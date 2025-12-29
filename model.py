"""
Full definition of a GPT Language Model, all of it in this single file.

This version adds **KV cache support** for fast incremental decoding.

KV cache API (new):
  - GPT.forward(idx, targets=None, use_cache=True, past_kv=...) ->
        returns (logits, loss, present_kv)

Where:
  - past_kv is either None, or a list length n_layer of (k, v) tuples
    with shapes: k, v = (B, n_head, T_past, head_dim)
  - present_kv is the same structure, with T_total = T_past + T_new

Notes on attention masking:
  - Prefill (past_kv is None): causal mask is required.
  - Decode-step (T_new == 1 with past): causal mask is NOT needed because keys contain only past+current.
  - Chunked decode (T_new > 1 with past): we apply an offset causal mask that prevents attending
    to “future” tokens within the newly provided chunk.

References:
1) OpenAI GPT-2 TF:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) HF Transformers GPT-2:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask for max block_size (prefill path)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def _build_offset_causal_mask(self, T_new: int, T_total: int, past_len: int, device):
        """
        Build a boolean mask (T_new, T_total) where mask[i, j] = True if key j is allowed
        for query position i in the new chunk, given there are past_len cached tokens.

        Allowed condition: j <= past_len + i
        """
        i = torch.arange(T_new, device=device)[:, None]                 # (T_new, 1)
        j = torch.arange(T_total, device=device)[None, :]               # (1, T_total)
        return j <= (past_len + i)                                      # (T_new, T_total) boolean

    def forward(self, x, use_cache: bool = False, past_kv=None):
        """
        Args:
          x: (B, T_new, C)
          use_cache: if True, returns present_kv = (k_total, v_total)
          past_kv: optional tuple (k_past, v_past) with shapes (B, nh, T_past, hs)

        Returns:
          y: (B, T_new, C)
          present_kv: (k_total, v_total) if use_cache else None
        """
        B, T_new, C = x.size()  # batch size, new sequence length, embedding dim
        hs = C // self.n_head

        # project
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # shape into heads
        k = k.view(B, T_new, self.n_head, hs).transpose(1, 2)  # (B, nh, T_new, hs)
        q = q.view(B, T_new, self.n_head, hs).transpose(1, 2)  # (B, nh, T_new, hs)
        v = v.view(B, T_new, self.n_head, hs).transpose(1, 2)  # (B, nh, T_new, hs)

        past_len = 0
        if past_kv is not None:
            k_past, v_past = past_kv
            past_len = k_past.size(2)
            k = torch.cat([k_past, k], dim=2)  # (B, nh, T_total, hs)
            v = torch.cat([v_past, v], dim=2)

        T_total = k.size(2)

        # Decide masking
        # - prefill (no past): causal mask required
        # - decode-step (past exists, T_new==1): no mask needed (no future keys exist)
        # - chunked decode (past exists, T_new>1): need offset causal mask
        need_mask = (past_kv is None) or (past_kv is not None and T_new > 1)

        if self.flash:
            if need_mask and past_kv is not None:
                # offset causal mask (T_new, T_total) boolean
                mask = self._build_offset_causal_mask(T_new, T_total, past_len, x.device)
                # SDPA expects either float mask with -inf or boolean mask depending on version.
                # Boolean mask works in modern PyTorch: True means "keep", False means "mask out".
                attn_mask = mask
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,  # we provide mask explicitly
                )
            else:
                # prefill: is_causal=True
                # decode-step with past and T_new==1: is_causal=False is safe and correct
                is_causal = (past_kv is None)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            # manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T_new, T_total)

            if past_kv is None:
                # standard causal mask for prefill
                att = att.masked_fill(self.bias[:, :, :T_new, :T_new] == 0, float("-inf"))
            else:
                if T_new > 1:
                    # offset causal mask for chunked decode
                    mask = self._build_offset_causal_mask(T_new, T_total, past_len, x.device)  # (T_new, T_total)
                    # expand to (1,1,T_new,T_total)
                    mask = mask.view(1, 1, T_new, T_total)
                    att = att.masked_fill(~mask, float("-inf"))
                # else T_new==1: no mask needed

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T_new, hs)

        # reassemble
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_dropout(self.c_proj(y))

        present_kv = (k, v) if use_cache else None
        return y, present_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, use_cache: bool = False, past_kv=None):
        a, present_kv = self.attn(self.ln_1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms like GPT-2


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # scaled init to residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report params
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache: bool = False, past_kv=None):
        """
        If use_cache=False (default): returns (logits, loss)
        If use_cache=True: returns (logits, loss, present_kv)

        past_kv:
          - None, or
          - list length n_layer of (k, v) tuples (or None entries),
            where k,v shape = (B, n_head, T_past, head_dim)

        Position embedding offset:
          When using cache, positions must start at T_past.
        """
        device = idx.device
        b, t_new = idx.size()

        if use_cache:
            if past_kv is None:
                past_kv = [None] * self.config.n_layer
            # determine past length from layer 0 (if present)
            if past_kv[0] is not None:
                T_past = past_kv[0][0].size(2)
            else:
                T_past = 0
        else:
            T_past = 0

        T_total = T_past + t_new
        assert T_total <= self.config.block_size, (
            f"Cannot forward total length {T_total}, block size is only {self.config.block_size}"
        )

        # position indices for the new tokens
        pos = torch.arange(T_past, T_total, dtype=torch.long, device=device)  # (t_new,)

        # token + pos embeddings
        tok_emb = self.transformer.wte(idx)      # (b, t_new, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (t_new, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        present = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            x, pkv = block(x, use_cache=use_cache, past_kv=(past_kv[i] if use_cache else None))
            if use_cache:
                present.append(pkv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward lm_head on last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if use_cache:
            return logits, loss, present
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Baseline generate (no KV cache).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
