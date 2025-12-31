"""
Full definition of a GPT Language Model, all of it in this single file.

This version adds:
  1) **KV cache support** for fast incremental decoding.
  2) **MHA / GQA / MQA** support via configurable n_kv_head.
  3) **use_sdpa** knob for deterministic tests (disable SDPA) or fast path (enable SDPA).

KV cache API:
  - GPT.forward(idx, targets=None, use_cache=True, past_kv=...) ->
        returns (logits, loss, present_kv)

Where:
  - past_kv is either None, or a list length n_layer of (k, v) tuples
    with shapes:
      k, v = (B, n_kv_head, T_past, head_dim)
  - present_kv is the same structure, with T_total = T_past + T_new

Attention head semantics:
  - Query heads: n_head
  - KV heads: n_kv_head
  - Constraint: n_head % n_kv_head == 0

Modes:
  - MHA: n_kv_head = n_head
  - GQA: 1 < n_kv_head < n_head
  - MQA: n_kv_head = 1

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

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        # KV heads
        self.n_kv_head = config.n_kv_head
        assert self.n_kv_head >= 1
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        kv_dim = self.n_kv_head * self.head_dim

        # Separate projections for Q and KV (KV packed as [K;V])
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_kv = nn.Linear(config.n_embd, 2 * kv_dim, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # SDPA availability
        self.flash = bool(config.use_sdpa) and hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Causal mask for manual attention (prefill path)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def _build_offset_causal_mask_bool(
        self, T_new: int, T_total: int, past_len: int, device
    ) -> torch.Tensor:
        """
        Boolean mask (T_new, T_total) where True means "allowed".
        Allowed condition: j <= past_len + i
        """
        i = torch.arange(T_new, device=device)[:, None]          # (T_new, 1)
        j = torch.arange(T_total, device=device)[None, :]        # (1, T_total)
        return j <= (past_len + i)                               # (T_new, T_total)

    def _build_offset_causal_mask_additive(
        self, T_new: int, T_total: int, past_len: int, device, dtype
    ) -> torch.Tensor:
        """
        Additive mask (T_new, T_total) with 0 for allowed and -inf for disallowed.
        Safer across PyTorch SDPA versions than boolean masks.
        """
        allowed = self._build_offset_causal_mask_bool(T_new, T_total, past_len, device=device)
        mask = torch.zeros((T_new, T_total), device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def _expand_kv_to_q_heads(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match query heads.
        Input:  (B, n_kv_head, T, hd)
        Output: (B, n_head,    T, hd)

        NOTE: repeat_interleave materializes expanded KV (simple, not optimal).
        """
        if self.n_kv_head == self.n_head:
            return kv
        g = self.n_head // self.n_kv_head
        return kv.repeat_interleave(g, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
          x: (B, T_new, C)
          use_cache: if True, returns present_kv = (k_total, v_total)
          past_kv: optional tuple (k_past, v_past) with shapes (B, n_kv_head, T_past, head_dim)

        Returns:
          y: (B, T_new, C)
          present_kv: (k_total, v_total) if use_cache else None
        """
        B, T_new, C = x.size()
        hd = self.head_dim

        # Projections
        q = self.c_q(x)                                # (B, T_new, n_embd)
        kv = self.c_kv(x)                              # (B, T_new, 2*(n_kv_head*hd))
        k, v = kv.split(kv.size(-1) // 2, dim=2)        # each (B, T_new, n_kv_head*hd)

        # Shape into heads
        q = q.view(B, T_new, self.n_head, hd).transpose(1, 2)          # (B, n_head, T_new, hd)
        k = k.view(B, T_new, self.n_kv_head, hd).transpose(1, 2)       # (B, n_kv_head, T_new, hd)
        v = v.view(B, T_new, self.n_kv_head, hd).transpose(1, 2)       # (B, n_kv_head, T_new, hd)

        past_len = 0
        if past_kv is not None:
            k_past, v_past = past_kv
            assert k_past.size(1) == self.n_kv_head, f"expected k_past heads={self.n_kv_head}, got {k_past.size(1)}"
            assert v_past.size(1) == self.n_kv_head, f"expected v_past heads={self.n_kv_head}, got {v_past.size(1)}"
            past_len = k_past.size(2)
            k = torch.cat([k_past, k], dim=2)  # (B, n_kv_head, T_total, hd)
            v = torch.cat([v_past, v], dim=2)

        T_total = k.size(2)

        # Expand KV to query-head space for compute
        kq = self._expand_kv_to_q_heads(k)  # (B, n_head, T_total, hd)
        vq = self._expand_kv_to_q_heads(v)

        # Masking rules:
        # - prefill: causal mask required
        # - decode-step (T_new == 1 with past): no mask needed
        # - chunked decode (T_new > 1 with past): offset causal mask required
        need_mask = (past_kv is None) or (past_kv is not None and T_new > 1)

        if self.flash:
            if need_mask and past_kv is not None:
                # offset mask for chunked decode with cache
                attn_mask = self._build_offset_causal_mask_additive(
                    T_new, T_total, past_len, device=x.device, dtype=q.dtype
                )
                y = F.scaled_dot_product_attention(
                    q, kq, vq,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
            else:
                # prefill => is_causal=True
                # decode-step with past => is_causal=False is safe
                is_causal = (past_kv is None)
                y = F.scaled_dot_product_attention(
                    q, kq, vq,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            # Manual attention
            att = (q @ kq.transpose(-2, -1)) * (1.0 / math.sqrt(kq.size(-1)))  # (B, n_head, T_new, T_total)

            if past_kv is None:
                att = att.masked_fill(self.bias[:, :, :T_new, :T_new] == 0, float("-inf"))
            else:
                if T_new > 1:
                    allowed = self._build_offset_causal_mask_bool(T_new, T_total, past_len, device=x.device)
                    allowed = allowed.view(1, 1, T_new, T_total)
                    att = att.masked_fill(~allowed, float("-inf"))
                # else: T_new == 1 with past => no mask needed

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ vq  # (B, n_head, T_new, hd)

        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_dropout(self.c_proj(y))

        # Cache stored in KV-head space (B, n_kv_head, T_total, hd)
        present_kv = (k, v) if use_cache else None
        return y, present_kv


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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

    # KV heads (None => default to n_head == MHA)
    n_kv_head: Optional[int] = None

    dropout: float = 0.0
    bias: bool = True
    use_sdpa: bool = True

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert isinstance(self.n_kv_head, int) and self.n_kv_head >= 1
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
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

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Scaled init to residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ):
        """
        If use_cache=False: returns (logits, loss)
        If use_cache=True:  returns (logits, loss, present_kv)

        past_kv:
          - None, or
          - list length n_layer of (k, v) tuples (or None entries),
            where k,v shape = (B, n_kv_head, T_past, head_dim)

        Position embedding offset:
          When using cache, positions must start at T_past.
        """
        device = idx.device
        b, t_new = idx.size()

        if use_cache:
            if past_kv is None:
                past_kv = [None] * self.config.n_layer
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

        pos = torch.arange(T_past, T_total, dtype=torch.long, device=device)  # (t_new,)

        tok_emb = self.transformer.wte(idx)  # (b, t_new, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t_new, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        present = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            layer_past = past_kv[i] if use_cache else None
            x, pkv = block(x, use_cache=use_cache, past_kv=layer_past)
            if use_cache:
                present.append(pkv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if use_cache:
            return logits, loss, present
        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None) -> "GPT":
        """
        Loads HF GPT-2 weights into this implementation.

        IMPORTANT:
          - Supported ONLY for MHA (n_kv_head == n_head).
          - If you want GQA/MQA with pretrained weights, you must define a projection
            remapping strategy (not implemented here).
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

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
        # MHA only
        config_args["n_kv_head"] = config_args["n_head"]

        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)

        # HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Our model state dict
        sd = model.state_dict()

        def _copy(dst_key: str, src_key: str, transpose_if_needed: bool = False) -> None:
            assert src_key in sd_hf, f"missing in HF: {src_key}"
            assert dst_key in sd, f"missing in ours: {dst_key}"
            src = sd_hf[src_key]
            dst = sd[dst_key]
            if transpose_if_needed and src.shape[::-1] == dst.shape:
                src = src.t()
            assert src.shape == dst.shape, f"shape mismatch for {dst_key}: src {src.shape} vs dst {dst.shape}"
            with torch.no_grad():
                dst.copy_(src)

        # Copy embeddings + final LN
        _copy("transformer.wte.weight", "transformer.wte.weight", transpose_if_needed=False)
        _copy("transformer.wpe.weight", "transformer.wpe.weight", transpose_if_needed=False)
        _copy("transformer.ln_f.weight", "transformer.ln_f.weight", transpose_if_needed=False)
        _copy("transformer.ln_f.bias", "transformer.ln_f.bias", transpose_if_needed=False)

        # Blocks
        for i in range(config.n_layer):
            # Layer norms
            _copy(f"transformer.h.{i}.ln_1.weight", f"transformer.h.{i}.ln_1.weight")
            _copy(f"transformer.h.{i}.ln_1.bias", f"transformer.h.{i}.ln_1.bias")
            _copy(f"transformer.h.{i}.ln_2.weight", f"transformer.h.{i}.ln_2.weight")
            _copy(f"transformer.h.{i}.ln_2.bias", f"transformer.h.{i}.ln_2.bias")

            # MLP (Conv1D weights need transpose)
            _copy(f"transformer.h.{i}.mlp.c_fc.weight", f"transformer.h.{i}.mlp.c_fc.weight", transpose_if_needed=True)
            _copy(f"transformer.h.{i}.mlp.c_fc.bias", f"transformer.h.{i}.mlp.c_fc.bias")
            _copy(f"transformer.h.{i}.mlp.c_proj.weight", f"transformer.h.{i}.mlp.c_proj.weight", transpose_if_needed=True)
            _copy(f"transformer.h.{i}.mlp.c_proj.bias", f"transformer.h.{i}.mlp.c_proj.bias")

            # Attention output proj (Conv1D weight needs transpose)
            _copy(f"transformer.h.{i}.attn.c_proj.weight", f"transformer.h.{i}.attn.c_proj.weight", transpose_if_needed=True)
            _copy(f"transformer.h.{i}.attn.c_proj.bias", f"transformer.h.{i}.attn.c_proj.bias")

            # Attention QKV mapping:
            # HF has fused c_attn producing [Q,K,V] each of size n_embd.
            # Our model has c_q (n_embd) and c_kv (2*n_embd), with MHA kv_dim = n_embd.
            w_hf = sd_hf[f"transformer.h.{i}.attn.c_attn.weight"]
            b_hf = sd_hf[f"transformer.h.{i}.attn.c_attn.bias"]

            # HF Conv1D weight stored transposed relative to nn.Linear:
            # In original nanoGPT, they copy with .t() into Linear weights.
            w = w_hf.t() if w_hf.shape[::-1] == sd[f"transformer.h.{i}.attn.c_q.weight"].shape[:1] + sd[f"transformer.h.{i}.attn.c_q.weight"].shape[1:] else None
            # More robust:
            # target shapes:
            wq_tgt = sd[f"transformer.h.{i}.attn.c_q.weight"]
            wkv_tgt = sd[f"transformer.h.{i}.attn.c_kv.weight"]
            if w_hf.shape[::-1] == wq_tgt.shape and False:
                pass

            # Robust transpose rule:
            if w_hf.shape[::-1] == (3 * config.n_embd, config.n_embd):
                # extremely unlikely; ignore
                w_full = w_hf
            else:
                # expected: HF (in, out) -> transpose to (out, in)
                w_full = w_hf.t()

            assert w_full.shape == (3 * config.n_embd, config.n_embd), f"unexpected HF c_attn weight shape after transpose: {w_full.shape}"

            w_q, w_k, w_v = w_full.split(config.n_embd, dim=0)
            b_q, b_k, b_v = b_hf.split(config.n_embd, dim=0)

            # Write into our params
            with torch.no_grad():
                sd[f"transformer.h.{i}.attn.c_q.weight"].copy_(w_q)
                sd[f"transformer.h.{i}.attn.c_q.bias"].copy_(b_q)

                w_kv = torch.cat([w_k, w_v], dim=0)   # (2*n_embd, n_embd)
                b_kv = torch.cat([b_k, b_v], dim=0)   # (2*n_embd,)
                sd[f"transformer.h.{i}.attn.c_kv.weight"].copy_(w_kv)
                sd[f"transformer.h.{i}.attn.c_kv.bias"].copy_(b_kv)

        # Commit state dict
        model.load_state_dict(sd, strict=True)
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
