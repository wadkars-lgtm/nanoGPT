import pytest
import torch

from model import GPT, GPTConfig


def seed_all(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tiny_cfg(*, n_head=12, n_kv_head=None):
    # Tiny + deterministic: no dropout, no flash dependence for correctness
    # IMPORTANT: n_embd must be divisible by n_head.
    # head_dim = 8 here.
    return GPTConfig(
        block_size=64,
        vocab_size=128,
        n_layer=2,
        n_head=n_head,
        n_embd=96,
        dropout=0.0,
        bias=True,
        # You will add this field when implementing GQA/MQA:
        n_kv_head=n_kv_head,
    )


@torch.no_grad()
def forward_no_cache(model: GPT, idx):
    # your forward returns (logits, loss) when use_cache=False
    logits, loss = model(idx, targets=None, use_cache=False, past_kv=None)
    return logits, loss


@torch.no_grad()
def forward_with_cache(model: GPT, idx, past_kv):
    # your forward returns (logits, loss, present_kv) when use_cache=True
    logits, loss, present = model(idx, targets=None, use_cache=True, past_kv=past_kv)
    return logits, loss, present


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@pytest.mark.parametrize("device", ["cpu"])
def test_baseline_equivalence_default_kv_heads(device):
    """
    Baseline equivalence:
      - n_kv_head is None => should behave exactly like MHA (n_kv_head == n_head)
    This prevents a silent mismatch between the "default" and explicit MHA setting.
    """
    seed_all(777)

    cfg_default = tiny_cfg(n_head=12, n_kv_head=None)
    cfg_mha = tiny_cfg(n_head=12, n_kv_head=12)

    # ensure identical init: reseed before each model init
    seed_all(777)
    m_default = GPT(cfg_default).to(device).eval()
    seed_all(777)
    m_mha = GPT(cfg_mha).to(device).eval()

    idx = torch.randint(0, cfg_default.vocab_size, (2, 17), device=device)

    logits_a, _ = forward_no_cache(m_default, idx)
    logits_b, _ = forward_no_cache(m_mha, idx)

    d = max_abs_diff(logits_a, logits_b)
    assert d < 1e-6, f"default n_kv_head(None) != explicit MHA (n_kv_head=n_head). max_diff={d}"


@pytest.mark.parametrize("device", ["cpu"])
def test_cache_equivalence_next_token_logits(device):
    """
    Cache equivalence:
      Compare logits for the last token between:
        A) full forward on (prompt + next_token) with no cache
        B) prefill(prompt) to get past_kv, then decode(next_token) using cache
    This catches cache indexing / concat bugs and attention masking mistakes.

    This test uses YOUR cache API:
      - past_kv is list[n_layer] of (k,v) tuples or None entries
    """
    seed_all(999)

    # pick a GQA-ish setting to validate grouped KV works too
    cfg = tiny_cfg(n_head=12, n_kv_head=3)
    model = GPT(cfg).to(device).eval()

    B = 2
    T_prompt = 13
    prompt = torch.randint(0, cfg.vocab_size, (B, T_prompt), device=device)
    next_tok = torch.randint(0, cfg.vocab_size, (B, 1), device=device)
    full = torch.cat([prompt, next_tok], dim=1)

    # A) full forward (no cache). Your inference optimization returns only last position logits
    logits_full, _ = forward_no_cache(model, full)
    full_last = logits_full[:, -1, :]

    # B) prefill prompt, then decode 1 token with cache
    _, _, past = forward_with_cache(model, prompt, past_kv=None)
    logits_dec, _, _ = forward_with_cache(model, next_tok, past_kv=past)
    dec_last = logits_dec[:, -1, :]

    d = max_abs_diff(full_last, dec_last)
    assert d < 1e-5, f"cache decode logits != full forward logits. max_diff={d}"
