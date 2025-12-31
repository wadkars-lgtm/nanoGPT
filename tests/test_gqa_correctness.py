import os
import math
import inspect
import pytest
import torch

from model import GPT, GPTConfig


def _seed_all(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_tiny_config(n_head=12, n_kv_head=None, device="cpu"):
    # Keep tiny so tests run fast on CPU
    cfg = GPTConfig(
        vocab_size=128,
        block_size=64,
        n_layer=2,
        n_head=n_head,
        n_embd=96,      # must be divisible by n_head -> 96/12=8
        n_kv_head=n_kv_head,
        dropout=0.0,
        bias=True,
        use_sdpa=False, # avoid device-specific SDPA differences for test determinism
    )
    return cfg


def _forward_supports_cache(model: GPT):
    # Try to detect whether your GPT forward supports kv caching.
    # This is intentionally permissive: it looks for typical names.
    sig = inspect.signature(model.forward)
    names = set(sig.parameters.keys())
    # common patterns:
    # forward(idx, targets=None, use_kv_cache=False, kv_cache=None, ...)
    has_flag = any(n in names for n in ["use_kv_cache", "kv_cache", "past_kv", "cache"])
    return has_flag


def _forward_no_targets(model: GPT, idx: torch.Tensor, **kwargs):
    # Helper to call forward without targets; returns logits only.
    out = model(idx, targets=None, **kwargs)
    if isinstance(out, tuple):
        logits = out[0]
        extras = out[1:]
        return logits, extras
    return out, ()


@pytest.mark.parametrize("device", ["cpu"])
def test_baseline_equivalence_mha_path(device):
    """
    Baseline equivalence test:
    - Build model with n_kv_head = n_head (MHA)
    - Ensure outputs are deterministic and stable under same seed,
      AND (importantly) that n_kv_head=None behaves the same as n_kv_head=n_head.
    This catches 'default n_kv_head' bugs and reshape/broadcast mistakes.
    """
    _seed_all(123)

    cfg_a = _build_tiny_config(n_head=12, n_kv_head=None)
    cfg_b = _build_tiny_config(n_head=12, n_kv_head=12)

    model_a = GPT(cfg_a).to(device).eval()
    model_b = GPT(cfg_b).to(device).eval()

    # Ensure identical init by re-seeding before each init (already done above),
    # but modules are created sequentially, so reseed again:
    _seed_all(123)
    model_a = GPT(cfg_a).to(device).eval()
    _seed_all(123)
    model_b = GPT(cfg_b).to(device).eval()

    idx = torch.randint(0, cfg_a.vocab_size, (2, 17), device=device)

    with torch.no_grad():
        logits_a, _ = _forward_no_targets(model_a, idx)
        logits_b, _ = _forward_no_targets(model_b, idx)

    max_diff = (logits_a - logits_b).abs().max().item()
    assert max_diff < 1e-6, f"MHA default n_kv_head(None) != n_kv_head(n_head). max_diff={max_diff}"


@pytest.mark.parametrize("device", ["cpu"])
def test_cache_equivalence_one_step(device):
    _seed_all(456)

    cfg = _build_tiny_config(n_head=12, n_kv_head=3, device=device)
    model = GPT(cfg).to(device)
    model.eval()

    B = 2
    T = 16
    vocab = cfg.vocab_size

    prompt = torch.randint(0, vocab, (B, T), device=device)

    # 1) Prefill -> cache
    with torch.no_grad():
        out = model(prompt, targets=None, use_cache=True, past_kv=None)

    assert isinstance(out, tuple) and len(out) == 3
    logits_prefill, loss_prefill, past = out
    assert past is not None and isinstance(past, list)
    assert len(past) == cfg.n_layer

    # 2) Decode 1 step using cache
    next_token = torch.randint(0, vocab, (B, 1), device=device)
    with torch.no_grad():
        logits_cached, loss_cached, present2 = model(
            next_token, targets=None, use_cache=True, past_kv=past
        )
    assert logits_cached.shape[:2] == (B, 1)

    # 3) Full forward on concatenated sequence (no cache)
    full = torch.cat([prompt, next_token], dim=1)
    with torch.no_grad():
        logits_full, loss_full = model(full, targets=None, use_cache=False)
    assert logits_full.shape[:2] == (B, 1)

    torch.testing.assert_close(logits_cached, logits_full, rtol=1e-5, atol=1e-5)