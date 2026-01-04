import torch
from nanogpt.model import GPT, GPTConfig

def run(use_rope, sliding_window_size, block_size=32):
    torch.manual_seed(123)
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=2,
        n_head=4,
        n_embd=128,
        n_kv_head=4,
        dropout=0.0,
        bias=True,
        use_sdpa=False,
        use_rope=use_rope,
        sliding_window_size=sliding_window_size,
    )
    m = GPT(cfg).eval()

    torch.manual_seed(999)
    idx = torch.randint(0, cfg.vocab_size, (2, block_size), dtype=torch.long)
    with torch.no_grad():
        logits, _ = m(idx, targets=idx)
    return logits

def test_rope_swa_full_window_matches_none():
    a = run(use_rope=True, sliding_window_size=None, block_size=32)
    b = run(use_rope=True, sliding_window_size=32, block_size=32)
    assert torch.allclose(a, b, atol=1e-6, rtol=0), "RoPE SWA full-window should match None"

def test_rope_swa_window_one_changes_behavior():
    a = run(use_rope=True, sliding_window_size=None, block_size=32)
    b = run(use_rope=True, sliding_window_size=1, block_size=32)
    diff = (a - b).abs().max().item()
    assert diff > 1e-4, "RoPE SWA window=1 should change logits"
