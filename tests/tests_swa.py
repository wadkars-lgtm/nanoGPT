import torch
from model import GPT, GPTConfig

def _run(sliding_window_size, block_size=64):
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
        use_sdpa=False,   # deterministic
        sliding_window_size=sliding_window_size,
    )
    model = GPT(cfg).eval()

    B, T = 2, block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(idx, targets=idx)

    return logits

def test_swa_none_matches_full_window():
    logits_none = _run(None)
    logits_full = _run(sliding_window_size=logits_none.size(1))

    diff = (logits_none - logits_full).abs().max().item()
    assert diff < 1e-6, f"SWA full window mismatch: max diff {diff}"

def test_swa_window_one_changes_behavior():
    logits_none = _run(None)
    logits_w1 = _run(sliding_window_size=1)

    diff = (logits_none - logits_w1).abs().max().item()
    assert diff > 1e-4, "SWA window=1 did not change logits"
