import torch
from nanogpt.model import GPT, GPTConfig

def run(sliding_window_size, block_size=64):
    # Reset RNG so weights and inputs are identical for every run
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
        use_sdpa=False,  # deterministic path
        sliding_window_size=sliding_window_size,
    )
    m = GPT(cfg).eval()

    B, T = 2, block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

    with torch.no_grad():
        logits, _ = m(idx, targets=idx)  # full logits
    return logits

def max_abs_diff(a, b):
    return (a - b).abs().max().item()

if __name__ == "__main__":
    logits_none = run(None, block_size=64)
    logits_full = run(64, block_size=64)   # full window == block_size
    logits_w1   = run(1,  block_size=64)

    print("\n=== SWA SANITY CHECK ===")
    d_full = max_abs_diff(logits_none, logits_full)
    d_w1   = max_abs_diff(logits_none, logits_w1)

    print(f"None vs full window: max |diff| = {d_full:.6g}")
    print(f"None vs window=1:   max |diff| = {d_w1:.6g}")

    print("\nExpected:")
    print(" - None vs full window → ~0 diff")
    print(" - None vs window=1   → large diff")
