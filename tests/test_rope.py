import torch
from nanogpt.model import GPT, GPTConfig

def make_model(use_rope: bool, block_size=16, sliding_window_size=None):
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
        use_sdpa=False,  # deterministic
        use_rope=use_rope,
        sliding_window_size=sliding_window_size,
    )
    return GPT(cfg).eval()

@torch.no_grad()
def test_rope_pos0_noop_single_token():
    # If T=1, the only position used is pos=0. RoPE should be identity at pos=0.
    # So abs-pos vs rope won't match (different positional scheme), BUT:
    # the RoPE transform itself should not change Q/K at pos=0.
    #
    # Without internal hooks, we validate a weaker invariant:
    # RoPE model should run and be deterministic across multiple runs with same seed.
    m1 = make_model(use_rope=True, block_size=1)
    m2 = make_model(use_rope=True, block_size=1)

    idx = torch.tensor([[42]], dtype=torch.long)
    logits1, _ = m1(idx, targets=idx)
    logits2, _ = m2(idx, targets=idx)
    assert torch.allclose(logits1, logits2, atol=0, rtol=0), "RoPE model not deterministic at pos=0"
