import torch
from nanogpt.model import GPT, GPTConfig

def make_model(use_rope: bool, block_size=32):
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
        sliding_window_size=None,
    )
    return GPT(cfg).eval()

@torch.no_grad()
def test_rope_full_forward_matches_prefill_plus_decode():
    B, T_total, T_prompt = 2, 24, 16
    assert T_prompt < T_total

    model = make_model(use_rope=True, block_size=T_total)

    torch.manual_seed(999)
    idx = torch.randint(0, model.config.vocab_size, (B, T_total), dtype=torch.long)

    # 1) Full forward logits for all positions
    logits_full, _ = model(idx, targets=idx)   # (B, T_total, vocab)

    # 2) Prefill with cache on first T_prompt tokens
    pre = idx[:, :T_prompt]
    # If you add past_pos (recommended):
    # logits_pre, _, past, pos = model(pre, targets=None, use_cache=True, past_kv=None, past_pos=0)
    logits_pre, _, past = model(pre, targets=None, use_cache=True, past_kv=None)

    # 3) Decode remaining tokens one-by-one with teacher forcing
    logits_steps = []
    cur_past = past
    # past_pos = T_prompt  # if you implement explicit past_pos

    for t in range(T_prompt, T_total):
        tok = idx[:, t:t+1]  # teacher-forced token
        # If you add past_pos:
        # lg, _, cur_past, past_pos = model(tok, targets=None, use_cache=True, past_kv=cur_past, past_pos=past_pos)
        lg, _, cur_past = model(tok, targets=None, use_cache=True, past_kv=cur_past)
        # lg is (B, 1, vocab)
        logits_steps.append(lg)

    logits_tf = torch.cat(logits_steps, dim=1)  # (B, T_total-T_prompt, vocab)

    # Compare against full forward logits at those positions
    ref = logits_full[:, T_prompt:T_total, :]
    assert torch.allclose(logits_tf, ref, atol=1e-5, rtol=1e-5), "RoPE cache path mismatch vs full forward"
