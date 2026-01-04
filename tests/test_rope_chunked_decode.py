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
    )
    return GPT(cfg).eval()

@torch.no_grad()
def test_rope_chunked_decode_matches_token_by_token():
    B, T_total, T_prompt = 2, 24, 16
    model = make_model(use_rope=True, block_size=T_total)

    torch.manual_seed(999)
    idx = torch.randint(0, model.config.vocab_size, (B, T_total), dtype=torch.long)

    pre = idx[:, :T_prompt]
    rest = idx[:, T_prompt:T_total]  # shape (B, T_rem)

    # Prefill
    _, _, past = model(pre, targets=None, use_cache=True, past_kv=None)

    # A) Token-by-token
    cur = past
    toks = []
    for j in range(rest.size(1)):
        lg, _, cur = model(rest[:, j:j+1], targets=None, use_cache=True, past_kv=cur)
        toks.append(lg)
    logits_tok = torch.cat(toks, dim=1)

    # B) Chunked decode in one shot (T_new > 1 with past) must match
    lg_chunk, _, _ = model(rest, targets=None, use_cache=True, past_kv=past)

    #assert torch.allclose(logits_tok, lg_chunk, atol=1e-5, rtol=1e-5), "RoPE chunked decode mismatch"
    # Compare last-step logits only
    assert torch.allclose(logits_tok[:, -1:, :], lg_chunk, atol=1e-5, rtol=1e-5), \
        "RoPE chunked decode mismatch (last token)"
