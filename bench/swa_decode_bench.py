import time
import torch
from nanogpt.model import GPT, GPTConfig

@torch.no_grad()
def bench(sliding_window_size, T_prompt=2048, steps=128, use_sdpa=True, device="cuda"):
    torch.manual_seed(0)

    cfg = GPTConfig(
        block_size=T_prompt + steps,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_head=12,
        dropout=0.0,
        bias=True,
        use_sdpa=use_sdpa,
        sliding_window_size=sliding_window_size,
    )
    m = GPT(cfg).to(device).eval()

    # prompt
    idx = torch.randint(0, cfg.vocab_size, (1, T_prompt), device=device)

    # prefill with cache
    logits, loss, past = m(idx, targets=None, use_cache=True, past_kv=None)

    # decode loop (1 token at a time)
    t0 = time.time()
    cur = idx[:, -1:]
    for _ in range(steps):
        logits, loss, past = m(cur, targets=None, use_cache=True, past_kv=past)
        cur = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"W={sliding_window_size}  steps={steps}  total={dt:.4f}s  ms/tok={(dt/steps)*1000:.3f}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    bench(None)           # baseline full attention
    bench(1)              # should be much faster, quality nonsense
    bench(2048)           # should match baseline closely (same as None for this prompt)
    bench(1024)           # the meaningful SWA regime
