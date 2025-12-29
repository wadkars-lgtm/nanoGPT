# GPT.forward() — Execution Spine (10 stages)

1) **Inputs + intent**
   - `idx`: token IDs, shape **(B, T_new)** (dtype `long`)
   - `targets`: optional, shape **(B, T_new)** (for training loss)
   - `use_cache`: controls whether we thread KV cache and return `present_kv`
   - `past_kv`: either `None` or a list of length `n_layer`, each entry either `None` or `(k, v)` where **k,v: (B, n_head, T_past, head_dim)**

2) **Normalize/initialize cache inputs**
   - If `use_cache=True` and `past_kv is None`, the code sets:
     - `past_kv = [None] * n_layer`
   - This guarantees we can index `past_kv[i]` for every layer in the loop.

3) **Infer `T_past` (past length)**
   - If caching is enabled and `past_kv[0]` exists, it infers:
     - `T_past = past_kv[0][0].size(2)`
   - Else:
     - `T_past = 0`
   - If `use_cache=False`, it forces `T_past = 0` always.

4) **Compute total length and enforce block_size**
   - `T_total = T_past + T_new`
   - Asserts `T_total <= block_size`

5) **Create position indices with the offset (cache correctness)**
   - `pos = arange(T_past, T_total)` produces shape **(T_new,)**
   - This offset ensures new tokens receive correct positional embeddings.

6) **Token embedding lookup**
   - `tok_emb = wte(idx)` → **(B, T_new, n_embd)**

7) **Position embedding lookup**
   - `pos_emb = wpe(pos)` → **(T_new, n_embd)**
   - Broadcast-add with `tok_emb` → **(B, T_new, n_embd)**

8) **Combine embeddings + dropout**
   - `x = drop(tok_emb + pos_emb)` → **(B, T_new, n_embd)**

9) **Transformer block loop (KV threading per layer)**
   - For each layer `i`:
     - `x, pkv = block(x, use_cache, past_kv[i])`
     - `pkv` (if present): **(B, n_head, T_total, head_dim)**
   - Collect all `pkv` into `present`.

10) **Final norm + logits (+ optional loss)**
   - `x = ln_f(x)`
   - Training: logits for all positions → **(B, T_new, vocab)**
   - Inference: logits for last position only → **(B, 1, vocab)**
