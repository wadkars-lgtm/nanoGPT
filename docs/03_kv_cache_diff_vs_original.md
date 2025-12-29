# nanoGPT KV-Cache Augmentation — What Changed vs Original

This document summarizes **how the augmented KV-cache version differs from the original nanoGPT model.py**.
The goal is to highlight *semantic and architectural differences*, not restate unchanged code.

---

## 1. High-level capability change

**Original code**
- Stateless Transformer forward pass.
- Every `forward()` recomputes attention over the entire sequence.
- Decode cost grows with sequence length (O(T²) per token).

**New code**
- Adds **explicit Key–Value (KV) cache support** for incremental decoding.
- Past attention keys and values are reused across forward passes.
- Decode cost becomes O(T) per token instead of O(T²).

---

## 2. Forward API contract change (core difference)

### Original
```python
logits, loss = model(idx, targets=None)
```

### New
```python
logits, loss, present_kv = model(
    idx,
    targets=None,
    use_cache=True,
    past_kv=...
)
```

**Key differences**
- `use_cache` flag controls cache behavior.
- `past_kv` is provided by the caller (client-owned cache).
- `present_kv` is returned and must be passed back on the next call.

The model itself remains **stateless** with respect to KV.

---

## 3. KV cache data structure (new)

- `past_kv`: list of length `n_layer`
- Each entry: `(k, v)` tensors
- Shape:
  ```
  k, v : (B, n_head, T_past, head_dim)
  ```

The original code has **no per-layer state threading**.

---

## 4. Position embedding offset (critical correctness fix)

### Original
```python
pos = arange(0, t)
```

### New
```python
pos = arange(T_past, T_past + T_new)
```

Why this matters:
- With KV cache, new tokens are *continuations*, not fresh sequences.
- Reusing position 0 would corrupt attention semantics.
- This change is mandatory for correct cached decoding.

---

## 5. Attention layer changes (CausalSelfAttention)

### Original
- Attention computed only over the current input `x`.
- Always uses a standard causal mask.
- No notion of past vs new tokens.

### New
- Attention accepts `past_kv` and concatenates:
  ```python
  k = cat([k_past, k_new], dim=2)
  v = cat([v_past, v_new], dim=2)
  ```
- Total attention length becomes `T_total = T_past + T_new`.

---

## 6. Masking logic expanded (new correctness cases)

**Original**
- Single case: standard causal mask.

**New**
Three explicit regimes:
1. **Prefill** (`past_kv is None`)
   - Full causal mask required.
2. **Decode-step** (`T_new == 1`)
   - No mask required (no future tokens exist).
3. **Chunked decode** (`T_new > 1`)
   - Offset causal mask required to prevent intra-chunk future attention.

This makes attention **correct beyond one-token decoding**.

---

## 7. Flash Attention usage updated

### Original
```python
scaled_dot_product_attention(..., is_causal=True)
```

### New
- `is_causal` is set dynamically.
- Explicit boolean masks are passed when required.
- Supports offset masking with Flash Attention safely.

This enables compatibility with:
- KV cache
- chunked decode
- modern PyTorch SDPA semantics

---

## 8. Block.forward() signature change

### Original
```python
x = block(x)
```

### New
```python
x, present_kv = block(x, use_cache, past_kv)
```

Each block:
- Consumes its own layer’s `past_kv`
- Produces its own `present_kv`

This per-layer threading **did not exist** in the original.

---

## 9. GPT.forward() internal control flow change

**Original**
- Simple sequential block application.

**New**
- Detects `T_past` from cache
- Offsets positions
- Threads cache through every block
- Collects per-layer cache into `present_kv`

This turns `GPT.forward()` into a **cache-aware execution spine**.

---

## 10. Generation behavior (unchanged by default)

- The provided `generate()` function is still **baseline**.
- It does *not* use KV cache unless rewritten.
- This is intentional: KV support is added at the model level, not forced on generation.

---

## Summary

The new code:
- Preserves original training behavior
- Adds inference-grade KV cache support
- Makes masking logic explicit and future-proof
- Keeps the model stateless and client-controlled
- Aligns nanoGPT with real-world inference systems (Transformers, vLLM, TRT-LLM)

The original code is a **correct training model**.
The new code is a **correct training + scalable inference model**.
