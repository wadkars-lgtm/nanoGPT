# Sliding Window Attention (SWA) — Current Implementation

This document describes the **Sliding Window Attention (SWA)** strategy currently implemented in this nanoGPT fork.

The implementation is **intentional, minimal, and correctness-first**:
- SWA is enforced **purely via attention masking**
- **No KV cache eviction** is performed
- Behavior is deterministic and compatible with MHA, GQA, and MQA
- Designed to be a stepping stone toward production-style SWA

---

## What Sliding Window Attention Means Here

For a given query token at position `t`, attention is restricted to:

```
[t - W + 1, ..., t]
```

Where:
- `W = sliding_window_size`
- Tokens outside the window are **masked out**
- The model **cannot attend to earlier tokens**, even though they exist in memory

This matches the *logical behavior* of SWA, but **not yet the memory optimization**.

---

## Key Design Choice: Mask-Only SWA

### What we do
- KV cache **continues to grow** with total sequence length
- A **banded causal mask** enforces the window constraint
- Masking is applied consistently across:
  - Prefill
  - Decode-step (`T_new == 1`)
  - Chunked decode (`T_new > 1 with past_kv`)

### What we do NOT do (yet)
- No KV cache truncation
- No KV eviction or rolling buffers
- No memory footprint reduction

This makes the implementation:
- Simple
- Correct
- Easy to reason about
- Easy to validate against full attention

---

## Why Mask-Only SWA Is Still Useful

Even without KV eviction, this implementation is valuable:

1. **Correctness validation**
   - Ensures attention semantics are correct
   - Matches expected behavior for edge cases

2. **Performance signal**
   - Reduces *effective attention width*
   - Can reduce compute when using optimized kernels

3. **Foundation for future work**
   - KV eviction can be layered later
   - RoPE-relative positioning issues can be explored cleanly

---

## Implementation Details

### Configuration

SWA is enabled via:

```python
GPTConfig(
    sliding_window_size = W  # int or None
)
```

- `None` → full causal attention
- `W == block_size` → equivalent to full attention
- `W == 1` → token can attend only to itself

---

### Mask Construction

For each query index `i` and key index `j`:

**Causal constraint**
```
j <= past_len + i
```

**Window constraint**
```
j >= (past_len + i) - (W - 1)
```

Both must be satisfied.

This is implemented as:
- Boolean masks for manual attention
- Additive `-inf` masks for SDPA

---

## Interaction with KV Cache

### Important Clarification

Even with SWA enabled:

- KV tensors still have shape:
  ```
  (B, n_kv_head, T_total, head_dim)
  ```
- Tokens outside the window:
  - Exist in KV cache
  - Are never attended to
  - Still consume memory

This is **intentional**.

---

## Expected Behavior (Sanity Checks)

The following sanity properties must hold:

| Configuration | Expected Result |
|--------------|----------------|
| `sliding_window_size = None` | Identical to baseline |
| `sliding_window_size = block_size` | Identical to baseline |
| `sliding_window_size = 1` | Severe quality collapse |
| Token-by-token decode vs chunked decode | Identical logits |

Dedicated sanity scripts validate these properties.

---

## Known Limitations

1. **No memory savings**
   - KV cache grows linearly with total sequence length

2. **RoPE interaction**
   - Absolute positional embeddings work trivially
   - RoPE requires careful handling of position offsets
   - Chunked decode tests are required for correctness

3. **Not production SWA**
   - Production systems also evict or roll KV cache
   - This implementation intentionally stops short of that

---

## Why This Design Was Chosen

This implementation prioritizes:

- Semantic correctness
- Testability
- Architectural clarity
- Compatibility with MHA / GQA / MQA
- A clean baseline for future KV eviction work

It avoids premature complexity while still modeling real SWA behavior.

---

## Next Logical Extensions (Not Implemented)

- KV cache eviction / rolling buffers
- Fixed-size KV storage
- RoPE-aware window-relative positioning
- Memory footprint measurements
- Attention kernel specialization

These can be added incrementally **without changing the SWA semantics** already in place.

---

## Summary

- SWA is enforced via masking, not memory management
- Behavior is correct and well-tested
- Memory usage is unchanged by design
- This is a **foundational SWA implementation**, not the final form

If you are reading this expecting memory reduction: that is the *next* layer, not this one.
