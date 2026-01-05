# Sliding Window Attention (SWA) — Mask-Only Design and Benchmark Results

This document describes the **current Sliding Window Attention (SWA) implementation** in this nanoGPT fork, explains **why it does not improve performance**, and records the **benchmark evidence** supporting that conclusion.

The design is **intentional, minimal, and correctness-first**.

---

## Executive Summary

- SWA is implemented **purely via attention masking**
- **No KV cache eviction or truncation** is performed
- Attention semantics are correct and deterministic
- **No performance or memory reduction is expected**
- In practice, masked SWA is **slower** under PyTorch SDPA

This implementation exists to validate **architectural behavior**, not to optimize inference speed.

---

## What Sliding Window Attention Means Here

For a query token at position `t`, attention is restricted to:

```
[t - W + 1, ..., t]
```

Where:
- `W = sliding_window_size`
- Tokens outside the window are **masked out**
- The model **cannot attend to earlier tokens**, even though they remain in memory

This enforces the **logical behavior** of SWA, but **not the systems optimization**.

---

## Key Design Choice: Mask-Only SWA

### What is implemented

- KV cache **continues to grow** with total sequence length
- A **banded causal attention mask** enforces the window constraint
- Masking is applied consistently across:
  - Prefill
  - Decode-step (`T_new == 1`)
  - Chunked decode (`T_new > 1 with past_kv`)

### What is intentionally not implemented

- No KV cache eviction
- No rolling buffers
- No truncation of K/V tensors
- No sparse or block-sparse attention kernels

As a result:
- **Tensor shapes do not shrink**
- **Attention matmul size is unchanged**
- **Memory footprint is unchanged**

---

## Why Mask-Only SWA Does NOT Improve Performance

There are two independent reasons.

### 1. Masking does not reduce compute

Even with SWA enabled:

- K/V tensors still have length `T_total`
- SDPA still computes attention against **all keys**
- The mask only zeroes out contributions *after* dot products

Unless keys/values are **evicted or sliced**, the attention kernel still performs full work.

---

### 2. Masked SDPA disables fast paths

When `sliding_window_size = None`, decode-step attention uses:

- `attn_mask = None`
- `is_causal = False`

This allows PyTorch to use its **fast SDPA / FlashAttention-style path**.

When SWA is enabled:

- An **explicit additive mask** is passed to SDPA
- This forces SDPA into a **more general, slower kernel path**

As a result, **mask-only SWA is strictly slower than baseline full attention**.

---

## Benchmark Evidence

The following benchmark measures **decode-step latency** with KV cache enabled.

### Command

```bash
python -m bench.swa_decode_bench
```

### Output

```
number of parameters: 123.69M
W=None  steps=128  total=0.6901s  ms/tok=5.391
number of parameters: 123.69M
W=1     steps=128  total=0.8123s  ms/tok=6.346
number of parameters: 123.69M
W=2048  steps=128  total=0.8036s  ms/tok=6.278
number of parameters: 123.69M
W=1024  steps=128  total=0.8092s  ms/tok=6.322
```

### Interpretation

- `W=None` (full attention) is the **fastest**
- All SWA configurations are **slower and nearly identical**
- Window size has **no impact on performance**

This confirms:

- No compute reduction is occurring
- Masked SDPA introduces overhead
- SWA here is a **semantic constraint only**

---

## Interaction with KV Cache

Even with SWA enabled:

```
(B, n_kv_head, T_total, head_dim)
```

- KV cache grows linearly with total sequence length
- Tokens outside the window:
  - Remain stored
  - Are never attended to
  - Still consume memory

This is **intentional and explicit**.

---

## Expected and Verified Properties

| Configuration | Expected Behavior | Status |
|--------------|------------------|--------|
| `W = None` | Baseline behavior | ✅ |
| `W = block_size` | Equivalent to baseline | ✅ |
| `W = 1` | Severe quality collapse | ✅ |
| Token-by-token vs chunked decode | Identical logits | ✅ |

---

## Why KV Eviction Is Not Implemented

KV eviction would:
- Reduce memory
- Reduce attention compute
- Introduce **significant complexity**

Specifically:
- RoPE position semantics become non-trivial
- Absolute vs relative position handling must change
- Cache bookkeeping becomes stateful and error-prone

For the purposes of **Deliverable B (architecture-level comparison)**, this complexity is unnecessary.

---

## Design Intent

This SWA implementation prioritizes:

- Semantic correctness
- Determinism
- Ease of reasoning
- Compatibility with MHA / GQA / MQA
- A clean baseline for architectural analysis

It intentionally stops short of production-grade SWA optimizations.

---

## What This Implementation Is (and Is Not)

**This is:**
- A correct SWA semantics implementation
- A validation tool
- An architectural reference point

**This is not:**
- A memory optimization
- A performance optimization
- A production SWA system

---

## Summary

- SWA is enforced via masking only
- KV cache size is unchanged
- Attention compute is unchanged
- Masked SDPA is slower than the fast causal path
- The benchmark results are expected and correct

This design is complete for its intended purpose. Further optimization belongs in a separate, explicitly systems-focused layer.

