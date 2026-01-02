# Rotary Positional Embeddings (RoPE) — Design Notes

## 1. Motivation

Transformer attention needs positional information to distinguish token order.
Classic **absolute positional embeddings** encode position as an additive vector,
but this approach breaks down for:

- Long contexts
- Sliding windows
- KV caching across segments
- Length generalization beyond training context

**RoPE (Rotary Positional Embeddings)** addresses these issues by encoding
*relative position* directly into the attention mechanism.

---

## 2. Core Idea

RoPE represents position by **rotating query and key vectors** in a complex plane.

Each pair of dimensions in Q and K is treated as a 2D vector and rotated by an
angle proportional to the token position.

This means:

- No positional embeddings are added to the residual stream
- Position information only affects **Q·Kᵀ**
- Relative distances between tokens are preserved

---

## 3. Mathematical Intuition (Lightweight)

For each head dimension pair:

- Position `p` defines an angle θ(p)
- `(x₁, x₂)` is rotated as:

```
(x₁', x₂') = (
  x₁ cos θ − x₂ sin θ,
  x₁ sin θ + x₂ cos θ
)
```

Crucially:

- Q and K are rotated with the **same angle**
- V is **not rotated**
- The dot product encodes relative offset

---

## 4. Why RoPE Works for Relative Positioning

When computing attention:

```
(Q_rotated · K_rotatedᵀ)
```

The rotation cancels absolute position and leaves **relative displacement**.

Effectively:

- Token at position 37 attending to 50
- Behaves like relative distance = +13
- Without storing absolute positions explicitly

---

## 5. Interaction with KV Cache

RoPE is **KV-cache friendly** because:

- K and V are stored *after* rotation
- Q is rotated at decode time using its position
- Cached K remains valid across decode steps

This avoids recomputing positional encodings for old tokens.

---

## 6. RoPE vs Absolute Position Embeddings

| Property | Absolute | RoPE |
|--------|----------|------|
| Additive | Yes | No |
| Relative awareness | Weak | Strong |
| KV cache friendly | Poor | Excellent |
| Sliding windows | Fragile | Natural |
| Length extrapolation | Poor | Good |

---

## 7. Sliding Window Compatibility

RoPE naturally supports **sliding window attention** *if* positions are
handled carefully.

Key constraint:

- The **relative offset** must be preserved
- Absolute indices may be rebased or clipped

Incorrect handling leads to the classic
“RoPE + SWA position drift” failure mode.

---

## 8. Current nanoGPT Integration

In this codebase:

- RoPE is **optional** via `use_rope`
- Rotation is applied inside attention
- No learned positional embeddings are used when enabled
- Compatible with:
  - MHA
  - GQA
  - MQA
  - KV caching

---

## 9. Known Limitations

Current implementation does **not** include:

- NTK scaling
- YaRN / LongRoPE
- Dynamic base adjustment
- Cross-window position rebasing helpers

These are intentionally excluded for clarity and correctness.

---

## 10. Design Philosophy

RoPE is treated as:

> A *geometric constraint on attention*, not an embedding trick.

This keeps:
- Semantics in the model weights
- Position as a structural property
- Memory usage minimal

---

## 11. Relationship to SWA Document

This document is intended to precede:

**02 — Sliding Window Attention (SWA) Design**

RoPE provides the positional foundation.
SWA constrains *which* tokens may attend.

The two are orthogonal but deeply complementary.
