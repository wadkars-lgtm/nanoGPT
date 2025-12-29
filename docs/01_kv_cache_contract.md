# KV Cache Contract (nanoGPT + KV augmentation)

This document defines the **exact contract** for KV cache usage in this nanoGPT codebase.
It is intentionally concrete: shapes, list structure, and behavioral rules are explicit.

This is the foundation for understanding attention, masking, and incremental decoding.


## A) What KV cache is in *this* codebase

KV cache stores the **Key (K)** and **Value (V)** tensors produced by the self-attention
layers for **past tokens**, so they do not need to be recomputed during generation.

Each Transformer layer owns its own KV cache.
During decoding, new queries attend over:
- cached K/V from previous tokens
- newly computed K/V for the current step or chunk

This reduces decode-time complexity from **O(T²)** to **O(T)** per generated token.


## B) Exact API contract

### GPT.forward signature

```python
GPT.forward(
    idx,
    targets=None,
    use_cache: bool = False,
    past_kv=None
)
```

### Return values

#### use_cache = False (default)
```text
returns: (logits, loss)
```

- logits: (B, T_new, vocab_size) during training
- logits: (B, 1, vocab_size) during inference
- loss: scalar or None

#### use_cache = True
```text
returns: (logits, loss, present_kv)
```

- present_kv: list of length `n_layer`
- each element is a tuple `(k, v)`


### KV list structure

```text
past_kv / present_kv:
[
  (k_0, v_0),   # layer 0
  (k_1, v_1),   # layer 1
  ...
  (k_{n-1}, v_{n-1})
]
```

- If no cache exists yet: `past_kv = [None] * n_layer`


## C) Shape invariants

Let:
- B = batch size
- T_new = number of new tokens in this forward pass
- T_past = number of cached tokens
- T_total = T_past + T_new
- C = n_embd
- n_head = number of attention heads
- head_dim = C / n_head

### Input
```text
idx: (B, T_new)
x:   (B, T_new, C)
```

### Attention projections (per layer)

Before concatenation:
```text
q, k, v: (B, n_head, T_new, head_dim)
```

Cached tensors:
```text
k_past, v_past: (B, n_head, T_past, head_dim)
```

After concatenation:
```text
k_total, v_total: (B, n_head, T_total, head_dim)
```

These `(k_total, v_total)` tensors become `present_kv[layer_id]`.


## D) Masking rules (three regimes)

### 1) Prefill (past_kv is None)
- Full sequence provided
- **Causal mask REQUIRED**
- Prevents attending to future tokens

```text
Reason: keys include future positions within the same input
```

---

### 2) Decode-step (past_kv exists, T_new == 1)
- Single new token
- **No causal mask needed**

```text
Reason: all keys are from the past or current token
No future positions exist
```

---

### 3) Chunked decode (past_kv exists, T_new > 1)
- Multiple new tokens appended at once
- **Offset causal mask REQUIRED**

```text
Rule: token i may only attend to keys <= (T_past + i)
Reason: prevent tokens inside the chunk from seeing "future" tokens in the same chunk
```


## E) Concrete toy example

Assume:
```text
B = 2
n_layer = 3
n_head = 4
n_embd = 32
head_dim = 8
```

### Case: decode-step
```text
T_past = 10
T_new  = 1
T_total = 11
```

Cached input:
```text
past_kv[0][0] (k): (2, 4, 10, 8)
past_kv[0][1] (v): (2, 4, 10, 8)
```

Current projections:
```text
q, k, v: (2, 4, 1, 8)
```

After concatenation:
```text
present_kv[0][0]: (2, 4, 11, 8)
present_kv[0][1]: (2, 4, 11, 8)
```

Masking:
```text
No mask needed (decode-step)
```


## Summary

- KV cache stores **per-layer K/V tensors**
- Cache structure is a **list of (k,v) tuples**
- Shapes are invariant and must be preserved
- Masking behavior depends strictly on `(past_kv, T_new)`
- Position offsets and masking correctness are critical to correctness
