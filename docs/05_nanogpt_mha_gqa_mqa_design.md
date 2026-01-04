# nanoGPT: MHA vs GQA vs MQA (Grouped / Multi-Query Attention)

This doc describes a **minimal, repo-local** extension to nanoGPT’s `CausalSelfAttention` so you can train
and compare:

- **MHA** (Multi-Head Attention): `n_kv_head = n_head`
- **GQA** (Grouped-Query Attention): `1 < n_kv_head < n_head`
- **MQA** (Multi-Query Attention): `n_kv_head = 1`

The key idea: **keep the number of query heads the same (`n_head`)**, but reduce the number of
**key/value heads (`n_kv_head`)**. During attention, **K/V are broadcast (repeated) across query heads**.

---

## What changes in the model

### New config field

Add `n_kv_head` to `GPTConfig`.

- If `n_kv_head is None`, default to `n_head` (standard MHA).
- Require:
  - `n_embd % n_head == 0`
  - `n_head % n_kv_head == 0` (so KV heads can be evenly shared across Q heads)

### Modified projections

Instead of a single `c_attn` projecting to `3 * n_embd`, use:

- `c_q`: `n_embd -> n_embd` (queries keep full head count)
- `c_kv`: `n_embd -> 2 * (n_kv_head * head_dim)` (keys/values use fewer heads)

This reduces K/V parameter count and compute, while keeping output shape identical.

### Attention computation

Shapes:
- `q`: `(B, n_head, T, head_dim)`
- `k,v`: `(B, n_kv_head, T, head_dim)`

Broadcast K/V:
- `k = k.repeat_interleave(g, dim=1)`
- `v = v.repeat_interleave(g, dim=1)`
where `g = n_head // n_kv_head`

Then run the same attention path you already have (Flash / SDPA fastpath if enabled), because final
head dimension count matches `n_head`.

---

## Training and saving: comparable checkpoints

### New CLI flags in `train.py`

Add:

- `--ckpt_name=<name>`: base filename for checkpoints inside `out_dir`
  - default: `ckpt`
  - files written:
    - `<ckpt_name>.pt` (latest)
    - `<ckpt_name>_best.pt` (best val loss)

This lets you keep **all variants in the same `out_dir`** and compare them later.

---

## Example commands

Below assumes you already have a standard config file like `config/train_shakespeare_char.py`.

### MHA (baseline)

```powershell
python -m bench.train  config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mha `
  --n_head=12 `
  --n_kv_head=12
```

### GQA (e.g., 2 KV heads, 6 Q heads)

```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_kv2 `
  --n_head=12 `
  --n_kv_head=3
```

### MQA (1 KV head)

```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mqa `
  --n_head=12 `
  --n_kv_head=1
```

**Rule of thumb:** Keep `n_layer`, `n_embd`, `n_head`, training steps identical across runs.
Only change `n_kv_head`.

---

## Accuracy comparison

| Variant | n_head | n_kv_head | Params (M) | best val loss | iter @ best |
|---|---:|---:|-----------:|--------------:|------------:|
|mha|12|12|      10.67 |          0.75 |        5000 |
|gqa|12|6|       82.5 |          0.75 |        5000 |
|mqa|12|1|       27.5 |              0.75 |        5000 |

---

## Expected results pattern

- **MQA**: biggest KV-cache benefit at inference (smallest KV per token), but might slightly hurt quality at a fixed 
model size unless you compensate elsewhere.
- **GQA**: usually the sweet spot: most KV-cache benefit with less quality hit.
- **MHA**: best flexibility/quality at fixed size, worst KV memory scaling.



