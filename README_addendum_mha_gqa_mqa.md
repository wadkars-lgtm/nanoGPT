# nanoGPT MHA / GQA / MQA + KV-Cache Runbook (End-to-End)

This runbook assumes your repo root (`{ROOT}`) contains `train.py`, `model.py`, `config/`, and `data/` (standard nanoGPT layout).
We add a few helper scripts **at repo root** (same level as `train.py`), and tests under `{ROOT}/tests`.

---

## 0) File placement (what goes where)

### `{ROOT}/tests/`
- `tests/test_attention_correctness.py`  
  - Contains **baseline equivalence** and **cache equivalence** tests.

###  `{ROOT}/` (same level as `train.py`)
- `eval_loss.py`  
  - Evaluates **val loss** + **perplexity** from a checkpoint.
- `sweep_n_kv_head.py`  
  - Sweeps `n_kv_head` across multiple prompt lengths and batch sizes, calling your benchmark script.
- `sample.py`  
  - Loads a trained checkpoint and **generates text** (sanity-check that the model can run inference end-to-end).
- (optional) `README_mha_gqa_mqa.md` (this file, if you want it in-repo)



---

## 1) Preconditions

### Python env
Use your existing venv/conda that runs nanoGPT training.

Minimum dependencies:
- `torch`
- `numpy`
- `pytest` (for tests)

Install pytest if needed:
```bash
pip install pytest
```

### Dataset
This runbook assumes the char-level Shakespeare dataset exists at:
```
{ROOT}/data/shakespeare_char/train.bin
{ROOT}/data/shakespeare_char/val.bin
{ROOT}/data/shakespeare_char/meta.pkl   (optional but typical)
```

If it doesn’t exist, generate it using nanoGPT’s data prep steps (whatever your repo already uses).

---

## 2) Sanity check: CLI overrides actually work

nanoGPT typically applies `--key=value` overrides via `configurator.py`.

Run this once to confirm your overrides actually take effect:
```bash
#Absolute pos
python -m bench.train config/train_shakespeare_char.py --n_head=12 --n_kv_head=3 --max_iters=1
#Rope based
python -m bench.train config/train_shakespeare_char.py --n_head=12 --n_kv_head=3 --max_iters=1 --use_rope=True --ckpt_name=ckpt_rope_smoke
```

You should see logging that reflects your chosen head counts (or add a one-line print in `train.py` to confirm).

If overrides are ignored, your comparisons are invalid until fixed.

---

## 3) Correctness tests

### 3.1 Add the tests

Create `{ROOT}/tests/test_attention_correctness.py` with the content you already have (baseline + cache equivalence).

### 3.2 Run tests (CPU)
```bash
pytest -q
```

#### What these tests guarantee
- **Baseline equivalence**: `n_kv_head=None` behaves exactly like `n_kv_head=n_head` (MHA).
- **Cache equivalence**: logits computed via:
  - full forward on `(prompt + next_token)`
  - vs prefill(prompt) + decode(next_token) using your KV cache  
  match within a tight tolerance.

If cache equivalence fails, do **not** benchmark. Fix correctness first.

---

## 4) Training runs (controlled set)

You said your baseline is `n_head=12`. Valid `n_kv_head` values must divide 12:
- 12, 6, 4, 3, 2, 1

All commands below keep `n_head=12` and only change `n_kv_head`.

> `--ckpt_name=...` is used to create distinct checkpoint names so you can compare models cleanly.

### 4.1 MHA baseline (12 Q heads / 12 KV heads)
```bash
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mha_h12_kv12 `
  --n_head=12 `
  --n_kv_head=12 `
  --use_rope=False

#iter 5000: loss 0.7437, lr 1.000e-04, dt 9499.10ms
```
```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mha_h12_kv12_rope `
  --n_head=12 `
  --n_kv_head=12 `
  --use_rope=True
```

**What it does**
- Trains a baseline MHA model.
- Saves checkpoints under `out-attn/` with a name prefix that includes `mha_h12_kv12`.

### 4.2 GQA (example: 12 Q heads / 3 KV heads)
```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --n_head=12 `
  --n_kv_head=3 `
  --use_rope=False
  
  #iter 5000: loss 0.7946, lr 1.000e-04, dt 9521.81ms
```

```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3_rope `
  --n_head=12 `
  --n_kv_head=3 `
  --use_rope=True
```

**What it does**
- Trains a GQA variant where 4 query heads share each KV head.
- Smaller KV cache than MHA, and typically better decode efficiency at long context.

### 4.3 MQA (12 Q heads / 1 KV head)

```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mqa_h12_kv1 `
  --n_head=12 `
  --n_kv_head=1 `
  --use_rope=False

#iter 5000: loss 0.8281, lr 1.000e-04, dt 9220.43ms
```

```powershell
python -m bench.train config/train_shakespeare_char.py `
  --out_dir=out-attn `
  --ckpt_name=mqa_h12_kv1_rope `
  --n_head=12 `
  --n_kv_head=1 `
  --use_rope=True
```

**What it does**
- Trains an MQA variant with minimal KV cache footprint.
- Usually the biggest decode-time win, but may reduce model quality depending on task/model size.

---

## 5) Evaluate accuracy: loss + perplexity

### 5.1 Add eval script at repo root

Create `{ROOT}/eval_loss.py` with the provided `eval_loss.py` content.

This script:
- loads a trained nanoGPT checkpoint
- runs evaluation on the validation split
- reports loss and perplexity

---

### 5.2 Run eval on best checkpoints

Your training loop typically writes some combination of:
- `..._best.pt`
- `ckpt.pt`
- `..._final.pt`

Pick the **best** checkpoint for each attention variant.

```bash
python -m bench.eval_loss --ckpt out-attn/mha_h12_kv12_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/mha_h12_kv12_rope_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_rope_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/mqa_h12_kv1_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/mqa_h12_kv1_rope_best.pt --dataset shakespeare_char
```
**What it does**
- Loads the checkpoint
- Evaluates `val_loss` over `--eval_iters` random batches
- Prints:
  - `val_loss=...`
  - `val_ppl=...`

### 5.3 Aggregate results for plotting

For convenience, you may record the outputs into a small CSV manually or via shell redirection:
```powershell
python -m bench.eval_loss --ckpt out-attn/mha_h12_kv12_best.pt --dataset shakespeare_char > eval_mha.txt
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_best.pt  --dataset shakespeare_char > eval_gqa.txt
python -m bench.eval_loss --ckpt out-attn/mqa_h12_kv1_best.pt  --dataset shakespeare_char > eval_mqa.txt
```

---

## 5) Evaluate accuracy: loss + perplexity

### 5.1 Add eval script at repo root

Create `{ROOT}/eval_loss.py` with the provided `eval_loss.py` content.

This script:
- loads a trained nanoGPT checkpoint
- runs evaluation on the validation split
- reports loss and perplexity

---

### 5.2 Run eval on best checkpoints

Your training loop typically writes some combination of:
- `..._best.pt`
- `ckpt.pt`
- `..._final.pt`

Pick the **best** checkpoint for each attention variant.

```bash
python -m bench.eval_loss --ckpt out-attn/mha_h12_kv12_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/mqa_h12_kv1_best.pt --dataset shakespeare_char
```

**What it does**
- Loads the checkpoint
- Evaluates `val_loss` over `--eval_iters` random batches
- Prints:
  - `val_loss=...`
  - `val_ppl=...`

---

### 5.3 Automated aggregation (recommended)

Instead of manually redirecting stdout, use the helper script
`run_eval_compare.py` to automate evaluation and aggregation.

This script:
- runs `eval_loss.py` for each checkpoint
- parses `val_loss` and `val_ppl` from stdout
- writes a single comparison CSV
- saves raw eval logs for auditability

```powershell
python -m bench.run_eval_compare `
  --dataset shakespeare_char `
  --out_dir results\gqa `
  --labels MHA,MHA_ROPE,GQA,GQA_ROPE,MQA,MQA_ROPE `
  --n_kv_heads 12,12,3,3,1,1 `
  --ckpts out-attn/mha_h12_kv12_best.pt,out-attn/mha_h12_kv12_rope_best.pt,out-attn/gqa_h12_kv3_best.pt,out-attn/gqa_h12_kv3_rope_best.pt,out-attn/mqa_h12_kv1_best.pt,out-attn/mqa_h12_kv1_rope_best.pt
```

This produces:
- `results/gqa/eval_compare.csv`
- `results/gqa/eval_logs/*.txt`

---

### 5.4 Plot accuracy comparison

Use `plot_eval_compare.py` to generate accuracy plots directly from the CSV:

```powershell
python -m bench.plot_eval_compare `
  --csv results\gqa\eval_compare.csv `
  --out_dir results\gqa\plots
```

Generated plots:
- `val_loss_vs_n_kv_head.png`
- `val_ppl_vs_n_kv_head.png`

These visualize the **quality ↔ KV-cache efficiency tradeoff**:
- **MHA** → best quality, largest KV cache
- **GQA** → strong quality/efficiency balance
- **MQA** → smallest KV cache, possible quality drop

This complements the decode-latency plots from Section 7 and completes the
accuracy–performance story.

---

### 5.5 Make comparisons fair

Use the **same**:
- dataset
- block size
- training steps / `max_iters`
- learning rate schedule
- seed (if your `train.py` supports `--seed=`; otherwise accept small noise)

---

## 6) Generate samples from the trained checkpoints (sanity-check inference)

This is a quick end-to-end “does the model actually run and produce text?” check using `sample.py`.

### 6.1 Why this matters
- It catches issues that **training + eval** can miss (tokenizer mismatch, wrong checkpoint name, bad `vocab_size`, etc.).
- If you see CUDA indexing asserts like `Indexing.cu: Assertion srcIndex < srcSelectDimSize failed`, it almost always means:
  - you are using the wrong tokenizer for a char-level model, or
  - your prompt encodes to token IDs >= `vocab_size`.

**For `shakespeare_char`, you should have:**
```
data/shakespeare_char/meta.pkl
```
and `sample.py` should use it automatically when `--dataset=shakespeare_char` is provided.

### 6.2 Commands (PowerShell)

#### MHA
```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=mha_h12_kv12 `
  --dataset=shakespeare_char `
  --start="`n" `
  --num_samples=3 `
  --max_new_tokens=300 `
  --temperature=0.8 `
  --top_k=200
```

#### GQA
```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --dataset=shakespeare_char `
  --start="`n" `
  --num_samples=3 `
  --max_new_tokens=300 `
  --temperature=0.8 `
  --top_k=200
```

#### MQA
```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=mqa_h12_kv1 `
  --dataset=shakespeare_char `
  --start="`n" `
  --num_samples=3 `
  --max_new_tokens=300 `
  --temperature=0.8 `
  --top_k=200
```

#### Prompt from a file
```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=mha_h12_kv12 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --num_samples=2 `
  --max_new_tokens=200
```

```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --num_samples=2 `
  --max_new_tokens=200
```

```powershell
python -m bench.sample `
  --out_dir=out-attn `
  --ckpt_name=mqa_h12_kv1 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --num_samples=2 `
  --max_new_tokens=200
```

---

## 7) Benchmark sweep for `n_kv_head`

This is the performance half: measure decode/prefill changes across KV head counts.

### 7.1 Add sweep script at repo root
Create `{ROOT}/sweep_n_kv_head.py` with the provided sweep script content.

### 7.2 Identify your benchmark entrypoint
The sweep script calls a benchmark script (default is `batch_infer.py`).

Make sure one of these is true:
- You have `{ROOT}/batch_infer.py`
- OR you edit `--script=...` to point at your benchmark script

Your benchmark script must accept these flags (or you modify sweep script accordingly):
- `--phase=decode|prefill`
- `--kv_cache=true|false`
- `--prompt_len=...`
- `--batch_size=...`
- `--max_new_tokens=...`
- `--warmup_iters=...`
- `--bench_iters=...`
- `--n_head=...`
- `--n_kv_head=...`

### 7.3 Run the sweep (decode + KV cache enabled)
```powershell
python -m bench.sweep_n_kv_head `   
      --script=batch_infer.py   `
      --phase=decode            ` 
      --kv_cache=true           `
      --n_head=12               `
      --n_kv_heads=12,6,4,3,2,1 `   
      --prompt_lens=128,512,1024,2048 `   
      --batch_sizes=1,4,8,16,32 `
```

**What it does**
- Runs a grid over:
  - `n_kv_head` in {12,6,4,3,2,1}
  - prompt lengths
  - batch sizes
- Captures stdout for each run into:
  - `results/gqa/rawlogs/*.log`
- Writes an index CSV:
  - `results/gqa/sweep.csv`

**Note:**
#### Important clarification

The command below **does not use your trained checkpoints** unless your `batch_infer.py` explicitly loads them 
(most nanoGPT-style benchmark scripts do **not**):

```powershell
python -m bench.sweep_n_kv_head '
  --script=batch_infer.py '
  --phase=decode '
  --kv_cache=true '
  --n_head=12 '
  --n_kv_heads=12,6,4,3,2,1 '
  --prompt_lens=128,512,1024,2048 '
  --batch_sizes=1,4,8,16,32
```

Instead, this is a **microbenchmark / synthetic sweep**.

---

##### What this sweep is actually doing

This sweep repeatedly runs your benchmark script (`batch_infer.py`) while varying:

- `n_kv_head ∈ {12, 6, 4, 3, 2, 1}`
- prompt length
- batch size

It answers the systems-level question:

> **“If I instantiate a model with `n_head = 12` and different `n_kv_head` values, how does decode performance change as context length and batch size grow?”**

This is about **hardware and memory behavior**, not model quality.

---

##### Why this is still valid even if you trained only `n_kv_head = 3`

Decode-time performance is dominated by:

- attention math
- KV-cache memory traffic
- tensor shapes and memory access patterns

These are driven primarily by:
- `n_head`
- `n_kv_head`
- sequence length (`T`)
- batch size

—not by how well the model is trained.

So the sweep is comparing **systems behavior** across attention layouts:

- **MHA**: `n_kv_head = 12`
- **GQA**: `n_kv_head = 6, 4, 3, 2`
- **MQA**: `n_kv_head = 1`

This makes it a valid and useful **performance study**, even with randomly initialized weights.

---

###### The catch (important)

What this sweep *actually measures* depends on how `batch_infer.py` initializes the model:

- If `batch_infer.py` **initializes random weights** (common):
  - ✅ The sweep is purely **shape- and bandwidth-driven**
  - ✅ Ideal for profiling decode behavior and KV-cache scaling

- If `batch_infer.py` **loads a trained checkpoint** whose `model_args` are fixed to `n_kv_head = 3`:
  - ❌ Running with `n_kv_head = 12, 6, 4, 2, 1` will either:
    - fail to load weights, or
    - silently benchmark something other than what you think

Always confirm how the model is initialized inside `batch_infer.py`.

---

###### Benchmarking trained models (different goal)

If your goal is to benchmark **your trained models**, then you should **sweep over checkpoints**, not over `n_kv_head` values.

In that case:
- Each run loads a specific checkpoint (MHA / GQA / MQA)
- `n_kv_head` comes from `model_args` inside the checkpoint
- You compare decode latency across *trained variants*


```powershell
python -m bench.batch_infer --out_dir=out-attn --ckpt_name=mha_h12_kv12 --phase=decode --kv_cache=true --prompt_len=2048 --batch_size=8 --max_new_tokens=128
python -m bench.batch_infer --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --phase=decode --kv_cache=true --prompt_len=2048 --batch_size=8 --max_new_tokens=128
python -m bench.batch_infer --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --phase=decode --kv_cache=true --prompt_len=2048 --batch_size=8 --max_new_tokens=128
```

### 7.4 Optional: run prefill sweep (KV cache doesn’t matter much for prefill)
```powershell
python -m bench.sweep_n_kv_head `   
  --script=batch_infer.py `   
  --phase=prefill   `
  --kv_cache=true   `
  --n_head=12   `
  --n_kv_heads=12,6,4,3,2,1 `   
  --prompt_lens=128,512,1024,2048 `   
  --batch_sizes=1,4,8,16,32 
```

### 7.5) Plot and visualize sweep results

After the sweep completes, you will have:

```
results/gqa/
  ├── sweep.csv
  ├── rawlogs/
  │     ├── decode_h12_kv12_T2048_B8.log
  │     └── ...
```

At this point you have **data**, but no plots yet.

This step converts the raw sweep outputs into **readable performance graphs**.

---

#### 7.5.1 What `plot_sweep.py` does

 
**What this script does**
- Reads `results/gqa/sweep.csv`
- Parses each corresponding log file in `results/gqa/rawlogs/`
- Extracts timing / throughput / memory metrics via regex
- Generates PNG plots under:
  ```
  results/gqa/plots/
  ```
- (Optional) writes a fully parsed table for debugging:
  ```
  results/gqa/parsed.csv
  ```

> This script runs **on CPU only**. It does **not** touch CUDA.

---

#### 7.5.2 Run the plotting step (PowerShell)

From repo root:

```powershell
python -m bench.plot_sweep --csv results\gqa\sweep.csv --write_parsed_csv
```

Expected outputs:

```
results/gqa/
  ├── plots/
  │     ├── decode_ms_per_tok_vs_promptlen_B8.png
  │     ├── decode_ms_per_tok_vs_batchsize_T2048.png
  │     └── ...
  ├── parsed.csv
```

---

#### 7.5.3 What plots you should expect

Depending on what your `batch_infer.py` prints, the script will automatically select the best available metric:

- **Decode**
  - `ms / token` (preferred)
  - OR `tokens / second`
  - OR total `decode_ms`

Typical plots:

1. **Decode latency vs prompt length**
   - One curve per attention layout (MHA / GQA / MQA)
   - Fixed batch size

2. **Decode latency vs batch size**
   - One curve per attention layout
   - Fixed prompt length

3. **Peak memory vs prompt length** (if logged)

These plots are where the **KV-cache scaling story becomes visually obvious**.

---

#### 7.5.4 If no metrics are detected (important)

If the script errors with:

```
No usable metrics found in logs
```

That means your `batch_infer.py` prints timings using different strings.

Fix is trivial:

1. Open any log file:
   ```
   results/gqa/rawlogs/decode_h12_kv3_T2048_B8.log
   ```
2. Find the exact timing line (e.g. `decode_ms/tok: 42.3`)
3. Update the regex patterns at the top of `plot_sweep.py`

You only need to do this **once**.

---

#### 7.5.5 What this step gives you (why it matters)

After this step, you have:

- Quantitative, visual evidence that:
  - MQA reduces KV-cache bandwidth pressure
  - GQA hits a quality / performance sweet spot
- Publication-ready plots suitable for:
  - blog posts
  - design docs
  - internal performance reviews
  - interviews

This turns your sweep from *“I ran benchmarks”* into  
**“I can explain the systems behavior of attention variants.”**

---

## 8) Interpreting results (what to expect)

### Decode phase
- As `n_kv_head` decreases (MHA → GQA → MQA), **KV cache shrinks**.
- This often improves **decode ms/token** at long context because:
  - less KV memory traffic
  - better cache locality / less bandwidth pressure
- Compute (QK^T, softmax, etc.) is still there, but KV reads become cheaper.

### Prefill phase
- Prefill is dominated by full attention on the whole prompt.
- `n_kv_head` can still matter, but improvements are usually smaller than decode.

### Accuracy
- On small models/datasets, MQA can degrade quality noticeably.
- GQA often hits a better tradeoff: smaller KV cache without dramatic quality loss.

---

## 9) Minimal “end-to-end” command list

Run these in order:

1) Tests:
```bash
pytest -q
```

2) Train:
```bash
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mha_h12_kv12 --n_head=12 --n_kv_head=12 --max_iters=1000 
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --n_head=12 --n_kv_head=3
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --n_head=12 --n_kv_head=1
```


```bash
#With Layer Norm
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mha_h12_kv12 --n_head=12 --n_kv_head=12 --norm_type=layernorm --use_rope=False --block_size=2049
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --n_head=12 --n_kv_head=3 --norm_type=layernorm --use_rope=False --block_size=2049
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --n_head=12 --n_kv_head=1 --norm_type=layernorm --use_rope=False --block_size=2049

#With RMS Norm
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mha_h12_kv12 --n_head=12 --n_kv_head=12 --norm_type=rmsnorm --use_rope=False --block_size=2049
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --n_head=12 --n_kv_head=3 --norm_type=rmsnorm --use_rope=False --block_size=2049
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --n_head=12 --n_kv_head=1 --norm_type=rmsnorm --use_rope=False --block_size=2049


#With Layer Norm and ROPE
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mha_h12_kv12 --n_head=12 --n_kv_head=12 --norm_type=layernorm  --use_rope=True
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --n_head=12 --n_kv_head=3 --norm_type=layernorm  --use_rope=True
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --n_head=12 --n_kv_head=1 --norm_type=layernorm  --use_rope=True

#With RMS Norm and ROPE
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mha_h12_kv12 --n_head=12 --n_kv_head=12 --norm_type=rmsnorm  --use_rope=True
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --n_head=12 --n_kv_head=3 --norm_type=rmsnorm  --use_rope=True
python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --n_head=12 --n_kv_head=1 --norm_type=rmsnorm  --use_rope=True

```


3) Eval:
```bash
python -m bench.eval_loss --ckpt out-attn/mha_h12_kv12_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_best.pt  --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/mqa_h12_kv1_best.pt  --dataset shakespeare_char

python -m bench.eval_loss --ckpt out-attn/qqa_h12_kv12_rmsnorm_best.pt --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/gqa_h12_kv3_rmsnorm_best.pt  --dataset shakespeare_char
python -m bench.eval_loss --ckpt out-attn/qqa_h12_kv1_rmsnorm_best.pt  --dataset shakespeare_char

```

4) Sample (PowerShell):
```powershell
python -m bench.sample --out_dir=out-attn --ckpt_name=mha_h12_kv12 --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200
python -m bench.sample --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200
python -m bench.sample --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200

python -m bench.sample --out_dir=out-attn --ckpt_name=mha_h12_kv12 --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200
python -m bench.sample --out_dir=out-attn --ckpt_name=gqa_h12_kv3  --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200
python -m bench.sample --out_dir=out-attn --ckpt_name=mqa_h12_kv1  --dataset=shakespeare_char --start="`n" --num_samples=1 --max_new_tokens=200

```

5) Sweep:
```bash
python -m bench.sweep_n_kv_head --script=batch_infer.py --phase=decode --kv_cache=true   --n_head=12 --n_kv_heads=12,6,4,3,2,1 --prompt_lens=128,512,1024,2048 --batch_sizes=1,4,8,16,32
```

```bash

python -m bench.sweep_n_kv_head --out_csv results/gqa --script=bench\batch_infer.py --phase=decode --kv_cache=true --max_new_tokens=32  --warmup_iters=3 --bench_iters=5 --n_head=12 --n_kv_heads=12,6,4,3,2,1 --prompt_lens=128,512,1024,2048 --batch_sizes=1,4,8,16,32 --norm_type=rmsnorm --use_rope=True --ignore_checkpoint=true --allow_unsafe_benchmark=true 
python -m bench.sweep_n_kv_head --out_csv results/gqa --script=bench\batch_infer.py --phase=decode --kv_cache=true  --max_new_tokens=32 --warmup_iters=3 --bench_iters=5 --n_head=12 --n_kv_heads=12,6,4,3,2,1 --prompt_lens=128,512,1024,2048 --batch_sizes=1,4,8,16,32 --norm_type=layernorm --use_rope=True --ignore_checkpoint=true   --allow_unsafe_benchmark=true 

```

6) Plot:
```bash
python -m bench.plot_sweep --csv results/gqa/sweep_rmsnorm_norope.csv --write_parsed_csv --norm_type=rmsnorm --use_rope=False
python -m bench.plot_sweep --csv results/gqa/sweep_layernorm_norope.csv --write_parsed_csv --norm_type=layernorm --use_rope=False

python -m bench.plot_sweep --csv results/gqa/sweep_rmsnorm_rope.csv --write_parsed_csv --norm_type=rmsnorm --use_rope=True
python -m bench.plot_sweep --csv results/gqa/sweep_layernorm_rope.csv --write_parsed_csv --norm_type=layernorm --use_rope=True

```

```bash
python -m bench.plot_sweep --csv results/gqa/sweep.csv --write_parsed_csv --norm_type=layernorm --use_rope=False
```
---

## 10) Common failure modes (fast triage)

- **`n_head % n_kv_head != 0`**  
  You chose an invalid grouping. With `n_head=12`, only use `12,6,4,3,2,1`.

- **Cache equivalence test fails**  
  Your cache concat / mask / head expansion has a bug. Fix before benchmarking.

- **Overrides ignored**  
  You’re not using nanoGPT’s `configurator.py` override pattern correctly. Confirm `--key=value` is honored.

- **Checkpoint filenames don’t match**  
  Your `train.py` may name checkpoints differently. Point `eval_loss.py --ckpt` at the actual produced file.

- **CUDA indexing asserts during `sample.py`**  
  Usually tokenizer mismatch: ensure `data/{dataset}/meta.pkl` exists for char-level datasets and run `sample.py` with `--dataset=shakespeare_char`.