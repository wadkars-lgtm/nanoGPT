"""
batch_infer.py

Single-run inference benchmark/profiling harness for nanoGPT.

What it can do:
1) Build a prefix of length prompt_len (random or model-generated), optionally cached to disk.
2) Time prefill on that prefix.
3) Time decode-only for max_new_tokens starting from that prefix.

Output:
Writes ONE JSON artifact per run under out_dir with semantic naming:
  infer_T{prompt_len}_B{batch_size}_KV{true|false}_P{phase}.json
Where phase is one of: both | prefill | decode

Defaults:
- phase = "both"  (best for benchmarking/sweeps)
- For profiling (Nsight), run phase="decode" or "prefill", warmup_iters=0, bench_iters=1
"""

import os
import json
import time
from contextlib import nullcontext
from typing import Any, Dict, Tuple

import torch
from model import GPTConfig, GPT
import hashlib

# -----------------------------------------------------------------------------
# Optional NVTX for Nsight filtering (works for both Nsight Systems + Compute)
try:
    import torch.cuda.nvtx as nvtx  # type: ignore
except Exception:
    nvtx = None

def nvtx_push(name: str, device_type: str) -> None:
    if nvtx is not None and device_type == "cuda":
        nvtx.range_push(name)

def nvtx_pop(device_type: str) -> None:
    if nvtx is not None and device_type == "cuda":
        nvtx.range_pop()

# -----------------------------------------------------------------------------
# Defaults (override via CLI / configurator.py)
batch_size = 4
prompt_len = 1024
max_new_tokens = 256

n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 50304

seed = 1337
device = "cuda"

# dtype can be: "float16", "bfloat16", "float32"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

compile = False

kv_cache = False

prefix_source = "random"   # "random" or "model"
prefix_seed_len = 1

warmup_iters = 2
bench_iters = 10

out_dir = "bench_out"
prefix_cache_dir = "bench_prefix"

# NEW: prefix caching toggle
# For profiling you often want this False to avoid disk/cache code.
cache_prefix = True

# NEW: phase control
# "both" (default): time prefill + decode
# "prefill": time only prefill
# "decode": time only decode-only
phase = "both"

# nanoGPT-style overrides (CLI: --key=value)
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

phase = str(phase).lower().strip()
if phase not in ("both", "prefill", "decode"):
    raise ValueError(f"phase must be one of both|prefill|decode, got: {phase}")

dtype = str(dtype).lower().strip()
if dtype not in ("float16", "bfloat16", "float32"):
    raise ValueError(f"dtype must be one of float16|bfloat16|float32, got: {dtype}")

device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(seed)
if device_type == "cuda":
    torch.cuda.manual_seed(seed)
    # TF32 only affects float32 matmul/conv on Ampere+; it does NOT change fp16/bf16 math.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Model init
gptconf = GPTConfig(
    block_size=prompt_len + max_new_tokens,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=0.0,
    bias=False,
)
model = GPT(gptconf).to(device).eval()

if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Utilities

def _reset_peak_mem():
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def _get_peak_mem() -> Tuple[int, int]:
    if device_type == "cuda":
        return (
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
        )
    return 0, 0

def _cuda_time_ms(fn) -> float:
    if device_type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    else:
        t0 = time.time()
        fn()
        return (time.time() - t0) * 1000.0

def _supports_kv_api(m) -> bool:
    import inspect
    sig = inspect.signature(m.forward)
    return ("use_cache" in sig.parameters) and ("past_kv" in sig.parameters)

# -----------------------------------------------------------------------------
# Prefix builders (optional caching)

def _prefix_cache_path() -> str:
    os.makedirs(prefix_cache_dir, exist_ok=True)
    key = {
        "prefix_source": prefix_source,
        "seed": seed,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "prefix_seed_len": prefix_seed_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dtype": dtype,
    }
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return os.path.join(prefix_cache_dir, f"prefix_{h}.pt")

@torch.no_grad()
def build_prefix_random() -> torch.Tensor:
    # purely random tokens (fastest, good for benchmarking mechanics)
    return torch.randint(vocab_size, (batch_size, prompt_len), device=device)

@torch.no_grad()
def build_prefix_model_generated() -> torch.Tensor:
    """
    Builds a prefix using the model's own next-token argmax loop.
    Uses KV to avoid O(T^2) generation when building long prefixes.
    """
    if not _supports_kv_api(model):
        raise RuntimeError("prefix_source=model requires KV cache support in model.forward (use_cache + past_kv).")

    seed_tok = torch.randint(vocab_size, (batch_size, prefix_seed_len), device=device)

    with ctx:
        nvtx_push("prefix_build_model_prefill_seed", device_type)
        logits, loss, past = model(seed_tok, None, use_cache=True, past_kv=None)  # type: ignore
        nvtx_pop(device_type)

        last = seed_tok[:, -1:].contiguous()
        tokens = [seed_tok]

        for _ in range(prompt_len - prefix_seed_len):
            nvtx_push("prefix_build_model_decode_step", device_type)
            logits, loss, past = model(last, None, use_cache=True, past_kv=past)  # type: ignore
            nvtx_pop(device_type)

            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(nxt)
            last = nxt

    return torch.cat(tokens, dim=1)

@torch.no_grad()
def get_or_build_prefix() -> torch.Tensor:
    if not cache_prefix:
        # no disk/cache logic at all (best for profiling purity)
        return build_prefix_model_generated() if prefix_source == "model" else build_prefix_random()

    path = _prefix_cache_path()
    if os.path.exists(path):
        prefix = torch.load(path, map_location=device)
        if tuple(prefix.shape) != (batch_size, prompt_len):
            raise RuntimeError(
                f"Cached prefix shape mismatch: got {tuple(prefix.shape)} expected {(batch_size, prompt_len)}"
            )
        return prefix

    prefix = build_prefix_model_generated() if prefix_source == "model" else build_prefix_random()
    torch.save(prefix, path)
    print(f"[INFO] Saved prefix cache: {os.path.abspath(path)}")
    return prefix

# -----------------------------------------------------------------------------
# Decode paths

@torch.no_grad()
def decode_only_no_kv(prefix: torch.Tensor):
    idx = prefix
    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model(idx, None)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)

@torch.no_grad()
def decode_only_kv(prefix: torch.Tensor):
    if not _supports_kv_api(model):
        raise RuntimeError("kv_cache=True requested but model.forward is not KV-capable.")

    nvtx_push("decode_kv_prefill", device_type)
    logits, loss, past = model(prefix, None, use_cache=True, past_kv=None)  # type: ignore
    nvtx_pop(device_type)

    last = prefix[:, -1:].contiguous()
    for _ in range(max_new_tokens):
        nvtx_push("decode_kv_step", device_type)
        logits, loss, past = model(last, None, use_cache=True, past_kv=past)  # type: ignore
        nvtx_pop(device_type)

        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        last = nxt

# -----------------------------------------------------------------------------
# Benchmark core

def bench_once(prefix: torch.Tensor) -> Dict[str, Any]:
    _reset_peak_mem()
    metrics: Dict[str, Any] = {}

    # Prefill timing (optional)
    if phase in ("both", "prefill"):
        def do_prefill():
            nvtx_push("prefill", device_type)
            with ctx:
                if kv_cache:
                    model(prefix, None, use_cache=True, past_kv=None)  # type: ignore
                else:
                    model(prefix, None)
            nvtx_pop(device_type)

        prefill_ms = _cuda_time_ms(do_prefill)
        metrics["prefill_ms"] = prefill_ms
        metrics["prefill_ms_per_tok"] = prefill_ms / float(prompt_len)

    # Decode-only timing (optional)
    if phase in ("both", "decode"):
        def do_decode():
            nvtx_push("decode", device_type)
            if kv_cache:
                decode_only_kv(prefix)
            else:
                decode_only_no_kv(prefix)
            nvtx_pop(device_type)

        decode_ms = _cuda_time_ms(do_decode)
        metrics["decode_total_ms"] = decode_ms
        metrics["decode_ms_per_tok"] = decode_ms / float(max_new_tokens)
        metrics["decode_tok_s"] = (1000.0 * float(max_new_tokens)) / decode_ms if decode_ms > 0 else 0.0

    peak_alloc, peak_reserved = _get_peak_mem()
    metrics["peak_alloc_bytes"] = peak_alloc
    metrics["peak_reserved_bytes"] = peak_reserved
    return metrics

# -----------------------------------------------------------------------------
# Main

def main():
    os.makedirs(out_dir, exist_ok=True)
    prefix = get_or_build_prefix()

    # Warmups
    for _ in range(int(warmup_iters)):
        bench_once(prefix)

    # Timed runs
    rows = [bench_once(prefix) for _ in range(int(bench_iters))]

    def avg(k: str) -> float:
        vals = [r[k] for r in rows if k in r]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    peak_alloc = max(r["peak_alloc_bytes"] for r in rows)
    peak_reserved = max(r["peak_reserved_bytes"] for r in rows)

    metrics_out: Dict[str, Any] = {
        "peak_alloc_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
    }

    # Only include metrics that were actually collected
    if phase in ("both", "prefill"):
        metrics_out["prefill_ms"] = avg("prefill_ms")
        metrics_out["prefill_ms_per_tok"] = avg("prefill_ms_per_tok")
    if phase in ("both", "decode"):
        metrics_out["decode_total_ms"] = avg("decode_total_ms")
        metrics_out["decode_ms_per_tok"] = avg("decode_ms_per_tok")
        metrics_out["decode_tok_s"] = avg("decode_tok_s")

    result = {
        "meta": {
            "device": device,
            "dtype": dtype,
            "torch_version": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "time_unix": time.time(),
        },
        "params": {
            "batch_size": batch_size,
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "kv_cache": kv_cache,
            "phase": phase,
            "prefix_source": prefix_source,
            "prefix_cache_dir": prefix_cache_dir,
            "cache_prefix": bool(cache_prefix),
            "prefix_seed_len": prefix_seed_len,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "vocab_size": vocab_size,
            "compile": bool(compile),
        },
        "metrics": metrics_out,
    }

    kv_str = "true" if kv_cache else "false"
    fname = f"infer_T{prompt_len}_B{batch_size}_KV{kv_str}_P{phase}.json"
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        print(f"[WARN] Overwriting existing file: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {out_path}")
    print("params:", json.dumps(result["params"], indent=2))
    print("metrics:", json.dumps(result["metrics"], indent=2))

if __name__ == "__main__":
    main()
