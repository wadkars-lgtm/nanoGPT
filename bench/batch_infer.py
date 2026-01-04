"""
batch_infer.py

Single-run inference benchmark/profiler harness for nanoGPT.

What it can do:
1) Build a prefix of length prompt_len (random or model-generated), optionally cached to disk.
2) Time prefill on that prefix.
3) Time decode-only for max_new_tokens starting from that prefix.

Output:
Writes ONE JSON artifact per run under out_dir with semantic naming:
  infer_h{n_head}_kv{n_kv_head}_T{prompt_len}_B{batch_size}_KV{true|false}_P{phase}{_rmsnorm_if_any}{_rope|_norope}.json
Where phase is one of: both | prefill | decode

Also writes ONE text log per run under raw_log_dir:
  infer_h{n_head}_kv{n_kv_head}_T{prompt_len}_B{batch_size}_KV{true|false}_P{phase}{_rmsnorm_if_any}{_rope|_norope}.log

Defaults:
- phase = "both"
- For profiling (Nsight), run phase="decode" or "prefill", warmup_iters=0, bench_iters=1

New knobs:
- allow_unsafe_benchmark:
    If use_rope=False and prompt_len+max_new_tokens exceeds block_size, by default we error.
    With allow_unsafe_benchmark=True we skip that safety check (unsafe for correctness, fine for perf/memory microbench).
- ignore_checkpoint:
    If True, we DO NOT load a checkpoint even if ckpt_name is provided.
    This enables pure performance/memory benchmarking with random weights.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from nanogpt.model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Optional NVTX for Nsight filtering
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
# Helpers for robust CLI/config parsing

def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "on"):
        return True
    if s in ("0", "false", "no", "n", "f", "off", ""):
        return False
    raise ValueError(f"Cannot parse bool from: {v!r}")


def _norm_suffix(nt: str) -> str:
    nt = str(nt).lower().strip()
    if nt == "layernorm":
        return ""
    if nt == "rmsnorm":
        return "_rmsnorm"
    raise ValueError(f"norm_type must be layernorm|rmsnorm, got: {nt!r}")


def _rope_suffix(use_rope: Any) -> str:
    return "_rope" if _parse_bool(use_rope) else "_norope"


# -----------------------------------------------------------------------------
# Defaults (override via CLI / configurator.py)

# benchmark params
batch_size = 4
prompt_len = 1024
max_new_tokens = 256
warmup_iters = 2
bench_iters = 10
phase = "both"  # "both" | "prefill" | "decode"
kv_cache = False

# model defaults (only used if no checkpoint is loaded or ignore_checkpoint=True)
n_layer = 12
n_head = 12
n_kv_head = 12
n_embd = 768
vocab_size = 50304
bias = True
use_rope = False

# norm knobs
norm_type = "layernorm"  # "layernorm" | "rmsnorm"
norm_eps = 1e-5

# runtime
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

# prefix
prefix_source = "random"  # "random" or "model"
prefix_seed_len = 1
prefix_cache_dir = "bench_prefix"
cache_prefix = True

# output
out_dir = "bench_out"
raw_log_dir = "raw_logs"

# checkpoint controls
ckpt_dir = "out-attn"
ckpt_name = ""  # if set and ignore_checkpoint=False, we load ckpt_dir/ckpt_name.pt

# NEW: benchmark safety & behavior knobs
allow_unsafe_benchmark = False
ignore_checkpoint = False

# nanoGPT-style overrides
exec(open("bench/configurator.py").read(), globals())
# -----------------------------------------------------------------------------


# Normalize/validate after configurator (fix common string-bool pitfalls)
kv_cache = _parse_bool(kv_cache)
use_rope = _parse_bool(use_rope)
cache_prefix = _parse_bool(cache_prefix)
compile = _parse_bool(compile)
allow_unsafe_benchmark = _parse_bool(allow_unsafe_benchmark)
ignore_checkpoint = _parse_bool(ignore_checkpoint)

phase = str(phase).lower().strip()
if phase not in ("both", "prefill", "decode"):
    raise ValueError(f"phase must be one of both|prefill|decode, got: {phase}")

dtype = str(dtype).lower().strip()
if dtype not in ("float16", "bfloat16", "float32"):
    raise ValueError(f"dtype must be one of float16|bfloat16|float32, got: {dtype}")

norm_type = str(norm_type).lower().strip()
norm_eps = float(norm_eps)
norm_suf = _norm_suffix(norm_type)

n_kv_head = int(n_kv_head)
n_head = int(n_head)
if n_head % n_kv_head != 0:
    raise ValueError(f"Invalid heads: n_head={n_head} must be divisible by n_kv_head={n_kv_head}")

device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(int(seed))
if device_type == "cuda":
    torch.cuda.manual_seed(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# Checkpoint-aware model construction

def _ckpt_path() -> Path:
    if not ckpt_name:
        return Path("")
    return Path(ckpt_dir) / f"{ckpt_name}.pt"


def _load_checkpoint() -> dict | None:
    # benchmark-only mode: do not load checkpoint even if provided
    if ignore_checkpoint:
        return None

    p = _ckpt_path()
    if not ckpt_name:
        return None
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return torch.load(str(p), map_location="cpu")


def _build_model_from_checkpoint_or_defaults() -> tuple[GPT, dict, dict | None]:
    """
    Returns: (model, model_args_used, checkpoint_dict_or_none)

    If ckpt_name provided and ignore_checkpoint=False, uses checkpoint["model_args"] to build model,
    then loads state_dict strictly.

    If no checkpoint (or ignore_checkpoint=True), builds a model from current globals with random weights.
    """
    run_block = int(prompt_len) + int(max_new_tokens)

    ckpt = _load_checkpoint()
    if ckpt is None:
        # no checkpoint: use current globals, and ensure block_size is big enough
        model_args = dict(
            block_size=run_block,
            vocab_size=int(vocab_size),
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_kv_head=int(n_kv_head),
            n_embd=int(n_embd),
            dropout=0.0,
            bias=bool(bias),
            use_rope=bool(use_rope),
            norm_type=str(norm_type),
            norm_eps=float(norm_eps),
        )
        conf = GPTConfig(**model_args)
        m = GPT(conf).to(device).eval()
        return m, model_args, None

    if "model_args" not in ckpt:
        raise RuntimeError("Checkpoint missing model_args. Re-train with saving model_args in checkpoint.")

    ckpt_args = dict(ckpt["model_args"])

    # Architecture must match checkpoint weights.
    ckpt_use_rope = _parse_bool(ckpt_args.get("use_rope", False))
    ckpt_block = int(ckpt_args.get("block_size", run_block))

    # Block size policy:
    # - With RoPE, we can safely expand block_size at runtime.
    # - Without RoPE, exceeding trained block_size is "unsafe correctness".
    #   For perf-only, allow skipping this check via allow_unsafe_benchmark.
    if ckpt_use_rope:
        block_size_used = run_block
    else:
        block_size_used = ckpt_block
        if run_block > ckpt_block and not allow_unsafe_benchmark:
            raise ValueError(
                f"Requested T={prompt_len} + new={max_new_tokens} = {run_block} exceeds ckpt block_size={ckpt_block} "
                f"and use_rope=False. Train with larger block_size, enable RoPE, or set --allow_unsafe_benchmark=true."
            )

        # If unsafe benchmark is enabled and run_block > ckpt_block, we still need a model config.
        # We keep block_size at ckpt_block to preserve param shapes (absolute pos emb tables).
        # This is "unsafe correctness" because the run is asking for longer sequences than the table supports.
        # But for pure perf/memory microbench, you *may* still want to proceed.
        # NOTE: If your model uses absolute learned positional embeddings, going past block_size can crash.
        # If that happens, enable RoPE or train with bigger block_size.
        block_size_used = ckpt_block

    model_args = dict(
        block_size=block_size_used,
        vocab_size=int(ckpt_args["vocab_size"]),
        n_layer=int(ckpt_args["n_layer"]),
        n_head=int(ckpt_args["n_head"]),
        n_kv_head=int(ckpt_args.get("n_kv_head", ckpt_args["n_head"])),
        n_embd=int(ckpt_args["n_embd"]),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        bias=bool(ckpt_args.get("bias", True)),
        use_rope=bool(ckpt_use_rope),
        norm_type=str(ckpt_args.get("norm_type", "layernorm")).lower().strip(),
        norm_eps=float(ckpt_args.get("norm_eps", 1e-5)),
    )

    # Enforce sweep knobs match ckpt (fail-fast)
    if int(n_head) != int(model_args["n_head"]):
        raise ValueError(f"n_head mismatch: run={n_head} ckpt={model_args['n_head']}")
    if int(n_kv_head) != int(model_args["n_kv_head"]):
        raise ValueError(f"n_kv_head mismatch: run={n_kv_head} ckpt={model_args['n_kv_head']}")
    if _parse_bool(use_rope) != bool(model_args["use_rope"]):
        raise ValueError(f"use_rope mismatch: run={use_rope} ckpt={model_args['use_rope']}")
    if str(norm_type).lower().strip() != str(model_args["norm_type"]).lower().strip():
        raise ValueError(f"norm_type mismatch: run={norm_type} ckpt={model_args['norm_type']}")

    conf = GPTConfig(**model_args)
    m = GPT(conf).to(device).eval()

    state_dict = ckpt["model"]
    m.load_state_dict(state_dict, strict=True)
    return m, model_args, ckpt


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
    t0 = time.time()
    fn()
    return (time.time() - t0) * 1000.0


def _supports_kv_api(m) -> bool:
    import inspect
    sig = inspect.signature(m.forward)
    return ("use_cache" in sig.parameters) and ("past_kv" in sig.parameters)


# -----------------------------------------------------------------------------
# Prefix builders (optional caching)

def _prefix_cache_path(model_args_used: dict) -> str:
    os.makedirs(prefix_cache_dir, exist_ok=True)
    key = {
        "prefix_source": prefix_source,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "prompt_len": int(prompt_len),
        "prefix_seed_len": int(prefix_seed_len),
        # tie prefix cache to architecture
        "model_args": model_args_used,
        "dtype": dtype,
    }
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return os.path.join(prefix_cache_dir, f"prefix_{h}.pt")


@torch.no_grad()
def build_prefix_random(vocab: int) -> torch.Tensor:
    return torch.randint(vocab, (int(batch_size), int(prompt_len)), device=device)


@torch.no_grad()
def build_prefix_model_generated(model: GPT, vocab: int) -> torch.Tensor:
    if not _supports_kv_api(model):
        raise RuntimeError("prefix_source=model requires KV cache support in model.forward (use_cache + past_kv).")

    seed_tok = torch.randint(vocab, (int(batch_size), int(prefix_seed_len)), device=device)

    with ctx:
        nvtx_push("prefix_build_model_prefill_seed", device_type)
        _logits, _loss, past = model(seed_tok, None, use_cache=True, past_kv=None)  # type: ignore
        nvtx_pop(device_type)

        last = seed_tok[:, -1:].contiguous()
        tokens = [seed_tok]

        for _ in range(int(prompt_len) - int(prefix_seed_len)):
            nvtx_push("prefix_build_model_decode_step", device_type)
            logits, _loss, past = model(last, None, use_cache=True, past_kv=past)  # type: ignore
            nvtx_pop(device_type)

            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(nxt)
            last = nxt

    return torch.cat(tokens, dim=1)


@torch.no_grad()
def get_or_build_prefix(model: GPT, model_args_used: dict) -> torch.Tensor:
    vocab = int(model_args_used["vocab_size"])

    if not bool(cache_prefix):
        return build_prefix_model_generated(model, vocab) if prefix_source == "model" else build_prefix_random(vocab)

    path = _prefix_cache_path(model_args_used)
    if os.path.exists(path):
        prefix = torch.load(path, map_location=device)
        if tuple(prefix.shape) != (int(batch_size), int(prompt_len)):
            raise RuntimeError(
                f"Cached prefix shape mismatch: got {tuple(prefix.shape)} expected {(int(batch_size), int(prompt_len))}"
            )
        return prefix

    prefix = build_prefix_model_generated(model, vocab) if prefix_source == "model" else build_prefix_random(vocab)
    torch.save(prefix, path)
    print(f"[INFO] Saved prefix cache: {os.path.abspath(path)}")
    return prefix


# -----------------------------------------------------------------------------
# Decode paths

@torch.no_grad()
def decode_only_no_kv(model: GPT, prefix: torch.Tensor):
    # WARNING: This is intentionally "realistic" for no-KV: each token recomputes full context.
    # For large prompt_len and large max_new_tokens, it can be very slow.
    idx = prefix
    for _ in range(int(max_new_tokens)):
        with ctx:
            logits, _loss = model(idx, None)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)


@torch.no_grad()
def decode_only_kv(model: GPT, prefix: torch.Tensor):
    if not _supports_kv_api(model):
        raise RuntimeError("kv_cache=True requested but model.forward is not KV-capable.")

    nvtx_push("decode_kv_prefill", device_type)
    _logits, _loss, past = model(prefix, None, use_cache=True, past_kv=None)  # type: ignore
    nvtx_pop(device_type)

    last = prefix[:, -1:].contiguous()
    for _ in range(int(max_new_tokens)):
        nvtx_push("decode_kv_step", device_type)
        logits, _loss, past = model(last, None, use_cache=True, past_kv=past)  # type: ignore
        nvtx_pop(device_type)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        last = nxt


# -----------------------------------------------------------------------------
# Benchmark core

def bench_once(model: GPT, prefix: torch.Tensor) -> Dict[str, Any]:
    _reset_peak_mem()
    metrics: Dict[str, Any] = {}

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

    if phase in ("both", "decode"):

        def do_decode():
            nvtx_push("decode", device_type)
            if kv_cache:
                decode_only_kv(model, prefix)
            else:
                decode_only_no_kv(model, prefix)
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
# Artifact naming

def _artifact_stem(model_args_used: dict) -> str:
    kv_str = "true" if kv_cache else "false"
    rope_suf = _rope_suffix(model_args_used.get("use_rope", use_rope))
    return (
        f"infer_h{n_head}_kv{n_kv_head}_T{prompt_len}_B{batch_size}_KV{kv_str}_P{phase}"
        f"{norm_suf}{rope_suf}"
    )


# -----------------------------------------------------------------------------
# Main

def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_log_dir, exist_ok=True)

    model, model_args_used, ckpt = _build_model_from_checkpoint_or_defaults()

    if compile:
        model = torch.compile(model)

    prefix = get_or_build_prefix(model, model_args_used)

    # Warmups
    for _ in range(int(warmup_iters)):
        bench_once(model, prefix)

    # Timed runs
    rows = [bench_once(model, prefix) for _ in range(int(bench_iters))]

    def avg(k: str) -> float:
        vals = [r[k] for r in rows if k in r]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    peak_alloc = max(int(r["peak_alloc_bytes"]) for r in rows)
    peak_reserved = max(int(r["peak_reserved_bytes"]) for r in rows)

    metrics_out: Dict[str, Any] = {
        "peak_alloc_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
    }

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
            "batch_size": int(batch_size),
            "prompt_len": int(prompt_len),
            "max_new_tokens": int(max_new_tokens),
            "kv_cache": bool(kv_cache),
            "phase": phase,
            "prefix_source": prefix_source,
            "prefix_cache_dir": prefix_cache_dir,
            "cache_prefix": bool(cache_prefix),
            "prefix_seed_len": int(prefix_seed_len),
            "compile": bool(compile),
            "allow_unsafe_benchmark": bool(allow_unsafe_benchmark),
            "ignore_checkpoint": bool(ignore_checkpoint),
            # resolved-from-ckpt architecture (authoritative)
            **model_args_used,
            # include ckpt identity if present
            "ckpt_dir": ckpt_dir,
            "ckpt_name": ckpt_name,
        },
        "metrics": metrics_out,
    }

    stem = _artifact_stem(model_args_used)

    out_path = os.path.join(out_dir, f"{stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log_path = os.path.join(raw_log_dir, f"{stem}.log")
    lines = [
        f"Wrote: {out_path}",
        f"Log:   {log_path}",
        "",
        "params:",
        json.dumps(result["params"], indent=2),
        "",
        "metrics:",
        json.dumps(result["metrics"], indent=2),
        "",
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote: {out_path}")
    print(f"Wrote: {log_path}")
    print("params:", json.dumps(result["params"], indent=2))
    print("metrics:", json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
