"""
batch_infer.py

Single-run inference benchmark that:
1) Builds a prefix of length prompt_len (random or model-generated).
2) Measures prefill on that prefix.
3) Measures decode-only for max_new_tokens starting from that prefix.

Writes ONE JSON artifact per run under out_dir with semantic naming:
  infer_T{prompt_len}_B{batch_size}_KV{true|false}.json
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
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

kv_cache = False

prefix_source = "random"   # "random" or "model"
prefix_seed_len = 1

warmup_iters = 2
bench_iters = 10

out_dir = "bench_out"
prefix_cache_dir = "bench_prefix"
prefix_source = "model"

# nanoGPT-style overrides
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(seed)
if device_type == "cuda":
    torch.cuda.manual_seed(seed)
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
    return "use_cache" in sig.parameters and "past_kv" in sig.parameters

# -----------------------------------------------------------------------------
# Prefix builders
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
def get_or_build_prefix() -> torch.Tensor:
    path = _prefix_cache_path()
    if os.path.exists(path):
        prefix = torch.load(path, map_location=device)
        if tuple(prefix.shape) != (batch_size, prompt_len):
            raise RuntimeError(f"Cached prefix shape mismatch: got {tuple(prefix.shape)} expected {(batch_size, prompt_len)}")
        return prefix

    if prefix_source == "model":
        prefix = build_prefix_model_generated()
    else:
        prefix = build_prefix_random()

    torch.save(prefix, path)
    print(f"[INFO] Saved prefix cache: {os.path.abspath(path)}")
    return prefix

@torch.no_grad()
def build_prefix_random() -> torch.Tensor:
    return torch.randint(vocab_size, (batch_size, prompt_len), device=device)

@torch.no_grad()
def build_prefix_model_generated() -> torch.Tensor:
    if not _supports_kv_api(model):
        raise RuntimeError("prefix_source=model requires KV cache support")

    seed_tok = torch.randint(vocab_size, (batch_size, prefix_seed_len), device=device)

    with ctx:
        _, _, past = model(seed_tok, None, use_cache=True, past_kv=None)
        last = seed_tok[:, -1:].contiguous()
        tokens = [seed_tok]

        for _ in range(prompt_len - prefix_seed_len):
            logits, _, past = model(last, None, use_cache=True, past_kv=past)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(nxt)
            last = nxt

    return torch.cat(tokens, dim=1)

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
    _, _, past = model(prefix, None, use_cache=True, past_kv=None)
    last = prefix[:, -1:].contiguous()
    for _ in range(max_new_tokens):
        logits, _, past = model(last, None, use_cache=True, past_kv=past)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        last = nxt

# -----------------------------------------------------------------------------
# Benchmark

def bench_once(prefix: torch.Tensor) -> Dict[str, Any]:
    _reset_peak_mem()

    def do_prefill():
        with ctx:
            if kv_cache:
                model(prefix, None, use_cache=True, past_kv=None)
            else:
                model(prefix, None)

    prefill_ms = _cuda_time_ms(do_prefill)
    prefill_ms_per_tok = prefill_ms / float(prompt_len)

    def do_decode():
        if kv_cache:
            decode_only_kv(prefix)
        else:
            decode_only_no_kv(prefix)

    decode_ms = _cuda_time_ms(do_decode)
    decode_ms_per_tok = decode_ms / float(max_new_tokens)
    decode_tok_s = (1000.0 * max_new_tokens) / decode_ms if decode_ms > 0 else 0.0

    peak_alloc, peak_reserved = _get_peak_mem()

    return {
        "prefill_ms": prefill_ms,
        "prefill_ms_per_tok": prefill_ms_per_tok,
        "decode_total_ms": decode_ms,
        "decode_ms_per_tok": decode_ms_per_tok,
        "decode_tok_s": decode_tok_s,
        "peak_alloc_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
    }

# -----------------------------------------------------------------------------
# Main

def main():
    os.makedirs(out_dir, exist_ok=True)

    prefix = get_or_build_prefix()

    # Warmups
    for _ in range(warmup_iters):
        bench_once(prefix)

    # Timed runs
    rows = [bench_once(prefix) for _ in range(bench_iters)]

    def avg(k: str) -> float:
        return sum(r[k] for r in rows) / len(rows)

    peak_alloc = max(r["peak_alloc_bytes"] for r in rows)
    peak_reserved = max(r["peak_reserved_bytes"] for r in rows)

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
            "prefix_source": prefix_source,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "vocab_size": vocab_size,
        },
        "metrics": {
            "prefill_ms": avg("prefill_ms"),
            "prefill_ms_per_tok": avg("prefill_ms_per_tok"),
            "decode_total_ms": avg("decode_total_ms"),
            "decode_ms_per_tok": avg("decode_ms_per_tok"),
            "decode_tok_s": avg("decode_tok_s"),
            "peak_alloc_bytes": int(peak_alloc),
            "peak_reserved_bytes": int(peak_reserved),
        },
    }

    kv_str = "true" if kv_cache else "false"
    fname = f"infer_T{prompt_len}_B{batch_size}_KV{kv_str}.json"
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        print(f"[WARN] Overwriting existing file: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
