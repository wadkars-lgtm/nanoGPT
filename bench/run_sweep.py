"""
run_sweep.py

Runs a grid sweep over prompt_len and batch_size by calling batch_infer.py using sys.executable.
Aggregates infer_*.json into CSV and plots.

Key guarantee:
- child runs with SAME python as parent (sys.executable)

This version adds DEBUG prints for:
- exact command executed
- resolved sys.executable
- cwd + batch_infer path
- stdout tail on success (optional, controlled by --debug_tail_lines)
- full stdout on failure (still tailed to avoid spam)
- discovery of JSON artifacts (count + newest file)
- row counts after filtering by kv mode

Assumptions:
- This file lives at: {REPO_ROOT}/bench/run_sweep.py
- batch_infer.py lives at: {REPO_ROOT}/bench/batch_infer.py
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def repo_root() -> str:
    # bench/run_sweep.py -> parents[1] is repo root
    return str(Path(__file__).resolve().parents[1])


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_child(cmd: List[str], cwd: str) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.returncode, p.stdout


def load_jsons(out_dir_abs: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(out_dir_abs, "infer_*.json")))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            # ignore unreadable/partial files
            pass
    return rows


def flatten(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("meta", {})
    params = obj.get("params", {})
    metrics = obj.get("metrics", {})
    out: Dict[str, Any] = {}
    out.update(params)
    out.update(metrics)
    out["gpu_name"] = meta.get("gpu_name")
    out["dtype"] = meta.get("dtype")
    out["torch_version"] = meta.get("torch_version")
    out["cuda_version"] = meta.get("cuda")
    out["time_unix"] = meta.get("time_unix")
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] Wrote CSV: {path} (rows={len(rows)}, cols={len(keys)})")


def plot_lines(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    series_key: str,
    out_path: str,
    title: str,
    y_label: str,
) -> None:
    if not rows:
        print(f"[WARN] No rows to plot: {out_path}")
        return

    series_vals = sorted({int(r[series_key]) for r in rows if series_key in r and r[series_key] is not None})
    x_vals = sorted({int(r[x_key]) for r in rows if x_key in r and r[x_key] is not None})

    idx: Dict[Tuple[int, int], float] = {}
    for r in rows:
        try:
            s = int(r[series_key])
            x = int(r[x_key])
            y = float(r[y_key])
            idx[(s, x)] = y
        except Exception:
            pass

    plt.figure()
    for s in series_vals:
        xs, ys = [], []
        for x in x_vals:
            if (s, x) in idx:
                xs.append(x)
                ys.append(idx[(s, x)])
        if xs:
            plt.plot(xs, ys, marker="o", label=f"{series_key}={s}")

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote plot: {out_path}")


def _tail(s: str, n: int) -> str:
    if n <= 0:
        return ""
    lines = s.splitlines()
    return "\n".join(lines[-n:])


def _fmt_cmd(cmd: List[str]) -> str:
    # readable, copy/paste-able
    parts = []
    for c in cmd:
        if " " in c and not (c.startswith('"') and c.endswith('"')):
            parts.append(f'"{c}"')
        else:
            parts.append(c)
    return " ".join(parts)


def _list_json_paths(out_dir_abs: str) -> List[str]:
    return sorted(glob(os.path.join(out_dir_abs, "infer_*.json")))


def _newest_path(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def _parse_bool_list(csv_str: str) -> List[bool]:
    out: List[bool] = []
    for tok in csv_str.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in ("true", "1", "t", "yes", "y"):
            out.append(True)
        elif t in ("false", "0", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Bad kv mode token: {tok}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # Recommend results/… by default; caller can override.
    ap.add_argument("--out_dir", default="results/sweeps")
    ap.add_argument("--prefix_cache_dir", default="results/prefix_cache")
    ap.add_argument("--prefix_source", default="model")  # "model" or "random"

    ap.add_argument("--prompt_lens", default="128,512,1024,2048,4096")
    ap.add_argument("--batch_sizes", default="1,2,4,8,12,16")
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--warmup_iters", type=int, default=2)
    ap.add_argument("--bench_iters", type=int, default=10)

    ap.add_argument("--dtype", default=None)
    ap.add_argument("--compile", default=None)

    # run one or both modes
    ap.add_argument("--kv_modes", default="False,True")  # "False" or "False,True" etc.

    # debug controls
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug prints per sweep point.")
    ap.add_argument(
        "--debug_tail_lines",
        type=int,
        default=25,
        help="Tail N lines of child stdout on success when --debug.",
    )
    ap.add_argument("--stop_on_fail", action="store_true", help="Stop immediately when a child run fails.")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint to evaluate (forwarded to batch_infer.py)")

    args = ap.parse_args()

    root = repo_root()
    bench_dir = os.path.join(root, "bench")

    # Resolve out_dir to an absolute path under repo root (even if caller passes relative)
    out_dir_abs = os.path.join(root, args.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    # prefix cache dir (keep it under repo root by default)
    prefix_cache_dir_abs = os.path.join(root, args.prefix_cache_dir)
    os.makedirs(prefix_cache_dir_abs, exist_ok=True)

    batch_infer = os.path.join(bench_dir, "batch_infer.py")
    if not os.path.exists(batch_infer):
        raise FileNotFoundError(f"Missing {batch_infer}")

    print("[INFO] run_sweep interpreter:", sys.executable)
    print("[INFO] run_sweep file:", os.path.abspath(__file__))
    print("[INFO] cwd(root):", root)
    print("[INFO] bench_dir:", bench_dir)
    print("[INFO] batch_infer path:", batch_infer)
    print("[INFO] out_dir_abs:", out_dir_abs)
    print("[INFO] prefix_cache_dir_abs:", prefix_cache_dir_abs)

    prompt_lens = parse_int_list(args.prompt_lens)
    batch_sizes = parse_int_list(args.batch_sizes)
    kv_modes = _parse_bool_list(args.kv_modes)

    py = sys.executable

    # Pre-sweep JSON snapshot (so we can tell if new artifacts appear)
    baseline_jsons = set(_list_json_paths(out_dir_abs))
    if args.debug:
        print(f"[DEBUG] existing infer_*.json count before sweep: {len(baseline_jsons)}")

    for kv in kv_modes:
        kv_str = "True" if kv else "False"
        print(f"\n=== Sweep kv_cache={kv_str} ===")

        attempted = 0
        succeeded = 0
        failed = 0

        for T in prompt_lens:
            for B in batch_sizes:
                attempted += 1
                t0 = time.time()

                # IMPORTANT: pass the SAME out_dir/prefix_cache_dir string to the child,
                # and run child with cwd=root so it resolves under repo root.
                cmd = [
                    py,
                    batch_infer,
                    f"--out_dir={args.out_dir}",
                    f"--prompt_len={T}",
                    f"--batch_size={B}",
                    f"--max_new_tokens={args.max_new_tokens}",
                    f"--kv_cache={kv_str}",
                    f"--prefix_source={args.prefix_source}",
                    f"--prefix_cache_dir={args.prefix_cache_dir}",
                    f"--warmup_iters={args.warmup_iters}",
                    f"--bench_iters={args.bench_iters}",
                ]
                if args.dtype is not None:
                    cmd.append(f"--dtype={args.dtype}")
                if args.compile is not None:
                    cmd.append(f"--compile={args.compile}")

                if args.debug:
                    print("\n[DEBUG] ----------------------------------------")
                    print(f"[DEBUG] point: kv={kv_str} T={T} B={B}")
                    print(f"[DEBUG] python: {py}")
                    print(f"[DEBUG] cwd: {root}")
                    print(f"[DEBUG] cmd: {_fmt_cmd(cmd)}")
                    print("[DEBUG] ----------------------------------------")

                before = set(_list_json_paths(out_dir_abs))
                rc, out = run_child(cmd, cwd=root)
                after = set(_list_json_paths(out_dir_abs))
                new_files = sorted(list(after - before))
                dt = time.time() - t0

                if rc != 0:
                    failed += 1
                    tail = _tail(out, 60)
                    print(f"[WARN] FAILED kv={kv_str} T={T} B={B} rc={rc} elapsed={dt:.2f}s")
                    if new_files:
                        print(f"[WARN] New JSONs despite failure ({len(new_files)}):")
                        for nf in new_files[:5]:
                            print("  -", nf)
                    print("[WARN] Child stdout tail:")
                    print(tail)
                    print()
                    if args.stop_on_fail:
                        raise RuntimeError(f"Stopping on failure at kv={kv_str} T={T} B={B}")
                    continue

                succeeded += 1
                if args.debug:
                    print(f"[DEBUG] OK kv={kv_str} T={T} B={B} elapsed={dt:.2f}s rc={rc}")
                    if new_files:
                        newest = _newest_path(new_files)
                        print(f"[DEBUG] New infer JSONs created: {len(new_files)} (newest={newest})")
                    else:
                        print("[DEBUG] No new infer_*.json detected for this run (check out_dir / run_id collisions).")

                    tail = _tail(out, args.debug_tail_lines)
                    if tail:
                        print("[DEBUG] Child stdout tail:")
                        print(tail)

        print(f"[INFO] mode kv={kv_str}: attempted={attempted} ok={succeeded} fail={failed}")

        # Aggregate per-mode CSV/plots
        objs = load_jsons(out_dir_abs)
        flat = [flatten(o) for o in objs]
        flat = [r for r in flat if bool(r.get("kv_cache")) == kv]

        print(f"[INFO] Aggregation kv={kv_str}: json_objs={len(objs)} rows_after_filter={len(flat)}")

        csv_path = os.path.join(out_dir_abs, f"sweep_kv_{kv_str}.csv")
        write_csv(csv_path, flat)

        # decode ms/tok plot
        plot_lines(
            flat,
            x_key="prompt_len",
            y_key="decode_ms_per_tok",
            series_key="batch_size",
            out_path=os.path.join(out_dir_abs, f"plot_decode_ms_per_tok_kv_{kv_str}.png"),
            title=f"Decode ms/token vs prompt_len (kv_cache={kv_str})",
            y_label="decode_ms_per_tok",
        )

        # peak alloc GiB plot
        flat2: List[Dict[str, Any]] = []
        for r in flat:
            rr = dict(r)
            try:
                rr["peak_alloc_gib"] = float(rr["peak_alloc_bytes"]) / (1024.0**3)
            except Exception:
                rr["peak_alloc_gib"] = None
            flat2.append(rr)

        plot_lines(
            flat2,
            x_key="prompt_len",
            y_key="peak_alloc_gib",
            series_key="batch_size",
            out_path=os.path.join(out_dir_abs, f"plot_peak_alloc_gib_kv_{kv_str}.png"),
            title=f"Peak alloc (GiB) vs prompt_len (kv_cache={kv_str})",
            y_label="peak_alloc_gib",
        )

    # Final sweep delta (how many new artifacts were created in total)
    final_jsons = set(_list_json_paths(out_dir_abs))
    created = sorted(list(final_jsons - baseline_jsons))
    print(f"\n[INFO] Total infer_*.json before: {len(baseline_jsons)} after: {len(final_jsons)} created: {len(created)}")
    if args.debug and created:
        newest = _newest_path(created)
        print(f"[DEBUG] newest created json: {newest}")


if __name__ == "__main__":
    main()
