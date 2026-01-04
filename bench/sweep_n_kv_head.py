# bench/sweep_n_kv_head.py
#
# Sweep runner for batch_infer.py.
#
# - Can run with checkpoints (default) OR ignore checkpoints for pure benchmarking.
# - Can allow "unsafe" benchmarks that exceed ckpt block_size when use_rope=False.
# - Writes sweep CSV and raw logs.
# - Output filenames and log folders can be auto-derived from norm + rope.
#
# Naming convention (requested):
#   gqa_h{heads}_kv{kv_heads}_{rope|seq}_{layernorm|rmsnorm}
#
# Example log name:
#   decode_gqa_h12_kv3_seq_rmsnorm_T2048_B32.log
#
# Example ckpt tag:
#   gqa_h12_kv3_seq_rmsnorm
#
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def _norm_tag(norm_type: str) -> str:
    nt = str(norm_type).strip().lower()
    if nt not in ("layernorm", "rmsnorm"):
        raise ValueError(f"Unknown norm_type={norm_type!r}. expected layernorm|rmsnorm")
    return nt


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t", "on")


def _rope_tag(use_rope: bool) -> str:
    return "rope" if bool(use_rope) else "seq"


def _rope_file(use_rope: bool) -> str:
    return "rope" if bool(use_rope) else "norope"


def ckpt_tag(n_head: int, n_kv_head: int, use_rope: bool, norm_type: str) -> str:
    """
    Requested naming convention:
      gqa_h{heads}_kv{kv_heads}_{rope|seq}_{layernorm|rmsnorm}
    """
    return f"gqa_h{n_head}_kv{n_kv_head}_{_rope_tag(use_rope)}_{_norm_tag(norm_type)}"


def _resolve_out_csv_path(out_csv_arg: str, norm_type: str, use_rope: bool) -> Path:
    """
    If --out_csv points to a directory (or ends with /), create:
      <dir>/sweep_<norm>_<ropefile>.csv
    Else, treat it as an explicit filename and use as-is.
    """
    p = Path(out_csv_arg)
    # Heuristic: if has suffix .csv, treat as file, otherwise as directory
    if p.suffix.lower() == ".csv":
        return p
    # Directory mode
    p.mkdir(parents=True, exist_ok=True)
    return p / f"sweep_{_norm_tag(norm_type)}_{_rope_file(use_rope)}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="bench/batch_infer.py")  # can pass batch_infer.py too

    # If this is a directory, we'll auto-name based on norm+rope.
    # If it ends with .csv, we use it verbatim.
    ap.add_argument("--out_csv", default="results/gqa")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")

    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_kv_heads", default="12,6,4,3,2,1")

    ap.add_argument("--prompt_lens", default="128,512,1024,2048,4096")
    ap.add_argument("--batch_sizes", default="1,4,8,16,32")

    ap.add_argument("--phase", default="decode", choices=["prefill", "decode"])
    ap.add_argument("--kv_cache", default="true", choices=["true", "false"])
    ap.add_argument("--max_new_tokens", type=int, default=1)
    ap.add_argument("--warmup_iters", type=int, default=5)
    ap.add_argument("--bench_iters", type=int, default=50)

    ap.add_argument("--norm_type", default="layernorm", choices=["layernorm", "rmsnorm"])
    ap.add_argument("--norm_eps", type=float, default=1e-5)
    ap.add_argument("--use_rope", default="false")  # parsed below

    # Checkpoint controls
    ap.add_argument("--ckpt_dir", default="out-attn")
    ap.add_argument("--on_missing_ckpt", default="fail", choices=["fail", "skip"])
    ap.add_argument("--ignore_checkpoint", default="false")  # NEW
    ap.add_argument("--allow_unsafe_benchmark", default="false")  # NEW

    args = ap.parse_args()

    use_rope = _parse_bool(args.use_rope)
    ignore_ckpt = _parse_bool(args.ignore_checkpoint)
    allow_unsafe = _parse_bool(args.allow_unsafe_benchmark)

    out_csv = _resolve_out_csv_path(args.out_csv, args.norm_type, use_rope)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Put logs next to the CSV but split per norm+rope to avoid collisions
    raw_dir = out_csv.parent / f"rawlogs_{_norm_tag(args.norm_type)}_{_rope_file(use_rope)}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_kv_list = [int(x) for x in args.n_kv_heads.split(",") if x.strip()]
    T_list = [int(x) for x in args.prompt_lens.split(",") if x.strip()]
    B_list = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    header = [
        "ts",
        "phase",
        "kv_cache",
        "dtype",
        "device",
        "use_rope",
        "norm_type",
        "n_head",
        "n_kv_head",
        "prompt_len",
        "batch_size",
        "ckpt_tag",
        "ckpt_path",
        "ignore_checkpoint",
        "allow_unsafe_benchmark",
        "rc",
        "log_path",
    ]

    py = sys.executable  # ensures venv interpreter

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for n_kv in n_kv_list:
            if args.n_head % n_kv != 0:
                print(f"SKIP n_kv={n_kv} (n_head % n_kv != 0)")
                continue

            tag = ckpt_tag(args.n_head, n_kv, use_rope=use_rope, norm_type=args.norm_type)
            ckpt_path = Path(args.ckpt_dir) / f"{tag}.pt"

            if not ignore_ckpt:
                if not ckpt_path.exists():
                    msg = f"Checkpoint not found: {ckpt_path}"
                    if args.on_missing_ckpt == "skip":
                        print(msg, "-> skipping all runs for this n_kv")
                        continue
                    raise FileNotFoundError(
                        f"{msg}\n"
                        f"use_rope={use_rope} norm_type={args.norm_type} implies ckpt_tag={tag}.\n"
                        f"Fix --ckpt_dir or train the missing checkpoint."
                    )

            for T in T_list:
                for B in B_list:
                    name = f"{args.phase}_{tag}_T{T}_B{B}"
                    log_path = raw_dir / f"{name}.log"

                    cmd = [
                        py,
                        args.script,
                        f"--device={args.device}",
                        f"--dtype={args.dtype}",
                        f"--phase={args.phase}",
                        f"--kv_cache={args.kv_cache}",
                        f"--prompt_len={T}",
                        f"--batch_size={B}",
                        f"--max_new_tokens={args.max_new_tokens}",
                        f"--warmup_iters={args.warmup_iters}",
                        f"--bench_iters={args.bench_iters}",
                        f"--n_head={args.n_head}",
                        f"--n_kv_head={n_kv}",
                        f"--use_rope={'true' if use_rope else 'false'}",
                        f"--norm_type={args.norm_type}",
                        f"--norm_eps={args.norm_eps}",
                        # NEW pass-throughs to batch_infer.py
                        f"--ignore_checkpoint={'true' if ignore_ckpt else 'false'}",
                        f"--allow_unsafe_benchmark={'true' if allow_unsafe else 'false'}",
                    ]

                    # Only pass ckpt args if we're not ignoring checkpoints
                    if not ignore_ckpt:
                        cmd += [
                            f"--ckpt_dir={args.ckpt_dir}",
                            f"--ckpt_name={tag}",
                        ]

                    print("RUN:", " ".join(cmd))
                    rc, out = run(cmd)
                    log_path.write_text(out, encoding="utf-8")

                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    w.writerow(
                        [
                            ts,
                            args.phase,
                            args.kv_cache,
                            args.dtype,
                            args.device,
                            use_rope,
                            args.norm_type,
                            args.n_head,
                            n_kv,
                            T,
                            B,
                            tag,
                            str(ckpt_path),
                            ignore_ckpt,
                            allow_unsafe,
                            rc,
                            str(log_path),
                        ]
                    )

                    if rc != 0:
                        print(out)
                        print(f"FAILED: {log_path}")

    print(f"Wrote sweep CSV: {out_csv}")
    print(f"Raw logs in:     {raw_dir}")


if __name__ == "__main__":
    main()
