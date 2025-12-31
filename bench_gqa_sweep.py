import os
import csv
import time
import argparse
import subprocess
from pathlib import Path


BATCH_INFER = "batch_infer.py"  # <-- change if your file lives elsewhere


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/gqa/sweep.csv")
    ap.add_argument("--out_dir", default="out-attn")
    ap.add_argument("--config", default="config/train_shakespeare_char.py")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_kv_heads", default="12,6,4,3,2,1")

    ap.add_argument("--prompt_lens", default="128,256,512,1024,2048")
    ap.add_argument("--batch_sizes", default="1,4,8,16,32")
    ap.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    ap.add_argument("--kv_cache", choices=["true", "false"], default="true")

    ap.add_argument("--max_new_tokens", type=int, default=1)
    ap.add_argument("--bench_iters", type=int, default=50)
    ap.add_argument("--warmup_iters", type=int, default=5)

    args = ap.parse_args()

    n_kv_list = [int(x) for x in args.n_kv_heads.split(",")]
    T_list = [int(x) for x in args.prompt_lens.split(",")]
    B_list = [int(x) for x in args.batch_sizes.split(",")]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # You should modify batch_infer.py to print a single parsable line like:
    # RESULT decode_total_ms=... ms_per_tok=... tok_per_s=... peak_mem_mb=...
    # If it doesn't, this sweep still stores raw logs so you can parse later.
    rows = []
    header = ["timestamp", "phase", "kv_cache", "n_head", "n_kv_head", "prompt_len", "batch_size", "rc", "notes"]
    rawlog_dir = out_csv.parent / "rawlogs"
    rawlog_dir.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for n_kv in n_kv_list:
            if args.n_head % n_kv != 0:
                print(f"SKIP n_kv={n_kv} (not divisible)")
                continue

            for T in T_list:
                for B in B_list:
                    ckpt_name = f"{args.phase}_h{args.n_head}_kv{n_kv}_T{T}_B{B}"

                    cmd = [
                        "python", BATCH_INFER,
                        f"--device={args.device}",
                        f"--dtype={args.dtype}",
                        f"--phase={args.phase}",
                        f"--kv_cache={args.kv_cache}",
                        f"--prompt_len={T}",
                        f"--batch_size={B}",
                        f"--max_new_tokens={args.max_new_tokens}",
                        f"--bench_iters={args.bench_iters}",
                        f"--warmup_iters={args.warmup_iters}",
                        f"--n_head={args.n_head}",
                        f"--n_kv_head={n_kv}",
                    ]

                    print("RUN:", " ".join(cmd))
                    rc, out = run_cmd(cmd)

                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_path = rawlog_dir / f"{ckpt_name}.log"
                    log_path.write_text(out, encoding="utf-8")

                    notes = f"log={log_path.as_posix()}"
                    w.writerow([ts, args.phase, args.kv_cache, args.n_head, n_kv, T, B, rc, notes])

                    if rc != 0:
                        print(out)
                        print(f"FAILED rc={rc}, see {log_path}")


if __name__ == "__main__":
    main()
