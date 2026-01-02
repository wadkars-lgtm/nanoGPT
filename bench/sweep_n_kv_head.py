import argparse
import csv
import time
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="batch_infer.py")  # change if needed
    ap.add_argument("--out_csv", default="results/gqa/sweep.csv")
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

    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_csv.parent / "rawlogs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_kv_list = [int(x) for x in args.n_kv_heads.split(",")]
    T_list = [int(x) for x in args.prompt_lens.split(",")]
    B_list = [int(x) for x in args.batch_sizes.split(",")]

    header = [
        "ts", "phase", "kv_cache", "n_head", "n_kv_head", "prompt_len", "batch_size",
        "rc", "log_path"
    ]

    py = sys.executable  # ✅ critical: ensures venv interpreter is used for subprocesses

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for n_kv in n_kv_list:
            if args.n_head % n_kv != 0:
                print(f"SKIP n_kv={n_kv} (n_head % n_kv != 0)")
                continue

            for T in T_list:
                for B in B_list:
                    name = f"{args.phase}_h{args.n_head}_kv{n_kv}_T{T}_B{B}"
                    log_path = raw_dir / f"{name}.log"

                    cmd = [
                        py, args.script,              # ✅ was "python"
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
                    ]

                    print("RUN:", " ".join(cmd))
                    rc, out = run(cmd)
                    log_path.write_text(out, encoding="utf-8")

                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    w.writerow([ts, args.phase, args.kv_cache, args.n_head, n_kv, T, B, rc, str(log_path)])

                    if rc != 0:
                        print(out)
                        print(f"FAILED: {log_path}")


if __name__ == "__main__":
    main()
