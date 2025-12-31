import argparse
import csv
import subprocess
import re
from pathlib import Path

LOSS_RE = re.compile(r"val_loss=([0-9.]+)")
PPL_RE = re.compile(r"val_ppl=([0-9.]+)")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout

def parse(out):
    loss = float(LOSS_RE.search(out).group(1))
    ppl = float(PPL_RE.search(out).group(1))
    return loss, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/gqa/eval_compare.csv")
    args = ap.parse_args()

    rows = [
        ("MHA", 12, "out-attn/mha_h12_kv12_best.pt"),
        ("GQA", 3,  "out-attn/gqa_h12_kv3_best.pt"),
        ("MQA", 1,  "out-attn/mqa_h12_kv1_best.pt"),
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_kv_head", "val_loss", "val_ppl"])

        for mode, kv, ckpt in rows:
            cmd = ["python", "eval_loss.py", "--ckpt", ckpt, "--dataset", "shakespeare_char"]
            print("RUN:", " ".join(cmd))
            out = run(cmd)
            loss, ppl = parse(out)
            w.writerow([mode, kv, loss, ppl])

    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
