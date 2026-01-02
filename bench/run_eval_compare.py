"""
run_eval_compare.py

Automate "Option A":
- Run eval_loss.py for multiple checkpoints
- Parse val_loss + val_ppl from stdout
- Write results/gqa/eval_compare.csv
- Save per-run raw logs for auditability

PowerShell example:
python .\run_eval_compare.py `
  --dataset shakespeare_char `
  --out_dir results\gqa `
  --labels MHA,GQA,MQA `
  --n_kv_heads 12,3,1 `
  --ckpts out-attn/mha_h12_kv12_best.pt,out-attn/gqa_h12_kv3_best.pt,out-attn/mqa_h12_kv1_best.pt
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


VAL_LOSS_RE = re.compile(r"\bval_loss\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
VAL_PPL_RE = re.compile(r"\bval_ppl\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class EvalRow:
    label: str
    n_kv_head: int
    ckpt: str
    val_loss: float
    val_ppl: float
    log_path: str


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def _extract_metrics(stdout: str) -> tuple[float, float]:
    m1 = VAL_LOSS_RE.search(stdout)
    m2 = VAL_PPL_RE.search(stdout)
    if not (m1 and m2):
        raise RuntimeError(
            "Could not parse val_loss/val_ppl from eval_loss.py output.\n"
            "Expected lines like:\n"
            "  val_loss=1.234\n"
            "  val_ppl=4.567\n"
            "---- output ----\n"
            f"{stdout}\n"
            "----------------"
        )
    return float(m1.group(1)), float(m2.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shakespeare_char")
    ap.add_argument("--out_dir", default="results/gqa")
    ap.add_argument("--csv_name", default="eval_compare.csv")

    ap.add_argument("--labels", required=True, help="Comma list, e.g. MHA,GQA,MQA")
    ap.add_argument("--n_kv_heads", required=True, help="Comma list, e.g. 12,3,1")
    ap.add_argument("--ckpts", required=True, help="Comma list of checkpoint paths")

    ap.add_argument("--eval_script", default="bench/eval_loss.py", help="Path to eval_loss.py")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    n_kv_heads = [int(x.strip()) for x in args.n_kv_heads.split(",") if x.strip()]
    ckpts = [x.strip() for x in args.ckpts.split(",") if x.strip()]

    if not (len(labels) == len(n_kv_heads) == len(ckpts)):
        raise ValueError(
            f"Lengths must match: labels({len(labels)}), n_kv_heads({len(n_kv_heads)}), ckpts({len(ckpts)})"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = out_dir / "eval_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable  # ensures venv interpreter

    rows: list[EvalRow] = []

    for label, kv, ckpt in zip(labels, n_kv_heads, ckpts):
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        cmd = [
            py, args.eval_script,
            "--ckpt", str(ckpt_path),
            "--dataset", str(args.dataset),
        ]
        print("RUN:", " ".join(cmd))
        rc, out = _run(cmd)

        log_path = logs_dir / f"eval_{label.lower()}_kv{kv}.txt"
        log_path.write_text(out, encoding="utf-8")

        if rc != 0:
            raise RuntimeError(f"eval_loss.py failed (rc={rc}). See log: {log_path}")

        val_loss, val_ppl = _extract_metrics(out)

        rows.append(
            EvalRow(
                label=label,
                n_kv_head=kv,
                ckpt=str(ckpt_path),
                val_loss=val_loss,
                val_ppl=val_ppl,
                log_path=str(log_path),
            )
        )

    # Write compare CSV
    csv_path = out_dir / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_kv_head", "val_loss", "val_ppl", "ckpt", "log_path"])
        for r in rows:
            w.writerow([r.label, r.n_kv_head, r.val_loss, r.val_ppl, r.ckpt, r.log_path])

    print(f"Wrote: {csv_path}")
    print(f"Saved logs in: {logs_dir}")


if __name__ == "__main__":
    main()
