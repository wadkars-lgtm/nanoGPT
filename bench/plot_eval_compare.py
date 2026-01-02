"""
plot_eval_compare.py

Reads a CSV produced by run_eval_compare.py and generates:
- val_loss_vs_n_kv_head.png
- val_ppl_vs_n_kv_head.png

PowerShell example:
python .\plot_eval_compare.py `
  --csv results\gqa\eval_compare.csv `
  --out_dir results\gqa\plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="results/gqa/plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # expected columns at minimum:
    # label, n_kv_head, val_loss, val_ppl
    for col in ["label", "n_kv_head", "val_loss", "val_ppl"]:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing required column: {col}")

    df = df.sort_values("n_kv_head", ascending=False).reset_index(drop=True)

    # Plot: val_loss vs n_kv_head
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(df["n_kv_head"], df["val_loss"], marker="o")
    for _, r in df.iterrows():
        ax.annotate(str(r["label"]), (r["n_kv_head"], r["val_loss"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("val_loss vs n_kv_head")
    ax.set_xlabel("n_kv_head")
    ax.set_ylabel("val_loss")
    _save(fig, out_dir / "val_loss_vs_n_kv_head.png")

    # Plot: val_ppl vs n_kv_head
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(df["n_kv_head"], df["val_ppl"], marker="o")
    for _, r in df.iterrows():
        ax.annotate(str(r["label"]), (r["n_kv_head"], r["val_ppl"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("val_ppl vs n_kv_head")
    ax.set_xlabel("n_kv_head")
    ax.set_ylabel("val_ppl")
    _save(fig, out_dir / "val_ppl_vs_n_kv_head.png")

    print(f"Done. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
