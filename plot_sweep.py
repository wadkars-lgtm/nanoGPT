"""
Plot sweep results produced by sweep_n_kv_head.py.

Inputs:
  - results/gqa/sweep.csv
  - results/gqa/rawlogs/*.log

Outputs:
  - results/gqa/plots/*.png
  - results/gqa/parsed.csv   (optional, handy for debugging)

This script does NOT run any CUDA. It only parses logs + plots.

Update:
- Your batch_infer.py prints a JSON-ish metrics block AND writes a JSON file:
    Wrote: bench_out\\infer_T2048_B16_KVtrue_Pdecode.json
  So we now:
    (1) Prefer parsing the "Wrote: ...json" artifact (most robust)
    (2) Fallback to regex parsing of the log text
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# JSON artifact detection
# -----------------------
WROTE_JSON_RE = re.compile(r"^\s*Wrote:\s*(.+?\.json)\s*$", re.MULTILINE)

# -----------------------
# Regex fallback (for logs that don't write JSON)
# -----------------------
# These patterns are tailored to your log snippet:
# metrics: {
#   "decode_total_ms": 513.47,
#   "decode_ms_per_tok": 513.47,
#   "decode_tok_s": 1.94,
#   "peak_alloc_bytes": 5350210560,
#   ...
# }
METRIC_REGEX = {
    "ms_per_tok": [
        re.compile(r'"decode_ms_per_tok"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),
        # legacy / other print styles
        re.compile(r"(?:ms\s*/\s*tok|ms_per_tok)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    "tok_per_s": [
        re.compile(r'"decode_tok_s"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),
        re.compile(r"(?:tok\s*/\s*s|tok_per_s|tokens\s*/\s*s)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    "decode_total_ms": [
        re.compile(r'"decode_total_ms"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),
        re.compile(r"(?:decode[_\s]*total[_\s]*ms|decode[_\s]*ms)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    "prefill_total_ms": [
        re.compile(r'"prefill_total_ms"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),
        re.compile(r"(?:prefill[_\s]*total[_\s]*ms|prefill[_\s]*ms)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    ],
    # We don't have "peak_mem_mb" in your logs, you have bytes.
    # We'll parse peak_alloc_bytes and convert -> MB.
    "peak_alloc_bytes": [
        re.compile(r'"peak_alloc_bytes"\s*:\s*(\d+)', re.IGNORECASE),
    ],
    "peak_reserved_bytes": [
        re.compile(r'"peak_reserved_bytes"\s*:\s*(\d+)', re.IGNORECASE),
    ],
}


def first_match(text: str, patterns: list[re.Pattern]) -> float | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _to_mb(x_bytes: int | float | None) -> float | None:
    if x_bytes is None:
        return None
    try:
        return float(x_bytes) / (1024.0 * 1024.0)
    except Exception:
        return None


def parse_metrics_from_json_artifact(log_text: str, log_path: Path) -> dict:
    """
    If the log contains:  Wrote: <path-to-json>
    load that JSON and return a normalized metric dict.
    """
    m = WROTE_JSON_RE.search(log_text)
    if not m:
        return {}

    raw = m.group(1).strip().strip('"').strip("'")

    # JSON path can be relative to repo root, or absolute.
    # If it's relative, interpret relative to repo root (cwd),
    # but if user ran from elsewhere, relative to log dir is a decent fallback.
    p = Path(raw)
    if not p.is_absolute():
        # prefer cwd-relative first
        p1 = Path(p)
        if p1.exists():
            p = p1
        else:
            # fallback: relative to the log file directory
            p2 = (log_path.parent / p).resolve()
            if p2.exists():
                p = p2

    if not p.exists():
        return {}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    params = data.get("params", {}) if isinstance(data, dict) else {}

    out: dict[str, float | None] = {}

    # Normalize the names our plotting code expects
    # decode
    if "decode_ms_per_tok" in metrics:
        out["ms_per_tok"] = float(metrics["decode_ms_per_tok"])
    if "decode_tok_s" in metrics:
        out["tok_per_s"] = float(metrics["decode_tok_s"])
    if "decode_total_ms" in metrics:
        out["decode_total_ms"] = float(metrics["decode_total_ms"])

    # prefill (if present)
    if "prefill_total_ms" in metrics:
        out["prefill_total_ms"] = float(metrics["prefill_total_ms"])
    if "prefill_ms_per_tok" in metrics and out.get("ms_per_tok") is None:
        # sometimes prefill-only runs might store prefill_ms_per_tok
        out["ms_per_tok"] = float(metrics["prefill_ms_per_tok"])

    # peak memory in MB (from bytes)
    peak_alloc_bytes = metrics.get("peak_alloc_bytes", None)
    peak_reserved_bytes = metrics.get("peak_reserved_bytes", None)

    if peak_alloc_bytes is not None:
        out["peak_mem_mb"] = _to_mb(float(peak_alloc_bytes))
    elif peak_reserved_bytes is not None:
        out["peak_mem_mb"] = _to_mb(float(peak_reserved_bytes))

    # Also optionally expose params if you ever want them later
    # (we don't rely on these for plotting)
    if "phase" in params:
        out["phase_from_json"] = params["phase"]

    return out


def parse_log(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")

    # 1) Prefer JSON artifact if available
    out = parse_metrics_from_json_artifact(txt, log_path)
    if out:
        return out

    # 2) Fallback to regex parsing the log itself
    out = {}
    for k, pats in METRIC_REGEX.items():
        out[k] = first_match(txt, pats)

    # derive peak_mem_mb if we have bytes
    peak_alloc = out.get("peak_alloc_bytes")
    peak_res = out.get("peak_reserved_bytes")

    peak_mem_mb = None
    if peak_alloc is not None:
        peak_mem_mb = _to_mb(peak_alloc)
    elif peak_res is not None:
        peak_mem_mb = _to_mb(peak_res)

    out["peak_mem_mb"] = peak_mem_mb

    return out


def kv_label(n_head: int, n_kv_head: int) -> str:
    if n_kv_head == n_head:
        return f"MHA (kv={n_kv_head})"
    if n_kv_head == 1:
        return f"MQA (kv={n_kv_head})"
    return f"GQA (kv={n_kv_head})"


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/gqa/sweep.csv")
    ap.add_argument("--plots_dir", default="results/gqa/plots")
    ap.add_argument("--write_parsed_csv", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    plots_dir = Path(args.plots_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sweep csv: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep only successful runs
    if "rc" in df.columns:
        df = df[df["rc"] == 0].copy()

    # Parse logs
    rows = []
    for _, r in df.iterrows():
        log_path = Path(r["log_path"])
        if not log_path.exists():
            continue
        metrics = parse_log(log_path)
        merged = dict(r)
        merged.update(metrics)
        rows.append(merged)

    if not rows:
        raise RuntimeError(
            "Parsed 0 rows. Either logs are missing, rc!=0, or parsing didn't find metrics.\n"
            "Open one rawlogs/*.log and confirm it contains either:\n"
            "  - a line like: Wrote: <path>.json  (preferred)\n"
            "  - or a metrics block that matches METRIC_REGEX.\n"
        )

    pdf = pd.DataFrame(rows)

    # Add labels for grouping
    pdf["mode"] = pdf.apply(lambda x: kv_label(int(x["n_head"]), int(x["n_kv_head"])), axis=1)

    # Optional debug CSV (great for sanity checking extraction)
    if args.write_parsed_csv:
        out_parsed = csv_path.parent / "parsed.csv"
        pdf.to_csv(out_parsed, index=False)
        print(f"Wrote parsed table: {out_parsed}")

    # Decide metric preference:
    # - If ms_per_tok exists, use it
    # - else if tok_per_s exists, use it
    # - else fallback to decode_total_ms for decode phase
    phase = str(pdf["phase"].iloc[0]) if "phase" in pdf.columns else "decode"

    metric = None
    if "ms_per_tok" in pdf.columns and pdf["ms_per_tok"].notna().any():
        metric = "ms_per_tok"
        ylab = "ms / token"
    elif "tok_per_s" in pdf.columns and pdf["tok_per_s"].notna().any():
        metric = "tok_per_s"
        ylab = "tokens / second"
    elif phase == "decode" and "decode_total_ms" in pdf.columns and pdf["decode_total_ms"].notna().any():
        metric = "decode_total_ms"
        ylab = "decode total (ms)"
    elif phase == "prefill" and "prefill_total_ms" in pdf.columns and pdf["prefill_total_ms"].notna().any():
        metric = "prefill_total_ms"
        ylab = "prefill total (ms)"
    else:
        # Dump a tiny hint to help debug quickly
        available = sorted([c for c in pdf.columns if c in ("ms_per_tok", "tok_per_s", "decode_total_ms", "prefill_total_ms", "peak_mem_mb")])
        raise RuntimeError(
            "No usable metrics found in logs.\n"
            f"Available parsed metric columns: {available}\n"
            "If your logs write JSON, ensure the log contains a line: Wrote: <path>.json\n"
        )

    print(f"Using metric: {metric} ({ylab})")
    print(f"Rows with metric present: {pdf[metric].notna().sum()} / {len(pdf)}")

    # -----------------------
    # Plot 1: metric vs prompt_len for each kv head, per batch size
    # -----------------------
    for B in sorted(pdf["batch_size"].unique()):
        sub = pdf[pdf["batch_size"] == B].copy()
        sub = sub[sub[metric].notna()]
        if sub.empty:
            continue

        fig = plt.figure()
        ax = plt.gca()

        for n_kv in sorted(sub["n_kv_head"].unique()):
            s2 = sub[sub["n_kv_head"] == n_kv].sort_values("prompt_len")
            ax.plot(
                s2["prompt_len"],
                s2[metric],
                marker="o",
                label=kv_label(int(s2["n_head"].iloc[0]), int(n_kv)),
            )

        ax.set_title(f"{phase}: {ylab} vs prompt_len (batch={B})")
        ax.set_xlabel("prompt_len")
        ax.set_ylabel(ylab)
        ax.legend()

        out_path = plots_dir / f"{phase}_{metric}_vs_promptlen_B{B}.png"
        save_plot(fig, out_path)

    # -----------------------
    # Plot 2: metric vs batch_size for each kv head, per prompt length
    # -----------------------
    for T in sorted(pdf["prompt_len"].unique()):
        sub = pdf[pdf["prompt_len"] == T].copy()
        sub = sub[sub[metric].notna()]
        if sub.empty:
            continue

        fig = plt.figure()
        ax = plt.gca()

        for n_kv in sorted(sub["n_kv_head"].unique()):
            s2 = sub[sub["n_kv_head"] == n_kv].sort_values("batch_size")
            ax.plot(
                s2["batch_size"],
                s2[metric],
                marker="o",
                label=kv_label(int(s2["n_head"].iloc[0]), int(n_kv)),
            )

        ax.set_title(f"{phase}: {ylab} vs batch_size (prompt_len={T})")
        ax.set_xlabel("batch_size")
        ax.set_ylabel(ylab)
        ax.legend()

        out_path = plots_dir / f"{phase}_{metric}_vs_batchsize_T{T}.png"
        save_plot(fig, out_path)

    # -----------------------
    # Plot 3: peak memory vs prompt_len (if present)
    # -----------------------
    if "peak_mem_mb" in pdf.columns and pdf["peak_mem_mb"].notna().any():
        for B in sorted(pdf["batch_size"].unique()):
            sub = pdf[(pdf["batch_size"] == B) & (pdf["peak_mem_mb"].notna())].copy()
            if sub.empty:
                continue

            fig = plt.figure()
            ax = plt.gca()

            for n_kv in sorted(sub["n_kv_head"].unique()):
                s2 = sub[sub["n_kv_head"] == n_kv].sort_values("prompt_len")
                ax.plot(
                    s2["prompt_len"],
                    s2["peak_mem_mb"],
                    marker="o",
                    label=kv_label(int(s2["n_head"].iloc[0]), int(n_kv)),
                )

            ax.set_title(f"{phase}: peak_mem_mb vs prompt_len (batch={B})")
            ax.set_xlabel("prompt_len")
            ax.set_ylabel("peak memory (MB)")
            ax.legend()

            out_path = plots_dir / f"{phase}_peakmem_vs_promptlen_B{B}.png"
            save_plot(fig, out_path)

    print(f"Done. Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
