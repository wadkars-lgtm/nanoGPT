"""
plot_sweep.py

Plot sweep results produced by sweep_n_kv_head.py.

Inputs:
  - results/gqa/sweep.csv
  - results/gqa/rawlogs/*.log

Outputs:
  - results/gqa/plots/*.png
  - results/gqa/parsed.csv   (optional, handy for debugging)

This script does NOT run any CUDA. It only parses logs + plots.

Robustness:
- Prefer parsing the JSON artifact referenced by a log line like:
    Wrote: bench_out\\infer_T2048_B16_KVtrue.json
- Fallback to regex parsing of the log text.

RoPE note:
- If your sweep includes BOTH use_rope=True and use_rope=False, plots can get overwritten
  unless we separate them. This script will:
    (1) split plots per use_rope value (if present in sweep.csv)
    (2) add a suffix (_rope / _norope) to filenames
    (3) include RoPE/NoRoPE in legend labels
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
METRIC_REGEX = {
    "ms_per_tok": [
        re.compile(r'"decode_ms_per_tok"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),
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

    # decode
    if "decode_ms_per_tok" in metrics:
        out["ms_per_tok"] = float(metrics["decode_ms_per_tok"])
    if "decode_tok_s" in metrics:
        out["tok_per_s"] = float(metrics["decode_tok_s"])
    if "decode_total_ms" in metrics:
        out["decode_total_ms"] = float(metrics["decode_total_ms"])

    # prefill
    if "prefill_total_ms" in metrics:
        out["prefill_total_ms"] = float(metrics["prefill_total_ms"])
    if "prefill_ms_per_tok" in metrics and out.get("ms_per_tok") is None:
        out["ms_per_tok"] = float(metrics["prefill_ms_per_tok"])

    # peak memory (bytes -> MB)
    peak_alloc_bytes = metrics.get("peak_alloc_bytes", None)
    peak_reserved_bytes = metrics.get("peak_reserved_bytes", None)

    if peak_alloc_bytes is not None:
        out["peak_mem_mb"] = _to_mb(float(peak_alloc_bytes))
    elif peak_reserved_bytes is not None:
        out["peak_mem_mb"] = _to_mb(float(peak_reserved_bytes))

    # optional
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


def rope_suffix(use_rope_val) -> str:
    # use_rope might be bool, int, or string depending on how csv got written
    if isinstance(use_rope_val, bool):
        return "rope" if use_rope_val else "norope"
    s = str(use_rope_val).strip().lower()
    return "rope" if s in ("1", "true", "yes", "y", "t") else "norope"


def rope_tag(use_rope_val) -> str:
    return "RoPE" if rope_suffix(use_rope_val) == "rope" else "NoRoPE"


def save_plot(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def pick_metric_and_label(pdf: pd.DataFrame) -> tuple[str, str, str]:
    # Decide metric preference:
    # - If ms_per_tok exists, use it
    # - else if tok_per_s exists, use it
    # - else fallback to decode_total_ms / prefill_total_ms
    phase = str(pdf["phase"].iloc[0]) if "phase" in pdf.columns else "decode"

    if "ms_per_tok" in pdf.columns and pdf["ms_per_tok"].notna().any():
        return phase, "ms_per_tok", "ms / token"
    if "tok_per_s" in pdf.columns and pdf["tok_per_s"].notna().any():
        return phase, "tok_per_s", "tokens / second"
    if phase == "decode" and "decode_total_ms" in pdf.columns and pdf["decode_total_ms"].notna().any():
        return phase, "decode_total_ms", "decode total (ms)"
    if phase == "prefill" and "prefill_total_ms" in pdf.columns and pdf["prefill_total_ms"].notna().any():
        return phase, "prefill_total_ms", "prefill total (ms)"

    available = sorted(
        [
            c
            for c in pdf.columns
            if c in ("ms_per_tok", "tok_per_s", "decode_total_ms", "prefill_total_ms", "peak_mem_mb")
        ]
    )
    raise RuntimeError(
        "No usable metrics found in logs.\n"
        f"Available parsed metric columns: {available}\n"
        "If your logs write JSON, ensure the log contains a line: Wrote: <path>.json\n"
    )


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

    # Optional debug CSV (great for sanity checking extraction)
    if args.write_parsed_csv:
        out_parsed = csv_path.parent / "parsed.csv"
        pdf.to_csv(out_parsed, index=False)
        print(f"Wrote parsed table: {out_parsed}")

    # Determine if we need to split by RoPE
    split_by_rope = "use_rope" in pdf.columns and pdf["use_rope"].nunique(dropna=False) > 1

    # Helper to add legend mode label
    def add_mode_col(frame: pd.DataFrame) -> pd.DataFrame:
        if "use_rope" in frame.columns:
            frame["mode"] = frame.apply(
                lambda x: f"{kv_label(int(x['n_head']), int(x['n_kv_head']))} [{rope_tag(x['use_rope'])}]",
                axis=1,
            )
        else:
            frame["mode"] = frame.apply(lambda x: kv_label(int(x["n_head"]), int(x["n_kv_head"])), axis=1)
        return frame

    # Plot per rope group if needed; else just once.
    rope_groups = [None]
    if split_by_rope:
        rope_groups = sorted(pdf["use_rope"].unique().tolist(), key=lambda v: rope_suffix(v))

    for rope_val in rope_groups:
        if rope_val is None:
            cur = pdf.copy()
            file_suffix = ""
            title_suffix = ""
        else:
            cur = pdf[pdf["use_rope"] == rope_val].copy()
            file_suffix = f"_{rope_suffix(rope_val)}"
            title_suffix = f" ({rope_tag(rope_val)})"

        if cur.empty:
            continue

        cur = add_mode_col(cur)

        phase, metric, ylab = pick_metric_and_label(cur)

        print(f"Using metric: {metric} ({ylab}){title_suffix}")
        print(f"Rows with metric present: {cur[metric].notna().sum()} / {len(cur)}")

        # -----------------------
        # Plot 1: metric vs prompt_len for each kv head, per batch size
        # -----------------------
        for B in sorted(cur["batch_size"].unique()):
            sub = cur[cur["batch_size"] == B].copy()
            sub = sub[sub[metric].notna()]
            if sub.empty:
                continue

            fig = plt.figure()
            ax = plt.gca()

            for mode in sorted(sub["mode"].unique()):
                s2 = sub[sub["mode"] == mode].sort_values("prompt_len")
                ax.plot(s2["prompt_len"], s2[metric], marker="o", label=mode)

            ax.set_title(f"{phase}: {ylab} vs prompt_len (batch={B}){title_suffix}")
            ax.set_xlabel("prompt_len")
            ax.set_ylabel(ylab)
            ax.legend()

            out_path = plots_dir / f"{phase}_{metric}_vs_promptlen_B{B}{file_suffix}.png"
            save_plot(fig, out_path)

        # -----------------------
        # Plot 2: metric vs batch_size for each kv head, per prompt length
        # -----------------------
        for T in sorted(cur["prompt_len"].unique()):
            sub = cur[cur["prompt_len"] == T].copy()
            sub = sub[sub[metric].notna()]
            if sub.empty:
                continue

            fig = plt.figure()
            ax = plt.gca()

            for mode in sorted(sub["mode"].unique()):
                s2 = sub[sub["mode"] == mode].sort_values("batch_size")
                ax.plot(s2["batch_size"], s2[metric], marker="o", label=mode)

            ax.set_title(f"{phase}: {ylab} vs batch_size (prompt_len={T}){title_suffix}")
            ax.set_xlabel("batch_size")
            ax.set_ylabel(ylab)
            ax.legend()

            out_path = plots_dir / f"{phase}_{metric}_vs_batchsize_T{T}{file_suffix}.png"
            save_plot(fig, out_path)

        # -----------------------
        # Plot 3: peak memory vs prompt_len (if present)
        # -----------------------
        if "peak_mem_mb" in cur.columns and cur["peak_mem_mb"].notna().any():
            for B in sorted(cur["batch_size"].unique()):
                sub = cur[(cur["batch_size"] == B) & (cur["peak_mem_mb"].notna())].copy()
                if sub.empty:
                    continue

                fig = plt.figure()
                ax = plt.gca()

                for mode in sorted(sub["mode"].unique()):
                    s2 = sub[sub["mode"] == mode].sort_values("prompt_len")
                    ax.plot(s2["prompt_len"], s2["peak_mem_mb"], marker="o", label=mode)

                ax.set_title(f"{phase}: peak_mem_mb vs prompt_len (batch={B}){title_suffix}")
                ax.set_xlabel("prompt_len")
                ax.set_ylabel("peak memory (MB)")
                ax.legend()

                out_path = plots_dir / f"{phase}_peakmem_vs_promptlen_B{B}{file_suffix}.png"
                save_plot(fig, out_path)

    print(f"Done. Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
