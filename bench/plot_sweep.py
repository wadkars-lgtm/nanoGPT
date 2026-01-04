"""
plot_sweep.py

Plot sweep results produced by sweep_n_kv_head.py.

Inputs:
  - results/gqa/sweep.csv
  - results/gqa/rawlogs/*.log

Outputs:
  - results/gqa/plots/*.png
  - results/gqa/parsed_<norm>_<rope>.csv   (optional, handy for debugging)

This script does NOT run any CUDA. It only parses logs + plots.

Robustness:
- Prefer parsing the JSON artifact referenced by a log line like:
    Wrote: bench_out\\infer_T2048_B16_KVtrue.json
- Fallback to regex parsing of the log text.

RoPE note:
- You can filter plots to a specific RoPE mode via --use_rope (true|false).
- If you DO NOT pass --use_rope, plots are split per use_rope value (if present).
- Output filenames ALWAYS include both norm + rope suffixes:
    ..._layernorm_rope.png / ..._layernorm_norope.png
    ..._rmsnorm_rope.png / ..._rmsnorm_norope.png

Norm note:
- You can filter plots to a specific norm type (layernorm or rmsnorm) via --norm_type.
- Norm type is taken from JSON params.norm_type when available.
- Fallback inference: if the JSON artifact filename contains 'rmsnorm', we treat it as rmsnorm,
  else layernorm.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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


def _parse_boolish(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "on"):
        return True
    if s in ("0", "false", "no", "n", "f", "off"):
        return False
    raise ValueError(f"Cannot parse bool from: {v!r}")


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


def _norm_from_path(p: Path) -> str:
    s = str(p).lower()
    return "rmsnorm" if "rmsnorm" in s else "layernorm"


def parse_metrics_from_json_artifact(log_text: str, log_path: Path) -> dict:
    """
    If the log contains:  Wrote: <path-to-json>
    load that JSON and return a normalized metric dict.

    Also attempts to pull:
      - phase (params.phase)
      - use_rope (params.use_rope)
      - norm_type (params.norm_type)
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

    out: dict[str, float | str | bool | None] = {}

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

    # optional: phase, rope, norm
    if "phase" in params:
        out["phase_from_json"] = str(params["phase"])

    if "use_rope" in params:
        out["use_rope_from_json"] = params["use_rope"]

    if "norm_type" in params:
        out["norm_type"] = str(params["norm_type"]).strip().lower()
    else:
        out["norm_type"] = _norm_from_path(p)

    # keep the json path around (useful for debugging)
    out["json_path"] = str(p)

    return out


def parse_log(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")

    # 1) Prefer JSON artifact if available
    out = parse_metrics_from_json_artifact(txt, log_path)
    if out:
        return out

    # 2) Fallback to regex parsing the log itself
    out2: dict[str, float | str | None] = {}
    for k, pats in METRIC_REGEX.items():
        out2[k] = first_match(txt, pats)

    peak_alloc = out2.get("peak_alloc_bytes")
    peak_res = out2.get("peak_reserved_bytes")

    peak_mem_mb = None
    if peak_alloc is not None:
        peak_mem_mb = _to_mb(peak_alloc)
    elif peak_res is not None:
        peak_mem_mb = _to_mb(peak_res)

    out2["peak_mem_mb"] = peak_mem_mb

    # best-effort inference for norm + rope
    out2["norm_type"] = _norm_from_path(log_path)
    s = str(log_path).lower()
    if "_rope_" in s or s.endswith("_rope.log"):
        out2["use_rope_from_json"] = True
    if "_seq_" in s or s.endswith("_seq.log") or "_norope" in s:
        out2["use_rope_from_json"] = False

    return out2


def kv_label(n_head: int, n_kv_head: int) -> str:
    if n_kv_head == n_head:
        return f"MHA (kv={n_kv_head})"
    if n_kv_head == 1:
        return f"MQA (kv={n_kv_head})"
    return f"GQA (kv={n_kv_head})"


def rope_suffix(use_rope_val) -> str:
    if isinstance(use_rope_val, bool):
        return "rope" if use_rope_val else "norope"
    s = str(use_rope_val).strip().lower()
    return "rope" if s in ("1", "true", "yes", "y", "t", "on") else "norope"


def rope_tag(use_rope_val) -> str:
    return "RoPE" if rope_suffix(use_rope_val) == "rope" else "NoRoPE"


def save_plot(fig, out_path: Path) -> None:
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


def _norm_file_suffix(norm_type: str) -> str:
    """
    ALWAYS include norm in filenames to avoid overwrites and make folders self-describing.
    """
    nt = str(norm_type).strip().lower()
    if nt not in ("layernorm", "rmsnorm"):
        raise ValueError(f"unknown norm_type={norm_type}")
    return f"_{nt}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/gqa/sweep.csv")
    ap.add_argument("--plots_dir", default="results/gqa/plots")
    ap.add_argument("--write_parsed_csv", action="store_true")
    ap.add_argument("--norm_type", default="layernorm", choices=["layernorm", "rmsnorm"])
    ap.add_argument(
        "--use_rope",
        default=None,
        help="Optional filter: true|false. If omitted, plots are split per use_rope (if present).",
    )

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

        # If JSON included phase/use_rope, prefer that
        if "phase_from_json" in merged and "phase" in merged:
            merged["phase"] = merged["phase_from_json"]

        # if we got rope from json, populate/override use_rope in frame
        if "use_rope_from_json" in merged:
            merged["use_rope"] = merged["use_rope_from_json"]

        rows.append(merged)

    if not rows:
        raise RuntimeError(
            "Parsed 0 rows. Either logs are missing, rc!=0, or parsing didn't find metrics.\n"
            "Open one rawlogs/*.log and confirm it contains either:\n"
            "  - a line like: Wrote: <path>.json  (preferred)\n"
            "  - or a metrics block that matches METRIC_REGEX.\n"
        )

    pdf = pd.DataFrame(rows)

    # -----------------------
    # Filter by norm type
    # -----------------------
    wanted_norm = str(args.norm_type).strip().lower()
    if "norm_type" not in pdf.columns:
        raise RuntimeError(
            "norm_type not found in parsed data. "
            "Ensure batch_infer.py writes params.norm_type into the JSON artifacts, "
            "or encode rmsnorm in the artifact filename (contains 'rmsnorm')."
        )

    pdf["norm_type"] = pdf["norm_type"].fillna("layernorm").astype(str).str.lower()
    pdf = pdf[pdf["norm_type"] == wanted_norm].copy()

    if pdf.empty:
        raise RuntimeError(
            f"No rows remain after filtering to norm_type={wanted_norm}. "
            "Confirm you ran the sweep with that norm_type and that logs point to JSON artifacts."
        )

    # -----------------------
    # Ensure use_rope is present + optionally filter it
    # -----------------------
    if "use_rope" not in pdf.columns:
        raise RuntimeError(
            "use_rope not found in parsed data. "
            "Ensure batch_infer.py writes params.use_rope into the JSON artifacts "
            "or sweep_n_kv_head.py records use_rope in sweep.csv."
        )

    # Normalize to bool for consistent grouping/filtering
    pdf["use_rope_bool"] = pdf["use_rope"].apply(_parse_boolish)

    wanted_rope: bool | None = None
    if args.use_rope is not None:
        wanted_rope = _parse_boolish(args.use_rope)
        pdf = pdf[pdf["use_rope_bool"] == wanted_rope].copy()
        if pdf.empty:
            raise RuntimeError(f"No rows remain after filtering to use_rope={wanted_rope} and norm_type={wanted_norm}.")

    # Decide rope groups:
    # - if --use_rope provided => one group
    # - else => split groups
    if wanted_rope is not None:
        rope_groups = [wanted_rope]
    else:
        rope_groups = sorted(pdf["use_rope_bool"].unique().tolist(), key=lambda v: rope_suffix(v))

    # Optional debug CSV(s)
    if args.write_parsed_csv:
        for rv in rope_groups:
            sub = pdf[pdf["use_rope_bool"] == rv].copy()
            if sub.empty:
                continue
            out_parsed = csv_path.parent / f"parsed_{wanted_norm}_{rope_suffix(rv)}.csv"
            sub.to_csv(out_parsed, index=False)
            print(f"Wrote parsed table: {out_parsed}")

    def add_mode_col(frame: pd.DataFrame) -> pd.DataFrame:
        frame["mode"] = frame.apply(
            lambda x: f"{kv_label(int(x['n_head']), int(x['n_kv_head']))} [{rope_tag(x['use_rope_bool'])}]",
            axis=1,
        )
        return frame

    norm_title = "RMSNorm" if wanted_norm == "rmsnorm" else "LayerNorm"
    norm_file_suffix = _norm_file_suffix(wanted_norm)  # ALWAYS explicit

    for rope_val in rope_groups:
        cur = pdf[pdf["use_rope_bool"] == rope_val].copy()
        if cur.empty:
            continue

        rope_file_suffix = f"_{rope_suffix(rope_val)}"  # ALWAYS explicit
        rope_title_suffix = f" ({rope_tag(rope_val)})"

        cur = add_mode_col(cur)

        phase, metric, ylab = pick_metric_and_label(cur)

        title_suffix = f" [{norm_title}]{rope_title_suffix}"
        file_suffix = f"{norm_file_suffix}{rope_file_suffix}"

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
