"""Build paper-ready tables and figures for Experiment 1 and Experiment 2.

Reads:
  - data/exp_results/old2/exp1_retrieval_{raw,parsed}_top{3,5}_results.csv  (Exp 1)
  - data/exp_results/scaling_per_condition.csv                              (Run B: 4 conds)
  - data/exp_results/scaling_per_log.csv                                    (Run B per-log)
  - experiments/logs/exp2_generation_20260520_191836.log                    (Run A: 3 conds)

Writes:
  - data/exp_results/paper_figures/exp1_table.csv
  - data/exp_results/paper_figures/exp2_table.csv
  - data/exp_results/paper_figures/tables.md
  - data/exp_results/paper_figures/fig_exp1_recall.png
  - data/exp_results/paper_figures/fig_exp2_quality.png
  - data/exp_results/paper_figures/fig_exp2_cost.png
  - data/exp_results/paper_figures/fig_exp2_oomtype_heatmap.png
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "data" / "exp_results"
OUT_DIR = RESULT_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# ---------------------------------------------------------------------------
# Exp 1: retrieval Recall@K  (Raw vs Structured)
# ---------------------------------------------------------------------------

def build_exp1_table() -> pd.DataFrame:
    src = RESULT_DIR / "old2"
    rows = []
    for mode in ("raw", "parsed"):
        for k in (3, 5):
            f = src / f"exp1_retrieval_{mode}_top{k}_results.csv"
            df = pd.read_csv(f)
            rows.append({
                "query_mode": "Raw log" if mode == "raw" else "Structured",
                "top_k": k,
                "n": len(df),
                f"recall_mean": float(df["recall"].mean()),
                f"recall_std": float(df["recall"].std(ddof=0)),
            })
    return pd.DataFrame(rows)


def plot_exp1(df: pd.DataFrame, path: Path) -> None:
    pivot = df.pivot(index="top_k", columns="query_mode", values="recall_mean").reindex(columns=["Raw log", "Structured"])
    ks = pivot.index.tolist()
    x = np.arange(len(ks))
    w = 0.32
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars_r = ax.bar(x - w/2, pivot["Raw log"], w, label="Raw log", color="#7f7f7f", edgecolor="black", linewidth=0.6)
    bars_s = ax.bar(x + w/2, pivot["Structured"], w, label="Structured", color="#1f77b4", edgecolor="black", linewidth=0.6)
    for bs in (bars_r, bars_s):
        for b in bs:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylim(0, max(pivot.max().max() * 1.35, 0.4))
    ax.set_ylabel("Mean Recall")
    ax.set_title("Experiment 1: Retrieval quality — Raw vs Structured query")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2: parse Run A log to recover 3 conditions, merge with Run B CSV
# ---------------------------------------------------------------------------

LOG_PATH_RUN_A = PROJECT_ROOT / "experiments" / "logs" / "exp2_generation_20260520_191836.log"

LINE_RE = re.compile(
    r"\[(\d{2})/56\]\s+(\S+)\s+cat=(\d+)\s+ev=([\d.]+)\s+ag=([\d.]+)\s+"
    r"t=([\d.]+)s\s+"
    r"(?:vram=(\S+)\s+)?"
    r"calls=(\d+)\s+tok=(\d+)/(\d+)\s+err=(\w)"
)


def parse_run_a_conditions() -> dict[str, dict]:
    """Walk log A and aggregate per-condition statistics for conditions that
    actually finished (have a successful `<<< Condition done` line)."""
    if not LOG_PATH_RUN_A.exists():
        return {}
    text = LOG_PATH_RUN_A.read_text(encoding="utf-8").splitlines()
    cond_label: str | None = None
    cond_model: str | None = None
    cond_mode: str | None = None
    data: dict[str, list[dict]] = {}
    done: set[str] = set()
    failed: set[str] = set()
    for L in text:
        m = re.search(r"=== Condition:\s+(.+?)\s+\(model=([\w\-\.]+),\s+mode=(\w+)\)\s+===", L)
        if m:
            cond_label, cond_model, cond_mode = m.group(1), m.group(2), m.group(3)
            data.setdefault(cond_label, [])
            continue
        if cond_label and "<<< Condition done" in L:
            done.add(cond_label)
            continue
        if cond_label and "!!! Condition failed" in L:
            failed.add(cond_label)
            continue
        if cond_label is None:
            continue
        m = LINE_RE.search(L)
        if m:
            data[cond_label].append({
                "cat": int(m.group(3)),
                "ev": float(m.group(4)),
                "ag": float(m.group(5)),
                "sec": float(m.group(6)),
                "vram": float(re.sub(r"GB$", "", m.group(7))) if (m.group(7) and m.group(7) not in ("NaN",)) else float("nan"),
                "calls": int(m.group(8)),
                "in_tok": int(m.group(9)),
                "out_tok": int(m.group(10)),
                "err": 1 if m.group(11) == "Y" else 0,
                "model": cond_model,
                "mode": cond_mode,
            })

    agg: dict[str, dict] = {}
    for label, rows in data.items():
        if label in failed or not rows or label not in done:
            continue
        n = len(rows)
        times = sorted([r["sec"] for r in rows])
        vrams = [r["vram"] for r in rows if not math.isnan(r["vram"])]
        agg[label] = {
            "label": label,
            "model": rows[0]["model"],
            "mode": rows[0]["mode"],
            "n": n,
            "category_match": sum(r["cat"] for r in rows) / n,
            "evidence_recall": sum(r["ev"] for r in rows) / n,
            "action_guide_similarity": sum(r["ag"] for r in rows) / n,
            "error_rate": sum(r["err"] for r in rows) / n,
            "sec_per_query": statistics.mean(times),
            "lat_p50": times[n // 2],
            "lat_p95": times[int(n * 0.95) if int(n * 0.95) < n else n - 1],
            "avg_input_tokens": sum(r["in_tok"] for r in rows) / n,
            "avg_output_tokens": sum(r["out_tok"] for r in rows) / n,
            "avg_calls": sum(r["calls"] for r in rows) / n,
            "vram_gb_per_query_mean": statistics.mean(vrams) if vrams else float("nan"),
            "vram_gb_peak": max(vrams) if vrams else float("nan"),
        }
    return agg


def build_exp2_table() -> pd.DataFrame:
    # Run B (small models RAG): scaling_per_condition.csv has full columns
    cond_csv = RESULT_DIR / "scaling_per_condition.csv"
    df_b = pd.read_csv(cond_csv)
    # Run A (Qwen 9B RAG/Naive, GPT-5.2 Naive): parsed from log
    a_aggs = parse_run_a_conditions()
    # Per-log-derived avg_calls for Run B
    per_log = pd.read_csv(RESULT_DIR / "scaling_per_log.csv")
    avg_calls_b = per_log.groupby("label")["n_llm_calls"].mean().to_dict()

    rows: list[dict] = []
    # Run B rows (keep all available columns)
    for _, r in df_b.iterrows():
        rows.append({
            "label": r["label"],
            "model": r["model"],
            "mode": r["mode"],
            "n": int(r["n"]),
            "category_match": r["category_match"],
            "evidence_recall": r["evidence_recall"],
            "action_guide_similarity": r["action_guide_similarity"],
            "error_rate": r["error_rate"],
            "json_first_try_rate": r["json_first_try_rate"],
            "sec_per_query": r["sec_per_query"],
            "lat_p50": r["lat_p50"],
            "lat_p95": r["lat_p95"],
            "avg_input_tokens": r["avg_input_tokens"],
            "avg_output_tokens": r["avg_output_tokens"],
            "avg_calls": float(avg_calls_b.get(r["label"], float("nan"))),
            "avg_cost_usd": r["avg_cost_usd"],
            "vram_gb_per_query_mean": r["vram_gb_per_query_mean"],
            "vram_gb_peak": r["vram_gb_peak"],
        })
    # Run A rows (json_first_try_rate, avg_cost_usd from Final aggregate; parsed otherwise)
    final_extras = {
        "Qwen3.5-9B (RAG)": {"json_first_try_rate": 0.98, "avg_cost_usd": 0.0008},
        "Qwen3.5-9B (Naive)": {"json_first_try_rate": 0.00, "avg_cost_usd": 0.0021},
        "gpt-5.2 (Naive)":    {"json_first_try_rate": 1.00, "avg_cost_usd": 0.0249},
    }
    for label, info in a_aggs.items():
        ex = final_extras.get(label, {})
        rows.append({
            "label": label,
            "model": info["model"],
            "mode": info["mode"],
            "n": info["n"],
            "category_match": info["category_match"],
            "evidence_recall": info["evidence_recall"],
            "action_guide_similarity": info["action_guide_similarity"],
            "error_rate": info["error_rate"],
            "json_first_try_rate": ex.get("json_first_try_rate", float("nan")),
            "sec_per_query": info["sec_per_query"],
            "lat_p50": info["lat_p50"],
            "lat_p95": info["lat_p95"],
            "avg_input_tokens": info["avg_input_tokens"],
            "avg_output_tokens": info["avg_output_tokens"],
            "avg_calls": info["avg_calls"],
            "avg_cost_usd": ex.get("avg_cost_usd", float("nan")),
            "vram_gb_per_query_mean": info["vram_gb_per_query_mean"],
            "vram_gb_peak": info["vram_gb_peak"],
        })

    df = pd.DataFrame(rows)
    # Order: RAG first (sorted by family/size), then Naive
    df["sort_key"] = df.apply(
        lambda r: (0 if r["mode"] == "rag" else 1, r["model"].startswith("gpt"), r["label"]),
        axis=1,
    )
    df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Visualizations (paper-ready)
# ---------------------------------------------------------------------------

def plot_exp2_quality(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars: Category / Evidence / Guide per condition."""
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    w = 0.27
    cats = df["category_match"].astype(float).tolist()
    evs = df["evidence_recall"].astype(float).tolist()
    ags = df["action_guide_similarity"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(x - w, cats, w, label="Category Match", color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(x,      evs,  w, label="Evidence Recall", color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.bar(x + w,  ags,  w, label="Action Guide Similarity", color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: Generation quality per condition")
    # Place legend OUTSIDE the axes (right side) to avoid any collision with
    # rotated x-tick labels below the plot.
    ax.legend(loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_cost_latency(df: pd.DataFrame, path: Path) -> None:
    """Scatter: sec/query (X) vs cost/query (Y), bubble = VRAM.

    Conditions with NaN VRAM (GPT) drawn with hollow marker. Label offsets are
    chosen per condition so they do not overlap each other or the legend.
    """
    # (dx_pts, dy_pts, anchor) per label, manually chosen to avoid overlap.
    label_offsets = {
        "gpt-5.2 (Naive)":     ( 10,  -4, "left"),
        "Qwen3.5-9B (Naive)":  (-10,   7, "right"),
        "Qwen3.5-9B (RAG)":    ( 10,   8, "left"),
        "Gemma4-E4B (RAG)":    (-10,  -8, "right"),
        "Gemma4-E2B (RAG)":    ( 10,  -4, "left"),
        "Qwen3.5-2B (RAG)":    ( 10,  -10, "left"),
        "Qwen3.5-0.8B (RAG)":  (-10,  -4, "right"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for _, r in df.iterrows():
        vram = r["vram_gb_per_query_mean"]
        size = (vram * 12) if not (isinstance(vram, float) and math.isnan(vram)) else 60
        face = "tab:blue" if r["mode"] == "rag" else "tab:orange"
        marker_filled = not (isinstance(vram, float) and math.isnan(vram))
        ax.scatter(r["sec_per_query"], r["avg_cost_usd"],
                   s=size, c=face if marker_filled else "white",
                   edgecolors="black", linewidths=1.0,
                   alpha=0.9 if marker_filled else 1.0, marker="o")
        dx, dy, ha = label_offsets.get(r["label"], (8, 4, "left"))
        ax.annotate(r["label"], (r["sec_per_query"], r["avg_cost_usd"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=ha,
                    va="center" if abs(dy) < 5 else ("bottom" if dy > 0 else "top"))
    ax.set_xlabel("Sec per query (lower is better)")
    ax.set_ylabel("Estimated cost per query (USD)")
    ax.set_yscale("log")
    ax.set_title("Experiment 2: Latency vs cost (bubble ∝ VRAM/query)")
    ax.set_xlim(left=-15)
    ax.margins(x=0.10, y=0.22)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="black", markersize=8, label="RAG"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markeredgecolor="black", markersize=8, label="Naive"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Remote (VRAM n/a)"),
    ]
    # Empty quadrant for legend is upper-right (high sec, high cost).
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_exp2_quality_vs_latency(df: pd.DataFrame, path: Path) -> None:
    """Scatter: sec/query (X) vs the three quality metrics (Y).

    Each condition contributes three markers at the same X (its latency):
    Category Match (circle), Evidence Recall (square), Action Guide
    Similarity (triangle). The model name is annotated once, next to its
    Category Match marker. Per-label offsets avoid collisions.
    """
    metrics = [
        ("category_match", "Category Match", "o", "#1f77b4"),
        ("evidence_recall", "Evidence Recall", "s", "#ff7f0e"),
        ("action_guide_similarity", "Action Guide Similarity", "^", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    # Drop a full-height black dotted line at each model's X (same sec/query)
    # so a reader can group that model's three metrics at a glance.
    for _, r in df.iterrows():
        ax.axvline(r["sec_per_query"], color="0.55",
                   linestyle=(0, (4, 5)), linewidth=1.0,
                   dash_capstyle="butt", zorder=1)
    for col, lbl, mk, color in metrics:
        ax.scatter(df["sec_per_query"].astype(float), df[col].astype(float),
                   s=70, marker=mk, c=color, edgecolors="black",
                   linewidths=0.6, alpha=0.9, label=lbl, zorder=3,
                   clip_on=False)

    # Anchor every model label at the TOP of its own dotted line (y=1.0),
    # centered on the line, so each label sits directly above its line and the
    # line carries the eye down to that model's three markers. Labels are
    # staggered into three vertical tiers so neighbours never collide.
    tier_dy = {
        "gpt-5.2 (Naive)":     8,
        "Gemma4-E2B (RAG)":    8,
        "Gemma4-E4B (RAG)":   26,
        "Qwen3.5-9B (RAG)":   46,
        "Qwen3.5-2B (RAG)":   26,
        "Qwen3.5-0.8B (RAG)":  8,
        "Qwen3.5-9B (Naive)":  8,
    }
    for _, r in df.iterrows():
        dy = tier_dy.get(r["label"], 8)
        ax.annotate(r["label"], (r["sec_per_query"], 1.0),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=7.5, ha="center", va="bottom")

    ax.set_xlabel("Sec per query (lower is better)")
    ax.set_ylabel("Score (higher is better)")
    # Scores are bounded at 1.0, so cap the axis there. The cat=1.0 markers sit
    # on the top spine (clip_on=False keeps them whole) and the model labels
    # render in the margin just above it.
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(left=-5)
    # Place legend inside the empty mid band (cat markers sit at 1.0, Ev/Ag
    # cluster at 0.55-0.66, leaving y~0.75-0.95 free across all x). Keeping it
    # inside preserves full plot width so the top-tier labels stay separated,
    # and it no longer covers the rightmost Qwen3.5-9B (Naive) Ev/Ag markers.
    ax.legend(loc="center", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_pareto(df: pd.DataFrame, path: Path) -> None:
    """Decision-support scatter: speed (X) vs VRAM (Y), colored by accuracy.

    A reader can pick a model by looking for the lower-left bubble that is
    still the darkest green. RAG vs Naive are encoded by marker shape; the
    remote GPT model (no local VRAM) sits on the y=0 baseline with a separate
    marker so it does not skew the Y axis.
    """
    # Combined accuracy = mean of three quality metrics (must precede subsetting)
    df = df.copy()
    df["accuracy_mean"] = (
        df["category_match"].astype(float)
        + df["evidence_recall"].astype(float)
        + df["action_guide_similarity"].astype(float)
    ) / 3.0

    rag = df[df["mode"] == "rag"].copy()
    naive_local = df[(df["mode"] == "naive") & ~df["vram_gb_per_query_mean"].isna()].copy()
    remote = df[df["vram_gb_per_query_mean"].isna()].copy()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=0.55, vmax=0.80)

    def _plot(sub: pd.DataFrame, marker: str, edge: str = "black", y_override=None):
        if sub.empty:
            return
        xs = sub["sec_per_query"].astype(float).to_numpy()
        ys = (sub["vram_gb_per_query_mean"].astype(float).to_numpy()
              if y_override is None else np.full(len(sub), y_override, dtype=float))
        cs = sub["accuracy_mean"].astype(float).to_numpy()
        ax.scatter(xs, ys, s=210, c=cs, cmap=cmap, norm=norm,
                   marker=marker, edgecolors=edge, linewidths=0.9, zorder=3)

    # Encode VRAM≈0 baseline for the remote GPT model so it stays on chart.
    BASELINE_Y = -1.5
    _plot(rag, "o")
    _plot(naive_local, "^")
    _plot(remote, "D", y_override=BASELINE_Y)

    # Per-label offsets to avoid clobbering each other.
    label_offsets = {
        "Gemma4-E2B (RAG)":   (  0, -12, "center"),
        "Gemma4-E4B (RAG)":   (  0, -12, "center"),
        "Qwen3.5-9B (RAG)":   ( 12,   8, "left"),
        "Qwen3.5-2B (RAG)":   (  0,   8, "center"),
        "Qwen3.5-0.8B (RAG)": ( 10,  -4, "left"),
        "Qwen3.5-9B (Naive)": (  0, -12, "center"),
        "gpt-5.2 (Naive)":    ( 12,  -2, "left"),
    }
    for _, r in df.iterrows():
        x = r["sec_per_query"]
        y = r["vram_gb_per_query_mean"]
        if pd.isna(y):
            y = BASELINE_Y
        dx, dy, ha = label_offsets.get(r["label"], (8, 6, "left"))
        ax.annotate(r["label"], (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, ha=ha,
                    va="bottom" if dy > 0 else "top")

    # Reference horizontal line marking the y=0 baseline used for remote models
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.text(ax.get_xlim()[1] if False else 175, 0, " local→ ", ha="left", va="bottom",
            fontsize=7.5, color="gray")
    ax.text(175, BASELINE_Y, " remote ", ha="left", va="center",
            fontsize=7.5, color="gray")

    ax.set_xlabel("Sec per query")
    ax.set_ylabel("VRAM per query (GB)")
    ax.set_xlim(left=-5, right=190)
    ax.set_ylim(bottom=-4, top=50)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Colorbar (accuracy)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy (mean of Cat, Ev, Ag)")

    # Marker shape legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="black",
                   markersize=11, label="RAG"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markeredgecolor="black",
                   markersize=11, label="Naive (local)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#888", markeredgecolor="black",
                   markersize=10, label="Naive (remote)"),
    ]
    # Legend outside the plot (right side) so it never overlaps a data label.
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
              bbox_to_anchor=(1.18, 1.0))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_oomtype_heatmap(path: Path) -> None:
    """Per oom_type Category Match heatmap.

    Pulls per-log labels from scaling_per_log.csv (Run B small models, RAG).
    For Run A's 3 conditions (Qwen 9B RAG/Naive, gpt-5.2 Naive), re-derive
    from the log file.
    """
    rows: list[dict] = []
    per_log_b = pd.read_csv(RESULT_DIR / "scaling_per_log.csv")
    if "expected_oom_type" in per_log_b.columns:
        tab_b = (per_log_b.groupby(["label", "expected_oom_type"])["category_match"]
                 .mean().reset_index())
        rows.extend(tab_b.to_dict("records"))

    # Run A per-log lines: parse to recover expected_oom_type via
    # log_id -> ground truth dataset
    qa_path = PROJECT_ROOT / "data" / "qa_ground_truth.jsonl"
    import json as _json
    qa_map = {}
    if qa_path.exists():
        with qa_path.open(encoding="utf-8") as f:
            for L in f:
                if not L.strip():
                    continue
                o = _json.loads(L)
                qa_map[o["log_id"]] = o.get("expected_oom_type", "")

    text = LOG_PATH_RUN_A.read_text(encoding="utf-8").splitlines()
    cond_label = None
    block: dict[str, list[tuple[str, int]]] = {}
    failed: set[str] = set()
    done: set[str] = set()
    line_id_re = re.compile(r"\[(\d+)/56\]\s+(\S+)\s+cat=(\d+)")
    for L in text:
        m = re.search(r"=== Condition:\s+(.+?)\s+\(model=", L)
        if m:
            cond_label = m.group(1)
            block.setdefault(cond_label, [])
            continue
        if cond_label and "<<< Condition done" in L:
            done.add(cond_label)
            continue
        if cond_label and "!!! Condition failed" in L:
            failed.add(cond_label)
            continue
        m = line_id_re.search(L)
        if m and cond_label:
            block[cond_label].append((m.group(2), int(m.group(3))))

    for label, items in block.items():
        if label in failed or label not in done or not items:
            continue
        per_type: dict[str, list[int]] = {}
        for log_id, cat in items:
            ot = qa_map.get(log_id, "unknown")
            per_type.setdefault(ot, []).append(cat)
        for ot, cats in per_type.items():
            rows.append({"label": label, "expected_oom_type": ot, "category_match": sum(cats) / len(cats)})

    if not rows:
        return
    tab = pd.DataFrame(rows)
    ordered_oom = ["cgroup_oom", "global_oom", "page_alloc_failure", "swap_exhaustion"]
    rag_labels = sorted(set(r["label"] for r in rows if "(RAG)" in r["label"]))
    naive_labels = sorted(set(r["label"] for r in rows if "(Naive)" in r["label"]))
    ordered_labels = rag_labels + naive_labels
    pivot = tab.pivot_table(index="label", columns="expected_oom_type", values="category_match", aggfunc="mean")
    pivot = pivot.reindex(index=ordered_labels, columns=ordered_oom)
    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.45 * len(pivot))))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if (np.isnan(v) or 0.3 < v < 0.8) else "white",
                    fontsize=9)
    ax.set_title("Experiment 2: Category Match per oom_type")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Category Match")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table renderer (compact paper format)
# ---------------------------------------------------------------------------

def _fmt(v, prec=3):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):d}"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "n/a"
        if prec is None:
            return f"{v}"
        return f"{v:.{prec}f}"
    return str(v)


def md_table(df: pd.DataFrame, header_map: dict[str, str], precs: dict[str, int]) -> str:
    cols = list(header_map.keys())
    out = ["| " + " | ".join(header_map[c] for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [_fmt(r[c], precs.get(c, 3)) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------- Exp 1 -------
    exp1 = build_exp1_table()
    exp1.to_csv(OUT_DIR / "exp1_table.csv", index=False, encoding="utf-8-sig")
    plot_exp1(exp1, OUT_DIR / "fig_exp1_recall.png")

    # ------- Exp 2 -------
    exp2 = build_exp2_table()
    exp2.to_csv(OUT_DIR / "exp2_table.csv", index=False, encoding="utf-8-sig")
    plot_exp2_quality(exp2, OUT_DIR / "fig_exp2_quality.png")
    plot_exp2_cost_latency(exp2, OUT_DIR / "fig_exp2_cost_latency.png")
    plot_exp2_quality_vs_latency(exp2, OUT_DIR / "fig_exp2_quality_vs_latency.png")
    plot_exp2_pareto(exp2, OUT_DIR / "fig_exp2_pareto.png")
    plot_exp2_oomtype_heatmap(OUT_DIR / "fig_exp2_oomtype_heatmap.png")

    # ------- Markdown tables -------
    md_parts = ["# Paper-ready consolidated tables", ""]
    md_parts.append("## Table 3. Experiment 1 — Retrieval Recall@K")
    md_parts.append(md_table(
        exp1[["query_mode", "top_k", "n", "recall_mean", "recall_std"]],
        {"query_mode": "Query Mode", "top_k": "Top-K", "n": "N",
         "recall_mean": "Recall (mean)", "recall_std": "Recall (sd)"},
        {"top_k": 0, "n": 0, "recall_mean": 3, "recall_std": 3},
    ))
    md_parts.append("")
    md_parts.append("## Table 4. Experiment 2 — Full per-condition metrics")
    md_parts.append(md_table(
        exp2[[
            "label", "category_match", "evidence_recall", "action_guide_similarity",
            "error_rate", "json_first_try_rate",
            "sec_per_query", "lat_p50", "lat_p95",
            "avg_calls", "avg_input_tokens", "avg_output_tokens",
            "avg_cost_usd", "vram_gb_per_query_mean", "vram_gb_peak",
        ]],
        {"label": "Condition",
         "category_match": "Cat",
         "evidence_recall": "Ev",
         "action_guide_similarity": "Ag",
         "error_rate": "Err",
         "json_first_try_rate": "JSON",
         "sec_per_query": "s/q",
         "lat_p50": "p50",
         "lat_p95": "p95",
         "avg_calls": "Calls",
         "avg_input_tokens": "Tok in",
         "avg_output_tokens": "Tok out",
         "avg_cost_usd": "$/q",
         "vram_gb_per_query_mean": "VRAM/q (GB)",
         "vram_gb_peak": "VRAM peak (GB)"},
        {"category_match": 3, "evidence_recall": 3, "action_guide_similarity": 3,
         "error_rate": 3, "json_first_try_rate": 2,
         "sec_per_query": 2, "lat_p50": 2, "lat_p95": 2,
         "avg_calls": 2, "avg_input_tokens": 0, "avg_output_tokens": 0,
         "avg_cost_usd": 4, "vram_gb_per_query_mean": 2, "vram_gb_peak": 2},
    ))
    (OUT_DIR / "tables.md").write_text("\n".join(md_parts), encoding="utf-8")

    print(f"✓ Exp 1 ({len(exp1)} rows) and Exp 2 ({len(exp2)} rows) tables built.")
    print(f"✓ Outputs under {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
