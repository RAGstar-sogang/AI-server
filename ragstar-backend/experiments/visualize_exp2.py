"""
Visualize model-scaling experiment results.

Reads:
  data/exp_results/scaling_per_log.csv
  data/exp_results/scaling_per_condition.csv

Writes plots to data/exp_results/scaling_plots/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "data" / "exp_results"
PLOT_DIR = RESULT_DIR / "scaling_plots"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _label_order(df: pd.DataFrame) -> list[str]:
    rag = df[df["mode"] == "rag"].copy()
    naive = df[df["mode"] == "naive"].copy()
    rag = rag.sort_values(by="label")
    naive = naive.sort_values(by="label")
    return list(rag["label"]) + list(naive["label"])


def plot_accuracy_bars(df: pd.DataFrame, path: Path) -> None:
    metrics = ["category_match", "evidence_recall", "action_guide_similarity"]
    labels = _label_order(df)
    df = df.set_index("label").loc[labels].reset_index()
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df)), 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, df[m].astype(float), w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy metrics per condition")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_naive_vs_rag(df: pd.DataFrame, path: Path) -> None:
    """For models that have both modes, show paired bars."""
    both = df.groupby("model")["mode"].nunique()
    paired_models = [m for m, n in both.items() if n >= 2]
    if not paired_models:
        return
    metrics = ["category_match", "evidence_recall", "action_guide_similarity"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(paired_models))
    w = 0.35
    for ax, m in zip(axes, metrics):
        naive_vals = []
        rag_vals = []
        for mdl in paired_models:
            naive_vals.append(df[(df["model"] == mdl) & (df["mode"] == "naive")][m].mean())
            rag_vals.append(df[(df["model"] == mdl) & (df["mode"] == "rag")][m].mean())
        ax.bar(x - w / 2, naive_vals, w, label="Naive")
        ax.bar(x + w / 2, rag_vals, w, label="RAG")
        ax.set_xticks(x)
        ax.set_xticklabels(paired_models, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(m)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Naive vs RAG, per shared model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_latency_vs_accuracy(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {"rag": "tab:blue", "naive": "tab:orange"}
    spq = _resolve_sec_per_query(df)
    for (_, r), x in zip(df.iterrows(), spq):
        c = color_map.get(r["mode"], "gray")
        ax.scatter(x, r["category_match"], c=c, s=80, edgecolor="black", linewidth=0.5)
        ax.annotate(r["label"], (x, r["category_match"]),
                    xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean inference time per log (sec)")
    ax.set_ylabel("Category Match")
    ax.set_ylim(0, 1.05)
    ax.set_title("Latency vs Accuracy (Pareto view)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=9, label="RAG"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=9, label="Naive"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _resolve_sec_per_query(df: pd.DataFrame) -> pd.Series:
    """Backwards-compat shim: prefer `sec_per_query`; fall back to legacy
    `lat_mean` / `throughput_qps_single` for older CSVs."""
    if "sec_per_query" in df.columns:
        return df["sec_per_query"].astype(float)
    if "lat_mean" in df.columns:
        return df["lat_mean"].astype(float)
    qps = df.get("throughput_qps_single")
    if qps is None:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return (1.0 / qps.astype(float)).replace([np.inf, -np.inf], float("nan"))


def _resolve_vram_x(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Prefer per-query measured mean (most meaningful for cost/comparison);
    fall back to condition-wide peak, then legacy mean, then estimate."""
    if "vram_gb_per_query_mean" in df.columns and df["vram_gb_per_query_mean"].notna().any():
        return df["vram_gb_per_query_mean"].astype(float), "Mean VRAM per query (GB, measured)"
    if "vram_gb_peak" in df.columns and df["vram_gb_peak"].notna().any():
        return df["vram_gb_peak"].astype(float), "Peak VRAM (GB, measured)"
    if "vram_gb_mean" in df.columns and df["vram_gb_mean"].notna().any():
        return df["vram_gb_mean"].astype(float), "Mean VRAM (GB, measured)"
    if "vram_gb_est" in df.columns:
        return df["vram_gb_est"].astype(float), "Estimated VRAM (GB, config ceiling)"
    return pd.Series([float("nan")] * len(df), index=df.index), "VRAM (GB)"


def plot_vram_vs_throughput(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["mode"] == "rag"].copy()
    if sub.empty:
        return
    y = _resolve_sec_per_query(sub)
    x, x_label = _resolve_vram_x(sub)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=90, c="tab:purple", edgecolor="black")
    for (xi, yi, lab) in zip(x, y, sub["label"]):
        ax.annotate(lab, (xi, yi),
                    xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sec per query (mean latency, lower is better)")
    ax.set_title("VRAM vs latency-per-query (RAG, single stream)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cost_per_query(df: pd.DataFrame, path: Path) -> None:
    labels = _label_order(df)
    df = df.set_index("label").loc[labels].reset_index()
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(df)), 4.5))
    ax.bar(df["label"], df["avg_cost_usd"], color="tab:green")
    ax.set_ylabel("Estimated cost per query (USD)")
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_title("Cost per query (local: power×time; remote: token price)")
    for i, v in enumerate(df["avg_cost_usd"]):
        ax.text(i, v, f"${v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_json_validity(df: pd.DataFrame, path: Path) -> None:
    """3-state stacked bar per condition.

    States:
      first_try_valid  raw LLM output parsed as strict JSON on first try
      extracted        raw was not strict JSON, but our extractor recovered it
      failed           extractor also failed -> condition's error_rate

    `extracted = 1 - first_try - error_rate`. (`json_repair_rate` from the CSV
    only counts the legacy brace-trim path and is misleading for outputs with
    CoT preambles or multiple JSON blocks; those are still recovered by
    `_extract_balanced_json_objects` but show up as `repair=0`.)
    """
    labels = _label_order(df)
    df = df.set_index("label").loc[labels].reset_index()
    first = df["json_first_try_rate"].astype(float)
    failed = df["error_rate"].astype(float)
    extracted = (1.0 - first - failed).clip(lower=0)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(df)), 4.5))
    ax.bar(x, first, label="first-try valid", color="#2ca02c")
    ax.bar(x, extracted, bottom=first, label="recovered via extraction", color="#ff7f0e")
    ax.bar(x, failed, bottom=first + extracted, label="failed", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("JSON output validity (final response)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_faithfulness(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["mode"] == "rag"].dropna(subset=["faithfulness_mean"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values(by="label").reset_index(drop=True)
    labels = sub["label"].tolist()
    vals = sub["faithfulness_mean"].astype(float).tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.5))
    ax.bar(x, vals, color="#2ca02c", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(0.1, max(vals) * 1.4 if vals else 0.1))
    ax.set_ylabel("Faithfulness (bigram overlap, mean over logs)")
    ax.set_title("RAG faithfulness — generated evidence vs retrieved chunks")
    for xx, v in zip(x, vals):
        ax.text(xx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_exp1_retrieval(exp1_dir: Path, path: Path) -> None:
    """Render Recall@K bar chart for Raw vs Structured retrieval (Exp 1)."""
    pairs = [("raw", 3), ("raw", 5), ("parsed", 3), ("parsed", 5)]
    data = []
    for mode, k in pairs:
        f = exp1_dir / f"exp1_retrieval_{mode}_top{k}_results.csv"
        if not f.exists():
            print(f"⚠ missing exp1 csv: {f}")
            continue
        df = pd.read_csv(f)
        mean_recall = float(df["recall"].mean()) if "recall" in df.columns else 0.0
        data.append({"mode": mode, "top_k": k, "recall": mean_recall, "n": len(df)})
    if not data:
        return
    dfx = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    modes = ["raw", "parsed"]
    ks = sorted(dfx["top_k"].unique())
    x = np.arange(len(ks))
    w = 0.35
    for i, m in enumerate(modes):
        sub = dfx[dfx["mode"] == m].set_index("top_k")
        vals = [float(sub.loc[k, "recall"]) if k in sub.index else 0.0 for k in ks]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label="Raw" if m == "raw" else "Structured")
        for xx, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylim(0, max(0.5, dfx["recall"].max() * 1.4))
    ax.set_ylabel("Mean Recall (n=56)")
    ax.set_title("Experiment 1: Retrieval quality — Raw vs Structured query")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


_OOM_TYPE_ORDER = ["cgroup_oom", "global_oom", "page_alloc_failure", "swap_exhaustion"]


def _per_oomtype_table(per_log_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = (
        per_log_df.groupby(["label", "expected_oom_type"])[metric]
        .mean()
        .unstack("expected_oom_type")
        .reindex(columns=_OOM_TYPE_ORDER)
    )
    return g


def plot_per_oomtype_heatmap(per_log_df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    pivot = _per_oomtype_table(per_log_df, metric)
    rag_labels = sorted(per_log_df[per_log_df["mode"] == "rag"]["label"].unique())
    naive_labels = sorted(per_log_df[per_log_df["mode"] == "naive"]["label"].unique())
    pivot = pivot.reindex(rag_labels + naive_labels)
    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.45 * len(pivot))))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if (np.isnan(v) or 0.3 < v < 0.8) else "white",
                    fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_oomtype_metric_bars(per_log_df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    rag_labels = sorted(per_log_df[per_log_df["mode"] == "rag"]["label"].unique())
    naive_labels = sorted(per_log_df[per_log_df["mode"] == "naive"]["label"].unique())
    ordered = rag_labels + naive_labels
    pivot = _per_oomtype_table(per_log_df, metric).reindex(ordered)

    n_types = len(_OOM_TYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(4.3 * n_types, 4.8), sharey=True)
    colors = ["tab:blue" if lab in rag_labels else "tab:orange" for lab in ordered]
    x = np.arange(len(ordered))
    for ax, ot in zip(axes, _OOM_TYPE_ORDER):
        vals = pivot[ot].astype(float).fillna(0).tolist()
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=8)
        n_logs = int((per_log_df["expected_oom_type"] == ot).sum() / max(per_log_df["label"].nunique(), 1))
        ax.set_title(f"{ot}  (n≈{n_logs}/cond)")
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel(metric)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="tab:blue", label="RAG"), Patch(facecolor="tab:orange", label="Naive")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Render a small pandas DF as a GitHub-flavored markdown table."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if pd.isna(v) else float_fmt.format(float(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _drop_excluded(df: pd.DataFrame, exclude_models_modes: list[tuple[str, str]]) -> pd.DataFrame:
    if not exclude_models_modes:
        return df
    mask = pd.Series(True, index=df.index)
    for (m, md) in exclude_models_modes:
        mask &= ~((df["model"] == m) & (df["mode"] == md))
    return df[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cond-csv", type=str,
                        default=str(RESULT_DIR / "scaling_per_condition.csv"))
    parser.add_argument("--per-log-csv", type=str,
                        default=str(RESULT_DIR / "scaling_per_log.csv"))
    parser.add_argument("--exp1-dir", type=str,
                        default=str(RESULT_DIR / "old2"),
                        help="Directory containing exp1_retrieval_*_topN_results.csv files.")
    parser.add_argument("--out", type=str, default=str(PLOT_DIR))
    parser.add_argument("--exclude", nargs="*", default=["qwen3.5-2b:naive"],
                        help='Conditions to exclude, format "model:mode" '
                             '(default: qwen3.5-2b:naive; pass empty to disable).')
    args = parser.parse_args()

    exclude_pairs = []
    for tok in args.exclude or []:
        if ":" in tok:
            m, md = tok.split(":", 1)
            exclude_pairs.append((m.strip(), md.strip()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables_md = ["# Experiment 2 — visualization tables", ""]
    if exclude_pairs:
        tables_md.append(f"_Excluded conditions: {exclude_pairs}_\n")

    # Exp 1 (retrieval) — from cached results
    exp1_dir = Path(args.exp1_dir)
    if exp1_dir.exists():
        plot_exp1_retrieval(exp1_dir, out / "00_exp1_retrieval_recall.png")
    else:
        print(f"⚠ exp1 source dir not found: {exp1_dir}")

    # Exp 2 (scaling) — from current run
    per_cond_path = Path(args.per_cond_csv)
    if not per_cond_path.exists():
        print(f"⚠ scaling per-condition csv not found yet: {per_cond_path}")
        return
    df_raw = pd.read_csv(per_cond_path)
    if df_raw.empty:
        print(f"⚠ empty condition table: {per_cond_path}")
        return
    df = _drop_excluded(df_raw, exclude_pairs)

    plot_accuracy_bars(df, out / "01_accuracy_bars.png")
    tables_md += ["## 01 Accuracy per condition",
                  _md_table(df[["label", "n", "category_match", "evidence_recall",
                                "action_guide_similarity"]].sort_values("label")),
                  ""]

    plot_naive_vs_rag(df, out / "02_naive_vs_rag.png")
    paired = (df.groupby("model")["mode"].nunique() >= 2)
    paired_models = paired[paired].index.tolist()
    if paired_models:
        sub = df[df["model"].isin(paired_models)].pivot_table(
            index="model", columns="mode",
            values=["category_match", "evidence_recall", "action_guide_similarity"],
        )
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        tables_md += ["## 02 Naive vs RAG (paired models)",
                      _md_table(sub.reset_index()),
                      ""]

    plot_latency_vs_accuracy(df, out / "03_latency_vs_accuracy.png")
    df_lat = df.copy()
    df_lat["sec_per_query"] = _resolve_sec_per_query(df_lat)
    lat_cols = ["label", "sec_per_query", "lat_p50", "lat_p95", "category_match"]
    tables_md += ["## 03 Latency vs accuracy",
                  _md_table(df_lat[lat_cols].sort_values("sec_per_query")),
                  ""]

    plot_vram_vs_throughput(df, out / "04_vram_vs_latency.png")
    rag_df = df[df["mode"] == "rag"].copy()
    if not rag_df.empty:
        rag_df["sec_per_query"] = _resolve_sec_per_query(rag_df)
        cols = ["label"]
        for c in ("vram_gb_per_query_mean", "vram_gb_peak", "vram_gb_mean", "vram_gb_est"):
            if c in rag_df.columns:
                cols.append(c)
        cols += ["sec_per_query"]
        sort_key = next(
            (c for c in ("vram_gb_per_query_mean", "vram_gb_peak", "vram_gb_est") if c in rag_df.columns),
            "sec_per_query",
        )
        tables_md += ["## 04 VRAM vs latency (RAG)",
                      _md_table(rag_df[cols].sort_values(sort_key),
                                float_fmt="{:.3f}"),
                      ""]

    plot_cost_per_query(df, out / "05_cost_per_query.png")
    df_cost = df.copy()
    df_cost["sec_per_query"] = _resolve_sec_per_query(df_cost)
    tables_md += ["## 05 Cost per query",
                  _md_table(df_cost[["label", "avg_cost_usd", "avg_input_tokens",
                                     "avg_output_tokens", "sec_per_query"]].sort_values("avg_cost_usd"),
                            float_fmt="{:.4f}"),
                  ""]

    plot_json_validity(df, out / "06_json_validity.png")
    json_tab = df[["label", "json_first_try_rate", "error_rate"]].copy()
    json_tab["extracted_recovery"] = (1.0 - json_tab["json_first_try_rate"] - json_tab["error_rate"]).clip(lower=0)
    tables_md += ["## 06 JSON validity",
                  _md_table(json_tab.sort_values("label")),
                  ""]

    # Per oom_type breakdown — requires per-log CSV
    per_log_path = Path(args.per_log_csv)
    if per_log_path.exists():
        per_log_df_raw = pd.read_csv(per_log_path)
        per_log_df = _drop_excluded(per_log_df_raw, exclude_pairs)
        plot_per_oomtype_heatmap(per_log_df, "category_match",
                                 "Category Match per oom_type",
                                 out / "08_per_oomtype_cat_heatmap.png")
        plot_per_oomtype_heatmap(per_log_df, "evidence_recall",
                                 "Evidence Recall per oom_type",
                                 out / "08b_per_oomtype_ev_heatmap.png")
        plot_per_oomtype_heatmap(per_log_df, "action_guide_similarity",
                                 "Action Guide Similarity per oom_type",
                                 out / "08c_per_oomtype_ag_heatmap.png")
        plot_per_oomtype_metric_bars(per_log_df, "category_match",
                                     "Category match per oom_type",
                                     out / "09_per_oomtype_cat_bars.png")
        plot_per_oomtype_metric_bars(per_log_df, "evidence_recall",
                                     "Evidence recall per oom_type",
                                     out / "10_per_oomtype_ev_bars.png")
        plot_per_oomtype_metric_bars(per_log_df, "action_guide_similarity",
                                     "Action guide similarity per oom_type",
                                     out / "11_per_oomtype_ag_bars.png")
        for metric, title in [
            ("category_match", "08 Category match per oom_type"),
            ("evidence_recall", "10 Evidence recall per oom_type"),
            ("action_guide_similarity", "11 Action guide similarity per oom_type"),
        ]:
            pivot = (per_log_df.groupby(["label", "expected_oom_type"])[metric]
                     .mean().unstack("expected_oom_type")
                     .reindex(columns=_OOM_TYPE_ORDER).reset_index())
            tables_md += [f"## {title}", _md_table(pivot), ""]
    else:
        print(f"⚠ per-log csv not found: {per_log_path} (skipping per-oom_type plots)")

    tables_path = out / "tables.md"
    tables_path.write_text("\n".join(tables_md), encoding="utf-8")
    print(f"✓ wrote plots to {out}")
    print(f"✓ wrote tables to {tables_path}")


if __name__ == "__main__":
    main()
