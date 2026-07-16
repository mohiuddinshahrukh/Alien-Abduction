"""
visualize_results.py

Generate a suite of figures comparing the 6 game variants across the 5 function
signatures, from a clembench results tree (default: results_gpt/).

The primary episode metric is SUCCESS RATE (% of targets solved exactly);
EFFICIENCY is the secondary metric. Each is addressed separately: the heatmap,
grouped-bar, family-rollup and radar views are produced once per metric.
Aggregation is over experiment name, i.e. each (signature, variant) cell, with
all 6 variants (including active_inputs and passive_labeled_pairs_oneshot) kept
as distinct columns.

Outputs are written as PDF into --out-dir (default: scripts/outputs/):
  Per metric (success_rate, efficiency):
    heatmap_<metric>.pdf         signatures x variants, mean metric
    bars_<metric>.pdf            grouped bars: variant clusters per signature
    family_rollup_<metric>.pdf   active vs passive vs oneshot (mean +/- SE)
    radar_<metric>.pdf           per-variant profile across the 5 signatures
  Metric-agnostic:
    heatmap_request_success.pdf  signatures x variants, format adherence
    efficiency_box.pdf           rounds used per variant (boxplot)
    success_vs_efficiency.pdf    success-rate vs efficiency trade-off scatter
    outcome_stacks.pdf           Success / Lose / Abort share per variant
    round_curve.pdf              mean request-success by round, per family

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --results-dir results_gpt
    python scripts/visualize_results.py --out-dir scripts/outputs
"""

import argparse
import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the shared loader whether run as a script or a module.
try:
    from results_loader import (
        CATEGORY_ORDER,
        CATEGORY_SIGNATURE,
        VARIANT_FAMILY,
        VARIANT_LABEL,
        VARIANT_ORDER,
        load_results,
        pivot_metric,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from results_loader import (
        CATEGORY_ORDER,
        CATEGORY_SIGNATURE,
        VARIANT_FAMILY,
        VARIANT_LABEL,
        VARIANT_ORDER,
        load_results,
        pivot_metric,
    )


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

VARIANT_LABELS_ORDERED = [VARIANT_LABEL[v] for v in VARIANT_ORDER]
SIGNATURE_TICKS = [f"{c}\n{CATEGORY_SIGNATURE[c]}" for c in CATEGORY_ORDER]
FAMILY_ORDER = ["active", "passive", "oneshot"]
FAMILY_COLORS = {"active": "#2166ac", "passive": "#f4a582", "oneshot": "#7b3294"}

# One distinct color per variant (6) and one distinct marker per signature (5),
# used by the accuracy-vs-efficiency scatter.
VARIANT_COLORS = {
    "active_inputs": "#1b9e77",
    "active_pair_checks": "#d95f02",
    "passive_examples": "#7570b3",
    "passive_labeled_pairs": "#e7298a",
    "passive_examples_oneshot": "#66a61e",
    "passive_labeled_pairs_oneshot": "#e6ab02",
}
SIGNATURE_MARKERS = {
    "NUMBERS": "o",         # circle
    "TWO_NUMBERS": "s",     # square
    "STRING": "^",          # triangle
    "LIST": "D",            # diamond
    "LOGIC": "P",           # plus (filled)
}


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1-3. Heatmaps
# ---------------------------------------------------------------------------
def heatmap(df, metric, title, out_dir, fname, cmap="viridis", vmin=None, vmax=None, fmt="{:.0f}"):
    table = pivot_metric(df, metric).reindex(index=CATEGORY_ORDER, columns=VARIANT_LABELS_ORDERED)
    data = table.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(VARIANT_LABELS_ORDERED)), VARIANT_LABELS_ORDERED, rotation=30, ha="right")
    ax.set_yticks(range(len(CATEGORY_ORDER)), SIGNATURE_TICKS)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                lum = im.norm(val)
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="white" if lum < 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label=metric.replace("_", " "))
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 4. Grouped bars
# ---------------------------------------------------------------------------
def grouped_bars(df, metric, out_dir, fname):
    table = pivot_metric(df, metric).reindex(index=CATEGORY_ORDER, columns=VARIANT_LABELS_ORDERED)
    n_cat = len(CATEGORY_ORDER)
    n_var = len(VARIANT_LABELS_ORDERED)
    x = np.arange(n_cat)
    width = 0.8 / n_var
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, var in enumerate(VARIANT_LABELS_ORDERED):
        vals = table[var].to_numpy(dtype=float)
        ax.bar(x + (j - n_var / 2 + 0.5) * width, vals, width, label=var, color=cmap(j % 10))
    ax.set_xticks(x, [f"{c}\n{CATEGORY_SIGNATURE[c]}" for c in CATEGORY_ORDER], fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 5. Family rollup (active vs passive vs oneshot)
# ---------------------------------------------------------------------------
def family_rollup(df, metric, out_dir, fname):
    g = df.groupby("family", observed=True)[metric]
    means = g.mean().reindex(FAMILY_ORDER)
    sems = g.sem().reindex(FAMILY_ORDER)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [FAMILY_COLORS[f] for f in FAMILY_ORDER]
    ax.bar(FAMILY_ORDER, means.to_numpy(), yerr=sems.to_numpy(), capsize=6, color=colors, alpha=0.9)
    for i, v in enumerate(means.to_numpy()):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(metric.replace("_", " "))
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 6. Efficiency boxplot (rounds used)
# ---------------------------------------------------------------------------
def efficiency_box(df, out_dir, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    data, labels, colors = [], [], []
    for v in VARIANT_ORDER:
        sub = df[df["variant"] == v]["round_count"].dropna().to_numpy()
        if len(sub):
            data.append(sub)
            labels.append(VARIANT_LABEL[v])
            colors.append(FAMILY_COLORS[VARIANT_FAMILY[v]])
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, tick_labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Rounds used (turns)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], alpha=0.6) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, title="family", fontsize=8)
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 7. Success rate vs efficiency scatter
# ---------------------------------------------------------------------------
def success_vs_efficiency(df, out_dir, fname):
    """
    Color = variant (6), marker = signature (5). One mean point per
    (variant, signature) cell, plotting success rate against efficiency (both
    0-100). Points that land on the same spot are fanned out into a small
    rosette so every symbol stays visible.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # 1. Aggregate to the cell mean (matches the heatmaps). Both axes 0-100.
    agg = (
        df.groupby(["variant", "category"], observed=True)
        .agg(eff=("efficiency", "mean"), acc=("success_rate", "mean"))
        .dropna()
        .reset_index()
    )

    # 2. Group near-identical coordinates and spread them on a small circle.
    agg["kx"] = agg["eff"].round(0)
    agg["ky"] = agg["acc"].round(0)
    radius = 1.2
    xs, ys = [], []
    for _key, grp in agg.groupby(["kx", "ky"], sort=False):
        idx = list(grp.index)
        n = len(idx)
        cx = grp["eff"].mean()
        cy = grp["acc"].mean()
        if n == 1:
            xs.append((idx[0], cx)); ys.append((idx[0], cy))
        else:
            for k, i in enumerate(idx):
                ang = 2 * np.pi * k / n
                xs.append((i, cx + radius * np.cos(ang)))
                ys.append((i, cy + radius * np.sin(ang)))
    x_off = dict(xs)
    y_off = dict(ys)

    # 3. Plot each point with its variant color + signature marker.
    for _, r in agg.iterrows():
        ax.scatter(
            x_off[r.name], y_off[r.name],
            s=95, alpha=0.9,
            color=VARIANT_COLORS[r["variant"]],
            marker=SIGNATURE_MARKERS[r["category"]],
            edgecolor="k", linewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Efficiency")
    ax.set_ylabel("Success rate")
    ax.margins(0.08)

    # Legend 1: variant colors.
    color_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=VARIANT_COLORS[v], markeredgecolor="k",
                   label=VARIANT_LABEL[v])
        for v in VARIANT_ORDER
    ]
    leg1 = ax.legend(handles=color_handles, title="Variant (color)",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.add_artist(leg1)

    # Legend 2: signature markers (shown in neutral gray so shape reads clearly).
    marker_handles = [
        plt.Line2D([0], [0], marker=SIGNATURE_MARKERS[c], linestyle="", markersize=9,
                   markerfacecolor="0.7", markeredgecolor="k",
                   label=f"{c}  {CATEGORY_SIGNATURE[c]}")
        for c in CATEGORY_ORDER
    ]
    ax.legend(handles=marker_handles, title="Signature (shape)",
              loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)

    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 8. Radar chart per variant across signatures
# ---------------------------------------------------------------------------
def radar_variants(df, metric, out_dir, fname):
    table = pivot_metric(df, metric).reindex(index=CATEGORY_ORDER, columns=VARIANT_LABELS_ORDERED)
    angles = np.linspace(0, 2 * np.pi, len(CATEGORY_ORDER), endpoint=False).tolist()
    angles += angles[:1]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    for j, var in enumerate(VARIANT_LABELS_ORDERED):
        vals = table[var].to_numpy(dtype=float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=var, color=cmap(j % 10), linewidth=1.8)
        ax.fill(angles, vals, color=cmap(j % 10), alpha=0.05)
    ax.set_xticks(angles[:-1], CATEGORY_ORDER)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 9. Outcome stacks (Success / Lose / Abort)
# ---------------------------------------------------------------------------
def outcome_stacks(df, out_dir, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = VARIANT_LABELS_ORDERED
    succ, lose, abort = [], [], []
    for v in VARIANT_ORDER:
        sub = df[df["variant"] == v]
        n = len(sub) or 1
        a = sub.get("Aborted")
        s = sub.get("Success")
        aborted = float(a.sum()) if a is not None else 0.0
        success = float(s.sum()) if s is not None else 0.0
        lost = len(sub) - success - aborted
        abort.append(aborted / n)
        succ.append(success / n)
        lose.append(max(lost, 0) / n)
    x = np.arange(len(labels))
    ax.bar(x, succ, label="Success", color="#4daf4a")
    ax.bar(x, lose, bottom=succ, label="Lose", color="#ff7f00")
    ax.bar(x, abort, bottom=np.array(succ) + np.array(lose), label="Aborted", color="#e41a1c")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Share of episodes")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=9)
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# 10. Per-round request-success curve, per family
# ---------------------------------------------------------------------------
def round_curve(df, out_dir, fname):
    # Re-read round scores from each episode's scores.json.
    series = {f: {} for f in FAMILY_ORDER}  # family -> {round_idx: [ratios]}
    for _, row in df.iterrows():
        fam = row["family"]
        if fam not in series:
            continue
        spath = os.path.join(str(row["episode_dir"]), "scores.json")
        if not os.path.exists(spath):
            continue
        with open(spath, "r", encoding="utf-8") as fh:
            rounds = json.load(fh).get("round scores", {})
        for idx, r in rounds.items():
            ratio = r.get("Request Success Ratio")
            if ratio is not None:
                series[fam].setdefault(int(idx), []).append(ratio)

    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in FAMILY_ORDER:
        by_round = series[fam]
        if not by_round:
            continue
        idxs = sorted(by_round)
        means = [np.mean(by_round[i]) for i in idxs]
        ax.plot(idxs, means, marker="o", label=fam, color=FAMILY_COLORS[fam], linewidth=2)
    ax.set_xlabel("Round index")
    ax.set_ylabel("Mean request-success ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="family")
    return _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# The two primary metrics, each addressed separately. (col, label, vmin, vmax, cmap, fmt)
METRIC_SPECS = {
    "success_rate": {"label": "Success rate", "vmin": 0.0, "vmax": 100.0, "cmap": "Greens", "fmt": "{:.0f}"},
    "efficiency": {"label": "Efficiency", "vmin": 0.0, "vmax": 100.0, "cmap": "magma", "fmt": "{:.0f}"},
    "accuracy": {"label": "Accuracy", "vmin": 0.0, "vmax": 100.0, "cmap": "viridis", "fmt": "{:.0f}"},
}


def generate_all(results_dir: str, out_dir: str,
                 metrics: List[str] = ("success_rate", "efficiency")) -> List[str]:
    df = load_results(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save the flattened table alongside the figures for reference.
    df.drop(columns=["episode_dir"]).to_csv(os.path.join(out_dir, "episodes.csv"), index=False)

    written: List[str] = []

    # --- Per-metric figures: success rate and efficiency handled separately --
    for metric in metrics:
        if metric not in df.columns:
            print(f"  (skipping {metric!r}: not present in results)")
            continue
        spec = METRIC_SPECS.get(metric, {"label": metric, "vmin": None, "vmax": None,
                                         "cmap": "viridis", "fmt": "{:.2f}"})
        written.append(heatmap(
            df, metric, f"{spec['label']} by signature x variant", out_dir,
            f"heatmap_{metric}.pdf", cmap=spec["cmap"],
            vmin=spec["vmin"], vmax=spec["vmax"], fmt=spec["fmt"]))
        written.append(grouped_bars(df, metric, out_dir, f"bars_{metric}.pdf"))
        written.append(family_rollup(df, metric, out_dir, f"family_rollup_{metric}.pdf"))
        written.append(radar_variants(df, metric, out_dir, f"radar_{metric}.pdf"))

    # --- Metric-agnostic figures -------------------------------------------
    if "Request_Success_Ratio" in df:
        written.append(heatmap(df, "Request_Success_Ratio", "Format adherence (request success)",
                               out_dir, "heatmap_request_success.pdf", cmap="Blues",
                               vmin=0, vmax=1, fmt="{:.2f}"))
    if "round_count" in df:
        written.append(efficiency_box(df, out_dir, "efficiency_box.pdf"))
    if "success_rate" in df and "efficiency" in df:
        written.append(success_vs_efficiency(df, out_dir, "success_vs_efficiency.pdf"))
    written.append(outcome_stacks(df, out_dir, "outcome_stacks.pdf"))
    written.append(round_curve(df, out_dir, "round_curve.pdf"))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize function_detective results.")
    parser.add_argument("--results-dir", default="results_gpt")
    parser.add_argument("--out-dir", default="scripts/outputs")
    parser.add_argument("--metrics", nargs="*", default=["success_rate", "efficiency"],
                        help="Episode metrics to produce per-metric figures for.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = generate_all(args.results_dir, args.out_dir, metrics=args.metrics)
    print(f"Wrote {len(paths)} figures (PDF) to {args.out_dir}:")
    for p in paths:
        print(f"  {os.path.basename(p)}")
