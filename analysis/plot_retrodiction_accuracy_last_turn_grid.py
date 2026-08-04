#!/usr/bin/env python3
"""
Retrodiction accuracy AT THE LAST TURN (the moment of decision -- each
instance's final valid turn before solving/giving up), one subplot per
model, x-axis = variant, y-axis = retrodiction accuracy (0-1), two lines --
Success, Loss. This is a different population/granularity from
plot_retrodiction_accuracy_by_variant_grid.py (which averages every turn>=2
across an instance's whole trajectory) and from
plot_retrodiction_accuracy_by_turn_grid.py (which plots the turn-by-turn
trajectory itself) -- here only the single last turn of each instance
counts, per explicit request.

Reads each model's already-computed <results_parent>/<model>/analysis/
last_turn_heatmap.json (from last_turn_analysis.py in this same folder) --
<model>.correctness_by_mode[mode][outcome] = {value, source}, not
recomputed here. "source" is "retrodiction" for the four multi-turn
variants (mean retrodiction_accuracy_corrected at that outcome's own last
turn -- genuinely separate Success/Loss numbers) and "assumed_from_outcome"
for the two single-turn ("_oneshot") variants, where real retrodiction is
undefined by construction (a definitional 1.0 for Success / 0.0 for Loss,
not a measured value) -- those two are filtered out here, same reasoning
as every other retrodiction chart in this folder excluding single-turn
modes.

Output:
    retrodiction_accuracy_last_turn_grid.png / .pdf   1xN subplots, one per
                                        model (ordered per display_names.
                                        sort_models). Each subplot: x-axis =
                                        variant (per display_names.
                                        sort_modes order, display_names.
                                        display_mode_overall labels), two
                                        lines -- Success, Loss last-turn
                                        retrodiction accuracy -- with
                                        markers and value labels, a
                                        line/marker omitted for a variant
                                        with no "retrodiction"-sourced value
                                        for that outcome. One shared y-axis
                                        (0-1, common scale across every
                                        model) labeled only on the first
                                        subplot; x-axis variant labels
                                        repeat on every subplot. A single
                                        figure-level legend distinguishes
                                        Success vs Loss.
    retrodiction_accuracy_last_turn_grid_success.png / .pdf   same chart,
                                        Success line only (no Loss), same
                                        axes/scale/layout.
    retrodiction_accuracy_last_turn_grid_bar.png / .pdf   same data, as
                                        grouped bars (Success, Loss) per
                                        variant instead of lines -- see
                                        plot_retrodiction_accuracy_last_turn_grid_bar().
    retrodiction_accuracy_last_turn_grid_bar_success.png / .pdf   same bar
                                        chart, Success only.

Usage:
    python plot_retrodiction_accuracy_last_turn_grid.py <results_parent> [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from display_names import (
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    display_mode_overall,
    display_model_name,
    iter_model_dirs,
    sort_modes,
    sort_models,
    style_axes,
)

SUCCESS_COLOR = PALETTE[0]  # teal
LOSS_COLOR = PALETTE[1]  # coral


def find_last_turn_retrodiction_by_model_and_mode(results_parent: Path):
    """Returns {model: {mode: {"success": value_or_None, "loss": value_or_None}}},
    "retrodiction"-sourced values only (excludes the two oneshot variants'
    definitional 1.0/0.0 placeholders)."""
    result = {}
    for model_dir in iter_model_dirs(results_parent):
        heatmap_path = model_dir / "analysis" / "last_turn_heatmap.json"
        if not heatmap_path.exists():
            print(f"No {heatmap_path} -- skipping {model_dir.name} (run last_turn_analysis.py first?)")
            continue
        with open(heatmap_path) as f:
            data = json.load(f)
        for model, entry in data.items():
            by_mode = {}
            for mode, outcomes in entry.get("correctness_by_mode", {}).items():
                values = {
                    outcome: info["value"]
                    for outcome, info in outcomes.items()
                    if info.get("source") == "retrodiction"
                }
                if values:
                    by_mode[mode] = values
            if by_mode:
                result[model] = by_mode
    return result


def plot_retrodiction_accuracy_last_turn_grid(accuracy_by_model, out_path_base, outcomes=None):
    """outcomes: subset of [("success", SUCCESS_COLOR, "Success"), ("loss", LOSS_COLOR, "Loss")]
    to plot -- defaults to both (the combined chart); pass just the Success
    tuple for a Success-only figure, etc."""
    if not accuracy_by_model:
        print(f"No last-turn retrodiction data found -- skipping {out_path_base.name}")
        return
    if outcomes is None:
        outcomes = [("success", SUCCESS_COLOR, "Success"), ("loss", LOSS_COLOR, "Loss")]

    models = sort_models(accuracy_by_model.keys())
    all_modes = {mode for per_mode in accuracy_by_model.values() for mode in per_mode}
    modes = sort_modes(all_modes)
    labels = [display_mode_overall(m) for m in modes]
    x = list(range(len(modes)))

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 5.0), facecolor=SURFACE, sharey=True)
    if len(models) == 1:
        axes = [axes]

    handles, plot_labels = None, None
    for ax, model in zip(axes, models):
        accuracy_by_mode = accuracy_by_model[model]
        for outcome, color, label in outcomes:
            xs, ys = [], []
            for idx, mode in zip(x, modes):
                value = accuracy_by_mode.get(mode, {}).get(outcome)
                if value is None:
                    continue
                xs.append(idx)
                ys.append(value)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=label, zorder=3)
            for xi, yi in zip(xs, ys):
                ax.annotate(
                    f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9, color=color, fontweight="bold", zorder=4,
                )
        if handles is None:
            handles, plot_labels = ax.get_legend_handles_labels()

        style_axes(ax, "Retrodiction accuracy (last turn)" if ax is axes[0] else "", xlabel="Variant", fontsize=15)
        ax.set_title(display_model_name(model), color=INK_PRIMARY, fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=13)
        ax.set_xlim(-0.4, len(modes) - 0.6)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    legend = fig.legend(
        handles, plot_labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(outcomes),
        frameon=False, fontsize=13,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    png_path = out_path_base.with_suffix(".png")
    pdf_path = out_path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


# Width < 2x the offset so the Success/Loss pair has a visible gap between
# them -- at width==2*offset the bars touch edge-to-edge, which leaves no
# room for a bar_label on a short bar not to be clipped by a much taller
# neighbor right next to it (same fix plot_tbu_by_variant_grid.py applied).
# Gap needs to be wide enough that it also survives the worst case this
# chart has and that one didn't -- Success and Loss landing on the EXACT
# same value (e.g. GPT5.4-mini's Active-Output, both 1.00): same y, so the
# two labels sit on the same line and collide head-on regardless of bar
# height, not just when one bar is short next to a tall one.
BAR_WIDTH = 0.26
BAR_OFFSET = 0.21


def plot_retrodiction_accuracy_last_turn_grid_bar(accuracy_by_model, out_path_base, outcomes=None):
    """Bar-chart version of plot_retrodiction_accuracy_last_turn_grid: same
    data (find_last_turn_retrodiction_by_model_and_mode), same per-model
    subplot grid, but each variant gets two grouped bars (Success, Loss)
    instead of two line-points -- there's no ordering/trend along the
    variant axis to show, just a direct per-variant value comparison.
    outcomes: same meaning as the line-chart version's parameter."""
    if not accuracy_by_model:
        print(f"No last-turn retrodiction data found -- skipping {out_path_base.name}")
        return
    if outcomes is None:
        outcomes = [("success", SUCCESS_COLOR, "Success"), ("loss", LOSS_COLOR, "Loss")]

    models = sort_models(accuracy_by_model.keys())
    all_modes = {mode for per_mode in accuracy_by_model.values() for mode in per_mode}
    modes = sort_modes(all_modes)
    labels = [display_mode_overall(m) for m in modes]
    x = list(range(len(modes)))
    offsets = [0] if len(outcomes) == 1 else [-BAR_OFFSET, BAR_OFFSET]
    bar_width = 0.5 if len(outcomes) == 1 else BAR_WIDTH

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 5.0), facecolor=SURFACE, sharey=True)
    if len(models) == 1:
        axes = [axes]

    handles, plot_labels = None, None
    for ax, model in zip(axes, models):
        accuracy_by_mode = accuracy_by_model[model]
        for offset, (outcome, color, label) in zip(offsets, outcomes):
            xs, ys = [], []
            for idx, mode in zip(x, modes):
                value = accuracy_by_mode.get(mode, {}).get(outcome)
                if value is None:
                    continue
                xs.append(idx + offset)
                ys.append(value)
            if not xs:
                continue
            bars = ax.bar(xs, ys, width=bar_width, color=color, label=label, zorder=3)
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, color=color, fontweight="bold")
        if handles is None:
            handles, plot_labels = ax.get_legend_handles_labels()

        style_axes(
            ax, "Retrodiction accuracy (last turn)" if ax is axes[0] else "", xlabel="Variant", fontsize=15,
        )
        ax.set_title(display_model_name(model), color=INK_PRIMARY, fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=13)
        ax.set_xlim(-0.6, len(modes) - 0.4)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    legend = fig.legend(
        handles, plot_labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(outcomes),
        frameon=False, fontsize=13,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    png_path = out_path_base.with_suffix(".png")
    pdf_path = out_path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def run(results_parent: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    accuracy_by_model = find_last_turn_retrodiction_by_model_and_mode(results_parent)
    plot_retrodiction_accuracy_last_turn_grid(accuracy_by_model, out_dir / "retrodiction_accuracy_last_turn_grid.png")
    plot_retrodiction_accuracy_last_turn_grid(
        accuracy_by_model, out_dir / "retrodiction_accuracy_last_turn_grid_success.png",
        outcomes=[("success", SUCCESS_COLOR, "Success")],
    )
    plot_retrodiction_accuracy_last_turn_grid_bar(
        accuracy_by_model, out_dir / "retrodiction_accuracy_last_turn_grid_bar.png",
    )
    plot_retrodiction_accuracy_last_turn_grid_bar(
        accuracy_by_model, out_dir / "retrodiction_accuracy_last_turn_grid_bar_success.png",
        outcomes=[("success", SUCCESS_COLOR, "Success")],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Parent directory containing one subfolder per model, each already analyzed "
        "(<results_parent>/<model>/analysis/last_turn_heatmap.json)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNG/PDF (default: <results_parent>/paper_plots)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots"
    run(results_parent, out_dir)


if __name__ == "__main__":
    main()
