#!/usr/bin/env python3
"""
Bar-chart counterpart to plot_efficiency_by_variant_grid.py, plotting the
new TBU (Turn Budget Usage) score instead of the game's own "efficiency"
field -- see tbu_analysis.py's module docstring for the metric definition
(tbu = (turns_used + 1) / 15, i.e. the fraction of the fixed 15-turn budget
a model actually used; 1.0 means it ran the full budget). Same per-model
grid layout and Success/Loss split as that script, but each (model,
variant) cell is drawn as two grouped bars (Success, Loss) instead of two
line-points, since there's no ordering/trend along the variant axis this
chart is trying to show -- just a direct value comparison per variant.

Reads each model's already-computed <results_parent>/<model>/analysis/
tbu_summary.json via find_tbu_by_model_and_mode -- not recomputed here.
Single-turn ("_oneshot") modes never appear in that file at all (see
tbu_analysis.py -- turns_used is trivially 0 there by construction, so
there's no real turn-budget decision to measure), so every subplot only
ever has the four multi-turn variants on its x-axis.

Output:
    tbu_by_variant_grid.png / .pdf   1xN subplots, one per model (ordered
                                        per display_names.sort_models). Each
                                        subplot: x-axis = variant (per
                                        display_names.MODE_ORDER with
                                        single-turn modes filtered out,
                                        display_names.display_mode_overall
                                        labels), two bars per variant --
                                        Success, Loss mean TBU score -- with
                                        value labels, a bar omitted for a
                                        variant with zero instances of that
                                        outcome. One shared y-axis (0-1,
                                        "TBU Score", common scale across
                                        every model) labeled only on the
                                        first subplot; x-axis variant labels
                                        repeat on every subplot. A single
                                        figure-level legend distinguishes
                                        Success vs Loss.

Usage:
    python plot_tbu_by_variant_grid.py <results_parent> [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from display_names import (
    INK_PRIMARY,
    INK_SECONDARY,
    LOSS_COLOR,
    MODE_FAMILY,
    SUCCESS_COLOR,
    SURFACE,
    display_mode_overall,
    display_model_name,
    iter_model_dirs,
    sort_models,
    style_axes,
)

NON_ONESHOT_MODES = [m for m in MODE_FAMILY if MODE_FAMILY[m] != "single_turn"]
# Width < 2x the offset below so the Success/Loss pair has a visible gap
# between them -- at width==2*offset the bars touch edge-to-edge, which
# left no room for a bar_label sitting on a short bar not to be clipped by
# a much taller neighbor right next to it.
BAR_WIDTH = 0.30
BAR_OFFSET = 0.19


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def find_tbu_by_model_and_mode(results_parent: Path):
    """Returns {model: {mode: {"success": mean_tbu, "loss": mean_tbu}}}."""
    result = {}
    for model_dir in iter_model_dirs(results_parent):
        tbu_path = model_dir / "analysis" / "tbu_summary.json"
        if not tbu_path.exists():
            print(f"No {tbu_path} -- skipping {model_dir.name} (run tbu_analysis.py first?)")
            continue
        by_model = _load(tbu_path)["by_model"]
        for model, entry in by_model.items():
            result[model] = {
                mode: {outcome: stats["mean_tbu"] for outcome, stats in outcomes.items()}
                for mode, outcomes in entry["by_mode"].items()
            }
    return result


def plot_tbu_by_variant_grid(tbu_by_model, out_path_base):
    if not tbu_by_model:
        print(f"No TBU data found -- skipping {out_path_base.name}")
        return

    models = sort_models(tbu_by_model.keys())
    modes = [m for m in NON_ONESHOT_MODES if any(m in per_mode for per_mode in tbu_by_model.values())]
    labels = [display_mode_overall(m) for m in modes]
    x = list(range(len(modes)))

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 5.0), facecolor=SURFACE, sharey=True)
    if len(models) == 1:
        axes = [axes]

    handles, plot_labels = None, None
    for ax, model in zip(axes, models):
        tbu_by_mode = tbu_by_model[model]
        for offset, outcome, color, label in (
            (-BAR_OFFSET, "success", SUCCESS_COLOR, "Success"),
            (BAR_OFFSET, "loss", LOSS_COLOR, "Loss"),
        ):
            xs, ys = [], []
            for idx, mode in zip(x, modes):
                value = tbu_by_mode.get(mode, {}).get(outcome)
                if value is None:
                    continue
                xs.append(idx + offset)
                ys.append(value)
            if not xs:
                continue
            bars = ax.bar(xs, ys, width=BAR_WIDTH, color=color, label=label, zorder=3)
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8.5, color=color, fontweight="bold")
        if handles is None:
            handles, plot_labels = ax.get_legend_handles_labels()

        style_axes(ax, "TBU Score" if ax is axes[0] else "", xlabel="Mode", fontsize=14)
        ax.set_title(display_model_name(model), color=INK_PRIMARY, fontsize=14, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
        ax.set_xlim(-0.6, len(modes) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    legend = fig.legend(
        handles, plot_labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=14,
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
    tbu_by_model = find_tbu_by_model_and_mode(results_parent)
    plot_tbu_by_variant_grid(tbu_by_model, out_dir / "tbu_by_variant_grid.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Parent directory containing one subfolder per model, each already analyzed "
        "(<results_parent>/<model>/analysis/tbu_summary.json)",
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
