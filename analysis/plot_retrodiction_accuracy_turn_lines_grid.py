#!/usr/bin/env python3
"""
Retrodiction accuracy across turns, one subplot per model (1x5 row),
x-axis = turn number, y-axis = mean retrodiction accuracy (0-1), one line
per variant (Active-Output/Verdict, Passive-Output/Verdict). This is the
line-chart sibling of plot_retrodiction_accuracy_by_turn_grid.py, which
puts variant on the x-axis and turn on the y-axis instead (accuracy shown
as a color-scaled scatter there, since both axes are otherwise taken) --
kept as a separate file rather than replacing that one, since both
orientations have been asked for and are useful for different reads (this
one for "does accuracy trend up/down as the game goes on", that one for
"which turn range does each variant actually run in").

Reads the same data as plot_retrodiction_accuracy_by_turn_grid.py --
find_retrodiction_accuracy_by_turn (retrodiction_aggregates.json,
Success-instances-only, (model, mode, turn) -> mean_retrodiction_accuracy)
and find_overall_median_success_turns (turn_level_summary.json's
overall.success.median_turns, every variant pooled) -- both imported from
that module, not reimplemented here. Single-turn ("_oneshot") modes never
appear (a turn-1-only game has no t>=2 to retrodict), so every subplot only
ever has the four multi-turn variants.

Output:
    retrodiction_accuracy_turn_lines_grid.png / .pdf   1x5 subplots, one
                                        per model (ordered per
                                        display_names.sort_models). Each
                                        subplot: x-axis = turn number,
                                        y-axis = mean retrodiction accuracy
                                        (0-1), one line per variant
                                        (display_names.MODE_COLORS,
                                        display_names.display_mode_overall
                                        labels), markers at each turn with
                                        data, sorted by turn along x, plus
                                        a neutral vertical dashed line at
                                        that model's overall median Success
                                        turns (every variant pooled). One
                                        shared y-axis (0-1, common scale
                                        across every model) labeled only on
                                        the first subplot; x-axis turn
                                        ticks repeat on every subplot. Two
                                        figure-level legends: the four
                                        variants (color) and the median-
                                        turns line (style).

Usage:
    python plot_retrodiction_accuracy_turn_lines_grid.py <results_parent> [--out-dir OUT_DIR]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from display_names import (
    INK_PRIMARY,
    INK_SECONDARY,
    MODE_COLORS,
    MODE_ORDER,
    SURFACE,
    display_mode_overall,
    display_model_name,
    sort_models,
    style_axes,
)
from plot_retrodiction_accuracy_by_turn_grid import (
    find_overall_median_success_turns,
    find_retrodiction_accuracy_by_turn,
)

NON_ONESHOT_MODES = [m for m in MODE_ORDER if "oneshot" not in m]


def plot_retrodiction_accuracy_turn_lines_grid(points_by_model, median_turns_by_model, out_path_base):
    if not points_by_model:
        print(f"No retrodiction data found -- skipping {out_path_base.name}")
        return

    models = sort_models(points_by_model.keys())
    modes = [m for m in NON_ONESHOT_MODES if any(m in per_mode for per_mode in points_by_model.values())]
    labels = [display_mode_overall(m) for m in modes]
    max_turn = max(
        (turn for per_mode in points_by_model.values() for points in per_mode.values() for turn, _ in points),
        default=2,
    )

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 5.0), facecolor=SURFACE, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        points_by_mode = points_by_model[model]
        for mode in modes:
            points = points_by_mode.get(mode)
            if not points:
                continue
            turns, accuracies = zip(*points)
            ax.plot(
                turns, accuracies, marker="o", markersize=4, linewidth=2,
                color=MODE_COLORS[mode], label=display_mode_overall(mode), zorder=3,
            )

        median_turns = median_turns_by_model.get(model)
        if median_turns is not None:
            ax.axvline(median_turns, color=INK_SECONDARY, linestyle=":", linewidth=1.95, zorder=2)

        style_axes(ax, "Retrodiction accuracy" if ax is axes[0] else "", xlabel="Turn", fontsize=15)
        ax.set_title(display_model_name(model), color=INK_PRIMARY, fontsize=14, pad=10)
        ax.set_xlim(1.5, max_turn + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Built from the global `modes` list (every variant that appears for
    # ANY model), not captured off one subplot's actual lines -- a model
    # missing one variant entirely (e.g. zero Success instances for it)
    # would otherwise silently drop that variant from the whole legend.
    variant_handles = [
        Line2D([0], [0], color=MODE_COLORS[mode], marker="o", markersize=4, linewidth=2, label=display_mode_overall(mode))
        for mode in modes
    ]
    median_handle = [
        Line2D([0], [0], color=INK_SECONDARY, linestyle=":", linewidth=1.8, label="Median turns (across all variants)"),
    ]
    variant_legend = fig.legend(
        handles=variant_handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(modes),
        frameon=False, fontsize=13,
    )
    median_legend = fig.legend(
        handles=median_handle, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=1, frameon=False, fontsize=11,
    )
    fig.add_artist(variant_legend)
    for legend in (variant_legend, median_legend):
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
    points_by_model = find_retrodiction_accuracy_by_turn(results_parent)
    median_turns_by_model = find_overall_median_success_turns(results_parent)
    plot_retrodiction_accuracy_turn_lines_grid(
        points_by_model, median_turns_by_model, out_dir / "retrodiction_accuracy_turn_lines_grid.png",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Parent directory containing one subfolder per model, each already analyzed "
        "(<results_parent>/<model>/analysis/{retrodiction_aggregates,turn_level_summary}.json)",
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
