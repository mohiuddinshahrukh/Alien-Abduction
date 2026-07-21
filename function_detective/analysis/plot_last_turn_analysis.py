#!/usr/bin/env python3
"""
Plot last_turn_analysis.py's last-turn confidence/correctness heatmap.

Reads last_turn_analysis.py's last_turn_heatmap.json and produces:

    last_turn_heatmap_success.png / last_turn_heatmap_loss.png
        Two separate heatmaps (not one chart split into rows), one for
        Success instances and one for Loss instances. Each: one row per
        mode (all of them, including the oneshot/singleturn ones -- unlike
        the retrodiction charts, this doesn't need turn >= 2 to have
        something to show), one column per model (today's data has one, but
        this fans out as more are added). Cell text is the mean last-turn
        confidence_score for that (mode, model, outcome); cell color is
        "hypothesis correctness" for that (mode, model) -- mean
        retrodiction_accuracy at the last turn for the four multi-turn
        modes, or success rate for the two oneshot/singleturn modes --
        the same value in both the Success and Loss image for a given
        (mode, model) cell, since correctness isn't split by outcome the
        way confidence is (see last_turn_analysis.py's compute_heatmap_data
        for why).

Usage:
    python plot_last_turn_analysis.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from display_names import display_label
from plot_results import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

# Bluish-green sequential scale for the heatmap's correctness color -- low
# correctness reads as pale/near-white, high correctness as a saturated
# blue-green, so the more "correct" a mode is, the more it visually stands out.
CORRECTNESS_CMAP = "GnBu"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def plot_last_turn_heatmap(heatmap_data, out_path_base):
    """
    heatmap_data: last_turn_analysis.py's last_turn_heatmap.json, {model:
    {modes, correctness_by_mode: {mode: {value, source}},
    confidence_by_mode_outcome: {success: {mode: v}, loss: {mode: v}}}}.

    Two separate heatmaps, not one chart split into rows -- one for Success
    instances, one for Loss instances (filename gets a "_success"/"_loss"
    suffix). Each: one row per mode, one column per model (today's data has
    one model, but this fans out cleanly as more are added). Cell text is
    mean last-turn confidence_score; cell color is "hypothesis
    correctness" for that (mode, model) -- the same underlying value in
    both images for a given cell, since correctness isn't split by outcome
    the way confidence is.
    """
    if not heatmap_data:
        print(f"No last_turn_heatmap data -- skipping {out_path_base.name}")
        return

    models = sorted(heatmap_data)
    all_modes = sorted({mode for model in models for mode in heatmap_data[model]["modes"]})
    cmap = plt.get_cmap(CORRECTNESS_CMAP)

    for outcome, outcome_label in (("success", "Success"), ("loss", "Loss")):
        # None (a (mode, model) combination with nothing to average) reads as 0
        # on the color scale for display purposes only -- imshow has no "missing" color.
        color_matrix = [
            [
                (heatmap_data.get(model, {}).get("correctness_by_mode", {}).get(mode) or {}).get("value") or 0.0
                for model in models
            ]
            for mode in all_modes
        ]
        text_matrix = [
            [
                heatmap_data.get(model, {}).get("confidence_by_mode_outcome", {}).get(outcome, {}).get(mode)
                for model in models
            ]
            for mode in all_modes
        ]

        # +2.2in fixed allowance for the colorbar regardless of column count --
        # without it, a narrow (few-model) figure lets the colorbar collide
        # with the last column instead of sitting clear of it.
        fig, ax = plt.subplots(
            figsize=(2.2 + 1.6 * len(models), max(3.0, 0.7 * len(all_modes))), facecolor=SURFACE,
        )
        im = ax.imshow(color_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for row_idx in range(len(all_modes)):
            for col_idx in range(len(models)):
                value = text_matrix[row_idx][col_idx]
                text = f"{value:.2f}" if value is not None else "n/a"
                rgba = cmap(color_matrix[row_idx][col_idx])
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else INK_PRIMARY
                ax.text(
                    col_idx, row_idx, text, ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold",
                )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_yticks(range(len(all_modes)))
        ax.set_yticklabels([display_label(m) for m in all_modes])
        ax.set_title(
            f"Last-turn confidence by mode -- {outcome_label} instances",
            color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([x - 0.5 for x in range(1, len(models))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(all_modes))], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

        # fraction/pad scale off the (here, very narrow -- one column wide)
        # axes box, not the figure, so the defaults sit too close when there
        # are only one or two models -- pinned to fixed-feeling values instead.
        cbar = fig.colorbar(im, ax=ax, fraction=0.15 / len(models), pad=0.25 / len(models))
        cbar.set_label("Hypothesis correctness", color=INK_SECONDARY, fontsize=9, labelpad=12)
        cbar.ax.tick_params(colors=INK_MUTED)

        fig.tight_layout()
        out_path = out_path_base.parent / f"{out_path_base.stem}_{outcome}{out_path_base.suffix}"
        fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_last_turn_heatmap(_load(in_dir / "last_turn_heatmap.json"), out_dir / "last_turn_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to last_turn_analysis.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing last_turn_heatmap.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNGs (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    plot(in_dir, out_dir)


if __name__ == "__main__":
    main()
