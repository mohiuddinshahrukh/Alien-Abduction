#!/usr/bin/env python3
"""
Plot the calibration reliability diagram from confidence_hypothesis_analysis.py's
calibration.json / calibration_by_experiment.json.

Reads calibration{,_by_experiment}.json (one record per model/mode-or-experiment/
hypothesis_committed/confidence_bucket: n, match_rate, mean_confidence) and produces:

    calibration.png
        A grid of small-multiple panels (one row per model, one column per game
        mode). Each panel plots actual match_rate against the player's stated
        confidence bucket, as two lines -- "hypothesis committed" and "still
        unknown" -- alongside a dashed gray diagonal marking perfect calibration
        (bucket midpoint == match rate). A line that tracks the diagonal means
        stated confidence is trustworthy; below it is overconfidence, above it
        is underconfidence.

    calibration_by_experiment.png
        The same chart, columns are experiments instead of modes (several
        experiments can share one mode, e.g. numbers_active_inputs and
        list_active_inputs both run "active_inputs").

Usage:
    python plot_confidence_hypothesis_analysis.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from display_names import display_label
from plot_results import SURFACE, _annotate, _legend, _style_axes

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

BUCKET_ORDER = ["0-20", "20-40", "40-60", "60-80", "80-100"]
BUCKET_MIDPOINTS = {"0-20": 10, "20-40": 30, "40-60": 50, "60-80": 70, "80-100": 90}

COMMITTED_COLOR = "#2a78d6"  # blue
UNCOMMITTED_COLOR = "#eb6834"  # orange
DIAGONAL_COLOR = "#c3c2b7"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def plot_calibration(records, out_path, facet_key="mode"):
    if not records:
        print(f"No calibration data -- skipping {out_path.name}")
        return

    models = sorted({r["model"] for r in records})
    facet_values = sorted({r[facet_key] for r in records})

    fig, axes = plt.subplots(
        len(models), len(facet_values), figsize=(6.5 * len(facet_values), 5 * len(models)),
        facecolor=SURFACE, squeeze=False, sharey=True,
    )

    for row_idx, model in enumerate(models):
        for col_idx, facet_value in enumerate(facet_values):
            ax = axes[row_idx][col_idx]
            panel_rows = [r for r in records if r["model"] == model and r[facet_key] == facet_value]
            if not panel_rows:
                ax.set_visible(False)
                continue

            xs = list(range(1, len(BUCKET_ORDER) + 1))
            ys = [BUCKET_MIDPOINTS[b] for b in BUCKET_ORDER]
            ax.plot(
                xs, ys, color=DIAGONAL_COLOR, linewidth=1.5, linestyle="--",
                zorder=2, label="perfect calibration",
            )

            for committed, color, label in (
                (True, COMMITTED_COLOR, "hypothesis committed"),
                (False, UNCOMMITTED_COLOR, "still unknown"),
            ):
                rows = sorted(
                    (r for r in panel_rows if r["hypothesis_committed"] == committed),
                    key=lambda r: BUCKET_ORDER.index(r["confidence_bucket"]),
                )
                if not rows:
                    continue
                x = [BUCKET_ORDER.index(r["confidence_bucket"]) + 1 for r in rows]
                y = [r["match_rate"] * 100 for r in rows]
                ax.plot(
                    x, y, color=color, linewidth=2, solid_capstyle="round",
                    marker="o", markersize=8, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=label, zorder=3,
                )
                y_offset = 10 if committed else -16
                # n= labels disabled -- uncomment to re-enable.
                # for xi, yi, r in zip(x, y, rows):
                #     _annotate(ax, xi, yi, f"n={r['n']}", color, y_offset)

            panel_title = (
                f"{model} -- {display_label(facet_value)}" if len(models) > 1 or len(facet_values) > 1
                else "Calibration: confidence vs. actual match rate"
            )
            _style_axes(ax, panel_title, "Actual match rate", xlabel="Stated confidence bucket")
            ax.set_xlim(0.5, len(BUCKET_ORDER) + 0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(BUCKET_ORDER)
            ax.set_ylim(-5, 105)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v >= 0 else ""))
            if row_idx == 0 and col_idx == len(facet_values) - 1:
                # n= labels disabled elsewhere on this chart -- note describing them
                # disabled too; uncomment both together to re-enable.
                # _legend(ax, 3, notes=["n = turns in this confidence bucket (not instances -- one instance can contribute several turns)"])
                _legend(ax, 3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(_load(in_dir / "calibration.json"), out_dir / "calibration.png", facet_key="mode")
    plot_calibration(
        _load(in_dir / "calibration_by_experiment.json"), out_dir / "calibration_by_experiment.png",
        facet_key="experiment",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py/confidence_hypothesis_analysis.py "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing calibration.json (default: <results_root>/analysis)",
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
