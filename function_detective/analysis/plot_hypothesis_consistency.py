#!/usr/bin/env python3
"""
Plot hypothesis_consistency_analysis.py's per-turn self-consistency scores.

Reads consistency_label_shares.json (one record per
model/mode/turn/consistency_label: count, n_total, share) and produces:

    hypothesis_consistency_by_turn.png
        Share of turns landing in each label (unknown / not_executable /
        mismatch / match) at each turn -- a grid of small-multiple panels,
        one row per model, one column per game mode. Reuses
        plot_results.plot_mode_label_shares (the same chart shape as
        mode_label_share_by_turn.png) rather than duplicating it.

    hypothesis_correctness_by_turn.png
        The same shape of chart, but from correctness_label_shares.json --
        hypothesis(input) compared against gm_output (the real answer) instead of
        predicted_output (what the player said). Consistency asks "is the player
        internally coherent"; correctness asks "is their theory actually right" --
        a turn can be consistency=match and correctness=mismatch at once (a coherent
        but wrong theory), which is exactly the gap between these two charts.

    The by-experiment variants of both (consistency_label_shares_by_experiment.json,
    correctness_label_shares_by_experiment.json) are still computed by
    hypothesis_consistency_analysis.py, but deliberately not plotted here.

Deliberately not plotted: mean score by turn, for either metric. Averaging
-100 (not_executable) together with -1/0/1 makes a mean that a single bad
turn can send to -40 or lower -- see analyze_results.py's docstring on this --
so the label-share view above is the only chart here; *_aggregates.json
still has the mean for anyone who wants to filter/inspect it directly.

Usage:
    python plot_hypothesis_consistency.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

from plot_results import plot_mode_label_shares

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mode_label_shares(
        _load(in_dir / "consistency_label_shares.json"), parser_error_records=[],
        out_path=out_dir / "hypothesis_consistency_by_turn.png",
        facet_key="mode", label_key="consistency_label", exclude_labels=frozenset(),
        chart_title="hypothesis self-consistency by turn",
    )

    plot_mode_label_shares(
        _load(in_dir / "correctness_label_shares.json"), parser_error_records=[],
        out_path=out_dir / "hypothesis_correctness_by_turn.png",
        facet_key="mode", label_key="correctness_label", exclude_labels=frozenset(),
        chart_title="hypothesis correctness (vs. gm_output) by turn",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py/hypothesis_consistency_analysis.py "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing consistency_label_shares.json (default: <results_root>/analysis)",
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
