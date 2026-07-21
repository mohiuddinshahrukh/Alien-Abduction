#!/usr/bin/env python3
"""
Per-turn coherence/correctness grid: one row per instance, one column per
turn, cell color = how the turn's hypothesis relates to what the player
said (predicted_output) and to the truth (gm_output).

Reads hypothesis_consistency.json (consistency_score/correctness_score per
non-parser-error turn, from hypothesis_consistency_analysis.py) and
turn_details.json (to also place parser_error turns in the grid, which
hypothesis_consistency.json excludes, and for its `match` field --
predicted_output == gm_output directly, independent of the hypothesis).

Three values are in play each turn: P (predicted_output), H
(hypothesis(input), executed), O (gm_output, the truth). Their equality
pattern has exactly five outcomes (the five ways to partition a 3-element
set), not four -- crossing "coherent" (P==H) with "correct" (H==O) alone
conflates two of them, since when neither holds, P could still happen to
equal O or not, and that's a real, different situation:

    converged             P=H=O            coherent AND correct -- the good case
    honestly_wrong        P=H, H!=O        coherent, not correct -- self-consistent, wrong theory
    right_despite_itself  H=O, H!=P        correct, not coherent -- code is actually right,
                                            but doesn't match their own stated answer
    lucky_guess           P=O, H!=P (!=O)  the stated answer is right, but the hypothesis
                                            code doesn't reproduce it -- P got there some other way
    decoupled             P, H, O all differ -- nothing agrees with anything
    unknown               hypothesis is bare "unknown", no code to check
    not_executable        code doesn't run / arity mismatch
    parser_error          the turn itself never parsed (no hypothesis at all)

Plain/print styling on purpose (white background, simple colored cells, no
rounding or gradients) so the figure drops straight into a LaTeX document
via \\includegraphics without further editing. Saves both .png and .pdf.

Usage:
    python plot_coherence_correctness_grid.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

CATEGORY_ORDER = [
    "converged", "honestly_wrong", "right_despite_itself", "lucky_guess", "decoupled",
    "unknown", "not_executable", "parser_error",
]
CATEGORY_LABELS = {
    "converged": "converged (coh+cor)",
    "honestly_wrong": "honestly wrong (coh only)",
    "right_despite_itself": "right despite itself (cor only)",
    "lucky_guess": "lucky guess (P=O, code disagrees)",
    "decoupled": "decoupled (all differ)",
    "unknown": "unknown",
    "not_executable": "not executable",
    "parser_error": "parser error",
}
# Plain, print-safe colors -- distinguishable in grayscale printing too
# (varying both hue and lightness), no theme/gradient styling.
CATEGORY_COLORS = {
    "converged": "#3a923a",
    "honestly_wrong": "#3a7ebf",
    "right_despite_itself": "#8064a2",
    "lucky_guess": "#4bacc6",
    "decoupled": "#c0504d",
    "unknown": "#e0a030",
    "not_executable": "#d98cb3",
    "parser_error": "#bfbfbf",
}

INK = "#0b0b0b"
GRID_LINE = "#ffffff"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def classify(consistency_score, correctness_score, predicted_matches_gm):
    if consistency_score == -1:
        return "unknown"
    if consistency_score == -100:
        return "not_executable"
    coherent = consistency_score == 1
    correct = correctness_score == 1
    if coherent and correct:
        return "converged"
    if coherent and not correct:
        return "honestly_wrong"
    if not coherent and correct:
        return "right_despite_itself"
    # Neither coherent nor correct -- P!=H and H!=O, but P vs O is still open:
    # a coincidental match there (the stated answer happens to be right even
    # though the hypothesis code disagrees with both it and the truth) is a
    # different situation from nothing lining up at all.
    return "lucky_guess" if predicted_matches_gm else "decoupled"


def row_label(experiment, instance):
    prefix = experiment.split("_")[0]
    suffix = instance.replace("instance_", "")[-3:]
    return f"{prefix}/{suffix}"


def build_grid(consistency_records, turn_details):
    """
    Returns (row_labels, matrix) where matrix[row_idx][turn-1] is a category
    string or None (no turn there -- the game ended before that turn). A
    blank row (row_label="") is inserted between experiment groups.
    """
    by_instance_scores = {}
    for r in consistency_records:
        key = (r["model"], r["experiment"], r["instance"])
        by_instance_scores.setdefault(key, {})[r["turn"]] = (r["consistency_score"], r["correctness_score"])

    row_labels = []
    matrix = []
    max_turn = 0

    for model in sorted(turn_details):
        for experiment in sorted(turn_details[model]):
            for mode in sorted(turn_details[model][experiment]):
                instances = sorted(turn_details[model][experiment][mode])
                if row_labels:
                    row_labels.append("")
                    matrix.append({})
                for instance in instances:
                    turns = turn_details[model][experiment][mode][instance]
                    key = (model, experiment, instance)
                    row = {}
                    for t in turns:
                        turn_num = t["turn"]
                        max_turn = max(max_turn, turn_num)
                        if t["parser_error"]:
                            row[turn_num] = "parser_error"
                        else:
                            scores = by_instance_scores.get(key, {}).get(turn_num)
                            if scores is not None:
                                row[turn_num] = classify(*scores, t["match"])
                    row_labels.append(row_label(experiment, instance))
                    matrix.append(row)

    return row_labels, matrix, max_turn


def plot_grid(row_labels, matrix, max_turn, out_path_base):
    n_rows = len(row_labels)
    fig_height = max(3.0, 0.28 * n_rows + 1.2)
    fig_width = max(6.0, 0.55 * max_turn + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")

    used_categories = []
    for row in matrix:
        for cat in row.values():
            if cat not in used_categories:
                used_categories.append(cat)
    legend_categories = [c for c in CATEGORY_ORDER if c in used_categories]

    for row_idx, row in enumerate(matrix):
        y = n_rows - 1 - row_idx
        for turn_num, category in row.items():
            ax.add_patch(
                Rectangle(
                    (turn_num - 1, y), 1, 1,
                    facecolor=CATEGORY_COLORS[category], edgecolor=GRID_LINE, linewidth=1,
                )
            )

    ax.set_xlim(0, max_turn)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(max_turn)])
    ax.set_xticklabels([str(i + 1) for i in range(max_turn)], fontsize=8, color=INK)
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(row_labels, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Turn", fontsize=9, color=INK)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS[c], edgecolor="none") for c in legend_categories
    ]
    legend_labels = [CATEGORY_LABELS[c] for c in legend_categories]
    ax.legend(
        legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, frameon=False, fontsize=8, handlelength=1.2, handleheight=1.2, columnspacing=1.2,
    )

    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path_base.with_suffix('.png')}")
    print(f"Wrote {out_path_base.with_suffix('.pdf')}")


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    consistency_path = in_dir / "hypothesis_consistency.json"
    turn_details_path = in_dir / "turn_details.json"
    if not consistency_path.exists():
        raise SystemExit(f"{consistency_path} not found -- run hypothesis_consistency_analysis.py first.")
    if not turn_details_path.exists():
        raise SystemExit(f"{turn_details_path} not found -- run analyze_results.py first.")

    consistency_records = _load(consistency_path)
    turn_details = _load(turn_details_path)

    row_labels, matrix, max_turn = build_grid(consistency_records, turn_details)
    plot_grid(row_labels, matrix, max_turn, out_dir / "coherence_correctness_grid")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py/hypothesis_consistency_analysis.py "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing hypothesis_consistency.json and turn_details.json "
        "(default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the figure (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    plot(in_dir, out_dir)


if __name__ == "__main__":
    main()
