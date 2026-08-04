#!/usr/bin/env python3
"""
For one model, one category: a single row of 4 panels (one per multi-turn
mode -- active_inputs, active_pair_checks, passive_examples,
passive_labeled_pairs; the single-turn *_oneshot variants are excluded)
showing, for every one of that category's 10 sampled callables, the held-out
test set's input space vs. the inputs the model actually queried/was shown
during play. Companion to plot_input_coverage_grid.py, which instead fixes
the mode and grids the 4 categories -- this one fixes the category and grids
the modes, so you can see how exploration for a single category changes
across active vs. passive
elicitation.

Each row's y-tick label is the same short synthetic ID scheme as
plot_input_coverage_grid.py (FLI#/FLO#/FN#/F2N#, numbered in the fixed order
the callables appear in instances.json -- identical across modes, so IDs
line up across panels here) plus "(success)"/"(failure)", colored black/red.
Real callable names are written once to function_id_mapping.json (shared
file, same content as plot_input_coverage_grid.py writes).

The x-axis is the single numeric proxy from
plot_solution_vs_truth_grid._x_value (LIST sums the list, LOGIC counts
Trues, NUMBERS is the scalar itself, TWO_NUMBERS sums the pair). NUMBERS and
TWO_NUMBERS use a symlog x-axis (linear inside +/-100, log beyond) for the
same reason as plot_input_coverage_grid.py: held-out sets pack densely near
0 plus sparse edge probes out near +/-1000/2000.

Reads instances.json directly for the held-out test_cases' args, plus each
model's raw per-instance interactions.json (evaldata) for the inputs it
actually queried/was shown -- both read fresh, nothing precomputed.

Output:
    coverage_by_mode_<category>_<model short name>.png
    function_id_mapping.json  {category: {id: real_callable_name}}

Usage:
    python plot_input_coverage_by_mode.py <results_parent> <model_dir_name> [--category LIST] [--out-dir OUT_DIR]
    (omit --category to generate all 4: LIST, LOGIC, NUMBERS, TWO_NUMBERS)

    results_parent must contain one subfolder per model (see
    display_names.iter_model_dirs), and its *parent* directory must contain
    function_detective/in/instances.json (i.e. results_parent is the game's
    own results/ folder, e.g. .../Alien-Abduction/results).
"""
#python3 plot_input_coverage_by_mode.py Alien-Abduction/results/ gpt-5.4 --modes active_inputs,passive_examples --category TWO_NUMBERS

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_solution_vs_truth_grid as svt
from plot_input_coverage_grid import (
    CATEGORIES, FUNC_PREFIX_BY_CATEGORY, PANEL_TITLE_BY_CATEGORY, SYMLOG_LINTHRESH_BY_CATEGORY,
    HELD_OUT_COLOR, QUERIED_COLOR, FAILURE_COLOR, SUCCESS_COLOR,
    _model_label, write_function_id_mapping,
)
from display_names import GRIDLINE, INK_PRIMARY, INK_SECONDARY, SURFACE, display_mode_overall

MODES = [
    "active_inputs",
    "active_pair_checks",
    "passive_examples",
    "passive_labeled_pairs",
]

# Low-cardinality categories (LOGIC's x-proxy is a count of Trues among 2
# bools, so only 0/1/2 are possible) put many points at identical (x, y)
# coordinates -- held-out and queried markers land exactly on top of each
# other and the queried diamond (drawn later, larger) fully occludes the
# held-out circle beneath it. A small deterministic y-jitter spreads
# same-row points apart vertically so both are visible; amplitude stays
# well under 0.5 so jittered points never cross into a neighboring row.
JITTER_AMPLITUDE = 0.3


def _jitter(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-JITTER_AMPLITUDE, JITTER_AMPLITUDE, size=n) if n else np.array([])


def _panel_rows(results_parent: Path, instances: list, model: str, mode: str, category: str):
    """[(callable_name, success, held_out_xs, queried_xs)] for this category's 10 callables."""
    experiment = f"{category.lower()}_{mode}"
    exp = next((e for e in instances if e["name"] == experiment), None)
    if exp is None:
        return []
    is_active = "active" in mode

    rows = []
    for gi in exp["game_instances"]:
        held_out = [tc["args"] for tc in gi["test_cases"]]
        hx = [svt._x_value(category, a) for a in held_out]

        path = results_parent / model / "results" / experiment / f"instance_{gi['game_id']:05d}" / "interactions.json"
        queried = []
        success = False
        if path.exists():
            interactions = json.load(open(path))
            success = interactions.get("Success", False)
            records = svt._turn_records(interactions, category, is_active=is_active)
            valid = [(t, args, gm, label) for t, status, args, gm, label in records if status == "valid"]
            if not valid:
                valid = svt._examples_from_game_instance(gi)
            queried = [args for _, args, _, _ in valid]
        qx = [svt._x_value(category, a) for a in queried]

        rows.append((gi["callable"], success, hx, qx))
    return rows


def plot_coverage_by_mode(results_parent: Path, instances: list, model: str, category: str, out_path: Path,
                           modes=None):
    modes = modes or MODES
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 6), dpi=170, facecolor=SURFACE)
    if len(modes) == 1:
        axes = [axes]
    prefix = FUNC_PREFIX_BY_CATEGORY[category]

    rng = np.random.default_rng(0)
    for ax, mode in zip(axes, modes):
        ax.set_facecolor(SURFACE)
        rows = _panel_rows(results_parent, instances, model, mode, category)

        labels = []
        label_colors = []
        for y, (callable_name, success, hx, qx) in enumerate(rows):
            outcome = "success" if success else "failure"
            labels.append(f"{prefix}{y + 1} ({outcome})")
            label_colors.append(SUCCESS_COLOR if success else FAILURE_COLOR)

            ax.scatter(hx, y + _jitter(rng, len(hx)), s=96, facecolors=HELD_OUT_COLOR, edgecolors="none",
                       alpha=0.75, zorder=2)
            if qx:
                ax.scatter(qx, y + _jitter(rng, len(qx)), s=106, facecolors=QUERIED_COLOR, edgecolors=INK_PRIMARY,
                           linewidths=0.5, marker="D", zorder=3)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=18.5)
        for tick, color in zip(ax.get_yticklabels(), label_colors):
            tick.set_color(color)
        ax.set_ylim(-0.5, max(len(rows), 1) - 0.5)
        ax.invert_yaxis()

        if category in SYMLOG_LINTHRESH_BY_CATEGORY:
            ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH_BY_CATEGORY[category])
        ax.set_xlabel("Input", fontsize=19.5, color=INK_PRIMARY)
        ax.set_title(display_mode_overall(mode), fontsize=21, color=INK_PRIMARY, fontweight="bold")
        ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=16.5)
        ax.tick_params(axis="y", labelsize=16.5)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=HELD_OUT_COLOR,
               markeredgecolor="none", markersize=14, label="held-out"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=QUERIED_COLOR,
               markeredgecolor=INK_PRIMARY, markersize=14, label="queried"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
               markeredgecolor="none", markersize=4, label=""),  # spacer
        Line2D([0], [0], marker="_", linestyle="none", markeredgecolor=SUCCESS_COLOR,
               markerfacecolor=SUCCESS_COLOR, markersize=14, markeredgewidth=3, label="success"),
        Line2D([0], [0], marker="_", linestyle="none", markeredgecolor=FAILURE_COLOR,
               markerfacecolor=FAILURE_COLOR, markersize=14, markeredgewidth=3, label="failure"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False,
               fontsize=22.5, bbox_to_anchor=(0.5, 1.03))

    #fig.suptitle(f"{PANEL_TITLE_BY_CATEGORY[category]} -- input coverage across modes", fontsize=14, color=INK_PRIMARY, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(out_path, facecolor=SURFACE, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def run(results_parent: Path, model: str, out_dir: Path, categories=None, modes=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    instances_path = results_parent.parent / "function_detective" / "in" / "instances.json"
    instances = json.load(open(instances_path))["experiments"]

    modes_suffix = "" if not modes or set(modes) == set(MODES) else "_" + "-".join(modes)
    for category in categories or CATEGORIES:
        plot_coverage_by_mode(
            results_parent, instances, model, category,
            out_dir / f"coverage_by_mode_{category.lower()}{modes_suffix}_{_model_label(model)}.png",
            modes=modes,
        )

    write_function_id_mapping(instances, "active_inputs", out_dir / "function_id_mapping.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Results folder containing one subfolder per model (each with results/ and "
        "analysis/ subfolders); its parent must contain function_detective/in/instances.json",
    )
    parser.add_argument("model", help="results/ subfolder name for the model to plot, e.g. gpt-5.4")
    parser.add_argument(
        "--category", default=None, choices=CATEGORIES,
        help="Restrict to a single category (default: generate all 4)",
    )
    parser.add_argument(
        "--modes", default=None,
        help="Comma-separated subset of modes to include, e.g. active_inputs,passive_examples "
        "(default: all 4 -- active_inputs, active_pair_checks, passive_examples, passive_labeled_pairs)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNGs and JSON (default: <results_parent>/paper_plots/input_coverage_by_mode)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots" / "input_coverage_by_mode"
    modes = args.modes.split(",") if args.modes else None
    run(results_parent, args.model, out_dir, categories=[args.category] if args.category else None, modes=modes)


if __name__ == "__main__":
    main()
