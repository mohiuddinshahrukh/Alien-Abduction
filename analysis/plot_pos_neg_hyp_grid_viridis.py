#!/usr/bin/env python3
"""
Cosmetic-only viridis restyle of plot_pos_neg_hyp_grid.py -- same data, same
category-block-of-2-mode-columns grid layout, same evidence-marker-plus-
hypothesis-overlay content, just re-skinned toward the visual language of a
reference "parameter sweep" chart the user showed (viridis color gradient,
a dashed reference line, one subplot grid). Does NOT modify or replace
plot_pos_neg_hyp_grid.py or plot_pos_neg_turns_hypothesis_overlay.py --
their own outputs are untouched; this is a new, separate script/output.

Explicitly NOT a semantic conversion -- there was no way to make that fit.
The reference chart plots a continuous signed score against a continuous
swept parameter, with a real y=0 crossing; this chart's data is categorical
(discrete turn number x-axis, 10 stacked function rows, evidence/hypothesis
MARKER TYPES, not a score), so there's nothing that actually crosses zero.
Two purely cosmetic borrows were applied instead, per explicit request for
"try the 1st [cosmetic-only] option":

  1. Viridis-inspired fixed marker colors: positive evidence (circle) is
     always blue, negative evidence (square) is always yellow -- per
     explicit follow-up request, replacing an earlier version of this
     script that colored each function ROW individually off the viridis
     colormap (that made color encode row identity, which was already
     unambiguous from the y-tick label, so the color added nothing and was
     dropped in favor of this simpler fixed-by-shape scheme, closer to the
     original chart's own fixed-color-pair convention, just recolored).
  2. Dashed reference line: the reference's dashed line marks y=0 on a
     continuous score axis, which has no equivalent here -- this chart's
     y-axis is just row identity. What IS meaningful on this chart's own
     x-axis (turn number) is "how long these instances typically ran", so a
     vertical dashed line at the median last-turn across the panel's own 10
     instances is drawn instead, echoing the reference's dashed-reference-
     line motif with something that's actually informative for this data
     rather than a decoration with no meaning.

The hypothesis-match/mismatch cross overlay keeps its original teal/red/
mauve colors unchanged -- recoloring that too would make it blend into the
new per-row viridis palette and lose the distinction it exists to show.

Output:
    pos_neg_hyp_grid_viridis_<category-suffix>_<model>.png

Usage:
    python plot_pos_neg_hyp_grid_viridis.py <results_parent> <model> [--categories LIST,LOGIC,...] [--out-dir OUT_DIR]
"""

#python3 plot_pos_neg_hyp_grid_viridis.py Alien-Abduction/results/
#python3 plot_pos_neg_hyp_grid_viridis.py Alien-Abduction/results/ --categories TWO_NUMBERS

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

import plot_pos_neg_turns_verdict_compare as pnv
import plot_solution_vs_truth_grid as svt
from display_names import GRIDLINE, INK_PRIMARY, INK_SECONDARY, SURFACE, display_mode_overall
from plot_input_coverage_grid import FUNC_PREFIX_BY_CATEGORY
from plot_pos_neg_turns_hypothesis_overlay import STATUS_MARK, _hyp_verdict_status

CATEGORIES = ["LIST", "LOGIC", "NUMBERS", "TWO_NUMBERS"]
# Sampled off the viridis colormap's own low/high end -- a clear blue and a
# clear yellow, per explicit request, rather than arbitrary named colors.
POS_MARKER_COLOR = plt.get_cmap("viridis")(0.25)  # blue
NEG_MARKER_COLOR = plt.get_cmap("viridis")(1.0)  # yellow


def _plot_model_pair_viridis(fig, gs, results_parent: Path, instances: list, model: str, modes: list,
                              category: str, callables: list, legend_state: dict):
    prefix = FUNC_PREFIX_BY_CATEGORY[category]
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(modes))]

    for ax, mode in zip(axes, modes):
        is_active = "active" in mode
        experiment = f"{category.lower()}_{mode}"
        ax.set_facecolor(SURFACE)

        labels, label_colors = [], []
        last_turns = []
        for row, callable_name in enumerate(callables):
            gi = pnv._game_instance(instances, experiment, callable_name)
            path = results_parent / model / "results" / experiment / f"instance_{gi['game_id']:05d}" / "interactions.json"
            success = None
            valid, error_turns = [], []
            hyp_status = {}
            if path.exists():
                interactions = json.load(open(path))
                success = interactions.get("Success", False)
                records = svt._turn_records(interactions, category, is_active=is_active)
                valid = [(t, args, gm, label) for t, status, args, gm, label in records if status == "valid"]
                error_turns = [t for t, status, _, _, _ in records if status == "parser_error"]
                if not valid:
                    valid = svt._examples_from_game_instance(gi)
                hyp_status = _hyp_verdict_status(interactions, category, is_active)

            if valid:
                last_turns.append(max(t for t, *_ in valid))

            outcome = "?" if success is None else ("success" if success else "failure")
            labels.append(f"{prefix}{row + 1} ({outcome})")
            label_colors.append(INK_PRIMARY if (success or success is None) else pnv.FAILURE_COLOR)

            pos_x = [t for t, _, _, label in valid if label]
            neg_x = [t for t, _, _, label in valid if not label]
            y = row
            pos_pts = ax.scatter(pos_x, [y] * len(pos_x), s=80, facecolors=POS_MARKER_COLOR, marker="o",
                                  edgecolors=INK_PRIMARY, linewidths=1.2, zorder=3)
            neg_pts = ax.scatter(neg_x, [y] * len(neg_x), s=80, facecolors=NEG_MARKER_COLOR, marker="s",
                                  edgecolors=INK_PRIMARY, linewidths=1.2, zorder=3)
            err_pts = None
            if error_turns:
                err_pts = ax.scatter(error_turns, [y] * len(error_turns), s=82, facecolors=svt.PARSE_ERROR_COLOR,
                                      marker="v", edgecolors="none", zorder=2)

            cross_handles = {}
            for status, spec in STATUS_MARK.items():
                xs = [t for t, _args, _gm, _label in valid if hyp_status.get(t) == status]
                if not xs:
                    continue
                h = ax.scatter(xs, [y - 0.3] * len(xs), zorder=4, **spec)
                cross_handles[status] = h

            # Each of pos/neg/err is captured independently, the first time
            # IT (not the others) has data -- the old "only when this same
            # row/mode has both pos AND neg AND an error" gate meant parser
            # error could vanish from the legend entirely if the first row
            # to have both pos and neg evidence happened not to also have a
            # parser-error turn, even though other rows further down did.
            if legend_state.get("pos") is None and pos_x:
                legend_state["pos"] = pos_pts
            if legend_state.get("neg") is None and neg_x:
                legend_state["neg"] = neg_pts
            if legend_state.get("err") is None and err_pts is not None:
                legend_state["err"] = err_pts
            for status, h in cross_handles.items():
                legend_state["cross_handles_state"].setdefault(status, h)

        if last_turns:
            ax.axvline(statistics.median(last_turns), color=INK_SECONDARY, linestyle="--", linewidth=1.6, zorder=1)

        ax.set_yticks(range(len(callables)))
        ax.set_yticklabels(labels, fontsize=18.5)
        for tick, color in zip(ax.get_yticklabels(), label_colors):
            tick.set_color(color)
        ax.set_ylim(-0.5, len(callables) - 0.5)
        ax.invert_yaxis()
        ax.set_xlim(0.5, svt.MAX_TURNS + 0.5)
        ax.set_xticks(range(1, svt.MAX_TURNS + 1))
        ax.set_xlabel("Turn", fontsize=18, color=INK_PRIMARY)
        ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=16.5)
        ax.tick_params(axis="y", labelsize=18.5)
        ax.set_title(display_mode_overall(mode), fontsize=21, color=INK_PRIMARY, fontweight="bold", pad=18)

    return axes


def plot_grid_viridis(results_parent: Path, instances_path: Path, model: str, modes: list, out_path: Path,
                       categories: list = None):
    categories = categories or CATEGORIES
    instances = json.load(open(instances_path))["experiments"]
    n_modes = len(modes)
    n_cat = len(categories)

    fig = plt.figure(figsize=(7.5 * n_modes * n_cat, 6), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(top=0.78)

    block_width = 1.0 / n_cat
    margin = 0.025
    inter_gap = 0.035
    legend_state = {"pos": None, "neg": None, "err": None, "cross_handles_state": {}}
    for i, category in enumerate(categories):
        left = margin if i == 0 else i * block_width + inter_gap / 2
        right = (1 - margin) if i == n_cat - 1 else (i + 1) * block_width - inter_gap / 2
        ref_experiment = f"{category.lower()}_{modes[0]}"
        callables = [gi["callable"] for gi in next(e for e in instances if e["name"] == ref_experiment)["game_instances"]]

        # wspace widened (0.29 -> 0.55) so a right-hand subplot's function-
        # name row labels get more clearance from the left-hand subplot's
        # turn-15 column, per explicit request.
        gs = fig.add_gridspec(1, n_modes, left=left, right=right, wspace=0.35)
        axes = _plot_model_pair_viridis(fig, gs, results_parent, instances, model, modes, category, callables, legend_state)

        x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        #fig.text(x_mid, 0.86, category, ha="center", va="top",
        #          fontsize=28, color=INK_PRIMARY, fontweight="bold")

    from matplotlib.lines import Line2D
    median_handle = Line2D([0], [0], color=INK_SECONDARY, linestyle="--", linewidth=1.6)

    # Built independently per marker type (see legend_state's collection
    # above) -- each entry included iff that marker type appeared ANYWHERE
    # in the whole grid, not gated on some other marker type also having
    # appeared in the same row.
    evidence_entries = [
        ("pos", "positive evidence"),
        ("neg", "negative evidence"),
        ("err", "parser error"),
    ]
    evidence_handles = [legend_state[key] for key, _ in evidence_entries if legend_state.get(key) is not None]
    evidence_labels = [label for key, label in evidence_entries if legend_state.get(key) is not None]

    cross_order = ["match", "mismatch", "unknown", "non_executable"]
    cross_text = {
        "match": "hypothesis matches",
        "mismatch": "hypothesis mismatches",
        "unknown": "hypothesis: unknown",
        "non_executable": "hypothesis not executable",
    }
    cross_handles = [legend_state["cross_handles_state"][k] for k in cross_order if k in legend_state["cross_handles_state"]]
    cross_labels = [cross_text[k] for k in cross_order if k in legend_state["cross_handles_state"]]

    all_handles = evidence_handles + [median_handle] + cross_handles
    all_labels = evidence_labels + ["median turns"] + cross_labels
    # ncol = half the entry count (rounded up) so the legend wraps into 2
    # rows instead of one long single-row strip, per explicit request.
    ncol = -(-len(all_handles) // 2)
    fig.legend(handles=all_handles, labels=all_labels, loc="upper center", ncol=ncol,
               frameon=False, fontsize=20, bbox_to_anchor=(0.5, 1.06))

    # No colorbar for the row-index gradient -- each row's identity is
    # already unambiguous from its own y-tick label (FN1...FN10), so a
    # color->index legend would explain something the chart doesn't need
    # explained (per explicit feedback: the color is a cosmetic echo of the
    # reference chart, not a load-bearing encoding the reader has to decode).
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def run(results_parent: Path, model: str, out_dir: Path, modes: list = None, categories: list = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    instances_path = results_parent.parent / "function_detective" / "in" / "instances.json"
    modes = modes or pnv.DEFAULT_MODES
    categories = categories or CATEGORIES

    cat_suffix = "" if categories == CATEGORIES else "_" + "-".join(c.lower() for c in categories)
    out_path = out_dir / f"pos_neg_hyp_grid_viridis{cat_suffix}_{pnv._model_label(model)}.png"
    plot_grid_viridis(results_parent, instances_path, model, modes, out_path, categories=categories)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Results folder containing one subfolder per model; its parent must contain "
        "function_detective/in/instances.json",
    )
    parser.add_argument("model", nargs="?", default="gpt-5.4", help="results/ subfolder name (default: gpt-5.4)")
    parser.add_argument(
        "--modes", default=",".join(pnv.DEFAULT_MODES),
        help=f"Comma-separated modes, in column order (default: {','.join(pnv.DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--categories", default=",".join(CATEGORIES),
        help=f"Comma-separated categories, in block order (default: {','.join(CATEGORIES)})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNG (default: <results_parent>/paper_plots/active_pair_checks_turns)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots" / "active_pair_checks_turns"
    run(results_parent, args.model, out_dir, modes=args.modes.split(","), categories=args.categories.split(","))


if __name__ == "__main__":
    main()
