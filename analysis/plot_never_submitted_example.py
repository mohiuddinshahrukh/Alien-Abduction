#!/usr/bin/env python3
"""
Deep-dive on a single "knew it, never submitted" episode: a model that lost
without ever getting a SOLVE accepted (source != "submitted_solution" from
plot_solution_vs_truth_grid._extract_final_solution), whose last-turn
hypothesis is nonetheless 100% correct against the held-out test_cases.

Two rows, each with the usual turn-level + held-out-test pair of panels:
    top row     what an accepted-SOLVE-only view would show for this
                episode -- since there isn't one, just flat, muted ground
                truth with no predictions at all (nothing to compare).
    bottom row  the model's actual last-turn hypothesis executed and
                compared against ground truth, in the normal
                blue-truth / green-match / red-mismatch styling.

The color split is deliberate: the top row is monochrome gray (signaling
"no submission, nothing to show"), the bottom row is the full color scheme
(signaling "here's the real comparison"), so the story reads at a glance
without needing the title.

Reads instances.json directly for ground truth/test_cases, plus every
model's raw per-instance interactions.json to find candidate episodes --
nothing precomputed, read fresh every run.

Output:
    never_submitted_example_<model>_<category>_<game_id>.png

Usage:
    # random pick among this model+category's "never submitted, 100% correct
    # anyway" episodes
    python plot_never_submitted_example.py <results_parent> <model_dir_name> --category NUMBERS [--mode passive_examples] [--seed N] [--out-dir OUT_DIR]

    # or target one exact episode directly
    python plot_never_submitted_example.py <results_parent> <model_dir_name> --experiment numbers_passive_examples --game-id 43 [--out-dir OUT_DIR]

    results_parent must contain one subfolder per model (see
    display_names.iter_model_dirs), and its *parent* directory must contain
    function_detective/in/instances.json (i.e. results_parent is the game's
    own results/ folder, e.g. .../Alien-Abduction/results).
"""

#Eg usage
# python3 plot_never_submitted_example.py Alien-Abduction/results/ gpt-5.4-mini --experiment numbers_passive_examples --game-id 43

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

import plot_solution_vs_truth_grid as svt
from display_names import GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, display_model_name

MODEL_DISPLAY_KEY_ALIASES = {"mistral-large-3": "mistral-large-3-azure"}

TRUTH_COLOR = "#2a78d6"
PRED_OK_COLOR = "#1baf7a"
PRED_WRONG_COLOR = "#e34948"
NO_SUBMISSION_COLOR = INK_MUTED  # deliberately flat/gray -- signals "nothing to compare"


def _model_label(model: str) -> str:
    return display_model_name(MODEL_DISPLAY_KEY_ALIASES.get(model, model))


def _find_candidates(results_parent: Path, instances: list, model: str, category: str = None, mode: str = None):
    """[(experiment, game_instance, source)] for every episode where this
    model Lost, never got a SOLVE accepted, and its last-turn hypothesis is
    nonetheless 100% correct against that instance's held-out test_cases."""
    out = []
    for exp in instances:
        exp_category = exp["game_instances"][0]["category"]
        exp_mode = exp["game_instances"][0]["mode"]
        if category and exp_category != category:
            continue
        if mode and exp_mode != mode:
            continue
        for gi in exp["game_instances"]:
            path = results_parent / model / "results" / exp["name"] / f"instance_{gi['game_id']:05d}" / "interactions.json"
            if not path.exists():
                continue
            interactions = json.load(open(path))
            if not interactions.get("Lose", False):
                continue
            fn, source = svt._extract_final_solution(interactions)
            if source == "submitted_solution" or fn is None:
                continue
            try:
                correct = sum(1 for tc in gi["test_cases"] if fn(*tc["args"]) == tc["expected"])
            except Exception:
                continue
            if correct == len(gi["test_cases"]):
                out.append((exp["name"], gi, source))
    return out


def plot_never_submitted_example(results_parent: Path, model: str, experiment: str, game_instance: dict,
                                  source: str, out_path: Path):
    d_solve_only = svt.build_panel_data(results_parent / model, game_instance, experiment, solve_only=True)
    d_hypothesis = svt.build_panel_data(results_parent / model, game_instance, experiment, solve_only=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=300, facecolor=SURFACE,
                              gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.3})

    rows = [
        ("No SOLVE submitted -- ground truth only", d_solve_only, False),
        (f"Last-turn hypothesis ({source}) vs ground truth", d_hypothesis, True),
    ]

    for row_i, (row_title, d, show_pred) in enumerate(rows[1:]):
        ax_turn, ax_test = axes
        truth_color = TRUTH_COLOR if show_pred else NO_SUBMISSION_COLOR

        ax_turn.set_facecolor(SURFACE)
        ax_turn.scatter(d["turn_xs"], d["turn_truth"], s=26, facecolors=truth_color,
                         edgecolors=INK_PRIMARY, linewidths=0.5, zorder=3, label="ground truth")
        if show_pred and d["turn_pred"] is not None:
            match_xs, match_ys, mismatch_xs, mismatch_ys = svt._split_by_match(
                d["turn_xs"], d["turn_truth"], d["turn_pred"])
            ax_turn.scatter(match_xs, match_ys, s=40, color=PRED_OK_COLOR, marker="x",
                             linewidths=1.4, zorder=4, label="hypothesis matches")
            ax_turn.scatter(mismatch_xs, mismatch_ys, s=40, color=PRED_WRONG_COLOR, marker="x",
                             linewidths=1.4, zorder=4, label="hypothesis differs")
        if d["error_turns"]:
            ax_turn.scatter(d["error_turns"], [0.06] * len(d["error_turns"]),
                             transform=ax_turn.get_xaxis_transform(), s=24, color=INK_MUTED,
                             marker="v", zorder=5, clip_on=False, label="parser error that turn")

        ax_test.set_facecolor(SURFACE)
        ax_test.scatter(d["test_xs"], d["test_truth"], s=10, facecolors=truth_color,
                         edgecolors=INK_PRIMARY, linewidths=0.3, zorder=3, label="ground truth")
        if show_pred and d["test_pred"] is not None:
            match_xs, match_ys, mismatch_xs, mismatch_ys = svt._split_by_match(
                d["test_xs"], d["test_truth"], d["test_pred"])
            ax_test.scatter(match_xs, match_ys, s=14, color=PRED_OK_COLOR, marker="x",
                             linewidths=0.9, zorder=4, label="hypothesis matches")
            ax_test.scatter(mismatch_xs, mismatch_ys, s=14, color=PRED_WRONG_COLOR, marker="x",
                             linewidths=0.9, zorder=4, label="hypothesis differs")

        for ax_ in (ax_turn, ax_test):
            ax_.grid(True, color=GRIDLINE, linewidth=0.5, zorder=0)
            ax_.set_axisbelow(True)
            for spine in ("top", "right"):
                ax_.spines[spine].set_visible(False)
            ax_.tick_params(colors=INK_SECONDARY, labelsize=8)

        ax_turn.set_xlim(0.5, svt.MAX_TURNS + 0.5)
        ax_turn.set_xticks([1, svt.MAX_TURNS])
        ax_turn.set_xlabel("Turn #", fontsize=12, color=INK_PRIMARY)
        ax_test.set_xlabel("Input", fontsize=12, color=INK_PRIMARY)
        ax_turn.set_ylabel("Output", fontsize=12, color=INK_PRIMARY)
        row_bg = "#eef0ee" if not show_pred else "#eaf6f0"
        """
        ax_turn.text(
            0.0, 1.12, row_title, transform=ax_turn.transAxes,
            va="bottom", ha="left", fontsize=10.5, color=INK_PRIMARY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=row_bg, edgecolor="none"),
        )
        """
        #ax_turn.legend(frameon=False, fontsize=12.5, loc="upper left")
        #ax_test.legend(frameon=False, fontsize=12.5, loc="upper left")
        break

    fig.tight_layout(rect=[0.02, 0, 1, 0.90])
    """
    fig.suptitle(
        f"{_model_label(model)} -- {game_instance['callable']} / {experiment} (instance {game_instance['game_id']})\n"
        f"episode outcome: {'success' if d_hypothesis['success'] else 'failure'} (never submitted SOLVE)",
        fontsize=12, color=INK_PRIMARY, y=1.02,
    )
    """

    # NOT out_path.with_suffix(".png") -- model names like "gpt-5.4-mini"
    # contain a dot, which with_suffix would treat as an existing
    # extension and mangle (e.g. truncating to "...gpt-5.png").
    png_path = out_path if out_path.suffix == ".png" else Path(f"{out_path}.png")
    fig.savefig(png_path, facecolor=SURFACE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


def run(results_parent: Path, model: str, out_dir: Path, category: str = None, mode: str = None,
        experiment: str = None, game_id: int = None, seed: int = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    instances_path = results_parent.parent / "function_detective" / "in" / "instances.json"
    instances = json.load(open(instances_path))["experiments"]

    if game_id is not None:
        if not experiment:
            raise ValueError("--game-id requires --experiment too")
        exp = next(e for e in instances if e["name"] == experiment)
        gi = next(g for g in exp["game_instances"] if g["game_id"] == game_id)
        interactions_path = results_parent / model / "results" / experiment / f"instance_{game_id:05d}" / "interactions.json"
        _, source = svt._extract_final_solution(json.load(open(interactions_path)))
        chosen_experiment, chosen_gi, chosen_source = experiment, gi, source
    else:
        candidates = _find_candidates(results_parent, instances, model, category=category, mode=mode)
        if not candidates:
            print(f"No 'never submitted, 100% correct anyway' episodes found for {model}"
                  f"{' / ' + category if category else ''}{' / ' + mode if mode else ''}")
            return
        rng = random.Random(seed)
        chosen_experiment, chosen_gi, chosen_source = rng.choice(candidates)
        print(f"Picked: {chosen_experiment} / {chosen_gi['callable']} / instance {chosen_gi['game_id']} "
              f"(source={chosen_source}) -- {len(candidates)} candidate(s) total")

    out_path = out_dir / f"never_submitted_example_{model}_{chosen_gi['category']}_{chosen_gi['game_id']}"
    plot_never_submitted_example(results_parent, model, chosen_experiment, chosen_gi, chosen_source, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Results folder containing one subfolder per model (each with results/ and "
        "analysis/ subfolders); its parent must contain function_detective/in/instances.json",
    )
    parser.add_argument("model", help="results/ subfolder name for the model to plot, e.g. gpt-5.4-mini")
    parser.add_argument("--category", default=None, help="Restrict random pick to this category, e.g. NUMBERS")
    parser.add_argument("--mode", default=None, help="Restrict random pick to this mode, e.g. passive_examples")
    parser.add_argument("--experiment", default=None, help="Exact experiment name (used with --game-id)")
    parser.add_argument("--game-id", type=int, default=None, help="Exact game_id (requires --experiment)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the candidate pick")
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNG (default: <results_parent>/paper_plots/never_submitted_examples)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots" / "never_submitted_examples"
    run(results_parent, args.model, out_dir, category=args.category, mode=args.mode,
        experiment=args.experiment, game_id=args.game_id, seed=args.seed)


if __name__ == "__main__":
    main()
