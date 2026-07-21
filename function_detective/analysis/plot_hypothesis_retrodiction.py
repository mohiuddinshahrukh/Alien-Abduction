#!/usr/bin/env python3
"""
Plot hypothesis_retrodiction_analysis.py's per-turn retrodiction accuracy.

Reads retrodiction_aggregates{,_by_experiment,_by_outcome}.json (one record
per model/mode-or-experiment-or-outcome/turn: n, mean_retrodiction_accuracy)
and produces:

    retrodiction_accuracy_by_turn.png
        Mean retrodiction accuracy at each turn -- does turn t's hypothesis
        correctly reproduce gm_output for every input the player has
        already probed, not just the current one? Faceted by model, one
        line per mode, reusing plot_results.plot_metric_by_turn (the same
        chart shape as confidence_by_turn.png).

    retrodiction_accuracy_by_turn_by_experiment.png
        Same, one line per experiment instead of per mode.

    retrodiction_accuracy_by_turn_by_outcome.png
        Same, one line per outcome (success vs. loss) instead of per mode --
        a separate graph answering a different question than the two above:
        does retrodiction accuracy actually track with winning, e.g. because
        a hypothesis that overfits the latest turn (and so retrodicts worse)
        is more likely to eventually be wrong? Aborted instances excluded
        (neither outcome applies to them).

    retrodiction_accuracy_by_turn_success.png / retrodiction_accuracy_by_turn_loss.png
        The same two outcome lines again, but as two separate single-line
        charts instead of sharing one legend -- easier to read or drop into
        a document individually. Kept alongside the combined chart above,
        not instead of it.

Read the sample size (the "n=" labels) before trusting a late-turn dip or
spike here -- only instances that ran long enough reach turn 9+, and in a
results folder where most instances solve early, that's a strong
selection effect (the stragglers still going at turn 12 are disproportionately
the ones that never converged), not necessarily "hypotheses get worse with
more evidence."

Usage:
    python plot_hypothesis_retrodiction.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

from plot_results import CATEGORICAL_PALETTE, plot_metric_by_turn, plot_metric_by_turn_grid

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_key, suffix in (("mode", ""), ("experiment", "_by_experiment"), ("outcome", "_by_outcome")):
        agg_records = _load(in_dir / f"retrodiction_aggregates{suffix}.json")
        if not agg_records:
            print(f"No retrodiction_aggregates{suffix}.json data -- skipping retrodiction_accuracy_by_turn{suffix}.png")
            continue

        models = sorted({r["model"] for r in agg_records})
        group_values = sorted({r[group_key] for r in agg_records})

        if len(group_values) > len(CATEGORICAL_PALETTE):
            print(
                f"{len(group_values)} distinct {group_key} values found -- too many for one shared-legend "
                f"panel, switching to one small-multiple subplot per {group_key} instead."
            )
            plot_metric_by_turn_grid(
                agg_records, models, group_values,
                value_key="mean_retrodiction_accuracy", n_key="n",
                title="Retrodiction accuracy by turn",
                ylabel="Retrodiction accuracy",
                out_path=out_dir / f"retrodiction_accuracy_by_turn{suffix}.png",
                as_percent=True,
                series_key=group_key,
            )
        else:
            colors = CATEGORICAL_PALETTE[: len(group_values)]
            plot_metric_by_turn(
                agg_records, models, group_values, colors,
                value_key="mean_retrodiction_accuracy", n_key="n",
                title="Retrodiction accuracy by turn",
                ylabel="Retrodiction accuracy",
                out_path=out_dir / f"retrodiction_accuracy_by_turn{suffix}.png",
                as_percent=True,
                series_key=group_key,
            )

        # The outcome facet additionally gets two separate single-line charts
        # (kept alongside the combined one above, not instead of it) -- easier
        # to read or share individually than picking one line out of a shared legend.
        if group_key == "outcome":
            outcome_colors = dict(zip(group_values, CATEGORICAL_PALETTE[: len(group_values)]))
            for outcome in group_values:
                outcome_records = [r for r in agg_records if r["outcome"] == outcome]
                plot_metric_by_turn(
                    outcome_records, models, [outcome], [outcome_colors[outcome]],
                    value_key="mean_retrodiction_accuracy", n_key="n",
                    title=f"Retrodiction accuracy by turn -- {outcome.capitalize()} instances",
                    ylabel="Retrodiction accuracy",
                    out_path=out_dir / f"retrodiction_accuracy_by_turn_{outcome}.png",
                    as_percent=True,
                    series_key="outcome",
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to hypothesis_retrodiction_analysis.py "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing retrodiction_aggregates.json (default: <results_root>/analysis)",
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
