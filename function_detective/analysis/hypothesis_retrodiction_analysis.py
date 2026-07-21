#!/usr/bin/env python3
"""
Retrodiction check: does turn t's hypothesis correctly explain every input
the player has already seen, not just the current one?

For each turn t (t >= 2) in an instance, take that turn's hypothesis code
and run it against every EARLIER turn's input (i = 1..t-1, excluding
parser_error turns), comparing the result to that earlier turn's gm_output
-- the ground truth for that historical probe, not the earlier turn's own
predicted_output or hypothesis. This is a stricter test than
hypothesis_consistency_analysis.py's correctness_score, which only checks
turn t's hypothesis against turn t's *own* input: a hypothesis can
correctly predict its own turn's input while silently breaking on
something the player already saw a few turns ago -- overfitting to the
latest data point instead of actually explaining the whole pattern.
Retrodiction accuracy tracks whether that happens, and whether it improves
as turns go on (a real theory should retrodict *better* over time, not
just predict the newest input).

Reuses hypothesis_consistency_analysis.py's sandbox-execution helpers
(_check_executable, _compare_against) unchanged -- same arity pre-check,
same Docker sandbox call, same -100/0/1 scoring per pair -- just pointed at
a different (input, expected) pair each time: (input_i, gm_output_i) for
every i < t, instead of (input_t, gm_output_t) or (input_t, predicted_output_t).

This calls the Docker sandbox once per (t, i) pair whose arity checks out --
that's roughly n*(n-1)/2 calls per instance instead of the ~n calls the
other hypothesis_*.py scripts make, so this is the slowest script in this
set by a wide margin. Expect several minutes, not seconds, on a results
folder with many multi-turn instances.

Scored for both Success AND Loss instances -- but sourced differently.
Success instances' turns come from turn_details.json (analyze_results.py),
same as every other script in this set. Loss instances never appear there
(analyze_results.py's load_instance only extracts turn data for Success
instances), so their turns are read directly from each Lose instance's
interactions.json instead (collect_loss_retrodiction_records, reusing
analyze_results.py's build_turn_row unchanged) and scored with the exact
same score_against_history sandbox calls. Practically, this roughly
triples the total runtime, since Lose instances tend to run longer before
giving up than Success ones do. Aborted instances are still excluded
entirely -- neither outcome applies to them.

Per turn t, per instance:
    hypothesis_unknown     turn t's hypothesis was bare "unknown" -- nothing
                            to check; retrodiction_accuracy is None
    n_prev                 number of earlier valid (non-parser-error) turns
    n_scoreable             of those, how many turn t's hypothesis could
                            actually be run against (excludes arity
                            mismatches / runtime errors on that input)
    n_match                 of the scoreable ones, how many turn t's
                            hypothesis reproduced correctly
    retrodiction_accuracy   n_match / n_scoreable (None if n_scoreable == 0)

Output JSON files (default: <results_root>/analysis, same as analyze_results.py):
    retrodiction_details.json                    flat list, one record per (instance, turn>=2),
                                                  Success and Loss instances both, tagged with
                                                  "outcome"
    retrodiction_aggregates.json                 (model, mode, turn) -> n, mean_retrodiction_accuracy
                                                  -- Success instances only, same population as
                                                  before Loss instances were scoreable at all
    retrodiction_aggregates_by_experiment.json   same, grouped by experiment -- also Success-only
    retrodiction_aggregates_by_outcome.json      same, grouped by outcome (success/loss) instead of
                                                  mode/experiment -- does a hypothesis retrodict worse
                                                  in instances that end up lost, e.g. because the player
                                                  latched onto an overfit theory? Aborted instances
                                                  excluded, same reasoning as
                                                  turn_level_summary_analysis.py -- an abort is neither
                                                  outcome, so it isn't comparable to either.

Usage:
    python hypothesis_retrodiction_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from analyze_results import build_turn_row, find_instance_files, get_model_name
from display_names import display_label
from hypothesis_consistency_analysis import _check_executable, _compare_against

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(records, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def build_outcome_lookup(instances):
    """Returns {(model, experiment, mode, instance): "success"|"loss"|"aborted"}."""
    outcomes = {}
    for model, by_experiment in instances.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, summary in by_instance.items():
                    if summary["Success"]:
                        outcome = "success"
                    elif summary["Lose"]:
                        outcome = "loss"
                    else:
                        outcome = "aborted"
                    outcomes[(model, experiment, mode, instance)] = outcome
    return outcomes


def group_instances(turn_details):
    """Returns {(model, experiment, mode, instance): [valid turns sorted by turn number]}."""
    groups = {}
    for model, by_experiment in turn_details.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, turns in by_instance.items():
                    valid_turns = sorted((t for t in turns if not t["parser_error"]), key=lambda t: t["turn"])
                    groups[(model, experiment, mode, instance)] = valid_turns
    return groups


def score_against_history(hypothesis_t, history):
    """
    history: list of (input_i, gm_output_i) for turns before t. Returns
    (n_prev, n_scoreable, n_match, n_not_executable). hypothesis_t is
    assumed not to be bare "unknown" -- callers check that first, since
    it's a property of turn t alone and doesn't need rechecking per i.
    """
    n_scoreable = n_match = n_not_executable = 0
    for input_i, gm_output_i in history:
        pre_score, args, _ = _check_executable(hypothesis_t, input_i)
        if pre_score is not None:
            n_not_executable += 1
            continue
        try:
            gm_value = ast.literal_eval(gm_output_i) if isinstance(gm_output_i, str) else gm_output_i
        except (ValueError, SyntaxError):
            n_not_executable += 1
            continue
        score, _ = _compare_against(hypothesis_t, args, gm_value)
        if score == 1:
            n_scoreable += 1
            n_match += 1
        elif score == 0:
            n_scoreable += 1
        else:
            n_not_executable += 1
    return len(history), n_scoreable, n_match, n_not_executable


def score_instance_turns(model, experiment, mode, instance, turns, outcome):
    """
    Per turn t (t >= 2) in this instance's valid turns, score retrodiction
    against every earlier turn. Shared by both the Success-instance loop
    (turns sourced from turn_details.json) and
    collect_loss_retrodiction_records (turns read directly from
    interactions.json, since turn_details.json only ever covers Success --
    see analyze_results.py's load_instance).
    """
    records = []
    for idx, t in enumerate(turns):
        if idx == 0:
            continue  # turn 1 has no earlier turns to retrodict
        history = [(prev["input"], prev["gm_output"]) for prev in turns[:idx]]

        if isinstance(t["hypothesis"], str) and t["hypothesis"].strip().lower() == "unknown":
            records.append(
                {
                    "model": model, "experiment": experiment, "mode": mode, "instance": instance,
                    "outcome": outcome, "turn": t["turn"], "hypothesis_unknown": True,
                    "n_prev": len(history), "n_scoreable": 0, "n_match": 0,
                    "n_not_executable": len(history), "retrodiction_accuracy": None,
                }
            )
            continue

        n_prev, n_scoreable, n_match, n_not_executable = score_against_history(t["hypothesis"], history)
        records.append(
            {
                "model": model, "experiment": experiment, "mode": mode, "instance": instance,
                "outcome": outcome, "turn": t["turn"], "hypothesis_unknown": False,
                "n_prev": n_prev, "n_scoreable": n_scoreable, "n_match": n_match,
                "n_not_executable": n_not_executable,
                "retrodiction_accuracy": (n_match / n_scoreable) if n_scoreable else None,
            }
        )
    return records


def collect_loss_retrodiction_records(results_root: Path):
    """
    Same per-turn retrodiction scoring as the Success-instance loop in
    analyze(), just sourced directly from each Lose instance's
    interactions.json instead of turn_details.json (which only ever covers
    Success instances -- see analyze_results.py's load_instance). Calls the
    Docker sandbox exactly the same way (score_against_history), so this is
    exactly as slow per-pair as the Success side -- Lose instances usually
    ran longer before giving up, so expect this to dominate the total
    runtime even more than the Success pass does.
    """
    groups = {}
    for interactions_path in find_instance_files(results_root):
        with open(interactions_path) as f:
            data = json.load(f)
        if not data.get("Lose"):
            continue

        experiment = interactions_path.parent.parent.name
        instance_id = interactions_path.parent.name
        model = get_model_name(data)
        mode = data.get("mode")

        turn_rows = [
            build_turn_row(entry, idx, experiment, instance_id, model, mode)
            for idx, entry in enumerate(data.get("evaldata", []), start=1)
        ]
        valid_turns = sorted((t for t in turn_rows if not t["parser_error"]), key=lambda t: t["turn"])
        groups[(model, experiment, mode, instance_id)] = valid_turns

    total_pairs = sum(len(turns) * (len(turns) - 1) // 2 for turns in groups.values())
    print(
        f"Scoring retrodiction across {len(groups)} Lose instances, up to {total_pairs} (turn, earlier-turn) "
        f"pairs against the Docker sandbox..."
    )

    records = []
    pairs_done = 0
    for (model, experiment, mode, instance), turns in groups.items():
        instance_records = score_instance_turns(model, experiment, mode, instance, turns, "loss")
        records.extend(instance_records)
        pairs_done += sum(r["n_prev"] for r in instance_records)
        print(f"  ...{instance} done ({pairs_done}/{total_pairs} pairs so far)")
    return records


def analyze(results_root: Path, in_dir: Path, out_dir: Path):
    turn_details_path = in_dir / "turn_details.json"
    if not turn_details_path.exists():
        raise SystemExit(f"{turn_details_path} not found -- run analyze_results.py first.")
    summary_path = in_dir / "instance_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found -- run analyze_results.py first.")

    groups = group_instances(_load(turn_details_path))
    outcomes = build_outcome_lookup(_load(summary_path)["instances"])
    total_pairs = sum(len(turns) * (len(turns) - 1) // 2 for turns in groups.values())
    print(
        f"Scoring retrodiction across {len(groups)} Success instances, up to {total_pairs} (turn, earlier-turn) "
        f"pairs against the Docker sandbox..."
    )

    records = []
    pairs_done = 0
    for (model, experiment, mode, instance), turns in groups.items():
        outcome = outcomes.get((model, experiment, mode, instance))
        instance_records = score_instance_turns(model, experiment, mode, instance, turns, outcome)
        records.extend(instance_records)
        pairs_done += sum(r["n_prev"] for r in instance_records)
        print(f"  ...{instance} done ({pairs_done}/{total_pairs} pairs so far)")

    records.extend(collect_loss_retrodiction_records(results_root))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(records, out_dir / "retrodiction_details.json")

    df = pd.DataFrame(records)
    scoreable_df = df[df["retrodiction_accuracy"].notna()]

    if scoreable_df.empty:
        print("No turn ever had a scoreable earlier turn to retrodict -- skipping aggregation.")
        for suffix in ("", "_by_experiment"):
            _write_json([], out_dir / f"retrodiction_aggregates{suffix}.json")
        _write_json([], out_dir / "retrodiction_aggregates_by_outcome.json")
        return

    # Mode/experiment breakdowns stay Success-only, same population as before
    # Loss instances were scoreable at all -- these two answer "how does
    # retrodiction accuracy vary by mode/experiment", not "by outcome", and
    # pooling Loss in here would silently change their meaning.
    success_scoreable_df = scoreable_df[scoreable_df["outcome"] == "success"]
    for group_key, suffix in (("mode", ""), ("experiment", "_by_experiment")):
        agg = (
            success_scoreable_df.groupby(["model", group_key, "turn"])
            .agg(n=("retrodiction_accuracy", "size"), mean_retrodiction_accuracy=("retrodiction_accuracy", "mean"))
            .reset_index()
            .sort_values(["model", group_key, "turn"])
        )
        _write_json(json.loads(agg.to_json(orient="records")), out_dir / f"retrodiction_aggregates{suffix}.json")

    # Success vs. loss instead of mode/experiment -- aborted instances excluded,
    # same reasoning as turn_level_summary_analysis.py (neither outcome applies).
    outcome_df = scoreable_df[scoreable_df["outcome"].isin(["success", "loss"])]
    if outcome_df.empty:
        _write_json([], out_dir / "retrodiction_aggregates_by_outcome.json")
    else:
        outcome_agg = (
            outcome_df.groupby(["model", "outcome", "turn"])
            .agg(n=("retrodiction_accuracy", "size"), mean_retrodiction_accuracy=("retrodiction_accuracy", "mean"))
            .reset_index()
            .sort_values(["model", "outcome", "turn"])
        )
        _write_json(
            json.loads(outcome_agg.to_json(orient="records")), out_dir / "retrodiction_aggregates_by_outcome.json",
        )

    print("\nMean retrodiction accuracy by (model, mode, turn) -- Success instances only:")
    mean_by_mode = success_scoreable_df.groupby(["model", "mode", "turn"])["retrodiction_accuracy"].mean()
    print(mean_by_mode.rename(index=display_label, level="mode").to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing turn_details.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the JSON files (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    analyze(Path(args.results_root), in_dir, out_dir)


if __name__ == "__main__":
    main()
