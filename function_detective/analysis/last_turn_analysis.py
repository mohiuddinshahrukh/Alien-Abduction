#!/usr/bin/env python3
"""
At the moment of decision (each instance's last turn before SOLVE/give-up),
pulls together each instance's last valid (non-parser-error) turn -- no new
sandbox calls, this is a join of analyze_results.py's turn_details.json and
hypothesis_retrodiction_analysis.py's retrodiction_details.json:

    confidence_score       the player's stated confidence at that last turn
    match                  predicted_output == gm_output at that last turn
                            (did they actually get the final guess right)
    retrodiction_accuracy   from retrodiction_details.json: does that same
                            turn's hypothesis correctly reproduce gm_output
                            for every earlier input in this instance, not
                            just the current one (None if the last turn was
                            turn 1 -- no earlier inputs to check, or if the
                            hypothesis had nothing scoreable to run against)

Output JSON files (default: <results_root>/analysis, same as analyze_results.py):
    last_turn_records.json                 flat list, one record per instance
    last_turn_heatmap.json                 per (model, mode): mean last-turn confidence_score,
                                            separately for Success and Loss instances (Aborted
                                            excluded), plus one "hypothesis correctness" value per
                                            mode used for both -- mean retrodiction_accuracy at the
                                            last turn for the four multi-turn modes, or success rate
                                            for the two oneshot/singleturn modes, where the last turn
                                            is always turn 1 and retrodiction_accuracy is always None
                                            (nothing earlier to retrodict against). See
                                            plot_last_turn_analysis.py's heatmap for how this is used.

                                            Loss instances' confidence_score is read directly from
                                            each Lose interactions.json (bypassing turn_details.json,
                                            which -- like everything built on top of it, e.g.
                                            retrodiction_details.json -- only ever covers Success
                                            instances; see analyze_results.py's load_instance). This
                                            reuses analyze_results.py's build_turn_row unchanged and
                                            makes no sandbox calls, so it's fast -- it does NOT compute
                                            retrodiction_accuracy for Loss instances (that would need
                                            the slow Docker sandbox pass), so the "hypothesis
                                            correctness" color stays Success-only for the four
                                            multi-turn modes exactly as before.

Usage:
    python last_turn_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_results import build_turn_row, find_instance_files, get_model_name
from hypothesis_retrodiction_analysis import build_outcome_lookup

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(records, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def find_last_turns(turn_details):
    """Returns {(model, experiment, mode, instance): last valid turn record}."""
    last_turns = {}
    for model, by_experiment in turn_details.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, turns in by_instance.items():
                    valid_turns = [t for t in turns if not t["parser_error"]]
                    if not valid_turns:
                        continue
                    last_turns[(model, experiment, mode, instance)] = max(valid_turns, key=lambda t: t["turn"])
    return last_turns


def build_last_turn_records(turn_details, retrodiction_records):
    last_turns = find_last_turns(turn_details)
    retrodiction_by_key = {
        (r["model"], r["experiment"], r["mode"], r["instance"], r["turn"]): r for r in retrodiction_records
    }

    records = []
    for (model, experiment, mode, instance), last_turn in last_turns.items():
        retrodiction = retrodiction_by_key.get((model, experiment, mode, instance, last_turn["turn"]))
        records.append(
            {
                "model": model, "experiment": experiment, "mode": mode, "instance": instance,
                "turn": last_turn["turn"], "confidence_score": last_turn["confidence_score"],
                "match": last_turn["match"],
                "n_prev": retrodiction["n_prev"] if retrodiction else 0,
                "n_scoreable": retrodiction["n_scoreable"] if retrodiction else 0,
                "retrodiction_accuracy": retrodiction["retrodiction_accuracy"] if retrodiction else None,
                "predicted_output": last_turn["predicted_output"], "gm_output": last_turn["gm_output"],
                "hypothesis": last_turn["hypothesis"],
            }
        )
    return records


def compute_success_rate_by_mode(instances):
    """
    {(model, mode): success_rate} -- every instance counts toward the
    denominator regardless of outcome, same definition as
    success_rate_analysis.py, just grouped by mode alone (pooling every
    experiment/domain that shares a mode) instead of by (mode, experiment).
    """
    rows = []
    for model, by_experiment in instances.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, summary in by_instance.items():
                    rows.append({"model": model, "mode": mode, "Success": summary["Success"]})
    df = pd.DataFrame(rows)
    return df.groupby(["model", "mode"])["Success"].mean().round(2).to_dict()


def collect_loss_last_turn_confidence(results_root: Path):
    """
    For every Lose instance, the confidence_score at its last valid
    (non-parser-error) turn -- read directly from interactions.json, since
    turn_details.json (and everything built from it, e.g.
    retrodiction_details.json) only ever covers Success instances (see
    analyze_results.py's load_instance). Reuses build_turn_row unchanged --
    it's outcome-agnostic, the Success-only restriction lives one level up
    -- and makes no sandbox calls, so this is fast even across hundreds of
    instances. Returns a flat list of
    {model, experiment, mode, instance, turn, confidence_score}.
    """
    records = []
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
        valid_turns = [t for t in turn_rows if not t["parser_error"]]
        if not valid_turns:
            continue
        last_turn = max(valid_turns, key=lambda t: t["turn"])
        records.append(
            {
                "model": model, "experiment": experiment, "mode": mode, "instance": instance_id,
                "turn": last_turn["turn"], "confidence_score": last_turn["confidence_score"],
            }
        )
    return records


def compute_heatmap_data(last_turn_df, instances, loss_last_turn_records):
    """
    Per (model, mode): mean last-turn confidence_score for Success instances
    and for Loss instances separately (Aborted excluded from both), plus one
    "hypothesis correctness" value per mode shared by both -- mean
    retrodiction_accuracy at the last turn for modes where that's defined,
    or success rate as a stand-in for the oneshot/singleturn modes where the
    last turn is always turn 1 and retrodiction_accuracy is always None.

    last_turn_df only ever contains Success instances (confirmed via
    build_outcome_lookup below rather than assumed), so it supplies the
    "success" side; loss_last_turn_records (from
    collect_loss_last_turn_confidence, a separate direct read of Lose
    instances' interactions.json) supplies the "loss" side.
    """
    outcomes = build_outcome_lookup(instances)
    df = last_turn_df.copy()
    df["outcome"] = df.apply(
        lambda r: outcomes.get((r["model"], r["experiment"], r["mode"], r["instance"])), axis=1,
    )

    success_confidence = (
        df[df["outcome"] == "success"].groupby(["model", "mode"])["confidence_score"].mean().round(2)
    )
    loss_df = pd.DataFrame(
        loss_last_turn_records, columns=["model", "experiment", "mode", "instance", "turn", "confidence_score"],
    )
    loss_confidence = (
        loss_df.groupby(["model", "mode"])["confidence_score"].mean().round(2)
        if not loss_df.empty else pd.Series(dtype=float)
    )

    retro_correctness = df[df["retrodiction_accuracy"].notna()].groupby(["model", "mode"])["retrodiction_accuracy"].mean()
    success_rate = compute_success_rate_by_mode(instances)

    all_model_modes = set(df[["model", "mode"]].itertuples(index=False, name=None))
    if not loss_df.empty:
        all_model_modes |= set(loss_df[["model", "mode"]].itertuples(index=False, name=None))

    by_model = {}
    for model, mode in sorted(all_model_modes):
        model_entry = by_model.setdefault(model, {"modes": [], "correctness_by_mode": {}, "confidence_by_mode_outcome": {"success": {}, "loss": {}}})
        model_entry["modes"].append(mode)

        if mode.endswith("_oneshot"):
            model_entry["correctness_by_mode"][mode] = {
                "value": success_rate.get((model, mode)), "source": "success_rate",
            }
        else:
            value = retro_correctness.get((model, mode))
            model_entry["correctness_by_mode"][mode] = {
                "value": round(value, 2) if value is not None else None, "source": "retrodiction",
            }

        for outcome, source in (("success", success_confidence), ("loss", loss_confidence)):
            value = source.get((model, mode))
            model_entry["confidence_by_mode_outcome"][outcome][mode] = value if value is None else float(value)

    return by_model


def analyze(results_root: Path, in_dir: Path, out_dir: Path):
    turn_details_path = in_dir / "turn_details.json"
    retrodiction_path = in_dir / "retrodiction_details.json"
    summary_path = in_dir / "instance_summary.json"
    if not turn_details_path.exists():
        raise SystemExit(f"{turn_details_path} not found -- run analyze_results.py first.")
    if not retrodiction_path.exists():
        raise SystemExit(f"{retrodiction_path} not found -- run hypothesis_retrodiction_analysis.py first.")
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found -- run analyze_results.py first.")

    records = build_last_turn_records(_load(turn_details_path), _load(retrodiction_path))
    if not records:
        raise SystemExit("No instances with a valid last turn found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(records, out_dir / "last_turn_records.json")

    df = pd.DataFrame(records)
    n_no_history = int(df["retrodiction_accuracy"].isna().sum())
    print(
        f"{len(df)} instances scored at their last turn "
        f"({n_no_history} with no retrodiction_accuracy -- solved on turn 1, or nothing scoreable)"
    )

    instances = _load(summary_path)["instances"]
    loss_records = collect_loss_last_turn_confidence(results_root)
    print(f"{len(loss_records)} Lose instances' last-turn confidence read directly from interactions.json")
    heatmap_data = compute_heatmap_data(df, instances, loss_records)
    _write_json(heatmap_data, out_dir / "last_turn_heatmap.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing turn_details.json, retrodiction_details.json, and "
        "instance_summary.json (default: <results_root>/analysis)",
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
