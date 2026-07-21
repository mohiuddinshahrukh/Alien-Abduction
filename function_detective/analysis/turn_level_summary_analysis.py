#!/usr/bin/env python3
"""
Turn-count distribution per model -- for Success instances, how many turns did
it typically take to solve; for Lose instances, how many turns did the player
get through before failing. Reports min/max/median/mean turns_used for each
outcome, at three levels of grouping: overall (every mode/experiment pooled),
by_mode, and by_experiment.

Reads instance_summary.json's "instances" section (from analyze_results.py).
Aborted instances are excluded entirely -- an abort is neither a solve nor a
failed guess, so its turns_used isn't comparable to either distribution.
Modes ending in "_oneshot" are also excluded, same reasoning as
turns_used_analysis.py: max_turns is capped at 1 there, so turns_used is
always 0 by construction regardless of outcome, not a real signal.

Output JSON (default: <results_root>/analysis, same as analyze_results.py):
    turn_level_summary.json
        {
          "by_model": {
            "<model>": {
              "overall":       {"success": {...}, "loss": {...}},
              "by_mode":       {"<mode>": {"success": {...}, "loss": {...}}, ...},
              "by_experiment": {"<experiment>": {"success": {...}, "loss": {...}}, ...}
            }
          }
        }
        where each {...} is {n, min_turns, max_turns, median_turns, mean_turns},
        or omitted if that (model, group, outcome) combination has zero instances.

Usage:
    python turn_level_summary_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote {path}")


def flatten_instances(instances):
    rows = []
    for model, by_experiment in instances.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                if mode.endswith("_oneshot"):
                    continue
                for instance, summary in by_instance.items():
                    if summary["Aborted"]:
                        continue
                    rows.append(
                        {
                            "model": model, "experiment": experiment, "mode": mode,
                            "outcome": "success" if summary["Success"] else "loss",
                            "turns_used": summary["turns_used"],
                        }
                    )
    return rows


def _turn_stats(turns):
    return {
        "n": len(turns),
        "min_turns": int(turns.min()),
        "max_turns": int(turns.max()),
        "median_turns": round(float(turns.median()), 2),
        "mean_turns": round(float(turns.mean()), 2),
    }


def _stats_by_outcome(df):
    stats = {}
    for outcome in ("success", "loss"):
        turns = df.loc[df["outcome"] == outcome, "turns_used"]
        if not turns.empty:
            stats[outcome] = _turn_stats(turns)
    return stats


def analyze(in_dir: Path, out_dir: Path):
    summary_path = in_dir / "instance_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found -- run analyze_results.py first.")

    data = _load(summary_path)
    rows = flatten_instances(data["instances"])
    if not rows:
        raise SystemExit("No Success/Lose instances found (outside oneshot modes) -- nothing to summarize.")

    df = pd.DataFrame(rows)

    by_model = {}
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        by_model[model] = {
            "overall": _stats_by_outcome(model_df),
            "by_mode": {
                mode: _stats_by_outcome(model_df[model_df["mode"] == mode])
                for mode in sorted(model_df["mode"].unique())
            },
            "by_experiment": {
                experiment: _stats_by_outcome(model_df[model_df["experiment"] == experiment])
                for experiment in sorted(model_df["experiment"].unique())
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json({"by_model": by_model}, out_dir / "turn_level_summary.json")

    print("\nOverall turns_used (min/max/median/mean) by model and outcome:")
    for model, model_stats in by_model.items():
        for outcome, stats in model_stats["overall"].items():
            print(
                f"  {model} / {outcome}: n={stats['n']} min={stats['min_turns']} "
                f"max={stats['max_turns']} median={stats['median_turns']} mean={stats['mean_turns']}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing instance_summary.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the JSON file (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    analyze(in_dir, out_dir)


if __name__ == "__main__":
    main()
