#!/usr/bin/env python3
"""
TBU (Turn Budget Usage) score per model -- for Success instances and for
Lose instances separately, at three levels of grouping: overall (every
mode/experiment pooled), by_mode, and by_experiment. New metric, sibling to
efficiency_analysis.py's "efficiency" (the game-computed field read straight
from interactions.json) -- TBU is instead derived here from each instance's
raw turn transcript:

    tbu = len(turns) / TOTAL_TURN_BUDGET

turns is interactions.json's own "turns" list (one entry per round actually
played, including a final SOLVE/submission round for instances that end by
solving or explicitly giving up) -- verified directly: an instance that
exhausts its full budget has len(turns) == max_turns == 15 exactly, no
off-by-one correction needed. TOTAL_TURN_BUDGET is 15, matching
plot_solution_vs_truth_grid.MAX_TURNS -- a score of 1.0 means the model used
its entire turn budget.

This intentionally does NOT use turns_used (the field
efficiency_analysis.py-style scripts would reach for first): turns_used
turned out to only count turns whose response parsed successfully, silently
excluding parser-error turns even though a parser-error turn still consumes
one of the 15 budgeted turns. Confirmed directly against raw data -- e.g. an
instance with parse_error_count == 15 (every single turn a parser error) has
turns_used == 0 despite its "turns" transcript running the full 15-turn
budget; another instance with 2 parser-error turns out of 15 has
turns_used == 13, exactly the count of turns that parsed, not 15. Across the
full paper-analysis results set turns_used undercounts len(turns) by
anywhere from 0 to 15 turns, correlated with parse_error_count -- not a
clean, correctable offset, so turns_used cannot be used for a "how much of
the budget did this model actually spend" metric: it would make a model that
wasted turns on invalid output look like it spent LESS budget, backwards
from what actually happened.

Single-Turn ("_oneshot") modes are excluded entirely, not just from the
plot -- there's no real turn-budget decision to measure there (no turns to
spend), same reasoning plot_efficiency_by_variant.py already applies when
filtering its x-axis.

Reads every interactions.json directly (via analyze_results.find_instance_files/
get_model_name, not reimplemented here) rather than instance_summary.json,
since the turn count this needs (len(turns)) was never carried into that
file. Aborted instances are excluded, same reasoning as efficiency_analysis.py
(neither outcome applies to them).

Output JSON (default: <results_root>/analysis):
    tbu_summary.json
        {
          "by_model": {
            "<model>": {
              "overall":       {"success": {...}, "loss": {...}},
              "by_mode":       {"<mode>": {"success": {...}, "loss": {...}}, ...},
              "by_experiment": {"<experiment>": {"success": {...}, "loss": {...}}, ...}
            }
          }
        }
        where each {...} is {n, mean_tbu}, or omitted if that
        (model, group, outcome) combination has zero instances.

Usage:
    python tbu_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_results import find_instance_files, get_model_name
from display_names import MODE_FAMILY

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")
TOTAL_TURN_BUDGET = 15  # matches plot_solution_vs_truth_grid.MAX_TURNS


def _write_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote {path}")


def collect_rows(results_root: Path):
    rows = []
    for interactions_path in find_instance_files(results_root):
        with open(interactions_path) as f:
            data = json.load(f)

        mode = data.get("mode") or ""
        if MODE_FAMILY.get(mode) == "single_turn":
            continue
        if data.get("Aborted"):
            continue
        if not (data.get("Success") or data.get("Lose")):
            continue

        rows.append(
            {
                "model": get_model_name(data),
                "experiment": interactions_path.parent.parent.name,
                "mode": mode,
                "outcome": "success" if data.get("Success") else "loss",
                "tbu": len(data.get("turns", [])) / TOTAL_TURN_BUDGET,
            }
        )
    return rows


def _tbu_stats(rows):
    return {
        "n": len(rows),
        "mean_tbu": round(float(rows["tbu"].mean()), 4),
    }


def _stats_by_outcome(df):
    stats = {}
    for outcome in ("success", "loss"):
        rows = df.loc[df["outcome"] == outcome]
        if not rows.empty:
            stats[outcome] = _tbu_stats(rows)
    return stats


def analyze(results_root: Path, out_dir: Path):
    rows = collect_rows(results_root)
    if not rows:
        raise SystemExit("No Success/Lose non-single-turn instances found -- nothing to summarize.")

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
    _write_json({"by_model": by_model}, out_dir / "tbu_summary.json")

    print("\nOverall mean TBU score by model and outcome:")
    for model, model_stats in by_model.items():
        for outcome, stats in model_stats["overall"].items():
            print(f"  {model} / {outcome}: n={stats['n']} mean_tbu={stats['mean_tbu']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to analyze_results.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the JSON file (default: <results_root>/analysis)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "analysis"
    analyze(results_root, out_dir)


if __name__ == "__main__":
    main()
