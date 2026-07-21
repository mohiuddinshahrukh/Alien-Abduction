#!/usr/bin/env python3
"""
Success rate per experiment, grouped by mode -- what fraction of instances
in each domain (numbers/list/logic/string/two_numbers/...) actually solved,
and does that vary by mode (active_inputs vs. passive_examples vs. ...)?

Reads instance_summary.json's "instances" section (from analyze_results.py).
Unlike turns_used_analysis.py, this counts every instance regardless of
outcome (Success/Lose/Aborted all count toward the denominator) -- success
rate is exactly n_success / n_total, so it needs the full outcome set, not
just the Success ones. Oneshot modes are NOT excluded here (unlike
turns_used_analysis.py): a oneshot instance's single guess is either right
or wrong, so success rate is just as meaningful there as anywhere else.

Output JSON (default: <results_root>/analysis, same as analyze_results.py):
    success_rate_by_experiment.json   one record per (model, mode, experiment):
                                       n, n_success, success_rate (fraction 0-1)

Usage:
    python success_rate_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from display_names import display_label

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(records, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def flatten_instances(instances):
    rows = []
    for model, by_experiment in instances.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, summary in by_instance.items():
                    rows.append(
                        {
                            "model": model, "experiment": experiment, "mode": mode,
                            "Success": summary["Success"],
                        }
                    )
    return rows


def analyze(in_dir: Path, out_dir: Path):
    summary_path = in_dir / "instance_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found -- run analyze_results.py first.")

    data = _load(summary_path)
    rows = flatten_instances(data["instances"])
    if not rows:
        raise SystemExit("No instances found.")

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["model", "mode", "experiment"])
        .agg(n=("Success", "size"), n_success=("Success", "sum"))
        .reset_index()
        .sort_values(["model", "mode", "experiment"])
    )
    agg["success_rate"] = (agg["n_success"] / agg["n"]).round(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(json.loads(agg.to_json(orient="records")), out_dir / "success_rate_by_experiment.json")

    print("\nSuccess rate by (model, mode, experiment):")
    display_agg = agg.copy()
    display_agg["mode"] = display_agg["mode"].map(display_label)
    display_agg["experiment"] = display_agg["experiment"].map(display_label)
    print(display_agg.to_string(index=False))


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
