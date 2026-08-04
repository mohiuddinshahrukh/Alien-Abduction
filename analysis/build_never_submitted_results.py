#!/usr/bin/env python3
"""
Among Lose episodes, how many never got a SOLVE accepted at all (source !=
"submitted_solution" from plot_solution_vs_truth_grid._extract_final_solution),
and of those, how many had a final hypothesis that was nonetheless 100%
correct against the held-out test_cases -- i.e. "knew the answer, never
submitted it" vs. "never converged." Broken down overall, per category, and
per mode, plus a breakdown of exactly which non-submission source
(evaldata_hypothesis / no_hypothesis / parser_error / executable_error /
solve_rejected_ok) each never-submitted episode falls into, since those are
different failure stories (never attempted SOLVE at all vs. attempted one
that didn't parse/exec).

Correctness is checked by executing the extracted hypothesis against every
one of the instance's own held-out test_cases (from instances.json) --
nothing precomputed, read fresh every run, same as the other scripts here.

Output:
    never_submitted_results.json
        {overall: {model: {...}}, by_category: {category: {model: {...}}},
         by_mode: {mode: {model: {...}}}}
    each {...} is: lose_total, never_submitted, never_submitted_pct,
    never_submitted_100pct_correct, never_submitted_100pct_correct_pct,
    never_submitted_by_source: {source: count}

Usage:
    python build_never_submitted_results.py <results_parent> [--out-dir OUT_DIR]

    results_parent must contain one subfolder per model (see
    display_names.iter_model_dirs), and its *parent* directory must contain
    function_detective/in/instances.json (i.e. results_parent is the game's
    own results/ folder, e.g. .../Alien-Abduction/results).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import plot_solution_vs_truth_grid as svt
from display_names import iter_model_dirs, sort_models

MODEL_DISPLAY_KEY_ALIASES = {"mistral-large-3": "mistral-large-3-azure"}


def _sort_model_dir_names(names):
    alias_to_raw = {MODEL_DISPLAY_KEY_ALIASES.get(n, n): n for n in names}
    return [alias_to_raw[a] for a in sort_models(list(alias_to_raw.keys()))]


def _new_bucket():
    return {
        "lose_total": 0,
        "never_submitted": 0,
        "never_submitted_100pct_correct": 0,
        "never_submitted_by_source": defaultdict(int),
    }


def _finalize(bucket: dict) -> dict:
    lose_total = bucket["lose_total"]
    never_submitted = bucket["never_submitted"]
    correct = bucket["never_submitted_100pct_correct"]
    return {
        "lose_total": lose_total,
        "never_submitted": never_submitted,
        "never_submitted_pct": round(100 * never_submitted / lose_total, 1) if lose_total else None,
        "never_submitted_100pct_correct": correct,
        "never_submitted_100pct_correct_pct": round(100 * correct / never_submitted, 1) if never_submitted else None,
        "never_submitted_by_source": dict(bucket["never_submitted_by_source"]),
    }


def build(results_parent: Path, instances_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _sort_model_dir_names([p.name for p in iter_model_dirs(results_parent)])
    experiments = json.load(open(instances_path))["experiments"]

    overall = defaultdict(_new_bucket)
    by_category = defaultdict(lambda: defaultdict(_new_bucket))
    by_mode = defaultdict(lambda: defaultdict(_new_bucket))

    for exp in experiments:
        exp_name = exp["name"]
        category = exp["game_instances"][0]["category"]
        mode = exp["game_instances"][0]["mode"]

        for gi in exp["game_instances"]:
            game_id = gi["game_id"]
            test_cases = gi["test_cases"]

            for model in models:
                path = results_parent / model / "results" / exp_name / f"instance_{game_id:05d}" / "interactions.json"
                if not path.exists():
                    continue
                interactions = json.load(open(path))
                if not interactions.get("Lose", False):
                    continue

                fn, source = svt._extract_final_solution(interactions)
                buckets = (overall[model], by_category[category][model], by_mode[mode][model])
                for b in buckets:
                    b["lose_total"] += 1

                if source == "submitted_solution":
                    continue
                for b in buckets:
                    b["never_submitted"] += 1
                    b["never_submitted_by_source"][source] += 1

                if fn is None:
                    continue
                try:
                    correct = sum(1 for tc in test_cases if fn(*tc["args"]) == tc["expected"])
                except Exception:
                    continue
                if correct == len(test_cases):
                    for b in buckets:
                        b["never_submitted_100pct_correct"] += 1

    result = {
        "overall": {model: _finalize(overall[model]) for model in models if overall[model]["lose_total"]},
        "by_category": {
            cat: {model: _finalize(by_category[cat][model]) for model in models if by_category[cat][model]["lose_total"]}
            for cat in by_category
        },
        "by_mode": {
            mode: {model: _finalize(by_mode[mode][model]) for model in models if by_mode[mode][model]["lose_total"]}
            for mode in by_mode
        },
    }

    out_path = out_dir / "never_submitted_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")


def run(results_parent: Path, out_dir: Path):
    instances_path = results_parent.parent / "function_detective" / "in" / "instances.json"
    build(results_parent, instances_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Results folder containing one subfolder per model (each with results/ and "
        "analysis/ subfolders); its parent must contain function_detective/in/instances.json",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the JSON file (default: <results_parent>/paper_plots/never_submitted_results)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots" / "never_submitted_results"
    run(results_parent, out_dir)


if __name__ == "__main__":
    main()
