#!/usr/bin/env python3
"""
Run the core function_detective analysis pipeline end-to-end, in the
required dependency order, from one command:

    analyze_results.py
    -> success_rate_analysis.py, turns_used_analysis.py, hypothesis_retrodiction_analysis.py
    -> last_turn_analysis.py
    -> plot_results.py, plot_success_rate.py, plot_turns_used.py, plot_hypothesis_retrodiction.py
    -> plot_last_turn_analysis.py

This is the "hand this to someone and say just run this" subset -- enough
to get the core turn-level charts, success rate, average turns,
retrodiction accuracy, and the last-turn confidence/correctness heatmap,
without anyone needing to know the run order or which script reads which
other script's JSON output. Everything else in this analysis/ folder
(hypothesis_consistency_analysis.py, confidence_hypothesis_analysis.py, the
coherence/correctness grid and quadrant, key_phrases.py, word_cloud.py,
...) is intentionally left out of this file -- run those individually,
only when actually wanted, exactly as before. They aren't needed to
produce the files above, so bundling them in here would just make this
slower for no benefit.

Each step is called directly (imported and invoked in-process, not
subprocess'd out to a separate `python3 script.py` call), so this only
ever needs one Python interpreter and one process. The only input is
results_root, optional -- defaults to <this file's parent>/../results if
not given, same convention as every other script's DEFAULT_RESULTS_ROOT in
this folder (they default to results_1; this one defaults to plain
results/, since that's the one folder meant to be handed to someone else
without them needing to know which results_N to point at). Output goes to
<results_root>/analysis either way.

hypothesis_retrodiction_analysis.py is by far the slowest step (it calls
a Docker sandbox up to roughly n*(n-1)/2 times per instance, now for both
Success and Loss instances) -- expect this to dominate the total runtime,
from under a minute on a small results folder to several minutes on a
large one. last_turn_analysis.py runs right after it since it reads
retrodiction_details.json, but makes no sandbox calls of its own (Loss
instances' confidence is read directly from interactions.json), so it's fast.

Usage:
    python run_analysis_pipeline.py [results_root]
"""
import argparse
import sys
import time
from pathlib import Path

# All the modules imported below live alongside this file -- make sure that's
# on sys.path regardless of the working directory this script is invoked from.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_RESULTS_ROOT = str(SCRIPT_DIR.parent / "results")

import analyze_results  # noqa: E402
import hypothesis_retrodiction_analysis  # noqa: E402
import last_turn_analysis  # noqa: E402
import plot_hypothesis_retrodiction  # noqa: E402
import plot_last_turn_analysis  # noqa: E402
import plot_results  # noqa: E402
import plot_success_rate  # noqa: E402
import plot_turns_used  # noqa: E402
import success_rate_analysis  # noqa: E402
import turns_used_analysis  # noqa: E402


def run(results_root: Path):
    out_dir = results_root / "analysis"

    # (label, callable) -- analysis steps take (source_dir, out_dir), plot
    # steps take (in_dir, out_dir); every step after the first reads from
    # and writes to the same <results_root>/analysis directory.
    steps = [
        ("analyze_results", lambda: analyze_results.analyze(results_root, out_dir)),
        ("success_rate_analysis", lambda: success_rate_analysis.analyze(out_dir, out_dir)),
        ("turns_used_analysis", lambda: turns_used_analysis.analyze(out_dir, out_dir)),
        ("hypothesis_retrodiction_analysis", lambda: hypothesis_retrodiction_analysis.analyze(results_root, out_dir, out_dir)),
        ("last_turn_analysis", lambda: last_turn_analysis.analyze(results_root, out_dir, out_dir)),
        ("plot_results", lambda: plot_results.plot(out_dir, out_dir)),
        ("plot_success_rate", lambda: plot_success_rate.plot(out_dir, out_dir)),
        ("plot_turns_used", lambda: plot_turns_used.plot(out_dir, out_dir)),
        ("plot_hypothesis_retrodiction", lambda: plot_hypothesis_retrodiction.plot(out_dir, out_dir)),
        ("plot_last_turn_analysis", lambda: plot_last_turn_analysis.plot(out_dir, out_dir)),
    ]

    overall_start = time.monotonic()
    for name, step in steps:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        step_start = time.monotonic()
        step()
        print(f"[{name} done in {time.monotonic() - step_start:.1f}s]")

    print(f"\nAll done in {time.monotonic() - overall_start:.1f}s. Output in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help="Root folder containing experiment subfolders of instance_* dirs "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    args = parser.parse_args()

    run(Path(args.results_root).resolve())


if __name__ == "__main__":
    main()
