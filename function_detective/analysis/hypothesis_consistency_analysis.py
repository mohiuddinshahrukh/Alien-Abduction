#!/usr/bin/env python3
"""
Does the player's stated hypothesis actually agree with their own stated
guess, and is it actually right?

For each non-parser-error turn, take the turn's `input`, `predicted_output`,
`gm_output` (the GM's ground truth, logged as a string), and `hypothesis`
(a Python function stub, e.g. "def f(x): ... return ..."). Execute the
hypothesis's code -- via function_detective's own sandbox
(utils.validate_function_logic, the same Docker-isolated runner the game
itself uses to score a real SOLVE submission -- not a reimplementation) --
on the turn's input, and compare the result against two different targets:

    consistency_score  hypothesis(input) vs. predicted_output -- does the
                        player's own code agree with the answer they gave?
                        (self-consistency: are they internally coherent)
    correctness_score   hypothesis(input) vs. gm_output -- does the
                        player's code actually implement the real function?
                        (ground-truth correctness: is their theory right)

Both share the same scale and the same pre-checks (see below), computed
together from the same sandbox call setup so a hypothesis that's unknown
or malformed gets the same -1/-100 on both without extra sandbox calls;
gm_output is parsed back to a native value with ast.literal_eval() (it's
always produced by this game's own format_value(), which for every type
used here -- int/float/bool/str/list/None -- is exactly literal_eval's
inverse) so it can be compared the same way predicted_output is.

Score per turn:
    -1   hypothesis is literally just the string "unknown" -- no code at
         all. Deliberately narrower than analyze_results.py's
         is_hypothesis_unknown() (used elsewhere for "has the player
         stated any belief yet"): here, if there's a `def ...` at all,
         it's code and gets executed and compared like any other, even if
         the body is trivial (e.g. `return 0` next to a stale "unknown"
         docstring) -- a trivial body that doesn't match the target is a
         real mismatch, not an absence of a hypothesis.
    -100 hypothesis is not executable. Checked two ways: first, whether the
         hypothesis's own declared signature's arity matches the number of
         arguments in `input` -- decided purely from the signature text
         (e.g. "def f(a, b):" -> arity 2), never guessed from the
         experiment name, since a list `input` can mean either "one list
         argument" (arity 1) or "several scalar arguments" (arity >1) and
         only the signature disambiguates which. A mismatch is reported as
         not-executable without ever calling the sandbox. If the arity
         matches, the sandbox is called, and a syntax/runtime error there
         also reports not-executable.
    0    the code runs fine but returns something other than the target
    1    the code runs and its result matches the target

Mean score is written for reference/filtering, but is a poor summary
statistic on its own -- a single -100 swamps any average it appears in.
The label share per turn (unknown / not_executable / mismatch / match) is
the intended read, plotted the same way as mode_label_share_by_turn.png
(see plot_hypothesis_consistency.py).

This calls out to `docker run` up to twice per scored turn (once per
target, skipped entirely for turns caught by the pre-checks), so it's
noticeably slower than the other analysis scripts here.

Output JSON files (default: <results_root>/analysis, same as analyze_results.py):
    hypothesis_consistency.json                  flat list, one record per scored turn,
                                                  with both consistency_* and correctness_* fields
    consistency_aggregates.json                  (model, mode, turn) -> n, mean_consistency_score
    consistency_aggregates_by_experiment.json    same, grouped by experiment
    consistency_label_shares.json                (model, mode, turn, consistency_label) ->
                                                  count, n_total, share
    consistency_label_shares_by_experiment.json  same, grouped by experiment
    correctness_aggregates{,_by_experiment}.json      same shape as consistency_aggregates, for correctness_score
    correctness_label_shares{,_by_experiment}.json    same shape as consistency_label_shares, for correctness_label

Usage:
    python hypothesis_consistency_analysis.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

FUNCTION_DETECTIVE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FUNCTION_DETECTIVE_DIR))
from utils import validate_function_logic  # noqa: E402 -- function_detective's own sandbox runner

from analyze_results import HYPOTHESIS_SIGNATURE_RE  # noqa: E402 -- reuse the same signature regex

DEFAULT_RESULTS_ROOT = str(FUNCTION_DETECTIVE_DIR / "results_1")

LABELS = {-1: "unknown", -100: "not_executable", 0: "mismatch", 1: "match"}


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(records, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def flatten_turn_details(nested):
    rows = []
    for model, by_experiment in nested.items():
        for experiment, by_mode in by_experiment.items():
            for mode, by_instance in by_mode.items():
                for instance, turns in by_instance.items():
                    for turn in turns:
                        if turn["parser_error"]:
                            continue
                        rows.append(
                            {"model": model, "experiment": experiment, "mode": mode, "instance": instance, **turn}
                        )
    return rows


def _check_executable(hypothesis, input_value):
    """
    Returns (short_circuit_score, args, detail):
      - (-1 or -100, None, detail) if there's no point calling the sandbox
        at all (bare "unknown", no signature, or arity mismatch) -- the
        same verdict applies to both consistency and correctness.
      - (None, args, None) if the hypothesis has a valid signature whose
        arity matches input_value's argument count, where args is what to
        pass as the sandbox call's positional arguments.
    """
    if isinstance(hypothesis, str) and hypothesis.strip().lower() == "unknown":
        return -1, None, None

    sig_match = HYPOTHESIS_SIGNATURE_RE.search(hypothesis) if isinstance(hypothesis, str) else None
    if not sig_match:
        return -100, None, "no function signature found in hypothesis text"

    arity = len([p for p in sig_match.group(1).split(",") if p.strip()])
    if arity == 1:
        # A single argument is a single argument regardless of its own shape --
        # even if input is itself a list, that's the one argument's value.
        return None, [input_value], None
    if isinstance(input_value, list) and len(input_value) == arity:
        return None, list(input_value), None
    return -100, None, f"signature expects {arity} argument(s) but input is {input_value!r}"


def _compare_against(hypothesis, args, expected_value):
    ok, _, message = validate_function_logic(hypothesis, [{"args": args, "expected": expected_value}])
    if ok:
        return 1, None
    if message.startswith("Runtime Error:") or "__ERROR__" in message:
        return -100, message
    return 0, message


def score_turn(hypothesis, input_value, predicted_output, gm_output):
    """Returns (consistency_score, consistency_detail, correctness_score, correctness_detail)."""
    pre_score, args, pre_detail = _check_executable(hypothesis, input_value)
    if pre_score is not None:
        return pre_score, pre_detail, pre_score, pre_detail

    consistency_score, consistency_detail = _compare_against(hypothesis, args, predicted_output)

    try:
        gm_value = ast.literal_eval(gm_output) if isinstance(gm_output, str) else gm_output
    except (ValueError, SyntaxError) as exc:
        correctness_score, correctness_detail = -100, f"could not parse gm_output {gm_output!r}: {exc}"
    else:
        correctness_score, correctness_detail = _compare_against(hypothesis, args, gm_value)

    return consistency_score, consistency_detail, correctness_score, correctness_detail


def write_score_outputs(df, score_col, label_col, out_dir, prefix):
    """Writes {prefix}_aggregates{,_by_experiment}.json and {prefix}_label_shares{,_by_experiment}.json."""
    for group_key, suffix in (("mode", ""), ("experiment", "_by_experiment")):
        agg = (
            df.groupby(["model", group_key, "turn"])
            .agg(n=(score_col, "size"), **{f"mean_{score_col}": (score_col, "mean")})
            .reset_index()
            .sort_values(["model", group_key, "turn"])
        )
        _write_json(json.loads(agg.to_json(orient="records")), out_dir / f"{prefix}_aggregates{suffix}.json")

        label_counts = df.groupby(["model", group_key, "turn", label_col]).size().rename("count").reset_index()
        totals = df.groupby(["model", group_key, "turn"]).size().rename("n_total")
        label_counts = label_counts.join(totals, on=["model", group_key, "turn"])
        label_counts["share"] = label_counts["count"] / label_counts["n_total"]
        label_counts = label_counts.sort_values(["model", group_key, "turn", label_col])
        _write_json(
            json.loads(label_counts.to_json(orient="records")), out_dir / f"{prefix}_label_shares{suffix}.json"
        )


def analyze(in_dir: Path, out_dir: Path):
    turn_details_path = in_dir / "turn_details.json"
    if not turn_details_path.exists():
        raise SystemExit(f"{turn_details_path} not found -- run analyze_results.py first.")

    rows = flatten_turn_details(_load(turn_details_path))
    if not rows:
        raise SystemExit(f"{turn_details_path} has no non-parser-error turns.")

    print(f"Scoring {len(rows)} turns against the Docker sandbox (up to two `docker run` calls per turn)...")
    records = []
    for i, row in enumerate(rows, start=1):
        consistency_score, consistency_detail, correctness_score, correctness_detail = score_turn(
            row["hypothesis"], row["input"], row["predicted_output"], row["gm_output"]
        )
        records.append(
            {
                "model": row["model"], "experiment": row["experiment"], "mode": row["mode"],
                "instance": row["instance"], "turn": row["turn"],
                "consistency_score": consistency_score, "consistency_label": LABELS[consistency_score],
                "consistency_detail": consistency_detail,
                "correctness_score": correctness_score, "correctness_label": LABELS[correctness_score],
                "correctness_detail": correctness_detail,
                "input": row["input"], "predicted_output": row["predicted_output"],
                "gm_output": row["gm_output"], "hypothesis": row["hypothesis"],
            }
        )
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} scored")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(records, out_dir / "hypothesis_consistency.json")

    df = pd.DataFrame(records)
    write_score_outputs(df, "consistency_score", "consistency_label", out_dir, "consistency")
    write_score_outputs(df, "correctness_score", "correctness_label", out_dir, "correctness")

    print("\nConsistency label counts overall (hypothesis vs. predicted_output):")
    print(df["consistency_label"].value_counts().to_string())
    print("\nCorrectness label counts overall (hypothesis vs. gm_output):")
    print(df["correctness_label"].value_counts().to_string())


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
    analyze(in_dir, out_dir)


if __name__ == "__main__":
    main()
