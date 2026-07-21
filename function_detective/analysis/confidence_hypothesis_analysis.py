#!/usr/bin/env python3
"""
Correlate confidence_score, hypothesis commitment, and match correctness
from analyze_results.py's turn_details.json.

Three analyses, each grouped by (model, mode) and again by (model, experiment):

    1. Calibration -- is confidence trustworthy? Turns are binned into
       confidence buckets (0-20, 20-40, ..., 80-100) and, within each
       bucket, split further by whether the player had already committed to
       a hypothesis (hypothesis != "unknown") at that turn. If confidence is
       well-calibrated, the actual match rate in the 80-100 bucket should be
       near 80-100% -- and that should hold whether or not a hypothesis has
       been committed to yet. If the committed/uncommitted split tells a
       different calibration story, confidence means something different
       depending on whether the player has a real guess behind it.

    2. Confident-but-wrong / unsure-but-right -- turns where stated
       confidence and actual correctness sharply disagree (confidence >=
       --high-confidence but match is False, or confidence <= --low-confidence
       but match is True). A mean-confidence-by-turn line averages these
       away entirely; this pulls them out individually with full context
       (hypothesis, input_rationale, current_mode_label) so you can read
       what the player was thinking when it went wrong.

    3. Commitment effect -- match rate and mean confidence split only by
       hypothesis commitment (collapsing over confidence buckets) -- does
       committing to *any* hypothesis actually correlate with being right,
       or do players commit prematurely?

parser_error turns are excluded throughout (no confidence/match to analyze
on a turn that didn't parse).

Output JSON files (default: <results_root>/analysis, same as analyze_results.py):
    calibration.json                    (model, mode, hypothesis_committed,
                                          confidence_bucket) -> n, match_rate, mean_confidence
    calibration_by_experiment.json      same, grouped by experiment instead of mode
    anomalies.json                      flat list of confident-wrong / unsure-right turns,
                                         with hypothesis/input_rationale/current_mode_label context
    anomaly_summary.json                (model, mode) -> n_turns, n_confident_wrong, n_unsure_right,
                                         confident_wrong_rate, unsure_right_rate
    anomaly_summary_by_experiment.json  same, grouped by experiment
    commitment_effect.json              (model, mode, hypothesis_committed) ->
                                         n, match_rate, mean_confidence
    commitment_effect_by_experiment.json  same, grouped by experiment

Usage:
    python confidence_hypothesis_analysis.py [results_root] [--out-dir OUT_DIR]
        [--high-confidence N] [--low-confidence N]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

CONFIDENCE_BUCKETS = [0, 20, 40, 60, 80, 100]
CONFIDENCE_BUCKET_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(records, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def flatten_turn_details(nested):
    """turn_details.json is nested model -> experiment -> mode -> instance -> [turns];
    re-attach the grouping keys and drop to a flat list for pandas, excluding
    parser_error turns (no confidence/match there)."""
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


def add_derived_columns(df):
    df = df.copy()
    df["hypothesis_committed"] = ~df["hypothesis_unknown"].astype(bool)
    df["confidence_bucket"] = pd.cut(
        df["confidence_score"], bins=CONFIDENCE_BUCKETS, labels=CONFIDENCE_BUCKET_LABELS, include_lowest=True,
    )
    return df


def compute_calibration(df, group_key):
    return (
        df.groupby(["model", group_key, "hypothesis_committed", "confidence_bucket"], observed=True)
        .agg(n=("match", "size"), match_rate=("match", "mean"), mean_confidence=("confidence_score", "mean"))
        .reset_index()
        .sort_values(["model", group_key, "hypothesis_committed", "confidence_bucket"])
    )


def compute_commitment_effect(df, group_key):
    return (
        df.groupby(["model", group_key, "hypothesis_committed"], observed=True)
        .agg(n=("match", "size"), match_rate=("match", "mean"), mean_confidence=("confidence_score", "mean"))
        .reset_index()
        .sort_values(["model", group_key, "hypothesis_committed"])
    )


def compute_anomalies(df, high_confidence, low_confidence):
    confident_wrong = df[(df["confidence_score"] >= high_confidence) & (~df["match"])].copy()
    confident_wrong["anomaly_type"] = "confident_wrong"
    unsure_right = df[(df["confidence_score"] <= low_confidence) & (df["match"])].copy()
    unsure_right["anomaly_type"] = "unsure_right"

    anomalies = pd.concat([confident_wrong, unsure_right], ignore_index=True)
    if anomalies.empty:
        return anomalies
    anomalies = anomalies.sort_values(["model", "experiment", "instance", "turn"])
    cols = [
        "model", "experiment", "mode", "instance", "turn", "anomaly_type",
        "confidence_score", "match", "predicted_output", "gm_output",
        "current_mode_label", "hypothesis", "input_rationale",
    ]
    return anomalies[cols]


def compute_anomaly_summary(df, anomalies, group_key, high_confidence, low_confidence):
    total = df.groupby(["model", group_key]).size().rename("n_turns")
    eligible_high = (
        df[df["confidence_score"] >= high_confidence].groupby(["model", group_key]).size()
        .rename("n_high_confidence_turns")
    )
    eligible_low = (
        df[df["confidence_score"] <= low_confidence].groupby(["model", group_key]).size()
        .rename("n_low_confidence_turns")
    )

    if not anomalies.empty:
        counts = (
            anomalies.groupby(["model", group_key, "anomaly_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["confident_wrong", "unsure_right"], fill_value=0)
        )
    else:
        counts = pd.DataFrame(columns=["confident_wrong", "unsure_right"])

    summary = pd.concat([total, eligible_high, eligible_low, counts], axis=1).fillna(0).reset_index()
    summary["n_confident_wrong"] = summary.get("confident_wrong", 0)
    summary["n_unsure_right"] = summary.get("unsure_right", 0)
    summary["confident_wrong_rate"] = summary["n_confident_wrong"] / summary["n_high_confidence_turns"].replace(0, pd.NA)
    summary["unsure_right_rate"] = summary["n_unsure_right"] / summary["n_low_confidence_turns"].replace(0, pd.NA)
    summary = summary.drop(columns=["confident_wrong", "unsure_right"], errors="ignore")
    int_cols = ["n_turns", "n_high_confidence_turns", "n_low_confidence_turns", "n_confident_wrong", "n_unsure_right"]
    summary[int_cols] = summary[int_cols].astype(int)
    return summary


def analyze(in_dir: Path, out_dir: Path, high_confidence: int, low_confidence: int):
    turn_details_path = in_dir / "turn_details.json"
    if not turn_details_path.exists():
        raise SystemExit(f"{turn_details_path} not found -- run analyze_results.py first.")

    nested = _load(turn_details_path)
    rows = flatten_turn_details(nested)
    if not rows:
        raise SystemExit(f"{turn_details_path} has no non-parser-error turns.")

    df = add_derived_columns(pd.DataFrame(rows))
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_key, suffix in (("mode", ""), ("experiment", "_by_experiment")):
        calib = compute_calibration(df, group_key)
        _write_json(json.loads(calib.to_json(orient="records")), out_dir / f"calibration{suffix}.json")

        effect = compute_commitment_effect(df, group_key)
        _write_json(json.loads(effect.to_json(orient="records")), out_dir / f"commitment_effect{suffix}.json")

    anomalies = compute_anomalies(df, high_confidence, low_confidence)
    _write_json(json.loads(anomalies.to_json(orient="records")), out_dir / "anomalies.json")

    for group_key, suffix in (("mode", ""), ("experiment", "_by_experiment")):
        summary = compute_anomaly_summary(df, anomalies, group_key, high_confidence, low_confidence)
        _write_json(json.loads(summary.to_json(orient="records")), out_dir / f"anomaly_summary{suffix}.json")

    n_confident_wrong = int((anomalies["anomaly_type"] == "confident_wrong").sum()) if not anomalies.empty else 0
    n_unsure_right = int((anomalies["anomaly_type"] == "unsure_right").sum()) if not anomalies.empty else 0
    print(f"\n{len(df)} non-parser-error turns analyzed")
    print(f"{len(anomalies)} anomalies found ({n_confident_wrong} confident-wrong, {n_unsure_right} unsure-right)")

    print("\nCalibration preview (mode-grouped):")
    print(compute_calibration(df, "mode").to_string(index=False))

    print("\nCommitment effect preview (mode-grouped):")
    print(compute_commitment_effect(df, "mode").to_string(index=False))


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
    parser.add_argument(
        "--high-confidence", type=int, default=70,
        help="confidence_score at/above this counts as 'confident' for the anomaly check (default: 70)",
    )
    parser.add_argument(
        "--low-confidence", type=int, default=30,
        help="confidence_score at/below this counts as 'unsure' for the anomaly check (default: 30)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    analyze(in_dir, out_dir, args.high_confidence, args.low_confidence)


if __name__ == "__main__":
    main()
