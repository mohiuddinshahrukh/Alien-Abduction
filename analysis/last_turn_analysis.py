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
                            just the current one. Raw value as
                            hypothesis_retrodiction_analysis.py computed it --
                            None if the last turn was turn 1 (no earlier
                            inputs to check at all) or if nothing was
                            scoreable, without distinguishing why.
    retrodiction_accuracy_corrected   the value actually used for aggregation
                            (see corrected_retrodiction) -- same as above,
                            except a hypothesis that had earlier turns to
                            check against but couldn't execute against any
                            of them (n_scoreable == 0) scores 0.0 here
                            instead of being silently excluded, and a bare
                            "unknown" hypothesis is excluded here but
                            flagged via hypothesis_unknown instead of just
                            vanishing.
    hypothesis_unknown     this last turn's hypothesis was the literal
                            string "unknown" -- nothing to retrodict,
                            reported separately (see
                            unknown_hypothesis_share_by_mode below) rather
                            than folded into retrodiction_accuracy_corrected.

Output JSON files (default: <results_root>/analysis, same as analyze_results.py):
    last_turn_records.json                 flat list, one record per instance
    last_turn_heatmap.json                 per (model, mode): mean last-turn confidence_score,
                                            separately for Success and Loss instances (Aborted
                                            excluded), plus a retrodiction-accuracy value per
                                            (mode, outcome) -- mean retrodiction_accuracy_corrected
                                            at each outcome's own last turn for the four
                                            multi-turn modes (genuinely separate Success/Loss
                                            numbers, not one copied onto the other; a
                                            non-executable-but-known hypothesis counts as 0, a
                                            bare "unknown" hypothesis is excluded here and
                                            reported instead via unknown_hypothesis_share_by_mode),
                                            or a definitional 1.0 (Success) / 0.0 (Loss) for the
                                            two oneshot/singleturn modes, where real retrodiction
                                            is undefined (see compute_heatmap_data). Also
                                            unknown_hypothesis_share_by_mode: per (model, mode,
                                            outcome), the share of that outcome's last-turn
                                            instances whose hypothesis was literally "unknown".
                                            See plot_last_turn_analysis.py's heatmap for how this
                                            is used.

                                            confidence_by_hypothesis_status_mode_outcome: per
                                            (model, mode, outcome), mean confidence_score split
                                            into "known" (hypothesis_unknown == False -- the same
                                            population that correctness_by_mode's retrodiction
                                            value is computed from) vs "unknown"
                                            (hypothesis_unknown == True) -- confidence_by_mode_
                                            outcome above pools both populations into one mean,
                                            which hides that "confident because retrodiction
                                            worked" and "confident despite never committing to a
                                            hypothesis" are different things. See
                                            plot_passive_output_variant.py's confidence/
                                            retrodiction chart for how this is used.

                                            Loss instances' confidence_score/current_mode_label are
                                            read directly from each Lose interactions.json
                                            (bypassing turn_details.json, which only ever covers
                                            Success instances; see analyze_results.py's
                                            load_instance) -- reuses build_turn_row unchanged and
                                            makes no sandbox calls. retrodiction_accuracy for Loss
                                            instances is looked up from
                                            hypothesis_retrodiction_analysis.py's
                                            retrodiction_details.json instead -- that script does
                                            score Loss instances (see its own docstring), it just
                                            wasn't being read here before; see
                                            collect_loss_last_turn_confidence.

    last_turn_mode_shares.json              per (model, mode, outcome): distribution of the
                                            player's self-reported current_mode_label at its
                                            last valid turn (probing, confirming, proposing,
                                            solving, dont_know -- see normalize_mode_label for
                                            how the "don't know"/"dont know" spelling variants
                                            get merged into one category). Flat list of
                                            {model, mode, outcome, current_mode_label, n,
                                            n_total, share}. See plot_last_turn_mode_overall.py.

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


def normalize_mode_label(label):
    """
    Collapses the model's free-text current_mode_label into a canonical
    value. Seen verbatim in this dataset: "probing", "confirming",
    "proposing", "solving", plus three spellings of the same "I don't know
    yet" idea -- "don't know", "dont know", and "don't know" with a curly
    apostrophe -- that need to be counted as one category, not three.
    Anything else passes through unchanged.
    """
    if not isinstance(label, str):
        return label
    normalized = label.replace("’", "'").strip().lower()
    if normalized in ("don't know", "dont know"):
        return "dont_know"
    return label


def _is_hypothesis_unknown(hypothesis):
    """Same check hypothesis_retrodiction_analysis.py uses: bare "unknown", nothing else."""
    return isinstance(hypothesis, str) and hypothesis.strip().lower() == "unknown"


def corrected_retrodiction(hypothesis, retrodiction_record):
    """
    Returns (retrodiction_accuracy_corrected, hypothesis_unknown) for an
    instance's last valid turn, given its raw hypothesis text and (if one
    exists) its hypothesis_retrodiction_analysis.py record for that turn.
    Three distinct cases, only two of which feed into retrodiction accuracy:

    - hypothesis is the literal string "unknown" -- nothing to retrodict,
      reported separately as "unknown" (hypothesis_unknown=True), not
      folded into retrodiction_accuracy_corrected at all (None).
    - no retrodiction record exists at all (last turn was turn 1 -- no
      earlier turns exist to retrodict against yet, structurally
      different from "the hypothesis failed to explain history it did
      have") -- excluded from both retrodiction and the unknown bucket
      (retrodiction_accuracy_corrected=None, hypothesis_unknown=False).
    - a record exists and the hypothesis isn't "unknown" -- counts toward
      retrodiction accuracy: n_match/n_scoreable if the hypothesis could
      be run against at least one earlier input, else 0.0. A hypothesis
      that can't even execute against evidence it already has is a
      failure, not "not applicable" -- excluding it from the mean instead
      of scoring it 0 silently inflates the average (this used to happen
      here; see the module docstring history / conversation that flagged it).
    """
    if _is_hypothesis_unknown(hypothesis):
        return None, True
    if retrodiction_record is None:
        return None, False
    n_scoreable = retrodiction_record["n_scoreable"]
    n_match = retrodiction_record["n_match"]
    return (n_match / n_scoreable if n_scoreable else 0.0), False


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
        corrected_value, is_unknown = corrected_retrodiction(last_turn["hypothesis"], retrodiction)
        records.append(
            {
                "model": model, "experiment": experiment, "mode": mode, "instance": instance,
                "turn": last_turn["turn"], "confidence_score": last_turn["confidence_score"],
                "match": last_turn["match"],
                "n_prev": retrodiction["n_prev"] if retrodiction else 0,
                "n_scoreable": retrodiction["n_scoreable"] if retrodiction else 0,
                "retrodiction_accuracy": retrodiction["retrodiction_accuracy"] if retrodiction else None,
                "retrodiction_accuracy_corrected": corrected_value,
                "hypothesis_unknown": is_unknown,
                "predicted_output": last_turn["predicted_output"], "gm_output": last_turn["gm_output"],
                "hypothesis": last_turn["hypothesis"],
                "current_mode_label": normalize_mode_label(last_turn["current_mode_label"]),
            }
        )
    return records


def collect_loss_last_turn_confidence(results_root: Path, retrodiction_records):
    """
    For every Lose instance, the confidence_score AND retrodiction_accuracy
    at its last valid (non-parser-error) turn.

    confidence_score/current_mode_label are read directly from
    interactions.json, since turn_details.json only ever covers Success
    instances (see analyze_results.py's load_instance) -- reuses
    build_turn_row unchanged (outcome-agnostic) and makes no sandbox calls.

    retrodiction_accuracy is looked up from
    hypothesis_retrodiction_analysis.py's retrodiction_details.json, keyed
    on (model, experiment, mode, instance, turn) and filtered to
    outcome == "loss" -- that script explicitly scores Loss instances too,
    not just Success (see its docstring: "Scored for both Success AND Loss
    instances"), it's just this script that was previously ignoring the
    Loss half and reusing the Success-side number instead. None here means
    the same thing it means on the Success side: last turn was turn 1 (no
    earlier turns to retrodict against), or nothing was scoreable.
    retrodiction_accuracy_corrected/hypothesis_unknown are the same
    correction applied on the Success side -- see corrected_retrodiction.

    Returns a flat list of {model, experiment, mode, instance, turn,
    confidence_score, current_mode_label, retrodiction_accuracy,
    retrodiction_accuracy_corrected, hypothesis_unknown}.
    """
    loss_retrodiction_by_key = {
        (r["model"], r["experiment"], r["mode"], r["instance"], r["turn"]): r
        for r in retrodiction_records if r["outcome"] == "loss"
    }

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
        retrodiction = loss_retrodiction_by_key.get((model, experiment, mode, instance_id, last_turn["turn"]))
        corrected_value, is_unknown = corrected_retrodiction(last_turn["hypothesis"], retrodiction)
        records.append(
            {
                "model": model, "experiment": experiment, "mode": mode, "instance": instance_id,
                "turn": last_turn["turn"], "confidence_score": last_turn["confidence_score"],
                "current_mode_label": normalize_mode_label(last_turn["current_mode_label"]),
                "retrodiction_accuracy": retrodiction["retrodiction_accuracy"] if retrodiction else None,
                "retrodiction_accuracy_corrected": corrected_value,
                "hypothesis_unknown": is_unknown,
            }
        )
    return records


def compute_heatmap_data(last_turn_df, instances, loss_last_turn_records):
    """
    Per (model, mode): mean last-turn confidence_score for Success instances
    and for Loss instances separately (Aborted excluded from both), plus a
    "retrodiction accuracy" value per (mode, outcome) for the four
    multi-turn modes -- genuinely separate numbers for Success and Loss
    (mean retrodiction_accuracy at each outcome's own last turn), not the
    same value copied onto both -- or a definitional 1.0/0.0 for the two
    oneshot/singleturn modes.

    Oneshot games don't probe with their own chosen inputs turn-by-turn --
    the player only ever sees revealed_examples baked into the prompt, which
    aren't logged as (input, gm_output) pairs anywhere in interactions.json,
    so there's nothing to actually retrodict against (real retrodiction
    accuracy is undefined for oneshot, not just small). Rather than leaving
    those cells blank, treat the outcome itself as the retrodiction signal:
    a oneshot Success instance's hypothesis is taken to have "explained" the
    examples that led to it (retrodiction_accuracy = 1.0); a Lose instance's
    didn't (0.0). This is definitional, not measured -- flagged via
    source="assumed_from_outcome" below.

    last_turn_df only ever contains Success instances (confirmed via
    build_outcome_lookup below rather than assumed), so it supplies the
    "success" side; loss_last_turn_records (from
    collect_loss_last_turn_confidence, a separate direct read of Lose
    instances' interactions.json, cross-referenced against
    hypothesis_retrodiction_analysis.py's retrodiction_details.json --
    that script does score Loss instances, see its docstring) supplies the
    "loss" side, for both confidence_score and retrodiction_accuracy.

    The retrodiction_accuracy mean uses retrodiction_accuracy_corrected
    (see corrected_retrodiction), not the raw retrodiction_accuracy field:
    an instance whose last-turn hypothesis was literally "unknown" is
    excluded here and counted instead in the new unknown_hypothesis_share
    output below; an instance whose hypothesis was real but couldn't
    execute against a single earlier input (n_scoreable == 0) counts as a
    0 in this mean rather than being silently dropped from it, which used
    to inflate the average toward whichever few instances happened to be
    executable.
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
        loss_last_turn_records,
        columns=[
            "model", "experiment", "mode", "instance", "turn", "confidence_score",
            "current_mode_label", "retrodiction_accuracy", "retrodiction_accuracy_corrected",
            "hypothesis_unknown",
        ],
    )
    loss_confidence = (
        loss_df.groupby(["model", "mode"])["confidence_score"].mean().round(2)
        if not loss_df.empty else pd.Series(dtype=float)
    )

    success_retro_correctness = (
        df[df["retrodiction_accuracy_corrected"].notna()]
        .groupby(["model", "mode"])["retrodiction_accuracy_corrected"].mean()
    )
    loss_retro_correctness = (
        loss_df[loss_df["retrodiction_accuracy_corrected"].notna()]
        .groupby(["model", "mode"])["retrodiction_accuracy_corrected"].mean()
        if not loss_df.empty else pd.Series(dtype=float)
    )

    success_unknown_share = df.groupby(["model", "mode"])["hypothesis_unknown"].mean()
    loss_unknown_share = (
        loss_df.groupby(["model", "mode"])["hypothesis_unknown"].mean()
        if not loss_df.empty else pd.Series(dtype=float)
    )

    # Confidence split by whether that same last turn's hypothesis was
    # "known" (retrodiction_accuracy_corrected computed, whether or not it
    # scored 0) or literally "unknown" -- confidence_by_mode_outcome above
    # averages across both, which hides the fact that these are two very
    # different populations of instances (one had a hypothesis to
    # retrodict, the other didn't even commit to one).
    success_confidence_by_hyp = df.groupby(["model", "mode", "hypothesis_unknown"])["confidence_score"].mean().round(2)
    loss_confidence_by_hyp = (
        loss_df.groupby(["model", "mode", "hypothesis_unknown"])["confidence_score"].mean().round(2)
        if not loss_df.empty else pd.Series(dtype=float)
    )

    all_model_modes = set(df[["model", "mode"]].itertuples(index=False, name=None))
    if not loss_df.empty:
        all_model_modes |= set(loss_df[["model", "mode"]].itertuples(index=False, name=None))

    by_model = {}
    for model, mode in sorted(all_model_modes):
        model_entry = by_model.setdefault(
            model,
            {
                "modes": [], "correctness_by_mode": {}, "confidence_by_mode_outcome": {"success": {}, "loss": {}},
                "unknown_hypothesis_share_by_mode": {},
            },
        )
        model_entry["modes"].append(mode)

        if mode.endswith("_oneshot"):
            model_entry["correctness_by_mode"][mode] = {
                "success": {"value": 1.0, "source": "assumed_from_outcome"},
                "loss": {"value": 0.0, "source": "assumed_from_outcome"},
            }
        else:
            success_value = success_retro_correctness.get((model, mode))
            loss_value = loss_retro_correctness.get((model, mode))
            model_entry["correctness_by_mode"][mode] = {
                "success": {
                    "value": round(success_value, 2) if success_value is not None else None,
                    "source": "retrodiction",
                },
                "loss": {
                    "value": round(loss_value, 2) if loss_value is not None else None,
                    "source": "retrodiction",
                },
            }

        success_unknown = success_unknown_share.get((model, mode))
        loss_unknown = loss_unknown_share.get((model, mode))
        model_entry["unknown_hypothesis_share_by_mode"][mode] = {
            "success": round(success_unknown, 4) if success_unknown is not None else None,
            "loss": round(loss_unknown, 4) if loss_unknown is not None else None,
        }

        for outcome, source in (("success", success_confidence), ("loss", loss_confidence)):
            value = source.get((model, mode))
            model_entry["confidence_by_mode_outcome"][outcome][mode] = value if value is None else float(value)

        model_entry.setdefault("confidence_by_hypothesis_status_mode_outcome", {})[mode] = {}
        for outcome, source in (("success", success_confidence_by_hyp), ("loss", loss_confidence_by_hyp)):
            known_value = source.get((model, mode, False))
            unknown_value = source.get((model, mode, True))
            model_entry["confidence_by_hypothesis_status_mode_outcome"][mode][outcome] = {
                "known": known_value if known_value is None else float(known_value),
                "unknown": unknown_value if unknown_value is None else float(unknown_value),
            }

    return by_model


def compute_last_turn_mode_shares(last_turn_df, instances, loss_last_turn_records):
    """
    Per (model, mode, outcome): how the player's self-reported
    current_mode_label at its last valid turn is distributed --
    e.g. what fraction of Loss instances in a given mode were still
    "probing" when they ran out of turns vs. "confirming" or "dont_know".
    Success comes from last_turn_df (restricted to Success via
    build_outcome_lookup, same join as compute_heatmap_data); Loss comes
    from loss_last_turn_records (collect_loss_last_turn_confidence's direct
    interactions.json read). Both already went through normalize_mode_label
    when built, so the "don't know" spelling variants are pre-merged here.

    Returns a flat list of {model, mode, outcome, current_mode_label, n,
    n_total, share}.
    """
    outcomes = build_outcome_lookup(instances)
    df = last_turn_df.copy()
    df["outcome"] = df.apply(
        lambda r: outcomes.get((r["model"], r["experiment"], r["mode"], r["instance"])), axis=1,
    )
    success_df = df[df["outcome"] == "success"]

    loss_df = pd.DataFrame(
        loss_last_turn_records,
        columns=["model", "experiment", "mode", "instance", "turn", "confidence_score", "current_mode_label"],
    )

    records = []
    for outcome, outcome_df in (("success", success_df), ("loss", loss_df)):
        if outcome_df.empty:
            continue
        for (model, mode), group in outcome_df.groupby(["model", "mode"]):
            n_total = len(group)
            for label, n in group["current_mode_label"].value_counts().items():
                records.append(
                    {
                        "model": model, "mode": mode, "outcome": outcome,
                        "current_mode_label": label, "n": int(n), "n_total": int(n_total),
                        "share": round(n / n_total, 4),
                    }
                )
    return records


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

    retrodiction_records = _load(retrodiction_path)
    records = build_last_turn_records(_load(turn_details_path), retrodiction_records)
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
    loss_records = collect_loss_last_turn_confidence(results_root, retrodiction_records)
    print(f"{len(loss_records)} Lose instances' last-turn confidence read directly from interactions.json")
    heatmap_data = compute_heatmap_data(df, instances, loss_records)
    _write_json(heatmap_data, out_dir / "last_turn_heatmap.json")

    mode_shares = compute_last_turn_mode_shares(df, instances, loss_records)
    _write_json(mode_shares, out_dir / "last_turn_mode_shares.json")


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
