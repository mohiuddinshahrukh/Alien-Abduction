#!/usr/bin/env python3
"""
Analyze function_detective clembench results.

Walks a results directory shaped like:
    <results_root>/.../<experiment_name>/instance_XXXXX/interactions.json

results_root can point above one model's results or above several -- the
model name isn't taken from the directory layout at all. Both grouping
dimensions are read straight out of each interactions.json:
    - model: the non-GM entry in `players[*].model_name` (falls back to
      meta.results_folder, then player_models[*].model_spec.model_name)
    - mode: the top-level `mode` field (e.g. active_inputs,
      passive_examples_oneshot)
so a results_root covering multiple model folders and/or multiple game
modes is grouped and plotted correctly without any path parsing.

For every instance, extracts the game outcome (Aborted/Lose/Success) and,
for Success == True instances, reads each `evaldata` entry (one per player
prediction/turn) directly -- each entry already carries both sides of the
comparison:
    - predicted_output : what the player guessed -- an int, float, str,
      bool, list, or list-of-lists, depending on what JSON type the
      player's response actually contained for that function signature
    - gm_output         : the GM's ground truth for that same input, always
      logged as a *string* (function_detective's format_value(): strings
      through repr(), floats to 3 decimals, everything else through str())
    - confidence_score  : the player's self-reported confidence
    - current_mode      : the player's self-reported strategy label
        (probing/confirming/random/...)
so no cross-referencing against `observed_pairs` is needed (an earlier
version of this script aligned evaldata to observed_pairs by input value;
that's been dropped now that gm_output is logged inline). Because
predicted_output keeps its native type while gm_output is always a
pre-formatted string, they're compared by applying the same format_value()
to predicted_output rather than by parsing gm_output back into a type --
that's what makes 7 == "7", "7" == "'7'" (no) and 3.14159 == "3.142" (yes,
same rounding the game itself scores with) all resolve the way the game
intends. See format_value() below.

Some evaldata entries are parse-error placeholders: when the player's
response for a round failed to parse, every field on that entry (input,
predicted_output, confidence_score, current_mode, ...) is literally the
string "parser_error" instead of real data. Those entries still occupy a
turn slot (the round happened), but contribute no comparable output/score,
so they're kept in turn_details.json with `parser_error: true` and
`match`/`confidence_score` set to null, and excluded from the numeric
aggregates -- parser_error_counts.json tracks how many landed at each
(model, mode, turn) so the plot can mark them explicitly instead of
silently dropping them from the sample.

Output JSON files (default: written to <results_root>/analysis/, or to --out-dir --
this keeps a separate results folder's output from ever overwriting another's):
    instance_summary.json      {"overall": {"by_model": {<model>: {n_instances, n_success, n_lose,
                                n_aborted, success_rate, lose_rate, aborted_rate (fractions 0-1,
                                2 decimals), "by_mode": {<mode>: {...same fields...}, ...},
                                "by_experiment": {<experiment>: {...same fields...}, ...}}, ...}},
                                "instances": nested model -> experiment -> mode -> instance ->
                                {Aborted, Lose, Success, turns_used, parse_error_count}}
                                (all outcomes, not just Success -- same nesting fashion as
                                turn_details.json, plus the aggregate tally that used to only
                                ever be printed to the console)
    turn_details.json          nested model -> experiment -> mode -> instance -> list of
                                turn records (turn, parser_error, input, predicted_output,
                                gm_output, match, confidence_score, current_mode_label,
                                input_rationale, hypothesis), Success instances only
    input_rationales.json      flat list of every turn's input_rationale (the player's
                                stated reason for that turn's chosen input) across every
                                instance/turn: {model, experiment, mode, instance, turn,
                                input_rationale}. parser_error turns and empty rationales
                                are skipped. Field was renamed from mode_rationale to
                                input_rationale partway through this game's development;
                                both are read so older logged runs still populate this.
    turn_aggregates.json       one record per (model, mode, turn): mean match rate /
                                confidence / unknown_hypothesis_rate, computed over
                                non-parser-error turns only. unknown_hypothesis_rate is a
                                cheap stand-in for clustering the free-text `hypothesis`
                                field: the share of instances still reporting "unknown"
                                at that turn (declining -> the player is converging on a
                                guess; not declining -> still lost)
    mode_label_shares.json     one record per (model, mode, turn, current_mode label):
                                share of instances reporting that label (parser_error
                                counts as its own label here, since it's categorical)
    parser_error_counts.json   one record per (model, mode, turn): how many instances
                                hit a parse error at that turn

    ...and the same three files again with an "_by_experiment" suffix
    (turn_aggregates_by_experiment.json, mode_label_shares_by_experiment.json,
    parser_error_counts_by_experiment.json) -- identical shape, but grouped by
    (model, experiment, turn) instead of (model, mode, turn). Several
    experiment folders can share one mode (numbers_active_inputs and
    list_active_inputs both run mode "active_inputs"), so the mode-grouped
    files pool across them while the _by_experiment files keep each
    experiment's numbers separate.

Usage:
    python analyze_results.py [results_root] [--out-dir OUT_DIR]
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from display_names import display_label

# Anchored to this file's location (analysis/analyze_results.py -> function_detective/results_5),
# rather than a path relative to the current working directory -- that broke as soon as this
# script was invoked from anywhere other than the repo root (e.g. `cd function_detective &&
# python3 analysis/analyze_results.py` resolved to a nonexistent nested results_5).
DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

PARSER_ERROR_SENTINEL = "parser_error"

HYPOTHESIS_DOCSTRING_RE = re.compile(r'"""(.*?)"""', re.DOTALL)
HYPOTHESIS_SIGNATURE_RE = re.compile(r"def\s+\w+\(([^)]*)\)")
TRIVIAL_RETURN_VALUES = {"0", "0.0", "false", "true", "none", '""', "''", "[]", "{}", "unknown"}


def is_hypothesis_unknown(hypothesis):
    """
    True if the player hasn't actually committed to a guess yet. Two logged
    formats exist: an early plain-text one where hypothesis is literally the
    string "unknown", and a later one where hypothesis is a full Python
    function stub -- e.g. "def f(x):" on one line, a docstring of just
    "unknown" on the next, then "return x".

    For the code-stub format, the docstring alone isn't reliable: a docstring
    of "unknown" next to a real, non-trivial return expression (e.g.
    `return x*4 if x < 4 else 2`) means the player actually has committed to
    something and just left the docstring stale -- that's a real hypothesis,
    not "no guess yet". So "unknown" is only reported when the docstring says
    so *and* the body is a single trivial placeholder return (a literal like
    0/False/None/"unknown", or a bare pass-through of one of the function's
    own parameters) -- anything else (arithmetic, conditionals, calls) counts
    as committed even if the docstring wasn't updated to match.
    """
    if not isinstance(hypothesis, str):
        return False
    text = hypothesis.strip()
    if text.lower() == "unknown":
        return True

    doc_match = HYPOTHESIS_DOCSTRING_RE.search(text)
    sig_match = HYPOTHESIS_SIGNATURE_RE.search(text)
    if not (doc_match and sig_match):
        return False
    if doc_match.group(1).strip().lower() != "unknown":
        return False

    body_lines = [line.strip() for line in text[doc_match.end():].strip().splitlines() if line.strip()]
    if len(body_lines) != 1 or not body_lines[0].startswith("return"):
        return False  # multi-line/branching body -- a real attempt regardless of the stale docstring

    expr = body_lines[0][len("return"):].strip()
    param_names = {p.strip() for p in sig_match.group(1).split(",") if p.strip()}
    return expr.lower() in TRIVIAL_RETURN_VALUES or expr in param_names


def find_instance_files(results_root: Path):
    """Yield every interactions.json that lives directly inside an instance_* dir."""
    for interactions_path in sorted(results_root.rglob("interactions.json")):
        if interactions_path.parent.name.startswith("instance_"):
            yield interactions_path


def get_model_name(data):
    """
    Identify which model played this instance, straight from the JSON
    content -- not from the results directory layout -- so a results_root
    spanning multiple model folders groups correctly.
    """
    for player in data.get("players", {}).values():
        if player.get("game_role") != "Game Master" and player.get("model_name"):
            return player["model_name"]
    results_folder = data.get("meta", {}).get("results_folder")
    if results_folder:
        return results_folder
    for player_model in data.get("player_models", {}).values():
        model_name = player_model.get("model_spec", {}).get("model_name")
        if model_name:
            return model_name
    return "unknown_model"


def format_value(value):
    """
    Mirror function_detective's own utils.format_value (see utils.py) so
    predicted_output is stringified exactly the way gm_output already was
    when the GM recorded it: strings go through repr (so "7" and 7 don't
    collide), floats are rounded to 3 decimals (matching the tolerance the
    game itself scores with), everything else (int/bool/list/tuple/None)
    through plain str(). predicted_output can be any of these types
    depending on what the player's JSON contained (an int, a list, a
    string, ...); gm_output is always already this same string form, so
    the comparison is format_value(predicted_output) == gm_output rather
    than trying to parse gm_output back into a native type.
    """
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_turn_row(entry, turn_idx, experiment, instance_id, model, mode):
    """
    Turn a single evaldata entry into a turn_details row. A parse-error
    round has every evaldata field set to the literal string
    "parser_error" -- detected via current_mode so it doesn't depend on
    input's type (input is otherwise an int/str/list/tuple depending on the
    function signature under test).

    `turn` is the 1-indexed position (T1, T2, ...) and is just
    current_round + 1 -- the two encode the same thing, so only one is
    kept. It's derived from current_round rather than the loop position
    whenever current_round is present, in case a future dataset has gaps.
    """
    is_parser_error = entry.get("current_mode") == PARSER_ERROR_SENTINEL
    current_round = entry.get("current_round")
    turn = current_round + 1 if isinstance(current_round, int) else turn_idx
    base = {
        "experiment": experiment,
        "instance": instance_id,
        "model": model,
        "mode": mode,
        "turn": turn,
        "parser_error": is_parser_error,
    }
    if is_parser_error:
        base.update(
            input=None,
            predicted_output=None,
            gm_output=None,
            match=None,
            confidence_score=None,
            current_mode_label=PARSER_ERROR_SENTINEL,
            input_rationale=None,
            hypothesis=None,
            hypothesis_unknown=None,
        )
        return base

    predicted_output = entry.get("predicted_output")
    gm_output = entry.get("gm_output")
    hypothesis = entry.get("hypothesis")
    base.update(
        input=entry.get("input"),
        predicted_output=predicted_output,
        gm_output=gm_output,
        match=format_value(predicted_output) == gm_output,
        confidence_score=entry.get("confidence_score"),
        current_mode_label=entry.get("current_mode"),
        # Field was renamed from mode_rationale to input_rationale partway through this
        # game's development; the fallback only matters for older logged runs, since a
        # parser_error placeholder round (handled above) is the only place mode_rationale
        # still shows up in current data.
        input_rationale=entry.get("input_rationale", entry.get("mode_rationale")),
        hypothesis=hypothesis,
        # Cheap stand-in for clustering the free-text hypothesis: has the
        # player committed to *any* guess yet, vs. still reporting "unknown".
        hypothesis_unknown=is_hypothesis_unknown(hypothesis),
    )
    return base


def nest_turn_rows(rows):
    """
    Group flat turn rows into model -> experiment -> mode -> instance -> [turns],
    dropping the now-redundant grouping keys from each turn record since
    they're captured by its position in the nesting.
    """
    nested = {}
    for row in rows:
        turns = (
            nested.setdefault(row["model"], {})
            .setdefault(row["experiment"], {})
            .setdefault(row["mode"], {})
            .setdefault(row["instance"], [])
        )
        turns.append({k: v for k, v in row.items() if k not in ("model", "experiment", "mode", "instance")})
    return nested


def nest_instance_summaries(summaries):
    """
    Group flat instance summaries into model -> experiment -> mode -> instance
    -> {Aborted, Lose, Success, turns_used, parse_error_count}, same fashion
    as nest_turn_rows -- one summary per instance, so the leaf is a single
    dict rather than a list.
    """
    nested = {}
    for summary in summaries:
        nested.setdefault(summary["model"], {}).setdefault(summary["experiment"], {}).setdefault(
            summary["mode"], {}
        )[summary["instance"]] = {
            k: v for k, v in summary.items() if k not in ("model", "experiment", "mode", "instance")
        }
    return nested


def compute_overall_summary(summaries):
    """
    Aborted/Lose/Success tallies, broken down by model -- no grand total
    across models, since that's rarely the number anyone wants when a
    results_root spans more than one -- and, within each model, broken down
    further by mode (variant) and by experiment (category). This used to
    only ever be printed to the console; now it's saved alongside the
    per-instance detail so it survives after the run.
    """

    def _tally(rows):
        n_instances = len(rows)
        n_success = sum(r["Success"] for r in rows)
        n_lose = sum(r["Lose"] for r in rows)
        n_aborted = sum(r["Aborted"] for r in rows)
        return {
            "n_instances": n_instances,
            "n_success": n_success,
            "n_lose": n_lose,
            "n_aborted": n_aborted,
            "success_rate": round(n_success / n_instances, 2) if n_instances else 0.0,
            "lose_rate": round(n_lose / n_instances, 2) if n_instances else 0.0,
            "aborted_rate": round(n_aborted / n_instances, 2) if n_instances else 0.0,
        }

    by_model = {}
    for model in sorted({r["model"] for r in summaries}):
        model_rows = [r for r in summaries if r["model"] == model]
        model_tally = _tally(model_rows)
        model_tally["by_mode"] = {
            mode: _tally([r for r in model_rows if r["mode"] == mode])
            for mode in sorted({r["mode"] for r in model_rows})
        }
        model_tally["by_experiment"] = {
            experiment: _tally([r for r in model_rows if r["experiment"] == experiment])
            for experiment in sorted({r["experiment"] for r in model_rows})
        }
        by_model[model] = model_tally

    return {"by_model": by_model}


def collect_input_rationales(rows):
    """
    Flatten every turn's `input_rationale` (the player's stated reason for
    choosing that turn's input, e.g. "Check behavior when both inputs are
    false to distinguish XOR/NAND/etc.") into one list spanning all
    instances/turns -- parser_error turns have no real rationale so they're
    skipped, as are the rare rows where it's missing/empty.
    """
    records = []
    for row in rows:
        if row["parser_error"]:
            continue
        rationale = row.get("input_rationale")
        if not rationale:
            continue
        records.append(
            {
                "model": row["model"],
                "experiment": row["experiment"],
                "mode": row["mode"],
                "instance": row["instance"],
                "turn": row["turn"],
                "input_rationale": rationale,
            }
        )
    return records


def load_instance(interactions_path: Path):
    with open(interactions_path) as f:
        data = json.load(f)

    experiment = interactions_path.parent.parent.name
    instance_id = interactions_path.parent.name
    model = get_model_name(data)
    mode = data.get("mode")

    summary = {
        "experiment": experiment,
        "instance": instance_id,
        "model": model,
        "mode": mode,
        "Aborted": bool(data.get("Aborted")),
        "Lose": bool(data.get("Lose")),
        "Success": bool(data.get("Success")),
        "turns_used": data.get("turns_used"),
        "parse_error_count": data.get("parse_error_count"),
    }

    turn_rows = []
    if summary["Success"]:
        for turn_idx, entry in enumerate(data.get("evaldata", []), start=1):
            turn_rows.append(
                build_turn_row(entry, turn_idx, experiment, instance_id, model, mode)
            )

    return summary, turn_rows


def _write_json(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {path}")


def compute_aggregates(turn_df, group_key, out_dir, suffix=""):
    """
    Write turn_aggregates{suffix}.json / mode_label_shares{suffix}.json /
    parser_error_counts{suffix}.json, grouped by (model, group_key, turn).

    Called once with group_key="mode" (the original, coarser grouping --
    several experiments can share one mode, e.g. numbers_active_inputs and
    list_active_inputs both run mode "active_inputs") and once with
    group_key="experiment" (the finer breakdown, one row per experiment
    folder), so both views exist side by side rather than one replacing
    the other.
    """
    valid_df = turn_df[~turn_df["parser_error"]]

    agg_df = (
        valid_df.groupby(["model", group_key, "turn"])
        .agg(
            n_instances=("match", "size"),
            match_rate=("match", "mean"),
            mean_confidence=("confidence_score", "mean"),
            std_confidence=("confidence_score", "std"),
            unknown_hypothesis_rate=("hypothesis_unknown", "mean"),
        )
        .reset_index()
        .sort_values(["model", group_key, "turn"])
    )
    _write_json(json.loads(agg_df.to_json(orient="records")), out_dir / f"turn_aggregates{suffix}.json")

    # Share of instances reporting each current_mode label, per (model, group_key, turn).
    # parser_error is kept as its own label here (categorical, so it plots fine as a line).
    label_counts = (
        turn_df.groupby(["model", group_key, "turn", "current_mode_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = turn_df.groupby(["model", group_key, "turn"]).size().rename("n_total")
    label_counts = label_counts.join(totals, on=["model", group_key, "turn"])
    label_counts["share"] = label_counts["count"] / label_counts["n_total"]
    label_counts = label_counts.sort_values(["model", group_key, "turn", "current_mode_label"])
    _write_json(json.loads(label_counts.to_json(orient="records")), out_dir / f"mode_label_shares{suffix}.json")

    # How many instances hit a parse error at each (model, group_key, turn) -- used to mark
    # match_accuracy_by_turn.png / confidence_by_turn.png where the line's sample size
    # quietly excludes them.
    parser_error_df = (
        turn_df.groupby(["model", group_key, "turn"])
        .agg(n_parser_error=("parser_error", "sum"), n_total=("parser_error", "size"))
        .reset_index()
    )
    parser_error_df = parser_error_df[parser_error_df["n_parser_error"] > 0]
    _write_json(
        json.loads(parser_error_df.to_json(orient="records")), out_dir / f"parser_error_counts{suffix}.json"
    )

    return agg_df


def analyze(results_root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_turn_rows = []
    for interactions_path in find_instance_files(results_root):
        print(f"Processing {interactions_path}")
        summary, turn_rows = load_instance(interactions_path)
        summaries.append(summary)
        all_turn_rows.extend(turn_rows)

    if not summaries:
        raise SystemExit(f"No instance_*/interactions.json files found under {results_root}")

    instance_df = pd.DataFrame(summaries)
    turn_df = pd.DataFrame(all_turn_rows)

    _write_json(
        {"overall": compute_overall_summary(summaries), "instances": nest_instance_summaries(summaries)},
        out_dir / "instance_summary.json",
    )
    _write_json(nest_turn_rows(all_turn_rows), out_dir / "turn_details.json")
    _write_json(collect_input_rationales(all_turn_rows), out_dir / "input_rationales.json")

    print(
        f"\n{len(instance_df)} instances found "
        f"({int(instance_df['Success'].sum())} Success, "
        f"{int(instance_df['Lose'].sum())} Lose, "
        f"{int(instance_df['Aborted'].sum())} Aborted)"
    )

    if turn_df.empty:
        print("No successful instances with turn data -- skipping aggregation.")
        for name in (
            "turn_aggregates.json", "mode_label_shares.json", "parser_error_counts.json",
            "turn_aggregates_by_experiment.json", "mode_label_shares_by_experiment.json",
            "parser_error_counts_by_experiment.json",
        ):
            _write_json([], out_dir / name)
        return instance_df, turn_df, None

    n_parser_errors = int(turn_df["parser_error"].sum())
    print(f"Parser-error turns: {n_parser_errors} / {len(turn_df)} total turns")

    # Two views of the same turn data: grouped by mode (coarser -- several
    # experiments can share one mode) and by experiment (finer -- one
    # breakdown per experiment folder, e.g. numbers_active_inputs vs.
    # list_active_inputs, even though both run mode "active_inputs").
    agg_df = compute_aggregates(turn_df, "mode", out_dir)
    compute_aggregates(turn_df, "experiment", out_dir, suffix="_by_experiment")

    print("\nPer-model / per-mode / per-turn aggregate preview:")
    display_agg = agg_df.copy()
    display_agg["mode"] = display_agg["mode"].map(display_label)
    print(display_agg.to_string(index=False))

    return instance_df, turn_df, agg_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=DEFAULT_RESULTS_ROOT,
        help="Root folder containing experiment subfolders of instance_* dirs "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write the JSON files (default: <results_root>/analysis, so results "
        "from different runs/folders never overwrite each other's output)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "analysis"
    analyze(results_root, out_dir)


if __name__ == "__main__":
    main()
