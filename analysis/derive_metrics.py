import json
from typing import List

import pandas as pd


MODE_ORDER: List[str] = [
    "passive_examples",
    "active_inputs",
    "passive_labeled_pairs",
    "active_pair_checks",
    "passive_examples_oneshot",
    "passive_labeled_pairs_oneshot",
]

DOMAIN_ORDER: List[str] = [
    "numbers",
    "two_numbers",
    "string",
    "list",
    "logic",
]

FAILURE_REASON_ORDER: List[str] = [
    "correct_guess",
    "protocol_or_format_error",
    "runtime_error",
    "wrong_function_guess",
    "aborted",
]


def _classify_failure_reason(row: pd.Series) -> str:
    if bool(row["aborted"]):
        return "aborted"
    if bool(row["correct_guess"]):
        return "correct_guess"
    if float(row["violated_request_count"]) > 0 or float(row["parse_error_count"]) > 0:
        return "protocol_or_format_error"
    if float(row["runtime_error_count"]) > 0:
        return "runtime_error"
    return "wrong_function_guess"


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()

    bool_columns = ["success", "lose", "aborted"]
    for column in bool_columns:
        derived[column] = derived[column].fillna(False).astype(bool)
    derived["played"] = derived["played"].fillna(True).astype(bool)

    numeric_defaults = {
        "main_score": 0.0,
        "quality_score": 0.0,
        "request_count": 0,
        "parsed_request_count": 0,
        "violated_request_count": 0,
        "parse_error_count": 0,
        "runtime_error_count": 0,
        "internal_consistency_score": 0.0,
        "internal_consistency_violations": 0,
        "round_count": 0,
        "turns_used": 0,
        "max_turns": 0,
        "efficiency_raw": 0.0,
        "tolerance_used": 0,
    }
    for column, default in numeric_defaults.items():
        derived[column] = pd.to_numeric(derived[column], errors="coerce").fillna(default)

    def _count_test_cases(instance_path: str) -> int:
        with open(instance_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return int(len(payload.get("test_cases", [])))

    derived["n_test_cases"] = derived.apply(
        lambda row: int(row["n_test_cases_logged"])
        if pd.notna(row.get("n_test_cases_logged"))
        else _count_test_cases(row["instance_path"]),
        axis=1,
    )
    derived["correct_guess"] = derived["success"].astype(bool)
    derived["binary_accuracy"] = derived["correct_guess"].astype(int)
    derived["passed_test_cases"] = derived.apply(
        lambda row: int(row["n_test_cases"]) if row["correct_guess"] else 0,
        axis=1,
    )
    derived["failed_test_cases"] = derived["n_test_cases"] - derived["passed_test_cases"]
    derived["pass_rate"] = derived["binary_accuracy"].astype(float)
    derived["closeness_bucket"] = derived["correct_guess"].map({True: "exact", False: "failure"})
    derived["failure_reason"] = derived.apply(_classify_failure_reason, axis=1)
    derived["observed_pair_consistency"] = derived["internal_consistency_score"]
    derived["guess_turn"] = derived["round_count"].clip(lower=0)
    derived["budget_used_ratio"] = derived.apply(
        lambda row: (float(row["guess_turn"]) / float(row["max_turns"])) if float(row["max_turns"]) > 0 else 0.0,
        axis=1,
    ).clip(lower=0.0, upper=1.0)
    derived["remaining_turn_budget"] = (derived["max_turns"] - derived["guess_turn"]).clip(lower=0)
    derived["guess_outcome"] = derived["correct_guess"].map({True: "right", False: "wrong"})
    derived["is_oneshot"] = derived["mode"].astype(str).str.endswith("_oneshot")
    derived["is_interactive"] = ~derived["is_oneshot"]
    derived["mode"] = pd.Categorical(derived["mode"], categories=MODE_ORDER, ordered=True)
    derived["domain"] = pd.Categorical(derived["domain"], categories=DOMAIN_ORDER, ordered=True)
    derived["failure_reason"] = pd.Categorical(
        derived["failure_reason"], categories=FAILURE_REASON_ORDER, ordered=True
    )
    return derived.sort_values(["model", "mode", "domain", "experiment", "game_id"]).reset_index(drop=True)
