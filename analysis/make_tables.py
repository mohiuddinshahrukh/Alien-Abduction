from pathlib import Path
from typing import Dict

import pandas as pd


def _summary(df: pd.DataFrame, group_cols):
    summary = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            episodes=("game_id", "count"),
            played_episodes=("played", "sum"),
            correct_guesses=("binary_accuracy", "sum"),
            avg_quality_score=("quality_score", "mean"),
            avg_binary_accuracy=("binary_accuracy", "mean"),
            avg_efficiency=("efficiency_raw", "mean"),
            avg_guess_turn=("guess_turn", "mean"),
            avg_budget_used_ratio=("budget_used_ratio", "mean"),
            avg_remaining_turn_budget=("remaining_turn_budget", "mean"),
            avg_passed_test_cases=("passed_test_cases", "mean"),
            avg_failed_test_cases=("failed_test_cases", "mean"),
            avg_internal_consistency_score=("internal_consistency_score", "mean"),
            avg_violated_request_count=("violated_request_count", "mean"),
            avg_runtime_error_count=("runtime_error_count", "mean"),
            avg_parse_error_count=("parse_error_count", "mean"),
        )
        .reset_index()
    )

    failed = df[df["binary_accuracy"] == 0]
    if failed.empty:
        summary["avg_internal_consistency_score_on_failed"] = pd.NA
    else:
        failed_summary = (
            failed.groupby(group_cols, dropna=False, observed=True)
            .agg(avg_internal_consistency_score_on_failed=("internal_consistency_score", "mean"))
            .reset_index()
        )
        summary = summary.merge(failed_summary, on=group_cols, how="left")
    return summary


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def make_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_summary = _summary(df, ["model"])
    domain_summary = _summary(df, ["domain"])
    model_domain_summary = _summary(df, ["model", "domain"])
    callable_summary = _summary(df, ["callable"])
    consistency_summary = _summary(df, ["domain"])

    failure_group_cols = ["failure_reason"] if df["model"].nunique() == 1 else ["model", "failure_reason"]
    failure_reason_summary = (
        df.groupby(failure_group_cols, dropna=False, observed=True)
        .agg(episodes=("game_id", "count"))
        .reset_index()
    )
    failure_reason_summary["share"] = (
        failure_reason_summary["episodes"] / max(1, int(len(df)))
    )

    protocol_metrics = (
        df.groupby(["model", "mode"], dropna=False, observed=True)
        .agg(
            episodes=("game_id", "count"),
            avg_request_count=("request_count", "mean"),
            avg_parsed_request_count=("parsed_request_count", "mean"),
            avg_violated_request_count=("violated_request_count", "mean"),
            avg_parse_error_count=("parse_error_count", "mean"),
            avg_runtime_error_count=("runtime_error_count", "mean"),
        )
        .reset_index()
    )

    scoring_components = (
        df[
            [
                "model",
                "mode",
                "domain",
                "experiment",
                "callable",
                "game_id",
                "quality_score",
                "binary_accuracy",
                "efficiency_raw",
                "guess_turn",
                "budget_used_ratio",
                "remaining_turn_budget",
                "tolerance_used",
                "n_test_cases",
                "passed_test_cases",
                "failed_test_cases",
            ]
        ]
        .copy()
    )

    guess_budget_summary = (
        df.groupby(["model", "domain", "guess_outcome"], dropna=False, observed=True)
        .agg(
            episodes=("game_id", "count"),
            avg_guess_turn=("guess_turn", "mean"),
            avg_budget_used_ratio=("budget_used_ratio", "mean"),
            avg_remaining_turn_budget=("remaining_turn_budget", "mean"),
            avg_quality_score=("quality_score", "mean"),
        )
        .reset_index()
    )

    root_outputs = {
        "model_summary": output_dir / "model_summary.csv",
        "domain_summary": output_dir / "domain_summary.csv",
        "callable_summary": output_dir / "callable_summary.csv",
        "failure_reason_summary": output_dir / "failure_reason_summary.csv",
        "guess_budget_summary": output_dir / "guess_budget_summary.csv",
    }
    _write(model_summary, root_outputs["model_summary"])
    _write(domain_summary, root_outputs["domain_summary"])
    _write(callable_summary, root_outputs["callable_summary"])
    _write(failure_reason_summary, root_outputs["failure_reason_summary"])
    _write(guess_budget_summary, root_outputs["guess_budget_summary"])

    table_outputs = {
        "table_01_model_summary": tables_dir / "table_01_model_summary.csv",
        "table_02_domain_summary": tables_dir / "table_02_domain_summary.csv",
        "table_03_model_domain_summary": tables_dir / "table_03_model_domain_summary.csv",
        "table_04_failure_reason_summary": tables_dir / "table_04_failure_reason_summary.csv",
        "table_05_callable_summary": tables_dir / "table_05_callable_summary.csv",
        "table_06_consistency_summary": tables_dir / "table_06_consistency_summary.csv",
        "table_07_protocol_metrics": tables_dir / "table_07_protocol_metrics.csv",
        "table_08_scoring_components": tables_dir / "table_08_scoring_components.csv",
        "table_09_guess_budget_summary": tables_dir / "table_09_guess_budget_summary.csv",
    }
    _write(model_summary, table_outputs["table_01_model_summary"])
    _write(domain_summary, table_outputs["table_02_domain_summary"])
    _write(model_domain_summary, table_outputs["table_03_model_domain_summary"])
    _write(failure_reason_summary, table_outputs["table_04_failure_reason_summary"])
    _write(callable_summary, table_outputs["table_05_callable_summary"])
    _write(consistency_summary, table_outputs["table_06_consistency_summary"])
    _write(protocol_metrics, table_outputs["table_07_protocol_metrics"])
    _write(scoring_components, table_outputs["table_08_scoring_components"])
    _write(guess_budget_summary, table_outputs["table_09_guess_budget_summary"])
    return {**root_outputs, **table_outputs}
