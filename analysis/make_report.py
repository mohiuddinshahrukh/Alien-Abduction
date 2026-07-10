from pathlib import Path
from typing import Dict

import pandas as pd


def _bullet_list(values) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def make_report(
    df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    figure_paths: Dict[str, Path],
    table_paths: Dict[str, Path],
) -> Path:
    report_path = output_dir / "report.md"

    model_summary = df.groupby("model", dropna=False, observed=True)["quality_score"].mean().sort_values(ascending=False)
    domain_summary = df.groupby("domain", dropna=False, observed=True)["binary_accuracy"].sum().sort_values(ascending=False)
    callable_summary = df.groupby("callable", dropna=False, observed=True)["binary_accuracy"].sum().sort_values(ascending=False)
    failure_counts = df[df["failure_reason"] != "correct_guess"]["failure_reason"].value_counts(dropna=False)
    failed = df[df["binary_accuracy"] == 0]
    avg_failed_consistency = float(failed["internal_consistency_score"].mean()) if not failed.empty else 0.0

    strongest_model = model_summary.index[0] if not model_summary.empty else "n/a"
    strongest_domain = domain_summary.index[0] if not domain_summary.empty else "n/a"
    weakest_domain = domain_summary.index[-1] if not domain_summary.empty else "n/a"
    strongest_callable = callable_summary.index[0] if not callable_summary.empty else "n/a"
    weakest_callable = callable_summary.index[-1] if not callable_summary.empty else "n/a"
    biggest_failure_reason = failure_counts.index[0] if not failure_counts.empty else "none"
    consistency_read = "mostly locally consistent" if avg_failed_consistency >= 0.5 else "mostly inconsistent"

    lines = [
        "# Function Detective Analysis",
        "",
        f"- Input folder analyzed: `{input_dir}`",
        f"- Output folder: `{output_dir}`",
        f"- Models found: {_bullet_list(sorted(df['model'].astype(str).unique()))}",
        f"- Modes found: {_bullet_list(sorted(df['mode'].astype(str).unique()))}",
        f"- Domains found: {_bullet_list(sorted(df['domain'].astype(str).unique()))}",
        f"- Test-case counts detected: {_bullet_list(sorted(int(value) for value in df['n_test_cases'].unique()))}",
        "",
        "## Scoring",
        "",
        "- Raw accuracy = held-out test accuracy of the submitted solution across evaluation cases.",
        "- Binary accuracy = 1 for an exact hidden-function recovery, else 0.",
        "- Interactive efficiency = `(T_total - T_used + 1) / T_total`, clamped to `[0, 1]`.",
        "- One-shot modes do not use the tolerance term and do not multiply correctness by efficiency.",
        "- Interactive quality score = `100 * Binary Accuracy * Efficiency`.",
        "- One-shot quality score = `100 * Binary Accuracy`.",
        "",
        "## Auto Findings",
        "",
        f"- Strongest model overall: `{strongest_model}`",
        f"- Strongest domain by correct guesses: `{strongest_domain}`",
        f"- Weakest domain: `{weakest_domain}`",
        f"- Strongest callable: `{strongest_callable}`",
        f"- Weakest callable: `{weakest_callable}`",
        f"- Biggest failure reason: `{biggest_failure_reason}`",
        f"- Average failed-episode observed-pair consistency: `{avg_failed_consistency:.3f}`",
        f"- Failure pattern read: `{consistency_read}`",
        "",
        "## Figures",
        "",
    ]

    for name, path in sorted(figure_paths.items()):
        rel = path.relative_to(output_dir)
        lines.append(f"- [{name}]({rel.as_posix()})")

    lines.extend(["", "## Tables", ""])
    for name, path in sorted(table_paths.items()):
        rel = path.relative_to(output_dir)
        lines.append(f"- [{name}]({rel.as_posix()})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
