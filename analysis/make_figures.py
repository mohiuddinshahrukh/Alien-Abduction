from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
    }
)

DOMAIN_LABELS = {
    "numbers": "Numbers",
    "two_numbers": "Two Numbers",
    "string": "String",
    "list": "List",
    "logic": "Logic",
}

MODE_LABELS = {
    "passive_examples": "Passive Examples",
    "active_inputs": "Active Inputs",
    "passive_labeled_pairs": "Passive Labeled Pairs",
    "active_pair_checks": "Active Pair Checks",
    "passive_examples_oneshot": "Passive Examples One-Shot",
    "passive_labeled_pairs_oneshot": "Passive Labeled Pairs One-Shot",
}


def _label_domain(value: str) -> str:
    return DOMAIN_LABELS.get(str(value), str(value).replace("_", " ").title())


def _humanize(text: str) -> str:
    return str(text).replace("_", " ").title()


def _label_mode(value: str) -> str:
    return MODE_LABELS.get(str(value), str(value))


def _label_outcome(value: str) -> str:
    return {"right": "Right Guess", "wrong": "Wrong Guess"}.get(str(value), str(value))


def _model_palette(df: pd.DataFrame) -> Dict[str, tuple]:
    models = list(pd.Series(df["model"].astype(str).unique()).sort_values())
    colors = sns.color_palette("Set2", n_colors=max(1, len(models)))
    return {model: colors[idx] for idx, model in enumerate(models)}


def _save(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    fig.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.10, right=0.97)
    fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _skip_if_empty(frame: pd.DataFrame) -> bool:
    return frame.empty or len(frame) == 0


def _barplot(data: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None, ax=ax)
    ax.set_title(title, pad=18)
    return fig


def _apply_domain_tick_labels(ax) -> None:
    ticks = ax.get_xticks()
    labels = [_label_domain(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _apply_mode_tick_labels(ax) -> None:
    ticks = ax.get_xticks()
    labels = [_label_mode(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=10)


def _apply_model_tick_labels(ax) -> None:
    ticks = ax.get_xticks()
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)


def _style_axes(ax, x_is_model: bool = False) -> None:
    ax.tick_params(axis="both", which="major", pad=8)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    if x_is_model:
        _apply_model_tick_labels(ax)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10)


def _style_mode_axis(ax) -> None:
    _apply_mode_tick_labels(ax)
    ax.tick_params(axis="x", pad=10)


def _annotate_bars(ax, fmt: str = "{:.1f}") -> None:
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=11)
        except Exception:
            continue


def make_figures(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    palette = _model_palette(df)
    hue = "model" if df["model"].nunique() > 1 else None

    def register(name: str):
        outputs[name] = figures_dir / f"{name}.png"

    model_summary = df.groupby("model", dropna=False, observed=True)["quality_score"].mean().reset_index()
    if len(model_summary) > 1:
        fig = _barplot(model_summary, "model", "quality_score", None, "Average Quality Score by Model")
        _style_axes(fig.axes[0], x_is_model=True)
        _annotate_bars(fig.axes[0], "{:.1f}")
        _save(fig, figures_dir, "fig_01_model_comparison_bar")
        register("fig_01_model_comparison_bar")

    domain_summary = df.groupby(["domain", "model"], dropna=False, observed=True)["quality_score"].mean().reset_index()
    if not _skip_if_empty(domain_summary):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=domain_summary, x="domain", y="quality_score", hue=hue, palette=palette if hue else None, errorbar=None, ax=ax)
        ax.set_title("Average Quality Score by Domain")
        ax.set_xlabel("Function Domain")
        ax.set_ylabel("Quality Score")
        _apply_domain_tick_labels(ax)
        _style_axes(ax)
        _annotate_bars(ax, "{:.1f}")
        _save(fig, figures_dir, "fig_02_domain_comparison_bar")
        register("fig_02_domain_comparison_bar")

    success_by_domain = df.groupby(["domain", "model"], dropna=False, observed=True)["binary_accuracy"].sum().reset_index()
    if not _skip_if_empty(success_by_domain):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=success_by_domain, x="domain", y="binary_accuracy", hue=hue, palette=palette if hue else None, errorbar=None, ax=ax)
        ax.set_title("Correct Guesses by Domain")
        ax.set_xlabel("Function Domain")
        ax.set_ylabel("Solved Functions")
        _apply_domain_tick_labels(ax)
        _style_axes(ax)
        _annotate_bars(ax, "{:.0f}")
        _save(fig, figures_dir, "fig_03_success_rate_by_domain")
        register("fig_03_success_rate_by_domain")

    testcase_summary = (
        df.groupby(["domain", "model"], dropna=False, observed=True)[["passed_test_cases", "failed_test_cases"]]
        .mean()
        .reset_index()
    )
    if not _skip_if_empty(testcase_summary):
        key = "domain" if df["domain"].nunique() > 1 else "model"
        stacked = df.groupby(key, dropna=False, observed=True)[["passed_test_cases", "failed_test_cases"]].mean()
        stacked = stacked.rename(columns={"passed_test_cases": "Passed", "failed_test_cases": "Failed"})
        if key == "domain":
            stacked.index = [_label_domain(idx) for idx in stacked.index]
        fig, ax = plt.subplots(figsize=(10, 6))
        stacked.plot(kind="bar", stacked=True, ax=ax, color=["#4C956C", "#D1495B"])
        ax.set_title("Average Passed vs Failed Evaluation Cases")
        ax.set_ylabel("Average Test Cases")
        ax.set_xlabel("Function Domain" if key == "domain" else "Model")
        ax.legend(title="")
        _save(fig, figures_dir, "fig_04_passed_failed_testcases_stacked")
        register("fig_04_passed_failed_testcases_stacked")

    heatmap_source = df.pivot_table(
        index="mode" if df["mode"].nunique() > 1 else "model",
        columns="domain",
        values="binary_accuracy",
        aggfunc="sum",
        observed=True,
    )
    if not _skip_if_empty(heatmap_source):
        if df["mode"].nunique() > 1:
            heatmap_source = heatmap_source.rename(index=lambda idx: _label_mode(idx))
        heatmap_source = heatmap_source.rename(columns=lambda col: _label_domain(col))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_source, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title("Correct Guesses Heatmap")
        ax.set_xlabel("Function Domain")
        ax.set_ylabel("Mode" if df["mode"].nunique() > 1 else "Model")
        _style_axes(ax)
        _save(fig, figures_dir, "fig_05_pass_rate_heatmap")
        register("fig_05_pass_rate_heatmap")

    failed = df[df["binary_accuracy"] == 0].copy()
    if not failed.empty:
        consistency_group = "domain" if failed["domain"].nunique() > 1 else ("model" if failed["model"].nunique() > 1 else "mode")
        failed_consistency = (
            failed.groupby(consistency_group, dropna=False, observed=True)["internal_consistency_score"]
            .mean()
            .reset_index()
        )
        if consistency_group == "mode":
            failed_consistency["mode"] = failed_consistency["mode"].astype(str).map(_label_mode)
        fig = _barplot(
            failed_consistency,
            consistency_group,
            "internal_consistency_score",
            None,
            "Observed-Pair Consistency on Failed Episodes by Domain" if consistency_group == "domain" else "Observed-Pair Consistency on Failed Episodes",
        )
        ax = fig.axes[0]
        if consistency_group == "domain":
            ax.set_xlabel("Function Domain")
            _apply_domain_tick_labels(ax)
        elif consistency_group == "mode":
            ax.set_xlabel("Mode")
            _style_mode_axis(ax)
        ax.set_ylabel("Observed-Pair Consistency")
        _style_axes(ax)
        _annotate_bars(ax, "{:.2f}")
        _save(fig, figures_dir, "fig_06_failed_zero_accuracy_consistency")
        register("fig_06_failed_zero_accuracy_consistency")

        if failed["domain"].nunique() > 1:
            failed_consistency_domain = (
                failed.groupby("domain", dropna=False, observed=True)["internal_consistency_score"]
                .mean()
                .reset_index()
            )
            fig = _barplot(
                failed_consistency_domain,
                "domain",
                "internal_consistency_score",
                None,
                "Observed-Pair Consistency on Failed Episodes by Function Domain",
            )
            ax = fig.axes[0]
            ax.set_xlabel("Function Domain")
            ax.set_ylabel("Observed-Pair Consistency")
            _apply_domain_tick_labels(ax)
            _annotate_bars(ax, "{:.2f}")
            _save(fig, figures_dir, "fig_06b_failed_zero_accuracy_consistency_by_domain")
            register("fig_06b_failed_zero_accuracy_consistency_by_domain")

        x_axis = "domain" if failed["domain"].nunique() > 1 else "mode"
        plot_failed = failed.copy()
        if x_axis == "mode":
            plot_failed["mode"] = plot_failed["mode"].astype(str).map(_label_mode)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=plot_failed, x=x_axis, y="internal_consistency_score", hue=hue, palette=palette if hue else None, ax=ax)
        sns.stripplot(data=plot_failed, x=x_axis, y="internal_consistency_score", color="black", size=7, alpha=0.6, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Consistency Distribution for Failed Episodes by Domain")
        ax.set_xlabel("Function Domain" if x_axis == "domain" else "Mode")
        ax.set_ylabel("Observed-Pair Consistency")
        if x_axis == "domain":
            _apply_domain_tick_labels(ax)
        else:
            _style_mode_axis(ax)
        _style_axes(ax)
        _save(fig, figures_dir, "fig_07_failed_zero_accuracy_consistency_distribution")
        register("fig_07_failed_zero_accuracy_consistency_distribution")

        if failed["failure_reason"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.boxplot(data=failed, x="failure_reason", y="internal_consistency_score", hue=hue, palette=palette if hue else None, ax=ax)
            sns.stripplot(data=failed, x="failure_reason", y="internal_consistency_score", color="black", size=7, alpha=0.6, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Consistency by Failure Reason")
            ax.set_xlabel("Failure Reason")
            ax.set_ylabel("Observed-Pair Consistency")
            ticks = ax.get_xticks()
            labels = [_humanize(tick.get_text()) for tick in ax.get_xticklabels()]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            _save(fig, figures_dir, "fig_08_consistency_vs_failure_reason")
            register("fig_08_consistency_vs_failure_reason")

    failure_counts = (
        df.groupby(["model", "failure_reason"], dropna=False, observed=True)["game_id"]
        .count()
        .reset_index(name="episodes")
    )
    if not _skip_if_empty(failure_counts):
        stacked = failure_counts.pivot(index="model", columns="failure_reason", values="episodes").fillna(0)
        fig, ax = plt.subplots(figsize=(11, 6))
        stacked.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Failure Reason Breakdown")
        ax.set_xlabel("Model")
        ax.set_ylabel("Episodes")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [_humanize(label) for label in labels], title="")
        _style_axes(ax, x_is_model=True)
        _save(fig, figures_dir, "fig_09_failure_reason_stacked")
        register("fig_09_failure_reason_stacked")

    if float(df["violated_request_count"].fillna(0).sum()) > 0:
        violation_summary = (
            df.groupby(["mode", "model"], dropna=False, observed=True)["violated_request_count"]
            .mean()
            .reset_index()
        )
        x_axis = "domain" if df["domain"].nunique() > 1 else "mode"
        violation_summary = (
            df.groupby([x_axis] + (["model"] if hue else []), dropna=False, observed=True)["violated_request_count"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=violation_summary, x=x_axis, y="violated_request_count", hue=hue, palette=palette if hue else None, errorbar=None, ax=ax)
        ax.set_title("Average Protocol Violation Count")
        ax.set_xlabel("Function Domain" if x_axis == "domain" else "Mode")
        ax.set_ylabel("Average Violations")
        if x_axis == "domain":
            _apply_domain_tick_labels(ax)
        elif x_axis == "mode":
            _style_mode_axis(ax)
        _style_axes(ax)
        _annotate_bars(ax, "{:.1f}")
        _save(fig, figures_dir, "fig_10_protocol_violation_bar")
        register("fig_10_protocol_violation_bar")

    interactive = df[df["is_interactive"]].copy()
    if not interactive.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        x_axis = "domain" if interactive["domain"].nunique() > 1 else "model"
        sns.boxplot(data=interactive, x=x_axis, y="efficiency_raw", hue=hue, palette=palette if hue else None, ax=ax)
        sns.stripplot(data=interactive, x=x_axis, y="efficiency_raw", color="black", size=7, alpha=0.6, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Efficiency Distribution by Domain")
        ax.set_xlabel("Function Domain" if x_axis == "domain" else "Model")
        ax.set_ylabel("Efficiency")
        if x_axis == "domain":
            _apply_domain_tick_labels(ax)
        _style_axes(ax, x_is_model=(x_axis == "model"))
        _save(fig, figures_dir, "fig_11_efficiency_distribution")
        register("fig_11_efficiency_distribution")

        if interactive["domain"].nunique() > 1:
            domain_eff = interactive.groupby("domain", observed=True)["efficiency_raw"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=domain_eff, x="domain", y="efficiency_raw", color="#5B8E7D", errorbar=None, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Average Efficiency by Domain")
            ax.set_xlabel("Function Domain")
            ax.set_ylabel("Efficiency")
            _apply_domain_tick_labels(ax)
            _annotate_bars(ax, "{:.2f}")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=interactive, x="request_count", y="binary_accuracy", hue="model", palette=palette, ax=ax)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Efficiency / Request Burden vs Exact Guess")
            ax.set_xlabel("Requests Used")
            ax.set_ylabel("Exact Guess")
        _save(fig, figures_dir, "fig_12_efficiency_vs_success")
        register("fig_12_efficiency_vs_success")

    callable_success = (
        df.groupby(["callable", "model"], dropna=False, observed=True)["binary_accuracy"]
        .sum()
        .reset_index()
    )
    if not _skip_if_empty(callable_success):
        callable_success = callable_success.sort_values(["binary_accuracy", "callable"], ascending=[True, True])
        fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * callable_success["callable"].nunique())))
        sns.scatterplot(data=callable_success, x="binary_accuracy", y="callable", hue=hue, palette=palette if hue else None, s=140, ax=ax)
        ax.set_title("Correct Guesses by Function")
        ax.set_xlabel("Solved Functions")
        ax.set_ylabel("Hidden Function")
        _save(fig, figures_dir, "fig_13_callable_success_dotplot")
        register("fig_13_callable_success_dotplot")

    callable_failed = (
        failed.groupby(["callable", "model"], dropna=False, observed=True)["internal_consistency_score"]
        .mean()
        .reset_index()
        if not failed.empty
        else pd.DataFrame()
    )
    if not _skip_if_empty(callable_failed):
        callable_failed = callable_failed.sort_values(["internal_consistency_score", "callable"], ascending=[True, True])
        fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * callable_failed["callable"].nunique())))
        sns.scatterplot(data=callable_failed, x="internal_consistency_score", y="callable", hue=hue, palette=palette if hue else None, s=140, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Callable Failed-Episode Consistency")
        ax.set_xlabel("Observed-Pair Consistency")
        ax.set_ylabel("Hidden Function")
        _save(fig, figures_dir, "fig_14_callable_consistency_dotplot")
        register("fig_14_callable_consistency_dotplot")

    budget_summary = (
        df.groupby(["domain", "guess_outcome"] + (["model"] if hue else []), dropna=False, observed=True)[
            ["budget_used_ratio", "guess_turn"]
        ]
        .mean()
        .reset_index()
    )
    if not _skip_if_empty(budget_summary):
        fig, ax = plt.subplots(figsize=(11, 6))
        palette_outcome = {"right": "#4C956C", "wrong": "#D1495B"}
        sns.barplot(
            data=budget_summary,
            x="domain",
            y="budget_used_ratio",
            hue="guess_outcome",
            palette=palette_outcome,
            errorbar=None,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title("Budget Used Before Guess by Domain and Outcome")
        ax.set_xlabel("Function Domain")
        ax.set_ylabel("Budget Used Ratio")
        _apply_domain_tick_labels(ax)
        _style_axes(ax)
        _annotate_bars(ax, "{:.2f}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [_label_outcome(label) for label in labels], title="")
        _save(fig, figures_dir, "fig_17_budget_used_by_domain_outcome")
        register("fig_17_budget_used_by_domain_outcome")

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(
            data=budget_summary,
            x="domain",
            y="guess_turn",
            hue="guess_outcome",
            palette=palette_outcome,
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Guess Turn by Domain and Outcome")
        ax.set_xlabel("Function Domain")
        ax.set_ylabel("Average Guess Turn")
        _apply_domain_tick_labels(ax)
        _style_axes(ax)
        _annotate_bars(ax, "{:.1f}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [_label_outcome(label) for label in labels], title="")
        _save(fig, figures_dir, "fig_18_guess_turn_by_domain_outcome")
        register("fig_18_guess_turn_by_domain_outcome")

    mode_families = df.copy()
    if mode_families["mode"].astype(str).str.contains("passive_|active_").any():
        mode_families["mode_family"] = mode_families["mode"].astype(str).map(
            lambda mode: (
                "oneshot"
                if mode.endswith("_oneshot")
                else "passive"
                if mode in {"passive_examples", "passive_labeled_pairs"}
                else "active"
            )
        )
        family_summary = (
            mode_families.groupby(["model", "mode_family"], dropna=False, observed=True)["binary_accuracy"]
            .mean()
            .reset_index()
        )
        if family_summary["mode_family"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            family_counts = mode_families.groupby(["model", "mode_family"], dropna=False, observed=True)["binary_accuracy"].sum().reset_index()
            sns.barplot(data=family_counts, x="mode_family", y="binary_accuracy", hue=hue, palette=palette if hue else None, errorbar=None, ax=ax)
            ax.set_title("Correct Guesses by Mode Family")
            ax.set_xlabel("Mode Family")
            ax.set_ylabel("Solved Functions")
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([_humanize(tick.get_text()) for tick in ax.get_xticklabels()], rotation=0)
            _style_axes(ax)
            _save(fig, figures_dir, "fig_15_active_passive_delta")
            register("fig_15_active_passive_delta")

        oneshot_delta = mode_families[mode_families["mode_family"].isin(["oneshot", "active", "passive"])].copy()
        if oneshot_delta["mode_family"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=oneshot_delta, x="mode_family", y="internal_consistency_score", hue=hue, palette=palette if hue else None, errorbar=None, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Interactive vs One-Shot Consistency")
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([_humanize(tick.get_text()) for tick in ax.get_xticklabels()], rotation=0)
            _style_axes(ax)
            _save(fig, figures_dir, "fig_16_oneshot_interactive_delta")
            register("fig_16_oneshot_interactive_delta")

    return outputs
