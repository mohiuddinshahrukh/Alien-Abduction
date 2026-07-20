from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from collect_results import collect_results
from derive_metrics import derive_metrics


MODE_LABELS = {
    "active_inputs": "Act_in",
    "active_pair_checks": "Act_in_pairs",
    "passive_examples": "pass_exp",
    "passive_examples_oneshot": "pass_exp_1s",
    "passive_labeled_pairs": "pass_lbl",
    "passive_labeled_pairs_oneshot": "pass_lbl_1s",
}

DOMAIN_LABELS = {
    "numbers": "nums",
    "two_numbers": "2 nums",
    "string": "String",
    "list": "List",
    "logic": "Logic",
}

MODEL_LABELS = {
    "gpt-4.1-nano": "4.1-n",
    "gpt-4.1-mini": "4.1-m",
    "gpt-5-nano": "5-n",
}

FAILURE_LABELS = {
    "correct_guess": "Correct Guess",
    "protocol_or_format_error": "Protocol/Format Error",
    "runtime_error": "Runtime Error",
    "wrong_function_guess": "Wrong Guess",
    "aborted": "Aborted",
}

FAILURE_COLORS = {
    "correct_guess": "#4C956C",
    "protocol_or_format_error": "#F4A259",
    "runtime_error": "#7D5BA6",
    "wrong_function_guess": "#D1495B",
    "aborted": "#8E9AAF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate callable quality-score plots from real results folders. "
            "Writes 1 combined plot plus 1 per mode."
        )
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root directory containing results_* mode folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plot images will be written.",
    )
    parser.add_argument(
        "--metric",
        default="quality_score",
        help="Episode-level metric to aggregate per (callable, model, mode).",
    )
    return parser.parse_args()


def mode_label(mode_name: str) -> str:
    return MODE_LABELS.get(mode_name, mode_name.replace("_", " ").title())


def domain_label(domain_name: str) -> str:
    return DOMAIN_LABELS.get(domain_name, domain_name.replace("_", " ").title())


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def failure_label(failure_name: str) -> str:
    return FAILURE_LABELS.get(failure_name, failure_name.replace("_", " ").title())


def mode_dir_map(results_root: Path) -> dict[str, Path]:
    discovered = {
        mode_name: results_root / f"results_{mode_name}"
        for mode_name in MODE_LABELS
    }
    discovered = {
        mode_name: mode_path
        for mode_name, mode_path in discovered.items()
        if mode_path.exists() and mode_path.is_dir()
    }
    missing = [mode for mode in MODE_LABELS if mode not in discovered]
    if missing:
        raise FileNotFoundError(
            f"Missing results folders for modes: {missing}. Looked under {results_root}"
        )
    return discovered


def load_real_results(results_root: Path, metric: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mode_name, mode_path in mode_dir_map(results_root).items():
        frame = derive_metrics(collect_results(mode_path))
        frame["mode"] = mode_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["callable"] = combined["callable"].astype(str)
    combined["model"] = combined["model"].astype(str)
    combined["mode"] = combined["mode"].astype(str)
    combined["domain"] = combined["domain"].astype(str)
    combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    combined = combined.dropna(subset=[metric, "callable", "model", "mode", "domain"])
    combined["failure_rate"] = (1.0 - combined["binary_accuracy"].astype(float)) * 100.0
    combined["wrong_guess_rate"] = (
        combined["failure_reason"].astype(str).eq("wrong_function_guess").astype(float) * 100.0
    )
    combined["protocol_error_rate"] = (
        (
            (combined["violated_request_count"] > 0)
            | (combined["parse_error_count"] > 0)
        ).astype(float)
        * 100.0
    )
    combined["success_rate"] = combined["binary_accuracy"].astype(float) * 100.0
    return combined


def write_summary_tables(episodes: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "callable_quality_summary.csv", index=False)
    (
        episodes.groupby(["model", "mode", "domain"], observed=True)
        .agg(
            episodes=("quality_score", "size"),
            avg_quality_score=("quality_score", "mean"),
            success_rate=("success_rate", "mean"),
            failure_rate=("failure_rate", "mean"),
            wrong_guess_rate=("wrong_guess_rate", "mean"),
            protocol_error_rate=("protocol_error_rate", "mean"),
            avg_guess_turn=("guess_turn", "mean"),
            avg_violated_request_count=("violated_request_count", "mean"),
        )
        .reset_index()
        .to_csv(output_dir / "model_mode_domain_summary.csv", index=False)
    )


def summarize_by_callable(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        frame.groupby(["callable", "model", "mode", "domain"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"avg_{metric}"})
    )
    return summary


def compute_offsets(names: list[str], span: float) -> dict[str, float]:
    if not names:
        return {}
    offsets = np.linspace(-span, span, num=len(names))
    return {name: float(offset) for name, offset in zip(names, offsets)}


def build_plot_frame(summary: pd.DataFrame, value_column: str, include_mode_offsets: bool) -> pd.DataFrame:
    plot_frame = summary.copy()
    callable_order = (
        plot_frame.groupby("callable", observed=True)[value_column]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    model_order = sorted(plot_frame["model"].unique().tolist())
    mode_order = sorted(plot_frame["mode"].unique().tolist())

    callable_positions = {name: idx for idx, name in enumerate(callable_order)}
    model_offsets = compute_offsets(model_order, span=0.18 if include_mode_offsets else 0.25)
    mode_offsets = compute_offsets(mode_order, span=0.18) if include_mode_offsets else {}

    plot_frame["callable_position"] = plot_frame["callable"].map(callable_positions)
    plot_frame["model_offset"] = plot_frame["model"].map(model_offsets)
    plot_frame["mode_offset"] = plot_frame["mode"].map(mode_offsets).fillna(0.0)
    plot_frame["x_position"] = plot_frame["callable_position"] + plot_frame["model_offset"]
    if include_mode_offsets:
        plot_frame["x_position"] = plot_frame["x_position"] + plot_frame["mode_offset"]
    return plot_frame.sort_values(["callable_position", "mode", "model"])


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ordered_modes(frame: pd.DataFrame) -> list[str]:
    present = set(frame["mode"].astype(str).tolist())
    return [mode_name for mode_name in MODE_LABELS if mode_name in present]


def ordered_domains(frame: pd.DataFrame) -> list[str]:
    present = set(frame["domain"].astype(str).tolist())
    return [domain_name for domain_name in DOMAIN_LABELS if domain_name in present]


def ordered_models(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["model"].astype(str).unique().tolist())


def plot_model_mode_heatmap(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    table = episodes.pivot_table(
        index="model",
        columns="mode",
        values="quality_score",
        aggfunc="mean",
        observed=True,
    ).reindex(index=ordered_models(episodes), columns=ordered_modes(episodes))
    table = table.rename(index=model_label, columns=mode_label)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
    ax.set_title("Average Quality Score by Model and Mode", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    save_figure(fig, output_path)


def plot_domain_mode_heatmaps_by_model(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    models = ordered_models(episodes)
    modes = ordered_modes(episodes)
    domains = ordered_domains(episodes)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.6), squeeze=False)

    for ax, model_name in zip(axes[0], models):
        model_frame = episodes[episodes["model"] == model_name]
        table = model_frame.pivot_table(
            index="domain",
            columns="mode",
            values="quality_score",
            aggfunc="mean",
            observed=True,
        ).reindex(index=domains, columns=modes)
        table = table.rename(index=domain_label, columns=mode_label)
        sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax, cbar=ax is axes[0][-1])
        ax.set_title(model_label(model_name))
        ax.set_xlabel("Mode")
        ax.set_ylabel("Domain")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    fig.suptitle("Average Quality Score by Domain and Mode, Split by Model", y=1.04)
    save_figure(fig, output_path)


def plot_failure_stacked_by_mode(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mode_order = ordered_modes(episodes)
    failure_order = [name for name in FAILURE_LABELS if name in set(episodes["failure_reason"].astype(str))]
    counts = (
        episodes.assign(failure_reason=episodes["failure_reason"].astype(str))
        .groupby(["mode", "failure_reason"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=mode_order, columns=failure_order, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100.0
    shares.index = [mode_label(mode_name) for mode_name in shares.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series(0.0, index=shares.index)
    for failure_name in failure_order:
        values = shares[failure_name]
        ax.bar(
            shares.index,
            values,
            bottom=bottom,
            color=FAILURE_COLORS[failure_name],
            label=failure_label(failure_name),
            width=0.72,
        )
        bottom = bottom + values
    ax.set_title("Outcome Mix by Mode", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.legend(title="Outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    save_figure(fig, output_path)


def plot_failure_stacked_by_model(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    model_order = ordered_models(episodes)
    failure_order = [name for name in FAILURE_LABELS if name in set(episodes["failure_reason"].astype(str))]
    counts = (
        episodes.assign(failure_reason=episodes["failure_reason"].astype(str))
        .groupby(["model", "failure_reason"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=model_order, columns=failure_order, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100.0
    shares.index = [model_label(model_name) for model_name in shares.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = pd.Series(0.0, index=shares.index)
    for failure_name in failure_order:
        values = shares[failure_name]
        ax.bar(
            shares.index,
            values,
            bottom=bottom,
            color=FAILURE_COLORS[failure_name],
            label=failure_label(failure_name),
            width=0.72,
        )
        bottom = bottom + values
    ax.set_title("Outcome Mix by Model", pad=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    save_figure(fig, output_path)


def plot_guess_turn_distribution(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = episodes.copy()
    plot_frame["mode_label"] = plot_frame["mode"].map(mode_label)
    plot_frame["model_label"] = plot_frame["model"].map(model_label)
    mode_labels = [mode_label(mode_name) for mode_name in ordered_modes(plot_frame)]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=plot_frame,
        x="mode_label",
        y="guess_turn",
        hue="model_label",
        order=mode_labels,
        ax=ax,
    )
    sns.stripplot(
        data=plot_frame,
        x="mode_label",
        y="guess_turn",
        hue="model_label",
        order=mode_labels,
        dodge=True,
        alpha=0.35,
        size=3,
        legend=False,
        ax=ax,
    )
    ax.set_title("Guess Turn Distribution by Mode and Model", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Guess Turn")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    save_figure(fig, output_path)


def plot_quality_vs_guess_turn(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = episodes.copy()
    plot_frame["mode_label"] = plot_frame["mode"].map(mode_label)
    plot_frame["model_label"] = plot_frame["model"].map(model_label)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_frame,
        x="guess_turn",
        y="quality_score",
        hue="model_label",
        style="mode_label",
        s=70,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Quality Score vs Guess Turn", pad=14)
    ax.set_xlabel("Guess Turn")
    ax.set_ylabel("Quality Score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    save_figure(fig, output_path)


def plot_protocol_error_rate_heatmap(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = episodes.copy()
    plot_frame["has_protocol_error"] = (
        (plot_frame["violated_request_count"] > 0) | (plot_frame["parse_error_count"] > 0)
    ).astype(float) * 100.0
    table = plot_frame.pivot_table(
        index="model",
        columns="mode",
        values="has_protocol_error",
        aggfunc="mean",
        observed=True,
    ).reindex(index=ordered_models(plot_frame), columns=ordered_modes(plot_frame))
    table = table.rename(index=model_label, columns=mode_label)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="OrRd", vmin=0, vmax=100, ax=ax)
    ax.set_title("Protocol or Format Error Rate by Model and Mode", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    save_figure(fig, output_path)


def plot_domain_failure_heatmap(episodes: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = episodes.copy()
    plot_frame["failure_rate"] = (1.0 - plot_frame["binary_accuracy"].astype(float)) * 100.0
    table = plot_frame.pivot_table(
        index="domain",
        columns="mode",
        values="failure_rate",
        aggfunc="mean",
        observed=True,
    ).reindex(index=ordered_domains(plot_frame), columns=ordered_modes(plot_frame))
    table = table.rename(index=domain_label, columns=mode_label)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=ax)
    ax.set_title("Failure Rate by Domain and Mode", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Domain")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    save_figure(fig, output_path)


def plot_model_mode_domain_heatmaps(
    episodes: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".1f",
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    domains = ordered_domains(episodes)
    models = ordered_models(episodes)
    modes = ordered_modes(episodes)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 4.7), squeeze=False)

    for ax, domain_name in zip(axes[0], domains):
        domain_frame = episodes[episodes["domain"] == domain_name]
        table = domain_frame.pivot_table(
            index="model",
            columns="mode",
            values=metric,
            aggfunc="mean",
            observed=True,
        ).reindex(index=models, columns=modes)
        table = table.rename(index=model_label, columns=mode_label)
        sns.heatmap(
            table,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=ax is axes[0][-1],
        )
        ax.set_title(domain_label(domain_name))
        ax.set_xlabel("Mode")
        ax.set_ylabel("Model")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    fig.suptitle(title, y=1.04)
    save_figure(fig, output_path)


def plot_model_mode_domain_bars(
    episodes: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_frame = (
        episodes.groupby(["domain", "mode", "model"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    plot_frame["domain_label"] = plot_frame["domain"].map(domain_label)
    plot_frame["mode_label"] = plot_frame["mode"].map(mode_label)
    plot_frame["model_label"] = plot_frame["model"].map(model_label)
    domain_order = [domain_label(domain_name) for domain_name in ordered_domains(episodes)]
    mode_order = [mode_label(mode_name) for mode_name in ordered_modes(episodes)]
    models = ordered_models(episodes)
    palette = dict(
        zip(
            [model_label(model_name) for model_name in models],
            sns.color_palette("tab10", n_colors=max(1, len(models))),
        )
    )

    grid = sns.catplot(
        data=plot_frame,
        kind="bar",
        x="mode_label",
        y=metric,
        hue="model_label",
        col="domain_label",
        col_order=domain_order,
        order=mode_order,
        palette=palette,
        height=4.6,
        aspect=0.95,
        errorbar=None,
        sharey=True,
        legend=False,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Mode", y_label)
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=90, pad=10, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_va("center")
            label.set_rotation_mode("anchor")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[model_label(model_name)])
        for model_name in models
    ]
    grid.fig.legend(
        handles,
        [model_label(model_name) for model_name in models],
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )
    grid.fig.suptitle(title, y=1.08)
    grid.fig.subplots_adjust(right=0.88, bottom=0.48, top=0.86, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(output_path, dpi=200, bbox_inches="tight")
    grid.fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(grid.fig)


def plot_combined(summary: pd.DataFrame, value_column: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    plot_frame = build_plot_frame(summary, value_column, include_mode_offsets=True)
    callable_names = plot_frame.sort_values("callable_position")["callable"].drop_duplicates().tolist()
    model_names = sorted(plot_frame["model"].unique().tolist())
    mode_names = sorted(plot_frame["mode"].unique().tolist())

    palette = dict(
        zip(model_names, sns.color_palette("tab10", n_colors=max(1, len(model_names))))
    )
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    marker_map = {mode_name: markers[idx % len(markers)] for idx, mode_name in enumerate(mode_names)}

    fig, ax = plt.subplots(figsize=(max(12, len(callable_names) * 0.35), 7.5))
    for model_name in model_names:
        for mode_name in mode_names:
            subset = plot_frame[
                (plot_frame["model"] == model_name) & (plot_frame["mode"] == mode_name)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["x_position"],
                subset[value_column],
                s=44,
                alpha=0.9,
                color=palette[model_name],
                marker=marker_map[mode_name],
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_title("Average Quality Score by Function, Model, and Mode", pad=14)
    ax.set_xlabel("Function")
    ax.set_ylabel("Average Quality Score")
    ax.set_xticks(range(len(callable_names)))
    ax.set_xticklabels(callable_names, rotation=65, ha="right", fontsize=9)
    ax.margins(x=0.01)

    model_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[model_name],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=model_label(model_name),
        )
        for model_name in model_names
    ]
    mode_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_map[mode_name],
            linestyle="",
            color="#444444",
            markerfacecolor="#444444",
            markersize=8,
            label=mode_label(mode_name),
        )
        for mode_name in mode_names
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=mode_handles,
        title="Mode",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.48),
        frameon=True,
    )

    save_figure(fig, output_path)


def plot_single_mode(summary: pd.DataFrame, mode_name: str, value_column: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    subset = summary[summary["mode"] == mode_name].copy()
    plot_frame = build_plot_frame(subset, value_column, include_mode_offsets=False)
    callable_names = plot_frame.sort_values("callable_position")["callable"].drop_duplicates().tolist()
    model_names = sorted(plot_frame["model"].unique().tolist())
    domain_names = [name for name in DOMAIN_LABELS if name in set(plot_frame["domain"].tolist())]
    palette = dict(
        zip(model_names, sns.color_palette("tab10", n_colors=max(1, len(model_names))))
    )
    domain_markers = {
        "numbers": "o",
        "two_numbers": "s",
        "string": "^",
        "list": "D",
        "logic": "P",
    }

    fig, ax = plt.subplots(figsize=(max(12, len(callable_names) * 0.35), 7.0))
    for model_name in model_names:
        for domain_name in domain_names:
            subset_frame = plot_frame[
                (plot_frame["model"] == model_name) & (plot_frame["domain"] == domain_name)
            ]
            if subset_frame.empty:
                continue
            ax.scatter(
                subset_frame["x_position"],
                subset_frame[value_column],
                s=52,
                alpha=0.9,
                color=palette[model_name],
                marker=domain_markers[domain_name],
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_title(f"Average Quality Score by Function and Model ({mode_label(mode_name)})", pad=14)
    ax.set_xlabel("Function")
    ax.set_ylabel("Average Quality Score")
    ax.set_xticks(range(len(callable_names)))
    ax.set_xticklabels(callable_names, rotation=65, ha="right", fontsize=9)
    ax.margins(x=0.01)

    model_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[model_name],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=model_label(model_name),
        )
        for model_name in model_names
    ]
    domain_handles = [
        plt.Line2D(
            [0], [0],
            marker=domain_markers[domain_name],
            linestyle="",
            color="#444444",
            markerfacecolor="#444444",
            markersize=8,
            label=domain_label(domain_name),
        )
        for domain_name in domain_names
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=domain_handles,
        title="Domain",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.54),
        frameon=True,
    )

    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    episodes = load_real_results(results_root, metric=args.metric)
    summary = summarize_by_callable(episodes, metric=args.metric)
    value_column = f"avg_{args.metric}"

    write_summary_tables(episodes, summary, output_dir)

    plot_combined(summary, value_column, output_dir / "callable_quality_all_modes.png")
    for mode_name in MODE_LABELS:
        plot_single_mode(
            summary=summary,
            mode_name=mode_name,
            value_column=value_column,
            output_path=output_dir / f"callable_quality_{mode_name}.png",
        )
    plot_model_mode_heatmap(episodes, output_dir / "heatmap_quality_model_by_mode.png")
    plot_domain_mode_heatmaps_by_model(episodes, output_dir / "heatmap_quality_domain_by_mode_per_model.png")
    plot_failure_stacked_by_mode(episodes, output_dir / "stacked_outcomes_by_mode.png")
    plot_failure_stacked_by_model(episodes, output_dir / "stacked_outcomes_by_model.png")
    plot_guess_turn_distribution(episodes, output_dir / "guess_turn_distribution_by_mode_model.png")
    plot_quality_vs_guess_turn(episodes, output_dir / "scatter_quality_vs_guess_turn.png")
    plot_protocol_error_rate_heatmap(episodes, output_dir / "heatmap_protocol_error_rate_model_by_mode.png")
    plot_domain_failure_heatmap(episodes, output_dir / "heatmap_failure_rate_domain_by_mode.png")
    plot_model_mode_domain_heatmaps(
        episodes=episodes,
        metric="quality_score",
        title="Quality Score by Model and Mode, Split by Domain",
        output_path=output_dir / "detailed_quality_model_mode_by_domain.png",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
    )
    plot_model_mode_domain_heatmaps(
        episodes=episodes,
        metric="failure_rate",
        title="Failure Rate by Model and Mode, Split by Domain",
        output_path=output_dir / "detailed_failure_rate_model_mode_by_domain.png",
        cmap="Reds",
        vmin=0,
        vmax=100,
    )
    plot_model_mode_domain_heatmaps(
        episodes=episodes,
        metric="wrong_guess_rate",
        title="Wrong-Guess Rate by Model and Mode, Split by Domain",
        output_path=output_dir / "detailed_wrong_guess_rate_model_mode_by_domain.png",
        cmap="OrRd",
        vmin=0,
        vmax=100,
    )
    plot_model_mode_domain_heatmaps(
        episodes=episodes,
        metric="protocol_error_rate",
        title="Protocol Error Rate by Model and Mode, Split by Domain",
        output_path=output_dir / "detailed_protocol_error_rate_model_mode_by_domain.png",
        cmap="OrRd",
        vmin=0,
        vmax=100,
    )
    plot_model_mode_domain_heatmaps(
        episodes=episodes,
        metric="guess_turn",
        title="Average Guess Turn by Model and Mode, Split by Domain",
        output_path=output_dir / "detailed_guess_turn_model_mode_by_domain.png",
        cmap="PuBu",
        vmin=0,
        vmax=15,
    )
    plot_model_mode_domain_bars(
        episodes=episodes,
        metric="quality_score",
        title="Quality Score by Model, Mode, and Domain",
        y_label="Average Quality Score",
        output_path=output_dir / "bars_quality_model_mode_by_domain.png",
    )
    plot_model_mode_domain_bars(
        episodes=episodes,
        metric="failure_rate",
        title="Failure Rate by Model, Mode, and Domain",
        y_label="Failure Rate (%)",
        output_path=output_dir / "bars_failure_rate_model_mode_by_domain.png",
    )


if __name__ == "__main__":
    main()
