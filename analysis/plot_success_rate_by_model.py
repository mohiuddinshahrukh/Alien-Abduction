#!/usr/bin/env python3
"""
Cross-model success rate, split by mode -- one cluster of bars per MODEL,
one bar per mode within each cluster (the transpose of
plot_success_rate_by_mode.py's mode-on-x-axis layout).

Reads each model's already-computed instance_summary.json (from
analyze_results.py in this same folder):
overall.by_model[model].by_mode[mode].success_rate is used directly, not
recomputed.

Output:
    success_rate_by_model.png / .pdf   grouped bar chart, x-axis = model
                                        (ordered per display_names.sort_models,
                                        horizontal labels), one bar per mode
                                        per model cluster, y-axis = success
                                        rate (%). Each mode's Output/Verdict
                                        pair shares one family color (Active/
                                        Passive/Single-Turn, see
                                        display_names.FAMILY_COLORS) and is
                                        told apart by fill style instead --
                                        Output plain, Verdict with a dark dot
                                        hatch on top (see
                                        display_names.mode_hatch_kwargs) --
                                        rather than each of the six modes
                                        getting its own hue.

    success_rate_by_model_with_ci.png / .pdf   same chart, plus a 95% Wilson
                                        score confidence interval (see
                                        display_names.wilson_ci) as an error
                                        bar on every bar -- n_success/
                                        n_instances come from the same
                                        instance_summary.json each bar's
                                        success_rate already comes from --
                                        plus each model's whole-model average
                                        success rate (every mode pooled) as a
                                        dashed reference line spanning that
                                        model's cluster width, labeled with
                                        its value (0-1, two decimals) directly
                                        above the line. success_rate_by_model.png
                                        itself is left untouched -- this is a
                                        separate file for anyone who wants the
                                        uncertainty shown, without cluttering
                                        the plain chart for anyone who doesn't.

Usage:
    python plot_success_rate_by_model.py <results_parent> [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from display_names import (
    INK_PRIMARY,
    INK_SECONDARY,
    INK_OVERALL_SUCCESS,
    SURFACE,
    display_mode_overall,
    display_model_name,
    iter_model_dirs,
    mode_family_color,
    mode_hatch_kwargs,
    sort_modes,
    sort_models,
    style_axes,
    wilson_ci,
)
from plot_success_rate_by_mode import find_success_rate_by_mode


def plot_success_rate_by_model(data_by_model, out_path_base):
    if not data_by_model:
        print(f"No success rate data found -- skipping {out_path_base.name}")
        return

    models = sort_models(data_by_model.keys())
    all_modes = {mode for per_mode in data_by_model.values() for mode in per_mode}
    modes = sort_modes(all_modes)

    bar_width = 0.8 / len(modes)
    fig_width = max(6.0, 1.8 * len(models))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), facecolor=SURFACE)

    for mode_idx, mode in enumerate(modes):
        offset = (mode_idx - (len(modes) - 1) / 2) * bar_width
        pairs = [
            (model_idx + offset, data_by_model[model].get(mode))
            for model_idx, model in enumerate(models)
            if data_by_model[model].get(mode) is not None
        ]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.bar(
            xs, ys, width=bar_width * 0.92, color=mode_family_color(mode),
            label=display_mode_overall(mode), zorder=3, **mode_hatch_kwargs(mode),
        )

    style_axes(ax, "Success rate (%)", xlabel="Model", fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=0, ha="center", fontsize=11)
    ax.set_ylim(0, 100, auto=True)


    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    png_path = out_path_base.with_suffix(".png")
    pdf_path = out_path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def find_success_rate_by_model_with_counts(results_parent: Path):
    """
    Returns {model: {mode: {"success_rate_pct": v, "n_success": n, "n_instances": n}}}
    -- same instance_summary.json as find_success_rate_by_mode, but keeping
    n_success/n_instances alongside the rate so a Wilson CI can be computed
    per (model, mode) bar.
    """
    result = {}
    for model_dir in iter_model_dirs(results_parent):
        summary_path = model_dir / "analysis" / "instance_summary.json"
        if not summary_path.exists():
            print(f"No {summary_path} -- skipping {model_dir.name} (not analyzed yet?)")
            continue
        with open(summary_path) as f:
            by_model = json.load(f)["overall"]["by_model"]
        for model, entry in by_model.items():
            result[model] = {
                mode: {
                    "success_rate_pct": stats["success_rate"] * 100,
                    "n_success": stats["n_success"],
                    "n_instances": stats["n_instances"],
                }
                for mode, stats in entry["by_mode"].items()
            }
    return result


def find_overall_success_rate_with_counts(results_parent: Path):
    """
    Returns {model: {"success_rate_pct": v, "n_success": n, "n_instances": n}}
    -- each model's whole-model rate (every mode pooled), from
    instance_summary.json's top-level by_model[model] fields (the same
    ones plot_success_rate.py's find_success_rate reads), kept alongside
    n_success/n_instances so a Wilson CI can be drawn for the overall-rate
    reference line too.
    """
    result = {}
    for model_dir in iter_model_dirs(results_parent):
        summary_path = model_dir / "analysis" / "instance_summary.json"
        if not summary_path.exists():
            print(f"No {summary_path} -- skipping {model_dir.name} (not analyzed yet?)")
            continue
        with open(summary_path) as f:
            by_model = json.load(f)["overall"]["by_model"]
        for model, entry in by_model.items():
            result[model] = {
                "success_rate_pct": entry["success_rate"] * 100,
                "n_success": entry["n_success"],
                "n_instances": entry["n_instances"],
            }
    return result


def plot_success_rate_by_model_with_ci(data_by_model, overall_by_model, out_path_base):
    if not data_by_model:
        print(f"No success rate data found -- skipping {out_path_base.name}")
        return

    models = sort_models(data_by_model.keys())
    all_modes = {mode for per_mode in data_by_model.values() for mode in per_mode}
    modes = sort_modes(all_modes)

    bar_width = 0.8 / len(modes)
    # Clusters are spaced model_spacing apart (not the usual 1) so the
    # overall-rate label past each cluster's right edge has room before the
    # next model's bars start -- widening just the gap between clusters,
    # not the bars/offsets within one, which stay in bar_width units.
    model_spacing = 1.15
    fig_width = max(6.0, 1.8 * len(models) * model_spacing)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), facecolor=SURFACE)

    for mode_idx, mode in enumerate(modes):
        offset = (mode_idx - (len(modes) - 1) / 2) * bar_width
        xs, ys, lower_err, upper_err = [], [], [], []
        for model_idx, model in enumerate(models):
            stats = data_by_model[model].get(mode)
            if stats is None:
                continue
            xs.append(model_idx * model_spacing + offset)
            rate = stats["success_rate_pct"] / 100
            ys.append(rate)
            lo, hi = wilson_ci(stats["n_success"], stats["n_instances"])
            lower_err.append(max(0.0, rate - lo))
            upper_err.append(max(0.0, hi - rate))
        if not xs:
            continue
        ax.bar(
            xs, ys, width=bar_width * 0.92, color=mode_family_color(mode),
            label=display_mode_overall(mode), zorder=3, **mode_hatch_kwargs(mode),
        )
        ax.errorbar(
            xs, ys, yerr=[lower_err, upper_err], fmt="none",
            ecolor=INK_PRIMARY, elinewidth=1.2, capsize=3, zorder=5,
        )

    # Each model's whole-model average success rate (every mode pooled), as
    # a dashed reference line spanning that model's cluster width, labeled
    # with its value directly (no CI band -- just the line, per explicit
    # request to drop the shaded hollow around it).
    cluster_half_width = (len(modes) / 2) * bar_width
    label_used = False
    for model_idx, model in enumerate(models):
        overall = overall_by_model.get(model)
        if overall is None:
            continue
        rate = overall["success_rate_pct"] / 100
        center = model_idx * model_spacing
        x_lo, x_hi = center - cluster_half_width, center + cluster_half_width
        ax.plot(
            [x_lo, x_hi], [rate, rate], linestyle="--", color=INK_SECONDARY, linewidth=1.8, zorder=4,
            label="Model overall success rate" if not label_used else None,
        )
        ax.text(
            x_hi + 0.02, rate, f"{rate:.2f}", ha="left", va="center",
            fontsize=10, fontweight="bold", color=INK_OVERALL_SUCCESS, zorder=6,
        )
        label_used = True

    style_axes(ax, "Success rate", xlabel="Model", fontsize=12)
    ax.set_xticks([model_idx * model_spacing for model_idx in range(len(models))])
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=0, ha="center", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    png_path = out_path_base.with_suffix(".png")
    pdf_path = out_path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def run(results_parent: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data_by_model = find_success_rate_by_mode(results_parent)
    plot_success_rate_by_model(data_by_model, out_dir / "success_rate_by_model.png")

    data_by_model_with_counts = find_success_rate_by_model_with_counts(results_parent)
    overall_by_model = find_overall_success_rate_with_counts(results_parent)
    plot_success_rate_by_model_with_ci(
        data_by_model_with_counts, overall_by_model, out_dir / "success_rate_by_model_with_ci.png",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_parent",
        help="Parent directory containing one subfolder per model, each already analyzed "
        "(<results_parent>/<model>/analysis/instance_summary.json)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNG/PDF (default: <results_parent>/paper_plots)",
    )
    args = parser.parse_args()

    results_parent = Path(args.results_parent).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results_parent / "paper_plots"
    run(results_parent, out_dir)


if __name__ == "__main__":
    main()
