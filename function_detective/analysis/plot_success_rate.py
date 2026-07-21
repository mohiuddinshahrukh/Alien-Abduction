#!/usr/bin/env python3
"""
Success rate per experiment, grouped by mode -- two views of the same
success_rate_by_experiment.json (from success_rate_analysis.py), mirroring
plot_turns_used.py's two chart styles:

    success_rate_by_mode.png / .pdf
        One compact grouped bar chart: x-axis = mode, one cluster of bars
        per mode; each bar within a cluster is one experiment's domain
        (numbers/list/logic/string/two_numbers/...), colored consistently
        across clusters, identified via a shared legend (not text on the
        bars). y-axis fixed to 0-1 (it's a rate).

    success_rate_grid.png / .pdf
        The same data as a grid of small-multiple subplots instead -- one
        subplot per mode (laid out as close to square as the mode count
        allows), each showing its own domains as bars, y-axis fixed 0-1.
        Same domain->color mapping, shared legend once for the whole grid.

    success_rate_mode_only.png / .pdf
        A separate, third chart: success rate pooled across every experiment
        that shares a mode (numbers_active_inputs and list_active_inputs
        both count toward "active_inputs"'s single bar) -- no domain
        breakdown at all, one plain bar per mode. n_success/n are summed
        across experiments before dividing, not averaged per-experiment, so
        a mode's bar reflects its true pooled rate rather than treating
        every experiment as equally sized. Deliberately its own file, not
        folded into either chart above.

One image per model (filename gets a "_<model>" suffix only if more than
one model is present), saved as both PNG (quick preview) and PDF (vector,
for dropping into a document).

Usage:
    python plot_success_rate.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from display_names import display_label
from plot_coherence_correctness_quadrant import derive_domain
from plot_results import CATEGORICAL_PALETTE, SURFACE, _annotate, _legend, _style_axes

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _add_domains(records):
    for r in records:
        r["domain"] = derive_domain(r["experiment"], r["mode"])
    return records


def _domain_colors(records):
    domains = sorted({r["domain"] for r in records})
    if len(domains) > len(CATEGORICAL_PALETTE):
        raise SystemExit(
            f"{len(domains)} domains found but only {len(CATEGORICAL_PALETTE)} palette "
            "slots defined -- fold extras into 'Other' or facet instead."
        )
    return domains, dict(zip(domains, CATEGORICAL_PALETTE))


def _savefig(fig, out_path_base):
    png_path = out_path_base.with_suffix(".png")
    pdf_path = out_path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_success_rate_clustered(records, out_path_base):
    if not records:
        print(f"No success_rate data -- skipping {out_path_base.name}")
        return

    records = _add_domains(records)
    domains, domain_colors = _domain_colors(records)
    models = sorted({r["model"] for r in records})
    modes = sorted({r["mode"] for r in records})

    # Same compact geometry as plot_turns_used.py's clustered chart: bars
    # within a cluster sit close together, clusters keep a visible gap.
    bar_width = 0.8 / max(len(domains), 1)
    fig_width = max(4.0, 1.1 * len(modes))
    fig, axes = plt.subplots(
        len(models), 1, figsize=(fig_width, 4.2 * len(models)), facecolor=SURFACE, squeeze=False,
    )

    for row_idx, model in enumerate(models):
        ax = axes[row_idx][0]
        model_records = [r for r in records if r["model"] == model]

        for domain_idx, domain in enumerate(domains):
            offset = (domain_idx - (len(domains) - 1) / 2) * bar_width
            xs, ys = [], []
            for mode_idx, mode in enumerate(modes):
                match = next((r for r in model_records if r["mode"] == mode and r["domain"] == domain), None)
                if match is None:
                    continue
                xs.append(mode_idx + offset)
                ys.append(match["success_rate"])
            if not xs:
                continue
            ax.bar(xs, ys, width=bar_width * 0.92, color=domain_colors[domain], label=domain, zorder=3)

        panel_title = "Success rate by mode and experiment" if len(models) == 1 else f"{model} -- success rate"
        _style_axes(ax, panel_title, "Success rate", xlabel="Mode")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([display_label(m) for m in modes], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        if row_idx == 0:
            _legend(ax, len(domains))

    fig.tight_layout()
    _savefig(fig, out_path_base)


def _grid_dims(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_success_rate_grid(records, out_path_base):
    if not records:
        print(f"No success_rate data -- skipping {out_path_base.name}")
        return

    records = _add_domains(records)
    domains, domain_colors = _domain_colors(records)
    models = sorted({r["model"] for r in records})

    for model in models:
        model_records = [r for r in records if r["model"] == model]
        modes = sorted({r["mode"] for r in model_records})
        nrows, ncols = _grid_dims(len(modes))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.3 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False,
        )

        for idx, mode in enumerate(modes):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            mode_records = sorted((r for r in model_records if r["mode"] == mode), key=lambda r: r["domain"])
            xs = list(range(len(mode_records)))
            colors = [domain_colors[r["domain"]] for r in mode_records]
            ax.bar(xs, [r["success_rate"] for r in mode_records], color=colors, zorder=3)
            # n= labels disabled -- uncomment to re-enable.
            # for x, r in zip(xs, mode_records):
            #     _annotate(ax, x, r["success_rate"], f"n={r['n']}", domain_colors[r["domain"]], y_offset=6)
            _style_axes(ax, display_label(mode), "Success rate", xlabel="")
            ax.set_xticks(xs)
            ax.set_xticklabels([r["domain"] for r in mode_records], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)

        for idx in range(len(modes), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        legend_handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                   markerfacecolor=domain_colors[d], markeredgecolor="none")
            for d in domains
        ]
        fig.legend(
            legend_handles, domains, loc="upper center", bbox_to_anchor=(0.5, 1.04),
            ncol=min(len(domains), 8), frameon=False, fontsize=9,
        )
        fig.suptitle(
            "Success rate by mode" if len(models) == 1 else f"Success rate by mode -- {model}",
            fontsize=13, fontweight="bold", y=1.1,
        )

        fig.tight_layout()
        suffix = f"_{model}" if len(models) > 1 else ""
        _savefig(fig, out_path_base.parent / f"{out_path_base.stem}{suffix}")


def plot_success_rate_mode_only(records, out_path_base):
    """
    Success rate pooled across every experiment sharing a mode -- no domain
    breakdown, one plain bar per mode. A separate chart from
    plot_success_rate_clustered's mode+domain view; kept as its own file,
    not merged into it.
    """
    if not records:
        print(f"No success_rate data -- skipping {out_path_base.name}")
        return

    models = sorted({r["model"] for r in records})
    modes = sorted({r["mode"] for r in records})
    bar_color = CATEGORICAL_PALETTE[0]

    fig, axes = plt.subplots(
        len(models), 1, figsize=(max(4.0, 1.1 * len(modes)), 4.2 * len(models)), facecolor=SURFACE, squeeze=False,
    )

    for row_idx, model in enumerate(models):
        ax = axes[row_idx][0]
        model_records = [r for r in records if r["model"] == model]

        xs, ys = [], []
        for mode_idx, mode in enumerate(modes):
            mode_records = [r for r in model_records if r["mode"] == mode]
            if not mode_records:
                continue
            n_total = sum(r["n"] for r in mode_records)
            n_success = sum(r["n_success"] for r in mode_records)
            xs.append(mode_idx)
            ys.append(round(n_success / n_total, 2) if n_total else 0.0)

        ax.bar(xs, ys, width=0.6, color=bar_color, zorder=3)

        panel_title = "Success rate by mode" if len(models) == 1 else f"{model} -- success rate by mode"
        _style_axes(ax, panel_title, "Success rate", xlabel="Mode")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([display_label(m) for m in modes], rotation=20, ha="right")
        ax.set_ylim(0, 1)

    fig.tight_layout()
    _savefig(fig, out_path_base)


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_success_rate_clustered(
        _load(in_dir / "success_rate_by_experiment.json"), out_dir / "success_rate_by_mode",
    )
    plot_success_rate_grid(
        _load(in_dir / "success_rate_by_experiment.json"), out_dir / "success_rate_grid",
    )
    plot_success_rate_mode_only(
        _load(in_dir / "success_rate_by_experiment.json"), out_dir / "success_rate_mode_only",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to success_rate_analysis.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing success_rate_by_experiment.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNGs/PDFs (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    plot(in_dir, out_dir)


if __name__ == "__main__":
    main()
