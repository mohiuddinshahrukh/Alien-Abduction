#!/usr/bin/env python3
"""
Average turns_used per experiment, clustered by mode -- two views of the
same turns_used_by_experiment.json (from turns_used_analysis.py):

    turns_used_by_mode.png / .pdf
        One grouped bar chart: x-axis = mode, one cluster of bars per mode;
        each bar within a cluster is one experiment's domain (numbers/list/
        logic/string/two_numbers/...), colored consistently across
        clusters -- so "numbers" is the same color in every mode's group.
        Good for comparing everything at a glance, but clusters get wide
        once there are many modes.

    turns_used_grid.png / .pdf
        The same data as a grid of small-multiple subplots instead -- one
        subplot per mode (laid out as close to square as the mode count
        allows, e.g. 3x2 for 6 modes), each showing its own domains as
        bars. Same domain->color mapping as the clustered version, shared
        legend once for the whole grid. Easier to read when there are
        several modes, since each mode gets its own uncluttered axes
        instead of a squeezed cluster.

Both are Success-instances-only (see turns_used_analysis.py's docstring),
one image per model (filename gets a "_<model>" suffix only if more than
one model is present), and saved as both PNG (quick preview) and PDF
(vector, for dropping into a document).

Usage:
    python plot_turns_used.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
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


def plot_turns_used_clustered(records, out_path_base):
    if not records:
        print(f"No turns_used data -- skipping {out_path_base.name}")
        return

    records = _add_domains(records)
    domains, domain_colors = _domain_colors(records)
    models = sorted({r["model"] for r in records})
    modes = sorted({r["mode"] for r in records})

    # Compact on purpose: width scales only with the number of mode clusters,
    # not clusters*domains -- bars within a cluster sit close together (wide
    # bar fraction), but clusters themselves keep a visible gap (well under
    # the full 1.0 mode-to-mode spacing) so each mode reads as its own group.
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
                ys.append(match["mean_turns_used"])
            if not xs:
                continue
            # Color alone identifies the domain here; the shared legend below
            # does the labeling instead of text on every bar.
            ax.bar(xs, ys, width=bar_width * 0.92, color=domain_colors[domain], label=domain, zorder=3)

        panel_title = "Average turns used by mode and experiment" if len(models) == 1 else f"{model} -- average turns used"
        _style_axes(ax, panel_title, "Average turns", xlabel="Mode")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([display_label(m) for m in modes], rotation=20, ha="right")
        ax.set_ylim(bottom=0)
        if row_idx == 0:
            _legend(ax, len(domains))

    fig.tight_layout()
    _savefig(fig, out_path_base)


def _grid_dims(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_turns_used_grid(records, out_path_base):
    if not records:
        print(f"No turns_used data -- skipping {out_path_base.name}")
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
            ax.bar(xs, [r["mean_turns_used"] for r in mode_records], color=colors, zorder=3)
            # n= labels disabled -- uncomment to re-enable.
            # for x, r in zip(xs, mode_records):
            #     _annotate(ax, x, r["mean_turns_used"], f"n={r['n']}", domain_colors[r["domain"]], y_offset=6)
            _style_axes(ax, display_label(mode), "Mean turns used", xlabel="")
            ax.set_xticks(xs)
            ax.set_xticklabels([r["domain"] for r in mode_records], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(bottom=0)

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
            "Average turns used by mode" if len(models) == 1 else f"Average turns used by mode -- {model}",
            fontsize=13, fontweight="bold", y=1.1,
        )

        fig.tight_layout()
        suffix = f"_{model}" if len(models) > 1 else ""
        _savefig(fig, out_path_base.parent / f"{out_path_base.stem}{suffix}")


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_turns_used_clustered(
        _load(in_dir / "turns_used_by_experiment.json"), out_dir / "turns_used_by_mode",
    )
    plot_turns_used_grid(
        _load(in_dir / "turns_used_by_experiment.json"), out_dir / "turns_used_grid",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to turns_used_analysis.py (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing turns_used_by_experiment.json (default: <results_root>/analysis)",
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
