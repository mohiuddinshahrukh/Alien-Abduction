#!/usr/bin/env python3
"""
Plot function_detective turn-level trends produced by analyze_results.py.

Reads the JSON files written by analyze_results.py and produces:

    match_accuracy_by_turn.png
        Mean match rate (predicted vs. observed output) at T1..T10 -- one
        small-multiple panel per model, one line per game `mode` within
        each panel (mode keeps the same color across every panel).

    confidence_by_turn.png
        Mean self-reported confidence_score at T1..T10, faceted the same way.

    unknown_hypothesis_by_turn.png
        Share of instances whose self-reported `hypothesis` is still the
        literal string "unknown" at each turn, faceted the same way. The
        hypothesis field is otherwise free text (one string per player per
        turn -- "identity function", "possibly square or identity-like
        arithmetic function", ...), too varied to plot directly without
        clustering it first; this is the cheap version of that: not
        clustering *what* the guess is, just tracking *whether* the player
        has committed to one yet. A declining line means the player is
        converging on a guess; a flat line stuck near 100% means it never did.

    mode_label_share_by_turn.png
        Share of instances reporting each current_mode label (probing,
        confirming, random, parser_error, ...) at each turn -- a grid of
        small-multiple panels, one row per model and one column per game
        mode, one line per label. Lets you see how the player's
        self-described strategy shifts over the course of a game.

    confidence_by_turn_overall.png
        The confidence_by_turn.png metric again, but as a single combined
        chart instead of one panel per model -- every (model, mode)
        combination as its own line on shared axes, so you can compare
        models/modes directly rather than eyeballing across panels. Two
        categorical dimensions can't both be color (color follows one
        entity only), so model is color and mode is line style
        (solid/dashed/dotted/dash-dot), with two separate legends.

All four of the first charts above are also produced a second time with an
"_by_experiment" suffix (match_accuracy_by_turn_by_experiment.png,
confidence_by_turn_by_experiment.png, unknown_hypothesis_by_turn_by_experiment.png,
mode_label_share_by_turn_by_experiment.png) -- identical chart, but the
within-panel series (or, for mode_label_share, the facet columns) is
`experiment` instead of `mode`. Several experiment folders can share one
mode (numbers_active_inputs and list_active_inputs both run mode
"active_inputs"), so the *_by_turn.png files pool across them and the
*_by_experiment.png files keep each experiment's numbers apart.

Both `model` and `mode`/`experiment` are read out of each interactions.json
rather than assumed from folder names (see analyze_results.py), so a
results tree covering several models and/or several modes/experiments fans
these charts out into more panels automatically -- nothing here is
hardcoded to today's run.

Reads from <results_root>/analysis (analyze_results.py's default output
location) unless --in-dir overrides it -- pass the same results_root you
gave analyze_results.py so the two agree on where the JSON lives.

Turns where the player's response failed to parse (analyze_results.py's
parser_error_counts.json) contribute no match/confidence value, so they'd
otherwise just quietly shrink the sample at that turn. match_accuracy_by_turn.png
and confidence_by_turn.png instead mark them explicitly: a red X below the
x-axis at that turn, labeled with how many instances hit it.

The x-axis always spans turns 1-10 (extended further only if some instance
actually ran longer), even though today's runs top out around turn 8 --
other experiment modes may use more of the 15-turn budget.

Usage:
    python plot_results.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from display_names import display_label

# Fixed-order categorical palette (dataviz skill reference palette, light mode).
# Assigned to categories in sorted order so a given category always gets the
# same color across re-runs, rather than being reshuffled by dict/set order.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# For charts that combine two categorical dimensions (model x mode) on one
# axes: color can only follow one entity, so the second dimension is
# distinguished by line style instead.
LINESTYLES = ["-", "--", ":", "-."]

MIN_TURN_AXIS = 10
PARSER_ERROR_SENTINEL = "parser_error"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _style_axes(ax, title, ylabel, xlabel="Turn"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _set_turn_axis(ax, turns_present):
    max_turn = max(MIN_TURN_AXIS, max(turns_present, default=MIN_TURN_AXIS))
    ax.set_xlim(0.5, max_turn + 0.5)
    ax.set_xticks(range(1, max_turn + 1))


def _legend(ax, n_series, notes=None):
    """
    notes: extra text-only lines appended to the legend (no marker/swatch),
    e.g. clarifying what "n=" means on *this* chart -- the same label shows
    up on charts throughout this pipeline with different denominators
    (instances vs. turns, or two different subsets on the same chart), so
    it's spelled out in the legend it appears in rather than assumed.
    """
    if n_series <= 1 and not notes:
        return
    handles, labels = ax.get_legend_handles_labels()
    for note in notes or []:
        handles.append(Line2D([], [], linestyle="none", marker="none"))
        labels.append(note)
    legend = ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _annotate(ax, x, y, text, color, y_offset=10):
    """
    Small value label anchored to a point, on its own surface-colored patch
    so it stays legible when a line (its own or another series') passes
    behind it -- plain text with no backing tended to blend into the line.
    """
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(0, y_offset),
        ha="center", va="bottom" if y_offset >= 0 else "top",
        fontsize=7, color=color, zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85),
    )


PARSER_ERROR_MARKER_Y = -6
PARSER_ERROR_COLOR = CATEGORICAL_PALETTE[7]  # red -- reserved for this, not a mode/model color


def plot_metric_by_turn(
    agg_records, models, series_values, colors, value_key, n_key, title, ylabel, out_path,
    as_percent=False, parser_error_records=None, series_key="mode",
):
    """
    series_key picks which field distinguishes the lines within a panel --
    "mode" (several experiments can share one mode, e.g. numbers_active_inputs
    and list_active_inputs both run "active_inputs") or "experiment" (the
    finer, per-experiment-folder breakdown). Panels are always faceted by
    model; series_values/colors must already be scoped to that key.
    """
    parser_error_records = parser_error_records or []
    series_colors = dict(zip(series_values, colors))
    all_turns = [row["turn"] for row in agg_records]

    fig, axes = plt.subplots(
        1, len(models), figsize=(7 * len(models), 5),
        facecolor=SURFACE, squeeze=False, sharey=True,
    )
    axes = axes[0]

    for panel_idx, (ax, model) in enumerate(zip(axes, models)):
        model_rows = [r for r in agg_records if r["model"] == model]
        model_series = [s for s in series_values if any(r[series_key] == s for r in model_rows)]

        for series_idx, series_value in enumerate(model_series):
            rows = sorted((r for r in model_rows if r[series_key] == series_value), key=lambda r: r["turn"])
            color = series_colors[series_value]
            x = [r["turn"] for r in rows]
            y = [r[value_key] * (100 if as_percent else 1) for r in rows]

            ax.plot(
                x, y,
                color=color, linewidth=2, solid_capstyle="round",
                marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=display_label(series_value), zorder=3,
            )
            # Offset alternates by series so two series' labels at the same
            # turn don't sit on top of each other.
            y_offset = 10 if series_idx % 2 == 0 else -16
            # n= labels disabled -- uncomment to re-enable.
            # for xi, yi, r in zip(x, y, rows):
            #     _annotate(ax, xi, yi, f"n={int(r[n_key])}", INK_SECONDARY, y_offset)

        # Parser-error turns contribute no match/confidence value, so they'd
        # otherwise just be an unexplained dip in n at that turn. Mark them
        # explicitly on their own row below the axis instead.
        pe_rows = sorted(
            (r for r in parser_error_records if r["model"] == model and r[series_key] in model_series),
            key=lambda r: r["turn"],
        )
        if pe_rows:
            ax.scatter(
                [r["turn"] for r in pe_rows], [PARSER_ERROR_MARKER_Y] * len(pe_rows),
                marker="x", s=70, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
                label="parser error",
            )
            # n= labels disabled -- uncomment to re-enable.
            # for r in pe_rows:
            #     _annotate(
            #         ax, r["turn"], PARSER_ERROR_MARKER_Y,
            #         f"n={int(r['n_parser_error'])}", PARSER_ERROR_COLOR, y_offset=-12,
            #     )

        panel_title = title if len(models) == 1 else f"{title} -- {model}"
        _style_axes(ax, panel_title, ylabel)
        _set_turn_axis(ax, all_turns)
        ax.set_ylim(-14, 105)
        if as_percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v >= 0 else ""))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 0 else ""))
        # One shared legend is enough since series->color is the same in every panel.
        if panel_idx == len(axes) - 1:
            # n= labels disabled elsewhere on this chart -- notes describing them
            # disabled too; uncomment both together to re-enable.
            # notes = ["n = instances contributing at that turn"]
            # if pe_rows:
            #     notes.append("n (red X) = instances with a parser error at that turn")
            _legend(ax, len(series_values) + (1 if pe_rows else 0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _grid_dims(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_metric_by_turn_grid(
    agg_records, models, series_values, value_key, n_key, title, ylabel, out_path,
    as_percent=False, parser_error_records=None, series_key="mode",
):
    """
    Same data as plot_metric_by_turn, for when series_values has too many
    distinct values to overlay as colored lines with a shared legend (more
    than CATEGORICAL_PALETTE has colors for -- typically the "experiment"
    facet, which can run into the dozens once several domains cross several
    modes). One small-multiple subplot per series_value instead, laid out as
    close to square as the count allows, one line per subplot in a single
    consistent color. One figure per model (filename gets a "_<model>"
    suffix only if more than one model is present).
    """
    parser_error_records = parser_error_records or []
    all_turns = [row["turn"] for row in agg_records]
    line_color = CATEGORICAL_PALETTE[0]

    for model in models:
        model_rows = [r for r in agg_records if r["model"] == model]
        model_series = [s for s in series_values if any(r[series_key] == s for r in model_rows)]
        if not model_series:
            continue
        nrows, ncols = _grid_dims(len(model_series))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows),
            facecolor=SURFACE, squeeze=False, sharey=True,
        )

        for idx, series_value in enumerate(model_series):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            rows = sorted((r for r in model_rows if r[series_key] == series_value), key=lambda r: r["turn"])
            x = [r["turn"] for r in rows]
            y = [r[value_key] * (100 if as_percent else 1) for r in rows]

            ax.plot(
                x, y, color=line_color, linewidth=2, solid_capstyle="round",
                marker="o", markersize=6, markerfacecolor=line_color,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
            )
            # n= labels disabled -- uncomment to re-enable.
            # for xi, yi, r in zip(x, y, rows):
            #     _annotate(ax, xi, yi, f"n={int(r[n_key])}", INK_SECONDARY, y_offset=8)

            pe_rows = sorted(
                (r for r in parser_error_records if r["model"] == model and r[series_key] == series_value),
                key=lambda r: r["turn"],
            )
            if pe_rows:
                ax.scatter(
                    [r["turn"] for r in pe_rows], [PARSER_ERROR_MARKER_Y] * len(pe_rows),
                    marker="x", s=50, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
                )
                # n= labels disabled -- uncomment to re-enable.
                # for r in pe_rows:
                #     _annotate(
                #         ax, r["turn"], PARSER_ERROR_MARKER_Y,
                #         f"n={int(r['n_parser_error'])}", PARSER_ERROR_COLOR, y_offset=-12,
                #     )

            _style_axes(ax, display_label(series_value), ylabel)
            _set_turn_axis(ax, all_turns)
            ax.set_ylim(-14, 105)
            if as_percent:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v >= 0 else ""))
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 0 else ""))

        for idx in range(len(model_series), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            title if len(models) == 1 else f"{title} -- {model}", fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        suffix = f"_{model}" if len(models) > 1 else ""
        model_out_path = out_path.parent / f"{out_path.stem}{suffix}{out_path.suffix}"
        fig.savefig(model_out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {model_out_path}")


def plot_overall_confidence(agg_records, parser_error_records, models, modes, out_path):
    """
    Every (model, mode) combination as one line on a single shared axes,
    instead of a small-multiple panel per model -- lets you compare models
    directly instead of eyeballing across separate panels. Color follows
    model (the dataviz-skill rule: color follows one entity only); mode is
    distinguished by line style, with its own separate legend.
    """
    if len(modes) > len(LINESTYLES):
        raise SystemExit(
            f"{len(modes)} modes found but only {len(LINESTYLES)} line styles defined "
            "for the combined overall chart -- fold extra modes into 'Other' or drop this chart."
        )
    model_colors = dict(zip(models, CATEGORICAL_PALETTE))
    mode_styles = dict(zip(modes, LINESTYLES))
    all_turns = [row["turn"] for row in agg_records]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)

    series_idx = 0
    for model in models:
        for mode in modes:
            rows = sorted(
                (r for r in agg_records if r["model"] == model and r["mode"] == mode),
                key=lambda r: r["turn"],
            )
            if not rows:
                continue
            color = model_colors[model]
            x = [r["turn"] for r in rows]
            y = [r["mean_confidence"] for r in rows]
            ax.plot(
                x, y,
                color=color, linestyle=mode_styles[mode], linewidth=2, solid_capstyle="round",
                marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3,
            )
            y_offset = 10 if series_idx % 2 == 0 else -16
            # n= labels disabled -- uncomment to re-enable.
            # for xi, yi, r in zip(x, y, rows):
            #     _annotate(ax, xi, yi, f"n={int(r['n_instances'])}", color, y_offset)
            series_idx += 1

    # Parser errors merged across every (model, mode) at each turn -- this
    # chart is the "overall" view, so per-model attribution is left to
    # confidence_by_turn.png.
    pe_by_turn = {}
    for r in parser_error_records:
        pe_by_turn[r["turn"]] = pe_by_turn.get(r["turn"], 0) + r["n_parser_error"]
    if pe_by_turn:
        xs = sorted(pe_by_turn)
        ax.scatter(
            xs, [PARSER_ERROR_MARKER_Y] * len(xs),
            marker="x", s=70, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
        )
        # n= labels disabled -- uncomment to re-enable.
        # for xi in xs:
        #     _annotate(ax, xi, PARSER_ERROR_MARKER_Y, f"n={int(pe_by_turn[xi])}", PARSER_ERROR_COLOR, y_offset=-12)

    _style_axes(ax, "Self-reported confidence by turn -- all models & modes", "Confidence score (1-100)")
    _set_turn_axis(ax, all_turns)
    ax.set_ylim(-14, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 0 else ""))

    model_handles = [
        Line2D([0], [0], color=model_colors[m], linewidth=2, marker="o",
               markerfacecolor=model_colors[m], markeredgecolor=SURFACE)
        for m in models
    ]
    model_legend = ax.legend(
        model_handles, models, title="Model", loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=False, fontsize=9, title_fontsize=9,
    )
    for text in model_legend.get_texts() + [model_legend.get_title()]:
        text.set_color(INK_SECONDARY)
    ax.add_artist(model_legend)

    mode_handles = [
        Line2D([0], [0], color=INK_SECONDARY, linewidth=2, linestyle=mode_styles[m]) for m in modes
    ]
    mode_labels = [display_label(m) for m in modes]
    if pe_by_turn:
        mode_handles.append(Line2D([0], [0], color=PARSER_ERROR_COLOR, marker="x", linestyle="none", markersize=8))
        mode_labels.append("parser error")
    # n= labels disabled elsewhere on this chart -- this legend-only entry
    # explaining them disabled too; uncomment both together to re-enable.
    # mode_handles.append(Line2D([], [], linestyle="none", marker="none"))
    # mode_labels.append("n = instances contributing at that turn")
    mode_legend_y = 1 - 0.09 * (len(models) + 1.5)
    mode_legend = ax.legend(
        mode_handles, mode_labels, title="Mode", loc="upper left", bbox_to_anchor=(1.02, mode_legend_y),
        frameon=False, fontsize=9, title_fontsize=9,
    )
    for text in mode_legend.get_texts() + [mode_legend.get_title()]:
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overall_confidence_grid(agg_records, parser_error_records, models, modes, out_path):
    """
    Same data as plot_overall_confidence, for when there are more modes than
    LINESTYLES has entries for -- one small-multiple subplot per mode instead
    of overlaying every mode as a linestyle on one shared axes. Color still
    follows model, with one legend shared across the whole grid; each
    subplot only needs a plain line per model since mode is now the facet
    instead of a second visual channel.
    """
    model_colors = dict(zip(models, CATEGORICAL_PALETTE))
    all_turns = [row["turn"] for row in agg_records]
    nrows, ncols = _grid_dims(len(modes))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False, sharey=True,
    )

    pe_by_turn = {}
    for r in parser_error_records:
        pe_by_turn[r["turn"]] = pe_by_turn.get(r["turn"], 0) + r["n_parser_error"]

    for idx, mode in enumerate(modes):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for series_idx, model in enumerate(models):
            rows = sorted(
                (r for r in agg_records if r["model"] == model and r["mode"] == mode),
                key=lambda r: r["turn"],
            )
            if not rows:
                continue
            color = model_colors[model]
            x = [r["turn"] for r in rows]
            y = [r["mean_confidence"] for r in rows]
            ax.plot(
                x, y, color=color, linewidth=2, solid_capstyle="round",
                marker="o", markersize=6, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
            )
            y_offset = 10 if series_idx % 2 == 0 else -16
            # n= labels disabled -- uncomment to re-enable.
            # for xi, yi, r in zip(x, y, rows):
            #     _annotate(ax, xi, yi, f"n={int(r['n_instances'])}", color, y_offset)

        if pe_by_turn:
            xs = sorted(pe_by_turn)
            ax.scatter(
                xs, [PARSER_ERROR_MARKER_Y] * len(xs),
                marker="x", s=50, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
            )

        _style_axes(ax, display_label(mode), "Confidence score (1-100)")
        _set_turn_axis(ax, all_turns)
        ax.set_ylim(-14, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 0 else ""))

    for idx in range(len(modes), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=model_colors[m], linewidth=2, marker="o",
               markerfacecolor=model_colors[m], markeredgecolor=SURFACE)
        for m in models
    ]
    fig.legend(
        legend_handles, models, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=min(len(models), 8), frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Self-reported confidence by turn -- all models & modes", fontsize=13, fontweight="bold", y=1.1,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mode_label_shares(
    label_records, parser_error_records, out_path, facet_key="mode",
    label_key="current_mode_label", exclude_labels=frozenset({PARSER_ERROR_SENTINEL}),
    chart_title="current_mode share by turn",
):
    """
    facet_key picks the column dimension of the panel grid -- "mode" (the
    original, coarser grouping) or "experiment" (one column per experiment
    folder). Rows are always models. label_key picks which field the plotted
    lines are grouped by; exclude_labels drops values from those lines
    (default: parser_error, which is instead drawn as its own red-X overlay
    from parser_error_records -- pass parser_error_records=[] and
    exclude_labels=frozenset() for label sets that don't have that concept).
    """
    if not label_records:
        print(f"No label-share data -- skipping {out_path.name}")
        return

    parser_error_records = parser_error_records or []
    models = sorted({r["model"] for r in label_records})
    facet_values = sorted({r[facet_key] for r in label_records})
    labels = sorted({r[label_key] for r in label_records} - set(exclude_labels))
    label_colors = dict(zip(labels, CATEGORICAL_PALETTE))

    # A single panel uses the full "{chart_title} -- {facet_value}" title (the
    # multi-panel grid uses the much shorter "{model} -- {facet_value}" instead),
    # which can run wider than the default panel width and get clipped -- widen
    # to fit whenever there's only one panel to draw it in.
    panel_width = 7.0
    if len(models) == 1 and len(facet_values) == 1:
        panel_width = max(panel_width, 0.115 * len(f"{chart_title} -- {display_label(facet_values[0])}"))

    fig, axes = plt.subplots(
        len(models), len(facet_values),
        figsize=(panel_width * len(facet_values), 5 * len(models)),
        facecolor=SURFACE, squeeze=False, sharey=True,
    )

    for row_idx, model in enumerate(models):
        for col_idx, facet_value in enumerate(facet_values):
            ax = axes[row_idx][col_idx]
            panel_rows = [
                r for r in label_records if r["model"] == model and r[facet_key] == facet_value
            ]
            all_turns = [r["turn"] for r in panel_rows]

            if not panel_rows:
                ax.set_visible(False)
                continue

            for label_idx, label in enumerate(labels):
                rows = sorted(
                    (r for r in panel_rows if r[label_key] == label),
                    key=lambda r: r["turn"],
                )
                if not rows:
                    continue
                x = [r["turn"] for r in rows]
                y = [r["share"] * 100 for r in rows]
                color = label_colors[label]
                ax.plot(
                    x, y, color=color, linewidth=2, solid_capstyle="round",
                    marker="o", markersize=7, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=label, zorder=3,
                )
                # Offset alternates by label so overlapping lines don't stack labels.
                y_offset = 10 if label_idx % 2 == 0 else -16
                # n= labels disabled -- uncomment to re-enable.
                # for xi, yi, r in zip(x, y, rows):
                #     _annotate(ax, xi, yi, f"n={int(r['count'])}", color, y_offset)

            pe_rows = sorted(
                (r for r in parser_error_records if r["model"] == model and r[facet_key] == facet_value),
                key=lambda r: r["turn"],
            )
            if pe_rows:
                ax.scatter(
                    [r["turn"] for r in pe_rows], [PARSER_ERROR_MARKER_Y] * len(pe_rows),
                    marker="x", s=70, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
                    label="parser error",
                )
                # n= labels disabled -- uncomment to re-enable.
                # for r in pe_rows:
                #     _annotate(
                #         ax, r["turn"], PARSER_ERROR_MARKER_Y,
                #         f"n={int(r['n_parser_error'])}", PARSER_ERROR_COLOR, y_offset=-12,
                #     )

            is_single_panel = len(models) == 1 and len(facet_values) == 1
            facet_display = display_label(facet_value)
            panel_title = f"{chart_title} -- {facet_display}" if is_single_panel else f"{model} -- {facet_display}"
            _style_axes(ax, panel_title, "Share of instances")
            _set_turn_axis(ax, all_turns)
            ax.set_ylim(-14, 105)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v >= 0 else ""))
            # One shared legend is enough since label->color is the same in every panel.
            if row_idx == 0 and col_idx == len(facet_values) - 1:
                # n= labels disabled elsewhere on this chart -- notes describing them
                # disabled too; uncomment both together to re-enable.
                # notes = ["n = instances with that label at that turn"]
                # if pe_rows:
                #     notes.append("n (red X) = instances with a parser error at that turn")
                _legend(ax, len(labels) + (1 if pe_rows else 0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mode_label_shares_grid(label_records, parser_error_records, out_path_base, chart_title="current_mode share by turn"):
    """
    current_mode share by turn, laid out as a proper square-ish grid (see
    _grid_dims) instead of plot_mode_label_shares' one-row-per-model,
    one-column-per-mode layout -- one subplot per mode. One figure per
    model (filename gets a "_<model>" suffix only if more than one model is
    present). Subplot titles are just the mode name -- no model name, since
    that's a per-figure/filename distinction here, not a per-panel one.
    """
    if not label_records:
        print(f"No label-share data -- skipping {out_path_base.name}")
        return

    parser_error_records = parser_error_records or []
    models = sorted({r["model"] for r in label_records})
    labels = sorted({r["current_mode_label"] for r in label_records} - {PARSER_ERROR_SENTINEL})
    label_colors = dict(zip(labels, CATEGORICAL_PALETTE))

    for model in models:
        model_records = [r for r in label_records if r["model"] == model]
        modes = sorted({r["mode"] for r in model_records})
        nrows, ncols = _grid_dims(len(modes))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False, sharey=True,
        )

        for idx, mode in enumerate(modes):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            panel_rows = [r for r in model_records if r["mode"] == mode]
            all_turns = [r["turn"] for r in panel_rows]

            for label in labels:
                rows = sorted((r for r in panel_rows if r["current_mode_label"] == label), key=lambda r: r["turn"])
                if not rows:
                    continue
                x = [r["turn"] for r in rows]
                y = [r["share"] * 100 for r in rows]
                color = label_colors[label]
                ax.plot(
                    x, y, color=color, linewidth=2, solid_capstyle="round",
                    marker="o", markersize=6, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
                )

            pe_rows = sorted(
                (r for r in parser_error_records if r["model"] == model and r["mode"] == mode),
                key=lambda r: r["turn"],
            )
            if pe_rows:
                ax.scatter(
                    [r["turn"] for r in pe_rows], [PARSER_ERROR_MARKER_Y] * len(pe_rows),
                    marker="x", s=50, linewidths=2, color=PARSER_ERROR_COLOR, zorder=4,
                )

            _style_axes(ax, display_label(mode), "Share of instances")
            _set_turn_axis(ax, all_turns)
            ax.set_ylim(-14, 105)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v >= 0 else ""))

        for idx in range(len(modes), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        legend_handles = [
            Line2D([0], [0], color=label_colors[label], linewidth=2, marker="o",
                   markerfacecolor=label_colors[label], markeredgecolor=SURFACE)
            for label in labels
        ]
        legend_labels = list(labels)
        if any(r["model"] == model for r in parser_error_records):
            legend_handles.append(Line2D([0], [0], color=PARSER_ERROR_COLOR, marker="x", linestyle="none", markersize=8))
            legend_labels.append("parser error")
        fig.legend(
            legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
            ncol=min(len(legend_labels), 8), frameon=False, fontsize=9,
        )
        fig.suptitle(chart_title, fontsize=13, fontweight="bold", y=1.1)

        fig.tight_layout()
        suffix = f"_{model}" if len(models) > 1 else ""
        model_out_path = out_path_base.parent / f"{out_path_base.stem}{suffix}{out_path_base.suffix}"
        fig.savefig(model_out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {model_out_path}")


def plot_grouped_charts(in_dir: Path, out_dir: Path, group_key, suffix=""):
    """
    The three per-turn charts (match accuracy, confidence, unknown-hypothesis
    rate), grouped by group_key ("mode" or "experiment") and reading from the
    matching *{suffix}.json files that analyze_results.py's
    compute_aggregates() wrote. Filenames get the same suffix, so the "mode"
    call (suffix="") keeps the original names and the "experiment" call
    (suffix="_by_experiment") writes alongside them without overwriting.

    current_mode label share is NOT one of these -- it's plotted separately,
    only for the "mode" facet (see plot()), not "experiment".

    Returns (models, group_values, agg_records, parser_error_records) so the
    caller can reuse them (currently: the mode-grouped combined-overall chart).
    """
    agg_records = _load(in_dir / f"turn_aggregates{suffix}.json")
    parser_error_records = _load(in_dir / f"parser_error_counts{suffix}.json")

    if not agg_records:
        print(f"No turn_aggregates{suffix}.json data -- skipping {group_key}-grouped charts.")
        return [], [], [], []

    models = sorted({r["model"] for r in agg_records})
    group_values = sorted({r[group_key] for r in agg_records})

    metrics = [
        ("match_rate", "n_instances", "Prediction match accuracy by turn", "Match accuracy",
         f"match_accuracy_by_turn{suffix}.png", True),
        ("mean_confidence", "n_instances", "Self-reported confidence by turn", "Confidence score (1-100)",
         f"confidence_by_turn{suffix}.png", False),
        ("unknown_hypothesis_rate", "n_instances", "Hypothesis still 'unknown' by turn", "Share still 'unknown'",
         f"unknown_hypothesis_by_turn{suffix}.png", True),
    ]

    if len(group_values) > len(CATEGORICAL_PALETTE):
        # Too many distinct values to color individually and keep a legend
        # readable -- one small-multiple subplot per value instead of
        # overlaid colored lines sharing one legend.
        print(
            f"{len(group_values)} distinct {group_key} values found -- too many for one shared-legend "
            f"panel, switching to one small-multiple subplot per {group_key} instead."
        )
        for value_key, n_key, title, ylabel, filename, as_percent in metrics:
            plot_metric_by_turn_grid(
                agg_records, models, group_values, value_key=value_key, n_key=n_key,
                title=title, ylabel=ylabel, out_path=out_dir / filename,
                as_percent=as_percent, parser_error_records=parser_error_records, series_key=group_key,
            )
    else:
        colors = CATEGORICAL_PALETTE[: len(group_values)]
        for value_key, n_key, title, ylabel, filename, as_percent in metrics:
            plot_metric_by_turn(
                agg_records, models, group_values, colors, value_key=value_key, n_key=n_key,
                title=title, ylabel=ylabel, out_path=out_dir / filename,
                as_percent=as_percent, parser_error_records=parser_error_records, series_key=group_key,
            )

    return models, group_values, agg_records, parser_error_records


def plot(in_dir: Path, out_dir: Path):
    turn_details = _load(in_dir / "turn_details.json")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not turn_details:
        print("No successful-instance turn data -- skipping accuracy/confidence/mode-label charts.")
        return

    models, modes, agg_records, parser_error_records = plot_grouped_charts(in_dir, out_dir, "mode")
    plot_grouped_charts(in_dir, out_dir, "experiment", suffix="_by_experiment")

    plot_mode_label_shares_grid(
        _load(in_dir / "mode_label_shares.json"), _load(in_dir / "parser_error_counts.json"),
        out_dir / "mode_label_share_by_turn.png",
    )

    if agg_records:
        if len(modes) > len(LINESTYLES):
            print(
                f"{len(modes)} modes found -- too many for one shared-axes chart with a linestyle "
                "per mode, switching to one small-multiple subplot per mode instead."
            )
            plot_overall_confidence_grid(
                agg_records, parser_error_records, models, modes,
                out_path=out_dir / "confidence_by_turn_overall.png",
            )
        else:
            plot_overall_confidence(
                agg_records, parser_error_records, models, modes,
                out_path=out_dir / "confidence_by_turn_overall.png",
            )


DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=DEFAULT_RESULTS_ROOT,
        help="Same results_root you passed to analyze_results.py -- only used to locate its "
        f"output at <results_root>/analysis (default: {DEFAULT_RESULTS_ROOT}). Ignored if "
        "--in-dir is given.",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing the JSON files from analyze_results.py "
        "(default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNGs (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    plot(in_dir, out_dir)


if __name__ == "__main__":
    main()
