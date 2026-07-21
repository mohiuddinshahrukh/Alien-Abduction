#!/usr/bin/env python3
"""
High-level quadrant view of hypothesis_consistency_analysis.py's output: one
point per experiment (the "variant" unit -- e.g. numbers_active_inputs,
list_passive_examples, ...), plotting:

    x = correctness rate  -- share of scoreable turns where hypothesis(input)
                             matched gm_output (the truth)
    y = coherence rate    -- share of scoreable turns where hypothesis(input)
                             matched predicted_output (what the player said)

"Scoreable" excludes turns scored unknown (-1) or not_executable (-100) --
those measure whether a hypothesis existed and ran at all, which is a
different question from whether it was right. A point's marker size
instead encodes scoreable_share (n_scoreable / n_total), so an experiment
where most turns were unknown/not_executable reads as a small, less
trustworthy point rather than being silently dropped or hidden in the rate.

The 50/50 lines split the plane into the same four categories as the
detailed per-turn grid (coherence_correctness_grid.png), now as regions
instead of per-cell colors:
    top-right    converged             (coherent and correct)
    top-left     honestly wrong        (coherent, not correct)
    bottom-right right despite itself  (correct, not coherent)
    bottom-left  decoupled             (neither)

Points are colored by mode and shaped by domain (numbers, list, logic,
string, two_numbers, ...) -- two independent categorical encodings instead
of per-point text labels, so this scales cleanly once other variants
(passive_examples, active_pair_checks, ...) are analyzed alongside
active_inputs: same domain keeps its marker shape across modes (so you can
track e.g. "numbers" across variants by shape alone), same mode keeps its
color across domains. Labels would start overlapping well before a second
mode's worth of points showed up; this doesn't.

Plain/print styling on purpose (white background, no gradients) so the
figure drops into a LaTeX document via \\includegraphics without further
editing. Saves both .png and .pdf.

Usage:
    python plot_coherence_correctness_quadrant.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from display_names import display_label

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_1")

# Plain, print-safe categorical colors -- same family as coherence_correctness_grid.py,
# but this is a separate fixed order (one slot per mode, not per outcome category).
MODE_COLORS = ["#3a7ebf", "#3a923a", "#8064a2", "#c0504d", "#e0a030", "#4bacc6", "#d98cb3", "#808080"]

# One marker per domain (the part of the experiment name before the mode suffix,
# e.g. "numbers_active_inputs" -> "numbers"). Fixed order so a domain keeps the
# same shape across charts/runs; extra domains beyond this list get assigned
# the remaining shapes in FALLBACK_MARKERS, in sorted order, so a brand new
# domain never crashes the script -- it just gets the next unused shape.
DOMAIN_MARKERS = {
    "numbers": "o", "two_numbers": "s", "string": "^", "list": "D", "logic": "v",
}
FALLBACK_MARKERS = ["P", "X", "*", "h", "8", "p", "<", ">"]

INK = "#0b0b0b"
MUTED = "#7f7f7f"
GRID_LINE = "#d9d9d9"

MIN_MARKER, MAX_MARKER = 60, 500


def derive_domain(experiment, mode):
    """"numbers_active_inputs", mode="active_inputs" -> "numbers". Falls back to
    the full experiment name if it doesn't end with "_<mode>" for some reason."""
    suffix = f"_{mode}"
    if experiment.endswith(suffix):
        return experiment[: -len(suffix)]
    return experiment


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def compute_variant_points(records):
    """One point per (model, experiment, mode): coherence/correctness rate over
    scoreable turns (score 0 or 1), plus scoreable_share for marker sizing."""
    groups = {}
    for r in records:
        key = (r["model"], r["experiment"], r["mode"])
        groups.setdefault(key, []).append(r)

    points = []
    for (model, experiment, mode), rows in groups.items():
        n_total = len(rows)
        scoreable = [r for r in rows if r["consistency_score"] in (0, 1)]
        n_scoreable = len(scoreable)
        if n_scoreable == 0:
            coherence_rate = correctness_rate = None
        else:
            coherence_rate = sum(r["consistency_score"] == 1 for r in scoreable) / n_scoreable
            correctness_rate = sum(r["correctness_score"] == 1 for r in scoreable) / n_scoreable
        points.append(
            {
                "model": model, "experiment": experiment, "mode": mode,
                "domain": derive_domain(experiment, mode),
                "n_total": n_total, "n_scoreable": n_scoreable,
                "scoreable_share": n_scoreable / n_total,
                "coherence_rate": coherence_rate, "correctness_rate": correctness_rate,
            }
        )
    return points


def plot_quadrant(points, out_path_base):
    points = [p for p in points if p["coherence_rate"] is not None]
    if not points:
        print("No scoreable points -- skipping quadrant plot")
        return

    modes = sorted({p["mode"] for p in points})
    mode_colors = dict(zip(modes, MODE_COLORS))

    domains = sorted({p["domain"] for p in points})
    domain_markers = {}
    next_fallback = 0
    for domain in domains:
        if domain in DOMAIN_MARKERS:
            domain_markers[domain] = DOMAIN_MARKERS[domain]
        else:
            domain_markers[domain] = FALLBACK_MARKERS[next_fallback % len(FALLBACK_MARKERS)]
            next_fallback += 1

    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="white")

    ax.axhline(50, color=GRID_LINE, linewidth=1, zorder=1)
    ax.axvline(50, color=GRID_LINE, linewidth=1, zorder=1)

    quadrant_labels = [
        (97, 97, "converged", "right", "top"),
        (3, 97, "honestly wrong", "left", "top"),
        (97, 3, "right despite itself", "right", "bottom"),
        (3, 3, "decoupled", "left", "bottom"),
    ]
    for x, y, label, ha, va in quadrant_labels:
        ax.annotate(label, (x, y), ha=ha, va=va, fontsize=9, color=MUTED, style="italic")

    for mode in modes:
        for domain in domains:
            group_points = [p for p in points if p["mode"] == mode and p["domain"] == domain]
            if not group_points:
                continue
            xs = [p["correctness_rate"] * 100 for p in group_points]
            ys = [p["coherence_rate"] * 100 for p in group_points]
            sizes = [MIN_MARKER + p["scoreable_share"] * (MAX_MARKER - MIN_MARKER) for p in group_points]
            ax.scatter(
                xs, ys, s=sizes, marker=domain_markers[domain], color=mode_colors[mode],
                edgecolor="white", linewidth=1, alpha=0.85, zorder=3,
            )

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Correctness rate (hypothesis(input) == gm_output)", fontsize=10, color=INK)
    ax.set_ylabel("Coherence rate (hypothesis(input) == predicted_output)", fontsize=10, color=INK)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_LINE)

    # Two independent legends -- color (mode) and marker shape (domain) -- since
    # a single legend can't show two categorical dimensions crossed like this.
    mode_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=mode_colors[m], markeredgecolor="white")
        for m in modes
    ]
    mode_legend = ax.legend(
        mode_handles, [display_label(m) for m in modes], title="Mode", loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=False, fontsize=9, title_fontsize=9,
    )
    ax.add_artist(mode_legend)

    domain_handles = [
        Line2D([0], [0], marker=domain_markers[d], linestyle="none", markersize=9,
               markerfacecolor=MUTED, markeredgecolor="white")
        for d in domains
    ]
    domain_legend_y = 1 - 0.075 * (len(modes) + 1.8)
    ax.legend(
        domain_handles, domains, title="Domain", loc="upper left", bbox_to_anchor=(1.02, domain_legend_y),
        frameon=False, fontsize=9, title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path_base.with_suffix('.png')}")
    print(f"Wrote {out_path_base.with_suffix('.pdf')}")


def plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    consistency_path = in_dir / "hypothesis_consistency.json"
    if not consistency_path.exists():
        raise SystemExit(f"{consistency_path} not found -- run hypothesis_consistency_analysis.py first.")

    records = _load(consistency_path)
    points = compute_variant_points(records)

    print("Per-experiment coherence/correctness rates:")
    for p in sorted(points, key=lambda p: (p["mode"], p["experiment"])):
        label = f"{display_label(p['mode'])}/{display_label(p['experiment'])}"
        if p["coherence_rate"] is None:
            print(f"  {label}: no scoreable turns")
        else:
            print(
                f"  {label}: coherence={p['coherence_rate']*100:.0f}% "
                f"correctness={p['correctness_rate']*100:.0f}% "
                f"scoreable={p['n_scoreable']}/{p['n_total']}"
            )

    plot_quadrant(points, out_dir / "coherence_correctness_quadrant")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root", nargs="?", default=DEFAULT_RESULTS_ROOT,
        help=f"Same results_root you passed to hypothesis_consistency_analysis.py "
        f"(default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing hypothesis_consistency.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the figure (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    plot(in_dir, out_dir)


if __name__ == "__main__":
    main()
