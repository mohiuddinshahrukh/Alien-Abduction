#!/usr/bin/env python3
"""
Cosmetic-only relabeling for mode/experiment names wherever a human reads
them -- chart titles, tick labels, legends, printed summary tables. Only
"oneshot" is affected, becoming "singleturn" (e.g. "passive_examples_oneshot"
-> "passive_examples_singleturn"); the rest of the name is untouched.

Nothing else changes: the underlying mode/experiment string used for
grouping, filtering (e.g. `mode.endswith("_oneshot")`), and JSON keys/values
is the literal string as it comes from the data everywhere else in this
pipeline -- this helper is applied only at the point a label is handed to
matplotlib or printed, never before.
"""
from math import sqrt
from pathlib import Path


def display_label(value: str) -> str:
    return value.replace("oneshot", "singleturn")


def iter_model_dirs(results_parent: Path):
    """
    Yields the subfolders of `results_parent` that are actually per-model
    result directories -- i.e. have both a results/ (raw interactions.json
    games) and an analysis/ (already-computed base JSON) subfolder -- sorted
    by name. Every plot script in this folder should discover models this
    way instead of `p.is_dir()` alone: results_parent commonly also holds
    sibling output folders from this same pipeline (analysis_overall/ from
    the old analysis/ folder's cross-model pipeline, paper_plots/ from this
    one) that are plain directories too but have neither subfolder, and
    would otherwise show up as a "model" with no data to skip past silently.
    """
    for p in sorted(results_parent.iterdir()):
        if p.is_dir() and (p / "results").is_dir() and (p / "analysis").is_dir():
            yield p


# Cosmetic-only relabeling for model folder names in the cross-model
# "overall" charts (plot_*_overall.py, plot_overall_montage.py) -- shortens
# known long/awkward folder names to something readable on an axis or in a
# legend. Only affects display; results_root folder names, dict keys used
# for joining data across scripts, etc. all stay the raw name everywhere else.
MODEL_DISPLAY_NAMES = {
    "gpt-5.4": "GPT5.4",
    "gpt-5.4-mini": "GPT5.4-mini",
    "Qwen3.6-35B-A3B-FP8-without-reasoning": "Qwen3.6-35B",
    "claude-opus-4-8-azure": "Claude-Opus-4.8",
    "mistral-large-3-azure": "Mistral-Large-3",
}


def display_model_name(model: str) -> str:
    """Falls back to the raw model name unchanged if it isn't in MODEL_DISPLAY_NAMES."""
    return MODEL_DISPLAY_NAMES.get(model, model)


# Even shorter than MODEL_DISPLAY_NAMES -- for space-constrained spots (an
# in-axes text annotation, a compact table) where the full display name
# would crowd everything else out.
MODEL_SHORT_NAMES = {
    "gpt-5.4": "GPT5.4",
    "gpt-5.4-mini": "GPT-mini",
    "Qwen3.6-35B-A3B-FP8-without-reasoning": "Qwen",
    "claude-opus-4-8-azure": "Claude",
    "mistral-large-3-azure": "Mistral",
}


def display_model_name_short(model: str) -> str:
    """Falls back to display_model_name(model) if it isn't in MODEL_SHORT_NAMES."""
    return MODEL_SHORT_NAMES.get(model, display_model_name(model))


# Cosmetic-only relabeling for mode/variant names, used only in the
# cross-model "overall" charts (plot_*_overall.py) -- friendlier variant
# names for a reader who doesn't necessarily know the underlying
# experiment-naming scheme. Every other chart in the pipeline (per-model,
# not "overall") keeps using display_label's lighter oneshot->singleturn
# substitution instead -- this mapping is intentionally scoped to just the
# overall scripts per an explicit request to only change it there for now.
MODE_DISPLAY_NAMES_OVERALL = {
    "active_inputs": "Active-Output",
    "active_pair_checks": "Active-Verdict",
    "passive_examples": "Passive-Output",
    "passive_labeled_pairs": "Passive-Verdict",
    "passive_examples_singleturn": "Single-Turn-Output",
    "passive_labeled_pairs_singleturn": "Single-Turn-Verdict",
}


def display_mode_overall(mode: str) -> str:
    """
    Falls back to display_label(mode) if it isn't in MODE_DISPLAY_NAMES_OVERALL.
    Normalizes through display_label first so "_oneshot" and "_singleturn"
    spellings of the same variant (inconsistent across models' experiment
    naming) both hit the same key.
    """
    normalized = display_label(mode)
    return MODE_DISPLAY_NAMES_OVERALL.get(normalized, normalized)


# Cosmetic-only relabeling for domain names (see plot_coherence_correctness_
# quadrant.py's derive_domain), used only in the overall-analysis turns-taken
# heatmap (plot_turn_level_summary.py, also tiled into the cross-model
# montage) -- friendlier domain names for a reader who doesn't necessarily
# know the underlying experiment-naming scheme.
DOMAIN_DISPLAY_NAMES_OVERALL = {
    "two_numbers": "Number Pairs",
}


def display_domain_overall(domain: str) -> str:
    """Falls back to the raw domain name unchanged if it isn't in DOMAIN_DISPLAY_NAMES_OVERALL."""
    return DOMAIN_DISPLAY_NAMES_OVERALL.get(domain, domain)


# ─── Canonical ordering + color, for every paper-analysis plot ───────────────
#
# Every chart in this folder that puts models or modes on an axis, in a
# legend, or in a groupby should go through sort_models()/sort_modes() (for
# ordering) and MODEL_COLORS/MODE_COLORS (for color) instead of relying on
# whatever order a dict/groupby happens to produce -- otherwise the model/mode
# order and the color assigned to e.g. "GPT5.4" would silently shuffle
# depending on which models are present in a given results_parent.

# x/y-axis order whenever models are the categorical axis, per explicit
# request: Qwen3.6, Mistral-Large-3, GPT5.4-Mini, GPT5.4, Claude.
MODEL_ORDER = [
    "Qwen3.6-35B-A3B-FP8-without-reasoning",
    "mistral-large-3-azure",
    "gpt-5.4-mini",
    "gpt-5.4",
    "claude-opus-4-8-azure",
]


def sort_models(models):
    """
    Returns `models` (any iterable) ordered per MODEL_ORDER. A model not in
    MODEL_ORDER (e.g. a new one added later) is appended at the end,
    alphabetically, rather than raising -- so a chart still renders, just
    with the new model trailing until MODEL_ORDER is updated for it.
    """
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    return known + unknown


# x/y-axis order whenever modes are the categorical axis, per explicit
# request: ActiveOutput, ActiveVerdict, PassiveOutput, PassiveVerdict,
# SingleTurnOutput, SingleTurnVerdict. Raw mode strings -- both the
# "_oneshot" and "_singleturn" spellings map to the same slot, same
# normalization display_mode_overall() already does via display_label().
MODE_ORDER = [
    "active_inputs",
    "active_pair_checks",
    "passive_examples",
    "passive_labeled_pairs",
    "passive_examples_oneshot",
    "passive_labeled_pairs_oneshot",
]


def sort_modes(modes):
    """Returns `modes` ordered per MODE_ORDER (normalized via display_label first
    so "_oneshot"/"_singleturn" spellings collapse to the same slot). A mode not
    in MODE_ORDER is appended at the end, alphabetically."""
    normalized_order = [display_label(m) for m in MODE_ORDER]
    by_normalized = {}
    for m in modes:
        by_normalized.setdefault(display_label(m), []).append(m)
    known = [m for key in normalized_order for m in by_normalized.get(key, [])]
    unknown = sorted(set(modes) - set(known))
    return known + unknown


# A muted, "eye pleasing" categorical palette -- picked to avoid the harsh
# bright red/green/blue/orange primaries of a raw tab10-style palette, then
# adjusted hue-by-hue and validated with the dataviz skill's
# scripts/validate_palette.js (OKLCH lightness band, chroma floor, CVD
# separation, normal-vision separation, contrast vs a light #fcfcfb surface)
# so "muted" doesn't end up meaning "washed out and hard to tell apart" --
# both the full 6-slot set and the first-5 subset pass every check (one WARN:
# slot 4's contrast vs surface sits at 2.53, below the 3:1 relief threshold --
# any chart using it needs a visible legend/tick label nearby, not a bare
# unlabeled fill, which every chart in this folder already has).
PALETTE = [
    "#2a9d8f",  # 1: teal
    "#e76f51",  # 2: coral
    "#3763a0",  # 3: steel-blue
    "#c9992e",  # 4: gold
    "#8859a8",  # 5: mauve
    "#4c9a6a",  # 6: sage
]

# Fixed model -> color, assigned in MODEL_ORDER so a given model always gets
# the same color across every chart in this folder, regardless of which
# other models are present.
MODEL_COLORS = dict(zip(MODEL_ORDER, PALETTE))

# Fixed mode -> color, assigned in MODE_ORDER, same reasoning.
MODE_COLORS = dict(zip(MODE_ORDER, PALETTE))

# Modes come in three families (Active/Passive/Single-Turn), each with an
# Output and a Verdict variant. Per explicit request: Output/Verdict siblings
# share one color (the family's) and are told apart by fill style instead --
# rather than each of the six modes getting its own hue. Use
# mode_family_color()/mode_edge_kwargs() (not MODE_COLORS) for any chart that
# wants this family-grouped look.
MODE_FAMILY = {
    "active_inputs": "active",
    "active_pair_checks": "active",
    "passive_examples": "passive",
    "passive_labeled_pairs": "passive",
    "passive_examples_oneshot": "single_turn",
    "passive_labeled_pairs_oneshot": "single_turn",
}
FAMILY_ORDER = ["active", "passive", "single_turn"]
# Picked (and validated the same way as PALETTE above) per explicit request:
# teal for Active, the same steel-blue used for GPT5.4-mini in
# success_rate_by_mode.png's MODEL_COLORS for Passive, and a lightened rust
# (#BE5103 blended ~10% toward white, per explicit follow-up request) for
# Single-Turn.
FAMILY_COLORS = {
    "active": "#2a9d8f",  # teal
    "passive": "#398cf8",  # steel-blue
    "single_turn": "#C578BF",  # rust (lightened from #BE5103)
}

# Verdict's fill -- each family color blended 35% of the way toward white,
# so the Verdict bar reads as a visibly lighter version of its Output
# sibling even before the dot hatch (below) is added on top.
FAMILY_LIGHT_COLORS = {
    "active": "#9fd8d1",
    "passive": "#aecdf7",
    "single_turn": "#EEC6EB",
}

# Verdict modes are the "_pair_checks"/"_labeled_pairs" half of each family
# (judging a given input/output pair) as opposed to the "_inputs"/"_examples"
# Output half (choosing/reading raw examples).
VERDICT_MODES = {"active_pair_checks", "passive_labeled_pairs", "passive_labeled_pairs_oneshot"}

# Output/Verdict have been through a diagonal hatch, then a dark perimeter
# outline, then a lighter fill tint on its own, then a dark dot hatch on the
# full-saturation fill -- all per explicit feedback on the previous
# attempt. Current approach combines the last two: Verdict is both the
# lighter tint (FAMILY_LIGHT_COLORS) AND has the dot hatch on top, so it's
# distinguishable from Output by fill lightness alone even if a short bar's
# hatch doesn't have room to show a dot. matplotlib draws a hatch using the
# same edgecolor as the bar's own border, which is why an earlier version of
# this also grew a solid black perimeter around every Verdict bar as an
# unintended side effect -- linewidth=0 in mode_hatch_kwargs drops the
# border (hatch strokes are a separate rcParam, hatch.linewidth, so they
# still render).
VERDICT_HATCH = ".."
VERDICT_HATCH_COLOR = "#0b0b0b"


def mode_family_color(mode: str) -> str:
    """The mode's fill color: full-saturation family color for an Output
    mode, a lighter tint of the same family color for its Verdict sibling
    (see FAMILY_COLORS / FAMILY_LIGHT_COLORS above)."""
    family = MODE_FAMILY[mode]
    return FAMILY_LIGHT_COLORS[family] if mode in VERDICT_MODES else FAMILY_COLORS[family]


def mode_hatch_kwargs(mode: str) -> dict:
    """matplotlib bar() kwargs for the Output/Verdict style distinction: a
    dark dot hatch for a Verdict mode, nothing (plain fill) otherwise.
    linewidth=0 so the bar doesn't also pick up a solid border in
    VERDICT_HATCH_COLOR -- edgecolor is needed for the hatch dots themselves,
    but a patch's border and its hatch strokes are two different draws
    (border via linewidth, hatch via the hatch.linewidth rcParam), so
    zeroing linewidth removes only the former."""
    if mode in VERDICT_MODES:
        return {"hatch": VERDICT_HATCH, "edgecolor": VERDICT_HATCH_COLOR, "linewidth": 0}
    return {}


FAMILY_DISPLAY_NAMES = {"active": "Active", "passive": "Passive", "single_turn": "Single-Turn"}


def family_modes(family: str):
    """Returns (output_mode, verdict_mode) -- the two raw mode strings
    belonging to `family` (one of FAMILY_ORDER), Output first."""
    modes_in_family = [m for m, f in MODE_FAMILY.items() if f == family]
    output_mode = next(m for m in modes_in_family if m not in VERDICT_MODES)
    verdict_mode = next(m for m in modes_in_family if m in VERDICT_MODES)
    return output_mode, verdict_mode


# Default single-series accent (e.g. a bar chart with one bar per model and
# no second categorical dimension to color by -- identity is already shown
# via the x-axis label, so every bar takes this same slot-1 hue rather than
# a different color per bar, per the "nominal categorical" rule: color only
# encodes a *second* dimension when there is one).
ACCENT_COLOR = PALETTE[0]

# Success/Loss color pair -- same teal/coral convention already established
# in analysis/plot_last_turn_confidence_overall.py, kept independent here
# since paper-analysis doesn't import from analysis/. For any chart that
# puts both outcomes on one axes (as two series/lines) rather than as
# separate figures.
SUCCESS_COLOR = "#2a9d8f"  # teal
LOSS_COLOR = "#e76f51"  # coral

# Chart chrome -- shared across every paper-analysis plot script so they
# render as one consistent visual system. Same light-mode values as
# analysis/plot_results.py (itself the dataviz skill's reference light
# surface/ink table), kept independent here since paper-analysis doesn't
# import from analysis/.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_OVERALL_SUCCESS = "#2938C2"
INK_MUTED = "#ADA79C"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def style_axes(ax, ylabel, xlabel="Model", fontsize=10):
    """Shared bar/line-chart axis styling -- flat baseline, recessive gridlines,
    no top/right spine, black axis labels/ticks for visibility. Same look
    across every chart in this folder."""
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_PRIMARY, fontsize=fontsize)
    ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=fontsize)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)
    ax.tick_params(colors=INK_PRIMARY, labelsize=9)


# 95% Wilson score interval for a binomial proportion -- shared by any
# paper-analysis chart that wants a confidence interval on a success rate.
# Same formula/reasoning as analysis/plot_success_rate_overall.py's
# wilson_ci: more reliable than the naive normal approximation
# (p +/- z*sqrt(p(1-p)/n)) when n is small (per-mode n is as low as ~10 in
# this dataset) or p sits near 0/1 (common here -- several Qwen modes are
# at or near 0% success) -- the normal approximation can produce an
# interval that dips below 0 or climbs above 1 in exactly those cases,
# Wilson's can't.
Z_95 = 1.96


def wilson_ci(n_success, n, z=Z_95):
    """Returns (lower, upper), both already clipped to [0, 1] fractions."""
    if not n:
        return (0.0, 0.0)
    p = n_success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
