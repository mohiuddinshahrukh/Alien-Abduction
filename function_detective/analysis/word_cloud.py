#!/usr/bin/env python3
"""
Render word clouds from key_phrases.py's key_phrases_by_turn.json.

Reads key_phrases_by_turn.json (one record per turn: {"turn": N,
"n_rationales": count, "top_terms": [[term, tfidf_score], ...]}) and produces:

    word_cloud_overall.png
        One combined cloud -- every turn's top_terms pooled, with a term's
        weight summed across all turns it appeared in (so "xor" showing up
        prominently at both turn 2 and turn 6 outweighs a term that only
        ever appeared once). Answers "what vocabulary shows up across this
        whole file", collapsing the turn dimension.

    word_cloud_by_turn.png
        One small-multiple panel per turn, each cloud built only from that
        turn's own top_terms/scores -- so you can see the vocabulary shift
        turn over turn instead of one blended picture.

Word size and color both encode weight (a single sequential blue ramp,
darker = higher-scoring) -- redundant encoding on purpose, since a word
cloud's layout can't carry a legend the way an axis can.

Usage:
    python word_cloud.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_5")
SURFACE = "#fcfcfb"


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _wordcloud(frequencies, width=900, height=500):
    return WordCloud(
        width=width, height=height,
        background_color=SURFACE,
        colormap="Blues",
        prefer_horizontal=0.95,
        relative_scaling=0.4,
        max_words=60,
    ).generate_from_frequencies(frequencies)


def plot_overall(records, out_path):
    combined = {}
    for record in records:
        for term, score in record["top_terms"]:
            combined[term] = combined.get(term, 0.0) + score
    if not combined:
        print(f"No terms found -- skipping {out_path.name}")
        return

    cloud = _wordcloud(combined)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Key phrases across all turns (input_rationale)", color="#0b0b0b",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_by_turn(records, out_path):
    records = [r for r in records if r["top_terms"]]
    if not records:
        print(f"No terms found -- skipping {out_path.name}")
        return

    fig, axes = plt.subplots(
        1, len(records), figsize=(4.5 * len(records), 5), facecolor=SURFACE, squeeze=False,
    )
    axes = axes[0]

    for ax, record in zip(axes, records):
        frequencies = {term: score for term, score in record["top_terms"]}
        cloud = _wordcloud(frequencies, width=500, height=500)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Turn {record['turn']} (n={record['n_rationales']})", color="#0b0b0b",
            fontsize=11, fontweight="bold", pad=10,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=DEFAULT_RESULTS_ROOT,
        help="Same results_root you passed to analyze_results.py/key_phrases.py -- only used "
        f"to locate <results_root>/analysis (default: {DEFAULT_RESULTS_ROOT}). Ignored if "
        "--in-dir is given.",
    )
    parser.add_argument(
        "--in-dir", default=None,
        help="Directory containing key_phrases_by_turn.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the PNGs (default: same as --in-dir)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phrases_path = in_dir / "key_phrases_by_turn.json"
    if not phrases_path.exists():
        raise SystemExit(f"{phrases_path} not found -- run key_phrases.py first.")

    records = _load(phrases_path)
    if not records:
        raise SystemExit(f"{phrases_path} has no records.")

    plot_overall(records, out_dir / "word_cloud_overall.png")
    plot_by_turn(records, out_dir / "word_cloud_by_turn.png")


if __name__ == "__main__":
    main()
