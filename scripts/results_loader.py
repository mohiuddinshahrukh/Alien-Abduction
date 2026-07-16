"""
results_loader.py

Walk a clembench-style results tree (e.g. results_gpt/<model>/function_detective/
<experiment>/instance_XXXXX/) and flatten every episode into a tidy table with one
row per episode: its signature/category, game variant, and all episode metrics.

The experiment name encodes both axes we care about:
    <category>_<variant>
        category  in {numbers, two_numbers, string, list, logic}
        variant   in {active_inputs, passive_examples, active_pair_checks,
                      passive_labeled_pairs, passive_examples_oneshot,
                      passive_labeled_pairs_oneshot}

Use from the command line to dump a CSV, or import `load_results` / `load_long`.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical ordering + human labels (keeps every figure's axes consistent)
# ---------------------------------------------------------------------------
CATEGORY_ORDER = ["NUMBERS", "TWO_NUMBERS", "STRING", "LIST", "LOGIC"]

CATEGORY_SIGNATURE = {
    "NUMBERS": "(x: int) -> int",
    "TWO_NUMBERS": "(a: int, b: int) -> int",
    "STRING": "(text: str) -> str",
    "LIST": "(items: List[int]) -> int",
    "LOGIC": "(a: bool, b: bool) -> bool",
}

# The 6 game variants, in a deliberate order: active / passive / oneshot pairs.
VARIANT_ORDER = [
    "active_inputs",
    "active_pair_checks",
    "passive_examples",
    "passive_labeled_pairs",
    "passive_examples_oneshot",
    "passive_labeled_pairs_oneshot",
]

VARIANT_LABEL = {
    "active_inputs": "Active I/O",
    "active_pair_checks": "Active Pair-Check",
    "passive_examples": "Passive I/O",
    "passive_labeled_pairs": "Passive Pairs",
    "passive_examples_oneshot": "Oneshot I/O",
    "passive_labeled_pairs_oneshot": "Oneshot Pairs",
}

# Higher-level groupings, useful for rollups.
VARIANT_FAMILY = {  # interaction style
    "active_inputs": "active",
    "active_pair_checks": "active",
    "passive_examples": "passive",
    "passive_labeled_pairs": "passive",
    "passive_examples_oneshot": "oneshot",
    "passive_labeled_pairs_oneshot": "oneshot",
}

VARIANT_TASK = {  # probe type
    "active_inputs": "io",
    "passive_examples": "io",
    "passive_examples_oneshot": "io",
    "active_pair_checks": "membership",
    "passive_labeled_pairs": "membership",
    "passive_labeled_pairs_oneshot": "membership",
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Return one row per episode found anywhere under `results_dir`.

    Columns include: model, category, signature, variant, variant_label, family,
    task_type, difficulty, callable, game_id, plus every episode-score metric and
    per-round summaries (round_count, mean_request_success).
    """
    rows: List[Dict[str, Any]] = []

    for dirpath, _dirnames, filenames in os.walk(results_dir):
        if "scores.json" not in filenames:
            continue

        scores = _load_json(os.path.join(dirpath, "scores.json"))
        meta = scores.get("meta", {})
        episode = scores.get("episode scores", {})

        # Prefer instance.json for the category/variant ground truth.
        instance: Dict[str, Any] = {}
        inst_path = os.path.join(dirpath, "instance.json")
        if os.path.exists(inst_path):
            instance = _load_json(inst_path)

        category = str(instance.get("category") or "").upper()
        variant = str(instance.get("mode") or "")

        # Fall back to parsing the experiment folder name: <category>_<variant>.
        exp_name = meta.get("experiment_name") or os.path.basename(os.path.dirname(dirpath))
        if not category or not variant:
            category, variant = _split_experiment_name(exp_name)

        # Per-round request-success summary.
        round_scores = scores.get("round scores", {})
        rsr = [
            r.get("Request Success Ratio")
            for r in round_scores.values()
            if r.get("Request Success Ratio") is not None
        ]
        mean_rsr = sum(rsr) / len(rsr) if rsr else None

        # Player model name (guesser).
        model_name = meta.get("results_folder")
        for player in scores.get("players", {}).values():
            if player.get("game_role") == "FunctionGuesser":
                model_name = player.get("model_name", model_name)

        row: Dict[str, Any] = {
            "model": model_name,
            "experiment": exp_name,
            "category": category,
            "signature": CATEGORY_SIGNATURE.get(category, ""),
            "variant": variant,
            "variant_label": VARIANT_LABEL.get(variant, variant),
            "family": VARIANT_FAMILY.get(variant, "other"),
            "task_type": VARIANT_TASK.get(variant, "other"),
            "difficulty": instance.get("difficulty"),
            "callable": instance.get("callable"),
            "game_id": meta.get("game_id"),
            "round_count": meta.get("round_count"),
            "mean_request_success": mean_rsr,
            "episode_dir": dirpath,
        }
        # Merge in every episode-score metric verbatim.
        for key, value in episode.items():
            row[key.replace(" ", "_")] = value

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No scores.json found under {results_dir!r}")

    df = pd.DataFrame(rows)

    # Normalize accuracy to a 0-100 scale (raw scores store it as 0-1), so every
    # downstream script and the exported CSV report accuracy on the same 0-100
    # scale as Main Score / efficiency.
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce") * 100.0

    # Success rate (0-100): binary_success is 0/1 per episode; averaged over a
    # cell this becomes the % of targets solved exactly. This is the primary
    # metric used for comparisons against efficiency.
    if "binary_success" in df.columns:
        df["success_rate"] = pd.to_numeric(df["binary_success"], errors="coerce") * 100.0

    # Categoricals with our canonical ordering, so plots sort correctly.
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["variant_label"] = pd.Categorical(
        df["variant_label"],
        categories=[VARIANT_LABEL[v] for v in VARIANT_ORDER],
        ordered=True,
    )
    return df.sort_values(["category", "variant"]).reset_index(drop=True)


def _split_experiment_name(exp_name: str) -> tuple[str, str]:
    """'two_numbers_passive_examples_oneshot' -> ('TWO_NUMBERS', 'passive_examples_oneshot')."""
    lowered = exp_name.lower()
    # Category prefixes are known; match the longest one first.
    for cat in sorted(CATEGORY_ORDER, key=len, reverse=True):
        prefix = cat.lower() + "_"
        if lowered.startswith(prefix):
            return cat, lowered[len(prefix):]
    return "", exp_name


def pivot_metric(df: pd.DataFrame, metric: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Signatures (rows) x variants (cols) table for one metric."""
    table = df.pivot_table(
        index="category", columns="variant_label", values=metric, aggfunc=aggfunc, observed=False
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten a results tree to a CSV.")
    parser.add_argument("--results-dir", default="results_gpt",
                        help="Root of the results tree to scan.")
    parser.add_argument("--out", default="scripts/episodes.csv",
                        help="Where to write the flattened CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = load_results(args.results_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Loaded {len(frame)} episodes from {args.results_dir!r}")
    print(f"  models:     {sorted(frame['model'].dropna().unique())}")
    print(f"  categories: {list(frame['category'].cat.categories)}")
    print(f"  variants:   {list(frame['variant'].cat.categories)}")
    print(f"Wrote {args.out}")
