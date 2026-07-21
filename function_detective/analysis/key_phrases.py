#!/usr/bin/env python3
"""
Extract per-turn "key phrases" from function_detective's input_rationale text.

Reads input_rationales.json (written by analyze_results.py: one record per
instance/turn with the player's stated reason for that turn's chosen input,
e.g. "Test differing inputs to confirm whether the function is XOR.") and
surfaces, for each turn number, the TF-IDF top terms/phrases across every
instance that reached that turn.

Why TF-IDF instead of raw word counts: input_rationale is mostly strategy
language ("check", "confirm", "distinguish", "verify") that repeats at
every turn regardless of what's actually being tested. Treating each
turn's combined rationale text as one document and scoring it against
every other turn's document downweights those constants and surfaces the
words that are distinctive to a given turn -- which tend to be the
function-type vocabulary the player has converged on by then (e.g. "xor",
"absolute value", "factorial", "distinct count", "identity").

Output: key_phrases_by_turn.json -- one record per turn, top terms sorted
by TF-IDF score descending:
    {"turn": 1, "n_rationales": 19, "top_terms": [["identity", 0.42], ...]}

Usage:
    python key_phrases.py [results_root] [--in-dir IN_DIR] [--out-dir OUT_DIR] [--top-n N]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / "results_5")

# input_rationale is largely experimental-process language ("check whether the
# function...", "verify behavior for this case") that repeats at every turn
# regardless of what's actually being tested -- sklearn's built-in English
# stopword list doesn't cover this domain-specific filler, so it dominated the
# first pass of results (e.g. "check", "simple", "start", "observe" outranking
# any actual function-type vocabulary). These are added on top of ENGLISH_STOP_WORDS
# so terms like "xor", "identity", "absolute value", "factorial", "palindrome"
# have a chance to surface instead.
EXTRA_STOPWORDS = {
    "function", "functions", "check", "checks", "checking", "test", "tests", "testing",
    "verify", "verifying", "confirm", "confirming", "distinguish", "distinguishing",
    "determine", "determining", "observe", "observing", "behavior", "behaviors",
    "case", "cases", "pattern", "patterns", "simple", "simply", "start", "starting",
    "baseline", "output", "outputs", "input", "inputs", "value", "values", "argument",
    "arguments", "exact", "final", "regardless", "whether", "etc", "like", "edge",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS)


def load_rationales_by_turn(path: Path):
    with open(path) as f:
        records = json.load(f)
    by_turn = defaultdict(list)
    for r in records:
        by_turn[r["turn"]].append(r["input_rationale"])
    return by_turn


def top_terms_by_turn(by_turn, top_n=8, ngram_range=(1, 2)):
    """
    One TF-IDF "document" per turn (every instance's rationale text at that
    turn, concatenated) -- so a term's score reflects how distinctive it is
    to that turn's document versus every other turn's document.
    """
    turns = sorted(by_turn)
    documents = [" ".join(by_turn[t]) for t in turns]

    vectorizer = TfidfVectorizer(lowercase=True, stop_words=STOPWORDS, ngram_range=ngram_range, min_df=1)
    matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    results = []
    for row_idx, turn in enumerate(turns):
        row = matrix[row_idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        top_terms = [[terms[i], round(float(row[i]), 4)] for i in top_idx if row[i] > 0]
        results.append({"turn": turn, "n_rationales": len(by_turn[turn]), "top_terms": top_terms})
    return results


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
        help="Directory containing input_rationales.json (default: <results_root>/analysis)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write key_phrases_by_turn.json (default: same as --in-dir)",
    )
    parser.add_argument(
        "--top-n", type=int, default=8,
        help="How many top terms to keep per turn (default: 8)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_root) / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rationales_path = in_dir / "input_rationales.json"
    if not rationales_path.exists():
        raise SystemExit(f"{rationales_path} not found -- run analyze_results.py first.")

    by_turn = load_rationales_by_turn(rationales_path)
    if not by_turn:
        raise SystemExit(f"{rationales_path} has no records.")

    results = top_terms_by_turn(by_turn, top_n=args.top_n)

    out_path = out_dir / "key_phrases_by_turn.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    print()
    for r in results:
        terms_str = ", ".join(f"{term} ({score})" for term, score in r["top_terms"])
        print(f"T{r['turn']} (n={r['n_rationales']}): {terms_str}")


if __name__ == "__main__":
    main()
