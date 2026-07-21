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


def display_label(value: str) -> str:
    return value.replace("oneshot", "singleturn")
