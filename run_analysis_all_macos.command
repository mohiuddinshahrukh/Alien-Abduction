#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -d "$REPO_ROOT/.venv" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
elif [[ -d "$REPO_ROOT/venv" ]]; then
  source "$REPO_ROOT/venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "Results directory not found: $RESULTS_ROOT" >&2
  exit 1
fi

for mode_dir in "$RESULTS_ROOT"/*(/); do
  mode="$(basename "$mode_dir")"

  echo
  echo "==================================================="
  echo "Analyzing mode: $mode"
  echo "==================================================="

  "$PYTHON_BIN" "$REPO_ROOT/analysis/run_all.py" \
    --input "$mode_dir" \
    --name "$mode/_all_models"

  for model_dir in "$mode_dir"/*(/); do
    model="$(basename "$model_dir")"

    echo "Analyzing model: $model"
    "$PYTHON_BIN" "$REPO_ROOT/analysis/run_all.py" \
      --input "$mode_dir" \
      --model "$model" \
      --name "$mode/$model"
  done
done

echo
echo "Finished analysis for all modes and models."
