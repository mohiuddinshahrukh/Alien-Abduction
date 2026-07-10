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

typeset -a MODE_RESULTS

for candidate in "$REPO_ROOT"/results_*; do
  if [[ -d "$candidate" ]]; then
    MODE_RESULTS+=("$candidate")
  fi
done

if [[ "${INCLUDE_COMBINED_RESULTS:-0}" == "1" && -d "$REPO_ROOT/results" ]]; then
  for candidate in "$REPO_ROOT"/results/*(/); do
    MODE_RESULTS+=("${candidate%/}")
  done
fi

if (( ${#MODE_RESULTS[@]} == 0 )); then
  echo "No per-mode results folders found under $REPO_ROOT" >&2
  exit 1
fi

MODE_RESULTS=(${(on)MODE_RESULTS})

for mode_dir in "${MODE_RESULTS[@]}"; do
  base_name="$(basename "$mode_dir")"
  if [[ "$base_name" == results_* ]]; then
    mode="${base_name#results_}"
  else
    mode="$base_name"
  fi

  echo
  echo "==================================================="
  echo "Analyzing mode: $mode"
  echo "==================================================="

  "$PYTHON_BIN" "$REPO_ROOT/analysis/run_all.py" \
    --input "$mode_dir" \
    --name "$mode/_all_models"

  model_root="$mode_dir"
  if [[ -d "$mode_dir/function_detective" ]]; then
    model_root="$mode_dir/function_detective"
  fi

  for model_dir in "$model_root"/*(/); do
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
