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

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO_ROOT/analysis_runs}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs}"

mkdir -p "$RESULTS_ROOT" "$ANALYSIS_ROOT" "$LOGS_DIR"

sanitize() {
  print -r -- "$1" | tr '/ :@' '_' | tr -cd '[:alnum:]_.-'
}

normalize_results_dir() {
  local raw_results_dir="$1"
  "$PYTHON_BIN" "$REPO_ROOT/analysis/normalize_results_layout.py" --results-root "$RESULTS_ROOT"
}

readarray_from_python() {
  local script="$1"
  "$PYTHON_BIN" -c "$script"
}

models=("${(@f)$(readarray_from_python '
import json
from pathlib import Path

repo = Path("'"$REPO_ROOT"'")
for candidate in (repo / "model_registry.json", repo / "function_detective" / "model_registry.json"):
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for entry in data:
            name = entry.get("model_name")
            if name:
                print(name)
        break
')}")

if (( ${#models[@]} == 0 )); then
  echo "No models found in model registry." >&2
  exit 1
fi

games=("${(@f)$(find "$REPO_ROOT" -mindepth 1 -maxdepth 2 -name clemgame.json -print | sed 's#/clemgame.json$##' | sort)}")

if (( ${#games[@]} == 0 )); then
  echo "No local Clembench games found." >&2
  exit 1
fi

for game_dir in "${games[@]}"; do
  game="$(basename "$game_dir")"
  instance_dir="$game_dir/in"
  manifests=()

  if [[ -d "$instance_dir" ]]; then
    while IFS= read -r manifest_path; do
      manifests+=("$(basename "$manifest_path" .json)")
    done < <(find "$instance_dir" -maxdepth 1 -name 'instance_*.json' | sort)
  fi

  if (( ${#manifests[@]} == 0 )); then
    manifests=("instances")
  fi

  for manifest in "${manifests[@]}"; do
    mode_name="${manifest#instance_}"
    raw_results_dir="$RESULTS_ROOT/${game}__${manifest}"
    final_results_dir="$RESULTS_ROOT/${mode_name}"
    mkdir -p "$raw_results_dir"

    echo
    echo "==================================================="
    echo "Game: $game"
    echo "Manifest: $manifest"
    echo "Results: $final_results_dir"
    echo "==================================================="

    for model in "${models[@]}"; do
      safe_model="$(sanitize "$model")"
      log_prefix="$LOGS_DIR/${game}.${manifest}.${safe_model}"
      run_cmd=(clem run -g "$game" -m "$model" -r "$raw_results_dir")
      if [[ "$manifest" != "instances" ]]; then
        run_cmd+=(-i "$manifest")
      fi

      echo "Running model: $model"
      { time "${run_cmd[@]}"; } 2>&1 | tee "${log_prefix}.run.log"
    done

    { time clem transcribe -g "$game" -r "$raw_results_dir"; } 2>&1 | tee "$LOGS_DIR/${game}.${manifest}.transcribe.log"
    { time clem score -g "$game" -r "$raw_results_dir"; } 2>&1 | tee "$LOGS_DIR/${game}.${manifest}.score.log"
    { time clem eval -r "$raw_results_dir"; } 2>&1 | tee "$LOGS_DIR/${game}.${manifest}.eval.log"
    normalize_results_dir "$raw_results_dir"

    analysis_name="$(basename "$final_results_dir")_analysis"
    { time "$PYTHON_BIN" "$REPO_ROOT/analysis/run_all.py" --input "$final_results_dir" --name "$analysis_name"; } 2>&1 | tee "$LOGS_DIR/${game}.${manifest}.analysis.log"
  done
done

echo
echo "Finished all runs, scoring, evaluation, and analysis."
