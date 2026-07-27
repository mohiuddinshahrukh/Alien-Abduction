#!/bin/bash

set -euo pipefail

if [ -d ".venv" ]; then
  # Local virtualenv is optional. Script still works if user activates env manually.
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

#export PYTHONPATH=.:$PYTHONPATH

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p logs "$RESULTS_DIR"

games=(
  "function_detective"
)

models=(
  #"gpt-5.4-mini"
  #"gpt-5.4"
  "claude-opus-4-8-azure"
  #"mistral-large-3-azure"
)

for game in "${games[@]}"; do
  for model in "${models[@]}"; do
    echo "Testing ${model} on ${game}"
    { time clem run -g "${game}" -m "${model}" -t 1 -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.${game}.${model}.log"
    { time clem transcribe -g "${game}" -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.transcribe.${game}.${model}.log"
    { time clem score -g "${game}" -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.score.${game}.${model}.log"
  done
done

echo "Evaluating results"
{ time clem eval -r "$RESULTS_DIR"; }
