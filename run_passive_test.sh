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
  # "gpt-5.4-mini"
  # "gpt-5.4"
  # "claude-opus-4-8-azure"
  "mistral-large-3-azure"
)

instances=(
  "instance_passive_examples_oneshot_allcat_3"
  "instance_passive_examples_oneshot_allcat_5"
  "instance_passive_examples_oneshot_allcat_7"
)

for game in "${games[@]}"; do
  for model in "${models[@]}"; do
    for instance in "${instances[@]}"; do
      echo "Testing ${model} on ${game} with instance ${instance}"
      { time clem run -g "${game}" -m "${model}" -i "${instance}" -t 1 -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.${game}.${model}.${instance}.log"
    done
  done

  { time clem transcribe -g "${game}" -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.transcribe.${game}.log"
  { time clem score -g "${game}" -r "$RESULTS_DIR"; } 2>&1 | tee "logs/runtime.score.${game}.log"
done

echo "Evaluating results"
{ time clem eval -r "$RESULTS_DIR"; }
