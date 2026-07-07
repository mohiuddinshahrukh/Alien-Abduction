# Alien Abduction

Code-only repository for Shahrukh Mohiuddin's Clembench game `function_detective`.

`function_detective` is a single-player program induction game. A model sees hidden Python function signature, probes function with inputs, observes outputs, then submits Python code intended to match hidden behavior.

## Repository Layout

- `function_detective/`: game package, prompts, tests, and generated instance files
- `sandbox_runner/`: isolated code runner used during evaluation
- `model_registry.json`: local model registry entries
- `game_registry.json`: game metadata for local Clembench setups
- `run_benchmark.sh`: helper script for local benchmark runs

## Running

Run from repository root after setting up Clembench environment:

```bash
clem run -g function_detective -m <model_name>
```

Run single mode into separate results folder:

```bash
clem run -g function_detective -m <model_name> -i instance_passive_examples -r results/passive_examples
clem run -g function_detective -m <model_name> -i instance_active_inputs -r results/active_inputs
clem run -g function_detective -m <model_name> -i instance_passive_labeled_pairs -r results/passive_labeled_pairs
clem run -g function_detective -m <model_name> -i instance_active_pair_checks -r results/active_pair_checks
clem run -g function_detective -m <model_name> -i instance_passive_examples_oneshot -r results/passive_examples_oneshot
clem run -g function_detective -m <model_name> -i instance_passive_labeled_pairs_oneshot -r results/passive_labeled_pairs_oneshot
```

Run every configured model across every local game and every per-mode instance manifest on macOS:

```bash
./run_everything_macos.command
```

This script runs `clem run`, then `clem transcribe`, `clem score`, `clem eval`, and finally the local analysis pipeline for each generated results folder.

Result layout is normalized after each mode run to:

```text
results/
  <mode>/
    results.csv
    raw.csv
    results.html
    <model_name>/
      run.json
      <experiment_name>/
        experiment.json
        instance_00000/
```

Regenerate instances:

```bash
cd function_detective
python instancegenerator.py
```

## Notes

- Internal runtime identifier stays `function_detective` for compatibility.
- Benchmark outputs do not belong in this repository.
- Secrets such as `key.json` must remain local only.
