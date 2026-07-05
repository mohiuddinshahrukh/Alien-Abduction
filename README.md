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
clem run -g function_detective -m <model_name> -i instance_example_mode -r results_example_mode
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
