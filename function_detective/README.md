# Function Detective

Implemented by: Shahrukh Mohiuddin

In this game a single player probes a hidden Python function through input-output interaction and must infer a functionally equivalent implementation. The player receives the function signature, can test candidate inputs, observes the resulting outputs, and eventually submits Python code as the final solution.

The game targets inductive reasoning over program behavior. It measures whether a model can choose informative probes, extract a rule from sparse observations, and produce code that matches the hidden function on held-out tests.

### Instantiation

We instantiate the game with function families grouped by difficulty and domain. Each experiment specifies a maximum number of turns and includes pre-generated hidden-function test cases used for evaluation. Instances are created from the registry in `functions.py`, and each instance stores the hidden callable name, signature, category, and evaluation tests.

### Evaluation

We measure the following metrics at the episode level:

1. **Success**: Whether the submitted solution functionally matches the hidden function on static tests.
2. **Abort**: Whether the interaction was aborted because of invalid behavior.
3. **Efficiency**: How quickly the player solved the task relative to the turn budget.
4. **Accuracy**: The final held-out test accuracy of the submitted solution.
5. **Internal Consistency**: Whether the submitted solution agrees with the outputs observed during probing.
6. **Probe Diversity**: How varied the tested inputs and resulting outputs were across the episode.

### Package Structure

The game package follows the standard Clembench layout:

- `master.py`: game master, scorer, benchmark, and runtime validation logic
- `instancegenerator.py`: instance generation from the hidden-function registry
- `functions.py`: hidden functions and registry metadata
- `utils.py`: test generation, parsing, and code validation helpers
- `protocol.py`: interaction tags used in prompts and responses
- `resources/`: prompt templates and other game assets
- `in/instances.json`: generated benchmark instances across all modes
- `in/instance_passive_examples.json`: only Passive Examples instances
- `in/instance_active_inputs.json`: only Active Inputs instances
- `in/instance_passive_labeled_pairs.json`: only Passive Labeled Pairs instances
- `in/instance_active_pair_checks.json`: only Active Pair Checks instances
- `in/instance_passive_examples_oneshot.json`: only Passive Examples One-Shot instances
- `in/instance_passive_labeled_pairs_oneshot.json`: only Passive Labeled Pairs One-Shot instances

### Running

Generate instances from the game directory:

```bash
python instancegenerator.py
```

Run the game with Clembench:

```bash
clem run -g function_detective -m <model_name>
```

Run a single mode by selecting its instance file:

```bash
clem run -g function_detective -m <model_name> -i instance_passive_examples -r results_passive_examples
clem run -g function_detective -m <model_name> -i instance_active_inputs -r results_active_inputs
clem run -g function_detective -m <model_name> -i instance_passive_labeled_pairs -r results_passive_labeled_pairs
clem run -g function_detective -m <model_name> -i instance_active_pair_checks -r results_active_pair_checks
clem run -g function_detective -m <model_name> -i instance_passive_examples_oneshot -r results_passive_examples_oneshot
clem run -g function_detective -m <model_name> -i instance_passive_labeled_pairs_oneshot -r results_passive_labeled_pairs_oneshot
```

Score the results:

```bash
clem score function_detective
```

### Results Layout

Results are now typically generated per mode in separate folders, for example:

- `results_passive_examples`
- `results_active_inputs`
- `results_passive_labeled_pairs`
- `results_active_pair_checks`
- `results_passive_examples_oneshot`
- `results_passive_labeled_pairs_oneshot`

Within a results folder, the stored run layout is organized by game, then model, then mode:

```text
results_<mode>/
  function_detective/
    <model_name>/
      <mode_name>/
        <experiment_name>/
          instance_00000/
```

### Evaluation Views

Running:

```bash
clem eval -r <results_folder>
```

now produces `results.csv` and `results.html` with:

- overall game performance
- mode-level performance
- domain-level performance for:
  - `scalar_math`
  - `pair_math`
  - `string`

This makes it easier to see which model performs well in which domain inside each mode-specific run folder.
