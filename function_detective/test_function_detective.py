import os
import sys
import unittest

try:
    from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS
except ModuleNotFoundError:
    BENCH_SCORE = "BenchScore"
    METRIC_ABORTED = "Aborted"
    METRIC_SUCCESS = "Success"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function_detective.protocol import NEXT_TAG
from function_detective.instancegenerator import (
    FunctionDetectiveInstanceGenerator,
    INTERACTIVE_PASSIVE_EXAMPLES,
    LOGIC_PASSIVE_EXAMPLES_MIN,
    MAX_TURNS,
    MODE_INSTANCE_FILES,
    NUM_TESTS,
)
from function_detective.utils import (
    create_passive_io_examples,
    create_passive_membership_examples,
    create_static_test_cases,
    extract_function_code,
    format_io_example,
    format_membership_example,
    is_sandbox_failure,
    parse_active_io_probe,
    parse_membership_probe,
    parse_signature_with_types,
    render_examples_block,
    type_matches_annotation,
)

try:
    from function_detective.master import (
        FunctionDetective,
        FunctionDetectiveGameState,
        FunctionDetectiveScorer,
        FunctionGuesser,
    )
    MASTER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    FunctionDetective = None
    FunctionDetectiveGameState = None
    FunctionDetectiveScorer = None
    FunctionGuesser = None
    MASTER_IMPORT_ERROR = exc


class FunctionDetectiveTestCase(unittest.TestCase):
    def setUp(self):
        self.static_tests = [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [-1], "expected": 1},
            {"args": [10], "expected": 10},
        ]

    def test_public_game_classes_exist(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")
        self.assertTrue(FunctionDetective)
        self.assertTrue(FunctionGuesser)
        self.assertTrue(FunctionDetectiveGameState)

    def test_extract_function_code(self):
        response = """SOLVE: ```python
def solution(x):
    return abs(x)
```"""
        self.assertEqual(extract_function_code(response), "def solution(x):\n    return abs(x)")

    def test_parse_signature_with_types(self):
        names, types, return_type = parse_signature_with_types("(x: int, y: int) -> int")
        self.assertEqual(names, ["x", "y"])
        self.assertEqual(types, ["int", "int"])
        self.assertEqual(return_type, "int")

    def test_membership_probe_parsing(self):
        args, candidate = parse_membership_probe('1, 2, 3', num_params=2)
        self.assertEqual(args, [1, 2])
        self.assertEqual(candidate, 3)

    def test_active_io_probe_parsing_single_value(self):
        values = parse_active_io_probe("7", num_params=1)
        self.assertEqual(values, (7,))

    def test_active_io_probe_rejects_non_iterable_multi_input(self):
        with self.assertRaises(ValueError):
            parse_active_io_probe("7", num_params=2)

    def test_type_matches_annotation(self):
        self.assertTrue(type_matches_annotation([1, 2], "List[int]"))
        self.assertTrue(type_matches_annotation(1, "float"))
        self.assertFalse(type_matches_annotation(True, "int"))

    def test_passive_io_examples_are_deterministic_subset(self):
        examples = create_passive_io_examples(self.static_tests, num_examples=3)
        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0]["kind"], "io")
        self.assertEqual(examples[0]["args"], [0])

    def test_passive_io_examples_repeat_when_unique_pool_is_small(self):
        examples = create_passive_io_examples(self.static_tests[:2], num_examples=6)
        self.assertEqual(len(examples), 6)
        self.assertEqual(examples[0]["args"], examples[2]["args"])
        self.assertEqual(examples[1]["args"], examples[3]["args"])

    def test_passive_membership_examples_include_true_and_false(self):
        examples = create_passive_membership_examples(
            static_tests=self.static_tests,
            return_type="int",
            category="SCALAR_MATH",
            num_examples=4,
        )
        labels = {example["is_member"] for example in examples}
        self.assertEqual(labels, {True, False})

    def test_example_formatters(self):
        self.assertEqual(format_io_example([1, 2], 3), "((1, 2), 3)")
        self.assertEqual(format_membership_example([1], 3, False), "((1, 3), False)")

    def test_render_examples_block(self):
        rendered = render_examples_block(create_passive_io_examples(self.static_tests, num_examples=2))
        self.assertIn("1. ", rendered)
        self.assertIn("(0, 0)", rendered)

    def test_quality_score_uses_binary_accuracy_and_interactive_efficiency(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")

        scorer = FunctionDetectiveScorer("function_detective", {}, {})
        scorer.compute_episode_scores(
            {
                METRIC_SUCCESS: True,
                METRIC_ABORTED: False,
                "mode": "active_inputs",
                "binary_accuracy": 1.0,
                "efficiency": 0.75,
                "efficiency_raw": 0.75,
                "tolerance_used": 1,
                "n_test_cases": 100,
            }
        )

        episode_scores = scorer.scores["episode scores"]
        self.assertEqual(episode_scores[BENCH_SCORE], 75.0)
        self.assertEqual(episode_scores["quality_score"], 75.0)
        self.assertEqual(episode_scores["efficiency"], 75.0)
        self.assertEqual(episode_scores["binary_accuracy"], 1.0)
        self.assertEqual(episode_scores["efficiency_raw"], 0.75)
        self.assertEqual(episode_scores["tolerance_used"], 1)
        self.assertEqual(episode_scores["n_test_cases"], 100)

    def test_oneshot_score_ignores_efficiency_multiplier(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")

        scorer = FunctionDetectiveScorer("function_detective", {}, {})
        scorer.compute_episode_scores(
            {
                METRIC_SUCCESS: True,
                METRIC_ABORTED: False,
                "mode": "passive_examples_oneshot",
                "binary_accuracy": 1.0,
                "efficiency": 0.25,
                "efficiency_raw": 1.0,
                "tolerance_used": 0,
            }
        )

        episode_scores = scorer.scores["episode scores"]
        self.assertEqual(episode_scores[BENCH_SCORE], 100.0)
        self.assertEqual(episode_scores["quality_score"], 100.0)

    def test_detects_sandbox_failure_feedback(self):
        self.assertTrue(
            is_sandbox_failure(
                "Sandbox Failure.\nStderr: 'failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine'"
            )
        )
        self.assertTrue(is_sandbox_failure("Docker is not running or not reachable. Start Docker Desktop and try again."))
        self.assertFalse(is_sandbox_failure("Failed Cases:\nInput: [1, 2] | Expected: 3 | Got: 4"))

    def test_protocol_exposes_next_tag(self):
        self.assertEqual(NEXT_TAG, "NEXT:")

    def test_generator_writes_mode_specific_instance_files(self):
        generator = FunctionDetectiveInstanceGenerator()
        generator.generate(seed=0)

        instances_dir = os.path.join(PROJECT_ROOT, "function_detective", "in")
        for instance_file in MODE_INSTANCE_FILES.values():
            path = os.path.join(instances_dir, f"{instance_file}.json")
            self.assertTrue(os.path.exists(path), path)

    def test_interactive_passive_example_budget_is_generous(self):
        self.assertEqual(INTERACTIVE_PASSIVE_EXAMPLES, NUM_TESTS)
        self.assertGreater(INTERACTIVE_PASSIVE_EXAMPLES, MAX_TURNS)

    def test_logic_category_uses_minimum_passive_example_budget(self):
        self.assertEqual(
            FunctionDetectiveInstanceGenerator._passive_example_count("LOGIC", None),
            INTERACTIVE_PASSIVE_EXAMPLES,
        )
        self.assertEqual(
            FunctionDetectiveInstanceGenerator._passive_example_count("LOGIC", "oneshot"),
            LOGIC_PASSIVE_EXAMPLES_MIN,
        )

    def test_string_static_test_generation_is_large_enough_for_passive_budget(self):
        def append_exclamation(value: str) -> str:
            return value + "!"

        test_cases = create_static_test_cases(
            append_exclamation,
            category="STRING",
            num_tests=NUM_TESTS,
            signature="(value: str) -> str",
            difficulty="easy",
        )
        self.assertGreaterEqual(len(test_cases), MAX_TURNS + 1)

    def test_oneshot_episode_summary_omits_interaction_only_fields(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")

        game = FunctionDetective.__new__(FunctionDetective)
        game._state = None
        game.mode = "passive_examples_oneshot"
        game.state = FunctionDetectiveGameState(
            max_turns=1,
            function_signature="(x: int) -> int",
            function_callable="add_one",
            mode="passive_examples_oneshot",
        )
        game.state.test_accuracy = 0.0
        game.state.test_efficiency = 1.0

        captured = {}
        game.log_to_self = lambda event_type, content: captured.update({"event_type": event_type, "content": content})
        game._log_episode_summary()

        summary = captured["content"]
        self.assertIn("Accuracy: 0.000", summary)
        self.assertIn("Efficiency: 1.000", summary)
        self.assertNotIn("Information requests", summary)
        self.assertNotIn("Revealed passive examples", summary)
        self.assertNotIn("Slack:", summary)
        self.assertNotIn("Observed-pair consistency", summary)
        self.assertNotIn("Membership balance", summary)

    def test_efficiency_formula_uses_interactive_tolerance(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")

        game = FunctionDetective.__new__(FunctionDetective)
        game._state = None
        game.mode = "active_inputs"
        game.state = FunctionDetectiveGameState(
            max_turns=15,
            function_signature="(x: int) -> int",
            function_callable="add_one",
            mode="active_inputs",
        )
        game.state.info_request_count = 4
        game._set_efficiency_metrics()

        self.assertAlmostEqual(game.state.test_efficiency, 0.8)
        self.assertAlmostEqual(game.state.efficiency_raw, 0.8)
        self.assertEqual(game.state.tolerance_used, 1)


if __name__ == "__main__":
    unittest.main()
