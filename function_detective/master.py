import importlib.util
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import DialogueGameMaster, GameBenchmark, GameError, GameMaster, GameScorer, GameSpec, ParseError, Player
from clemcore.clemgame.master import GameState
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS

try:
    from protocol import (
        ACTIVE_IO_MODE,
        ACTIVE_MEMBERSHIP_MODE,
        MEMBERSHIP_MODES,
        NEXT_TAG,
        ONESHOT_MODES,
        OUTPUT_TAG,
        PASSIVE_IO_MODE,
        PASSIVE_IO_ONESHOT_MODE,
        PASSIVE_MEMBERSHIP_MODE,
        PASSIVE_MEMBERSHIP_ONESHOT_MODE,
        PASSIVE_MODES,
        SOLVE_TAG,
        TEST_TAG,
    )
    from utils import (
        extract_function_code,
        format_input_object,
        format_io_example,
        format_membership_example,
        format_value,
        get_sandbox_status,
        is_sandbox_failure,
        parse_active_io_probe,
        parse_membership_probe,
        parse_signature_with_types,
        render_examples_block,
        type_matches_annotation,
        validate_function_logic,
    )
except ModuleNotFoundError:
    from function_detective.protocol import (
        ACTIVE_IO_MODE,
        ACTIVE_MEMBERSHIP_MODE,
        MEMBERSHIP_MODES,
        NEXT_TAG,
        ONESHOT_MODES,
        OUTPUT_TAG,
        PASSIVE_IO_MODE,
        PASSIVE_IO_ONESHOT_MODE,
        PASSIVE_MEMBERSHIP_MODE,
        PASSIVE_MEMBERSHIP_ONESHOT_MODE,
        PASSIVE_MODES,
        SOLVE_TAG,
        TEST_TAG,
    )
    from function_detective.utils import (
        extract_function_code,
        format_input_object,
        format_io_example,
        format_membership_example,
        format_value,
        get_sandbox_status,
        is_sandbox_failure,
        parse_active_io_probe,
        parse_membership_probe,
        parse_signature_with_types,
        render_examples_block,
        type_matches_annotation,
        validate_function_logic,
    )


logger = logging.getLogger(__name__)


@dataclass
class FunctionDetectiveGameState(GameState):
    max_turns: int
    function_signature: str
    function_callable: str
    mode: str
    baseline_kind: str = ""

    success: bool = False
    failure: bool = False
    aborted: bool = False

    probe_count: int = 0
    unique_output_count: int = 0
    output_novelty_rate: float = 0.0
    unique_input_count: int = 0
    unique_input_rate: float = 0.0

    probe_args: List[Any] = field(default_factory=list)
    probe_round: int = 0
    probe_output: Any = ""

    test_slack: float = 0.0
    test_accuracy: float = 0.0
    test_efficiency: float = 0.0
    efficiency_raw: float = 0.0
    tolerance_used: int = 0
    n_test_cases: int = 0

    parse_error_count: int = 0
    runtime_error_count: int = 0

    info_request_count: int = 0
    revealed_example_count: int = 0
    passive_index: int = 0
    passive_examples_total: int = 0
    membership_true_count: int = 0
    membership_false_count: int = 0

    observed_pairs: List[Dict] = field(default_factory=list)
    revealed_examples: List[Dict] = field(default_factory=list)

    internal_consistency_score: float = 0.0
    internal_consistency_all_observed: bool = False
    internal_consistency_violations: int = 0

    def __post_init__(self):
        super().__init__()


class FunctionGuesser(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages):
        sanitized = []
        for message in messages:
            sanitized.append(str(message.content) if hasattr(message, "content") else str(message))
        return " ".join(sanitized)


class FunctionDetective(DialogueGameMaster):
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.experiment = experiment

    def _build_mode_prompt_config(self) -> Dict[str, str]:
        if self.mode == ACTIVE_IO_MODE:
            return {
                "mode_name": "Active Inputs",
                "mode_description": (
                    "You choose test inputs x, and I return the exact output f(x)."
                ),
                "action_choices": f"- {TEST_TAG} <inputs>\n- {SOLVE_TAG} ```python ... ```",
                "probe_guide": (
                    f"To request function values, write exactly one line: `{TEST_TAG} <inputs>`.\n"
                    "Provide one value per function argument, in signature order."
                ),
                "example_test_line": f"{TEST_TAG} {self._build_active_io_example_input()}",
                "example_output_line": f"{OUTPUT_TAG} {self._build_active_io_example_output()}",
                "preloaded_examples": "No examples are preloaded in this mode.",
            }
        if self.mode == ACTIVE_MEMBERSHIP_MODE:
            return {
                "mode_name": "Active Pair Checks",
                "mode_description": (
                    "You propose function inputs together with a candidate output, and I tell you whether that pair is correct."
                ),
                "action_choices": f"- {TEST_TAG} <inputs>, <candidate_output>\n- {SOLVE_TAG} ```python ... ```",
                "probe_guide": (
                    f"To query graph membership, write exactly one line: `{TEST_TAG} <input1>, ..., <candidate_output>`.\n"
                    "The last value is the candidate function output; the reply is True iff it matches the hidden function."
                ),
                "example_test_line": f"{TEST_TAG} {self._build_active_membership_example_input()}",
                "example_output_line": f"{OUTPUT_TAG} True",
                "preloaded_examples": "No examples are preloaded in this mode.",
            }
        if self.mode == PASSIVE_IO_MODE:
            return {
                "mode_name": "Passive Examples",
                "mode_description": (
                    "I control which valid input-output examples you see. You can request them one at a time."
                ),
                "action_choices": f"- {NEXT_TAG}\n- {SOLVE_TAG} ```python ... ```",
                "probe_guide": (
                    f"To receive the next valid example, write exactly `{NEXT_TAG}`.\n"
                    "Each reply is a valid pair (x, f(x))."
                ),
                "example_test_line": NEXT_TAG,
                "example_output_line": f"{OUTPUT_TAG} {self._build_passive_example_output()}",
                "preloaded_examples": "No examples are preloaded in this mode.",
            }
        if self.mode == PASSIVE_MEMBERSHIP_MODE:
            return {
                "mode_name": "Passive Labeled Pairs",
                "mode_description": (
                    "I reveal candidate input-output pairs one at a time and tell you whether each one is correct."
                ),
                "action_choices": f"- {NEXT_TAG}\n- {SOLVE_TAG} ```python ... ```",
                "probe_guide": (
                    f"To receive the next graph-membership statement, write exactly `{NEXT_TAG}`.\n"
                    "Each reply has the form ((x, y), True/False)."
                ),
                "example_test_line": NEXT_TAG,
                "example_output_line": f"{OUTPUT_TAG} {self._build_passive_example_output()}",
                "preloaded_examples": "No examples are preloaded in this mode.",
            }
        if self.mode == PASSIVE_IO_ONESHOT_MODE:
            return {
                "mode_name": "Passive Examples One-Shot",
                "mode_description": (
                    "I have already given you a fixed list of valid input-output examples. Solve using those examples only."
                ),
                "action_choices": f"- {SOLVE_TAG} ```python ... ```",
                "probe_guide": "No further example requests are allowed in this one-shot baseline.",
                "example_test_line": SOLVE_TAG + " ```python",
                "example_output_line": "def solution(...):\n    ...\n```",
                "preloaded_examples": render_examples_block(self.passive_examples),
            }
        return {
            "mode_name": "Passive Labeled Pairs One-Shot",
            "mode_description": (
                "I have already given you a fixed list of candidate input-output pairs with true or false labels. Solve using those statements only."
            ),
            "action_choices": f"- {SOLVE_TAG} ```python ... ```",
            "probe_guide": "No further example requests are allowed in this one-shot baseline.",
            "example_test_line": SOLVE_TAG + " ```python",
            "example_output_line": "def solution(...):\n    ...\n```",
            "preloaded_examples": render_examples_block(self.passive_examples),
        }

    def _build_initial_prompt(self) -> str:
        prompt = self.experiment["guesser_initial_prompt"]
        config = self._build_mode_prompt_config()
        param_list_str = ", ".join(
            f"{name}: {param_type}" for name, param_type in zip(self.param_names, self.param_types)
        ) if self.param_names else "none"
        variables = ", ".join(f"<value{i + 1}>" for i in range(self.num_params))

        replacements = {
            "$TEST_TAG$": TEST_TAG,
            "$SOLVE_TAG$": SOLVE_TAG,
            "$OUTPUT_TAG$": OUTPUT_TAG,
            "$NEXT_TAG$": NEXT_TAG,
            "$SIGNATURE_OF_CURRENT_FUNCTION$": self.signature,
            "$NUM_PARAMS$": str(self.num_params),
            "$PARAM_LIST$": param_list_str,
            "$CATEGORY$": str(self.category),
            "$MAX_TURNS$": str(self.max_turns),
            "$VARIABLES$": variables,
            "$MODE_NAME$": config["mode_name"],
            "$MODE_DESCRIPTION$": config["mode_description"],
            "$ACTION_CHOICES$": config["action_choices"],
            "$PROBE_FORMAT_GUIDE$": config["probe_guide"],
            "$EXAMPLE_TEST_LINE$": config["example_test_line"],
            "$EXAMPLE_OUTPUT_LINE$": config["example_output_line"],
            "$PRELOADED_EXAMPLES_SECTION$": (
                "Preloaded examples:\n" + config["preloaded_examples"] if self.mode in ONESHOT_MODES else ""
            ),
        }
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def _build_active_io_example_input(self) -> str:
        if not self.param_names:
            return "<inputs>"
        return ", ".join(self._placeholder_input_values())

    def _build_active_io_example_output(self) -> str:
        return self._placeholder_output_value()

    def _build_active_membership_example_input(self) -> str:
        values = self._placeholder_input_values()
        values.append("<candidate_output>")
        return ", ".join(values)

    def _build_passive_example_output(self) -> str:
        if self.passive_examples:
            return self._render_passive_example(self.passive_examples[0])
        input_object = self._placeholder_input_object()
        if self.mode in MEMBERSHIP_MODES:
            return f"(({input_object}, <candidate_output>), True)"
        return f"({input_object}, {self._placeholder_output_value()})"

    def _placeholder_input_values(self) -> List[str]:
        values = []
        for index, param_type in enumerate(self.param_types or [], start=1):
            values.append(self._placeholder_value_for_type(param_type, index=index))
        return values or ["<input_value>"]

    def _placeholder_input_object(self) -> str:
        values = self._placeholder_input_values()
        if len(values) == 1:
            return values[0]
        return "(" + ", ".join(values) + ")"

    def _placeholder_output_value(self) -> str:
        return self._placeholder_value_for_type(self.return_type, index=1, is_output=True)

    def _placeholder_value_for_type(self, annotation: str, index: int, is_output: bool = False) -> str:
        annotation = (annotation or "").lower()
        role = "output" if is_output else f"input{index}"
        if "str" in annotation:
            return f'"<{role}_string>"'
        if "bool" in annotation:
            return f"<{role}_bool>"
        if "float" in annotation:
            return f"<{role}_number>"
        if "int" in annotation:
            return f"<{role}_number>"
        if "list" in annotation or "tuple" in annotation:
            return f"<{role}_sequence>"
        return f"<{role}_value>"

    def _record_information_request(self) -> None:
        self.state.info_request_count += 1
        self.state.probe_round += 1
        self.state.probe_count = self.state.info_request_count

    def _record_positive_observation(self, args: List[Any], result: Any) -> None:
        self.state.probe_args = list(args)
        self.state.probe_output = result
        self.state.observed_pairs.append(
            {
                "round": self.state.probe_round,
                "args": list(args),
                "output": result,
            }
        )

    def _record_membership_balance(self, is_member: bool) -> None:
        if is_member:
            self.state.membership_true_count += 1
        else:
            self.state.membership_false_count += 1

    def _set_efficiency_metrics(self) -> None:
        if self.mode in ONESHOT_MODES:
            raw_efficiency = 1.0
            self.state.tolerance_used = 0
            self.state.efficiency_raw = raw_efficiency
            self.state.test_efficiency = raw_efficiency
            self.state.test_slack = 0.0
            return
        tolerance = 1
        raw_efficiency = (
            (self.state.max_turns - self.state.info_request_count + tolerance) / self.state.max_turns
            if self.state.max_turns
            else 0.0
        )
        self.state.tolerance_used = tolerance
        self.state.efficiency_raw = raw_efficiency
        self.state.test_efficiency = min(1.0, max(0.0, raw_efficiency))
        self.state.test_slack = max(0, self.state.max_turns - self.state.info_request_count)

    def _build_observed_cases(self) -> List[Dict[str, Any]]:
        return [{"args": pair["args"], "expected": pair["output"]} for pair in self.state.observed_pairs]

    def _compute_internal_consistency(self, guessed_code: str) -> None:
        observed_cases = self._build_observed_cases()
        if not observed_cases:
            self.state.internal_consistency_score = 1.0
            self.state.internal_consistency_all_observed = True
            self.state.internal_consistency_violations = 0
            return
        obs_all_correct, obs_accuracy, feedback = validate_function_logic(guessed_code, observed_cases)
        if is_sandbox_failure(feedback):
            raise RuntimeError(feedback)
        self.state.internal_consistency_score = float(obs_accuracy)
        self.state.internal_consistency_all_observed = bool(obs_all_correct)
        self.state.internal_consistency_violations = int(round((1.0 - obs_accuracy) * len(observed_cases)))

    def _stable_key(self, obj: Any) -> str:
        try:
            return json.dumps(obj, default=str, sort_keys=True)
        except Exception:
            return str(obj)

    def _compute_probe_summary(self) -> Dict[str, float]:
        pairs = self.state.observed_pairs or []
        probe_count = len(pairs)
        unique_input_keys = {self._stable_key(pair.get("args")) for pair in pairs}
        unique_output_keys = {self._stable_key(pair.get("output")) for pair in pairs}
        unique_input_count = len(unique_input_keys)
        unique_output_count = len(unique_output_keys)
        novelty_rate = (unique_output_count / probe_count) if probe_count else 0.0
        input_diversity = (unique_input_count / probe_count) if probe_count else 0.0
        output_entropy_bits = 0.0
        if probe_count:
            from collections import Counter
            import math

            counts = Counter(self._stable_key(pair.get("output")) for pair in pairs)
            for count in counts.values():
                probability = count / probe_count
                output_entropy_bits -= probability * math.log2(probability)
        return {
            "probe_count": probe_count,
            "unique_input_count": unique_input_count,
            "unique_output_count": unique_output_count,
            "novelty_rate": float(novelty_rate),
            "input_diversity": float(input_diversity),
            "output_entropy_bits": float(output_entropy_bits),
        }

    def _log_probe_summary(self) -> None:
        summary = self._compute_probe_summary()
        for key, value in summary.items():
            self.log_key(key, value)
        self.log_key("mode", self.mode)
        self.log_key("info_request_count", self.state.info_request_count)
        self.log_key("revealed_example_count", self.state.revealed_example_count)
        self.log_key("membership_true_count", self.state.membership_true_count)
        self.log_key("membership_false_count", self.state.membership_false_count)
        self.log_key("internal_consistency_score", self.state.internal_consistency_score)
        self.log_key("internal_consistency_all_observed", self.state.internal_consistency_all_observed)
        self.log_key("internal_consistency_violations", self.state.internal_consistency_violations)

    def _log_probe_result(self) -> None:
        self.log_to_self(
            "probe_result",
            (
                f"Round: {self.state.probe_round}\n"
                f"Inputs: {format_input_object(self.state.probe_args)}\n"
                f"Output: {format_value(self.state.probe_output)}"
            ),
        )

    def _log_episode_summary(self) -> None:
        lines = [f"Mode: {self.mode}", f"Accuracy: {self.state.test_accuracy:.3f}"]

        if self.mode not in ONESHOT_MODES:
            lines.extend(
                [
                    f"Information requests: {self.state.info_request_count}/{self.state.max_turns}",
                    f"Revealed passive examples: {self.state.revealed_example_count}",
                    f"Efficiency: {self.state.test_efficiency:.3f}",
                    f"Slack: {int(self.state.test_slack)}",
                    f"Observed-pair consistency: {self.state.internal_consistency_score:.3f}",
                ]
            )
            if self.mode in MEMBERSHIP_MODES:
                lines.append(
                    f"Membership balance: {self.state.membership_true_count} true / {self.state.membership_false_count} false"
                )
        else:
            lines.append(f"Efficiency: {self.state.test_efficiency:.3f}")

        lines.extend(
            [
                f"Parse errors: {self.state.parse_error_count}",
                f"Runtime errors: {self.state.runtime_error_count}",
            ]
        )

        self.log_to_self(
            "episode_summary",
            "\n".join(lines),
        )

    def _format_parse_error_instructions(self) -> str:
        if self.mode in ONESHOT_MODES:
            return f"Output ONLY:\n{SOLVE_TAG} ```python ... ```"
        if self.mode in PASSIVE_MODES:
            return f"Output ONLY:\n{NEXT_TAG}\nOR\n{SOLVE_TAG} ```python ... ```"
        return f"Output ONLY:\n{TEST_TAG} ...\nOR\n{SOLVE_TAG} ```python ... ```"

    def _clean_source_for_reveal(self, func) -> str:
        try:
            source_lines = inspect.getsource(func).splitlines()
        except OSError:
            return "# Source code unavailable"

        def_idx = None
        for index, line in enumerate(source_lines):
            if line.lstrip().startswith("def "):
                def_idx = index
                break
        if def_idx is None:
            return "# Source code unavailable"

        lines = source_lines[def_idx:]
        j = 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines):
            stripped = lines[j].lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and stripped.strip() != quote:
                    lines.pop(j)
                else:
                    lines.pop(j)
                    while j < len(lines):
                        if quote in lines[j]:
                            lines.pop(j)
                            break
                        lines.pop(j)
        return "\n".join(lines).strip()

    def load_game_function(self, function_name: str):
        module_path = os.path.join(os.path.dirname(__file__), "functions.py")
        spec = importlib.util.spec_from_file_location("_function_detective_functions", module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module from '{module_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            return getattr(module, function_name)
        except AttributeError as exc:
            raise ValueError(f"Function '{function_name}' not found in '{module_path}'") from exc

    def _inputs_match_signature(self, parsed_values: Tuple[Any, ...]) -> bool:
        if len(parsed_values) != self.num_params:
            return False
        return all(type_matches_annotation(value, expected) for value, expected in zip(parsed_values, self.param_types))

    def _membership_probe_matches_signature(self, args: List[Any], candidate_output: Any) -> bool:
        if len(args) != self.num_params:
            return False
        if not all(type_matches_annotation(value, expected) for value, expected in zip(args, self.param_types)):
            return False
        return type_matches_annotation(candidate_output, self.return_type)

    def _render_passive_example(self, example: Dict[str, Any]) -> str:
        if example["kind"] == "io":
            return format_io_example(example["args"], example["expected"])
        return format_membership_example(example["args"], example["candidate_output"], example["is_member"])

    def _reveal_next_passive_example(self) -> None:
        self._record_information_request()
        if self.state.passive_index >= len(self.passive_examples):
            self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} NO_MORE_EXAMPLES")
            return

        example = self.passive_examples[self.state.passive_index]
        self.state.passive_index += 1
        self.state.revealed_example_count += 1
        self.state.revealed_examples.append(example)
        self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {self._render_passive_example(example)}")

        if example["kind"] == "io":
            self._record_positive_observation(example["args"], example["expected"])
        else:
            self._record_membership_balance(example["is_member"])
            if example["is_member"]:
                self._record_positive_observation(example["args"], example["candidate_output"])

    def _seed_prompt_passive_example(self) -> None:
        if self.mode not in PASSIVE_MODES or self.mode in ONESHOT_MODES or not self.passive_examples:
            return

        example = self.passive_examples[0]
        self.state.passive_index = 1
        self.state.revealed_example_count = 1
        self.state.revealed_examples.append(example)

        if example["kind"] == "io":
            self._record_positive_observation(example["args"], example["expected"])
        else:
            self._record_membership_balance(example["is_member"])
            if example["is_member"]:
                self._record_positive_observation(example["args"], example["candidate_output"])

    def _handle_active_io_action(self, input_line: str) -> None:
        self._record_information_request()
        func = self.load_game_function(self.state.function_callable)
        args = list(parse_active_io_probe(input_line[len(TEST_TAG):].strip(), self.num_params))
        try:
            result = func(*args)
        except Exception as exc:
            self.state.runtime_error_count += 1
            self.set_context_for(self.guesser_player, f"Error executing function: {exc}")
            return
        self._record_positive_observation(args, result)
        self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {format_value(result)}")
        self._log_probe_result()

    def _handle_active_membership_action(self, input_line: str) -> None:
        self._record_information_request()
        func = self.load_game_function(self.state.function_callable)
        args, candidate_output = parse_membership_probe(input_line[len(TEST_TAG):].strip(), self.num_params)
        try:
            actual_output = func(*args)
        except Exception as exc:
            self.state.runtime_error_count += 1
            self.set_context_for(self.guesser_player, f"Error executing function: {exc}")
            return
        is_member = actual_output == candidate_output
        self._record_membership_balance(is_member)
        if is_member:
            self._record_positive_observation(args, candidate_output)
        self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {is_member}")

    def _handle_solve_action(self, guessed_code: str) -> None:
        actual_func = self.load_game_function(self.state.function_callable)
        actual_code = self._clean_source_for_reveal(actual_func)
        is_correct, _accuracy, feedback = validate_function_logic(guessed_code, self.test_cases)

        if is_sandbox_failure(feedback):
            self.state.aborted = True
            self.state.failure = False
            self.state.success = False
            abort_message = (
                "Infrastructure Failure: Docker sandbox is unavailable.\n"
                f"{feedback}\n\n"
                "Start Docker Desktop and rerun the benchmark."
            )
            logger.error(abort_message)
            print(abort_message)
            self.log_to_self("outcome", "Game Verdict: ABORTED (sandbox unavailable)")
            self.log_to_self("infrastructure_error", abort_message)
            return

        self._compute_internal_consistency(guessed_code)
        self.state.test_accuracy = 1.0 if is_correct else 0.0
        self._set_efficiency_metrics()

        if is_correct:
            self.state.success = True
            self.log_to_self("outcome", "Game Verdict: WIN")
            feedback_msg = "That is correct!"
        else:
            self.state.success = False
            self.state.failure = True
            self.log_to_self("outcome", "Game Verdict: LOSS")
            feedback_msg = f"Incorrect.\n{feedback}"

        self.log_to_self("solution_feedback", feedback_msg)
        self.log_to_self("hidden_function", actual_code)
        self.log_to_self("submitted_solution", guessed_code)
        self._log_episode_summary()

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.signature = game_instance["signature"]
        self.mode = game_instance.get("mode", self.experiment.get("mode", ACTIVE_IO_MODE))
        self.baseline_kind = game_instance.get("baseline_kind", self.experiment.get("baseline_kind") or "")
        self.max_turns = self.experiment["max_turns"]
        self.test_cases = game_instance.get("test_cases", [])
        self.passive_examples = game_instance.get("passive_examples", [])
        self.param_names, self.param_types, self.return_type = parse_signature_with_types(self.signature)
        self.num_params = len(self.param_names)
        self.category = game_instance.get("category", "MATH")
        prompt = self._build_initial_prompt()
        self.guesser_player = FunctionGuesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=prompt)

        self.state = FunctionDetectiveGameState(
            max_turns=self.max_turns,
            function_signature=self.signature,
            function_callable=game_instance["callable"],
            mode=self.mode,
            baseline_kind=self.baseline_kind,
        )
        self.state.n_test_cases = len(self.test_cases)
        self.state.passive_examples_total = len(self.passive_examples)
        self._seed_prompt_passive_example()

        sandbox_ok, sandbox_message = get_sandbox_status()
        if not sandbox_ok:
            user_message = (
                "Docker sandbox is not available. FunctionDetective cannot validate solutions without it.\n"
                f"{sandbox_message}\n"
                "Start Docker Desktop and rerun the benchmark."
            )
            logger.error(user_message)
            print(user_message)
            self.state.aborted = True
            self.log_to_self("outcome", "Game Verdict: ABORTED (sandbox unavailable)")
            self.log_to_self("infrastructure_error", user_message)

    def _parse_response(self, player: Player, response: str) -> Tuple[str, Any]:
        response_str = str(response).strip()
        if response_str.startswith(SOLVE_TAG):
            lines = response_str.splitlines()
            forbidden_tags = [TEST_TAG, NEXT_TAG] if self.mode not in ONESHOT_MODES else [TEST_TAG, NEXT_TAG]
            if any(any(line.strip().startswith(tag) for tag in forbidden_tags) for line in lines[1:]):
                raise ParseError("Only one action per turn.")
            extracted_code = extract_function_code(response_str)
            if not extracted_code:
                raise ParseError(f"{SOLVE_TAG} must contain a markdown code block (```python ... ```).")
            return "solve", extracted_code

        if self.mode in ONESHOT_MODES:
            raise ParseError(f"Only {SOLVE_TAG} is allowed in this one-shot baseline.")

        if self.mode in {PASSIVE_IO_MODE, PASSIVE_MEMBERSHIP_MODE}:
            if response_str == NEXT_TAG:
                return "next", None
            raise ParseError(f"Output must be exactly {NEXT_TAG} or {SOLVE_TAG} ```python ... ```.")

        if response_str.startswith(TEST_TAG):
            input_line = response_str.splitlines()[0].strip()
            for line in response_str.splitlines()[1:]:
                stripped = line.strip()
                if stripped.startswith(TEST_TAG):
                    raise ParseError(f"Only one {TEST_TAG} is allowed per turn.")
                if stripped.startswith(SOLVE_TAG) or stripped.startswith(NEXT_TAG):
                    raise ParseError("Only one action per turn.")

            try:
                if self.mode == ACTIVE_IO_MODE:
                    values = parse_active_io_probe(input_line[len(TEST_TAG):].strip(), self.num_params)
                    if not self._inputs_match_signature(values):
                        raise ParseError(f"{TEST_TAG} values do not match the required signature/types.")
                else:
                    args, candidate_output = parse_membership_probe(input_line[len(TEST_TAG):].strip(), self.num_params)
                    if not self._membership_probe_matches_signature(args, candidate_output):
                        raise ParseError(f"{TEST_TAG} values do not match the required signature/types.")
            except ParseError:
                raise
            except Exception as exc:
                raise ParseError(str(exc))
            return "test", {"input_line": input_line}

        raise ParseError("Invalid format for the current mode.")

    def _advance_game(self, player: Player, parsed_response: Tuple[str, Any]):
        action_type, content = parsed_response
        if action_type == "next":
            self._reveal_next_passive_example()
        elif action_type == "test":
            if self.mode == ACTIVE_IO_MODE:
                self._handle_active_io_action(content["input_line"])
            else:
                self._handle_active_membership_action(content["input_line"])
        elif action_type == "solve":
            self._handle_solve_action(content)

    def _on_parse_error(self, error: GameError):
        self.set_context_for(
            self.guesser_player,
            f"Game Violation: {error}\n{self._format_parse_error_instructions()}",
        )
        self.state.parse_error_count += 1
        print(f"Parse Error: {error}. Consuming a turn.")

    def _does_game_proceed(self):
        if self.current_round >= self.max_turns:
            return False
        return not (self.state.aborted or self.state.failure or self.state.success)

    def compute_turn_score(self):
        return 1 if self.state.success else 0

    def compute_episode_score(self):
        return 100 if self.state.success else 0

    def _on_after_game(self):
        if self.current_round >= self.max_turns and not (self.state.success or self.state.failure or self.state.aborted):
            self.state.failure = True
            self.log_to_self("outcome", "Game Verdict: LOSS (turn limit reached)")

        self.log_key(METRIC_ABORTED, self.state.aborted)
        self.log_key(METRIC_LOSE, self.state.failure)
        self.log_key(METRIC_SUCCESS, self.state.success)
        self.log_key("efficiency", getattr(self.state, "test_efficiency", 0.0))
        self.log_key("efficiency_raw", getattr(self.state, "efficiency_raw", 0.0))
        self.log_key("slack", getattr(self.state, "test_slack", 0.0))
        self.log_key("accuracy", getattr(self.state, "test_accuracy", 0.0))
        self.log_key("binary_accuracy", getattr(self.state, "test_accuracy", 0.0))
        self.log_key("tolerance_used", getattr(self.state, "tolerance_used", 0))
        self.log_key("n_test_cases", getattr(self.state, "n_test_cases", len(self.test_cases)))
        self.log_key("parse_error_count", self.state.parse_error_count)
        self.log_key("runtime_error_count", self.state.runtime_error_count)
        self.log_key("turns_used", self.state.info_request_count)
        self.log_key("max_turns", self.state.max_turns)
        self.log_key("observed_pairs", self.state.observed_pairs)
        self.log_key("revealed_examples", self.state.revealed_examples)
        self._log_probe_summary()


class FunctionDetectiveScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, "response_received", 1)

    def compute_episode_scores(self, interactions: Dict):
        binary_success = 1 if interactions.get(METRIC_SUCCESS, False) else 0
        binary_accuracy = float(interactions.get("binary_accuracy", binary_success))
        efficiency = float(interactions.get("efficiency", 0.0))
        efficiency_raw = float(interactions.get("efficiency_raw", efficiency))
        tolerance_used = int(interactions.get("tolerance_used", 0))
        n_test_cases = int(interactions.get("n_test_cases", 0))
        mode = interactions.get("mode", self.experiment.get("mode"))
        quality_score = 100.0 * binary_accuracy if mode in ONESHOT_MODES else 100.0 * binary_accuracy * efficiency
        if interactions.get(METRIC_ABORTED, False):
            quality_score = np.nan

        self.log_episode_score(BENCH_SCORE, quality_score)
        self.log_episode_score("quality_score", quality_score)
        self.log_episode_score("binary_success", binary_success)
        self.log_episode_score("binary_accuracy", binary_accuracy)
        self.log_episode_score("efficiency", efficiency * 100.0)
        self.log_episode_score("efficiency_raw", efficiency_raw)
        self.log_episode_score("tolerance_used", tolerance_used)
        self.log_episode_score("n_test_cases", n_test_cases)


class FunctionDetectiveGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return FunctionDetective(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return FunctionDetectiveScorer(self.game_name, experiment, game_instance)
