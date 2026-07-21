import ast
import inspect
import json
import random
import re
import string
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args, get_origin
import logging

from tomlkit import value

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "functionigma-sandbox"
_SANDBOX_STATUS_CACHE: Optional[Tuple[bool, str]] = None


def get_sandbox_status(force_refresh: bool = False) -> Tuple[bool, str]:
    global _SANDBOX_STATUS_CACHE
    if _SANDBOX_STATUS_CACHE is not None and not force_refresh:
        return _SANDBOX_STATUS_CACHE

    try:
        docker_version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        _SANDBOX_STATUS_CACHE = (
            False,
            "Docker is not installed or not on PATH. Function validation requires Docker Desktop.",
        )
        return _SANDBOX_STATUS_CACHE
    except subprocess.TimeoutExpired:
        _SANDBOX_STATUS_CACHE = (False, "Docker did not respond in time. Start Docker Desktop and try again.")
        return _SANDBOX_STATUS_CACHE
    except Exception as exc:
        _SANDBOX_STATUS_CACHE = (False, f"Docker availability check failed: {exc}")
        return _SANDBOX_STATUS_CACHE

    if docker_version.returncode != 0:
        stderr = docker_version.stderr.strip() or docker_version.stdout.strip()
        _SANDBOX_STATUS_CACHE = (
            False,
            "Docker is not running or not reachable. "
            f"Start Docker Desktop and try again. Details: {stderr}",
        )
        return _SANDBOX_STATUS_CACHE

    image_check = subprocess.run(
        ["docker", "image", "inspect", SANDBOX_IMAGE],
        text=True,
        capture_output=True,
        timeout=5,
    )
    if image_check.returncode != 0:
        stderr = image_check.stderr.strip() or image_check.stdout.strip()
        _SANDBOX_STATUS_CACHE = (
            False,
            f"Docker sandbox image '{SANDBOX_IMAGE}' is missing or not accessible. Details: {stderr}",
        )
        return _SANDBOX_STATUS_CACHE

    _SANDBOX_STATUS_CACHE = (True, "")
    return _SANDBOX_STATUS_CACHE


def is_sandbox_failure(feedback: str) -> bool:
    if not feedback:
        return False

    markers = (
        "Sandbox Failure.",
        "Docker execution failed:",
        "Docker is not installed or not on PATH.",
        "Docker did not respond in time.",
        "Docker is not running or not reachable.",
        f"Docker sandbox image '{SANDBOX_IMAGE}' is missing or not accessible.",
        "failed to connect to the docker API",
        "dockerdesktoplinuxengine",
        "no such image",
    )
    lowered = feedback.lower()
    return any(marker.lower() in lowered for marker in markers)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


def _parse_signature_types_from_callable(actual_callable: Callable) -> List[Any]:
    sig = inspect.signature(actual_callable)
    annotations = []
    for param in sig.parameters.values():
        annotations.append(param.annotation if param.annotation is not inspect._empty else int)
    return annotations


def generate_random_value(annotation: Any, category: str = None) -> Any:
    cat = (category or "").lower()
    origin = get_origin(annotation)

    if origin in (list, List):
        args = get_args(annotation)
        inner_type = args[0] if args else int
        return [generate_random_value(inner_type, category) for _ in range(random.randint(0, 6))]

    if annotation is int:
        edge_cases = [0, 1, -1, 2, -2, 10, -10]
        if "math" in cat:
            edge_cases += [9, 11, 99, 100, -100, 1000]
        return random.choice(edge_cases + [random.randint(-50, 50)])

    if annotation is float:
        return random.choice([0.0, 1.0, -1.0, 0.5, -0.5, random.uniform(-100.0, 100.0)])

    if annotation is str:
        if "string" in cat:
            candidates = [
                "",
                " ",
                "a",
                "ab",
                "aba",
                "   ",
                "hello world",
                "12345",
                "a-b",
            ]
            if random.random() < 0.5:
                return random.choice(candidates)
            alphabet = string.ascii_letters + string.digits + "-_ "
            length = random.randint(1, 12)
            return "".join(random.choice(alphabet) for _ in range(length))
        if "logic" in cat or "formal" in cat:
            candidates = ["", "()", "(]", "([{}])", "([)]", "(())", ")(", "_x", "x1", "1x", "x-1"]
            return random.choice(candidates)
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for _ in range(random.randint(0, 8)))

    if annotation is bool:
        return random.choice([True, False])

    return random.randint(-50, 50)


def _edge_values_for_annotation(annotation: Any, category: Optional[str] = None) -> List[Any]:
    origin = get_origin(annotation)

    if origin in (list, List):
        inner = get_args(annotation)[0] if get_args(annotation) else int
        return [
            [],
            [generate_random_value(inner, category)],
            [0],
            [1, 1],
            [2, 1],
            [1, 2, 3],
            [3, 2, 1],
            [-1, 0, 1],
            [5, -5, 5],
        ]

    if annotation is int:
        base = [0, 1, -1, 2, -2, 10, -10, 99, -99, 1000, -1000]
        if category and category.lower() == "math":
            base += [100, -100, 50, -50]
        return base

    if annotation is float:
        return [0.0, 1.0, -1.0, 0.5, -0.5, 10.25, -10.25]

    if annotation is str:
        return ["", " ", "a", "ab", "aba", "A", "aA", "hello", "hello world", "123", "a-b", "   "]

    if annotation is bool:
        return [True, False]

    return [generate_random_value(annotation, category)]


def _special_cases_for_known_functions(function_name: str) -> List[List[Any]]:
    if function_name == "first_missing_positive":
        return [[[]], [[1]], [[2]], [[1, 2, 0]], [[3, 4, -1, 1]], [[1, 1, 2, 2]]]
    if function_name == "balanced_parentheses":
        return [[""], ["()"], ["(]"], ["([{}])"], ["([)]"], ["abc(def)"], [")("]]
    if function_name == "edit_distance":
        return [["", ""], ["", "a"], ["a", ""], ["a", "a"], ["kitten", "sitting"], ["flaw", "lawn"]]
    return []


def create_static_test_cases(
    actual_callable: Callable,
    category: str,
    num_tests: int = 50,
    signature: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[Dict]:
    try:
        param_annotations = _parse_signature_types_from_callable(actual_callable)
        sig = inspect.signature(actual_callable)
        param_count = len(sig.parameters)
    except ValueError:
        return []

    function_name = getattr(actual_callable, "__name__", "")
    edge_ratio = {"easy": 0.30, "medium": 0.45, "hard": 0.60}.get((difficulty or "").lower(), 0.40)
    edge_pools = [_edge_values_for_annotation(ann, category) for ann in param_annotations]

    test_cases: List[Dict] = []
    seen_args = set()

    def add_case(args: List[Any]) -> None:
        key = json.dumps(_json_safe(args), sort_keys=True)
        if key in seen_args:
            return
        try:
            expected = actual_callable(*args)
        except Exception:
            return
        test_cases.append({"args": args, "expected": _json_safe(expected)})
        seen_args.add(key)

    for args in _special_cases_for_known_functions(function_name):
        if len(args) == param_count:
            add_case(args)

    edge_budget = int(round(num_tests * edge_ratio))
    edge_attempts = 0
    while len(test_cases) < min(num_tests, edge_budget) and edge_attempts < max(200, num_tests * 20):
        edge_attempts += 1
        add_case([random.choice(pool) for pool in edge_pools])

    safety = 0
    while len(test_cases) < num_tests and safety < num_tests * 50:
        safety += 1
        add_case([generate_random_value(ann, category) for ann in param_annotations])

    return test_cases


def validate_function_logic(guessed_code: str, test_cases: List[Dict]) -> Tuple[bool, float, str]:
    sandbox_ok, sandbox_message = get_sandbox_status()
    if not sandbox_ok:
        return False, 0.0, sandbox_message

    input_args_list = [case["args"] for case in test_cases]
    expected_outputs = [case["expected"] for case in test_cases]
    payload = {"guessed_code": guessed_code, "test_cases": input_args_list}
    payload_json = json.dumps(payload)

    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "-i", SANDBOX_IMAGE],
            input=payload_json,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return False, 0.0, "Sandbox execution timed out."
    except Exception as exc:
        return False, 0.0, f"Docker execution failed: {exc}"

    try:
        response = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return False, 0.0, f"Sandbox Failure.\nStdout: '{result.stdout}'\nStderr: '{result.stderr}'"

    if response.get("status") != "ok":
        return False, 0.0, f"Runtime Error: {response.get('error')}"

    sandbox_results = response.get("results", [])
    if len(sandbox_results) != len(test_cases):
        return False, 0.0, "Mismatch in result count."

    passed_count = 0
    failures = []
    for i, (expected_raw, actual) in enumerate(zip(expected_outputs, sandbox_results)):
        expected = _json_safe(expected_raw)
        if expected == actual:
            passed_count += 1
        else:
            failures.append(f"Input: {input_args_list[i]} | Expected: {expected} | Got: {actual}")

    accuracy = passed_count / len(test_cases) if test_cases else 0.0
    if passed_count == len(test_cases):
        return True, 1.0, "All tests passed!"

    return False, accuracy, "Failed Cases:\n" + "\n".join(failures[:3])


def extract_function_code(text: str) -> Optional[str]:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_signature_with_types(signature: str) -> Tuple[List[str], List[str], str]:
    src = signature.strip()
    if src.startswith("def "):
        tree = ast.parse(src)
        func = tree.body[0]
        params = [arg.arg for arg in func.args.args]
        types = [ast.unparse(arg.annotation) if arg.annotation is not None else "Any" for arg in func.args.args]
        ret = ast.unparse(func.returns) if func.returns is not None else "Any"
        return params, types, ret

    fake = f"def _f({src.split('(')[1]}"
    try:
        tree = ast.parse(fake)
        func = tree.body[0]
        params = [arg.arg for arg in func.args.args]
        types = [ast.unparse(arg.annotation) if arg.annotation is not None else "Any" for arg in func.args.args]
        ret = src.split("->")[-1].strip() if "->" in src else "Any"
        return params, types, ret
    except Exception:
        match = re.match(r".*\((.*)\)\s*->\s*(.+)", src)
        if not match:
            return [], [], "Any"
        params_raw = match.group(1).strip()
        parts = [part.strip() for part in params_raw.split(",")] if params_raw else []
        names, types_out = [], []
        for part in parts:
            if ":" in part:
                name, type_name = part.split(":", 1)
                names.append(name.strip())
                types_out.append(type_name.strip())
            else:
                names.append(part.split()[0])
                types_out.append("Any")
        return names, types_out, match.group(2).strip()


def type_matches_annotation(value: Any, expected: str) -> bool:
    normalized = (expected or "Any").strip().replace("typing.", "")
    if normalized in {"Any", ""}:
        return True
    if normalized in {"int", "Integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"float", "double"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized in {"str", "string"}:
        return isinstance(value, str)
    if normalized in {"bool", "boolean"}:
        return isinstance(value, bool)
    if normalized.startswith("List[") or normalized.startswith("list[") or normalized == "list":
        return isinstance(value, list)
    return True


def parse_active_io_probe(raw: str, num_params: int) -> Tuple[Any, ...]:
    #logger.info(f"Parsing active I/O probe: raw='{raw}', num_params={num_params}")
    payload = raw.strip()
    if num_params == 0:
        return tuple()
    tuple_text = f"({payload})" if num_params > 1 else payload
    #logger.info(f"Tuple text for parsing: '{tuple_text}', {type(tuple_text)}'")
    #logger.info(repr(tuple_text))
    #print(ast.literal_eval(tuple_text))    
    parsed = ast.literal_eval(str(tuple_text))
    if num_params == 1:
        return (parsed,)
    if isinstance(parsed, tuple):
        return parsed
    if isinstance(parsed, list):
        return tuple(parsed)
    raise ValueError("Active I/O probe must contain one value per input parameter.")


def parse_membership_probe(raw: str, num_params: int) -> Tuple[List[Any], Any]:
    payload = raw.strip()
    parsed = ast.literal_eval(f"({payload})")
    if not isinstance(parsed, tuple):
        raise ValueError("Membership probe must parse to a tuple.")
    if len(parsed) != num_params + 1:
        #Check if the input is an iterable of inputs followed by a candidate output
        if len(parsed) == 2 and isinstance(parsed[0], (list, tuple)):
            inputs, candidate_output = parsed
            if len(inputs) == num_params:
                return list(inputs), candidate_output
        raise ValueError("Membership probe must contain function inputs followed by a candidate output.")
    values = list(parsed)
    return values[:-1], values[-1]


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_input_object(args: List[Any]) -> str:
    if len(args) == 0:
        return "()"
    if len(args) == 1:
        return format_value(args[0])
    return "(" + ", ".join(format_value(arg) for arg in args) + ")"


def format_io_example(args: List[Any], expected: Any) -> str:
    return f"({format_input_object(args)}, {format_value(expected)})"


def format_membership_example(args: List[Any], candidate_output: Any, is_member: bool) -> str:
    return f"(({format_input_object(args)}, {format_value(candidate_output)}), {is_member})"


def render_examples_block(passive_examples: List[Dict]) -> str:
    if not passive_examples:
        return "No examples are preloaded."
    lines = []
    for index, example in enumerate(passive_examples, start=1):
        if example["kind"] == "io":
            rendered = format_io_example(example["args"], example["expected"])
        else:
            rendered = format_membership_example(example["args"], example["candidate_output"], example["is_member"])
        lines.append(f"{index}. {rendered}")
    return "\n".join(lines)


def _case_signature(case: Dict) -> str:
    return json.dumps(_json_safe(case), sort_keys=True)


def select_informative_cases(test_cases: List[Dict], num_examples: int) -> List[Dict]:
    if num_examples <= 0:
        return []

    chosen: List[Dict] = []
    seen_outputs = set()
    seen_inputs = set()

    for case in test_cases:
        input_key = json.dumps(_json_safe(case["args"]), sort_keys=True)
        output_key = json.dumps(_json_safe(case["expected"]), sort_keys=True)
        if output_key not in seen_outputs or input_key not in seen_inputs:
            chosen.append(case)
            seen_inputs.add(input_key)
            seen_outputs.add(output_key)
        if len(chosen) >= num_examples:
            return chosen

    for case in test_cases:
        key = _case_signature(case)
        if any(_case_signature(existing) == key for existing in chosen):
            continue
        chosen.append(case)
        if len(chosen) >= num_examples:
            break

    return chosen


def _alternate_bool(expected: Any) -> bool:
    return not bool(expected)


def _alternate_int(expected: Any, observed_values: List[Any]) -> int:
    deltas = [1, -1, 2, -2, 10]
    for delta in deltas:
        candidate = int(expected) + delta
        if candidate != expected and candidate not in observed_values:
            return candidate
    return int(expected) + 1


def _alternate_float(expected: Any, observed_values: List[Any]) -> float:
    deltas = [0.5, -0.5, 1.0, -1.0]
    for delta in deltas:
        candidate = float(expected) + delta
        if candidate != expected and candidate not in observed_values:
            return candidate
    return float(expected) + 0.5


def _alternate_str(expected: Any, observed_values: List[Any]) -> str:
    candidates = [
        f"{expected}_alt",
        f"{expected}!",
        f"{expected}{len(str(expected))}",
        str(expected)[::-1],
        "",
    ]
    for candidate in candidates:
        if candidate != expected and candidate not in observed_values:
            return candidate
    return f"{expected}_neg"


def generate_negative_output(expected: Any, return_type: str, observed_values: Optional[List[Any]] = None) -> Any:
    observed_values = observed_values or []
    normalized = (return_type or "Any").strip().replace("typing.", "")

    if normalized in {"bool", "boolean"}:
        return _alternate_bool(expected)
    if normalized in {"int", "Integer"}:
        return _alternate_int(expected, observed_values)
    if normalized in {"float", "double"}:
        return _alternate_float(expected, observed_values)
    if normalized in {"str", "string"}:
        return _alternate_str(expected, observed_values)
    if normalized.startswith("List[") or normalized.startswith("list[") or normalized == "list":
        candidate = list(expected) if isinstance(expected, list) else [expected]
        candidate = candidate + ["alt"]
        return candidate if candidate != expected else candidate + ["x"]
    return f"{expected}_neg"


def create_passive_io_examples(static_tests: List[Dict], num_examples: int) -> List[Dict]:
    informative_cases = select_informative_cases(static_tests, num_examples)
    examples = [
        {"kind": "io", "args": case["args"], "expected": case["expected"]}
        for case in informative_cases
    ]

    if not examples:
        return []

    # Small finite domains such as boolean logic can exhaust unique valid pairs quickly.
    # Repeat the informative pool so passive modes still have a stable evidence budget.
    while len(examples) < num_examples:
        source = informative_cases[len(examples) % len(informative_cases)]
        examples.append({"kind": "io", "args": source["args"], "expected": source["expected"]})

    return examples[:num_examples]


def create_passive_membership_examples(
    static_tests: List[Dict],
    return_type: str,
    category: str,
    num_examples: int,
) -> List[Dict]:
    informative_cases = select_informative_cases(static_tests, max(num_examples, 2))
    observed_outputs = [case["expected"] for case in informative_cases]
    examples: List[Dict] = []

    for index, case in enumerate(informative_cases):
        if len(examples) >= num_examples:
            break

        if index % 2 == 0:
            examples.append(
                {
                    "kind": "membership",
                    "args": case["args"],
                    "candidate_output": case["expected"],
                    "is_member": True,
                }
            )
        else:
            examples.append(
                {
                    "kind": "membership",
                    "args": case["args"],
                    "candidate_output": generate_negative_output(case["expected"], return_type, observed_outputs),
                    "is_member": False,
                }
            )

    true_count = sum(1 for example in examples if example["is_member"])
    false_count = len(examples) - true_count
    case_iter = iter(informative_cases)

    while len(examples) < num_examples:
        case = next(case_iter, informative_cases[len(examples) % len(informative_cases)])
        needs_true = true_count <= false_count
        if needs_true:
            examples.append(
                {
                    "kind": "membership",
                    "args": case["args"],
                    "candidate_output": case["expected"],
                    "is_member": True,
                }
            )
            true_count += 1
        else:
            examples.append(
                {
                    "kind": "membership",
                    "args": case["args"],
                    "candidate_output": generate_negative_output(case["expected"], return_type, observed_outputs),
                    "is_member": False,
                }
            )
            false_count += 1

    return examples[:num_examples]
