from enum import Enum
from typing import Any, Callable, Dict, List


class Category(Enum):
    NUMBERS = "NUMBERS"
    TWO_NUMBERS = "TWO_NUMBERS"
    STRING = "STRING"
    LIST = "LIST"
    LOGIC = "LOGIC"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


FUNCTION_REGISTRY: List[Dict[str, Any]] = []


def register(category: Category, difficulty: Difficulty, confusers=None, family_id: str = "", hint: str = ""):
    confusers = confusers or []

    def decorator(function: Callable):
        FUNCTION_REGISTRY.append(
            {
                "function_name": function.__name__,
                "callable": function,
                "category": category.value,
                "difficulty": difficulty.value,
                "signature": _get_signature_str(function),
                "confusers": confusers,
                "family_id": family_id,
                "hint": hint,
            }
        )
        return function

    return decorator


def _get_signature_str(function):
    import inspect

    return str(inspect.signature(function))


# ============================================================
# NUMBERS
# ============================================================


@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["add_one", "sign_label"],
    family_id="numbers_basic",
    hint="Probe negative, zero, positive.",
)
def absolute_value(x: int) -> int:
    return abs(x)


@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["absolute_value", "clamp_to_0_9"],
    family_id="numbers_basic",
    hint="Simple arithmetic shift.",
)
def add_one(x: int) -> int:
    return x + 1


@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=["absolute_value", "sign_label"],
    family_id="numbers_boundaries",
    hint="Boundary fn. Test below 0, inside range, above 9.",
)
def clamp_to_0_9(x: int) -> int:
    if x < 0:
        return 0
    if x > 9:
        return 9
    return x


@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["sign_label", "add_one"],
    family_id="numbers_parity",
    hint="Parity fn. Test even, odd, negative.",
)
def is_even(x: int) -> bool:
    return x % 2 == 0


@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=["absolute_value", "is_even"],
    family_id="numbers_sign",
    hint="Three-way sign split. Test negative, zero, positive.",
)
def sign_label(x: int) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


# ============================================================
# TWO_NUMBERS
# ============================================================


@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["max_value", "first_minus_second"],
    family_id="two_numbers_basic",
    hint="Smaller value always wins.",
)
def min_value(a: int, b: int) -> int:
    return min(a, b)


@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["min_value", "sum_two"],
    family_id="two_numbers_basic",
    hint="Larger value always wins.",
)
def max_value(a: int, b: int) -> int:
    return max(a, b)


@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.EASY,
    confusers=["first_minus_second", "abs_diff"],
    family_id="two_numbers_arithmetic",
    hint="Plain addition. Order not important.",
)
def sum_two(a: int, b: int) -> int:
    return a + b


@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=["first_minus_second", "min_value"],
    family_id="two_numbers_arithmetic",
    hint="Distance fn. Swap inputs, output same.",
)
def abs_diff(a: int, b: int) -> int:
    return abs(a - b)


@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=["sum_two", "abs_diff"],
    family_id="two_numbers_order",
    hint="Order matters. Swap inputs, sign flips.",
)
def first_minus_second(a: int, b: int) -> int:
    return a - b


# ============================================================
# STRING
# ============================================================


@register(
    category=Category.STRING,
    difficulty=Difficulty.EASY,
    confusers=["first_char", "uppercase_str"],
    family_id="string_basic",
    hint="Reverse text. Non-palindrome best test.",
)
def reverse_str(text: str) -> str:
    return text[::-1]


@register(
    category=Category.STRING,
    difficulty=Difficulty.EASY,
    confusers=["reverse_str", "add_exclamation"],
    family_id="string_basic",
    hint="Short prefix fn. Test empty, one-char, longer.",
)
def first_char(text: str) -> str:
    return text[:1]


@register(
    category=Category.STRING,
    difficulty=Difficulty.EASY,
    confusers=["uppercase_str", "remove_space"],
    family_id="string_suffix",
    hint="Adds fixed punctuation suffix.",
)
def add_exclamation(text: str) -> str:
    return text + "!"


@register(
    category=Category.STRING,
    difficulty=Difficulty.EASY,
    confusers=["add_exclamation", "remove_space"],
    family_id="string_case",
    hint="Case conversion. Test lowercase, mixed, spaces.",
)
def uppercase_str(text: str) -> str:
    return text.upper()


@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=["uppercase_str", "reverse_str"],
    family_id="string_cleanup",
    hint="Remove all spaces. Test leading, middle, trailing.",
)
def remove_space(text: str) -> str:
    return text.replace(" ", "")


# ============================================================
# LIST
# ============================================================


@register(
    category=Category.LIST,
    difficulty=Difficulty.EASY,
    confusers=["count_zeros", "sum_list"],
    family_id="list_basic",
    hint="Length fn. Test empty, short, longer.",
)
def list_len(items: List[int]) -> int:
    return len(items)


@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=["list_len", "count_zeros"],
    family_id="list_access",
    hint="First item only. Empty list special case matters.",
)
def ret_first_item(items: List[int]) -> int:
    return items[0] if items else 0


@register(
    category=Category.LIST,
    difficulty=Difficulty.EASY,
    confusers=["count_zeros", "list_len"],
    family_id="list_arithmetic",
    hint="Aggregate sum across all items.",
)
def sum_list(items: List[int]) -> int:
    return sum(items)


@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=["list_len", "ret_first_item"],
    family_id="list_count",
    hint="Count zero entries only.",
)
def count_zeros(items: List[int]) -> int:
    return sum(1 for item in items if item == 0)


@register(
    category=Category.LIST,
    difficulty=Difficulty.HARD,
    confusers=["count_zeros", "ret_first_item"],
    family_id="list_order",
    hint="Remove duplicates, preserve first-seen order.",
)
def rmv_duplicates(items: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ============================================================
# LOGIC
# ============================================================


@register(
    category=Category.LOGIC,
    difficulty=Difficulty.EASY,
    confusers=["logic_or", "logic_xor"],
    family_id="logic_binary",
    hint="True only if both inputs true.",
)
def logic_and(a: bool, b: bool) -> bool:
    return a and b


@register(
    category=Category.LOGIC,
    difficulty=Difficulty.EASY,
    confusers=["logic_and", "logic_xor"],
    family_id="logic_binary",
    hint="True if at least one input true.",
)
def logic_or(a: bool, b: bool) -> bool:
    return a or b


@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=["logic_or", "logic_and"],
    family_id="logic_binary",
    hint="True if exactly one input true.",
)
def logic_xor(a: bool, b: bool) -> bool:
    return (a and not b) or (not a and b)


@register(
    category=Category.LOGIC,
    difficulty=Difficulty.EASY,
    confusers=["logic_xor", "logic_and"],
    family_id="logic_unary",
    hint="Unary negation. Test both truth values.",
)
def not_value(value: bool) -> bool:
    return not value


@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=["logic_or", "logic_and"],
    family_id="logic_relation",
    hint="Implication. Only false when first true, second false.",
)
def implies(a: bool, b: bool) -> bool:
    return (not a) or b
