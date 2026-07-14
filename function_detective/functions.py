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


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns twice the input plus three.',
)
def double_and_add_three(x: int) -> int:
    return x * 2 + 3


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the square of the input minus the input itself.',
)
def square_minus_input(x: int) -> int:
    return x * x - x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the absolute distance between the input and ten.',
)
def absolute_distance_from_ten(x: int) -> int:
    return abs(x - 10)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the input unchanged if even, otherwise its negation.',
)
def negate_if_odd(x: int) -> int:
    if x % 2 == 0:
        return x
    return -x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of the decimal digits of the absolute value of the input.',
)
def sum_of_decimal_digits(x: int) -> int:
    total = 0
    for ch in str(abs(x)):
        total += ord(ch) - ord('0')
    return total


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns 1 for positive inputs, -1 for negative inputs, and 0 for zero.',
)
def cube_sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the number of ones in the binary representation of the absolute value.',
)
def count_set_bits(x: int) -> int:
    return bin(abs(x)).count('1')


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the triangular number of the absolute value of the input.',
)
def triangular_number_of_abs(x: int) -> int:
    n = abs(x)
    return n * (n + 1) // 2


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the input with its decimal digits reversed while preserving sign.',
)
def reverse_digits_preserve_sign(x: int) -> int:
    sign = -1 if x < 0 else 1
    return sign * int(str(abs(x))[::-1])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the greatest multiple of five less than or equal to the input.',
)
def nearest_multiple_of_five_below(x: int) -> int:
    return x - (x % 5)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all integers in the list.',
)
def sum_of_items(items: List[int]) -> int:
    return sum(items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Counts how many integers in the list are negative.',
)
def count_negative_values(items: List[int]) -> int:
    return sum(1 for x in items if x < 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the difference between the largest and smallest values, or 0 for an empty list.',
)
def max_minus_min(items: List[int]) -> int:
    if not items:
        return 0
    return max(items) - min(items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all even integers in the list.',
)
def sum_of_even_items(items: List[int]) -> int:
    return sum(x for x in items if x % 2 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the product of all nonzero integers, or 1 if there are none.',
)
def product_of_nonzero_items(items: List[int]) -> int:
    result = 1
    for x in items:
        if x != 0:
            result *= x
    return result


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of the absolute values of all integers in the list.',
)
def sum_of_absolute_values(items: List[int]) -> int:
    return sum(abs(x) for x in items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the last item minus the first item, or 0 if the list has fewer than two elements.',
)
def last_minus_first(items: List[int]) -> int:
    if len(items) < 2:
        return 0
    return items[-1] - items[0]


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Counts adjacent positions where the next value is greater than the current one.',
)
def count_strictly_increasing_steps(items: List[int]) -> int:
    count = 0
    for i in range(len(items) - 1):
        if items[i + 1] > items[i]:
            count += 1
    return count


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the alternating sum of the list, adding even-indexed items and subtracting odd-indexed items.',
)
def alternating_sum(items: List[int]) -> int:
    total = 0
    for i, x in enumerate(items):
        total += x if i % 2 == 0 else -x
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of values that appear exactly once in the list.',
)
def sum_of_unique_values(items: List[int]) -> int:
    total = 0
    for x in items:
        if items.count(x) == 1:
            total += x
    return total


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the input string with characters in reverse order.',
)
def reverse_text(text: str) -> str:
    return text[::-1]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Alternates uppercase and lowercase letters across the string.',
)
def alternate_case(text: str) -> str:
    result = ''
    for i, ch in enumerate(text):
        result += ch.upper() if i % 2 == 0 else ch.lower()
    return result


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Repeats every character in the input exactly twice.',
)
def duplicate_characters(text: str) -> str:
    return ''.join(ch * 2 for ch in text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes all lowercase and uppercase vowels from the string.',
)
def remove_vowels(text: str) -> str:
    vowels = 'aeiouAEIOU'
    return ''.join(ch for ch in text if ch not in vowels)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Splits on whitespace and returns the words sorted alphabetically.',
)
def sort_words_alphabetically(text: str) -> str:
    words = text.split()
    words.sort()
    return ' '.join(words)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Wraps each whitespace-separated word in square brackets.',
)
def surround_with_brackets(text: str) -> str:
    return ' '.join('[' + word + ']' for word in text.split())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Normalizes whitespace by collapsing runs into single spaces and trimming ends.',
)
def collapse_spaces(text: str) -> str:
    return ' '.join(text.split())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Moves the first character to the end of the string.',
)
def rotate_first_char_to_end(text: str) -> str:
    if not text:
        return ''
    return text[1:] + text[0]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the string followed by a bar and its reverse.',
)
def mirror_with_separator(text: str) -> str:
    return text + '|' + text[::-1]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Replaces every numeric digit in the string with a hash symbol.',
)
def replace_digits_with_hash(text: str) -> str:
    return ''.join('#' if ch.isdigit() else ch for ch in text)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True only when both inputs are True.',
)
def both_true(a: bool, b: bool) -> bool:
    return a and b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when at least one input is True.',
)
def either_true(a: bool, b: bool) -> bool:
    return a or b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when exactly one of the two inputs is True.',
)
def exactly_one_true(a: bool, b: bool) -> bool:
    return a != b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when the two inputs have the same truth value.',
)
def both_equal(a: bool, b: bool) -> bool:
    return a == b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns whether a logically implies b.',
)
def a_implies_b(a: bool, b: bool) -> bool:
    return (not a) or b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns whether b logically implies a.',
)
def b_implies_a(a: bool, b: bool) -> bool:
    return (not b) or a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is True and b is False.',
)
def a_and_not_b(a: bool, b: bool) -> bool:
    return a and (not b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is False or b is True.',
)
def not_a_or_b(a: bool, b: bool) -> bool:
    return (not a) or b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True only when both inputs are False.',
)
def neither_true(a: bool, b: bool) -> bool:
    return not (a or b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when no more than one input is True.',
)
def at_most_one_true(a: bool, b: bool) -> bool:
    return not (a and b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the squares of the two inputs.',
)
def sum_of_squares(a: int, b: int) -> int:
    return a * a + b * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute difference between the two inputs.',
)
def absolute_difference(a: int, b: int) -> int:
    return a - b if a >= b else b - a


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the larger input minus the smaller input.',
)
def larger_minus_smaller(a: int, b: int) -> int:
    if a > b:
        return a - b
    return b - a


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the product of the inputs increased by one.',
)
def product_plus_one(a: int, b: int) -> int:
    return a * b + 1


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the larger input multiplied by the smaller input.',
)
def max_times_min(a: int, b: int) -> int:
    if a >= b:
        return a * b
    return b * a


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of both inputs after replacing negatives with zero.',
)
def nonnegative_sum(a: int, b: int) -> int:
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    return a + b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns two times the first input plus three times the second.',
)
def twice_first_plus_thrice_second(a: int, b: int) -> int:
    return 2 * a + 3 * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the square of the smaller input plus the larger input.',
)
def smaller_squared_plus_larger(a: int, b: int) -> int:
    if a <= b:
        return a * a + b
    return b * b + a


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the difference between the absolute values of the inputs.',
)
def distance_from_zero_difference(a: int, b: int) -> int:
    aa = a if a >= 0 else -a
    bb = b if b >= 0 else -b
    return aa - bb


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the inputs plus one if they have the same parity.',
)
def shared_parity_bonus(a: int, b: int) -> int:
    bonus = 1 if (a % 2) == (b % 2) else 0
    return a + b + bonus
