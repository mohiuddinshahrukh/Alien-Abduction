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


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Appends an exclamation mark to the string.',
)
def add_exclamation(text: str) -> str:
    return text + '!'


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
    hint='Returns the square of the input minus the input.',
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
    hint='Returns the sum of the decimal digits of the absolute value of x.',
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
    hint='Counts the number of one bits in the binary form of the absolute input.',
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
    hint='Reverses decimal digits and preserves the original sign.',
)
def reverse_digits_preserve_sign(x: int) -> int:
    s = str(abs(x))[::-1]
    n = int(s)
    return -n if x < 0 else n


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
    total = 0
    for value in items:
        if value % 2 == 0:
            total += value
    return total


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
    hint='Removes all vowels from the string.',
)
def remove_vowels(text: str) -> str:
    return ''.join(ch for ch in text if ch not in 'aeiouAEIOU')


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
    hint='Replaces each digit with a hash sign.',
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
    hint='Returns True when exactly one input is True.',
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
    hint='Returns True unless b is True and a is False.',
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
    hint='Returns the sum of the squares of the two integers.',
)
def sum_of_squares(a: int, b: int) -> int:
    return a * a + b * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute difference between the integers.',
)
def absolute_difference(a: int, b: int) -> int:
    return abs(a - b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the difference between the larger integer and the smaller one.',
)
def larger_minus_smaller(a: int, b: int) -> int:
    if a >= b:
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


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns x plus the triangular number of the absolute value of x.',
)
def triangular_offset(x: int) -> int:
    n = abs(x)
    return x + n * (n + 1) // 2


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the difference between the maximum and minimum values, or 0 for an empty list.',
)
def range_span(items: List[int]) -> int:
    if not items:
        return 0
    return max(items) - min(items)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Reverses the order of whitespace-separated words in the input text.',
)
def reverse_word_order(text: str) -> str:
    words = text.split()
    return ' '.join(reversed(words))


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Duplicates each lowercase and uppercase vowel in the input text.',
)
def duplicate_vowels(text: str) -> str:
    vowels = 'aeiouAEIOU'
    return ''.join(ch * 2 if ch in vowels else ch for ch in text)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True only when both inputs are True.',
)
def both_are_true(a: bool, b: bool) -> bool:
    return a and b


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Subtracts the reversal of the absolute decimal digits from the input.',
)
def difference_from_reversed_absolute(x: int) -> int:
    return x - int(str(abs(x))[::-1])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the smallest divisor of the absolute input that is at least two, or zero when none exists.',
)
def smallest_divisor_ge_two_abs(x: int) -> int:
    n = abs(x)
    if n < 2:
        return 0
    for i in range(2, n + 1):
        if n % i == 0:
            return i
    return 0


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the absolute difference between the input and nine.',
)
def distance_from_square_of_three(x: int) -> int:
    return abs(x - 9)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Adds seven to negative inputs and returns nonnegative inputs unchanged.',
)
def add_seven_if_negative(x: int) -> int:
    if x < 0:
        return x + 7
    return x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the larger of the input and its negation.',
)
def max_with_negative_self(x: int) -> int:
    return max(x, -x)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns four times the input modulo nine.',
)
def times_four_mod_nine(x: int) -> int:
    return (4 * x) % 9


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the floor average of the input and ten.',
)
def integer_average_with_ten(x: int) -> int:
    return (x + 10) // 2


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the cube of the input modulo eleven.',
)
def modulo_of_cube_by_eleven(x: int) -> int:
    return (x * x * x) % 11


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the absolute difference between the input remainder mod eleven and five.',
)
def absolute_modulus_gap_eleven(x: int) -> int:
    return abs((x % 11) - 5)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the product of the first and last decimal digits of the absolute input.',
)
def leading_trailing_digit_product(x: int) -> int:
    s = str(abs(x))
    return int(s[0]) * int(s[-1])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the absolute value plus one if the input is odd.',
)
def absolute_value_plus_parity(x: int) -> int:
    return abs(x) + (x % 2)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Removes the first decimal digit from the absolute input.',
)
def remove_first_digit(x: int) -> int:
    s = str(abs(x))
    return int(s[1:]) if len(s) > 1 else 0


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Multiplies the first and last decimal digits of the absolute input.',
)
def last_digit_times_first_digit(x: int) -> int:
    s = str(abs(x))
    return int(s[0]) * int(s[-1])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns three times the input modulo five.',
)
def times_three_mod_five(x: int) -> int:
    return (3 * x) % 5


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns one when the input is divisible by three, else zero.',
)
def is_multiple_of_three_indicator(x: int) -> int:
    return 1 if x % 3 == 0 else 0


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the bit length of the absolute input.',
)
def bit_length_of_absolute(x: int) -> int:
    return abs(x).bit_length()


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns x squared for even inputs and x cubed for odd inputs.',
)
def even_square_odd_cube(x: int) -> int:
    if x % 2 == 0:
        return x * x
    return x * x * x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns one if the input is negative, else zero.',
)
def negative_indicator(x: int) -> int:
    return 1 if x < 0 else 0


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Adds the floor quotient and remainder when dividing the input by five.',
)
def remainder_plus_quotient_by_five(x: int) -> int:
    q, r = divmod(x, 5)
    return q + r


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Counts the nonzero decimal digits of the absolute input.',
)
def nonzero_digit_count(x: int) -> int:
    return sum(1 for ch in str(abs(x)) if ch != '0')


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Forms a three-digit number by repeating the last digit of the absolute input.',
)
def repeat_last_digit_thrice(x: int) -> int:
    d = str(abs(x))[-1]
    return int(d + d + d)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the smallest multiple of four greater than or equal to the input.',
)
def next_multiple_of_four(x: int) -> int:
    r = x % 4
    return x if r == 0 else x + (4 - r)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Adds one to the input if its absolute value has an odd number of digits.',
)
def increment_until_even_count_digits(x: int) -> int:
    if len(str(abs(x))) % 2 == 1:
        return x + 1
    return x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Subtracts the remainder of the absolute value modulo four from the input.',
)
def input_minus_absolute_mod_four(x: int) -> int:
    return x - (abs(x) % 4)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the triangular number of the absolute value of the input.',
)
def triangular_number_of_absolute(x: int) -> int:
    n = abs(x)
    return n * (n + 1) // 2


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of the odd decimal digits of the absolute input.',
)
def sum_of_odd_digits(x: int) -> int:
    total = 0
    for ch in str(abs(x)):
        d = int(ch)
        if d % 2 == 1:
            total += d
    return total


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of the first and last decimal digits of the absolute input.',
)
def leading_trailing_digit_sum(x: int) -> int:
    s = str(abs(x))
    return int(s[0]) + int(s[-1])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the first decimal digit of the absolute input.',
)
def leading_digit_value(x: int) -> int:
    return int(str(abs(x))[0])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Adds the input to the remainder of its absolute value modulo four.',
)
def input_plus_absolute_mod_four(x: int) -> int:
    return x + (abs(x) % 4)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns how far the input is above the previous multiple of ten.',
)
def distance_from_previous_ten(x: int) -> int:
    return x % 10


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the difference between the next integer and the input.',
)
def difference_of_consecutive_pair(x: int) -> int:
    return (x + 1) - x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Subtracts one when the input is divisible by four, otherwise returns it unchanged.',
)
def decrement_if_divisible_by_four(x: int) -> int:
    if x % 4 == 0:
        return x - 1
    return x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the count of odd digits minus the count of even digits in the absolute input.',
)
def odd_even_balance(x: int) -> int:
    odds = 0
    evens = 0
    for ch in str(abs(x)):
        if int(ch) % 2:
            odds += 1
        else:
            evens += 1
    return odds - evens


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Counts the positive divisors of the absolute input, treating zero as zero.',
)
def count_divisors_of_absolute(x: int) -> int:
    n = abs(x)
    if n == 0:
        return 0
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of the previous and next integers around the input.',
)
def sum_of_previous_and_next(x: int) -> int:
    return (x - 1) + (x + 1)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of squares of the decimal digits of the absolute input.',
)
def sum_of_digit_squares(x: int) -> int:
    return sum(int(ch) * int(ch) for ch in str(abs(x)))


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the least common multiple of the absolute input and six, using zero when the input is zero.',
)
def lcm_with_six(x: int) -> int:
    a = abs(x)
    if a == 0:
        return 0
    b = 6
    g1, g2 = a, b
    while g2:
        g1, g2 = g2, g1 % g2
    return a * b // g1


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the largest even integer less than or equal to the input.',
)
def largest_even_not_above(x: int) -> int:
    return x if x % 2 == 0 else x - 1


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Multiplies the input by the number of digits in its absolute value.',
)
def multiply_by_digit_count(x: int) -> int:
    return x * len(str(abs(x)))


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the sum of factorials of the decimal digits of the absolute input.',
)
def sum_of_factorials_of_digits(x: int) -> int:
    total = 0
    for ch in str(abs(x)):
        d = int(ch)
        f = 1
        for i in range(2, d + 1):
            f *= i
        total += f
    return total


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Squares the sum of digits of the absolute input.',
)
def square_of_digit_sum(x: int) -> int:
    s = sum(int(ch) for ch in str(abs(x)))
    return s * s


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Removes the last decimal digit from the absolute input.',
)
def remove_last_digit(x: int) -> int:
    return abs(x) // 10


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns negative one minus the input.',
)
def ones_complement_like(x: int) -> int:
    return -1 - x


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the square of the input modulo seven.',
)
def modulo_of_square_by_seven(x: int) -> int:
    return (x * x) % 7


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the absolute value of the input plus three.',
)
def absolute_plus_three(x: int) -> int:
    return abs(x) + 3


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Swaps the first and last digits of the absolute input.',
)
def swap_first_last_digits(x: int) -> int:
    s = str(abs(x))
    if len(s) == 1:
        return int(s)
    return int(s[-1] + s[1:-1] + s[0])


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns twice the input plus one.',
)
def double_plus_one(x: int) -> int:
    return x * 2 + 1


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of interior elements greater than both neighbors.',
)
def sum_local_maxima(items: List[int]) -> int:
    return sum(items[i] for i in range(1, len(items) - 1) if items[i] > items[i - 1] and items[i] > items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the maximum value minus the sum of all other values.',
)
def max_minus_sum_rest(items: List[int]) -> int:
    return 2 * max(items) - sum(items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many running prefix sums are odd.',
)
def count_prefix_sums_odd(items: List[int]) -> int:
    total = 0
    count = 0
    for x in items:
        total += x
        if total % 2 != 0:
            count += 1
    return count


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the list sum multiplied by the list length.',
)
def sum_items_times_length(items: List[int]) -> int:
    return sum(items) * len(items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns count of even values minus count of odd values.',
)
def parity_balance(items: List[int]) -> int:
    return sum(1 for x in items if x % 2 == 0) - sum(1 for x in items if x % 2 != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the second-to-last element, or zero if it does not exist.',
)
def second_last_or_zero(items: List[int]) -> int:
    return items[-2] if len(items) > 1 else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many values are single-digit in absolute value.',
)
def count_single_digit_values(items: List[int]) -> int:
    return sum(1 for x in items if abs(x) < 10)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of times the sequence decreases between neighbors.',
)
def count_decreases(items: List[int]) -> int:
    total = 0
    for i in range(1, len(items)):
        if items[i] < items[i - 1]:
            total += 1
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of odd values located at odd indices.',
)
def sum_odd_indices_odd_values(items: List[int]) -> int:
    return sum(items[i] for i in range(1, len(items), 2) if items[i] % 2 != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the difference between the maximum and minimum values.',
)
def range_width(items: List[int]) -> int:
    return max(items) - min(items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of each value multiplied by its one-based position.',
)
def weighted_one_based_sum(items: List[int]) -> int:
    return sum((i + 1) * x for i, x in enumerate(items))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of adjacent pairs with equal values.',
)
def count_adjacent_equal_pairs(items: List[int]) -> int:
    return sum(1 for i in range(len(items) - 1) if items[i] == items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of distinct integers in the list.',
)
def sum_distinct_values(items: List[int]) -> int:
    return sum(set(items))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the second element, or zero if it does not exist.',
)
def second_element_or_zero(items: List[int]) -> int:
    return items[1] if len(items) > 1 else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of absolute values of all elements.',
)
def sum_absolute_values(items: List[int]) -> int:
    return sum(abs(x) for x in items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many elements have an odd absolute value.',
)
def count_values_with_odd_absolute(items: List[int]) -> int:
    return sum(1 for x in items if abs(x) % 2 != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of running minimum values.',
)
def sum_prefix_minima(items: List[int]) -> int:
    total = 0
    current = 0
    seen = False
    for x in items:
        if not seen or x < current:
            current = x
            seen = True
        total += current
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many elements are even.',
)
def count_even_values(items: List[int]) -> int:
    return sum(1 for x in items if x % 2 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all odd numbers in the list.',
)
def sum_odd_values(items: List[int]) -> int:
    return sum(x for x in items if x % 2 != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of times the sequence increases between neighbors.',
)
def count_increases(items: List[int]) -> int:
    total = 0
    for i in range(1, len(items)):
        if items[i] > items[i - 1]:
            total += 1
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of squares of negative values.',
)
def sum_of_negative_squares(items: List[int]) -> int:
    return sum(x * x for x in items if x < 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all elements except one minimum value.',
)
def sum_excluding_min(items: List[int]) -> int:
    return sum(items) - min(items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of positive integers.',
)
def count_positive_values(items: List[int]) -> int:
    return sum(1 for x in items if x > 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of elements that are greater than the previous element.',
)
def sum_greater_than_previous(items: List[int]) -> int:
    return sum(items[i] for i in range(1, len(items)) if items[i] > items[i - 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of absolute values of even elements.',
)
def sum_absolute_even_values(items: List[int]) -> int:
    return sum(abs(x) for x in items if x % 2 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of differences between adjacent elements.',
)
def sum_neighbors_differences(items: List[int]) -> int:
    return sum(items[i + 1] - items[i] for i in range(len(items) - 1))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of zeros in the list.',
)
def count_zero_values(items: List[int]) -> int:
    return sum(1 for x in items if x == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of elements that come after the first positive value.',
)
def sum_values_after_first_positive(items: List[int]) -> int:
    for i, x in enumerate(items):
        if x > 0:
            return sum(items[i + 1:])
    return 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of elements except the first and last.',
)
def count_middle_excluding_ends(items: List[int]) -> int:
    return max(len(items) - 2, 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many elements are greater than the sum of all previous elements.',
)
def count_greater_than_sum_previous(items: List[int]) -> int:
    total = 0
    count = 0
    for x in items:
        if x > total:
            count += 1
        total += x
    return count


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the smallest absolute value, or zero if empty.',
)
def min_absolute_value(items: List[int]) -> int:
    return min(abs(x) for x in items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many interior elements are less than both neighbors.',
)
def count_local_minima(items: List[int]) -> int:
    return sum(1 for i in range(1, len(items) - 1) if items[i] < items[i - 1] and items[i] < items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many even indices contain even values.',
)
def count_even_indices_with_even_values(items: List[int]) -> int:
    return sum(1 for i in range(0, len(items), 2) if items[i] % 2 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of sums of all adjacent pairs.',
)
def sum_adjacent_pair_sums(items: List[int]) -> int:
    return sum(items[i] + items[i + 1] for i in range(len(items) - 1))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of products of adjacent pairs.',
)
def sum_neighbors_products(items: List[int]) -> int:
    return sum(items[i] * items[i + 1] for i in range(len(items) - 1))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of running maximum values.',
)
def sum_prefix_maxima(items: List[int]) -> int:
    total = 0
    current = 0
    seen = False
    for x in items:
        if not seen or x > current:
            current = x
            seen = True
        total += current
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many values end with digit zero.',
)
def count_ends_with_zero(items: List[int]) -> int:
    return sum(1 for x in items if x % 10 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many interior elements are greater than both neighbors.',
)
def count_local_maxima(items: List[int]) -> int:
    return sum(1 for i in range(1, len(items) - 1) if items[i] > items[i - 1] and items[i] > items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all nonzero elements.',
)
def sum_nonzero_values(items: List[int]) -> int:
    return sum(x for x in items if x != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many values read the same forward and backward ignoring sign.',
)
def count_palindrome_values(items: List[int]) -> int:
    return sum(1 for x in items if str(abs(x)) == str(abs(x))[::-1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of the first value in each adjacent equal pair.',
)
def sum_adjacent_equal_values(items: List[int]) -> int:
    return sum(items[i] for i in range(len(items) - 1) if items[i] == items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many running prefix sums are positive.',
)
def count_prefix_sums_positive(items: List[int]) -> int:
    total = 0
    count = 0
    for x in items:
        total += x
        if total > 0:
            count += 1
    return count


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of even values located at even indices.',
)
def sum_even_indices_even_values(items: List[int]) -> int:
    return sum(items[i] for i in range(0, len(items), 2) if items[i] % 2 == 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of each value multiplied by its index.',
)
def weighted_index_sum(items: List[int]) -> int:
    return sum(i * x for i, x in enumerate(items))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many running prefix sums are even.',
)
def count_prefix_sums_even(items: List[int]) -> int:
    total = 0
    count = 0
    for x in items:
        total += x
        if total % 2 == 0:
            count += 1
    return count


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of absolute values of odd elements.',
)
def sum_absolute_odd_values(items: List[int]) -> int:
    return sum(abs(x) for x in items if x % 2 != 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the maximum value, or zero for an empty list.',
)
def max_or_zero(items: List[int]) -> int:
    return max(items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of the first half of the list.',
)
def sum_first_half(items: List[int]) -> int:
    return sum(items[:len(items) // 2])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of the first value of each suffix, which equals the list sum.',
)
def sum_suffix_start_values(items: List[int]) -> int:
    total = 0
    for i in range(len(items)):
        total += items[i]
    return total


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the list length minus the sum of the elements.',
)
def length_minus_sum(items: List[int]) -> int:
    return len(items) - sum(items)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Wraps the string in double quote characters.',
)
def wrap_in_double_quotes(text: str) -> str:
    return '"' + text + '"'


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the first half of the string.',
)
def first_half(text: str) -> str:
    return text[:len(text) // 2]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the string without its last character.',
)
def remove_last_character(text: str) -> str:
    return text[:-1]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Keeps every second character starting from the first.',
)
def every_second_character(text: str) -> str:
    return text[::2]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Swaps uppercase letters to lowercase and vice versa.',
)
def swap_case_text(text: str) -> str:
    return text.swapcase()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Wraps the string in parentheses.',
)
def surround_with_parentheses(text: str) -> str:
    return '(' + text + ')'


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Appends a colon and the string length.',
)
def suffix_colon_and_length(text: str) -> str:
    return text + ':' + str(len(text))


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the first whitespace-separated word or empty string.',
)
def first_word(text: str) -> str:
    parts = text.split()
    return parts[0] if parts else ''


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Wraps the string in angle brackets.',
)
def wrap_in_angle_brackets(text: str) -> str:
    return '<' + text + '>'


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns whether the string reads the same backward.',
)
def is_palindrome_as_string(text: str) -> str:
    return str(text == text[::-1])


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Repeats the string three times.',
)
def repeat_three_times(text: str) -> str:
    return text * 3


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Trims outer whitespace and converts to uppercase.',
)
def strip_and_uppercase(text: str) -> str:
    return text.strip().upper()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Alternates character case starting with lowercase.',
)
def alternating_case_start_lower(text: str) -> str:
    result = ''
    for i, ch in enumerate(text):
        result += ch.lower() if i % 2 == 0 else ch.upper()
    return result


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Pads the string on the left with dots to length 8.',
)
def left_pad_dots_to_8(text: str) -> str:
    return text.rjust(8, '.')


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Inserts dashes between all characters.',
)
def insert_dashes_between_characters(text: str) -> str:
    return '-'.join(text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Collapses all runs of whitespace to single spaces.',
)
def collapse_whitespace(text: str) -> str:
    return ' '.join(text.split())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes matching surrounding single or double quotes.',
)
def remove_surrounding_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Keeps only ASCII characters.',
)
def keep_ascii_only(text: str) -> str:
    return ''.join(ch for ch in text if ord(ch) < 128)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the smallest character or empty string.',
)
def lexicographically_smallest_char(text: str) -> str:
    return min(text) if text else ''


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Replaces newline characters with pipes.',
)
def replace_newlines_with_pipes(text: str) -> str:
    return text.replace('\n', '|')


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Adds a trailing slash if missing.',
)
def ensure_suffix_slash(text: str) -> str:
    return text if text.endswith('/') else text + '/'


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the shortest whitespace-separated word, preferring the first tie.',
)
def shortest_word(text: str) -> str:
    words = text.split()
    return min(words, key=len) if words else ''


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Centers the string in width 10 using dots.',
)
def center_with_dots_width_10(text: str) -> str:
    return text.center(10, '.')


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns at most the first five characters.',
)
def trim_to_five_characters(text: str) -> str:
    return text[:5]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Reverses the string and converts it to uppercase.',
)
def reverse_and_uppercase(text: str) -> str:
    return text[::-1].upper()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes all whitespace characters.',
)
def remove_whitespace(text: str) -> str:
    return ''.join(ch for ch in text if not ch.isspace())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns space-separated decimal code points for each character.',
)
def decimal_code_points(text: str) -> str:
    return ' '.join(str(ord(ch)) for ch in text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the last character or an empty string.',
)
def last_character_only(text: str) -> str:
    return text[-1:]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Applies aggressive lowercase conversion using casefold.',
)
def casefold_text(text: str) -> str:
    return text.casefold()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the longest prefix with even length.',
)
def take_even_length_prefix(text: str) -> str:
    return text[:len(text) - len(text) % 2]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes all digit characters.',
)
def remove_digits(text: str) -> str:
    return ''.join(ch for ch in text if not ch.isdigit())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes all space characters from the string.',
)
def remove_spaces(text: str) -> str:
    return text.replace(' ', '')


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Inserts commas between all characters.',
)
def comma_separate_characters(text: str) -> str:
    return ','.join(text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Converts all letters in the string to lowercase.',
)
def lowercase_text(text: str) -> str:
    return text.lower()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Alternates character case starting with uppercase.',
)
def alternating_case_start_upper(text: str) -> str:
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
    hint='Applies ROT13 to lowercase letters only.',
)
def rot13_lowercase_only(text: str) -> str:
    result = ''
    for ch in text:
        if 'a' <= ch <= 'z':
            result += chr((ord(ch) - 97 + 13) % 26 + 97)
        else:
            result += ch
    return result


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Repeats each character twice.',
)
def double_each_character(text: str) -> str:
    return ''.join(ch * 2 for ch in text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the number of whitespace-separated words as text.',
)
def count_words_as_string(text: str) -> str:
    return str(len(text.split()))


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Reverses the order of whitespace-separated words.',
)
def reverse_words_order(text: str) -> str:
    return ' '.join(text.split()[::-1])


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Keeps only non-ASCII characters.',
)
def drop_ascii_only(text: str) -> str:
    return ''.join(ch for ch in text if ord(ch) >= 128)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the largest character or empty string.',
)
def lexicographically_largest_char(text: str) -> str:
    return max(text) if text else ''


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Removes trailing whitespace only.',
)
def right_strip_spaces(text: str) -> str:
    return text.rstrip()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Adds a hash character to the start of the string.',
)
def prepend_hash(text: str) -> str:
    return '#' + text


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns up to the last three characters.',
)
def take_last_three(text: str) -> str:
    return text[-3:]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Reverses both word order and letters in each word.',
)
def reverse_word_letters_and_order(text: str) -> str:
    return ' '.join(word[::-1] for word in text.split()[::-1])


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Replaces each vowel with an asterisk.',
)
def replace_vowels_with_star(text: str) -> str:
    return ''.join('*' if ch in 'aeiouAEIOU' else ch for ch in text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns the middle character, or empty for empty text.',
)
def middle_character(text: str) -> str:
    return text[len(text) // 2:len(text) // 2 + 1]


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the negation of b.',
)
def not_b(a: bool, b: bool) -> bool:
    return not b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True unless both inputs are False.',
)
def a_or_b_but_not_both_false(a: bool, b: bool) -> bool:
    return not ((not a) and (not b))


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True unless both inputs are True.',
)
def sum_not_two(a: bool, b: bool) -> bool:
    return int(a) + int(b) != 2


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when the bit-encoded inputs equal one.',
)
def encoded_is_one(a: bool, b: bool) -> bool:
    return ((int(a) << 1) | int(b)) == 1


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when at least one input is False via integer product.',
)
def product_is_zero(a: bool, b: bool) -> bool:
    return int(a) * int(b) == 0


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the negation of the bitwise XOR result.',
)
def bitwise_xnor_bool(a: bool, b: bool) -> bool:
    return not (a ^ b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b matches or exceeds a numerically.',
)
def second_is_majority(a: bool, b: bool) -> bool:
    return int(b) >= int(a)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b is True and a is False.',
)
def b_without_a(a: bool, b: bool) -> bool:
    return b and (not a)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when both inputs have the same truth value.',
)
def same_truth_value(a: bool, b: bool) -> bool:
    return a == b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is True or b is False.',
)
def a_dominates_false_b(a: bool, b: bool) -> bool:
    return True if a else not b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns False when a is True, otherwise returns b.',
)
def false_if_a_else_b(a: bool, b: bool) -> bool:
    if a:
        return False
    return b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a being True requires b to be True.',
)
def a_requires_b(a: bool, b: bool) -> bool:
    if a and not b:
        return False
    return True


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b is True or a is False.',
)
def b_dominates_false_a(a: bool, b: bool) -> bool:
    return True if b else not a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint="Computes conjunction using De Morgan's law.",
)
def a_and_b_via_de_morgan(a: bool, b: bool) -> bool:
    return not ((not a) or (not b))


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when exactly one input is True using list count.',
)
def single_true_via_count(a: bool, b: bool) -> bool:
    return [a, b].count(True) == 1


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the negation of both_true.',
)
def nand_result(a: bool, b: bool) -> bool:
    return not (a and b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the disjunction of the first and second tuple elements.',
)
def tuple_first_or_second(a: bool, b: bool) -> bool:
    values = (a, b)
    return values[0] or values[1]


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is True or b is False.',
)
def a_or_not_b(a: bool, b: bool) -> bool:
    return a or (not b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the negation of a.',
)
def not_a(a: bool, b: bool) -> bool:
    return not a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Flips b only when a is True.',
)
def conditional_flip_on_a(a: bool, b: bool) -> bool:
    if a:
        return not b
    return b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns False when inputs are equal, otherwise True.',
)
def false_when_equal(a: bool, b: bool) -> bool:
    if a == b:
        return False
    return True


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b is True, otherwise returns a.',
)
def true_if_b_else_a(a: bool, b: bool) -> bool:
    if b:
        return True
    return a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a and b are different singleton booleans.',
)
def a_is_not_b(a: bool, b: bool) -> bool:
    return a is not b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when exactly one input is True using branches.',
)
def xor_via_if(a: bool, b: bool) -> bool:
    if a:
        return not b
    return b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns a unchanged.',
)
def return_a(a: bool, b: bool) -> bool:
    return a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a matches at least one of the inputs.',
)
def a_matches_any(a: bool, b: bool) -> bool:
    return a in (a, b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when the integer forms of the inputs are equal.',
)
def difference_is_zero(a: bool, b: bool) -> bool:
    return int(a) - int(b) == 0


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when True values are at least as many as False values.',
)
def at_least_as_many_true(a: bool, b: bool) -> bool:
    values = [a, b]
    return values.count(True) >= values.count(False)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns the negation of either_true.',
)
def nor_result(a: bool, b: bool) -> bool:
    return not (a or b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when both inputs are False by counting them.',
)
def sum_equals_zero(a: bool, b: bool) -> bool:
    return a + b == 0


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is True and b is False.',
)
def a_without_b(a: bool, b: bool) -> bool:
    return a and (not b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Always returns True.',
)
def always_true(a: bool, b: bool) -> bool:
    return True


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when the inputs differ using inequality.',
)
def xor_via_inequality(a: bool, b: bool) -> bool:
    return a != b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is False and b is True by numeric comparison.',
)
def first_less_than_second(a: bool, b: bool) -> bool:
    return a < b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when at least one input is True but not both are required.',
)
def any_not_both(a: bool, b: bool) -> bool:
    return any((a, b)) and not all((a, b))


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b implies a.',
)
def reverse_implication(a: bool, b: bool) -> bool:
    return (not b) or a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a matches or exceeds b numerically.',
)
def first_is_majority(a: bool, b: bool) -> bool:
    return int(a) >= int(b)


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when the bit-encoded inputs equal two.',
)
def encoded_is_two(a: bool, b: bool) -> bool:
    return ((int(a) << 1) | int(b)) == 2


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Uses a to choose between the negation of b and b.',
)
def choose_by_not_a(a: bool, b: bool) -> bool:
    if a:
        return not b
    return b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True only when both inputs are True using nested conditionals.',
)
def a_and_b_via_if(a: bool, b: bool) -> bool:
    if a:
        if b:
            return True
    return False


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when b being True requires a to be True.',
)
def b_requires_a(a: bool, b: bool) -> bool:
    if b and not a:
        return False
    return True


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a weighted sum using a left shift is positive.',
)
def left_shift_sum_positive(a: bool, b: bool) -> bool:
    return ((int(a) << 1) + int(b)) > 0


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a and b are the same singleton boolean.',
)
def a_is_b(a: bool, b: bool) -> bool:
    return a is b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns False when b is True, otherwise returns a.',
)
def false_if_b_else_a(a: bool, b: bool) -> bool:
    if b:
        return False
    return a


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is False and b is True.',
)
def not_a_and_b(a: bool, b: bool) -> bool:
    return (not a) and b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is True, otherwise returns b.',
)
def true_if_a_else_b(a: bool, b: bool) -> bool:
    if a:
        return True
    return b


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a equals the negation of b.',
)
def a_equals_not_b(a: bool, b: bool) -> bool:
    return a == (not b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the larger and smaller absolute values.',
)
def sum_of_max_and_min_abs(a: int, b: int) -> int:
    return max(abs(a), abs(b)) + min(abs(a), abs(b))


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the integer that is farthest from zero, preferring the larger one on ties.',
)
def farthest_from_zero(a: int, b: int) -> int:
    if abs(a) > abs(b):
        return a
    if abs(b) > abs(a):
        return b
    return a if a > b else b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the first integer if it is even, otherwise the second.',
)
def first_if_even_else_second(a: int, b: int) -> int:
    return a if a % 2 == 0 else b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the remainder when the difference is divided by four.',
)
def modulo_difference_by_four(a: int, b: int) -> int:
    return (a - b) % 4


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the two integers are equal, otherwise 0.',
)
def is_equal_flag(a: int, b: int) -> int:
    return 1 if a == b else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns twice the larger of the two integers.',
)
def larger_times_two(a: int, b: int) -> int:
    return 2 * (a if a > b else b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the first integer plus the absolute value of the second.',
)
def sum_with_second_abs(a: int, b: int) -> int:
    return a + abs(b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of inputs that are negative.',
)
def sum_if_negative(a: int, b: int) -> int:
    return (a if a < 0 else 0) + (b if b < 0 else 0)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the sum is even, otherwise 0.',
)
def sum_divisible_by_two_flag(a: int, b: int) -> int:
    return 1 if (a + b) % 2 == 0 else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the product is divisible by five, otherwise 0.',
)
def product_divisible_by_five_flag(a: int, b: int) -> int:
    return 1 if (a * b) % 5 == 0 else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the product plus the smaller integer.',
)
def product_plus_min(a: int, b: int) -> int:
    return a * b + (a if a < b else b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the first integer is greater than the second, otherwise 0.',
)
def is_first_greater_flag(a: int, b: int) -> int:
    return 1 if a > b else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum plus the smaller integer.',
)
def sum_plus_min(a: int, b: int) -> int:
    return a + b + (a if a < b else b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the digit-reversed absolute values.',
)
def reverse_digits_sum(a: int, b: int) -> int:
    return int(str(abs(a))[::-1]) + int(str(abs(b))[::-1])


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the product plus the absolute difference.',
)
def product_plus_abs_difference(a: int, b: int) -> int:
    return a * b + abs(a - b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of each integer reduced modulo 5.',
)
def sum_mod_five(a: int, b: int) -> int:
    return (a % 5) + (b % 5)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the next integers after each input.',
)
def sum_of_successors(a: int, b: int) -> int:
    return (a + 1) + (b + 1)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the product is negative, otherwise 0.',
)
def ones_if_product_negative(a: int, b: int) -> int:
    return 1 if a * b < 0 else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the Manhattan distance from the origin to the point (a, b).',
)
def manhattan_to_pair(a: int, b: int) -> int:
    return abs(a) + abs(b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute value of the first integer plus the second.',
)
def sum_with_first_abs(a: int, b: int) -> int:
    return abs(a) + b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if both integers have the same parity, otherwise 0.',
)
def parity_match_flag(a: int, b: int) -> int:
    return 1 if a % 2 == b % 2 else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the first integer minus the second.',
)
def difference_ab(a: int, b: int) -> int:
    return a - b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum minus the smaller absolute value.',
)
def sum_minus_smaller_abs(a: int, b: int) -> int:
    return a + b - min(abs(a), abs(b))


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the smaller integer plus the absolute value of the larger.',
)
def min_plus_abs_max(a: int, b: int) -> int:
    m = a if a < b else b
    x = a if a > b else b
    return m + abs(x)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the second integer if it is odd, otherwise the first.',
)
def second_if_odd_else_first(a: int, b: int) -> int:
    return b if b % 2 != 0 else a


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of each integer floor-divided by two.',
)
def sum_of_halves_floor(a: int, b: int) -> int:
    return a // 2 + b // 2


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the first decimal digits ignoring signs.',
)
def first_digit_sum(a: int, b: int) -> int:
    return int(str(abs(a))[0]) + int(str(abs(b))[0])


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum if the integers share parity, otherwise their difference.',
)
def common_parity_value(a: int, b: int) -> int:
    return a + b if a % 2 == b % 2 else a - b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute product plus one.',
)
def abs_product_plus_one(a: int, b: int) -> int:
    return abs(a * b) + 1


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if the absolute values have the same number of digits, otherwise 0.',
)
def count_digits_equal(a: int, b: int) -> int:
    return 1 if len(str(abs(a))) == len(str(abs(b))) else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns 1 if both integers have the same sign or either is zero, otherwise 0.',
)
def sign_agreement_flag(a: int, b: int) -> int:
    return 1 if a * b >= 0 else 0


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the remainder when the sum is divided by three.',
)
def modulo_sum_by_three(a: int, b: int) -> int:
    return (a + b) % 3


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the first integer times the absolute value of the second.',
)
def product_with_second_abs(a: int, b: int) -> int:
    return a * abs(b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns how many steps separate the two integers on the number line.',
)
def taxicab_gap_to_equal(a: int, b: int) -> int:
    return abs(a - b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum plus the product of the integers.',
)
def sum_plus_product(a: int, b: int) -> int:
    return a + b + a * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the product of the last decimal digits ignoring signs.',
)
def digit_product_last_digits(a: int, b: int) -> int:
    return (abs(a) % 10) * (abs(b) % 10)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the smaller of the two integers.',
)
def min_of_two(a: int, b: int) -> int:
    return a if a < b else b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the square of the sum of the integers.',
)
def sum_squared(a: int, b: int) -> int:
    s = a + b
    return s * s


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the arithmetic mean rounded down.',
)
def average_rounded_down(a: int, b: int) -> int:
    return (a + b) // 2


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the larger absolute value of the two integers.',
)
def larger_absolute_value(a: int, b: int) -> int:
    return abs(a) if abs(a) > abs(b) else abs(b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of distances from 10 for both integers.',
)
def distance_from_ten_sum(a: int, b: int) -> int:
    return abs(a - 10) + abs(b - 10)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum after increasing the first integer by one.',
)
def sum_after_incrementing_first(a: int, b: int) -> int:
    return (a + 1) + b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute sum plus the absolute difference.',
)
def absolute_sum_plus_absolute_difference(a: int, b: int) -> int:
    return abs(a + b) + abs(a - b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns a modulo one more than the absolute value of b.',
)
def remainder_a_by_abs_b_plus_one(a: int, b: int) -> int:
    return a % (abs(b) + 1)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the difference in digit counts of the absolute values.',
)
def difference_in_digit_lengths(a: int, b: int) -> int:
    return len(str(abs(a))) - len(str(abs(b)))


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the previous integers before each input.',
)
def sum_of_predecessors(a: int, b: int) -> int:
    return (a - 1) + (b - 1)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the signs of the two integers.',
)
def sum_of_signs(a: int, b: int) -> int:
    sa = 1 if a > 0 else -1 if a < 0 else 0
    sb = 1 if b > 0 else -1 if b < 0 else 0
    return sa + sb


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns how many of the two integers are even.',
)
def even_count(a: int, b: int) -> int:
    return (1 if a % 2 == 0 else 0) + (1 if b % 2 == 0 else 0)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum multiplied by the sign of the first integer.',
)
def sum_times_sign_of_a(a: int, b: int) -> int:
    return (a + b) * (1 if a > 0 else -1 if a < 0 else 0)


# [generated]
@register(
    category=Category.NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_int_to_int',
    hint='Returns the Fibonacci number at position abs(x) modulo twenty.',
)
def fibonacci_index_small(x: int) -> int:
    n = abs(x) % 20
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all elements except one maximum value.',
)
def sum_excluding_max(items: List[int]) -> int:
    return sum(items) - max(items) if items else 0


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of each value modulo three.',
)
def sum_mod_three(items: List[int]) -> int:
    return sum(x % 3 for x in items)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of adjacent pairs that are exact opposites.',
)
def count_adjacent_opposite_pairs(items: List[int]) -> int:
    return sum(1 for i in range(len(items) - 1) if items[i] == -items[i + 1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of the smaller value from each adjacent pair.',
)
def sum_smaller_of_neighbors(items: List[int]) -> int:
    return sum(items[i] if items[i] < items[i + 1] else items[i + 1] for i in range(len(items) - 1))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the number of distinct integers in the list.',
)
def count_distinct_values(items: List[int]) -> int:
    return len(set(items))


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of squares of positive values.',
)
def sum_of_positive_squares(items: List[int]) -> int:
    return sum(x * x for x in items if x > 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of all negative integers.',
)
def sum_negative_values(items: List[int]) -> int:
    return sum(x for x in items if x < 0)


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns the sum of values whose absolute decimal form is a palindrome.',
)
def sum_palindrome_values(items: List[int]) -> int:
    return sum(x for x in items if str(abs(x)) == str(abs(x))[::-1])


# [generated]
@register(
    category=Category.LIST,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_list_to_int',
    hint='Returns how many elements are odd.',
)
def count_odd_values(items: List[int]) -> int:
    return sum(1 for x in items if x % 2 != 0)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Keeps only whitespace characters.',
)
def keep_whitespace_only(text: str) -> str:
    return ''.join(ch for ch in text if ch.isspace())


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Appends a pipe and the reversed string.',
)
def mirror_with_pipe(text: str) -> str:
    return text + '|' + text[::-1]


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Trims outer whitespace and converts to lowercase.',
)
def strip_and_lowercase(text: str) -> str:
    return text.strip().lower()


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Returns space-separated hexadecimal code points for each character.',
)
def hex_code_points(text: str) -> str:
    return ' '.join(format(ord(ch), 'x') for ch in text)


# [generated]
@register(
    category=Category.STRING,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_str_to_str',
    hint='Keeps only alphabetic characters.',
)
def letters_only(text: str) -> str:
    return ''.join(ch for ch in text if ch.isalpha())


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when both inputs are exactly True.',
)
def both_are_true_identity(a: bool, b: bool) -> bool:
    return a is True and b is True


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when any value in the tuple of inputs is True.',
)
def bool_from_any_tuple(a: bool, b: bool) -> bool:
    return any((a, b))


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when either input is True using conditionals.',
)
def a_or_b_via_if(a: bool, b: bool) -> bool:
    if a:
        return True
    if b:
        return True
    return False


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when a is False and b is True via integer difference.',
)
def difference_negative(a: bool, b: bool) -> bool:
    return int(a) - int(b) < 0


# [generated]
@register(
    category=Category.LOGIC,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_bools_to_bool',
    hint='Returns True when both inputs match, either all True or all False.',
)
def all_or_none(a: bool, b: bool) -> bool:
    return all((a, b)) or not any((a, b))


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum of the last decimal digits ignoring signs.',
)
def last_digit_sum(a: int, b: int) -> int:
    return abs(a) % 10 + abs(b) % 10


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns zero when the integers are opposites, otherwise their product.',
)
def zero_if_opposite_else_product(a: int, b: int) -> int:
    return 0 if a == -b else a * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the absolute sum minus the absolute difference.',
)
def absolute_sum_minus_absolute_difference(a: int, b: int) -> int:
    return abs(a + b) - abs(a - b)


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the square of the difference a - b.',
)
def difference_squared(a: int, b: int) -> int:
    d = a - b
    return d * d


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the sum plus the larger absolute value.',
)
def sum_plus_larger_abs(a: int, b: int) -> int:
    return a + b + max(abs(a), abs(b))


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns three times the sum minus the product.',
)
def triple_sum_minus_product(a: int, b: int) -> int:
    return 3 * (a + b) - a * b


# [generated]
@register(
    category=Category.TWO_NUMBERS,
    difficulty=Difficulty.MEDIUM,
    confusers=[],
    family_id='generated_two_ints_to_int',
    hint='Returns the product reduced modulo 7.',
)
def product_mod_seven(a: int, b: int) -> int:
    return (a * b) % 7
