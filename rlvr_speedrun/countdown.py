"""Countdown arithmetic puzzles: generator, prompt format, and verifier.

A puzzle gives `n` numbers and a target. A solution is an arithmetic
expression using each number exactly once with + - * / that evaluates to
the target. The verifier is pure CPU, deterministic, and sub-millisecond.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

MALFORMED_PENALTY = -0.1
CORRECT_REWARD = 1.0
INCORRECT_REWARD = 0.0
ALL_OPS = "+-*/"


def ops_text(ops: str) -> str:
    """How the prompt lists the allowed operators ("+, -, *, /" for the default set)."""
    return ", ".join(ops)


@dataclass(frozen=True)
class Puzzle:
    numbers: tuple[int, ...]
    target: int
    solution: str | None = None  # one known-valid expression, never shown to the model
    ops: str = ALL_OPS  # operators the answer may use; stated in the prompt and enforced by `verify`
    level: str | None = field(default=None, compare=False)  # difficulty label for reporting (e.g. ladder level)

    def to_dict(self) -> dict:
        d = {"numbers": list(self.numbers), "target": self.target, "solution": self.solution}
        if self.ops != ALL_OPS:
            d["ops"] = self.ops
        if self.level is not None:
            d["level"] = self.level
        return d

    @staticmethod
    def from_dict(d: dict) -> Puzzle:
        return Puzzle(tuple(int(x) for x in d["numbers"]), int(d["target"]), d.get("solution"), d.get("ops", ALL_OPS), d.get("level"))


@dataclass
class VerifyResult:
    reward: float
    correct: bool
    malformed: bool
    reason: str
    expression: str | None = None
    value: Fraction | None = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _random_expression(rng: random.Random, leaves: list[int], allowed: str = ALL_OPS) -> tuple[str, Fraction] | None:
    """Build a random binary tree over `leaves`, keeping every intermediate a
    positive integer so the puzzle is guaranteed solvable with clean arithmetic.
    Only operators in `allowed` are used; returns None if no allowed operator fits a merge."""
    items: list[tuple[str, Fraction]] = [(str(x), Fraction(x)) for x in leaves]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        (sa, va), (sb, vb) = items[i], items[j]
        ops = ["+", "*"]
        if va != vb:
            ops.append("-")
        hi, lo = (va, vb) if va >= vb else (vb, va)
        if lo != 0 and hi % lo == 0 and lo != 1:
            ops.append("/")
        if allowed != ALL_OPS:  # the default path draws exactly as before, so v1/v2 data is unchanged
            ops = [o for o in ops if o in allowed]
            if not ops:
                return None
        op = rng.choice(ops)
        if op == "+":
            s, v = f"({sa} + {sb})", va + vb
        elif op == "*":
            s, v = f"({sa} * {sb})", va * vb
        elif op == "-":
            s, v = (f"({sa} - {sb})", va - vb) if va > vb else (f"({sb} - {sa})", vb - va)
        else:
            s, v = (f"({sa} / {sb})", va / vb) if va >= vb else (f"({sb} / {sa})", vb / va)
        for k in sorted((i, j), reverse=True):
            items.pop(k)
        items.append((s, v))
    return items[0]


def generate_puzzle(
    rng: random.Random,
    n_numbers: int = 4,
    number_range: tuple[int, int] = (1, 25),
    target_range: tuple[int, int] = (10, 200),
    max_tries: int = 1000,
    ops: str = ALL_OPS,
) -> Puzzle:
    """Sample a solvable puzzle. Difficulty is tuned via `n_numbers`,
    `number_range`, `target_range` and the allowed operators `ops`."""
    for _ in range(max_tries):
        nums = [rng.randint(*number_range) for _ in range(n_numbers)]
        built = _random_expression(rng, nums, ops)
        if built is None:
            continue
        expr, val = built
        if val.denominator == 1 and target_range[0] <= val <= target_range[1]:
            target = int(val)
            if target in nums and n_numbers > 1:
                continue  # too trivial: target is one of the inputs
            return Puzzle(tuple(nums), target, expr[1:-1] if expr.startswith("(") else expr, ops)
    raise RuntimeError("failed to generate a puzzle within max_tries")


def generate_puzzles(seed: int, n: int, **kwargs) -> list[Puzzle]:
    rng = random.Random(seed)
    return [generate_puzzle(rng, **kwargs) for _ in range(n)]


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You may use {ops} and each number exactly once. "
    "Think briefly, then give the final expression inside <answer></answer> tags, "
    "for example <answer>(1 + 2) * 3</answer>.\n"
)


FEW_SHOT_EXAMPLES = [
    (
        Puzzle((1, 2, 3), 6),
        "Add the three numbers: 1 + 2 + 3 = 6.",
        "<answer>1 + 2 + 3</answer>",
    ),
    (
        Puzzle((2, 4, 5, 7), 20),
        "Multiply 4 by 2, then add 7 and 5: 4 * 2 + 7 + 5 = 20.",
        "<answer>7 + 5 + 4 * 2</answer>",
    ),
    (
        Puzzle((3, 4, 6, 8), 13),
        "Add 6, 8, and 3, then subtract 4: 6 + 8 + 3 - 4 = 13.",
        "<answer>6 + 8 + 3 - 4</answer>",
    ),
]


# Zero-shot "think, then answer" prompt in the style of TinyZero / DeepSeek-R1-Zero: the completion
# starts inside <think>, so the model may reason at length before <answer>. Candidate for the next
# benchmark version; "v1" (above, with few-shot examples) is the current eval prompt.
R1_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process step by step and then gives the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. "
    "You may use {ops} and each number exactly once. Show your work in <think> </think> tags, "
    "then give the final expression inside <answer></answer> tags, for example <answer>(1 + 2) * 3</answer>.\n"
    "Assistant: Let me solve this step by step.\n<think>"
)
# "search": the r1 format preceded by three worked examples whose reasoning is trial and error
# (propose, compute, reject, retry). The base model then emits well-formed answers and gets some
# right, so GRPO groups have reward variance from the first step. With the zero-shot r1 prompt,
# only about 1 % of 0.5B groups do. Every example's arithmetic is checked in tests/test_countdown.py.
SEARCH_EXAMPLES = [
    (
        Puzzle((3, 7, 8, 2), 26),
        "Try 8 * 3 = 24, then 24 + 7 - 2 = 29. Not 26. Try 7 * 3 = 21, then 21 + 8 - 2 = 27. Close, but not 26. "
        "Try 8 * 2 = 16, then 16 + 7 + 3 = 26. That works and uses every number once.",
        "8 * 2 + 7 + 3",
    ),
    (
        Puzzle((4, 9, 6, 5), 21),
        "The sum is 4 + 9 + 6 + 5 = 24, which is 3 too many. Try 9 * 5 = 45, then remove 24: 6 * 4 = 24, "
        "and 45 - 24 = 21. That works.",
        "9 * 5 - 6 * 4",
    ),
    (
        Puzzle((2, 10, 5, 3), 8),
        "The sum is 2 + 10 + 5 + 3 = 20, too big. Try 10 - 5 + 3 = 8, but then 2 is unused, and (10 - 5 + 3) * 2 = 16. "
        "Not 8. Try 10 / 5 = 2, then 2 * 3 = 6, and 6 + 2 = 8. That works.",
        "10 / 5 * 3 + 2",
    ),
]
PROMPT_STYLES = ("v1", "r1", "search")

_PREFIX_SPLIT = "Using the numbers"


def split_prompt(puzzle: Puzzle, few_shot: int = 0, style: str = "v1") -> tuple[str, str]:
    """Split the prompt into (shared prefix, puzzle-specific suffix).

    The prefix is identical for every puzzle at a given `few_shot`, and the split
    falls on a whitespace boundary so BPE tokenization of the parts concatenates to
    the tokenization of the whole prompt. `"".join(split_prompt(p, k)) == format_prompt(p, k)`.
    `style="r1"` ignores `few_shot`.
    """
    if style in ("r1", "search"):
        head, _, tail = R1_TEMPLATE.format(numbers=list(puzzle.numbers), target=puzzle.target, ops=ops_text(puzzle.ops)).partition(_PREFIX_SPLIT)
        if style == "search":
            header, _, user = head.partition("User: ")
            shots = "".join(
                "User: " + R1_TEMPLATE.format(numbers=list(p.numbers), target=p.target, ops=ops_text(p.ops)).partition("User: ")[2]
                + f" {reasoning}</think>\n<answer>{answer}</answer>\n\n"
                for p, reasoning, answer in SEARCH_EXAMPLES
            )
            head = header + shots + "User: " + user
        return head + _PREFIX_SPLIT, tail
    if style != "v1":
        raise ValueError(f"unknown prompt style {style!r} (expected one of {PROMPT_STYLES})")
    question = PROMPT_TEMPLATE.format(numbers=list(puzzle.numbers), target=puzzle.target, ops=ops_text(puzzle.ops))
    head, _, tail = question.partition(_PREFIX_SPLIT)
    if few_shot <= 0:
        return head + _PREFIX_SPLIT, tail
    examples = []
    for example_puzzle, reasoning, answer in FEW_SHOT_EXAMPLES[: min(few_shot, len(FEW_SHOT_EXAMPLES))]:
        example = PROMPT_TEMPLATE.format(numbers=list(example_puzzle.numbers), target=example_puzzle.target, ops=ops_text(example_puzzle.ops))
        examples.append(f"Question: {example}Reasoning: {reasoning}\n{answer}")
    return "\n\n".join(examples + [f"Question: {head}{_PREFIX_SPLIT}"]), tail


def format_prompt(puzzle: Puzzle, few_shot: int = 0, style: str = "v1") -> str:
    """Format a puzzle, optionally preceded by up to three worked examples."""
    return "".join(split_prompt(puzzle, few_shot, style))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class _Unsafe(ValueError):
    pass


_OP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _eval_node(node: ast.AST, leaves: list[int], ops_used: set[str] | None = None) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, leaves, ops_used)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _Unsafe("only integer literals allowed")
        leaves.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, leaves, ops_used)
        right = _eval_node(node.right, leaves, ops_used)
        if ops_used is not None and type(node.op) in _OP_SYMBOL:
            ops_used.add(_OP_SYMBOL[type(node.op)])
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _Unsafe("division by zero")
            return left / right
        raise _Unsafe(f"operator not allowed: {type(node.op).__name__}")
    raise _Unsafe(f"syntax not allowed: {type(node).__name__}")


def safe_eval(expression: str, ops_used: set[str] | None = None) -> tuple[Fraction, list[int]]:
    """Evaluate an arithmetic expression over integer literals with + - * /.
    Returns (value, leaf numbers in order). Raises ValueError on anything else."""
    expression = expression.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    if len(expression) > 500:
        raise _Unsafe("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise _Unsafe(f"syntax error: {e.msg}") from None
    leaves: list[int] = []
    value = _eval_node(tree, leaves, ops_used)
    return value, leaves


def extract_expression(response: str) -> str | None:
    """Pull the candidate expression out of a model response. Prefers the
    last <answer> block; falls back to the last non-empty line, with an
    optional trailing '= N' stripped."""
    matches = ANSWER_RE.findall(response)
    if matches:
        cand = matches[-1].strip()
    else:
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        cand = lines[-1]
    cand = cand.split("=")[0].strip()
    return cand or None


def verify(response: str, puzzle: Puzzle) -> VerifyResult:
    """Binary reward with a small malformed-output penalty.

    correct   -> +1.0
    parseable but wrong (wrong value or wrong number usage) -> 0.0
    unparseable / not a valid arithmetic expression -> -0.1
    """
    expr = extract_expression(response)
    if expr is None:
        return VerifyResult(MALFORMED_PENALTY, False, True, "no expression found")
    ops_used: set[str] = set()
    try:
        value, leaves = safe_eval(expr, ops_used)
    except ValueError as e:
        return VerifyResult(MALFORMED_PENALTY, False, True, str(e), expression=expr)
    if not ops_used <= set(puzzle.ops):
        return VerifyResult(INCORRECT_REWARD, False, False, f"operators {sorted(ops_used - set(puzzle.ops))} not allowed", expr, value)
    if sorted(leaves) != sorted(puzzle.numbers):
        return VerifyResult(
            INCORRECT_REWARD, False, False, f"numbers used {sorted(leaves)} != {sorted(puzzle.numbers)}", expr, value
        )
    if value != puzzle.target:
        return VerifyResult(INCORRECT_REWARD, False, False, f"evaluates to {value}, target {puzzle.target}", expr, value)
    return VerifyResult(CORRECT_REWARD, True, False, "correct", expr, value)


def reward(response: str, puzzle: Puzzle) -> float:
    return verify(response, puzzle).reward


LEVELS = ("sum", "+-", "+-*", "+-*/")


@lru_cache(maxsize=1 << 16)  # bounded: a training stream filtered by level sees an endless supply of number sets
def reachable(nums: tuple[Fraction, ...], ops: str) -> frozenset[Fraction]:
    """Every value an expression using each of `nums` exactly once and only `ops` can take."""
    if len(nums) == 1:
        return frozenset(nums)
    out = set()
    n = len(nums)
    for mask in range(1, (1 << n) - 1):
        left = tuple(sorted(nums[i] for i in range(n) if mask >> i & 1))
        right = tuple(sorted(nums[i] for i in range(n) if not mask >> i & 1))
        for x in reachable(left, ops):
            for y in reachable(right, ops):
                if "+" in ops:
                    out.add(x + y)
                if "-" in ops:
                    out.add(x - y)
                if "*" in ops:
                    out.add(x * y)
                if "/" in ops and y != 0:
                    out.add(x / y)
    return frozenset(out)


def level(puzzle: Puzzle) -> str:
    """Difficulty: "sum" if the target is the sum of all numbers, else the smallest of "+-", "+-*", "+-*/" that reaches it."""
    nums = tuple(sorted(Fraction(x) for x in puzzle.numbers))
    if sum(nums) == puzzle.target:
        return "sum"
    return next(ops for ops in LEVELS[1:] if Fraction(puzzle.target) in reachable(nums, ops))
