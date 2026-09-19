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
from typing import Optional

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

MALFORMED_PENALTY = -0.1
CORRECT_REWARD = 1.0
INCORRECT_REWARD = 0.0


@dataclass(frozen=True)
class Puzzle:
    numbers: tuple[int, ...]
    target: int
    solution: Optional[str] = None  # one known-valid expression, never shown to the model

    def to_dict(self) -> dict:
        return {"numbers": list(self.numbers), "target": self.target, "solution": self.solution}

    @staticmethod
    def from_dict(d: dict) -> "Puzzle":
        return Puzzle(tuple(int(x) for x in d["numbers"]), int(d["target"]), d.get("solution"))


@dataclass
class VerifyResult:
    reward: float
    correct: bool
    malformed: bool
    reason: str
    expression: Optional[str] = None
    value: Optional[Fraction] = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _random_expression(rng: random.Random, leaves: list[int]) -> tuple[str, Fraction]:
    """Build a random binary tree over `leaves`, keeping every intermediate a
    positive integer so the puzzle is guaranteed solvable with clean arithmetic."""
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
) -> Puzzle:
    """Sample a solvable puzzle. Difficulty is tuned via `n_numbers`,
    `number_range` and `target_range`."""
    for _ in range(max_tries):
        nums = [rng.randint(*number_range) for _ in range(n_numbers)]
        expr, val = _random_expression(rng, nums)
        if val.denominator == 1 and target_range[0] <= val <= target_range[1]:
            target = int(val)
            if target in nums and n_numbers > 1:
                continue  # too trivial: target is one of the inputs
            return Puzzle(tuple(nums), target, expr[1:-1] if expr.startswith("(") else expr)
    raise RuntimeError("failed to generate a puzzle within max_tries")


def generate_puzzles(seed: int, n: int, **kwargs) -> list[Puzzle]:
    rng = random.Random(seed)
    return [generate_puzzle(rng, **kwargs) for _ in range(n)]


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You may use +, -, *, / and each number exactly once. "
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


def format_prompt(puzzle: Puzzle, few_shot: int = 0) -> str:
    """Format a puzzle, optionally preceded by up to three worked examples."""
    if few_shot <= 0:
        return PROMPT_TEMPLATE.format(numbers=list(puzzle.numbers), target=puzzle.target)
    examples = []
    for example_puzzle, reasoning, answer in FEW_SHOT_EXAMPLES[: min(few_shot, len(FEW_SHOT_EXAMPLES))]:
        question = PROMPT_TEMPLATE.format(numbers=list(example_puzzle.numbers), target=example_puzzle.target)
        examples.append(f"Question: {question}Reasoning: {reasoning}\n{answer}")
    question = PROMPT_TEMPLATE.format(numbers=list(puzzle.numbers), target=puzzle.target)
    return "\n\n".join(examples + [f"Question: {question}"])


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class _Unsafe(ValueError):
    pass


def _eval_node(node: ast.AST, leaves: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, leaves)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _Unsafe("only integer literals allowed")
        leaves.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, leaves)
        right = _eval_node(node.right, leaves)
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


def safe_eval(expression: str) -> tuple[Fraction, list[int]]:
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
    value = _eval_node(tree, leaves)
    return value, leaves


def extract_expression(response: str) -> Optional[str]:
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
    try:
        value, leaves = safe_eval(expr)
    except ValueError as e:
        return VerifyResult(MALFORMED_PENALTY, False, True, str(e), expression=expr)
    if sorted(leaves) != sorted(puzzle.numbers):
        return VerifyResult(
            INCORRECT_REWARD, False, False, f"numbers used {sorted(leaves)} != {sorted(puzzle.numbers)}", expr, value
        )
    if value != puzzle.target:
        return VerifyResult(INCORRECT_REWARD, False, False, f"evaluates to {value}, target {puzzle.target}", expr, value)
    return VerifyResult(CORRECT_REWARD, True, False, "correct", expr, value)


def reward(response: str, puzzle: Puzzle) -> float:
    return verify(response, puzzle).reward
