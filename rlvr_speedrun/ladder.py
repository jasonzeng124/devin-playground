"""Difficulty ladder for benchmark v3 (draft): Countdown levels scored with equal weight.

Each level fixes how many numbers a puzzle has, which operators it allows (stated in the prompt,
enforced by the verifier) and what the target must need, so that an easy trick cannot carry the
score. With the generator's natural mix, 40 % of 4-number puzzles are solvable with + and -
alone and 7 % are the plain sum (docs/v3_design.md).

    L1  4 numbers, + -         not the plain sum
    L2  4 numbers, + - *       needs *
    L3  4 numbers, + - * /     needs /
    L4  5 numbers, + - * /     needs * or /
    L5  6 numbers, + - * /     needs * or /, targets 100-999
    L6  4 numbers, one of {+ *}, {- *}, {+ /}, {- /}; both operators needed

Puzzles are built from a random expression over the allowed operators, so every puzzle is
solvable. "Needs" is checked by brute force (`countdown.reachable`) for 4 numbers and by the
cheap +/- closed form for 5-6.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .countdown import Puzzle, generate_puzzle, reachable

REPO_ROOT = Path(__file__).resolve().parents[1]
# dev: for in-loop evals and recipe tuning (it was the draft eval set of every v3 pilot).
# test: scored once per seed by `python -m rlvr_speedrun.ladder_confirm`; never used during training.
LADDER_DEV_PATH = "data/ladder_v3_dev.jsonl"
LADDER_TEST_PATH = "data/ladder_v3_test.jsonl"
LADDER_DEV_SEED = 30_000_000  # far from the v1/v2 generation seeds and train_puzzle_stream's 1,000,000 + seed
LADDER_TEST_SEED = 31_000_000
LADDER_EVAL_PATH = LADDER_DEV_PATH
PER_LEVEL = 400


@dataclass(frozen=True)
class Level:
    name: str
    n_numbers: int
    ops: tuple[str, ...]  # one is picked per puzzle
    needs: str  # "" or operators of which at least one must be needed ("*", "/", "*/"); "all" = every allowed op
    target_range: tuple[int, int] = (10, 200)


LEVELS = (
    Level("L1", 4, ("+-",), ""),
    Level("L2", 4, ("+-*",), "*"),
    Level("L3", 4, ("+-*/",), "/"),
    Level("L4", 5, ("+-*/",), "*/"),
    Level("L5", 6, ("+-*/",), "*/", (100, 999)),
    Level("L6", 4, ("+*", "-*", "+/", "-/"), "all"),
)
LEVEL_NAMES = tuple(level.name for level in LEVELS)


def _plus_minus_values(numbers: tuple[int, ...]) -> set[int]:
    """Every value an expression over all `numbers` with only + and - can take: any sign pattern except all minus."""
    return {sum(s * x for s, x in zip(signs, numbers)) for signs in itertools.product((1, -1), repeat=len(numbers)) if 1 in signs}


def _meets_needs(puzzle: Puzzle, level: Level) -> bool:
    nums = tuple(sorted(Fraction(x) for x in puzzle.numbers))
    target = Fraction(puzzle.target)
    if level.needs == "":
        return sum(puzzle.numbers) != puzzle.target
    if level.needs == "all":  # no proper subset of the allowed operators reaches the target
        return all(target not in reachable(nums, "".join(sub)) for k in range(1, len(puzzle.ops))
                   for sub in itertools.combinations(puzzle.ops, k))
    if level.n_numbers > 4:  # full brute force is too slow; + and - alone must not reach the target
        return puzzle.target not in _plus_minus_values(puzzle.numbers)
    without = "".join(op for op in puzzle.ops if op not in level.needs)
    return target not in reachable(nums, without)


def generate_level_puzzle(rng: random.Random, level: Level, max_tries: int = 10_000) -> Puzzle:
    ops = rng.choice(level.ops)  # chosen once, so operator sets that rarely qualify keep their share
    for _ in range(max_tries):
        puzzle = generate_puzzle(rng, n_numbers=level.n_numbers, target_range=level.target_range, ops=ops)
        if _meets_needs(puzzle, level):
            return Puzzle(puzzle.numbers, puzzle.target, puzzle.solution, puzzle.ops, level.name)
    raise RuntimeError(f"no puzzle for {level.name} within {max_tries} tries")


def level_by_name(name: str) -> Level:
    return next(level for level in LEVELS if level.name == name)


def ladder_stream(seed: int, names: tuple[str, ...] = LEVEL_NAMES, exclude: frozenset = frozenset()):
    """Unbounded training stream, uniform over `names`, skipping (numbers, target, ops) keys in `exclude`."""
    rng = random.Random(1_000_000 + seed)
    levels = [level_by_name(name) for name in names]
    while True:
        puzzle = generate_level_puzzle(rng, rng.choice(levels))
        if puzzle_key(puzzle) not in exclude:
            yield puzzle


def puzzle_key(puzzle: Puzzle) -> tuple:
    return tuple(sorted(puzzle.numbers)), puzzle.target, puzzle.ops


def build_ladder_eval(per_level: int = PER_LEVEL, seed: int = LADDER_DEV_SEED, exclude: frozenset = frozenset()) -> list[Puzzle]:
    puzzles, seen = [], set(exclude)
    for i, level in enumerate(LEVELS):
        rng = random.Random(seed + i)
        count = 0
        while count < per_level:
            puzzle = generate_level_puzzle(rng, level)
            if puzzle_key(puzzle) not in seen:
                seen.add(puzzle_key(puzzle))
                puzzles.append(puzzle)
                count += 1
    return puzzles


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def load_ladder(path: str | Path = LADDER_TEST_PATH) -> list[Puzzle]:
    return [Puzzle.from_dict(json.loads(line)) for line in _resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ladder_keys(paths=(LADDER_DEV_PATH, LADDER_TEST_PATH)) -> frozenset:
    """(numbers, target, ops) keys of the committed ladder sets; training streams skip them."""
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    return frozenset(puzzle_key(p) for path in paths if _resolve(path).exists() for p in load_ladder(path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write a frozen ladder set (the test set excludes the dev set's puzzles).")
    parser.add_argument("split", choices=("dev", "test"))
    parser.add_argument("--per-level", type=int, default=PER_LEVEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = REPO_ROOT / (LADDER_DEV_PATH if args.split == "dev" else LADDER_TEST_PATH)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    if args.split == "dev":
        puzzles = build_ladder_eval(args.per_level, LADDER_DEV_SEED)
    else:
        puzzles = build_ladder_eval(args.per_level, LADDER_TEST_SEED, ladder_keys(LADDER_DEV_PATH))
    out.write_text("".join(json.dumps(p.to_dict()) + "\n" for p in puzzles), encoding="utf-8")
    print(f"wrote {len(puzzles)} puzzles to {out}")


if __name__ == "__main__":
    main()
