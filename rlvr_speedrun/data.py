"""Dataset utilities and the frozen Countdown held-out sets (eval, confirmation, OOD)."""

from __future__ import annotations

import argparse
import functools
import json
import random
from collections.abc import Iterator
from pathlib import Path

from . import benchmark
from .countdown import Puzzle, generate_puzzle

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_puzzles(path: str | Path) -> list[Puzzle]:
    with Path(path).open(encoding="utf-8") as f:
        return [Puzzle.from_dict(json.loads(line)) for line in f if line.strip()]


@functools.cache
def heldout_keys() -> frozenset:
    """Keys (sorted numbers, target) of every puzzle in the committed held-out sets."""
    keys = set()
    for relpath in benchmark.HELDOUT_PATHS:
        path = REPO_ROOT / relpath
        if path.is_file():
            keys.update(benchmark.puzzle_key(p.numbers, p.target) for p in load_puzzles(path))
    return frozenset(keys)


def train_puzzle_stream(seed: int, exclude_heldout: bool = True, **difficulty) -> Iterator[Puzzle]:
    """Yield an unbounded training stream that never contains a held-out puzzle.

    Seeds are mapped to ``1_000_000 + seed``, away from the held-out generation seeds.
    That alone is not enough: the generator's distribution is concentrated (about 0.4 %
    of raw training draws are exact eval puzzles), so puzzles whose (sorted numbers,
    target) appears in any held-out set are skipped (benchmark v2).
    """
    rng = random.Random(1_000_000 + seed)
    excluded = heldout_keys() if exclude_heldout else frozenset()
    while True:
        puzzle = generate_puzzle(rng, **difficulty)
        if benchmark.puzzle_key(puzzle.numbers, puzzle.target) not in excluded:
            yield puzzle


def _distinct_puzzles(seed: int, n: int, exclude: frozenset = frozenset(), **difficulty) -> list[Puzzle]:
    rng = random.Random(seed)
    seen = set(exclude)
    puzzles = []
    while len(puzzles) < n:
        puzzle = generate_puzzle(rng, **difficulty)
        key = benchmark.puzzle_key(puzzle.numbers, puzzle.target)
        if key not in seen:
            seen.add(key)
            puzzles.append(puzzle)
    return puzzles


def _write(path: Path, puzzles: list[Puzzle], force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing held-out set: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle.to_dict(), separators=(",", ":")) + "\n")
    return path


def build_eval(out: str | Path, force: bool = False) -> Path:
    from .countdown import generate_puzzles

    # v1 set, kept byte-identical (it contains 2 duplicate puzzles; see tests/test_frozen_data.py)
    return _write(Path(out), generate_puzzles(seed=benchmark.EVAL_SEED, n=benchmark.EVAL_SIZE), force)


def build_confirm(out: str | Path, force: bool = False) -> Path:
    exclude = frozenset(benchmark.puzzle_key(p.numbers, p.target) for p in load_puzzles(REPO_ROOT / benchmark.EVAL_PATH))
    return _write(Path(out), _distinct_puzzles(benchmark.CONFIRM_SEED, benchmark.CONFIRM_SIZE, exclude), force)


def build_ood(out: str | Path, force: bool = False) -> Path:
    puzzles = _distinct_puzzles(benchmark.OOD_SEED, benchmark.OOD_SIZE, n_numbers=benchmark.OOD_N_NUMBERS)
    return _write(Path(out), puzzles, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builders = {
        "build-eval": (build_eval, benchmark.EVAL_PATH, "the frozen stopping/eval set"),
        "build-confirm": (build_confirm, benchmark.CONFIRM_PATH, "the frozen confirmation set (disjoint from eval)"),
        "build-ood": (build_ood, benchmark.OOD_PATH, "the frozen 5-number OOD report set"),
    }
    for name, (_, default, help_text) in builders.items():
        build = sub.add_parser(name, help=f"write {help_text}")
        build.add_argument("--out", default=str(default))
        build.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = builders[args.command][0](args.out, args.force)
    print(f"wrote {sum(1 for _ in path.open(encoding='utf-8'))} puzzles to {path}")


if __name__ == "__main__":
    main()
