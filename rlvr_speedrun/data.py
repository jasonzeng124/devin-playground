"""Dataset utilities and the frozen Countdown evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from .countdown import Puzzle, generate_puzzle, generate_puzzles


def load_puzzles(path: str | Path) -> list[Puzzle]:
    with Path(path).open(encoding="utf-8") as f:
        return [Puzzle.from_dict(json.loads(line)) for line in f if line.strip()]


def train_puzzle_stream(seed: int, **difficulty) -> Iterator[Puzzle]:
    """Yield an unbounded stream from seeds offset from frozen eval seed space.

    The frozen evaluation set uses seed 20240601. Training seeds are mapped to
    ``1_000_000 + seed`` so a training seed cannot equal that eval seed.
    """
    import random

    rng = random.Random(1_000_000 + seed)
    while True:
        yield generate_puzzle(rng, **difficulty)


def build_eval(out: str | Path, force: bool = False) -> Path:
    path = Path(out)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing eval set: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    puzzles = generate_puzzles(seed=20240601, n=2000)
    with path.open("w", encoding="utf-8") as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle.to_dict(), separators=(",", ":")) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-eval", help="write the frozen evaluation set")
    build.add_argument("--out", default="data/countdown_eval.jsonl")
    build.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build-eval":
        path = build_eval(args.out, args.force)
        print(f"wrote {sum(1 for _ in path.open(encoding='utf-8'))} puzzles to {path}")


if __name__ == "__main__":
    main()
