"""Zero-shot evaluation and checkpoint sweep harness."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

from .countdown import format_prompt, verify
from .data import load_puzzles
from .model_utils import generate_completions, load_causal_model, load_tokenizer


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def aggregate_results(completions: Iterable[str], puzzles) -> dict:
    rows = [verify(completion, puzzle) for completion, puzzle in zip(completions, puzzles)]
    n = len(rows)
    passes = sum(row.correct for row in rows)
    malformed = sum(row.malformed for row in rows)
    lo, hi = wilson_interval(passes, n)
    return {
        "pass_rate": passes / n if n else 0.0,
        "malformed_rate": malformed / n if n else 0.0,
        "n": n,
        "wilson_95_ci": [lo, hi],
    }


def evaluate_model(
    model_name: str,
    puzzles,
    max_new_tokens: int = 64,
    batch_size: int = 16,
    temperature: float = 0.0,
    device: str = "auto",
    few_shot: int = 3,
    n_samples: int = 1,
) -> dict:
    n_samples = max(1, n_samples)
    tokenizer = load_tokenizer(model_name)
    model, resolved = load_causal_model(model_name, device)
    sampled_completions: list[list[str]] = []
    for _ in range(n_samples):
        completions: list[str] = []
        for start in range(0, len(puzzles), batch_size):
            batch = puzzles[start : start + batch_size]
            completions.extend(
                generate_completions(
                    model,
                    tokenizer,
                    [format_prompt(puzzle, few_shot=few_shot) for puzzle in batch],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    device=resolved,
                )
            )
        sampled_completions.append(completions)
    if n_samples == 1:
        result = aggregate_results(sampled_completions[0], puzzles)
        all_completions = sampled_completions[0]
    else:
        per_puzzle = [
            [verify(sampled_completions[sample][index], puzzle) for sample in range(n_samples)]
            for index, puzzle in enumerate(puzzles)
        ]
        passes = sum(any(row.correct for row in rows) for rows in per_puzzle)
        malformed = sum(result.malformed for rows in per_puzzle for result in rows)
        lo, hi = wilson_interval(passes, len(puzzles))
        result = {
            "pass_rate": passes / len(puzzles) if puzzles else 0.0,
            "malformed_rate": malformed / (len(puzzles) * n_samples) if puzzles else 0.0,
            "n": len(puzzles),
            "n_samples": n_samples,
            "wilson_95_ci": [lo, hi],
        }
        all_completions = sampled_completions[0]
    result["model"] = model_name
    result["examples"] = [
        {
            "puzzle": puzzle.to_dict(),
            "completion": completion,
            "reward": verify(completion, puzzle).reward,
        }
        for puzzle, completion in list(zip(puzzles, all_completions))[:5]
    ]
    return result


def _print_result(result: dict) -> None:
    lo, hi = result["wilson_95_ci"]
    print(
        f"{result.get('model', '')}: pass_rate={result['pass_rate']:.4f} "
        f"malformed_rate={result['malformed_rate']:.4f} n={result['n']} "
        f"Wilson95%=[{lo:.4f}, {hi:.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--models", help="comma-separated checkpoints for a sweep")
    parser.add_argument("--eval", default="data/countdown_eval.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out")
    args = parser.parse_args()
    names = [name.strip() for name in args.models.split(",")] if args.models else [args.model]
    if not names or any(not name for name in names):
        parser.error("provide --model or --models")
    puzzles = load_puzzles(args.eval)
    if args.limit is not None:
        puzzles = puzzles[: args.limit]
    results = [
        evaluate_model(
            name,
            puzzles,
            args.max_new_tokens,
            args.batch_size,
            args.temperature,
            args.device,
            args.few_shot,
            args.n_samples,
        )
        for name in names
    ]
    if len(results) > 1:
        print("| model | pass rate | malformed rate | n | Wilson 95% CI |")
        print("|---|---:|---:|---:|---|")
        for result in results:
            lo, hi = result["wilson_95_ci"]
            print(f"| {result['model']} | {result['pass_rate']:.4f} | {result['malformed_rate']:.4f} | {result['n']} | [{lo:.4f}, {hi:.4f}] |")
    else:
        _print_result(results[0])
        for example in results[0]["examples"]:
            print(f"example: {example['completion']!r} reward={example['reward']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results if len(results) > 1 else results[0], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
