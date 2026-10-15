"""Score one seed's final checkpoint on the frozen test ladder (benchmark v3, docs/RULES_v3.md).

    python -m rlvr_speedrun.ladder_confirm --model <seed>/final --out <seed>/confirm_v3.json [--constrain] [--think-budget N]

Every test puzzle gets `V3_SAMPLES_PER_PUZZLE` independent samples with the fixed zero-shot r1
prompt and at most `V3_MAX_NEW_TOKENS` new tokens. A sample is correct if the verifier accepts it.
The seed's score is the mean over levels of each level's pass@1 (the mean correctness of its
samples), so every level weighs the same. The sampler settings are the entrant's (temperature,
answer-constrained decoding, think budget) and are recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import benchmark
from .countdown import format_prompt, verify
from .ladder import load_ladder


def score(correct: list[list[bool]], levels: list[str]) -> dict:
    """Per-level pass@1 and the level-weighted ladder score from per-puzzle sample correctness."""
    by_level = {
        level: statistics.fmean(statistics.fmean(c) for c, lv in zip(correct, levels) if lv == level)
        for level in dict.fromkeys(levels)
    }
    return {"ladder_score": statistics.fmean(by_level.values()), "pass@1_by_level": by_level,
            f"pass@{len(correct[0])}": statistics.fmean(any(c) for c in correct)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--constrain", action="store_true", help="answer-constrained decoding (rlvr_speedrun/constrained.py)")
    parser.add_argument("--think-budget", type=int, default=0, help="with --constrain: force <answer> after this many free tokens")
    parser.add_argument("--max-new-tokens", type=int, default=benchmark.V3_MAX_NEW_TOKENS)
    parser.add_argument("--limit-per-level", type=int, default=0, help="smoke tests only; a confirmation uses the whole set")
    args = parser.parse_args()
    if args.max_new_tokens > benchmark.V3_MAX_NEW_TOKENS:
        raise SystemExit(f"--max-new-tokens is capped at {benchmark.V3_MAX_NEW_TOKENS}")
    if args.think_budget and not args.constrain:
        raise SystemExit("--think-budget needs --constrain")

    from vllm import LLM, SamplingParams

    puzzles = load_ladder(benchmark.LADDER_TEST_PATH)
    if args.limit_per_level:
        puzzles = [p for level in benchmark.LADDER_LEVELS for p in [q for q in puzzles if q.level == level][: args.limit_per_level]]
    prompts = [format_prompt(p, 0, "r1") for p in puzzles]
    extra = {}
    if args.constrain:
        from .vllm_constrained import AnswerLogitsProcessor

        extra["logits_processors"] = [AnswerLogitsProcessor]
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=min(4096, 1024 + args.max_new_tokens),
              seed=0, **extra)
    common = dict(n=benchmark.V3_SAMPLES_PER_PUZZLE, temperature=args.temperature, max_tokens=args.max_new_tokens,
                  stop=["</answer>"], include_stop_str_in_output=True, seed=0)
    if args.constrain:
        params = [SamplingParams(**common, extra_args={"countdown": {"numbers": list(p.numbers), "ops": p.ops,
                                                                     "think_budget": args.think_budget}}) for p in puzzles]
    else:
        params = SamplingParams(**common)
    outputs = llm.generate(prompts, params)
    correct = [[verify(c.text, p).correct for c in o.outputs] for o, p in zip(outputs, puzzles)]
    result = {
        "benchmark_version": 3,
        "model": args.model,
        "test_set": str(benchmark.LADDER_TEST_PATH),
        "n_puzzles": len(puzzles),
        "samples_per_puzzle": benchmark.V3_SAMPLES_PER_PUZZLE,
        "sampler": {"temperature": args.temperature, "constrain": args.constrain, "think_budget": args.think_budget,
                    "max_new_tokens": args.max_new_tokens},
        **score(correct, [str(p.level) for p in puzzles]),
        "truncated": statistics.fmean(c.finish_reason == "length" for o in outputs for c in o.outputs),
        "mean_tokens": statistics.fmean(len(c.token_ids) for o in outputs for c in o.outputs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model", "ladder_score", "pass@1_by_level", "sampler")}))


if __name__ == "__main__":
    main()
