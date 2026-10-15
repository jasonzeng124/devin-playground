"""What does a checkpoint actually solve? Break the stopping-set pass rate down by puzzle difficulty.

    python scripts/analyze_solves.py <model or checkpoint> [...] --out results/solves.json

Every stopping-set puzzle gets a difficulty level from a brute-force solver: the smallest set of
operators that reaches the target using all four numbers once ("sum" = the target is the sum of
all four; then "+-", "+-*", "+-*/"). For each model this prints the greedy pass rate per level
and which operators the model's correct answers use. It also samples a few rollouts at the
training temperature on puzzles that need * or /. The examples go to --out.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import torch

from rlvr_speedrun.countdown import LEVELS, Puzzle, format_prompt, level, verify
from rlvr_speedrun.model_utils import generate_completions, load_causal_model, load_tokenizer


def ops_used(expression: str) -> str:
    names = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    found = {names[type(node.op)] for node in ast.walk(ast.parse(expression, mode="eval")) if isinstance(node, ast.BinOp)}
    return "".join(op for op in "+-*/" if op in found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--data", default="data/countdown_eval.jsonl")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt-style", default="v1")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    puzzles = [Puzzle.from_dict(json.loads(line)) for line in open(args.data)]
    levels = [level(p) for p in puzzles]
    share = {lv: levels.count(lv) / len(puzzles) for lv in LEVELS}
    print("puzzles needing each level: " + ", ".join(f"{lv} {share[lv]:.3f}" for lv in LEVELS))
    hard = [p for p, lv in zip(puzzles, levels) if lv in ("+-*", "+-*/")]
    random.Random(0).shuffle(hard)
    report = {"level_share": share, "models": {}}
    tokenizer = load_tokenizer(args.models[0])
    for name in args.models:
        model, device = load_causal_model(name, "cuda")
        completions = []
        for i in range(0, len(puzzles), args.batch_size):
            batch = puzzles[i : i + args.batch_size]
            completions += generate_completions(model, tokenizer, [format_prompt(p, 3, args.prompt_style) for p in batch], max_new_tokens=args.max_new_tokens,
                                                temperature=0.0, device=device)
        rows = [verify(c, p) for c, p in zip(completions, puzzles)]
        by_level = {lv: [r.correct for r, x in zip(rows, levels) if x == lv] for lv in LEVELS}
        used = {}
        for r in rows:
            if r.correct:
                key = ops_used(r.expression)
                used[key] = used.get(key, 0) + 1
        n_ok = sum(r.correct for r in rows)
        print(f"\n== {name}: pass {n_ok / len(puzzles):.3f}")
        print("   pass rate by level needed: " + ", ".join(f"{lv} {sum(v) / len(v):.3f}" for lv, v in by_level.items()))
        print("   contribution to pass rate: " + ", ".join(f"{lv} {sum(v) / len(puzzles):.3f}" for lv, v in by_level.items()))
        print("   operators in correct answers: " + ", ".join(f"{k or 'none'} {v}" for k, v in sorted(used.items(), key=lambda kv: -kv[1])))
        correct_hard = [(p, c) for p, c, r, lv in zip(puzzles, completions, rows, levels) if r.correct and lv in ("+-*", "+-*/")]
        torch.manual_seed(0)
        sampled = []
        for p in hard[:4]:
            outs = generate_completions(model, tokenizer, [format_prompt(p, 3, args.prompt_style)] * 8, max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature, device=device)
            sampled.append({"puzzle": p.to_dict(), "level": level(p),
                            "samples": [{"completion": o, "reward": verify(o, p).reward} for o in outs]})
        report["models"][name] = {
            "pass_rate": n_ok / len(puzzles),
            "pass_by_level": {lv: sum(v) / len(v) for lv, v in by_level.items()},
            "ops_in_correct": used,
            "greedy_correct_hard_examples": [{"puzzle": p.to_dict(), "completion": c} for p, c in correct_hard[:10]],
            "greedy_first_20": [{"puzzle": p.to_dict(), "level": lv, "completion": c, "reward": r.reward}
                                for p, lv, c, r in list(zip(puzzles, levels, completions, rows))[:20]],
            "sampled_hard": sampled,
        }
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
