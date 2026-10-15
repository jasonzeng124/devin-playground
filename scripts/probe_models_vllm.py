"""Which base models give RL something to work with? Sample each model on stopping-set puzzles with vLLM.

    python scripts/probe_models_vllm.py Qwen/Qwen2.5-1.5B Qwen/Qwen3-1.7B-Base ... --out results/probe_models.json

For each model, with the given prompt style and token budget, it reports:
* greedy pass rate;
* mean sampled pass rate (pass@1) and pass@k, overall and per difficulty level
  (scripts/analyze_solves.py);
* the share of puzzles whose k samples disagree, which are the groups GRPO can learn from;
* mean completion length and truncation rate.
Needs the vLLM environment (not the pinned benchmark stack).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlvr_speedrun.countdown import Puzzle, format_prompt, level, verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--data", default="data/countdown_eval.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--prompt-style", default="r1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--constrain", action="store_true", help="answer-constrained decoding (rlvr_speedrun/constrained.py); keys get a ' +constrained' suffix")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    load_all = [Puzzle.from_dict(json.loads(line)) for line in open(args.data)]
    puzzles = load_all[: args.n] if not any(p.level for p in load_all) else load_all
    levels = [p.level or level(p) for p in puzzles]  # ladder files store their level
    if any(p.level for p in puzzles):  # ladder: take --n spread evenly over the levels
        names = list(dict.fromkeys(p.level for p in load_all))
        per = max(1, args.n // len(names))
        puzzles = [p for name in names for p in [q for q in load_all if q.level == name][:per]]
        levels = [p.level for p in puzzles]
    prompts = [format_prompt(p, 3, args.prompt_style) for p in puzzles]
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "models": {}}
    if args.out.exists():
        report = json.loads(args.out.read_text())
    for model in args.models:
        name = model + (" +constrained" if args.constrain else "")
        if name in report["models"]:
            continue
        extra = {}
        if args.constrain:
            from rlvr_speedrun.vllm_constrained import AnswerLogitsProcessor

            extra["logits_processors"] = [AnswerLogitsProcessor]
        llm = LLM(model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=min(4096, 1024 + args.max_tokens), seed=0, **extra)
        stop = dict(stop=["</answer>"], include_stop_str_in_output=True, max_tokens=args.max_tokens)

        def params(stop=stop, **kw):
            if not args.constrain:
                return SamplingParams(**kw, **stop)
            return [SamplingParams(**kw, **stop, extra_args={"countdown": {"numbers": list(p.numbers), "ops": p.ops}}) for p in puzzles]

        greedy = llm.generate(prompts, params(temperature=0.0))
        sampled = llm.generate(prompts, params(temperature=args.temperature, n=args.k, seed=0))
        g_ok = [verify(o.outputs[0].text, p).correct for o, p in zip(greedy, puzzles)]
        per = [[verify(c.text, p).correct for c in o.outputs] for o, p in zip(sampled, puzzles)]
        lens = [len(c.token_ids) for o in sampled for c in o.outputs]
        trunc = [c.finish_reason == "length" for o in sampled for c in o.outputs]
        row = {
            "greedy": statistics.fmean(g_ok),
            "pass@1": statistics.fmean(statistics.fmean(x) for x in per),
            f"pass@{args.k}": statistics.fmean(any(x) for x in per),
            "informative_groups": statistics.fmean(0 < sum(x) < len(x) for x in per),
            "mean_tokens": statistics.fmean(lens),
            "truncated": statistics.fmean(trunc),
            "greedy_by_level": {lv: statistics.fmean(ok for ok, lev in zip(g_ok, levels) if lev == lv) for lv in dict.fromkeys(levels) if lv in levels},
            "greedy_truncated": statistics.fmean(o.outputs[0].finish_reason == "length" for o in greedy),
            "pass@1_by_level": {lv: statistics.fmean(statistics.fmean(x) for x, lev in zip(per, levels) if lev == lv) for lv in dict.fromkeys(levels) if lv in levels},
            f"pass@{args.k}_by_level": {lv: statistics.fmean(any(x) for x, lev in zip(per, levels) if lev == lv) for lv in dict.fromkeys(levels) if lv in levels},
            "examples": [{"puzzle": p.to_dict(), "level": lv, "samples": [{"text": c.text, "correct": ok} for c, ok in zip(o.outputs[:4], x[:4])]}
                         for p, lv, o, x in list(zip(puzzles, levels, sampled, per))[:12]],
        }
        report["models"][name] = row
        print(f"{name}: greedy {row['greedy']:.3f} pass@1 {row['pass@1']:.3f} pass@{args.k} {row[f'pass@{args.k}']:.3f} "
              f"informative {row['informative_groups']:.2f} tokens {row['mean_tokens']:.0f} truncated {row['truncated']:.2f}", flush=True)
        print("   greedy by level " + json.dumps({k: round(v, 3) for k, v in row["greedy_by_level"].items()}) + f" greedy truncated {row['greedy_truncated']:.2f}", flush=True)
        print("   pass@1 by level " + json.dumps({k: round(v, 3) for k, v in row["pass@1_by_level"].items()}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
