"""Canonical confirmation eval for a finished seed (benchmark v2, RULES.md).

    python -m rlvr_speedrun.confirm --model <seed>/final --out <seed>/confirm.json

Scores the final checkpoint on the frozen confirmation set (never seen during the
timed run) and the 5-number OOD report set with the fixed protocol from
`rlvr_speedrun.benchmark`: greedy decoding, 3-shot prompt, 64 new tokens. Nothing
here reads the record's config, so a trainer's own eval settings cannot leak in.
Untimed, like the capability probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from . import benchmark
from .countdown import format_prompt
from .data import REPO_ROOT, load_puzzles
from .eval import aggregate_results
from .model_utils import generate_completions, load_causal_model, load_tokenizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(model, tokenizer, puzzles, device, batch_size: int) -> dict:
    # sort by prompt length so batches pad little; order does not affect the pass rate
    order = sorted(range(len(puzzles)), key=lambda i: len(format_prompt(puzzles[i], benchmark.FEW_SHOT)))
    completions: list[str] = [""] * len(puzzles)
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        prompts = [format_prompt(puzzles[i], few_shot=benchmark.FEW_SHOT) for i in index]
        outputs = generate_completions(model, tokenizer, prompts, max_new_tokens=benchmark.MAX_EVAL_NEW_TOKENS, temperature=0.0, device=device)
        for i, text in zip(index, outputs):
            completions[i] = text
    return aggregate_results(completions, puzzles)


def run_confirm(model_path: str, device: str = "auto", batch_size: int = 256, limit: int | None = None) -> dict:
    tokenizer = load_tokenizer(model_path)
    model, resolved = load_causal_model(model_path, device)
    result = {
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "model": model_path,
        "protocol": {"few_shot": benchmark.FEW_SHOT, "max_new_tokens": benchmark.MAX_EVAL_NEW_TOKENS, "decoding": "greedy"},
        "gpu": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else resolved.type,
    }
    for name, relpath in (("confirm", benchmark.CONFIRM_PATH), ("ood", benchmark.OOD_PATH)):
        path = REPO_ROOT / relpath
        puzzles = load_puzzles(path)[:limit]
        start = time.perf_counter()
        with torch.no_grad():
            scored = score(model, tokenizer, puzzles, resolved, batch_size)
        scored.update({"data": str(relpath), "data_sha256": _sha256(path), "eval_s": time.perf_counter() - start})
        result[name] = scored
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint directory (a seed's final/) or HF model id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="smoke tests only; a record's confirm.json must score the full sets")
    args = parser.parse_args()
    result = run_confirm(args.model, args.device, args.batch_size, args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name in ("confirm", "ood"):
        lo, hi = result[name]["wilson_95_ci"]
        print(f"{name}: pass_rate={result[name]['pass_rate']:.4f} n={result[name]['n']} Wilson95%=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
