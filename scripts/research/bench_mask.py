"""CPU benchmark of the native sampler's grammar mask (LOG 2026-11-24, the s_mask_cpu item; no GPU, no model).

Replays `fast_rollout.generate`'s per-token grammar work over simulated rollouts at r106's shape (64 ladder puzzles a
step, 8 rows each, the TinyLlama tokenizer, think budget 0): each row writes a few free tokens, then OPEN, then a
random walk over the ids the grammar allows (seeded), then CLOSE. It times, per decoding step, the two halves of the
timed block: the rows' `allowed_ids` calls, and `mask_rows_cached` on a `meta` tensor (the Python and dispatcher
work only; on the GPU the kernels are launched, not run, on this path). It also counts the allowed sets per step,
the index-cache misses (each a host-to-device copy on the GPU) and the ids per step. With --profile it prints the
top cProfile rows of the whole loop.
  uv run python scripts/research/bench_mask.py --steps 20 [--factored] [--fast] [--profile]
"""
import argparse
import cProfile
import faulthandler
import io
import pstats
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from rlvr_speedrun.constrained import (
    CLOSE,
    AnswerGrammar,
    MaskTable,
    RequestConstraint,
    mask_rows_cached,
    open_token_ids,
    token_strings,
)
from rlvr_speedrun.grpo import LADDER_LEVELS
from rlvr_speedrun.ladder import ladder_stream


def simulate_step(grammar, tok_close, puzzles, group, rng, index_cache, device, stats, table=None, max_tokens=256, free_max=8):
    """One sampling step's grammar work; appends per-decoding-step timings to `stats`."""
    rows = [(pz, g) for pz in puzzles for g in range(group)]
    constraints = [RequestConstraint(grammar, pz.numbers, pz.ops, index_cache, 0) for pz, _ in rows]
    vocab = len(grammar.strings)
    plain = [i for i, s in enumerate(grammar.strings) if s.strip().isalpha() and len(s) > 2][:2000]
    plans = []  # per row: the free tokens, then OPEN's ids
    for _ in rows:
        plans.append([rng.choice(plain) for _ in range(rng.randint(0, free_max))] + list(grammar.open_ids))
    outputs = [[] for _ in rows]
    closing = [None] * len(rows)  # CLOSE's ids still to write once the answer is done
    active = list(range(len(rows)))
    logits = torch.empty(len(rows), vocab, device=device)
    for _ in range(max_tokens):
        t0 = time.perf_counter()
        allowed = [(r, constraints[r].allowed_ids(outputs[r])) for r in active]
        t1 = time.perf_counter()
        misses = sum(1 for _, a in allowed if a and a not in index_cache)
        if table is not None:
            table(logits, allowed)
        else:
            mask_rows_cached(logits, allowed, index_cache)
        t2 = time.perf_counter()
        sets = [a for _, a in allowed if a]
        stats.append({"rows": len(active), "allowed_s": t1 - t0, "mask_s": t2 - t1, "sets": len(sets),
                      "unique": len({id(a) for a in sets}), "misses": misses, "ids": sum(len(a) for a in sets)})
        still = []
        for (r, a) in allowed:
            if len(outputs[r]) < len(plans[r]):
                tid = plans[r][len(outputs[r])]
            elif closing[r] is not None:
                tid = closing[r].pop(0)
            elif a:
                tid = rng.choice(a)
            else:  # answer done (or broken): write CLOSE
                closing[r] = list(tok_close)
                tid = closing[r].pop(0)
            outputs[r].append(tid)
            text = "".join(grammar.strings[t] for t in outputs[r][-6:])
            if CLOSE in text or len(outputs[r]) >= max_tokens or len(outputs[r]) > len(plans[r]) + 60:
                continue
            still.append(r)
        active = still
        if not active:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(Path.home() / "orcd/pool/data/tinyllama-tokenizer"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--factored", action="store_true")
    ap.add_argument("--fast", action="store_true", help="grammar_fast_path: interned states and MaskTable")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)  # where a slow stage is, every 90 s
    print("imports done", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print("tokenizer loaded", flush=True)
    t = time.perf_counter()
    grammar = AnswerGrammar(token_strings(tok), open_token_ids(tok), factored=args.factored, interned=args.fast)
    table = MaskTable() if args.fast else None
    print(f"grammar built in {time.perf_counter() - t:.1f} s; vocab {len(grammar.strings)}, open ids {grammar.open_ids}")
    tok_close = tuple(tok.encode(CLOSE, add_special_tokens=False))
    stream = ladder_stream(args.seed, LADDER_LEVELS)
    rng = random.Random(args.seed)
    index_cache: dict = {}
    device = torch.device("meta")
    per_step = []
    prof = cProfile.Profile() if args.profile else None
    for step in range(args.steps):
        puzzles = [next(stream) for _ in range(args.prompts)]
        stats: list = []
        if args.fast:
            grammar.trim(400_000)  # as `NativeRollout.generate` does (GRAMMAR_CACHE_LIMIT)
        if prof:
            prof.enable()
        simulate_step(grammar, tok_close, puzzles, args.group, rng, index_cache, device, stats, table)
        if prof:
            prof.disable()
        row = {k: sum(d[k] for d in stats) for k in ("allowed_s", "mask_s", "sets", "misses", "ids", "rows")}
        row["unique"] = statistics.mean(d["unique"] for d in stats)
        row["decode_steps"] = len(stats)
        per_step.append(row)
        print(f"step {step}: {len(stats)} decoding steps, {row['rows']} row-tokens, allowed {row['allowed_s'] * 1e3:.1f} ms, "
              f"mask {row['mask_s'] * 1e3:.1f} ms, sets {row['sets']}, unique/decode step {row['unique']:.1f}, "
              f"misses {row['misses']}, ids {row['ids']}, grammar cache {len(grammar._cache)}, interned {len(grammar._info)}, index cache {len(index_cache)}")
    late = per_step[len(per_step) // 2 :]
    for k in ("allowed_s", "mask_s"):
        v = statistics.median(r[k] for r in late)
        tokens = statistics.median(r["rows"] for r in late)
        print(f"median over the second half: {k} {v * 1e3:.1f} ms a step, {v / tokens * 1e6:.2f} us a row-token")
    if prof:
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(25)
        print(out.getvalue())


if __name__ == "__main__":
    main()
