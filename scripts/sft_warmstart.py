"""Step-limited SFT warm start for track D: teach the answer distribution before RL.

    python scripts/sft_warmstart.py --model <pretrained dir> --out <dir> --steps 200

Each example is a generated training puzzle (r1 prompt) followed by `</think>\\n<answer>SOLUTION</answer>`,
where SOLUTION is the generator's own expression: answers only, no worked reasoning. Puzzles come
from the training streams (held-out and ladder eval puzzles excluded): a third 2-3 numbers, the
rest the ladder levels. Loss on completion tokens only. `--steps x --batch` caps how many solved
examples the model ever sees, which is the quantity a track rule would limit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr_speedrun.countdown import format_prompt
from rlvr_speedrun.data import train_puzzle_stream
from rlvr_speedrun.ladder import ladder_keys, ladder_stream


def examples(seed: int):
    rng = random.Random(seed)
    streams = [
        train_puzzle_stream(seed + 700_000, n_numbers=2, target_range=(1, 50)),
        train_puzzle_stream(seed + 800_000, n_numbers=3, target_range=(1, 50)),
        ladder_stream(seed + 900_000, exclude=ladder_keys()),
    ]
    while True:
        stream = streams[rng.choice((0, 1, 2, 2, 2, 2))]
        puzzle = next(stream)
        yield format_prompt(puzzle, style="r1"), f"</think>\n<answer>{puzzle.solution}</answer>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    data = examples(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "sft_log.jsonl").open("w")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, step / 20)
        rows, labels = [], []
        for prompt, completion in itertools.islice(data, args.batch):
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            rows.append(p + c)
            labels.append([-100] * len(p) + c)
        width = max(map(len, rows))
        ids = torch.tensor([r + [tok.eos_token_id] * (width - len(r)) for r in rows]).cuda()
        lab = torch.tensor([lb + [-100] * (width - len(lb)) for lb in labels]).cuda()
        mask = torch.arange(width).cuda().unsqueeze(0) < torch.tensor([len(r) for r in rows]).cuda().unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask.long()).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == 1:
            row = {"step": step, "loss": loss.item(), "examples": step * args.batch, "s": time.perf_counter() - start}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
    model.to(torch.bfloat16).save_pretrained(out)
    tok.save_pretrained(out)


if __name__ == "__main__":
    main()
