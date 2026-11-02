"""Step-limited SFT on a published worked-solution dataset, before track D RL (PLAN D3).

    python scripts/research/sft_public.py --model <pretrained dir> --out <dir> --data <gsm8k train.jsonl> --steps 300

Data: GSM8K train (`openai/gsm8k`, `main`, revision 740312add88f781978c0658806c59bc2815b9866), each
line {"question", "answer"} as published. The question goes into the r1 chat template in place of
the Countdown request, and the published solution goes inside <think>, with its final number in
<answer>: `"<solution before ####></think>\\n<answer><number after ####></answer>"`. Nothing is
generated; the text is the dataset's. Questions matching a numbers-game keyword are dropped
(RULES_v3, "Both tracks: external data"). Loss on completion tokens only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
from pathlib import Path

from rlvr_speedrun.countdown import R1_TEMPLATE

NUMBERS_GAME = re.compile(r"countdown|24 game|game of 24|make 24", re.IGNORECASE)
HEADER = R1_TEMPLATE.partition("User: ")[0]


def format_example(question: str, answer: str) -> tuple[str, str]:
    """(prompt, completion) for one published GSM8K row."""
    reasoning, _, final = answer.partition("####")
    prompt = f"{HEADER}User: {question.strip()}\nAssistant: Let me solve this step by step.\n<think>"
    return prompt, f" {reasoning.strip()}</think>\n<answer>{final.strip()}</answer>"


def load_rows(path: str) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in open(path)]
    return [format_example(r["question"], r["answer"]) for r in rows if not NUMBERS_GAME.search(r["question"]) and "####" in r["answer"]]


def batches(rows: list, batch: int, seed: int):
    """Shuffled epochs, each row once per epoch."""
    rng = random.Random(seed)
    while True:
        order = rows[:]
        rng.shuffle(order)
        yield from (order[i : i + batch] for i in range(0, len(order) - batch + 1, batch))


def main() -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    rows = load_rows(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "sft_log.jsonl").open("w")
    print(f"sft_public: {len(rows)} rows from {args.data}, {args.steps} steps x {args.batch}", flush=True)
    start = time.perf_counter()
    for step, group_rows in enumerate(itertools.islice(batches(rows, args.batch, args.seed), args.steps), 1):
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, step / 20)
        ids_rows, labels = [], []
        for prompt, completion in group_rows:
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids_rows.append((p + c)[: args.max_len])
            labels.append(([-100] * len(p) + c)[: args.max_len])
        width = max(map(len, ids_rows))
        ids = torch.tensor([r + [tok.eos_token_id] * (width - len(r)) for r in ids_rows]).cuda()
        lab = torch.tensor([lb + [-100] * (width - len(lb)) for lb in labels]).cuda()
        mask = torch.arange(width).cuda().unsqueeze(0) < torch.tensor([len(r) for r in ids_rows]).cuda().unsqueeze(1)
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
