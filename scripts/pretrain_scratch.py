"""Pretrain a small Llama-style model from scratch on FineWeb-Edu (the track D baseline's pretraining).

    python scripts/pretrain_scratch.py tokenize --parquet <dir> --tokenizer <dir> --out tokens.bin
    python scripts/pretrain_scratch.py train --tokens tokens.bin --tokenizer <dir> --out <model dir> [--minutes 60]

`tokenize` streams parquet row groups through the tokenizer and writes uint16 ids, EOS after each document. `train` runs next-token
prediction with AdamW or Muon (`--optimizer`; warmup, cosine decay), bf16 autocast and torch.compile, stopping after
--minutes of training or --max-tokens, and saves a `LlamaForCausalLM` plus the tokenizer so
`rlvr_speedrun.grpo` and vLLM can load it like any HF model. Deliberately plain (no speedrun
tricks): the question is whether RL can start from a model of roughly GPT-2-small quality.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _tokenize_row_group(args):
    """Tokenize one parquet row group to a uint16 array (EOS after each document)."""
    path, group, tokenizer_dir = args
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    texts = pq.ParquetFile(path).read_row_group(group, columns=["text"]).column("text").to_pylist()
    parts = []
    for start in range(0, len(texts), 1000):
        for ids in tok(texts[start : start + 1000], add_special_tokens=False)["input_ids"]:
            parts.append(np.asarray(ids + [tok.eos_token_id], dtype=np.uint16))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def tokenize(args) -> None:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(args.parquet, "**", "*.parquet"), recursive=True))
    tasks = [(f, g, args.tokenizer) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    total = 0
    with open(args.out, "wb") as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for array in pool.map(_tokenize_row_group, tasks):  # in order, streamed to disk
            array.tofile(out)
            total += len(array)
    print(f"{len(files)} files, {len(tasks)} row groups -> {total / 1e9:.3f}B tokens in {args.out}")


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token
    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    val, data = data[: 2_000_000], data[2_000_000:]
    config = LlamaConfig(
        vocab_size=len(tok), hidden_size=args.width, intermediate_size=int(args.width * 8 / 3 / 64) * 64, num_hidden_layers=args.layers,
        num_attention_heads=args.width // 64, num_key_value_heads=args.width // 64, max_position_embeddings=2048, tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, torch_dtype="bfloat16",
    )
    model = LlamaForCausalLM(config).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    if args.optimizer == "muon":  # Muon for hidden matrices, AdamW for embeddings and norms (modded-nanogpt's split)
        hidden = [p for n, p in model.named_parameters() if p.dim() == 2 and "embed" not in n and "lm_head" not in n]
        rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "embed" not in n and "lm_head" not in n)]
        opts = [torch.optim.Muon(hidden, lr=args.lr, weight_decay=0.1, adjust_lr_fn="match_rms_adamw"),
                torch.optim.AdamW([{"params": [p for p in rest if p.dim() >= 2], "weight_decay": 0.1},
                                   {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), fused=True)]
    else:
        opts = [torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95), fused=True)]
    compiled = torch.compile(model)
    seq, micro = args.seq_len, args.micro_batch
    accum = max(1, args.batch_tokens // (seq * micro))
    total_steps = args.max_tokens // (seq * micro * accum)
    rng = np.random.default_rng(args.seed)

    def batch(source):
        idx = rng.integers(0, len(source) - seq - 1, micro)
        x = torch.from_numpy(np.stack([source[i : i + seq + 1].astype(np.int64) for i in idx])).cuda(non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def lr_at(step, frac_time):
        frac = max(step / total_steps, frac_time)  # decay on whichever budget runs out first
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = batch(val)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = compiled(input_ids=x).logits
            losses.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "pretrain_log.jsonl").open("w")
    start, step, tokens_seen = time.perf_counter(), 0, 0
    budget = args.minutes * 60
    model.train()
    while step < total_steps:
        elapsed = time.perf_counter() - start
        if elapsed >= budget:
            break
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, elapsed / budget)
        for _ in range(accum):
            x, y = batch(data)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = compiled(input_ids=x).logits
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += seq * micro * accum
        if step % 50 == 0 or step == 1:
            row = {"step": step, "tokens": tokens_seen, "minutes": (time.perf_counter() - start) / 60, "train_loss": loss.item() * accum,
                   "lr": opts[0].param_groups[0]["lr"], "tok_per_s": tokens_seen / (time.perf_counter() - start)}
            if step % 500 == 0:
                row["val_loss"] = val_loss()
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
    final = {"step": step, "tokens": tokens_seen, "minutes": (time.perf_counter() - start) / 60, "val_loss": val_loss(), "params_m": n_params / 1e6}
    log.write(json.dumps(final) + "\n")
    print(final, flush=True)
    model.to(torch.bfloat16).save_pretrained(out)
    tok.save_pretrained(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--parquet", required=True)
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("train")
    r.add_argument("--tokens", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--width", type=int, default=768)
    r.add_argument("--layers", type=int, default=12)
    r.add_argument("--seq-len", type=int, default=1024)
    r.add_argument("--micro-batch", type=int, default=64)
    r.add_argument("--batch-tokens", type=int, default=524_288)
    r.add_argument("--lr", type=float, default=1.5e-3)
    r.add_argument("--warmup", type=int, default=200)
    r.add_argument("--max-tokens", type=int, default=1_400_000_000)
    r.add_argument("--minutes", type=float, default=60.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    args = parser.parse_args()
    tokenize(args) if args.cmd == "tokenize" else train(args)


if __name__ == "__main__":
    main()
