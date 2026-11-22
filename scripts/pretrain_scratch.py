"""Pretrain a small Llama-style model from scratch on FineWeb-Edu (the track D baseline's pretraining).

    python scripts/pretrain_scratch.py tokenize --parquet <dir> --tokenizer <dir> --out tokens.bin
    python scripts/pretrain_scratch.py train --tokens tokens.bin --tokenizer <dir> --out <model dir> [--minutes 60]

`tokenize` streams parquet row groups through the tokenizer and writes uint16 ids, EOS after each document. `train` runs next-token
prediction with AdamW or Muon (`--optimizer`; warmup, cosine decay), bf16 autocast and torch.compile, stopping after
--minutes of training or --max-tokens, and saves a `LlamaForCausalLM` plus the tokenizer so
`rlvr_speedrun.grpo` and vLLM can load it like any HF model. Deliberately plain (no speedrun
tricks): the question is whether RL can start from a model of roughly GPT-2-small quality.

`--phase2 a.bin:0.5,b.bin:0.25,...` (off by default) switches the training draws, once the run's progress (the
fraction the lr schedule decays on: the larger of steps and time) reaches `--phase2-start`, from --tokens to the
listed token files, each micro-batch row from one file picked by weight. A file named the same as --tokens is read
without its validation slice; validation stays on --tokens.

`--ddp` (off by default) trains data-parallel under torchrun, one process a GPU: every rank starts from the same
weights and draws its own rows, the gradients are averaged in flat buckets after each step's micro-batches, and rank
0's clock is broadcast every step, so all ranks take the same lr and stop together. --batch-tokens stays the global
batch. At each validation the ranks' weight sums are compared (`rank_spread` in the log) and rank 0's weights are
sent to all.
`--resume-every-minutes N` (0, the default: never) saves the model, the optimizers, the step, the tokens and the
elapsed training time to `<out>/resume/state.pt` every N minutes; `--resume` continues from that file if it exists
(the clock and the budget carry on from the saved time, the log is appended to, so rows logged after the save
appear twice, and each rank's draws are reseeded from the seed, its rank and the step).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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


def parse_phase2(spec: str, tokens_path: str, train_data) -> list:
    """`path:weight,...` -> [(token array, weight)]; --tokens itself maps to its training part (no validation slice)."""
    out = []
    for item in spec.split(","):
        path, sep, weight = item.strip().rpartition(":")
        if not sep or not path or float(weight) <= 0:
            raise SystemExit(f"--phase2: expected path:weight with a positive weight, got {item!r}")
        same = os.path.realpath(path) == os.path.realpath(tokens_path)
        out.append((train_data if same else np.memmap(path, dtype=np.uint16, mode="r"), float(weight)))
    return out


def step_layout(batch_tokens: int, seq: int, micro: int, world: int, max_tokens: int) -> tuple:
    """(micro-batches a step on each rank, tokens a step over all ranks, total steps); world 1 is the one-GPU layout."""
    accum = max(1, batch_tokens // (seq * micro * world))
    step_tokens = seq * micro * accum * world
    return accum, step_tokens, max_tokens // step_tokens


def allreduce_grads(params, bucket_bytes: int = 256 << 20) -> None:
    """Average every parameter's .grad over the default process group, in place, in flat buckets of one dtype."""
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    buckets, current, size = [], [], 0
    for g in (p.grad for p in params if p.grad is not None):
        nbytes = g.numel() * g.element_size()
        if current and (g.dtype != current[0].dtype or size + nbytes > bucket_bytes):
            buckets.append(current)
            current, size = [], 0
        current.append(g)
        size += nbytes
    if current:
        buckets.append(current)
    for bucket in buckets:
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for g in bucket:
            g.copy_(flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()


def save_resume(path, model, opts, meta: dict) -> None:
    """Write the model's and the optimizers' state with `meta` (step, tokens, elapsed seconds) to `path`, atomically."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts], "meta": meta}, tmp)
    os.replace(tmp, path)


def load_resume(path, model, opts, device=None) -> dict:
    """Load what save_resume wrote into `model` and `opts`; returns its meta."""
    import torch

    state = torch.load(path, map_location=device, weights_only=True)
    if len(state["opts"]) != len(opts):
        raise SystemExit(f"--resume: {path} has {len(state['opts'])} optimizers, this run has {len(opts)}")
    model.load_state_dict(state["model"])
    for opt, saved in zip(opts, state["opts"]):
        opt.load_state_dict(saved)
    return state["meta"]


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    rank, world = 0, 1
    if args.ddp:
        import torch.distributed as dist

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    main_rank = rank == 0
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
    if args.ddp:  # the same seed gives the same weights; rank 0's are sent anyway
        for p in model.parameters():
            dist.broadcast(p.data, 0)
    n_params = sum(p.numel() for p in model.parameters())
    if main_rank:
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
    resume_path = Path(args.out) / "resume" / "state.pt"
    resumed = load_resume(resume_path, model, opts, "cuda") if args.resume and resume_path.exists() else None
    ckw = {"mode": args.compile_mode} if args.compile_mode else {}
    compiled = torch.compile(model, **ckw)

    def _fused(x, y):  # only with --fused-loss: Inductor fuses the fp32 upcast into the loss
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, use_cache=False).logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))

    fused_loss = torch.compile(_fused, **ckw) if args.fused_loss else None
    seq, micro = args.seq_len, args.micro_batch
    accum, step_tokens, total_steps = step_layout(args.batch_tokens, seq, micro, world, args.max_tokens)
    if resumed:
        rng = np.random.default_rng([args.seed, rank, resumed["step"]])
    else:
        rng = np.random.default_rng(args.seed if rank == 0 else [args.seed, rank])
    phase2 = parse_phase2(args.phase2, args.tokens, data) if args.phase2 else []
    p2_weights = np.array([w for _, w in phase2]) / sum(w for _, w in phase2) if phase2 else None
    progress = [0.0]  # the run's progress, as lr_at measures it; set each step

    def batch(source):
        idx = rng.integers(0, len(source) - seq - 1, micro)
        x = torch.from_numpy(np.stack([source[i : i + seq + 1].astype(np.int64) for i in idx])).cuda(non_blocking=True)
        return x[:, :-1], x[:, 1:]

    # --prefetch: the same draws in the same order from rng, gathered and pinned on a thread ahead of use. Val
    # batches still draw from rng when they run, so after the first val eval the training draws are offset.
    pool, ahead = (ThreadPoolExecutor(max_workers=2), deque()) if args.prefetch else (None, None)

    def draw():
        """(token array, offset) for each row of one training micro-batch."""
        if phase2 and progress[0] >= args.phase2_start:
            return [(phase2[k][0], int(rng.integers(0, len(phase2[k][0]) - seq - 1))) for k in rng.choice(len(phase2), micro, p=p2_weights)]
        return [(data, i) for i in rng.integers(0, len(data) - seq - 1, micro)]

    def _rows(rows):
        return np.stack([source[i : i + seq + 1].astype(np.int64) for source, i in rows])

    def _gather(rows):
        return torch.from_numpy(_rows(rows)).pin_memory()

    def train_batch():
        if pool is None:
            x = torch.from_numpy(_rows(draw())).cuda(non_blocking=True)
            return x[:, :-1], x[:, 1:]
        while len(ahead) <= args.prefetch:
            ahead.append(pool.submit(_gather, draw()))
        x = ahead.popleft().result().cuda(non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def lr_at(step, frac_time):
        frac = max(step / total_steps, frac_time)  # decay on whichever budget runs out first
        progress[0] = frac
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = batch(val)
            if fused_loss is not None:
                losses.append(fused_loss(x, y).item())
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = compiled(input_ids=x).logits
            losses.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item())
        model.train()
        mean = sum(losses) / len(losses)
        if args.ddp:
            total = torch.tensor([mean], device="cuda", dtype=torch.float64)
            dist.all_reduce(total)
            mean = total.item() / world
        return mean

    @torch.no_grad()
    def rank_spread():
        """The ranks' largest minus smallest sum of all weights (0.0 while they agree); then rank 0's weights go to all."""
        total = torch.stack([p.double().sum() for p in model.parameters()]).sum().reshape(1)
        gathered = [torch.zeros_like(total) for _ in range(world)]
        dist.all_gather(gathered, total)
        for p in model.parameters():
            dist.broadcast(p.data, 0)
        values = torch.cat(gathered)
        return (values.max() - values.min()).item()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "pretrain_log.jsonl").open("a" if resumed else "w") if main_rank else None
    start, step, tokens_seen = time.perf_counter(), 0, 0
    if resumed:
        start -= resumed["elapsed"]
        step, tokens_seen = resumed["step"], resumed["tokens"]
        if main_rank:
            print(f"resumed from {resume_path} at step {step}, {resumed['elapsed'] / 60:.2f} min", flush=True)
    last_save = time.perf_counter() - start
    budget = args.minutes * 60
    model.train()
    while step < total_steps:
        elapsed = time.perf_counter() - start
        if args.ddp:  # rank 0's clock, so every rank takes the same lr and stops at the same step
            clock = torch.tensor([elapsed], device="cuda", dtype=torch.float64)
            dist.broadcast(clock, 0)
            elapsed = clock.item()
        if elapsed >= budget:
            break
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, elapsed / budget)
        for _ in range(accum):
            x, y = train_batch()
            if fused_loss is not None:
                loss = fused_loss(x, y) / accum
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = compiled(input_ids=x).logits
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / accum
            loss.backward()
        if args.ddp:
            allreduce_grads(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += step_tokens
        if args.resume_every_minutes > 0 and elapsed - last_save >= args.resume_every_minutes * 60:
            if main_rank:  # the other ranks wait in the next step's broadcast
                save_resume(resume_path, model, opts, {"step": step, "tokens": tokens_seen, "elapsed": time.perf_counter() - start})
            last_save = elapsed
        if step % 50 == 0 or step == 1:
            row = {"step": step, "tokens": tokens_seen, "minutes": (time.perf_counter() - start) / 60, "train_loss": loss.item() * accum,
                   "lr": opts[0].param_groups[0]["lr"], "tok_per_s": tokens_seen / (time.perf_counter() - start)}
            if phase2:
                row["phase2"] = progress[0] >= args.phase2_start
            if step % 500 == 0:
                row["val_loss"] = val_loss()
                if args.ddp:
                    row["rank_spread"] = rank_spread()
            if main_rank:
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(row, flush=True)
    final = {"step": step, "tokens": tokens_seen, "minutes": (time.perf_counter() - start) / 60, "val_loss": val_loss(), "params_m": n_params / 1e6}
    if args.ddp:
        final["gpus"] = world
    if main_rank:
        log.write(json.dumps(final) + "\n")
        print(final, flush=True)
        model.to(torch.bfloat16).save_pretrained(out)
        tok.save_pretrained(out)
    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()


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
    r.add_argument("--fused-loss", action="store_true",
                   help="compile the forward and the cross-entropy as one graph (no fp32 logits outside it); off by default")
    r.add_argument("--compile-mode", default=None, help="torch.compile mode (e.g. max-autotune-no-cudagraphs); default: torch's default")
    r.add_argument("--phase2", default="",
                   help="path:weight,... token files to draw from once progress reaches --phase2-start (default: none)")
    r.add_argument("--phase2-start", type=float, default=0.5, help="the run's progress at which --phase2 begins")
    r.add_argument("--prefetch", type=int, default=0,
                   help="gather this many training micro-batches ahead into pinned memory on a thread (0, the default: none)")
    r.add_argument("--ddp", action="store_true", help="data-parallel under torchrun, one process a GPU; off by default")
    r.add_argument("--resume-every-minutes", type=float, default=0.0,
                   help="save <out>/resume/state.pt every this many minutes of training (0, the default: never)")
    r.add_argument("--resume", action="store_true", help="continue from <out>/resume/state.pt if it exists; off by default")
    args = parser.parse_args()
    tokenize(args) if args.cmd == "tokenize" else train(args)


if __name__ == "__main__":
    main()
