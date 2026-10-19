"""A minimal, readable GRPO implementation using plain PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
import weakref
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import StaticCache

from . import benchmark
from .constrained import AnswerGrammar, mask_logits, open_token_ids, token_strings
from .countdown import LEVELS, level, safe_eval, split_prompt, verify
from .data import load_puzzles, train_puzzle_stream
from .distributed import SINGLE, Dist, destroy, init_from_env
from .eval import aggregate_results
from .ladder import LEVEL_NAMES as LADDER_LEVELS
from .ladder import ladder_keys, ladder_stream
from .model_utils import answer_stopping_criteria, load_causal_model, load_tokenizer


@dataclass
class GRPOConfig:
    model: str = ""
    ref_model: str | None = None
    group_size: int = 8
    prompts_per_step: int = 8
    oversample: int = 1
    gen_batch_size: int = 0
    compile: bool = False
    pad_to_multiple: int = 0
    cache_len: int = 0
    rollout_backend: str = "hf"  # "hf" (model.generate) or "vllm" (colocated engine; rlvr_speedrun/vllm_rollout.py)
    vllm_gpu_memory: float = 0.3  # share of GPU memory given to the vLLM engine (weights + KV cache)
    adv_norm: str = "std"
    lr: float = 1e-6
    # learning rate over training time (wall clock minus evals): a linear warmup over lr_warmup_s, then
    # "constant", or a "cosine" / "linear" decay to lr * lr_final_frac at time_budget_s
    lr_schedule: str = "constant"
    lr_warmup_s: float = 0.0
    lr_final_frac: float = 0.1
    kl_coef: float = 0.02
    # partial credit for well-formed wrong answers, up to this weight (see `partial_credit`); never searches for a solution
    shaping: float = 0.0
    # after "<answer>", sample only well-formed uses of the puzzle's numbers and operators (rlvr_speedrun/constrained.py; vLLM backend)
    constrain_answer: bool = False
    think_budget: int = 0  # with constrain_answer: force "<answer>" after this many free tokens (0 = never)
    cov_clip_frac: float = 0.0  # Clip-Cov: drop the gradient of this fraction of tokens with the largest log-prob x advantage covariance
    malformed_reward: float | None = None  # training reward for unparseable (e.g. truncated) outputs; None = the verifier's -0.1
    temperature: float = 1.0
    max_new_tokens: int = 48
    few_shot: int = 3
    prompt_style: str = "v1"  # countdown.PROMPT_STYLES; used for training rollouts
    eval_prompt_style: str = ""  # prompt for the in-loop eval; "" = same as prompt_style
    max_steps: int = 100
    time_budget_s: float = 0.0  # if > 0, stop once training time (wall clock minus evals) reaches this, then eval once
    eval_every: int = 10
    eval_start_step: int = 0
    eval_limit: int = 200
    eval_batch_size: int = 32
    target_solve_rate: float = 1.0
    grad_clip: float = 1.0
    micro_batch_size: int = 16
    micro_batch_tokens: int = 0  # if > 0, length-sorted micro-batches of up to this many padded tokens (overrides micro_batch_size)
    grad_checkpoint: bool = False  # activation checkpointing in the training forward (less memory, ~30 % more compute)
    curriculum_steps: int = 0
    # eval set; a ladder file (rlvr_speedrun/ladder.py) is truncated per level and adds "ladder_score"
    eval_set: str = "data/countdown_eval.jsonl"
    eval_samples: int = 0  # 0 = greedy; k > 0 = mean pass rate of k samples at T=1 per puzzle (vLLM backend)
    train_mix: str = ""  # "" = the generator's 4-number mix; "ladder" = uniform over the ladder levels
    train_levels: str = ""  # comma-separated countdown.LEVELS to train on (e.g. "+-*,+-*/"); empty = the generator's full mix
    save_every: int = 0
    # resumable state (weights, optimizer, RNG, logs) in resume_dir every resume_every_s training
    # seconds; a run started with the same resume_dir continues from it (e.g. after preemption)
    resume_dir: str = ""
    resume_every_s: float = 0.0
    sample_every: int = 0  # write every kept group's rollouts to samples.jsonl every N steps (analysis only)
    seed: int = 0
    device: str = "auto"
    dtype: str | None = None
    # fp32 weights and AdamW state with a bf16-autocast forward/backward; off = pure-bf16 AdamW, where
    # updates smaller than ~1/512 of a weight round away (as in the v3 records)
    master_fp32: bool = False
    out_dir: str = "runs/grpo"
    no_save: bool = False


def lr_at(config: GRPOConfig, train_s: float) -> float:
    """The learning rate after `train_s` seconds of training (`GRPOConfig.lr_schedule`)."""
    if config.lr_schedule not in ("constant", "cosine", "linear"):
        raise ValueError(f"unknown lr_schedule {config.lr_schedule!r} (expected 'constant', 'cosine' or 'linear')")
    if config.lr_warmup_s > 0 and train_s < config.lr_warmup_s:
        return config.lr * max(0.0, train_s) / config.lr_warmup_s
    if config.lr_schedule == "constant":
        return config.lr
    if config.time_budget_s <= config.lr_warmup_s:
        raise ValueError(f"lr_schedule {config.lr_schedule!r} needs time_budget_s > lr_warmup_s")
    frac = min(1.0, (train_s - config.lr_warmup_s) / (config.time_budget_s - config.lr_warmup_s))
    floor = config.lr * config.lr_final_frac
    if config.lr_schedule == "cosine":
        return floor + (config.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
    return config.lr - (config.lr - floor) * frac


def _set_seed(seed: int, rank: int = 0) -> None:
    """Seed all RNGs; ranks > 0 get a distinct torch seed so their rollouts are independent draws."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed + 1_000_003 * rank)


def _completion_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    temperature: float = 1.0,
    prefix_len: int = 0,
    allowed=None,
):
    """Per-token log-probs (at the sampling temperature) and entropies of completion tokens.

    `allowed` (per row, per completion position: allowed token ids or None) renormalises over the
    tokens the constrained sampler could pick, so log-probs are those of the constrained policy.

    Positions follow the attention mask (as in `generate`), so padding may sit anywhere.
    If `prefix_len > 0`, columns `[:prefix_len]` are identical and fully attended in
    every row: that prefix is run once and its KV cache broadcast to the batch, which
    is exact under causal attention and skips ~`(rows-1) * prefix_len` tokens of
    forward/backward compute.
    """
    start = max(0, prompt_width - 1)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    if 0 < prefix_len <= start:
        prefix = model.base_model(input_ids=input_ids[:1, :prefix_len], position_ids=position_ids[:1, :prefix_len], use_cache=True)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(input_ids.shape[0])
        hidden = model.base_model(
            input_ids=input_ids[:, prefix_len:],
            attention_mask=attention_mask,
            position_ids=position_ids[:, prefix_len:],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        hidden = hidden[:, start - prefix_len : -1]
    else:
        hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).last_hidden_state
        hidden = hidden[:, start:-1]
    labels = input_ids[:, start + 1 :]
    completion_mask = attention_mask[:, start + 1 :].bool()
    # the vocabulary projection only for real completion tokens (not padding): with long,
    # right-padded rollouts most of the rectangle is padding and the fp32 logits dominate memory
    logits = model.get_output_embeddings()(hidden[completion_mask]).float() / temperature
    if allowed is not None:
        pairs, k = [], 0
        for row, n in enumerate(completion_mask.sum(1).tolist()):
            pairs.extend((k + j, ids) for j, ids in enumerate(allowed[row][:n]) if ids is not None)
            k += n
        logits = mask_logits(logits, pairs)
    log_z = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, labels[completion_mask].unsqueeze(-1)).squeeze(-1) - log_z
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        ent = log_z - (probs * logits.masked_fill(probs == 0, 0.0)).sum(-1)
    token_lp = torch.zeros(labels.shape, dtype=lp.dtype, device=lp.device)
    token_lp[completion_mask] = lp
    entropy = torch.zeros(labels.shape, dtype=ent.dtype, device=ent.device)
    entropy[completion_mask] = ent
    return token_lp, completion_mask, entropy


class StaticCachePool:
    """One persistent `StaticCache` per batch size, reset between `generate` calls.

    HF's built-in `cache_implementation="static"` rebuilds the cache whenever the
    batch size changes, and every fresh cache object re-triggers `torch.compile`
    (~25 s on an H100); with alternating rollout/eval batch sizes that happens every
    switch. Reusing the same objects keeps the compiled graphs' guards valid.

    `min_len` (config.cache_len) allocates every cache at least that long up front.
    Otherwise a longer prompt than any seen so far replaces the cache, and the compiled
    graphs of the old one may stay alive (suspected cause of the ~20 GB growth and OOM in the
    2026-09 L40S plateau runs).
    """

    def __init__(self, model, min_len: int = 0):
        self.model = model
        self.min_len = min_len
        self._caches: dict[int, StaticCache] = {}

    def get(self, rows: int, max_cache_len: int) -> StaticCache:
        max_cache_len = max(max_cache_len, self.min_len)
        cache = self._caches.get(rows)
        if cache is None or cache.max_cache_len < max_cache_len:
            cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
            self._caches[rows] = cache
        else:
            cache.reset()
        return cache


def _cache_kwargs(caches: StaticCachePool | None, rows: int, prompt_width: int, config: GRPOConfig) -> dict:
    if caches is None:
        return {}
    return {"past_key_values": caches.get(rows, prompt_width + config.max_new_tokens)}


def _generate_chunked(model, tokenizer, encoded, config: GRPOConfig, caches: StaticCachePool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `group_size` completions per prompt; returns (sequences, finished_at).

    Sampling is from the full tempered softmax (`top_k=0`, `top_p=1.0`; some
    transformers versions default to `top_k=50`) so the sampler matches the policy
    log-probs used in the loss.

    `finished_at[i]` is the generated length of row i: the position at which it
    emitted `</answer>`, or its chunk's generated width if it never did. Tokens at
    or beyond it (forced pads, cross-chunk padding) are excluded from the loss.
    """
    n_prompts = encoded["input_ids"].shape[0]
    chunk = config.gen_batch_size if config.gen_batch_size > 0 else n_prompts
    outputs, finished = [], []
    with torch.no_grad():
        for i in range(0, n_prompts, chunk):
            criteria = answer_stopping_criteria(tokenizer)
            ids = encoded["input_ids"][i : i + chunk]
            outputs.append(
                model.generate(
                    input_ids=ids,
                    attention_mask=encoded["attention_mask"][i : i + chunk],
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_k=0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=config.group_size,
                    stopping_criteria=criteria,
                    **_cache_kwargs(caches, ids.shape[0] * config.group_size, ids.shape[1], config),
                )
            )
            rows, gen_len = outputs[-1].shape[0], outputs[-1].shape[1] - ids.shape[1]
            finished_at = criteria[0].finished_at
            if finished_at is None:
                finished_at = torch.full((rows,), -1, dtype=torch.long, device=outputs[-1].device)
            finished.append(torch.where(finished_at >= 0, finished_at, torch.full_like(finished_at, gen_len)))
    width = max(out.shape[1] for out in outputs)
    padded = [torch.nn.functional.pad(out, (0, width - out.shape[1]), value=tokenizer.pad_token_id) for out in outputs]
    return torch.cat(padded, dim=0), torch.cat(finished, dim=0)


def select_groups(rewards: torch.Tensor, keep: int) -> torch.Tensor:
    """Return indices of the `keep` most informative groups (rows of `rewards`).

    Groups with reward variance carry gradient; among those, groups containing a
    correct solve rank above groups whose only variance is malformed-vs-wrong.
    Zero-variance groups are used last (their advantages are zero).
    """
    n = rewards.shape[0]
    if keep >= n:
        return torch.arange(n, device=rewards.device)
    stds = rewards.std(dim=1, unbiased=False)
    solved = (rewards.max(dim=1).values >= 1.0).float()
    informative = (stds > 0).float()
    score = informative * (1.0 + solved) * 10.0 + stds
    return torch.argsort(score, descending=True, stable=True)[:keep]


def group_advantages(groups: torch.Tensor, adv_norm: str) -> torch.Tensor:
    """Group-relative advantages: `std` is GRPO's per-group z-score, `none` is the
    Dr. GRPO variant (mean-centred only), which removes the difficulty bias that
    up-weights groups where only one or two samples disagree with the rest."""
    centred = groups - groups.mean(dim=1, keepdim=True)
    if adv_norm == "none":
        return centred
    if adv_norm != "std":
        raise ValueError(f"unknown adv_norm {adv_norm!r} (expected 'std' or 'none')")
    stds = groups.std(dim=1, unbiased=False, keepdim=True)
    return torch.where(stds > 0, centred / (stds + 1e-4), torch.zeros_like(centred))


_SPLIT_IS_EXACT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _encode_prompts(tokenizer, puzzles, config: GRPOConfig, device: torch.device, style: str | None = None) -> tuple[dict, int]:
    """Tokenize prompts as `[shared prefix][pad...][puzzle suffix]`; returns (encoding, prefix_len).

    Padding between prefix and suffix (rather than on the left) keeps the prefix
    column-aligned across rows. With mask-derived positions this is equivalent to
    left padding for generation, and it lets training reuse the prefix's KV cache.
    Falls back to plain left padding (prefix_len 0) if the tokenizer does not split
    cleanly at the prefix boundary.
    """
    style = style or config.prompt_style
    parts = [split_prompt(puzzle, few_shot=config.few_shot, style=style) for puzzle in puzzles]
    prefix_ids = tokenizer.encode(parts[0][0])
    suffix_ids = [tokenizer.encode(suffix, add_special_tokens=False) for _, suffix in parts]
    exact = _SPLIT_IS_EXACT.setdefault(tokenizer, {})
    if style not in exact:
        exact[style] = tokenizer.encode(parts[0][0] + parts[0][1]) == prefix_ids + suffix_ids[0]
    multiple = max(1, config.pad_to_multiple)
    if not exact[style]:
        prompts = [prefix + suffix for prefix, suffix in parts]
        kwargs = {"pad_to_multiple_of": multiple} if multiple > 1 else {}
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, **kwargs)
        return {"input_ids": encoded["input_ids"].to(device), "attention_mask": encoded["attention_mask"].to(device)}, 0
    width = len(prefix_ids) + max(len(ids) for ids in suffix_ids)
    width = -(-width // multiple) * multiple
    pad = tokenizer.pad_token_id
    rows, mask = [], []
    for ids in suffix_ids:
        fill = width - len(prefix_ids) - len(ids)
        rows.append(prefix_ids + [pad] * fill + ids)
        mask.append([1] * len(prefix_ids) + [0] * fill + [1] * len(ids))
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }, len(prefix_ids)


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _is_vllm(engine) -> bool:
    return type(engine).__name__ == "VLLMRollout"


def _sample_batch(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device, caches: StaticCachePool | None = None):
    encoded, prefix_len = _encode_prompts(tokenizer, puzzles, config, device)
    if _is_vllm(caches):
        generated, finished_at = caches.generate(encoded, config.group_size, config.temperature, config.max_new_tokens, tokenizer.pad_token_id,
                                                 puzzles=puzzles if config.constrain_answer else None, think_budget=config.think_budget)
    else:
        assert not config.constrain_answer, "constrain_answer needs rollout_backend='vllm'"
        generated, finished_at = _generate_chunked(model, tokenizer, encoded, config, caches)
    prompt_width = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    expanded = [puzzle for puzzle in puzzles for _ in range(config.group_size)]
    results = [verify(text, puzzle) for text, puzzle in zip(completions, expanded)]
    if config.malformed_reward is None:
        values = [r.reward for r in results]
    else:
        values = [config.malformed_reward if r.malformed else r.reward for r in results]
    if config.shaping:
        values = [v + config.shaping * partial_credit(r, p) for v, r, p in zip(values, results, expanded)]
    rewards = torch.tensor(values, dtype=torch.float32, device=device)
    prompt_attention = encoded["attention_mask"].repeat_interleave(config.group_size, dim=0)
    completion = generated[:, prompt_width:]
    positions = torch.arange(completion.shape[1], device=device).unsqueeze(0)
    valid = positions < finished_at.unsqueeze(1)
    if tokenizer.eos_token_id is not None and completion.shape[1]:
        eos_positions = torch.where(
            completion.eq(tokenizer.eos_token_id),
            positions,
            torch.full_like(positions, completion.shape[1]),
        )
        first_eos = eos_positions.min(dim=1, keepdim=True).values
        valid &= positions <= first_eos
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            valid &= completion.ne(tokenizer.pad_token_id)
    completion_attention = valid.to(prompt_attention.dtype)
    attention = torch.cat([prompt_attention, completion_attention], dim=1)
    malformed = torch.tensor([r.malformed for r in results], dtype=torch.bool, device=device)
    return generated, attention, completions, expanded, rewards, prompt_width, prefix_len, malformed


def _micro_batches(completion_valid: torch.Tensor, prompt_width: int, config: GRPOConfig) -> list[torch.Tensor]:
    """Row groups for gradient accumulation; each is trimmed to its own longest completion.

    Fixed `micro_batch_size` rows in order by default. With `micro_batch_tokens > 0`, rows are
    sorted by completion length and packed greedily up to that many (padded) tokens per group,
    so short rollouts share a micro-batch instead of being padded to the longest in the batch.
    The loss is a sum over tokens / the global token count, so the grouping does not change it.
    """
    n = completion_valid.shape[0]
    if config.micro_batch_tokens <= 0:
        micro = max(1, config.micro_batch_size)
        return [torch.arange(i, min(i + micro, n), device=completion_valid.device) for i in range(0, n, micro)]
    lengths = completion_valid.sum(1)
    order = torch.argsort(lengths, descending=True).tolist()
    lengths = lengths.tolist()
    groups, current = [], []
    for row in order:
        width = prompt_width + (lengths[current[0]] if current else lengths[row])  # first row is the longest
        if current and (len(current) + 1) * width > config.micro_batch_tokens:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return [torch.tensor(g, device=completion_valid.device) for g in groups]


def _cov_clip(token_lp: torch.Tensor, token_adv: torch.Tensor, frac: float) -> torch.Tensor:
    """Clip-Cov (Cui et al. 2025): no gradient through the `frac` of tokens with the largest
    covariance between log-prob and advantage, the updates that drive entropy collapse.
    Covariance is centred on this micro-batch's means."""
    with torch.no_grad():
        cov = (token_lp - token_lp.mean()) * (token_adv - token_adv.mean())
        k = max(1, int(frac * cov.numel()))
        clip = torch.zeros_like(cov, dtype=torch.bool)
        clip[torch.topk(cov, k).indices] = True
    return torch.where(clip, token_lp.detach(), token_lp)


def _update(
    model,
    ref_model,
    generated: torch.Tensor,
    attention: torch.Tensor,
    advantages: torch.Tensor,
    prompt_width: int,
    prefix_len: int,
    config: GRPOConfig,
    optimizer,
    device: torch.device,
    dist: Dist,
    allowed=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One optimizer step on this rank's rows; returns (loss, kl, entropy_sum, token_count) sums.

    Under data parallelism the loss is normalised by the global completion-token
    count and gradients are summed across ranks, so the step equals the
    single-process step on the concatenated batch.
    """
    completion_valid = attention[:, prompt_width:].bool()
    token_count = dist.all_reduce_sum_(completion_valid.sum()).clamp_min(1)
    optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)
    entropy_total = torch.zeros((), device=device)
    for rows in _micro_batches(completion_valid, prompt_width, config):
        width = prompt_width + int(completion_valid[rows].any(0).nonzero().max().item()) + 1 if completion_valid[rows].any() else prompt_width + 1
        ids, mask, adv = generated[rows, :width], attention[rows, :width], advantages[rows]
        if config.grad_checkpoint:  # HF checkpoints only in train mode, and then drops the KV cache the prefix trick needs
            model.train()
        rows_allowed = None if allowed is None else [allowed[r] for r in rows.tolist()]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.master_fp32):
            policy_lp, completion_mask, entropy = _completion_logprobs(
                model, ids, mask, prompt_width, config.temperature, 0 if config.grad_checkpoint else prefix_len, rows_allowed
            )
        model.eval()
        policy_lp = policy_lp[completion_mask]
        entropy_total += entropy[completion_mask].sum()
        repeated_advantages = adv.unsqueeze(1).expand(-1, completion_mask.shape[1])[completion_mask]
        if config.cov_clip_frac > 0 and policy_lp.numel() > 1:
            policy_lp = _cov_clip(policy_lp, repeated_advantages, config.cov_clip_frac)
        policy_loss = -(repeated_advantages * policy_lp).sum() / token_count
        if ref_model is None:
            kl = torch.zeros((), device=device)
            loss = policy_loss
        else:
            with torch.no_grad():
                ref_lp, _, _ = _completion_logprobs(ref_model, ids, mask, prompt_width, config.temperature, prefix_len, rows_allowed)
            ref_lp = ref_lp[completion_mask]
            kl_tokens = torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
            kl = kl_tokens.sum() / token_count
            loss = policy_loss + config.kl_coef * kl
        loss.backward()
        loss_total += loss.detach()
        kl_total += kl.detach()
    dist.all_reduce_grads_(model.parameters())
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return loss_total, kl_total, entropy_total, token_count


_GRAMMARS: dict = {}


def _allowed_tokens(tokenizer, generated, attention, prompt_width, puzzles, think_budget: int = 0) -> list:
    """Per row, per completion token: the ids the constrained sampler allowed there (None = free text)."""
    key = tokenizer.name_or_path
    if key not in _GRAMMARS:
        _GRAMMARS[key] = AnswerGrammar(token_strings(tokenizer), open_token_ids(tokenizer))
    grammar = _GRAMMARS[key]
    valid = attention[:, prompt_width:].bool()
    completion = generated[:, prompt_width:]
    return [grammar.allowed_per_position(completion[r][valid[r]].tolist(), p.numbers, p.ops, think_budget) for r, p in enumerate(puzzles)]


def grpo_step(
    model,
    ref_model,
    tokenizer,
    puzzles,
    config: GRPOConfig,
    optimizer,
    device: torch.device,
    caches: StaticCachePool | None = None,
    dist: Dist = SINGLE,
) -> dict:
    """One GRPO step on `puzzles` (this rank's shard of the global prompt batch)."""
    t0 = _now(device)
    generated, attention, completions, expanded, rewards, prompt_width, prefix_len, malformed = _sample_batch(
        model, tokenizer, puzzles, config, device, caches
    )
    t_sample = _now(device) - t0
    all_rewards = rewards
    all_groups = all_rewards.reshape(len(puzzles), config.group_size)
    informative_groups = int((all_groups.std(dim=1, unbiased=False) > 0).sum().item())
    kept = select_groups(all_groups, config.prompts_per_step // dist.world_size)
    if kept.shape[0] < len(puzzles):
        rows = (kept.unsqueeze(1) * config.group_size + torch.arange(config.group_size, device=kept.device)).reshape(-1)
        generated, attention, rewards = generated[rows], attention[rows], rewards[rows]
        row_list = rows.tolist()
        completions = [completions[r] for r in row_list]
        expanded = [expanded[r] for r in row_list]
    groups = rewards.reshape(kept.shape[0], config.group_size)
    advantages = group_advantages(groups, config.adv_norm).reshape(-1)

    allowed = _allowed_tokens(tokenizer, generated, attention, prompt_width, expanded, config.think_budget) if config.constrain_answer else None
    loss, kl, entropy_total, token_count = _update(
        model, ref_model, generated, attention, advantages, prompt_width, prefix_len, config, optimizer, device, dist, allowed
    )
    t_update = _now(device) - t0 - t_sample
    # global batch statistics: sums over ranks of [reward, solved, malformed, samples, informative, kept, entropy, kl, loss]
    stats = dist.all_reduce_sum_(
        torch.stack(
            [
                all_rewards.sum(),
                (all_rewards == 1.0).float().sum(),
                malformed.float().sum(),
                torch.tensor(float(all_rewards.numel()), device=device),
                torch.tensor(float(informative_groups), device=device),
                torch.tensor(float(kept.shape[0]), device=device),
                entropy_total,
                kl.detach(),
                loss.detach(),
            ]
        )
    ).tolist()
    reward_sum, solved, malformed, n_samples, informative_sum, kept_sum, entropy_sum, kl_sum, loss_sum = stats
    return {
        "mean_reward": reward_sum / n_samples,
        "solve_rate_in_batch": solved / n_samples,
        "malformed_rate": malformed / n_samples,
        "informative_groups": int(informative_sum),
        "kept_groups": int(kept_sum),
        "entropy": entropy_sum / token_count.item(),
        "kl": kl_sum,
        "loss": loss_sum,
        "tokens": int(token_count.item()),
        "t_sample": t_sample,
        "t_update": t_update,
        "completions": completions,
        "rewards": rewards.detach().cpu().tolist(),
        "puzzles": expanded,
    }


def _evaluate(
    model,
    tokenizer,
    puzzles,
    config: GRPOConfig,
    device: torch.device,
    caches: StaticCachePool | None = None,
    dist: Dist = SINGLE,
) -> dict:
    """Greedy pass rate on `puzzles`; under data parallelism each rank decodes its shard and all ranks get the full result."""
    local = dist.shard(puzzles)
    local_completions = []
    with torch.no_grad():
        for start in range(0, len(local), config.eval_batch_size):
            batch = local[start : start + config.eval_batch_size]
            n_real = len(batch)
            if config.compile and n_real < config.eval_batch_size:
                batch = batch + [batch[0]] * (config.eval_batch_size - n_real)
            encoded, _ = _encode_prompts(tokenizer, batch, config, device, config.eval_prompt_style or config.prompt_style)
            if _is_vllm(caches):
                k = max(1, config.eval_samples)
                generated, _ = caches.generate(encoded, k, 1.0 if config.eval_samples else 0.0, config.max_new_tokens, tokenizer.pad_token_id,
                                               puzzles=batch if config.constrain_answer else None, think_budget=config.think_budget)
                texts = tokenizer.batch_decode(generated[: n_real * k, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
                local_completions.extend(texts[i * k : (i + 1) * k] if config.eval_samples else texts[i] for i in range(n_real))
                continue
            assert not config.eval_samples, "eval_samples needs rollout_backend='vllm'"
            assert not config.constrain_answer, "constrain_answer needs rollout_backend='vllm'"
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=answer_stopping_criteria(tokenizer),
                **_cache_kwargs(caches, encoded["input_ids"].shape[0], encoded["input_ids"].shape[1], config),
            )
            local_completions.extend(tokenizer.batch_decode(generated[:n_real, encoded["input_ids"].shape[1] :], skip_special_tokens=True))
    gathered = [item for part in dist.all_gather_object(local_completions) for item in part]
    samples = gathered if config.eval_samples else [[text] for text in gathered]  # k completions per puzzle
    result = aggregate_results([t for texts in samples for t in texts], [p for p, texts in zip(puzzles, samples) for _ in texts])
    if config.eval_samples:
        result["samples_per_puzzle"] = config.eval_samples
    all_completions = [texts[0] for texts in samples]
    by_level: dict[str, list[float]] = {}
    for puzzle, texts in zip(puzzles, samples):
        by_level.setdefault(puzzle.level or level(puzzle), []).append(sum(verify(t, puzzle).correct for t in texts) / len(texts))
    order = LADDER_LEVELS + LEVELS
    result["pass_by_level"] = {lv: sum(v) / len(v) for lv, v in sorted(by_level.items(), key=lambda kv: order.index(kv[0]))}
    if any(puzzle.level for puzzle in puzzles):  # ladder: every level weighs the same
        result["ladder_score"] = sum(result["pass_by_level"].values()) / len(result["pass_by_level"])
    result["examples"] = [
        {
            "puzzle": puzzle.to_dict(),
            "completion": completion,
            "reward": verify(completion, puzzle).reward,
        }
        for puzzle, completion in list(zip(puzzles, all_completions))[:2]
    ]
    return result


def environment_info(device: torch.device, world_size: int = 1, model=None) -> dict:
    """Hardware/software identity recorded in result.json (RULES.md, 'Reproducibility')."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "world_size": world_size,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "git_commit": commit,
        # Hugging Face commit the base weights were loaded from (None for a local path)
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
    }


def _warmup(
    model, tokenizer, config: GRPOConfig, eval_puzzles, device: torch.device, caches: StaticCachePool, dist: Dist = SINGLE
) -> None:
    """Untimed warm-up: compile the decode graph for the rollout and eval batch shapes.

    Uses a puzzle stream disjoint from training and performs no optimizer step, and
    restores the RNG state afterwards, so the policy and the seeded sampling sequence
    are unchanged when the clock starts.
    """
    warm_stream = train_puzzle_stream(config.seed + 700_000)
    rollout = dist.shard([next(warm_stream) for _ in range(config.prompts_per_step * max(1, config.oversample))])
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        _sample_batch(model, tokenizer, rollout, config, device, caches)
        _evaluate(model, tokenizer, eval_puzzles[: config.eval_batch_size * dist.world_size], config, device, caches, dist)


def _check_shardable(config: GRPOConfig, dist: Dist) -> None:
    if config.prompts_per_step % dist.world_size:
        raise ValueError(f"prompts_per_step={config.prompts_per_step} must be divisible by the number of ranks ({dist.world_size})")


def partial_credit(result, puzzle) -> float:
    """In [0, 1) for a well-formed wrong answer (0 for correct or malformed ones). An expression that
    uses exactly the puzzle's numbers earns 0.5 plus up to 0.5 for closeness to the target; one with
    two or more numbers but the wrong multiset earns 0.25 x their overlap; a bare number earns nothing.
    Track D pilots hacked looser versions within ~100 steps by answering the target itself or one
    puzzle number (docs/v3_design.md)."""
    if result.correct or result.malformed or result.expression is None:
        return 0.0
    try:
        value, leaves = safe_eval(result.expression)
    except ValueError:
        return 0.0
    if sorted(leaves) == sorted(puzzle.numbers):
        closeness = max(0.0, 1.0 - abs(float(value) - puzzle.target) / max(abs(puzzle.target), 1))
        return min(0.99, 0.5 + 0.5 * closeness)
    if len(leaves) < 2:
        return 0.0
    common = sum((Counter(leaves) & Counter(puzzle.numbers)).values())
    return 0.25 * common / max(len(leaves), len(puzzle.numbers))


_RESUME_LOGS = ("train_log.jsonl", "eval_log.jsonl", "samples.jsonl")


def _load_eval_set(config: GRPOConfig):
    puzzles = load_puzzles(config.eval_set)
    if not any(p.level for p in puzzles):
        return puzzles[: config.eval_limit]
    per_level = max(1, config.eval_limit // len({p.level for p in puzzles}))  # truncate each level, not the file
    counts: dict[str, int] = {}
    kept = []
    for p in puzzles:
        counts[p.level] = counts.get(p.level, 0) + 1
        if counts[p.level] <= per_level:
            kept.append(p)
    return kept


def _train_streams(config: GRPOConfig, offset: int = 0):
    """The main training stream (filtered to `train_levels`) and the two easy curriculum streams."""
    if config.train_mix == "ladder":
        stream = ladder_stream(config.seed + offset * 10_000_000, exclude=ladder_keys())
    elif config.train_mix:
        raise ValueError(f"unknown train_mix {config.train_mix!r} (expected '' or 'ladder')")
    else:
        stream = train_puzzle_stream(config.seed + offset * 10_000_000)
    if config.train_levels:
        keep = set(config.train_levels.split(","))
        assert keep <= set(LEVELS), f"train_levels must be a subset of {LEVELS}"
        stream = (p for p in stream if level(p) in keep)
    easy_streams = [
        train_puzzle_stream(config.seed + 500_000 + offset * 10_000_000, n_numbers=2, target_range=(1, 50)),
        train_puzzle_stream(config.seed + 600_000 + offset * 10_000_000, n_numbers=3, target_range=(1, 50)),
    ]
    return stream, easy_streams


def _rng_state(mix_rng: random.Random, device: torch.device) -> dict:
    return {"mix": mix_rng.getstate(), "random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _set_rng_state(state: dict, mix_rng: random.Random, device: torch.device) -> None:
    mix_rng.setstate(state["mix"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _save_state(state_path: Path, out_dir: Path, model, optimizer, mix_rng, device, progress: dict) -> None:
    """Write logs then state.pt, each atomically; on resume, log lines past the state's step are dropped."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    for name in _RESUME_LOGS:
        if (out_dir / name).exists():
            tmp = state_path.parent / (name + ".tmp")
            shutil.copyfile(out_dir / name, tmp)
            os.replace(tmp, state_path.parent / name)
    tmp = state_path.with_suffix(".tmp")
    torch.save({**progress, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(mix_rng, device)}, tmp)
    os.replace(tmp, state_path)


def run(config: GRPOConfig, dist: Dist = SINGLE) -> dict:
    """Train one seed; with `dist.world_size > 1` every rank calls this and only rank 0 writes `out_dir`."""
    _check_shardable(config, dist)
    scheduled = config.lr_schedule != "constant" or config.lr_warmup_s > 0
    lr_at(config, config.lr_warmup_s)  # fail on a bad schedule before loading anything
    _set_seed(config.seed, dist.rank)
    out_dir = Path(config.out_dir)
    if dist.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    tokenizer = load_tokenizer(config.model)
    engine = None
    if config.rollout_backend == "vllm":  # before the training model, so vLLM's memory profiling sees an empty GPU
        from .vllm_rollout import VLLMRollout

        engine = VLLMRollout(config.model, 512 + config.max_new_tokens, config.vllm_gpu_memory, config.seed + 1_000_003 * dist.rank,
                             constrain_answer=config.constrain_answer)
    elif config.rollout_backend != "hf":
        raise ValueError(f"unknown rollout_backend {config.rollout_backend!r} (expected 'hf' or 'vllm')")
    model, device = load_causal_model(config.model, config.device, "fp32" if config.master_fp32 else config.dtype)
    if config.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    ref_model = None
    if config.kl_coef != 0:
        ref_name = config.ref_model or config.model
        ref_model, _ = load_causal_model(ref_name, str(device), config.dtype)
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    stream, easy_streams = _train_streams(config)
    mix_rng = random.Random(config.seed)
    state_path = Path(config.resume_dir) / "state.pt" if config.resume_dir else None
    resumed = None
    if state_path is not None and state_path.exists():
        resumed = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        _set_rng_state(resumed["rng"], mix_rng, device)
        # fresh puzzle streams rather than replaying every draw consumed so far
        stream, easy_streams = _train_streams(config, offset=1_000 + resumed["step"])
        if dist.is_main:
            for name in _RESUME_LOGS:
                src = Path(config.resume_dir) / name
                lines = src.read_text(encoding="utf-8").splitlines(keepends=True) if src.exists() else []
                (out_dir / name).write_text("".join(x for x in lines if json.loads(x)["step"] <= resumed["step"]), encoding="utf-8")
            print(f"resumed from {state_path} at step {resumed['step']}", flush=True)
    eval_puzzles = _load_eval_set(config)
    log_mode = "a" if resumed else "w"
    train_log = (out_dir / "train_log.jsonl").open(log_mode, encoding="utf-8") if dist.is_main else None
    eval_log = (out_dir / "eval_log.jsonl").open(log_mode, encoding="utf-8") if dist.is_main else None
    caches = engine if engine is not None else StaticCachePool(model, config.cache_len) if config.compile else None
    warmup_s = 0.0
    if caches is not None and engine is None:
        warm_start = time.perf_counter()
        _warmup(model, tokenizer, config, eval_puzzles, device, caches, dist)
        warmup_s = time.perf_counter() - warm_start
        if dist.is_main:
            print(f"warmup (untimed): {warmup_s:.1f}s", flush=True)
    dist.barrier()
    start = time.perf_counter()
    threshold_time = None
    total_steps = 0
    final_pass_rate = 0.0
    eval_s = 0.0  # wall-clock time spent in evals (and resume saves); train_s = wall_s - eval_s
    first_step = 1
    if resumed:
        start -= resumed["wall_s"]
        eval_s, total_steps, final_pass_rate = resumed["eval_s"], resumed["step"], resumed["final_pass_rate"]
        first_step = resumed["step"] + 1
        if engine is not None:
            engine.sync(model)
    last_save = time.perf_counter() - start - eval_s
    try:
        for step in range(first_step, config.max_steps + 1):
            easy_frac = max(0.0, 1.0 - step / config.curriculum_steps) if config.curriculum_steps > 0 else 0.0
            puzzles = []
            for _ in range(config.prompts_per_step * max(1, config.oversample)):
                u = mix_rng.random()
                source = stream if u >= easy_frac else easy_streams[int(u >= easy_frac / 2)]
                puzzles.append(next(source))
            if scheduled:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(config, time.perf_counter() - start - eval_s)
            step_start = time.perf_counter()
            info = grpo_step(model, ref_model, tokenizer, dist.shard(puzzles), config, optimizer, device, caches, dist)
            if engine is not None:
                engine.sync(model)
            step_elapsed = time.perf_counter() - step_start
            elapsed = time.perf_counter() - start
            total_steps = step
            tokens = info.pop("tokens")
            if config.sample_every and step % config.sample_every == 0 and dist.is_main:
                with (out_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
                    for text, r, puzzle in zip(info["completions"], info["rewards"], info["puzzles"]):
                        f.write(json.dumps({"step": step, "numbers": list(puzzle.numbers), "target": puzzle.target,
                                            "reward": r, "completion": text}) + "\n")
            info = {key: value for key, value in info.items() if key not in {"completions", "rewards", "puzzles"}}
            info.update({"step": step, "wall_s": elapsed, "train_s": elapsed - eval_s, "tokens/s": tokens / max(step_elapsed, 1e-9)})
            if scheduled:
                info["lr"] = optimizer.param_groups[0]["lr"]
            out_of_time = False
            if config.time_budget_s > 0:  # rank 0 decides so all ranks stop on the same step
                flag = torch.tensor(float(dist.is_main and elapsed - eval_s >= config.time_budget_s), device=device)
                out_of_time = dist.all_reduce_sum_(flag).item() > 0
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                info["gpu_mem_gb"] = (total - free) / 2**30  # whole-process use, incl. CUDA graph pools
            if train_log is not None:
                train_log.write(json.dumps(info) + "\n")
                train_log.flush()
            if config.save_every and step % config.save_every == 0 and dist.is_main:
                checkpoint = out_dir / f"ckpt_{step}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
            if (step >= config.eval_start_step and step % config.eval_every == 0) or step == config.max_steps or out_of_time:
                eval_start = time.perf_counter()
                result = _evaluate(model, tokenizer, eval_puzzles, config, device, caches, dist)
                eval_s += time.perf_counter() - eval_start
                result.update({"step": step, "wall_s": time.perf_counter() - start, "train_s": time.perf_counter() - start - eval_s})
                final_pass_rate = result["pass_rate"]
                if eval_log is not None:
                    eval_log.write(json.dumps(result) + "\n")
                    eval_log.flush()
                if threshold_time is None and result["pass_rate"] >= config.target_solve_rate:
                    threshold_time = result["wall_s"]
                    break
            train_s = time.perf_counter() - start - eval_s
            if (state_path is not None and config.resume_every_s > 0 and train_s - last_save >= config.resume_every_s
                    and not out_of_time and step < config.max_steps and dist.is_main):
                save_start = time.perf_counter()
                train_log.flush()
                eval_log.flush()
                _save_state(state_path, out_dir, model, optimizer, mix_rng, device, {
                    "step": step, "wall_s": save_start - start, "eval_s": eval_s, "final_pass_rate": final_pass_rate})
                eval_s += time.perf_counter() - save_start  # off the clock, like evals
                last_save = train_s
            if out_of_time:
                break
    finally:
        if train_log is not None:
            train_log.close()
        if eval_log is not None:
            eval_log.close()
    total_wall = time.perf_counter() - start
    if not config.no_save and dist.is_main:
        model.save_pretrained(out_dir / "final")
        tokenizer.save_pretrained(out_dir / "final")
    result = {
        "final_pass_rate": final_pass_rate,
        "pass_rate": final_pass_rate,
        "time_to_threshold_s": threshold_time,
        "total_steps": total_steps,
        "total_wall": total_wall,
        "warmup_s": warmup_s,
        "max_steps": config.max_steps,
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "environment": environment_info(device, dist.world_size, model),
    }
    if dist.is_main:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    return result


def _config_from_args(args) -> GRPOConfig:
    names = {field.name for field in fields(GRPOConfig)}
    values = {}
    if args.config:
        loaded = json.loads(Path(args.config).read_text(encoding="utf-8"))
        values.update({key: value for key, value in loaded.items() if key in names})
    for field in fields(GRPOConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    return GRPOConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    for field in fields(GRPOConfig):
        arg = "--" + field.name.replace("_", "-")
        if field.name in {"no_save", "compile"}:
            parser.add_argument(arg, action="store_true", default=None)
        else:
            parser.add_argument(arg, type=type(field.default) if field.default is not None else str, default=None)
    args = parser.parse_args()
    config = _config_from_args(args)
    if not config.model:
        parser.error("--model is required (or set it in --config)")
    dist, device = init_from_env(config.device)
    config.device = device
    try:
        result = run(config, dist)
    finally:
        destroy()
    if dist.is_main:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
