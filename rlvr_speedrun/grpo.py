"""A minimal, readable GRPO implementation using plain PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import time
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.checkpoint import checkpoint
from transformers import StaticCache

from . import benchmark
from .constrained import OPEN, AnswerGrammar, mask_logits, open_token_ids, token_strings
from .countdown import LEVELS, level, safe_eval, split_prompt, verify
from .data import load_puzzles, train_puzzle_stream
from .distributed import SINGLE, Dist, destroy, init_from_env
from .eval import aggregate_results
from .ladder import LEVEL_NAMES as LADDER_LEVELS
from .ladder import MINI_LEVEL_NAMES, ladder_keys, ladder_stream, level_by_name, puzzle_key
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
    # the update runs each decoder layer through torch.compile (`_compile_layers`): one set of graphs shared by the
    # layers, dynamic batch and length, the KV-cache update left eager; the same maths up to rounding, and the
    # compilation counts in the clock. Not with rollout_backend 'hf' or grad_checkpoint (LOG r123)
    compile_update: bool = False
    # the update builds every micro-batch's indices (rows, width, completion tokens, shared-prompt groups, the
    # constrained-logit mask) on the host at its start and sends them to the GPU in one copy, so the micro-batches
    # run without device syncs; the replayed rows' nll is read after the optimizer step. The same maths (LOG r124)
    update_host_index: bool = False
    # the update's model body (everything before the vocabulary projection) runs as CUDA graphs, captured once per
    # bucketed shape (`_UpdateCapture`) and replayed, with the gradients bound to fixed .grad tensors; the same maths
    # up to rounding, and the captures count in the clock. Needs update_host_index and micro_batch_tokens; not with
    # grad_checkpoint or compile_update (LOG r125)
    update_capture: bool = False
    # update_capture's buckets: width a multiple of update_capture_width_step, rows a multiple of
    # update_capture_row_step (0: steps of 8 up to 64 rows, 16 up to 128, 32 up to 256, then 64) and shared-prompt
    # groups a multiple of update_capture_group_step; finer steps pad less and capture more shapes (LOG r127)
    update_capture_width_step: int = 64
    update_capture_row_step: int = 0
    update_capture_group_step: int = 8
    pad_to_multiple: int = 0
    cache_len: int = 0
    rollout_backend: str = "hf"  # "hf" (model.generate), "vllm" (colocated engine; rlvr_speedrun/vllm_rollout.py) or "native" (fast_rollout.py, Llama only)
    vllm_gpu_memory: float = 0.3  # share of GPU memory given to the vLLM engine (weights + KV cache)
    # extra keyword arguments for vllm.LLM, as a JSON object (e.g. '{"async_scheduling": true}'; PLAN D7); "" = none
    vllm_engine_args: str = ""
    # constrain_answer's vLLM mask applied to the whole batch in one go instead of request by request (same mask; PLAN D7)
    constrain_batched: bool = False
    # rollout_backend 'native' with constrain_answer: the update takes the allowed token sets the sampler used at each
    # position instead of recomputing them from the tokens (`_allowed_tokens`); the same values, less Python per step
    reuse_sampler_allowed: bool = False
    # rollout_backend 'native' with constrain_answer: the sampler builds a new grammar state's allowed set from parts
    # instead of walking every candidate token (`AnswerGrammar(factored=True)`); the same sets, less CPU per step
    factored_grammar: bool = False
    # rollout_backend 'native' with constrain_answer: rows step through interned grammar states by (state, token)
    # lookups, and allowed sets live as rows of a device table that one gather masks with (`MaskTable`); the same
    # masks, less CPU per decoding step (LOG 2026-11-24, the s_mask_cpu item)
    grammar_fast_path: bool = False
    # the update runs each distinct prompt once (its KV cache indexed out to its group's rows) instead of once
    # per row; exact like the shared-prefix trick, fewer token positions per update (INBOX 11-17 14:59)
    share_group_prompt: bool = False
    # synced wall-clock timers around the parts of the RL update, logged as t_part_* (INBOX 11-17 17:14); the
    # synchronisations slow the step a little, so timing runs only
    time_update_parts: bool = False
    # train_mix 'ladder': the draw's "needs" test uses integer maths (`countdown.reaches`) instead of Fractions; the
    # same puzzles in the same order, less host time before each step (LOG 2026-11-25 00:23: 77 ms of a 413 ms step)
    fast_puzzle_draw: bool = False
    # draw the next step's puzzles on a background thread while this step runs, so the draw leaves the step's wall
    # time (INBOX 11-25 12:05); the same puzzles in the same order, and the mix rng is saved as without it
    prefetch_puzzle_draw: bool = False
    adv_norm: str = "std"
    # multiplies the positive advantages (W-REINFORCE, Zhu et al. 2025): below 1 damps the push toward sampled
    # solutions, which narrows the policy, and keeps the push away from failures; 1 = unchanged
    pos_adv_scale: float = 1.0
    lr: float = 1e-6
    # learning rate over training time (wall clock minus evals): a linear warmup over lr_warmup_s, then
    # "constant", or a "cosine" / "linear" decay to lr * lr_final_frac at time_budget_s
    lr_schedule: str = "constant"
    lr_warmup_s: float = 0.0
    lr_final_frac: float = 0.1
    kl_coef: float = 0.02
    # partial credit for well-formed wrong answers, up to this weight (see `partial_credit`); never searches for a solution
    shaping: float = 0.0
    # credit, up to this weight, for legal and correct Countdown moves written in <think> (see `process_credit`; PLAN D4)
    process_reward: float = 0.0
    # after "<answer>", sample only well-formed uses of the puzzle's numbers and operators (rlvr_speedrun/constrained.py; vLLM backend)
    constrain_answer: bool = False
    think_budget: int = 0  # with constrain_answer: force "<answer>" after this many free tokens (0 = never)
    cov_clip_frac: float = 0.0  # Clip-Cov: drop the gradient of this fraction of tokens with the largest log-prob x advantage covariance
    # if > 0, subtract entropy_coef x the mean token entropy of the updated rows from the loss (a bonus against the
    # policy sharpening), on steps after one whose logged entropy was below entropy_target (0 = on every step); the
    # first step after a start or resume has no logged entropy, so it runs without the bonus unless the target is 0
    entropy_coef: float = 0.0
    entropy_target: float = 0.0
    # update only the rows with a nonzero advantage: with kl_coef == 0 and no Clip-Cov the others carry no
    # gradient, and the loss keeps the full token count, so the step is the same, only faster
    skip_zero_advantage: bool = False
    # vLLM backend: also return the sampler's log-prob of each sampled token (after the answer constraint) and log
    # how far the training model's log-probs are from it (train_log "mismatch_*", over the updated rows); the
    # sampled tokens are unchanged
    log_mismatch: bool = False
    # truncated importance sampling (Yao et al. 2025): weight each token's policy-gradient term by
    # min(pi_train / pi_sampler, tis_cap) with the ratio detached, correcting for the sampler/trainer
    # mismatch; 0 = off. Implies log_mismatch
    tis_cap: float = 0.0
    malformed_reward: float | None = None  # training reward for unparseable (e.g. truncated) outputs; None = the verifier's -0.1
    malformed_reward_until: int = 0  # if > 0, malformed_reward applies before this step only, then the verifier's -0.1
    # training reward for malformed (e.g. truncated) outputs in groups where some sample is correct, overriding the
    # above there: with malformed_reward 0, e.g. -0.5 pushes finished searches over truncated ones only where the
    # policy can solve the puzzle, and never toward one quick guess in unsolved groups; None = off
    malformed_reward_solved: float | None = None
    # soft overlong penalty (DAPO, Yu et al. 2025): a training completion of L tokens loses
    # overlong_coef * clamp((L - (max_new_tokens - overlong_buffer)) / overlong_buffer, 0, 1) reward, so the
    # last overlong_buffer tokens before the cap cost up to overlong_coef; 0 = off
    overlong_buffer: int = 0
    overlong_coef: float = 0.0
    # the overlong penalty only on correct completions, so a long or truncated attempt never scores below a quick
    # wrong guess (r79: penalising truncated samples too collapsed track A into short guesses)
    overlong_correct_only: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 48
    # if > 0, the in-loop eval's token limit (needs cache_len 0); training rollouts keep max_new_tokens, so a run can
    # train under a shorter limit than it is scored with
    eval_max_new_tokens: int = 0
    few_shot: int = 3
    prompt_style: str = "v1"  # countdown.PROMPT_STYLES; used for training rollouts
    eval_prompt_style: str = ""  # prompt for the in-loop eval; "" = same as prompt_style
    max_steps: int = 100
    time_budget_s: float = 0.0  # if > 0, stop once training time (wall clock minus evals) reaches this, then eval once
    # if > 0, count the budget in steps instead of training time: stop after budget_steps steps (then eval once),
    # and read step s as (s - 1) / budget_steps of time_budget_s for the lr schedule, so a run takes the same
    # schedule and stops on the same step on any GPU or GPU share (step-capped screens); train_s still logs the
    # wall clock, and clock_s the budget's
    budget_steps: int = 0
    # with budget_steps: also stop once training time reaches time_budget_s, as a time-budgeted run does, so a
    # step-capped record still keeps to its clock
    budget_steps_time_stop: bool = False
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
    # with train_mix "ladder": comma-separated ladder levels to train on (e.g. "L2,L3,L4,L5,L6"); empty = all six. The
    # list may also name ladder.MINI_LEVELS (M2, M3, M6: 3-number training-only levels); the mix is uniform over the names
    train_ladder_levels: str = ""
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
    # RULES v3.4: end the run on the weights of its best in-loop eval (a tie goes to the later eval) instead of the
    # last step's. After each eval at least as good as the best so far the run copies its weights to CPU memory, and
    # it restores the kept copy before the final save. The evals and the copies then count toward the clock (train_s
    # includes them, so run_seed_v3.sh excludes nothing), and a time-budgeted run's final eval comes after the time
    # stop, so time_budget_s has to leave room for it. One process only, and the eval set may not be the test ladder
    keep_best_dev: bool = False
    # rollback (RULES v3.4; needs keep_best_dev): when an in-loop eval scores more than rollback_drop below the kept
    # best, training continues from the kept weights (the optimizer state carries on). 0 = off. The eval and the
    # restore count on the clock, like keep_best_dev's. Meant for the late-schedule dev slide (PLAN A8 (c))
    rollback_drop: float = 0.0
    # self-imitation (allowed with limits, docs/research/INBOX.md 2026-11-10): every verifier-correct sample of the
    # run's own rollouts goes into a buffer, as sampled and with the generator puzzle it answered, and each step
    # replays up to sil_rows of them (0 = off) as extra rows of the same optimizer step, each weighing like a
    # policy-gradient row with advantage sil_coef (plain log-likelihood: no entropy bonus, KL or importance weight).
    # The buffer starts empty in every run, is never saved (a resumed run refills it) and keeps the newest sil_buffer
    # entries per pool. sil_by_level: one pool per puzzle level label and draws spread evenly over the pools that
    # hold anything (a training mix over the generator's levels); off = one pool. No entry is replayed twice in a
    # step, so a step replays fewer rows while a pool is small. One process only
    sil_rows: int = 0
    sil_coef: float = 1.0
    # leave the tokens that spell a completion's first "<answer>" out of the policy-gradient term (their advantage is
    # set to 0; they still count in the token mean), so when to stop thinking and answer stays at the model's own
    # prior instead of being pushed by the reward. Entropy bonus, KL and self-imitation rows are unchanged. Off = as before
    mask_answer_open: bool = False
    sil_buffer: int = 4096
    sil_by_level: bool = True
    # hindsight relabelling. NOT ALLOWED in records, promotion screens or anything stacked into a ranked arm: the
    # maintainer has not ruled on it, and one unranked screen measures its effect (docs/research/INBOX.md 2026-11-10
    # 14:54, PLAN D12; validate_record_v3.py rejects a record that used it). A sample that is a well-formed
    # expression over the puzzle's own numbers and operators but misses the target goes into the buffer, tokens as
    # sampled, with the puzzle relabelled to the value it does reach, when that value is an integer inside the
    # level's generator target range and the relabelled puzzle is in neither ladder set. Nothing solves or searches
    # a relabelled pair (the level's "needs" rule is not checked on it). Each step replays up to hindsight_rows of
    # them (0 = off) the way self-imitation replays its rows, weighted hindsight_coef. The buffer is
    # self-imitation's (sil_buffer, sil_by_level; empty in every run, never saved). Needs the ladder training mix
    # and one process, and can't be combined with sil_rows. Meant for answer-only completions (track D): a think
    # section would still reason about the original target
    hindsight_rows: int = 0
    hindsight_coef: float = 1.0
    # pretraining during RL (maintainer idea U14, PLAN D9; track D): each step also takes lm_rows windows of
    # lm_seq_len + 1 tokens from lm_tokens (a uint16 token file as scripts/pretrain_scratch.py reads it, in the
    # policy's tokenizer) and adds lm_coef * their mean next-token cross-entropy to the step's loss (0 rows = off).
    # The window starts are drawn from (seed, step), so a resumed run draws the same ones with nothing saved.
    # One process only
    lm_rows: int = 0
    lm_coef: float = 1.0
    lm_seq_len: int = 1024
    lm_tokens: str = ""


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


def _entropy(logits: torch.Tensor, log_z: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) per row, given log_z = logsumexp(logits); -inf (masked) logits add nothing."""
    probs = torch.softmax(logits, dim=-1)
    return log_z - (probs * logits.masked_fill(probs == 0, 0.0)).sum(-1)


def _mask_by_index(logits: torch.Tensor, rows: torch.Tensor, ids: torch.Tensor, constrained_rows: torch.Tensor) -> torch.Tensor:
    """`mask_logits` with its (row, allowed id) pairs and constrained rows already on the device (update_host_index)."""
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[rows, ids] = True
    constrained = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    constrained[constrained_rows] = True
    return logits.masked_fill(constrained.unsqueeze(1) & ~keep, float("-inf"))


_CAPTURES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # update_capture: model -> its _UpdateCapture
_MAX_GRAPHS = 128  # update_capture: shapes past this many run eager


def _hidden(model, input_ids, attention_mask, position_ids, start: int, prefix_len: int, groups):
    """The final-norm hidden states that predict columns [start + 1:], [rows, width - 1 - start, hidden size]
    (`_completion_logprobs`): with `groups` (first, inverse) the shared prompts' columns [:start] run once a group,
    else with 0 < prefix_len <= start the batch-wide prefix runs once, else one pass."""
    if groups is not None:
        first, inverse = groups
        cache = None
        if prefix_len > 0:
            cache = model.base_model(input_ids=input_ids[:1, :prefix_len], position_ids=position_ids[:1, :prefix_len], use_cache=True).past_key_values
            cache.batch_repeat_interleave(first.numel())
        cache = model.base_model(
            input_ids=input_ids[first, prefix_len:start],
            attention_mask=attention_mask[first, :start],
            position_ids=position_ids[first, prefix_len:start],
            past_key_values=cache,
            use_cache=True,
        ).past_key_values
        cache.batch_select_indices(inverse)
        hidden = model.base_model(
            input_ids=input_ids[:, start:],
            attention_mask=attention_mask,
            position_ids=position_ids[:, start:],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        return hidden[:, :-1]
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
        return hidden[:, start - prefix_len : -1]
    hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).last_hidden_state
    return hidden[:, start:-1]


def _completion_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    temperature: float = 1.0,
    prefix_len: int = 0,
    allowed=None,
    entropy_grad: bool = False,
    share_prompt: bool = False,
    parts: dict | None = None,
    index: _MicroIndex | None = None,
    body=None,
):
    """Per-token log-probs (at the sampling temperature) and entropies of completion tokens.

    `allowed` (per row, per completion position: allowed token ids or None) renormalises over the
    tokens the constrained sampler could pick, so log-probs are those of the constrained policy.
    The entropies carry a gradient only with `entropy_grad` (for the entropy bonus).

    Positions follow the attention mask (as in `generate`), so padding may sit anywhere.
    If `prefix_len > 0`, columns `[:prefix_len]` are identical and fully attended in
    every row: that prefix is run once and its KV cache broadcast to the batch, which
    is exact under causal attention and skips ~`(rows-1) * prefix_len` tokens of
    forward/backward compute.

    `parts` (time_update_parts): forward times go to its t_part_fwd_body / _mask / _head (`_lap`).

    `share_prompt`: rows whose columns `[:prompt_width - 1]` (ids and mask) match, a GRPO group's, run
    those columns once; the cache is then indexed out to the rows (gradients sum back through the
    index), exact for the same reason as the prefix.

    `index` (update_host_index): the micro-batch's `_MicroIndex`, whose host-built indices stand in for the
    completion mask's boolean indexing, the shared-prompt grouping and the constrained mask (the same values in
    the same order, without device syncs); `allowed` is then None.

    `body` (update_capture): the update's `_UpdateCapture`, which gives the hidden states in place of the eager
    forward, or None, and the eager forward runs.
    """
    start = max(0, prompt_width - 1)
    share = share_prompt and prefix_len < start
    hidden = None if body is None else body(model, input_ids, attention_mask, start, prefix_len, share, index)
    if hidden is None:
        groups = None
        if share:
            if index is not None and index.groups is not None:
                groups = index.groups
            else:
                key = input_ids[:, :start] * 2 + attention_mask[:, :start].long()
                _, inverse = torch.unique(key, dim=0, return_inverse=True)
                rows = torch.arange(input_ids.shape[0], device=input_ids.device)
                first = torch.full((int(inverse.max()) + 1,), rows.numel(), device=rows.device).scatter_reduce_(0, inverse, rows, "amin")
                groups = (first, inverse)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = _hidden(model, input_ids, attention_mask, position_ids, start, prefix_len, groups)
    _lap(parts, "fwd_body", input_ids.device)
    labels = input_ids[:, start + 1 :]
    completion_mask = attention_mask[:, start + 1 :].bool()
    # the vocabulary projection only for real completion tokens (not padding): with long,
    # right-padded rollouts most of the rectangle is padding and the fp32 logits dominate memory
    sel = completion_mask if index is None else index.tokens  # the same tokens in the same (row-major) order
    logits = model.get_output_embeddings()(hidden[sel]).float() / temperature
    if allowed is not None:
        pairs, k = [], 0
        for row, n in enumerate(completion_mask.sum(1).tolist()):
            pairs.extend((k + j, ids) for j, ids in enumerate(allowed[row][:n]) if ids is not None)
            k += n
        logits = mask_logits(logits, pairs)
        _lap(parts, "fwd_mask", input_ids.device)
    elif index is not None and index.mask is not None:
        logits = _mask_by_index(logits, *index.mask)
        _lap(parts, "fwd_mask", input_ids.device)
    log_z = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, labels[sel].unsqueeze(-1)).squeeze(-1) - log_z
    if entropy_grad and torch.is_grad_enabled():  # for the entropy bonus; recomputed in the backward to keep memory flat
        ent = checkpoint(_entropy, logits, log_z, use_reentrant=False)
    else:
        with torch.no_grad():
            ent = _entropy(logits, log_z)
    token_lp = torch.zeros(labels.shape, dtype=lp.dtype, device=lp.device)
    token_lp[sel] = lp
    entropy = torch.zeros(labels.shape, dtype=ent.dtype, device=ent.device)
    entropy[sel] = ent
    _lap(parts, "fwd_head", input_ids.device)
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


def group_advantages(groups: torch.Tensor, adv_norm: str, pos_scale: float = 1.0) -> torch.Tensor:
    """Group-relative advantages: `std` is GRPO's per-group z-score, `none` is the
    Dr. GRPO variant (mean-centred only), which removes the difficulty bias that
    up-weights groups where only one or two samples disagree with the rest.
    `pos_scale` then multiplies the positive ones (`GRPOConfig.pos_adv_scale`)."""
    centred = groups - groups.mean(dim=1, keepdim=True)
    if adv_norm == "none":
        adv = centred
    elif adv_norm == "std":
        stds = groups.std(dim=1, unbiased=False, keepdim=True)
        adv = torch.where(stds > 0, centred / (stds + 1e-4), torch.zeros_like(centred))
    else:
        raise ValueError(f"unknown adv_norm {adv_norm!r} (expected 'std' or 'none')")
    return adv if pos_scale == 1.0 else torch.where(adv > 0, pos_scale * adv, adv)


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


def _lap(parts: dict | None, name: str, device: torch.device) -> None:
    """time_update_parts: adds the synced wall time since the previous lap (`parts["_t"]`) to parts["t_part_" + name]."""
    if parts is None:
        return
    t = _now(device)
    key = "t_part_" + name
    parts[key] = parts.get(key, 0.0) + t - parts["_t"]
    parts["_t"] = t


def _is_vllm(engine) -> bool:
    """A rollout engine with `VLLMRollout.generate`'s interface: vLLM's, or the same-process sampler (`native`)."""
    return type(engine).__name__ in ("VLLMRollout", "NativeRollout")


def _wants_sampler_logprobs(config: GRPOConfig) -> bool:
    return config.log_mismatch or config.tis_cap > 0


def _sample_batch(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device, caches: StaticCachePool | None = None,
                  parts: dict | None = None):
    """`parts` (time_update_parts): sampling's part times go to t_part_s_* (`_lap`), and the constrained vLLM
    processor's CPU time to t_part_s_mask_cpu (inside t_part_s_generate, not added to the sum)."""
    encoded, prefix_len = _encode_prompts(tokenizer, puzzles, config, device)
    _lap(parts, "s_encode", device)
    sampler_lp = None  # per completion token, the sampler's log-prob (with log_mismatch / tis_cap)
    if _is_vllm(caches):
        native = type(caches).__name__ == "NativeRollout"
        if parts is not None and not native:
            from . import vllm_constrained

            vllm_constrained.APPLY_TIME = 0.0  # a float turns the processor's timer on
        sampled = caches.generate(encoded, config.group_size, config.temperature, config.max_new_tokens, tokenizer.pad_token_id,
                                  puzzles=puzzles if config.constrain_answer else None, think_budget=config.think_budget,
                                  return_logprobs=_wants_sampler_logprobs(config))
        generated, finished_at = sampled[:2]
        if _wants_sampler_logprobs(config):
            sampler_lp = sampled[2]
        if parts is not None and native:
            parts["t_part_s_mask_cpu"] = caches.mask_cpu
        elif parts is not None:
            parts["t_part_s_mask_cpu"], vllm_constrained.APPLY_TIME = vllm_constrained.APPLY_TIME, None
    else:
        assert not _wants_sampler_logprobs(config), "log_mismatch and tis_cap need rollout_backend='vllm'"
        assert not config.constrain_answer, "constrain_answer needs rollout_backend='vllm' or 'native'"
        generated, finished_at = _generate_chunked(model, tokenizer, encoded, config, caches)
    _lap(parts, "s_generate", device)
    prompt_width = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    _lap(parts, "s_decode", device)
    expanded = [puzzle for puzzle in puzzles for _ in range(config.group_size)]
    results = [verify(text, puzzle) for text, puzzle in zip(completions, expanded)]
    if config.malformed_reward is None:
        values = [r.reward for r in results]
    else:
        values = [config.malformed_reward if r.malformed else r.reward for r in results]
    if config.malformed_reward_solved is not None:
        g = config.group_size
        solved = [any(r.correct for r in results[i - i % g : i - i % g + g]) for i in range(len(results))]
        values = [config.malformed_reward_solved if r.malformed and s else v for v, r, s in zip(values, results, solved)]
    if config.shaping:
        values = [v + config.shaping * partial_credit(r, p) for v, r, p in zip(values, results, expanded)]
    if config.process_reward:
        assert config.process_reward + 0.99 * config.shaping < 1.0, "a wrong answer must stay below a correct one"
        values = [v + config.process_reward * process_credit(t, p) for v, t, p in zip(values, completions, expanded)]
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
    if config.overlong_coef:
        start = config.max_new_tokens - config.overlong_buffer
        over = (valid.sum(dim=1).float() - start).clamp(min=0) / config.overlong_buffer
        penalty = config.overlong_coef * over.clamp(max=1.0)
        if config.overlong_correct_only:
            penalty = penalty * torch.tensor([r.correct for r in results], dtype=penalty.dtype, device=penalty.device)
        rewards = rewards - penalty
    attention = torch.cat([prompt_attention, completion_attention], dim=1)
    malformed = torch.tensor([r.malformed for r in results], dtype=torch.bool, device=device)
    correct = torch.tensor([r.correct for r in results], dtype=torch.bool, device=device)  # the verifier's verdict per sample
    _lap(parts, "s_reward", device)
    return generated, attention, completions, expanded, rewards, prompt_width, prefix_len, malformed, sampler_lp, correct


def hindsight_puzzle(text: str, puzzle, exclude: frozenset = frozenset()):
    """Hindsight relabelling (`GRPOConfig.hindsight_rows`; unranked screens only): the puzzle that `text` does
    solve when it misses `puzzle` by its value alone, else None. That is `puzzle` with its target set to the value
    of `text`'s expression, when the value is an integer inside the level's generator target range, the verifier
    accepts `text` for the relabelled puzzle (so it uses each of the puzzle's numbers once and only its
    operators), and the relabelled puzzle's key is not in `exclude` (the ladder sets). Nothing here solves or
    searches: the level's "needs" rule is not checked on the relabelled puzzle."""
    result = verify(text, puzzle)
    if result.correct or result.value is None or result.value.denominator != 1 or not puzzle.level:
        return None
    low, high = level_by_name(puzzle.level).target_range
    if not low <= result.value <= high:
        return None
    relabelled = replace(puzzle, target=int(result.value), solution=None)
    if puzzle_key(relabelled) in exclude or not verify(text, relabelled).correct:
        return None
    return relabelled


class _Replay:
    """The self-imitation buffer (`GRPOConfig.sil_rows`): (puzzle, completion tokens) of the run's own
    verifier-correct samples, the newest `capacity` per pool. It lives in memory only: `run` starts it empty and
    `_save_state` never sees it. With `GRPOConfig.hindsight_rows` (unranked screens only) it holds relabelled
    pairs instead (`hindsight_puzzle`), and `exclude` is the puzzle keys that may not be relabelled to."""

    def __init__(self, capacity: int, by_level: bool, seed: int = 0, exclude: frozenset = frozenset()):
        self.capacity, self.by_level, self.exclude = capacity, by_level, exclude
        self.pools: dict[str, list] = {}
        self.added: Counter = Counter()  # per pool, entries ever added (the next slot to overwrite once it is full)
        self.rng = random.Random(seed)  # its own stream, so the puzzle mix draws as it does without replay

    def __len__(self) -> int:
        return sum(len(pool) for pool in self.pools.values())

    def add(self, puzzle, tokens: np.ndarray) -> None:
        key = (puzzle.level or "") if self.by_level else ""
        pool = self.pools.setdefault(key, [])
        if len(pool) < self.capacity:
            pool.append((puzzle, tokens))
        else:
            pool[self.added[key] % self.capacity] = (puzzle, tokens)  # the oldest entry
        self.added[key] += 1

    def draw(self, n: int) -> list:
        """Up to `n` entries: each draw picks a pool uniformly among those holding anything, then the pool gives
        that many distinct entries (all of them if it holds fewer)."""
        keys = sorted(self.pools)
        if not keys:
            return []
        wanted = Counter(self.rng.choice(keys) for _ in range(n))
        return [self.pools[key][i] for key in keys for i in self.rng.sample(range(len(self.pools[key])), min(wanted[key], len(self.pools[key])))]


@dataclass
class _ReplayBatch:
    """Replayed rows laid out like a rollout batch (`[prompt][completion, right-padded]`), for `_update`."""

    ids: torch.Tensor
    attention: torch.Tensor
    prompt_width: int
    prefix_len: int
    allowed: list | None = None  # as `_allowed_tokens` gives it, with constrain_answer
    nll: float = 0.0  # set by `_update`: mean negative log-prob per replayed token, before the step


@dataclass
class _LMBatch:
    """This step's pretraining windows (`GRPOConfig.lm_rows`), for `_update`."""

    ids: torch.Tensor  # [lm_rows, lm_seq_len + 1]
    loss: float = 0.0  # set by `_update`: mean next-token cross-entropy, before the step


def _lm_batch(data: np.ndarray, config: GRPOConfig, step: int, vocab: int, device: torch.device) -> _LMBatch:
    """lm_rows windows of `data` (a token file) at offsets drawn from (seed, step)."""
    span = config.lm_seq_len + 1
    starts = np.random.default_rng((config.seed, step)).integers(0, len(data) - span + 1, config.lm_rows)
    windows = np.stack([np.asarray(data[i : i + span], dtype=np.int64) for i in starts])
    if windows.max() >= vocab:
        raise ValueError(f"lm_tokens holds token id {int(windows.max())}, outside the policy's vocabulary ({vocab}): another tokenizer?")
    return _LMBatch(torch.from_numpy(windows).to(device))


def _replay_rows(config: GRPOConfig) -> tuple[int, float]:
    """Rows replayed per step and a replayed row's weight: hindsight relabelling's when it is on, else
    self-imitation's."""
    return (config.hindsight_rows, config.hindsight_coef) if config.hindsight_rows > 0 else (config.sil_rows, config.sil_coef)


def _replay_batch(replay: _Replay, tokenizer, config: GRPOConfig, device: torch.device) -> _ReplayBatch | None:
    """This step's draws from the buffer (None while it is empty). Each row is the stored puzzle's prompt followed
    by the stored completion tokens, unchanged."""
    drawn = replay.draw(_replay_rows(config)[0])
    if not drawn:
        return None
    puzzles = [puzzle for puzzle, _ in drawn]
    encoded, prefix_len = _encode_prompts(tokenizer, puzzles, config, device)
    width = max(1, max(len(tokens) for _, tokens in drawn))
    completion = torch.full((len(drawn), width), tokenizer.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(drawn), width), dtype=encoded["attention_mask"].dtype)
    for row, (_, tokens) in enumerate(drawn):
        completion[row, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
        mask[row, : len(tokens)] = 1
    ids = torch.cat([encoded["input_ids"], completion.to(device)], dim=1)
    attention = torch.cat([encoded["attention_mask"], mask.to(device)], dim=1)
    prompt_width = encoded["input_ids"].shape[1]
    allowed = _allowed_tokens(tokenizer, ids, attention, prompt_width, puzzles, config.think_budget) if config.constrain_answer else None
    return _ReplayBatch(ids, attention, prompt_width, prefix_len, allowed)


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
    return [torch.tensor(g, device=completion_valid.device) for g in _micro_groups(completion_valid, prompt_width, config)]


def _micro_groups(completion_valid: torch.Tensor, prompt_width: int, config: GRPOConfig) -> list[list[int]]:
    """`_micro_batches`' row groups as lists (the lengths are sorted on `completion_valid`'s device, so ties fall the same)."""
    n = completion_valid.shape[0]
    if config.micro_batch_tokens <= 0:
        micro = max(1, config.micro_batch_size)
        return [list(range(i, min(i + micro, n))) for i in range(0, n, micro)]
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
    return groups


@dataclass
class _MicroIndex:
    """One micro-batch's indices on the device, built on the host at the start of the update (update_host_index)."""

    rows: torch.Tensor  # the batch rows it takes
    width: int  # prompt_width + its longest completion
    tokens: tuple  # (row, column) of each completion token in its [rows, width - prompt_width] rectangle, row-major as a boolean mask
    groups: tuple | None = None  # share_prompt: (the first row of each distinct prompt, each row's prompt), numbered as by `torch.unique`
    mask: tuple | None = None  # constrained logits: (logit row, allowed id) pairs and the constrained logit rows, as `mask_logits` builds them


def _host_micro(valid: torch.Tensor, rows: torch.Tensor, prompt_width: int, width: int, allowed, key) -> tuple[list[torch.Tensor], bool, bool]:
    """One `_MicroIndex`'s host tensors: rows, token rows, token columns, then (first, inverse) with `key` (shared
    prompts) and (logit rows, ids, constrained rows) when `allowed` constrains a token; and whether each is there."""
    rect = valid[rows, : width - prompt_width]
    tensors = [rows, *rect.nonzero(as_tuple=True)]
    if key is not None:
        _, inverse = torch.unique(key[rows], dim=0, return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), rows.numel(), dtype=torch.long)
        tensors += [first.scatter_reduce_(0, inverse, torch.arange(rows.numel()), "amin"), inverse]
    logit_rows, ids, constrained, k = [], [], [], 0
    if allowed is not None:
        for row, n in zip(rows.tolist(), rect.sum(1).tolist()):
            for j, allowed_ids in enumerate(allowed[row][:n]):
                if allowed_ids is not None:
                    constrained.append(k + j)
                    logit_rows.extend([k + j] * len(allowed_ids))
                    ids.extend(allowed_ids)
            k += n
    if constrained:
        tensors += [torch.tensor(logit_rows, dtype=torch.long), torch.tensor(ids, dtype=torch.long), torch.tensor(constrained, dtype=torch.long)]
    return tensors, key is not None, bool(constrained)


def _plan_update(completion_valid, active, generated, attention, prompt_width: int, prefix_len: int, config: GRPOConfig, allowed,
                 replayed: _ReplayBatch | None) -> tuple[list[_MicroIndex], list[_MicroIndex]]:
    """update_host_index: each micro-batch's `_MicroIndex`, the batch's and then the replayed rows', grouped as by
    `_micro_batches`, from one copy of the masks to the host, and sent to the device in one transfer."""
    valid, rows_of = completion_valid.cpu(), active.cpu()
    start = max(0, prompt_width - 1)
    key = (generated[:, :start] * 2 + attention[:, :start].long()).cpu() if config.share_group_prompt and prefix_len < start else None
    specs = []
    for group in _micro_groups(completion_valid[active], prompt_width, config):
        rows = rows_of[group]
        used = valid[rows].any(0).nonzero()
        width = prompt_width + (int(used.max()) + 1 if used.numel() else 1)
        specs.append((width, *_host_micro(valid, rows, prompt_width, width, allowed, key)))
    main = len(specs)
    if replayed is not None:
        replay_valid = replayed.attention[:, replayed.prompt_width :].bool()
        valid = replay_valid.cpu()
        for group in _micro_groups(replay_valid, replayed.prompt_width, config):
            rows = torch.tensor(group, dtype=torch.long)
            width = replayed.prompt_width + max(1, int(valid[rows].sum(1).max()))  # completions are right-padded
            specs.append((width, *_host_micro(valid, rows, replayed.prompt_width, width, replayed.allowed, None)))
    flat = [t for _, tensors, _, _ in specs for t in tensors]
    joined = torch.cat(flat) if flat else torch.zeros(0, dtype=torch.long)
    if generated.device.type == "cuda":
        joined = joined.pin_memory().to(generated.device, non_blocking=True)
    else:
        joined = joined.to(generated.device)
    pieces = iter(joined.split([t.numel() for t in flat]))
    index = []
    for width, _, grouped, masked in specs:
        rows, tokens = next(pieces), (next(pieces), next(pieces))
        groups = (next(pieces), next(pieces)) if grouped else None
        mask = (next(pieces), next(pieces), next(pieces)) if masked else None
        index.append(_MicroIndex(rows, width, tokens, groups, mask))
    return index[:main], index[main:]


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


# sums kept by `_sampler_mismatch_` (one more slot after them holds the largest |log ratio|)
_MISMATCH_SUMS = ("tokens", "k1", "k3", "far", "tis_clipped", "rows", "seq_abs")


def _sampler_mismatch_(sums: torch.Tensor, token_lp: torch.Tensor, sampler_lp: torch.Tensor, completion_mask: torch.Tensor,
                       tis_cap: float) -> torch.Tensor | None:
    """Adds one micro-batch's sampler/trainer mismatch to `sums` and returns its truncated importance
    weights min(pi_train / pi_sampler, tis_cap) per completion token (as `token_lp[completion_mask]`),
    or None when tis_cap is 0. `token_lp` (detached) and `sampler_lp` are [rows, completion width]."""
    with torch.no_grad():
        log_ratio = torch.where(completion_mask, token_lp - sampler_lp, torch.zeros_like(token_lp))  # log(pi_train / pi_sampler)
        flat = log_ratio[completion_mask]
        ratio = flat.exp()
        far = (flat.abs() > math.log(2.0)).float()
        clipped = (ratio > tis_cap).float() if tis_cap > 0 else torch.zeros_like(far)
        sums[:-1] += torch.stack(
            [flat.new_tensor(float(flat.numel())), -flat.sum(), (ratio - 1 - flat).sum(), far.sum(), clipped.sum(),
             flat.new_tensor(float(completion_mask.shape[0])), log_ratio.sum(1).abs().sum()]
        )
        if flat.numel():
            sums[-1] = torch.maximum(sums[-1], flat.abs().max())
        return ratio.clamp(max=tis_cap) if tis_cap > 0 else None


def _mismatch_stats(sums: torch.Tensor, dist: Dist, tis_cap: float) -> dict:
    """train_log fields from `_sampler_mismatch_` sums, with r = pi_train / pi_sampler per updated token:
    mismatch_k1 = mean -log r and mismatch_k3 = mean r - 1 - log r (two estimates of KL(sampler || train)),
    mismatch_far = share of tokens with r outside [1/2, 2], mismatch_seq = mean |sum of log r| per row,
    mismatch_max = largest |log r| (this rank's), tis_clipped = share of tokens with r > tis_cap."""
    largest = sums[-1].item()
    tokens, k1, k3, far, clipped, rows, seq_abs = dist.all_reduce_sum_(sums[:-1].clone()).tolist()
    tokens, rows = max(tokens, 1.0), max(rows, 1.0)
    stats = {"mismatch_k1": k1 / tokens, "mismatch_k3": k3 / tokens, "mismatch_far": far / tokens, "mismatch_seq": seq_abs / rows,
             "mismatch_max": largest}
    if tis_cap > 0:
        stats["tis_clipped"] = clipped / tokens
    return stats


def _capture_rows(rows: int, step: int = 0) -> int:
    """update_capture's row bucket: a multiple of `step`, or for step 0 steps of 8 up to 64 rows, 16 up to 128, 32 up
    to 256, then 64."""
    if step <= 0:
        step = 8 if rows <= 64 else 16 if rows <= 128 else 32 if rows <= 256 else 64
    return -(-rows // step) * step


class _CaptureEntry:
    """One captured shape: its eager body (`run`), static inputs, graphs (CUDA) and output / output-gradient views."""

    def __init__(self, run, ids, mask, first, inverse, shape: tuple):
        self.run, self.ids, self.mask, self.first, self.inverse, self.shape = run, ids, mask, first, inverse, shape
        self.fwd = self.bwd = self.out = self.grad = self.h = None


class _GraphedBody(torch.autograd.Function):
    """A captured body as one autograd node: the forward replays its forward graph; the backward replays its backward
    graph, which adds the body's parameter gradients into the bound .grad tensors itself (so no input gets one)."""

    @staticmethod
    def forward(ctx, anchor, capture, entry):
        ctx.capture, ctx.entry = capture, entry
        ctx.serial = capture.forward(entry)
        return entry.out.detach()

    @staticmethod
    def backward(ctx, grad):
        ctx.capture.backward(ctx.entry, ctx.serial, grad)
        return None, None, None


class _UpdateCapture:
    """update_capture (LOG r125): the update's model body (`_hidden`, everything before the vocabulary projection)
    runs as CUDA graphs, one forward and one backward graph per bucketed shape (`_capture_rows` rows, width a
    multiple of 64, shared-prompt groups a multiple of 8, by default; LOG r127 adds finer steps), captured the first
    time the shape comes and replayed after; the same maths up to rounding, and the captures count in the clock.

    Padded rows copy row 0 (padded groups copy group 0) and padded columns are id 0 / mask 0 at the end, so under
    causal attention the real rows' columns are unchanged; the caller gets the real [rows, width - 1 - start] slice,
    so the padding's gradient is zero. Parameter gradients go into .grad tensors bound once (`bind_grads`, in place
    of zero_grad(set_to_none)), which the eager parts of the update add into in place, so the backward graph can add
    into fixed addresses. A body's backward must follow its own forward, once (each micro-batch's backward comes
    before the next forward); otherwise it raises. On CPU the same path runs eagerly, which the tests use."""

    def __init__(self, model, device: torch.device, micro_batch_tokens: int, width_step: int = 64, row_step: int = 0,
                 group_step: int = 8):
        self.device, self.cuda = device, device.type == "cuda"
        self.width_step, self.row_step, self.group_step = width_step, row_step, group_step
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.grads = [torch.zeros_like(p) for p in self.params]
        self.ptrs = [p.data_ptr() for p in self.params]
        self.anchor = torch.zeros((), device=device, requires_grad=True)
        self.cap_tokens = 2 * micro_batch_tokens + 4096  # _micro_groups keeps rows x width <= micro_batch_tokens
        self.entries: dict = {}
        self.stats = {"keys": 0, "captures": 0, "capture_s": 0.0, "replays": 0, "fallbacks": 0, "resets": 0}
        # summed over replays: the micro-batches' real rows x width and their buckets' (host ints, no sync)
        self.stats.update(real_tokens=0, bucket_tokens=0, real_groups=0, bucket_groups=0)
        self.serial = self.done = 0
        self.ids_buf = self.mask_buf = self.out_buf = self.grad_buf = None
        self.warmed = False
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.pool = torch.cuda.graph_pool_handle() if self.cuda else None

    def bind_grads(self) -> None:
        """Zero the bound gradients and set them as the parameters' .grad, at the start of each update."""
        ptrs = [p.data_ptr() for p in self.params]
        if ptrs != self.ptrs:  # the parameters moved, and the graphs read the old addresses
            self.entries.clear()
            self.ptrs = ptrs
            self.stats["resets"] += 1
        torch._foreach_zero_(self.grads)
        for p, g in zip(self.params, self.grads, strict=True):
            p.grad = g

    def check_bound(self) -> None:
        if any(p.grad is not g for p, g in zip(self.params, self.grads, strict=True)):
            raise RuntimeError("update_capture: a parameter's .grad was replaced during the update")

    def __call__(self, model, input_ids, attention_mask, start: int, prefix_len: int, share: bool, index):
        """The micro-batch's hidden states through the captured body, or None when it runs eager instead."""
        if not torch.is_grad_enabled() or index is None or (share and index.groups is None):
            self.stats["fallbacks"] += 1
            return None
        R, W = input_ids.shape
        first = inverse = None
        if share:
            kind, plen = "share", prefix_len
            first, inverse = index.groups
        elif 0 < prefix_len <= start:
            kind, plen = "prefix", prefix_len
        else:
            kind, plen = "plain", 0
        rows, width = _capture_rows(R, self.row_step), -(-W // self.width_step) * self.width_step
        groups = 0 if first is None else -(-first.numel() // self.group_step) * self.group_step
        dev = input_ids.device.type
        key = (kind, plen, start, rows, width, groups, torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev))
        e = self.entries.get(key)
        if e is None:
            if rows * width > self.cap_tokens or len(self.entries) >= _MAX_GRAPHS:
                self.stats["fallbacks"] += 1
                return None
            e = self.entries[key] = self._entry(model, key, input_ids, attention_mask, first, inverse)
            self.stats["keys"] += 1
        self.stats["real_tokens"] += R * W
        self.stats["bucket_tokens"] += rows * width
        if first is not None:
            self.stats["real_groups"] += first.numel()
            self.stats["bucket_groups"] += groups
        e.ids[:R, :W].copy_(input_ids)
        e.ids[R:, :W].copy_(input_ids[:1])
        e.ids[:, W:].zero_()
        e.mask[:R, :W].copy_(attention_mask)
        e.mask[R:, :W].copy_(attention_mask[:1])
        e.mask[:, W:].zero_()
        if first is not None:
            G = first.numel()
            e.first[:G].copy_(first)
            e.first[G:].copy_(first[:1])
            e.inverse[:R].copy_(inverse)
            e.inverse[R:].copy_(inverse[:1])
        if self.cuda and e.fwd is None:
            self._capture(e)
        out = _GraphedBody.apply(self.anchor, self, e)
        return out[:R, : W - 1 - start]

    def _entry(self, model, key: tuple, input_ids, attention_mask, first, inverse) -> _CaptureEntry:
        kind, plen, start, rows, width, groups, amp, amp_dtype = key
        dev = input_ids.device
        if self.ids_buf is None:  # every entry's static ids and mask are views of one buffer, refilled before each replay
            self.ids_buf = torch.zeros(self.cap_tokens, dtype=input_ids.dtype, device=dev)
            self.mask_buf = torch.zeros(self.cap_tokens, dtype=attention_mask.dtype, device=dev)
        ids = self.ids_buf[: rows * width].view(rows, width)
        mask = self.mask_buf[: rows * width].view(rows, width)
        if kind == "share":
            first = torch.zeros(groups, dtype=first.dtype, device=dev)
            inverse = torch.zeros(rows, dtype=inverse.dtype, device=dev)
        ref = weakref.ref(model)  # the entry lives in _CAPTURES[model], so it must not hold the model

        def run():
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp, cache_enabled=False):
                pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
                return _hidden(ref(), ids, mask, pos, start, plen, (first, inverse) if kind == "share" else None)

        return _CaptureEntry(run, ids, mask, first, inverse, (rows, width - 1 - start))

    def _buffers(self, hidden_dim: int, dtype) -> None:
        """One output and one output-gradient buffer for all entries (each micro-batch's forward and backward finish
        before the next forward, and the caller copies what it takes out of the output)."""
        if self.out_buf is None:
            self.out_buf = torch.zeros(self.cap_tokens * hidden_dim, dtype=dtype, device=self.device)
            self.grad_buf = torch.zeros_like(self.out_buf)

    def _views(self, e: _CaptureEntry) -> None:
        shape = (*e.shape, self.out_buf.numel() // self.cap_tokens)
        n = math.prod(shape)
        e.out, e.grad = self.out_buf[:n].view(shape), self.grad_buf[:n].view(shape)

    def _capture(self, e: _CaptureEntry) -> None:
        current = torch.cuda.current_stream(self.device)
        t0 = _now(self.device)
        if not self.warmed:  # once, eagerly on the capture stream, so its libraries are set up before any capture
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                h = e.run()
                torch.autograd.grad(h, self.params, torch.zeros_like(h), allow_unused=True)
                hidden_dim, dtype = h.shape[-1], h.dtype
                del h
            current.wait_stream(self.stream)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            self._buffers(hidden_dim, dtype)
            self.warmed = True
        self._views(e)
        e.fwd, e.bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            e.fwd.capture_begin(pool=self.pool)
            try:
                h = e.run()
                assert h.dtype == e.out.dtype and h.shape == e.out.shape, (h.dtype, h.shape, e.out.shape)
                e.out.copy_(h)
            finally:
                e.fwd.capture_end()
            e.bwd.capture_begin(pool=self.pool)
            try:
                grads = torch.autograd.grad(h, self.params, e.grad, allow_unused=True)
                pairs = [(b, g) for b, g in zip(self.grads, grads, strict=True) if g is not None]
                torch._foreach_add_([b for b, _ in pairs], [g for _, g in pairs])
            finally:
                e.bwd.capture_end()
            del h, grads, pairs
        current.wait_stream(self.stream)
        self.stats["captures"] += 1
        self.stats["capture_s"] += _now(self.device) - t0

    def forward(self, e: _CaptureEntry) -> int:
        if self.cuda:
            e.fwd.replay()
        else:
            with torch.enable_grad():
                e.h = e.run()
            self._buffers(e.h.shape[-1], e.h.dtype)
            self._views(e)
            e.out.copy_(e.h.detach())
        self.serial += 1
        self.stats["replays"] += 1
        return self.serial

    def backward(self, e: _CaptureEntry, serial: int, grad: torch.Tensor) -> None:
        if serial != self.serial or self.done == serial:
            raise RuntimeError("update_capture: a captured body's backward must follow its own forward, once")
        self.done = serial
        if self.cuda:
            e.grad.copy_(grad)
            e.bwd.replay()
            return
        grads = torch.autograd.grad(e.h, self.params, grad, allow_unused=True)
        e.h = None
        pairs = [(b, g) for b, g in zip(self.grads, grads, strict=True) if g is not None]
        torch._foreach_add_([b for b, _ in pairs], [g for _, g in pairs])


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
    sampler_lp: torch.Tensor | None = None,
    replayed: _ReplayBatch | None = None,
    lm: _LMBatch | None = None,
    pg_skip: torch.Tensor | None = None,
    parts: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """One optimizer step on this rank's rows; returns (loss, kl, entropy_sum, token_count) sums,
    with `sampler_lp` (the sampler's per-token log-probs, [rows, completion width]) the
    `_sampler_mismatch_` sums (else None), and the gradient norm before clipping (None without
    `grad_clip`).

    `replayed` (self-imitation) adds -sil_coef * (the replayed rows' summed log-probs) / token_count to the
    step's loss, so a replayed row weighs like one of the batch's rows with advantage sil_coef (hindsight_coef
    when the rows are hindsight relabelling's, `_replay_rows`). The returned sums leave it out; its mean per
    token is written to `replayed.nll`.

    `pg_skip` ([rows, completion width] bool, `answer_open_mask`) zeroes the advantage of those tokens in the
    policy-gradient term only.

    `parts` (time_update_parts) collects the update's part times (`_lap`) and its micro-batch counts.

    `lm` (pretraining during RL) adds lm_coef * the windows' mean next-token cross-entropy, left out of the
    returned sums too; the mean is written to `lm.loss`.

    Under data parallelism the loss is normalised by the global completion-token
    count and gradients are summed across ranks, so the step equals the
    single-process step on the concatenated batch.
    """
    completion_valid = attention[:, prompt_width:].bool()
    token_count = dist.all_reduce_sum_(completion_valid.sum()).clamp_min(1)
    active = torch.arange(generated.shape[0], device=generated.device)
    if config.skip_zero_advantage:
        active = (advantages != 0).nonzero().squeeze(1)
    capture = None
    if config.update_capture:
        capture = _CAPTURES.get(model)
        if capture is None:
            capture = _CAPTURES[model] = _UpdateCapture(model, device, config.micro_batch_tokens, config.update_capture_width_step,
                                                        config.update_capture_row_step, config.update_capture_group_step)
        capture.bind_grads()
    else:
        optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)
    entropy_total = torch.zeros((), device=device)
    mismatch = None if sampler_lp is None else torch.zeros(len(_MISMATCH_SUMS) + 1, device=device)
    bonus = config.entropy_coef > 0
    if bonus:  # the bonus is a mean over the updated rows' tokens
        bonus_tokens = dist.all_reduce_sum_(completion_valid[active].sum()).clamp_min(1)
    plan = None
    if config.update_host_index:
        plan = _plan_update(completion_valid, active, generated, attention, prompt_width, prefix_len, config, allowed, replayed)
    _lap(parts, "setup", device)
    for sub in (_micro_batches(completion_valid[active], prompt_width, config) if plan is None else plan[0]):
        index = None if plan is None else sub
        if index is None:
            rows = active[sub]
            width = prompt_width + int(completion_valid[rows].any(0).nonzero().max().item()) + 1 if completion_valid[rows].any() else prompt_width + 1
        else:
            rows, width = index.rows, index.width
        ids, mask, adv = generated[rows, :width], attention[rows, :width], advantages[rows]
        if config.grad_checkpoint:  # HF checkpoints only in train mode, and then drops the KV cache the prefix trick needs
            model.train()
        rows_allowed = None if allowed is None or index is not None else [allowed[r] for r in rows.tolist()]
        if parts is not None:
            parts["n_micro"] = parts.get("n_micro", 0) + 1
            _lap(parts, "micro_setup", device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.master_fp32):
            token_lp, completion_mask, entropy = _completion_logprobs(
                model, ids, mask, prompt_width, config.temperature, 0 if config.grad_checkpoint else prefix_len, rows_allowed,
                entropy_grad=bonus, share_prompt=config.share_group_prompt and not config.grad_checkpoint, parts=parts, index=index,
                body=capture,
            )
        model.eval()
        sel = completion_mask if index is None else index.tokens
        policy_lp = token_lp[sel]
        entropy_total += entropy[sel].detach().sum()
        repeated_advantages = adv.unsqueeze(1).expand(-1, completion_mask.shape[1])[sel]
        if pg_skip is not None:
            repeated_advantages = repeated_advantages * (~pg_skip[rows, : width - prompt_width])[sel]
        if config.cov_clip_frac > 0 and policy_lp.numel() > 1:
            policy_lp = _cov_clip(policy_lp, repeated_advantages, config.cov_clip_frac)
        if sampler_lp is not None:
            weights = _sampler_mismatch_(mismatch, token_lp.detach(), sampler_lp[rows, : width - prompt_width], completion_mask, config.tis_cap)
            if weights is not None:
                repeated_advantages = repeated_advantages * weights
        policy_loss = -(repeated_advantages * policy_lp).sum() / token_count
        if bonus:
            policy_loss = policy_loss - config.entropy_coef * entropy[sel].sum() / bonus_tokens
        if ref_model is None:
            kl = torch.zeros((), device=device)
            loss = policy_loss
        else:
            with torch.no_grad():
                ref_lp, _, _ = _completion_logprobs(
                    ref_model, ids, mask, prompt_width, config.temperature, prefix_len, rows_allowed, share_prompt=config.share_group_prompt,
                    index=index,
                )
            ref_lp = ref_lp[sel]
            kl_tokens = torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
            kl = kl_tokens.sum() / token_count
            loss = policy_loss + config.kl_coef * kl
        _lap(parts, "loss", device)
        loss.backward()
        _lap(parts, "bwd", device)
        loss_total += loss.detach()
        kl_total += kl.detach()
    if config.skip_zero_advantage:
        if active.numel() == 0 and capture is None:  # an all-zero gradient still moves AdamW (momentum, weight decay), as without skipping; update_capture's bound gradients are already zero
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter) if parameter.requires_grad else None
        # the logged entropy (entropy_total / token_count) becomes the mean over the updated rows
        entropy_total = entropy_total * token_count / completion_valid[active].sum().clamp_min(1)
    if replayed is not None:
        replay_valid = replayed.attention[:, replayed.prompt_width :].bool()
        nll_total = torch.zeros((), device=device)
        for rows in (_micro_batches(replay_valid, replayed.prompt_width, config) if plan is None else plan[1]):
            index = None if plan is None else rows
            if index is None:
                width = replayed.prompt_width + max(1, int(replay_valid[rows].sum(1).max().item()))  # completions are right-padded
            else:
                rows, width = index.rows, index.width
            if config.grad_checkpoint:
                model.train()
            rows_allowed = None if replayed.allowed is None or index is not None else [replayed.allowed[r] for r in rows.tolist()]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.master_fp32):
                token_lp, completion_mask, _ = _completion_logprobs(
                    model, replayed.ids[rows, :width], replayed.attention[rows, :width], replayed.prompt_width, config.temperature,
                    0 if config.grad_checkpoint else replayed.prefix_len, rows_allowed, index=index, body=capture,
                )
            model.eval()
            nll = -token_lp[completion_mask if index is None else index.tokens].sum()
            (_replay_rows(config)[1] * nll / token_count).backward()
            nll_total += nll.detach()
            if parts is not None:
                parts["n_micro_sil"] = parts.get("n_micro_sil", 0) + 1
        replay_nll = nll_total / replay_valid.sum().clamp_min(1)
        if plan is None:
            replayed.nll = replay_nll.item()
    _lap(parts, "sil", device)
    if lm is not None:
        per_micro = max(1, config.micro_batch_tokens // lm.ids.shape[1] if config.micro_batch_tokens > 0 else config.micro_batch_size)
        lm_tokens = lm.ids[:, 1:].numel()
        ce_total = torch.zeros((), device=device)
        for first in range(0, lm.ids.shape[0], per_micro):
            window = lm.ids[first : first + per_micro]
            if config.grad_checkpoint:
                model.train()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.master_fp32):
                hidden = model.base_model(input_ids=window[:, :-1], use_cache=False).last_hidden_state
                logits = model.get_output_embeddings()(hidden).float()
            model.eval()
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), window[:, 1:].reshape(-1), reduction="sum")
            (config.lm_coef * ce / lm_tokens).backward()
            ce_total += ce.detach()
        lm.loss = (ce_total / lm_tokens).item()
    _lap(parts, "lm", device)
    if capture is not None:
        capture.check_bound()
    dist.all_reduce_grads_(model.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip) if config.grad_clip > 0 else None
    _lap(parts, "clip", device)
    optimizer.step()
    if plan is not None and replayed is not None:  # read once the step is queued, so the update runs without a sync
        replayed.nll = replay_nll.item()
    _lap(parts, "opt", device)
    return loss_total, kl_total, entropy_total, token_count, mismatch, grad_norm


_GRAMMARS: dict = {}


def _grammar(tokenizer) -> AnswerGrammar:
    key = tokenizer.name_or_path
    if key not in _GRAMMARS:
        _GRAMMARS[key] = AnswerGrammar(token_strings(tokenizer), open_token_ids(tokenizer))
    return _GRAMMARS[key]


def _allowed_tokens(tokenizer, generated, attention, prompt_width, puzzles, think_budget: int = 0, sampled: list | None = None) -> list:
    """Per row, per completion token: the ids the constrained sampler allowed there (None = free text).

    `sampled` (reuse_sampler_allowed): per row, the sets the sampler recorded as it went. A row takes them as they are
    when its valid tokens are exactly the first len(sampled[r]) positions; any other row is recomputed."""
    grammar = _grammar(tokenizer)
    valid = attention[:, prompt_width:].bool()
    completion = generated[:, prompt_width:]
    reuse = [False] * len(puzzles)
    if sampled is not None:
        lengths = torch.tensor([len(a) for a in sampled], device=valid.device).unsqueeze(1)
        reuse = valid.eq(torch.arange(valid.shape[1], device=valid.device).unsqueeze(0) < lengths).all(1).tolist()
    return [sampled[r] if reuse[r] else grammar.allowed_per_position(completion[r][valid[r]].tolist(), p.numbers, p.ops, think_budget)
            for r, p in enumerate(puzzles)]


def answer_open_mask(tokenizer, generated, attention, prompt_width) -> torch.Tensor:
    """[rows, completion width] bool: the completion tokens whose text overlaps the first OPEN ("<answer>") of their
    row (`mask_answer_open`). Rows without OPEN are all False."""
    grammar = _grammar(tokenizer)
    valid = attention[:, prompt_width:].bool()
    out = torch.zeros(valid.shape, dtype=torch.bool)
    completion = generated[:, prompt_width:].tolist()
    for r, cols in enumerate(valid.cpu().tolist()):
        text, starts = "", []
        for c, ok in enumerate(cols):
            if not ok:
                continue
            starts.append((c, len(text)))
            text += grammar.text(completion[r][c])
            at = text.find(OPEN)
            if at >= 0:
                for col, start in reversed(starts):  # the tokens up to here that end after OPEN begins
                    if start + len(grammar.text(completion[r][col])) <= at:
                        break
                    out[r, col] = True
                break
    return out.to(valid.device)


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
    replay: _Replay | None = None,
    lm: _LMBatch | None = None,
) -> dict:
    """One GRPO step on `puzzles` (this rank's shard of the global prompt batch). With `lm` the step also
    trains on those pretraining windows (`GRPOConfig.lm_rows`). With `replay` (self-imitation)
    the step also replays rows drawn from it, then adds this step's verifier-correct samples to it. With
    `config.hindsight_rows` (unranked screens only) it adds the samples `hindsight_puzzle` relabels instead."""
    t0 = _now(device)
    parts = {"_t": t0} if config.time_update_parts else None
    generated, attention, completions, expanded, rewards, prompt_width, prefix_len, malformed, sampler_lp, correct = _sample_batch(
        model, tokenizer, puzzles, config, device, caches, parts
    )
    t_sample = _now(device) - t0
    sampler_allowed = caches.last_allowed if config.reuse_sampler_allowed else None  # per generated row
    hindsight, n_sampled = config.hindsight_rows > 0, len(completions)
    store = []  # (row, the puzzle it is stored with), over all sampled rows (kept group or not)
    if replay is not None and hindsight:  # every sample that misses by its value alone, with its relabelled puzzle
        relabelled = [None if solved else hindsight_puzzle(text, puzzle, replay.exclude)
                      for text, puzzle, solved in zip(completions, expanded, correct.tolist())]
        store = [(row, puzzle) for row, puzzle in enumerate(relabelled) if puzzle is not None]
    elif replay is not None:  # self-imitation: every verifier-correct sample, with the puzzle it answered
        store = [(row, expanded[row]) for row in correct.nonzero().squeeze(1).tolist()]
    solved_samples = []  # what goes into the buffer after the update: the puzzle and the sample's tokens as the update sees them
    if store:
        sampled, sampled_valid = generated[:, prompt_width:].cpu().numpy(), attention[:, prompt_width:].bool().cpu().numpy()
        solved_samples = [(puzzle, sampled[row][sampled_valid[row]].astype(np.int32)) for row, puzzle in store]
    all_rewards = rewards
    all_groups = all_rewards.reshape(len(puzzles), config.group_size)
    informative_groups = int((all_groups.std(dim=1, unbiased=False) > 0).sum().item())
    kept = select_groups(all_groups, config.prompts_per_step // dist.world_size)
    if kept.shape[0] < len(puzzles):
        rows = (kept.unsqueeze(1) * config.group_size + torch.arange(config.group_size, device=kept.device)).reshape(-1)
        generated, attention, rewards = generated[rows], attention[rows], rewards[rows]
        if sampler_lp is not None:
            sampler_lp = sampler_lp[rows]
        row_list = rows.tolist()
        completions = [completions[r] for r in row_list]
        if sampler_allowed is not None:
            sampler_allowed = [sampler_allowed[r] for r in row_list]
        expanded = [expanded[r] for r in row_list]
    groups = rewards.reshape(kept.shape[0], config.group_size)
    advantages = group_advantages(groups, config.adv_norm, config.pos_adv_scale).reshape(-1)

    _lap(parts, "select", device)  # sampling's end to here: stored samples, group selection, advantages
    allowed = (_allowed_tokens(tokenizer, generated, attention, prompt_width, expanded, config.think_budget, sampler_allowed)
               if config.constrain_answer else None)
    _lap(parts, "allowed", device)
    replayed = _replay_batch(replay, tokenizer, config, device) if replay is not None else None
    pg_skip = answer_open_mask(tokenizer, generated, attention, prompt_width) if config.mask_answer_open else None
    _lap(parts, "replay_batch", device)
    loss, kl, entropy_total, token_count, mismatch, grad_norm = _update(
        model, ref_model, generated, attention, advantages, prompt_width, prefix_len, config, optimizer, device, dist, allowed, sampler_lp,
        replayed, lm, pg_skip, parts,
    )
    for puzzle, tokens in solved_samples:  # after the update: a step never replays its own samples
        replay.add(puzzle, tokens)
    _lap(parts, "post", device)
    t_update = _now(device) - t0 - t_sample
    # global batch statistics: sums over ranks of [reward, solved, malformed, samples, informative, kept, entropy, kl, loss]
    stats = dist.all_reduce_sum_(
        torch.stack(
            [
                all_rewards.sum(),
                (all_rewards >= 1.0).float().sum(),  # correct answers can add process credit
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
    info = {
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
    if config.process_reward:
        info["process_credit"] = sum(process_credit(t, p) for t, p in zip(completions, expanded)) / len(completions)
    if parts is not None:
        parts.pop("_t")
        info.update(parts)
    if grad_norm is not None:
        info["grad_norm"] = grad_norm.item()  # before clipping; the same on every rank (grads are all-reduced)
    if mismatch is not None:
        info.update(_mismatch_stats(mismatch, dist, config.tis_cap))
    if lm is not None:
        info["lm_loss"] = lm.loss
    if replay is not None:
        tag = "hs" if hindsight else "sil"
        info.update({f"{tag}_rows": 0 if replayed is None else replayed.ids.shape[0], f"{tag}_nll": 0.0 if replayed is None else replayed.nll,
                     f"{tag}_added": len(solved_samples), f"{tag}_buffer": len(replay)})
        if hindsight:
            info["hs_share"] = len(solved_samples) / n_sampled  # of all this step's samples, those that gave a usable relabelled pair
    return info


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
    limit = config.eval_max_new_tokens or config.max_new_tokens
    with torch.no_grad():
        for start in range(0, len(local), config.eval_batch_size):
            batch = local[start : start + config.eval_batch_size]
            n_real = len(batch)
            if config.compile and n_real < config.eval_batch_size:
                batch = batch + [batch[0]] * (config.eval_batch_size - n_real)
            encoded, _ = _encode_prompts(tokenizer, batch, config, device, config.eval_prompt_style or config.prompt_style)
            if _is_vllm(caches):
                k = max(1, config.eval_samples)
                generated, _ = caches.generate(encoded, k, 1.0 if config.eval_samples else 0.0, limit, tokenizer.pad_token_id,
                                               puzzles=batch if config.constrain_answer else None, think_budget=config.think_budget)
                texts = tokenizer.batch_decode(generated[: n_real * k, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
                local_completions.extend(texts[i * k : (i + 1) * k] if config.eval_samples else texts[i] for i in range(n_real))
                continue
            assert not config.eval_samples, "eval_samples needs rollout_backend='vllm' or 'native'"
            assert not config.constrain_answer, "constrain_answer needs rollout_backend='vllm' or 'native'"
            generated = model.generate(
                **encoded,
                max_new_tokens=limit,
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


def _compile_layers(model, backend: str = "inductor") -> int:
    """compile_update: compile each decoder layer of `model` in place (`nn.Module.compile`, so parameter names and the
    samplers' weight copies are unchanged), with dynamic batch and length; returns the number of layers.

    The layers share one set of graphs: the layer index stays an unspecialised int
    (allow_unspec_int_on_nn_module), and the KV-cache update, which indexes the cache's per-layer lists by it, runs
    eagerly (a graph break on each side). Without both the graph count grows with the depth.
    """
    import torch._dynamo
    from transformers.cache_utils import DynamicCache

    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    if not getattr(DynamicCache.update, "_rlvr_eager", False):
        eager_update = torch.compiler.disable(DynamicCache.update)
        eager_update._rlvr_eager = True
        DynamicCache.update = eager_update
    layers = model.base_model.layers
    for layer in layers:
        layer.compile(backend=backend, dynamic=True)
    return len(layers)


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


_STEP_RE = re.compile(r"(\d+)\s*([-+*/])\s*(\d+)\s*=\s*(?:<<[^>]*>>\s*)?(-?\d+)")


def process_credit(text: str, puzzle) -> float:
    """In [0, 1]: the share of the puzzle's n - 1 combining steps that the <think> text writes as legal
    moves. A step `a op b = c` (GSM8K's `a op b = <<a op b = c>>c` also counts once) is legal when a and b
    are still in the pool (the puzzle's numbers, then earlier results, each used once), op is allowed and
    c is exact; c joins the pool. The first illegal step ends the chain, so restating or inventing
    numbers earns nothing. It checks the model's own arithmetic and never searches for a solution."""
    think = text.split("</think>")[0].split("<answer>")[0]
    think = think.replace("×", "*").replace("÷", "/").replace("−", "-")
    pool = Counter(puzzle.numbers)
    legal = 0
    for a, op, b, c in _STEP_RE.findall(think):
        a, b, c = int(a), int(b), int(c)
        need = Counter([a, b])
        if op not in puzzle.ops or need & pool != need:
            break
        exact = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b and a % b == 0 else None}[op]
        if exact != c:
            break
        pool -= need
        pool[c] += 1
        legal += 1
    return min(1.0, legal / max(1, len(puzzle.numbers) - 1))


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
        names = tuple(config.train_ladder_levels.split(",")) if config.train_ladder_levels else LADDER_LEVELS
        allowed = LADDER_LEVELS + MINI_LEVEL_NAMES  # the mini levels (3 numbers) are training-only and off unless named
        assert set(names) <= set(allowed), f"train_ladder_levels must be a subset of {allowed}"
        stream = ladder_stream(config.seed + offset * 10_000_000, names, exclude=ladder_keys(), fast=config.fast_puzzle_draw)
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


def _draw_puzzles(config: GRPOConfig, step: int, rng: random.Random, stream, easy_streams) -> list:
    """The step's puzzles: each from the main stream or, early in the curriculum, from one of the easy streams."""
    easy_frac = max(0.0, 1.0 - step / config.curriculum_steps) if config.curriculum_steps > 0 else 0.0
    puzzles = []
    for _ in range(config.prompts_per_step * max(1, config.oversample)):
        u = rng.random()
        source = stream if u >= easy_frac else easy_streams[int(u >= easy_frac / 2)]
        puzzles.append(next(source))
    return puzzles


class _DrawAhead:
    """prefetch_puzzle_draw: one background thread draws step s + 1's puzzles while step s runs. The thread owns
    the streams and its own copy of the mix rng; `get` returns a step's puzzles and the mix rng's state after them,
    which the caller sets on its mix rng, so saved states (and resumes) are as without the thread."""

    def __init__(self, config: GRPOConfig, mix_rng: random.Random, stream, easy_streams, first_step: int):
        self.config, self.stream, self.easy_streams = config, stream, easy_streams
        self.rng = random.Random()
        self.rng.setstate(mix_rng.getstate())
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="draw")
        self.next_step = first_step
        self.future = self.pool.submit(self._draw, first_step)

    def _draw(self, step: int):
        puzzles = _draw_puzzles(self.config, step, self.rng, self.stream, self.easy_streams)
        return puzzles, self.rng.getstate()

    def get(self, step: int):
        assert step == self.next_step, (step, self.next_step)
        puzzles, state = self.future.result()
        self.next_step = step + 1
        if step < self.config.max_steps:
            self.future = self.pool.submit(self._draw, step + 1)
        return puzzles, state

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


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


def _cpu_weights(model, into: dict | None = None) -> dict:
    """A detached copy of the model's weights in CPU memory (keep_best_dev), written over `into` when given."""
    if into is None:
        return {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
    for name, tensor in model.state_dict().items():
        into[name].copy_(tensor)
    return into


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
    if config.skip_zero_advantage and (config.kl_coef != 0 or config.cov_clip_frac > 0 or dist.world_size > 1):
        raise ValueError("skip_zero_advantage needs kl_coef == 0, cov_clip_frac == 0 and one process (only then do the skipped rows carry no gradient)")
    if config.pos_adv_scale < 0:
        raise ValueError(f"pos_adv_scale must be >= 0, got {config.pos_adv_scale}")
    if config.tis_cap < 0:
        raise ValueError(f"tis_cap must be >= 0, got {config.tis_cap}")
    if _wants_sampler_logprobs(config) and config.rollout_backend != "vllm":
        raise ValueError("log_mismatch and tis_cap need rollout_backend='vllm' (the sampler's log-probs come from vLLM)")
    if config.overlong_coef and not 0 < config.overlong_buffer <= config.max_new_tokens:
        raise ValueError(f"overlong_coef needs 0 < overlong_buffer <= max_new_tokens, got {config.overlong_buffer}")
    if config.overlong_correct_only and not config.overlong_coef:
        raise ValueError("overlong_correct_only needs overlong_coef")
    if config.malformed_reward_until > 0 and config.malformed_reward is None:
        raise ValueError("malformed_reward_until needs malformed_reward (it switches from it to the verifier's -0.1)")
    if config.vllm_engine_args:
        if config.rollout_backend != "vllm":
            raise ValueError("vllm_engine_args needs rollout_backend='vllm'")
        if not isinstance(json.loads(config.vllm_engine_args), dict):
            raise ValueError(f"vllm_engine_args must be a JSON object, got {config.vllm_engine_args!r}")
    if config.constrain_batched and not (config.constrain_answer and config.rollout_backend == "vllm"):
        raise ValueError("constrain_batched needs constrain_answer and rollout_backend='vllm'")
    if config.reuse_sampler_allowed and not (config.constrain_answer and config.rollout_backend == "native"):
        raise ValueError("reuse_sampler_allowed needs constrain_answer and rollout_backend='native'")
    if config.factored_grammar and not (config.constrain_answer and config.rollout_backend == "native"):
        raise ValueError("factored_grammar needs constrain_answer and rollout_backend='native'")
    if config.grammar_fast_path and not (config.constrain_answer and config.rollout_backend == "native"):
        raise ValueError("grammar_fast_path needs constrain_answer and rollout_backend='native'")
    if config.budget_steps < 0:
        raise ValueError(f"budget_steps must be >= 0, got {config.budget_steps}")
    if config.budget_steps_time_stop and not (config.budget_steps > 0 and config.time_budget_s > 0):
        raise ValueError("budget_steps_time_stop needs budget_steps and time_budget_s")
    if config.entropy_coef < 0 or config.entropy_target < 0:
        raise ValueError(f"entropy_coef and entropy_target must be >= 0, got {config.entropy_coef} and {config.entropy_target}")
    if config.eval_max_new_tokens < 0 or (config.eval_max_new_tokens and config.cache_len):
        raise ValueError(f"eval_max_new_tokens must be >= 0 and needs cache_len 0, got {config.eval_max_new_tokens}")
    if config.keep_best_dev and (dist.world_size > 1 or Path(config.eval_set).name == Path(benchmark.LADDER_TEST_PATH).name):
        raise ValueError("keep_best_dev needs one process and an in-loop eval set that is not the test ladder (RULES v3.4)")
    if config.rollback_drop > 0 and not config.keep_best_dev:
        raise ValueError("rollback_drop needs keep_best_dev")
    if config.sil_rows < 0 or config.sil_coef < 0 or (config.sil_rows > 0 and (config.sil_buffer < 1 or dist.world_size > 1)):
        raise ValueError(f"sil_rows and sil_coef must be >= 0, and self-imitation needs sil_buffer >= 1 and one process, got {config.sil_rows}, {config.sil_coef}, {config.sil_buffer}")
    if config.hindsight_rows < 0 or config.hindsight_coef < 0 or (
        config.hindsight_rows > 0 and (config.sil_rows > 0 or config.sil_buffer < 1 or dist.world_size > 1 or config.train_mix != "ladder")
    ):
        raise ValueError("hindsight_rows and hindsight_coef must be >= 0, and hindsight relabelling needs sil_rows 0, sil_buffer >= 1, "
                         f"train_mix 'ladder' and one process, got {config.hindsight_rows}, {config.hindsight_coef}, {config.sil_rows}, "
                         f"{config.sil_buffer}, {config.train_mix!r}")
    if config.lm_rows < 0 or config.lm_coef < 0 or config.lm_seq_len < 1 or (
        config.lm_rows > 0 and (dist.world_size > 1 or not Path(config.lm_tokens).is_file())
    ):
        raise ValueError(f"lm_rows and lm_coef must be >= 0 and lm_seq_len >= 1, and pretraining during RL needs one process and an "
                         f"lm_tokens file, got {config.lm_rows}, {config.lm_coef}, {config.lm_seq_len}, {config.lm_tokens!r}")
    late_config = replace(config, malformed_reward=None) if config.malformed_reward_until > 0 else config
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

        engine = VLLMRollout(config.model, 512 + max(config.max_new_tokens, config.eval_max_new_tokens), config.vllm_gpu_memory, config.seed + 1_000_003 * dist.rank,
                             constrain_answer=config.constrain_answer, logprobs=_wants_sampler_logprobs(config),
                             engine_args=json.loads(config.vllm_engine_args) if config.vllm_engine_args else None,
                             constrain_batched=config.constrain_batched)
    elif config.rollout_backend not in ("hf", "native"):
        raise ValueError(f"unknown rollout_backend {config.rollout_backend!r} (expected 'hf', 'vllm' or 'native')")
    model, device = load_causal_model(config.model, config.device, "fp32" if config.master_fp32 else config.dtype)
    if config.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if config.compile_update:
        if config.rollout_backend == "hf" or config.grad_checkpoint:
            raise ValueError("compile_update needs rollout_backend 'native' or 'vllm' and no grad_checkpoint")
        _compile_layers(model)
    if config.update_capture and (not config.update_host_index or config.micro_batch_tokens <= 0 or config.grad_checkpoint or config.compile_update):
        raise ValueError("update_capture needs update_host_index and micro_batch_tokens > 0, and no grad_checkpoint or compile_update")
    if config.update_capture_width_step < 1 or config.update_capture_row_step < 0 or config.update_capture_group_step < 1:
        raise ValueError("update_capture's steps: width_step and group_step at least 1, row_step at least 0")
    ref_model = None
    if config.kl_coef != 0:
        ref_name = config.ref_model or config.model
        ref_model, _ = load_causal_model(ref_name, str(device), config.dtype)
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    stream, easy_streams = _train_streams(config)
    mix_rng = random.Random(config.seed)
    replay = _Replay(config.sil_buffer, config.sil_by_level, config.seed) if config.sil_rows > 0 else None  # empty, also after a resume
    if config.hindsight_rows > 0:  # the same buffer, holding relabelled pairs; never one in a ladder set
        replay = _Replay(config.sil_buffer, config.sil_by_level, config.seed, ladder_keys())
    lm_data = np.memmap(config.lm_tokens, dtype=np.uint16, mode="r") if config.lm_rows > 0 else None
    state_path = Path(config.resume_dir) / "state.pt" if config.resume_dir else None
    resumed = None
    if state_path is not None and state_path.exists():
        resumed = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed.pop("model"))  # pop: free the loaded copies (a second fp32 model would not fit)
        optimizer.load_state_dict(resumed.pop("optimizer"))
        if resumed.get("kept"):  # keep_best_dev's copy lives in CPU memory
            resumed["kept"]["state"] = {name: tensor.cpu() for name, tensor in resumed["kept"]["state"].items()}
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
    if config.rollout_backend == "native":  # after the model (and any resume), whose weights it copies
        from .fast_rollout import NativeRollout

        engine = NativeRollout(model, tokenizer, 512 + max(config.max_new_tokens, config.eval_max_new_tokens), config.seed + 1_000_003 * dist.rank,
                               constrain_answer=config.constrain_answer, factored_grammar=config.factored_grammar,
                               grammar_fast_path=config.grammar_fast_path)
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
    kept = None  # keep_best_dev: {"step", "pass_rate", "state"} of the best in-loop eval so far
    if resumed:
        start -= resumed["wall_s"]
        eval_s, total_steps, final_pass_rate = resumed["eval_s"], resumed["step"], resumed["final_pass_rate"]
        first_step = resumed["step"] + 1
        kept = resumed.pop("kept", None) if config.keep_best_dev else None
        if engine is not None:
            engine.sync(model)
    last_save = time.perf_counter() - start - eval_s
    last_entropy = None  # the previous step's logged entropy, for entropy_target
    ahead = _DrawAhead(config, mix_rng, stream, easy_streams, first_step) if config.prefetch_puzzle_draw else None
    try:
        for step in range(first_step, config.max_steps + 1):
            draw_start = time.perf_counter()
            if ahead is not None:
                puzzles, mix_state = ahead.get(step)
                mix_rng.setstate(mix_state)
            else:
                puzzles = _draw_puzzles(config, step, mix_rng, stream, easy_streams)
            if scheduled:
                spent = (step - 1) / config.budget_steps * config.time_budget_s if config.budget_steps else time.perf_counter() - start - eval_s
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(config, spent)
            step_start = time.perf_counter()
            loop_parts = {"t_part_draw": step_start - draw_start} if config.time_update_parts else {}
            step_config = late_config if 0 < config.malformed_reward_until <= step else config
            if config.entropy_coef > 0 and config.entropy_target > 0 and not (last_entropy is not None and last_entropy < config.entropy_target):
                step_config = replace(step_config, entropy_coef=0.0)
            lm = None if lm_data is None else _lm_batch(lm_data, config, step, model.get_input_embeddings().num_embeddings, device)
            info = grpo_step(model, ref_model, tokenizer, dist.shard(puzzles), step_config, optimizer, device, caches, dist, replay, lm)
            last_entropy = info["entropy"]
            if config.entropy_coef > 0:
                info["entropy_coef"] = step_config.entropy_coef
            if engine is not None:
                sync_start = time.perf_counter()
                engine.sync(model)
                if config.time_update_parts:  # the weight sync's synced wall time, next to the update's part times
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    loop_parts["t_part_sync"] = time.perf_counter() - sync_start
            step_elapsed = time.perf_counter() - step_start
            elapsed = time.perf_counter() - start
            total_steps = step
            tokens = info.pop("tokens")
            if config.time_update_parts:  # the loop's own part times and the update's token count, for its MFU
                info.update(loop_parts, t_part_step=step_elapsed, update_tokens=tokens)
            if config.sample_every and step % config.sample_every == 0 and dist.is_main:
                with (out_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
                    for text, r, puzzle in zip(info["completions"], info["rewards"], info["puzzles"]):
                        f.write(json.dumps({"step": step, "numbers": list(puzzle.numbers), "target": puzzle.target,
                                            "reward": r, "completion": text}) + "\n")
            info = {key: value for key, value in info.items() if key not in {"completions", "rewards", "puzzles"}}
            info.update({"step": step, "wall_s": elapsed, "train_s": elapsed - eval_s, "tokens/s": tokens / max(step_elapsed, 1e-9)})
            if scheduled:
                info["lr"] = optimizer.param_groups[0]["lr"]
            if config.budget_steps:
                info["clock_s"] = step / config.budget_steps * config.time_budget_s
            out_of_time = False
            if config.budget_steps:
                out_of_time = step >= config.budget_steps
            if config.time_budget_s > 0 and (not config.budget_steps or config.budget_steps_time_stop):
                # rank 0 decides so all ranks stop on the same step
                flag = torch.tensor(float(dist.is_main and elapsed - eval_s >= config.time_budget_s), device=device)
                out_of_time = dist.all_reduce_sum_(flag).item() > 0 or out_of_time
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
                if config.keep_best_dev:  # the eval and the copy stay on the clock (RULES v3.4)
                    if kept is None or result["pass_rate"] >= kept["pass_rate"]:
                        kept = {"step": step, "pass_rate": result["pass_rate"], "state": _cpu_weights(model, kept and kept["state"])}
                    elif config.rollback_drop > 0 and result["pass_rate"] < kept["pass_rate"] - config.rollback_drop:
                        model.load_state_dict(kept["state"])
                        if engine is not None:
                            engine.sync(model)
                        result["rolled_back_to"] = kept["step"]
                    result["kept_step"] = kept["step"]
                else:
                    eval_s += time.perf_counter() - eval_start
                now = time.perf_counter() - start
                result.update({"step": step, "wall_s": now, "train_s": now - eval_s})
                if config.budget_steps:
                    result["clock_s"] = info["clock_s"]
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
                    "step": step, "wall_s": save_start - start, "eval_s": eval_s, "final_pass_rate": final_pass_rate,
                    **({"kept": kept} if config.keep_best_dev else {})})
                eval_s += time.perf_counter() - save_start  # off the clock, like evals
                last_save = train_s
            if out_of_time:
                break
    finally:
        if ahead is not None:
            ahead.close()
        if train_log is not None:
            train_log.close()
        if eval_log is not None:
            eval_log.close()
    last_pass_rate = final_pass_rate
    if kept is not None:  # keep_best_dev: back to the best eval's weights, before the clock stops
        if kept["step"] != total_steps:
            model.load_state_dict(kept["state"])
        final_pass_rate = kept["pass_rate"]
    total_wall = time.perf_counter() - start
    if not config.no_save and dist.is_main:
        model.save_pretrained(out_dir / "final")
        tokenizer.save_pretrained(out_dir / "final")
    if config.compile_update and dist.is_main:
        from torch._dynamo.utils import counters

        print(f"compile_update: dynamo stats {dict(counters['stats'])}, graph breaks {sum(counters['graph_break'].values())}", flush=True)
    if config.update_capture and dist.is_main and model in _CAPTURES:
        print(f"update_capture: {_CAPTURES[model].stats}", flush=True)
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
    if config.keep_best_dev:
        result.update({"kept_step": kept["step"] if kept else None, "last_pass_rate": last_pass_rate})
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
