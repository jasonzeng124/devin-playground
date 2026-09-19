# RLVR Speedrun — Rules

The speedrun measures **wall-clock time to teach a language model to solve
Countdown arithmetic puzzles with reinforcement learning from verifiable
rewards (RLVR)**. Everything is scored by a deterministic, CPU-only verifier
(`rlvr_speedrun/countdown.py`). Learned reward models are never allowed — that
would break the "verifiable" in RLVR.

## The task

A Countdown puzzle gives 4 integers in `[1, 25]` and a target in `[10, 200]`.
The model must output an arithmetic expression using `+ - * /` and **each
number exactly once** that evaluates exactly to the target (exact rational
arithmetic — `7 / 2 * 2` is fine, `7 / 2` rounded is not).

* Reward: `+1` correct, `0` well-formed but wrong, `-0.1` malformed
  (unparseable / not a pure arithmetic expression). See `verify()`.
* Held-out eval: `data/countdown_eval.jsonl` — 2,000 puzzles generated once
  from seed `20240601` and frozen. **Never train on it.** Training puzzles are
  sampled from the infinite generator (`train_puzzle_stream`), which uses a
  disjoint seed space.
* Eval protocol: greedy decoding, `--few-shot 3` fixed prompt, `max_new_tokens
  <= 64`. `pass_rate` on all 2,000 puzzles is the headline metric. Report the
  Wilson 95% CI that the eval script prints.

## Tracks

### Track A — fixed base model (RL-only speedrun)

Everyone starts from the **same frozen base checkpoint** (listed in
`records/track_a/README.md`, currently the smallest open model whose few-shot
pass rate is non-zero — see `results/base_sweep.md`). Free to change: RL
algorithm, reward shaping (must be a deterministic function of the verifier
output), KL schedule, curriculum, group size, rollout batching, prompt
template used *during training*, systems/kernels.

* Clock starts at the first RL optimizer step and stops at the first
  evaluation where `pass_rate >= 0.50` on the full 2,000-puzzle eval set.
  Evaluation time counts.
* **Capability guardrail** (anti-reward-hacking): the final model must stay
  within `0.05` nats of the base model's loss on the fixed FineWeb validation
  shard used by modded-nanogpt (`fineweb_val_000000.bin`, first 10M tokens),
  measured with `scripts/capability_probe.py` (planned — until it lands,
  report base vs. final loss on any 1M-token FineWeb sample and include the
  command).

### Track B — full stack (pretrain + RL speedrun)

Anything goes, including architecture and pretraining data, timed from
**random initialization**. Win condition is a dual threshold, so capability
preservation is part of the objective by construction:

* FineWeb val loss `<= X` **and** Countdown `pass_rate >= Y`.
* `X` and `Y` are calibrated together from the first baseline record, not
  chosen independently; the first accepted Track B record sets them and they
  are then frozen in `records/track_b/README.md`.

## Hardware tiers

Records are grouped by hardware so single-GPU contributors can participate:

| tier | spec |
|------|------|
| `1xGPU` | one 24 GB consumer GPU (RTX 3090/4090) or one 80 GB datacenter GPU — state which |
| `8xH100` | one 8xH100 SXM node (the modded-nanogpt spec) |

Wall-clock times are only comparable within a tier.

## Seeds and statistics

RL is high-variance. Every record must include **N >= 3 seeds**. Report the
mean, standard deviation and 95% CI of `time_to_threshold_s` and final
`pass_rate` (`scripts/validate_record.py` computes these). A record beats the
previous one if its mean time is lower and the CIs do not substantially
overlap; use judgement and say so in the PR.

## Submitting a record

1. Copy `records/track_a/000_smoke_cpu/` as a template.
2. Name it `records/<track>/<NNN>_<short_slug>/` with the next number.
3. Include: `README.md` (what changed, hardware, exact command), `config.json`,
   and `seeds/<seed>/{train_log.jsonl,eval_log.jsonl,result.json}` for each seed.
   Do not commit model weights.
4. Run `python scripts/validate_record.py records/<track>/<NNN>_<slug>` — it
   must pass.
5. Open a PR; put the summary table in the description. Add yourself to the
   leaderboard in `README.md`.

## Forbidden

* Training on, or peeking at, `data/countdown_eval.jsonl`.
* Any learned or LLM-based reward.
* Hard-coding a solver into the model's inference path (the model itself must
  emit the expression).
* Changing the verifier, the eval set, or the prompt used *at evaluation*.
