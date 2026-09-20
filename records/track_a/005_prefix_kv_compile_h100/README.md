# 005 — same recipe as 004, 2x faster trainer (Qwen2.5-0.5B, 1x H100)

Pure systems record: RL hyperparameters are identical to
[004](../004_nokl_guardrail_h100) (KL=0, curriculum 200, lr 5e-6, T=0.8,
group 16 x 8 prompts, evals from step 150). What changed is how each step is
executed:

* **Shared-prefix KV reuse in training.** The 3-shot prompt is 420 tokens, of
  which the first 351 are identical for every puzzle. Prompts are laid out as
  `[prefix][pad][puzzle]` (equivalent to left padding once positions follow the
  attention mask), the prefix is run once per micro-batch and its KV cache is
  broadcast to all rows. This is exact under causal attention (tested against
  the full forward to 1e-5) and cuts policy forward/backward tokens ~3.5x.
* **Compiled decode with a persistent static cache** (`--compile`). One
  `StaticCache` per batch shape (rollout: 128 rows, eval: 1000 rows), reset
  between calls, so `torch.compile` fires once per shape during an untimed
  warm-up (93-161 s, see RULES) instead of every rollout/eval switch.
* `--micro-batch-size 128` (one backward per step; memory freed by the prefix
  trick) and `--eval-batch-size 1000`.
* Rollouts sample from the full tempered softmax (`top_k=0`); transformers 4.x
  silently defaulted to `top_k=50`, so the sampler now matches the policy
  log-probs used in the loss.

Step time 2.4 s -> 1.18 s (rollout 0.81 s, update 0.36 s); full 2,000-puzzle
eval 15 s -> 9.5 s.

| seed | time to 10% | steps | final pass rate | FineWeb-Edu loss Δ | MMLU-lite Δ |
|-----:|------------:|------:|----------------:|-------------------:|------------:|
| 0 | 417.6 s | 300 | 0.1070 | +0.0016 | +0.004 |
| 1 | 650.5 s | 450 | 0.1090 | +0.0019 | -0.002 |
| 2 | 464.1 s | 325 | 0.1125 | +0.0018 | +0.004 |
| 3 | 522.8 s | 350 | 0.1030 | +0.0016 | +0.006 |
| 4 | 650.2 s | 425 | 0.1065 | +0.0016 | -0.008 |
| 5 | 388.7 s | 275 | 0.1140 | +0.0017 | +0.008 |
| 6 | 344.7 s | 250 | 0.1085 | +0.0013 | +0.012 |
| 7 | 190.0 s | 150 | 0.1040 | +0.0009 | -0.010 |
| 8 | 780.9 s | 525 | 0.1065 | +0.0013 | -0.002 |
| 9 | 191.3 s | 150 | 0.1045 | +0.0012 | +0.000 |

**460.1 ± 195.7 s** (N=10, 95% CI [320, 600] s). Guardrail passes on all
seeds (base: FineWeb-Edu 2.5937 nats, MMLU-lite 0.476).

**vs 004** (958.1 ± 348.2 s, N=10): difference −498 s, one-sided Welch
t = −3.94, df = 14.2, **p = 0.0007**; exact permutation p = 0.0002. 005 is the
record holder. (`python scripts/validate_record.py records/track_a/005_prefix_kv_compile_h100 --compare-to records/track_a/004_nokl_guardrail_h100`)

As first submitted with seeds 0-2 this read 510.7 ± 123.0 s against 004's
655 ± 116 s — a gap that was **not** significant (Welch p = 0.11, permutation
p = 0.15). Seeds 3-9 were added to both records to measure the real
variance; the std is ~200 s here and ~350 s for 004, so N=3 estimates of the
mean are ±250 s at 95% and cannot separate recipes that differ by less.

Steps to threshold: 150-525 here (mean 320) vs 250-600 for 004 (mean 390).
So the speedup is not purely the 2x step time: this trainer also tends to
need fewer steps. The RL hyperparameters are identical; the candidate cause is
the `top_k=0` sampler fix (004 sampled from a top-50-truncated distribution
while computing the loss on the full one). Two seeds (7 and 9) crossed 10% at
the very first eval (step 150), so the `eval_start_step` deferral now costs
time on some seeds — an adaptive eval trigger is an obvious next record.

## Seeds 3-9 (added later; N: 3 -> 10)

Run one at a time on a RunPod `NVIDIA H100 80GB HBM3` (SXM) pod, torch
2.5.1+cu124, transformers 4.57.6, Python 3.11, driver 580.126, with the trainer
at commit `2bc076e` (seeds 0-2 ran on Modal H100 80GB HBM3, torch 2.5.1 /
transformers 4.57, on the pre-review version of the same trainer; the review
fixes in between — RNG restore after warm-up, padding mask across prefix
chunks — do not change the recipe). Step time matches Modal (1.2-1.5 s/step).
Untimed warm-up (`warmup_s`, in `stdout.log`): 67 s for seed 3 (cold inductor
cache), 36-40 s for the rest. Same `--args` as below plus
`--seed N --device cuda --out-dir <record>/seeds/N`, then
`python -m rlvr_speedrun.capability --model <out-dir>/final --base Qwen/Qwen2.5-0.5B --out <out-dir>/probe.json`.

Disclosure: 8 launches for 7 seeds, ~1.2 GPU-hours including probes, no
exploratory runs of this recipe (fixed since 003). Seed 8 was censored once
(below). Under the initial 600-step cap 1 of 7 new seeds was censored
(1 of 10 overall, including the seed-1 attempt below at cap 400).

## Disclosure: seeds 1 and 8 were each run twice

**Seed 1.** The first 3-seed launch used `--max-steps 400`; seed 1 ended at
8.55% after 400 steps (579 s) without reaching threshold (logs in
`extra/seed1_first_attempt_400steps/`). Seed 1 was re-run with
`--max-steps 600` and reached 10.9% at step 450; that run is the one in
`seeds/1`.

**Seed 8.** First attempt at `--max-steps 600` was censored: 9.6% at step 600
(904 s wall; `extra/seed8_censored_600steps/`). Re-run with `--max-steps 1200`
and otherwise identical arguments, it crossed 10.65% at step 525 (780.9 s);
that run is in `seeds/8`. The two attempts have identical step-1 losses and
diverge from step 2 (the first attempt then sat at 7.25% held-out from step
200 to 450), so the rerun is a **fresh draw, not a continuation**: on GPU with
`torch.compile` and bf16 reductions the trainer is not bit-reproducible for a
given seed. This is exactly the downward bias RULES.md warns about.
Sensitivity check: counting seed 8 at its censoring time (904 s, a lower
bound) instead of 780.9 s gives 472.4 ± 220.5 s and Welch p = 0.0010 against
004 — the verdict does not change.

`max_steps` is only a cap — the clock is time-to-threshold — but a seed that
never gets there is a failed seed, so all attempts are kept here.

## Command

```
modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record9_h100_prefixcache_compile \
  --args "--model Qwen/Qwen2.5-0.5B --group-size 16 --prompts-per-step 8 --micro-batch-size 128 \
    --few-shot 3 --max-new-tokens 64 --temperature 0.8 --lr 5e-6 --kl-coef 0 \
    --curriculum-steps 200 --max-steps 600 --eval-every 25 --eval-start-step 150 \
    --eval-limit 2000 --eval-batch-size 1000 --target-solve-rate 0.1 --compile --pad-to-multiple 64"
```

## Negative results along the way (same trainer, 3 seeds each, all pass the guardrail)

| experiment | time to 10% | notes |
|---|---:|---|
| dynamic sampling (`--oversample 4`, keep 8 most informative of 32 groups), T=0.8 | 1226 / 354 / 477 s | 4x rollout tokens per step; two seeds converge in 100 steps, one needs 250 |
| same, T=1.0 | 1365 / 1121 / 870 s | in-batch solve rate high early, entropy collapses, held-out lags |
| Dr. GRPO advantages (`--adv-norm none`) + oversample 4 + compile, T=0.8 | 1187 / 886 / 783 s | no better than std-normalised advantages here |

Oversampling buys informative groups but at this model size the base solve
rate on 2-3-number puzzles is already high enough that most groups are
informative anyway; the extra generation is not paid back.
