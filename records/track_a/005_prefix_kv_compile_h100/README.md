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

**510.7 ± 123.0 s** (004: 655 ± 116 s; 003: 573 ± 214 s). Guardrail passes on
all seeds (base: FineWeb-Edu 2.5937 nats, MMLU-lite 0.476).

The gain is smaller than the 2x step-time reduction because these seeds needed
more steps (300/450/325 vs 325/275/250 in 004) — with three seeds, steps-to-
threshold noise is still the dominant term. Learning curves (in-batch solve
rate, malformed rate) are indistinguishable from 003/004, as expected for a
change that leaves the RL math untouched.

## Disclosure: seed 1 was run twice

The first 3-seed launch used `--max-steps 400`; seed 1 ended at 8.55% after 400
steps (579 s) without reaching threshold (logs in
`extra/seed1_first_attempt_400steps/`). Seed 1 was re-run with
`--max-steps 600` and reached 10.9% at step 450; that run is the one in
`seeds/1`. `max_steps` is only a cap — the clock is time-to-threshold — but a
seed that never gets there is a failed seed, so both runs are kept here.

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
