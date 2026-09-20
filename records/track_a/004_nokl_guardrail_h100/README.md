# 004 — record 003 recipe with the capability guardrail measured

Identical hyperparameters to [003](../003_nokl_deferred_eval_h100) (KL=0, no
reference model, curriculum 200, evals from step 150). Re-run so that every
seed's final checkpoint is probed with `rlvr_speedrun.capability` against the
base model; probe time is not on the clock.

| seed | time to 10% | steps | final pass rate | FineWeb-Edu loss Δ | MMLU-lite Δ |
|-----:|------------:|------:|----------------:|-------------------:|------------:|
| 0 | 778.2 s | 325 | 0.1015 | +0.0010 | +0.000 |
| 1 | 640.8 s | 275 | 0.1130 | +0.0014 | +0.002 |
| 2 | 546.8 s | 250 | 0.1165 | +0.0012 | +0.000 |
| 3 | 785.3 s | 350 | 0.1115 | +0.0010 | +0.006 |
| 4 | 1075.5 s | 425 | 0.1015 | +0.0010 | +0.002 |
| 5 | 1548.2 s | 600 | 0.1010 | +0.0012 | +0.008 |
| 6 | 1535.8 s | 600 | 0.1020 | +0.0014 | -0.004 |
| 7 | 862.8 s | 350 | 0.1050 | +0.0014 | +0.004 |
| 8 | 748.8 s | 300 | 0.1070 | +0.0011 | +0.006 |
| 9 | 1059.3 s | 425 | 0.1010 | +0.0017 | +0.000 |

**958.1 ± 348.2 s** (N=10, 95% CI [709, 1207] s); guardrail passes on all
seeds (limits: +0.05 nats, −0.03 acc). Base: FineWeb-Edu 2.5937 nats,
MMLU-lite 0.476.

As first submitted with seeds 0-2 this record read **655.3 ± 116.3 s**. Seeds
3-9 were added afterwards to measure the variance of the recipe (see
"Seeds 3-9" below) and roughly doubled the estimate: the first three seeds
were lucky draws. Steps to threshold range from 250 to 600 (the cap) and the
distribution is right-skewed — two seeds crawl at 9-10% for hundreds of
steps before crossing.

With ~250-600 steps of lr 5e-6 and no KL the policy barely moves off the base
on general text, so the guardrail is far from binding at this threshold. It
will matter once records push harder (higher LR, longer runs, or a raised
threshold).

## Seeds 3-9 (added later; N: 3 -> 10)

Run on a RunPod `NVIDIA H100 80GB HBM3` (SXM) pod, torch 2.5.1+cu124,
transformers 4.57.6, Python 3.11, driver 580.126, with the trainer at commit
`fa9e0c3` (the pre-005 trainer used for seeds 0-2, which ran on Modal H100
80GB HBM3 with torch 2.5.1 / transformers 4.57). Step time is the same on both providers
(2.3-2.6 s/step). This trainer predates `warmup_s` / `environment` in
`result.json`; it does no untimed warm-up (eager generation), so the clock
starts at the first rollout with cold kernels.

The only config difference is `--max-steps 600` instead of 400. `max_steps` is
a censoring cap, not a hyperparameter (RULES.md): a seed's trajectory and
time-to-threshold do not depend on it. It was raised because seed 1 of record
005 had already shown that 400 is too low for this recipe; seeds 5 and 6
crossed 10% exactly at step 600 and would have been censored under the old
cap. `config.json` now says 600.

Disclosure: seeds 3-9 were launched once each, no retries, no censored runs,
no exploratory runs of this recipe (it was fixed in 003). GPU time: 7 seeds,
~2.2 GPU-hours including the probes.

## Command

```
modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record5_h100_nokl_probe \
  --args "--model Qwen/Qwen2.5-0.5B --group-size 16 --prompts-per-step 8 --micro-batch-size 32 \
    --few-shot 3 --max-new-tokens 64 --temperature 0.8 --lr 5e-6 --kl-coef 0 \
    --curriculum-steps 200 --max-steps 600 --eval-every 25 --eval-start-step 150 \
    --eval-limit 2000 --eval-batch-size 250 --target-solve-rate 0.1"
```
(`--probe` is on by default; seeds 0-2 were run with `--max-steps 400`.)

Seeds 3-9 were run one at a time on a single pod with the same `--args`, plus
`--seed N --device cuda --out-dir <record>/seeds/N`, followed by
`python -m rlvr_speedrun.capability --model <out-dir>/final --base Qwen/Qwen2.5-0.5B --out <out-dir>/probe.json`.
