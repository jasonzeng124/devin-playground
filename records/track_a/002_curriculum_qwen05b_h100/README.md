# Record 002 — GRPO + easy-puzzle curriculum, Qwen2.5-0.5B, 1xH100

First real Track A entry (tier `1xH100`). 3 seeds, all reach the 10% threshold
on the full 2,000-puzzle frozen eval.

| seed | time to 10% | steps | final pass rate |
|------|------------:|------:|----------------:|
| 0 | 649 s | 225 | 0.1005 |
| 1 | 1032 s | 375 | 0.1015 |
| 2 | 670 s | 225 | 0.1100 |

**time to 10%: 783 ± 216 s (mean ± std, N=3)**. Base model few-shot pass rate
is 2.6%; malformed rate goes to ~0 within 50 steps.

Hardware: one H100 80GB on Modal (`scripts/modal_run.py`), torch 2.5.1, bf16,
~3.6k completion tokens/s, ~2.9 s/step including reference-model scoring.

Command (seeds 0,1,2 run in parallel on separate H100s):

```bash
modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record2_h100 \
  --args "--model Qwen/Qwen2.5-0.5B --group-size 16 --prompts-per-step 8 \
  --micro-batch-size 32 --few-shot 3 --max-new-tokens 64 --temperature 0.8 \
  --lr 5e-6 --kl-coef 0.02 --curriculum-steps 200 --max-steps 400 \
  --eval-every 25 --eval-limit 2000 --eval-batch-size 250 --target-solve-rate 0.1"
```

## What made it work

Plain GRPO on 4-number puzzles is flat: at T=0.8 the base model solves ~0% of
sampled 4-number puzzles, so nearly every group of 16 has identical reward and
zero advantage (see record 001). The fix is a curriculum over the training
stream only (the eval set is untouched): for step `s < curriculum_steps`, a
fraction `1 - s/200` of prompts are drawn from easy 2-number and 3-number
puzzles (targets 1-50), where the base model solves 5% / 2% of samples. That
creates reward variance immediately; solve rate on 4-number held-out puzzles
follows within ~75 steps.

## Caveats

* Capability guardrail (`scripts/capability_probe.py`) is not yet implemented,
  so base-vs-final FineWeb loss is not reported. KL to the reference model
  stays at ~0.1 nats/token.
* Eval every 25 steps on the full 2,000 puzzles costs ~15 s per eval; this
  is included in the time (as the rules require). Evaluating less often
  is a legal speedup.
* Seed 1 is a clear outlier (plateaued at 7% for 150 steps). Expect high
  variance; the CI is wide.

## Good next records

* Skip evals until in-batch solve rate suggests 10% is close.
* Lower KL coefficient / higher LR — KL is far from binding.
* Tune the curriculum length (200 steps may be too long; seeds 0/2 crossed
  10% only 25 steps after the curriculum ended).
* Drop the reference model forward pass (KL=0) and check the guardrail holds.
