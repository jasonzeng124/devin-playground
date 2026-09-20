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

**655.3 ± 116.3 s**; guardrail passes on all seeds (limits: +0.05 nats,
−0.03 acc). Base: FineWeb-Edu 2.5937 nats, MMLU-lite 0.476.

The time is within seed noise of 003 (572 ± 214 s); with ~250-325 steps of
lr 5e-6 and no KL the policy barely moves off the base on general text, so the
guardrail is far from binding at this threshold. It will matter once records
push harder (higher LR, longer runs, or a raised threshold).

## Command

```
modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record5_h100_nokl_probe \
  --args "--model Qwen/Qwen2.5-0.5B --group-size 16 --prompts-per-step 8 --micro-batch-size 32 \
    --few-shot 3 --max-new-tokens 64 --temperature 0.8 --lr 5e-6 --kl-coef 0 \
    --curriculum-steps 200 --max-steps 400 --eval-every 25 --eval-start-step 150 \
    --eval-limit 2000 --eval-batch-size 250 --target-solve-rate 0.1"
```
(`--probe` is on by default.)
