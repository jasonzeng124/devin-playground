# 003 — no KL, deferred eval (Qwen2.5-0.5B, 1x H100)

Record 002's recipe with two changes, both pure clock savings:

* `--kl-coef 0`: no reference model is loaded, so each micro-batch does one
  0.5B forward pass instead of two. Step time 2.9 s -> 2.4 s.
* `--eval-start-step 150`: the first 5 full evals (~15 s each) are skipped;
  no seed in record 002 was anywhere near 10% before step 150.

| seed | time to 10% | steps | final pass rate |
|-----:|------------:|------:|----------------:|
| 0 | 503.4 s | 200 | 0.1060 |
| 1 | 812.5 s | 325 | 0.1095 |
| 2 | 401.6 s | 175 | 0.1005 |

**572.5 ± 214.2 s** (record 002: 783.6 ± 215.7 s). Seed-to-seed variance is
unchanged and large; seed 1 again lags by ~150 steps.

## Command

```
modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record4_h100_nokl \
  --args "--model Qwen/Qwen2.5-0.5B --group-size 16 --prompts-per-step 8 --micro-batch-size 32 \
    --few-shot 3 --max-new-tokens 64 --temperature 0.8 --lr 5e-6 --kl-coef 0 \
    --curriculum-steps 200 --max-steps 400 --eval-every 25 --eval-start-step 150 \
    --eval-limit 2000 --eval-batch-size 250 --target-solve-rate 0.1"
```

## Caveats

* Capability guardrail (`scripts/capability_probe.py`) is still unimplemented,
  so like 002 this record has no drift measurement. Dropping KL makes that
  guardrail more important, not less — the policy is now unconstrained.
* An earlier attempt with `--curriculum-steps 100` (same KL=0 / deferred eval)
  was *slower* (850 ± 170 s): learning stalled at ~7% once the easy puzzles
  ran out. Curriculum length matters more than step time here.

## Good next records

* Curriculum 250-300, or a slower anneal, may pull seed 1 in.
* Eval only when in-batch solve rate exceeds ~15% instead of a fixed step.
* Higher LR now that KL no longer restrains the policy.
