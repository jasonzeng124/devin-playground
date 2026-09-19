# GPU GRPO smoke record — Qwen2.5-0.5B

Pipeline smoke test on a single RTX 4090 (RunPod `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`),
not a real leaderboard entry: 1 seed, 60 steps, no threshold reached.

Command:

```bash
python -m rlvr_speedrun.grpo --model Qwen/Qwen2.5-0.5B \
  --group-size 8 --prompts-per-step 16 --micro-batch-size 16 --few-shot 3 \
  --max-new-tokens 64 --lr 2e-6 --kl-coef 0.02 --max-steps 60 \
  --eval-every 20 --eval-limit 200 --eval-batch-size 64 --device cuda --seed 0
```

Result: 60 steps in 272 s (~4.5 s/step, ~1.9k completion tokens/s). Malformed rate
on the 200-puzzle eval fell 0.035 -> 0.005 while pass rate stayed within noise
(0.020 -> 0.005, CI overlaps the base rate). With ~2.6% base solve rate at T=1.0,
most groups have zero reward variance, so the gradient is dominated by the
malformed penalty. Lower temperature, larger groups, or an easier curriculum
should be the first real record.
