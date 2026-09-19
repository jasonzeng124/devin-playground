# Unranked reproduction of record 002 on 1x RTX 4090

Same recipe as the parent record (Qwen2.5-0.5B, GRPO, 2->3->4-number
curriculum over 200 steps, group 16, 8 prompts/step, T=0.8, lr 5e-6,
kl 0.02, 64 new tokens), run on a single RTX 4090 (RunPod, torch 2.4.1,
transformers 4.57, bf16) before the H100 record existed. Differences from the
ranked run, which is why this is a reproduction and **not a leaderboard entry**:

- eval on 500 of the 2,000 frozen puzzles (`--eval-limit 500`), not the full set
- fixed 300 steps, `--target-solve-rate 0.2` (never reached, so no early stop)
- `micro_batch_size 16`

| seed | first eval >= 10% (step, wall) | final pass rate @300 |
|-----:|-------------------------------:|---------------------:|
| 0 | step 250, 1262 s | 0.118 |
| 1 | step 225, 1133 s | 0.096 |
| 2 | step 125, 628 s | 0.124 |

~5.0 s/step vs ~2.9 s/step on the H100 (the 4090 also spends less time per
eval because it only scores 500 puzzles). All three seeds learn the same
curve shape as the H100 seeds: near zero until ~step 50, 7-9% plateau during
the curriculum tail, then a lift into 10-13% once the mix is all 4-number
puzzles. Total cost: ~$1.10 of RunPod credit.
