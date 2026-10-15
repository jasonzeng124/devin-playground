# Track D 000: baseline

From scratch in 6 h on one H100:

1. **Pretraining, 112 min:** a 110M Llama-style model (`scripts/pretrain_scratch.py`), AdamW,
   on the pinned FineWeb-Edu shards tokenized with the TinyLlama tokenizer
   (`scripts/prepare_track_d_data.py`, CPU, untimed).
2. **RL, 234 min:** Dr. GRPO (`adv_norm: none`), lr 3e-5, 64 prompts x 8 samples, on the ladder
   mix, with answer-constrained decoding in rollouts and 256 new tokens.

Scored with `constrain: true`, T = 1, no think budget.

This is calibration/sweep_d10 `pre120`, shortened to fit the 6 h with ~10 min to spare, and with
dev evals every 500 steps instead of 50. They are excluded from the clock, but at every 50 steps
they would push the job past the cluster's 6 h limit.

## Results

All seeds ran on MIT Engaging, one NVIDIA H100 80GB HBM3 (node2906), with torch 2.13.0+cu130,
transformers 5.17.0, vLLM 0.30.0. The token file was copied to node-local disk before the
clock started (`LOCAL_DATA_DIR`), because random reads from the network filesystem slowed
pretraining ~9x. Training and scoring ran as separate jobs (`STAGE=train`, then `STAGE=confirm`).
The clock column is pretraining + RL = total.

| seed | test ladder | L1 | L2 | L3 | L4 | L5 | L6 | mean tokens | clock (min) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0979 | 0.312 | 0.091 | 0.023 | 0.018 | 0.002 | 0.142 | 28 | 112 + 236 = 348 |
| 1 | 0.1391 | 0.477 | 0.147 | 0.000 | 0.048 | 0.006 | 0.157 | 25 | 112 + 236 = 348 |
| 2 | 0.1120 | 0.349 | 0.080 | 0.056 | 0.013 | 0.002 | 0.172 | 29 | 112 + 236 = 348 |
| 3 | 0.1266 | 0.400 | 0.095 | 0.063 | 0.035 | 0.003 | 0.164 | 21 | 112 + 236 = 348 |
| 4 | 0.1075 | 0.444 | 0.044 | 0.000 | 0.014 | 0.005 | 0.139 | 20 | 112 + 236 = 348 |

Mean test ladder score **0.1166** (sd 0.0163, 5 seeds).

## What the model does

It doesn't reason. Without the constraint it copies the target or prints a constant
(`<answer> 14</answer>`) and scores 0. With the constraint it samples expressions over the given
numbers, shaped like answers, and never writes intermediate arithmetic. FineWeb-Edu has almost no
written arithmetic (~0.005 equations per document), which is the main open problem for this track.
