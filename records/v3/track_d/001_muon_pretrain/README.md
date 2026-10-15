# Track D 001: Muon pretraining (not a record)

Changes from 000_baseline: pretraining uses Muon at lr 4e-3 for 56 min instead of AdamW for
112 min (`--optimizer muon --lr 4e-3`, already in `scripts/pretrain_scratch.py`). The 56 saved
minutes go to RL (290 min instead of 234). The RL recipe and the sampler are unchanged.

In the pilots (H200), Muon at 60 min reached the same validation loss and dev ladder score as
AdamW at 120 min (calibration/sweep_d11, sweep_d13). With 5 h of RL after 60 min of Muon,
`muon60_rl5h` reached dev 0.172 (best 0.183), against 0.175 / 0.167 / 0.137 for the
baseline recipe. The pilot gain is within seed noise, so this record tests whether extra RL time
helps once RL is near its plateau.

## Results

All seeds ran on MIT Engaging, one NVIDIA H100 80GB HBM3 (node2906), with torch 2.13.0+cu130,
transformers 5.17.0, vLLM 0.30.0: the same machine and software as 000_baseline. Data preparation ran before the clock started. The clock column is
pretraining + RL = total.

| seed | test ladder | L1 | L2 | L3 | L4 | L5 | L6 | mean tokens | clock (min) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1241 | 0.451 | 0.078 | 0.000 | 0.018 | 0.007 | 0.192 | 23 | 56 + 292 = 348 |
| 1 | 0.1001 | 0.317 | 0.068 | 0.041 | 0.022 | 0.011 | 0.142 | 28 | 56 + 292 = 348 |
| 2 | 0.1395 | 0.474 | 0.101 | 0.064 | 0.025 | 0.004 | 0.168 | 21 | 56 + 292 = 348 |
| 3 | 0.1078 | 0.355 | 0.079 | 0.048 | 0.021 | 0.004 | 0.140 | 22 | 56 + 292 = 348 |
| 4 | 0.1096 | 0.374 | 0.101 | 0.014 | 0.036 | 0.002 | 0.130 | 24 | 56 + 292 = 348 |

Mean test ladder score **0.1162** (sd 0.0156, 5 seeds).

**Not a record.** Against 000_baseline (0.1166, sd 0.0163): Welch t = -0.04, one-sided
p = 0.52 (`scripts/validate_record_v3.py --compare-to`). The two recipes score the same.

On the H100, 56 min of Muon reached a final validation loss of 2.844-2.887 (mean 2.869),
against 2.806-2.852 (mean 2.825) for 112 min of AdamW. In the H200 pilots the two matched.
The extra RL time gave more steps (13,200-16,200 against 11,100-13,600) but no better score,
which agrees with RL being near its plateau by ~4 h. Kept as a negative result: halving the
pretraining time with Muon costs nothing here, so the saved minutes are free for other uses.
