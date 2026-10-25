# Track D 002: DeepMind arithmetic in pretraining, fp32 RL with skipping

Changes from 000_baseline:

1. **Pretraining data (112 min; same model, optimizer and schedule):** DeepMind
   mathematics_dataset arithmetic, as published, is added to the baseline's FineWeb-Edu tokens.
   It makes up 25.8 % of the token file: two copies of its 0.30 B tokens beside FineWeb-Edu's
   1.72 B, 2.32 B in all. Pretraining samples uniformly from the file and reads 2.47-2.49 B tokens
   in its 112 minutes.
2. **RL (234 min budget):**
   - fp32 master weights (`master_fp32: true`; forward and backward passes under bf16 autocast)
     at lr 3e-6, instead of 000's bf16 weights at lr 3e-5.
   - Rows whose advantage is zero (every sample in the group got the same reward) carry no
     gradient, so their forward and backward passes are skipped (`skip_zero_advantage: true`). The
     update is unchanged, and the time saved goes to more steps.
   - Everything else is 000's RL: Dr. GRPO, 64 prompts x 8 samples on the ladder mix,
     answer-constrained rollouts, 256 new tokens.

Scored as 000_baseline: `constrain: true`, T = 1, no think budget.

## Why

FineWeb-Edu has almost no written arithmetic (~0.005 equations per document; 000_baseline's
README), and 000's model never writes intermediate arithmetic. The DeepMind modules are public
question-answer pairs of plain arithmetic, so the pretrained model has seen arithmetic written out.

The research screens leading here scored on the dev ladder (in-loop greedy evals at the end of RL,
600 puzzles; never the test ladder):

| recipe | seeds | final dev |
|---|---|---|
| 000_baseline (its five record seeds) | 5 | 0.119 |
| 000's pretraining, RL with fp32 master weights at lr 3e-6 | 5 | 0.137 ± 0.009 |
| DeepMind mix in pretraining, 000's RL | 1 | 0.118 |
| DeepMind mix in pretraining, RL with fp32 at lr 3e-6 and skipping (this record) | 5 | 0.152 / 0.143 / 0.153 / 0.158 / 0.157 = 0.153 |

The last row's seeds 0-2 started RL from saved pretrained models of the same recipe. Seeds 3-4
pretrained themselves, as the record does, and scored slightly higher.

The DeepMind mix gave no gain with 000's RL in its one seed, but did with the fp32 RL. Skipping
wasn't tested on its own in track D.

## External data

- **DeepMind mathematics_dataset v1.0** (Saxton et al. 2019, "Analysing Mathematical Reasoning
  Abilities of Neural Models").
  - Source: the published release,
    `https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz`, md5
    `c5f9590beaf2d2445b55240ce0dab9a5` (the file that HF `deepmind/math_dataset` loads).
  - Used: the six arithmetic modules (`add_or_sub`, `add_sub_multiple`, `div`, `mixed`, `mul`,
    `mul_div_multiple`), splits train-easy, train-medium and train-hard.
  - The question and answer lines are kept exactly as published, a few consecutive pairs per
    document. Nothing is generated, filtered or rewritten.
  - RULES_v3 ("Both tracks: external data") admits public datasets used as published that aren't
    Countdown-specific.
- **FineWeb-Edu and the TinyLlama tokenizer:** 000_baseline's pinned revisions
  (`scripts/prepare_track_d_data.py`).

Data preparation (CPU, before the clock starts):

```bash
python scripts/prepare_track_d_data.py
python scripts/research/prepare_dm_math.py --out <dir> --tokenizer <pinned TinyLlama tokenizer> \
    --base <prepare_track_d_data.py's fineweb-edu-tinyllama.bin> --fracs 0.2
```

`prepare_dm_math.py` downloads and checks the release, extracts the modules, and tokenizes them
with `scripts/pretrain_scratch.py tokenize`. Then `scripts/math_mix.py mix` writes
`<dir>/mix_0.2.bin`, the token file in `config.json`.

## Results

All five seeds ran on MIT Engaging's `mit_preemptable` partition, one NVIDIA H100 80GB HBM3 each
(every scored run on node2700). The software was torch 2.13.0+cu130, transformers 5.17.0 and vLLM
0.30.0, the same as 000_baseline. The token file was copied to node-local disk before the clock
started. Each seed trained and then scored the test ladder in one job (`STAGE=all`). The clock
column is pretraining + RL = total.

| seed | test ladder | L1 | L2 | L3 | L4 | L5 | L6 | pass@16 | mean tokens | clock (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1496 | 0.471 | 0.121 | 0.083 | 0.024 | 0.000 | 0.199 | 0.183 | 25 | 112.2 + 235.9 = 348.1 |
| 1 | 0.1375 | 0.463 | 0.133 | 0.008 | 0.045 | 0.008 | 0.169 | 0.188 | 24 | 112.2 + 237.3 = 349.5 |
| 2 | 0.1361 | 0.473 | 0.076 | 0.062 | 0.024 | 0.005 | 0.176 | 0.171 | 28 | 112.2 + 236.1 = 348.3 |
| 3 | 0.1857 | 0.563 | 0.154 | 0.080 | 0.074 | 0.016 | 0.227 | 0.234 | 21 | 112.2 + 235.9 = 348.1 |
| 4 | 0.1469 | 0.492 | 0.084 | 0.070 | 0.029 | 0.008 | 0.198 | 0.180 | 25 | 112.2 + 236.2 = 348.4 |

Mean test ladder score **0.1512** (sd 0.0202, 5 seeds). Against 000_baseline (0.1166, sd 0.0163):
Welch t = 2.98, df = 7.7, one-sided p = 0.0092
(`scripts/validate_record_v3.py --compare-to records/v3/track_d/000_baseline`). Three seeds score
above 000's best seed (0.1391); seeds 1 and 2 are just below it and above 000's other four.

In-loop final dev (greedy, 600 puzzles): 0.167 / 0.137 / 0.150 / 0.193 / 0.155, mean 0.160.

Seed 1's RL steps were slower than the other seeds' on the same node: sampling took 1.18 s per
step on average against 0.74-0.83 s. So it ran 9,887 RL steps against 14,106-15,693. Its
pretraining ran at the usual speed (2.47 B tokens). I don't know the cause.

Seed 1 took three attempts. The first two are kept in `extra/`, each with a note:

- `extra/interrupted_1_1`: preempted 110 of its 112 pretraining minutes in, before any RL.
- `extra/failed_1_1`: crashed at its first CUDA call ("CUDA-capable device(s) is/are busy or
  unavailable") on a GPU that failed every job sent to it that afternoon. No training ran.

Both are infrastructure failures, so seed 1 was rerun under the same seed (RULES_v3, "Seeds and
statistics").

## What the model does

Like 000_baseline, it answers at once. All 284 dev-eval examples logged by its five seeds start
with `<answer>`, so they contain no intermediate arithmetic. Test-ladder samples average 21-28
tokens, and none are truncated. Pretraining on written arithmetic didn't make it write any.

It is more accurate at every level (means of five seeds):

| | L1 | L2 | L3 | L4 | L5 | L6 | pass@16 |
|---|---|---|---|---|---|---|---|
| 000_baseline | 0.397 | 0.091 | 0.028 | 0.026 | 0.004 | 0.155 | 0.163 |
| this record | 0.492 | 0.114 | 0.061 | 0.039 | 0.008 | 0.194 | 0.191 |

- Nearly half of the gain is on L1 (+0.095; four numbers, + and -), and about a fifth on L6
  (+0.039).
- L3, which needs division, more than doubles to 0.061.
- L4 and L5 (five and six numbers) stay under 0.04.
