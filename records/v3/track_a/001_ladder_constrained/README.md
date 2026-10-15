# Track A 001: ladder mix + answer-constrained decoding

Changes from 000_baseline:

* `train_mix: "ladder"`: training puzzles from the generators of all six ladder levels
  (fresh puzzles; the dev and test ladders are excluded), instead of the plain generator mix;
* `constrain_answer: true`: in rollouts, once the model opens `<answer>`, decoding is restricted
  to well-formed expressions over the puzzle's numbers and operators, each number used once;
* lr 4e-6 instead of 2e-6.

Scored with the same constraint (`constrain: true`, T = 1, no think budget). 000_baseline is
scored with free sampling, so the two records use different samplers; the diagnostic below
measures how much that matters. The reference sampler implements the constraint
(`ladder_confirm.py --constrain`); this record adds no code.

Pilots (H200, 5 h, in-loop dev evals: 600 dev puzzles, greedy). The numbers are the final evals
in the Engaging logs on 2026-10-12; the repo's copies of calibration/sweep_a01 and sweep_a02 are
earlier snapshots (sweep_a02 holds only the configs):

* calibration/sweep_a01, lr 2e-6: the ladder mix alone reached 0.148 and 0.097 (2 seeds), against
  000_baseline's 0.118-0.145; adding the constraint gave 0.062 and 0.050;
* sweep_a01, lr 4e-6 with the constraint (this recipe): 0.235 and 0.233;
* calibration/sweep_a02, the same at lr 6e-6 and 8e-6 (one seed each): both collapsed into
  ~40-token answers within the first hour, and scored 0.208 (at 278 of 300 min; preempted, not yet
  finished) and 0.188.

## Results

All seeds ran on MIT Engaging, one NVIDIA H100 80GB HBM3 (node2906), with torch 2.13.0+cu130,
transformers 5.17.0, vLLM 0.30.0: the same machine and software as 000_baseline. The test ladder
was scored in the same job, on the same GPU, after the clock stopped. The clock counts all of
the GPU wall time except the in-training dev evals (RULES_v3).

| seed | test ladder | L1 | L2 | L3 | L4 | L5 | L6 | pass@16 | mean tokens | steps | clock (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2236 | 0.735 | 0.183 | 0.104 | 0.051 | 0.016 | 0.253 | 0.460 | 198 | 1,068 | 292 |
| 1 | 0.2522 | 0.825 | 0.226 | 0.138 | 0.067 | 0.016 | 0.240 | 0.577 | 453 | 1,036 | 291 |
| 2 | 0.1860 | 0.598 | 0.152 | 0.098 | 0.050 | 0.017 | 0.201 | 0.246 | 40 | 2,514 | 291 |
| 3 | 0.2724 | 0.866 | 0.246 | 0.165 | 0.084 | 0.019 | 0.255 | 0.566 | 466 | 989 | 291 |
| 4 | 0.3064 | 0.943 | 0.324 | 0.171 | 0.104 | 0.019 | 0.277 | 0.547 | 415 | 970 | 291 |

Mean test ladder score **0.2481** (sd 0.0460, 5 seeds). Against 000_baseline (0.1295, sd 0.0098):
Welch t = 5.64, df 4.4, one-sided p = 0.0019, so this is a new record.

pass@16 is the share of puzzles solved by at least one of the 16 samples. For scale, a random
valid expression scores pass@1 0.021 and pass@16 0.195 on the test ladder.

### Seed 2 collapsed

* By minute 60 its policy entropy had fallen to 0.15 nats per token (other seeds: 0.65-0.68), and
  its answers to ~40 tokens. They stayed there: it answers with a single guess, without a search.
* Short rollouts made its steps fast, so it ran 2,514 steps against 970-1,068 for the others.
* It still scores 0.186, above every 000_baseline seed. But its pass@16 (0.246) is barely above a
  random valid expression's: it concentrates probability on a few guesses rather than solving
  more puzzles.
* Seed 0 started to shorten its answers in the last ~40 minutes (from ~430 to ~210 tokens; mean
  198 on the test ladder), so it may have been heading the same way.

Because of seed 2, this recipe's seed sd is 4.7x the baseline's. To be detected with 5 seeds at
80% power, a challenger needs a gain of about +0.08 if it is as noisy as this record, or about
+0.06 if it is as stable as the baseline.

### How much of the gain is the constraint at scoring

calibration/diag_sampler scores the final checkpoints with the other sampler. To keep the test
ladder untouched, the diagnostic runs on the dev ladder (2,400 puzzles x 16 samples, the same
scorer). Dev and test agree closely: 000_baseline seeds 0 and 1 score 0.128 / 0.120 on dev and
0.130 / 0.124 on test with free sampling.

| model | free sampling | with the constraint |
|---|---:|---:|
| base Qwen3-1.7B | 0.003 (dev) | 0.011 (dev) |
| 000_baseline, 5 seeds | 0.1295 (test) | 0.140 (dev) |
| 001, seeds 0, 1, 3, 4 | 0.247 (dev) | 0.264 (test) |
| 001, seed 2 | 0.112 (dev) | 0.186 (test) |
| 001, 5 seeds | 0.220 (dev) | 0.248 (test) |

* The constraint at scoring adds +0.011-0.014 to 000_baseline and about +0.016 to 001's healthy
  seeds. The collapsed seed 2 depends on it much more (+0.074).
* On a common sampler, 001 still beats 000 by about +0.09 (free) to +0.11 (constrained). Most of
  the gain comes from training.

### What is not attributed

001 changes three things at once (training mix, the constraint during training, lr x2), and no
run separates them. At lr 2e-6 the ladder mix alone was no better than the baseline and adding
the constraint halved the score; only the combination at lr 4e-6 worked. The deciding run
(ladder mix, lr 4e-6, no constraint in training) hasn't been done.

The learning rate may also matter for a numerical reason. The trainer updates bf16 weights
directly, with no fp32 master copy, so an update of about the learning rate only lands on
weights smaller than roughly lr x 512. After the full run, 4.4% of the weights differ from the
base model here, against 2.2% for 000_baseline (seeds 0 and 2 here, seeds 0 and 3 there).
Doubling the learning rate roughly doubles how many weights can move.

## What the model does

Seeds 0, 1, 3 and 4 do the same guess-and-check search as the baseline, more briefly and with
more structure ("22 - (7 - 3) - 2 = ... 16 (close, but not 14)"), then answer. Inside
`<answer>` the constraint only allows well-formed expressions over the puzzle's numbers, so a
draft that misuses a number is redirected into a valid expression. Seed 2 skips the search and
answers with a single guess.

The constraint switches off if the token that completes `<answer>` carries extra text (for
example `>` followed by a newline). This never happened in the 1,044 logged eval examples of the
v3 records.
