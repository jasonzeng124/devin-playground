# Track A 001: ladder mix + answer-constrained decoding

Changes from 000_baseline:

* `train_mix: "ladder"`: training puzzles from the generators of all six ladder levels
  (fresh puzzles; the dev and test ladders are excluded), instead of the plain generator mix;
* `constrain_answer: true`: in rollouts, once the model opens `<answer>`, decoding is restricted
  to well-formed expressions that use each given number once, so every answer can be checked;
* lr 4e-6 instead of 2e-6.

Scored with the same constraint (`constrain: true`, T = 1, no think budget). The reference
sampler implements the constraint (`ladder_confirm.py --constrain`); this record adds no code.

In the pilots (calibration/sweep_a01, H200, one seed each), the ladder mix alone reached dev
0.148 against the baseline's ~0.12. Adding the constraint at lr 2e-6 got 0.06, and at lr 4e-6 it
got 0.250.

## Results

All seeds ran on MIT Engaging, one NVIDIA H100 80GB HBM3 (node2906), with torch 2.13.0+cu130,
transformers 5.17.0, vLLM 0.30.0: the same machine and software as 000_baseline.

<!-- RESULTS -->

## What the model does

It does the same guess-and-check search as the baseline, more briefly and with more structure
("22 - (7 - 3) - 2 = ... 16 (close, but not 14)"). The constraint means a search line that
reaches the target turns into a valid answer, instead of a malformed one such as
`(22 - 7 - 3 - 2) = 14`.
