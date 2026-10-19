# Autoresearch log

Newest entries last. Tables come from `scripts/summarize_screens.py`: final and best dev score,
first time dev reached 0.10 (`t.10`), steps per minute over the last 30 min, entropy and
informative groups over the last 20 steps, collapse mode, and dev at 30/60/120/180/240 min.

## 2026-10-19: weekend screens (calibration/screens_w42), rules frozen, r1 queued

Screens ran on H100s (dev ladder only), 2026-10-16 to 10-17.

| arm | n | dev final | ok / one-guess / diverged | median t.10 |
|---|---|---|---|---|
| A 001 + fp32 master weights (lr 4e-6) | 5 | 0.106 ± 0.097 | 0/3/2 | 21 min |
| A 001 without the answer constraint | 5 | 0.197 ± 0.055 | 2/3/0 | 54 min |
| A 000 + fp32 master weights (lr 2e-6) | 3 | 0.097 ± 0.022 | 0/3/0 | 168 min |
| D 000 RL + fp32 (paired on D 000's pretrained seeds 1-4) | 4 | 0.035 ± 0.041 | 0/3/1 | never |
| (A 001 record) | 5 | 0.240 ± 0.045 | 4/1/0 | 150 min |
| (A 000 record) | 5 | 0.129 ± 0.011 | 5/0/0 | 220 min |
| (D 000 record) | 5 | 0.119 ± 0.023 | 0/5/0 | 85 min |

What it means:
- **fp32 master weights are a large effective learning-rate increase.** In pure bf16 most AdamW
  updates round away, so the records' nominal lr overstates what was applied. With fp32 at the
  same lr, A 001's recipe reaches dev 0.10 in 11-22 min instead of 128-166, and seed 1 reached
  0.288 at 49 min while searching properly (it tried combinations and checked them). Then every
  seed failed: two diverged into repeated tokens ("设置设置…", "inside inside…"), three fell
  into one-guess mode. Track D at fp32 is worse than D 000 on every paired seed. Next: the same
  recipes at lower lr (r1).
- **Collapse comes in three modes.** One-guess: the policy writes one equation and answers
  (`(22 - 7) - (3 + 2) = 14`), steps/min triple because outputs are short, and fewer than 10 of 64
  groups stay informative, so learning stalls at dev ~0.16. Divergence: malformed output, score
  0. Healthy search: lists candidate combinations and checks them, 20-45 informative groups, and
  it is the only mode that scores above ~0.2. The records' 000 seeds are all healthy; every track
  D record seed is one-guess.
- **The ablation (A 001 without the constraint):** takeoff is faster (54 min against 150) but 3 of
  5 seeds collapse to one-guess; the 2 healthy seeds score 0.282 and 0.227, as high as A 001's
  best. The constraint's part in A 001 looks like stability rather than speed. 5 seeds and a
  bimodal outcome: Welch against A 001 is not significant (0.197 vs 0.240).
- The FineMath pretraining pilot (sweep_d15, 1 seed, H200) ended at dev 0.135: no change of regime.
- d00_fp32 seed 0 crashed (host `MemoryError` while compiling); it was requeued.

Also on 2026-10-19: the maintainer froze the rules with external data allowed (RULES_v3 at
13dbe32); research runs from branch `research/2026-10`.

Queued as r1 (`calibration/research_r1`, 26 runs, ~120 GPU-h):
- A 001 recipe + fp32 at lr 2.5e-7, 5e-7, 1e-6, 2e-6 (3 seeds each);
- A 001 without the constraint + fp32 at lr 5e-7, 1e-6 (2 seeds each);
- D 000 RL + fp32 at lr 1e-6, 3e-6, 1e-5 (3 seeds each, paired on D 000's pretrained seeds 0-2);
- d00_fp32 seed 0 again.

## 2026-10-19 (evening): d00_fp32 done, first r1 seeds, preemptions

`d00_fp32` (D 000's RL with fp32 master weights at D 000's lr 3e-5) has its 5th seed: seed 0
ended at 0.090 (D 000's own seed 0: 0.087).

| arm | n | dev final | ok / one-guess / diverged | median t.10 |
|---|---|---|---|---|
| D 000 RL + fp32 at lr 3e-5 (paired on D 000's pretrained seeds 0-4) | 5 | 0.046 ± 0.043 | 0/4/1 | never |
| (D 000 record) | 5 | 0.119 ± 0.023 | 0/5/0 | 85 min |

Paired by seed, fp32 is level on seed 0 and worse on seeds 1-4; this arm is closed.

First r1 seeds (seed 0 of each arm; partial, dev at 30/60/120/180/240 min):

| arm (seed 0) | reached | dev then | curve | mode |
|---|---|---|---|---|
| A 001 + fp32, lr 1e-6 | 254 of 290 min (preempted) | 0.188 (best 0.192) | 0.082 0.125 0.153 0.150 0.173 | one-guess from ~150 min |
| A 001 + fp32, lr 5e-7 | 159 min (preempted later) | 0.150 | 0.043 0.057 0.113 0.150 | one-guess from ~100 min |
| D 000 RL + fp32, lr 3e-6 | 144 min (preempted later) | 0.107 (best 0.123) | 0.067 0.077 0.123 | one-guess |
| (A 001 record, mean of 5) | 290 min | 0.240 | 0.053 0.063 0.089 0.133 0.195 | 4 ok / 1 one-guess |

What it means so far:
- Lowering the fp32 lr 4-8x keeps the early speed (A 001 + fp32 at 1e-6 leads the record's mean
  curve until ~180 min) but does not prevent one-guess collapse; it only delays it. In one-guess
  mode dev still creeps up (0.15 -> 0.19), but A 001's healthy seeds gain ~0.1 in their last two
  hours, which one-guess seeds don't. Seed 0 of lr 2.5e-7 and 2e-6 is still to come; seeds 1-2
  of each arm are decided once all seed 0s are in.
- Five preemptions on mit_preemptable between 11:45 and 15:20 EDT restarted r1 screens from
  scratch (~14 GPU-h lost; a01_fp32_lr1e-6 seed 0 at 254 of 290 min). The weekend's preemptable
  screens had none. Since a1e4502, screens on mit_preemptable save resumable state every hour
  (off the clock) and a requeued seed continues from it. The a01_fp32_lr1e-6 seed 0 rerun
  (started 15:36 EDT) is the first to use it. The partial runs stay in the pool as
  `<seed>.interrupted.<stamp>/`.
