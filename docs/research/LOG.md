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

## 2026-10-19 (night): r1 seed 0s done; r1 pruned; r2 queued with a faster update

All six r1 seed 0s finished (dev at 30/60/120/180/240 min, final at 290 / 234 min):

| arm (seed 0) | dev final | best | curve | mode |
|---|---|---|---|---|
| A 001 + fp32, lr 2e-6 | 0.062 | 0.287 (92 min) | 0.077 0.098 0.287 0.118 0.173 | one-guess |
| A 001 + fp32, lr 1e-6 | **0.290** | 0.327 (120 min) | 0.095 0.187 0.327 0.283 0.242 | ok |
| (same seed, first attempt, preempted at 254 min) | 0.188 | 0.192 | 0.082 0.125 0.153 0.150 0.173 | one-guess |
| A 001 + fp32, lr 5e-7 | 0.222 | 0.222 | 0.038 0.052 0.163 0.172 0.202 | one-guess |
| (A 001 record: seed 0 / mean of 5) | 0.203 / 0.240 | | mean 0.053 0.063 0.089 0.133 0.195 | 4 ok / 1 one-guess |
| D 000 RL + fp32, lr 1e-6 | 0.098 | 0.103 | 0.062 0.085 0.088 0.095 | one-guess |
| D 000 RL + fp32, lr 3e-6 | **0.147** | 0.163 | 0.088 0.127 0.138 0.148 | one-guess |
| D 000 RL + fp32, lr 1e-5 | 0.080 | 0.095 | 0.087 0.060 0.082 0.087 | one-guess |
| (D 000 record: seed 0 / mean of 5) | 0.087 / 0.119 | | | all one-guess |

What it means:
- Track A: lr 1e-6 is the only fp32 lr whose seed 0 ended healthy, at 0.290 against A 001's
  seed-0 0.203, but the same seed's first attempt drifted into one-guess mode, so the outcome
  looks bimodal, as for A 001 itself. lr 2e-6 peaks earliest and highest, then collapses
  (0.287 -> 0.062); 5e-7 is slow and ends one-guess.
- Track D: lr 3e-6 is the best fp32 lr. On the same pretrained model (D 000's seed 0) its RL
  ends at 0.147 against D 000's 0.087 (+0.060); 1e-6 gains +0.011, 1e-5 loses 0.007, and 3e-5
  was worse on 4 of 5 seeds (w42).

Decisions:
- r1 keeps only a01_fp32_lr1e-6 (seeds 1-2) and d00_fp32_lr3e-6 (seeds 1-4, paired on D 000's
  pretrained seeds 1-4: a 5-seed screen for the promotion rule). The 15 dropped items are listed
  in `queue/r1_dropped_2026-10-19.txt` (A lr 2.5e-7, 5e-7, 2e-6; the no-constraint arms; D lr
  1e-6 and 1e-5).
- **Faster update (9dcb8be, `skip_zero_advantage`, off by default).** Groups whose 8 samples got
  the same reward have zero advantage and, with no KL term, zero gradient, but the update still
  ran their forward and backward passes. In A 001 seed 1 only 28-35 of 64 groups were
  informative, and the update took 11-16 s of each 16-22 s step; in D 000, 5-9 of 64. Skipping
  those rows gives the same optimizer step (tested) with the same loss normaliser, so only the
  time changes: an estimated ~1.6x more steps in the same budget on both tracks.
- r2 (`queue/main.txt`), 3 seeds each, in this order: A 001 + skip (`a01_skip`), fp32 lr 1e-6 +
  skip + cosine decay to 0.1 (`a01_fp32_lr1e-6_skip_cos`), D fp32 lr 3e-6 + skip
  (`d00_fp32_lr3e-6_skip`), then fp32 lr 1e-6 with only cosine decay (`_cos`) and with malformed
  reward 0 (`_mal0`). The D2 data screens moved to `queue/long.txt` (6 h items, preemptable
  workers only).
- Workers replaced (the pending ones served r1 only): from the end of the maintenance
  (2026-10-20 21:00 EDT), 4 preemptable workers (2 take `long` first, 2 take r1 then main;
  RESERVE_MIN 380 so a 6 h item never starts too late) and 6 mit_normal_gpu workers (r1 then
  main, one item each, 2 at a time).

## 2026-10-19 (late): sampler temperature on the holders' weights

The record seeds' final weights scored on the whole dev ladder (2400 puzzles, 16 samples, the
record sampler: constrained, think budget 0) at three temperatures (`calibration/research_r2/samp_*`):

| | T | seed 0 | 1 | 2 | 3 | 4 | mean | pass@16 |
|---|---|---|---|---|---|---|---|---|
| A 001 | 1.0 | 0.2202 | 0.2495 | 0.1886 | 0.2705 | 0.3031 | 0.2464 | 0.471 |
| A 001 | 0.75 | 0.2224 | 0.2593 | 0.1889 | 0.2805 | 0.3070 | 0.2516 | 0.458 |
| A 001 | 0.5 | 0.2235 | 0.2676 | 0.1892 | 0.2855 | 0.3087 | 0.2549 | 0.432 |
| D 000 | 1.0 | 0.0896 | 0.1356 | 0.1060 | 0.1205 | 0.1111 | 0.1126 | 0.160 |
| D 000 | 0.75 | 0.0915 | 0.1385 | 0.1072 | 0.1220 | 0.1120 | 0.1142 | 0.151 |
| D 000 | 0.5 | 0.0925 | 0.1396 | 0.1073 | 0.1230 | 0.1125 | 0.1150 | 0.141 |

What it means:
- T 0.5 beats T 1.0 on all 10 seeds, by +0.0085 on average for A 001 (+0.0007 to +0.018) and
  +0.0024 for D 000. The gain is largest on A 001's healthy-search seeds 1 and 3, whose outputs
  grow longer at low temperature (truncation at 1024 tokens rises to 4-8 %) and still score higher.
  A free, small gain: later record attempts will use T 0.5 unless their own dev check says otherwise.
- Dev at T 1.0 matches the record seeds' test scores closely (A 001: 0.220 / 0.250 / 0.189 / 0.271 /
  0.303 against test 0.224 / 0.252 / 0.186 / 0.272 / 0.306), so the dev ladder is a faithful proxy.

## 2026-10-20: when track A runs collapse (logs of the 19 runs so far); r3 queued

Engaging's maintenance ended early (~13:45 EDT); the four preemptable workers started, were all
preempted within 25 min and wait in the queue again, so no new results. Meanwhile, from the train
logs of every track A run (A 001's seeds, w42, r1 seed 0s): mean completion length (tokens per
sample, from `tokens/s` and the step time), entropy and informative groups by training step.

| run | len @ step 100 | 200 | 300 | entropy @ 300 | informative @ 300 | dev final | mode |
|---|---|---|---|---|---|---|---|
| A 001 s0 | 94 | 104 | 132 | 0.63 | 16 | 0.203 | ok |
| A 001 s1 | 100 | 127 | 146 | 0.68 | 18 | 0.237 | ok |
| A 001 s2 | 87 | 54 | 38 | 0.19 | 15 | 0.193 | one-guess |
| A 001 s3 | 131 | 135 | 135 | 0.68 | 18 | 0.268 | ok |
| A 001 s4 | 123 | 129 | 160 | 0.68 | 17 | 0.300 | ok |
| w42 a01 nc s0 | 45 | 44 | 45 | 0.03 | 6 | 0.157 | one-guess |
| w42 a01 nc s1 | 90 | 47 | 43 | 0.06 | 11 | 0.158 | one-guess |
| w42 a01 nc s2 | 135 | 143 | 276 | 0.26 | 21 | 0.282 | ok |
| w42 a01 nc s3 | 113 | 45 | 46 | 0.07 | 12 | 0.163 | one-guess |
| w42 a01 nc s4 | 111 | 123 | 174 | 0.33 | 19 | 0.227 | ok |
| w42 a01 fp32 4e-6 s0 | 21 | 32 | 34 | 0.07 | 6 | 0.172 | one-guess |
| w42 a01 fp32 4e-6 s1 | 619 | 700 | 20 | 0.25 | 9 | 0.000 | diverged |
| w42 a01 fp32 4e-6 s2 | 21 | 20 | 69 | 0.43 | 9 | 0.170 | one-guess |
| w42 a01 fp32 4e-6 s3 | 370 | 126 | 40 | 0.14 | 13 | 0.000 | diverged |
| w42 a01 fp32 4e-6 s4 | 44 | 26 | 22 | 0.10 | 5 | 0.190 | one-guess |
| r1 fp32 2e-6 s0 | 189 | 501 | 546 | 0.03 | 17 | 0.062 | one-guess (late) |
| r1 fp32 1e-6 s0 | 49 | 114 | 429 | 0.07 | 20 | 0.290 | ok |
| r1 fp32 1e-6 s0 (1st attempt) | 40 | 38 | 38 | 0.05 | 6 | 0.188 | one-guess |
| r1 fp32 5e-7 s0 | 85 | 70 | 29 | 0.25 | 14 | 0.222 | one-guess |

What it means:
- **One-guess collapse is decided early, by step ~200-300 (30-60 min).** Every run starts long
  (300-550 tokens, 30-55 % malformed while it learns the format) and shortens. Healthy runs stop
  at ~100-140 tokens and then grow (300-600 tokens by 2-4 h: they list candidates and check them);
  one-guess runs keep shortening to 20-60 tokens and never come back. At step 300 the two groups
  don't overlap in length (>= 130 vs <= 69), and no run that was short at step 300 recovered.
- fp32 master weights cut entropy much faster (0.15-0.2 at 30 min against 0.65-0.75 for A 001 in
  bf16), so the length dip is deeper (~50 tokens at step 100 for both lr 1e-6 attempts); whether
  the run then regrows looks like chance (same seed, opposite outcomes). Lowering the lr to 5e-7
  only delays the dip. Late failures are a separate, high-lr problem (lr 2e-6, 4e-6).
- A cheap test follows: a collapse-prevention change can be judged from 90-min runs (the length,
  entropy and informative groups at the end) instead of 5-h runs.

Queue changes (GPU time is the bottleneck, INBOX 2026-10-20):
- Dropped `a01_fp32_lr1e-6` seeds 1-2 (r1) and r2's `_cos` / `_mal0` arms (no skipping): with
  `skip_zero_advantage` the optimizer step is the same, so r2's `a01_fp32_lr1e-6_skip_cos` seeds
  give the same early-collapse evidence and also test the full candidate. Listed in
  `queue/{r1,main}_dropped_2026-10-20.txt`.
- r3 collapse screens (`calibration/research_r3/c90_*`, 4 seeds each, 90 training minutes, fp32
  lr 1e-6, constant lr): malformed reward 0 (`c90_mal0`, skipping), training temperature 1.2
  (`c90_t12`, skipping), Clip-Cov on 0.1 % of tokens (`c90_cc1e-3`, no skipping). KL to the base
  model is left out: fp32 runs already use ~79 of 80 GB, and the reference model needs ~3.4 GB more.
  The baseline is r2's skip_cos seeds (identical per step for the first hour) plus r1's two attempts.
- r3 sampler evals on the holders' weights: A 001 at T 0.5 with a think budget of 960 (A 001's
  long-search seed 1 truncates 7.9 % of its samples at T 0.5, and a truncated sample scores 0),
  A 001 at T 0.25, D 000 at T 0.25.
- The short items sit in `queue/long.txt`, which only the preemptable workers serve: a 90-min item
  would end a mit_normal_gpu worker early (RESERVE_MIN 330 of its 6 h). Order: collapse seeds 0-1,
  D2 seeds 1, collapse seeds 2-3, D2 seeds 2, evals. `main` holds r2's 9 skip items; `r1` holds
  d00_fp32_lr3e-6 seeds 3-4 (seeds 1-2 in flight).
- Also noted: the in-loop dev eval (the promotion rule's "dev") is greedy decoding on 600 dev
  puzzles (100 per level), not 16 samples; the sampler evals use the whole dev ladder.

## 2026-10-20 (evening): still waiting for GPUs; two more workers, a fourth collapse fix, D sampler evals first

No results yet: Engaging is crowded after its maintenance (2342 jobs pending on mit_preemptable;
Slurm's estimate for our workers ~22:45 EDT).

- Workers: the mit_normal_gpu workers' first start slipped to 2026-10-23 07:30 (that partition
  has one 4-GPU H100 node), so two preemptable workers were added (26450849-50, `r1:main:long`) and
  the six mit_normal_gpu workers now wait for both to end (`afterany`). At most 6 GPUs at a time.
- `r1` now starts with track D sampler evals (~1 min each): the fp32 lr 3e-6 screen's seed 0
  weights at T 1.0 / 0.5 / 0.25 and D 000's seeds at T 0.25. A D record from that arm picks its
  sampler on its own weights, and the decision comes with the promotion decision instead of
  waiting behind the long queue.
- `main`: seeds 3-4 of a01_skip, d00_fp32_lr3e-6_skip and a01_fp32_lr1e-6_skip_cos queued behind
  seeds 0-2, so an arm that needs 5 seeds for the promotion rule doesn't wait; dropped before they
  start if seeds 0-2 rule an arm out.
- A fourth collapse fix (da706a8, `pos_adv_scale`, off by default): positive advantages times
  0.1, as in W-REINFORCE (Zhu et al. 2025, "The Surprising Effectiveness of Negative
  Reinforcement in LLM Reasoning"). Reinforcing sampled solutions narrows the policy, which is
  how one-guess collapse starts; pushing away from failures spreads probability over the
  alternatives. `c90_pos01` (4 seeds, 90 min) sits beside the other three in `long`.

## 2026-10-20 (night): A 001's dev curves were still climbing; a01_skip seeds 0-2 moved first

Still no GPU time: the four workers that started at 13:46 were preempted after 15-17 min, and
Slurm's estimate for our next start is ~01:55 (2,652 jobs pending on mit_preemptable).

Meanwhile, A 001's in-loop dev curves (greedy, 600 dev puzzles) from its record logs, with the
training minutes in brackets:

| seed | step 600 | step 800 | step 900 | end (290 min) | steps |
|---|---|---|---|---|---|
| 0 | 0.087 (137) | 0.197 (206) | 0.218 (247) | 0.203 (0.240 at step 1050) | 1068 |
| 1 | 0.088 (138) | 0.140 (199) | 0.192 (236) | 0.237 | 1036 |
| 2, one-guess | 0.133 (89) | 0.143 (110) | 0.145 (120) | 0.193 | 2514 |
| 3 | 0.100 (142) | 0.163 (211) | 0.225 (252) | 0.268 | 989 |
| 4 | 0.115 (150) | 0.210 (223) | 0.262 (262) | 0.300 | 970 |

In plain words: every healthy seed sits at 0.06-0.09 for its first ~600 steps (~140 min). It then
gains 0.12-0.19 over its last ~400 steps, until the budget stops it. Seeds 1, 3 and 4 are still
rising at the end; seed 0 is noisy around 0.19-0.24 over its last 170 steps. The holder's dev
number (0.240) is the mean of these final values, so it is set by how many steps fit in 290 min.
`skip_zero_advantage` leaves each step's update unchanged and should fit 1.6-2.1x more steps
(estimated from A 001's step times). That makes a01_skip the likeliest A record, unless collapse
gets more frequent with it (seed 2 went one-guess by step ~350 and stayed at 0.14-0.19). In-loop
evals cost 7-9 s each and are outside the budget, so more steps do not stretch a screen much.

- `main` reordered: a01_skip seeds 0-2 and a01_fp32_lr1e-6_skip_cos seed 0 first, then the fp32
  arm's seeds 1-2 and d00_fp32_lr3e-6_skip seeds 0-2, then seeds 3-4. a01_skip's 3-seed promotion
  check then comes with the first wave (~12:00 on 10-21, if the workers start as estimated), not
  the second. d00_fp32_lr3e-6_skip waits one wave. That costs little, because it only matters
  once a D record exists.
- Asked the maintainer (QUESTIONS.md) whether AWS may also run a01_skip's dev screens while
  Engaging is this congested; the approval so far covers record attempts.
- For comparison, r1's healthy fp32 lr 1e-6 seed 0 takes off near step 200 instead of 600: dev
  0.187 at step 200 (43 min), 0.290 at step 300 (84 min), peak 0.327 at step 350 (109 min). It
  then fell into a one-guess spell: from step ~500, steps took 10-14 s instead of 26-36 s, entropy
  stayed near 0.035, the batch reward fell from 0.41 to 0.14 and dev to 0.125 at step 550. Entropy
  then rose again (0.15-0.22) and dev recovered to 0.290 at the end. RULES_v3 scores each seed
  on its final weights and forbids choosing a checkpoint. So the peak only counts if the run is
  still there at the end, which is what r2's cosine decay (`a01_fp32_lr1e-6_skip_cos`) and the
  collapse screens aim at. Track D is different: fp32 lr 3e-6 seed 0 has been flat at 0.14-0.16
  since step ~3,500 (75 of 234 min), so extra steps from skip should matter less there.

## 2026-10-20 (late): a01_skip seeds 0-2 on AWS

The maintainer approved AWS for a01_skip's dev screens (INBOX, via the laptop session): seeds
0-2 run on spot p5.4xlarge (1 H100 SXM, us-east-1, from ~23:20; launcher change 78975d7,
STAGE=train, so the test ladder is never read), and the laptop session removed them from
`queue/main.txt`. Seeds 3-4 stay on Engaging behind the fp32 arm's seeds 0-2. Engaging's
preemptable estimate meanwhile slipped to ~05:20 on 10-21 (2,925 jobs pending), so the AWS
seeds should finish (~05:00-06:00) before Engaging reaches `main`. Then: 3 healthy seeds all
>= 0.270 promote a01_skip (record config = A 001 + skip_zero_advantage, sampler T 0.5) and seeds
3-4 leave the queue. Anything less moves seeds 3-4 to the front for the 5-seed form.

## 2026-10-22 (night): Ubuntu nodes failed 35 items; D fp32 lr 3e-6 misses promotion; skip + cosine seed 0 at 0.300

The agent loop stalled on a permission prompt after its 10-21 00:45 update and was restarted at
23:14 on 10-22 (eb3becd). Meanwhile the workers ran, and 35 items failed for a host reason until all
three queues were empty. The last GPU item ended at 10:27 on 10-22.

**What failed and why.** Worker 26424658 failed 32 items on node1800 (10-21 11:11-18:48), and three
more failed on node2700, node2801 and node2701 (10-22). All four nodes run Ubuntu 24.04; the
other H100 nodes run Rocky 8 (8 of the ~35 H100 nodes on mit_preemptable are Ubuntu). The cluster's
gcc 12.2.0 (Spack, built for Rocky) does not search Ubuntu's multiarch directories, so
`#include <stdlib.h>` fails (`bits/libc-header-start.h: No such file or directory`). Triton then
cannot build its C launcher when vLLM compiles the model, and the run dies at startup. Items whose
compiled kernels were already in vLLM's compile cache (keyed by model and config) ran fine on those
nodes, so the failures looked random. New models or configs failed, and the D2 items died at the
RL start after their 112-min pretraining. A CPU probe job on node1800 reproduced the failure and
confirmed the fix: with `C_INCLUDE_PATH`/`CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu` and
`LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`, gcc, g++ and triton's launcher build all succeed. (`module
load` also fails on Ubuntu nodes, because the lmod path differs; gcc comes from the inherited PATH.)

Fixes (`queue_worker.sbatch`, tested with stubbed Slurm commands):
- Before popping anything, the worker checks that gcc can build a small C extension. On Ubuntu
  nodes it adds the multiarch paths; if gcc still fails, it exits with nothing popped.
- After two failures in a row that each took under 5 minutes, it puts the second item back at the
  front of the queue it came from and exits (the laptop's request). One bad node can no longer
  drain the queues.
- New workers exclude node1800 (INBOX).

**Results that landed before the queues emptied.**

D 000's RL with fp32 master weights at lr 3e-6 (r1, paired on D 000's pretrained seeds; seed 1 reached
193 of 234 min before preemption and resumes from its saved state at 180 min, where dev was 0.132):

| seed | D 000 | fp32 lr 3e-6 | dev at 30/60/120/180 min (fp32) |
|---|---|---|---|
| 0 | 0.087 | 0.147 | 0.088 0.127 0.138 0.148 |
| 1 | 0.152 | 0.145 (193 min) | 0.077 0.088 0.108 0.132 |
| 2 | 0.115 | 0.132 | 0.072 0.083 0.098 0.112 |
| 3 | 0.127 | 0.125 | 0.087 0.083 0.102 0.100 |
| 4 | 0.115 | 0.140 | 0.072 0.088 0.103 0.132 |
| mean | 0.119 ± 0.023 | 0.138 ± 0.009 | |

- In plain words: fp32 RL lifts the weak seeds and narrows the spread (0.125-0.147 against
  0.087-0.152), so the mean gains +0.019. Promotion needs +0.03 (mean >= 0.149), so this arm does
  not promote, whatever seed 1 ends at. All five seeds sit above the holder's 0.119.
- Seeds 1-4 were still rising at the end (seed 2: 0.112 at 180 min -> 0.132 at 234). Seed 0's
  early plateau was the exception, so the split screen that moves RL time to pretraining is not
  queued. More RL steps in the same time (`skip_zero_advantage`) is the more promising change.

A 001 + fp32 lr 1e-6 + skip + cosine decay (`a01_fp32_lr1e-6_skip_cos`), seed 0: **0.300**
(A 001's seed 0: 0.203). Dev at 30/60/120/180/240 min: 0.115 0.170 0.242 0.288 0.287, and
0.28-0.30 over the last 500 steps as the lr decayed. It ran 4,127 steps in 290 min; A 001's healthy
seeds ran ~1,000. Mean completion length by step:

| step | 100 | 300 | 1000 | 2000 | 2500 | 3000 | 3500 | 4120 |
|---|---|---|---|---|---|---|---|---|
| minutes | 18 | 26 | 52 | 86 | 101 | 141 | 197 | 289 |
| tokens | 19 | 25 | 26 | 28 | 41 | 86 | 179 | 237 |

- This is the first run that recovered from the early one-guess dip, which the 10-20 analysis
  had found permanent. For ~100 min it answered with one short guess, then it grew back into a
  long search (237 tokens per sample at the end), with dev rising throughout. The 4x more steps
  (skip, plus fast steps while outputs were short) probably made the difference. `summarize_screens.py` still labels it
  one-guess, because fewer than 10 of 64 groups are informative when the policy solves most easy
  prompts 8 of 8 times. The label is a training-signal heuristic and misreads strong policies.
- It is one seed. Seeds 1-2 decide the 3-seed promotion form (both >= 0.270).

D2 (D 000's pretraining with DeepMind mathematics arithmetic): dm26 seed 0 with D 000's RL ended
at 0.118 (D 000's seed 0: 0.087; best 0.130). The five failed D2 seeds (dm26 seeds 1-2, dm51
seeds 0-2) all finished pretraining (2.46-2.50 B tokens at ~370 k tokens/s, like D 000's record)
before the RL start died. Their pretrained models are saved.

Sampler on the holders' weights (whole dev ladder, 16 samples):

| | T 0.5 (10-19) | T 0.5 + think budget 960 | T 0.25 |
|---|---|---|---|
| A 001 (5 seeds; T 0.25: seeds 1, 3, 4) | 0.2549 | 0.2551 | 0.2881 (vs 0.2873 at T 0.5 on those seeds) |
| D 000 (5 seeds) | 0.1150 | | 0.1154 |
| D fp32 lr 3e-6 seed 0 | 0.1502 | | 0.1500 (T 1.0: 0.1493) |

The think budget removes A 001's truncation (7.9 % -> 0 on seed 1) but gains only +0.0002, and
T 0.25 changes -0.0002 to +0.0008. T 0.5 stays the record sampler.

**Decisions.**
- Promotion: none. The D candidate is now fp32 + skip, on D 000's pretraining and on the D2
  pretraining. The A candidates are `a01_fp32_lr1e-6_skip_cos` and `a01_skip` (AWS seeds 0-2 not in
  yet).
- Promotion rule, clarified before the seeds it applies to (PLAN.md): a "collapsed" seed is a
  diverged one, or a one-guess one whose final dev is below the holder's dev mean. Read
  literally, the label would block every track D arm, since the holder's own seeds are all
  one-guess, and it would block seed 0 above at 0.300. QUESTIONS.md asks the maintainer.
- Requeued: 29 failed screen items, their failed attempts moved to `extra/failed_<seed>_1/` with
  a note. d00_fp32_lr3e-6 seed 1 resumes from its 180-min state (step 8,567).
- D2: rather than rerun 6-h pipelines, r4 runs the RL stage from the saved D2 pretrained models,
  with the fp32 + skip RL (`calibration/research_r4/{dm26,dm51}_fp32_lr3e-6_skip`, seeds 0-2,
  `screen_pretrained`). The baseline is `d00_fp32_lr3e-6_skip` on D 000's pretraining, and
  r1's no-skip arm isolates skip. D 000's own RL on the D2 models (the old D2 design for the
  remaining seeds) is dropped, because the D candidate RL is now fp32.
- Queues: `main` = skip_cos seeds 1-2, dm26 0-2, d00 skip 0-2, dm51 0-2, a01_skip 3-4, then
  seeds 3-4 of skip_cos and d00 skip. `long` = d00_fp32 seed 1 (resume), the two A 001 T 0.25
  evals, then the c90 collapse screens (seeds 0-1, then 2-3). About 105 GPU-hours.
- Workers: 6 on mit_preemptable (26644470 `long:main`, 26644471-75 `main:long`). Slurm's estimate
  is ~00:20 on 10-24 (2,374 jobs pending). mit_normal_gpu's estimate is 10-25 09:00 for H100 and
  H200 alike, so no workers wait there.

## 2026-10-23 (early): workers run on Ubuntu nodes; D fp32 lr 3e-6 ends at 0.137; T 0.25 adds nothing

The first four workers started at 23:59 and 00:26, all on Ubuntu 24 nodes (node2700, node2701).
Their gcc check added the multiarch paths, and every item so far got through vLLM's startup
compile: two A screens, the r4 dm26 screen, the resumed D seed and two sampler evals. A probe on
node2801 showed that lmod isn't installed on Ubuntu nodes at all (no `/usr/share/lmod`, `MODULEPATH`
unset), so the worker's `module load cuda/13.0.1` fails there; training and evals don't need it.
This corrects the 10-22 entry, which put it down to a different lmod path.

D 000's RL with fp32 master weights at lr 3e-6, all five seeds (seed 1 resumed from its 180-min
state):

| seed | D 000 | fp32 lr 3e-6 |
|---|---|---|
| 0 | 0.087 | 0.147 |
| 1 | 0.152 | 0.143 |
| 2 | 0.115 | 0.132 |
| 3 | 0.127 | 0.125 |
| 4 | 0.115 | 0.140 |
| mean | 0.119 ± 0.023 | 0.137 ± 0.009 |

All five seeds end above the holder's 0.119, but the mean gains only +0.018, short of the 0.149 that
promotion needs. Seed 1 is the one seed where fp32 ends below D 000 (0.143 against 0.152). The arm
stays the no-skip baseline for the fp32 + skip D screens (`d00_fp32_lr3e-6_skip`, r4).

A 001's weights at T 0.25 against T 0.5 (whole dev ladder, 16 samples):

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| T 0.5 | 0.2235 | 0.2676 | 0.1892 | 0.2855 | 0.3087 | 0.2549 |
| T 0.25 | 0.2237 | 0.2663 | 0.1901 | 0.2886 | 0.3094 | 0.2556 |

+0.0007 on the mean, while pass@16 drops (seed 1: 0.530 -> 0.478). T 0.5 stays the record sampler.

Workers: four run and two pend (26644474-75). Successors 26649456-59 each wait on one running worker
(`afterany`), so the 6-GPU cap holds.

## 2026-10-23 (02:45): malformed reward 0 reaches 0.322 in 90 min; r5 queued first

The first collapse screen finished. `c90_mal0` is A 001 + fp32 lr 1e-6 + skip at constant lr, with
training reward 0 instead of -0.1 for malformed or truncated outputs, stopped at 90 min. On seed 0:

| | dev at step 100 | step 200 | final (step 254, 90 min) |
|---|---|---|---|
| pass | 0.075 | 0.288 | **0.322** |
| eval malformed | 0.398 | 0.683 | 0.622 |

By level at the end: L1 0.97, L2 0.30, L3 0.16, L4 0.16, L5 0.03, L6 0.31. For comparison, the
skip + cosine seeds (the same recipe for the first hour, with the -0.1 penalty) were at
0.170-0.192 after 60 min, and seed 0 ended at 0.300 after 290 min.

- In plain words: without the penalty, the policy never retreats to one short guess. It writes
  long searches, and 62 % of its greedy eval answers run past 1,024 tokens and score 0, yet it is
  already ahead of every penalized run at the same point. The steps are slow (254 in 90 min,
  t_sample ~15 s), because outputs are long.
- `summarize_screens.py` labels the seed "diverged" (malformed > 0.5). Here malformed means
  truncated searches, not garbage. If a mal0 arm reaches the promotion numbers with that label,
  QUESTIONS.md gets the question before anything is decided.
- The 62 % truncation is an opening for the sampler: a think budget forces `<answer>` before token
  1,024 (allowed by RULES_v3), so a truncated search would still answer. On A 001 it changed
  nothing, because A 001 rarely truncates.

Decisions:
- r5 `a01_fp32_lr1e-6_skip_cos_mal0`: the skip + cosine candidate with malformed reward 0, 290 min,
  seeds 0-2 first in `main`. Final weights are kept (~7 GB per seed) so the sampler, think budget
  included, can be chosen on the dev ladder.
- `c90_mal0` seed 1 moved to the front of `long` (running since 02:35).

## 2026-10-23 (04:45): malformed reward 0 replicates (0.383); dm26 + fp32 + skip seed 0 at 0.152

`c90_mal0` seed 1 (90 min) ended at **0.383**, after seed 0's 0.322:

| seed | dev at step 100 / 200 / final | eval malformed (final) | L1 L2 L3 L4 L5 L6 (final) |
|---|---|---|---|
| 0 | 0.075 / 0.288 / 0.322 (step 254) | 0.62 | 0.97 0.30 0.16 0.16 0.03 0.31 |
| 1 | 0.122 / 0.322 / 0.383 (step 243) | 0.50 | 0.98 0.34 0.25 0.19 0.07 0.47 |

The skip + cosine seeds, the same recipe with the -0.1 penalty, were at 0.17-0.25 at 90 min. Both
mal0 seeds write long searches from the start, and half or more of their greedy eval answers still
hit the 1,024-token limit. The r5 arm (290 min, weights kept) started seed 0 at 04:22.

D RL screens on seed 0 (dev, final):

| RL \ pretraining | D 000's | DM-math 26 % (dm26) |
|---|---|---|
| D 000's RL (bf16, lr 3e-5) | 0.087 | 0.118 |
| fp32 lr 3e-6 | 0.147 | |
| fp32 lr 3e-6 + skip | (queued) | **0.152** |

- `dm26_fp32_lr3e-6_skip` seed 0 is the first D screen seed at or above the promotion line
  (0.149). One seed: the 3-seed form needs seeds 1-2 there too. `d00_fp32_lr3e-6_skip` seed 0
  (queued) separates what the DM-math data adds from what skip adds.

Queues: dm26 seed 1 moved to the front of `long` (worker 26644470 takes it after c90_t12 seed 0,
~05:40), dm26 seed 2 follows the r5 seeds in `main`. The c90 screens keep seed 0 of t12, Clip-Cov and
pos01; their seeds 1-3 and mal0's seeds 2-3 are parked in `queue/later.txt`, which no worker reads.
mal0 is now tested at full length (r5), and the other three test fixes for a collapse that mal0
avoids.

## 2026-10-23 (05:40): skip + cosine ends at 0.300 / 0.270 / 0.472; the 3-seed form misses by one puzzle

`a01_fp32_lr1e-6_skip_cos` (A 001 + fp32 lr 1e-6 + skip + cosine decay to 1e-7), seeds 0-2, 290 min:

| seed | A 001 | final dev | dev at 30/60/120/180/240 min | steps | L1 L2 L3 L4 L5 L6 (final) | mode at the end |
|---|---|---|---|---|---|---|
| 0 | 0.203 | 0.300 | 0.115 0.170 0.242 0.288 0.287 | 4,127 | 0.93 0.21 0.22 0.10 0.04 0.30 | long searches regrown after ~100 min |
| 1 | 0.237 | 0.270 | 0.133 0.187 0.220 0.245 0.262 | 10,610 | 0.74 0.28 0.15 0.07 0.06 0.32 | one guess throughout |
| 2 | 0.193 | **0.472** | 0.112 0.192 0.353 0.428 0.455 | 1,984 | 1.00 0.44 0.25 0.39 0.30 0.45 | long searches throughout |
| mean | 0.211 | **0.347 ± 0.109** | | | | |

- In plain words: the arm splits by whether the policy searches. Seed 2 never fell into the
  one-guess mode and reached 0.472, the best dev seen in this project, with L4 0.39 and L5 0.30.
  Seed 0 fell in and climbed back out. Seed 1 stayed in it for 290 min (10,610 short steps), and
  its guessing alone reached 0.270. All three beat A 001 on the same seed (+0.10, +0.03, +0.28).
- Promotion, 3-seed form: every seed must be at least the holder's dev mean + 0.03. A 001's exact
  dev mean is 0.2403 (0.203, 0.237, 0.193, 0.268, 0.300), so the line is 0.2703, i.e. 163 of 600
  puzzles. Seed 1 has 162. PLAN's rounded 0.240 would pass it, but I read the rule with the exact
  mean, so the 3-seed form is not met. None of the seeds counts as collapsed: seed 1 is labelled
  one-guess, but its 0.270 is above the holder's mean.
- The 5-seed form (mean >= 0.2703, 4 of 5 above 0.2403, at most 1 collapsed) needs seeds 3-4. One of
  them above 0.2403 and a combined 0.31 for the two would do it.
- Decision, made at the same time as the r5 malformed-reward-0 screens (the same recipe without
  the -0.1 penalty, which avoided the one-guess mode on both 90-min seeds): skip_cos seeds 3-4 stay
  queued, and the call comes when r5's seeds 0-1 end (~09:20-10:05). If r5 meets the promotion rule,
  r5 becomes the A record candidate. If not, skip_cos seeds 3-4 move to the front for the 5-seed
  form. All four of our preemptable GPUs (the QOS limit) are busy until ~09:20 either way.

`c90_t12` seed 0 (training temperature 1.2) is at 0.202 after 85 min and one-guess, like the
penalized runs: the higher temperature does not prevent the mode.

## 2026-10-23 (07:50): r5 seed 0 collapses at 140-160 min; skip + cosine seeds 3-4 to the front

r5 (`a01_fp32_lr1e-6_skip_cos_mal0`: skip + cosine with reward 0 for truncated outputs) seed 0 was
the strongest run so far at 139 min, then lost it all within 20 minutes. Seed 0, in-loop dev and
training rows (reward = solve rate, since truncation now scores 0):

| min | dev (malformed) | training reward | training truncated | entropy | informative groups |
|---|---|---|---|---|---|
| 117 | 0.380 (0.59) | 0.50 | 0.44 | 0.086 | 24 |
| 139 | **0.392** (0.57) | 0.39 | 0.58 | 0.063 | 28 |
| 150 | | 0.19 | 0.70 | 0.043 | 21 |
| 158 | | 0.06 | 0.82 | 0.115 | 6 |
| 160 | 0.043 (0.75) | 0.04 | 0.73 | 0.155 | 7 |
| 171 | 0.013 (0.27) | 0.02 | 0.24 | 0.399 | 7 |
| 190 | 0.043 (0.01) | 0.03 | 0.01 | 0.228 | 8 |

- In plain words: the searches kept getting longer until most of them ran out of tokens.
  Truncation costs nothing in this arm, so nothing pushed back. With most rollouts scoring 0, few
  groups carried a signal. The policy then fell into answering at once, at a solve rate (2-5 %)
  below the base model's. Seeds 1 and 2 are healthy so far: 0.378 at 146 min and 0.392 at 127 min,
  with 30-50 % truncated in training and entropy 0.08-0.10. Seed 0 went wrong when its entropy fell
  to 0.04 and its truncation passed 0.7.
- Promotion: seed 0 will end far below the holder's dev mean whatever the label says, so it is
  collapsed under any reading, and r5 cannot meet the 3-seed form. The 5-seed form would need
  seeds 3-4 both healthy and above 0.2403. I decide on that when seeds 1-2 end (~10:05, ~10:30).
- Decision now: skip + cosine (no collapse in 3 seeds; 0.300 / 0.270 / 0.472) is the safer A
  candidate. Its seeds 3-4 go to the front of `main`, with dm26 skip seed 2 between them, so they
  start ~09:25 (as r5 seed 0 ends) and ~10:30. If seed 3 alone reaches 0.31 and beats 0.2403, the
  5-seed form is met whatever seed 4 scores.
- Next for the long-search recipe: a think budget in training (`think_budget`, plumbed through
  rollouts, update and eval), which forces the answer open after N tokens. Then a search that runs
  long ends in a guess that can still score, never in a truncation. The r5 sampler evals
  (T 0.5 with budgets 960 and 768) on seeds 1-2's weights pick N first; they go to the front of
  `long` once the weights exist.

## 2026-10-23 (08:55): r5 seed 1 collapses too; r5 is out; r6 trains with a think budget

r5 seed 1 went the same way at 175-185 min: in-loop dev 0.378 (146 min), 0.333 (166 min), then 0.025
(183 min); now 0.05 at 215 min with nearly every answer immediate. Its entropy fell to 0.033 at
164-170 min, then truncation jumped from 0.2 to 0.6 and the solve rate from 0.33 to 0.05 within
10 minutes. Seed 2 is still at 0.377 (185 min), but its entropy is down to 0.044, where the other two
broke.

- In plain words: with truncation free, the arm learns long searches fast and then breaks once the
  policy is sharp enough. Two of three seeds lost everything in under 20 minutes. Skip + cosine
  (same recipe with the -0.1 penalty) never broke in three seeds.
- Promotion: two collapsed seeds fail both forms (at most 1 of 5), so r5 is out whatever seeds 3-4
  would do. Seed 2 finishes (~10:30); its weights get the sampler evals only if it ends healthy.
- r6 (`calibration/research_r6/a01_fp32_lr1e-6_skip_cos_tb768`): skip + cosine with
  `think_budget: 768` in training and in the in-loop eval. A completion that hasn't opened the answer
  after 768 free tokens has to answer, so nothing truncates, the malformed penalty never applies,
  and a search that runs long ends in a guess that can still score. The aim is r5's long searches
  without its runaway. The budget, 768 of the 1,024 tokens, leaves room for long searches while
  capping every rollout. Weights are saved for sampler evals.
- Queue: r6 seed 0 at the front of `long` (worker 26644470, ~09:50, ahead of the two c90 collapse
  screens); seeds 1-2 in `main` after skip_cos seeds 3-4 and dm26 seed 2 (~14:00).

## 2026-10-23 (09:55): all three r5 seeds collapsed; dm26 skip seed 1 ends at 0.143 (D 3-seed form missed)

r5 seed 2 broke too (0.407 at 166 min, 0.06-0.08 from ~240 min), so all three seeds collapsed:

| seed | best in-loop dev (min) | final dev | collapse window |
|---|---|---|---|
| 0 | 0.392 (139) | 0.113 | 140-160 min |
| 1 | 0.378 (146) | ~0.06 (ends ~10:05) | 175-185 min |
| 2 | 0.407 (166) | ~0.07 (ends ~10:30) | ~190-240 min |

- In plain words: without the truncation penalty, the long-search policy peaks near 0.40 and then
  breaks on every seed. Its final weights are no use for sampler evals, so those are dropped.

r4 `dm26_fp32_lr3e-6_skip` (fp32 lr 3e-6 + skip RL on the 26 % DM-math pretraining), seed 1 done:

| seed | D 000 | final dev | puzzles | steps | L1 L2 L3 L4 L5 L6 |
|---|---|---|---|---|---|
| 0 | 0.087 | 0.152 | 91 | 16,317 | 0.59 0.09 0.03 0.04 0.00 0.16 |
| 1 | 0.152 | 0.143 | 86 | 17,550 | 0.48 0.10 0.04 0.04 0.00 0.20 |
| mean | 0.119 | 0.148 | | | |

- Promotion, 3-seed form: needs every seed >= 0.149 (90 puzzles); seed 1 has 86, so it is out. The
  5-seed form needs a mean >= 0.149, so seeds 2-4 must average >= 0.150. Seed 2 runs now (ends
  ~14:05). Seeds 3-4 have no saved D2 pretraining, so each would be the full 6-h pipeline; I decide
  on them when seed 2 is in.
- Queue: the 51 % DM-math seeds (`dm51_fp32_lr3e-6_skip` 0-2) move ahead of `d00_fp32_lr3e-6_skip`
  0-2. Those three only isolate skip's effect on D 000's pretraining, while dm51 is a D record path.

Correction to the 08:55 entry: r6's think budget removes truncation, but not completions that end
(EOS) before opening the answer. Those still score -0.1 as in skip + cosine. r6 seed 0's first
steps: malformed 0.29-0.37, against 0.58-0.72 for skip + cosine's first steps (truncations gone);
the update takes ~23 s per step, as skip + cosine's first steps did, because the penalty makes most
groups informative. Both fall once the policy learns to answer.

## 2026-10-23 (10:50): r7 stops r5 at step 300, before its collapse; r6 learns slowly

`scripts/summarize_screens.py` (r5 done; the others running):

```
  a01_fp32_lr1e-6_skip_cos_mal0      s0 done  290m final 0.113 best 0.392 t.10   50 spm  31.1 ent 0.06 inf  6.1 one-guess | 0.098 0.215 0.380 0.023 0.068
  a01_fp32_lr1e-6_skip_cos_mal0      s1 done  290m final 0.063 best 0.378 t.10   48 spm  11.5 ent 0.26 inf 12.7 ok        | 0.087 0.168 0.322 0.333 0.047
  a01_fp32_lr1e-6_skip_cos_mal0      s2 done  290m final 0.083 best 0.407 t.10   34 spm   8.8 ent 0.11 inf 10.2 ok        | 0.067 0.252 0.373 0.407 0.062
= a01_fp32_lr1e-6_skip_cos_mal0        n=3 final 0.087 ± 0.025  ok/1-guess/div 2/1/0  median t.10 48m
  a01_fp32_lr1e-6_skip_cos           s3 RUN    85m final 0.208 best 0.228 t.10   21 spm  40.3 ent 0.04 inf  5.9 one-guess
  a01_fp32_lr1e-6_skip_cos           s4 RUN    27m final 0.118 best 0.118 t.10   25 spm   9.9 ent 0.03 inf  6.2 one-guess
  a01_fp32_lr1e-6_skip_cos_tb768     s0 RUN    68m final 0.087 best 0.087 t.10    - spm   2.5 ent 0.90 inf 19.3 ok
  dm26_fp32_lr3e-6_skip              s2 RUN    48m final 0.093 best 0.097 t.10    - spm  68.3 ent 0.05 inf  3.1 one-guess
```

r5's in-loop dev by step (greedy dev malformed in brackets, i.e. searches cut at 1,024 tokens):

| seed | step 300 | step 350 | step 400 | step 450 | step 500 | first collapsed eval |
|---|---|---|---|---|---|---|
| 0 | 0.380 (0.59), 119 min | 0.392 (0.57) | 0.043 | | | step 400 |
| 1 | 0.355 (0.56), 125 min | 0.378 (0.43) | 0.333 | 0.025 | | step 450 |
| 2 | 0.373 (0.25), 107 min | 0.392 (0.36) | 0.392 | 0.407 | 0.377 | step 550 |

- In plain words: r5 is fine for its first 350 steps on every seed and breaks between steps 350 and
  550, so its trouble is all after ~2 h. A run that stops at step 300 would have scored 0.380 /
  0.355 / 0.373, every seed above the A line (0.2703), with 50 or more steps to spare before the
  earliest collapse.
- r7 (`calibration/research_r7/a01_fp32_lr1e-6_skip_cos_mal0_s300`): r5's config with
  `max_steps: 300`. The lr schedule is time-based over the 17,400-s budget, so the run is r5's first
  ~2 h and then stops: an eval at step 300, then the final weights. The budget is a cap, so a
  ~2-h track A run is legal. Weights are saved: 25-59 % of the greedy dev outputs at step 300
  were cut at 1,024 tokens, so a think budget at eval may add points. Eval arms
  `samp_s300_t050`, `_tb960` and `_tb768` are ready and queue when the weights exist.
- Queue: r7 seeds 0-2 at the front of `main` (~2.2 h each); they start at ~14:05, ~14:20 and ~15:20
  as running items end. Seeds at r5's level would meet the 3-seed form (every seed >= 0.2703,
  none collapsed; truncation-heavy evals count as collapsed only below the holder's mean, per the
  10-23 07:00 clarification).
- skip + cosine seed 3 has been one-guess since ~30 min (entropy 0.05, ~1-s rollouts) and is at
  0.208-0.228 at 85 min. Seeds 0 and 1 looked the same at that point (0.202 / 0.185 at 90 min) and
  ended at 0.300 / 0.270. Seed 4 is at 0.118 after 27 min.
- r6 seed 0 (think budget 768) learns slowly: 0.050 / 0.072 / 0.087 at steps 50 / 100 / 150 (66 min),
  against 0.14-0.215 for r5 at step 150. Its steps take ~27 s (rollouts run to the budget), entropy
  is still 1.1 at 60 min and the batch reward is 0.05 (r5: 0.23-0.30). My reading: with the budget,
  a search that rambles still ends in a forced guess that sometimes scores, so nothing pushes the
  policy to finish a search; r5's zero reward for truncation did. Its greedy outputs separate steps
  with stray code tokens ("\npaque Step 2: ..."); skip + cosine seed 2 (".debugLine") and r5 seed 0
  ("ibraries") showed the same kind of tokens early, so that is not the budget's doing. r6 seeds 1-2
  now wait behind r7; seed 0's end (~14:35) decides whether they run.
- dm26 skip seed 2: 0.093 at 48 min (seeds 0-1 had 0.098 / 0.087 at that point).

## 2026-10-23 (14:25): skip + cosine promotes (5-seed form); A 002 record config committed. dm26 seed 2 ends at 0.153

`a01_fp32_lr1e-6_skip_cos` (A 001 + fp32 master weights at lr 1e-6, cosine to 1e-7, zero-advantage
rows skipped), in-loop final dev:

| seed | A 001 (same seed) | final dev | puzzles | best | mode |
|---|---|---|---|---|---|
| 0 | 0.203 | 0.300 | 180 | 0.300 | one-guess (237-token searches at the end) |
| 1 | 0.237 | 0.270 | 162 | 0.277 | one-guess |
| 2 | 0.193 | 0.472 | 283 | 0.482 | ok |
| 3 | 0.268 | 0.315 | 189 | 0.327 | one-guess |
| 4 | 0.300 | 0.263 at 238 min (ends ~15:20) | | | one-guess |

- Promotion, 5-seed form: A 001's record seeds total 721 of 3,000 (0.2403), so the line is 0.2703,
  811 puzzles over five seeds. Seeds 0-3 already have 814, so the mean holds whatever seed 4 scores;
  seeds 0-3 are all above 0.2403 and none is collapsed (one-guess seeds count as collapsed only
  below the holder's mean). The rule is met.
- In plain words: across five seeds, skip + cosine beats A 001 on four (seed 4 is behind A 001's
  seed 4, 0.263 against 0.300, with 50 min left), by +0.11 on average over seeds 0-3 (0.339 against 0.225). It is the first
  arm of the month to meet the rule.
- Record: `records/v3/track_a/002_fp32_skip_cosine/config.json` is the screen's config with
  final weights saved and the scoring sampler at T 0.5, constrained, no think budget. T 0.5 was
  picked on A 001's dev evals (`calibration/research_r2/samp_a001_t050` against T 0.75 and 1.0); this
  arm's final evals truncate 0-0.2 % of outputs, so a think budget would not change anything.
  N = 5 (seeds 0-4), `scripts/run_seed_v3.sh` STAGE=all, all five reported, then
  `validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`.
- Where: Engaging's only mit_normal_gpu H100 node (node2906) estimates its next start at 10-26 06:46
  (`sbatch --test-only`), so AWS via the laptop session is likely sooner. All five seeds must train on
  one provider, so I have not submitted Engaging record workers; STATUS.md flags the promotion.

r4 `dm26_fp32_lr3e-6_skip` seed 2 ended at 0.153 (92 puzzles; L1-L6 0.55 0.06 0.06 0.02 0.00 0.23).
Seeds 0-2: 0.152 / 0.143 / 0.153, 269 puzzles, mean 0.1494.

- Promotion: the 3-seed form is out (seed 1). The 5-seed form needs 447 puzzles over five seeds, so
  seeds 3-4 must total 178 (0.148 each on average, the arm's mean so far): about even odds.
- Seeds 3-4 run as full pipelines (`calibration/research_r4/dm26_fp32_lr3e-6_skip_pipe`: the same
  26 % DM-math pretraining for 112 min, then the same RL), since only seeds 0-2 have saved
  pretraining. They follow r7 in `main` (~16:05 and ~16:30 starts) and finish ~22:00-22:30.
  Pretraining is time-budgeted, so I'll check their tokens/s against the ~370k of seeds 0-2.

r6 seed 0 (think budget 768 in training) is at 0.338-0.340 from 249 to 281 min, entropy 0.05-0.08,
22-23 informative groups: no sign of r5's collapse at the same time. It ends ~14:35. r7 seed 0's first
eval (step 50) is 0.027, as r5's seeds were at step 50 (0.027-0.042).

## 2026-10-23 (15:35): skip + cosine's five seeds average 0.324; r6 seed 0 ends at 0.318 (r6 parked); how r5 collapsed; r8 queued

```
  a01_fp32_lr1e-6_skip_cos           s0 done  290m final 0.300 best 0.300 t.10   21 spm   6.4 ent 0.08 inf  9.1 one-guess | 0.115 0.170 0.242 0.288 0.287
  a01_fp32_lr1e-6_skip_cos           s1 done  290m final 0.270 best 0.277 t.10   21 spm  39.3 ent 0.05 inf  4.6 one-guess | 0.133 0.187 0.220 0.245 0.262
  a01_fp32_lr1e-6_skip_cos           s2 done  290m final 0.472 best 0.482 t.10   30 spm   5.3 ent 0.06 inf 12.0 ok        | 0.112 0.192 0.353 0.428 0.455
  a01_fp32_lr1e-6_skip_cos           s3 done  290m final 0.315 best 0.327 t.10   21 spm  37.4 ent 0.04 inf  2.7 one-guess | 0.128 0.172 0.225 0.297 0.312
  a01_fp32_lr1e-6_skip_cos           s4 done  290m final 0.263 best 0.272 t.10   25 spm  32.3 ent 0.02 inf  2.7 one-guess | 0.135 0.168 0.233 0.248 0.267
= a01_fp32_lr1e-6_skip_cos             n=5 final 0.324 ± 0.085  ok/1-guess/div 1/4/0  median t.10 21m
  a01_fp32_lr1e-6_skip_cos_tb768     s0 done  290m final 0.318 best 0.340 t.10   86 spm   2.5 ent 0.10 inf 23.4 ok        | 0.050 0.072 0.212 0.305 0.318
```

- Skip + cosine, all five seeds: 972 of 3,000 dev puzzles (0.324) against A 001's 721 (0.240) on the
  same seeds, ahead on four of five (seed 4: 0.263 against 0.300). None collapsed. The promotion
  (14:25) stands; seed 4 changes nothing.
- r6 seed 0 (think budget 768 in training, otherwise skip + cosine) ended at 0.318 (191 puzzles),
  best 0.340 at step 550. It stayed healthy all 5 h: 2.5 steps/min of long searches, no
  truncation, 20-30 informative groups to the end. But per step it learns much more slowly than r5
  (malformed reward 0, no budget). Dev at step 300 is 0.255 against r5's 0.355-0.380, and 0.10 is
  first reached at 86 min against r5's ~25 min. A forced answer after 768 tokens is sometimes right,
  so a search that never finishes still earns reward now and then. That blurs the signal.
- Decision: r6 seeds 1-2 move to `later`. Even a steady 0.32-0.34 on every seed would not beat skip +
  cosine's 0.324 by the +0.03 a later attempt needs once A 002 is the holder. Its weights get two dev
  sampler evals (`calibration/research_r6/samp_t050`, `_tb768`, at the front of `main`, ~5 min each)
  to measure the T 0.5 record sampler with and without the training budget.

How r5 collapsed (seed 0's training log). From step 125 on, about half the training samples ran
out of tokens (0.44-0.59 malformed), while dev kept rising to 0.39 at step 350. Then the share rose
quickly: 0.59 at step 355, 0.73 at 360-370, 0.82 at 395. Solve rate in the batch fell from 0.39 to
0.04, and informative groups fell from 25-29 to 6-7. With reward 0 for a truncated output, a group
whose samples all truncate carries no signal and is skipped. So the slide into truncation removed
the gradient that could undo it. By step 410-420 the policy had moved elsewhere: entropy 0.04 -> 0.46,
outputs short again (0.2 malformed), solve rate 0.02. Seeds 1 and 2 went the same way 50-150 steps
later.

Why not a smaller penalty such as -0.03? Under GRPO's per-group z-score (`adv_norm: std`), a group
with no correct answer has rewards 0 (wrong) or the malformed reward, and the z-score removes the
scale. Any negative malformed reward pushes those groups toward finishing just as hard as -0.1,
which is skip + cosine's push toward one-guess early on, when most groups have no correct answer. So
the penalty can only usefully change over time.

r8 (`calibration/research_r8/a01_fp32_lr1e-6_skip_cos_mal0_until300`) is r5 (malformed reward 0)
for the first 300 steps, then the verifier's -0.1 for malformed outputs, over the full 290-min
budget. Hypothesis: r5's early search survives, because by step 300 half the batch's samples are correct
(r5 seed 0: solve rate 0.50), so the +1s dominate most groups' signal. The penalty then stops the
slide into truncation that ended r5. The risk is the one-guess push in groups with no correct
answer, as in skip + cosine. The arm needs a new option, `malformed_reward_until` (GRPOConfig; 0 = off, so other configs are
unchanged). It refuses to run without `malformed_reward`, and it has a test (122 pass). Seeds 0-2 are queued after
the dm26 pipelines; weights are saved for sampler evals.

Also:
- Preemption: both workers on node2701 were preempted at 14:59, after 15 h. Worker 26644470 lost
  c90_cc1e-3 seed 0 at 28 min, so it reruns. Worker 26644471 lost r7 seed 0 at ~68 min; its
  resumable state was saved at 14:54. The requeued pair rank below the two never-started workers
  26644474-75 (priority 35826 against 39536). So r7 seed 0 would have waited for a running worker to
  end, likely ~10-24 evening. I moved the item to the front of `main` and kept 26644471's in-hand file
  as `queue/running.26644471.moved_to_main_1600`, so 26644471 pops a fresh item when it starts. Its
  logs come back from the resume state.
- r7 seeds 3-4 (2 h each) are queued after the pipelines, so the 5-seed form is ready if the 3-seed
  form misses narrowly. If seeds 0-2 settle it, I take them out.
- r7 so far: seed 1 0.223 at step 150 (56 min; r5's seed 1 was 0.168 there), seed 2 0.065 at step 50.

## 2026-10-23 (18:10): r7 seeds 0-2 at 0.385 / 0.348 / 0.238 (3-seed form missed); record sampler on seed 1: 0.375

```
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s0 done  133m final 0.385 best 0.385 t.10   31 spm   2.1 ent 0.08 inf 22.9 ok        | 0.027 0.183 0.340 0.385 0.385
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s1 done  143m final 0.348 best 0.348 t.10   28 spm   1.9 ent 0.11 inf 27.8 ok        | 0.118 0.223 0.308 0.348 0.348
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s2 done  130m final 0.238 best 0.372 t.10   34 spm   2.3 ent 0.04 inf 15.9 ok        | 0.065 0.105 0.372 0.238 0.238
= a01_fp32_lr1e-6_skip_cos_mal0_s300   n=3 final 0.324 ± 0.076  ok/1-guess/div 3/0/0  median t.10 31m
  samp_s300_t050                     s1 dev16 0.3748 pass@16 0.598 tokens  746 trunc 0.565 | 0.969 0.415 0.262 0.172 0.044 0.388
```

- Seeds 0-1 repeat r5's step-300 values (r5: 0.380 / 0.355; r7: 0.385 / 0.348). Seed 2 reached
  0.372 at step 250 (r5's seed 2 had 0.373 at step 300) and fell to 0.238 by step 300. This was not
  r5's slide into truncation: 0.21-0.33 of its training samples truncated over steps 240-300. Its
  entropy fell from 0.09 to 0.034 and the batch solve rate from ~0.4 to 0.22. At the end L1 was 0.83
  (seed 0: 1.00) and L3-L5 were 0.01-0.06. Runs of the same seed differ (vLLM sampling isn't
  bitwise reproducible), so stopping at step 300 makes a collapse less likely but doesn't rule it out.
- Promotion against A 001 (line 0.2703):
  - The 3-seed form needs every seed at least 0.2703. Seed 2 misses it.
  - The 5-seed form needs 811 puzzles; seeds 0-2 have 583 (231 + 209 + 143). Seeds 3-4 must total
    228 or more and both finish above 0.2403, because seed 2 is already below the holder's mean.
  - Seeds 3-4 (about 2.2 h each) moved to the front of `main`.
- Record sampler, seed 1: T 0.5, constrained, no think budget scores dev16 0.375 (pass@16 0.598),
  above its greedy 0.348, even though 56.5 % of samples hit the 1,024-token limit (mean 746
  tokens). Those count as failures, so forcing an answer may score higher still.
  - Seed 1's tb960 and tb768 evals are next.
  - Seeds 0 and 2 get T 0.5 with and without tb960 after r7's seeds 3-4.
  - Each eval takes ~30 min here (long outputs), against 2-6 min for A 001's.
- Greedy in-loop dev truncates 24-34 % of this arm's outputs at steps 250-300 and counts them as
  wrong. The promotion rule stays on in-loop dev, as for every arm; the record-sampler evals pick the
  sampler.
- Preemption again: 26644474 started on node3009 at 17:35 and was preempted at 17:37, holding
  seed 1's tb960 eval. I moved the eval to the front of `main`; 26644474's in-hand file is kept as
  `queue/running.26644474.moved_to_main_1802`.
- r6: its T 0.5 eval (seed 0, no budget) is running. The tb768 eval went to `later`: r6 is parked,
  and r7's evals answer the think-budget question for the arm that matters.

## 2026-10-23 (19:35): r7's sampler, seed 1: a think budget adds nothing at T 0.5

```
  samp_s300_t050                     s1 dev16 0.3748 pass@16 0.598 tokens  746 trunc 0.565 | 0.969 0.415 0.262 0.172 0.044 0.388
  samp_s300_t050_tb960               s1 dev16 0.3762 pass@16 0.605 tokens  722 trunc 0.000 | 0.968 0.416 0.264 0.173 0.045 0.391
  samp_s300_t050_tb768               s1 dev16 0.3647 pass@16 0.593 tokens  610 trunc 0.000 | 0.963 0.396 0.244 0.161 0.039 0.385
  samp_t050 (r6 seed 0)              s0 dev16 0.3607 pass@16 0.585 tokens  709 trunc 0.516 | 0.952 0.406 0.265 0.131 0.040 0.370
```

- Forcing the answer at 960 tokens turns r7 seed 1's 56.5 % truncated samples into answers, and
  almost none of them are right: +0.0014. Forcing it at 768 costs 0.010, because it also cuts off
  searches that would have finished correctly. A search that hasn't found the answer by ~960 tokens
  has nearly always failed.
- r7's sampler is therefore T 0.5, constrained, no think budget, the same as A 002's. The tb960
  evals for seeds 0 and 2 are dropped. Their no-budget evals stay, after the dm26 pipelines.
- r6 seed 0 scores 0.361 at T 0.5 without a budget, against its in-loop 0.318. The in-loop eval
  uses the training config's think budget (768), so that number was greedy with forced answers at
  768. The record sampler gains more over greedy on long-search arms than on A 001 (+0.015 on
  average): r7 seed 1 +0.027, r6 seed 0 +0.043. QUESTIONS.md now asks whether promotion should use
  dev under the record sampler; until then the rule stays on greedy in-loop dev. r6 stays parked.

## 2026-10-23 (21:10): r7's five seeds average 0.324, so r7 is promoted as A 003 (tied with A 002 on dev)

```
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s0 done  133m final 0.385 best 0.385 t.10   31 spm   2.1 ent 0.08 inf 22.9 ok        | 0.027 0.183 0.340 0.385 0.385
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s1 done  143m final 0.348 best 0.348 t.10   28 spm   1.9 ent 0.11 inf 27.8 ok        | 0.118 0.223 0.308 0.348 0.348
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s2 done  130m final 0.238 best 0.372 t.10   34 spm   2.3 ent 0.04 inf 15.9 ok        | 0.065 0.105 0.372 0.238 0.238
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s3 done  115m final 0.348 best 0.348 t.10   42 spm   2.2 ent 0.13 inf 26.1 diverged  | 0.077 0.105 0.348 0.348 0.348
  a01_fp32_lr1e-6_skip_cos_mal0_s300 s4 done  118m final 0.302 best 0.315 t.10   49 spm   2.2 ent 0.04 inf 23.4 ok        | 0.098 0.178 0.302 0.302 0.302
= a01_fp32_lr1e-6_skip_cos_mal0_s300   n=5 final 0.324 ± 0.056  ok/1-guess/div 4/0/1  median t.10 34m
```

- Seed 3 ended at 0.348 (209 puzzles) and was still climbing (0.277 at step 250). Seed 4 ended at
  0.302 (181), just under its 0.315 at step 250.
- Promotion against A 001, 5-seed form: 973 of 3,000 puzzles (0.3243) against the 811 needed.
  - Seeds 0, 1, 3 and 4 are above A 001's 0.2403. Seed 2 (0.238) is the one allowed below it.
  - No seed collapsed. Seed 2 is not one-guess (15.9 informative groups) and not truncated past
    half its outputs.
  - Seed 3's "diverged" label means 60.5 % of its greedy outputs hit the 1,024-token limit. Its
    0.348 is above the holder's mean, so it doesn't count as collapsed (PLAN.md, clarified 10-23
    07:00).
- So r7 is a record attempt, `records/v3/track_a/003_fp32_mal0_s300`.
  - Its `config.json` is r7's config with a new description: max_steps 300, malformed reward 0,
    sampler T 0.5, constrained, no think budget.
  - N = 5 (seeds 0-4), `scripts/run_seed_v3.sh` STAGE=all, then
    `validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`.
- A 002 and A 003 tie on greedy dev (972 and 973 of 3,000). They differ elsewhere:
  - Spread: A 002's seeds are 0.263-0.315 plus one 0.472 (median 0.300). r7's are 0.302-0.385
    plus one 0.238 (median 0.348).
  - Cost: r7 trains for 115-143 min instead of 290, so a record seed costs under half the GPU time.
  - Record sampler: on seed 1 it adds 0.027 over greedy. A 002's gain can't be measured on dev,
    because its screens didn't save weights.
  - r7's sampler evals for seeds 0, 2, 3 and 4 (~30 min each) run first in `main`. Seed 0's
    started at 20:53.
- Neither record has landed, so both are measured against A 001. Whichever lands first becomes the
  holder. Then the other needs that holder's dev mean + 0.03 (0.354) and no longer qualifies. So
  either both start before one lands, or one is chosen. STATUS.md says this at the top for the
  laptop session.
- Engaging's node2906 (the only H100 80GB HBM3 node on mit_normal_gpu) now estimates 10-26 15:07
  for a 6-h job, and my six preemptable workers fill the 6-GPU cap. I have not submitted record
  workers; AWS via the laptop is sooner.
- The dm26 pipeline seeds 3 and 4 pretrain at 367k tokens/s (node2642, node2700), within the
  359-373k of D 000's record. They end around 02:00 and 02:30.
- Queue: r7's seed 3-4 sampler evals follow seed 2's. r8 seeds 3-4 are at the end of `main`, about
  a day out. They get dropped if r8's first three seeds fail.

## 2026-10-23 (22:50): A 003 under its record sampler: 0.337 over five seeds, 0.013 above greedy

```
  samp_s300_t050                     s0 dev16 0.3988 pass@16 0.597 tokens  715 trunc 0.553 | 0.992 0.436 0.290 0.210 0.068 0.397
  samp_s300_t050                     s1 dev16 0.3748 pass@16 0.598 tokens  746 trunc 0.565 | 0.969 0.415 0.262 0.172 0.044 0.388
  samp_s300_t050                     s2 dev16 0.2621 pass@16 0.391 tokens  781 trunc 0.665 | 0.810 0.256 0.126 0.057 0.008 0.316
  samp_s300_t050                     s3 dev16 0.3566 pass@16 0.596 tokens  758 trunc 0.592 | 0.960 0.379 0.280 0.132 0.025 0.365
  samp_s300_t050                     s4 dev16 0.2924 pass@16 0.503 tokens  756 trunc 0.485 | 0.930 0.274 0.237 0.034 0.009 0.270
= samp_s300_t050                       n=5 dev16 0.3369 ± 0.0575
```

- r7's seeds score 0.337 on average with A 003's sampler (T 0.5, constrained, no think budget),
  against 0.324 greedy. The gain is +0.013, about A 001's +0.015.
  - By seed: +0.014, +0.027, +0.024, +0.008 and -0.009. Seed 1's gain, measured first, was the
    high end.
  - 49-67 % of the samples hit the 1,024-token limit. Seed 2, which collapsed late, has the most
    (66.5 %) and the lowest pass@16 (0.391).
- So the sampler doesn't separate A 003 from A 002. A 002's gain can't be measured on dev; it
  should be near A 001's, since its outputs rarely truncate.
  - A 003's case is cost (training stops at step 300, ~2.2 h) and a higher median seed (0.348
    against 0.300).
  - A 002's case is its strongest seed (0.472).
  - STATUS.md's top note now says this.
- Under the record-sampler rule that QUESTIONS.md asks about (line 0.255 + 0.03 = 0.285), r7 still
  qualifies: mean 0.337, all five seeds above 0.255. That entry has the five-seed numbers now.
- r8 seed 0 (malformed reward 0 until step 300, then -0.1) started at 22:37 on node2700. Seeds 1-2
  start when the dm26 pipelines end (~02:00, ~02:30).

## 2026-10-24 (00:55): r8 seed 0 collapsed before its reward switch, so A 003's recipe failed on a sixth run

r8 is r5's config with the -0.1 truncation penalty back from step 300 on. Up to step 299 it is the
same training as r7 (A 003) and r5. Its seed 0 collapsed at steps 264-284, before the switch:

```
step  min   solve  trunc  inf  entropy   (training batch)      dev: 250 0.363, 300 0.053
 260  99.4  0.402  0.248   31  0.038
 264 101.5  0.357  0.277   29  0.040
 268 103.5  0.131  0.430   12  0.067
 276 107.2  0.049  0.449   11  0.141
 284 110.0  0.018  0.459    7  0.256
 296 113.2  0.027  0.215    8  0.284
```

- The batch solve rate fell from 0.36 to 0.13 in four steps and to 0.02 by step 284. Entropy rose
  from its low of 0.038 to 0.26-0.36. Truncation stayed at 0.2-0.48 throughout. This is not r5's
  late slide into truncation. It looks like r7 seed 2's fall (0.372 at step 250 to 0.238 at 300,
  entropy down to 0.034), gone further.
- Since step 300 the -0.1 penalty has cut truncation to ~0 at a dev of 0.04-0.05 (step 472: solve
  0.09, 17 informative groups). The run tells nothing about r8's question, which is whether the
  penalty after step 300 holds off r5's late collapse. Seeds 1-2 start at ~02:00 and still ask it.
- Every run of this recipe through step 300, in dev puzzles of 600:

  | run | step 200 | step 250 | step 300 |
  |---|---|---|---|
  | r5 s0 / s1 / s2 | 165 / 163 / 151 | 200 / 193 / 176 | 228 / 213 / 224 |
  | r7 s0 / s1 / s2 / s3 / s4 | 181 / 158 / 193 / 122 / 164 | 204 / 185 / 223 / 166 / 189 | 231 / 209 / 143 / 209 / 181 |
  | r8 s0 | 188 | 218 | 32 |

  Over the nine runs, step 300 averages 0.309 (1,670 of 5,400). Two runs fell between steps 250
  and 300, and they had the two highest step-250 scores (223 and 218). The collapse seems to come
  once a run is far enough along. Faster seeds reach that point before step 300, and r5's other
  seeds reached it after step 350.
- What this means for the records:
  - A 003's promotion stands under the rule, which is met on r7's own five seeds. The nine runs
    would still meet it (mean 0.309 against the 0.2703 line, 7 of 9 above 0.2403, one collapsed).
  - In expectation, though, A 003 is below A 002: about 0.309 against 0.324. About one record seed
    in five may collapse.
  - A 002's five seeds all ran the full 290 min without collapsing.
  - STATUS.md's top note now recommends A 002 if only one runs. That reverses the 21:15 note,
    which had preferred A 003 for its cost and median.
- The non-collapsed runs reach 0.35-0.385 by step 300, above A 002's typical seed. A stable version
  of this regime is the most direct way past the next line, which will be A 002's or A 003's dev
  mean + 0.03 = 0.354. Collapse fixes already in the code: Clip-Cov (`cov_clip_frac`, needs
  skipping off), a KL penalty (`kl_coef`, same), a lower lr, and r8's switch.

## 2026-10-24 (02:35): dm26's five seeds average 0.153, so dm26 is promoted as D 002 and its record seeds are queued here

```
  dm26_fp32_lr3e-6_skip              s0 done  234m final 0.152 best 0.162 t.10   34 spm  73.2 ent 0.06 inf  5.8 one-guess | 0.098 0.118 0.128 0.145 0.152
  dm26_fp32_lr3e-6_skip              s1 done  234m final 0.143 best 0.147 t.10   80 spm  79.9 ent 0.04 inf  3.1 one-guess | 0.087 0.085 0.137 0.138 0.143
  dm26_fp32_lr3e-6_skip              s2 done  234m final 0.153 best 0.153 t.10   54 spm  44.7 ent 0.06 inf  4.8 one-guess | 0.082 0.103 0.120 0.137 0.153
  dm26_fp32_lr3e-6_skip_pipe         s3 done  234m final 0.158 best 0.160 t.10   41 spm  77.0 ent 0.05 inf  4.6 one-guess | 0.070 0.112 0.143 0.147 0.158
  dm26_fp32_lr3e-6_skip_pipe         s4 done  234m final 0.157 best 0.173 t.10   62 spm  71.9 ent 0.05 inf  3.6 one-guess | 0.065 0.095 0.142 0.165 0.157
```

- Pipeline seeds 3-4 included their own pretraining (112 min at 367-369k tokens/s on node2642 and
  node2700). They ended at 0.158 and 0.157 (95 and 94 puzzles), slightly above seeds 0-2, which ran
  RL from r2's saved pretraining.
- Both show FAILED in `worker.log`. That comes from the last line of `run_seed_v3.sh`
  STAGE=train, which moves the final weights, and these `no_save` screens have none. Training, the
  evals and `clock.json` all finished (20,995 s of the 21,600 s budget). Nothing to requeue.
- Promotion against D 000 (dev 0.1190; line 0.149 = 447 of 3,000):
  - The five seeds total 458 (0.1527) and all are above 0.119.
  - All five are one-guess, as every track D seed is (D 000's too). None is below the holder's
    mean, so none collapsed.
  - The 5-seed form is met.
- So dm26 becomes the record attempt D 002, `records/v3/track_d/002_dm_math_fp32`.
  - Its `config.json` is the pipeline config without `no_save`, because the test-ladder stage
    needs the weights.
  - Sampler: T 1.0, constrained, no budget, as D 000's. These one-guess policies score the same at
    T 0.25, 0.5 and 1.0 (0.150, 0.150, 0.149 on an fp32 D model, r3 `samp_d00fp32_*`).
  - The DeepMind arithmetic is the published v1.0 release with its lines as published
    (`scripts/research/prepare_dm_math.py`, md5-checked).
- The record runs on Engaging, which AGENT.md allows on preemptable for track D. N = 5 (seeds
  0-4).
  - The screen workers on mit_preemptable H100 80GB HBM3 run it, because new jobs have waited ~10 h
    to start. `run_item.sh` now runs items under `records/v3/` as record seeds (STAGE=all
    `run_seed_v3.sh`, token file copied to node-local disk; 4093bab).
  - A preempted seed's partial run goes to the record's `extra/` (the worker's restart rule) and
    the seed reruns. Every run is reported.
- Queue: D 002 seeds 0-1 lead `main`, then r9 seed 0, D 002 seeds 2-3, r9 seed 1, D 002 seed 4,
  then the rest. Seed 0 starts when r8 seed 0 ends (~03:30); the last should end ~18:00-22:00.
  - 26644472-73 hit their time limit at 10-25 00:26. A D seed (~6.1 h) popped by them after ~18:20
    would be cut, so I check the queue head around then.
- After the five seeds: README, `validate_record_v3.py --compare-to records/v3/track_d/000_baseline`,
  and the leaderboard if it beats D 000.

## 2026-10-24 (07:30): r8 seed 2 peaked at 0.415 and ended at 0.315, so the late penalty doesn't hold the peak; r8 seeds 3-4 parked

r8 is r5 (malformed reward 0, full 290-min budget) with the -0.1 truncation penalty back from step
300 on (`malformed_reward_until`). Two of its three seeds are done; seed 1's worker was preempted
at 03:55 (dev 0.390 at 124 min) and the seed resumes from its hourly checkpoint when the worker
restarts.

```
  a01_fp32_lr1e-6_skip_cos_mal0_unti s0 done  290m final 0.205 best 0.363 t.10   48 spm  35.7 ent 0.04 inf  5.1 one-guess | 0.093 0.242 0.037 0.173 0.193
  a01_fp32_lr1e-6_skip_cos_mal0_unti s1 RUN   124m final 0.390 best 0.390 t.10   47 spm   2.1 ent 0.10 inf 34.2 ok        | 0.097 0.203 0.390 0.390 0.390
  a01_fp32_lr1e-6_skip_cos_mal0_unti s2 done  290m final 0.315 best 0.415 t.10   32 spm   1.4 ent 0.03 inf 48.5 ok        | 0.067 0.212 0.373 0.397 0.373
```

Seed 2's dev after the switch (greedy, 600 puzzles; malformed = truncated at 1,024 tokens):

| step | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 570 (final) |
|---|---|---|---|---|---|---|---|---|
| minute | 79 | 103 | 139 | 175 | 209 | 244 | 280 | 295 |
| dev | 0.292 | 0.373 | 0.367 | 0.397 | 0.373 | 0.415 | 0.348 | 0.315 |
| truncated | 0.550 | 0.448 | 0.215 | 0.155 | 0.188 | 0.167 | 0.237 | 0.243 |

What it means:
- The penalty did what it was meant to at first: truncation fell from 45 % to 16 % within 100 steps,
  and dev rose a little, to 0.415 at step 500.
- Then dev fell 0.10 in the last 70 steps (35 min). The training solve rate fell from ~0.43 to
  0.23-0.26 over steps 541-559, truncation crept back to 24 %, and entropy kept falling (0.038 to
  0.029). That's the same late decline as r5's, only later and smaller. The lr was already near
  its cosine floor (1e-7 at 290 min), so a lower lr alone may not stop it.
- Seed 0 collapsed before the switch (logged 10-24 00:55) and ended at 0.205 (one-guess, so
  collapsed). With one seed collapsed and the other ending at 0.315, r8 can't beat A 002 (0.324,
  no collapses), so seeds 3-4 move from `main` to `later`. Seed 1 still finishes when its worker
  restarts.
- For the A 002 / A 003 choice nothing changes. A 003's training (r7) stops at step 300, before
  this decline; its risk is the earlier collapse (2 of 9 runs before step 300).

Also this morning (details in STATUS.md):
- D 002 seed 0 started at 03:33 on node2700 (pretraining at 367-370k tokens/s, inside D 000's
  359-373k) and is in RL. Seed 2 started at 07:12 (367k tokens/s).
- D 002 seed 1 was preempted at 06:42, at 110 of its 112 pretraining minutes. Its seed dir held only
  stdout.log, which the worker's restart check doesn't look for, so the rerun would have overwritten
  it. `run_item.sh` now moves such a dir to `extra/interrupted_<seed>_<n>` before the rerun (commit
  66c1f30), and the README will report the interrupted run.
- Preempted workers restart in backfill gaps (their time limits fell to 8-11 h, and can go to the
  5:30 `--time-min`), and the worker redoes its item in hand without checking the time left. A D
  record seed needs ~6.1 h, so `run_item.sh` now refuses a record seed with less time left than its
  budgets + 30 min (376 min for D 002; commit 5ea30f0). A refused seed shows as FAILED in
  worker.log with a "not started, requeue it" note, and I requeue it by hand.
- Four preemptions between 02:45 and 06:42 (26644470, -75, -74, -71). Two workers run; four are
  pending with items in hand.
- node2906 (mit_normal_gpu, the only HBM3 node there) estimates 10-26 10:45 for a 6-h job, so
  Engaging still can't run the track A records sooner than AWS.

## 2026-10-24 (07:35): r10 queued: A 002 with mean-centred advantages (Dr. GRPO)

The step logs show why the two track A recipes end where they do:

| run | steps in 290 min | informative groups (of 64) | sampling / update s per step | end |
|---|---|---|---|---|
| A 002 seed 1 (-0.1 penalty, z-score) | 10,610 | 1-8 | 1.0 / 0.5 | one-guess from ~20 min, 0.270 |
| A 002 seed 2 (same recipe) | 1,984 | ~15 | 7.0 / 5.4 | searching, 0.472 |
| r8 seed 2 (reward 0 for truncation to step 300) | 570 | 47-51 | 15.8 / 31.1 | searching, peak 0.415, 0.315 |

Four of A 002's five seeds were one-guess by ~20 minutes, and dev then grew slowly over thousands of
cheap steps. Only the seed that kept searching got far. With reward 0 for truncated outputs every
seed keeps searching (r5, r7, r8), with plenty of signal per step, but the searches then truncate
or the policy drifts.

The 10-23 entry explains the early push to one-guess: under the per-group z-score, a group with no
correct answer turns the 0 vs -0.1 gap between wrong and truncated outputs into full-scale
advantages toward shorter outputs. r8 tried changing the penalty over time. r10
(`calibration/research_r10/a01_fp32_lr1e-6_skip_cos_advnone`) instead keeps A 002's -0.1 penalty
and switches the advantage to mean-centring only (`adv_norm: none`, Dr. GRPO, as track D already
uses). Then a group with no correct answer gets advantages of at most 0.1 in size, while a solve in
a 1-of-8 group still gets +0.875. That's a weak push against truncation throughout, instead of none
(mal0) or a full-scale one (z-score). It is A 002's screen config with one key added (no code
change).

If it works, the seeds keep searching past step 300 as r7's did, with truncation held down as A 002's
were, and end above A 002's 0.324 with no collapses; the 3-seed form needs every seed >= 0.2703.
Seeds 0-2 are in `main` after D 002's seeds 3-4, interleaved with r9 and dm51. Oversampling was the
other candidate, but the search runs already have 47-51 informative groups per step, so it would
help only the one-guess runs, whose ceiling is low.

## 2026-10-24 (09:35): D 002 seed 0 scores 0.150 on the test ladder

The first D 002 record seed finished at 09:25.
- Pretraining: 112.0 min and 2.49 B tokens (370k tokens/s), ending at val loss 2.887.
- RL: 15,693 steps. The in-loop dev score (greedy) was 0.167 at the end and 0.170 at best.
- Total time: 348 of 360 min, within budget (`clock.json`).

Test ladder (2,400 puzzles × 16 samples, T 1.0, constrained):

| | ladder score | pass@16 | mean tokens |
|---|---|---|---|
| D 002 seed 0 | 0.1496 | 0.183 | 25 |
| D 000 seeds 0-4 | 0.098 / 0.139 / 0.112 / 0.127 / 0.108 (mean 0.1166) | 0.134-0.202 | 20-29 |

By level: L1 0.471, L2 0.121, L3 0.083, L4 0.024, L5 0, L6 0.199.

What it means:
- This seed scores above all five D 000 seeds, 0.010 above the best of them.
- The record is judged on the five-seed mean (`validate_record_v3.py`), so this is one result of
  five; every seed counts, and nothing about the run changes.
- Remaining seeds:
  - Seed 2 ends ~13:00.
  - Seed 3 started at 09:25 on the same worker and ends ~15:15.
  - Seed 4 starts when seed 2 ends.
  - Seed 1 reruns when 26644471 restarts.

## 2026-10-24 (09:40): in the step logs, the mal0 collapses start with a truncation spike, not low entropy

The training steps (reward / entropy / truncated fraction), for the four runs that trained with
reward 0 for truncated outputs and collapsed:

| run | before | collapse | after |
|---|---|---|---|
| r5 seed 0 | step 300: 0.50 / 0.086 / 0.44 | step 375-400: 0.19-0.04 / 0.04-0.16 / 0.70-0.73 | step 450: 0.02 / 0.40 / 0.24 |
| r5 seed 1 | step 400: 0.40 / 0.031 / 0.20 | step 425: 0.11 / 0.08 / 0.59 | step 450: 0.03 / 0.22 / 0.23 |
| r5 seed 2 | step 500: 0.35 / 0.048 / 0.28 | step 525: 0.13 / 0.05 / 0.47 | step 550: 0.04 / 0.13 / 0.29 |
| r8 seed 0 | step 264: 0.36 / 0.040 / 0.28 | step 266-276: 0.19-0.05 / 0.05-0.14 / 0.43-0.45 | step 286: 0.02 / 0.30 / 0.41 |

What it means:
- Each collapse has the same order:
  1. Entropy has decayed to 0.03-0.09.
  2. Within ~25 steps the truncated share jumps to 0.45-0.73, and the reward falls.
  3. Entropy then rises to 0.2-0.4.
  4. The reward stays at 0.02-0.12 for the rest of the run; none of the four recovered.
- Low entropy alone doesn't do this. A 002 seed 2 (reward -0.1 for truncated outputs, 0.472 at the
  end) reached entropy 0.017 by step ~1,070 with 0.00-0.01 truncated throughout, and kept improving
  to step 1,984.
- So the trigger is the truncation that the reward-0 runs leave unchecked, which r8's late penalty
  slowed but didn't stop. It is less likely to be kernel-level sampler/trainer mismatch, which
  would also strike the low-entropy -0.1 runs. I'm not starting the mismatch-logging and
  importance-correction project for now.
- r10 (the -0.1 penalty with mean-centred advantages) tests the remaining middle ground: a push
  against truncation from the start that is too weak to drive the early one-guess collapse.

## 2026-10-24 (13:15): D 002 seed 2 scores 0.136 on the test ladder; seed 4 started

Seed 2 finished at 13:04:
- Pretraining: 112.0 min and 2.48 B tokens (370k tokens/s), ending at val loss 2.870.
- RL: 14,846 steps. The in-loop dev score was 0.150 at the end and 0.152 at best.
- Total time: 348 of 360 min.
- Test ladder: 0.1361 (pass@16 0.171, 28 mean tokens, nothing truncated). By level: L1 0.473,
  L2 0.076, L3 0.062, L4 0.024, L5 0.005, L6 0.176.

So far D 002 has 0.1496 and 0.1361 on test. D 000's five seeds scored 0.098-0.139 (mean 0.1166), so
seed 2 is 0.003 below D 000's best seed. Seed 4 started at 13:04 on the same worker and ends
~18:50. Seed 3 ends ~15:15. Seed 1 still waits for its worker (pending on Resources).

## 2026-10-24 (15:25): D 002 seed 3 scores 0.186 on the test ladder; seed 1's rerun crashed on node2801 and is requeued

Seed 3 finished at 15:14:
- Pretraining: 112.0 min and 2.48 B tokens (370k tokens/s), ending at val loss 2.867.
- RL: 15,394 steps. The in-loop dev score was 0.193 at the end and 0.195 at best.
- Total time: 348 of 360 min.
- Test ladder: 0.1857 (pass@16 0.234, 21 mean tokens, nothing truncated). By level: L1 0.563,
  L2 0.154, L3 0.080, L4 0.074, L5 0.016, L6 0.227.

D 002's test scores so far are 0.1496 / 0.1361 / 0.1857 (seeds 0, 2, 3; mean 0.157). D 000's mean
is 0.1166.

Seed 1, attempt 2 (attempt 1 is `extra/interrupted_1_1`):
- Worker 26644471 restarted at 15:05 on node2801 and reran seed 1. `run_item.sh` moved attempt 1
  to `extra/interrupted_1_1` first, as intended.
- After the 4.6 GB data copy (~13 min on that node), pretraining crashed at its first CUDA call
  (`model.cuda()`): "CUDA error: CUDA-capable device(s) is/are busy or unavailable".
- That is an infrastructure failure. RULES_v3 ("Seeds and statistics") allows such a seed to be
  rerun if the failed attempt goes in `extra/` with a note. It is now in `extra/failed_1_1`.
- Seed 1 is requeued at the head of `long`. Worker 26644470 reads `long` first and runs on
  node2700, where seeds 0, 2 and 3 ran. It will have ~9.6 h left when its 90-min c90 screen ends at
  ~16:45, and the time guard requires 376 min. Attempt 3 should then end at ~22:35.
- At 15:23, nvidia-smi inside the job showed the GPU idle (0 MiB, Default compute mode, no
  processes), so the fault may have been transient. r9 seed 1 is starting on the same GPU. If it
  fails the same way, I'll requeue it and exclude node2801 from later submissions.

## 2026-10-24 (16:35): a node2801 GPU that CUDA can't open stopped four workers; run_item.sh now tries CUDA first

What failed:
- r9 seed 1 failed on that GPU at 15:27 with the same "CUDA-capable device(s) is/are busy or
  unavailable" error as D 002 seed 1's second attempt.
- Worker 26644471 then failed r10 seed 1 and dm51 seed 0 within 10 s each, and exited. The
  two-quick-failures rule put dm51 back.
- The freed GPU went to my two pending workers in turn, and both exited the same way:
  - 26644474 at 15:30: r9 seed 0 failed, dm51 seed 0 put back.
  - 26644475 at 15:32: r8 seed 1 failed, dm51 seed 0 put back.
- 26649457, 26644471's successor, took dm51 seed 0 at 15:53. It was preempted three times in 30 min
  (node2904, node3009, node3209), then reached node2801 at 16:29. It failed dm51 seed 0, put r9
  seed 0 back, and exited.

Every failure is the same error at the first CUDA call. Other users' jobs hold five of the node's
eight GPUs. This is node1800's black-hole pattern (10-21) with a different fault.

What I changed:
- Requeued the host failures, once each: dm51 seed 0, r9 seeds 0-1, r10 seed 1 and r8 seed 1.
  r8 seed 1's resume state from 03:50 (20.6 GB) is intact.
- `main` (16) is now: dm51 s0, r9 s0, r10 s1, dm51 s1, r9 s1, r10 s2, r9 s2, dm51 s2, r8 s1, then
  d00 / a01_skip as before.
- `scripts/research/run_item.sh` (16:32) runs `torch.zeros(1, device="cuda")` before each item,
  with a 150-s timeout.
  - If that fails, it puts the item back at the front of the worker's first queue and empties the
    worker's in-hand file. The worker's exit after two quick failures then uses nothing up.
  - I tested it in a sandbox with a fake python: a failure on a `main:long` worker, a failure on a
    `long:main` worker, a pass for a screen, a pass for a record, and outside Slurm.
- Three replacement workers (26786843, 26786844, 26786872; `main:long`) exclude node2801.
- The three pending successors were submitted before this and don't exclude it. QUESTIONS.md asks
  whether I may update them.
- Workers that could run together: 3 running chains plus the 3 new ones, which makes 6.

D 002 seed 1's third attempt is still at the head of `long` for 26644470 (node2700, ~16:45).

## 2026-10-24 (16:57): Clip-Cov alone (c90_cc1e-3) stays in one-guess mode at 0.162 after 90 min; D 002 seed 1's third attempt is pretraining

The 90-min collapse screens (fp32, lr 1e-6, constant lr), in-loop greedy dev at the end:

| arm | seeds | final dev | mode | entropy at the end |
|---|---|---|---|---|
| `c90_mal0` (truncated = 0, skipping) | 0, 1 | 0.322, 0.383 | searching | 0.11 |
| `c90_t12` (training temperature 1.2, skipping) | 0 | 0.212 | one-guess | 0.05 |
| `c90_cc1e-3` (Clip-Cov on 0.1 % of tokens, no skipping) | 0 | 0.162 (best 0.182) | one-guess | 2.25 |

In `c90_cc1e-3` seed 0:
- Entropy fell from 1.92 to 0.40 by step 100, then rose steadily to 2.25 at step 549.
- The training solve rate rose from 0.06 to 0.22.
- Informative groups stayed at 8-17 of 64.
- No output was truncated.

In plain words: Clip-Cov at 0.1 % does stop entropy from collapsing, but on its own it doesn't get
the model to search. It answers in one guess, and greedy dev trails the training reward.

r9 (r5's malformed-reward-0 recipe plus the same Clip-Cov) still tests the case that matters: does
keeping entropy up prevent the truncation collapse that follows entropy decay in r5 and r8 (LOG
09:40)? Its 3 seeds stay queued. The other `c90_cc1e-3` and `c90_t12` seeds stay parked in `later`.
`c90_pos01` seed 0 is next in `long`.

D 002 seed 1's third attempt started at 16:39 on node2700 and passed the new CUDA check. It is
pretraining at 364k tokens/s. RL starts at ~18:32 and the seed ends at ~22:35.

## 2026-10-24 (18:00): the CUDA check now puts an item back only on a CUDA error

- Worker 26786843 started at 17:06 on node4109. Its first CUDA check timed out after 150 s with no
  output, so it put dm51 seed 0 back and took it again. It was preempted at 17:09, during the
  second check. dm51 seed 0 is still in its hand, so nothing was lost.
- I don't know whether that was a slow first import of torch from the pool venv or CUDA hanging at
  start-up. node4109 is the slow node from D's pretraining notes (224k tokens/s).
- `run_item.sh` now puts the item back only when the check fails with a CUDA error. A timeout or
  any other failure gets a note in worker.log, and the item runs as before.
- Tested in the sandbox: a CUDA error, a timeout, an import error, a pass, outside Slurm, and a
  record on a `long:main` worker.

## 2026-10-24 (19:00): D 002 seed 4 scores 0.147 on the test ladder; four seeds average 0.155

Seed 4 finished at 19:00:
- Pretraining: 112.0 min and 2.49 B tokens (370k tokens/s), ending at val loss 2.851.
- RL: 14,106 steps. The in-loop dev score was 0.155 at the end and 0.162 at best.
- Total time: 348 of 360 min.
- Test ladder: 0.1469 (pass@16 0.180, 24.7 mean tokens, nothing truncated). By level: L1 0.492,
  L2 0.084, L3 0.070, L4 0.029, L5 0.008, L6 0.198.

| D 002 seed | test ladder | pass@16 | in-loop final dev |
|---|---|---|---|
| 0 | 0.1496 | 0.183 | 0.167 |
| 2 | 0.1361 | 0.171 | 0.150 |
| 3 | 0.1857 | 0.234 | 0.193 |
| 4 | 0.1469 | 0.180 | 0.155 |
| mean of four | 0.1546 | | |

D 000's five seeds score 0.098-0.139 (mean 0.1166), so every D 002 seed so far beats D 000's best
seed. Seed 1's third attempt is in RL and ends at ~22:35.

Worker 26644473 had 325 min left when seed 4 finished, so it exited instead of starting a 5-h
item. Its successor 26649459 takes over the slot.

## 2026-10-24 (20:30): r10 seed 0 (A 002 with mean-centred advantages) switched to searching and ends at 0.340 without collapsing

r10 seed 0 (`calibration/research_r10/a01_fp32_lr1e-6_skip_cos_advnone`) finished at 20:15 after
the full 290 min. Same seed and recipe as A 002 except for the advantage (LOG 07:35):

| run | advantage | steps in 290 min | informative groups (mean, of 64) | mode | final dev (best) |
|---|---|---|---|---|---|
| r10 seed 0 | mean-centred | 599 | 26.7 | searching | 0.340 (0.378 at step 500) |
| A 002 seed 0 | z-score | 4,127 | 9.1 | one-guess | 0.300 (0.300) |
| A 002 seeds 0-4 | z-score | 1,984-11,147 | 2.7-12.0 | 1 searching, 4 one-guess | 0.263-0.472, mean 0.324 |

The training steps:
- Steps 1-100 looked like A 002's one-guess seeds. The truncated share fell from 0.69 to 0.004,
  informative groups from 58 to 23 and entropy from 1.9 to 0.69; dev was 0.058 at step 100.
- Around steps 120-150 it switched to searching. At step 150, 31 % of samples were truncated, 63
  groups were informative, and sampling took 14.5 s per step; dev was 0.200.
- From step 200 to the end, the truncated share stayed at 0.02-0.11 in training (0-0.115 of the
  greedy dev outputs) while entropy fell to 0.03. Dev reached 0.378 at step 500 and ended at 0.340.

What it means:
- This seed did what r10 was queued to test: it searched for the whole run with truncation held
  down and didn't collapse in 290 min. The reward-0 runs that trained past step 300 (r5, r8)
  collapsed at steps 264-550 when the truncated share jumped to 0.45-0.73 (LOG 09:40); here it
  never rose above 0.12 after step 200.
- 0.340 is above A 002's seed 0 (0.300) and the 0.324 mean of A 002 and of r7, but inside A 002's
  spread, and it is one seed. A 002's seed 2 also searched and reached 0.472. The question is
  whether every r10 seed searches. The 3-seed form of the promotion rule needs seeds 1 and 2 at
  >= 0.2703 too, with none collapsed. Seed 1 is in 26786872's hand, seed 2 second in `main`.

Workers:
- 26644472 exited after r10 seed 0 (less than 330 min left). Its successor 26649458 started at
  20:23 on node2801. The CUDA check failed there twice ("busy or unavailable"), put r9 seed 1 back
  both times, and the worker exited after its two quick failures. Nothing was used up; r9 seed 1
  is at the front of `main` again.
- That ended the chain, so I submitted 26803673 (`main:long`, node2801 excluded). 26644470 runs
  and five workers are pending (26649459, 26786843, 26786844, 26786872, 26803673), 6 in all;
  26649456 waits on 26644470.
- Between 19:48 and 20:24, 26649459, 26786843, 26786844 and 26786872 each started and were
  preempted within 1-4 min. They hold dm51 seed 1, dm51 seed 0, r9 seed 0 and r10 seed 1 and
  redo them when they restart.

## 2026-10-24 (22:39): D 002 is a record: 0.1512 ± 0.0202 over five seeds, against D 000's 0.1166 (one-sided Welch p = 0.0092)

Seed 1 (attempt 3) finished at 22:34:
- Pretraining: 112.0 min and 2.47 B tokens (367k tokens/s), ending at val loss 2.871.
- RL: 9,887 steps; the other seeds ran 14,106-15,693. Its sampling took 1.18 s per step on
  average against 0.74-0.83 s for the others, on the same node. I don't know why. The in-loop dev
  score was 0.137 at the end and 0.143 at best.
- Total time: 349.5 of 360 min.
- Test ladder: 0.1375 (pass@16 0.188, 24 mean tokens, nothing truncated). By level: L1 0.463,
  L2 0.133, L3 0.008, L4 0.045, L5 0.008, L6 0.169.

| D 002 seed | test ladder | pass@16 | in-loop final dev |
|---|---|---|---|
| 0 | 0.1496 | 0.183 | 0.167 |
| 1 | 0.1375 | 0.188 | 0.137 |
| 2 | 0.1361 | 0.171 | 0.150 |
| 3 | 0.1857 | 0.234 | 0.193 |
| 4 | 0.1469 | 0.180 | 0.155 |
| mean | 0.1512 (sd 0.0202) | 0.191 | 0.160 (481 of 3,000) |

`validate_record_v3.py --compare-to records/v3/track_d/000_baseline`: Welch t = 2.98, df = 7.7,
one-sided p = 0.0092, "new record".

What it means:
- D 002 is the track D holder. Committed: all five seeds, seed 1's two failed attempts in
  `extra/`, the README, and the leaderboard row in README.md. I also updated the README.md sentence
  that said public datasets "haven't been tried yet".
- The gain over D 000 (+0.035) is spread over all six levels; nearly half of it is on L1. The
  model still answers at once and writes no intermediate arithmetic (record README, "What the
  model does").
- Correction to the 19:00 entry, which said every D 002 seed so far beat D 000's best seed
  (0.1391). Seed 2 (0.1361) doesn't, as the 13:15 entry said, and neither does seed 1 (0.1375).
  Three of the five do.
- New track D promotion line: D 002's dev mean is the record seeds' in-loop final dev, 0.160 (as
  for D 000's 0.119, PLAN.md "Reference points"). So a D candidate needs a mean of 0.190 (571 of
  3,000) with 4 of 5 seeds above 0.160, or 3 seeds each at >= 0.190 (115 of 600). No D screen so
  far reaches that: the best five-seed screen is D 002's own recipe at 0.153.
- `validate_all_records.py` passes D 002 and fails only A 002 and A 003. Their folders hold just
  `config.json` until their seeds run.

Workers: 26644470 exited after seed 1 (less than 330 min left). Its successor 26649456 is pending.

## 2026-10-24 (22:43): r11 queued: D 002's RL at lr 1e-5 and 3e-5

To beat D 002, a track D screen now needs 0.190 on dev. The fp32 RL learning rates tried so far
(final dev, in-loop):

| RL | pretraining | seeds | final dev | dev at 30 / 60 / 120 / 180 min |
|---|---|---|---|---|
| bf16 weights, lr 3e-5 (D 000) | D 000's | 5 | 0.119 | |
| fp32, lr 1e-6 | D 000's | 1 | 0.098 | |
| fp32, lr 3e-6 | D 000's | 5 | 0.137 ± 0.009 | |
| fp32, lr 3e-6, skipping (D 002's RL) | DeepMind mix (dm26) | 5 | 0.153 | seed 0: 0.098 / 0.118 / 0.128 / 0.145 |

- The best rate is the highest one tried (1e-6 -> 3e-6 added 0.039), and dev was still rising at
  the end of the 234 min (dm26 seed 0: 0.145 at 180 min, 0.152 at the end). So the RL looks
  under-stepped, not stuck.
- r11 runs D 002's RL with only the lr changed, at 1e-5 and at 3e-5
  (`calibration/research_r11/dm26_fp32_lr1e-5_skip`, `dm26_fp32_lr3e-5_skip`). It starts from the
  saved dm26 pretrained models of r4's seeds 0-2, so each seed is paired with r4's 0.152 / 0.143 /
  0.153.
- With fp32 master weights an update of lr 3e-5 is applied in full, unlike D 000's bf16 weights,
  so 3e-5 may be too high. The two rates bracket the next step.
- Promotion against D 002 (dev 0.160): the 3-seed form needs each seed at >= 0.190 (115 of 600),
  none collapsed.

Queue (`main`, 16):
- Seed 0 of each rate goes second and fourth, between r9 seed 1 and r10 seed 2.
- Seeds 1-2 of each go after d00_fp32_lr3e-6_skip seeds 0-2. I'll move them up if a seed 0 is
  clearly above its r4 pair.
- d00_fp32_lr3e-6_skip seeds 3-4 moved to `later`. Three seeds paired with d00_fp32_lr3e-6's are
  enough to measure skipping on its own, because D seeds spread by only 0.005-0.009.

## 2026-10-25 (00:47): A 002's record workers submitted, then cancelled on the laptop's HOLD (the reference GPU is changing to H200)

INBOX (laptop, committed 00:12):
- I may update or cancel my own pending jobs without asking.
- AWS found no H100 capacity in 48 h, on-demand or spot. Nothing ran there and $0 was spent, so the
  a01_skip seeds 0-2 never ran.
- Start A 002's record on Engaging (mit_normal_gpu, HBM3), seeds 0-4 in order. Hold A 003.

What I did:
- Cancelled 26649456, a pending preemptable successor with no item in hand. That kept 4
  preemptable workers + 2 record workers at the 6-GPU cap.
- Queued A 002 seeds 0-4 in `queue/record-002_fp32_skip_cosine.txt`. At 00:45:43 I submitted five
  chained mit_normal_gpu workers, 6 h and one seed each: 26821229 and 26821230, then 26821231,
  26821232 and 26821234 `afterany` on them. They used `scripts/research/run_item.sh` (a CUDA check,
  a time-left check, then `STAGE=all scripts/run_seed_v3.sh`, as D 002's seeds ran).
  - `sbatch --test-only` put the first start on node2906 at 10-27 09:04, with 34 H100 jobs pending
    there.
  - mit_normal_gpu has one H100 node (4 GPUs) beside 13 H200 nodes (8 GPUs each).

A second laptop note was committed at 00:45:29, 14 s before my submission: hold record workers,
because the maintainer is changing the reference GPU to H200, with details within the hour.
- I cancelled all five record workers at 00:46:34, before any started.
- The queue file is set aside as `queue/record-002_fp32_skip_cosine.held`.
- A 003 stays held. Screens continue on H100 HBM3 until the details say otherwise.

a01_skip (A 001 + skipping): not requeued. A 002 includes skipping, so a01_skip would now only be
an ablation of it. Its seeds 3-4 moved from `main` to `later`.

Preemptions: 26786844 (r9 seed 0) at 23:40, and 26803673 (dm51 seed 1) and 26786872 (r10 seed 1)
at 23:51. All three are pending and resume from their hourly saves. 26786843 (dm51 seed 0, node2700)
still runs.

## 2026-10-25 (01:24): v3.1, the reference GPU is now the H200 SXM; three H200 reruns are queued, H100 screens wind down, and r12 is the first H200 screen

The maintainer's change (commit 18a5b37, INBOX 01:40):
- The reference GPU is the H200 SXM (`NVIDIA H200`), not the H100 80GB SXM.
- v3.0's H100 records (A 000-001, D 000-002) keep their scores on a separate H100 board, and
  `validate_record_v3.py --compare-to` refuses mixed-GPU comparisons.
- The H200 board starts from reruns of the H100 holders' configs.
- Screens may use any GPU type, compared only within one type, but the promotion screen that gates a
  record runs on the H200.
- Until the reruns land, promotion uses the H100 lines (A 0.2703, D 0.190); then they come from the
  reruns' in-loop final dev (PLAN).

Records (laptop's order: A 004, A 002, D 003; A 003 waits):

| record | config | workers |
|---|---|---|
| track_a/004_ladder_constrained_h200 | A 001's, copied byte for byte | chain 1, 26824033-37 (A 004 first, then A 002) |
| track_a/002_fp32_skip_cosine | as committed on 10-23 | chain 2, 26824038-42 (A 002 first, then A 004) |
| track_d/003_dm_math_fp32_h200 | D 002's, copied byte for byte | 26824147-50 on mit_preemptable |

- Track A runs on mit_normal_gpu, 6 h jobs of one seed each, `--gres=gpu:h200:1`,
  REQUIRE_GPU="NVIDIA H200", `scripts/research/run_item.sh` -> `STAGE=all scripts/run_seed_v3.sh`.
  `sbatch --test-only` estimates node2434 at 10-26 21:44. mit_normal_gpu has 13 H200 nodes.
- A D seed needs 376 min (budgets + 30), more than a 6-h normal job holds, so D 003 runs on
  preemptable H200 workers. They exclude the NVL nodes, use `--time-min=06:30:00` so a short
  backfill gap can't start one, and RESERVE_MIN 380. Each is `afterany` on one of the four H100
  workers.
- Like D 002's, D 003's seeds keep their pretrained models in
  `~/orcd/pool/rlvr-ckpts/records/v3/track_d/003_dm_math_fp32_h200/seeds/<seed>/pretrained`. So
  RL-only H200 screens can start from them (`screen_pretrained`), as r4 and r11 do from r2's H100
  models.

H100 screens:
- The four H100 workers finish the items in their hands and then find `main` and `long` empty:
  - dm51 seeds 0-1;
  - r9 seed 0;
  - r10 seed 1.
- The other 15 items went to `queue/h100.txt`, which no worker reads:
  - r9 seeds 1-2 and r10 seed 2;
  - r11's six seeds (paired with r4's H100 seeds);
  - dm51 seed 2;
  - r8 seed 1;
  - d00_fp32_lr3e-6_skip seeds 0-2;
  - c90_pos01.
- They still count for comparisons within the H100. But promotion now needs H200 screens, and
  r11's pairs are H100 runs. So H200 time goes to H200 arms.

r12 (`calibration/research_r12/a01_fp32_lr1e-6_skip_cos_advnone_h200`, `queue/h200.txt`, seeds 0-2):
- It is r10's config unchanged (A 002 + mean-centred advantages) on the H200.
- r10 is the best open track A idea: on the H100 its seed 0 searched through the whole run and
  ended at 0.340, against A 002's seed 0 at 0.300.
- A 002's H200 record seeds run the same recipe with the per-group z-score, so they pair with r12
  seed for seed.
- The D 003 workers turn to `h200` once the record seeds are taken.

GPU cap: mit_normal_gpu runs 2 (the chain heads), and mit_preemptable 4 (the H100 workers now, their
H200 successors after them). That is 6.

## 2026-10-25 (02:28): r9 seed 0 diverged and is stopped; H200 waits; two H100 workers get r11's seed 0s

r9 seed 0 (`a01_fp32_lr1e-6_cos_mal0_cc1e-3`: r5's recipe, malformed reward 0, with Clip-Cov 0.1 %
and skipping off) diverged:

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|
| dev | 0.040 | 0.072 | 0.058 | 0.088 | 0.095 | 0.002 | 0.002 |
| malformed | 0.455 | 0.308 | 0.328 | 0.150 | 0.068 | 0.960 | 0.983 |

- Training entropy went 1.9 (step 1) -> 3.6 (step 161) -> 2.2 (step 281) -> 5.7-6.1 (steps
  321-361).
- So Clip-Cov at this rate first raised entropy without producing search, then the policy blew up.
  Its dev never got past r5's or A 001's early values.
- I cancelled its worker (26786844) at 02:27, 129 min in. Nothing could recover from 98 % malformed
  output at entropy 6. Its logs are in the seed dir; I removed its 20 GB resume state.
- r9 seeds 1-2 moved to `later` (dropped). Together with c90_cc1e-3 (one-guess at 0.162 with
  entropy rising to 2.25; LOG 10-24 16:57), Clip-Cov is closed for track A.

Still running on the H100:

| seed | at | dev now | same seed, other arm |
|---|---|---|---|
| dm51 seed 0 (51 % DeepMind share) | 224 of 234 min | 0.167 | dm26 seed 0 ended at 0.152 |
| r10 seed 1 (A 002 + mean-centred advantages) | 140 min | 0.285 | A 002 seed 1 ended at 0.270; r10 seed 0 ended at 0.340 |

H200 capacity (02:24):
- mit_normal_gpu: Slurm plans my two A record chain heads for 10-28 07:00 and 07:30. A new 6-h
  job is estimated for 10-27 02:10.
- mit_preemptable: a new H200 job is estimated for 10-25 21:48.
- B200 on mit_normal_gpu would start at once by preempting preemptable jobs, but B200 runs are
  unranked.
- QUESTIONS.md asks whether track A's H200 record seeds may also run on mit_preemptable.

To keep GPUs busy without holding up D 003, the four H100 workers split:
- `main` gets r11's two seed 0s (D 002's RL at lr 1e-5 and 3e-5, paired with r4's dm26 seed 0 at
  0.152). The next two H100 workers to finish an item take them: 26786843 at ~02:45 and 26803673
  at ~04:00.
- The other two exit: 26786844 now (r9), and 26786872 at ~04:40 after r10 seed 1. Their H200
  successors (26824148, then 26824150) then wait for H200 capacity, and can start D 003 at once
  when it frees.
- The other two H200 successors follow when their H100 workers run out of items.
- `h100.txt` keeps 13 items, including r10 seed 2 and dm51 seed 2. If H200 stays this far off, I
  give more of them to the H100 workers.

## 2026-10-25 (05:18): r10 seed 1 searched, peaked at 0.315 and slid back to 0.180, so r10 stops at two seeds; dm51 seed 1 ends at 0.178

Both finished at 05:00 on the H100 (node2905):

```
  a01_fp32_lr1e-6_skip_cos_advnone   s0 done  290m final 0.340 best 0.378 t.10   53 spm   2.1 ent 0.03 inf 26.7 ok        | 0.062 0.200 0.275 0.352 0.317
  a01_fp32_lr1e-6_skip_cos_advnone   s1 done  290m final 0.180 best 0.315 t.10   44 spm   9.2 ent 0.04 inf 14.6 ok        |   -   0.115 0.237 0.285 0.130
= a01_fp32_lr1e-6_skip_cos_advnone     n=2 final 0.260 ± 0.113  ok/1-guess/div 2/0/0  median t.10 49m
  a01_fp32_lr1e-6_skip_cos           s0 done  290m final 0.300 best 0.300 t.10   21 spm   6.4 ent 0.08 inf  9.1 one-guess | 0.115 0.170 0.242 0.288 0.287
  a01_fp32_lr1e-6_skip_cos           s1 done  290m final 0.270 best 0.277 t.10   21 spm  39.3 ent 0.05 inf  4.6 one-guess | 0.133 0.187 0.220 0.245 0.262
= a01_fp32_lr1e-6_skip_cos             n=5 final 0.324 ± 0.085  ok/1-guess/div 1/4/0  median t.10 21m
  dm51_fp32_lr3e-6_skip              s0 done  234m final 0.155 best 0.167 t.10   45 spm  42.0 ent 0.07 inf  5.2 one-guess | 0.057 0.110 0.128 0.147 0.155
  dm51_fp32_lr3e-6_skip              s1 done  234m final 0.178 best 0.190 t.10   35 spm  73.4 ent 0.04 inf  3.4 one-guess | 0.098 0.098 0.125 0.137 0.178
= dm51_fp32_lr3e-6_skip                n=2 final 0.167 ± 0.016  ok/1-guess/div 0/2/0  median t.10 40m
  dm26_fp32_lr3e-6_skip              s0 done  234m final 0.152 best 0.162 t.10   34 spm  73.2 ent 0.06 inf  5.8 one-guess | 0.098 0.118 0.128 0.145 0.152
  dm26_fp32_lr3e-6_skip              s1 done  234m final 0.143 best 0.147 t.10   80 spm  79.9 ent 0.04 inf  3.1 one-guess | 0.087 0.085 0.137 0.138 0.143
  dm26_fp32_lr3e-6_skip              s2 done  234m final 0.153 best 0.153 t.10   54 spm  44.7 ent 0.06 inf  4.8 one-guess | 0.082 0.103 0.120 0.137 0.153
= dm26_fp32_lr3e-6_skip                n=3 final 0.149 ± 0.005  ok/1-guess/div 0/3/0  median t.10 54m
```

(A 002's seeds 2-4 are left out of the table; they are unchanged from LOG 10-23.)

r10 seed 1 (`a01_fp32_lr1e-6_skip_cos_advnone`: A 002 with mean-centred advantages), by 50-step
window:

| steps | min | training reward | truncated | informative groups | entropy | sampling s/step | dev at window end |
|---|---|---|---|---|---|---|---|
| 1-50 | 32 | -0.007 | 0.340 | 54 | 1.38 | 11.1 | 0.047 |
| 101-150 | 61 | 0.115 | 0.027 | 30 | 0.41 | 8.1 | 0.168 |
| 151-200 | 105 | 0.220 | 0.152 | 54 | 0.16 | 16.5 | 0.237 |
| 251-300 | 157 | 0.329 | 0.049 | 37 | 0.09 | 10.4 | 0.315 |
| 351-400 | 194 | 0.272 | 0.035 | 25 | 0.03 | 9.7 | 0.240 |
| 451-500 | 229 | 0.142 | 0.016 | 21 | 0.05 | 5.8 | 0.130 |
| 651-700 | 255 | 0.151 | 0.003 | 15 | 0.05 | 3.6 | 0.143 |
| 951-1000 | 288 | 0.174 | 0.002 | 14 | 0.04 | 3.7 | 0.178 |

The final eval (step 1017, 290 min) is 0.180.

What it means:
- Seed 1 switched to searching at steps 150-200, as seed 0 did (54 informative groups, 16.5 s of
  sampling per step), and reached 0.315 at step 300 (157 min). That is above A 002 seed 1's final
  0.270.
- Then it slid back. Over steps 300-500 (157-229 min, lr 4.9e-7 down to 1.9e-7), the training
  reward fell from 0.33 to 0.14 and sampling fell from 10 s to 4-6 s per step. The searches got
  shorter and stopped finding solutions, on the policy's own training prompts. It stayed in that
  short-output mode for the last hour (0.13-0.18).
- This is not the reward-0 runs' collapse (LOG 10-24 09:40): truncation stayed under 0.14
  throughout, and entropy was already low (0.03-0.09) before the slide.
- `summarize_screens.py` labels the seed "ok" (14.6 informative groups), so it is not "collapsed"
  under PLAN's definition. But it ends below A 001's dev mean (0.240).
- Against A 002 on the same seeds: +0.040 (seed 0) and -0.090 (seed 1), mean -0.025. The 3-seed
  form is out (seed 1 is below 0.2703). The 5-seed form would need seeds 2-4 to average 0.277 with
  all three above 0.240. Even then, r10 would be one more recipe tied with A 002, which is already
  promoted. So r10 stops at two seeds; seed 2 goes to `later`.
- r12 (r10's config on the H200) stays first in `queue/h200.txt` only as a fallback, so the D 003
  workers find work there if I'm not around. I'll replace it with a better H200 screen before the
  D 003 seeds are all taken (not before 10-26 ~04:00, even if all four H200 workers start
  tonight).

dm51 seed 1 (D 002's RL from r2's `d00_dm51` pretrained seed 1, with the DeepMind arithmetic at 51 %
of the pretraining tokens):
- It ended at 0.178 (best 0.190 at step 15,500), against dm26 seed 1's 0.143.
- Paired with dm26: +0.003 (seed 0) and +0.035 (seed 1), mean +0.019. Two seeds can't separate that
  from noise.
- dm51 can't promote on its own. Its seed 0 (0.155) is below D 002's dev mean (0.160), which rules
  out the 3-seed form. The 5-seed form would need seeds 2-4 to average 0.206 (H100 D line: 0.190).
- The data share still matters for later track D records on the H200, which reads more tokens in
  the same 112 min. Seed 2 (RL only, ~4.2 h) is second in `main`.

r11 seed 0 at lr 1e-5 (D 002's RL, paired with dm26 seed 0) is at 0.105 after 156 min. dm26 seed 0
was at 0.128 at 120 min and 0.145 at 180. It ends ~06:40.

Workers and queues:
- 26786872 found `main` empty at 05:00 and exited. Its H200 successor 26824150 now waits for a GPU
  beside 26824148.
- 26803673 took r11's lr 3e-5 seed 0 at 05:00 (ends ~09:05).
- At 05:15 Slurm estimated a new mit_preemptable job at 21:20-22:00 tonight: H200, H100 and B200
  alike. So the remaining H100 workers keep running H100 screens instead of exiting.
- `main` now holds:
  - `calibration/research_r3/c90_pos01` seed 0. This is the W-REINFORCE collapse screen (positive
    advantages x 0.1, 90 min); its 10-21 attempts failed on node1800 and it never ran. It is
    comparable with the other r3 screens: c90_mal0 0.322 / 0.383 searching, c90_t12 0.212 and
    c90_cc1e-3 0.162, both one-guess.
  - dm51 seed 2.
- 26786843 takes c90_pos01 at ~06:40 and dm51 seed 2 after it. I'll decide what 26803673 gets
  before ~09:05.
- `h100.txt` keeps 8 items, which no worker reads.
- mit_normal_gpu: Slurm plans the A record chain heads for 10-27 20:30 and 21:00 (at 04:29).
- GPU cap: 2 normal (pending), 2 H100 running and 2 H200 pending make 6. 26824147 and 26824149
  wait on the H100 workers.

## 2026-10-25 (09:19): r11's higher learning rates and W-REINFORCE both lose; r13's KL holds the policy at the base; D 003 seeds 0-1 run on the H200

```
  dm26_fp32_lr1e-5_skip              s0 done  234m final 0.142 best 0.152 t.10   95 spm  51.3 ent 0.06 inf  3.0 one-guess | 0.040 0.082 0.100 0.132 0.142
  dm26_fp32_lr3e-5_skip              s0 done  234m final 0.068 best 0.098 t.10    - spm  51.9 ent 0.05 inf  3.8 one-guess | 0.082 0.080 0.070 0.048 0.068
  dm26_fp32_lr3e-6_skip              s0 done  234m final 0.152 best 0.162 t.10   34 spm  73.2 ent 0.06 inf  5.8 one-guess | 0.098 0.118 0.128 0.145 0.152
  c90_pos01                          s0 done   90m final 0.153 best 0.162 t.10   35 spm  41.1 ent 0.03 inf  2.4 one-guess | 0.055 0.142 0.153 0.153 0.153
  c90_t12                            s0 done   90m final 0.212 best 0.218 t.10   19 spm  46.1 ent 0.05 inf  4.0 one-guess | 0.170 0.203 0.212 0.212 0.212
  c90_cc1e-3                         s0 done   90m final 0.162 best 0.182 t.10   44 spm   7.6 ent 2.25 inf  7.9 one-guess |   -   0.110 0.162 0.162 0.162
= c90_mal0                             n=2 final 0.353 ± 0.044  ok/1-guess/div 0/0/2  median t.10 44m
```

r11 (D 002's RL at higher learning rates, from r2's dm26 pretrained seed 0):
- lr 1e-5 ended at 0.142 and lr 3e-5 at 0.068, against 0.152 at D 002's lr 3e-6 on the same
  pretrained model.
- Both took off later: 95 min to reach 0.10 at 1e-5, never at 3e-5, against 34 min at 3e-6.
- r11 is closed. Its seeds 1-2 go to `later`; track D keeps lr 3e-6.

c90_pos01 (A 001 + fp32 at lr 1e-6, positive advantages x 0.1, the W-REINFORCE screen):
- One-guess by 60 min, with 2.4 informative groups at the end.
- At 0.153 after 90 min it is the lowest of the r3 screens, beside Clip-Cov's 0.162 and
  temperature 1.2's 0.212.
- Damping the push toward sampled solutions didn't keep the policy searching here. W-REINFORCE is
  closed for track A, and c90_pos01's other seeds stay in `later`.

r13 seed 1 (r10 + KL to the base model at 0.05, skipping off; started 08:06 on node2700) has barely
moved from the base model in 80 steps:

| step | r10 seed 1: reward / truncated / entropy | r13 seed 1: reward / truncated / entropy / KL |
|---|---|---|
| 1 | -0.060 / 0.71 / 1.96 | -0.062 / 0.72 / 1.95 / 0.0012 |
| 20 | -0.016 / 0.42 / 1.59 | -0.032 / 0.55 / 1.70 / 0.0017 |
| 40 | +0.033 / 0.10 / 1.08 | -0.031 / 0.53 / 1.84 / 0.0018 |
| 80 | +0.067 / 0.02 / 0.70 | -0.007 / 0.48 / 1.72 / 0.0020 |

Its dev at step 50 was 0.010, with 92 % of outputs truncated (r10 seed 1: 0.047 with 29 %).
- Why: under mean-centred advantages, a group with no solve (most groups early on) gives
  advantages of about +-0.05, from the -0.1 for truncation. The KL pull per token is 0.05 times
  the log-prob change. So once a token's probability has moved by about one nat, the pull matches
  the early learning signal, and the policy stays near the base model.
- The KL of 0.0012 at step 1, before any update, is the gap between the fp32 policy (bf16
  autocast) and the pure-bf16 reference, not training.
- So a constant KL weight doesn't suit mean-centred advantages: 0.05 freezes the start, while a
  weight small enough for the start (~0.005) would be far too weak against the late slide.
- r13 runs ~48 s per step, against r10's ~17 s average over its run. Most of that is outputs
  staying long (half of them truncated); KL and the 4,096-token micro-batches add the rest.
- It will get ~350 steps in 290 min, so it can't reach the steps where r10 seed 1 slid (300-500),
  and it can't answer the question it was queued for. I let it finish anyway: stopping it would
  give the H100 back to the cluster, and Slurm estimates a new H100 or H200 job at 00:51
  tomorrow.

Records on the H200:
- D 003 seed 0 started at 08:09 (node4900) and seed 1 at 08:43 (node5001). These are the first
  H200 record seeds.
- Pretraining runs at ~385k tokens/s, about D 000's 359-373k on the H100. This 110M model's
  pretraining is not limited by memory bandwidth, so the H200 adds little there.
- Seeds 2-3 follow on the same two workers at ~14:30-15:05. Seed 4 starts ~20:50.
- The second worker then turns to `queue/h200.txt` at ~21:25. Its first item, r12, has to be
  decided by then, not by 10-26 04:00 as I wrote at 05:18.
- Slurm now plans the mit_normal_gpu chain heads (A 004, A 002) for 10-27 16:00.

H100 plan:
- r10 seed 2 moved from `later` to `main`. One H100 worker takes it after dm51 seed 2 (~13:20) or
  r13 (~13:40).
- By ~16:00 it shows whether r10's mean-centred recipe searches again and holds past step 300.
- If it slides like seed 1, r12 comes out of `h200` before 21:25. The fallback is then the next
  record seed or an empty queue.
- GPU cap: 4 preemptable running (2 H200, 2 H100) and 2 mit_normal_gpu pending make 6.
  26824147 and 26824149 (H200) wait on the H100 workers.

## 2026-10-25 (10:39): A 002's seeds split at steps 100-200; sampler/trainer mismatch logging and truncated importance sampling added; r14 (r5 + TIS) queued on the H100

Why A 002's seeds end so far apart (0.263-0.315, and 0.472 for seed 2), by 100-step window of the
screen seeds' train logs (sampling s/step tracks the output length):

| seed | steps | min | reward | truncated | informative groups | entropy | sampling s/step | dev at window end |
|---|---|---|---|---|---|---|---|---|
| 2 | 1-100 | 20 | 0.032 | 0.115 | 28.7 | 0.89 | 4.7 | 0.072 |
| 2 | 101-200 | 30 | 0.093 | 0.001 | 17.9 | 0.33 | 2.8 | 0.112 |
| 2 | 501-600 | 82 | 0.281 | 0.001 | 14.6 | 0.033 | 4.6 | 0.250 |
| 2 | 1001-1100 | 137 | 0.402 | 0.000 | 10.7 | 0.019 | 5.1 | 0.427 |
| 2 | 1901-1984 | 290 | 0.480 | 0.001 | 10.8 | 0.056 | 7.2 | 0.472 |
| 1 | 1-100 | 21 | 0.033 | 0.120 | 28.6 | 0.73 | 4.9 | 0.103 |
| 1 | 101-200 | 25 | 0.125 | 0.000 | 10.9 | 0.080 | 1.1 | 0.128 |
| 1 | 2201-2300 | 80 | 0.225 | 0.000 | 2.2 | 0.032 | 0.9 | 0.202 |

- Every seed starts the same way: long rambling outputs, 12 % truncated, and ~29 informative groups.
  Most of those are unsolved groups where the only contrast is truncated (-0.1) against not (0),
  which the per-group z-score scales up to full size.
- By step 200 every seed has stopped truncating. Seeds 0, 1, 3 and 4 overshoot to immediate answers
  (sampling ~1 s/step, entropy 0.05) and then creep up slowly. Seed 2 kept short searches (3-7 s/step)
  and climbed steadily for 2,000 steps.
- So the early one-guess collapse is the z-score's push against truncation. Arms that weaken it
  keep searching: mal0 (r5, r7, r8: every seed searched) and mean-centred advantages (r10: both
  seeds). But each long-search run declined later, after entropy fell to ~0.03: r5 (3 of 3
  collapsed at steps 375-525), r8 seed 2 (0.415 to 0.315) and r10 seed 1 (0.315 to 0.180). Seed 2's
  shorter searches didn't, at the same entropy.

A cause I set aside on 10-24 (09:40) fits this pattern: the mismatch between vLLM's sampling
probabilities and the training model's. RL reports (Yao et al. 2025; Liu et al. 2025) find it grows
as entropy falls and with output length, and that collapses follow it. My argument then was that
A 002 seed 2 reached entropy 0.017 without collapsing; but its outputs were short. Nothing measured
the mismatch so far, so I added the measurement and the standard correction (commit 92238ab, off by
default):
- `log_mismatch`: vLLM also returns each sampled token's log-prob after temperature and the answer
  constraint (`logprobs_mode="processed_logprobs"`). The train log gets, over the updated rows:
  `mismatch_k1` and `mismatch_k3` (two estimates of KL(sampler || trainer) per token),
  `mismatch_far` (share of tokens whose ratio pi_train / pi_sampler is outside [1/2, 2]),
  `mismatch_seq` (mean |sequence log-ratio|) and `mismatch_max`.
- `tis_cap`: truncated importance sampling. Each token's policy-gradient term is weighted by
  min(pi_train / pi_sampler, cap), with the ratio detached. It also turns on the logging.
- Unchanged by default: default runs build the same engine and sampling parameters and take the same
  path through the update. Tests check the weighting (a capped constant ratio scales the gradient
  exactly), the alignment of the sampler's log-probs with the kept groups' tokens, and the
  extraction from vLLM's flat log-prob lists. 130 tests pass. The GPU path is untested, so r14 is
  also its smoke test.

r14 (`calibration/research_r14/a01_fp32_lr1e-6_skip_cos_mal0_tis2`): r5's config plus
`tis_cap: 2.0` and `no_save`. Seeds 0-1 are in `main` after r10 seed 2, on the H100, so they compare
with r5's three H100 seeds:
- r5 seed 0 collapsed at steps 375-400 (~150 min) and seed 1 at step 425.
- If r14 holds past step 550 with dev near r5's 0.38-0.41 peak, the next step is an H200 screen of
  it, with the A 002 / A 003 record seeds as the H200 reference.
- If r14 collapses like r5 while `mismatch_k3` stays tiny, the mismatch idea is out.

The H100 workers now take r10 seed 2 and r14 seed 0 at ~13:20 instead of exiting. Slurm's estimate
for a new mit_preemptable job (H200 or H100) was 10-26 00:51 at 09:15, so their H200 successors
would wait about that long anyway. The cap is unchanged: 2 normal (pending) + 4 preemptable.

## 2026-10-25 (11:46): track A's H200 record seeds also run on preemptable H200 workers (INBOX 11:15)

The laptop answered QUESTIONS 10-25 02:28: yes. mit_preemptable is the same cluster, so the same
provider. The mit_normal_gpu chains stay, and all workers share the record queues under flock. A
preempted seed is an infrastructure stop: its partial attempt goes to `extra/` with a note, and the
same seed reruns.

Routing:
- The two running H200 workers (26824148, 26824150) read `record-003_dm_math_fp32_h200:h200`, which
  was fixed when they were submitted. So A 004's and A 002's seeds 0-1 moved to the front of `h200`,
  interleaved (A 004 0, A 002 0, A 004 1, A 002 1), ahead of r12's three seeds. After D 003's seeds,
  these workers take them at about 21:25 (26824150), then about 02:45, 03:10 and 08:05.
- The two pending H200 successors were replaced, because a pending job's queue list can't be
  changed:
  - 26824147 became 26854533 (`afterany` 26786843). It reads `record-003`, then A 004's queue, then
    A 002's, then `h200`.
  - 26824149 became 26854534 (`afterany` 26803673). It reads the same, with A 002 before A 004.
  - Both use RESERVE_MIN 380, `--time-min=06:30:00`, `REQUIRE_GPU="NVIDIA H200"` and the NVL nodes
    excluded, like the D 003 workers.
- The mit_normal_gpu chains keep reading the A record queues, which now hold seeds 2-4 of each.
- While those queues still hold seeds, each wake keeps at least two A seeds at the front of `h200`,
  so the running workers don't reach r12 first.

Start estimates at 11:45 (`sbatch --test-only`):
- a new mit_preemptable H200 job: 10-26 05:46;
- B200 or H100: 10-26 04:54;
- the chain heads (Slurm's plan): 10-27 10:00.

If nothing is preempted, the two running workers alone can finish about 11 A seeds before their
48-h limits (10-27 08:09 and 08:43). With the successors, both A records could land around 10-26
night, instead of about 11-01 on mit_normal_gpu alone. The GPU cap is unchanged: 2 normal (pending)
plus 4 preemptable.

D 003 at 11:45:
- Seed 0 (node4900): 90 of its 234 RL minutes, dev 0.120 at step 3000.
- Seed 1 (node5001): 64 RL minutes, dev 0.130 at step 4000.

## 2026-10-25 (12:50): preemptions: D 003 seed 1 stopped 88 RL minutes in and reruns; r13 seed 1 stopped (closed arm); r14 seed 0 moved to the front of `main`

Between 11:48 and 12:43, mit_preemptable preempted three of my four running workers, two of them
more than once:
- 26824150 (D 003 seed 1, node5001) at 12:10, 88.5 of its 234 RL minutes in (in-loop dev 0.137 at
  step 6000).
  - The attempt is in `records/v3/track_d/003_dm_math_fp32_h200/extra/interrupted_1_1`, with a note
    (moved there by hand, as the worker would on restart).
  - When 26824150 restarts, seed 1 reruns from the start (~6.2 h), as RULES_v3 allows for an
    infrastructure interruption. Its pretraining rewrites the pretrained model at the same pool
    path (`pretrain_scratch.py` creates the directory if needed and overwrites the files).
- 26786843 (r13 seed 1): at 11:48 on node2700, then again after restarts on node4109 and node1703
  (12:43). On both nodes the restart spent its first 150 s on a CUDA check that timed out, as on
  node3209 below.
- 26803673 (dm51 seed 2): on node2905, then node3209. It resumed on node2700 at 12:24 from its hourly
  save and was at step 14083 (199 RL minutes) at 12:47.
- 26824148 (D 003 seed 0, node4900) wasn't preempted.

r13 seed 1 stopped by hand:
- Its arm is closed: KL 0.05 holds the policy at the base (dev 0.010, 0.002, 0.005, 0.003, 0.002 at
  steps 50-250; LOG 09:19).
- It would have resumed from the 11:11 save and needed about 2 more H100-hours.
- I emptied its worker's in-hand file, so 26786843 takes the next `main` item when it restarts. Its
  logs through step 265 (217 min) are in the seed dir. I removed its 20 GB resume state.

`main` is now r14 seed 0, r10 seed 2, r14 seed 1:
- r14 seed 0 goes first because it is the first GPU run of the mismatch code, and it asks the main
  open track A question.
- r10 seed 2 decides r12, which can't run before the A record seeds anyway.

GPU count:
- running: 26824148 (H200), 26803673 (H100);
- requeued and pending: 26824150 (H200), 26786843 (H100);
- pending: the 2 mit_normal_gpu chain heads.

That makes 6. The H200 successors wait on the H100 workers. Until 26824150 restarts, 26824148
alone takes D 003's seeds 2-4 and then the A seeds at the front of `h200`.

## 2026-10-25 (14:30): dm51 closed at +0.010 over dm26; node4900 halves track D's RL speed (excluded); box 1 (8x B200 tomorrow) gets r15, eight step-capped track A arms

r4 dm51 seed 2 ended at 0.145:

```
  dm51_fp32_lr3e-6_skip              s0 done  234m final 0.155 best 0.167 t.10   45 spm  42.0 ent 0.07 inf  5.2 one-guess | 0.057 0.110 0.128 0.147 0.155
  dm51_fp32_lr3e-6_skip              s1 done  234m final 0.178 best 0.190 t.10   35 spm  73.4 ent 0.04 inf  3.4 one-guess | 0.098 0.098 0.125 0.137 0.178
  dm51_fp32_lr3e-6_skip              s2 done  234m final 0.145 best 0.145 t.10  129 spm  67.7 ent 0.06 inf  2.7 one-guess | 0.073 0.082 0.080 0.142 0.145
= dm51_fp32_lr3e-6_skip                n=3 final 0.159 ± 0.017  ok/1-guess/div 0/3/0  median t.10 45m
= dm26_fp32_lr3e-6_skip                n=3 final 0.149 ± 0.005  ok/1-guess/div 0/3/0  median t.10 54m
```

- Paired with dm26 on the same seeds: +0.003, +0.035, -0.008, so +0.010 on the mean. That is within the
  seed noise, and 0.159 is far from the D line (0.190). Doubling the DeepMind share from 26 % to 51 %
  doesn't change the regime. dm51 is closed.

node4900 (D 003 seed 0's node) is CPU-bound, not GPU-bound:
- Its pretraining read 2.69 B tokens in 112 min, the same as node5001 (2.67 B). Its RL ran 1.68 s per
  step (sampling 1.38 s) against node5001's 0.83 s (0.65 s): half speed, all of it in vLLM's sampling
  of this small model, which is bound by the CPU.
- `scontrol show node`: all 120 CPUs allocated to other jobs, load 74 (node5001: 46 allocated, load 13).
- A track A seed would lose there too: its steps spend hundreds of decoding iterations per step on
  the same CPU-side scheduling.
- D 003 seed 0 counts as it ran (in-loop dev 0.165 at step 8,000, 224 min). The node is now excluded
  from all 13 pending workers (`scontrol update ... ExcNodeList`).
- Worker 26824148 runs on node4900 and would have taken D 003 seed 2 next. I set its two queue files
  aside (`record-003_dm_math_fp32_h200`, `h200`) so it exits when seed 0 ends, then put them back. No
  other running worker reads them; 26824150 reruns seed 1 first when it restarts.
  - It finished seed 0 at 14:20 and exited on the empty queues; the files went back at 14:22.
  - Replacement: 26862717 (mit_preemptable H200, node4900 and the NVL nodes excluded,
    `--time-min=06:30:00`, RESERVE_MIN 380). It reads `record-003`, A 004's queue, A 002's, then
    `h200`. Slurm estimates its start at 10-26 07:45, about when the other H200 workers' estimates
    fall. The cap stays at 6: 2 H100 running, 2 H200 and 2 mit_normal_gpu pending.
- D 003 seed 0 scores **0.1590 on the test ladder** (pass@16 0.203; L1-L6 0.566 0.109 0.064 0.035
  0.005 0.176), against D 002's seed 0 at 0.1496. In-loop final dev 0.168 after 8,365 RL steps: half
  of a normal node's ~17,000, yet as high as D 002's seeds. Track D's RL had flattened by then
  (dev 0.148-0.165 from step 5,500 on).

Box 1 (8x B200, 2026-10-26 11:30Z to 10-27 11:30Z). The maintainer wants experiments, not records,
two or three runs per GPU, and step caps (INBOX 18:00):
- **`budget_steps`** (commit afd8b9f, off by default): the run stops after that many steps and reads
  step s as (s - 1) / budget_steps of `time_budget_s` for the lr schedule, so it follows the same
  schedule and stops on the same step on any GPU or GPU share. `clock_s` in the logs is that clock, and
  `summarize_screens.py` uses it for these runs. With 800 steps over the 17,400-s budget, the lr at
  step 300 is 7.24e-7, against 6.5-7.3e-7 at r5's step 300 (107-125 of 290 min): the schedule of r5's
  first hours.
- **One fixed step cap per track doesn't work for track A.** A 002's H100 seeds did 1,984 to 10,610
  steps in the same 290 min, depending on whether they searched (~30 s per step) or guessed (~1-3 s).
  So the cap is set per experiment. r15 is about long searches: the H100 runs that searched to the end
  did 570 (r8 seed 2) and 669 (r6 seed 0) steps in 290 min, and an H200 is faster, so 800 steps, past
  the 250-550 window where r5, r7 and r8 seeds collapsed.
- **Packing**: two runs per B200. An fp32 track A trainer needs ~51 GB on the H100 (79 GB in use from
  step 5 on, with vLLM's 28 GB), so three can't fit in a B200's
  ~180 GB with any useful vLLM cache. Two fit with `vllm_gpu_memory` 0.2 (36 GB each) and
  `micro_batch_tokens` 4096 (smaller logit buffers; the update is the same). An Engaging B200 can't
  start before the block (`sbatch --test-only`: 10-26 06:37 preemptable, 15:36 mit_normal_gpu), so
  `scripts/research/pack_probe.sh` runs on one box GPU at the start (1 vs 2 track A runs from r7's
  searching weights, 1 vs 4 track D runs), while 14 workers start on the other seven.
- **`malformed_reward_solved`** (commit b66cf05, off by default): the training reward for a malformed
  (truncated) output in a group with a correct sample. With reward 0 elsewhere, a truncated search
  then scores below a finished wrong one only where the policy can solve the puzzle. In a group with 3
  correct, 4 truncated and 1 wrong, the z-scored advantages of the truncated and the wrong samples
  are -0.90 and -0.18 at -0.5 (-0.84 and -0.49 at -0.2), against -0.77 for both under r5. Unsolved
  groups stay all-zero, so nothing pushes them toward one quick guess, which the -0.1 penalty does
  (LOG 10-25 10:39).

r15 (`calibration/research_r15`, `queue/b200.txt`, seeds 0-3 of eight arms, 32 items): r5's recipe
(A 001 + fp32 lr 1e-6 + skip + malformed reward 0) with the 800-step cap, and fixes for its collapses:

| arm | change | hypothesis it tests |
|---|---|---|
| `mal0_st800` | none (control) | r5's three H100 seeds collapsed at steps 350-550 |
| `mal0_st800_trs05` | truncation -0.5 in solved groups | the slide into truncation (r5 seed 0) |
| `mal0_st800_trs02` | truncation -0.2 in solved groups | the same, milder |
| `mal0_st800_tis2` | truncated importance sampling, cap 2 | sampler/trainer mismatch (r14 tests it on the H100) |
| `mal0_st800_lr5e-7` | lr 5e-7 | step size once the policy is sharp |
| `advnone_st800` | Dr. GRPO with the -0.1 penalty | r10's recipe under the same cap |
| `mal0_st800_t11` | training temperature 1.1 | the entropy fall (0.03-0.04) before each collapse |
| `mal0_st800_kl05` | KL 0.05 to the base (skipping off) | drift once the policy is sharp |

- Seed 0 and 1 of every arm come first (16 runs, one per worker), then seeds 2-3. The comparison is
  within the batch (same GPU type and packing). A winner gets an H200 screen before any record.
- Weights are not saved (`no_save`); the box disk goes with the block.
- `scripts/aws/box_sync.sh` copies the code, data, configs, queue and base model to the box (`to`)
  and brings the logs back (`from`) every wake. `box_workers.sh` takes `GPUS` and `FIRST` to start
  workers on some GPUs only.

## 2026-10-25 (15:45): box 1 pre-checks; r14 seed 0 and r10 seed 2 are searching at steps 219-243

Nothing finished since 14:30. Both H100 screens are in the searching regime, ahead of the windows
where their recipes slid before:

| seed | steps (min) | dev at step 200 | training reward | truncated | informative groups | entropy |
|---|---|---|---|---|---|---|
| r14 seed 0 (r5 + TIS cap 2) | 226-250 (133) | 0.253 | 0.32 | 0.54 | 26 | 0.15 |
| r10 seed 2 (A 002 + mean-centred advantages) | 226-233 (119) | 0.263 | 0.30 | 0.09 | 47 | 0.15 |

- r14 seed 0 has truncated about half its outputs since step 150 (reward 0 for those under mal0),
  with sampling at ~33 s per step, as r5's seeds did before they collapsed (steps 375-525).
- Its sampler/trainer mismatch is small and falling: `mismatch_k3` per token went 0.0011 (steps
  1-100) to 0.00055 (steps 226-250) while entropy fell from 1.8 to 0.15. Almost no token's ratio
  is outside [1/2, 2] (`mismatch_far` 1e-5), and the cap of 2 has clipped nothing. So far the
  mismatch doesn't grow as entropy falls in this run, so TIS has nothing to correct yet.
- r14's answer comes at ~17:30-19:00. Both runs end at ~18:30.

Box 1 (8x B200, 10-26 11:30Z), checks at 15:40-15:55:
- AWS (read-only calls): capacity block `cr-03fec7969fbda8633` is `scheduled` for
  10-26 11:30Z to 10-27 11:30Z (p6-b200.48xlarge, us-east-1d). Its three EventBridge launch
  schedules (+1, +10, +25 min) are enabled. The launch template uses ENI `eni-03f79eb9d98654069`
  (100.29.118.132) and a 2 TB gp3 root volume. The security group allows ssh from anywhere.
- mit_normal CPU jobs are estimated to start on 10-28, so the copy can't run there tomorrow.
  mit_quicktest starts at once (15-min limit; the copy is ~10.6 GB: 6.9 GB for r7's probe weights,
  3.5 GB of Qwen3-1.7B-Base, 0.2 GB for D 002's pretrained seed 0). A quicktest job on node1619
  egressed as 18.13.54.83, inside the box's allowed range. `mit_data_transfer` (dtn001-002) is
  another option, but I haven't checked its egress.
- New `scripts/aws/box_ssh.sh`: ssh to a box with the agent's key in BatchMode, keeping the box's
  host key in `agent/known_hosts_<ip>`. A key that an earlier instance left for the same address
  (a TEST launch, or box 4, which reuses box 1's address) can't block the login then.
  `box_sync.sh` uses the same file.
- vLLM's cu130 wheels JIT-compile kernels with nvcc. `box_workers.sh` points CUDA_HOME at
  `/usr/local/cuda`, the AMI's toolkit, whose version I'll check before starting runs.
- The start is staged so a broken setup costs two items, not fourteen. First the probe on GPU 7
  and two workers on GPU 0. Once those runs log their first steps, two workers each on GPUs 1-6.
  GPU 7 gets its two workers when the probe ends.

## 2026-10-25 (18:46): r14 seed 0 doesn't collapse (dev 0.398), but its importance sampling did nothing and its slow sampling had annealed the lr; r10 closes at 0.261; r16/st500 tests the lr path by step

```
  a01_fp32_lr1e-6_skip_cos_mal0_tis2 s0 done  290m final 0.398 best 0.410 t.10   53 spm   1.3 ent 0.11 inf 26.4 ok        | 0.068 0.150 0.253 0.343 0.358
  a01_fp32_lr1e-6_skip_cos_advnone   s0 done  290m final 0.340 best 0.378 t.10   53 spm   2.1 ent 0.03 inf 26.7 ok        | 0.062 0.200 0.275 0.352 0.317
  a01_fp32_lr1e-6_skip_cos_advnone   s1 done  290m final 0.180 best 0.315 t.10   44 spm   9.2 ent 0.04 inf 14.6 ok        |   -   0.115 0.237 0.285 0.130
  a01_fp32_lr1e-6_skip_cos_advnone   s2 done  291m final 0.263 best 0.328 t.10   66 spm   1.7 ent 0.04 inf 46.5 ok        | 0.060 0.093 0.263 0.328 0.295
= a01_fp32_lr1e-6_skip_cos_advnone     n=3 final 0.261 ± 0.080  ok/1-guess/div 3/0/0  median t.10 53m
= a01_fp32_lr1e-6_skip_cos_mal0 (r5)   n=3 final 0.087 ± 0.025  ok/1-guess/div 2/1/0  (bests 0.392, 0.378, 0.407)
```

r14 seed 0 (r5 + truncated importance sampling, cap 2) ended at **0.398 (best 0.410 at step 450)**, the
best final dev of any track A screen, and it still searched at the end (training reward 0.40,
entropy 0.11, 26 informative groups, a third of its outputs truncated). r5's three seeds, on the
same H100 node (node2700) for two of them, collapsed from ~0.39-0.41:

| 50-step window | r5 s0 | r5 s1 | r5 s2 | r14 s0 |
|---|---|---|---|---|
| steps 251-300: minutes, lr at step 300, dev | 117 min, 6.9e-7, 0.380 | 123, 6.6e-7, 0.355 | 105, 7.4e-7, 0.373 | 172, 4.3e-7, 0.343 |
| 351-400 | dev 0.043 (collapsed) | 0.333 | 0.392 | 243 min, lr 1.6e-7, 0.393 |
| 401-450 | 0.013 | 0.025 (collapsed) | 0.407 | 280 min, lr 1.0e-7, 0.410 |
| 501-550 | 0.025 | 0.050 | 0.033 (collapsed) | (ended at step 463) |

Why it held is not the importance sampling:
- The correction had nothing to correct. Per token, `mismatch_k3` stayed at 0.0005-0.0006 from step
  200 to the end (a typical |log ratio| of ~0.035), and the cap clipped ~1e-5 of tokens. The weights
  were 1 ± a few percent, so the gradient was r5's.
- The mismatch didn't grow as entropy fell (0.0011 at steps 1-100, 0.00058 at 401-463).
- **Sampling ran 2.2x slower** with the sampler's log-probs requested (34.7 s per step at steps
  151-200 against r5's 15.8-18.1 s at the same truncation rate, 0.57; tokens/s 8.4k against
  13.9k; the update time is unchanged, 14 s). r14 seed 1, which started at 18:22, shows the same
  (23.7 s against 9.4-12.3 s at steps 1-50). Requesting log-probs from vLLM costs that much here.
- So r14 took ~40 % fewer steps per minute, and its time-based cosine had annealed by the steps
  where r5's seeds broke: lr 2.8e-7 at step 350 and 1.6e-7 at step 400. r5's 10-step training
  reward fell below 0.25 at step 375 (150 min, lr 5.3e-7), step 422 (175 min, 4.1e-7) and step 523
  (195 min, 3.2e-7).

The working hypothesis is now the lr at each step once the policy is sharp: r14 seed 0 crossed steps
375-463 at 1.0-2.3e-7, below the 3.2-5.3e-7 at which r5's seeds collapsed. Seed 2 collapsed at the
lowest lr and the latest step, so the threshold isn't a fixed number. A 003 (stop at step 300)
avoids the window but leaves ~150 of its 290 minutes unused. With one seed this is a lead, not a
result. The tests:
- **r16 `calibration/research_r16/mal0_st500`** (H100): r5's config with the cosine counted over 500
  steps (`budget_steps` 500, no TIS). By step it follows r14 seed 0's lr: 4.1e-7 at step 300, 2.9e-7
  at 350 and 1.9e-7 at 400, against r14's 4.25e-7, 2.82e-7 and 1.60e-7. Seed 0 is in `main`.
  26803673 takes it after r14 seed 1 (~23:20) and should finish ~03:30. This is also the first GPU run
  of `budget_steps`, ahead of box 1.
- **Box 1, `calibration/research_r15/mal0_st500`**: the same cap of 500 steps on r15's settings,
  seeds 0-3. It replaces `advnone_st800` in `queue/b200.txt`, in the same positions, because r10's
  recipe is closed (below).
- r15's `mal0_st800_tis2` stays: its four seeds log the mismatch through the 800-step schedule,
  where the control is expected to collapse. If the mismatch stays flat through a collapse, the
  mismatch idea is out. Its items will take ~1.5-2x the control's time (sampling).
- r15's `mal0_st800_lr5e-7` tests the same idea with a lower peak. `mal0_st500` keeps the peak of
  1e-6 for the search phase and anneals early.
- r14 seed 1 keeps running: a second seed of the same slow-clock condition, with the mismatch logged.

r10 (A 002 + mean-centred advantages) closes at **0.261 ± 0.080** on three seeds, below A 002's 0.324
and below the A line (0.2703). Seed 2 peaked at 0.328 at step 300 and slid to 0.263, as seed 1 did
(0.315 to 0.180), with entropy at 0.04. Its H200 follow-up r12 left `h200`. Its configs stay
unqueued.

Queues: `h200` is A 004 seeds 0-1 and A 002 seeds 0-1 (interleaved). `main` is r16 seed 0.
`b200` has 32 items, with `mal0_st500` where `advnone_st800` was.

GPUs: 26803673 runs r14 seed 1 (H100). Pending: 26854533 (eligible since 26786843 exited at 18:42,
less than 330 min left), 26824150, 26862717 (all H200) and the two mit_normal_gpu chain heads
(estimated 10-27 10:00). 26854534 waits on 26803673. That makes 6.

## 2026-10-25 (23:05): AWS boxes off (INBOX 22:00); r15's 500-step arm runs on Engaging as r16 seeds 0-2

The capacity blocks are under a refund review and must stay unused, so box 1 doesn't run
tomorrow and no AWS spending is allowed (the $500 emergency budget too). A read-only check at 23:00
found no EventBridge schedules and no instances in us-east-1 or us-east-2. `queue/b200.txt`
(r15, 32 items) is parked.

What moves to Engaging: only r15's `mal0_st500`, as r16 (`calibration/research_r16/mal0_st500`,
r5's H100 settings), seeds 0-2 in `main` on the H100 worker 26803673. They follow r14 seed 1, which
ends ~23:45 (step 378 at 22:53, dev 0.368 at step 350, still searching). Each seed takes ~4.2 h, so
the three end at ~04:00, ~08:15 and ~12:30. r16 is step-capped, so this 3-seed screen can gate a
record (PLAN). The bar is every seed at least 0.2703 and none collapsed. The other r15 arms wait on
its result: lr 5e-7 and truncation -0.5 in solved groups come next if r16 fails, and r16 + the
truncation reward if it holds with a third of its outputs truncated, as r14 seed 0 did.

GPUs: `sbatch --test-only` puts a new mit_preemptable H100 or H200 job at 10-26 16:02 and a
mit_normal_gpu H200 job at 22:33. Trading a pending slot (the mit_normal_gpu chain heads,
estimated 10-27 10:00) for another screen worker would therefore not start anything sooner, so the
six GPU jobs stay as they are. 26854534, the H200 successor of the H100 worker, waits until r16
seed 2 ends (~12:30). The other three H200 workers are ahead of it in the queue.

## 2026-10-26 (00:05): three H200 workers started at 23:11-23:36, two were preempted within a minute; D 003 seed 2 runs; r14 seed 1 resumes after a preemption

- 26803673 (H100) was preempted on node2700 at 23:05, with r14 seed 1 at step 396 (229 min, dev
  0.368 at step 350, still searching). It restarted on node2904 at 23:40 and resumes from the
  hourly state saved at 22:18 (`resume/1/state.pt`, 20.6 GB). It redoes the steps since then and should end
  ~02:00. r16 seeds 0-2 follow, now ending ~06:15, ~10:30 and ~14:45.
- 26824150 started D 003 seed 1 on node4300 at 23:11 and was preempted at 23:12. 26854533 started
  D 003 seed 3 there at 23:36 and was preempted at 23:36:48. Both stopped before training, during
  the CUDA check: no seed directory was written, so nothing goes to `extra/`. Each keeps its seed in
  hand (`queue/running/`) and reruns it when it restarts. Both are pending again (reason None).
- 26862717 runs D 003 seed 2 on node4300 since 23:30 (pretraining, then RL; ~6.2 h, ends ~05:45).
- Six GPU jobs: 2 running, 4 pending (26824150, 26854533, the two mit_normal_gpu chain heads), and
  26854534 waits on 26803673.

## 2026-10-26 (03:00): r14 seed 1 holds too (dev 0.403), so r14 closes at 0.401 on two seeds against r5's 0.087 with the same recipe; r16 seed 0 starts

```
  a01_fp32_lr1e-6_skip_cos_mal0      s0 done  290m final 0.113 best 0.392 t.10   50 spm  31.1 ent 0.06 inf  6.1 one-guess | 0.098 0.215 0.380 0.023 0.068
  a01_fp32_lr1e-6_skip_cos_mal0      s1 done  290m final 0.063 best 0.378 t.10   48 spm  11.5 ent 0.26 inf 12.7 ok        | 0.087 0.168 0.322 0.333 0.047
  a01_fp32_lr1e-6_skip_cos_mal0      s2 done  290m final 0.083 best 0.407 t.10   34 spm   8.8 ent 0.11 inf 10.2 ok        | 0.067 0.252 0.373 0.407 0.062
= a01_fp32_lr1e-6_skip_cos_mal0        n=3 final 0.087 ± 0.025  ok/1-guess/div 2/1/0  median t.10 48m
  a01_fp32_lr1e-6_skip_cos_mal0_tis2 s0 done  290m final 0.398 best 0.410 t.10   53 spm   1.3 ent 0.11 inf 26.4 ok        | 0.068 0.150 0.253 0.343 0.358
  a01_fp32_lr1e-6_skip_cos_mal0_tis2 s1 done  290m final 0.403 best 0.403 t.10   57 spm   1.8 ent 0.11 inf 27.0 diverged  | 0.010 0.132 0.250 0.348 0.377
= a01_fp32_lr1e-6_skip_cos_mal0_tis2   n=2 final 0.401 ± 0.004  ok/1-guess/div 1/0/1  median t.10 55m
```

- r14 seed 1 ended at dev 0.403, its best, still rising: 0.348 at step 300, 0.377 at 400, 0.390 at
  500, 0.403 at its last step, 526. It resumed at step 333 after the 23:05 preemption; its rerun of
  steps 334-350 scored 0.350 at step 350, where the first pass had 0.368. Seed 0 ran 463 steps.
- The summarizer flags seed 1 "diverged" because half its outputs are malformed (0.47 in training,
  0.505 in the last eval). That is the mal0 recipe, where a truncated answer scores 0 instead of a
  penalty, and not a collapse: L1 is 0.99, and the pass rate climbed to the end. The malformed half
  is room for the truncation reward (r15's `trs05`) later.
- The sampler/trainer mismatch stayed flat (k3 ~0.00058 per token at step 452), and the clip did
  nothing, as in seed 0.
- So r14 = r5 + per-token logprobs + a cap that never clipped. Its two seeds hold at 0.40, where
  r5's three seeds peaked at 0.38-0.41 and fell to 0.06-0.11. The difference that remains is speed:
  r14 samples 2.2x slower, ran 463-526 steps in 290 min (r5: 1300-3179, most of them fast,
  collapsed steps), and its time-based cosine reached the low lr by step count much earlier. r16
  (`calibration/research_r16/mal0_st500`: r5's config, fast sampling, the cosine over 500 steps)
  tests that directly. r16 seed 0 started at 02:41 on node2904 (H100); seeds 0-2 end ~07:00, ~11:15,
  ~15:30.
- r14 itself can't gate a record: it is time-based and ran on the H100, and its lr path would change
  with the H200's sampling speed. If r16 passes (every seed at least 0.2703, none collapsed), the
  record config is r16's with a hard time stop added.

Elsewhere: 26824150 started D 003 seed 1 on node5100 at 01:10 and was preempted at 01:27, two
minutes into pretraining (`seeds/1` will move to `extra/interrupted_1_2` when it reruns). 26854533
restarted D 003 seed 3 on node4300 at 02:00. 26862717 is in D 003 seed 2's RL (ends ~05:45). GPUs:
3 running, 26824150 and the two mit_normal_gpu chain heads pending: 6.

## 2026-10-26 (06:40): D 003's finished seeds are 124-127 s over the 6-h budget, from slow start-up on the nodes; A records held; r16 runs five seeds across the H100 and H200 workers

D 003 seed 2 finished at 05:42: test 0.1455, dev 0.173 at its last eval. Seed 0 had test 0.1590
and dev 0.168. Both clocks are over budget:

| seed | node | pretrain_s (112 min = 6720) | GRPO wall over its training clock | total_s (budget 21600) |
|---|---|---|---|---|
| 0 | node4900 | 6876 | 810 | 21727 |
| 2 | node4300 | 7206 | 477 | 21724 |
| D 002, H100 (5 seeds) | | 6729-6733 | 114-197 | 20884-20971 |

`validate_record_v3.py` fails a seed over budget ("over budget"), and dropping a seed disqualifies
a record. So D 003 can't pass as it stands. I didn't catch this on seed 0 at 10-25: I logged its
test score but didn't check its `within_budget`.

The excess is start-up: imports, model creation and vLLM initialisation before each trainer's own
clock starts, plus the saves. The trainers' own clocks are fixed (112 min + 14040 s = 20760 s, which
leaves 840 s of margin). On the H100 the start-ups took 2-3.5 min a seed; here they take 16 min.

The pattern points to cold reads of the Python environment (`~/rlvr-venv`, NFS):
- The worker's 150-s CUDA check (`import torch` + one CUDA op) timed out on 15 of the 42 item
  starts since 10-24 23:00: node4109 and node3409 three times each, node3209, node2905 and node2904
  twice each, node5100, node4300 and node1703 once. Other starts passed it, some on the same nodes:
  node4300 at 02:00 and 05:42, node4301, node2700, node2901.
- r16 seed 0 needed 15 min on node2904 from launch to vLLM's first log line. Its restart on node2901
  had vLLM ready in 3.5 min.
- The login node imports torch in 3-6 s.

Track A has less margin: 600 s (18000 against 17400). A 001 used 33-98 s on the H100, so an A
record seed that starts on a cold node would probably be over too.

Actions:
- Held all ten A 004 / A 002 items in `queue/held_records.txt`, which no worker reads. Neither
  record has started a seed, so nothing is lost.
- D 003 seeds 1, 3 and 4 keep running: stopping them would drop runs, and they give the H200 D line
  either way.
- Asked in QUESTIONS (06:40) whether records may use an untimed warm-up before each timed stage. It
  would read the environment into the page cache and import torch, transformers and vLLM in a
  process that sees no GPU. RULES_v3 starts the clock at the first GPU operation, and this does
  none. I also asked how to handle D 003 (report it as failed and rerun D 002's config as D 004, or
  shorten budgets). I'll build the warm-up as an off-by-default option and measure it on screens.
  No record uses it until the answer.
- r16 now runs five seeds instead of three: seed 1 on the H100 worker after seed 0, seeds 2-4 in
  `h200` for the H200 workers as D 003's seeds finish (~08:15, ~10:20, ~11:50). It is step-capped,
  so its schedule doesn't depend on the GPU. The 5-seed promotion form allows one collapsed seed,
  which suits a recipe whose cousin r5 collapsed. All five should end by ~15:30.
- The mit_normal_gpu chain heads read only the two A record queues, which are now empty. If they
  start (Slurm's plan: 10-27 11:00 and 14:00) they exit at once. I'll deal with them before then.

## 2026-10-26 (11:45): r16's first three seeds end 0.232 / 0.405 / 0.412, seed 0 sliding into truncation over its last 50 steps; D 003 fails validation with two seeds over budget; the warm start fixes cold starts on screens; r17 (truncation penalty in solved groups) and r18 (track D RL, 4x prompts) queued

This session was down from 06:50 to 11:30. The restart script's prefix match found `research-watch`, so the dead
agent was never restarted (fixed in a98dfa7). While it was down:
- The two mit_normal_gpu chain heads started on node2434 at 06:51 and 06:53, a day earlier than Slurm's plan. They
  read only the A record queues, which are held empty, so both exited at once.
- 26803673 (H100) finished r16 seeds 0 and 1, and exited on an empty `main` at 10:50.
- 26862717 finished D 003 seed 4 at 11:35 and exited on empty queues, two minutes before r17 was queued.

r16 (`calibration/research_r16/mal0_st500`: r5's recipe with the cosine over 500 steps):

```
  mal0_st500                         s0 done  290m final 0.232 best 0.395 t.10   87 spm   1.7 ent 0.03 inf 18.9 diverged  | 0.050 0.092 0.335 0.355 0.395
  mal0_st500                         s1 done  290m final 0.405 best 0.405 t.10   87 spm   1.7 ent 0.08 inf 22.3 diverged  | 0.033 0.065 0.277 0.345 0.385
  mal0_st500                         s2 done  290m final 0.412 best 0.418 t.10   58 spm   1.7 ent 0.06 inf 21.8 ok        | 0.000 0.105 0.268 0.345 0.397
  mal0_st500                         s3 RUN   162m final 0.300 best 0.300 t.10   87 spm   1.7 ent 0.15 inf 28.8 diverged  | 0.055 0.097 0.270 0.300 0.300
  mal0_st500                         s4 RUN    56m final 0.035 best 0.035 t.10    - spm   1.7 ent 0.80 inf 15.8 diverged  | 0.035 0.035 0.035 0.035 0.035
= mal0_st500                           n=3 final 0.349 ± 0.102  ok/1-guess/div 1/0/2  median t.10 87m
```

(Minutes are the step clock: 500 steps count as the 290-min budget. Seeds 0-1 ran on the H100 and seed 2 on the
H200; the step cap makes the GPU irrelevant to the trajectory.)

- Seeds 1 and 2 held to the end, at 0.405 and 0.412, and their dev was still at its best at step 450-500. That
  matches r14's 0.398 and 0.403, now with fast sampling.
- Seed 0 peaked at 0.395 (step 400) and fell to 0.232 at step 500. It wasn't the lr: the cosine had reached
  1.1-1.0e-7. It was truncation. Seed 0 had truncated 54-64 % of its training outputs since step 300 (seeds 1-2:
  41-47 % and 21-24 %), and in its last 50 steps the share rose while the reward fell:

| steps | lr | training reward | truncated | entropy |
|---|---|---|---|---|
| 401-425 | 1.7e-7 | 0.404 | 0.54 | 0.048 |
| 426-450 | 1.4e-7 | 0.398 | 0.54 | 0.036 |
| 451-475 | 1.1e-7 | 0.337 | 0.59 | 0.031 |
| 476-500 | 1.0e-7 | 0.263 | 0.64 | 0.033 |

  Under malformed reward 0 a truncated search costs nothing more than a wrong answer, so nothing pushes back.
- Promotion: the 3-seed form fails (seed 0 is below 0.2703). For the 5-seed form, seed 0 counts as collapsed
  (labelled diverged, final below the holder's 0.240), the one allowed. r16 promotes if seeds 3 and 4 both end
  above 0.240 without collapsing; the mean then clears 0.2703 on its own (it needs seeds 3 + 4 >= 0.3025).
  Seed 3 is at step ~280 (dev 0.300 at step 250) and ends ~13:25, seed 4 ~14:45.
- On an H200, 500 steps took 12,124 s of wall from the trainer's start (seed 2). An r16-style record
  (`budget_steps` 500 with `budget_steps_time_stop`) would end ~200 min into its 300-min budget, so it can't run
  over. But the H200 board needs A 004 (A 001's config, time-based, all 17,400 s) to land first, and A 004 has the
  600-s margin problem. The warm-up answer gates all of H200 track A.

D 003 (`records/v3/track_d/003_dm_math_fp32_h200`, D 002's config on the H200) has all five seeds:

| seed | node | test | dev (final) | pretrain_s | GRPO overhead | total_s | within 21,600 |
|---|---|---|---|---|---|---|---|
| 0 | node4900 | 0.1590 | 0.168 | 6876 | 811 | 21727 | no |
| 1 | node4301 | 0.1385 | 0.125 | 6779 | 88 | 20906 | yes |
| 2 | node4300 | 0.1455 | 0.173 | 7206 | 478 | 21724 | no |
| 3 | node4300 | 0.1734 | 0.183 | 6734 | 147 | 20921 | yes |
| 4 | node4300 | 0.1526 | 0.153 | 6732 | 150 | 20922 | yes |

(GRPO overhead = total - pretrain_s - the 14,040-s training clock.)

- `validate_record_v3.py` fails seeds 0 and 2 (over budget), so D 003 can't be a record as it stands. Its five
  seeds average 0.1538 ± 0.0134 on the test ladder (D 002, H100: 0.1512 ± 0.0202), dev 0.160.
- Seeds 1, 3 and 4 used 20,906-20,922 s, as D 002 did on the H100 (20,884-20,971), so the recipe fits when
  start-up is warm. The same node gave 7,206 s of pretraining at 03:30 and 6,732-6,734 s later.
- Its RL hadn't plateaued: dev rose through the last hour on seeds 2-4 (0.147 to 0.173, 0.173 to 0.183, 0.148
  to 0.153). Seed 0's flat end was the slow node. So moving RL time to pretraining isn't promising.

Warm start on screens (`scripts/warm_start.sh`, run by `run_item.sh` since bc57b3a, before the CUDA check; off
for record seeds):

| item | node | warm start | item start to vLLM's first log line |
|---|---|---|---|
| r16 s1 | node2901 | 38 s | 56 s |
| r16 s2 | node4300 | 30 s | 40 s |
| r16 s3 | node4301 | 52 s | 62 s |
| r16 s4 | node4300 | 9 s | 20 s |

Cold, the CUDA check timed out at 150 s on 15 of 42 starts, and r16 seed 0 took 15 min to reach vLLM's first log
on node2904. With the warm start, none of the four timed out. I added this to the open question (QUESTIONS
06:40).

Queued in `h200` (16 items):
- **r17** (`calibration/research_r17`), r16 plus `malformed_reward_solved` (a truncated output scores this value
  in groups with a correct sample, 0 elsewhere), against seed 0's slide and the 20-60 % truncated outputs that
  score zero at eval: `mal0_st500_trs05` (-0.5) and `mal0_st500_trs02` (-0.2), seeds 0-2 first, then seeds 3-4
  at the end of the queue. The control is r16, step-capped like them.
- **r18** (`calibration/research_r18`), track D: D 003's RL from D 003's own pretrained models
  (`screen_pretrained`) with `prompts_per_step` 256 instead of 64, at lr 3e-6 (`d003rl_p256_lr3e-6`) and 6e-6
  (`d003rl_p256_lr6e-6`), seeds 1-3. In D 003's RL, sampling takes 0.9 of 1.05 s per step (vLLM scheduling a
  110M model on the CPU), only ~5 of 64 groups are informative, and entropy is ~0.055. Four times the prompts
  should cost well under four times the step time and give ~20 informative groups per update. Seeds 1-3 pair
  with D 003's seeds on the same pretrained models and GPU type. Seed 0's control is the slow node4900 run.

Workers:
- Cancelled the two mit_normal_gpu chains (26824034-37 and 26824039-42). They read only the held A record queues,
  so each start would exit at once, as their heads did at 06:51. My priority there is fair-share-bound (age 1,130
  of 246,012), so resubmitting costs nothing.
- Submitted two new chains of three, reading A 004's and A 002's queues and then `h200`: 26927304-06 and
  26927308, 26927310, 26927311. They take record seeds first once the records are released, and screens until
  then. One 6-h job runs one r17 item.
- 26927517 replaces 26862717: mit_preemptable H200, A record queues and then `h200`, RESERVE_MIN 380.
- GPUs: 26824150 (r16 seed 3) and 26854533 (r16 seed 4) running. 26854534, 26927517 and the two normal chain heads
  pending: 6.

## 2026-10-26 (13:05): the maintainer allows an untimed warm-up (RULES_v3 v3.2); D 003 reported as failed and rerun as D 004; A 004 released to the front of `h200`

The answers (INBOX 12:00, committed at 11:41) arrived two minutes before the 11:43 commit, so I saw them at 12:43:
1. Record seeds may use an untimed warm-up: libraries and the base model read into the page cache, and imports
   in a process that sees no GPU. A GPU warm-up pass is also allowed, as modded-nanogpt does: engine start-up,
   CUDA graphs, compilation, and one dummy step of each stage. Afterwards every piece of trained state must be
   exactly as without it, the warm-up must use no ladder data, and its time is logged separately. The maintainer
   asked for it in RULES_v3 as a v3.2 clarification, for measurements on screens first, and for each README to
   say whether the record used it.
2. D 003 is reported as failed, and D 002's config reruns unchanged, with the warm-up, as D 004.

Done (8910562):
- RULES_v3: a v3.2 line in the header and the ruling as "Untimed warm-up" under "Clock and hardware".
- The warm-up records use is the one without the GPU, `scripts/warm_start.sh`, which screens have run since
  07:07. On r16 seeds 1 and 2, vLLM's engine was up 55 s (H100) and 30 s (H200) after launch: weights loaded
  in 0.7-0.8 s, `init engine` took 14-24 s, graph capture 3 s. Cold starts took up to 15 min before vLLM's first
  log line. That brings start-up back to the H100's level (A 001's seeds lost 33-98 s in all), well inside
  track A's 600-s margin. So I haven't built the GPU pass. It would save the remaining ~0.5 min per stage, but it
  needs every piece of state restored exactly.
- `run_item.sh` now warm-starts record seeds as well (`WARM_START=0` turns it off). For a record it appends the
  time to `<record>/warm_start.log` as well as worker.log, so each README can cite it. The file was replaced
  atomically. Its running copies had already exec'd into their seed scripts.
- `records/v3/track_d/003_dm_math_fp32_h200/README.md` reports the failure: all five seeds with test scores and
  clocks, and why seeds 0 and 2 ran over.
- `records/v3/track_d/004_dm_math_fp32_h200_warm/config.json` is D 002's config.json, byte for byte.

Queues and workers:
- Only the two running workers are certain for the next ~15 h. `sbatch --test-only` puts a new mit_preemptable
  H200 job at 10-27 04:37, and the mit_normal_gpu chain heads at 10-29.
- A 004's seeds 0-4 are at the front of `h200`, which every worker reads. The two running workers take seeds 0
  and 1 when r16 seeds 3 and 4 end (~13:25 and ~14:45), and seeds 2-4 after that. If nothing is preempted, A 004
  lands ~05:30 tomorrow. It goes first because every track A challenger needs it as the holder, and it is
  itself the first H200 track A record.
- D 004's seeds 0-4 are in their own queue, `record-004_dm_math_fp32_h200_warm`. I cancelled the two pending
  preemptable workers (26854534, 26927517) and resubmitted them reading that queue and then `h200`, as
  26933254-55. Their age priority was ~180 points, so nothing was lost. D seeds need 376 min, so they stay off
  the 6-h normal jobs, which read `h200`.
- A 002 is at the end of `h200`, after r17 and r18, not beside A 004 as the 10-25 plan had it:
  - Its H100 seeds (dev 0.300 / 0.270 / 0.472 / 0.315 / 0.263) have sd 0.085.
  - Against A 001's 0.240 ± 0.045, the same gap would give a one-sided Welch p of about 0.05, so it is about
    even odds to pass.
  - r16 decides at ~14:45: if it promotes, its record (call it A 005) is the stronger challenger.
  - r17 tells us whether the truncation penalty lifts the recipe further.
  - If the laptop session wants A 002 sooner, moving its five lines to the front of `h200` is enough.
- `held_records.txt` is emptied (a one-line note).
- GPUs: 26824150 and 26854533 running (r16 seeds 3 and 4); pending 26933254, 26933255 and the two mit_normal_gpu
  chain heads (26927304, 26927308): 6.

## 2026-10-26 (17:45): B200 packing probe submitted (INBOX 16:45: screens may use B200 too; records stay on the H200)

The laptop note allows screen-only workers on B200 as well, after a measurement of one B200 with one
and then two runs. `scripts/research/pack_probe.sh` (written for box 1 on 10-25, never run) does that:
- track A: one run, then two at once (90 s apart), 30 steps each from r7 `s300` seed 1's searching
  weights at constant lr, `vllm_gpu_memory` 0.2 and `micro_batch_tokens` 4096 (r15's settings for two
  runs per B200), no saves;
- track D: one run, then four, 300 steps each from D 002's pretrained seed 0 (`vllm_gpu_memory` 0.08).

It prints each run's s/step over steps 6-30 and the summed steps per hour. I added the CUDA module
load the queue workers do (vLLM compiles kernels with nvcc). Job 26953974: mit_preemptable B200, 2.5 h,
16 CPUs, 128 GB, dev ladder only (a 60-puzzle final eval). `sbatch --test-only` puts it at 10-27
15:22 (a short H200 job: 11:42; mit_normal_gpu B200: 10-28 09:25).

Cap: the later mit_normal_gpu chain head, 26927308 (estimated 10-29 12:00), now waits on the probe
(afterany), so the probe takes its slot: 4 preemptable H200 workers + 26927304 + the probe = 6.

H200 numbers to compare with, from r16 seed 2's log (r16's settings: `vllm_gpu_memory` 0.35,
`micro_batch_tokens` 8192): 16.6 s/step at steps 6-30 (~525 tokens per sample), 27.1 s at steps
250-300 (~740), 24.8 s at steps 450-500 (~717); sampling is 13-15 s of each step and the update 4-11 s.

What follows depends on the result. If two runs on a B200 add up to clearly more steps per hour than
one H200 run at matched output lengths, step-capped track A screens move to screen-only B200 workers,
two per GPU, with r15's memory settings, from a queue with no record items (`h200` holds A 004 and
A 002, which must stay on the H200). Screens with a time budget (r18, track D RL) stay on the H200,
since their results depend on the GPU's speed.

## 2026-10-28 (19:25): preemptions cost A 004 seed 0 at 99%; r16 seed 3 0.332; the B200 probe failed (libstdc++), fixed and requeued

I was paused from 10-27 08:50 to 10-28 19:00 (usage limit). What ran, from sacct and
`queue/worker.log` (EDT):
- 26933254 started D 004 seed 0 on node4500 at 10-27 16:26 and was preempted 4.5 min in
  (`extra/interrupted_0_1`). It started seed 0 again on node5100 at 10-28 12:17 and was preempted at
  15:15, about 3 h in; that attempt moves to `extra/` when the worker restarts.
- 26933255 started D 004 seed 1 on node5100 at 10-27 18:29 and was preempted at 19:36
  (`extra/interrupted_1_1`). It started seed 1 again on node4400 at 10-28 13:21. That run ended
  within budget: total_s 20,993 of 21,600 (pretraining 6,739 s, GRPO 14,287 s; the 57-s warm start
  is logged separately), in-loop final dev 0.130. Its test-ladder eval is running.
- 26824150 resumed r16 seed 3 on node4500 at 10-27 19:13 and finished it at 19:34. It then started
  A 004 seed 0, which was preempted at 10-28 00:25 at train_s 17,271 of 17,400 (step 2,167; in-loop dev
  0.15-0.16 from step 800 on). This was an infrastructure stop, so the attempt goes to `extra/` and
  seed 0 reruns.
- 26854533 resumed r16 seed 4 on node4400 at 10-27 20:16 and was preempted at 21:10, before its next
  hourly save, so seed 4 resumes again from step ~296 (dev 0.303).
- Probe 26953974 was preempted 3 min in at 11:10. It restarted at 12:38 on a B200, and every run
  failed within 9 min (below).

r16 (r5 plus cosine over 500 steps) has 4 of 5 seeds: 0.232 (diverged below 0.240, so it counts as
collapsed), 0.405, 0.412 and 0.332; mean 0.345. If seed 4 ends above 0.240, r16 promotes: the mean
is then at least 0.324 against the 0.2703 line, 4 of 5 seeds are above 0.240, and 1 is collapsed.

Why the probe failed: every run died loading vLLM's attention kernel. On the B200, vLLM picks the
FLASHINFER backend (on the H200 it uses FlashAttention). FlashInfer JIT-builds its decode kernel
(`~/.cache/flashinfer/0.6.18.post1/100a/cached_ops/fmha_gen/fmha_gen.so`) with `$CXX`. Engaging's
default environment loads Spack GCC 12.2.0 (`CXX` points to it), so the kernel needs GLIBCXX_3.4.26
or later. All 12 B200 nodes run Rocky 8, whose system libstdc++ is GCC 8.5's (6.0.25). The process
already holds the system library (from torch) when the kernel loads, so the kernel's RPATH to GCC
12's lib64 doesn't help. Fix: B200 runs put `$CXX`'s libstdc++ directory first in `LD_LIBRARY_PATH`
(`pack_probe.sh`, `packed_worker.sbatch`). `queue_worker.sbatch` and the H200 workers are unchanged.
A CPU job on a Rocky 8 node (27161845, with libcuda from the CUDA stubs) checked it. Without the fix
it reproduced the error. With the fix it loaded GCC 12's libstdc++ 6.0.30 and the version error was
gone. The bare dlopen then stopped at a symbol that FlashInfer's loader (tvm_ffi) provides, so a
second job, 27162074, loads the kernel through tvm_ffi.

The probe is requeued as 27162075 (mit_preemptable B200, 2.5 h). It waits on 26933255 (afterany),
which exits once seed 1's eval ends (less than RESERVE_MIN left), so the probe takes that slot. A new
preemptable H200 worker, 27162158, reads D 004's queue and then `h200`. It waits on the probe, so
D 004 gets a second reader. Cap: 26933255 running, plus 26854533, 26824150, 26933254, 26927304 and
26927308 pending, makes 6. The probe and 27162158 queue behind 26933255, so they never add a seventh.

## 2026-10-28 (19:35): D 004 seed 1 test 0.1296; FlashInfer's loader works with the fix; the probe is next

- D 004 seed 1 (rerun, node4400): test ladder 0.1296, pass@16 0.160, in-loop final dev 0.130,
  within budget (total_s 20,993). D 003's five H200 seeds scored 0.139-0.173 (seed 1: 0.1385).
  Committed with the two preempted first attempts (`extra/interrupted_0_1`, `interrupted_1_1`, with
  notes).
- CPU check 27162074 (node2507, Rocky 8): with `$CXX`'s libstdc++ first, `tvm_ffi.load_module` (the
  loader FlashInfer uses) loads the cached B200 decode kernel. So the probe should get past the
  error that stopped it.
- 26933255 exited after seed 1 (less than 380 min left). Probe 27162075 is now eligible, and 27162158
  (the second D 004 reader) waits on it. Cap: 0 running + 6 pending (26824150, 26854533, 26933254,
  26927304, 26927308, 27162075).

## 2026-10-28 (20:25): the B200 probe: one track A run is 1.3x an H200 and a second adds 11 %, track D packs 2.5x with three runs; r17 moves to two B200 workers, one run each

Probe 27162075 ran on node5800-c1 (mit_preemptable B200) from 19:16 to 20:14 with the libstdc++ fix
(LOG 19:25). Track A started from r7 `s300` seed 1 (long searching outputs), track D from D 002's
pretrained seed 0, as planned (LOG 10-26 17:45).

| phase | settings | s/step (sampling + update) | tokens per sample | steps/h, summed |
|---|---|---|---|---|
| A, one run | `vllm_gpu_memory` 0.2, `micro_batch_tokens` 4096 | 20.5 (10.0 + 10.3) | 758 | 176 |
| A, two runs | the same, started 90 s apart | 38.6 and 35.3 | 755 | 195 |
| D, one run | `vllm_gpu_memory` 0.08 | 1.26 | 27 | 2,860 |
| D, four runs | the same | 1.50 / 1.60 / 1.47; run 2 failed | 26 | 7,093 (three runs) |

The H200 number to compare with is r16 seed 2's log (`vllm_gpu_memory` 0.35, `micro_batch_tokens` 8192):
27.1 s/step at steps 250-300 (15.5 s sampling + 11.5 s update), at 740 tokens per sample, which is
133 steps/h.

What it means:
- One track A run on a B200 is 1.3x an H200 run (20.5 against 27.1 s/step), even with the packing
  settings, which give vLLM less cache and the update half-size micro-batches. Sampling is the gain:
  10.0 s against 15.5 s.
- A second run on the same B200 adds only 11 % (195 against 176 steps/h): one run already keeps the
  GPU busy. One run at the arms' own settings (0.35 and 8192, too big to fit twice in 183 GB) should be
  at least as fast as at the packing settings, so packing track A would need new configs for a few
  percent at most. I don't pack track A on Engaging. Counted per GPU in the 6-GPU limit, a B200 does
  1.3-1.5x an H200's track A screening.
- Track D packs well: three runs together made 2.5x one run's steps per hour, and four would make about
  3.3x. Run 2 died at start-up: vLLM found no memory for its cache ("No available memory for the
  cache blocks"), because run 1 was still growing while run 2's engine measured the free memory. A longer
  stagger or a larger share would fix that. Only step-capped track D RL screens could use this; r18 has
  a time budget and stays on the H200.

The cluster at 20:20: 436 jobs were pending for H200s on mit_preemptable and 23 for B200s, and 9
B200s were free. None of my H200 jobs has started since 19:15, and `squeue --start` puts the
mit_normal_gpu heads at 10-29 18:00 and 10-30 09:00.

Decisions:
- r17's ten items (`mal0_st500_trs05` and `trs02`, seeds 0-4) move from `h200` to a new screen-only
  queue, `screens_b200`, with no record items. Two B200 workers read it, one run per GPU at the arms'
  own settings: 27168549 and 27168550 (mit_preemptable, `packed_worker.sbatch` with PER_GPU=1 for its
  libstdc++ line, REQUIRE_GPU "NVIDIA B200", RESERVE_MIN 240, 8 CPUs, 64 GB). r17's configs are
  unchanged. r17 is step-capped at 500 steps, so its B200 seeds compare with r16's H200 seeds. The first
  seeds' logs will give the real B200 speed at those settings.
- AGENT.md wants the screen that gates a record to run on the H200. If r17 meets the promotion rule on
  the B200, three seeds rerun on the H200 before a record attempt.
- Cap: 27162158 (the second D 004 reader) now waits on 27168549, and the mit_normal_gpu head 26927308
  waits on 27168550 (afterany). Six jobs can run together: the 2 B200 workers, the 3 preemptable H200
  workers and 26927304. When `screens_b200` is empty, the B200 workers exit and the other two become
  eligible.
- `h200` now holds A 004 seeds 1-4, then r18 (6) and A 002 (5): 15 items.

The first two B200 workers, 27167715 and 27167716, started at 20:22 on node5502-c1 and node5501-c1 and
exited 3 s later ("less than 240 min left"), with nothing popped. `packed_worker.sbatch` took its
sub-workers' deadline from `SLURM_JOB_END_TIME`, and on this Slurm (26.05.4) that is the start plus
`--time-min`, not the time limit. A 1-CPU test job (27168049, the same `--time` and `--time-min`)
printed END - START = 14,400 s while `squeue` gave 1-23:59:59 left. My earlier test of the script ran
outside Slurm with DEADLINE set, so it missed this. The packed worker now reads the time left from
`squeue` (%L), as `queue_worker.sbatch` itself does; ruff and the 133 tests pass. The workers were
resubmitted as 27168549 and 27168550, and the two H200 jobs were chained behind those instead.

## 2026-10-28 (20:34): r17 `trs05` seed 0 failed at start-up on the shared Triton cache; requeued once

Both B200 workers started their first items at the same moment (20:28), and vLLM's kernel warm-up
compiled the same new sm_100 Triton kernels on both nodes into the shared cache under my home
directory. `trs05` seed 0 (node5600-c1) read a file that the other node was replacing: `OSError: [Errno
116] Stale file handle` in Triton's compiler, 4 min in, before any step. It is a host failure, so the
item goes back to the front of `screens_b200`, once (`screen_seed.sh` keeps the failed log as
`stdout.interrupted.<stamp>.log`). The other node's run (`trs02` seed 0) got past the warm-up, and the
same worker's next item (`trs05` seed 1) found the kernels cached. Only simultaneous first compiles
race, so later B200 workers start a few minutes apart.

## 2026-10-28 (23:10): r16 seed 4 ends at dev 0.438, so r16 promotes; its record, A 005, is queued on the H200s after A 004

r16 (`calibration/research_r16/mal0_st500`) is r5's recipe, with the cosine to 0.1x counted over 500
steps. r5's recipe is A 001 plus fp32 master weights at lr 1e-6, with zero-advantage rows skipped
and training reward 0 for truncated outputs. All five seeds are in. Seed 4 started on 10-26 on
mit_preemptable H200s. It was preempted twice and resumed from its saved state each time (node4300,
node4400, node4301), and ended at 03:06Z. A step cap makes it GPU-independent.

```
  mal0_st500   s0 done  290m final 0.232 best 0.395 t.10   87 spm   1.7 ent 0.03 inf 18.9 diverged  | 0.050 0.092 0.335 0.355 0.395
  mal0_st500   s1 done  290m final 0.405 best 0.405 t.10   87 spm   1.7 ent 0.08 inf 22.3 diverged  | 0.033 0.065 0.277 0.345 0.385
  mal0_st500   s2 done  290m final 0.412 best 0.418 t.10   58 spm   1.7 ent 0.06 inf 21.8 ok        | 0.000 0.105 0.268 0.345 0.397
  mal0_st500   s3 done  290m final 0.332 best 0.393 t.10   87 spm   1.7 ent 0.04 inf 27.4 ok        | 0.055 0.097 0.270 0.343 0.393
  mal0_st500   s4 done  290m final 0.438 best 0.438 t.10   87 spm   1.7 ent 0.07 inf 20.8 ok        | 0.035 0.078 0.268 0.368 0.415
= mal0_st500     n=5 final 0.364 ± 0.084  ok/1-guess/div 3/0/2  median t.10 87m
```

("290m" is the step clock: 500 steps count as 17,400 s.)

- Promotion, 5-seed form, against the H100 line until A 004 lands:
  - The mean, 0.364, clears 0.2703.
  - Four seeds end above the holder's 0.240 (0.405, 0.412, 0.332, 0.438).
  - Only seed 0 counts as collapsed: it is labelled diverged and ends below 0.240. Seed 1 is also
    labelled diverged, but on this answer-constrained arm that counts only below 0.240 (PLAN, 10-23).

  **r16 promotes.**
- The last 100 steps show where the seeds part:

| seed | steps 401-450: reward / truncated | steps 451-500: reward / truncated | final dev |
|---|---|---|---|
| 0 | 0.401 / 0.54 | 0.300 / 0.61 | 0.232 |
| 1 | 0.425 / 0.42 | 0.414 / 0.46 | 0.405 |
| 2 | 0.432 / 0.21 | 0.409 / 0.22 | 0.412 |
| 3 | 0.405 / 0.45 | 0.372 / 0.43 | 0.332 |
| 4 | 0.439 / 0.19 | 0.444 / 0.19 | 0.438 |

  The lr is the same for every seed: 1.5e-7, then 1.1e-7. The seeds that kept their searches short
  (2 and 4, about a fifth truncated) end highest. Seed 0 slid into truncation. r17's penalties on
  truncated outputs target that slide.
- Record: `records/v3/track_a/005_fp32_mal0_cos500`. Its config is r16's, with two changes: weights
  are saved (no `no_save`), and `budget_steps_time_stop` is set, so training also stops at
  17,400 s of real time. The sampler is unchanged (`constrain`, T = 1). On an H200, 500 steps took
  12,124 s of wall time (seed 2), so a seed should end about 200 min into its 300-min budget.
  `validate_record_v3.py` checks the wall-clock `clock.json`, not the step clock. N = 5, seeds 0-4,
  in `h200` after A 004's seeds 2-4 and before r18 and A 002. Every H200 worker reads that queue:
  mit_normal_gpu, and mit_preemptable under INBOX 10-26 (a preempted seed's attempt goes to `extra/`
  and the same seed reruns).
- It is compared with the H200 holder, A 004, once both land. A 004's seed 1 started on 26854533
  right after r16's seed 4.
- B200 against H200 at r16's settings, steps 251-500:

| run | GPU | s/step | sampling | update | tokens/sample |
|---|---|---|---|---|---|
| r17 `trs05` seed 1 | B200 | 15.2 | 8.4 | 6.8 | ~654 |
| r16 seed 4 | H200 | 22.1 | 11.4 | 10.7 | ~697 |
| r16 seed 2 | H200 | 25.2 | 14.4 | 10.8 | ~712 |

  Per token, the B200 is 1.4-1.5x faster. `trs05` seed 1 (malformed_reward_solved -0.5) ended at dev
  0.357. r17 gets its own entry when its arms finish.
- What r17 must beat: A 005 becomes the holder if it lands, and its line would then be about
  0.364 + 0.03. So an r17 arm that clears today's line but not r16's dev waits for A 005's result
  rather than becoming another record attempt.

## 2026-10-29 (00:30): A 005 stops at step 400 instead of 500 (r16's dev at step 400: 0.397 ± 0.011); r17's first seeds overcorrect or slide

r16's five seeds, in-loop dev by step (dev truncated share after the slash):

| seed | 300 | 350 | 400 | 450 | 500 |
|---|---|---|---|---|---|
| 0 | 0.355/0.61 | 0.390/0.57 | 0.395/0.56 | 0.382/0.56 | 0.232/0.68 |
| 1 | 0.345/0.43 | 0.370/0.45 | 0.385/0.51 | 0.402/0.51 | 0.405/0.52 |
| 2 | 0.345/0.25 | 0.362/0.21 | 0.397/0.23 | 0.418/0.22 | 0.412/0.22 |
| 3 | 0.343/0.45 | 0.368/0.37 | 0.393/0.38 | 0.382/0.48 | 0.332/0.37 |
| 4 | 0.368/0.22 | 0.368/0.23 | 0.415/0.20 | 0.423/0.23 | 0.438/0.19 |
| mean ± sd | 0.351 ± 0.011 | 0.372 ± 0.011 | **0.397 ± 0.011** | 0.401 ± 0.020 | 0.364 ± 0.084 |

- All five seeds rise together to step 400. After that, two seeds keep gaining (2 and 4) and two slide
  (0 and 3). So the last 100 steps add risk without adding to the mean.
- A 005 had no seed started, so its config now stops at step 400 (`max_steps` 400). The rest is
  unchanged: the cosine is still counted over 500 steps (`budget_steps` 500), and
  `budget_steps_time_stop` is still set. Up to step 400 the run is the one r16's seeds made, and its
  final eval is the in-loop eval at step 400.
  - r16's seeds at step 400 serve as its screen: mean 0.397 (line 0.2703), five of five above 0.240,
    none collapsed. Seeds 0 and 1 have more than half their dev outputs truncated, but on this
    constrained arm that counts as collapse only below 0.240 (PLAN, 10-23).
  - The folder is renamed `records/v3/track_a/005_fp32_mal0_cos500_s400`, and its five `h200` lines
    follow.
- Choosing the stop from the same seeds' evals flatters the estimate a little. But the curve is smooth,
  the sd at 400 is 0.011, and the test ladder is held out, so the reported number is unaffected.
- Cost: 500 steps on an H200 took 12,124 s of wall time (r16 seed 2). Steps 401-500 are the slowest
  (longest outputs), so 400 steps should take about 9,600 s, about 2.7 h of the 4.8-h training budget.
  That leaves room to spend the extra time on bigger batches later (agenda A2/A3).
- r16 ran on two GPU types under a step cap (PLAN: the gating screen is on the H200 or step-capped):
  seeds 0-1 on H100s (10-26), seeds 2-4 on H200s. A 002 (H100 dev 0.324) is still at the end of `h200`
  in the laptop's order, after A 005.

r17 (r16 + a training penalty on truncated outputs, on the B200s), the three finished seeds:

| arm (penalty) | seed | dev at 400 | final dev | steps 451-500: reward / truncated / tokens per sample |
|---|---|---|---|---|
| `trs05` (-0.5) | 0 | 0.177 | 0.190 | 0.177 / 0.00 / 165 |
| `trs05` (-0.5) | 1 | 0.332 | 0.357 | 0.345 / 0.12 / 661 |
| `trs02` (-0.2) | 0 | 0.395 | 0.230 | 0.279 / 0.29 / 723 |

- At -0.5, seed 0's truncation fell to zero by step 300. Its outputs then shrank from about 580 to
  165 tokens, and it stopped searching: the opposite failure from r16 seed 0's.
- At -0.2, seed 0 still slid in its last 50 steps: truncated 0.19, then 0.29, and dev 0.395, then
  0.230.
- Two seeds per arm can't settle it, given how widely seeds differ. The remaining seven items run
  (about 1.5 h each on the B200s, and nothing else is waiting for those GPUs). But the arms would
  need about 0.43+ to beat A 005's expected dev, so r17 is now a question about the slide, not a
  record candidate.

## 2026-10-29 (01:45): r19 trains with a shorter token limit than it is scored with (768 or 896, eval at 1,024); queued first on the B200s

- The idea. The 1,024-token limit is where r16's points go:
  - By step 400, 20-60 % of its dev outputs were truncated (seed 0 0.56, seed 4 0.20), and every
    truncated output scores 0.
  - Its two weakest finals (seeds 0 and 3) came from searches that grew into the limit after step
    450.
  - r17's flat penalty on truncated rollouts either overcorrected (short guesses) or didn't hold.

  r19 instead caps the training rollouts below the eval limit. A truncated rollout still gets
  reward 0, no worse than a wrong answer, so the only pressure is the positive advantage of shorter
  correct searches. The policy should learn searches that finish within 768 or 896 tokens, leaving
  slack at 1,024.
- New option `eval_max_new_tokens` (`rlvr_speedrun/grpo.py`, off by default, so existing configs run
  as before):
  - If set, the in-loop eval uses it as its token limit, and training rollouts keep
    `max_new_tokens`.
  - vLLM's context is sized for the larger of the two.
  - It needs `cache_len` 0 (the HF static caches are sized by `max_new_tokens`).

  A new test checks both limits and the check. Ruff and the 134 tests pass. A record's test ladder
  is scored by the reference scorer at 1,024 regardless.
- Arms (`calibration/research_r19/`): `mal0_st500_tc768` and `mal0_st500_tc896`, which are r16's config
  with `max_new_tokens` 768 or 896 and `eval_max_new_tokens` 1,024. Both are step-capped at 500 steps,
  so they compare with r16's H100/H200 seeds. In-loop evals run every 50 steps, so the step-400
  numbers compare with A 005.
- Queue: seeds 0-2 of both, interleaved, at the front of `screens_b200`, ahead of r17's remaining five.
  Each takes about 1.5 h on a B200, so the six take about 4.5 h on the two workers.
- What counts: a final or step-400 dev clearly above r16's (0.364 final, 0.397 at step 400) with no
  slide. A record candidate needs about 0.43 (A 005's expected 0.397 + 0.03), with its gating seeds
  on the H200 (AGENT.md).

## 2026-10-29 (02:55): A 004 seed 0: test 0.3135, within budget

`records/v3/track_a/004_ladder_constrained_h200`, seed 0 (rerun after the 10-28 preemption), on
node4301 (mit_preemptable H200), with a 31-s warm start:

| test ladder | pass@16 | truncated | mean tokens | in-loop final dev | steps | total_s (budget 18,000) |
|---|---|---|---|---|---|---|
| 0.3135 | 0.5754 | 0.012 | 525 | 0.287 | 998 | 17,463 (within) |

Test pass@1 by level: L1 0.921, L2 0.307, L3 0.235, L4 0.079, L5 0.010, L6 0.329.

- A 001's config scored 0.2481 on the H100 board (dev 0.240). On the H200 the same config fits 998
  steps into the budget. One seed is not a mean, but the H200 line for later arms will likely sit near
  0.29 + 0.03 once A 004's five in-loop finals replace the H100 number (PLAN). A 005's expected 0.397
  clears that.
- Seed 1 runs on 26854533 (from 23:06, about 04:10 to the end of its training budget). Seed 2 started
  on 26824150 at 02:50. Seeds 3-4 are next in `h200`, then A 005.

## 2026-10-29 (04:20): A 004 seed 1: test 0.2358, within budget

| seed | node | test ladder | pass@16 | truncated | mean tokens | in-loop final dev | steps | total_s |
|---|---|---|---|---|---|---|---|---|
| 0 | node4301 | 0.3135 | 0.5754 | 0.012 | 525 | 0.287 | 998 | 17,463 |
| 1 | node4301 | 0.2358 | 0.5579 | 0.005 | 425 | 0.230 | 1042 | 17,438 |

Seed 1 had a 9-s warm start. Its test pass@1 by level: L1 0.778, L2 0.202, L3 0.130, L4 0.063,
L5 0.014, L6 0.228. The two seeds spread as A 001's did on the H100 (dev 0.240 ± 0.045). Seed 3
started on 26854533 at 04:14, and seed 2 runs on 26824150. Seed 4 is next in `h200`, then A 005.

## 2026-10-29 (04:50): node5501-c1's GPU failed; r19 `tc896` seed 1 requeued once; replacement B200 worker 27206873

- What happened on node5501-c1:
  - At 04:43, B200 worker 27168549 finished r19 `tc896` seed 0 (final dev 0.352).
  - The next item, `tc896` seed 1, failed 21 s after its start, in `torch._C._cuda_init()`.
  - The item after that failed the worker's CUDA check ("CUDA unknown error"). That made two quick
    failures in a row, so the worker put the second item (`tc768` seed 2) back and exited.

  The cause is the node's GPU, not the item. So `tc896` seed 1 goes back to the front of
  `screens_b200`, once.
- New B200 worker 27206873: the same settings as 27168549, plus `--exclude=node5501-c1`. It starts
  with warm Triton kernels in the shared cache (the other B200 worker is mid-run), so the
  start-up race of 10-28 20:34 doesn't apply.
- Cap: 27162158 (D 004's second reader) waits on 27206873 again (afterany). That leaves 3 running
  (2 H200, 1 B200), plus 3 that could run (27206873, 26933254, 26927304): 6.
- r19 so far, as dev at step 400 / final (r16: 0.397 / 0.364):
  - `tc768` seed 0: 0.362 / 0.272.
  - `tc896` seed 0: 0.375 / 0.352. Its dev truncated share went from 0.54 at step 400 to 0.27 at
    step 500.
  - `tc768` seed 1 is at step ~300, with dev 0.325 at step 300.

## 2026-10-29 (04:55): A 004 seeds 2 and 3 preempted on node4301; they rerun when their workers restart

- mit_preemptable preempted both H200 workers on node4301 at 04:48:56. Both were requeued and are
  pending again:
  - 26824150 was running A 004 seed 2, started 02:50, about 2 h of its 4.8-h training budget in.
  - 26854533 was running A 004 seed 3, started 04:14, 34 min in.

  When each worker restarts, it moves the partial attempt to `extra/interrupted_<seed>_<n>` and
  reruns the same seed. These are infrastructure stops (INBOX 10-26), and each gets a `note.txt`
  then.
- The replacement B200 worker 27206873 started at 04:48 on node5800-c1, with r19 `tc896` seed 1.
- Cap: 2 running (both B200), plus 4 that could run (26824150, 26854533, 26933254, 26927304): 6.

## 2026-10-29 (07:00): r17 closes on five seeds (truncation penalties lose to r16); r20 (training temperature 1.1, 96 prompts per step) replaces its last five items on the B200s

r17 (r16 + a training penalty on truncated rollouts in solved groups, B200, step-capped):

```
  mal0_st500_trs05  s0 done  290m final 0.190 best 0.305 t.10  87 spm  1.7 ent 0.03 inf 16.1 ok  | 0.032 0.097 0.305 0.240 0.177
  mal0_st500_trs05  s1 done  290m final 0.357 best 0.357 t.10  87 spm  1.7 ent 0.08 inf 22.7 ok  | 0.030 0.090 0.212 0.293 0.332
  mal0_st500_trs05  s2 done  290m final 0.340 best 0.340 t.10  116 spm  1.7 ent 0.08 inf 20.8 ok  | 0.018 0.078 0.212 0.283 0.322
= mal0_st500_trs05  n=3 final 0.296 ± 0.092  ok/1-guess/div 3/0/0  median t.10 87m
  mal0_st500_trs02  s0 done  290m final 0.230 best 0.395 t.10  58 spm  1.7 ent 0.05 inf 18.6 ok  | 0.047 0.110 0.250 0.382 0.395
  mal0_st500_trs02  s1 done  290m final 0.392 best 0.400 t.10  87 spm  1.7 ent 0.08 inf 19.7 ok  | 0.042 0.075 0.292 0.357 0.392
= mal0_st500_trs02  n=2 final 0.311 ± 0.114  ok/1-guess/div 2/0/0  median t.10 72m
```

- Both arms end below r16 (0.364 final, 0.397 at step 400). At -0.5, one seed gave up searching
  (outputs fell to 165 tokens, dev 0.190). At -0.2, one seed still slid at the end (0.395 to 0.230).
  r17 closes here.
- Its remaining five items (`trs02` seeds 2-4, `trs05` seeds 3-4) are dropped from `screens_b200`
  rather than run as filler on shared B200s.

r20 (`calibration/research_r20/`, r16's config with one change each, step-capped at 500 steps, B200):

| arm | change | what it tests |
|---|---|---|
| `mal0_st500_t11` | training rollouts at temperature 1.1 (log-probs at the same temperature) | the entropy fall: r16's and r19's seeds that slid had entropy near 0.03, and r19 `tc768` seed 0 slid with its truncated share flat |
| `mal0_st500_p96` | 96 prompts per step instead of 64 | lower-noise steps on the same lr path. A 005 uses about 2.7 h of the 4.8-h budget, so 1.5x the batch fits 400 steps on an H200 (about 4 h) |

- Seeds 0-2 of both, interleaved, follow r19's last two seeds. A `t11` seed takes about 1.5 h on a
  B200, a `p96` seed about 3 h.
- The in-loop dev eval stays greedy. A record scores at T = 1 (`v3.sampler`) whatever the training
  temperature.
- When `screens_b200` empties, the B200 workers exit, and the two chained H200 jobs (27162158,
  26927308) become eligible.
- H200s: all four pending H200 workers wait on Resources (A 004's two preempted seeds, D 004, and the
  mit_normal_gpu head), so only the B200s are running.

## 2026-10-29 (08:45): r19 closes: a shorter training token limit doesn't help (step 400: 0.359 and 0.351 against r16's 0.397)

```
  mal0_st500_tc768  s0 done  290m final 0.272 best 0.372 t.10  87 spm  1.7 ent 0.04 inf 17.3 ok  | 0.020 0.093 0.262 0.362 0.362
  mal0_st500_tc768  s1 done  290m final 0.365 best 0.365 t.10  87 spm  1.7 ent 0.07 inf 24.5 ok  | 0.012 0.083 0.247 0.325 0.350
  mal0_st500_tc768  s2 done  290m final 0.342 best 0.365 t.10  87 spm  1.7 ent 0.04 inf 21.3 diverged  | 0.037 0.052 0.227 0.347 0.365
= mal0_st500_tc768  n=3 final 0.326 ± 0.049  ok/1-guess/div 2/0/1  median t.10 87m
  mal0_st500_tc896  s0 done  290m final 0.352 best 0.375 t.10  116 spm  1.7 ent 0.05 inf 23.4 ok  | 0.030 0.037 0.225 0.303 0.375
  mal0_st500_tc896  s1 done  290m final 0.338 best 0.338 t.10  87 spm  1.7 ent 0.07 inf 20.9 diverged  | 0.023 0.058 0.223 0.278 0.322
  mal0_st500_tc896  s2 done  290m final 0.210 best 0.357 t.10  116 spm  1.7 ent 0.04 inf 20.4 diverged  | 0.040 0.068 0.167 0.297 0.357
= mal0_st500_tc896  n=3 final 0.300 ± 0.078  ok/1-guess/div 1/0/2  median t.10 116m
```

Dev by step, mean ± sd of three seeds (r16: 0.351, 0.372, 0.397, 0.401, 0.364):

| arm | 300 | 350 | 400 | 450 | 500 |
|---|---|---|---|---|---|
| `tc768` | 0.344 ± 0.018 | 0.353 ± 0.018 | 0.359 ± 0.008 | 0.340 ± 0.018 | 0.326 ± 0.049 |
| `tc896` | 0.293 ± 0.013 | 0.312 ± 0.019 | 0.351 ± 0.027 | 0.320 ± 0.032 | 0.300 ± 0.078 |

- Both arms are below r16 at every step from 350 on, and both still slide after step 400 (`tc896`
  seed 2: 0.357 to 0.210).
- The dev truncated share stayed at 0.5-0.6 for `tc896` and for `tc768` seed 2, even with the
  slack. Training under a lower limit didn't teach the policy to finish sooner at eval.
- r19 closes. Together with r17, this says the late slide isn't a truncation problem. Stopping at
  step 400 (A 005) stays the answer until r20 (temperature 1.1, 96 prompts) or something else holds
  past it.
- r20 runs on both B200s: `t11` seed 0 on node5600-c1 (dev 0.143 at step 250, a slow takeoff after
  a high-entropy start), and `p96` seed 0 on node5800-c1 (from 08:40).

## 2026-10-29 (10:10): r20 `t11` closes on two seeds (0.290, 0.183); `p96` learns faster per step so far; A 004 seed 4 and D 004 seed 0's rerun started

```
  mal0_st500_t11  s0 done  290m final 0.290 best 0.292 t.10  145 spm  1.7 ent 0.04 inf 18.6 ok  | 0.017 0.058 0.082 0.213 0.275
  mal0_st500_t11  s1 done  290m final 0.183 best 0.183 t.10  116 spm  1.7 ent 0.06 inf  8.6 one-guess | 0.032 0.063 0.128 0.130 0.157
= mal0_st500_t11  n=2 final 0.237 ± 0.075  ok/1-guess/div 1/1/0  median t.10 130m
```

- Training at temperature 1.1:
  - The base model's first 50 steps are near-random long outputs (entropy 5.8, three quarters
    truncated).
  - The policy then shrinks to about 150-token guesses, and only starts searching again around
    step 250.
  - Seed 1 never did. It took 26 min of B200 time for 500 steps and ended one-guess at 0.183.

  The entropy stays higher in the middle, but both seeds end far below r16 (0.364). `t11` closes,
  and its seed 2 is dropped from `screens_b200`.
- `p96` (96 prompts per step), dev by step against r16's five-seed mean:

| | 100 | 150 | 200 |
|---|---|---|---|
| `p96` seed 0 | 0.147 | 0.273 | 0.318 |
| `p96` seed 1 | 0.130 | | |
| r16 mean | 0.087 | 0.183 | 0.284 |

  It learns faster per step so far. Seeds 0 and 1 end about 11:40 and 12:10, and seed 2 follows.
- H200 workers that got GPUs:
  - 26927304 (mit_normal_gpu, node3100) started A 004 seed 4 at 08:07 (59-s warm start).
  - 26933254 (node4300) restarted at 08:42 and moved D 004 seed 0's second attempt to
    `extra/interrupted_0_2`, with a `note.txt`. That attempt started 10-28 16:19Z on node5100 and
    was preempted at 15:15 EDT, 3,644 s into RL after the full pretraining. The rerun started at
    08:43.
- A 004 seeds 2 and 3 still wait on their preempted workers (26824150, 26854533, Resources).

## 2026-10-29 (12:10): r20 `p96` collapses after step 300 (seed 0 ends at 0.045); the collapse follows the entropy, not the step count; B200 worker 27206873 cancelled

`p96` (96 prompts per step, r16's lr path):

| seed | 250 | 300 | 350 | 400 | 450 | 500 |
|---|---|---|---|---|---|---|
| 0 | 0.377 | 0.395 | 0.357 | 0.227 | 0.172 | 0.045 |
| 1 | 0.380 | 0.385 | 0.362 | 0.182 | (running) | |
| r16 mean | 0.317 | 0.351 | 0.372 | 0.397 | 0.401 | 0.364 |

Seed 0's training, by window:

| steps | reward | truncated | entropy | lr |
|---|---|---|---|---|
| 251-300 | 0.412 | 0.53 | 0.068 | 4.8e-7 |
| 301-350 | 0.403 | 0.54 | 0.044 | 3.5e-7 |
| 351-400 | 0.305 | 0.41 | 0.036 | 2.3e-7 |
| 401-450 | 0.231 | 0.28 | 0.041 | 1.5e-7 |
| 451-500 | 0.107 | 0.29 | 0.061 | 1.1e-7 |

Seed 1: reward 0.425 at steps 251-300, then 0.282 and 0.167. Its entropy went 0.078, 0.047, 0.035,
0.056.

- Bigger batches learn faster per step: 0.39 by step 300, which r16 reached at step 400. They also
  reach the low-entropy state sooner. In both seeds, the reward fell within 50-100 steps of entropy
  reaching 0.035-0.045, with truncation falling, not rising.
- That matches every slide so far: r16 seeds 0 and 3, r19, r17 `trs02`. The trigger is entropy near
  0.04, whatever the step or batch. Stopping before it (A 005 at step 400 on r16's path) works, but it
  caps the recipe near 0.40.
- The next lever is keeping entropy up once the policy is sharp. Entropy control doesn't exist in
  `grpo.py` yet: KL 0.05 to the base held the policy at the base (r13), and Clip-Cov blew up (r9).
  r21 will add an entropy bonus that turns on below a target entropy (off by default).
- `p96` closes. Seed 1 finishes on 27168550, and then that worker exits on the empty queue.
- B200 worker 27206873 was preempted at 11:58, 8 min into `p96` seed 2, and was pending. I cancelled
  it (my own listed job) and removed its held item, so seed 2 isn't rerun. That makes 27162158 (D 004
  then `h200`, H200) eligible.
- Cap: 3 running (26927304, 26933254, 27168550), plus 3 that could run (26824150, 26854533,
  27162158): 6.

## 2026-10-29 (12:25): r21, an entropy bonus that turns on below a target entropy; queued on the B200 worker

New options in `rlvr_speedrun/grpo.py`, off by default, so existing configs run as before:
- `entropy_coef`: subtracts that coefficient times the mean token entropy of the updated rows from
  the loss.
- `entropy_target`: turns the bonus on only on steps after one whose logged entropy was below the
  target. 0 means every step.

Details:
- The entropy is the exact per-token entropy that is already logged. With the bonus on, it keeps
  its gradient, through `torch.utils.checkpoint`, so its softmax is recomputed in the backward
  instead of being stored. The memory then stays near the current peak: 104 GB of 141 on an H200
  for r16, 117 of 180 on a B200.
- The train log gets `entropy_coef`, the coefficient applied that step, for runs that set it.
- The first step after a start or resume has no previous entropy, so it runs without the bonus
  unless the target is 0.
- Two new tests:
  - On a sharpened tiny model, a step with zero advantages raises the entropy only with the bonus.
  - The target gates the bonus as described.

  Ruff and the 136 tests pass.

r21 (`calibration/research_r21/`, r16's config, step-capped at 500 steps, B200):

| arm | entropy_coef | entropy_target |
|---|---|---|
| `mal0_st500_ent01_t10` | 0.01 | 0.1 |
| `mal0_st500_ent05_t10` | 0.05 | 0.1 |

- r16's entropy passes 0.1 around steps 250-300, so the bonus acts on the second half of the run. It
  works like a thermostat: once the entropy is back above 0.1, the next step runs without it.
- Under Adam each step's size is set by the lr either way, so the coefficient sets how much of the
  step goes to entropy instead of reward, not how big the step is.
- Seeds 0-2 of both, interleaved, run in `screens_b200` on 27168550 after `p96` seed 1. That is one
  B200 worker only, so the eligible H200 jobs keep their places for the records. Each seed takes
  about 1.5-2 h.
- What counts: dev that keeps rising past step 400 without a slide. A record candidate needs about
  0.43 (A 005's expected 0.397 + 0.03).

## 2026-10-29 (13:10): A 004 seed 4: test 0.1834, a one-guess seed, within budget

| seed | node | test ladder | pass@16 | truncated | mean tokens | in-loop final dev | steps | total_s |
|---|---|---|---|---|---|---|---|---|
| 0 | node4301 | 0.3135 | 0.5754 | 0.012 | 525 | 0.287 | 998 | 17,463 |
| 1 | node4301 | 0.2358 | 0.5579 | 0.005 | 425 | 0.230 | 1042 | 17,438 |
| 4 | node3100 | 0.1834 | 0.2429 | 0.000 | 47 | 0.185 | 2501 | 17,508 |

- Seed 4 ran on mit_normal_gpu worker 26927304, with a 59-s warm start. It never learned to search:
  47-token answers, 2,501 short steps, and pass@16 0.243 against 0.56-0.58 for the searching seeds.
  It is the one-guess mode that A 001 also showed in one of five seeds on the H100 (PLAN reference
  points: 4/1/0).
- Its test pass@1 by level: L1 0.595, L2 0.151, L3 0.084, L4 0.058, L5 0.017, L6 0.195.
- Every seed counts. Seeds 2 and 3 rerun when their preempted workers (26824150, 26854533) restart.
- 26927304 exited after the seed: less than 330 min was left of its 6 h. The next job in its chain,
  26927305, is now eligible and reads A 004's and A 002's (empty) queues, then `h200`, whose head is
  A 005 seed 0.

## 2026-10-29 14:45 EDT: D 004 seed 0 within budget

D 004 (`004_dm_math_fp32_h200_warm`, D 002's recipe on the H200 with the untimed warm-up), seed 0's
third attempt, on preemptable worker 26933254 (node4300, 36-s warm start):

| seed | test ladder | pass@16 | truncated | mean tokens | final dev | RL steps | total s (of 21,600) |
|---|---|---|---|---|---|---|---|
| 0 | 0.1832 | 0.2367 | 0.000 | 25 | 0.182 | 14,955 | 20,816 (pretrain 6,731) |

- It is within budget and scores above D 002's H100 test mean (0.1512) and D 004 seed 1 (0.1296).
- Seed 1 and seed 0 are done. Seed 2 started on the same worker at 18:34Z (144-s warm start); seeds 3
  and 4 stay queued for 26933254 and 27162158.

## 2026-10-29 15:45 EDT: r21 ent01 seed 0, no slide

r21 `mal0_st500_ent01_t10` (r16 + entropy bonus 0.01 below an entropy of 0.1), seed 0, on B200 worker
27168550 (12:27-15:18), 500 steps:

| step | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| dev (greedy) | 0.117 | 0.273 | 0.368 | 0.378 | 0.400 |
| logged entropy | 0.391 | 0.126 | 0.101 | 0.104 | 0.104 |

- The bonus first engaged at step 237 and was on for 57 of the remaining 264 steps. It held entropy
  at about 0.10. Without it, r16, r17, r19 and r20 drifted down to 0.035-0.045 and then slid.
- Dev kept rising to the end with no slide (r16's five seeds at 500: 0.232-0.438, mean 0.364). One
  seed is not enough to tell it from r16's good seeds (0.405-0.438). The bonus also cost nothing on
  memory (no OOM).
- `ent05_t10` seed 0 started on the same worker at 19:18Z; seeds 1-2 of both arms follow.

## 2026-10-29 18:45 EDT: r21 ent05 seed 0 ends at 0.395

r21 `mal0_st500_ent05_t10` (r16 + entropy bonus 0.05 below an entropy of 0.1), seed 0, on B200 worker
27168550 (15:18-18:21), 500 steps:

| step | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| dev (greedy) | 0.072 | 0.232 | 0.313 | 0.372 | 0.395 |
| logged entropy | 1.200 | 0.196 | 0.137 | 0.104 | 0.106 |

- This seed's entropy fell more slowly than ent01 seed 0's, so the bonus was on for only 6 steps (first
  at step 364). Dev rose to the end, with no slide. The final dev by level: L1 0.99, L2 0.39, L3 0.26,
  L4 0.17, L5 0.08, L6 0.48. Max memory was 120 GB.
- Quirk: step 12 logged entropy 0.0 (a step with no informative groups, so nothing was updated). That
  turned the bonus on for step 13. It is harmless (one step at 1.6 entropy), but the target check should
  skip such steps. I'll fix it as a new option-free change only after r21's queued seeds finish, since
  they read this code.
- Both r21 seeds so far end at about 0.40 with no slide. That is r16's good-seed level, not yet the
  0.43 that a new record candidate needs. `ent01_t10` seed 1 started at 22:21Z; seeds 1-2 of
  `ent05_t10` and seed 2 of `ent01_t10` follow.

## 2026-10-29 19:50 EDT: H200 waits are days long; r22 design, a second B200 worker

- No H200 job has started since 08:43. Slurm puts 26927305 (mit_normal_gpu) at 11-03 10:30, and many
  mit_normal_gpu H200 nodes are under user reservations until 11-04 or later. A 005 seed 0 (head of
  `h200`), A 004 seeds 2-3 and D 004 seeds 3-4 wait on that. Records stay on the H200, so for now
  B200 screens are the work that moves.
- r22: r21's two arms at lr 2e-6 (`calibration/research_r22/mal0_st500_{ent01,ent05}_t10_lr2e-6`, seeds
  0-1 each, queued in `screens_b200` after r21's three remaining seeds). In r1-r3, lr 2e-6 peaked
  earliest and highest, then collapsed. r21 shows the entropy floor stops the late slide at lr 1e-6, so
  the open question is whether it makes the faster lr usable within 500 steps.
- A second B200 screen worker, 27283068 (`packed_worker.sbatch`, PER_GPU=1, `screens_b200`), starts
  after 26933254 ends. 26933254 runs D 004 seed 2 until about 20:30 and then exits (less than 6.5 h
  left). That keeps the cap at 2 running + 4 eligible pending.

## 2026-10-29 20:45 EDT: D 004 seed 2 within budget

| seed | test ladder | pass@16 | truncated | mean tokens | final dev | RL steps | total s (of 21,600) |
|---|---|---|---|---|---|---|---|
| 2 | 0.1384 | 0.2100 | 0.000 | 22 | 0.137 | 14,972 | 20,892 (pretrain 6,728) |

- Seed 2 ran on 26933254 (node4300, 18:34-00:29Z, 144-s warm start).
- D 004 so far: seed 0 0.1832, seed 1 0.1296, seed 2 0.1384 (mean 0.1504; D 002's H100 mean 0.1512).
- Seeds 3-4 are left for 27162158, which waits on the H200 shortage (LOG 19:50).
- 26933254 then exited. B200 screen worker 27283068 started on node5702-c1 at 00:32Z with r21
  `ent05_t10` seed 1.

## 2026-10-29 23:55 EDT: r21 three more seeds; the entropy floor stops slides but doesn't raise the level

The B200 runs take about 2.3 h each (500 steps), so three more seeds finished on 27168550 and 27283068.

| arm | seed | dev at 100 / 200 / 300 / 400 / 500 | final entropy | bonus steps |
|---|---|---|---|---|
| `ent01_t10` | 0 | 0.117 / 0.273 / 0.368 / 0.378 / 0.400 | 0.10 | 57 |
| `ent01_t10` | 1 | 0.085 / 0.283 / 0.343 / 0.395 / 0.405 | 0.10 | 32 |
| `ent01_t10` | 2 | 0.092 / 0.280 / 0.307 / 0.318 / 0.302 | 0.09 | 156 |
| `ent05_t10` | 0 | 0.072 / 0.232 / 0.313 / 0.372 / 0.395 | 0.11 | 7 |
| `ent05_t10` | 1 | 0.067 / 0.263 / 0.345 / 0.372 / 0.390 | 0.11 | - |

- `ent01_t10`, n=3: final 0.369 ± 0.058; at step 400 0.364. r16 (no bonus, n=5): final 0.364 ± 0.084, at step 400 0.397.
- No seed slid, so the bonus does what it was meant to. But seed 2 stopped improving at about 0.31 from
  step 250 on (L2 0.28, L4 0.09 at step 500), with the bonus on for 156 steps. The other seeds end near 0.40,
  the same as r16's good seeds. The step-400 mean is below A 005's 0.397, so neither arm is a
  record candidate (needs about 0.43).
- `ent05_t10` seed 2 is running; r22 (the same arms at lr 2e-6) has started (`ent05` seed 0). The question
  now is whether the floor lets a higher lr pay off. I added r22 seed 2 for both arms to `screens_b200`
  so the two workers have about 7 h of work.

## 2026-10-30 01:45 EDT: r21 done; preemptions on the second B200 worker

| arm | n | dev at step 400 | final dev (500) | seeds |
|---|---|---|---|---|
| `ent01_t10` | 3 | 0.364 | 0.369 ± 0.058 | 0.400 / 0.405 / 0.302 |
| `ent05_t10` | 3 | 0.381 | 0.399 ± 0.011 | 0.395 / 0.390 / 0.412 |
| r16 (no bonus) | 5 | 0.397 | 0.364 ± 0.084 | 0.232 / 0.405 / 0.412 / 0.332 / 0.438 |

- `ent05_t10` seed 2 ended at 0.412 (L4 0.22, L6 0.49), with the bonus on for 11 steps. None of the three
  `ent05` seeds slid, and they ended close together.
- r21 is closed. `ent05_t10` is the steadier run, but its mean (0.399) is about A 005's expected 0.397,
  far from the 0.43 a record needs. Its step-400 dev (0.381) is below r16's. The bonus is worth keeping
  as a guard for longer or higher-lr runs, which is what r22 tests.
- B200 worker 27283068 was preempted twice, at 00:35 (about 1.5 h into r22 `ent05` seed 0) and at 01:38
  (about 1 h in). Slurm requeued it each time. It restarted on node5702-c1 and is running that seed again
  from the start; the partial runs are kept as `stdout.interrupted.*.log` in the seed dir. 27168550 has
  started r22 `ent01` seed 0. `screens_b200` holds four more r22 items (about 7 h on the two workers).

## 2026-10-30 02:50 EDT: A 004 seed 2 rerun started; B200 worker replaced

- 26824150 (preemptable H200) restarted on node4301 and started the A 004 seed 2 rerun at 05:59Z, after a
  98-s warm start. The preempted attempt is in `extra/interrupted_2_1` with a note: step 1000, 7,026 s of
  the 17,400-s budget used. It doesn't count.
- B200 worker 27168550 was preempted at about 01:38 while running r22 `ent01` seed 0. Slurm requeued it
  onto node5501-c1, the node with the failed GPU; that job was submitted before the exclusion was added.
  CUDA failed twice in 8 s there, so the worker put the item back at the front of `screens_b200` and exited
  as designed. Its partial run of `ent01` seed 0 reruns from the start.
- 27283068 was preempted a third time (now on node5800-c1, from 02:05) and is on r22 `ent05` seed 0 again.
- New B200 worker 27318631 (same settings, excludes node5501-c1). I moved the dependency of the
  mit_normal_gpu H200 job 26927308 from 27168550 to 27318631, so it still waits for a B200 worker to end.
  The cap stays at 6: running 26824150, 27283068; eligible pending 26927305, 26854533, 27162158, 27318631.

## 2026-10-30 03:50 EDT: r22 `ent05_t10_lr2e-6` seed 0: faster, same ceiling

| step | 100 | 200 | 300 | 400 | 450 | 500 |
|---|---|---|---|---|---|---|
| dev | 0.250 | 0.378 | 0.402 | 0.387 | 0.403 | 0.360 |

- At lr 2e-6 the run reaches 0.40 by step 300, against step 500 at lr 1e-6 (r21). It then stays at 0.39-0.40
  and drops to 0.360 at step 500 (L1 1.00, L2 0.33, L4 0.07, malformed 0.34). Entropy stayed at 0.10-0.14
  (bonus on 14 steps). That's no slide in the r16 sense, but no gain either.
- Correction to the 01:45 entry: preempted screens don't start over. They resume from the last
  checkpoint (this seed resumed at step 225 after its last preemption), so a preemption costs well
  under an hour.
- Seed 1 of this arm and `ent01_t10_lr2e-6` seed 0 are running on 27283068 and 27318631.

## 2026-10-30 04:50 EDT: A 004 seed 3 rerun started; r22 `ent01_t10_lr2e-6` seed 0 restarted

- A 004 seed 3: worker 26854533 restarted on node4301 and began the rerun at 08:30:46Z (7-s warm start).
  The worker had moved attempt 1 (preempted 10-29 at step 126, 11 % of the budget) to
  `extra/interrupted_3_1`; I added its `note.txt`. Seeds 2 and 3 are both rerunning now.
- B200 worker 27318631 was preempted and requeued at 03:16 (same node). Its run of `ent01_t10_lr2e-6` seed 0
  had reached step 200, before the first resume checkpoint (step 225), so it started over. The worker kept the
  partial run as `seeds/0.interrupted.1793344598` in the pool. Correction to 03:50: resuming needs a
  checkpoint, so a preemption before step 225 still costs the whole run so far.
- The restarted seed is at dev 0.43 at step 300, the best step-300 dev of any arm so far (malformed 0.51).
  `ent05_t10_lr2e-6` seed 1 is at 0.405 at its latest eval.

## 2026-10-30 05:55 EDT: A 004 seeds 2 and 3 preempted again; r22 two more seeds; r23 queued

- Both A 004 reruns on node4301 were preempted at 04:53:04 EDT (08:53:04Z). Seed 2's attempt 2 had reached
  step 728 (10,221 s of 17,400, 59 %); seed 3's attempt 2 reached step 75 (1,299 s, 7 %). Workers 26824150 and
  26854533 are pending again; they move the partial runs to `extra/` on restart, and I'll add the
  `interrupted_2_2` and `interrupted_3_2` notes then. Neither attempt counts.
- r22 (lr 2e-6, cosine over 500 steps, entropy floor at 0.1). Dev every 50 steps:

| arm | seed | 100 | 200 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|
| `ent01_t10_lr2e-6` | 0 | 0.245 | 0.362 | 0.430 | 0.437 | **0.458** | 0.438 | 0.427 | 0.47 |
| `ent05_t10_lr2e-6` | 0 | 0.250 | 0.378 | 0.402 | 0.397 | 0.387 | 0.403 | 0.360 | 0.34 |
| `ent05_t10_lr2e-6` | 1 | 0.290 | 0.410 | 0.428 | 0.437 | 0.400 | 0.382 | 0.387 | 0.25 |

  - All three peak between steps 300 and 400 (0.403-0.458) and lose 0.03-0.05 by step 500, while the lr
    is already small (0.2-0.4e-6). Entropy stayed at 0.10-0.14 throughout, so this isn't the
    entropy slide.
  - `ent01` seed 0 is the best single run so far: 0.427 final, L4 0.19, L6 0.58. The summarizer labels it
    "diverged" only because training malformed passed 0.5 early (0.72 at step 100). It ends at 0.47,
    the most truncation of the three; r16's best seeds also kept the most.
  - The step-400 mean of the three is 0.415, against r16's 0.397 (which A 005 uses). A stop at step 400 is
    the natural record form here too, but one seed of `ent01` isn't enough.
- Queued on `screens_b200`: `ent01_t10_lr2e-6` seeds 3-4 (with seeds 1-2 that makes 5 for a step-400
  estimate; it needs about 0.43 to beat A 005's expected 0.397). Also r23 `ent01_t10_lr3e-6` seeds 0-1:
  does the floor keep lr 3e-6 stable too? About 6 h for the two B200 workers.

## 2026-10-30 07:50 EDT: r22 `ent01_t10_lr2e-6` seed 1 collapses at step 350; grad-norm logging; r23 switches to grad_clip 0.3

- r22 `ent01_t10_lr2e-6` seed 1 (one clean run on 27283068, node5800-c1, no resume) ends at **0.078**.
  Dev by step: 100 0.252, 200 0.352, 300 0.372, 350 0.223, 400 0.065, 450 0.058, 500 0.078.
  - The mechanism matches the known mal0 collapses (LOG 10-24 09:40): a truncation spike first. Training
    malformed went from 0.25 to 0.70 between steps 335 and 366, while entropy drifted from 0.11 to 0.077
    with the bonus on. Reward fell from 0.37 to 0.02. Then malformed dropped to 0.01 with reward stuck at
    0.03-0.07: a degenerate short-answer policy that doesn't recover.
  - So the entropy floor stops the slow entropy slide but not this jump. The arm now stands at 0.427 and
    0.078 (seeds 2-4 to come); one collapse in five is allowed by the promotion rule, but the mean needs
    about 0.43.
- Code: `_update` now returns the gradient norm before clipping, and `grpo_step` logs it as `grad_norm`
  in train_log (only when `grad_clip` > 0, the default). Nothing else changes. ruff and pytest
  (136 passed) are clean. Runs that start from now on log it; it shows whether the clip binds before
  and during a collapse.
- r23 `ent01_t10_lr3e-6` is dropped before it ran: a faster lr makes no sense while 2e-6 still
  collapses. In its place, r23 `mal0_st500_ent01_t10_lr2e-6_gc03`: r22 `ent01` with `grad_clip` 0.3
  instead of 1.0 (PLAN A2, tighter clipping), seeds 0-2. `screens_b200` is now r22 `ent01` seeds 3-4 and
  gc03 seeds 0-2, interleaved; with r22 `ent01` s2 and `ent05` s2 running, about 7 h for the two B200 workers.

## 2026-10-30 09:50 EDT: r22 `ent05_t10_lr2e-6` seed 2: 0.440

- Seed 2 of `ent05_t10_lr2e-6` ends at **0.440** (300 0.447, 350 0.443, 400 0.427, 450 0.443, 500 0.440;
  malformed 0.46, entropy 0.11 at the end). It is the first r22 seed with no late decline.
- The arm closes at n=3: final 0.396 ± 0.041 (0.360, 0.387, 0.440), step-400 0.405 (0.387, 0.400, 0.427).
  That's about r16's 0.397 at step 400, so not a gain worth a record. Still running: r22 `ent01` seeds 2-3.
  Queued: `ent01` seed 4, r23 grad_clip 0.3 seeds 0-2.

## 2026-10-30 10:55 EDT: r22 `ent01_t10_lr2e-6` seeds 2-3: 0.445, 0.448; grad norms are small, so r23 moves to clip 0.15

- r22 `ent01_t10_lr2e-6`, dev every 50 steps (seed 4 running):

| seed | 100 | 200 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.245 | 0.362 | 0.430 | 0.437 | 0.458 | 0.438 | 0.427 | 0.52 |
| 1 | 0.252 | 0.352 | 0.372 | 0.223 | 0.065 | 0.058 | 0.078 | 0.01 |
| 2 | 0.237 | 0.335 | 0.383 | 0.397 | 0.422 | 0.440 | **0.445** | 0.21 |
| 3 | 0.190 | 0.373 | 0.398 | 0.413 | 0.430 | 0.437 | **0.448** | 0.33 |

  - Seeds 2 and 3 climb to the end with no late decline. The three that didn't collapse average 0.440 at
    step 500, the best of any arm. But seed 1's collapse puts the mean of four at 0.350, and a mean of five
    can't reach 0.43 now. The arm's problem is the collapse risk, not its ceiling.
- The first `grad_norm` logs (r22 `ent01` seed 4 and r23 gc03 seed 0, steps 1-175): median 0.17, max 0.34.
  So `grad_clip` 1.0 has never bound in these runs, and gc03 bound on only 4 of 176 steps: it's nearly the
  same run as r22 `ent01`. gc03 seed 0 keeps running (it logs the norms through step 500, including any spike
  before a collapse). Its queued seeds 1-2 are replaced by r23 `mal0_st500_ent01_t10_lr2e-6_gc015`
  (`grad_clip` 0.15, which binds on 122 of those 176 steps), seeds 0-3.

## 2026-10-30 12:50 EDT: D 004 seed 3: test 0.1677; r22 `ent01` closes at 0.355; r23 gc03 seed 0: 0.378

- D 004 seed 3 (worker 27162158, node2500, H200, after a 59-s warm start): test **0.1677** (L1 0.57,
  L6 0.20, pass@16 0.228), last in-loop dev 0.183. Within budget: 20,941 s of 21,600 (pretrain 6,736 s,
  GRPO 14,239 s). Seed 4 started on the same worker at 16:36Z (12:36 EDT) after an 11-s warm start.
  D 004 so far: 0.1832, 0.1296, 0.1384, 0.1677.
- r22 `ent01_t10_lr2e-6` seed 4: 0.378 (0.405 at step 450). The arm closes at n=5: 0.427, 0.078, 0.445,
  0.448, 0.378, mean **0.355 ± 0.157**. Step 400: 0.458, 0.065, 0.422, 0.430, 0.382, mean 0.351.
  Without the collapse it would be about 0.425; with it, below r16's 0.364 and A 005's step-400 0.397.
  r22 is closed. The entropy floor removes the slow slide; one collapse in eight r22 runs remains.
- r23 `ent01_t10_lr2e-6_gc03` seed 0: 0.378 final, 0.428 at steps 350 and 450.
- Grad norms over full runs (r22 `ent01` s4 / gc03 s0), median per 100 steps: 0.17/0.17, 0.18/0.19,
  0.20/0.17, 0.23/0.20, 0.27/0.23. They grow late as the lr decays, and no run so far shows a spike
  (max 0.38 and 0.50). So clip 0.3 binds mostly in the last 100 steps (26 and 6 of them). Neither run
  collapsed, so there's no norm trace of a collapse yet.
- Queue: r23 gc015 seeds 0-1 running, seeds 2-4 and gc03 seed 1 queued (about 6 h for the two B200 workers).

## 2026-10-30 14:55 EDT: r23 clip 0.15: seed 0 0.392, seed 1 collapses (0.018); no grad-norm spike before it, so r24 tries ent05 at lr 3e-6

- r23 `ent01_t10_lr2e-6_gc015` (`grad_clip` 0.15), dev every 50 steps:

| seed | 100 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.213 | 0.328 | 0.377 | **0.435** | 0.417 | 0.393 | 0.390 | 0.392 | 0.57 |
| 1 | 0.237 | 0.387 | 0.427 | 0.422 | 0.408 | 0.278 | 0.063 | 0.018 | 0.62 |

- Seed 1's collapse, from the training log: reward fell from 0.45 (step 345) to 0.17 (step 393) while the batch's
  malformed rate stayed at 0.27-0.29. Truncation followed later (0.48 at steps 425-450, 0.67 after). So this one
  starts as wrong but well-formed answers, not as a slide into truncation. The pre-clip grad norm stayed at
  0.19-0.26 through the fall and only rose (0.30-0.48) once reward was already near 0.1. Entropy stayed at the
  0.10 floor throughout. A clip can't catch a collapse that has no spike before it.
- ent01-based runs so far: 2 collapses in 8 (r22 `ent01` seed 1, this one). Seed 0 kept a high malformed rate
  (0.55) the whole run and was stable but flat after step 300.
- With seeds 0-1 at 0.392 and 0.018, gc015's mean of five can't reach 0.43. Seeds 2-3 are running and finish;
  seed 4 is dropped from the queue. In its place, r24 `mal0_st500_ent05_t10_lr3e-6` (r22 `ent05` at lr 3e-6) seeds
  0-1. `ent05` had no collapse in three seeds at 2e-6 but peaked lower. gc03 seed 1 stays queued.
- Next larger item if the bonus arms stay near 0.40: PLAN A4 (SFT on a public math-reasoning set before RL).
  The existing `scripts/sft_warmstart.py` trains on generator solutions, which v3 doesn't allow, so it would
  need new code and a data job.

## 2026-10-30 17:55 EDT: r23 clip 0.15 closes with 2 collapses in 4; A 004 seeds 2-3 rerun (attempt 3)

- r23 `ent01_t10_lr2e-6_gc015`, seeds 2-3 (seed 4 was dropped):

| seed | 100 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.282 | 0.362 | 0.385 | 0.082 | 0.012 | 0.002 | 0.003 | 0.005 | 0.54 |
| 3 | 0.248 | 0.380 | 0.343 | 0.385 | 0.417 | 0.405 | 0.407 | **0.435** | 0.13 |

  - The arm closes at n=4: 0.392, 0.018, 0.005, 0.435, mean **0.212 ± 0.233**. Clip 0.15 doesn't help and may
    hurt (2 collapses in 4, against 1 in 5 for r22 `ent01` without it). ent01-based runs: 3 collapses in 10.
  - Seed 2 was preempted and resumed from its step-234 checkpoint at 15:59 (worker 27318631, node5600-c1).
    It collapsed about 40 steps later, with the same signature as seed 1: training reward fell (0.34 at step 264
    to 0.04 at step 306) before truncation rose (0.30 to 0.62), and the grad norm fell rather than spiked
    (0.25 to 0.08). The resume isn't the cause: seed 1 and r22 `ent01` seed 1 collapsed without one, and r22
    `ent05` seed 0 resumed four times at step 225 and finished at 0.360.
  - Seed 3 is the r22-style good run: still climbing at step 500, with truncation falling to 0.13.
- Worker 27318631 exited at 16:58 (less than 240 min left after its requeue). `screens_b200` keeps one worker
  (27283068, gc03 seed 1, then r24 seeds 0-1, about 7 h). I don't start another while the cap is full: A 004's
  workers restarted, and 26927308 became eligible when 27318631 ended.
- A 004 seeds 2 and 3: workers 26824150 and 26854533 restarted on node4801 at 21:08Z (17:08 EDT) and rerun both
  seeds after a 71-s warm start (attempt 3). Attempts 2 are in `extra/interrupted_2_2` (step 728, 59 % of the
  budget) and `extra/interrupted_3_2` (step 75, 7 %), with notes.

## 2026-10-30 18:55 EDT: D 004 complete, the first track D record on the H200 (0.1552 ± 0.0217); D line now 0.189

Seed 4 finished at about 18:30 on node2500 (worker 27162158, after an 11-s warm start). It scored
test 0.1572, and its total was 20,932 s, within budget. All five seeds are in:

| seed | node | test ladder | pass@16 | final dev | total_s | within 21,600 s |
|---|---|---|---|---|---|---|
| 0 | node4300 | 0.1832 | 0.2367 | 0.182 | 20816 | yes |
| 1 | node4400 | 0.1296 | 0.1604 | 0.130 | 20993 | yes |
| 2 | node4300 | 0.1384 | 0.2100 | 0.137 | 20892 | yes |
| 3 | node2500 | 0.1677 | 0.2283 | 0.183 | 20941 | yes |
| 4 | node2500 | 0.1572 | 0.2037 | 0.163 | 20932 | yes |

- Mean test 0.1552 ± 0.0217. On the H100, D 002 scored 0.1512 ± 0.0202, and the invalid D 003
  scored 0.1538.
- With the warm-up, start-up and saves took 56-233 s per seed. Without it, D 003's two
  over-budget seeds took about 965 s.
- `validate_record_v3.py` passes. The README is written, and the README leaderboard lists 004 as
  the track D (H200) holder.
- Three preempted attempts are in `extra/`, each with a note.
- PLAN's D line moves from 0.190 (H100) to 0.189: dev mean 0.159 + 0.03.
- Worker 27162158 went on to `h200` and is running A 005 seed 0.
- r23 `gc03` seed 1 is at step 200, dev 0.362.

## 2026-10-30 21:55 EDT: A 005 seed 0 ran about 3x slow on node2500 and was preempted; node2500 excluded

Worker 27162158 started A 005 seed 0 at 18:28 on node2500, after a 10-s warm start. It was
preempted between 21:03 and 21:44, at step 165, with 9,273 s of real time. The worker is requeued
(pending), and on restart it moves the partial attempt to `extra/` and reruns seed 0.

The attempt was far too slow. r16, whose config this is, ran on the H100 at about 3,460 s by step
165 and 10,190 s by step 400. Here both stages were slower from the first steps:

| step | A 005 s0 (node2500 H200): sample / update s | r16 s0 (H100): sample / update s |
|---|---|---|
| 5 | 32.4 / 8.5 | 11.1 / 2.6 |
| 50 | 23.3 / 8.5 | 8.0 / 3.8 |
| 150 | 45.0 / 43.9 | 16.3 / 14.0 |

Generation at 4,200-7,000 tokens/s against 13,000-22,000 points to a slow GPU or node, not the
config. At this speed the real-time stop (18,000 s) would end a seed near step 320 instead of 400.

D 004's seeds 3 and 4 ran on node2500 too. Their RL steps were 13,299 and 15,912 against
14,955-16,522 elsewhere, so seed 3's GPU may have been slow as well. Those seeds are already
counted and within budget.

I added node2500 to the exclude list of every pending H200 worker: 27162158, 26927305, 26927306,
26927308, 26927310 and 26927311. node4900 was excluded the same way (10-25). Future H200 workers
exclude node2500 as well. This is an infrastructure exclusion made before seed 0's rerun, and it
does not depend on any seed's score.

## 2026-10-30 22:55 EDT: A 004 complete, the first track A record on the H200 (0.2671 ± 0.0565); A line now 0.285

Seeds 2 and 3 finished their third attempts on node4801 at 22:20 (workers 26824150 and 26854533,
after 71-s warm starts). Both are within budget. All five seeds are in:

| seed | node | test ladder | pass@16 | mean tokens | final dev | total_s | within 18,000 s |
|---|---|---|---|---|---|---|---|
| 0 | node4301 | 0.3135 | 0.5754 | 525 | 0.287 | 17463 | yes |
| 1 | node4301 | 0.2358 | 0.5579 | 425 | 0.230 | 17438 | yes |
| 2 | node4801 | 0.3132 | 0.5429 | 453 | 0.310 | 17527 | yes |
| 3 | node4801 | 0.2895 | 0.5704 | 543 | 0.263 | 17535 | yes |
| 4 | node3100 | 0.1834 | 0.2429 | 47 | 0.185 | 17508 | yes |

- Mean test 0.2671 ± 0.0565. A 001 scored 0.2481 ± 0.0460 on the H100.
- Seed 4 collapsed into single guesses, as A 001's seed 2 did on the H100.
- `validate_record_v3.py` passes. The README is written, and the README leaderboard lists 004 as
  the track A (H200) holder.
- The five preempted attempts are in `extra/`, each with a note.
- PLAN's A line moves from 0.2703 (H100) to 0.285: dev mean 0.255 + 0.03. The 4-of-5 bar is now
  the holder's dev mean, 0.255.
- A 005 (step-capped dev 0.397 at step 400) and A 002 (H100 dev 0.324) clear the new line, so both
  stay queued.

Both workers then exited with less than 380 min left. That left no GPU job running and four
eligible pending: 27162158, 27283068, 26927305 and 26927308. I submitted two new mit_preemptable
H200 workers on `h200`, 27412796 and 27412797 (the second starts 5 min later). They exclude
node2500 and node4900. That makes 6 jobs that could run together.

## 2026-10-31 01:50 EDT: r23 clip 0.3 seed 1 slides to 0.125; gc03 closes; r22 `ent05` seed 3 queued

- r23 `ent01_t10_lr2e-6_gc03` seed 1 finished at 01:10 (B200 worker 27283068, node5602-c1, after four
  preemptions; `stdout.interrupted.*` logs kept):

| seed | 100 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.260 | 0.415 | 0.390 | 0.428 | 0.407 | 0.425 | 0.428 | 0.378 | 0.19 |
| 1 | 0.135 | 0.362 | 0.408 | 0.417 | 0.272 | 0.222 | 0.135 | **0.125** | 0.00 |

  - Seed 1 peaked at 0.417 (step 300) and then lost accuracy at every level: L1 fell from 0.99 to 0.50 and
    L6 from 0.53 to 0.14. This slide is different from the earlier ones. Truncation fell to zero instead of
    rising, answers kept their length (about 350 characters), and the grad norm rose from 0.22 to about
    0.5 while training reward fell from 0.48 (step 275) to 0.11.
  - The arm closes at n=2: 0.378 and 0.125. Clipping at 0.3 doesn't stop the slides either. With r23 `gc015`
    (2 collapses in 4), gradient clipping is closed. ent01-based runs: 4 collapses or slides in 11.
- `screens_b200`: the worker moved on to r24 `ent05_lr3e-6` seed 0 (seed 1 queued). I added r22
  `ent05_t10_lr2e-6` seed 3 after it. That arm had no collapse in three seeds (0.396 ± 0.041), and a fourth
  seed tells us more about its collapse rate. This keeps about 6 h of work queued for the one worker.
- H200 jobs: all seven still pending (27162158 holds A 005 seed 0).

## 2026-10-31 03:50 EDT: r24 `ent05_lr3e-6` seed 0 ends at 0.413; new B200 worker 27432002

- r24 `mal0_st500_ent05_t10_lr3e-6` seed 0 finished at 03:36 (worker 27283068, node5602-c1):

| seed | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.333 | 0.400 | 0.432 | 0.390 | 0.443 | **0.470** | 0.382 | 0.412 | 0.413 | 0.22 |

  - It ends at 0.413 with no collapse. Its peak (0.470 at step 350) is above every eval of r22 `ent05` (best 0.447).
    It is also ahead of r22 `ent05` at lr 2e-6 at every step from 100 to 350 (r22's seeds ended at 0.360-0.440).
    One seed is too few to judge; seed 1 is next.
- Worker 27283068 then exited (less than 240 min left). I submitted 27432002 in its place, with the same
  settings: B200, `screens_b200`, excludes node5501-c1 and node4900. Queue: r24 seed 1, then r22 `ent05` seed 3.
- Cap: 0 running, 6 eligible pending (27162158, 26927305, 26927308, 27412796, 27412797, 27432002).

## 2026-10-31 06:50 EDT: r24 `ent05_lr3e-6` seed 1 collapses at step 430; r24 closes; r22 `ent05` seed 4 queued; B200 worker 27441594

- r24 `mal0_st500_ent05_t10_lr3e-6` seed 1 finished at 06:10 (worker 27432002, node5602-c1):

| seed | 100 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | malformed (end) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.333 | 0.432 | 0.390 | 0.443 | 0.470 | 0.382 | 0.412 | 0.413 | 0.22 |
| 1 | 0.328 | 0.365 | 0.402 | 0.375 | 0.373 | 0.347 | 0.017 | **0.065** | 0.17 |

  - Seed 1 collapsed between steps 400 and 440. Training reward fell from 0.40 to 0.03 while truncation rose
    from 0.25 to 0.56 and entropy rose from 0.13 to 0.19. The grad norm stayed between 0.15 and 0.21. The
    entropy bonus was off throughout (`entropy_coef` 0), since entropy stayed above the 0.10 target.
  - The arm closes at n=2: 0.413 and 0.065. lr 3e-6 is faster early but collapses like the `ent01` runs. The
    lr 2e-6 arm (`ent05`, three seeds and no collapse) stays the reference.
- Every B200 worker so far ran one item and then exited with less than 240 min left: Slurm grants them close
  to their 4-h `--time-min`. The new worker, 27441594, has `--time-min=08:00:00` so it can run two
  screens per start. Its other settings are the same.
- `screens_b200`: r22 `ent05_t10_lr2e-6` seeds 3 and 4 (about 4.6 h). Five seeds of that arm show whether
  its no-collapse run holds up.
- Next larger step, at the 10:45 full wake: plan PLAN A4 (SFT on a public math-reasoning set before RL).
  RL-only stability arms keep landing near 0.40 and do not beat A 005's expected 0.397.
  `scripts/sft_warmstart.py` trains on generator solutions, which v3's data rules don't allow. A4 needs a
  published dataset instead.

## 2026-10-31 10:50 EDT: A 005 seeds 0-2 running on the H200; B200 worker preempted 7 times since 06:49

- H200 workers started at last. The A 005 record seeds are running on mit_preemptable (step clock
  34.8 s per step, 400 steps):

| seed | worker | node | started (EDT) | step at 10:45 |
|---|---|---|---|---|
| 0 | 27162158 | node4300 | 10-31 10:03 (attempt 3) | 156 |
| 1 | 27412796 | node4400 | 10-31 09:22 | 226 |
| 2 | 27412797 | node4400 | 10-31 10:03 | 178 |

  - Seed 0's attempt 2 started on node4400 at 09:02 and was preempted at 09:03 at step 1. Both
    preempted attempts are in `extra/interrupted_0_1` (node2500, step 165) and `extra/interrupted_0_2`,
    each with a `note.txt`. Seed 0 reran under the same seed, as RULES_v3 allows.
  - Real time runs at about 15-21 s per step on these nodes, so each seed should finish in about 2-2.5 h.
- B200 worker 27441594 (node5800-c1) has been preempted and requeued 7 times since 06:49. The longest
  stretch was 63 min. Every restart begins r22 `ent05` seed 3 again from step 0, because screens save no
  weights. No screen step has finished since 06:10. I'm leaving it queued: the B200s are contested in
  the daytime, and the worker loses nothing else.
- Cap: 4 running (27162158, 27412796, 27412797, 27441594) and 2 eligible pending on mit_normal_gpu
  (26927305, 26927308).
- PLAN A4 (SFT on public data before RL, track A): RULES_v3 ("Both tracks: external data") allows any
  public dataset that isn't Countdown-specific, including published distilled sets. The training counts
  toward the 5-h budget. Data preparation is untimed, done by a deterministic script with no model.
  Design, to build next:
  - A new, off-by-default `sft_steps` phase in `rlvr_speedrun/grpo.py`. It runs next-token loss on a
    Qwen-tokenized text file before the first RL step, inside the timed run.
  - Data: DeepMind `mathematics_dataset` v1.0 arithmetic, already downloaded and checked
    (`~/orcd/pool/data/dm_math/parquet`). It needs a Qwen tokenization, a CPU job on mit_normal.
  - First screen: r16-at-400's config plus 100-300 SFT steps, step-capped on the B200.
  - I'll write and test the code while no new A 005 seed is about to start: seeds 3-4 read
    `grpo.py` when they start, so the default path must stay byte-for-byte the same.

## 2026-10-31 11:50 EDT: rules v3.3, no external data on track A; PLAN A4 dropped

- INBOX 10-31 12:00 (commit 71dde06): the maintainer withdrew the external-data amendment for track
  A. Its data is again generator puzzles and verifier verdicts only. Track D keeps external data.
  No track A record or screen used external data. The A4 design in LOG 10:50 is dropped and no code
  was written for it.
- Track A goes on with RL-only arms. Its next screens must beat A 005's expected dev of 0.397 by
  about 0.03.
- External data moves to track D. D 004 holds there with test 0.1552 and dev 0.159. PLAN D3 is SFT on
  public worked solutions before RL. `scripts/sft_warmstart.py` already runs a track D SFT phase, but
  on generator solutions, which RULES_v3 forbids. It can be pointed at a published dataset instead,
  such as DeepMind `mathematics_dataset` arithmetic, which is already downloaded and checked. I'll plan
  that at the next full wake.
- A 005 at 11:43: seed 0 step 310, seed 1 step 377, seed 2 step 324 (of 400).

## 2026-10-31 12:50 EDT: A 005 seeds 0-1 test 0.3995 and 0.4076; seed 2 preempted in its test ladder; INBOX: track D first; r25 (track D, B200) queued

A 005 (`records/v3/track_a/005_fp32_mal0_cos500_s400`, H200, 400 steps):

| seed | node | test ladder | pass@16 | truncated | mean tokens | in-loop final dev | total_s | within 18,000 s |
|---|---|---|---|---|---|---|---|---|
| 0 (attempt 3) | node4300 | 0.3995 | 0.6346 | 0.529 | 709 | 0.375 | 7,781 | yes |
| 1 | node4400 | 0.4076 | 0.6608 | 0.549 | 723 | 0.4033 | 8,867 | yes |
| 2 | node4400 | none: preempted | | | | 0.36 | 7,645 | (training yes) |
| 3 | node4400 | running since 12:16 | | | | | | |
| 4 | node4300 | running since 12:38 | | | | | | |

- Seeds 0 and 1 are the best track A test scores so far: A 004 holds at 0.2671. Both seeds stopped on
  their step count (400) at less than half the budget.
- Seed 2 finished training, but worker 27412797 was preempted at 12:22 during the test ladder, so it
  has no `confirm_v3.json`. That counts as an infrastructure stop under INBOX 10-25: when the worker
  restarts, the runner moves the attempt to `extra/interrupted_2_1` and reruns seed 2 from the start.
  I'll write its `note.txt` then. Its rerun runs the whole seed again, training included.

INBOX 10-31 12:15 (commit ba1f819): track D matters more to the maintainer. It gets at least half of
research effort and screen capacity from now on; A 005 and A 002 still run as planned.

Why track D stalls: every D 004 seed ends with almost no learning signal. Seed 0's last step has 1 of 64
groups informative, entropy 0.020 and batch reward 0.20. From step ~1,500 on, it has 3-10 informative
groups and entropy about 0.05. Its dev still creeps up across all ~15,000 steps (seed 0: 0.083 at step
500, 0.122 at 6,000, 0.182 at the end), so most of the 234 min of RL is spent on a few groups per step.
This is the one-guess mode of LOG 10-19. Two existing options (both off by default) attack that
directly:

- `shaping`: partial credit for wrong answers (`partial_credit` in grpo.py). With the answer
  constrained, almost every wrong answer uses the puzzle's numbers and earns 0.5-0.99 of the weight by
  closeness to the target. Groups stop being all-equal, so most of them carry a gradient.
- the entropy bonus with a target (r21/r22; `ent05` was track A's steadiest arm).

r25 (`calibration/research_r25`), track D, RL from D 004's own pretrained seeds 0-2, step-capped at 6,000 of
D 004's ~15,000 steps, D 004's config otherwise (`vllm_gpu_memory` 0.08 for packing):

| arm | change | seeds |
|---|---|---|
| `d004rl_shape02` | `shaping` 0.2 | 0-2 |
| `d004rl_shape05` | `shaping` 0.5 | 0-2 |
| `d004rl_ent05_t10` | `entropy_coef` 0.05 below entropy 0.10 | 0-2 |
| `d004rl_ctl` | none | 0 |

- The paired control is free: D 004's own eval logs at step 6,000 (0.122 / 0.115 / 0.075 for seeds
  0-2, mean 0.104), on the same pretrained weights. `d004rl_ctl` seed 0 checks that the B200 and the
  packing reproduce it.
- An arm worth following needs a clear gain at step 6,000 with the informative groups still above about
  10. The full-length gating seeds then run on the H200 against the D line of 0.189.
- New B200 worker 27460856 (`packed_worker.sbatch`, PER_GPU=3, queue `screens_d_b200`, RESERVE_MIN 180,
  `--time-min=08:00:00`). The probe ran three track D runs at 1.47-1.60 s/step, so each run is about
  2.5 h and r25 is about 10 h on one B200.
- To stay at 6 GPUs, I cancelled my pending mit_normal_gpu chain 26927308 -> 26927310 -> 26927311. Its
  priority was fair-share-bound and it had not started since 10-29. The other chain (26927305 ->
  26927306) stays.
- Capacity: the B200s now run one track A worker (r22 `ent05` seeds 3-4) and one track D worker with
  three runs. r18 (track D, time-budgeted) stays in `h200`.

## 2026-10-31 13:50 EDT: A 005 seed 3 preempted at step 130; r25 started on the B200

- Worker 27412796 was preempted at 12:48 (16:48:19Z) on node4400 with A 005 seed 3 at step 130 of 400
  (1,861 s of training). It is pending and holds seed 3; on restart the runner moves the attempt to
  `extra/interrupted_3_1` and reruns seed 3. 27412797 (seed 2, preempted in its test ladder) is also
  pending. Seed 4 is at step 231 on node4300 (27162158).
- B200 worker 27460856 started at 13:36 on node5800-c1 with r25's first three runs (`shape02`,
  `shape05`, `ent05_t10`, seed 0); 27441594 restarted on the same node at 13:46 (r22 `ent05` seed 3).

## 2026-10-31 15:50 EDT: A 005 seed 4 test 0.2899, within budget; worker 27162158 moves on to r18

| seed | node | test ladder | pass@16 | truncated | mean tokens | in-loop final dev | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|
| 4 | node4300 | 0.2899 | 0.5875 | 0.232 | 554 | 0.248 | 8,075 | yes | 17 s |

- Seed 4 is well below seeds 0-1 (0.3995, 0.4076), with shorter outputs and fewer truncated. Its dev
  at step 400 (0.248) is also the lowest of the four seeds that trained, so this looks like seed spread
  rather than a test-ladder effect. The mean of the three scored seeds is 0.366.
- Seeds 2 and 3 wait for their workers (27412797, 27412796, both pending). Worker 27162158 popped the
  next `h200` item, r18 `d003rl_p256_lr3e-6` seed 1 (track D).
- B200 workers 27441594 and 27460856 restarted on node5800-c1 at 15:38; r25's first three runs resume.

## 2026-10-31 18:50 EDT: r25 restarted from step 1 at every preemption; its worker now saves every 10 min

- r25's three seed-0 runs had only 27-135 steps after six hours. Worker 27460856 was preempted eight
  times on node5800-c1, each time within an hour of starting. Screens save resumable state every
  3,600 training seconds (`RESUME_EVERY_S`, `scripts/slurm/screen_seed.sh`), so no `state.pt` was
  ever written and each restart began again at step 1. The 15:50 entry's "resume" was wrong. The
  partial attempts are in the weights dir as `0.interrupted.<stamp>` (logs only, `no_save`).
- At 3 runs per B200, a track D run takes 2.4 s per step, so the 6,000 steps need about 4 h of
  training. A 110M fp32 model's state is about 1.3 GB, so saving every 10 min costs little.
- I cancelled 27460856 while it was pending, put its three held items back at the front of
  `screens_d_b200`, and submitted 27483976 with the same settings plus `RESUME_EVERY_S=600`. The track A
  B200 worker (27441594, r22) keeps hourly saves: a 1.7B state is 20 GB, and r22 seed 3 has one.
- No other results since 15:50. r18 `d003rl_p256_lr3e-6` seed 1 runs on node4300 (27162158). A 005 seeds 2
  and 3 wait for 27412797 and 27412796 (pending, Resources). Both B200 workers wait on
  ReqNodeNotAvail.

## 2026-10-31 19:50 EDT: r18 `d003rl_p256_lr3e-6` seed 1 dev 0.148 (D 003 seed 1: 0.125); new H200 worker

r18 is D 003's RL from D 003's own pretrained models, with 256 prompts per step instead of 64. Same
time budget (14,040 s), H200, paired with D 003's seed.

| arm | seed | final dev | D 003 same seed | steps | informative groups (late) | entropy (late) | s per step |
|---|---|---|---|---|---|---|---|
| `d003rl_p256_lr3e-6` | 1 | 0.148 | 0.125 | 4,334 | 11-28 of 256 | 0.06-0.08 | 4.5 (sampling 4.1) |

- Dev by 500 steps: 0.073, 0.100, 0.103, 0.127, 0.137, 0.150, 0.147, 0.148, 0.148 (step 4,334). It
  gains 0.023 on its pair but flattens from step 3,000. Informative groups stay near 8 % of the batch,
  about the share D 003 had with 64 prompts, so 4x prompts gives 4x signal per step at about 4.3x the step
  time. That is close to break-even, so the gain is no more than one seed's noise (D 004's seeds span
  0.130-0.183).
- Seed 1 is one of six r18 items; the rest (`lr3e-6` seeds 2-3, `lr6e-6` seeds 1-3) stay in `h200`. A
  gating run needs about 0.19 (line 0.189).
- Worker 27162158 exited after it (less than 380 min left). I submitted 27487788 (same H200 settings,
  queue `h200`). With 27483976 (r25, now running on node5800-c1) and 27441594 that makes 6 GPU jobs.

## 2026-11-01 00:45 EDT: r25 resumed from its 10-minute saves; early dev ahead of D 004 for `shape05`

B200 workers 27441594 and 27483976 restarted on node5800-c1 at 00:17. r25's runs resumed from their
saved state (`shape02` at step 278, `shape05` at 270) instead of step 1, and they now save every
10 minutes. At three runs per B200 a step takes about 1.8 s, so seed 0 should finish around 03:30.

Dev ladder so far, against D 004 seed 0's own RL run at the same step (same pretrained weights):

| run | step 500 | step 1,000 | entropy now | informative groups of 64 now |
|---|---|---|---|---|
| D 004 seed 0 (control) | 0.083 | 0.087 | | |
| `shape02` (partial credit 0.2) | 0.065 | 0.093 | 0.06 at step 1,158 | 34 |
| `shape05` (partial credit 0.5) | 0.092 | 0.137 | 0.08 at step 1,061 | 36 |
| `ent05_t10` (entropy bonus) | 0.067 | | 0.16 at step 734 | 14 |

One seed at step 1,000 decides nothing, but `shape05` keeps about half its groups informative. D 004
had 1-10 by step 1,500. r22 `ent05_t10_lr2e-6` seed 3 resumed on 27441594 from its saved state at step 224 and is at
step 308.

## 2026-11-01 01:45 EST: A 005 seed 2 reruns; B200 workers preempted again

Clocks moved back to EST at 02:00 EDT; entries from here on use EST.

H200 worker 27412797 restarted on node4500 at 00:49 EST and is running A 005 seed 2 again. The
preempted attempt (trained to dev 0.36, stopped in its test ladder on 10-31) is in
`extra/interrupted_2_1` with its `note.txt`; no test score had been written. Both B200 workers were
preempted again around 01:40 EDT. r25 has saved every 10 minutes, so its runs resume where they
stopped: `shape02` at step 2,033, `shape05` at 1,530 (dev 0.135 at 1,500 against D 004's 0.102),
`ent05_t10` at 1,234.

## 2026-11-01 03:45 EST: A 005 seed 2 test 0.3821 (within budget); new H200 worker

A 005 seed 2's rerun (node4500, worker 27412797) finished at 03:40: test ladder 0.3821, pass@16
0.625, in-loop final dev 0.342, total_s 8,775 of 18,000 (within budget). Mean tokens 756 and 54 % of
outputs truncated, like seeds 0 and 1.

| seed | test ladder | within budget |
|---|---|---|
| 0 | 0.3995 | yes |
| 1 | 0.4076 | yes |
| 2 | 0.3821 | yes |
| 3 | waits for worker 27412796 | |
| 4 | 0.2899 | yes |

Four of five seeds: mean 0.370 ± 0.054, above A 004's 0.2671 ± 0.0565. Seed 3 is the last one.
27412797 then exited (less than 380 min left). New H200 worker 27530015 (same settings, queue `h200`,
head: r18 `lr6e-6` seed 1) keeps six GPU jobs: 27441594 running; 27483976, 27412796, 27487788,
26927305 and 27530015 pending.

## 2026-11-01 05:50 EST: r25 `shape05` leads D 004 by about 0.05 dev; r26 queued (its H200 gating screen and shaping 1.0)

r25 is about halfway (B200, three runs per GPU, step-capped at 6,000; worker 27483976 running again
since 00:17, resumed from its 10-minute saves). Dev ladder every 500 steps, seed 0 of each arm, against
D 004 seed 0's own dev at the same steps:

| step | D 004 seed 0 (no shaping) | `shape02` | `shape05` | `ent05_t10` |
|---|---|---|---|---|
| 500 | 0.083 | 0.065 | 0.092 | 0.067 |
| 1000 | 0.087 | 0.093 | 0.137 | 0.072 |
| 1500 | 0.102 | 0.090 | 0.142 | 0.077 |
| 2000 | 0.107 | 0.095 | 0.138 | 0.078 |
| 2500 | 0.105 | 0.105 | 0.163 | 0.080 |
| 3000 | 0.100 | 0.115 | 0.157 | 0.055 |
| 3500 | 0.093 | 0.115 | | |
| 6000 | 0.122 | | | |

- `shape05` (partial credit 0.5 for a wrong answer) leads at every eval from step 1,000, by 0.03-0.06.
  It keeps more informative groups (36 of 64 at step 3,400, against 1-10 for D 004 after step 1,500)
  and entropy 0.071.
- `shape02` is about level with D 004.
- `ent05_t10` is diverging: entropy rose to 0.49 and only 4 groups are informative. Its queued seeds 1
  and 2 are removed. Seed 0 runs to its end.

One seed, so far. Next:

- **r26 `d004rl_shape05_h200`, seeds 0-2, at the head of `h200`.** This is D 004's full RL (14,040 s,
  time-budgeted, H200) with shaping 0.5, from D 004's pretrained seeds. It is the screen that gates
  a track D record. Line: 0.189 for every seed (3-seed form). Paired with D 004's in-loop final dev:
  0.182 / 0.130 / 0.137.
- **r26 `d004rl_shape10`, seeds 0-1, in `screens_d_b200`.** As `shape05`, with shaping 1.0. Partial
  credit stays at most 0.99, below a correct answer's 1.

The B200 track A worker 27441594 kept resuming r22 `ent05_t10_lr2e-6` seed 3 from its only save
(step 224; hourly saves never came before the next preemption). I cancelled it and put the seed back
at the front of `screens_b200`. Its replacement, 27534692, uses the same settings with
`RESUME_EVERY_S=1800`. I used 1,800 s, not 600 s, because track A's state is about 20 GB per save.

Six GPU jobs:

- Running: 27483976, 27534692.
- Pending: 27412796 (holds A 005 seed 3), 27487788, 27530015, 26927305.

## 2026-11-01 09:50 EST: A 005 complete, 0.3400 ± 0.0815, validated, but not a new record (Welch p = 0.072)

A 005 seed 3's rerun (node4801, worker 27412796) finished at 09:35. Test ladder 0.2209, pass@16
0.474, in-loop final dev 0.197, total_s 7,618 (within budget). Its first attempt is in
`extra/interrupted_3_1` with a note.

| seed | test ladder | in-loop final dev | mean tokens | truncated | within budget |
|---|---|---|---|---|---|
| 0 | 0.3995 | 0.375 | 709 | 0.53 | yes |
| 1 | 0.4076 | 0.403 | 723 | 0.55 | yes |
| 2 | 0.3821 | 0.342 | 756 | 0.54 | yes |
| 3 | 0.2209 | 0.197 | 403 | 0.14 | yes |
| 4 | 0.2899 | 0.248 | 554 | 0.23 | yes |

- Mean of all five: 0.3400 ± 0.0815. `validate_record_v3.py` passes.
- Against A 004 (0.2671 ± 0.0565), the one-sided Welch test gives t = 1.64, df 7.1, p = 0.072.
  That is not significant, so **A 004 stays the track A (H200) holder**. A 005 is a valid record
  attempt that falls short, and its README says so.
- The r16 screen's five seeds were within 0.02 of each other on the dev ladder at step 400
  (0.397 ± 0.011). The record's seeds spread far wider: in-loop dev 0.197-0.403.
- Seeds that kept long outputs (about 54 % truncated) scored 0.38-0.41. Seeds 3 and 4 shortened
  their outputs (14 % and 23 % truncated) and scored 0.22 and 0.29. The step-400 stop doesn't prevent
  that drift; it just happens before step 400 on some seeds.

Track A next: A 002 still runs as queued (INBOX 10-31). Track D stays the priority. The r26 H200
gating screen is at the head of `h200`.

Workers: H200 worker 27487788 started at about 09:00 on node4300 and took r26
`d004rl_shape05_h200` seed 0. B200 worker 27483976 (r25) was preempted again and is pending. It
resumes from its 10-minute saves.

### r25 seed 0 finished (step 6,000): `shape05` 0.188 against 0.113 for the paired control

All four seed-0 arms reached step 6,000 on the B200, three runs per GPU. Dev ladder:

| step | `ctl` (D 004's RL, same packing) | `shape02` | `shape05` | `ent05_t10` |
|---|---|---|---|---|
| 1000 | 0.068 | 0.093 | 0.137 | 0.072 |
| 2000 | 0.085 | 0.095 | 0.138 | 0.078 |
| 3000 | 0.093 | 0.115 | 0.157 | 0.055 |
| 4000 | 0.092 | 0.137 | 0.172 | 0.063 |
| 5000 | 0.107 | 0.135 | 0.180 | 0.073 |
| 6000 | 0.113 | 0.140 | 0.188 | 0.072 |

- D 004 seed 0's own H200 run had 0.122 at step 6,000 and 0.182 at its end (about 15,000 steps).
- `shape05` matches that final number by step 6,000, still rising, with 36 of 64 groups
  informative and entropy 0.057.
- `shape02` gains about half as much. `ent05_t10` is closed.
- One seed per arm so far. `shape05` and `shape02` seeds 1-2 are running or queued, then
  `shape10` seeds 0-1.
- The r26 H200 gating screen (`d004rl_shape05_h200`) runs seeds 0 and 1 now, on 27412796 and
  27487788. Seed 2 heads `h200`.

### r22 `ent05_t10_lr2e-6`: four seeds 0.360 / 0.387 / 0.440 / 0.357 at step 500

Mean 0.386. That doesn't reach the 0.43 a new track A arm needs. Seed 4 is running (27534692);
after it, the arm closes and `screens_b200` is empty.

## 2026-11-01 11:50 EST: r25 seed 1 finished; r22 `ent05_t10_lr2e-6` closed; r27 queued

r25 seed 1 reached step 6,000 (B200, three runs per GPU). D 004's own dev at step 6,000, from its H200 record run, is the paired control.

| arm | seed 0 | seed 1 | mean |
|---|---|---|---|
| D 004's own run (control) | 0.122 | 0.115 | 0.119 |
| `shape02` | 0.140 | 0.185 | 0.163 |
| `shape05` | 0.188 | 0.140 | 0.164 |

At step 6,000, both shaping weights lead the control by about 0.045 on each of four seed pairs. The data can't tell 0.2 from 0.5 apart. Seed 2 of both arms is running.

The H200 gating screen (r26 `shape05_h200`, full D 004 RL budget) is not done. Seed 1 is at dev 0.155 at step 9,000 of about 14,000; D 004's seed 1 ended at 0.130. The promotion line is 0.189 on every seed. Seed 0's worker (27412796) was preempted about 10:30 and holds the item. Seed 1 should finish about 13:10 EST.

r22 `ent05_t10_lr2e-6` seed 4 finished at 0.470. The five seeds are 0.360 / 0.387 / 0.440 / 0.357 / 0.470, mean 0.403. That is below the 0.43 a new arm needs, so the arm is closed.

Both B200 workers on node5800-c1 (27483976 and 27534692) were preempted and requeued four times between 11:00 and 11:30 EST. r26 `shape10` seed 0 has no save yet: its worker saves every 1,800 s, and every attempt was stopped sooner. Slot 3 of the track D worker found its queue empty and exited. It comes back at the worker's next restart.

Next, r27 `d004rl_shape05_lr6e-6`, seeds 0-2: r25 `shape05` at twice the lr, to see whether the denser reward can take a larger step. It is queued on `screens_d_b200`.

## 2026-11-01 13:50 EST: r26 H200 seed 1 ends at 0.148, so shaping 0.5 does not promote; r28 queued

r26 `shape05_h200` seed 1 ran D 004's full RL budget on the H200 (15,183 steps) with shaping 0.5. Its final dev score is 0.148, against 0.130 for D 004's seed 1. The 3-seed promotion form needs 0.189 on every seed, so r26 cannot promote. Seeds 0 and 2 still run, as paired evidence of how much shaping adds over the full budget.

Seed 1's dev score held at 0.15-0.16 from step 8,500 on. At step 6,000, shaping led D 004 by 0.02-0.07 per seed. Over the full budget the lead shrank to 0.018, because D 004 keeps climbing to step 15,000 while the shaped run levels off.

| run (B200 runs stop at step 6,000) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D 004's own run at step 6,000 (control) | 0.122 | 0.115 | 0.075 |
| r25 `shape02` | 0.140 | 0.185 | 0.148 |
| r25 `shape05` | 0.188 | 0.140 | running |
| r26 `shape10` | 0.123 | | |
| D 004 final (H200, full budget) | 0.182 | 0.130 | 0.137 |
| r26 `shape05_h200` final (H200, full budget) | running | 0.148 | running |

Shaping 1.0 is no better than the control on seed 0, so weights from 0.2 to 0.5 are the useful range. Shaping makes RL climb faster but levels off near 0.15-0.16; a record needs something that lifts that level.

The shaped runs end with about 30 informative groups out of 64. Next, r28 `d004rl_shape05_g16`: group 16 with 32 prompts, the same rollouts per step. Larger groups should leave more groups with a nonzero advantage on hard puzzles. Seed 0 goes to `screens_b200`, which would otherwise empty after `shape10` seed 1, and seeds 1-2 to `screens_d_b200`, after r27 seed 2.

## 2026-11-01 14:45 EST: r25 complete; `shape10` closed

Dev scores at step 6,000. The control is D 004's own run at the same step.

| run | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| D 004 (control) | 0.122 | 0.115 | 0.075 | 0.104 |
| `shape02` | 0.140 | 0.185 | 0.148 | 0.158 |
| `shape05` | 0.188 | 0.140 | 0.153 | 0.160 |
| `shape10` | 0.123 | 0.118 | | 0.121 |

Shaping at 0.2 or 0.5 beats the control on all six seed pairs, by 0.05 on average. Shaping 1.0 matches the control on both of its seeds, so `shape10` is closed. Whether the faster climb lasts over the full budget is r26's question (seed 1 gained only 0.018; LOG 13:50).

## 2026-11-01 15:45 EST: r27 seed 0 0.175; r29 queued

r27 `shape05_lr6e-6` seed 0 reached 0.175 at step 6,000. At that step, r25 `shape05` (lr 3e-6) scored 0.188 and D 004 scored 0.122. It climbs faster early: 0.145 at step 1,500. Seeds 1-2 are running.

The worker queues would empty within about 4 h, so I queued r29 `d004rl_shape05_lr1e-5`, seeds 0-2, to bound the lr sweep from above. Seed 0 goes to `screens_b200` and seeds 1-2 to `screens_d_b200`.

## 2026-11-01 16:45 EST: r28 seed 0 0.138, r27 seed 1 0.152; new B200 worker 27585540

Dev scores at step 6,000:

| run | seed 0 | seed 1 |
|---|---|---|
| D 004 (control) | 0.122 | 0.115 |
| r25 `shape05` (lr 3e-6) | 0.188 | 0.140 |
| r27 `shape05_lr6e-6` | 0.175 | 0.152 |
| r28 `shape05_g16` (group 16 x 32 prompts) | 0.138 | running |

So far, doubling the lr is about even with r25 (0.164 against 0.164 over two seeds). Larger groups did worse on seed 0. r27 seed 2 and r28 seeds 1-2 are running.

B200 worker 27534692 ended normally at 16:17. Its time limit left less than 240 min, so it exited after r28 seed 0. Its replacement is 27585540, with the same settings but RESUME_EVERY_S 600, because node5800-c1 preempts often (LOG 11:50). It runs `screens_b200`, starting with r29 seed 0.

## 2026-11-01 17:45 EST: r26 H200 seed 2 ends at 0.183 (D 004: 0.137); r27 complete; r30/r31 queued

r26 `shape05_h200` seed 2 (shaping 0.5, D 004's full time budget on the H200) ended at 0.183 on the dev ladder at step 13,718. D 004's seed 2 ended at 0.137.

| full budget, H200 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D 004 (final dev) | 0.182 | 0.130 | 0.137 |
| r26 `shape05_h200` | pending | 0.148 | 0.183 |

Shaping is ahead on both pairs, by 0.018 and 0.046. The promotion line is 0.189, and seed 1 is under it, so the 3-seed form can't be met. Seed 0 (preempted job 27412796, about step 2,500) stays queued for the paired comparison.

Dev scores at step 6,000 (B200 packed):

| run | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| D 004 (control) | 0.122 | 0.115 | 0.075 | 0.104 |
| r25 `shape05` (lr 3e-6) | 0.188 | 0.140 | 0.153 | 0.160 |
| r27 `shape05_lr6e-6` | 0.175 | 0.152 | 0.180 | 0.169 |
| r28 `shape05_g16` | 0.138 | 0.158 | running | |

r27 is complete. Against r25 it is -0.013, +0.012 and +0.027 by seed: about level, maybe slightly ahead. r28 seed 1 beat r25's seed 1 (0.158 against 0.140), so groups of 16 are 1-1 so far. r28 seed 2 and r29 (lr 1e-5) are running.

All of these runs end with one guess per answer: L1 near 0.6 and L6 near 0.2, with L3-L5 under 0.1. The open question is whether anything lifts the level reached at the end of a full run. The queues were empty, so I added two runs at the full length, step-capped at 15,000 steps (about what D 004's RL reaches on the H200, about 6.3 h packed):

- r30 `d004rl_shape05_lr6e-6_s15k`: r27 at full length.
- r31 `d004rl_shape05_lr6e-6_cos_s15k`: the same with a cosine lr decay to 0.1x over the 15,000 steps. Track D has used a constant lr so far.

Seeds 0-2 of each are interleaved in `screens_d_b200`; r30 seed 2 is in `screens_b200`. The comparison is with r26's H200 finals. r30/r31 are step-capped, so the GPU type doesn't change them. r26 was time-capped, and its seeds reached 13,700-15,200 steps. If r29 shows that lr 1e-5 is clearly better, I'll swap the queued r30/r31 items for that lr.

## 2026-11-01 18:45 EST: r28 complete (groups of 16 no better); worker 27598456 queued behind 27483976

r28 `shape05_g16` seed 2 reached 0.168 at step 6,000. r28 is complete:

| run, dev at step 6,000 | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| r25 `shape05` (group 8 x 64 prompts) | 0.188 | 0.140 | 0.153 | 0.160 |
| r28 `shape05_g16` (group 16 x 32 prompts) | 0.138 | 0.158 | 0.168 | 0.155 |

Pairs are -0.050, +0.018 and +0.015: no gain from larger groups at the same rollouts per step. r28 is closed.

Track D worker 27483976 has less than its 180-min reserve left (its 8:40 limit ends 21:05 EST), so slot 1 exited after r28 seed 2 instead of taking r31 seed 0. Its other two slots finish r29 seeds 1-2. I submitted 27598456, which has the same settings (PER_GPU 3, `screens_d_b200`), with `--dependency=afterany:27483976`. It can't run while 27483976 holds its GPU, so the cap stays at 6. Until then one B200 slot is idle.

## 2026-11-01 19:45 EST: r29 (lr 1e-5) complete, mean 0.174 at step 6,000; the lr sweep with shaping 0.5 is flat to slightly up

Dev scores with shaping 0.5, B200 packed, at steps 1,500 / 3,000 / 6,000 (×1000 for the first two):

| run | seed 0 | seed 1 | seed 2 | mean at 6,000 |
|---|---|---|---|---|
| D 004 (control, lr 3e-6, no shaping) | 0.122 | 0.115 | 0.075 | 0.104 |
| r25 lr 3e-6 | 142 / 157 / 0.188 | 88 / 120 / 0.140 | 118 / 138 / 0.153 | 0.160 |
| r27 lr 6e-6 | 145 / 153 / 0.175 | 110 / 112 / 0.152 | 122 / 142 / 0.180 | 0.169 |
| r29 lr 1e-5 | 143 / 148 / 0.170 | 143 / 163 / 0.172 | 143 / 135 / 0.182 | 0.174 |

- Against r25, r29 gives -0.018, +0.032 and +0.029 by seed. Its seeds are closer together (0.170-0.182), and all three are at 0.143 by step 1,500.
- Each lr step adds only about 0.005-0.01 to the mean, inside the seed spread. All three arms are flat from about step 4,500 (r29: 0.160-0.183 at step 4,500, and 0.170-0.182 at 6,000).
- r29 closes the step-6,000 lr sweep. A higher lr gets to the same level sooner rather than to a higher one.

27483976 ended at 00:39 UTC after r29 seeds 1-2. 27585540 finished r29 seed 0 and took r30 seed 2. 27598456 (3 slots) waits for a B200 and takes r31 seed 0, r30 seed 0 and r31 seed 1 next. I'm keeping r30/r31 at lr 6e-6 so that they differ only in the cosine. The queue holds about 9 h on 4 slots. If the cosine helps, a cosine from lr 1e-5 is the next arm.

## 2026-11-01 21:45 EST: r18 `d003rl_p256_lr6e-6` seed 1 dev 0.142 (D 003 seed 1: 0.125)

r18 is D 003's RL with 256 prompts per step, from D 003's pretrained seeds, at D 003's full time budget on the H200.

| arm | seed 1 final dev | steps |
|---|---|---|
| D 003 (64 prompts, lr 3e-6) | 0.125 | |
| r18 `lr3e-6` (256 prompts) | 0.148 | 4,334 |
| r18 `lr6e-6` (256 prompts) | 0.142 | 3,857 |

Dev by 500 steps peaked at 0.148 at step 3,000 and ended at 0.142. So doubling the lr adds nothing at 256 prompts. Both r18 arms land in the 0.14-0.15 band that every shaping and lr arm reaches. Remaining in `h200`: `lr3e-6` seed 2 and `lr6e-6` seed 2 (running), then `lr3e-6` seed 3, `lr6e-6` seed 3 and A 002 seeds 0-4.

## 2026-11-01 22:45 EST: r18 `d003rl_p256_lr3e-6` seed 2 dev 0.147 (D 003 seed 2: 0.173)

| arm | seed 1 final dev | seed 2 final dev | steps (seed 2) |
|---|---|---|---|
| D 003 (64 prompts, lr 3e-6) | 0.125 | 0.173 | |
| r18 `lr3e-6` (256 prompts) | 0.148 | 0.147 | 3,908 |

Dev by 500 steps rose steadily to 0.158 at step 3,500 and ended at 0.147. On this seed, 256 prompts lands 0.026 below D 003, after landing 0.023 above it on seed 1. So far r18 sits in the same 0.14-0.16 band as every other arm, with no paired gain. 27530015 took `lr3e-6` seed 3. `lr6e-6` seed 2 is still running on 27487788.

## 2026-11-01 23:50 EST: maintainer intervention (review PLAN before more track D RL knobs); r30 seed 2 reaches dev 0.220 at step 15,000; D3 next

**Maintainer intervention.** INBOX 11-01 23:30 (commit e7b49b5): about a dozen track D RL arms since 10-31 all ended in the 0.14-0.18 band, in one-guess mode. Before queuing more screens that only vary RL settings, review PLAN.md's untried track D items, then either queue the best of them or say here why the RL side is still the better bet.

**r30 seed 2** (`d004rl_shape05_lr6e-6_s15k`, from D 004's pretrained seed 2, alone on a B200, step-capped at 15,000) finished at 22:48:

| step | 3,000 | 6,000 | 7,500 | 9,000 | 11,500 | 13,000 | 15,000 |
|---|---|---|---|---|---|---|---|
| dev | 0.135 | 0.168 | 0.197 | 0.203 | 0.225 | 0.227 | 0.220 |

| from D 004 pretrained seed 2 | final dev | steps |
|---|---|---|
| D 004 (lr 3e-6, no shaping, H200) | 0.137 | time-capped |
| r26 (lr 3e-6, shaping 0.5, H200) | 0.183 | time-capped, about 14,000-15,000 |
| r30 (lr 6e-6, shaping 0.5, B200) | 0.220 | 15,000 |

This is the first track D run above the 0.189 line. It kept rising after step 6,000, where all the step-6,000 screens stopped. It is still one-guess (last training batch: 23 of 64 groups informative, entropy 0.044). One seed, so nothing follows yet. r30 seeds 0-1 and r31 seeds 0-2 finish by about 09:00.

**PLAN review, track D:**
- D1 (fp32 and lr): done. It gave D 002.
- D2 (math pretraining mixes): done, all closed. dm26 gave D 002. dm51 was +0.010 (LOG 10-25 14:30). The FineMath pilot ended at 0.135.
- D3 (SFT on public worked solutions before RL): **untried.** It was planned on 10-31 but never built. It is the item aimed directly at the one-guess mode. Pretraining on FineWeb-Edu plus DeepMind question-answer pairs never shows the model a chain of intermediate arithmetic, so RL has no search behaviour to reinforce. GSM8K's published solutions are human-written chains of arithmetic (`48/2 = <<48/2=24>>24`).
- D4 (process rewards for correct intermediate arithmetic): untried, but it needs a policy that writes intermediate steps, so it comes after D3.
- D5 (model size): no reason yet. A larger model in the same data regime would still guess once.

**Decision:** No new RL-knob screens. The running r30/r31 runs finish as queued, since they test whether full length is what the 6,000-step screens missed. The next build is D3:
1. Download GSM8K train (`openai/gsm8k`, `main` config, revision pinned and md5 recorded) to `~/orcd/pool/data/gsm8k`, as published.
2. Write an off-by-default SFT step before RL, separate from `scripts/sft_warmstart.py` (that script trains on generator solutions, which v3 forbids). It runs next-token loss on GSM8K question → solution text, from D 004's pretrained seeds, for a fixed step count.
3. Screen r32: D 004 pretrained, then SFT, then r30's RL (lr 6e-6, shaping 0.5, 15,000 steps), on B200 seeds 0-2. Check at step 0 whether the policy writes equations in `<think>`, and compare with r30 by matched step.

**GPUs:** 27585540 left when `screens_b200` emptied. New B200 worker 27627658 (PER_GPU=2, queue `screens_d_b200`) takes r30 seed 1 and r31 seed 2, which were waiting for 27598456 to finish at about 03:00. Cap: 3 running plus 3 pending (27412796, 26927305, 27627658) = 6.

## 2026-11-02 00:30 EST: D3 built; r32 (GSM8K SFT, then r30's RL) queued

- Data: GSM8K train, `openai/gsm8k` config `main` at revision `740312add88f781978c0658806c59bc2815b9866`, downloaded as published to `~/orcd/pool/data/gsm8k/main/train-00000-of-00001.parquet` (md5 979c9d17cd5753bc11d745e945289382, 7,473 rows). It is converted field for field to `train.jsonl` (md5 a3a079b998bb13fa3aa9fc479e1e6dd0) so the jobs need no pyarrow. The numbers-game keyword filter finds 2 rows, both false positives ("24 games" of a season, "make 24 pies"). They are dropped anyway, as the rule's keyword filter allows.
- `scripts/research/sft_public.py`: SFT with completion-only loss. Each row is the r1 chat template with the GSM8K question in place of the Countdown request, then `<think> {published solution before ####}</think>\n<answer>{published final number}</answer>`. Nothing is generated: the only added text is the benchmark's own template. Shuffled epochs, AdamW, 20-step warm-up, saved in bf16 like the pretrained models. Tests are in `tests/test_sft_public.py`.
- `scripts/slurm/screen_seed.sh`: an optional `"screen_sft"` config block runs that SFT on the starting model before GRPO, into `$CKPT_DIR/<dir>/seeds/<seed>/sft`. A resumed seed reuses it, and it is deleted when the seed finishes. Without the block, nothing changes. The file was replaced by `mv`, so running workers keep reading the old copy. `grpo.py` is untouched. Ruff is clean and pytest passes (139).
- r32 `d004rl_gsm8k300_shape05_lr6e-6_s15k`: D 004's pretrained seed, then 300 SFT steps of 64 (about 2.6 epochs, lr 1e-4), then r30's RL unchanged. Seeds 0-2 are on `screens_d_b200`, with seed 0 first so that a bug shows up early. I'll read three things: the step-0 dev eval and whether the completions write equations in `<think>`, the informative-group count against r30's 20-25, and dev by matched step against r30.

## 2026-11-02 00:55 EST: r32 seed 0 failed at import; fixed and requeued

r32 seed 0 started at 23:53 on worker 27598456 and stopped at once. `sft_public.py`, run as a file, could not import `rlvr_speedrun`, because only `-m` puts the checkout on `sys.path`. No GPU time was used. `screen_seed.sh` now runs it with `PYTHONPATH="$PWD"`, and `--help` works that way with the worker venv. The failed stdout is kept as `seeds/0/stdout.failed.importerror.log`. Seed 0 is back at the front of `screens_d_b200`, and it starts when a B200 slot frees.

## 2026-11-02 01:55 EST: r18 `d003rl_p256_lr6e-6` seed 2 dev 0.158 (D 003 seed 2: 0.173)

| arm | seed 1 final dev | seed 2 final dev | steps (seed 2) |
|---|---|---|---|
| D 003 (64 prompts, lr 3e-6) | 0.125 | 0.173 | |
| r18 `lr3e-6` (256 prompts) | 0.148 | 0.147 | 3,908 |
| r18 `lr6e-6` (256 prompts) | 0.142 | 0.158 | 4,342 |

Dev by 500 steps: 0.078, 0.095, 0.122, 0.132, 0.137, 0.148, 0.160 (step 3,500), 0.157, then 0.158 at the end. 11 of 256 groups were informative at the end, with entropy 0.089. Over two seeds, 256 prompts gains on D 003's weak seed 1 and loses on its strong seed 2, for both lrs. That is no paired gain, and both arms stay in the 0.14-0.16 band. Seed 3 of each arm is running (27487788, 27530015). After that, r18 closes and the `h200` queue moves on to A 002 seeds 0-4.

## 2026-11-02 03:05 EST: lr 6e-6 at full length clears the D line on 4 B200 seeds; r33 H200 promotion screen queued; r32 SFT runs

r30 seed 0, r31 seeds 0-1 and r18 `lr3e-6` seed 3 finished. Final dev on the dev ladder. Every arm starts RL from the same D 004 pretrained seed.

| pretrained seed | D 004 (lr 3e-6, no shaping, H200) | r26 (lr 3e-6, shaping 0.5, H200, time-capped) | r30 (lr 6e-6, shaping 0.5, constant, 15,000 steps, B200) | r31 (same, cosine) |
|---|---|---|---|---|
| 0 | 0.182 | pending | 0.192 | 0.188 |
| 1 | 0.130 | 0.148 | running | 0.217 |
| 2 | 0.137 | 0.183 | 0.220 | running |

- Matched by seed, lr 6e-6 beats r26 by +0.069 (seed 1, r31) and +0.037 (seed 2, r30). The 4 finished lr 6e-6 runs average 0.204. The lowest, 0.188, is on D 004's best seed.
- Constant and cosine are within 0.004 of each other on seed 0. Cosine's curves are a little lower until about step 9,000 and catch up by the end.
- Dev by 1,000 steps keeps rising to the end, as on r30 seed 2. For example, r31 seed 1 goes 0.158 (step 4,500), 0.187 (7,500), 0.208 (10,500), 0.217 (15,000). The early-stopped screens (6,000 steps or fewer) missed this. That is why every shaping and lr arm looked like it sat at 0.14-0.18.
- r18 `lr3e-6` seed 3: 0.177 at step 4,340. No D 003 pair for seed 3. r18 closes when `lr6e-6` seed 3 finishes.

**Decision:** queue r33 `d004rl_shape05_lr6e-6_h200` on `h200`, seeds 0-4, ahead of A 002 seeds 1-4. It is r26's config with only lr changed to 6e-6 (constant, time-capped at 14,040 s, vllm memory 0.35). It is the H200 gating screen in the 5-seed form: mean ≥ 0.189, 4 of 5 seeds above 0.159, and at most one collapsed. The D 004 seeds 3-4 have no lr 6e-6 run yet, so I'm using the 5-seed form, not the 3-seed one. I'm using constant lr rather than cosine because it doesn't depend on how many steps the GPU fits into the time budget. Track D comes first per INBOX 10-31. A 002 seed 0 is running on 27530015, and seeds 1-4 follow r33.

**r32 (GSM8K SFT, then r30's RL):** all three seeds started on 27598456 after the import fix. The SFT phase works. 7,471 rows, 300 steps of 64, about 60 s. Loss went from 2.3-2.45 at step 1 to 0.51-0.53 at step 300 (about 2.6 epochs). No dev evals have been written yet.

r30 seed 1 and r31 seed 2 were preempted on 27627658, and both resumed (from steps 3,409 and 3,627).

## 2026-11-02 05:00 EST: r31 seed 2 dev 0.197; r34 (lr 1e-5, 15,000 steps) queued on B200

| pretrained seed | D 004 | r26 (lr 3e-6, H200) | r30 (lr 6e-6, constant) | r31 (lr 6e-6, cosine) |
|---|---|---|---|---|
| 0 | 0.182 | pending | 0.192 | 0.188 |
| 1 | 0.130 | 0.148 | 0.175 at step 14,500 (running) | 0.217 |
| 2 | 0.137 | 0.183 | 0.220 | 0.197 |

r31 seed 2 rose from 0.150 at step 4,500 to 0.185 at step 9,500 and 0.197 at step 15,000. At the end, 22 of 64 groups were informative, with entropy 0.067. The 5 finished-or-nearly lr 6e-6 runs average about 0.194. Seed for seed, constant and cosine are within 0.03 of each other, in both directions. All five beat r26 on the same seed, but two are only just at the 0.189 line. r33 on the H200 decides.

**Next:** r34 `d004rl_shape05_lr1e-5_s15k`, r30 at lr 1e-5, seeds 0-2, queued in `screens_d_b200`. At step 6,000, lr 1e-5 was 0.174 against 0.169 for lr 6e-6. It hasn't been run at full length, and full length is where lr 6e-6 pulled ahead. 27627658's second slot left on the empty queue. When r30 seed 1 finishes, that worker ends and I'll submit a B200 worker for r34. Submitting now would go past 6 GPUs.

## 2026-11-02 05:55 EST: r30 seed 1 dev 0.180; r18 closes; r33 seed 0 started

| pretrained seed | D 004 | r26 (lr 3e-6, H200) | r30 (lr 6e-6, constant) | r31 (lr 6e-6, cosine) |
|---|---|---|---|---|
| 0 | 0.182 | pending | 0.192 | 0.188 |
| 1 | 0.130 | 0.148 | 0.180 | 0.217 |
| 2 | 0.137 | 0.183 | 0.220 | 0.197 |
| mean | 0.150 | | 0.197 | 0.201 |

All 6 lr 6e-6 runs are done, with a mean of 0.199. Five of the six are at or above 0.188. Seed 1 under constant lr is the low one, at 0.180. It moved slowly from step 4,500 (0.162) and climbed late (0.178 at step 11,500). All six beat r26 on the same seed, by +0.032 to +0.069. Constant and cosine are equal within seed noise (means 0.197 vs 0.201), so r33 keeps constant lr.

**r18 closes.** `lr6e-6` seed 3 finished at 0.165 at step 3,486 (7 of 256 groups informative, entropy 0.044). Over seeds 1-3, 256 prompts per step gave 0.148 / 0.147 / 0.177 at lr 3e-6 and 0.142 / 0.158 / 0.165 at lr 6e-6. That is no better than 64 prompts, and it fits fewer steps in the budget.

27487788 took r33 seed 0 at 05:40 (H200). 27627658's remaining slot took r34 seed 0. r34 seeds 1-2 wait for a B200 slot. 6 GPU jobs, at the cap.

## 2026-11-02 20:05 EST: r32 (GSM8K SFT first) ties r30; r34 lr 1e-5 seed 0 0.182; all workers preempted; r35 queued

This loop was suspended from 06:51 to 19:50, so nothing was checked during that gap. All six workers (27412796, 27487788, 27530015, 27598456, 27627658, and 26927305) are back in the queue, pending on priority. The last line of r33 seed 0's log is at 06:57, at step 7,459 of its time-capped run (dev 0.180 at step 7,000). On restart, each H200 worker redoes the item it was running, so r33 seed 0 starts again from the beginning. Nothing needs to be requeued by hand.

Track D, final dev at step 15,000 (B200), on D 004's pretrained seeds:

| pretrained seed | r30 (lr 6e-6) | r32 (300 steps of GSM8K SFT, then r30's RL) | r34 (lr 1e-5) |
|---|---|---|---|
| 0 | 0.192 | 0.187 | 0.182 |
| 1 | 0.180 | 0.203 | (preempted, waiting) |
| 2 | 0.220 | 0.220 | (preempted, waiting) |
| mean | 0.197 | 0.203 | |

- **r32 (PLAN D3).** Paired differences against r30 are -0.005, +0.023 and 0.000, with a mean of +0.006. That is within seed noise. It kept more groups informative to the end (29-34 of 64, against about 20-25 for r30), at about the same entropy (0.048-0.058), but that did not turn into a higher score. GSM8K SFT is not worth its extra step for a record. Close it unless a later arm gives a reason to pair it.
- **r34 seed 0.** 0.182, against 0.192 for r30 on the same seed. The curve looks like r30's (0.202 at step 12,500, then down). Seeds 1-2 resume when 27627658 runs again.
- **Next: r35, groups of 16 on 32 prompts per step.** That is r30 with the same 512 rollouts per step. The question is whether bigger groups find more informative groups once most of a group of 8 agrees. 3 seeds are queued on `screens_d_b200`, step-capped at 15,000.

Holding for r33. The track D record decision waits on its 5 seeds (H200, time-capped, 5-seed form).

## 2026-11-03 03:50 EST: r34 (lr 1e-5) complete, ties r30; r36 (training temperature 1.2) queued

Track D, final dev at step 15,000 (B200), on D 004's pretrained seeds:

| pretrained seed | r30 (lr 6e-6) | r34 (lr 1e-5) | difference |
|---|---|---|---|
| 0 | 0.192 | 0.182 | -0.010 |
| 1 | 0.180 | 0.173 | -0.007 |
| 2 | 0.220 | 0.245 | +0.025 |
| mean | 0.197 | 0.200 | +0.003 |

r34 is a tie with r30. Seed 2 at 0.245 is the best single track D dev score so far, but seeds 0 and 1 are a little lower than r30, and seed 1 fell from 0.187 at step 12,000 to 0.173 at the end. At the end, 26-29 of 64 groups were informative, with entropy 0.046-0.056, about the same as r30. With shaping 0.5, lr from 6e-6 to 1e-5 gives the same result at full length. r33 keeps lr 6e-6, and r34 closes.

r35 (groups of 16 on 32 prompts) is running: seed 2 is on 27627658, and seeds 0-1 are held by 27598456, which was preempted again. Seeds 0-2 are at about step 5,500-7,500, and their dev scores (0.150-0.158) are in line with r30 at the same steps.

**Next: r36 `d004rl_shape05_lr6e-6_t12_s15k`.** This is r30 with the training rollouts sampled at temperature 1.2. The dev eval sampler does not read this setting, so dev scores stay comparable. By the end, entropy is down to about 0.05 and only a third of the groups are informative. A hotter sampler should keep more groups mixed. The risk is a gap between training at 1.2 and the record's eval at T=1. Seeds 0-2 are queued on `screens_d_b200`. 27627658's free slot already exited on the empty queue, so r36 starts when 27598456 resumes or when 27627658 finishes r35 seed 2. 6 GPU jobs, at the cap.

## 2026-11-03 05:50 EST: r35 (groups of 16) seed 2 ends at 0.182, against 0.220 for r30

r35 `d004rl_shape05_lr6e-6_g16p32_s15k` seed 2 reached step 15,000 at 0.182 on the dev ladder (0.150 at step 9,500, 0.167 at step 12,500). r30 got 0.220 on the same pretrained seed. At the end, 19 of 32 groups were informative, with entropy 0.061. That is a larger share than r30's groups of 8, but the score is lower. Seeds 0 and 1 are at step 10,300 (0.163 and 0.175 at step 9,500) on 27598456. r36 (training temperature 1.2) seeds 0 and 1 started at 03:55 and 05:35, and seed 2 is queued. 6 GPU jobs.

## 2026-11-03 07:50 EST: r35 (groups of 16) complete, ties r30; r37 (entropy floor) queued

Track D, final dev at step 15,000 (B200), on D 004's pretrained seeds:

| pretrained seed | r30 (groups of 8 on 64 prompts) | r35 (groups of 16 on 32 prompts) | difference |
|---|---|---|---|
| 0 | 0.192 | 0.213 | +0.021 |
| 1 | 0.180 | 0.190 | +0.010 |
| 2 | 0.220 | 0.182 | -0.038 |
| mean | 0.197 | 0.195 | -0.002 |

r35 ties r30 and closes. Like r34 (lr 1e-5) and r31 (cosine), it moves single seeds by 0.01-0.04 in both directions and leaves the mean where it was. All three r35 seeds were still rising at step 12,500-15,000. At the end, 17-19 of 32 groups were informative.

r36 (training temperature 1.2) at step 8,000: 0.180 and 0.177 on seeds 0 and 1, with entropy 0.068-0.095 and 22-26 of 64 groups informative. That entropy is higher than r30's, and the scores are at or a little above r30 at the same step. Seed 2 started at 07:39.

r33 seed 0 (the H200 gating screen) is running again on 27487788, at 0.215 at step 12,500 and 0.207 at step 13,000.

**Next: r37 `d004rl_shape05_lr6e-6_ent05_s15k`.** This is r30 with the entropy floor from track A (coef 0.05 below entropy 0.1). It is a second way to keep the policy from settling, alongside r36's hotter sampler. Seeds 0-2 are queued on `screens_d_b200` and start as r36's seeds finish. 6 GPU jobs.

## 2026-11-03 08:50 EST: r33 (H200 gating screen) seed 0 ends at 0.205

r33 `d004rl_shape05_lr6e-6_h200` seed 0 finished at 07:49. It ran D 004's full RL time budget on the H200 (13,728 steps) and ended at 0.205 on the dev ladder (0.215 at step 12,500, 0.212 at step 13,500). At the end, 24 of 64 groups were informative, with entropy 0.047. On the same pretrained seed, D 004 got 0.182 and r30 on B200 got 0.192. 0.205 is above the 0.189 line and above the holder's dev mean of 0.159.

This attempt resumed at step 7,115 from the screen's own resume checkpoint, which was written before the 11-02 preemption, with the time budget carried over. So the earlier note that seed 0 starts again from the beginning was wrong. The interrupted stdout is kept beside the seed. 27487788 took r33 seed 1 at 07:49. Seeds 2-4 are queued on `h200`, ahead of A 002 seeds 1-4. The record decision waits for all 5 seeds (5-seed form: mean ≥ 0.189, at least 4 of 5 above 0.159, at most 1 collapsed).

## 2026-11-03 09:50 EST: r36 (training temperature 1.2) seeds 0-1: 0.203 and 0.195

| pretrained seed | r30 (temperature 1.0) | r36 (temperature 1.2) | difference |
|---|---|---|---|
| 0 | 0.192 | 0.203 | +0.011 |
| 1 | 0.180 | 0.195 | +0.015 |
| 2 | 0.220 | running (started 07:39) | |

Both finished seeds beat r30 by a little, and r30's weakest seed (1) ends above the 0.189 line. At the end, 23-31 of 64 groups were informative, with entropy 0.055-0.062. That is about r30's level, so the hotter sampler's extra spread is gone by step 15,000. Seed 1 dipped to 0.165 at step 12,500 before ending at 0.195. Gains of +0.01 are within the seed noise seen in r31, r34 and r35, so seed 2 decides it. r37 (entropy floor) seeds 0 and 1 started at 09:25 and 09:39, and seed 2 is queued.

## 2026-11-03 11:50 EST: A 002 seed 0 on test is 0.252; preemption on node4500

Record A 002 `002_fp32_skip_cosine` seed 0 finished at 10:57 on the H200 (5,485 steps, 17,756 s wall).

| seed | final dev | test ladder (T 0.5, 16 samples) | A 004 test mean | A promotion line |
|---|---|---|---|---|
| 0 | 0.243 | 0.252 | 0.267 | 0.285 |

Test pass@1 by level: L1 0.764, L2 0.216, L3 0.162, L4 0.092, L5 0.034, L6 0.243. The seed sits 0.015 below A 004's mean. The seed's earlier preempted attempt is in `extra/interrupted_0_1`, as the INBOX rule says. Seeds 1-4 stay queued on `h200` behind r33. Every seed counts.

At 11:02-11:04 both H200 workers on node4500 were preempted:
- 27487788 was running r33 seed 1, at step 10,148 with dev 0.173 at step 10,000. It resumes from its own checkpoint when the job runs again.
- 27530015 had started r33 seed 2 at 10:57 and ran for 7 minutes.

Both are `--requeue` jobs and are pending again (Resources). B200 worker 27598456 (r36 seed 2, r37 seed 1) was also preempted and is pending. 27627658 (r37 seed 0) is still running.

**Queued:** r30 `d004rl_shape05_lr6e-6_s15k` seeds 3-4 on `screens_d_b200`, behind r37 seed 2. r30 so far pairs only on D 004's pretrained seeds 0-2. With seeds 3-4, all five pretrained seeds pair against every later arm, including r36 and r37 if either needs more seeds. The GPU count stays at 6.

## 2026-11-03 12:50 EST: maintainer intervention: the 11-05 H100 box is back on (INBOX 11-03 13:30)

**Maintainer intervention (INBOX 2026-11-03 13:30 EST, logged as such).** This lifts INBOX 10-25 22:00 for the 2026-11-05 block only.

The block:
- 8x H100 in us-east-2b, at `zjason@16.58.142.215`, with template `rlvr-box-cr-070d093aa92f43751`.
- It runs from 11:30Z 11-05 to 11:30Z 11-06, which is 06:30 EST 11-05 to 06:30 EST 11-06.

The terms:
- Screens only, split about 50/50 between track A and track D. Records and test ladders stay on Engaging.
- The box's GPUs don't count toward the 6-GPU limit.
- Scheduled launch at +1, +10 and +25 min. The +30 min `run-instances` fallback for this template is allowed. No other AWS calls and no AWS spending.
- `box_sync.sh from` every wake, and a last time before 11:00Z 11-06.
- While the box is up, I wake every 20-30 min, which overrides the 6-hour full-wake rule for that day. The 11-18 and 11-28 blocks stay off.

Prep plan, to finish by 11-05 05:30 EST (the first 20-30 min wake):
1. **Packing on an 80 GB H100.**
   - Track A fp32 needs about 51 GB for the trainer, so one run per GPU (memory notes from 10-25).
   - Track D (110M) on B200 runs at `vllm_gpu_memory` 0.35 of 192 GB. On an H100 that share must shrink so 2-4 runs fit.
   - I'll set the D packing from the B200 runs' measured peak memory. `pack_probe.sh` on box GPU 7 in the first hour checks it.
2. **Step caps.** Track D arms reuse the `s15k` configs (15,000 RL steps from D 004's pretrained models; an H200 does 13,728 in the budget), each batch with its own r30 control on the same packing. Track A arms get a per-experiment cap with their own control, as r15 did.
3. **What to run.**
   - Track D: the most promising of r36 (temperature 1.2) and r37 (entropy floor) once their B200 seeds finish, plus PLAN items that need no new data.
   - Track A: A 005's recipe and its seed spread (0.221-0.408 on test, two seeds shortening their outputs). The box's four A GPUs go to stabilising it.
4. **Copies.** The checkout, `~/orcd/pool/data/...` that the configs read, `~/.cache/huggingface`, and D 004's `pretrained` checkpoints (five seeds). They go out by `box_sync.sh to` from a mit_quicktest job once the box answers ssh.
5. The venv freeze file is fixed (`agent/vllm-venv-freeze.txt`). If the venv still fails on the box, fix it there.

Engaging this hour:
- r26 seed 0 (27412796) started at 12:05 on node4301.
- Both H200 workers and B200 worker 27598456 are still pending after the 11:02 preemptions.
- 27627658 runs r37 seed 0.

## 2026-11-03 13:50 EST: r37 (entropy floor) seed 0 ends at 0.205

| pretrained seed | r30 (no floor) | r37 (entropy floor 0.05 below 0.1) | difference |
|---|---|---|---|
| 0 | 0.192 | 0.205 | +0.013 |
| 1 | 0.180 | held by 27598456 (preempted at 11:02, pending) | |
| 2 | 0.220 | queued | |

r37 seed 0 dev by 2,000 steps:

| step | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 | 12,000 | 14,000 | 15,000 |
|---|---|---|---|---|---|---|---|---|
| dev | 0.125 | 0.162 | 0.180 | 0.183 | 0.182 | 0.198 | 0.198 | 0.205 |

- At the end, entropy was 0.134 against r30's 0.076, so the floor holds entropy up as designed. Informative groups were 18 of 64 against r30's 19, so the extra entropy doesn't buy more mixed groups.
- On seed 0, r37 lands where r36 (temperature 1.2) does: 0.203 vs 0.205. Both exploration knobs are a little above r30 on every finished seed, by +0.011 to +0.015 over three paired runs. Seed 2, where r30 is strongest, decides both.
- If both hold, temperature 1.2 combined with the floor is a candidate for the H100 box's track D half.

**GPUs:** 27627658 finished its item with less than 180 min left in its 10.5-h allocation and exited. New B200 worker 27800892 (`packed_worker.sbatch`, PER_GPU=3, `screens_d_b200`, otherwise the same settings) takes r37 seed 2 and r30 seeds 3-4. GPU jobs: 6.

## 2026-11-03 15:50 EST: r30 seed 4 failed at vLLM start-up and is requeued once

B200 worker 27800892 started at 14:52 on node5800-c1 and launched r30 seeds 3 and 4 in two slots, two minutes apart. Seed 4 failed after 26 s with vLLM's "No available memory for the cache blocks" error. The seeds dir holds only `config.json`, so it hadn't trained.

This is the first such failure in r30-r37, which all ran three per B200 at `vllm_gpu_memory` 0.35. It is a host-side start-up failure, so it is requeued once at the end of `screens_d_b200`, as AGENT.md allows. If it recurs, the D B200 workers go to PER_GPU=2.

The slot's `stdout.log` stays in `calibration/.../seeds/4` until the retry moves it aside. Both B200 workers (27598456 with r36 seed 2 and r37 seeds 1-2; 27800892 with r30 seed 3) were preempted again on node5800-c1 and are pending. They redo their held items. r26 seed 0 (27412796) runs on node4301.

## 2026-11-03 16:50 EST: r26 seed 0 dev 0.183; r26 closes; H200 worker 27814763

r26 seed 0 (D 004's recipe at lr 3e-6, on H200) finished at 16:01 after two preemptions. It reached step 11,659 when the 14,040 s budget ran out, with a final dev score of 0.183 (0.190 at step 11,500). At the end it had 29 informative groups and an entropy of 0.063.

| pretrained seed | D 004 | r26 (lr 3e-6, H200) | r30 (lr 6e-6, B200) |
|---|---|---|---|
| 0 | 0.182 | 0.183 | 0.192 |
| 1 | 0.130 | 0.148 | 0.180 |
| 2 | 0.137 | 0.183 | 0.220 |
| mean | 0.150 | 0.171 | 0.197 |

r26 is done. On these three seeds it beats D 004 by 0.021, but it is 0.026 below lr 6e-6 (r30). That backs the choice to gate record 005 with lr 6e-6 (r33). r26 tests nothing that is still open, so it gets no more seeds.

Worker 27412796 exited, which freed one GPU slot. I submitted H200 worker 27814763 for `queue/h200`, which holds r33 seeds 3-4 and A 002 seeds 1-4. That brings the total back to 6 GPU jobs.

**Next track D arm:** r38 = r30 with both exploration knobs (rollout temperature 1.2 from r36, entropy floor 0.05 below 0.1 from r37), seeds 0-2 queued in `screens_d_b200` (which was empty). If r36 and r37 hold up on seed 2, r38 tests whether their gains stack; it is also the candidate arm for the 11-05 box's track D half.

## 2026-11-03 22:45 EST: 11-05 H100 box queues ready (track A r39-r42, track D r36-r38 seeds 3-4, r43, r44)

All Engaging jobs have been pending since about 19:45 (preempted, then waiting on priority). Nothing finished since 16:50.

Box plan for 06:30 EST 11-05 to 06:30 EST 11-06: GPUs 0-3 run track A and GPUs 4-7 run track D (the 50/50 split from INBOX). All arms are step-capped, so the results compare with the Engaging seeds of the same configs. Each batch also has its own control on the box.

| queue | GPUs, packing | arms | items |
|---|---|---|---|
| `box_a` | 0-3, 1 per GPU (fp32 trainer about 51 GB plus vLLM 28 GB, as r16 on the H100) | r39 `a005_ctl` (A 005's recipe as a dev screen, stop at 400 of a 500-step cosine); r40 `a005_ent05` (entropy floor 0.05 below 0.1); r41 `a005_t11` (rollout temperature 1.1); r42 `a005_lr7e-7` | 20: seeds 0-4, one seed of every arm per round |
| `box_d` | 4-7, 3 per GPU (`vllm_gpu_memory` 0.08) | r44 (r30's config in its own directory, so it doesn't collide with r30's Engaging seeds 3-4); r36, r37 and r38 seeds 3-4; r43 (r30 with rollout temperature 1.4) seeds 0-4 | 13 |

- **Track A goal:** stabilise A 005. Its seeds 3-4 shortened their outputs. Entropy on seed 3 fell to 0.049 by step 300, while seed 0 held at 0.115-0.13. Each arm targets that. On the H100, r16 reached step 400 in about 10,200 s, so one round is about 3 h, and five rounds fit in the block's roughly 22.5 usable hours.
- **Track D:** the box completes r36, r37 and r38 to five seeds (Engaging has or will have seeds 0-2). The Engaging queues must not get these seeds 3-4. On a B200 at 3 per GPU, 15k steps took 21,558 s. If the H100 runs at 9-11 h, two rounds fit.
- **Copies:** run `box_sync.sh to box_a:box_d` with extra paths: D 004's pretrained checkpoints (from `screen_pretrained` `records/v3/track_d/004_dm_math_fp32_h200_warm`, path in the pool still to check) and the D tokenizer, which is already in the default list. Use mit_quicktest once the box answers. Workers: `QUEUE=box_a GPUS="0 1 2 3" PER_GPU=1 RUNNER=scripts/research/run_item.sh`, then `QUEUE=box_d GPUS="4 5 6 7" PER_GPU=3 FIRST=1`.
- **Still to do before the block:** find the checkpoint path, and dry-run that `run_item.sh` resolves `screen_pretrained` on a relative pool path on the box.

## 2026-11-04 04:50 EST: box prep finished; Engaging jobs all pending

- **Checkpoint path.** `scripts/slurm/screen_seed.sh` (which `run_item.sh` calls for every config with `screen_pretrained`) loads `${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}/<screen_pretrained>/seeds/<seed>/pretrained`. On the box that is `~/orcd/pool/rlvr-ckpts/records/v3/track_d/004_dm_math_fp32_h200_warm/seeds/<seed>/pretrained`. All five seeds exist on Engaging, and each has its own tokenizer files. `box_sync.sh to` runs `mkdir -p` on every extra path before the rsync, so these nested paths land correctly. The configs' only other pool path, `dm_math/mix_0.2.bin`, sits under `v3.pretrain`, which screens with `screen_pretrained` skip. So it doesn't go to the box.
- **Copy** (from mit_quicktest once the box answers ssh; about 5-6 GB):
  `sbatch -p mit_quicktest --time=00:15:00 -c 4 --mem=8G --chdir=/home/zjason/rlvr-screens --wrap 'BOX=zjason@16.58.142.215 bash scripts/aws/box_sync.sh to box_a:box_d $(for s in 0 1 2 3 4; do echo orcd/pool/rlvr-ckpts/records/v3/track_d/004_dm_math_fp32_h200_warm/seeds/$s/pretrained; done)'`
- **Engaging:** 7 jobs (6 counted) are pending on Priority. B200 27598456 ran from 03:40 to about 04:30 and was then preempted again. No runs have finished since 16:01 EST 11-03.

## 2026-11-04 07:55 EST: r30 seed 3 dev 0.233; r45 (no L1 in training) queued

r30 `d004rl_shape05_lr6e-6_s15k` seed 3 (from D 004's pretrained seed 3, B200 packed 3 per GPU) reached step 15,000 at 12:27Z with 0.233 on the dev ladder. That is its best seed so far. By level it scored L1 0.73, L2 0.12, L3 0.07, L4 0.13, L5 0.04 and L6 0.31.

| r30 seed (D 004 pretrained seed) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| dev at step 15,000 | 0.192 | 0.180 | 0.220 | 0.233 | running |

So far no other arm has a seed 3 to pair with this one; the box's r44, r36, r37 and r38 seeds 3-4 will. r30 seed 4 is running on 27800892, and r38 seeds 1-2 started there at 11:34Z and 12:27Z.

**Next: r45 `d004rl_shape05_lr6e-6_noL1_s15k`** is r30 with training prompts drawn from ladder levels L2-L6 only (new option `train_ladder_levels`, off by default; `uv run ruff check . && uv run pytest -q` pass, 140 tests). L1 (`+-` only) is solved about three times in four by the end, so many of its groups are all-correct and `skip_zero_advantage` drops them. Dropping L1 should leave more informative groups per step at the same step cost. The risk is L1 regressing; the dev ladder still scores every level. Seeds 0-2 are queued on `screens_d_b200`, which had emptied once r38 seed 2 started. The GPU count stays at 6.

## 2026-11-04 08:45 EST: r30 seed 4 dev 0.193; r30 complete at five seeds; r45 seed 0 started

r30 seed 4 (the retry after the vLLM start-up failure) reached step 15,000 at 13:14Z with 0.193 on the dev ladder (L1 0.64, L2 0.16, L3 0.09, L4 0.07, L5 0.03, L6 0.17).

| r30 seed (D 004 pretrained seed) | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| dev at step 15,000 (B200, step cap) | 0.192 | 0.180 | 0.220 | 0.233 | 0.193 | 0.204 |

All five seeds are above the holder's dev mean of 0.159, and the mean is above the 0.189 line. This is B200 with a step cap, not the H200 gating screen; r33 (the same RL on H200, time-capped) is that screen and is still pending. The box's r44 seeds 3-4 repeat r30's config on the H100 as the control for its batch. r45 seed 0 took the freed slot on 27800892; seeds 1-2 are queued.

## 2026-11-04 10:45 EST: r37 seed 1 dev 0.175; r45 seed 1 started

| seed 1, dev at step 15,000 (B200, step cap) | r30 (base) | r36 (temp 1.2) | r37 (entropy floor) |
|---|---|---|---|
| dev | 0.180 | 0.195 | 0.175 |

- r37 (entropy floor 0.05 below 0.1) is 0.005 under r30 on seed 1, after +0.011 on seed 0. That is noise-sized either way, so the entropy floor alone shows no clear gain so far. r36 is still +0.013 to +0.015 on both seeds.
- The freed B200 slot took r45 seed 1 (no L1 in training). r45 seed 2 is still queued. r36 seed 2, r37 seed 2 and r38 seeds 0-2 are running. The 11-05 box covers r36, r37 and r38 seeds 3-4.

## 2026-11-04 12:10 EST: r37 complete at three seeds (no gain); box runs can now finish on Engaging (INBOX 11-04)

| dev at step 15,000 (B200, step cap) | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| r30 (base) | 0.192 | 0.180 | 0.220 | 0.197 |
| r37 (entropy floor 0.05 below 0.1) | 0.203 | 0.175 | 0.215 | 0.198 |

- The entropy floor alone does nothing measurable: +0.011, -0.005, -0.005 against r30, seed by seed. Seeds 3-4 stay in the box queue, so the arm closes at five seeds at little cost. r36 (temperature 1.2) seed 2 and r38 (both) are still running.
- 27598456 (B200) was preempted at 11:29 and requeued. Its seeds (r36 seed 2, r45 seed 1 and the item that followed r37 seed 2) resume from their saved state when it restarts.

**INBOX 11-04 12:00 (5d1ab11): unfinished box runs finish on Engaging.** Two changes, both off by default:
- `scripts/slurm/screen_seed.sh`: `SCREEN_RESUME=1` saves resume state outside `mit_preemptable` too. On the box there is no Slurm partition, so before this no state was saved. The file was replaced atomically; with the variable unset nothing changes for Engaging workers.
- `scripts/aws/box_sync.sh from`: `STATE=1` also copies each unfinished seed's `$CKPT_DIR/<dir>/resume/<i>/state.pt`, before its logs. Logs copied after the state can only be ahead of it, and the trainer truncates logs past the state's step. I checked the rsync filter on a dummy tree: it picks only `state.pt`. Ruff passes, and pytest passes all 140 tests.
- What carries over: the state holds the step, `wall_s`, `eval_s`, model, optimizer and RNG, so step caps, time budgets and log truncation work as for a preempted Engaging seed. The 15,000- and 500-step caps make a box-then-Engaging run comparable to single-GPU-type runs. I'll still mark resumed seeds in the tables. On Engaging, `screen_seed.sh` moves the box `stdout.log` aside, as `stdout.interrupted.<stamp>.log`, and keeps the box weights dir because a state exists.
- Sizes: track D state 1.3 GB, track A about 20 GB (1.7B fp32 weights plus AdamW). The pool is at 51 %.

Box start commands (replacing those in the 04:50 entry): resume saves every 20 min for A and 10 min for D, and items keep starting until 30 min before the worker deadline:
```
cd ~/rlvr-screens && SCREEN_RESUME=1 RESUME_EVERY_S=1200 RESERVE_MIN=30 QUEUE=box_a GPUS="0 1 2 3" PER_GPU=1 RUNNER=scripts/research/run_item.sh bash scripts/aws/box_workers.sh
SCREEN_RESUME=1 RESUME_EVERY_S=600 RESERVE_MIN=30 QUEUE=box_d GPUS="4 5 6 7" PER_GPU=3 FIRST=1 RUNNER=scripts/research/run_item.sh bash scripts/aws/box_workers.sh
```
Last sync, about 10:30Z 11-06 (the deadline is 11:00Z). Run one `mit_quicktest` job per arm directory: quicktest's 15-minute limit fits one directory's states, and the jobs run in parallel.
```
for d in <arm dirs with unfinished seeds>; do sbatch -p mit_quicktest --time=00:15:00 -c 2 --mem=4G --chdir=/home/zjason/rlvr-screens --wrap "STATE=1 BOX=zjason@16.58.142.215 bash scripts/aws/box_sync.sh from box1105 $d"; done
```
Then queue each unfinished item on Engaging: track D on `screens_d_b200` (same `vllm_gpu_memory` 0.08), track A on an H200 worker queue. Check the first resumed seed's step count, cap and logs against its box logs.

## 2026-11-04 12:50 EST: r36 seed 2 dev 0.207; both B200 workers preempted

| dev at step 15,000 (B200, step cap) | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| r30 (base) | 0.192 | 0.180 | 0.220 | 0.197 |
| r36 (temperature 1.2) | 0.205 | 0.195 | 0.207 | 0.202 |
| r37 (entropy floor) | 0.203 | 0.175 | 0.215 | 0.198 |

- Against r30, seed by seed, r36 is +0.013, +0.015, -0.013. As expected, seed 2 decided it, and it didn't hold. Temperature 1.2 is +0.005 on the mean, inside seed noise. The box runs seeds 3-4 for r36, r37 and r38 tomorrow.
- 27598456 (11:29) and 27800892 (11:55) were both preempted and are pending again. On restart they redo their six in-hand items, which resume from saved state: r38 seeds 0-2 and r45 seeds 0-2. The B200 queue file is empty. I'll refill it at the next full wake, since the in-hand items are several hours of work.

## 2026-11-04 13:50 EST: r46 queued, moving time between pretraining and RL (track D)

The RL changes started from D 004's pretrained models all land at 0.198-0.204 on dev at 15,000 steps:

| arm (B200, 15,000 steps) | change from r30 | seeds | dev mean |
|---|---|---|---|
| r30 | (D 004 RL, shaping 0.5, lr 6e-6) | 5 | 0.204 |
| r34 | lr 1e-5 | 3 | 0.200 |
| r36 | temperature 1.2 | 3 | 0.202 |
| r37 | entropy floor | 3 | 0.198 |
| r32 | GSM8K SFT first (D3) | 3 | null (about +0.006) |

So the limit looks like the pretrained model, not the RL settings. PLAN's track D items don't
address it: D2's math mixes are closed, D3 is null, D4 needs a policy that writes steps, and D5 has
no reason yet. One thing nobody has screened is the 6 h split itself. D 000 set 112 min of
pretraining and 234 min of RL, and every record since has kept it. r30's dev curve is still rising
slowly at the end (0.19-0.20 from 11,000 to 15,000 steps), so neither side has obviously
saturated.

**r46** has two arms, each paired with D 004's seeds (same data stream, seed and lr peak; only the
cosine length changes):

| arm | pretraining tokens (H200 minutes) | RL steps (B200 cap) |
|---|---|---|
| r30 (reference) | 2.67B (112 min) | 15,000 |
| `pt150` -> `pt150_rl12500` | 3.58B (150 min) | 12,500 |
| `pt80` -> `pt80_rl17000` | 1.91B (80 min) | 17,000 |

- **Pretraining is capped by tokens**, so B200 runs read what D 004's H200 pretraining reads in that
  many minutes. D 004 read 2.67B tokens in 112.0 min, which is 23.9M per minute. `minutes` is set to
  400 so the token cap is what sets the cosine schedule. The file is 2.32B tokens, sampled at
  random positions, so pt150 sees about 1.5 epochs where D 004 saw about 1.15.
- **RL caps** scale r30's 15,000 steps by the RL time left in the 6 h budget. The H200 rate is about
  0.98 steps/s (r33 seed 0 ran 13,728 steps in its 14,040 s budget).
- **New option, off by default:** `"pretrain_only": true` in run_item.sh runs
  `STAGE=pretrain scripts/run_seed_v3.sh`. That pretrains, keeps the weights under `$CKPT_DIR/<dir>`,
  and skips GRPO and the test ladder. The RL arms then start from those weights through
  `screen_pretrained`, so they resume after preemption the way the other B200 screens do. A
  preempted pretraining reruns from the start (about 1-2 h on a packed B200).
- **Queued:** the four pretraining items (pt150 and pt80, seeds 0 and 1) on `screens_d_b200`, after
  the six in-hand items. I'll queue the RL items by hand once each pretraining finishes, then check
  that its `pretrain_log.jsonl` reached the token cap. If the time cap ended it instead, the
  schedule doesn't match. Seeds 2-4 only go ahead if one side beats r30 on both seeds.
- ruff and pytest pass (140). Both scripts were replaced atomically, and running jobs keep their
  open copies.

## 2026-11-04 14:55 EST: r33 seed 3 dev 0.215 (H200, the track D gating screen)

Worker 26927305 finished r33 `d004rl_shape05_lr6e-6_h200` seed 3 at 13:57 EST. It ran 13,351 steps
in the 14,040 s budget, then exited because less than 330 min were left in its job.

| seed | dev | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.205 | | | | | | |
| 3 | 0.215 | 0.63 | 0.15 | 0.07 | 0.07 | 0.03 | 0.34 |

- Both seeds are above the D line of 0.189 and above D 004's dev mean of 0.159. Two of five seeds
  are in, mean 0.210.
- Seeds 1 and 2 are held by pending H200 workers 27487788 and 27530015. Seed 4 is first in `h200`,
  and the dependency worker 26927306 is now pending on Priority.
- If all five pass the 5-seed form, the next step is track D record 005.

## 2026-11-04 19:50 EST: no GPUs since 14:00 (cluster congestion); waiting

None of my six GPU jobs has run since r33 seed 3 finished at 13:57.

| job | GPU | holds | waiting since |
|---|---|---|---|
| 26927306 | H200 | `h200` queue: r33 seed 4, then A 002 | 13:57 |
| 27487788 | H200 | r33 seed 1 | earlier |
| 27530015 | H200 | r33 seed 2 | earlier |
| 27814763 | H200 | `h200` queue | earlier |
| 27598456 | B200 | r38 seed 0, r45 seeds 1-2 | 11:29 |
| 27800892 | B200 | r38 seeds 1-2, r45 seed 0 | 11:55 |

- All six are pending on Priority. Slurm estimates 26927306 won't start until 11-09.
- The partitions are full. About 500 H200 GPU jobs and 50 B200 GPU jobs are pending on
  mit_normal_gpu and mit_preemptable. Several H200 and B200 nodes are drained, and a few H200 nodes
  are held by other users' reservations.
- The message of the day announces no maintenance.
- Nothing to fix on my side. The six jobs already fill the limit of 6, and moving screens to
  another GPU type would need new controls.
- The 11-05 box (8x H100, 06:30 EST 11-05 to 06:30 EST 11-06) doesn't depend on Engaging, so it
  goes ahead as planned (LOG 11-04 04:50 and 12:10).

## 2026-11-05 01:45 EST: full wake; r46 moves to the box

- Engaging: all six GPU jobs are pending again. r33 seeds 1-2 (27487788, 27530015) ran on mit_preemptable from 22:28 and 22:39 and were preempted at 23:12. Both resumed from saved state (seed 1 at step 9,558) and will again. The message of the day announces no maintenance. HOME is at 41 % and the pool at 51 %.
- **r46 (pretraining/RL split) moves from `screens_d_b200` to the head of `box_d`.** The B200 workers haven't run since 11:55 yesterday, and r46 is the track D question that matters most now (LOG 11-04 13:50). `box_workers.sh` starts one worker per GPU before the second round, so the four pretraining items land one per GPU (4-7), each next to two RL screens.
  - The two pretraining configs have never run. I changed their wall cap from 400 to 900 min so the token cap (3.58B / 1.91B) still binds when a packed H100 pretrains more slowly than an H200. The cosine follows whichever budget runs out first, so the arms stay defined by tokens.
  - The RL items (`pt150_rl12500`, `pt80_rl17000`) are added to the box queue by hand as each pretraining finishes. If one can't finish on the box, its pretrained weights come back with its state (INBOX 11-04).
  - RL control on the box: r44 (r30's config, same GPU and packing). The r30 B200 seeds stay as the paired reference, marked as cross-GPU in the tables.
- The pretraining needs `mix_0.2.bin` (4.6 GB), which `box_sync.sh to` doesn't copy. `agent/box_copy_mix.sh` copies just that file. Box start, after the `to` job from the 04:50 entry:
  ```
  sbatch -p mit_quicktest --time=00:15:00 -c 2 --mem=4G --chdir=/home/zjason/rlvr-screens --wrap 'BOX=zjason@16.58.142.215 bash agent/box_copy_mix.sh'
  ```
  `box_d` now holds 17 items: r46 ×4, then r44/r36/r37/r38 seeds 3-4, then r43 ×5.

## 2026-11-05 07:05 EST: 11-05 box not up at +30 min; fallback launch

- The block started at 11:30Z. ssh to 16.58.142.215 timed out from a compute node at 11:44Z (job 27974167) and again at 12:03Z (27974851), so none of the scheduled launches (+1/+10/+25 min) produced a box I could reach.
- At 12:04Z I ran the allowed fallback (INBOX 11-03), `aws ec2 run-instances --region us-east-2 --launch-template LaunchTemplateName=rlvr-box-cr-070d093aa92f43751`. It returned i-0aaa8c10dbb9e5888 (p5.48xlarge, us-east-2b, pending). The call went through, so the capacity block was still unused. I made no other AWS calls.
- Next: wait for `/var/lib/rlvr-box.ready`, then the `to` sync, the token-file copy and the workers (LOG 11-05 01:45).
- Engaging: both B200 workers are running again since 06:44 (27598456 resumed r45 seeds 1-2 from state). The four H200 jobs are still pending.

## 2026-11-05 07:55 EST: 11-05 box running

- The fallback box answered at 12:25Z with the venv built (torch 2.13, vLLM 0.30, 8x H100 80 GB). The `to` sync (27975489) and the token-file copy (27975490) took about a minute each. Workers started at 12:29Z: `box_a` on GPUs 0-3 (1 per GPU), `box_d` on GPUs 4-7 (3 per GPU planned).
- Track D packing: **3 per GPU doesn't fit next to a pretraining run.** The r46 pretraining uses 31.6 GB, so after one RL screen the third run's vLLM or trainer ran out of memory within about 2 min. The four third-slot workers stopped after two quick failures each (as `queue_worker.sbatch` is meant to), which leaves pretraining plus one RL screen per D GPU.
  - `give_back` returns only the item in hand at the exit, so the six items that failed first (r44 s4, r36 s3/s4, r38 s4, r43 s2/s3) went to `done.txt` as FAILED. I put them back at the head of the box's `box_d` under the queue lock. No GPU time was lost beyond those 2-minute starts.
- Running at 12:51Z:

  | GPU | runs |
  |---|---|
  | 0-3 | A: r39 ctl, r40 ent05, r42 lr7e-7, r41 t11 (seed 0 each) |
  | 4 | r46 pt150 s1 (236k tok/s), r44 s3 |
  | 5 | r46 pt80 s1 (212k tok/s), r38 s3 |
  | 6 | r46 pt80 s0 (237k tok/s), r37 s4 |
  | 7 | r46 pt150 s0 (266k tok/s), r37 s3 |

- At these rates pt80 (1.91B tokens) ends at about 15:00Z and pt150 (3.58B) at about 16:30Z. Both are well inside the 900-min wall cap. When each finishes, its RL item (`pt80_rl17000` / `pt150_rl12500`) goes to the head of `box_d`.
- Helper scripts for this block are `agent/box_start_1105.sh`, `agent/box_health_1105.sh`, `agent/box_q_1105.sh` and `agent/box_requeue_1105.sh`. They run through `box_ssh.sh` inside mit_quicktest jobs, because the box accepts ssh only from Engaging compute nodes.

## 2026-11-05 08:40 EST: box: two dead D runs restarted

- At step ~435, r37 s4 (GPU 6) and r38 s3 (GPU 5) had entropy 0.0, grad norm 0, loss 0 and 100 % malformed outputs. They had been like that since step 1, so the policy never produced a usable completion.
- The other box D runs from the same `004/seeds/3/pretrained` model are healthy: r44 s3 is at reward 0.46 (step 563) and r37 s3 at 0.44 (step 609). So the model files are fine. The cause isn't known. Both bad runs were on the GPUs that share with the pt80 pretraining runs, which may be coincidence.
- I killed the two trainers by PID at 13:34Z and deleted their resume state by exact path, so neither resumes from the garbage weights. I put both items back at the head of `box_d`, and they restarted at once. I'll check their step-1 entropy on the next wake.
- Progress at 13:30Z:

  | run | step | reward |
  |---|---|---|
  | A r39 ctl s0 | 150 | |
  | A r40 ent05 s0 | 168 | |
  | A r41 t11 s0 | 224 | |
  | A r42 lr7e-7 s0 | 176 | |
  | D r44 s3 | 563 | 0.46 |
  | D r37 s3 | 609 | 0.44 |

  pt80 is at about 0.76B of 1.91B tokens, and pt150 at about 0.83B of 3.58B. The RL train/eval logs are written under `$CKPT_DIR/<arm>/seeds/<s>/`, and `box_sync.sh from` picks them up there.

## 2026-11-05 10:05 EST: box: r46 pt80 pretraining done; r38 s3 dead again

- r46 pt80 seeds 0 and 1 both stopped at the token cap, 1.9105B tokens at step 3644. They took 136 and 143 min on a shared H100, and val loss is 2.907 and 2.927. For comparison, D 004 seed 0 reached 2.891 after 2.67B tokens in 112 min on an H200. I queued `r46/pt80_rl17000` seeds 0 and 1 at the head of `box_d`. They will take the slots pt150 frees around 16:30-17:00Z.
- **r38 s3 broke the same way again after its restart**: entropy 0, all outputs malformed and no informative groups from step 1, this time on GPU 6. r37 s4 restarted cleanly on GPU 5, and r44 s4 runs fine on GPU 6. So neither the GPU nor the pretrained model explains it, and so far only this (config, seed) pair repeats. Step 1 already reads 1.000 malformed, while healthy runs from the same model read 0.994. It looks as if vLLM samples garbage from the start, there is never a non-zero advantage, and the policy never moves. I killed it again by PID, deleted its resume state, and put it at the tail of `box_d`. If it breaks a third time I'll drop it from the box and report it with r38's results.
- **Packing:** GPUs 5 and 6 now run 2 RL D runs each, at 53 and 81 GB with no OOM so far.
- Track A r41 t11 s0 finished on the box (`ok`), and r39 ctl s1 started on GPU 3.

## 2026-11-05 10:55 EST: box: r38 s4 dead too, r38 taken off the box; track A box seeds 0 done

**r38 (temperature 1.2 + entropy floor) on the box.** Seed 4, restarted 44 min earlier on GPU 5 (shared with r37 s4), went dead at step 3. Every step since reads entropy 0.0, malformed 1.000, reward -0.100 (623 steps). This is the third dead r38 run on the box: s3 died twice, s4 once. r38 seeds 0 and 2 on Engaging's B200 finished fine. I killed it (verified PID 57292 by cmdline; `agent/box_fix4_1105.sh`), deleted its box resume state, and took r38 s3 out of `box_d`. I also removed r38 from `agent/box_dirs_1105.txt`, so the box sync can't overwrite Engaging's copies. The deleted local jsonl copies in `resume/{3,4}` had no `state.pt`. r38 seeds 3 and 4 are now in `screens_d_b200` on Engaging, with a new B200 worker 27987194 (`packed_worker.sbatch`, PER_GPU=2, RESERVE_MIN 180, RESUME_EVERY_S 600). The dead runs' logs stay in `$CKPT_DIR/.../seeds/{3,4}`, and the worker moves them aside on start. pt80_rl17000 seed 0 took the freed GPU 5 slot.

A dead run's first step sometimes reads entropy 0.0 and malformed 1.000. r37 s4's restart did, and it was healthy from step 2. The cause is still unknown. Every dead run so far has been on the box and had an entropy floor (r37 s4, r38 s3 and s4).

**Track A box screens, seed 0 at step 400 (dev ladder, H100, step cap):**

| arm | change from control | dev at step 400 |
|---|---|---|
| r39 ctl | none (control) | 0.380 |
| r41 t11 | temperature 1.1 | 0.332 |
| r42 lr7e-7 | learning rate 7e-7 | 0.293 |
| r40 ent05 | entropy floor | 0.338 (step 350, still running) |

All three changes trail the control on one seed. Seed 1 of each is running.

**Engaging:** B200 worker 27598456 was preempted at 10:30 and is pending, holding r38 and r45 items that resume from their saves. Worker 27800892 exited at 09:23 because its queue was empty. GPU jobs: 4 H200 + 2 B200 = 6.

## 2026-11-05 12:25 EST: box: r46 pt150 pretraining done; its RL follow-ups queued

On the box, r46 `pt150` pretraining (track D, pretraining-only, token cap 3.582B) finished on both seeds:

| run | tokens | minutes | val loss |
|---|---|---|---|
| pt150 s0 | 3.5819B | 223 | 2.796 |
| pt150 s1 | 3.5819B | 262 | 2.824 |
| pt80 s0 / s1 (for reference) | 1.9105B | 136 / 143 | 2.907 / 2.927 |
| D 004 s0 (112 min on H200, for reference) | 2.67B | 112 | 2.891 |

Both pt150 seeds hit the token cap exactly. Each ran beside an RL run on the same H100, so its minutes are not timed results. `pt150_rl12500` seed 0 started at 16:48Z and was healthy at step 701 (reward 0.48, malformed 0.00, 30 informative groups). Seed 1 is now at the head of `box_d`. `pt80_rl17000` seeds 0-1 are running.

**Track A box screens:** r40 `ent05` seed 0 finished at 0.375 at step 400 (control r39 0.380). All four arms' seed 1 runs are running.

**Engaging:** B200 worker 27598456 is running again (since 12:08 on node5802), resuming its r38/r45 items. 27987194 (r38 seeds 3-4) was preempted and is pending.

## 2026-11-05 13:58 EST: box: track A seed 1 done; control still leads

All four track A box arms finished seed 1 at the step cap (400 steps, dev ladder, H100, same packing as the control):

| arm | change from control | seed 0 | seed 1 | mean |
|---|---|---|---|---|
| r39 ctl | none (control) | 0.380 | 0.383 | 0.382 |
| r40 ent05 | entropy floor | 0.375 | 0.380 | 0.378 |
| r41 t11 | temperature 1.1 | 0.332 | 0.328 | 0.330 |
| r42 lr7e-7 | learning rate 7e-7 | 0.293 | 0.295 | 0.294 |

The seeds agree closely. The entropy floor is level with the control. Temperature 1.1 and the lower learning rate are clearly worse. Seed 2 of all four is running. If they hold, r41 and r42 are dropped after seed 2 and their box slots go to other arms.

**Box track D:** all runs are healthy. r44 s3 is at step 7012 (reward 0.59), r37 s3 at 8041, and `pt150_rl12500` s0 at 3762 (reward 0.51). `pt150_rl12500` s1 is first in `box_d`.

**Engaging:** 6 GPU jobs (4 H200, 2 B200), all pending. B200 27598456 ran briefly from 13:24 and is pending again.

## 2026-11-05 16:10 EST: box: r41 (temperature 1.1) dropped; two new track A arms, r47 and r48

r41 `t11` seed 2 finished at 0.267 at step 400. Its three seeds are 0.332 / 0.328 / 0.267, against the control's 0.380 / 0.383 (r39 seed 2 is still running). It is dropped: seeds 3 and 4 are out of `box_a`.

Two new arms (dev ladder, box, step cap 400, control r39 on the same GPU type and packing) take the slots, at the head of `box_a`:

| arm | change from control (A 005's recipe) | why |
|---|---|---|
| r47 `a005_lr1.5e-6` | learning rate 1.5e-6 instead of 1e-6 | lr 7e-7 (r42) trailed the control by 0.09 on two seeds, so test the other direction |
| r48 `a005_cos400` | cosine over 400 steps instead of 500 | the run stops at 400 steps, so the lr now reaches its floor at the stop |

Each gets seeds 0 and 1 first. r42 `lr7e-7` seeds 3-4 come after them. r42 is dropped too if its seed 2 also trails.

## 2026-11-05 16:40 EST: box: track A seed 2 done; r42 (lr 7e-7) dropped

| arm | change from control | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|---|
| r39 ctl | none (control) | 0.380 | 0.383 | 0.395 | 0.386 |
| r40 ent05 | entropy floor | 0.375 | 0.380 | 0.385 | 0.380 |
| r41 t11 | temperature 1.1 | 0.332 | 0.328 | 0.267 | 0.309 (dropped 16:10) |
| r42 lr7e-7 | learning rate 7e-7 | 0.293 | 0.295 | 0.303 | 0.297 |

All on the dev ladder at the 400-step cap, H100, same packing. r42 trails the control by 0.09 on all three seeds, so its seeds 3-4 are out of `box_a`. r47 and r48 seed 2 go to the end of the queue in their place. The entropy floor stays level with the control (a gap of 0.006 on three seeds), so it does not look worth a promotion screen by itself. r39 and r40 seed 3 and r47 and r48 seed 0 are running.

**Engaging:** H200 worker 27814763 started at 16:24 on node4400. Five more GPU jobs are pending (6 in all).

## 2026-11-05 17:45 EST: box: r37 (entropy floor, track D) complete; level with the control

r37's box seeds 3 and 4 finished. With its three B200 seeds it now has all 5, and it ties r30 (the
D 004 RL control) exactly:

| seed | r30 (B200) | r37 entropy floor | r37 GPU | r44 box control |
|---|---|---|---|---|
| 0 | 0.192 | 0.205 | B200 | – |
| 1 | 0.180 | 0.175 | B200 | – |
| 2 | 0.220 | 0.215 | B200 | – |
| 3 | 0.233 | 0.207 | box H100 | running (step 14,749) |
| 4 | 0.193 | 0.217 | box H100 | 0.203 |
| mean | 0.204 | 0.204 | | |

- Seeds 3 and 4 are cross-GPU against r30, so the same-GPU comparison is r44 (r30's config on the
  box). It has only seed 4 so far, where r37 leads by 0.013; seed 3 lands in about 10 min.
- Either way the entropy floor doesn't move track D's dev score, which matches r40 on track A.
  r37 is closed. r36 (temperature 1.2) seeds 3-4, r43 (temperature 1.4) and the r46 split arms are
  the box's remaining D questions.
- r36 s3 restarted cleanly on GPU 7 (reward 0.44 at step 483). The stale 18-step log from its 07:35
  failed start is replaced. r43 s2 started on GPU 6.

## 2026-11-05 18:50 EST: box: first r46 RL result, the box D control, and a late drop in r39 seed 3

**Track D, all at their step caps on the box (H100, packed 2 per GPU):**

| arm | seed | dev | same-seed reference |
|---|---|---|---|
| r44 box control (r30 config) | 3 | 0.227 | r30 B200 0.233 |
| r44 box control | 4 | 0.203 | r30 B200 0.193 |
| r36 temperature 1.2 | 4 | 0.240 | r44 box 0.203 |
| r46 pt150_rl12500 (150 min pretrain, 12,500 RL steps) | 0 | 0.183 | r30 B200 0.192 (no box s0 yet) |

- The box control matches r30's B200 seeds to within 0.01, so box and B200 D results line up at the
  same step caps.
- r36 s4 is r36's best seed by 0.03 (its B200 seeds were 0.195-0.207). r36 s3 (step 2,195) finishes
  around 02:30 EST. One seed isn't a signal yet.
- r46's longer pretraining starts below the control on seed 0. pt150 seed 1 (step 3,616) and
  pt80_rl17000 seeds 0-1 (steps 15,751 and 14,697 of 17,000) are still running.
- **Queued:** `box_d` was down to r43 s4. I appended r44 seeds 0-2 (`agent/box_r44more_1105.sh`), so
  r43 and r46 get same-GPU control seeds. Any that can't finish before the block ends come back
  with their state and finish on Engaging (INBOX 11-04).

**Track A, r39 (the control) seed 3: 0.253**, against 0.380-0.395 for seeds 0-2. This wasn't a bad
start. Its dev score climbed normally to 0.352 at step 350, then fell to 0.253 at step 400. The
batch reward fell with it, from 0.45 at step 360 to 0.27 at step 400, while entropy stayed around
0.06 and malformed answers rose only from 0.23 to 0.25. Scores fell at every level (L1 0.97 to 0.89,
L3 0.26 to 0.12). The cosine schedule (budget 500 over 400 steps) still leaves about 10 % of the
peak lr at step 400, so this is a late instability, not the end of a decayed run. r48 (cosine over
400, so lr reaches 0 at the end) tests exactly that. Its seed 0 finishes in about 1 h. I'll judge
r40/r47/r48 against the control by seed pairs, not only by the mean.

## 2026-11-05 19:20 EST: box: r47 (lr 1.5e-6) seed 0 collapsed; r40 holds up on seed 3

**Track A, box seeds at 400 steps (dev):**

| arm | s0 | s1 | s2 | s3 | mean over s0-s3 |
|---|---|---|---|---|---|
| r39 control | 0.380 | 0.383 | 0.395 | 0.253 | 0.353 |
| r40 entropy floor 0.5 | 0.375 | 0.380 | 0.385 | 0.378 | 0.380 |
| r47 lr 1.5e-6 | 0.007 | running | – | – | |

- **r47 s0 collapsed.** Its dev score peaked at 0.373 at step 250, then fell to 0.307 at 300, 0.017
  at 350 and 0.007 at 400. Malformed answers went from 0.60 to 0.96, and the batch reward fell from
  0.37 to 0.02 between steps 300 and 350. Even before the collapse it was below the control's
  curve. A higher peak lr makes the late instability that hit r39 s3 worse, so I took r47 s2 out
  of `box_a` (`agent/box_r47drop_1105.sh`). s1 is already 36 min in and keeps running, to see
  whether the collapse repeats.
- **r40 s3 didn't drop where r39 s3 did.** It climbed steadily to 0.398 at step 350 and ended at
  0.378. So r40 now leads the control by 0.027 over four seeds. All of that comes from one seed
  pair, and r40 and r39 are level on the other three. Seed 4 of both (r39 s4 started, r40 s4
  queued) and r48 (cosine to 0 at step 400) tell whether the entropy floor is protecting against the
  late instability.
- Engaging: the two B200 workers (27987194 for r38 s3-4, 27598456 for r38/r45) started at 18:56 on
  mit_preemptable. There are still 6 GPU jobs.

## 2026-11-05 20:05 EST: box: r48 (cosine over 400) is worse; r46 pt80 beats r30 on both seeds, so seeds 2-4 go ahead

| arm (dev ladder, step-capped) | seed | dev | compared with |
|---|---|---|---|
| r48 a005_cos400 (track A, cosine ends at step 400) | 0 | 0.300 | r39 control s0 0.380 |
| r46 pt80_rl17000 (track D, 80 min pretraining, 17,000 RL steps) | 0 | 0.213 | r30 s0 0.192 |
| r46 pt80_rl17000 | 1 | 0.222 | r30 s1 0.180 |
| r46 pt150_rl12500 (150 min pretraining, 12,500 RL steps) | 0 | 0.183 | r30 s0 0.192 |

- **r48 is dropped.** Its eval curve was behind the control the whole way (0.300 at step 300, then
  0.298 and 0.300). Ending the cosine at step 400 means a higher lr through the middle, and that
  doesn't help. I killed seed 1 (45 min in) and removed seed 2 from the queue.
- **r47 (lr 1.5e-6) is dropped** after the seed 0 collapse (LOG 19:20). I killed seed 1 too; one
  collapse at a 1.5x lr is enough reason to stop the arm.
- That leaves two track A arms: r40 (entropy floor) and the r39 control. r40 seeds 0-3 average 0.380,
  against 0.353 for the control. Most of that gap comes from the control's seed 3 drop (0.253), so I
  queued paired seeds 5, 6 and 7 for both arms on box GPUs 0-3. Seeds 5 started at 01:00Z; seeds 7
  should finish around 10:30Z, before the block ends.
- **r46 pt80 beats r30 on both seeds:** a mean of 0.217 against 0.186 on the same seeds, about
  +0.03. The pt80 RL ran on the box (H100), and r30 ran on B200. But the box D control (r44) has
  been within 0.01 of r30 seed for seed, so I count this as a real lead for now. LOG 11-04 13:50
  said seeds 2-4 go ahead only if one side beats r30 on both seeds, and pt80 does. **pt80 seeds 2-4
  (pretraining, token-capped at 1.91B) are queued on `screens_d_b200`.** That queue was empty, so
  the B200 workers would have gone idle after their in-hand r38/r45 items. I'll queue the RL items
  (pt80_rl17000 seeds 2-4) once each pretraining reaches its token cap.
- pt150 seed 0 is below r30. Seed 1 is still running on the box (step about 6,300 of 12,500).
- If pt80 holds up on 5 seeds, the next step is an H200 timed screen of the full 6 h split (80 min
  pretraining, then RL to the time limit). That's what would gate a D record.

## 2026-11-05 22:35 EST: box: seed 4 halves r40's lead over the control (track A)

| seed | r39 control | r40 entropy floor | r40 - r39 |
|---|---|---|---|
| 0 | 0.380 | 0.375 | -0.005 |
| 1 | 0.383 | 0.380 | -0.003 |
| 2 | 0.395 | 0.385 | -0.010 |
| 3 | 0.253 | 0.378 | +0.125 |
| 4 | 0.377 | 0.342 | -0.035 |
| mean | 0.358 | 0.372 | +0.014 |

- (Corrected 23:05 at the maintainer's request, INBOX 11-05 23:00: the control's mean was first
  written as 0.378 and the difference as -0.006. Computed by script, they are 0.358 and +0.014.)
- On four seeds of five, the entropy floor is a little below the control. On seed 3 it avoided the
  control's late drop (0.352 at step 350 down to 0.253 at step 400). So it doesn't raise the
  typical run, but it may guard against the late collapses that cost A 005 its seeds 3 and 4 on
  test. Seeds 5-7 of both arms are running on the box and should tell whether that holds. If the
  control drops again and r40 doesn't, an H200 3-seed screen of r40 is the next step. If not, r40 is
  null.
- r40 seed 4 rose slowly to 0.342 at step 400 (0.305 at step 250). The control's seed 4 got to 0.377.
- B200 worker 27987194 was preempted again at about 22:20 and is pending. Its in-hand items resume
  from saved state when it restarts. The pt80 seed 2 pretraining reruns from the start if that was
  one of them.

## 2026-11-05 23:05 EST: correction to 22:35; r39 control seed 5 at 0.328

- The maintainer pointed out an arithmetic error in the 22:35 table (INBOX 11-05 23:00). The r39
  control's mean over seeds 0-4 is 0.358, not 0.378, so r40 leads by +0.014, not -0.006. I fixed
  the table and marked the fix there. From now on I compute table means with a script.
- That changes the reading: r40 is still ahead on the mean, and all of the lead comes from the
  control's seed 3 drop. On the other four seeds r40 is 0.003-0.035 lower.
- **Control seed 5: 0.328** at step 400 (0.308 at step 350). That's a weaker control seed again,
  though not a drop like seed 3's. Over seeds 0-5 the control's mean is 0.353 (script). r40 seed 5
  is at step 389, and seeds 6-7 of both arms are running. The decision waits for those.

## 2026-11-05 23:45 EST: r40 seed 5 beats the control; pt150 seed 1 ties r30; pt80 RL seeds 2-4 queued

Track A, box, dev ladder, step 400 (all means by script):

| seed | r39 control | r40 entropy floor | r40 - r39 |
|---|---|---|---|
| 5 | 0.328 | 0.358 | +0.030 |
| mean, seeds 0-5 | 0.353 | 0.370 | +0.017 |

- r40 wins seed 5 by 0.030, but the control's seed 5 didn't collapse late (0.308 at step 350, up to
  0.328), so this is a plain weaker control seed rather than the late drop r40 is meant to prevent.
  r40 now leads on 2 of 6 seeds. Seeds 6 and 7 of both arms are running (steps 104-203); the
  decision still waits for them.

Track D, box, dev ladder (r46: token-capped pretraining, then RL to the 6 h budget):

| seed | r30 (112 min pretrain) | pt150_rl12500 | difference |
|---|---|---|---|
| 0 | 0.192 | 0.183 | -0.009 |
| 1 | 0.180 | 0.187 | +0.007 |
| mean | 0.186 | 0.185 | -0.001 |

- Longer pretraining (3.58B tokens, less RL) is a tie with r30. I drop pt150. pt80 (shorter
  pretraining, more RL) led by +0.031 on two seeds, so it's the one to follow up.
- The pt80 pretrainings for seeds 2-4 all reached their 1.911B-token cap on Engaging B200
  (val loss 2.907, 2.940, 2.931). I queued `pt80_rl17000` seeds 2-4 on `screens_d_b200`.
- r33 (track D record gate, H200): seed 4 finished at 0.190. Seeds 0, 3 and 4 are done
  (0.205, 0.215, 0.190; mean 0.203 by script, above the 0.189 line). Seeds 1 and 2 wait for H200
  workers.
- B200 worker 27598456 was preempted at 23:07 and is pending. 27987194 is running.

## 2026-11-06 01:45 EST: control seeds 6-7 hold up, so r40's lead stays one seed; r36 seed 3 ties its control

Track A, box, dev ladder, step 400 (means by script):

| seed | r39 control | r40 entropy floor | r40 - r39 |
|---|---|---|---|
| 6 | 0.398 | 0.388 | -0.010 |
| 7 | 0.352 | running (step 344) | |
| mean, seeds 0-6 | 0.359 | 0.372 | +0.013 |

- Neither control seed collapsed late: seed 6 rose to 0.398, and seed 7 held about 0.35 from step 300
  on. r40 wins 2 seeds of 7. Its lead of +0.013 still comes almost all from the control's seed 3.
  Unless r40's seeds 7-9 show something new, an H200 screen of r40 isn't worth it. Seeds 8-9 of both
  arms started on the box at 01:20-01:35 (the last should finish around 05:00).

Track D, box, dev ladder (15k-step cap):

| seed | r44 control | r36 temperature 1.2 | r43 temperature 1.4 |
|---|---|---|---|
| 2 | running | 0.207 | 0.228 |
| 3 | 0.227 | 0.227 | running |
| 4 | 0.203 | 0.240 | running |

- r36 seed 3 ties its control. On the two seeds paired so far, r36 leads 0.233 to 0.215. Its
  five-seed mean is 0.214. r44 seeds 0-2 give the other three pairs: seed 0 should finish on the box,
  and seeds 1-2 will resume on Engaging after the block.
- I queued seed 5 of r36 and r44 at 01:15, but it can't run. These arms start from record D 004's
  pretrained seeds, and only 0-4 exist, so r36 seed 5 failed after 6 s, and I took r44 seed 5 off
  `box_d` before it started. Neither left any files. GPU 7's second slot stays empty, so r44
  seed 0 runs alone there and is more likely to finish before the block ends.

## 2026-11-06 02:15 EST: r40 seed 7 beats its control by 0.041; r43 (temperature 1.4) finishes three more seeds

**Track A box (dev, step 400).** r40 adds an entropy floor to A 005; r39 is the plain A 005 control.

| seed | r39 control | r40 entropy floor | r40 minus r39 |
|---|---|---|---|
| 7 | 0.352 | 0.393 | +0.041 |
| mean of seeds 0-7 | 0.358 | 0.375 | +0.017 |
| mean of seeds 0-7 without seed 3 | 0.373 | 0.374 | +0.001 |

r40 now wins 3 of 8 seeds. Most of its lead still comes from r39's seed 3 drop (0.253); without
that seed the two arms tie. Seeds 8-9 of both arms are running and should all be done by about
05:30. I'll decide on an H200 screen of r40 after those.

**Track D box.** r43 runs D 004's RL with sampling temperature 1.4; r36 uses 1.2.

| seed | r43 (1.4) | r36 (1.2) | r44 control |
|---|---|---|---|
| 0 | 0.197 | 0.203 | running |
| 1 | 0.232 | 0.195 | running |
| 2 | 0.228 | 0.207 | running |
| 3 | 0.228 | 0.227 | 0.227 |
| 4 | running | 0.240 | 0.203 |
| mean of seeds 0-3 | 0.221 | 0.208 | - |

On the seeds that both have finished, r43 is ahead of r36 by 0.013 on average (seed 0: -0.006,
seed 1: +0.037, seed 2: +0.021, seed 3: +0.001). On the one seed it shares with the control
(seed 3), r43 ties it (+0.001). Seed 4 is at step 12.2k of 15k.

- One box GPU is idle, and so are two half slots. I'm leaving them empty. The D arms on the box
  start from D 004's pretrained seeds, and only seeds 0-4 exist; each of those seeds is either
  finished or running for every open arm. A new arm would also be a new decision. It couldn't
  finish in the four hours left, so it would only add resumes on Engaging.

## 2026-11-06 03:45 EST: r43 (temperature 1.4) finishes 5 seeds at 0.222; the box control is level with both temperature arms so far

**Track D box (dev, 15k-step cap).** r44 is the D 004 RL control (temperature 1.0), r36 uses 1.2, and r43 uses 1.4.

| seed | r44 control | r36 (1.2) | r43 (1.4) |
|---|---|---|---|
| 0 | 0.218 | 0.203 | 0.197 |
| 1 | running | 0.195 | 0.232 |
| 2 | running | 0.207 | 0.228 |
| 3 | 0.227 | 0.227 | 0.228 |
| 4 | 0.203 | 0.240 | 0.225 |
| mean of seeds 0-4 | - | 0.214 | 0.222 |
| mean of seeds 0, 3, 4 | 0.216 | 0.223 | 0.217 |

- r43's 5-seed mean is 0.008 above r36's. Seed by seed it is ahead on 2, level on 2 and behind on 1,
  so the two temperatures aren't distinguishable yet.
- On the three seeds the control has finished, neither temperature beats it. The control's mean
  is 0.216, r43's is 0.217 and r36's is 0.223. Seed 0 favours the control, seed 4 favours the
  temperature arms, and seed 3 ties. The comparison waits on control seeds 1 and 2 (now at steps
  9.1k and 5.3k). They'll resume on Engaging after the block.
- Two box GPUs (4 and 7) and one half slot are now free. They stay empty for the reason in the 02:15
  entry.

## 2026-11-06 05:20 EST: box block ends. With ten seeds each, the entropy floor (r40) doesn't beat its control on the mean; it only narrows the spread. r44 seeds 1-2 come back to Engaging

**Track A box (dev, step 400), all ten seeds.** r39 is the A 005 control; r40 adds the entropy floor.

| seed | r39 control | r40 entropy floor | r40 minus r39 |
|---|---|---|---|
| 0 | 0.380 | 0.375 | -0.005 |
| 1 | 0.383 | 0.380 | -0.003 |
| 2 | 0.395 | 0.385 | -0.010 |
| 3 | 0.253 | 0.378 | +0.125 |
| 4 | 0.377 | 0.342 | -0.035 |
| 5 | 0.328 | 0.358 | +0.030 |
| 6 | 0.398 | 0.388 | -0.010 |
| 7 | 0.352 | 0.393 | +0.041 |
| 8 | 0.392 | 0.385 | -0.007 |
| 9 | 0.405 | 0.382 | -0.023 |
| mean of 10 | 0.366 | 0.377 | +0.010 |
| mean without seed 3 | 0.379 | 0.376 | -0.002 |
| standard deviation | 0.046 | 0.015 | - |

- r40 loses 7 of 10 seeds. Its +0.010 lead on the mean comes from one control seed (seed 3, 0.253);
  without that seed r40 is 0.002 behind. The standard error of the paired difference is 0.015, so
  even +0.010 is well inside the noise.
- What r40 does change is the spread. Its worst seed is 0.342, and the control has two seeds below
  0.34 (0.253 and 0.328). But 2 low seeds in 10 against 0 in 10 is weak evidence: a Fisher test gives
  p of about 0.47. I'm closing r40 without an H200 screen. The H200 queue is already full with record
  002 seeds 1-4 and r33, and a screen would cost three H200 runs for a result the box says is null on
  the mean. If low A seeds turn up in the record runs, this is the arm to come back to.

**Track D: pulling unfinished runs back (INBOX 11-04).** The only box runs left unfinished are r44
(D control, 15k cap) seeds 1 and 2. At 10:16Z a `mit_quicktest` job ran `STATE=1 box_sync.sh from`
on the r44 arm directory and copied `resume/1/state.pt` and `resume/2/state.pt` (1.3 GB each),
then the logs. `grpo.py` writes `state.pt` through a `.tmp` file and `os.replace`, so the copy
can't be a partial save. The logs at copy time end at step 12,911 for seed 1 and step 8,582 for
seed 2, and the trainer drops log lines past the state's step on resume. Both seeds are queued on
`screens_d_b200`. Their config still caps them at `budget_steps` 15000, so they'll be marked as
box-then-B200 seeds in the table. I'll check the first resumed seed's starting step and logs when it
starts. This was the last box sync; no further syncs run, so the box can't overwrite the Engaging
logs. The box's GPUs ran screens only, and I made no AWS calls.

Final box D tally, dev at 15k steps: r36 (temperature 1.2) 0.214 over 5 seeds, r43 (1.4) 0.222 over
5, r37 (entropy floor) 0.204 over 5. Control r44 has 3 of 5 seeds at 0.216. The temperature
question is decided when r44 seeds 1-2 finish.

## 2026-11-06 10:40 EST: the pretraining/RL split (r46) ties the control; r49 screens PLAN D4, a reward for legal moves written in the reasoning

**r46 (80 minutes of dm_math pretraining, 1.91B tokens, then RL to 17,000 steps) is done.** Dev
pass rate; the r46 column is at step 17,000, the arm's full budget, and the controls are at their
15,000-step cap (label corrected at 12:40, INBOX 11-06 11:30; step-15,000 values in the 12:40 entry):

| seed | r46 pt80, step 17,000 | r30 control (Engaging) | r44 control (box) |
|---|---|---|---|
| 0 | 0.213 | 0.192 | 0.218 |
| 1 | 0.222 | 0.180 | 0.212 (box, then B200 from step 12,505) |
| 2 | 0.210 | 0.220 | still running |
| 3 | 0.227 | 0.233 | 0.227 |
| 4 | 0.208 | 0.193 | 0.203 |
| mean | 0.216 | 0.204 | 0.215 (4 seeds) |

- Against r30, r46 at step 17,000 is +0.012 on the mean with a standard error of 0.0095, and it wins 3 of 5 seeds.
- Against the box control on the four seeds both have, r46 at step 17,000 is +0.003. The box control runs about
  0.016 above r30 on the same seeds, which accounts for r46's lead over r30. I'm closing r46 as a
  tie with no H200 screen.
- r44 seed 1 resumed from the pulled-back state at step 12,505 and seed 2 at step 8,582, as the
  stdout logs show. Seed 2 should finish around 11:00; then the temperature question (r43 0.222,
  r36 0.214) gets its full 5-seed control.

**r49: process reward (PLAN D4).** D3's GSM8K SFT run (r32) was null, and its samples show why:
the policy writes one GSM8K-style equation that ignores the puzzle numbers, for example
`14/2=<<14/2=7>>7.</think><answer>7 +(22-2-3)</answer>`. D4 was waiting for a policy that writes
steps, and r32's policy does write them. r49 is r32 unchanged plus a new option, `process_reward`
0.25. It gives credit for the share of the puzzle's n - 1 combining steps that the `<think>` text
writes as legal, exact moves, chained on the numbers still left (`grpo.process_credit`). The
check stops at the first illegal or wrong step. It checks the model's own arithmetic against the
puzzle's numbers and never searches for a solution, so no solver output is involved. An assert
keeps a wrong answer with full process credit below a correct one (0.25 + 0.99 x 0.5 < 1). The
option is off by default, and the solved count now counts rewards >= 1, which is the same for
every existing config. The train log gets a `process_credit` mean so I can see whether the credit
moves. Seeds 0-4 are queued on `screens_d_b200`, with r32 and r30 as the per-seed comparison.

## 2026-11-06 12:40 EST: r33 passes the promotion rule; track D record 005 starts. Label fix for r46

**r33 (D 004 RL at lr 6e-6 with shaping 0.5, H200, gating screen) has all 5 seeds.** In-loop final dev:

| seed | r33 | D 004 holder |
|---|---|---|
| 0 | 0.205 | 0.182 |
| 1 | 0.183 | 0.130 |
| 2 | 0.212 | 0.137 |
| 3 | 0.215 | 0.183 |
| 4 | 0.190 | 0.163 |
| mean | 0.201 | 0.159 |

- The 5-seed form needs a mean of at least 0.189 (D line), at least 4 of 5 seeds above 0.159 (the
  holder's dev mean) and at most 1 collapsed seed. r33 has a mean of 0.201, all 5 seeds above 0.159,
  and no collapsed seed: none is below 0.159, and the highest malformed rate is 0.003. It passes.
- It beats the holder on every seed, by +0.042 on the mean.
- The record is `records/v3/track_d/005_dm_math_fp32_shape05_lr6e-6`. It's D 004's config with RL
  at lr 6e-6 and shaping 0.5, which is r33's config without `screen_pretrained` and `no_save`. So
  each seed pretrains from scratch, as D 004 did. N = 5 (seeds 0-4), fixed now. The seeds are
  queued on `h200` after A 002 seed 4, with `scripts/run_seed_v3.sh` through the record path of
  `run_item.sh` and the untimed warm-up. Track D records may run on `mit_preemptable` (AGENT.md).
  A preempted seed's partial run goes to `extra/`, and the same seed reruns.

**Label fix (INBOX 11-06 11:30).** The 10:40 entry called r46's `pt80` "80 % of the budget". It is
80 minutes of pretraining (1.91B tokens). Its column was also labelled as step 15,000, but it is the
final dev at step 17,000, the arm's full budget. I corrected the labels in place. At the matched step,
15,000, r46 reads:

| seed | r46 pt80, step 15,000 | r30 control |
|---|---|---|
| 0 | 0.210 | 0.192 |
| 1 | 0.212 | 0.180 |
| 2 | 0.195 | 0.220 |
| 3 | 0.207 | 0.233 |
| 4 | 0.198 | 0.193 |
| mean | 0.204 | 0.204 |

That is +0.001 against r30 (standard error 0.012), and -0.008 against the box control on seeds
0, 1, 3, 4. The tie holds, and at matched steps there's no lead left to explain. r46 stays closed.

## 2026-11-06 13:25 EST: A 002 seed 2 finishes; replacement H200 worker

- Record A 002 seed 2 (worker 27487788, node4301) ended `ok`: test ladder score 0.2809, within
  budget (`clock.json`: 17,448 s of 18,000 s, NVIDIA H200).
- 27487788 then exited, with less than 380 minutes left. Its replacement is 28117249, with the same
  settings (mit_preemptable H200, `--time-min=06:30:00`, RESERVE_MIN 380, queue `h200`). The `h200`
  queue holds A 002 seed 4, then D 005 seeds 0-4. GPU jobs: 6.

## 2026-11-06 16:25 EST: A 002 seed 3 finishes; replacement H200 worker

- Record A 002 seed 3 (worker 27530015, node4301) ended `ok`. Its test ladder score is 0.2123, within
  budget (17,448 of 18,000 s), and the warm-up took 41 s. Final dev score: 0.183. This seed fell into
  a long-output mode: 271 tokens on average on the test set, against 64 for seed 2. It ran 1,345
  training steps in the same time, where seed 2 ran 9,020. Its test pass@16 is the highest so far
  (0.350), but pass@1 is lower on L4-L5. It counts as given; seeds are not re-run or dropped.

  | Seed | Test ladder score | Dev score at the end | Mean tokens (test) |
  |---|---|---|---|
  | 0 | see the 11-05 entry | 0.243 | 117 |
  | 2 | 0.2809 | 0.265 | 64 |
  | 3 | 0.2123 | 0.183 | 271 |

- Worker 27530015 exited with less than 380 min left; it is replaced by 28133807 (same submit line).
  Seed 1 is still claimed by pending 27814763, and the h200 queue holds seed 4, then D 005 seeds 0-4.
  GPU jobs: 6.

## 2026-11-06 21:20 EST: A 002 seed 4 preempted; D 005 seed 0 starts

- Record A 002 seed 4 (worker 28117249, node4800) started at 19:30, with a 64 s warm-up. It was preempted
  after about an hour and is pending again. The worker keeps its claim on seed 4, so the same seed reruns
  when the job restarts and the partial run goes to `extra/`, as for seeds 0 and 1.
- Record D 005 seed 0 started on worker 28133807 (node4300) at 20:39, with a 53 s warm-up.
- GPU jobs: 6 (1 running, 5 pending).

## 2026-11-07 03:15 EST: D 005 seed 0 finishes (test 0.215)

- Record D 005 seed 0 (worker 28133807, node4300) ended `ok`. It was within budget: 20,915 of 21,600 s
  (pretraining 6,731 s, then GRPO). The warm-up took 53 s. Its test ladder score is 0.2150 and its final
  dev score 0.193, with no malformed outputs. Mean output length on the test set is 25 tokens.
- For comparison, the holder D 004 has a test mean of 0.1553 over 5 seeds (0.1832, 0.1296, 0.1384,
  0.1677, 0.1572). Seed 0 alone is above every D 004 seed; seeds 1-4 decide.
- The same worker started D 005 seed 1 at 02:34, with a 10 s warm-up. A 002 seed 1 is running on
  27814763. GPU jobs: 6 (2 running, 4 pending).

## 2026-11-07 05:15 EST: A 002 seed 1 finishes (test 0.215); D 005 seed 2 starts

- Record A 002 seed 1 (worker 27814763, node4802) ended `ok` at 05:00. It was within budget: 17,554 of
  18,000 s. The warm-up took 52 s. The two earlier partial runs are in `extra/interrupted_1_1` and `_1_2`.
- A 002 so far (test is the test ladder score, dev the final dev score, tokens the mean test output length):

  | Seed | Test | Dev | Tokens |
  |---|---|---|---|
  | 0 | 0.2518 | 0.243 | 117 |
  | 1 | 0.2149 | 0.210 | 190 |
  | 2 | 0.2809 | 0.265 | 64 |
  | 3 | 0.2123 | 0.183 | 271 |
  | Mean of 4 | 0.2400 | 0.225 | |

  The holder A 004 has a test mean of 0.2671. Seed 4 can't lift A 002 above that mean: it would need a test
  score of 0.376 or more, above every A 002 seed so far. Seed 4 still runs (every seed counts).
- The same worker started D 005 seed 2 at 05:00, with a 13 s warm-up. D 005 seed 1 is running on 28133807.
  The `h200` queue holds D 005 seeds 3-4. GPU jobs: 6 (2 running, 4 pending).

## 2026-11-07 09:15 EST: D 005 seed 1 finishes (test 0.171); seed 3 starts

- Record D 005 seed 1 (worker 28133807, node4300) ended `ok` at 08:25. It was within budget: 20,876 of
  21,600 s (pretraining 6,732 s). The warm-up took 10 s. Test ladder score 0.1711, final dev score 0.187,
  mean test output length 24 tokens.
- D 005 so far, against the holder D 004 (test mean 0.1553):

  | Seed | Test | Dev |
  |---|---|---|
  | 0 | 0.2150 | 0.193 |
  | 1 | 0.1711 | 0.187 |
  | Mean of 2 | 0.1931 | 0.190 |

- The same worker started D 005 seed 3 at 08:25, with a 6 s warm-up. Seed 2 is running on 27814763 (since
  05:00); seed 4 is the last item in `h200`. A 002 seed 4 is still claimed by pending 28117249.
- B200 workers 27598456 and 27987194 are pending on unavailable nodes (`ReqNodeNotAvail`); no maintenance
  notice in the motd. GPU jobs: 6 (2 running, 4 pending).

## 2026-11-07 11:15 EST: D 005 seed 2 finishes (test 0.218); A 002 seed 4 restarts; r49 moves to H200

- Record D 005 seed 2 (worker 27814763, node4802) ended `ok` at 10:51. It was within budget: 20,875 of
  21,600 s (pretraining 6,728 s). The warm-up took 13 s. Test ladder score 0.2183, final dev score 0.210,
  mean test output length 27 tokens.

  | Seed | Test | Dev |
  |---|---|---|
  | 0 | 0.2150 | 0.193 |
  | 1 | 0.1711 | 0.187 |
  | 2 | 0.2183 | 0.210 |
  | Mean of 3 | 0.2015 | 0.197 |

  D 004 (holder) has a test mean of 0.1553. The same worker started D 005 seed 4 at 10:51 (6 s warm-up);
  seed 3 is running on 28133807 since 08:25.
- Worker 28117249 restarted on node4803 at 10:27 and reran A 002 seed 4 (46 s warm-up). The partial run
  from 11-06 19:30 is in `extra/interrupted_4_1`.
- **r49 moves from `screens_d_b200` to `h200`.** The B200 workers 27598456 and 27987194 have been pending
  on `ReqNodeNotAvail` since about 08:00, and the `h200` queue was empty after D 005 seed 4. r49 uses
  `budget_steps` 15000, which stops a run at step 15,000 and reads the lr schedule by step, so its
  step-15,000 result doesn't depend on the GPU type and stays comparable with r32 and r30. On H200 it
  should take about 4.7 h per seed (r33 ran about 12,400 steps in 14,040 s there). The H200 workers
  take seeds 0-4 as they free up, from about 14:25. The B200 workers keep their claimed items
  (r38 seeds 3-4, r44 seed 2).
- GPU jobs: 6 (3 H200 running; 26927306 and 2 B200 pending).

## 2026-11-07 15:10 EST: D 005 seed 3 finishes (test 0.252); A 002 seed 4 preempted again

- Record D 005 seed 3 (worker 28133807, node4300) ended `ok` at 14:17. It was within budget: 20,874 of
  21,600 s (pretraining 6,729 s). The warm-up took 6 s. Test ladder score 0.2516, final dev score 0.217,
  mean test output length 29 tokens.

  | Seed | Test | Dev |
  |---|---|---|
  | 0 | 0.2150 | 0.193 |
  | 1 | 0.1711 | 0.187 |
  | 2 | 0.2183 | 0.210 |
  | 3 | 0.2516 | 0.217 |
  | Mean of 4 | 0.2140 | 0.202 |

  All four dev scores are above D 004's dev mean (0.159), and none collapsed. Seed 4 (running on
  27814763 since 10:51, due about 16:50) needs a test score of 0.089 or more for the 5-seed mean to
  reach the 0.189 promotion line.
- Worker 28133807 then exited with less than 380 min left. I submitted a replacement H200 worker,
  28232037, which will take r49 from the `h200` queue.
- A 002 seed 4 was preempted again about 12:16 (its stdout stops at 12:15:58). Worker 28117249 is
  pending on Priority; when it restarts, the partial run should move to `extra/interrupted_4_2` and the
  same seed reruns.
- GPU jobs: 6 (27814763 running; 28117249, 28232037, 26927306 and B200 27598456, 27987194 pending).

## 2026-11-07 17:05 EST: D 005 seed 4 finishes (test 0.219); D 005 is the new track D record (0.2150)

- Record D 005 seed 4 (worker 27814763, node4802) ended `ok` at 16:43. It was within budget: 20,883 of
  21,600 s (pretraining 6,729 s). The warm-up took 6 s. Test ladder score 0.2191, final dev score 0.200,
  mean test output length 25 tokens.

  | Seed | Test | pass@16 | Dev | Total s |
  |---|---|---|---|---|
  | 0 | 0.2150 | 0.2700 | 0.193 | 20,915 |
  | 1 | 0.1711 | 0.2408 | 0.187 | 20,876 |
  | 2 | 0.2183 | 0.2892 | 0.210 | 20,875 |
  | 3 | 0.2516 | 0.3125 | 0.217 | 20,874 |
  | 4 | 0.2191 | 0.2742 | 0.200 | 20,883 |
  | Mean | 0.2150 (sd 0.0287) | 0.2773 | 0.201 | |

- I wrote the README. `validate_record_v3.py --compare-to records/v3/track_d/004_dm_math_fp32_h200_warm`
  passes: 0.2150 against 0.1552, Welch t = 3.72, df = 7.4, one-sided p = 0.0034, "new record".
  **D 005 is the track D (H200) holder**, +0.060 over D 004. The leaderboard in `README.md` is
  updated. No attempt was preempted, so there is no `extra/`.
- What it shows: on top of D 004's pretraining, doubling the RL lr to 6e-6 and giving well-formed
  wrong answers partial credit 0.5 raised the test score by about 2 sd of D 004's seeds. The gating
  screen r33 predicted it: dev mean 0.201 there, 0.201 here.
- New D line: D 005's dev mean 0.201 + 0.03 = **0.231** (STATUS, PLAN). In the 5-seed form, at least
  4 of 5 seeds must also be above 0.201.
- Next for track D: r49 (D 005's RL plus 300 steps of GSM8K SFT and process reward 0.25, step-capped at
  15,000) is first in the `h200` queue. Its comparators are r32 (the same without process reward) and
  r30, at step 15,000. Ideas from PLAN still untried for D: D5 (model size).
- Worker 27814763 then exited with less than 380 min left. I submitted a replacement H200 worker,
  28239994. GPU jobs: 6, all pending (3 H200 workers, 26927306, 2 B200 on ReqNodeNotAvail).

## 2026-11-07 23:05 EST: r49 starts on H200; A 002 seed 4 preempted a third time

- A 002 seed 4 (worker 28117249) restarted at 19:43 EST on node4301. It was preempted within a minute.
  The earlier partial run moved to `extra/interrupted_4_2` as it should. The new partial (under a
  minute) is in `seeds/4/` and will move to `extra/interrupted_4_3` when the seed reruns.
  The job is pending again on preemptable (Resources) and still holds the claim.
- r49 seeds 0 and 1 started at about 22:10 EST on node4602 (workers 28232037 and 28239994).
  The GSM8K SFT finished and RL is running. Seed 0 at step 1130 logs `process_credit` 0.81 and
  mean reward 0.64, so the process-reward term is active. The comparison with r32 and r30 waits
  for step 15,000 (about 4.7 h per seed). The `h200` queue holds r49 seeds 2-4, about 9 h of work
  for 2 workers.
- GPU jobs: 6 (r49 running on 2 H200; pending: 28117249, 26927306, and 2 B200 on ReqNodeNotAvail).

## 2026-11-08 01:56 EST: A 002 seed 4 preempted a fourth time

- The seed restarted at 00:28 EST on node4602 (warm-up 14 s; the previous partial moved to
  `extra/interrupted_4_3`). It was preempted at 01:48 EST, about 80 min in. The partial stays in
  `seeds/4/` and will move to `extra/interrupted_4_4` on the next rerun. Worker 28117249 is pending
  again and holds the claim.
- I am keeping it on preemptable. `mit_normal_gpu` (6 h, enough for a ~5 h seed) is no real
  alternative: 26927306 has been pending there on Priority since 10-31.

## 2026-11-08 02:56 EST: r49 seeds 0 and 1 preempted; A 002 seed 4 running again

- node4602 was taken back at about 02:00 EST. r49 seed 1 (worker 28239994, step 6099 of 15,000) was
  preempted at 01:57 and seed 0 (worker 28232037, step 5603) at 02:04. Both workers are pending and
  keep their claims, so the seeds rerun from the start when the workers get a GPU.
- A 002 seed 4 (worker 28117249) restarted at 01:57 EST on node4301, warm-up 77 s. The 80-min
  partial moved to `extra/interrupted_4_4`. If it is not preempted it finishes at about 07:00 EST.

## 2026-11-08 07:40 EST: A 002 lands at 0.2784 ± 0.0904, short of A 004 (Welch p = 0.41)

Seed 4's fifth attempt finished at 07:10 EST on node4301 (77-s warm start). It scored 0.4320 on test, the best
track A seed on the H200 board. A 002's five seeds:

| seed | test ladder | in-loop final dev | pass@16 | mean tokens | within budget |
|---|---|---|---|---|---|
| 0 | 0.2518 | 0.243 | 0.28 | 117 | yes |
| 1 | 0.2149 | 0.210 | 0.29 | 190 | yes |
| 2 | 0.2809 | 0.265 | 0.31 | 64 | yes |
| 3 | 0.2123 | 0.183 | 0.35 | 271 | yes |
| 4 | 0.4320 | 0.417 | 0.58 | 509 | yes |

- Mean of all five: 0.2784 ± 0.0904. `validate_record_v3.py` passes. Against A 004 (0.2671 ± 0.0565), the
  one-sided Welch test gives t = 0.24, df 6.7, p = 0.41. Not significant, so **A 004 stays the track A (H200)
  holder**. A 002 is a valid attempt that falls short; its README says so, and it is not on the leaderboard.
- As in A 004 and A 005, answer length decides the seed. Seeds 0-3 shortened their answers (64-271 tokens) and
  scored 0.21-0.28. Seed 4 kept 509-token answers and scored 0.43.
- Seven preempted attempts are in `extra/` (seed 0 once, seed 1 twice, seed 4 four times), each with a `note.txt`.
  All stopped before the test ladder.
- Worker 28117249 exited (less than 380 min left). H200 worker 28312159 replaces it, reading `h200` (r49 seeds
  2-4).
- **A 003 is not started.** The INBOX hold lifts now that A 002 has landed, but A 003 was promoted on the H100
  (dev 0.324), and a record now needs an H200 promotion screen (AGENT.md). A 005's screen was better (dev 0.397
  at step 400) and its record still fell short (p = 0.072). A 003 shares the collapse that costs these recipes
  their seeds, so its H200 GPU-hours go to track D instead. The maintainer can overrule this in INBOX.

## 2026-11-08 08:50 EST: r49 seed 1 at step 15,000 is 0.217; r50 (PLAN D5, model size) queued

- r49 seed 1 (worker 28239994, node4301) finished: dev 0.217 at step 15,000, against r32 seed 1 0.203
  (+0.013) and r30 seed 1 0.180 (+0.037). That is one seed. Seed 0 is running again on node5000
  after its preemption; seeds 2-4 follow (seed 2 started 07:35 on node4300, worker 28312159).
- Worker 28239994 exited (less than 380 min left). H200 worker 28316532 replaces it.
- **r50 queued (PLAN D5, the last untried D item).** D 005's 110M model reads 2.67B tokens in its 112 min of
  pretraining, about 24 per parameter, so the size was never screened against the time budget. LOG 11-01
  23:50 said D5 had no reason while the policy guessed once. r49 now writes legal moves in `<think>`
  (process credit about 0.75-0.81), which gives it a reason. Two arms, D 005's recipe in full with only the
  model changed (`--width`/`--layers` in the pretraining args):

  | arm | model | parameters | tokens per parameter in 112 min (estimate) |
  |---|---|---|---|
  | `calibration/research_r50/d005_w640_l10` | 640 x 10 layers | about 70M | about 60 |
  | `calibration/research_r50/d005_w1024_l12` | 1024 x 12 layers | about 184M | about 9 |

  Each item pretrains for 112 min and runs RL for the record's 14,040 s, with no step cap. The final in-loop
  dev then compares by seed with D 005's own H200 seeds (dev 0.193, 0.187, 0.210 for seeds 0-2).
  Seeds 0-2 of both arms are in `h200` after r49 seeds 3-4, about 6 h per item.
- GPU jobs: 6 (2 H200 running r49 seeds 0 and 2; pending: H200 28316532, 26927306, and 2 B200 on
  ReqNodeNotAvail). The `h200` queue holds about 36 h of work.

## 2026-11-08 12:25 EST: r49 seed 0 at step 15,000 is 0.190, level with its comparators

- r49 seed 0 (worker 28232037, node5000, restarted after a preemption) finished: dev 0.190 at step 15,000,
  against r32 seed 0 0.187 (+0.003) and r30 seed 0 0.192 (-0.002).

  | seed | r49 | r32 | r30 | r49 - r32 | r49 - r30 |
  |---|---|---|---|---|---|
  | 0 | 0.190 | 0.187 | 0.192 | +0.003 | -0.002 |
  | 1 | 0.217 | 0.203 | 0.180 | +0.014 | +0.037 |
  | mean of 2 | 0.203 | 0.195 | 0.186 | | |

  Two seeds: ahead of r32 on both, but seed 0 adds nothing. No call until seeds 2-4 land (seed 2 is at
  step 6,928 on node4300).
- Worker 28232037 exited (less than 380 min left). H200 worker 28334442 replaces it. GPU jobs: 6.

## 2026-11-08 18:05 EST: r49 seeds 2-3 land at 0.205 and 0.233; r49 is level with r32 and r30 so far

- r49 seed 2 (worker 28312159, node4300) dev 0.205 and seed 3 (worker 28316532, node4601) dev 0.233, both at
  step 15,000.

  | seed | r49 | r32 | r30 | r49 - r32 | r49 - r30 |
  |---|---|---|---|---|---|
  | 0 | 0.190 | 0.187 | 0.192 | +0.003 | -0.002 |
  | 1 | 0.217 | 0.203 | 0.180 | +0.014 | +0.037 |
  | 2 | 0.205 | 0.220 | 0.220 | -0.015 | -0.015 |
  | 3 | 0.233 | - | 0.233 | - | +0.000 |

  Means over shared seeds: r49 0.204 vs r32 0.203 (seeds 0-2); r49 0.211 vs r30 0.206 (seeds 0-3).
  GSM8K SFT plus process reward 0.25 does not move the step-15,000 dev; the seed-1 lead was noise.
  Seed 4 (started 17:37 on node4601) finishes the arm; unless it is far above 0.193 (r30 seed 4), r49 closes
  as no gain.
- r50 w640 seed 0 started 17:41 on node4300 (worker 28312159). Both workers kept going (2-day limit).

## 2026-11-08 22:10 EST: the box control r44 is complete (0.219 over 5 seeds); neither sampling temperature beats it

- r44 seed 2 (box until step 8,582, then B200 packed job 27987194 slot 2 on node5600-c1) finished at
  21:58: dev 0.235 at step 15,000. Seed 1 (box, then B200) had already finished at 0.212 and wasn't
  logged; it's in the table now. Dev at step 15,000:

  | seed | r44 control | r36 temp 1.2 | r43 temp 1.4 | r36 - r44 | r43 - r44 |
  |---|---|---|---|---|---|
  | 0 | 0.218 | 0.203 | 0.197 | -0.015 | -0.022 |
  | 1 | 0.212 | 0.195 | 0.232 | -0.017 | +0.020 |
  | 2 | 0.235 | 0.207 | 0.228 | -0.028 | -0.007 |
  | 3 | 0.227 | 0.227 | 0.228 | +0.000 | +0.002 |
  | 4 | 0.203 | 0.240 | 0.225 | +0.037 | +0.022 |
  | mean | 0.219 | 0.214 | 0.222 | -0.005 (SE 0.011) | +0.003 (SE 0.008) |

  The temperature question is closed: 1.2 is -0.005 and 1.4 is +0.003 against the control, both well
  inside one standard error. Temperature stays at 1.0 for track D.
- Preemptions this hour: B200 27598456 (r38 seed 4) at 21:17 and H200 28334442 (r50 w1024 seed 0, started
  21:04 on node4800) at 21:54. Both are requeued and pending; each keeps its claimed item and reruns it when it starts.
  B200 27987194 is running r38 seed 3 on node5600-c1. GPU jobs: 6 (3 running).

## 2026-11-08 23:55 EST: r50 w640 (70M) seed 0 ties D 005 seed 0 at 0.193

- r50 `d005_w640_l10` seed 0 (worker 28312159, node4300) finished at 23:30. It pretrained for 112 min, then ran
  RL for the full 14,040 s. Total clock 20,877 s, within the 21,600 s budget.

  | seed 0 | r50 w640 (69M) | D 005 (110M) |
  |---|---|---|
  | pretraining tokens | 3.71B | 2.67B |
  | final pretraining val loss | 2.888 | 2.887 |
  | RL steps | 14,400 | 10,272 |
  | final in-loop dev | 0.193 | 0.193 |

  The smaller model reads 39 % more tokens and reaches the same val loss and the same dev. It also fits more RL
  steps into the same time, but that doesn't help here. Seeds 1-2 of both arms are still to come.
- The worker logged this item as FAILED. That's harmless. `run_seed_v3.sh` moves `final/` to the weights
  directory after the timed run, but the r50 configs set `no_save`, so no `final/` exists and the `mv` fails.
  The results were written before that point, and the worker went on to w640 seed 1 (23:30, node4300). The
  other r50 items will log FAILED the same way. I'm leaving the script alone while r50 items use it.
- Preemptions: r49 seed 4 (28316532) at 22:13, at step 9,601 with its resume state saved, and both B200 jobs
  (27598456, r38 seed 4, and 27987194, r38 seed 3) at about 22:54. All three are pending. r50 w1024 seed 0 is
  running on node4601 (28334442). GPU jobs: 6 (2 running).

## 2026-11-09 02:55 EST: r38 (temperature 1.2 + entropy floor) seed 3 never learned; the arm can't promote

- r38 `d004rl_shape05_lr6e-6_t12_ent05_s15k` seed 3 (B200 27987194) finished at 02:02 with dev 0.000. Every
  eval from step 500 to 15,000 is 100 % malformed. The completions repeat `<!DOCTYPE think>`.
- This is a real result, not an infrastructure fault, so the seed counts. It started from the same D 004 seed-3
  pretrained model as r30, r36 and r44. At temperature 1.2, the first six steps had no well-formed sample at
  all. With every reward equal (-0.1), no group has an advantage, so nothing gets updated. Logged entropy,
  KL and grad norm read 0 for that reason, not because the policy is deterministic. A few later steps had one
  well-formed sample (the first at step 7, the last at 14,768), but the policy never escaped. r36
  (temperature 1.2 alone) seed 3 got its first well-formed sample at step 4 and escaped. r30 and r44 (temperature
  1.0) had one at step 1.

  | final in-loop dev | seed 0 | seed 1 | seed 2 | seed 3 | mean (0-3) |
  |---|---|---|---|---|---|
  | r30 (control) | 0.192 | 0.180 | 0.220 | 0.233 | 0.206 |
  | r44 (box control) | 0.218 | 0.212 | 0.235 | 0.227 | 0.223 |
  | r38 (t 1.2 + entropy 0.05) | 0.215 | 0.202 | 0.215 | 0.000 | 0.158 |

  Over seeds 0-2 alone, r38 is +0.013 vs r30 (SE 0.009) and -0.011 vs r44 (SE 0.005), so even without the
  stall it shows no gain. The mean can't reach the 0.231 line: seed 4 would need 0.525. Once seed 4 lands
  (27598456, running on node5600-c1 since 02:04), I'll close r38 as no gain. Lesson for track D: from
  this pretrained model, a hotter sampler can stall RL completely before it learns the format.
- This is the same "dead run" seen three times on the AWS box: r38 seeds 3 and 4, and r37 seed 4 (LOG 11-05 10:05
  and 10:55). That entry said every dead run so far had been on the box and had an entropy floor. This one ran
  on Engaging B200, so the box was not the cause. Every dead run so far had an entropy floor, and the r38 ones
  also had temperature 1.2. An entropy floor isn't worth retrying for track D without a guard against an
  all-malformed start.
- B200 27987194 exited on an empty B200 queue. GPU jobs: 5 (2 running: 28312159 r50 w640 seed 1, 27598456
  r38 seed 4; pending: 26927306, 28316532 r49 seed 4, 28334442 r50 w1024 seed 0).

## 2026-11-09 06:00 EST: r38 seed 4 never learned either, so r38 is closed; r50 w640 seed 1 is 0.170, below D 005

- r38 seed 4 (B200 27598456) finished at 05:40 with dev 0.000. It stalled like seed 3: no well-formed sample in
  the first five steps, 30 steps out of 15,000 with any, and 100 % malformed at every eval.

  | final in-loop dev | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | r38 minus |
  |---|---|---|---|---|---|---|---|
  | r30 (control) | 0.192 | 0.180 | 0.220 | 0.233 | 0.193 | 0.204 | -0.077 (SE 0.056) |
  | r44 (box control) | 0.218 | 0.212 | 0.235 | 0.227 | 0.203 | 0.219 | -0.093 (SE 0.050) |
  | r38 (t 1.2 + entropy 0.05) | 0.215 | 0.202 | 0.215 | 0.000 | 0.000 | 0.126 | |

  Closed. Across the box and Engaging, r38 went dead on 4 of its 9 starts.
- r50 `d005_w640_l10` seed 1 (H200 28312159) finished at 05:19, within budget (20,892 s).

  | r50 w640 (69M) vs D 005 (110M) | seed 0 | seed 1 | mean |
  |---|---|---|---|
  | final pretraining val loss | 2.888 / 2.887 | 2.916 / 2.858 | |
  | RL steps | 14,400 / 10,272 | 13,060 / 13,427 | |
  | final in-loop dev | 0.193 / 0.193 | 0.170 / 0.187 | 0.182 / 0.190 |

  The smaller model is 0.009 behind on two seeds, and seed 1's pretraining loss is worse. Seed 2 is queued.
  The 184M arm (w1024) decides D5: seed 1 started on 26927306 (mit_normal_gpu, node2434) at 05:48, and seed 0
  is preempted and pending (28334442).
- Workers: 28312159 exited after w640 seed 1 (less than 380 min left). I replaced it with 28439655 (same
  settings, queue `h200`, which holds r50 w640 seed 2 and w1024 seed 2). B200 27598456 exited on an empty queue.
  GPU jobs: 4 (26927306 running; 28316532 r49 seed 4, 28334442 r50 w1024 seed 0, 28439655 pending).
- The track D agenda is used up. PLAN D1-D5 have all been run. So have the RL knobs (group 16, 256 prompts,
  no L1, temperature, entropy floor) and the pretraining choices (51 % DM-math, the pt80 and pt150 splits).
  Every arm since D 005 has landed between 0.18 and 0.22. The two free GPU slots stay empty until w1024 seed 0
  or 1 lands (about 11:30). If the larger model helps, the next screen follows from that. If not, I'll
  write a new direction into PLAN before queuing anything.

## 2026-11-09 11:55 EST: r49 closes level with r30; r50 w1024 (182M) seed 1 is 0.215, ahead of D 005 seed 1 (0.187)

- r49 seed 4 (worker 28316532, node4300, resumed at step 9,601) finished at about 11:30: dev 0.215 at step
  15,000. The arm is complete. Dev at step 15,000:

  | seed | r49 GSM8K SFT + process reward | r30 control | r49 - r30 |
  |---|---|---|---|
  | 0 | 0.190 | 0.192 | -0.002 |
  | 1 | 0.217 | 0.180 | +0.037 |
  | 2 | 0.205 | 0.220 | -0.015 |
  | 3 | 0.233 | 0.233 | +0.000 |
  | 4 | 0.215 | 0.193 | +0.022 |
  | mean | 0.212 | 0.204 | +0.008 (SE 0.009) |

  r49 (PLAN D4) is closed as no gain: +0.008 is inside one standard error, and the dev ladder can't show a
  gain that small.
- r50 `d005_w1024_l12` seed 1 (mit_normal_gpu 26927306, node2434, NVIDIA H200) finished at 11:37. Total clock
  20,929 s, within the 21,600 s budget. Logged FAILED, harmless, as before (no `final/` under `no_save`).

  | seed 1 | r50 w1024 (182M) | D 005 (110M) | r50 w640 (69M) |
  |---|---|---|---|
  | pretraining tokens | 1.90B | 2.68B | 3.71B |
  | final pretraining val loss | 2.840 | 2.858 | 2.916 |
  | RL steps | 14,027 | 13,427 | 13,060 |
  | final in-loop dev | 0.215 | 0.187 | 0.170 |

  The 182M model reads 29 % fewer tokens but reaches a lower val loss, and its dev is 0.028 above D 005 on
  this seed. It's one seed, and the spread between seeds is about 0.02, so this is a lead to check, not a
  result. Per level against D 005 seed 1 (in-loop dev): L1 0.69 / 0.64, L6 0.32 / 0.22, L2-L5 within 0.01.
  Seeds 0 and 2 are running (28334442 and 28439655, node4301), and w640 seed 2 is running on 28316532.
  To finish five seeds for the promotion rule (at least 4 of 5 above 0.201, on H200), I queued w1024
  seeds 3 and 4 on `h200` and submitted two H200 workers: 28473022 (mit_normal_gpu, 6 h, RESERVE_MIN 330, as
  the 26927306 chain) and 28473023 (mit_preemptable, the usual settings).
- 26927306 exited after this item (less than 330 min left), which ends that mit_normal_gpu chain.
- GPU jobs: 5 (3 running: 28316532 r50 w640 seed 2, 28334442 w1024 seed 0, 28439655 w1024 seed 2;
  pending: 28473022, 28473023). Track A has no screen running; I'll review PLAN's track A items for the
  sixth slot at the next full wake.

## 2026-11-09 14:00 EST: INBOX 13:00 (keep track A running); r51 screens oversample 2 on A 005's recipe

INBOX 13:00 asks for one or two GPU slots on track A screens whenever there's capacity, with track D still at least half. Track A had nothing running, so the sixth slot goes to track A.

Choice of direction: PLAN A2 lists five knobs. Four have been screened already: lr decay (r48), KL (r13), entropy floor (r40) and tighter grad clipping (r23). The fifth, dynamic sampling (`oversample`), never has been. A 005's record seeds suggest it has room:

| A 005 record seeds (H200) | value |
|---|---|
| time used for 400 steps | 7,686 to 8,959 s of the 17,400-s budget |
| informative groups (not all right or all wrong), steps 1-100 | about 11-14 of 64 |
| informative groups, steps 301-400 | about 22-26 of 64 |

So about half the budget is unused, and most groups give no gradient (skip_zero_advantage drops them). With `oversample: 2`, each step samples 128 prompts and keeps the 64 most informative groups. That should double the useful gradient per step for about twice the sampling time. The estimate is about 40 s per step, or about 16,000 s for 400 steps, so it should still fit the budget, and `budget_steps_time_stop` guards the limit. The code path is already tested (`test_grpo_step_with_oversample_keeps_prompts_per_step_groups`).

r51 `a005_os2`: A 005's config (via the r39 control) with only `oversample` changed from 1 to 2. Dev ladder, no_save, seeds 0-4. It runs on its own queue `screens_a_h200`, so it doesn't wait behind the w1024 seeds. Worker 28486959 is H200 preemptable. The comparison is with A 005's record seeds' in-loop dev at step 400 (mean 0.313), paired by seed. Screens of it shortening its answers will show up in the same seed-by-seed table.

GPU jobs: 6. Running: 28316532 (r50 w640 seed 2), 28334442 (w1024 seed 0), 28439655 (w1024 seed 2). Pending: 28473022 and 28473023 (w1024 seeds 3-4), plus 28486959 (r51). That is 5 on track D and 1 on track A.

## 2026-11-09 18:00 EST: r50 w640 closes 0.008 behind D 005; w1024 is +0.011 over seeds 0-2; r52 tries 270M

- Three r50 seeds finished between 16:46 and 17:45, each with the full 14,040 s of RL: w1024 seed 0 (28334442), w640 seed 2 (28316532) and w1024 seed 2 (28439655). All three log FAILED for the usual harmless reason (`mv final` under no_save). Workers 28316532 and 28334442 went on to w1024 seeds 4 and 3. 28439655 exited after its item.

  | final in-loop dev | seed 0 | seed 1 | seed 2 | mean | vs D 005 |
  |---|---|---|---|---|---|
  | r50 w640 (69M) | 0.193 | 0.170 | 0.202 | 0.188 | -0.008 (SE 0.005) |
  | D 005 (110M) | 0.193 | 0.187 | 0.210 | 0.197 | |
  | r50 w1024 (182M) | 0.212 | 0.215 | 0.197 | 0.208 | +0.011 (SE 0.012) |

  | w1024 / D 005 | seed 0 | seed 2 |
  |---|---|---|
  | pretraining tokens | 1.91B / 2.67B | 1.94B / 2.66B |
  | final pretraining val loss | 2.842 / 2.887 | 2.822 / 2.872 |
  | RL steps | 11,692 / 10,272 | 11,050 / 14,463 |

  The larger model reaches a lower pretraining val loss on every seed, by 0.02-0.05, even though it reads 28 % fewer tokens. Its dev is higher on two of three seeds. Seed 2 is behind on L6 (0.16 vs 0.28) but level or ahead on L1-L4.
- w640 is closed: three seeds, 0.008 behind D 005, with a worse val loss on two of three seeds.
- w1024 can't promote by the rule: the D line is 0.231 and its mean is 0.208. Seeds 3-4 are running. They decide whether the size gain holds over five seeds, which would make 184M the base for the next track D arms.
- r52 `d005_w1280_l12` takes size one step further, to width 1280 and 12 layers (about 270M parameters, tied embeddings). At 112 min it should read about 1.3B tokens. The question is whether the val loss and dev trend keeps going or turns over. Seeds 0-4 are on queue `h200` and go to the two pending track D workers (28473022, 28473023), and then to 28316532 and 28334442 after w1024 seeds 3-4.
- Track A: a second worker for r51 (28523219, H200 preemptable, queue `screens_a_h200`), so track A has two slots, per INBOX 13:00.
- The array job 28518521 (`eng-board1`, CPU only, `~/pokerbot26`) is the maintainer's, not this loop's. It uses no GPUs and I'm leaving it alone.
- GPU jobs: 6, 4 track D and 2 track A. Running: 28316532 (w1024 seed 4), 28334442 (w1024 seed 3). Pending: 28473022, 28473023 (r52), 28486959, 28523219 (r51).

## 2026-11-10 10:20 EST: r50 closes level with D 005 over five seeds; r52 (276M) is +0.011 over seeds 0-2; r53 tests depth

The loop restarted at 09:40 EST after it reached its maximum duration at 20:44 EST 11-09. AGENT.md is unchanged, there is nothing new in INBOX (last b789aeb), and the pool is at 53.5 %.

**r50 w1024 (182M), all five seeds** (in-loop final dev; D 005 = the record's own seeds; computed with python):

| seed | tokens | val loss | RL steps | w1024 dev | D 005 dev | diff |
|---|---|---|---|---|---|---|
| 0 | 1.91B | 2.842 | 11,692 | 0.212 | 0.193 | +0.019 |
| 1 | 1.90B | 2.840 | 14,027 | 0.215 | 0.187 | +0.028 |
| 2 | 1.94B | 2.822 | 11,050 | 0.197 | 0.210 | -0.013 |
| 3 | 1.92B | 2.848 | 12,599 | 0.188 | 0.217 | -0.029 |
| 4 | 1.91B | 2.826 | 12,753 | 0.207 | 0.200 | +0.007 |
| mean | | | | 0.204 | 0.201 | +0.002 (SE 0.010) |

Seeds 3 and 4 took back most of the early lead: three seeds of five are above D 005's 0.201 and the mean is 0.027 short of the D line 0.231. A 182M model trained on fewer tokens (1.9B against 2.7B) reaches the same val loss and the same dev score. r50 is closed: neither 69M nor 182M beats 110M.

**r52 w1280 x 12 (276M), seeds 0-2.** All three are within the 21,600 s budget (20,894 / 20,952 / 20,892 s).

| seed | tokens | val loss | RL steps | w1280 dev | D 005 dev | diff |
|---|---|---|---|---|---|---|
| 0 | 1.42B | 2.845 | 12,017 | 0.203 | 0.193 | +0.010 |
| 1 | 1.42B | 2.859 | 8,118 | 0.198 | 0.187 | +0.011 |
| 2 | 1.44B | 2.856 | 12,003 | 0.222 | 0.210 | +0.012 |
| mean | | | | 0.208 | 0.197 | +0.011 (SE 0.001) |

Every seed is about 0.011 ahead of its D 005 partner, which is very steady, but w1024 looked the same after three seeds (+0.011) and ended at +0.002. No seed is near 0.231, so the 3-seed form of the rule is not met. Seed 3 is running and seed 4 is queued. The val loss is flat across sizes (2.82-2.89): at a fixed 112 min of pretraining, a bigger model sees fewer tokens and ends at about the same loss. Per level, the 276M seeds are not better at L1 (0.60-0.72) but are slightly better at L3-L4.

**Next for track D: r53 `d005_w768_l24`**, PLAN D5 by depth instead of width. It is D 005's recipe with 24 layers at width 768 (about 195M parameters). Countdown needs several dependent arithmetic steps, and depth is the usual lever for that, while widening has not helped. Seeds 0-2 are queued on `h200` after r52 seed 4. PLAN's other track D items have each been tried (D2 data mixes closed, D3 GSM8K SFT null in r32, D4 process reward null in r49), so size and shape are what's left on the agenda.

**Workers.** Two workers had exited (less than 380 min left), so there were only 4 GPU jobs. New jobs:
- 28595678: H200 preemptable, queue `h200` (track D).
- 28595679: H200 mit_normal_gpu (6 h), queue `screens_a_h200` (r51, track A). r51's two preemptable workers (28486959, 28523219) have been pending on Resources since 11-09, and nothing from r51 has started.

GPU jobs: 6. Track D: 28316532 (running, r52 seed 3), 28473022 and 28595678. Track A: 28486959, 28523219 and 28595679. Track D gets half the slots, per INBOX 10-31 and 11-09.

## 2026-11-10 10:33 EST: handover to Fable 5.2; the run is extended to 2026-12-19; the 11-18 and 11-28 AWS boxes are back on; r54 (A 006 promotion screen) and r55 (mini-ladder) queued

INBOX 2026-11-10 10:00 (maintainer): from the restart at about 10:00 EST this session runs on Fable 5.2 (`claude-fable-5-2`) at effort xhigh. It was Opus 5.5 at effort high until then. It is the same conversation, resumed, and the notes below are written by the new model from this entry on. New commits carry the trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. The run now ends on 2026-12-19, not 2026-11-19. The rules, the promotion rule, the 6-GPU cap and the track split (D at least half, A not idle) are unchanged. That note left AWS as it was. A second note, committed at 10:21 EST, turns the two remaining boxes back on (see "The AWS boxes" below). I re-read AGENT.md in full (new dates, the new trailer, and the note that the harness ends a /loop after 7 days and cron re-issues it).

Two slips in the entries just before this one:
- The previous entry and STATUS are headed "10:20 EST" but were written at about 09:55 EST. I had estimated the time instead of reading the clock. From now on an entry's heading carries the time `date` gives when the entry is finished.
- r45 never got a result entry. It is below, under "Closed since the last full table".

### Where things stand at the handover

**Record holders** (test ladder, 5 seeds each; README leaderboard):

| track | holder | test | in-loop dev mean | promotion line (dev mean + 0.03) |
|---|---|---|---|---|
| A | `004_ladder_constrained_h200` | 0.2671 ± 0.0565 | 0.255 | 0.285 |
| D | `005_dm_math_fp32_shape05_lr6e-6` | 0.2150 ± 0.0287 | 0.201 | 0.231 |

`records/v3/track_a/005_fp32_mal0_cos500_s400` (A 005) is valid but not a record: test 0.3400 ± 0.0815 against A 004, Welch p = 0.072. Its mean is 0.073 above the holder's, and two low seeds (about 0.22 and 0.29) cost it the test.

**Open arms** (in-loop final dev):

| arm | track | what it is | state |
|---|---|---|---|
| r52 `d005_w1280_l12` | D | D 005's recipe at width 1280 (276M) | seeds 0-2: 0.203 / 0.198 / 0.222, mean 0.208, +0.011 paired against D 005's seeds 0-2. Seed 3 is running (dev 0.183 at step 8,500; it ends at about 11:00 EST; its D 005 partner ended at 0.217). Seed 4 is queued. |
| r53 `d005_w768_l24` | D | D 005's recipe with 24 layers at width 768 (about 195M) | seeds 0-2 queued |
| r51 `a005_os2` | A | A 005's recipe with `oversample: 2` | seed 0 running since 09:48 EST at 1.7 steps per minute, so 400 steps take about 235 min; seeds 1-4 queued |

**Closed since the last full table.** r45 (r30's B200 recipe, 15,000 steps, with no L1 prompts in training) ended at 0.188 / 0.157 / 0.187, mean 0.177 ± 0.018. Its control r30 has 0.192 / 0.180 / 0.220 on the same seeds, so the paired difference is -0.020 (computed with python). Dropping the easy level did not help. If anything it hurt, which fits the idea below that easy puzzles carry useful signal.

**Jobs at 10:18 EST** (6 GPU jobs, the cap):

| job | partition | queue | state |
|---|---|---|---|
| 28316532 | preemptable H200 | `h200` (track D) | running r52 seed 3 on node4300 |
| 28473022 | mit_normal_gpu H200 | `h200` | pending (Priority) |
| 28595678 | preemptable H200 | `h200` | pending |
| 28486959 | preemptable H200 | `screens_a_h200` (track A) | running r51 seed 0 on node4301 |
| 28523219 | preemptable H200 | `screens_a_h200` | pending (Resources) |
| 28595679 | mit_normal_gpu H200 | `screens_a_h200` | pending |

The CPU jobs under this user that I did not submit (the array `eng-board1`, 28597816 `bash` on mit_quicktest, 28597977 `afpga-sif` on mit_normal) are not mine, and I leave them alone. The pool is at 53.5 %.

### What I looked at before revising the plan

**Track D is flat because the reward is sparse, not because the model is small.** Every RL knob, both wider models and the extra data stages ended at 0.198-0.222 against the 0.231 line. By level, r52 seed 2 has L1 0.72, L2 0.13, L3 0.08, L4 0.09, L5 0.03 and L6 0.28. About 26-29 of 64 groups give a gradient in the second half of a run.

**Dev is still rising slowly when the budget ends.** D 005 seed 2 by step: 0.085 at 500, 0.123 at 2,500, 0.160 at 4,500, 0.182 at 6,500, 0.193 at 8,500, 0.203 at 10,500, 0.215 at 12,500 and 0.210 at 14,463. That is about +0.03 from step 6,000 to 12,000. The number of steps a seed gets depends on the node: D 005's seeds reached between 10,272 and 14,463.

**Where a track D step goes** (second half of each run, seconds per step):

| run | step | sampling | update | other |
|---|---|---|---|---|
| D 005 seed 0 | 1.393 | 1.060 | 0.285 | 0.048 |
| D 005 seed 2 | 0.955 | 0.666 | 0.253 | 0.037 |
| D 005 seed 4 | 0.979 | 0.704 | 0.236 | 0.038 |
| r52 seed 2 (276M) | 1.082 | 0.673 | 0.372 | 0.038 |

Sampling 512 short completions through vLLM is about 70 % of a step. Puzzle generation is in "other", which is 3-4 %. I had guessed that L3's generator (23 ms per puzzle on the login node) cost 15-20 % of a step, and that guess was wrong.

**Track A has the mean and lacks the consistency.** On the 11-05 box (H100, capped at 400 steps, ten seeds each) A 005's recipe (r39) scored 0.366 with SD 0.046, and the same recipe with an entropy floor (r40) scored 0.377 with SD 0.015. The floor did not move the mean, but it removed the low seeds (r39's worst is 0.253, r40's worst is 0.342). A 005 missed the record on its low seeds, so the floor is the direct fix. A 005 also stops at step 400 after 7,700-9,000 s of its 17,400 s, so about half of the budget is unused.

### The AWS boxes (INBOX 2026-11-10 10:40)

Maintainer: "surely just use it". Both remaining capacity blocks are back on, for screens only:

| block (EST) | GPUs | region | launch template |
|---|---|---|---|
| 11-18 06:30 to 11-19 06:30 | 8x H100 | us-east-1f | `rlvr-box-cr-0261a6f8bb30fc123` |
| 11-28 06:30 to 11-29 06:30 | 8x H200 | us-east-1d | `rlvr-box-cr-08927241900b5884c` |

What the note changes for me:
- Nothing launches the box automatically. From the block start I run `aws ec2 describe-instances` and then `aws ec2 run-instances` with the block's template, and retry every couple of minutes until it succeeds. Those two commands, for these two templates, are the only AWS calls allowed. The $500 budget stays suspended.
- Each box is split about 50/50 between the tracks. Its 8 GPUs do not count toward the 6-GPU limit on Engaging. Records and the test ladder stay on Engaging.
- Engaging's maintenance is 11-17 00:00-21:00 EST, and the first block starts 9.5 h after it ends. So the 11-18 queue files, configs and copy list have to be committed by 11-16.
- While a box is up I wake every 20-30 min, sync results back every wake, and bring unfinished runs' state back before 06:00 EST on the end day.
- The maintainer may still take the 11-28 block for other work. I plan for it until a note says so.

How I plan to use them (PLAN, "Box days"; the arms are fixed by 11-16 and 11-26 from what r54, r55 and their follow-ups show):
- 11-18, H100, step caps, each batch with its own control arm. Track A on 4 GPUs, one fp32 run per GPU: the longer schedule under the entropy floor (PLAN A7), about 12 runs of 700 steps. Track D on 4 GPUs, packed: RL-only arms from D 005's checkpoints (the mini-ladder follow-ups and level weights, PLAN D6 and D8).
- 11-28, H200: the stacked track D arm and track A's best A7 arm, with more seeds than Engaging can give in a day.

### New work

**r54 `a005_ent05_h200` (track A): the H200 promotion screen for record A 006.** It is r40's config unchanged (A 005's recipe plus `entropy_coef` 0.05 below `entropy_target` 0.1). r40 ran on the box's H100s under a step cap. AGENT.md wants the gating screen on H200, so this is that screen. Rule, fixed now: if seeds 0-2 all end at 0.285 or above with none collapsed (the 3-seed form), record A 006 starts with this config and 5 fixed seeds. Otherwise seeds 3-4 run and the 5-seed form decides. Seeds 0-2 are at the front of `screens_a_h200`, ahead of r51 seeds 1-4.

**r55 `d005rl_mini3` (track D): a mini-ladder in the training mix.** New training-only levels M2, M3 and M6 are L2, L3 and L6 on 3 numbers instead of 4: the same operator sets and the same "needs" filters. The arm trains on nine names (L1-L6, M2, M3, M6) uniformly, so a third of the prompts are 3-number puzzles. A 3-number puzzle has far fewer candidate expressions, so a policy that guesses once is right much more often, and more groups carry a gradient for "use * here" or "use / here". The bet is that this transfers to the 4-number levels. It is RL-only from D 005's own pretrained checkpoints (same seed), 14,040 s, paired with D 005's in-loop dev. Seeds 0-2 are queued on `h200`, interleaved with r53.

Why I think the mini levels are within the rules. RULES_v3 allows "any RL algorithm, curriculum or training mix over generated puzzles", from "`countdown.generate_puzzle`, `ladder.ladder_stream`, or any other program that produces (numbers, target, operators) without solving them". The mini levels go through the same `generate_level_puzzle` and `_meets_needs` code as the six ladder levels. No solution reaches the model: it sees the numbers, the target and the allowed operators, and gets the verifier's reward. They are never scored and are in no eval set. I have noted this reading in QUESTIONS.md so the maintainer can object.

Code (off unless a config names the levels): `ladder.MINI_LEVELS`, `level_by_name` finds them, and `train_ladder_levels` accepts their names. Two tests cover it. `ruff check` is clean and 142 tests pass.

**PLAN's agenda is revised for the extra month** (new items A6-A7 and D6-D10, each old item marked with how it ended). In short:
- Track A: get A 006 on the board with the entropy floor, then spend the unused half of the budget (a longer schedule under the floor, or r51's oversampling).
- Track D: test ideas that give more reward signal (the mini-ladder, level weights) and more steps per budget (the sampler), then stack whatever gains hold up with r52's width and run one 5-seed promotion screen against 0.231.

**Two questions for the maintainer** are in QUESTIONS.md, and I am not doing either until answered: hindsight relabelling (I assume it is not allowed), and self-imitation on the policy's own verified samples kept in a buffer (I think it is allowed, as an RL algorithm). The same entry carries the note on the mini levels.

STATUS.md is rewritten as a short page (it had grown to 138 lines of history that the LOG already holds).

GPU jobs: 6. Queues: `h200` has r52 seed 4, r55 seeds 0-2 and r53 seeds 0-2 (about 36 GPU-hours). `screens_a_h200` has r54 seeds 0-2 and r51 seeds 1-4 (about 23 GPU-hours).

## 2026-11-10 11:37 EST: r52 seed 3 ends at 0.200 (four-seed paired difference +0.004); both track A workers preempted; r55 moved ahead of r53

**r52 (track D, width 1280, 276M), seed 3.** Final in-loop dev, paired with D 005's same seeds:

| seed | r52 w1280 | D 005 | difference | r52 RL steps | D 005 RL steps |
|---|---|---|---|---|---|
| 0 | 0.203 | 0.193 | +0.010 | 12,017 | 10,272 |
| 1 | 0.198 | 0.187 | +0.012 | 8,118 | 13,427 |
| 2 | 0.222 | 0.210 | +0.012 | 12,003 | 14,463 |
| 3 | 0.200 | 0.217 | -0.017 | 10,265 | 14,058 |
| mean | 0.206 ± 0.011 | 0.202 | +0.004 (SD 0.014) | 10,601 | 13,055 |

What it means: the +0.011 that seeds 0-2 showed is down to +0.004 over four seeds, and the SD of the difference (0.014) is three times that. This is the same pattern as width 1024 (r50: +0.011 over three seeds, +0.002 over five). The wider model does about 19 % fewer RL steps in the same budget on average, and what it gains per step does not make up for more than that. So model size is not what holds track D at 0.20. Seed 4 has been running since 10:50 EST (job 28316532, node4300) and ends at about 16:50 EST. It would need 0.332 for the five-seed mean to reach the line (0.231), so r52 cannot promote on its own. I let it finish so the arm closes on five seeds. Width stays out of the D10 stack unless seed 4 changes this.

**Both track A workers were preempted at about 11:00 EST** (node4301): 28486959 with r51 seed 0 and 28523219 with r54 seed 0. Slurm requeued both jobs (Restarts=1, pending), and a requeued worker first redoes the item it held (`queue/running`), so nothing needs requeueing by hand. r51 seed 0 has resume state at step 118 (dev 0.138 at step 100) and continues from there. r54 seed 0 was 8 minutes in and starts again.

**Queue order.** `h200` is now r55 seeds 0-2, then r53 seeds 0-2 (it had r53 seed 0 second). r55 (the mini-ladder) is the result the 11-18 box plan depends on, and its follow-up arms have to be fixed by 11-16. r53 (depth 24) is the last model-size arm, and model size now looks closed.

GPU jobs: 6 (1 running, 5 pending: two on `mit_normal_gpu` waiting on priority, three on preemptable). Queued: `h200` about 30 GPU-hours, `screens_a_h200` about 25 GPU-hours with the two items to redo.

## 2026-11-10 13:39 EST: five workers still pending after 2.5 h (the cluster is full); r54 moved to the running worker's queue

Only 28316532 (r52 seed 4) has run since the two track A workers were preempted at 11:00 EST. The cluster is full: `squeue` shows 3,043 pending jobs on `mit_preemptable` and about 550 pending requests for H200 GPUs on the two GPU partitions. Slurm's estimate for 28473022 (`mit_normal_gpu`) is 11-15 01:00. `sbatch --test-only` gives the same answer for H200, B200, H100 and L40S on preemptable (no start before about 16:00 EST on 11-11), which I read as "not within the scheduler's planning window" and not as a real time. So another GPU type would not start sooner, and I left the job set as it is (6 GPU jobs: 1 running, 5 pending).

What I changed is the order of work for the one GPU that runs. Worker 28316532 reads `h200`. It is free at about 16:50 EST and can start items until about 02:10 EST on 11-11 (its job ends at 08:31, less the 380 min reserve). `h200` is now: r54 seeds 0, 1, 2; r55 seeds 0-2; r53 seeds 0-2. r54 is the promotion screen for A 006 and was waiting on the preempted track A workers. If no other worker starts, its three seeds end at about 00:20 EST and r55 seed 0 follows. r54 seed 0 was held by the preempted worker 28523219; I removed its `queue/running/28523219` file while the job was pending, so the seed is not run twice. That worker pops r51 when it restarts. `screens_a_h200` is now r51 seeds 1-4 (r51 seed 0 stays with 28486959).

Track split: track D has had this GPU all day (r52 seeds 3 and 4, 05:01 to about 16:50 EST). Track A then has it for about 7.5 h, and r55 follows. Workers that start in the meantime take the next items in the same order.

## 2026-11-10 14:59 EST: no GPU has run since 13:37; step-capped screens moved to B200 (r51, new arms r56 and r57); r52 closed on four seeds; INBOX answers on self-imitation and relabelling

**Correction to the 13:39 entry.** That entry said worker 28316532 was running r52 seed 4. The worker had been preempted at 13:36:50 EST, just before I wrote it. So from 13:37 EST all six workers were pending and no GPU was running. r52 seed 4 was 2 h 46 min into its run (2,601 RL steps, last dev 0.165) and had no resume state, so that run is lost.

**How long the lines are.** The 13:39 entry also concluded that no GPU type would start sooner than another, because `sbatch --test-only` gave the same far date for all of them. That was wrong. I counted the pending jobs ahead of mine by priority, per GPU type, at 14:40-14:50 EST:

| partition | GPU type | jobs ahead | GPUs they ask for | users |
|---|---|---|---|---|
| `mit_preemptable` | H200 | 88 | 175 | 23 |
| `mit_preemptable` | L40S | | 96 | |
| `mit_preemptable` | H100 | | 30 | 6 |
| `mit_preemptable` | B200 | | 16 | 4 |
| `mit_preemptable` | A100 | 0 | 0 | |
| `mit_normal_gpu` | H200 | 53 | 66 | 20 |
| `mit_normal_gpu` | B200 | 1 | 1 | 1 |

Slurm's estimates for my two `mit_normal_gpu` H200 jobs are 11-14 14:30 and 11-15 13:00. B200 is the short line on both partitions. The B200 nodes were full, but the `mit_normal_gpu` jobs on them (about 15 GPUs) all end within 2 h, and the other pending B200 requests there are held by a per-user memory limit or by dependencies. 4-8 A100 GPUs were free, but that type is untested with the vLLM environment and slower, so I left it as a fallback.

**What I changed.** Everything step-capped now goes to B200 (INBOX 10-25 01:40 allows any Engaging type for screens, compared within one type or under step caps). The H200 jobs keep the work that needs an H200: the A 006 gate (r54) and the time-budgeted track D arms (r55, r53).

- Cancelled 28523219, a pending preemptable H200 worker that held no item.
- Submitted three B200 workers at 14:53 EST, all `packed_worker.sbatch` with `REQUIRE_GPU=NVIDIA B200`, `SCREEN_RESUME=1` and node5501-c1 excluded (failed GPU on 10-29):

| job | partition | queue | runs per GPU | time limit | reserve | resume state every |
|---|---|---|---|---|---|---|
| 28623801 | `mit_normal_gpu` | `screens_d_b200` (track D) | 3 | 6 h | 330 min | 10 min |
| 28623802 | `mit_normal_gpu` | `screens_b200` (track A) | 1 | 6 h | 200 min | 30 min |
| 28623803 | `mit_preemptable` | `screens_d_b200` (track D) | 3 | 2 days | 400 min | 10 min |

- **The 6-GPU cap.** Eight GPU jobs are now queued, but at most six can run together: `mit_normal_gpu` lets one user run 2 GPUs at a time (QOS `gres/gpu=2`), so of my four jobs there at most two run, and four are on preemptable. A side effect: while the two B200 jobs run on `mit_normal_gpu`, my two older H200 jobs there are held by that limit. They keep their age. If they get near the front of the H200 line, I stop resubmitting normal B200 workers so the H200 jobs can start.
- **r51 (track A, A 005 + oversample 2) moved to B200**: seeds 0-4 are in `screens_b200`, one run per GPU (the fp32 track A trainer needs about 51 GB). The arm is step-capped at 400 steps on the 500-step schedule, so the GPU type does not change what it trains. Seed 0 continues from its H200 resume state at step 118, so that seed is part H200 and part B200.
- **r56, new (track D, PLAN D6): the mini levels, step-capped.** `calibration/research_r56/d004rl_shape05_lr6e-6_mini3_s15k` is r30's config with `train_ladder_levels` `L1,L2,L3,L4,L5,L6,M2,M3,M6`: RL from D 004's pretrained checkpoints for 15,000 steps, packed 3 per B200. It asks the same question as r55 (do the 3-number levels help?) without waiting for an H200. Paired per seed with r30 (0.192 / 0.180 / 0.220 / 0.233 / 0.193, mean 0.204).
- **r57, new (track D, PLAN D8): no L5 prompts in training.** `calibration/research_r57/d004rl_shape05_lr6e-6_noL5_s15k` is r30's config with `train_ladder_levels` `L1,L2,L3,L4,L6`. L5 (6 numbers) has a dev pass rate of about 0.03 and the longest completions, so its prompts give almost no gradient. The dev ladder still scores all six levels. Paired with r30 in the same way.
- `screens_d_b200` holds r56 seeds 0-2, r57 seeds 0-2, r56 seeds 3-4, r57 seeds 3-4, so each worker's first three runs are one arm's seeds 0-2.
- **H200 queues.** `h200`: r54 seeds 0-1, r55 seeds 0-1, r53 seeds 0-2. `screens_a_h200`: r54 seed 2, r55 seed 2. Whichever H200 worker starts first takes an r54 seed. The rule for r54 is unchanged (LOG 11-10 10:33).
- **r52 (width 1280) is closed on four seeds; seed 4 is not rerun.** Seeds 0-3 ended at 0.203 / 0.198 / 0.222 / 0.200 (mean 0.206), +0.004 paired with D 005's seeds 0-3. Seed 4 would need 0.332 to bring the five-seed mean to the line of 0.231, and its lost run was at 0.165 after 2,601 steps. Six more H200-hours on it would not change the decision, and the H200 line is days long. The partial run is under `calibration/research_r52/d005_w1280_l12/extra/interrupted_4_1`. Width stays out of the D10 stack.

**A rule for the `mit_normal_gpu` workers.** A job that ends at its 6-h limit is not requeued, and the packed track D runs can take 4-6 h. After a TIMEOUT I put the `queue/running/<job>-k` items back at the front of their queue by hand; the next worker continues them from the resume state.

**INBOX 14:50 and 14:54 EST: the answers to QUESTIONS 11-10.**

- Mini levels are fine. r55 and r56 stand.
- **Self-imitation on the policy's own verified samples is allowed**, within limits: the buffer starts empty in each seed's timed run and is never carried between runs or seeds; each completion is stored verbatim with its original generator puzzle; the verifier's verdict is the only filter; sampling done to fill the buffer counts toward the budget. If an arm built on it meets the promotion rule, I say so at the top of STATUS before the record starts. This is now PLAN D11, and it is the next code I write. It targets what the 11-10 reading names as track D's limit (most groups give no gradient).
- **Hindsight relabelling: one unranked screen is allowed, to measure its effect.** Track D, three seeds, dev ladder only, paired against D 005, marked unranked in its config and here, never promoted or stacked. The relabelled target is the value of the model's own expression on the puzzle's own numbers, with no solver or search on relabelled pairs. I report the paired difference, the per-level numbers and the share of samples that gave a usable pair. It stays not allowed in ranked work. This is PLAN D12. It comes after r54, r55 and the self-imitation arm, and the 11-18 box has room for it.

Both options will be off by default, with tests. The first self-imitation arm will be step-capped on B200 and paired with r30, like r56 and r57.

**Track split.** Track D: r56 and r57 on two B200 workers, r55 and r53 on H200. Track A: r51 on one B200 worker, r54 first on both H200 queues.

GPU jobs: 8 queued, 0 running, at most 6 able to run together. Queued work: `screens_d_b200` 10 runs (about 17 GPU-hours packed), `screens_b200` 5 runs (about 19 GPU-hours), `h200` about 31 GPU-hours, `screens_a_h200` about 6 GPU-hours.

## 2026-11-10 15:19 EST: INBOX 14:58 and 15:04 (the maintainer's ideas; RULES v3.4 allows checkpoint choice); r56 started on B200

**INBOX 14:58 and 15:04 EST: the maintainer's ideas.** The list (`~/rlvr-ideas/IDEAS.md`) is private and stays outside this checkout, and so do my assessments of it. The first pass is written in `~/rlvr-ideas/STATUS.md`: one row and a short assessment per idea, each judged as I would judge my own (INBOX 15:04). When an arm is built on one of them, its LOG entry names the id.

**RULES v3.4 (INBOX 15:04): a run may choose among its own checkpoints, or roll back and train on.** The conditions: the run's own code decides before the clock stops; it uses training signals and in-loop dev evals only, never the test ladder; snapshots, restores and the dev evals it acts on count toward the budget.

What that is worth on track A, from A 005's own eval logs (in-loop dev, greedy, 600 puzzles, eight evals per seed at steps 50 to 400):

| seed | final (step 400) | best of the eight | at step |
|---|---|---|---|
| 0 | 0.375 | 0.375 | 400 |
| 1 | 0.403 | 0.403 | 400 |
| 2 | 0.342 | 0.342 | 400 |
| 3 | 0.197 | 0.335 | 250 |
| 4 | 0.248 | 0.365 | 350 |
| mean | 0.313 | 0.364 | |
| SD | 0.087 | 0.027 | |

- Seeds 3 and 4 are the two that slid, and they are why A 005 missed the record (Welch p = 0.072). Before the slide they were level with the other three.
- The cost is small: seed 3's eight evals took 117 s (`grpo_eval_s` in its `clock.json`). The run used 7618 s of the 17,400 s budget.
- The best of eight noisy evals overstates dev (one eval's standard error is about 0.02), so the "best" column can't be compared with a final-eval mean as it stands. The part that should carry over to the test ladder is the two slid seeds.

The plan (PLAN A8, new):

- **An option, off by default, that keeps the best in-loop dev checkpoint.** When it is on, every in-loop dev eval counts toward the clock, the weights are copied when an eval is the best so far, and the best copy is written as final. It gets tests, and before a record relies on it I check that `clock.json` and `validate_record_v3.py` count those evals.
- **The promotion screen for A 006 becomes r54's config with the option on (r58)**, if the code is ready before an H200 worker starts r54. The option doesn't change the training path, so one run's eval log gives both readings (the last eval and the kept one). If r54 starts first, it runs as queued, both readings come from its log, and the option gets a short run on B200 before the record.
- **Rollback and train on comes second (maintainer idea U12)**: restore the best snapshot when dev falls or the malformed share jumps, lower the lr and continue. It is for the 700-step schedule on the 11-18 box, where a slide at step 450 leaves time to use.
- Track D runs don't slide, so no track D arm is planned on it.
- The code order is now: the checkpoint option, then self-imitation (D11), then the unranked relabelling screen (D12).

**B200: the first worker started.** 28623801 (`mit_normal_gpu`, node5600-c1) started at 15:03 EST and runs r56 seeds 0-2 packed on one B200. At 15:19 EST all three were sampling and none had failed. They have written no train log yet, so the check for M2/M3/M6 puzzles in the training mix is still open. The job's 6-h limit ends at 21:03 EST. The packed runs took 3.8-5.8 h in r30, so some may not finish; they save resume state every 10 min, and I put unfinished items back at the front of `screens_d_b200` and submit a replacement.

GPU jobs: 8 (1 running, 7 pending), at most 6 able to run together. Queued work is unchanged apart from r56 seeds 0-2 being in hand.

## 2026-11-10 15:24 EST: r56's training mix holds the mini levels

The check left open in the last entry. A running screen writes its train log under the pool checkpoint directory, not the arm's `seeds/` directory, until it ends; that is why I didn't find it.

- A stream drawn with r56's settings (`train_mix` ladder, the nine level names, 900 puzzles) gives each level about a ninth: L1 106, L2 104, L3 87, L4 94, L5 109, L6 98, M2 94, M3 97, M6 111. So a third of the training puzzles are mini levels.
- r56 seed 0's training solve rate over steps 300-500 is 0.121, against 0.069 for r30 seed 0 over the same steps, as easier puzzles in the mix should give. It says nothing yet about the dev ladder.
- Speed is the same as r30's: 1.57 s per step against 1.62 s over those steps. Seed 0 was at step 593 at 15:24 EST.

## 2026-11-10 15:46 EST: the checkpoint option is in (`keep_best_dev`); r58 replaces r54 as A 006's promotion screen; r51 started on B200

**The option (PLAN A8 (a); commits adbb18d and 73c8b2b).** `keep_best_dev` in `grpo.py`, off by default. With it on:

- After each in-loop dev eval that is at least as good as the best so far (a tie goes to the later eval), the run copies its weights to CPU memory. Before the clock stops it restores the kept copy, so the final weights and `final_pass_rate` are the best eval's. `kept_step` and `last_pass_rate` (the last eval's score) are logged beside them.
- The evals and the copies stay on the clock. The trainer counts no eval time as excluded, so `train_s` equals `wall_s` in every eval row, `run_seed_v3.sh` writes `grpo_eval_s` 0 and `total_s` is the whole run. Neither the script nor the existing validator checks needed a change. `validate_record_v3.py` now also fails a record whose eval log has `kept_step` while eval time is taken off.
- It refuses an eval set named like the test ladder, and more than one process.
- A time-budgeted run's last eval comes after the time stop, so `time_budget_s` has to leave room for it. An eval takes about 15 s; track A's margin is 600 s and track D's about 840 s.
- The kept copy rides in the resume state, so a preempted screen keeps it. Records don't resume.
- Host memory: A 005's seed 3 peaked at 11.6 GB (`sacct` MaxRSS) in a 64 GB job, and the copy adds 6.8 GB.
- `summarize_screens.py` reports the kept eval as such a run's final and prints the last eval beside it.
- Tests (145 pass): with the option off, a run ends on its last weights and its eval time is excluded as before; with it on, the run ends on the right eval's weights with the evals on the clock; the kept copy survives a resume; the validator case.

For configs without the option, the one change is that an eval row's `wall_s` and `train_s` come from one clock reading instead of two.

**r58 `calibration/research_r58/a005_ent05_keepbest_h200` replaces r54 as the H200 promotion screen for A 006 (PLAN A8 (b)).** r54 never got past its first minutes (seed 0 ran 8 minutes on 28523219 this morning before the preemption), so nothing is lost. r58 is r54's config with `keep_best_dev` on. The option doesn't touch the training path, so each seed gives both readings: the kept eval, which is what a record with the option is scored from, and the last eval, which is what r54 would have reported. Seeds 0-1 are first in `h200` and seed 2 is first in `screens_a_h200`.

The rule for r58, fixed before any seed runs: the kept eval is the best of eight noisy evals and reads about 0.01-0.02 high, so both forms of the promotion rule read the kept dev against 0.305 (the line 0.285 plus 0.02). Three seeds all at 0.305 or above and none collapsed start record A 006 with the option on; otherwise seeds 3-4 and the five-seed form. On A 005's logs the kept evals were 0.335 to 0.403. A 006's README will say that the run chose a checkpoint (RULES v3.4).

**r51 started on B200.** 28623802 (`mit_normal_gpu`, node5801-c1) started at 15:41 EST with r51 seed 0, which continues from its H200 resume state at step 118. Its 6-h limit ends at 21:41 EST. Both `mit_normal_gpu` slots now hold B200 jobs, so my two normal H200 jobs wait behind them; the three preemptable H200 jobs are still pending.

GPU jobs: 8 (2 running, 6 pending), at most 6 able to run together. Next in the code order: self-imitation (D11), the unranked relabelling screen (D12), then rollback and train on (A8 (c), by 11-16 for the 11-18 box).

## 2026-11-10 16:16 EST: self-imitation is in (`sil_rows`, PLAN D11); r59 is its first arm, first in the track D B200 queue

**The option (commit 08fe4a3).** Four settings in `grpo.py`, off by default (`sil_rows` 0). With `sil_rows` above 0:

- **What goes in.** Every sample of the run's own rollouts that the verifier marks correct is stored: its completion tokens as sampled, with the generator puzzle it answered. That includes the groups an oversampled step doesn't keep for its update. Nothing else is stored.
- **The buffer** lives in memory only. `run` starts it empty, it is not in the resume state and it is never written to disk, so a resumed screen starts it empty again and refills it. It keeps the newest `sil_buffer` entries (4,096) per pool and overwrites the oldest.
- **The replay.** Each step draws up to `sil_rows` entries and adds their negative log-likelihood to the same optimizer step, scaled so a replayed row weighs like a policy-gradient row with advantage `sil_coef`: the term is `sil_coef` times the replayed tokens' summed negative log-prob, divided by the step's RL token count. The log-probs are the constrained policy's at the sampling temperature, as in the RL term. Replayed rows get no entropy bonus, KL term or importance weight.
- A step's own samples go in after its update, so a step never replays what it just sampled. No entry is replayed twice in one step, so a step replays fewer rows while a pool is small.
- **`sil_by_level`** (on by default): one pool per puzzle level label, and each draw picks a pool evenly among those that hold anything. With one pool for everything, the replay would be mostly L1 solutions, which the policy already produces nine times in ten; drawn by level, five sixths of the replayed rows are L2-L6 solutions, which are rare in the rollouts. Off gives the single pool.
- It samples nothing extra: the RL rollouts already on the clock fill the buffer. The replay's forward and backward pass is on the clock like the rest of the update.
- One process only. The log gains `sil_rows`, `sil_nll` (mean negative log-prob per replayed token, before the step), `sil_added` and `sil_buffer`.
- Tests (150 pass): the buffer keeps the newest entries per level and spreads draws over levels; a replay batch is each stored puzzle's prompt followed by its stored tokens; the replay's gradient equals that of the same rows as policy-gradient rows with advantage `sil_coef`, on top of an unchanged RL term, also on a step with no informative group; a step stores every verified sample and replays only earlier ones; a run with the option off logs nothing about it; a resumed run starts with an empty buffer and its resume state holds none of it. I also built a replay batch with the real track D tokenizer and `constrain_answer` on the login node (no model): the allowed-token sets line up with the stored tokens.

**How it meets the limits of INBOX 11-10 14:50.**

- The buffer starts empty in each seed's run and is never carried over: it exists only in the process's memory.
- Each completion is stored verbatim with its original puzzle: the stored tokens are the ones the RL update trains on, and no code edits, trims or relabels them.
- The verifier's verdict is the only filter on what goes in. Entries leave by age only.
- No sampling is done to fill it beyond the RL rollouts.
- One reading of mine: drawing evenly over levels uses the generator's level label of each stored puzzle. I take that as a training mix over generated puzzles, which RULES_v3 allows and `train_ladder_levels` already does for the rollouts. `sil_by_level` false is the plain buffer if the maintainer reads it differently.

**r59 `calibration/research_r59/d004rl_shape05_lr6e-6_sil64c05_s15k`** is r30's config with `sil_rows` 64, `sil_coef` 0.5, 4,096 entries per level, drawn by level: 15,000 steps on B200 packed 3 per GPU, from D 004's checkpoints, paired per seed with r30 (0.192 / 0.180 / 0.220 / 0.233 / 0.193). Seeds 0-2 are first in `screens_d_b200`, ahead of r57.

- **Why 64 rows at 0.5.** r30 has no advantage normalisation and shaping 0.5, so a lone success in a group of 8 has an advantage of about 0.6; a replayed row at 0.5 weighs a little less than that. r30's seed 0 updates on 46 informative groups of 8 early (steps 300-500) and 23 late (the last 1,000 steps), so 64 replayed rows are between a sixth and a third of an update's rows.
- **The risk.** r30's entropy is already low (0.10 nats per token at steps 300-500, 0.07 at the end). A likelihood term on solutions sharpens the policy further and could cut the number of informative groups. r59's `entropy` and `informative_groups` are read against r30's for that.
- **The cost.** A step cap hides the extra update time. r30's seed 0 spent 0.69 s sampling and 0.61-0.86 s updating per step; r59's `t_update` is reported against that, and a gain here still has to hold in a time-budgeted H200 gate before it counts.
- **A caveat of the B200 setup.** 15,000 steps take about 6.5 h packed (r56 runs at 1.56 s per step), longer than a 6-h `mit_normal_gpu` job, so a seed can continue in a second job, where its buffer starts empty again. A record run doesn't resume.
- **What follows.** A clear paired gain over three seeds leads to seeds 3-4, a coefficient and row-count sweep on the 11-18 box, and a time-budgeted H200 gate from D 005's checkpoints against 0.231. If an arm built on self-imitation meets the promotion rule, that goes to the top of STATUS before any record starts (INBOX 11-10 14:50).

r56 seeds 0-2 and r51 seed 0 are still running on B200 (28623801 to 21:03 EST, 28623802 to 21:41 EST). No H200 worker has started. GPU jobs: 8 (2 running, 6 pending), at most 6 able to run together. Next in the code order: the unranked relabelling screen (D12), then rollback and train on (A8 (c), by 11-16 for the 11-18 box).

## 2026-11-10 19:31 EST: r51 seed 0 (oversample 2) reaches the same 0.39 ceiling in half the steps, then collapses to 0.053 at step 400; r60 (the 700-step schedule under the floor) goes first on B200

**r51 `a005_os2` seed 0** finished at 18:43 EST (steps 1-118 on H200, 119-400 on B200; the run is step-capped). Its worker, 28623802, then exited with less than 200 min left.

```
  a005_os2                           s0 done  232m final 0.053 best 0.390 t.10   58 spm   1.7 ent 0.11 inf 39.5 ok        | 0.013 0.138 0.350 0.387 0.053
```

In-loop dev by step, beside A 005's record seeds (same recipe without oversampling, H200) and r40's seeds 0-4 (A 005 plus the entropy floor, H100 box):

| step | r51 seed 0 | A 005 seed 0 | difference | A 005, 5 seeds | r40, seeds 0-4 | r51 train time (s) | A 005 seed 0 train time (s) |
|---|---|---|---|---|---|---|---|
| 50 | 0.013 | 0.038 | -0.025 | 0.036 | 0.031 | 1,209 | 595 |
| 100 | 0.138 | 0.085 | +0.053 | 0.084 | 0.075 | 2,685 | 1,120 |
| 150 | 0.275 | 0.180 | +0.095 | 0.182 | 0.174 | 5,121 | 2,080 |
| 200 | 0.350 | 0.243 | +0.107 | 0.256 | 0.260 | 7,302 | 3,322 |
| 250 | 0.360 | 0.305 | +0.055 | 0.308 | 0.294 | 9,348 | 4,460 |
| 300 | 0.387 | 0.347 | +0.040 | 0.325 | 0.332 | 11,027 | 5,577 |
| 350 | 0.390 | 0.357 | +0.033 | 0.315 | 0.368 | 12,665 | 6,664 |
| 400 | 0.053 | 0.375 | -0.322 | 0.313 | 0.372 | 14,164 | 7,724 |

What it means, on one seed:

- **Ahead per step, not per unit of time.** From step 100 to 350 it is 0.03-0.11 above A 005's seed 0. A step costs 35.4 s against 19.3 s (1.8x), so when A 005's seed 0 ended (7,724 s, dev 0.375) r51 was at 0.350 (step 200, 7,302 s). The 400 steps took 14,164 s, inside the 17,400-s budget. The B200 was no faster than the H200 here: 51 s per step on the H200 over steps 100-118 and 47 s on the B200 over steps 119-150.
- **The same ceiling.** 0.387 and 0.390 at steps 300 and 350. r40's seeds 0-4 are at 0.372 at step 400 (the best of each seed's eight evals: 0.375 / 0.383 / 0.385 / 0.398 / 0.342, mean 0.377), and A 005's best-of-eight mean is 0.364. One eval has an SE of about 0.019, so these are level. By level at step 350 r51 has 0.97 / 0.35 / 0.31 / 0.23 / 0.02 / 0.46, the profile of r40's finals (0.99 / 0.34 / 0.23 / 0.19 / 0.07 / 0.41): L1 is solved and L2-L5 hold the score down.
- **Then it collapsed.** Final 0.053, with L1 down from 0.97 to 0.20. The share of malformed dev answers fell from 0.54 to 0.24 at the same time, so the model ended on well-formed wrong answers. In training, over ten-step windows from step 350: grad norm 0.73, 1.23, 1.49, 1.98, 2.56 (0.46 over steps 300-350); solve rate in the batch 0.39, 0.33, 0.24, 0.15, 0.10; entropy 0.064, 0.058, 0.065, 0.093, 0.136. The lr was 2.8e-7 falling to 1.9e-7, so a small lr did not prevent it.
- **Why I think it slid.** Keeping the 64 most informative of 128 groups doubles the informative groups in each update (58 / 51 / 44 of 64 over steps 100-200 / 200-300 / 300-400, against 26 / 23 / 22 for A 005's seed 0). The grad norm was about twice A 005's before the slide (0.25 and 0.34 over steps 100-200 and 200-300, against 0.15), and entropy fell faster (0.101 against 0.148 over steps 200-300, 0.074 over steps 300-350). So oversampling acts like a larger step: it gets to the ceiling sooner, and sooner to the entropy of 0.05-0.09 at which A 005's seeds 3 and 4 slid. The floor (r40) holds entropy at 0.1.
- With `keep_best_dev` this run would have ended on step 350's weights (0.390).
- A 005 itself slid on 2 of 5 seeds, so one collapse doesn't show that oversampling makes a slide more likely.

**What follows.**

- r58 stays A 006's promotion screen. Nothing here argues for changing the gate.
- r51 seeds 1 and 2 stay queued, and seeds 3 and 4 go last. A7 (b), oversampling with the floor, waits for them. It costs 1.8x per step, so it needs a higher ceiling to be worth having, not a faster start.
- **New arm r60 `calibration/research_r60/a005_ent05_cos875_s700_keepbest`** (PLAN A7 (a)): r58's config (A 005's recipe, the floor, `keep_best_dev`) on a cosine over 875 steps, stopped at 700. The lr is 6.1e-7 at step 400 and ends at the same 1.9e-7. Why now: a higher lr under the floor (PLAN A7, r22-r24), the floor itself and now oversampling all end between 0.37 and 0.40, and nothing has trained past step 500 under the floor. The 11-18 box plan gives about 12 runs to this schedule, and its arms are fixed by 11-16, so a first reading on B200 is worth one slot now.
  - Time: A 005's seeds took 7,569-8,826 s for 400 steps (19-22 s per step), so 700 steps take about 13,200-15,500 s, plus 14 evals on the clock (A 005's eight took 117 s). The 17,400-s budget still stops the run, so a slower run ends early on its kept eval, as a record would.
  - How it is read: the evals from step 450 on against r40's 0.372 at step 400, and the kept dev against r40's best-of-eight mean 0.377. A kept dev of 0.41 or above on at least two of three seeds keeps the schedule in the box's track A batch and leads to an H200 gate. If the evals past step 400 stay at 0.37-0.40, the box's track A half is planned again by 11-16.
- `screens_b200` now holds, in order: r60 seed 0, r51 seed 1, r60 seed 1, r51 seed 2, r60 seed 2, r51 seeds 3 and 4. Each takes about 4 h, one per 6-h job.

Jobs: 28646346 (B200, `mit_normal_gpu`, `screens_b200`) submitted at 19:29 EST. 28623801 (r56 seeds 0-2) runs until 21:03 EST at the latest; its three seeds then go back to the front of `screens_d_b200` with a replacement worker. No H200 worker has started; Slurm's estimate for 28473022 is now 11-14 05:30. GPU jobs: 8 (1 running, 7 pending), at most 6 able to run together.

## 2026-11-10 21:21 EST: r56 seeds 0-2 (the mini levels in track D's training mix): +0.019 paired with r30 at step 15,000, all of it at L2; r55 stays; no GPU job of mine has run since 20:54 EST

r56 `calibration/research_r56/d004rl_shape05_lr6e-6_mini3_s15k` (PLAN D6) is r30's config with the mini levels M2, M3 and M6 added to the training mix (a third of the prompts): RL from D 004's pretrained checkpoints for 15,000 steps, three runs packed on one B200, paired per seed with r30. Seeds 0-2 finished inside job 28623801, which completed at 20:54 EST after 5 h 52 min. The 19:31 entry expected the 6-h limit to cut them. The 15,000 steps took 5.6-5.8 h: 1.38 s per step over the whole run, against 1.56 s in the first minutes.

```
  d004rl_shape05_lr6e-6_mini3_s15k   s0 done  234m final 0.217 best 0.235 t.10   23 spm  64.1 ent 0.05 inf 22.1 ok        | 0.100 0.133 0.182 0.195 0.217
  d004rl_shape05_lr6e-6_mini3_s15k   s1 done  234m final 0.212 best 0.217 t.10   16 spm  64.1 ent 0.06 inf 23.4 ok        | 0.132 0.170 0.185 0.188 0.212
  d004rl_shape05_lr6e-6_mini3_s15k   s2 done  234m final 0.222 best 0.232 t.10    8 spm  64.1 ent 0.07 inf 23.1 ok        | 0.123 0.170 0.165 0.210 0.222
= d004rl_shape05_lr6e-6_mini3_s15k     n=3 final 0.217 ± 0.005  ok/1-guess/div 3/0/0  median t.10 16m
  d004rl_shape05_lr6e-6_s15k         s0 done  234m final 0.192 best 0.200 t.10   16 spm  64.1 ent 0.07 inf 22.1 ok        | 0.145 0.162 0.192 0.177 0.192
  d004rl_shape05_lr6e-6_s15k         s1 done  234m final 0.180 best 0.180 t.10   16 spm  64.1 ent 0.05 inf 28.3 ok        | 0.102 0.153 0.163 0.178 0.180
  d004rl_shape05_lr6e-6_s15k         s2 done  234m final 0.220 best 0.227 t.10   16 spm  64.1 ent 0.05 inf 27.2 ok        | 0.120 0.150 0.197 0.225 0.220
  d004rl_shape05_lr6e-6_s15k         s3 done  234m final 0.233 best 0.233 t.10   16 spm  64.1 ent 0.07 inf 26.3 ok        | 0.120 0.147 0.185 0.205 0.233
  d004rl_shape05_lr6e-6_s15k         s4 done  234m final 0.193 best 0.202 t.10   16 spm  64.1 ent 0.06 inf 26.7 ok        | 0.127 0.158 0.165 0.198 0.193
= d004rl_shape05_lr6e-6_s15k           n=5 final 0.204 ± 0.022  ok/1-guess/div 5/0/0  median t.10 16m
```

Final eval (step 15,000; dev ladder, greedy, 600 puzzles):

| seed | r56 | r30 | difference |
|---|---|---|---|
| 0 | 0.217 | 0.192 | +0.025 |
| 1 | 0.212 | 0.180 | +0.032 |
| 2 | 0.222 | 0.220 | +0.002 |
| mean | 0.217 | 0.197 | **+0.019** (SE 0.009, t 2.14 on 2 df, p 0.17) |

r30's seeds 3 and 4 ended at 0.233 and 0.193, so its five-seed mean is 0.204 (SD 0.022), and r56's three seeds are +0.013 above it. r56's seeds are close together (SD 0.005).

Paired difference over the run (the mean of the three pairs at each eval; evals every 500 steps):

| steps | mean difference | range | evals above zero |
|---|---|---|---|
| 500-2,500 | −0.008 | −0.014 to −0.001 | 0 of 5 |
| 3,000-7,500 | +0.004 | −0.007 to +0.020 | 6 of 10 |
| 8,000-15,000 | +0.014 | +0.004 to +0.029 | 15 of 15 |

Over the last three evals (steps 14,000-15,000) the pairs are +0.035, +0.032 and 0.000, mean +0.022.

By level at the final eval (pass@1, 100 puzzles per level):

| | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|
| r56, seeds 0-2 | 0.677 | 0.190 | 0.087 | 0.080 | 0.030 | 0.237 |
| r30, seeds 0-2 | 0.667 | 0.117 | 0.067 | 0.060 | 0.027 | 0.247 |
| r30, seeds 0-4 | 0.674 | 0.126 | 0.072 | 0.076 | 0.030 | 0.244 |
| difference (seeds 0-2) | +0.010 | +0.073 | +0.020 | +0.020 | +0.003 | −0.010 |

What it means:

- **A small gain that shows up late.** The arm is behind for the first 2,500 steps (a third of its prompts are mini puzzles), level until about step 7,500, and ahead at every eval from step 8,000, by 0.014 on average and 0.019 at the end. Three pairs don't make that significant (p 0.17). Seed 2's pair is level because r30's seed 2 was one of its two strong seeds.
- **All of it is at L2.** L2 goes from 0.12 to 0.19, and every r56 seed's L2 (0.18-0.21) is above every one of r30's five (0.11-0.16). Treating each puzzle-seed as independent (300 a side), that is about 2.5 standard errors. L3 and L4 are +0.02 each, inside the noise; L1 and L5 are level and L6 is −0.01. So M2 transfers to L2, and M3 and M6 show no transfer to L3 and L6.
- **Training signals.** Over steps 14,000-15,000 r56 has 21.9 / 24.1 / 22.9 informative groups of 64 against r30's 23.3 / 28.8 / 27.6 (seeds 0-2), and entropy 0.050 / 0.062 / 0.071 against 0.070 / 0.059 / 0.046. So the mix gives slightly fewer informative groups late in training (the train log doesn't split them by level), and entropy is no different. Time per step varies more between seeds than between arms: the update takes 0.42 / 0.38 / 0.57 s for r56's seeds, which shared one job, and 0.61 / 0.29 / 0.21 s for r30's. So this screen says nothing about the cost of a step.
- **Against the D line.** r30's five seeds are at 0.204 and D 005's in-loop dev mean is 0.201. A gain of 0.02 puts the mini levels alone near 0.22, short of 0.231.

**What follows.**

- The rule for r56 was: a clear gain goes to an H200 gate, and a clearly negative result drops r55. It is neither. **r55 stays queued** (RL from D 005's checkpoints with the mini levels, time-budgeted on H200, paired with D 005's seeds 0-2). It asks the same question under the record's clock and from the record's checkpoints.
- r56 seeds 3 and 4 stay in `screens_d_b200`, after r57's first three seeds. Five pairs would bring the SE to about 0.007.
- The mini levels become a part of the D10 stack, with whatever r59 (self-imitation) and r57 (no L5) show. None of the three looks likely to reach the line alone.
- D6's follow-ups for the 11-18 box's track D half, fixed by 11-16: M2 alone (the mini level that transferred) and a larger share of it, and intermediate levels aimed at L3-L5, where dev is 0.03-0.09 and the mini levels did nothing. Fading the mini levels out is dropped, because the difference was still growing at the end.

**Jobs.** No GPU job of mine has run since 20:54 EST. Every pending job waits on priority. Mine is about 365,000 on `mit_normal_gpu` (fairshare 4,513 after three weeks of use) against 515,000-565,000 for the jobs Slurm has planned onto the free B200 and H200 GPUs, and my preemptable jobs are at 36,000-38,000. `sbatch --test-only` puts a new job of any length, on any GPU type, at 11-12 to 11-14. Those estimates have been too late before: two B200 jobs started within minutes at 15:03 and 15:41 today. What I did:

- 28653623 (B200, `mit_normal_gpu`, `screens_d_b200`, 3 runs packed), submitted at 21:14 EST, takes 28623801's place. r59 seeds 0-2 are first in its queue.
- 28653733 (H200, `mit_normal_gpu`, new queue `gate_a_h200`), submitted at 21:18 EST with `--time-min=03:15:00` and `RESERVE_MIN=180`. Slurm may start it in a gap of 3 h 15 min, where the other H200 workers need 6 h, and an r58 seed takes about 2.5 h. The queue holds only r58 seeds 1 and 2 (moved out of `h200` and `screens_a_h200`), because a worker with a short reserve must not pop r55 or r53, which need 4-6 h. r58 seed 0 stays first in `h200`, where 28473022 and two preemptable workers read it.
  The submit line, from `slurm_logs`:
  ```
  sbatch --parsable -p mit_normal_gpu --time=06:00:00 --time-min=03:15:00 --gres=gpu:h200:1 --export="ALL,RLVR_DIR=/home/zjason/rlvr-screens,RUNNER=scripts/research/run_item.sh,RESERVE_MIN=180,REQUIRE_GPU=NVIDIA H200,VENV=/home/zjason/orcd/pool/vllm-venv,HF_HUB_OFFLINE=1,QUEUE=gate_a_h200" ../scripts/slurm/queue_worker.sbatch
  ```
- GPU jobs: 9 pending, none running. Five are on `mit_normal_gpu`, where two run at a time, and four on preemptable, so at most 6 can run together. Slurm's estimate for 28646346 (B200, r60 seed 0) is 02:38 EST on 11-11. For the H200 jobs 28473022 and 28595679 it is 11-14 15:00 and 11-15 12:30 (read at 21:13 EST; squeue prints local time).

Queued work: `screens_d_b200` 10 runs (about 19 GPU-hours packed), `screens_b200` 7 runs (about 28 GPU-hours), and the three H200 queues about 37 GPU-hours.

## 2026-11-10 21:46 EST: hindsight relabelling is coded for its one unranked screen (`hindsight_rows`, PLAN D12); still no GPU job running

INBOX 11-10 14:54 allows one unranked track D screen of hindsight relabelling, to measure its effect. It stays not allowed in records, promotion screens and anything stacked into a ranked arm. The option is in the trainer (commit 6d3aeab) and off by default, so existing configs and the queued arms behave as before.

**What it does.** With `hindsight_rows` above 0, each step looks at every sample the verifier rejected. A sample gives a usable relabelled pair when all of these hold:

- It is a well-formed expression that uses each of the puzzle's numbers once and only the puzzle's operators, so it missed by its value alone.
- Its value is an integer inside the level's generator target range (10-200; 100-999 for L5). This is a range check on the value, which keeps relabelled targets where the generator draws them.
- The relabelled puzzle (the same numbers, operators and level, with that value as its target) is in neither ladder set.

The pair is the relabelled puzzle with the sample's tokens exactly as sampled. Nothing solves or searches a relabelled pair, and the level's "needs" rule is not checked on it, as the note asks. The pairs go into self-imitation's buffer: in memory, empty at each run's start and after a resume, never saved, the newest `sil_buffer` pairs per level. Each step replays up to `hindsight_rows` of them, drawn evenly over levels, and each replayed row weighs `hindsight_coef` times a policy-gradient row. A step never replays its own samples. The option can't be combined with `sil_rows`, and it needs the ladder training mix and one process.

**What it logs.** Each step writes `hs_share`, the usable pairs over all of the step's samples (the share the note asks for), with `hs_added`, `hs_buffer`, `hs_rows` and `hs_nll`.

**The guard.** `validate_record_v3.py` now fails a record whose seed config has `hindsight_rows` on.

**Checks.** `ruff` is clean and all 155 tests pass. Five are new: what `hindsight_puzzle` accepts and rejects, the replayed rows' weight, a step, a run (off by default, the ladder exclusion, the refused combinations) and the validator.

**The arm.** It goes on the 11-18 H100 box, where it costs the ranked arms nothing. Both arms are RL with D 005's settings from saved pretrained checkpoints, step-capped, three seeds each on the same packing: a control arm and the relabel arm. The relabel arm replays 64 rows at coefficient 0.5, the same as r59 (self-imitation), so the two kinds of replay can be read side by side. Its `description` and its directory name will say "unranked". The configs are due with the rest of the box's by 11-16. The result will give the paired difference, the per-level numbers and the mean `hs_share`.

One limit belongs with the result. Track D's completions are answer-only, so a stored completion fits its relabelled prompt exactly. On track A a think section would still reason about the old target, so this screen says nothing about track A.

**Jobs.** No GPU job of mine has run since 20:54 EST. The 9 workers are pending, 8 on priority and one on unavailable nodes. At 21:45 EST squeue's estimates were 03:00 EST on 11-11 for 28646346 (B200, r60 seed 0) and 05:00 EST for 28653623 (B200, r59 seeds 0-2). They are estimates and have moved at every read.

Next in the code order: rollback and train on (PLAN A8 (c)), by 11-16, for the box.

## 2026-11-11 12:03 EST: workers resubmitted after the Slurm-side cancellation

All nine of my pending workers were cancelled at 08:12:56 EST 11-11. The maintainer checked (INBOX
12:01): more than 1,800 other users' pending jobs were cancelled between 08:12 and 08:16 with no
cancelling uid, so it was a Slurm-side event. The queues were intact; the same nine workers are
resubmitted (job IDs 28702298-28702307, see STATUS), at most 6 GPUs together. Still in the
low-usage window until 18:00 EST.

## 2026-11-11 19:12 EST: r58 seeds 0-1 pass the gate so far; B200 and node5200 host failures

```
  a005_ent05_keepbest_h200   s0 done 232m final 0.363 best 0.363 ok        | 0.055 0.085 0.223 0.315 0.363 | kept step 400, last 0.363
  a005_ent05_keepbest_h200   s1 done 232m final 0.398 best 0.398 diverged  | 0.017 0.107 0.273 0.368 0.392 | kept step 350, last 0.392
  d005rl_mini3               s0 done 234m final 0.233 best 0.237 ok        | 0.120 0.130 0.173 0.213 0.233
```

- r58 (A 006's promotion screen, H200): kept dev 0.363 and 0.398, both above the 0.305 bar. Seed 1's
  "diverged" label does not count as collapsed: r58 is answer-constrained and its final dev is far above
  the holder's dev mean (PLAN, clarification of 10-23). Seed 2 decides; it is first in `h200`.
- r55 seed 0 (D 005 + mini3, H200): 0.233 dev against D 005's dev mean 0.201. One seed; read at 3.
- Host failures, requeued once at the front: the three B200 workers that started at 12:04 EST all
  landed on node5500-c1 and every item failed within about a minute with CUDA out of memory (118 MiB
  free of 178 GiB, this process 17.7 GiB). r60 s0, r59 s0-2 and r57 s0-1 were lost from the queues
  and are back. node5500-c1 is excluded with node5501-c1.
- 28702305/28702306 (preemptable H200, node5200) ran 7 h with an empty log and no CPU time:
  cancelled, node5200 excluded, replaced by 28742593.

## 2026-11-12 00:11 EST — r58 meets its rule; record A 006 queued

- r58 seed 2 kept dev 0.330 at step 300 (evals 0.225, 0.302, 0.330), so all three seeds are at or above the 0.305
  bar (0.363 / 0.398 / 0.330) and none collapsed: r58's 3-seed rule is met. A kept value can only rise, so seed 2's
  remaining evals cannot undo this; its final value goes in the record README.
- Record A 006: `records/v3/track_a/006_fp32_mal0_cos500_s400_ent05_keepbest` (config f04191c) is r58's config without
  `no_save`. Seeds 0-4 are at the front of `h200`, whose workers run `run_item.sh`; it runs a records/v3 item as
  `STAGE=all scripts/run_seed_v3.sh`, as A 005's seeds ran. No separate record-<slug> queue: the H200 workers that
  could take one are already in the 6-GPU count, and these keep their place in Slurm's priority.
- r60 seed 0 (B200, 700-step cosine + keep_best_dev): kept 0.428 at step 450, last eval 0.337. Read at 3 seeds
  against PLAN A7 (a) (kept >= 0.41 on two of three).

## 2026-11-12 01:11 EST — r58 seed 2 preempted at step 350; not rerun

- 28702304 (node4901) was preempted at 00:43 EST while running r58 seed 2. Its evals: 0.035, 0.060, 0.158, 0.225,
  0.302, 0.330, 0.388 (steps 50-350), so it kept 0.388 at step 350. r58's three kept values are 0.363 / 0.398 / 0.388+.
- I removed the worker's redo file (`queue/running/28702304`), so on restart it skips the 5-h rerun of r58 seed 2 and
  takes A 006 seed 0 from the front of `h200`. The rerun could not change r58's verdict, which was already met at
  step 300, and the record needs the H200. r58 seed 2 stays a partial seed (no done.txt line); the A 006 README
  reports it as such.

## 2026-11-12 06:08 EST — r59 (self-imitation): clear gain over r30; H200 gate r61 queued; A 006 seeds 0-1 running

Track D, B200 packed 3 per GPU, step-capped at 15000, RL from D 004's checkpoints (summarize_screens):

| arm | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| r59 sil64 c0.5 | 0.278 | 0.248 | 0.232 | | | 0.253 (n=3) |
| r30 control | 0.192 | 0.180 | 0.220 | 0.233 | 0.193 | 0.204 (n=5); 0.197 on s0-2 |

- Paired by seed (python): +0.086, +0.068, +0.012, mean +0.055. All three seeds gained, two by a lot, and every r59
  seed is at or above D's line 0.231. This is the largest single-change gain on track D since the shaping change.
  The step cap hides the replay's update time, so the number that counts is the time-budgeted one.
- Next, as PLAN D11 says for a clear gain: r59 seeds 3-4 go to the front of `screens_d_b200`; the time-budgeted H200
  gate r61 (`calibration/research_r61/d005rl_sil64c05`: D 005's RL phase from D 005's own checkpoints plus r59's
  self-imitation settings, 5 seeds, against 0.231, paired with D 005's in-loop dev 0.193 / 0.187 / 0.210 / 0.217 /
  0.200) is queued in `h200`, alternating with A 006's seeds 2-4; a coefficient and row-count sweep goes to the
  11-18 box. STATUS says at the top that a self-imitation arm reached the promotion numbers on its screen, as
  INBOX 11-10 14:50 asks; a D 006 record waits for the maintainer's look.
- r51 (A 005 with oversample 2, B200) seeds 0-1: last dev 0.053 / 0.160 after peaks of 0.390 / 0.363. It peaks like
  A 005 and then falls apart; without keep_best_dev it is worse than A 005. Read at more seeds.
- r60 seed 1 (B200, 700-step cosine + keep_best_dev): kept 0.322 at step 300, still running.
- A 006: seed 0 on 28742593 (node4601) since 03:58 EST, seed 1 on 28702304 (node4801, restarted after the
  preemption) since 06:03 EST.

## 2026-11-12 09:07 EST — A 006 seeds 0-1 done; r60 meets PLAN A7 (a)'s bar on 2 of 2

- Record A 006 (test ladder): seed 0 test 0.397 (kept step 400), seed 1 test 0.375 (kept step 350);
  mean so far 0.386. Seeds 2-4 follow (seed 2 on 28742593 since 06:42 EST); the README and validation come
  after all five. (Holder A 004: 0.2671.)
- r60 (r58's config on a cosine schedule over 875 steps stopped at 700, keep_best_dev, B200 step-capped): kept
  0.428 (step 450, last 0.337) and 0.462 (step 650, last 0.457), mean 0.445. PLAN A7 (a) asks
  for a kept dev of 0.41 or above on two of three seeds, so the 700-step schedule stays in the 11-18 box batch;
  seed 2 is still queued in `screens_b200`. Seed 1 kept improving to step 650, which is the first sign that the
  extra steps raise the level rather than only holding it. Whether 700 steps fit in A's 5 h H200 budget with the
  evals is checked at the next full wake (A 006's 400 steps took 8413 s).
- r61 (the H200 self-imitation gate) seed 0 started on 28702304 (node4801) at 08:55 EST.

## 2026-11-12 12:06 EST — Correction: A 006's "test" numbers were kept dev; r62 (700 steps on H200) queued; r61 seed 0 ahead

- **Correction** (INBOX 11-12 11:56). The entries of 11-12 09:07 EST and the 10:07 status gave A 006's seeds as
  "test 0.397 / 0.375 (/ 0.397)". Those are the kept in-loop dev values (`result.json` `final_pass_rate`, 600
  dev puzzles). The test-ladder scores are in each seed's `confirm_v3.json` (2,400 puzzles x 16 samples):

  | A 006 seed | kept step | kept dev | test |
  |---|---|---|---|
  | 0 | 400 | 0.397 | 0.4099 |
  | 1 | 350 | 0.375 | 0.3932 |
  | 2 | 400 | 0.397 | 0.4148 |

  The runs themselves are fine: the in-loop evals read the dev ladder only. The README will take its numbers from
  `confirm_v3.json`. Seed 3 runs on 28742593 since 09:23 EST; seed 4 is next in `h200`.
- **INBOX 11-12 11:56:** a self-imitation record (D 006) may go ahead without another look if r61 meets the
  promotion rule, under the limits of 11-10 14:50. r61 seed 0 is at dev 0.235 after 188 min (best 0.247), against
  D 005's seed 0 at 0.193.
- **r62 queued** (`calibration/research_r62/a006_cos875_s700_h200`): r60's config unchanged (cosine over 875 steps,
  stop at 700, keep_best_dev, 17,400-s budget), time-budgeted on H200, 3 seeds interleaved with r61 in `h200`.
  A 006's 400 steps took 8,464 s, so 700 should take about 14,800 s. It is the promotion screen for A 007 under
  PLAN A7 (a), read kept-against-kept against A 006's line once A 006 is done. r60 seed 2 (B200) stays queued.
- r51 seed 2 is at 0.402 (step 350) and still running; seeds 0-1 collapsed to 0.053 / 0.160 after peaks of 0.390 /
  0.363. r57 seeds 0-2 (no level 5 in training) are at 0.185 / 0.193 / 0.212 near the end, against r30's
  0.192 / 0.180 / 0.220: level.

## 2026-11-12 13:06 EST — A 006 seed 3 test 0.4149; r61 seed 0 dev 0.250 (D 005 seed 0: 0.193)

- A 006 seed 3: kept step 400, kept dev 0.392, test 0.4149 (`confirm_v3.json`). Seed 4 runs on 28702304 since
  12:51 EST, so the record closes about 15:20 EST.
- r61 (H200 self-imitation gate, time-budgeted) seed 0: final dev 0.250, at or above D's line 0.231, paired
  +0.057 over D 005's seed 0. One seed of five; seed 1 runs on 28742593.

## 2026-11-12 16:05 EST — Record A 006: test 0.4123 ± 0.0127, the new track A holder (Welch vs A 004 p = 0.0019); r51 closed

- **A 006** (`records/v3/track_a/006_fp32_mal0_cos500_s400_ent05_keepbest`), all five seeds, test from `confirm_v3.json`:

  | seed | node | kept step | kept dev | test | pass@16 | mean tokens | total_s |
  |---|---|---|---|---|---|---|---|
  | 0 | node4601 | 400 | 0.397 | 0.4099 | 0.6750 | 703 | 8464 |
  | 1 | node4801 | 350 | 0.375 | 0.3932 | 0.6825 | 726 | 8722 |
  | 2 | node4601 | 400 | 0.397 | 0.4148 | 0.6879 | 718 | 8315 |
  | 3 | node4601 | 400 | 0.392 | 0.4149 | 0.6763 | 712 | 9018 |
  | 4 | node4801 | 400 | 0.380 | 0.4285 | 0.6846 | 697 | 9586 |

  Mean test 0.4123 ± 0.0127; kept dev mean 0.388. `validate_record_v3.py --compare-to` A 004: t = 5.61, df 4.4,
  one-sided p = 0.0019, "new record". Against A 005 (not a holder): +0.072, p = 0.059. No seed was preempted.
  In plain words: A 005 had two seeds of five that shortened their outputs and scored 0.22 / 0.29; here every seed
  kept long outputs (697-726 tokens) and the spread fell from 0.082 to 0.013. The checkpoint choice moved one seed
  (seed 1, step 350 over 400, 0.375 against 0.365 on dev). README written, leaderboard updated.
- **A 006's line for r62 (A 007's promotion screen), kept against kept:** 0.388 + 0.03 = **0.418**. Both sides are
  kept evals, so no extra +0.02. r62's 3-seed form: each seed's kept dev at least 0.418, none collapsed.
- **r51 (A 005 + oversample 2, B200, U11) closed at 3 seeds:** last dev 0.053 / 0.160 / 0.177 after peaks of
  0.390 / 0.363 / 0.402. Every seed collapsed late. Even read at its peaks (mean 0.385) it is level with A 006's
  kept dev, not above it, so oversampling at this lr is no help. Seeds 3-4 are taken out of `screens_b200` and
  replaced by r60 seeds 3-4 (the 700-step schedule the 11-18 box batch builds on), so that arm reads at five seeds.

## 2026-11-12 18:04 EST — r61 seed 1 dev 0.247 (+0.060 over D 005's seed 1); both B200 workers preempted

- r61 (H200 self-imitation gate) seed 1: final dev 0.247 (`summarize_screens`: best 0.247, not collapsed), against
  D 005's seed 1 at 0.187. Seeds 0-1: 0.250 / 0.247, both at or above 0.231, paired +0.057 / +0.060. Seed 2 runs on
  28702304 since 16:03 EST; seeds 3-4 follow in `h200`.
- r62 (A 007's promotion screen) seed 0 runs on 28742593 since 16:16 EST; read against 0.418 (kept against kept).
- 28742594 and 28742595 (B200) were preempted about 16:30 EST and wait in the queue. Their items (r56 seed 3, r57
  seed 1, r59 seed 4, r60 seed 2) are in `queue/running/` and go back to their queues when the workers restart.
  `mit_normal_gpu` 28702299/28702300 still pending (Slurm estimates 11-14). Six GPUs counted, two running.

## 2026-11-12 20:05 EST — r61 passes D's promotion rule (3-seed form); record D 006 (self-imitation) queued

- r61 (H200, time-budgeted, D 005's RL from D 005's own pretrained checkpoints, plus self-imitation: 64 replayed
  rows per step, coefficient 0.5, buffer 4,096) seed 2 ended at dev 0.237, against D 005's seed 2 at 0.210.

  | seed | r61 final dev | D 005 in-loop dev | paired |
  |---|---|---|---|
  | 0 | 0.250 | 0.193 | +0.057 |
  | 1 | 0.247 | 0.187 | +0.060 |
  | 2 | 0.237 | 0.210 | +0.027 |

  `summarize_screens`: n = 3, final 0.244 ± 0.007, none collapsed. Every seed is at least D's line 0.231 (D 005's dev
  mean 0.201 + 0.03), which is PLAN's 3-seed form of the promotion rule. In plain words: the replay of the run's own
  verified answers adds about 0.04-0.05 dev on top of D 005, the same size r59 showed on B200 from D 004's checkpoints.
- INBOX 11-12 11:56 lets the self-imitation record go ahead without another look once r61 passes, under the limits of
  11-10 14:50. `records/v3/track_d/006_dm_math_fp32_shape05_lr6e-6_sil64c05/config.json` (D 005's config plus the
  four `sil_*` keys, nothing else changed) is committed as 3c4e32f. Its five seeds are in
  `queue/record-006_dm_math_fp32_shape05_lr6e-6_sil64c05.txt`. The buffer starts empty in every seed, is never
  carried between runs, replays completions verbatim with their own puzzle, and the verifier is its only filter.
- Workers: I cancelled my two pending `mit_normal_gpu` jobs (28702299 `h200`, 28702300 `screens_a_h200`; Slurm put
  their start at 11-14, and a 6-h job cannot hold a ~5.8-h track D seed plus its test scoring) and submitted two
  preemptable H200 record workers, 28891874 and 28891875 (`QUEUE=record-006_…:h200`, `--time-min=06:30:00`,
  RESERVE_MIN 380), the settings D 005 ran under. They take D 006's seeds first, then `h200`. A preempted seed is an
  infrastructure interruption (INBOX 10-25): the partial goes to `extra/` and the same seed reruns.
- r61 seed 3 keeps running on 28702304 (it started at 19:59); seed 4 is taken out of `h200`, since the promotion is
  decided and the record's seeds are the replication. r55 seed 2 moves from `screens_a_h200` to the end of `h200`.
  GPU jobs: 6 (4 running, 2 pending).

## 2026-11-12 21:05 EST — r62 seed 0 kept dev 0.463 (A 006's line 0.418)

- r62 (A 006's recipe, cosine over 875 steps stopped at 700, keep_best_dev, time-budgeted on H200) seed 0 ended
  `ok` after 232 min: dev 0.013 / 0.097 / 0.380 / 0.435 / 0.463 over its evals, kept step 700 at 0.463. That is above
  A 006's kept line 0.418 and +0.066 over A 006 seed 0's kept 0.397. Seed 1 started on 28742593 at 20:32; seed 2
  follows in `h200`. The 3-seed form needs every seed at 0.418 or more with none collapsed.
- D 006's record workers 28891874/28891875 are still pending. GPU jobs: 6 (4 running, 2 pending).


## 2026-11-13 00:04 EST — r61 seed 3 0.228; r60 seed 3 kept 0.435; r63 (96 prompts per step) queued on B200

- r61 (self-imitation gate, H200) seed 3 ended `ok`: dev 0.228 (best 0.253), against D 005 seed 3's 0.217, +0.011.
  Over its four seeds r61 averages 0.2405 (0.250 / 0.247 / 0.237 / 0.228). It had already passed in the 3-seed form;
  record D 006's five seeds are the replication (workers 28891874/28891875 still pending).
- r60 (A 006's recipe on the 700-step cosine schedule, B200, step-capped) seed 3 kept 0.435 at step 650. Kept dev over
  four seeds: 0.428 / 0.462 / 0.380 (one-guess, slid) / 0.435, mean 0.4263. Seed 4 runs on 28742595.
- r62 seed 1 (H200 gate) is at 198 min: kept 0.412 at step 400, last eval 0.315. It needs 0.418 or more for the
  3-seed form; seed 2 started on 28702304 at 23:55.
- `screens_b200` was about to run dry, so I queued r63 (PLAN A7 (c), `calibration/research_r63/a006_p96_cos583_s467`,
  seeds 0-2): r60's config with 96 prompts per step, cosine over 583 steps stopped at 467, so it sees the same number
  of prompts as r60 (44,832 against 44,800). Read paired against r60's B200 seeds. r20 tried 96 prompts without the
  entropy floor. GPU jobs: 6 (4 running, 2 pending).

## 2026-11-13 01:03 EST — r62 seed 1 kept 0.412, under A 006's line 0.418; r62 goes to five seeds

- r62 seed 1 ended `ok` after 232 min. Dev rose to 0.412 at step 400, then fell late in the schedule: 0.382 at step 500,
  0.093 at step 600, 0.065 at step 700. keep_best_dev kept step 400 (0.412). That is under the line 0.418, so the
  3-seed form can no longer pass. The 5-seed form is still open (mean at least 0.418 with 4 of 5 seeds above it and
  at most 1 collapsed). So far: 0.463 / 0.412, mean 0.438.
- Seed 1's late collapse is the risk of the longer schedule: the lr is still high past step 500. r60's B200 seed 2
  (0.380, one-guess) slid the same way. keep_best_dev protects the score but not the run.
- Seed 2 (67 min in) is at 0.317 at step 200. Its "diverged" flag in the summary is the early malformed rate (0.64 at
  step 200); seed 0 was at 0.53 at the same step and recovered.
- I queued r62 seeds 3-4 at the front of `h200`, ahead of r53 seeds 1-2 and r55 seed 2.

## 2026-11-13 02:03 EST — r59 complete: self-imitation +0.054 over its control r30 on all five seeds (B200)

| seed | r59 (self-imitation) | r30 (control) | diff |
|---|---|---|---|
| 0 | 0.278 | 0.192 | +0.086 |
| 1 | 0.248 | 0.180 | +0.068 |
| 2 | 0.232 | 0.220 | +0.012 |
| 3 | 0.247 | 0.233 | +0.014 |
| 4 | 0.285 | 0.193 | +0.092 |
| mean | 0.2580 | 0.2036 | +0.0544 |

- Seeds 3 and 4 ended `ok` at 0.247 and 0.285. All five seeds beat their paired control (smallest gap +0.012).
  This is the B200 screen behind the H200 gate r61, which passed. Record D 006's five seeds are queued.
- 28742594 now runs r57 seeds 3-4 and r56 seed 4. `screens_d_b200` is empty after these, and the box queue
  rebuild comes at the 06:00 full wake.

## 2026-11-13 04:02 EST — r60 complete (B200, 700-step schedule): kept 0.427 mean, but 3 of 5 seeds slid late

| seed | kept dev (step) | last eval (step 700) |
|---|---|---|
| 0 | 0.428 (450) | 0.337 |
| 1 | 0.462 (650) | 0.457 |
| 2 | 0.380 (300) | 0.105 (one-guess) |
| 3 | 0.435 (650) | 0.418 |
| 4 | 0.432 (450) | 0.230 |
| mean | 0.4274 ± 0.0297 | 0.3094 |

- On kept dev, r60 is +0.039 over A 006's kept mean of 0.388 (B200 step caps; A 006 ran on H200). But seeds 0, 2 and 4
  fell 0.09 to 0.28 after their best eval. The last half of the longer cosine schedule is unstable, and keep_best_dev is
  what holds the score up. r62 (H200) shows the same pattern: seed 1 fell from 0.412 to 0.065, and seed 2 is at 0.432
  kept (step 500) with a last eval of 0.353 at 225 min.
- This points at the end of the schedule, not its length: a lower lr floor or faster decay after step 450, or a
  rollback when dev drops (PLAN A8 (c)). These go into the 11-18 box's track A batch. r63 seed 0 (96 prompts per
  step) started on 28742595 at 03:40.

## 2026-11-13 06:02 EST — r62 seed 2 kept 0.432 (last 0.233); D 006 seeds 0-1 moved onto the running H200 workers

- r62 (H200 gate, 700-step schedule) seed 2 ended `ok`: kept 0.432 at step 500, last eval 0.233 at step 700. It slid
  late like seed 1. Kept so far: 0.463 / 0.412 / 0.432, mean 0.4357. Seed 3 is at 0.385 (step 300, 102 min),
  and seed 4 is queued. For the 5-seed form, seeds 3 and 4 both need 0.418 or more (seed 1 is already under the line).
- The two D 006 record workers (28891874/5) are pending with `QOSMaxGRESPerUser`: the four running preemptable jobs
  fill the per-user GPU cap, so they cannot start until one of them ends. The running `h200` workers use
  `run_item.sh`, which runs a `records/v3/` item as a record (A 006's seeds ran this way on 28742593), and they are on
  H200 SXM nodes (node4601, node4801). I moved D 006 seeds 0-1 to the front of `h200`, so they start as soon as r53
  seed 0 (~06:30) and r62 seed 3 (~08:05) end. Seeds 2-4 stay in the record queue for whichever worker frees first.
  Every seed still runs under `run_seed_v3.sh` on an H200, as RULES_v3 requires.
- In progress: r63 seed 0 (96 prompts per step, B200) kept 0.427 at step 300 (152 min). r53 seed 0 (d005_w768_l24) is
  at 0.212 after 222 min, against D 005 seed 0's 0.193. r57 seeds 3-4 and r56 seed 4 are on the B200 track D worker.

### 2026-11-13 06:05 EST: rollback option, r64 queued

- `rollback_drop` added to grpo.py (off by default; needs keep_best_dev; RULES v3.4 allows rollback when its evals
  count on the clock). An in-loop eval more than `rollback_drop` below the kept best sends training back to the kept
  weights; the optimizer state carries on, and the eval log marks the row `rolled_back_to`. Tested (156 pass).
- r64 = r60's config + `rollback_drop` 0.05, seeds 0-2 queued on B200 behind r63 s1-2 (PLAN A8 (c)). It targets the
  late-schedule slide (r60 s0/s2/s4, r62 s1-2). Read it against r60's B200 seeds: kept dev, and whether the
  rolled-back runs climb past their kept eval. Also a candidate for the 11-18 box batch.

### 2026-11-13 08:10 EST: r62 fails promotion (s3 kept 0.390); r62 s4 dropped; r53 s0

- r62 s3 (H200): kept 0.390 at step 350, then collapsed to one-guess (0.042 at step 400, last 0.053). With s1
  (0.412) also under the 0.418 line, two of five seeds are below it, so the 5-seed form can't pass. No A 007 from r62.
  Kept dev over four seeds: mean 0.4243 ± 0.0310. r62 s4 is removed from the h200 queue (it
  can't change the verdict); the H200 time goes to D 006 and the D screens.
- The slide is now in four of r62's seeds after the kept eval (s1 0.065, s2 0.233, s3 0.053 at the end), worse
  than r60 on B200. That is what r64 (rollback_drop 0.05) tests; keep_best_dev alone leaves the kept eval early.
- r53 s0 (D, w768 l24) finished: final 0.200, best 0.222 (D 005 s0 0.193). The worker marked it FAILED only because
  the post-run `mv` of `final/` found nothing (no_save); the result is complete and is not rerun.

### 2026-11-13 12:00 EST: r63 s0-1, r56 and r57 results; two mit_normal_gpu B200 screen workers

- r63 (A 006 + 96 prompts per step, 583-step cosine stopped at 467, B200) kept dev against r60's B200 seeds:

  | seed | r63 kept (step) | r63 last | r60 kept | diff |
  |---|---|---|---|---|
  | 0 | 0.4583 (350) | 0.420 | 0.4283 | +0.0300 |
  | 1 | 0.4383 (400) | 0.432 | 0.4617 | -0.0233 |

  Mean diff +0.0033 over two seeds. No late slide in either seed. Seed 2 is queued.
- r56 (D, mini3) against r30, all five seeds paired:

  | seed | 0 | 1 | 2 | 3 | 4 | mean |
  |---|---|---|---|---|---|---|
  | r56 | 0.2167 | 0.2117 | 0.2217 | 0.2250 | 0.2150 | 0.2180 |
  | r30 | 0.1917 | 0.1800 | 0.2200 | 0.2333 | 0.1933 | 0.2037 |
  | diff | +0.0250 | +0.0317 | +0.0017 | -0.0083 | +0.0217 | +0.0143 |

  Positive in 4 of 5 and with a much tighter spread (sd 0.0053), but no seed reaches the 0.231 D line. It is a
  variance reducer, not a record; it is a candidate to stack on D 006's setup.
- r57 (D, no L5) seeds 2-4 against r30: 0.2233/0.2250/0.2100 vs 0.2200/0.2333/0.1933, mean diff +0.0039. Null.
  Seeds 0-1 failed on node5800-c1 (vLLM "No available memory for the cache blocks" after a preemption restart,
  already requeued once); they are not rerun, since three seeds already show no effect.
- The preemptable QOS holds at 4 GPUs (all D 006 seeds). Two mit_normal_gpu B200 workers (28967959, 28967960; 6 h)
  take `screens_b200` (r63 s2, r64 s0-2), so the total is 6.

### 2026-11-13 13:00 EST: D 006 seed 0 done; B200 workers hit a bad node

- D 006 seed 0 (self-imitation record) finished: test ladder_score 0.2252 (confirm_v3.json), dev 0.220. D 005's
  test is 0.2150. Seeds 1-3 are running and seed 4 waits; no verdict until all five are in.
- Both mit_normal_gpu B200 workers (28967959, 28967960) landed on node5801-c1 and failed in seconds: the node's
  `/lib64/libstdc++.so.6` lacks GLIBCXX_3.4.26, so the cached flashinfer `fmha_gen.so` won't load. r63 s2 and r64 s0
  went to FAILED and are back at the head of `screens_b200`; r64 s1-2 were put back by the worker. Resubmitted as
  28973352 and 28973353 excluding node5500-c1, node5501-c1, node5800-c1 and node5801-c1.

### 2026-11-13 14:10 EST: the B200 failures were my submit environment, not the nodes

- The resubmitted worker 28973352 failed the same way on node5701-c1, where r60 and r63 ran fine. The cause is
  that B200 workers must start through `packed_worker.sbatch`, which puts Spack GCC 12's libstdc++ first on
  LD_LIBRARY_PATH (LOG 2026-10-28). I had used plain `queue_worker.sbatch` from a shell without the login
  environment. Resubmitted as packed workers (PER_GPU=1, 6 h, mit_normal_gpu) 28978213 and 28978214 from a login
  shell; r63 s2 is back at the head of `screens_b200`. The node exclusions in the 13:00 entry were unnecessary.

### 2026-11-13 15:00 EST: D 006 seeds 1-2 in; B200 packed workers running

- D 006 test ladder_score (confirm_v3.json) against D 005's same seeds:

  | seed | D 006 | D 005 | diff |
  |---|---|---|---|
  | 0 | 0.2252 | 0.2150 | +0.0102 |
  | 1 | 0.2487 | 0.1711 | +0.0775 |
  | 2 | 0.2662 | 0.2183 | +0.0479 |

  Mean over three seeds 0.2467 (+0.0452 paired). Seed 3 is running and seed 4 started at 19:34Z on
  node4801. The verdict waits for all five.
- The packed B200 workers 28978213 and 28978214 started r63 s2 and r64 s0 on node5601-c1 and are past warm start.

### 2026-11-13 17:00 EST: D 006 seed 3

- D 006 seed 3: test 0.2335 against D 005 seed 3's 0.2516 (-0.0182), the first seed below its D 005 pair. Four-seed
  mean 0.2434. Seed 4 is running on node4801; its worker 28891875 moved on to r55 s2.

### 2026-11-13 18:05 EST: r63 seed 2, r64 seed 0, r53 seed 1; r65-r67 queued (D stack, sil coef, the D12 screen)

- r63 (A 006 + 96 prompts per step, B200, step-capped) against r60's B200 seeds, kept dev:

  | seed | r63 kept | r60 kept | diff |
  |---|---|---|---|
  | 0 | 0.4583 | 0.4283 | +0.0300 |
  | 1 | 0.4383 | 0.4617 | -0.0233 |
  | 2 | 0.4350 | 0.3800 | +0.0550 |

  Mean diff +0.0206 (sd 0.040). Seed 2's gain is mostly r60 seed 2's slide (kept at step 300). r63 never slid
  (last within 0.02 of kept in all three seeds). Seeds 3-4 are queued on B200 so the pairing covers all five of r60's seeds.
- r64 (r60 + rollback_drop 0.05) seed 0: kept 0.4417 at step 650, last 0.4367, against r60 seed 0's 0.4283 kept at step
  450 and last 0.337 (+0.0133 kept). The slide r60 seed 0 had is gone. Seeds 1-2 are queued.
- r53 (D 005 at 24 layers) seeds 0-1 against D 005's dev: 0.2000 / 0.1933 vs 0.1933 / 0.1867, +0.0067 each. Seed 1's
  worker marked it FAILED only because `mv final/` found nothing under no_save (its result and clock are complete,
  within budget); it is not rerun. Seed 2 is running. Depth looks like width (D5): no gain.
- The two packed B200 workers exited at 22:18Z and 22:24Z with under 250 min left, as designed.
- Queued on `h200` (preemptable H200, time-budgeted, from D 005's pretrained checkpoints, 3 seeds each, paired against
  r61):
  - r65 = r61 (D 006's RL setup) + the mini levels M2/M3/M6 (PLAN D10 stack; r56 was +0.014 paired with a tight spread).
  - r67 = the one unranked hindsight-relabelling screen (PLAN D12, INBOX 11-10 14:54): D 005's RL with hindsight_rows
    64, coef 0.5, no self-imitation. Paired against D 005's dev and r61. Never promoted or stacked. It runs here instead
    of on the 11-18 box.
  - r66 = r61 with sil_coef 1.0.
- Workers: 29004220 (preemptable H200, `h200`), 29004221 and 29004222 (mit_normal_gpu B200 packed, PER_GPU=1,
  `screens_b200`: r64 s1-2, then r63 s3-4). With D 006 s4, r55 s2 and r53 s2 running, that is 6 GPUs.

### 2026-11-13 19:05 EST: B200 worker on node5500-c1 failed (GPU memory held), resubmitted with the exclusion

- 29004221 landed on node5500-c1, where vLLM found only 17 of 178 GiB free at startup (another process held the GPU).
  r64 s1 went to FAILED and r63 s3 was put back after the second quick failure. 18:00's submit left out the
  `--exclude=node5500-c1,node5501-c1` that the earlier B200 workers carried. r64 s1 is back at the head of
  `screens_b200`; replacement 29011155 carries the exclusion. 29004222 runs r64 s2 on node5601-c1.

### 2026-11-13 20:02 EST: r53 closed (24 layers, +0.0072 over three seeds)

- r53 seed 2: dev 0.2183 against D 005 seed 2's 0.2100 (+0.0083). With seeds 0-1 (+0.0067 each), the mean is +0.0072.
  Depth gives nothing, like width (r50, r52); PLAN D5 is closed. The FAILED mark is again only the no_save `mv final/`.
- The H200 worker 28702304 moved on to r65 seed 0. The replacement B200 worker 29011155 runs r64 seed 1 on node5702-c1.

### 2026-11-13 21:02 EST: Record D 006 (D 005 + self-imitation) is the new track D holder, 0.2444 vs 0.2150

- All five seeds ran on preemptable H200 without preemption (348.0-348.9 min each). Test is `confirm_v3.json`'s
  ladder_score; dev is result.json's kept pass rate. D 005's seeds pair by seed number.

  | seed | test | D 005 test | diff | dev |
  |---|---|---|---|---|
  | 0 | 0.2252 | 0.2150 | +0.0102 | 0.2200 |
  | 1 | 0.2487 | 0.1711 | +0.0776 | 0.2383 |
  | 2 | 0.2662 | 0.2183 | +0.0479 | 0.2433 |
  | 3 | 0.2335 | 0.2516 | -0.0181 | 0.2200 |
  | 4 | 0.2485 | 0.2191 | +0.0294 | 0.2250 |
  | mean | 0.2444 ± 0.0158 | 0.2150 ± 0.0287 | +0.0294 | 0.2293 |

- `validate_record_v3.py --compare-to 005`: "new 0.2444 vs holder 0.2150: Welch t=2.00 df=6.2 one-sided p=0.0451 -> new
  record". The README states that the record uses self-imitation and its limits (empty buffer per seed, never carried
  between runs, completions replayed verbatim with their own puzzle, the verifier the only filter). The leaderboard
  now lists 006 as holder. Committed in f73c704.
- The track D promotion line moves to D 006's dev mean + 0.03 = 0.259 (5-seed form), or every seed +0.03 over the
  paired r61 seed (3-seed form). r65 and r66 are paired against r61, D 006's own screen.
- r55 (D 005 + mini levels M2/M3/M6, no self-imitation) seed 2: dev 0.1950 vs D 005's 0.2100. Three seeds: 0.2333 /
  0.1900 / 0.1950 vs 0.1933 / 0.1867 / 0.2100, diffs +0.040 / +0.003 / -0.015, mean +0.009. Not a gain on its own;
  r65 tests it on top of self-imitation.

### 2026-11-13 23:05 EST: r64 (rollback_drop 0.05) done on three seeds: +0.013 kept, the slides are gone

- r64 = r60 (A 006's config on the 700-step schedule, B200, step-capped) + `rollback_drop` 0.05. Kept dev / kept step /
  last dev, against r60's same seed:

  | seed | r64 kept | step | r64 last | r60 kept | step | r60 last | kept diff |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.4417 | 650 | 0.4367 | 0.4283 | 450 | 0.3367 | +0.0133 |
  | 1 | 0.4067 | 500 | 0.3983 | 0.4617 | 650 | 0.4567 | -0.0550 |
  | 2 | 0.4600 | 650 | 0.4500 | 0.3800 | 300 | 0.1050 | +0.0800 |
  | mean | 0.4361 | | 0.4283 | 0.4233 | | 0.2994 | +0.0128 (sd 0.068) |

- Rollback does what it should: the last eval sits within 0.01 of the kept one in every seed, where r60 slid to 0.105 in
  seed 2. The kept score gains only +0.013 on average, though, and seed 1 lost 0.055 (its run took a worse path after a
  rollback). Seed 1 is under A's line 0.418, so r64 does not pass the 3-seed form. r63 (96 rows, s0-2 0.4583 / 0.4383 /
  0.4350, all above 0.418) remains track A's lead; its seeds 3-4 go next.
- Both B200 workers exited normally with under 250 min left. Two new ones (29037156, 29037159, with the node5500/5501
  exclusion) take r63 s3 and s4 from `screens_b200`.

## 2026-11-14 00:10 EST: B200 first (INBOX 11-13 23:43 and 23:45); r65 s0; new B200 screens r68-r70

- INBOX 23:43 and 23:45 read and followed. Progress now comes before records: nothing waits for an H200. The running H200
  workers finish what they hold and are not cancelled. When a worker ends, a B200 worker replaces it. Gates and records
  wait in a backlog listed in STATUS until the maintainer says otherwise.
- **r63 is not yet gated.** Its B200 seeds 0-2 (kept 0.4583 / 0.4383 / 0.4350) all clear A's line 0.418, so they meet the
  3-seed form on B200. The H200 gate, about 14,900 s on an H200 by the B200 timing, goes into the backlog as the first item.
- r65 (r61 plus three minibatches, timed H200 from D 005's checkpoints), seed 0: final dev 0.2483 vs r61 seed 0's
  0.2500 (-0.002). Seeds 1-2 are running.
- `h200` loses r66's three seeds (sil coef 1.0 from D 005's checkpoints, timed), so the H200 workers exit sooner. Only r67
  seed 2 (the unranked hindsight screen) stays there. The same change returns as B200 step-capped screens with r59 as the
  paired control (config commit 65ce423):
  - r68: r59 with sil coef 1.0 (seeds 0-2, `screens_d_b200`, packed 3 per GPU).
  - r70: r59 with 128 replayed rows (seeds 0-2, `screens_d_b200`).
  - r69: r63 plus D 006's self-imitation (64 rows, coef 0.5, by level). It is track A's first self-imitation screen, read
    against r63's B200 seeds (seeds 0-2, `screens_b200`).
- Jobs: two B200 workers on r63 s3-4; four preemptable H200 workers on r65 s1-2 and r67 s0-1. All six slots are in use.
  As each H200 worker ends, a B200 worker takes its slot (D queue: PER_GPU=3).

## 2026-11-14 01:04 EST: r65 (self-imitation plus three minibatches) is null against r61; a B200 track D worker takes the free slot

| seed | r65 (mini3) | r61 | diff |
|---|---|---|---|
| 0 | 0.2483 | 0.2500 | -0.0017 |
| 1 | 0.2333 | 0.2467 | -0.0133 |
| 2 | 0.2517 | 0.2367 | +0.0150 |

- The mean diff is -0.0000 (r65 mean 0.2444). Three minibatches per step add nothing on top of self-imitation from D
  005's checkpoints, as r55 found without it (+0.009). r65 is closed; nothing to gate.
- 28891875 ended with `h200` empty. 28891874 took r67 seed 2 at 05:22Z and runs it to the end (INBOX: let the H200
  workers finish). The free slot goes to 29053189 (mit_normal_gpu B200, PER_GPU=3, `screens_d_b200`: r68 seeds 0-2).

## 2026-11-14 03:04 EST: r63 complete on five B200 seeds: kept mean 0.441, +0.014 over r60; meets the 5-seed form, not yet gated

| seed | r63 kept (96 prompts) | kept step | r60 kept (64 prompts) | diff |
|---|---|---|---|---|
| 0 | 0.4583 | 350 | 0.4283 | +0.0300 |
| 1 | 0.4383 | 400 | 0.4617 | -0.0233 |
| 2 | 0.4350 | 450 | 0.3800 | +0.0550 |
| 3 | 0.4300 | 350 | 0.4350 | -0.0050 |
| 4 | 0.4450 | 400 | 0.4317 | +0.0133 |

- Mean 0.4413 vs 0.4273, diff +0.014 (3 of 5 seeds up). Every r63 seed clears A 006's dev mean 0.388 and the mean
  clears A's line 0.418, so r63 meets the 5-seed form on B200. It is **not yet gated**, and its H200 gate stays first
  in the STATUS backlog (INBOX 11-13 23:45). Most of the gain over A 006 comes from the longer schedule (r60 is 0.427
  too); 96 prompts per step adds a little and narrows the spread (0.430-0.458 against 0.380-0.462).
- 29037156 and 29037159 exited normally. 29060211 and 29060212 (preemptable B200, PER_GPU=3) run r68 s0-2 and r70
  s0-2. Two new mit_normal_gpu B200 workers (29067858, 29067860) take r69 s0-1, and r69 s2 waits in `screens_b200`.
  The mit_normal_gpu worker submitted at 01:04 (29053189) sat on QOSMaxCpuPerUserLimit while two B200 jobs ran there,
  so it was cancelled and replaced on preemptable.

## 2026-11-14 05:04 EST: r67 (unranked hindsight screen, D12) done: above D 005, below self-imitation; full wake, B200 only

| seed | r67 (hindsight) | D 005 | r61 (self-imitation) | vs D 005 | vs r61 |
|---|---|---|---|---|---|
| 0 | 0.2183 | 0.1933 | 0.2500 | +0.0250 | -0.0317 |
| 1 | 0.2217 | 0.1867 | 0.2467 | +0.0350 | -0.0250 |
| 2 | 0.2133 | 0.2100 | 0.2367 | +0.0033 | -0.0233 |

- Hindsight relabelling (64 rows, coef 0.5, timed H200 from D 005's checkpoints) gains +0.021 over D 005 on average
  but trails self-imitation by -0.027 on every seed. As INBOX 11-10 14:54 says, r67 is unranked and is never promoted
  or stacked. D12 is closed.
- All H200 workers have now finished their items and exited; none was cancelled. Engaging work is B200 only from here.
- **Preemptable B200 churn:** 29060211 and 29060212 (r68 and r70, packed 3 each) were preempted twice between 08:00Z
  and 08:45Z, once by this agent's own mit_normal_gpu job landing on node5701-c1. They are pending again with their six
  items in hand. Screens are no_save, so each preemption costs the minutes run so far. mit_normal_gpu takes only two
  B200 jobs per user (the third pends on QOSMaxCpuPerUserLimit), so the other four slots have to be preemptable.
- New: r71 (commit above) is r63 with 128 prompts per step, 350 of 437 cosine steps, paired with r63. Its seeds 0-2
  go into `screens_b200` after r69 s2.
- Jobs, 6 GPUs: 29067858 and 29067860 (mit_normal_gpu B200: r69 s0-1); pending preemptable B200: 29060211 and
  29060212 (r68 and r70), 29073366 (r69 s2) and 29076377 (r71).

## 2026-11-14 11:00 EST: r69 (track A self-imitation) is worse than r63 on every seed, closed; r72 (r63 + rollback) queued

| seed | r69 kept | step | r63 kept | step | diff |
|---|---|---|---|---|---|
| 0 | 0.3800 | 467 | 0.4583 | 350 | -0.0783 |
| 1 | 0.3583 | 450 | 0.4383 | 400 | -0.0800 |
| 2 | 0.3433 | 467 | 0.4350 | 450 | -0.0917 |

- r69 = r63 + self-imitation (64 rows, coef 0.5, buffer 4096, one pool per level), B200, step-capped. Mean kept
  0.361 against r63's 0.444 (-0.083), every seed about 0.08 down, and the best eval comes at or near the last step,
  so the run is slower, not sliding. On track A, replaying the model's own correct answers costs more than it gives;
  the D11 gain does not carry over to the pretrained model. Closed; no further track A self-imitation arms.
- New: r72 = r63 + `rollback_drop` 0.05 (r64 gave +0.013 kept on r60's schedule and removed the slides). Seeds 0-2
  queued in `screens_b200` behind r71, paired with r63. If it holds up, it is the stack the backlog's A gate would use.
- Track D on B200 packed 3 takes about 7.1-7.4 h a run (r56/r57, 11-13), too long for mit_normal_gpu's 6 h, so it
  stays on mit_preemptable, where 29060211 and 29060212 were preempted again at about 09:00 and 10:50 EST. r68 and
  r70 have not completed a run yet.
- Jobs, 6 GPUs, all B200: mit_normal_gpu 29085753 (r71 s0) and 29094497 (new, takes the next `screens_b200` item);
  preemptable 29076377 (r71 s2) and 29073366 (r71 s1) running, 29060211 and 29060212 (r68 and r70) pending.

## 2026-11-14 16:10 EST: r72 s0 kept 0.422 (-0.037 vs r63); preempted work moved to mit_normal_gpu

- r72 s0 (r63 + `rollback_drop` 0.05, B200): kept 0.4217 at step 350 against r63 s0's 0.4583 (-0.037). Its dev fell
  to 0.388 at step 400 and 0.308 at step 450, which triggered the rollback to step 350; step 467 read 0.398. Its
  malformed share sat near 0.54 from step 150 on, against about 0.20 for r63 s0. r63 s1 was also near 0.42, so seeds
  differ that much; one seed is not a reading. Seeds 1-2 are running. Not yet gated.
- **Preemptions:** at about 19:00Z all four preemptable B200 workers were preempted again. 29076377 lost r72 s1 at
  4.8 h; 29060211 and 29060212 (r68, r70) and 29073366 (r71 s1) restarted and were preempted within the hour. No
  preemptable track D run has finished today.
- **Change:** track D alone on a B200 took 3.95 h (r30 s2, 11-01), which fits a 6-h `mit_normal_gpu` job (step caps
  make the trajectory the same at any packing). I cancelled the pending preemptable workers 29076377, 29060212 and
  29073366, and returned their items to `screens_b200`: r72 s1 (now running on 29110624), then r71 s1, r68 s0, r68 s2
  and r70 s1. Three more `mit_normal_gpu` B200 workers (29115015-17, PER_GPU=1) wait behind the two running ones;
  mit_normal_gpu runs two per user at a time, so they start as those finish. 29060211 (r68 s1, r70 s0, r70 s2,
  packed 3) stays on preemptable.
- Jobs, 6 GPUs, all B200: running 29102770 (r72 s2) and 29110624 (r72 s1); pending 29115015-17 (`mit_normal_gpu`)
  and 29060211 (preemptable).

## 2026-11-14 18:15 EST: r72 (A, r63 + rollback_drop 0.05) -0.021 vs r63, closed; preemptable slots back to H200

r72 is r63 (A, 96 prompts per step, B200, step-capped) plus `rollback_drop` 0.05. All three seeds finished on
`mit_normal_gpu` B200 (not yet gated). Kept dev against r63's paired seeds:

| seed | r72 | r63 | diff |
|---|---|---|---|
| 0 | 0.4217 | 0.4583 | -0.0366 |
| 1 | 0.4267 | 0.4383 | -0.0116 |
| 2 | 0.4217 | 0.4350 | -0.0133 |
| mean | 0.4234 | 0.4439 | -0.0205 |

All three seeds are below their pair. The rollback fired only on s0 (dev 0.308 at step 450, back to step 350);
s1 and s2 kept step 450 and still lost 0.012-0.013, so most of the gap looks like seed-to-seed noise on top of no
gain. Rolling back on a dev drop doesn't help r63. Closed; r63 stays the A candidate in the gate backlog.

**Correction to LOG 11-14 05:04 and 16:10.** Those entries said a preemption loses the whole run. It doesn't:
`screen_seed.sh` saves resumable state every `RESUME_EVERY_S` training seconds on mit_preemptable (default 3600), and
a requeued seed continues from it (INBOX 11-14 17:04). r72 s1 resumed at step 156 on 29110624, and r68 s0 resumed at
step 1403 on 29115016. What the preemptable B200 churn cost was the 16 of 18 incarnations that died inside their
first hour, before any save.

**Preemptable slots (INBOX 11-14 17:01).** Preemptable B200 lost 77 % of its GPU-hours to preemption against 38 % for
preemptable H200, so the four preemptable slots go back to H200. r68 and r70 are step-capped at 15,000 steps, so they
follow the same trajectory on any GPU type and packing, and they stay comparable with r59 (B200).
- Cancelled 29060211 (preemptable B200, pending, no saved state for its items) and moved r68 s1, r70 s0 and r70 s2
  to a new queue, `screens_h200`.
- Submitted 29124241 and 29124243: preemptable H200, `packed_worker`, PER_GPU=2, `RESUME_EVERY_S=1800` (D's state is
  1.3 GB, so a half-hourly save is cheap and halves what a preemption costs).
- Jobs, 6 GPUs: `mit_normal_gpu` B200 running 29115015 (r71 s1) and 29115016 (r68 s0), pending 29115017 and 29119664
  (`screens_b200`: r68 s2, r70 s1); preemptable H200 pending 29124241 and 29124243 (`screens_h200`). One H200 slot is
  spare once the three D items are placed; it takes the next screens I queue.

## 2026-11-14 19:10 EST: node5200 hang; H200 worker resubmitted

29124241 (preemptable H200) sat 43 min on node5200 with both `queue_worker` processes in uninterruptible I/O wait
(state D, no children, GPU idle) before popping an item. Cancelled; 29128857 replaces it with node5200 added to the
H200 exclusions. r70 s2 is still first in `screens_h200`. 29124243 runs r68 s1 and r70 s0 on node5001.

## 2026-11-14 21:20 EST: r71 (A, 128 prompts per step) -0.004 vs r63, closed; r73 (D8 mid levels x2) queued

r71 is r63 with 128 prompts per step instead of 96, the prompts seen kept equal (cosine over 437 steps, stopped at
350). All three seeds on `mit_normal_gpu` B200, step-capped (not yet gated). Kept dev against r63's paired seeds:

| seed | r71 | r63 | diff |
|---|---|---|---|
| 0 | 0.4533 | 0.4583 | -0.0050 |
| 1 | 0.4183 | 0.4383 | -0.0200 |
| 2 | 0.4483 | 0.4350 | +0.0133 |
| mean | 0.4400 | 0.4439 | -0.0039 |

Null: within seed noise either way, and every seed kept step 350, its last eval but one. Wider steps don't buy
anything over r63 at the same prompts seen, and they cost time per step. Closed. With r69 (self-imitation, -0.083)
and r72 (rollback, -0.021) also closed, r63 stays the A candidate in the gate backlog; its five seeds sit at
0.42-0.44 from step 350 on, with the kept step at 350-450.

**Next D arm, r73 (PLAN D8 level weights).** r59's config (D 006's recipe) with `train_ladder_levels`
L1,L2,L2,L3,L3,L4,L4,L5,L6, so L2-L4 are drawn 2/9 each instead of 1/6 and L1, L5, L6 1/9 each. r59 ends at 0.06-0.22
dev on L2-L4, where groups are most often mixed; r45 (no L1, -0.020) and r57 (no L5, null) said to keep the end levels
in, so this reweights instead of dropping. `ladder_stream` draws uniformly over the names, so a repeated name is
drawn twice as often. Step-capped at 15,000 steps; read paired with r59 (0.278 / 0.248 / 0.232). Seeds 0-2 are
appended to `screens_b200` after r70 s1, for the `mit_normal_gpu` B200 slots that free up overnight.

- 29124243 (preemptable H200, r68 s1 and r70 s0) was preempted after about 1.5 h, past its first 30-min save, and
  is pending again; 29128857 (r70 s2) is still pending. B200: 29115016 (r68 s0) and 29115017 (r68 s2) running,
  29119664 (r70 s1) pending.

## 2026-11-14 23:35 EST: second venv hang (node5100), guard added; GPU use opened up per INBOX 22:03; r72 correction

**INBOX 22:03** makes GPU use mine to optimise and explore, counts only GPUs that can run at once (2 on
`mit_normal_gpu` plus 4 on `mit_preemptable`), and leaves a backlog gate on an H200 slot to my judgment.

**Hang, again.** 29128857 (preemptable H200, node5100) ran 59 min with an empty `.out` and no item: both
queue_worker processes in D state, as on node5200 at 19:10. A probe on the node, each path under `timeout`:
`~/orcd/pool` and the queue lock stat fine, but `~/orcd/pool/vllm-venv/bin/activate` does not answer in 8 s. So it is
the node's Lustre client on the venv, not the worker. Cancelled (it held nothing; r70 s2 is still queued), node5100
added to the H200 excludes, and `packed_worker.sbatch` now stats the venv under `timeout 60` before starting any
worker, logging the host and exiting if it hangs, so a bad node costs a minute instead of hours. (`timeout` does get
out of this hang: the probe returned.) Two of five preemptable H200 starts today hit it.

**Trying GPU types.** One change at a time, so yield can be read from `sacct -D` per type:
- 29147202: preemptable B200, PER_GPU=2, `RESUME_EVERY_S=600` (track D state is 1.3 GB, so a 10-min save is cheap).
  It takes the cancelled H200 worker's slot and draws `screens_h200` then `screens_b200`. All queued items are track D.
- 29147203 / 29147204: `mit_normal_gpu` B200 workers queued behind the running pair (QUEUE=screens_b200:screens_h200),
  so those two slots refill as soon as a 6-h worker exits instead of waiting for a wake.
- Not yet: A100 (the gres does not say 40 or 80 GB) and RTX PRO 6000 (untested here). Both wait until a slot is
  short of work.
- 29137875 (submitted 21:04): a third preemptable H200 worker on `screens_h200:screens_b200`.

**r72 correction (18:15 entry).** Rollback fired only once in r72: seed 0, back to step 350. Seeds 1 and 2 never
rolled back, so they ran r63's trajectory in effect, and still kept 0.4267 / 0.4217 against r63's 0.4383 / 0.4350.
So -0.012 / -0.013 is the run-to-run difference between identical configs on the same seed (B200 vs B200), and
r72's -0.021 says little about rollback itself; only seed 0 tested it (-0.037). The closing stands: on r63's
schedule rollback seldom fires, so it can add little. The noise figure matters more: same-seed pairs carry about
0.01-0.015 of noise on track A, so a 3-seed pair needs a gain well above that.

**Results in.** r68 s0 (track D, sil coef 1.0): final dev 0.240 against r59 s0's 0.278 (-0.038). s1 and s2 pending.

## 2026-11-15 01:05 EST: worker resubmit, r68 s2

- 29137875 (preemptable H200, submitted before the node5100 exclude) also landed on node5100 and hung on the venv. Cancelled and resubmitted as 29151450 (PER_GPU=2, node5100 and node5200 excluded). Slurm copies the batch script at submit, so the 30cd3ae guard covers only workers submitted after it: 29147202/3/4 and 29119664/29124243 run the old copy.
- r68 (sil coef 1.0) s2: 0.2383 vs r59 s2 0.232. With s0 (0.240 vs 0.278), r68 stands at -0.016 over two seeds; s1 is running.
- Queues: screens_h200 r70 s2; screens_b200 r73 s1, s2. Next track D arm: maintainer idea U14 (PLAN D9, an LM loss on the pretraining mix during RL), being coded.

## 2026-11-15 01:40 EST: U14 coded (lm_rows), r74 queued

- 2ec078b adds `lm_rows`/`lm_coef`/`lm_seq_len`/`lm_tokens` to grpo.py (maintainer idea U14, PLAN D9), off by default: each step adds lm_coef x the mean next-token cross-entropy of lm_rows windows of the pretraining token file, drawn at offsets from (seed, step). One process only; a token id outside the policy's vocabulary stops the run. Three tests; ruff and pytest (159) pass.
- r74 = r59 + 8 windows of 1024 tokens of D 004's own mix_0.2.bin at coef 0.25 (8k LM tokens per step; r59's RL grad norm is about 0.15, so the LM term should sit below it). Seeds 0-2 queued on screens_b200 after r73, paired with r59 (0.278 / 0.248 / 0.232).


## 2026-11-15 04:10 EST: r68 closed (null); r70 at -0.013; preemptable B200 trial; r63's H200 gate (r75) queued

- r68 (r59 + sil coef 1.0) closed at 3 seeds: 0.240 / 0.2633 / 0.2383 (mean 0.2472) against r59's 0.2783 / 0.2483 /
  0.2317 (mean 0.2528), -0.006 paired (-0.038 / +0.015 / +0.007). Null; coef 0.5 stays.
- r70 (128 sil rows) s0-s1: 0.2667 / 0.2333, -0.012 / -0.015 against r59 (mean -0.013). s2 runs on 29151450.
- Preemptable B200 trial (INBOX 11-14 22:03): 29147202 started 03:51:37 EST on node5700-c1 and was preempted at
  03:53:11, after 94 s. It is requeued and holds r74 s0. One data point; the B200s in mit_preemptable look heavily
  claimed by their owners. 29119664 and 29124243 ended COMPLETED (queues empty).
- r75 = r63's H200 gate: r63's config unchanged (kept dev 0.441 over 5 B200 seeds, against A 006's line 0.418), run
  alone on an NVIDIA H200, read seed by seed against 0.418 as r62 was. INBOX 11-14 22:03 leaves backlog gates to my
  judgement; this is the strongest track A candidate. Seeds 0-2 are in queue `gate_a_h200`; worker 29173680
  (mit_preemptable H200, PER_GPU=1 so the timed run has the GPU to itself, RESUME_EVERY_S=1800; the resumed run keeps
  its clock) is pending. GPUs: 3 running, 2 pending.

## 2026-11-15 06:40 EST: r73 s0; B200 worker replaced; track D's H200 runs get 10,900 RL steps, not 15,000 (PLAN D7)

- r73 (D8, mid levels x2) s0: 0.255 against r59's 0.2783 (-0.023). s1 runs on 29151450, s2 on 29147204.
- 29147203 (`mit_normal_gpu` B200) ended COMPLETED after r73 s0, with under 250 min left. Replacement 29183441, same
  settings (PER_GPU=1, QUEUE=screens_b200:screens_h200, RESERVE_MIN=250); r74 s2 is the only item left in either queue.
- Correction: the 11-14 01:04 entry calls r65 "three minibatches". r65's `mini3` is the mini levels M2, M3, M6
  (`train_ladder_levels` L1-L6,M2,M3,M6), as in r56. The verdict stands: null on self-imitation from D 005.
- **The step gap.** Record D 006's five H200 seeds ran 10,912 / 11,380 / 11,237 / 10,068 / 10,621 RL steps in the
  14,040-s budget (1.29 s per step on seed 0: t_sample 0.76-0.82 s, t_update 0.33-0.37 s, about 0.14 s else). r33
  (no self-imitation) made 13,728. The B200 screens stop at 15,000 steps, so they read the curve further than an H200
  record gets. r59's five-seed mean dev by step: 0.163 / 0.192 / 0.219 / 0.239 / 0.246 / 0.258 at 2,500 / 5,000 /
  7,500 / 10,000 / 12,500 / 15,000. D 006's dev mean was 0.229, about where r59 is at 10,000. Reaching 15,000 steps in
  the H200 budget is worth about +0.02 by that curve, near the D line 0.259 on its own.
- So PLAN D7 (more steps per budget) moves ahead of further self-imitation and level-weight arms. Sampling is the
  biggest part: about 512 completions of about 25-30 tokens in 0.8 s, so the decode loop's per-token CPU cost dominates
  (the custom answer-constraint logits processor runs on every token; vLLM 0.30). Next: a short profiling job on one
  GPU (not the login node) that times one D 006 step's sampling with the constraint on and off and with vLLM's async
  scheduling, then the cheapest fix that gives 1.4x or more, off by default and screened against r59.

## 2026-11-15 06:15 EST: PLAN D7 profiling (r76) submitted

- `vllm_engine_args` (e979806): extra `vllm.LLM` keyword arguments as a JSON object, off by default; it needs
  `rollout_backend` "vllm".
- r76 is four 300-step runs of r59's config, for timing only (not screens): unchanged, with vLLM's
  `async_scheduling`, with `constrain_answer` off, and with both. They run one after another, alone on a preemptable
  H200 (worker 29183714, queue `profile_d7`). They will be read on the median `t_sample` and `t_update` per step.
  The fix needs to reach about 15,000 RL steps in D's 14,040 s, so about 1.4x the steps.

## 2026-11-15 07:18 EST: r76 profile read; the per-request answer mask is the cost; `constrain_batched` coded

Median seconds per step over steps 50-300, one run each, alone on an H200 (node4800):

| r76 variant | t_sample | t_update | wall for 300 steps |
|---|---|---|---|
| r59 unchanged | 0.748 | 0.649 | 503 s |
| vLLM `async_scheduling` | 0.822 | 0.719 | 551 s |
| `constrain_answer` off | 0.234 | 0.041 | 159 s |
| both | 0.236 | 0.041 | 160 s |

- Async scheduling does not help; drop it.
- Without the constraint the policy collapses at once (entropy 0, no solved puzzles, 0-3 informative groups). So
  that run's `t_update` and its shorter completions are not comparable. But its `t_sample` bounds what the mask
  costs: about 0.5 s of the 0.75 s.
- The cause: vLLM's `AdapterLogitsProcessor.apply` calls each request's processor on its own row. For 512
  requests, that is about 1,500 small kernels per decoded token.
- `constrain_batched` (989199f, off by default) gathers every request's allowed set (cached on the GPU by tuple) and
  masks the batch in a fixed handful of kernels. A test checks the mask equals the per-request one.
- Being timed now: r76 `d7prof_batched` and an unchanged `d7prof_base2`, on one H200 (worker 29188304, same node
  for both).
- Also: this run's `t_update` (0.65) is about twice D 006 seed 0's (0.33-0.37). Read it again on base2 before
  blaming the node.

## 2026-11-15 08:01 EST: `constrain_batched` saves 0.25 s a step; trie-built allowed sets; r76 timing round 3

Same H200 (node3200), steps 50-300, medians:

| r76 variant | t_sample | t_update | wall for 300 steps |
|---|---|---|---|
| `d7prof_base2` (r59 unchanged) | 0.842 | 0.712 | 551 s |
| `d7prof_batched` | 0.593 | 0.686 | 467 s |

- The batched run's entropy (0.26 vs 0.27) and reward (0.40 vs 0.39) match the unchanged run.
- The high `t_update` is early-run cost, not the node. D 006 seed 0 has the same over steps 50-300 (t_sample 0.88,
  t_update 0.70), falling to 0.76 / 0.37 by step 8000.
- So `constrain_batched` alone saves about 0.25 s of a late step's 1.29 s, about 1.24x the steps. That is short
  of the 1.4x target.
- A CPU microbenchmark (TinyLlama tokenizer, random 3-5 number puzzles, 512 rows) puts the rest in Python:
  building the allowed-token set for each new grammar state tested all 477 candidate tokens, at 191 us per state.
- c81d9e2 walks a character trie of the candidates instead. It gives the same ids at 17 us per state; a test
  compares it with testing every candidate. The output is unchanged, so this is not an option. It also speeds
  up the trainer's `allowed_per_position`.
- Timing r76 `d7prof_batched_trie` and `d7prof_base_trie` next (worker 29191186).

## 2026-11-15 08:45 EST: r76 round 3, the trie does not cut sampling; r75 s0 passes 0.418; r77 = D 006 + constrain_batched, H200 gate

- r76 round 3 on node4500, medians over steps 50-300, alone on an H200: `d7prof_base_trie` t_sample 0.98 s,
  t_update 0.62 s, wall 579 s; `d7prof_batched_trie` 0.614 / 0.602, wall 467 s. The trie gives no sampling gain:
  states repeat within each group of 8, so the per-state cache already hides the cost of building allowed sets.
  `constrain_batched` saves 0.25-0.37 s per step against the unchanged mask on the same node (node3200: 0.842 ->
  0.593; node4500: 0.98 -> 0.614), with matching entropy and reward. Late in a D run (about 1.29 s per step) that
  is about 1.2-1.3x the steps: an estimated 13.5k-15k RL steps in 14,040 s against D 006's 10.9k. r59's five-seed
  dev was 0.239 at step 10,000 and 0.258 at 15,000.
- r75 (r63's H200 gate) s0: dev 0.450 at step 467 against A 006's line 0.418. s1 runs on 29173680.
- r77 = D 006's record config unchanged (its own pretraining, the same time budgets) plus `constrain_batched: true`
  and `no_save`, run alone on an H200, read seed by seed against D's line 0.259 on dev. D 006's own seeds kept dev
  0.220 / 0.238 / 0.243 / 0.220 / 0.225. Seeds 0-2 are in queue `gate_d_h200`; worker 29194209 (mit_preemptable
  H200, PER_GPU=1, RESUME_EVERY_S=1800) is pending. GPUs: 5 running, 1 pending.

## 2026-11-15 09:55 EST: r70 and r73 close flat; r73 s2 cut at step 14,500 by the B200 time limit; r77 worker resubmitted; r78 = r59 at 20k steps

- r70 (128 sil rows) s2: 0.2617 against r59 s2's 0.2317 (+0.030). Over three seeds -0.012 / -0.015 / +0.030,
  mean +0.001: no gain. Closed.
- r73 (D8, mid levels x2) s1: 0.270 against 0.2483 (+0.022). s2: 29147204 (`mit_normal_gpu` B200, no saved state)
  hit its 6-h limit at step 14,500 of 15,000, 21,464 s in. Read at step 14,500 against r59 s2 at the same step:
  0.250 against 0.235 (+0.015). Over three seeds -0.023 / +0.022 / +0.015, mean +0.005, within the
  seed noise. Closed without a rerun.
- Alone on a B200, a D screen ran about 1.48 s per step, so 15k steps barely fit the 6-h `mit_normal_gpu` limit.
  New D screens set `constrain_batched` (the same mask, so step-capped results don't change) to fit.
- 29194209 (r77's worker) exited after 3 s: I left out `--time`, so the job got the partition's short default
  ("less than 90 min left"). Resubmitted as 29198329 with the documented flags (`--time=2-00:00:00`, VENV,
  RLVR_DIR, HF_HUB_OFFLINE).
- r75 s1 was preempted; 29173680 is requeued and resumes from its half-hourly state.
- r78 = r59 with `budget_steps` 20000 (the lr schedule reads step s as s/20000 of the time budget) and
  `constrain_batched`: what a D run gains from 20k RL steps instead of 15k, which sizes the value of more
  speed work (PLAN D7). Three seeds in queue `screens_r78`, worker 29198428 (mit_preemptable H200, PER_GPU=3,
  RESUME_EVERY_S=1800). GPUs: 2 running (29183441 r74 s2, 29147202 r74 s0/s1), 3 pending; one `mit_normal_gpu`
  slot is free for the next screen.

## 2026-11-15 11:05 EST: r74 s0-s1; second r75 gate worker; U10 coded (`overlong_buffer`, `overlong_coef`) and screened as r79 on B200

- r74 (maintainer idea U14: r59 plus an LM loss on the pretraining mix) s0: 0.2733 (-0.005 against r59), s1: 0.2300
  (-0.018). s2 runs on 29183441. 29147202 ended COMPLETED after s0.
- r75: s1 resumed on 29173680. A second `gate_a_h200` worker, 29202527 (same flags), takes s2 so the gate doesn't
  wait about 5 h for it.
- U10 (maintainer idea: a longer training cap plus a length penalty) is coded: `overlong_buffer` and
  `overlong_coef`, both off by default. They add DAPO's soft overlong penalty to the training reward only: a completion of
  L tokens loses overlong_coef * clamp((L - (max_new_tokens - overlong_buffer)) / overlong_buffer, 0, 1). A test
  checks the ramp. ruff and the 163 tests pass.
- r79 = r63 (A's strongest config, kept dev 0.441 over 5 B200 seeds) with a training cap of 1,536 tokens,
  in-loop dev still at 1,024 (`eval_max_new_tokens`, the test ladder's limit), and the penalty over tokens
  1,024-1,536 (coef 0.5). So a correct answer past 1,024 tokens, which the ladder would score as a failure,
  earns 1.0 down to 0.5, still above a truncated one (0). r40's seeds lost 41 % of late samples to the cap.
  It is step-capped like r63, so I'll compare s/step with r63's on B200 (12,533 s for seed 0) to see what the
  longer samples cost. Three seeds are in `screens_b200`, with workers 29202659 and 29202660 (`mit_normal_gpu`
  B200, PER_GPU=1, RESERVE_MIN=250). This is earlier than the 11-18 box planned in STATUS, so the box can
  take other work.
- GPUs: 29183441, 29173680, 29198329 and 29198428 are running; 29202527, 29202659 and 29202660 are pending.


## 2026-11-15 11:50 EST: r74 closes flat; r79 collapses (my design error), cancelled; `overlong_correct_only`; r80 queued

- r74 s2: 0.235 against r59 s2's 0.2317 at the same step (+0.003). Over three seeds the mean is -0.007 against r59.
  Maintainer idea U14 closes as flat.
- r79 s0 collapsed: dev 0.2017 against r63's 0.441, and it finished its 467 steps in 2,001 s against r63's 12,533 s.
  Compared with r63 s0 at the same steps (step 161): malformed 0.0 vs 0.53, t_sample 0.83 s vs 15.5 s, and mean reward
  0.08 vs 0.31. It learned to give short guesses. The cause is my reward design, not a length bug. A truncated sample
  sits at the 1,536 cap, so it took the full -0.5 penalty on top of malformed_reward 0.0. That put it below any quick
  wrong guess (0), so short outputs were always the safer choice (step 1's mean reward was -0.11). The 11:05 entry's
  "still above a truncated one (0)" was wrong. DAPO applies the penalty to every sample, but their truncated samples
  are rare. Here r40 lost 41 % of late samples to the cap. I cancelled s1 and s2 (29202659, 29202660, both mine).
- New option `overlong_correct_only` (default off): the penalty applies only to correct completions. A long or
  truncated attempt then scores the same as a short wrong one (0), and a correct answer past 1,024 tokens gets 1.0
  down to 0.5. A test covers the correct, wrong and truncated cases at the cap. ruff and the 164 tests pass.
- r80 = r79 + `overlong_correct_only`. Three seeds are in `screens_b200`, with workers 29208466 and 29208467
  (B200, same flags). time_budget_s 17,400 fits the 6 h limit.
- GPUs: 29198329 (r77), 29198428 (r78) and 29173680 (r75 s1) are running; 29202527 (r75 s2), 29208466 and
  29208467 are pending.

## 2026-11-15 13:50 EST: r75 s1 0.445; r80 not collapsing; r77 s1 worker

- r75 (r63's H200 gate) s1: kept dev 0.445 at step 450 (last 0.423). With s0 at 0.450, both seeds so far clear
  A 006's line of 0.418. s2 runs on 29202527. 29173680 ended COMPLETED.
- r80 (U10 with `overlong_correct_only`) s0 and s1 are at about step 125. Unlike r79, they track r63's trajectory
  (reward 0.2-0.25, malformed 0.5, entropy 0.18). Samples take about 25 s against r63's 15 s, because the training cap
  is now 1,536 tokens.
- A third preemptable worker, 29218868 (`gate_d_h200`), takes r77 s1 in the slot 29173680 freed.
- GPUs: 29198329 (r77 s0), 29198428 (r78), 29202527 (r75 s2), 29208466 and 29208467 (r80 s0-s1) are running;
  29218868 is pending.

## 2026-11-15 15:05 EST: INBOX 14:03 and 14:09 (U16-U18 judged; RULES v3.5); r81 (U16, Muon) configured

- INBOX 14:03: three new maintainer ideas, U16 to U18. Each has a row in the private status file. The `rlvr-chat` jobs
  on `mit_quicktest` are the maintainer's.
- INBOX 14:09: RULES v3.5 lets a record train and score outside vLLM, through `ladder_confirm` with its own generation
  backend. Nothing here needs that yet, so `ladder_confirm` is unchanged.
- r81 (maintainer idea U16) = r77 (D 006 + `constrain_batched`, its own pretraining) with Muon pretraining at lr 4e-3
  for 56 min instead of AdamW for 112 min. The 3,360 s saved go to RL (time_budget_s 17,400), so the total stays
  20,760 s. Why: in sweep_d11 Muon at 60 min reached val loss 2.83, against AdamW's 2.96 at 60 min and 2.81 at
  120 min. RL steps are now worth more than they were then (r59: dev 0.239 at 10k steps, 0.258 at 15k), and 56 more
  min is about 2.6k more steps at 1.3 s per step. Muon hasn't been run on the maths mix yet, so its pretraining val loss
  is the first thing to read. The pretraining LR follows a cosine over wall time, so 56 min gets a full decay.
  Three seeds are in `screens_r81` (H200, read against r77's seeds). No worker yet: all four preemptable slots are
  busy, and if r75 s2 passes, the A 007 record seeds go first.

## 2026-11-15 15:25 EST: INBOX 15:13/15:16 (maintainer idea U19 judged); r77 s1 preempted

- Read INBOX through 4b9e48c. Maintainer idea U19 is judged in the private STATUS file: run now, in steps. First
  pretraining-only screens read by val loss at fixed minutes, data unchanged, then the best end to end like r81.
- U19 baseline from r77 seed 0 (H200, AdamW, lr 1.5e-3, mix_0.2): val 3.079 at 51 min, 3.013 at 62 min,
  2.892 at 112 min; 370k tok/s at the end. r81 (Muon, 56 min) is read against these.
- r77 seed 1 (29218868) was preempted at 15:11, at pretraining minute 58. Pretraining has no resume, so it restarts. It is
  pending on QOSMaxMemoryPerUser, the per-user memory cap shared with the maintainer's CPU arrays.
- Running: r80 seeds 0-1 (B200), r75 seed 2, r77 seed 0, r78 (H200 preemptable).

## 2026-11-15 16:05 EST: r75 passes 3/3; A 007 record queued; r77 seed 0 under the line

- r75 (r63's config on an H200, A 006's line 0.418), `summarize_screens`:

      a006_p96_cos583_s467_h200  s0 final 0.450  kept step 467, last 0.450
      a006_p96_cos583_s467_h200  s1 final 0.445  kept step 450, last 0.423
      a006_p96_cos583_s467_h200  s2 final 0.438  kept step 467, last 0.438
      n=3 final 0.444 ± 0.006, none collapsed, 232 min each

  Every seed clears 0.418, which is PLAN's 3-seed promotion form. In plain words: 96 prompts per step with a longer
  cosine adds about 0.056 dev over A 006's kept dev mean of 0.388, and the gain from B200 (r63, 0.441) holds on the H200.
- Record config `records/v3/track_a/007_fp32_mal0_p96_cos583_s467_ent05_keepbest` (f589beb) is r75's config minus
  `no_save`. Nothing else changed (checked key by key against A 006). Its five seeds are in
  `queue/record-007_fp32_mal0_p96_cos583_s467_ent05_keepbest.txt`, run by two preemptable H200 packed workers,
  29229546 and 29229547 (PER_GPU=1, RESERVE_MIN 380, no resume saves, so the record's clock carries none). They are
  pending, with "Reserved for maintenance" as the reason; `--time-min` lets them start before the 11-17 window. A
  preempted seed goes to `extra/` and reruns, as the worker does.
- r77 seed 0 (D 006 + constrain_batched on an H200) kept dev 0.253 against D's line 0.259. That is +0.024 over D 006's
  dev mean of 0.229, but the 3-seed form already fails. The 5-seed form would need the mean of seeds 1-2 plus two more
  seeds to reach 0.259. Seeds 1 and 2 run on. Clock: pretraining 6,878 s plus GRPO 14,259 s, 21,110 s against 21,600.
  `done.txt` says FAILED only because `run_seed_v3.sh` (STAGE=train) moves a `final/` that a `no_save` screen
  never writes. The log and result are complete. I left the record script unchanged while a record runs through it.
- r80 (U10, B200) seeds 0-1 at step 350: dev 0.412 / 0.408. They are still running, to be read against r63's kept 0.441.

## 2026-11-15 17:05 EST: r80 closes (overlong penalty on correct answers, -0.031); r82 pretraining-only screens on B200; A 007 seed 0 started

- r80 (maintainer idea U10: r63 + max_new_tokens 1536 + an overlong penalty on correct completions only, B200), `summarize_screens`:

      r80 s0 final 0.412  kept step 350, last 0.382   | r63 s0 0.458
      r80 s1 final 0.408  kept step 400, last 0.385   | r63 s1 0.438
      n=2 0.410 ± 0.002 against r63's 0.441 ± 0.011 (n=5)

  Paired by seed: -0.046 and -0.030. In plain words: with `overlong_correct_only` the r79 collapse is gone, but the
  longer cap still costs dev, and neither seed comes near r63. Closed; seed 2 is taken out of `screens_b200`.
- A 007 seed 0 started at 15:58 on 29229546 (node4602, H200). 29229547 waits for a slot.
- r82, a pretraining-only batch (maintainer ideas U19 and U16; `pretrain_only`, no RL). Track D's 110M model runs 56 min on
  mix_0.2 with seed 0, alone on a B200, and is read by final val loss against the AdamW control in the same batch. Arms:
  AdamW lr 1.5e-3 (the control), Muon at lr 2e-3, 4e-3 and 8e-3, and Muon 4e-3 with half the batch (262,144 tokens).
  Two `mit_normal_gpu` B200 workers, 29234617 and 29234618, run with RESERVE_MIN 75: an item is about 65 min, and the
  old 250 made r80's workers exit with 4 h left. r81 (Muon 56 min end to end, H200) stays queued for a preemptable slot.

## 2026-11-15 19:55 EST: r82 closes (Muon lr 4e-3 best, -0.115 val loss against AdamW at 56 min); r78 (20k RL steps) +0.032 over r59; r83 queued

- r82, final val loss after 56 min alone on a B200 (seed 0, same data order; lower is better):

      AdamW lr 1.5e-3 (control)    2.919
      Muon lr 2e-3                 2.836   -0.083
      Muon lr 4e-3                 2.805   -0.115
      Muon lr 8e-3                 2.818   -0.102
      Muon lr 4e-3, batch 262k     2.818   -0.102

  Muon wins at every lr tried, and lr 4e-3 with the full batch is best, which is the setting r81 already uses, so r81
  stays as queued. All arms ran about 600k tok/s. One seed per arm; the spread between Muon settings (0.013-0.031) is
  small next to the gap to AdamW.
- r83 (maintainer ideas U19/U16): r82's Muon 4e-3 and AdamW arms for 112 min instead of 56, on B200. It asks whether
  the gain holds at twice the tokens, and how Muon at 56 min (2.805) compares with AdamW at 112 min on the same GPU.
  That decides whether r81's split (56 min pretraining, the rest to RL) is the right use of the saved time. Queue
  `screens_b200`, workers 29250819 and 29250820 (RESERVE_MIN 130).
- r78 = r59 with `budget_steps` 20000 and `constrain_batched` (H200 packed 3 per GPU; 34,542 s of wall time, so this
  is a step-capped screen, not a fit for the budget). Final dev against r59's 15k-step seeds:

      r78 s0 0.295  (0.277 at step 15,000) | r59 s0 0.278
      r78 s1 0.255  (0.250)                | r59 s1 0.248
      r78 s2 0.305  (0.273)                | r59 s2 0.232
      mean 0.285 (0.267 at 15k) against 0.253; paired +0.017 / +0.007 / +0.073, mean +0.032

  Dev still climbs from step 15,000 to 20,000 (+0.018 within r78, and the longer cosine itself adds a little at
  15k). So a D run that fits about a third more RL steps into its budget is worth about +0.02 to +0.03 dev. That is
  the value of the speed work (PLAN D7), and of r81's move of 56 min from pretraining to RL.
- r77 s1's worker 29218868 was preempted again and is pending; it resumes from its half-hourly state. r77 s2 runs on
  29198329. A 007 s0 runs on 29229546; 29229547 is pending.

## 2026-11-15 20:03 EST: 11-18 H100 box planned (r85 track A, r86 track D); r84 Muon checkpoints queued on B200

- r84 (B200, screens_b200 after r83): Muon lr 4e-3 pretraining, saved, seeds 0-4, capped by tokens
  (cosine over tokens, minutes 600 so wall time never binds): muon_1338m (1.338B tokens, H200 56 min,
  ~37 min on B200) and muon_2676m (2.676B tokens = D 004's, ~75 min). They are the starting points
  of the box's D arms and must finish before the 11-17 maintenance.
- r85 (box GPUs 0-3, 12 runs): A 007 (r75) control against lr 1.3e-6, entropy target 0.15 and
  cosine horizon 467; seeds 0-2; time stop off so the 467-step cap binds on H100.
- r86 (box GPUs 4-7, 3 per GPU, 18 runs): D 006 RL from saved checkpoints, step-capped: control on
  D 004 checkpoints at 11441 steps (r77's H200 count) against Muon 1.338B + 14180 steps (r81's split),
  Muon 2.676B + 11441 steps, and (last) Muon 1.338B + 11441 steps; seeds 0-4.
- Plan, queue contents and copy list: docs/research/box_1118.md.

## 2026-11-15 21:03 EST: A 007 seed 0 finished (test 0.456, dev 0.435); seed 1 and r77 seed 1 preempted

- A 007 seed 0 (H200, 29229546): 467 steps, kept step 400, dev 0.435, test ladder 0.456 (A 006: 0.4123).
  The same worker went on to seed 2.
- Seed 1 (29229547) and r77 seed 1 (29218868) were both preempted on node4600 and are pending again.
  The requeued worker redoes the item and moves the partial seed to extra/interrupted_1_<n>, as RULES_v3 asks.

## 2026-11-15 22:03 EST: r83 closes: Muon's lead shrinks at 112 min (-0.034 val loss, from -0.115 at 56 min)

| arm (B200, seed 0, 112 min) | tokens | val loss |
|---|---|---|
| AdamW lr 1.5e-3 (control) | 4.11B | 2.776 |
| Muon lr 4e-3 | 4.04B | 2.742 |

- Muon still wins at twice the time, but by 0.034 instead of 0.115: most of its lead is faster early
  progress. Muon at 56 min (2.805, r82) is still behind AdamW at 112 min (2.776).
- B200 does ~4.1B tokens in 112 min against H200's 2.68B, so these runs are further along than any H200
  pretraining. At the box checkpoints' budgets the lead should be at least r82's at 1.34B tokens (below
  r82's 2.0B) and between the two readings at 2.68B. r86 on the 11-18 box tests it end to end.
- B200 workers moved on to r84 (Muon checkpoints) at 21:52/21:57.

## 2026-11-16 02:02 EST: A 007 seed 2 test 0.453; r77 seed 2 dev 0.273 (over D's line); r84 at 8/10 checkpoints

- A 007 seed 2 (H200, 29229546): 467 steps, kept step 450, dev 0.442, test ladder 0.453 (seed 0: 0.456;
  A 006: 0.4123). The worker went on to seed 3; seed 1 runs on 29229547; seed 4 is queued.
- r77 seed 2 (D 006 + constrain_batched, H200) finished at 21:43 after 11,074 steps: dev 0.273 against D's
  line 0.259 (seed 0: 0.253, 11,441 steps). D 006's own seeds kept 0.220-0.243. Seed 1 is still running.
- r84: muon_1338m seeds 0-4 and muon_2676m seeds 0-2 are saved (2676m val loss 2.792 / 2.774 / 2.781 at
  2.68B tokens). Seeds 3-4 run on B200 workers 29274632/29274633 and end well before the 11-17 maintenance.

## 2026-11-16 03:03 EST: r77 closes under D's line (mean 0.248); A 007 seed 1 test 0.439; r84 done; r81 and r87 start

- r77 (D 006 + constrain_batched, alone on an H200) seed 1 kept dev 0.218. It ran 16,055 RL steps in the same
  14,040 s (seeds 0/2: 11,441 / 11,074), so that node was faster per step, yet it scored lowest.

      r77 s0 0.253 | s1 0.218 | s2 0.273   mean 0.248 against the line 0.259
      D 006 s0-2: 0.220 / 0.238 / 0.243 (mean 0.234); paired +0.033 / -0.020 / +0.030, mean +0.014

  Batched constrained sampling alone does not clear D's line, so no record from it. It stays in the D configs
  (r81, r86) since it is faster per step and does not hurt.
- A 007 seed 1 (H200, 29229547, preempted twice; partials in extra/interrupted_1_1 and _2): 467 steps, kept step
  350, dev 0.427, test ladder 0.439. Seeds 0-2: 0.456 / 0.439 / 0.453 (mean 0.449; A 006 0.4123). Seeds 3 and 4
  run on 29229546 / 29229547 from about 03:00.
- r84 is complete: muon_1338m seeds 0-4 and muon_2676m seeds 0-4 saved in the pool (2676m seeds 3-4: val loss
  2.792 / 2.782). The 11-18 box's checkpoints are all in place before the maintenance.
- r81 (Muon 56 min end to end, H200, seeds 0-2): preemptable workers 29292521 / 29292522, PER_GPU 1.
- r87 (B200, seed 0): r86's arms muon1338m_s14180 and d004_ctl_s11441 unchanged, one per B200 (workers 29292528 /
  29292529, RESERVE_MIN 300), as an early end-to-end look at the Muon checkpoint a day before the box. Not gated;
  read paired against each other only.
- GPUs: 6 (2 A 007 seeds, 2 r81, 2 r87 B200).

## 2026-11-16 06:02 EST: A 007 seed 3 test 0.469; r81 seed 2 gets the free H200 slot

- A 007 seed 3 (H200, 29229546): 467 steps, kept step 467, dev 0.427, test ladder 0.469. Seeds 0-3: 0.456 / 0.439 /
  0.453 / 0.469 (A 006 0.4123). Seed 4 runs on 29229547.
- r81 workers 29292521/22 sat pending behind the 11-17 maintenance reservation (2-day limit); resubmitted at 04:05
  with a 19 h limit as 29297581/29297587 (seeds 0 and 1, running since 04:27). Seed 2 now on a third preemptable
  H200 worker (17.5 h limit). GPUs: 6.

## 2026-11-16 13:53 EST: A 007 is the new track A record (test 0.4527, p = 0.0004); r81 passes D's line, D 007 starts

- A 007 seed 4 (H200, 29229547): 467 steps, kept step 467, dev 0.413, test 0.446. All five: 0.456 / 0.439 / 0.453 /
  0.469 / 0.446, mean 0.4527 ± 0.0113, all within 18,000 s (246.6-263.6 min). `validate_record_v3.py --compare-to`
  A 006: Welch t = 5.31, df 7.9, one-sided p = 0.0004, so A 007 holds track A. README and leaderboard updated. Kept dev
  mean 0.429, so A's line is now 0.459.
- r81 (D 006 + `constrain_batched` + Muon 56 min pretraining, RL budget 17,400 s; H200): final dev 0.262 / 0.268 /
  0.275 (mean 0.268), each at or above D's line 0.259: the 3-seed promotion rule. Promoted to
  `records/v3/track_d/007_dm_math_sil64c05_batched_muon56` (r81's config minus `no_save`). It uses self-imitation
  (as D 006 does); said at the top of STATUS before its seeds start (INBOX 11-12).
- Risk for D 007: r81 seed 2's process exited about 15 min after its loop ended (GRPO wall 18,430 s for 17,428 s of
  loop), so its clock was 21,770 s against 21,600; seeds 0 and 1 were at 20,880 s and 20,885 s. The other recent D
  runs had 55-220 s of overhead. Asked in QUESTIONS 11-16 13:54 how such a seed should count.
- r87 (B200, seed 0, not yet gated): Muon checkpoint `muon1338m_s14180` final dev 0.240 against the AdamW control
  `d004_ctl_s11441` 0.263, one paired seed (-0.023). The box's r86 gives more seeds.
- D 007: seeds 0-3 on four preemptable H200 workers (RESERVE_MIN 380, no resume saves), limits ending before the 11-17
  00:00 maintenance; a fifth worker for seed 4 waits for the maintenance to end.


## 2026-11-16 14:30 EST: U18 coded (answer-opening mask, off by default); r88 on the two B200 slots

`mask_answer_open` (commit 701b913) sets the advantage of the tokens that spell a completion's first `<answer>` to 0
in the policy-gradient term only (they still count in the token mean; entropy bonus, KL and self-imitation rows are
unchanged), so when to answer stays at the model's prior. Tests: the mask marks exactly the tokens overlapping the first
OPEN, and masking a row's tokens gives the gradient of zeroing its advantage. ruff and pytest pass (166).

r88 `calibration/research_r88/d004_maskopen_s11441` is r87's control `d004_ctl_s11441` plus the mask. Seeds 0 and 1 run
on two mit_normal_gpu B200 workers (29360942, 29360944; about 3 h each, so they end before the 11-17 maintenance). Read
by final dev, paired by seed with r87's control (seed 0: 0.263; its seed 1 runs after the maintenance). Not gated.


## 2026-11-16 18:47 EST: r88 (answer-opening mask) seed 0 at 0.235 against its control's 0.263

| arm (B200, not gated) | seed | final dev | best in-loop dev | final entropy |
|---|---|---|---|---|
| r87 `d004_ctl_s11441` (control) | 0 | 0.263 | 0.270 | 0.090 |
| r88 `d004_maskopen_s11441` | 0 | 0.235 | 0.248 | 0.095 |
| r88 `d004_maskopen_s11441` | 1 | 0.237 | 0.243 | 0.098 |

Seed 0, paired with the control: -0.028. Seed 1's control (r87 `d004_ctl_s11441` seed 1) started at 18:06 on B200
worker 29387381 and should end about 21:20, before the maintenance. Masking the gradient on the answer-opening
tokens does not help on the one complete pair; leaving when to answer at the model's prior seems to cost dev rather
than gain it. The verdict waits for the seed 1 pair. Seed 0's node (node5702-c1) ran about 25 min slower than seed 1's;
both runs are step-capped at 11,441, so the scores compare.


## 2026-11-16 20:26 EST: D 007 seeds 0-3 done, all within budget; test mean so far 0.263

| seed | node | test ladder | final dev | pretrain_s | grpo_wall_s | total_s | within 21,600 s | warm start |
|---|---|---|---|---|---|---|---|---|
| 0 | node4901 | 0.2606 | 0.245 | 3370 | 17539 | 20880 | yes | 41 s |
| 1 | node4901 | 0.2538 | 0.252 | 3370 | 17534 | 20879 | yes | 41 s |
| 2 | node4602 | 0.2646 | 0.250 | 3369 | 17538 | 20879 | yes | 25 s |
| 3 | node4801 | 0.2726 | 0.255 | 3369 | 17536 | 20875 | yes | 11 s |

Four seeds: test mean 0.2629 (sd 0.0079), dev mean 0.250; D 006 holds at 0.2444, so +0.019 so far. Every seed
ended about 720 s inside the budget: none had r81 seed 2's slow exit (QUESTIONS 11-16 13:54). Seed 4 runs after the
maintenance (worker 29355448, pending). No GPU work fits before 00:00 on the freed H200s, so they stay idle until then.


## 2026-11-16 21:26 EST: r88 (answer-opening mask) read: no effect on two paired seeds; U18 parked

| seed | r87 control `d004_ctl_s11441` | r88 `d004_maskopen_s11441` | r88 - control |
|---|---|---|---|
| 0 | 0.263 | 0.235 | -0.028 |
| 1 | 0.217 | 0.237 | +0.020 |
| mean | 0.240 | 0.236 | -0.004 |

Final dev, B200, step-capped at 11,441, not gated. Control seed 1 (worker 29387381, node5600-c1): best in-loop dev
0.227, final entropy 0.070. The two pairs point in opposite directions and the control's two seeds differ by 0.047,
so the mask shows no effect larger than seed noise. U18 is parked: no further GPU time on it unless another result
points back to when the model answers.

## 2026-11-17 12:17 EST: maintenance over; r89 (U19 step 2: pretraining throughput, fused loss) on two B200 slots

INBOX 11-16 22:35 (RULES v3.6) read: a record seed over budget only because of idle exit time counts as it is, if its
logs show the last training step and the final evaluation inside the budget and the README names the seed with both
times; such a seed goes to QUESTIONS when it happens. D 007 seeds 0-3 did not need it. Seed 4 (29355448, node4802,
H200) started 16:00Z after the maintenance, warm start 216 s.

U19's step 2 is pretraining throughput in the Llama class. r82 ran at about 600k tok/s on B200 (370k on H200, about
26 % of peak). The loop in `scripts/pretrain_scratch.py` upcasts the full logits to fp32 outside the compiled model
(32 x 1024 x 32,000 floats, about 4 GB per micro-batch) before the cross-entropy. New option `--fused-loss` (off by
default; the default path is unchanged): the forward and the cross-entropy are compiled as one graph, so Inductor can
fuse the upcast into the loss. r89, pretraining only, 56 min, seed 0, B200, against r82's `pt56_muon_lr4e-3` (602k
tok/s, val 2.805): `pt56_muon_lr4e-3_fused` (micro-batch 32, as r82) and `pt56_muon_lr4e-3_fused_mb64` (micro-batch
64; the batch stays 524,288 tokens). Read by tok/s and final val loss. A clear gain in tokens per minute goes end to
end like r81.

## 2026-11-17 12:51 EST: r89 runs at +12 % and +16 % tokens per second; r90, short speed benchmarks on one H200 (INBOX 11-17 12:25)

INBOX 11-17 12:25 (maintainer): a change that only affects speed is ranked by a few hundred steps after the compile
warm-up (tokens per second), with the loss at matching steps against the old path as the check; long runs are for
changes to the maths.

r89 at about 19 min (B200), against r82's `pt56_muon_lr4e-3` at the same steps:

| step | r82 train / val | r89 fused mb32 train / val | r89 fused mb64 train / val |
|---|---|---|---|
| 500 | 2.553 / 3.445 | 2.552 / 3.450 | 2.803 / 3.440 |
| 1000 | 2.783 / 3.203 | 2.786 / 3.206 | 2.644 / 3.180 |
| 1500 | 2.351 / 3.076 | 2.360 / 3.084 | 2.453 / 3.085 |

Cumulative tokens per second (compile included): 602k (r82), 677k (fused, micro-batch 32), 696k (fused, micro-batch
64). The val losses agree within the 20-batch noise (micro-batch 64 draws different rows, so its train losses differ
batch by batch). Both arms run to 56 min for the val readout; the learning rate decays by wall time, so the faster arms
sit at a higher lr at the same step.

r90 ranks speed-only variants on one preemptable H200 (worker 29445218, queue `bench_h200`), 300 optimizer steps each
(157M tokens, the lr schedule by step), read as tokens per second between step 50 and the last step: the default path
at micro-batch 32 (reference), `--fused-loss` at micro-batch 32, 64 and 128, `--prefetch 4` at 64 and 128, and
`--compile-mode max-autotune-no-cudagraphs` at 64. Two new options, off by default: `--prefetch N` gathers the next N
training micro-batches into pinned memory on a thread (same draws from the rng; after a val eval the draws are offset),
and `--compile-mode` passes a mode to `torch.compile`.

## 2026-11-17 13:35 EST: r89 final: +16 % and +19 % tokens per second, the same val loss at 56 min; D 007 seed 4 preempted; r91 (r90 on B200)

r89 (B200, pretraining only, 56 min, seed 0) against r82's `pt56_muon_lr4e-3`:

| arm | steps | tokens | tok/s (cumulative) | val at 56 min |
|---|---|---|---|---|
| r82 `pt56_muon_lr4e-3` (default path, micro-batch 32) | 3857 | 2.02B | 602k | 2.805 |
| `pt56_muon_lr4e-3_fused` (micro-batch 32) | 4459 | 2.34B | 695k | 2.816 |
| `pt56_muon_lr4e-3_fused_mb64` | 4584 | 2.40B | 715k | 2.805 |

Val loss at 2000/3000/3500 steps: r82 2.985 / 2.883 / 2.832; fused 2.997 / 2.900 / 2.842; fused mb64 3.013 / 2.893 /
2.837 (the lr decays by wall time, so the faster arms are at a higher lr at the same step). In plain words: the fused
loss is 16-19 % faster, but the extra 16-19 % of tokens did not lower the 56-min val loss in this one seed (20 val
batches, noise about 0.01). The speed is worth having as time moved to RL (the same tokens in about 48 min), not as a
better checkpoint at 56 min; that is the end-to-end test once the benchmarks pick the fastest path.

D 007 seed 4 (job 29355448, node4802) was preempted at 13:19 EST after about 3 h 15 min (pretraining done, GRPO
running); Slurm requeued it. As the rules say, the partial goes to `extra/` and seed 4 reruns from the start (the
worker does this).

r91 is r90's seven benchmarks on B200 (`screens_b200`), since the H200 worker for r90 is still pending. The first two
(13:31 EST), 300 steps, tok/s between step 50 and step 300:

| arm | tok/s | train at 100 / 200 / 300 | val at 300 |
|---|---|---|---|
| `bench_base_mb32` | 615k | 4.296 / 3.516 / 3.278 | 3.767 |
| `bench_fused_mb32` | 701k (+14 %) | 4.311 / 3.510 / 3.279 | 3.769 |

The two r89 workers exited after these (under 300 min left); worker 29448582 (B200, 1.5 h) runs the other five.

## 2026-11-17 14:07 EST: r91 (speed benchmarks, B200): fused loss at micro-batch 128 is +21 %; r92 spends it as RL time (H200, end to end)

r91, 300 optimizer steps each (B200, seed 0), tokens per second between step 50 and step 300, and the losses at
matching steps as the check that only the speed changed:

| arm | tok/s | vs default | step 50 at | train at 100 / 200 / 300 | val at 300 |
|---|---|---|---|---|---|
| `bench_base_mb32` (default path) | 615k | | 69 s | 4.296 / 3.516 / 3.278 | 3.767 |
| `bench_fused_mb32` | 701k | +14.1 % | 58 s | 4.311 / 3.510 / 3.279 | 3.769 |
| `bench_fused_mb64` | 730k | +18.8 % | 56 s | 4.311 / 3.461 / 3.245 | 3.765 |
| `bench_fused_mb128` | 746k | +21.3 % | 55 s | 4.304 / 3.507 / 3.142 | 3.764 |
| `bench_fused_mb64_pf4` | 730k | +18.8 % | 44 s | 4.308 / 3.459 / 3.245 | 3.766 |
| `bench_fused_mb128_pf4` | 748k | +21.7 % | 43 s | 4.307 / 3.510 / 3.140 | 3.755 |
| `bench_fused_mb64_autotune` | 742k | +20.6 % | 207 s | 4.313 / 3.457 / 3.245 | 3.766 |

(Train losses at one step are one micro-batch, so they match only between arms with the same micro-batch, and do.)
In plain words: the fused loss gives most of the gain, a bigger micro-batch adds a little (the batch stays 524,288
tokens), prefetching adds nothing to the steady rate (only the first steps are quicker), and max-autotune adds 1.6 %
for 2.5 min more compiling. Fastest simple path: `--fused-loss --micro-batch 128`. r90 (the same on H200) is still
waiting for a preemptable slot; it will say whether the gain holds on the record GPU (track D pretraining runs at about
380k tok/s there).

r92 (U19 step 3) spends the speed as RL time, the way r81 spent Muon's: `calibration/research_r92/d007_fused128_pt46_rl18000`
is r81 / D 007 with `--fused-loss --micro-batch 128`, pretraining 46 min instead of 56 (about the tokens 56 min gave on
the old path) and RL 18,000 s instead of 17,400; the same 20,760 s total. Seeds 0-2 on H200 (queue `screens_r92`,
preemptable workers 29450693 / 29450694 / 29450702, 12 h limit), read paired against r81's seeds 0-2 (dev 0.262 /
0.268 / 0.275). Its pretraining log also gives the H200 tokens at 46 min against D 007's (1.288B at 56 min). If the H200
gain proves much smaller than on B200, 46 min gives fewer tokens than D 007 and the arm is read with that in mind.
`calibration/research_r92/pt46_muon_fused_mb128` (B200 worker 29450704, pretraining only, seed 0) checks the premise:
46 min on the fast path against r82's 56 min (val 2.805).

GPUs: 6 counting pending (r90 H200, D 007 seed 4, three r92 H200, one B200).

## 2026-11-17 15:12 EST: r92's B200 pretraining check (46 min fast path matches r82 at matching steps); INBOX 15:02; r93 (shared prompt in the RL update)

r92 `pt46_muon_fused_mb128` (B200, seed 0, pretraining only): 3,836 steps, 2.011B tokens in 46.0 min (729k tok/s), final
val 2.832. r82 `pt56_muon_lr4e-3` (micro-batch 32, unfused) reached 2.022B tokens in 56 min with val 2.805. Val at
matching steps (the speed-only check INBOX 11-17 12:25 asks for; the two runs' lr schedules are nearly the same at each
step, since both stop at about 3,850 steps):

| step | tokens | r82 (56 min, old path) | r92 (46 min, fused mb128) | diff |
|---|---|---|---|---|
| 500 | 0.262B | 3.4450 | 3.4503 | +0.005 |
| 1000 | 0.524B | 3.2028 | 3.1989 | -0.004 |
| 1500 | 0.786B | 3.0759 | 3.0880 | +0.012 |
| 2000 | 1.049B | 2.9849 | 2.9812 | -0.004 |
| 2500 | 1.311B | 2.9202 | 2.9082 | -0.012 |
| 3000 | 1.573B | 2.8835 | 2.8747 | -0.009 |
| 3500 | 1.835B | 2.8322 | 2.8474 | +0.015 |
| end | 2.01-2.02B | 2.8046 | 2.8322 | +0.028 |

In plain words: the fast path follows the old one step for step within the val estimate's noise (each val is 20
fresh random batches; r82's bt262k arm moved 2.805 -> 2.837 -> 2.818 between neighbouring evals). The final gap of
0.028 is at the edge of that noise and has no maths behind it (same batch of 524,288 tokens, same optimizer, the
loss only fused). So 46 min on this path gives about the checkpoint 56 min gave: r92's premise holds on B200.

INBOX 15:02 (maintainer: "more train time generally = more flexibility anyway"): a speed change with the same maths is
kept without having to win a paired screen; the time it frees can go to RL, pretraining or a larger model. Promotion
still follows PLAN. So `--fused-loss --micro-batch 128` becomes the default track D pretraining path for new D arms,
and r92 is read as where to spend the time, not whether to keep the path.

r93 (INBOX 11-17 14:59, item 1: share the prompt per group in the RL update). New option `share_group_prompt` (off by
default, commit 612d049): in `_completion_logprobs`, rows whose prompt columns match (a GRPO group) run those columns
once on top of the batch-wide prefix cache, and the cache is indexed out to the rows, so the gradients sum back through
the index. Exact like the prefix trick; tests check log-probs, entropies and gradients against the per-row path (GPT-2
and Llama, with and without a prefix, groups out of order, padded rows) and the whole `_update` with
`skip_zero_advantage` under both micro-batch modes. Groups split across length-sorted micro-batches share only within
each, so one arm also raises `micro_batch_tokens` to 131,072. Six items on one B200 (worker 29456195, mit_normal_gpu,
queue `screens_b200`), 300 RL steps each, D 007's config, from D 007 seed 0's pretrained checkpoint (start of RL) and
its final weights (late RL; INBOX 12:25 asks for both): `{start,final}_{ctl,share,share_mbt131k}`. Read by `t_update`
after step 50 and by loss and reward at matching steps.

GPUs: 6 counting pending (r90 H200, D 007 seed 4, three r92 H200, one B200 for r93).

## 2026-11-17 16:32 EST: r93 (shared prompt in the RL update): exact, no faster alone; with 4x micro-batches the update is 33 % faster; r94 separates the two

r93 (B200, worker 29456195, all six items ok). Means after step 50 of 300; memory is the whole process (vLLM's 0.35
share included, 63 GB on a B200), peak.

| checkpoint | arm | t_sample s | t_update s | wall s/step | peak GB |
|---|---|---|---|---|---|
| start | ctl (mbt 32,768) | 0.834 | 0.566 | 1.520 | 95.7 |
| start | share | 0.837 | 0.569 | 1.533 | 94.1 |
| start | share + mbt 131,072 | 0.741 | 0.378 | 1.241 | 168.9 |
| final | ctl | 0.783 | 0.390 | 1.295 | 83.5 |
| final | share | 0.676 | 0.389 | 1.203 | 76.8 |
| final | share + mbt 131,072 | 0.639 | 0.267 | 1.026 | 89.0 |

Step 1's loss is identical in every arm of a checkpoint (-0.2848 start, -0.0004 final); after that sampling differs
and reward stays in line (start step 300: 0.386 / 0.400 / 0.418; final: 0.592 / 0.610 / 0.598). `t_sample` moves with
answer lengths, not with the update path.

In plain words: sharing the prompt is exact but saves nothing by itself. After the batch-wide prefix only the puzzle's
few tokens are per-row, against up to 256 completion tokens, and length-sorted packing scatters a group across
micro-batches. With 4x larger micro-batches the update is 33 % faster on both checkpoints, but that arm changed two
things, and at the start of RL it peaks at 169 GB: on an H200 (141 GB, vLLM's share 49 GB) it would not fit.

r94 (B200, worker 29462557, queue `screens_b200`): `{start,final}_{ctl_mbt65k, share_mbt65k, ctl_mbt131k}`, the same
300-step timing. It separates the micro-batch size from the sharing and checks 65,536 tokens, which should fit an H200.

GPUs: 6 counting pending (r90 H200, D 007 seed 4, three r92 H200, one B200 for r94).

## 2026-11-17 17:36 EST: r94 (micro-batch size vs shared prompt): both together give the gain; 65,536 tokens shared fits the H200; r95 times the update's parts

r94 (B200 worker 29462557, 300 steps each, no_save; means over steps 51-300; memory is the whole-process peak, vLLM's
0.35 share included) with r93's 32,768-token rows for reference:

| from | arm | t_update s | wall/step s | peak GB |
|---|---|---|---|---|
| D 007 seed 0 start | ctl, 32,768 tokens (r93) | 0.566 | 1.520 | 95.7 |
| | share, 32,768 (r93) | 0.569 | | 94.1 |
| | ctl, 65,536 | 0.524 | | 127.8 |
| | share, 65,536 | 0.484 | | 121.7 |
| | ctl, 131,072 | 0.553 | | 174.1 |
| | share, 131,072 (r93) | 0.378 | 1.241 | 168.9 |
| D 007 seed 0 final | ctl, 32,768 (r93) | 0.390 | 1.295 | 83.5 |
| | share, 32,768 (r93) | 0.389 | | 76.8 |
| | ctl, 65,536 | 0.358 | | 102.3 |
| | share, 65,536 | 0.318 | 1.106 | 85.6 |
| | ctl, 131,072 | 0.337 | | 112.4 |
| | share, 131,072 (r93) | 0.267 | 1.026 | 89.0 |

Step 1 losses are the same in every arm (-0.2848 start, -0.0004 final); rewards in line. Larger micro-batches alone
save little (-7 % at 65,536, -2 % to -14 % at 131,072); the shared prompt alone nothing; together -14 % / -18 % at
65,536 and -33 % at 131,072. Rows are length-sorted and packed, so a group's rows only share a micro-batch (and so its
prompt) when micro-batches are large. H200 fit: at the start of RL 131,072 peaks at about 169 GB on the B200, i.e.
about 106 GB of training memory plus vLLM's 49 GB on an H200 = 155 GB > 141 GB, which does not fit; shared 65,536 is
about 59 + 49 = 108 GB, which does. `share_group_prompt` with `micro_batch_tokens` 65,536 is the speed setting for new
D arms (same maths, INBOX 15:02); the records keep their configs.

INBOX 17:14 (dbd2333): the update is not bound by token positions, so time its parts first. New `time_update_parts`
(off by default): synced timers logged as `t_part_*`: group selection, `_allowed_tokens`, the replay batch, the
forward's body / allowed-token mask / head, the loss, the backward, self-imitation, the clip and the optimizer step,
plus micro-batch counts; a test checks the gradient is unchanged. r95 (B200 worker 29468588, queue `screens_b200`,
150 steps each): ctl and shared-65,536 from D 007 seed 0's start and final weights, to find the largest part before
fixing it.

INBOX 17:23 (403f9af): the 6-GPU cap counts running GPUs only (2 on mit_normal_gpu, 4 on mit_preemptable); pending
jobs do not count. My H200 jobs (D 007 seed 4, r92 seeds 0-2, r90) are still pending; B200 jobs start in minutes.

## 2026-11-17 18:12 EST: r95 (parts of the RL update, B200): the update is GPU-bound; sampling is half the step; r96 times sampling

r95 (B200 worker 29468588, queue `screens_b200`, 150 steps each, D 007 seed 0's start and final weights,
`time_update_parts`). Means over steps 51-150, seconds per step; the parts add up to `t_update`.

| arm | wall | sample | update | fwd body | bwd | SIL replay | allowed | fwd mask | replay batch | fwd head | clip+opt | micro-batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start_ctl | 1.273 | 0.629 | 0.519 | 0.136 | 0.206 | 0.058 | 0.065 | 0.024 | 0.012 | 0.007 | 0.003 | 4.00 |
| start_share_mbt65k | 1.192 | 0.646 | 0.431 | 0.108 | 0.158 | 0.055 | 0.067 | 0.017 | 0.010 | 0.006 | 0.003 | 2.55 |
| final_ctl | 0.952 | 0.488 | 0.342 | 0.077 | 0.117 | 0.056 | 0.054 | 0.012 | 0.013 | 0.004 | 0.003 | 2.43 |
| final_share_mbt65k | 0.850 | 0.464 | 0.260 | 0.050 | 0.071 | 0.055 | 0.050 | 0.011 | 0.012 | 0.004 | 0.003 | 1.50 |

Reading (INBOX 11-17 17:14 suspects): the head and log-softmax over the vocabulary (0.004-0.007 s) and the optimizer
(0.003 s) are negligible. The Python grammar work (`_allowed_tokens` 0.05-0.07 s, plus the mask and the replay batch)
is 0.07-0.10 s, 7-8 % of the step: real but not the largest. The forward and backward body is the largest part of the
update, and it already shrinks with the shared prompt and larger micro-batches. The self-imitation replay costs about one
micro-batch (0.055 s). The largest part of the whole step is sampling (vLLM): 0.46-0.65 s, 50-55 %.
So the next timing splits sampling: r96 (B200, queue `screens_b200`) reruns r95's two shared-prompt arms with new
sampling laps under the same flag (encode, vLLM generate, decode, verify and rewards; plus the constrained logits
processor's CPU time inside generate, `t_part_s_mask_cpu`). The fix then goes to whichever is largest: the processor's
per-token grammar work, or the sampler itself (U17).

## 2026-11-17 19:16 EST: r96 (parts of sampling, B200): vLLM's generate is 95 % of sampling; the answer mask's CPU is 10-12 % of it; a sampling profile runs

r96 (B200 worker 29471792, node5502-c1, 150 steps each, r95's shared-prompt arms). Means over steps 51-150, seconds:

| arm | sample | encode | vLLM generate | of which mask CPU | decode | verify + rewards | update |
|---|---|---|---|---|---|---|---|
| start_share_mbt65k | 0.618 | 0.006 | 0.590 | 0.069 | 0.007 | 0.014 | 0.404 |
| final_share_mbt65k | 0.482 | 0.006 | 0.459 | 0.044 | 0.004 | 0.013 | 0.273 |

- Everything around the engine (prompt encoding, decoding, the verifier, rewards) is about 0.025 s. The constrained
  processor's own CPU time is 0.04-0.07 s. The other 0.4-0.5 s is inside vLLM, for about 512 completions of 25-30
  tokens: tens of milliseconds per decode step for a 110M model, so per-request engine overhead rather than the GPU.
- vLLM's default `max_num_seqs` is 1,024 on both GPUs, so the 512 requests are not split into waves.
- Next: `scripts/research/profile_sampling.py` (timing only, no training) times `generate` on fresh batches with
  the final weights, with and without the constraint, and takes a cProfile of three calls, to find the vLLM
  functions that cost the time (B200 job 29477068). The fix depends on it: vLLM settings if the cost is
  detokenization and stop strings, else a same-process sampler (U17).

## 2026-11-17 23:09 EST: INBOX 20:23 (more use out of the B200s); r97 (D speed stack end to end, B200); H200 gates held

- INBOX 11-17 20:23: U20 is about getting more use out of the B200s; records stay on the H200. Since noon my jobs had
  used 5.5 GPU-hours out of about 50, and nothing had run since 18:53. The agent was not woken between 19:49 and 23:06.
- The H200s are jammed: 236 H200 jobs pending on mit_preemptable, and mine have waited since 13:19. The B200 nodes
  there have free GPUs.
- r97 (B200, not yet gated, queue `screens_d_b200`, preemptable workers 29497701/02/05, one run per GPU): D 007 end
  to end with every same-maths speed change so far (`d007_speed_b200`: r92's pretraining path, `--fused-loss`,
  micro-batch 128, Muon 46 min, 600 s moved to RL; plus r94's `share_group_prompt` and 65,536-token micro-batches),
  against D 007's config on the same GPU type (`d007_ctl_b200`), seeds 0-2. It reads RL steps reached and final dev.
  More RL steps have raised D before (r59: 0.239 at 10k steps, 0.258 at 15k; r78: 20k steps +0.032 over r59).
- r97's arm contains r92's change, so r92's three H200 workers (29450693/94/702) and the r90 H200 benchmark (29445218;
  r91 measured the same on B200) are held, not cancelled. That keeps my running preemptable GPUs at 4 or fewer if
  D 007 seed 4 (29355448, still pending on an H200) starts while r97 runs.
- The sampling profiler failed twice more: outside the worker it needs the worker's environment (CUDA module,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, and GCC's libstdc++ first on `LD_LIBRARY_PATH` for FlashInfer's kernels, as in
  `packed_worker.sbatch`). Resubmitted with all three as B200 job 29497515.
- The 11-18 box: `box_sync.sh to` needs the box up, so it runs after the 06:30 EST launch.

## 2026-11-18 01:10 EST: r96 profile (vLLM's per-request bookkeeping is the sampling cost); a same-process sampler next

- Profile job 29497515 (B200, D 007 seed 0's final weights, `profile_sampling.py`, timing only). Constrained:
  median 0.473 s per `generate` call (mean 0.505) for 512 completions of mean length 24.7 and mean longest 36.3
  tokens, about 13.9 ms per decode step; the constraint's own CPU time is 0.050 s. Unconstrained: median 2.18 s,
  mean length 166, longest 256, 8.6 ms per decode step.
- cProfile of three constrained calls (3.48 s under the profiler, 112 engine steps, about 340 live sequences per
  step): scheduler 1.08 s (slot allocation 0.58), `update_from_output` 0.61, output processing 0.43 (detokenizer
  0.32), the constraint 0.39, `execute_model` 0.55 (of which sampling with the logits processor 0.46). The model
  forward itself is under about 0.1 s. Prefix-cache hashing is about 0.17 s.
- So per-request Python bookkeeping in vLLM, not the GPU, sets the time for a 110M model with short answers.
  Tuning vLLM (no prefix cache, no detokenizer) would remove a fifth to a third of it at most.
- Next (INBOX 11-17 14:59, "surely you can do better than vllm"): a sampler of our own in the training process,
  behind a new `rollout_backend` value that is off by default. It will have a bf16 Llama decode step captured in a
  CUDA graph and a static KV cache. Each distinct prompt is prefilled once and its cache shared across the group.
  It keeps the same `RequestConstraint` grammar and the same stop rules (`</answer>` or EOS). Expected: sampling
  from about 0.47 s to 0.15-0.2 s per step, about 25-30 % more RL steps in D's budget. It will be timed against
  r96's arms on B200 before any end-to-end screen.


## 2026-11-18 01:18 EST: the same-process sampler is in (`rollout_backend: "native"`, off by default); r98 times it

- `rlvr_speedrun/fast_rollout.py` (`NativeRollout`, f9cf5c2): a bf16 copy of the Llama policy with fused qkv and
  gate/up weights, a static KV cache, each distinct prompt prefilled once and its cache copied to its n rows, the
  decode step captured as a CUDA graph per (rows, key length) bucket, the answer grammar applied in the same process,
  and exact sampling (exponential-noise argmax; argmax at temperature 0). It stops on EOS, on `</answer>` or at the
  token limit, and returns what `VLLMRollout` returns. `sync` copies the weights in after each update.
- Tests: greedy tokens equal the HF model's argmax (full and grouped KV heads, and again after `sync`); constrained
  samples follow the grammar and stop at the token that closes the answer; `grpo.run` trains with it. 177 passed.
- r98 (B200 job 29508506, timing only, D 007 seed 0's final weights): `profile_sampling.py --backend native` with and
  without CUDA graphs, then vLLM on the same node. It reads wall time per call and completion lengths (the lengths
  should match vLLM's, since both sample the same distribution). If it is well under vLLM's 0.47 s, 300-step
  benchmarks against r96's arms follow (tok/s, loss at matching steps), then an end-to-end D arm.

## 2026-11-18 03:00 EST: r98 profile: the native sampler is 2.2x faster than vLLM per generate call; 300-step benchmarks queued

Job 29508506 (B200, node5601-c1; D 007 seed 0's final weights, 64 prompts x 8, 20 timed calls each, timing only):

| sampler | constrained median s/call | mean length | mean longest | constraint s/call | unconstrained s/call (256 tokens) |
|---|---|---|---|---|---|
| vLLM (as in D 007) | 0.441 | 24.68 | 36.3 | 0.046 | 2.066 |
| native, CUDA graphs | 0.199 | 24.69 | 36.25 | 0.155 (includes waiting on the GPU) | 1.479 |
| native, no graphs | 0.239 | 24.69 | 36.25 | 0.067 | 1.748 |

- Same lengths as vLLM, as expected from the same distribution. 2.22x faster constrained, 1.40x unconstrained.
  The weight copy after each update (`sync`) takes 0.6 ms.
- With graphs the host waits for the GPU inside the grammar step, so its 0.155 s is not CPU work. Without graphs the
  grammar's real CPU time is 0.067 s, a third of the call. That is the next piece to overlap with the GPU.
- Sampling was 0.48-0.62 s of a 0.95-1.0 s step in r96, so this should save roughly a quarter of the step.
- Next, r98's benchmarks (B200 worker 29516197, queue `screens_b200`, 300 steps each, no_save): r94's
  `share_mbt65k` arms from D 007 seed 0's start and final weights, native against vLLM on the same node. The
  native arms turn off `constrain_batched`, which is a vLLM option. They read tok/s and step time, and loss and
  reward at matching steps.

## 2026-11-18 04:18 EST: r98 benchmarks: the native sampler makes D's RL step 1.4-1.6x faster at matching reward; r99 runs it end to end

Worker 29516197 (B200, node5801-c1), 300 steps each, no_save, r94's `share_mbt65k` arms from D 007 seed 0's start and
final weights, native sampler against vLLM on the same node:

| weights | sampler | wall s | tok/s | sample s/step | update s/step | reward, steps 1-100 / 101-200 / 201-300 | entropy, 201-300 | peak GB | dev at 300 |
|---|---|---|---|---|---|---|---|---|---|
| start | vLLM | 385.4 | 13,549 | 0.714 | 0.477 | 0.2999 / 0.3626 / 0.3887 | 0.354 | 122.0 | 0.0717 |
| start | native | 241.4 | 25,845 | 0.157 | 0.396 | 0.2941 / 0.3588 / 0.3848 | 0.313 | 75.2 | 0.0617 |
| final | vLLM | 246.4 | 18,569 | 0.445 | 0.257 | 0.5957 / 0.5992 / 0.5947 | 0.0911 | 85.4 | 0.2500 |
| final | native | 175.6 | 34,693 | 0.147 | 0.287 | 0.5966 / 0.5968 / 0.5964 | 0.0875 | 39.4 | 0.2517 |

- Wall time 1.60x (start) and 1.40x (final) faster; tok/s 1.91x and 1.87x (wall includes vLLM's longer start-up).
  Sampling drops from 0.45-0.71 s to 0.15-0.16 s per step; the update is now most of the step.
- Rewards match within noise at every 100-step window from both weights, so the maths is the same, as built.
  The start arm's entropy (0.313 vs 0.354) and dev (0.062 vs 0.072) are a little lower with native: one seed, 300
  steps, and the dev at step 300 of a start-weights run is noise-sized; the end-to-end arm reads it properly.
- Peak memory drops by about 46-47 GB (no vLLM pool).
- Speed change with the same maths (INBOX 11-17 15:02): kept. `rollout_backend: "native"` is the sampler for new D
  arms; it stays off by default.
- r99 (`calibration/research_r99/d007_native_b200`): r97's speed arm with the native sampler, seeds 0-2, end to end,
  on queue `screens_d_b200` interleaved as speed / ctl / native per seed. r97's three workers take the first two
  rounds; the third round needs workers when those end. Reads RL steps reached and final dev, paired by seed.
- Next for speed: the grammar's 0.067 s of CPU per call runs while the GPU waits; overlapping it is the next piece.

## 2026-11-18 04:22 EST: `reuse_sampler_allowed` (off by default): the update takes the sampler's allowed sets; r100 times it

- r95 measured the update's `allowed` part (the grammar's allowed sets recomputed in Python from each row's tokens, for
  the constrained training mask) at 0.050-0.067 s per step. With the native sampler that is now about a fifth of the
  update. The native sampler already computes the same sets as it samples, so it now records them per row
  (`NativeRollout.last_allowed`), and with `reuse_sampler_allowed` the update takes them instead of recomputing. A row
  whose valid tokens are not exactly the sampled prefix (a sampled pad token that is not EOS) is recomputed as before.
  Same values, so the same maths; the self-imitation replay batch still recomputes (0.012 s).
- Tests: the recorded sets equal `allowed_per_position` on every constrained row; `grpo.run` with the flag on checks
  that the sets it uses equal the recomputed ones at every step; a row with a non-prefix valid mask is recomputed.
  179 passed. (The first version shadowed the prompt ids in `generate`; the tests caught it before any job read it.)
- r100 (B200 worker 29521171, queue `screens_b200`, 300 steps each, `time_update_parts` on): r98's native arms from D
  007 seed 0's start and final weights, flag off against on. Reads t_update, the `allowed` part and tok/s, with reward
  at matching steps.

## 2026-11-18 12:03 EST: the 11-18 H100 box is up, 5 h 16 min late; all eight GPUs run r85 and r86

- Launched at 11:46 EST (i-0502d26f396883825, p5.48xlarge, 174.129.98.131) with the two allowed AWS commands, not at
  06:30: the previous account's weekly usage limit stopped this session at about 04:25 (INBOX 11-18 11:47), so no wake
  ran at 06:30. The box ends at 06:30 EST on 11-19 (worker deadline 05:55): about 18 h of its 24 h are left.
- Ready at 11:49. The copy job (29551164, mit_quicktest) sat pending behind other jobs, so on the maintainer's word
  (via the laptop session) the copy ran from the login node, niced: code, configs, queues, base model and tokenizer
  in 42 s, then the 15 D checkpoints (3.2 GB) in 34 s. That is network-bound rsync, about a minute of light CPU. The
  quicktest job started at the same time and finished as a second pass over the same files (SYNC_OK); a duplicate on
  mit_normal (29551402) was cancelled before it started.
- Workers (docs/research/box_1118.md, with `SCREEN_RESUME=1` as on the 11-05 box): A on GPUs 0-3 at 11:58
  (`box_a_1118`, one per GPU), D on GPUs 4-7 from 11:58 to 12:01 (`box_d_1118`, three per GPU). At 12:02 all 16 runs
  were training (vLLM's kernel warmup took about 2 min on the fresh box).
- Left in the queues: 8 A items and 6 D items. D's second round fills only half of GPUs 4-7, so more D items go on
  once the first rates are read. Results come back with `box_sync.sh from box1118` each wake.

## 2026-11-18 12:03 EST: r100: `reuse_sampler_allowed` is bit-identical over 300 steps and cuts the update 14.5 % on the final weights; kept

| weights | flag | tok/s | sample s/step | update s/step | allowed s/step |
|---|---|---|---|---|---|
| final | off | 29,345 | 0.152 | 0.293 | 0.0487 |
| final | on | 32,749 | 0.150 | 0.251 | 0.0002 |
| start | off | 20,219 | 0.245 | 0.442 | 0.0570 |
| start | on | 16,051 | 0.440 | 0.451 | 0.0002 |

(B200 node5801, worker 29521171, 300 steps each, means over steps 21-300.)

- **Final weights:** reward, loss, entropy and grad norm are equal at every one of the 300 steps, flag on against off,
  and both equal r98's native run. So the flag is the same maths in a full run, not only in the tests. The update is
  0.043 s shorter (-14.5 %) and tok/s is +11.6 %.
- **Start weights:** not a usable whole-step comparison. The two runs differ from step 2, but so do r98's native run
  and r100's control, which are the same maths: at step 1 the loss is equal to every digit and the grad norm differs
  in the fifth, and at entropy 3.2 that changes the samples one step later. The three same-maths runs average
  21,637, 20,219 and 16,051 tok/s at equal tokens per step; the flag-on run took the
  highest-entropy path, where sampling ran 0.44 s against 0.24 s per step (the sampler code is the same
  in both arms). The allowed part itself fell from 0.057 s to 0.0002 s there too.
- **Verdict:** kept (INBOX 11-17 15:02: same maths, faster). New D arms on the native sampler set
  `reuse_sampler_allowed`. r99 is already queued without it and stays as it is.

## 2026-11-18 12:09 EST: box: three D runs per GPU do not fit an 80 GB H100; seven r86 runs died at about step 100 and are requeued at two per GPU

- **What failed.** Each r86 run takes about 40 GB once its update has seen a long batch (about 32 GB in PyTorch plus
  vLLM's 6.4 GB share), not the 15 GB it showed at step 40. With three per GPU the third run to grow hit
  `CUDA out of memory` between 12:02 and 12:05: seven runs failed (`d004_ctl_s11441` seeds 2 and 3,
  `muon1338m_s14180` seeds 2 and 3, `muon2676m_s11441` seeds 1, 2 and 3) and five workers left after two quick
  failures. No run had saved state yet, so nothing but 4-8 minutes of GPU time was lost. The plan's "three runs per
  GPU" (box_1118.md) copied the H200 packing (r78, 141 GB per GPU) and was not checked against the H100's 80 GB.
- **Fix (12:07).** The seven items went back to the front of the box's `box_d_1118` queue and a second worker started
  on GPU 5. GPUs 4-7 now hold two D runs each (80.2-80.9 GB of 81.6 GB in use), GPUs 0-3 one A run each. All eight
  are busy.
- **Rates.** D: 2.15 s per step with two per GPU (1.47 alone), so 6.8 h for an 11,441-step run and
  8.5 h for a 14,180-step run. Two rounds of eight fit before the 05:55 deadline; the last items start after
  01:00 and will be cut. A: 25.5 s per step over the first 15 steps, which would be 3.3 h per run against the
  plan's 5.5 h; later steps are expected to be slower.
- **Queues.** So that no GPU runs dry: seeds 3 and 4 of the four r85 arms were appended to the box's A queue (16
  items waiting), and `muon1338m_s11441` seeds 3 and 4 to its D queue (12 waiting for eight slots).
- **Watch.** Two D runs leave about 1 GB free per GPU. A run that fails later is requeued and resumes from its saved
  state (every 600 s).

## 2026-11-18 12:36 EST: Engaging: the B200 preemptable workers were preempted within 21 min each; r101 (D 007's RL with the native speed stack, RL only) on two normal B200 slots

- **What happened.** Workers 29497701 and 29497702 (r97 seed 0, control and speed arm) started at 08:35 and were
  preempted at 08:54 and 08:56. Worker 29497705 (r99 seed 0) ran 08:45-08:50 and 11:28-11:39. All three are pending
  again and every run was still in pretraining, so r97 and r99 have no result. No Engaging GPU was running at 12:30.
- **r101 (B200, not yet gated, RL only).** The same question as r99 in a form that fits the 6 h `mit_normal_gpu`
  limit, where jobs are not preempted: the RL stage starts from D 007's own seed-s pretrained checkpoint
  (`screen_pretrained`), `time_budget_s` 17,400 in both arms.
  - `d007rl_ctl_b200`: D 007's RL unchanged (vLLM sampler, `constrain_batched`).
  - `d007rl_native_b200`: every same-maths RL speed change so far: `rollout_backend: "native"`,
    `reuse_sampler_allowed` (r100), `share_group_prompt` with `micro_batch_tokens` 65,536 (r94).
- **Read.** Paired by seed: RL steps done in 17,400 s and final dev pass rate. More RL steps have helped D, so the
  expectation is a higher pass rate from about 1.5x the steps (r98's step rate). If the steps rise and the pass
  rate does not, the speed stack alone does not justify a record attempt.
- **Workers.** 29554919 and 29554920 (`mit_normal_gpu`, 6 h, one item each), queue `screens_rl_b200`, seeds 0-2, one
  pair per round of about 5 h. The r97 and r99 items stay queued for the preemptable workers.
- **Box (12:36).** All eight GPUs busy, no new failure. D 1.9-2.1 s per step at steps 800-1,060. A 19-28 s per step
  at steps 83-97 and rising, so A's per-run time is still open (2.7-3.5 h).

## 2026-11-18 13:00 EST: box: A's step time settled at 40-50 s, so three A rounds fill the box and no sixth is needed; r101's control started

- **A on the box (12:59).** Steps 105-127 after about 3,450 s, 40.5-50.0 s per step over the last 40 steps (19-28 s
  before step 100). This is the profile of A 007's seed 0 on the H200: 18.7 s per step over steps 1-100, 37.2 s over
  steps 121-467, 15,708 s in all. Projected from each run's rate: 4.78 h (cos467), 4.99 h (ctl), 5.50 h (ent15),
  5.98 h (lr1.3e-6), mean 5.31 h, close to the plan's 5.5 h.
- **So.** Three rounds (seeds 0-2 of the four arms) end about 03:54 on 11-19 on average, and 05:54 if every round
  on a GPU ran at the slowest rate. The 2.7-3.5 h in the 12:36 entry was read before the slowdown and is wrong.
  No sixth round is needed. A fourth round will start after about 03:54 and be cut at 05:55.
- **D on the box.** 1.6-2.0 s per step at steps 1,446-1,806: about 6.1 h per 11,441-step run and 7.5 h per
  14,180-step run. No new failure; all eight GPUs busy.
- **r101.** Worker 29554919 started `d007rl_ctl_b200` seed 0 at 12:39 on node5600-c1: 882 steps in the first 19 min,
  about 13,500 steps at 17,400 s if the rate holds. Worker 29554920 (`d007rl_native_b200` seed 0) is pending, with
  Slurm's start estimate at 19:00, so the pair will not run side by side.

## 2026-11-18 14:14 EST: INBOX 13:35 and 13:39 read; box: A runs take about 6.4 h, so A's third seed moves to the D GPUs

- **INBOX 13:35 (aceaf15).** The 11-28 H200 box has to launch and run with nobody awake: an AWS-side launch at 11:30Z
  with retries, the box fetching code, queues, checkpoints and venv pins at boot and starting its own workers, results
  and saved state leaving the box without a pull, a 24 h queue packed against 141 GB, and no clash with a launch by me.
  It is to be tested (a dry run under the launching role, one boot of a cheap instance) for a few dollars, and built and
  written up in `docs/research/box_1128.md` by 11-25. I read the note as allowing the AWS calls this build needs (the
  earlier notes allowed only `describe-instances` and `run-instances`); every charge will be logged. Not started: today's
  box comes first.
- **INBOX 13:39 (480d57d).** Two more 8x H100 days: 11-25 (`cr-03ab208e99ca6ca06`) and 12-09 (`cr-0813172fa29139b78`),
  11:30Z to 11:30Z, us-east-1f, same IP and launch path as today, screens only. Each needs a launch with its own
  reservation id, a 24 h queue and a packing checked against 80 GB. Still to check: that both show `scheduled`.
- **A on the box (14:12).** The 13:00 entry's 5.3 h was too low. Over the last 100 steps:

  | arm (seed 0) | step | s per step | projected total | projected end |
  |---|---|---|---|---|
  | `a007_cos467_h100` | 204 | 55.5 | 6.25 h | 18:15 |
  | `a007_ctl_h100` | 203 | 55.4 | 6.25 h | 18:16 |
  | `a007_ent15_h100` | 189 | 56.3 | 6.53 h | 18:32 |
  | `a007_lr1.3e-6_h100` | 190 | 54.9 | 6.41 h | 18:25 |

  The mean is 6.36 h. Round 2 then ends about 00:30-01:05, and a third round on GPUs 0-3 would end after 06:45, past the
  block (the box may go at 06:00). If late steps ease as they did on the H200 (42 s to 35-36 s), a run takes about
  5.7 h and a third round still ends after 05:05, which is too close to plan on.
- **D on the box (14:12).** At the last 800 steps' rates the five 11,441-step runs end 17:24-17:46 (5.33-5.76 h) and
  the three 14,180-step runs end 18:32-19:11 (6.52-7.11 h).
- **Plan for the tail, to set up about 17:10 (before the first D run ends).**
  - A's seed 2 (four arms) goes on GPUs 4-7, one run per GPU as each GPU empties of D runs (about 17:45 to 18:45), and
    ends about 00:10-01:10. A's seeds 3 and 4 leave the box queue; they never fitted.
  - D's 12 queued runs wait, then run one per GPU on all eight GPUs as each A run ends (00:10-01:10). A pair of
    11,441-step runs on one GPU needs 5.6 h, which only the first free GPU might have. So 8 or 9 of the 12 fit; the ones
    left out are from `muon1338m_s11441`, the arm box_1118.md calls extra.
  - That gives A 12 runs (three seeds per arm) and D 16 or 17. Leaving the queues alone gives A 8 finished runs and D at
    most 20, with A's third round cut.
  - Not measured yet: a D run's speed alone on an H100. One run needs under about 1.45 s per step to finish 11,441
    steps in the 4.7 h the last GPUs will have. The first reading comes when one run of a pair ends (about 17:24).
- **Engaging.** r101's control (29554919) is 94 min in; 29554920 is pending with a 17:28 start estimate. D 007 seed 4 (29355448) is still
  pending.

## 2026-11-18 15:33 EST: box: A control seed 0 ran out of GPU memory at step 271; requeued, it resumes from step 261

- **What happened.** `calibration/research_r85/a007_ctl_h100` seed 0 failed on GPU 2 at 15:13:47 EST with
  `torch.OutOfMemoryError` at `loss.backward()` ("Tried to allocate 3.74 GiB ... 3.67 GiB is free"), about step 271 of
  467. Its worker recorded it FAILED and started the next queue item, `a007_ctl_h100` seed 1, at once.
- **Why.** Every A run on the box has used 79.0-79.17 GB of the H100's 79.18 GB since its first 100 steps
  (`gpu_mem_gb` in the train logs): the sampler's share (`vllm_gpu_memory` 0.35, 28 GB here against 49 GB on an H200)
  plus training fills the card. One long batch is enough to fail, so the other A runs can fail the same way.
- **Fix today.** The run's saved state from step 261 was intact (written 15:03). At 15:25 I put the item back at the
  front of the box's `queue/box_a_1118.txt`; the next free A worker resumes it with 206 steps left (about 2.8 h at
  49 s per step). Each wake I check `queue/done.txt` for new FAILED lines (8 now) and put failed runs back at the
  front. I did not change the config or the scripts that the running workers read.
- **Tail plan, adjusted.** As in the 14:14 entry, set up about 17:10, with this A order: control seed 0 (resume), seed 1
  of the three other arms, then seed 2 of all four. Control seed 1 already runs on GPU 2 and ends about 21:37, so
  GPU 2 takes D runs after that.
- **For the next boxes.** The 11-25 and 12-09 H100 days need A configs that leave headroom on 80 GB (a smaller sampler
  share, same maths), and the self-running box must retry a failed run from its saved state without anyone awake.
- **Reservations.** `describe-capacity-reservations` showed both new blocks (`cr-03ab208e99ca6ca06`,
  `cr-0813172fa29139b78`) as `scheduled` at 14:17 (INBOX 13:39).
- **Box at 15:33.** All eight GPUs busy, 12 runs, queues 16 (A) and 12 (D). No AWS charge so far beyond the blocks.

## 2026-11-18 17:07 EST: box: the tail runs from per-GPU queues with a keeper on the box

- **Why the 15:33 tail plan was dropped.** One shared tail queue with the A runs first would have started the remaining
  D runs as pairs at about 00:40-01:35, and a paired 11,441-step D run takes about 5.6 h here, so none would have
  finished before the box ends. After one A run (about 6.4 h) a GPU has only 4.3-5.3 h left. A shared queue cannot say
  which GPU takes what, so each GPU now has its own queue.
- **What is on the box now.** At 16:59 EST I moved the two old queues to hold files (`queue/hold_a_1118.txt`,
  `queue/hold_d_1118.txt`), so each old worker exits when its run ends. At 17:06 I wrote `queue/box_g<g>_1118.txt` for
  each GPU and a shared overflow queue `queue/box_t_1118.txt`, and started `~/tail_keeper.sh` on the box (a copy is in
  `agent/box_tail_keeper_1118.sh`). Once a minute it looks at every GPU. When a GPU is empty on two passes and its
  queue or the overflow queue holds an item, it starts a worker with `QUEUE=box_g<g>_1118:box_t_1118`. It starts a
  second worker beside a lone D run only when the GPU's next item is a D run and at least 400 min remain. Workers start
  nothing with less than 240 min left (after 01:55 EST). The workers use `scripts/aws/box_runner.sh`, which reruns a
  failed item from its saved state (up to two tries without progress) before it is recorded FAILED.
  Stop the keeper with `touch ~/tail_keeper.off` on the box.
- **Plan per GPU** (times EST, projected from the run times measured so far):

  | GPU | free from | its queue | projected end |
  |---|---|---|---|
  | 0 | 18:15 | A `lr1.3e-6` seed 1 | 00:40 |
  | 1 | 18:25 | A `ent15` seed 1 | 00:50 |
  | 2 | 21:37 | D `muon1338m_s14180` seeds 2, 3 (pair) | 04:40 |
  | 3 | 18:32 | A `cos467` seed 1 | 00:55 |
  | 4 | 19:05 | A control seed 2 | 01:30 |
  | 5 | 17:35-17:42 | D `muon2676m_s11441` seeds 1, 2, then 3, 4 (two pairs) | 04:50 |
  | 6 | 18:32 at the latest | A control seed 0 (resume from step 261), then A `ent15` seed 2 | 03:40 |
  | 7 | 19:11 | A `lr1.3e-6` seed 2 | 01:35 |

  Overflow queue, taken by GPUs 0, 1, 3, 4 and 7 after their A run, one D run per GPU: `d004_ctl_s11441` seed 3, then
  `muon1338m_s11441` seeds 1, 2, 0, 3, 4. Held back: A `cos467` seed 2 and A seeds 3-4.
- **Not known yet.** How fast one D run goes alone on an H100 (all D runs so far ran as pairs, about 1.75 s per step).
  `muon1338m_s14180` seed 0 runs alone on GPU 6 from about 17:24, which gives the number. A single 11,441-step run has
  to finish in 4.3-5.2 h in those slots; if it cannot, I take the overflow items out rather than start runs that
  cannot finish.
- **Also open.** The resumed A control seed 0 meets the same long batch at about step 271, a few minutes after it
  starts on GPU 6. If it fails twice there, the worker goes on to the next item and I look at it again.
- **Box at 17:06.** All eight GPUs busy, 12 runs, 8 FAILED lines in `queue/done.txt` (unchanged). No AWS charge so far
  beyond the blocks.

## 2026-11-18 17:55 EST: AWS: a box that launches and runs by itself passed its cheap test; box A runs move to a 0.30 sampler share after three ran out of memory

- **Set up on AWS for INBOX 11-18 13:35** (17:09-17:16 EST, all through `scripts/aws/box_auto.sh`):
  - `iam`: role `rlvr-scheduler` (the role whose 11-05 launches were refused) now has an inline policy `run-rlvr-box`
    that allows RunInstances on launch templates tagged `rlvr-box=true` and on the instance, subnet, security group,
    network interface, volume, key pair, capacity reservation, image and snapshot the launch touches, tagging at
    launch, and passing role `rlvr-box` to EC2. Role and instance profile `rlvr-box` are new and are what the box
    itself runs as: it can read `box/` and `box-state/` in the bucket and write only `box/<name>/out/` and
    `box-state/`.
  - `lifecycle`: the bucket drops `box-state/` objects after 7 days and `box/` objects after 90 days.
  - `template test`: launch template `rlvr-box-test`, a t3.large in us-east-1d on the 11-28 box's network interface
    with an 80 GB disk, which shuts itself down after 40 min at most.
- **Dry runs under `rlvr-scheduler`** (`box_auto.sh dryrun`, the call that failed on 11-05): the 11-28 template
  `rlvr-box-cr-08927241900b5884c` at 17:10:21 EST and the test template at 17:15:28 both returned "Request would have
  succeeded, but DryRun flag is set".
- **The test.** Queue `scripts/aws/test/box_test.txt`: seven stub items of 24 steps each on two worker slots, every
  item built to fail once at step 6 and one item (`bad1`) built to fail always. Three one-time EventBridge schedules
  at +3, +8 and +18 min stood in for the 11:30Z launch and its two retries. Times EST:

  | time | what happened |
  |---|---|
  | 17:15:39 | the three schedules created (`box_auto.sh schedule test 3 8 18`) |
  | 17:18:58 | schedule 1: RunInstances by `rlvr-scheduler` succeeded, instance 1 |
  | 17:20:39 | `scripts/aws/box_boot.sh` started from user data; plan and queue fetched from S3 by 17:20:46; two workers started at 17:20:48 |
  | 17:23:46 | schedule 2: RunInstances refused, `InvalidNetworkInterface.InUse` (the box's network interface was taken by instance 1) |
  | 17:25:25 | the first two items finished ok, each after one rerun in place from its saved state; `bad1` recorded FAILED and put back once by the keeper |
  | 17:26:39 | results, queue snapshot, logs and a status file were in S3 |
  | 17:27:52 | the saved states of the two running items were in S3 |
  | 17:28:06 | I terminated instance 1 with both running items at step 12 |
  | 17:34:26 | schedule 3: RunInstances succeeded, instance 2 |
  | 17:36:11 | instance 2 restored the queue snapshot ("restored: 2 done, 2 running, 3 queued") and both saved states |
  | 17:43:21 | the last items finished ok and the queue was empty; status written at 17:43:53; instance 2 shut itself down |

- **Result of the test.** `done.txt` ended with 6 ok and 2 FAILED (both `bad1`, as built). The two interrupted items
  hold steps 1-24 in order, 9 written by instance 1 and 15 by instance 2 (the three steps after the last saved state
  were run again). `scripts/aws/box_s3.sh pull rlvr-box-test boxtest calibration/box_test` brought logs and results
  back to Engaging (the six `train_log.jsonl` are committed with this entry as the record of the test).
  `box_auto.sh unschedule test` deleted the three schedules; no schedule, instance or loose volume remains.
  `HOURS=2 box_auto.sh show test` lists the three scheduler calls (ok, refused, ok) and both dry runs from CloudTrail.
- **What the test shows for each point of the INBOX note.** Launch: an AWS-side schedule under the fixed role starts
  the instance and a later schedule starts another after the first is gone. No clash: while an instance holds the
  network interface, any further launch from the template is refused, whoever makes it. Start of work: the box
  fetched code, plan and queue from S3 with its own role and started workers with no ssh from anyone. Results:
  finished runs, logs and saved states left the box by themselves, and a replacement instance carried on from them.
- **What the test does not show.** The stub items load no model, the instance had no GPU and built no venv, and the
  template's capacity-block settings were only dry-run. The first real boot through the venv build, the checkpoint
  fetch and a GPU worker will be on a paid block.
- **Charge.** Two t3.large instances for 1183 s in total and two 80 GB gp3 volumes for the same time: $0.03 by
  the list prices ($0.0832/h and $0.08 per GB-month), plus S3 requests and a few MB of storage (under one cent).
  Running total of AWS charges made by the agent, outside the blocks the maintainer bought: about $0.03.
- **Left in place.** The test template, both roles, the bucket rules, and the test's output under
  `box/rlvr-box-test/out/` in the bucket (for the laptop's audit). Before any re-test run
  `box_s3.sh clear rlvr-box-test`; otherwise a new instance restores the finished snapshot and shuts down.
- **Not done yet.** The self-running template version, staged inputs and schedules for the 11-28 block (and for
  11-25 and 12-09), the queues, and `docs/research/box_1128.md`.
- **Box: three A runs ran out of GPU memory at `vllm_gpu_memory` 0.35.** r85 control seed 0 (about step 271),
  `cos467` seed 0 (about step 420) and control seed 1 (about step 216) each failed on a 3.7 GB allocation that was
  20-140 MB short. Every A run on the H100 sits at 78.5-79.2 GB of the card's 79.2 GB. At 17:47 I added
  `calibration/research_r85/a007_{ctl,lr1.3e-6,ent15,cos467}_h100m30`: the same arms with `vllm_gpu_memory` 0.30,
  which leaves the trainer about 4 GB more. It changes speed only, so an arm's `_h100` and `_h100m30` seeds are read
  together. Every A item that had not started was renamed to these in the box's queues, and the three failed runs'
  saved states were moved to the new directories (a resume checks no config and takes its logs from the state
  directory). The first of them, `cos467` seed 0, started at 17:52 and printed "resumed from ... at step 407".
  Control seeds 0 and 1 resume at steps 261 and 200 when GPUs 6 and 1 are free. `lr1.3e-6` seed 1 (GPU 0, started
  17:28) still runs at 0.35.
- **D step times on the H100** (from five finished r86 runs). Two runs on one GPU: 1.88-2.53 s per step over steps
  1-300, 1.81-2.04 s over steps 600-2000, 1.52-1.63 s after step 6000, so an 11,441-step run takes 5.2-5.6 h and a
  14,180-step run 6.2-6.8 h. One run alone on a GPU: 1.14-1.15 s per step late in the run, so about 3.65 h and
  4.5 h. This answers the 17:07 entry's open point: a single 11,441-step run fits the slots after the A runs.
- **First finished box results (final dev; nothing can be read from them yet).**
  - r86 `d004_ctl_s11441`: 0.2700 (seed 0), 0.2467 (seed 1), 0.2317 (seed 2), 0.2400 (seed 4). Seed 3 failed early and is in the overflow queue.
  - r86 `muon2676m_s11441` seed 0: 0.2550, -0.0150 against the control's seed 0.
  - r85 `a007_lr1.3e-6_h100` seed 0: 0.4367. The control's seed 0 is the run that resumes at step 261.
- **r101** (B200, not yet gated). `d007rl_ctl_b200` seed 0 finished: final dev 0.2483 after
  18,215 steps. The native arm's seed 0 is next in the queue; workers 29554920 and 29580582 are
  pending on `mit_normal_gpu`.
- **Box at 17:55.** All eight GPUs busy, 10 runs, 10 FAILED lines in `queue/done.txt` (8 at 17:06, plus `cos467`
  seed 0 and control seed 1).

## 2026-11-18 18:23 EST: AWS: self-running launch templates for all three blocks pass the dry run; the 0.30 sampler share costs 1-3 % of sampling time

- **Launch templates.** `scripts/aws/box_auto.sh template` ran for the three reservations at 18:20-18:21 (code at
  1206dbb). Each template's default version now carries the instance profile `rlvr-box` and the self-running stage
  (at boot it fetches `box/<template name>/in/code.tgz` from the bucket and starts `scripts/aws/box_boot.sh`). Each
  passed `box_auto.sh dryrun`, which is the real `RunInstances` call with `--dry-run` under the `rlvr-scheduler` role.

  | block | reservation | template | default version | instance | zone and network interface |
  |---|---|---|---|---|---|
  | 11-25 | `cr-03ab208e99ca6ca06` | `rlvr-box-cr-03ab208e99ca6ca06` (lt-0b73f76d15a2a42d2) | 1 (new) | p5.48xlarge | us-east-1f, eni-00b53b4c24ca888cd |
  | 11-28 | `cr-08927241900b5884c` | `rlvr-box-cr-08927241900b5884c` (lt-0c1649ffec152a168) | 3 (was 2) | p5en.48xlarge | us-east-1d, eni-03f79eb9d98654069 |
  | 12-09 | `cr-0813172fa29139b78` | `rlvr-box-cr-0813172fa29139b78` (lt-01af37ff7f391e126) | 1 (new) | p5.48xlarge | us-east-1f, eni-00b53b4c24ca888cd |

  - The 11-28 template's version 2 (no self-running stage) is kept. `aws ec2 modify-launch-template
    --launch-template-name rlvr-box-cr-08927241900b5884c --default-version 2` goes back to it.
  - No schedule exists yet, so nothing launches by itself. With nothing staged in the bucket, a box started from
    these versions comes up as the 11-18 one did: the fetch gives up after about 3 min and the venv is still built.
  - No charge: templates and dry runs are free. The running total stays at about $0.03.
  - Next: the 11-25 queues and plan (first what the 11-18 box leaves unfinished), `box_s3.sh stage`, `home` and
    `check`, then the schedules, then `docs/research/box_1128.md`.
- **The 0.30 sampler share on the box.** All three A runs that ran out of memory at 0.35 have resumed under the
  `_h100m30` configs. By 18:17 `cos467` seed 0 (step 433) and control seed 0 (step 274) were past the steps where
  they had failed (about 420 and 271). Seconds per step, the last 30 steps at 0.35 against the first steps at 0.30:

  | run | resumed at step | sampling at 0.35 | sampling at 0.30 | change | update at 0.35 | update at 0.30 | steps at 0.30 |
  |---|---|---|---|---|---|---|---|
  | `cos467` seed 0 | 407 | 26.8 | 27.3 | +1.9 % | 20.2 | 20.7 | 26 |
  | control seed 0 | 261 | 27.0 | 27.3 | +1.1 % | 17.5 | 18.0 | 13 |
  | control seed 1 | 200 | 27.6 | 28.3 | +2.5 % | 21.7 | 19.4 | 12 |

  Sampling is 1.1 to 2.5 % slower. The update time moves both ways and shows no cost. `nvidia-smi` still
  shows the cards nearly full (the trainer's allocator takes what the sampler leaves), so the test of the change is
  that no further A run fails for memory. `lr1.3e-6` seed 1 on GPU 0 still runs at 0.35.
- **Two more finished box results (final dev; one seed each, nothing can be read from them yet).**
  - r86 `muon1338m_s14180` seed 0: 0.2550 after 14,180 steps, -0.0150 against the control's seed 0
    (0.2700 after 11,441 steps). `muon2676m_s11441` seed 0 was 0.2550 (-0.0150).
  - r85 `a007_ent15_h100` seed 0: 0.4383 (`a007_lr1.3e-6_h100` seed 0 was 0.4367). The A control's seed 0 is
    due about 21:00.
- **Box at 18:22.** All eight GPUs busy, 10 runs, still 10 FAILED lines in `queue/done.txt`.

## 2026-11-18 18:50 EST: box: three more seeds finished; `muon1338m_s14180` is -0.0083 against the control over 3 paired seeds so far

- **r86 (D, RL only from saved checkpoints, final dev).** `muon1338m_s14180` (14,180 steps) against
  `d004_ctl_s11441` (11,441 steps), the seeds both arms have finished:

  | seed | control | `muon1338m_s14180` | difference |
  |---|---|---|---|
  | 0 | 0.2700 | 0.2550 | -0.0150 |
  | 1 | 0.2467 | 0.2333 | -0.0133 |
  | 4 | 0.2400 | 0.2433 | +0.0033 |

  Mean difference -0.0083 over 3 seeds. Seeds 2 and 3 of the Muon arm run on GPU 2 and are due about 01:00;
  the control's seed 3 failed early and waits in the box's shared queue. No verdict before the arm is complete.
- **r85 (A 007's config, 467 steps, final dev).** `a007_cos467` seed 0: 0.4167 (its steps up to 407 ran under the
  `_h100` config and the rest under `_h100m30`; the result is in the `_h100m30` directory). Seed 0 of the other
  arms: `lr1.3e-6` 0.4367, `ent15` 0.4383. The control's seed 0 is due about 21:00, so nothing is paired yet.
- **Box at 18:46.** All eight GPUs busy, 10 runs, still 10 FAILED lines. The keeper started GPU 4 (`a007_ctl_h100m30`
  seed 2, 18:26) and GPU 7 (`a007_lr1.3e-6_h100m30` seed 2, 18:37); GPU 3's worker went on to `a007_cos467_h100m30`
  seed 1 by itself. No A run has failed for memory since the move to the 0.30 share.

## 2026-11-18 19:25 EST: box: results and saved states now also leave the box by themselves (S3); `cos467` gets a third seed

- **Why.** INBOX 11-18 11:47 puts getting every result off the box first, and until now results left it only when
  I ran `box_sync.sh from`. If this session stopped overnight, the runs would finish and be lost at 06:00.
- **What I did (19:23).** Attached the instance profile `rlvr-box` to the running instance
  (`aws ec2 associate-iam-instance-profile`, free; the role can only write under `box/*/out/` and `box-state/` of
  the bucket) and started a loop on the box (`~/box_push.sh`, copy in `agent/box_push_1118.sh`, log
  `~/box_push.log`, stop with `touch ~/box_push.off`). Every 10 min it sends the r85 and r86 seeds' logs and
  results, the queue files and the worker logs to `box/box1118/out/`, with a status file. Saved states go to
  `box-state/box1118/` (expires after 7 days): track D states (1.3 GB) every hour, and every changed state,
  the 28.8 GB track A ones too, on each pass from 04:40 on. Layout as `box_boot.sh` writes it, so
  `box_s3.sh status box1118` and `box_s3.sh pull box1118 <tag> calibration/research_r85 calibration/research_r86`
  (`STATE=1` for the states) read it from Engaging or the laptop.
- **Checked.** The first pass took 15 s: 220 objects (143 MB) under `out/` and four D states (5.3 GB);
  `box_s3.sh status box1118` shows the box's running list and queues. `box_sync.sh from` stays the main route.
- **Cost.** S3 storage only: at most about 100 GB of states for 7 days, about $0.54. Agent AWS charges so
  far: about $0.03 (the self-launch test) plus this, about $0.57 at most.
- **Queue change (19:20).** Moved `a007_cos467_h100m30` seed 2 from the hold list to GPU 0's queue. GPU 0's worker
  takes it when `a007_lr1.3e-6_h100` seed 1 ends (expected about 23:25) and it should end between 05:05 and 05:40,
  inside the 05:55 deadline. Otherwise GPU 0 would have taken a single D run and sat empty for the last 1.5 h.
  `cos467` then has three seeds planned; one `muon1338m_s11441` seed fewer runs on this box.
- **H100 memory, measured 18:51** (cards hold 81,559 MiB). One A run at the 0.35 share: 79,836 MiB. One A run at
  0.30: 77,996 to 80,996 MiB over five runs. Two D runs on a GPU: 39,192 to 41,172 MiB each. So for 11-25 and
  12-09: one A run (0.30 share) or two D runs per GPU, never an A beside a D. The A runs still nearly fill the card.
- **Plan for the rest of the night** (times are estimates). GPU 6 about 21:00 and GPU 1 about 21:54 go on to
  `a007_ent15_h100m30` seeds 2 and 1; GPU 5 about 23:03 to `muon2676m_s11441` seeds 3 and 4; GPUs 4, 7 and 3 about
  00:30-00:45 to single D runs from the shared list; GPU 2 about 01:07, where only one D run may start that late
  (a pair would not finish), so I leave it one item before 01:00.

## 2026-11-18 21:20 EST: box: the track A control's seed 0 is in (0.4250); the last 0.35-share A run failed for memory and continues at 0.30

- **r85 (A 007's config, 467 steps on an H100, final dev), seed 0 of each arm:**

  | arm | final dev | against the control | kept step | hours |
  |---|---|---|---|---|
  | `a007_ctl` | 0.4250 | | 450 | 5.97 |
  | `a007_lr1.3e-6` | 0.4367 | +0.0117 | 350 | 5.83 |
  | `a007_ent15` | 0.4383 | +0.0133 | 350 | 6.05 |
  | `a007_cos467` | 0.4167 | -0.0083 | 467 | 6.09 |

  One seed each, so this says nothing yet: the differences are the size of one seed's noise. Seeds 1 and 2 of all
  four arms are running or queued for tonight. (The control's seed 0 and `cos467` seed 0 ran their first
  steps under the `_h100` config, 0.35 sampler share, and the rest under `_h100m30`, 0.30; same maths.)
- **`a007_lr1.3e-6_h100` seed 1 failed for memory (GPU 0).** It was the one run still at the 0.35 share. It ran
  out of memory at 20:47, restarted from its saved state, and ran out again at 20:57, which
  used up the runner's two tries: FAILED line 11 in the box's `queue/done.txt`. Its saved state (step 278 of 467,
  saved 20:43) is whole. At 21:16 I moved it to `a007_lr1.3e-6_h100m30`'s resume directory and queued that seed
  on GPU 7 behind its current run (free about 00:20; 189 steps left, about 2.5 h). To make room,
  `muon1338m_s11441` seed 1 went to the hold file. No run at the 0.30 share has failed for memory so far.
- **GPU 0 and GPU 6 moved on by themselves:** GPU 0 started `a007_cos467_h100m30` seed 2 at 20:57 and GPU 6
  started `a007_ent15_h100m30` seed 2 at 21:03. Both are due about 03:00.
- **Queue change at 20:52 (box).** The shared queue `box_t_1118` is now empty and GPUs 2, 3, 4 and 7 each got
  one item of their own: GPU 4 `d004_ctl_s11441` seed 3, GPU 3 `muon1338m_s11441` seed 2, GPU 2
  `muon1338m_s11441` seed 0, GPU 7 as above. Why: GPU 2 has two live workers, and both would have taken a D item
  from the shared queue when their pair ends about 01:07; a D pair needs 5.2 h or more and the box ends at 06:00,
  so neither would have finished. Alone a D run takes about 3.65 h. `muon1338m_s11441` seeds 1, 3 and 4 wait in
  `queue/hold_d2_1118.txt` on the box (not run tonight).
- **Engaging.** r101 worker 29554920 started about 21:03 on node5600-c1 (`d007rl_native_b200` seed 0).

## 2026-11-18 22:59 EST: box: `muon2676m_s11441` seeds 1 and 2 are in (mean -0.0233 against the control over 3 seeds); the A control's seed 1 is 0.4000

- **r86 (track D, RL only from the saved checkpoints, final dev), paired by seed against `d004_ctl_s11441`:**

  | arm | seed | final dev | control | difference |
  |---|---|---|---|---|
  | `muon1338m_s14180` | 0 | 0.2550 | 0.2700 | -0.0150 |
  | `muon1338m_s14180` | 1 | 0.2333 | 0.2467 | -0.0133 |
  | `muon1338m_s14180` | 4 | 0.2433 | 0.2400 | +0.0033 |
  | `muon2676m_s11441` | 0 | 0.2550 | 0.2700 | -0.0150 |
  | `muon2676m_s11441` | 1 | 0.2133 | 0.2467 | -0.0333 |
  | `muon2676m_s11441` | 2 | 0.2100 | 0.2317 | -0.0217 |

  Means so far: `muon2676m_s11441` -0.0233 over 3 seeds, `muon1338m_s14180` -0.0083 over
  3 seeds. Neither arm is ahead of the control. Two seeds of each are still to come: `muon1338m_s14180`
  seeds 2 and 3 about 00:20, `muon2676m_s11441` seeds 3 and 4 about 04:00 (GPU 5 started them by itself at 22:41
  and 22:46). The control's seed 3, which failed early this afternoon, runs on GPU 4 from about 00:15.
- **r85 (track A), the control's seed 1: 0.4000** (kept step 300; the last evaluation of the run was
  0.2317, so the run fell apart late and the kept checkpoint is what counts). Seed 0 was 0.4250. No arm has
  a seed 1 yet. GPU 1 started `a007_ent15_h100m30` seed 1 at 21:51.
- **Queue change at 22:41 (box).** The run-time projection now puts the end of GPU 2's pair (`muon1338m_s14180` seeds
  2 and 3) at about 00:20, not 01:07. So GPU 2 gets a pair again: `muon1338m_s11441` seeds 0 and 1, both workers
  taking one each. A pair of 11,441-step runs took 5.2 to 5.3 h on GPU 5, which puts their end about 05:40,
  20 min before AWS may terminate the box (06:00). If they do not finish, their saved states are in S3 and they
  can be finished on the 11-25 box. `muon1338m_s11441` seeds 3 and 4 stay in the hold file.
- **Engaging.** The second r101 worker, 29580582, started at 22:58 on node5800-c1 with `d007rl_ctl_b200` seed 1.

## 2026-11-19 00:14 EST: box: the track A control has three seeds (mean 0.4217); GPUs that free up after the keeper's 01:55 cutoff will start track A seeds 3 and 4, to be finished on the 11-25 box

- **r85 (track A, final dev; difference from the control's same seed in brackets):**

  | seed | `a007_ctl` (kept step) | `a007_lr1.3e-6` | `a007_ent15` | `a007_cos467` |
  |---|---|---|---|---|
  | 0 | 0.4250 (step 450) | 0.4367 (+0.0117) | 0.4383 (+0.0133) | 0.4167 (-0.0083) |
  | 1 | 0.4000 (step 300) | running | running | running |
  | 2 | 0.4400 (step 467) | 0.4133 (-0.0267) | running | running |

  - New: `a007_lr1.3e-6_h100m30` seed 2 is 0.4133 (kept step 250; its last evaluation was 0.0817, so the run
    fell apart after the kept checkpoint; 5.19 h). The control's seed 2 is 0.4400 (kept step 467, the last one;
    5.73 h).
  - The control's three seeds are 0.4250, 0.4000 and 0.4400: mean 0.4217, range 0.0400. `lr1.3e-6` is +0.0117 and
    -0.0267 on its two paired seeds (mean -0.0075). No arm has enough seeds to read yet.
  - GPU 7's worker finished `lr1.3e-6` seed 2 at 23:49 and went on to the rest of `a007_lr1.3e-6_h100m30` seed 1 (from step
    278 of 467). GPU 4's worker finished the control's seed 2 at 00:11 and started `d004_ctl_s11441` seed 3 (alone on
    its GPU; 40,819 MiB in use at 00:13).
- **The idle tails.** The tail keeper starts nothing once fewer than 240 min remain (01:55), and a worker that
  finishes an item after that exits. By the run times so far, seven GPUs would then sit empty until the box ends
  (06:00): GPU 7 from about 02:00, GPUs 0 and 6 from 02:10 to 03:10, GPU 1 from 03:00 to 03:55, GPUs 4, 5 and 3 from
  about 04:00. That is roughly 17 to 19 GPU-hours. A track A run needs 5.2 to 6.1 h here, so none can finish, but one
  can be continued elsewhere: `state.pt` holds the weights, the optimizer state and the kept-best copy (`"kept"` in
  the saved state, `rlvr_speedrun/grpo.py`), and a run continued from a state moved to another directory already
  worked (LOG 11-18 17:55: `cos467` seed 0 resumed at step 407 and finished). INBOX 11:47 puts keeping all eight
  GPUs busy first, so the tails run the next track A seeds as far as they get and the 11-25 box (same card, same
  configs) finishes them.
- **What runs on the box for this (started 00:07):**
  - `queue/box_tail_1118.txt`, taken out of `hold_a_1118.txt`, in this order: `a007_ctl_h100m30` 3, `a007_ent15_h100m30` 3,
    `a007_lr1.3e-6_h100m30` 3, `a007_cos467_h100m30` 3, then seed 4 of the control, `ent15` and `lr1.3e-6`. The hold file keeps
    `a007_cos467_h100m30` 4.
  - `~/tail_fill.sh` (copy in `agent/box_tail_fill_1118.sh`; log `~/tail_fill.log`; stop with `touch ~/tail_fill.off`).
    Each minute it starts one worker on a GPU that has been empty for two passes (no worker, no running file, no
    compute process, under 2 GB in use), reading that queue, with a saved state every 20 min. So a track A run
    never starts beside another run. It leaves a GPU to the keeper while the keeper can still start something
    there, starts nothing with under 75 min left (04:40), and starts at most two workers per GPU.
  - `~/box_push_res.sh` (copy in `agent/box_push_res_1118.sh`; log `~/box_push_res.log`): results, queue files and logs to
    `box/box1118/out/` every 5 min, beside the existing push loop (every 10 min, plus the saved states).
- **Upload speed measured.** One 28.77 GB track A state went from the box to the bucket in 5 s and 5.7 s (two
  tries; size checked in the listing, then removed). So the push loop's last stretch (every changed state on each
  pass from 04:40) is short even with seven track A states, and the saved states of the tail runs will be at most
  about 30 min old when the box ends.
- **After the box ends:** copy the tail runs' states inside the bucket from `box-state/box1118/` to the 11-25 box's
  prefix and remove the originals, and put those seven items first in the 11-25 queues. `box_boot.sh` brings back
  every saved state under its own prefix at boot and the queue items continue from them.
- **AWS cost of this (estimate).** Storage: seven states of 28.77 GB for 7 days, about $1.08 (the bucket keeps
  no old versions, and `box-state/` objects expire after 7 days). Upload requests: about $0.017 per state
  upload if the parts are 8 MiB, so at most about $0.63 for the two test uploads and five uploads of each state.
  New: about $1.72 at most. **Running total of AWS charges made by the agent: about $2.29 at most** (the
  self-launch test $0.03, the 11-18 box's results and D states $0.54, this).
- Engaging: D 007 seed 4 (29355448) has run 1 h 29 min on node4802; r101's two workers are running.

## 2026-11-19 00:36 EST: box: `muon1338m_s14180` has all five seeds (-0.0025 against the control over 4 paired seeds); `cos467` seed 1 is 0.3933

- **r86 (track D, RL only from the saved checkpoints, final dev), paired by seed against `d004_ctl_s11441`:**

  | arm | seed | final dev | control | difference |
  |---|---|---|---|---|
  | `muon1338m_s14180` | 0 | 0.2550 | 0.2700 | -0.0150 |
  | `muon1338m_s14180` | 1 | 0.2333 | 0.2467 | -0.0133 |
  | `muon1338m_s14180` | 2 | 0.2467 | 0.2317 | +0.0150 |
  | `muon1338m_s14180` | 3 | 0.2317 | running | |
  | `muon1338m_s14180` | 4 | 0.2433 | 0.2400 | +0.0033 |
  | `muon2676m_s11441` | 0 | 0.2550 | 0.2700 | -0.0150 |
  | `muon2676m_s11441` | 1 | 0.2133 | 0.2467 | -0.0333 |
  | `muon2676m_s11441` | 2 | 0.2100 | 0.2317 | -0.0217 |

  - `muon1338m_s14180` (Muon pretraining stopped at 1.338B tokens, the time moved to RL: 14,180 steps against 11,441):
    all five seeds are in, mean 0.2420 (range 0.2317 to 0.2550). Against the control on the four
    seeds that have a pair: -0.0150, -0.0133, +0.0150, +0.0033; mean -0.0025. The control's four seeds have
    mean 0.2471 (range 0.2317 to 0.2700). So the split gives the same dev score as the control within the
    seed noise: no gain from it on the H100. The control's seed 3 (GPU 4, started 00:11) gives the fifth pair
    at about 03:50.
  - `muon2676m_s11441` is unchanged: -0.0233 over 3 seeds; seeds 3 and 4 run on GPU 5 until about 04:00.
- **r85 (track A, final dev; difference from the control's same seed in brackets):**

  | seed | `a007_ctl` (kept step) | `a007_lr1.3e-6` | `a007_ent15` | `a007_cos467` |
  |---|---|---|---|---|
  | 0 | 0.4250 (step 450) | 0.4367 (+0.0117) | 0.4383 (+0.0133) | 0.4167 (-0.0083) |
  | 1 | 0.4000 (step 300) | running | running | 0.3933 (-0.0067) |
  | 2 | 0.4400 (step 467) | 0.4133 (-0.0267) | running | running |

  New: `a007_cos467_h100m30` seed 1 is 0.3933 (kept step 450, last evaluation 0.3867, 5.70 h).
  `cos467` is -0.0083 and -0.0067 on its two paired seeds (mean -0.0075); its seed 2 runs on GPU 0.
- **Hand-overs on the box, all by the workers themselves:** GPU 2's two workers finished `muon1338m_s14180` seeds 2
  and 3 at 00:18 and 00:19 and started `muon1338m_s11441` seeds 0 and 1 as a pair (80,033 MiB in use at 00:35). A
  pair of 11,441-step runs took 5.2 to 5.6 h here, so they end between 05:30 and 05:55, close to the end of the
  box (06:00); their saved states go to S3 every 10 min from 04:40 in case they do not make it. GPU 3's worker
  finished `cos467` seed 1 at 00:28 and started `muon1338m_s11441` seed 2 alone (40,621 MiB; about 3.65 h, so until
  about 04:10). All eight GPUs are busy; the box's `queue/done.txt` still has 11 FAILED lines (none new).

## 2026-11-19 02:05 EST: INBOX 01:32 answered. r101's native arm ends at 0.3033 dev after 34,893 steps (control 0.2483 after 18,215); r102, its H200 promotion screen, is queued; the box's tail runs the native sampler

- **INBOX 01:32 (the laptop session's critique of the 11-18 box plan), read at 01:34. I agree with its main points**:
  the r85 arms were too small for three seeds, and the plan should have been redone on 11-17 from the speed results.
  What changed within the hour is below; the point-by-point reply is the last item.
- **r101 (B200, RL only from D 007's checkpoints, both arms 17,400 s, one seed each so far; not yet gated).** The
  native arm (same-process sampler, `reuse_sampler_allowed`, `share_group_prompt`, 65,536-token micro-batches)
  finished its seed 0 at 01:56:

  | arm | seed | RL steps in 17,400 s | final dev | mean of the last 5 evals |
  |---|---|---|---|---|
  | `d007rl_native_b200` | 0 | 34,893 | 0.3033 | 0.2920 |
  | `d007rl_ctl_b200` | 0 | 18,215 | 0.2483 | 0.2463 |

  Dev by step (the control's seed 1 is still running, at step 12,000):

  | step | native s0 | control s0 | control s1 |
  |---|---|---|---|
  | 2,000 | 0.1450 | 0.1317 | 0.1433 |
  | 4,000 | 0.1733 | 0.1533 | 0.1733 |
  | 6,000 | 0.1950 | 0.1933 | 0.1767 |
  | 8,000 | 0.2250 | 0.2000 | 0.1883 |
  | 10,000 | 0.2350 | 0.2267 | 0.2000 |
  | 12,000 | 0.2450 | 0.2250 | 0.2150 |
  | 14,000 | 0.2483 | 0.2450 |  |
  | 16,000 | 0.2550 | 0.2533 |  |
  | 18,000 | 0.2633 | 0.2517 |  |

  The native arm alone after that: 20,000 0.2650, 22,000 0.2583, 24,000 0.2700, 26,000 0.2617, 28,000 0.2683, 30,000 0.2900, 32,000 0.2917, 34,000 0.2883; final (34,893) 0.3033.
  - **1.92x the steps in the same time, and +0.0550 dev.** At equal steps the native arm is level with or a little above
    the control (native minus control s0 at the nine steps above: mean +0.0117, range +0.0017 to +0.0250; the
    control's own two seeds differ by up to 0.0267 at these steps), so most of the gain comes from the extra
    steps, and dev was still rising at 30,000 steps. This is the first run that turns the speed work
    into score. It is one seed on a B200; the control's seed 1 and the native arm's seed 1 (worker 29618971,
    submitted 01:58) follow.
- **r102, the H200 promotion screen for a D record with the speed stack (queued 01:52).** The gating screen has to
  run on the H200 and that queue is slow, so it starts now and does not wait for more B200 seeds.
  `calibration/research_r102/d007_speed_h200` is D 007 end to end with every same-maths speed change: pretraining
  with Muon on the fused path for 46 min in place of 56 (r92), then RL for 18,000 s in place of 17,400 with r101's
  native settings; the same 20,760 s in total. It is r99's arm plus `reuse_sampler_allowed`. Seeds 0-2 are in
  `queue/screens_r92.txt` (r92's own arm is dropped: r102 contains its change), read by H200 preemptable workers
  29450693/94/702, which I released at 01:52. B200 preemptable workers 29497701/02/05 (r97, r99) are held so that
  no more than four preemptable GPUs run (those three and D 007's seed 4).
  - Read against D's line, 0.259 (D 006 holds). D 007's four finished seeds have kept dev 0.2450 / 0.2517 / 0.2500 / 0.2550
    (mean 0.2504), so the line becomes about 0.280 if D 007 lands; I read r102 against both. If
    the three-seed rule is met, a five-seed record attempt starts (r102's config without `no_save`).
  - Expected effect, written before it runs: +0.03 to +0.05 dev over D 007 (r101: +0.055 on one B200 seed; r78:
    +0.032 for a third more steps). D 007's four seeds have a dev sd of 0.0042 (r77's three had 0.0278, on nodes
    of different speed), so three seeds resolve an effect of that size only if the spread stays near D 007's.
  - Risks: this is the native sampler's first run through pretraining and RL in one job (r99 was preempted
    before RL each time), and the workers are preemptable.
- **The box's tail queue changed at 01:44.** `queue/box_tail_1118.txt` held seeds 3 and 4 of the four track A arms,
  none of which could finish. They moved to `queue/hold_a_1118.txt` (not run). The tail now holds two new r86
  arms, `muon1338m_native_s14180` (seeds 0-4) and `muon1338m_native_s11441` (seeds 0-2): tonight's vLLM arms with
  r101's five native settings, same checkpoints and step caps, one run alone on each GPU as it frees (first
  start about 02:22 on GPU 7). Expected: the same dev as the vLLM arm of the same seed (it is the same maths, so
  this is a check and not a score arm), and the step rate and memory of the native sampler on an H100, which
  the 11-25 queue needs for its step caps and packing. A 14,180-step run should take about 2.5 h if the B200's
  1.9x holds, so the first three or four finish; later ones end as partials with their states in S3.
  - Why not the A seeds: the control's three seeds are 0.4250 / 0.4000 / 0.4400 (sd 0.0202), so a paired
    difference has an sd of about 0.029 per seed, and seeing +-0.015 at two standard errors takes about
    15 pairs at 5.7 GPU-hours a run: more than a box day for one arm.
- **r81 against r86: I believe r86.**
  - r81's margin was never firm. r77 (H200, timed): 0.253 / 0.218 / 0.273, mean 0.2483, sd 0.0278. r81:
    0.262 / 0.268 / 0.275, mean 0.2683. Difference +0.0200 with a standard error of 0.0165
    (three unpaired seeds each; r77's seeds ran 11,074 to 16,055 steps on different nodes).
  - The same config's next four seeds, D 007's, have kept dev mean 0.2504: r77's level. Over all seven
    seeds of the config the mean is 0.2581, +0.0098 over r77 with a standard error of 0.0166.
  - r86 says the same with paired seeds and equal step counts: Muon for 56 min plus 24 % more RL steps against
    AdamW for 112 min is -0.0025 (standard error 0.0072, 4 pairs), and Muon for the full 112 min
    at equal steps is -0.0233 (3 pairs).
  - So the Muon split is about neutral on dev: the shorter pretraining costs about what the extra RL steps give
    back. D 007's dev gain over D 006 (0.229) is mostly the batched sampler's extra steps (r77 alone is 0.248).
    D 007 as a record rests on its test ladder, 0.2606 / 0.2538 / 0.2646 / 0.2726 (mean 0.2629) against D 006's
    0.2444 +- 0.0158, with seed 4 running; that comparison is unchanged.
  - One caveat on r86: its 14,180-step cap came from r77 seed 0's rate, and D 007's seeds ran 17,359 to
    19,735 steps on the H200, so r86's Muon arm is D 007 as a slow node would run it.
  - My error: I promoted r81 on three seeds that cleared the line without asking what the control's spread
    allowed. The three-seed rule permits that; the lesson is to write the expected effect and the control's
    noise next to each arm before it runs, as the note says.
- **Track A's clock (the note's last question), from A 007's record logs.** The five seeds reached step 467 after
  15,708 / 15,087 / 14,737 / 15,573 / 14,994 s of a 17,400 s budget: the step cap ended every run with 1,692 to
  2,663 s unused. A step takes 32.2 s on average: 17.7 s sampling (55 %) and 14.4 s
  update. At each seed's own overall rate (evals included) 17,400 s would hold 517 to 551 steps, so A has
  room for about 50 more steps before any speed work (less a margin for slow nodes), and both halves are worth timing: a short H200 run with `time_update_parts` is next for A.
- **Reply to the four suggestions.**
  1. Speed into score first: yes. r102 is that run on the H200, queued tonight. The 11-25 box (H100, screens
     only) gives the speed stack seeds against its control at step caps taken from tonight's native runs.
  2. Fewer arms, more seeds: yes. From now each arm's LOG line carries the expected effect and its control's seed
     sd before it gets GPUs, and an arm inside the noise gets the seeds to resolve it or does not run. No more
     small moves around A 007; `hold_a_1118` stays unrun.
  3. The rest goes to changes that could be large: U17 and U19, and the settings that nearly twice the steps may
     have moved (the pretraining/RL split, prompts per step, the lr schedule's length).
  4. Each box queue is re-planned the day before from the results in hand: 11-24 for 11-25, 11-27 for 11-28,
     12-08 for 12-09.
- **Box at 02:05:** all eight GPUs busy (ten runs), 11 FAILED lines (none new), sync fine at 02:05. No track A state has gone
  to S3 (the box's push log at 02:00 shows none) and none will unless an A run is still going at 04:40 (the last ends about 03:40), so the $1.72 logged at
  00:14 is not expected to be spent; I correct the total when the box's final S3 contents are logged.

## 2026-11-19 02:43 EST: box: the native sampler runs at 0.80 s/step on the H100 after its first 400 steps (1.64x the B200's time), so the 11,441-step native runs go first; `a007_lr1.3e-6` seed 1 is 0.4367

- **The first native-sampler run on an H100** (`muon1338m_native_s14180` seed 0, alone on GPU 7, started 02:27:54;
  step 621 at wall 722 s when read at 02:40). Seconds per RL step by step range, against the B200 run of r101
  (`d007rl_native_b200` seed 0: the same five native settings and model size, another checkpoint):

  | steps | H100 s/step | B200 s/step | H100 / B200 |
  |---|---|---|---|
  | 11 to 80 | 1.921 | 1.094 | 1.76 |
  | 80 to 201 | 1.389 | 0.655 | 2.12 |
  | 201 to 401 | 1.014 | 0.515 | 1.97 |
  | 401 to 500 | 0.839 | 0.479 | 1.75 |
  | 421 to 621 | 0.800 | 0.487 | 1.64 |

  - Over the last 200 steps the H100 spends 0.29 s sampling and 0.46 s in the update per step. The slow start is
    the same on both cards (the decode graphs are captured as new batch and key-length buckets appear).
  - Whole-GPU memory: 69.5 GB at step 1, 75.0 GB at step 79, 75.1 GB at step 499, then 48.2 GB at step 511
    (after the first dev evaluation at step 500) and 51.7 GB at step 621. The B200 run: 69.1, 75.1, 75.1, 42.1 and
    46.3 GB at the same steps. So on an 80 GB card a native D run needs its GPU to itself for its first 500 steps,
    unless that early use can be cut (the packing test below). One run already keeps the GPU at 94 % use, so a
    second run beside it may gain little.
- **Correction to the 02:05 entry.** It said a 14,180-step native run "should take about 2.5 h if the B200's 1.9x
  holds"; that was the B200's rate, not a measurement on this card. A flat 0.80 s/step would give 3.21 h, but the
  B200 run slowed after its first 600 steps (0.583 s/step from step 621 to 14,180 against 0.487 over steps 421 to 621).
  With the B200 run's remaining time scaled by 1.64, the run on GPU 7 needs 3.81 h and would end about 06:16, after
  the box goes at 06:00: it ends unfinished near step 12,712 at 05:55 unless the ratio falls to 1.48 (it fell from
  2.12 to 1.64 over the first 621 steps). An 11,441-step run needs 3.14 h on the same reckoning, so it finishes by 05:55
  only if started by about 02:46.
- **Tail queue reordered at 02:31** (on the box, under the queue lock): the three `muon1338m_native_s11441` items
  (seeds 2, 0, 1) now come first, then `muon1338m_native_s14180` seeds 1 to 4. GPUs 0 and 6 free up at about 02:42
  and 02:48, so their two 11,441-step runs are due at about the deadline and may or may not finish; the workers do
  not stop a run at 05:55, the box's end at 06:00 does. Runs that start from 03:40 on (GPUs 1, 4, 5, 3) end
  unfinished. Every unfinished run's state goes to S3, and its dev curve at equal steps is still read against the
  vLLM arm of the same seed.
- **Planned for GPU 1 (free about 03:40, too late for a run to finish): a packing test for 11-25.** The fall from
  75 to 48 GB at the first evaluation matches `_ensure_capacity` in `fast_rollout.py`, which drops the decode graphs
  and calls `torch.cuda.empty_cache()` when a larger batch arrives, so part of the 75 GB is PyTorch's cache and not
  need. Once the native run that starts there has passed step 500 I start a second native run beside it by hand:
  whether two fit in 80 GB, and the pair's step rate against one run alone.
- **r85, `a007_lr1.3e-6` (track A, 467 steps, kept dev), now three seeds paired with the control:**

  | seed | `a007_ctl` | `a007_lr1.3e-6` | difference |
  |---|---|---|---|
  | 0 | 0.4250 | 0.4367 (kept step 350) | +0.0117 |
  | 1 | 0.4000 | 0.4367 (kept step 350) | +0.0367 |
  | 2 | 0.4400 | 0.4133 (kept step 250) | -0.0267 |

  Mean difference +0.0072 (sd 0.0319, SE 0.0184). Seed 1's last evaluation was 0.4250. This is level
  with the control within the noise, as the 02:05 entry expected; the arm gets no more seeds.
- **Box at 02:43:** all eight GPUs busy (ten runs), 11 FAILED lines (none new), sync fine at 02:40.

## 2026-11-19 03:04 EST: box: the native run on the H100 is now as fast as the B200's (0.62 s/step) and all three native runs should finish by about 05:00; `a007_ent15` seed 2 is 0.4533, `a007_cos467` seed 2 is 0.4033

- **Correction to the 02:43 entry: the H100's step time kept falling, and the native runs finish.** The run on GPU 7
  (`muon1338m_native_s14180` seed 0) was at step 2,619 after 2078 s when read at 03:02. Seconds per RL step
  against the B200 run of r101 at the same steps:

  | steps | H100 s/step | B200 s/step | H100 / B200 |
  |---|---|---|---|
  | 11 to 80 | 1.921 | 1.094 | 1.76 |
  | 80 to 201 | 1.389 | 0.655 | 2.12 |
  | 201 to 401 | 1.014 | 0.515 | 1.97 |
  | 401 to 621 | 0.810 | 0.486 | 1.67 |
  | 621 to 1500 | 0.711 | 0.623 | 1.14 |
  | 1500 to 2500 | 0.656 | 0.635 | 1.03 |
  | 2419 to 2619 | 0.625 | 0.628 | 1.00 |

  - The 02:43 entry scaled the B200 run's remaining time by 1.64, the ratio at steps 421 to 621. That ratio was still
    falling and is now 1.00: only the first 600 steps are slower on the H100. Over the last 200 steps the run spends
    0.19 s sampling and 0.39 s in the update per step, and the GPU holds 51.7 GB.
  - With the B200's remaining time scaled by 1.00, the GPU 7 run needs 2.42 h in all and ends about 04:52. On the
    same reckoning an 11,441-step run needs 2.01 h, against 3.65 to 3.8 h for a vLLM run alone on an H100 and 5.2 to 5.6 h for a
    vLLM pair (two runs). So one native run per GPU gives 1.29 to 1.39 times the runs per GPU-hour of the vLLM pair.
  - **The filler started the two 11,441-step runs** when the track A runs ended: `muon1338m_native_s11441` seed 2 on
    GPU 0 (its clock started at 02:55) and seed 0 on GPU 6 (02:58). They should end about 04:55 and 04:58. Their first
    steps were faster than GPU 7's: 1.74 and 1.77 s/step over steps 11 to 80 against 1.92, and 0.82 against
    1.01 over steps 201 to 401 on GPU 0 (GPU 6 had not got that far).
  - An 11,441-step run finishes by 05:55 if it starts by about 03:54, a 14,180-step run if it starts by about
    03:30. So `muon1338m_native_s11441` seed 1 (GPU 1, free about 03:40) can finish; the four
    `muon1338m_native_s14180` items that follow (GPUs 4, 5, 3 from about 03:51) cannot: a start at 03:55 reaches
    about step 11,400 by 05:55, a start at 04:15 about step 9,200. They are read at equal steps against the
    finished vLLM seeds.
- **Dev at equal steps, native against vLLM (seed 0 of the 14,180-step arms, so far):**

  | step | native | vLLM | difference |
  |---|---|---|---|
  | 500 | 0.0817 | 0.0883 | -0.0066 |
  | 1000 | 0.1083 | 0.1067 | +0.0016 |
  | 1500 | 0.1433 | 0.1283 | +0.0150 |
  | 2000 | 0.1333 | 0.1550 | -0.0217 |
  | 2500 | 0.1500 | 0.1600 | -0.0100 |

  Mean difference -0.0043 over 5 evaluations of one seed; too few to read. The full table comes with the finished
  runs.
- **The packing test moves from GPU 1 to GPU 4.** GPU 1's run can now finish alone, and a second run beside it would
  stop that. GPU 4's run (free about 03:51 to 04:00) cannot finish in any case, so the second native run goes beside
  it once it has passed step 500. After 04:40 the filler stops; GPUs 7, 0 and 6 are then free from about
  04:52 with an hour left, enough for a second packing test (two native runs started together on an empty GPU).
  I decide that at 04:40 from the first test.
- **r85 (track A, 467 steps, kept dev; difference from `a007_ctl` of the same seed in brackets).** Control: 0.4250,
  0.4000, 0.4400.

  | arm | seed 0 | seed 1 | seed 2 | mean difference |
  |---|---|---|---|---|
  | `a007_lr1.3e-6` | 0.4367 (+0.0117) | 0.4367 (+0.0367) | 0.4133 (-0.0267) | +0.0072 (sd 0.0319, SE 0.0184) |
  | `a007_ent15` | 0.4383 (+0.0133) | running | 0.4533 (+0.0133) | +0.0133 (2 seeds) |
  | `a007_cos467` | 0.4167 (-0.0083) | 0.3933 (-0.0067) | 0.4033 (-0.0367) | -0.0172 (sd 0.0169, SE 0.0097) |

  - `a007_cos467` has its three seeds: all below the control, mean -0.0172 (SE 0.0097). Ending the cosine
    at step 467 does not help; closed.
  - `a007_ent15` seed 2 kept its last step (0.4533, the highest track A dev on this box). Two seeds at +0.0133
    each; seed 1 is due about 03:40. The paired differences of the two finished arms have an sd of 0.017 and
    0.032 per seed, so three seeds near +0.013 would not separate it from the control either.
- **Box at 03:04:** all eight GPUs busy (ten runs), 11 FAILED lines (none new), sync fine at 03:01.

## 2026-11-19 03:45 EST: box: no GPU that frees from 04:00 can finish a native run, so they run packing tests for the 11-25 box; correction to 03:04

- **Correction.** The 03:04 entry said GPU 1's next run "can now finish alone". It took `a007_ent15` seed 1 to end
  about 03:40, an old estimate I did not re-read. At 03:43 that run was at step 443 of 467 at 50.5 s a step, so it
  ends about 04:03. An 11,441-step native run needs 2.01 h, and the workers' deadline is 05:55, so it has to
  start by 03:54. The first GPU frees about 04:01. No native run started from now can finish.

  | GPU | run | step at 03:43 | s/step (last 200; A: last 20) | ends about |
  |---|---|---|---|---|
  | 4 | `d004_ctl_s11441` 3 | 10,487 of 11,441 | 1.13 | 04:01 |
  | 5 | `muon2676m_s11441` 3 | 10,778 of 11,441 | 1.48 | 03:59 |
  | 5 | `muon2676m_s11441` 4 | 10,533 of 11,441 | 1.51 | 04:06 |
  | 1 | `a007_ent15_h100m30` 1 | 443 of 467 | 50.52 | 04:03 |
  | 3 | `muon1338m_s11441` 2 | 9,251 of 11,441 | 1.18 | 04:26 |
  | 7 | `muon1338m_native_s14180` 0 | 6,536 of 14,180 | 0.61 | 05:01 |
  | 0 | `muon1338m_native_s11441` 2 | 3,993 of 11,441 | 0.65 | 05:03 |
  | 6 | `muon1338m_native_s11441` 0 | 3,706 of 11,441 | 0.63 | 05:04 |
  | 2 | `muon1338m_s11441` 0 | 7,125 of 11,441 | 1.62 | 05:39 |
  | 2 | `muon1338m_s11441` 1 | 7,080 of 11,441 | 1.62 | 05:41 |

  The three native runs still end by about 05:04; the vLLM pair on GPU 2 ends about 14 min before the
  deadline.
- **What the freed GPUs do instead: packing tests.** The 11-25 and 12-09 boxes are these 80 GB H100s. One native run
  at 65,536-token micro-batches holds 52 GB after its first evaluation and 75 GB before it, so tonight one runs per
  GPU. If two fit, each of those boxes does up to twice the track D runs; the same readings set the H200 packing for
  11-28 (141 GB). The tests need no finished run, only 10 to 60 min of steps.
  - New config `calibration/research_r86/muon1338m_native_mb32k_s11441`: the native 11,441-step arm with
    `micro_batch_tokens` 32768 (the only key that differs, checked by script). The update is the same sum over tokens,
    computed in smaller pieces (r94), so it is a memory setting. The vLLM arms here use 32,768 and take about 40 GB
    with vLLM's share.
  - Test 1, first GPU to free: two 32k runs started within a few minutes of each other.
  - Test 2, second GPU: one 32k run alone until its first evaluation (step 500; this gives its alone rate and
    memory), then a second beside it.
  - Test 3, third GPU: one 65k run alone until its first evaluation, then a second 65k run beside it.
  - GPU 3 (free about 04:26) and GPUs 7, 0, 6 (about 05:00): decided from the first readings.
  - What I expect: a 32k run holds 30 to 35 GB, so two fit and a GPU does 1.3 to 1.8 times the steps of one 65k run
    alone (0.61 to 0.65 s a step now, with the GPU at 52 to 71 % use in one sample at 03:37); the second 65k run fails for
    memory. A pair started together may also fail in warm-up, when each holds more than later.
  - Readings: memory per process, FAILED lines beyond the 11 so far, each run's s/step against the alone rates.
- **How.** The tail queue (rewritten 03:38) now starts `muon1338m_native_mb32k_s11441` seeds 0 and 2, then
  `muon1338m_native_s11441` seed 1, then the 14,180-step items; the filler starts one worker per empty GPU from it until
  04:40. I start each second worker by hand with `agent/box_pack_start_1118.sh` (its own queue file, one item, and it
  stops at the first failure). None of these runs can finish; their states go to S3 from 04:40 with the others'.
- **Box at 03:45:** all eight GPUs busy (ten runs), 11 FAILED lines (none new), sync fine.

## 2026-11-19 05:26 EST: D 007's fifth seed is in (five seeds, test 0.2644 against D 006's 0.2444); box: r85 and r86 results, native against vLLM at equal steps, the packing and resume tests, and where track A's step goes (r103)

**D 007 has its five seeds** (`records/v3/track_d/007_dm_math_sil64c05_batched_muon56`, H200, job 29355448 ended at 04:36). Every seed is within the 21,600 s
budget.

| seed | RL steps | kept dev | test ladder | total_s | test minus D 006's seed |
|---|---|---|---|---|---|
| 0 | 17,359 | 0.2450 | 0.2606 | 20,880 | +0.0354 |
| 1 | 19,641 | 0.2517 | 0.2538 | 20,879 | +0.0051 |
| 2 | 18,335 | 0.2500 | 0.2646 | 20,879 | -0.0016 |
| 3 | 19,735 | 0.2550 | 0.2726 | 20,875 | +0.0391 |
| 4 | 16,318 | 0.2533 | 0.2703 | 20,856 | +0.0218 |

- Test mean 0.2644 ± 0.0076, against D 006's 0.2444 ± 0.0158: +0.0200. Dev mean 0.2510 (D 006: 0.2293).
- Still to do, after the box ends at 06:00: the README, `validate_record_v3.py --compare-to` D 006, the leaderboard
  row and the commit of the seeds' files. Until the validation passes it is not on the board. If it passes, D's
  promotion line becomes 0.2810 (its dev mean + 0.03).

**r85 (track A on the H100s, 467 steps, three seeds each; final = kept dev).**

| arm | seeds 0 / 1 / 2 | mean | paired with the control | mean difference |
|---|---|---|---|---|
| `a007_ctl` | 0.4250 / 0.4000 / 0.4400 | 0.4217 | (sd 0.0202) | |
| `a007_ent15` | 0.4383 / 0.4400 / 0.4533 | 0.4439 | +0.0133 / +0.0400 / +0.0133 | +0.0222 (SE 0.0089) |
| `a007_lr1.3e-6` | 0.4367 / 0.4367 / 0.4133 | 0.4289 | +0.0117 / +0.0367 / -0.0267 | +0.0072 (SE 0.0184) |
| `a007_cos467` | 0.4167 / 0.3933 / 0.4033 | 0.4044 | -0.0083 / -0.0067 / -0.0367 | -0.0172 (SE 0.0097) |

- `a007_ent15` is the one arm whose three pairs all have the same sign. Its mean is still under A's line (0.459),
  and +0.0222 is about one standard deviation of the control's seeds, so three pairs do not settle it (INBOX
  01:32's point). It is not a gate. It goes into the next track A arm together with more steps, at five seeds or
  more; that is decided when the 11-25 box is planned on 11-24.
- `a007_lr1.3e-6` and `a007_cos467` are closed: no consistent sign, and `cos467` is below the control in all three.

**r86 (track D on the H100s, step caps, five seeds; final dev).**

| arm | seeds 0-4 | mean | paired with the control |
|---|---|---|---|
| `d004_ctl_s11441` | 0.2700 / 0.2467 / 0.2317 / 0.2767 / 0.2400 | 0.2530 | (sd 0.0195) |
| `muon1338m_s14180` | 0.2550 / 0.2333 / 0.2467 / 0.2317 / 0.2433 | 0.2420 | -0.0110 (SE 0.0101) |
| `muon2676m_s11441` | 0.2550 / 0.2133 / 0.2100 / 0.2333 / 0.2200 | 0.2263 | -0.0267 (SE 0.0051) |
| `muon1338m_s11441` | seed 2: 0.2250; seeds 0 and 1 are running | | -0.0067 (one pair) |

- `muon2676m_s11441` is closed (below the control in all five pairs). `muon1338m_s14180` is level with or below the
  AdamW control with 24 % more RL steps. This is the 02:05 reading with all pairs in: at equal RL steps the Muon
  pretraining checkpoints do not beat D 004's.
- `muon1338m_s11441` seeds 0 and 1 (GPU 2) end at about 05:40.

**The native sampler against vLLM at equal steps, same seeds** (r86's native arms; each native run alone on its
GPU, each vLLM run with a second vLLM run beside it).

| run | steps | final dev | vLLM arm, same seed | difference | mean difference at the shared evaluations from step 2,000 | wall s | s/step from step 600 | vLLM arm's s/step |
|---|---|---|---|---|---|---|---|---|
| `muon1338m_native_s14180` 0 | 14,180 | 0.2333 | 0.2550 | -0.0217 | -0.0116 over 26 | 8,931 | 0.606 | 1.508 |
| `muon1338m_native_s11441` 2 | 11,441 | 0.2200 | 0.2250 | -0.0050 | +0.0020 over 20 | 7,524 | 0.640 | 1.227 |
| `muon1338m_native_s11441` 0 | 11,441 | 0.2167 | running |  | -0.0017 over 17 | 7,497 | 0.635 | 1.654 |

- The three native runs that could finish did. `muon1338m_native_s11441` seed 0's twin is still running; at step
  10,000, the last evaluation both have, they stood at 0.2233 (native) and 0.2483 (vLLM).
- I read the two finished pairs as level: the differences are inside the control's seed spread (0.0195),
  and the same seed does not give the same samples under the two samplers. Seed 0 of the 14,180-step pair sits
  below its twin through most of the run, so this is not yet shown to be zero. r101 found the two level at equal
  steps on the B200, and r102 on the H200 is the test that counts.
- Per GPU on the H100, one native run does 1.65 steps/s against 1.33 for a pair of vLLM runs
  (14,180-step runs, seed 0).

**r101 (B200, RL only, not yet gated).** The control's seed 1 finished: 0.2383 after 20,032 steps (seed 0:
0.2483 after 18,215). The native arm's seed 0 was 0.3033 after 34,893. The native arm's seed 1 and
both arms' seed 2 wait on `mit_normal_gpu` workers 29618971 and 29626073 (pending on priority). r102's three H200
workers (29450693, 29450694, 29450702) are also pending on priority.

**Packing tests on the 80 GB H100 (for 11-25 and 12-09), read at 05:12.** Correction to 03:45 first: I expected a
32,768-token run to hold 30 to 35 GB and a pair to do 1.3 to 1.8 times the steps of one 65,536-token run. Both
were wrong. A native run holds much more until its first evaluation at step 500, and the gain is smaller.

| run alone | whole-card GB at step 1 / 10 / 500 / 501 / 600 | s/step over steps 20-80 / 80-200 / 200-400 / 400-500 / 520-600 |
|---|---|---|
| 32,768-token micro-batches (`mb32k` seed 0, GPU 4) | 43.2 / 47.2 / 47.3 / 30.4 / 36.3 | 1.81 / 1.22 / 1.04 / 0.97 / 0.89 |
| 32,768 (`mb32k` seed 2, GPU 1) | 42.5 / 46.2 / 46.3 / 30.6 / 36.6 | 1.79 / 1.42 / 1.10 / 0.97 / 0.97 |
| 65,536 (`muon1338m_native_s11441` seed 1, GPU 5) | 68.1 / 70.7 / 74.2 / 40.1 / 51.1 | 1.47 / 1.15 / 0.90 / 0.74 / 0.81 |

- The memory falls at the first evaluation and settles at about 36.3 GB (32,768) and 51.7 GB (65,536). I
  read the fall as the sampler freeing and resizing its caches there (`_ensure_capacity` in
  `rlvr_speedrun/fast_rollout.py`); I have not checked that.
- **Two started together: fails.** The second run died within 20 s in its first update (`torch.OutOfMemoryError`,
  the first run holding 47.18 GiB of the card's 79.18). This is FAILED line 12 on the box.
- **Second started after the first's first evaluation: fits, twice.** The card is full while the second warms up
  (80.5 to 80.7 GB of 81.6), and the second run's log has allocator warnings from that time (29 on GPU 4, 42 on
  GPU 1; none since). Neither failed. Once both are past step 500 the card holds 73.3 GB (GPU 4) and
  74.1 GB (GPU 1).

| GPU | run (step now) | s/step, last 300 | pair, steps/s | against one 65,536 run alone |
|---|---|---|---|---|
| 4 | `mb32k` 0 (3,556), `mb32k` 3 (2,650) | 1.094, 1.115 | 1.81 | +20 % |
| 1 | `mb32k` 2 (3,032), `mb32k` 4 (1,763) | 1.230, 1.406 | 1.52 | +1 % |
| 5 | the 65,536 run alone, steps 1,800 to 3,500 | 0.664 | 1.51 | |

- The pair on GPU 4 gains about a fifth. The pair on GPU 1 gains nothing so far (its second run is still in its
  slower early part, and its first run was also slower alone than GPU 4's). I do not know why the two GPUs differ.
- **For 11-25: one native run per H100 at 65,536-token micro-batches stays the planning number.** A staggered pair
  gains between nothing and 20 %, runs with the card full for 10 to 20 min, and failed when started together.
- For the H200's 141 GB (11-28): two 65,536 runs, the second started after the first's first evaluation, need
  74.2 + 51.7 = 125.9 GB on paper. That has to be checked on an H200 before the 11-28 plan uses it.

**The native backend resumes from its saved state (tested on GPU 3, `muon1338m_native_s14180` seed 1).**

- The first state was saved at 04:48:40 (1.31 GB, 1,200 s of training, step 1,375). I stopped the trainer with a
  plain `kill` at 04:49:16, at step 1,427.
- `scripts/aws/box_runner.sh` restarted it 40 s later; it printed "resumed from ... at step 1375", cut the train
  log back to step 1,375 and went on from 1,376 with the wall clock continuous. No FAILED line.
- One stop cost 52 steps, 40 s and a slow stretch: 0.72 s/step before the stop, then 1.35 (steps 1,380-1,400),
  0.90 (1,400-1,500), 0.80 (1,500-1,600) and 0.67 now.
- A resumed run skips the warm-up peak: 37.9 GB at step 1,380, 47.5 at 1,400, 47.5 at 1,500, 49.7 at 1,600, 51.6 now,
  against 74.2 GB before the first evaluation of a fresh run.
- This removes r102's main preemption risk, and tonight's unfinished native runs can be continued from S3.

**Where track A's step goes on the H100 (INBOX 01:32's last paragraph).** From `a007_ctl_h100m30` seed 0's train
log (467 steps, 21,464 s):

| steps | sampling s | update s | wall s per step | k tokens per step (tokens/s times sampling plus update time) | informative groups of 96 | solve rate in batch |
|---|---|---|---|---|---|---|
| 1-40 | 18.2 | 4.9 | 23.6 | 380 | 12 | 0.02 |
| 41-80 | 11.1 | 5.9 | 18.6 | 224 | 23 | 0.05 |
| 81-120 | 20.2 | 13.2 | 39.6 | 406 | 36 | 0.12 |
| 121-200 | 28.8 | 20.0 | 55.3 | 593 | 40 | 0.26 |
| 201-300 | 27.5 | 19.0 | 51.7 | 557 | 37 | 0.37 |
| 301-400 | 26.7 | 19.3 | 51.9 | 543 | 38 | 0.40 |
| 401-467 | 27.2 | 18.9 | 51.0 | 550 | 36 | 0.42 |

- A step costs 19 to 24 s for the first 80 steps and about 52 s from step 120. Between steps 80 and 120 the
  completions grow from about 224k to about 593k tokens a step while the solve rate starts to climb.
  The first 100 steps take 2,186 s; the
  other 367 take 90 % of the run.
- From step 121 sampling is 27.5 s and the update 19.3 s of a step. The update's cost follows the number of
  informative groups (`skip_zero_advantage`); sampling follows the tokens.
- So the steps that matter for "can 5 h hold more than 467" are the late ones, and both halves are worth
  attacking. Three same-maths options exist in the code and had never been run on track A:
  `constrain_batched` (the batched answer mask in vLLM), and `grad_checkpoint: false` with `share_group_prompt`
  (A trains with activation checkpointing, which `rlvr_speedrun/grpo.py` documents as about 30 % more compute).

**r103: track A timing runs in the box's last hour** (configs under `calibration/research_r85/`; A 007's H100
config with `time_update_parts`, at most 140 steps, no evaluations, no saved state). Not a score comparison. They
are read by the time of each part, tokens per second and loss at matching steps. The box ends them at about step
100 to 125, so they reach only the start of the costly regime.

- `a007_time_h100m30` (control, GPU 7, started 04:57) and `a007_time_batched_h100m30` (`constrain_batched`, GPU 0,
  05:01) are running: steps 71 and 52 at 05:26.
- `a007_time_nockpt_h100m30` (no checkpointing, shared prompt, 8,192-token micro-batches; GPU 6) **ran out of
  memory in its first update** (the process, vLLM's 0.30 share included, held 77.33 GiB of the card's 79.18). This
  is FAILED line 13. So that setting does not fit an 80 GB card at A's micro-batch size.
- `a007_time_nockpt_mb4k_h100m30` (the same at 4,096-token micro-batches, a memory setting only; GPU 6, started
  05:05) fits so far: step 45 at 05:26, with the card at 79.0 GB. `a007_time_nockpt_mb2k_h100m30` (2,048) is the unused fallback.
- The readout comes in the next entry.

**The box at 05:26.** All eight GPUs are busy: three track A timing runs (GPUs 0, 6, 7), the two 32,768 pairs
(GPUs 1, 4), the vLLM pair that ends at about 05:40 (GPU 2), the resumed native run (GPU 3) and the 65,536 run
(GPU 5). 13 FAILED lines (11 before 21:15, two from tonight's tests). The push loop has made 80 state uploads. No
track A state exists, so none goes to S3; the AWS total is unchanged (about $2.29 at most). AWS ends the instance
at 06:00. The next entry has the timing readout, `muon1338m_s11441` seeds 0 and 1 and the final S3 contents.

## 2026-11-19 06:32 EST: D 007 is the track D holder (D's line is now 0.281); the 11-18 box has ended: native against vLLM at equal steps, track A's timing runs (r103), what is left in S3 and what it cost

**D 007 is validated, on the leaderboard and committed (1b804ff, 05:40).** The validator's line:
`new 0.2644 vs holder 0.2444: Welch t=2.55 df=5.7 one-sided p=0.0226 -> new record`. Test 0.2644 ± 0.0076 over
five seeds against D 006's 0.2444 ± 0.0158; in-loop final dev mean 0.2510 (D 006: 0.2293). **D's
promotion line is now 0.2810** (it was 0.2593). r102, the speed stack's H200 screen, is read against it. The
record's README says where the gain comes from (mostly the extra RL steps of batched masking, as LOG 02:05 and
05:26 found).

**Correction to LOG 11-17 13:35.** That entry says D 007 seed 4's first attempt was preempted "after about 3 h
15 min". It ran about 2 h 15 min: it started at 16:04:22Z (11:04 EST) and was preempted at about 18:19Z (13:19 EST),
at RL step 3,917 with 4,550 s of RL done. The record's README and `extra/interrupted_4_1/note.txt` have the right
times.

**The box ended at about 06:00 EST**, half an hour before the block's end at 06:30, which is when a capacity
block's instances are reclaimed. I did not stop it and no script of mine shuts a box down; the instance showed
`shutting-down` at 06:01 and `terminated` at 06:13, with the state reason "User initiated". Its last status,
pushed at 05:50, counted 33 runs ok and 13 FAILED lines in `queue/done.txt` (the same 13 as at 05:26). Every
finished result was on Engaging before the end, and the last fetch of the running runs' logs was at 05:58.

**r86's last vLLM arm: `muon1338m_s11441`** (the 56-min Muon checkpoint with the control's 11,441 RL steps;
three seeds ran, final dev).

| arm | seeds 0 / 1 / 2 | mean | paired with the control's seeds 0-2 |
|---|---|---|---|
| `d004_ctl_s11441` | 0.2700 / 0.2467 / 0.2317 | 0.2494 | |
| `muon1338m_s11441` | 0.2533 / 0.2233 / 0.2250 | 0.2339 | -0.0167 / -0.0233 / -0.0067: -0.0156 (SE 0.0048) |
| `muon1338m_s14180` (05:26), same seeds | 0.2550 / 0.2333 / 0.2467 | 0.2450 | -0.0150 / -0.0133 / +0.0150: -0.0044 (SE 0.0097) |

- At equal RL steps the 56-min Muon checkpoint is below the 112-min AdamW one in all three pairs (-0.0156).
  On these three seeds the 24 % more steps that the saved pretraining time pays for win back +0.0111
  (SE 0.0058), to -0.0044. Over all five pairs `muon1338m_s14180` is -0.0110 (SE 0.0101, 05:26). This agrees
  with the reading in D 007's README: on dev, the Muon split with its extra RL is about level with AdamW for
  112 min, not ahead.
- The control's five seeds have an sd of 0.0195, and one evaluation of 600 dev puzzles has a standard error of
  about 0.017 by itself, so differences of this size need more than three pairs to be firm.

**Native against vLLM at equal steps: three finished pairs** (H100, same checkpoint, seed and step cap; the 05:26
table's row for `muon1338m_native_s11441` seed 0 had its twin still running and is replaced by the row here).

| native run, seed | steps | native final | vLLM final | difference | mean difference over shared evaluations (every 500 steps from 2,000) | native wall s | vLLM wall s | native s/step from step 600 | vLLM s/step from step 600 |
|---|---|---|---|---|---|---|---|---|---|
| `muon1338m_native_s14180` 0 | 14,180 | 0.2333 | 0.2550 | -0.0217 | -0.0116 over 26 | 8,931 | 21,762 | 0.606 | 1.508 |
| `muon1338m_native_s11441` 0 | 11,441 | 0.2167 | 0.2533 | -0.0366 | -0.0062 over 20 | 7,497 | 19,065 | 0.635 | 1.644 |
| `muon1338m_native_s11441` 2 | 11,441 | 0.2200 | 0.2250 | -0.0050 | +0.0020 over 20 | 7,524 | 14,256 | 0.640 | 1.227 |

The two native runs that did not finish, at their last evaluation, against the vLLM twin at the same step:

| native run, seed | step | native | vLLM | difference | mean difference over shared evaluations |
|---|---|---|---|---|---|
| `muon1338m_native_s11441` 1 | 9,500 | 0.1917 | 0.2117 | -0.0200 | -0.0058 over 16 |
| `muon1338m_native_s14180` 1 | 7,500 | 0.1867 | 0.1983 | -0.0117 | -0.0164 over 12 |

- The three finals are all below their vLLM twin: mean -0.0211 (SE 0.0091, three pairs). Over the evaluations
  each pair shares, the curves differ by -0.0076 on average over all five pairs (SE 0.0031;
  -0.0116 / -0.0062 / +0.0020 / -0.0058 / -0.0164; 4 of 5 negative), and by -0.0053 over the three finished pairs.
- My reading: the native sampler is probably a little behind vLLM at equal steps, by something like 0.01 on dev.
  The size is not known well: five pairs from one checkpoint, a final is one evaluation (standard error about
  0.017), and a native run and its vLLM twin do not draw the same samples from the same seed, so each pair is two
  different runs. A deficit of that size is small against what twice the steps gave in r101 (B200, not gated;
  seed 0: 0.3033 after 34,893 native steps against 0.2483 after 18,215 with vLLM).
- r102 on the H200 is the test that counts: the whole stack in the record's time budget, read against 0.2810.
- Wall time: the native runs took 0.39 to 0.53 of their twin's wall time, with one native run per GPU
  against two vLLM runs per GPU.

The twins' training logs, by step range (each cell: native / vLLM, the mean over the pairs that cover the range):

| steps | pairs | mean reward | in-batch solve rate | malformed rate | entropy | pairs with the higher entropy in the native run (mean ratio) | pairs with the higher solve rate in the native run |
|---|---|---|---|---|---|---|---|
| 1-100 | 5 | 0.3009 / 0.2990 | 0.0234 / 0.0235 | 0.0797 / 0.0813 | 0.8216 / 0.8401 | 2 of 5 (0.983) | 3 of 5 |
| 101-500 | 5 | 0.4032 / 0.3997 | 0.0650 / 0.0650 | 0.0004 / 0.0005 | 0.2563 / 0.2475 | 4 of 5 (1.048) | 1 of 5 |
| 501-2,000 | 5 | 0.4678 / 0.4615 | 0.1131 / 0.1085 | 0.0001 / 0.0001 | 0.1665 / 0.1562 | 5 of 5 (1.068) | 3 of 5 |
| 2,001-5,000 | 5 | 0.5180 / 0.5173 | 0.1598 / 0.1591 | 0.0000 / 0.0000 | 0.1328 / 0.1219 | 5 of 5 (1.091) | 2 of 5 |
| 5,001-9,000 | 4 | 0.5532 / 0.5557 | 0.2005 / 0.2025 | 0.0000 / 0.0000 | 0.1198 / 0.1117 | 3 of 4 (1.075) | 2 of 4 |

- In the first 100 steps, from the same checkpoint, the two samplers give the same reward, solve rate and
  malformed rate, so they sample alike at the start.
- From step 501 to 5,000 the native runs' entropy is higher in all five pairs (1.068 and 1.091 times the
  twin's), and at 5,001-9,000 in 3 of the 4 pairs that reach it (1.075), while reward and the in-batch solve
  rate stay level on average. So training under the native sampler seems to drift to a slightly less sharp
  policy. I do not know why yet.
- Open: whether the lower dev score comes from that training drift or from the evaluation. A native run also
  evaluates with the native sampler (`_evaluate` takes whichever engine the run samples with), and no run
  evaluates before its first 500 steps, so the logs cannot separate the two. The test is cheap: one saved
  checkpoint, greedy dev evaluation by both paths, as two short GPU jobs. It goes in the queue next.

**Unfinished D runs when the box ended** (their logs to the 05:58 fetch; their saved states are in S3, up to
about 10 to 20 min behind these steps).

| run, seed | micro-batch tokens, placement | step | wall s | s/step over the last 200 steps | last evaluation | time left at that rate |
|---|---|---|---|---|---|---|
| `muon1338m_native_s11441` 1 | 65,536, alone | 9,959 of 11,441 | 6,687 | 0.602 | 0.1917 at 9,500 | 0.2 h |
| `muon1338m_native_s14180` 1 | 65,536, alone, resumed at 1,375 | 7,583 of 14,180 | 5,285 | 0.608 | 0.1867 at 7,500 | 1.1 h |
| `muon1338m_native_mb32k_s11441` 0 | 32,768, pair on GPU 4 | 6,151 of 11,441 | 6,875 | 1.005 | 0.2133 at 6,000 | 1.5 h |
| `muon1338m_native_mb32k_s11441` 3 | 32,768, pair on GPU 4 | 5,134 of 11,441 | 6,127 | 1.100 | 0.1833 at 5,000 | 1.9 h |
| `muon1338m_native_mb32k_s11441` 2 | 32,768, pair on GPU 1 | 5,298 of 11,441 | 6,751 | 1.164 | 0.1933 at 5,000 | 2.0 h |
| `muon1338m_native_mb32k_s11441` 4 | 32,768, pair on GPU 1 | 3,903 of 11,441 | 5,940 | 1.192 | 0.1550 at 3,500 | 2.5 h |

- `muon1338m_native_mb32k_s11441` seed 1 has an empty log (the pair that was started together ran out of memory,
  05:26).
- Packing, final numbers. Over their last 200 steps the GPU 4 pair ran 1.005 and 1.100 s/step, 1.90 steps/s
  together, against 1.56 steps/s for the 65,536 run alone over the same steps (+22 %). The GPU 1 pair ran
  1.164 and 1.192 s/step, 1.70 steps/s together against 1.53 (+11 %). So staggered pairs at
  32,768 gave 11 to 22 % more steps per GPU at these steps, with each run taking 1.7 to 2 times as long and a
  failed start if the stagger slips. The planning number for 11-25 and 12-09 stays one native run per H100 at
  65,536-token micro-batches; pairs are an option that is decided with the queue on 11-24. I do not know why
  GPU 1's pair was slower than GPU 4's.
- GPU 2 was empty from about 05:39 to the end: nothing that could start then would have given a result (the
  unused `a007_time_nockpt_mb2k_h100m30` arm would only have been slower in the update than the 4,096 arm).

**r103: where track A's step goes on the H100, and two same-maths speed options** (`a007_ctl_h100m30` with
`time_update_parts`, capped at 140 steps; none reached the cap before the box ended, so there is no result file,
only the train logs). The control reached step 124, `constrain_batched` step 106 and no checkpointing with the
shared prompt at 4,096-token micro-batches step 97. No checkpointing at 8,192 ran out of memory in its first
update (05:26). The arms see different completions, so the same step holds different token counts; the fair
comparison is seconds per 1,000 tokens. Each cell: sampling s + update s per step, tokens per step, then sampling
+ update s per 1,000 tokens.

| steps | control | `constrain_batched` | no checkpointing, shared prompt, 4,096 |
|---|---|---|---|
| 1-20 | 21.8 + 4.3, 420k, 0.052 + 0.010 | 25.4 + 5.0, 422k, 0.060 + 0.012 | 21.1 + 7.6, 409k, 0.051 + 0.019 |
| 21-40 | 18.5 + 5.4, 376k, 0.049 + 0.014 | 20.9 + 5.6, 354k, 0.059 + 0.016 | 15.2 + 8.6, 324k, 0.047 + 0.027 |
| 41-60 | 14.5 + 6.8, 315k, 0.046 + 0.022 | 16.6 + 6.4, 287k, 0.058 + 0.022 | 13.0 + 10.9, 275k, 0.047 + 0.040 |
| 61-80 | 13.8 + 7.6, 298k, 0.046 + 0.026 | 19.1 + 8.9, 333k, 0.058 + 0.027 | 16.2 + 14.9, 337k, 0.048 + 0.044 |
| 81-100 | 19.7 + 11.4, 394k, 0.050 + 0.029 | 27.3 + 14.2, 445k, 0.061 + 0.032 | 24.1 + 26.7, 487k, 0.049 + 0.055 (17 steps) |
| 101-120 | 25.6 + 17.1, 530k, 0.048 + 0.032 | 31.9 + 17.1, 521k, 0.061 + 0.033 (6 steps) |  |
| 121-140 | 29.6 + 19.8, 604k, 0.049 + 0.033 (4 steps) |  |  |

- `constrain_batched` does not help track A as it is: its sampling costs 1.16 to 1.27 times the control's per
  token. Building the masks takes 5.7 to 11.0 s of CPU time a step (`t_part_s_mask_cpu`, which is inside
  `t_part_s_generate`), 26 to 35 % of its sampling time. Without the mask time its sampling would cost 0.80 to
  0.87 times the control's per token, so a mask builder that cost nothing would save at most 20 % of the
  sampling time (the control's own per-request masks are inside its `generate` time and are not timed
  separately). In track D the same option was the gain behind D 007. Closed for track A as it is; open as a
  lead: a faster mask builder.
- No checkpointing with the shared prompt at 4,096 loses on an 80 GB card: the update costs 1.73 to 1.89 times
  the control's per token, with 2.0 to 2.9 times as many micro-batches, and the card
  is full either way (peak 79.0 GB against the control's 78.1). Whether it wins at 8,192 or more on an H200
  (141 GB) is open; it is one short run there.
- Inside the control's update at steps 121-124 the backward pass is 13.3 of 19.8 s (67 %) and the body's
  forward 4.3 s (22 %); the head, the mask and the allowed sets are small. Sampling is still the larger part
  of the step (29.6 s against 19.8 s), and 99.5 % of it is vLLM's `generate`.
- Sampling costs the same per token all the way (0.046 to 0.052 s per 1,000 tokens). The update's cost per
  token rises from 0.010 to 0.033 s as the reward rises and more groups carry a gradient. The step grows from
  21.3 s at its low point to 49.3 s at steps 121-124 because the tokens per step double (298k to 604k)
  and the update gets dearer.
- So neither same-maths option tested here buys track A steps on an H100. What is left for track A's speed is
  the sampler itself (track A cannot use the native sampler yet: it is written for the Llama model only), a
  faster mask builder and the H200 check of no checkpointing. LOG 02:05 put the room in A 007's budget at about 50 more steps without any
  speed change; that arm, with `a007_ent15`'s setting, is the track A candidate for 11-25 (decided 11-24).

**What is left in S3 and what the box cost outside the block.**

- `box/box1118/`: 408 objects, 0.355 GB (results, logs, queue files; kept 90 days). `box-state/box1118/`: 58 objects,
  25.07 GB: 19 saved states of about 1.31 GB each with their logs (deleted after 7 days). Six belong to the
  unfinished runs in the table above; the other 13 are states of runs that finished. The whole bucket holds 582
  objects, 25.46 GB. The push loop made 92 state uploads; its last pass was at 05:50 EST.
- No track A state was ever sent, so the $1.72 that LOG 00:14 allowed for them was not spent. The cost by this
  count: states 25.07 GB for 7 days $0.135; `box/` for 90 days $0.025; the 92 multipart uploads about $0.07
  (about 159 requests each); the result pushes every 10 min about $0.04 (my estimate); the self-launch test of
  11-18 $0.03. **Total about $0.30, and at most about $0.40**, which replaces the "about $2.29 at most" of
  LOG 00:14 and 05:26. I have not read the bill. Data sent out of AWS to Engaging was logs and result files (well
  under the 100 GB a month that is free).
- Decision on the states: the 13 finished runs' states are left to expire (deleting them would save about $0.09).
  Whether an unfinished run continues on the 11-25 box is decided on 11-24, when that box is planned; the states
  expire from 11-26, so a continuing run's state is copied to the new box's prefix before then. The two 65,536 runs
  would each add a native-against-vLLM pair for 0.2 and 1.1 h of one GPU; the 32,768 runs have little value.

**Committed with this entry.** For the 33 finished box runs: `config.json`, `eval_log.jsonl`, `result.json` and
`train_log.jsonl` (132 files; the result and config files were copied from the pool beside the logs). The three
r103 train logs. `calibration/research_r101/d007rl_ctl_b200` seed 1 (its numbers are in LOG 05:26).

**Engaging at 06:32.** None of my GPUs is running. r102's three H200 preemptable workers (29450693, 29450694,
29450702) are pending on `Resources` (05:26 said priority; 174 H200 jobs were pending in the preemptable queue at
06:07). r101's two `mit_normal_gpu` B200 workers (29618971, 29626073) are pending on priority, so at 06:09 I added
one preemptable B200 worker for the same queue (29632479, 14 h, room for two items). If everything starts that is
3 + 1 preemptable and 2 normal GPUs, six in all, the cap. The r97/r99 workers and r90's stay held.

**Next.** r102's three seeds against 0.2810 (three-seed form; then the five-seed record attempt if it is met).
r101's remaining seeds, paired by seed. The 11-28 self-running box: queue, plan, staging, schedules and
`docs/research/box_1128.md`, to be ready by 11-22. On 11-24 the 11-25 queue is planned from what is known then:
the speed stack against its control at step caps with five seeds or more, the track A arm with more steps and
`a007_ent15`'s setting, and U17/U19 arms; small moves around the holders are out (INBOX 01:32).

## 2026-11-19 06:57 EST: r104, the native evaluation-path test, is built and queued (first in the B200 queue)

The 06:32 entry left one question open: on the box the native runs ended about 0.01 behind their vLLM twins on dev
at equal steps, and a native run is also evaluated by the native sampler, so the logs cannot say whether the gap is
training or evaluation. r104 is the test (commit 5e5db85). Nothing has run on a GPU yet.

- `scripts/research/sampler_paths_eval.py` scores saved checkpoints on the config's dev set by both in-loop
  samplers. It builds the two engines as `grpo.run` builds them and hands each checkpoint's weights to both
  through `sync`, as after a training step. For each checkpoint and sampler it records `_evaluate`'s own greedy
  number, a second greedy pass that keeps every completion (so the two are compared puzzle by puzzle: identical
  completions, vLLM-only and native-only solves, the first differing token, completions at the length limit), and
  16 samples a puzzle at T = 1 in calls with as many rows as a training step's (512): solve rate, length, distinct
  completions per prompt, and the mean log-prob and entropy of the sampled tokens under the fp32 weights without
  the answer mask. It refuses an `eval_set` whose name contains "test".
- `scripts/research/run_item.sh` runs it for an item that has a `sampler_paths.json` (checkpoint list and sample
  count), under a 1,500 s timeout. No existing item has that file, so nothing queued changes.
- The item is `calibration/research_r104/d007_paths_b200`: D 007's five final checkpoints (seeds 0-4; in-loop
  vLLM dev finals 0.245 / 0.252 / 0.250 / 0.255 / 0.253) with `d007rl_ctl_b200`'s sampler settings. Dev only;
  nothing is trained or saved. My estimate is 6 to 15 min on a B200.

Checked on the login node only, without a model: a dry run of the script with stand-in engines and a stand-in
model (two checkpoints, six puzzles) ran end to end, and the runner branch built the right command with a stub
interpreter. ruff passes; pytest 183 passed.

**How it will be read.** If the two samplers give the same greedy completions on the same weights (or differ on a
few puzzles with no net change in solves), the box's gap is not evaluation, and it comes from training. If the
native greedy pass solves fewer, the native runs' dev numbers are low by that much and the equal-step comparison
is redone with it in mind. The T = 1 numbers say whether the two samplers draw alike under the answer mask: if
solve rate, length and the log-prob and entropy per sampled token agree, they do, and then the 5 to 9 % higher
training entropy of the native arms comes from their other settings (`micro_batch_tokens`, `share_group_prompt`,
`reuse_sampler_allowed`, and the vLLM arm's `constrain_batched`), not from the sampler.

**Where it runs.** It is the first item of `queue/screens_rl_b200.txt`, ahead of r101's three remaining items, so
the first B200 worker to start runs it (29618971 and 29626073 on `mit_normal_gpu`, 29632479 preemptable; all
pending at 06:57; the normal workers keep their room for one r101 item after it). I did not submit a job of its
own: `sbatch --test-only` at about 06:54 put a new 30-minute one-GPU job on 11-21 (preemptable, B200 or H200) or
11-23 (`mit_normal_gpu`), while Slurm's estimate for the two pending normal workers was 18:30 EST today when
asked a little earlier (it shows no estimate now). None of my GPUs is running; r102's three H200 workers are
still pending on `Resources`.

## 2026-11-19 09:04 EST: r104: on the same weights the two samplers score alike, so the box's native gap is not evaluation; r101's native run ran a third faster once its node emptied; a node trace runs beside r101 and r102

**r104 (the native evaluation-path test).** It ran from 08:41 to 08:49 EST on a B200 (node5602-c1, worker
29618971; 193 s of scoring). The 06:32 entry had the box's native runs about 0.01 behind their vLLM twins on dev
at equal steps, with a native run also evaluated by the native sampler. Here D 007's five final checkpoints are
scored on the 600 dev puzzles by both samplers, with the same weights handed to both. Greedy, as the in-loop
evaluation runs:

| D 007 checkpoint | in-loop final (H200, vLLM) | vLLM here | native here | native - vLLM | identical completions of 600 | solved by vLLM only | solved by native only |
|---|---|---|---|---|---|---|---|
| seed 0 | 0.2450 | 0.2450 | 0.2417 | -0.0033 | 572 | 2 | 0 |
| seed 1 | 0.2517 | 0.2550 | 0.2517 | -0.0033 | 564 | 3 | 1 |
| seed 2 | 0.2500 | 0.2517 | 0.2517 | +0.0000 | 576 | 1 | 1 |
| seed 3 | 0.2550 | 0.2567 | 0.2533 | -0.0033 | 572 | 3 | 1 |
| seed 4 | 0.2533 | 0.2550 | 0.2583 | +0.0033 | 576 | 1 | 3 |
| mean or total | 0.2510 | 0.2527 | 0.2513 | -0.0013 | 2,860 | 10 | 6 |

- Native minus vLLM is -0.0013 (SE 0.0013 over the five checkpoints), under one puzzle of 600. Of the 3,000
  completions 2,860 are the same token for token. The 140 that differ part ways at token 11 (median; 3 to
  25) and mostly end with the same verdict: 10 are solved by vLLM only and 6 by the native sampler only
  (sign test on the 16: p = 0.45). An example: `(21-7) - 6 + 2` against `(21-6) - 7 + 2`, both correct. I take
  them for near-ties that the two engines' arithmetic breaks differently; that is not checked.
- The vLLM sampler here on a B200 is +0.0017 from the same checkpoints' in-loop finals on the H200 (0 to
  2 puzzles per checkpoint), so one sampler on another GPU differs by as much as the two samplers do.
- No greedy completion was malformed or ran to the length limit under either sampler, and each sampler's
  `_evaluate` number equals the kept-completions pass.

Sampled at T = 1, 16 completions a puzzle (9,600 per checkpoint and sampler; each cell vLLM / native; log-prob and
entropy are per sampled token under the fp32 weights without the answer mask):

| D 007 checkpoint | solve rate | native - vLLM (SE over puzzles) | tokens | distinct per prompt | log-prob | entropy |
|---|---|---|---|---|---|---|
| seed 0 | 0.2414 / 0.2372 | -0.0042 (0.0024) | 24.78 / 24.79 | 4.78 / 4.96 | -2.309 / -2.316 | 0.966 / 0.966 |
| seed 1 | 0.2414 / 0.2418 | +0.0004 (0.0022) | 21.64 / 21.64 | 4.37 / 4.32 | -2.773 / -2.775 | 0.712 / 0.710 |
| seed 2 | 0.2487 / 0.2499 | +0.0011 (0.0021) | 21.75 / 21.77 | 4.54 / 4.55 | -2.677 / -2.677 | 0.885 / 0.886 |
| seed 3 | 0.2521 / 0.2553 | +0.0032 (0.0023) | 22.40 / 22.41 | 4.21 / 4.14 | -2.409 / -2.412 | 0.725 / 0.729 |
| seed 4 | 0.2499 / 0.2505 | +0.0006 (0.0022) | 23.73 / 23.73 | 3.90 / 4.01 | -2.695 / -2.693 | 0.732 / 0.734 |
| mean | 0.2467 / 0.2469 | +0.0003 (0.0012, over checkpoints) | 22.86 / 22.87 | 4.36 / 4.40 | -2.573 / -2.574 | 0.804 / 0.805 |

- Solve rate differs by +0.0003 (SE 0.0012 over checkpoints; -0.0042 to +0.0032, none beyond 1.7 of its own SEs).
- Entropy per token: native / vLLM = 1.0006 (0.997 to 1.004 by checkpoint). Log-prob per token differs by
  -0.0018 (SE 0.0015). Native completions are +0.011 tokens longer (SE 0.004), 0.05 % of their length.

**What r104 means.**

- The evaluation is cleared. A native run's dev numbers are not low because the native sampler scores it.
- At these weights the two samplers also draw alike under the answer mask. So the native arms' 5 to 9 % higher
  training entropy on the box is not the sampler drawing differently, at least at weights like these. The train
  log's entropy is the model's entropy over the rows the update uses, and the native arms differ from their twins
  in update settings too (`share_group_prompt`, `reuse_sampler_allowed`, 65,536-token micro-batches), which r104
  does not run. What is left for the box's gap is training under those settings, or chance: the gap rests on
  three pairs of finals (-0.021, SE 0.009) and five pairs of curves (-0.008, SE 0.003).
- Limits: only D 007's final weights (the box's entropy gap opened between steps 100 and 2,000 from a pretrained
  start, while the policy was still sharpening), one GPU type, and no update. r104's entropy (0.80) is taken
  without the answer mask, so it is not the train log's number (0.11 to 0.13 after step 2,000 on the box); it
  compares the two samplers' draws and nothing more.
- Decision: no GPU time for the equal-step gap by itself. At equal time the native stack is far ahead (r101 seed
  0: 0.3033 after 34,893 steps against 0.2483 after 18,215), and a gap of 0.01 in finals needs tens of seeds to
  measure. The entropy difference is the cheap handle: it showed in all five pairs within 5,000 steps. Next, on
  the login node, I read whether the logged entropy counts the same rows and mask under the native update
  settings as under the control's. If it does, a few short step-capped runs that switch one update setting at a
  time would say which one moves it; they can ride on a box or the B200 queue.

**r101's native seed 0, read more closely from its train log** (B200, node5600-c1, 21:05 EST 11-18 to 01:56 EST
11-19, 34,893 steps, final dev 0.3033).

**It was still climbing at the end.** Dev pass rate by level, means of ten evaluations of 600 puzzles:

| steps | pass | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| 500-5,000 | 0.152 | 0.555 | 0.096 | 0.056 | 0.021 | 0.013 | 0.173 |
| 5,500-10,000 | 0.216 | 0.686 | 0.161 | 0.127 | 0.058 | 0.026 | 0.240 |
| 10,500-15,000 | 0.248 | 0.729 | 0.187 | 0.163 | 0.072 | 0.057 | 0.279 |
| 15,500-20,000 | 0.259 | 0.762 | 0.213 | 0.175 | 0.083 | 0.041 | 0.280 |
| 20,500-25,000 | 0.270 | 0.795 | 0.224 | 0.183 | 0.083 | 0.046 | 0.286 |
| 25,500-30,000 | 0.275 | 0.826 | 0.200 | 0.184 | 0.091 | 0.057 | 0.291 |
| 30,500-34,893 | 0.294 | 0.838 | 0.226 | 0.208 | 0.111 | 0.085 | 0.299 |
| control seed 0, 15,500-18,215 (7 evaluations) | 0.248 | 0.721 | 0.186 | 0.133 | 0.099 | 0.051 | 0.299 |
| control seed 1, 15,500-20,032 (11 evaluations) | 0.240 | 0.806 | 0.172 | 0.130 | 0.047 | 0.026 | 0.258 |

A straight line through the 41 evaluations from step 15,000 rises 0.0023 per 1,000 steps (SE 0.0002; the
evaluations scatter 0.008 around it), so 30 % more steps at this length (about 10,500) would be worth
about +0.024. From steps 15,500-20,000 to the last block every level gained:
L1 +0.076, L5 +0.044, L3 +0.033, L4 +0.028, L6 +0.019, L2 +0.013. One seed; the two controls' last
blocks are there for scale.

**The update is the larger part of a native step and shortens by itself.**

| steps | sampling s | update s | update share | informative groups of 64 | solve rate in batch | tok/s |
|---|---|---|---|---|---|---|
| 100-4,999 | 0.172 | 0.381 | 0.69 | 49.4 | 0.149 | 22,828 |
| 5,000-14,999 | 0.169 | 0.336 | 0.67 | 36.6 | 0.243 | 25,240 |
| 15,000-24,999 | 0.166 | 0.269 | 0.62 | 31.8 | 0.284 | 29,346 |
| 25,000-34,893 | 0.145 | 0.201 | 0.58 | 29.9 | 0.313 | 37,490 |

`skip_zero_advantage` is already on in D 007's config and in r101's and r102's, so groups without an advantage
signal are already left out of the update, and the update shortens as the informative groups fall from 49 to 30.
That is not a new lever.

**The speed changed by a third within three minutes, with nothing changed inside the run.**

| | steps 24,000-28,000 | steps 29,000-34,893 | after / before |
|---|---|---|---|
| tokens per second | 31,256 | 41,491 | 1.33 |
| sampling, s per step | 0.161 | 0.135 | 0.84 |
| update, s per step | 0.249 | 0.171 | 0.69 |
| tokens per step | 12,363 | 12,449 | 1.01 |
| informative groups | 30.6 | 29.5 | 0.96 |
| wall per step with evaluations, s | 0.459 | 0.337 | 0.73 |

By 250-step block the rate was 31.3k tok/s at 01:15 and 01:17 EST, 30.5k at 01:19, 37.2k at 01:21 and 41.1k at
01:22:34, then 41.1k to 41.9k to the end. GPU memory stayed at about 52 GB. Slurm's accounting for the node shows
two other one-GPU jobs ending at 01:19:32 and 01:20:58 (and one at 01:09); longer multi-GPU jobs stayed, and a
two-GPU job that ran from 01:26 to 02:28 did not slow the run again. My run shared its GPU with nothing.

What this rules out: the run's own work (tokens per step, informative groups and memory are the same on both
sides) and another user on the same cores (the cluster gives out whole cores, `CR_CORE_MEMORY`). What is left is
the GPU side (power or thermal limits, or clocks, while neighbouring GPUs are busy) or the CPU side (the clock of
the one busy core on a loaded socket, cache and memory bandwidth): a native step is many short GPU calls from one
Python thread, and the run used about one of its 16 cores (sacct: 4:47:55 of CPU over 4:52:10), though that alone
proves little because CUDA waits spin. The cause is not known.

Older logs do not say which state is the normal one. Update time per informative group: 6.9 ms in r100's
`final_native_reuse` timing run (node5801-c1, 11-18, steps 100-300), 8.1 ms here before the change and 5.8 ms
after, the fastest native rate seen so far.

**What it means.**

- A time-budgeted native run's step count can move by a quarter to a third with the state of its node. At the
  late slope that is 0.02 to 0.03 dev, as large as the promotion margin. Had the later rate held from the start,
  this run would have made about 47,000 steps (a rough upper figure: the earlier wall time divided by 1.36).
- So each native result is reported with its steps and its node, arms are compared at step caps (PLAN's box rule
  already), and r102's three H200 seeds, which are time-budgeted as a record would be, are read with their step
  counts beside them.
- The vLLM controls are steadier: from one 250-step block to the next their rate moved by at most about 1.10 up
  and 0.85 down.
- If the cause turns out to be something a job can choose (a node, a core, a power state), it is worth up to a
  third more steps at no cost. That is the reason for the trace below.

**The node trace (commit aa41eb8).** `scripts/research/node_trace.sh` appends one csv row every 30 s beside a
run: the job's own GPU from `nvidia-smi` (utilisation, power, temperature, memory, and whichever clock and
throttle fields the driver answers) and node-wide numbers that name nobody (1-min load, runnable tasks, the mean
CPU clock over the job's cores and over all cores, the wall time of a fixed shell loop as a single-core speed
reading, the pressure averages, the job cgroup's CPU seconds). It reads no other process and no other user's job.
`run_item.sh` starts it detached for an item whose directory holds a `node_trace` file, never for a record; the
run and its config are untouched. The file is in r101's two item directories and in r102's, so r101's three
remaining items and r102's seeds write `seeds/<seed>/node_trace.csv` (r102's only if its workers run items
through `run_item.sh`, which I expect and check when the first one starts).

Tested first on the login node with stand-ins for `nvidia-smi` and for the trainer: the runner returned when the
trainer ended, rows were written at the set interval, no trace process was left, and without the trigger file
nothing was written. ruff passes; pytest 183 passed. Its first real rows came at 08:49 on node5602-c1, beside
both r101 runs: the driver answers all eight optional fields (GPU clock, the throttle and event reasons, the
power-cap and slowdown flags, the enforced power limit, the performance state), the cgroup CPU seconds are filled
(they count the whole Slurm job, so a worker's second item starts above zero), and the three pressure columns are
empty because the compute node has no pressure files.

**Jobs at 09:04.** Two GPUs are running, both B200s on node5602-c1 under `mit_normal_gpu`, hours before Slurm's
estimates (which ranged from 09:57 to 18:30 when read between 06:50 and 08:45). Worker 29618971 started at 08:41,
ran r104 and has run r101's control seed 2 since 08:49. Worker 29626073 started at 08:48 and runs r101's native
seed 1. Both items are traced, both have a 17,400 s budget and end at about 13:45, and the two share a node, so a
node-wide change should show in both traces. One item is left in the B200 queue, r101's native seed 2, for
29632479 (preemptable, waiting for nodes). r102's three H200 workers wait on priority. The B200 queue needs new
items today; that is the next decision.

## 2026-11-19 09:15 EST: the update's code does not explain the native entropy gap; r104 gets a second item, the pretrained weights

The 09:04 entry named a code read as the next step on the box's native gap (native runs train at 5 to 9 % higher
entropy after their first 100 steps and end about 0.01 behind at equal steps). It is done, on the login node,
with no GPU.

- **The logged entropy is the same quantity in both arms.** `_entropy` in `rlvr_speedrun/grpo.py` is the entropy
  of the policy after the answer mask. `_update` sums it over the completion tokens of the updated rows (the rows
  with a non-zero advantage, since `skip_zero_advantage` is on in both arms) and divides by those rows' tokens.
  The micro-batch size and the shared prompt change neither the rows nor the mask.
- **One optimizer step per GRPO step in both arms.** Micro-batches only accumulate gradients, so 32,768 against
  65,536 tokens changes memory and speed, not the number or size of updates.
- **The reused allowed sets are the recomputed ones.** `_allowed_tokens` takes the sampler's sets only for a row
  whose valid tokens are exactly the sampled prefix, and recomputes any other row. The tests assert the
  sampler-recorded sets equal the recomputed sets, and that the shared prompt gives the same update gradient
  (`tests/test_fast_rollout.py`, `tests/test_grpo.py`; both on a tiny model, not at scale).
- **The box pair's configs differ in five keys only** (`muon1338m_s11441` against `muon1338m_native_s11441`):
  `rollout_backend`, `constrain_batched`, `micro_batch_tokens`, `reuse_sampler_allowed`, `share_group_prompt`.
  Both start from the same pretrained checkpoint.

So nothing in the update's maths explains the gap, and the runs I named at 09:04 (switching one update setting at
a time) are dropped: they would test settings that are equal by construction. Two explanations are left. One is
drift between runs that draw different samples: the entropy is higher in 5 of 5 pairs in two step bands, but the
bands share the same five pairs, so that is one sign test at about p = 0.06, not two. The other is a sampler
difference that shows only at early weights. r104 scored D 007's final weights, where the policy is sharp, and
the box's gap opened between steps 100 and 2,000 from a pretrained start.

**r104's second item covers that case.** `calibration/research_r104/d007_pre_paths_b200` scores D 007's five
pretrained (pre-RL) checkpoints by both samplers, greedy and with 16 samples a puzzle at T = 1, with the first
item's settings. It is first in `queue/screens_rl_b200.txt` (queued 09:09) and has not run. I expect about 8
minutes on a B200; completions may be longer at these weights, and the 1,500 s timeout applies.

How it will be read: if the T = 1 solve rate, length, log-prob and entropy per sampled token and the greedy pass
rate agree as they did at the final weights, the sampler is cleared at both ends of training, the gap is put down
to drift, and the equal-step question is closed until r102 gives end-to-end numbers. If they differ, there is a
sampler difference early in training, and it is fixed before the speed stack becomes the record path.

Jobs at 09:15: unchanged since 09:04. Two B200s run r101's native seed 1 and control seed 2 until about 13:45.
29632479 (preemptable B200) and r102's three H200 workers are pending. The B200 queue still needs new arms today.

## 2026-11-19 09:28 EST: r105 queued (lr 2x and 4x on the native RL-only item, B200); where the native step's time goes

**The arms.** `calibration/research_r105/d007rl_native_lr12e-6_b200` and `..._lr24e-6_b200` are r101's
`d007rl_native_b200` with lr 1.2e-5 and 2.4e-5 (2x and 4x D 007's 6e-6, constant as there). Nothing else differs:
the same D 007 seed-n pretrained checkpoint, 17,400 s, the speed settings and a node trace. Seeds 0-2 of each are
queued after r101's native seed 2, which is their control's third seed.

**Why lr, and what I expect (written before any of it runs, as INBOX 01:32 asks).** The earlier lr results were
ties, but in a regime that had stopped learning: r29's arms were flat from step 4,500, and r34 (lr 1e-5 against
6e-6 over 15,000 steps, before self-imitation) gave -0.010, -0.007 and +0.025 by seed. r29's reading was that a
higher lr gets to the same level sooner. The native control is not flat: seed 0 still gained 0.0023 dev per 1,000
steps over its last 20,000 steps. If a higher lr again gets to a given level sooner, on a curve that is still
rising it ends higher. My expectation for 2x is about +0.02, with a range of -0.02 to +0.05 (the top is twice the
steps at the late slope with some flattening; the bottom is a lower plateau from noisier updates or an earlier
entropy collapse). 4x is there for the shape of the response: it may fail, and that is information too.

**Against the noise.** The vLLM control's two finished seeds have last-block dev means of 0.248 and 0.240, D 007's
five finals have an SD of 0.004, and single evals scatter by 0.008 around the native curve's fit. For block means
of ten evals, paired by seed, three pairs give an SE of about 0.005 (my estimate from those numbers). An effect
of +0.02 is four of those; anything inside +-0.01 reads as a tie.

**How it is read.** Paired by seed against `d007rl_native_b200` at matched steps on the dev curves (block means of
ten evals), since a time-budgeted run's steps move with its node; the finals, steps and node are reported beside
that. A gain of 0.015 or more over the three pairs puts that lr into the next H200 gate config. A tie closes lr
for the native regime, and no further lr arm runs. Cost: six runs of 4.83 h, about 29 GPU-hours.

**Considered and not queued.** `oversample` (the uninformative groups are sampled either way, so it is a larger
batch by another name; r18 and r71 found nothing there); level weights (r45, r57 and r73 were flat or worse);
groups of 16 (r35 tied); seeds 3-4 of r101 (they wait for the first three pairs).

**Where the native step's time goes (r100's timing runs, B200, 300 steps each with `time_update_parts`; means of
steps 51-300; no new GPU time).** At the final weights with `reuse_sampler_allowed` (`final_native_reuse`):

| part | s per step |
|---|---|
| step (sampling + update) | 0.392 |
| sampling | 0.148 |
| - of which the answer constraint (CPU) | 0.087 |
| update | 0.244 |
| - forward, body | 0.068 |
| - backward | 0.083 |
| - self-imitation replay | 0.054 |
| - mask in the forward pass | 0.015 |
| - replay batch | 0.014 |

The constraint is 22 % of the step there and 18 to 22 % across r100's four runs. It is 59 % of
sampling: one Python grammar call per row per decoded token (512 rows), then the mask. The model's own decoding
is the smaller part. It is the largest single piece left that is not the model's forward and backward pass, and
it is CPU work, which fits a run whose speed moved when its node's other jobs ended (the traces will say).
**Next code item:** fewer grammar calls with the same result (rows of a group that share a prefix share one
call), behind an option that is off by default, timed as r100 was. Halving the constraint's cost would give
about 12 % more steps and removing it 28 % (by the 09:04 slope about +0.010 and +0.023 dev on a
35,000-step run; to be measured, not assumed).

Jobs at 09:28: the two B200s still run r101's native seed 1 and control seed 2 (to about 13:45). Submitted
29644110 and 29644111 (`mit_normal_gpu`, B200, 6 h, one item each), each set to start only after one of the
running workers ends, so no more than two normal GPUs run. 29632479 and r102's three H200 workers are still
pending. The B200 queue now holds r104's second item, r101's native seed 2 and r105's six items.

## 2026-11-19 10:12 EST: the answer constraint's cost is in new grammar states; `factored_grammar` builds them 3x faster on the CPU; r106 queued to time it

The 09:28 entry named the native sampler's answer constraint as the next speed item (0.087 s of the 0.392 s step
at the final weights) and planned to let rows with the same prefix share one grammar call. I measured first, and
the measurement changed the plan. Everything here ran on the login node's CPU without a model (three scripts,
under a minute each, two threads); nothing has run on a GPU yet.

**Where the time goes (simulated GRPO steps: 64 puzzles x 8 rows, each row a random walk through the grammar,
the track D tokenizer; about 13,972 row-tokens per step).**

- The grammar call is 92 % of the constraint's CPU time (0.1017 of 0.1107 s per step). The log of allowed
  sets, the hashing and the torch part together are 0.009 s.
- Inside the grammar call, a state seen before is a dictionary hit, and a state seen for the first time walks
  the trie of all 477 candidate tokens (52.7 `step` calls and 18.8 us on average, for a set of 11.1 tokens).
  A step meets 4,412 new states. New states are 82 % of the first pass: replaying the same sequences, all
  hits, costs 0.0185 s against 0.1014 s.

So sharing a call among rows with the same prefix would save little (those are hits already). The cost is
building a new state's set, and that is what the change makes cheaper.

**The change (commit 1c7883d, `factored_grammar`, off by default; same sets, so the same maths).** Of the 477
candidate tokens of the track D tokenizer, 457 hold no digit and 20 are single digits. Whether a token without a
digit is allowed depends only on the state's shape (mode, how many numbers remain, bracket depth, progress
through the closing tag, whether the last character was a space, and inside a number whether the digits so far
complete a remaining number), not on which numbers remain. Those tokens are walked once per shape and cached.
An all-digit token is allowed exactly when the digits so far plus the token start one of the remaining numbers,
which is read off the numbers without a walk. Tokens that mix digits with other characters (none in this
tokenizer) are still walked. If a puzzle's `ops` held a digit the full walk is used.

Checks: on every reachable state of seven puzzles the factored set equals the walk's, with the SmolLM2 tokenizer
and with a synthetic vocabulary that has multi-digit and mixed tokens, where both also equal a brute-force feed
of every candidate. Two native engines with the option off and on sample the same sequences and record the same
sets, and the GRPO-level native test now runs with it on. ruff passes; pytest 186 passed.

**CPU benchmark of the change (the same sequences through both grammars, order alternated).** 35,297 states,
every allowed set equal; 572 shapes were walked for them.

| per simulated step | walk | factored |
|---|---|---|
| first pass, s | 0.1193 | 0.0357 |
| a new state, us | 22.85 | 4.06 |
| replay (all hits), s | 0.0184 | 0.0178 |

The first pass takes 0.30 of the walk's time. The cache of sets per state is unbounded, as before; the two
finished long B200 workers peaked at 12.6 and 12.2 GB of their 128 GB (one ran 34,893 native steps), so it
needs no bound.

**What I expect on the B200, written before the result.** A trained policy repeats more states than a random
walk, so less of its constraint time is new states and the saving should be smaller than the benchmark's. The
B200's constraint cost is 6.6 us per row-token (0.0868 s for about 13,072 tokens), between the benchmark's
all-hits 1.3 us and its first-pass 8.5 us on a different CPU, which fits a large share of new states. At
the benchmark's ratio the step would go from 0.392 to 0.331 s (+18 % steps); I expect `t_part_s_mask_cpu` to
fall by between a third and 70 %, which is +8 % to +18 % steps at the final weights.

**r106 (B200, timing only, 300 steps each, `time_update_parts`, traced every 10 s).** Six items copied from
r100's `final_native_reuse` and `start_native_reuse` (D 007 seed 0's final and pretrained weights, the settled
speed settings): `final_native_ctl`, `final_native_factored`, `final_native_factored_b`, `final_native_ctl_b`
(the pair twice, in the order control, factored, factored, control) and `start_native_ctl`,
`start_native_factored`. About 3 min each at the final weights and 5 min at the start weights. In
`queue/screens_rl_b200.txt` the four final items follow r104's second item, ahead of the long items; the start
pair sits after two long items, so a 6 h worker's 35 min of slack is never spent on short items alone.

How it will be read: means over steps 51-300 of `t_part_s_mask_cpu`, `t_sample`, `t_update` and tokens per
second, control against factored on the same node (the node and worker of each item from `queue/worker.log`), and
`loss` and `mean_reward` step by step. At the final weights the four runs should match exactly: r100's two
final-weights runs were equal on all 300 steps. At the start weights same-maths runs part from step 2 (LOG 11-18
12:03: the grad norm differs in the fifth digit at step 1, and at entropy 3.2 that changes the next samples), so
the start pair is read on its timings only, with tokens per step beside them. If the final runs match and the
constraint time falls, `factored_grammar: true` joins the settled speed settings for new D arms and the box
queues (r102 and the items already queued keep their configs). If a factored final run differs from its control
while the two controls match each other, that is a bug, to be found before the option is used anywhere. A
cheaper cache-hit path is decided only after this result.

## 2026-11-19 10:47 EST: launch schedules exist for all three paid blocks; the box setup can now stage a pretraining screen's token file

Two pieces of the self-running box (INBOX 11-18 13:35), and what is still missing.

**Schedules.** At about 10:25 I ran `bash scripts/aws/box_auto.sh schedule <reservation id>` for the 11-25, 11-28
and 12-09 blocks. Each block now has 11 one-time EventBridge schedules, `rlvr-box-<reservation id>-m0001`,
`-m0005`, `-m0010`, `-m0025`, `-m0060`, `-m0180`, `-m0360`, `-m0540`, `-m0720`, `-m0900` and `-m1080`: 1, 5, 10, 25
and 60 min and 3, 6, 9, 12, 15 and 18 h after the block starts (11:31Z to 05:30Z the next day). Each calls
RunInstances with the block's launch template under role `rlvr-scheduler`; the template's default version is the
self-running one. Schedules cost nothing. `box_auto.sh show` for each block at 10:32 EST: reservation
`scheduled`, default template version self-running (1, 3 and 1), 11 schedules `ENABLED` with the right template
name and times, no instance, no RunInstances call in the last 6 h.

What follows from them:

- Nobody has to launch a box by hand. A launch while an instance holds the block's network interface is refused
  (tested on 11-18), so a hand launch and a scheduled one cannot both succeed, and the later schedules do nothing
  while the box is up.
- If the box dies, the next schedule starts a new one, which takes the queue less what the bucket lists as done
  (tested on 11-18 with the cheap instance). So before I terminate a box on purpose inside its block, I must run
  `box_auto.sh unschedule <reservation id>` first.
- The stage has to be in the bucket before the block starts. The user data tries `in/code.tgz` nine times, 20 s
  apart, and then stops trying; such a box comes up like the 11-18 one and has to be run by hand.
  **Nothing is staged for any of the three blocks yet.**

**End-to-end screens could not be staged; now they can.** INBOX 11-19 01:32 asks for the speed stack end to end
with enough seeds before anything else gets box time. An end-to-end D item pretrains, and pretraining reads the
token file `mix_0.2.bin` (4.64 GB). `box_s3.sh` staged a screen's starting checkpoint, the base model's cache
and the tokenizer, but not that file, and its `check` refused every item with its own pretraining. Changes
(this commit):

- `home_paths` lists the token file of a screen that pretrains (no `screen_pretrained`; both `STAGE=train` and
  `pretrain_only` items) as `<path under ~>.tar`. `home` uploads it as one tar to the shared `box/home/` prefix,
  so one upload serves all three blocks, and `box_boot.sh` unpacks a `.tar` path in place, where the configs'
  absolute path points. An RL-only screen does not pull the file.
- `home` now checks that a `.tar` path's source exists before it pipes `tar` into the upload. Before, a missing
  source made `tar` fail while it still wrote an empty 10 kB archive into the pipe (checked locally); the upload
  would have stored it, and a second run of `home` and `check` would have taken that object as present.
- The warning that made `check` fail is narrowed to what the box still cannot run: a record seed, and a
  pretraining item whose token file or tokenizer is outside the home directory.
- `tests/test_box_s3.py` (4 tests) runs the script's own `items` and `home_paths` text against a small git
  repository of configs: an RL-only screen, an end-to-end and a `pretrain_only` screen, a track A screen with
  `HOME_EXTRA`, and the four refused or warned cases. ruff passes; pytest 190 passed.

Checked beyond the unit tests: on the real configs (r102 seed 0 and r101's native seed 2) the function prints
the token tar, the tokenizer and D 007's seed 2 checkpoint with no warning; and the tar, copy, unpack and remove
commands of `home` and `box_boot.sh`, run by hand on a 1 kB file in a scratch directory, give back the same
bytes at the path the config would name.

**Not tested yet.** None of this has run against the bucket or on a box. The 11-18 cheap test staged an empty
`home.txt` (I read it in the bucket today), and the 11-18 GPU box was filled by hand, so no box has yet fetched
any path from `box/home/`, directory or tar. That is one more boot of the cheap instance with a small directory
and a small tar in `home.txt` (a few cents), to do before the first real stage is relied on.

**How a pretraining item behaves after a failure on a box (read from the code, not run).** `queue_worker.sbatch`
moves a seed directory that holds a `train_log.jsonl` or `pretrain_log.jsonl` to `extra/interrupted_<seed>_<n>`
before it starts the item, so a failed or restored end-to-end item starts again from pretraining. It has no saved
state to resume from (`box_runner.sh` resumes in place only when a `state.pt` exists). The keeper puts a FAILED
item back once. The worker redoes an item in hand without checking the time left, so a rerun late in the block
can be cut off by the block's end. Pretraining minutes and `time_budget_s` are wall-clock budgets, so timed
end-to-end runs go one per GPU. The logs are written into the seed directory as the run goes, so the box's
pushes every 5 min carry a cut-off run's partial logs.

**Disk.** Each pretraining run copies the token file to `/tmp` (eight at once is 37 GB) and saves a
checkpoint of about 0.44 GB. The three templates' default versions have a 2,000 GB gp3 volume (read from AWS
today), so disk is not a limit.

**AWS charges:** unchanged, about $0.30 in total (at most about $0.40). The token tar in the bucket will add
about $0.11 a month (about $0.32 over the 90 days the prefix keeps it) once it is uploaded.

**Next for the boxes, by about 11-22:** the cheap boot that fetches from `box/home/`; a baseline queue and plan
for 11-25 and 11-28, staged with `stage`, `home` (a `mit_normal` CPU job) and `check`, so a box that starts
with me down has work; `docs/research/box_1128.md`. Each queue is planned again the day before its block
(11-24, 11-27, 12-08) from what has been learnt. I asked in QUESTIONS (10:47) whether an end-to-end screen on the
11-28 H200 box may serve as the gating screen.

## 2026-11-19 11:08 EST: cheap boot test of the fetch from `box/home/`: a directory, a one-file tar and the tokenizer reached the box and matched by sha256

The 10:47 entry left one piece of the self-running boxes untested: no box had fetched anything from the shared
`box/home/` prefix, and the new token-file staging had never run against the bucket. This test covers both with
small files, on a t3.large (no GPU), for about one cent.

**What was staged (commit d41d304).** Test files made on Engaging under `~/orcd/pool/boxtest/`: a directory
`ckpt/` (a 1,000,000-byte random `model.bin`, a `config.json` and a file in a subdirectory) and a
20,000,000-byte random `tokens.bin`. `box_s3.sh stage`, `home` and `check` ran with
`HOME_EXTRA="orcd/pool/boxtest/ckpt orcd/pool/boxtest/tokens.bin.tar orcd/pool/data/tinyllama-tokenizer"`: the
directory went up by sync, the file as one tar stream (the route a pretraining screen's token file takes), and
the real track D tokenizer (three files, 3.6 MB) as a directory. The real boxes need the tokenizer too, and
it now stays in `box/home/`. `check` printed "check ok". These were the first runs of the three commands with
the 10:47 code. In all 24,629,400 bytes.

**How the box checked it.** The test stub (`scripts/aws/box_test_runner.sh`, commit ad9a26f) takes
`TEST_NEED="<path under ~>=<sha256> ..."` from the plan's worker line and fails an item when a file is missing
or differs. The plan named five files with checksums computed on Engaging: the three in `ckpt/`, the untarred
`tokens.bin` and the tokenizer's `tokenizer.json`. The queue held two new stub arms (`calibration/box_test/h1`
and `h2`; six 10 s steps, one planned failure at step 3), so the results do not mix with the 11-18 test's.

**What happened (11-19, times in UTC; instance i-0c5daecf12dda9c42).**

| time | event |
|---|---|
| 15:59:02 | the one-time schedule fires |
| 15:59:12 | `RunInstances` under the `rlvr-scheduler` role succeeds |
| 16:01:01 | `box_boot.sh` starts, 1 min 49 s after the launch |
| 16:01:12 | "fetched: the plan, 1 queues, 3 paths under ~" (11 s) |
| 16:01:22 | both workers start their items |
| 16:02:53 | both items `ok` |
| 16:03:00 | "idle: last push, then shutdown", 3 min 48 s after the launch; the instance terminated itself |

Each item's log shows five "TEST_NEED ok" lines with the right sizes (1,000,000, 27, 7, 20,000,000 and
3,619,016 bytes), and each failed once at step 3 as planned and "resumed from step 3". The box's log has no
"NOT FETCHED" line. The results, the done list and the box's logs were in the bucket before the shutdown, and
`box_s3.sh pull` brought them into the checkout. So the whole path has now run once with nobody on the box:
schedule, launch, code, plan, queue, files under `~` from `box/home/` (a directory and a tar), workers, a failed
item resumed from its saved state, results off the box, shutdown.

**Two things the test turned up.**

- The test template's block end is set to three hours after the template is made, so the 11-18 version's end
  was long past. A box from it would have started workers that take nothing (their deadline is the block end
  less 35 min). I made version 2 (`box_auto.sh template test`) and ran `dryrun test` before scheduling. A
  re-test needs `template test` first. The real templates carry their block's true end and are not affected.
- `box_s3.sh clear rlvr-box-test` removes the test box's output, which STATUS listed as kept for the audit. I
  first copied the 11-18 test's `box/rlvr-box-test/` to `box/rlvr-box-test-1118/` within the bucket (42
  objects, 0.98 MB).

**Still not tested.** The real 4.64 GB token-file tar (not uploaded yet: it goes up as a `mit_normal` CPU job
when a real queue with a pretraining item is staged), a real checkpoint directory or the model cache through
this route on the new boxes, and anything with a GPU. One known weakness in `box_boot.sh`: after a directory
sync or an untar it only checks that the path exists, so a fetch that stopped part-way would pass. The
commands' exit codes should count too. That is a few lines; I will make the change and repeat this test (about
a cent) before staging the real queues.

**Left in place for a re-test and for the audit:** the test files under `~/orcd/pool/boxtest/` on Engaging
(21 MB) and under `box/home/orcd/pool/boxtest/` in the bucket, the staged `box/rlvr-box-test/in/` and this
test's `out/`. No instance, loose volume or test schedule remains (checked after the test).

**AWS charge.** One t3.large for at most 5 min 35 s and its 80 GB volume: $0.009 by list prices, so about
$0.01; the S3 uploads (25 MB) are far under a cent. Running total about $0.31, at most about $0.41. I have
not read the bill.

## 2026-11-19 12:28 EST: repeat of the cheap fetch test at 344eb32: a cut-off tar is reported as not fetched, the rest arrives

**The change (344eb32).** In `scripts/aws/box_boot.sh` a copy, sync or untar that fails now marks its path as not
fetched, even when it left part of the path behind. Before, only the path's existence was checked. Checked by hand
first: a 1,003,520-byte tar cut to 600,000 bytes unpacks to a 599,040-byte file with `tar: Unexpected EOF in
archive`; the old check accepted that file, the new one reports it. ruff clean, 190 tests pass.

**The test.** Staged at 344eb32 for `rlvr-box-test`: a plan, the queue `box_home2` (stub items
`calibration/box_test/h3` and `h4`, each checking five fetched files by sha256) and four paths under `box/home/`:
the three of the 11:08 test and `orcd/pool/boxtest/bad.bin.tar`, the cut-off tar, uploaded by hand. `check ok`.
The test template was made again (default version 3), the dry run under `rlvr-scheduler` passed, and a one-time
schedule launched the box. The first 11-19 test's stage and output were copied to `box/rlvr-box-test-1119/` first.

| UTC | event | after the launch |
|---|---|---|
| 17:17:38 | the schedule fires | |
| 17:17:48 | RunInstances (t3.large) | |
| 17:19:37 | `box_boot.sh` starts | 1 min 49 s |
| 17:19:44 | `NOT FETCHED: orcd/pool/boxtest/bad.bin` after tar's `Unexpected EOF in archive` | |
| 17:19:49 | the fetch ends (12 s; the other three paths are there) | |
| 17:19:51 | the two workers start | |
| 17:21:28 | both items `ok` (five sha256 checks each, one planned failure at step 3, resumed) | |
| 17:21:34 | idle: last push, then shutdown | 3 min 46 s |

**Result: as intended.** The box's `status.txt` carried `NOT FETCHED: orcd/pool/boxtest/bad.bin` and the
`fetch_failed` marker from its first write, the workers started all the same, and both items passed because
neither reads that path. On a real box an item that needs a missing path fails by itself, and the status file
names the path. The instance was `terminated` when next read at 17:26:33; no volume and no test schedule is left.

**What it does not cover.** Only failures that the copy, sync or untar reports. A file that is complete but wrong
passes; `box_s3.sh check` looks at the stage from the Engaging side before a launch. The log line after the
fetch still reads "4 paths under ~", which counts the paths listed and not the paths fetched; the `NOT FETCHED`
line and the status file are what to read. I will reword it with the next change to that file.

**Still not tested:** the 4.64 GB token-file tar (not uploaded yet), a real checkpoint directory or the model
cache through this route on the new boxes, and anything with a GPU.

**Left in place.** Template `rlvr-box-test` at default version 3 (its block end is 3 h after it is made, so it is
made again before any re-test). In the bucket: the repeat's stage and output under `box/rlvr-box-test/`
(20 objects, 963,092 bytes), the first 11-19 test under `box/rlvr-box-test-1119/` (20 objects, 962,564 bytes),
the 11-18 test under `box/rlvr-box-test-1118/`, and `box/home/orcd/pool/boxtest/bad.bin.tar` (600,000 bytes)
beside the other test files, all to go once the build is audited.

**AWS charge.** One t3.large for at most 8 min 45 s and its 80 GB volume: $0.013 by list prices, so about
$0.01; the S3 uploads (under 2 MB) are far under a cent. Running total about $0.32, at most about $0.42. I have
not read the bill.

## 2026-11-19 12:40 EST: `docs/research/box_1128.md` written; pretraining screens keep no saved state on a box

**The write-up asked for in INBOX 11-18 13:35 exists** (177 lines): what is on AWS for the three blocks
(reservations, launch templates and their default versions, the 33 schedule names and launch times, roles
`rlvr-scheduler` and `rlvr-box` with their policies, the bucket prefixes and expiry rules), what a box does by
itself from launch to block end, a read-only command for each piece, the rules for working by hand, the three
cheap tests, and what is not tested. It says plainly that no queue is staged yet and that nothing has run on a
GPU instance. Per block it still lacks the staged commit, the queue names, the plan lines and the output of
`check`; those go in when each stage is made.

**One limit found while writing it.** "Each run's saved state every so often" (INBOX 13:35, point 3) holds for
screens run through `scripts/slurm/screen_seed.sh`: track A screens and track D's RL-only items, which save a
state every 60 min and resume from it on a new instance (shown on the cheap box with stub items, and by the
11-18 box's states reaching S3). A track D screen with its own pretraining runs through
`scripts/run_seed_v3.sh`, which saves no resumable state. On a box, if the instance is lost, such a run starts
again from the beginning; only its logs so far are in S3.

**What that means for the plans.** The many-seed end-to-end screen planned for the 11-28 box is made of such
runs: 46 min of pretraining and 5 h of RL, 5.77 h each before evaluation. An instance
lost late in a run costs up to that much on each of eight GPUs. I am not changing `run_seed_v3.sh` (records run
through it). The plan for 11-28 keeps this in view in two ways: the end-to-end runs go first in the block, so a
rerun still fits, and I will look at whether the pretrained checkpoint of each seed can be kept and sent to S3
when pretraining ends, so that a replacement instance repeats only the RL part. That is a new option, off by
default, and it is decided when the 11-28 queue is planned, not now.

No AWS call was made for this entry and nothing was charged.

## 2026-11-19 13:01 EST: r107 queued: the GPU memory of fused pretraining at micro-batch 128, which was never measured

**Why.** While gathering facts for the 11-25 baseline queue I found that the memory of the settled pretraining
path (`--fused-loss --micro-batch 128`) is not in any log: `pretrain_scratch.py` logs none, and r91 and r92 ran
on a B200 (180 GB) without a node trace. Two plans rest on it. r102, the H200 gate, starts with this pretraining
on a 141 GB card, and that path has never run on an H200 (r90, its H200 benchmark, is still held). The 11-25 box
has 80 GB cards. The only logged number is 31.6 GB for the old path at micro-batch 32 on an H100 (LOG 11-05).

**The item.** `calibration/research_r107/pt_fused128_mem` seed 0: r91's `bench_fused_mb128` config again (300
optimizer steps, about 5 min, pretraining only) with a node trace every 5 s. It is first in `screens_rl_b200`
and runs when the next B200 worker starts (about 13:50 EST, when r101's two runs end).

**What I expect and what follows.** Somewhere between 50 and 90 GB; I have no better basis than the size of the
activations. The trace reads the card's `memory.used`, which is what the process holds, so it is an upper bound
on what the run needs.
- Under about 70 GB: both plans stand, and the 11-25 end-to-end arm uses micro-batch 128.
- Between 70 and 141 GB: r102 stands; the H100 arm uses `--micro-batch 64` (+19 % tokens per second in r91
  against +21 %) unless a run on an 80 GB card shows 128 fits.
- Over 141 GB: r102 would fail at once on the H200, and its queue gets a config with micro-batch 64 before its
  workers start.

**One more fact for the box plans.** On a box the worker's time check uses the block's deadline: an item is
started only while `RESERVE_MIN` minutes remain (`queue_worker.sbatch`, one value per `box_workers.sh` call).
A plan line for six-hour end-to-end runs therefore sets `RESERVE_MIN` to the run's full length, so that no run
starts that cannot finish.

## 2026-11-19 14:35 EST: r101's second native seed and third control seed (B200, RL only): both pairs +0.05; the gain is the extra steps

**What ran.** r101 starts from D 007's pretrained checkpoints and gives RL 17,400 s on a B200, so it is not yet
gated. The native arm (`d007rl_native_b200`: native sampler, `reuse_sampler_allowed`, shared prompt,
65,536-token micro-batches) runs against the control (`d007rl_ctl_b200`: D 007's own RL settings on vLLM). Native
seed 1 and control seed 2 ended at 13:40 and 13:42 EST, both on node5602-c1. Native seed 2 is still queued.

| arm | seed | node | RL steps | final dev | mean of last five evals | best eval | steps/s |
|---|---|---|---|---|---|---|---|
| native | 0 | node5600-c1 | 34,893 | 0.3033 | 0.2920 | 0.3033 | 2.01 |
| native | 1 | node5602-c1 | 44,862 | 0.2883 | 0.2857 | 0.2883 | 2.58 |
| control | 0 | node5600-c1 | 18,215 | 0.2483 | 0.2463 | 0.2533 | 1.05 |
| control | 1 | node5800-c1 | 20,032 | 0.2383 | 0.2393 | 0.2583 | 1.15 |
| control | 2 | node5602-c1 | 17,608 | 0.2750 | 0.2707 | 0.2800 | 1.01 |

Final dev: native mean 0.2958 (two seeds), control mean 0.2539 (three seeds, sd 0.0190).
By pair: seed 0 +0.0550, seed 1 +0.0500, mean +0.0525. Both native finals are above
D's line of 0.281; all three control finals are below it.

**At equal steps the two arms are level.** Over the evaluation steps both runs of a pair reached, native minus
control is +0.0124 for seed 0 (36 evaluations, up to step 18,000) and +0.0045 for seed 1 (40, up to 20,000).
Their standard errors (0.0018 and 0.0021) treat the evaluations of one run as independent, which they are not, so
they are too small; against a control spread of 0.2383 to 0.2750 these differences are nothing. Control
seed 2 shows the size of the seed effect: it did the fewest steps of the three (17,608) and ended highest.

**So the gain is the extra steps.** Native dev around a step count (mean of the nine evaluations within 2,000
steps):

| around step | native seed 0 | native seed 1 |
|---|---|---|
| 18,000 | 0.2598 | 0.2385 |
| 20,000 | 0.2643 | 0.2420 |
| 25,000 | 0.2704 | 0.2583 |
| 29,000 | 0.2824 | 0.2694 |
| 32,000 | 0.2930 | 0.2724 |
| 35,000 | not reached | 0.2743 |
| 40,000 | not reached | 0.2744 |

Seed 0 gained +0.033 between steps 18,000 and 32,000 and was still rising when its time ran out
(+0.0030 dev per 1,000 steps from step 25,000, SE 0.0006). Seed 1 gained +0.036 between 18,000 and 35,000
and +0.0002 from there to 40,000; its slope from step 25,000 is +0.0009 per 1,000 steps
(SE 0.0001), a third of seed 0's. Seed 1
did 9,969 more steps than seed 0 and ended 0.015 lower. Two seeds do not say where the curve
flattens; they say that steps beyond about 35,000 are worth less than the steps before.

**What this means for r102, the H200 gate.** r102 gives RL 18,000 s. On the 11-18 box a native run settled at
0.62 s/step on an H100 (LOG 11-19 05:26). The H200 has the same compute chip with more and faster memory, so
that is my planning number until r102 shows its own. At that speed r102 reaches about
29,000 steps, not the 35,000 to 45,000 of these B200 runs. Around step 29,000 the two native seeds here
stood at 0.2824 and 0.2694, mean 0.2759: +0.036 and +0.030 over their paired controls' last five evaluations
and +0.024 over the mean of all three controls' last five. That is near the line of 0.281, not clear of it. I wrote on
11-19 02:05 that I expect r102 at +0.03 to +0.05 over D 007; with the second seed and the H100's speed I now
expect +0.02 to +0.04 (finals between about 0.27 and 0.29), so r102 may meet the three-seed rule or miss it
by a seed. r102 also pretrains by itself (fused, 46 min), which r101 does not test. What I read first when r102
runs is its steps per second, since that fixes where on this table it ends.

**If r102 lands near the line**, the two things already queued decide the next gate config: r106
(`factored_grammar`: +8 % to +18 % steps expected, the same maths) and r105 (lr 2x and 4x in the native
regime, more dev per step if it works). Neither is a small move around a holder: both act on the one lever
that has paid, RL progress inside the budget.

**The node traces (first two read).** Both runs were on node5602-c1 from 08:49 EST, traced every 30 s
(579 and 582 rows). Hourly means, with seconds per step from each train log over the same hour:

| UTC hour (EST) | node load | my CPUs' MHz, native / control | spin ms, native / control | native s/step | native GPU use % | control s/step | control GPU use % |
|---|---|---|---|---|---|---|---|
| 13:00 (08:00) | 20 | 2,272 / 2,336 | 93 / 91 | 0.531 | 60 | 1.104 | 24 |
| 14:00 (09:00) | 17 | 2,285 / 2,353 | 95 / 93 | 0.415 | 68 | 0.877 | 39 |
| 15:00 (10:00) | 15 | 2,292 / 2,375 | 90 / 91 | 0.367 | 65 | 0.856 | 41 |
| 16:00 (11:00) | 99 | 2,141 / 2,020 | 113 / 117 | 0.371 | 60 | 1.022 | 28 |
| 17:00 (12:00) | 151 | 2,023 / 1,931 | 116 / 126 | 0.387 | 55 | 1.151 | 25 |
| 18:00 (13:00) | 155 | 2,024 / 1,903 | 118 / 124 | 0.386 | 57 | 1.153 | 24 |

The first row starts at 13:49 and the last ends at 18:40; clock times of steps are good to about a minute.

From 11:00 EST other jobs raised the node's load from about 15 to about 153. My CPUs' clocks fell and the trace's
fixed CPU spin took 34 % longer (91 to 121 ms). Taking the last quiet half hour (10:30 to 11:00 EST) against
12:00 to 13:30 EST: the vLLM control ran 35 % slower (0.854 to 1.152 s/step), nearly all of it in sampling
(0.461 to 0.705 s per step; its update +10 %), and its GPU sat idle more. The native run ran 9 % slower
(0.354 to 0.387; sampling +11 %, update +5 %). It had been getting faster by itself until then (from
0.427 s/step at 09:00 EST), so that figure may understate its loss. At their 10:30 speeds the control would have
done about 2,500 more steps (14 % of its total) and the native run about 1,900 more (4 %). The GPU's clock
stayed at 1,965 MHz in every row and no throttle flag was set, so the slowdown is on the CPU side; the trace
cannot say whether it is the CPU clocks, shared caches or memory bandwidth.

Two consequences. A vLLM run's step count depends on its node's other jobs more than a native run's does, so
a time-budgeted comparison between the two paths is tilted when a node is busy. And this does not explain
native seed 0's jump in speed at 01:20 on node5600-c1 (LOG 11-19 09:04): that was the same direction when other
jobs ended, but three times the size and mostly in the update (-31 %), which moved by +5 % here. Its cause
stays unknown. Card memory in the steady state
(`memory.used`, MiB / 1024): native 51.5 GiB, control 95.9 GiB.

**Jobs.** No GPU of mine is running. The two normal B200 workers (29644110, 29644111) have been eligible since
13:42 and wait for capacity; Slurm's estimate for both is 01:30 on 11-20, and `sbatch --test-only` puts a new
worker at 11-23 on the B200 and at 11-21 (preemptable) or 11-23 (normal) on the H200.
So the pending workers keep their place and I submit nothing. The queue is unchanged: r107 first, then r104's
second item, r106, r101's native seed 2 and r105.

## 2026-11-19 16:51 EST: INBOX 14:40, 14:29, 14:32 and 16:28 answered: standby queues staged for all three boxes (r108, r109, r110); bolder arms, fewer seeds

**Maintainer intervention (INBOX 11-19 14:29, 14:32 and 16:28; commits 388b8f5, e74a9c1 and f48eb31; logged as
such).** Three notes arrived between 14:13 and 14:32 EST, while the r101 entry was being written; I read them at
about 14:37. A fourth came at 16:28 and I read it at 16:32. The 14:40 note (commit 13a347a) is the laptop
session's own.

- **14:29, the maintainer on the AWS boxes:** "also for running screens on aws i feel like you can be a bit more
  aggressive and use less seeds?", "i think its better to make more bold changes in general", "wait speed work
  should be done on engaging. anything else that might be useful can be run on aws." The laptop withdraws point 2
  of its 01:32 note ("fewer arms, more seeds") in favour of this; writing each arm's expected effect and its
  control's noise before it runs stays. The first two remarks are filed as U21. I adopt them: the rule is below.
  The note also asks for U20's verdict in the ideas status file: written at 15:29.
- **14:32, my 10:47 question (the 11-28 box as the gating screen):** "i think this is too early to be talking
  about it, but its fine to just let the agent do whatever as long as it is reasonable". So the decision is mine,
  made when the time comes, with the reasoning written here. The maintainer's leaning is speed work on Engaging.
  Today nothing needs deciding: r102 stays the gate on Engaging, and the standby below holds no end-to-end
  speed-stack screen, so either decision still works. QUESTIONS 10:47 is marked answered.
- **14:40, the laptop session:** a box that starts while I am down must find work, so stage a standby queue for
  each block now and replace it on the planning day. Done today for all three blocks; the rest of this entry says
  what is staged and why.
- **16:28, the maintainer:** "in general unless proven otherwise itd be wise toa ssume aws and engaging run at
  diff speseds". Adopted as a rule: a timing on a box, or a result from a time-budgeted run there, is not carried
  over to Engaging (or the reverse) until the same item has been timed on both. The standby fits it: each arm is
  read only against the control on its own box, and a winning arm is re-run and then gated on an Engaging H200,
  where its steps per second are read afresh. One number below mixes machines and is marked: the control's
  expected score takes its step count from the 11-18 AWS H100 box and its score at that step count from r101 on an
  Engaging B200.

**The seeds rule for box screens (U21).** I take the seed sd of a native RL-only run's final dev as 0.019: that is
r101's three B200 control seeds (0.2483, 0.2383, 0.2750). D 007's five H200 dev seeds had sd 0.0038, so 0.019 is
the cautious figure. Every arm on a box is read against the same five-seed control mean.

| seeds per arm | sd of (arm mean - control mean) | expected best of nine arms that do nothing |
|---|---|---|
| 1 | 0.021 | +0.028 |
| 2 | 0.016 | +0.020 |
| 3 | 0.014 | +0.016 |

The last column is the expected largest of nine standard normals (1.485) times 0.019/sqrt(seeds). The control
mean's own error (0.0085) moves all nine differences together and does not change which arm comes out on top. An
effect of 0.04 to 0.05 shows in one or two seeds (r101's two pairs were +0.0550 and +0.0500), and that is the size
an arm should be able to reach to get box time. The rule:

- a bold arm gets three seeds that finish, against a five-seed control (two seeds where the queue cannot fit
  three);
- the top two or three arms are re-run on new seeds before anything is built on them, because the best of nine is
  partly luck (+0.016 expected from nothing at three seeds);
- an arm whose three-seed mean is below the control's is closed;
- each arm's expected effect and the control's sd are written before it runs (the table below);
- timing work stays on Engaging.

**The standby: r108 (11-25, 8x H100), r109 (11-28, 8x H200), r110 (12-09, 8x H100).** The same ten configs, in one
batch per box so that pulled results do not land on each other and arms are compared within one card type. All are
track D, RL only from D 007's five pretrained checkpoints (seed N starts from D 007's seed N), on r101's native
config (native sampler, shared prompt, 65,536-token micro-batches, reused allowed sets), 17,400 s, dev ladder
only, no_save. Configs: commit ca72b2f. Plan files: 849c85c.

Why arms that vary RL settings (INBOX 11-01 23:30 asks the LOG to say so when I queue them). First, the gain
measured this week is in RL steps: r101's +0.05 came from about twice the steps. The size of self-imitation, the
lr and its schedule, the group shape and the level mix were last tuned when sampling was the expensive part and a
run had half the steps. Second, a standby must run with nobody awake. An RL-only item from an existing checkpoint
saves its state every 60 min and carries on after a lost instance, and that route has run on a box. Arms that need
their own pretraining (U17, U19, the pretraining/RL split) go through the end-to-end route, which has never run on
a box and saves no state there. They join on a planning day if that route is ready.

Why all track D. INBOX 11-10 10:40 asks for each box to be split about 50/50 between the tracks ("which screens is
your call"). The standby does not do that, on purpose. The 11-18 box's track A half (r85) was three small moves
around A 007 that three seeds could not read, and I have no track A arm ready today that could move the score by
0.04. INBOX 10-31 puts track D first. Track A is a question for each planning day (11-24, 11-27, 12-08): a change
on A that could move the score by 0.04 gets its half of the box.

| arm | change against the control | seeds that finish in 24 h | expected final dev against the control (written before any run) |
|---|---|---|---|
| ctl | r101's native config, unchanged | 5 | final about 0.27 to 0.29 on an H100 (see below); the H200's step count is not measured |
| long2x | 34,800 s of RL instead of 17,400 s | 2 | +0.02 to +0.04; under +0.01 would say more steps alone are used up. Not a record config |
| sil256 | 256 replayed rows a step instead of 64 | 3 | -0.05 to +0.04; about 1.6 times the update rows late in a run, so it costs steps |
| silc2 | self-imitation coefficient 2.0 instead of 0.5 | 3 | -0.05 to +0.04 |
| mini3 | the mini levels M2, M3 and M6 join training (r56's setting) | 3 | -0.01 to +0.03, centre +0.01 |
| hard2x | L2 to L5 drawn twice as often as L1 and L6 | 3 | -0.01 to +0.03 |
| g32p16 | 16 prompts a step in groups of 32 (64 x 8 in the control) | 3 | -0.02 to +0.03 |
| lr12cos | lr 12e-6 with cosine decay to a tenth (6e-6 constant in the control) | 3 | -0.02 to +0.04 |
| lr24cos | lr 24e-6 with cosine decay to a tenth | 2 | -0.05 to +0.04; may collapse |
| stack3 | mini3 + sil256 + lr12cos | 3 | -0.03 to +0.06 |

The control's expectation: on the 11-18 AWS H100 box the native sampler settled at 0.62 s a step, which gives
about 28,000 steps in 17,400 s, and r101's two native seeds on an Engaging B200 stood at 0.2824 and 0.2694 around
step 29,000. The step count is from one machine and the score at that step count from another, so this is a
cross-machine estimate (INBOX 16:28): good for telling a normal control from a broken one, no more.

Where the ranges come from. mini3: r56 was +0.019 paired at 15,000 vLLM steps without self-imitation, all of it at
L2. With the mini levels the replay buffer holds nine level pools instead of six and draws from the pools evenly,
so about a third of the replayed rows in mini3 and stack3 are mini-level solves. hard2x: L1 passed 0.84 late in
r101's native seed 0 and L2 to L5 hold the room; dropping L1 altogether cost 0.020 in r45. g32p16: groups of 16
with 32 prompts tied 64 x 8 in r28 and r35, before self-imitation and the shared prompt; this goes one step
further. sil256: late in r101's native seed 0 about 30 of the 64 groups were informative, about 239 rows plus 64
replayed; with 256 replayed rows an update has about 495.

By the rule above mini3, hard2x and g32p16 are the weak ones: their upper ends are +0.03, under the 0.04 an arm
should be able to reach. They are in the standby because mini3 is a part of stack3, and the other two are one-line
changes to the level mix and the group shape that I could vouch for today. They are the first to be replaced on a
planning day.

**Queue and packing.** Each queue has 47 lines: two long2x seeds, the control's five seeds and five seeds of each
of the other eight arms. The seeds that finish first differ between the boxes (0, 1, 2 on 11-25; 2, 3, 4 on 11-28;
3, 4, 0 on 12-09; the control always runs 0 to 4). One run per GPU (`PER_GPU=1`, eight workers). On an H100 a
native run at 65,536-token micro-batches peaked at 74.2 GB before its first evaluation and settled at 51.7 GB (LOG
11-19 05:26), so two do not fit 80 GB. Two would fit an H200's 141 GB on paper (125.9 GB) only if their starts
were staggered, and `box_workers.sh` staggers worker index levels, not GPUs. A record config runs alone on its
H200, so one run per card is also the like-for-like comparison. Two per H200 is a question for 11-27.

Expected course, on my estimates of about 5.1 h for a 17,400 s item (training plus loading and evaluations) and
about 9.9 h for the 34,800 s one: lines 1 to 8 start together; at about 5.1 h six GPUs take lines 9 to 14; at
about 10 h all eight take lines 15 to 22; at about 15 h lines 23 to 30; at about 20.3 h the tail, lines 31 to 38,
which the block's end cuts after about 2.5 to 3 h. `RESERVE_MIN=90` lets the tail start (INBOX 11-18 11:47:
keeping all eight GPUs busy to the end comes first). A cut run still has a dev evaluation every 500 steps in
`eval_log.jsonl`, pushed every 5 min, and can be read at matched steps. Finished in 24 h: 30 runs (five control
seeds, two long2x, three seeds of seven arms, two of lr24cos). Lines 39 to 47 are spares for slots freed by an arm
that fails.

**Left out, and why.** `oversample` 2 or 4 and 128 prompts a step: with `skip_zero_advantage` the uninformative
groups already cost nothing in the update, r18 found bigger batches no better, and sampling 1,024 or more
sequences at once risks the 80 GB card. Groups of 16 with 64 prompts: memory unknown on an 80 GB card. 32 prompts
a step: acts like a higher lr, which r105 covers on Engaging. A process reward: r49 was inside the noise.
Temperature, the entropy floor and shaping: flat before. The LM loss: closed flat (r74), and it needs the 4.64 GB
token file on the box. End-to-end arms (the pretraining/RL split, U17, U19): see above.

**Not yet run on a GPU: r111.** `sil_rows` 256, groups of 32 with 16 prompts, the cosine schedule and
`train_ladder_levels` have never run with the native sampler. The mini levels' three-number puzzles with reused
allowed sets are the least certain. Card memory is unmeasured for these arms; by design the 65,536-token
micro-batch bounds the update. So five 90 s smoke runs are queued on the B200 behind the two long items (lines 9
to 13 of `queue/screens_rl_b200.txt`): `smoke_mini3`, `smoke_sil256`, `smoke_g32p16`, `smoke_lr12cos` and
`smoke_stack3`, each the r108 arm's config with `time_budget_s` 90 and a node trace every 5 s. What I read from
each: that it reaches its final evaluation without an error, the peak card memory (a B200 reading, so only
indicative for an 80 GB card), `lr` falling in the two cosine runs, and steps per second (for Engaging only; by
INBOX 16:28 it says nothing about the boxes). The score means nothing. Not smoked: silc2 (a coefficient), lr24cos
(lr12cos's path), hard2x (mini3's path; a level named twice passes straight through `_train_streams`, read in the
code) and long2x (the control with more time).

If a smoke run fails, a fixed config goes into a new batch and the boxes are staged again. If an arm fails on a
box all the same: `box_runner.sh` retries it from its saved state, the keeper retries a FAILED item once, and the
worker moves on to the next line. An arm that fails at once uses up its five lines, and the queue still holds 42
lines for the block's 38 slots.

**Staged and checked.** `box_s3.sh stage` ran for the three box names at commit 849c85c at 20:22 to 20:23Z (15:22
EST): 516 files in `code.tgz`, 47 items in one queue, 6 paths in `home.txt` (the tokenizer and D 007's five
pretrained checkpoints). The first `check` runs (15:31 and 15:33) failed only on the five checkpoints not being in
S3. The upload is a CPU job, and the one I had submitted to `mit_normal` showed a planned start of 11-26 01:00,
after the first box starts. `sbatch --test-only` put a 15-minute job on `mit_quicktest` within a quarter of an
hour, so the same upload went there as job 29681585 and ran from 15:47:28 to 15:47:49 EST (30 objects,
1,113,502,953 bytes). The `mit_normal` copy (29678885) is cancelled. `check` at 15:49 and again at 16:16 EST reads
"check ok" for all three boxes. `docs/research/box_1128.md` now has the table of what is staged and the `check`
output.

Lesson: a small CPU job goes to `mit_quicktest` when `mit_normal` shows a far start, and I read a job's planned
start right after submitting anything a deadline depends on.

Not shown yet: the self-running route (fetch, venv build, workers on real GPUs) has still not run on a GPU
instance. The 11-25 box is its first. Each stage is replaced on its planning day.

**What a standby result is for.** A winning standby arm is not a gate. It is re-run on new seeds, then goes into
an H200 gate config together with the speed stack. Where that gate runs I decide then (INBOX 14:32); today's
answer would be Engaging, and by INBOX 16:28 a box's step counts are not assumed to hold there.

**If I am down through 12-09.** The standby runs unchanged on all three boxes. The two H100 boxes (the same AWS
instance type; each read against its own control first) then give the seven three-seed arms all five seeds, seed 0
twice (0, 1, 2 on 11-25 and 3, 4, 0 on 12-09), lr24cos and long2x four seeds each (0, 1 and 3, 4), and ten control
runs.

**AWS charge (by list prices; the bill is not read).** D 007's checkpoints in `box/home/`: 1.04 GiB at $0.023 per
GB-month is $0.024 a month, about $0.07 over the 90 days until they expire. The upload's requests and the three
stages (about 0.9 MB each) are under a tenth of a cent. Running total: about $0.39, at most about $0.50 (before:
$0.32 and $0.42).

**Jobs (16:51 EST).** No GPU is running. 29644110 and 29644111 (`mit_normal_gpu` B200, 6 h, queue
`screens_rl_b200`): both pending, no planned start (reason `None`). Slurm's planned starts for the two have moved
all afternoon (01:30 on 11-20 when read at 14:35, 15:56 at 15:46, 17:16 and 17:25 at 16:16, none from 16:23), so I
plan nothing on them. 29632479 (preemptable B200, 14 h): pending, no planned start (reason `Resources`). r102's
three H200 workers (29450693/94/702): all three pending, no planned start (reason `Resources`). Nothing more is
submitted: the pending pair would fill the cap's two normal slots. The queue holds 20 lines: the five r111 items
were added at lines 9 to 13 and the rest is unchanged.

## 2026-11-19 21:02 EST: INBOX 20:45 answered: the next two pairs of B200 workers are queued now so that they gain age (29715907 to 29715910); a dependency stops age

**INBOX 11-19 20:45 (laptop session, operational; a suggestion).** The note explains why my Engaging jobs wait: a
job's priority is partition + fair share + age, my fair-share part is 1,287 of 150,000, and age (up to 50,000
after seven days pending) is the one part I control. It suggests submitting the next pair of workers while the
current pair runs or waits, because Slurm's per-user limit holds them. Adopted, with one correction to my own
practice (the third point below). No GPU of mine has run since about 14:35 today, so the wait for a card is the
main limit on the work now.

**What I checked (read-only, 20:55 to 21:00).**

- The per-user limits are Slurm's own: 2 GPUs, 32 CPUs and 515 GB on `mit_normal_gpu` (64 submitted jobs), 4 GPUs
  on `mit_preemptable` (256 submitted jobs). Together they are the cap of six running GPUs, so extra pending
  workers cannot break it. Two 16-CPU workers are exactly the 32-CPU limit.
- The weights are as the note says: fair share up to 150,000, age up to 50,000 over 7 days, job size up to 10,000,
  and the partition's part (360,000 normal, 30,000 preemptable). Age grows by 298 an hour, 7,143 a day.
- `PriorityFlags` is empty, so a job gains age only while it is free to run: a held job, or one waiting on a
  dependency, gains none. The pending pair shows it. They were submitted at 09:23:27 with `--dependency=afterany`
  on the two workers then running (LOG today, jobs at 09:28), and their `AccrueTime` is 13:43:47, when the
  dependency cleared: 4.3 h of age lost, about 1,291 points.
- `sprio` at 21:02: the pair stands at 363,444 (age 2,153, fair share 1,287, partition 360,000) and the four new
  workers at 361,291 (age 0). The preemptable workers have age 4,398 (29632479) and 5,669 (r102's first) on a
  partition part of 30,000; their reason is `Resources`, not `Priority`.
- My own `sbatch --test-only` at 18:54 (nothing submitted) put a new normal worker at about 01:28 on 11-24 on a
  B200 and 01:08 on an H200, 103 h away, the same with 8 CPUs or a 2 h limit. That agrees with the note's test:
  the worker's size, time limit and card are not levers.

**Submitted at 20:58: 29715907, 29715908, 29715909 and 29715910.** They have the pending pair's form
(`mit_normal_gpu`, one B200, 6 h, 16 CPUs, 128 GB, queue `screens_rl_b200`, `RESERVE_MIN=325`) and no dependency.
Why four: the queue's 20 lines hold 7 long items (about 5.1 h each). A 6 h worker takes the short items and then
one long one; the pending pair takes two and the 14 h preemptable worker at most two, so at least three more
worker starts are needed, and four makes two full pairs. A worker that starts on an empty queue exits at once and
loses only its place.

**What to expect (not measured).** By the note's reading of the B200 line, the heavy users' jobs sit at 362,000 to
370,000 and the light users' at 506,000 to 520,000. A day of age lifts a worker over most of the first group and
two days over all of it; no age reaches the second group. So queueing early shortens the wait behind the heavy
group only. The check is each worker's time from submission to start, which I log as they start.

**Practice from now.**

- Early workers carry no `--dependency`; the per-user limit does the holding. Still to see: that a worker held
  only by that limit keeps gaining age (the note saw one such job at the top of the line). I read `sprio` for it
  once a pair of mine runs.
- Two pairs stay pending behind the running or next pair on `mit_normal_gpu` while the queue has work for them,
  topped up at the wake where a pair starts.
- Not changed: the workers' 16 CPUs and 128 GB (timed arms are compared within one worker shape, and a smaller
  worker starts no sooner); the preemptable side, where each of the cap's four slots already has a pending worker;
  the held jobs (r97/r99 and r90, superseded). Fair share is charged by CPUs and most of the account's usage is
  not this project's (the note), so that part stays as it is.

**Jobs (21:02 EST).** No GPU is running. Normal B200 workers: 29644110 and 29644111 both pending, planned start
11-20 11:30; 29715907 to 29715910 all four pending, no planned start (reason `None`). 29632479 (preemptable B200,
14 h): pending, no planned start (reason `Resources`). r102's three H200 workers (29450693/94/702): all three
pending, no planned start (reason `Resources`). The queue is unchanged at 20 lines; the four held jobs stay held.

## 2026-11-19 23:29 EST: why no GPU has started since 14:35: the cluster is full and my jobs wait on age; two stale node exclusions dropped; all nodes are reserved for maintenance on 12-15

Full wake. INBOX is unchanged since 20:45 (answered at 21:02). No job of mine is running; nothing finished or
failed. The message of the day has no notice. Quota at 23:00: pool 64.47 % of 1 TB, home 47.06 %. Since no GPU of
mine has run for almost nine hours, I looked at why (all read-only, 23:05 to 23:10) and whether anything but
waiting helps.

**What the cluster looks like.**

- B200 (`mit_normal_gpu`): of 12 nodes, 3 are out of service and 2 are inside a reservation of other users until
  10:00 on 11-20. Of the seven B200 GPUs that `sinfo` showed as free at 23:05, six were on those two reserved
  nodes, which are not open to me, and one was on an open node. 7 nodes are open to normal jobs now, and 7 after
  10:00 on 11-20 (one node comes back, another is reserved until 11-22). One more node is reserved from 11-30 to
  12-07.
- H200 on `mit_normal_gpu`: 72 GPUs, none free. Pending there with reason `Priority`: 9 B200 requests (6 of them
  mine), 132 for an H200, 103 for an L40S, 45 for an H100 and 78 for any GPU.
- H200 on `mit_preemptable` (r102's line): one GPU free at 23:09. 182 pending requests could run on an H200 there;
  74 of them stood above my three gate workers (37,617 then), the highest at 49,839. On B200 the same count is 7
  of 24 above 29632479's line. My first count of free preemptable H200s (8) was a parsing error from narrow
  `sinfo` columns; the re-count with wide columns gave one.

**How the scheduler treats these jobs (from `scontrol show config`).**

- `bf_min_prio_reserve=50000`: as I read Slurm's documentation, the scheduler holds a place (a planned start) only
  for jobs at priority 50,000 or more. Every normal job is above that (partition part 360,000). My preemptable
  workers are below it (37,716 for r102's three, 36,445 for 29632479), so they never show a planned start: they
  start only when a GPU they may use is free at the moment the scheduler tries them. With age alone r102's workers
  reach 50,000 on 11-21 about 16:00 (41 h from now), and from then they should be given a planned start like the
  normal jobs. That is the clearest date I have for the gate, and it is not soon.
- Preemption is by partition (`preempt/partition_prio`, mode `REQUEUE`), as known: a preemptable run is requeued
  when a normal job needs its GPU.
- `sbatch --test-only` (nothing submitted) gave one answer for a new normal B200 worker, 05:22 on 11-24, at every
  time limit from 30 min to 6 h and with half the CPUs and memory; an L40S, an H200 or an H100 came out later, and
  every preemptable request at 11-22. So no other card, shorter limit or smaller worker gets a job in sooner (the
  20:45 note found the same). The estimate itself is four days later than the planned starts my pending workers
  have shown tonight, so I do not plan by it.

**First change: the exclusion of `node5501-c1` is dropped on the seven B200 workers.** The node was excluded after
a GPU failed on it on 10-29 (LOG 10-30). At 23:07 all eight of its GPUs were allocated, to jobs that had run from
6 minutes to 23 hours, so it reads as repaired. I set `ExcNodeList=node5500-c1` (the drained node stays excluded)
with `scontrol update` on 29715910 first: its `AccrueTime` and age were unchanged. Then on 29644110, 29644111,
29715907, 29715908, 29715909 and 29632479. My workers can now use all 7 open B200 nodes instead of 6. The risk is
small: on a bad GPU the worker puts its item back at the front of the queue and exits (it did so on that node on
10-30), which costs one worker's place and no item. If it happens the exclusion goes back. New B200 workers are
submitted with `--exclude=node5500-c1` only.

**Planned starts of the six normal workers.** At 21:02 only the pair had one (11:30 on 11-20). At 23:08, after the
change: 11:30, 11:30, 14:30, 14:30, 16:00 and 17:30. At 23:17: 05:00, 05:00, 07:00, 10:00, 11:00 and 11:00. At
23:22 and 23:24: none (reason `Priority`). At 23:29: 06:30, 07:00, 10:00, 11:00, 11:30, 11:30. All times are on
11-20. They come and go with every pass of the scheduler, so I read them as rough and do not credit the earlier
times to the change.

**Second change: node5200 comes off the exclusions of r102's H200 workers, after a probe.** The three workers
excluded 50 of the 258 preemptable H200 GPUs that are in service. 18 are on AGENT.md's list. 32 were my additions,
four nodes of eight GPUs: node2500 and node4900 ran my jobs slowly (node2500 about 3x on 10-30) and stay excluded
because the gate is a timed run; node5100 and node5200 hung on reading the venv on 11-14 (LOG 11-14). I tested
those two with probe jobs 29727371 (node5100) and 29727372 (node5200): one CPU and at most 6 minutes each on
`mit_preemptable`, script `agent/probe_venv_1119.sh` (a `stat` of two venv files and of one file in home, then one
`import torch`, each under a timeout). Both started at 23:18, 111 s or less after submission. node5200 passed: the
venv files were read in 0.008 s or less and `import torch` took 28 s. node5100 failed: both venv files timed out
at 30 s while the file in home took 0.003 s, and the import was killed at its limit (155 s). So the pool
filesystem still hangs on node5100, and that node stays excluded. node5200 is off the three workers' list since
23:23 (`AccrueTime` and age unchanged), which leaves 42 of 258 excluded. The worker script also guards against
this case since 11-14: on a node where the venv cannot be read within 60 s it exits holding no item.

**Maintenance on 12-15.** A reservation covers every node from 00:00 to 21:00 on 2026-12-15 (the message of the
day does not announce it yet). Nothing runs then, and a job whose time limit reaches past 00:00 on 12-15 does not
start before it: a 6 h worker must start by 18:00 on 12-14, a 12 h or 14 h preemptable worker by 12:00 or 10:00
(later if its `--time-min` lets it shrink). A record's five seeds must be planned around that day; with the run's
end on 12-19 it leaves four days after it.

**What this means for the plan.** On Engaging, age is the only lever and it is already in use: six normal workers
(age 2,897 and 720), one preemptable B200 worker (5,142) and the three gate workers (6,413). Nothing else I could
find starts a job sooner. The three paid boxes (11-25, 11-28, 12-09) are therefore most of the GPU time this run
can count on, and their queues deserve the idle wakes: before the planning day of 11-24 I prepare bold end-to-end
track D arms (U17, U19) to replace part of the RL-only standby, which needs the pretraining token file in
`box/home/` (it goes up with the `home` job once such an arm is in a queue). Whether the 11-28 H200 box should
also carry the gate stays my decision for 11-27 (INBOX 14:32); tonight's reading of r102's line is a fact for it.

**Jobs (23:29 EST).** No GPU is running. Normal B200 workers 29644110, 29644111 and 29715907 to 29715910: pending,
planned starts as above. 29632479 (preemptable B200, 14 h) and r102's three H200 workers (29450693/94/702):
pending, no planned start. Both probes are finished. The queue is unchanged at 20 lines; the four held jobs stay
held.

## 2026-11-20 01:21 EST: r112 queued: seconds per RL step by model size with the native sampler; the B200 queue put into short blocks so that a worker starting alone still gets a long item

**State.** No GPU of mine is running (none since about 14:35 on 11-19); every job is pending. INBOX is unchanged
since 11-19 20:45. This entry is design work in an idle wake (LOG 11-19 16:51): it queues one measurement and
changes the order of the B200 queue. It holds no new result.

**Why model size is open again.** r50, r52 and r53 closed model size (PLAN D5): against D 005's 768x12 model,
640x10 was at -0.008, 1024x12 at +0.002, 1280x12 at +0.004 and 768x24 at +0.007. All four ran through vLLM at
11,000 to 14,000 RL steps, where sampling was about 70 % of a step (PLAN D7), so a smaller model bought few extra
steps and a bigger one lost few. The native sampler changed that. In r101's two native runs (B200, RL only from D
007's checkpoints) the update is the larger of the two timed parts of a step. The update is the model's forward
and backward pass, so it grows with the model.

| seed | part of the run | `t_sample` s | `t_update` s | the update's share |
|---|---|---|---|---|
| 0 | steps 50 to 400 | 0.187 | 0.336 | 64 % |
| 0 | the middle 1,000 steps | 0.166 | 0.289 | 63 % |
| 0 | the last 1,000 steps | 0.138 | 0.168 | 55 % |
| 1 | steps 50 to 400 | 0.176 | 0.313 | 64 % |
| 1 | the middle 1,000 steps | 0.135 | 0.176 | 57 % |
| 1 | the last 1,000 steps | 0.141 | 0.201 | 59 % |

The whole runs made 2.01 and 2.58 steps per second; at steps 50 to 400 the two timed parts alone allow 1.91 and
2.05. Dev was also still rising when those runs ended (mean of the evaluations in each window of 5,000 steps: ten,
or nine in a run's last window):

| seed | steps | 0-5k | 5-10k | 10-15k | 15-20k | 20-25k | 25-30k | 30-35k | 35-40k | 40-45k | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 34,893 | 0.152 | 0.216 | 0.248 | 0.259 | 0.269 | 0.275 | 0.294 |  |  | 0.3033 |
| 1 | 44,862 | 0.148 | 0.206 | 0.225 | 0.237 | 0.249 | 0.267 | 0.273 | 0.274 | 0.281 | 0.2883 |

So with the native sampler more steps still buy score, and the model's size sets a large part of the time a step
takes. How far the step rate moves with size has not been measured in this regime.

**r112.** Six 90 s smoke runs on a B200 in r111's form: RL only, native sampler, D 007's RL settings (64
self-imitation rows), `no_save`, a node trace every 5 s, and one final dev evaluation whose score means nothing.
They start from checkpoints that already exist, all pretrained with D 005's recipe (AdamW, 112 min), so nothing is
pretrained for this:

- `ctl_w768_l12`: D 005's own 768x12 model (110M), seeds 0 and 1: the control;
- `w640_l10`: r50's 640x10 (69M), seeds 0 and 1;
- `w1024_l12`: r50's 1024x12 (182M), seed 0;
- `w768_l24`: r53's 768x24 (about 195M), seed 0.

Each config differs from r101's native config only in the checkpoint it starts from, the pretraining block that
describes that checkpoint, and the 90 s budget (checked by the script that wrote them). I read `t_sample`,
`t_update` and steps per second over steps 20 on, and the peak card memory from the trace. The control and 640x10
run twice, once in each of two blocks of the queue, so each block has its own pair on one node. Each run should
cost a worker about 3 minutes, taken from the minutes before its long item.

**Expected, written before it runs.** The compute for a token is about proportional to the parameter count, less
the parts of a step that do not depend on the model. Against the control's steps per second on the same node I
expect 640x10 at 1.25x to 1.5x, 1024x12 at 0.6x to 0.75x and 768x24 at 0.55x to 0.7x. The two bigger models may
run out of card memory at 65,536-token micro-batches; that would be their answer. They are in the batch to fix the
slope and to read memory. I do not expect an arm from them.

**Rule.** 640x10 at 1.4x or more: size moves the step count for real here, and a small model gets seeds: 640x10
against 768x12 from these same checkpoints (RL only, native, 17,400 s on the B200), and a 512x8 model (41M), which
has no checkpoint and so needs a pretraining item first. 1.2x or less: size stays closed with the native sampler
too, and the work goes to the parts of a step that do not depend on the model. In between: decided with the memory
reading.

**What a win would be worth.** The control's seed sd on dev is 0.019. In the table above, seed 1's dev rose by
0.008 from the 30-35k window to the 40-45k window (about 30 % more steps), and 640x10 was 0.008 below 768x12 at
equal time in r50. So I expect 640x10 at -0.01 to +0.02 against the control, inside the noise: by the seeds rule
of LOG 11-19 16:51 it is not a bold arm by itself. A 41M model at about twice the steps is the bold form, -0.03 to
+0.04, never measured. r112 is the cheap reading that says whether that pretraining item is worth a slot.

**Limits.** A 90 s run reads the start of a run. In r101 a step got faster and the update's share fell as the run
went on (first table), so a ratio at the start is a first reading only. Timings are compared within one node and
one card type, and none is carried over to the AWS boxes (INBOX 11-19 16:28).

**The queue order.** The old order was written for workers that start in pairs. The six normal workers' planned
starts now differ by worker (05:30, 05:30, 10:30, 11:30, 11:30, 11:30 on 11-20), so a worker may start alone. A
worker takes items only in its first 35 minutes (`RESERVE_MIN=325` in a 6 h job). One that is still on short items
at minute 35 exits without a long item, and the GPU it waited a day for is lost. By my estimates the old order had
about 26 and 25 minutes of short items before a long one, and r112 would have added to that. The new order (26
lines) is in blocks, each a few short items and then one long item (an RL-only run of 17,400 s, about 5.1 h):

- lines 1-2: r107 (memory of fused pretraining) and r104's second item, about 15 min; line 3: r101's native seed 2
  (long);
- lines 4-7: r106's four final-weights timing items, about 13 min; line 8: r105 lr 12e-6 seed 0 (long);
- lines 9-13: the five r111 smoke runs, 10 to 15 min; line 14: r105 lr 24e-6 seed 0 (long);
- lines 15-19: r106's two start-weights timing items, then r112's control, 1024x12 and 640x10 (seed 0 each), about
  19 min; line 20: r105 lr 12e-6 seed 1 (long);
- lines 21-23: r112's control (seed 1), 768x24 and 640x10 (seed 1), about 8 min; lines 24-26: r105's last three
  items (long).

The times are my estimates from earlier items of the same kind on a B200 (r96 and r100: 3 to 5.5 min each; r104's
first item 7.4 min); the 90 s smokes have not run yet. Two more rules shaped the order. r106's four final-weights
items stay together, because they are an ABBA pairing on one node. And a worker that sees two failures of under
300 s in a row puts the item back and exits, so each bigger-model smoke sits between two items I expect to pass.
No worker was running, so the change touched nothing in flight. The seven long items match the seven pending B200
workers.

**Jobs (01:21 EST).** No GPU is running. Normal B200 workers 29644110, 29644111 and 29715907 to 29715910: pending
at priority 364,746 and 362,569, planned starts as above. 29632479 (preemptable B200, 14 h) and r102's three H200
workers (29450693/94/702): pending at 36,991 and 38,262, no planned start. The four held jobs stay held.

## 2026-11-20 04:12 EST: the first B200 pair started 13 h after submission; r107: fused pretraining at micro-batch 128 holds 51.6 GiB; r104's second item: the samplers agree at the pretrained weights too; r106: `factored_grammar` gives no speed at the final weights

**Full wake (03:25 to 04:12 EST), because both workers of the first pair started and six short items finished.**
The message of the day has no maintenance or outage line. Quota (03:04): home 94.1 of 200 GB, pool 660.4 of 1,024
GB (64.5 %). INBOX is unchanged (e66970b, answered). No item failed.

**The first pair started, 38 minutes apart.** 29644110 started at 02:30:07 on node5600-c1, 12 h 46 min after it
was submitted (13:43:47 on 11-19), and 29644111 at 03:08:16 on node5602-c1, 13 h 24 min after. So each started
alone, which the 01:21 entry allowed for, and the block order of the queue did its job. 29644110 ran r107 (7.7
min, of which 133 s was the first warm start) and r104's second item (13.4 min; I had estimated 8), and took
r101's native seed 2 at 02:51, 21 minutes into its 35-minute window. 29644111 ran r106's four final-weights items
(11.7 min in all; warm starts of 6 to 8 s) and took r105's lr 12e-6 seed 0 at 03:20, 12 minutes in. The two long
items should end about 08:00 and 08:26.

**A worker that waits only on the per-user limit keeps gaining age (the open point of INBOX 11-19 20:45).** Since
03:08 the four pending normal workers (29715907 to 29715910) wait with the reason `QOSMaxCpuPerUserLimit`: the
partition allows 32 CPUs per user and the running pair holds 16 each. Their age part read 1,885 at about 03:25,
1,935 at 03:33 and 2,112 at 04:12 (priority 363,403). Had it stopped when the limit began to bind, it would still
read about 1,814. That is the usual 298 an hour. So workers queued early behind a running pair do build age, and
the practice of 11-19 21:02 stands as written.

**r107 (card memory of fused pretraining at micro-batch 128).** `calibration/research_r107/pt_fused128_mem` seed 0
ran alone on a B200 (node5600-c1) from 02:30 to 02:37 for 300 optimizer steps, with the node trace every 5 s.
`memory.used` rose to 52,866 MiB and stayed there: all 40 readings with the GPU more than half busy show 52,866
MiB (51.6 GiB; 55.4 GB in decimal units). Between steps 50 and 300 it ran at 749k tokens a second (r92 on a B200:
729k) and the train loss fell to 3.14 by step 300 (157M tokens), so this was the real pretraining path and not an
idle allocation. The rule (LOG 11-19 13:01) expected 50 to 90 GB and said that under about 70 GB both plans stand.
They stand: r102 keeps `--micro-batch 128` on the H200, and an end-to-end arm on the 80 GB H100 boxes (81,559 MiB)
can use it with about 28 GiB to spare. The reading is from a B200; I have not checked that an H100 allocates the
same.

**r104's second item (the two samplers on D 007's five pretrained checkpoints, before any RL).** It ran from 02:37
to 02:51 EST on the same B200 (worker 29644110; 660 s of scoring). The settings are the first item's (LOG 11-19
09:04): the 600 dev puzzles, at most 256 new tokens, the answer constraint on in both paths. Greedy, as the
in-loop evaluation runs:

| D 007 pretrained checkpoint | vLLM | native | native - vLLM | identical completions of 600 | solved by vLLM only | solved by native only |
|---|---|---|---|---|---|---|
| seed 0 | 0.0000 | 0.0017 | +0.0017 | 323 | 0 | 1 |
| seed 1 | 0.0000 | 0.0000 | +0.0000 | 383 | 0 | 0 |
| seed 2 | 0.0017 | 0.0000 | -0.0017 | 508 | 1 | 0 |
| seed 3 | 0.0033 | 0.0050 | +0.0017 | 453 | 0 | 1 |
| seed 4 | 0.0000 | 0.0000 | +0.0000 | 565 | 0 | 0 |
| mean or total | 0.0010 | 0.0013 | +0.0003 | 2,232 | 1 | 2 |

- Native minus vLLM is +0.0003 (SE 0.0006). Both pass rates are near zero, as they should be for weights that have
  had no RL: vLLM solves 3 of 3,000 puzzles and native 4.
- 2,232 of 3,000 completions are identical token for token, against 2,860 at the final weights. The 768 that
  differ part at token 4 in the median (range 1 to 224). The sign test on the 3 puzzles solved by one path only
  gives p = 1.0.
- Why so many more differ here than at the final weights: untrained weights put near-equal probability on many
  tokens, so the argmax flips on small differences in the two engines' arithmetic, and after the first flipped
  token the completions go separate ways. That is my reading and it is not checked. The sampled statistics below
  do not depend on it.
- The five checkpoints write very different completions (mean greedy length by seed, vLLM/native: 147/152,
  250/249, 22/23, 25/25, 6/6 tokens; malformed rate 0.515/0.537, 0.962/0.965, 0.000/0.000, 0.000/0.000,
  0.972/0.970), and the two paths agree on that checkpoint by checkpoint.

Sampled at T = 1, 16 completions a puzzle (9,600 per checkpoint and sampler; each cell vLLM / native; log-prob and
entropy are per sampled token under the fp32 weights without the answer mask):

| D 007 pretrained checkpoint | solve rate | native - vLLM (SE over puzzles) | tokens | distinct per prompt | log-prob | entropy |
|---|---|---|---|---|---|---|
| seed 0 | 0.0023 / 0.0011 | -0.0011 (0.0006) | 138.87 / 138.71 | 15.72 / 15.73 | -3.326 / -3.308 | 3.307 / 3.290 |
| seed 1 | 0.0002 / 0.0003 | +0.0001 (0.0002) | 208.61 / 205.78 | 16.00 / 16.00 | -3.568 / -3.560 | 3.563 / 3.555 |
| seed 2 | 0.0004 / 0.0005 | +0.0001 (0.0003) | 197.40 / 200.28 | 16.00 / 16.00 | -3.547 / -3.546 | 3.541 / 3.541 |
| seed 3 | 0.0029 / 0.0023 | -0.0006 (0.0008) | 153.58 / 154.89 | 15.95 / 15.95 | -3.359 / -3.347 | 3.343 / 3.334 |
| seed 4 | 0.0011 / 0.0011 | +0.0000 (0.0005) | 141.33 / 145.75 | 15.61 / 15.71 | -3.576 / -3.575 | 3.567 / 3.569 |
| mean | 0.0014 / 0.0011 | -0.0003 (0.0002, over checkpoints) | 167.96 / 169.08 | 15.85 / 15.88 | -3.475 / -3.467 | 3.464 / 3.458 |

- Solve rate: native minus vLLM is -0.0003 (SE 0.0002 over the five checkpoints; -0.0011 to +0.0001). The rates
  are near zero, so this says little.
- Entropy per token: native / vLLM = 0.9981 (0.9949 to 1.0005 by checkpoint), a difference of -0.0065 (SE 0.0034).
  Log-prob per token differs by +0.0080 (SE 0.0032). Native completions are +1.12 tokens longer (SE 1.25).
- So at these weights the native sampler's draws are, if anything, a touch less spread than vLLM's: 0.2 % lower
  entropy, about 1.9 standard errors over five checkpoints (2.5 for the log-prob). I do not call that zero, and I
  do not call it a finding either: there are five checkpoints, and in two of them most completions run into the
  token limit.

**What r104's second item means.**

- The rule (LOG 11-19 09:15) asked whether the T = 1 solve rate, length, log-prob and entropy per sampled token
  and the greedy pass rate agree at the pretrained weights as they did at the final ones. They do, to within 0.2
  %. The gap the test was looking for is native runs training at 5 to 9 % higher entropy after their first 100
  steps. What shows here is 0.2 % in the other direction, some 26 to 48 times smaller.
- By the rule the sampler is cleared at both ends of training, the box's equal-step gap (about 0.01 on dev) is put
  down to drift between runs, and the question is closed until r102 gives end-to-end numbers. No GPU time goes to
  it.
- Limits: one GPU type; no update step; the entropy is taken without the answer mask, so it is not the train log's
  number; and with a quarter of the greedy completions differing, a greedy evaluation of an untrained checkpoint
  cannot be compared completion by completion between the two paths (the pass rates can, and are near zero in
  both).

**r106, the four final-weights items (`factored_grammar` timed on a GPU). My expectation failed.** Four 300-step
runs from D 007's seed-0 final weights on one B200 (node5602-c1, worker 29644111), in the order control, factored,
factored, control, from 03:08 to 03:20. The only config difference is `factored_grammar`. Means over steps 51 to
300 (steps/s from `train_s` over the same steps):

| run, in the order run | `factored_grammar` | `t_part_s_mask_cpu`, mean (median) s | `t_sample` s | `t_update` s | steps/s | tokens/s | dev at step 300 |
|---|---|---|---|---|---|---|---|
| `final_native_ctl` | off | 0.0868 (0.0868) | 0.1481 | 0.2445 | 2.108 | 33,276 | 0.2517 |
| `final_native_factored` | on | 0.0903 (0.0849) | 0.1523 | 0.2386 | 2.056 | 33,390 | 0.2517 |
| `final_native_factored_b` | on | 0.0865 (0.0866) | 0.1502 | 0.2428 | 2.107 | 33,255 | 0.2517 |
| `final_native_ctl_b` | off | 0.0875 (0.0873) | 0.1509 | 0.2537 | 2.045 | 32,266 | 0.2517 |

- **Same maths: yes.** `loss`, `mean_reward`, `entropy`, `grad_norm`, the batch solve rate and the self-imitation
  loss are equal on all 300 steps in all four runs, and all four end at dev 0.2517. By the rule's test the option
  is safe.
- **Faster: no.** I expected the constraint timer to fall by a third to 70 %. It reads 0.0868 and 0.0875 s a step
  in the controls and 0.0903 and 0.0865 in the factored runs, and the whole step runs at 2.108 and 2.045 steps a
  second against 2.056 and 2.107. The two controls differ from each other by more than the arms do. Factored is
  lower only in the first 50 steps (0.0921 and 0.0921 s against 0.0993 and 0.0986); from step 51 on there is
  nothing. (The first factored run reads 0.0986 and 0.0998 over steps 101 to 200, which is why its mean is above
  its median; I take that for something else on the node and have not checked it.)
- **By the rule** (LOG 11-19 10:12: the option joins the settled settings "if the final runs match and the
  constraint time falls"), `factored_grammar` does not join the settled speed settings. It stays in the code, off
  by default.
- **Why the CPU benchmark did not carry over.** The option changes only how the allowed set of a grammar state is
  built the first time that state is seen (the cache-miss branch of `AnswerGrammar.allowed`). My benchmark walked
  the grammar at random and met mostly new states. A trained policy meets the same states again and again, so
  after the first steps nearly every call is a dictionary hit and the option has nothing left to speed up; the
  difference in the first 50 steps fits that. I had read the distance between the B200's 6.6 us per row-token and
  the benchmark's 1.3 us for hits as misses. That was a guess, and a count of hits and misses in one real run
  would have tested it before I wrote an expectation on it.
- **What the 0.087 s is, then, is not established.** For each token step the timer wraps the Python that feeds
  each row's new token to its grammar state and looks up the allowed set, and `mask_rows_cached`, which joins one
  cached index tensor per row on the GPU and masks the logits. The benchmark puts the Python part at about 0.018 s
  a step when every lookup is a hit (on the login node's CPU). Two candidates for the other 0.07 s: (a) one of the
  torch calls waits for the previous forward step to finish on the GPU, so the timer is partly forward time under
  another name and a faster constraint gains nothing; (b) overhead of the per-row torch work (hundreds of small
  GPU tensors joined at every token step), which one index built on the CPU and copied once would remove. I do not
  know which.
- **How it gets decided, at no extra GPU time.** r112's size smokes (queued) log the same timer for 640x10,
  768x12, 1024x12 and 768x24 models on one node. Read per sampled token (the checkpoints write completions of
  different lengths): if the timer grows with the model it is mostly a wait for the forward step (a); if it is
  flat across sizes it is overhead (b), and a same-maths change to `mask_rows_cached` is worth writing and timing.
  r106's start-weights pair (queued) shows whether new states matter early in training, the one place the option
  could still earn its keep. No code change before those two readings.

**Jobs (04:12 EST).** Running on `mit_normal_gpu`: 29644110 (node5600-c1; r101's native seed 2, to about 08:00)
and 29644111 (node5602-c1; r105 lr 12e-6 seed 0, to about 08:26). Each exits after its long item. Pending there:
29715907 to 29715910 (priority 363,403; the first two can start when the running pair ends). Pending on
`mit_preemptable` with the reason `Resources` and no planned start: 29632479 (B200, priority 37,825) and r102's
three H200 gate workers 29450693/94/702 (priority 39,096; 50,000 about 16:00 on 11-21). Held as before:
29497701/02/05 and 29445218. The queue has 18 lines: the five r111 smokes, then r105 lr 24e-6 seed 0; r106's
start-weights pair and three r112 smokes, then r105 lr 12e-6 seed 1; the other three r112 smokes, then r105's last
three items. That is five long items for the five pending B200 workers and it runs out about 20:00 today, so more
work is queued next in this wake, in its own entry with the expectations written first.

## 2026-11-20 04:34 EST: r113 queued (three RL-only arms on the B200 from D 007's checkpoints: sil256, silc2, stack3; expectations and rule written first); one pending worker for each queued long item

**What.** r113 is three RL-only arms with the native sampler, 17,400 s each, seeds 0 and 1, from D 007's
pretrained checkpoints. Each config is the r101 native control (`calibration/research_r101/d007rl_native_b200`)
with only the named settings changed (checked by the script that wrote them): `d007rl_native_sil256_b200` has
`sil_rows` 256 against 64; `d007rl_native_silc2_b200` has `sil_coef` 2.0 against 0.5; `d007rl_native_stack3_b200`
has three changes at once, lr 12e-6 on a cosine schedule, `sil_rows` 256, and the mini levels M2, M3 and M6 in the
training mix. They are the B200 copies of the staged box arms `d007rl_sil256`, `d007rl_silc2` and `d007rl_stack3`
under `calibration/research_r108/` (the settings are equal, also checked). Each has a 5 s node trace. The results
are B200 results, not yet gated.

**Why these three.** They are the arms of the standby table (LOG 11-19 16:51) that are not weak by that table and
are not yet on Engaging. The lr arms have their constant-lr forms in r105. `long2x` (34,800 s) does not fit a 6 h
worker. `mini3`, `hard2x` and `g32p16` were the table's weak arms, and `mini3` runs inside `stack3`. An end-to-end
item (pretraining and RL, about 6.1 h) does not fit the 6 h normal workers that share this queue, so the bold
end-to-end arms stay for the boxes. A B200 reading of these three before 11-24 lets the 11-25 box be planned
around what the arms do, not around the standby guess.

**Why RL settings again (INBOX 11-01 asks for the reason).** The native sampler about doubles the RL steps in the
budget: the control reached 34,893 and 44,862 steps against D 007's 16,318 to 19,735. `sil_rows` 64, `sil_coef`
0.5 and lr 6e-6 were chosen at about half the steps. These are 4x and 2x moves, not small ones, and each has a
written range that includes a loss. In the second half of the control's runs 30.6 and 28.6 of the 64 groups a step
had a non-zero advantage, so about 301 rows went into an update (those groups times the group size of 8, plus the
64 replayed rows). The final dev by level (seed 0, seed 1): L1 0.84 and 0.88, L2 0.26 and 0.16, L3 0.24 and 0.16,
L4 0.09 and 0.09, L5 0.08 and 0.10, L6 0.31 and 0.34. L2 to L5 are where the score is lost, and the replay buffer
(4,096 completions a level) is where solved hard puzzles are seen again.

**Expected, written before any item runs.** Final dev against the control at the same seed (the standby table's
ranges): sil256 -0.05 to +0.04; silc2 -0.05 to +0.04; stack3 -0.03 to +0.06. The control's noise: the vLLM
control's seed sd is 0.019, and the native control's two seeds ended at 0.3033 and 0.2883, on nodes that ran 2.01
and 2.58 steps a second. Cost per step, an estimate only: sil256 takes an update from about 301 to about 493 rows,
and if the update time grows with the rows (the control's timers read 0.149 s of sampling and 0.213 s of update a
step) a step costs about 38 % more, so sil256 and stack3 would end about 27 % short of the control's steps on an
equal node. silc2 changes a weight only and should cost nothing per step.

**Rule for reading.** Each arm seed is read against the control's same seed two ways: at matched steps (block
means of ten dev evals), and at the final, with the steps and the node of both runs. The B200 nodes differ in
speed, so the matched-steps reading judges the setting and the final is the score the budget gives on that node.
For sil256 and stack3 a gain at matched steps is not enough: it has to be larger than what the control gains over
the steps the arm loses, read from the control's own curve. An arm that is +0.02 or more at the final in both
pairs, and ahead at matched steps too, gets a third seed here and more seeds on the 11-25 box, and is a candidate
for the next H200 gate config. An arm below the control in both pairs, both ways, is closed and its box arm is
replaced on the planning day. Anything between stays in the box queue and the box's seeds decide. If the
`smoke_sil256` or `smoke_stack3` run of r111 fails, the matching r113 items are pulled before a worker pops them;
silc2 has no smoke because it changes a weight and no code path.

**Queue and workers.** The six items were appended at 04:21 EST (seed 0 of each arm, then seed 1), so
`screens_rl_b200` has 24 lines and 11 long items: r105's five, then r113's six. If each pair starts as the one
before ends, the queue lasts to about 14:00 on 11-21; the first pair waited 13 h, so it may last longer. Six more
normal B200 workers were submitted at 04:21 EST, in the same form as the four of 11-19 20:58 and with no
dependency: 29755538 to 29755543 (priority 361,351 at age 60; the older four are at 363,527). All ten wait on the
per-user limit and gain age there. **This changes the practice of 11-19 21:02** (two pairs pending): now one
pending worker for each queued long item, submitted when the item is queued. A pending job costs nothing and takes
no node, a worker that starts on an empty queue exits at once, and with waits of 13 h the age gained early counts
against jobs submitted later (it does nothing against jobs already ahead, which age at the same rate). The
eleventh long item is for 29632479 (preemptable, 14 h) or a later worker. The running pair is unchanged: 29644110
(r101's native seed 2, to about 08:00) and 29644111 (r105 lr 12e-6 seed 0, to about 08:26). r102's three H200 gate
workers are pending at priority 39,220 and 29632479 at 37,949. INBOX is unchanged since 11-19 20:45.

## 2026-11-20 09:23 EST: full wake: r101's native seed 2 (0.2833 after 44,987 steps; three native seeds 0.292 against the vLLM control's 0.254); r105's first pair (lr 12e-6 level with its control, not the gain I expected); no GPU from 08:11 until a worker started at 09:14; r113 moved ahead of r105's later seeds

**What finished.** The first B200 pair ran its two long items and left. r101's native seed 2
(`d007rl_native_b200`, worker 29644110, node5600-c1) ran 02:51 to 07:42 EST and r105's lr 12e-6 seed 0
(`d007rl_native_lr12e-6_b200`, worker 29644111, node5602-c1) ran 03:20 to 08:11. Each took 4 h 51 min from start
to "ok" for a 17,400 s budget (17,464 s and 17,463 s), so a native RL-only item costs about a minute beyond its
budget, not the 5.1 h I had in STATUS. Both workers then exited on their reserve (less than 325 min left). No
failures. Both results are B200, RL only from D 007's pretrained checkpoints, and not yet gated.

**r101, all six runs (native sampler against the vLLM control, 17,400 s each).**

| arm | seed | node | RL steps | final dev | mean of last five evals | best eval | steps/s |
|---|---|---|---|---|---|---|---|
| native | 0 | node5600-c1 | 34,893 | 0.3033 | 0.2920 | 0.3033 | 2.01 |
| native | 1 | node5602-c1 | 44,862 | 0.2883 | 0.2857 | 0.2883 | 2.58 |
| native | 2 | node5600-c1 | 44,987 | 0.2833 | 0.2803 | 0.2917 | 2.59 |
| control | 0 | node5600-c1 | 18,215 | 0.2483 | 0.2463 | 0.2533 | 1.05 |
| control | 1 | node5800-c1 | 20,032 | 0.2383 | 0.2393 | 0.2583 | 1.15 |
| control | 2 | node5602-c1 | 17,608 | 0.2750 | 0.2707 | 0.2800 | 1.01 |

Final dev: native mean 0.2917 (sd 0.0104), control mean 0.2539 (sd 0.0190). By pair: seed 0 +0.0550, seed 1
+0.0500, seed 2 +0.0083; mean +0.0378, sd 0.0256, SE 0.0148. The third pair is small because the control's seed 2
ended high (0.2750, the best of the three controls on the fewest steps), not because the native run was low:
native seed 2 sits with seed 1. All three native finals are above D's line of 0.281 (0.3033, 0.2883, 0.2833) and
all three control finals are below it. That is not a gate: these runs start from D 007's checkpoints and ran on a
B200. Native seed 2's final by level: L1 0.90, L2 0.19, L3 0.20, L4 0.06, L5 0.03, L6 0.32; malformed 0.0000.

**At equal steps the arms are still level.** Over the evaluation steps both runs of a pair reached, native minus
control is +0.0124 for seed 0 (36 evaluations), +0.0045 for seed 1 (40) and -0.0157 for seed 2 (35, up to step
17,500); mean +0.0004. So the gain is still the extra steps, as on 11-19 14:35. Native dev around a step count
(mean of the nine evaluations within 2,000 steps):

| around step | native seed 0 | native seed 1 | native seed 2 |
|---|---|---|---|
| 18,000 | 0.2598 | 0.2385 | 0.2328 |
| 20,000 | 0.2643 | 0.2420 | 0.2424 |
| 25,000 | 0.2704 | 0.2583 | 0.2504 |
| 29,000 | 0.2824 | 0.2694 | 0.2633 |
| 32,000 | 0.2930 | 0.2724 | 0.2698 |
| 35,000 | not reached | 0.2743 | 0.2743 |
| 40,000 | not reached | 0.2744 | 0.2804 |
| 43,000 | not reached | 0.2822 | 0.2802 |

Seeds 1 and 2 had the same node speed (2.58 and 2.59 steps/s) and their curves agree to within 0.01 all the way.
From around step 35,000 to around 43,000 they gained +0.0080 and +0.0059: 8,000 steps for about half a point. Seed
0, on a slower day of the same node as seed 2 (2.01 steps/s on node5600-c1 against 2.59), stood 0.01 to 0.03 above
the other two at every step count and ended highest on the fewest steps. So among the native seeds the seed
matters more than 10,000 extra steps, and steps beyond about 35,000 are worth little with these settings. For r102
(about 29,000 steps expected on an H200) the three seeds around step 29,000 are 0.2824, 0.2694 and 0.2633, mean
0.2717: under the line of 0.281 by about 0.01 in the mean. My expectation for r102 stays at finals of about 0.27
to 0.29, so the three-seed rule may be missed by a seed or two, and what is needed next is more dev per step,
which is what r105 and r113 test.

**r105, first pair: lr 12e-6 against the native control, seed 0.** The arm did 36,808 steps on node5602-c1 (2.12
steps/s) and ended at 0.2900 (best 0.3050, mean of the last five evals 0.2903); the control's seed 0 did 34,893 on
node5600-c1 (2.01 steps/s) and ended at 0.3033. The two happen to have nearly the same steps, so the block means
line up over the whole run:

| steps (ten evals to) | lr 12e-6 seed 0 | control seed 0 | difference |
|---|---|---|---|
| 5,000 | 0.1557 | 0.1523 | +0.0033 |
| 10,000 | 0.2143 | 0.2163 | -0.0020 |
| 15,000 | 0.2387 | 0.2478 | -0.0092 |
| 20,000 | 0.2435 | 0.2590 | -0.0155 |
| 25,000 | 0.2598 | 0.2695 | -0.0097 |
| 30,000 | 0.2767 | 0.2748 | +0.0018 |
| 35,000 | 0.2893 | 0.2945 | -0.0052 |

Mean of the 7 block differences -0.0052; over the 69 shared evaluations (steps 500 to 34,500) the mean difference
is -0.0051 (sd 0.0112). Final: -0.0133 with 1,915 more steps. Final by level, arm minus control: L1 -0.05, L2
-0.06, L3 +0.00, L4 +0.01, L5 -0.04, L6 +0.06. I expected about +0.02 (range -0.02 to +0.05) and wrote that a gain
of 0.015 or more over the three pairs puts the lr into the next gate config and a tie closes lr for the native
regime. This pair is a tie that leans slightly down: twice the lr bought nothing per step at any point of the run,
early or late. One pair against a seed sd of about 0.01 to 0.02 at the final does not close it, and the control's
seed 0 is the highest of its three, so seeds 1 and 2 stay queued; but I no longer expect a gain from lr 12e-6, and
I lower what I expect from the two cosine arms that carry it (`lr12cos` and `stack3`: the upper ends of their
ranges, +0.04 and +0.06, now look too high; the schedule itself is untested). lr 24e-6 seed 0 is the next long
item and says whether a larger step hurts or is also level.

**Jobs.** No GPU of mine ran from 08:11 to 09:14. Then 29715907 started on node5800-c1 (09:14:52, 12 h 16 min
after its submission at 20:58 on 11-19) and took r111's five smoke runs; r105's lr 24e-6 seed 0 follows them in
its block (4 h 51 min, so to about 14:20). The other nine normal B200 workers (29715908 to 29715910, priority
364,902; 29755538 to 29755543, priority 362,726) are pending on `Priority`, no longer on the per-user limit. The
scheduler's planned starts are estimates and they move: at 08:43 the first read 10:00, at 08:52 it read 11:30, at
09:10 none of the ten had one (four minutes before one started), and at 09:23 all nine read between 10:53 and
17:00 today. I use them only to pick the next wake. My fair-share part of the priority fell from 1,287 to 1,223
(of 150,000) since last night. 29632479 (preemptable B200) and r102's three H200 gate workers are pending at
priority 39,324 and 40,595 with no planned start; nothing of mine has started on `mit_preemptable` since D 007's
last record seed on 11-18. The line at 08:38, as counts only: 88 H200 and 42 H100 requests pending on `Priority`
on `mit_normal_gpu`; 116 H200 and 12 B200 requests pending on `mit_preemptable`.

**Decisions.** (1) No native control seeds 3 and 4: the arms pair against seeds 0 to 2 and the GPU time is scarce.
(2) **Queue order changed at 09:02** (under the lock; nothing added or dropped): r113's arms now run before r105's
later seeds. The long items in order: r105 lr 24e-6 seed 0; r113 silc2, sil256 and stack3 seed 0; r105 lr 12e-6
seed 1; r113 silc2, sil256 and stack3 seed 1; r105 lr 24e-6 seed 1, lr 12e-6 seed 2, lr 24e-6 seed 2. Why: lr
12e-6's first pair is not a gain, the box queues for 11-25 are planned on 11-24 and r113's three arms are box arms
whose result changes that plan, and INBOX 11-19 14:29 asks for bold arms on few seeds. r105's three-pair rule is
unchanged; its later seeds only run later. The first two long items a worker can reach (lr 24e-6 and silc2) need
no r111 smoke; sil256 and stack3 come after the smokes are read. (3) **The gate's place in the line is the next
decision.** r102 and any track D record need more than the 6 h of `mit_normal_gpu`, so they sit on
`mit_preemptable` at a priority near 40,000, behind every normal job on the shared nodes, and r102 has now waited
31 h. A gate that fits a 6 h normal H200 worker (46 min of fused pretraining and a shorter RL budget, on its own
queue) would wait at about 362,000 like the B200 workers, which do start. Before I queue one I am checking that an
end-to-end item fits 6 h with its loading and evaluation time, and what budget a record run of the same config
could have, because a record adds the test ladder and `run_item.sh` wants the budgets plus 30 min left before it
starts one. If it holds up it gets its own entry with expectations and a rule before anything is submitted; r102's
workers stay pending either way.

**Checks.** The message of the day has no maintenance or outage line. Quota at 08:39: home 47 %, pool 64 % (660 of
1,024 GB), under the 80 % line. INBOX is unchanged since 11-19 20:45 (e66970b).

## 2026-11-20 10:26 EST: r111's five smoke runs pass; r106: `factored_grammar` saves constraint time only early; r112: 640x10 runs 1.10x the control, model size closed; r102 seed 0 runs on an H200 and seed 1 was preempted; r114 queued (the gate in a form that fits a 6 h normal worker); STATUS

The second part of the 08:36 full wake (the first is the 09:23 entry). From this entry on the agent runs on Opus
5.5.

**r111: the five standby-arm smoke runs pass.** 29715907 started at 09:14 on node5800-c1 and ran r111's five 90 s
items (RL only, native sampler, D 007 seed 0's pretrained checkpoint, each the r108 box arm with `time_budget_s`
90) from 09:14 to 09:25. Every one reached its final evaluation (n = 600) with no error line in `stdout.log`:

| smoke (B200, 90 s) | steps | steps/s (rows 6+) | final dev | `gpu_mem_gb` max | trace peak (MiB) | max `sil_rows` | tokens/s | lr first / middle / last |
|---|---|---|---|---|---|---|---|---|
| `smoke_mini3` | 55 | 0.66 | 0.0233 | 70.69 | 72,384 | 62 | 10,402 | constant |
| `smoke_sil256` | 51 | 0.59 | 0.0533 | 74.98 | 76,780 | 140 | 10,648 | constant |
| `smoke_g32p16` | 65 | 0.74 | 0.0233 | 72.40 | 74,134 | 60 | 11,794 | constant |
| `smoke_lr12cos` | 60 | 0.71 | 0.0083 | 70.85 | 72,544 | 55 | 11,416 | 1.2e-5 / 5.0e-6 / 1.2e-6 |
| `smoke_stack3` | 56 | 0.64 | 0.0317 | 70.70 | 72,398 | 184 | 8,736 | 1.2e-5 / 5.8e-6 / 1.2e-6 |

The control (r101's native seeds 0-2, other B200 nodes) over the same first 65 steps: steps/s 0.82, 0.78 and 0.69;
`gpu_mem_gb` max 75.07, 75.11 and 74.26. What it means: all five code paths run with the native sampler (the mini
levels, groups of 32 with 16 prompts, the cosine falling from 1.2e-5 to 1.2e-6 as set, 256 self-imitation rows);
no arm held more card memory in its first steps than the control; steps per second this early say little (the long
runs settle later and differ by node), and the 90 s scores mean nothing. One limit: sil256 and stack3 drew at most
140 and 184 self-imitation rows in about 55 steps, so a full 256-row draw has not been measured (by design the
65,536-token micro-batch bounds the update); r113's 17,400 s B200 runs of both measure it, before 11-24 if a
worker reaches them. So no r113 item is pulled (LOG 11-20 04:34), the three box queues stay as staged, and one run
per H100 stands.

**r106, the start-weights pair.** After r111, the second normal worker 29715908 ran r106's last two timing items
on node5701-c1: 300 steps each from D 007 seed 0's pretrained (start) weights with the settled speed settings and
`time_update_parts`, the control (10:04 to 10:09) and then `factored_grammar` (10:09 to 10:13). At these weights
two same-maths runs part early (the grad norm at step 1 is 0.050677 against 0.050673, and the loss differs from
step 2), so the pair is read on its timings, means over steps 51 to 300:

| run (B200, 300 steps from the start weights) | `t_part_s_mask_cpu` s, mean (median) | per sampled token | `t_sample` s | `t_update` s | steps/s | tokens a step | mask s, steps 1-50 | trace peak (MiB) |
|---|---|---|---|---|---|---|---|---|
| `start_native_ctl` | 0.1271 (0.1137) | 10.4 us | 0.2543 | 0.4432 | 1.283 | 12,278 | 0.3109 | 72,514 |
| `start_native_factored` | 0.0991 (0.0870) | 8.2 us | 0.1936 | 0.3943 | 1.489 | 12,148 | 0.2861 | 77,138 |

The constraint timer is lower with the option: 0.0991 against 0.1271 s a step (-22 %), 8.2 against 10.4 us per
sampled token, and lower in the first 50 steps too. So early in training new grammar states do cost time and the
option saves part of it, as the CPU benchmark said; at the final weights (LOG 11-20 04:12) there was nothing left
to save. The whole run was +16 % in steps a second, but most of that is not the option: `t_update`, which it does
not touch, was -11 % too, so the two runs differ in their samples or the node's state, and one pair in one order
cannot split those. What the option could be worth in a run: the saving fades as the policy settles, and even 0.03
s a step over the first 2,000 steps would be about a minute of a 4.8 h run. The factored run also peaked 4,624 MiB
higher on the card, which I cannot separate from its samples either. So `factored_grammar` stays off by default
and out of the settled settings; no arm is built on it.

**r112, the first block of size smokes** (29715908 on node5701-c1, 10:14 to 10:19: RL only, native sampler, 90 s
each from checkpoints pretrained with D 005's recipe, D 007's RL settings; steps a second from step 20 on):

| model (B200, 90 s, from step 20) | parameters | steps | steps/s | against the control | `t_sample` s | `t_update` s | tokens a step | `gpu_mem_gb` max | trace peak (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| `ctl_w768_l12`: 768x12 (control) | 110M | 71 | 0.916 | 1.00x | 0.3576 | 0.5528 | 11,965 | 74.21 | 75,988 |
| `w640_l10`: 640x10 | 69M | 75 | 1.004 | 1.10x | 0.3250 | 0.4675 | 11,550 | 61.21 | 62,674 |
| `w1024_l12`: 1024x12 | 182M | 51 | 0.688 | 0.75x | 0.5689 | 0.6710 | 11,987 | 89.51 | 91,662 |

Expected (LOG 11-20 01:21): 640x10 at 1.25x to 1.5x and 1024x12 at 0.6x to 0.75x. 1024x12 came at the top of its
range and 640x10 well under its own. A straight line through the three step times leaves 0.68 s of a step that
does not grow with the model, 62 % of the control's step at this point of a run: with the native sampler a step is
mostly a fixed cost, not compute per parameter. **By the rule** (1.2x or less), model size stays closed with the
native sampler too: no small-model seeds and no 512x8 pretraining item. The speed work goes to the parts of a step
that do not depend on the model. 1024x12 also held 91,662 MiB at these micro-batches, more than an 80 GB H100 has.
The second block (control and 640x10 seed 1, 768x24 seed 0; about 6 minutes of a worker's slack before its long
item) stays queued as a check on another node. One thing I wrote at 04:12 was wrong: r112's configs do not set
`time_update_parts`, so they give no constraint timer by model size, and what the timer's 0.087 s at the final
weights is (a wait for the forward step, or per-row overhead in `mask_rows_cached`) is still open. The next speed
item finds where the fixed part of a native step goes, timed before any code changes; I write it at a quiet wake.

**r102 on the H200.** Seed 0 (29450693) started at 09:40 on node4802, 31 h 49 min after the three workers were
released at 01:52 on 11-19: the first of my preemptable jobs to start since 11-18. The fused pretraining
(micro-batch 128) stood at step 2,300 after 43.4 min, 1.206B tokens, 463k tokens a second so far and 473k over the
last 100 steps; at that pace 46 min gives about 1.28B tokens, against 1.325B in 56 min for D 007's unfused
pretraining (394k a second on the H200; mean of five seeds, final validation loss 2.883) and 2.011B in 46 min on
the B200 (r92). So on the H200 the 46-minute checkpoint is about D 007's, as planned. The pass into native RL
(never run end to end before) is due about 10:30; its RL steps per second on the H200 decide the expected score
(below) and are read at the next wake. Seed 1 (29450694) started at 09:54 on node4603 and was preempted at
10:04:16 at pretraining step 400 (210M tokens, about 10 GPU-minutes lost). Slurm requeued the job; the worker's
marker in `queue/running/` makes it redo seed 1 when it starts again (the seed directory holds no training log, so
nothing moves to `extra/`). The requeue reset the job's age: its priority is now 31,298, against 40,892 for seed
2's worker 29450702 (pending, with 32 h of age). Each r102 item will end `FAILED` at `run_seed_v3.sh`'s final `mv`
by construction (`no_save` writes no `final` directory), with every result written, as r81 did.

**r114: the gate in a form that fits a 6 h normal worker.** r102's items need 46 min of pretraining, 300 of RL and
`run_item.sh`'s 30-minute margin for a record seed: 376 min, more than a 6 h `mit_normal_gpu` worker. So r102 runs
only on `mit_preemptable` (priority about 31,000 to 41,000 there, about 361,000 to 365,000 on `mit_normal_gpu`),
where a 5.8 h run can be preempted, as seed 1 was today, and a requeued job starts its wait again. r114
(`calibration/research_r114/d007_speed_h200_6h`) is r102's config with `time_budget_s` 16,440 instead of 18,000
and nothing else changed: 46 + 274 + 30 = 350 min, so a seed fits a 6 h normal worker as a screen and, without
`no_save`, as a record seed (320 min of the 360-minute budget). The cost is 1,560 s of RL, about 9 % of the steps;
on r101's native curves that is worth little at these step counts (dev around step 33,000 was 0.277 and around
36,000 0.277). Seeds 0-2 are in `queue/gate_r114_h200.txt`, and at 09:50 I submitted one normal H200 worker for
each (29777239 to 29777241: 6 h, 16 CPUs, 128 GB, `RESERVE_MIN` 335, `REQUIRE_GPU` NVIDIA H200, the H200 workers'
node exclusions). Their planned starts read the evening of 11-25 at some readings and none at others; on
`mit_normal_gpu` they share my two-GPU cap with the eight pending B200 workers, which are ahead of them (priority
361,387 against 363,023 and more). So r114 is not yet a faster gate; it is the record form gaining age. Which goes
first on the normal cap, the gate or the B200 items, I decide once r102 seed 0 shows the H200's native RL rate,
before the next normal slot opens (about 14:20).

Expected final dev, from r101's native seeds (RL only on the B200 from D 007's checkpoints, not gated), averaged
over the evaluations within 1,000 steps of the step count a rate gives:

| H200 native RL rate | r114 (16,440 s of RL) | r102 (18,000 s) |
|---|---|---|
| 1.61 steps/s (the AWS H100's, 11-18) | about 26,500 steps: 0.262 | about 29,000: 0.273 |
| 2.0 steps/s | about 32,900: 0.277 | about 36,000: 0.277 (two seeds) |
| 2.6 steps/s | about 42,700: 0.279 (two seeds) | about 46,800: past the B200 runs (about 0.28) |

At any of these rates the gate sits near the line 0.281 and not clear of it, so the three-seed form (every seed at
or above 0.281) needs a faster card than these or some luck. The end-to-end runs also differ from r101 in the
checkpoint (46 min fused, not D 007's 56 min) and the card. **Rule, written before any r114 seed runs:** each
seed's final dev is read against 0.281 with its steps and node. Three seeds at or above 0.281: a five-seed record
attempt with r114's config without `no_save`, on 6 h normal H200 workers. Short of that, the gap says how much an
arm must add, and r114's queued seeds can be swapped for a stronger config in a new directory before a worker
starts them (or the workers cancelled). No end-to-end smoke ran first: r102 seed 0 is that test, and if its
hand-off into RL fails, a 2-minute end-to-end smoke on the B200 checks the fix before any more gate seeds start.

**Jobs.** The second normal B200 worker, 29715908, started at 10:04 on node5701-c1, 13 h 05 min after its
submission at 20:58 on 11-19. It ran r106's start-weights pair and r112's first three size smokes (above), and
since 10:19 it runs r113 silc2 seed 0 (to about 15:15). So both normal GPUs of the cap are busy: 29715907 with
r105 lr 24e-6 seed 0 (to about 14:20) and 29715908. Eight normal B200 workers are pending on
`QOSMaxCpuPerUserLimit` (none can start before one of the two running ones ends) at priority 365,199 and 363,023;
the three r114 workers at 361,387; the preemptable B200 worker 29632479 at 39,621; r102's seed 1 and 2 workers at
31,298 and 40,892. Fair-share part 1,222. The B200 queue has 12 lines, 9 of them long items; held as before:
29497701/02/05 and 29445218. INBOX is unchanged since 11-19 20:45 (e66970b).

## 2026-11-20 11:48 EST: r112's second block: the control ran 1.44x faster from its seed-1 checkpoint than from seed 0's, so a 90 s size ratio follows the starting weights; model size stays closed; 768x24 needs 116,730 MiB; 29632479 preempted after 28 minutes and its item put back at the front of the queue; r102 seed 0 at 1.85 RL steps a second on the H200

**What ran.** The preemptable B200 worker 29632479 started at 11:18 on node5600-c1 (priority about
39,600, under the 50,000 at which the scheduler plans starts, so it started when a card was free as it was
tried). It ran r112's second block of 90 s size smokes (the control and 640x10 from the seed-1 checkpoints of D 005
and r50, 768x24 from r53's seed 0), then started r113 sil256 seed 0 at 11:24, and was preempted at 11:46. All six
r112 runs (steps a second from step 20 on; median GPU use, my CPUs' clock in MHz and the node's running tasks from
the 5 s trace):

| node | model | seed | steps | steps/s | against the control on that node | `t_sample` s | `t_update` s | tokens a step | GPU use | my CPUs MHz | running tasks | trace peak (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| node5701-c1 | `ctl_w768_l12` 768x12 (control) | 0 | 71 | 0.916 | 1.00x | 0.3576 | 0.5528 | 11,965 | 34 % | 2,360 | 15 | 75,988 |
| node5701-c1 | `w640_l10` 640x10 | 0 | 75 | 1.004 | 1.10x | 0.3250 | 0.4675 | 11,550 | 54 % | 2,011 | 60 | 62,674 |
| node5701-c1 | `w1024_l12` 1024x12 | 0 | 51 | 0.688 | 0.75x | 0.5689 | 0.6710 | 11,987 | 52 % | 1,992 | 60 | 91,662 |
| node5600-c1 | `ctl_w768_l12` 768x12 (control) | 1 | 93 | 1.317 | 1.00x | 0.2134 | 0.3971 | 11,902 | 56 % | 2,332 | 12 | 74,826 |
| node5600-c1 | `w640_l10` 640x10 | 1 | 80 | 1.018 | 0.77x | 0.3799 | 0.4533 | 12,475 | 30 % | 2,496 | 24 | 62,982 |
| node5600-c1 | `w768_l24` 768x24 | 0 | 37 | 0.465 | 0.35x | 1.0935 | 0.8695 | 12,776 | 92 % | 2,322 | 14 | 116,730 |

**What it means.** The second block does not repeat the first. 640x10 ran at the same rate in both blocks
(1.004 and 1.018 steps a second), but the control ran 1.317 from D 005's seed-1 weights
against 0.916 from seed 0's, so 640x10 was 0.77x the control in the second block and 1.10x
in the first. The two control runs had the same tokens a step (11,965 and 11,902), the same CPU
clock and about as many running tasks on the node, so CPU contention does not explain the gap; the faster run kept
the GPU busier (56 % against 34 %). With two nodes and two checkpoints in
two runs I cannot tell node from weights apart. My guess, untested: the native sampler decodes until the longest
row of a batch is done, so a step's sampling time follows the longest completion, not the mean, and how often one
row runs long depends on the weights. The train log has no field for the longest completion, so this cannot be
read from these runs. Three consequences:

- **Model size stays closed, more firmly.** 640x10 never reached the rule's 1.4x (1.10x and 0.77x). No
  small-model seeds and no 512x8 pretraining item.
- **The 0.68 s / 62 % fixed part of a step from 10:26 is not a measurement to build on.** It came from one set of
  starting weights. The second block puts 69M slower than 110M, so no straight line fits it. That a native step is
  mostly not compute per parameter still stands (GPU use 30-56 % for the two smaller models); how
  much is fixed does not.
- **The timing item for the fixed part** has to compare runs from one checkpoint on one node, and log the longest
  completion of each batch (the number of decode steps) next to `t_sample`. Speed smokes from different seeds'
  checkpoints are not comparable at 90 s; the speed pairs of r101 and r106 started from the same seed's weights.

768x24 (about 195M) ran 0.465 steps a second, 0.35x the control on its node (0.51x the first
block's control), under the 0.55x-0.70x I expected, at 116,730 MiB: over an 80 GB H100 and inside a
141 GB H200. Its GPU use was 92 %, the only run near full use. No arm comes from it.

**29632479's preemption.** It was preempted at 11:46:07, 22 minutes into r113 sil256 seed 0, and requeued; its
age starts from zero again (priority 31,226). Its marker in `queue/running/` would have made it redo that item
whenever it starts again, which could be days. I put the item back at the front of `screens_rl_b200` under the
queue lock (what the worker's own `give_back` does) and removed the marker, so the next of the nine workers pending
on that queue (eight normal at 363,400 to 365,600, and 29632479) takes it. The queue has 9 long items for those 9
workers. The partial run's files stay in its seed directory; the worker sets an interrupted run's log aside
before it reruns.

**r102 seed 0 (H200, node4802).** In RL since about 10:29: step 8,446 after 4,799 s,
1.760 steps a second overall, 1.313 over the first 1,000 steps and 1.847 since. At that rate it ends
its 18,000 s near step 32,827, against D 007's 16,318 to 19,735. Dev at step 8,000 is
0.1917; D 007's five seeds read 0.225, 0.185, 0.195, 0.195, 0.205 there (mean 0.2010), so it is
inside their range at equal steps, a little under their mean, with a pretraining loss of 2.905 against their 2.883.
On the table of LOG 11-20 10:26 (1.61 steps a second gives 0.273, 2.0 gives 0.277), 1.85 reads
0.275 for r102 and a little less for r114's 16,440 s: both under the 0.281 line. That is the expectation;
the reading is seed 0's final dev at about 15:35.

**The order on `mit_normal_gpu`.** The next two normal slots open at about 14:20 and 15:15; the B200 workers'
priority (363,400 to 365,600) is above r114's (361,774), and all age at the same rate, so the B200 workers take
both. r114 cannot start before r102 seed 0 finishes, so the decision moves to 15:35 at no cost: if seed 0 ends at
or above 0.281, I let r114's three workers go ahead of the remaining B200 workers (by holding B200 workers, which
loses their age, or by a nice on them, untested here); if it ends near the 0.275 expected, the B200 workers
keep their order.

INBOX is unchanged since 11-19 20:45 (e66970b).

## 2026-11-20 14:25 EST: r105's second pair (lr 24e-6 against the native control, seed 0): level over the whole run; with lr 12e-6 also level, I expect no gain from the lr in the native regime

r105's lr 24e-6 seed 0 ran on node5800-c1 (worker 29715907) and ended at 0.2967 after 45,636 steps (2.62 steps/s; best 0.3167, mean of the last five evals 0.3060). The control's seed 0 (LOG 11-20 09:23) did 34,893 steps on node5600-c1 (2.01 steps/s) and ended at 0.3033 (last five 0.2920). Block means at equal steps, with lr 12e-6 seed 0 beside them:

| steps (ten evals to) | lr 24e-6 seed 0 | control seed 0 | difference | lr 12e-6 seed 0 |
|---|---|---|---|---|
| 5,000 | 0.1583 | 0.1523 | +0.0060 | 0.1557 |
| 10,000 | 0.2182 | 0.2163 | +0.0018 | 0.2143 |
| 15,000 | 0.2427 | 0.2478 | -0.0052 | 0.2387 |
| 20,000 | 0.2627 | 0.2590 | +0.0037 | 0.2435 |
| 25,000 | 0.2757 | 0.2695 | +0.0062 | 0.2598 |
| 30,000 | 0.2825 | 0.2748 | +0.0077 | 0.2767 |

Over the 69 shared evaluations (steps 500 to 34,500) the mean difference is +0.0020 (sd 0.0113); lr 12e-6's was -0.0051 over its 69. Final: -0.0067 with 10,743 more steps; the last five evals +0.0140. Final by level, arm minus control: L1 +0.00, L2 -0.07, L3 -0.03, L4 +0.01, L5 -0.02, L6 +0.07. I expected a step past the useful range to show early as faster gains then a fall, or as instability; neither happened: entropy and grad norm fell smoothly, and the arm is within 0.008 of the control at every block. Four times the control's lr and twice it both sit within about 0.005 of the control at equal steps, with no order by dose (-0.005 at 2x, +0.002 at 4x), so in the native regime the lr is not what limits the score per step. The rule (a gain of 0.015 over three pairs puts the lr into the next gate config; a tie closes it) needs the later pairs to close formally; I no longer expect a gain. Decision: seeds 1 of both doses stay in the queue (one more pair each); the two seed-2 items are replaced at the next full wake by a bold item, keeping one pending worker for each long item (INBOX 11-19 14:29: one or two seeds against the control, wide and shallow).

**Step rates.** The three arms started from the same seed-0 weights on three nodes and ran at 2.01, 2.12 and 2.62 steps/s overall (control, 12e-6, 24e-6). Within a run the update time falls as the informative groups fall (the control: 0.360 s at 61 informative groups of 64 over steps 1 to 1,000, 0.170 s at 30 over 30,000 to 31,000, at about the same tokens per step, 12,721 and 12,460), but at steps 5,000 to 6,000 the control and lr 24e-6 had the same count (40 and 41) and update times of 0.374 s and 0.256 s on different nodes. So part of the rate gap is the node, as in r112's two blocks (LOG 11-20 11:48); the arm's 10,743 extra steps are not credited to the lr, and the comparison above is at equal steps.

**r102 seed 0** (H200 gate, native RL from fused pretraining): dev 0.2717 at step 29,500, the last six evals 0.258 to 0.282 (mean 0.2678; train step 29,643 at 14,200 s of 18,000, about 63 min to go), against the 0.281 line; the final dev at about 15:35 is the reading, and the order on `mit_normal_gpu` (LOG 11-20 11:48) is decided then.

**Jobs.** 29715907 exited after the item (less than 325 min left). 29715909 started at 14:18 on node5801-c1 and took r113 sil256 seed 0 (the item 29632479 gave back at 11:46). Running: 29715908 (r113 silc2 seed 0, to about 15:15), 29715909 and r102 seed 0 (29450693). INBOX is unchanged since 11-19 20:45 (e66970b).

## 2026-11-20 15:32 EST: a pool quota failure (15:03 to 15:15) cost two running B200 seeds and three queued items; r113 silc2 seed 0 read from its partial (behind at matched steps); r102 seed 0 ends at 0.270, under the line; r114 taken off

**The quota failure.** At 15:03 both running B200 items failed with `OSError: [Errno 122] Disk quota exceeded` on their train logs, which a screen writes in the pool (`~/orcd/pool/rlvr-ckpts/<dir>/seeds/<seed>`) and copies to the checkout at the end: r113 silc2 seed 0 at train step 38,448 (16,978 s of its 17,400 s, 422 s short of the end), and r113 sil256 seed 0 at step 4,100 (2,615 s). At 15:14 and 15:15 three more workers each failed an item within seconds at `mkdir` in the pool: r113 stack3 seed 0, r105 lr12e-6 seed 1 and r102 seed 2 (preemptable H200 29450702, which then found its queue empty and exited). The worker drops a failed item unless it is the second quick failure in a row, so those three left the queues; silc2 seed 1 and sil256 seed 1 were put back after second quick failures and started normally at 15:24 on node5801-c1. r102 seed 0 writes its logs in the checkout and ran on. Pool writes work again (a write probe just before 15:28); `df` shows 661 of 1,024 GB (65 %) at 15:28 and the quota cache showed 64.5 % at 15:12, so I have not found what reached the limit; nothing of mine grows by hundreds of GB in minutes. Such a failure shows as `Disk quota exceeded` in the slurm logs, and I check for it each wake.

**Recovery.** sil256 seed 0, stack3 seed 0 and lr12e-6 seed 1 are back at the front of `screens_rl_b200` (sil256's 4,100-step partial stays in its pool directory, which the rerun moves aside), and r102 seed 2 is back in `screens_r92`. Three normal B200 workers (29808850-52) and one preemptable H200 worker (29808853, with 29450702's submit line) replace the workers that exited, so every queued long item again has one pending worker. silc2 seed 0 is not rerun (below); its pool logs are copied to its seed directory as `train_log.partial.jsonl` and `eval_log.partial.jsonl`.

**r113 silc2 seed 0 (partial) against the native control's seed 0**, dev means over shared eval steps:

| steps | evals | silc2 seed 0 | control seed 0 | difference |
|---|---|---|---|---|
| 0 to 5,000 | 10 | 0.1015 | 0.1523 | -0.0508 |
| 5,001 to 10,000 | 10 | 0.1667 | 0.2163 | -0.0497 |
| 10,001 to 20,000 | 20 | 0.2228 | 0.2534 | -0.0307 |
| 20,001 to 30,000 | 20 | 0.2663 | 0.2722 | -0.0058 |
| 30,001 to 34,500 | 9 | 0.2961 | 0.2935 | +0.0026 |

Over all 69 shared evals (to step 34,500) the mean difference is -0.0248 (sd 0.0227). silc2 starts about 0.05 behind, closes most of the gap by step 20,000 and is level after that. Its last eval is 0.3050 at step 38,000 against the control's final 0.3033 at 34,893 steps; its last five evals average 0.3100 against the control's 0.2920, but it had run 3,555 more steps by then (node5701-c1 against the control's node5600-c1). The r113 rule for a third seed is +0.02 at the final in both pairs and ahead at matched steps; this pair is behind at matched steps whatever its final would have been, so a third seed is off, and a rerun (about 4.8 B200 hours) could not change that. Seed 1's pair (started 15:24) completes the reading. sil256 seed 0's partial, 8 shared evals to step 4,000: -0.0425, the same early lag as silc2's first 5,000 steps (-0.0508); it reruns.

**r102 seed 0's final.** 0.2700 at step 38,914 (18,000 s of GRPO at 2.162 steps a second after 2,775 s of pretraining; 20,798 s of the 21,600 s budget). Its last five evals average 0.2720, best 0.2817. I expected 0.275 (LOG 11-20 11:48). It is +0.019 over D 007's dev mean (0.2510) and under the 0.281 line.

**The order on `mit_normal_gpu`, and r114.** Seed 0 ended under 0.281, so by the 11:48 rule the B200 workers keep their order. I also take r114 off. It is the same config in a 6 h normal form with 16,440 s of GRPO, which on r102 seed 0's curve is step 35,541, where its evals average 0.2746 (n 4). With the spread of r101's three native seeds (sd 0.0104), one seed reaches 0.281 with probability 0.145 at a mean of 0.270 and 0.269 at 0.2746, so three of three is 0.003 to 0.019: not worth about 18 normal GPU hours. r114's three items are out of `gate_r114_h200` and its workers 29777239-41 are cancelled. r102 seeds 1 and 2 still run (preemptable) to give the speed config's three-seed mean. The INBOX order (4a72abc) was speed into score first, then large changes; speed alone has given about +0.02 on one seed against the +0.03 a record needs, so the normal slots go to the queued B200 screens and then to the bold item that replaces r105's two seed-2 items, written at the next full wake, before those items can start.

**Jobs.** Running: 29755539 and 29755540 (r113 silc2 seed 1 and sil256 seed 1, node5801-c1, since 15:24). Pending: six normal B200 workers (29755541-43, 29808850-52) and preemptable 29632479 for the seven `screens_rl_b200` items; preemptable H200 workers 29450694 (r102 seed 1, its resume state kept) and 29808853 (r102 seed 2). 29450693 exited after r102 seed 0 (less than 380 min left). GPUs in use: 2 of 6. INBOX is unchanged since 11-19 20:45 (e66970b).

## 2026-11-20 18:42 EST: the pool's file server filled up at 18:27 and failed r113 silc2 and sil256 seed 1 at about 3 h of their 4 h 51 min, and four queued items; both read from their partials (silc2 behind at every step, sil256 closing); new workers write to scratch

**The second failure.** At 18:27:26 and 18:27:28 both running B200 items failed with `OSError: [Errno 122] Disk quota exceeded` on their pool train logs: r113 sil256 seed 1 at train step 19,362 (10,948 s) and silc2 seed 1 at step 22,761 (10,944 s), both on node5801-c1 since 15:24. Their workers (29755540 and 29755539) then exited with less than 325 minutes left. At 18:31 two more normal workers (29755541 and 29755542) started on the same node and each failed two items within seconds at the pool: sil256 seed 0 and stack3 seed 0 were dropped, and lr12e-6 seed 1 and stack3 seed 1 were put back after second quick failures. This time `df` on the pool shows a size of 661 GB with 661 GB used and 0 available (18:33 to 18:39), where at 15:28 it showed 661 of 1,024 GB. The quota cache still shows 660.6 of 1,024 GB (64.5 %), and my use has not changed since 15:28, so the shared file server behind the pool is full rather than my quota. That is outside my control. Whether to tell the cluster admins is the maintainer's call (QUESTIONS). The 15:03 failure probably had the same cause.

**Recovery.** I held the seven pending workers that write to the pool, and cancelled them at 18:40:
- normal B200 29755543 and 29808850-52;
- preemptable B200 29632479;
- preemptable H200 29450694, whose in-hand r102 seed 1 is back at the front of `screens_r92`;
- preemptable H200 29808853.

sil256 seed 0 and stack3 seed 0 are back at the front of `screens_rl_b200`, which now holds seven long items.

New workers write to my scratch space instead. Each gets `CKPT_DIR=/home/zjason/orcd/scratch/rlvr-ckpts` in its `--export`. `queue_worker.sbatch`, `screen_seed.sh` and `run_seed_v3.sh` all take `CKPT_DIR` from the environment, so no script changed. The RL screens read their starting weights through symlinks in that tree to D 007's three `pretrained` directories in the pool, since reading from the pool still works. r102 pretrains into scratch. The new workers are:
- six normal B200 workers (29837202, 29837204, 29837206, 29837208, 29837209 and 29837211) and preemptable B200 29837212, for the seven `screens_rl_b200` items;
- preemptable H200 29837213 and 29837214, for r102 seeds 1 and 2.

r102 seed 1's stdout and node trace from its preempted start are kept in its seed directory as `*.interrupted.1795218024.*`. Its pool directory holds a partial `pretrained` and no resume state, so it starts over. The older held workers (29445218, 29497701, 29497702 and 29497705) are unchanged. Scratch holds 31.7 of 1,024 GB.

**r113 seed 1 partials against the native control's seed 1** (dev means over shared eval steps; both arms on node5801-c1, the control on node5602-c1; the control ended at 0.2883 after 44,862 steps):

| arm | steps | evals | arm mean | control | difference |
|---|---|---|---|---|---|
| silc2 | 0 to 5,000 | 10 | 0.1093 | 0.1478 | -0.0385 |
| silc2 | 5,001 to 10,000 | 10 | 0.1640 | 0.2062 | -0.0422 |
| silc2 | 10,001 to 20,000 | 20 | 0.2003 | 0.2311 | -0.0308 |
| silc2 | 20,001 to 22,500 | 5 | 0.2207 | 0.2457 | -0.0250 |
| sil256 | 0 to 5,000 | 10 | 0.1318 | 0.1478 | -0.0160 |
| sil256 | 5,001 to 10,000 | 10 | 0.1917 | 0.2062 | -0.0145 |
| sil256 | 10,001 to 20,000 | 18 | 0.2271 | 0.2300 | -0.0029 |

silc2 seed 1 trails the control at every point: -0.0344 over 45 evals (sd 0.0125) to step 22,500. Its last five evals (steps 20,500 to 22,500) average 0.2207 against the control's 0.2457 at the same steps. sil256 seed 1 is -0.0094 over 38 evals (sd 0.0143) to step 19,000, with the gap closing from about -0.015 before step 10,000 to about -0.003 after it. Its last five evals average 0.2360 against 0.2397. In the same 10,944 s the control reached step 28,148; silc2 ran 19 % fewer steps and sil256 31 % fewer. sil256's extra update rows were expected to cost about 38 % more per step. silc2 adds no rows, so its shortfall is more likely the node or the two runs sharing one, but I have not checked that.

**What this means under the r113 rule** (LOG 11-20 04:34).
- **silc2** is behind at matched steps in both pairs: -0.0248 for seed 0 and -0.0344 for seed 1. The third-seed bar, +0.02 at the final in both pairs *and* ahead at matched steps, cannot be met. It is not closed under the rule's wording either, because seed 0 was level at the end, so it falls in "between", which the rule leaves to the box's seeds. I don't rerun seed 1 here. When I plan the 11-25 box on 11-24, it is a weak candidate against the other arms.
- **sil256** seed 0 reruns in full (first in the queue). I expect it to end level with the control's seed 0, between -0.02 and +0.01 (seed sd about 0.010 to 0.019). Seed 1 reruns only if seed 0 ends at least +0.02 over the control's 0.3033 and ahead at matched steps, since the rule needs both pairs. Otherwise its reading is this partial, which saves about 4.8 B200 hours.
- **stack3** has not started; both of its seeds are queued.

**Jobs.** Nothing of mine is running. Pending: the nine scratch workers above and the four older held workers. GPUs in use: 0 of 6. INBOX is unchanged since 11-19 20:45 (e66970b). Each wake checks the pool's `df` and the slurm logs for `Disk quota exceeded`. The workers stay on scratch until the pool has room again and I have checked it with a write probe.

## 2026-11-21 02:00 EST: r113 sil256 seed 0 ends level with the native control (0.3033 against 0.3033; -0.0010 at matched steps), so it falls between and is left to the box; where the native step's time goes late in a run; r115 (half the prompts per step) replaces r105's two seed-2 items, with its expected effect and rule written first

**r113 sil256 seed 0** (`sil_rows` 256, the rerun on scratch, node5602-c1, 20:45 11-20 to 01:36 11-21) against r101's native control seed 0 (node5600-c1). The read is by script, at the final and over the dev evals both runs made at the same step.

| | arm | control |
|---|---|---|
| final dev (n = 600) | 0.3033 | 0.3033 |
| steps in 17,400 s | 29,892 | 34,893 |
| steps/s | 1.718 | 2.005 |
| mean `t_sample` / `t_update` (s) | 0.169 / 0.375 | 0.163 / 0.286 |
| mean of the last five evals | 0.3047 | 0.2920 |

| steps | shared evals | arm | control | arm minus control |
|---|---|---|---|---|
| 1 to 5,000 | 10 | 0.1228 | 0.1523 | -0.0295 |
| 5,001 to 10,000 | 10 | 0.1880 | 0.2163 | -0.0283 |
| 10,001 to 20,000 | 20 | 0.2552 | 0.2534 | +0.0017 |
| 20,001 to 29,500 | 19 | 0.2968 | 0.2712 | +0.0255 |

Over all 59 shared evals the arm is -0.0010 (sd 0.0253, se 0.0033). It trails by about 0.03 before step 10,000 and leads by about 0.025 after step 20,000. It made 14 % fewer steps because the update carries four times the self-imitation rows. At the arm's last step the control stood at 0.2850, so the arm's final is +0.0183 over the control at equal steps, but level at equal time.

**What this means under the r113 rule** (LOG 11-20 04:34). The final is level, not +0.02, so the third-seed bar is not met. The arm is ahead at matched steps over the second half, so it is not closed either. It falls between, as I expected in LOG 11-20 18:42 (level, -0.02 to +0.01). Seed 1 was to rerun only if seed 0 cleared +0.02, so it does not rerun. Its partial stays as read (-0.009 to step 20,000, closing). The late lead at equal steps comes from one seed pair on two nodes, against a control seed sd of about 0.011, so I do not count it as a gain. It does make sil256 the strongest of r113's three arms for the 11-25 box, where more rows cost less if the update is faster. I weigh it there on 11-24.

**Where the native step's time goes late in a run** (r101 control seed 0, node5600-c1; 512 sampled rows a step):

| steps | `t_sample` s | `t_update` s | informative groups of 64 | solve rate in batch | tokens a step | wall s per step |
|---|---|---|---|---|---|---|
| 100 to 2,000 | 0.176 | 0.376 | 57 | 0.10 | 12,256 | 0.601 |
| 10,000 to 12,000 | 0.169 | 0.326 | 36 | 0.25 | 12,295 | 0.553 |
| 30,000 to 34,893 | 0.134 | 0.171 | 29 | 0.32 | 12,450 | 0.336 |

Tokens per step stay near 12,300 (about 24 per completion). As the policy improves, fewer groups carry a nonzero advantage, and with `skip_zero_advantage` the update shrinks from 0.376 s to 0.171 s. The sampler only falls from 0.176 s to 0.134 s. Late in a run the sampler is 44 % of the two timed parts.

I read the decode loop (`rlvr_speedrun/fast_rollout.py`, the per-token step after the CUDA-graphed forward). Each token copies the sampled ids to the CPU (`sampled.tolist()`) and runs a Python loop over the rows for the answer grammar (`allowed_ids`) and the stop check. `mask_rows_cached` (`constrained.py`) copies a pageable CPU tensor with `non_blocking=True`. Such a copy waits for the forward step on the stream. So the answer-constraint timer (0.087 s of a 0.392 s step, LOG 11-19 09:28) includes forward time, and it cannot be read as the CPU's own cost.

The speed work that follows, in order:
1. The grammar and stop checks as tables on the GPU, with a sync every k tokens instead of every token.
2. The update, whose time follows the node's CPU (LOG 11-20 11:48).

Each gets a timing item before any code is written. Both keep the same maths and are off by default.

**r115: half the prompts per step** (the bold item that replaces r105's two seed-2 items, as STATUS planned). The arm is `calibration/research_r115/d007rl_native_p32_b200` on the B200, RL only, not gated, seeds 0 and 1. It is r101's native control with these changes:
- `prompts_per_step` 32 against 64, with `group_size` 8 kept, so 256 sampled rows a step;
- `sil_rows` 32 against 64, so the self-imitation share per step is unchanged;
- `eval_every` 1000, so it evaluates about as often in time as the control does at 500.

Everything else is the same: lr 6e-6, `time_budget_s` 17,400 and D 007's seed-matched pretrained weights. It is a config-only change, and wide rather than fine.

Why: the control's curve still rises at the end (about +0.002 per 1,000 steps), and its gain over the vLLM control came from extra steps (LOG 11-19 14:35). A smaller batch buys more, noisier steps in the same time. The step rate also measures how much of a native step scales with the rows and how much is fixed, which the speed work needs.

Expected:
- **Step rate:** 1.45x to 1.6x the control's, about 50,000 steps. Late in a run I expect `t_sample` about 0.09 s and `t_update` about 0.10-0.12 s, since the per-row Python work halves but the forward step barely does.
- **Score:** centre 0.00 at the final, range -0.04 to +0.03. The paired noise is about 0.015 per pair (control seed sd 0.011).

I compare at the final and at matched train time (`train_s`), not matched steps, since a step holds half the rows. Node speeds differ, which I will report with each pair.

Rule, written before it runs:
- **+0.02 or more at the final in both pairs, and ahead at matched train time in both:** seed 2, and a place in the next H200 gate config and on the box list.
- **Below the control at the final and at matched train time in both pairs:** closed.
- **Anything between:** left there.

Whatever the score, the step-rate reading goes to the speed work.

**Queue** (`queue/screens_rl_b200.txt`, edited under the lock):
1. r105 lr 12e-6 seed 1
2. r113 stack3 seed 1
3. r115 p32 seed 0
4. r115 p32 seed 1
5. r105 lr 24e-6 seed 1

r105's two seed-2 items are dropped. Their seeds 1 stay, as LOG 11-20 14:24 decided. The five pending B200 workers (normal 29837206, 29837208, 29837209 and 29837211; preemptable 29837212) match the five items.

**Jobs.**
- **Running:** 29837204 runs r113 stack3 seed 0 on node5701-c1 (since 21:37, ends about 02:30).
- **Pending:**
  - the B200 workers above;
  - H200 preemptable 29837213 and 29837214 for r102 seeds 1 and 2;
  - the four older held workers.
- **GPUs in use:** 1 of 6.
- **Storage:** the pool is at 65 %. Scratch is at 3 % of space and 65 % of its file limit. The slurm logs show no quota errors.
- **INBOX:** unchanged since 11-19 20:45 (e66970b).

## 2026-11-21 02:39 EST: r113 stack3 seed 0 ends behind the native control (0.2733 against 0.3033; -0.0081 at matched steps); seed 1 decides between "closed" and "between"

**r113 stack3 seed 0** (lr 12e-6 on a cosine schedule, `sil_rows` 256 and the mini levels M2, M3 and M6 in the training mix; node5701-c1, 21:37 11-20 to 02:28 11-21) against r101's native control seed 0 (node5600-c1). The read is by script, as for sil256 (LOG 11-21 02:00).

| | arm | control |
|---|---|---|
| final dev (n = 600) | 0.2733 | 0.3033 |
| steps in 17,400 s | 31,122 | 34,893 |
| steps/s | 1.789 | 2.005 |
| mean `t_sample` / `t_update` (s) | 0.155 / 0.367 | 0.163 / 0.286 |
| mean entropy | 0.081 | 0.100 |
| mean of the last five evals | 0.2800 | 0.2920 |

| steps | shared evals | arm | control | arm minus control |
|---|---|---|---|---|
| 1 to 5,000 | 10 | 0.1480 | 0.1523 | -0.0043 |
| 5,001 to 10,000 | 10 | 0.2103 | 0.2163 | -0.0060 |
| 10,001 to 20,000 | 20 | 0.2365 | 0.2534 | -0.0169 |
| 20,001 to 30,000 | 20 | 0.2714 | 0.2722 | -0.0008 |
| 30,001 to 31,000 | 2 | 0.2792 | 0.3017 | -0.0225 |

Over all 62 shared evals the arm is -0.0081 (sd 0.0153, se 0.0019). It is behind at 42 of them. The training reward is not comparable, because the mini levels are in the mix.

**What this means under the r113 rule** (LOG 11-20 04:34). Seed 0 is below the control both at the final (-0.0300) and at matched steps. That is the bottom of the range I wrote (-0.03 to +0.06). Its parts did no better alone:
- lr 12e-6 seed 0 (constant) was level at matched steps and -0.013 at the final (LOG 11-20 09:23).
- sil256 seed 0 was level at both (LOG 11-21 02:00).

So on one seed the three changes do not add up to a gain. Seed 1 started at 02:35 on node5501-c1 (29837208) and ends about 07:30.
- If seed 1 is also below both ways, stack3 is closed and comes off the box queues on 11-24.
- Otherwise it is between, and a weak box candidate.

I let seed 1 run rather than stop it, since the rule needs both pairs.

**Jobs.**
- **Running (2 of 6 GPUs):**
  - 29837206 (node5702-c1): r105 lr 12e-6 seed 1, since 02:06;
  - 29837208 (node5501-c1): stack3 seed 1, since 02:35.
- **Pending:**
  - normal B200 29837209 and 29837211 (at the two-normal-GPU limit);
  - preemptable B200 29837212;
  - H200 29837213 and 29837214 (r102 seeds 1 and 2);
  - the four older held workers.
- **Queue:** the B200 queue holds r115 p32 seeds 0 and 1 and r105 lr 24e-6 seed 1.
- **Storage:** pool 65 % (df 364 GB available); scratch 3 % of space and 65 % of its file limit; no quota errors in the slurm logs.
- **INBOX:** unchanged since 11-19 20:45 (e66970b).

## 2026-11-21 07:39 EST: r105 lr 12e-6 seed 1 ends ahead of the native control (0.3233 against 0.2883; +0.0243 at matched steps, ahead at 85 of 89 evals); over two pairs +0.0096, so seed 2 is queued again for r105's three-pair rule; r113 stack3 seed 1 is level at matched steps, so stack3 is between (weak), not closed

Both seeds ran on normal B200 workers from scratch and are read by script against r101's native control seed 1 (node5602-c1).

**r105 lr 12e-6 seed 1** (lr 2x D 007's 6e-6, nothing else changed; node5702-c1, 02:06 to 06:57):

| | lr 12e-6 seed 1 | control seed 1 | lr 12e-6 seed 0 | control seed 0 |
|---|---|---|---|---|
| final dev (n = 600) | 0.3233 | 0.2883 | 0.2900 | 0.3033 |
| steps in 17,400 s | 48,070 | 44,862 | 36,808 | 34,893 |
| steps/s | 2.763 | 2.578 | 2.115 | 2.005 |
| mean `t_sample` / `t_update` (s) | 0.138 / 0.191 | 0.141 / 0.210 | 0.180 / 0.250 | 0.163 / 0.286 |
| mean of the last five evals | 0.3223 | 0.2857 | 0.2903 | 0.2920 |

| steps (seed 1) | shared evals | seed 1: arm minus control | seed 0, same block |
|---|---|---|---|
| 1 to 5,000 | 10 | +0.0035 | +0.0033 |
| 5,001 to 10,000 | 10 | +0.0140 | -0.0020 |
| 10,001 to 20,000 | 20 | +0.0268 | -0.0123 |
| 20,001 to 30,000 | 20 | +0.0202 | -0.0039 |
| 30,001 to 44,500 | 29 | +0.0360 | -0.0044 |

Over all 89 shared evals, seed 1's arm is +0.0243 (sd 0.0140, se 0.0015), ahead at 85 and behind at 3. The lead grows from about +0.004 in the first 5,000 steps to about +0.036 after step 30,000. At the control's last step (44,862) the arm stood at 0.3150, so the gain at the final is not only from its 3,208 extra steps. Seed 0's pair was -0.0051 over 69 evals and -0.0133 at the final (LOG 11-20 09:23).

**What this means under r105's rule** (LOG 11-19 09:28). The rule needs a gain of 0.015 or more at matched steps over three pairs to put lr 12e-6 into the next H200 gate config. A tie closes lr for the native regime. Over two pairs the mean is +0.0096 at matched steps and +0.0108 at the final, so the rule is neither met nor closed. The seed-2 pair would need +0.0258 or more at matched steps. That is about seed 1's size, so I give it well under even odds. The two pairs differ by 0.029, more than the paired noise I assumed (SE about 0.005 over three pairs). So the lr effect, if real, varies by seed, or one pair is noise. I expected about +0.02 at 2x (range -0.02 to +0.05), and the two-pair mean is inside that range, below its centre.

The seed-2 item was dropped at 02:00 for r115. It goes back into the queue now, ahead of lr 24e-6 seed 1, because only it can complete the rule for the one dose with a large pair. lr 24e-6 seed 2 stays out. lr 24e-6 seed 0 was +0.0020, so its seed 2 counts only if its seed 1 is large too.

**r113 stack3 seed 1** (lr 12e-6 cosine, `sil_rows` 256, mini levels M2, M3, M6; node5501-c1, 02:36 to 07:26):

| | stack3 seed 1 | control seed 1 |
|---|---|---|
| final dev | 0.2733 | 0.2883 |
| steps / steps/s | 38,299 / 2.201 | 44,862 / 2.578 |
| mean `t_sample` / `t_update` (s) | 0.127 / 0.298 | 0.141 / 0.210 |
| mean of the last five evals | 0.2800 | 0.2857 |

| steps | shared evals | arm minus control |
|---|---|---|
| 1 to 5,000 | 10 | -0.0013 |
| 5,001 to 10,000 | 10 | -0.0048 |
| 10,001 to 20,000 | 20 | -0.0039 |
| 20,001 to 30,000 | 20 | +0.0036 |
| 30,001 to 38,000 | 16 | +0.0046 |

Over 76 shared evals: +0.0001 (sd 0.0104), ahead at 36 and behind at 33. At the arm's last step the control stood at 0.2683, so the arm's final is +0.0050 at equal steps. The final is -0.0150 at equal time, because the 256 self-imitation rows lengthen the update.

**Under the r113 rule** (LOG 11-20 04:34). Seed 0 was below both ways (-0.0300 at the final, -0.0081 at matched steps; LOG 11-21 02:39). Seed 1 is below at the final but level at matched steps. "Closed" needs below both ways in both pairs, so stack3 is between, not closed. Among r113's arms it is the weakest candidate for the 11-25 box, behind sil256.

On the same seed, stack3 is -0.0217 against lr 12e-6 at matched steps (76 shared evals). stack3 peaks at the same lr, uses a cosine schedule, and adds sil256 and the mini levels. One seed cannot say which part takes back lr 12e-6's lead, so I draw no cause from it.

**Queue** (`queue/screens_rl_b200.txt`, edited under the lock at 07:38):
1. r105 lr 12e-6 seed 2
2. r105 lr 24e-6 seed 1

Workers:
- normal B200 29942141 (submitted 07:38 for the re-added item);
- preemptable 29837212.

**Jobs.**
- **Ran and exited:** 29837206 (lr 12e-6 seed 1) and 29837208 (stack3 seed 1) exited after their items.
- **Running:**
  - 29837209: r115 p32 seed 0 on node5701-c1, since 07:10, to about 12:00;
  - 29837211: r115 p32 seed 1 on node5800-c1, since 07:35, to about 12:30.
- **Pending:**
  - the two B200 workers above;
  - preemptable H200 29837213 and 29837214 (r102 seeds 1 and 2);
  - the four held workers.
- **GPUs in use:** 2 of 6.
- **Queue depth:** less than a day of B200 work is queued, to about 17:30 if the workers start at 12:00. The next items depend on r115 seed 0's step rate, which sets the speed timing item. I add them at the wake after it ends.

**Storage.**
- Pool: 64.5 % (df: 661G used, 364G free).
- Scratch: 3.2 % of space and 69.6 % of its file limit (696K files, up from 652K at 02:22). Each seed directory these runs leave on scratch holds four files, so the growth is not from their outputs. I watch it. Above 85 %, the workers go back to the pool after a write probe there.
- The slurm logs show no quota errors.

**INBOX:** unchanged since 11-19 20:45 (e66970b).

## 2026-11-21 12:40 EST: r115 (half the prompts per step) is closed: behind the native control in both pairs at the final (-0.033, -0.032), at matched train time (-0.012, -0.027) and at matched steps (-0.041, -0.040); much of a step is fixed, so r117 (twice the prompts per step) is queued; the sampler timing puts the update first in the speed work; r116 (U24) pretrains on a mix with keyword-filtered FineMath, its expected effect and rule written first

**r115 p32, both pairs** (`calibration/research_r115/d007rl_native_p32_b200`, B200, RL only; seed 0 on node5701-c1 from 07:10 to 12:11, seed 1 on node5800-c1 from 07:35 to 12:36; the control `d007rl_native_b200` ran seed 0 on node5600-c1 and seed 1 on node5602-c1):

| | seed 0 arm | seed 0 control | seed 1 arm | seed 1 control |
|---|---|---|---|---|
| final dev | 0.2700 | 0.3033 | 0.2567 | 0.2883 |
| steps | 63,559 | 34,893 | 57,695 | 44,862 |
| steps/s | 3.653 (1.82x) | 2.005 | 3.316 (1.29x) | 2.578 |
| mean of the last five evals | 0.2740 | 0.2920 | 0.2510 | 0.2857 |
| at the control's last step | 0.2467 | | 0.2433 | |

| arm minus control | seed 0 | seed 1 |
|---|---|---|
| at the final | -0.0333 | -0.0317 |
| at matched train time (evals; ahead / behind) | -0.0124 (64; 7 / 57) | -0.0273 (58; 3 / 55) |
| at matched steps (evals; ahead / behind) | -0.0409 (34; 0 / 34) | -0.0396 (44; 0 / 44) |
| matched train time, by hour 1 to 5 | -0.006, -0.016, -0.013, -0.012, -0.014 | -0.021, -0.025, -0.026, -0.035, -0.029 |

The step's parts, early (steps 100 to 2,100) and late (the last 2,000 steps), in seconds:

| | `t_sample` | `t_update` | wall per step | informative groups |
|---|---|---|---|---|
| seed 0 arm, early | 0.180 | 0.219 | 0.422 | 30.1 |
| seed 0 arm, late | 0.100 | 0.136 | 0.256 | 14.2 |
| seed 0 control, early | 0.175 | 0.377 | 0.602 | 56.6 |
| seed 0 control, late | 0.136 | 0.170 | 0.338 | 29.2 |
| seed 1 arm, early | 0.111 | 0.189 | 0.330 | 28.7 |
| seed 1 arm, late | 0.097 | 0.172 | 0.293 | 14.3 |
| seed 1 control, early | 0.149 | 0.276 | 0.460 | 56.7 |
| seed 1 control, late | 0.143 | 0.202 | 0.385 | 26.1 |

Seed 1's peak GPU memory was 61.5 GB (the control's, 75 GB). Neither stdout has an error line.

**Under r115's rule** (LOG 11-21 02:00): below the control at the final and at matched train time in both pairs, so
r115 is **closed**. I expected a step rate of 1.45x to 1.6x and a final at 0.00 (range -0.04 to +0.03). The final is
near the low end of that range. The step rate is 1.29x on seed 1. Seed 0 shows
1.82x, but its control ran on the slow node5600-c1 (2.005 steps/s against 2.578), so that
figure measures the node as much as the arm.

**What it says.** On seed 1, half the rows cut the sampler's time by 26 % early and
32 % late, and the update's by 31 % early and
15 % late. So a large part of a step does not scale with the rows. Each step with half the
rows learns less: the arm is behind at every shared eval at matched steps, by about 0.034 before step 10,000 and 0.040
to 0.044 after. More, smaller steps lose more per step than they gain in steps. The other direction is the open one:
twice the rows a step, for fewer steps (r117 below).

**r117: twice the prompts per step.** The arm is `calibration/research_r117/d007rl_native_p128_b200` on the B200, RL
only, not gated, seeds 0 and 1. It is r101's native control with these changes:
- `prompts_per_step` 128 against 64, with `group_size` 8 kept, so 1,024 sampled rows a step;
- `sil_rows` 128 against 64, so the self-imitation share per sampled row is unchanged;
- `eval_every` 250, so it evaluates every 32,000 prompts, as the control does at 500 (evals are not counted in the
  training time).

Everything else is the same: lr 6e-6, `time_budget_s` 17,400 and D 007's seed-matched pretrained weights. It is a
config-only change.

Expected:
- **Step rate:** 0.63x to 0.71x the control's, 28,000 to 32,000 steps on seed 1's node. A straight line through each
  part's time at 256 and 512 rows (seed 1, above) gives about 0.22 s to sample and 0.45 s to update early in a run,
  and 0.24 s and 0.26 s late.
- **Memory:** about 103 GB at peak (75 GB at 512 rows, 61.5 GB at 256), within the B200's 180 GB.
- **Score:** centre +0.005 at the final, range -0.03 to +0.035. Per step it should gain about what r115 lost at
  matched steps (0.04 for half the rows), but it makes about two thirds of the control's steps, and the control's
  curve still rises at the end. The paired noise is about 0.015 per pair (control seed sd 0.011).

Rule, written before it runs (as r115's):
- **+0.02 or more at the final in both pairs, and ahead at matched train time in both:** seed 2, and a place in the
  next H200 gate config and on the box list.
- **Below the control at the final and at matched train time in both pairs:** closed.
- **Anything between:** left there.

Its matched-steps reading goes to the speed work in any case: with r115 it gives the gain per step at three batch
sizes.

**Queue** (`queue/screens_rl_b200.txt`, edited under the lock at 12:35):
1. r105 lr 24e-6 seed 1
2. r117 p128 seed 0
3. r117 p128 seed 1

Normal B200 workers 29966173 and 29966174 were submitted when the two r117 items were queued; with preemptable
29837212 that is one worker per long item. A worker takes the queue's first item, so 29966173 runs lr 24e-6 seed 1
(corrected at 12:47; this said the preemptable worker keeps it).

**The sampler timing** (speed item 1 of LOG 11-21 02:00). Job 29963843 (normal B200, node5800-c1, from 12:28) ran
`scripts/research/profile_sampling.py --backend native --floor` on r101's config. That is the native sampler with the
answer grammar, then its decode loop's GPU work alone:
- `forward`: the CUDA-graph forward per token;
- `sample`: the forward plus the sampling kernels;
- `sample_sync`: also the per-token copy of the sampled ids to the host.

There is no grammar and no stop check in these three, and each runs for the constrained calls' mean longest
completion. The `--floor` and `--prompts-per-step` options are new in the script and off by default. It is timing
only. It ran from two checkpoints, each at 64 and 32 prompts:
- late: r93's `d007s0_final`, after RL;
- early: D 007's seed-0 pretrained weights.

Times in ms:

| checkpoint, prompts | longest completion | call wall | prefill | decode step, constrained | `forward` | `sample` | `sample_sync` | grammar and stop share |
|---|---|---|---|---|---|---|---|---|
| late 64 (512 rows) | 36.2 | 202 | 19.6 | 5.02 | 4.43 | 4.58 | 4.62 | 9 to 18 % |
| late 32 (256 rows) | 35.4 | 130 | 10.9 | 3.38 | 2.97 | 3.05 | 3.08 | 10 to 17 % |
| early 64 (512 rows) | 256.0 | 1537 | 19.6 | 5.93 | 5.41 | 5.55 | 5.59 | 6 to 7 % |
| early 32 (256 rows) | 256.0 | 974 | 10.9 | 3.76 | 3.28 | 3.36 | 3.39 | 11 to 13 % |

"Decode step, constrained" is the call's wall time less the prefill, over the longest completion. The share is that
step's excess over `sample`. The first figure takes out the prefill and the second does not.

The rule I set when I submitted it, but did not log, was this:
- At 512 rows, if the grammar and stop checks take 40 % or more of a decode step, write the GPU-table sampler.
- If under 20 %, the update goes first.

They take 9 to 18 %, so **the update goes first**. On the late checkpoint, at
most 18 to 36 ms of a 202 ms call could be
removed. That is under a tenth of a step. The per-token copy to the host costs about 0.04 ms a token. The constrained
calls' CPU time for the grammar (0.16 s late, 0.96 s early) mostly overlaps the GPU's work.

The forward is the bulk of a decode step: 2.97 ms at 256 rows and 4.43 ms at 512, so most of it scales
with the rows. A faster forward would mean lower-precision matmuls, which need a loss comparison (INBOX), so I do not
start it now. Speed item 2 is the update. Its timing item comes first: where `t_update`'s 0.17 to 0.38 s goes (the
policy rows' forward and backward, the self-imitation rows, the optimizer step), on one node and both checkpoints,
before any code.

**r116 (U24): D 007's pretraining on a mix with keyword-filtered FineMath.**

The data:
- **Source:** FineMath, HuggingFaceTB/finemath, the `finemath-4plus` subset at revision
  e92b25a616738fe95dc186b64dfb19f9c8525594, shards 0 to 2 of 64. They are already on the pool from the FineMath
  pilot (sweep_d15).
- **Filter:** `scripts/keyword_filter.py drop` (new, CPU only, no model; tests in `tests/test_keyword_filter.py`)
  removes Countdown items by a keyword rule. It drops every document that matches any of:
  - Countdown or a count-down;
  - the 24 game;
  - "make" or "reach" 24;
  - a numbers game, round or puzzle;
  - Krypto;
  - a target number, value or total;
  - each, every or all of the numbers used once.

  It over-drops on purpose. RULES_v3 allows a general dataset "once those items are removed (a keyword filter is
  enough)".
- **Mix:** `scripts/research/prep_r116_data.sh` (new) filters, tokenizes with D 007's tokenizer, and appends two
  copies to `mix_0.2.bin` (`math_mix.py mix --frac 0.3 --max-repeat 2`). By tokens that is about 31 % FineMath, 14 %
  dm_math and 55 % FineWeb-Edu, against D 007's 20 % dm_math and 80 % FineWeb-Edu. Pretraining samples windows
  uniformly from the file.
- **Reasoning cues:** the script also counts them per million characters (`keyword_filter.py cues`) in FineWeb-Edu,
  dm_math, FineMath and the filtered FineMath, to show how much written reasoning each holds. Examples are "step by
  step", "let's", "therefore" and "the answer is".
- **The job:** CPU job 29965651 (`mit_normal`, 16 cores, 2 h) builds it on scratch, with a planned start of 15:00.
  The drop counts, cue counts and md5 sums go into the LOG when it ends.

The arms:
- **Pretraining:** `calibration/research_r116/d_fmkw2_muon1338m_pre` (pretrain_only), seeds 0 and 1. It is D 007's
  recipe (Muon lr 4e-3, micro-batch 32), stopped at D 007's 1,337,982,976 tokens as r84's `muon_1338m` is, on a
  B200.
- **RL:** `calibration/research_r116/d116rl_native_fmkw2_b200`. It is r101's native control, unchanged, from those
  weights. Each seed is paired with `d007rl_native_b200`'s.

The control's weights are D 007's own: an H200 run stopped by time at about the same token count. So a pair compares
the data and the pretraining GPU together. A B200 control pretraining on `mix_0.2.bin` (r84's config) runs only if
the arm clears the bar.

Why: this is U24. With `think_budget` 0, the `<answer>` tag is never forced, so free text before it is allowed. The
record seeds answer in about 24 tokens. More worked maths in pretraining could change what RL finds.

Expected:
- **Score at the final:** centre -0.01, range -0.05 to +0.03. The FineMath pilot (sweep_d15) did not change the
  regime. The math mixes of D2 are closed (dm51 was +0.010). r86 found twice D 007's tokens worse (LOG 11-19 05:26).
  Here dm_math, the closest data to the task, falls from 20 % to 14 % of the tokens.
- **Completion length:** unchanged at about 24 tokens late in a run. If it moves to written reasoning (a mean over 40
  tokens), I report that whatever the score.

Rule, written before it runs:
- **+0.02 or more at the final in both pairs:** seed 2, and the B200 control pretraining (r84's config) for seeds 0
  to 2, so that the data's effect is read alone. The H200 gate follows if the gain holds.
- **Below the control at the final and at matched steps in both pairs:** closed.
- **Anything between:** left there.

The regime reading goes into the LOG in any case.

**Running r116.** When the data is built, the two pretraining items go into a new queue `pre_r116`, with one
preemptable B200 worker (2 h 30 min). The RL items follow when the weights exist, with one worker each.

**Other ideas.** U22 was answered at twice D 007's tokens by r86 (LOG 11-19 05:26). U23 goes into the 11-25 box plan,
written on 11-24.

**Jobs.**
- **Exited:**
  - 29837209 (r115 seed 0, to 12:11);
  - 29837211 (r115 seed 1, to 12:36);
  - 29963843 (the sampler timing, 12:28 to 12:37).
- **Running:**
  - 29942141: r105 lr 12e-6 seed 2, normal B200 node5701-c1, from about 12:04 to about 17:00;
  - 29966173: r105 lr 24e-6 seed 1 (the queue's first item), normal B200 node5501-c1, from 12:38 to about 17:40.
    Corrected at 12:47: this line said r117 seed 0; the worker log shows lr 24e-6 seed 1.
- **Pending:**
  - normal B200 29966174 and preemptable B200 29837212, for r117 seeds 0 and 1 (still one worker per queued
    long item);
  - preemptable H200 29837213 and 29837214 (r102 seeds 1 and 2);
  - CPU 29965651 (r116 data);
  - the four held workers.
- **GPUs in use:** 2 of 6.
- **Queued work:** about 19 GPU-hours on the B200 (lr 12e-6 seed 2's rest, r117's two seeds, lr 24e-6 seed 1), about
  31 with r116's four items, and 12 on the H200 (r102 seeds 1 and 2). That is more than a day. The update's timing
  item follows when a normal worker is free.

**Storage.**
- Pool: 64.5 % (df: 661G used, 364G free).
- Scratch: 3.2 % of space and 74.7 % of its file limit (747K files at 12:30, up from 696K at 07:29). My runs add a few
  files per seed directory, so the growth is not theirs. Above 85 %, the workers go back to the pool after a
  write probe there.
- The slurm logs show no quota errors.

**INBOX:** unchanged since 11-19 20:45 (e66970b).

## 2026-11-21 14:29 EST: r118 profiles the update step (timing only) at r106's final-weights settings; the expected result and what it decides are written first

**Why.** A native-control step takes about 0.39 s on the B200: about 0.24 s in the update and 0.15 s in the sampler
(r106 at D 007's final weights; the sampler timing in LOG 11-21 12:40). The update's parts in r106, in seconds per step:

| part | s per step |
|---|---|
| body forward | 0.068 |
| mask | 0.015 |
| head | 0.004 |
| backward | 0.082 |
| SIL pass | 0.055 |
| replay batch | 0.014 |
| optimizer | 0.002 |

The SIL pass holds 64 rows, an eighth of the 512 policy rows, but takes about a third of their time. So the update
is dominated by a fixed cost per call, not by the tokens. r106's node trace showed GPU utilisation between 28 and 93 %
and power between 340 and 530 W.

**r118.** `scripts/research/profile_update.py` runs grpo with r106 `final_native_ctl`'s config (D 007 seed 0's final
weights; no saving, no evals, the parts timing off). It times 20 updates, then profiles 8 more one at a time with
torch.profiler (CPU and CUDA), and stops. It reports:
- the GPU's busy fraction in the update (device activity over wall time);
- kernel launches, synchronisations and device-to-host copies per update;
- the top ops by self CUDA time and by self CPU time.

Nothing is saved or scored. One preemptable B200 job, 29980436, about 15 minutes; results go to
`calibration/research_r118/final_native_ctl/seeds/0/profile.json`.

**Expected, written first.**
- Busy fraction 0.3 to 0.6 of the update's wall time.
- Over 2,000 kernel launches per update.
- A handful of syncs: one `.item()` per micro-batch and a few more.

The profiler slows the CPU side, so profiled updates run longer than the 20 unprofiled ones. The busy fraction is
read against both.

**What it decides.**
- **Busy under 0.6, with many launches:** the next arm captures the training forward and backward (CUDA graphs or
  `torch.compile`, off by default). Removing half the idle time would cut the update by about 0.06 s, about 15 %
  more steps in the same time. The maths is the same, so a timing check keeps it without a paired win (INBOX); the
  loss comparison applies if the precision changes.
- **Busy over 0.8:** the update is compute-bound, so the work is cut instead: the padding in the micro-batches, or
  the separate SIL pass.
- **Host waits on syncs over 10 % of the update:** those syncs go first, as the cheapest change.

**Jobs.**
- **Running:** 29942141 (r105 lr 12e-6 seed 2, to about 17:00) and 29966173 (r105 lr 24e-6 seed 1, to about 17:40).
- **Pending:**
  - 29980436 (r118, preemptable B200);
  - 29966174 and 29837212 (r117 seeds 0 and 1);
  - 29837213 and 29837214 (r102 seeds 1 and 2, H200);
  - CPU 29976793 (r116 data; its time limit was cut to 2 h after its planned start moved to 11-26);
  - the four held workers.
- **GPUs:** 2 in use. If every pending GPU job started, that would be 6.

## 2026-11-21 14:59 EST: r118: the update step is overhead-bound (GPU busy 0.40 of the update's wall time, 11,689 kernel launches per update), so the next speed arm removes overhead; cuDNN's attention takes 81 ms of CPU per update, so r119 first tries the memory-efficient attention kernel; the r116 data job moves to a GPU node, as CPU nodes do not reach the pool

**r118 result.** One preemptable B200 (29980436, node5800-c1, torch 2.13.0+cu130), 80 s; numbers from its
`profile.json`, medians over the updates:

| measure | expected (written first) | r118 |
|---|---|---|
| wall per update, 19 unprofiled | - | 0.417 s |
| wall per update, 8 profiled | - | 0.322 s |
| GPU busy per update | - | 0.128 s |
| busy fraction | 0.3 to 0.6 | 0.40 of the profiled wall (0.31 of the unprofiled) |
| kernel launches per update | over 2,000 | 11,689 |
| synchronisations per update | a handful | 64.5 |
| host time waiting in syncs | the 10 % line | 26 ms, 8 % of the profiled wall |
| peak GPU memory | - | 31 to 33 GB |

There are two kinds of update: 5 of the 8 had 9,531 launches and 54 syncs, 3 had about
15,300 and 82. Which part adds the rest is not checked yet. The 19 unprofiled updates (the run's
first) took longer than the profiled ones, so the profiler's own cost is not what sets the wall time here; the
busy fraction is under 0.6 against either.

Where the time goes, per update:
- **CPU (self time):** cuDNN attention backward 41.6 ms and forward
  39.4 ms for 143 calls, about 0.57 ms of CPU per call;
  `cudaLaunchKernel` 31.8 ms (9,878 calls); `cudaStreamSynchronize` 26.0 ms; `aten::mm` 24.0 ms.
- **GPU (self time by op, out of 128 ms busy):** copies 26.1 ms (2,936 calls), multiplies
  22.0 ms, matmuls 16.1 ms, `cat` 11.0 ms, cuDNN attention 20.1 ms. The matmuls are a small
  part; elementwise work and copies lead. 1,249 of the copies per update are bf16 casts (5.8 ms), which
  fits autocast casting the fp32 master weights again for each micro-batch (not checked).

**Decision, by the rule written first.** Busy is under 0.6 with many launches, so the next arm removes overhead
from the training forward and backward. Host waits in syncs are 8 %, under the 10 % line, so the syncs do not
go first.

**r119, written first.** One cheaper same-maths test goes before capture or `torch.compile`. cuDNN's attention
uses 81 ms of CPU per update for 20 ms of GPU work; a likely cause is a new cuDNN plan for each new
micro-batch shape (not checked). `profile_update.py --sdpa` (new, off by default) limits the update's attention to
named PyTorch backends. One preemptable B200 job profiles the update twice on one node, with the settings of r118
(20 timed updates, then 12 profiled):
- `sdpa_efficient`: PyTorch's memory-efficient kernel, which takes the padding mask. The flash kernel takes no
  mask, so it is not tried.
- `sdpa_default`: PyTorch's own choice (cuDNN) again, as the noise reference against r118.

Expected:
- the efficient kernel costs tens of microseconds of CPU per call, so about 70 of the 81 ms leave the CPU side;
- its GPU kernels are slower than cuDNN's (say 5 to 15 ms more busy time), which fits in the idle time;
- so the update's wall time falls by 0.04 to 0.07 s (10 to 17 %) in both medians, and the busy fraction rises
  towards 0.5.

What it decides:
- **Down by 0.03 s or more in both medians, and by more than `sdpa_default` moves against r118:** an `sdpa_backend`
  option goes into GRPOConfig (off by default). Then one native-control seed runs with it on the B200 as a timing
  check. The maths is the same, so it is kept on time without a paired win (INBOX); its score is read against the
  control's pairs only as a check that nothing broke. Capture or compile then goes on top.
- **Less:** capture or compile goes next. The micro-batches' padded lengths vary, so capture needs a fixed set of
  lengths.

**r116 data job.** 29981583 failed its 5 min start check on node2702 at 14:42. A diagnostic (29982547, node1702,
from 14:46) found:
- the pool is not mounted there: `ls -d` of my pool directory, a `stat` of the venv, the venv's python and a
  `stat` of a FineMath shard each did not finish in 90 s;
- home and scratch respond in 0.5 s.

So CPU nodes do not reach the pool (at least today), which also explains 29965651 (node3107), 29980655 (node3409)
and 29981583. GPU nodes do: r118 read the venv and the weights from it.

The prep now runs after r119's two profiles in the same preemptable B200 job, 29983376, which leaves the GPU idle
during the prep. An L40S job for the prep alone (29982809, 14:51) was cancelled before it started: B200 nodes are
known to mount the pool, and one job carries both.

**Jobs.**
- **Running:**
  - 29942141 (r105 lr 12e-6 seed 2, to about 17:00);
  - 29966173 (r105 lr 24e-6 seed 1, to about 17:40);
  - 29837214 (r102 seed 2, H200 node4601, started again at 14:40 after a preemption).
- **Pending:**
  - 29983376 (r119 and the r116 data, preemptable B200);
  - 29966174 and 29837212 (r117 seeds 0 and 1);
  - 29837213 (r102 seed 1, preempted at 11:20 after 32 min);
  - the four held workers.
- **GPUs:** 3 in use. If every pending GPU job started, that would be 6.

## 2026-11-21 17:13 EST: r105 lr 12e-6 meets its rule (+0.016 over three pairs at matched steps); r120 written first: r102's H200 gate with lr 1.2e-5

**r105 lr 12e-6 seed 2** (normal B200 29942141, node5701-c1, ended 16:55): 0.3067 after 40,426 steps against the
native control's seed 2, 0.2833 after 44,987 (node5600-c1).
- Matched steps: +0.0290 over 80 shared evals (ahead at 77, behind at 2). By step block: 0-5k +0.0073, 5-10k
  +0.0193, 10-20k +0.0368, 20-30k +0.0361, 30-40k +0.0299. Last five evals: 0.3123 against 0.2803.
- The control stood at 0.2733 at step 40,000, so the arm is +0.0333 at equal steps at its final.
- The arm took 2.32 steps a second against the control's 2.59 (mean t_sample 0.147 s against 0.141, t_update
  0.240 s against 0.209). Seeds 0 and 1 took more steps than their controls, so I put this down to the node, not
  the lr.

All three pairs (computed by script from the eval logs):

| seed | lr 12e-6 final (steps) | control final (steps) | final diff | matched steps (shared evals) |
|---|---|---|---|---|
| 0 | 0.2900 (36,808) | 0.3033 (34,893) | -0.0133 | -0.0051 (69) |
| 1 | 0.3233 (48,070) | 0.2883 (44,862) | +0.0350 | +0.0243 (89) |
| 2 | 0.3067 (40,426) | 0.2833 (44,987) | +0.0233 | +0.0290 (80) |
| mean | 0.3067 | 0.2917 | +0.0150 (SE 0.0146) | +0.0161 (SE 0.0107) |

The rule (LOG 11-19 09:28): "A gain of 0.015 or more over the three pairs puts that lr into the next H200 gate
config. A tie closes lr for the native regime." **Met, at the bar:** +0.0161 at matched steps and +0.0150 at the
finals, with seed 0 behind. The arm's finals also spread more (sd 0.0167 against the control's 0.0104). Not gated:
B200, RL only from D 007's checkpoints. lr 24e-6: seed 0 was +0.0020 at matched steps and -0.0067 at the final;
seed 1 (29966173) ends about 17:40; its seed 2 runs only if seed 1 is large.

**r120, written first.** `calibration/research_r120/d007_speed_lr12e-6_h200`: r102's gate config with lr 1.2e-5
and nothing else changed: fused Muon pretraining for 46 min, then the native RL settings for 18,000 s, no_save.
Seeds 0-2 on `queue/gate_r120_h200.txt`, preemptable H200 workers.
- **Why the 18,000 s form, not r114's 6 h normal form:** H200 waits on `mit_normal_gpu` ran to a median of 214 h
  in November (INBOX 11-13), so a preemption costs less than that wait. This form also pairs with r102's seeds,
  and a record from it runs on preemptable H200 workers, as D 007's did.
- **Expected:** r102 seed 0 ended at 0.270; seeds 1 and 2 are still running. Add r105's +0.015 and the centre is
  about 0.285 a seed, if the B200 RL-only gain carries over to the end-to-end H200 run. The seed sd is between
  0.0104 (r101's native seeds) and 0.0167 (r105 lr 12e-6). On that:
  - one seed reaches 0.281 with probability 0.59 to 0.65;
  - all three reach it with probability 0.21 to 0.27;
  - a five-seed mean reaches it with probability 0.70 to 0.81.
  If the gain does not carry over (centre 0.278), those are 0.39 to 0.43, 0.06 to 0.08 and 0.26 to 0.34.
- **Rule:**
  - All three seeds at 0.281 or more: the three-seed form is met. A five-seed record attempt follows with r120's
    config without no_save (a `record-<slug>` queue, `scripts/run_seed_v3.sh`, H200), its `config.json`
    committed first.
  - Otherwise, a mean of the three of 0.276 or more: seeds 3 and 4 run for the five-seed form (mean at least
    0.281, four of five above D 007's dev mean 0.251, at most one collapsed).
  - Mean under 0.276: no record from lr alone. The pairs with r102's seeds at matched steps say whether the gain
    carried over to the H200. The next gate waits for another B200 gain to stack on it.
  - Paired with r102 by seed number, at matched steps and at the final, with the steps and node beside each.

**Placement.** INBOX 11-13: when H200 work is waiting, keep slots for it. The four preemptable slots now go to H200
work (r102 seeds 1 and 2, r120) plus r119 and r116's data. B200 screens move to the two normal slots, where B200
waits are short (median 0.1 h).
- At 17:13 I cancelled preemptable B200 worker 29837212 (pending, for r117 seed 1). I submitted preemptable H200
  worker 29995938 for r120's first item.
- r117 seed 1 gets a normal B200 worker when 29966173 exits (about 17:40). Until then the cap is full.
- r120 seeds 1 and 2 get preemptable H200 workers as preemptable slots free: 29983376 after r119 and the data,
  and r102's two workers when they end.

**Jobs.**
- **Running:**
  - 29966173 (r105 lr 24e-6 seed 1, normal B200 node5501-c1, to about 17:40);
  - 29966174 (r117 p128 seed 0, normal B200 node5501-c1, from 17:02, to about 22:00);
  - 29837214 (r102 seed 2, preemptable H200 node4601, since 14:40).
- **Pending:**
  - 29837213 (r102 seed 1, preemptable H200);
  - 29983376 (r119, then r116's data, preemptable B200);
  - 29995938 (r120, preemptable H200);
  - the four held workers.
- **GPUs:** 3 in use. If every pending GPU job started, that would be 6.

## 2026-11-21 17:59 EST: r105 lr 24e-6 seed 1 is large (+0.031 at matched steps, ahead at 88 of 89 evals), so its seed 2 runs; U25 (a doubling search on pretraining size for where reasoning appears, without RL) is planned, with what counts as "reasoning appears" written before any run; the data job moves to a normal B200

**r105 lr 24e-6 seed 1** (normal B200 29966173, node5501-c1, 12:38 to 17:29): 0.3367 after 47,812 steps against
the native control's seed 1, 0.2883 after 44,862 (node5602-c1).
- Matched steps: +0.0311 over 89 shared evals (ahead at 88, behind at 0). By step block: 0-5k +0.029, 5-10k
  +0.031, 10-20k +0.026, 20-30k +0.031, 30-50k +0.036. The lead is there from the first 5,000 steps
  and does not shrink.
- Final: +0.0483.

Both pairs (computed by script from the eval logs):

| seed | lr 24e-6 final (steps) | control final (steps) | final diff | matched steps (shared evals) |
|---|---|---|---|---|
| 0 | 0.2967 (45,636) | 0.3033 (34,893) | -0.0066 | +0.0020 (69) |
| 1 | 0.3367 (47,812) | 0.2883 (44,862) | +0.0484 | +0.0311 (89) |
| mean | 0.3167 | 0.2958 | +0.0208 (SE 0.0275) | +0.0166 (SE 0.0146) |

**What this means.** lr 24e-6's seed 2 was to run only if its seed 1 is large (LOG 11-21 07:39). +0.031 is the
largest pair in r105, so seed 2 runs. Under r105's rule (0.015 or more at matched steps over three pairs) it needs
about +0.012 or more. lr 12e-6's seed-2 pair was +0.0290, so I give it better than even odds.
- At both doses the seed-0 pair is level (-0.005 and +0.002) and the others are large. The control's seed 0 ended
  highest of its three (0.3033 against 0.2883 and 0.2833), so the seed-0 pairs say more about that control run
  than about lr.
- At seed 1, 4x is ahead of 2x: +0.0311 against +0.0243 at matched steps, finals 0.3367 against 0.3233.
- If seed 2 meets the rule and the three-pair mean at matched steps is above lr 12e-6's +0.0161, the next H200
  gate after r120 uses lr 2.4e-5 (a new config, written then; r120 is not changed). If it meets the rule without
  being above, lr stays at 1.2e-5. If it falls short, lr 24e-6 is closed for the native regime.

**Placement.** Data job 29983376 (r119, then r116's data) had waited 3 h on preemptable B200 (reason Priority, at
about 32,000). r116 and U25's second-phase data both wait on it.
- At 17:50 I cancelled it and the normal B200 worker 29999120 (pending, submitted at 17:48 for lr 24e-6 seed 2).
- The same job went to `mit_normal_gpu` as 29999304 and started at once on node5800-c1 (2:30 limit).
- lr 24e-6 seed 2 moved to preemptable B200 worker 29999306. Three of the four preemptable slots stay with H200
  work (r102 seeds 1 and 2, r120 seed 0).
- When 29999304 ends, its normal slot goes to a normal B200 worker for r117 seed 1.

**U25 (INBOX 11-21, 17:21 to 17:26).** This is a request to run it, not only to judge it:
- a doubling search on pretraining size for where a track D model starts to reason;
- measured on the pretrained model without RL, outside the rules and unranked;
- the first rung is 3x D 007's size; halve down if reasoning shows, double up if not;
- no keyword filter is needed, and reasoning-heavy data can go in the second half;
- judging by eye is fine, and size, data mix and the rest come after reasoning appears.

The plan below is written before any U25 run.

**Rungs.** Parameters and tokens grow together at 20 tokens per parameter (the INBOX estimate: about 330M on 6.6B
tokens).
- The architecture is D 007's: Llama-style, TinyLlama tokenizer, tied embeddings, 64-wide heads.
- The optimizer is Muon with D 007's settings.
- Parameter counts were computed by script.

| rung | width x layers | parameters | tokens | 6NT FLOPs |
|---|---|---|---|---|
| 1x, D 007's base (exists) | 768 x 12 | 110M | 1.34B as trained (12 per parameter) | 0.9e18 |
| 3x, first | 1024 x 24 | 332M | 6.6B | 1.3e19 |
| 6x | 1280 x 32 | 668M | 13.4B | 5.4e19 |
| 12x | 2048 x 24 | 1,270M | 25.4B | 1.9e20 |
| 1.5x | 896 x 16 | 182M | 3.6B | 4.0e18 |
| 0.5x | 640 x 10 | 69M | 1.4B | 5.7e17 |

**Where each rung runs.**
- **1x:** D 007's base (seed 0's pretrained weights in the pool) is probed first, as the reference.
- **3x:** one normal B200 job on Engaging for 330 minutes, with the cosine ending on time (the script decays the lr
  on the larger of the step and time fractions).
  - At r84's 18 % MFU that is about 4.0B tokens, 12 per parameter (D 007's ratio); at 25 % it is about 5.5B.
  - The 6.6B form needs 8 to 10 h on one B200. That is longer than a normal job, and the script has no resume yet.
  - It starts on 11-22, when the code below is in, and is read on 11-23.
- **The 11-25 box** (8x H100, 24 h) runs the next rung, chosen from 3x's result.
  - This needs data-parallel training in `scripts/pretrain_scratch.py` (torchrun, one process a GPU). It is added
    and tested on two Engaging GPUs before 11-24.
  - **If 3x is no:** 6x on all eight GPUs, about 6 to 7 h at 30 % MFU, probed on the box. If 6x is also no, 12x
    needs about 23 h on eight H100s at 20 tokens per parameter (14 h at 12). It then goes to the 11-28 or 12-09
    box, planned the day before.
  - **If 3x is yes:** 1.5x, 1x on the new data and 0.5x run on single GPUs side by side, an hour or two each. The
    rest of the box goes to the A arms (U23).
- U25 on Engaging stays inside the six-GPU cap. Its outputs never feed a ranked screen or a record.

**Data.** Two phases, split by run progress (the same fraction the lr schedule uses):
- **First half:** D 007's `mix_0.2.bin` (FineWeb-Edu with dm_math, 2.32B tokens).
- **Second half,** reasoning-heavy, sampled by weight:
  - FineMath-4plus 0.5 (shards 0-2 as downloaded, 529M tokens; not keyword-filtered, which INBOX allows for U25);
  - dm_math 0.25 (299M tokens);
  - FineWeb-Edu 0.25.
- At 3x on Engaging, FineMath repeats about twice in the second phase. The box rungs download more FineMath shards
  on the box.
- The weights are a first choice, not tuned (INBOX: the data mix comes after reasoning appears).

**Measure.** `scripts/research/probe_reasoning.py` (written next) probes base weights only, without RL. Every rung
and D 007's base get the same settings, on one GPU for some minutes. Three prompt sets:
- **A:** Countdown with track D's own `r1` prompt (zero-shot; the completion opens in `<think>`). The first 200 dev
  puzzles (`data/ladder_v3_dev.jsonl`), 4 samples each at temperature 1.0, 256 new tokens.
- **B:** Countdown with the `search` prompt (three worked trial-and-error examples, in `rlvr_speedrun/countdown.py`).
  The same puzzles and sampling.
- **C:** GSM8K word problems: 100 questions from the pool copy (train split), with 4 other worked questions in
  front, greedy, 256 new tokens. It checks that what B finds is not only the Countdown format.

Numbers for each set:
- step accuracy: the share of written `a op b = c` steps with correct arithmetic;
- grounded-chain rate: the share of samples with at least two correct steps where every operand is a given number
  or an earlier result;
- solve rate (Countdown: a valid expression equal to the target; GSM8K: the final number);
- well-formed rate;
- median length of the working, in tokens;
- loop rate (a 4-gram repeated four or more times).

By eye: 20 samples of each set go to a markdown file next to the run. They are the same puzzles and questions for
every rung, read beside D 007's base.

**What counts as "reasoning appears"** (fixed now):
- **Yes** when set B shows both of these:
  - by eye, at least 10 of the 20 samples are working, not imitation: the model combines the puzzle's own numbers,
    computes mostly right, compares with the target, and tries again or stops (not copied example text, not a
    loop, not number salad);
  - a grounded-chain rate of 0.5 or more and step accuracy of 0.7 or more, each at least 0.2 above D 007's base.
- **Partial** when B's chain rate is at least 0.1 above D 007's base but under the yes bar, or 5 to 9 of the 20
  samples are working. For the direction of the search, partial counts as no.
- **No** otherwise.
- Sets A and C are reported beside it. A yes on B together with working in at least 5 of 20 samples on A (no
  examples given) is a strong yes, because A is the prompt RL starts from.
- **The search:** yes at 3x goes down (1.5x, then 1x on the new data, then 0.5x) until a rung says no. No at 3x
  goes up (6x, then 12x). The smallest yes is where the size, data and length work starts (INBOX).

**Expected**, before any run:
- D 007's base copies the examples' form in B, with step accuracy under 0.5 and few grounded chains. dm_math gives
  it some arithmetic, and after RL the record seeds answer in about 24 tokens.
- 3x on the two-phase data: partial (chain rate 0.3 to 0.5).
- 6x: about even odds of a yes.
- 12x: likely yes on B, still uncertain on A.

**Order:**
1. 11-22:
   - the two-phase data option in `pretrain_scratch.py` (off by default, with tests);
   - `probe_reasoning.py`;
   - D 007's base probed (one B200, under 30 min);
   - the 3x config `calibration/research_r121/u25_3x_twophase` (pretraining only).
2. 11-22 to 11-23: 3x on a normal B200, then probed. Read in the LOG with samples.
3. By 11-24: data-parallel training and resume in `pretrain_scratch.py`, tested on two GPUs. The 11-25 box plan
   with U25's next rung.
4. 11-25: the box.

**Jobs.**
- **Running:**
  - 29966174 (r117 p128 seed 0, normal B200 node5501-c1, from 17:02 to about 22:00);
  - 29999304 (r119's two profiles, then r116's data; normal B200 node5800-c1, from 17:50, 2:30 at most);
  - 29837214 (r102 seed 2, preemptable H200 node4601, since 14:40).
- **Pending:**
  - 29837213 (r102 seed 1, preemptable H200);
  - 29995938 (r120 seed 0, preemptable H200);
  - 29999306 (r105 lr 24e-6 seed 2, preemptable B200);
  - the four held workers.
- **GPUs:** 3 in use. If every pending GPU job started, that would be 6.

## 2026-11-21 18:16 EST: r119 does not meet its rule (the memory-efficient attention takes 0.16 s off the unprofiled update but only 0.009 s off the profiled one), so capture or compile is next; r116's data is ready and its pretraining runs; U25's code is in, its second phase uses the keyword-filtered FineMath, and D 007's base is probed first, with a short smoke run of the 3x rung

**r119** (LOG 11-21 14:59; job 29999304 on one normal B200, node5800-c1, 17:51 to 17:55). The update profiled on the
same node with the memory-efficient SDPA kernel and with PyTorch's own choice (cuDNN); r118 (another node, the same
settings) is the earlier reference. Numbers are per update, from `profile.json`:

| arm | wall s, unprofiled (median) | wall s, profiled (median) | GPU busy fraction | GPU busy ms | kernel launches | syncs | device-to-host copies | cuDNN attention CPU ms |
|---|---|---|---|---|---|---|---|---|
| r118 final_native_ctl | 0.4173 | 0.3219 | 0.402 | 127.7 | 11,689 | 64 | 266 | 81.0 |
| r119 sdpa_default | 0.4146 | 0.3668 | 0.366 | 129.8 | 11,453 | 63 | 266 | 111.0 |
| r119 sdpa_efficient | 0.2557 | 0.3583 | 0.415 | 146.5 | 13,339 | 73 | 272 | 0.0 |

- Unprofiled, the efficient kernel is 0.159 s faster (38 %); profiled, 0.0085 s.
- The rule needed 0.03 s or more in both medians, and more than `sdpa_default` moved against r118. Between those
  two, the profiled median moved by +0.045 s and the unprofiled by -0.003 s, so the profiled numbers differ between
  runs by more than the efficient kernel's gain there.
- **Not met.** As written for that case, capture or compile goes next.
- A reading, not checked: `sdpa_default`'s profiled updates (which come after the 20 timed ones) were faster than its
  timed ones (0.367 s against 0.415 s), the reverse of the usual profiler cost. That fits cuDNN building plans
  for new micro-batch shapes in the early updates, a cost that would fade over a long run. If so, most of the
  unprofiled gain is a start-up cost, not a steady one.
- The efficient kernel adds 1,886 launches and 17 ms of GPU busy time per update (expected: 5 to 15 ms).
- Capture or compile builds on the efficient kernel. It was not slower in either median, and capture needs fixed
  shapes with no per-shape planning. The arm is timed on a real run's update times, not on profiles alone. It
  comes after U25's code (done) and data-parallel pretraining (due 11-24).

**r116 data** (same job, 17:55 to 18:01). The Countdown keyword rule (`scripts/keyword_filter.py`) dropped 2,230 of
314,040 FineMath-4plus documents (0.71 % of documents, 1.80 % of characters). Matches by phrase (a document can match more
than one): "countdown" 694, "target number" 688, "numbers game" 677, "make 24" 210, "each number once" 182, "24 game" 115, "krypto" 6.
- `finemath_kw.bin`: 0.521B tokens, md5 46a89589818162629cc0035eb47b0fd5.
- `mix_0.2_finemath_kw2.bin`: 2.321B + 2 x 0.521B = 3.362B tokens, FineMath share 31.0 %, md5
  ea5a93c9b6b5cda51221b5c38a07de1f.
- Reasoning cues in 2,000 windows of 1,024 tokens from each file (the job's own counts):

| cue (per million characters) | FineWeb-Edu | dm_math | FineMath-4plus | FineMath, filtered |
|---|---|---|---|---|
| therefore / thus | 80.6 | 0.0 | 220.4 | 208.9 |
| we get / we have | 43.1 | 0.0 | 167.9 | 170.2 |
| let's / let us | 22.3 | 0.0 | 78.5 | 92.8 |
| check / verify | 30.9 | 0.0 | 59.5 | 76.6 |
| the answer is | 1.5 | 0.0 | 16.7 | 8.8 |
| step by step | 1.8 | 0.0 | 14.4 | 18.4 |
| wait / hmm / okay | 9.2 | 0.0 | 20.1 | 18.9 |

FineMath has 1.9 to 11.1 times FineWeb-Edu's rate of each cue; dm_math has none of them (it is question and
answer only). The filter moves each rate by at most 29 %, except "the answer is", which falls from 16.7 to 8.8.

**r116 pretraining.** Worker 30000622 (normal B200, node5701-c1, from 18:08, queue `pre_r116`) runs
`d_fmkw2_muon1338m_pre` seeds 0 and 1 (about 42 min each at 530k tokens a second). Then `d116rl_native_fmkw2_b200`
seeds 0 and 1 go into a B200 queue with their workers.

**U25 code** (plan: LOG 11-21 17:59).
- f3a6340: `pretrain_scratch.py --phase2 path:weight,...` with `--phase2-start` (off by default; without it the draws
  are unchanged). From that run progress on, each row of a micro-batch comes from one file picked by weight.
- 6cab95e: `scripts/research/probe_reasoning.py`, the probe as planned (sets A, B and C; `probe.json`, `samples.md`
  for reading by eye, `completions.jsonl`).
- 45e92eb: `scripts/research/u25_rung.sh` (one rung: copies the three token files to node-local disk, pretrains,
  then probes) and the r121 configs `u25_3x_twophase` and `u25_1x_d007base`.

**Two changes to the written plan**, both made before any U25 run:
- **The second phase uses the keyword-filtered FineMath** (`finemath_kw.bin`), not the unfiltered shards. INBOX said
  a filter is not needed for U25, but the probe is Countdown, the filter found 2,230 Countdown-like documents in
  FineMath, and the hard data rule bans Countdown data. It costs 1.5 % of FineMath's tokens. The weights by row are
  `mix_0.2.bin` 0.3125, `dm_math.bin` 0.1875 and `finemath_kw.bin` 0.5. That is FineWeb-Edu 0.25, dm_math 0.25 and
  FineMath 0.5, as planned.
- **The fused loss at micro-batch 64.** The settled track D path is the fused loss at 128 (r91, r107: 52.9 GB at
  110M). At 1024 x 24 the activations per token are about 2.7 times larger, and 128 might not fit in a B200. In
  r91, 64 was about 2 % slower than 128.

**U25 jobs.**
1. When 30000622 ends (about 19:50), one normal B200 job (1 h limit):
   - D 007's base probed (rung 1x, the reference);
   - then a 6-minute 3x smoke run of `u25_rung.sh` into `u25_3x_smoke`. The phase switch happens at 3 minutes. It
     checks memory at 1024 x 24, the copy, the switch and the probe on a 332M model before the long run.
2. The 3x rung: one normal B200 job, 6 h limit, 330 minutes of pretraining, then the probe. It starts when
   29966174 ends (about 22:00). If the smoke run fails, it waits for a fix. Read on 11-22 at about 04:00.

The smoke run's weights and outputs are not results. They are deleted after the read.

**Jobs.**
- **Running:**
  - 30000622 (r116 pretraining, normal B200 node5701-c1, from 18:08);
  - 29966174 (r117 p128 seed 0, normal B200 node5501-c1, to about 22:00);
  - 29837214 (r102 seed 2, preemptable H200 node4601).
- **Pending:**
  - 29837213 (r102 seed 1, H200);
  - 29995938 (r120 seed 0, H200);
  - 29999306 (B200 preemptable, queue `screens_rl_b200`: r105 lr 24e-6 seed 2, then r117 seed 1);
  - the four held workers.
- **GPUs:** 3 in use. If every pending GPU job started, that would be 6.

## 2026-11-21 19:57 EST: U25's reference probe: D 007's base writes no working on Countdown; the 3x smoke run passed and the 3x rung is submitted now; r116's pretraining is done and its RL seeds are queued

**U25, D 007's base probed (rung 1x, the reference).** Normal B200 job 30008197 (node5601-c1). `probe_reasoning.py`
with its defaults: sets A and B have 200 dev puzzles x 4 samples; set C has 100 GSM8K train questions; temperature
1.0; 256 new tokens. Output: `calibration/research_r121/u25_1x_d007base/seeds/0/probe/` (`probe.json`, `samples.md`).

| set | samples | steps a sample | step accuracy | grounded chains | solved | loops | median tokens |
|---|---|---|---|---|---|---|---|
| A, track D's r1 prompt | 800 | 0.00 | none | 0.00 | 0.00 | 0.91 | 184.50 |
| B, the trial-and-error few-shot prompt | 800 | 0.00 | none | 0.00 | 0.00 | 0.67 | 143.00 |
| C, GSM8K with worked examples | 100 | 0.44 | 0.00 | 0.00 | 0.03 | 0.56 | 255.00 |

Read by eye (the 20 shown samples of set B): none has working. The samples are repeated `<answer>` tags and web-page
text (comment boxes, share links, prose). On set A, 0.91 of samples loop. On set C, 0.44 arithmetic steps a sample,
none right, and 3 of 100 final answers match (read as chance: short numbers). So D 007's base gives a floor of zero
for the "reasoning appears" test. The test in LOG 11-21 17:59 thus reduces to its absolute parts on set B: at least
10 of 20 samples real working, a grounded-chain rate of 0.5 or more and step accuracy of 0.7 or more.

**The 3x smoke run passed.** The same job ran `u25_rung.sh` at 1024 x 24 (331.7M parameters) for 6 minutes.
- The copy to node-local disk took 7 s.
- Micro-batch 64 with the fused loss fit in the B200. Peak memory was not logged.
- The phase switch came at half the time: step 50 (2.2 min) was in phase 1, step 100 (3.8 min) in phase 2.
- Throughput rose to 244k tokens/s at step 150. At that rate, 330 minutes is about 4.82B tokens, 14.5 per
  parameter. That is within the 4.0B to 5.5B range planned (LOG 11-21 17:59).
- The probe ran on the 332M model in 1 min 22 s.
- The smoke run's weights and outputs are deleted, as planned. Its output is in the job's Slurm log.

**The 3x rung is submitted now** (normal B200 job 30010825, 6 h limit), not at 22:00: the smoke job ended at 19:42 and
left the normal slot free. It was pending at 19:58. Read it about 5.7 h after it starts (about 01:45 on 11-22 if it
starts soon).

**r116's pretraining is done** (normal B200 job 30000622, node5701-c1).
- Seed 0: 38.0 min for 1,337,982,976 tokens, final val loss 2.9721.
- Seed 1: 37.4 min, final val loss 2.9810.
- These val losses are on the new mix's held-out rows, so they are not compared with D 007's.
- The two RL items (`d116rl_native_fmkw2_b200`, seeds 0 and 1) are queued on `screens_rl_b200`, after r105 lr 24e-6
  seed 2 and r117 p128 seed 1. They get workers as slots free.

**r102 seed 1 lost its worker to a node.**
- Preemptable H200 29837213, requeued, restarted on node4301 at 18:20.
- There, the venv was unreadable within 60 s, so the worker exited at once, as the guard intends.
- The item stayed in hand in `queue/running/`. I moved it back to `screens_r92` under the lock.
- Preemptable H200 30006765 (node4301 now excluded) took it on node4602 at 19:25. This is an infrastructure
  interruption of a screen seed, not a record seed.

**Jobs.**
- **Running:**
  - 29966174 (r117 p128 seed 0, normal B200, to about 22:00);
  - 29837214 (r102 seed 2), 30006765 (r102 seed 1) and 29995938 (r120 seed 0, since 18:38), preemptable H200.
- **Pending:**
  - 30010825 (U25 3x, normal B200);
  - 29999306 (preemptable B200, `screens_rl_b200`);
  - the four held workers.
- **GPUs:** 6 if both pending jobs start. r120 seeds 1 and 2 get preemptable H200 workers as slots free.

## 2026-11-21 20:52 EST: r102 seed 2 ends at 0.300 on dev, above D's line; r102's rule for seed 1 written first; r120 seed 1 gets a worker

**r102 seed 2's final** (preemptable H200 29837214, node4601, 14:40 to 20:28). Dev 0.300 after 42,588 steps
(18,028 s of GRPO at 2.360 steps a second, after 2,774 s of pretraining; 20,795 s of the 21,600 s budget). The last
five evals average 0.3070 and the best is 0.3217. That is +0.049 over D 007's dev mean (0.2510) and +0.019 over the
line (0.281). Seed 0 ended at 0.270 after 38,914 steps at 2.156 steps a second (node4802). Seed 2's node gave it
9 % more steps, so part of the gap between the two seeds is the node, not the seed. The worker log says "FAILED"
for both seeds: the runner's last `mv` looks for a `final/` folder, which a `no_save` screen does not write. The
results are complete.

**r102's rule for seed 1, written before its final** (preemptable H200 30006765, node4602, due about 01:15 on 11-22):
- The three-seed form cannot be met, since seed 0 is under 0.281. r102 is not a record candidate on three seeds.
- The two-seed mean is 0.285. A three-seed mean of 0.281 or more needs seed 1 at 0.273 or more.
- If the three-seed mean is 0.281 or more, r102's seeds 3 and 4 join `gate_r120_h200` after r120's seeds 1 and 2,
  for the five-seed form on r102 itself (mean at least 0.281, four of five above 0.251, at most one collapsed).
  r120, the same config with lr 1.2e-5, is expected about +0.015 higher (LOG 11-21 17:13). If both reach the
  five-seed form, the record attempt uses r120's config; if only r102 does, r102's.
- If it is under 0.281, r102's seeds stay as r120's pairs, and r120's rule alone decides the next record attempt.

**Jobs.** I submitted preemptable H200 30017487 for `gate_r120_h200` (r120 seed 1) into 29837214's freed slot.
- **Running:** 29966174 (r117 p128 seed 0, normal B200, to about 22:00); 29995938 (r120 seed 0, preemptable H200,
  node5000, since 18:38); 30006765 (r102 seed 1, preemptable H200, node4602, since 19:25).
- **Pending:** 30010825 (U25 3x, normal B200; Slurm's estimate is 20:29, passed); 29999306 (preemptable B200,
  `screens_rl_b200`); 30017487 (preemptable H200, `gate_r120_h200`); the four held workers.
- **GPUs:** 6 if all pending jobs start.

## 2026-11-21 21:58 EST: INBOX 21:25 puts speed work first: r122 times 600 updates to settle r119, then capture or compile of the update (r123), ahead of U25's data-parallel pretraining; r117 seed 0 is ahead at matched train time (+0.007) but 0.008 behind at the final, so r117 is left where it is and its seed 1 is not run

**r117 p128 seed 0** (`calibration/research_r117/d007rl_native_p128_b200`, B200, RL only; node5501-c1, 17:02 to 21:53;
the control `calibration/research_r101/d007rl_native_b200` seed 0 ran on node5600-c1, the slow node):

| | arm | control |
|---|---|---|
| final dev | 0.2950 | 0.3033 |
| steps | 21,695 | 34,893 |
| steps/s | 1.247 (0.62x) | 2.005 |
| mean of the last five evals | 0.3000 | 0.2920 |

| arm minus control | seed 0 |
|---|---|
| at the final | -0.0083 |
| at matched train time (evals; ahead / behind) | +0.0067 (86; 66 / 20) |
| matched train time, by hour 1 to 5 | -0.019, +0.007, +0.015, +0.014, +0.009 |
| at matched steps (evals; ahead / behind) | +0.0293 (43; 42 / 1) |

The step's parts, early (steps 100 to 2,100) and late (the last 2,000 steps), in seconds:

| | `t_sample` | `t_update` | wall per step | informative groups |
|---|---|---|---|---|
| arm, early | 0.368 | 0.602 | 1.061 | 106.6 |
| arm, late | 0.240 | 0.340 | 0.649 | 56.3 |
| control, early | 0.175 | 0.377 | 0.602 | 56.6 |
| control, late | 0.136 | 0.170 | 0.338 | 29.2 |

No error line in stdout. Expected (LOG 11-21 12:40): 0.63x to 0.71x the control's step rate, and a final at +0.005
(range -0.03 to +0.035). The step rate is at the low end. Late in the run the update took 2.0x the control's time
for 1.9x the informative groups, so at this size it scales with the rows. In r115 (half the rows) it fell by
only 15 to 20 %.
The final is inside the expected range, below its centre. Per step it gains about what r115 lost (+0.029 here
against -0.040 there), and it is ahead at matched train time from the second hour to the fifth, by 0.007 to 0.015.
The control's last evals rise faster, and it ends 0.008 ahead. One final eval has a noise of about 0.015 per pair,
so the final and the last-five mean (+0.008) point opposite ways within that noise.

**Under r117's rule** (written first): not +0.02 at the final, and not behind at matched train time, so seed 0
alone puts r117 **between: left there**. Seed 1 cannot change that: both other outcomes need both pairs. So I took
r117 seed 1 out of `screens_rl_b200`, and its 4.8 B200 hours go to the speed work. The rows-per-step question is
asked again once the update is captured, since capture changes the fixed part of a step that the comparison turns on.

**INBOX 21:25 (maintainer).** Speed work goes first, own kernels are in scope, capture of the update is endorsed,
and r119's 0.16 s needs a timing on a real run's later updates. The order of code work is now:
1. r122 (below), a timing job, tonight.
2. Capture or compile of the update (r123), from 11-22, on the attention kernel r122 leaves.
3. Own kernels (Triton or CUDA) where the r122 and r123 profiles show a library kernel is the cost, with
   same-maths checks.
4. U25's data-parallel pretraining. It is needed only if the 3x rung says no, since the 6x rung then goes to
   the 11-25 box. The 3x read is due about 5.7 h after 30010825 starts. If it says no, data-parallel work starts
   by 11-23 09:00 EST whatever the speed work's state, so it is tested on Engaging by 11-24. If it says yes,
   the next rungs (1.5x, 1x, 0.5x) fit single B200 jobs and data-parallel waits.
5. U26 is new in the ideas file; it is picked up after the speed items (verdict in the private status file).

**Reference point for r122.** The native control's train logs give a late update of 0.170 s (seed 0, node5600-c1)
and 0.202 s (seed 1), at about 29 informative groups. r118's and r119's profiles, at the final weights, measured
0.415 to 0.417 s for the first 20 updates with PyTorch's attention choice, and 0.256 s with the memory-efficient
kernel. The real run's late update is far below either, so most of the 0.16 s is likely a start-up cost: cuDNN
planning for new micro-batch shapes, the allocator and the first kernel loads.

**r122, written first.** `scripts/research/profile_update.py` now writes every timed update's wall time and the
median by window of updates (1-49, 50-199, 200-399, 400-599). One normal B200 job profiles the update at r118's
settings (r106 `final_native_ctl`'s config, D 007 seed 0's final weights) with 600 timed updates and then 12
profiled ones, twice on one node: PyTorch's choice (cuDNN), then the memory-efficient kernel. Results go to
`calibration/research_r122/sdpa_{default,efficient}_long/seeds/0/profile.json`; nothing is trained, saved or scored.

Expected:
- PyTorch's choice: the median falls from about 0.41 s (updates 1-49) to 0.17 to 0.22 s from update 200 on, as
  cuDNN's plans for the shapes seen are reused.
- The efficient kernel: about 0.25 s early and 0.16 to 0.22 s from update 200 on, within 0.03 s of PyTorch's choice.
- A busy fraction of 0.5 to 0.75 in the late profiled updates, for both.

What it decides:
- **The efficient kernel 0.03 s or more faster from update 200 on:** an `sdpa_backend` option goes into GRPOConfig
  (off by default), and one native-control seed runs with it as a timing check, as r119's rule said.
- **Under 0.03 s:** r119's gain was a start-up cost and is closed. Capture or compile still uses the efficient
  kernel (its shapes need no per-shape plans) and is timed against PyTorch's choice from update 200 on.
- **Late busy fraction under 0.6:** r123 compiles or captures the update's forward and backward, expecting to
  remove about half the idle time. **Over 0.8:** the update's GPU work is cut instead, starting with the top ops
  in the late profile (own kernels where a library kernel is the cost).

Cost: one normal B200 job, about 30 min. It takes the normal slot freed by 29966174; the 3x rung (30010825) has
the other. A normal B200 worker for `screens_rl_b200` follows when r122 ends.

**r123, outline (its rule is written when the code is in).** A new GRPOConfig option, off by default, sends the
update's forward, loss and backward through `torch.compile` (first without CUDA graphs, then with them if the
shapes allow). The micro-batches' padded widths are rounded up to a multiple of 64 so the compiled graphs see a
small set of shapes. The same-maths check runs on fixed batches: per-token log-probs and the loss within bf16
tolerance of the eager path, and the gradient's relative difference under 1e-2. Then one native-control seed runs
on the B200 as the timing check. It is kept on time without a paired win (INBOX), unless the precision changes.

## 2026-11-21 22:20 EST: r123 code in (compile_update), its rule written first

**Code.** A new GRPOConfig option, `compile_update` (off by default; no existing config changes).
- `_compile_layers` compiles each decoder layer in place with `nn.Module.compile(dynamic=True)` (inductor). The
  update's forward and backward then run through compiled graphs, while the head, the loss and the cache
  bookkeeping stay eager. Parameter names are unchanged, so saving and the samplers' weight copies are too.
- Two settings let all the layers share one set of graphs: the layer index stays an unspecialised int, and
  `DynamicCache.update` runs eagerly (it indexes the cache's per-layer lists by that index). Without both, each
  layer gets its own graphs.
- On CPU (aot_eager, a tiny Llama) the graph count is 10 for both 3 and 6 layers, with five graph breaks, all
  around the cache update. Log-probs are identical and gradients agree within 2e-6.
- A new test checks the three log-prob paths (plain, prefix cache, shared prompt) against eager for 2 and 3
  layers, and that the graph count does not change with depth. `ruff check .` is clean; pytest: 207 passed.
- The native sampler copies the weights into its own buffers and never calls the layers, so sampling is unchanged.
  `rollout_backend 'hf'` and `grad_checkpoint` raise an error with this option.
- Compilation happens in the first updates, inside the clock. The run prints dynamo's graph counts at the end.

**Why the layers and not the whole update.** r119's profile gave 13,339 launches per update. My count is about 50
forward kernels per layer per call, of which 7 are matmuls or attention; the rest are pointwise (norm pieces,
autocast casts, rotary, residual adds, SiLU-and-multiply), and inductor fuses those within a layer. Capturing the
whole update in CUDA graphs needs fixed shapes, but the micro-batches' widths vary, so it is the next step only if
this one still leaves the GPU idle. The outline's rounding of widths to a multiple of 64 is not needed with
dynamic shapes.

**r123 rule, written first.** One normal B200 job (r122's settings: r106 `final_native_ctl`'s config, D 007 seed
0's final weights). It takes the normal slot r122 frees.
1. `scripts/research/check_compile_update.py` builds 16 training prompts (the ladder training stream, which
   excludes dev and test) with 8 sampled completions each. It runs forward and backward on the shared-prompt path
   three ways: fp32 eager, bf16 autocast eager, and bf16 autocast compiled. **Passes** if compiled is no further
   from fp32 than eager is (mean |delta log-prob| at most 1.1x eager's, gradient cosine to fp32 at least eager's
   minus 1e-4) and the compiled-vs-eager gradient cosine is above 0.999.
2. `profile_update.py --set compile_update=true --warm 600 --steps 12`, once with PyTorch's attention choice and
   once with the efficient kernel. Each is read against r122's run on the same kernel.

Output goes to `calibration/research_r123/{check,compiled_default_long,compiled_efficient_long}/seeds/0/`. Nothing
is trained, saved or scored.

Expected:
- Launches per update down by 40 to 60 %, and the busy fraction up from about 0.4 to 0.6 to 0.75.
- The late median (updates 200-599) at 0.11 to 0.13 s, against the 0.17 to 0.22 s r122 is expected to show.
- Compilation makes the first updates slower, by 2 to 5 min in total (1 to 2 % of 17,400 s).
- At the native control's 2.0 steps a second, the update is about a third of a step, so 0.05 s less per update
  means about 10 % more steps.

Noise: a median over 400 updates on one node should hold to about 0.005 s. r122's two arms show the spread
between runs, so a 0.03 s difference is clear.

What it decides:
- **The check fails:** the option stays off, and the per-layer comparison finds the op before anything else runs.
- **The check passes and the late median is 0.03 s or more under r122's on the same kernel:** one native-control
  timing seed runs with compile_update on a B200, paired with `d007rl_native_b200` seed 0 (0.3033, 2.005 steps a
  second). It is the same maths, so it is kept on time without a paired win (INBOX 11-19). Its score must stay
  within one seed's noise (about 0.015).
- **Under 0.03 s:** compiling the layers is closed. Capturing the whole micro-batch in CUDA graphs, with the widths
  rounded to fixed sizes, becomes r124.
- **The late busy fraction stays under 0.6 after compiling:** the remaining gap is the head and loss (eager) and
  the step's own Python, and r124 targets those.

## 2026-11-21 22:54 EST: r122 stopped on max_steps; rerun with r123 after it in one job

r122 (30024430) started on node5701-c1 at 22:41 and both arms stopped in 30 s with `max_steps 300 must exceed
warm + steps (612)`. r106's `final_native_ctl` config, which the profiles run, has max_steps 300, and
`profile_update.py` checks it before it starts. Nothing ran on the GPU.

The fix sets `--set max_steps=1000000` on every profile (time_budget_s stays 17,400 s, far more than 612 updates
need). The config and the code are unchanged. r122 and r123 now run as one normal B200 job, 30028921 (submitted at
22:54, 2.5 h limit): r122's two arms, then r123's check and its two compiled arms. That puts all four timings on the
same node, so the r123 rule compares like with like. The rules (LOG 11-21 21:58 and 22:20) are unchanged. After the
job, the normal slot goes to a `screens_rl_b200` worker.

## 2026-11-22 00:00 EST: r122 and r123 read: r119 is closed, compile_update stays off, and r124 goes after the update's synchronisations

30028921 ran on node5700-c1 (B200) from 23:19 to 23:43 EST. It ran r122's two arms, then r123's check, then r123's
two compiled arms, and all five steps exited 0. Each profile timed 600 updates of r106's `final_native_ctl` (D 007
seed 0's final weights, max_steps raised) and then profiled 12. The numbers below come from the JSONs by script.
"Late" means updates 200-599. Busy, launches, synchronisations and device-to-host copies are per profiled update.

| arm | late median | late mean | late p90 | 600 updates | first 50 | busy frac | busy s | launches | syncs | D2H |
|---|---|---|---|---|---|---|---|---|---|---|
| r122 PyTorch's choice (cuDNN) | 0.1756 | 0.2170 | 0.3185 | 142.6 s | 19.4 s | 0.507 | 0.124 | 11,486 | 63 | 266 |
| r122 efficient | 0.1677 | 0.1918 | 0.2368 | 119.2 s | 12.1 s | 0.519 | 0.122 | 10,977 | 61 | 264 |
| r123 compiled, PyTorch's choice | 0.1525 | 0.1855 | 0.2248 | 141.9 s | 33.6 s | 0.429 | 0.093 | 6,020 | 59 | 394 |
| r123 compiled, efficient | 0.1576 | 0.1954 | 0.2813 | 142.3 s | 29.9 s | 0.353 | 0.095 | 6,582 | 63 | 410 |

**Noise is larger than I expected.** Each arm is its own RL run from the same weights and seed, so the completions
(and the tokens in each update) drift apart, and tokens per update are not logged. The window medians show the
drift: PyTorch's-choice arm has 0.174 s for updates 200-399 but 0.227 s for 400-599, while the others hold or fall.
So arm differences under about 0.02 s are within run-to-run noise, not the 0.005 s I wrote for r123.

**r122 (rule LOG 11-21 21:58).** The efficient kernel's late median is 0.008 s under PyTorch's choice, which is
under the 0.03 s bar. **r119 is closed**: its 0.16 s was a start-up cost (the first 50 updates take 12.1 s against
19.4 s). The late means differ by 0.025 s because of cuDNN's tail (p90 0.32 against 0.24 s, single updates up to
2.4 s), and that is also under the bar. No `sdpa_backend` option is added. As the rule says, capture or compile
uses the efficient kernel. The late busy fraction is 0.51 to 0.52, under 0.6, which pointed r123 at compiling.

**r123's check (rule LOG 11-21 22:20): rule_met false.** It used 128 rows of width 448, with 26,499 completion
tokens.

| comparison | mean abs dlp | grad cos | grad rel err |
|---|---|---|---|
| eager vs fp32 | 0.0120 | 0.910 | 0.43 |
| compiled vs fp32 | 0.0116 | 0.943 | 0.36 |
| compiled vs eager | 0.0097 | 0.9944 | 0.11 |

- The first two clauses hold: compiled is closer to fp32 than eager is on both measures.
- The third clause fails: the compiled-vs-eager cosine is 0.9944, and the bar is 0.999.
- That bar was set without knowing the bf16 floor. Eager bf16 itself sits at cosine 0.910 from fp32, so 0.999
  between two bf16 paths asks for more than bf16 gives.
- The rule stands as written, so compile_update fails its check. Any later same-maths check sets its tolerance
  from the measured bf16-vs-fp32 floor, written before the run.
- Other results:
  - The first compiled call took 67 s.
  - Median call: eager 0.152 s, compiled 0.117 s.
  - 11 graphs, with no new graph on a narrower, shorter second batch.
  - 5 graph breaks.

**r123's timing.** Correction (00:02): the compiled "efficient" arm did not use the efficient kernel. Its profile
shows cuDNN attention (72 forward and 68 backward calls per update), as the other compiled arm and r122's
PyTorch's-choice arm do. The kernel choice made around the update did not reach the compiled layers, so both
compiled arms ran the same configuration. Their gap (0.005 s late median, 0.4 s over 600 updates) measures
run-to-run noise. Against r122's cuDNN arm, the compiled late medians are 0.023 and 0.018 s lower. Both are under
the 0.03 s bar. Over all 600
updates, compiling gains nothing: 141.9 s against 142.6 s, and 142.3 s against the efficient kernel's 119.2 s, because compilation adds
10 to 20 s in the first 50 updates. **Compiling the layers is closed. compile_update stays off by default and no
seed runs with it.** The code stays as an option, and its test and check script stay too.

**Why the gain was small, and what r124 targets.**
- Compiling cut the GPU's busy time per update from 0.122 to 0.094 s, and launches by 40 to 48 %.
- The wall time fell only 0.01 to 0.02 s, so the busy fraction went down (0.35 to 0.43), not up.
- The update is bound on the CPU side, not by kernels:
  - about 60 synchronisations per update (cudaStreamSynchronize 16 to 20 ms of CPU time);
  - 264 to 410 device-to-host copies;
  - about 30 ms of cudaLaunchKernel in eager mode.
- Each synchronisation drains the GPU's queue, so the CPU can never run ahead.
- The sources are in `_update` and the loss. Each micro-batch does:
  - boolean-mask indexing (`token_lp[completion_mask]`, the entropy, the repeated advantages, the skip mask), each
    with a hidden nonzero;
  - `.item()` for the micro-batch width;
  - `.nonzero()` for the active rows.
- Also, `unique_dim` takes 4.6 ms of GPU time per update.
- CUDA-graph capture needs these regions free of synchronisations too.

**r124, outline (its rule is written when the code is in):**
1. A GRPOConfig option, off by default, computes the loss sums as masked multiply-and-sum over the padded tensors
   instead of boolean-mask indexing.
2. The same option takes the micro-batch widths and active rows from one host copy per update, made before the
   micro-batch loop.
3. Same maths up to the order of summation. It gets a CPU test, and a GPU check whose tolerance comes from the
   bf16 floor.
4. It is timed the way r122 was. Whole-micro-batch CUDA-graph capture, with fixed widths and the efficient kernel,
   follows only if the synchronisations come down to a few per update.
5. Own Triton kernels (INBOX 21:25) come in where a profile then shows a library kernel as the cost.
   unique_dim is the first candidate.

U25's 3x rung (30010825) is still running, and its read is due about 03:40. The normal slot that 30028921 freed went
to a `screens_rl_b200` worker, 30033104 (submitted 23:57).

## 2026-11-22 00:25 EST: r124 code in (update_host_index), its rule written first

**What changed from the 00:00 outline.** A masked multiply-and-sum would change the order of summation. Instead the
option keeps the update's indexing ops and feeds them indices built on the host once, at the start of the update.
For each micro-batch these are:
- its rows and width;
- the (row, column) of each completion token, row-major, which is what the boolean mask's hidden nonzero gives;
- the shared-prompt groups, from `torch.unique` on the CPU, which numbers them the same way;
- the constrained-logit mask's index tensors.

The replayed (SIL) rows get the same. Everything goes to the GPU in one pinned copy, and the replayed rows' nll is
read after `optimizer.step()`. The ops and their inputs are unchanged, so the maths is too. The CPU test
(`test_update_host_index_gives_the_same_update`) asserts bit-identical gradients, logged sums and nll in three
cases, each with constrained logits and replayed rows:
1. fixed micro-batches;
2. token packing with skipped rows and shared prompts;
3. the same with the entropy bonus, TIS, the skip mask and KL.

Left as they are:
- the copies to the host at the start of the update;
- the TIS mismatch sums;
- the LM rows' loss;
- whatever the model's attention-mask code does.

The TIS sums and the LM loss are off in r106's config. The option is off by default, and the pending jobs that
import the new `grpo.py` run the same path as before. Full suite: 210 passed.

**Expected effect and noise.**
- Synchronisations per profiled update should fall from 61 to 63 (r122) to about 10. The GPU's queue is then no
  longer drained once per micro-batch.
- r122's PyTorch's-choice arm had a late median of 0.176 s. The GPU was busy for 0.124 s of it (busy fraction
  0.51), with 11,486 launches.
- Once the CPU can run ahead, the wall time falls toward the longer of the GPU's busy time and the CPU's launch
  time. Expected gain: 0.01 to 0.05 s per update.
- Noise: separate arms differ by up to 0.02 s at late medians, because each arm is its own RL run (00:00 entry).
  So the timing is paired on the same inputs instead.

**Rule.**
1. **Same maths.** `scripts/research/check_update_host_index.py` runs r106's `final_native_ctl` config and checks
   updates 2, 20 and 100. At each one it runs the update three ways on the same inputs: default, default again,
   then on. It passes if, at every checked update, on vs default is within the default's own repeat spread:
   - gradient cosine ≥ min(0.9999, the default-vs-default cosine − 1e-4);
   - sums and nll within max(1e-5, twice the default-vs-default difference), relative.

   Bit-identical is expected, up to the order of the GPU's atomic adds. This sets the tolerance from the measured
   floor, as r123 taught. If the check fails, the option is not used and the timing is not read.
2. **Timing, paired.** In the same run, at every 10th update from 110 to 590 (49 pairs), the update runs once
   each way on the same inputs, with a step that does nothing. The first of each pair alternates.
   - **Kept** if the median of default minus host_index is at least 0.01 s, and host_index is faster in at least
     34 of 49 pairs (one-sided sign test, p < 0.01).
   - Kept means later configs (screens and records) turn it on, as a same-maths speed change.
3. **Profiles.** Two separate arms follow in the same job: default, then host_index. Each runs 600 updates plus
   12 profiled ones, with PyTorch's attention choice.
   - They report synchronisations, launches and busy time per update. host_index should show fewer
     synchronisations.
   - Their late medians are read but not ruled on, since separate arms differ by up to 0.02 s.
4. **If it is not kept,** the option stays off. Capture (r125) needs a loop without synchronisations, so it builds
   on this code either way.
5. **Next, in this order:** capture, then own Triton kernels. unique_dim has already moved to the CPU.

**When it runs.** All six GPU slots are in use or pending. r124's job (one B200) takes the first slot it can:
- the one r102 seed 1 frees about 01:15, unless r102's rule needs that slot;
- otherwise the normal slot U25's 3x rung frees about 03:40.

## 2026-11-22 01:14 EST: r102 seed 1 ends at 0.310 on dev, so r102's three-seed mean is 0.293 and its seeds 3 and 4 are queued; r124 submitted

**r102 seed 1's final** (preemptable H200 30006765, node4602, 19:25 to 01:12):
- Dev 0.310 after 43,073 steps: 18,045 s of GRPO at 2.387 steps a second, after 2,771 s of pretraining.
- 20,788 s of the 21,600 s budget.
- The last five evals average 0.301, and the best is 0.310.
- As for seeds 0 and 2, the worker log's "FAILED" comes from the runner's last `mv`. It looks for a `final/` folder,
  which a `no_save` screen does not write. The results are complete.

| seed | dev | steps | steps a second | last five evals | best |
|---|---|---|---|---|---|
| 0 | 0.270 | 38,914 | 2.156 | 0.272 | 0.282 |
| 1 | 0.310 | 43,073 | 2.387 | 0.301 | 0.310 |
| 2 | 0.300 | 42,588 | 2.360 | 0.307 | 0.322 |
| mean | 0.293 | | | | |

The seed with the slowest node (seed 0, node4802) is again the lowest.

**r102's rule (LOG 11-21 20:52) is met.** The three-seed mean is 0.293, so it is at least 0.281.
- r102's seeds 3 and 4 joined `gate_r120_h200` at 01:14, after r120's seeds 1 and 2.
- They are for the five-seed form on r102: mean at least 0.281, four of five above D 007's 0.251, and at most one
  collapsed. All three seeds so far are above 0.251.
- If r120 also reaches its form, the record attempt uses r120's config, as that rule says.

**Workers.** The gate queue now holds four items: r120's seeds 1 and 2, and r102's seeds 3 and 4.
- One worker is pending for it (30017487), besides the requeued r120 seed 0 (29995938).
- The six-GPU cap holds back the other items' workers. They are submitted as slots free.

**r124 submitted** into the freed slot as preemptable B200 30039308.
- Both normal slots are in use: U25's 3x rung runs, and 30033104 waits.
- Its rule is the 00:25 entry. If it is preempted, it reruns from the start.

## 2026-11-22 03:52 EST: U25's 3x rung says no (no working on set B by eye; chain rate 0.05, step accuracy 0.44), so the search goes up to 6x on the 11-25 box; r124 moved to a normal B200 (30051395), and a second r120 gate worker (30051396)

**The 3x pretraining** (`calibration/research_r121/u25_3x_twophase`, 1024 x 24, 331.7M parameters; normal B200 job
30010825 on node5501-c1) ran from 21:59 to 03:30 and stopped on its 330-minute clock. The probe ended at 03:32, rc 0.
- Step 10,209 and 5,352,456,192 tokens: 16.1 a parameter, at 270,323 tokens a second. That is more than the smoke
  run's projection (4.8B) and less than the 6.6B (20 a parameter) of the full form.
- 6NT a second is 5.38e14, an MFU of 0.24 against the B200's dense bf16 peak of 2.25e15.
- The lr ended at 4.04e-4, the schedule's floor (0.1 of the 4e-3 peak), as designed.
- Final val loss 2.644. The val slice is the first 2M tokens of `mix_0.2.bin`, the same as D 007's base, whose five
  seeds ended at 2.864 to 2.903.

**The probe**, the same settings as the 1x reference (LOG 11-21 19:57). Output: `probe/probe.json` and
`probe/samples.md` in the run's seed directory (committed; `completions.jsonl` is not, as for 1x).

| set | rung | steps a sample | step accuracy | grounded chains | solved | well-formed | loops | median tokens |
|---|---|---|---|---|---|---|---|---|
| A, track D's r1 prompt | 1x | 0.00 | none | 0.00 | 0.00 | 0.00 | 0.91 | 184.5 |
| A | 3x | 0.76 | 0.49 | 0.02 | 0.00 | 0.01 | 0.11 | 256 |
| B, the trial-and-error few-shot prompt | 1x | 0.00 | none | 0.00 | 0.00 | 0.00 | 0.67 | 143 |
| B | 3x | 4.84 | 0.44 | 0.05 | 0.00 | 0.17 | 0.07 | 171 |
| C, GSM8K with worked examples | 1x | 0.44 | 0.00 | 0.00 | 0.03 | 0.58 | 0.56 | 255 |
| C | 3x | 1.49 | 0.75 | 0.10 | 0.01 | 0.68 | 0.10 | 55.5 |

**Read by eye** (the 20 samples of each set beside 1x's):
- **Set B:** none of the 20 is working. Every sample copies the examples' sentence form ("The sum is ...", "Try
  ...", "That works") but fills it with numbers that are not the puzzle's, wrong sums and false claims.
  - The closest is B9 (numbers 8, 17, 20, 14, target 19). It writes 20 - 14 = 6 and 6 * 8 = 48, both right and
    from the puzzle's numbers, then says "That works" for 48.
  - B19 writes the sum of 7, 20, 18 and 6 as 40 (it is 51). B1 drifts into forum-page text.
- **Set A:** mostly web text after the `<think>` tag; A1 is a comment thread.
- **Set C:** mostly one short, wrong number (C1 to C4), so well-formed is 0.68 while only 1 of 100 is solved.

**Verdict: no, not partial.** Against the test fixed in LOG 11-21 17:59, set B has 0 of 20 by eye (yes needs 10,
partial 5), a chain rate of 0.05 (partial needs 0.10 over 1x's 0.00; yes needs 0.5) and step accuracy of 0.44 (yes
needs 0.7). It also falls short of the expected "partial, chain rate 0.3 to 0.5" written before the run. What 3x has
gained over 1x is format: well-formed answers on B (0.17 against 0.00), far fewer loops (0.07 against 0.67 on B,
0.11 against 0.91 on A), and arithmetic in the GSM8K examples' form (step accuracy 0.75 on C, against 0.00).

**Next: 6x on the 11-25 box**, as the search rule says (no at 3x goes up).
- 6x is 1280 x 32, 668M parameters. At 3x's MFU (0.24) on eight H100s (dense bf16 peak 0.989e15 each) with perfect
  scaling, 13.4B tokens (20 a parameter) take about 7.9 h, and 8.0B tokens (12 a parameter) about 4.7 h. The box
  has 24 h, so the 20-a-parameter form fits, with the probe on the box.
- This needs data-parallel pretraining (torchrun, one process a GPU) and resume in `scripts/pretrain_scratch.py`,
  off by default, with tests, run on two Engaging GPUs before 11-24. It is the one U25 item that is blocked by a box
  date, so it goes in beside the speed items (INBOX 11-21 21:25: speed items first unless box-date-blocked).
- **Data repetition at 6x.** The first half (6.7B tokens) repeats `mix_0.2.bin` (2.32B) about 2.9 times. In the
  second half, the keyword-filtered FineMath (about 520M tokens) at weight 0.5 repeats about 6.4 times (2.6 at
  3x). So the box plan adds keyword-filtered FineMath shards downloaded on the box, filtered with
  `scripts/keyword_filter.py` before use, as the hard data rule requires. The plan is written on 11-24.
- If 6x is also no, 12x (2048 x 24, 1,270M) goes to the 11-28 or 12-09 box, planned the day before.

**Jobs.**
- **r124 moved to the freed normal slot.** Its preemptable B200 job 30039308 was cancelled before it started. Its
  run is now normal B200 30051395 (the same script and rule, LOG 11-22 00:25), submitted at 03:47 and running on
  node5501-c1 since 03:50 (2 h limit): the check with the paired timing, then the two profiles.
- **A second worker for `gate_r120_h200`:** preemptable H200 30051396 (the same submit line as 30017487), for the
  items after r120's seed 1 (seed 2, then r102's seeds 3 and 4).
- 30033104 (normal B200) has run r105 lr 24e-6 seed 2 on node5601-c1 since 03:35.
- Pending: 29999306 (preemptable B200, `screens_rl_b200`), 29995938 (preemptable H200,
  requeued: r120 seed 0 again, the partial to `extra/`), 30017487 and 30051396 (preemptable H200, `gate_r120_h200`).
  Six GPUs with the two running. The held jobs (29497701/02/05, 29445218) stay held.

## 2026-11-22 04:03 EST: U25's data-parallel pretraining and resume are in (off by default), with a GPU check whose rule is written first

**What changed.** `scripts/pretrain_scratch.py` has three new options, all off by default, so the existing runs
and `scripts/research/u25_rung.sh` behave as before.
- **`--ddp`** trains data-parallel under torchrun, one process a GPU.
  - Every rank starts from rank 0's weights and draws its own rows. Rank 0 keeps the seed's old stream; the
    others draw from (seed, rank).
  - After each step's micro-batches, the gradients are averaged in flat buckets of up to 256 MB, one all_reduce
    each, before the clip. There is no DDP wrapper, so the reduction does not overlap the backward pass.
  - Rank 0's clock is broadcast every step, so the lr, the phase-2 switch and the stop are the same on every rank.
  - --batch-tokens stays the global batch: each rank runs batch / (seq x micro x GPUs) micro-batches a step.
  - The val loss is averaged over the ranks, each drawing its own 20 batches.
  - At each val (every 500 steps) the log gets `rank_spread`, the spread of the ranks' weight sums, and rank 0's
    weights are then sent to all. Only rank 0 logs and saves.
- **`--resume-every-minutes N` and `--resume`.** Every N minutes, rank 0 saves the model, both optimizers, the
  step, the tokens and the elapsed training time to `<out>/resume/state.pt` (a tmp file, then a rename).
  - `--resume` loads it. The clock and the budget carry on from the saved time, and the log is appended to.
  - Draws after a resume are reseeded from (seed, rank, step). So a resumed run's weights and optimizer states
    continue exactly, but its later rows are not the ones an uninterrupted run would have drawn.
- **CPU tests** (`tests/test_pretrain_ddp.py`, gloo with two processes):
  - the bucketed average equals the ranks' mean gradient, over two dtypes, a parameter without a gradient and
    several buckets;
  - the one-GPU step layout is the old one, and eight GPUs keep the global batch;
  - a resumed run (Muon and AdamW) continues bit-identically to the uninterrupted one;
  - a resume file with another number of optimizers is refused.
- Full suite: 217 passed; ruff clean.

**Expected overhead at 6x.** 668M parameters' fp32 gradients are 2.7 GB a step. A ring all_reduce over eight GPUs
moves 1.75 times that per GPU: about 26 ms at 180 GB/s, less on the H100s' NVLink. A step of 524,288 tokens is
6 x 668M x 524,288 = 2.1e15 FLOPs, about 1.1 s at 3x's MFU on eight H100s. So without overlap the cost should be
under 3 %.

**The GPU check.** `scripts/research/u25_ddp_check.sh` runs three arms on one node, read by
`scripts/research/u25_ddp_read.py` into `check.json`. Each arm trains D 007's shape (768 x 12, Muon lr 4e-3, the
fused loss) on `mix_0.2.bin` for 1,000 steps of 131,072 tokens.
- **single:** one process, without --ddp.
- **ddp:** torchrun on every GPU the node has, with --ddp.
- **resume:** as ddp, saving every 0.5 min. It is stopped by SIGTERM at 100 s, then run again with --resume.

**Rule.**
1. **Ranks identical:** every `rank_spread` is 0.0. The averaged gradients are the same on every rank, and the
   update is deterministic.
2. ddp's final val loss is within 0.05 of single's.
3. ddp's steady tokens a second (from step 500 to the last logged row) is at least 0.9 x GPUs x single's.
4. **Resume:** the first part was stopped by the timeout, and the rerun started from a saved step above 0. It
   ended at ddp's step and tokens, with a final val loss within 0.05 of ddp's.

**Expected and noise.** On one GPU, ddp draws the same rows as single, so their val losses should agree to about
0.01; compiled kernels can differ from run to run. On two GPUs, rank 1 draws other rows, so the runs differ like
seeds. The 0.05 bounds are loose on purpose. The CPU tests check the averaging and the saved states exactly; this
checks that the pieces run together on GPUs, at speed.

**When.**
- The one-GPU form goes on a normal B200 when r124's job ends; its two profiles are running now. With one GPU,
  ddp is torchrun with one process.
- The two-GPU form goes on two normal B200s when 30033104's item ends (r105's lr 24e-6 arm, seed 2, started 03:35).
- Both are well before 11-24. If both meet the rule, the 6x box plan (written 11-24) uses --ddp with
  --resume-every-minutes 30. A memory check of the micro-batch at 6x on an 80 GB H100 goes in that plan.

## 2026-11-22 04:13 EST: r124 read: update_host_index gives the same maths and removes most synchronisations, but its paired gain of 0.0061 s misses the rule, so it stays off

Job: normal B200 30051395 (node5501-c1, 03:50 to 04:08). Outputs: `calibration/research_r124/check/seeds/0/check.json`
and `calibration/research_r124/{default,host_index}_long/seeds/0/profile.json`.

**Rule 1, same maths: met.** At updates 2, 20 and 100, on and default gave bit-identical gradients, logged sums and nll
(`all_identical` true).

**Rule 2, paired timing: not met. Not kept.**
- 49 pairs from update 110 to 590. host_index was faster in 32 of 49 (needs 34); one-sided sign test
  p = 0.022 (needs under 0.01).
- Median of default minus host_index: 0.0061 s (needs at least 0.01). The medians were
  0.2386 s and 0.2313 s, a gain of about 2.6 %.
- Per rule 4, the option stays off, so screens and records run as before.
- The INBOX lets a same-maths speed change be kept without a paired win. The 00:25 rule asked for one, and I keep to
  the rule written first. The question comes back with r125: capture needs a loop without synchronisations, so it
  turns this option on as part of its own arm and is timed against the default.

**Rule 3, profiles: read, not ruled on.** Each arm ran 600 updates plus 12 profiled ones, with PyTorch's attention choice.

| per update | default | host_index |
|---|---|---|
| synchronisations | 63.3 | 17.7 |
| device-to-host copies | 265.7 | 234.7 |
| kernel launches | 11486 | 11351 |
| GPU busy, s (fraction of the wall time) | 0.127 (0.54) | 0.118 (0.56) |
| cudaLaunchKernel on the CPU, ms (calls) | 30.5 (9682) | 29.7 (9549) |
| cudaStreamSynchronize on the CPU, ms (calls) | 25.8 (60) | 14.1 (15) |
| aten::copy_ calls | 2878 | 3140 |
| unprofiled median, updates 400-599, s | 0.188 | 0.170 |

- Synchronisations fell from 63 to 18 an update; I expected about 10. The rest
  may come from the parts left as they were (the copies at the start of the update, the model's attention-mask code).
  r125's design has to find them, since capture allows none.
- The late medians differ by 0.019 s. That is within the 0.02 s by which separate
  arms differ, so it is not read as a gain. The paired timing is the measure.

**What it shows.** Even with most synchronisations gone, the GPU is busy only about 0.56 of
the update. The CPU spends about 30 ms an update in cudaLaunchKernel alone
(9549 calls), on top of the ops' own CPU time. So the CPU, not waiting, sets the pace, and removing
waits alone moves little. Fewer launches is what should help: capture (one replay for many kernels) or fused kernels.

**Next, as the 00:25 rule says:**
1. r125, capture of the update on top of this option, with the same-maths check and paired timing;
2. then own Triton kernels where a library kernel is the cost (aten::copy_ and aten::mul lead the GPU time).

## 2026-11-22 04:48 EST: U25's one-GPU data-parallel check meets its rule; the two-GPU form waits for both normal B200s

30052724 (normal B200, node5700-c1, 04:31 to 04:44) ran `scripts/research/u25_ddp_check.sh` with one GPU, read by
`u25_ddp_read.py` into `calibration/research_r121/u25_ddp_check_1gpu/check.json` (rule in LOG 11-22 04:03). Each arm
trained 1,000 steps, 131,072,000 tokens.

| arm | final val loss | steady tokens/s (step 500 on) | minutes | rank_spread |
|---|---|---|---|---|
| single | 3.4036 | 641,080 | 3.96 | |
| ddp (torchrun, one process) | 3.4052 | 661,726 | 3.37 | 0.0, 0.0 |
| resume | 3.3975 | 648,478 | 3.55 | 0.0, 0.0 |

**Against the rule: met.**
1. Ranks identical: both spreads are 0.0. With one rank this only shows that the spread code runs.
2. ddp's val loss is 0.0016 from single's (at most 0.05).
3. Scaling: ddp's steady rate is 1.03 x single's (at least 0.9).
   - With one process, ddp does single's work plus the bucketed average, so the 3 % is variation, not a gain.
   - single ran first. Its first 500 steps took about 32 s longer than ddp's, likely compiling into a cache that
     ddp then used.
4. Resume:
   - The first part was stopped by the timeout (rc 124).
   - The rerun loaded the file saved at step 274 (1.01 min) and ended at ddp's step and tokens.
   - Its val loss is 0.0077 from ddp's. Its rows after step 274 are reseeded, so it differs from ddp like a
     near-identical seed.

**Notes for the 6x box plan.**
- The resume file was 975 MB at 110M parameters (the weights, Muon's momentum, AdamW's two moments). By
  proportion it is about 6 GB at 668M, which is fine every 30 minutes on a box's local disk.
- At exit, torch warns that barrier() guesses the device because init_process_group was not given one. That is
  harmless here, as each process sets its device first.

**Next.** The two-GPU form (rank 1 draws other rows, so the spread and the averaging are tested for real) needs both
normal B200s at once.
- 30033104 (r105 lr 24e-6 seed 2) holds one.
- A second `screens_rl_b200` worker holds the other: 30054646, submitted 04:48 for r116's queued RL seeds, so that
  the slot freed at 04:44 did not stay empty while r125's code is written.
- r125's check takes the first normal slot that frees, as speed comes first (INBOX 11-21 21:25). The two-GPU form
  goes the first time both are free, well before 11-24.

## 2026-11-22 06:32 EST: r125, update_capture: the update's model body as CUDA graphs (off by default); its check's rule is written first

**What changed.** `rlvr_speedrun/grpo.py` has `update_capture`, off by default. It needs update_host_index and
micro_batch_tokens, and does not work with grad_checkpoint or compile_update.
- **What is captured.** The update's model body is everything before the vocabulary projection, in the three forms
  the update uses: shared-prompt groups, a batch-wide prefix, or one pass. Each bucketed shape gets a pair of CUDA
  graphs, a forward and a backward, captured the first time the shape comes and replayed after. The vocabulary
  projection, the loss and the optimizer stay eager.
- **Buckets.**
  - Rows round up in steps of 8 to 64, 16 to 128, 32 to 256, then 64.
  - The width rounds up to a multiple of 64, and shared-prompt groups to a multiple of 8.
  - The key also holds the form, the prompt width, the prefix and the autocast setting.
  - At most 128 shapes are kept. A micro-batch past that, or over 2 x micro_batch_tokens + 4,096 padded tokens,
    runs eagerly.
- **Padding.** Extra rows copy row 0, extra groups copy group 0, and extra columns go at the end with id 0 and
  mask 0. Under causal attention the real rows' real columns see the same keys. Only the real slice goes on, so the
  padding gets no gradient.
- **Gradients.**
  - Each update binds zeroed .grad tensors once, in place of zero_grad(set_to_none).
  - The backward graph adds the body's gradients into them, and the eager parts add into them in place.
  - Before the step, a check raises if any .grad was replaced.
- **Safety.**
  - A guard raises if a body's backward does not follow its own forward, exactly once.
  - All graphs share one memory pool. This is safe because each micro-batch's forward and backward finish before
    the next forward.
  - The first capture runs the body once eagerly on the capture stream, to set up the libraries.
- **Clock and logging.** The captures count in the clock. The run prints `update_capture: {counters}` at its end.
- **CPU path.** On CPU the same path runs eagerly with the same padding; the tests use it.
- **CPU tests.**
  - Over two updates, with and without shared prompts, the gradient and the replayed rows' nll match the eager
    update's, and the bound gradients keep their storage.
  - The guard refuses an out-of-order or repeated backward and a replaced .grad.
  - The option is refused without update_host_index.
- Full suite: 221 passed; ruff clean.

**Expected effect and noise.** In r124's paired timing (r106's config on a B200) the update took 0.239 s median.
- The body is 12 layers at 110M parameters, so its forward and backward are many small kernels per micro-batch.
  Replaying them as graphs removes the launch gaps and the remaining host work between them.
- Expected gain: about 0.05 s an update, in a range of 0.02 to 0.1.
- Noise: r124's 49 pairs resolved a 0.006 s median gain at p 0.022, so a gain of 0.01 s or more should be clear.
- Capture's error against fp32 should match the default's. Padding changes the summation order the way any
  change of shape does. Replays of the same graphs should be bit-identical, up to the GPU's atomic adds.
- Memory: the pool holds about one micro-batch's activations at the largest shape. On top of that there are two
  output buffers of up to 415 MB each (fp32 hidden states at the token cap). The check reports the peak reserved
  memory.

**The check (r125).** `scripts/research/check_update_capture.py` runs on one normal B200. It uses r106's
final_native_ctl config with max_steps 1,000,000, and D 007 seed 0's weights.
- **Checked updates (2, 20 and 100).** Every arm runs with update_host_index on. Each update runs five times on the
  same inputs, with an optimizer step that does nothing:
  1. default;
  2. default again;
  3. capture (capturing new shapes);
  4. capture again (replaying them);
  5. an fp32 reference without autocast.
- **Paired timing.** Updates 110 to 590, every 10, give 49 pairs of default and capture, alternating which runs
  first.
- **Profiles.** Then two profile arms (host_index and capture), each 600 updates plus 12 profiled.

**Rule.**
1. **Same maths up to rounding.** At every checked update:
   - capture's gradient relative error to the fp32 reference is at most 1.25 x the default's + 1e-4;
   - capture's largest relative error in the logged sums, and in the replayed rows' nll, is at most
     max(2 x the default's, 1e-4).
2. **Timing.** At least 30 pairs, a median of default minus capture of at least 0.01 s, and a one-sided sign test
   with p < 0.01.

**What follows.**
- If both parts hold, new configs turn update_capture on (r102–r121's stay frozen), and own kernels come next where
  a library kernel is the cost.
- If the maths holds but the timing does not, the option stays in the code, off.

**When.** It runs on the first normal B200 that frees. Both are busy now:
- 30033104 holds r105's lr 24e-6 arm, seed 2;
- 30054646 holds r116's seed 0.

## 2026-11-22 08:31 EST: r105 lr 24e-6 seed 2 is +0.010 at matched steps, so its three pairs give +0.0144, short of the 0.015 rule; lr 24e-6 is closed for the native regime and the gates keep lr 1.2e-5; r125's check is submitted on the freed normal B200

**r105 lr 24e-6 seed 2** (normal B200 30033104, node5601-c1, 03:36 to 08:27): 0.2950 after 50,132 steps against
the native control's seed 2, 0.2833 after 44,987 (node5600-c1).
- Matched steps: +0.0100 over 89 shared evals (ahead at 66, behind at 19). By step block: 0-5k +0.015, 5-10k
  +0.019, 10-20k +0.018, 20-30k +0.012, 30-50k -0.002. The lead is gone after 30,000 steps.
- Last five evals: 0.2993 against 0.2803. Final: +0.0117.
- The arm took 2.88 steps a second against the control's 2.59 (mean t_sample 0.128 s against 0.141, t_update
  0.184 s against 0.209). lr 12e-6's seed 2 went the other way (2.32 against 2.59), so I put this down to the node.

All three pairs (computed by script from the eval logs):

| seed | lr 24e-6 final (steps) | control final (steps) | final diff | matched steps (shared evals) |
|---|---|---|---|---|
| 0 | 0.2967 (45,636) | 0.3033 (34,893) | -0.0067 | +0.0020 (69) |
| 1 | 0.3367 (47,812) | 0.2883 (44,862) | +0.0483 | +0.0311 (89) |
| 2 | 0.2950 (50,132) | 0.2833 (44,987) | +0.0117 | +0.0100 (89) |
| mean | 0.3094 | 0.2917 | +0.0178 (SE 0.0162) | +0.0144 (SE 0.0087) |

**Rule** (LOG 11-19 09:28): a gain of 0.015 or more over the three pairs at matched steps. Seed 2 needed about
+0.012 (LOG 11-21 17:59).
- **Not met: +0.0144 at matched steps.** The finals give +0.0178, but the rule is at matched steps because the
  arms run different numbers of steps on different nodes.
- As written at 17:59, lr 24e-6 is closed for the native regime. r120's lr 1.2e-5 stays the lr in the H200 gates,
  and no 2.4e-5 gate config is written.
- **What it says.** At matched steps both doses are ahead of the native control by about the same amount: +0.016
  at 2x and +0.014 at 4x. So 4x is not better than 2x. In seed 2 the 4x arm's lead fades after 30,000 steps.

**Placement.** 30033104's worker exited, with less than 325 minutes left and its queue empty.
- r125's check (LOG 06:32) was submitted to `mit_normal_gpu` as 30065238 (2 h limit). It is pending on resources.
- The six GPUs are:
  - two normal B200s: r116 seed 0 (30054646) and r125;
  - one preemptable B200, pending: r116 seed 1 (29999306), which was requeued after preemption and has started
    five times since 05:56;
  - three preemptable H200s: r120 seeds 0-2.
- r102's seeds 3 and 4 wait on `gate_r120_h200` behind r120's seeds.
- When r125 and r116 seed 0 have both ended, U25's two-GPU check takes both normal B200s (it is box-date-bound,
  by 11-24). If 29999306 is still being preempted then, r116 seed 1 goes to a normal B200 after that.

## 2026-11-22 09:06 EST: r125 update_capture meets both parts of its rule, so new configs turn it on with update_host_index

**The check** (normal B200 30065238, node5601-c1, 08:30 to 08:47 EST;
`calibration/research_r125/check/seeds/0/check.json`): r106's `final_native_ctl` config from D 007 seed 0's
checkpoint, update_host_index on in both arms, capture against the default path.

1. **Maths, met.** Gradient error against the fp32 reference, capture and the default, at three updates:

| update | capture grad_rel_err | default grad_rel_err | sums_max_rel capture / default | nll_rel capture / default | capture vs default |
|---|---|---|---|---|---|
| 2 | 0.0593 | 0.0569 | 0.0626 / 0.0626 | 7.4e-05 / 1.9e-03 | 0.0271 |
| 20 | 0.0603 | 0.0611 | 0.0240 / 0.0240 | 3.3e-03 / 1.1e-02 | 0.0313 |
| 100 | 0.0517 | 0.0521 | 0.0708 / 0.0708 | 1.1e-03 / 4.0e-03 | 0.0377 |

   Each capture error is within 1.25 x the default's + 1e-4, the sums match, and the loss is closer to fp32 than
   the default's. Repeats of each path are identical (rel err 0), and so are the graph replays.
2. **Timing, met.** 49 pairs from update 110 to 590: capture was faster in 35, sign p
   0.0019, median gain 0.0183 s (rule: at least 30 pairs, median 0.01 s or more,
   p under 0.01). The median update went from 0.228 to 0.188 s. By updates: 110-290: 19 pairs, median 0.007 s, 10 faster; 300-440: 15 pairs, median 0.042 s, 13 faster; 450-590: 15 pairs, median 0.019 s, 12 faster.
   So the gain holds late in the run (r122 found that r119's start-up gain did not). A few capture updates were slow
   (0.385 s against 0.231 at update 120), where a new shape was captured.
- **Captures:** 14 shapes, 141 replays, 0 fallbacks, 0 resets; 3.36 s of capturing in all,
  inside the timed update, so in the clock. The check's peak reserved memory, over all its arms, was
  62.2 GB.
- torch warns once that the AccumulateGrad node's stream differs from the stream that made the gradient. Any
  synchronisation that costs is inside the timings above.

**The two 600-update profiles** (same job; `calibration/research_r125/{host_index_long,capture_long}/seeds/0/profile.json`).
They are not paired: the paths' rounding differs, so the rollouts part after the first update.
- Unprofiled update, median of 600: 0.223 s with update_host_index alone, 0.179 s with capture. By
  window, host_index against capture: 1-49: 0.291 against 0.200 s; 50-199: 0.231 against 0.184 s; 200-399: 0.172 against 0.177 s; 400-599: 0.195 against 0.172 s.
- At updates 600 to 611 (profiled): kernel launches 11351 to 288 an update, synchronisations
  17.7 in both, GPU busy 0.49 to 0.87 of the wall time. Median allocated peak 28.3 to 20.8 GB.
- GPU busy time rose from 0.117 to 0.151 s an update. Part of that should be the padding to bucket
  shapes (rows to steps of 8 to 64, width to 64, groups to 8). How much is padding is the next measurement.
- The update is now bound by GPU work, not launches. What is left is that work: the padding; the elementwise and
  copy kernels (the four largest add to 40 ms an update; which are inside the graphs and which are in the
  optimizer step is part of the next measurement); and attention, where cuDNN runs sm80 flash kernels on the
  B200 (17 ms an update).

**Per the rule written first (LOG 11-22 06:32): met.** New configs turn update_capture on, with
update_host_index, which it needs. r102 to r121 are frozen, so r120's record attempt, if its rule is met, uses
r120's config as written. The next gate config is the first to carry capture. Gates and records run on H200 and
this check ran on a B200, so the same check, without the profiles (about 10 minutes), runs on an H200 before a
gate uses it, on the first free preemptable H200 slot. Own kernels come next, starting with the padding count
and where the elementwise work comes from.

**Jobs (11-22 09:06).** 30054646 (normal B200, node5700-c1) has run r116 seed 0 since 04:50. 29999306 (preemptable
B200) started r116 seed 1 five times between 05:56 and 07:57, and each start was preempted; it is pending again.
Three preemptable H200 workers run r120 seeds 0 to 2 (29995938 and 30051396 on node4803, 30017487 on node4601).
5 GPUs in use or pending. The free normal slot stays free: when r116 seed 0 ends, U25's two-GPU check takes both
normal B200s (by 11-24), then r116 seed 1 goes to a normal B200 if 29999306 is still being preempted.

## 2026-11-22 09:10 EST: r126, capture's check on an H200, with padding counts; its rule is written first

**What changed** (60c9d0b, stats only, the same maths). update_capture's stats now add up two counts over all
replays:
- the micro-batches' own tokens (rows x width) and their buckets' tokens (bucket rows x bucket width);
- with the shared prompt prefix on, their groups and their buckets' groups.

They are host integers, so they add no synchronisation. A micro-batch's own width is already its longest row's, so
the counts measure only the padding that bucketing adds on top. CPU tests: 221 passed; ruff clean.

**The check.** r125's check (`scripts/research/check_update_capture.py`, r106's config, D 007's seed-0 weights,
timing at updates 100 to 600, every 10) runs again, without the profiles, on one preemptable H200 (30067245).
Its output goes to `calibration/research_r125/check_pad_h200/seeds/0/check.json`. Gates and records run on
H200s, and r125 ran on a B200.

**Rule.**
1. **r125's rule** (LOG 11-22 06:32), both parts, on the H200.
   - If both hold, the next gate config turns update_capture on, with update_host_index.
   - If the maths holds but the timing does not, gates keep it off; B200 screens may still turn it on.
2. **Padding.** padding = 1 - own tokens / bucket tokens, over all replays.
   - At 0.10 or more, finer buckets are the next item, with a paired timing check: width in steps of 16, and rows
     in steps of 8 throughout.
   - Under 0.10, the padding is not where the busy time went, and the elementwise and copy kernels are next.
   - The groups' count is read, not ruled on.

**Expected and noise.** My guess is 0.10 to 0.25: if widths are a few hundred tokens, padding them to 64 adds
about 32 columns a row on average, and rows are padded about half a step on average. I have not measured the
widths; that is what the count is for. The count is exact for this run's shapes. The shapes come from the rollouts, which differ by GPU and seed,
so it stands for this config at these weights, not for every config.

**GPUs.** 6 with this job: 30054646 (normal B200, r116 seed 0), 29999306 (preemptable B200, pending, r116
seed 1), three preemptable H200s (r120 seeds 0 to 2) and 30067245. When r116 seed 0 ends, U25's two-GPU check
waits for 30067245 to end, so that the total stays at 6. r125's B200 check took 6 minutes.

## 2026-11-22 09:37 EST: r126, on an H200 capture keeps the maths but is slower, so gates keep it off; padding is 0.196 of the bucket tokens, so finer buckets are next

Preemptable H200 30067245 (node4601) ran r125's check from 09:11 to 09:17. The output is
`calibration/research_r125/check_pad_h200/seeds/0/check.json`. Both arms run with update_host_index on; "default"
is without capture.

**Maths: holds.** At each checked update, capture's gradient error is within its bound. The logged sums and the
replayed rows' nll equal the default's. Replays are identical.

| update | capture's grad rel. error vs fp32 | default's | bound (1.25 x default + 1e-4) |
|---|---|---|---|
| 2 | 0.0454 | 0.0455 | 0.0569 |
| 20 | 0.0933 | 0.0933 | 0.1168 |
| 100 | 0.0506 | 0.0506 | 0.0634 |

**Timing: not met.**
- Overall: 49 pairs; 17 faster with capture; sign test p 0.989; median gain (default minus capture) -0.022 s.
- Unpaired medians: 0.228 s default, 0.203 s capture. The times are spread out, so the paired gain is what the
  rule reads.

By update band, H200 against r125's B200:

| updates | H200 median gain | H200 pairs faster | B200 median gain | B200 pairs faster |
|---|---|---|---|---|
| 110-290 | -0.021 s | 6/19 | 0.007 s | 10/19 |
| 300-440 | -0.024 s | 5/15 | 0.042 s | 13/15 |
| 450-590 | -0.008 s | 6/15 | 0.019 s | 12/15 |

By the default's own time:

| default's time | H200 pairs | faster | median gain | B200 pairs | faster | median gain |
|---|---|---|---|---|---|---|
| under 0.2 s | 24 | 0 | -0.025 s | 21 | 12 | 0.003 s |
| 0.2-0.3 s | 22 | 15 | 0.009 s | 19 | 16 | 0.043 s |
| over 0.3 s | 3 | 2 | 0.014 s | 9 | 7 | 0.119 s |

- So on the H200 capture loses on the short updates and gains little on the others.
- Times over 0.3 s: 8 with capture and 3 without (on the B200, 8 and 9). I don't know why. A new shape's capture
  inside a timed update would explain it, but the check does not record where captures happen.

**Captures and memory.** 14 shapes, 135 replays, no fallbacks, 2.74 s of capturing. The peak reserved memory
over all arms was 62.4 GB.

**Padding.** Over all replays, the micro-batches' own tokens were 4,413,298 and their buckets' were 5,490,688:
padding 0.196. Their groups were 2,168 and the buckets' 2,488: 0.129. The guess written first was 0.10 to 0.25.

**Per the rule written first (LOG 11-22 09:10):**
1. **Maths holds, timing does not.** Gate and record configs keep update_capture off. This changes LOG 11-22
   09:06, where new configs were to turn it on after this check. B200 screen configs may still turn it on.
2. **Padding 0.196 is at least 0.10.** Finer buckets (r127) are the next item:
   - new options, off by default: width in steps of 16, and rows in steps of 8 throughout;
   - the same paired check with r125's rule, on an H200, the GPU where capture loses now;
   - the check also records which timed updates capture a new shape.

   Finer buckets mean more shapes, and each capture costs about 0.2 s on the H200 (2.74 s over 14), inside the
   clock. So the r127 entry gives the number of shapes and the capture seconds beside the gain.
3. The elementwise and copy kernels, then attention, follow r127.

**Jobs (11-22 09:37).** 30054646 (normal B200, r116 seed 0) has run since 04:50; 29999306 (preemptable B200, r116 seed
1) is pending; three preemptable H200s run r120 seeds 0 to 2. 5 GPUs. When r116 seed 0 ends, U25's two-GPU check
takes both normal B200s (6 GPUs), and r127's H200 check goes in after it.


## 2026-11-22 09:44 EST: r127, capture with finer buckets, timed with warm pairs on an H200; its rule is written first

**What changes.** Three new options, off by default (their defaults are the buckets r125 and r126 used):
- `update_capture_width_step` (64): the width is padded to a multiple of it;
- `update_capture_row_step` (0): rows are padded to a multiple of it; 0 keeps the tiered steps (8 up to 64 rows,
  16 up to 128, 32 up to 256, then 64);
- `update_capture_group_step` (8): shared-prompt groups are padded to a multiple of it.

The check gets `--warm`: before each timed pair, each arm runs once, untimed, on the same inputs. So the timed
capture arm replays shapes that are already captured. Each timed row now also records the capture arm's new
captures and its micro-batches' shapes.

**Why warm pairs.** In r126 the capture arm captured each new shape inside the update that first met it. Timed
updates may have been among them; the check did not record which. A real run captures a shape once. B200 gate
screens made 21,695 to 63,559 updates each (r113, r115 and r117 result.json total_steps). The captures
still count in a run's clock, so the entry reports their number and seconds beside the gain. At most 128 shapes are
captured (`_MAX_GRAPHS`); at r126's 0.2 s each, that is at most about 26 s per run. A median gain of 0.01 s over
21,000 updates is 210 s.

**Arms.** One preemptable H200 job runs both, in this order:
1. **fine:** width step 16, row step 8, group step 8, warm pairs.
2. **coarse:** the default buckets, warm pairs. This is r126 again with warm pairs. It shows how much of r126's
   loss came from captures inside timed updates.

Everything else is as in r125 and r126:
- the r106 `final_native_ctl` config with max_steps raised, and the r93 seed 0 weights;
- checked updates 2, 20 and 100;
- timed updates 100 to 590, every 10th.

**Expected effect (guesses).**
- Fine padding: about 0.05 of the bucket tokens (r126: 0.196).
- Fine shapes: 30 to 80 (r126: 14).
- r126's updates under 0.2 s lost 0.025 s each. If most of that is padding, finer buckets win back about half of
  it. So I expect fine to be near even on the H200 and to miss the timing rule. I expect coarse with warm pairs to
  stay below even too.

**Noise.** r126's 49 paired gains had median -0.022 s, quartiles -0.026 s and 0.009 s, and SD 0.055 s.

**Rule.**
1. **Maths.** r125's rule must hold at every checked update, in each arm.
2. **Fine wins.** If fine meets r125's timing rule with warm pairs (at least 30 pairs, median gain at least 0.01 s,
   sign test p < 0.01) and has no fallbacks, new H200 gate and record configs turn update_capture on with the fine
   steps.
3. **Only coarse wins.** If only coarse meets it, they turn it on with the default buckets.
4. **Both win.** The arm with the larger median gain is used.
5. **Neither wins.** Capture stays off on the H200, and the elementwise and copy kernels are next.

The pairs without a new capture are reported in every case.

**Jobs (2026-11-22 09:44).** 5 GPUs are in use, so r127 takes the 6th now, ahead of U25's two-GPU check, which was to come
first (LOG 11-22 09:37). U25 starts when r116 seed 0 (30054646) has ended and r127 is done, keeping 6 at most.


## 2026-11-22 09:49 EST: r116 seed 0 (U24) is +0.037 over its control at the final; the rule waits for seed 1

**The run.** `calibration/research_r116/d116rl_native_fmkw2_b200` seed 0: r101's native control, unchanged, from
the FineMath-keyword pretraining (LOG 11-21 12:40). Worker 30054646, normal B200 node5700-c1, 04:50 to 09:41. Its
pair is r101's `d007rl_native_b200` seed 0 (node5600-c1). Both trained for 17,400 s. Numbers are from
`read_time.py` and a length script (scratchpad), dev only.

| | r116 seed 0 | control seed 0 |
|---|---|---|
| final dev | 0.3400 (step 48,567) | 0.3033 (step 34,893) |
| mean of the last five evals | 0.3317 | 0.2920 |
| steps per second | 2.791 | 2.005 |
| steps in each fifth of train time (58 min) | 7,416 / 9,183 / 10,311 / 10,735 / 10,922 | 5,655 / 6,041 / 6,650 / 7,358 / 9,189 |
| late wall per step (last 2,000) | 0.317 s | 0.338 s |
| late informative groups / batch solve rate | 24.8 / 0.366 | 29.2 / 0.322 |
| completion length, steps 101 to 2,100 | 24.0 | 26.0 |
| completion length, last 2,000 steps | 24.8 | 26.8 |

- **Final:** +0.0367.
- **Matched train time:** 98 evals, mean +0.0462 (SD 0.0140), ahead in 98 and behind in 0. By hour: +0.041,
  +0.040, +0.055, +0.049, +0.043.
- **Matched steps** (to step 34,500): 69 evals, mean +0.0316, ahead in 69 and behind in 0. Steps 1 to 10,000:
  +0.030; 10,001 to 20,000: +0.028; 20,001 to 35,000: +0.035.
- **Completion length** is the median over the steps of tokens per sample (`tokens/s` x step time / 512). It stays
  near 24 to 25 tokens, below the 40 that would mean written reasoning, so the regime is unchanged.
- **Speed:** the arm made 39 % more steps. The two ran on different nodes, so part of that may be the node; this
  was not measured. The matched-steps gain (+0.032) does not depend on speed.

**Against what I expected.** The centre was -0.01 and the range -0.05 to +0.03. Seed 0 is above the range.

**Rule (LOG 11-21 12:40).** It needs both pairs. Seed 0's pair is over the +0.02 bar at the final and ahead at
matched steps. Nothing is decided until seed 1 ends.

**Seed 1.** 29999306 (preemptable B200) started it six times between 05:56 and 09:43, and each start was preempted.
No-save screens restart from the beginning, so I cancelled 29999306 while it was pending. The item went back into
`screens_rl_b200`, and normal B200 worker 30069082 (6 h) was submitted for it at 09:49.

**Order change.** U25's two-GPU check (1 h, both normal B200s) was to come first (LOG 11-22 09:06). It now runs
after seed 1 ends, at about 15:00 if 30069082 starts soon; that is still well before 11-24. So seed 1 uses the
normal slot that is free now, and r116's next step is not held back by about 1.5 h.

**GPUs (09:49).** 5: three preemptable H200s on r120 seeds 0 to 2, r127 (30068951, preemptable H200) and 30069082
(normal B200, pending).

## 2026-11-22 10:21 EST: r127, finer buckets meet capture's rule on the H200 (faster in 49 of 49 pairs, median 0.012 s), so new H200 gate and record configs turn capture on with them

**The check** (preemptable H200 30068951, node4602, 09:45 to 09:57 EST;
`calibration/research_r127/{fine,coarse}_warm_h200/seeds/0/check.json`). Both arms ended with rc 0, on r106's
`final_native_ctl` config and r93 seed 0's weights, as written first.

**Maths: holds in both arms.** Gradient error to the fp32 reference:

| update | fine, capture | fine, default | coarse, capture | coarse, default |
|---|---|---|---|---|
| 2 | 0.0508 | 0.0508 | 0.0528 | 0.0528 |
| 20 | 0.0654 | 0.0654 | 0.0639 | 0.0640 |
| 100 | 0.0579 | 0.0580 | 0.0541 | 0.0542 |

Each capture error is within 1.25 x the default's + 1e-4, and the logged sums and the replayed rows' nll are within
their bounds. Replays are identical in both arms.

**Timing** (warm pairs at updates 100 to 590, every 10th; gain = default minus capture):

| arm | pairs | capture faster | sign p | median gain | quartiles | default median | capture median | pairs with a new capture |
|---|---|---|---|---|---|---|---|---|
| fine | 49 | 49 | 1.78e-15 | 0.012 s | 0.008, 0.041 s | 0.182 s | 0.174 s | 0 |
| coarse | 49 | 16 | 0.995 | -0.009 s | -0.023, 0.011 s | 0.181 s | 0.203 s | 0 |

Fine by band: 100-290: 19 of 19 faster, median 0.010 s; 300-440: 15 of 15 faster, median 0.017 s; 450-590: 15 of 15 faster, median 0.010 s. So the gain holds through the window.

**Captures, padding and memory** (padding = 1 - own / bucket, over all replays):

| arm | shapes captured | capturing | replays | fallbacks | padding, tokens | padding, groups | peak reserved |
|---|---|---|---|---|---|---|---|
| fine | 20 | 3.65 s | 254 | 0 | 0.058 | 0.117 | 57.9 GB |
| coarse | 13 | 2.85 s | 244 | 0 | 0.190 | 0.123 | 62.0 GB |

- The timed updates' micro-batches had 61 distinct own shapes in fine and 52 in coarse.
- Bucket tokens per replay were 34,551 in fine against 41,044 in coarse (0.84 x). The capture
  medians were 0.174 against 0.203 s (0.86 x). So capture's time fell about as much as its padded tokens did.

**The default's time depends on the order.** Split by the arm timed first:

| check | timed first | pairs | default median | capture median | median gain | capture faster |
|---|---|---|---|---|---|---|
| r126 (cold) | default | 25 | 0.179 s | 0.203 s | -0.024 s | 7 |
| r126 (cold) | capture | 24 | 0.237 s | 0.204 s | -0.007 s | 10 |
| r127 fine | default | 25 | 0.181 s | 0.174 s | +0.010 s | 25 |
| r127 fine | capture | 24 | 0.240 s | 0.199 s | +0.027 s | 24 |
| r127 coarse | default | 25 | 0.181 s | 0.203 s | -0.007 s | 8 |
| r127 coarse | capture | 24 | 0.180 s | 0.203 s | -0.013 s | 8 |

- The default took about 0.24 s when it was timed second, in r126 and in fine, and about 0.18 s otherwise. In
  coarse with warm pairs the order made no difference. I don't know why.
- A real run has one path and runs each update once, so neither order is a real run's update. My guess is that the
  rows with the default timed first are the fairer ones: there fine's median gain is 0.0096 s, faster in
  25 of 25. That is just under 0.01 s. The rule written first reads all 49 pairs (0.012 s), so it is
  met. In every pair, in either order, fine was faster.

**Per the rule written first (LOG 11-22 09:44): fine wins.**
- The maths holds in both arms.
- Fine meets r125's timing rule with warm pairs, with no fallbacks; coarse does not.
- So new H200 gate and record configs turn update_capture on, with update_host_index and the fine steps:
  `update_capture_width_step` 16, `update_capture_row_step` 8 and `update_capture_group_step` 8.
- r102 to r121 stay frozen: if r120's rule is met, its record uses r120's config as written.

**Against the guesses.**
- Padding: 0.058 (guess: about 0.05).
- Shapes: 20 captured (guess: 30 to 80); the 61 distinct own shapes fell into fewer buckets than I expected.
- Fine was to be near even and to miss the rule; it met it. Coarse stayed below even, as guessed.

**What a real run should get (a guess).**
- r102's H200 seeds made 38,914 to 43,073 RL updates in 18,000 s. At 0.0096 to 0.012 s an update, that is
  about 372 to 517 s, 2 to 3 % of the RL time.
- Less the captures: 3.6 s for 20 shapes here. Past 128 shapes the rest run eager, and the run prints its counts
  at the end.
- The end-to-end check is the first H200 gate with capture: its RL steps against those of seeds without capture
  that differ in nothing else (r102, r120), by seed.
- B200: r125 met the rule there with the default buckets; the fine steps have not been checked on a B200.

**Next speed item** (LOG 11-22 09:37): the elementwise and copy kernels, then attention. First, a profile of the
update with fine capture on an H200 (`scripts/research/profile_update.py`). It shows which of those kernels run
inside the graphs and which in the optimizer step. Its rule is written with it.

**Jobs (10:21).**
- 30069082 (normal B200, node5700-c1) has run r116 seed 1 since 09:51.
- Three preemptable H200s run r120 seeds 0 to 2 (29995938 and 30051396 on node4803, 30017487 on node4601).
  r102 seeds 3 and 4 follow on `queue/gate_r120_h200.txt`.
- 4 GPUs.

## 2026-11-22 10:26 EST: r128 rule, where a step's time goes on an H200 with fine capture (one preemptable H200, 30071155)

**Why.** r127 put capture with the fine steps into the next H200 gate and record configs. The next speed items
(INBOX 21:25: efficiency first, own Triton or CUDA kernels welcome) need to know where a step's time goes now.
- r125's B200 profile (capture, default buckets) put 139 ms of an update's device time in the two graph ops. After
  them came eager elementwise and copy kernels (13.4, 13.1 and 7.2 ms an update for the top three).
- r106's part timers (B200, no capture, medians over steps 150 to 299 of seed 0) put the optimizer step at
  1.8 ms and clipping at 1.0 ms a step. So those kernels are not mostly in the optimizer step, as the
  10:21 entry guessed. My guess now: the head (vocabulary projection, mask, logsumexp, gather, entropy and their
  backward), the loss and the bound gradients' adds.
- The same r106 medians: fwd_mask 10.3 ms, fwd_head 3.3 ms, fwd_body 66.7 ms,
  bwd 84.0 ms, sil 53.7 ms, replay_batch 14.1 ms, s_generate 126.4 ms
  with the sampler mask's CPU time (s_mask_cpu) 87.0 ms inside it.

**Arms** (`calibration/research_r128/`, `scripts/research/profile_update.py`). Both use r106's config (the gate's
RL settings), D 007 seed 0's final weights, max_steps 1,000,000 and capture with the fine steps (16, 8, 8):
- `fine_profile_h200`: 600 timed updates, then 12 under torch.profiler. Device time by op (inside the graphs:
  `_GraphedBody`, `_GraphedBodyBackward`; the rest by kernel), busy fraction, launches.
- `fine_parts_h200`: the same with `time_update_parts` on (a sync at each part), grpo's train log kept. Each step's
  sampling parts (s_encode, s_generate with s_mask_cpu inside it, s_decode, s_reward), the update's parts (fwd_body,
  fwd_mask, fwd_head, loss, bwd, sil, clip, opt) and the rest (select, allowed, replay_batch, post).

**Rule (picks the next item; nothing here changes a config).** Medians over steps 100 to 599 of fine_parts, and
fine_profile's device time per update. The candidates, each its own piece of work:
- (a) the head's forward: fwd_mask + fwd_head;
- (b) the sampler mask's CPU time: s_mask_cpu. At the final weights r106's `factored_grammar` did not lower it
  (LOG 11-20 04:12 and 10:26), so a change here would be in how the mask is applied per token;
- (c) replay_batch;
- (d) attention inside the graphs: the attention kernels' device time per update in fine_profile.

The next item is the largest of (a) to (d), if it is at least 10 ms a step. Within 20 % of each other, the one with a
same-maths change (no loss comparison needed) goes first. Also reported: the update's device time outside the two
graph ops (the head's backward, the loss, the adds, the optimizer), to size the head's backward.

**Expected (guesses).**
- Update wall time in fine_profile 0.17 to 0.20 s (r127's warm pairs: 0.174 s median); the graphs 55 to 70 % of its
  device time; busy fraction 0.85 to 0.95.
- (a) 10 to 20 ms; (b) 50 to 90 ms, inside s_generate of 100 to 150 ms; (c) 10 to 15 ms; (d) 15 to 30 ms.
- So my guess for the next item is (b), the sampler's mask, if most of its CPU time is on the step's path. If the
  GPU waits on it, s_generate less s_mask_cpu should be small.

**Noise.** The part timers' syncs lengthen each part, so I read the order and shares, not absolute times. Medians are
over 500 steps of one seed, on one node.

**Jobs (10:25).** 30071155 (preemptable H200, 50 min limit) runs both arms. The first submission, 30071118, started
on node5100, where the pool venv hangs, and I cancelled it at 10:26; 30071155 has the H200 workers' exclude list. r116 seed 1 (30069082) and r120 seeds 0 to
2 (three preemptable H200s) as at 10:21. 5 GPUs.

## 2026-11-22 11:02 EST: r128 read, the sampler mask's CPU time is the largest item (104 ms a step); the next speed item is the native sampler's mask

**Result.** 30071155 ran both arms on node4300 (10:27 to 10:38 EST, COMPLETED). D 007 seed 0's final weights, r106's
config, capture with the fine steps, one H200.
- fine_parts, medians over steps 100 to 599 (500 steps; one main and one sil micro-batch at the median):
  - the step's wall time (differences of the train log's wall_s): 417.9 ms;
  - sampling 173.0 ms: s_generate 151.4 with s_mask_cpu 104.1 inside it
    (median share 0.69), s_reward 11.8, s_encode 5.7, s_decode 3.8;
  - the update 188.7 ms: bwd 64.3, sil 53.8, fwd_body 34.9, setup
    9.5, fwd_head 4.5, fwd_mask 3.5, opt 2.6, clip 0.9,
    loss 0.4;
  - between them: replay_batch 13.3, select 0.4, allowed 0.2, post 0.2;
  - about 40 ms of the step is in no timed part (the step's median less the medians of sample + update, replay_batch,
    select, allowed and post; medians don't add exactly).
- fine_profile: the update's unprofiled wall time 174.5 ms median (172.0 ms over
  updates 400 to 599), device busy 158.5 ms (fraction 0.895). The two graph ops: 147.0 ms an
  update (93.2 backward, 53.9 forward), 0.93 of the busy time. Outside them: 11 to 21 ms an
  update (busy time less the graphs, or half the table's total less the graphs).
- The op table counts each kernel under its own name and under the op that launched it: its total, 335 ms an
  update, is about twice the busy time. So its kernel rows include the graphs' own kernels. The top elementwise rows
  (13.4, 13.4, 10.3 ms an update, with 1050, 870, 1532 calls an update against
  2.3 graph replays) are probably inside the graphs. My guess is that r125's rows,
  which the 10:26 entry read as eager kernels, were mostly the graphs' kernels too.
- The attention kernels (cuDNN flash forward and backward): 4.1 + 7.4 = 11.4 ms an update, inside the graphs or not.

**Rule.**
- (a) the head's forward (median of fwd_mask + fwd_head): 7.9 ms;
- (b) s_mask_cpu: 104.1 ms;
- (c) replay_batch: 13.3 ms;
- (d) attention: at most 11.4 ms.

(b) is the largest, at least 10 ms, and more than 20 % above the next ((c)). **So the next speed item is (b), the
sampler mask's CPU time.**

**Against the guesses.**
- Inside the ranges: the update 0.175 s (0.17 to 0.20); busy fraction
  0.895 (0.85 to 0.95); (c) 13.3 (10 to 15).
- Outside them:
  - the graphs' share is 0.93 of the busy time against 55 to 70 %, probably because the elementwise kernels are inside them;
  - (a) 7.9 (guess 10 to 20);
  - (b) 104.1 (guess 50 to 90), with s_generate 151.4 (guess 100 to 150);
  - (d) 11.4 (guess 15 to 30).
- s_generate less s_mask_cpu is 47.4 ms (median), so most of s_generate is the mask's CPU time.

**Also seen (not a candidate here).** sil, one micro-batch of 64 replayed rows, takes 53.8 ms. That is
0.54 of the main batch's fwd_body + bwd (99.2 ms for 512 rows). A guess why: replayed rows don't share
prompts, while the main batch runs one prompt pass for each prompt's 8 completions. So each replayed row carries its
whole prompt. This is a later item.

**Next.** (b) In the native sampler (`rlvr_speedrun/fast_rollout.py`), each token step does three things in order:
- runs the answer grammar for every active row in Python (`RequestConstraint.allowed_ids`);
- masks the logits (`mask_rows_cached`);
- samples and calls `.tolist()`, which waits for the GPU.

The GPU waits while the CPU builds the mask. The next entry gives the change, how I'll check it and its rule before
anything runs. A change that gives exactly the same masks needs no loss comparison.

**Files.** Both arms' profile.json; fine_parts' run/config.json and run/train_log.jsonl. The stdout.logs are
gitignored.

**Jobs (11:02).**
- r116 seed 1 on a normal B200 (30069082).
- r120 seeds 0 to 2 on three preemptable H200s (30017487, 29995938, 30051396).
- 4 GPUs.
## 2026-11-22 11:06 EST: r120 seeds 0 and 1 end at 0.310 and 0.302 on dev, both above D's line; the rule waits for seed 2

**The runs.** `calibration/research_r120/d007_speed_lr12e-6_h200`, r102's H200 gate with lr 1.2e-5 (LOG 11-21
17:13). Each pairs with r102's seed of the same number (`d007_speed_h200`). Numbers are from `read_time.py` with a
second seed directory (scratchpad), dev only. Both arms trained for 18,000 s.

| | seed 0 | seed 1 |
|---|---|---|
| r120 final dev (steps, node) | 0.3100 (43,659, node4803) | 0.3017 (46,332, node4601) |
| r102 final dev (steps, node) | 0.2700 (38,914, node4802) | 0.3100 (43,073, node4602) |
| final diff | +0.0400 | -0.0083 |
| matched train time: evals, mean, ahead / behind | 86, +0.0363, 86 / 0 | 91, +0.0135, 73 / 16 |
| matched steps: evals, mean, ahead / behind | 77 (to 38,500), +0.0283, 76 / 0 | 86 (to 43,000), +0.0097, 64 / 20 |
| r120 at r102's last step | 0.3067 | 0.2883 |
| mean of the last five evals, r120 / r102 | 0.3217 / 0.2720 | 0.2950 / 0.3010 |
| steps per second, r120 / r102 | 2.425 / 2.162 (1.122) | 2.574 / 2.393 (1.076) |
| late wall per step, r120 / r102 | 0.374 / 0.376 s | 0.355 / 0.373 s |

- **Seed 0** is ahead in every matched eval, by about the same amount each hour (+0.032, +0.036, +0.037, +0.036,
  +0.039). At matched steps: +0.022, +0.030, +0.031, +0.029 for steps 1 to 10,000, 10,001 to 20,000, 20,001 to
  35,000 and 35,001 on.
- **Seed 1** is ahead early and even late. By hour: +0.019, +0.028, +0.019, +0.003, +0.001. At matched steps:
  +0.018, +0.019, +0.004, -0.003 over the same step ranges. r102's seed 1 was the highest of its three.
- **Speed:** r120 made 12 % and 8 % more steps. Each pair ran on different nodes and days, so this may be the node
  or the code at the time; it was not measured. Late wall per step is close in both pairs.
- **Clocks:** 20,797 s and 20,789 s of 21,600 (pretraining 2,775 s and 2,772 s).
- Both workers logged FAILED at the final `mv`; this is the no_save gate's known line, and result.json and
  clock.json are complete. Seed 0's first start was preempted; its partial is in `extra/interrupted_0_1`.

**Against what I expected.** The centre was about 0.285 a seed if r105's B200 gain carried over, 0.278 if not.
Both seeds are above it. The two-seed mean is 0.3058. Over the two pairs, r120 is +0.016 at the
final, +0.025 at matched train time and +0.019 at matched steps. r105's lr 12e-6 was +0.016 at
matched steps over three B200 pairs. Here one pair is large and one small.

**Rule (LOG 11-21 17:13).** Both seeds are at 0.281 or more.
- If seed 2 is at 0.281 or more, the three-seed form is met, and a five-seed record attempt follows with r120's
  config as written, without no_save (a `record-<slug>` queue, `scripts/run_seed_v3.sh`, H200; its
  `config.json` committed first).
- Otherwise the three-seed mean is 0.276 or more if seed 2 is at least 0.2163; then r120's seeds 3 and 4 run
  for the five-seed form.

**Jobs (11:06).**
- r120 seed 2 runs on node4803 (30051396) since 06:27.
- 29995938 (seed 0) and 30017487 (seed 1) exited after their seeds, with less than 380 min left.
- r102 seeds 3 and 4 are still on `queue/gate_r120_h200.txt`; I submitted two preemptable H200 workers for them
  at 11:05 (30073610, 30073611), one per item. Both are pending.
- r116 seed 1 on a normal B200 (30069082).
- 4 GPUs.

## 2026-11-22 16:28 EST: r120 seed 2 meets the line, so D 008's record attempt starts; r116 meets its rule (seed 2 and the control queued)

Low-usage window (INBOX fee62d8), so this entry is short.

**r120 seed 2** (`calibration/research_r120/d007_speed_lr12e-6_h200`, node4803): final dev 0.3217 at step 42,987,
against r102 seed 2's 0.3000 (42,588): +0.0217 at the final, +0.0025 at matched train time (85 evals, ahead 45,
behind 40), +0.0047 at matched steps. Mean of the last five evals 0.3200 / 0.3070. Clock 20,798 s of 21,600
(pretraining 2,771 s).
- **Rule (LOG 11-21 17:13):** seeds 0-2 are 0.3100, 0.3017 and 0.3217, all at 0.281 or more, so the three-seed form
  is met (three-seed mean 0.3111). Over the three pairs r120 leads r102 at the final in two; on mean dev at matched time it is
  ahead in all three, by +0.036, +0.014 and +0.003.
- **D 008:** `records/v3/track_d/008_native_fused46_lr12e-6/config.json` is r120's config as written without
  `no_save` (committed in 7fb0ef3 before any seed). Queue `record-008_native_fused46_lr12e-6`, seeds 0-4, through
  run_item.sh's record branch (`STAGE=all scripts/run_seed_v3.sh`, 376 min a seed). Two preemptable H200 workers
  (30099011, 30099012; 24 h limit, so each can take up to three seeds; mit_normal_gpu's 6 h limit is too short for
  a record seed). Two more go in when r102 seeds 3 and 4 free their preemptable slots.

**r116 seed 1** (`calibration/research_r116/d116rl_native_fmkw2_b200`, node5700-c1): final dev 0.3333 at step 43,175,
against r101 seed 1's 0.2883 (44,862): +0.0450 at the final, +0.0285 at matched train time (ahead 75, behind 11),
+0.0345 at matched steps (ahead 77, behind 8). Last five evals 0.3380 / 0.2857. Steps per second 2.481 / 2.578.
Completion length is unchanged: 24.4 tokens early and 26.4 late, against 25.1 and 25.5 for r101 seed 1.
- **Rule (r116's entry):** seed 0 was +0.0367 and seed 1 is +0.0450 at the final (mean +0.0408), both +0.02 or more:
  seed 2, and the B200 control pretraining on `mix_0.2.bin` for seeds 0-2.
- **Control:** r84's `muon_1338m` already is that pretraining (the same settings as r116's arm, B200, mix_0.2.bin,
  1,337,982,976 tokens), with weights for seeds 0-4 on the pool. So it is not run again; the new RL config
  `calibration/research_r116/d116rl_native_r84ctl_b200` is r101's native control from those weights (6e21e76).
  Scratch holds symlinks to them, since the workers use `CKPT_DIR` in scratch.
- **Jobs:** `pre_r116` holds the arm's pretraining seed 2 and then its RL seed 2, for one normal B200 worker
  (30099099, `RESERVE_MIN=305`, so it runs both if the pretraining takes 39 min as before).
  `screens_rl_b200` holds control seeds 0-2 (30099100, one item per 6 h job).

**FineMath download (INBOX 5b83148).** CPU job 30099150 (mit_normal) fetches FineMath-4plus shards 3-31 at r116's
revision into scratch, with a manifest of sizes and md5 sums. Each shard is about 273 MB.

**GPUs at this entry.** Running: r102 seeds 3 and 4 (preemptable H200), r116 arm (normal B200). Pending: two D 008
workers (preemptable H200), the control (normal B200). Six in all, at the cap. The compile and MFU work of INBOX
5b83148 starts after the window (11-25 18:00).

## 2026-11-22 21:41 EST: r102 ends its five seeds at a dev mean of 0.2943; r116's control seed 0 ends 0.055 under r116; D 008 seeds 0 and 1 running

- **r102 seeds 3 and 4** (the H200 gate for D 007's speed stack): dev 0.2983 and 0.2933 (clock 20,787 and 20,790 s
  of 21,600). As before, run_seed_v3.sh's final mv logs FAILED for a no_save gate; result.json and clock.json are
  complete. All five seeds: 0.2700, 0.3100, 0.3000, 0.2983, 0.2933, mean 0.2943. All five are above the holder's dev
  mean 0.251, four of five reach 0.281 and none collapsed, so r102 meets the five-seed form too. It gets no record
  attempt of its own: D 008 runs r120 (lr 1.2e-5), whose seeds 0-2 averaged 0.3111 against r102's 0.2933 on the same
  seeds. r102 seeds 1-4's files are committed here.
- **r116's control, seed 0** (`d116rl_native_r84ctl_b200`: r101's native RL from r84 muon_1338m's weights, the same
  pretraining settings on mix_0.2 without FineMath; B200): dev 0.2850 at the end against 0.3400 for r116 seed 0,
  +0.0550. At matched train time +0.0543 (r116 ahead at all 96 paired evals); at matched steps +0.0435 (84 evals to
  step 42,000). r116 ran more steps (48,567 against 42,449), most of the gap early in the run (early sample time
  0.19 s against 0.33 s per step). The two ran on different nodes (node5700-c1 and node5502-c1), so part of that may
  be the node; the matched-steps figure doesn't depend on speed. r116 seed 2's RL is running; control seeds 1 and 2
  are queued.
- **FineMath download** (CPU job 30099150, 2 min 39 s): FineMath-4plus shards 3-31, 29 files and 8.31 GB, are in
  scratch with a manifest of sizes and md5s. They are not tokenized yet; that waits for the window to end.
- **Jobs at 21:41:** D 008 seed 0 runs on 30099011 (node4803, since 16:38) and seed 1 on 30099012 (node4802, since
  18:20). Two more record workers (30132623/25, pending) are for seeds 2-4. r116 seed 2's RL runs on 30099099
  (normal B200, since 17:05). The control's seed 1 worker 30132627 (normal B200) is pending. Six GPUs.

## 2026-11-23 02:53 EST: D 008 seed 0 ends at dev 0.2950, test 0.3188; three D 008 workers preempted; r116 seed 2 ends +0.0367 over r101's control; r116's control seed 1 ends 0.0767 under r116

- **D 008 seed 0** (30099011, node4803, H200): dev 0.2950, test (confirm_v3.json ladder_score) 0.3188. Clock 20,790 s
  of 21,600 (pretraining 2,770 s, GRPO 18,049 s). Seed 4 started on the same worker at 22:28.
- **Preemptions:** at 00:02-00:06 Slurm preempted the other three record workers: 30099012 (seed 1, after 5 h 46 min,
  near its end), 30132623 (seed 2) and 30132625 (seed 3), both after about 2 h. All three were requeued. 30132623
  started again at 02:07 on node4602; its worker moved seed 2's partial run to `extra/interrupted_2_1` and started
  seed 2 from the beginning. Seeds 1 and 3 restart the same way when 30099012 and 30132625 start. None of the partial
  runs reached a test score; the record's README will list them.
- **r116 seed 2** (FineMath-keyword pretraining, then r101's native RL; B200 30099099): dev 0.3200 at the end against
  0.2833 for r101's native control seed 2, +0.0367; +0.0285 at matched train time. Over seeds 0-2 r116 ends
  +0.0367, +0.0450 and +0.0367 over r101's control, mean +0.0395.
- **r116's control, seed 1** (r84's weights; 30132627, node5700-c1): dev 0.2567 against 0.3333 for r116 seed 1,
  +0.0767 for r116; +0.0388 at matched train time (r116 ahead at 76 of 87 paired evals, behind by 0.017 in the first
  hour). Over seeds 0-1 the control ends 0.0550 and 0.0767 under r116. Control seed 2 waits for 30165584 (normal
  B200, submitted at 02:53).
- **Window (INBOX c080338):** the low-usage window now pauses from 11-24 15:00 to the 11-25 box start for a full replan
  of the 11-25 box; `agent/slow_until` reads 2026-11-24 15:00. Once the box runs, the window resumes until 11-25 18:00.
- **Jobs at 02:53:** D 008 seed 4 runs on 30099011 (since 22:28) and seed 2 on 30132623 (since 02:07). 30099012 and
  30132625 (preemptable H200) are pending for seeds 1 and 3, 30165584 (normal B200) for control seed 2. Five GPUs.
  r116 seed 2's and control seed 1's logs are committed here; D 008's seed files wait for the record's commit.

## 2026-11-23 12:57 EST: D 008 seeds 2 and 4 end at test 0.3176 and 0.3371; seeds 1 and 3 preempted twice more; r116's control seed 2 ends 0.0183 under r116

- **D 008 seeds 2 and 4** (H200, preemptable): seed 2 (30132623, node4602) dev 0.3083, test 0.3176, clock 20,791 s;
  seed 4 (30099011, node4803) dev 0.3233, test 0.3371, clock 20,792 s. With seed 0, three of five seeds are done:
  dev mean 0.3089, test mean 0.3245 (the holder D 007: dev 0.2510, test 0.2644).
- **Seeds 1 and 3, more preemptions:** both restarted at 04:12 and 04:34 (partials of the midnight runs in
  `extra/interrupted_3_1` and `extra/interrupted_1_1`), were preempted again at 09:01 (seed 3 after 4 h 49 min,
  seed 1 after 4 h 27 min), restarted at 09:12 on node4601 (partials in `extra/interrupted_1_2` and `extra/interrupted_3_2`) and were
  preempted a third time at 09:18, six minutes in. 30099012 and 30132625 are pending again; each restart moves the
  partial to extra/. Each seed needs about 6 h 16 min, longer than mit_normal_gpu's 6 h limit, so they stay on
  preemptable.
- **r116's control, seed 2** (r84's weights; 30165584, node5702-c1): dev 0.3017 at the end against 0.3200 for r116
  seed 2, +0.0183 for r116; +0.0105 at matched train time (ahead at 67 of 89 paired evals), +0.0124 at matched steps
  (63 of 82). Over the three seeds the control ends 0.0550, 0.0767 and 0.0183 under r116, mean
  0.0500; r116 ends above r101's control by a mean of +0.0395. So the FineMath-keyword data still leads the
  same pretraining on mix_0.2 on all three seeds, by less on seed 2. The r116 work is done; a follow-up within the
  record's pretraining budget is for the 11-24 box replan. Control seed 2's logs are committed here.
- **Jobs at 12:57:** 30099012 and 30132625 (preemptable H200, D 008 seeds 1 and 3) pending. Two GPUs. The two normal
  B200 slots are idle: nothing queued fits them without a new config, which waits for the 11-24 15:00 replan.

## 2026-11-24 09:56 EST: the low-usage window is over; D 008 seed 1 is preempted a fourth time and seed 3 runs; U25's two-GPU check is submitted; r129 puts r116's FineMath mix into r120's 46-min pretraining, its rule written first

- **Window:** INBOX 11-24 ends the low-usage window, so the wakes are hourly again. It also asks me to fill the
  B200 queue with what is planned; r129 below is the first part of that.
- **D 008, seeds 1 and 3:** both were pending from the 09:18 preemption on 11-23 until early 11-24 (about 20 h).
  - Seed 3 restarted at 05:52 on node4800 (30132625); the 09:12 run's partial is in `extra/interrupted_3_3`. At
    09:55 it had run 4 h 03 min with 43 dev evals logged.
  - Seed 1 restarted at 05:34 on node4802 (30099012); the 09:12 run's partial is in `extra/interrupted_1_3`. It
    was preempted a fourth time at 06:47, after 1 h 13 min, and is pending again. Its partial will move to
    `extra/interrupted_1_4` when it restarts.
- **U25, the two-GPU check:** 30347362 (`mit_normal_gpu`, two B200s, 75 min; it replaced 30346665 before that
  started) runs `scripts/research/u25_ddp_check.sh calibration/research_r121/u25_ddp_check_2gpu`, which writes
  `check.json`. It is read against the rule in LOG 2026-11-22 04:03. In the same job,
  `scripts/research/u25_mem_smoke.sh calibration/research_r121/u25_mem_smoke_6x 78` runs U25's 6x size (1280x32)
  with DDP on the two GPUs for at most 6 min at micro-batch 64, then 32, with each process's CUDA memory capped
  at 78 GiB (`scripts/research/mem_cap_run.py`), to stand in for an 80 GB H100. It gives the peak memory and
  tokens/s for sizing the 6x rung on the 11-25 box. It is pending on Resources.

**r129: r116's FineMath mix inside r120's 46-min pretraining (B200, end to end).**

r116 (U24) ended above its own control on all three seeds (mean +0.0500) and above r101's control (mean
+0.0395). Its pretraining, though, is D 007's recipe stopped at 1,337,982,976 tokens on a B200, not the
record's timed 46-min run, so its gain is not yet a record candidate. r129 asks whether the gain survives inside
the record's recipe.
- **Arm:** `calibration/research_r129/d120_fmkw2_b200`. It is r120's gate config
  (`calibration/research_r120/d007_speed_lr12e-6_h200`) with one change: `pretrain.tokens` is r116's
  `mix_0.2_finemath_kw2.bin` on scratch (6,723,632,390 bytes; about 31 % FineMath, 14 % dm_math and 55 %
  FineWeb-Edu by tokens). Pretraining samples windows uniformly from the file, so the share is the same at any
  budget. The validation slice (the first 2M tokens) is mix_0.2's own start, the same as the control's.
- **Control:** `calibration/research_r129/d120_ctl_b200`, r120's config unchanged, on a B200.
- Both arms: 46 min of Muon pretraining on the fused path, then 18,000 s of native RL, `no_save`. Seeds 0 to 2,
  interleaved by seed in a new queue `r129_b200`, on two preemptable B200 workers, so each pair runs at the same
  time.
- **Expected:** a smaller gain than r116's, perhaps +0.01 to +0.03 at the final. Both arms see about the same
  number of tokens in 46 min, so the FineMath share replaces FineWeb-Edu and dm_math tokens instead of adding to
  them. r120's lr (1.2e-5) is also not the lr r116 was read at.
- **Rule, written before it runs:** take the paired final dev differences (arm minus control) for seeds 0 to 2.
  The arm meets the rule if the mean is at least +0.015 and at least 2 of the 3 are positive. Matched train time
  and matched steps are reported too, but they don't decide.
  - If it meets the rule, the next step is an H200 gate of the arm's config: three seeds against the track D
    promotion line in force then, and a record if that is met.
  - If not, r129 is closed, and the note is that r116's gain needs the longer pretraining.

## 2026-11-24 10:06 EST: an r129 worker lands on a node without scratch; CPU nodes reach the pool again; more keyword-filtered FineMath for U25's 6x rung

- **r129:** worker 30347230 started on node5800-c1 at 10:02 and failed both items within a minute: `mkdir:
  cannot create directory '/home/zjason/orcd/scratch': File exists`, so scratch was not mounted there. It is a
  host failure; `d120_fmkw2_b200` seed 0 is put back at the top of `r129_b200`, and one replacement worker,
  30348240, is submitted with node5800-c1 excluded too. The other worker, 30347231, runs `d120_ctl_b200` seed 0 on
  node5600-c1 since 10:05, so seed 0's pair will not run at the same time; the pair is still read as a pair.
- **Pool from CPU nodes:** probe 30347889 (`mit_normal`, 1 CPU) stat'ed `dm_math.bin` and read 1 MB of
  `mix_0.2.bin` on node3104 in 10 s, so the 11-21 note (CPU nodes don't reach the pool) no longer holds there.
  CPU preparation jobs can use the pool's venv again.
- **U25 6x data:** at 6x (1280x32, 668M parameters), r116's keyword-filtered FineMath (shards 0-2, 0.52B tokens)
  at weight 0.5 in the second half would repeat about 6 times. `scripts/research/prep_u25_finemath.sh` applies
  the same keyword rule (`scripts/keyword_filter.py drop`) to FineMath-4plus shards 3-31 of the same revision
  (already downloaded to scratch on 11-22), tokenizes them with the pinned tokenizer, and appends them to r116's
  `finemath_kw.bin` as `finemath_kw_0_31.bin` on scratch. My guess is about 5B more tokens, so the 6x rung's
  FineMath share repeats less than once. It is a `mit_normal` CPU job (16 CPUs); U25 is unranked.

## 2026-11-24 10:21 EST: U25's two-GPU check misses its speed rule (the rest holds); 6x fits at micro-batch 32; the 11-25 box replanned

**Two-GPU check (30347362, two normal B200s on node5600-c1, 10:03 to 10:18).** `check.json` in
`calibration/research_r121/u25_ddp_check_2gpu`, against the rule of LOG 11-22 04:03:
1. Ranks identical: met (every `rank_spread` 0.0).
2. Val loss: met. single 3.4047, ddp 3.4049.
3. Speed: **not met.** single 634,878 tokens a second, ddp 909,999, so 0.717 per GPU against the rule's 0.9.
4. Resume: met. The first part was stopped by the timeout and the rerun started at step 480. It ended at step
   1,000 with val loss 3.4079.

So the rule is not met. Rule 3 measures D 007's shape (110M) at 131,072 tokens a step. There a step's compute is
24 times smaller than at 6x and its gradient 6 times smaller, so the all_reduce's share is about 4 times larger.
The update does not overlap the all_reduce with the backward pass. I am still using DDP on the box: it is the only
way to train 668M on a day's box, and the run is bounded by its minutes cap, not by its token target. At 0.72 per
GPU, 480 min on eight H100s would give about 9.7B tokens (about 15 a parameter) instead of 10.7B. The box's own
`tok_per_s` rows give the scaling at 6x. Overlapping the all_reduce with the backward pass is a speed item for
later.

**6x memory smoke (same job, `calibration/research_r121/u25_mem_smoke_6x`).** 1280 x 32 (667.6M parameters),
torchrun on two GPUs with --ddp and --resume-every-minutes 1, each process capped at 78 GiB (an 80 GB H100 less
its margin):
- micro-batch 64 ran out of memory (peak allocated 77.78 GiB);
- micro-batch 32 ran: peak allocated 56.15 GiB, reserved 57.99 GiB. Its 20 steps took 0.62 min after the first,
  so about 266k tokens a second on two B200s, including warm-up.

The box rung therefore starts at micro-batch 32. A step is still 524,288 tokens.

**The 6x rung: `calibration/research_r130/u25_6x`.** It is unranked and outside the rules.
- Setup: D 007's architecture at 1280 x 32. Muon lr 4e-3, the fused loss, micro-batch 32, DDP over the box's
  eight H100s, a resume state every 30 min.
- Length: up to 10.7B tokens (16 a parameter) or 480 min, whichever comes first.
- Data: the 3x rung's two phases. The FineMath file is `finemath_kw_0_31.bin` (prep job 30348265, LOG 11-24
  10:06) instead of shards 0 to 2.
- Then the base weights get the 3x rung's probe. The probe's GSM8K default is now `~/orcd/pool/data/...`, so the
  same file is found on the box. The weights are copied to the bucket (`box/<box>/out/u25/`) before the box ends.

The 6x rule is the one written for 3x (LOG 11-21 17:59 and 11-22 03:52):
- **yes:** set B at least 10/20 working by eye, chain 0.5 and step accuracy 0.7;
- **partial:** 5/20 and chain 0.10;
- **no:** anything less. Then 12x goes to the 11-28 or 12-09 box.

**Code.**
- `scripts/research/u25_rung.sh` gets three options, all off by default (the 3x call behaves as before):
  - `NPROC` (torchrun with --ddp);
  - `RESUME_MIN` (resume state; a trained rung only probes; a log without a state is moved aside);
  - `LOCAL_DIR=none` (read the token files in place).
- `scripts/aws/u25_box_item.sh` is the box's runner for the rung:
  - it reads the shape from the config's `u25` section;
  - it tries up to 4 times, resuming each time, and halves the micro-batch on an out-of-memory failure that
    happens before any state is saved;
  - it copies the weights to S3;
  - it hands the screens' queue over under `queue/lock`;
  - it always exits 0.
- `tests/test_u25_box_item.py` covers it with a fake interpreter. Full suite: 225 passed; ruff clean.

**The 11-25 box (8x H100, 24 h from 06:30 EST): `agent/plan_box_1125.txt`.**
- **Line A:** `u25_1125` = the 6x rung on GPU 0's worker, using all eight GPUs.
- **Line B:** one worker per GPU on `box_1125`. It is staged empty, so these workers exit at boot. When the rung
  ends, `u25_box_item.sh` appends `box_1125_after` to it, and the keeper starts them within 10 min. RESERVE_MIN
  is 330, so no 5.3 h screen starts if the box's end would cut it.
- **`box_1125_after`, in order:**
  1. **Track A, U23's base:** r85's `a007_ent15_h100m30` and `a007_ctl_h100m30`, seeds 3 and 4. That makes five
     pairs of the entropy floor (r85: +0.0222 over 3 pairs, SE 0.0089), which r85 asked for before it goes into
     the next track A arm. The m30 dirs, because 0.35 ran out of memory on the H100.
  2. **r108:** ctl, sil256 and mini3, seeds 0 to 2, then hard2x seeds 0 to 2. That is 16 items, two rounds of
     eight.
  3. **Tail, if a third round fits:** lr12cos and g32p16 seeds 0 to 2, then sil256 seeds 3 and 4.
- Not on this box: U23's code arms (collapse guard, rollback, EMA clip) are not written yet, so they wait for
  the 11-28 box.
- Timeline: if the rung uses its full 480 min, the screens start about 20:30 UTC, and two rounds end before the
  box's deadline.

**Staging** is next. The FineMath prep job is still pending; the files list adds the dm_math, FineMath, tokenizer
and GSM8K paths to what the screens' configs need.

**r129 gets two more workers.** 30349981 and 30349982: preemptable B200s on `r129_b200`, with the same
submission as 30347231 (node5800-c1 excluded too). Its seed 1 pair now runs next to seed 0's. With D 008's two
H200 workers that makes six GPUs, the cap. Both are pending until preemption allows; the queue holds 4 items for
4 workers.

## 2026-11-24 11:07 EST: a faster grammar mask for the native sampler (`grammar_fast_path`, off by default); r131 will time it on a GPU

**Why.** At r128, the native sampler's mask CPU time was 104 ms of s_generate's 151 ms a step. The sampler waits on
the host, so less host work in the mask should mean less generate time.

**What changed** (`rlvr_speedrun/constrained.py`, `fast_rollout.py`, `grpo.py`). A new option `grammar_fast_path`
needs `constrain_answer` and the native backend. It does two things:
- **Interned grammar states.** Each (state, ops) pair gets an int id. A row steps its id one token at a time
  through a `(id, token) -> id` table, instead of re-running the grammar over its answer text. `generate` clears
  the tables when they hold more than 400,000 entries.
- **`MaskTable`.** A device bool table of blocked ids, 4096 slots by vocab, keyed by the allowed-id tuple. New sets
  are written in one batched host-to-device copy. The mask is one `masked_fill` with a gather of the rows' slots.
  On overflow it resets for the call.

The masks are identical to `mask_rows_cached`'s: the tests check the allowed ids against the old path, and the
mask tensors element by element. A rollout test checks the same sequences with all four (factored, fast) settings.
So no loss comparison is needed. Full suite: 228 passed; ruff clean.

**CPU benchmark** (`scripts/research/bench_mask.py`, job 30354739, a `meta` device, so it measures host work
only; medians over the second half of 8 sampling steps, ms a step):

| grammar | allowed_ids | mask |
|---|---|---|
| factored | 32.9 | 56.8 |
| factored, fast path | 31.5 | 25.1 |
| plain | 84.0 | 60.4 |
| plain, fast path | 86.6 | 34.2 |

- `MaskTable` saves 26 to 32 ms a step of host work.
- Interning gives nothing in this simulation, but its rows take random walks and rarely share states. Rows in
  a real group share more, so the GPU run is the test.

**r131, rule written first.** One GPU, r128's settings (r106's `final_native_ctl` config, D 007 seed 0's final
weights, the fine capture steps, part timers on). Four arms in ABBA order: off, on, on, off, 300 timed updates
each (`scripts/research/r131_mask_ab.sh`). The read is the median over updates 100 to 299 of s_mask_cpu,
s_generate and update wall time.
- If s_generate drops by at least 10 ms in both pairs (off_a vs on_a, off_b vs on_b), new configs turn
  `grammar_fast_path` on.
- Otherwise it stays off, and the LOG says how far it got.

It goes in when a GPU frees up. The 6 GPUs are all in use now: 4 r129 workers and D 008's 2 pending H200 seeds.

## 2026-11-24 11:40 EST: the 11-25 box's missing S3 files are uploaded; U25's FineMath prep moves to a preemptable CPU job

INBOX 11-24 11:50: most of the box's `home.txt` was not in S3.
- **Upload.** `box_s3.sh home` ran as job 30359132 on `mit_data_transfer` (dtn001), with an idle partition and
  pool access, in 2 minutes. S3 now has the Qwen3-1.7B-Base tar, `dm_math.bin.tar`, `mix_0.2.bin.tar` and `gsm8k`,
  next to what was there.
- **Check.** `box_s3.sh check` (with the stage's HOME_EXTRA) reports one problem: the U25 FineMath file.
- **FineMath prep.** The mit_normal prep job 30348265 had a planned start of 11-28, so I cancelled it. The same
  command runs as 30359131 on a `mit_preemptable` CPU node (node3300, 16 CPUs, no GPU, so not under the GPU cap),
  behind `agent/pool_guard_1124.sh`, which exits if the node cannot read the pool. When the file exists,
  `box_s3.sh home` runs again to upload it.
- **Fallback.** If the file is not ready by about 20:00 EST, the plan's FINEMATH points at r116's
  `finemath_kw.bin` (shards 0-2, which the 6x rung repeats about 6 times), and the box is restaged.
