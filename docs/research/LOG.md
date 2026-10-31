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
