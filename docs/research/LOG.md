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
