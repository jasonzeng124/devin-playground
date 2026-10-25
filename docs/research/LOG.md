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
