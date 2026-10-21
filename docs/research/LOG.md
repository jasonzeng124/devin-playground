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
