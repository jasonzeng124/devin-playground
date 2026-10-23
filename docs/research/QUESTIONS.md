# Open questions for the maintainer

## 2026-10-20 18:25 EDT: AWS for dev screens while Engaging is congested?

**Answered 2026-10-21 (laptop, INBOX): approved for a01_skip; seeds 0-2 run on AWS, seeds 3-4 stay
on Engaging.**

Since the maintenance ended (13:45), our workers have barely run. The four that started at 13:46
were preempted after 15-17 min; one restarted at 14:07 and was preempted after 6 min. Nothing has
run since. 2,652 jobs are pending on
mit_preemptable, and Slurm's estimate for our next start is ~01:55 tonight. The two
mit_normal_gpu slots can't start before 2026-10-23.

The AWS approval covers record attempts. Would you also approve dev-ladder screens there for the
strongest candidate, so its promotion decision doesn't wait on Engaging?

- Candidate: `calibration/research_r2/a01_skip` (A 001 + `skip_zero_advantage`). A 001's healthy
  seeds were still rising fast when its budget ended (seed 3: 0.188 -> 0.268 and seed 4: 0.248 ->
  0.300 over the last 50-80 min; LOG 2026-10-20 night). From A 001's step times, skip should fit
  1.6-2.1x more steps into the same budget (estimated; r2 measures it).
- Size: 3 seeds (the 3-seed promotion form) at about 5.3 h each on one H100 80GB HBM3, so about 16
  instance-hours; 5 seeds, about 27.
- Speed-up: on Engaging these 3 seeds are now first in `queue/main.txt` and should finish around
  12:30 on 10-21, if the workers start as estimated and nobody preempts them.
- If yes: say in INBOX which seeds AWS runs, or delete those lines from `queue/main.txt` under
  `flock queue/lock`. Otherwise Engaging would run them too, and I'd have two runs of one seed.

If not, everything stays on Engaging, which is fine.

## 2026-10-21 00:45 EDT: AWS for the track D promotion screen too?

**On hold (laptop, INBOX 2026-10-22): until AWS has shown it can get capacity. Meanwhile the D
candidate changed: fp32 lr 3e-6 alone ended at a 0.138 mean (no promotion); the open D screens
are fp32 + skip on D 000's and the D2 pretraining (LOG 2026-10-22 night).**

Engaging still runs nothing (last GPU minute 14:13 on 10-20). 3,071 jobs are pending on
mit_preemptable (2,925 at 22:25), and Slurm now gives our workers no start estimate at all.

The D candidate is fp32 master weights at lr 3e-6 on D 000's recipe. Its seed 0 scored 0.147
against D 000's 0.087 on the same pretrained seed. Promotion needs a 5-seed dev mean >= 0.149
with 4 of 5 seeds above 0.119, so seeds 1-4 decide it. On Engaging they wait behind the stalled
workers.

- Proposal: seeds 1-4 on AWS as dev-only screens of the full pipeline, with no new launcher code:
  `V3=1 STAGE=train RECORD=calibration/research_r2/d00_fp32_lr3e-6_full bash scripts/aws/spot_launch.sh 1 2 3 4`.
  This config (committed) is D 000's record config with `lr` 3e-6 and `master_fp32`, so each
  seed pretrains as D 000 did (112 min) and then runs the fp32 RL (234 min): the record recipe
  minus the test ladder. About 6 h per seed plus D data prep on each instance, so roughly 25
  instance-hours.
- The 5-seed check would count r1's seed 0 (the same RL, from D 000's seed-0 pretraining) with
  these four.
- If yes, say so in INBOX. I then take d00_fp32_lr3e-6 seeds 1-4 out of the Engaging queues
  (seeds 3-4 from `r1`; seeds 1-2 are in `queue/running/`), so nothing runs twice. If Engaging has
  started some of them by then, STATUS.md says which, and AWS can skip those.
- Related: if a01_skip's AWS seeds 0-2 miss the 3-seed form but stay close, the 5-seed form needs
  seeds 3-4. They are near the front of `main` but wait on the same workers. May they also go to
  AWS (about 11 instance-hours)?

If not, everything stays on Engaging.

## 2026-10-22 23:55 EDT: what counts as a "collapsed" seed for promotion?

The promotion rule (PLAN.md) allows at most 1 collapsed seed of 5, and none in the 3-seed form.
`summarize_screens.py` labels a seed one-guess when fewer than 10 of 64 groups are informative.
Read literally, that label blocks every track D arm, because D 000's own seeds are all one-guess.
It also blocks strong track A seeds: `a01_fp32_lr1e-6_skip_cos` seed 0 ended at dev 0.300 with
237-token searches but only ~9 informative groups, because it solves most easy prompts 8 of 8 times.

Clarification applied from now on (PLAN.md, written before seeds 1-2 of that arm or any r4 seed came
in): collapsed = diverged, or one-guess with a final dev below the holder's dev mean. If you want
the literal reading instead, say so in INBOX and I will apply it.

## 2026-10-23 07:00 EDT: is a constrained seed "diverged" when its searches run out of tokens?

`summarize_screens.py` labels a seed diverged when more than half its outputs are malformed, in
training or in the last dev eval ("the policy emits garbage"). Under answer-constrained decoding
(every track A arm since A 001), every answer is well-formed by construction. So "malformed" there
only counts completions that hit the 1,024-token limit before closing the answer, and the dev
score already counts those as failures.

Arms that reward a truncated completion 0 instead of -0.1 (`malformed_reward: 0.0`) learn long
searches. `c90_mal0` ended its 90 min at 0.322 and 0.383 with 62 % and 50 % of greedy dev outputs
truncated, so both seeds are labeled diverged. r5 (`a01_fp32_lr1e-6_skip_cos_mal0`) runs the same
recipe for 290 min; seed 0 was at 0.380 after 117 min with 59 % truncated. Read literally, the label
blocks any such arm from promotion however high it scores.

Reading applied from now on (PLAN.md, written before any r5 seed ended): on constrained arms, a
diverged seed counts as collapsed only when its final dev is below the holder's dev mean, as for
one-guess seeds since 10-22. The record's sampler may then add a think budget (`--think-budget`,
which forces the answer open after N tokens, so nothing truncates), chosen on dev as RULES_v3
requires, and the test ladder decides as for any record. If you want the literal reading, say so in
INBOX: r5 would then not promote, and I would screen penalties that keep truncation under half.

## 2026-10-23 19:35 EDT: should promotion compare dev under the record sampler instead of greedy?

The promotion rule compares in-loop final dev, which is greedy decoding on the 600 dev puzzles. A
record is scored with its own sampler (16 samples at T 0.5, constrained here), and that gains more
over greedy on arms with long searches:

| weights | greedy in-loop | T 0.5, 16 samples | gain |
|---|---|---|---|
| A 001 record seeds (mean of 5) | 0.240 | 0.255 | +0.015 |
| r7 `a01_fp32_lr1e-6_skip_cos_mal0_s300` seed 1 | 0.348 | 0.375 | +0.027 |
| r6 `a01_fp32_lr1e-6_skip_cos_tb768` seed 0 | 0.318 (with r6's think budget 768) | 0.361 | +0.043 |

So the rule may hold back arms that would do well as records, and it can't rank them against arms
that didn't save weights (skip + cosine didn't). Applied: nothing. The rule stays on greedy
in-loop dev, as for every arm so far. If you prefer dev under the record sampler, say so in INBOX.
Arms would then save weights and get a dev eval at T 0.5. The holder line would be A 001's 0.255 +
0.03 = 0.285, and A 002's line would come from its record seeds' evals.
