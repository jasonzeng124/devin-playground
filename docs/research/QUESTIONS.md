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
