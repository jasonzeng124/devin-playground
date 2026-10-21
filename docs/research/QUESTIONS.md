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
