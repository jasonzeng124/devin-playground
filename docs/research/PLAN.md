# Autoresearch plan (2026-10-19 to 2026-11-19)

The maintainer froze the v3 rules on 2026-10-19 (`docs/RULES_v3.md` at 13dbe32, with external
data allowed) and asked for a month of autonomous research: "just autoresearch for a month and
see what you get". This file is the plan and the operating rules; `LOG.md` is the dated lab
notebook. Both live on branch `research/2026-10`.

**Goal (maintainer, 2026-10-19): log the best records under the current rules, and run a record
attempt for any substantial improvement.** Screens exist to find record candidates; the month is
judged by records logged on the leaderboard, in either track.

## Operating rules

- Compute: MIT Engaging only, at most **6 GPUs at once** (mit_normal_gpu allows 2 per user;
  the rest on mit_preemptable). AWS only for the capacity-block boxes in AGENT.md (screens only). Heavy work goes through Slurm, never the login node.
- Code: research runs from the checkout `~/rlvr-screens` (this branch). `~/rlvr` stays as it
  was when the records ran (its code matches f8d9ade); don't touch it.
- Screens are dev-ladder only (`scripts/slurm/screen_seed.sh`, queue files in
  `~/rlvr-screens/queue/`, workers from `scripts/slurm/queue_worker.sbatch` with
  `RLVR_DIR=~/rlvr-screens RUNNER=scripts/slurm/screen_seed.sh`). The test ladder is used only by
  record runs (below).
- Workers: `RESERVE_MIN=330` so a worker never starts a 5 h item it can't finish;
  `--time-min=05:30:00` so jobs can start before a maintenance reservation (monthly, e.g.
  2026-10-20 00:00-21:00). Since 2026-10-25 (v3.1) the reference GPU is `NVIDIA H200`
  (H200 SXM). Screens may use any GPU type MIT has (H200, B200, H100, ...), but compare arms
  only within one GPU type, or use step caps, since a faster GPU fits more steps into a budget.
  The 5-seed promotion screen that gates a record runs on `NVIDIA H200` (or is step-capped).
  The H200 NVL nodes (4-GPU, preemptable only: node4002, node4009, node4010, node5102, node5105,
  node5201) are not the reference GPU.
- Keep at least a day of queued work so a lapse in the ssh connection (Duo) costs nothing.
- RL screens set `"no_save": true` unless their weights are needed (probes, later stages):
  fp32 track A weights are ~7 GB each.
- Every finished batch gets a LOG entry with the `scripts/summarize_screens.py` table and what
  it means, and the branch is pushed. Commits use `zjason <zjason@mit.edu>`.

## Reference points (dev ladder, final in-loop eval)

| | dev final | collapse modes (ok / one-guess / diverged) | takeoff (first dev >= 0.10) |
|---|---|---|---|
| A 000 (bf16, lr 2e-6) | 0.129 ± 0.011 | 5/0/0 | ~220 min |
| A 001 (bf16, lr 4e-6, ladder mix, constraint) | 0.240 ± 0.045 | 4/1/0 | ~150 min |
| D 000 (bf16, lr 3e-5) | 0.119 ± 0.023 | 0/5/0 (track D policies are one-guess) | ~85 min |

## Record promotion rule

An arm becomes a record attempt when its screen seeds show **mean dev >= the holder's dev mean
+ 0.03, at least 4 of 5 seeds above the holder's dev mean, and at most 1 collapsed seed**
(holder dev means: A 0.240, D 0.160 since D 002 landed on 2026-10-24). A 3-seed screen with every seed at least +0.03 above the
holder's dev mean and none collapsed also qualifies (don't wait for seeds 4-5). A record run
counts as a screen too: its seeds' dev logs feed the next decision. After a new record lands, it
becomes the holder for later attempts. Then: a record folder `records/v3/track_{a,d}/<NNN>_<slug>`
on this branch, N = 5 fixed in advance, `scripts/run_seed_v3.sh` (STAGE=all) on H200 SXMs
(`QUEUE=record-<slug>`, `--gres=gpu:h200:1`, `REQUIRE_GPU="NVIDIA H200"`, mit_normal_gpu), all 5 seeds
reported, `validate_record_v3.py --compare-to <holder>`. No PR without the maintainer.

H200 lines (v3.1, INBOX 2026-10-25): the first H200 holders will be the reruns
`records/v3/track_a/004_ladder_constrained_h200` (A 001's config) and
`records/v3/track_d/003_dm_math_fp32_h200` (D 002's config). Until they land, promotion uses the
H100 lines: A 0.2703 (A 001's record dev 0.2403 + 0.03) and D 0.190 (D 002's 0.1603 + 0.03). When a
rerun lands, its five seeds' in-loop final dev replaces the H100 number here, with the date.
D 003 failed (two seeds over budget); its rerun `004_dm_math_fp32_h200_warm` landed on 2026-10-30:
in-loop final dev 0.182 / 0.130 / 0.137 / 0.183 / 0.163, mean 0.1590, so **the D line is now 0.189**
(holder dev mean 0.159).

"Collapsed" in this rule (clarified 2026-10-22, before the seeds it first applies to; QUESTIONS.md
asks the maintainer): a seed that diverged, or a one-guess seed whose final dev is below the
holder's dev mean. `summarize_screens.py` labels a seed one-guess when fewer than 10 of 64 groups
are informative. That label marks every track D seed, the holder's included, and strong policies
that solve most training prompts 8 of 8 times, such as `a01_fp32_lr1e-6_skip_cos` seed 0 (dev 0.300,
237-token searches at the end).

"Diverged" on answer-constrained arms (clarified 2026-10-23 07:00, before any `research_r5` seed
ended; QUESTIONS.md asks the maintainer). The label means "more than half the outputs malformed",
meant as "the policy emits garbage". Under `constrain_answer` every answer is well-formed by
construction, so "malformed" there counts only completions that hit the 1,024-token limit before
closing the answer, and the dev score already counts those as failures. On constrained arms a
diverged seed therefore counts as collapsed only when its final dev is below the holder's dev
mean, as for one-guess seeds.

## Agenda

Track A (start: Qwen3-1.7B-Base, 5 h):
- A1. fp32 master weights + lr sweep (r1, running): which lr keeps fp32's fast takeoff without
  diverging or collapsing?
- A2. Stability on the best fp32 lr: lr decay over the budget, KL to the base model, an entropy
  floor, dynamic sampling (oversample), tighter grad clipping.
- A3. Throughput: where the step time goes (t_sample vs t_update), batch shapes, vLLM settings.
- A4. External data: a short SFT on a public math-reasoning set before RL.
- A5. The budget question: time-to-target curves of the best recipe (would 1-2 h rank the same?).

Track D (from scratch, 6 h):
- D1. fp32 master weights + lr sweep (r1, running), paired on D 000's pretrained seeds.
- D2. External data for pretraining: math mixes (FineMath, OpenWebMath, DeepMind
  `mathematics_dataset` arithmetic). Screen the pretrained model first (dev pass@16 against the
  random-guess 0.195, and whether it writes arithmetic) and only RL candidates that beat it.
- D3. SFT on public worked solutions (GSM8K-style chains of arithmetic) before RL.
- D4. Process rewards for correct intermediate arithmetic (allowed since launch).
- D5. Model size once the data gives a reason to write steps.
