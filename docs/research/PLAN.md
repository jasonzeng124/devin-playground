# Autoresearch plan (2026-10-19 to 2026-12-19)

The maintainer froze the v3 rules on 2026-10-19 (`docs/RULES_v3.md` at 13dbe32, with external
data allowed) and asked for a month of autonomous research: "just autoresearch for a month and
see what you get". This file is the plan and the operating rules; `LOG.md` is the dated lab
notebook. Both live on branch `research/2026-10`. On 2026-11-10 the maintainer extended the run
by a month, to 2026-12-19 (INBOX 2026-11-10), and the agenda at the end was revised that day.

**Goal (maintainer, 2026-10-19): log the best records under the current rules, and run a record
attempt for any substantial improvement.** Screens exist to find record candidates; the run is
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
- The maintainer's own ideas (INBOX 2026-11-10 14:58 and 15:04) are judged in
  `~/rlvr-ideas/STATUS.md`, outside this checkout. The list is private, so neither it nor the
  assessments are copied here. Read that file at each full wake together with this one, and
  update it when a verdict or a result changes. An arm built on one of the ideas names its id in
  LOG.

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
A 004 (`004_ladder_constrained_h200`) landed on 2026-10-30: in-loop final dev 0.287 / 0.230 / 0.310 /
0.263 / 0.185, mean 0.255, so **the A line is now 0.285** (holder dev mean 0.255).
D 005 (`005_dm_math_fp32_shape05_lr6e-6`, D 004 + RL lr 6e-6 + partial credit 0.5) landed on 2026-11-07
and beat D 004 (Welch p = 0.0034): in-loop final dev 0.193 / 0.187 / 0.210 / 0.217 / 0.200, mean 0.201,
so **the D line is now 0.231** (holder dev mean 0.201).

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

Revised on 2026-11-10 for the extension to 2026-12-19. Each original item is marked with how it
ended (LOG.md has the entries). New items start at A6 and D6.

Track A (start: Qwen3-1.7B-Base, 5 h; no external data since v3.3):
- A1. fp32 master weights + lr sweep. **Done:** fp32 at lr 1e-6 is the base of A 005.
- A2. Stability on the best fp32 lr: lr decay over the budget, KL to the base model, an entropy
  floor, dynamic sampling (oversample), tighter grad clipping. **Mostly done:** cosine decay and
  malformed reward 0 are in A 005; the KL and clipping arms are in the LOG (r13, r23); the entropy
  floor is A6; oversample is r51 (running).
- A3. Throughput: where the step time goes (t_sample vs t_update), batch shapes, vLLM settings.
  **Open, low priority:** A 005 stops at step 400 with about half of its 17,400 s unused, so
  throughput does not bind until A7 makes it bind.
- A4. External data: a short SFT on a public math-reasoning set before RL. **Not allowed** since
  v3.3 (track A may use no external data).
- A5. The budget question: time-to-target curves of the best recipe (would 1-2 h rank the same?).
  **Open, low priority.**
- A6. **Record A 006 by cutting the spread.** A 005 (test 0.3400 ± 0.0815) missed the record at
  Welch p = 0.072 because of two low seeds. With the entropy floor (coef 0.05 below entropy 0.1)
  ten box seeds had the same mean and a third of the SD (r40: 0.377, SD 0.015; control r39: 0.366,
  SD 0.046). Steps: r54, the H200 promotion screen of r40's config (3-seed rule), then record
  A 006 with 5 fixed seeds. What the test needs, with the validator's Welch test against A 004's
  0.2671 ± 0.0565: a test mean of 0.34 passes at SD 0.04 (p = 0.025) and 0.33 passes at SD 0.04
  (p = 0.040), but 0.32 does not (p = 0.065).
- A7. **Use the unused half of the budget, under the floor.** First read r40's per-level dev to see
  which levels hold the score near 0.38-0.40. Higher lr is not the way: lr 2e-6 and 3e-6 under the
  floor reached the same ceiling sooner and collapsed more often (r22-r24, closed). Candidates,
  each screened on H200 against r54's seeds:
  - (a) a longer schedule, for example cosine over 875 steps, stopped at 700. At 500 steps the
    floor "stops slides but doesn't raise the level" (r21), and nothing longer has run under it:
    the 800-step arms planned for r15 were dropped when the box was turned off.
  - (b) `oversample: 2` (r51) with the floor, if r51 shows a gain.
  - (c) more prompts per step (r20 tried 96 without the floor).

  Whatever clears the promotion rule against A 006's line becomes A 007.
- A8. **Checkpoint choice and rollback** (allowed since RULES v3.4, 2026-11-10). A run may keep
  the best of its own checkpoints, or roll back and train on, if its own code decides before the
  clock stops, from training signals and in-loop dev evals only, with snapshots, restores and the
  dev evals it acts on counted in the budget. The size, on A 005's own logs: final in-loop dev
  0.313 (SD 0.087), best of each seed's eight evals 0.364 (SD 0.027). The two seeds that slid
  (0.197 and 0.248) were at 0.335 and 0.365 at steps 250 and 350. The eight evals took 117 s. The
  best of eight noisy evals overstates dev by about 0.01-0.02, so read the low seeds and the test
  score, not the selected dev mean alone.
  - (a) Done 11-10 (`keep_best_dev`, LOG 11-10 15:46 EST): an option, off by default, that keeps the
    best in-loop dev checkpoint. Every in-loop dev eval counts toward the clock, the weights are
    copied when an eval is at least as good as the best so far, and the kept copy is the final.
    `clock.json` then takes no eval time off, and `validate_record_v3.py` checks that. The
    record's README says it chose a checkpoint.
  - (b) Queued 11-10: r58, r54's config with (a) on, is the promotion screen for A 006 (r54 had
    not run). The training path is r54's, so one eval log gives both readings (the last eval and
    the kept one). The kept dev is read against 0.305, not 0.285: the best of eight evals reads
    about 0.01-0.02 high.
  - (c) Rollback and train on (maintainer idea U12): restore the best snapshot when dev falls or
    the malformed share jumps, lower the lr and continue. For the 700-step schedule (A7 (a)) on
    the 11-18 box, where a slide at step 450 leaves time to use.
  - Track D runs don't slide, so no track D arm is planned on it.

Track D (from scratch, 6 h):
- D1. fp32 master weights + lr sweep. **Done** (in D 002 to D 005).
- D2. External data for pretraining: math mixes (FineMath, OpenWebMath, DeepMind
  `mathematics_dataset` arithmetic). **Closed:** the dm_math mix at 0.2 is in D 004 and D 005, and
  the other mixes were no better.
- D3. SFT on public worked solutions (GSM8K-style chains of arithmetic) before RL. **Closed:** no
  gain (r32).
- D4. Process rewards for correct intermediate arithmetic. **Closed:** +0.008 over five seeds
  (r49), inside the noise.
- D5. Model size. **Nearly closed:** widths 640 and 1024 ended level with 110M (r50); width 1280 is
  +0.004 paired over four seeds (r52, closed 11-10: seed 4 was preempted and is not rerun); 24
  layers is r53 (queued, last in the H200 line).
- RL knobs (not on the original list): lr 1e-5, temperature 1.2 and 1.4, an entropy floor, groups
  of 16 and the pretrain/RL split all ended at 0.198-0.222 against the 0.231 line. No more
  RL-knob-only screens (INBOX 2026-10-31).

  The reading on 2026-11-10: the limit is sparse reward on the levels that need * or /, not model
  size or data. A policy that guesses once is right on 3-13 % of L2-L5 prompts, so most groups give
  no gradient. Dev is still rising at the end of the budget (about +0.03 from step 6,000 to 12,000).
- D6. **Mini-ladder curriculum.** Training-only 3-number levels with the operator rules of L2, L3
  and L6 (`ladder.MINI_LEVELS`), as a third of the training mix (r55 on H200 from D 005's checkpoints;
  r56 step-capped on B200 from D 004's checkpoints, paired with r30). If it helps: vary the
  share, fade the mini levels out over the run, and try 2-number or other intermediate levels.
- D7. **More steps per budget.** Sampling 512 short completions through vLLM is about 70 % of a
  step (0.67 of 0.96 s); the update is 0.25 s. First profile the sampling path (tokens per
  completion, the answer constraint's cost, the weight sync), then try the cheapest fix that
  profile points to. Worth building only if it gives 1.5 times the steps or more. With cheap
  sampling, `oversample` also becomes affordable in track D, which would fill each update with
  informative groups (now 26-29 of 64).
- D8. **Level weights in the training mix.** For example no L5 prompts (dev 0.03, the longest
  completions), or more weight on the levels that give informative groups. r45 (no L1) was -0.020
  over three seeds, so the easy levels stay in. r57 is the mix without L5, step-capped on B200 and
  paired with r30. Weights need no code: a level named twice in `train_ladder_levels` is drawn
  twice as often.
- D9. **dm_math during RL** (lower priority): an interleaved LM loss on the published dm_math text
  during RL, so the arithmetic learned in pretraining does not drift.
- D10. **Stack and promote.** Combine what holds up (the mini-ladder, level weights, more steps,
  self-imitation; not width, r52) into one arm, run the 5-seed promotion screen on H200 against
  0.231, and run record D 006 if it passes.
- D11. **Self-imitation on the policy's own verified samples** (allowed, INBOX 2026-11-10 14:50).
  Completions the verifier accepted during this run's RL go into a buffer with their original
  generator puzzle, and a likelihood loss on samples from the buffer is added to the RL loss. The
  limits: the buffer starts empty in each seed's timed run and is never saved or carried between
  runs or seeds; completions are stored verbatim (no relabelling, no edits or trims); the
  verifier's verdict is the only filter; sampling done to fill the buffer counts toward the
  budget. Coded 11-10 (`sil_rows`, `sil_coef`, `sil_buffer`, `sil_by_level`; off by default; LOG
  2026-11-10 16:16 EST). First arm r59: r30's config with 64 replayed rows per step at
  coefficient 0.5, drawn evenly over levels, step-capped on B200 from D 004's checkpoints, paired
  with r30. A clear gain: seeds 3-4, a coefficient and row-count sweep on the 11-18 box, and a
  time-budgeted H200 gate (the step cap hides the replay's update time). Track A may use it too. If an arm built on it meets the promotion rule, say so at the
  top of STATUS before the record starts.
- D12. **Hindsight relabelling: one unranked screen, to measure its effect** (INBOX 2026-11-10
  14:54). It stays not allowed in records, promotion screens and anything stacked into a ranked
  arm. One arm on track D, three seeds, dev ladder only, paired against D 005, marked unranked in
  its config and in LOG, never promoted or stacked. The relabelled target is the value of the
  model's own expression on the puzzle's own numbers. No solver and no search over expressions on
  relabelled pairs (no `reachable` or "needs" filter). Off by default in code. Report the paired
  difference, the per-level numbers and the share of samples that gave a usable relabelled pair.
  It does not outrank r54, r55 or other ranked work; the 11-18 box has room.

Box days (INBOX 2026-11-10 10:40: both remaining AWS blocks are back on, screens only, about
50/50 between the tracks, launched by hand; AGENT.md "AWS boxes"). Box runs use step caps, and
each batch carries its own control arm:
- 11-18 06:30 EST to 11-19 06:30 EST, 8x H100. Track A (4 GPUs, one fp32 run per GPU): A7 (a),
  the 700-step schedule under the floor, with and without rollback (A8 (c)), about 12 runs. Track D (4 GPUs, packed): RL-only arms
  from D 005's checkpoints for D6's follow-ups and D8. The arms are fixed by 11-16 from r54, r55
  and what follows them. Queue files, configs and the copy list are committed by 11-16, because
  Engaging's maintenance is 11-17 00:00-21:00 EST.
- 11-28 06:30 EST to 11-29 06:30 EST, 8x H200 (the maintainer may still take this block). The D10
  stack and track A's best A7 arm, with more seeds than Engaging gives in a day. The arms are
  fixed by 11-26.

Rough timeline for the extension:
- 11-10 to 11-14: code in this order: the checkpoint option (A8, done), self-imitation (D11,
  done), the unranked relabelling screen (D12). r58, then A 006's five record seeds if the rule is
  met. r55, r53. On B200: r51, r56, r59 (the first self-imitation arm, D11), r57.
  The D7 sampling profile.
- 11-15 to 11-16: prepare the 11-18 box (above). 11-17: Engaging maintenance.
- 11-18 to 11-25: the H100 box day, then its results. D6 follow-ups, D7 and D8 on Engaging.
- 11-26 to 12-06: the H200 box day (11-28). The D10 stack and its promotion screen on Engaging.
  The A7 promotion screen.
- 12-07 to 12-16: record runs for whatever qualifies, and last screens.
- 12-17 to 12-19: no new arms. The final LOG summary, then the loop stops.
