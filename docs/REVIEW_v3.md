# Benchmark v3: what to review

This is the reviewer's guide to the v3 work. It lists every decision taken on your behalf, the
current approach for each track, and what failed and why. Details and raw numbers are in
`docs/v3_design.md` (a chronological lab notebook). The spec itself is `docs/RULES_v3.md`.

Suggested reading order (~30 min):
1. `docs/RULES_v3.md`: the rules (about 225 lines).
2. "Decisions" below: each one with the alternative and the evidence.
3. "Failure cases" below.
4. Optional: `rlvr_speedrun/constrained.py` (the answer grammar that every track D result depends on)
   and `rlvr_speedrun/ladder_confirm.py` (the scorer).

All scores are level-weighted ladder pass@1 (mean over 6 levels of the per-level solve rate).
"Dev" is the in-loop eval on `data/ladder_v3_dev.jsonl` (greedy, 100 per level) unless noted.
"Test" is `ladder_confirm` on `data/ladder_v3_test.jsonl` (2,400 puzzles x 16 samples, T=1).

## Decisions taken for you (please confirm or overturn)

| # | decision | alternative | why (evidence) |
|---|---|---|---|
| 1 | Fixed compute budget, score = ladder pass@1 (v2 was time to a threshold) | keep time-to-threshold | One threshold can't fit both a 1.7B model and a scratch model, and a ladder shows where gains come from |
| 2 | Ladder: 6 levels x 400, equal weights; separate frozen dev and test sets | single eval set | Dev predicts test closely (d07 finals: dev 0.141/0.122/0.122 vs test 0.140/0.121/0.125), so tuning on dev is safe |
| 3 | Two tracks: A = Qwen3-1.7B-Base (pinned), 5 h; D = from scratch on FineWeb-Edu, 6 h incl. pretraining | more or other tracks | Your choice (verifier-only for both) |
| 4 | Training on solutions is banned (generator `solution` field, solver output, other models' text) | allow a capped SFT warm start | 1,000 SFT steps on generator solutions lift D from ~0.14 to 0.24, so D would turn into distillation |
| 5 | The eval sampler is part of the submission (**you confirmed 2026-10-07**; custom samplers get careful review and may be rejected) (temperature, answer-constrained decoding, think budget, or own code) but must not evaluate arithmetic, call the verifier or pick among samples | one fixed sampler for everyone | A D model trained with the constraint scores **0.000** without it (the constraint is part of its policy). The constraint alone solves nothing (bases 0.003-0.011). It adds +0.011 to a Qwen pilot trained without it and +0.011-0.016 to the track A records' healthy seeds, but +0.074 to A 001's collapsed seed (`calibration/diag_sampler`) |
| 6 | 16 samples per puzzle, at most 1,024 new tokens | 2,048 tokens | The best Qwen scores 0.157 at 2,048 vs 0.155 at 1,024: the cap doesn't bind |
| 7 | >= 5 seeds, one-sided Welch p < 0.05 vs the holder | 10 seeds (v2) | Record seed sds are 0.0098 (A 000) and 0.0163 (D 000), so N=5 has 80 % power for gains of ~0.017 (A) and ~0.028 (D). A 001's sd is 0.046 (one collapsed seed), so beating it takes ~+0.06-0.08. You said n=5 is fine |
| 8 | Track D tokenizer: any tokenizer trained on the allowed data, or the pinned TinyLlama one | pin one tokenizer | A GPT-2 tokenizer (it merges digits) halves the score (0.067 vs 0.135), so the choice is a real design lever, not a loophole |
| 9 | Track D baseline = 110M Llama, AdamW pretraining + RL (pilots 120 min + 4 h; record 112 + 234 min) | Muon 60 min + 5 h RL (pilots: 0.158 over 5 seeds vs 0.159 over 3) | Plain AdamW keeps the baseline simple. Muon was then tried on the H100 as D 001 and did not beat it (0.1162 vs 0.1166, p = 0.52) |
| 10 | Clock = GPU wall clock on one H100 SXM; CPU data prep untimed but scripted; in-loop dev evals may be excluded | time everything | Matches v2; tokenization is CPU-bound and not the interesting part |
| 11 | Track D labelled **experimental**; you may revise its ladder at any time (new version number, old records keep their leaderboard). You set this on 2026-10-07; my draft said only if nothing passed 0.20 within 6 months | revise the ladder before launch (e.g. easier L5) | See "the track D plateau" below |
| 12 | In track D each seed pretrains its own model (the pilots shared one pretrained model across RL seeds) | allow sharing | Otherwise the seed spread ignores pretraining noise, and a record could pick its best pretrained model |
| 13 | Baseline budgets leave room for overhead: D = 112 min pretraining + 234 min RL, A = 290 min RL; `clock.json` checks the real total (start-up, loading, vLLM init and saving all count) | 120 + 240 / 300 nominal | Overhead counts under the rules, so nominal splits would run over. Measured: A 291 of 300 min, D 348 of 360; the overhead was 1-2 min, so ~9-12 min went unused |
| 14 | Track D data is pinned: two FineWeb-Edu `sample/10BT` shards (1.72B TinyLlama tokens) at a fixed dataset revision, and the TinyLlama tokenizer at a fixed commit (`benchmark.V3_TRACK_D_*`); `scripts/prepare_track_d_data.py` rebuilds them (untimed CPU step). Track A's Qwen revision is pinned too and is recorded per seed (`model_revision` in result.json) | "any FineWeb-Edu subset" | The 112-min baseline samples 2.47-2.51B tokens with replacement from the 1.72B-token pool (~1.45 passes), a level of repetition that usually costs little, and records may use any FineWeb-Edu subset. Pinning makes the baseline reproducible bit for bit. Rebuilding on AWS gave the same 1.723B tokens as on Engaging |
| 15 | Test-ladder scoring is untimed and may run as a separate job on any GPU, but always in the reference software (vLLM 0.30.0 / torch 2.13 / transformers 5.17). Training software is recorded, not frozen (as in v2) | time the scoring, or freeze all software | Track D's 6 h clock plus scoring doesn't fit Engaging's 6 h job limit. Scoring is not what's being raced, and pinning the scorer keeps scores comparable. All 10 track D seeds were scored this way, on L40S nodes |
| 16 | A run that crashed for infrastructure reasons (OOM, preemption, a lost machine) may be rerun under the same seed if the failed attempt stays in `extra/` with a note | any rerun disqualifies | Spot instances and preemptible nodes make this routine. "All runs reported" still holds, since nothing is deleted |
| 17 | Track A baseline uses `micro_batch_tokens` 8192 instead of the pilots' 16384 | lower vLLM's memory share | The pilot setting used 99-108 GB on H200s and ran out of memory at step 2 on an 80 GB H100 (`records/v3/track_a/000_baseline/extra/oom_0`). Smaller micro-batches change memory and speed, not the update (the loss is normalised by the global token count) |
| 18 | Process rewards and value functions are allowed: any reward or training signal (a program or a learned critic/value/PRM) whose only inputs are the model's own outputs, the puzzle and the verifier's verdicts. Learned ones are trained on the run's own rollouts, within the budget. No solver, search, other pretrained model or external data. Reward code is reviewed like samplers. Training them counts toward the budget, and they may not be used at eval (you set both on 2026-10-07) | rewards only from the verifier's output (old wording) | You leaned this way on 2026-10-07 ("value functions should be allowed to exist"). It stays verifier-only: every signal traces back to the verifier and the model's own samples. Track D's policies do no arithmetic, so step-level signal is a real lever on the plateau. Reward hacks show up in the test score |
| 19 | Synthetic arithmetic text (equations, expression-value pairs) is explicitly banned from training data, in pretraining and RL | allow it, or a separate "D+" track that allows it | A random expression with its value is a puzzle with its solution, which is the banned SFT shortcut (0.14 to 0.24 on track D). This only makes the old implicit ban explicit |
| 20 | All seeds of a record train on one provider (added to RULES_v3 on 2026-10-14; before that it was only in CONTRIBUTING) | allow mixed providers | Host differences would mix into the seed spread: v1's record 004 had Modal seeds much faster than its RunPod seeds (655 vs 1088 s). A 000's Modal duplicate of seed 0 sits in `extra/` |

**2026-10-07: you approved the rules as a whole; edge cases are to be arbitrated later.** You have confirmed #5 (with careful review of custom samplers) and changed #11 (ladder revision at any time). Still open for your eyes: #15-17, #20, and "Open questions" at the end.

## Current approach

**Track A baseline** (`calibration/sweep_008` `q3_17b_lr2e-6`, record A 000): GRPO on generator
puzzles, lr 2e-6, r1 zero-shot prompt, 1,024 tokens, 5 h. H200 pilots: test 0.153 / 0.123 (2 seeds,
trained before the test ladder existed); 8 runs of the recipe family score 0.123-0.155 on dev. The
H100 record scores 0.1295 ± 0.0098 on test. It passes v2's capability guardrail easily (FineWeb
loss +0.003 nats, MMLU-lite -0.006). Track A has had little rules attention since you said it
"seems fine ish".

**Track A record 001** (ladder mix + answer-constrained decoding + lr 4e-6): test 0.2481 ± 0.0460,
p = 0.0019 against 000. One seed collapsed into ~40-token single guesses by minute 60 and still
scored 0.186. Its three changes were never separated (see its README). The capability probes
(on Engaging) put the collapsed seed's FineWeb loss at +0.0041 nats and its MMLU-lite at the
base model's, like the healthy seeds, so v2's guardrail can't detect this kind of collapse.

**Track D baseline** (`calibration/sweep_d10` `pre120`):
1. Pretrain a 110M Llama-style model (width 768, 12 layers) on FineWeb-Edu with AdamW for 120 min (112 in the record), using the TinyLlama tokenizer (`scripts/pretrain_scratch.py`).
2. Run Dr. GRPO (no std normalisation) at lr 3e-5, 64 prompts x 8 samples, 256 tokens, uniform over the ladder levels, no easy curriculum, no reward shaping.
3. Use **answer-constrained decoding**: after `<answer>`, only well-formed expressions over exactly
   the puzzle's numbers and operators can be sampled. The trainer renormalises its log-probs over
   the same allowed tokens.

H200 pilot dev scores: 0.175 / 0.167 / 0.137. The H100 record scores 0.1166 ± 0.0163 on test, and its
final in-loop dev evals average 0.119 (see "Infrastructure" below).

What the D policies actually do: a few fixed answer shapes, with the number placement learned per
puzzle. 4 shapes cover 75 % of correct 4-number answers (`N+N+N-N`, `N+N-(N-N)`,
`N*N+N+N`, `N+N-N-N`); with 5-6 numbers it is "one product, then add or subtract the rest". No
reasoning text, no search. L3 (needs /) and L5 (6 numbers, targets 100-999) stay near 0.

**How a record seed runs** (`scripts/run_seed_v3.sh <record> <seed>`, same on Slurm and AWS):
1. Track D only: pretraining under a wall-clock cap.
2. GRPO under `time_budget_s`.
3. `clock.json`, which records the GPU, each phase's wall time, the total and `within_budget`.
4. `ladder_confirm` on the test ladder with the record's own sampler (the `v3` block of its `config.json`).

On AWS, `scripts/aws/spot_launch.sh` with `V3=1` gives each seed its own self-terminating EC2 instance. The instance:
- builds a fresh env (vllm 0.30.0, transformers 5.17.0, torch 2.13.0+cu130);
- prepares the data;
- runs the seed;
- pushes everything to S3.

A smoke run on an A10G (2026-10-06, 3 min pretraining + 4 min RL) passed every step end to end. The
weights were too young to score, so it got 0.000. Its logs are not in the repo, and no H100 has
launched through this route (see "Why not AWS" below). Modal (`scripts/modal_v3.py`) has: it
produced A 000's `extra/modal_0`.

## Failure cases (what broke and what it implies)

Reward and training:
- **Partial-credit shaping was hacked twice.** First the model wrote the bare target, then a single number. Shaping is now off (shaping 0). Implication: verifier-only, binary reward.
- **Without the answer constraint, track D never learns.** 0 solves in d01-d03: the model never copied the puzzle's numbers into the answer.
- **Collapse to "add everything".** Under the generator mix (where ~7 % of targets are the plain sum), constrained D runs learned to sum all the numbers and stopped there. Fixed by training on the ladder mix, where no target is the plain sum.
- **lr too high.** At lr 6e-5 the policy peaked early and grew a junk prefix ("COVID at"); at 1e-4 it collapsed into free text that never reaches `<answer>`.
- **Clip-Cov (entropy-preserving clipping) blew up.** Entropy rose to ~8.5 and the completions became 1,000+ chars of junk; the ladder score went to 0.
- **No reasoning, even when it is never punished.** Without a think budget, >99.9 % of samples answer immediately. Cause: free text risks the 256-token cap and the -0.1 malformed penalty, while a constrained answer is always well-formed, so answering at once is always safe (you spotted this). A forced-answer think budget makes thinking safe, yet it scored 0.07 vs 0.13: the text before the answer is FineWeb-style filler and doesn't help, and the 120-min model learns to skip it. Left for a second record.
- **The track D plateau.** The best H200 pilots reached ~0.15-0.17 on dev, and nothing went past that:
  - pretraining 30 / 60 / 120 min;
  - AdamW vs Muon;
  - 110M vs 335M;
  - temperature 1.2, 32x16 groups, oversample 2, Clip-Cov;
  - a think budget (lower: 0.07-0.12).

  Lower pretraining loss stops helping below ~2.8 nats (Muon 120 min reached val loss 2.72 but scored 0.122 after RL). The signal-shortage idea doesn't explain it either. After ~20 min only 3-5 of 64 groups per step have mixed outcomes, but oversampling 4x, 8x or 16x to keep only informative groups (sweep_d14, one seed each) ended at 0.168 / 0.177 / 0.137, against a 5-seed mean of 0.158 without oversampling. The limit looks more like the model (fixed templates) than the RL signal, though in a fixed 6 h a 3x larger model did no better. These are H200 pilot numbers; the H100 records sit lower (see "Infrastructure").
- **Pure-bf16 optimizer (found 2026-10-14).** The trainer runs AdamW directly on bf16 weights, with no fp32 master copy, so an update only lands on weights smaller than about lr x 512 (2^-10 at lr 2e-6). After 5 h, 2.2 % (A 000) and 4.4 % (A 001) of the weights differ from the base model. The learning rate therefore also sets how many weights can move at all, which may be part of why A 001's lr 4e-6 mattered. Not changed, since it would change every track A result; an opt-in fp32 master copy is the obvious next experiment.
- **Sampler dependence.** D models score 0 without their training-time constraint. This is why decision #5 exists. It also means the reference constraint code (`constrained.py`) is effectively part of the benchmark for D.

Infrastructure (affects how the baselines must be redone):
- **The pilots ran on H200s, not H100s.** Every D pretraining and most RL pilots ran on Engaging H200s, but the rules time an H100, so the baselines were rerun on node2906's H100. Track D came out lower: final in-loop dev evals average 0.119, against 0.159 for the pre120 pilots (one-sided Welch p = 0.025). Speed doesn't explain it: RL ran 47-58 steps/min on the H100 against 52-55 on the H200, and pretraining 359-373k tokens/s against ~383k. Each pilot arm shared one pretrained model across its seeds, so one lucky pretraining run is a candidate. The pilots' sbatch files on Engaging were not rechecked.
- **Preemption and NFS.** Pilot runs on mit_preemptable were preempted (resumed or rerun) and one died on a stale NFS handle (rerun). None of that affects the reported numbers, which come from completed runs only.
- **The network filesystem slowed track D pretraining ~9x on the H100 node.** Each step does 512 random reads of the token file (16 micro-batches of 32 windows), and cold reads over the network disk took 5.4 s per 256. Copying the file to local disk first (63 s, untimed preparation; `LOCAL_DATA_DIR` in `run_seed_v3.sh`) restored 359-373k tokens/s, close to the H200 pilots' ~383k. Seeds that train from a network disk would be handicapped without anyone noticing.
- **The GPT-2 tokenization job ran out of memory.** It stopped after a 1.09B-token prefix, which the GPT-2 pilot then used. That tokenizer comparison is confounded (less data, 124M params, slower), but a 2x gap is too big for those to explain.

## Status (2026-10-14)

- Done: the v3 validator (`scripts/validate_record_v3.py` + `tests/test_validate_record_v3.py`, also run by `validate_all_records.py`). Since 2026-10-14 it checks the rules it can see, not just file formats: each seed's config against the record's (dev-ladder evals, no resuming, track D's own pretrained model), clock arithmetic against the eval log, one GPU, a `confirm_v3.json` that scored the seed's own final weights on the test ladder and whose score is the mean of its levels, and distinct track D pretraining logs. CI also requires every ranked v3 record to be linked from README.md. Also done: the `records/v3/` layout, README and CONTRIBUTING for v3, and READMEs with results for all four records.
- **Record runs (5 seeds each), all on Engaging's H100 node (node2906, which reports exactly `NVIDIA H100 80GB HBM3`):**
  - Track A 000: done, all 5 seeds scored (test ladder 0.1297, 0.1236, 0.1221, 0.1463, 0.1258, mean 0.1295; all within the 18,000 s clock).
  - Track D 000: done, all 5 seeds scored (test ladder 0.0979, 0.1391, 0.1120, 0.1266, 0.1075, mean 0.1166; all 348 of 360 min). Training and scoring ran as separate jobs (STAGE=train on the H100, then STAGE=confirm on L40S nodes; the scoring sbatch file is on Engaging, not in the repo). Track D seed logs are ~5 MB each (train_log.jsonl has 11,089-13,641 steps); git already compresses them.
  - Track D 001 (Muon pretraining): done, **not a record**: 0.1162 ± 0.0156 vs 0.1166, Welch p = 0.52. Written up as a negative result and listed under "Not records" in README.md. On the H100, 56 min of Muon pretrained to a slightly worse loss than 112 min of AdamW (2.869 vs 2.825; they matched on the H200 pilots), and the extra RL time didn't help.
  - Track A 001 (ladder mix + constrained answers, lr 4e-6): done, **new record**: 0.2236, 0.2522, 0.1860, 0.2724, 0.3064 (mean 0.2481, sd 0.0460) against the baseline's 0.1295, Welch p = 0.0019. Seed 2 collapsed to short answers (40 tokens on average vs 415-466 for seeds 1, 3 and 4) and still scored 0.186.
  - **Why not AWS:** the 001 records were first launched on AWS (on-demand p5.4xlarge fleets, 2026-10-08 to 10-11, us-east-1 and us-east-2). In 72 h no instance launched (InsufficientInstanceCapacity in every us-east-1 zone, and quota for under one instance in us-east-2). The support case asking for P quota and Capacity Blocks has had no human reply. Same GPU string as the baselines, so the ranking isn't affected; it also keeps one provider per record (#20). `scripts/aws/` was smoke-tested on an A10G only, so the docs now name Modal (`scripts/modal_v3.py`) as the route that has produced a full seed, and say the AWS defaults must be overridden.
  - Track D: in-training dev evals run every 500 steps instead of 50. The clock excludes them, but Engaging's 6 h job limit doesn't. Each eval takes ~1 s, so every 50 steps would have added only ~4 min; 500 was a cautious choice.
- **You kept the H100 SXM as the ranked GPU (2026-10-08)**, although Engaging has ~24x more H200s in the same partition, because single H100s are much easier for contributors to rent.
- A re-read of the track A rules (you said "laterTM").
- Test-ladder use (score once, never tune on it) rests on the honour system and the scripted pipeline. The validator checks that in-loop evals used the dev ladder and that each `confirm_v3.json` scored the seed's own final weights, but it can't see tuning. Before the rules were frozen, the test ladder was scored 7 times on 5 pilot checkpoints (`docs/v3_design.md`, 2026-10-03), which informed decisions #2 and #5.

## Corrections after launch (2026-10-14)

A review of the launched v3 found these; RULES_v3 and the docs now say:
- The answer is the last `<answer>` block, or the last line if there is none (RULES_v3 said "first"). This matches the verifier; scoring didn't change.
- Training streams match ladder puzzles on numbers, target and operators, so the same numbers and target can be drawn under another operator set (0.06-0.07 % of ladder-mix draws, at most ~0.015 of a score). Documented and left for the next version.
- Detectable gains are ~0.017 (A) and ~0.028 (D) under the actual Welch test, not ~0.015 / ~0.025.
- One provider per record is a rule (#20), not only CONTRIBUTING advice.
- The track D plateau and the H200-speed explanation are corrected (see the failure cases and "Infrastructure").
- The pure-bf16 optimizer is disclosed (RULES_v3, Baselines).
- Measured clocks are quoted next to the budgets, and the README leaderboard has a sampler column.
- The record READMEs disclose where scoring ran (track D: L40S) and report pilots without picking the best seed.

## Open questions

Defaults in brackets; nothing waits on them.
- Optional stopping: the rules don't say whether N must be fixed before scoring. Adding seeds after a near-miss inflates the false-record rate. [N is fixed before the first test-ladder score; every scored seed counts.]
- Checkpoint choice: the rules say "final checkpoint" but don't explicitly ban picking weights on dev while dev evals are off the clock. [Final checkpoint only.]
- L1 (+ and - only) supplies 51-78 % of every record's score and L5 sits near 0, so the equal-weight ladder mostly measures L1, L2 and L6. [Keep it for v3; revisit for a v3.1.]
