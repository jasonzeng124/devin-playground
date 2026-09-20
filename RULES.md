# RLVR Speedrun — Rules

The speedrun measures **wall-clock time to teach a language model to solve
Countdown arithmetic puzzles with reinforcement learning from verifiable
rewards (RLVR)**. Everything is scored by a deterministic, CPU-only verifier
(`rlvr_speedrun/countdown.py`). Learned reward models are never allowed — that
would break the "verifiable" in RLVR.

## The task

A Countdown puzzle gives 4 integers in `[1, 25]` and a target in `[10, 200]`.
The model must output an arithmetic expression using `+ - * /` and **each
number exactly once** that evaluates exactly to the target (exact rational
arithmetic — `7 / 2 * 2` is fine, `7 / 2` rounded is not).

* Reward: `+1` correct, `0` well-formed but wrong, `-0.1` malformed
  (unparseable / not a pure arithmetic expression). See `verify()`.
* Held-out eval: `data/countdown_eval.jsonl` — 2,000 puzzles generated once
  from seed `20240601` and frozen. **Never train on it.** Training puzzles are
  sampled from the infinite generator (`train_puzzle_stream`), which uses a
  disjoint seed space.
* Eval protocol: greedy decoding, `--few-shot 3` fixed prompt, `max_new_tokens
  <= 64`. `pass_rate` on all 2,000 puzzles is the headline metric. Report the
  Wilson 95% CI that the eval script prints.

## Tracks

### Track A — fixed base model (RL-only speedrun)

Everyone starts from the **same frozen base checkpoint** (below; it is the
smallest open model whose few-shot pass rate is non-zero — see
`results/base_sweep.json` and the table in `README.md`). Free to change: RL
algorithm, reward shaping (must be a deterministic function of the verifier
output), KL schedule, curriculum, group size, rollout batching, prompt
template used *during training*, systems/kernels.

* **Base model**: `Qwen/Qwen2.5-0.5B` — the pretrained checkpoint (no SFT /
  RLHF / RLVR post-training; Qwen's pretraining mix does contain synthetic
  math, so "base" means no post-training, not "never saw arithmetic"). Records
  must state the Hugging Face revision they loaded (`main` at the time of the
  run is fine; a pinned commit hash is better).
* **Threshold**: `pass_rate >= 0.10` on the full 2,000-puzzle eval set with
  the fixed eval protocol above. Set from the first Qwen2.5-0.5B baseline
  (~10-12% in 300 GRPO steps); it will be raised once records saturate it —
  a raised threshold starts a new leaderboard table.
* **Clock**: `time_to_threshold_s` is measured by the training process with a
  monotonic clock. It **starts** immediately before the first training rollout
  (nothing that updates the weights or produces training data may run before
  it) and **stops** when the first full 2,000-puzzle evaluation with
  `pass_rate >= 0.10` finishes. Everything in between counts: rollouts,
  scoring, optimizer steps, checkpointing, and every evaluation — including
  evaluations that miss the threshold, so `eval_every`/`eval_start_step` are
  part of the recipe and an eval that fires too early is a real cost.
* **Untimed warm-up** (before the clock starts) may load the model, compile,
  capture CUDA graphs and run warm-up rollouts/evals, as long as it performs
  **no optimizer step**, does not touch the RNG state the training run will
  use, and uses puzzles that are never reused for training (`--compile` in
  the reference GRPO does exactly this from a disjoint puzzle stream inside
  `torch.random.fork_rng`). `warmup_s` is recorded in `result.json` and
  reported in the record README.
* **Step cap**: `max_steps` is a censoring limit, not a tuning knob — set it
  generously (at least 2x the steps you expect to need). A seed that hits the
  cap without reaching the threshold has no `time_to_threshold_s` and is a
  **censored seed** (see Seeds and statistics).
* **Capability guardrail** (anti-reward-hacking): after the run stops, probe the
  final checkpoint with `python -m rlvr_speedrun.capability --model <ckpt>
  --base Qwen/Qwen2.5-0.5B`. Every seed must have
  `fineweb_loss_delta <= 0.05` nats and `mmlu_lite_acc_delta >= -0.03`.
  Probe time does **not** count toward the clock. Records 002/003 predate this
  probe, are marked “guardrail: not measured” and are the only records the
  validator exempts.

### Track B — full stack (pretrain + RL speedrun)

Anything goes, including architecture and pretraining data, timed from
**random initialization**. Win condition is a dual threshold, so capability
preservation is part of the objective by construction:

* FineWeb val loss `<= X` **and** Countdown `pass_rate >= Y`.
* `X` and `Y` are calibrated together from the first baseline record, not
  chosen independently; the first accepted Track B record sets them and they
  are then frozen in `records/track_b/README.md`.

## Hardware and reproducibility

The leaderboard hardware is **one NVIDIA H100 80GB SXM** (`torch.cuda.get_device_name()`
== `NVIDIA H100 80GB HBM3`; this is what Modal, RunPod "H100 SXM" and Lambda
sell for ~$2.5-4/hr). Only runs on exactly this GPU are ranked: the H100
PCIe has ~40% less memory bandwidth (2.0 vs 3.35 TB/s) and the H100 NVL more
(94 GB, 3.9 TB/s), and a 0.5B-model rollout loop is bandwidth-bound, so
neither is interchangeable. An `8xH100` tier will be opened once a
distributed trainer exists and single-GPU records saturate the threshold.

Runs on other hardware (an RTX 4090, an H100 PCIe) are welcome as unranked
reproductions inside a record's folder (`repro_<gpu>/`) — they help
contributors without an H100 verify a recipe, but their times are never
compared to ranked ones.

Software is **recorded, not frozen**: kernel and library upgrades are
legitimate systems speedups. Every `result.json` written by the reference
trainer carries an `environment` block (`gpu`, `cuda`, `torch`,
`transformers`, `python`, `git_commit`); the record README must additionally
give the exact install (`pip install torch==... transformers==...`) and the
exact command that produced the seeds, so that a third party can reproduce
the record on a rented H100 with no guesswork. The reference environment is
`torch 2.5.1` + `transformers 4.57` + bf16 (`scripts/modal_run.py`).

The same GPU model on different hosts is not identical: between Modal and
RunPod H100 SXM pods we measured a 5-8 % per-step gap (host CPU, driver,
PCIe/NVLink topology), which is small next to seed-to-seed variance in
steps-to-threshold but not zero. A record's README must therefore say which
provider/host each seed ran on, and a record whose seeds are split across
providers should report per-provider step times so readers can separate
hardware from learning variance.

## Seeds and statistics

RL time-to-threshold is high-variance and right-skewed: over 10 seeds each,
record 004 is 958 ± 348 s and record 005 is 460 ± 196 s (std ≈ 40 % of the
mean), and the first three seeds of 004 had read 655 ± 116 s. The rules below
exist so that a leaderboard claim is a statement about the recipe, not about a
lucky draw.

**Seed protocol.**

* A record declares its seeds in `config.json` (`"seeds": [0, 1, ..., N-1]`)
  and must run exactly the consecutive seeds `0..N-1` with **N >= 3**; the
  `seeds/` directory must contain exactly those. No picking seed IDs.
* Fix the recipe **before** launching the declared seeds. Exploratory /
  tuning runs are fine but are not the record; the README must say how many
  were done (count and GPU-hours) so readers can judge the garden of forking
  paths.
* **Every launched run with the record's configuration is reported.** Runs
  that are not part of `seeds/` (censored seeds, crashed runs, re-runs) live
  under `extra/<why>/` with their logs; dropping a run is the one thing that
  gets a record rejected outright.
* **Censored seeds** (cap hit, threshold not reached) have no time and make
  the record invalid as submitted — a record's times are only meaningful when
  all N seeds reached the threshold. The remedy is to raise `max_steps` and
  re-run **that seed** (the cap cannot have affected seeds that finished
  under it; the rest of the config must be byte-identical); the censored
  attempt stays in `extra/` and the README says so. Note this replaces a
  known-slow draw with a fresh one and so biases the mean slightly downward
  (GPU training with `torch.compile`/bf16 is not bit-reproducible, so the
  re-run is a new trajectory, not the old one continued — record 005 seed 8
  is an example). Report the comparison both ways (censored seed at its
  re-run time, and at its censoring time as a lower bound); if more than one
  seed in three is censored, the cap was too low and all seeds must be re-run
  under the new cap.
* You may add seeds after seeing results (N -> M, still consecutive), never
  remove them.

**Reporting.** `scripts/validate_record.py <record>` prints, over the N
seeds, the mean, sample standard deviation and Student-t 95% CI of
`time_to_threshold_s` and final `pass_rate`, and checks every rule above.
The leaderboard shows `mean ± std (N)`.

**Beating a record.** A new record replaces the current best only if

```
python scripts/validate_record.py records/track_a/<new> --compare-to records/track_a/<current best>
```

reports **`p < 0.05`** for the one-sided Welch t-test of
`mean(time_new) < mean(time_best)` over the two records' per-seed times
(both records must be ranked-hardware, same track, same threshold). The
validator also prints the exact permutation-test p-value for the difference
in means as a distribution-free cross-check; if the two disagree about
`0.05`, say so in the PR and expect discussion. Power is low at N=3 (against
std ≈ 200-350 s a gap of several hundred seconds is needed to reach
`p < 0.05`; with N=10 per side a ≈ 250 s gap is detectable), so a recipe that
is faster but not significant at N=3 should run more seeds, not argue. A
record that is valid but not significantly faster than the current holder is
still merged and listed (it documents a recipe), but does not
become the holder. The **record holder** is the most recent record that beat
the then-holder at `p < 0.05`; the README leaderboard marks it.

Comparisons are unpaired: seed `i` of two records does not share a
trajectory (any change to the recipe changes how the RNG stream is
consumed), so identical seed IDs buy reproducibility, not pairing.

Records 002-004 predate this section and were accepted on `mean` alone;
005 is the first record whose claim over its predecessor was tested: not
significant at N=3 (p = 0.11), significant after both records were extended
to N=10 (p = 0.0007).

## Submitting a record

1. Copy `records/track_a/000_smoke_cpu/` as a template.
2. Name it `records/<track>/<NNN>_<short_slug>/` with the next number.
3. Include: `README.md` (what changed, hardware, exact install + command,
   number of exploratory runs, warm-up time), `config.json` (with `track`,
   `model`, `hardware`, `seeds` and the RL hyperparameters), and
   `seeds/<seed>/{config.json,train_log.jsonl,eval_log.jsonl,result.json,probe.json}`
   for each declared seed; `extra/` for everything else that was launched.
   Do not commit model weights.
4. Run `python scripts/validate_record.py records/<track>/<NNN>_<slug>` — it
   must pass. Run it again with `--compare-to` against the current record
   holder and paste both outputs into the PR.
5. Open a PR; put the summary table in the description. Add yourself to the
   leaderboard in `README.md`.

## Forbidden

* Training on, or peeking at, `data/countdown_eval.jsonl`.
* Any learned or LLM-based reward.
* Hard-coding a solver into the model's inference path (the model itself must
  emit the expression).
* Changing the verifier, the eval set, or the prompt used *at evaluation*.
* Omitting any run launched with the record's configuration, or choosing
  which seeds to report.
