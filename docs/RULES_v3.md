# RLVR Speedrun v3 — Rules

Status: final for launch; the maintainer approved it on 2026-10-07 (`docs/REVIEW_v3.md` lists every
decision, and the corrections made after launch). **Frozen on 2026-10-19** with one amendment,
external data (see "Both tracks: external data"); the rules don't change while they are frozen.
**v3.1 (2026-10-25):** the maintainer moved the reference GPU from the H100 80GB SXM to the
**H200 SXM** (see "Clock and hardware"); nothing else changed.
Edge cases the text doesn't settle are arbitrated by the maintainer, and rulings are added
here. v3 replaces v2's "time to a threshold" with a **fixed compute budget, scored on a
difficulty ladder**, and has two tracks. Evidence for each choice is in `docs/v3_design.md`.
Constants live in `rlvr_speedrun/benchmark.py` (v3 block); `scripts/validate_record_v3.py`
checks what can be checked from a record's files (see Submissions).

## The task

Countdown: given numbers and a target, write an arithmetic expression that uses every number
exactly once, only the operators the puzzle allows, and evaluates to the target. The prompt is
the fixed zero-shot `r1` prompt (`countdown.format_prompt(puzzle, 0, "r1")`). The answer is the
expression inside the last `<answer>...</answer>` block or, if there is none, the last non-empty
line of the completion, with anything after an `=` dropped (`countdown.extract_expression`). The
scorer stops each sample at its first `</answer>`, so a scored sample has at most one block.
`countdown.verify` decides correctness with exact rational arithmetic. The prompt and the verifier are part of the benchmark and may not be
changed.

## Scoring

The **test ladder** (`data/ladder_v3_test.jsonl`, frozen) has 400 puzzles at each of 6 levels:

| level | numbers | operators | the target needs |
|---|---|---|---|
| L1 | 4 | + - | not the plain sum |
| L2 | 4 | + - * | * |
| L3 | 4 | + - * / | / |
| L4 | 5 | + - * / | * or / |
| L5 | 6 | + - * / | * or /, targets 100-999 |
| L6 | 4 | one of {+ *}, {- *}, {+ /}, {- /} | both operators |

After training, each seed's final checkpoint is scored once with
`python -m rlvr_speedrun.ladder_confirm`: 16 independent samples per puzzle, at most 1,024 new
tokens each. A seed's score is the mean over levels of the level's pass@1 (mean correctness of
its samples). Levels weigh the same, so no single easy trick carries the score (a random valid
expression scores ~0.02; "add everything" scores 0 on L1).

The **dev ladder** (`data/ladder_v3_dev.jsonl`, same construction, disjoint) is for in-loop
evals and tuning. The test ladder may not be evaluated during training or used to pick
checkpoints, hyperparameters or seeds. Training streams skip every puzzle of both ladders
(`data.train_puzzle_stream` matches puzzles on numbers and target, `ladder.ladder_stream` on
numbers, target and operators). The ladder mix can therefore still draw a ladder puzzle's numbers
and target under another operator set: 0.06-0.07 % of its draws, worth at most ~0.015 of a seed's
score even if every such puzzle were memorised. Closing this gap changes the training streams, so
it is left for the next benchmark version.

## Tracks

### Track A — pinned base model

- Start from `Qwen/Qwen3-1.7B-Base` at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Budget: **5 h** of GPU wall clock on one H200 SXM (see "Clock and hardware").
- Data: puzzles from the generator (`countdown.generate_puzzle`, `ladder.ladder_stream`, or any
  other program that produces (numbers, target, operators) without solving them), the
  verifier's verdicts, and public datasets that aren't Countdown-specific (see "Both tracks:
  external data"). No other models.

### Track D — from scratch

**Status: experimental.** In the H200 pilots the best recipes reached about 0.15-0.17 on the dev
ladder, and nothing tried went past that: longer pretraining (30-120 min), Muon instead of AdamW,
a 335M model instead of 110M, more exploration (temperature 1.2, dynamic sampling, oversampling,
Clip-Cov) or a think budget (which scored lower, 0.07-0.12). The H100 records score lower still
(see Baselines). The models learn a few fixed answer shapes (one product, then additions and
subtractions), and L3 and L5 stay near 0. Records are welcome, but the maintainer may revise this
track's ladder (e.g. an easier L5) at any time. A revision gets a new version number, and the
existing records stay on the old leaderboard. Since 2026-10-19 public datasets other than
FineWeb-Edu are allowed (see "Both tracks: external data"), and no pilot has used them yet.

- Start from randomly initialised weights; any architecture and size.
- Budget: **6 h** on one H200 SXM, **pretraining included**. How to split it between
  pretraining and RL is the entrant's choice.
- Pretraining data: any allowed dataset (see "Both tracks: external data"). The baseline uses
  `HuggingFaceFW/fineweb-edu` at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Filtering
  and reweighting may not use an external model. Tokenizer: any tokenizer trained on allowed data, or the pinned
  TinyLlama tokenizer (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` at revision
  `59f6f375b26bde864a6ca194a9a3044570490064`). A GPT-2 tokenizer, which merges digits, halves the
  pilot score, so the tokenizer is a real part of an entry.
- The baseline uses two shards (`benchmark.V3_TRACK_D_SHARDS`, 1.72B tokens), prepared by
  `scripts/prepare_track_d_data.py`.
- RL data: as in track A (generator puzzles, verifier verdicts and allowed public datasets).

### Both tracks: what counts as "verifier-only"

Allowed: any RL algorithm, curriculum or training mix over generated puzzles; any sampler during
training. Rewards and training signals may be any program or learned model whose only inputs are
the model's own outputs, the puzzle and the verifier's verdicts (or, for a problem from an allowed
external dataset, that dataset's published answer). For example:
- partial credit or process rewards, e.g. checking that the arithmetic the model writes
  (`7*3=21`) is correct;
- learned value functions, critics or process reward models, trained from scratch (or from the
  entrant's own model) on the run's own rollouts and their verdicts.

These may not run a solver or a search over expressions, use another pretrained model, or be
trained on external data (external data may train the policy, below, but not reward signals). Reward code beyond the verifier is part of the submission and gets the same
careful review as custom samplers. Training them counts toward the time budget. They are for
training only: the evaluation sampler may not use a value function, critic or reward model.

Forbidden:
- Training on solutions: the generator's `Puzzle.solution` field, the output of any solver or
  search over expressions, or Countdown-specific data. (A 1,000-step SFT warm start on generator
  solutions lifts track D from ~0.14 to 0.24, so allowing them would turn the track into
  distillation.)
- Running another model (anything but track A's pinned base model and the entrant's own
  weights), as a teacher, data generator, reward model or starting point.
- Using the test ladder for anything but the final score.
- Models or data other than those listed for the track. This includes synthetic or
  programmatically generated text that the entrant generates (arithmetic facts, equations,
  expression-value pairs), in any stage: a random expression with its value is a puzzle with its
  solution. The model may be rewarded for arithmetic it writes itself, and trained on arithmetic
  text from an allowed public dataset, but not on arithmetic text generated for the submission.

### Both tracks: external data

From 2026-10-19, any public dataset may be used in either track, at any stage (pretraining,
fine-tuning, or RL on its problems against its published answers), unless it is
Countdown-specific. A dataset is Countdown-specific if it
was made for Countdown, the 24 game or other numbers-game puzzles, or for this benchmark. A
general dataset that contains some such items may be used once those items are removed (a
keyword filter is enough). A published dataset that another model wrote (a distillation or
synthetic set) counts as a dataset and is allowed. Running another model is not.

Data the entrant generates is limited as before: generator puzzles, verifier verdicts and the
model's own outputs. So the amendment admits public expression-value data, such as the
arithmetic modules of DeepMind's `mathematics_dataset`, which the earlier rules banned as
synthetic arithmetic text. It does not admit the same kind of text generated for a submission.

Training on external data counts toward the time budget. Downloading, filtering, mixing and
tokenizing are untimed preparation, done by a deterministic script in the submission that uses no
model. The record README names each dataset and its revision. The records made before the
amendment (A 000, A 001, D 000, D 001) used no external data and stay valid.

## The evaluation sampler

The submission is the final weights plus a sampler configuration. Allowed: temperature,
answer-constrained decoding (`--constrain`: after `<answer>`, only well-formed uses of the
puzzle's numbers and operators), a think budget (`--think-budget`, which needs `--constrain`:
it forces `<answer>` after N free tokens), or another sampler the entrant implements, provided it:

- produces each sample independently, in one pass, from the prompt alone;
- never evaluates arithmetic, never sees the target's relation to an expression, and never calls
  the verifier (enforcing syntax and number usage is fine; checking values is search);
- is included in the submission and runs inside `ladder_confirm`.

Best-of-n selection, majority voting or retries are not allowed: each of the 16 samples is
scored on its own.

Samplers beyond the built-in options (temperature, `--constrain`, `--think-budget`) are reviewed
carefully. The README must describe what the sampler changes and why, and the code must be short
enough to read. The maintainer may reject a sampler that meets the letter of the rules but does
part of the solving (e.g. a grammar that encodes which expressions tend to be right).

Why the sampler is part of the submission: a model trained with answer-constrained decoding can
score 0.000 without it (a track D pilot did), because the constraint is part of its policy. The
constraint by itself solves nothing (base models: 0.003-0.011). On trained Qwen models it adds
+0.011 (a pilot trained without it; `docs/v3_design.md`), +0.011-0.014 (track A 000) and about
+0.016 (track A 001's healthy seeds), but +0.074 for 001's collapsed seed
(`calibration/diag_sampler`; 001's README says which numbers come from the dev ladder).

## Clock and hardware

- The budget is GPU wall clock on one H200 SXM 141GB (`NVIDIA H200`; the `NVIDIA H200 NVL` is
  unranked), from the first
  GPU operation of the run to the final checkpoint being written. In-loop dev evals may be
  excluded from the clock if logged separately; everything else on the GPU counts (pretraining,
  rollouts, updates, compilation, warmup).
- CPU-only preparation (download, tokenizer training, tokenization, filtering) is not timed,
  but must be a deterministic script in the submission that uses only the allowed data.
- Scoring on the test ladder happens after the clock stops and is not timed. It may run as a
  separate job, on any GPU, but always with `ladder_confirm` under the reference software below.
- Training runs on other GPUs are unranked.
- v3.0 (2026-10-07 to 2026-10-24) used one H100 80GB SXM (`NVIDIA H100 80GB HBM3`). Its records
  (track A 000-001, track D 000-002) keep their scores on a separate H100 leaderboard and are
  never compared with H200 records. The H200 leaderboard starts from reruns of the H100 holders'
  configs (A 001, D 002) on the H200, as ordinary 5-seed records. Why: Engaging's
  non-preemptable queue has one H100 node and thirteen H200 SXM nodes, so H100 records queued
  behind a single node.
- All seeds of a record train on one provider (one cloud, or one cluster), so differences between
  hosts don't mix into the seed spread. Runs elsewhere go in `extra/` with a note (for example
  track A 000's Modal run, `extra/modal_0`).
- Reference software (`benchmark.V3_REFERENCE_SOFTWARE`): vLLM 0.30.0, torch 2.13.0+cu130,
  transformers 5.17.0. The baselines use it, and the scorer always runs in it. Records may train with
  other software if the README says which.

## Seeds and statistics

- At least **5 seeds** (`0..N-1`), all reported; dropping a run disqualifies the record. A run that
  crashed or was interrupted by the infrastructure (out of memory, preemption, a lost
  machine) may be rerun under the same seed, if the failed attempt goes in `extra/` with a note.
- In track D a seed is the whole pipeline: each seed pretrains its own model (with that seed)
  before RL. Sharing one pretrained model across seeds is not allowed.
- The record's score is the mean seed score. The baselines' seed standard deviations are 0.0098
  (track A) and 0.0163 (track D). With 5 seeds against 5, the one-sided Welch test has 80 % power
  for gains of about 0.017 (A) and 0.028 (D), about 1.75 sd. A noisier holder needs larger gains:
  track A 001's sd is 0.046 because one seed collapsed, so a challenger needs about +0.06 if it is
  as stable as 000, or about +0.08 if it is as noisy as 001.
- A new record must beat the holder's mean with a one-sided Welch t-test at p < 0.05 on the
  seed scores.
- N is fixed before the first test-ladder score, and every scored seed counts: adding seeds
  after a near-miss is not allowed.
- Each seed is scored on its final weights. Choosing a checkpoint (on the dev ladder or
  otherwise) is not allowed.

## Submissions

`scripts/run_seed_v3.sh <record> <seed>` runs one seed end to end and writes `clock.json` with
the timed total. The steps are pretraining (track D), GRPO, then the test ladder with the record's
sampler. `STAGE=train` / `STAGE=confirm` split the timed part from the scoring.
`scripts/modal_v3.py` runs a seed on Modal. `scripts/aws/spot_launch.sh` (`V3=1`) has not run on
an H100 yet, and its account defaults must be overridden.

`records/v3/track_{a,d}/<NNN>_<slug>/` holds:
- `config.json`: the trainer config plus a `v3` block with the track, sampler and track D
  pretraining;
- `README.md`: recipe, clock time per seed, sampler and test score;
- any code that differs from this repository;
- per seed, `seeds/<i>/{config.json, train_log.jsonl, eval_log.jsonl, result.json, clock.json,
  confirm_v3.json}` as `run_seed_v3.sh` writes them, plus `pretrain_log.jsonl` for track D;
- `extra/`: every other run.

`python scripts/validate_record_v3.py <record> [--compare-to <holder>]` must pass. It checks:
- the seeds, and that each seed's config matches the record's and ran its in-loop evals on the
  dev ladder;
- the clock: ranked GPU, one GPU, within budget, and arithmetic that matches the eval log;
- a full test-ladder score of each seed's own final weights with the record's sampler;
- the pinned track A revision, and per-seed pretraining for track D;
- with `--compare-to`, the Welch test.

It can't see how the test ladder was used, which provider ran the seeds, or what the training
data was. Those rest on the submission's scripts and on review.

## Baselines (records 000)

- Track A (`records/v3/track_a/000_baseline`): GRPO on the generator mix, lr 2e-6, r1 prompt,
  1,024 tokens, a 290-min GRPO budget (291 min on the clock), free sampling at T=1. This is
  calibration/sweep_008 `q3_17b_lr2e-6` with `micro_batch_tokens` lowered from 16384 to 8192 so
  it fits in 80 GB, which doesn't change the update. H200 pilots: test ladder 0.153 / 0.123. Those
  pilots were trained before the test ladder was drawn, so their streams were not filtered
  against it (sweep_008 seed 0's stream holds ~8.5 % of the test ladder's (numbers, target)
  pairs).
- Track D (`records/v3/track_d/000_baseline`): 110M Llama-style model, 112 min of AdamW pretraining
  on FineWeb-Edu, then a 234-min Dr. GRPO budget (236 min on the clock) at lr 3e-5 on the ladder
  mix with answer-constrained decoding, scored with `--constrain`. This is calibration/sweep_d10
  `pre120` (120 + 240 min) shortened to fit. H200 pilots: dev ladder 0.175 / 0.167 / 0.137.

The measured clocks are 291 of 300 min (A) and 348 of 360 min (D). Start-up, loading and saving
take 1-2 min, so about 9-12 min of each budget is unused.

The trainer updates bf16 weights directly with AdamW, with no fp32 master copy, so an update
smaller than half a weight's bf16 spacing is lost. At track A's learning rates most updates
round to zero: after 5 h, 2.2 % of the weights differ from the base model in 000 (lr 2e-6) and
4.4 % in 001 (lr 4e-6). The trainer is not part of the benchmark, so records may change this.

Pilots against records: the record READMEs give the H100 test scores, and the pilot numbers above
are from H200s. In track A the H200 pilots ran 3-18 % more steps per minute, for 300 min instead
of 290. In track D speed doesn't explain the gap. RL ran 47-58 steps/min on the H100 against 52-55
on the H200, and pretraining 359-373k tokens/s against ~383k. Yet the H100 baseline's final
in-loop dev evals average 0.119, against 0.159 for the pre120 pilots (one-sided Welch p = 0.025).
Each pilot arm shared one pretrained model across its seeds, so one lucky pretraining run could
explain it. The pilots' pretraining commands are in sbatch files on Engaging and were not
rechecked. Muon at 60 min of pretraining matched AdamW at 120 min in the pilots, but on the H100
it didn't help (track D 001: 0.1162 against 0.1166).
