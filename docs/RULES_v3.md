# RLVR Speedrun v3 — Rules

Status: final for launch; the maintainer approved it on 2026-10-07 (`docs/REVIEW_v3.md` lists every
decision). Edge cases the text doesn't settle are arbitrated by the maintainer, and rulings are added
here. v3 replaces v2's "time to a threshold" with a **fixed compute budget, scored on a
difficulty ladder**, and has two tracks. Evidence for each choice is in `docs/v3_design.md`.
Constants live in `rlvr_speedrun/benchmark.py` (v3 block); `scripts/validate_record_v3.py`
enforces these rules.

## The task

Countdown: given numbers and a target, write an arithmetic expression that uses every number
exactly once, only the operators the puzzle allows, and evaluates to the target. The prompt is
the fixed zero-shot `r1` prompt (`countdown.format_prompt(puzzle, 0, "r1")`). The answer is the
expression inside the first `<answer>...</answer>`; `countdown.verify` decides correctness with
exact rational arithmetic. The prompt and the verifier are part of the benchmark and may not be
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
(`data.train_puzzle_stream`, `ladder.ladder_stream`).

## Tracks

### Track A — pinned base model

- Start from `Qwen/Qwen3-1.7B-Base` at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Budget: **5 h** of training on one H100 80GB SXM.
- Data: puzzles from the generator (`countdown.generate_puzzle`, `ladder.ladder_stream`, or any
  other program that produces (numbers, target, operators) without solving them) and the
  verifier's verdicts. Nothing else: no other models, datasets or text.

### Track D — from scratch

**Status: experimental.** Pilots plateau at ~0.15-0.17 (dev ladder) whatever the pretraining
length, optimizer (AdamW or Muon), model size (110M or 335M), exploration settings or think
budget. The models learn a few fixed answer shapes (one product, then additions and
subtractions), and L3 and L5 stay near 0. Records are welcome, but the maintainer may revise this
track's ladder (e.g. an easier L5) at any time. A revision gets a new version number, and the
existing records stay on the old leaderboard.

- Start from randomly initialised weights; any architecture and size.
- Budget: **6 h** on one H100 80GB SXM, **pretraining included**. How to split it between
  pretraining and RL is the entrant's choice.
- Pretraining data: `HuggingFaceFW/fineweb-edu` at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (any subset, any filtering or reweighting that
  does not use an external model). Tokenizer: any tokenizer trained on that data, or the pinned
  TinyLlama tokenizer (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` at revision
  `59f6f375b26bde864a6ca194a9a3044570490064`). A GPT-2 tokenizer, which merges digits, halves the
  pilot score, so the tokenizer is a real part of an entry.
- The baseline uses two shards (`benchmark.V3_TRACK_D_SHARDS`, 1.72B tokens), prepared by
  `scripts/prepare_track_d_data.py`.
- RL data: as in track A (generator puzzles and verifier verdicts only).

### Both tracks: what counts as "verifier-only"

Allowed: any RL algorithm, curriculum or training mix over generated puzzles; any sampler during
training. Rewards and training signals may be any program or learned model whose only inputs are
the model's own outputs, the puzzle and the verifier's verdicts. For example:
- partial credit or process rewards, e.g. checking that the arithmetic the model writes
  (`7*3=21`) is correct;
- learned value functions, critics or process reward models, trained from scratch (or from the
  entrant's own model) on the run's own rollouts and their verdicts.

These may not run a solver or a search over expressions, use another pretrained model, or use
external data. Reward code beyond the verifier is part of the submission and gets the same
careful review as custom samplers. Training them counts toward the time budget. They are for
training only: the evaluation sampler may not use a value function, critic or reward model.

Forbidden:
- Training on solutions: the generator's `Puzzle.solution` field, the output of any solver or
  search over expressions, or any text produced by another model. (A 1,000-step SFT warm start
  on generator solutions lifts track D from ~0.14 to 0.24, so allowing them would turn the track
  into distillation.)
- Using the test ladder for anything but the final score.
- Models or data other than those listed for the track. This includes synthetic or
  programmatically generated text (arithmetic facts, equations, expression-value pairs), in
  pretraining or RL: a random expression with its value is a puzzle with its solution. The model
  may be rewarded for arithmetic it writes itself, but not trained on arithmetic text it didn't.

## The evaluation sampler

The submission is the final weights plus a sampler configuration. Allowed: temperature,
answer-constrained decoding (`--constrain`: after `<answer>`, only well-formed uses of the
puzzle's numbers and operators), a think budget (`--think-budget`), or another sampler the
entrant implements, provided it:

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

Why the sampler is part of the submission: a model trained with answer-constrained decoding
scores 0.000 without it, because the constraint is part of its policy. The constraint by itself
solves nothing (bases: 0.003-0.011), and it adds only +0.011 to a trained Qwen model.

## Clock and hardware

- The budget is GPU wall clock on one H100 80GB SXM (`NVIDIA H100 80GB HBM3`), from the first
  GPU operation of the run to the final checkpoint being written. In-loop dev evals may be
  excluded from the clock if logged separately; everything else on the GPU counts (pretraining,
  rollouts, updates, compilation, warmup).
- CPU-only preparation (download, tokenizer training, tokenization, filtering) is not timed,
  but must be a deterministic script in the submission that uses only the allowed data.
- Scoring on the test ladder happens after the clock stops and is not timed. It may run as a
  separate job, on any GPU, but always with `ladder_confirm` under the reference software below.
- Training runs on other GPUs are unranked.
- Reference software (`benchmark.V3_REFERENCE_SOFTWARE`): vLLM 0.30.0, torch 2.13.0+cu130,
  transformers 5.17.0. The baselines use it, and the scorer always runs in it. Records may train with
  other software if the README says which.

## Seeds and statistics

- At least **5 seeds** (`0..N-1`), all reported; dropping a run disqualifies the record. A run that
  crashed or was interrupted by the infrastructure (out of memory, preemption, a lost
  machine) may be rerun under the same seed, if the failed attempt goes in `extra/` with a note.
- In track D a seed is the whole pipeline: each seed pretrains its own model (with that seed)
  before RL. Sharing one pretrained model across seeds is not allowed.
- The record's score is the mean seed score. The baselines' seed-to-seed standard deviations
  are 0.010 (track A) and 0.016 (track D), so 5 seeds detect gains of ~0.015 and ~0.025.
- A new record must beat the holder's mean with a one-sided Welch t-test at p < 0.05 on the
  seed scores.

## Submissions

`scripts/run_seed_v3.sh <record> <seed>` runs one seed end to end and writes `clock.json` with
the timed total. The steps are pretraining (track D), GRPO, then the test ladder with the record's
sampler. `STAGE=train` / `STAGE=confirm` split the timed part from the scoring.
`scripts/aws/spot_launch.sh` (`V3=1`) runs seeds on fresh AWS instances.

`records/v3/track_{a,d}/<NNN>_<slug>/` holds:
- `config.json`: the trainer config plus a `v3` block with the track, sampler and track D
  pretraining;
- `README.md`: recipe, clock time per seed, sampler and test score;
- any code that differs from this repository;
- per seed, `seeds/<i>/{train_log.jsonl, result.json, clock.json, confirm_v3.json}`, plus
  `pretrain_log.jsonl` for track D;
- `extra/`: every other run.

`python scripts/validate_record_v3.py <record> [--compare-to <holder>]` must pass. It checks the
seeds, the clock (ranked GPU, within budget), a full test-ladder score with the record's own
sampler, the pinned track A revision, per-seed pretraining for track D, and the Welch test.

## Baselines (records 000)

- Track A (`records/v3/track_a/000_baseline`): GRPO on the generator mix, lr 2e-6, r1 prompt,
  1,024 tokens, 290 min, free sampling at T=1. This is calibration/sweep_008 `q3_17b_lr2e-6`
  with `micro_batch_tokens` lowered from 16384 to 8192 so it fits in 80 GB, which doesn't change the
  update. H200 pilots: test ladder 0.153 / 0.123.
- Track D (`records/v3/track_d/000_baseline`): 110M Llama-style model, 112 min of AdamW pretraining
  on FineWeb-Edu, then 234 min of Dr. GRPO at lr 3e-5 on the ladder mix with answer-constrained
  decoding, scored with `--constrain` (calibration/sweep_d10 `pre120`). H200 pilots: dev ladder
  0.175 / 0.167 / 0.137.

Both leave ~10 min of the budget for start-up, loading and saving, which count. Muon at 60 min of
pretraining matched AdamW at 120 min in the pilots. The baseline keeps plain AdamW and leaves
Muon for records. The record READMEs give the H100 test scores; the pilot numbers above are
from H200s, which do more work per minute.
