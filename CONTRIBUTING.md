# Contributing

Two kinds of PRs: **records** (a new leaderboard entry) and **code** (trainer,
validator, docs). Both run the same CI: tests, frozen-data checksums, and
`scripts/validate_all_records.py`.

## Submitting a record

The full rules are in [RULES.md](RULES.md); this is the workflow.

1. **Start from the holder.** `python scripts/new_record.py <slug> --provider <host>`
   copies the current holder's `config.json` into `records/track_a/<NNN>_<slug>/`
   and writes a README skeleton. Edit `description` and the hyperparameters you
   change; anything that is not a trainer field (`description`, `parent`,
   `hardware`, ...) is metadata and ignored by `--config`.
2. **Train N ≥ 3 seeds, all on one provider, on a single H100 80GB SXM.**
   Locally:
   ```bash
   for SEED in 0 1 2; do
     python -m rlvr_speedrun.grpo --config records/track_a/<NNN>_<slug>/config.json \
         --device cuda --seed $SEED --out-dir records/track_a/<NNN>_<slug>/seeds/$SEED
     python -m rlvr_speedrun.capability --model records/track_a/<NNN>_<slug>/seeds/$SEED/final \
         --base Qwen/Qwen2.5-0.5B --out records/track_a/<NNN>_<slug>/seeds/$SEED/probe.json
   done
   ```
   or on Modal, all seeds in parallel with the probe included:
   ```bash
   modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/<NNN>_<slug> \
       --args "--config records/track_a/<NNN>_<slug>/config.json"
   ```
   then copy `results/<NNN>_<slug>/seeds/` into the record. Seeds are
   `0..N-1`, chosen before you look at results; a seed that never reaches the
   threshold cannot be dropped: RULES ("Seeds and statistics") says when a
   censored seed may be re-run with a higher `max_steps` and how to report it.
3. **Validate and compare.**
   ```bash
   python scripts/validate_record.py records/track_a/<NNN>_<slug>
   python scripts/validate_record.py records/track_a/<NNN>_<slug> --compare-to records/track_a/<holder>
   ```
   Holder status needs one-sided Welch `p < 0.05` against the current holder.
   A record that is not significantly faster can still be merged as a ranked
   entry (negative and neutral results are useful) — say so in the README.
4. **Disclose everything.** Every run launched with the record's configuration
   goes in `seeds/` (official), `extra/` (exploratory, censored, retries) or
   the README. `validate_record.py` cannot check this; reviewers will ask.
5. **Open the PR** with the template filled in and a row added to the
   leaderboard in `README.md` (CI fails if a ranked record is not linked).

Unranked entries — a smoke test, a 4090 reproduction, a run on an H100 PCIe —
set `"unranked": true` in `config.json` and skip the seed/probe/hardware checks.
They are welcome inside an existing record's folder (`repro_<gpu>/`) or as
their own folder, but never appear in the leaderboard table.

## Code changes

* `uv sync && uv run pytest -q && uv run ruff check .` before pushing.
* Anything on the timed path (rollouts, scoring, optimizer, eval) or in the
  verifier/eval prompt is a benchmark change. Say in the PR how existing
  records are affected; changes to `data/` fail CI by design (the eval set is
  frozen — regenerating it means a new benchmark version).
* Timing-neutral tooling (validator, scripts, docs) can land without a record.
  Speedups should come with a record that demonstrates them; a trainer change
  without a record is fine if it is off by default.

## Good first records

See the list at the bottom of [README.md](README.md). If you have a single
H100 for an hour, the cheapest experiment is three seeds of the holder recipe
with one change; `new_record.py` sets that up in one command.
