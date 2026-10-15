# Contributing

The current benchmark is **v3** ([docs/RULES_v3.md](docs/RULES_v3.md)). Records are pull requests
that add a folder under `records/v3/track_{a,d}/`. CI validates every record on each PR.

## Submitting a record

1. **Develop on the dev ladder.** Tune on `data/ladder_v3_dev.jsonl` (the trainer's in-loop eval).
   Never evaluate on the test ladder until the final runs. The smoke configs in
   `records/v3/smoke/` run in a few minutes on any GPU and exercise the whole pipeline.
2. **Create the record folder**, `records/v3/track_{a,d}/<NNN>_<slug>/`, with:
   - `config.json`: the trainer config plus a `v3` block:
     `{"track": "A"|"D", "sampler": {"temperature", "constrain", "think_budget"}, "pretrain": {...}}`
     (`pretrain` for track D only; see the baselines);
   - `README.md`: what changed and why, the test score, the clock time per seed, the sampler, and
     the software used;
   - any code that differs from this repository: a custom sampler, reward code, a pretraining
     script. Custom samplers and reward code are reviewed carefully (RULES_v3).
3. **Run >= 5 seeds** (`0..N-1`) on one H100 80GB SXM (`NVIDIA H100 80GB HBM3`), all on one
   provider:

   ```bash
   python scripts/prepare_track_d_data.py              # track D, once per machine (CPU, untimed)
   bash scripts/run_seed_v3.sh records/v3/track_d/<NNN>_<slug> <seed>
   ```

   Each seed writes `seeds/<seed>/{train_log.jsonl, result.json, clock.json, confirm_v3.json}`
   (plus `pretrain_log.jsonl` for track D). `STAGE=train` and `STAGE=confirm` run the timed part
   and the scoring as separate jobs. Scoring is untimed and may run on any GPU. If the token file
   sits on a network filesystem, set `LOCAL_DATA_DIR` to a node-local disk: random reads over
   the network slowed pretraining ~9x on one cluster.
4. **Report every run.** A run that crashed for infrastructure reasons (out of memory,
   preemption, a lost machine) may be rerun under the same seed if the failed attempt goes in
   `extra/` with a note. Dropping a run disqualifies the record.
5. **Validate and open a PR:**

   ```bash
   uv run python scripts/validate_record_v3.py records/v3/track_d/<NNN>_<slug> --compare-to <current holder>
   ```

   A new record must beat the holder's mean test score with a one-sided Welch t-test at
   p < 0.05. Add the record to the leaderboard table in [README.md](README.md).

Running 5 seeds costs ~25-30 H100-hours: roughly $100-200 at cloud on-demand prices.
`scripts/aws/spot_launch.sh` (`V3=1`) runs one seed per self-terminating AWS instance and pushes
the results to S3.

## Code changes

* Run `uv sync && uv run pytest -q && uv run ruff check .` before pushing.
* The verifier, the prompt, the ladders and the scorer (`countdown.py`, `ladder.py`,
  `ladder_confirm.py`, `data/`) define the benchmark. Changing them means a new benchmark
  version, and changes to `data/` fail CI by design.
* Trainer changes are fine. A speedup should come with a record that demonstrates it, or be off
  by default.

## Good first records

* **Track D:** the plateau at ~0.17 is the open problem. Things not tried yet: process rewards
  for correct intermediate arithmetic, learned critics, other architectures, better use of the
  6 h split, math-aware filtering of FineWeb-Edu (it has almost no written arithmetic:
  0.005 equations per document in the baseline shards).
* **Track A:** training on the ladder mix with answer-constrained decoding roughly doubled the
  dev score in pilots (record 001). Tuning around it is cheap to try.
* Both tracks: anything that gets more learning out of each GPU hour (rollout throughput,
  dynamic sampling, curriculum).
