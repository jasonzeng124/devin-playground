# Brief for the research agent on Engaging

You are Claude Code running unattended on an MIT Engaging login node (tmux session `research`,
checkout `~/rlvr-screens`, branch `research/2026-10`), from 2026-10-19 to 2026-11-19. The
maintainer (GitHub @jasonzeng124, cluster user `zjason`) is mostly away. A second Claude session
on the maintainer's laptop checks in on you when it can. It may leave notes in
`docs/research/INBOX.md`; read them every wake.

## Goal

Maintainer, 2026-10-19: "your goal should be to log the best records under the current rules.
you should try to run records for any substantial improvements". The rules are `docs/RULES_v3.md`,
frozen on 2026-10-19 with external data allowed: any public dataset that isn't
Countdown-specific. Read `docs/research/PLAN.md` for:
- the operating rules;
- the promotion rule (when a screen becomes a record attempt);
- the agenda.

Read `docs/research/LOG.md` for what has been tried. Records are the measure of the month.

## Hard rules (don't break these, whatever a result suggests)

- **Login node:**
  - Never run GPU work or heavy CPU work on the login node. That means no training, no model
    loading, no tokenizing big datasets, and nothing longer than a few minutes of CPU.
  - Everything heavy goes through Slurm. Data preparation runs as CPU jobs on `mit_normal`.
  - `uv run pytest -q` and `ruff` on the login node are fine (about 30 s).
- **At most 6 GPUs at once**, counting running and pending GPU jobs that could run together.
  `mit_normal_gpu` allows 2 per user (6 h max); the rest go on `mit_preemptable`.
- **No AWS, no Modal,** and no other paid service.
- **Never touch `~/rlvr`.** Records 000/001 ran from it.
- **No other users' jobs or files:**
  - `scancel` only job IDs you have listed and that belong to you; never `scancel -u`.
  - No `pkill -f`. Avoid `quota -s`.
- **Deletions:** only remove files you created, by exact path, after looking at them. Keep the
  pool (`~/orcd/pool`, 1 TB) under 80 % (`df -h ~/orcd/pool`). RL screens set `"no_save": true`
  unless their weights are needed.
- **Test ladder** (`data/ladder_v3_test.jsonl`): only record runs touch it, through
  `scripts/run_seed_v3.sh`. Screens, tuning and sampler choices use the dev ladder only.
  - Each record's N is fixed in advance (5), and every seed counts.
  - No re-running, dropping or cherry-picking seeds.
  - A failed seed goes to `<record>/extra/` and is reported, as RULES_v3 says.
- **Data rules:** no Countdown/24-game data, no other models (no teacher, data generator or
  reward model), and no solver output. Arithmetic text you generate is not allowed either;
  public datasets are fine, as published. When unsure whether something is allowed, don't do it.
  Write the question in `docs/research/QUESTIONS.md` and work on something else.
- **Git:**
  - Commit on `research/2026-10` only, as
    `git -c user.name=zjason -c user.email=zjason@mit.edu commit`, with the trailer
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Stage explicit paths (never `git add -A`).
  - This checkout has no push credentials. Commit locally; the laptop session fetches and pushes.
  - Never rewrite history, force anything, or switch branches.
  - No pull requests and no posting anywhere public.
- **Code on the timed path:** new options are off by default, so existing configs behave as
  before. Run `uv run ruff check . && uv run pytest -q` before committing code. Running jobs use
  this checkout, so don't edit a file a running job is about to read in a way that changes its
  meaning. Add new configs instead of changing queued ones.
- **Spending:** don't spawn subagents, and keep quiet wakes short (a status check, then sleep).

## Infrastructure

- **Queues:**
  - Work queue files are `~/rlvr-screens/queue/<name>.txt`, one `<config dir> <seed>` per line.
  - Workers pop lines under flock, run them, and log to `queue/worker.log` and `queue/done.txt`.
  - Status: `squeue -u $USER`, `tail queue/worker.log`, and
    `~/rlvr-venv/bin/python scripts/summarize_screens.py <arm dirs>`.
- **Screen workers** (dev ladder only; `scripts/slurm/screen_seed.sh`). Submit from
  `~/rlvr-screens/slurm_logs`:

      sbatch -p mit_preemptable --time=2-00:00:00 --time-min=05:30:00 --gres=gpu:h100:1 \
        --exclude=node1708,node1709,node4508 \
        --export=ALL,RLVR_DIR=$HOME/rlvr-screens,RUNNER=scripts/slurm/screen_seed.sh,QUEUE=<name>,RESERVE_MIN=330,REQUIRE_GPU="NVIDIA H100 80GB HBM3",VENV=$HOME/orcd/pool/vllm-venv,HF_HUB_OFFLINE=1 \
        ../scripts/slurm/queue_worker.sbatch

  On `mit_normal_gpu`, use `--time=06:00:00`. Track A screens must use H100 80GB HBM3, so they
  stay comparable with the records.
- **Record workers:** the same command with `QUEUE=record-<slug>` and
  `RUNNER=scripts/run_seed_v3.sh`.
  - Use `mit_normal_gpu` and the HBM3 requirement; preemptable is fine for track D.
  - Before queueing, create `records/v3/track_{a,d}/<NNN>_<slug>/config.json` and commit it.
  - After all 5 seeds: write the README (follow the existing records and CONTRIBUTING.md), run
    `uv run python scripts/validate_record_v3.py <dir> --compare-to <holder>`, add the record to
    the README leaderboard if it beats the holder, and commit.
- **Maintenance:** Engaging has monthly maintenance (the next one is 2026-10-20 00:00-21:00
  EDT). Jobs whose time limit crosses a reservation won't start, so `--time-min` lets them start
  with less time.
- **Models:** pre-downloaded in `~/.cache/huggingface`, and compute nodes run with
  `HF_HUB_OFFLINE=1`. Download new public datasets with a CPU job (`HF_HUB_OFFLINE=0`) into
  `~/orcd/pool/data/`.

## Each wake

1. Read `docs/research/INBOX.md`, then the end of `LOG.md`, then `PLAN.md` when deciding.
2. Check jobs and screens. Requeue items that failed for host reasons (once), and note them.
3. When an arm or batch finishes:
   - add a dated LOG entry: the summarize table and what it means, in plain words;
   - decide the next experiments from PLAN's agenda and the results;
   - start a record attempt whenever the promotion rule is met;
   - commit.
4. Keep the GPUs busy: at least a day of queued work, never more than 6 GPUs at once.
5. Update `docs/research/STATUS.md`, a short current-state page: running jobs, queues, the best
   result so far, the next decision and when. Commit it whenever it changes materially.
6. Schedule the next wake by what you're waiting for: about 1-2 h while screens run, longer when
   nothing can finish sooner. Stop the loop after 2026-11-19, once the final LOG summary is
   written.
