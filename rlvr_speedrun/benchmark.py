"""The benchmark protocol in one place (RULES.md). Standard library only: the record
validator imports this without torch.

Version 2 (records 006+) adds a separate confirmation set that judges every record,
an out-of-distribution report set, a training stream that excludes all held-out
puzzles, and N >= 10 seeds. Records 000-005 were run under version 1.
"""

from __future__ import annotations

from pathlib import Path

BENCHMARK_VERSION = 2
FIRST_V2_RECORD = 6

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Stopping set: the trainer evaluates on it during the timed run and stops at the first
# eval with pass_rate >= its `target_solve_rate`. Frozen since v1.
EVAL_PATH = Path("data/countdown_eval.jsonl")
EVAL_SEED = 20240601
EVAL_SIZE = 2000

# Confirmation set: scored once on each seed's final checkpoint, never during training.
# The record counts only if its mean confirmation pass rate reaches THRESHOLD.
CONFIRM_PATH = Path("data/countdown_confirm.jsonl")
CONFIRM_SEED = 20260923
CONFIRM_SIZE = 4000

# Out-of-distribution report set (5 numbers): reported, not scored.
OOD_PATH = Path("data/countdown_ood5.jsonl")
OOD_SEED = 20260924
OOD_SIZE = 1000
OOD_N_NUMBERS = 5

# Fixed evaluation protocol for both the timed eval and the confirmation eval.
FEW_SHOT = 3
MAX_EVAL_NEW_TOKENS = 64

# v2 threshold, frozen 2026-09-26 from the plateau calibration and recipe sweeps
# (docs/plateau_calibration.md): record 005's recipe is still climbing at ~0.18 after 2,000
# steps, while stronger baselines get every seed to 0.20. v1 records used 0.10.
# Changing it starts a new leaderboard table.
THRESHOLD = 0.20

MIN_SEEDS = 10

# ---------------------------------------------------------------------------------------
# Version 3 (draft, docs/RULES_v3.md): fixed-time runs scored on the difficulty ladder.
# The score of a seed is its level-weighted sampled pass@1 on the test ladder; a record is the
# mean over seeds. Track A starts from a pinned Qwen3-1.7B-Base; track D trains from scratch.
LADDER_DEV_PATH = Path("data/ladder_v3_dev.jsonl")  # in-loop evals and tuning
LADDER_TEST_PATH = Path("data/ladder_v3_test.jsonl")  # scored once per seed (rlvr_speedrun.ladder_confirm)
LADDER_LEVELS = ("L1", "L2", "L3", "L4", "L5", "L6")
LADDER_PER_LEVEL = 400
V3_SAMPLES_PER_PUZZLE = 16
V3_MAX_NEW_TOKENS = 1024
V3_MIN_SEEDS = 5
V3_TRACK_A_MODEL = "Qwen/Qwen3-1.7B-Base"
V3_TRACK_A_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
V3_TRACK_D_DATASET = ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")  # any subset
V3_TRACK_D_SHARDS = ("sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet")  # the baseline's subset (~1.7B tokens)
V3_TRACK_D_TOKENIZER = ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "59f6f375b26bde864a6ca194a9a3044570490064")
V3_TRACK_D_DATA_DIR = Path("data/track_d")  # written by scripts/prepare_track_d_data.py (gitignored)
V3_REFERENCE_SOFTWARE = {"vllm": "0.30.0", "torch": "2.13.0+cu130", "transformers": "5.17.0"}
V3_TIME_BUDGET_S = {"A": 5 * 3600, "D": 6 * 3600}  # GPU wall clock on 1x H100 80GB SXM; D includes pretraining

HELDOUT_PATHS = (EVAL_PATH, CONFIRM_PATH, OOD_PATH, LADDER_DEV_PATH, LADDER_TEST_PATH)


def puzzle_key(numbers, target) -> tuple[tuple[int, ...], int]:
    """Identity of a puzzle up to the order of its numbers."""
    return tuple(sorted(int(n) for n in numbers)), int(target)
