"""Score a checkpoint on the DEV ladder with the reference scorer (research only, never the test ladder).

    python scripts/research/dev_ladder_eval.py --model <dir> --out <json> [--temperature T] [--constrain] [--limit-per-level N]

Runs `rlvr_speedrun.ladder_confirm` unchanged (16 samples per puzzle, 1,024 tokens, the same
flags) with its puzzle file pointed at `data/ladder_v3_dev.jsonl`, so sampler choices can be
tuned on the dev ladder (RULES_v3: the test ladder is only for a record's final score). The
output's "test_set" field names the dev file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlvr_speedrun import benchmark, ladder_confirm  # noqa: E402

if __name__ == "__main__":
    benchmark.LADDER_TEST_PATH = benchmark.LADDER_DEV_PATH
    ladder_confirm.main()
