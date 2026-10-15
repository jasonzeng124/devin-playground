"""The held-out sets and capability-probe data are frozen: any change to them breaks
comparability of every record, so their checksums are pinned here (RULES.md,
"Forbidden"). Regenerating them on purpose means a new benchmark version."""

import hashlib
import itertools
from pathlib import Path

import pytest

from rlvr_speedrun import benchmark
from rlvr_speedrun.data import build_confirm, build_ood, heldout_keys, load_puzzles, train_puzzle_stream

ROOT = Path(__file__).resolve().parents[1]

FROZEN = {
    "data/countdown_eval.jsonl": "ca3ab9fbb06a0824bc8aec3a9136ff7ff735eae04485d248a3d27e94c049a332",
    "data/countdown_confirm.jsonl": "100b040eb7d3885c21b6e881797da0ab22e52044bd8721020f98684ae2e9715d",
    "data/countdown_ood5.jsonl": "13665afdf37d0e4174b384fff2d1a1f979a19ede0962820d11d84c4b5616855a",
    "data/capability/fineweb_edu_val.jsonl": "8d2026791506c49629559aafe4379fb6ff503d638a1e5c0b66a5d53a8044e858",
    "data/ladder_v3_dev.jsonl": "4ebf05414ce6cbcd116e5bd0ca58855e3e6d8f73ab9af7c5edd9c7d4fbf5c9aa",
    "data/ladder_v3_test.jsonl": "b38fe0123a68e988b55b9982527b20e14c2637ac15279eb1377d28c59e4083e0",
    "data/capability/mmlu_lite.jsonl": "51c8caf4dc19efe9af30f9f0f98c1ca961d10a579ff7e2c6bc691fb6f783a9a0",
}


def _keys(relpath):
    return [benchmark.puzzle_key(p.numbers, p.target) for p in load_puzzles(ROOT / relpath)]


@pytest.mark.parametrize("relpath", sorted(FROZEN))
def test_frozen_file_unchanged(relpath):
    digest = hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()
    assert digest == FROZEN[relpath], f"{relpath} changed; the frozen data must not be edited"


def test_heldout_set_sizes():
    assert len(_keys(benchmark.EVAL_PATH)) == benchmark.EVAL_SIZE
    assert len(set(_keys(benchmark.EVAL_PATH))) == benchmark.EVAL_SIZE - 2  # two duplicates, frozen in v1
    assert len(set(_keys(benchmark.CONFIRM_PATH))) == benchmark.CONFIRM_SIZE
    ood = load_puzzles(ROOT / benchmark.OOD_PATH)
    assert len({benchmark.puzzle_key(p.numbers, p.target) for p in ood}) == benchmark.OOD_SIZE
    assert all(len(p.numbers) == benchmark.OOD_N_NUMBERS for p in ood)


def test_confirmation_set_is_disjoint_from_eval():
    assert not set(_keys(benchmark.EVAL_PATH)) & set(_keys(benchmark.CONFIRM_PATH))


def test_builders_reproduce_committed_sets(tmp_path):
    assert build_confirm(tmp_path / "c.jsonl").read_bytes() == (ROOT / benchmark.CONFIRM_PATH).read_bytes()
    assert build_ood(tmp_path / "o.jsonl").read_bytes() == (ROOT / benchmark.OOD_PATH).read_bytes()


def test_training_stream_never_yields_heldout_puzzles():
    keys = heldout_keys()
    v2 = {benchmark.puzzle_key(p.numbers, p.target) for path in benchmark.HELDOUT_PATHS[:3] for p in load_puzzles(ROOT / path)}
    assert len(v2) == benchmark.EVAL_SIZE - 2 + benchmark.CONFIRM_SIZE + benchmark.OOD_SIZE
    ladder = {benchmark.puzzle_key(p.numbers, p.target) for path in (benchmark.LADDER_DEV_PATH, benchmark.LADDER_TEST_PATH)
              for p in load_puzzles(ROOT / path)}
    assert keys == v2 | ladder
    drawn = [benchmark.puzzle_key(p.numbers, p.target) for p in itertools.islice(train_puzzle_stream(0), 20_000)]
    assert not set(drawn) & keys
    # without the filter the raw generator does hit the eval set, which is why it exists
    raw = [benchmark.puzzle_key(p.numbers, p.target) for p in itertools.islice(train_puzzle_stream(0, exclude_heldout=False), 20_000)]
    assert set(raw) & keys


def test_ladder_sets_are_frozen_disjoint_and_balanced():
    from rlvr_speedrun.ladder import (
        LADDER_DEV_SEED,
        LADDER_TEST_SEED,
        build_ladder_eval,
        ladder_keys,
        load_ladder,
        puzzle_key,
    )

    dev, test = load_ladder(benchmark.LADDER_DEV_PATH), load_ladder(benchmark.LADDER_TEST_PATH)
    assert not {puzzle_key(p) for p in dev} & {puzzle_key(p) for p in test}
    for puzzles in (dev, test):
        assert [p.level for p in puzzles] == [lv for lv in benchmark.LADDER_LEVELS for _ in range(benchmark.LADDER_PER_LEVEL)]
    assert build_ladder_eval(seed=LADDER_DEV_SEED) == dev
    assert build_ladder_eval(seed=LADDER_TEST_SEED, exclude=ladder_keys(benchmark.LADDER_DEV_PATH)) == test
