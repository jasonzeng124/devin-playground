import json
from pathlib import Path

from rlvr_speedrun import benchmark
from scripts.validate_record import RANKED_GPU
from scripts.validate_record_v3 import LADDER_SIZE, compare, validate

SAMPLER = {"temperature": 1.0, "constrain": True, "think_budget": 0}


def make_record(root: Path, track: str = "d", scores=(0.15, 0.16, 0.17, 0.15, 0.16), name: str = "000_baseline", **overrides) -> Path:
    record = root / "records" / "v3" / f"track_{track}" / name
    record.mkdir(parents=True)
    v3 = {"track": track.upper(), "sampler": SAMPLER}
    if track == "d":
        v3["pretrain"] = {"minutes": 112}
    config = {"model": benchmark.V3_TRACK_A_MODEL if track == "a" else "scratch", "v3": v3, **overrides.pop("config", {})}
    (record / "config.json").write_text(json.dumps(config))
    (record / "README.md").write_text("baseline\n")
    budget = benchmark.V3_TIME_BUDGET_S[track.upper()]
    for i, score in enumerate(scores):
        seed = record / "seeds" / str(i)
        seed.mkdir(parents=True)
        (seed / "train_log.jsonl").write_text("{}\n")
        if track == "d":
            (seed / "pretrain_log.jsonl").write_text("{}\n")
        clock = {"gpu": RANKED_GPU, "total_s": budget - 60, "budget_s": budget, "within_budget": True, **overrides.get("clock", {})}
        (seed / "clock.json").write_text(json.dumps(clock))
        env = {"model_revision": benchmark.V3_TRACK_A_REVISION if track == "a" else None}
        (seed / "result.json").write_text(json.dumps({"environment": env}))
        confirm = {"benchmark_version": 3, "n_puzzles": LADDER_SIZE, "samples_per_puzzle": benchmark.V3_SAMPLES_PER_PUZZLE,
                   "sampler": {**SAMPLER, "max_new_tokens": benchmark.V3_MAX_NEW_TOKENS}, "ladder_score": score, **overrides.get("confirm", {})}
        (seed / "confirm_v3.json").write_text(json.dumps(confirm))
    return record


def test_valid_records(tmp_path):
    for track in ("a", "d"):
        ok, message = validate(make_record(tmp_path, track))
        assert ok, message
        assert "mean test ladder score 0.1580" in message


def test_too_few_seeds(tmp_path):
    ok, message = validate(make_record(tmp_path, scores=(0.15, 0.16, 0.17)))
    assert not ok and "at least 5 seeds" in message


def test_unranked_smoke_allows_few_seeds_and_other_gpus(tmp_path):
    record = make_record(tmp_path, scores=(0.0,), config={"unranked": True}, clock={"gpu": "NVIDIA A10G"})
    assert validate(record)[0]


def test_over_budget_and_wrong_gpu(tmp_path):
    ok, message = validate(make_record(tmp_path, clock={"total_s": 6 * 3600 + 1, "gpu": "NVIDIA H200"}))
    assert not ok and "over budget" in message and "not the ranked" in message


def test_partial_confirm_and_wrong_sampler(tmp_path):
    ok, message = validate(make_record(tmp_path, confirm={"n_puzzles": 30, "sampler": {**SAMPLER, "constrain": False, "max_new_tokens": 1024}}))
    assert not ok and "not a full test-ladder run" in message and "confirm sampler" in message


def test_track_a_revision_pinned(tmp_path):
    record = make_record(tmp_path, "a")
    (record / "seeds" / "2" / "result.json").write_text(json.dumps({"environment": {"model_revision": "main"}}))
    ok, message = validate(record)
    assert not ok and "seed 2: model_revision" in message


def test_track_d_needs_own_pretraining(tmp_path):
    record = make_record(tmp_path)
    (record / "seeds" / "1" / "pretrain_log.jsonl").unlink()
    assert "seed 1: missing pretrain_log" in validate(record)[1]


def test_compare(tmp_path):
    old = make_record(tmp_path, scores=(0.15, 0.16, 0.17, 0.15, 0.16))
    better = make_record(tmp_path, scores=(0.20, 0.21, 0.19, 0.20, 0.22), name="001_better")
    same = make_record(tmp_path, scores=(0.16, 0.15, 0.17, 0.16, 0.15), name="002_same")
    assert compare(better, old)[0]
    assert not compare(same, old)[0]
