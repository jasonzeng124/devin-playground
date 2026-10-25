import json
from pathlib import Path

from rlvr_speedrun import benchmark
from scripts.validate_record_v3 import H100_GPU, LADDER_SIZE, LEVELS, RANKED_GPU, compare, validate

SAMPLER = {"temperature": 1.0, "constrain": True, "think_budget": 0}


def make_record(root: Path, track: str = "d", scores=(0.15, 0.16, 0.17, 0.15, 0.16), name: str = "010_baseline", **overrides) -> Path:
    """A record that follows the rules; `config`, `clock` and `confirm` overrides break them on purpose."""
    record = root / "records" / "v3" / f"track_{track}" / name
    record.mkdir(parents=True)
    v3 = {"track": track.upper(), "sampler": SAMPLER}
    if track == "d":
        v3["pretrain"] = {"minutes": 112}
    config = {"model": benchmark.V3_TRACK_A_MODEL if track == "a" else "scratch", "lr": 2e-6, "eval_set": str(benchmark.LADDER_DEV_PATH),
              "v3": v3, **overrides.pop("config", {})}
    (record / "config.json").write_text(json.dumps(config))
    (record / "README.md").write_text("baseline\n")
    budget = benchmark.V3_TIME_BUDGET_S[track.upper()]
    weights = f"/ckpt/records/v3/track_{track}/{name}/seeds"
    for i, score in enumerate(scores):
        seed = record / "seeds" / str(i)
        seed.mkdir(parents=True)
        seed_config = {k: v for k, v in config.items() if k not in ("v3", "unranked", "description")}
        seed_config["seed"] = i
        if track == "d":
            seed_config["model"] = f"{weights}/{i}/pretrained"
            (seed / "pretrain_log.jsonl").write_text(json.dumps({"seed": i, "loss": 3.0}) + "\n")
        (seed / "config.json").write_text(json.dumps(seed_config))
        (seed / "train_log.jsonl").write_text(json.dumps({"step": 1}) + "\n")
        (seed / "eval_log.jsonl").write_text(json.dumps({"wall_s": 110.0, "train_s": 100.0}) + "\n")
        pre = 6000.0 if track == "d" else 0.0
        clock = {"gpu": RANKED_GPU, "pretrain_s": pre, "grpo_wall_s": budget - 60 - pre + 10, "grpo_eval_s": 10.0,
                 "total_s": budget - 60, "budget_s": budget, "within_budget": True, **overrides.get("clock", {})}
        (seed / "clock.json").write_text(json.dumps(clock))
        env = {"model_revision": benchmark.V3_TRACK_A_REVISION if track == "a" else None, "gpu": clock["gpu"], "gpu_count": 1, "world_size": 1}
        (seed / "result.json").write_text(json.dumps({"environment": env}))
        confirm = {"benchmark_version": 3, "n_puzzles": LADDER_SIZE, "samples_per_puzzle": benchmark.V3_SAMPLES_PER_PUZZLE,
                   "sampler": {**SAMPLER, "max_new_tokens": benchmark.V3_MAX_NEW_TOKENS}, "ladder_score": score,
                   "pass@1_by_level": dict.fromkeys(LEVELS, score), "test_set": str(benchmark.LADDER_TEST_PATH),
                   "model": f"{weights}/{i}/final", **overrides.get("confirm", {})}
        (seed / "confirm_v3.json").write_text(json.dumps(confirm))
    return record


def edit_json(path: Path, **changes) -> None:
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


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
    ok, message = validate(make_record(tmp_path, clock={"total_s": 6 * 3600 + 1, "gpu": "NVIDIA H200 NVL"}))
    assert not ok and "over budget" in message and "not the ranked" in message


def test_partial_confirm_and_wrong_sampler(tmp_path):
    ok, message = validate(make_record(tmp_path, confirm={"n_puzzles": 30, "sampler": {**SAMPLER, "constrain": False, "max_new_tokens": 1024}}))
    assert not ok and "not a full test-ladder run" in message and "confirm sampler" in message


def test_track_a_revision_pinned(tmp_path):
    record = make_record(tmp_path, "a")
    result = record / "seeds" / "2" / "result.json"
    edit_json(result, environment={**json.loads(result.read_text())["environment"], "model_revision": "main"})
    ok, message = validate(record)
    assert not ok and "seed 2: model_revision" in message


def test_track_d_needs_own_pretraining(tmp_path):
    record = make_record(tmp_path)
    seeds = record / "seeds"
    (seeds / "1" / "pretrain_log.jsonl").unlink()
    (seeds / "3" / "pretrain_log.jsonl").write_text((seeds / "2" / "pretrain_log.jsonl").read_text())
    edit_json(seeds / "4" / "config.json", model="/ckpt/records/v3/track_d/000_baseline/seeds/0/pretrained")
    ok, message = validate(record)
    assert not ok
    assert "seed 1: missing pretrain_log" in message
    assert "seed 3: same pretrain_log.jsonl as seed 2" in message
    assert "seed 4: GRPO must start from this seed's own pretrained model" in message


def test_rule_checks_catch_a_fabricated_record(tmp_path):
    """The kind of record the 2026-10 audit fabricated, which the format-only checks passed."""
    record = make_record(tmp_path, "a")
    seeds = record / "seeds"
    edit_json(seeds / "0" / "config.json", eval_set=str(benchmark.LADDER_TEST_PATH))
    edit_json(seeds / "1" / "confirm_v3.json", ladder_score=0.52)
    edit_json(seeds / "2" / "clock.json", grpo_eval_s=9999.0)
    edit_json(seeds / "3" / "confirm_v3.json", test_set=str(benchmark.LADDER_DEV_PATH), model="/somewhere/seeds/0/final")
    result = seeds / "4" / "result.json"
    edit_json(result, environment={**json.loads(result.read_text())["environment"], "gpu_count": 8, "world_size": 8})
    edit_json(seeds / "4" / "config.json", lr=1e-4, resume_every_s=1200.0)
    ok, message = validate(record)
    assert not ok
    for expected in ("seed 0: in-training evals use the test ladder",
                     "seed 1: ladder_score 0.52 != the mean of its levels",
                     "seed 2: clock total_s", "seed 2: grpo_eval_s 9999 != the in-loop eval time",
                     "seed 3: confirm_v3.json was not scored on the test ladder",
                     "seed 3: confirm_v3.json scored '/somewhere/seeds/0/final', not this seed's final weights",
                     "seed 4: trained with gpu_count 8", "seed 4: config.json has lr=0.0001", "seed 4: resuming was on"):
        assert expected in message, expected


def test_compare(tmp_path):
    old = make_record(tmp_path, scores=(0.15, 0.16, 0.17, 0.15, 0.16))
    better = make_record(tmp_path, scores=(0.20, 0.21, 0.19, 0.20, 0.22), name="001_better")
    same = make_record(tmp_path, scores=(0.16, 0.15, 0.17, 0.16, 0.15), name="002_same")
    assert compare(better, old)[0]
    assert not compare(same, old)[0]


def test_h100_era_records_keep_their_gpu(tmp_path):
    old = make_record(tmp_path, name="000_baseline", clock={"gpu": H100_GPU})
    assert validate(old)[0]
    edit_json(old / "seeds" / "0" / "clock.json", gpu=RANKED_GPU)
    ok, message = validate(old)
    assert not ok and f"not the ranked {H100_GPU!r}" in message
    ok, message = validate(make_record(tmp_path, name="011_h100", clock={"gpu": H100_GPU}))
    assert not ok and f"not the ranked {RANKED_GPU!r}" in message


def test_compare_refuses_mixed_gpus(tmp_path):
    old = make_record(tmp_path, name="000_baseline", clock={"gpu": H100_GPU})
    new = make_record(tmp_path, scores=(0.3, 0.31, 0.32, 0.3, 0.31), name="012_h200")
    ok, message = compare(new, old)
    assert not ok and "different GPUs" in message
