import json
from pathlib import Path

import pytest

from scripts.validate_record import (
    RANKED_GPU,
    compare_records,
    permutation_one_sided,
    summary_stats,
    t_cdf,
    t_ppf,
    validate_record,
    welch_one_sided,
)

ENV = {"gpu": RANKED_GPU, "cuda": "12.4", "torch": "2.5.1", "transformers": "4.57.6", "python": "3.11.10", "git_commit": None}


def make_record(
    root: Path,
    name: str,
    times: list[float | None],
    probes: bool = True,
    env: dict | None = ENV,
    declare_seeds: bool = True,
    config_extra: dict | None = None,
    seed_names: list[str] | None = None,
) -> Path:
    record = root / "track_a" / name
    config = {"track": "track_a", "model": "Qwen/Qwen2.5-0.5B", "hardware": "1x H100", "provider": "modal", "target_solve_rate": 0.1, "eval_limit": 2000}
    if declare_seeds:
        config["seeds"] = list(range(len(times)))
    config.update(config_extra or {})
    record.mkdir(parents=True)
    (record / "README.md").write_text("test\n")
    (record / "config.json").write_text(json.dumps(config) + "\n")
    names = seed_names or [str(i) for i in range(len(times))]
    for seed, time in zip(names, times):
        seed_dir = record / "seeds" / seed
        seed_dir.mkdir(parents=True)
        result = {"final_pass_rate": 0.11, "time_to_threshold_s": time, "total_steps": 300}
        if env is not None:
            result["environment"] = env
        (seed_dir / "result.json").write_text(json.dumps(result) + "\n")
        (seed_dir / "train_log.jsonl").write_text('{"step": 1}\n')
        if probes:
            (seed_dir / "probe.json").write_text(json.dumps({"fineweb_loss_delta": 0.001, "mmlu_lite_acc_delta": 0.0}) + "\n")
    return record


def test_validate_record_smoke_fixture(tmp_path):
    record = tmp_path / "track_a" / "001_test"
    (record / "seeds" / "0").mkdir(parents=True)
    (record / "README.md").write_text("test\n")
    (record / "config.json").write_text("{}\n")
    (record / "seeds" / "0" / "result.json").write_text(
        json.dumps({"final_pass_rate": 0.5, "time_to_threshold_s": None}) + "\n"
    )
    (record / "seeds" / "0" / "train_log.jsonl").write_text('{"step": 1}\n')
    valid, message = validate_record(record, allow_fewer_seeds=True)
    assert valid
    assert "pass_rate" in message
    invalid, message = validate_record(record)
    assert not invalid
    assert "at least 3" in message


def test_full_record_passes(tmp_path):
    record = make_record(tmp_path, "006_good", [400.0, 500.0, 450.0])
    valid, message = validate_record(record)
    assert valid, message
    assert "n=3" in message
    assert RANKED_GPU in message


def test_t_distribution_matches_tables():
    assert t_cdf(2.132, 4) == pytest.approx(0.95, abs=1e-3)
    assert t_ppf(0.975, 2) == pytest.approx(4.303, abs=1e-3)
    assert t_ppf(0.975, 9) == pytest.approx(2.262, abs=1e-3)
    mean, std, low, high = summary_stats([1.0, 2.0, 3.0])
    assert (mean, std) == (2.0, 1.0)
    assert low == pytest.approx(2.0 - 4.303 / 3**0.5, abs=1e-3)
    assert high == pytest.approx(2.0 + 4.303 / 3**0.5, abs=1e-3)


def test_welch_and_permutation():
    new, old = [400.0, 420.0, 410.0], [600.0, 640.0, 620.0]
    t, _df, p = welch_one_sided(new, old)
    assert t < 0 and p < 0.01
    p_perm, exact = permutation_one_sided(new, old)
    assert exact and p_perm == pytest.approx(1 / 20)
    # no evidence when the new record is slower
    _, _, p_slow = welch_one_sided(old, new)
    assert p_slow > 0.99
    assert permutation_one_sided(old, new)[0] == pytest.approx(1.0)


def test_compare_records_significant_and_not(tmp_path):
    old = make_record(tmp_path, "006_old", [600.0, 640.0, 620.0])
    new = make_record(tmp_path, "007_new", [400.0, 420.0, 410.0])
    significant, report = compare_records(new, old)
    assert significant
    assert "BEATS" in report and "welch" in report and "permutation" in report
    close = make_record(tmp_path, "008_close", [590.0, 630.0, 610.0])
    significant, report = compare_records(close, old)
    assert not significant
    assert "does not beat" in report


def test_compare_records_rejects_incompatible(tmp_path):
    old = make_record(tmp_path, "006_old", [600.0, 640.0, 620.0])
    other_model = make_record(tmp_path, "007_model", [400.0, 420.0, 410.0], config_extra={"model": "other/model"})
    ok, report = compare_records(other_model, old)
    assert not ok and "different base models" in report
    other_threshold = make_record(tmp_path, "008_thr", [400.0, 420.0, 410.0], config_extra={"target_solve_rate": 0.2})
    ok, report = compare_records(other_threshold, old)
    assert not ok and "different thresholds" in report
    censored = make_record(tmp_path, "009_cens", [400.0, None, 410.0])
    ok, report = compare_records(censored, old)
    assert not ok and "censored" in report
    pcie = make_record(tmp_path, "010_pcie", [400.0, 420.0, 410.0], env={**ENV, "gpu": "NVIDIA H100 PCIe"})
    ok, report = compare_records(pcie, old)
    assert not ok and "ranked hardware" in report


def test_censored_seed_invalidates_record(tmp_path):
    record = make_record(tmp_path, "006_cens", [400.0, None, 410.0])
    valid, message = validate_record(record)
    assert not valid
    assert "censored seeds" in message and "['1']" in message
    valid, message = validate_record(record, allow_fewer_seeds=True)
    assert valid


def test_probe_required_after_record_003(tmp_path):
    exempt = make_record(tmp_path, "003_legacy", [400.0, 500.0, 450.0], probes=False, env=None)
    valid, message = validate_record(exempt)
    assert valid and "exempt" in message
    new = make_record(tmp_path, "006_noprobe", [400.0, 500.0, 450.0], probes=False)
    valid, message = validate_record(new)
    assert not valid and "missing probe.json" in message


def test_failing_probe_fails_record(tmp_path):
    record = make_record(tmp_path, "006_hack", [400.0, 500.0, 450.0])
    (record / "seeds" / "2" / "probe.json").write_text(json.dumps({"fineweb_loss_delta": 0.2, "mmlu_lite_acc_delta": -0.1}) + "\n")
    valid, message = validate_record(record)
    assert not valid and "seed 2" in message and "FAIL" in message


def test_environment_required_after_record_005(tmp_path):
    legacy = make_record(tmp_path, "005_legacy", [400.0, 500.0, 450.0], env=None)
    assert validate_record(legacy)[0]
    new = make_record(tmp_path, "006_noenv", [400.0, 500.0, 450.0], env=None)
    valid, message = validate_record(new)
    assert not valid and "environment" in message


def test_unranked_gpu_fails_ranked_validation(tmp_path):
    record = make_record(tmp_path, "006_pcie", [400.0, 500.0, 450.0], env={**ENV, "gpu": "NVIDIA H100 PCIe"})
    valid, message = validate_record(record)
    assert not valid and "UNRANKED" in message
    assert validate_record(record, allow_fewer_seeds=True)[0]


def test_single_provider_required_after_record_005(tmp_path):
    legacy = make_record(tmp_path, "005_legacy", [400.0, 500.0, 450.0], config_extra={"provider": {"0": "modal", "1": "modal", "2": "runpod"}})
    valid, message = validate_record(legacy)
    assert valid and "MIXED" in message and "grandfathered" in message
    mixed = make_record(tmp_path, "006_mixed", [400.0, 500.0, 450.0], config_extra={"provider": {"0": "modal", "1": "modal", "2": "runpod"}})
    valid, message = validate_record(mixed)
    assert not valid and "one provider" in message
    undeclared = make_record(tmp_path, "007_noprov", [400.0, 500.0, 450.0], config_extra={"provider": None})
    valid, message = validate_record(undeclared)
    assert not valid and "declare 'provider'" in message
    partial = make_record(tmp_path, "008_partial", [400.0, 500.0, 450.0], config_extra={"provider": {"0": "modal"}})
    valid, message = validate_record(partial)
    assert not valid and "declare 'provider'" in message
    runpod = make_record(tmp_path, "009_runpod", [400.0, 500.0, 450.0], config_extra={"provider": "RunPod"})
    valid, message = validate_record(runpod)
    assert valid and "provider: runpod" in message


def test_compare_reports_same_provider_subsets(tmp_path):
    split = {str(i): ("modal" if i < 3 else "runpod") for i in range(6)}
    old = make_record(tmp_path, "004_old", [600.0, 640.0, 620.0, 900.0, 950.0, 1000.0], config_extra={"provider": split})
    new = make_record(tmp_path, "005_new", [400.0, 420.0, 410.0, 430.0, 450.0, 440.0], config_extra={"provider": split})
    significant, report = compare_records(new, old)
    assert significant
    assert "span providers" in report and "modal: n=3 vs 3" in report and "runpod: n=3 vs 3" in report
    other = make_record(tmp_path, "006_lambda", [400.0, 420.0, 410.0], config_extra={"provider": "lambda"})
    same = make_record(tmp_path, "007_modal", [600.0, 640.0, 620.0])
    _, report = compare_records(other, same)
    assert "no provider in common" in report
    _, report = compare_records(make_record(tmp_path, "008_modal", [400.0, 420.0, 410.0]), same)
    assert "span providers" not in report


def test_seed_protocol(tmp_path):
    undeclared = make_record(tmp_path, "006_undeclared", [400.0, 500.0, 450.0], declare_seeds=False)
    valid, message = validate_record(undeclared)
    assert not valid and "declare 'seeds'" in message
    picked = make_record(tmp_path, "007_picked", [400.0, 500.0, 450.0], config_extra={"seeds": [0, 1, 7]}, seed_names=["0", "1", "7"])
    valid, message = validate_record(picked)
    assert not valid and "0..N-1" in message
    mismatch = make_record(tmp_path, "008_mismatch", [400.0, 500.0, 450.0], seed_names=["0", "1", "3"])
    valid, message = validate_record(mismatch)
    assert not valid and "exactly the declared seeds" in message


def test_validate_all_records_ranked_vs_unranked(tmp_path):
    from scripts.validate_all_records import validate_all

    records = tmp_path / "records"
    make_record(records, "006_ranked", [400.0, 500.0, 450.0])
    smoke = make_record(records, "000_smoke", [None], config_extra={"unranked": True, "seeds": None})
    readme = tmp_path / "README.md"

    readme.write_text("| 1 | [006_ranked](records/track_a/006_ranked) |\n")
    ok, lines = validate_all(records, readme)
    assert ok, "\n".join(lines)
    assert any("000_smoke (unranked): OK" in line for line in lines)
    assert any("006_ranked (ranked): OK" in line for line in lines)

    readme.write_text("empty leaderboard\n")
    ok, lines = validate_all(records, readme)
    assert not ok and any("not linked" in line for line in lines)

    (smoke / "config.json").write_text(json.dumps({"track": "track_a"}) + "\n")
    ok, lines = validate_all(records, None)
    assert not ok and any("000_smoke (ranked): FAIL" in line for line in lines)
