import hashlib
import itertools
import json
from pathlib import Path

import pytest

from rlvr_speedrun import benchmark
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

ENV = {"gpu": RANKED_GPU, "cuda": "12.4", "torch": "2.5.1", "transformers": "4.57.6", "python": "3.11.10", "git_commit": None, "model_revision": "abc123"}
ROOT = Path(__file__).resolve().parents[1]


def ten(*values):
    """Cycle `values` to the v2 minimum of 10 seeds."""
    return list(itertools.islice(itertools.cycle(values), benchmark.MIN_SEEDS))


def confirm_json(pass_rate: float = benchmark.THRESHOLD + 0.005) -> dict:
    def block(relpath, n, rate):
        return {"pass_rate": rate, "n": n, "data_sha256": hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()}

    return {
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "protocol": {"few_shot": 3, "max_new_tokens": 64, "decoding": "greedy"},
        "confirm": block(benchmark.CONFIRM_PATH, benchmark.CONFIRM_SIZE, pass_rate),
        "ood": block(benchmark.OOD_PATH, benchmark.OOD_SIZE, 0.01),
    }


def make_record(
    root: Path,
    name: str,
    times: list[float | None],
    probes: bool = True,
    env: dict | None = ENV,
    declare_seeds: bool = True,
    config_extra: dict | None = None,
    seed_names: list[str] | None = None,
    confirm: float | None = benchmark.THRESHOLD + 0.005,
) -> Path:
    record = root / "track_a" / name
    config = {"track": "track_a", "model": "Qwen/Qwen2.5-0.5B", "hardware": "1x H100", "provider": "modal", "target_solve_rate": benchmark.THRESHOLD, "eval_limit": 2000, "few_shot": 3, "max_new_tokens": 64}
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
        result = {"final_pass_rate": benchmark.THRESHOLD + 0.01, "time_to_threshold_s": time, "total_steps": 300, "benchmark_version": benchmark.BENCHMARK_VERSION}
        if env is not None:
            result["environment"] = env
        (seed_dir / "result.json").write_text(json.dumps(result) + "\n")
        (seed_dir / "train_log.jsonl").write_text('{"step": 1}\n')
        if probes:
            (seed_dir / "probe.json").write_text(json.dumps({"fineweb_loss_delta": 0.001, "mmlu_lite_acc_delta": 0.0}) + "\n")
        if confirm is not None:
            (seed_dir / "confirm.json").write_text(json.dumps(confirm_json(confirm)) + "\n")
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
    record = make_record(tmp_path, "006_good", ten(400.0, 500.0, 450.0))
    valid, message = validate_record(record)
    assert valid, message
    assert "n=10" in message
    assert RANKED_GPU in message
    assert "steps_to_threshold" in message and "s_per_step" in message and "confirm pass_rate" in message


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
    old = make_record(tmp_path, "006_old", ten(600.0, 640.0, 620.0))
    new = make_record(tmp_path, "007_new", ten(400.0, 420.0, 410.0))
    significant, report = compare_records(new, old)
    assert significant
    assert "BEATS" in report and "welch" in report and "permutation" in report
    close = make_record(tmp_path, "008_close", ten(590.0, 630.0, 610.0))
    significant, report = compare_records(close, old)
    assert not significant
    assert "does not beat" in report


def test_compare_records_rejects_incompatible(tmp_path):
    old = make_record(tmp_path, "006_old", ten(600.0, 640.0, 620.0))
    other_model = make_record(tmp_path, "007_model", ten(400.0, 420.0, 410.0), config_extra={"model": "other/model"})
    ok, report = compare_records(other_model, old)
    assert not ok and "different base models" in report
    later_stop = make_record(tmp_path, "008_later", ten(400.0, 420.0, 410.0), config_extra={"target_solve_rate": benchmark.THRESHOLD + 0.02})
    ok, report = compare_records(later_stop, old)
    assert "different thresholds" not in report  # v2: stopping later is a recipe choice
    v1_old = make_record(tmp_path, "004_v1", [600.0, 640.0, 620.0], confirm=None)
    v1_other = make_record(tmp_path, "005_v1thr", [400.0, 420.0, 410.0], confirm=None, config_extra={"target_solve_rate": 0.3})
    ok, report = compare_records(v1_other, v1_old)
    assert not ok and "different thresholds" in report
    censored = make_record(tmp_path, "009_cens", ten(400.0, None, 410.0))
    ok, report = compare_records(censored, old)
    assert not ok and "censored" in report
    pcie = make_record(tmp_path, "010_pcie", ten(400.0, 420.0, 410.0), env={**ENV, "gpu": "NVIDIA H100 PCIe"})
    ok, report = compare_records(pcie, old)
    assert not ok and "ranked hardware" in report


def test_censored_seed_invalidates_record(tmp_path):
    record = make_record(tmp_path, "006_cens", ten(400.0, None, 410.0))
    valid, message = validate_record(record)
    assert not valid
    assert "censored seeds" in message and "['1', '4', '7']" in message
    valid, message = validate_record(record, allow_fewer_seeds=True)
    assert valid


def test_probe_required_after_record_003(tmp_path):
    exempt = make_record(tmp_path, "003_legacy", [400.0, 500.0, 450.0], probes=False, env=None)
    valid, message = validate_record(exempt)
    assert valid and "exempt" in message
    new = make_record(tmp_path, "006_noprobe", ten(400.0, 500.0, 450.0), probes=False)
    valid, message = validate_record(new)
    assert not valid and "missing probe.json" in message


def test_failing_probe_fails_record(tmp_path):
    record = make_record(tmp_path, "006_hack", ten(400.0, 500.0, 450.0))
    (record / "seeds" / "2" / "probe.json").write_text(json.dumps({"fineweb_loss_delta": 0.2, "mmlu_lite_acc_delta": -0.1}) + "\n")
    valid, message = validate_record(record)
    assert not valid and "seed 2" in message and "FAIL" in message


def test_environment_required_after_record_005(tmp_path):
    legacy = make_record(tmp_path, "005_legacy", [400.0, 500.0, 450.0], env=None)
    assert validate_record(legacy)[0]
    new = make_record(tmp_path, "006_noenv", ten(400.0, 500.0, 450.0), env=None)
    valid, message = validate_record(new)
    assert not valid and "environment" in message


def test_unranked_gpu_fails_ranked_validation(tmp_path):
    record = make_record(tmp_path, "006_pcie", ten(400.0, 500.0, 450.0), env={**ENV, "gpu": "NVIDIA H100 PCIe"})
    valid, message = validate_record(record)
    assert not valid and "UNRANKED" in message
    assert validate_record(record, allow_fewer_seeds=True)[0]


def test_single_provider_required_after_record_005(tmp_path):
    legacy = make_record(tmp_path, "005_legacy", [400.0, 500.0, 450.0], config_extra={"provider": {"0": "modal", "1": "modal", "2": "runpod"}})
    valid, message = validate_record(legacy)
    assert valid and "MIXED" in message and "grandfathered" in message
    mixed = make_record(tmp_path, "006_mixed", ten(400.0, 500.0, 450.0), config_extra={"provider": {str(i): "modal" if i else "runpod" for i in range(10)}})
    valid, message = validate_record(mixed)
    assert not valid and "one provider" in message
    undeclared = make_record(tmp_path, "007_noprov", ten(400.0, 500.0, 450.0), config_extra={"provider": None})
    valid, message = validate_record(undeclared)
    assert not valid and "declare 'provider'" in message
    partial = make_record(tmp_path, "008_partial", ten(400.0, 500.0, 450.0), config_extra={"provider": {"0": "modal"}})
    valid, message = validate_record(partial)
    assert not valid and "declare 'provider'" in message
    runpod = make_record(tmp_path, "009_runpod", ten(400.0, 500.0, 450.0), config_extra={"provider": "RunPod"})
    valid, message = validate_record(runpod)
    assert valid and "provider: runpod" in message


def test_compare_reports_same_provider_subsets(tmp_path):
    split = {str(i): ("modal" if i < 3 else "runpod") for i in range(6)}
    old = make_record(tmp_path, "004_old", [600.0, 640.0, 620.0, 900.0, 950.0, 1000.0], config_extra={"provider": split})
    new = make_record(tmp_path, "005_new", [400.0, 420.0, 410.0, 430.0, 450.0, 440.0], config_extra={"provider": split})
    significant, report = compare_records(new, old)
    assert significant
    assert "span providers" in report and "modal: n=3 vs 3" in report and "runpod: n=3 vs 3" in report
    other = make_record(tmp_path, "006_lambda", ten(400.0, 420.0, 410.0), config_extra={"provider": "lambda"})
    same = make_record(tmp_path, "007_modal", ten(600.0, 640.0, 620.0))
    _, report = compare_records(other, same)
    assert "no provider in common" in report
    _, report = compare_records(make_record(tmp_path, "008_modal", ten(400.0, 420.0, 410.0)), same)
    assert "span providers" not in report


def test_seed_protocol(tmp_path):
    undeclared = make_record(tmp_path, "006_undeclared", ten(400.0, 500.0, 450.0), declare_seeds=False)
    valid, message = validate_record(undeclared)
    assert not valid and "declare 'seeds'" in message
    picked = make_record(tmp_path, "007_picked", ten(400.0, 500.0, 450.0), config_extra={"seeds": [0, 1, 7]}, seed_names=["0", "1", "7"])
    valid, message = validate_record(picked)
    assert not valid and "0..N-1" in message
    mismatch = make_record(tmp_path, "008_mismatch", ten(400.0, 500.0, 450.0), seed_names=["0", "1", "3"])
    valid, message = validate_record(mismatch)
    assert not valid and "exactly the declared seeds" in message


def test_validate_all_records_ranked_vs_unranked(tmp_path):
    from scripts.validate_all_records import validate_all

    records = tmp_path / "records"
    make_record(records, "006_ranked", ten(400.0, 500.0, 450.0))
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


def test_multi_gpu_run_is_unranked(tmp_path):
    record = make_record(tmp_path, "006_ddp", ten(400.0, 500.0, 450.0), env={**ENV, "world_size": 8})
    valid, message = validate_record(record)
    assert not valid and "world_size 1" in message
    assert validate_record(record, allow_fewer_seeds=True)[0]


def test_v2_requires_ten_seeds(tmp_path):
    record = make_record(tmp_path, "006_three", [400.0, 500.0, 450.0])
    valid, message = validate_record(record)
    assert not valid and "at least 10 seeds" in message
    legacy = make_record(tmp_path, "005_three", [400.0, 500.0, 450.0], confirm=None)
    assert validate_record(legacy)[0]


@pytest.mark.parametrize(
    ("override", "field"),
    [({"few_shot": 0}, "few_shot"), ({"max_new_tokens": 256}, "max_new_tokens"), ({"eval_limit": 500}, "eval_limit"), ({"target_solve_rate": 0.05}, "target_solve_rate"), ({"model": "Qwen/Qwen2.5-0.5B-Instruct"}, "base model")],
)
def test_v2_enforces_eval_protocol(tmp_path, override, field):
    record = make_record(tmp_path, "006_proto", ten(400.0, 500.0, 450.0), config_extra=override)
    valid, message = validate_record(record)
    assert not valid and field in message


def test_v2_stopping_target_may_exceed_threshold(tmp_path):
    record = make_record(tmp_path, "006_later", ten(400.0, 500.0, 450.0), config_extra={"target_solve_rate": benchmark.THRESHOLD + 0.01})
    assert validate_record(record)[0]


def test_v2_confirmation_required_and_scored(tmp_path):
    missing = make_record(tmp_path, "006_noconfirm", ten(400.0, 500.0, 450.0), confirm=None)
    valid, message = validate_record(missing)
    assert not valid and "missing confirm.json" in message
    lucky = make_record(tmp_path, "007_lucky", ten(400.0, 500.0, 450.0), confirm=benchmark.THRESHOLD - 0.005)
    valid, message = validate_record(lucky)
    assert not valid and f"mean confirm pass_rate {benchmark.THRESHOLD - 0.005:.4f} < threshold" in message
    assert "stopping-eval optimism (crossing - confirm): +0.0150" in message
    partial = make_record(tmp_path, "008_partial", ten(400.0, 500.0, 450.0))
    confirm = confirm_json()
    confirm["confirm"]["n"] = 500
    (partial / "seeds" / "3" / "confirm.json").write_text(json.dumps(confirm) + "\n")
    valid, message = validate_record(partial)
    assert not valid and "3 confirm must score all 4000 puzzles" in message


def test_v2_requires_version_and_model_revision(tmp_path):
    record = make_record(tmp_path, "006_old_trainer", ten(400.0, 500.0, 450.0), env={**ENV, "model_revision": None})
    result_path = record / "seeds" / "0" / "result.json"
    result = json.loads(result_path.read_text())
    del result["benchmark_version"]
    result_path.write_text(json.dumps(result) + "\n")
    valid, message = validate_record(record)
    assert not valid and "0: result.json benchmark_version" in message and "model_revision" in message


def test_compare_uses_log_time_and_refuses_cross_version(tmp_path):
    # one slow outlier: raw-time Welch is not significant, log-time Welch is (skewed data)
    old = make_record(tmp_path, "006_old", [600.0, 620.0, 640.0, 610.0, 630.0, 600.0, 620.0, 640.0, 610.0, 630.0])
    new = make_record(tmp_path, "007_new", [400.0, 410.0, 420.0, 405.0, 415.0, 400.0, 410.0, 420.0, 405.0, 1400.0])
    significant, report = compare_records(new, old)
    assert "log time" in report and "geometric-mean ratio" in report
    raw_p = float(report.split("welch one-sided on raw time: ")[1].split("p=")[1].split()[0])
    assert significant and raw_p > 0.05
    legacy = make_record(tmp_path, "005_legacy", [600.0, 640.0, 620.0], confirm=None)
    ok, report = compare_records(new, legacy)
    assert not ok and "different benchmark versions" in report
