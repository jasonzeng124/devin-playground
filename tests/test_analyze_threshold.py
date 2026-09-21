from pathlib import Path

from scripts.analyze_threshold import GENERATED_MARKER, analyze, load_record, wilson_halfwidth

ROOT = Path(__file__).resolve().parent.parent


def test_wilson_halfwidth_matches_logged_ci():
    # eval_log.jsonl rows carry wilson_95_ci; at p=0.107, n=2000 the logged interval is [0.0942, 0.1213]
    assert abs(wilson_halfwidth(0.107, 2000) - (0.12130887 - 0.107)) < 1e-6


def test_analyze_record_005():
    seeds = load_record(ROOT / "records/track_a/005_prefix_kv_compile_h100")
    assert len(seeds) == 10 and not any(s.censored for s in seeds)
    s0 = seeds[0]
    assert s0.crossing is not None and s0.crossing.step == 300 and s0.before_crossing.step == 275
    assert abs(s0.time_to_threshold_s - s0.crossing.wall_s) < 1e-6
    # seeds 7 and 9 crossed at the very first eval -> no slope
    assert seeds[7].slope_per_step is None and seeds[9].slope_per_step is None
    report = "\n".join(analyze(seeds))
    assert "## 005_prefix_kv_compile_h100" in report
    assert "steps-to-threshold explains" in report
    assert "| 0.10 | 10/10 |" in report


def test_doc_is_up_to_date_with_generator(tmp_path):
    """docs/threshold_calibration.md below the marker must equal the generator output for the two N=10 records."""
    doc = (ROOT / "docs/threshold_calibration.md").read_text()
    assert GENERATED_MARKER in doc
    generated = "\n".join(
        line
        for record in ("004_nokl_guardrail_h100", "005_prefix_kv_compile_h100")
        for line in analyze(load_record(ROOT / "records/track_a" / record))
    )
    assert generated.strip() in doc
