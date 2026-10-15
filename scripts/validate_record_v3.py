#!/usr/bin/env python3
"""Validate a benchmark v3 record (docs/RULES_v3.md); optionally test it against the holder.

    python scripts/validate_record_v3.py records/v3/track_{a,d}/<NNN>_<slug> [--compare-to <holder>]

Checks: the record path and `v3` config block, consecutive seeds 0..N-1 with N >= 5, and per seed
`clock.json` (ranked GPU, the track's budget, within it), a full test-ladder `confirm_v3.json`
scored with the record's own sampler, `pretrain_log.jsonl` for track D, and the pinned base
model and revision for track A. `--compare-to` runs the one-sided Welch test on seed scores.
Records with `"unranked": true` (smoke tests) skip the seed-count and hardware checks.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from rlvr_speedrun import benchmark
from scripts.validate_record import ALPHA, RANKED_GPU, welch_one_sided

LADDER_SIZE = 2400  # 6 levels x 400 (data/ladder_v3_test.jsonl)
TRACKS = {"track_a": "A", "track_d": "D"}


def _read_json(path: Path, errors: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing {path}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable {path}: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path} is not a JSON object")
        return None
    return data


def load_scores(record: Path) -> tuple[list[float], list[str], list[str]]:
    """Seed scores in seed order, error messages, and summary lines."""
    errors: list[str] = []
    lines: list[str] = []
    track = TRACKS.get(record.parent.name)
    if track is None or record.parent.parent.name != "v3":
        errors.append("record must be under records/v3/track_a or records/v3/track_d")
        return [], errors, lines
    if not re.fullmatch(r"\d{3}_[a-z0-9_.-]+", record.name):
        errors.append(f"record directory must be named NNN_slug, got {record.name!r}")
    config = _read_json(record / "config.json", errors)
    if config is None:
        return [], errors, lines
    v3 = config.get("v3")
    if not isinstance(v3, dict):
        errors.append('config.json needs a "v3" block (track, sampler, pretrain for track D)')
        return [], errors, lines
    if v3.get("track") != track:
        errors.append(f"v3.track must be {track!r} for {record.parent.name}, got {v3.get('track')!r}")
    sampler = v3.get("sampler")
    if not isinstance(sampler, dict):
        errors.append('v3 block needs a "sampler" (temperature, constrain, think_budget)')
        sampler = {}
    if track == "A" and config.get("model") != benchmark.V3_TRACK_A_MODEL:
        errors.append(f"track A base model must be {benchmark.V3_TRACK_A_MODEL}, got {config.get('model')!r}")
    if track == "D" and not isinstance(v3.get("pretrain"), dict):
        errors.append('track D v3 block needs a "pretrain" entry')
    if not (record / "README.md").is_file():
        errors.append("missing README.md")
    unranked = config.get("unranked") is True

    seeds_dir = record / "seeds"
    names = sorted((p.name for p in seeds_dir.iterdir() if p.is_dir()), key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0, n)) if seeds_dir.is_dir() else []
    if names != [str(i) for i in range(len(names))]:
        errors.append(f"seeds must be 0..N-1, got {names}")
    if len(names) < benchmark.V3_MIN_SEEDS and not unranked:
        errors.append(f"need at least {benchmark.V3_MIN_SEEDS} seeds, got {len(names)}")

    want_sampler = {"temperature": sampler.get("temperature", 1.0), "constrain": bool(sampler.get("constrain", False)),
                    "think_budget": sampler.get("think_budget", 0) if sampler.get("constrain") else 0}
    scores: list[float] = []
    for name in names:
        seed = seeds_dir / name
        n_err = len(errors)
        if not (seed / "train_log.jsonl").is_file():
            errors.append(f"seed {name}: missing train_log.jsonl")
        if track == "D" and not (seed / "pretrain_log.jsonl").is_file():
            errors.append(f"seed {name}: missing pretrain_log.jsonl (each track D seed pretrains its own model)")
        clock = _read_json(seed / "clock.json", errors)
        if clock is not None:
            if clock.get("budget_s") != benchmark.V3_TIME_BUDGET_S[track]:
                errors.append(f"seed {name}: clock budget_s {clock.get('budget_s')} != {benchmark.V3_TIME_BUDGET_S[track]}")
            if not isinstance(clock.get("total_s"), (int, float)) or clock["total_s"] > benchmark.V3_TIME_BUDGET_S[track]:
                errors.append(f"seed {name}: over budget (total_s {clock.get('total_s')})")
            if clock.get("gpu") != RANKED_GPU and not unranked:
                errors.append(f"seed {name}: GPU {clock.get('gpu')!r} is not the ranked {RANKED_GPU!r}")
        result = _read_json(seed / "result.json", errors)
        if result is not None and track == "A":
            revision = (result.get("environment") or {}).get("model_revision")
            if revision != benchmark.V3_TRACK_A_REVISION:
                errors.append(f"seed {name}: model_revision {revision!r} != pinned {benchmark.V3_TRACK_A_REVISION}")
        confirm = _read_json(seed / "confirm_v3.json", errors)
        if confirm is not None:
            if confirm.get("benchmark_version") != 3 or confirm.get("n_puzzles") != LADDER_SIZE \
                    or confirm.get("samples_per_puzzle") != benchmark.V3_SAMPLES_PER_PUZZLE:
                errors.append(f"seed {name}: confirm_v3.json is not a full test-ladder run "
                              f"({confirm.get('n_puzzles')} puzzles x {confirm.get('samples_per_puzzle')} samples)")
            got = dict(confirm.get("sampler") or {})
            if got.pop("max_new_tokens", None) != benchmark.V3_MAX_NEW_TOKENS:
                errors.append(f"seed {name}: confirm max_new_tokens must be {benchmark.V3_MAX_NEW_TOKENS}")
            if got != want_sampler:
                errors.append(f"seed {name}: confirm sampler {got} != the record's {want_sampler}")
            if len(errors) == n_err and isinstance(confirm.get("ladder_score"), (int, float)):
                scores.append(float(confirm["ladder_score"]))
                minutes = clock["total_s"] / 60 if clock else float("nan")
                lines.append(f"seed {name}: test {confirm['ladder_score']:.4f}  clock {minutes:.1f} min  {clock.get('gpu') if clock else ''}")
    if scores:
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lines.append(f"track {track}: mean test ladder score {statistics.fmean(scores):.4f} (sd {sd:.4f}, n={len(scores)})")
    return scores, errors, lines


def validate(record: Path) -> tuple[bool, str]:
    _, errors, lines = load_scores(record)
    return not errors, "\n".join(lines + [f"FAIL: {e}" for e in errors])


def compare(new: Path, old: Path) -> tuple[bool, str]:
    new_scores, new_errors, _ = load_scores(new)
    old_scores, old_errors, _ = load_scores(old)
    if new_errors or old_errors:
        return False, "\n".join(f"FAIL: {e}" for e in new_errors + old_errors)
    if new.parent.name != old.parent.name:
        return False, "FAIL: records are on different tracks"
    t, df, p = welch_one_sided(old_scores, new_scores)  # H1: mean(old) < mean(new)
    ok = p < ALPHA
    return ok, (f"new {statistics.fmean(new_scores):.4f} vs holder {statistics.fmean(old_scores):.4f}: "
                f"Welch t={-t:.2f} df={df:.1f} one-sided p={p:.4f} -> {'new record' if ok else 'not significant'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path)
    parser.add_argument("--compare-to", type=Path, help="current holder; exits 0 only if this record is significantly better")
    args = parser.parse_args()
    ok, message = validate(args.record)
    print(message)
    if ok and args.compare_to:
        ok, message = compare(args.record, args.compare_to)
        print(message)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
