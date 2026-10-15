#!/usr/bin/env python3
"""Validate a benchmark v3 record (docs/RULES_v3.md); optionally test it against the holder.

    python scripts/validate_record_v3.py records/v3/track_{a,d}/<NNN>_<slug> [--compare-to <holder>]

Checks: the record path and `v3` config block, consecutive seeds 0..N-1 with N >= 5, and per seed:
- `config.json` matches the record's config, in-training evals never use the test ladder, no
  resuming (its checkpoint saves are taken off the clock), and for track D GRPO starts from the
  seed's own pretrained model;
- `clock.json`: ranked GPU, the track's budget and within it, and total_s = pretrain_s +
  grpo_wall_s - grpo_eval_s, with grpo_eval_s matching the in-loop evals in `eval_log.jsonl`;
- `result.json`: one GPU, the same GPU as the clock, and the pinned revision for track A;
- `confirm_v3.json`: a full test-ladder run of the seed's own final weights, scored with the
  record's sampler, whose score is the mean of its six levels;
- track D: a `pretrain_log.jsonl` per seed, no two alike.
`--compare-to` runs the one-sided Welch test on seed scores. Records with `"unranked": true`
(smoke tests) skip the seed-count and hardware checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from rlvr_speedrun import benchmark
from scripts.validate_record import ALPHA, RANKED_GPU, welch_one_sided

LADDER_SIZE = 2400  # 6 levels x 400 (data/ladder_v3_test.jsonl)
LEVELS = ("L1", "L2", "L3", "L4", "L5", "L6")
TRACKS = {"track_a": "A", "track_d": "D"}
RECORD_ONLY_KEYS = {"description", "v3", "unranked"}  # record metadata, not trainer flags
CLOCK_TOLERANCE_S = 5.0


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


def _last_row(path: Path, errors: list[str], label: str) -> dict | None:
    """The last row of a .jsonl log; None (with an error) if the log is missing, empty or unreadable."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = json.loads(lines[-1]) if lines else None
    except FileNotFoundError:
        errors.append(f"{label}: missing {path.name}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: unreadable {path.name}: {exc}")
        return None
    if not isinstance(row, dict):
        errors.append(f"{label}: {path.name} has no rows")
        return None
    return row


def _ends_with(path: object, *parts: str) -> bool:
    return Path(str(path or "")).parts[-len(parts):] == parts


def _check_seed_config(seed: Path, name: str, track: str, config: dict, errors: list[str]) -> None:
    label = f"seed {name}"
    seed_config = _read_json(seed / "config.json", errors)
    if seed_config is None:
        return
    for key, value in config.items():
        if key in RECORD_ONLY_KEYS or (track == "D" and key == "model"):
            continue
        if seed_config.get(key) != value:
            errors.append(f"{label}: config.json has {key}={seed_config.get(key)!r}, the record's config.json {value!r}")
    if seed_config.get("seed") != int(name):
        errors.append(f"{label}: config.json has seed {seed_config.get('seed')!r}")
    if Path(str(seed_config.get("eval_set") or "")).name == benchmark.LADDER_TEST_PATH.name:
        errors.append(f"{label}: in-training evals use the test ladder ({seed_config['eval_set']}); use the dev ladder")
    if seed_config.get("resume_every_s") or seed_config.get("resume_dir"):
        errors.append(f"{label}: resuming was on; its checkpoint saves are taken off the clock, which RULES_v3 doesn't allow")
    if track == "D" and not _ends_with(seed_config.get("model"), "seeds", name, "pretrained"):
        errors.append(f"{label}: GRPO must start from this seed's own pretrained model (seeds/{name}/pretrained), "
                      f"got {seed_config.get('model')!r}")


def _check_clock(seed: Path, name: str, clock: dict, budget: float, errors: list[str]) -> None:
    label = f"seed {name}"
    parts = [clock.get(k) for k in ("pretrain_s", "grpo_wall_s", "grpo_eval_s", "total_s")]
    if not all(isinstance(v, (int, float)) for v in parts):
        errors.append(f"{label}: clock.json needs numeric pretrain_s, grpo_wall_s, grpo_eval_s and total_s")
        return
    pre, wall, eval_s, total = parts
    if abs(pre + wall - eval_s - total) > CLOCK_TOLERANCE_S:
        errors.append(f"{label}: clock total_s {total:.0f} != pretrain_s + grpo_wall_s - grpo_eval_s ({pre + wall - eval_s:.0f})")
    if clock.get("within_budget") is not (total <= budget):
        errors.append(f"{label}: clock within_budget {clock.get('within_budget')!r} doesn't match total_s {total:.0f}")
    if not (seed / "eval_log.jsonl").is_file():
        if eval_s > CLOCK_TOLERANCE_S:
            errors.append(f"{label}: grpo_eval_s {eval_s:.0f} is taken off the clock, but there is no eval_log.jsonl")
        return
    last = _last_row(seed / "eval_log.jsonl", errors, label)
    if last is not None:
        logged = (last.get("wall_s") or 0.0) - (last.get("train_s") or 0.0)
        if abs(logged - eval_s) > CLOCK_TOLERANCE_S:
            errors.append(f"{label}: grpo_eval_s {eval_s:.0f} != the in-loop eval time in eval_log.jsonl ({logged:.0f})")


def _check_confirm(name: str, confirm: dict, errors: list[str]) -> None:
    label = f"seed {name}"
    if Path(str(confirm.get("test_set") or "")).name != benchmark.LADDER_TEST_PATH.name:
        errors.append(f"{label}: confirm_v3.json was not scored on the test ladder (test_set {confirm.get('test_set')!r})")
    if not _ends_with(confirm.get("model"), "seeds", name, "final"):
        errors.append(f"{label}: confirm_v3.json scored {confirm.get('model')!r}, not this seed's final weights")
    levels = confirm.get("pass@1_by_level")
    if not isinstance(levels, dict) or sorted(levels) != list(LEVELS) \
            or not all(isinstance(v, (int, float)) for v in levels.values()):
        errors.append(f"{label}: confirm_v3.json needs pass@1 for levels {', '.join(LEVELS)}")
    elif isinstance(confirm.get("ladder_score"), (int, float)) and abs(statistics.fmean(levels.values()) - confirm["ladder_score"]) > 1e-6:
        errors.append(f"{label}: ladder_score {confirm['ladder_score']} != the mean of its levels ({statistics.fmean(levels.values()):.4f})")


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
    pretrain_logs: dict[str, str] = {}  # sha256 -> seed
    budget = benchmark.V3_TIME_BUDGET_S[track]
    for name in names:
        seed = seeds_dir / name
        n_err = len(errors)
        _check_seed_config(seed, name, track, config, errors)
        last = _last_row(seed / "train_log.jsonl", errors, f"seed {name}")
        if last is not None and "step" not in last:
            errors.append(f"seed {name}: train_log.jsonl rows have no step")
        if track == "D":
            if not (seed / "pretrain_log.jsonl").is_file():
                errors.append(f"seed {name}: missing pretrain_log.jsonl (each track D seed pretrains its own model)")
            else:
                digest = hashlib.sha256((seed / "pretrain_log.jsonl").read_bytes()).hexdigest()
                if digest in pretrain_logs:
                    errors.append(f"seed {name}: same pretrain_log.jsonl as seed {pretrain_logs[digest]} (each track D seed pretrains its own model)")
                pretrain_logs.setdefault(digest, name)
        clock = _read_json(seed / "clock.json", errors)
        if clock is not None:
            if clock.get("budget_s") != budget:
                errors.append(f"seed {name}: clock budget_s {clock.get('budget_s')} != {budget}")
            if not isinstance(clock.get("total_s"), (int, float)) or clock["total_s"] > budget:
                errors.append(f"seed {name}: over budget (total_s {clock.get('total_s')})")
            if clock.get("gpu") != RANKED_GPU and not unranked:
                errors.append(f"seed {name}: GPU {clock.get('gpu')!r} is not the ranked {RANKED_GPU!r}")
            _check_clock(seed, name, clock, budget, errors)
        result = _read_json(seed / "result.json", errors)
        if result is not None:
            env = result.get("environment") or {}
            if track == "A" and env.get("model_revision") != benchmark.V3_TRACK_A_REVISION:
                errors.append(f"seed {name}: model_revision {env.get('model_revision')!r} != pinned {benchmark.V3_TRACK_A_REVISION}")
            if not unranked:
                if env.get("gpu_count", 1) != 1 or env.get("world_size", 1) != 1:
                    errors.append(f"seed {name}: trained with gpu_count {env.get('gpu_count')} / world_size {env.get('world_size')}; "
                                  "ranked runs use one GPU")
                if "gpu" in env and clock is not None and env["gpu"] != clock.get("gpu"):
                    errors.append(f"seed {name}: result.json GPU {env['gpu']!r} != clock.json GPU {clock.get('gpu')!r}")
        confirm = _read_json(seed / "confirm_v3.json", errors)
        if confirm is not None:
            _check_confirm(name, confirm, errors)
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
