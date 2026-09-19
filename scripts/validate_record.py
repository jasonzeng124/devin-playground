#!/usr/bin/env python3
"""Validate and summarize a speedrun record directory."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def _summary(values: list[float]) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    z = 1.96
    margin = z * std / math.sqrt(len(values)) if values else 0.0
    return f"{mean:.4f} ± {std:.4f} (95% CI [{mean - margin:.4f}, {mean + margin:.4f}])"


def validate_record(record: Path, allow_fewer_seeds: bool = False) -> tuple[bool, str]:
    errors = []
    if record.parent.name not in {"track_a", "track_b"}:
        errors.append("record must be under records/track_a or records/track_b")
    if not re.fullmatch(r"\d{3}_.+", record.name):
        errors.append("record directory must match NNN_slug")
    for name in ("README.md", "config.json"):
        if not (record / name).is_file():
            errors.append(f"missing {name}")
    if (record / "config.json").is_file():
        try:
            json.loads((record / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json ({exc})")
    seeds_dir = record / "seeds"
    seed_dirs = sorted(path for path in seeds_dir.glob("*") if path.is_dir()) if seeds_dir.is_dir() else []
    if len(seed_dirs) < 3 and not allow_fewer_seeds:
        errors.append(f"expected at least 3 seed directories, found {len(seed_dirs)}")
    records = []
    for seed_dir in seed_dirs:
        result_path = seed_dir / "result.json"
        log_path = seed_dir / "train_log.jsonl"
        if not result_path.is_file():
            errors.append(f"{seed_dir.name}: missing result.json")
            continue
        if not log_path.is_file():
            errors.append(f"{seed_dir.name}: missing train_log.jsonl")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            records.append(result)
            pass_rate = result.get("final_pass_rate", result.get("pass_rate"))
            if not isinstance(pass_rate, (int, float)):
                errors.append(f"{seed_dir.name}: result missing final_pass_rate/pass_rate")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{seed_dir.name}: invalid result.json ({exc})")
    if errors:
        return False, "\n".join(errors)
    pass_rates = [
        float(item["final_pass_rate"] if "final_pass_rate" in item else item["pass_rate"])
        for item in records
    ]
    times = [float(item["time_to_threshold_s"]) for item in records if item.get("time_to_threshold_s") is not None]
    lines = [
        f"record: {record}",
        f"seeds: {len(records)}",
        f"pass_rate: {_summary(pass_rates)}",
        f"time_to_threshold_s: {_summary(times) if times else 'no thresholds reached'}",
    ]
    probes = []
    for seed_dir in seed_dirs:
        probe_path = seed_dir / "probe.json"
        if probe_path.is_file():
            try:
                probe = json.loads(probe_path.read_text(encoding="utf-8"))
                loss_delta = float(probe["fineweb_loss_delta"])
                acc_delta = float(probe["mmlu_lite_acc_delta"])
                passed = loss_delta <= 0.05 and acc_delta >= -0.03
                probes.append((seed_dir.name, loss_delta, acc_delta, passed))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{seed_dir.name}: invalid probe.json ({exc})")
    if errors:
        return False, "\n".join(errors)
    if not probes:
        lines.append("guardrail: not measured")
    else:
        if len(probes) != len(seed_dirs):
            lines.append("guardrail: FAIL (missing probe.json for one or more seeds)")
        for seed, loss_delta, acc_delta, passed in probes:
            lines.append(
                f"guardrail: seed {seed}: fineweb_loss Δ={loss_delta:+.3f}, "
                f"mmlu_lite Δ={acc_delta:+.3f} ({'pass' if passed else 'FAIL'})"
            )
        mean_loss = statistics.mean(item[1] for item in probes)
        mean_acc = statistics.mean(item[2] for item in probes)
        lines.append(
            f"guardrail: fineweb_loss Δ={mean_loss:+.3f}, mmlu_lite Δ={mean_acc:+.3f} "
            f"({'pass' if len(probes) == len(seed_dirs) and all(item[3] for item in probes) else 'FAIL'})"
        )
        if len(probes) != len(seed_dirs) or not all(item[3] for item in probes):
            return False, "\n".join(lines)
    return True, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--allow-fewer-seeds", action="store_true")
    args = parser.parse_args()
    valid, message = validate_record(args.record, args.allow_fewer_seeds)
    print(message)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
