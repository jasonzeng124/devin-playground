#!/usr/bin/env python3
"""Validate and summarize a speedrun record directory; optionally test it against another.

Implements the checks in RULES.md ("Seeds and statistics", "Hardware and
reproducibility") with the standard library only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

RANKED_GPU = "NVIDIA H100 80GB HBM3"
ALPHA = 0.05
# Records accepted before the probe / environment / single-provider requirements existed (RULES.md).
PROBE_EXEMPT_MAX_RECORD = 3
ENVIRONMENT_EXEMPT_MAX_RECORD = 5
PROVIDER_EXEMPT_MAX_RECORD = 5
MAX_EXACT_PERMUTATIONS = 500_000


# --- Student t distribution (no scipy) -------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with `df` degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t by bisection (enough precision for a CI)."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- statistics ------------------------------------------------------------------------


def summary_stats(values: list[float]) -> tuple[float, float, float, float]:
    """(mean, sample std, ci_low, ci_high) with a Student-t 95% CI."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    std = statistics.stdev(values)
    margin = t_ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def _summary(values: list[float]) -> str:
    mean, std, low, high = summary_stats(values)
    return f"{mean:.4f} ± {std:.4f} (n={len(values)}, 95% CI [{low:.4f}, {high:.4f}])"


@dataclass
class Comparison:
    n_new: int
    n_old: int
    mean_new: float
    mean_old: float
    t: float
    df: float
    p_welch: float
    p_perm: float
    p_perm_exact: bool

    @property
    def significant(self) -> bool:
        return self.p_welch < ALPHA


def welch_one_sided(new: list[float], old: list[float]) -> tuple[float, float, float]:
    """One-sided Welch t-test for mean(new) < mean(old): (t, df, p)."""
    n1, n2 = len(new), len(old)
    m1, m2 = statistics.mean(new), statistics.mean(old)
    v1 = statistics.variance(new) / n1
    v2 = statistics.variance(old) / n2
    if v1 + v2 == 0:
        return (-math.inf if m1 < m2 else math.inf if m1 > m2 else 0.0), float(n1 + n2 - 2), (0.0 if m1 < m2 else 1.0)
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return t, df, t_cdf(t, df)


def permutation_one_sided(new: list[float], old: list[float]) -> tuple[float, bool]:
    """P(mean(new*) - mean(old*) <= observed) over relabelings; exact when feasible.

    Returns (p, exact). Falls back to a seeded Monte Carlo estimate when the number of
    relabelings exceeds MAX_EXACT_PERMUTATIONS.
    """
    pooled = new + old
    n1, total = len(new), len(pooled)
    observed = statistics.mean(new) - statistics.mean(old)
    pooled_sum = sum(pooled)

    def diff(indices) -> float:
        s1 = sum(pooled[i] for i in indices)
        return s1 / n1 - (pooled_sum - s1) / (total - n1)

    if math.comb(total, n1) <= MAX_EXACT_PERMUTATIONS:
        count = sum(1 for idx in itertools.combinations(range(total), n1) if diff(idx) <= observed + 1e-12)
        return count / math.comb(total, n1), True
    rng = random.Random(0)
    draws = 200_000
    count = sum(1 for _ in range(draws) if diff(rng.sample(range(total), n1)) <= observed + 1e-12)
    return (count + 1) / (draws + 1), False


def compare_times(new: list[float], old: list[float]) -> Comparison:
    if len(new) < 2 or len(old) < 2:
        raise ValueError("need at least 2 complete seeds in each record to compare")
    t, df, p = welch_one_sided(new, old)
    p_perm, exact = permutation_one_sided(new, old)
    return Comparison(len(new), len(old), statistics.mean(new), statistics.mean(old), t, df, p, p_perm, exact)


# --- record loading and rule checks ----------------------------------------------------


def _record_number(record: Path) -> int | None:
    match = re.fullmatch(r"(\d{3})_.+", record.name)
    return int(match.group(1)) if match else None


@dataclass
class LoadedRecord:
    path: Path
    config: dict
    seeds: list[str]
    results: dict[str, dict]
    probes: dict[str, dict]


def _load_record(record: Path, allow_fewer_seeds: bool) -> tuple[LoadedRecord | None, list[str]]:
    errors: list[str] = []
    if record.parent.name not in {"track_a", "track_b"}:
        errors.append("record must be under records/track_a or records/track_b")
    if _record_number(record) is None:
        errors.append("record directory must match NNN_slug")
    for name in ("README.md", "config.json"):
        if not (record / name).is_file():
            errors.append(f"missing {name}")
    config: dict = {}
    if (record / "config.json").is_file():
        try:
            config = json.loads((record / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                errors.append("config.json must be a JSON object")
                config = {}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json ({exc})")
    seeds_dir = record / "seeds"
    seed_dirs = sorted(path for path in seeds_dir.glob("*") if path.is_dir()) if seeds_dir.is_dir() else []
    if len(seed_dirs) < 3 and not allow_fewer_seeds:
        errors.append(f"expected at least 3 seed directories, found {len(seed_dirs)}")
    declared = config.get("seeds")
    if declared is not None:
        if not (isinstance(declared, list) and all(isinstance(s, int) for s in declared)):
            errors.append("config.json 'seeds' must be a list of integers")
        else:
            if sorted(declared) != list(range(len(declared))):
                errors.append(f"declared seeds must be exactly 0..N-1, got {declared}")
            found = sorted(int(path.name) for path in seed_dirs if path.name.isdigit())
            if found != sorted(declared) or len(found) != len(seed_dirs):
                errors.append(f"seeds/ must contain exactly the declared seeds {sorted(declared)}, found {[p.name for p in seed_dirs]}")
    elif not allow_fewer_seeds:
        errors.append("config.json must declare 'seeds': [0, ..., N-1]")
    results: dict[str, dict] = {}
    probes: dict[str, dict] = {}
    for seed_dir in seed_dirs:
        result_path = seed_dir / "result.json"
        if not result_path.is_file():
            errors.append(f"{seed_dir.name}: missing result.json")
            continue
        if not (seed_dir / "train_log.jsonl").is_file():
            errors.append(f"{seed_dir.name}: missing train_log.jsonl")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{seed_dir.name}: invalid result.json ({exc})")
            continue
        pass_rate = result.get("final_pass_rate", result.get("pass_rate"))
        if not isinstance(pass_rate, (int, float)):
            errors.append(f"{seed_dir.name}: result missing final_pass_rate/pass_rate")
            continue
        results[seed_dir.name] = result
        probe_path = seed_dir / "probe.json"
        if probe_path.is_file():
            try:
                probe = json.loads(probe_path.read_text(encoding="utf-8"))
                probes[seed_dir.name] = {
                    "fineweb_loss_delta": float(probe["fineweb_loss_delta"]),
                    "mmlu_lite_acc_delta": float(probe["mmlu_lite_acc_delta"]),
                }
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{seed_dir.name}: invalid probe.json ({exc})")
    if errors:
        return None, errors
    return LoadedRecord(record, config, [p.name for p in seed_dirs], results, probes), []


def _pass_rate(result: dict) -> float:
    return float(result["final_pass_rate"] if "final_pass_rate" in result else result["pass_rate"])


def _providers(loaded: LoadedRecord) -> dict[str, str] | None:
    """Seed -> provider from config.json 'provider' (a string for all seeds, or a {seed: provider} map)."""
    declared = loaded.config.get("provider")
    if isinstance(declared, str):
        return {seed: declared.lower() for seed in loaded.seeds}
    if isinstance(declared, dict) and all(isinstance(v, str) for v in declared.values()):
        return {seed: declared[seed].lower() for seed in loaded.seeds if seed in declared}
    return None


def _provider_summary(providers: dict[str, str]) -> str:
    groups: dict[str, list[str]] = {}
    for seed, provider in providers.items():
        groups.setdefault(provider, []).append(seed)
    return ", ".join(f"{p} (seeds {','.join(s)})" for p, s in sorted(groups.items()))


def _is_ranked_hardware(loaded: LoadedRecord) -> bool:
    gpus = [res.get("environment", {}).get("gpu") for res in loaded.results.values()]
    if all(isinstance(g, str) for g in gpus) and gpus:
        return all(g == RANKED_GPU for g in gpus)
    return "H100" in str(loaded.config.get("hardware", ""))


def validate_record(record: Path, allow_fewer_seeds: bool = False) -> tuple[bool, str]:
    loaded, errors = _load_record(record, allow_fewer_seeds)
    if loaded is None:
        return False, "\n".join(errors)
    number = _record_number(record) or 0
    lines = [f"record: {record}", f"seeds: {len(loaded.results)}"]

    pass_rates = [_pass_rate(res) for res in loaded.results.values()]
    lines.append(f"pass_rate: {_summary(pass_rates)}")
    censored = [seed for seed, res in loaded.results.items() if res.get("time_to_threshold_s") is None]
    times = [float(res["time_to_threshold_s"]) for res in loaded.results.values() if res.get("time_to_threshold_s") is not None]
    lines.append(f"time_to_threshold_s: {_summary(times) if times else 'no thresholds reached'}")
    if censored and not allow_fewer_seeds:
        errors.append(f"censored seeds (threshold not reached): {censored} — raise max_steps and re-run them, keep these under extra/")
    elif censored:
        lines.append(f"censored seeds: {censored}")
    extra = record / "extra"
    if extra.is_dir():
        lines.append(f"extra runs disclosed: {sorted(p.name for p in extra.iterdir() if p.is_dir())}")

    # environment / hardware identity
    envs = {seed: res.get("environment") for seed, res in loaded.results.items()}
    if number > ENVIRONMENT_EXEMPT_MAX_RECORD and not allow_fewer_seeds:
        for seed, env in envs.items():
            if not isinstance(env, dict) or not isinstance(env.get("gpu"), str):
                errors.append(f"{seed}: result.json lacks an 'environment' block (re-run with the current trainer)")
    gpus = sorted({env["gpu"] for env in envs.values() if isinstance(env, dict) and isinstance(env.get("gpu"), str)})
    if gpus:
        lines.append(f"gpu: {', '.join(gpus)}" + ("" if gpus == [RANKED_GPU] else f" (UNRANKED: leaderboard hardware is {RANKED_GPU})"))
        if gpus != [RANKED_GPU] and not allow_fewer_seeds:
            errors.append(f"ranked records must run on {RANKED_GPU}; found {gpus}")
        software = sorted({f"torch {env.get('torch')} / transformers {env.get('transformers')} / cuda {env.get('cuda')}" for env in envs.values() if isinstance(env, dict)})
        lines.append(f"software: {'; '.join(software)}")
    else:
        lines.append(f"hardware (declared): {loaded.config.get('hardware', 'unknown')}")

    # provider: all seeds of a ranked record on one declared host/provider
    providers = _providers(loaded)
    if providers is None or set(providers) != set(loaded.seeds):
        if number > PROVIDER_EXEMPT_MAX_RECORD and not allow_fewer_seeds:
            errors.append("config.json must declare 'provider' (the cloud/host every seed ran on)")
        elif "provider" in loaded.config:
            errors.append("config.json 'provider' must be a string or a {seed: provider} map covering every seed")
        else:
            lines.append("provider: not declared (record predates the rule)")
    else:
        distinct = sorted(set(providers.values()))
        if len(distinct) == 1:
            lines.append(f"provider: {distinct[0]}")
        else:
            lines.append(f"provider: MIXED — {_provider_summary(providers)}" + (" (grandfathered)" if number <= PROVIDER_EXEMPT_MAX_RECORD else ""))
            if number > PROVIDER_EXEMPT_MAX_RECORD and not allow_fewer_seeds:
                errors.append(f"all seeds of a ranked record must run on one provider; found {distinct}")

    # capability guardrail
    probes = loaded.probes
    if not probes:
        if number <= PROBE_EXEMPT_MAX_RECORD:
            lines.append("guardrail: not measured (record predates the probe; exempt)")
        elif allow_fewer_seeds:
            lines.append("guardrail: not measured")
        else:
            errors.append("guardrail: missing probe.json for every seed (required for records after 003)")
    else:
        for seed in loaded.seeds:
            if seed not in probes:
                errors.append(f"guardrail: {seed} missing probe.json")
                continue
            loss_delta, acc_delta = probes[seed]["fineweb_loss_delta"], probes[seed]["mmlu_lite_acc_delta"]
            passed = loss_delta <= 0.05 and acc_delta >= -0.03
            lines.append(f"guardrail: seed {seed}: fineweb_loss Δ={loss_delta:+.3f}, mmlu_lite Δ={acc_delta:+.3f} ({'pass' if passed else 'FAIL'})")
            if not passed:
                errors.append(f"guardrail: {seed} fails (fineweb Δ={loss_delta:+.3f} > 0.05 or mmlu_lite Δ={acc_delta:+.3f} < -0.03)")
        mean_loss = statistics.mean(p["fineweb_loss_delta"] for p in probes.values())
        mean_acc = statistics.mean(p["mmlu_lite_acc_delta"] for p in probes.values())
        ok = not any(e.startswith("guardrail") for e in errors)
        lines.append(f"guardrail: fineweb_loss Δ={mean_loss:+.3f}, mmlu_lite Δ={mean_acc:+.3f} ({'pass' if ok else 'FAIL'})")

    if errors:
        return False, "\n".join(lines + errors)
    return True, "\n".join(lines)


def compare_records(new: Path, old: Path) -> tuple[bool, str]:
    """Test `new` faster than `old` (RULES.md 'Beating a record'). Returns (significant, report)."""
    problems = []
    loaded = {}
    for label, path in (("new", new), ("old", old)):
        rec, errors = _load_record(path, allow_fewer_seeds=False)
        if rec is None:
            problems.extend(f"{label} ({path.name}): {e}" for e in errors)
            continue
        loaded[label] = rec
    if problems:
        return False, "\n".join(problems)
    a, b = loaded["new"], loaded["old"]
    if a.path.parent.name != b.path.parent.name:
        problems.append("records are in different tracks")
    if a.config.get("model") != b.config.get("model"):
        problems.append(f"different base models: {a.config.get('model')} vs {b.config.get('model')}")
    if a.config.get("target_solve_rate") != b.config.get("target_solve_rate"):
        problems.append(f"different thresholds: {a.config.get('target_solve_rate')} vs {b.config.get('target_solve_rate')}")
    if a.config.get("eval_limit") != b.config.get("eval_limit"):
        problems.append(f"different eval sizes: {a.config.get('eval_limit')} vs {b.config.get('eval_limit')}")
    for label, rec in (("new", a), ("old", b)):
        if not _is_ranked_hardware(rec):
            problems.append(f"{label} ({rec.path.name}) was not run on ranked hardware ({RANKED_GPU})")
        censored = [s for s, r in rec.results.items() if r.get("time_to_threshold_s") is None]
        if censored:
            problems.append(f"{label} ({rec.path.name}) has censored seeds {censored}; cannot compare")
    if problems:
        return False, "\n".join(problems)
    t_new = [float(r["time_to_threshold_s"]) for r in a.results.values()]
    t_old = [float(r["time_to_threshold_s"]) for r in b.results.values()]
    cmp = compare_times(t_new, t_old)
    lines = [
        f"compare: {a.path.name} (n={cmp.n_new}, mean {cmp.mean_new:.1f} s) vs {b.path.name} (n={cmp.n_old}, mean {cmp.mean_old:.1f} s)",
        f"difference: {cmp.mean_new - cmp.mean_old:+.1f} s",
        f"welch one-sided (new < old): t={cmp.t:.3f}, df={cmp.df:.2f}, p={cmp.p_welch:.4f}",
        f"permutation one-sided ({'exact' if cmp.p_perm_exact else 'monte carlo'}): p={cmp.p_perm:.4f}",
        f"verdict: {'BEATS' if cmp.significant else 'does not beat'} {b.path.name} at alpha={ALPHA}",
    ]
    if (cmp.p_perm < ALPHA) != cmp.significant:
        lines.append("note: Welch and permutation tests disagree at alpha; report both in the PR")
    prov_new, prov_old = _providers(a), _providers(b)
    if prov_new is not None and prov_old is not None:
        shared = sorted(set(prov_new.values()) & set(prov_old.values()))
        if len(set(prov_new.values()) | set(prov_old.values())) > 1:
            lines.append("note: records span providers (host speed differs by a few % per step); same-provider subsets:")
            if not shared:
                lines.append("  no provider in common — the difference includes a host effect; reproduce one record on the other's provider")
            for provider in shared:
                sub_new = [float(a.results[s]["time_to_threshold_s"]) for s, p in prov_new.items() if p == provider]
                sub_old = [float(b.results[s]["time_to_threshold_s"]) for s, p in prov_old.items() if p == provider]
                if len(sub_new) < 2 or len(sub_old) < 2:
                    lines.append(f"  {provider}: n={len(sub_new)} vs {len(sub_old)}, too few to test")
                    continue
                sub = compare_times(sub_new, sub_old)
                lines.append(f"  {provider}: n={sub.n_new} vs {sub.n_old}, mean {sub.mean_new:.1f} vs {sub.mean_old:.1f} s, welch p={sub.p_welch:.4f}, permutation p={sub.p_perm:.4f}")
    return cmp.significant, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--allow-fewer-seeds", action="store_true", help="smoke/unranked records: relax seed, probe, censoring and hardware checks")
    parser.add_argument("--compare-to", type=Path, help="previous record; exits 0 only if this record is significantly faster")
    args = parser.parse_args()
    valid, message = validate_record(args.record, args.allow_fewer_seeds)
    print(message)
    if args.compare_to is not None:
        significant, report = compare_records(args.record, args.compare_to)
        print(report)
        valid = valid and significant
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
