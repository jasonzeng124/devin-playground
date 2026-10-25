"""One line per screen seed, then one line per arm: dev score, takeoff time and collapse mode.

    python scripts/summarize_screens.py calibration/research_r1/* [records/v3/track_a/001_ladder_constrained]

A seed's logs are read from <arm>/seeds/<i>/ when the run has finished, else from the running
run's output under $CKPT_DIR/<arm>/seeds/<i>/ (scripts/slurm/screen_seed.sh). Collapse modes,
from the last training rows:
  diverged   malformed rate > 0.5 in training or in the last eval (the policy emits garbage)
  one-guess  fewer than 10 of 64 groups informative (answers at once with one guess, little signal
             left; healthy runs keep 20-45)
  ok         anything else
Step-capped runs (grpo `budget_steps`) are timed by their budget clock (`clock_s`), not the wall clock.
An arm with an eval.json (scripts/research/run_item.sh) is a dev-ladder sampler eval: one line per
seed's dev_confirm.json (16 samples per puzzle, the reference scorer) and their mean.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

CKPT = Path(os.environ.get("CKPT_DIR", Path.home() / "orcd/pool/rlvr-ckpts"))


def _rows(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _clock(row: dict) -> float:
    """Training seconds by the run's budget clock: the wall clock, or a step-capped run's clock_s (grpo budget_steps)."""
    return row.get("clock_s", row["train_s"])


def seed_summary(arm: Path, seed: str) -> dict | None:
    local = arm / "seeds" / seed  # screens copy their logs here when done; run_seed_v3.sh (track D pretraining) writes here as it runs
    done = (local / "train_log.jsonl").exists() and ((local / "clock.json").exists() or not (local / "pretrain_log.jsonl").exists())
    src = local if (local / "train_log.jsonl").exists() else CKPT / arm / "seeds" / seed
    train, evals = _rows(src / "train_log.jsonl"), _rows(src / "eval_log.jsonl")
    if not train or not evals:
        return None
    last = train[-1]
    minutes = _clock(last) / 60
    recent = [r for r in train if _clock(r) >= _clock(last) - 1800]
    spm = (recent[-1]["step"] - recent[0]["step"]) / max(1e-9, (_clock(recent[-1]) - _clock(recent[0])) / 60)
    tail = train[-20:]
    ent = statistics.mean(r.get("entropy", 0) for r in tail)
    inf = statistics.mean(r.get("informative_groups", 0) for r in tail)
    mal = statistics.mean(r.get("malformed_rate", 0) for r in tail)
    mode = "diverged" if max(mal, evals[-1].get("malformed_rate", 0)) > 0.5 else "one-guess" if inf < 10 else "ok"
    score = [e.get("pass_rate", e.get("ladder_score", 0)) for e in evals]
    t10 = next((_clock(e) / 60 for e, x in zip(evals, score) if x >= 0.10), None)

    def at(m):
        c = [x for e, x in zip(evals, score) if _clock(e) <= m * 60]
        return c[-1] if c else float("nan")

    return {"seed": seed, "done": done, "min": minutes, "final": score[-1], "best": max(score), "t10": t10, "spm": spm,
            "ent": ent, "inf": inf, "mal": mal, "mode": mode, "at": [at(m) for m in (30, 60, 120, 180, 240)]}


def eval_arm(arm: Path) -> None:
    done = sorted((p for p in (arm / "seeds").glob("*/dev_confirm.json")), key=lambda p: int(p.parent.name))
    scores = []
    for path in done:
        r = json.loads(path.read_text())
        scores.append(r["ladder_score"])
        levels = " ".join(f"{v:.3f}" for v in r["pass@1_by_level"].values())
        print(f"  {arm.name[:34]:34} s{path.parent.name} dev16 {r['ladder_score']:.4f} pass@16 {r['pass@16']:.3f} "
              f"tokens {r['mean_tokens']:4.0f} trunc {r['truncated']:.3f} | {levels}")
    if scores:
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"= {arm.name:36} n={len(scores)} dev16 {statistics.mean(scores):.4f} ± {sd:.4f}")


def main(args: list[str]) -> None:
    for a in args:
        arm = Path(a.rstrip("/"))
        if (arm / "eval.json").exists():
            eval_arm(arm)
            continue
        seeds_dir = arm / "seeds"
        names = {p.name for p in seeds_dir.iterdir() if p.is_dir() and p.name.isdigit()} if seeds_dir.exists() else set()
        ck = CKPT / arm / "seeds"
        if ck.exists():
            names |= {p.name for p in ck.iterdir() if p.is_dir() and p.name.isdigit()}
        rows = [r for r in (seed_summary(arm, s) for s in sorted(names, key=int)) if r]
        if not rows:
            continue
        for r in rows:
            t10 = f"{r['t10']:.0f}" if r["t10"] is not None else "-"
            at = " ".join("  -  " if x != x else f"{x:.3f}" for x in r["at"])
            print(f"  {arm.name[:34]:34} s{r['seed']} {'done' if r['done'] else 'RUN '} {r['min']:4.0f}m final {r['final']:.3f} "
                  f"best {r['best']:.3f} t.10 {t10:>4} spm {r['spm']:5.1f} ent {r['ent']:.2f} inf {r['inf']:4.1f} {r['mode']:9} | {at}")
        done = [r for r in rows if r["done"]]
        if done:
            finals = [r["final"] for r in done]
            sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
            modes = {m: sum(r["mode"] == m for r in done) for m in ("ok", "one-guess", "diverged")}
            t10s = [r["t10"] for r in done if r["t10"] is not None]
            print(f"= {arm.name:36} n={len(done)} final {statistics.mean(finals):.3f} ± {sd:.3f}  "
                  f"ok/1-guess/div {modes['ok']}/{modes['one-guess']}/{modes['diverged']}  "
                  f"median t.10 {statistics.median(t10s):.0f}m" if t10s else
                  f"= {arm.name:36} n={len(done)} final {statistics.mean(finals):.3f} ± {sd:.3f}  "
                  f"ok/1-guess/div {modes['ok']}/{modes['one-guess']}/{modes['diverged']}  t.10 never")


if __name__ == "__main__":
    main(sys.argv[1:])
