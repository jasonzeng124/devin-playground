"""One line per screen seed, then one line per arm: dev score, takeoff time and collapse mode.

    python scripts/summarize_screens.py calibration/research_r1/* [records/v3/track_a/001_ladder_constrained]

A seed's logs are read from <arm>/seeds/<i>/ when the run has finished, else from the running
run's output under $CKPT_DIR/<arm>/seeds/<i>/ (scripts/slurm/screen_seed.sh). Collapse modes,
from the last training rows:
  diverged   malformed rate > 0.5 (the policy emits garbage)
  one-guess  entropy < 0.03 and fewer than 10 informative groups (answers at once, stops exploring)
  ok         anything else
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


def seed_summary(arm: Path, seed: str) -> dict | None:
    done = (arm / "seeds" / seed / "train_log.jsonl").exists()
    src = arm / "seeds" / seed if done else CKPT / arm / "seeds" / seed
    train, evals = _rows(src / "train_log.jsonl"), _rows(src / "eval_log.jsonl")
    if not train or not evals:
        return None
    last = train[-1]
    minutes = last["train_s"] / 60
    recent = [r for r in train if r["train_s"] >= last["train_s"] - 1800]
    spm = (recent[-1]["step"] - recent[0]["step"]) / max(1e-9, (recent[-1]["train_s"] - recent[0]["train_s"]) / 60)
    tail = train[-20:]
    ent = statistics.mean(r.get("entropy", 0) for r in tail)
    inf = statistics.mean(r.get("informative_groups", 0) for r in tail)
    mal = statistics.mean(r.get("malformed_rate", 0) for r in tail)
    mode = "diverged" if mal > 0.5 else "one-guess" if ent < 0.03 and inf < 10 else "ok"
    score = [e.get("pass_rate", e.get("ladder_score", 0)) for e in evals]
    t10 = next((e["train_s"] / 60 for e, x in zip(evals, score) if x >= 0.10), None)

    def at(m):
        c = [x for e, x in zip(evals, score) if e["train_s"] <= m * 60]
        return c[-1] if c else float("nan")

    return {"seed": seed, "done": done, "min": minutes, "final": score[-1], "best": max(score), "t10": t10, "spm": spm,
            "ent": ent, "inf": inf, "mal": mal, "mode": mode, "at": [at(m) for m in (30, 60, 120, 180, 240)]}


def main(args: list[str]) -> None:
    for a in args:
        arm = Path(a.rstrip("/"))
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
