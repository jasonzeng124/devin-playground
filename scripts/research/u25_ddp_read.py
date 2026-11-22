"""Read U25's data-parallel check (scripts/research/u25_ddp_check.sh) against its rule (LOG 2026-11-22).

    python scripts/research/u25_ddp_read.py <dir>

Writes <dir>/check.json and prints it. Rule:
1. the ranks stay identical: every rank_spread in the ddp and resume arms is 0.0;
2. ddp's final val loss is within 0.05 of single's;
3. ddp's steady tokens a second (from the step-500 row to the last logged row) is at least 0.9 x GPUs x single's;
4. resume: the first part was stopped by the timeout, the rerun started from a saved step above 0 and ended at
   ddp's step and tokens, and its final val loss is within 0.05 of ddp's.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _steady(rows: list) -> float | None:
    logged = [r for r in rows if "lr" in r]
    first = next((r for r in logged if r["step"] == 500), None)
    if first is None or logged[-1]["step"] <= first["step"]:
        return None
    last = logged[-1]
    return (last["tokens"] - first["tokens"]) / ((last["minutes"] - first["minutes"]) * 60)


def _arm(root: Path, name: str) -> dict:
    rows = _rows(root / name / "pretrain_log.jsonl")
    final = rows[-1] if rows and "lr" not in rows[-1] else None
    return {"final": final, "steady_tok_per_s": _steady(rows),
            "rank_spread": [r["rank_spread"] for r in rows if "rank_spread" in r]}


def main() -> None:
    root = Path(sys.argv[1])
    arms = {name: _arm(root, name) for name in ("single", "ddp", "resume")}
    single, ddp, resume = (arms[k]["final"] for k in ("single", "ddp", "resume"))
    stdout = (root / "resume" / "stdout.log").read_text() if (root / "resume" / "stdout.log").exists() else ""
    resumed = re.findall(r"resumed from \S+ at step (\d+)", stdout)
    rc_file = root / "resume" / "first_rc.txt"
    first_rc = int(rc_file.read_text().strip()) if rc_file.exists() else None
    gpus = ddp.get("gpus") if ddp else None

    spreads = arms["ddp"]["rank_spread"] + arms["resume"]["rank_spread"]
    rule = {"ranks_identical": bool(spreads) and all(s == 0.0 for s in spreads)}
    rule["val_close"] = bool(single and ddp) and abs(ddp["val_loss"] - single["val_loss"]) <= 0.05
    s1, s2 = arms["single"]["steady_tok_per_s"], arms["ddp"]["steady_tok_per_s"]
    scaling = s2 / (gpus * s1) if s1 and s2 and gpus else None
    rule["scaling"] = scaling is not None and scaling >= 0.9
    rule["resume"] = (first_rc == 124 and bool(resumed) and int(resumed[-1]) > 0 and bool(resume and ddp)
                      and resume["step"] == ddp["step"] and resume["tokens"] == ddp["tokens"]
                      and abs(resume["val_loss"] - ddp["val_loss"]) <= 0.05)
    result = {"gpus": gpus, "arms": arms, "scaling_per_gpu": scaling, "first_part_rc": first_rc,
              "resumed_at_step": int(resumed[-1]) if resumed else None, "rule": rule, "rule_met": all(rule.values())}
    (root / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"gpus": gpus, "scaling_per_gpu": scaling, "resumed_at_step": result["resumed_at_step"], **rule,
                      "rule_met": result["rule_met"]}))


if __name__ == "__main__":
    main()
