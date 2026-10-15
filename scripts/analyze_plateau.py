"""Where does a recipe plateau, and which threshold gives the least noisy clock?

Reads `seeds/*/eval_log.jsonl` of a calibration run (threshold disabled, long
`max_steps`; e.g. calibration/plateau_005) and prints a markdown report:

* the mean learning curve across seeds (pass rate vs step);
* the plateau, estimated as the mean pass rate over the last quarter of evals;
* for each candidate threshold: how many seeds cross it, the mean and CV of
  steps-to-first-crossing, the mean curve's slope there, and how much of the
  crossing step is eval noise (Wilson half-width / slope) — the quantities
  docs/threshold_calibration.md asks for. Steps, not seconds, so any GPU works.

    python scripts/analyze_plateau.py calibration/plateau_005 [--md docs/plateau_calibration.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

Z95 = 1.959964


def load_curves(run: Path) -> dict[str, list[tuple[int, float, int]]]:
    curves = {}
    for seed_dir in sorted((run / "seeds").iterdir()):
        path = seed_dir / "eval_log.jsonl"
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            curves[seed_dir.name] = [(int(r["step"]), float(r["pass_rate"]), int(r["n"])) for r in rows]
    return curves


def mean_curve(curves: dict[str, list[tuple[int, float, int]]]) -> list[tuple[int, float, float, int]]:
    """(step, mean, sd, n_seeds) at every step that at least two seeds evaluated."""
    by_step: dict[int, list[float]] = {}
    for rows in curves.values():
        for step, rate, _ in rows:
            by_step.setdefault(step, []).append(rate)
    return [(s, statistics.mean(v), statistics.stdev(v), len(v)) for s, v in sorted(by_step.items()) if len(v) >= 2]


def first_crossing(rows: list[tuple[int, float, int]], threshold: float) -> int | None:
    return next((step for step, rate, _ in rows if rate >= threshold), None)


def wilson_halfwidth(p: float, n: int) -> float:
    denom = 1 + Z95**2 / n
    return Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom


def report(run: Path) -> list[str]:
    curves = load_curves(run)
    if not curves:
        raise SystemExit(f"no seeds/*/eval_log.jsonl under {run}")
    curve = mean_curve(curves)
    tail = curve[-max(1, len(curve) // 4) :]
    plateau = statistics.mean(m for _, m, _, _ in tail)
    n_eval = next(iter(curves.values()))[0][2]
    lines = [
        f"# Plateau calibration: {run.name}",
        "",
        f"Seeds: {len(curves)} ({', '.join(curves)}); evals on n = {n_eval}; last step {curve[-1][0]}.",
        f"Plateau estimate (mean of the last {len(tail)} mean-curve points, steps {tail[0][0]}-{tail[-1][0]}): **{plateau:.3f}**.",
        "",
        "## Mean curve",
        "",
        "| step | mean pass rate | sd across seeds | seeds |",
        "|--:|--:|--:|--:|",
    ]
    lines += [f"| {s} | {m:.4f} | {sd:.4f} | {k} |" for s, m, sd, k in curve]
    lines += [
        "",
        "## Candidate thresholds",
        "",
        "Steps to the first eval at or above the threshold. `noise (steps)` is the Wilson 95 % half-width at the",
        "threshold divided by the mean curve's local slope: how far eval sampling noise alone moves a crossing.",
        "",
        "| threshold | % of plateau | seeds crossed | steps mean ± sd | CV | slope (pp/100 steps) | noise (steps) | noise / sd |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    top = max(m for _, m, _, _ in curve)
    for i in range(2, 40):
        threshold = round(0.05 * i / 2, 3)
        if threshold > top:
            break
        steps = [c for c in (first_crossing(rows, threshold) for rows in curves.values()) if c is not None]
        above = next((j for j, (_, m, _, _) in enumerate(curve) if m >= threshold), None)
        slope = None
        if above is not None and above > 0:
            (s0, m0, _, _), (s1, m1, _, _) = curve[above - 1], curve[above]
            slope = (m1 - m0) / (s1 - s0) if s1 > s0 else None
        noise = wilson_halfwidth(threshold, n_eval) / slope if slope and slope > 0 else None
        if len(steps) >= 2:
            mean, sd = statistics.mean(steps), statistics.stdev(steps)
            stats = f"{mean:.0f} ± {sd:.0f} | {sd / mean:.2f}"
            ratio = f"{noise / sd:.2f}" if noise is not None and sd > 0 else "–"
        else:
            stats, ratio = "– | –", "–"
        lines.append(
            f"| {threshold:.3f} | {threshold / plateau:.0%} | {len(steps)}/{len(curves)} | {stats} | "
            f"{slope * 1e4:.2f} | {noise:.0f} | {ratio} |" if slope and noise is not None else
            f"| {threshold:.3f} | {threshold / plateau:.0%} | {len(steps)}/{len(curves)} | {stats} | – | – | {ratio} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    text = "\n".join(report(args.run)) + "\n"
    print(text)
    if args.md:
        args.md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
