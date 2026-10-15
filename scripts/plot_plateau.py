"""Plots for a plateau calibration run (companion to scripts/analyze_plateau.py).

    uv run --with matplotlib python scripts/plot_plateau.py calibration/plateau_005 --out docs/img/plateau_005

Writes <out>_curves.png (every seed's stopping-set curve, the mean with a ±1 sd band, and the
candidate thresholds), <out>_crossings.png (steps to first reach each threshold, one dot per seed)
and <out>_confirm.png (final stopping-set vs confirmation-set pass rate, with Wilson 95% bars).
Seeds whose curve stops short of max_steps (e.g. killed runs) are drawn dashed and left out of
the mean and the crossing plot's summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
CURRENT = 0.20  # benchmark.THRESHOLD (v2)


def load(run: Path) -> tuple[dict[str, list[dict]], dict[str, dict], int]:
    max_steps = json.loads((run / "config.json").read_text())["max_steps"]
    curves, confirms = {}, {}
    for seed in sorted((run / "seeds").iterdir(), key=lambda p: int(p.name)):
        log = seed / "eval_log.jsonl"
        if log.exists():
            curves[seed.name] = [json.loads(line) for line in log.open()]
        if (seed / "confirm.json").exists():
            confirms[seed.name] = json.loads((seed / "confirm.json").read_text())
    return curves, confirms, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="path prefix for the PNGs")
    args = parser.parse_args()
    curves, confirms, max_steps = load(args.run)
    full = {s: c for s, c in curves.items() if c and c[-1]["step"] >= max_steps}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    cmap = plt.get_cmap("tab20")

    # 1. learning curves
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (seed, curve) in enumerate(curves.items()):
        ax.plot([e["step"] for e in curve], [e["pass_rate"] for e in curve], lw=1, alpha=0.55, color=cmap(i % 20),
                ls="-" if seed in full else "--", label=f"seed {seed}" + ("" if seed in full else " (partial)"))
    if full:
        steps = [e["step"] for e in next(iter(full.values()))]
        rows = [[e["pass_rate"] for e in c] for c in full.values()]
        mean = [statistics.fmean(col) for col in zip(*rows)]
        sd = [statistics.stdev(col) if len(col) > 1 else 0.0 for col in zip(*rows)]
        ax.fill_between(steps, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)], color="k", alpha=0.08)
        ax.plot(steps, mean, color="k", lw=2.4, label=f"mean of {len(full)} full seeds (±1 sd)")
    for t in THRESHOLDS:
        ax.axhline(t, color="crimson" if t == CURRENT else "grey", lw=1.2 if t == CURRENT else 0.6, ls=":", zorder=0)
        ax.text(max_steps * 1.005, t, f"{t:.3f}", va="center", fontsize=8, color="crimson" if t == CURRENT else "grey")
    ax.set(xlabel="training step", ylabel="pass rate (2,000-puzzle stopping set, greedy)", xlim=(0, max_steps), ylim=(0, None),
           title="Qwen2.5-0.5B GRPO on Countdown: record-005 recipe, threshold disabled")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{args.out}_curves.png")

    # 2. steps to first crossing per threshold
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for j, t in enumerate(THRESHOLDS):
        firsts = [next((e["step"] for e in curve if e["pass_rate"] >= t), None) for curve in full.values()]
        hits, misses = [f for f in firsts if f is not None], firsts.count(None)
        xs = [j + (k - (len(hits) - 1) / 2) * 0.06 for k in range(len(hits))]
        ax.scatter(xs, sorted(hits), s=22, color="crimson" if t == CURRENT else "steelblue", zorder=3)
        if len(hits) > 1:
            m, s = statistics.fmean(hits), statistics.stdev(hits)
            ax.errorbar(j + 0.32, m, yerr=s, fmt="D", color="k", ms=4, capsize=3)
            ax.text(j, -0.09, f"CV {s / m:.2f}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        if misses:
            ax.text(j, max_steps * 0.97, f"{misses} never", ha="center", fontsize=8, color="grey")
    ax.set_xticks(range(len(THRESHOLDS)), [f"{t:.3f}" for t in THRESHOLDS])
    ax.set(xlabel="\nthreshold (pass rate)", ylabel="steps to first eval ≥ threshold", ylim=(0, max_steps),
           title=f"How noisy is the clock at each threshold? ({len(full)} seeds; diamond = mean ± sd)")
    fig.tight_layout()
    fig.savefig(f"{args.out}_crossings.png")

    # 3. stopping set vs confirmation set at the final checkpoint
    if confirms:
        fig, ax = plt.subplots(figsize=(5.2, 5))
        for seed, c in confirms.items():
            stop, conf = curves[seed][-1]["pass_rate"], c["confirm"]
            lo, hi = conf["wilson_95_ci"]
            ax.errorbar(stop, conf["pass_rate"], yerr=[[conf["pass_rate"] - lo], [hi - conf["pass_rate"]]], fmt="o", color="steelblue", capsize=2)
            ax.annotate(seed, (stop, conf["pass_rate"]), xytext=(4, -10), textcoords="offset points", fontsize=7)
        lim = max(max(curves[s][-1]["pass_rate"] for s in confirms), max(c["confirm"]["pass_rate"] for c in confirms.values())) * 1.15
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="final stopping-set pass rate (n=2,000)", ylabel="confirmation-set pass rate (n=4,000)",
               title="Stopping set vs held-out confirmation set")
        fig.tight_layout()
        fig.savefig(f"{args.out}_confirm.png")
    print(f"wrote {args.out}_*.png ({len(curves)} seeds, {len(full)} full)")


if __name__ == "__main__":
    main()
