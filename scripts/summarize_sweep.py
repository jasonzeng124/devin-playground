"""Compare the variants of a sweep made by scripts/slurm/make_sweep.py.

    python scripts/summarize_sweep.py calibration/sweep_001 [--thresholds 0.10 0.15 0.20] [--plot docs/img/sweep_001.png]

Sweep runs have the threshold disabled, so each seed's eval_log gives the time to any threshold:
the clock value (`wall_s`) of its first eval at or above it. Per variant and threshold: seeds that
reached it, steps and seconds to it (mean, CV). Also seconds per step and the final and
confirmation pass rates. Seconds come from whatever GPU ran the seed (H200 on Engaging), so
compare them only within a sweep; the ranked clock is the H100. Unfinished seeds count only if
their curve already crossed (use `done` to see how many finished).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def stat(xs: list[float]) -> str:
    if not xs:
        return "-"
    m = statistics.fmean(xs)
    return f"{m:.0f} ({statistics.stdev(xs) / m:.2f})" if len(xs) > 1 else f"{m:.0f}"


def load(variant: Path) -> list[dict]:
    seeds = []
    for seed in sorted((variant / "seeds").glob("*")):
        log = seed / "eval_log.jsonl"
        if not log.exists():
            continue
        evals = [json.loads(line) for line in log.open()]
        train = [json.loads(line) for line in (seed / "train_log.jsonl").open()] if (seed / "train_log.jsonl").exists() else []
        confirm = seed / "confirm.json"
        seeds.append({
            "evals": evals,
            "done": (seed / "result.json").exists(),
            "s_per_step": train[-1]["wall_s"] / train[-1]["step"] if train else None,
            "confirm": json.loads(confirm.read_text())["confirm"]["pass_rate"] if confirm.exists() else None,
        })
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    variants = {v.name: s for v in sorted(p for p in args.sweep.iterdir() if p.is_dir()) if (s := load(v))}
    head = ["variant", "done"] + [f"≥{t:.2f}: hit / steps / s (CV)" for t in args.thresholds] + ["s/step", "final", "confirm"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    secs_at = {}
    for name, seeds in variants.items():
        cells = [name, f"{sum(s['done'] for s in seeds)}/{len(seeds)}"]
        for t in args.thresholds:
            first = [next((e for e in s["evals"] if e["pass_rate"] >= t), None) for s in seeds]
            hits = [f for f in first if f is not None]
            secs_at[name, t] = [f["wall_s"] for f in hits]
            cells.append(f"{len(hits)}/{len(seeds)} / {stat([f['step'] for f in hits])} / {stat(secs_at[name, t])}")
        rates = [s["s_per_step"] for s in seeds if s["s_per_step"]]
        finals = [s["evals"][-1]["pass_rate"] for s in seeds if s["done"]]
        confirms = [s["confirm"] for s in seeds if s["confirm"] is not None]
        cells += [f"{statistics.fmean(rates):.2f}" if rates else "-", f"{statistics.fmean(finals):.3f}" if finals else "-",
                  f"{statistics.fmean(confirms):.3f}" if confirms else "-"]
        print("| " + " | ".join(cells) + " |")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(args.thresholds), figsize=(4.2 * len(args.thresholds), 4.4), dpi=150, sharey=True)
        for ax, t in zip(axes if len(args.thresholds) > 1 else [axes], args.thresholds):
            for i, name in enumerate(variants):
                secs = secs_at[name, t]
                ax.scatter([i] * len(secs), secs, color="steelblue", s=18, zorder=3)
                if secs:
                    ax.scatter(i, statistics.geometric_mean(secs), marker="_", s=300, color="k", zorder=4)
                missed = len(variants[name]) - len(secs)
                if missed:
                    ax.text(i, 1.02, f"{missed}✗", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="crimson")
            ax.set_xticks(range(len(variants)), list(variants), rotation=45, ha="right", fontsize=8)
            ax.set(yscale="log", title=f"threshold {t:.2f}")
            ax.spines[["top", "right"]].set_visible(False)
        (axes[0] if len(args.thresholds) > 1 else axes).set_ylabel("training seconds to threshold (log; bar = geo. mean)")
        fig.suptitle(f"{args.sweep.name}: time to threshold per seed (✗ = seeds that never reached it)")
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
