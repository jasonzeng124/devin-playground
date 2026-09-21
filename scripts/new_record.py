"""Scaffold a new record folder from an existing one and print the commands to fill it.

    python scripts/new_record.py <slug> --provider runpod [--from records/track_a/005_...] [--seeds 3]

Creates records/<track>/<NNN>_<slug>/ (next free number) with a config.json copied
from the parent record (RL hyperparameters + metadata; edit it to describe your
change) and a README.md skeleton. The record config doubles as the trainer
config: `python -m rlvr_speedrun.grpo --config <record>/config.json` reads every
key that matches a GRPOConfig field and ignores the rest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.validate_record import RANKED_GPU

METADATA_KEYS = ("track", "description", "model", "hardware", "provider", "seeds", "unranked", "parent")

README_TEMPLATE = """# {number:03d} — {title} ({model}, 1x H100)

<!-- What changed relative to {parent_name} and why. Keep the numbers below honest:
     RULES.md requires every launched run with this config to be disclosed. -->

Hardware: one {gpu} on {provider}; software versions are in
`seeds/*/result.json` (`environment`). Untimed warm-up: see `warmup_s` in each
`result.json`.

Exploratory runs before these seeds: <N> (describe or put them under `extra/`).

Command (one process per seed):

```bash
{train_command}
```

| seed | time to {target:.0%} | steps | final pass rate | FineWeb-Edu loss Δ | MMLU-lite Δ |
|-----:|------------:|------:|----------------:|-------------------:|------------:|
{seed_rows}

**mean ± sd s** (N={n}, 95% CI [..., ...] s) — paste the `validate_record.py`
output, then the `--compare-to {parent_name}` output:

```
<validate_record.py output>
```
"""


def next_number(track_dir: Path) -> int:
    numbers = [int(m.group(1)) for p in track_dir.glob("*") if (m := re.fullmatch(r"(\d{3})_.+", p.name))]
    return max(numbers, default=-1) + 1


def latest_ranked(track_dir: Path) -> Path:
    candidates = []
    for path in sorted(track_dir.glob("*")):
        if not path.is_dir() or not re.fullmatch(r"\d{3}_.+", path.name):
            continue
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("unranked") is not True:
            candidates.append(path)
    if not candidates:
        raise SystemExit(f"no ranked record under {track_dir} to copy from; pass --from")
    return candidates[-1]


def scaffold(records_dir: Path, track: str, slug: str, provider: str, parent: Path | None, n_seeds: int) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise SystemExit("slug must be lowercase letters, digits and underscores")
    track_dir = records_dir / track
    track_dir.mkdir(parents=True, exist_ok=True)
    parent = parent or latest_ranked(track_dir)
    parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    number = next_number(track_dir)
    record = track_dir / f"{number:03d}_{slug}"
    if record.exists():
        raise SystemExit(f"{record} already exists")
    record.mkdir()

    config = {key: value for key, value in parent_config.items() if key not in METADATA_KEYS}
    config = {
        "track": track,
        "description": f"TODO: what changed relative to {parent.name}",
        "parent": parent.name,
        "model": parent_config.get("model", "Qwen/Qwen2.5-0.5B"),
        "hardware": f"1x {RANKED_GPU}",
        "provider": provider.lower(),
        "seeds": list(range(n_seeds)),
        **config,
    }
    (record / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_command = (
        f"python -m rlvr_speedrun.grpo --config {record}/config.json --device cuda \\\n"
        f"    --seed $SEED --out-dir {record}/seeds/$SEED\n"
        f"python -m rlvr_speedrun.capability --model {record}/seeds/$SEED/final \\\n"
        f"    --base {config['model']} --out {record}/seeds/$SEED/probe.json"
    )
    seed_rows = "\n".join(f"| {seed} | | | | | |" for seed in range(n_seeds))
    (record / "README.md").write_text(
        README_TEMPLATE.format(
            number=number,
            title=slug.replace("_", " "),
            model=config["model"],
            parent_name=parent.name,
            gpu=RANKED_GPU,
            provider=provider,
            train_command=train_command,
            target=float(config.get("target_solve_rate", 0.1)),
            seed_rows=seed_rows,
            n=n_seeds,
        ),
        encoding="utf-8",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slug", help="short_snake_case name, e.g. muon_optimizer")
    parser.add_argument("--provider", required=True, help="cloud/host every seed will run on (modal, runpod, lambda, ...)")
    parser.add_argument("--track", default="track_a", choices=["track_a", "track_b"])
    parser.add_argument("--from", dest="parent", type=Path, help="record to copy the config from (default: latest ranked record)")
    parser.add_argument("--seeds", type=int, default=3, help="number of official seeds (0..N-1, N >= 3)")
    parser.add_argument("--records-dir", type=Path, default=Path("records"))
    args = parser.parse_args()
    if args.seeds < 3:
        parser.error("ranked records need at least 3 seeds")
    record = scaffold(args.records_dir, args.track, args.slug, args.provider, args.parent, args.seeds)
    print(f"created {record}/")
    print(f"  1. edit {record}/config.json (description + the hyperparameters you changed)")
    print(f"  2. train every seed in {record}/config.json['seeds'] on {args.provider} (commands in README.md), or:")
    print(f"     modal run scripts/modal_run.py --gpu H100 --seeds {','.join(map(str, range(args.seeds)))} --out results/{record.name} --args \"--config {record}/config.json\"")
    print(f"  3. python scripts/validate_record.py {record} --compare-to <current holder>")
    print("  4. add the row to README.md and open a PR (the PR template lists the checklist)")


if __name__ == "__main__":
    main()
