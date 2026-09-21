"""Validate every record under records/ (what CI runs on each PR).

Records whose config.json has "unranked": true (pipeline smoke tests, cheap-GPU
reproductions) are validated with --allow-fewer-seeds; everything else must
pass the full ranked-record checks and be linked from the README leaderboard.

    python scripts/validate_all_records.py [--records-dir records] [--readme README.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate_record import validate_record


def _is_unranked(record: Path) -> bool:
    config_path = record / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("unranked") is True


def validate_all(records_dir: Path, readme: Path | None) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    readme_text = readme.read_text(encoding="utf-8") if readme is not None and readme.is_file() else None
    records = sorted(p for track in sorted(records_dir.glob("track_*")) for p in track.iterdir() if p.is_dir())
    if not records:
        return False, [f"no records found under {records_dir}"]
    for record in records:
        unranked = _is_unranked(record)
        valid, message = validate_record(record, allow_fewer_seeds=unranked)
        label = "unranked" if unranked else "ranked"
        lines.append(f"== {record.parent.name}/{record.name} ({label}): {'OK' if valid else 'FAIL'}")
        lines.extend("   " + line for line in message.splitlines())
        ok = ok and valid
        if valid and not unranked and readme_text is not None:
            link = f"{records_dir.name}/{record.parent.name}/{record.name}"
            if link not in readme_text:
                ok = False
                lines.append(f"   FAIL: ranked record is not linked from {readme} (add it to the leaderboard table)")
    return ok, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-dir", type=Path, default=Path("records"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"), help="leaderboard file every ranked record must link from; pass '' to skip")
    args = parser.parse_args()
    readme = args.readme if str(args.readme) not in {"", "."} else None
    ok, lines = validate_all(args.records_dir, readme)
    print("\n".join(lines))
    print("all records valid" if ok else "some records failed validation")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
