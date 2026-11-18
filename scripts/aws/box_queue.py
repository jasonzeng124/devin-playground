"""Queue bookkeeping for a self-running AWS box (scripts/aws/box_boot.sh). Standard library only; runs on the box.

    flock queue/lock python3 scripts/aws/box_queue.py snap             # copy the queue state to queue/snap/
    python3 scripts/aws/box_queue.py restore <worker ids of the plan>  # rebuild queue/ on a new instance
    flock queue/lock python3 scripts/aws/box_queue.py retry            # put each FAILED item back once

The files, all under queue/ of the checkout (scripts/slurm/queue_worker.sbatch reads and writes the first three):

    <q>.txt        a queue, one "<config dir> <seed>" per line; any other *.txt (a holding file) only parks items
    done.txt       "<config dir> <seed> ok|FAILED", appended by the workers
    running/<id>   the item worker <id> has in hand
    retried.txt    items `retry` has put back
    staged/<q>.txt the queues as staged for this box (never edited on the box)
    snap/          a copy of every *.txt above plus running.txt ("<id> <config dir> <seed>")
    snap.tar       snap/ as one file: box_boot.sh sends it to S3 after the results, so a snapshot is whole and never
                   lists as done a result that is not in S3 yet

`restore` tolerates a snapshot whose files are from different moments: an item is run again only when it is neither
done, running nor queued, and never twice at once.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

NOT_QUEUES = {"done.txt", "retried.txt", "running.txt"}


def _lines(path: Path) -> list[str]:
    return [" ".join(ln.split()) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def _write(path: Path, lines: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(ln + "\n" for ln in lines))
    os.replace(tmp, path)


def _item(line: str) -> str:
    return " ".join(line.split()[:2])


def _done(lines: list[str]) -> tuple[set[str], Counter]:
    ok = {_item(ln) for ln in lines if ln.split()[-1] == "ok"}
    failed = Counter(_item(ln) for ln in lines if ln.split()[-1] == "FAILED")
    return ok, failed


def _staged(q: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    queues = {f.stem: _lines(f) for f in sorted((q / "staged").glob("*.txt"))}
    home = {}
    for name, items in queues.items():
        for item in items:
            home.setdefault(item, name)
    return queues, home


def _finished(item: str) -> bool:
    """A screen's results are copied into the checkout only when it succeeds, and its log then ends with '=== done'."""
    record, seed = item.split()
    out = Path(record) / "seeds" / seed
    log = out / "stdout.log"
    return (out / "train_log.jsonl").exists() and log.exists() and "=== done" in log.read_text(errors="replace")[-400:]


def snap(q: Path) -> None:
    new = q / "snap.new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    for f in sorted(q.glob("*.txt")):
        shutil.copy2(f, new / f.name)
    running = []
    for f in sorted((q / "running").glob("*")) if (q / "running").is_dir() else []:
        item = " ".join(f.read_text().split())
        if item:
            running.append(f"{f.name} {item}")
    _write(new / "running.txt", running)
    shutil.rmtree(q / "snap", ignore_errors=True)
    os.rename(new, q / "snap")
    with tarfile.open(q / "snap.tar.tmp", "w") as tar:
        tar.add(q / "snap", arcname="snap")
    os.replace(q / "snap.tar.tmp", q / "snap.tar")


def restore(q: Path, slots: list[str], finished=_finished) -> list[str]:
    """Rebuild queue/ from queue/snap.tar (an earlier instance's last snapshot) and queue/staged/; returns what it did."""
    queues, home = _staged(q)
    snapshot = q / "snap"
    if (q / "snap.tar").exists():
        shutil.rmtree(snapshot, ignore_errors=True)
        with tarfile.open(q / "snap.tar") as tar:
            tar.extractall(q)
    (q / "running").mkdir(parents=True, exist_ok=True)
    if not (snapshot / "running.txt").exists():
        for name, items in queues.items():
            _write(q / f"{name}.txt", items)
        return [f"fresh start: {sum(map(len, queues.values()))} items in {len(queues)} queues"]

    report = []
    done = _lines(snapshot / "done.txt")
    retried = _lines(snapshot / "retried.txt")
    ok, failed = _done(done)
    first = next(iter(queues), "orphans")
    front: dict[str, list[str]] = {}  # items to put at the front of a queue
    running: dict[str, str] = {}
    for line in _lines(snapshot / "running.txt"):
        worker, item = line.split()[0], _item(line.split(None, 1)[1])
        if item in ok or item in running.values() or any(item in items for items in front.values()):
            continue
        if finished(item):
            done.append(f"{item} ok")
            ok.add(item)
            report.append(f"{item}: its results are complete, marked ok")
        elif worker not in slots or worker in running:
            front.setdefault(home.get(item, first), []).append(item)
            report.append(f"{item}: worker {worker} is not in the plan, put at the front of its queue")
        else:
            running[worker] = item

    seen = set(running.values()) | {item for items in front.values() for item in items}
    files: dict[str, list[str]] = {name: [] for name in queues}
    for f in sorted(snapshot.glob("*.txt")):
        if f.name in NOT_QUEUES:
            continue
        kept = []
        for item in _lines(f):
            if item in ok or item in seen:
                continue
            seen.add(item)
            kept.append(item)
        files[f.stem] = kept
    for name, items in queues.items():
        for item in items:  # popped but in no running file yet, or lost between two files of a mixed snapshot
            if item not in ok and item not in seen and failed[item] <= retried.count(item):
                seen.add(item)
                front.setdefault(name, []).append(item)
                report.append(f"{item}: in no queue, not running and not done; put at the front of {name}")
    for name, items in front.items():
        files[name] = items + files.get(name, [])

    for name, items in files.items():
        _write(q / f"{name}.txt", items)
    _write(q / "done.txt", done)
    _write(q / "retried.txt", retried)
    for f in (q / "running").glob("*"):
        f.unlink()
    for worker, item in running.items():
        (q / "running" / worker).write_text(item + "\n")
    report.insert(0, f"restored: {len(ok)} done, {len(running)} running, {sum(map(len, files.values()))} queued")
    return report


def retry(q: Path) -> list[str]:
    """Put each FAILED item that is not done, queued, running or already retried at the front of its queue, once."""
    queues, home = _staged(q)
    ok, failed = _done(_lines(q / "done.txt"))
    retried = _lines(q / "retried.txt")
    busy = {_item(f.read_text()) for f in (q / "running").glob("*") if f.read_text().strip()}
    for f in q.glob("*.txt"):
        if f.name not in NOT_QUEUES:
            busy.update(_lines(f))
    back = [item for item in failed if item not in ok and item not in busy and item not in retried and item in home]
    for item in back:
        path = q / f"{home[item]}.txt"
        _write(path, [item] + _lines(path))
    if back:
        _write(q / "retried.txt", retried + back)
    return back


def main() -> None:
    q = Path("queue")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "snap":
        snap(q)
    elif command == "restore":
        print("\n".join(restore(q, sys.argv[2:])))
    elif command == "retry":
        for item in retry(q):
            print(f"put back once after a failure: {item}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
