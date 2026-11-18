import shutil
from pathlib import Path

from scripts.aws import box_queue


def _put(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(ln + "\n" for ln in lines))


def _get(path: Path):
    return path.read_text().splitlines()


def test_a_first_instance_starts_from_the_staged_queues(tmp_path):
    q = tmp_path / "queue"
    _put(q / "staged" / "a.txt", ["x/a 0", "x/a 1"])
    _put(q / "staged" / "d.txt", ["x/d 0"])
    assert box_queue.restore(q, ["w1"]) == ["fresh start: 3 items in 2 queues"]
    assert _get(q / "a.txt") == ["x/a 0", "x/a 1"] and _get(q / "d.txt") == ["x/d 0"]
    assert list((q / "running").iterdir()) == []


def test_a_later_instance_runs_every_unfinished_item_once(tmp_path):
    old = tmp_path / "old" / "queue"
    _put(old / "staged" / "a.txt", [f"x/a {s}" for s in range(7)])
    _put(old / "staged" / "d.txt", [f"x/d {s}" for s in range(4)])
    _put(old / "a.txt", ["x/a 4", "x/a 5", "x/a 4"])  # a repeated line
    _put(old / "d.txt", ["x/d 2"])
    _put(old / "hold.txt", ["x/d 3"])  # parked by hand
    _put(old / "done.txt", ["x/a 0 ok", "x/d 0 FAILED", "x/d 0 ok", "x/d 1 FAILED"])
    _put(old / "retried.txt", ["x/d 0"])
    _put(old / "running" / "w1", ["x/a 1"])  # was running
    _put(old / "running" / "w2", ["x/a 2"])  # finished, and its results reached S3, after this snapshot
    _put(old / "running" / "gone", ["x/a 3"])  # a worker the plan no longer has
    _put(old / "running" / "w3", [])  # idle
    box_queue.snap(old)  # x/a 6 is in no file: popped, not yet written to a running file
    assert _get(old / "snap" / "running.txt") == ["gone x/a 3", "w1 x/a 1", "w2 x/a 2"]

    new = tmp_path / "new" / "queue"
    shutil.copytree(old / "staged", new / "staged")
    shutil.copy(old / "snap.tar", new / "snap.tar")
    report = box_queue.restore(new, ["w1", "w2", "w3"], finished=lambda item: item == "x/a 2")
    assert report[0] == "restored: 3 done, 1 running, 6 queued"
    assert _get(new / "a.txt") == ["x/a 3", "x/a 6", "x/a 4", "x/a 5"]
    assert _get(new / "d.txt") == ["x/d 2"]  # x/d 1 failed once: `retry` puts it back, not `restore`
    assert _get(new / "hold.txt") == ["x/d 3"]
    assert {f.name: _get(f) for f in (new / "running").iterdir()} == {"w1": ["x/a 1"]}
    assert _get(new / "done.txt")[-1] == "x/a 2 ok"

    assert box_queue.retry(new) == ["x/d 1"]
    assert _get(new / "d.txt") == ["x/d 1", "x/d 2"]
    assert box_queue.retry(new) == []  # once only
    box_queue.snap(new)
    again = tmp_path / "again" / "queue"
    shutil.copytree(new / "staged", again / "staged")
    shutil.copy(new / "snap.tar", again / "snap.tar")
    box_queue.restore(again, ["w1", "w2", "w3"], finished=lambda item: False)
    for name in ("a.txt", "d.txt", "hold.txt", "done.txt", "retried.txt"):
        assert _get(again / name) == _get(new / name)


def test_restore_never_gives_one_item_to_two_workers(tmp_path):
    q = tmp_path / "queue"
    _put(q / "staged" / "a.txt", ["x/a 0", "x/a 1", "x/a 2"])
    _put(q / "snap" / "a.txt", ["x/a 0", "x/a 1", "x/a 2"])  # older than running.txt
    _put(q / "snap" / "running.txt", ["w1 x/a 0", "w2 x/a 0", "w3 x/a 1"])
    box_queue.restore(q, ["w1", "w2", "w3"], finished=lambda item: False)
    assert _get(q / "a.txt") == ["x/a 2"]
    assert {f.name: _get(f) for f in (q / "running").iterdir()} == {"w1": ["x/a 0"], "w3": ["x/a 1"]}


def test_retry_leaves_items_that_are_queued_running_done_or_unknown(tmp_path):
    q = tmp_path / "queue"
    _put(q / "staged" / "a.txt", [f"x/a {s}" for s in range(5)])
    _put(q / "a.txt", ["x/a 0"])
    _put(q / "running" / "w1", ["x/a 1"])
    _put(q / "done.txt", [f"x/a {s} FAILED" for s in range(5)] + ["x/a 2 ok", "x/other 0 FAILED"])
    assert box_queue.retry(q) == ["x/a 3", "x/a 4"]
    assert _get(q / "a.txt") == ["x/a 4", "x/a 3", "x/a 0"]
    assert _get(q / "retried.txt") == ["x/a 3", "x/a 4"]
