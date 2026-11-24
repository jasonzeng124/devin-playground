"""scripts/aws/u25_box_item.sh and u25_rung.sh's NPROC/RESUME_MIN options, with a fake python (no GPU, no torch)."""
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE = r"""#!/bin/bash
# Fake interpreter: records its arguments; pretrain_scratch.py fails with an OOM at micro 64 when FAKE_OOM64=1.
echo "$*" >>"$FAKE_CALLS"
if [[ $1 == -c ]]; then exec "$REAL_PY" "$@"; fi
args=("$@")
for i in "${!args[@]}"; do
  case ${args[$i]} in
    --out) out=${args[$((i + 1))]} ;;
    --out-dir) outdir=${args[$((i + 1))]} ;;
    --micro-batch) micro=${args[$((i + 1))]} ;;
  esac
done
if [[ " $* " == *" scripts/pretrain_scratch.py "* ]]; then
  if [[ ${FAKE_OOM64:-0} == 1 && $micro == 64 ]]; then echo "torch.OutOfMemoryError: CUDA out of memory"; exit 1; fi
  mkdir -p "$out/resume"; echo x >"$out/resume/state.pt"
  echo '{"step": 1}' >>"$out/pretrain_log.jsonl"; echo w >"$out/model.safetensors"; exit 0
fi
if [[ " $* " == *" scripts/research/probe_reasoning.py "* ]]; then mkdir -p "$outdir"; exit 0; fi
exit 3
"""


@pytest.fixture
def box(tmp_path):
    work = tmp_path / "work"
    for sub in ("scripts/aws", "scripts/research"):
        (work / sub).mkdir(parents=True)
    for f in ("scripts/aws/u25_box_item.sh", "scripts/research/u25_rung.sh"):
        shutil.copy(REPO / f, work / f)
    (work / "queue").mkdir()
    (work / "queue" / "lock").touch()
    run = work / "calibration/research_x/u25_6x"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"u25": {"width": 64, "layers": 2, "max_tokens": 1000, "minutes": 5}}))
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PYTHON=str(fake), REAL_PY=sys.executable, FAKE_CALLS=str(tmp_path / "calls"),
               CKPT_DIR=str(tmp_path / "ckpt"), DATA=str(tmp_path / "data"), FINEMATH=str(tmp_path / "fm.bin"),
               U25_RETRY_S="0")
    return work, env, tmp_path


def test_box_item_oom_fallback_ddp_and_handover(box):
    work, env, tmp = box
    (work / "queue/box_after.txt").write_text("calibration/a 0\ncalibration/b 1\n")
    (work / "queue/box.txt").write_text("")
    env.update(U25_GPUS="0,1", U25_AFTER="queue/box_after.txt", U25_TO="queue/box.txt", FAKE_OOM64="1")
    r = subprocess.run(["bash", "scripts/aws/u25_box_item.sh", "calibration/research_x/u25_6x", "0"], cwd=work, env=env)
    assert r.returncode == 0
    calls = (tmp / "calls").read_text().splitlines()
    train = [c for c in calls if "pretrain_scratch.py" in c]
    assert len(train) == 2 and "--micro-batch 64" in train[0] and "--micro-batch 32" in train[1]
    assert all("--nproc_per_node 2" in c and "--ddp" in c and "--resume-every-minutes 30 --resume" in c for c in train)
    out = tmp / "ckpt/calibration/research_x/u25_6x/seeds/0/pretrained"
    assert (out / "model.safetensors").exists() and not (out / "resume").exists()
    assert (work / "calibration/research_x/u25_6x/seeds/0/pretrain_log.jsonl").exists()
    assert (work / "queue/box.txt").read_text() == "calibration/a 0\ncalibration/b 1\n"
    assert (work / "queue/box_after.txt").read_text() == ""
    # a second run: already trained, so probe only; the handover does not repeat
    r = subprocess.run(["bash", "scripts/aws/u25_box_item.sh", "calibration/research_x/u25_6x", "0"], cwd=work, env=env)
    assert r.returncode == 0
    assert sum("pretrain_scratch.py" in c for c in (tmp / "calls").read_text().splitlines()) == 2
    assert (work / "queue/box.txt").read_text() == "calibration/a 0\ncalibration/b 1\n"


def test_rung_defaults_unchanged(box):
    work, env, tmp = box
    out = tmp / "ckpt/calibration/research_x/u25_6x/seeds/0/pretrained"
    out.mkdir(parents=True)
    (out / "pretrain_log.jsonl").write_text("{}\n")
    r = subprocess.run(["bash", "scripts/research/u25_rung.sh", "calibration/research_x/u25_6x", "64", "2", "1000", "5"],
                       cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "not overwriting" in r.stderr
    # with RESUME_MIN and no state, the old log is moved aside and training starts again, single process
    for name in ("mix_0.2.bin", "dm_math.bin"):
        (tmp / "data/dm_math").mkdir(parents=True, exist_ok=True)
        (tmp / "data/dm_math" / name).write_bytes(b"\0\0")
    (tmp / "fm.bin").write_bytes(b"\0\0")
    env.update(RESUME_MIN="5", LOCAL_DIR=str(tmp / "local"))
    r = subprocess.run(["bash", "scripts/research/u25_rung.sh", "calibration/research_x/u25_6x", "64", "2", "1000", "5"],
                       cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "pretrain_log.failed.0.jsonl").exists()
    train = [c for c in (tmp / "calls").read_text().splitlines() if "pretrain_scratch.py" in c]
    assert len(train) == 1 and "torch.distributed.run" not in train[0] and "--ddp" not in train[0]
