"""scripts/aws/box_s3.sh home_paths: the paths under ~ that a box's queued items read, and what it refuses.

The script itself needs AWS, so the test runs its two pure functions (items, home_paths) as they stand in the file
against a small git repository of configs.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aws" / "box_s3.sh"
HOME = "/home/someone"
TOKENS = f"{HOME}/orcd/pool/data/dm_math/mix_0.2.bin"
TOKENIZER = f"{HOME}/orcd/pool/data/tinyllama-tokenizer"

pytestmark = pytest.mark.skipif(
    not all(shutil.which(x) for x in ("bash", "git", "jq")), reason="needs bash, git and jq"
)


def home_paths(tmp_path, configs, queue, extra=""):
    repo = tmp_path / "repo"
    repo.mkdir()
    for d, cfg in configs.items():
        (repo / d).mkdir(parents=True)
        (repo / d / "config.json").write_text(json.dumps(cfg))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(git + ["add", "."], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "configs"], cwd=repo, check=True)
    q = tmp_path / "q.txt"
    q.write_text(queue)
    text = SCRIPT.read_text()
    items = next(line for line in text.splitlines() if line.startswith("items() {"))
    start = text.index("home_paths() {")
    func = text[start : text.index("\n}\n", start) + 3]
    prog = f'set -euo pipefail\n{items}\n{func}\nhome_paths HEAD "$1" | sort -u\n'
    r = subprocess.run(
        ["bash", "-c", prog, "bash", str(q)],
        cwd=repo,
        env={**os.environ, "HOME": HOME, "HOME_EXTRA": extra},
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.split(), r.stderr


def test_rl_only_screen_reads_its_checkpoint_and_not_the_token_file(tmp_path):
    cfg = {
        "screen_pretrained": "records/v3/track_d/007_x",
        "model": "(pretrained by run_seed_v3.sh)",
        "v3": {"pretrain": {"tokens": TOKENS, "tokenizer": TOKENIZER, "minutes": 56}},
    }
    paths, warn = home_paths(tmp_path, {"calibration/r1/rl": cfg}, "calibration/r1/rl 3\n\n# a comment\n")
    assert paths == [
        "orcd/pool/data/tinyllama-tokenizer",
        "orcd/pool/rlvr-ckpts/records/v3/track_d/007_x/seeds/3/pretrained",
    ]
    assert warn == ""


def test_screen_with_its_own_pretraining_reads_the_token_file_as_a_tar(tmp_path):
    cfg = {"model": "(pretrained by run_seed_v3.sh)", "v3": {"pretrain": {"tokens": TOKENS, "tokenizer": TOKENIZER}}}
    only = dict(cfg, pretrain_only=True)
    paths, warn = home_paths(
        tmp_path,
        {"calibration/r2/e2e": cfg, "calibration/r2/pre": only},
        "calibration/r2/e2e 0\ncalibration/r2/e2e 1\ncalibration/r2/pre 0\n",
    )
    assert paths == ["orcd/pool/data/dm_math/mix_0.2.bin.tar", "orcd/pool/data/tinyllama-tokenizer"]
    assert warn == ""


def test_track_a_screen_reads_the_model_cache_and_home_extra_is_added(tmp_path):
    paths, warn = home_paths(
        tmp_path, {"calibration/r3/a": {"model": "Qwen/Qwen3-1.7B-Base"}}, "calibration/r3/a 0\n", extra="orcd/pool/more"
    )
    assert paths == [".cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base.tar", "orcd/pool/more"]
    assert warn == ""


def test_items_the_box_cannot_run_are_warned_about(tmp_path):
    outside = {"v3": {"pretrain": {"tokens": "/scratch/tokens.bin", "tokenizer": TOKENIZER}}}
    no_tok = {"v3": {"pretrain": {"tokens": TOKENS}}}
    record = {"v3": {"pretrain": {"tokens": TOKENS, "tokenizer": TOKENIZER}}}
    paths, warn = home_paths(
        tmp_path,
        {"calibration/r4/outside": outside, "calibration/r4/no_tok": no_tok, "records/v3/track_d/008_x": record},
        "calibration/r4/outside 0\ncalibration/r4/no_tok 0\nrecords/v3/track_d/008_x 0\ncalibration/r4/missing 0\n",
    )
    lines = warn.strip().split("\n")
    assert len(lines) == 4 and all(x.startswith("WARNING: ") for x in lines)
    assert "calibration/r4/outside 0 pretrains from files outside ~" in lines[0]
    assert "calibration/r4/no_tok 0 pretrains from files outside ~" in lines[1]
    assert "records/v3/track_d/008_x 0 is a record seed" in lines[2]
    assert "calibration/r4/missing has no config.json" in lines[3]
    # what can be staged is still listed, so `check` shows every problem at once
    assert "orcd/pool/data/dm_math/mix_0.2.bin.tar" in paths and "orcd/pool/data/tinyllama-tokenizer" in paths
