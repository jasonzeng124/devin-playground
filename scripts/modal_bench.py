"""Run an arbitrary `python -m` / script command from this repo on a Modal GPU and stream its output.

    modal run scripts/modal_bench.py --gpu H100 --cmd "python scripts/bench_generate.py --fixed-length"
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers<5", "numpy", "accelerate")
    .add_local_dir(REPO / "rlvr_speedrun", "/root/rlvr/rlvr_speedrun")
    .add_local_dir(REPO / "scripts", "/root/rlvr/scripts")
    .add_local_dir(REPO / "data", "/root/rlvr/data")
)
app = modal.App("rlvr-speedrun-bench", image=image)


def _run(cmd: str) -> str:
    proc = subprocess.run(shlex.split(cmd), cwd="/root/rlvr", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/root/rlvr"})
    output = proc.stdout + "\n--- stderr ---\n" + proc.stderr[-6000:]
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{output}")
    return output


@app.function(gpu="H100", timeout=3600)
def run_h100(cmd: str) -> str:
    return _run(cmd)


@app.function(gpu="A100-80GB", timeout=3600)
def run_a100(cmd: str) -> str:
    return _run(cmd)


@app.local_entrypoint()
def main(gpu: str = "H100", cmd: str = "python scripts/bench_generate.py"):
    fn = run_h100 if gpu.upper() == "H100" else run_a100
    print(fn.remote(cmd))
