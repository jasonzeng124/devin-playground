"""Run a GRPO record (one or more seeds) on a Modal GPU.

    modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record2 \
        --args "--model Qwen/Qwen2.5-0.5B --max-steps 300 ..."

`--args` is passed through to `python -m rlvr_speedrun.grpo` (seed/device/out-dir are set here).
Logs are written to a Modal volume and copied back to `--out` locally.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOLUME_NAME = "rlvr-speedrun-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers<5", "numpy", "accelerate")
    .add_local_dir(REPO / "rlvr_speedrun", "/root/rlvr/rlvr_speedrun")
    .add_local_dir(REPO / "data", "/root/rlvr/data")
)
app = modal.App("rlvr-speedrun", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run(seed: int, run_name: str, grpo_args: list[str]) -> str:
    out_dir = f"/results/{run_name}/seeds/{seed}"
    cmd = ["python", "-m", "rlvr_speedrun.grpo", *grpo_args, "--seed", str(seed), "--device", "cuda", "--no-save", "--out-dir", out_dir]
    proc = subprocess.run(cmd, cwd="/root/rlvr", capture_output=True, text=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "stdout.log").write_text(proc.stdout + "\n" + proc.stderr)
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])
    return (Path(out_dir) / "result.json").read_text()


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/results": volume})
def run_h100(seed: int, run_name: str, grpo_args: list[str]) -> str:
    return _run(seed, run_name, grpo_args)


@app.function(gpu="A100-80GB", timeout=6 * 3600, volumes={"/results": volume})
def run_a100(seed: int, run_name: str, grpo_args: list[str]) -> str:
    return _run(seed, run_name, grpo_args)


@app.local_entrypoint()
def main(gpu: str = "H100", seeds: str = "0", out: str = "results/modal_run", args: str = ""):
    run_name = Path(out).name
    fn = run_h100 if gpu.upper() == "H100" else run_a100
    grpo_args = shlex.split(args)
    seed_list = [int(s) for s in seeds.split(",")]
    results = list(fn.map(seed_list, kwargs={"run_name": run_name, "grpo_args": grpo_args}))
    for seed, result in zip(seed_list, results):
        print(f"seed {seed}: {result}")
    local = Path(out)
    local.mkdir(parents=True, exist_ok=True)
    for entry in volume.listdir(f"/{run_name}", recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            dest = local / Path(entry.path).relative_to(run_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                for chunk in volume.read_file(entry.path):
                    fh.write(chunk)
    print(f"copied results to {local}")
