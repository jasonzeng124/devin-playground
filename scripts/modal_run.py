"""Run a GRPO record (one or more seeds) on a Modal GPU.

    modal run scripts/modal_run.py --gpu H100 --seeds 0,1,2 --out results/record2 \
        --args "--model Qwen/Qwen2.5-0.5B --max-steps 300 ..."

`--args` is passed through to `python -m rlvr_speedrun.grpo` (seed/device/out-dir are set here).
Logs are written to a Modal volume and copied back to `--out` locally.
"""

from __future__ import annotations

import json
import shlex
import shutil
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


def _flag_value(grpo_args: list[str], flag: str) -> str | None:
    for index, arg in enumerate(grpo_args):
        if arg == flag and index + 1 < len(grpo_args):
            return grpo_args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _model_from_args(grpo_args: list[str]) -> str:
    """Resolve the base model the way `rlvr_speedrun.grpo` does: `--model` overrides `--config`.

    Runs locally (the config file lives on the launching machine) before any GPU is
    started, so a missing model fails fast instead of after training.
    """
    model = _flag_value(grpo_args, "--model")
    config_path = _flag_value(grpo_args, "--config")
    if model is None and config_path is not None:
        model = json.loads(Path(config_path).read_text(encoding="utf-8")).get("model")
    if not model:
        raise ValueError("--model is required in --args (or in --config) when --probe is enabled")
    return model


def _run(seed: int, run_name: str, grpo_args: list[str], probe: bool = True, base_model: str | None = None) -> str:
    out_dir = f"/results/{run_name}/seeds/{seed}"
    train_args = [arg for arg in grpo_args if arg != "--no-save"]
    if not probe:
        train_args.append("--no-save")
    cmd = ["python", "-m", "rlvr_speedrun.grpo", *train_args, "--seed", str(seed), "--device", "cuda", "--out-dir", out_dir]
    proc = subprocess.run(cmd, cwd="/root/rlvr", capture_output=True, text=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "stdout.log").write_text(proc.stdout + "\n" + proc.stderr)
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])
    if probe:
        probe_cmd = [
            "python",
            "-m",
            "rlvr_speedrun.capability",
            "--model",
            f"{out_dir}/final",
            "--base",
            base_model or _model_from_args(grpo_args),
            "--device",
            "cuda",
            "--out",
            f"{out_dir}/probe.json",
        ]
        probe_proc = subprocess.run(probe_cmd, cwd="/root/rlvr", capture_output=True, text=True)
        (Path(out_dir) / "stdout.log").write_text(
            proc.stdout + "\n" + proc.stderr + "\n--- capability probe ---\n" + probe_proc.stdout + "\n" + probe_proc.stderr
        )
        if probe_proc.returncode != 0:
            raise RuntimeError(probe_proc.stderr[-4000:])
        shutil.rmtree(f"{out_dir}/final")
        volume.commit()
    return (Path(out_dir) / "result.json").read_text()


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/results": volume})
def run_h100(seed: int, run_name: str, grpo_args: list[str], probe: bool = True, base_model: str | None = None) -> str:
    return _run(seed, run_name, grpo_args, probe, base_model)


@app.function(gpu="A100-80GB", timeout=6 * 3600, volumes={"/results": volume})
def run_a100(seed: int, run_name: str, grpo_args: list[str], probe: bool = True, base_model: str | None = None) -> str:
    return _run(seed, run_name, grpo_args, probe, base_model)


@app.local_entrypoint()
def main(gpu: str = "H100", seeds: str = "0", out: str = "results/modal_run", args: str = "", probe: bool = True):
    run_name = Path(out).name
    fn = run_h100 if gpu.upper() == "H100" else run_a100
    grpo_args = shlex.split(args)
    base_model = _model_from_args(grpo_args) if probe else None
    seed_list = [int(s) for s in seeds.split(",")]
    results = list(
        fn.map(seed_list, kwargs={"run_name": run_name, "grpo_args": grpo_args, "probe": probe, "base_model": base_model})
    )
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
