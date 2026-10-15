"""Benchmark v3 on Modal, in the v3 reference software: record seeds and the track D math-data pilot.

    modal run scripts/modal_v3.py::gpu_name               # which GPU "H100!" gets (ranked: NVIDIA H100 80GB HBM3)
    modal run --detach scripts/modal_v3.py::record_seed --record records/v3/track_a/000_baseline --seed 0
    modal volume get rlvr-v3 records/v3/track_a/000_baseline/seeds/0 <dest>   # logs only; weights stay in /tmp
    modal run scripts/modal_v3.py::prep                   # CPU: track D data + FineMath mix on the volume
    modal run --detach scripts/modal_v3.py::pilot         # 1 GPU: 120 min pretraining, then 4 h of RL
    modal volume get rlvr-v3 runs/finemath120 calibration/sweep_d15/finemath120   # small logs only

`record_seed` runs scripts/run_seed_v3.sh on one H100 SXM ("H100!" disables Modal's upgrade to
H200) and writes the seed directory to the volume. Track D needs `prep` first (data on the volume,
copied to local disk before the clock starts). The pilot mirrors the Engaging scripts: `prep` is d15_data.sbatch (finemath arm), `pilot` is d15_pre.sbatch
plus the queued RL run. Unranked: Modal's "H100" may be upgraded to an H200, as in the other pilots.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = Path("/vol")
MIX = VOL / "mix_finemath.bin"
RUN = VOL / "runs" / "finemath120"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.11")
    .uv_pip_install("vllm==0.30.0", "transformers==5.17.0", "pyarrow", "numpy", "accelerate", "huggingface_hub")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(REPO / "rlvr_speedrun", "/root/rlvr/rlvr_speedrun")
    .add_local_dir(REPO / "scripts", "/root/rlvr/scripts")
    .add_local_dir(REPO / "data", "/root/rlvr/data")
    .add_local_dir(REPO / "records" / "v3", "/root/rlvr/records/v3", ignore=["**/seeds/**", "**/extra/**"])
    .add_local_dir(REPO / "calibration" / "sweep_d15", "/root/rlvr/calibration/sweep_d15")
)
app = modal.App("rlvr-v3", image=image)
volume = modal.Volume.from_name("rlvr-v3", create_if_missing=True)


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/rlvr")


@app.function(cpu=16, memory=65536, timeout=4 * 3600, volumes={VOL: volume})
def prep() -> None:
    from huggingface_hub import hf_hub_download

    d = VOL / "track_d"
    sh("python", "scripts/prepare_track_d_data.py", "--out", str(d), "--workers", "16")
    volume.commit()
    if not (VOL / "finemath.bin").exists():
        for i in range(3):
            hf_hub_download("HuggingFaceTB/finemath", f"finemath-4plus/train-{i:05d}-of-00064.parquet",
                            repo_type="dataset", local_dir=VOL / "finemath")
        sh("python", "scripts/pretrain_scratch.py", "tokenize", "--parquet", str(VOL / "finemath"), "--tokenizer",
           str(d / "tokenizer"), "--out", str(VOL / "finemath.partial"), "--workers", "16")
        (VOL / "finemath.partial").rename(VOL / "finemath.bin")
    if not MIX.exists():
        sh("python", "scripts/math_mix.py", "mix", "--base", str(d / "fineweb-edu-tinyllama.bin"),
           "--extra", str(VOL / "finemath.bin"), "--frac", "0.2", "--out", str(MIX))
    volume.commit()


@app.function(gpu="H100", cpu=8, memory=98304, timeout=int(6.6 * 3600), volumes={VOL: volume})
def pilot() -> None:
    sh("nvidia-smi", "-L")
    local = Path("/tmp/mix.bin")
    shutil.copy(MIX, local)  # random reads from the network volume would slow pretraining (cf. RULES_v3 LOCAL_DATA_DIR)
    model = Path("/tmp/d110m_finemath_120min")
    sh("python", "scripts/pretrain_scratch.py", "train", "--tokens", str(local), "--tokenizer", str(VOL / "track_d" / "tokenizer"),
       "--out", str(model), "--minutes", "120", "--micro-batch", "32", "--max-tokens", "20000000000", "--seed", "0")
    out = RUN / "seeds" / "0"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(model / "pretrain_log.jsonl", out / "pretrain_log.jsonl")
    config = json.loads(Path("/root/rlvr/calibration/sweep_d15/finemath120/config.json").read_text())
    config["model"] = str(model)
    (RUN / "config.json").write_text(json.dumps(config, indent=2))
    volume.commit()
    try:
        sh("python", "-m", "rlvr_speedrun.grpo", "--config", str(RUN / "config.json"), "--device", "cuda", "--seed", "0",
           "--out-dir", str(out), "--no-save")
    finally:
        volume.commit()


@app.function(gpu="H100!", timeout=600)
def gpu_name() -> None:
    sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")


@app.function(gpu="H100!", cpu=8, memory=98304, timeout=int(7.5 * 3600), volumes={VOL: volume})
def record_seed(record: str, seed: int) -> None:
    import os

    repo = Path("/root/rlvr")
    seeds = VOL / record / "seeds"
    if (seeds / str(seed)).exists():
        raise SystemExit(f"{seeds / str(seed)} already exists on the volume; move it to extra/ first")
    seeds.mkdir(parents=True, exist_ok=True)
    (repo / record / "seeds").symlink_to(seeds)
    env = {**os.environ, "CKPT_DIR": "/tmp/ckpts"}
    if "track_d" in record:  # untimed data preparation: the pinned data from the volume (`prep`), on local disk
        (repo / "data" / "track_d").symlink_to(VOL / "track_d")
        env["LOCAL_DATA_DIR"] = "/tmp"
    else:  # untimed, as on AWS: fetch the pinned base model before the clock starts
        sh("python", "-c", "from huggingface_hub import snapshot_download; from rlvr_speedrun import benchmark as b; "
           "snapshot_download(b.V3_TRACK_A_MODEL, revision=b.V3_TRACK_A_REVISION)")
    try:
        subprocess.run(["bash", "scripts/run_seed_v3.sh", record, str(seed)], check=True, cwd=repo, env=env)
    finally:
        volume.commit()
