"""Profile GRPO's update step (timing only: nothing is saved, scored or kept).

    python scripts/research/profile_update.py --config <dir>/config.json --model <weights> --out profile.json \
        [--warm 20] [--steps 8] [--seed 0] [--parts] [--set key=json ...]

It runs grpo.run with the config's settings (no saving, no evals, `time_update_parts` off unless --parts) and wraps
grpo._update. The first `warm` updates are timed only (CUDA-synchronised wall time); each of the next `steps` runs
under its own torch.profiler session (CPU and CUDA), so the sampling between updates stays out of the profile. Then
the run stops (no final eval). It prints and writes:
  - wall time per update, unprofiled and profiled;
  - GPU busy fraction per update: the union of device activity (kernels, copies) over the update's wall time;
  - kernel launches, synchronisations and device-to-host copies per update;
  - the top ops by self CUDA time and by self CPU time, summed over the profiled updates.
It answers whether the update is compute-bound or overhead-bound before a speed change is tried. --set overrides
config fields (JSON values) to profile a variant, e.g. --set micro_batch_tokens=131072.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import fields

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from rlvr_speedrun import grpo

COUNTERS = {
    "kernel launches": ("LaunchKernel",),
    "synchronisations": ("Synchronize",),
    "device-to-host copies": ("_local_scalar_dense", "Memcpy DtoH"),
}


class _Done(Exception):
    pass


def _device_us(avg) -> float:
    value = getattr(avg, "self_device_time_total", None)
    return float(value if value is not None else getattr(avg, "self_cuda_time_total", 0.0))


def _busy_us(events) -> tuple[float, int]:
    """Union of device activity in microseconds, and the number of device events."""
    spans = sorted((e.time_range.start, e.time_range.end) for e in events if e.device_type == DeviceType.CUDA)
    busy = 0.0
    end = None
    for s, e in spans:
        if end is None or s > end:
            busy += e - s
            end = e
        elif e > end:
            busy += e - end
            end = e
    return busy, len(spans)


def _config(args) -> grpo.GRPOConfig:
    values = {f.name: None for f in fields(grpo.GRPOConfig)}
    values.update(model=args.model, seed=args.seed, device="cuda", out_dir=args.out_dir or tempfile.mkdtemp(prefix="profile_update_"),
                  no_save=True, eval_every=10**9, eval_start_step=10**9, time_update_parts=args.parts)
    for item in args.set:
        key, _, raw = item.partition("=")
        if key not in values:
            raise SystemExit(f"--set: {key} is not a GRPOConfig field")
        values[key] = json.loads(raw)
    config = grpo._config_from_args(argparse.Namespace(config=args.config, **values))
    if config.max_steps <= args.warm + args.steps:
        raise SystemExit(f"max_steps {config.max_steps} must exceed warm + steps ({args.warm + args.steps})")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="JSON results")
    parser.add_argument("--out-dir", default="", help="grpo's out_dir (train log only); default a temporary directory")
    parser.add_argument("--warm", type=int, default=20)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parts", action="store_true", help="keep time_update_parts on (adds a sync per part)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    config = _config(args)

    original = grpo._update
    walls: list[float] = []
    per_update: list[dict] = []
    ops: dict[str, dict] = {}

    def wrapped(*a, **k):
        i = len(walls)
        prof = None
        if i >= args.warm:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = original(*a, **k)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        walls.append(wall)
        if prof is not None:
            prof.stop()
            busy, n_device = _busy_us(prof.events())
            row = {"update": i, "wall_s": wall, "busy_s": busy / 1e6, "busy_frac": busy / 1e6 / wall, "device_events": n_device,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            for avg in prof.key_averages():
                op = ops.setdefault(avg.key, {"count": 0, "self_cpu_us": 0.0, "self_cuda_us": 0.0})
                op["count"] += avg.count
                op["self_cpu_us"] += avg.self_cpu_time_total
                op["self_cuda_us"] += _device_us(avg)
                for name, keys in COUNTERS.items():
                    if any(key in avg.key for key in keys):
                        row[name] = row.get(name, 0) + avg.count
            per_update.append(row)
            print(json.dumps(row), flush=True)
            if len(per_update) == args.steps:
                raise _Done
        return out

    grpo._update = wrapped
    dist, device = grpo.init_from_env(config.device)
    config.device = device
    try:
        grpo.run(config, dist)
        raise SystemExit("the run ended before the profiled updates")
    except _Done:
        pass
    finally:
        grpo._update = original
        grpo.destroy()

    n = len(per_update)
    unprofiled = walls[1:args.warm]  # the first update carries one-off allocation and autotuning
    summary = {
        "config": args.config, "model": args.model, "set": args.set, "parts": args.parts, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "warm": args.warm, "steps": n,
        "wall_s_unprofiled_median": statistics.median(unprofiled) if unprofiled else None,
        "wall_s_profiled_median": statistics.median(r["wall_s"] for r in per_update),
        "busy_frac_median": statistics.median(r["busy_frac"] for r in per_update),
        "busy_s_median": statistics.median(r["busy_s"] for r in per_update),
        "per_update": per_update,
    }
    for name in COUNTERS:
        summary[name + " per update"] = sum(r.get(name, 0) for r in per_update) / n
    total_cuda = sum(op["self_cuda_us"] for op in ops.values()) or 1.0
    total_cpu = sum(op["self_cpu_us"] for op in ops.values()) or 1.0
    for key, total, label in (("self_cuda_us", total_cuda, "top_cuda"), ("self_cpu_us", total_cpu, "top_cpu")):
        top = sorted(ops.items(), key=lambda kv: -kv[1][key])[:args.top]
        summary[label] = [{"op": name[:120], "per_update_ms": op[key] / n / 1e3, "share": op[key] / total, "calls_per_update": op["count"] / n}
                          for name, op in top]
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_update", "top_cuda", "top_cpu")}, indent=1))
    for label in ("top_cuda", "top_cpu"):
        print(f"--- {label}")
        for row in summary[label]:
            print(f"{row['per_update_ms']:9.3f} ms {row['share']:6.1%} {row['calls_per_update']:8.1f} calls  {row['op']}")
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
