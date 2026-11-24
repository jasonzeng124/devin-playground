"""Run a Python script with this process's CUDA memory capped, then print its peak use (research tool, U25).

    python scripts/research/mem_cap_run.py <cap GiB> <script.py> [args ...]

It caps the caching allocator of the process's device (LOCAL_RANK under torchrun, else 0) at <cap GiB> with
torch.cuda.set_per_process_memory_fraction, so a larger card stands in for a smaller one (e.g. a B200 for an
80 GB H100), runs the script as __main__, and at exit prints one line `mem_cap_run {...}` with the rank, the cap
and the peak allocated and reserved GiB. An allocation over the cap fails as an out-of-memory error would.
"""
import atexit
import json
import os
import runpy
import sys

import torch

GIB = 1024 ** 3


def main() -> None:
    cap, script = float(sys.argv[1]), sys.argv[2]
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    total = torch.cuda.get_device_properties(rank).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * GIB / total), rank)

    def report() -> None:
        print("mem_cap_run " + json.dumps({
            "rank": rank, "cap_gib": cap, "total_gib": round(total / GIB, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(rank) / GIB, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(rank) / GIB, 2)}), flush=True)

    atexit.register(report)
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
