"""Same-maths check for update_host_index (LOG r124): the update with host-built indices vs the default update.

    python scripts/research/check_update_host_index.py --config <dir>/config.json --model <weights> --out check.json \
        [--at 2,20,100] [--time 100:600:10] [--seed 0] [--set key=json ...]

It runs grpo.run with the config's settings (no saving, no evals, update_host_index as the config has it) and wraps
grpo._update, as scripts/research/profile_update.py does. At each update numbered in --at it first runs the update
three more times on the same inputs, each with an optimizer whose step does nothing: default, default again (the GPU's
run-to-run spread: its atomic adds may sum in another order), then update_host_index on. Then the real update runs, so
the run itself goes on unchanged. For each pair it reports the gradients' cosine and relative error, the largest
relative difference of the logged sums (loss, KL, entropy, tokens, mismatch, grad norm) and of the replayed rows' nll,
whether the gradients are bit-identical, and the three CUDA-synchronised wall times. The run stops after the last one.
With --time start:stop:step it also times the update in pairs on the same inputs at those updates (not the checked
ones): default and update_host_index each run once with the no-op step, the first of the pair alternating, so the run's
drift (each RL run's completions differ) cancels out of each pair.
Rule (LOG r124): at every checked update, on vs default is within the default's own repeat spread: gradient cosine
>= min(0.9999, the default-vs-default cosine - 1e-4), and the sums and the nll within max(1e-5, twice the
default-vs-default difference), relative. Bit-identical is expected up to the order of the GPU's atomic adds.
Timing rule (LOG r124): at least 30 pairs, the median of default minus host_index at least 0.01 s, and host_index
faster in enough pairs for a one-sided sign test p < 0.01.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import tempfile
import time
from dataclasses import fields

import torch

from rlvr_speedrun import grpo


class _Done(Exception):
    pass


class _NoStep:
    """The run's optimizer, except that `step` does nothing (the checked updates leave the weights alone)."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        pass


def _config(args) -> grpo.GRPOConfig:
    values = {f.name: None for f in fields(grpo.GRPOConfig)}
    values.update(model=args.model, seed=args.seed, device="cuda", out_dir=tempfile.mkdtemp(prefix="check_host_index_"),
                  no_save=True, eval_every=10**9, eval_start_step=10**9)
    for item in args.set:
        key, _, raw = item.partition("=")
        if key not in values:
            raise SystemExit(f"--set: {key} is not a GRPOConfig field")
        values[key] = json.loads(raw)
    return grpo._config_from_args(argparse.Namespace(config=args.config, **values))


def _floats(sums) -> list:
    out = []
    for s in sums:
        if s is None:
            out.append(None)
        else:
            out.extend(float(v) for v in torch.as_tensor(s).detach().float().flatten().tolist())
    return out


def _rel(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _compare(a: dict, b: dict) -> dict:
    cos = torch.nn.functional.cosine_similarity(a["grad"], b["grad"], dim=0).item()
    sums = max((_rel(x, y) for x, y in zip(a["sums"], b["sums"])), default=0.0) if len(a["sums"]) == len(b["sums"]) else float("inf")
    return {"grad_cos": cos, "grad_rel_err": ((a["grad"] - b["grad"]).norm() / b["grad"].norm().clamp_min(1e-30)).item(),
            "grad_identical": bool(torch.equal(a["grad"], b["grad"])), "sums_max_rel": sums, "nll_rel": _rel(a["nll"], b["nll"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="JSON results")
    parser.add_argument("--at", default="2,20,100", help="the updates (0-based) to check")
    parser.add_argument("--time", default="", help="start:stop:step, the updates (0-based) timed in pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    config = _config(args)
    at = sorted({int(x) for x in args.at.split(",")})
    timed = set(range(*(int(x) for x in args.time.split(":")))) - set(at) if args.time else set()
    last = max([*at, *timed])
    if config.max_steps <= last + 1:
        raise SystemExit(f"max_steps {config.max_steps} must exceed the last checked or timed update ({last})")

    original = grpo._update
    seen = [0]
    checks: list[dict] = []
    timing: list[dict] = []

    def once(a, on: bool, grads: bool = True) -> dict:
        a = list(a)
        a[7] = dataclasses.replace(a[7], update_host_index=on)  # config
        a[8] = _NoStep(a[8])  # optimizer
        replayed = a[13] if len(a) > 13 else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sums = original(*a)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if not grads:
            return {"wall_s": wall}
        model = a[0]
        grad = torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])
        return {"grad": grad, "sums": _floats(sums), "nll": None if replayed is None else float(replayed.nll), "wall_s": wall}

    def wrapped(*a, **k):
        if k:
            raise SystemExit("grpo._update was called with keyword arguments; the check expects them positional")
        i = seen[0]
        seen[0] += 1
        if i in at:
            off = once(a, False)
            off2 = once(a, False)
            on = once(a, True)
            row = {"update": i, "on_vs_default": _compare(on, off), "default_vs_default": _compare(off2, off),
                   "wall_s": {"default": off["wall_s"], "default_again": off2["wall_s"], "host_index": on["wall_s"]},
                   "sums_default": off["sums"], "nll_default": off["nll"]}
            del off, off2, on
            checks.append(row)
            print(json.dumps(row), flush=True)
        elif i in timed:
            first_on = len(timing) % 2 == 1
            walls = {}
            for on in (first_on, not first_on):
                walls[on] = once(a, on, grads=False)["wall_s"]
            timing.append({"update": i, "first": "host_index" if first_on else "default", "default_s": walls[False],
                           "host_index_s": walls[True]})
        out = original(*a)
        if i == last:
            raise _Done
        return out

    grpo._update = wrapped
    dist, device = grpo.init_from_env(config.device)
    config.device = device
    try:
        grpo.run(config, dist)
        raise SystemExit("the run ended before the last checked update")
    except _Done:
        pass
    finally:
        grpo._update = original
        grpo.destroy()

    def met(c: dict) -> bool:
        on, dd = c["on_vs_default"], c["default_vs_default"]
        return (on["grad_cos"] >= min(0.9999, dd["grad_cos"] - 1e-4) and on["sums_max_rel"] <= max(1e-5, 2 * dd["sums_max_rel"])
                and on["nll_rel"] <= max(1e-5, 2 * dd["nll_rel"]))

    rule = bool(checks) and all(met(c) for c in checks)
    gains = [t["default_s"] - t["host_index_s"] for t in timing]
    faster = sum(g > 0 for g in gains)
    n = len(gains)
    sign_p = sum(math.comb(n, k) for k in range(faster, n + 1)) / 2**n if n else 1.0
    timing_summary = {"pairs": n, "faster": faster, "sign_p": sign_p,
                      "median_gain_s": statistics.median(gains) if n else None,
                      "default_median_s": statistics.median(t["default_s"] for t in timing) if n else None,
                      "host_index_median_s": statistics.median(t["host_index_s"] for t in timing) if n else None}
    timing_summary["timing_rule_met"] = n >= 30 and timing_summary["median_gain_s"] >= 0.01 and sign_p < 0.01
    result = {"config": args.config, "model": args.model, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "at": at, "sets": args.set, "checks": checks, "rule_met": rule,
              "all_identical": all(c["on_vs_default"]["grad_identical"] for c in checks),
              "time": args.time, "timing_summary": timing_summary, "timing": timing}
    print(json.dumps({"rule_met": rule, "all_identical": result["all_identical"], **timing_summary}), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
