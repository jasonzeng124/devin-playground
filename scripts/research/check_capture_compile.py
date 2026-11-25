"""Equivalence check and quick benchmark for compile_update inside update_capture (LOG 2026-11-25; INBOX c5e94e0).

    python scripts/research/check_capture_compile.py --model <weights> --out check.json [--prompts 16] [--reps 20]

The batch is check_compile_update.py's (`prompts` training prompts, 8 sampled completions each, the shared-prefix
layout), with a random per-row advantage. Each way runs grpo._update (shared prompts, host index, one micro-batch of
up to 65,536 tokens, SGD at lr 0 so the weights stay put) and reads the summed gradient:
  fp32:     no autocast, no capture (the reference);
  captured: autocast bf16 over the fp32 weights with update_capture, as r106's configs run;
  compiled: the same after `_compile_layers`, so the compiled layers run inside the captured graphs.
Rule, written before the run (the new standing rule: a change that leaves training the same needs an equivalence
check and a benchmark): compiled is no further from fp32 than captured is (gradient cosine to fp32 at least
captured's minus 1e-4, relative error at most 1.1x captured's), and every gradient is finite. The benchmark is
the median wall time of `reps` updates each way after two warm updates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_compile_update import GROUP, _batch  # noqa: E402

from rlvr_speedrun import grpo  # noqa: E402
from rlvr_speedrun.model_utils import load_causal_model, load_tokenizer  # noqa: E402


def _config(**kw) -> grpo.GRPOConfig:
    return grpo.GRPOConfig(model="", kl_coef=0.0, grad_clip=0.0, micro_batch_tokens=65536, share_group_prompt=True,
                           update_host_index=True, temperature=1.0, **kw)


def _step(model, ids, mask, adv, prompt_width, prefix_len, config, optimizer) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    grpo._update(model, None, ids, mask, adv, prompt_width, prefix_len, config, optimizer, ids.device, grpo.SINGLE)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None]), wall


def _compare(a, b) -> dict:
    return {"grad_cos": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            "grad_rel_err": ((a - b).norm() / b.norm()).item(), "finite": bool(torch.isfinite(a).all())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model, _ = load_causal_model(args.model, "cuda", "fp32")
    model.eval()
    ids, mask, prompt_width, prefix_len = _batch(model, tokenizer, args.prompts, args.max_new, args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    adv = torch.randn(ids.shape[0], generator=gen).to(ids.device)
    shape = {"rows": ids.shape[0], "width": ids.shape[1], "prompt_width": prompt_width, "prefix_len": prefix_len,
             "completion_tokens": int(mask[:, prompt_width:].sum()), "group": GROUP}
    print(json.dumps(shape), flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    run = (model, ids, mask, adv, prompt_width, prefix_len)

    g32, _ = _step(*run, _config(master_fp32=False), optimizer)
    captured = _config(master_fp32=True, update_capture=True)
    walls = {}
    for name, config in (("captured", captured), ("compiled", None)):
        if config is None:
            grpo._CAPTURES.pop(model, None)  # the eager captures read the uncompiled layers
            grpo._compile_layers(model)
            config = _config(master_fp32=True, update_capture=True, compile_update=True)
        t0 = time.perf_counter()
        grad, _ = _step(*run, config, optimizer)
        first = time.perf_counter() - t0
        _step(*run, config, optimizer)
        walls[name] = {"first_s": first, "walls": [_step(*run, config, optimizer)[1] for _ in range(args.reps)]}
        walls[name]["median_s"] = statistics.median(walls[name]["walls"])
        walls[name]["grad"] = grad
        walls[name]["stats"] = dict(grpo._CAPTURES[model].stats)

    c, k = walls["captured"], walls["compiled"]
    result = {
        "model": args.model, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, **shape,
        "captured_vs_fp32": _compare(c["grad"], g32), "compiled_vs_fp32": _compare(k["grad"], g32),
        "compiled_vs_captured": _compare(k["grad"], c["grad"]),
        **{f"{n}_{f}": walls[n][f] for n in walls for f in ("first_s", "median_s", "walls", "stats")},
    }
    result["speedup"] = c["median_s"] / k["median_s"]
    a, b = result["captured_vs_fp32"], result["compiled_vs_fp32"]
    result["rule_met"] = (b["grad_cos"] >= a["grad_cos"] - 1e-4 and b["grad_rel_err"] <= 1.1 * a["grad_rel_err"]
                          and a["finite"] and b["finite"])
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_walls")}, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
