"""Same-maths check for compile_update (LOG r123): compiled vs eager log-probs and gradients on an update-like batch.

    python scripts/research/check_compile_update.py --model <weights> --out check.json [--prompts 16] [--reps 10]

Builds an update micro-batch on the GPU: `prompts` training prompts (the ladder training stream, seed 0, which
excludes the dev and test sets; the r1 prompt in `_encode_prompts`'s shared-prefix layout), each repeated 8 times,
with completions sampled from the model (HF generate, temperature 1, up to 256 new tokens, cut after the first EOS).
Then it runs `_completion_logprobs` on the shared-prompt path, forward and backward with a per-row weight standing in
for the advantage, three ways:
  fp32:     the eager model without autocast (the reference);
  eager:    autocast bf16 over the fp32 weights, as the update runs;
  compiled: the same after `_compile_layers` (inductor).
It reports compiled-vs-eager and each-vs-fp32 differences (max and mean |delta log-prob| over completion tokens, and
the gradients' cosine and relative error), the first compiled call's wall time (compilation), the median wall time of
`reps` later calls each way, and whether a narrower, shorter second batch compiled new graphs.
Rule (LOG r123): compiled is no further from fp32 than eager is (mean |delta lp| at most 1.1x eager's, gradient
cosine to fp32 at least eager's minus 1e-4), and the compiled-vs-eager gradient cosine is above 0.999.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from rlvr_speedrun import grpo
from rlvr_speedrun.model_utils import load_causal_model, load_tokenizer

GROUP = 8


def _batch(model, tokenizer, prompts: int, max_new: int, seed: int):
    config = grpo.GRPOConfig(model="", prompt_style="r1", few_shot=0, pad_to_multiple=64, train_mix="ladder", seed=seed)
    stream, _ = grpo._train_streams(config)
    puzzles = [next(stream) for _ in range(prompts)]
    encoded, prefix_len = grpo._encode_prompts(tokenizer, puzzles, config, model.device)
    ids = encoded["input_ids"].repeat_interleave(GROUP, 0)
    mask = encoded["attention_mask"].repeat_interleave(GROUP, 0)
    prompt_width = ids.shape[1]
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                             max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    completion = out[:, prompt_width:]
    eos = (completion == tokenizer.eos_token_id).long()
    keep = (eos.cumsum(1) - eos) == 0  # up to and including the first EOS
    completion = completion.masked_fill(~keep, tokenizer.pad_token_id)
    ids = torch.cat([ids, completion], 1)
    mask = torch.cat([mask, keep.long()], 1)
    return ids, mask, prompt_width, prefix_len


def _run(model, ids, mask, prompt_width, prefix_len, weight, autocast: bool):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        lp, cm, _ = grpo._completion_logprobs(model, ids, mask, prompt_width, 1.0, prefix_len, share_prompt=True)
    loss = -(lp * cm * weight[:, None]).sum() / cm.sum()
    loss.backward()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    grad = torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])
    return lp.detach()[cm].float(), grad, wall


def _compare(a_lp, a_grad, b_lp, b_grad) -> dict:
    d = (a_lp - b_lp).abs()
    return {"max_abs_dlp": d.max().item(), "mean_abs_dlp": d.mean().item(),
            "grad_cos": torch.nn.functional.cosine_similarity(a_grad, b_grad, dim=0).item(),
            "grad_rel_err": ((a_grad - b_grad).norm() / b_grad.norm()).item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model, _ = load_causal_model(args.model, "cuda", "fp32")
    model.eval()
    ids, mask, prompt_width, prefix_len = _batch(model, tokenizer, args.prompts, args.max_new, args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    weight = (torch.randn(ids.shape[0], generator=gen)).to(ids.device)
    shape = {"rows": ids.shape[0], "width": ids.shape[1], "prompt_width": prompt_width, "prefix_len": prefix_len,
             "completion_tokens": int(mask[:, prompt_width:].sum())}
    print(json.dumps(shape), flush=True)

    lp32, g32, _ = _run(model, ids, mask, prompt_width, prefix_len, weight, autocast=False)
    lp_e, g_e, _ = _run(model, ids, mask, prompt_width, prefix_len, weight, autocast=True)
    eager_walls = [_run(model, ids, mask, prompt_width, prefix_len, weight, autocast=True)[2] for _ in range(args.reps)]

    from torch._dynamo.utils import counters

    grpo._compile_layers(model)
    lp_c, g_c, first_wall = _run(model, ids, mask, prompt_width, prefix_len, weight, autocast=True)
    compiled_walls = [_run(model, ids, mask, prompt_width, prefix_len, weight, autocast=True)[2] for _ in range(args.reps)]
    graphs_before = counters["stats"]["unique_graphs"]
    rows2, width2 = ids.shape[0] - 3 * GROUP, prompt_width + max(8, args.max_new // 2)  # a narrower, shorter batch
    second = _run(model, ids[:rows2, :width2].contiguous(), mask[:rows2, :width2].contiguous(), prompt_width, prefix_len, weight[:rows2],
                  autocast=True)[2]

    result = {
        "model": args.model, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, **shape,
        "eager_vs_fp32": _compare(lp_e, g_e, lp32, g32),
        "compiled_vs_fp32": _compare(lp_c, g_c, lp32, g32),
        "compiled_vs_eager": _compare(lp_c, g_c, lp_e, g_e),
        "first_compiled_call_s": first_wall,
        "eager_median_s": statistics.median(eager_walls), "compiled_median_s": statistics.median(compiled_walls),
        "eager_walls": eager_walls, "compiled_walls": compiled_walls,
        "unique_graphs": graphs_before, "new_graphs_second_batch": counters["stats"]["unique_graphs"] - graphs_before,
        "second_batch_s": second, "graph_breaks": sum(counters["graph_break"].values()),
    }
    e32, c32, ce = result["eager_vs_fp32"], result["compiled_vs_fp32"], result["compiled_vs_eager"]
    result["rule_met"] = (c32["mean_abs_dlp"] <= 1.1 * e32["mean_abs_dlp"] and c32["grad_cos"] >= e32["grad_cos"] - 1e-4
                          and ce["grad_cos"] > 0.999)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_walls")}, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
