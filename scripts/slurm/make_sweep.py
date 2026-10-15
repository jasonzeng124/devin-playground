"""Write a hyperparameter sweep as record-style directories and queue their seeds.

    python scripts/slurm/make_sweep.py calibration/sweep_001 --seeds 4 >> queue/pending.txt

The sweep's variants are SWEEPS[<basename of out>] below.

Each variant is record 005's config with the threshold disabled (full --max-steps curve, eval
every 50 steps, so one run gives the time to any candidate threshold) plus the overrides below;
its directory gets a config.json that run_seed.sh can use directly. Prints one "<dir> <seed>"
line per run, interleaved by seed so every variant gets early results.
Compare variants with scripts/summarize_sweep.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "records/track_a/005_prefix_kv_compile_h100/config.json"
KEEP = {"model", "few_shot", "eval_limit", "curriculum_steps", "group_size", "prompts_per_step", "temperature", "lr",
        "kl_coef", "eval_batch_size", "max_new_tokens", "micro_batch_size", "compile", "pad_to_multiple"}

# sweep_001 (2026-09-26): which recipe reliably reaches 0.15-0.20 (the user wants a higher v2
# threshold than 0.10, which record 005 cannot carry)? Does more signal per step (bigger groups
# / batches, more curriculum, informative-prompt oversampling) buy fewer and less variable steps
# to a threshold, and at what cost per step?
SWEEPS = {"sweep_001": {
    "base": {},
    "g32": {"group_size": 32},
    "g64": {"group_size": 64},
    "p16": {"prompts_per_step": 16},
    "g32_p16": {"group_size": 32, "prompts_per_step": 16},
    "g32_p16_lr1e-5": {"group_size": 32, "prompts_per_step": 16, "lr": 1e-5},
    "cur500": {"curriculum_steps": 500},
    "os2": {"oversample": 2},
}}
# sweep_002: the two baseline candidates for a 0.20 threshold (the first v2 record must get every
# seed there) get 8 more seeds each (--first-seed 4; seeds 0-3 are copied from sweep_001).
# Does "every seed reaches 0.20" hold at N = 12?
SWEEPS["sweep_002"] = {name: SWEEPS["sweep_001"][name] for name in ("os2", "g32_p16")}
# sweep_003: os2 misses 0.20 in about 1 run in 10 (3 of 32 across sweep_002 and record 006), and
# the misses are entropy collapse (policy entropy -> ~0, no informative groups left) at the
# record-005 sampling temperature of 0.8. A record needs every one of >= 10 seeds to cross, so at
# 90 % per seed it fails two times in three. Does a higher sampling temperature keep exploration
# alive without costing speed? 16 seeds each, 2,500 steps.
SWEEPS["sweep_003"] = {
    "os2_t0.9": {"oversample": 2, "temperature": 0.9},
    "os2_t1.0": {"oversample": 2, "temperature": 1.0},
}
# rollouts_os2: the record-006 recipe with sample_every=100, so there are rollouts to read
# (samples.jsonl) and checkpoints every 500 steps for scripts/analyze_solves.py. Seeds 1 (healthy), 4 and 16 (collapsed in record 006; GPU nondeterminism means
# they need not collapse again).
SWEEPS["rollouts_os2"] = {"os2": {"oversample": 2, "sample_every": 100, "save_every": 500}}

# sweep_004 (exploratory; the validator caps records at 64 tokens): does a larger generation
# budget let the model try more than one approach and solve puzzles that need * or /? At 64
# tokens a second attempt is cut off before <answer> and scores -0.1, below a wrong guess. The
# budget applies to training and eval alike. micro_batch_size is lowered for the longer sequences.
SWEEPS["sweep_004"] = {
    "os2_tok256": {"oversample": 2, "max_new_tokens": 256, "micro_batch_size": 32, "cache_len": 1024},
    "os2_tok512": {"oversample": 2, "max_new_tokens": 512, "micro_batch_size": 16, "cache_len": 1024},
}

# sweep_005 (benchmark v3 design pilot): the v2 definition (64 tokens, a "think briefly" few-shot
# prompt) cannot reward search, so try the TinyZero-style zero-shot think-then-answer prompt with a
# 1,024-token budget, scored as pass rate after a fixed training-time budget (evals excluded).
# Questions: how far does each base model get in 1-4 hours on one GPU, and does learning look
# like search (multiple attempts, solving puzzles that need * or /)? 2 seeds each, 4 hours of training.
_R1 = {"prompt_style": "r1", "max_new_tokens": 1024, "cache_len": 0, "temperature": 1.0, "oversample": 2,
       "curriculum_steps": 0, "micro_batch_size": 8, "eval_batch_size": 500, "eval_start_step": 0,
       "eval_every": 25, "time_budget_s": 14400, "sample_every": 25}
SWEEPS["sweep_005"] = {
    "r1_05b_lr2e-6": {**_R1, "lr": 2e-6},
    "r1_05b_lr5e-6": {**_R1, "lr": 5e-6},
    "r1_15b_lr1e-6": {**_R1, "model": "Qwen/Qwen2.5-1.5B", "lr": 1e-6},
    "r1_05b_instruct_lr2e-6": {**_R1, "model": "Qwen/Qwen2.5-0.5B-Instruct", "lr": 2e-6},
}

# sweep_006 (after pilot 1): is the 0.5B failure the model, or a recipe tuned for 64-token answers
# that does not carry over to 1,024-token reasoning? Pilot 1 used the
# speed-record settings (8 prompts x 16 samples per step, lr 2-5e-6, no KL), which are noisy and
# collapse entropy within an hour. Standard GRPO recipes use far larger batches and lower lr.
# All variants use the vLLM rollout backend and a 2 h training budget.
_V = {"rollout_backend": "vllm", "vllm_gpu_memory": 0.35, "compile": False, "cache_len": 0, "temperature": 1.0,
      "curriculum_steps": 0, "micro_batch_size": 8, "eval_batch_size": 2000, "eval_start_step": 0, "eval_every": 10,
      "time_budget_s": 7200, "sample_every": 10, "oversample": 1}
_R1V = {**_V, "prompt_style": "r1", "max_new_tokens": 1024}
SWEEPS["sweep_006"] = {
    "r1_p8g16_lr1e-6": {**_R1V, "prompts_per_step": 8, "group_size": 16, "lr": 1e-6},  # pilot-1 batch, lower lr
    "r1_p64g8_lr1e-6": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6},  # standard-size batch
    "r1_p64g8_lr1e-6_kl": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "kl_coef": 0.001},
    # at 1,024 tokens a truncated attempt scores -0.1, below a short wrong answer (0), which rewards
    # giving up on long reasoning; train with malformed = 0 (the eval still uses the verifier)
    "r1_p64g8_lr1e-6_mal0": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "malformed_reward": 0.0},
    "r1_p64g8_lr3e-6": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 3e-6},
    "v1_p64g8_lr1e-6": {**_V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "max_new_tokens": 64},  # v2 task, big batch
    "r1_p64g8_lr1e-6_15b": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "model": "Qwen/Qwen2.5-1.5B"},
    # the zero-shot r1 prompt gives 0.5B almost no signal (1 % of groups have any reward variance,
    # scripts/probe_models_vllm.py), so the same recipes trained with the worked-example "search"
# prompt but scored zero-shot (eval_prompt_style r1), the proposed fixed eval prompt:
    "search_p64g8_lr1e-6": {**_R1V, "prompt_style": "search", "eval_prompt_style": "r1", "prompts_per_step": 64, "group_size": 8, "lr": 1e-6},
    "search_p64g8_lr1e-6_mal0": {**_R1V, "prompt_style": "search", "eval_prompt_style": "r1", "prompts_per_step": 64, "group_size": 8, "lr": 1e-6,
                                 "malformed_reward": 0.0},
    "search_p64g8_lr1e-6_15b": {**_R1V, "prompt_style": "search", "eval_prompt_style": "r1", "prompts_per_step": 64, "group_size": 8, "lr": 1e-6,
                                "model": "Qwen/Qwen2.5-1.5B"},
}

# sweep_007: 0.5B barely learns zero-shot in 2 h (sweep_006: about 1 % throughout, no signal), and
# the vLLM probe (results/probe_models_r1.json) gives the share of 16-sample groups with any
# reward variance: 0.5B 1 %, 1.5B 3 %, Qwen3-1.7B 5 %, Qwen2.5-3B 13 %, Qwen3-4B 26 %. Try the
# larger bases with the same zero-shot recipe, plus 0.5B with the easy-puzzle curriculum
# (2- and 3-number puzzles first), which gave the v1/v2 recipes their early reward.
SWEEPS["sweep_007"] = {
    "q3_17b": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "model": "Qwen/Qwen3-1.7B-Base"},
    "q25_3b": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "model": "Qwen/Qwen2.5-3B", "micro_batch_size": 4},
    "q3_4b": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 1e-6, "model": "Qwen/Qwen3-4B-Base", "micro_batch_size": 4},
    "q25_05b_cur200_lr2e-6": {**_R1V, "prompts_per_step": 64, "group_size": 8, "lr": 2e-6, "curriculum_steps": 200},
}

# sweep_008: in sweep_007 only Qwen3-4B-Base learned steadily (0.009 -> 0.083 over 103 steps, still
# rising), but 88 % of its 70 s step was the gradient update, which padded every row to 1,024
# tokens. The trainer now packs length-sorted micro-batches (micro_batch_tokens) and projects to
# the vocabulary only for real tokens. These are longer runs (5 h of training) to see where 4B and
# 1.7B level off, and whether lr 2e-6 is faster than 1e-6.
_LONG = {**_R1V, "prompts_per_step": 64, "group_size": 8, "time_budget_s": 18000, "eval_every": 20}
SWEEPS["sweep_008"] = {
    "q3_4b_lr1e-6": {**_LONG, "model": "Qwen/Qwen3-4B-Base", "lr": 1e-6, "micro_batch_tokens": 8192, "grad_checkpoint": True},
    "q3_4b_lr2e-6": {**_LONG, "model": "Qwen/Qwen3-4B-Base", "lr": 2e-6, "micro_batch_tokens": 8192, "grad_checkpoint": True},
    "q3_17b_lr2e-6": {**_LONG, "model": "Qwen/Qwen3-1.7B-Base", "lr": 2e-6, "micro_batch_tokens": 16384, "grad_checkpoint": True},
}

# sweep_009: in sweep_008 every run's gains came from puzzles that need only + and -. Qwen3-1.7B
# (0.37 overall) solves 0.77 of those but 0.11 of those that need *, and every 4B run collapsed to
# one-line guesses. Can the 1.7B baseline learn * at all if it trains only on puzzles that need it?
# Evals now log pass_by_level.
_Q17 = {**_LONG, "model": "Qwen/Qwen3-1.7B-Base", "lr": 2e-6, "micro_batch_tokens": 16384, "grad_checkpoint": True}
SWEEPS["sweep_009"] = {
    "q3_17b_hard": {**_Q17, "train_levels": "+-*,+-*/"},
    "q3_17b_all": _Q17,
}

# sweep_010: the full-mix 1.7B baseline is reproducible (4 seeds at 0.29-0.37 after 5 h) but
# puzzles that need * stay at 0.08-0.11. Hard puzzles give few informative groups at 8 samples
# each; larger groups at the same 512 rollouts per step trade prompts for within-group coverage.
SWEEPS["sweep_010"] = {
    "q3_17b_p16g32": {**_Q17, "prompts_per_step": 16, "group_size": 32},
    "q3_17b_p8g64": {**_Q17, "prompts_per_step": 8, "group_size": 64},
}

# sweep_d01 (track D feasibility): GRPO from a ~110M model pretrained from scratch on FineWeb-Edu
# for 60 min (scripts/pretrain_scratch.py). Easy 2-3 number puzzles first (curriculum), partial
# credit for well-formed answers that use the right numbers and land near the target (shaping).
# The question is only whether the easiest puzzles start getting solved.
_D = {**_V, "model": "/home/zjason/orcd/pool/scratch-models/d110m_60min", "prompt_style": "r1", "max_new_tokens": 256,
      "prompts_per_step": 64, "group_size": 8, "shaping": 0.5, "curriculum_steps": 3000, "time_budget_s": 7200,
      "eval_set": "data/ladder_v3_dev.jsonl", "eval_limit": 600, "eval_every": 50, "micro_batch_tokens": 32768}
SWEEPS["sweep_d01"] = {
    "d110m_lr1e-5": {**_D, "lr": 1e-5},
    "d110m_lr3e-5": {**_D, "lr": 3e-5},
}

# sweep_d02: sweep_d01 hacked the shaping within ~100 steps by answering the bare target
# ("<answer> 21</answer>" earned closeness 1.0); closeness now counts only for answers that use
# exactly the puzzle's numbers. shaping 0 is the verifier-only control. Run with --max-steps 6000.
SWEEPS["sweep_d02"] = {
    "d110m_lr1e-5": {**_D, "lr": 1e-5},
    "d110m_lr3e-5": {**_D, "lr": 3e-5},
    "d110m_lr3e-5_noshape": {**_D, "lr": 3e-5, "shaping": 0.0},
}

# sweep_d03: sweep_d02 hacked the overlap term by answering one puzzle number (0.0625); the
# verifier-only control collapsed to "<answer>1</answer>" with zero reward throughout. Partial
# credit now pays mainly for expressions that use exactly the puzzle's numbers (0.5 + closeness).
SWEEPS["sweep_d03"] = {
    "d110m_lr3e-6": {**_D, "lr": 3e-6},
    "d110m_lr1e-5": {**_D, "lr": 1e-5},
}

# sweep_d04: sweep_d03's model never used the puzzle's numbers (0 of ~100k samples), so shaping had
# nothing to reward. Decoding after "<answer>" is now constrained to well-formed uses of the
# puzzle's numbers and operators (constrain_answer), which makes the plain verifier reward
# informative on easy puzzles. Verifier-only at two lrs, plus closeness shaping.
_DC = {**_D, "constrain_answer": True, "shaping": 0.0}
SWEEPS["sweep_d04"] = {
    "c_lr1e-5": {**_DC, "lr": 1e-5},
    "c_lr3e-6": {**_DC, "lr": 3e-6},
    "c_lr1e-5_shape": {**_DC, "lr": 1e-5, "shaping": 0.5},
}

# sweep_d05: sweep_d04 collapsed to "add every number" (ladder 0.000; in-batch solves only on easy
# puzzles). Step-limited SFT warm start on the generator's answers (no reasoning; scripts/sft_warmstart.py,
# 200 or 1000 steps x 64 examples), then the same constrained verifier-only RL.
_SFT = "/home/zjason/orcd/pool/scratch-models/d110m_sft{}"
SWEEPS["sweep_d05"] = {
    "sft200_lr3e-6": {**_DC, "model": _SFT.format(200), "lr": 3e-6},
    "sft1000_lr3e-6": {**_DC, "model": _SFT.format(1000), "lr": 3e-6},
}

# sweep_d06: the sum template is the RL optimum within reach under the generator mix (7 % of 4-number
# targets are the plain sum, and all-fail groups never push it down), so remove the payoff instead
# of warm-starting: train on the ladder mix (no plain-sum answers) with no easy curriculum (2-number
# puzzles are often sums). With the constraint, a random L1 answer hits ~1 in 8.
_DL = {**_DC, "train_mix": "ladder", "curriculum_steps": 0}
SWEEPS["sweep_d06"] = {
    "lad_lr3e-6": {**_DL, "lr": 3e-6},
    "lad_lr3e-6_drgrpo": {**_DL, "lr": 3e-6, "adv_norm": "none"},
    "lad_lr1e-5": {**_DL, "lr": 1e-5},
}

# sweep_d07: d06 is the first verifier-only from-scratch learning above chance (lr 1e-5: ladder 0.05,
# L1 0.25 vs ~0.02 / 0.09 for random valid answers); d05's SFT gains decayed under the sum-paying mix.
# 4 h runs: Dr. GRPO at higher lrs on the ladder mix, and SFT warm start + ladder mix.
_DL4 = {**_DL, "time_budget_s": 14400}
SWEEPS["sweep_d07"] = {
    "lad_lr1e-5_drgrpo": {**_DL4, "lr": 1e-5, "adv_norm": "none"},
    "lad_lr3e-5_drgrpo": {**_DL4, "lr": 3e-5, "adv_norm": "none"},
    "sft1000_lad_lr3e-6": {**_DL4, "model": _SFT.format(1000), "lr": 3e-6},
}

# sweep_d08: d07 lr 3e-5 Dr. GRPO reached ladder 0.147 at 4 h (best 0.155; L1 0.52, L2 0.12, L6 0.19),
# level with the best Qwen3-1.7B run. Higher lrs; d07's lr 3e-5 gets seeds 1-2 for spread.
SWEEPS["sweep_d08"] = {
    "lad_lr6e-5_drgrpo": {**_DL4, "lr": 6e-5, "adv_norm": "none"},
    "lad_lr1e-4_drgrpo": {**_DL4, "lr": 1e-4, "adv_norm": "none"},
}

# sweep_d09: lr 3e-5 Dr. GRPO is the best recipe (3 seeds 0.147 / 0.130 / 0.127 at 4 h); 6e-5 peaks
# early and decays, 1e-4 stops answering. Every run ends near-deterministic (entropy 0.05-0.09,
# 5-9 informative groups of 64), so: wider groups, hotter sampling, and an 8 h run to see the plateau.
_BEST = {**_DL4, "lr": 3e-5, "adv_norm": "none"}
SWEEPS["sweep_d09"] = {
    "p32g16": {**_BEST, "prompts_per_step": 32, "group_size": 16},
    "t1.2": {**_BEST, "temperature": 1.2},
    "8h": {**_BEST, "time_budget_s": 28800},
}

# sweep_d10 (evidence for the track D rules): how pretraining length trades against RL time
# (30 / 120 min pretraining vs d07's 60; 120 min repeats the 1.7B-token corpus ~1.6x), and whether
# training on generator solutions gives an edge at the best RL recipe (decides whether to ban them).
_PRE = "/home/zjason/orcd/pool/scratch-models/d110m_{}min"
SWEEPS["sweep_d10"] = {
    "pre30": {**_BEST, "model": _PRE.format(30)},
    "pre120": {**_BEST, "model": _PRE.format(120)},
    "sft1000": {**_BEST, "model": _SFT.format(1000)},
}

# sweep_d11 (is the ~0.17 plateau exploration or model?): dynamic sampling (oversample 2, keep the
# informative groups) and Clip-Cov on d110m_60min; Muon pretraining (two lrs) and a GPT-2
# (digit-merging) tokenizer, each 60 min, with the baseline RL recipe.
SWEEPS["sweep_d11"] = {
    "os2": {**_BEST, "oversample": 2},
    "covclip": {**_BEST, "cov_clip_frac": 0.002},
    "muon_lr1.5e-3": {**_BEST, "model": "/home/zjason/orcd/pool/scratch-models/d110m_muon1.5e-3_60min"},
    "muon_lr4e-3": {**_BEST, "model": "/home/zjason/orcd/pool/scratch-models/d110m_muon4e-3_60min"},
    "gpt2tok": {**_BEST, "model": "/home/zjason/orcd/pool/scratch-models/d110m_gpt2tok_60min"},
}

# sweep_d12 (does it ever try to reason?): d07 runs answered at once from step ~10 on (0.1 % of
# samples had any text first) because free text risks the 256-token cap and the malformed penalty.
# think_budget forces "<answer>" after N free tokens, so thinking is never punished, only unhelpful.
SWEEPS["sweep_d12"] = {
    "think160": {**_BEST, "think_budget": 160},
    "think64": {**_BEST, "think_budget": 64},
    "think160_pre120": {**_BEST, "think_budget": 160, "model": _PRE.format(120)},
}

# sweep_d13 (is there headroom in track D?): no reasoning emerged (d12) and exploration knobs did
# not move the ~0.17 plateau (d11), but Muon pretraining at 60 min matched AdamW at 120 min. Spend
# the 6 h budget on better pretraining: Muon 60 min + 5 h RL, Muon 120 min + 4 h RL, and a ~335M
# model (width 1024, 24 layers, Muon 120 min) + 4 h RL to see whether the plateau moves with size.
_M = "/home/zjason/orcd/pool/scratch-models/{}"
SWEEPS["sweep_d13"] = {
    "muon60_rl5h": {**_BEST, "model": _M.format("d110m_muon4e-3_60min"), "time_budget_s": 18000},
    "muon120": {**_BEST, "model": _M.format("d110m_muon4e-3_120min")},
    "d335m_muon120": {**_BEST, "model": _M.format("d335m_muon4e-3_120min")},
    "d335m_muon120_lr1e-5": {**_BEST, "model": _M.format("d335m_muon4e-3_120min"), "lr": 1e-5},
}

# sweep_d14 (is the plateau a signal shortage?): after ~20 min of RL only 3-5 of the 64 groups per
# step have mixed outcomes (the rest are all-fail, or all-pass), so ~93 % of each update carries no
# gradient. Oversample k: generate k x 64 groups and keep the 64 most informative (os2 in d11 gave
# ~10 informative groups and no gain; these aim for 20-60). Muon 60 min model, 5 h RL.
_M60 = {**_BEST, "model": _M.format("d110m_muon4e-3_60min"), "time_budget_s": 18000}
SWEEPS["sweep_d14"] = {f"os{k}": {**_M60, "oversample": k} for k in (4, 8, 16)}

# sweep_a01 (track A headroom, the obvious first records): the track A baseline trains on the
# generator mix with free decoding. Train on the ladder mix (what the score measures), and on the
# ladder mix with answer-constrained decoding (scored with --constrain), at the baseline lr and 2x.
_A = {**_Q17, "eval_set": "data/ladder_v3_dev.jsonl", "eval_limit": 600, "eval_every": 50, "eval_start_step": 50}
SWEEPS["sweep_a01"] = {
    "ladder": {**_A, "train_mix": "ladder", "curriculum_steps": 0},
    "ladder_constrained": {**_A, "train_mix": "ladder", "curriculum_steps": 0, "constrain_answer": True},
    "ladder_constrained_lr4e-6": {**_A, "train_mix": "ladder", "curriculum_steps": 0, "constrain_answer": True, "lr": 4e-6},
}


# sweep_d15 (does math text in pretraining lift the track D plateau?): FineWeb-Edu has little
# written arithmetic. 120 min AdamW pretraining (as sweep_d10 pre120) on FineWeb-Edu plus ~20 % of
# (a) its own documents with >= 3 written equations (scripts/math_mix.py filter; legal today) or
# (b) FineMath-4+ (needs a rule change), then the baseline RL recipe for 4 h. Control: d10 pre120.
SWEEPS["sweep_d15"] = {
    "mathfilter120": {**_BEST, "model": _M.format("d110m_mathfilter_120min")},
    "finemath120": {**_BEST, "model": _M.format("d110m_finemath_120min")},
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", help="queue only these variants (configs are still written for all)")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    base = {k: v for k, v in json.loads(Path(BASE).read_text()).items() if k in KEEP}
    base.update(target_solve_rate=1.01, max_steps=args.max_steps, eval_start_step=50, eval_every=50,
                cache_len=512, unranked=True, parent="005_prefix_kv_compile_h100")
    variants = SWEEPS[args.out.name]
    for name, overrides in variants.items():
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        config = {**base, **overrides, "description": f"{args.out.name}/{name}: record 005 recipe with {overrides or 'no changes'}"}
        (d / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        for name in args.only or variants:
            print(f"{args.out / name} {seed}")


if __name__ == "__main__":
    main()
