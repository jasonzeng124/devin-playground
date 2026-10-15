# Benchmark v3 design notes (lab notebook, 2026-09-27 to 2026-10-14)

v2 (threshold 0.20, 64 tokens, few-shot "think briefly" prompt) was retired for v3. The analysis in
docs/plateau_calibration.md and `scripts/analyze_solves.py` showed it cannot reward search:

* 39.7 % of stopping-set puzzles can be solved with + and - only, so 0.20 needs no multiplication.
  The 0.07 shelf is the "add all four numbers" template (6.95 % of targets, 139 of 2,000, are the sum).
* At 64 tokens a second attempt is cut off before `<answer>` and scores -0.1, which is below a
  one-line wrong guess (0). RL therefore learns to guess in one line, and its "reasoning" is
  decoration with false arithmetic.
* The few-shot eval prompt primes one-line answers ("Think briefly", one-line worked examples).
* Time-to-threshold needs every seed to cross. About 1 os2 run in 10 collapses (entropy -> 0),
  so a 10-seed record fails about two times in three. Stopping at the first eval above the target
  also needs the 0.21 workaround for selection bias.
* RL at this scale barely moves the model: FineWeb loss changed by +0.002 nats and MMLU was
  unchanged. 5-number OOD went from 0.000 (base) to 0.030.

## Direction

| | v2 | v3 candidate |
|---|---|---|
| score | time to 0.20 | held-out pass rate after a fixed training-time budget on one H100 (evals off the clock) |
| prompt | 3-shot, "think briefly" | zero-shot think-then-answer (`prompt_style="r1"`, TinyZero-style) |
| generation budget | 64 tokens | about 1,024 (to be set from pilots) |
| eval set | 2,000 4-number puzzles | larger and harder mix (4-6 numbers), reported per level (sum / +- / needs * / needs /) |
| seeds | all N must cross | mean score over N seeds; stalled seeds just score low |

The fixed-budget format removes censoring, the stopping-set selection bias and the separate
confirmation set, since there is one eval at the end. It also never saturates, which matters if
RLVR done well reaches 80-90 %.

Open questions, answered by pilots:

1. Which base model? Qwen2.5-0.5B may not learn to search at all. Try 0.5B, 0.5B-Instruct and
   1.5B (`calibration/sweep_005`).
2. What budget (1 h? 2 h?) leaves the baseline far from the ceiling but clearly learning?
3. Which difficulty mix gives the baseline about 20-40 % with headroom?
4. Is the eval-time token cap binding? Also compare 256 and 512 tokens with the old prompt
   (`calibration/sweep_004`).

Dropped: sweep_003 (the temperature fix for v2's collapse) and the record 006 rerun. They only
matter under the v2 definition.

## Pilot 1 results (`calibration/sweep_005`, 2026-09-27; 4 h of training each on an H200, HF generate)

| variant | seeds finished | pass after 1 h | pass after 4 h | what happened |
|---|---|---|---|---|
| Qwen2.5-0.5B, lr 2e-6 | 2 | 0.006, 0.004 | 0.073, 0.077 | collapsed to one-line `a - b + c + d` answers, all 16 samples identical |
| Qwen2.5-0.5B, lr 5e-6 | 2 | 0.000, 0.070 | 0.000, 0.070 | one seed never learned the format; the other collapsed to the sum template |
| Qwen2.5-0.5B-Instruct, lr 2e-6 | 1 | 0.087 | 0.145 | "First, let's try using addition." then a sum; entropy 0.01 |
| Qwen2.5-1.5B, lr 1e-6 | 2 | 0.015, 0.018 | 0.017, 0.013 | reasons for real (tries sums, adjusts), but 53 s/step gave only 270 steps |

Base models with the r1 prompt at 1,024 tokens (greedy): 0.5B 0.005, 0.5B-Instruct 0.002,
1.5B 0.015.

Takeaways:
* Even with room to think, 0.5B RL finds a one-line template and its entropy collapses within
  about an hour. This matches TinyZero's report that 0.5B models do not learn to search on
  Countdown.
* 1.5B produces real trial-and-error reasoning, but HF `generate` at 1,024 tokens is far too
  slow (53 s/step). A usable baseline needs a real inference engine for rollouts. The trainer now
  has `rollout_backend="vllm"` (a colocated engine with weights synced every step).
* Next: probe the candidate base models with vLLM sampling (`scripts/probe_models_vllm.py`:
  pass@1, pass@16, the share of groups with disagreeing samples, length and truncation per
  model), then run pilots with the vLLM backend on the most promising 1-4B models.

## Prompt policy (decided with the user, 2026-09-27)

* **The eval prompt is fixed and zero-shot** (`r1`: think in `<think>`, answer in `<answer>`). If
  it could vary, records could win by prompting instead of learning, or smuggle in format
  scaffolding. A zero-shot eval asks whether the model learned the task.
* **The training prompt is free**, as are reward shaping, curriculum and so on
  (`prompt_style`, `eval_prompt_style`, `malformed_reward`). The zero-shot cold start is part of
  the challenge: with `r1`, only about 1 % of Qwen2.5-0.5B groups (16 samples) have any reward
  variance, and 3 % for 1.5B. Training on worked examples (`search`) and scoring zero-shot is
  one legitimate way around it.

## Pilot 2: model probe and 0.5B algorithm sweep (2026-09-28)

vLLM probe with the zero-shot r1 prompt, 500 stopping-set puzzles, 16 samples each at T=1, up to
1,024 tokens (`results/probe_models_r1.json`):

| model | greedy | pass@16 | groups with reward variance | truncated |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.002 | 0.006 | 0.01 | 0.02 |
| Qwen3-0.6B-Base | 0.000 | 0.006 | 0.01 | 0.08 |
| Qwen2.5-1.5B | 0.016 | 0.032 | 0.03 | 0.02 |
| Qwen3-1.7B-Base | 0.008 | 0.048 | 0.05 | 0.33 |
| Qwen2.5-1.5B-Instruct | 0.006 | 0.084 | 0.08 | 0.00 |
| Qwen2.5-3B | 0.038 | 0.134 | 0.13 | 0.02 |
| Qwen3-4B-Base | 0.008 | 0.262 | 0.26 | 0.21 |

In `calibration/sweep_006` (vLLM backend, 2 h of training), 0.5B stays at about 1 % zero-shot in
every variant: 8x16 or 64x8 batches, lr 1e-6, KL 0.001, and training on worked examples. Entropy
stays at 0.4-0.8, so nothing collapses; there is simply no reward to learn from. `sweep_007`
tries Qwen3-1.7B, Qwen2.5-3B and Qwen3-4B zero-shot, plus 0.5B with the easy-puzzle curriculum.

## Pilot 3 (2026-09-28): larger models, zero-shot, 2 h of training on an H200

| run | steps | s/step | pass after 1 h | pass after 2 h | notes |
|---|---|---|---|---|---|
| Qwen3-4B-Base, lr 1e-6 | 103 | 70 | 0.042, 0.035 | 0.041, 0.079 (max 0.083) | rises every eval; update is 61 of 70 s |
| Qwen2.5-3B | 170 | 42 | 0.041, 0.038 | 0.041, 0.040 | flat |
| Qwen3-1.7B-Base | 204 | 35 | 0.011, 0.013 | 0.031, 0.032 | about 45 % malformed (truncated) |
| 0.5B + easy curriculum, lr 2e-6 | 1,280 | 5.6 | 0.070 | 0.070 | collapsed to the sum template (entropy 0.02) |
| 0.5B, lr 3e-6 | 1,420 | 5 | 0.000 | 0.000 | collapsed |
| 0.5B, v1 prompt at 64 tokens, 64x8, T=1 | 1,390 | 5 | 0.000 | 0.000 | collapsed (max 0.022) |

Base-model probe with the worked-example `search` prompt, for reference only, since the training
prompt is fixed: the share of groups with reward variance is 0.5B 4 %, Qwen3-0.6B 15 %,
Qwen2.5-1.5B 23 %, Qwen3-1.7B 20 %, Qwen2.5-3B 34 % and Qwen3-4B 63 %.

The trainer was update-bound, since every row was padded to the longest rollout. It now has
`micro_batch_tokens` (length-sorted, trimmed micro-batches), vocabulary projection only for real
tokens, and optional `grad_checkpoint`. `sweep_008` runs Qwen3-4B (lr 1e-6 and 2e-6) and
Qwen3-1.7B (lr 2e-6) for 5 h of training each. The first launch went out of memory in backward
within a minute: token-packed micro-batches without checkpointing, next to vLLM's 35 % share.
Relaunched on 2026-09-29 with `grad_checkpoint` on.

## Pilot 4 (2026-09-29): sweep_008 results, 5 h of training on an H200

Final checkpoints probed with vLLM (`results/probe_sweep008_*.json` on Engaging; 1,000 eval
puzzles and 500 five-number OOD puzzles, 8 samples each at T=1, 1,024 tokens):

| model | greedy | pass@1 | pass@8 | greedy: sum / +- / needs * / needs / | greedy truncated | OOD5 greedy / pass@8 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 0.009 | 0.003 | 0.023 | 0.00 / 0.02 / 0.01 / 0.00 | 0.88 | 0.004 / 0.012 |
| Qwen3-4B-Base | 0.006 | 0.020 | 0.144 | 0.00 / 0.01 / 0.00 / 0.00 | 0.96 | 0.000 / 0.048 |
| 1.7B, lr 2e-6 | 0.374 | 0.334 | 0.612 | 0.89 / 0.77 / 0.11 / 0.01 | 0.29 | 0.100 / 0.354 |
| 4B, lr 1e-6 (2 seeds) | 0.241, 0.244 | 0.23, 0.25 | 0.36, 0.35 | 0.71 / 0.48 / 0.06 / 0.00 | 0.00 | 0.07 / 0.16 |
| 4B, lr 2e-6 | 0.304 | 0.304 | 0.342 | 0.80 / 0.61 / 0.09 / 0.00 | 0.00 | 0.088 / 0.114 |

* Every 4B run collapsed to a one-line answer with no reasoning (about 20 tokens, entropy 0.01-0.08).
* 1.7B learned visible trial-and-error search ("= 94 (close, but not right)") and was still
  improving at 5 h. It is the leading base-model candidate.
* Every run's gains are on puzzles that need only + and -. Puzzles that need * (52 % of the set)
  stay at or below 0.11. An eval set of the generator's full mix mostly rewards +/- search.
* Greedy decoding loops: 88-96 % of base-model greedy completions and 29 % for trained 1.7B hit the
  cap, against 1-3 % of samples. Mean sampled pass@1 is a more robust score than greedy.

`sweep_009` trains 1.7B only on puzzles that need * or / (`train_levels`) against the full mix,
with `pass_by_level` in every eval.

## Pilot 5 (2026-09-30): sweep_009, Qwen3-1.7B, 5 h of training, pass_by_level in every eval

| run | overall | sum | +- | needs * | needs / |
|---|---|---|---|---|---|
| full mix, seed 0 | 0.298 | 0.93 | 0.57 | 0.09 | 0.00 |
| full mix, seed 1 (at 289 min) | 0.340 | 0.89 | 0.68 | 0.11 | 0.00 |
| sweep_008 full mix, seed 1 (at 277 min) | 0.293 | 0.90 | 0.58 | 0.08 | 0.00 |
| only puzzles that need * or /, seed 0 | 0.084 | 0.02 | 0.09 | 0.10 | 0.01 |
| only puzzles that need * or /, seed 1 | 0.065 | 0.01 | 0.01 | 0.12 | 0.00 |

* The full-mix baseline is reproducible: 4 seeds at 0.29-0.37 after 5 h with near-identical curves.
* Training only on * and / puzzles gives the same 0.10-0.12 on them and loses +- (one-line templates).
* Late in training the greedy-eval malformed rate climbs to 0.2-0.5 (longer reasoning hitting the cap).
* The trainer now resumes after preemption (`resume_dir`, `resume_every_s`; on by default in
  `run_seed.sh` when CKPT_DIR is set, except for the record queue).

`sweep_010` keeps 512 rollouts per step but uses larger groups (16x32 and 8x64).

### sweep_010 (2026-09-30): group size at 512 rollouts per step, 5 h, 2 seeds each

| prompts x samples | overall | +- | needs * |
|---|---|---|---|
| 64 x 8 (sweep_009 baseline) | 0.298, 0.342 | 0.57, 0.68 | 0.09, 0.12 |
| 16 x 32 | 0.330, 0.327 | 0.63, 0.62 | 0.12, 0.11 |
| 8 x 64 | 0.310, 0.338 | 0.58, 0.66 | 0.11, 0.12 |

Group size makes no difference at 5 h (8 x 64 is slightly ahead in the first 90 minutes). The
~0.33 plateau and the ~0.11 on * puzzles are not a shortage of informative groups. Greedy-eval
malformed rates of 0.3-0.6 late in training (looping to the cap) may be hiding part of the
progress; the ladder probe scores sampled pass@1.

## Ladder v3 draft and track D pilot (2026-09-30)

* `rlvr_speedrun/ladder.py` + `data/ladder_v3_eval.jsonl` (draft; since renamed `data/ladder_v3_dev.jsonl`, byte-identical): 6 levels x 400.
  Puzzles carry their allowed operators (`Puzzle.ops`, stated in the prompt, enforced by `verify`;
  the default prompt text is unchanged). The trainer takes `eval_set`, `eval_samples` (sampled
  pass@1 via vLLM) and `train_mix="ladder"`, and logs `ladder_score` (the mean over levels).
* Track D pilot: `scripts/pretrain_scratch.py` (plain Llama-style ~110M, TinyLlama tokenizer with
  single-digit tokens, 2 FineWeb-Edu shards, 60 min on one GPU), then GRPO from it.

### Ladder baseline (probe, 200 puzzles per level, 8 samples at T=1, 1,024 tokens)

| model | ladder pass@1 | pass@8 | greedy by level L1 / L2 / L3 / L4 / L5 / L6 | greedy truncated |
|---|---|---|---|---|
| Qwen3-1.7B-Base | 0.004 | 0.027 | 0.015 / 0.005 / 0 / 0 / 0 / 0 | 0.87 |
| 1.7B full mix, 5 h (sweep_008 s0) | **0.149** | 0.319 | 0.75 / 0.16 / 0.03 / 0.02 / 0.01 / 0.10 | 0.39 |
| 1.7B full mix, 5 h (sweep_009 s0) | 0.118 | 0.280 | 0.63 / 0.11 / 0.01 / 0.02 / 0.02 / 0.08 | 0.44 |
| 1.7B only * and / puzzles, 5 h | 0.034 | 0.087 | 0.02 / 0.06 / 0.01 / 0.04 / 0 / 0.05 | 0.00 |
| 4B lr 2e-6 (one-line guesser) | 0.124 | 0.147 | 0.62 / 0.08 / 0 / 0.04 / 0 / 0.05 | 0.00 |

The best 5 h baseline scores about 0.15 on the ladder, almost all from L1. L3-L5 are near zero
for every model, so the ladder has a lot of headroom; the guessing 4B model no longer looks
competitive once levels weigh the same.

### Track D pilot: sweep_d01 hacked the shaping; sweep_d02

GRPO from `d110m_60min` (steps ~0.5-1 s). The answer format was learned within ~20-60 steps (malformed 0.99 -> <0.01), then both lrs converged on answering the bare target (`<answer> 21</answer>`): closeness 1.0 x 0.5 x shaping 0.5 = 0.25 reward, entropy -> 0.02-0.1, informative groups ~2/64, ladder 0.000 at steps 303 (1e-5) and 440 (3e-5). `partial_credit` now gives closeness only when the answer uses exactly the puzzle's numbers (overlap credit unchanged). sweep_d02 reruns both lrs plus a shaping-0 (verifier-only) control.

sweep_d02 (closeness gated on exact numbers): still 0 solves. Both shaped runs moved to answering a single puzzle number (`<answer> 6</answer>`, 0.25 x 1/4 overlap x 0.5 = 0.0625); the shaping-0 control never earned reward and collapsed to `<answer>1</answer>` within ~20 steps (the -0.1 malformed penalty is the only signal). Early samples do contain operators (~5% in the first 20 steps), so the model can write expressions; the reward just never preferred them. `partial_credit` now pays 0.5 + 0.5 x closeness only for expressions using exactly the puzzle's numbers, 0.25 x overlap for 2+ numbers, 0 for a bare number. sweep_d03: lr 3e-6 and 1e-5.

sweep_d03 (credit mainly for using exactly the puzzle's numbers): 0 solves, cancelled at steps ~320 (1e-5) and ~1,940 (3e-6). Of ~100k samples at 3e-6, 24 got any credit and **none** used the puzzle's numbers: the rewarded ones are generic templates (`(1 + 2) * 3`) that ignore the prompt. Both runs drifted to `<answer>1</answer>` like the verifier-only control. Conclusion: the 60-min FineWeb-only 110M model does not copy numbers from the prompt into the answer, so verifier-only RL from it has nothing to reinforce; reward shaping cannot fix that. Track D needs a model that attends to the prompt at step 0 (more pretraining / an entrant-chosen pretraining mix) or a structured action space (decoding constrained to the puzzle's numbers, operators and parentheses).

### Track D: constrained answers (sweep_d04)

`constrain_answer` (rlvr_speedrun/constrained.py): free text until `<answer>`, then only tokens that keep the text a prefix of `EXPR</answer>` with each puzzle number used exactly once, only the puzzle's operators, balanced parentheses, at most one space in a row and no more open parentheses than operators still to come. vLLM applies it as a per-request logits processor; the trainer renormalises its log-probs over the same allowed tokens (so the gradient is the constrained policy's). Division by zero is the one malformed answer the grammar cannot exclude. Smoke run (d110m_60min, lr 1e-5, verifier-only, easy curriculum): in-batch solve 0.02 -> 0.23 in 30 steps, malformed 0.82 -> 0.02 (the remaining malformed answers are free text that never reaches `<answer>`), ladder eval 0.000 at step 30. sweep_d04: verifier-only at lr 1e-5 and 3e-6, and closeness shaping 0.5 at 1e-5, 2 h each.

sweep_d04 at ~50-56 min: all three runs collapsed to "add every number" (`(24 + 1 + 10 + 9)`), the template the 1.7B model also found. In-batch solves (0.08-0.26) come from the easy 2-3 number curriculum puzzles; the ladder stays 0.000 (L1 excludes plain sums). lr 3e-6 keeps more entropy (0.5-0.9, ~20 informative groups vs ~1 at 1e-5) but samples the same template. Shaping only made the sum template pay more (0.37 for a wrong sum). Next, sweep_d05: a step-limited SFT warm start on generator answers (expressions only; 200 or 1000 steps x 64), then the same constrained verifier-only RL at lr 3e-6.

sweep_d05 / d06 (2 h each; random valid answers score ladder ~0.02: L1 0.09, L2 0.008, L6 0.014):
- d05 (SFT warm start, then constrained RL on the sum-paying generator mix): SFT alone gives ladder 0.07-0.09 at the first eval (L1 ~0.3, L2-L4 1-6 %); RL then decays it (sft1000 0.088 -> 0.077, sft200 0.070 -> 0.053) as answers drift toward + and -.
- d06 (no SFT, ladder mix, no curriculum): lr 1e-5 reaches 0.05 at 34 min (L1 0.25, L6 0.05) with varied operators: the first verifier-only from-scratch learning above chance. Dr. GRPO at 3e-6: 0.037 at 2 h, still rising, entropy ~1. std at 3e-6: 0.000 greedy all run (sampled solves ~1 %).
Next, sweep_d07 (4 h): Dr. GRPO at 1e-5 and 3e-5 on the ladder mix, and SFT warm start + ladder mix.

sweep_d07 (4 h, ladder mix, constrained, verifier-only; greedy ladder, 100 per level):
| run | ladder at 4 h (best) | L1 | L2 | L6 |
|---|---|---|---|---|
| Dr. GRPO lr 3e-5 | 0.147 (0.155) | 0.52 | 0.12 | 0.19 |
| SFT 1000 + lr 3e-6 | 0.103 (0.107) | 0.40 | 0.06 | 0.08 |
| Dr. GRPO lr 1e-5 | 0.088 | 0.32 | 0.04 | 0.14 |
d06 lr 1e-5 (std) ended at 0.053. The from-scratch 110M with verifier-only RL matches the best Qwen3-1.7B run (0.149 sampled / 0.176 greedy, unconstrained) on this ladder; L3 and L5 stay ~0. sweep_d08: lr 6e-5 and 1e-4, plus seeds 1-2 of lr 3e-5.

sweep_d08 + d07 seeds (4 h, greedy ladder at the end / best):
| run | ladder | best | notes |
|---|---|---|---|
| lr 3e-5 Dr. GRPO s0 / s1 / s2 | 0.147 / 0.130 / 0.127 | 0.155 / 0.140 / 0.132 | answers immediately (0 chars before `<answer>`), entropy 0.05-0.09 |
| lr 6e-5 | 0.077 | 0.108 (step 3,100) | peaks early, decays; learned a junk prefix ("COVID at <answer>") |
| lr 1e-4 | 0.000 | 0.080 | collapsed to free text that never reaches `<answer>` (256-token truncation) |
The policies are one-shot pattern matchers: no reasoning text, plus learned tricks such as multiplying by a 1 (`(1) * 13 + 10 + 3`). Full 2,400-puzzle sampled + greedy ladder probe (constrained d07 finals, d110m base, Qwen3-1.7B best constrained and unconstrained, Qwen base constrained) -> results/probe_ladder_full.json. sweep_d09 (lr 3e-5 Dr. GRPO): 32x16 groups, temperature 1.2, an 8 h run.

## Evidence for the v3 rule decisions (in progress, 2026-10-02)

| decision | evidence being gathered |
|---|---|
| eval sampler fixed vs part of the submission | d07 finals with and without the answer constraint; Qwen3-1.7B (base and best) with the constraint (results/probe_ladder_full.json) |
| generator solutions allowed? | sweep_d10 sft1000: SFT warm start + the best RL recipe vs d07's 0.13-0.15 without it |
| track D budget / pretrain-RL split | sweep_d10 pre30 / pre120 vs d07 (60 min); sweep_d09 8h run for the RL plateau |
| token cap (A) | Qwen best at 2,048 vs 1,024 tokens (results/probe_ladder_2048.json) |
| seeds needed | full-ladder sampled pass@1 of 8 track A finals (results/probe_ladder_A.json) and 3 track D seeds |
| keep anti-hacking checks (A)? | capability probe of the best Qwen final (results/capability_q3_17b_s0.json) |
| exploration in the baseline recipe | sweep_d09: 32x16 groups, temperature 1.2 |

Results (2026-10-02; greedy ladder, 100 per level, one seed each unless noted; d07 baseline = lr 3e-5 Dr. GRPO on d110m_60min, 3 seeds 0.147 / 0.130 / 0.127 at 4 h):
| run | ladder at end (best) | notes |
|---|---|---|
| sft1000 + baseline recipe (d10) | **0.240** (0.248) | starts at 0.125; L1 0.78, L3 0.11. Generator solutions are a large edge. |
| 8 h RL (d09) | 0.173 (0.190) at 8 h | 0.147 at ~4 h, 0.172 at ~5 h, then flat |
| 120 min pretrain, val 2.81 (d10) | 0.175 (0.180) | reaches 0.18 by ~2 h of RL |
| temperature 1.2 (d09) | 0.168 (0.172) | still rising at 4 h |
| 30 min pretrain, val 3.21 (d10) | 0.140 (0.163) | within the 60-min seed range |
| 32 x 16 groups (d09) | 0.078 (0.085) | worse: fewer puzzles per step |
Seeds 1-2 of temperature 1.2 and 120 min pretraining queued; probes moved to mit_preemptable.

Full-ladder probes (2,400 puzzles, sampled pass@1 at T=1 with k=16, level-weighted; results/probe_ladder_{A,2048,full}.json on Engaging):
- Track A seed spread: 8 Qwen3-1.7B finals of the 5 h recipe family score 0.123-0.155 (mean 0.138, sd 0.011). One-sided alpha 0.05 / power 0.8 detects a gain of ~0.022 at N=3, 0.017 at N=5, 0.012 at N=10. Track D (3 d07 seeds, 100-per-level greedy) has the same sd, 0.011. (Superseded: the 5-seed H100 records measure 0.0098 for A and 0.0163 for D, and the Welch test needs ~0.017 / ~0.028 for 80 % power; RULES_v3.)
- Token cap: the best Qwen at 2,048 tokens scores 0.157 vs 0.155 at 1,024 (truncation 1 % vs 2 %); the cap does not bind once training ends. (Greedy truncation is 39 %: greedy loops, sampling does not.)
- Constraint on Qwen: the best Qwen with constrained answers scores 0.168 vs 0.155 (+0.013, L1 0.71 vs 0.69, L2 0.12 vs 0.10). The constraint helps a trained model only a little; its effect on the d110m models and on the bases is pending (probe_D rerun after a fix: Qwen samples padded vocab ids past the tokenizer).
- Capability guardrail: best Qwen final FineWeb loss +0.003 nats, MMLU-lite -0.006 vs base; far inside v2's 0.05 / -0.03 bounds.
- Extra seeds (4 h): temperature 1.2 = 0.168 / 0.105 / 0.130 (mean 0.134, same as T=1); 120 min pretraining s1/s2 still running.

sweep_d11 (queued 2026-10-02; is the ~0.17 plateau exploration or the model?): 2 seeds each of the baseline RL recipe with dynamic sampling (oversample 2), Clip-Cov (`cov_clip_frac` 0.002, new in grpo.py), Muon pretraining (`--optimizer muon`, lr 1.5e-3 / 4e-3 with match_rms_adamw, 60 min), and a GPT-2 tokenizer (digits merged: " 128" is one token; fineweb-edu-gpt2.bin, 60 min AdamW).

Did track D ever try to reason? No: in d07 seed 0, >99.9 % of ~620k samples open `<answer>` at once from step 10; ~60 had any text first (FineWeb-style rambling, not arithmetic) and 2 contained an equation (both wrong). Cause: with the 256-token cap, free text risks truncation and the -0.1 malformed penalty, while an immediate (constrained, always well-formed) answer scores >= 0, so "answer at once" is rewarded whether or not thinking helps. `think_budget` (grpo.py, constrained.py) forces "\n<answer>" after N free tokens (forced tokens carry no gradient), so thinking is never punished. sweep_d12: budgets 160 and 64 on d110m_60min, 160 on d110m_120min; queued ahead of d11.

## Rules draft (2026-10-02)

docs/RULES_v3.md. Frozen: `data/ladder_v3_dev.jsonl` (the pilots' draft set, renamed; seed 30,000,000) and `data/ladder_v3_test.jsonl` (seed 31,000,000, disjoint by (numbers, target, ops); 3 test puzzles share numbers and target with a dev puzzle under different operators). Both are checksummed in tests/test_frozen_data.py and excluded from every training stream (benchmark.HELDOUT_PATHS). Scoring: `python -m rlvr_speedrun.ladder_confirm` (16 samples per puzzle, 1,024 tokens, level-weighted pass@1, entrant's sampler recorded). Pins: Qwen3-1.7B-Base ea980cb, fineweb-edu 87f0914, vLLM 0.30.0 / torch 2.13.0+cu130 / transformers 5.17.0. Test-ladder baselines (d07 constrained and free, Qwen sweep_008 s0/s1) -> results/confirm_v3/ on Engaging.

## Results for the rule decisions (2026-10-03)

Test ladder (`ladder_confirm`, 2,400 puzzles x 16 samples, T=1, results/confirm_v3/ on Engaging):
| model | sampler | ladder | L1 | L2 | L3 | L4 | L5 | L6 | truncated |
|---|---|---|---|---|---|---|---|---|---|
| d07 s0 | constrained | 0.140 | 0.47 | 0.13 | 0.00 | 0.05 | 0.01 | 0.19 | 0 |
| d07 s1 | constrained | 0.121 | 0.40 | 0.09 | 0.05 | 0.03 | 0.00 | 0.15 | 0 |
| d07 s2 | constrained | 0.125 | 0.42 | 0.09 | 0.07 | 0.01 | 0.00 | 0.16 | 0 |
| d07 s0 | free | **0.000** | 0 | 0 | 0 | 0 | 0 | 0 | 0.27 |
| Qwen sweep_008 s0 | free | 0.153 | 0.68 | 0.10 | 0.02 | 0.02 | 0.01 | 0.09 | 0.02 |
| Qwen sweep_008 s0 | constrained | 0.164 | 0.69 | 0.12 | 0.02 | 0.03 | 0.01 | 0.12 | 0.02 |
| Qwen sweep_008 s1 | free | 0.123 | 0.57 | 0.07 | 0.01 | 0.02 | 0.00 | 0.07 | 0.03 |

- Dev is a faithful proxy: the same d07 finals score 0.141 / 0.122 / 0.122 on the dev ladder (sampled) and 0.140 / 0.121 / 0.125 on test; their in-loop greedy 100-per-level evals were 0.147 / 0.130 / 0.127.
- Eval sampler: a model trained with the constraint scores 0 without it (free decoding never writes a valid answer; 27 % truncated). The constraint is part of the policy, so it must be allowed at eval. On Qwen it adds +0.011 (test) / +0.013 (dev): a small, uniform edge. Bases with the constraint: Qwen3-1.7B-Base 0.011, d110m_60min 0.003 (dev), so the constraint alone solves nothing.
- Pretraining length (in-loop dev, greedy, 4 h of RL): 120 min = 0.175 / 0.167 / 0.137 (mean 0.160) vs 60 min = 0.147 / 0.130 / 0.127 (mean 0.135); 30 min = 0.140 (n=1). At a 6 h total, 120 min + 4 h RL (0.160, n=3) is about level with 60 min + 5 h RL (d09: 0.172 at 5 h, n=1). Caveat: every d110m pretrain ran on an H200; the baseline record must redo it on an H100, where 60 min buys fewer tokens (~1.38B tokens / 60 min on H200).
- Think budget (sweep_d12, 4 h, n=1): budget 160 = 0.072, budget 64 = 0.070, budget 160 on the 120-min model = 0.115 (vs 0.13-0.16 without). With the budget, the 60-min model writes ~90-320 chars of FineWeb-style text before answering, and that text does not help; the 120-min model learns to skip it (0 chars). No reasoning emerges at this scale and budget; it is left to later records.
- Exploration (sweep_d11, 4 h, 2 seeds): dynamic sampling (oversample 2) 0.113 / 0.138, no gain. Clip-Cov 0.002 blew up: entropy rose to ~8.5, the completions turned into >1,000 chars of junk that never reach `<answer>`, ladder 0 at the end (best 0.083). The ~0.17 plateau is not fixed by these exploration knobs.
- Muon pretraining (60 min, then the baseline 4 h of RL, 2 seeds; in-loop dev greedy): lr 4e-3 = 0.157 / 0.170 (mean 0.164; val loss 2.83, vs 2.96 for AdamW at 60 min and 2.81 at 120 min); lr 1.5e-3 = 0.133 / 0.145. Muon at 60 min matches AdamW at 120 min, so the pretraining stage is a real lever for entrants.
- GPT-2 tokenizer (digits merged, 60 min AdamW, 2 seeds): 0.065 / 0.068, half the TinyLlama tokenizer's 0.135 (which splits numbers into single digits). Confounds: it trained on a 1.09B-token prefix (~1.1 epochs), has 124M parameters (larger embedding) and ran 18 % fewer tokens per second. The tokenizer matters a lot, and choosing it is a legitimate part of an entry, so the rule stays "any tokenizer trained on the allowed data, or the pinned TinyLlama one".

What the track D policies learned (correct answers in the last ~60k training samples, templates with numbers replaced by N, `templates.py` on Engaging): a few fixed operator shapes, with the numbers' placement learned per puzzle. For d07 s0, 4 shapes cover 75 % of correct 4-number answers (`(N)+N+N-N` 27 %, `(N)+N-(N-N)` 21 %, `(N)*N+N+N` 15 %, `(N)+N-N-N` 13 %). With 5-6 numbers it is "one product, then add or subtract the rest", e.g. 80 % of pre120's correct 6-number answers are `(N)+N*N+N+N+N`. The pre120 and Muon models have the same shapes, with somewhat more variety (29-35 shapes vs 25). The `*1` trick is rare (0.3-3 % of correct answers). So the score measures learned sign and placement choice under a template, not search. L3 (needs /) and L5 (6 numbers, targets 100-999) stay near 0.

sweep_d13 (queued 2026-10-04, does better pretraining move the plateau?): Muon 60 min + 5 h RL (3 seeds); Muon 120 min (110M) + 4 h RL (2 seeds); a 335M model (width 1024, 24 layers, Muon 120 min) + 4 h RL at lr 3e-5 (2 seeds) and 1e-5 (1 seed).

sweep_d13 results (2026-10-05; in-loop dev ladder, greedy):
| run | seeds | notes |
|---|---|---|
| Muon 60 min + 5 h RL | 0.177 / 0.137 | s0 died on a stale NFS handle; s0, s3, s4 requeued |
| Muon 120 min (110M, val 2.72) + 4 h RL | 0.122 / 0.122 (best 0.150 / 0.127) | worse than Muon 60 (0.164) despite lower val loss |
| 335M, Muon 120 min (val 2.79, 0.97B tokens) + 4 h RL, lr 3e-5 | 0.113 / 0.162 | the bigger model gets fewer tokens and slower RL in the same time |
| same, lr 1e-5 | 0.087 | |
Lower pretraining loss does not reliably buy RL score beyond ~2.8 nats, and in a fixed 6 h budget a 3x larger model is not better. The plateau stays ~0.15-0.17.
Muon 60 min + 5 h RL, all seeds: 0.142 / 0.177 / 0.137 / 0.165, plus s3 at 0.172 when preempted at 202 min (it resumes). Mean ~0.155, level with AdamW 120 min + 4 h (0.160). (Final, all 5 seeds at 5 h: 0.142 / 0.177 / 0.137 / 0.172 / 0.165, mean 0.158.)

sweep_d14 (queued 2026-10-05): oversample 4 / 8 / 16 on the Muon 60 min model, 5 h RL, 1 seed each. In d13 runs only 3-5 of 64 groups per step are informative after the first ~20 min.

sweep_d14 interim results (2026-10-06): all three runs were preempted at 4.3-4.6 h of 5 h and are requeued to resume. Dev score at the last eval:

| run | hours | last dev | best dev |
|---|---|---|---|
| os16 | 4.3 | 0.140 | 0.140 |
| os8 | 4.5 | 0.173 | 0.175 |
| os4 | 1.2 (restarted) | 0.118 | 0.123 |
| muon60_rl5h s3 (oversample 1, reference) | 4.6 | 0.177 | 0.183 |

Oversampling does not lift the plateau. os8 matches the oversample-1 reference, and os16 is lower because each step costs more. Informative groups being scarce is not the bottleneck: when uninformative groups are filtered out, the policy still settles on the same few answer templates.

sweep_d15 (queued 2026-10-07): does math text in pretraining lift the track D plateau? FineWeb-Edu has little written arithmetic. Two arms, each 120 min of AdamW pretraining (as d10 pre120) and then 4 h of baseline RL:
- **mathfilter120:** FineWeb-Edu plus ~20 % of its own documents with >= 3 written equations, found by a hand-written regex (`scripts/math_mix.py`). This is legal under the current rules.
- **finemath120:** FineWeb-Edu plus ~20 % FineMath-4+ (3 shards at revision e92b25a6). This would need a rule change.

Control: d10 pre120 (dev 0.175 / 0.167 / 0.137).

sweep_d15 data (2026-10-07): **FineWeb-Edu has almost no written arithmetic.** The two pinned shards hold 1,455,000 documents and 7,899 regex-matched equations (0.005 per document). Only 810 documents (0.06 %; 0.14 % of characters; 3M tokens) have >= 3. Examples: `55+21=76`, `35 * 15 = 525`.

The legal arm (mathfilter120) reaches only a 1.2 % share even with 8 repeats, so it was cancelled: reaching 20 % would mean ~100 repeats of 3M tokens. The FineMath arm (finemath120: 0.53B tokens, 23.5 % of the mix) stays queued.

Later (2026-10-12) finemath120 ran on Engaging, but its RL config (calibration/sweep_d15/finemath120/config.json) inherited `few_shot: 3` and `max_steps: 2000` from an older template, so it is not comparable with the control and no conclusion is drawn from it.

## Outcomes (2026-10-14)

Records, all trained on Engaging's H100 node2906 (test ladder, 5 seeds each):
| record | test (mean ± sd) | notes |
|---|---|---|
| A 000 baseline | 0.1295 ± 0.0098 | free sampling |
| A 001 ladder mix + constraint + lr 4e-6 | **0.2481 ± 0.0460** | new record, Welch p = 0.0019; seed 2 collapsed into ~40-token guesses by minute 60 and still scored 0.186 |
| D 000 baseline | 0.1166 ± 0.0163 | scored on L40S nodes |
| D 001 Muon 56 min + 290 min RL | 0.1162 ± 0.0156 | not a record (p = 0.52) |

- Track A pilots for 001 (calibration/sweep_a01, sweep_a02; the record's README has the numbers): at lr 2e-6 the ladder mix alone matched the baseline and adding the constraint halved the score; only the combination at lr 4e-6 worked, and lr 6e-6 / 8e-6 collapsed into short answers within the first hour. The ablation that would separate the three changes (ladder mix at lr 4e-6 without the constraint) hasn't been run.
- Sampler diagnostic (calibration/diag_sampler, dev ladder): the constraint at scoring adds +0.011-0.014 to A 000 and ~+0.016 to A 001's healthy seeds, but +0.074 to the collapsed seed. On a common sampler, A 001 still beats A 000 by +0.09 to +0.11.
- Track D on the H100 came out below the H200 pilots (in-loop dev 0.119 vs 0.159, p = 0.025) at about the same RL steps per minute; each pilot arm had shared one pretrained model.
- The trainer runs AdamW on bf16 weights with no fp32 master copy: after 5 h only 2.2 % (A 000) and 4.4 % (A 001) of the weights differ from the base model.
- The corrections made to the launched docs are listed in docs/REVIEW_v3.md.
