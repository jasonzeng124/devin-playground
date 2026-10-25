"""Colocated vLLM sampler for GRPO rollouts and evals (`--rollout-backend vllm`).

vLLM runs in the training process on the same GPU, with a fixed slice of memory. After every
optimizer step the trainer copies the policy weights into it (`sync`). Rollouts come back in the
layout `grpo._generate_chunked` returns (prompt ids as encoded, then right-padded completion ids,
plus each row's completion length), so the loss code is unchanged. vLLM's sampling
probabilities differ from the training model's by kernel-level noise; the loss recomputes
log-probs with the training model, as TRL and verl do. An engine built with `logprobs=True` can
also return the sampler's log-prob of each sampled token, after temperature and the answer
constraint (`logprobs_mode="processed_logprobs"`), to measure or correct that difference.
"""

from __future__ import annotations

import os

import torch


class VLLMRollout:
    def __init__(self, model_name: str, max_len: int, gpu_memory_utilization: float, seed: int, dtype: str = "bfloat16",
                 constrain_answer: bool = False, logprobs: bool = False):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep the engine in-process so weights can be copied in
        from vllm import LLM

        processors = []
        if constrain_answer:
            from .vllm_constrained import AnswerLogitsProcessor

            processors.append(AnswerLogitsProcessor)
        extra = {"logprobs_mode": "processed_logprobs"} if logprobs else {}  # the distribution actually sampled from
        self.llm = LLM(model_name, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_len,
                       seed=seed, enable_prefix_caching=True, logits_processors=processors or None, **extra)
        self.logprobs = logprobs
        self._runner_model = None

    def _model(self):
        if self._runner_model is None:
            engine = self.llm.llm_engine
            candidates = [
                lambda: engine.model_executor.driver_worker.model_runner.model,  # V0
                lambda: engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,  # V1 in-process
                lambda: engine.model_executor.driver_worker.worker.model_runner.model,
            ]
            for get in candidates:
                try:
                    self._runner_model = get()
                    break
                except AttributeError:
                    continue
            else:
                raise RuntimeError("cannot find the vLLM model to load weights into; check the vLLM version")
        return self._runner_model

    @torch.no_grad()
    def sync(self, model) -> None:
        self._model().load_weights((name, param.data) for name, param in model.named_parameters())
        self.llm.reset_prefix_cache()

    def _params(self, *, n: int, temperature: float, max_tokens: int, seed: int | None, extra_args: dict | None = None,
                logprobs: bool = False):
        from vllm import SamplingParams

        # logprobs=0: only the sampled token's log-prob, kept as flat lists rather than a dict per token
        lp = {"logprobs": 0, "flat_logprobs": True} if logprobs else {}
        return SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1 if temperature > 0 else -1, max_tokens=max_tokens,
                              stop=["</answer>"], include_stop_str_in_output=True, seed=seed, extra_args=extra_args, **lp)

    @torch.no_grad()
    def generate(self, encoded: dict, n: int, temperature: float, max_tokens: int, pad_id: int, seed: int | None = None,
                 puzzles=None, think_budget: int = 0, return_logprobs: bool = False):
        """Sample `n` completions per prompt row; returns (sequences, finished_at) like grpo._generate_chunked.

        With `puzzles` (one per row), each request carries its numbers and operators for the answer constraint.
        With `return_logprobs` (engine built with `logprobs=True`), also returns the sampled tokens'
        log-probs, [rows, completion width] float32, right-padded with 0."""
        from vllm.inputs import TokensPrompt

        if return_logprobs and not self.logprobs:
            raise ValueError("return_logprobs needs an engine built with logprobs=True (processed log-probs)")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        prompts = [TokensPrompt(prompt_token_ids=row[m.bool()].tolist()) for row, m in zip(ids, mask)]
        if puzzles is None:
            params = self._params(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed, logprobs=return_logprobs)
        else:
            params = [self._params(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed, logprobs=return_logprobs,
                                   extra_args={"countdown": {"numbers": list(p.numbers), "ops": p.ops, "think_budget": think_budget}}) for p in puzzles]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        samples = [c for out in outputs for c in out.outputs]
        completions = [list(c.token_ids) for c in samples]
        width = max(1, max(len(c) for c in completions))
        comp = torch.full((len(completions), width), pad_id, dtype=torch.long)
        for i, c in enumerate(completions):
            comp[i, : len(c)] = torch.tensor(c, dtype=torch.long)
        comp = comp.to(ids.device)
        finished_at = torch.tensor([len(c) for c in completions], dtype=torch.long, device=ids.device)
        sequences = torch.cat([ids.repeat_interleave(n, dim=0), comp], dim=1)
        if not return_logprobs:
            return sequences, finished_at
        lp = torch.zeros((len(samples), width), dtype=torch.float32)
        for i, c in enumerate(samples):
            flat = c.logprobs  # vllm.logprobs.FlatLogprobs; a position's first entry is its sampled token
            if len(flat.start_indices) != len(c.token_ids):
                raise RuntimeError(f"vLLM returned {len(flat.start_indices)} log-probs for {len(c.token_ids)} tokens")
            lp[i, : len(c.token_ids)] = torch.tensor([flat.logprobs[j] for j in flat.start_indices], dtype=torch.float32)
        return sequences, finished_at, lp.to(ids.device)
