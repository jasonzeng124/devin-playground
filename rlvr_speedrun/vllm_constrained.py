"""vLLM hook for answer-constrained decoding (rlvr_speedrun/constrained.py); imports vLLM, so only the vLLM backend loads it."""

from __future__ import annotations

import time

from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .constrained import AnswerGrammar, RequestConstraint, mask_rows_cached, open_token_ids, token_strings

APPLY_TIME: float | None = None  # a float (set by grpo with time_update_parts): the batched processor's CPU seconds add up here


class AnswerLogitsProcessor(AdapterLogitsProcessor):
    """Applies `RequestConstraint` to requests whose `extra_args` carry {"countdown": {"numbers", "ops"}}."""

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer)
        self.grammar = AnswerGrammar(token_strings(tokenizer), open_token_ids(tokenizer))
        self.index_cache: dict = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        puzzle = (params.extra_args or {}).get("countdown")
        if puzzle is None:
            return None
        return RequestConstraint(self.grammar, puzzle["numbers"], puzzle["ops"], self.index_cache, puzzle.get("think_budget", 0))


class BatchedAnswerLogitsProcessor(AnswerLogitsProcessor):
    """The same mask as `AnswerLogitsProcessor`, applied to the whole batch at once (PLAN D7): the adapter's
    `apply` runs each request's processor on its own row, a few kernels per request per decoded token."""

    def apply(self, logits):
        global APPLY_TIME
        start = time.perf_counter() if APPLY_TIME is not None else None
        rows = [(row, lp.func.allowed_ids(*lp.args)) for row, lp in self.req_info.items()]
        out = mask_rows_cached(logits, rows, self.index_cache)
        if start is not None:
            APPLY_TIME += time.perf_counter() - start
        return out
