"""vLLM hook for answer-constrained decoding (rlvr_speedrun/constrained.py); imports vLLM, so only the vLLM backend loads it."""

from __future__ import annotations

from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .constrained import AnswerGrammar, RequestConstraint, open_token_ids, token_strings


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
