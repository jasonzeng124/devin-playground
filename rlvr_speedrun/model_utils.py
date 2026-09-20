"""Small shared helpers for loading and sampling causal language models."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


_CLOSE_TOKEN_IDS: dict[int, torch.Tensor] = {}


def _tokens_containing(tokenizer, text: str) -> torch.Tensor:
    """Vocabulary ids whose decoded string contains `text` (cached per tokenizer)."""
    key = id(tokenizer)
    if key not in _CLOSE_TOKEN_IDS:
        ids = [i for i in range(len(tokenizer)) if text in tokenizer.decode([i], skip_special_tokens=False)]
        _CLOSE_TOKEN_IDS[key] = torch.tensor(ids, dtype=torch.long)
    return _CLOSE_TOKEN_IDS[key]


class AnswerTagStoppingCriteria(StoppingCriteria):
    """Per-sequence stop as soon as a row emits `</answer>`.

    Only rows whose newest token can close a tag (contains '>') are decoded, so the
    check is a GPU mask plus a handful of decodes per step instead of one decode per
    row per step. `finished_at[i]` is the generated length of row i when it was
    stopped (-1 if it never emitted the tag), which lets callers mask the pad tokens
    the generator writes after a stopped row.
    """

    def __init__(self, tokenizer, tail_tokens: int = 16):
        self.tokenizer = tokenizer
        self.tail_tokens = tail_tokens
        self.start_length = None
        self.finished_at: torch.Tensor | None = None
        self._close_ids = _tokens_containing(tokenizer, ">")

    def __call__(self, input_ids, scores, **kwargs) -> torch.Tensor:
        if self.start_length is None or self.finished_at is None:
            self.start_length = input_ids.shape[1] - 1
            self.finished_at = torch.full((input_ids.shape[0],), -1, dtype=torch.long, device=input_ids.device)
            self._close_ids = self._close_ids.to(input_ids.device)
        gen_len = input_ids.shape[1] - self.start_length
        candidates = torch.isin(input_ids[:, -1], self._close_ids) & (self.finished_at < 0)
        if bool(candidates.any()):
            tail_start = max(self.start_length, input_ids.shape[1] - self.tail_tokens)
            for row in candidates.nonzero().flatten().tolist():
                if "</answer>" in self.tokenizer.decode(input_ids[row, tail_start:], skip_special_tokens=False):
                    self.finished_at[row] = gen_len
        return self.finished_at >= 0


def answer_stopping_criteria(tokenizer) -> StoppingCriteriaList:
    return StoppingCriteriaList([AnswerTagStoppingCriteria(tokenizer)])


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(device)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(model_name: str, device: str = "auto", dtype: str | None = None):
    resolved = resolve_device(device)
    if dtype is None:
        dtype = "bf16" if resolved.type == "cuda" else "fp32"
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(resolved)
    model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
    model.eval()
    return model, resolved


def generate_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: torch.device | None = None,
) -> list[str]:
    if not prompts:
        return []
    if device is None:
        device = next(model.parameters()).device
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "stopping_criteria": answer_stopping_criteria(tokenizer),
    }
    if do_sample:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**encoded, **kwargs)
    width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)
