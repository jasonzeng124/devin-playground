"""Small shared helpers for loading and sampling causal language models."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class AnswerTagStoppingCriteria(StoppingCriteria):
    """Stop batched generation once every sequence has emitted an answer tag."""

    def __init__(self, tokenizer, tail_tokens: int = 64):
        self.tokenizer = tokenizer
        self.tail_tokens = tail_tokens
        self.start_length = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.start_length is None:
            self.start_length = input_ids.shape[1] - 1
        generated = input_ids[:, self.start_length :]
        tails = generated[:, -self.tail_tokens :]
        return all("</answer>" in self.tokenizer.decode(tail, skip_special_tokens=False) for tail in tails)


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
