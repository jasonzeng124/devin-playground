"""Capability-preservation probes for RLVR checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .model_utils import load_causal_model, load_tokenizer

MAX_FINEWEB_LOSS_INCREASE = 0.05
MAX_MMLU_LITE_DROP = 0.03
FINEWEB_PATH = Path("data/capability/fineweb_edu_val.jsonl")
MMLU_PATH = Path("data/capability/mmlu_lite.jsonl")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def fineweb_loss(model, tokenizer, docs, device, max_tokens: int = 512, batch_size: int = 16) -> float:
    """Return mean next-token NLL in nats over the supplied documents."""
    texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in docs]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in _batches(texts, batch_size):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                ).to(device)
                logits = model(**encoded).logits[:, :-1].float()
                labels = encoded["input_ids"][:, 1:]
                valid = encoded["attention_mask"][:, 1:].bool()
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)
                total_nll += nll.masked_select(valid).sum().item()
                total_tokens += int(valid.sum().item())
    finally:
        tokenizer.padding_side = old_padding_side
    return total_nll / total_tokens if total_tokens else float("nan")


def _mmlu_prompt(shots: Sequence[dict], row: dict, include_answer: bool = False) -> str:
    parts = ["The following are multiple choice questions (with answers).", ""]
    for item in [*shots, row]:
        parts.append(item["question"])
        parts.extend(f"{letter}. {choice}" for letter, choice in zip("ABCD", item["choices"]))
        if item is row and not include_answer:
            parts.append("Answer:")
        else:
            parts.append(f"Answer: {'ABCD'[item['answer']]}")
        parts.append("")
    if include_answer:
        return "\n".join(parts)
    # The final blank line is unnecessary and makes exact prompt inspection harder.
    return "\n".join(parts[:-1])


def mmlu_lite_accuracy(model, tokenizer, rows, device, batch_size: int = 16) -> float:
    """Score next-token multiple-choice predictions for the supplied rows."""
    shots = [row for row in rows if row.get("kind") == "shot"]
    tests = [row for row in rows if row.get("kind") == "test"]
    if not tests:
        return float("nan")
    answer_ids = []
    for letter in "ABCD":
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"tokenizer does not encode ' {letter}' as one token: {ids}")
        answer_ids.append(ids[0])
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    correct = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in _batches(tests, batch_size):
                prompts = [_mmlu_prompt(shots, row) for row in batch]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                logits = model(**encoded).logits[:, -1, :].float()
                predictions = logits[:, answer_ids].argmax(dim=-1).tolist()
                correct += sum(prediction == row["answer"] for prediction, row in zip(predictions, batch))
    finally:
        tokenizer.padding_side = old_padding_side
    return correct / len(tests)


def _load_target(target, device: str, dtype: str | None, tokenizer=None):
    if isinstance(target, (str, Path)):
        tokenizer = tokenizer or load_tokenizer(str(target))
        model, resolved = load_causal_model(str(target), device, dtype)
        return model, tokenizer, resolved
    if tokenizer is None:
        raise ValueError("a tokenizer is required when probing an in-memory model")
    resolved = torch.device(device) if isinstance(device, str) else device
    return target, tokenizer, resolved


def run_probe(
    model_name,
    base_name=None,
    device: str = "auto",
    dtype: str | None = None,
    fineweb_limit: int = 256,
    mmlu_limit: int = 500,
    batch_size: int = 16,
    fineweb_docs=None,
    mmlu_rows=None,
    tokenizer=None,
) -> dict:
    docs = fineweb_docs if fineweb_docs is not None else _load_jsonl(FINEWEB_PATH)[:fineweb_limit]
    rows = mmlu_rows if mmlu_rows is not None else _load_jsonl(MMLU_PATH)
    tests = [row for row in rows if row.get("kind") == "test"][:mmlu_limit]
    if mmlu_rows is None:
        rows = [row for row in rows if row.get("kind") == "shot"] + tests
    model, model_tokenizer, resolved = _load_target(model_name, device, dtype, tokenizer)
    result = {
        "model": str(model_name),
        "fineweb_loss": fineweb_loss(model, model_tokenizer, docs, resolved, batch_size=batch_size),
        "mmlu_lite_acc": mmlu_lite_accuracy(model, model_tokenizer, rows, resolved, batch_size=batch_size),
        "n_docs": len(docs),
        "n_questions": len(tests),
    }
    if base_name is not None:
        if base_name == model_name:
            base, base_tokenizer, base_device = model, model_tokenizer, resolved
        else:
            base, base_tokenizer, base_device = _load_target(base_name, device, dtype, tokenizer)
        result.update(
            {
                "base": str(base_name),
                "base_fineweb_loss": fineweb_loss(base, base_tokenizer, docs, base_device, batch_size=batch_size),
                "base_mmlu_lite_acc": mmlu_lite_accuracy(base, base_tokenizer, rows, base_device, batch_size=batch_size),
            }
        )
        result["fineweb_loss_delta"] = result["fineweb_loss"] - result["base_fineweb_loss"]
        result["mmlu_lite_acc_delta"] = result["mmlu_lite_acc"] - result["base_mmlu_lite_acc"]
        result["passes_guardrail"] = (
            result["fineweb_loss_delta"] <= MAX_FINEWEB_LOSS_INCREASE
            and result["mmlu_lite_acc_delta"] >= -MAX_MMLU_LITE_DROP
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype")
    parser.add_argument("--out")
    parser.add_argument("--fineweb-limit", type=int, default=256)
    parser.add_argument("--mmlu-limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = run_probe(
        args.model,
        args.base,
        device=args.device,
        dtype=args.dtype,
        fineweb_limit=args.fineweb_limit,
        mmlu_limit=args.mmlu_limit,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
