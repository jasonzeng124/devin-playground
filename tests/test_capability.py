import json

import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

import rlvr_speedrun.capability as capability
from rlvr_speedrun.capability import fineweb_loss, mmlu_lite_accuracy, run_probe


class CapabilityTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        if text in {" A", " B", " C", " D"}:
            return [10 + "ABCD".index(text[-1])]
        return [2 + (ord(char) % 30) for char in text]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=False, max_length=None):
        rows = [self.encode(text) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(map(len, rows))
        if self.padding_side == "left":
            input_ids = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        else:
            input_ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        mask = [[int(token != self.pad_token_id) for token in row] for row in input_ids]
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        )


def _tiny_model():
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=40,
            n_positions=256,
            n_ctx=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


def _rows(n_tests=8):
    shots = [
        {"kind": "shot", "question": "One plus one?", "choices": ["1", "2", "3", "4"], "answer": 1},
        {"kind": "shot", "question": "Two plus two?", "choices": ["2", "3", "4", "5"], "answer": 2},
    ]
    tests = [
        {"kind": "test", "question": f"Question {i}?", "choices": ["A", "B", "C", "D"], "answer": i % 4}
        for i in range(n_tests)
    ]
    return shots + tests


def test_fineweb_loss_is_finite_and_batch_invariant():
    torch.manual_seed(0)
    model = _tiny_model()
    tokenizer = CapabilityTokenizer()
    docs = [{"text": f"Document {i} about a small language model."} for i in range(4)]
    one = fineweb_loss(model, tokenizer, docs, torch.device("cpu"), batch_size=1)
    four = fineweb_loss(model, tokenizer, docs, torch.device("cpu"), batch_size=4)
    assert 0 < one < float("inf")
    assert abs(one - four) < 1e-3


def test_mmlu_lite_accuracy_is_batch_invariant():
    torch.manual_seed(1)
    model = _tiny_model()
    tokenizer = CapabilityTokenizer()
    rows = _rows()
    one = mmlu_lite_accuracy(model, tokenizer, rows, torch.device("cpu"), batch_size=1)
    four = mmlu_lite_accuracy(model, tokenizer, rows, torch.device("cpu"), batch_size=4)
    assert 0 <= one <= 1
    assert one == four


def test_run_probe_same_model_has_zero_deltas(monkeypatch):
    torch.manual_seed(2)
    model = _tiny_model()
    tokenizer = CapabilityTokenizer()
    monkeypatch.setattr(capability, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(capability, "load_causal_model", lambda _name, _device, _dtype: (model, torch.device("cpu")))
    result = run_probe(
        "tiny",
        "tiny",
        device="cpu",
        fineweb_docs=[{"text": "A short probe document."}] * 4,
        mmlu_rows=_rows(),
        batch_size=4,
    )
    assert abs(result["fineweb_loss_delta"]) < 1e-7
    assert abs(result["mmlu_lite_acc_delta"]) < 1e-7
    assert result["passes_guardrail"] is True


def test_validate_record_rejects_failing_probe(tmp_path):
    from scripts.validate_record import validate_record

    record = tmp_path / "track_a" / "001_probe"
    seed = record / "seeds" / "0"
    seed.mkdir(parents=True)
    (record / "README.md").write_text("probe\n")
    (record / "config.json").write_text("{}\n")
    (seed / "result.json").write_text(json.dumps({"pass_rate": 0.1, "time_to_threshold_s": 1}) + "\n")
    (seed / "train_log.jsonl").write_text('{"step": 1}\n')
    (seed / "probe.json").write_text(
        json.dumps(
            {
                "fineweb_loss_delta": 0.2,
                "mmlu_lite_acc_delta": -0.1,
                "passes_guardrail": False,
            }
        )
        + "\n"
    )
    valid, message = validate_record(record, allow_fewer_seeds=True)
    assert not valid
    assert "FAIL" in message
