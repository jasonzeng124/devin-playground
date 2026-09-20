import json
import copy

import pytest
import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from rlvr_speedrun.countdown import Puzzle
import rlvr_speedrun.grpo as grpo
from rlvr_speedrun.grpo import GRPOConfig, grpo_step, run


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 40

    def __call__(self, prompts, return_tensors="pt", padding=True):
        rows = []
        for prompt in prompts:
            ids = [2 + (ord(char) % 30) for char in prompt[:12]]
            rows.append(ids)
        width = max(map(len, rows))
        input_ids = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        mask = [[int(token != self.pad_token_id) for token in row] for row in input_ids]
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        )

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "<answer>1 + 2</answer>" if index % 2 == 0 else "<answer>1 + 1</answer>"
            for index, _ in enumerate(sequences)
        ]

    def decode(self, sequence, skip_special_tokens=False):
        return "</answer>"


def test_single_grpo_step_has_finite_loss_and_updates_parameters():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=40,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    reference = copy.deepcopy(model)
    tokenizer = TinyTokenizer()
    run_config = GRPOConfig(group_size=2, prompts_per_step=1, max_new_tokens=4, temperature=1.0, kl_coef=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = grpo_step(
        model,
        reference,
        tokenizer,
        [Puzzle((1, 2), 3)],
        run_config,
        optimizer,
        torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_grpo_step_without_reference_model_has_zero_kl():
    torch.manual_seed(1)
    config = GPT2Config(
        vocab_size=40,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    tokenizer = TinyTokenizer()
    run_config = GRPOConfig(group_size=2, prompts_per_step=1, max_new_tokens=4, temperature=1.0, kl_coef=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = grpo_step(
        model,
        None,
        tokenizer,
        [Puzzle((1, 2), 3)],
        run_config,
        optimizer,
        torch.device("cpu"),
    )
    assert result["kl"] == 0.0


def _patch_tiny_training(monkeypatch, tmp_path, calls=None):
    config = GPT2Config(
        vocab_size=40,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    tokenizer = TinyTokenizer()
    if calls is None:
        calls = []

    def load_model(_name, _device, _dtype):
        calls.append(True)
        return model, torch.device("cpu")

    monkeypatch.setattr(grpo, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(grpo, "load_causal_model", load_model)
    return model, tokenizer, calls


def test_run_defers_periodic_eval_until_eval_start_step(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    config = GRPOConfig(
        model="tiny",
        group_size=2,
        prompts_per_step=1,
        max_new_tokens=4,
        max_steps=2,
        eval_every=1,
        eval_start_step=3,
        eval_limit=1,
        kl_coef=0.0,
        no_save=True,
        out_dir=str(tmp_path / "deferred"),
    )
    run(config)
    lines = (tmp_path / "deferred" / "eval_log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == 2


def test_run_with_zero_kl_does_not_load_reference_model(monkeypatch, tmp_path):
    _model, _tokenizer, calls = _patch_tiny_training(monkeypatch, tmp_path)
    config = GRPOConfig(
        model="tiny",
        group_size=2,
        prompts_per_step=1,
        max_new_tokens=4,
        max_steps=2,
        eval_every=1,
        eval_limit=1,
        kl_coef=0.0,
        no_save=True,
        out_dir=str(tmp_path / "zero_kl"),
    )
    result = run(config)
    assert result["total_steps"] == 2
    assert len(calls) == 1


def _tiny_model():
    return GPT2LMHeadModel(
        GPT2Config(vocab_size=40, n_positions=64, n_ctx=64, n_embd=16, n_layer=1, n_head=2, pad_token_id=0, eos_token_id=1)
    )


def test_select_groups_prefers_solved_then_informative_then_zero_variance():
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # zero variance
            [0.0, -0.1, 0.0],  # informative, no solve
            [1.0, 0.0, 0.0],  # informative, contains solve
            [-0.1, -0.1, -0.1],  # zero variance
            [1.0, 1.0, 0.0],  # informative, contains solve
        ]
    )
    kept = grpo.select_groups(rewards, 3).tolist()
    assert set(kept[:2]) == {2, 4}
    assert kept[2] == 1
    assert grpo.select_groups(rewards, 4).tolist()[3] in {0, 3}
    assert grpo.select_groups(rewards, 5).tolist() == [0, 1, 2, 3, 4]
    assert grpo.select_groups(rewards, 9).tolist() == [0, 1, 2, 3, 4]


def test_grpo_step_with_oversample_keeps_prompts_per_step_groups():
    torch.manual_seed(2)
    model = _tiny_model()
    tokenizer = TinyTokenizer()
    config = GRPOConfig(group_size=2, prompts_per_step=2, oversample=3, max_new_tokens=4, kl_coef=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    puzzles = [Puzzle((1, 2), 3)] * (config.prompts_per_step * config.oversample)
    result = grpo_step(model, None, tokenizer, puzzles, config, optimizer, torch.device("cpu"))
    assert result["kept_groups"] == 2
    assert 0 <= result["informative_groups"] <= 6
    assert len(result["rewards"]) == 4
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_chunked_generation_matches_unchunked_shapes():
    torch.manual_seed(3)
    model = _tiny_model()
    tokenizer = TinyTokenizer()
    puzzles = [Puzzle((1, 2), 3), Puzzle((3, 4), 7), Puzzle((5, 6), 11)]
    base = GRPOConfig(group_size=2, prompts_per_step=3, max_new_tokens=4)
    chunked = GRPOConfig(group_size=2, prompts_per_step=3, max_new_tokens=4, gen_batch_size=1)
    out_a = grpo._sample_batch(model, tokenizer, puzzles, base, torch.device("cpu"))
    out_b = grpo._sample_batch(model, tokenizer, puzzles, chunked, torch.device("cpu"))
    assert out_a[0].shape[0] == out_b[0].shape[0] == 6
    assert out_a[1].shape[0] == out_b[1].shape[0] == 6
    assert out_a[0].shape[1] == out_a[1].shape[1] and out_b[0].shape[1] == out_b[1].shape[1]
    assert len(out_b[4]) == 6 and out_a[5] == out_b[5]


def test_group_advantages_std_and_none():
    groups = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    centred = grpo.group_advantages(groups, "none")
    assert torch.allclose(centred[0], torch.tensor([0.75, -0.25, -0.25, -0.25]))
    assert torch.all(centred[1] == 0)
    z = grpo.group_advantages(groups, "std")
    assert torch.all(z[1] == 0)
    assert torch.allclose(z[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert z[0][0] > centred[0][0]
    with pytest.raises(ValueError):
        grpo.group_advantages(groups, "bogus")
