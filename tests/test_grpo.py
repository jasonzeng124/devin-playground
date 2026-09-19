import copy

import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from rlvr_speedrun.countdown import Puzzle
from rlvr_speedrun.grpo import GRPOConfig, grpo_step


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

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
