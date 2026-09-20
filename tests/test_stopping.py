import torch
from transformers import AutoTokenizer

from rlvr_speedrun.model_utils import AnswerTagStoppingCriteria


def test_answer_tag_stopping_flags_rows_individually_and_records_length():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    prompt = tokenizer.encode("Puzzle:", add_special_tokens=False)
    done = tokenizer.encode(" 1 + 2</answer>", add_special_tokens=False)
    open_only = tokenizer.encode(" <answer>1 + 2", add_special_tokens=False)
    width = max(len(done), len(open_only))
    done = done + [tokenizer.eos_token_id] * (width - len(done))
    open_only = open_only + [tokenizer.eos_token_id] * (width - len(open_only))
    criteria = AnswerTagStoppingCriteria(tokenizer)
    scores = torch.zeros(2, 1)
    finished_steps = []
    for step in range(1, width + 1):
        ids = torch.tensor([prompt + done[:step], prompt + open_only[:step]])
        flags = criteria(ids, scores)
        assert flags.shape == (2,)
        assert not flags[1], "an opening tag must not stop generation"
        if flags[0] and not finished_steps:
            finished_steps.append(step)
    assert finished_steps, "row emitting </answer> must be flagged"
    assert criteria.finished_at.tolist()[0] == finished_steps[0]
    assert criteria.finished_at.tolist()[1] == -1
    assert "</answer>" in tokenizer.decode(done[: finished_steps[0]])
    assert "</answer>" not in tokenizer.decode(done[: finished_steps[0] - 1])
