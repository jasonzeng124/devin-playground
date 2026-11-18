import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from rlvr_speedrun.constrained import CLOSE, OPEN
from rlvr_speedrun.countdown import Puzzle
from rlvr_speedrun.fast_rollout import NativeRollout


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")


def _llama(tokenizer, kv_heads=2, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=kv_heads, max_position_embeddings=128, rope_theta=10000.0,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                         tie_word_embeddings=True)
    return LlamaForCausalLM(config).eval()


def _left_padded(rows, pad):
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r) :] = torch.tensor(r)
        mask[i, width - len(r) :] = 1
    return {"input_ids": ids, "attention_mask": mask}


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_greedy_tokens_are_the_hf_models_argmax(tokenizer, kv_heads):
    model = _llama(tokenizer, kv_heads)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    rows = [tokenizer.encode(t, add_special_tokens=False) for t in ["Use 3 and 7 to make 10.", "Hi", "a b c d e f g h i j k"]]
    encoded = _left_padded(rows, pad)
    engine = NativeRollout(model, tokenizer, 64, seed=0, dtype=torch.float32, cuda_graphs=False, batch_bucket=4, key_bucket=8)
    sequences, finished = engine.generate(encoded, 2, 0.0, 12, pad)
    width = encoded["input_ids"].shape[1]
    assert sequences.shape[0] == 6
    for r in range(6):
        prompt = rows[r // 2]
        completion = sequences[r, width : width + finished[r]].tolist()
        full = torch.tensor([prompt + completion])
        with torch.no_grad():
            logits = model(full).logits[0]
        expected = logits[len(prompt) - 1 : len(prompt) - 1 + len(completion)].argmax(-1).tolist()
        assert completion == expected
        assert sequences[r, width + finished[r] :].eq(pad).all()

    # after an update, sync() copies the new weights in
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    engine.sync(model)
    sequences, finished = engine.generate(encoded, 1, 0.0, 6, pad)
    completion = sequences[0, width : width + finished[0]].tolist()
    with torch.no_grad():
        logits = model(torch.tensor([rows[0] + completion])).logits[0]
    assert completion == logits[len(rows[0]) - 1 : len(rows[0]) - 1 + len(completion)].argmax(-1).tolist()


def test_constrained_samples_follow_the_grammar_and_stop_at_close(tokenizer):
    model = _llama(tokenizer, seed=1)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    puzzles = [Puzzle((3, 7), 10, ops="+-"), Puzzle((2, 5, 9), 1, ops="+-*/")]
    rows = [tokenizer.encode(f"Numbers {p.numbers}, target {p.target}.", add_special_tokens=False) for p in puzzles]
    encoded = _left_padded(rows, pad)
    engine = NativeRollout(model, tokenizer, 128, seed=3, dtype=torch.float32, constrain_answer=True, cuda_graphs=False,
                           batch_bucket=8, key_bucket=16)
    budget, n, max_tokens = 3, 4, 60
    sequences, finished = engine.generate(encoded, n, 1.0, max_tokens, pad, puzzles=puzzles, think_budget=budget)
    width = encoded["input_ids"].shape[1]
    closed = 0
    for r in range(len(puzzles) * n):
        puzzle = puzzles[r // n]
        tokens = sequences[r, width : width + finished[r]].tolist()
        allowed = engine.grammar.allowed_per_position(tokens, puzzle.numbers, puzzle.ops, budget)
        assert all(a is None or t in a for t, a in zip(tokens, allowed))
        text = "".join(engine.strings[t] for t in tokens)
        assert OPEN in text  # the think budget forces the answer open
        if CLOSE in text:
            closed += 1
            assert "".join(engine.strings[t] for t in tokens[:-1]).find(CLOSE) < 0  # stopped at the token that closed it
        else:
            assert len(tokens) == max_tokens or tokens[-1] in engine.eos
    assert closed > 0


def test_grpo_run_trains_with_the_native_backend(monkeypatch, tmp_path, tokenizer):
    import copy
    import json

    from rlvr_speedrun import grpo

    tokenizer = copy.deepcopy(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # as load_tokenizer does
    model = _llama(tokenizer, seed=2)
    monkeypatch.setattr(grpo, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(grpo, "load_causal_model", lambda _name, _device, _dtype: (model, torch.device("cpu")))
    config = grpo.GRPOConfig(model="tiny", rollout_backend="native", constrain_answer=True, think_budget=4, group_size=2, prompts_per_step=2,
                             max_new_tokens=12, eval_max_new_tokens=12, max_steps=2, eval_every=2, eval_limit=2, kl_coef=0.0,
                             no_save=True, out_dir=str(tmp_path / "native"))
    result = grpo.run(config)
    assert result["total_steps"] == 2
    assert len((tmp_path / "native" / "eval_log.jsonl").read_text().splitlines()) >= 1
    assert json.loads((tmp_path / "native" / "train_log.jsonl").read_text().splitlines()[-1])["step"] == 2
