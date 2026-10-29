import copy
import json
import math
import sys
import types

import pytest
import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

import rlvr_speedrun.grpo as grpo
from rlvr_speedrun.countdown import Puzzle
from rlvr_speedrun.countdown import split_prompt as _split_prompt
from rlvr_speedrun.grpo import GRPOConfig, grpo_step, run


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 40

    def encode(self, text, add_special_tokens=True):
        return [2 + (ord(char) % 30) for char in text[:12]]

    def __call__(self, prompts, return_tensors="pt", padding=True, pad_to_multiple_of=None):
        rows = [self.encode(prompt) for prompt in prompts]
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


def test_cross_chunk_padding_is_excluded_even_when_pad_equals_eos(monkeypatch):
    class PadIsEosTokenizer(TinyTokenizer):
        pad_token_id = 1
        eos_token_id = 1

    tokenizer = PadIsEosTokenizer()
    puzzles = [Puzzle((1, 2), 3), Puzzle((3, 4), 7)]
    config = GRPOConfig(group_size=1, prompts_per_step=2, max_new_tokens=4, gen_batch_size=1)
    encoded, _ = grpo._encode_prompts(tokenizer, puzzles, config, torch.device("cpu"))
    prompt_width = encoded["input_ids"].shape[1]
    # chunk 0 generated 4 tokens (no EOS); chunk 1 generated 2 and was padded with pad==eos to width 4
    completions = torch.tensor([[5, 6, 7, 8], [5, 6, 1, 1]])
    generated = torch.cat([encoded["input_ids"], completions], dim=1)
    finished_at = torch.tensor([4, 2])
    monkeypatch.setattr(grpo, "_generate_chunked", lambda *args, **kwargs: (generated, finished_at))
    _, attention, *_ = grpo._sample_batch(model=None, tokenizer=tokenizer, puzzles=puzzles, config=config, device=torch.device("cpu"))
    assert attention[:, prompt_width:].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_malformed_reward_solved_applies_only_in_groups_with_a_correct_sample(monkeypatch):
    tokenizer = TinyTokenizer()
    puzzles = [Puzzle((1, 2), 3), Puzzle((3, 4), 7)]
    config = GRPOConfig(group_size=3, prompts_per_step=2, max_new_tokens=4, malformed_reward=0.0, malformed_reward_solved=-0.5)
    encoded, _ = grpo._encode_prompts(tokenizer, puzzles, config, torch.device("cpu"))
    generated = torch.cat([encoded["input_ids"].repeat_interleave(3, dim=0), torch.full((6, 4), 5)], dim=1)
    monkeypatch.setattr(grpo, "_generate_chunked", lambda *args, **kwargs: (generated, torch.full((6,), 4)))
    # group 0: correct, wrong, malformed; group 1: wrong, malformed, malformed (no correct sample)
    verdicts = iter([(1.0, True, False), (0.0, False, False), (-0.1, False, True), (0.0, False, False), (-0.1, False, True), (-0.1, False, True)])
    monkeypatch.setattr(grpo, "verify", lambda text, puzzle: types.SimpleNamespace(**dict(zip(("reward", "correct", "malformed"), next(verdicts)))))
    rewards = grpo._sample_batch(None, tokenizer, puzzles, config, torch.device("cpu"))[4]
    assert rewards.tolist() == [1.0, 0.0, -0.5, 0.0, 0.0, 0.0]


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


def test_pos_scale_damps_only_positive_advantages():
    groups = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 1.0]])
    for norm in ("none", "std"):
        plain = grpo.group_advantages(groups, norm)
        damped = grpo.group_advantages(groups, norm, pos_scale=0.1)
        assert torch.allclose(damped, torch.where(plain > 0, 0.1 * plain, plain))
        assert torch.equal(grpo.group_advantages(groups, norm, pos_scale=1.0), plain)
    assert torch.allclose(grpo.group_advantages(groups, "none", pos_scale=0.1)[0], torch.tensor([0.075, -0.25, -0.25, -0.25]))


class CharTokenizer(TinyTokenizer):
    """Exact per-character tokenizer, so prefix/suffix tokenization concatenates cleanly."""

    def encode(self, text, add_special_tokens=True):
        return [2 + (ord(char) % 30) for char in text]


def test_encode_prompts_aligns_shared_prefix_and_pads_in_the_middle():
    tokenizer = CharTokenizer()
    puzzles = [Puzzle((1, 2), 3), Puzzle((10, 20, 30), 60)]
    config = GRPOConfig(few_shot=0, pad_to_multiple=8)
    encoded, prefix_len = grpo._encode_prompts(tokenizer, puzzles, config, torch.device("cpu"))
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    assert prefix_len == len(tokenizer.encode("Using the numbers"))
    assert ids.shape[1] % 8 == 0
    assert torch.equal(ids[0, :prefix_len], ids[1, :prefix_len])
    assert bool(mask[:, :prefix_len].all()) and bool(mask[:, -1].all())
    assert not bool(mask[0].all())  # the shorter prompt has internal padding
    row = ids[0][mask[0].bool()].tolist()
    assert row == tokenizer.encode("".join(_split_prompt(puzzles[0], 0)))


def test_prefix_cache_logprobs_match_full_forward():
    torch.manual_seed(3)
    model = _tiny_model().eval()  # GPT2Config defaults to dropout>0, which breaks exact comparison
    prefix_len, width, prompt_width = 6, 20, 14
    ids = torch.randint(2, 40, (3, width))
    ids[:, :prefix_len] = ids[0, :prefix_len]
    mask = torch.ones_like(ids)
    mask[1, prefix_len : prefix_len + 3] = 0
    mask[2, prefix_len : prefix_len + 1] = 0
    ids[mask == 0] = 0

    def score(prefix):
        model.zero_grad(set_to_none=True)
        lp, cm, ent = grpo._completion_logprobs(model, ids, mask, prompt_width, 0.7, prefix)
        (lp * cm).sum().backward()
        grad = model.transformer.wte.weight.grad.clone()
        return lp.detach(), cm, ent, grad

    lp_full, cm_full, ent_full, grad_full = score(0)
    lp_pref, cm_pref, ent_pref, grad_pref = score(prefix_len)
    assert torch.equal(cm_full, cm_pref)
    assert torch.allclose(lp_full[cm_full], lp_pref[cm_pref], atol=1e-5)
    assert torch.allclose(ent_full[cm_full], ent_pref[cm_pref], atol=1e-5)
    assert torch.allclose(grad_full, grad_pref, atol=1e-5)


def test_grpo_step_with_prefix_cache_path():
    torch.manual_seed(4)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=40, n_positions=512, n_ctx=512, n_embd=16, n_layer=1, n_head=2, pad_token_id=0, eos_token_id=1)
    )
    tokenizer = CharTokenizer()
    config = GRPOConfig(group_size=2, prompts_per_step=2, few_shot=0, max_new_tokens=4, kl_coef=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = grpo_step(model, None, tokenizer, [Puzzle((1, 2), 3), Puzzle((10, 20, 30), 60)], config, optimizer, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_length_sorted_token_micro_batches_give_the_same_gradient():
    """micro_batch_tokens regroups and trims rows; the loss (sum over tokens / global count) must not change."""
    torch.manual_seed(0)
    prompt_width, rows = 4, 7
    generated = torch.randint(2, 40, (rows, prompt_width + 9))
    attention = torch.ones_like(generated)
    for row, length in enumerate([9, 1, 5, 3, 9, 2, 6]):  # ragged, right-padded completions
        attention[row, prompt_width + length :] = 0
    advantages = torch.randn(rows)
    grads, base = [], _tiny_model().eval()  # no dropout, as for loaded models
    for extra in ({"micro_batch_size": 3}, {"micro_batch_tokens": 30}):
        model = copy.deepcopy(base)
        config = GRPOConfig(model="tiny", kl_coef=0.0, grad_clip=0.0, **extra)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        grpo._update(model, None, generated, attention, advantages, prompt_width, 0, config, optimizer, torch.device("cpu"), grpo.SINGLE)
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("extra", [{"micro_batch_size": 3}, {"micro_batch_tokens": 30}])
def test_skipping_zero_advantage_rows_gives_the_same_gradient(extra):
    torch.manual_seed(1)
    prompt_width, rows = 4, 8
    generated = torch.randint(2, 40, (rows, prompt_width + 9))
    attention = torch.ones_like(generated)
    for row, length in enumerate([9, 1, 5, 3, 9, 2, 6, 4]):
        attention[row, prompt_width + length :] = 0
    base = _tiny_model().eval()
    for advantages in (torch.tensor([0.0, 0.0, 1.2, -1.2, 0.0, 0.0, 0.7, -0.7]), torch.zeros(rows)):
        grads = []
        for skip in (False, True):
            model = copy.deepcopy(base)
            config = GRPOConfig(model="tiny", kl_coef=0.0, grad_clip=0.0, skip_zero_advantage=skip, **extra)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            grpo._update(model, None, generated, attention, advantages, prompt_width, 0, config, optimizer, torch.device("cpu"), grpo.SINGLE)
            grads.append([p.grad.clone() for p in model.parameters()])
        for a, b in zip(*grads):
            assert torch.allclose(a, b, atol=1e-6)


def test_skip_zero_advantage_refuses_a_kl_term(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    config = GRPOConfig(model="tiny", kl_coef=0.02, skip_zero_advantage=True, out_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="skip_zero_advantage"):
        run(config)


def test_malformed_reward_until_switches_to_the_verifier_reward(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    seen = []
    step = grpo.grpo_step

    def recording_step(model, ref_model, tokenizer, puzzles, config, *args):
        seen.append(config.malformed_reward)
        return step(model, ref_model, tokenizer, puzzles, config, *args)

    monkeypatch.setattr(grpo, "grpo_step", recording_step)
    common = dict(model="tiny", group_size=2, prompts_per_step=1, max_new_tokens=4, max_steps=3, eval_every=10,
                  eval_limit=1, kl_coef=0.0, no_save=True)
    run(GRPOConfig(**common, malformed_reward=0.0, malformed_reward_until=3, out_dir=str(tmp_path / "until")))
    assert seen == [0.0, 0.0, None]
    with pytest.raises(ValueError, match="malformed_reward_until"):
        run(GRPOConfig(**common, malformed_reward_until=3, out_dir=str(tmp_path / "bad")))


def test_run_resumes_from_saved_state_after_an_interruption(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    config = GRPOConfig(
        model="tiny",
        group_size=2,
        prompts_per_step=1,
        max_new_tokens=4,
        max_steps=4,
        eval_every=1,
        eval_limit=1,
        kl_coef=0.0,
        no_save=True,
        sample_every=1,
        resume_dir=str(tmp_path / "resume"),
        resume_every_s=1e-9,
        out_dir=str(tmp_path / "run"),
    )
    real_step = grpo.grpo_step
    calls = []

    def preempted_step(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise KeyboardInterrupt("preempted")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(grpo, "grpo_step", preempted_step)
    with pytest.raises(KeyboardInterrupt):
        run(config)
    resumed_calls = []
    monkeypatch.setattr(grpo, "grpo_step", lambda *a, **k: resumed_calls.append(True) or real_step(*a, **k))
    result = run(config)
    assert len(resumed_calls) == 2  # steps 3 and 4 only
    steps = [json.loads(x)["step"] for x in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    evals = [json.loads(x)["step"] for x in (tmp_path / "run" / "eval_log.jsonl").read_text().splitlines()]
    samples = {json.loads(x)["step"] for x in (tmp_path / "run" / "samples.jsonl").read_text().splitlines()}
    assert steps == evals == [1, 2, 3, 4]
    assert samples == {1, 2, 3, 4}
    assert result["total_steps"] == 4


class VLLMRollout:  # stands in for rlvr_speedrun.vllm_rollout.VLLMRollout (grpo._is_vllm checks the class name)
    def __init__(self):
        self.calls = []

    def generate(self, encoded, n, temperature, max_tokens, pad_id, puzzles=None, think_budget=0):
        self.calls.append((n, temperature))
        ids = encoded["input_ids"].repeat_interleave(n, dim=0)
        completion = torch.full((ids.shape[0], 2), 5, dtype=torch.long)
        return torch.cat([ids, completion], dim=1), torch.full((ids.shape[0],), 2, dtype=torch.long)


def test_sampled_ladder_eval_scores_every_sample_and_weighs_levels_equally():
    # TinyTokenizer decodes even rows to a correct "1 + 2" and odd rows to a wrong "1 + 1"
    puzzles = [Puzzle((1, 2), 3, level="L1"), Puzzle((1, 2), 3, level="L1"), Puzzle((1, 2), 3, level="L2")]
    engine = VLLMRollout()
    config = GRPOConfig(model="tiny", eval_samples=2, eval_batch_size=8, max_new_tokens=4)
    result = grpo._evaluate(_tiny_model(), TinyTokenizer(), puzzles, config, torch.device("cpu"), engine)
    assert engine.calls == [(2, 1.0)]
    assert result["n"] == 6 and result["pass_rate"] == 0.5
    assert result["pass_by_level"] == {"L1": 0.5, "L2": 0.5}
    assert result["ladder_score"] == 0.5
    greedy = grpo._evaluate(_tiny_model(), TinyTokenizer(), puzzles, GRPOConfig(model="tiny", eval_batch_size=8), torch.device("cpu"), engine)
    assert engine.calls[-1] == (1, 0.0)
    assert greedy["pass_by_level"] == {"L1": 0.5, "L2": 1.0} and greedy["ladder_score"] == 0.75


def test_ladder_eval_set_is_truncated_per_level(tmp_path):
    path = tmp_path / "ladder.jsonl"
    rows = [Puzzle((1, 2), 3, level=lv).to_dict() for lv in ["L1"] * 5 + ["L2"] * 5]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = grpo._load_eval_set(GRPOConfig(model="tiny", eval_set=str(path), eval_limit=4))
    assert [p.level for p in kept] == ["L1", "L1", "L2", "L2"]


def test_partial_credit_rewards_right_numbers_and_closeness_only_for_wrong_answers():
    from rlvr_speedrun.countdown import verify

    puzzle = Puzzle((3, 7, 8, 2), 26)
    credit = lambda text: grpo.partial_credit(verify(text, puzzle), puzzle)  # noqa: E731
    assert credit("<answer>8 * 2 + 7 + 3</answer>") == 0.0  # correct: the verifier's 1.0 is the reward
    assert credit("no answer") == 0.0
    assert credit("<answer>8 * 3 + 7 - 2</answer>") > credit("<answer>8 * 3 + 7 - 2 + 5 + 5</answer>")  # extra numbers
    assert credit("<answer>8 * 3 + 7 - 2</answer>") > credit("<answer>8 * 3 + 2 - 7</answer>")  # 29 is closer to 26 than 19
    assert credit("<answer>26</answer>") == 0.0  # writing the target is not close to solving it
    assert credit("<answer>8</answer>") == 0.0  # nor is writing one of the numbers
    assert credit("<answer>8 * 3 + 7</answer>") == 0.25 * 3 / 4  # partial numbers: overlap only, no closeness
    assert credit("<answer>8 * 3 + 7</answer>") < 0.5 <= credit("<answer>8 + 3 + 7 + 2</answer>")  # using every number pays most


def test_cov_clip_stops_gradient_only_through_the_highest_covariance_tokens():
    lp = torch.tensor([-0.1, -3.0, -0.2, -2.0], requires_grad=True)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    # covariances: confident + positive (token 0) and unlikely + negative (token 3) are largest
    out = grpo._cov_clip(lp, adv, 0.5)
    (out * adv).sum().backward()
    assert lp.grad.tolist() == [0.0, 1.0, -1.0, 0.0]
    assert torch.equal(out.detach(), lp.detach())


def test_lr_schedule_warms_up_then_decays_over_the_time_budget():
    base = dict(lr=1e-6, time_budget_s=1000.0, lr_final_frac=0.1)
    assert grpo.lr_at(GRPOConfig(**base), 500.0) == 1e-6
    cosine = GRPOConfig(**base, lr_schedule="cosine", lr_warmup_s=100.0)
    assert grpo.lr_at(cosine, 0.0) == 0.0
    assert grpo.lr_at(cosine, 50.0) == pytest.approx(0.5e-6)
    assert grpo.lr_at(cosine, 100.0) == pytest.approx(1e-6)
    assert grpo.lr_at(cosine, 550.0) == pytest.approx(0.55e-6)
    assert grpo.lr_at(cosine, 1000.0) == pytest.approx(0.1e-6)
    assert grpo.lr_at(cosine, 2000.0) == pytest.approx(0.1e-6)
    linear = GRPOConfig(**base, lr_schedule="linear")
    assert grpo.lr_at(linear, 250.0) == pytest.approx(0.775e-6)
    with pytest.raises(ValueError):
        grpo.lr_at(GRPOConfig(lr_schedule="cosine"), 0.0)  # no time budget to decay over
    with pytest.raises(ValueError):
        grpo.lr_at(GRPOConfig(lr_schedule="step"), 0.0)


def test_run_with_lr_schedule_sets_and_logs_the_learning_rate(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    common = dict(model="tiny", group_size=2, prompts_per_step=1, max_new_tokens=4, max_steps=3, eval_every=10,
                  eval_limit=1, kl_coef=0.0, no_save=True)
    run(GRPOConfig(**common, lr_schedule="linear", time_budget_s=1e6, out_dir=str(tmp_path / "sched")))
    rows = [json.loads(line) for line in (tmp_path / "sched" / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(0.99e-6 < row["lr"] <= 1e-6 for row in rows)
    run(GRPOConfig(**common, out_dir=str(tmp_path / "const")))
    rows = [json.loads(line) for line in (tmp_path / "const" / "train_log.jsonl").read_text().splitlines()]
    assert all("lr" not in row for row in rows)


def test_budget_steps_counts_the_budget_and_the_lr_schedule_in_steps(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    common = dict(model="tiny", group_size=2, prompts_per_step=1, max_new_tokens=4, max_steps=10, eval_every=100,
                  eval_limit=1, kl_coef=0.0, no_save=True, lr_schedule="linear", lr_final_frac=0.1)
    # a 300-s budget counted as 3 steps: the run stops after step 3 however long the steps take
    result = run(GRPOConfig(**common, time_budget_s=300.0, budget_steps=3, out_dir=str(tmp_path / "steps")))
    rows = [json.loads(line) for line in (tmp_path / "steps" / "train_log.jsonl").read_text().splitlines()]
    evals = [json.loads(line) for line in (tmp_path / "steps" / "eval_log.jsonl").read_text().splitlines()]
    assert result["total_steps"] == 3 and [row["step"] for row in rows] == [1, 2, 3]
    assert [row["lr"] for row in rows] == pytest.approx([1e-6, 0.7e-6, 0.4e-6])  # lr at 0, 100 and 200 s of 300
    assert [row["clock_s"] for row in rows] == pytest.approx([100.0, 200.0, 300.0])
    assert [(e["step"], e["clock_s"]) for e in evals] == [(3, pytest.approx(300.0))]
    # max_steps still stops a run first, on the same schedule
    run(GRPOConfig(**{**common, "max_steps": 2}, time_budget_s=300.0, budget_steps=3, out_dir=str(tmp_path / "early")))
    rows = [json.loads(line) for line in (tmp_path / "early" / "train_log.jsonl").read_text().splitlines()]
    assert [row["lr"] for row in rows] == pytest.approx([1e-6, 0.7e-6])
    with pytest.raises(ValueError, match="budget_steps"):
        run(GRPOConfig(**common, time_budget_s=300.0, budget_steps=-1, out_dir=str(tmp_path / "bad")))


def test_budget_steps_time_stop_also_stops_a_step_capped_run_at_its_time_budget(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    common = dict(model="tiny", group_size=2, prompts_per_step=1, max_new_tokens=4, max_steps=10, eval_every=100,
                  eval_limit=1, kl_coef=0.0, no_save=True, time_budget_s=1e-6, budget_steps=3)
    # every step outlasts a 1-us budget: by default the run still takes its 3 steps; with the time stop it ends
    # after step 1, with its final eval there
    assert run(GRPOConfig(**common, out_dir=str(tmp_path / "off")))["total_steps"] == 3
    assert run(GRPOConfig(**common, budget_steps_time_stop=True, out_dir=str(tmp_path / "on")))["total_steps"] == 1
    evals = [json.loads(line) for line in (tmp_path / "on" / "eval_log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [1]
    with pytest.raises(ValueError, match="budget_steps_time_stop"):
        run(GRPOConfig(**{**common, "budget_steps": 0}, budget_steps_time_stop=True, out_dir=str(tmp_path / "bad")))


def test_eval_max_new_tokens_sets_the_eval_limit_and_leaves_training_rollouts_at_max_new_tokens(monkeypatch, tmp_path):
    model, _, _ = _patch_tiny_training(monkeypatch, tmp_path)
    limits = []
    generate = model.generate

    def spy(*args, **kwargs):  # (sampled = a training rollout, limit)
        limits.append((kwargs["do_sample"], kwargs["max_new_tokens"]))
        return generate(*args, **kwargs)

    monkeypatch.setattr(model, "generate", spy)
    common = dict(model="tiny", group_size=2, prompts_per_step=1, max_new_tokens=3, max_steps=1, eval_every=100,
                  eval_limit=1, kl_coef=0.0, no_save=True)
    run(GRPOConfig(**common, out_dir=str(tmp_path / "same")))
    assert set(limits) == {(True, 3), (False, 3)}
    limits.clear()
    run(GRPOConfig(**common, eval_max_new_tokens=5, out_dir=str(tmp_path / "longer")))
    assert set(limits) == {(True, 3), (False, 5)}
    with pytest.raises(ValueError, match="eval_max_new_tokens"):
        run(GRPOConfig(**common, eval_max_new_tokens=5, cache_len=16, out_dir=str(tmp_path / "bad")))


def _ragged_batch(seed: int, lengths: list[int], prompt_width: int = 4):
    torch.manual_seed(seed)
    generated = torch.randint(2, 40, (len(lengths), prompt_width + max(lengths)))
    attention = torch.ones_like(generated)
    for row, length in enumerate(lengths):  # right-padded completions
        attention[row, prompt_width + length :] = 0
    return generated, attention, prompt_width


@pytest.mark.parametrize("offset, cap, scale", [(0.0, 2.0, 1.0), (-5.0, 2.0, 2.0), (1.0, 2.0, math.exp(-1.0))])
def test_truncated_importance_weights_scale_the_policy_gradient(offset, cap, scale):
    """Sampler log-probs = the policy's + offset give every token the ratio exp(-offset), capped at tis_cap."""
    lengths = [7, 2, 5, 3, 7, 1]
    generated, attention, prompt_width = _ragged_batch(2, lengths)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
    base = _tiny_model().eval()
    with torch.no_grad():
        token_lp, _, _ = grpo._completion_logprobs(base, generated, attention, prompt_width)
    grads, sums = [], None
    for extra, sampler_lp in (({}, None), ({"tis_cap": cap}, token_lp + offset)):
        model = copy.deepcopy(base)
        config = GRPOConfig(model="tiny", kl_coef=0.0, grad_clip=0.0, micro_batch_size=4, **extra)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        *_, sums = grpo._update(
            model, None, generated, attention, advantages, prompt_width, 0, config, optimizer, torch.device("cpu"), grpo.SINGLE, None, sampler_lp
        )
        grads.append([p.grad.clone() for p in model.parameters()])
    for plain, weighted in zip(*grads):
        assert torch.allclose(scale * plain, weighted, atol=1e-6)
    stats = grpo._mismatch_stats(sums, grpo.SINGLE, cap)
    assert stats["mismatch_k1"] == pytest.approx(offset, abs=1e-5)
    assert stats["mismatch_k3"] == pytest.approx(math.exp(-offset) - 1 + offset, abs=1e-4)
    assert stats["mismatch_far"] == (1.0 if abs(offset) > math.log(2) else 0.0)
    assert stats["mismatch_seq"] == pytest.approx(abs(offset) * sum(lengths) / len(lengths), abs=1e-4)
    assert stats["mismatch_max"] == pytest.approx(abs(offset), abs=1e-5)
    assert stats["tis_clipped"] == (1.0 if math.exp(-offset) > cap else 0.0)


def test_grpo_step_logs_the_sampler_mismatch_of_the_kept_rows():
    """The sampler's log-probs follow the group selection: only kept rows are compared, position by position."""

    class FirstTokenTokenizer(TinyTokenizer):  # a row is correct when its first completion token is even
        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["<answer>1 + 2</answer>" if int(row[0]) % 2 == 0 else "<answer>1 + 1</answer>" for row in sequences]

    model = _tiny_model().eval()
    lengths, first, shift = [3, 5, 2, 4, 5, 1, 4, 3], [3, 5, 7, 9, 4, 5, 6, 7], 0.3  # groups 0-1 all wrong, 2-3 mixed

    class VLLMRollout:  # grpo._is_vllm checks the class name
        def generate(self, encoded, n, temperature, max_tokens, pad_id, puzzles=None, think_budget=0, return_logprobs=False):
            torch.manual_seed(6)
            ids = encoded["input_ids"].repeat_interleave(n, dim=0)
            completion = torch.full((ids.shape[0], max(lengths)), pad_id, dtype=torch.long)
            valid = torch.zeros_like(completion)
            for row, (length, token) in enumerate(zip(lengths, first)):
                completion[row, :length] = torch.randint(2, 40, (length,))
                completion[row, 0] = token
                valid[row, :length] = 1
            sequences = torch.cat([ids, completion], dim=1)
            attention = torch.cat([encoded["attention_mask"].repeat_interleave(n, dim=0), valid], dim=1)
            with torch.no_grad():
                lp, _, _ = grpo._completion_logprobs(model, sequences, attention, ids.shape[1])
            assert return_logprobs
            return sequences, torch.tensor(lengths), lp - shift

    config = GRPOConfig(model="tiny", group_size=2, prompts_per_step=2, few_shot=0, kl_coef=0.0, log_mismatch=True, rollout_backend="vllm")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    info = grpo_step(model, None, FirstTokenTokenizer(), [Puzzle((1, 2), 3)] * 4, config, optimizer, torch.device("cpu"), VLLMRollout())
    assert info["kept_groups"] == 2
    assert info["mismatch_k1"] == pytest.approx(-shift, abs=1e-5)
    assert info["mismatch_k3"] == pytest.approx(math.exp(shift) - 1 - shift, abs=1e-5)
    assert info["mismatch_far"] == 0.0 and "tis_clipped" not in info
    assert info["mismatch_seq"] == pytest.approx(shift * sum(lengths[4:]) / 4, abs=1e-5)  # rows 4-7, the kept groups
    assert info["mismatch_max"] == pytest.approx(shift, abs=1e-5)


def test_sampler_logprob_options_need_the_vllm_backend(monkeypatch, tmp_path):
    _patch_tiny_training(monkeypatch, tmp_path)
    for extra in ({"log_mismatch": True}, {"tis_cap": 2.0}):
        with pytest.raises(ValueError, match="rollout_backend='vllm'"):
            run(GRPOConfig(model="tiny", out_dir=str(tmp_path / "run"), **extra))
    with pytest.raises(ValueError, match="tis_cap"):
        run(GRPOConfig(model="tiny", tis_cap=-1.0, rollout_backend="vllm", out_dir=str(tmp_path / "run")))


def test_vllm_rollout_returns_each_sampled_tokens_logprob(monkeypatch):
    from rlvr_speedrun.vllm_rollout import VLLMRollout as Rollout

    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(Rollout, "_params", lambda self, **kwargs: kwargs)

    def flat(values):  # vllm.logprobs.FlatLogprobs with one entry (the sampled token) per position
        return types.SimpleNamespace(start_indices=list(range(len(values))), logprobs=values)

    sample = lambda ids, lps: types.SimpleNamespace(token_ids=ids, logprobs=flat(lps))  # noqa: E731
    outputs = [types.SimpleNamespace(outputs=[sample([5, 6, 7], [-0.1, -0.2, -0.3]), sample([8], [-1.5])])]
    requests = []
    engine = object.__new__(Rollout)
    engine.logprobs = True
    engine.llm = types.SimpleNamespace(generate=lambda prompts, params, use_tqdm: requests.append(params) or outputs)
    encoded = {"input_ids": torch.tensor([[0, 3, 4]]), "attention_mask": torch.tensor([[0, 1, 1]])}
    sequences, finished_at, lp = engine.generate(encoded, 2, 1.0, 4, 0, return_logprobs=True)
    assert requests[0]["logprobs"] is True
    assert sequences.tolist() == [[0, 3, 4, 5, 6, 7], [0, 3, 4, 8, 0, 0]] and finished_at.tolist() == [3, 1]
    assert torch.allclose(lp, torch.tensor([[-0.1, -0.2, -0.3], [-1.5, 0.0, 0.0]]))
    assert len(engine.generate(encoded, 2, 1.0, 4, 0)) == 2 and requests[1]["logprobs"] is False
    engine.logprobs = False
    with pytest.raises(ValueError, match="logprobs=True"):
        engine.generate(encoded, 2, 1.0, 4, 0, return_logprobs=True)
