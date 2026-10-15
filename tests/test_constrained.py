import random

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from rlvr_speedrun import grpo
from rlvr_speedrun.constrained import (
    OPEN,
    AnswerGrammar,
    RequestConstraint,
    feed,
    initial_state,
    open_token_ids,
    token_strings,
)
from rlvr_speedrun.countdown import Puzzle, verify


def _done(numbers, text, ops="+-*/"):
    state = feed(initial_state(numbers), text, ops)
    return state is not None and state[2] == "done"


def test_grammar_admits_exactly_the_well_formed_uses_of_the_numbers():
    assert _done((3, 7, 8, 2), "(3 + 7) * 2 - 8</answer>")
    assert _done((1, 13), " 13-1 </answer>")  # 1 is a prefix of 13
    assert feed(initial_state((3, 7)), "3 + 3", "+-*/") is None  # a number used twice
    assert feed(initial_state((3, 7)), "3 * 7", "+-") is None  # an operator the puzzle does not allow
    assert feed(initial_state((3, 7)), "3</", "+-*/") is None  # closing before every number is used
    assert feed(initial_state((3, 7)), "(3 + 7</", "+-*/") is None  # unbalanced
    assert feed(initial_state((3, 7)), "3 + 7)", "+-*/") is None
    assert feed(initial_state((3, 7)), "-3", "+-*/") is None  # no unary minus
    assert feed(initial_state((3, 7)), "37", "+-*/") is None
    assert feed(initial_state((3, 7)), "3  +", "+-*/") is None  # at most one space in a row
    assert feed(initial_state((3, 7)), "((3", "+-*/") is None  # more "(" than operators left
    assert _done((3, 7, 8), "((3 + 7) * 8)</answer>")


@pytest.fixture(scope="module")
def grammar_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    return AnswerGrammar(token_strings(tokenizer), open_token_ids(tokenizer)), tokenizer


def test_random_constrained_answers_are_always_well_formed(grammar_and_tokenizer):
    grammar, _ = grammar_and_tokenizer
    rng = random.Random(0)
    for numbers, ops in [((3, 7, 8, 2), "+-*/"), ((1, 13, 25), "+-"), ((4, 4, 10, 6, 75), "-/"), ((2, 50), "+*")]:
        puzzle = Puzzle(numbers, 1, ops=ops)
        for _ in range(20):
            state, text = initial_state(numbers), ""
            while state[2] != "done":
                token = rng.choice(grammar.allowed(state, ops))
                text += grammar.strings[token]
                state = feed(state, grammar.strings[token], ops)
            result = verify(OPEN + text, puzzle)
            if result.reason != "division by zero":  # the only failure a grammar cannot rule out
                assert not result.malformed and "numbers used" not in result.reason and "not allowed" not in result.reason, text


def test_allowed_per_position_is_free_before_the_answer_and_covers_each_sampled_token(grammar_and_tokenizer):
    grammar, tokenizer = grammar_and_tokenizer
    ids = tokenizer.encode("I will try <answer>(3 + 7) * 2</answer>", add_special_tokens=False)
    allowed = grammar.allowed_per_position(ids, (2, 3, 7), "+-*/")
    first = next(i for i, a in enumerate(allowed) if a is not None)
    assert OPEN in tokenizer.decode(ids[:first]) and OPEN not in tokenizer.decode(ids[: first - 1])
    assert all(a is None or tid in a for tid, a in zip(ids, allowed))
    assert all(a is not None for a in allowed[first:])


def test_request_constraint_masks_logits_only_inside_the_answer(grammar_and_tokenizer):
    grammar, tokenizer = grammar_and_tokenizer
    constraint = RequestConstraint(grammar, (3, 7), "+-", {})
    output = tokenizer.encode("ok <answer>3", add_special_tokens=False)
    logits = torch.randn(len(tokenizer))
    assert torch.equal(constraint(output[:1], logits.clone()), logits)
    masked = constraint(output, logits.clone())
    finite = set(torch.isfinite(masked).nonzero().flatten().tolist())
    assert finite == set(grammar.allowed(feed(initial_state((3, 7)), "3", "+-"), "+-"))
    assert {grammar.strings[i] for i in finite} >= {"+", "-", " +", " "} and not {"*", "  ", "7", ")"} & {grammar.strings[i] for i in finite}


def test_constrained_logprobs_are_renormalised_over_the_allowed_tokens():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=40, n_positions=64, n_embd=16, n_layer=1, n_head=2)).eval()
    prompt = torch.tensor([[3, 4, 5]])
    total = 0.0
    for label in (7, 9, 11):
        ids = torch.cat([prompt, torch.tensor([[6, label]])], dim=1)
        mask = torch.ones_like(ids)
        lp, _, entropy = grpo._completion_logprobs(model, ids, mask, 3, allowed=[[None, (7, 9, 11)]])
        total += lp[0, 1].exp().item()
        free, _, _ = grpo._completion_logprobs(model, ids, mask, 3)
        assert lp[0, 0] == free[0, 0] and lp[0, 1] > free[0, 1]
        assert torch.isfinite(entropy).all() and entropy[0, 1] <= torch.log(torch.tensor(3.0)) + 1e-5
    assert total == pytest.approx(1.0, abs=1e-5)


def test_ids_past_the_tokenizer_are_treated_as_empty_text(grammar_and_tokenizer):
    grammar, tokenizer = grammar_and_tokenizer
    constraint = RequestConstraint(grammar, (3, 7), "+-", {})
    logits = torch.randn(len(tokenizer) + 64)  # padded embedding matrix
    assert torch.equal(constraint([len(tokenizer) + 10], logits.clone()), logits)
    assert grammar.allowed_per_position([len(tokenizer) + 10], (3, 7), "+-") == [None]


def test_think_budget_forces_the_answer_open_and_then_constrains_it(grammar_and_tokenizer):
    grammar, tokenizer = grammar_and_tokenizer
    budget, open_ids = 5, grammar.open_ids
    constraint = RequestConstraint(grammar, (3, 7), "+-", {}, think_budget=budget)
    rambling = tokenizer.encode("the cat sat on the mat and then", add_special_tokens=False)[:budget]
    output = list(rambling)
    for k in range(len(open_ids)):  # each forced position allows exactly the next token of "\n<answer>"
        masked = constraint(output, torch.randn(len(tokenizer)))
        assert torch.isfinite(masked).nonzero().flatten().tolist() == [open_ids[k]]
        output.append(open_ids[k])
    masked = constraint(output, torch.randn(len(tokenizer)))
    assert set(torch.isfinite(masked).nonzero().flatten().tolist()) == set(grammar.allowed(initial_state((3, 7)), "+-"))
    allowed = grammar.allowed_per_position(output + [grammar.allowed(initial_state((3, 7)), "+-")[0]], (3, 7), "+-", budget)
    assert allowed[:budget] == [None] * budget and allowed[budget : budget + len(open_ids)] == [(t,) for t in open_ids]
    assert allowed[-1] == grammar.allowed(initial_state((3, 7)), "+-")
    assert RequestConstraint(grammar, (3, 7), "+-", {}, think_budget=0)(rambling, torch.zeros(len(tokenizer))).isfinite().all()
