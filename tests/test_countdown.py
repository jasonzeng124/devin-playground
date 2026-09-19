import random
import time

import pytest

from rlvr_speedrun.countdown import (
    CORRECT_REWARD,
    INCORRECT_REWARD,
    MALFORMED_PENALTY,
    FEW_SHOT_EXAMPLES,
    Puzzle,
    extract_expression,
    generate_puzzle,
    generate_puzzles,
    safe_eval,
    verify,
    format_prompt,
)

P = Puzzle((3, 5, 7, 2), 24)


def test_correct_with_tags():
    r = verify("some thinking...\n<answer>(3 + 5) * (7 - 2) / 2 + 4</answer>", Puzzle((3, 5, 7, 2, 2, 4), 24))
    assert r.correct and r.reward == CORRECT_REWARD


def test_correct_plain_last_line():
    r = verify("we can do\n3 * 5 + 7 + 2", P)
    assert r.correct


def test_trailing_equals_stripped():
    assert verify("<answer>3 * 5 + 7 + 2 = 24</answer>", P).correct


def test_wrong_value():
    r = verify("<answer>3 + 5 + 7 + 2</answer>", P)
    assert not r.correct and not r.malformed and r.reward == INCORRECT_REWARD


def test_number_reuse_rejected():
    r = verify("<answer>3 * 7 + 3</answer>", P)
    assert not r.correct and not r.malformed


def test_missing_number_rejected():
    assert not verify("<answer>3 * 7 + 5 - 2 + 0</answer>", P).correct


def test_malformed_penalty():
    for bad in ["", "hello world", "<answer>__import__('os')</answer>", "<answer>3 ** 5</answer>", "<answer>x + 1</answer>"]:
        r = verify(bad, P)
        assert r.malformed and r.reward == MALFORMED_PENALTY, bad


def test_division_by_zero_is_malformed():
    assert verify("<answer>3 / (5 - 5)</answer>", Puzzle((3, 5, 5), 1)).malformed


def test_exact_division_fractions():
    assert verify("<answer>(7 / 2) * 2 + 3 + 5</answer>", Puzzle((7, 2, 2, 3, 5), 15)).correct


def test_unicode_operators():
    assert safe_eval("3 × 5 ÷ 5")[0] == 3


def test_extract_prefers_last_answer_block():
    assert extract_expression("<answer>1+1</answer> no wait <answer>2+2</answer>") == "2+2"


def test_few_shot_examples_are_valid_and_prompt_is_bounded():
    for puzzle, _reasoning, answer in FEW_SHOT_EXAMPLES:
        assert verify(answer, puzzle).correct
    assert format_prompt(P) == format_prompt(P, few_shot=0)
    assert "Question:" in format_prompt(P, few_shot=3)
    assert format_prompt(P, few_shot=99).count("Question:") == len(FEW_SHOT_EXAMPLES) + 1


def test_generated_puzzles_are_solvable_and_deterministic():
    a = generate_puzzles(seed=0, n=200)
    b = generate_puzzles(seed=0, n=200)
    assert a == b
    for p in a:
        assert verify(f"<answer>{p.solution}</answer>", p).correct
        assert p.target not in p.numbers


def test_difficulty_knobs():
    rng = random.Random(1)
    p = generate_puzzle(rng, n_numbers=6, number_range=(1, 100), target_range=(100, 999))
    assert len(p.numbers) == 6 and 100 <= p.target <= 999


def test_verifier_is_fast():
    t = time.perf_counter()
    for _ in range(1000):
        verify("<answer>(3 + 5) * (7 - 2) / 2 + 4</answer>", P)
    assert (time.perf_counter() - t) / 1000 < 0.01  # <10ms per call


@pytest.mark.parametrize("expr", ["1e3", "1.5 + 2", "True + 1", "(1)(2)", "[1, 2]"])
def test_reject_non_integer_or_odd_syntax(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)
