import random
import time

import pytest

from rlvr_speedrun.countdown import (
    CORRECT_REWARD,
    FEW_SHOT_EXAMPLES,
    INCORRECT_REWARD,
    MALFORMED_PENALTY,
    Puzzle,
    extract_expression,
    format_prompt,
    generate_puzzle,
    generate_puzzles,
    safe_eval,
    verify,
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


def test_search_examples_are_correct_and_prompts_split_exactly():
    from rlvr_speedrun.countdown import PROMPT_STYLES, SEARCH_EXAMPLES, split_prompt

    for puzzle, _, answer in SEARCH_EXAMPLES:
        assert verify(f"<answer>{answer}</answer>", puzzle).correct
    for style in PROMPT_STYLES:
        prefixes = {split_prompt(p, 3, style)[0] for p in (P, Puzzle((1, 2, 3, 4), 10))}
        assert len(prefixes) == 1  # shared prefix, so its KV cache can be reused across the batch
        assert "".join(split_prompt(P, 3, style)) == format_prompt(P, 3, style)
    assert format_prompt(P, 3, "search").endswith("<think>")


def test_level_is_the_smallest_operator_set_that_reaches_the_target():
    from rlvr_speedrun.countdown import Puzzle, level

    assert level(Puzzle([1, 2, 3, 4], 10)) == "sum"
    assert level(Puzzle([1, 2, 3, 4], 2)) == "+-"  # 4 + 1 - 3 ... any +- expression
    assert level(Puzzle([2, 3, 5, 7], 210)) == "+-*"
    assert level(Puzzle([1, 3, 4, 6], 24)) == "+-*/"  # 6 / (1 - 3 / 4)


def test_restricted_operators_are_stated_in_the_prompt_and_enforced():
    from rlvr_speedrun.countdown import Puzzle, format_prompt, verify

    default, restricted = Puzzle((3, 7, 8, 2), 26), Puzzle((3, 7, 8, 2), 26, ops="+-*")
    assert "You may use +, -, *, / and" in format_prompt(default, 0, "r1")
    assert "You may use +, -, * and" in format_prompt(restricted, 0, "r1")
    assert verify("<answer>8 * 2 + 7 + 3</answer>", restricted).correct
    result = verify("<answer>8 / 2 * 7 - 3 + 1</answer>", Puzzle((8, 2, 7, 3, 1), 26, ops="+-*"))
    assert not result.correct and not result.malformed and "not allowed" in result.reason


def test_ladder_levels_meet_their_requirements():
    import random
    from fractions import Fraction

    from rlvr_speedrun.countdown import reachable, verify
    from rlvr_speedrun.ladder import (
        LEVELS,
        _plus_minus_values,
        build_ladder_eval,
        generate_level_puzzle,
        ladder_stream,
        puzzle_key,
    )

    rng = random.Random(0)
    for level in LEVELS:
        for _ in range(20):
            p = generate_level_puzzle(rng, level)
            assert len(p.numbers) == level.n_numbers and p.ops in level.ops and p.level == level.name
            assert verify(f"<answer>{p.solution}</answer>", p).correct
            nums = tuple(sorted(Fraction(x) for x in p.numbers))
            if level.name == "L1":
                assert p.target != sum(p.numbers)
            elif level.name == "L2":
                assert Fraction(p.target) not in reachable(nums, "+-")
            elif level.name == "L3":
                assert Fraction(p.target) not in reachable(nums, "+-*")
            elif level.name in ("L4", "L5"):
                assert p.target not in _plus_minus_values(p.numbers)
    small = build_ladder_eval(per_level=5)
    assert small == build_ladder_eval(per_level=5)  # deterministic
    keys = frozenset(puzzle_key(p) for p in small)
    stream = ladder_stream(0, exclude=keys)
    assert all(puzzle_key(next(stream)) not in keys for _ in range(50))
