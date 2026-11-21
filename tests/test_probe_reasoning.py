from rlvr_speedrun.countdown import SEARCH_EXAMPLES, Puzzle
from scripts.research.probe_reasoning import (
    cut_countdown,
    cut_gsm8k,
    grounded_chain,
    gsm8k_gold,
    gsm8k_shot,
    loops,
    score_countdown,
    score_gsm8k,
    steps,
    summarise,
)


def test_search_examples_are_all_right_and_grounded():
    for puzzle, reasoning, _ in SEARCH_EXAMPLES:
        found = steps(reasoning)
        assert len(found) >= 3 and all(s["right"] for s in found), found
        assert grounded_chain(found, puzzle.numbers) == len(found)


def test_steps_read_chains_brackets_and_sentences():
    found = steps("Try (10 - 5 + 3) * 2 = 16. 16 + 1 = 18, so 0.2 x 50 = 10 and 12/60 = 0.2 and 10 / 3 = 3.33.")
    assert [(s["lhs"], s["right"]) for s in found] == [
        ("(10 - 5 + 3) * 2", True), ("16 + 1", False), ("0.2 * 50", True), ("12/60", True), ("10 / 3", True)]
    assert steps("x = 5 and the answer = 26 and [3, 7] = 2") == []


def test_grounding_needs_given_numbers_or_earlier_results():
    found = steps("8 * 3 = 24, then 24 + 1 = 25, and 1 + 2 = 3, then 3 * 2 = 6")
    assert grounded_chain(found, (1, 2)) == 2  # 8 and 24 are not given; 1 + 2 and then 3 * 2 are


def test_cuts_and_scores():
    p = Puzzle((3, 7, 8, 2), 26)
    text = cut_countdown(" 8 * 2 = 16, 16 + 7 + 3 = 26</think>\n<answer>8 * 2 + 7 + 3</answer>\n\nUser: more")
    assert text.endswith("</answer>")
    assert score_countdown(text, p) == {"well_formed": True, "solved": True, "reason": "correct"}
    assert score_countdown(" no tags here", p)["well_formed"] is False
    assert cut_gsm8k(" She has 3 + 4 = 7 apples.\nThe answer is 7.\n\nQuestion: next") == " She has 3 + 4 = 7 apples.\nThe answer is 7."


def test_gsm8k_format_and_score():
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips.\n#### 1,272"
    assert gsm8k_gold(answer) == "1272"
    shot = gsm8k_shot("How many?", answer)
    assert "<<" not in shot and shot.endswith("The answer is 1,272.\n\n")
    assert score_gsm8k(" so 1,000 + 272 = 1,272.\nThe answer is $1,272.", "1272")["solved"]
    loose = score_gsm8k(" 3 + 4 = 7 so 7", "7")
    assert loose["solved"] and not loose["well_formed"]


def test_loops_and_summary():
    assert loops("a b c d " * 4) and not loops("a b c d " * 3)
    rows = [{"n_steps": 2, "n_right": 1, "chain": 2, "solved": True, "well_formed": True, "tokens": 10, "loop": False},
            {"n_steps": 0, "n_right": 0, "chain": 0, "solved": False, "well_formed": False, "tokens": 30, "loop": True}]
    s = summarise(rows)
    assert s["step_accuracy"] == 0.5 and s["grounded_chain_rate"] == 0.5 and s["median_tokens"] == 20 and s["loop_rate"] == 0.5
