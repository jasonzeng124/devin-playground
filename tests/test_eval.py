from rlvr_speedrun.countdown import Puzzle
from rlvr_speedrun.eval import aggregate_results, wilson_interval


def test_wilson_interval_bounds_and_zero_case():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert 0 <= low < 0.5 < high <= 1


def test_aggregate_results():
    puzzles = [Puzzle((1, 2), 3), Puzzle((2, 2), 4), Puzzle((1, 3), 5)]
    result = aggregate_results(
        ["<answer>1 + 2</answer>", "<answer>2 + 2</answer>", "not arithmetic"],
        puzzles,
    )
    assert result["n"] == 3
    assert result["pass_rate"] == 2 / 3
    assert result["malformed_rate"] == 1 / 3
    assert len(result["wilson_95_ci"]) == 2
