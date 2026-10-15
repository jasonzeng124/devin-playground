import pytest

from rlvr_speedrun.ladder_confirm import score


def test_ladder_score_weighs_levels_equally_whatever_their_size():
    correct = [[True, False], [False, False], [True, True]]  # L1: 0.5 and 0; L2: 1.0
    result = score(correct, ["L1", "L1", "L2"])
    assert result["pass@1_by_level"] == {"L1": 0.25, "L2": 1.0}
    assert result["ladder_score"] == pytest.approx(0.625)
    assert result["pass@2"] == pytest.approx(2 / 3)
