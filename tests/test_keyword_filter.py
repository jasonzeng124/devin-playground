from scripts.keyword_filter import countdown_matches, cue_counts


def test_countdown_rule_drops_numbers_game_items():
    for text in (
        "In the Countdown numbers round you get six numbers.",
        "Play the 24 game: use 3, 3, 8, 8.",
        "A classic game of 24 puzzle.",
        "Use + - * / to make 24 from 1, 2, 3, 4.",
        "Reach the target number with the cards.",
        "Use each of the four numbers exactly once.",
        "Krypto is a card game.",
        "a count-down timer",
    ):
        assert countdown_matches(text), text


def test_countdown_rule_keeps_ordinary_math():
    for text in (
        "There are 24 students in the class.",
        "A day has 24 hours, so 3 days have 72 hours.",
        "Solve 2x + 3 = 11. First, subtract 3 from both sides.",
        "The numbers 4 and 6 have a least common multiple of 12.",
    ):
        assert not countdown_matches(text), text


def test_cue_counts():
    c = cue_counts("Let's solve it step by step. First, we get 4. Therefore the answer is 4. Let me check.")
    assert c["step by step"] == 1
    assert c["let's / let us"] == 1
    assert c["let me"] == 1
    assert c["first,"] == 1
    assert c["therefore / thus"] == 1
    assert c["the answer is"] == 1
    assert c["check / verify"] == 1
    assert c["we get / we have"] == 1
    assert c["wait / hmm / okay"] == 0
