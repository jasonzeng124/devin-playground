import json

import rlvr_speedrun.confirm as confirm
from rlvr_speedrun import benchmark
from rlvr_speedrun.countdown import Puzzle
from rlvr_speedrun.data import REPO_ROOT, load_puzzles


def _fake_generate(solved):
    """Answers correctly exactly for puzzles whose target is in `solved`; records the protocol used."""
    calls = []

    def generate(model, tokenizer, prompts, max_new_tokens, temperature, device):
        calls.append((max_new_tokens, temperature, len(prompts)))
        outputs = []
        for prompt in prompts:
            target = int(prompt.rsplit("equals ", 1)[1].split(".")[0])
            outputs.append(f"<answer>{solved[target]}</answer>" if target in solved else "<answer>1 + 1</answer>")
        return outputs

    return generate, calls


def test_score_keeps_puzzle_order_and_fixed_protocol(monkeypatch):
    puzzles = [Puzzle((1, 2, 3, 4), 24, "1 * 2 * 3 * 4"), Puzzle((5, 5, 1, 1), 27, "5 * 5 + 1 + 1"), Puzzle((2, 3, 4, 5), 16, "(5 - 3 + 2) * 4")]
    generate, calls = _fake_generate({24: "1 * 2 * 3 * 4", 16: "(5 - 3 + 2) * 4"})
    monkeypatch.setattr(confirm, "generate_completions", generate)
    result = confirm.score(None, None, puzzles, device=None, batch_size=2)
    assert result["n"] == 3 and result["pass_rate"] == 2 / 3
    assert all(tokens == benchmark.MAX_EVAL_NEW_TOKENS and temperature == 0.0 for tokens, temperature, _ in calls)
    assert [n for _, _, n in calls] == [2, 1]


def test_run_confirm_scores_full_frozen_sets(monkeypatch, tmp_path):
    rows = load_puzzles(REPO_ROOT / benchmark.CONFIRM_PATH)
    generate, _ = _fake_generate({p.target: p.solution for p in rows[:10]})
    monkeypatch.setattr(confirm, "generate_completions", generate)
    monkeypatch.setattr(confirm, "load_tokenizer", lambda name: None)
    monkeypatch.setattr(confirm, "load_causal_model", lambda name, device: (None, confirm.torch.device("cpu")))
    result = confirm.run_confirm("some/ckpt", device="cpu", batch_size=512)
    json.dumps(result)
    assert result["protocol"] == {"few_shot": 3, "max_new_tokens": 64, "decoding": "greedy"}
    assert result["confirm"]["n"] == benchmark.CONFIRM_SIZE and result["ood"]["n"] == benchmark.OOD_SIZE
    assert result["confirm"]["pass_rate"] > 0
