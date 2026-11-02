import importlib.util
import json
from pathlib import Path

from rlvr_speedrun.countdown import R1_TEMPLATE

spec = importlib.util.spec_from_file_location("sft_public", Path(__file__).parents[1] / "scripts/research/sft_public.py")
sft_public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft_public)


def test_format_example_keeps_published_text():
    prompt, completion = sft_public.format_example("How many?", "48/2 = <<48/2=24>>24 clips.\n#### 24")
    assert prompt.startswith(R1_TEMPLATE.partition("User: ")[0] + "User: How many?\n")
    assert prompt.endswith(R1_TEMPLATE.partition("Assistant: ")[1] + R1_TEMPLATE.partition("Assistant: ")[2])
    assert completion == " 48/2 = <<48/2=24>>24 clips.</think>\n<answer>24</answer>"


def test_load_rows_drops_numbers_game_questions(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"question": "Play the 24 game with 1 2 3 4.", "answer": "x\n#### 24"}, {"question": "Sum 2 and 3.", "answer": "2+3=5\n#### 5"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = sft_public.load_rows(str(path))
    assert len(loaded) == 1 and loaded[0][1].endswith("<answer>5</answer>")


def test_batches_cover_each_row_once_per_epoch():
    rows = list(range(10))
    gen = sft_public.batches(rows, 5, seed=0)
    assert sorted(next(gen) + next(gen)) == rows
