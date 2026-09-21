"""The eval set and capability-probe data are frozen: any change to them breaks
comparability of every record, so their checksums are pinned here (RULES.md,
"Forbidden"). Regenerating them on purpose means a new benchmark version."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FROZEN = {
    "data/countdown_eval.jsonl": "ca3ab9fbb06a0824bc8aec3a9136ff7ff735eae04485d248a3d27e94c049a332",
    "data/capability/fineweb_edu_val.jsonl": "8d2026791506c49629559aafe4379fb6ff503d638a1e5c0b66a5d53a8044e858",
    "data/capability/mmlu_lite.jsonl": "51c8caf4dc19efe9af30f9f0f98c1ca961d10a579ff7e2c6bc691fb6f783a9a0",
}


@pytest.mark.parametrize("relpath", sorted(FROZEN))
def test_frozen_file_unchanged(relpath):
    digest = hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()
    assert digest == FROZEN[relpath], f"{relpath} changed; the frozen data must not be edited"


def test_eval_set_size():
    lines = (ROOT / "data/countdown_eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2000
