import json

import pytest

from scripts.new_record import scaffold
from scripts.validate_record import validate_record


def _write_record(records, name, config):
    record = records / "track_a" / name
    record.mkdir(parents=True)
    (record / "config.json").write_text(json.dumps(config) + "\n")
    (record / "README.md").write_text("x\n")
    return record


def test_scaffold_copies_holder_config_and_numbers_next(tmp_path):
    records = tmp_path / "records"
    _write_record(records, "000_smoke", {"unranked": True, "track": "track_a", "lr": 1.0})
    holder = _write_record(
        records,
        "005_holder",
        {"track": "track_a", "model": "Qwen/Qwen2.5-0.5B", "provider": {"0": "modal", "1": "runpod"}, "seeds": [0, 1], "lr": 5e-6, "compile": True},
    )
    record = scaffold(records, "track_a", "muon", "RunPod", None, 3)
    assert record.name == "006_muon" and record.is_dir()
    config = json.loads((record / "config.json").read_text())
    assert config["parent"] == holder.name
    assert config["provider"] == "runpod" and config["seeds"] == [0, 1, 2]
    assert config["lr"] == 5e-6 and config["compile"] is True
    assert "unranked" not in config
    readme = (record / "README.md").read_text()
    assert "--config" in readme and "005_holder" in readme

    # the scaffold alone must not pass ranked validation (no seeds yet)
    valid, message = validate_record(record)
    assert not valid and "seed" in message

    assert scaffold(records, "track_a", "muon", "runpod", None, 3).name == "007_muon"
    with pytest.raises(SystemExit):
        scaffold(records, "track_a", "Bad-Slug", "runpod", None, 3)
