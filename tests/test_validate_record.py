import json

from scripts.validate_record import validate_record


def test_validate_record_smoke_fixture(tmp_path):
    record = tmp_path / "track_a" / "001_test"
    (record / "seeds" / "0").mkdir(parents=True)
    (record / "README.md").write_text("test\n")
    (record / "config.json").write_text("{}\n")
    (record / "seeds" / "0" / "result.json").write_text(
        json.dumps({"final_pass_rate": 0.5, "time_to_threshold_s": None}) + "\n"
    )
    (record / "seeds" / "0" / "train_log.jsonl").write_text('{"step": 1}\n')
    valid, message = validate_record(record, allow_fewer_seeds=True)
    assert valid
    assert "pass_rate" in message
    invalid, message = validate_record(record)
    assert not invalid
    assert "at least 3" in message
