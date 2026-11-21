import numpy as np
import pytest

from scripts.pretrain_scratch import parse_phase2


def test_phase2_reads_files_and_weights(tmp_path):
    main = tmp_path / "main.bin"
    other = tmp_path / "other.bin"
    np.arange(100, dtype=np.uint16).tofile(main)
    np.arange(50, dtype=np.uint16).tofile(other)
    train = np.memmap(main, dtype=np.uint16, mode="r")[10:]
    out = parse_phase2(f"{main}:0.25, {other}:0.75", str(main), train)
    assert [w for _, w in out] == [0.25, 0.75]
    assert out[0][0] is train  # --tokens maps to its training part, without the validation slice
    assert len(out[1][0]) == 50 and int(out[1][0][7]) == 7


@pytest.mark.parametrize("spec", ["a.bin", "a.bin:0", "a.bin:-1", ":0.5"])
def test_phase2_rejects_bad_items(spec):
    with pytest.raises(SystemExit):
        parse_phase2(spec, "main.bin", None)
