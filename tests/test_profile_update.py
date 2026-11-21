from types import SimpleNamespace

import pytest
from torch.autograd import DeviceType
from torch.nn.attention import SDPBackend

from scripts.research.profile_update import _backends, _busy_us


def _event(start, end, device=DeviceType.CUDA):
    return SimpleNamespace(time_range=SimpleNamespace(start=start, end=end), device_type=device)


def test_busy_is_the_union_of_device_spans():
    events = [_event(0, 10), _event(5, 12), _event(20, 25), _event(21, 22), _event(0, 100, DeviceType.CPU)]
    assert _busy_us(events) == (17, 4)
    assert _busy_us([]) == (0.0, 0)


def test_sdpa_backends_parse_in_order_and_reject_unknown_names():
    assert _backends("") == []
    assert _backends("efficient") == [SDPBackend.EFFICIENT_ATTENTION]
    assert _backends("flash, efficient") == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    with pytest.raises(SystemExit):
        _backends("cudnn,fast")
