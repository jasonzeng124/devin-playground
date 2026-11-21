from types import SimpleNamespace

from torch.autograd import DeviceType

from scripts.research.profile_update import _busy_us


def _event(start, end, device=DeviceType.CUDA):
    return SimpleNamespace(time_range=SimpleNamespace(start=start, end=end), device_type=device)


def test_busy_is_the_union_of_device_spans():
    events = [_event(0, 10), _event(5, 12), _event(20, 25), _event(21, 22), _event(0, 100, DeviceType.CPU)]
    assert _busy_us(events) == (17, 4)
    assert _busy_us([]) == (0.0, 0)
