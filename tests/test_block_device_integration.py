"""Real Linux block-device integration, enabled only by one explicit path."""

import gc
import os
import stat
import sys
from pathlib import Path

import pytest

import strideweave as sw

pytestmark = pytest.mark.block_device_integration


def _configured_device() -> tuple[Path, int]:
    raw_path = os.environ.get("STRIDEWEAVE_BLOCK_DEVICE")
    raw_capacity = os.environ.get("STRIDEWEAVE_BLOCK_DEVICE_CAPACITY")
    if raw_path is None or raw_capacity is None:
        pytest.skip("no explicit real block-device path and capacity were supplied")
    path = Path(raw_path)
    if not path.is_absolute():
        raise AssertionError("real block-device path must be absolute")
    return path, int(raw_capacity)


def _cpu_tensor(values: list[float], layout: sw.Layout) -> sw.Tensor:
    carrier = sw.CPU(layout.cosize, dtype=sw.DType.Float32)
    for index, value in enumerate(values):
        carrier[index] = value
    return sw.Tensor(carrier, 0, layout)


def _exercise_device(device: sw.BlockDevice, expected_capacity: int) -> None:
    assert device.capacity_bytes == expected_capacity
    assert device.logical_block_size > 0
    assert device.physical_block_size > 0
    assert expected_capacity % device.logical_block_size == 0
    assert device.physical_block_size % device.logical_block_size == 0

    first = device.allocate(1, empty=True)
    second = device.allocate(1, empty=True)
    assert first.device_offset_bytes == 0
    assert second.device_offset_bytes >= first.extent_bytes
    assert first.extent_bytes % device.allocation_alignment == 0
    assert second.device_offset_bytes % device.allocation_alignment == 0
    first[0] = 1.25
    second[0] = -2.5
    assert first[0] == pytest.approx(1.25)
    assert second[0] == pytest.approx(-2.5)

    first.release()
    replacement = device.allocate(1, empty=True)
    assert replacement.device_offset_bytes == 0
    second.release()
    replacement.release()

    layout = sw.Layout(sw.Shape(3), sw.Stride(2))
    source = _cpu_tensor([1.0, 91.0, 2.0, 92.0, 3.0], layout)
    block = device.allocate(layout.cosize, empty=True)
    on_block = sw.move(source, block)
    result = sw.move(on_block, sw.CPU(layout.cosize, dtype=sw.DType.Float32))
    assert result.layout == layout
    assert [result[index] for index in range(result.size())] == [1.0, 2.0, 3.0]
    assert [result.carrier[index] for index in range(layout.cosize)] == [
        1.0,
        91.0,
        2.0,
        92.0,
        3.0,
    ]

    undersized = device.allocate(1, empty=True)
    too_large = _cpu_tensor([4.0, 5.0], sw.Layout(sw.Shape(2), sw.Stride(1)))
    with pytest.raises(ValueError, match="too small"):
        sw.move(too_large, undersized)
    assert not too_large.carrier.is_released()
    undersized.release()

    final = device.allocate(expected_capacity // 4, empty=True)
    final[final.size() - 1] = 7.5
    assert final[final.size() - 1] == pytest.approx(7.5)
    with pytest.raises(RuntimeError, match="Carrier object exists"):
        device.close()
    assert not device.is_closed()
    final.release()


def test_public_carrier_on_exact_disposable_linux_block_device() -> None:
    path, expected_capacity = _configured_device()
    assert sys.platform.startswith("linux")
    assert os.geteuid() != 0, "the carrier test itself must run unprivileged"
    assert stat.S_ISBLK(path.stat().st_mode)

    device = sw.BlockDevice(path)
    _exercise_device(device, expected_capacity)
    gc.collect()

    device.close()
    assert device.is_closed()
    device.close()
