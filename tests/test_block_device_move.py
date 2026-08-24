import ctypes
import errno
import struct
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

import strideweave as sw
from strideweave import CPU, DType, Layout, Shape, Stride, Tensor
from strideweave.carriers.block_device.carrier import BlockDevice
from strideweave.carriers.move import (
    BlockDeviceToCpuMoveOperation,
    CpuToBlockDeviceMoveOperation,
    dispatch_move,
)

_block_device = import_module("strideweave._block_device")

CAPACITY = 32 * 1024
LOGICAL_BLOCK_SIZE = 512
PHYSICAL_BLOCK_SIZE = 4096


def make_device(
    tmp_path: Path,
    *,
    state: Any | None = None,
    initial: bytes | None = None,
) -> tuple[BlockDevice, Any, Path]:
    path = tmp_path / "block-move-storage"
    path.parent.mkdir(parents=True, exist_ok=True)
    if initial is None:
        initial = bytes(CAPACITY)
    if len(initial) > CAPACITY:
        raise ValueError("initial test storage exceeds capacity")
    path.write_bytes(initial + bytes(CAPACITY - len(initial)))
    if state is None:
        state = _block_device._TestIoState()
    handle = _block_device._open_block_device_for_test(
        str(path),
        CAPACITY,
        LOGICAL_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE,
        state,
    )
    device = BlockDevice._from_handle_for_test(str(path), handle)  # pyright: ignore[reportAttributeAccessIssue]
    return device, state, path


def make_cpu_carrier(values: list[float], *, size: int | None = None) -> CPU:
    if size is None:
        size = len(values)
    carrier = CPU(size, dtype=DType.Float32)
    for index, value in enumerate(values):
        carrier[index] = value
    return carrier


def tensor(carrier: sw.Carrier, layout: Layout, *, offset: int = 0) -> Tensor:
    return Tensor(carrier, offset, layout)


def flat_layout(size: int) -> Layout:
    return Layout(Shape(size), Stride(1))


def physical_values(carrier: sw.Carrier) -> list[Any]:
    return [carrier[index] for index in range(carrier.size())]


def logical_values(value: Tensor) -> list[Any]:
    return [value[index] for index in range(value.size())]


def cpu_bytes(carrier: CPU, byte_count: int) -> bytes:
    return ctypes.string_at(carrier.pointer(), byte_count)


def test_block_move_operations_are_public_and_registered() -> None:
    assert sw.CpuToBlockDeviceMoveOperation is CpuToBlockDeviceMoveOperation
    assert sw.BlockDeviceToCpuMoveOperation is BlockDeviceToCpuMoveOperation
    assert dispatch_move(CPU, sw.BlockDeviceCarrier) is CpuToBlockDeviceMoveOperation
    assert dispatch_move(sw.BlockDeviceCarrier, CPU) is BlockDeviceToCpuMoveOperation


def test_cpu_block_roundtrip_preserves_a_gapped_physical_span(
    tmp_path: Path,
) -> None:
    device, state, _ = make_device(tmp_path)
    source = make_cpu_carrier([1.0, 91.0, 2.0, 92.0, 3.0])
    layout = Layout(Shape(3), Stride(2))
    source_tensor = tensor(source, layout)
    block = device.allocate(layout.cosize, empty=True)
    block_version = block.version

    on_block = sw.move(source_tensor, block)

    assert on_block.carrier is block
    assert on_block.offset == 0
    assert on_block.layout == layout
    assert on_block.dtype() is DType.Float32
    assert logical_values(on_block) == [1.0, 2.0, 3.0]
    assert physical_values(block) == [1.0, 91.0, 2.0, 92.0, 3.0]
    assert block.version == block_version + 1
    assert source.is_released()

    destination = make_cpu_carrier([-1.0] * layout.cosize)
    destination_version = destination.version
    result = sw.move(on_block, destination)

    assert result.carrier is destination
    assert result.offset == 0
    assert result.layout == layout
    assert logical_values(result) == [1.0, 2.0, 3.0]
    assert physical_values(destination) == [1.0, 91.0, 2.0, 92.0, 3.0]
    assert destination.version == destination_version + 1
    assert block.is_released()
    assert state.write_attempts == 1
    assert state.read_attempts == 9  # eight scalar checks plus one bulk read
    assert not device.is_faulted()


def test_roundtrip_preserves_stride_zero_layout_and_physical_cosize(
    tmp_path: Path,
) -> None:
    device, _, _ = make_device(tmp_path)
    layout = Layout(Shape([4, 3]), Stride([0, 1]))
    source = make_cpu_carrier([2.0, 5.0, 7.0])

    on_block = sw.move(tensor(source, layout), device.allocate(3, empty=True))
    result = sw.move(on_block, CPU(3, dtype=DType.Float32))

    assert result.layout == layout
    assert physical_values(result.carrier) == [2.0, 5.0, 7.0]
    assert [result[i, j] for i in range(4) for j in range(3)] == [
        2.0,
        5.0,
        7.0,
    ] * 4


def test_nonzero_cpu_offset_reads_into_exact_size_block_destination(
    tmp_path: Path,
) -> None:
    device, state, _ = make_device(tmp_path)
    source = make_cpu_carrier([50.0, 51.0, 1.0, 2.0, 3.0, 52.0])
    source_view = tensor(source, flat_layout(3), offset=2)
    block = device.allocate(3, empty=True)

    result = sw.move(source_view, block)

    assert result.offset == 0
    assert result.carrier is block
    assert physical_values(block) == [1.0, 2.0, 3.0]
    assert state.write_attempts == 1


def test_nonzero_block_offset_reads_into_exact_size_cpu_destination(
    tmp_path: Path,
) -> None:
    initial = struct.pack("=6f", 50.0, 51.0, 1.0, 2.0, 3.0, 52.0)
    device, state, _ = make_device(tmp_path, initial=initial)
    block = device.allocate(6, empty=True)
    block_view = tensor(block, flat_layout(3), offset=2)
    destination = CPU(3, dtype=DType.Float32)

    result = sw.move(block_view, destination)

    assert result.offset == 0
    assert result.carrier is destination
    assert physical_values(destination) == [1.0, 2.0, 3.0]
    assert state.read_attempts == 1


def test_bulk_moves_change_only_the_selected_destination_prefix(
    tmp_path: Path,
) -> None:
    device, _, _ = make_device(tmp_path)
    source = make_cpu_carrier([1.0, 2.0, 3.0])
    block = device.allocate(6, empty=True)
    for index in range(6):
        block[index] = 9.0
    block_version = block.version

    on_block = sw.move(tensor(source, flat_layout(3)), block)

    assert physical_values(block) == [1.0, 2.0, 3.0, 9.0, 9.0, 9.0]
    assert block.version == block_version + 1

    destination = make_cpu_carrier([-7.0] * 6)
    destination_version = destination.version
    result = sw.move(on_block, destination)

    assert physical_values(destination) == [1.0, 2.0, 3.0, -7.0, -7.0, -7.0]
    assert destination.version == destination_version + 1
    assert result.layout == flat_layout(3)


@pytest.mark.parametrize("destination_size", [0, 1])
def test_fixed_block_destination_rejects_zero_or_undersized_storage(
    tmp_path: Path,
    destination_size: int,
) -> None:
    device, state, _ = make_device(tmp_path)
    source = make_cpu_carrier([1.0, 2.0])
    source_tensor = tensor(source, flat_layout(2))
    destination = device.allocate(destination_size, empty=True)
    before = (
        destination.size(),
        destination.device_offset_bytes,
        destination.extent_bytes,
        destination.version,
    )

    with pytest.raises(ValueError, match="too small"):
        sw.move(source_tensor, destination)

    assert (
        destination.size(),
        destination.device_offset_bytes,
        destination.extent_bytes,
        destination.version,
    ) == before
    assert not source.is_released()
    assert state.write_attempts == 0
    assert not device.is_faulted()


def test_complete_bulk_transfers_retry_interrupts_and_short_progress(
    tmp_path: Path,
) -> None:
    script = [("interrupt", 0), ("progress", 2), ("progress", 3)]
    state = _block_device._TestIoState(
        read_script=script,
        write_script=script,
    )
    device, _, _ = make_device(tmp_path, state=state)
    source = make_cpu_carrier([1.0, 2.0])

    on_block = sw.move(tensor(source, flat_layout(2)), device.allocate(2, empty=True))
    destination = CPU(2, dtype=DType.Float32)
    result = sw.move(on_block, destination)

    assert physical_values(destination) == [1.0, 2.0]
    assert result.carrier is destination
    assert state.write_attempts == 4
    assert state.read_attempts == 4
    assert state.write_steps_remaining == 0
    assert state.read_steps_remaining == 0


@pytest.mark.parametrize(
    ("terminal", "prefix", "error_type", "message"),
    [
        ("error", 0, OSError, "Input/output error"),
        ("error", 2, OSError, "Input/output error"),
        ("zero", 0, RuntimeError, "completed 0 of 8 bytes"),
        ("zero", 2, RuntimeError, "completed 2 of 8 bytes"),
    ],
)
def test_cpu_to_block_terminal_failure_faults_and_keeps_source(
    tmp_path: Path,
    terminal: str,
    prefix: int,
    error_type: type[BaseException],
    message: str,
) -> None:
    script = [] if prefix == 0 else [("progress", prefix)]
    script.append((terminal, errno.EIO if terminal == "error" else 0))
    state = _block_device._TestIoState(write_script=script)
    initial = b"\xa5" * CAPACITY
    device, _, path = make_device(tmp_path, state=state, initial=initial)
    source = make_cpu_carrier([1.25, -2.5])
    source_bytes = cpu_bytes(source, 8)
    source_version = source.version
    destination = device.allocate(2, empty=True)

    with pytest.raises(error_type, match=message):
        sw.move(tensor(source, flat_layout(2)), destination)

    stored = path.read_bytes()[:8]
    assert stored[:prefix] == source_bytes[:prefix]
    assert stored[prefix:] == initial[prefix:8]
    assert destination.version == 1
    assert destination.size() == 2
    assert not destination.is_released()
    assert source.version == source_version
    assert not source.is_released()
    assert device.is_faulted()


@pytest.mark.parametrize(
    ("terminal", "prefix", "error_type", "message"),
    [
        ("error", 0, OSError, "Input/output error"),
        ("error", 2, OSError, "Input/output error"),
        ("zero", 0, RuntimeError, "completed 0 of 8 bytes"),
        ("zero", 2, RuntimeError, "completed 2 of 8 bytes"),
    ],
)
def test_block_to_cpu_terminal_failure_exposes_only_completed_prefix(
    tmp_path: Path,
    terminal: str,
    prefix: int,
    error_type: type[BaseException],
    message: str,
) -> None:
    source_bytes = struct.pack("=2f", 1.25, -2.5)
    script = [] if prefix == 0 else [("progress", prefix)]
    script.append((terminal, errno.EIO if terminal == "error" else 0))
    state = _block_device._TestIoState(read_script=script)
    device, _, _ = make_device(tmp_path, state=state, initial=source_bytes)
    source = device.allocate(2, empty=True)
    destination = make_cpu_carrier([7.0, 8.0])
    before_bytes = cpu_bytes(destination, 8)
    destination_version = destination.version

    with pytest.raises(error_type, match=message):
        sw.move(tensor(source, flat_layout(2)), destination)

    after_bytes = cpu_bytes(destination, 8)
    assert after_bytes[:prefix] == source_bytes[:prefix]
    assert after_bytes[prefix:] == before_bytes[prefix:]
    assert destination.version == destination_version + 1
    assert not source.is_released()
    assert source.size() == 2
    assert device.is_faulted()


def test_faulted_block_storage_is_rejected_before_another_transfer(
    tmp_path: Path,
) -> None:
    state = _block_device._TestIoState(write_script=[("error", errno.EIO)])
    device, _, _ = make_device(tmp_path, state=state)
    destination = device.allocate(1, empty=True)
    first_source = make_cpu_carrier([1.0])
    with pytest.raises(OSError, match="Input/output error"):
        sw.move(tensor(first_source, flat_layout(1)), destination)
    attempts = state.write_attempts
    destination_version = destination.version
    later_source = make_cpu_carrier([2.0])

    with pytest.raises(RuntimeError, match="faulted"):
        sw.move(tensor(later_source, flat_layout(1)), destination)

    assert state.write_attempts == attempts
    assert destination.version == destination_version
    assert not later_source.is_released()


def test_faulted_block_source_is_rejected_before_mutating_another_cpu(
    tmp_path: Path,
) -> None:
    state = _block_device._TestIoState(read_script=[("error", errno.EIO)])
    device, _, _ = make_device(tmp_path, state=state, initial=struct.pack("=f", 1.0))
    source = device.allocate(1, empty=True)
    source_tensor = tensor(source, flat_layout(1))
    first_destination = CPU(1, dtype=DType.Float32)
    with pytest.raises(OSError, match="Input/output error"):
        sw.move(source_tensor, first_destination)
    attempts = state.read_attempts
    later_destination = make_cpu_carrier([9.0])
    later_version = later_destination.version

    with pytest.raises(RuntimeError, match="faulted"):
        sw.move(source_tensor, later_destination)

    assert state.read_attempts == attempts
    assert later_destination[0] == 9.0
    assert later_destination.version == later_version
    assert not source.is_released()


def test_exact_operations_revalidate_ownership_before_transfer(
    tmp_path: Path,
) -> None:
    device, state, _ = make_device(tmp_path)
    block_destination = device.allocate(1, empty=True)
    destination_token = block_destination._claim_ownership()
    cpu_source = make_cpu_carrier([1.0])
    with pytest.raises(RuntimeError, match="must be mutable"):
        sw.move(tensor(cpu_source, flat_layout(1)), block_destination)
    block_destination._relinquish_ownership(destination_token)

    block_source = device.allocate(1, empty=True)
    source_token = block_source._claim_ownership()
    with pytest.raises(RuntimeError, match="owned by another carrier"):
        sw.move(tensor(block_source, flat_layout(1)), CPU(1, dtype=DType.Float32))
    block_source._relinquish_ownership(source_token)

    assert state.read_attempts == 0
    assert state.write_attempts == 0
    assert not device.is_faulted()


def test_wrong_dtype_fails_before_block_io_or_version_change(tmp_path: Path) -> None:
    device, state, _ = make_device(tmp_path)
    destination = device.allocate(1, empty=True)
    source = CPU(1, dtype=DType.Int32)
    source[0] = 1

    with pytest.raises(TypeError, match="dtype must match"):
        CpuToBlockDeviceMoveOperation().forward(
            tensor(source, flat_layout(1)), destination
        )

    assert destination.version == 0
    assert state.write_attempts == 0
    assert not source.is_released()


def test_concrete_block_operations_reject_wrong_pinned_classes_before_policy(
    tmp_path: Path,
) -> None:
    device, state, _ = make_device(tmp_path)
    first_block = device.allocate(1, empty=True)
    second_block = device.allocate(1, empty=True)

    with pytest.raises(TypeError, match="requires a CPU source"):
        CpuToBlockDeviceMoveOperation().forward(
            tensor(first_block, flat_layout(1)), second_block
        )
    with pytest.raises(TypeError, match="requires a BlockDeviceCarrier source"):
        BlockDeviceToCpuMoveOperation().forward(
            tensor(make_cpu_carrier([1.0]), flat_layout(1)),
            CPU(1, dtype=DType.Float32),
        )

    assert state.read_attempts == 0
    assert state.write_attempts == 0
    assert first_block.version == 0
    assert second_block.version == 0


def test_exact_move_lifecycle_and_cpu_ownership_failures_precede_transfer(
    tmp_path: Path,
) -> None:
    device, state, _ = make_device(tmp_path)

    released_cpu_source = make_cpu_carrier([1.0])
    released_cpu_tensor = tensor(released_cpu_source, flat_layout(1))
    released_cpu_source.release()
    block_destination = device.allocate(1, empty=True)
    with pytest.raises(RuntimeError, match="tensor carrier is released"):
        sw.move(released_cpu_tensor, block_destination)

    released_block_source = device.allocate(1, empty=True)
    released_block_tensor = tensor(released_block_source, flat_layout(1))
    released_block_source.release()
    cpu_destination = CPU(1, dtype=DType.Float32)
    with pytest.raises(RuntimeError, match="tensor carrier is released"):
        sw.move(released_block_tensor, cpu_destination)

    live_block_source = device.allocate(1, empty=True)
    released_cpu_destination = CPU(1, dtype=DType.Float32)
    released_cpu_destination.release()
    with pytest.raises(RuntimeError, match="destination carrier is released"):
        sw.move(tensor(live_block_source, flat_layout(1)), released_cpu_destination)

    immutable_cpu_destination = CPU(1, mutable=False, dtype=DType.Float32)
    with pytest.raises(RuntimeError, match="destination carrier must be mutable"):
        sw.move(tensor(live_block_source, flat_layout(1)), immutable_cpu_destination)

    owned_cpu_source = make_cpu_carrier([1.0])
    source_hierarchy = sw.Evictable(
        owned_cpu_source, sw.Generic([0.0], dtype=DType.Float32)
    )
    with pytest.raises(RuntimeError, match="owned by another carrier"):
        sw.move(tensor(owned_cpu_source, flat_layout(1)), block_destination)
    assert source_hierarchy[0] == 1.0

    owned_cpu_destination = CPU(1, dtype=DType.Float32)
    destination_hierarchy = sw.Evictable(
        owned_cpu_destination, sw.Generic([0.0], dtype=DType.Float32)
    )
    with pytest.raises(RuntimeError, match="destination carrier must be mutable"):
        sw.move(tensor(live_block_source, flat_layout(1)), owned_cpu_destination)
    assert destination_hierarchy[0] == 0.0

    assert state.read_attempts == 0
    assert state.write_attempts == 0
    assert cpu_destination.version == 0
    assert not device.is_faulted()


def test_bulk_operations_never_call_block_scalar_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, state, _ = make_device(tmp_path)
    block = device.allocate(2, empty=True)
    source = make_cpu_carrier([1.0, 2.0])

    def scalar_access_is_forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("bulk movement reached scalar block access")

    monkeypatch.setattr(sw.BlockDeviceCarrier, "get_value", scalar_access_is_forbidden)
    monkeypatch.setattr(sw.BlockDeviceCarrier, "set_value", scalar_access_is_forbidden)

    on_block = sw.move(tensor(source, flat_layout(2)), block)
    destination = CPU(2, dtype=DType.Float32)
    result = sw.move(on_block, destination)

    assert physical_values(destination) == [1.0, 2.0]
    assert result.carrier is destination
    assert state.write_attempts == 1
    assert state.read_attempts == 1


def test_cpu_block_roundtrip_backpropagates_through_released_prototypes(
    tmp_path: Path,
) -> None:
    device, _, _ = make_device(tmp_path)
    original = make_cpu_carrier([1.0, 2.0, 3.0])
    source = tensor(original, flat_layout(3))

    on_block = sw.move(source, device.allocate(3, empty=True))
    result = sw.move(on_block, CPU(3, dtype=DType.Float32))
    gradient = tensor(make_cpu_carrier([10.0, 11.0, 12.0]), flat_layout(3))
    result.backward(gradient)

    assert source.grad is not None
    assert type(source.grad.carrier) is CPU
    assert logical_values(source.grad) == [10.0, 11.0, 12.0]
    assert result.layout == source.layout
