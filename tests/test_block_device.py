import errno
import gc
import struct
import threading
import weakref
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

import strideweave as sw
import strideweave.operation as operation
from strideweave.carriers.block_device.carrier import BlockDevice
from strideweave.carriers.operation_policy import resolve_operation_plan

_block_device = import_module("strideweave._block_device")

CAPACITY = 16 * 1024
LOGICAL_BLOCK_SIZE = 512
PHYSICAL_BLOCK_SIZE = 4096


@dataclass(frozen=True)
class Outcome:
    status: str = "complete"
    completed_bytes: int = 0
    error_number: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class CloseOutcome:
    error_number: int = 0

    @property
    def succeeded(self) -> bool:
        return self.error_number == 0


class MemoryHandle:
    def __init__(self, capacity: int = CAPACITY, *, close_error: int = 0) -> None:
        self.capacity_bytes = capacity
        self.logical_block_size = LOGICAL_BLOCK_SIZE
        self.physical_block_size = PHYSICAL_BLOCK_SIZE
        self.data = bytearray(capacity)
        self.open = True
        self.close_error = close_error
        self.read_calls: list[tuple[int, int]] = []
        self.write_calls: list[tuple[int, bytes]] = []
        self.zero_calls: list[tuple[int, int]] = []
        self.before_read: Any = None
        self.before_write: Any = None

    def is_open(self) -> bool:
        return self.open

    def read_bytes(self, byte_offset: int, byte_count: int) -> tuple[Outcome, bytes]:
        if self.before_read is not None:
            self.before_read()
        self.read_calls.append((byte_offset, byte_count))
        return Outcome(completed_bytes=byte_count), bytes(
            self.data[byte_offset : byte_offset + byte_count]
        )

    def write_bytes(self, byte_offset: int, values: bytes) -> Outcome:
        if self.before_write is not None:
            self.before_write()
        self.write_calls.append((byte_offset, values))
        self.data[byte_offset : byte_offset + len(values)] = values
        return Outcome(completed_bytes=len(values))

    def zero_fill(self, byte_offset: int, byte_count: int) -> Outcome:
        self.zero_calls.append((byte_offset, byte_count))
        self.data[byte_offset : byte_offset + byte_count] = bytes(byte_count)
        return Outcome(completed_bytes=byte_count)

    def close(self) -> CloseOutcome:
        self.open = False
        return CloseOutcome(self.close_error)


def make_memory_device(
    *, capacity: int = CAPACITY, close_error: int = 0
) -> tuple[BlockDevice, MemoryHandle]:
    handle = MemoryHandle(capacity, close_error=close_error)
    return BlockDevice._from_handle_for_test("test-device", handle), handle  # pyright: ignore[reportAttributeAccessIssue]


def make_native_device(
    tmp_path: Path,
    *,
    capacity: int = CAPACITY,
    state: Any | None = None,
) -> tuple[BlockDevice, Any, Path]:
    path = tmp_path / "block-device-carrier-storage"
    path.write_bytes(bytes(capacity))
    if state is None:
        state = _block_device._TestIoState()
    handle = _block_device._open_block_device_for_test(
        str(path),
        capacity,
        LOGICAL_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE,
        state,
    )
    return BlockDevice._from_handle_for_test(str(path), handle), state, path  # pyright: ignore[reportAttributeAccessIssue]


def test_public_exports_and_direct_construction_boundary() -> None:
    assert sw.BlockDevice is BlockDevice
    assert sw.carriers.BlockDevice is BlockDevice
    assert sw.BlockDeviceCarrier is sw.carriers.BlockDeviceCarrier
    assert {"BlockDevice", "BlockDeviceCarrier"} <= set(sw.__all__)
    assert {"BlockDevice", "BlockDeviceCarrier"} <= set(vars(sw.carriers)["__all__"])
    assert {
        "BlockDeviceToCpuMoveOperation",
        "CpuToBlockDeviceMoveOperation",
    } <= set(operation.__all__)

    with pytest.raises(TypeError, match=r"use BlockDevice\.allocate"):
        sw.BlockDeviceCarrier()


def test_path_protocol_is_preserved_without_normalization(monkeypatch) -> None:
    observed: list[str] = []
    handle = MemoryHandle()

    class RelativePath:
        def __fspath__(self) -> str:
            return "devices/model-tier"

    def open_handle(path: str) -> MemoryHandle:
        observed.append(path)
        return handle

    monkeypatch.setattr(
        "strideweave.carriers.block_device.carrier._open_native_handle", open_handle
    )

    device = BlockDevice(RelativePath())

    assert device.path == "devices/model-tier"
    assert observed == ["devices/model-tier"]


def test_embedded_nul_path_fails_before_open(monkeypatch) -> None:
    observed: list[str] = []

    class EmbeddedNulPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            return "devices/model-tier\0ignored-suffix"

    def open_handle(path: str) -> MemoryHandle:
        observed.append(path)
        return MemoryHandle()

    monkeypatch.setattr(
        "strideweave.carriers.block_device.carrier._open_native_handle", open_handle
    )
    path_like = EmbeddedNulPath()

    with pytest.raises(ValueError, match="embedded NUL"):
        BlockDevice("devices/model-tier\0ignored-suffix")
    with pytest.raises(ValueError, match="embedded NUL"):
        BlockDevice(path_like)

    assert path_like.calls == 1
    assert observed == []


def test_invalid_path_protocol_fails_before_open(monkeypatch) -> None:
    opened = False

    def open_handle(path: str) -> MemoryHandle:
        nonlocal opened
        opened = True
        return MemoryHandle()

    monkeypatch.setattr(
        "strideweave.carriers.block_device.carrier._open_native_handle", open_handle
    )

    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"device"

    class ExplodingPath:
        def __fspath__(self) -> str:
            raise LookupError("path callback failed")

    with pytest.raises(TypeError, match="string filesystem representation"):
        BlockDevice(BytesPath())  # pyright: ignore[reportArgumentType]
    with pytest.raises(LookupError, match="path callback failed"):
        BlockDevice(ExplodingPath())
    assert not opened


def test_geometry_and_lifecycle_queries_are_side_effect_free() -> None:
    device, handle = make_memory_device()

    assert device.path == "test-device"
    assert device.capacity_bytes == CAPACITY
    assert device.logical_block_size == LOGICAL_BLOCK_SIZE
    assert device.physical_block_size == PHYSICAL_BLOCK_SIZE
    assert device.allocation_alignment == PHYSICAL_BLOCK_SIZE
    assert not device.is_closed()
    assert not device.is_faulted()
    assert handle.read_calls == []
    assert handle.write_calls == []


def test_allocations_are_lowest_first_aligned_and_non_overlapping(
    tmp_path: Path,
) -> None:
    device, _, _ = make_native_device(tmp_path)

    first = device.allocate(1)
    second = device.allocate(2)

    assert first.device is device
    assert first.device_offset_bytes == 0
    assert first.extent_bytes == PHYSICAL_BLOCK_SIZE
    assert second.device_offset_bytes == PHYSICAL_BLOCK_SIZE
    assert second.extent_bytes == PHYSICAL_BLOCK_SIZE
    assert first.size() == 1
    assert second.size() == 2
    assert first.device_offset_bytes + first.extent_bytes <= second.device_offset_bytes


def test_released_ranges_are_reused_lowest_first_and_coalesced(
    tmp_path: Path,
) -> None:
    device, _, _ = make_native_device(tmp_path)
    first = device.allocate(1, empty=True)
    second = device.allocate(1, empty=True)
    third = device.allocate(1, empty=True)

    first.release()
    replacement = device.allocate(1, empty=True)
    assert replacement.device_offset_bytes == 0

    replacement.release()
    second.release()
    combined = device.allocate(PHYSICAL_BLOCK_SIZE // 2, empty=True)

    assert combined.extent_bytes == 2 * PHYSICAL_BLOCK_SIZE
    assert combined.device_offset_bytes == 0
    assert third.device_offset_bytes == 2 * PHYSICAL_BLOCK_SIZE


def test_capacity_exhaustion_is_atomic_and_recoverable(tmp_path: Path) -> None:
    device, state, _ = make_native_device(tmp_path, capacity=2 * PHYSICAL_BLOCK_SIZE)
    first = device.allocate(1, empty=True)
    second = device.allocate(1, empty=True)

    with pytest.raises(MemoryError, match="no aligned extent"):
        device.allocate(1, empty=True)

    assert first.device_offset_bytes == 0
    assert second.device_offset_bytes == PHYSICAL_BLOCK_SIZE
    assert state.write_attempts == 0

    first.release()
    replacement = device.allocate(1, empty=True)
    assert replacement.device_offset_bytes == 0


def test_zero_size_allocations_reserve_nothing_and_remain_fixed(tmp_path: Path) -> None:
    device, state, _ = make_native_device(tmp_path)

    first = device.allocate(0)
    second = device.allocate(0, empty=True)
    positive = device.allocate(1, empty=True)

    assert (first.size(), first.device_offset_bytes, first.extent_bytes) == (0, 0, 0)
    assert (second.size(), second.device_offset_bytes, second.extent_bytes) == (0, 0, 0)
    assert positive.device_offset_bytes == 0
    assert state.write_attempts == 0


def test_allocation_validates_exact_flags_dtype_and_checked_size(
    tmp_path: Path,
) -> None:
    device, state, _ = make_native_device(tmp_path)

    for name in ("mutable", "empty"):
        arguments = {name: 1}
        with pytest.raises(TypeError, match=f"{name} must be a bool"):
            device.allocate(1, **arguments)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="dtype must be a DType"):
        device.allocate(1, dtype="Float32")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"dtype must be DType\.Float32"):
        device.allocate(1, dtype=sw.DType.Int32)
    with pytest.raises(ValueError, match="size must be non-negative"):
        device.allocate(-1)
    with pytest.raises(OverflowError, match="byte length overflows"):
        device.allocate((1 << 63) // 4)

    assert state.write_attempts == 0


def test_float32_is_the_exact_complete_block_storage_dtype_set(
    tmp_path: Path,
) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(0, empty=True)

    for dtype in sw.DType.registered():
        expected = dtype is sw.DType.Float32
        assert carrier.supports_storage_dtype(dtype) is expected
        if expected:
            allocated = device.allocate(0, dtype=dtype, empty=True)
            assert allocated.dtype() is dtype
            continue
        with pytest.raises(ValueError, match="BlockDeviceCarrier"):
            device.allocate(0, dtype=dtype, empty=True)

    with pytest.raises(TypeError, match="storage dtype must be a DType"):
        carrier.supports_storage_dtype("Float32")  # pyright: ignore[reportArgumentType]


def test_size_index_callback_reenters_before_outer_allocation(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    nested: list[sw.BlockDeviceCarrier] = []

    class ReentrantSize:
        def __index__(self) -> int:
            nested.append(device.allocate(1, empty=True))
            return 1

    outer = device.allocate(ReentrantSize(), empty=True)  # type: ignore[arg-type]

    assert nested[0].device_offset_bytes == 0
    assert outer.device_offset_bytes == PHYSICAL_BLOCK_SIZE


def test_initialized_and_new_like_storage_is_binary32(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    initialized = device.allocate(3)

    assert [initialized[index] for index in range(3)] == [0.0, 0.0, 0.0]

    values = initialized.new_like([1.1, None, -2.5])
    expected = struct.unpack("=f", struct.pack("=f", 1.1))[0]
    assert values[0] == expected
    assert values[1] == 0.0
    assert values[2] == -2.5
    assert values.version == 0


def test_scalar_access_normalizes_values_and_versions_at_transfer_start(
    tmp_path: Path,
) -> None:
    device, state, _ = make_native_device(tmp_path)
    carrier = device.allocate(2)
    initial_attempts = state.write_attempts

    carrier[0] = 1.1
    carrier.set_value(1, -2)

    assert carrier[0] == struct.unpack("=f", struct.pack("=f", 1.1))[0]
    assert carrier.get_value(1) == -2.0
    assert carrier.version == 2
    assert state.write_attempts == initial_attempts + 2


def test_scalar_preflight_rejects_bounds_and_mutability_before_io(
    tmp_path: Path,
) -> None:
    device, state, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, mutable=False, empty=True)

    with pytest.raises(IndexError, match="out of range"):
        carrier.get_value(-1)
    with pytest.raises(IndexError, match="out of range"):
        carrier.get_value(1)
    with pytest.raises(RuntimeError, match="not mutable"):
        carrier.set_value(0, 1.0)

    assert carrier.version == 0
    assert state.read_attempts == 0
    assert state.write_attempts == 0


def test_float_conversion_reenters_before_write_effect(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, empty=True)
    nested: list[sw.BlockDeviceCarrier] = []

    class ReentrantFloat:
        def __float__(self) -> float:
            nested.append(device.allocate(0))
            return 3.5

    carrier.set_value(0, ReentrantFloat())

    assert carrier[0] == 3.5
    assert nested[0].size() == 0


def test_new_like_materialization_and_cleanup_reenter_outside_the_lock(
    tmp_path: Path,
) -> None:
    device, _, _ = make_native_device(tmp_path)
    prototype = device.allocate(0)
    cleanup_allocations: list[sw.BlockDeviceCarrier] = []

    class ReentrantValue:
        def __float__(self) -> float:
            return 2.0

        def __del__(self) -> None:
            cleanup_allocations.append(device.allocate(0))

    def values():
        yield ReentrantValue()

    result = prototype.new_like(values())

    assert result[0] == 2.0
    assert len(cleanup_allocations) == 1


def test_terminal_scalar_write_faults_shared_device_and_retains_version(
    tmp_path: Path,
) -> None:
    state = _block_device._TestIoState(write_script=[("error", errno.EIO)])
    device, _, _ = make_native_device(tmp_path, state=state)
    destination = device.allocate(1, empty=True)
    sibling = device.allocate(1, empty=True)

    with pytest.raises(OSError, match="Input/output error") as error:
        destination.set_value(0, 1.0)

    assert error.value.errno == errno.EIO
    assert destination.version == 1
    assert device.is_faulted()
    assert destination.size() == 1
    assert sibling.size() == 1
    with pytest.raises(RuntimeError, match="faulted"):
        sibling.get_value(0)
    with pytest.raises(RuntimeError, match="faulted"):
        device.allocate(1)


def test_terminal_zero_completion_faults_and_reports_progress(tmp_path: Path) -> None:
    state = _block_device._TestIoState(write_script=[("progress", 2), ("zero", 0)])
    device, _, _ = make_native_device(tmp_path, state=state)
    carrier = device.allocate(1, empty=True)

    with pytest.raises(RuntimeError, match="completed 2 of 4 bytes"):
        carrier.set_value(0, 1.0)

    assert carrier.version == 1
    assert device.is_faulted()


def test_terminal_initialization_failure_returns_no_carrier_and_faults(
    tmp_path: Path,
) -> None:
    state = _block_device._TestIoState(write_script=[("error", errno.EIO)])
    device, _, _ = make_native_device(tmp_path, state=state)

    with pytest.raises(OSError, match="Input/output error"):
        device.allocate(1)

    assert device.is_faulted()
    with pytest.raises(RuntimeError, match="faulted"):
        device.allocate(1, empty=True)


def test_release_is_idempotent_zeros_metrics_and_keeps_factory_behavior(
    tmp_path: Path,
) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(2, mutable=False, empty=True)

    carrier.release()
    carrier.release()

    assert carrier.is_released()
    assert carrier.size() == 0
    assert carrier.device_offset_bytes == 0
    assert carrier.extent_bytes == 0
    assert carrier.device is device
    assert carrier.dtype() is sw.DType.Float32
    assert not carrier.is_mutable()
    assert carrier.supports_storage_dtype(sw.DType.Float32)
    assert not carrier.supports_storage_dtype(sw.DType.Int32)
    with pytest.raises(RuntimeError, match="released"):
        carrier.get_value(0)

    replacement = carrier.allocate_like(1)
    from_values = carrier.new_like([3.0])
    assert replacement.device_offset_bytes == 0
    assert from_values[0] == 3.0


def test_faulted_release_remains_available_and_performs_no_io(tmp_path: Path) -> None:
    state = _block_device._TestIoState(write_script=[("error", errno.EIO)])
    device, _, _ = make_native_device(tmp_path, state=state)
    carrier = device.allocate(1, empty=True)
    with pytest.raises(OSError, match="Input/output error"):
        carrier.set_value(0, 1.0)
    attempts = state.write_attempts

    carrier.release()

    assert carrier.is_released()
    assert carrier.size() == 0
    assert state.write_attempts == attempts


def test_destroying_an_unreleased_carrier_retires_its_extent(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, empty=True)
    reference = weakref.ref(carrier)
    assert carrier.device_offset_bytes == 0

    del carrier
    gc.collect()

    assert reference() is None
    replacement = device.allocate(1, empty=True)
    assert replacement.device_offset_bytes == 0


def test_close_refuses_live_and_released_carrier_objects(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, empty=True)

    with pytest.raises(RuntimeError, match="Carrier object exists"):
        device.close()
    assert not device.is_closed()

    carrier.release()
    with pytest.raises(RuntimeError, match="Carrier object exists"):
        device.close()

    del carrier
    gc.collect()
    device.close()
    device.close()
    assert device.is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        device.allocate(1)


def test_carrier_holds_device_strongly_until_object_destruction(tmp_path: Path) -> None:
    device, _, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, empty=True)
    device_reference = weakref.ref(device)

    del device
    gc.collect()

    assert device_reference() is carrier.device
    assert not carrier.device.is_closed()


def test_context_manager_preserves_and_chains_exceptions() -> None:
    device, _ = make_memory_device()

    with pytest.raises(ValueError, match="managed failure"):
        with device:
            raise ValueError("managed failure")
    assert device.is_closed()

    blocked, _ = make_memory_device()
    carrier = blocked.allocate(0)
    with pytest.raises(RuntimeError, match="Carrier object exists") as close_error:
        with blocked:
            raise LookupError("managed failure")
    assert isinstance(close_error.value.__context__, LookupError)
    assert not blocked.is_closed()
    carrier.release()


def test_close_error_invalidates_and_marks_the_device_closed() -> None:
    device, handle = make_memory_device(close_error=errno.EIO)

    with pytest.raises(OSError, match="Input/output error") as error:
        device.close()

    assert error.value.errno == errno.EIO
    assert device.is_closed()
    assert not handle.is_open()
    device.close()


def test_storage_only_dispatch_capabilities_scatter_and_dlpack(tmp_path: Path) -> None:
    device, state, _ = make_native_device(tmp_path)
    carrier = device.allocate(1, empty=True)
    plan = resolve_operation_plan("relu", sw.DType.Float32)

    assert carrier.operation_capabilities() == ()
    assert not carrier.supports_operation_plan(plan)
    with pytest.raises(sw.UnsupportedOperationPlan):
        carrier.require_operation_plan(plan)
    for operation_name in ("add", "view"):
        with pytest.raises(NotImplementedError, match=operation_name):
            carrier.dispatch_op(operation_name)
    with pytest.raises(TypeError):
        carrier.dispatch_op(1)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="scatter"):
        carrier.scatter(None, None, None)
    with pytest.raises(BufferError, match="not supported"):
        carrier.dlpack_info()

    assert state.read_attempts == 0
    assert state.write_attempts == 0


def test_disjoint_operations_are_serialized_by_one_device_lock() -> None:
    device, handle = make_memory_device()
    first = device.allocate(1, empty=True)
    second = device.allocate(1, empty=True)
    entered = threading.Event()
    continue_first = threading.Event()
    second_done = threading.Event()
    calls = 0

    def block_first_write() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert continue_first.wait(timeout=2)

    handle.before_write = block_first_write

    first_thread = threading.Thread(target=first.set_value, args=(0, 1.0))

    def write_second() -> None:
        second.set_value(0, 2.0)
        second_done.set()

    second_thread = threading.Thread(target=write_second)
    first_thread.start()
    assert entered.wait(timeout=2)
    second_thread.start()
    assert not second_done.wait(timeout=0.05)
    continue_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first[0] == 1.0
    assert second[0] == 2.0


def test_release_waits_for_active_transfer_before_extent_reuse() -> None:
    device, handle = make_memory_device()
    carrier = device.allocate(1, empty=True)
    entered = threading.Event()
    continue_write = threading.Event()
    release_done = threading.Event()

    def block_write() -> None:
        entered.set()
        assert continue_write.wait(timeout=2)

    handle.before_write = block_write
    writer = threading.Thread(target=carrier.set_value, args=(0, 1.0))

    def release() -> None:
        carrier.release()
        release_done.set()

    releaser = threading.Thread(target=release)
    writer.start()
    assert entered.wait(timeout=2)
    releaser.start()
    assert not release_done.wait(timeout=0.05)
    continue_write.set()
    writer.join(timeout=2)
    releaser.join(timeout=2)

    assert carrier.is_released()
    replacement = device.allocate(1, empty=True)
    assert replacement.device_offset_bytes == 0
