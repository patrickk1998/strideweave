import errno
import gc
import struct
import sys
from array import array
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

_block_device = import_module("strideweave._block_device")

CAPACITY = 4096
LOGICAL_BLOCK_SIZE = 512
PHYSICAL_BLOCK_SIZE = 4096


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    path = tmp_path / "block-device-test-storage"
    path.write_bytes(bytes(CAPACITY))
    return path


def open_test_handle(
    storage_path: Path,
    *,
    state: Any | None = None,
    capacity: int = CAPACITY,
    logical_block_size: int = LOGICAL_BLOCK_SIZE,
    physical_block_size: int = PHYSICAL_BLOCK_SIZE,
) -> tuple[Any, Any]:
    if state is None:
        state = _block_device._TestIoState()
    handle = _block_device._open_block_device_for_test(
        str(storage_path),
        capacity,
        logical_block_size,
        physical_block_size,
        state,
    )
    return handle, state


def test_private_handle_freezes_validated_geometry(storage_path: Path) -> None:
    handle, state = open_test_handle(storage_path)

    assert handle.capacity_bytes == CAPACITY
    assert handle.logical_block_size == LOGICAL_BLOCK_SIZE
    assert handle.physical_block_size == PHYSICAL_BLOCK_SIZE
    assert handle.is_open()
    assert state.opened_descriptor >= 0

    close_result = handle.close()

    assert close_result.succeeded
    assert close_result.error_number == 0
    assert not handle.is_open()
    assert state.close_attempts == 1
    assert state.last_closed_descriptor == state.opened_descriptor
    assert state.descriptor_was_closed


@pytest.mark.parametrize(
    ("capacity", "logical_block_size", "physical_block_size", "message"),
    [
        (0, 512, 4096, "capacity must be positive"),
        (4096, 0, 4096, "logical block size must be positive"),
        (4096, 512, 0, "physical block size must be positive"),
        (4097, 512, 4096, "capacity must be a multiple"),
        (4096, 512, 768, "physical block size must be a multiple"),
    ],
)
def test_invalid_test_geometry_closes_the_opened_descriptor(
    storage_path: Path,
    capacity: int,
    logical_block_size: int,
    physical_block_size: int,
    message: str,
) -> None:
    state = _block_device._TestIoState()

    with pytest.raises(ValueError, match=message):
        open_test_handle(
            storage_path,
            state=state,
            capacity=capacity,
            logical_block_size=logical_block_size,
            physical_block_size=physical_block_size,
        )

    assert state.opened_descriptor >= 0
    assert state.close_attempts == 1
    assert state.last_closed_descriptor == state.opened_descriptor
    assert state.descriptor_was_closed


def test_production_open_rejects_a_regular_file_without_writing_it(
    storage_path: Path,
) -> None:
    before = storage_path.read_bytes()

    with pytest.raises(ValueError, match=r"not a .* block device"):
        _block_device._open_block_device(str(storage_path))

    assert storage_path.read_bytes() == before


def test_embedded_nul_path_is_rejected_before_native_open(storage_path: Path) -> None:
    before = storage_path.read_bytes()
    embedded_nul_path = f"{storage_path}\0ignored-suffix"
    state = _block_device._TestIoState()

    with pytest.raises(ValueError, match="embedded NUL"):
        _block_device._open_block_device(embedded_nul_path)
    with pytest.raises(ValueError, match="embedded NUL"):
        _block_device._open_block_device_for_test(
            embedded_nul_path,
            CAPACITY,
            LOGICAL_BLOCK_SIZE,
            PHYSICAL_BLOCK_SIZE,
            state,
        )

    assert state.opened_descriptor == -1
    assert state.close_attempts == 0
    assert state.read_attempts == 0
    assert state.write_attempts == 0
    assert storage_path.read_bytes() == before


def test_production_open_reports_an_os_error_for_a_missing_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-device"

    with pytest.raises(OSError, match="No such file or directory") as error:
        _block_device._open_block_device(str(missing))

    assert error.value.errno == errno.ENOENT
    assert error.value.filename == str(missing)


def test_zero_byte_transfer_accepts_null_at_the_end(storage_path: Path) -> None:
    handle, state = open_test_handle(storage_path)

    read_result = handle.read_into(0, CAPACITY, 0)
    write_result = handle.write_from(0, CAPACITY, 0)

    assert read_result.status == "complete"
    assert read_result.completed_bytes == 0
    assert read_result.error_number == 0
    assert write_result.status == "complete"
    assert state.read_attempts == 0
    assert state.write_attempts == 0


def test_transfer_preflight_rejects_null_and_out_of_range_inputs_before_io(
    storage_path: Path,
) -> None:
    handle, state = open_test_handle(storage_path)
    pointer_bits = struct.calcsize("P") * 8

    with pytest.raises(ValueError, match="pointer must be non-null"):
        handle.read_into(0, 0, 1)
    with pytest.raises(ValueError, match="byte_offset must be non-negative"):
        handle.read_into(1, -1, 1)
    with pytest.raises(ValueError, match="byte_count must be non-negative"):
        handle.read_into(1, 0, -1)
    with pytest.raises(ValueError, match="byte range exceeds"):
        handle.read_into(1, CAPACITY, 1)
    with pytest.raises(OverflowError, match="pointer byte endpoint"):
        handle.read_into((1 << pointer_bits) - 1, 0, 1)

    assert state.read_attempts == 0


def test_transfer_preflight_detects_index_endpoint_overflow(
    storage_path: Path,
) -> None:
    state = _block_device._TestIoState()
    handle, _ = open_test_handle(
        storage_path,
        state=state,
        capacity=sys.maxsize,
        logical_block_size=1,
        physical_block_size=1,
    )

    with pytest.raises(OverflowError, match="byte endpoint overflows Index"):
        handle.read_into(1, sys.maxsize, 1)

    assert state.read_attempts == 0


def test_final_in_range_access_and_end_boundary_are_exact(storage_path: Path) -> None:
    handle, state = open_test_handle(storage_path)

    write_result = handle.write_bytes(CAPACITY - 4, b"last")
    read_result, payload = handle.read_bytes(CAPACITY - 4, 4)

    assert write_result.is_complete
    assert write_result.completed_bytes == 4
    assert read_result.is_complete
    assert payload == b"last"
    assert state.write_attempts == 1
    assert state.read_attempts == 1

    with pytest.raises(ValueError, match="byte range exceeds"):
        handle.write_bytes(CAPACITY - 3, b"last")


def test_transfer_retries_interrupts_and_completes_repeated_short_progress(
    storage_path: Path,
) -> None:
    script = [("interrupt", 0), ("progress", 1), ("progress", 2)]
    state = _block_device._TestIoState(
        read_script=script,
        write_script=script,
    )
    handle, _ = open_test_handle(storage_path, state=state)

    write_result = handle.write_bytes(8, b"data")
    read_result, payload = handle.read_bytes(8, 4)

    assert write_result.status == "complete"
    assert write_result.completed_bytes == 4
    assert read_result.status == "complete"
    assert read_result.completed_bytes == 4
    assert payload == b"data"
    assert state.write_attempts == 4
    assert state.read_attempts == 4
    assert state.write_steps_remaining == 0
    assert state.read_steps_remaining == 0


@pytest.mark.parametrize("prefix", [0, 2])
def test_terminal_zero_completion_reports_the_exact_prefix(
    storage_path: Path, prefix: int
) -> None:
    script = [] if prefix == 0 else [("progress", prefix)]
    script.append(("zero", 0))
    state = _block_device._TestIoState(write_script=script)
    handle, _ = open_test_handle(storage_path, state=state)

    result = handle.write_bytes(0, b"abcd")

    assert result.status == "zero_completion"
    assert result.completed_bytes == prefix
    assert result.error_number == 0
    assert storage_path.read_bytes()[:prefix] == b"abcd"[:prefix]


@pytest.mark.parametrize("prefix", [0, 2])
def test_terminal_os_error_reports_errno_and_the_exact_prefix(
    storage_path: Path, prefix: int
) -> None:
    script = [] if prefix == 0 else [("progress", prefix)]
    script.append(("error", errno.EIO))
    state = _block_device._TestIoState(read_script=script)
    handle, _ = open_test_handle(storage_path, state=state)

    result, payload = handle.read_bytes(0, 4)

    assert result.status == "os_error"
    assert result.completed_bytes == prefix
    assert result.error_number == errno.EIO
    assert payload[:prefix] == bytes(prefix)


def test_pointer_methods_transfer_directly_to_and_from_cpu_memory(
    storage_path: Path,
) -> None:
    handle, _ = open_test_handle(storage_path)
    source = array("B", b"pointer-transfer")
    destination = array("B", bytes(len(source)))

    write_result = handle.write_from(source.buffer_info()[0], 32, len(source))
    read_result = handle.read_into(destination.buffer_info()[0], 32, len(destination))

    assert write_result.is_complete
    assert read_result.is_complete
    assert destination.tobytes() == source.tobytes()


def test_zero_fill_completes_in_native_code(storage_path: Path) -> None:
    storage_path.write_bytes(b"x" * CAPACITY)
    state = _block_device._TestIoState(write_script=[("interrupt", 0), ("progress", 3)])
    handle, _ = open_test_handle(storage_path, state=state)

    result = handle.zero_fill(32, 128)

    assert result.is_complete
    assert result.completed_bytes == 128
    assert storage_path.read_bytes()[32:160] == bytes(128)
    assert state.write_attempts == 3


def test_close_is_idempotent_and_never_retries_a_descriptor(storage_path: Path) -> None:
    handle, state = open_test_handle(storage_path)

    first = handle.close()
    second = handle.close()

    assert first.succeeded
    assert second.succeeded
    assert state.close_attempts == 1
    assert state.descriptor_was_closed
    with pytest.raises(RuntimeError, match="handle is closed"):
        handle.read_into(0, 0, 0)


def test_close_error_still_invalidates_the_descriptor_before_returning(
    storage_path: Path,
) -> None:
    state = _block_device._TestIoState(close_error=errno.EIO)
    handle, _ = open_test_handle(storage_path, state=state)

    result = handle.close()
    repeated = handle.close()

    assert not result.succeeded
    assert result.error_number == errno.EIO
    assert repeated.succeeded
    assert not handle.is_open()
    assert state.close_attempts == 1
    assert state.descriptor_was_closed


def test_handle_destruction_closes_exactly_once(storage_path: Path) -> None:
    handle, state = open_test_handle(storage_path)
    opened_descriptor = state.opened_descriptor

    del handle
    gc.collect()

    assert state.close_attempts == 1
    assert state.last_closed_descriptor == opened_descriptor
    assert state.descriptor_was_closed
