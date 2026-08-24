"""Synchronous Float32 carrier storage over one caller-supplied block device."""

from __future__ import annotations

import os
import struct
import threading
import weakref
from collections.abc import Iterable
from importlib import import_module
from operator import index as operator_index
from typing import Any, Protocol, Self, cast, final

from ..base import Carrier, reject_carrier_subclass
from ..dtype import DType, accepts_storage_dtype, validate_storage_dtype

_native = import_module("strideweave._block_device")

_FLOAT32_BYTES = 4
_INDEX_MAX = (1 << 63) - 1
_POINTER_MAX = (1 << (struct.calcsize("P") * 8)) - 1
_BLOCK_DEVICE_DTYPES = (DType.Float32,)


class _TransferResult(Protocol):
    status: str
    completed_bytes: int
    error_number: int
    is_complete: bool


class _CloseResult(Protocol):
    error_number: int
    succeeded: bool


class _DeviceHandle(Protocol):
    capacity_bytes: int
    logical_block_size: int
    physical_block_size: int

    def is_open(self) -> bool: ...

    def read_bytes(
        self, byte_offset: int, byte_count: int
    ) -> tuple[_TransferResult, bytes]: ...

    def write_bytes(self, byte_offset: int, values: bytes) -> _TransferResult: ...

    def read_into(
        self, destination_pointer: int, byte_offset: int, byte_count: int
    ) -> _TransferResult: ...

    def write_from(
        self, source_pointer: int, byte_offset: int, byte_count: int
    ) -> _TransferResult: ...

    def zero_fill(self, byte_offset: int, byte_count: int) -> _TransferResult: ...

    def close(self) -> _CloseResult: ...


def _normalize_path(path: str | os.PathLike[str]) -> str:
    normalized = os.fspath(path)
    if not isinstance(normalized, str):
        raise TypeError("BlockDevice path must have a string filesystem representation")
    if "\0" in normalized:
        raise ValueError("BlockDevice path must not contain embedded NUL characters")
    return normalized


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return cast(bool, value)


def _block_device_dtype(dtype: object) -> DType:
    return validate_storage_dtype(
        dtype, carrier="BlockDeviceCarrier", accepted=_BLOCK_DEVICE_DTYPES
    )


def _normalize_size(size: object) -> int:
    normalized = operator_index(cast(Any, size))
    if normalized < 0:
        raise ValueError("BlockDevice allocation size must be non-negative")
    if normalized > _INDEX_MAX // _FLOAT32_BYTES:
        raise OverflowError("BlockDevice allocation byte length overflows Index")
    return normalized


def _extent_length(size: int, alignment: int) -> tuple[int, int]:
    payload_bytes = size * _FLOAT32_BYTES
    if payload_bytes == 0:
        return 0, 0
    if payload_bytes > _INDEX_MAX - (alignment - 1):
        raise OverflowError("BlockDevice aligned extent length overflows Index")
    extent_bytes = ((payload_bytes + alignment - 1) // alignment) * alignment
    if extent_bytes > _INDEX_MAX:
        raise OverflowError("BlockDevice aligned extent length overflows Index")
    return payload_bytes, extent_bytes


def _checked_slot_window(
    offset: int, count: int, size: int, *, endpoint: str
) -> tuple[int, int, int]:
    if offset < 0 or count < 0:
        raise ValueError(f"{endpoint} slot window must be non-negative")
    if offset > _INDEX_MAX - count:
        raise OverflowError(f"{endpoint} slot endpoint overflows Index")
    end = offset + count
    if end > size:
        raise ValueError(f"{endpoint} slot window exceeds carrier storage")
    if offset > _INDEX_MAX // _FLOAT32_BYTES:
        raise OverflowError(f"{endpoint} byte offset overflows Index")
    if count > _INDEX_MAX // _FLOAT32_BYTES:
        raise OverflowError(f"{endpoint} byte count overflows Index")
    if end > _INDEX_MAX // _FLOAT32_BYTES:
        raise OverflowError(f"{endpoint} byte endpoint overflows Index")
    return (
        offset * _FLOAT32_BYTES,
        count * _FLOAT32_BYTES,
        end * _FLOAT32_BYTES,
    )


def _checked_cpu_pointer(carrier: Any, offset: int, count: int) -> tuple[int, int]:
    byte_offset, byte_count, _ = _checked_slot_window(
        offset, count, carrier.size(), endpoint="CPU"
    )
    pointer = carrier.pointer()
    if pointer < 0 or pointer > _POINTER_MAX:
        raise OverflowError("CPU base pointer is not representable as uintptr_t")
    if byte_offset > _POINTER_MAX - pointer:
        raise OverflowError("CPU pointer byte offset overflows uintptr_t")
    selected = pointer + byte_offset
    if byte_count > _POINTER_MAX - selected:
        raise OverflowError("CPU pointer byte endpoint overflows uintptr_t")
    return selected, byte_count


def _normalize_float32(value: object) -> bytes:
    if value is None:
        return bytes(_FLOAT32_BYTES)
    return struct.pack("=f", value)


def _open_native_handle(path: str) -> _DeviceHandle:
    return cast(_DeviceHandle, _native._open_block_device(path))


class BlockDevice:
    """Process-local allocation arena over one explicit block-device node.

    The device path is opened exactly as supplied and its capacity and block
    sizes are frozen for this object's lifetime. Allocations from one arena are
    aligned, non-overlapping, and ordered through one synchronous lock. Sharing
    means reusing this same object; independently opening the same node creates
    an independent arena and is not coordinated.

    Args:
        path: String or filesystem-path object naming the exact supported Linux
            or buffered macOS block-device node to open for reading and writing.

    Examples:
        >>> import strideweave as sw
        >>> # device = sw.BlockDevice("/dev/loop0")
        >>> # carrier = device.allocate(1024)
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        normalized_path = _normalize_path(path)
        handle = _open_native_handle(normalized_path)
        self._initialize(normalized_path, handle)

    def _initialize(self, path: str, handle: _DeviceHandle) -> None:
        self._path = path
        self._handle = handle
        self._capacity_bytes = handle.capacity_bytes
        self._logical_block_size = handle.logical_block_size
        self._physical_block_size = handle.physical_block_size
        self._allocation_alignment = max(
            self._logical_block_size, self._physical_block_size
        )
        self._lock = threading.Lock()
        self._faulted = False
        self._closed = False
        self._free_ranges: list[tuple[int, int]] = [(0, self._capacity_bytes)]
        self._allocations: dict[int, tuple[int, int]] = {}
        self._carrier_refs: dict[int, weakref.ReferenceType[BlockDeviceCarrier]] = {}
        self._next_allocation_token = 1

    @classmethod
    def _from_handle_for_test(cls, path: str, handle: _DeviceHandle) -> BlockDevice:
        """Build a device over the private deterministic native test seam."""
        device = cls.__new__(cls)
        device._initialize(path, handle)
        return device

    @property
    def path(self) -> str:
        """Return the exact filesystem string supplied at construction."""
        return self._path

    @property
    def capacity_bytes(self) -> int:
        """Return the immutable byte capacity reported by the opened device."""
        return self._capacity_bytes

    @property
    def logical_block_size(self) -> int:
        """Return the immutable logical block size in bytes."""
        return self._logical_block_size

    @property
    def physical_block_size(self) -> int:
        """Return the immutable physical block size in bytes."""
        return self._physical_block_size

    @property
    def allocation_alignment(self) -> int:
        """Return the byte alignment used for every positive allocation."""
        return self._allocation_alignment

    def is_closed(self) -> bool:
        """Return whether this arena has closed its native device handle."""
        return self._closed

    def is_faulted(self) -> bool:
        """Return whether a terminal transfer failure made storage inaccessible."""
        return self._faulted

    def allocate(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType = DType.Float32,
        empty: bool = False,
    ) -> BlockDeviceCarrier:
        """Allocate one fixed-size Float32 carrier from this device.

        Args:
            size: Number of Float32 physical slots in the carrier payload.
            mutable: Whether public writes are permitted.
            dtype: Storage dtype, which must be the identical ``DType.Float32``.
            empty: Whether initialization may be skipped.

        Returns:
            A fresh carrier whose padded extent overlaps no other live allocation.

        Examples:
            >>> import strideweave as sw
            >>> # carrier = sw.BlockDevice("/dev/loop0").allocate(4)
        """
        normalized_size = _normalize_size(size)
        normalized_mutable = _exact_bool(mutable, "mutable")
        normalized_dtype = _block_device_dtype(dtype)
        normalized_empty = _exact_bool(empty, "empty")
        return self._allocate_staged(
            normalized_size,
            mutable=normalized_mutable,
            dtype=normalized_dtype,
            empty=normalized_empty,
            payload=None,
        )

    def _new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: object,
        dtype: object,
    ) -> BlockDeviceCarrier:
        normalized_mutable = _exact_bool(mutable, "mutable")
        normalized_dtype = _block_device_dtype(dtype)
        materialized = list(values)
        encoded_values = [_normalize_float32(value) for value in materialized]
        payload = b"".join(encoded_values)
        return self._allocate_staged(
            _normalize_size(len(materialized)),
            mutable=normalized_mutable,
            dtype=normalized_dtype,
            empty=False,
            payload=payload,
        )

    def _allocate_staged(
        self,
        size: int,
        *,
        mutable: bool,
        dtype: DType,
        empty: bool,
        payload: bytes | None,
    ) -> BlockDeviceCarrier:
        payload_bytes, extent_bytes = _extent_length(size, self._allocation_alignment)
        if payload is not None and len(payload) != payload_bytes:
            raise RuntimeError("BlockDevice staged payload length is inconsistent")

        carrier = BlockDeviceCarrier._create_pending(
            self, size=size, mutable=mutable, dtype=dtype
        )
        transfer_failure: tuple[_TransferResult, str, int] | None = None
        with self._lock:
            self._require_available_locked()
            offset = self._reserve_locked(extent_bytes)
            try:
                result: _TransferResult | None = None
                if payload is not None:
                    result = self._handle.write_bytes(offset, payload)
                elif not empty:
                    result = self._handle.zero_fill(offset, payload_bytes)
                if result is not None and not result.is_complete:
                    self._faulted = True
                    transfer_failure = (result, "initialization", payload_bytes)
                    self._free_locked(offset, extent_bytes)
                else:
                    self._activate_carrier_locked(carrier, offset, extent_bytes)
            except BaseException:
                self._free_locked(offset, extent_bytes)
                raise

        if transfer_failure is not None:
            result, operation, requested = transfer_failure
            raise self._transfer_exception(result, operation, requested)
        return carrier

    def _require_available_locked(self) -> None:
        if self._closed:
            raise RuntimeError("BlockDevice is closed")
        if self._faulted:
            raise RuntimeError("BlockDevice is faulted")

    def _reserve_locked(self, extent_bytes: int) -> int:
        if extent_bytes == 0:
            return 0
        alignment = self._allocation_alignment
        for index, (start, length) in enumerate(self._free_ranges):
            if start > _INDEX_MAX - (alignment - 1):
                continue
            candidate = ((start + alignment - 1) // alignment) * alignment
            if candidate > _INDEX_MAX - extent_bytes:
                continue
            endpoint = candidate + extent_bytes
            range_endpoint = start + length
            if endpoint > range_endpoint:
                continue

            replacement: list[tuple[int, int]] = []
            if candidate > start:
                replacement.append((start, candidate - start))
            if endpoint < range_endpoint:
                replacement.append((endpoint, range_endpoint - endpoint))
            self._free_ranges[index : index + 1] = replacement
            return candidate
        raise MemoryError("BlockDevice has no aligned extent large enough")

    def _free_locked(self, offset: int, extent_bytes: int) -> None:
        if extent_bytes == 0:
            return
        ranges = [*self._free_ranges, (offset, extent_bytes)]
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, length in ranges:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous_start, previous_length = merged[-1]
                merged[-1] = (previous_start, previous_length + length)
            else:
                merged.append((start, length))
        self._free_ranges = merged

    def _activate_carrier_locked(
        self, carrier: BlockDeviceCarrier, offset: int, extent_bytes: int
    ) -> None:
        token = self._next_allocation_token
        self._next_allocation_token += 1
        finalizer: weakref.finalize | None = None
        try:
            finalizer = weakref.finalize(carrier, self._carrier_collected, token)
            self._allocations[token] = (offset, extent_bytes)
            self._carrier_refs[token] = weakref.ref(carrier)
            carrier._activate(token, offset, extent_bytes, finalizer)
        except BaseException:
            if finalizer is not None:
                finalizer.detach()
            self._allocations.pop(token, None)
            self._carrier_refs.pop(token, None)
            raise

    def _carrier_collected(self, token: int) -> None:
        with self._lock:
            allocation = self._allocations.pop(token, None)
            self._carrier_refs.pop(token, None)
            if allocation is not None:
                self._free_locked(*allocation)

    def _retire_carrier(self, carrier: BlockDeviceCarrier) -> None:
        with self._lock:
            token = carrier._allocation_token
            reference = self._carrier_refs.get(token)
            if reference is None or reference() is not carrier:
                raise RuntimeError("BlockDeviceCarrier allocation identity is invalid")
            allocation = self._allocations.pop(token, None)
            if allocation is not None:
                self._free_locked(*allocation)
            carrier._mark_extent_released()

    def _transfer_exception(
        self, result: _TransferResult, operation: str, requested: int
    ) -> BaseException:
        if result.status == "os_error":
            return OSError(
                result.error_number, os.strerror(result.error_number), self.path
            )
        if result.status == "zero_completion":
            return RuntimeError(
                f"incomplete block-device {operation}: completed "
                f"{result.completed_bytes} of {requested} bytes"
            )
        return RuntimeError(f"unknown block-device {operation} transfer outcome")

    def close(self) -> None:
        """Close this device after every carrier object has been destroyed.

        Returns:
            ``None``. Repeated successful closes are idempotent.

        Raises:
            RuntimeError: If any live or released carrier object still exists.
            OSError: If the non-retried native close reports an operating-system
                error after invalidating its descriptor.

        Examples:
            >>> import strideweave as sw
            >>> # device = sw.BlockDevice("/dev/loop0")
            >>> # device.close()
        """
        close_error = 0
        with self._lock:
            if self._closed:
                return
            if self._carrier_refs:
                raise RuntimeError(
                    "BlockDevice cannot close while a BlockDeviceCarrier object exists"
                )
            result = self._handle.close()
            self._closed = True
            close_error = result.error_number
        if close_error:
            raise OSError(close_error, os.strerror(close_error), self.path)

    def __enter__(self) -> Self:
        """Return this device for a managed block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Always close the managed device and preserve ordinary exception context."""
        del exc_type, exc_value, traceback
        self.close()
        return False


@final
class BlockDeviceCarrier(Carrier):
    """Fixed-size Float32 carrier allocated by :class:`BlockDevice`.

    This storage-only carrier is public for type checks but has no direct
    construction form. Use ``BlockDevice.allocate`` or a carrier factory.
    It performs complete synchronous scalar access and advertises no compute
    capability.
    """

    _device: BlockDevice
    _size: int
    _mutable: bool
    _dtype: DType
    _device_offset_bytes: int
    _extent_bytes: int
    _allocation_token: int
    _allocation_live: bool
    _object_finalizer: weakref.finalize | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        reject_carrier_subclass("BlockDeviceCarrier")

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "BlockDeviceCarrier cannot be constructed directly; use "
            "BlockDevice.allocate"
        )

    @classmethod
    def _create_pending(
        cls, device: BlockDevice, *, size: int, mutable: bool, dtype: DType
    ) -> BlockDeviceCarrier:
        carrier = cls.__new__(cls)
        Carrier.__init__(carrier)
        carrier._device = device
        carrier._size = size
        carrier._mutable = mutable
        carrier._dtype = dtype
        carrier._device_offset_bytes = 0
        carrier._extent_bytes = 0
        carrier._allocation_token = 0
        carrier._allocation_live = False
        carrier._object_finalizer = None
        return carrier

    def _activate(
        self,
        token: int,
        offset: int,
        extent_bytes: int,
        finalizer: weakref.finalize,
    ) -> None:
        self._allocation_token = token
        self._device_offset_bytes = offset
        self._extent_bytes = extent_bytes
        self._allocation_live = True
        self._object_finalizer = finalizer

    def _mark_extent_released(self) -> None:
        self._allocation_live = False
        self._size = 0
        self._device_offset_bytes = 0
        self._extent_bytes = 0

    @property
    def device(self) -> BlockDevice:
        """Return the arena that created this carrier."""
        return self._device

    @property
    def device_offset_bytes(self) -> int:
        """Return the live allocation's aligned device byte offset, or zero."""
        return self._device_offset_bytes

    @property
    def extent_bytes(self) -> int:
        """Return the live padded extent length in bytes, or zero."""
        return self._extent_bytes

    def size(self) -> int:
        return self._size

    def dtype(self) -> DType:
        return self._dtype

    def _is_mutable(self) -> bool:
        return self._mutable

    def _supports_storage_dtype(self, dtype: DType) -> bool:
        return accepts_storage_dtype(dtype, _BLOCK_DEVICE_DTYPES)

    def _require_storage_locked(self) -> None:
        if self.is_released() or not self._allocation_live:
            raise RuntimeError("BlockDeviceCarrier is released")
        self._device._require_available_locked()

    def _normalized_index(self, index: object) -> int:
        normalized = operator_index(cast(Any, index))
        if normalized < 0 or normalized >= self._size:
            raise IndexError("Carrier index out of range")
        return normalized

    def _checked_block_window_locked(self, offset: int, count: int) -> tuple[int, int]:
        byte_offset, byte_count, byte_end = _checked_slot_window(
            offset, count, self._size, endpoint="block-device"
        )
        payload_bytes = self._size * _FLOAT32_BYTES
        if byte_end > payload_bytes:
            raise ValueError("block-device byte window exceeds carrier payload")
        if byte_end > self._extent_bytes:
            raise ValueError("block-device byte window exceeds reserved extent")
        if self._device_offset_bytes > _INDEX_MAX - byte_offset:
            raise OverflowError("absolute block-device byte offset overflows Index")
        absolute_offset = self._device_offset_bytes + byte_offset
        if self._device_offset_bytes > _INDEX_MAX - byte_end:
            raise OverflowError("absolute block-device byte endpoint overflows Index")
        absolute_end = self._device_offset_bytes + byte_end
        if absolute_end > self._device._capacity_bytes:
            raise ValueError("block-device byte window exceeds device capacity")
        if self._device_offset_bytes > _INDEX_MAX - payload_bytes:
            raise OverflowError(
                "absolute block-device payload endpoint overflows Index"
            )
        if absolute_end > self._device_offset_bytes + payload_bytes:
            raise ValueError("block-device byte window exceeds absolute payload")
        if self._device_offset_bytes > _INDEX_MAX - self._extent_bytes:
            raise OverflowError("absolute block-device extent endpoint overflows Index")
        if absolute_end > self._device_offset_bytes + self._extent_bytes:
            raise ValueError("block-device byte window exceeds absolute extent")
        return absolute_offset, byte_count

    def _require_move_source_locked(self) -> None:
        self._require_storage_locked()
        if self.is_owned() and not self._has_owner_access():
            raise RuntimeError(
                "tensor carrier is owned by another carrier object and cannot be "
                "moved directly"
            )

    def _write_from_cpu(
        self,
        source: Any,
        *,
        source_offset: int,
        destination_offset: int,
        element_count: int,
    ) -> None:
        transfer_failure: tuple[_TransferResult, str, int] | None = None
        with self._device._lock:
            self._require_storage_locked()
            if not self.is_mutable():
                raise RuntimeError("destination carrier must be mutable")
            if source.is_released():
                raise RuntimeError("tensor carrier is released")
            if source.is_owned() and not source._has_owner_access():
                raise RuntimeError(
                    "tensor carrier is owned by another carrier object and cannot be "
                    "moved directly"
                )
            if source.dtype() is not DType.Float32 or self._dtype is not DType.Float32:
                raise TypeError("CPU/block-device movement requires DType.Float32")
            source_pointer, source_bytes = _checked_cpu_pointer(
                source, source_offset, element_count
            )
            destination_byte_offset, destination_bytes = (
                self._checked_block_window_locked(destination_offset, element_count)
            )
            if source_bytes != destination_bytes:
                raise RuntimeError("CPU/block-device transfer lengths disagree")
            self._increment_version()
            result = self._device._handle.write_from(
                source_pointer, destination_byte_offset, destination_bytes
            )
            if not result.is_complete:
                self._device._faulted = True
                transfer_failure = (result, "write", destination_bytes)
        if transfer_failure is not None:
            result, operation, requested = transfer_failure
            raise self._device._transfer_exception(result, operation, requested)

    def _read_into_cpu(
        self,
        destination: Any,
        *,
        source_offset: int,
        destination_offset: int,
        element_count: int,
    ) -> None:
        transfer_failure: tuple[_TransferResult, str, int] | None = None
        with self._device._lock:
            self._require_move_source_locked()
            if destination.is_released():
                raise RuntimeError("destination carrier is released")
            if not destination.is_mutable():
                raise RuntimeError("destination carrier must be mutable")
            if (
                self._dtype is not DType.Float32
                or destination.dtype() is not DType.Float32
            ):
                raise TypeError("CPU/block-device movement requires DType.Float32")
            source_byte_offset, source_bytes = self._checked_block_window_locked(
                source_offset, element_count
            )
            destination_pointer, destination_bytes = _checked_cpu_pointer(
                destination, destination_offset, element_count
            )
            if source_bytes != destination_bytes:
                raise RuntimeError("CPU/block-device transfer lengths disagree")
            destination._increment_version()
            result = self._device._handle.read_into(
                destination_pointer, source_byte_offset, source_bytes
            )
            if not result.is_complete:
                self._device._faulted = True
                transfer_failure = (result, "read", source_bytes)
        if transfer_failure is not None:
            result, operation, requested = transfer_failure
            raise self._device._transfer_exception(result, operation, requested)

    def get_value(self, index: int) -> float:
        normalized = operator_index(index)
        transfer_failure: tuple[_TransferResult, str, int] | None = None
        payload = b""
        with self._device._lock:
            self._require_storage_locked()
            if normalized < 0 or normalized >= self._size:
                raise IndexError("Carrier index out of range")
            byte_offset = self._device_offset_bytes + normalized * _FLOAT32_BYTES
            result, payload = self._device._handle.read_bytes(
                byte_offset, _FLOAT32_BYTES
            )
            if not result.is_complete:
                self._device._faulted = True
                transfer_failure = (result, "read", _FLOAT32_BYTES)
        if transfer_failure is not None:
            result, operation, requested = transfer_failure
            raise self._device._transfer_exception(result, operation, requested)
        return cast(float, struct.unpack("=f", payload)[0])

    def set_value(self, index: int, value: Any) -> None:
        normalized = operator_index(index)
        payload = _normalize_float32(value)
        transfer_failure: tuple[_TransferResult, str, int] | None = None
        with self._device._lock:
            self._require_storage_locked()
            if not self.is_mutable():
                raise RuntimeError("Carrier is not mutable")
            if normalized < 0 or normalized >= self._size:
                raise IndexError("Carrier index out of range")
            byte_offset = self._device_offset_bytes + normalized * _FLOAT32_BYTES
            self._increment_version()
            result = self._device._handle.write_bytes(byte_offset, payload)
            if not result.is_complete:
                self._device._faulted = True
                transfer_failure = (result, "write", _FLOAT32_BYTES)
        if transfer_failure is not None:
            result, operation, requested = transfer_failure
            raise self._device._transfer_exception(result, operation, requested)

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> BlockDeviceCarrier:
        """Allocate and initialize a fresh carrier from staged Float32 values."""
        return self._device._new_like(
            values,
            mutable=mutable,
            dtype=self._dtype if dtype is None else dtype,
        )

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> BlockDeviceCarrier:
        """Allocate a fresh fixed-size extent from the same device."""
        return self._device.allocate(
            size,
            mutable=mutable,
            dtype=self._dtype if dtype is None else dtype,
            empty=empty,
        )

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("BlockDeviceCarrier does not support scatter")

    def _dispatch_op(self, operation_name: str) -> Any:
        raise NotImplementedError(
            f"BlockDeviceCarrier does not support operation {operation_name!r}"
        )

    def _release(self) -> None:
        self._device._retire_carrier(self)


__all__ = [
    "BlockDevice",
    "BlockDeviceCarrier",
]
