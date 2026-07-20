from __future__ import annotations

import atexit
import shutil
import struct
import tempfile
import weakref
from collections.abc import Iterable
from operator import index as operator_index
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..base import Carrier
from ..dtype import DType

_STRUCT_FORMATS = {
    DType.Floating: "d",
    DType.Float32: "f",
    DType.Int32: "i",
}

_session_directory_path: Path | None = None


def _session_directory() -> Path:
    global _session_directory_path
    if _session_directory_path is None:
        path = Path(tempfile.gettempdir()) / f".strideweave-filebacked-{uuid4().hex}"
        path.mkdir(parents=True)
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        _session_directory_path = path
    return _session_directory_path


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _validate_file_backed_dtype(dtype: DType) -> DType:
    if not isinstance(dtype, DType):
        raise TypeError("FileBacked dtype must be a DType")
    if dtype not in _STRUCT_FORMATS:
        raise ValueError(
            "FileBacked dtype must be DType.Floating, DType.Float32, or DType.Int32"
        )
    return dtype


class FileBacked(Carrier):
    """Carrier storage backed by a raw binary file on disk.

    Values live in a file inside a hidden per-process directory under the
    system temporary directory. The file is allocated when the carrier is
    created and removed when the carrier is deleted; the whole directory
    is removed when the interpreter exits. FileBacked supports no dispatched
    tensor operations — tensors backed by it can only be read, written, and
    moved to another carrier with ``strideweave.move``.

    Args:
        filename: Bare file name inside the hidden session directory, or
            ``None`` to generate a random name.
        mutable: Whether values may be written after creation.
        dtype: Numeric element type; one of ``DType.Floating``,
            ``DType.Float32``, or ``DType.Int32``.

    Examples:
        >>> from strideweave import FileBacked
        >>> carrier = FileBacked("weights.bin")
        >>> carrier.size()
        0
    """

    def __init__(
        self,
        filename: str | None = None,
        *,
        mutable: bool = True,
        dtype: DType = DType.Floating,
    ):
        super().__init__()
        self._mutable = bool(mutable)
        self._dtype = _validate_file_backed_dtype(dtype)
        self._format = _STRUCT_FORMATS[self._dtype]
        self._itemsize = struct.calcsize(self._format)

        if filename is None:
            filename = f"{uuid4().hex}.bin"
        if Path(filename).name != filename:
            raise ValueError("FileBacked filename must be a bare file name")
        self._path = _session_directory() / filename
        with open(self._path, "xb"):
            pass
        self._finalizer = weakref.finalize(self, _remove_file, self._path)

    @property
    def path(self) -> Path:
        return self._path

    def _require_file(self) -> Path:
        if self.is_released():
            raise RuntimeError("Carrier is released")
        return self._path

    def size(self) -> int:
        if self.is_released():
            return 0
        return self._path.stat().st_size // self._itemsize

    def dtype(self) -> DType:
        return self._dtype

    def _is_mutable(self) -> bool:
        return self._mutable

    def get_value(self, index: int) -> Any:
        path = self._require_file()
        normalized = self._validate_index(index)
        with open(path, "rb") as file:
            file.seek(normalized * self._itemsize)
            (value,) = struct.unpack(self._format, file.read(self._itemsize))
        return value

    def set_value(self, index: int, value: Any) -> None:
        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        self._write_value(index, value)
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> FileBacked:
        if type(self) is not FileBacked:
            raise NotImplementedError(
                "FileBacked carrier factory only supports exact FileBacked carriers"
            )
        result = FileBacked(
            mutable=mutable, dtype=self._dtype if dtype is None else dtype
        )
        materialized = list(values)
        result._allocate(len(materialized))
        for index, value in enumerate(materialized):
            if value is None:
                continue
            result._write_value(index, value)
        return result

    def empty_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> FileBacked:
        normalized_size = operator_index(size)
        if normalized_size < 0:
            raise ValueError("FileBacked allocation size must be non-negative")
        result = FileBacked(
            mutable=mutable, dtype=self._dtype if dtype is None else dtype
        )
        result._allocate(normalized_size)
        return result

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("FileBacked carriers do not support scatter")

    def _allocate(self, size: int) -> None:
        if size < 0:
            raise ValueError("FileBacked allocation size must be non-negative")
        path = self._require_file()
        with open(path, "r+b") as file:
            file.truncate(size * self._itemsize)

    def _release(self) -> None:
        self._finalizer()

    def _validate_index(self, index: int) -> int:
        normalized = operator_index(index)
        if normalized < 0 or normalized >= self.size():
            raise IndexError("Carrier index out of range")
        return normalized

    def _write_value(self, index: int, value: Any) -> None:
        path = self._require_file()
        normalized = self._validate_index(index)
        if self._dtype is DType.Int32:
            packed = struct.pack(self._format, operator_index(value))
        else:
            packed = struct.pack(self._format, value)
        with open(path, "r+b") as file:
            file.seek(normalized * self._itemsize)
            file.write(packed)


__all__ = [
    "FileBacked",
]
