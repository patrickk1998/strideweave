from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..base import Data
from ..dtype import DataType

def _session_directory() -> Path: ...

class FileBacked(Data):
    _itemsize: int
    def __init__(
        self,
        filename: str | None = None,
        *,
        mutable: bool = True,
        dtype: DataType = DataType.Floating,
    ) -> None: ...
    @property
    def path(self) -> Path: ...
    def size(self) -> int: ...
    def type(self) -> DataType: ...
    def is_mutable(self) -> bool: ...
    def _is_mutable(self) -> bool: ...
    def get_value(self, index: int) -> Any: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> FileBacked: ...
    def empty_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> FileBacked: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
    def _allocate(self, size: int) -> None: ...
