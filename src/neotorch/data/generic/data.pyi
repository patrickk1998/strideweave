from collections.abc import Iterable
from os import PathLike
from typing import Any

from ..base import Data
from ..dtype import DataType

class Generic(Data):
    def __init__(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType = DataType.Floating,
    ) -> None: ...
    def size(self) -> int: ...
    def type(self) -> DataType: ...
    def get_value(self, index: int) -> Any: ...
    def is_mutable(self) -> bool: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> Generic: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
    @staticmethod
    def dispatch_op(operation_name: str) -> Any: ...

class GenericEvictable(Generic):
    path: str
    def __init__(
        self,
        values: Iterable[Any],
        path: str | PathLike[str],
        *,
        mutable: bool = True,
        dtype: DataType = DataType.Floating,
    ) -> None: ...
    def size(self) -> int: ...
    def is_evictable(self) -> bool: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> GenericEvictable: ...
