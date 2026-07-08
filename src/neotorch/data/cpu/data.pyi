from collections.abc import Iterable
from typing import Any

from ..base import Data
from ..dtype import DataType

class CPU(Data):
    def __init__(
        self,
        size: int,
        pointer: int | None = None,
        *,
        mutable: bool = True,
        dtype: DataType = DataType.Float32,
    ) -> None: ...
    def size(self) -> int: ...
    def type(self) -> DataType: ...
    def get_value(self, index: int) -> float | int: ...
    def is_mutable(self) -> bool: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> CPU: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
    def pointer(self) -> int: ...
    @staticmethod
    def dispatch_op(operation_name: str) -> Any: ...
