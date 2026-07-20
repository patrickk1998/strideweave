from collections.abc import Iterable
from typing import Any

from ..base import Carrier
from ..dtype import DType

class CPU(Carrier):
    def __init__(
        self,
        size: int,
        pointer: int | None = None,
        *,
        mutable: bool = True,
        dtype: DType = DType.Float32,
    ) -> None: ...
    def size(self) -> int: ...
    def dtype(self) -> DType: ...
    def get_value(self, index: int) -> float | int: ...
    def is_mutable(self) -> bool: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> CPU: ...
    def empty_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> CPU: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
    def pointer(self) -> int: ...
    def dlpack_info(self) -> dict[str, int]: ...
    def dispatch_op(self, operation_name: str) -> Any: ...
