from collections.abc import Iterable
from typing import Any

from ..base import Carrier
from ..dtype import DType

class Generic(Carrier):
    def __init__(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType = ...,
    ) -> None: ...
    def size(self) -> int: ...
    def dtype(self) -> DType: ...
    def get_value(self, index: int) -> Any: ...
    def is_mutable(self) -> bool: ...
    def _is_mutable(self) -> bool: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> Generic: ...
    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> Generic: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
    def _dispatch_op(self, operation_name: str) -> Any: ...
