from collections.abc import Iterable
from os import PathLike
from typing import Any, Never, Self, final

from ..base import Carrier
from ..dtype import DType

class BlockDevice:
    def __init__(self, path: str | PathLike[str]) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def capacity_bytes(self) -> int: ...
    @property
    def logical_block_size(self) -> int: ...
    @property
    def physical_block_size(self) -> int: ...
    @property
    def allocation_alignment(self) -> int: ...
    def is_closed(self) -> bool: ...
    def is_faulted(self) -> bool: ...
    def allocate(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType = ...,
        empty: bool = False,
    ) -> BlockDeviceCarrier: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool: ...

@final
class BlockDeviceCarrier(Carrier):
    def __init__(self, *args: Never, **kwargs: Never) -> Never: ...
    @property
    def device(self) -> BlockDevice: ...
    @property
    def device_offset_bytes(self) -> int: ...
    @property
    def extent_bytes(self) -> int: ...
    def size(self) -> int: ...
    def dtype(self) -> DType: ...
    def get_value(self, index: int) -> float: ...
    def set_value(self, index: int, value: Any) -> None: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> BlockDeviceCarrier: ...
    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> BlockDeviceCarrier: ...
    def _write_from_cpu(
        self,
        source: Any,
        *,
        source_offset: int,
        destination_offset: int,
        element_count: int,
    ) -> None: ...
    def _read_into_cpu(
        self,
        destination: Any,
        *,
        source_offset: int,
        destination_offset: int,
        element_count: int,
    ) -> None: ...
    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None: ...
