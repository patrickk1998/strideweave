from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any, final

from ...layout import Layout
from ...tensor import Tensor
from ..base import Carrier
from ..dtype import DType
from ..move.await_result import AwaitMove
from ..operation_capability import DependentCarrier, OperationCapability
from .projection import AwaitProjection
from .residency import AwaitResidency, ResidencyPolicy
from .values import ResidencyPlan, TileSet

class _TileState:
    location: str
    valid: bool
    dirty: bool

class _WorkLease:
    plan: ResidencyPlan
    protected: set[tuple[int, ...]]
    promotion: AwaitResidency
    pins_released: bool
    finished: bool

def _decode(index: int, shape: tuple[int, ...]) -> tuple[int, ...]: ...
def _encode(coordinate: tuple[int, ...], shape: tuple[int, ...]) -> int: ...
def _coordinates(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]: ...
def _dense_layout(shape: tuple[int, ...]) -> Layout: ...
def _owner_access(owned: Any) -> AbstractContextManager[Carrier]: ...
def _validate_full_tensor(
    carrier: TiledEvictable, value: object, name: str
) -> Tensor: ...

@final
class TiledEvictable(DependentCarrier):
    def __init__(
        self,
        primary: Carrier,
        secondary: Carrier,
        grid_shape: Iterable[int],
        tile_shape: Iterable[int],
        *,
        residency_policy: ResidencyPolicy | None = ...,
    ) -> None: ...
    @property
    def primary(self) -> Carrier: ...
    @property
    def secondary(self) -> Carrier: ...
    @property
    def grid_shape(self) -> tuple[int, ...]: ...
    @property
    def tile_shape(self) -> tuple[int, ...]: ...
    @property
    def logical_shape(self) -> tuple[int, ...]: ...
    @property
    def residency_policy(self) -> ResidencyPolicy | None: ...
    @property
    def primary_tiles(self) -> TileSet: ...
    @property
    def secondary_tiles(self) -> TileSet: ...
    @property
    def valid_tiles(self) -> TileSet: ...
    @property
    def dirty_tiles(self) -> TileSet: ...
    def tile_state(self, coordinate: tuple[int, ...]) -> _TileState: ...
    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = ...,
        dtype: DType | None = ...,
    ) -> TiledEvictable: ...
    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = ...,
        dtype: DType | None = ...,
        empty: bool = ...,
    ) -> TiledEvictable: ...
    def promote_async(self, tiles: object) -> AwaitResidency: ...
    def evict_async(self, tiles: object) -> AwaitResidency: ...
    def promote(self, tiles: object) -> None: ...
    def evict(self, tiles: object) -> None: ...
    def project(
        self,
        tensor: object,
        selection: object,
        destination: object | None = ...,
    ) -> AwaitProjection: ...
    def scatter(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        compact: object,
        template: object,
        selection: object,
        *,
        reduction: str = ...,
    ) -> AwaitMove[Tensor]: ...
    def _promote_async(self, tiles: object) -> AwaitResidency: ...
    def _evict_async(self, tiles: object) -> AwaitResidency: ...
    def _implicit_zero_like(self) -> TiledEvictable: ...
    def _tile_and_local(self, index: int) -> tuple[tuple[int, ...], int]: ...
    def _install_primary_tiles(
        self,
        values: dict[tuple[int, ...], list[object]],
        *,
        dirty: bool,
        increment_version: bool,
    ) -> None: ...
    def _plan_residency(self, required: TileSet, purpose: str) -> ResidencyPlan: ...
    def _pin_tiles_for_work(
        self, required: TileSet, purpose: str
    ) -> AbstractContextManager[ResidencyPlan]: ...
    def _composite_capabilities(self) -> tuple[OperationCapability, ...]: ...
    def _composite_result_carriers(self, invocation: object) -> tuple[type, ...]: ...

__all__ = ["TiledEvictable"]
