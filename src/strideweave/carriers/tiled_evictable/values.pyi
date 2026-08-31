from collections.abc import Iterable
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TileSet:
    coordinates: tuple[tuple[int, ...], ...]
    def __init__(self, coordinates: Iterable[Iterable[int]] = ...) -> None: ...
    def _validate_grid_shape(self, grid_shape: Iterable[int]) -> None: ...

@dataclass(frozen=True, slots=True)
class TileSelection:
    per_axis: tuple[tuple[int, ...], ...]
    def __init__(self, per_axis: Iterable[Iterable[int]]) -> None: ...
    @property
    def rank(self) -> int: ...
    @property
    def compact_grid_shape(self) -> tuple[int, ...]: ...
    @property
    def coordinates(self) -> tuple[tuple[int, ...], ...]: ...
    @property
    def tile_set(self) -> TileSet: ...
    def _validate_grid_shape(self, grid_shape: Iterable[int]) -> None: ...

@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    promote: TileSet
    retain: TileSet
    evict: TileSet
    def __init__(
        self,
        promote: TileSet | None = ...,
        retain: TileSet | None = ...,
        evict: TileSet | None = ...,
    ) -> None: ...
