"""Immutable tile-selection and residency control values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product


def _materialize_iterable(value: object, name: str) -> tuple[object, ...]:
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a finite iterable") from error


def _normalize_coordinate(
    value: object,
    *,
    name: str,
) -> tuple[int, ...]:
    raw = _materialize_iterable(value, name)
    coordinate: list[int] = []
    for component in raw:
        if type(component) is not int:
            raise TypeError(f"{name} components must be integers")
        if component < 0:
            raise ValueError(f"{name} components must be non-negative")
        coordinate.append(component)
    return tuple(coordinate)


def _normalize_grid_shape(grid_shape: object) -> tuple[int, ...]:
    raw = _materialize_iterable(grid_shape, "tile grid shape")
    shape: list[int] = []
    for extent in raw:
        if type(extent) is not int:
            raise TypeError("tile grid extents must be integers")
        if extent <= 0:
            raise ValueError("tile grid extents must be positive")
        shape.append(extent)
    return tuple(shape)


def _validate_coordinates_against_grid(
    coordinates: tuple[tuple[int, ...], ...],
    *,
    rank: int,
    grid_shape: object,
) -> None:
    shape = _normalize_grid_shape(grid_shape)
    if rank != len(shape):
        raise ValueError(
            f"tile coordinate rank {rank} does not match grid rank {len(shape)}"
        )
    for coordinate in coordinates:
        if any(component >= shape[axis] for axis, component in enumerate(coordinate)):
            raise ValueError(
                f"tile coordinate {coordinate!r} is outside grid shape {shape!r}"
            )


@dataclass(frozen=True, slots=True, init=False)
class TileSet:
    """Capture an immutable arbitrary set of full-rank tile coordinates.

    Coordinates are sorted in StrideWeave first-mode-fastest order. Unlike a
    :class:`TileSelection`, a tile set need not describe a Cartesian product.

    Args:
        coordinates: Finite iterable of same-rank, non-negative integer
            coordinates. The default creates an empty rank-agnostic set.

    Examples:
        >>> import strideweave as sw
        >>> tiles = sw.TileSet([(0, 1), (1, 0), (0, 0)])
        >>> tiles.coordinates
        ((0, 0), (1, 0), (0, 1))
    """

    coordinates: tuple[tuple[int, ...], ...]

    def __init__(self, coordinates: Iterable[Iterable[int]] = ()) -> None:
        raw_coordinates = _materialize_iterable(coordinates, "coordinates")
        normalized: list[tuple[int, ...]] = []
        rank: int | None = None
        seen: set[tuple[int, ...]] = set()

        for index, raw_coordinate in enumerate(raw_coordinates):
            coordinate = _normalize_coordinate(
                raw_coordinate,
                name=f"coordinates[{index}]",
            )
            if rank is None:
                rank = len(coordinate)
            elif len(coordinate) != rank:
                raise ValueError("tile coordinates must all have the same rank")
            if coordinate in seen:
                raise ValueError(f"duplicate tile coordinate {coordinate!r}")
            seen.add(coordinate)
            normalized.append(coordinate)

        object.__setattr__(
            self,
            "coordinates",
            tuple(sorted(normalized, key=lambda coordinate: coordinate[::-1])),
        )

    def _validate_grid_shape(self, grid_shape: Iterable[int]) -> None:
        """Validate this set against one concrete tile grid."""

        if not self.coordinates:
            _normalize_grid_shape(grid_shape)
            return
        _validate_coordinates_against_grid(
            self.coordinates,
            rank=len(self.coordinates[0]),
            grid_shape=grid_shape,
        )


@dataclass(frozen=True, slots=True, init=False)
class TileSelection:
    """Describe an ordered Cartesian product of source tiles.

    Each axis retains caller order. ``coordinates`` enumerates the full product
    in StrideWeave first-mode-fastest order, so axis zero varies fastest.

    Args:
        per_axis: One non-empty finite iterable of unique, non-negative integer
            tile coordinates per selected grid axis.

    Examples:
        >>> import strideweave as sw
        >>> selection = sw.TileSelection(([7, 2], [4, 9]))
        >>> selection.coordinates
        ((7, 4), (2, 4), (7, 9), (2, 9))
    """

    per_axis: tuple[tuple[int, ...], ...]

    def __init__(self, per_axis: Iterable[Iterable[int]]) -> None:
        raw_axes = _materialize_iterable(per_axis, "per_axis")
        if not raw_axes:
            raise ValueError("TileSelection requires at least one axis")

        axes: list[tuple[int, ...]] = []
        for axis_index, raw_axis in enumerate(raw_axes):
            raw_coordinates = _materialize_iterable(
                raw_axis,
                f"per_axis[{axis_index}]",
            )
            if not raw_coordinates:
                raise ValueError(f"per_axis[{axis_index}] must not be empty")

            coordinates: list[int] = []
            seen: set[int] = set()
            for coordinate in raw_coordinates:
                if type(coordinate) is not int:
                    raise TypeError("tile coordinates must be integers")
                if coordinate < 0:
                    raise ValueError("tile coordinates must be non-negative")
                if coordinate in seen:
                    raise ValueError(
                        f"per_axis[{axis_index}] contains duplicate coordinate "
                        f"{coordinate}"
                    )
                seen.add(coordinate)
                coordinates.append(coordinate)
            axes.append(tuple(coordinates))

        object.__setattr__(self, "per_axis", tuple(axes))

    @property
    def rank(self) -> int:
        """Return the number of selected grid axes."""

        return len(self.per_axis)

    @property
    def compact_grid_shape(self) -> tuple[int, ...]:
        """Return the selected tile count on each compact grid axis."""

        return tuple(len(axis) for axis in self.per_axis)

    @property
    def coordinates(self) -> tuple[tuple[int, ...], ...]:
        """Return source tile coordinates in first-mode-fastest order."""

        return tuple(
            tuple(reversed(coordinate))
            for coordinate in product(*reversed(self.per_axis))
        )

    @property
    def tile_set(self) -> TileSet:
        """Return this selection's coordinates as an arbitrary tile set."""

        return TileSet(self.coordinates)

    def _validate_grid_shape(self, grid_shape: Iterable[int]) -> None:
        """Validate this selection against one concrete tile grid."""

        _validate_coordinates_against_grid(
            self.coordinates,
            rank=self.rank,
            grid_shape=grid_shape,
        )


@dataclass(frozen=True, slots=True, init=False)
class ResidencyPlan:
    """Capture one immutable, disjoint tiled residency decision.

    Args:
        promote: Tiles that must become primary-resident, or ``None``.
        retain: Primary-resident tiles that must remain resident, or ``None``.
        evict: Tiles that should move out of primary storage, or ``None``.

    Examples:
        >>> import strideweave as sw
        >>> plan = sw.ResidencyPlan(promote=sw.TileSet([(0,)]))
        >>> plan.promote.coordinates
        ((0,),)
    """

    promote: TileSet
    retain: TileSet
    evict: TileSet

    def __init__(
        self,
        promote: TileSet | None = None,
        retain: TileSet | None = None,
        evict: TileSet | None = None,
    ) -> None:
        values: dict[str, TileSet] = {}
        for name, value in (
            ("promote", promote),
            ("retain", retain),
            ("evict", evict),
        ):
            if value is None:
                values[name] = TileSet()
            elif not isinstance(value, TileSet):
                raise TypeError(f"{name} must be a TileSet or None")
            else:
                values[name] = value

        promote_coordinates = set(values["promote"].coordinates)
        retain_coordinates = set(values["retain"].coordinates)
        evict_coordinates = set(values["evict"].coordinates)
        if (
            promote_coordinates & retain_coordinates
            or promote_coordinates & evict_coordinates
            or retain_coordinates & evict_coordinates
        ):
            raise ValueError("promote, retain, and evict tile sets must be disjoint")

        object.__setattr__(self, "promote", values["promote"])
        object.__setattr__(self, "retain", values["retain"])
        object.__setattr__(self, "evict", values["evict"])


__all__ = ["ResidencyPlan", "TileSelection", "TileSet"]
