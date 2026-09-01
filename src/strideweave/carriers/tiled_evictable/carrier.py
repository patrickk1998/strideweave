"""Definition-backed tiled storage and transactional residency control."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Any, cast, final

from ...core.layout import Layout, Shape, Stride
from ...core.tensor import Tensor
from .._background import _TILED_RESIDENCY_EXECUTOR
from ..base import Carrier, reject_carrier_subclass
from ..dtype import DType, storage_zero
from ..extension import (
    CarrierDefinition,
    CompositeProvider,
    PreparedComposite,
    StorageProvider,
    Unsupported,
    register_carrier_definition,
)
from ..move.async_move import _move_async_with_route, _resolve_route
from ..move.await_result import _CompletionState
from ..operation_capability import DependentCarrier, OperationCapability
from .projection import AwaitProjection
from .residency import (
    AwaitResidency,
    ResidencyPolicy,
    TiledResidencyFacet,
)
from .values import ResidencyPlan, TileSelection, TileSet

_INVALID = object()


def _implicit_zero(dtype: DType) -> object:
    return storage_zero(dtype)


def _same_value(lhs: object, rhs: object) -> bool:
    try:
        result = lhs == rhs
    except BaseException:
        return False
    return type(result) is bool and result


def _shape(value: object, name: str) -> tuple[int, ...]:
    try:
        materialized = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a finite iterable") from error
    if not materialized:
        raise ValueError(f"{name} must contain at least one axis")
    if any(type(extent) is not int for extent in materialized):
        raise TypeError(f"{name} extents must be integers")
    if any(cast(int, extent) <= 0 for extent in materialized):
        raise ValueError(f"{name} extents must be positive")
    return cast(tuple[int, ...], materialized)


def _decode(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinate: list[int] = []
    remainder = index
    for extent in shape:
        coordinate.append(remainder % extent)
        remainder //= extent
    return tuple(coordinate)


def _encode(coordinate: tuple[int, ...], shape: tuple[int, ...]) -> int:
    result = 0
    scale = 1
    for component, extent in zip(coordinate, shape, strict=True):
        result += component * scale
        scale *= extent
    return result


def _coordinates(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(_decode(index, shape) for index in range(prod(shape)))


def _dense_layout(shape: tuple[int, ...]) -> Layout:
    strides: list[int] = []
    scale = 1
    for extent in shape:
        strides.append(scale)
        scale *= extent
    return Layout(Shape(shape), Stride(tuple(strides)))


@contextmanager
def _owner_access(owned: _OwnedCarrier) -> Iterator[Carrier]:
    owned.carrier._begin_owner_access(owned.token)
    try:
        yield owned.carrier
    finally:
        owned.carrier._end_owner_access(owned.token)


@dataclass(slots=True)
class _OwnedCarrier:
    carrier: Carrier
    token: int


@dataclass(slots=True)
class _TileRecord:
    primary: _OwnedCarrier | None = None
    secondary: _OwnedCarrier | None = None
    implicit_value: object = _INVALID
    dirty: bool = False

    @property
    def valid(self) -> bool:
        return (
            self.primary is not None
            or self.secondary is not None
            or self.implicit_value is not _INVALID
        )


@dataclass(frozen=True, slots=True)
class _TileState:
    location: str
    valid: bool
    dirty: bool


@dataclass(slots=True)
class _StagedTransfer:
    coordinate: tuple[int, ...]
    source: Carrier
    destination: Carrier
    handle: object


@dataclass(slots=True)
class _WorkLease:
    plan: ResidencyPlan
    protected: set[tuple[int, ...]]
    promotion: AwaitResidency
    pins_released: bool = False
    finished: bool = False


def _release_unowned(carrier: Carrier) -> None:
    try:
        if not carrier.is_released() and not carrier.is_owned():
            carrier.release()
    except BaseException:
        pass


def _release_owned(owned: _OwnedCarrier) -> None:
    released = False
    try:
        if not owned.carrier.is_released():
            with _owner_access(owned) as carrier:
                carrier.release()
        released = True
    finally:
        if released and owned.carrier.is_owned():
            owned.carrier._relinquish_ownership(owned.token)


@final
class TiledEvictable(DependentCarrier):
    """Own one flat tensor extent with independently resident logical tiles.

    The supplied tier carriers are full-value seeds or empty allocation
    prototypes. A full primary initializes every tile in primary storage; a
    full secondary initializes every tile in secondary storage; two empty
    tiers create an invalid tiled allocation. The carrier takes exclusive
    ownership of both tier instances and creates independently movable child
    storage for each valid tile.

    Args:
        primary: Live, unowned compute-tier Carrier with either zero slots or
            exactly the full logical extent.
        secondary: Distinct live, unowned backing-tier Carrier with either zero
            slots or exactly the full logical extent and the same dtype.
        grid_shape: Positive tile count on every logical axis.
        tile_shape: Positive within-tile extent on every logical axis.
        residency_policy: Optional structural policy used by later operation,
            projection, scatter, and backward requests.

    Examples:
        >>> import strideweave as sw
        >>> tiled = sw.TiledEvictable(
        ...     sw.Generic([1.0, 2.0], dtype=sw.DType.Float32), sw.Generic([], dtype=sw.DType.Float32), (2,), (1,)
        ... )
        >>> tiled.grid_shape
        (2,)
        >>> tiled.size()
        2
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        reject_carrier_subclass("TiledEvictable")

    def __init__(
        self,
        primary: Carrier,
        secondary: Carrier,
        grid_shape: Iterable[int],
        tile_shape: Iterable[int],
        *,
        residency_policy: ResidencyPolicy | None = None,
    ) -> None:
        if not isinstance(primary, Carrier):
            raise TypeError("primary must be a Carrier instance")
        if not isinstance(secondary, Carrier):
            raise TypeError("secondary must be a Carrier instance")
        if isinstance(primary, DependentCarrier):
            raise TypeError("primary must be an ordinary compute Carrier")
        if isinstance(secondary, DependentCarrier):
            raise TypeError("secondary must be an ordinary backing Carrier")
        if primary is secondary:
            raise ValueError("primary and secondary must be distinct carriers")
        if primary.is_released():
            raise RuntimeError("primary carrier is released")
        if secondary.is_released():
            raise RuntimeError("secondary carrier is released")
        if primary.is_owned():
            raise RuntimeError("primary carrier is already owned")
        if secondary.is_owned():
            raise RuntimeError("secondary carrier is already owned")
        if primary.dtype() is not secondary.dtype():
            raise TypeError("primary and secondary dtypes must match")
        tiled_dtype = primary.dtype()
        normalized_grid = _shape(grid_shape, "grid_shape")
        normalized_tile = _shape(tile_shape, "tile_shape")
        if len(normalized_grid) != len(normalized_tile):
            raise ValueError("grid_shape and tile_shape ranks must match")
        logical_shape = tuple(
            grid * tile
            for grid, tile in zip(normalized_grid, normalized_tile, strict=True)
        )
        logical_size = prod(logical_shape)
        primary_size = primary.size()
        secondary_size = secondary.size()
        for name, carrier, carrier_size in (
            ("primary", primary, primary_size),
            ("secondary", secondary, secondary_size),
        ):
            if carrier_size not in (0, logical_size):
                raise ValueError(
                    f"{name} carrier must be empty or have the full physical extent"
                )
            if not carrier.supports_storage_dtype(tiled_dtype):
                raise TypeError(f"{name} carrier cannot store the tiled dtype")
        if residency_policy is not None and not isinstance(
            residency_policy, ResidencyPolicy
        ):
            raise TypeError("residency_policy must satisfy ResidencyPolicy or be None")
        if residency_policy is not None and not callable(
            getattr(residency_policy, "plan", None)
        ):
            raise TypeError("residency_policy.plan must be callable")
        is_mutable = primary.is_mutable()

        self._grid_shape = normalized_grid
        self._tile_shape = normalized_tile
        self._logical_shape = logical_shape
        self._logical_size = logical_size
        self._tile_size = prod(normalized_tile)
        self._lock = threading.RLock()
        self._prototype_lock = threading.RLock()
        self._pending_tiles: set[tuple[int, ...]] = set()
        self._pending_requests = 0
        self._active_work_requests = 0
        self._deferred_evictions: set[tuple[int, ...]] = set()
        self._pin_counts = {
            coordinate: 0 for coordinate in _coordinates(normalized_grid)
        }
        self._residency_policy = residency_policy
        self._primary = primary
        self._secondary = secondary
        self._primary_owned: _OwnedCarrier | None = None
        self._secondary_owned: _OwnedCarrier | None = None
        self._tiles: dict[tuple[int, ...], _TileRecord] = {
            coordinate: _TileRecord() for coordinate in _coordinates(normalized_grid)
        }

        created: list[Carrier] = []
        claimed: list[_OwnedCarrier] = []
        try:
            primary_owned = _OwnedCarrier(primary, primary._claim_ownership())
            claimed.append(primary_owned)
            secondary_owned = _OwnedCarrier(secondary, secondary._claim_ownership())
            claimed.append(secondary_owned)
            self._primary_owned = primary_owned
            self._secondary_owned = secondary_owned

            if primary_size == logical_size:
                with _owner_access(primary_owned) as owned_primary:
                    for coordinate, values in self._split_seed(owned_primary).items():
                        child = owned_primary.new_like(
                            values,
                            mutable=True,
                            dtype=owned_primary.dtype(),
                        )
                        created.append(child)
                        self._tiles[coordinate].primary = _OwnedCarrier(child, -1)
            elif secondary_size == logical_size:
                with _owner_access(secondary_owned) as owned_secondary:
                    for coordinate, values in self._split_seed(owned_secondary).items():
                        child = owned_secondary.new_like(
                            values,
                            mutable=True,
                            dtype=owned_secondary.dtype(),
                        )
                        created.append(child)
                        self._tiles[coordinate].secondary = _OwnedCarrier(child, -1)

            for record in self._tiles.values():
                child = record.primary or record.secondary
                if child is None:
                    continue
                child.token = child.carrier._claim_ownership()
                claimed.append(child)
            super().__init__(logical_size, dtype=tiled_dtype, mutable=is_mutable)
            self._finalize_dependent_capabilities()
        except BaseException:
            for owned in reversed(claimed):
                try:
                    if owned.carrier.is_owned():
                        owned.carrier._relinquish_ownership(owned.token)
                except BaseException:
                    pass
            for carrier in created:
                _release_unowned(carrier)
            raise

    def __del__(self) -> None:
        for record in getattr(self, "_tiles", {}).values():
            for owned in (record.primary, record.secondary):
                if owned is None:
                    continue
                try:
                    if owned.carrier.is_owned():
                        owned.carrier._relinquish_ownership(owned.token)
                except BaseException:
                    pass
        for name in ("_primary_owned", "_secondary_owned"):
            owned = getattr(self, name, None)
            if owned is None:
                continue
            try:
                if owned.carrier.is_owned():
                    owned.carrier._relinquish_ownership(owned.token)
            except BaseException:
                pass

    def _split_seed(self, carrier: Carrier) -> dict[tuple[int, ...], list[object]]:
        result: dict[tuple[int, ...], list[object]] = {
            coordinate: [None] * self._tile_size
            for coordinate in _coordinates(self._grid_shape)
        }
        for index in range(self._logical_size):
            coordinate, local_index = self._tile_and_local(index)
            result[coordinate][local_index] = carrier[index]
        return result

    def _tile_and_local(self, index: int) -> tuple[tuple[int, ...], int]:
        logical = _decode(index, self._logical_shape)
        tile_coordinate = tuple(
            component // extent
            for component, extent in zip(logical, self._tile_shape, strict=True)
        )
        local_coordinate = tuple(
            component % extent
            for component, extent in zip(logical, self._tile_shape, strict=True)
        )
        return tile_coordinate, _encode(local_coordinate, self._tile_shape)

    @property
    def primary(self) -> Carrier:
        """Return the owned compute-tier seed and allocation prototype."""

        return self._primary

    @property
    def secondary(self) -> Carrier:
        """Return the owned backing-tier seed and allocation prototype."""

        return self._secondary

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Return the immutable tile-grid extents."""

        return self._grid_shape

    @property
    def tile_shape(self) -> tuple[int, ...]:
        """Return the immutable within-tile logical extents."""

        return self._tile_shape

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """Return the canonical full logical extents."""

        return self._logical_shape

    @property
    def residency_policy(self) -> ResidencyPolicy | None:
        """Return the immutable optional residency policy."""

        return self._residency_policy

    def tile_state(self, coordinate: tuple[int, ...]) -> _TileState:
        """Return immutable location, validity, and dirtiness metadata.

        Args:
            coordinate: One full-rank tile coordinate in the configured grid.

        Returns:
            Immutable state snapshot whose location is ``primary``,
            ``secondary``, ``implicit``, or ``invalid``.

        Examples:
            >>> import strideweave as sw
            >>> tiled = sw.TiledEvictable(
            ...     sw.Generic([1.0], dtype=sw.DType.Float32), sw.Generic([], dtype=sw.DType.Float32), (1,), (1,)
            ... )
            >>> tiled.tile_state((0,)).location
            'primary'
        """

        validated = TileSet((coordinate,))
        validated._validate_grid_shape(self._grid_shape)
        with self._lock:
            record = self._tiles[validated.coordinates[0]]
            if record.primary is not None:
                location = "primary"
            elif record.secondary is not None:
                location = "secondary"
            elif record.implicit_value is not _INVALID:
                location = "implicit"
            else:
                location = "invalid"
            return _TileState(location, record.valid, record.dirty)

    def _tile_set_for(self, predicate: Any) -> TileSet:
        with self._lock:
            return TileSet(
                coordinate
                for coordinate, record in self._tiles.items()
                if predicate(record)
            )

    @property
    def primary_tiles(self) -> TileSet:
        """Return the arbitrary set of primary-resident valid tiles."""

        return self._tile_set_for(lambda record: record.primary is not None)

    @property
    def secondary_tiles(self) -> TileSet:
        """Return the arbitrary set of secondary-resident valid tiles."""

        return self._tile_set_for(lambda record: record.secondary is not None)

    @property
    def valid_tiles(self) -> TileSet:
        """Return every valid stored or implicit tile."""

        return self._tile_set_for(lambda record: record.valid)

    @property
    def dirty_tiles(self) -> TileSet:
        """Return primary tiles whose values require preservation on eviction."""

        return self._tile_set_for(lambda record: record.dirty)

    def _read_owned(self, owned: _OwnedCarrier, index: int) -> object:
        with _owner_access(owned) as carrier:
            return carrier[index]

    def _read(self, index: int) -> object:
        coordinate, local_index = self._tile_and_local(index)
        with self._lock:
            record = self._tiles[coordinate]
            if record.primary is not None:
                return self._read_owned(record.primary, local_index)
            if record.secondary is not None:
                return self._read_owned(record.secondary, local_index)
            if record.implicit_value is not _INVALID:
                return record.implicit_value
        raise RuntimeError(f"tile {coordinate!r} is invalid and not resident")

    def _new_primary_tile(self, value: object) -> Carrier:
        prototype = cast(_OwnedCarrier, self._primary_owned)
        with self._prototype_lock:
            with _owner_access(prototype) as carrier:
                return carrier.new_like(
                    [value] * self._tile_size,
                    mutable=True,
                    dtype=self.dtype(),
                )

    def _allocate_primary_tile(self, value: object) -> _OwnedCarrier:
        child = self._new_primary_tile(value)
        return _OwnedCarrier(child, child._claim_ownership())

    def _write(self, index: int, value: object) -> None:
        coordinate, local_index = self._tile_and_local(index)
        while True:
            with self._lock:
                if coordinate in self._pending_tiles:
                    raise RuntimeError(
                        "tile has pending residency work; wait for that request"
                    )
                if self._pin_counts[coordinate] > 0:
                    raise RuntimeError("tile is pinned by active tiled work")
                record = self._tiles[coordinate]
                if record.primary is not None:
                    with _owner_access(record.primary) as carrier:
                        carrier[local_index] = value
                    record.dirty = True
                    record.implicit_value = _INVALID
                    return
                if record.secondary is None:
                    if record.implicit_value is _INVALID:
                        raise RuntimeError(
                            f"tile {coordinate!r} is invalid and cannot be written "
                            "before explicit initialization"
                        )
                    if _same_value(value, record.implicit_value):
                        return
                    fill = record.implicit_value
                    record.primary = self._allocate_primary_tile(fill)
                    record.implicit_value = _INVALID
                    continue
            self.promote(TileSet((coordinate,)))

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> TiledEvictable:
        """Create a same-geometry tiled carrier initialized in primary storage.

        Args:
            values: Full logical value sequence in tensor layout order.
            mutable: Whether public mutation is permitted on the result.
            dtype: Optional result dtype; the current dtype is preserved by default.

        Returns:
            Fresh tiled carrier with the same grid, tiles, and residency policy.

        Examples:
            >>> callable(TiledEvictable.new_like)
            True
        """

        materialized = tuple(values)
        if len(materialized) != self._logical_size:
            raise ValueError("values must fill the tiled carrier's full extent")
        result_dtype = self.dtype() if dtype is None else dtype
        primary_prototype = cast(_OwnedCarrier, self._primary_owned)
        secondary_prototype = cast(_OwnedCarrier, self._secondary_owned)
        with self._prototype_lock:
            with _owner_access(primary_prototype) as primary:
                result_primary = primary.new_like(
                    materialized, mutable=mutable, dtype=result_dtype
                )
            with _owner_access(secondary_prototype) as secondary:
                result_secondary = secondary.allocate_like(
                    0, mutable=True, dtype=result_dtype, empty=True
                )
        result = TiledEvictable(
            result_primary,
            result_secondary,
            self._grid_shape,
            self._tile_shape,
            residency_policy=self._residency_policy,
        )
        zero = _implicit_zero(result_dtype)
        if zero is not None:
            tile_values: dict[tuple[int, ...], list[object]] = {
                coordinate: [None] * self._tile_size
                for coordinate in _coordinates(self._grid_shape)
            }
            for index, value in enumerate(materialized):
                coordinate, local_index = result._tile_and_local(index)
                tile_values[coordinate][local_index] = value
            for coordinate, values in tile_values.items():
                if not all(_same_value(value, zero) for value in values):
                    continue
                record = result._tiles[coordinate]
                if record.primary is not None:
                    _release_owned(record.primary)
                record.primary = None
                record.implicit_value = zero
        return result

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> TiledEvictable:
        """Create an invalid or zero-filled same-geometry tiled carrier.

        Args:
            size: Full logical extent, which must match this carrier.
            mutable: Whether public mutation is permitted on the result.
            dtype: Optional result dtype; the current dtype is preserved by default.
            empty: Whether to leave every result tile explicitly invalid.

        Returns:
            Fresh tiled carrier with the same grid, tiles, and residency policy.

        Examples:
            >>> callable(TiledEvictable.allocate_like)
            True
        """

        if type(size) is not int:
            raise TypeError("size must be an integer")
        if size != self._logical_size:
            raise ValueError("tiled allocation must preserve the full extent")
        result_dtype = self.dtype() if dtype is None else dtype
        if not empty:
            return self.new_like(
                [storage_zero(result_dtype)] * size,
                mutable=mutable,
                dtype=result_dtype,
            )
        primary_prototype = cast(_OwnedCarrier, self._primary_owned)
        secondary_prototype = cast(_OwnedCarrier, self._secondary_owned)
        with self._prototype_lock:
            with _owner_access(primary_prototype) as primary:
                result_primary = primary.allocate_like(
                    0, mutable=mutable, dtype=result_dtype, empty=True
                )
            with _owner_access(secondary_prototype) as secondary:
                result_secondary = secondary.allocate_like(
                    0, mutable=True, dtype=result_dtype, empty=True
                )
        return TiledEvictable(
            result_primary,
            result_secondary,
            self._grid_shape,
            self._tile_shape,
            residency_policy=self._residency_policy,
        )

    def _implicit_zero_like(self) -> TiledEvictable:
        result = self.allocate_like(
            self._logical_size, mutable=True, dtype=self.dtype(), empty=True
        )
        zero = _implicit_zero(self.dtype())
        if zero is None:
            raise TypeError("the tiled dtype has no implicit zero value")
        for record in result._tiles.values():
            record.implicit_value = zero
        return result

    def _install_primary_tiles(
        self,
        values: dict[tuple[int, ...], list[object]],
        *,
        dirty: bool,
        increment_version: bool,
    ) -> None:
        normalized = TileSet(values)
        normalized._validate_grid_shape(self._grid_shape)
        if any(len(tile_values) != self._tile_size for tile_values in values.values()):
            raise ValueError("every installed tile must contain its full extent")
        created: dict[tuple[int, ...], Carrier] = {}
        claimed: dict[tuple[int, ...], _OwnedCarrier] = {}
        prototype = cast(_OwnedCarrier, self._primary_owned)
        try:
            with self._prototype_lock:
                with _owner_access(prototype) as carrier:
                    for coordinate in normalized.coordinates:
                        created[coordinate] = carrier.new_like(
                            values[coordinate],
                            mutable=True,
                            dtype=self.dtype(),
                        )
            for coordinate, child in created.items():
                claimed[coordinate] = _OwnedCarrier(child, child._claim_ownership())
            with self._lock:
                for coordinate in normalized.coordinates:
                    if self._tiles[coordinate].valid:
                        raise RuntimeError("installed tile must start invalid")
                for coordinate, owned in claimed.items():
                    record = self._tiles[coordinate]
                    record.primary = owned
                    record.implicit_value = _INVALID
                    record.dirty = dirty
        except BaseException:
            for owned in claimed.values():
                if owned.carrier.is_owned():
                    owned.carrier._relinquish_ownership(owned.token)
            for child in created.values():
                _release_unowned(child)
            raise
        if increment_version:
            self._increment_version()

    def _normalize_tiles(self, tiles: object) -> TileSet:
        if isinstance(tiles, TileSelection):
            tiles = tiles.tile_set
        if not isinstance(tiles, TileSet):
            raise TypeError("tiles must be a TileSet or TileSelection")
        tiles._validate_grid_shape(self._grid_shape)
        return tiles

    def _stage_transfer(
        self,
        coordinate: tuple[int, ...],
        source: _OwnedCarrier,
        target: _OwnedCarrier,
        route: object,
    ) -> _StagedTransfer:
        with _owner_access(source) as carrier:
            values = [carrier[index] for index in range(self._tile_size)]
            source_clone = carrier.new_like(values, mutable=True, dtype=self.dtype())
        try:
            with self._prototype_lock:
                with _owner_access(target) as carrier:
                    destination = carrier.allocate_like(
                        self._tile_size,
                        mutable=True,
                        dtype=self.dtype(),
                        empty=True,
                    )
        except BaseException:
            _release_unowned(source_clone)
            raise
        try:
            tensor = Tensor(source_clone, 0, _dense_layout(self._tile_shape))
            handle = _move_async_with_route(
                tensor,
                destination,
                cast(Any, route),
                build_graph=False,
            )
        except BaseException:
            _release_unowned(source_clone)
            _release_unowned(destination)
            raise
        return _StagedTransfer(coordinate, source_clone, destination, handle)

    def _finish_pending(self, coordinates: tuple[tuple[int, ...], ...]) -> None:
        with self._lock:
            self._pending_tiles.difference_update(coordinates)
            self._pending_requests -= 1

    def _cleanup_transfers(self, transfers: list[_StagedTransfer]) -> None:
        for transfer in transfers:
            _release_unowned(transfer.source)
            _release_unowned(transfer.destination)

    def _cleanup_materialized(
        self, materialized: dict[tuple[int, ...], Carrier]
    ) -> None:
        for carrier in materialized.values():
            _release_unowned(carrier)

    def _commit_transition(
        self,
        kind: str,
        coordinates: tuple[tuple[int, ...], ...],
        transfers: list[_StagedTransfer],
        outputs: dict[tuple[int, ...], Carrier],
    ) -> None:
        claimed: dict[tuple[int, ...], _OwnedCarrier] = {}
        try:
            for coordinate, carrier in outputs.items():
                claimed[coordinate] = _OwnedCarrier(carrier, carrier._claim_ownership())
        except BaseException:
            for owned in claimed.values():
                owned.carrier._relinquish_ownership(owned.token)
            raise

        discarded: list[_OwnedCarrier] = []
        try:
            with self._lock:
                if self.is_released():
                    raise RuntimeError(
                        "TiledEvictable was released during residency work"
                    )
                for coordinate in coordinates:
                    record = self._tiles[coordinate]
                    replacement = claimed.get(coordinate)
                    if kind == "promote" and replacement is not None:
                        if record.secondary is not None:
                            discarded.append(record.secondary)
                        record.primary = replacement
                        record.secondary = None
                        record.implicit_value = _INVALID
                        record.dirty = False
                    elif kind == "evict" and record.primary is not None:
                        discarded.append(record.primary)
                        record.primary = None
                        if replacement is not None:
                            if record.secondary is not None:
                                discarded.append(record.secondary)
                            record.secondary = replacement
                            record.implicit_value = _INVALID
                        record.dirty = False
        except BaseException:
            for owned in claimed.values():
                if owned.carrier.is_owned():
                    owned.carrier._relinquish_ownership(owned.token)
            raise

        for owned in discarded:
            try:
                _release_owned(owned)
            except BaseException:
                pass
        for transfer in transfers:
            _release_unowned(transfer.source)

    def _run_transition(
        self,
        kind: str,
        coordinates: tuple[tuple[int, ...], ...],
        transfers: list[_StagedTransfer],
        materialized: dict[tuple[int, ...], Carrier],
        state: _CompletionState[None],
        setup_error: BaseException | None,
    ) -> None:
        error = setup_error
        outputs = dict(materialized)
        for transfer in transfers:
            try:
                output = cast(Any, transfer.handle).wait()
                outputs[transfer.coordinate] = cast(Carrier, output.carrier)
            except BaseException as observed:
                if error is None:
                    error = observed
        if error is not None:
            self._cleanup_transfers(transfers)
            self._cleanup_materialized(materialized)
            with self._lock:
                self._pending_tiles.difference_update(coordinates)
                self._pending_requests -= 1
                state.fail(error)
            return
        try:
            self._commit_transition(kind, coordinates, transfers, outputs)
        except BaseException as observed:
            self._cleanup_transfers(transfers)
            self._cleanup_materialized(materialized)
            with self._lock:
                self._pending_tiles.difference_update(coordinates)
                self._pending_requests -= 1
                state.fail(observed)
            return
        with self._lock:
            self._pending_tiles.difference_update(coordinates)
            self._pending_requests -= 1
            state.succeed(None)

    def _transition_async(
        self,
        kind: str,
        tiles: object,
        *,
        pinned_owner: set[tuple[int, ...]] | None = None,
    ) -> AwaitResidency:
        if self.is_released():
            raise RuntimeError("TiledEvictable is released")
        requested = self._normalize_tiles(tiles)
        coordinates = requested.coordinates
        routes: dict[tuple[int, ...], object] = {}
        implicit_values: dict[tuple[int, ...], object] = {}
        with self._lock:
            conflict = self._pending_tiles.intersection(coordinates)
            if conflict:
                raise RuntimeError(
                    "tile has a conflicting pending residency request; wait for it"
                )
            actionable: list[tuple[tuple[int, ...], _OwnedCarrier, _OwnedCarrier]] = []
            for coordinate in coordinates:
                record = self._tiles[coordinate]
                if kind == "promote":
                    if record.primary is not None:
                        continue
                    owner_pin = pinned_owner is not None and coordinate in pinned_owner
                    other_pins = self._pin_counts[coordinate] - int(owner_pin)
                    if other_pins > 0 or (
                        self._pin_counts[coordinate] > 0 and not owner_pin
                    ):
                        raise RuntimeError("tile is pinned by active tiled work")
                    if record.secondary is None:
                        if record.implicit_value is _INVALID:
                            raise RuntimeError(
                                f"tile {coordinate!r} is invalid and cannot be promoted"
                            )
                        implicit_values[coordinate] = record.implicit_value
                        continue
                    actionable.append(
                        (
                            coordinate,
                            record.secondary,
                            cast(_OwnedCarrier, self._primary_owned),
                        )
                    )
                else:
                    if record.primary is None:
                        continue
                    owner_pin = pinned_owner is not None and coordinate in pinned_owner
                    other_pins = self._pin_counts[coordinate] - int(owner_pin)
                    if other_pins > 0 or (
                        self._pin_counts[coordinate] > 0 and not owner_pin
                    ):
                        raise RuntimeError("tile is pinned by active tiled work")
                    actionable.append(
                        (
                            coordinate,
                            record.primary,
                            cast(_OwnedCarrier, self._secondary_owned),
                        )
                    )
            for coordinate, source, target in actionable:
                routes[coordinate] = _resolve_route(
                    type(source.carrier), type(target.carrier)
                )
            if not actionable and not implicit_values:
                state: _CompletionState[None] = _CompletionState()
                state.succeed(None)
                return cast(AwaitResidency, AwaitResidency._create(state))
            self._pending_tiles.update(coordinates)
            self._pending_requests += 1

        state = _CompletionState()
        handle = cast(AwaitResidency, AwaitResidency._create(state))
        transfers: list[_StagedTransfer] = []
        materialized: dict[tuple[int, ...], Carrier] = {}
        setup_error: BaseException | None = None
        try:
            for coordinate, value in implicit_values.items():
                materialized[coordinate] = self._new_primary_tile(value)
        except BaseException:
            self._cleanup_materialized(materialized)
            self._finish_pending(coordinates)
            raise
        for coordinate, source, target in actionable:
            try:
                transfers.append(
                    self._stage_transfer(coordinate, source, target, routes[coordinate])
                )
            except BaseException as error:
                if not transfers:
                    self._cleanup_materialized(materialized)
                    self._finish_pending(coordinates)
                    raise
                setup_error = error
                break
        worker = partial(
            self._run_transition,
            kind,
            coordinates,
            transfers,
            materialized,
            state,
            setup_error,
        )
        try:
            _TILED_RESIDENCY_EXECUTOR.submit(worker)
        except BaseException:
            for transfer in transfers:
                try:
                    cast(Any, transfer.handle).wait()
                except BaseException:
                    pass
            self._cleanup_transfers(transfers)
            self._cleanup_materialized(materialized)
            self._finish_pending(coordinates)
            raise
        return handle

    def _promote_async(self, tiles: object) -> AwaitResidency:
        return self._transition_async("promote", tiles)

    def _evict_async(self, tiles: object) -> AwaitResidency:
        return self._transition_async("evict", tiles)

    def promote_async(self, tiles: object) -> AwaitResidency:
        """Eagerly make an immutable tile set primary-resident.

        Args:
            tiles: ``TileSet`` selecting the tiles to promote.

        Returns:
            Blocking-only completion handle for the residency request.

        Examples:
            >>> callable(TiledEvictable.promote_async)
            True
        """

        facet = cast(Any, self.require_facet(TiledResidencyFacet))
        return cast(AwaitResidency, facet.promote_async(self, tiles))

    def evict_async(self, tiles: object) -> AwaitResidency:
        """Eagerly preserve and remove an immutable tile set from primary.

        Args:
            tiles: ``TileSet`` selecting the tiles to evict.

        Returns:
            Blocking-only completion handle for the residency request.

        Examples:
            >>> callable(TiledEvictable.evict_async)
            True
        """

        facet = cast(Any, self.require_facet(TiledResidencyFacet))
        return cast(AwaitResidency, facet.evict_async(self, tiles))

    def promote(self, tiles: object) -> None:
        """Promote tiles and block until the residency request completes.

        Args:
            tiles: ``TileSet`` selecting the tiles to promote.

        Returns:
            None after the selected tiles become primary-resident.

        Examples:
            >>> callable(TiledEvictable.promote)
            True
        """

        self.promote_async(tiles).wait()

    def evict(self, tiles: object) -> None:
        """Evict tiles and block until the residency request completes.

        Args:
            tiles: ``TileSet`` selecting the tiles to evict.

        Returns:
            None after the selected tiles leave primary residency.

        Examples:
            >>> callable(TiledEvictable.evict)
            True
        """

        self.evict_async(tiles).wait()

    def project(
        self,
        tensor: object,
        selection: object,
        destination: object | None = None,
    ) -> AwaitProjection:
        """Eagerly gather selected tiles into an ordinary compact Tensor.

        Args:
            tensor: Full Tensor backed by this exact tiled carrier.
            selection: Immutable ordered ``TileSelection`` to gather.
            destination: Optional mutable compute carrier for the compact result.

        Returns:
            Blocking-only completion handle for the compact Tensor.

        Examples:
            >>> callable(TiledEvictable.project)
            True
        """

        from .selection_ops import project

        return project(cast(Any, self), tensor, selection, destination)

    def scatter(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        compact: object,
        template: object,
        selection: object,
        *,
        reduction: str = "replace",
    ) -> Any:
        """Eagerly scatter a compact Tensor into a fresh tiled result.

        Args:
            compact: Ordinary compute-carrier Tensor containing selected values.
            template: Full Tensor backed by this exact tiled carrier.
            selection: Immutable ordered ``TileSelection`` describing placement.
            reduction: ``"replace"`` or ``"add"``.

        Returns:
            ``AwaitMove`` handle for the fresh full-shaped Tensor.

        Examples:
            >>> callable(TiledEvictable.scatter)
            True
        """

        from .selection_ops import scatter

        return scatter(
            cast(Any, self),
            compact,
            template,
            selection,
            reduction=reduction,
        )

    def _plan_residency(
        self,
        required: TileSet,
        purpose: str,
        *,
        default_promote_required: bool = True,
    ) -> ResidencyPlan:
        if purpose not in ("operation", "projection", "scatter", "backward"):
            raise ValueError("unsupported residency policy purpose")
        required._validate_grid_shape(self._grid_shape)
        policy = self._residency_policy
        if policy is None:
            return ResidencyPlan(promote=required if default_promote_required else None)
        plan = policy.plan(required, purpose)
        if not isinstance(plan, ResidencyPlan):
            raise TypeError("ResidencyPolicy.plan must return ResidencyPlan")
        for tiles in (plan.promote, plan.retain, plan.evict):
            tiles._validate_grid_shape(self._grid_shape)
        return plan

    def _begin_tiled_work(
        self,
        required: TileSet,
        purpose: str,
        *,
        ensure_required_values: bool = True,
    ) -> _WorkLease:
        plan = self._plan_residency(
            required,
            purpose,
            default_promote_required=ensure_required_values,
        )
        to_promote_coordinates = set(plan.promote.coordinates)
        if ensure_required_values:
            to_promote_coordinates.update(required.coordinates)
        to_promote = TileSet(to_promote_coordinates)
        protected = (
            set(required.coordinates)
            | set(plan.promote.coordinates)
            | set(plan.retain.coordinates)
        )
        with self._lock:
            conflict = self._pending_tiles.intersection(protected)
            if conflict:
                raise RuntimeError(
                    "required tile has pending residency work; wait for it"
                )
            for coordinate in protected:
                self._pin_counts[coordinate] += 1
            self._active_work_requests += 1
            try:
                promotion = self._transition_async(
                    "promote",
                    to_promote,
                    pinned_owner=protected,
                )
            except BaseException:
                for coordinate in protected:
                    self._pin_counts[coordinate] -= 1
                self._active_work_requests -= 1
                raise
        return _WorkLease(plan, protected, promotion)

    def _end_tiled_work(
        self,
        lease: _WorkLease,
        *,
        apply_policy_evictions: bool,
        release_active: bool,
    ) -> None:
        if lease.pins_released:
            raise RuntimeError("tiled work lease was already released")
        eviction: AwaitResidency | None = None
        ready: set[tuple[int, ...]] = set()
        error: BaseException | None = None
        with self._lock:
            for coordinate in lease.protected:
                self._pin_counts[coordinate] -= 1
            lease.pins_released = True
            if apply_policy_evictions:
                self._deferred_evictions.update(lease.plan.evict.coordinates)
            ready = {
                coordinate
                for coordinate in self._deferred_evictions
                if self._pin_counts[coordinate] == 0
                and coordinate not in self._pending_tiles
            }
            if ready:
                self._deferred_evictions.difference_update(ready)
                try:
                    eviction = self._transition_async("evict", TileSet(ready))
                except BaseException as observed:
                    self._deferred_evictions.update(ready)
                    error = observed
        if eviction is not None:
            try:
                eviction.wait()
            except BaseException as observed:
                with self._lock:
                    self._deferred_evictions.update(ready)
                error = observed
        if release_active:
            with self._lock:
                self._active_work_requests -= 1
                lease.finished = True
        if error is not None:
            raise error

    def _complete_tiled_work(
        self,
        lease: _WorkLease,
        state: Any,
        *,
        result: object = _INVALID,
        error: BaseException | None = None,
    ) -> None:
        if not lease.pins_released or lease.finished:
            raise RuntimeError("tiled work lease is not ready for completion")
        with self._lock:
            self._active_work_requests -= 1
            lease.finished = True
            if error is None:
                state.succeed(result)
            else:
                state.fail(error)

    @contextmanager
    def _pin_tiles_for_work(
        self, required: TileSet, purpose: str
    ) -> Iterator[ResidencyPlan]:
        lease = self._begin_tiled_work(required, purpose)
        try:
            lease.promotion.wait()
        except BaseException:
            self._end_tiled_work(
                lease,
                apply_policy_evictions=False,
                release_active=True,
            )
            raise

        body_error: BaseException | None = None
        try:
            yield lease.plan
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                self._end_tiled_work(
                    lease,
                    apply_policy_evictions=True,
                    release_active=True,
                )
            except BaseException:
                if body_error is None:
                    raise

    def _composite_capabilities(self) -> tuple[OperationCapability, ...]:
        from .direct_ops import composite_capabilities

        return composite_capabilities(cast(Any, self))

    def _composite_result_carriers(self, invocation: object) -> tuple[type, ...]:
        from .direct_ops import composite_result_carriers

        return composite_result_carriers(cast(Any, self), invocation)

    def _prepare_composite(self, invocation: object) -> PreparedComposite | Unsupported:
        from .direct_ops import prepare_composite

        return prepare_composite(cast(Any, self), invocation)

    def _composite_vjp_scope(self, invocation: object) -> Any:
        from .direct_ops import composite_vjp_scope

        return composite_vjp_scope(cast(Any, self), cast(Any, invocation))

    def _release_storage(self) -> None:
        with self._lock:
            if self._pending_requests or self._active_work_requests:
                raise RuntimeError(
                    "TiledEvictable has pending work; wait before release"
                )
            owned = []
            for record in self._tiles.values():
                if record.primary is not None:
                    owned.append(record.primary)
                if record.secondary is not None:
                    owned.append(record.secondary)
            for prototype in (self._primary_owned, self._secondary_owned):
                if prototype is not None:
                    owned.append(prototype)
        error: BaseException | None = None
        for item in owned:
            try:
                _release_owned(item)
            except BaseException as observed:
                if error is None:
                    error = observed
        if error is not None:
            raise error
        with self._lock:
            for record in self._tiles.values():
                record.primary = None
                record.secondary = None
                record.implicit_value = _INVALID
                record.dirty = False
            self._primary_owned = None
            self._secondary_owned = None
            self._deferred_evictions.clear()


def _validate_full_tensor(carrier: TiledEvictable, value: object, name: str) -> Tensor:
    """Validate one full Tensor against the carrier's exact tile footprints."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.carrier is not carrier:
        raise TypeError(f"{name} must be backed by this exact TiledEvictable")
    if carrier.is_released():
        raise RuntimeError("TiledEvictable is released")
    if len(value.layout) != len(carrier.logical_shape):
        raise ValueError(f"{name} rank is incompatible with tiled geometry")
    extents = tuple(
        mode if isinstance(mode, int) else cast(Any, mode).logical_size
        for mode in value.layout.shape.top_level
    )
    if extents != carrier.logical_shape:
        raise ValueError(f"{name} shape is incompatible with tiled geometry")

    expected = {coordinate: set() for coordinate in _coordinates(carrier.grid_shape)}
    actual = {coordinate: set() for coordinate in expected}
    for logical_index in range(carrier.size()):
        tile, _ = carrier._tile_and_local(logical_index)
        expected[tile].add(logical_index)
        actual[tile].add(value.offset + value.layout.index(logical_index))
    if any(
        len(actual[coordinate]) != carrier._tile_size
        or actual[coordinate] != expected[coordinate]
        for coordinate in expected
    ):
        raise ValueError(
            f"{name} Layout is incompatible with TiledEvictable tile footprints"
        )
    return value


def _storage_size(carrier: object) -> int:
    return cast(TiledEvictable, carrier)._logical_size


def _storage_supports_dtype(carrier: Carrier, dtype: DType) -> bool:
    tiled = cast(TiledEvictable, carrier)
    return tiled.primary.supports_storage_dtype(
        dtype
    ) and tiled.secondary.supports_storage_dtype(dtype)


def _storage_allocate(carrier: object, slots: int) -> None:
    tiled = cast(TiledEvictable, carrier)
    if slots != tiled._logical_size:
        raise ValueError("TiledEvictable allocation must preserve its full extent")


def _storage_release(carrier: object) -> None:
    cast(TiledEvictable, carrier)._release_storage()


def _storage_read(carrier: object, index: int) -> object:
    return cast(TiledEvictable, carrier)._read(index)


def _storage_write(carrier: object, index: int, value: object) -> None:
    cast(TiledEvictable, carrier)._write(index, value)


_TILED_DEFINITION = CarrierDefinition(
    StorageProvider(
        _storage_supports_dtype,
        _storage_size,
        _storage_allocate,
        _storage_release,
        _storage_read,
        _storage_write,
    ),
    composite=CompositeProvider(
        lambda carrier: cast(TiledEvictable, carrier)._composite_capabilities(),
        lambda carrier, invocation: cast(
            TiledEvictable, carrier
        )._composite_result_carriers(invocation),
        lambda carrier, invocation: cast(TiledEvictable, carrier)._prepare_composite(
            invocation
        ),
    ),
    facets=(TiledResidencyFacet(),),
)
register_carrier_definition(TiledEvictable, _TILED_DEFINITION)


__all__ = ["TiledEvictable"]
