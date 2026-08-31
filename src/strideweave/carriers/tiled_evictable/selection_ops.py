"""Eager projection, functional scatter, and their tiled VJPs."""

from __future__ import annotations

from importlib import import_module
from math import prod
from typing import Any, cast

from ...core.tensor import Tensor
from ...functional import add
from .._background import _TILED_SELECTION_EXECUTOR
from ..base import Carrier
from ..extension import _observe_carrier_definition
from ..move.async_move import (
    _allocate_destination,
    _preflight_destination_extent,
    _without_graph,
)
from ..move.await_result import AwaitMove, _CompletionState
from ..operation_capability import DependentCarrier
from ..operation_helpers import Operation
from .carrier import (
    TiledEvictable,
    _decode,
    _dense_layout,
    _owner_access,
    _validate_full_tensor,
    _WorkLease,
)
from .projection import AwaitProjection
from .values import TileSelection

_operation = import_module("strideweave._operation")
_MISSING = object()


def _compact_shape(
    carrier: TiledEvictable, selection: TileSelection
) -> tuple[int, ...]:
    return tuple(
        count * tile
        for count, tile in zip(
            selection.compact_grid_shape,
            carrier.tile_shape,
            strict=True,
        )
    )


def _validate_selection(carrier: TiledEvictable, selection: object) -> TileSelection:
    if not isinstance(selection, TileSelection):
        raise TypeError("selection must be a TileSelection")
    selection._validate_grid_shape(carrier.grid_shape)
    return selection


def _selection_mapping(
    carrier: TiledEvictable, selection: TileSelection
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    compact_shape = _compact_shape(carrier, selection)
    mapping = []
    for compact_index in range(prod(compact_shape)):
        compact_coordinate = _decode(compact_index, compact_shape)
        compact_tile = tuple(
            component // extent
            for component, extent in zip(
                compact_coordinate, carrier.tile_shape, strict=True
            )
        )
        local = tuple(
            component % extent
            for component, extent in zip(
                compact_coordinate, carrier.tile_shape, strict=True
            )
        )
        source_tile = tuple(
            selection.per_axis[axis][compact_tile[axis]]
            for axis in range(len(compact_shape))
        )
        source_coordinate = tuple(
            tile * extent + component
            for tile, extent, component in zip(
                source_tile,
                carrier.tile_shape,
                local,
                strict=True,
            )
        )
        mapping.append((compact_index, source_coordinate, source_tile, local))
    return tuple(mapping)


def _release_destination(destination: Carrier, token: int, *, release: bool) -> None:
    if destination.is_owned():
        destination._relinquish_ownership(token)
    if release and not destination.is_released():
        destination.release()


def _prepare_destination(
    carrier: TiledEvictable,
    destination: object | None,
    compact_size: int,
) -> tuple[Carrier, int, bool]:
    created = destination is None
    if destination is None:
        prototype = cast(Any, carrier)._primary_owned
        with cast(Any, carrier)._prototype_lock:
            with _owner_access(prototype) as primary:
                destination = primary.allocate_like(
                    compact_size,
                    mutable=True,
                    dtype=carrier.dtype(),
                    empty=True,
                )
    if not isinstance(destination, Carrier):
        raise TypeError("destination must be a Carrier or None")
    if isinstance(destination, DependentCarrier):
        raise TypeError("destination must be an ordinary compute Carrier")
    if type(destination) is not type(carrier.primary):
        raise TypeError("destination must use the configured compute carrier class")
    if destination.is_released():
        raise RuntimeError("destination is released")
    if destination.is_owned():
        raise RuntimeError("destination is already owned")
    if not destination.is_mutable():
        raise RuntimeError("destination must be publicly mutable")
    if destination.dtype() is not carrier.dtype():
        raise TypeError("destination dtype must match the tiled dtype")
    definition = _observe_carrier_definition(type(destination))
    _preflight_destination_extent(destination, compact_size, definition)
    token = destination._claim_ownership()
    try:
        destination._begin_owner_access(token)
        try:
            _allocate_destination(destination, compact_size, definition)
        finally:
            destination._end_owner_access(token)
    except BaseException:
        _release_destination(destination, token, release=created)
        raise
    return destination, token, created


def _write_destination(destination: Carrier, token: int, values: list[object]) -> None:
    destination._begin_owner_access(token)
    try:
        for index, value in enumerate(values):
            destination[index] = value
    finally:
        destination._end_owner_access(token)


def _full_implicit_gradient(template: Tensor) -> Tensor:
    carrier = cast(TiledEvictable, template.carrier)._implicit_zero_like()
    return Tensor(carrier, template.offset, template.layout)


def _projection_gradient(
    source: Tensor, compact: Tensor, selection: TileSelection
) -> Tensor:
    result = _full_implicit_gradient(source)
    carrier = cast(TiledEvictable, source.carrier)
    for compact_index, source_coordinate, _, _ in _selection_mapping(
        carrier, selection
    ):
        result[source_coordinate] = compact[compact_index]
    return result


def _gather_gradient(
    compact_template: Tensor,
    full: Tensor,
    carrier: TiledEvictable,
    selection: TileSelection,
) -> Tensor:
    values = [
        full[source_coordinate]
        for _, source_coordinate, _, _ in _selection_mapping(carrier, selection)
    ]
    return Tensor(
        compact_template.carrier.new_like(
            values,
            mutable=True,
            dtype=compact_template.dtype(),
        ),
        0,
        compact_template.layout,
    )


def _template_gradient(
    template: Tensor,
    full: Tensor,
    selection: TileSelection,
    reduction: str,
) -> Tensor:
    result = _full_implicit_gradient(template)
    carrier = cast(TiledEvictable, template.carrier)
    selected = set(selection.coordinates)
    for index in range(prod(carrier.logical_shape)):
        coordinate = _decode(index, carrier.logical_shape)
        tile = tuple(
            component // extent
            for component, extent in zip(coordinate, carrier.tile_shape, strict=True)
        )
        if reduction == "replace" and tile in selected:
            continue
        result[coordinate] = full[coordinate]
    return result


class _ProjectionOperation(Operation):
    def __init__(self, source: Tensor, selection: TileSelection) -> None:
        super().__init__()
        self._selection = selection
        self.store_inputs(source)

    def _forward(self, *inputs: Any) -> Any:
        del inputs
        raise RuntimeError("projection operations are attached after eager completion")

    def backward(self, gradient: Any) -> tuple[Tensor]:
        if not isinstance(gradient, Tensor):
            raise TypeError("projection gradient must be a Tensor")
        (source,) = self.inputs()
        return (_projection_gradient(source, gradient, self._selection),)


class _ScatterOperation(Operation):
    def __init__(
        self,
        compact: Tensor,
        template: Tensor,
        selection: TileSelection,
        reduction: str,
    ) -> None:
        super().__init__()
        self._selection = selection
        self._reduction = reduction
        self.store_inputs(compact, template)

    def _forward(self, *inputs: Any) -> Any:
        del inputs
        raise RuntimeError("scatter operations are attached after eager completion")

    def backward(self, gradient: Any) -> tuple[Tensor, Tensor]:
        if not isinstance(gradient, Tensor):
            raise TypeError("scatter gradient must be a Tensor")
        compact, template = self.inputs()
        carrier = cast(TiledEvictable, template.carrier)
        return (
            _gather_gradient(compact, gradient, carrier, self._selection),
            _template_gradient(
                template,
                gradient,
                self._selection,
                self._reduction,
            ),
        )


def _finish_work_failure(
    carrier: TiledEvictable,
    lease: _WorkLease,
    state: Any,
    error: BaseException,
) -> None:
    terminal_error = error
    if not lease.pins_released:
        try:
            cast(Any, carrier)._end_tiled_work(
                lease,
                apply_policy_evictions=False,
                release_active=False,
            )
        except BaseException as cleanup_error:
            terminal_error = cleanup_error
    cast(Any, carrier)._complete_tiled_work(
        lease,
        state,
        error=terminal_error,
    )


def project(
    carrier: TiledEvictable,
    tensor: object,
    selection: object,
    destination: object | None = None,
) -> AwaitProjection:
    source = _validate_full_tensor(carrier, tensor, "tensor")
    selected = _validate_selection(carrier, selection)
    compact_layout = _dense_layout(_compact_shape(carrier, selected))
    destination_carrier, destination_token, created = _prepare_destination(
        carrier,
        destination,
        compact_layout.cosize,
    )
    operation = (
        _ProjectionOperation(source, selected)
        if bool(_operation.is_grad_enabled()) and source.is_differentiable()
        else None
    )
    try:
        lease = cast(Any, carrier)._begin_tiled_work(
            selected.tile_set,
            "projection",
        )
    except BaseException:
        _release_destination(
            destination_carrier,
            destination_token,
            release=created,
        )
        raise

    state: _CompletionState[Tensor] = _CompletionState()
    handle = cast(AwaitProjection, AwaitProjection._create(state))
    mapping = _selection_mapping(carrier, selected)
    values: list[object] = [_MISSING] * compact_layout.shape.logical_size
    setup_error: BaseException | None = None
    try:
        ready_tiles = carrier.primary_tiles.coordinates
        for compact_index, source_coordinate, source_tile, _ in mapping:
            location = carrier.tile_state(source_tile).location
            if source_tile in ready_tiles or location == "implicit":
                values[compact_index] = source[source_coordinate]
    except BaseException as error:
        setup_error = error

    def run() -> None:
        try:
            if setup_error is not None:
                raise setup_error
            lease.promotion.wait()
            with _without_graph():
                for compact_index, source_coordinate, _, _ in mapping:
                    if values[compact_index] is _MISSING:
                        values[compact_index] = source[source_coordinate]
                _write_destination(destination_carrier, destination_token, values)
            result = Tensor(destination_carrier, 0, compact_layout)
            if operation is not None:
                result.autograd_ctx = operation
            cast(Any, carrier)._end_tiled_work(
                lease,
                apply_policy_evictions=True,
                release_active=False,
            )
            _release_destination(
                destination_carrier,
                destination_token,
                release=False,
            )
            cast(Any, carrier)._complete_tiled_work(
                lease,
                state,
                result=result,
            )
        except BaseException as error:
            _release_destination(
                destination_carrier,
                destination_token,
                release=created,
            )
            _finish_work_failure(carrier, lease, state, error)

    try:
        _TILED_SELECTION_EXECUTOR.submit(run)
    except BaseException as error:
        _release_destination(
            destination_carrier,
            destination_token,
            release=created,
        )
        _finish_work_failure(carrier, lease, state, error)
        raise
    return handle


def _tile_buffers(
    carrier: TiledEvictable,
    template: Tensor,
    selection: TileSelection,
    values: list[object],
) -> dict[tuple[int, ...], list[object]]:
    buffers = {
        coordinate: [_MISSING] * prod(carrier.tile_shape)
        for coordinate in selection.coordinates
    }
    for compact_index, source_coordinate, source_tile, _ in _selection_mapping(
        carrier, selection
    ):
        physical = template.offset + template.layout.index(source_coordinate)
        destination_tile, local_index = carrier._tile_and_local(physical)
        if destination_tile != source_tile:
            raise RuntimeError("validated scatter footprint changed during execution")
        buffers[source_tile][local_index] = values[compact_index]
    if any(value is _MISSING for buffer in buffers.values() for value in buffer):
        raise RuntimeError("scatter did not fill every selected tile value")
    return buffers


def _compact_values(compact: Tensor, token: int) -> list[object]:
    compact.carrier._begin_owner_access(token)
    try:
        return [compact[index] for index in range(compact.size())]
    finally:
        compact.carrier._end_owner_access(token)


def _add_values(
    carrier: TiledEvictable,
    compact: Tensor,
    compact_token: int,
    template: Tensor,
    selection: TileSelection,
) -> list[object]:
    template_values = [
        template[source_coordinate]
        for _, source_coordinate, _, _ in _selection_mapping(carrier, selection)
    ]
    template_carrier: Carrier | None = None
    combined: Tensor | None = None
    compact.carrier._begin_owner_access(compact_token)
    try:
        template_carrier = compact.carrier.new_like(
            template_values,
            mutable=True,
            dtype=compact.dtype(),
        )
        template_compact = Tensor(template_carrier, 0, compact.layout)
        with _without_graph():
            combined = add(template_compact, compact)
        combined_tensor = cast(Tensor, combined)
        values = [combined_tensor[index] for index in range(combined_tensor.size())]
    finally:
        compact.carrier._end_owner_access(compact_token)
        if template_carrier is not None and not template_carrier.is_released():
            template_carrier.release()
        if combined is not None and not combined.carrier.is_released():
            combined.carrier.release()
    return values


def scatter(
    carrier: TiledEvictable,
    compact: object,
    template: object,
    selection: object,
    *,
    reduction: str = "replace",
) -> AwaitMove[Tensor]:
    if reduction not in ("replace", "add"):
        raise ValueError("reduction must be 'replace' or 'add'")
    full = _validate_full_tensor(carrier, template, "template")
    selected = _validate_selection(carrier, selection)
    if not isinstance(compact, Tensor):
        raise TypeError("compact must be a Tensor")
    if isinstance(compact.carrier, DependentCarrier):
        raise TypeError("compact must use an ordinary compute Carrier")
    if type(compact.carrier) is not type(carrier.primary):
        raise TypeError("compact must use the configured compute carrier class")
    if compact.carrier.is_released():
        raise RuntimeError("compact carrier is released")
    if compact.carrier.is_owned():
        raise RuntimeError("compact carrier is already owned")
    if compact.dtype() is not carrier.dtype():
        raise TypeError("compact dtype must match the tiled dtype")
    compact_layout = _dense_layout(_compact_shape(carrier, selected))
    if compact.layout != compact_layout:
        raise ValueError("compact shape and dense layout must match the selection")
    if reduction == "add" and not set(selected.coordinates).issubset(
        carrier.valid_tiles.coordinates
    ):
        raise RuntimeError("add scatter requires valid selected template tiles")

    operation = (
        _ScatterOperation(compact, full, selected, reduction)
        if bool(_operation.is_grad_enabled())
        and (compact.is_differentiable() or full.is_differentiable())
        else None
    )
    compact_token = compact.carrier._claim_ownership()
    try:
        lease = cast(Any, carrier)._begin_tiled_work(
            selected.tile_set,
            "scatter",
            ensure_required_values=reduction == "add",
        )
    except BaseException:
        compact.carrier._relinquish_ownership(compact_token)
        raise

    state: _CompletionState[Tensor] = _CompletionState()
    handle = cast(AwaitMove[Tensor], AwaitMove._create(state))

    def run() -> None:
        result_carrier: TiledEvictable | None = None
        try:
            lease.promotion.wait()
            with _without_graph():
                values = (
                    _compact_values(compact, compact_token)
                    if reduction == "replace"
                    else _add_values(
                        carrier,
                        compact,
                        compact_token,
                        full,
                        selected,
                    )
                )
                result_carrier = carrier.allocate_like(
                    carrier.size(),
                    mutable=True,
                    dtype=carrier.dtype(),
                    empty=True,
                )
                result_carrier._install_primary_tiles(
                    _tile_buffers(carrier, full, selected, values),
                    dirty=True,
                    increment_version=True,
                )
                result = Tensor(result_carrier, full.offset, full.layout)
            if operation is not None:
                result.autograd_ctx = operation
            cast(Any, carrier)._end_tiled_work(
                lease,
                apply_policy_evictions=True,
                release_active=False,
            )
            compact.carrier._relinquish_ownership(compact_token)
            cast(Any, carrier)._complete_tiled_work(
                lease,
                state,
                result=result,
            )
        except BaseException as error:
            if compact.carrier.is_owned():
                compact.carrier._relinquish_ownership(compact_token)
            if result_carrier is not None and not result_carrier.is_released():
                try:
                    result_carrier.release()
                except BaseException:
                    pass
            _finish_work_failure(carrier, lease, state, error)

    try:
        _TILED_SELECTION_EXECUTOR.submit(run)
    except BaseException as error:
        compact.carrier._relinquish_ownership(compact_token)
        _finish_work_failure(carrier, lease, state, error)
        raise
    return handle


__all__ = ["project", "scatter"]
