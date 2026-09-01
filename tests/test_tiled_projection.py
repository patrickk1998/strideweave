"""Contract tests for eager TiledEvictable projection and gather autograd."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from importlib import import_module
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave.carriers.extension import (
    CarrierDefinition,
    PreparedTransfer,
    StorageProvider,
    TransferProvider,
    TransferRequest,
    TransferRoute,
    register_carrier_definition,
)
from strideweave.carriers.tiled_evictable import (
    AwaitProjection,
    TiledEvictable,
    TileSelection,
)
from strideweave.friendly import column_major
from strideweave.friendly import tensor as make_tensor

GRID = (2, 2)
TILE = (2, 2)
FULL_SHAPE = (4, 4)
FULL_SIZE = 16


def _seed_values() -> list[float]:
    # Physical order is first-mode-fastest, matching column_major(4, 4).
    return [float(i + 10 * j) for j in range(4) for i in range(4)]


def _tiled(*, secondary: bool = False) -> TiledEvictable:
    values = _seed_values()
    primary = sw.Generic([] if secondary else values, dtype=sw.DType.Float32)
    backing = sw.Generic(values if secondary else [], dtype=sw.DType.Float32)
    return TiledEvictable(primary, backing, GRID, TILE)


def _tensor(carrier: TiledEvictable) -> sw.Tensor:
    return sw.Tensor(carrier, 0, column_major(*FULL_SHAPE))


def _within_tile_permutation() -> sw.Layout:
    return sw.Layout(
        sw.Shape([[2, 2], [2, 2]]),
        sw.Stride([[4, 2], [1, 8]]),
    )


def _incompatible_layout(kind: str) -> sw.Layout:
    if kind == "spans":
        return sw.Layout(
            sw.Shape([[2, 2], [2, 2]]),
            sw.Stride([[1, 2], [8, 4]]),
        )
    if kind == "remaps":
        return sw.Layout(sw.Shape(FULL_SHAPE), sw.Stride((4, 1)))
    if kind == "aliases":
        return sw.Layout(sw.Shape(FULL_SHAPE), sw.Stride((1, 0)))
    assert kind == "omits"
    return sw.Layout(
        sw.Shape([[2, 2], [2, 2]]),
        sw.Stride([[1, 8], [2, 4]]),
    )


def _expected_projection(source: sw.Tensor, selection: TileSelection) -> list[float]:
    values: list[float] = []
    compact_shape = tuple(
        axis_count * tile_extent
        for axis_count, tile_extent in zip(
            selection.compact_grid_shape, TILE, strict=True
        )
    )
    # Compact logical coordinates are mapped back through the caller's axis
    # order before reading the full source tensor.
    for flat in range(1 * compact_shape[0] * compact_shape[1]):
        remainder = flat
        compact = []
        for extent in compact_shape:
            compact.append(remainder % extent)
            remainder //= extent
        source_coordinate = tuple(
            selection.per_axis[axis][compact[axis] // TILE[axis]] * TILE[axis]
            + compact[axis] % TILE[axis]
            for axis in range(2)
        )
        values.append(float(source[source_coordinate]))
    return values


def test_projection_exports_and_handle_contract() -> None:
    import strideweave.carriers.tiled_evictable as tiled

    assert sw.AwaitProjection is tiled.AwaitProjection is AwaitProjection
    assert "AwaitProjection" in sw.__all__
    assert "AwaitProjection" in tiled.__all__
    assert not hasattr(AwaitProjection, "__await__")


def test_selection_order_layout_dtype_and_carrier_identity() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    selection = TileSelection(([1, 0], [1]))
    before = (
        source.layout,
        source.carrier,
        source.version,
        tuple(source[i] for i in range(FULL_SIZE)),
    )

    pending = carrier.project(source, selection)
    assert isinstance(pending, AwaitProjection)
    result = pending.wait()

    assert result is pending.wait()
    assert result.carrier is not carrier
    assert type(result.carrier) is type(carrier.primary)
    assert result.dtype() is sw.DType.Float32
    assert result.layout == column_major(4, 2)
    assert [result[i] for i in range(result.size())] == _expected_projection(
        source, selection
    )
    assert (source.layout, source.carrier, source.version) == before[:3]
    assert tuple(source[i] for i in range(FULL_SIZE)) == before[3]


def test_projection_preserves_within_tile_permutation_and_backward_values() -> None:
    carrier = _tiled()
    source = sw.Tensor(carrier, 0, _within_tile_permutation())
    selection = TileSelection(([1, 0], [1]))

    projected = carrier.project(source, selection).wait()

    assert [projected[index] for index in range(projected.size())] == (
        _expected_projection(source, selection)
    )
    gradient_values = [float(index + 1) for index in range(projected.size())]
    projected.backward(
        sw.Tensor(
            sw.Generic(gradient_values, dtype=sw.DType.Float32),
            0,
            projected.layout,
        )
    )
    assert source.grad is not None
    assert source.grad.layout == source.layout
    expected = [0.0] * FULL_SIZE
    compact_shape = (4, 2)
    for compact_index in range(projected.size()):
        compact_x = compact_index % compact_shape[0]
        compact_y = compact_index // compact_shape[0]
        source_x = selection.per_axis[0][compact_x // TILE[0]] * TILE[0] + (
            compact_x % TILE[0]
        )
        source_y = selection.per_axis[1][compact_y // TILE[1]] * TILE[1] + (
            compact_y % TILE[1]
        )
        expected[source_x + FULL_SHAPE[0] * source_y] = gradient_values[compact_index]
    assert [source.grad[index] for index in range(FULL_SIZE)] == expected


@pytest.mark.parametrize("kind", ["spans", "remaps", "aliases", "omits"])
def test_projection_rejects_incompatible_tile_footprints_before_work(
    kind: str,
) -> None:
    carrier = _tiled()
    source = sw.Tensor(carrier, 0, _incompatible_layout(kind))
    selection = TileSelection(([0], [0]))
    destination = sw.Generic([], dtype=sw.DType.Float32)
    before = (
        carrier.version,
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        tuple(carrier[index] for index in range(FULL_SIZE)),
    )

    with pytest.raises(ValueError, match="tile footprints"):
        carrier.project(source, selection, destination)

    assert destination.size() == 0
    assert not destination.is_owned()
    assert (
        carrier.version,
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        tuple(carrier[index] for index in range(FULL_SIZE)),
    ) == before


@pytest.mark.parametrize(
    ("tensor", "selection", "error"),
    [
        (object(), TileSelection(([0], [0])), TypeError),
        (make_tensor([1.0, 2.0]), TileSelection(([0], [0])), TypeError),
        (None, object(), TypeError),
    ],
)
def test_projection_rejects_non_tensor_or_non_selection(
    tensor: object, selection: object, error: type[Exception]
) -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    with pytest.raises(error):
        carrier.project(tensor if tensor is not None else source, selection)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "selection",
    [TileSelection(([0],)), TileSelection(([2], [0])), TileSelection(([0], [0, 1, 2]))],
)
def test_projection_rejects_rank_range_or_geometry_mismatch(
    selection: TileSelection,
) -> None:
    carrier = _tiled()
    with pytest.raises(ValueError, match=r"rank|range|grid|geometry"):
        carrier.project(_tensor(carrier), selection)


def test_projection_destination_validation_is_preflight() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    selection = TileSelection(([0], [0]))
    before = (carrier.primary_tiles, carrier.secondary_tiles, carrier.version)

    with pytest.raises(TypeError):
        carrier.project(source, selection, destination=carrier)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        carrier.project(
            source, selection, destination=sw.Generic([0] * 4, dtype=sw.DType.Int32)
        )
    with pytest.raises(ValueError, match=r"extent|size|capacity|small"):
        carrier.project(
            source,
            selection,
            destination=sw.Generic([0.0] * 2, dtype=sw.DType.Float32),
        )
    with pytest.raises(RuntimeError):
        carrier.project(
            source,
            selection,
            destination=sw.Generic([0.0] * 4, mutable=False, dtype=sw.DType.Float32),
        )

    assert (carrier.primary_tiles, carrier.secondary_tiles, carrier.version) == before


def test_projection_rejects_released_source_before_work() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    carrier.release()
    with pytest.raises(RuntimeError):
        carrier.project(source, TileSelection(([0], [0])))


class _PendingCompletion:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self._event.wait(timeout=5)
        if self._error is not None:
            raise self._error

    def finish(self, error: BaseException | None = None) -> None:
        self._error = error
        self._event.set()


def _pending_pair(
    completion: _PendingCompletion,
) -> tuple[type[sw.Carrier], type[sw.Carrier]]:
    source_class = type("ProjectionPendingPrimary", (sw.Carrier,), {})
    destination_class = type("ProjectionPendingSecondary", (sw.Carrier,), {})

    def allocate(carrier: Any, slots: int) -> None:
        carrier.values = [0.0] * slots

    def release(carrier: Any) -> None:
        carrier.values = []

    def size(carrier: Any) -> int:
        return len(carrier.values)

    def read(carrier: Any, index: int) -> object:
        return carrier.values[index]

    def write(carrier: Any, index: int, value: object) -> None:
        carrier.values[index] = value

    storage = StorageProvider(
        lambda _carrier, dtype: dtype is sw.DType.Float32,
        size,
        allocate,
        release,
        read,
        write,
    )

    def prepare(request: TransferRequest) -> PreparedTransfer:
        source = cast(Any, request.tensor.carrier)
        destination = cast(Any, request.destination)

        def submit() -> _PendingCompletion:
            destination.values[: request.physical_span] = source.values[
                : request.physical_span
            ]
            return completion

        return PreparedTransfer(submit)

    forward = TransferRoute(
        source_class,
        destination_class,
        "secondary-to-primary",
        TransferProvider(prepare),
    )
    reverse = TransferRoute(
        destination_class,
        source_class,
        "primary-to-secondary",
        TransferProvider(prepare),
    )
    register_carrier_definition(
        source_class, CarrierDefinition(storage, transfers=(forward,))
    )
    register_carrier_definition(
        destination_class, CarrierDefinition(storage, transfers=(reverse,))
    )
    return source_class, destination_class


def test_projection_eagerly_returns_incomplete_handle_and_rejects_as_operand() -> None:
    completion = _PendingCompletion()
    primary_class, secondary_class = _pending_pair(completion)
    primary = primary_class(0, dtype=sw.DType.Float32)
    secondary = secondary_class(FULL_SIZE, dtype=sw.DType.Float32)
    for index, value in enumerate(_seed_values()):
        secondary[index] = value
    carrier = TiledEvictable(primary, secondary, GRID, TILE)
    handle = carrier.project(_tensor(carrier), TileSelection(([1], [1])))

    assert isinstance(handle, AwaitProjection)
    assert not handle.done
    assert not isinstance(handle, sw.Tensor)
    assert not hasattr(handle, "__await__")
    with pytest.raises(TypeError):
        sw.add(handle, handle)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match=r"wait|pending"):
        carrier.release()
    with pytest.raises(RuntimeError, match=r"pinned|pending|wait"):
        carrier[10] = -1.0

    unrelated = make_tensor([1.0, 2.0])
    assert [sw.add(unrelated, unrelated)[index] for index in range(2)] == [2.0, 4.0]

    completion.finish()
    assert handle.wait()[0] == 22.0


def test_many_delayed_projections_use_bounded_selection_workers() -> None:
    background = import_module("strideweave.carriers._background")
    completion = _PendingCompletion()
    primary_class, secondary_class = _pending_pair(completion)
    carriers: list[TiledEvictable] = []
    handles: list[AwaitProjection] = []
    for _ in range(24):
        primary = primary_class(0, dtype=sw.DType.Float32)
        secondary = secondary_class(FULL_SIZE, dtype=sw.DType.Float32)
        for index, value in enumerate(_seed_values()):
            secondary[index] = value
        carrier = TiledEvictable(primary, secondary, GRID, TILE)
        carriers.append(carrier)
        handles.append(carrier.project(_tensor(carrier), TileSelection(([0], [0]))))

    executor = cast(Any, background)._TILED_SELECTION_EXECUTOR
    workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(executor.thread_name_prefix)
    ]
    assert len(workers) <= executor.max_workers
    assert all(not handle.done for handle in handles)

    dropped = handles.pop()
    dropped_reference = weakref.ref(dropped)
    del dropped
    gc.collect()
    assert dropped_reference() is None

    completion.finish()
    results = [handle.wait() for handle in handles]
    assert all(result[0] == 0.0 for result in results)
    deadline = time.monotonic() + 5
    while (
        any(cast(Any, carrier)._pending_requests for carrier in carriers)
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert all(cast(Any, carrier)._pending_requests == 0 for carrier in carriers)


def test_projection_failure_is_atomic_and_repeated_wait_raises_same_error() -> None:
    completion = _PendingCompletion()
    primary_class, secondary_class = _pending_pair(completion)
    primary = primary_class(0, dtype=sw.DType.Float32)
    secondary = secondary_class(FULL_SIZE, dtype=sw.DType.Float32)
    for index, value in enumerate(_seed_values()):
        secondary[index] = value
    carrier = TiledEvictable(primary, secondary, GRID, TILE)
    source = _tensor(carrier)
    before = tuple(source[index] for index in range(FULL_SIZE))
    handle = carrier.project(source, TileSelection(([0], [0])))
    failure = RuntimeError("projection provider failed")
    completion.finish(failure)

    observed: list[BaseException] = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            handle.wait()
        observed.append(caught.value)
    assert observed[0] is observed[1] is failure
    assert tuple(source[index] for index in range(FULL_SIZE)) == before
    assert carrier.valid_tiles == carrier.secondary_tiles
    assert not carrier.is_released()


def test_projection_backward_scatter_adds_full_shaped_implicit_zero_gradient() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    result = carrier.project(source, TileSelection(([1], [0]))).wait()
    gradient = sw.Tensor(
        sw.Generic([1.0] * result.size(), dtype=sw.DType.Float32), 0, result.layout
    )
    result.backward(gradient)

    assert source.grad is not None
    assert type(source.grad.carrier) is TiledEvictable
    assert source.grad.layout == source.layout
    expected = [0.0] * FULL_SIZE
    for j in range(2):
        for i in range(2, 4):
            expected[i + 4 * j] = 1.0
    assert [source.grad[index] for index in range(FULL_SIZE)] == expected
    assert source.grad.carrier.primary_tiles == sw.TileSet(((1, 0),))
    assert source.grad.carrier.valid_tiles == sw.TileSet(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    assert source.grad.carrier.tile_state((0, 0)).location == "implicit"


def test_projection_then_ordinary_operation_has_one_visible_gather_node() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    projected = carrier.project(source, TileSelection(([0], [0]))).wait()

    assert type(projected.autograd_ctx).__name__ == "_ProjectionOperation"
    assert cast(Any, projected.autograd_ctx).inputs() == (source,)
    result = sw.add(projected, projected)
    result.backward(
        sw.Tensor(
            sw.Generic([1.0] * result.size(), dtype=sw.DType.Float32), 0, result.layout
        )
    )

    assert source.grad is not None
    assert [source.grad[index] for index in (0, 1, 4, 5)] == [2.0] * 4


def test_projection_saved_version_rejects_later_source_mutation() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    result = carrier.project(source, TileSelection(([0], [0]))).wait()
    source[0] = 99.0

    with pytest.raises(RuntimeError, match=r"modified in-place|version"):
        result.backward(
            sw.Tensor(
                sw.Generic([1.0] * result.size(), dtype=sw.DType.Float32),
                0,
                result.layout,
            )
        )


def test_overlapping_projection_gradients_accumulate() -> None:
    carrier = _tiled()
    source = _tensor(carrier)
    first = carrier.project(source, TileSelection(([0], [0]))).wait()
    second = carrier.project(source, TileSelection(([0], [0]))).wait()
    first.backward(
        sw.Tensor(
            sw.Generic([1.0] * first.size(), dtype=sw.DType.Float32), 0, first.layout
        )
    )
    second.backward(
        sw.Tensor(
            sw.Generic([2.0] * second.size(), dtype=sw.DType.Float32), 0, second.layout
        )
    )

    assert source.grad is not None
    assert source.grad[0] == 3.0
    assert source.grad[1] == 3.0
    assert source.grad[4] == 3.0
    assert source.grad[5] == 3.0
    assert source.grad[6] == 0.0
    assert isinstance(source.grad.carrier, sw.TiledEvictable)
    gradient_carrier = cast(sw.TiledEvictable, source.grad.carrier)
    assert gradient_carrier.primary_tiles == sw.TileSet(((0, 0),))
    assert gradient_carrier.tile_state((1, 1)).location == "implicit"
