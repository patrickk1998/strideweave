"""Focused functional scatter tests for ``TiledEvictable``.

The tests use a two-dimensional, first-mode-fastest tiled tensor so that the
selection order and the within-tile affine mapping are both observable.  The
scatter contract intentionally distinguishes an ordinary partial result (only
selected tiles are valid) from the implicit-zero tiled gradients produced by
autograd.
"""

from __future__ import annotations

import threading
from importlib import import_module
from math import prod
from typing import Any, cast

import pytest

import strideweave as sw

GRID = (2, 2)
TILE = (2, 2)
FULL_SHAPE = (4, 4)
FULL_SIZE = 16


def _encode(coordinate: tuple[int, ...], shape: tuple[int, ...]) -> int:
    stride = 1
    index = 0
    for component, extent in zip(coordinate, shape, strict=True):
        index += component * stride
        stride *= extent
    return index


def _decode(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinate: list[int] = []
    for extent in shape:
        coordinate.append(index % extent)
        index //= extent
    return tuple(coordinate)


def _layout(shape: tuple[int, ...]) -> sw.Layout:
    stride: list[int] = []
    product = 1
    for extent in shape:
        stride.append(product)
        product *= extent
    return sw.Layout(sw.Shape(shape), sw.Stride(stride))


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


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _tiled_carrier(tensor: sw.Tensor) -> sw.TiledEvictable:
    assert isinstance(tensor.carrier, sw.TiledEvictable)
    return cast(sw.TiledEvictable, tensor.carrier)


def _tiled(values: list[float] | None = None) -> sw.TiledEvictable:
    return sw.TiledEvictable(
        sw.Generic([] if values is None else values, dtype=sw.DType.Float32),
        sw.Generic([], dtype=sw.DType.Float32),
        GRID,
        TILE,
    )


def _full_tensor(
    values: list[float] | None = None,
) -> tuple[sw.TiledEvictable, sw.Tensor]:
    carrier = _tiled(list(range(FULL_SIZE)) if values is None else values)
    return carrier, sw.Tensor(carrier, 0, _layout(FULL_SHAPE))


def _compact_shape(selection: sw.TileSelection) -> tuple[int, ...]:
    return tuple(
        count * extent
        for count, extent in zip(selection.compact_grid_shape, TILE, strict=True)
    )


def _compact_tensor(
    selection: sw.TileSelection,
    values: list[float] | None = None,
) -> sw.Tensor:
    shape = _compact_shape(selection)
    size = 1
    for extent in shape:
        size *= extent
    compact_values = list(range(100, 100 + size)) if values is None else values
    assert len(compact_values) == size
    return sw.Tensor(
        sw.Generic(compact_values, dtype=sw.DType.Float32),
        0,
        _layout(shape),
    )


def _source_flat_indices(selection: sw.TileSelection) -> list[int]:
    """Return source slots in dense compact logical order."""

    compact_shape = _compact_shape(selection)
    result: list[int] = []
    for compact_index in range(prod(compact_shape)):
        compact = _decode(compact_index, compact_shape)
        logical = tuple(
            selection.per_axis[axis][compact[axis] // TILE[axis]] * TILE[axis]
            + compact[axis] % TILE[axis]
            for axis in range(len(compact_shape))
        )
        result.append(_encode(logical, FULL_SHAPE))
    return result


def _selected_values(tensor: sw.Tensor, selection: sw.TileSelection) -> list[Any]:
    return [tensor[index] for index in _source_flat_indices(selection)]


def _snapshot(tensor: sw.Tensor) -> tuple[list[Any], Any, Any, Any, Any]:
    carrier = _tiled_carrier(tensor)
    return (
        _values(tensor),
        carrier.version,
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
    )


def _assert_snapshot(tensor: sw.Tensor, snapshot: tuple[Any, ...]) -> None:
    values, version, primary_tiles, secondary_tiles, valid_tiles = snapshot
    carrier = _tiled_carrier(tensor)
    assert _values(tensor) == values
    assert carrier.version == version
    assert carrier.primary_tiles == primary_tiles
    assert carrier.secondary_tiles == secondary_tiles
    assert carrier.valid_tiles == valid_tiles


def test_replace_scatter_is_fresh_full_shaped_and_preserves_selection_order() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([1, 0], [1]))
    compact = _compact_tensor(selection)
    template_before = _snapshot(template)
    compact_before = _values(compact)
    compact_version = compact.carrier.version

    handle = carrier.scatter(compact, template, selection)
    assert isinstance(handle, sw.AwaitMove)
    result = handle.wait()
    assert result is handle.wait()

    assert isinstance(result, sw.Tensor)
    assert result is not template
    assert result.carrier is not carrier
    assert isinstance(result.carrier, sw.TiledEvictable)
    result_carrier = _tiled_carrier(result)
    assert result.layout == template.layout
    assert result.size() == FULL_SIZE
    assert result.dtype() is compact.dtype() is sw.DType.Float32
    assert result_carrier.version == 1
    assert result_carrier.valid_tiles == selection.tile_set
    assert result_carrier.dirty_tiles == selection.tile_set
    for coordinate in selection.coordinates:
        state = result_carrier.tile_state(coordinate)
        assert state.location == "primary"
        assert state.valid is True
        assert state.dirty is True
    for coordinate in {(0, 0), (1, 0), (0, 1), (1, 1)} - set(selection.coordinates):
        assert result_carrier.tile_state(coordinate).valid is False

    assert _selected_values(result, selection) == compact_before
    _assert_snapshot(template, template_before)
    assert _values(compact) == compact_before
    assert compact.carrier.version == compact_version

    projected = result_carrier.project(result, selection).wait()
    assert _values(projected) == compact_before
    assert projected.layout == compact.layout


def test_add_scatter_accumulates_selected_template_tiles_only() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([0, 1], [0]))
    compact_values = [1.0 + index for index in range(8)]
    compact = _compact_tensor(selection, compact_values)
    template_before = _values(template)
    template_version = template.carrier.version
    compact_version = compact.carrier.version

    result = carrier.scatter(compact, template, selection, reduction="add").wait()
    result_carrier = _tiled_carrier(result)

    expected = [
        value + update
        for value, update in zip(
            _selected_values(template, selection), compact_values, strict=True
        )
    ]
    assert _selected_values(result, selection) == expected
    assert result_carrier.valid_tiles == selection.tile_set
    assert result_carrier.dirty_tiles == selection.tile_set
    assert _values(template) == template_before
    assert _values(compact) == compact_values
    assert template.carrier.version == template_version
    assert compact.carrier.version == compact_version


def test_scatter_preserves_within_tile_permutation_values_metadata_and_vjp() -> None:
    carrier = _tiled([float(index) for index in range(FULL_SIZE)])
    template = sw.Tensor(carrier, 0, _within_tile_permutation())
    selection = sw.TileSelection(([1, 0], [1]))
    compact_values = [float(100 + index) for index in range(8)]
    compact = _compact_tensor(selection, compact_values)

    replaced = carrier.scatter(compact, template, selection).wait()
    replaced_carrier = _tiled_carrier(replaced)
    projected = replaced_carrier.project(replaced, selection).wait()

    assert _values(projected) == compact_values
    assert replaced.layout == template.layout
    assert replaced.offset == template.offset == 0
    assert replaced.dtype() is template.dtype() is sw.DType.Float32
    assert replaced_carrier.grid_shape == carrier.grid_shape
    assert replaced_carrier.tile_shape == carrier.tile_shape
    assert type(replaced_carrier.primary) is type(carrier.primary)
    assert type(replaced_carrier.secondary) is type(carrier.secondary)

    added = carrier.scatter(compact, template, selection, reduction="add").wait()
    assert _selected_values(added, selection) == [
        before + update
        for before, update in zip(
            _selected_values(template, selection), compact_values, strict=True
        )
    ]
    incoming = sw.Tensor(
        sw.Generic(
            [float(index + 1) for index in range(FULL_SIZE)],
            dtype=sw.DType.Float32,
        ),
        0,
        _within_tile_permutation(),
    )
    added.backward(incoming)

    assert compact.grad is not None
    assert template.grad is not None
    assert _values(compact.grad) == _selected_values(incoming, selection)
    assert template.grad.layout == template.layout
    assert _values(template.grad) == _values(incoming)


@pytest.mark.parametrize("kind", ["spans", "remaps", "aliases", "omits"])
def test_scatter_rejects_incompatible_tile_footprints_before_work(
    kind: str,
) -> None:
    carrier = _tiled([float(index) for index in range(FULL_SIZE)])
    template = sw.Tensor(carrier, 0, _incompatible_layout(kind))
    selection = sw.TileSelection(([0], [0]))
    compact = _compact_tensor(selection)
    before = _snapshot(template)
    compact_before = _values(compact)

    with pytest.raises(ValueError, match="tile footprints"):
        carrier.scatter(compact, template, selection)

    _assert_snapshot(template, before)
    assert _values(compact) == compact_before
    assert not compact.carrier.is_owned()


def test_add_scatter_accepts_declared_implicit_zero_tiles() -> None:
    source_carrier, _ = _full_tensor()
    zero_carrier = source_carrier._implicit_zero_like()
    template = sw.Tensor(zero_carrier, 0, _layout(FULL_SHAPE))
    selection = sw.TileSelection(([1], [0, 1]))
    compact_values = [float(index + 1) for index in range(8)]
    compact = _compact_tensor(selection, compact_values)

    result = zero_carrier.scatter(compact, template, selection, reduction="add").wait()
    result_carrier = _tiled_carrier(result)

    assert _selected_values(result, selection) == compact_values
    assert result_carrier.valid_tiles == selection.tile_set
    assert result_carrier.dirty_tiles == selection.tile_set
    assert _values(template) == [0.0] * FULL_SIZE


def test_replace_scatter_can_initialize_selected_tiles_of_invalid_template() -> None:
    template_carrier = _tiled()
    template = sw.Tensor(template_carrier, 0, _layout(FULL_SHAPE))
    selection = sw.TileSelection(([0], [1]))
    compact = _compact_tensor(selection)

    result = template_carrier.scatter(compact, template, selection).wait()

    assert _tiled_carrier(result).valid_tiles == selection.tile_set
    assert _selected_values(result, selection) == _values(compact)


def test_add_scatter_rejects_an_uninitialized_selected_template_tile() -> None:
    template_carrier = _tiled()
    template = sw.Tensor(template_carrier, 0, _layout(FULL_SHAPE))
    selection = sw.TileSelection(([0], [0]))
    compact = _compact_tensor(selection, [3.0, 4.0, 5.0, 6.0])

    with pytest.raises(RuntimeError):
        template_carrier.scatter(compact, template, selection, reduction="add")


@pytest.mark.parametrize(
    ("compact_factory", "selection_factory", "reduction", "error"),
    [
        (
            lambda selection: _compact_tensor(selection, [1.0] * 8),
            lambda: sw.TileSelection(([0, 1], [0])),
            "overwrite",
            ValueError,
        ),
        (
            lambda selection: sw.Tensor(
                sw.Generic([1.0] * 8, dtype=sw.DType.Float32),
                0,
                _layout((2, 4)),
            ),
            lambda: sw.TileSelection(([0, 1], [0])),
            "replace",
            ValueError,
        ),
        (
            lambda selection: sw.Tensor(
                sw.Generic([1] * 8, dtype=sw.DType.Int32),
                0,
                _layout((4, 2)),
            ),
            lambda: sw.TileSelection(([0, 1], [0])),
            "replace",
            TypeError,
        ),
        (
            lambda selection: _compact_tensor(selection, [1.0] * 8),
            lambda: sw.TileSelection(([0, 2], [0])),
            "replace",
            ValueError,
        ),
    ],
)
def test_scatter_validation_is_atomic(
    compact_factory: Any,
    selection_factory: Any,
    reduction: str,
    error: type[Exception],
) -> None:
    carrier, template = _full_tensor()
    valid_selection = sw.TileSelection(([0, 1], [0]))
    compact = _compact_tensor(valid_selection)
    before_template = _snapshot(template)
    before_compact = _values(compact)
    invalid_selection = selection_factory()
    invalid_compact = compact_factory(valid_selection)
    invalid_compact_before = _values(invalid_compact)

    with pytest.raises(error):
        carrier.scatter(
            invalid_compact, template, invalid_selection, reduction=reduction
        )

    _assert_snapshot(template, before_template)
    assert _values(compact) == before_compact
    assert _values(invalid_compact) == invalid_compact_before


def test_scatter_rejects_non_tensor_and_foreign_template_before_work() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([0, 1], [0]))
    compact = _compact_tensor(selection)
    before = _snapshot(template)

    with pytest.raises(TypeError):
        carrier.scatter(object(), template, selection)
    foreign = sw.Tensor(
        sw.Generic([0.0] * FULL_SIZE, dtype=sw.DType.Float32), 0, _layout(FULL_SHAPE)
    )
    with pytest.raises(TypeError):
        carrier.scatter(compact, foreign, selection)

    _assert_snapshot(template, before)


def test_replace_scatter_vjp_gathers_compact_gradient_and_masks_template() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([1, 0], [1]))
    compact = _compact_tensor(selection)
    result = carrier.scatter(compact, template, selection).wait()
    assert type(result.autograd_ctx).__name__ == "_ScatterOperation"
    incoming_values = [float(index + 1) for index in range(FULL_SIZE)]
    incoming = sw.Tensor(
        sw.Generic(incoming_values, dtype=sw.DType.Float32),
        0,
        _layout(FULL_SHAPE),
    )

    result.backward(incoming)

    assert compact.grad is not None
    assert template.grad is not None
    assert _values(compact.grad) == [
        incoming_values[i] for i in _source_flat_indices(selection)
    ]
    expected_template = incoming_values[:]
    for index in _source_flat_indices(selection):
        expected_template[index] = 0.0
    assert _values(template.grad) == expected_template
    template_gradient_carrier = _tiled_carrier(template.grad)
    assert template_gradient_carrier.primary_tiles == sw.TileSet(
        tuple(
            coordinate
            for coordinate in ((0, 0), (1, 0), (0, 1), (1, 1))
            if coordinate not in selection.coordinates
        )
    )
    for coordinate in selection.coordinates:
        assert template_gradient_carrier.tile_state(coordinate).location == "implicit"


def test_add_scatter_vjp_reaches_both_contributing_inputs() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([0, 1], [0]))
    compact = _compact_tensor(selection)
    result = carrier.scatter(compact, template, selection, reduction="add").wait()
    incoming = sw.Tensor(
        sw.Generic([1.0] * FULL_SIZE, dtype=sw.DType.Float32),
        0,
        _layout(FULL_SHAPE),
    )

    result.backward(incoming)

    assert compact.grad is not None
    assert template.grad is not None
    assert _values(compact.grad) == [1.0] * compact.size()
    assert _values(template.grad) == [1.0] * FULL_SIZE


def test_scatter_saved_versions_reject_later_compact_mutation() -> None:
    carrier, template = _full_tensor()
    selection = sw.TileSelection(([0], [0]))
    compact = _compact_tensor(selection)
    result = carrier.scatter(compact, template, selection).wait()
    compact[0] = -1.0
    incoming = sw.Tensor(
        sw.Generic([1.0] * FULL_SIZE, dtype=sw.DType.Float32),
        0,
        _layout(FULL_SHAPE),
    )

    with pytest.raises(RuntimeError, match=r"modified in-place|version"):
        result.backward(incoming)


def test_many_delayed_scatters_use_bounded_selection_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background = import_module("strideweave.carriers._background")
    selection_ops = import_module("strideweave.carriers.tiled_evictable.selection_ops")
    executor = cast(Any, background)._TILED_SELECTION_EXECUTOR
    original = cast(Any, selection_ops)._compact_values
    gate = threading.Event()
    active = 0
    active_condition = threading.Condition()

    def delayed(compact: sw.Tensor, token: int) -> list[object]:
        nonlocal active
        with active_condition:
            active += 1
            active_condition.notify_all()
        gate.wait(timeout=5)
        return original(compact, token)

    monkeypatch.setattr(selection_ops, "_compact_values", delayed)
    requests: list[tuple[sw.Tensor, sw.AwaitMove[sw.Tensor]]] = []
    for _ in range(24):
        carrier, template = _full_tensor()
        selection = sw.TileSelection(([0], [0]))
        compact = _compact_tensor(selection)
        requests.append((compact, carrier.scatter(compact, template, selection)))

    with active_condition:
        assert active_condition.wait_for(
            lambda: active == executor.max_workers,
            timeout=2,
        )
    workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(executor.thread_name_prefix)
    ]
    assert len(workers) <= executor.max_workers
    assert all(not handle.done for _, handle in requests)

    gate.set()
    results = [handle.wait() for _, handle in requests]
    assert all(
        _selected_values(result, sw.TileSelection(([0], [0]))) for result in results
    )
    assert all(not compact.carrier.is_owned() for compact, _ in requests)
