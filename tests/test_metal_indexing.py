from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.base import Carrier
from strideweave.carriers.metal import _indexing_executor as indexing_executor
from strideweave.carriers.metal import indexing_ops
from strideweave.carriers.metal.indexing_ops import MetalIndexingOperation
from strideweave.carriers.operation_policy import resolve_operation_plan

pytestmark = pytest.mark.metal

_DISPATCH_NAMES = (
    "_sort_indices",
    "_sort_values",
    "_topk_indices",
    "_topk_values",
    "gather",
    "scatter",
    "scatter_add",
)


def _tensor_from_storage(
    storage: list[Any],
    layout: sw.Layout,
    *,
    metal: bool,
    offset: int = 0,
    dtype: sw.DType = sw.DType.Float32,
) -> sw.Tensor:
    if metal:
        carrier = sw.Metal(len(storage), dtype=dtype)
        for index, value in enumerate(storage):
            carrier[index] = value
    else:
        carrier = sw.Generic(storage, dtype=dtype)
    return sw.Tensor(carrier, offset, layout)


def _tensor_from_logical(
    values: list[Any],
    layout: sw.Layout,
    *,
    metal: bool,
    offset: int = 0,
    dtype: sw.DType = sw.DType.Float32,
) -> sw.Tensor:
    zero: Any = 0 if dtype is sw.DType.Int32 else -91.0
    storage = [zero] * (offset + layout.cosize)
    for logical_index, value in enumerate(values):
        storage[offset + layout.index(logical_index)] = value
    return _tensor_from_storage(
        storage,
        layout,
        metal=metal,
        offset=offset,
        dtype=dtype,
    )


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _canonical(shape: sw.Shape) -> sw.Layout:
    from strideweave.carriers.operation_helpers import _canonical_layout_for_shape

    return _canonical_layout_for_shape(shape)


def _assert_matches(actual: sw.Tensor, expected: sw.Tensor) -> None:
    assert actual.dtype() is expected.dtype()
    assert actual.layout == expected.layout
    assert _values(actual) == pytest.approx(
        _values(expected), rel=2e-6, abs=2e-6, nan_ok=True
    )


def test_every_indexing_dispatch_is_fresh_and_metal_owned() -> None:
    carrier = sw.Metal.__new__(sw.Metal)
    Carrier.__init__(carrier)

    for name in _DISPATCH_NAMES:
        first = carrier.dispatch_op(name)
        second = carrier.dispatch_op(name)
        assert isinstance(first, MetalIndexingOperation)
        assert isinstance(second, MetalIndexingOperation)
        assert first is not second
        assert first._dispatch_carrier_class is sw.Metal
        assert first._operation_name == name


def test_gather_scatter_and_scatter_add_match_generic() -> None:
    base_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 5]))
    index_layout = sw.Layout(sw.Shape(2), sw.Stride(2))
    update_layout = _canonical(sw.Shape([2, 2]))
    base_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    index_values = [2, 0]
    update_values = [10.0, 20.0, 30.0, 40.0]

    metal_base = _tensor_from_logical(base_values, base_layout, metal=True, offset=2)
    generic_base = _tensor_from_logical(base_values, base_layout, metal=False, offset=2)
    metal_indices = _tensor_from_logical(
        index_values,
        index_layout,
        metal=True,
        offset=1,
        dtype=sw.DType.Int32,
    )
    generic_indices = _tensor_from_logical(
        index_values,
        index_layout,
        metal=False,
        offset=1,
        dtype=sw.DType.Int32,
    )
    metal_updates = _tensor_from_logical(update_values, update_layout, metal=True)
    generic_updates = _tensor_from_logical(update_values, update_layout, metal=False)

    _assert_matches(
        sw.gather(metal_base, metal_indices, 1),
        sw.gather(generic_base, generic_indices, 1),
    )
    _assert_matches(
        sw.scatter(metal_base, metal_indices, metal_updates, 1),
        sw.scatter(generic_base, generic_indices, generic_updates, 1),
    )
    _assert_matches(
        sw.scatter_add(metal_base, metal_indices, metal_updates, 1),
        sw.scatter_add(generic_base, generic_indices, generic_updates, 1),
    )
    assert _values(metal_base) == base_values


def test_sort_and_topk_match_generic_special_value_order() -> None:
    values = [
        3.0,
        float("nan"),
        -0.0,
        0.0,
        2.0,
        float("nan"),
        -float("inf"),
        float("inf"),
    ]
    layout = sw.Layout(sw.Shape(8), sw.Stride(2))
    metal = _tensor_from_logical(values, layout, metal=True, offset=1)
    generic = _tensor_from_logical(values, layout, metal=False, offset=1)

    for descending in (False, True):
        actual = sw.sort(metal, descending=descending)
        expected = sw.sort(generic, descending=descending)
        _assert_matches(actual.values, expected.values)
        assert _values(actual.indices) == _values(expected.indices)
    for largest in (False, True):
        actual = sw.topk(metal, 4, largest=largest)
        expected = sw.topk(generic, 4, largest=largest)
        _assert_matches(actual.values, expected.values)
        assert _values(actual.indices) == _values(expected.indices)

    ascending = sw.sort(metal)
    assert math.copysign(1.0, ascending.values[1]) == -1.0
    assert math.copysign(1.0, ascending.values[2]) == 1.0


def test_gather_repeated_index_vjp_accumulates_on_metal() -> None:
    tensor = _tensor_from_logical([1.0, 2.0, 3.0], _canonical(sw.Shape(3)), metal=True)
    indices = _tensor_from_storage(
        [1],
        sw.Layout(sw.Shape(3), sw.Stride(0)),
        metal=True,
        dtype=sw.DType.Int32,
    )

    result = sw.gather(tensor, indices, 0)
    result.backward(_tensor_from_logical([1.0, 2.0, 4.0], result.layout, metal=True))

    assert _values(result) == [2.0, 2.0, 2.0]
    assert tensor.grad is not None
    assert type(tensor.grad.carrier) is sw.Metal
    assert _values(tensor.grad) == [0.0, 7.0, 0.0]
    with pytest.raises(RuntimeError, match="non-differentiable"):
        _ = indices.grad


@pytest.mark.parametrize(
    ("operation", "expected_base_gradient"),
    ((sw.scatter, [0.0, 2.0, 0.0]), (sw.scatter_add, [1.0, 2.0, 4.0])),
)
def test_scatter_value_vjps_remain_metal(
    operation: Any,
    expected_base_gradient: list[float],
) -> None:
    layout = _canonical(sw.Shape(3))
    base = _tensor_from_logical([1.0, 2.0, 3.0], layout, metal=True)
    indices = _tensor_from_logical(
        [2, 0],
        _canonical(sw.Shape(2)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    updates = _tensor_from_logical([20.0, 10.0], _canonical(sw.Shape(2)), metal=True)

    result = operation(base, indices, updates, 0)
    result.backward(_tensor_from_logical([1.0, 2.0, 4.0], result.layout, metal=True))

    assert base.grad is not None
    assert updates.grad is not None
    assert type(base.grad.carrier) is sw.Metal
    assert type(updates.grad.carrier) is sw.Metal
    assert _values(base.grad) == expected_base_gradient
    assert _values(updates.grad) == [4.0, 1.0]


def test_scatter_add_uses_first_mode_fast_binary32_order() -> None:
    base = _tensor_from_logical([0.0], _canonical(sw.Shape(1)), metal=True)
    indices = _tensor_from_logical(
        [0, 0, 0],
        _canonical(sw.Shape(3)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    updates = _tensor_from_logical(
        [1.0e8, 1.0, -1.0e8], _canonical(sw.Shape(3)), metal=True
    )

    assert _values(sw.scatter_add(base, indices, updates, 0)) == [0.0]


def test_sort_and_topk_value_vjps_route_source_ordinals_on_metal() -> None:
    sort_tensor = _tensor_from_logical(
        [3.0, 1.0, 3.0, 2.0], _canonical(sw.Shape(4)), metal=True
    )
    sorted_result = sw.sort(sort_tensor)
    sorted_result.values.backward(
        _tensor_from_logical(
            [10.0, 20.0, 30.0, 40.0], sorted_result.values.layout, metal=True
        )
    )
    assert sort_tensor.grad is not None
    assert type(sort_tensor.grad.carrier) is sw.Metal
    assert _values(sort_tensor.grad) == [30.0, 10.0, 40.0, 20.0]

    topk_tensor = _tensor_from_logical(
        [4.0, 7.0, 7.0, 2.0], _canonical(sw.Shape(4)), metal=True
    )
    topk_result = sw.topk(topk_tensor, 2)
    topk_result.values.backward(
        _tensor_from_logical([10.0, 20.0], topk_result.values.layout, metal=True)
    )
    assert topk_tensor.grad is not None
    assert type(topk_tensor.grad.carrier) is sw.Metal
    assert _values(topk_tensor.grad) == [0.0, 10.0, 20.0, 0.0]
    assert topk_result.indices.dtype() is sw.DType.Int32
    with pytest.raises(RuntimeError, match="non-differentiable"):
        _ = topk_result.indices.grad


def test_hierarchical_gather_sort_and_topk_match_generic() -> None:
    gather_shape = sw.Shape([2, [3, 2], 2])
    gather_layout = sw.Layout.permute(_canonical(gather_shape), [2, 1, 0])
    gather_values = [float(index - 5) for index in range(gather_layout.size)]
    index_layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([2, 5]))
    index_values = [5, 0, 3, 1]
    metal_source = _tensor_from_logical(gather_values, gather_layout, metal=True)
    generic_source = _tensor_from_logical(gather_values, gather_layout, metal=False)
    metal_indices = _tensor_from_logical(
        index_values,
        index_layout,
        metal=True,
        offset=1,
        dtype=sw.DType.Int32,
    )
    generic_indices = _tensor_from_logical(
        index_values,
        index_layout,
        metal=False,
        offset=1,
        dtype=sw.DType.Int32,
    )

    _assert_matches(
        sw.gather(metal_source, metal_indices, 1),
        sw.gather(generic_source, generic_indices, 1),
    )

    selection_shape = sw.Shape([2, [2, 2]])
    selection_layout = sw.Layout(selection_shape, sw.Stride([3, [1, 7]]))
    selection_values = [4.0, 1.0, 3.0, 2.0, 8.0, 6.0, 7.0, 5.0]
    metal_selection = _tensor_from_logical(
        selection_values, selection_layout, metal=True, offset=2
    )
    generic_selection = _tensor_from_logical(
        selection_values, selection_layout, metal=False, offset=2
    )
    actual_sort = sw.sort(metal_selection, axis=1)
    expected_sort = sw.sort(generic_selection, axis=1)
    _assert_matches(actual_sort.values, expected_sort.values)
    assert _values(actual_sort.indices) == _values(expected_sort.indices)
    actual_topk = sw.topk(metal_selection, 2, axis=1)
    expected_topk = sw.topk(generic_selection, 2, axis=1)
    _assert_matches(actual_topk.values, expected_topk.values)
    assert _values(actual_topk.indices) == _values(expected_topk.indices)


@pytest.mark.parametrize("operation", (sw.scatter, sw.scatter_add))
def test_hierarchical_scatter_variants_match_generic(operation: Any) -> None:
    base_shape = sw.Shape([2, [2, 2]])
    base_layout = sw.Layout(base_shape, sw.Stride([3, [1, 7]]))
    index_layout = sw.Layout(sw.Shape([2, 1]), sw.Stride([2, 5]))
    update_layout = _canonical(sw.Shape([2, 2, 1]))
    base_values = [float(index + 1) for index in range(base_layout.size)]
    index_values = [3, 0]
    update_values = [10.0, 20.0, 30.0, 40.0]

    actual = operation(
        _tensor_from_logical(base_values, base_layout, metal=True, offset=2),
        _tensor_from_logical(
            index_values,
            index_layout,
            metal=True,
            offset=1,
            dtype=sw.DType.Int32,
        ),
        _tensor_from_logical(update_values, update_layout, metal=True),
        1,
    )
    expected = operation(
        _tensor_from_logical(base_values, base_layout, metal=False, offset=2),
        _tensor_from_logical(
            index_values,
            index_layout,
            metal=False,
            offset=1,
            dtype=sw.DType.Int32,
        ),
        _tensor_from_logical(update_values, update_layout, metal=False),
        1,
    )
    _assert_matches(actual, expected)


def test_stride_zero_indices_and_sources_preserve_logical_aliases() -> None:
    source_layout = sw.Layout(sw.Shape(3), sw.Stride(0))
    metal_source = _tensor_from_storage([4.0], source_layout, metal=True)
    generic_source = _tensor_from_storage([4.0], source_layout, metal=False)
    index_layout = sw.Layout(sw.Shape(3), sw.Stride(0))
    metal_indices = _tensor_from_storage(
        [1], index_layout, metal=True, dtype=sw.DType.Int32
    )
    generic_indices = _tensor_from_storage(
        [1], index_layout, metal=False, dtype=sw.DType.Int32
    )

    _assert_matches(
        sw.gather(metal_source, metal_indices, 0),
        sw.gather(generic_source, generic_indices, 0),
    )
    sorted_result = sw.sort(metal_source)
    assert _values(sorted_result.values) == [4.0, 4.0, 4.0]
    assert _values(sorted_result.indices) == [0, 1, 2]

    updates = _tensor_from_logical([1.0, 2.0, 3.0], _canonical(sw.Shape(3)), metal=True)
    with pytest.raises(ValueError, match="distinct"):
        sw.scatter(metal_source, metal_indices, updates, 0)
    assert _values(sw.scatter_add(metal_source, metal_indices, updates, 0)) == [
        4.0,
        10.0,
        4.0,
    ]


def test_invalid_indices_fail_before_result_allocation_and_preserve_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _tensor_from_logical([1.0, 2.0, 3.0], _canonical(sw.Shape(3)), metal=True)
    invalid = _tensor_from_logical(
        [-1],
        _canonical(sw.Shape(1)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    too_large = _tensor_from_logical(
        [3],
        _canonical(sw.Shape(1)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    repeated = _tensor_from_logical(
        [1, 1],
        _canonical(sw.Shape(2)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    one_update = _tensor_from_logical([9.0], _canonical(sw.Shape(1)), metal=True)
    repeated_updates = _tensor_from_logical(
        [9.0, 8.0], _canonical(sw.Shape(2)), metal=True
    )
    snapshots = [
        (tensor, _values(tensor), tensor.carrier.version)
        for tensor in (
            base,
            invalid,
            too_large,
            repeated,
            one_update,
            repeated_updates,
        )
    ]

    def unexpected_allocation(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid index reached result allocation")

    monkeypatch.setattr(sw.Metal, "allocate_like", unexpected_allocation)
    with pytest.raises(IndexError, match="out of range"):
        sw.gather(base, invalid, 0)
    with pytest.raises(ValueError, match="distinct"):
        sw.scatter(base, repeated, repeated_updates, 0)

    with pytest.raises(IndexError, match="out of range"):
        sw.gather(base, too_large, 0)

    for tensor, values, version in snapshots:
        assert _values(tensor) == values
        assert tensor.carrier.version == version


@pytest.mark.parametrize(
    "invoke",
    (
        lambda tensor, indices, updates: sw.gather(tensor, indices, 2),
        lambda tensor, indices, updates: sw.scatter(tensor, indices, updates, 0),
        lambda tensor, indices, updates: sw.sort(tensor, descending=1),
        lambda tensor, indices, updates: sw.topk(tensor, 0),
        lambda tensor, indices, updates: sw.topk(tensor, 4),
        lambda tensor, indices, updates: sw.topk(tensor, 1, largest="yes"),
    ),
)
def test_axis_shape_k_and_bool_validation_precedes_numeric_execution(
    invoke: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = _tensor_from_logical([1.0, 2.0, 3.0], _canonical(sw.Shape(3)), metal=True)
    indices = _tensor_from_logical(
        [0, 1],
        _canonical(sw.Shape(2)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    updates = _tensor_from_logical([4.0, 5.0, 6.0], _canonical(sw.Shape(3)), metal=True)

    def unexpected_execution(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid request reached a numeric executor")

    monkeypatch.setattr(indexing_ops, "execute_gather", unexpected_execution)
    monkeypatch.setattr(indexing_ops, "execute_scatter", unexpected_execution)
    monkeypatch.setattr(indexing_ops, "execute_selection", unexpected_execution)
    with pytest.raises((TypeError, ValueError)):
        invoke(tensor, indices, updates)


def test_wrong_dtype_plan_refuses_before_index_validation_or_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integer_data = _tensor_from_logical(
        [1, 2],
        _canonical(sw.Shape(2)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    indices = _tensor_from_logical(
        [0],
        _canonical(sw.Shape(1)),
        metal=True,
        dtype=sw.DType.Int32,
    )

    def unexpected_work(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unsupported plan reached Metal work")

    monkeypatch.setattr(indexing_ops, "validate_indices", unexpected_work)
    monkeypatch.setattr(indexing_ops, "execute_gather", unexpected_work)
    with pytest.raises(TypeError, match="Float32"):
        sw.gather(integer_data, indices, 0)


def test_unadvertised_index_plan_refuses_at_the_common_gate_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _tensor_from_logical([1.0, 2.0], _canonical(sw.Shape(2)), metal=True)
    indices = _tensor_from_logical(
        [0],
        _canonical(sw.Shape(1)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    near_match = replace(
        resolve_operation_plan("gather", sw.DType.Float32, sw.DType.Int32),
        output=sw.DType.Int32,
    )

    def unexpected_work(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unadvertised index plan reached Metal work")

    monkeypatch.setattr(indexing_ops, "resolve_operation_plan", lambda *_: near_match)
    monkeypatch.setattr(indexing_ops, "validate_indices", unexpected_work)
    monkeypatch.setattr(indexing_ops, "execute_gather", unexpected_work)
    monkeypatch.setattr(sw.Metal, "allocate_like", unexpected_work)

    with pytest.raises(sw.UnsupportedOperationPlan, match="Metal declares no"):
        sw.gather(data, indices, 0)


def test_scatter_rejects_an_update_profile_mismatch_before_numeric_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _tensor_from_logical([1.0, 2.0], _canonical(sw.Shape(2)), metal=True)
    indices = _tensor_from_logical(
        [0, 1],
        _canonical(sw.Shape([1, 2])),
        metal=True,
        dtype=sw.DType.Int32,
    )
    updates = _tensor_from_logical([3.0, 4.0], _canonical(sw.Shape(2)), metal=True)

    def unexpected_work(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid update profile reached Metal work")

    monkeypatch.setattr(indexing_ops, "validate_indices", unexpected_work)
    monkeypatch.setattr(indexing_ops, "execute_scatter", unexpected_work)
    with pytest.raises(ValueError, match="shape and profile"):
        sw.scatter(base, indices, updates, 0)


def test_topk_k_boundaries_and_singleton_slices_match_generic() -> None:
    values = [3.0, 1.0, 2.0]
    metal = _tensor_from_logical(values, _canonical(sw.Shape(3)), metal=True)
    generic = _tensor_from_logical(values, _canonical(sw.Shape(3)), metal=False)
    for k in (1, 3):
        actual = sw.topk(metal, k)
        expected = sw.topk(generic, k)
        _assert_matches(actual.values, expected.values)
        assert _values(actual.indices) == _values(expected.indices)

    singleton = _tensor_from_logical(
        [float("nan")], _canonical(sw.Shape(1)), metal=True
    )
    singleton_result = sw.topk(singleton, 1, largest=False)
    assert math.isnan(singleton_result.values[0])
    assert _values(singleton_result.indices) == [0]


def test_indexing_uses_tilelang_without_host_element_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[Any] = []
    original_compile = indexing_executor._compile_kernel

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    def unexpected_read(_self: sw.Metal, _index: int) -> Any:
        raise AssertionError("Metal indexing performed a host element read")

    layout = _canonical(sw.Shape(3))
    source = _tensor_from_logical([3.0, 1.0, 2.0], layout, metal=True)
    indices = _tensor_from_logical(
        [2, 0],
        _canonical(sw.Shape(2)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    updates = _tensor_from_logical([5.0, 6.0], _canonical(sw.Shape(2)), metal=True)
    indexing_executor._JIT_CACHE.clear()
    monkeypatch.setattr(indexing_executor, "_compile_kernel", capture)
    monkeypatch.setattr(sw.Metal, "get_value", unexpected_read)

    results = (
        sw.gather(source, indices, 0),
        sw.scatter_add(source, indices, updates, 0),
        sw.sort(source).values,
        sw.topk(source, 2).indices,
    )
    assert all(
        result.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
        for result in results
    )
    assert compiled
    assert all(item.facts.device_source and item.facts.host_source for item in compiled)


def test_holey_and_stride_zero_value_gradients_use_valid_metal_layouts() -> None:
    holey_layout = sw.Layout(sw.Shape(4), sw.Stride(2))
    holey = _tensor_from_logical(
        [3.0, 1.0, 4.0, 2.0], holey_layout, metal=True, offset=1
    )
    sorted_result = sw.sort(holey)
    sorted_result.values.backward(
        _tensor_from_logical(
            [10.0, 20.0, 30.0, 40.0], sorted_result.values.layout, metal=True
        )
    )
    assert holey.grad is not None
    assert holey.grad.layout == holey.layout
    assert _values(holey.grad) == [30.0, 10.0, 40.0, 20.0]

    broadcast = _tensor_from_storage(
        [2.0], sw.Layout(sw.Shape(3), sw.Stride(0)), metal=True
    )
    broadcast_result = sw.sort(broadcast)
    broadcast_result.values.backward(
        _tensor_from_logical(
            [1.0, 2.0, 4.0], broadcast_result.values.layout, metal=True
        )
    )
    assert broadcast.grad is not None
    assert broadcast.grad.layout == _canonical(sw.Shape(3))
    assert _values(broadcast.grad) == [7.0, 7.0, 7.0]


def test_saved_index_mutation_rejects_backward_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = _tensor_from_logical([1.0, 2.0], _canonical(sw.Shape(2)), metal=True)
    indices = _tensor_from_logical(
        [1],
        _canonical(sw.Shape(1)),
        metal=True,
        dtype=sw.DType.Int32,
    )
    result = sw.gather(tensor, indices, 0)
    indices.carrier[0] = 0

    def unexpected_backward(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("mutated input reached Metal backward")

    monkeypatch.setattr(indexing_ops, "execute_gather_backward", unexpected_backward)
    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(_tensor_from_logical([1.0], result.layout, metal=True))


def test_selection_default_backward_releases_graph_state() -> None:
    tensor = _tensor_from_logical([3.0, 1.0, 2.0], _canonical(sw.Shape(3)), metal=True)
    operation = tensor.carrier.dispatch_op("_sort_values")
    result = operation.forward(tensor)
    result.backward(_tensor_from_logical([1.0, 2.0, 3.0], result.layout, metal=True))

    assert operation.inputs() == ()
    assert operation.input_versions() == ()
    assert len(operation.ctx) == 0
    with pytest.raises(RuntimeError, match="retain_graph=True"):
        result.backward(
            _tensor_from_logical([1.0, 2.0, 3.0], result.layout, metal=True)
        )


def test_backward_cache_specializes_oversized_source_and_gradient_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _canonical(sw.Shape(2))
    small = _tensor_from_storage([3.0, 1.0], layout, metal=True)
    large = _tensor_from_storage([3.0, 1.0, 91.0, 92.0, 93.0], layout, metal=True)
    small_operation = small.carrier.dispatch_op("_sort_values")
    large_operation = large.carrier.dispatch_op("_sort_values")
    small_result = small_operation.forward(small)
    large_result = large_operation.forward(large)
    small_gradient = _tensor_from_storage([1.0, 2.0], layout, metal=True)
    large_gradient = _tensor_from_storage(
        [1.0, 2.0, 81.0, 82.0, 83.0], layout, metal=True
    )
    compiled: list[Any] = []
    original_compile = indexing_executor._compile_kernel

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    indexing_executor._JIT_CACHE.clear()
    monkeypatch.setattr(indexing_executor, "_compile_kernel", capture)
    small_result.backward(small_gradient)
    large_result.backward(large_gradient)

    assert len(compiled) == 2
    assert compiled[0].facts.device_source != compiled[1].facts.device_source


def test_public_shape_boundary_excludes_empty_selection_slices() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        sw.Shape(0)
