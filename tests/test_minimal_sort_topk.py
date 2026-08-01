"""Focused tests for the isolated Generic selection backend."""

from __future__ import annotations

import math

import pytest

import strideweave as sw
import strideweave.functional.api as functional_api
import strideweave.operation as operation
from strideweave import DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.selection_ops import (
    GenericSortIndicesOperation,
    GenericSortValuesOperation,
    GenericTopKIndicesOperation,
    GenericTopKValuesOperation,
)
from strideweave.carriers.operation_helpers import _canonical_layout_for_shape


def _tensor(
    values: list[float | int], shape: Shape, dtype: DType = DType.Float32
) -> Tensor:
    return Tensor(
        Generic(values, dtype=dtype),
        0,
        _canonical_layout_for_shape(shape),
    )


def _values(tensor: Tensor) -> list[object]:
    return [tensor[index] for index in range(tensor.size())]


def test_sort_orders_special_values_and_keeps_equal_source_ordinals_stable() -> None:
    tensor = _tensor(
        [3.0, float("nan"), -0.0, 0.0, 2.0, float("nan"), -float("inf"), float("inf")],
        Shape(8),
    )

    ascending_values = GenericSortValuesOperation().forward(tensor)
    ascending_indices = GenericSortIndicesOperation().forward(tensor)
    descending_values = GenericSortValuesOperation().forward(tensor, -1, True)
    descending_indices = GenericSortIndicesOperation().forward(tensor, -1, True)

    assert _values(ascending_indices) == [6, 2, 3, 4, 0, 7, 1, 5]
    assert _values(descending_indices) == [1, 5, 7, 0, 4, 2, 3, 6]
    assert _values(ascending_values)[:2] == [-float("inf"), -0.0]
    assert math.copysign(1.0, ascending_values[1]) == -1.0
    assert math.copysign(1.0, ascending_values[2]) == 1.0
    assert all(math.isnan(ascending_values[index]) for index in (6, 7))
    assert all(math.isnan(descending_values[index]) for index in (0, 1))


def test_sort_and_topk_result_types_have_one_public_runtime_identity() -> None:
    assert sw.SortResult is operation.SortResult is functional_api.SortResult
    assert sw.TopKResult is operation.TopKResult is functional_api.TopKResult

    tensor = _tensor([2.0, 1.0], Shape(2))
    sorted_result = sw.sort(tensor)
    topk_result = sw.topk(tensor, 1)

    assert isinstance(sorted_result, sw.SortResult)
    assert isinstance(topk_result, sw.TopKResult)
    assert sorted_result._fields == ("values", "indices")
    assert topk_result._fields == ("values", "indices")
    assert tuple(sorted_result) == (sorted_result.values, sorted_result.indices)
    assert tuple(topk_result) == (topk_result.values, topk_result.indices)


def test_sort_flattens_a_hierarchical_axis_and_uses_a_canonical_layout() -> None:
    shape = Shape([2, [2, 2]])
    tensor = _tensor([4.0, 1.0, 3.0, 2.0, 8.0, 6.0, 7.0, 5.0], shape)

    values = GenericSortValuesOperation().forward(tensor, -1)
    indices = GenericSortIndicesOperation().forward(tensor, -1)

    assert values.layout.shape == shape
    assert values.layout == _canonical_layout_for_shape(shape)
    assert _values(values) == [3.0, 1.0, 4.0, 2.0, 7.0, 5.0, 8.0, 6.0]
    assert _values(indices) == [1, 0, 0, 1, 3, 3, 2, 2]
    assert _values(tensor) == [4.0, 1.0, 3.0, 2.0, 8.0, 6.0, 7.0, 5.0]


def test_topk_replaces_hierarchical_axis_and_returns_sorted_largest_values() -> None:
    shape = Shape([2, [2, 2]])
    tensor = _tensor([4.0, 1.0, 3.0, 2.0, 8.0, 6.0, 7.0, 5.0], shape)

    values = GenericTopKValuesOperation().forward(tensor, 2, 1)
    indices = GenericTopKIndicesOperation().forward(tensor, 2, 1)
    smallest_values = GenericTopKValuesOperation().forward(tensor, 2, 1, False)
    smallest_indices = GenericTopKIndicesOperation().forward(tensor, 2, 1, False)

    expected_shape = Shape([2, 2])
    assert values.layout.shape == expected_shape
    assert values.layout == _canonical_layout_for_shape(expected_shape)
    assert _values(values) == [8.0, 6.0, 7.0, 5.0]
    assert _values(indices) == [2, 2, 3, 3]
    assert _values(smallest_values) == [3.0, 1.0, 4.0, 2.0]
    assert _values(smallest_indices) == [1, 0, 0, 1]


@pytest.mark.parametrize(
    ("operation", "arguments", "message"),
    [
        (GenericTopKValuesOperation, (0,), "1 <= k"),
        (GenericTopKIndicesOperation, (5,), "1 <= k"),
        (GenericTopKValuesOperation, ("two",), "k must be an integer"),
    ],
)
def test_topk_validates_k(
    operation: type, arguments: tuple[object], message: str
) -> None:
    tensor = _tensor([1.0, 2.0, 3.0], Shape(3))
    with pytest.raises((TypeError, ValueError), match=message):
        operation().forward(tensor, *arguments)


def test_selection_validates_dtype_and_top_level_axis() -> None:
    integer_tensor = _tensor([1, 2], Shape(2), DType.Int32)
    float_tensor = _tensor([1.0, 2.0], Shape(2))

    with pytest.raises(TypeError, match="Float32"):
        GenericSortValuesOperation().forward(integer_tensor)
    with pytest.raises(ValueError, match="axis out of range"):
        GenericSortValuesOperation().forward(float_tensor, 2)
    with pytest.raises(TypeError, match="axis must be an integer"):
        GenericSortIndicesOperation().forward(float_tensor, "last")


def test_sort_and_topk_values_route_vjps_to_source_ordinals() -> None:
    tensor = _tensor([3.0, 1.0, 3.0, 2.0], Shape(4))
    sort_operation = GenericSortValuesOperation()
    sorted_values = sort_operation.forward(tensor)
    sorted_values.backward(_tensor([10.0, 20.0, 30.0, 40.0], Shape(4)))
    sort_gradient = tensor.grad
    assert sort_gradient is not None
    assert _values(sort_gradient) == [30.0, 10.0, 40.0, 20.0]

    topk_tensor = _tensor([4.0, 7.0, 7.0, 2.0], Shape(4))
    topk_operation = GenericTopKValuesOperation()
    topk_values = topk_operation.forward(topk_tensor, 2)
    topk_values.backward(_tensor([10.0, 20.0], Shape(2)))
    topk_gradient = topk_tensor.grad
    assert topk_gradient is not None
    assert _values(topk_gradient) == [0.0, 10.0, 20.0, 0.0]


def test_indices_are_int32_and_non_differentiable() -> None:
    tensor = _tensor([3.0, 1.0], Shape(2))
    result = GenericSortIndicesOperation().forward(tensor)

    assert result.dtype() is DType.Int32
    with pytest.raises(RuntimeError, match="non-differentiable"):
        _ = result.grad


def test_selection_outputs_canonicalize_a_gapped_input_layout() -> None:
    tensor = Tensor(
        Generic([3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0], dtype=DType.Float32),
        0,
        Layout(Shape(4), Stride(2)),
    )

    result = GenericSortValuesOperation().forward(tensor)

    assert result.layout == _canonical_layout_for_shape(Shape(4))
    assert _values(result) == [1.0, 1.0, 2.0, 3.0]
