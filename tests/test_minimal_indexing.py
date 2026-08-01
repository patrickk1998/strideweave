"""Focused tests for the isolated Generic section-7 indexing backend."""

from __future__ import annotations

import pytest

from strideweave import DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.indexing_ops import (
    GenericGatherOperation,
    GenericScatterAddOperation,
    GenericScatterOperation,
)
from strideweave.carriers.generic.numerics import binary32, float32_scalar


def _layout(shape: Shape) -> Layout:
    from strideweave.carriers.operation_helpers import _canonical_layout_for_shape

    return _canonical_layout_for_shape(shape)


def _tensor(values: list[float | int], shape: Shape, dtype: DType) -> Tensor:
    layout = _layout(shape)
    return Tensor(
        Generic(values, dtype=dtype),
        0,
        layout,
    )


def _values(tensor: Tensor) -> list[object]:
    return [tensor[index] for index in range(tensor.size())]


def _gradient(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_gather_replaces_a_hierarchical_axis_with_the_full_index_shape() -> None:
    shape = Shape([2, [3, 2], 2])
    tensor = _tensor(
        [float(value) for value in range(shape.logical_size)], shape, DType.Float32
    )
    indices = _tensor([5, 0, 3, 1], Shape([2, 2]), DType.Int32)

    result = GenericGatherOperation().forward(tensor, indices, -2)

    assert result.layout.shape == Shape([2, 2, 2, 2])
    expected = []
    for trailing in range(2):
        for index_column in range(2):
            for index_row in range(2):
                index = (5, 0, 3, 1)[index_row + 2 * index_column]
                for outer in range(2):
                    source = outer + 2 * index + 2 * 6 * trailing
                    expected.append(float(source))
    assert _values(result) == expected
    assert result.layout.is_injective
    assert result.carrier is not tensor.carrier


def test_gather_allows_repeated_and_stride_zero_indices_and_sums_vjp() -> None:
    tensor = _tensor([1.0, 2.0, 3.0], Shape(3), DType.Float32)
    indices = Tensor(
        Generic([1], dtype=DType.Int32),
        0,
        Layout(Shape(3), Stride(0)),
    )

    operation = GenericGatherOperation()
    result = operation.forward(tensor, indices, 0)

    assert _values(result) == [2.0, 2.0, 2.0]
    result.backward(
        Tensor(Generic([1.0, 2.0, 4.0], dtype=DType.Float32), 0, result.layout)
    )
    assert _values(_gradient(tensor)) == [0.0, 7.0, 0.0]


def test_gather_rejects_every_invalid_index_before_result_allocation() -> None:
    tensor = _tensor([1.0, 2.0], Shape(2), DType.Float32)
    indices = _tensor([-1], Shape(1), DType.Int32)

    with pytest.raises(IndexError, match="out of range"):
        GenericGatherOperation().forward(tensor, indices, 0)


def test_scatter_requires_distinct_indices_and_is_functional() -> None:
    base = _tensor([1.0, 2.0, 3.0], Shape(3), DType.Float32)
    indices = _tensor([2, 0], Shape(2), DType.Int32)
    updates = _tensor([20.0, 10.0], Shape(2), DType.Float32)

    result = GenericScatterOperation().forward(base, indices, updates, -1)

    assert _values(result) == [10.0, 2.0, 20.0]
    assert _values(base) == [1.0, 2.0, 3.0]

    repeated = _tensor([1, 1], Shape(2), DType.Int32)
    with pytest.raises(ValueError, match="distinct"):
        GenericScatterOperation().forward(base, repeated, updates, 0)


def test_scatter_vjp_zeros_written_base_positions_and_gathers_updates() -> None:
    base = _tensor([1.0, 2.0, 3.0], Shape(3), DType.Float32)
    indices = _tensor([2, 0], Shape(2), DType.Int32)
    updates = _tensor([20.0, 10.0], Shape(2), DType.Float32)
    operation = GenericScatterOperation()
    result = operation.forward(base, indices, updates, 0)

    result.backward(
        Tensor(Generic([1.0, 2.0, 4.0], dtype=DType.Float32), 0, result.layout)
    )

    assert _values(_gradient(base)) == [0.0, 2.0, 0.0]
    assert _values(_gradient(updates)) == [4.0, 1.0]


def test_scatter_add_accumulates_repeated_indices_in_first_mode_fast_order() -> None:
    base = _tensor([0.0], Shape(1), DType.Float32)
    indices = _tensor([0, 0, 0], Shape(3), DType.Int32)
    updates = _tensor([1.0e8, 1.0, -1.0e8], Shape(3), DType.Float32)

    result = GenericScatterAddOperation().forward(base, indices, updates, 0)

    expected = binary32(
        float32_scalar(
            float32_scalar(binary32(float32_scalar(1.0e8) + float32_scalar(1.0)))
            + float32_scalar(-1.0e8)
        )
    )
    assert _values(result) == [expected]


def test_scatter_add_vjp_passes_base_gradient_and_gathers_each_update() -> None:
    base = _tensor([1.0, 2.0], Shape(2), DType.Float32)
    indices = _tensor([1, 1], Shape(2), DType.Int32)
    updates = _tensor([3.0, 4.0], Shape(2), DType.Float32)
    operation = GenericScatterAddOperation()
    result = operation.forward(base, indices, updates, 0)

    result.backward(Tensor(Generic([2.0, 5.0], dtype=DType.Float32), 0, result.layout))

    assert _values(_gradient(base)) == [2.0, 5.0]
    assert _values(_gradient(updates)) == [5.0, 5.0]


@pytest.mark.parametrize("operation", [GenericGatherOperation, GenericScatterOperation])
def test_indexing_requires_declared_float32_or_int32_dtypes(operation: type) -> None:
    float_tensor = _tensor([1.0, 2.0], Shape(2), DType.Floating)
    int_indices = _tensor([0], Shape(1), DType.Int32)
    updates = _tensor([1.0], Shape(1), DType.Float32)
    if operation is GenericGatherOperation:

        def invoke() -> object:
            return operation().forward(float_tensor, int_indices, 0)
    else:

        def invoke() -> object:
            return operation().forward(float_tensor, int_indices, updates, 0)

    with pytest.raises(TypeError, match="Float32"):
        invoke()
