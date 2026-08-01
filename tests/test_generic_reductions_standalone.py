"""Standalone tests for the Generic reduction/scan implementations.

The integration task wires these classes into carrier dispatch and the central
capability registry.  These tests intentionally instantiate the operation classes
directly so their coordinate and VJP contracts can be checked independently.
"""

from __future__ import annotations

import math

import pytest

from strideweave import DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.reduction_ops import (
    GenericArgMaxOperation,
    GenericArgMinOperation,
    GenericCumsumOperation,
    GenericReduceMaxOperation,
    GenericReduceMinOperation,
    GenericReduceProdOperation,
    GenericReduceSumOperation,
)


def tensor(values, shape):
    normalized_shape = Shape(shape)
    layout = Layout(
        normalized_shape,
        Stride([1, int(normalized_shape[0])] if len(normalized_shape) == 2 else 1),
    )
    storage = [*values, *([0.0] * (layout.cosize - len(values)))]
    return Tensor(
        Generic(storage, dtype=DType.Floating),
        0,
        layout,
    )


def values_of(value):
    return [value[index] for index in range(value.size())]


def test_sum_and_product_use_first_mode_fast_fibers_and_vjps():
    summed = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    sum_result = GenericReduceSumOperation().forward(summed)
    assert values_of(sum_result) == [9.0, 12.0]
    sum_result.backward(Tensor(Generic([10.0, 20.0]), 0, sum_result.layout))
    assert values_of(summed.grad) == [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]

    product = tensor([0.0, 2.0, 3.0], [1, 3])
    product_result = GenericReduceProdOperation().forward(product)
    assert values_of(product_result) == [0.0]
    product_result.backward(Tensor(Generic([1.0]), 0, product_result.layout))
    # Direct products of the other members define zero behavior without division.
    assert values_of(product.grad) == [6.0, 0.0, 0.0]


def test_extreme_reductions_propagate_nan_and_choose_signed_zero():
    maximum = tensor([0.0, -0.0], [1, 2])
    max_result = GenericReduceMaxOperation().forward(maximum)
    assert not math.copysign(1.0, max_result[0]) < 0
    max_result.backward(Tensor(Generic([2.0]), 0, max_result.layout))
    assert values_of(maximum.grad) == [1.0, 1.0]

    minimum = tensor([0.0, -0.0], [1, 2])
    min_result = GenericReduceMinOperation().forward(minimum)
    assert math.copysign(1.0, min_result[0]) < 0

    asymmetric_minimum = tensor([3.0, 1.0, 2.0], [1, 3])
    assert values_of(GenericReduceMinOperation().forward(asymmetric_minimum)) == [1.0]

    nan_input = tensor([float("nan"), 1.0], [1, 2])
    nan_result = GenericReduceMaxOperation().forward(nan_input)
    nan_result.backward(Tensor(Generic([3.0]), 0, nan_result.layout))
    assert all(math.isnan(value) for value in values_of(nan_input.grad))


def test_arg_reductions_return_first_nan_or_tie_as_int32():
    values = tensor([1.0, float("nan"), float("nan")], [1, 3])
    argmax = GenericArgMaxOperation().forward(values)
    assert argmax.dtype() is DType.Int32
    assert values_of(argmax) == [1]

    tied = tensor([3.0, 1.0, 3.0], [1, 3])
    argmin = GenericArgMinOperation().forward(tied)
    assert values_of(argmin) == [1]


def test_cumsum_is_inclusive_and_backward_is_reverse_inclusive():
    source = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    result = GenericCumsumOperation().forward(source, 1)
    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert values_of(result) == [1.0, 2.0, 4.0, 6.0, 9.0, 12.0]

    result.backward(Tensor(Generic([1.0] * 6), 0, result.layout))
    assert values_of(source.grad) == [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]


@pytest.mark.parametrize("axis", [-1, 2])
def test_cumsum_accepts_negative_axis_and_rejects_out_of_range(axis):
    source = tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    if axis == -1:
        assert values_of(GenericCumsumOperation().forward(source, axis)) == [
            1.0,
            2.0,
            4.0,
            6.0,
        ]
    else:
        with pytest.raises(ValueError, match="out of range"):
            GenericCumsumOperation().forward(source, axis)
