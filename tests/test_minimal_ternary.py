"""Focused Generic tests for the minimum ternary operation set."""

from __future__ import annotations

import math

import pytest

from strideweave import DType, Generic, Shape, Stride, Tensor
from strideweave.carriers.generic.ternary_ops import (
    GenericClampOperation,
    GenericSelectOperation,
)
from strideweave.carriers.operation_helpers import _canonical_layout_for_shape


def _tensor(values, *, dtype=DType.Float32, shape=None):
    if shape is None:
        shape = Shape(len(values))
    layout = _canonical_layout_for_shape(shape)
    return Tensor(Generic(values, dtype=dtype), 0, layout)


def _values(tensor):
    return [tensor[index] for index in range(tensor.size())]


def test_select_reads_only_selected_branch_and_returns_float32():
    condition = _tensor([True, False], dtype=DType.Bool)
    on_true = _tensor([1.5, math.nan])
    on_false = _tensor([math.nan, 2.5])

    result = GenericSelectOperation().forward(condition, on_true, on_false)

    assert _values(result) == [1.5, 2.5]
    assert result.dtype() is DType.Float32
    assert result.layout.is_injective
    assert result.layout.stride == Stride(1)


def test_select_aligns_three_operands_simultaneously():
    condition = _tensor([True, False, True], dtype=DType.Bool, shape=Shape([1, 3]))
    on_true = _tensor([10.0, 20.0], shape=Shape([2, 1]))
    on_false = _tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=Shape([2, 3]))

    result = GenericSelectOperation().forward(condition, on_true, on_false)

    assert _values(result) == [10.0, 20.0, 3.0, 4.0, 10.0, 20.0]
    assert result.layout.shape == Shape([2, 3])


def test_select_vjp_masks_value_inputs_and_condition_is_nondifferentiable():
    # The default helper is Float32 because the two value inputs use that dtype;
    # the condition is rebuilt with concrete Bool storage.
    condition = _tensor([True, False], dtype=DType.Bool)
    on_true = _tensor([3.0, 4.0])
    on_false = _tensor([5.0, 6.0])
    operation = GenericSelectOperation()
    operation.forward(condition, on_true, on_false)
    gradient = _tensor([7.0, 11.0])

    condition_gradient, true_gradient, false_gradient = operation.backward(gradient)

    assert condition_gradient is None
    assert _values(true_gradient) == [7.0, 0.0]
    assert _values(false_gradient) == [0.0, 11.0]


def test_select_rejects_non_float32_values_before_result_allocation():
    condition = _tensor([True], dtype=DType.Bool)
    integer = _tensor([1], dtype=DType.Int32)
    floating = _tensor([2.0])

    with pytest.raises(TypeError, match=r"DType\.Float32"):
        GenericSelectOperation().forward(condition, integer, floating)


def test_clamp_matches_ordered_maximum_then_minimum_for_scalar_bounds():
    tensor = _tensor([-2.0, 0.5, 2.0])

    result = GenericClampOperation().forward(tensor, 0.0, 1.0)

    assert _values(result) == [0.0, 0.5, 1.0]
    assert result.dtype() is DType.Float32


def test_clamp_tensor_bounds_broadcast_and_vjp_follow_two_stages():
    tensor = _tensor([1.0], shape=Shape([1, 1]))
    lower = _tensor([1.0], shape=Shape([1, 1]))
    upper = _tensor([1.0], shape=Shape([1, 1]))
    operation = GenericClampOperation()
    result = operation.forward(tensor, lower, upper)
    gradient = _tensor([4.0], shape=Shape([1, 1]))

    tensor_gradient, lower_gradient, upper_gradient = operation.backward(gradient)

    assert _values(result) == [1.0]
    assert _values(tensor_gradient) == [1.0]
    assert _values(lower_gradient) == [1.0]
    assert _values(upper_gradient) == [2.0]


def test_clamp_preserves_nan_and_signed_zero_and_lower_upper_order():
    tensor = _tensor([math.nan, -0.0, 2.0])
    result = GenericClampOperation().forward(tensor, 1.0, -1.0)

    assert math.isnan(result[0])
    assert result[1] == -1.0
    assert result[2] == -1.0


def test_clamp_scalar_bounds_have_no_scalar_gradient_inputs():
    tensor = _tensor([-1.0, 2.0])
    operation = GenericClampOperation()
    operation.forward(tensor, 0.0, 1.0)

    (tensor_gradient,) = operation.backward(_tensor([3.0, 5.0]))

    assert _values(tensor_gradient) == [0.0, 0.0]
