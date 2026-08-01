"""Focused regression tests for Generic's concrete numeric reference paths."""

from __future__ import annotations

import math
from typing import Any

import numpy
import pytest

import strideweave as sw
from strideweave import DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.numerics import safe_int_power_checked
from strideweave.carriers.generic.reduction_ops import GenericReduceSumOperation

ONE_MODE = Layout(Shape(4), Stride(1))


def _tensor(values: list[float]) -> Tensor:
    return Tensor(Generic(values, dtype=DType.Float32), 0, ONE_MODE)


def _values(tensor: Tensor) -> list[float]:
    return [tensor[index] for index in range(tensor.size())]


def _f32(value: Any) -> numpy.float32:
    return numpy.float32(value)


def _sigmoid(value: numpy.float32) -> numpy.float32:
    one = _f32(1.0)
    if value >= _f32(0.0):
        return _f32(one / (one + numpy.exp(-value, dtype=numpy.float32)))
    exponential = numpy.exp(value, dtype=numpy.float32)
    return _f32(exponential / (one + exponential))


def _expected_activation(
    name: str, value: numpy.float32
) -> tuple[numpy.float32, numpy.float32]:
    zero = _f32(0.0)
    one = _f32(1.0)
    half = _f32(0.5)
    if name == "sigmoid":
        output = _sigmoid(value)
        return output, _f32(output * (one - output))
    if name == "silu":
        sigmoid = _sigmoid(value)
        return _f32(value * sigmoid), _f32(sigmoid + value * sigmoid * (one - sigmoid))
    if name == "gelu":
        inverse_sqrt2 = _f32(math.sqrt(0.5))
        erf_value = _f32(math.erf(float(value * inverse_sqrt2)))
        output = _f32(half * value * (one + erf_value))
        inverse_sqrt_2pi = _f32(1.0 / math.sqrt(2.0 * math.pi))
        derivative = _f32(
            half * (one + erf_value)
            + value
            * numpy.exp(-half * value * value, dtype=numpy.float32)
            * inverse_sqrt_2pi
        )
        return output, derivative
    if name == "softplus":
        output = _f32(
            numpy.log1p(
                numpy.exp(-numpy.abs(value), dtype=numpy.float32),
                dtype=numpy.float32,
            )
            + max(value, zero)
        )
        return output, _sigmoid(value)
    if name == "elu":
        output = value if value > zero else _f32(numpy.expm1(value))
        derivative = one if value > zero else _f32(numpy.exp(value))
        return output, derivative
    if name == "leaky_relu":
        slope = _f32(0.01)
        output = value if value >= zero else _f32(slope * value)
        derivative = one if value >= zero else slope
        return output, derivative
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name", ["sigmoid", "silu", "gelu", "softplus", "elu", "leaky_relu"]
)
def test_float32_activations_round_every_primitive(name: str) -> None:
    values = [-3.125, -0.5, 0.5, 7.25]
    tensor = _tensor(values)
    gradient = _tensor([0.25, 0.5, 0.75, 1.25])

    with numpy.errstate(all="ignore"):
        result = getattr(sw, name)(tensor)
        result.backward(gradient)

    expected_values = [_expected_activation(name, _f32(value))[0] for value in values]
    expected_gradients = [
        _f32(_f32(grad) * _expected_activation(name, _f32(value))[1])
        for value, grad in zip(values, [0.25, 0.5, 0.75, 1.25], strict=True)
    ]
    assert _values(result) == [float(value) for value in expected_values]
    assert tensor.grad is not None
    assert _values(tensor.grad) == [float(value) for value in expected_gradients]


def test_int32_pow_rejects_huge_permitted_exponent_before_bigint_allocation() -> None:
    assert safe_int_power_checked(2, 30) == 2**30
    with pytest.raises(OverflowError, match="out of int32 range"):
        safe_int_power_checked(2, 2**31 - 1)

    tensor = Tensor(Generic([2], dtype=DType.Int32), 0, Layout(Shape(1), Stride(1)))
    with pytest.raises(OverflowError, match="out of int32 range"):
        _ = tensor ** (2**31 - 1)


def test_reduce_sum_uses_the_planned_reduction_operation() -> None:
    tensor = Tensor(
        Generic([1.0, 2.0], dtype=DType.Float32),
        0,
        Layout(Shape([1, 2]), Stride([1, 1])),
    )
    result = sw.reduce_sum(tensor, "a b -> a")
    assert isinstance(result.autograd_ctx, GenericReduceSumOperation)
    assert type(result.autograd_ctx).__module__.endswith("reduction_ops")
