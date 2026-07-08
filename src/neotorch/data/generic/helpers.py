"""Generic backend dtype and scalar math helpers."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

from ..dtype import DataType


def _is_integral_number(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _generic_binary_dtype(lhs: Any, rhs: Any) -> DataType:
    if lhs.dtype() is DataType.Floating or rhs.dtype() is DataType.Floating:
        return DataType.Floating
    return DataType.Any


def _generic_scalar_mul_dtype(tensor: Any, scalar: Any) -> DataType:
    if tensor.dtype() is DataType.Floating or not _is_integral_number(scalar):
        return DataType.Floating
    return DataType.Any


def _generic_pow_dtype(tensor: Any, exponent: Any) -> DataType:
    if tensor.dtype() is DataType.Floating:
        return DataType.Floating
    if not _is_integral_number(exponent) or exponent < 0:
        return DataType.Floating
    return DataType.Any


def _sigmoid_value(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus_value(value: float) -> float:
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


_INV_SQRT2 = math.sqrt(0.5)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_LEAKY_RELU_NEGATIVE_SLOPE = 0.01


def _gelu_value(value: float) -> float:
    return 0.5 * value * (1.0 + math.erf(value * _INV_SQRT2))


def _gelu_derivative(value: float) -> float:
    return (
        0.5 * (1.0 + math.erf(value * _INV_SQRT2))
        + value * math.exp(-0.5 * value * value) * _INV_SQRT_2PI
    )
