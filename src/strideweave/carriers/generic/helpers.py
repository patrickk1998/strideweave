"""Generic carrier dtype and scalar math helpers."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

from ..dtype import DType
from .numerics import (
    _is_binary32_scalar,
    float32_scalar,
    safe_abs,
    safe_divide,
    safe_erf,
    safe_exp,
    safe_expm1,
    safe_log1p,
)


def _is_integral_number(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _generic_binary_dtype(lhs: Any, rhs: Any) -> DType:
    if lhs.dtype() is DType.Floating or rhs.dtype() is DType.Floating:
        return DType.Floating
    return DType.Any


def _generic_scalar_mul_dtype(tensor: Any, scalar: Any) -> DType:
    if tensor.dtype() is DType.Floating or not _is_integral_number(scalar):
        return DType.Floating
    return DType.Any


def _generic_pow_dtype(tensor: Any, exponent: Any) -> DType:
    if tensor.dtype() is DType.Floating:
        return DType.Floating
    if not _is_integral_number(exponent) or exponent < 0:
        return DType.Floating
    return DType.Any


def _sigmoid_value(value: Any) -> Any:
    """Evaluate a numerically stable sigmoid in the input arithmetic.

    Concrete ``Float32`` operations must round each primitive, rather than
    widening the complete formula to Python ``float`` and narrowing once at
    the end.  The scalar helpers in :mod:`.numerics` preserve that distinction
    while this function keeps the legacy opaque path's historical ``math``
    behavior.
    """
    if _is_binary32_scalar(value):
        zero = float32_scalar(0.0)
        one = float32_scalar(1.0)
        if value >= zero:
            inverse = safe_exp(-value)
            return safe_divide(one, one + inverse)
        exponential = safe_exp(value)
        return safe_divide(exponential, one + exponential)
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus_value(value: Any) -> Any:
    """Evaluate stable softplus, retaining binary32 primitive boundaries."""
    if _is_binary32_scalar(value):
        zero = float32_scalar(0.0)
        return safe_log1p(safe_exp(-safe_abs(value))) + max(value, zero)
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


_INV_SQRT2 = math.sqrt(0.5)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_LEAKY_RELU_NEGATIVE_SLOPE = 0.01


def _gelu_value(value: Any) -> Any:
    """Evaluate GELU with binary32-rounded constants and primitives."""
    if _is_binary32_scalar(value):
        half = float32_scalar(0.5)
        inverse_sqrt2 = float32_scalar(_INV_SQRT2)
        one = float32_scalar(1.0)
        return half * value * (one + safe_erf(value * inverse_sqrt2))
    return 0.5 * value * (1.0 + math.erf(value * _INV_SQRT2))


def _gelu_derivative(value: Any) -> Any:
    """Evaluate the GELU derivative in the input arithmetic."""
    if _is_binary32_scalar(value):
        half = float32_scalar(0.5)
        one = float32_scalar(1.0)
        inverse_sqrt2 = float32_scalar(_INV_SQRT2)
        inverse_sqrt_2pi = float32_scalar(_INV_SQRT_2PI)
        exponent = -half * value * value
        return (
            half * (one + safe_erf(value * inverse_sqrt2))
            + value * safe_exp(exponent) * inverse_sqrt_2pi
        )
    return (
        0.5 * (1.0 + math.erf(value * _INV_SQRT2))
        + value * math.exp(-0.5 * value * value) * _INV_SQRT_2PI
    )


def _elu_value(value: Any) -> Any:
    """Evaluate ELU (alpha one) in the input arithmetic."""
    if _is_binary32_scalar(value):
        return value if value > float32_scalar(0.0) else safe_expm1(value)
    return value if value > 0.0 else math.expm1(value)


def _elu_derivative(value: Any) -> Any:
    """Evaluate the ELU derivative in the input arithmetic."""
    if _is_binary32_scalar(value):
        return float32_scalar(1.0) if value > float32_scalar(0.0) else safe_exp(value)
    return 1.0 if value > 0.0 else math.exp(value)


def _leaky_relu_value(value: Any) -> Any:
    """Evaluate leaky ReLU using a binary32 slope for concrete inputs."""
    if _is_binary32_scalar(value):
        slope = float32_scalar(_LEAKY_RELU_NEGATIVE_SLOPE)
        return value if value >= float32_scalar(0.0) else slope * value
    return value if value >= 0.0 else _LEAKY_RELU_NEGATIVE_SLOPE * value


def _leaky_relu_derivative(value: Any) -> Any:
    """Evaluate the leaky ReLU derivative in the input arithmetic."""
    if _is_binary32_scalar(value):
        return (
            float32_scalar(1.0)
            if value >= float32_scalar(0.0)
            else float32_scalar(_LEAKY_RELU_NEGATIVE_SLOPE)
        )
    return 1.0 if value >= 0.0 else _LEAKY_RELU_NEGATIVE_SLOPE
