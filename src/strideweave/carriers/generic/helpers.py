"""Generic carrier dtype and scalar math helpers."""

from __future__ import annotations

import math
from typing import Any

from .numerics import (
    float32_scalar,
    safe_abs,
    safe_divide,
    safe_erf,
    safe_exp,
    safe_expm1,
    safe_log1p,
)


def _sigmoid_value(value: Any) -> Any:
    """Evaluate a numerically stable sigmoid in the input arithmetic.

    ``Float32`` operations round each primitive rather than widening the
    complete formula to Python ``float`` and narrowing once at the end.
    """
    zero = float32_scalar(0.0)
    one = float32_scalar(1.0)
    if value >= zero:
        inverse = safe_exp(-value)
        return safe_divide(one, one + inverse)
    exponential = safe_exp(value)
    return safe_divide(exponential, one + exponential)


def _softplus_value(value: Any) -> Any:
    """Evaluate stable softplus, retaining binary32 primitive boundaries."""
    zero = float32_scalar(0.0)
    return safe_log1p(safe_exp(-safe_abs(value))) + max(value, zero)


_INV_SQRT2 = math.sqrt(0.5)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_LEAKY_RELU_NEGATIVE_SLOPE = 0.01


def _gelu_value(value: Any) -> Any:
    """Evaluate GELU with binary32-rounded constants and primitives."""
    half = float32_scalar(0.5)
    inverse_sqrt2 = float32_scalar(_INV_SQRT2)
    one = float32_scalar(1.0)
    return half * value * (one + safe_erf(value * inverse_sqrt2))


def _gelu_derivative(value: Any) -> Any:
    """Evaluate the GELU derivative in the input arithmetic."""
    half = float32_scalar(0.5)
    one = float32_scalar(1.0)
    inverse_sqrt2 = float32_scalar(_INV_SQRT2)
    inverse_sqrt_2pi = float32_scalar(_INV_SQRT_2PI)
    exponent = -half * value * value
    return (
        half * (one + safe_erf(value * inverse_sqrt2))
        + value * safe_exp(exponent) * inverse_sqrt_2pi
    )


def _elu_value(value: Any) -> Any:
    """Evaluate ELU (alpha one) in the input arithmetic."""
    return value if value > float32_scalar(0.0) else safe_expm1(value)


def _elu_derivative(value: Any) -> Any:
    """Evaluate the ELU derivative in the input arithmetic."""
    return float32_scalar(1.0) if value > float32_scalar(0.0) else safe_exp(value)


def _leaky_relu_value(value: Any) -> Any:
    """Evaluate leaky ReLU using a binary32 slope for concrete inputs."""
    slope = float32_scalar(_LEAKY_RELU_NEGATIVE_SLOPE)
    return value if value >= float32_scalar(0.0) else slope * value


def _leaky_relu_derivative(value: Any) -> Any:
    """Evaluate the leaky ReLU derivative in the input arithmetic."""
    return (
        float32_scalar(1.0)
        if value >= float32_scalar(0.0)
        else float32_scalar(_LEAKY_RELU_NEGATIVE_SLOPE)
    )
