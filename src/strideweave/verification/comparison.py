"""Bitwise and numerical comparison utilities for encoded Float32 results."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass

from .model import Deviations, Tolerance


def float32_bits(value: float) -> int:
    """Return the IEEE-754 binary32 word obtained by encoding ``value``.

    Args:
        value: Numeric value to encode as Float32.

    Returns:
        Unsigned integer containing the encoded Float32 bits.

    Examples:
        >>> hex(float32_bits(1.0))
        '0x3f800000'
    """
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _ordered_float32(bits: int) -> int:
    return (~bits & 0xFFFF_FFFF) if bits & 0x8000_0000 else bits | 0x8000_0000


def float32_ulp_distance(expected: float, actual: float) -> int:
    """Measure the ordered-ULP distance between two encoded Float32 values.

    Args:
        expected: Reference value.
        actual: Value produced by the backend.

    Returns:
        Ordered ULP distance, with distinct NaN encodings treated as maximally
        different.

    Examples:
        >>> float32_ulp_distance(1.0, 1.0)
        0
    """
    if math.isnan(expected) or math.isnan(actual):
        return 0 if float32_bits(expected) == float32_bits(actual) else 0xFFFF_FFFF
    if expected == actual == 0.0:
        return 0
    return abs(
        _ordered_float32(float32_bits(expected))
        - _ordered_float32(float32_bits(actual))
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Immutable summary of elementwise Float32 differences."""

    deviations: Deviations
    mismatches: int
    signed_zero_mismatches: int
    nan_payload_mismatches: int

    def within(self, tolerance: Tolerance) -> bool:
        if self.signed_zero_mismatches or self.nan_payload_mismatches:
            return False
        deviations = self.deviations
        if (
            deviations.maximum_absolute is None
            or deviations.maximum_relative is None
            or deviations.maximum_ulps is None
        ):
            return False
        return self.mismatches == 0 or (
            deviations.maximum_absolute <= tolerance.absolute
            and deviations.maximum_relative <= tolerance.relative
            and deviations.maximum_ulps <= tolerance.ulps
        )


def compare_float32(expected: Iterable[float], actual: Iterable[float]) -> Comparison:
    """Compare two equal-length sequences using encoded Float32 semantics.

    Args:
        expected: Reference values.
        actual: Values produced by the backend.

    Returns:
        Comparison containing mismatch counts and maximum deviations.

    Examples:
        >>> compare_float32((1.0,), (1.0,)).mismatches
        0
    """
    expected_values = tuple(expected)
    actual_values = tuple(actual)
    if len(expected_values) != len(actual_values):
        raise ValueError("comparison inputs must have the same length")

    max_absolute = 0.0
    max_relative = 0.0
    max_ulps = 0
    mismatches = 0
    zero_mismatches = 0
    nan_mismatches = 0
    for expected_value, actual_value in zip(
        expected_values, actual_values, strict=True
    ):
        expected_bits = float32_bits(expected_value)
        actual_bits = float32_bits(actual_value)
        if expected_bits == actual_bits:
            continue
        mismatches += 1
        if expected_value == actual_value == 0.0:
            zero_mismatches += 1
        if math.isnan(expected_value) or math.isnan(actual_value):
            nan_mismatches += 1
            max_ulps = max(max_ulps, 0xFFFF_FFFF)
            max_absolute = math.inf
            max_relative = math.inf
            continue
        absolute = abs(actual_value - expected_value)
        scale = max(abs(expected_value), abs(actual_value))
        relative = absolute / scale if scale != 0.0 else 0.0
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        max_ulps = max(max_ulps, float32_ulp_distance(expected_value, actual_value))
    return Comparison(
        Deviations(max_absolute, max_relative, max_ulps),
        mismatches,
        zero_mismatches,
        nan_mismatches,
    )


def gamma_bound(unit_roundoff: float, terms: int, sum_absolute_terms: float) -> float:
    """Compute the analytic ``gamma_terms`` floating-point error envelope.

    Args:
        unit_roundoff: Unit roundoff of the accumulator format.
        terms: Number of terms in the reduction.
        sum_absolute_terms: Sum of absolute exact terms.

    Returns:
        Upper bound on accumulated absolute error.

    Examples:
        >>> gamma_bound(2.0**-24, 2, 3.0) > 0.0
        True
    """
    if (
        not math.isfinite(unit_roundoff)
        or not math.isfinite(sum_absolute_terms)
        or unit_roundoff <= 0.0
        or terms < 0
        or sum_absolute_terms < 0.0
    ):
        raise ValueError("gamma-bound inputs must be non-negative and epsilon positive")
    product = terms * unit_roundoff
    if product >= 1.0:
        raise ValueError("gamma bound is undefined when terms * unit_roundoff >= 1")
    return product / (1.0 - product) * sum_absolute_terms
