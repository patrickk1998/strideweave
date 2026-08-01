"""Direct regression tests for Generic's plan-owned accumulation helpers."""

from __future__ import annotations

import math

import pytest

from strideweave import DType, Generic
from strideweave.carriers.generic.execution import arithmetic_for_plan
from strideweave.carriers.generic.numerics import float32_scalar
from strideweave.carriers.operation_policy import resolve_operation_plan


def planned_total(operation: str, values: list[float]):
    """Execute one resolved Float32 accumulation directly through its adapter."""
    plan = resolve_operation_plan(operation, DType.Float32)
    arithmetic = arithmetic_for_plan(plan, Generic)
    return arithmetic.total([float32_scalar(value) for value in values])


@pytest.mark.parametrize(
    ("operation", "values", "negative_zero"),
    [
        ("reduce_max", [0.0, -0.0], False),
        ("reduce_max", [-0.0, 0.0], False),
        ("reduce_min", [0.0, -0.0], True),
        ("reduce_min", [-0.0, 0.0], True),
    ],
)
def test_planned_extrema_total_use_numpy_signed_zero(
    operation: str, values: list[float], negative_zero: bool
) -> None:
    result = planned_total(operation, values)

    assert (math.copysign(1.0, float(result)) < 0) is negative_zero


@pytest.mark.parametrize("operation", ["reduce_max", "reduce_min"])
def test_planned_extrema_total_retains_the_first_nan(operation: str) -> None:
    plan = resolve_operation_plan(operation, DType.Float32)
    arithmetic = arithmetic_for_plan(plan, Generic)
    values = [
        float32_scalar(1.0),
        float32_scalar(float("nan")),
        float32_scalar(2.0),
        float32_scalar(float("nan")),
    ]

    result = arithmetic.total(values)

    assert math.isnan(float(result))
    assert result is values[1]


@pytest.mark.parametrize(
    ("operation", "values", "expected"),
    [
        ("argmax", [1.0, 4.0, 4.0], 1),
        ("argmin", [1.0, 1.0, 2.0], 0),
        ("argmax", [1.0, float("nan"), float("nan")], 1),
        ("argmin", [1.0, float("nan"), float("nan")], 1),
        ("argmax", [-0.0, 0.0], 0),
        ("argmin", [-0.0, 0.0], 0),
    ],
)
def test_planned_arg_extrema_total_uses_first_nan_and_first_winner(
    operation: str, values: list[float], expected: int
) -> None:
    assert planned_total(operation, values) == expected
