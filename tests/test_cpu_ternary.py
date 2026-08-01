"""Focused native CPU tests for select and clamp."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest

import strideweave as sw
from strideweave import CPU, DType, Layout, Shape, Stride, Tensor


def _tensor(
    values: Iterable[float | bool],
    *,
    dtype: DType = DType.Float32,
    layout: Layout | None = None,
) -> Tensor:
    logical_values = list(values)
    layout = layout or Layout(Shape(len(logical_values)), Stride(1))
    carrier = CPU(layout.cosize, dtype=dtype)
    for index, value in enumerate(logical_values):
        carrier[layout.index(index)] = value
    return Tensor(carrier, 0, layout)


def _values(tensor: Tensor) -> list[float | bool]:
    return [tensor[index] for index in range(tensor.size())]


def test_cpu_select_is_native_canonical_and_masks_branch_gradients() -> None:
    gapped = Layout(Shape(4), Stride(2))
    condition = _tensor([True, False, True, False], dtype=DType.Bool, layout=gapped)
    on_true = _tensor([1.0, math.nan, 3.0, math.nan], layout=gapped)
    on_false = _tensor([math.nan, 2.0, math.nan, 4.0], layout=gapped)

    result = sw.select(condition, on_true, on_false)
    result.backward(_tensor([1.0, 2.0, 3.0, 4.0]))

    assert type(result.autograd_ctx).__name__ == "_CPUSelectOperation"
    assert result.dtype() is DType.Float32
    assert result.layout == Layout(Shape(4), Stride(1))
    assert _values(result) == [1.0, 2.0, 3.0, 4.0]
    assert on_true.grad is not None
    assert _values(on_true.grad) == [1.0, 0.0, 3.0, 0.0]
    assert on_false.grad is not None
    assert _values(on_false.grad) == [0.0, 2.0, 0.0, 4.0]


def test_cpu_clamp_is_native_and_uses_staged_equal_vjp() -> None:
    tensor = _tensor([1.0])
    lower = _tensor([1.0])
    upper = _tensor([1.0])

    result = sw.clamp(tensor, lower, upper)
    result.backward(_tensor([8.0]))

    assert type(result.autograd_ctx).__name__ == "_CPUClampOperation"
    assert _values(result) == [1.0]
    assert tensor.grad is not None
    assert lower.grad is not None
    assert upper.grad is not None
    assert _values(tensor.grad) == [2.0]
    assert _values(lower.grad) == [2.0]
    assert _values(upper.grad) == [4.0]


@pytest.mark.parametrize(
    ("tensor_value", "lower", "upper", "negative_zero"),
    [
        (-0.0, -0.0, -0.0, True),
        (0.0, 0.0, 0.0, False),
        (-0.0, 0.0, 0.0, False),
        (0.0, -0.0, -0.0, True),
    ],
)
def test_cpu_clamp_matches_numpy_signed_zero_composition(
    tensor_value: float,
    lower: float,
    upper: float,
    negative_zero: bool,
) -> None:
    result = sw.clamp(_tensor([tensor_value]), lower, upper)

    assert (math.copysign(1.0, result[0]) < 0.0) is negative_zero


def test_cpu_clamp_scalar_bounds_are_ordered_and_canonical() -> None:
    tensor = _tensor([-2.0, 0.5, 3.0], layout=Layout(Shape(3), Stride(2)))

    result = sw.clamp(tensor, 2.0, -1.0)

    assert _values(result) == [-1.0, -1.0, -1.0]
    assert result.layout == Layout(Shape(3), Stride(1))
