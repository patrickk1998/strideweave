"""Shared validation, dtype, and tensor-construction helpers for operations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from importlib import import_module
from numbers import Integral, Number
from typing import Any, cast

from .data import DataType
from .layout import Layout, Shape, Stride

_operation = import_module("neotorch._operation")
Operation = cast(type[Any], _operation.Operation)
_is_grad_enabled = cast(Callable[[], bool], _operation.is_grad_enabled)
_set_grad_enabled = cast(Callable[[bool], None], _operation.set_grad_enabled)


def _as_tensor(value: Any, name: str) -> Any:
    from .tensor import Tensor

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _dispatch_unary(operation_name: str, tensor: Any) -> Any:
    tensor = _as_tensor(tensor, "tensor")
    return type(tensor.data).dispatch_op(operation_name)


def _dispatch_binary(operation_name: str, lhs: Any, rhs: Any) -> Any:
    lhs = _as_tensor(lhs, "lhs")
    rhs = _as_tensor(rhs, "rhs")
    if type(lhs.data) is not type(rhs.data):
        raise TypeError("Tensor backing data classes must match")
    return type(lhs.data).dispatch_op(operation_name)


def _require_unevicted_tensor(value: Any, name: str) -> Any:
    tensor = _as_tensor(value, name)
    if tensor.data.is_evicted():
        raise RuntimeError(f"{name} data is evicted")
    return tensor


def _require_same_layout(lhs: Any, rhs: Any) -> None:
    if lhs.layout != rhs.layout:
        raise ValueError("Tensor layouts must match")


def _require_layout(tensor: Any, layout: Layout) -> None:
    if tensor.layout != layout:
        raise ValueError("Tensor layouts must match")


def _require_two_mode_tensor(tensor: Any, name: str) -> Any:
    tensor = _require_unevicted_tensor(tensor, name)
    if len(tensor.layout) != 2:
        raise ValueError(f"{name} must have a two-mode layout")
    return tensor


def _require_number(value: Any, name: str) -> Number:
    if not isinstance(value, Number):
        raise TypeError(f"{name} must be a numerical scalar")
    return value


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


def _logical_values(tensor: Any) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def _mode_shape(layout: Layout, mode: int) -> Any:
    return layout.shape.top_level[mode]


def _mode_logical_size(layout: Layout, mode: int) -> int:
    shape = _mode_shape(layout, mode)
    if isinstance(shape, int):
        return shape
    return cast(int, shape.logical_size)


def _mode_stride(layout: Layout, mode: int) -> Any:
    return layout.stride.top_level[mode]


def _shape_from_modes(*modes: Any) -> Shape:
    if len(modes) == 1:
        return Shape(modes[0])
    return Shape(list(modes))


def _canonical_stride_level(shape_level: Any, stride: int) -> tuple[Any, int]:
    if isinstance(shape_level, int):
        return stride, stride * shape_level

    stride_level = []
    next_stride = stride
    for shape in shape_level:
        child_stride, next_stride = _canonical_stride_level(shape, next_stride)
        stride_level.append(child_stride)
    return stride_level, next_stride


def _canonical_layout_from_modes(*modes: Any) -> Layout:
    shape = _shape_from_modes(*modes)
    stride, _ = _canonical_stride_level(shape.top_level, 1)
    return Layout(shape, Stride(stride))


def _layout_from_modes(shapes: Iterable[Any], strides: Iterable[Any]) -> Layout:
    return Layout(Shape(list(shapes)), Stride(list(strides)))


def _physical_values_for_layout(
    layout: Layout, logical_values: Iterable[Any]
) -> list[Any]:
    values = list(logical_values)
    if len(values) != layout.shape.logical_size:
        raise ValueError("Logical values length must match layout size")

    cache = layout._cache
    physical_values: list[Any] = [None] * cache.cosize
    for logical_index, value in enumerate(values):
        physical_values[cache.get_index(logical_index)] = value
    return physical_values


def _tensor_with_layout_like(
    target: Any,
    layout: Layout,
    logical_values: Iterable[Any],
    dtype: DataType | None = None,
) -> Any:
    from .tensor import Tensor

    values = _physical_values_for_layout(layout, logical_values)
    if dtype is None:
        data = target.data.new_like(values)
    else:
        data = target.data.new_like(values, dtype=dtype)
    return Tensor(data, 0, layout)


def _detached_tensor_like(
    target: Any, values: Iterable[Any], dtype: DataType | None = None
) -> Any:
    return _tensor_with_layout_like(target, target.layout, values, dtype)


def _zero_tensor_like(target: Any) -> Any:
    return _detached_tensor_like(target, [0] * target.size())


def _copy_gradient_for(target: Any, gradient: Any) -> Any:
    _require_same_layout(target, gradient)
    return _detached_tensor_like(target, _logical_values(gradient))


def _copy_gradient_to_layout(target: Any, gradient: Any) -> Any:
    if target.size() != gradient.size():
        raise ValueError("Tensor layouts must have the same logical size")
    return _tensor_with_layout_like(target, target.layout, _logical_values(gradient))
