"""Shared validation, dtype, and tensor-construction helpers for operations."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from numbers import Number
from typing import Any, cast

from ..layout import Layout, Shape, Stride
from .dtype import DType

_operation = import_module("strideweave._operation")
Operation = cast(type[Any], _operation.Operation)


def _as_tensor(value: Any, name: str) -> Any:
    from ..tensor import Tensor

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _require_live_tensor(value: Any, name: str) -> Any:
    tensor = _as_tensor(value, name)
    if tensor.carrier.is_released():
        raise RuntimeError(f"{name} carrier is released")
    return tensor


def _require_same_layout(lhs: Any, rhs: Any) -> None:
    if lhs.layout != rhs.layout:
        raise ValueError("Tensor layouts must match")


def _require_layout(tensor: Any, layout: Layout) -> None:
    if tensor.layout != layout:
        raise ValueError("Tensor layouts must match")


def _require_two_mode_tensor(tensor: Any, name: str) -> Any:
    tensor = _require_live_tensor(tensor, name)
    if len(tensor.layout) != 2:
        raise ValueError(f"{name} must have a two-mode layout")
    return tensor


def _require_number(value: Any, name: str) -> Number:
    if not isinstance(value, Number):
        raise TypeError(f"{name} must be a numerical scalar")
    return value


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
    dtype: DType | None = None,
) -> Any:
    from ..tensor import Tensor

    values = _physical_values_for_layout(layout, logical_values)
    if dtype is None:
        carrier = target.carrier.new_like(values)
    else:
        carrier = target.carrier.new_like(values, dtype=dtype)
    return Tensor(carrier, 0, layout)


def _detached_tensor_like(
    target: Any, values: Iterable[Any], dtype: DType | None = None
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
