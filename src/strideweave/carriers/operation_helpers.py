"""Shared validation, dtype, and tensor-construction helpers for operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from numbers import Number
from typing import Any, cast

from ..layout import Layout, Shape, Stride
from .dtype import DType, storage_zero

_operation = import_module("strideweave._operation")
Operation = cast(type[Any], _operation.Operation)
execute_lowered_operation = cast(Callable[..., Any], _operation._execute_lowered)


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


def _require_same_shape(lhs: Any, rhs: Any) -> None:
    if lhs.layout.shape != rhs.layout.shape:
        raise ValueError("Tensor shapes must match")


def _require_layout(tensor: Any, layout: Layout) -> None:
    if tensor.layout != layout:
        raise ValueError("Tensor layouts must match")


def _require_shape_layout(tensor: Any, layout: Layout) -> None:
    """Require a gradient's logical shape while allowing any valid strides.

    A view can receive a cotangent materialized by another operation with the
    same hierarchical shape but different strides.  In particular, a
    stride-zero cotangent is valid when every logical coordinate is the same
    value, so only the logical shape is part of this contract.  Tensor
    construction has already validated the gradient's storage bounds.
    """
    if tensor.layout.shape != layout.shape:
        raise ValueError("Tensor gradient shape must match")


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


def _canonical_layout_for_shape(shape: Shape) -> Layout:
    stride, _ = _canonical_stride_level(shape.top_level, 1)
    return Layout(shape, Stride(stride))


def _canonical_layout_from_modes(*modes: Any) -> Layout:
    shape = _shape_from_modes(*modes)
    return _canonical_layout_for_shape(shape)


def _format_shape_profile(shape_level: Any) -> str:
    parts = (
        "leaf" if isinstance(child, int) else _format_shape_profile(child)
        for child in shape_level
    )
    return f"({', '.join(parts)})"


def _align_binary_operands(lhs: Any, rhs: Any) -> tuple[Any, Any, Layout]:
    """Align two structurally broadcast-compatible tensor operands."""
    lhs_profile = lhs.layout.profile
    rhs_profile = rhs.layout.profile
    if lhs_profile != rhs_profile:
        lhs_rendered = _format_shape_profile(lhs.layout.shape.top_level)
        rhs_rendered = _format_shape_profile(rhs.layout.shape.top_level)
        raise ValueError(
            "Tensor shape profiles are not congruent: "
            f"lhs={lhs_rendered}, rhs={rhs_rendered}. "
            "Insert singleton modes with rearrange so both profiles match."
        )

    rendered_profile = _format_shape_profile(lhs.layout.shape.top_level)

    def common_level(
        lhs_level: Any, rhs_level: Any, path: tuple[int, ...]
    ) -> list[Any]:
        common: list[Any] = []
        for index, (lhs_extent, rhs_extent) in enumerate(
            zip(lhs_level, rhs_level, strict=True)
        ):
            leaf_path = (*path, index)
            if isinstance(lhs_extent, int):
                if lhs_extent == rhs_extent or rhs_extent == 1:
                    common.append(lhs_extent)
                elif lhs_extent == 1:
                    common.append(rhs_extent)
                else:
                    position = ".".join(str(component) for component in leaf_path)
                    raise ValueError(
                        "Tensor extents are not broadcast-compatible at leaf "
                        f"{position} within profile {rendered_profile}: "
                        f"lhs={lhs_extent}, rhs={rhs_extent}"
                    )
                continue
            common.append(common_level(lhs_extent, rhs_extent, leaf_path))
        return common

    shape = Shape(
        common_level(
            lhs.layout.shape.top_level,
            rhs.layout.shape.top_level,
            (),
        )
    )
    lhs_layout = lhs.layout.broadcast_to(shape)
    rhs_layout = rhs.layout.broadcast_to(shape)
    aligned_lhs = (
        lhs
        if lhs_layout == lhs.layout
        else lhs.carrier.dispatch_op("broadcast_to").forward(lhs, shape)
    )
    aligned_rhs = (
        rhs
        if rhs_layout == rhs.layout
        else rhs.carrier.dispatch_op("broadcast_to").forward(rhs, shape)
    )
    return aligned_lhs, aligned_rhs, _canonical_layout_for_shape(shape)


def _layout_from_modes(shapes: Iterable[Any], strides: Iterable[Any]) -> Layout:
    return Layout(Shape(list(shapes)), Stride(list(strides)))


def _physical_values_for_layout(
    layout: Layout, logical_values: Iterable[Any], hole: Any = None
) -> list[Any]:
    """Place logical values at their physical slots, filling the holes.

    A layout whose strides leave gaps has physical slots no logical index
    addresses. Those slots still have to hold something the storage dtype can
    represent, so concrete storage fills them with that dtype's zero; legacy
    opaque storage keeps ``None``.
    """
    values = list(logical_values)
    if len(values) != layout.shape.logical_size:
        raise ValueError("Logical values length must match layout size")

    cache = layout._cache
    physical_values: list[Any] = [hole] * cache.cosize
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

    storage_dtype = target.carrier.dtype() if dtype is None else dtype
    values = _physical_values_for_layout(
        layout, logical_values, storage_zero(storage_dtype)
    )
    if dtype is None:
        carrier = target.carrier.new_like(values)
    else:
        carrier = target.carrier.new_like(values, dtype=dtype)
    return Tensor(carrier, 0, layout)


def _detached_tensor_like(
    target: Any, values: Iterable[Any], dtype: DType | None = None
) -> Any:
    layout = target.layout
    if not layout.is_injective:
        layout = _canonical_layout_for_shape(layout.shape)
    return _tensor_with_layout_like(target, layout, values, dtype)


def _zero_tensor_like(target: Any) -> Any:
    return _detached_tensor_like(target, [0] * target.size())


def _copy_gradient_for(target: Any, gradient: Any) -> Any:
    _require_same_layout(target, gradient)
    return _detached_tensor_like(target, _logical_values(gradient))


def _copy_gradient_to_layout(target: Any, gradient: Any) -> Any:
    if target.size() != gradient.size():
        raise ValueError("Tensor layouts must have the same logical size")
    return _detached_tensor_like(target, _logical_values(gradient))
