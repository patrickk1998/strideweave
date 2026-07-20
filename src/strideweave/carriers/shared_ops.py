"""Carrier-neutral operations owned by the carrier dispatch layer."""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

from ..layout import Layout, Tree
from .operation_helpers import (
    Operation,
    _as_tensor,
    _copy_gradient_to_layout,
    _layout_from_modes,
    _mode_shape,
    _mode_stride,
    _require_layout,
    _zero_tensor_like,
)


def _normalize_view_key(key: Any, rank: int) -> tuple[Any, ...]:
    normalized = key if isinstance(key, tuple) else (key,)
    if len(normalized) != rank:
        raise ValueError("View keys must include exactly one key per top-level mode")
    return normalized


def _normalize_int_key(value: Any, extent: int, name: str) -> int:
    try:
        normalized = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or slice") from exc

    if normalized < 0 or normalized >= extent:
        raise ValueError("View integer key is out of domain")
    return normalized


def _normalize_leaf_slice(key: slice, extent: int) -> tuple[int, int]:
    if key.step not in (None, 1):
        raise ValueError("View slices do not support steps")
    start, stop, _ = key.indices(extent)
    if stop <= start:
        raise ValueError("View slices must be non-empty")
    return start, stop


def _is_whole_slice(key: slice) -> bool:
    return key.start is None and key.stop is None and key.step in (None, 1)


def _view_layout_and_mapping(tensor: Any, key: Any) -> tuple[int, Layout]:
    normalized_key = _normalize_view_key(key, len(tensor.layout))
    offset_delta = 0
    output_shapes = []
    output_strides = []

    for mode_index, mode_key in enumerate(normalized_key):
        mode_layout = tensor.layout[mode_index]
        mode_shape = _mode_shape(tensor.layout, mode_index)
        mode_stride = _mode_stride(tensor.layout, mode_index)

        if isinstance(mode_key, slice):
            if mode_layout.is_leaf:
                extent = int(mode_layout.shape)
                stride = int(mode_layout.stride)
                start, stop = _normalize_leaf_slice(mode_key, extent)
                offset_delta += start * stride
                output_shapes.append(stop - start)
                output_strides.append(stride)
                continue

            if not _is_whole_slice(mode_key):
                raise ValueError("Only whole slices are supported for non-leaf modes")
            output_shapes.append(mode_shape)
            output_strides.append(mode_stride)
            continue

        if mode_layout.is_leaf:
            extent = int(mode_layout.shape)
            stride = int(mode_layout.stride)
            index = _normalize_int_key(mode_key, extent, "View key")
            offset_delta += index * stride
            continue

        extent = mode_layout.shape.logical_size
        index = _normalize_int_key(mode_key, extent, "View key")
        offset_delta += mode_layout.index(index)

    return offset_delta, _layout_from_modes(output_shapes, output_strides)


class GenericViewOperation(Operation):
    """Generic tensor view operation sharing the input backing carrier."""

    def _forward(self, tensor: Any, key: Any) -> Any:
        from ..tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        offset_delta, output_layout = _view_layout_and_mapping(tensor, key)

        self.ctx["mapping_layout"] = output_layout
        self.ctx["mapping_offset"] = offset_delta
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.carrier, tensor.offset + offset_delta, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        scatter_onto = _zero_tensor_like(tensor)
        scatter_onto.carrier.scatter(
            gradient,
            scatter_onto,
            self.ctx["mapping_layout"],
            self.ctx["mapping_offset"],
        )
        return (scatter_onto,)


class RearrangeOperation(Operation):
    """Autograd operation that rearranges a tensor layout without copying values."""

    def _forward(self, tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
        from ..tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        if not isinstance(output, Tree):
            raise TypeError("output must be a Tree")
        if selection is not None and not isinstance(selection, Tree):
            raise TypeError("selection must be a Tree or None")

        effective_selection = selection
        if effective_selection is None:
            effective_selection = Layout._default_selection_tree(tensor.layout)

        output_layout = Layout.rearrange(tensor.layout, output, effective_selection)

        self.ctx["output"] = output
        self.ctx["selection"] = effective_selection
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.carrier, tensor.offset, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        reverse_output, reverse_selection = Layout.reverse_rearrange(
            self.ctx["output"], self.ctx["selection"]
        )
        inverse_layout = Layout.rearrange(
            gradient.layout, reverse_output, reverse_selection
        )

        from ..tensor import Tensor

        inverse_gradient = Tensor(gradient.carrier, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


class PermuteOperation(Operation):
    """Autograd operation that permutes top-level layout modes."""

    def _forward(self, tensor: Any, *order: Any) -> Any:
        from ..tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        normalized_order = Layout._normalize_permute_order(order, len(tensor.layout))
        output_layout = Layout.permute(tensor.layout, normalized_order)

        self.ctx["order"] = normalized_order
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.carrier, tensor.offset, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        order = self.ctx["order"]
        inverse_order = [0] * len(order)
        for output_mode, input_mode in enumerate(order):
            inverse_order[input_mode] = output_mode
        inverse_layout = Layout.permute(gradient.layout, inverse_order)

        from ..tensor import Tensor

        inverse_gradient = Tensor(gradient.carrier, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


__all__ = [
    "GenericViewOperation",
    "PermuteOperation",
    "RearrangeOperation",
]
