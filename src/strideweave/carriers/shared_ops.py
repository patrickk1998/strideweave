"""Carrier-neutral operations owned by the carrier dispatch layer."""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

from ..core._representation import Subtensor, TensorRepresentation
from ..layout import Layout, Node, Shape, Stride, Tree
from .dtype import storage_zero
from .operation_helpers import (
    Operation,
    _as_tensor,
    _canonical_layout_for_shape,
    _canonical_stride_level,
    _copy_gradient_to_layout,
    _detached_tensor_like,
    _layout_from_modes,
    _mode_shape,
    _mode_stride,
    _require_layout,
    _require_shape_layout,
    _zero_tensor_like,
)


def _is_multi_subtensor(tensor: Any) -> bool:
    """Return whether ``tensor`` uses the generalized multi-plane form."""
    return not bool(tensor._representation.is_single_subtensor)


def _transform_c0_representation(tensor: Any, transform: Any) -> Any:
    """Apply ``transform`` to c0-domain layouts and rebuild a validated view.

    Placement layout ``L_0`` and adjacent layout ``S_0`` share the outermost
    logical domain.  Deeper levels are intentionally left untouched by every
    v0 layout view.
    """
    from ..tensor import Tensor

    representation = tensor._representation
    subtensors = tuple(
        Subtensor(
            subtensor.dtype,
            subtensor.carrier,
            subtensor.offset,
            transform(subtensor.layout) if level == 0 else subtensor.layout,
        )
        for level, subtensor in enumerate(representation.subtensors)
    )
    adjacent = tuple(
        transform(layout) if level == 0 else layout
        for level, layout in enumerate(representation.adjacent_layouts)
    )
    output = TensorRepresentation(representation.logical_dtype, subtensors, adjacent)
    return Tensor._from_representation(output)


def _transform_gradient_c0(tensor: Any, transform: Any) -> Any:
    """Transform a cotangent's c0-domain layouts without copying its planes."""
    return _transform_c0_representation(tensor, transform)


def _rearrange_c0_layout(layout: Layout, output: Tree, selection: Tree) -> Layout:
    """Rearrange one layout's top-level modes through the shared tree algebra."""
    return Layout.rearrange(layout, output, selection)


def _top_level_selection(rank: int) -> Tree:
    return Tree(*(Node.Leaf for _ in range(rank)))


def _unsqueeze_trees(rank: int, dim: int) -> tuple[Tree, Tree]:
    output: list[Any] = [Node.id(index) for index in range(rank)]
    output.insert(dim, Node.Leaf)
    return Tree(*output), _top_level_selection(rank)


def _squeeze_trees(rank: int, dim: int) -> tuple[Tree, Tree]:
    output = [Node.id(index) for index in range(rank) if index != dim]
    return Tree(*output), _top_level_selection(rank)


def _normalize_dim(value: Any, rank: int, *, insertion: bool, name: str) -> int:
    try:
        dim = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    upper = rank + 1 if insertion else rank
    if dim < 0:
        dim += upper
    if dim < 0 or dim >= upper:
        raise ValueError(f"{name} is out of range for rank {rank}")
    return dim


class BroadcastOperation(Operation):
    """Autograd operation that widens singleton layout leaves using stride zero."""

    def _forward(self, tensor: Any, target: Shape) -> Any:
        tensor = _as_tensor(tensor, "tensor")
        if not isinstance(target, Shape):
            raise TypeError("target must be a Shape")

        output_layout = tensor.layout.broadcast_to(target)
        mapping_layout = _canonical_layout_for_shape(tensor.layout.shape).broadcast_to(
            target
        )
        self.ctx["mapping_layout"] = mapping_layout
        self.ctx["output_layout"] = output_layout
        return _transform_c0_representation(
            tensor, lambda layout: layout.broadcast_to(target)
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        output_layout = self.ctx["output_layout"]
        if _is_multi_subtensor(tensor):
            # Compound dtypes are not currently differentiable, but preserving
            # the validated representation here keeps direct operation VJPs
            # coherent for custom carriers that opt into that capability.
            return (
                _transform_gradient_c0(
                    gradient,
                    lambda layout: Layout(
                        tensor.layout.shape,
                        layout.stride,
                    ),
                ),
            )
        if (
            gradient.layout.shape != output_layout.shape
            or not gradient.layout.is_injective
        ):
            raise ValueError(
                "Broadcast gradient must have the output shape in an injective layout"
            )

        mapping_layout = self.ctx["mapping_layout"]
        totals: list[Any | None] = [None] * tensor.size()
        for logical_index in range(gradient.size()):
            input_index = mapping_layout.index(logical_index)
            contribution = gradient[logical_index]
            current = totals[input_index]
            totals[input_index] = (
                contribution if current is None else current + contribution
            )
        if any(total is None for total in totals):
            raise RuntimeError("Broadcast backward did not cover every input element")
        return (_detached_tensor_like(tensor, totals),)


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


def _normalize_leaf_slice(key: slice, extent: int) -> tuple[int, int, int]:
    """Normalize one positive-step leaf slice and enforce exact divisibility."""
    try:
        step = 1 if key.step is None else operator_index(key.step)
    except TypeError as exc:
        raise TypeError("View slice step must be an integer") from exc
    if step <= 0:
        raise ValueError("View slices require a positive step")

    start, stop, step = key.indices(extent)
    if stop <= start:
        raise ValueError("View slices must be non-empty")
    if (stop - start) % step:
        raise ValueError("View slice interval must be divisible by its step")
    return start, stop, step


def _is_whole_slice(key: slice) -> bool:
    return key.start is None and key.stop is None and key.step in (None, 1)


def _view_layout_and_mapping(
    layout: Layout, key: Any, *, require_zero_origin: bool = False
) -> tuple[int, Layout]:
    normalized_key = _normalize_view_key(key, len(layout))
    offset_delta = 0
    output_shapes = []
    output_strides = []

    for mode_index, mode_key in enumerate(normalized_key):
        mode_layout = layout[mode_index]
        mode_shape = _mode_shape(layout, mode_index)
        mode_stride = _mode_stride(layout, mode_index)

        if isinstance(mode_key, slice):
            if mode_layout.is_leaf:
                extent = int(mode_layout.shape)
                stride = int(mode_layout.stride)
                start, stop, step = _normalize_leaf_slice(mode_key, extent)
                if require_zero_origin and start != 0:
                    raise NotImplementedError(
                        "Multi-subtensor slices require every selected leaf to "
                        "have normalized start 0"
                    )
                offset_delta += start * stride
                output_shapes.append((stop - start) // step)
                output_strides.append(stride * step)
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
            if require_zero_origin and index != 0:
                raise NotImplementedError(
                    "Multi-subtensor slices require every selected leaf to have "
                    "normalized start 0"
                )
            offset_delta += index * stride
            continue

        extent = mode_layout.shape.logical_size
        index = _normalize_int_key(mode_key, extent, "View key")
        if require_zero_origin and index != 0:
            raise NotImplementedError(
                "Multi-subtensor slices require every selected leaf to have "
                "normalized start 0"
            )
        offset_delta += mode_layout.index(index)

    return offset_delta, _layout_from_modes(output_shapes, output_strides)


def _scatter_multi_subtensor_view_gradient(
    tensor: Any, gradient: Any, mapping: Layout
) -> Any:
    """Scatter a compound view cotangent into fresh input-shaped plane storage."""
    from ..tensor import Tensor

    source = tensor._representation
    cotangent = gradient._representation
    if cotangent.logical_dtype is not source.logical_dtype:
        raise ValueError("View gradient dtype must match the input dtype")
    if len(cotangent.subtensors) != len(source.subtensors):
        raise ValueError("View gradient representation must match the input")

    result_subtensors = []
    for level, (target, incoming) in enumerate(
        zip(source.subtensors, cotangent.subtensors, strict=True)
    ):
        values = [storage_zero(target.dtype)] * target.layout.cosize
        if level == 0:
            for output_index in range(mapping.size):
                input_index = mapping.index(output_index)
                values[target.layout.index(input_index)] = incoming.carrier.get_value(
                    incoming.offset + incoming.layout.index(output_index)
                )
        else:
            if incoming.layout != target.layout:
                raise ValueError(
                    "View gradient deeper placement layouts must match the input"
                )
            for logical_index in range(target.layout.shape.logical_size):
                values[target.layout.index(logical_index)] = incoming.carrier.get_value(
                    incoming.offset + incoming.layout.index(logical_index)
                )
        result_subtensors.append(
            Subtensor(
                target.dtype,
                target.carrier.new_like(values),
                0,
                target.layout,
            )
        )

    return Tensor._from_representation(
        TensorRepresentation(
            source.logical_dtype,
            tuple(result_subtensors),
            source.adjacent_layouts,
        )
    )


class GenericViewOperation(Operation):
    """Generic tensor view operation sharing the input backing carrier."""

    def _forward(self, tensor: Any, key: Any) -> Any:
        from ..tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        multi_subtensor = _is_multi_subtensor(tensor)
        offset_delta, output_layout = _view_layout_and_mapping(
            tensor.layout, key, require_zero_origin=multi_subtensor
        )
        canonical_input = _canonical_layout_for_shape(tensor.layout.shape)
        mapping_offset, mapping_layout = _view_layout_and_mapping(
            canonical_input, key, require_zero_origin=multi_subtensor
        )

        self.ctx["mapping_layout"] = mapping_layout
        self.ctx["mapping_offset"] = mapping_offset
        self.ctx["output_layout"] = output_layout
        if multi_subtensor:

            def transform(layout: Layout) -> Layout:
                transformed_offset, transformed = _view_layout_and_mapping(
                    layout, key, require_zero_origin=True
                )
                if transformed_offset != 0:
                    raise RuntimeError(
                        "A zero-origin multi-subtensor slice changed a layout origin"
                    )
                return transformed

            return _transform_c0_representation(tensor, transform)
        return Tensor(tensor.carrier, tensor.offset + offset_delta, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        if _is_multi_subtensor(tensor):
            return (
                _scatter_multi_subtensor_view_gradient(
                    tensor, gradient, self.ctx["mapping_layout"]
                ),
            )
        scatter_onto = _zero_tensor_like(tensor)
        scatter_onto.carrier.scatter(
            gradient,
            scatter_onto,
            self.ctx["mapping_layout"],
            self.ctx["mapping_offset"],
        )
        return (scatter_onto,)


def _reshape_layout(layout: Layout, target_shape: Shape) -> Layout:
    """Return a first-mode-fast zero-copy reshape of one layout.

    A reshape is representable only when the source layout coalesces to one
    logical leaf.  The resulting target hierarchy receives compact strides
    scaled by that leaf's base stride, preserving physical holes and aliases.
    """
    if not isinstance(target_shape, Shape):
        raise TypeError("target_shape must be a Shape")
    if target_shape.logical_size != layout.shape.logical_size:
        raise ValueError(
            "reshape target must have the same logical size as the input layout"
        )

    coalesced = Layout.coalesce(layout)
    if len(coalesced) != 1:
        raise ValueError("reshape requires a layout that coalesces to one logical leaf")
    base_stride = int(coalesced.stride)
    target_stride, _ = _canonical_stride_level(target_shape.top_level, base_stride)
    return Layout(target_shape, Stride(target_stride))


class ReshapeOperation(Operation):
    """Autograd operation that reshapes the outermost layout without copying."""

    def _forward(self, tensor: Any, target_shape: Shape) -> Any:
        tensor = _as_tensor(tensor, "tensor")
        if not isinstance(target_shape, Shape):
            raise TypeError("target_shape must be a Shape")

        output_layout = _reshape_layout(tensor.layout, target_shape)
        self.ctx["output_layout"] = output_layout
        self.ctx["target_shape"] = target_shape
        return _transform_c0_representation(
            tensor, lambda layout: _reshape_layout(layout, target_shape)
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        from ..tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        input_shape = tensor.layout.shape
        if _is_multi_subtensor(tensor):
            return (
                _transform_gradient_c0(
                    gradient, lambda layout: _reshape_layout(layout, input_shape)
                ),
            )

        inverse_layout = _reshape_layout(gradient.layout, input_shape)
        inverse_gradient = Tensor(gradient.carrier, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


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
        from ..tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_shape_layout(gradient, self.ctx["output_layout"])

        reverse_output, reverse_selection = Layout.reverse_rearrange(
            self.ctx["output"], self.ctx["selection"]
        )
        inverse_layout = Layout.rearrange(
            gradient.layout, reverse_output, reverse_selection
        )

        inverse_gradient = Tensor(gradient.carrier, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


class PermuteOperation(Operation):
    """Autograd operation that permutes top-level layout modes."""

    def _forward(self, tensor: Any, *order: Any) -> Any:
        tensor = _as_tensor(tensor, "tensor")
        normalized_order = Layout._normalize_permute_order(order, len(tensor.layout))
        output_layout = Layout.permute(tensor.layout, normalized_order)

        self.ctx["order"] = normalized_order
        self.ctx["output_layout"] = output_layout
        return _transform_c0_representation(
            tensor, lambda layout: Layout.permute(layout, normalized_order)
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        from ..tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        order = self.ctx["order"]
        inverse_order = [0] * len(order)
        for output_mode, input_mode in enumerate(order):
            inverse_order[input_mode] = output_mode
        inverse_layout = Layout.permute(gradient.layout, inverse_order)

        if _is_multi_subtensor(tensor):
            return (
                _transform_gradient_c0(
                    gradient, lambda layout: Layout.permute(layout, inverse_order)
                ),
            )

        inverse_gradient = Tensor(gradient.carrier, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


class UnsqueezeOperation(Operation):
    """Autograd operation that inserts one top-level singleton layout mode."""

    def _forward(self, tensor: Any, dim: Any) -> Any:
        tensor = _as_tensor(tensor, "tensor")
        rank = len(tensor.layout)
        normalized_dim = _normalize_dim(dim, rank, insertion=True, name="dim")
        output, selection = _unsqueeze_trees(rank, normalized_dim)

        def transform(layout: Layout) -> Layout:
            return _rearrange_c0_layout(layout, output, selection)

        self.ctx["output"] = output
        self.ctx["selection"] = selection
        self.ctx["output_layout"] = transform(tensor.layout)
        return _transform_c0_representation(tensor, transform)

    def backward(self, gradient: Any) -> tuple[Any]:
        from ..tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        reverse_output, reverse_selection = Layout.reverse_rearrange(
            self.ctx["output"], self.ctx["selection"]
        )

        def inverse(layout: Layout) -> Layout:
            return Layout.rearrange(layout, reverse_output, reverse_selection)

        if _is_multi_subtensor(tensor):
            return (_transform_gradient_c0(gradient, inverse),)
        inverse_gradient = Tensor(
            gradient.carrier, gradient.offset, inverse(gradient.layout)
        )
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


class SqueezeOperation(Operation):
    """Autograd operation that removes one explicitly selected singleton mode."""

    def _forward(self, tensor: Any, dim: Any) -> Any:
        tensor = _as_tensor(tensor, "tensor")
        rank = len(tensor.layout)
        normalized_dim = _normalize_dim(dim, rank, insertion=False, name="dim")
        selected = tensor.layout[normalized_dim]
        if not selected.is_leaf or int(selected.shape) != 1:
            raise ValueError("squeeze dim must select a top-level extent-one leaf")
        output, selection = _squeeze_trees(rank, normalized_dim)

        def transform(layout: Layout) -> Layout:
            return _rearrange_c0_layout(layout, output, selection)

        self.ctx["output"] = output
        self.ctx["selection"] = selection
        self.ctx["output_layout"] = transform(tensor.layout)
        return _transform_c0_representation(tensor, transform)

    def backward(self, gradient: Any) -> tuple[Any]:
        from ..tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        reverse_output, reverse_selection = Layout.reverse_rearrange(
            self.ctx["output"], self.ctx["selection"]
        )

        def inverse(layout: Layout) -> Layout:
            return Layout.rearrange(layout, reverse_output, reverse_selection)

        if _is_multi_subtensor(tensor):
            return (_transform_gradient_c0(gradient, inverse),)
        inverse_gradient = Tensor(
            gradient.carrier, gradient.offset, inverse(gradient.layout)
        )
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


__all__ = [
    "BroadcastOperation",
    "GenericViewOperation",
    "PermuteOperation",
    "RearrangeOperation",
    "ReshapeOperation",
    "SqueezeOperation",
    "UnsqueezeOperation",
]
