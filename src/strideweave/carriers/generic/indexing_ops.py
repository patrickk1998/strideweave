"""Generic reference implementations for section-7 indexing primitives.

The public dispatch and capability wiring lives with the carrier integration.
This module deliberately owns only the reference mapping, validation, and VJPs
so it can be exercised independently while that wiring is assembled.
"""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _require_layout,
    _require_live_tensor,
    _tensor_with_layout_like,
)
from .execution import arithmetic_for_plan, resolve_operation_plan
from .numerics import binary32, float32_errstate, float32_scalar

__all__ = [
    "GenericGatherOperation",
    "GenericScatterAddOperation",
    "GenericScatterOperation",
]


def _normalize_axis(axis: Any, rank: int) -> int:
    """Normalize one top-level axis using Python's negative-axis convention."""
    try:
        normalized = operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {normalized} is out of range for rank {rank}")
    return normalized


def _require_dtype(tensor: Any, name: str, dtype: DType) -> Any:
    tensor = _require_live_tensor(tensor, name)
    if tensor.dtype() is not dtype:
        raise TypeError(f"{name} must have dtype DType.{dtype.name}")
    return tensor


def _require_operation_plan(operation: str, *tensors: Any) -> None:
    """Resolve and gate one concrete indexing plan before materialization."""
    plan = resolve_operation_plan(operation, *(tensor.dtype() for tensor in tensors))
    arithmetic_for_plan(plan, type(tensors[0].carrier))


def _mode_extent(mode: Any) -> int:
    return mode if isinstance(mode, int) else mode.logical_size


def _output_shape(base: Any, indices: Any, axis: int) -> Any:
    from ...layout import Shape

    base_modes = base.layout.shape.top_level
    index_modes = indices.layout.shape.top_level
    return Shape(
        [
            *base_modes[:axis],
            *index_modes,
            *base_modes[axis + 1 :],
        ]
    )


def _top_level_coords(shape: Any, ordinal: int) -> tuple[int, ...]:
    """Decode a logical ordinal into first-mode-fast top-level mode ordinals."""
    from ...layout import Layout

    return tuple(Layout.expand_int(ordinal, shape.top_level))


def _ordinal_from_top_level_coords(shape: Any, coords: tuple[int, ...]) -> int:
    """Encode top-level mode ordinals in StrideWeave's first-mode-fast order."""
    ordinal = 0
    factor = 1
    if len(coords) != len(shape.top_level):
        raise ValueError("Coordinate rank does not match tensor shape")
    for mode, coordinate in zip(shape.top_level, coords, strict=True):
        extent = _mode_extent(mode)
        if coordinate < 0 or coordinate >= extent:
            raise ValueError("Coordinate is outside tensor shape")
        ordinal += coordinate * factor
        factor *= extent
    return ordinal


def _tensor_at_top_level_coords(tensor: Any, coords: tuple[int, ...]) -> Any:
    """Read a tensor with one flattened ordinal per top-level mode."""
    return tensor[coords]


def _validate_indices(indices: Any, extent: int, *, unique: bool) -> tuple[int, ...]:
    """Materialize and validate every logical index before result allocation."""
    values: list[int] = []
    seen: set[int] = set()
    for logical_index in range(indices.size()):
        try:
            value = operator_index(indices[logical_index])
        except TypeError as exc:
            raise TypeError("indices must contain only integers") from exc
        if value < 0 or value >= extent:
            raise IndexError(
                f"index {value} is out of range for selected axis extent {extent}"
            )
        if unique and value in seen:
            raise ValueError("scatter indices must contain distinct logical values")
        seen.add(value)
        values.append(value)
    return tuple(values)


def _validate_common(
    base: Any,
    indices: Any,
    axis: Any,
    *,
    operation: str,
    unique: bool,
    data_name: str = "base",
    updates: Any | None = None,
) -> tuple[Any, Any, int, Any, Any, tuple[int, ...]]:
    base = _require_dtype(base, data_name, DType.Float32)
    indices = _require_dtype(indices, "indices", DType.Int32)
    if updates is None:
        _require_operation_plan(operation, base, indices)
    else:
        updates = _require_dtype(updates, "updates", DType.Float32)
        _require_operation_plan(operation, base, indices, updates)
    normalized_axis = _normalize_axis(axis, len(base.layout))
    extent = _mode_extent(base.layout.shape.top_level[normalized_axis])
    values = _validate_indices(indices, extent, unique=unique)
    output_shape = _output_shape(base, indices, normalized_axis)
    output_layout = _canonical_layout_for_shape(output_shape)
    return base, indices, normalized_axis, output_shape, output_layout, values


def _gather_values(
    source: Any,
    indices: Any,
    axis: int,
    output_shape: Any,
    validated_indices: tuple[int, ...] | None = None,
) -> list[Any]:
    """Read the gather mapping in first-mode-fast output order."""
    index_rank = len(indices.layout)
    if validated_indices is None:
        extent = _mode_extent(source.layout.shape.top_level[axis])
        validated_indices = _validate_indices(indices, extent, unique=False)
    values: list[Any] = []
    with float32_errstate():
        for output_ordinal in range(output_shape.logical_size):
            output_coords = _top_level_coords(output_shape, output_ordinal)
            index_ordinal = _ordinal_from_top_level_coords(
                indices.layout.shape,
                output_coords[axis : axis + index_rank],
            )
            source_coords = (
                *output_coords[:axis],
                validated_indices[index_ordinal],
                *output_coords[axis + index_rank :],
            )
            values.append(binary32(_tensor_at_top_level_coords(source, source_coords)))
    return values


def _scatter_values(
    base: Any,
    indices: Any,
    updates: Any,
    axis: int,
    output_shape: Any,
    validated_indices: tuple[int, ...],
    *,
    add: bool,
) -> list[Any]:
    """Materialize a canonical base copy and apply updates functionally."""
    base_shape = base.layout.shape
    result = [binary32(base[logical_index]) for logical_index in range(base.size())]
    index_rank = len(indices.layout)
    with float32_errstate():
        for update_ordinal in range(updates.size()):
            output_coords = _top_level_coords(output_shape, update_ordinal)
            index_ordinal = _ordinal_from_top_level_coords(
                indices.layout.shape,
                output_coords[axis : axis + index_rank],
            )
            base_coords = (
                *output_coords[:axis],
                validated_indices[index_ordinal],
                *output_coords[axis + index_rank :],
            )
            destination = _ordinal_from_top_level_coords(base_shape, base_coords)
            update = float32_scalar(updates[update_ordinal])
            if add:
                result[destination] = binary32(
                    float32_scalar(result[destination]) + update
                )
            else:
                result[destination] = binary32(update)
    return result


def _require_updates(updates: Any, output_shape: Any) -> Any:
    updates = _require_dtype(updates, "updates", DType.Float32)
    if (
        updates.layout.shape != output_shape
        or updates.layout.profile != _canonical_layout_for_shape(output_shape).profile
    ):
        raise ValueError(
            "updates must have exactly the gather-result shape and profile"
        )
    return updates


class GenericGatherOperation(Operation):
    """Generic Float32 gather operation over one top-level logical axis."""

    def _forward(self, tensor: Any, indices: Any, axis: Any) -> Any:
        tensor, indices, normalized_axis, output_shape, output_layout, values = (
            _validate_common(
                tensor,
                indices,
                axis,
                operation="gather",
                unique=False,
                data_name="tensor",
            )
        )
        self.ctx["axis"] = normalized_axis
        self.ctx["output_layout"] = output_layout
        self.ctx["output_shape"] = output_shape
        return _tensor_with_layout_like(
            tensor,
            output_layout,
            _gather_values(tensor, indices, normalized_axis, output_shape, values),
            DType.Float32,
        )

    def backward(self, gradient: Any) -> tuple[Any, None]:
        (tensor, indices) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        axis = self.ctx["axis"]
        output_shape = self.ctx["output_shape"]
        extent = _mode_extent(tensor.layout.shape.top_level[axis])
        index_values = _validate_indices(indices, extent, unique=False)
        grad_values = [binary32(0.0)] * tensor.size()
        index_rank = len(indices.layout)
        with float32_errstate():
            for output_ordinal in range(output_shape.logical_size):
                output_coords = _top_level_coords(output_shape, output_ordinal)
                index_ordinal = _ordinal_from_top_level_coords(
                    indices.layout.shape,
                    output_coords[axis : axis + index_rank],
                )
                source_coords = (
                    *output_coords[:axis],
                    index_values[index_ordinal],
                    *output_coords[axis + index_rank :],
                )
                destination = _ordinal_from_top_level_coords(
                    tensor.layout.shape, source_coords
                )
                grad_values[destination] = binary32(
                    float32_scalar(grad_values[destination])
                    + float32_scalar(gradient[output_ordinal])
                )
        return _detached_tensor_like(tensor, grad_values, DType.Float32), None


class GenericScatterOperation(Operation):
    """Generic functional overwrite scatter with distinct logical indices."""

    def _forward(self, base: Any, indices: Any, updates: Any, axis: Any) -> Any:
        (
            base,
            indices,
            normalized_axis,
            output_shape,
            _,
            values,
        ) = _validate_common(
            base,
            indices,
            axis,
            operation="scatter",
            unique=True,
            updates=updates,
        )
        updates = _require_updates(updates, output_shape)
        output_layout = _canonical_layout_for_shape(base.layout.shape)
        self.ctx["axis"] = normalized_axis
        self.ctx["output_layout"] = output_layout
        self.ctx["output_shape"] = output_shape
        self.ctx["indices"] = values
        result_values = _scatter_values(
            base,
            indices,
            updates,
            normalized_axis,
            output_shape,
            values,
            add=False,
        )
        return _tensor_with_layout_like(
            base, output_layout, result_values, DType.Float32
        )

    def backward(self, gradient: Any) -> tuple[Any, None, Any]:
        base, indices, updates = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        axis = self.ctx["axis"]
        output_shape = self.ctx["output_shape"]
        index_values = self.ctx["indices"]
        index_rank = len(indices.layout)

        base_gradient = [binary32(gradient[i]) for i in range(base.size())]
        with float32_errstate():
            for update_ordinal in range(updates.size()):
                output_coords = _top_level_coords(output_shape, update_ordinal)
                index_ordinal = _ordinal_from_top_level_coords(
                    indices.layout.shape,
                    output_coords[axis : axis + index_rank],
                )
                base_coords = (
                    *output_coords[:axis],
                    index_values[index_ordinal],
                    *output_coords[axis + index_rank :],
                )
                destination = _ordinal_from_top_level_coords(
                    base.layout.shape, base_coords
                )
                base_gradient[destination] = binary32(0.0)

        update_gradient_values = _gather_values(
            gradient,
            indices,
            axis,
            output_shape,
            index_values,
        )
        return (
            _detached_tensor_like(base, base_gradient, DType.Float32),
            None,
            _detached_tensor_like(updates, update_gradient_values, DType.Float32),
        )


class GenericScatterAddOperation(Operation):
    """Generic functional scatter-add with deterministic binary32 accumulation."""

    def _forward(self, base: Any, indices: Any, updates: Any, axis: Any) -> Any:
        (
            base,
            indices,
            normalized_axis,
            output_shape,
            _,
            values,
        ) = _validate_common(
            base,
            indices,
            axis,
            operation="scatter_add",
            unique=False,
            updates=updates,
        )
        updates = _require_updates(updates, output_shape)
        output_layout = _canonical_layout_for_shape(base.layout.shape)
        self.ctx["axis"] = normalized_axis
        self.ctx["output_layout"] = output_layout
        self.ctx["output_shape"] = output_shape
        self.ctx["indices"] = values
        result_values = _scatter_values(
            base,
            indices,
            updates,
            normalized_axis,
            output_shape,
            values,
            add=True,
        )
        return _tensor_with_layout_like(
            base, output_layout, result_values, DType.Float32
        )

    def backward(self, gradient: Any) -> tuple[Any, None, Any]:
        base, indices, updates = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        update_gradient_values = _gather_values(
            gradient,
            indices,
            self.ctx["axis"],
            self.ctx["output_shape"],
            self.ctx["indices"],
        )
        return (
            _detached_tensor_like(
                base,
                [binary32(gradient[i]) for i in range(base.size())],
                DType.Float32,
            ),
            None,
            _detached_tensor_like(updates, update_gradient_values, DType.Float32),
        )
