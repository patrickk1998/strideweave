"""Stable Float32 sort and top-k operations for the generic carrier.

The operations in this module intentionally expose the single-output pieces
of selection.  A higher-level integrator packages the values and indices
results into the public multi-result API.
"""

from __future__ import annotations

import math
from functools import cmp_to_key
from operator import index as operator_index
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _mode_logical_size,
    _require_layout,
    _require_live_tensor,
    _tensor_with_layout_like,
)
from .execution import arithmetic_for_plan, resolve_operation_plan
from .numerics import INT32_MAX, binary32

__all__ = [
    "GenericSortIndicesOperation",
    "GenericSortValuesOperation",
    "GenericTopKIndicesOperation",
    "GenericTopKValuesOperation",
]


def _require_float32_tensor(tensor: Any, name: str) -> Any:
    """Validate a live Float32 tensor and return it."""

    tensor = _require_live_tensor(tensor, name)
    if tensor.dtype() is not DType.Float32:
        raise TypeError("selection operations require a Float32 tensor")
    return tensor


def _require_operation_plan(operation: str, tensor: Any) -> None:
    """Resolve and gate a concrete selection plan before materialization."""
    plan = resolve_operation_plan(operation, tensor.dtype())
    arithmetic_for_plan(plan, type(tensor.carrier))


def _normalize_axis(tensor: Any, axis: Any) -> int:
    """Normalize a top-level axis using Python's index protocol."""

    try:
        axis = operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc

    rank = len(tensor.layout)
    if rank == 0:
        raise ValueError("selection operations require a non-scalar tensor")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("axis out of range")
    return axis


def _normalize_count(k: Any, extent: int) -> int:
    """Validate a top-k count against the selected axis extent."""

    try:
        k = operator_index(k)
    except TypeError as exc:
        raise TypeError("k must be an integer") from exc
    if k < 1 or k > extent:
        raise ValueError("k must satisfy 1 <= k <= selected axis extent")
    return k


def _require_bool(value: Any, name: str) -> bool:
    """Require an actual bool for an ordering mode."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _mode_extents(tensor: Any) -> tuple[int, ...]:
    """Return logical extents of the tensor's top-level modes."""

    return tuple(
        _mode_logical_size(tensor.layout, mode) for mode in range(len(tensor.layout))
    )


def _logical_coordinates(logical: int, extents: tuple[int, ...]) -> list[int]:
    """Decode a first-mode-fast logical ordinal into top-level coordinates."""

    coordinates: list[int] = []
    remainder = logical
    for extent in extents:
        coordinates.append(remainder % extent)
        remainder //= extent
    return coordinates


def _logical_ordinal(coordinates: list[int], extents: tuple[int, ...]) -> int:
    """Encode top-level coordinates using first-mode-fast ordering."""

    logical = 0
    factor = 1
    for coordinate, extent in zip(coordinates, extents, strict=True):
        logical += coordinate * factor
        factor *= extent
    return logical


def _compare_values(
    lhs: tuple[float, int], rhs: tuple[float, int], descending: bool
) -> int:
    """Compare values with the selection ordering contract."""

    lhs_value, lhs_ordinal = lhs
    rhs_value, rhs_ordinal = rhs
    lhs_nan = math.isnan(lhs_value)
    rhs_nan = math.isnan(rhs_value)

    # NaNs form one final (or initial, for descending) equivalence class.
    if lhs_nan or rhs_nan:
        if lhs_nan and rhs_nan:
            return lhs_ordinal - rhs_ordinal
        if descending:
            return -1 if lhs_nan else 1
        return 1 if lhs_nan else -1

    # Equality intentionally includes signed zero.  Equal values retain the
    # source ordinal order in both ascending and descending modes.
    if lhs_value == rhs_value:
        return lhs_ordinal - rhs_ordinal
    if descending:
        return -1 if lhs_value > rhs_value else 1
    return -1 if lhs_value < rhs_value else 1


def _ordered_axis_ordinals(values: list[float], descending: bool) -> list[int]:
    """Return stable source ordinals for one selection fiber."""

    tagged = [(value, ordinal) for ordinal, value in enumerate(values)]
    tagged.sort(key=cmp_to_key(lambda lhs, rhs: _compare_values(lhs, rhs, descending)))
    return [ordinal for _, ordinal in tagged]


def _selection_shape(tensor: Any, axis: int, k: int | None) -> Any:
    """Return the output shape, replacing a top-level mode for top-k."""

    if k is None:
        return tensor.layout.shape
    top_level = list(tensor.layout.shape.top_level)
    top_level[axis] = k
    # Import lazily to keep this module's import surface aligned with the
    # operation helpers used by the other generic operations.
    from ...layout import Shape

    return Shape(top_level)


def _selection_forward(
    tensor: Any,
    axis: int,
    descending: bool,
    k: int | None,
) -> tuple[Any, Any, tuple[int, ...], Any]:
    """Compute logical values, source ordinals, and the canonical output."""

    input_extents = _mode_extents(tensor)
    axis_extent = input_extents[axis]
    if axis_extent <= 0:
        raise ValueError("selected axis must have a positive extent")
    if axis_extent > INT32_MAX:
        raise OverflowError("selected axis extent is out of int32 range")

    output_shape = _selection_shape(tensor, axis, k)
    output_extents = tuple(
        level if isinstance(level, int) else level.logical_size
        for level in output_shape.top_level
    )
    output_layout = _canonical_layout_for_shape(output_shape)
    output_size = 1
    for extent in output_extents:
        output_size *= extent

    fiber_orders: dict[tuple[int, ...], tuple[list[float], list[int]]] = {}
    result_values: list[float] = []
    result_indices: list[int] = []
    source_logicals: list[int] = []

    for output_logical in range(output_size):
        output_coordinates = _logical_coordinates(output_logical, output_extents)
        fiber_key = tuple(
            coordinate
            for mode, coordinate in enumerate(output_coordinates)
            if mode != axis
        )
        fiber = fiber_orders.get(fiber_key)
        if fiber is None:
            source_coordinates = [0] * len(input_extents)
            key_index = 0
            for mode in range(len(input_extents)):
                if mode != axis:
                    source_coordinates[mode] = fiber_key[key_index]
                    key_index += 1
            fiber_values = []
            for axis_ordinal in range(axis_extent):
                source_coordinates[axis] = axis_ordinal
                source_logical = _logical_ordinal(source_coordinates, input_extents)
                fiber_values.append(float(tensor[source_logical]))
            fiber = (
                fiber_values,
                _ordered_axis_ordinals(fiber_values, descending),
            )
            fiber_orders[fiber_key] = fiber

        fiber_values, ordered_ordinals = fiber
        selected_position = output_coordinates[axis]
        source_axis_ordinal = ordered_ordinals[selected_position]
        source_coordinates = [0] * len(input_extents)
        key_index = 0
        for mode in range(len(input_extents)):
            if mode == axis:
                source_coordinates[mode] = source_axis_ordinal
            else:
                source_coordinates[mode] = fiber_key[key_index]
                key_index += 1
        source_logical = _logical_ordinal(source_coordinates, input_extents)
        result_values.append(binary32(fiber_values[source_axis_ordinal]))
        result_indices.append(source_axis_ordinal)
        source_logicals.append(source_logical)

    return (
        _tensor_with_layout_like(
            tensor, output_layout, result_values, dtype=DType.Float32
        ),
        _tensor_with_layout_like(
            tensor, output_layout, result_indices, dtype=DType.Int32
        ),
        tuple(source_logicals),
        output_layout,
    )


def _backward_values(operation: Any, gradient: Any) -> tuple[Any]:
    """Route a values cotangent to the source logical ordinals."""

    gradient = _require_float32_tensor(gradient, "gradient")
    _require_layout(gradient, operation.ctx["output_layout"])
    tensor = operation.ctx["input_tensor"]
    source_logicals = operation.ctx["source_logicals"]
    gradient_values = [0.0] * tensor.size()
    for output_logical, source_logical in enumerate(source_logicals):
        gradient_values[source_logical] = binary32(float(gradient[output_logical]))
    return (_detached_tensor_like(tensor, gradient_values, dtype=DType.Float32),)


class GenericSortValuesOperation(Operation):
    """Stable Float32 sort values operation for the generic carrier."""

    def _forward(self, tensor: Any, axis: Any = -1, descending: Any = False) -> Any:
        tensor = _require_float32_tensor(tensor, "tensor")
        _require_operation_plan("_sort_values", tensor)
        axis = _normalize_axis(tensor, axis)
        descending = _require_bool(descending, "descending")
        values, _, source_logicals, output_layout = _selection_forward(
            tensor, axis, descending, None
        )
        self.ctx["input_tensor"] = tensor
        self.ctx["source_logicals"] = source_logicals
        self.ctx["output_layout"] = output_layout
        return values

    def backward(self, gradient: Any) -> tuple[Any]:
        """Route a full sort cotangent back through its source permutation."""

        return _backward_values(self, gradient)


class GenericSortIndicesOperation(Operation):
    """Stable Float32 sort source-ordinal operation for the generic carrier."""

    def _forward(self, tensor: Any, axis: Any = -1, descending: Any = False) -> Any:
        tensor = _require_float32_tensor(tensor, "tensor")
        _require_operation_plan("_sort_indices", tensor)
        axis = _normalize_axis(tensor, axis)
        descending = _require_bool(descending, "descending")
        _, indices, _, _ = _selection_forward(tensor, axis, descending, None)
        return indices

    def backward(self, gradient: Any) -> tuple[None]:
        """Indices are integer and therefore non-differentiable."""

        return (None,)


class GenericTopKValuesOperation(Operation):
    """Stable Float32 top-k values operation for the generic carrier."""

    def _forward(self, tensor: Any, k: Any, axis: Any = -1, largest: Any = True) -> Any:
        tensor = _require_float32_tensor(tensor, "tensor")
        _require_operation_plan("_topk_values", tensor)
        axis = _normalize_axis(tensor, axis)
        largest = _require_bool(largest, "largest")
        axis_extent = _mode_logical_size(tensor.layout, axis)
        k = _normalize_count(k, axis_extent)
        values, _, source_logicals, output_layout = _selection_forward(
            tensor, axis, largest, k
        )
        self.ctx["input_tensor"] = tensor
        self.ctx["source_logicals"] = source_logicals
        self.ctx["output_layout"] = output_layout
        return values

    def backward(self, gradient: Any) -> tuple[Any]:
        """Scatter selected top-k cotangents into a zero input gradient."""

        return _backward_values(self, gradient)


class GenericTopKIndicesOperation(Operation):
    """Stable Float32 top-k source-ordinal operation for the generic carrier."""

    def _forward(self, tensor: Any, k: Any, axis: Any = -1, largest: Any = True) -> Any:
        tensor = _require_float32_tensor(tensor, "tensor")
        _require_operation_plan("_topk_indices", tensor)
        axis = _normalize_axis(tensor, axis)
        largest = _require_bool(largest, "largest")
        axis_extent = _mode_logical_size(tensor.layout, axis)
        k = _normalize_count(k, axis_extent)
        _, indices, _, _ = _selection_forward(tensor, axis, largest, k)
        return indices

    def backward(self, gradient: Any) -> tuple[None]:
        """Indices are integer and therefore non-differentiable."""

        return (None,)
