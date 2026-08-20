"""Metal-owned indexing, functional scatter, sort, and top-k operations."""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

from ...layout import Shape
from ...tensor import Tensor
from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _mode_logical_size,
    _require_layout,
    _require_live_tensor,
)
from ..operation_policy import resolve_operation_plan
from ._indexing_executor import (
    execute_gather,
    execute_gather_backward,
    execute_scatter,
    execute_scatter_base_backward,
    execute_scatter_updates_backward,
    execute_selection,
    execute_selection_backward,
    validate_indices,
)
from .capabilities import _require_executable_plan
from .carrier import Metal

_INDEXING_OPERATIONS = frozenset({"gather", "scatter", "scatter_add"})
_SELECTION_OPERATIONS = frozenset(
    {"_sort_values", "_sort_indices", "_topk_values", "_topk_indices"}
)
_OPERATIONS = frozenset({*_INDEXING_OPERATIONS, *_SELECTION_OPERATIONS})


def _require_metal_tensor(value: Any, name: str) -> Tensor:
    tensor = _require_live_tensor(value, name)
    if type(tensor.carrier) is not Metal:
        raise ValueError(f"{name} must use the exact Metal carrier class")
    return tensor


def _require_dtype(tensor: Tensor, name: str, dtype: DType) -> Tensor:
    if tensor.dtype() is not dtype:
        raise TypeError(f"{name} must have dtype DType.{dtype.name}")
    return tensor


def _normalize_axis(axis: Any, rank: int) -> int:
    try:
        normalized = operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {normalized} is out of range for rank {rank}")
    return normalized


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _normalize_count(k: Any, extent: int) -> int:
    try:
        normalized = operator_index(k)
    except TypeError as exc:
        raise TypeError("k must be an integer") from exc
    if normalized < 1 or normalized > extent:
        raise ValueError("k must satisfy 1 <= k <= selected axis extent")
    return normalized


def _output_shape(base: Tensor, indices: Tensor, axis: int) -> Shape:
    base_modes = base.layout.shape.top_level
    return Shape(
        [
            *base_modes[:axis],
            *indices.layout.shape.top_level,
            *base_modes[axis + 1 :],
        ]
    )


def _require_updates(updates: Tensor, output_shape: Shape) -> Tensor:
    canonical = _canonical_layout_for_shape(output_shape)
    if (
        updates.layout.shape != output_shape
        or updates.layout.profile != canonical.profile
    ):
        raise ValueError(
            "updates must have exactly the gather-result shape and profile"
        )
    return updates


def _gradient_layout(tensor: Tensor) -> Any:
    if tensor.layout.is_injective:
        return tensor.layout
    return _canonical_layout_for_shape(tensor.layout.shape)


class MetalIndexingOperation(Operation):
    """Private Metal operation selected by one indexing or selection name."""

    def __init__(self, operation_name: str) -> None:
        super().__init__()
        if operation_name not in _OPERATIONS:
            raise ValueError(f"unknown Metal indexing operation {operation_name!r}")
        self._metal_operation_name = operation_name

    def _forward(self, *operands: Any) -> Tensor:
        if self._metal_operation_name == "gather":
            return self._forward_gather(operands)
        if self._metal_operation_name in {"scatter", "scatter_add"}:
            return self._forward_scatter(operands)
        return self._forward_selection(operands)

    def _forward_gather(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 3:
            raise TypeError("gather expects tensor, indices, and axis")
        tensor = _require_dtype(
            _require_metal_tensor(operands[0], "tensor"), "tensor", DType.Float32
        )
        indices = _require_dtype(
            _require_metal_tensor(operands[1], "indices"), "indices", DType.Int32
        )
        plan = _require_executable_plan(
            resolve_operation_plan("gather", tensor.dtype(), indices.dtype())
        )
        axis = _normalize_axis(operands[2], len(tensor.layout))
        axis_extent = _mode_logical_size(tensor.layout, axis)
        validate_indices(indices, extent=axis_extent, unique=False)
        output_shape = _output_shape(tensor, indices, axis)
        output_layout = _canonical_layout_for_shape(output_shape)
        axis_stride = 1
        for mode in range(axis):
            axis_stride *= _mode_logical_size(tensor.layout, mode)
        self.ctx["axis"] = axis
        self.ctx["axis_extent"] = axis_extent
        self.ctx["axis_stride"] = axis_stride
        self.ctx["index_size"] = indices.size()
        self.ctx["output_layout"] = output_layout
        return execute_gather(
            tensor,
            indices,
            output_layout=output_layout,
            axis_stride=axis_stride,
            axis_extent=axis_extent,
            plan=plan,
        )

    def _forward_scatter(self, operands: tuple[Any, ...]) -> Tensor:
        operation = self._metal_operation_name
        if len(operands) != 4:
            raise TypeError(f"{operation} expects base, indices, updates, and axis")
        base = _require_dtype(
            _require_metal_tensor(operands[0], "base"), "base", DType.Float32
        )
        indices = _require_dtype(
            _require_metal_tensor(operands[1], "indices"), "indices", DType.Int32
        )
        updates = _require_dtype(
            _require_metal_tensor(operands[2], "updates"),
            "updates",
            DType.Float32,
        )
        plan = _require_executable_plan(
            resolve_operation_plan(
                operation, base.dtype(), indices.dtype(), updates.dtype()
            )
        )
        axis = _normalize_axis(operands[3], len(base.layout))
        output_shape = _output_shape(base, indices, axis)
        updates = _require_updates(updates, output_shape)
        axis_extent = _mode_logical_size(base.layout, axis)
        validate_indices(
            indices,
            extent=axis_extent,
            unique=operation == "scatter",
        )
        output_layout = _canonical_layout_for_shape(base.layout.shape)
        axis_stride = 1
        for mode in range(axis):
            axis_stride *= _mode_logical_size(base.layout, mode)
        self.ctx["axis"] = axis
        self.ctx["axis_extent"] = axis_extent
        self.ctx["axis_stride"] = axis_stride
        self.ctx["index_size"] = indices.size()
        self.ctx["output_layout"] = output_layout
        return execute_scatter(
            operation,
            base,
            indices,
            updates,
            output_layout=output_layout,
            axis_stride=axis_stride,
            axis_extent=axis_extent,
            plan=plan,
        )

    def _forward_selection(self, operands: tuple[Any, ...]) -> Tensor:
        operation = self._metal_operation_name
        topk = operation.startswith("_topk")
        expected = 4 if topk else 3
        if len(operands) < expected - 2 or len(operands) > expected:
            raise TypeError(
                "topk expects tensor and k" if topk else "sort expects a tensor"
            )
        tensor = _require_dtype(
            _require_metal_tensor(operands[0], "tensor"), "tensor", DType.Float32
        )
        plan = _require_executable_plan(
            resolve_operation_plan(operation, tensor.dtype())
        )
        if len(tensor.layout) == 0:
            raise ValueError("selection operations require a non-scalar tensor")
        if topk:
            assert len(operands) >= 2
            axis = _normalize_axis(
                operands[2] if len(operands) >= 3 else -1, len(tensor.layout)
            )
            descending = _require_bool(
                operands[3] if len(operands) >= 4 else True, "largest"
            )
            k = _normalize_count(operands[1], _mode_logical_size(tensor.layout, axis))
        else:
            axis = _normalize_axis(
                operands[1] if len(operands) >= 2 else -1, len(tensor.layout)
            )
            descending = _require_bool(
                operands[2] if len(operands) >= 3 else False, "descending"
            )
            k = _mode_logical_size(tensor.layout, axis)
        axis_extent = _mode_logical_size(tensor.layout, axis)
        if axis_extent <= 0:
            raise ValueError("selected axis must have a positive extent")
        if axis_extent > 2**31 - 1:
            raise OverflowError("selected axis extent is out of int32 range")
        top_level = list(tensor.layout.shape.top_level)
        if topk:
            top_level[axis] = k
        output_layout = _canonical_layout_for_shape(Shape(top_level))
        axis_stride = 1
        for mode in range(axis):
            axis_stride *= _mode_logical_size(tensor.layout, mode)
        self.ctx["axis_extent"] = axis_extent
        self.ctx["axis_stride"] = axis_stride
        self.ctx["descending"] = descending
        self.ctx["k"] = k
        self.ctx["output_layout"] = output_layout
        return execute_selection(
            operation,
            tensor,
            output_layout=output_layout,
            axis_stride=axis_stride,
            axis_extent=axis_extent,
            k=k,
            descending=descending,
            output_dtype=plan.output,
            plan=plan,
        )

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        operation = self._metal_operation_name
        if operation in {"_sort_indices", "_topk_indices"}:
            raise RuntimeError("selection index results are non-differentiable")
        if operation == "gather":
            tensor, indices = self.inputs()
            gradient = _require_dtype(
                _require_metal_tensor(gradient, "gradient"),
                "gradient",
                DType.Float32,
            )
            _require_layout(gradient, self.ctx["output_layout"])
            result = execute_gather_backward(
                indices,
                gradient,
                output_layout=_gradient_layout(tensor),
                axis_stride=self.ctx["axis_stride"],
                axis_extent=self.ctx["axis_extent"],
                source_size=tensor.size(),
            )
            return result, None
        if operation in {"scatter", "scatter_add"}:
            base, indices, updates = self.inputs()
            gradient = _require_dtype(
                _require_metal_tensor(gradient, "gradient"),
                "gradient",
                DType.Float32,
            )
            _require_layout(gradient, self.ctx["output_layout"])
            base_gradient = execute_scatter_base_backward(
                operation,
                indices,
                gradient,
                output_layout=_gradient_layout(base),
                axis_stride=self.ctx["axis_stride"],
                axis_extent=self.ctx["axis_extent"],
            )
            updates_gradient = execute_scatter_updates_backward(
                indices,
                gradient,
                output_layout=_gradient_layout(updates),
                axis_stride=self.ctx["axis_stride"],
                axis_extent=self.ctx["axis_extent"],
            )
            return base_gradient, None, updates_gradient

        (tensor,) = self.inputs()
        gradient = _require_dtype(
            _require_metal_tensor(gradient, "gradient"),
            "gradient",
            DType.Float32,
        )
        _require_layout(gradient, self.ctx["output_layout"])
        result = execute_selection_backward(
            tensor,
            gradient,
            output_layout=_gradient_layout(tensor),
            axis_stride=self.ctx["axis_stride"],
            axis_extent=self.ctx["axis_extent"],
            k=self.ctx["k"],
            descending=self.ctx["descending"],
        )
        return (result,)


def metal_indexing_operation(operation_name: str) -> MetalIndexingOperation:
    """Return a fresh Metal-owned operation for an indexing or selection name."""
    return MetalIndexingOperation(operation_name)


__all__: list[str] = []
