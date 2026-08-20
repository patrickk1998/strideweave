"""Metal-owned reduction and inclusive-scan operations."""

from __future__ import annotations

import math
from typing import Any

from ...layout import Shape
from ...tensor import Tensor
from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _mode_logical_size,
    _mode_shape,
    _require_layout,
    _require_live_tensor,
    _require_two_mode_tensor,
)
from ..operation_policy import resolve_operation_plan
from ._reduction_executor import (
    execute_cumsum,
    execute_reduction_backward,
    execute_reduction_forward,
)
from .capabilities import _require_executable_plan
from .carrier import Metal

_REDUCTIONS = frozenset(
    {"argmax", "argmin", "reduce_max", "reduce_min", "reduce_prod", "reduce_sum"}
)
_DIFFERENTIABLE_REDUCTIONS = frozenset(
    {"reduce_max", "reduce_min", "reduce_prod", "reduce_sum"}
)
_OPERATIONS = frozenset({*_REDUCTIONS, "cumsum"})


def _require_metal_tensor(value: Any, name: str) -> Tensor:
    tensor = _require_live_tensor(value, name)
    if type(tensor.carrier) is not Metal:
        raise ValueError(f"{name} must use the exact Metal carrier class")
    return tensor


def _normalize_axis(axis: Any, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("cumsum dimension must be an integer top-level mode")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("cumsum dimension is out of range")
    return axis


def _gradient_layout(tensor: Tensor) -> Any:
    if tensor.layout.is_injective:
        return tensor.layout
    return _canonical_layout_for_shape(tensor.layout.shape)


class MetalReductionOperation(Operation):
    """Private Metal operation selected by one reduction or scan name."""

    def __init__(self, operation_name: str) -> None:
        super().__init__()
        if operation_name not in _OPERATIONS:
            raise ValueError(f"unknown Metal reduction operation {operation_name!r}")
        self._metal_operation_name = operation_name

    def _forward(self, *operands: Any) -> Tensor:
        if self._metal_operation_name == "cumsum":
            return self._forward_cumsum(operands)
        return self._forward_reduction(operands)

    def _forward_reduction(self, operands: tuple[Any, ...]) -> Tensor:
        operation = self._metal_operation_name
        if len(operands) != 1:
            raise TypeError(f"{operation} expects one tensor operand")
        tensor = _require_metal_tensor(operands[0], "tensor")
        tensor = _require_two_mode_tensor(tensor, "tensor")
        plan = _require_executable_plan(
            resolve_operation_plan(
                operation,
                tensor.dtype(),
                options=self._execution_options,
            )
        )
        row_count = _mode_logical_size(tensor.layout, 0)
        fiber_size = _mode_logical_size(tensor.layout, 1)
        if fiber_size <= 0:
            raise ValueError("Reduction fibers must be nonempty")
        if operation in {"argmax", "argmin"} and fiber_size > 2**31 - 1:
            raise OverflowError("Reduction fiber extent is out of Int32 range")
        output_layout = _canonical_layout_for_shape(
            Shape(_mode_shape(tensor.layout, 0))
        )
        self.ctx["output_layout"] = output_layout
        self.ctx["row_count"] = row_count
        self.ctx["fiber_size"] = fiber_size
        return execute_reduction_forward(
            operation,
            tensor,
            output_layout=output_layout,
            output_dtype=plan.output,
            row_count=row_count,
            fiber_size=fiber_size,
            plan=plan,
        )

    def _forward_cumsum(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 2:
            raise TypeError("cumsum expects a tensor and dimension")
        tensor = _require_metal_tensor(operands[0], "tensor")
        axis = _normalize_axis(operands[1], len(tensor.layout))
        extents = tuple(
            _mode_logical_size(tensor.layout, index)
            for index in range(len(tensor.layout))
        )
        if not extents or any(extent <= 0 for extent in extents):
            raise ValueError("cumsum requires nonempty tensor modes")
        plan = _require_executable_plan(
            resolve_operation_plan(
                "cumsum",
                tensor.dtype(),
                options=self._execution_options,
            )
        )
        output_layout = _canonical_layout_for_shape(tensor.layout.shape)
        axis_stride = math.prod(extents[:axis])
        self.ctx["output_layout"] = output_layout
        self.ctx["axis"] = axis
        self.ctx["axis_stride"] = axis_stride
        self.ctx["axis_extent"] = extents[axis]
        return execute_cumsum(
            tensor,
            output_layout=output_layout,
            axis_stride=axis_stride,
            axis_extent=extents[axis],
            plan=plan,
        )

    def backward(self, gradient: Any) -> tuple[Tensor]:
        operation = self._metal_operation_name
        if operation in {"argmax", "argmin"}:
            raise RuntimeError("arg reductions are not differentiable")
        (tensor,) = self.inputs()
        tensor = _require_metal_tensor(tensor, "tensor")
        gradient = _require_metal_tensor(gradient, "gradient")
        if gradient.dtype() is not DType.Float32:
            raise TypeError("Metal reduction gradients must use DType.Float32")
        _require_layout(gradient, self.ctx["output_layout"])
        output_layout = _gradient_layout(tensor)
        if operation == "cumsum":
            result = execute_cumsum(
                gradient,
                output_layout=output_layout,
                axis_stride=self.ctx["axis_stride"],
                axis_extent=self.ctx["axis_extent"],
                reverse=True,
            )
            return (result,)
        if operation not in _DIFFERENTIABLE_REDUCTIONS:
            raise AssertionError(f"unknown Metal reduction VJP {operation!r}")
        result = execute_reduction_backward(
            f"grad_{operation}",
            tensor,
            gradient,
            output_layout=output_layout,
            row_count=self.ctx["row_count"],
            fiber_size=self.ctx["fiber_size"],
        )
        return (result,)


def metal_reduction_operation(operation_name: str) -> MetalReductionOperation:
    """Return a fresh Metal-owned operation for a reduction or scan name."""
    return MetalReductionOperation(operation_name)


__all__: list[str] = []
