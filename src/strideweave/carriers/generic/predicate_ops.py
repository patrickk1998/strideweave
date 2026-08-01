"""Generic predicate operations producing non-differentiable ``Bool`` tensors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _align_binary_operands,
    _detached_tensor_like,
    _require_live_tensor,
    _tensor_with_layout_like,
)
from .execution import arithmetic_for_plan, resolve_operation_plan


def _require_float32_tensor(value: Any, name: str) -> Any:
    """Validate that one predicate operand is a concrete ``Float32`` tensor."""
    tensor = _require_live_tensor(value, name)
    if tensor.dtype() is not DType.Float32:
        raise TypeError(f"{name} must have DType.Float32 for a predicate operation")
    return tensor


def _require_operation_plan(operation: str, *tensors: Any) -> None:
    """Resolve and gate one concrete predicate plan before any materialization."""
    plan = resolve_operation_plan(operation, *(tensor.dtype() for tensor in tensors))
    arithmetic_for_plan(plan, type(tensors[0].carrier))


def _nondifferentiable_backward(_self: Any, _gradient: Any) -> tuple[Any, ...]:
    """Reject direct VJP requests; predicate results never build an autograd graph."""
    raise RuntimeError("predicate operation results are non-differentiable")


def _binary_predicate(
    owner: Any,
    operation: str,
    lhs: Any,
    rhs: Any,
    predicate: Callable[[float, float], bool],
) -> Any:
    lhs = _require_float32_tensor(lhs, "lhs")
    rhs = _require_float32_tensor(rhs, "rhs")
    _require_operation_plan(operation, lhs, rhs)
    lhs, rhs, result_layout = _align_binary_operands(lhs, rhs)
    if owner.inputs():
        owner.store_inputs(lhs, rhs)
    values = [bool(predicate(lhs[i], rhs[i])) for i in range(lhs.size())]
    return _tensor_with_layout_like(lhs, result_layout, values, DType.Bool)


class GenericEqOperation(Operation):
    """Generic Float32 equality predicate returning Bool."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_predicate(
            self, "eq", lhs, rhs, lambda left, right: left == right
        )

    backward = _nondifferentiable_backward


class GenericNeOperation(Operation):
    """Generic Float32 inequality predicate returning Bool."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_predicate(
            self, "ne", lhs, rhs, lambda left, right: left != right
        )

    backward = _nondifferentiable_backward


class GenericLtOperation(Operation):
    """Generic Float32 less-than predicate returning Bool."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_predicate(self, "lt", lhs, rhs, lambda left, right: left < right)

    backward = _nondifferentiable_backward


class GenericLeOperation(Operation):
    """Generic Float32 less-than-or-equal predicate returning Bool."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_predicate(
            self, "le", lhs, rhs, lambda left, right: left <= right
        )

    backward = _nondifferentiable_backward


class GenericLogicalNotOperation(Operation):
    """Generic Float32 logical-not predicate returning Bool."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_float32_tensor(tensor, "tensor")
        _require_operation_plan("logical_not", tensor)
        values = [tensor[i] == 0.0 for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values, DType.Bool)

    backward = _nondifferentiable_backward


__all__ = [
    "GenericEqOperation",
    "GenericLeOperation",
    "GenericLogicalNotOperation",
    "GenericLtOperation",
    "GenericNeOperation",
]
