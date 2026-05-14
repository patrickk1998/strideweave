from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast

from .data import Generic

Operation = cast(type[Any], import_module("neotorch._operation").Operation)


def _as_tensor(value: Any, name: str) -> Any:
    from .tensor import Tensor

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _require_exact_generic_tensor(value: Any, name: str) -> Any:
    tensor = _as_tensor(value, name)
    if type(tensor.data) is not Generic:
        raise TypeError(f"{name} must be backed by exact Generic data")
    return tensor


def _require_same_layout(lhs: Any, rhs: Any) -> None:
    if lhs.layout != rhs.layout:
        raise ValueError("Tensor layouts must match")


def _logical_values(tensor: Any) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def _detached_tensor_like(target: Any, values: Iterable[Any]) -> Any:
    from .tensor import Tensor

    data = target.data.new_like(list(values))
    return Tensor(data, 0, target.layout)


def _copy_gradient_for(target: Any, gradient: Any) -> Any:
    _require_same_layout(target, gradient)
    return _detached_tensor_like(target, _logical_values(gradient))


class GenericAddOperation(Operation):
    def forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_exact_generic_tensor(lhs, "lhs")
        rhs = _require_exact_generic_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        self.store_inputs(lhs, rhs)
        values = [lhs[i] + rhs[i] for i in range(lhs.size())]
        result = _detached_tensor_like(lhs, values)
        result.autograd_ctx = self
        return result

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_exact_generic_tensor(gradient, "gradient")
        return _copy_gradient_for(lhs, gradient), _copy_gradient_for(rhs, gradient)


def add(lhs: Any, rhs: Any) -> Any:
    return GenericAddOperation().forward(lhs, rhs)


__all__ = [
    "GenericAddOperation",
    "Operation",
    "add",
]
