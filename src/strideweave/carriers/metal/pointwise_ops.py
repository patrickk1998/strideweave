"""Metal-owned pointwise, predicate, and ternary operations."""

from __future__ import annotations

from typing import Any

from ...tensor import Tensor
from ..dtype import DType
from ..generic.numerics import normalize_storage_value
from ..generic.ternary_ops import _align_ternary_operands
from ..operation_helpers import (
    Operation,
    _align_binary_operands,
    _canonical_layout_for_shape,
    _require_layout,
    _require_live_tensor,
    _require_same_shape,
)
from ..operation_policy import OperationPlan, resolve_operation_plan
from ._executor import execute_expression
from .capabilities import _require_executable_plan
from .carrier import Metal

_UNARY_OPERATIONS = frozenset(
    {
        "abs",
        "ceil",
        "cos",
        "elu",
        "erf",
        "exp",
        "exp2",
        "floor",
        "gelu",
        "leaky_relu",
        "log",
        "log2",
        "neg",
        "recip",
        "relu",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "silu",
        "sin",
        "softplus",
        "sqrt",
        "tanh",
    }
)
_PREDICATE_OPERATIONS = frozenset({"eq", "le", "logical_not", "lt", "ne"})
_BINARY_OPERATIONS = frozenset(
    {
        "add",
        "div",
        "elementwise_mul",
        "maximum",
        "minimum",
        "rem",
        "sub",
    }
)
_TERNARY_OPERATIONS = frozenset({"clamp", "select"})
_POINTWISE_OPERATIONS = frozenset(
    {
        *_UNARY_OPERATIONS,
        *_PREDICATE_OPERATIONS,
        *_BINARY_OPERATIONS,
        *_TERNARY_OPERATIONS,
        "mul",
        "pow",
    }
)


def _is_tensor(value: Any) -> bool:
    return isinstance(value, Tensor)


def _require_metal_tensor(value: Any, name: str) -> Tensor:
    tensor = _require_live_tensor(value, name)
    if type(tensor.carrier) is not Metal:
        raise ValueError(f"{name} must use the exact Metal carrier class")
    return tensor


def _require_same_metal_carrier(tensors: tuple[Tensor, ...]) -> None:
    if any(type(tensor.carrier) is not Metal for tensor in tensors):
        raise ValueError("pointwise tensor operands must use the exact Metal carrier")


def _materialize_scalar(plan: OperationPlan, index: int, value: Any) -> Any:
    return normalize_storage_value(
        plan.operands[index].convert_to,
        value,
        name="Metal weak scalar",
    )


def _gradient_expression(
    expression: str,
    operands: tuple[Any, ...],
    target: Tensor,
) -> Tensor:
    convert_dtypes = tuple(
        DType.Bool
        if isinstance(operand, Tensor) and operand.dtype() is DType.Bool
        else DType.Float32
        for operand in operands
    )
    return execute_expression(
        expression,
        operands,
        output_layout=_canonical_layout_for_shape(target.layout.shape),
        output_dtype=DType.Float32,
        convert_dtypes=convert_dtypes,
    )


def _nondifferentiable_backward(_self: Any, _gradient: Any) -> tuple[Any, ...]:
    raise RuntimeError("predicate operation results are non-differentiable")


class MetalPointwiseOperation(Operation):
    """Private Metal operation selected by one registered dispatch name."""

    def __init__(self, operation_name: str) -> None:
        super().__init__()
        if operation_name not in _POINTWISE_OPERATIONS:
            raise ValueError(f"unknown Metal pointwise operation {operation_name!r}")
        self._metal_operation_name = operation_name

    def _forward(self, *operands: Any) -> Tensor:
        operation = self._metal_operation_name
        if operation in _UNARY_OPERATIONS or operation == "logical_not":
            return self._forward_unary(operation, operands)
        if operation in _BINARY_OPERATIONS or operation in {"eq", "le", "lt", "ne"}:
            return self._forward_binary(operation, operands)
        if operation == "mul":
            return self._forward_mul(operands)
        if operation == "pow":
            return self._forward_pow(operands)
        if operation == "select":
            return self._forward_select(operands)
        return self._forward_clamp(operands)

    def _forward_unary(self, operation: str, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 1:
            raise TypeError(f"{operation} expects one tensor operand")
        tensor = _require_metal_tensor(operands[0], "tensor")
        plan = _require_executable_plan(
            resolve_operation_plan(operation, tensor.dtype())
        )
        output_layout = _canonical_layout_for_shape(tensor.layout.shape)
        return execute_expression(
            operation,
            (tensor,),
            output_layout=output_layout,
            output_dtype=plan.output,
            plan=plan,
        )

    def _forward_binary(self, operation: str, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 2:
            raise TypeError(f"{operation} expects two tensor operands")
        lhs = _require_metal_tensor(operands[0], "lhs")
        rhs = _require_metal_tensor(operands[1], "rhs")
        _require_same_metal_carrier((lhs, rhs))
        plan = _require_executable_plan(
            resolve_operation_plan(operation, lhs.dtype(), rhs.dtype())
        )
        lhs, rhs, output_layout = _align_binary_operands(lhs, rhs)
        if self.inputs():
            self.store_inputs(lhs, rhs)
        return execute_expression(
            operation,
            (lhs, rhs),
            output_layout=output_layout,
            output_dtype=plan.output,
            plan=plan,
        )

    def _forward_mul(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 2:
            raise TypeError("mul expects a tensor and tensor or weak scalar")
        tensor = _require_metal_tensor(operands[0], "tensor")
        other = operands[1]
        if _is_tensor(other):
            other = _require_metal_tensor(other, "other")
            _require_same_metal_carrier((tensor, other))
            plan = _require_executable_plan(
                resolve_operation_plan("mul", tensor.dtype(), other.dtype())
            )
            tensor, other, output_layout = _align_binary_operands(tensor, other)
            self.ctx["binary"] = True
            if self.inputs():
                self.store_inputs(tensor, other)
            return execute_expression(
                "mul",
                (tensor, other),
                output_layout=output_layout,
                output_dtype=plan.output,
                plan=plan,
            )

        plan = _require_executable_plan(
            resolve_operation_plan("mul", tensor.dtype(), other)
        )
        scalar = _materialize_scalar(plan, 1, other)
        self.ctx["scalar"] = scalar
        return execute_expression(
            "mul",
            (tensor, scalar),
            output_layout=_canonical_layout_for_shape(tensor.layout.shape),
            output_dtype=plan.output,
            plan=plan,
        )

    def _forward_pow(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 2:
            raise TypeError("pow expects one tensor and a tensor or weak scalar")
        base, exponent = operands
        base_tensor = _is_tensor(base)
        exponent_tensor = _is_tensor(exponent)
        if base_tensor and exponent_tensor:
            base = _require_metal_tensor(base, "base")
            exponent = _require_metal_tensor(exponent, "exponent")
            _require_same_metal_carrier((base, exponent))
            plan = _require_executable_plan(
                resolve_operation_plan("pow", base.dtype(), exponent.dtype())
            )
            base, exponent, output_layout = _align_binary_operands(base, exponent)
            self.ctx["binary"] = True
            if self.inputs():
                self.store_inputs(base, exponent)
            return execute_expression(
                "pow",
                (base, exponent),
                output_layout=output_layout,
                output_dtype=plan.output,
                plan=plan,
            )
        if exponent_tensor:
            exponent = _require_metal_tensor(exponent, "exponent")
            plan = _require_executable_plan(
                resolve_operation_plan("pow", base, exponent.dtype())
            )
            scalar = _materialize_scalar(plan, 0, base)
            self.ctx["scalar_base"] = scalar
            return execute_expression(
                "pow",
                (scalar, exponent),
                output_layout=_canonical_layout_for_shape(exponent.layout.shape),
                output_dtype=plan.output,
                plan=plan,
            )
        if base_tensor:
            base = _require_metal_tensor(base, "base")
            plan = _require_executable_plan(
                resolve_operation_plan("pow", base.dtype(), exponent)
            )
            scalar = _materialize_scalar(plan, 1, exponent)
            self.ctx["scalar_exponent"] = scalar
            return execute_expression(
                "pow",
                (base, scalar),
                output_layout=_canonical_layout_for_shape(base.layout.shape),
                output_dtype=plan.output,
                plan=plan,
            )
        raise TypeError("pow requires at least one Tensor operand")

    def _forward_select(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 3:
            raise TypeError("select expects condition, on_true, and on_false")
        condition = _require_metal_tensor(operands[0], "condition")
        on_true = _require_metal_tensor(operands[1], "on_true")
        on_false = _require_metal_tensor(operands[2], "on_false")
        _require_same_metal_carrier((condition, on_true, on_false))
        plan = _require_executable_plan(
            resolve_operation_plan(
                "select", condition.dtype(), on_true.dtype(), on_false.dtype()
            )
        )
        (condition, on_true, on_false), shape = _align_ternary_operands(
            condition, on_true, on_false
        )
        if self.inputs():
            self.store_inputs(condition, on_true, on_false)
        output_layout = _canonical_layout_for_shape(shape)
        self.ctx["output_layout"] = output_layout
        return execute_expression(
            "select",
            (condition, on_true, on_false),
            output_layout=output_layout,
            output_dtype=plan.output,
            plan=plan,
        )

    def _forward_clamp(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 3:
            raise TypeError("clamp expects tensor, lower, and upper")
        tensor = _require_metal_tensor(operands[0], "tensor")
        lower, upper = operands[1:]
        lower_tensor = (
            _require_metal_tensor(lower, "lower") if _is_tensor(lower) else None
        )
        upper_tensor = (
            _require_metal_tensor(upper, "upper") if _is_tensor(upper) else None
        )
        tensor_operands = tuple(
            operand
            for operand in (tensor, lower_tensor, upper_tensor)
            if operand is not None
        )
        _require_same_metal_carrier(tensor_operands)
        plan = _require_executable_plan(
            resolve_operation_plan(
                "clamp",
                tensor.dtype(),
                lower_tensor.dtype() if lower_tensor is not None else lower,
                upper_tensor.dtype() if upper_tensor is not None else upper,
            )
        )
        aligned, shape = _align_ternary_operands(*tensor_operands)
        cursor = 0
        tensor = aligned[cursor]
        cursor += 1
        if lower_tensor is not None:
            lower_tensor = aligned[cursor]
            cursor += 1
        if upper_tensor is not None:
            upper_tensor = aligned[cursor]

        lower_operand = (
            lower_tensor
            if lower_tensor is not None
            else _materialize_scalar(plan, 1, lower)
        )
        upper_operand = (
            upper_tensor
            if upper_tensor is not None
            else _materialize_scalar(plan, 2, upper)
        )
        self.ctx["lower_is_tensor"] = lower_tensor is not None
        self.ctx["upper_is_tensor"] = upper_tensor is not None
        if lower_tensor is None:
            self.ctx["lower_value"] = lower_operand
        if upper_tensor is None:
            self.ctx["upper_value"] = upper_operand
        stored = [tensor]
        if lower_tensor is not None:
            stored.append(lower_tensor)
        if upper_tensor is not None:
            stored.append(upper_tensor)
        if self.inputs():
            self.store_inputs(*stored)
        output_layout = _canonical_layout_for_shape(shape)
        self.ctx["output_layout"] = output_layout
        return execute_expression(
            "clamp",
            (tensor, lower_operand, upper_operand),
            output_layout=output_layout,
            output_dtype=plan.output,
            plan=plan,
        )

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        operation = self._metal_operation_name
        if operation in _PREDICATE_OPERATIONS:
            return _nondifferentiable_backward(self, gradient)
        if operation in _UNARY_OPERATIONS:
            return self._backward_unary(operation, gradient)
        if operation in _BINARY_OPERATIONS:
            return self._backward_binary(operation, gradient)
        if operation == "mul":
            return self._backward_mul(gradient)
        if operation == "pow":
            return self._backward_pow(gradient)
        if operation == "select":
            return self._backward_select(gradient)
        return self._backward_clamp(gradient)

    def _backward_unary(self, operation: str, gradient: Any) -> tuple[Tensor]:
        (tensor,) = self.inputs()
        gradient = _require_metal_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)
        operands = (
            (gradient,)
            if operation in {"ceil", "floor", "neg", "round", "sign"}
            else (gradient, tensor)
        )
        return (_gradient_expression(f"grad_{operation}", operands, tensor),)

    def _backward_binary(self, operation: str, gradient: Any) -> tuple[Tensor, Tensor]:
        lhs, rhs = self.inputs()
        gradient = _require_metal_tensor(gradient, "gradient")
        _require_same_shape(lhs, gradient)
        _require_same_shape(rhs, gradient)
        if operation == "add":
            lhs_expression = rhs_expression = "grad_add"
        elif operation == "sub":
            lhs_expression, rhs_expression = "grad_sub_lhs", "grad_sub_rhs"
        elif operation in {"elementwise_mul"}:
            lhs_expression, rhs_expression = "grad_mul_lhs", "grad_mul_rhs"
        elif operation == "div":
            lhs_expression, rhs_expression = "grad_div_lhs", "grad_div_rhs"
        elif operation in {"maximum", "minimum"}:
            lhs_expression = f"grad_{operation}_lhs"
            rhs_expression = f"grad_{operation}_rhs"
        elif operation == "rem":
            lhs_expression, rhs_expression = "grad_rem_lhs", "grad_rem_rhs"
        else:
            raise AssertionError(f"unexpected Metal binary operation {operation!r}")

        if operation in {"add", "sub"}:
            lhs_operands = rhs_operands = (gradient,)
        elif operation == "elementwise_mul":
            lhs_operands = (gradient, rhs)
            rhs_operands = (gradient, lhs)
        elif operation == "div":
            lhs_operands = (gradient, rhs)
            rhs_operands = (gradient, lhs, rhs)
        elif operation == "rem":
            lhs_operands = (gradient,)
            rhs_operands = (gradient, lhs, rhs)
        else:
            lhs_operands = (gradient, lhs, rhs)
            rhs_operands = (gradient, lhs, rhs)
        return (
            _gradient_expression(lhs_expression, lhs_operands, lhs),
            _gradient_expression(rhs_expression, rhs_operands, rhs),
        )

    def _backward_mul(self, gradient: Any) -> tuple[Any, ...]:
        gradient = _require_metal_tensor(gradient, "gradient")
        if self.ctx.get("binary", False):
            lhs, rhs = self.inputs()
            _require_same_shape(lhs, gradient)
            _require_same_shape(rhs, gradient)
            return (
                _gradient_expression("grad_mul_lhs", (gradient, rhs), lhs),
                _gradient_expression("grad_mul_rhs", (gradient, lhs), rhs),
            )
        (tensor,) = self.inputs()
        _require_same_shape(tensor, gradient)
        return (
            _gradient_expression(
                "grad_mul_lhs", (gradient, self.ctx["scalar"]), tensor
            ),
        )

    def _backward_pow(self, gradient: Any) -> tuple[Any, ...]:
        gradient = _require_metal_tensor(gradient, "gradient")
        if self.ctx.get("binary", False):
            base, exponent = self.inputs()
            _require_same_shape(base, gradient)
            _require_same_shape(exponent, gradient)
            operands = (gradient, base, exponent)
            return (
                _gradient_expression("grad_pow_base", operands, base),
                _gradient_expression("grad_pow_exponent", operands, exponent),
            )
        (tensor,) = self.inputs()
        _require_same_shape(tensor, gradient)
        if "scalar_base" in self.ctx:
            return (
                _gradient_expression(
                    "grad_pow_scalar_base",
                    (gradient, self.ctx["scalar_base"], tensor),
                    tensor,
                ),
            )
        return (
            _gradient_expression(
                "grad_pow_scalar_exponent",
                (gradient, tensor, self.ctx["scalar_exponent"]),
                tensor,
            ),
        )

    def _backward_select(self, gradient: Any) -> tuple[None, Tensor, Tensor]:
        condition, on_true, on_false = self.inputs()
        gradient = _require_metal_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        return (
            None,
            _gradient_expression("grad_select_true", (condition, gradient), on_true),
            _gradient_expression("grad_select_false", (condition, gradient), on_false),
        )

    def _backward_clamp(self, gradient: Any) -> tuple[Any, ...]:
        inputs = self.inputs()
        tensor = inputs[0]
        cursor = 1
        lower = (
            inputs[cursor] if self.ctx["lower_is_tensor"] else self.ctx["lower_value"]
        )
        cursor += int(self.ctx["lower_is_tensor"])
        upper = (
            inputs[cursor] if self.ctx["upper_is_tensor"] else self.ctx["upper_value"]
        )
        gradient = _require_metal_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        operands = (gradient, tensor, lower, upper)
        gradients: list[Tensor] = [
            _gradient_expression("grad_clamp_tensor", operands, tensor)
        ]
        if isinstance(lower, Tensor):
            gradients.append(_gradient_expression("grad_clamp_lower", operands, lower))
        if isinstance(upper, Tensor):
            gradients.append(_gradient_expression("grad_clamp_upper", operands, upper))
        return tuple(gradients)


def metal_pointwise_operation(operation_name: str) -> MetalPointwiseOperation:
    """Return a fresh Metal-owned operation for a registered pointwise name."""
    return MetalPointwiseOperation(operation_name)


__all__: list[str] = []
