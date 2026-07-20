"""Generic (Python-backed) autograd operation classes."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_from_modes,
    _copy_gradient_for,
    _detached_tensor_like,
    _mode_logical_size,
    _mode_shape,
    _require_layout,
    _require_live_tensor,
    _require_number,
    _require_same_layout,
    _require_two_mode_tensor,
    _tensor_with_layout_like,
)
from .helpers import (
    _LEAKY_RELU_NEGATIVE_SLOPE,
    _gelu_derivative,
    _gelu_value,
    _generic_binary_dtype,
    _generic_pow_dtype,
    _generic_scalar_mul_dtype,
    _sigmoid_value,
    _softplus_value,
)


def _unary_elementwise_operation(
    class_name: str,
    docstring: str,
    *,
    compute: Callable[[Any], tuple[Any, Any]],
    gradient_multiplier: Callable[[Any, Any], Any],
    result_dtype: DType | None = DType.Floating,
) -> type[Any]:
    """Build a Generic unary elementwise operation class.

    ``compute`` maps one input value to ``(output_value, saved_value)``;
    saved values are stored in the autograd context for the backward pass.
    ``gradient_multiplier`` maps ``(input_value, saved_value)`` to the local
    derivative that scales the incoming gradient.
    """

    def _forward(self: Any, tensor: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")

        values = []
        saved_values = []
        for i in range(tensor.size()):
            value, saved = compute(tensor[i])
            values.append(value)
            saved_values.append(saved)
        self.ctx["saved_values"] = saved_values
        return _detached_tensor_like(tensor, values, result_dtype)

    def backward(self: Any, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        saved_values = self.ctx["saved_values"]
        values = [
            gradient[i] * gradient_multiplier(tensor[i], saved_values[i])
            for i in range(gradient.size())
        ]
        return (_detached_tensor_like(tensor, values),)

    return type(
        class_name,
        (Operation,),
        {"__doc__": docstring, "_forward": _forward, "backward": backward},
    )


def _exp_compute(value: Any) -> tuple[Any, Any]:
    output = math.exp(value)
    return output, output


def _sigmoid_compute(value: Any) -> tuple[Any, Any]:
    output = _sigmoid_value(value)
    return output, output


def _tanh_compute(value: Any) -> tuple[Any, Any]:
    output = math.tanh(value)
    return output, output


def _silu_compute(value: Any) -> tuple[Any, Any]:
    sigmoid = _sigmoid_value(value)
    return value * sigmoid, sigmoid


GenericExpOperation = _unary_elementwise_operation(
    "GenericExpOperation",
    "Generic elementwise exponential operation.",
    compute=_exp_compute,
    gradient_multiplier=lambda _value, output: output,
)

GenericReLUOperation = _unary_elementwise_operation(
    "GenericReLUOperation",
    "Generic elementwise rectified linear unit operation.",
    compute=lambda value: (max(0, value), None),
    gradient_multiplier=lambda value, _saved: 1 if value > 0 else 0,
    result_dtype=None,
)

GenericSigmoidOperation = _unary_elementwise_operation(
    "GenericSigmoidOperation",
    "Generic elementwise logistic sigmoid operation.",
    compute=_sigmoid_compute,
    gradient_multiplier=lambda _value, output: output * (1.0 - output),
)

GenericTanhOperation = _unary_elementwise_operation(
    "GenericTanhOperation",
    "Generic elementwise hyperbolic tangent operation.",
    compute=_tanh_compute,
    gradient_multiplier=lambda _value, output: 1.0 - output**2,
)

GenericGELUOperation = _unary_elementwise_operation(
    "GenericGELUOperation",
    "Generic elementwise Gaussian error linear unit operation.",
    compute=lambda value: (_gelu_value(float(value)), None),
    gradient_multiplier=lambda value, _saved: _gelu_derivative(float(value)),
)

GenericSiLUOperation = _unary_elementwise_operation(
    "GenericSiLUOperation",
    "Generic elementwise sigmoid linear unit operation.",
    compute=_silu_compute,
    gradient_multiplier=lambda value, sigmoid: (
        sigmoid + value * sigmoid * (1.0 - sigmoid)
    ),
)

GenericSoftplusOperation = _unary_elementwise_operation(
    "GenericSoftplusOperation",
    "Generic elementwise softplus operation.",
    compute=lambda value: (_softplus_value(value), None),
    gradient_multiplier=lambda value, _saved: _sigmoid_value(value),
)

GenericELUOperation = _unary_elementwise_operation(
    "GenericELUOperation",
    "Generic elementwise exponential linear unit operation.",
    compute=lambda value: (
        float(value) if value > 0 else math.expm1(value),
        None,
    ),
    gradient_multiplier=lambda value, _saved: 1 if value > 0 else math.exp(value),
)

GenericLeakyReLUOperation = _unary_elementwise_operation(
    "GenericLeakyReLUOperation",
    "Generic elementwise leaky rectified linear unit operation.",
    compute=lambda value: (
        float(value) if value >= 0 else _LEAKY_RELU_NEGATIVE_SLOPE * value,
        None,
    ),
    gradient_multiplier=lambda value, _saved: (
        1 if value >= 0 else _LEAKY_RELU_NEGATIVE_SLOPE
    ),
)


class GenericAddOperation(Operation):
    """Generic elementwise tensor addition operation with autograd support."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_live_tensor(lhs, "lhs")
        rhs = _require_live_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        dtype = _generic_binary_dtype(lhs, rhs)
        values = [lhs[i] + rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        return _copy_gradient_for(lhs, gradient), _copy_gradient_for(rhs, gradient)


class GenericSubOperation(Operation):
    """Generic elementwise tensor subtraction operation with autograd support."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_live_tensor(lhs, "lhs")
        rhs = _require_live_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        dtype = _generic_binary_dtype(lhs, rhs)
        values = [lhs[i] - rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(rhs, gradient)

        rhs_values = [-gradient[i] for i in range(gradient.size())]
        return (
            _copy_gradient_for(lhs, gradient),
            _detached_tensor_like(rhs, rhs_values),
        )


class GenericScalarMulOperation(Operation):
    """Generic tensor-by-scalar multiplication operation."""

    def _forward(self, tensor: Any, scalar: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")
        scalar = _require_number(scalar, "scalar")

        self.ctx["scalar"] = scalar
        dtype = _generic_scalar_mul_dtype(tensor, scalar)
        values = [tensor[i] * scalar for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        scalar = self.ctx["scalar"]
        values = [gradient[i] * scalar for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericElementwiseMulOperation(Operation):
    """Generic elementwise tensor multiplication operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_live_tensor(lhs, "lhs")
        rhs = _require_live_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        dtype = _generic_binary_dtype(lhs, rhs)
        values = [lhs[i] * rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(lhs, gradient)

        lhs_values = [gradient[i] * rhs[i] for i in range(gradient.size())]
        rhs_values = [gradient[i] * lhs[i] for i in range(gradient.size())]
        return _detached_tensor_like(lhs, lhs_values), _detached_tensor_like(
            rhs, rhs_values
        )


class GenericDivOperation(Operation):
    """Generic elementwise tensor division operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_live_tensor(lhs, "lhs")
        rhs = _require_live_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        values = [lhs[i] / rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, DType.Floating)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(lhs, gradient)

        lhs_values = [gradient[i] / rhs[i] for i in range(gradient.size())]
        rhs_values = [
            -gradient[i] * lhs[i] / (rhs[i] ** 2) for i in range(gradient.size())
        ]
        return _detached_tensor_like(lhs, lhs_values), _detached_tensor_like(
            rhs, rhs_values
        )


class GenericPowOperation(Operation):
    """Generic elementwise power-to-scalar operation."""

    def _forward(self, tensor: Any, exponent: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")
        exponent = _require_number(exponent, "exponent")

        self.ctx["exponent"] = exponent
        dtype = _generic_pow_dtype(tensor, exponent)
        values = [tensor[i] ** exponent for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        exponent = self.ctx["exponent"]
        values = [
            gradient[i] * exponent * (tensor[i] ** (exponent - 1))
            for i in range(gradient.size())
        ]
        return (_detached_tensor_like(tensor, values),)


class GenericReduceSumOperation(Operation):
    """Generic two-mode sum reduction over the second mode."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_two_mode_tensor(tensor, "tensor")

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        output_layout = _canonical_layout_from_modes(_mode_shape(tensor.layout, 0))

        self.ctx["output_layout"] = output_layout
        values = [sum(tensor[i, j] for j in range(m_size)) for i in range(n_size)]
        return _tensor_with_layout_like(tensor, output_layout, values)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        values = [gradient[i] for _j in range(m_size) for i in range(n_size)]
        return (_detached_tensor_like(tensor, values),)


class GenericMatmulOperation(Operation):
    """Generic two-mode matrix multiplication operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_two_mode_tensor(lhs, "lhs")
        rhs = _require_two_mode_tensor(rhs, "rhs")

        n_size = _mode_logical_size(lhs.layout, 0)
        lhs_k_size = _mode_logical_size(lhs.layout, 1)
        m_size = _mode_logical_size(rhs.layout, 0)
        rhs_k_size = _mode_logical_size(rhs.layout, 1)
        if lhs_k_size != rhs_k_size:
            raise ValueError("Matmul inner dimensions must match")

        output_layout = _canonical_layout_from_modes(
            _mode_shape(lhs.layout, 0), _mode_shape(rhs.layout, 0)
        )

        self.ctx["output_layout"] = output_layout
        dtype = _generic_binary_dtype(lhs, rhs)
        values = [
            sum(lhs[i, k] * rhs[j, k] for k in range(lhs_k_size))
            for j in range(m_size)
            for i in range(n_size)
        ]
        return _tensor_with_layout_like(lhs, output_layout, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        n_size = _mode_logical_size(lhs.layout, 0)
        k_size = _mode_logical_size(lhs.layout, 1)
        m_size = _mode_logical_size(rhs.layout, 0)

        lhs_values = [
            sum(gradient[i, j] * rhs[j, k] for j in range(m_size))
            for k in range(k_size)
            for i in range(n_size)
        ]
        rhs_values = [
            sum(gradient[i, j] * lhs[i, k] for i in range(n_size))
            for k in range(k_size)
            for j in range(m_size)
        ]
        return _detached_tensor_like(lhs, lhs_values), _detached_tensor_like(
            rhs, rhs_values
        )
