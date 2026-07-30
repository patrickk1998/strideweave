"""Generic (Python-backed) autograd operation classes."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _align_binary_operands,
    _canonical_layout_from_modes,
    _detached_tensor_like,
    _mode_logical_size,
    _mode_shape,
    _require_layout,
    _require_live_tensor,
    _require_same_shape,
    _require_two_mode_tensor,
    _tensor_with_layout_like,
)
from .execution import (
    binary_arithmetic,
    executing,
    gradient_arithmetic,
    scalar_arithmetic,
    unary_arithmetic,
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
from .numerics import safe_exp


def _unary_elementwise_operation(
    class_name: str,
    docstring: str,
    *,
    operation: str,
    compute: Callable[[Any], tuple[Any, Any]],
    gradient_multiplier: Callable[[Any, Any], Any],
    result_dtype: DType | None = DType.Floating,
) -> type[Any]:
    """Build a Generic unary elementwise operation class.

    ``compute`` maps one input value to ``(output_value, saved_value)``;
    saved values are stored in the autograd context for the backward pass.
    ``gradient_multiplier`` maps ``(input_value, saved_value)`` to the local
    derivative that scales the incoming gradient. Both run through the
    arithmetic resolved for the operand, so a concrete operand computes in
    binary32 or exact ``Int32`` while legacy storage keeps Python arithmetic.
    """

    def _forward(self: Any, tensor: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")
        arithmetic = unary_arithmetic(operation, tensor, result_dtype)

        values = []
        saved_values = []
        with executing(arithmetic):
            for i in range(tensor.size()):
                value, saved = compute(arithmetic.convert(tensor[i]))
                values.append(arithmetic.store(value))
                saved_values.append(saved)
        self.ctx["saved_values"] = saved_values
        return _detached_tensor_like(tensor, values, arithmetic.result_dtype)

    def backward(self: Any, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)

        saved_values = self.ctx["saved_values"]
        arithmetic = gradient_arithmetic(tensor)
        with executing(arithmetic):
            values = [
                arithmetic.store(
                    arithmetic.convert(gradient[i])
                    * gradient_multiplier(
                        arithmetic.convert(tensor[i]), saved_values[i]
                    )
                )
                for i in range(gradient.size())
            ]
        return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)

    return type(
        class_name,
        (Operation,),
        {"__doc__": docstring, "_forward": _forward, "backward": backward},
    )


def _binary_elementwise_result(
    owner: Any,
    lhs: Any,
    rhs: Any,
    *,
    operation: str,
    compute: Callable[[Any, Any], Any],
    legacy_dtype: DType | None = None,
) -> Any:
    """Validate Generic binary operands and construct their detached result.

    ``legacy_dtype`` overrides the dtype Generic reports on its legacy path;
    by default the operands' historical promotion decides it.
    """
    lhs = _require_live_tensor(lhs, "lhs")
    rhs = _require_live_tensor(rhs, "rhs")
    lhs, rhs, result_layout = _align_binary_operands(lhs, rhs)
    if owner.inputs():
        owner.store_inputs(lhs, rhs)

    if legacy_dtype is None:
        legacy_dtype = _generic_binary_dtype(lhs, rhs)
    arithmetic = binary_arithmetic(operation, lhs, rhs, legacy_dtype)
    with executing(arithmetic):
        values = [
            arithmetic.store(
                compute(arithmetic.convert(lhs[i]), arithmetic.convert(rhs[i]))
            )
            for i in range(lhs.size())
        ]
    return _tensor_with_layout_like(lhs, result_layout, values, arithmetic.result_dtype)


def _gradient_values(
    arithmetic: Any, size: int, compute: Callable[[int], Any]
) -> list[Any]:
    """Build one operand's gradient values through ``arithmetic``.

    ``compute`` receives a logical index and returns the gradient term, already
    expressed in the arithmetic's compute representation.
    """
    return [arithmetic.store(compute(i)) for i in range(size)]


def _gradient_tensor(
    target: Any, arithmetic: Any, compute: Callable[[int], Any]
) -> Any:
    """Build one operand's gradient tensor through ``arithmetic``."""
    with executing(arithmetic):
        values = _gradient_values(arithmetic, target.size(), compute)
    return _detached_tensor_like(target, values, arithmetic.result_dtype)


def _copy_gradient(target: Any, gradient: Any) -> Any:
    """Copy an incoming gradient into fresh storage shaped like ``target``.

    A concrete operand's gradient is ``Float32`` even when the operand itself is
    ``Int32``, because gradients are floating; the framework only consumes the
    gradients of differentiable operands.
    """
    _require_same_shape(target, gradient)
    arithmetic = gradient_arithmetic(target)
    return _gradient_tensor(
        target, arithmetic, lambda i: arithmetic.convert(gradient[i])
    )


def _exp_compute(value: Any) -> tuple[Any, Any]:
    output = safe_exp(value)
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
    operation="exp",
    compute=_exp_compute,
    gradient_multiplier=lambda _value, output: output,
)

GenericReLUOperation = _unary_elementwise_operation(
    "GenericReLUOperation",
    "Generic elementwise rectified linear unit operation.",
    operation="relu",
    compute=lambda value: (max(0, value), None),
    gradient_multiplier=lambda value, _saved: 1 if value > 0 else 0,
    result_dtype=None,
)

GenericSigmoidOperation = _unary_elementwise_operation(
    "GenericSigmoidOperation",
    "Generic elementwise logistic sigmoid operation.",
    operation="sigmoid",
    compute=_sigmoid_compute,
    gradient_multiplier=lambda _value, output: output * (1.0 - output),
)

GenericTanhOperation = _unary_elementwise_operation(
    "GenericTanhOperation",
    "Generic elementwise hyperbolic tangent operation.",
    operation="tanh",
    compute=_tanh_compute,
    gradient_multiplier=lambda _value, output: 1.0 - output**2,
)

GenericGELUOperation = _unary_elementwise_operation(
    "GenericGELUOperation",
    "Generic elementwise Gaussian error linear unit operation.",
    operation="gelu",
    compute=lambda value: (_gelu_value(float(value)), None),
    gradient_multiplier=lambda value, _saved: _gelu_derivative(float(value)),
)

GenericSiLUOperation = _unary_elementwise_operation(
    "GenericSiLUOperation",
    "Generic elementwise sigmoid linear unit operation.",
    operation="silu",
    compute=_silu_compute,
    gradient_multiplier=lambda value, sigmoid: (
        sigmoid + value * sigmoid * (1.0 - sigmoid)
    ),
)

GenericSoftplusOperation = _unary_elementwise_operation(
    "GenericSoftplusOperation",
    "Generic elementwise softplus operation.",
    operation="softplus",
    compute=lambda value: (_softplus_value(value), None),
    gradient_multiplier=lambda value, _saved: _sigmoid_value(value),
)

GenericELUOperation = _unary_elementwise_operation(
    "GenericELUOperation",
    "Generic elementwise exponential linear unit operation.",
    operation="elu",
    compute=lambda value: (
        float(value) if value > 0 else math.expm1(value),
        None,
    ),
    gradient_multiplier=lambda value, _saved: 1 if value > 0 else math.exp(value),
)

GenericLeakyReLUOperation = _unary_elementwise_operation(
    "GenericLeakyReLUOperation",
    "Generic elementwise leaky rectified linear unit operation.",
    operation="leaky_relu",
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
        return _binary_elementwise_result(
            self, lhs, rhs, operation="add", compute=lambda x, y: x + y
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        return _copy_gradient(lhs, gradient), _copy_gradient(rhs, gradient)


class GenericSubOperation(Operation):
    """Generic elementwise tensor subtraction operation with autograd support."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_elementwise_result(
            self, lhs, rhs, operation="sub", compute=lambda x, y: x - y
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(rhs, gradient)

        rhs_arithmetic = gradient_arithmetic(rhs)
        return (
            _copy_gradient(lhs, gradient),
            _gradient_tensor(
                rhs,
                rhs_arithmetic,
                lambda i: -rhs_arithmetic.convert(gradient[i]),
            ),
        )


class GenericScalarMulOperation(Operation):
    """Generic tensor-by-scalar multiplication operation."""

    def _forward(self, tensor: Any, scalar: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")

        # The scalar is validated inside `scalar_arithmetic`, by the policy for
        # concrete storage and by the legacy check otherwise, so both backends
        # reject the same scalars with the same message.
        arithmetic = scalar_arithmetic(
            "mul", tensor, scalar, _generic_scalar_mul_dtype(tensor, scalar), "scalar"
        )
        # The scalar is materialized once, before the loop, and the materialized
        # value is what backward reuses. Saving the original object instead
        # would convert it a second time, so a scalar whose numeric conversion
        # is not stable could scale the gradient by a different value than the
        # forward pass used. On the legacy path conversion is the identity, so
        # this stores the supplied object exactly as before.
        materialized = arithmetic.convert(scalar)
        self.ctx["scalar"] = materialized
        with executing(arithmetic):
            values = [
                arithmetic.store(arithmetic.convert(tensor[i]) * materialized)
                for i in range(tensor.size())
            ]
        return _detached_tensor_like(tensor, values, arithmetic.result_dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)

        arithmetic = gradient_arithmetic(tensor)
        scalar = self.ctx["scalar"]
        return (
            _gradient_tensor(
                tensor,
                arithmetic,
                lambda i: arithmetic.convert(gradient[i]) * scalar,
            ),
        )


class GenericElementwiseMulOperation(Operation):
    """Generic elementwise tensor multiplication operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_elementwise_result(
            self,
            lhs,
            rhs,
            operation="elementwise_mul",
            compute=lambda x, y: x * y,
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(lhs, gradient)
        _require_same_shape(rhs, gradient)

        lhs_arithmetic = gradient_arithmetic(lhs)
        rhs_arithmetic = gradient_arithmetic(rhs)
        return (
            _gradient_tensor(
                lhs,
                lhs_arithmetic,
                lambda i: (
                    lhs_arithmetic.convert(gradient[i]) * lhs_arithmetic.convert(rhs[i])
                ),
            ),
            _gradient_tensor(
                rhs,
                rhs_arithmetic,
                lambda i: (
                    rhs_arithmetic.convert(gradient[i]) * rhs_arithmetic.convert(lhs[i])
                ),
            ),
        )


class GenericDivOperation(Operation):
    """Generic elementwise tensor division operation.

    Division is always floating. On concrete storage it follows IEEE-754, so a
    zero divisor yields an infinity or NaN rather than raising, in forward and
    in backward alike.
    """

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        # Division is floating on both paths, so legacy storage reports
        # Floating rather than the operands' promoted category.
        return _binary_elementwise_result(
            self,
            lhs,
            rhs,
            operation="div",
            compute=lambda x, y: x / y,
            legacy_dtype=DType.Floating,
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(lhs, gradient)
        _require_same_shape(rhs, gradient)

        lhs_arithmetic = gradient_arithmetic(lhs)
        rhs_arithmetic = gradient_arithmetic(rhs)
        return (
            _gradient_tensor(
                lhs,
                lhs_arithmetic,
                lambda i: (
                    lhs_arithmetic.convert(gradient[i]) / lhs_arithmetic.convert(rhs[i])
                ),
            ),
            _gradient_tensor(
                rhs,
                rhs_arithmetic,
                lambda i: (
                    -rhs_arithmetic.convert(gradient[i])
                    * rhs_arithmetic.convert(lhs[i])
                    / (rhs_arithmetic.convert(rhs[i]) ** rhs_arithmetic.convert(2))
                ),
            ),
        )


class GenericPowOperation(Operation):
    """Generic elementwise power-to-scalar operation."""

    def _forward(self, tensor: Any, exponent: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")

        arithmetic = scalar_arithmetic(
            "pow", tensor, exponent, _generic_pow_dtype(tensor, exponent), "exponent"
        )
        # Materialized once and reused by backward, for the reason given in
        # GenericScalarMulOperation.
        materialized = arithmetic.convert(exponent)
        self.ctx["exponent"] = materialized
        with executing(arithmetic):
            values = [
                arithmetic.store(arithmetic.convert(tensor[i]) ** materialized)
                for i in range(tensor.size())
            ]
        return _detached_tensor_like(tensor, values, arithmetic.result_dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)

        arithmetic = gradient_arithmetic(tensor)
        exponent = self.ctx["exponent"]
        one = arithmetic.convert(1)
        return (
            _gradient_tensor(
                tensor,
                arithmetic,
                lambda i: (
                    arithmetic.convert(gradient[i])
                    * exponent
                    * (arithmetic.convert(tensor[i]) ** (exponent - one))
                ),
            ),
        )


class GenericReduceSumOperation(Operation):
    """Generic two-mode sum reduction over the second mode."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_two_mode_tensor(tensor, "tensor")

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        output_layout = _canonical_layout_from_modes(_mode_shape(tensor.layout, 0))

        self.ctx["output_layout"] = output_layout
        arithmetic = unary_arithmetic("reduce", tensor, None)
        with executing(arithmetic):
            values = [
                arithmetic.store(
                    arithmetic.total(
                        arithmetic.convert(tensor[i, j]) for j in range(m_size)
                    )
                )
                for i in range(n_size)
            ]
        return _tensor_with_layout_like(
            tensor, output_layout, values, arithmetic.result_dtype
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        arithmetic = gradient_arithmetic(tensor)
        # Each reduced element receives its row's gradient. The repeated values
        # are converted and stored in the same pass that generates them, so only
        # one list of the input's size is ever live.
        with executing(arithmetic):
            stored = [
                arithmetic.store(arithmetic.convert(gradient[i]))
                for _j in range(m_size)
                for i in range(n_size)
            ]
        return (_detached_tensor_like(tensor, stored, arithmetic.result_dtype),)


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
        arithmetic = binary_arithmetic(
            "matmul", lhs, rhs, _generic_binary_dtype(lhs, rhs)
        )
        with executing(arithmetic):
            values = [
                arithmetic.store(
                    arithmetic.total(
                        arithmetic.convert(lhs[i, k]) * arithmetic.convert(rhs[j, k])
                        for k in range(lhs_k_size)
                    )
                )
                for j in range(m_size)
                for i in range(n_size)
            ]
        return _tensor_with_layout_like(
            lhs, output_layout, values, arithmetic.result_dtype
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        n_size = _mode_logical_size(lhs.layout, 0)
        k_size = _mode_logical_size(lhs.layout, 1)
        m_size = _mode_logical_size(rhs.layout, 0)

        lhs_arithmetic = gradient_arithmetic(lhs)
        rhs_arithmetic = gradient_arithmetic(rhs)
        with executing(lhs_arithmetic):
            lhs_values = [
                lhs_arithmetic.store(
                    lhs_arithmetic.total(
                        lhs_arithmetic.convert(gradient[i, j])
                        * lhs_arithmetic.convert(rhs[j, k])
                        for j in range(m_size)
                    )
                )
                for k in range(k_size)
                for i in range(n_size)
            ]
        with executing(rhs_arithmetic):
            rhs_values = [
                rhs_arithmetic.store(
                    rhs_arithmetic.total(
                        rhs_arithmetic.convert(gradient[i, j])
                        * rhs_arithmetic.convert(lhs[i, k])
                        for i in range(n_size)
                    )
                )
                for k in range(k_size)
                for j in range(m_size)
            ]
        return (
            _detached_tensor_like(lhs, lhs_values, lhs_arithmetic.result_dtype),
            _detached_tensor_like(rhs, rhs_values, rhs_arithmetic.result_dtype),
        )
