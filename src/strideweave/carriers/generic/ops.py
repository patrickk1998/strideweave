"""Generic (Python-backed) autograd operation classes."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _align_binary_operands,
    _canonical_layout_for_shape,
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
    arithmetic_for_plan,
    binary_arithmetic,
    executing,
    gradient_arithmetic,
    resolve_operation_plan,
    scalar_arithmetic,
    unary_arithmetic,
)
from .helpers import (
    _elu_derivative,
    _elu_value,
    _gelu_derivative,
    _gelu_value,
    _generic_binary_dtype,
    _generic_pow_dtype,
    _generic_scalar_mul_dtype,
    _leaky_relu_derivative,
    _leaky_relu_value,
    _sigmoid_value,
    _softplus_value,
)
from .numerics import (
    safe_abs,
    safe_ceil,
    safe_cos,
    safe_divide,
    safe_erf,
    safe_exp,
    safe_exp2,
    safe_floor,
    safe_fmod,
    safe_int_power_checked,
    safe_log,
    safe_log2,
    safe_maximum,
    safe_minimum,
    safe_power,
    safe_recip,
    safe_round,
    safe_sign,
    safe_sin,
    safe_sqrt,
    safe_tanh,
    safe_trunc,
)


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
        # Unary results always use fresh canonical injective storage.  Keeping
        # the input stride would make a gapped-but-injective view leak its
        # storage holes into the result, contrary to the v0 elementwise
        # contract.
        result_layout = _canonical_layout_for_shape(tensor.layout.shape)
        return _tensor_with_layout_like(
            tensor, result_layout, values, arithmetic.result_dtype
        )

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


def _neg_compute(value: Any) -> tuple[Any, Any]:
    output = -value
    return output, None


def _abs_compute(value: Any) -> tuple[Any, Any]:
    output = safe_abs(value)
    return output, None


def _sign_compute(value: Any) -> tuple[Any, Any]:
    output = safe_sign(value)
    return output, None


def _recip_compute(value: Any) -> tuple[Any, Any]:
    output = safe_recip(value)
    return output, output


def _sqrt_compute(value: Any) -> tuple[Any, Any]:
    output = safe_sqrt(value)
    return output, output


def _rsqrt_compute(value: Any) -> tuple[Any, Any]:
    output = safe_recip(safe_sqrt(value))
    return output, output


def _exp2_compute(value: Any) -> tuple[Any, Any]:
    output = safe_exp2(value)
    return output, output


def _log_compute(value: Any) -> tuple[Any, Any]:
    output = safe_log(value)
    return output, None


def _log2_compute(value: Any) -> tuple[Any, Any]:
    output = safe_log2(value)
    return output, None


def _sin_compute(value: Any) -> tuple[Any, Any]:
    return safe_sin(value), None


def _cos_compute(value: Any) -> tuple[Any, Any]:
    return safe_cos(value), None


def _erf_compute(value: Any) -> tuple[Any, Any]:
    return safe_erf(value), None


def _floor_compute(value: Any) -> tuple[Any, Any]:
    return safe_floor(value), None


def _ceil_compute(value: Any) -> tuple[Any, Any]:
    return safe_ceil(value), None


def _round_compute(value: Any) -> tuple[Any, Any]:
    return safe_round(value), None


def _sigmoid_compute(value: Any) -> tuple[Any, Any]:
    output = _sigmoid_value(value)
    return output, output


def _tanh_compute(value: Any) -> tuple[Any, Any]:
    output = safe_tanh(value)
    return output, output


def _silu_compute(value: Any) -> tuple[Any, Any]:
    sigmoid = _sigmoid_value(value)
    return value * sigmoid, sigmoid


def _power_value(arithmetic: Any, base: Any, exponent: Any) -> Any:
    """Evaluate power through the arithmetic's checked integer boundary."""
    if arithmetic.result_dtype is DType.Int32:
        return safe_int_power_checked(int(base), int(exponent))
    return safe_power(base, exponent)


_LOG2 = math.log(2.0)
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


GenericNegOperation = _unary_elementwise_operation(
    "GenericNegOperation",
    "Generic elementwise negation operation.",
    operation="neg",
    compute=_neg_compute,
    gradient_multiplier=lambda _value, _saved: -1,
    result_dtype=None,
)

GenericAbsOperation = _unary_elementwise_operation(
    "GenericAbsOperation",
    "Generic elementwise absolute-value operation.",
    operation="abs",
    compute=_abs_compute,
    gradient_multiplier=lambda value, _saved: 0 if value == 0 else safe_sign(value),
    result_dtype=None,
)

GenericSignOperation = _unary_elementwise_operation(
    "GenericSignOperation",
    "Generic elementwise sign operation.",
    operation="sign",
    compute=_sign_compute,
    gradient_multiplier=lambda _value, _saved: 0,
    result_dtype=None,
)

GenericRecipOperation = _unary_elementwise_operation(
    "GenericRecipOperation",
    "Generic elementwise reciprocal operation.",
    operation="recip",
    compute=_recip_compute,
    gradient_multiplier=lambda value, _saved: -safe_divide(1, value * value),
)

GenericSqrtOperation = _unary_elementwise_operation(
    "GenericSqrtOperation",
    "Generic elementwise square-root operation.",
    operation="sqrt",
    compute=_sqrt_compute,
    gradient_multiplier=lambda _value, output: safe_divide(1, 2 * output),
)

GenericRsqrtOperation = _unary_elementwise_operation(
    "GenericRsqrtOperation",
    "Generic elementwise reciprocal-square-root operation.",
    operation="rsqrt",
    compute=_rsqrt_compute,
    gradient_multiplier=lambda _value, output: -(output * output * output) / 2,
)

GenericExp2Operation = _unary_elementwise_operation(
    "GenericExp2Operation",
    "Generic elementwise base-two exponential operation.",
    operation="exp2",
    compute=_exp2_compute,
    gradient_multiplier=lambda _value, output: _LOG2 * output,
)

GenericLogOperation = _unary_elementwise_operation(
    "GenericLogOperation",
    "Generic elementwise natural-logarithm operation.",
    operation="log",
    compute=_log_compute,
    gradient_multiplier=lambda value, _saved: safe_recip(value),
)

GenericLog2Operation = _unary_elementwise_operation(
    "GenericLog2Operation",
    "Generic elementwise base-two logarithm operation.",
    operation="log2",
    compute=_log2_compute,
    gradient_multiplier=lambda value, _saved: safe_divide(1, value * _LOG2),
)

GenericSinOperation = _unary_elementwise_operation(
    "GenericSinOperation",
    "Generic elementwise sine operation.",
    operation="sin",
    compute=_sin_compute,
    gradient_multiplier=lambda value, _saved: safe_cos(value),
)

GenericCosOperation = _unary_elementwise_operation(
    "GenericCosOperation",
    "Generic elementwise cosine operation.",
    operation="cos",
    compute=_cos_compute,
    gradient_multiplier=lambda value, _saved: -safe_sin(value),
)

GenericErfOperation = _unary_elementwise_operation(
    "GenericErfOperation",
    "Generic elementwise error-function operation.",
    operation="erf",
    compute=_erf_compute,
    gradient_multiplier=lambda value, _saved: (
        _TWO_OVER_SQRT_PI * safe_exp(-(value * value))
    ),
)

GenericFloorOperation = _unary_elementwise_operation(
    "GenericFloorOperation",
    "Generic elementwise floor operation.",
    operation="floor",
    compute=_floor_compute,
    gradient_multiplier=lambda _value, _saved: 0,
    result_dtype=None,
)

GenericCeilOperation = _unary_elementwise_operation(
    "GenericCeilOperation",
    "Generic elementwise ceil operation.",
    operation="ceil",
    compute=_ceil_compute,
    gradient_multiplier=lambda _value, _saved: 0,
    result_dtype=None,
)

GenericRoundOperation = _unary_elementwise_operation(
    "GenericRoundOperation",
    "Generic elementwise ties-to-even round operation.",
    operation="round",
    compute=_round_compute,
    gradient_multiplier=lambda _value, _saved: 0,
    result_dtype=None,
)


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
    compute=lambda value: (_gelu_value(value), None),
    gradient_multiplier=lambda value, _saved: _gelu_derivative(value),
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
    compute=lambda value: (_elu_value(value), None),
    gradient_multiplier=lambda value, _saved: _elu_derivative(value),
)

GenericLeakyReLUOperation = _unary_elementwise_operation(
    "GenericLeakyReLUOperation",
    "Generic elementwise leaky rectified linear unit operation.",
    operation="leaky_relu",
    compute=lambda value: (_leaky_relu_value(value), None),
    gradient_multiplier=lambda value, _saved: _leaky_relu_derivative(value),
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

        # The v0 binary surface also permits ``mul(tensor, other_tensor)``.
        # Keep this overload in the carrier operation so callers that dispatch
        # ``mul`` directly still get the same structural alignment and
        # canonical result as the historical ``elementwise_mul`` entry point.
        from ...tensor import Tensor

        if isinstance(scalar, Tensor):
            self.ctx["binary"] = True
            return _binary_elementwise_result(
                self,
                tensor,
                scalar,
                operation="mul",
                compute=lambda x, y: x * y,
            )

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
        return _tensor_with_layout_like(
            tensor,
            _canonical_layout_for_shape(tensor.layout.shape),
            values,
            arithmetic.result_dtype,
        )

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        if self.ctx.get("binary", False):
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
                        lhs_arithmetic.convert(gradient[i])
                        * lhs_arithmetic.convert(rhs[i])
                    ),
                ),
                _gradient_tensor(
                    rhs,
                    rhs_arithmetic,
                    lambda i: (
                        rhs_arithmetic.convert(gradient[i])
                        * rhs_arithmetic.convert(lhs[i])
                    ),
                ),
            )

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


def _extrema_gradient_multiplier(
    lhs: Any, rhs: Any, *, maximum: bool, lhs_side: bool
) -> Any:
    """Return one operand's NumPy-style max/min gradient multiplier."""
    lhs_value = float(lhs)
    rhs_value = float(rhs)
    if math.isnan(lhs_value) or math.isnan(rhs_value):
        return math.nan
    if lhs_value == rhs_value and math.isfinite(lhs_value):
        return 0.5
    if maximum:
        selected_lhs = lhs_value > rhs_value
    else:
        selected_lhs = lhs_value < rhs_value
    return 1 if selected_lhs is lhs_side else 0


def _extrema_operation(class_name: str, operation: str, *, maximum: bool) -> type[Any]:
    """Build a Generic maximum/minimum operation with split-tie VJPs."""

    compute = safe_maximum if maximum else safe_minimum

    def _forward(self: Any, lhs: Any, rhs: Any) -> Any:
        return _binary_elementwise_result(
            self, lhs, rhs, operation=operation, compute=compute
        )

    def backward(self: Any, gradient: Any) -> tuple[Any, Any]:
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
                    lhs_arithmetic.convert(gradient[i])
                    * _extrema_gradient_multiplier(
                        lhs[i], rhs[i], maximum=maximum, lhs_side=True
                    )
                ),
            ),
            _gradient_tensor(
                rhs,
                rhs_arithmetic,
                lambda i: (
                    rhs_arithmetic.convert(gradient[i])
                    * _extrema_gradient_multiplier(
                        lhs[i], rhs[i], maximum=maximum, lhs_side=False
                    )
                ),
            ),
        )

    return type(
        class_name,
        (Operation,),
        {
            "__doc__": f"Generic elementwise {operation} operation.",
            "_forward": _forward,
            "backward": backward,
        },
    )


GenericMaximumOperation = _extrema_operation(
    "GenericMaximumOperation", "maximum", maximum=True
)
GenericMinimumOperation = _extrema_operation(
    "GenericMinimumOperation", "minimum", maximum=False
)


class GenericRemOperation(Operation):
    """Generic truncating-remainder operation with autograd support."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        return _binary_elementwise_result(
            self, lhs, rhs, operation="rem", compute=safe_fmod
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(lhs, gradient)
        _require_same_shape(rhs, gradient)
        lhs_arithmetic = gradient_arithmetic(lhs)
        rhs_arithmetic = gradient_arithmetic(rhs)

        def rhs_multiplier(i: int) -> Any:
            x = rhs_arithmetic.convert(lhs[i])
            divisor = rhs_arithmetic.convert(rhs[i])
            quotient = safe_divide(x, divisor)
            return -safe_trunc(quotient)

        return (
            _gradient_tensor(
                lhs,
                lhs_arithmetic,
                lambda i: lhs_arithmetic.convert(gradient[i]),
            ),
            _gradient_tensor(
                rhs,
                rhs_arithmetic,
                lambda i: rhs_arithmetic.convert(gradient[i]) * rhs_multiplier(i),
            ),
        )


class GenericPowOperation(Operation):
    """Generic elementwise power operation with scalar overloads."""

    def _forward(self, tensor: Any, exponent: Any) -> Any:
        from ...tensor import Tensor

        # Tensor-tensor power is a binary pointwise operation.  The aligned
        # inputs are retained by ``_binary_elementwise_result`` for backward,
        # and the result uses its canonical injective layout.
        if isinstance(tensor, Tensor) and isinstance(exponent, Tensor):
            self.ctx["binary"] = True
            return _binary_elementwise_result(
                self,
                tensor,
                exponent,
                operation="pow",
                compute=safe_power,
            )

        # A weak scalar base is the second v0 overload.  Only the tensor
        # exponent receives a gradient; the scalar itself is not an input
        # tensor.  This path is intentionally kept separate from the
        # tensor-base scalar exponent path below because their VJPs differ.
        if isinstance(exponent, Tensor) and not isinstance(tensor, Tensor):
            exponent = _require_live_tensor(exponent, "exponent")
            exponent_dtype = exponent.carrier.dtype()
            if exponent_dtype in (DType.Float32, DType.Int32, DType.Bool):
                # ``scalar_arithmetic`` resolves only the tensor-weak-scalar
                # overload.  Reverse power has the weak-scalar-tensor role
                # order, so ask the shared policy for that overload directly.
                plan = resolve_operation_plan("pow", tensor, exponent_dtype)
                arithmetic = arithmetic_for_plan(plan, type(exponent.carrier))
            else:
                # Legacy opaque storage has no policy plan and retains Generic's
                # historical number validation and Python arithmetic.
                arithmetic = scalar_arithmetic(
                    "pow",
                    exponent,
                    tensor,
                    _generic_pow_dtype(exponent, tensor),
                    "base",
                )
            materialized = arithmetic.convert(tensor)
            self.ctx["scalar_base"] = materialized
            self.ctx["output_values"] = []
            with executing(arithmetic):
                values = [
                    arithmetic.store(
                        _power_value(
                            arithmetic,
                            materialized,
                            arithmetic.convert(exponent[i]),
                        )
                    )
                    for i in range(exponent.size())
                ]
            self.ctx["output_values"] = values
            return _tensor_with_layout_like(
                exponent,
                _canonical_layout_for_shape(exponent.layout.shape),
                values,
                arithmetic.result_dtype,
            )

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
                arithmetic.store(
                    _power_value(
                        arithmetic, arithmetic.convert(tensor[i]), materialized
                    )
                )
                for i in range(tensor.size())
            ]
        return _tensor_with_layout_like(
            tensor,
            _canonical_layout_for_shape(tensor.layout.shape),
            values,
            arithmetic.result_dtype,
        )

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        if self.ctx.get("binary", False):
            base, exponent = self.inputs()
            gradient = _require_live_tensor(gradient, "gradient")
            _require_same_shape(base, gradient)
            _require_same_shape(exponent, gradient)
            base_arithmetic = gradient_arithmetic(base)
            exponent_arithmetic = gradient_arithmetic(exponent)

            def base_term(i: int) -> Any:
                x = base_arithmetic.convert(base[i])
                p = base_arithmetic.convert(exponent[i])
                return base_arithmetic.convert(gradient[i]) * p * safe_power(x, p - 1)

            def exponent_term(i: int) -> Any:
                x = exponent_arithmetic.convert(base[i])
                p = exponent_arithmetic.convert(exponent[i])
                y = safe_power(x, p)
                return exponent_arithmetic.convert(gradient[i]) * y * safe_log(x)

            return (
                _gradient_tensor(base, base_arithmetic, base_term),
                _gradient_tensor(exponent, exponent_arithmetic, exponent_term),
            )

        # Weak scalar base: only the tensor exponent receives a gradient.
        if "scalar_base" in self.ctx:
            (exponent,) = self.inputs()
            gradient = _require_live_tensor(gradient, "gradient")
            _require_same_shape(exponent, gradient)
            arithmetic = gradient_arithmetic(exponent)
            base = arithmetic.convert(self.ctx["scalar_base"])

            return (
                _gradient_tensor(
                    exponent,
                    arithmetic,
                    lambda i: (
                        arithmetic.convert(gradient[i])
                        * safe_power(base, arithmetic.convert(exponent[i]))
                        * safe_log(base)
                    ),
                ),
            )

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
                    * safe_power(arithmetic.convert(tensor[i]), exponent - one)
                ),
            ),
        )


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
            "matmul",
            lhs,
            rhs,
            _generic_binary_dtype(lhs, rhs),
            options=self._execution_options,
        )
        if arithmetic.plan is not None:
            self.ctx["accumulator_dtype"] = arithmetic.plan.accumulator_dtype
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

        accumulator_dtype = self.ctx.get("accumulator_dtype") or DType.Float32
        lhs_arithmetic = gradient_arithmetic(lhs, accumulator_dtype)
        rhs_arithmetic = gradient_arithmetic(rhs, accumulator_dtype)
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
