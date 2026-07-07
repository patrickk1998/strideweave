"""Generic (Python-backed) autograd operation classes."""

from __future__ import annotations

import math
from collections.abc import Callable
from operator import index as operator_index
from typing import Any

from ._operation_helpers import (
    _LEAKY_RELU_NEGATIVE_SLOPE,
    Operation,
    _as_tensor,
    _canonical_layout_from_modes,
    _copy_gradient_for,
    _copy_gradient_to_layout,
    _detached_tensor_like,
    _gelu_derivative,
    _gelu_value,
    _generic_binary_dtype,
    _generic_pow_dtype,
    _generic_scalar_mul_dtype,
    _layout_from_modes,
    _mode_logical_size,
    _mode_shape,
    _mode_stride,
    _require_layout,
    _require_number,
    _require_same_layout,
    _require_two_mode_tensor,
    _require_unevicted_tensor,
    _sigmoid_value,
    _softplus_value,
    _tensor_with_layout_like,
    _zero_tensor_like,
)
from .data import DataType
from .layout import Layout, Tree


def _unary_elementwise_operation(
    class_name: str,
    docstring: str,
    *,
    compute: Callable[[Any], tuple[Any, Any]],
    gradient_multiplier: Callable[[Any, Any], Any],
    result_dtype: DataType | None = DataType.Floating,
) -> type[Any]:
    """Build a Generic unary elementwise operation class.

    ``compute`` maps one input value to ``(output_value, saved_value)``;
    saved values are stored in the autograd context for the backward pass.
    ``gradient_multiplier`` maps ``(input_value, saved_value)`` to the local
    derivative that scales the incoming gradient.
    """

    def _forward(self: Any, tensor: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")

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
        gradient = _require_unevicted_tensor(gradient, "gradient")
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
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        dtype = _generic_binary_dtype(lhs, rhs)
        values = [lhs[i] + rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        return _copy_gradient_for(lhs, gradient), _copy_gradient_for(rhs, gradient)


class GenericScalarMulOperation(Operation):
    """Generic tensor-by-scalar multiplication operation."""

    def _forward(self, tensor: Any, scalar: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")
        scalar = _require_number(scalar, "scalar")

        self.ctx["scalar"] = scalar
        dtype = _generic_scalar_mul_dtype(tensor, scalar)
        values = [tensor[i] * scalar for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        scalar = self.ctx["scalar"]
        values = [gradient[i] * scalar for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericElementwiseMulOperation(Operation):
    """Generic elementwise tensor multiplication operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        dtype = _generic_binary_dtype(lhs, rhs)
        values = [lhs[i] * rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(lhs, gradient)

        lhs_values = [gradient[i] * rhs[i] for i in range(gradient.size())]
        rhs_values = [gradient[i] * lhs[i] for i in range(gradient.size())]
        return _detached_tensor_like(lhs, lhs_values), _detached_tensor_like(
            rhs, rhs_values
        )


class GenericDivOperation(Operation):
    """Generic elementwise tensor division operation."""

    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        values = [lhs[i] / rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values, DataType.Floating)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
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
        tensor = _require_unevicted_tensor(tensor, "tensor")
        exponent = _require_number(exponent, "exponent")

        self.ctx["exponent"] = exponent
        dtype = _generic_pow_dtype(tensor, exponent)
        values = [tensor[i] ** exponent for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values, dtype)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
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
        gradient = _require_unevicted_tensor(gradient, "gradient")
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
        gradient = _require_unevicted_tensor(gradient, "gradient")
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


def _normalize_view_key(key: Any, rank: int) -> tuple[Any, ...]:
    normalized = key if isinstance(key, tuple) else (key,)
    if len(normalized) != rank:
        raise ValueError("View keys must include exactly one key per top-level mode")
    return normalized


def _normalize_int_key(value: Any, extent: int, name: str) -> int:
    try:
        normalized = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or slice") from exc

    if normalized < 0 or normalized >= extent:
        raise ValueError("View integer key is out of domain")
    return normalized


def _normalize_leaf_slice(key: slice, extent: int) -> tuple[int, int]:
    if key.step not in (None, 1):
        raise ValueError("View slices do not support steps")
    start, stop, _ = key.indices(extent)
    if stop <= start:
        raise ValueError("View slices must be non-empty")
    return start, stop


def _is_whole_slice(key: slice) -> bool:
    return key.start is None and key.stop is None and key.step in (None, 1)


def _view_layout_and_mapping(tensor: Any, key: Any) -> tuple[int, Layout]:
    normalized_key = _normalize_view_key(key, len(tensor.layout))
    offset_delta = 0
    output_shapes = []
    output_strides = []

    for mode_index, mode_key in enumerate(normalized_key):
        mode_layout = tensor.layout[mode_index]
        mode_shape = _mode_shape(tensor.layout, mode_index)
        mode_stride = _mode_stride(tensor.layout, mode_index)

        if isinstance(mode_key, slice):
            if mode_layout.is_leaf:
                extent = int(mode_layout.shape)
                stride = int(mode_layout.stride)
                start, stop = _normalize_leaf_slice(mode_key, extent)
                offset_delta += start * stride
                output_shapes.append(stop - start)
                output_strides.append(stride)
                continue

            if not _is_whole_slice(mode_key):
                raise ValueError("Only whole slices are supported for non-leaf modes")
            output_shapes.append(mode_shape)
            output_strides.append(mode_stride)
            continue

        if mode_layout.is_leaf:
            extent = int(mode_layout.shape)
            stride = int(mode_layout.stride)
            index = _normalize_int_key(mode_key, extent, "View key")
            offset_delta += index * stride
            continue

        extent = mode_layout.shape.logical_size
        index = _normalize_int_key(mode_key, extent, "View key")
        offset_delta += mode_layout.index(index)

    return offset_delta, _layout_from_modes(output_shapes, output_strides)


class GenericViewOperation(Operation):
    """Generic tensor view operation sharing the input backing data."""

    def _forward(self, tensor: Any, key: Any) -> Any:
        from .tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        offset_delta, output_layout = _view_layout_and_mapping(tensor, key)

        self.ctx["mapping_layout"] = output_layout
        self.ctx["mapping_offset"] = offset_delta
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.data, tensor.offset + offset_delta, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        scatter_onto = _zero_tensor_like(tensor)
        scatter_onto.data.scatter(
            gradient,
            scatter_onto,
            self.ctx["mapping_layout"],
            self.ctx["mapping_offset"],
        )
        return (scatter_onto,)


class RearrangeOperation(Operation):
    """Autograd operation that rearranges a tensor layout without copying data."""

    def _forward(self, tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
        from .tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        if not isinstance(output, Tree):
            raise TypeError("output must be a Tree")
        if selection is not None and not isinstance(selection, Tree):
            raise TypeError("selection must be a Tree or None")

        effective_selection = selection
        if effective_selection is None:
            effective_selection = Layout._default_selection_tree(tensor.layout)

        output_layout = Layout.rearrange(tensor.layout, output, effective_selection)

        self.ctx["output"] = output
        self.ctx["selection"] = effective_selection
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.data, tensor.offset, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        reverse_output, reverse_selection = Layout.reverse_rearrange(
            self.ctx["output"], self.ctx["selection"]
        )
        inverse_layout = Layout.rearrange(
            gradient.layout, reverse_output, reverse_selection
        )

        from .tensor import Tensor

        inverse_gradient = Tensor(gradient.data, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)


class PermuteOperation(Operation):
    """Autograd operation that permutes top-level layout modes."""

    def _forward(self, tensor: Any, *order: Any) -> Any:
        from .tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        normalized_order = Layout._normalize_permute_order(order, len(tensor.layout))
        output_layout = Layout.permute(tensor.layout, normalized_order)

        self.ctx["order"] = normalized_order
        self.ctx["output_layout"] = output_layout
        return Tensor(tensor.data, tensor.offset, output_layout)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _as_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        order = self.ctx["order"]
        inverse_order = [0] * len(order)
        for output_mode, input_mode in enumerate(order):
            inverse_order[input_mode] = output_mode
        inverse_layout = Layout.permute(gradient.layout, inverse_order)

        from .tensor import Tensor

        inverse_gradient = Tensor(gradient.data, gradient.offset, inverse_layout)
        return (_copy_gradient_to_layout(tensor, inverse_gradient),)
