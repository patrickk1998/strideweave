from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib import import_module
from numbers import Integral, Number
from operator import index as operator_index
from typing import Any, cast, overload

from .data import DataType
from .layout import Layout, Shape, Stride, Tree

_get_index = cast(Any, import_module("neotorch._index")).get_index

_operation = import_module("neotorch._operation")
Operation = cast(type[Any], _operation.Operation)
_is_grad_enabled = cast(Callable[[], bool], _operation.is_grad_enabled)
_set_grad_enabled = cast(Callable[[bool], None], _operation.set_grad_enabled)
_REDUCE_DESCRIPTION_MISSING = object()


def is_grad_enabled() -> bool:
    """Return whether autograd graph construction is enabled.

    The value is thread-local and controls whether operation calls attach
    autograd context to their result tensors.

    Args:
        None.

    Returns:
        ``True`` when operations build an autograd graph in the current thread.

    Examples:
        >>> import neotorch
        >>> neotorch.is_grad_enabled()
        True
    """

    return _is_grad_enabled()


def set_grad_enabled(enabled: bool) -> None:
    """Set the current thread's autograd graph construction state.

    Args:
        enabled: ``True`` to build autograd graphs for future operations in the
            current thread, or ``False`` to skip graph construction.

    Returns:
        ``None``.

    Examples:
        >>> import neotorch
        >>> previous = neotorch.is_grad_enabled()
        >>> neotorch.set_grad_enabled(False)
        >>> neotorch.set_grad_enabled(previous)
    """

    _set_grad_enabled(enabled)


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable autograd graph construction.

    The previous thread-local grad-enabled state is restored when the context
    exits, including when the block raises.

    Args:
        None.

    Returns:
        Context manager that yields ``None`` while gradients are disabled.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> with neotorch.no_grad():
        ...     y = neotorch.mul(x, 2)
        >>> y.autograd_ctx is None
        True
    """

    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)


def _as_tensor(value: Any, name: str) -> Any:
    from .tensor import Tensor

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _dispatch_unary(operation_name: str, tensor: Any) -> Any:
    tensor = _as_tensor(tensor, "tensor")
    return type(tensor.data).dispatch_op(operation_name)


def _dispatch_binary(operation_name: str, lhs: Any, rhs: Any) -> Any:
    lhs = _as_tensor(lhs, "lhs")
    rhs = _as_tensor(rhs, "rhs")
    if type(lhs.data) is not type(rhs.data):
        raise TypeError("Tensor backing data classes must match")
    return type(lhs.data).dispatch_op(operation_name)


def _require_unevicted_tensor(value: Any, name: str) -> Any:
    tensor = _as_tensor(value, name)
    if tensor.data.is_evicted():
        raise RuntimeError(f"{name} data is evicted")
    return tensor


def _require_same_layout(lhs: Any, rhs: Any) -> None:
    if lhs.layout != rhs.layout:
        raise ValueError("Tensor layouts must match")


def _require_layout(tensor: Any, layout: Layout) -> None:
    if tensor.layout != layout:
        raise ValueError("Tensor layouts must match")


def _require_two_mode_tensor(tensor: Any, name: str) -> Any:
    tensor = _require_unevicted_tensor(tensor, name)
    if len(tensor.layout) != 2:
        raise ValueError(f"{name} must have a two-mode layout")
    return tensor


def _require_number(value: Any, name: str) -> Number:
    if not isinstance(value, Number):
        raise TypeError(f"{name} must be a numerical scalar")
    return value


def _is_integral_number(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _generic_binary_dtype(lhs: Any, rhs: Any) -> DataType:
    if lhs.dtype() is DataType.Floating or rhs.dtype() is DataType.Floating:
        return DataType.Floating
    return DataType.Any


def _generic_scalar_mul_dtype(tensor: Any, scalar: Any) -> DataType:
    if tensor.dtype() is DataType.Floating or not _is_integral_number(scalar):
        return DataType.Floating
    return DataType.Any


def _generic_pow_dtype(tensor: Any, exponent: Any) -> DataType:
    if tensor.dtype() is DataType.Floating:
        return DataType.Floating
    if not _is_integral_number(exponent) or exponent < 0:
        return DataType.Floating
    return DataType.Any


def _logical_values(tensor: Any) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def _mode_shape(layout: Layout, mode: int) -> Any:
    return layout.shape.top_level[mode]


def _mode_logical_size(layout: Layout, mode: int) -> int:
    shape = _mode_shape(layout, mode)
    if isinstance(shape, int):
        return shape
    return cast(int, shape.logical_size)


def _mode_stride(layout: Layout, mode: int) -> Any:
    return layout.stride.top_level[mode]


def _shape_from_modes(*modes: Any) -> Shape:
    if len(modes) == 1:
        return Shape(modes[0])
    return Shape(list(modes))


def _canonical_stride_level(shape_level: Any, stride: int) -> tuple[Any, int]:
    if isinstance(shape_level, int):
        return stride, stride * shape_level

    stride_level = []
    next_stride = stride
    for shape in shape_level:
        child_stride, next_stride = _canonical_stride_level(shape, next_stride)
        stride_level.append(child_stride)
    return stride_level, next_stride


def _canonical_layout_from_modes(*modes: Any) -> Layout:
    shape = _shape_from_modes(*modes)
    stride, _ = _canonical_stride_level(shape.top_level, 1)
    return Layout(shape, Stride(stride))


def _layout_from_modes(shapes: Iterable[Any], strides: Iterable[Any]) -> Layout:
    return Layout(Shape(list(shapes)), Stride(list(strides)))


def _cosize(layout: Layout) -> int:
    if layout.shape.logical_size == 0:
        return 0
    return max(_get_index(layout, i) for i in range(layout.shape.logical_size)) + 1


def _physical_values_for_layout(
    layout: Layout, logical_values: Iterable[Any]
) -> list[Any]:
    values = list(logical_values)
    if len(values) != layout.shape.logical_size:
        raise ValueError("Logical values length must match layout size")

    physical_values: list[Any] = [None] * _cosize(layout)
    for logical_index, value in enumerate(values):
        physical_values[_get_index(layout, logical_index)] = value
    return physical_values


def _tensor_with_layout_like(
    target: Any,
    layout: Layout,
    logical_values: Iterable[Any],
    dtype: DataType | None = None,
) -> Any:
    from .tensor import Tensor

    values = _physical_values_for_layout(layout, logical_values)
    if dtype is None:
        data = target.data.new_like(values)
    else:
        data = target.data.new_like(values, dtype=dtype)
    return Tensor(data, 0, layout)


def _detached_tensor_like(
    target: Any, values: Iterable[Any], dtype: DataType | None = None
) -> Any:
    return _tensor_with_layout_like(target, target.layout, values, dtype)


def _zero_tensor_like(target: Any) -> Any:
    return _detached_tensor_like(target, [0] * target.size())


def _copy_gradient_for(target: Any, gradient: Any) -> Any:
    _require_same_layout(target, gradient)
    return _detached_tensor_like(target, _logical_values(gradient))


def _copy_gradient_to_layout(target: Any, gradient: Any) -> Any:
    if target.size() != gradient.size():
        raise ValueError("Tensor layouts must have the same logical size")
    return _tensor_with_layout_like(target, target.layout, _logical_values(gradient))


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


class GenericExpOperation(Operation):
    """Generic elementwise exponential operation."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")

        values = [math.exp(tensor[i]) for i in range(tensor.size())]
        self.ctx["output_values"] = values
        return _detached_tensor_like(tensor, values, DataType.Floating)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        output_values = self.ctx["output_values"]
        values = [gradient[i] * output_values[i] for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericReLUOperation(Operation):
    """Generic elementwise rectified linear unit operation."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")

        values = [max(0, tensor[i]) for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        values = [gradient[i] if tensor[i] > 0 else 0 for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericSigmoidOperation(Operation):
    """Generic elementwise logistic sigmoid operation."""

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")

        values = [1.0 / (1.0 + math.exp(-tensor[i])) for i in range(tensor.size())]
        self.ctx["output_values"] = values
        return _detached_tensor_like(tensor, values, DataType.Floating)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        output_values = self.ctx["output_values"]
        values = [
            gradient[i] * output_values[i] * (1.0 - output_values[i])
            for i in range(gradient.size())
        ]
        return (_detached_tensor_like(tensor, values),)


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
        values = [gradient[i] for j in range(m_size) for i in range(n_size)]
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


class GenericViewOperation(Operation):
    """Generic tensor view operation sharing the input backing data."""

    def _forward(self, tensor: Any, key: Any) -> Any:
        from .tensor import Tensor

        tensor = _as_tensor(tensor, "tensor")
        offset_delta, output_layout = _view_layout_and_mapping(tensor, key)

        self.ctx["key"] = _normalize_view_key(key, len(tensor.layout))
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


def add(lhs: Any, rhs: Any) -> Any:
    """Add two tensors with matching layouts.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand with the same layout and backing data class.

    Returns:
        Tensor containing the elementwise sum.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([1, 2]), 0, layout)
        >>> y = Tensor(Generic([3, 4]), 0, layout)
        >>> neotorch.add(x, y)[1]
        6
    """

    return _dispatch_binary("add", lhs, rhs).forward(lhs, rhs)


def elementwise_mul(lhs: Any, rhs: Any) -> Any:
    """Multiply two tensors elementwise.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand with the same layout and backing data class.

    Returns:
        Tensor containing elementwise products.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([2, 3]), 0, layout)
        >>> y = Tensor(Generic([4, 5]), 0, layout)
        >>> neotorch.elementwise_mul(x, y)[1]
        15
    """

    return _dispatch_binary("elementwise_mul", lhs, rhs).forward(lhs, rhs)


def mul(tensor: Any, scalar: Any) -> Any:
    """Multiply a tensor by a scalar or another tensor.

    Tensor inputs dispatch to elementwise multiplication; non-tensor scalar
    inputs dispatch to scalar multiplication for the tensor's data backend.

    Args:
        tensor: Tensor operand to scale or multiply elementwise.
        scalar: Numerical scalar or tensor operand.

    Returns:
        Tensor containing scaled or elementwise multiplied values.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.mul(x, 10)[1]
        30
    """

    from .tensor import Tensor

    if isinstance(scalar, Tensor):
        return elementwise_mul(tensor, scalar)
    return _dispatch_unary("mul", tensor).forward(tensor, scalar)


def div(lhs: Any, rhs: Any) -> Any:
    """Divide two tensors elementwise.

    Args:
        lhs: Numerator tensor.
        rhs: Denominator tensor with the same layout and backing data class.

    Returns:
        Tensor containing elementwise quotients.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([8, 9]), 0, layout)
        >>> y = Tensor(Generic([2, 3]), 0, layout)
        >>> neotorch.div(x, y)[1]
        3.0
    """

    return _dispatch_binary("div", lhs, rhs).forward(lhs, rhs)


def exp(tensor: Any) -> Any:
    """Apply the exponential function elementwise.

    Args:
        tensor: Tensor whose logical values should be exponentiated.

    Returns:
        Tensor containing ``math.exp`` applied to each element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.exp(x)[0]
        1.0
    """

    return _dispatch_unary("exp", tensor).forward(tensor)


def relu(tensor: Any) -> Any:
    """Apply the rectified linear unit function elementwise.

    ReLU maps negative values to ``0`` and keeps positive values unchanged. Its
    autograd derivative is ``0`` for values less than or equal to ``0`` and
    ``1`` for values greater than ``0``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing ``max(0, value)`` for each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([-1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.relu(x)[1]
        2
    """

    return _dispatch_unary("relu", tensor).forward(tensor)


def sigmoid(tensor: Any) -> Any:
    """Apply the logistic sigmoid function elementwise.

    Sigmoid maps each value ``x`` to ``1 / (1 + math.exp(-x))``. Its autograd
    derivative multiplies the incoming gradient by ``sigmoid(x) * (1 -
    sigmoid(x))``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the logistic sigmoid of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.sigmoid(x)[0]
        0.5
    """

    return _dispatch_unary("sigmoid", tensor).forward(tensor)


def pow(tensor: Any, exponent: Any) -> Any:
    """Raise each tensor element to a scalar exponent.

    Args:
        tensor: Tensor containing base values.
        exponent: Numerical scalar exponent.

    Returns:
        Tensor containing each element raised to ``exponent``.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.pow(x, 3)[1]
        27
    """

    return _dispatch_unary("pow", tensor).forward(tensor, exponent)


@overload
def reduce(tensor: Any) -> Any: ...


@overload
def reduce(tensor: Any, description: str) -> Any: ...


def reduce(tensor: Any, description: Any = _REDUCE_DESCRIPTION_MISSING) -> Any:
    """Sum-reduce a tensor.

    With no description, the tensor must have two top-level modes and the second
    mode is summed. With a string description, omitted dimensions are reduced
    through the Neotorch hierarchical-layout lowering path.

    Args:
        tensor: Tensor to reduce.
        description: Optional Neotorch layout reduce command.

    Returns:
        Tensor containing the sum-reduced values.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.reduce(x)[1]
        12
        >>> neotorch.reduce(x, "a b -> a")[1]
        12
    """

    if description is _REDUCE_DESCRIPTION_MISSING:
        return _dispatch_unary("reduce", tensor).forward(tensor)
    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from .einops import reduce as einops_reduce

    return einops_reduce(tensor, description)


def matmul(lhs: Any, rhs: Any) -> Any:
    """Multiply two two-mode tensors.

    The first mode of each input is kept, and the second modes must have the
    same logical size and are contracted with a dot product.

    Args:
        lhs: Left two-mode tensor.
        rhs: Right two-mode tensor with matching second-mode logical size.

    Returns:
        Tensor with layout formed from the first mode of each input.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.matmul(lhs, rhs)[1, 1]
        22
    """

    return _dispatch_binary("matmul", lhs, rhs).forward(lhs, rhs)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    """Contract two tensors using a Neotorch contraction description.

    The string form is parsed by ``neotorch.einops`` and lowered into
    rearrange, matmul, and final rearrange operations.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor containing the requested contraction result.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.einsum(lhs, rhs, "a b, c b -> a c")[1, 1]
        22
    """

    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from .einops import einsum as einops_einsum

    return einops_einsum(lhs, rhs, description)


@overload
def rearrange(tensor: Any, output: str) -> Any: ...


@overload
def rearrange(tensor: Any, output: Tree, selection: Tree | None = None) -> Any: ...


def rearrange(tensor: Any, output: Tree | str, selection: Tree | None = None) -> Any:
    """Rearrange a tensor layout.

    Tree inputs call the lower-level rearrange operation directly. A string
    output is parsed as a Neotorch layout command and must not be combined with
    an explicit selection.

    Args:
        tensor: Tensor whose layout should be rearranged.
        output: Output Tree or Neotorch layout rearrange description.
        selection: Optional Tree selecting source layout subtrees.

    Returns:
        Tensor view with the rearranged layout.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Node, Shape, Stride, Tensor, Tree
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.rearrange(x, "a b -> b a")[2, 1]
        6
        >>> output = Tree(Node.id(1), Node.id(0))
        >>> selection = Tree(Node.Leaf, Node.Leaf)
        >>> neotorch.rearrange(x, output, selection)[2, 1]
        6
    """

    if isinstance(output, str):
        if selection is not None:
            raise TypeError(
                "String rearrange descriptions do not accept an explicit selection"
            )
        from .einops import rearrange as einops_rearrange

        return einops_rearrange(tensor, output)
    return _dispatch_unary("rearrange", tensor).forward(tensor, output, selection)


def permute(tensor: Any, *order: Any) -> Any:
    """Permute top-level tensor layout modes.

    Args:
        tensor: Tensor whose layout modes should be reordered.
        order: Permutation of every top-level mode index.

    Returns:
        Tensor view with top-level modes in the requested order.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.permute(x, 1, 0)[2, 1]
        6
    """

    return _dispatch_unary("permute", tensor).forward(tensor, *order)


def view(tensor: Any, key: Any) -> Any:
    """Create a tensor view from integer and slice keys.

    Integers select and remove a mode where supported; slices preserve a mode
    while adjusting the output layout and offset.

    Args:
        tensor: Tensor to view.
        key: Integer and slice key tuple for top-level modes.

    Returns:
        Tensor view sharing the input backing data with an adjusted layout.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> row = neotorch.view(x, (1, slice(None)))
        >>> row[2]
        6
    """

    return _dispatch_unary("view", tensor).forward(tensor, key)


__all__ = [
    "GenericAddOperation",
    "GenericDivOperation",
    "GenericElementwiseMulOperation",
    "GenericExpOperation",
    "GenericMatmulOperation",
    "GenericPowOperation",
    "GenericReLUOperation",
    "GenericReduceSumOperation",
    "GenericScalarMulOperation",
    "GenericSigmoidOperation",
    "GenericViewOperation",
    "Operation",
    "PermuteOperation",
    "RearrangeOperation",
    "add",
    "div",
    "elementwise_mul",
    "einsum",
    "exp",
    "is_grad_enabled",
    "matmul",
    "mul",
    "no_grad",
    "permute",
    "pow",
    "rearrange",
    "reduce",
    "relu",
    "set_grad_enabled",
    "sigmoid",
    "view",
]
