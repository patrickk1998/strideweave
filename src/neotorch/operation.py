from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib import import_module
from numbers import Number
from operator import index as operator_index
from typing import Any, cast, overload

from .layout import Layout, Shape, Stride, Tree

_get_index = cast(Any, import_module("neotorch._index")).get_index

_operation = import_module("neotorch._operation")
Operation = cast(type[Any], _operation.Operation)
is_grad_enabled = cast(Callable[[], bool], _operation.is_grad_enabled)
set_grad_enabled = cast(Callable[[bool], None], _operation.set_grad_enabled)
_REDUCE_DESCRIPTION_MISSING = object()


@contextmanager
def no_grad() -> Iterator[None]:
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
    target: Any, layout: Layout, logical_values: Iterable[Any]
) -> Any:
    from .tensor import Tensor

    data = target.data.new_like(_physical_values_for_layout(layout, logical_values))
    return Tensor(data, 0, layout)


def _detached_tensor_like(target: Any, values: Iterable[Any]) -> Any:
    return _tensor_with_layout_like(target, target.layout, values)


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
    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        values = [lhs[i] + rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values)

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        return _copy_gradient_for(lhs, gradient), _copy_gradient_for(rhs, gradient)


class GenericScalarMulOperation(Operation):
    def _forward(self, tensor: Any, scalar: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")
        scalar = _require_number(scalar, "scalar")

        self.ctx["scalar"] = scalar
        values = [tensor[i] * scalar for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        scalar = self.ctx["scalar"]
        values = [gradient[i] * scalar for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericElementwiseMulOperation(Operation):
    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        values = [lhs[i] * rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values)

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
    def _forward(self, lhs: Any, rhs: Any) -> Any:
        lhs = _require_unevicted_tensor(lhs, "lhs")
        rhs = _require_unevicted_tensor(rhs, "rhs")
        _require_same_layout(lhs, rhs)

        values = [lhs[i] / rhs[i] for i in range(lhs.size())]
        return _detached_tensor_like(lhs, values)

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
    def _forward(self, tensor: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")

        values = [math.exp(tensor[i]) for i in range(tensor.size())]
        self.ctx["output_values"] = values
        return _detached_tensor_like(tensor, values)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_unevicted_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        output_values = self.ctx["output_values"]
        values = [gradient[i] * output_values[i] for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericPowOperation(Operation):
    def _forward(self, tensor: Any, exponent: Any) -> Any:
        tensor = _require_unevicted_tensor(tensor, "tensor")
        exponent = _require_number(exponent, "exponent")

        self.ctx["exponent"] = exponent
        values = [tensor[i] ** exponent for i in range(tensor.size())]
        return _detached_tensor_like(tensor, values)

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
        values = [
            sum(lhs[i, k] * rhs[j, k] for k in range(lhs_k_size))
            for j in range(m_size)
            for i in range(n_size)
        ]
        return _tensor_with_layout_like(lhs, output_layout, values)

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
    return _dispatch_binary("add", lhs, rhs).forward(lhs, rhs)


def elementwise_mul(lhs: Any, rhs: Any) -> Any:
    return _dispatch_binary("elementwise_mul", lhs, rhs).forward(lhs, rhs)


def mul(tensor: Any, scalar: Any) -> Any:
    from .tensor import Tensor

    if isinstance(scalar, Tensor):
        return elementwise_mul(tensor, scalar)
    return _dispatch_unary("mul", tensor).forward(tensor, scalar)


def div(lhs: Any, rhs: Any) -> Any:
    return _dispatch_binary("div", lhs, rhs).forward(lhs, rhs)


def exp(tensor: Any) -> Any:
    return _dispatch_unary("exp", tensor).forward(tensor)


def pow(tensor: Any, exponent: Any) -> Any:
    return _dispatch_unary("pow", tensor).forward(tensor, exponent)


@overload
def reduce(tensor: Any) -> Any: ...


@overload
def reduce(tensor: Any, description: str) -> Any: ...


def reduce(tensor: Any, description: Any = _REDUCE_DESCRIPTION_MISSING) -> Any:
    if description is _REDUCE_DESCRIPTION_MISSING:
        return _dispatch_unary("reduce", tensor).forward(tensor)
    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from .einops import reduce as einops_reduce

    return einops_reduce(tensor, description)


def matmul(lhs: Any, rhs: Any) -> Any:
    return _dispatch_binary("matmul", lhs, rhs).forward(lhs, rhs)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from .einops import einsum as einops_einsum

    return einops_einsum(lhs, rhs, description)


@overload
def rearrange(tensor: Any, output: str) -> Any: ...


@overload
def rearrange(tensor: Any, output: Tree, selection: Tree | None = None) -> Any: ...


def rearrange(tensor: Any, output: Tree | str, selection: Tree | None = None) -> Any:
    if isinstance(output, str):
        if selection is not None:
            raise TypeError(
                "String rearrange descriptions do not accept an explicit selection"
            )
        from .einops import rearrange as einops_rearrange

        return einops_rearrange(tensor, output)
    return _dispatch_unary("rearrange", tensor).forward(tensor, output, selection)


def permute(tensor: Any, *order: Any) -> Any:
    return _dispatch_unary("permute", tensor).forward(tensor, *order)


def view(tensor: Any, key: Any) -> Any:
    return _dispatch_unary("view", tensor).forward(tensor, key)


__all__ = [
    "GenericAddOperation",
    "GenericDivOperation",
    "GenericElementwiseMulOperation",
    "GenericExpOperation",
    "GenericMatmulOperation",
    "GenericPowOperation",
    "GenericReduceSumOperation",
    "GenericScalarMulOperation",
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
    "set_grad_enabled",
    "view",
]
