from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from numbers import Number
from typing import Any, cast

from .data import Generic
from .layout import Layout, Shape, Stride, Tree

_get_index = cast(Any, import_module("neotorch._index")).get_index

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


def _require_layout(tensor: Any, layout: Layout) -> None:
    if tensor.layout != layout:
        raise ValueError("Tensor layouts must match")


def _require_two_mode_tensor(tensor: Any, name: str) -> Any:
    tensor = _require_exact_generic_tensor(tensor, name)
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


def _copy_gradient_for(target: Any, gradient: Any) -> Any:
    _require_same_layout(target, gradient)
    return _detached_tensor_like(target, _logical_values(gradient))


def _copy_gradient_to_layout(target: Any, gradient: Any) -> Any:
    if target.size() != gradient.size():
        raise ValueError("Tensor layouts must have the same logical size")
    return _tensor_with_layout_like(target, target.layout, _logical_values(gradient))


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


class GenericScalarMulOperation(Operation):
    def forward(self, tensor: Any, scalar: Any) -> Any:
        tensor = _require_exact_generic_tensor(tensor, "tensor")
        scalar = _require_number(scalar, "scalar")

        self.store_inputs(tensor)
        self.ctx["scalar"] = scalar
        values = [tensor[i] * scalar for i in range(tensor.size())]
        result = _detached_tensor_like(tensor, values)
        result.autograd_ctx = self
        return result

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_exact_generic_tensor(gradient, "gradient")
        _require_same_layout(tensor, gradient)

        scalar = self.ctx["scalar"]
        values = [gradient[i] * scalar for i in range(gradient.size())]
        return (_detached_tensor_like(tensor, values),)


class GenericReduceSumOperation(Operation):
    def forward(self, tensor: Any) -> Any:
        tensor = _require_two_mode_tensor(tensor, "tensor")

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        output_layout = _canonical_layout_from_modes(_mode_shape(tensor.layout, 0))

        self.store_inputs(tensor)
        self.ctx["output_layout"] = output_layout
        values = [sum(tensor[i, j] for j in range(m_size)) for i in range(n_size)]
        result = _tensor_with_layout_like(tensor, output_layout, values)
        result.autograd_ctx = self
        return result

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_exact_generic_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])

        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        values = [gradient[i] for j in range(m_size) for i in range(n_size)]
        return (_detached_tensor_like(tensor, values),)


class GenericMatmulOperation(Operation):
    def forward(self, lhs: Any, rhs: Any) -> Any:
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

        self.store_inputs(lhs, rhs)
        self.ctx["output_layout"] = output_layout
        values = [
            sum(lhs[i, k] * rhs[j, k] for k in range(lhs_k_size))
            for j in range(m_size)
            for i in range(n_size)
        ]
        result = _tensor_with_layout_like(lhs, output_layout, values)
        result.autograd_ctx = self
        return result

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, rhs = self.inputs()
        gradient = _require_exact_generic_tensor(gradient, "gradient")
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


class RearrangeOperation(Operation):
    def forward(self, tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
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

        self.store_inputs(tensor)
        self.ctx["output"] = output
        self.ctx["selection"] = effective_selection
        self.ctx["output_layout"] = output_layout
        result = Tensor(tensor.data, tensor.offset, output_layout)
        result.autograd_ctx = self
        return result

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


def add(lhs: Any, rhs: Any) -> Any:
    return GenericAddOperation().forward(lhs, rhs)


def mul(tensor: Any, scalar: Any) -> Any:
    return GenericScalarMulOperation().forward(tensor, scalar)


def reduce(tensor: Any) -> Any:
    return GenericReduceSumOperation().forward(tensor)


def matmul(lhs: Any, rhs: Any) -> Any:
    return GenericMatmulOperation().forward(lhs, rhs)


def rearrange(tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
    return RearrangeOperation().forward(tensor, output, selection)


__all__ = [
    "GenericAddOperation",
    "GenericMatmulOperation",
    "GenericReduceSumOperation",
    "GenericScalarMulOperation",
    "Operation",
    "RearrangeOperation",
    "add",
    "matmul",
    "mul",
    "rearrange",
    "reduce",
]
