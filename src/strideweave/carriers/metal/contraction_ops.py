"""Metal-owned matrix multiplication and convolution operations."""

from __future__ import annotations

from typing import Any

from ...layout import Shape
from ...tensor import Tensor
from ..dtype import DType
from ..generic.convolution_ops import (
    _as_index,
    _mode_extents,
    _normalize_dims,
    _normalize_padding,
    _normalize_positive_sequence,
    _require_spatial_leaves,
)
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _canonical_layout_from_modes,
    _mode_logical_size,
    _mode_shape,
    _require_layout,
    _require_live_tensor,
    _require_two_mode_tensor,
)
from ..operation_policy import resolve_operation_plan
from ._contraction_executor import ConvGeometry, execute_conv, execute_matmul
from .capabilities import _require_executable_plan
from .carrier import Metal

_OPERATIONS = frozenset({"matmul", "conv_general"})


def _require_metal_tensor(value: Any, name: str) -> Tensor:
    tensor = _require_live_tensor(value, name)
    if type(tensor.carrier) is not Metal:
        raise ValueError(f"{name} must use the exact Metal carrier class")
    return tensor


def _gradient_layout(tensor: Tensor) -> Any:
    if tensor.layout.is_injective:
        return tensor.layout
    return _canonical_layout_for_shape(tensor.layout.shape)


class MetalContractionOperation(Operation):
    """Private Metal operation selected by a contraction name."""

    def __init__(self, operation_name: str) -> None:
        super().__init__()
        if operation_name not in _OPERATIONS:
            raise ValueError(f"unknown Metal contraction operation {operation_name!r}")
        self._metal_operation_name = operation_name

    def _forward(self, *operands: Any) -> Tensor:
        if self._metal_operation_name == "matmul":
            return self._forward_matmul(operands)
        return self._forward_conv(operands)

    def _forward_matmul(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) != 2:
            raise TypeError("matmul expects lhs and rhs tensors")
        lhs = _require_two_mode_tensor(_require_metal_tensor(operands[0], "lhs"), "lhs")
        rhs = _require_two_mode_tensor(_require_metal_tensor(operands[1], "rhs"), "rhs")
        plan = _require_executable_plan(
            resolve_operation_plan(
                "matmul",
                lhs.dtype(),
                rhs.dtype(),
                options=self._execution_options,
            )
        )
        n_size = _mode_logical_size(lhs.layout, 0)
        k_size = _mode_logical_size(lhs.layout, 1)
        m_size = _mode_logical_size(rhs.layout, 0)
        if k_size != _mode_logical_size(rhs.layout, 1):
            raise ValueError("Matmul inner dimensions must match")
        output_layout = _canonical_layout_from_modes(
            _mode_shape(lhs.layout, 0), _mode_shape(rhs.layout, 0)
        )
        self.ctx.update(
            {
                "output_layout": output_layout,
                "n_size": n_size,
                "m_size": m_size,
                "k_size": k_size,
            }
        )
        return execute_matmul(
            "matmul_forward",
            lhs,
            rhs,
            output_layout=output_layout,
            n_size=n_size,
            m_size=m_size,
            k_size=k_size,
            plan=plan,
        )

    def _forward_conv(self, operands: tuple[Any, ...]) -> Tensor:
        if len(operands) < 4 or len(operands) > 10:
            raise TypeError(
                "conv_general expects lhs, kernel, strides, padding, and optional "
                "dilation, group, and role-permutation arguments"
            )
        lhs = _require_metal_tensor(operands[0], "lhs")
        kernel = _require_metal_tensor(operands[1], "kernel")
        plan = _require_executable_plan(
            resolve_operation_plan("conv_general", lhs.dtype(), kernel.dtype())
        )
        lhs_rank = len(lhs.layout)
        kernel_rank = len(kernel.layout)
        spatial_rank = lhs_rank - 2
        if spatial_rank < 1 or kernel_rank - 2 != spatial_rank:
            raise ValueError(
                "conv_general requires lhs and kernel with the same spatial rank "
                "of at least one"
            )
        lhs_roles = _normalize_dims(
            operands[7] if len(operands) >= 8 else None, lhs_rank, "lhs_dims"
        )
        kernel_roles = _normalize_dims(
            operands[8] if len(operands) >= 9 else None, kernel_rank, "kernel_dims"
        )
        output_roles = _normalize_dims(
            operands[9] if len(operands) >= 10 else None,
            spatial_rank + 2,
            "output_dims",
        )
        _require_spatial_leaves(lhs, lhs_roles, "lhs")
        _require_spatial_leaves(kernel, kernel_roles, "kernel")
        canonical_roles = tuple(range(lhs_rank))
        if lhs_roles != canonical_roles:
            lhs = lhs.carrier.dispatch_op("permute").forward(lhs, *lhs_roles)
        if kernel_roles != canonical_roles:
            kernel = kernel.carrier.dispatch_op("permute").forward(
                kernel, *kernel_roles
            )
        self.store_inputs(lhs, kernel)
        strides = _normalize_positive_sequence(operands[2], spatial_rank, "strides")
        padding = _normalize_padding(operands[3], spatial_rank)
        lhs_dilation = (
            (1,) * spatial_rank
            if len(operands) < 5 or operands[4] is None
            else _normalize_positive_sequence(operands[4], spatial_rank, "lhs_dilation")
        )
        kernel_dilation = (
            (1,) * spatial_rank
            if len(operands) < 6 or operands[5] is None
            else _normalize_positive_sequence(
                operands[5], spatial_rank, "kernel_dilation"
            )
        )
        groups = _as_index(operands[6] if len(operands) >= 7 else 1, "feature_groups")
        if groups <= 0:
            raise ValueError("feature_groups must be positive")

        lhs_extents = _mode_extents(lhs)
        kernel_extents = _mode_extents(kernel)
        batch_size, input_features = lhs_extents[:2]
        output_features, kernel_input_features = kernel_extents[:2]
        if input_features % groups:
            raise ValueError(
                "lhs input feature extent must be divisible by feature_groups"
            )
        if output_features % groups:
            raise ValueError(
                "kernel output feature extent must be divisible by feature_groups"
            )
        channels_per_group = input_features // groups
        if kernel_input_features != channels_per_group:
            raise ValueError(
                "kernel input feature extent must equal lhs input features divided "
                "by feature_groups"
            )
        input_spatial = lhs_extents[2:]
        kernel_spatial = kernel_extents[2:]
        output_spatial: list[int] = []
        for dimension in range(spatial_rank):
            effective_input = (input_spatial[dimension] - 1) * lhs_dilation[
                dimension
            ] + 1
            effective_kernel = (kernel_spatial[dimension] - 1) * kernel_dilation[
                dimension
            ] + 1
            numerator = (
                padding[dimension][0]
                + effective_input
                + padding[dimension][1]
                - effective_kernel
            )
            extent = numerator // strides[dimension] + 1
            if extent <= 0:
                raise ValueError("conv_general output spatial extents must be positive")
            output_spatial.append(extent)
        output_shape = Shape(
            [
                lhs.layout.shape.top_level[0],
                kernel.layout.shape.top_level[0],
                *output_spatial,
            ]
        )
        output_layout = _canonical_layout_for_shape(output_shape)
        geometry = ConvGeometry(
            batch_size,
            input_features,
            output_features,
            channels_per_group,
            input_spatial,
            kernel_spatial,
            tuple(output_spatial),
            strides,
            padding,
            lhs_dilation,
            kernel_dilation,
            groups,
        )
        self.ctx.update(
            {
                "output_layout": output_layout,
                "geometry": geometry,
                "output_dims": output_roles,
            }
        )
        return execute_conv(
            "conv_forward",
            lhs,
            kernel,
            output_layout=output_layout,
            geometry=geometry,
            plan=plan,
        )

    def backward(self, gradient: Any) -> tuple[Tensor, Tensor]:
        lhs, rhs = self.inputs()
        lhs = _require_metal_tensor(lhs, "lhs")
        rhs = _require_metal_tensor(rhs, "rhs")
        gradient = _require_metal_tensor(gradient, "gradient")
        if gradient.dtype() is not DType.Float32:
            raise TypeError("Metal contraction gradients must use DType.Float32")
        _require_layout(gradient, self.ctx["output_layout"])
        if self._metal_operation_name == "matmul":
            n_size = self.ctx["n_size"]
            m_size = self.ctx["m_size"]
            k_size = self.ctx["k_size"]
            lhs_gradient = execute_matmul(
                "matmul_grad_lhs",
                gradient,
                rhs,
                output_layout=_gradient_layout(lhs),
                n_size=n_size,
                m_size=m_size,
                k_size=k_size,
            )
            rhs_gradient = execute_matmul(
                "matmul_grad_rhs",
                gradient,
                lhs,
                output_layout=_gradient_layout(rhs),
                n_size=n_size,
                m_size=m_size,
                k_size=k_size,
            )
            return lhs_gradient, rhs_gradient

        geometry = self.ctx["geometry"]
        lhs_gradient = execute_conv(
            "conv_grad_lhs",
            lhs,
            rhs,
            output_layout=_gradient_layout(lhs),
            geometry=geometry,
            gradient=gradient,
        )
        kernel_gradient = execute_conv(
            "conv_grad_kernel",
            lhs,
            rhs,
            output_layout=_gradient_layout(rhs),
            geometry=geometry,
            gradient=gradient,
        )
        return lhs_gradient, kernel_gradient


def metal_contraction_operation(operation_name: str) -> MetalContractionOperation:
    """Return a fresh Metal-owned operation for a contraction name."""
    return MetalContractionOperation(operation_name)


__all__: list[str] = []
