"""Reference implementation of the Generic ``conv_general`` primitive.

The carrier and public-function wiring for this operation is deliberately kept
out of this module.  The operation here owns the coordinate contract: role
permutations are normalized to ``[batch, feature, spatial...]``; spatial
coordinates use the first-mode-fast ordinal convention; and every contraction
is accumulated sequentially in binary32.  The same explicit loops are the
reference a native backend can compare against.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as operator_index
from typing import Any

from ...layout import Shape
from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _require_layout,
    _require_live_tensor,
    _tensor_with_layout_like,
)
from .execution import binary_arithmetic, executing, gradient_arithmetic
from .numerics import binary32

__all__ = ["GenericConvGeneralOperation"]


def _mode_extent(mode: Any) -> int:
    """Return the flattened logical extent of one top-level mode."""
    return mode if isinstance(mode, int) else mode.logical_size


def _mode_extents(tensor: Any) -> tuple[int, ...]:
    """Return first-mode-fast extents for the tensor's top-level modes."""
    return tuple(_mode_extent(mode) for mode in tensor.layout.shape.top_level)


def _logical_ordinal(coordinates: Sequence[int], extents: Sequence[int]) -> int:
    """Encode top-level coordinates in StrideWeave's first-mode-fast order."""
    if len(coordinates) != len(extents):
        raise ValueError("Coordinate rank does not match tensor rank")
    ordinal = 0
    factor = 1
    for coordinate, extent in zip(coordinates, extents, strict=True):
        if coordinate < 0 or coordinate >= extent:
            raise ValueError("Coordinate is outside tensor shape")
        ordinal += coordinate * factor
        factor *= extent
    return ordinal


def _logical_coordinates(ordinal: int, extents: Sequence[int]) -> tuple[int, ...]:
    """Decode a first-mode-fast logical ordinal into top-level coordinates."""
    coordinates: list[int] = []
    remainder = ordinal
    for extent in extents:
        coordinates.append(remainder % extent)
        remainder //= extent
    return tuple(coordinates)


def _as_index(value: Any, name: str) -> int:
    """Normalize one integer configuration value and reject booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _normalize_positive_sequence(values: Any, rank: int, name: str) -> tuple[int, ...]:
    """Validate one positive per-spatial-dimension integer sequence."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain one integer per spatial dimension")
    try:
        normalized = tuple(_as_index(value, name) for value in values)
    except TypeError as exc:
        if str(exc).startswith(f"{name} must"):
            raise
        raise TypeError(
            f"{name} must contain one integer per spatial dimension"
        ) from exc
    if len(normalized) != rank:
        raise ValueError(f"{name} must have one entry per spatial dimension")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integers")
    return normalized


def _normalize_padding(values: Any, rank: int) -> tuple[tuple[int, int], ...]:
    """Validate non-negative ``(low, high)`` pairs for each spatial mode."""
    if isinstance(values, (str, bytes)):
        raise TypeError("padding must contain one (low, high) pair per dimension")
    try:
        pairs = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "padding must contain one (low, high) pair per dimension"
        ) from exc
    if len(pairs) != rank:
        raise ValueError("padding must have one pair per spatial dimension")
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if isinstance(pair, (str, bytes)):
            raise TypeError("padding entries must be (low, high) integer pairs")
        try:
            low, high = tuple(pair)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "padding entries must be (low, high) integer pairs"
            ) from exc
        low_index = _as_index(low, "padding values")
        high_index = _as_index(high, "padding values")
        if low_index < 0 or high_index < 0:
            raise ValueError("padding values must be non-negative")
        normalized.append((low_index, high_index))
    return tuple(normalized)


def _normalize_dims(spec: Any, rank: int, name: str) -> tuple[int, ...]:
    """Normalize a role specification to a permutation of top-level modes.

    A role specification lists source top-level mode indices in canonical role
    order.  For example, ``lhs_dims=(0, 2, 1)`` means batch mode ``0``, feature
    mode ``2``, and spatial mode ``1``.  ``None`` is the canonical ordering.
    """
    if spec is None:
        return tuple(range(rank))
    if isinstance(spec, (str, bytes)):
        raise TypeError(f"{name} must be a permutation of top-level mode indices")
    try:
        values = tuple(_as_index(value, name) for value in spec)
    except TypeError as exc:
        if str(exc).startswith(f"{name} must be"):
            raise
        raise TypeError(
            f"{name} must be a permutation of top-level mode indices"
        ) from exc
    if len(values) != rank or set(values) != set(range(rank)):
        raise ValueError(f"{name} must be a permutation of all top-level modes")
    return values


def _require_float32_tensor(tensor: Any, name: str) -> Any:
    """Validate a live Float32 tensor operand."""
    tensor = _require_live_tensor(tensor, name)
    if tensor.dtype() is not DType.Float32:
        raise TypeError(f"{name} must have dtype DType.Float32")
    return tensor


def _require_compatible_carriers(lhs: Any, kernel: Any) -> None:
    """Require operands to use one dispatching carrier implementation."""
    if type(lhs.carrier) is not type(kernel.carrier):
        raise ValueError("lhs and kernel must use compatible carrier classes")


def _require_spatial_leaves(tensor: Any, dims: tuple[int, ...], name: str) -> None:
    """Require each selected spatial mode to be a top-level leaf."""
    for mode in dims[2:]:
        if not isinstance(tensor.layout.shape.top_level[mode], int):
            raise ValueError(f"{name} spatial role modes must be leaves")


def _read_role(tensor: Any, dims: tuple[int, ...], coordinates: Sequence[int]) -> Any:
    """Read a tensor using coordinates expressed in canonical role order."""
    source_extents = _mode_extents(tensor)
    source_coordinates = [0] * len(source_extents)
    for role_coordinate, source_mode in zip(coordinates, dims, strict=True):
        source_coordinates[source_mode] = role_coordinate
    return tensor[_logical_ordinal(source_coordinates, source_extents)]


def _role_ordinal(
    dims: tuple[int, ...], coordinates: Sequence[int], tensor: Any
) -> int:
    """Encode canonical role coordinates as an operand logical ordinal."""
    source_extents = _mode_extents(tensor)
    source_coordinates = [0] * len(source_extents)
    for role_coordinate, source_mode in zip(coordinates, dims, strict=True):
        source_coordinates[source_mode] = role_coordinate
    return _logical_ordinal(source_coordinates, source_extents)


def _conv_arithmetic(lhs: Any, kernel: Any) -> Any:
    """Resolve the policy-owned Float32 arithmetic for convolution."""
    return binary_arithmetic("conv_general", lhs, kernel, DType.Float32)


class GenericConvGeneralOperation(Operation):
    """Generic Float32 arbitrary-rank grouped cross-correlation operation."""

    def _forward(
        self,
        lhs: Any,
        kernel: Any,
        strides: Any,
        padding: Any,
        lhs_dilation: Any = None,
        kernel_dilation: Any = None,
        feature_groups: Any = 1,
        lhs_dims: Any = None,
        kernel_dims: Any = None,
        output_dims: Any = None,
    ) -> Any:
        lhs = _require_float32_tensor(lhs, "lhs")
        kernel = _require_float32_tensor(kernel, "kernel")
        _require_compatible_carriers(lhs, kernel)

        lhs_rank = len(lhs.layout)
        kernel_rank = len(kernel.layout)
        spatial_rank = lhs_rank - 2
        if spatial_rank < 1 or kernel_rank - 2 != spatial_rank:
            raise ValueError(
                "conv_general requires lhs and kernel with the same spatial rank "
                "of at least one"
            )
        lhs_roles = _normalize_dims(lhs_dims, lhs_rank, "lhs_dims")
        kernel_roles = _normalize_dims(kernel_dims, kernel_rank, "kernel_dims")
        output_rank = spatial_rank + 2
        output_roles = _normalize_dims(output_dims, output_rank, "output_dims")
        _require_spatial_leaves(lhs, lhs_roles, "lhs")
        _require_spatial_leaves(kernel, kernel_roles, "kernel")

        normalized_strides = _normalize_positive_sequence(
            strides, spatial_rank, "strides"
        )
        normalized_lhs_dilation = (
            (1,) * spatial_rank
            if lhs_dilation is None
            else _normalize_positive_sequence(
                lhs_dilation, spatial_rank, "lhs_dilation"
            )
        )
        normalized_kernel_dilation = (
            (1,) * spatial_rank
            if kernel_dilation is None
            else _normalize_positive_sequence(
                kernel_dilation, spatial_rank, "kernel_dilation"
            )
        )
        normalized_padding = _normalize_padding(padding, spatial_rank)
        groups = _as_index(feature_groups, "feature_groups")
        if groups <= 0:
            raise ValueError("feature_groups must be positive")

        lhs_extents = _mode_extents(lhs)
        kernel_extents = _mode_extents(kernel)
        batch_size = lhs_extents[lhs_roles[0]]
        input_features = lhs_extents[lhs_roles[1]]
        output_features = kernel_extents[kernel_roles[0]]
        kernel_input_features = kernel_extents[kernel_roles[1]]
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

        input_spatial = tuple(lhs_extents[mode] for mode in lhs_roles[2:])
        kernel_spatial = tuple(kernel_extents[mode] for mode in kernel_roles[2:])
        output_spatial: list[int] = []
        for dimension in range(spatial_rank):
            effective_input = (input_spatial[dimension] - 1) * normalized_lhs_dilation[
                dimension
            ] + 1
            effective_kernel = (
                kernel_spatial[dimension] - 1
            ) * normalized_kernel_dilation[dimension] + 1
            numerator = (
                normalized_padding[dimension][0]
                + effective_input
                + normalized_padding[dimension][1]
                - effective_kernel
            )
            extent = numerator // normalized_strides[dimension] + 1
            if extent <= 0:
                raise ValueError("conv_general output spatial extents must be positive")
            output_spatial.append(extent)

        # The primitive result is always canonical.  ``output_dims`` is retained
        # as validated metadata for the public frontend's subsequent permute.
        batch_shape = lhs.layout.shape.top_level[lhs_roles[0]]
        output_feature_shape = kernel.layout.shape.top_level[kernel_roles[0]]
        output_shape = Shape([batch_shape, output_feature_shape, *output_spatial])
        output_layout = _canonical_layout_for_shape(output_shape)
        self.ctx.update(
            {
                "output_layout": output_layout,
                "lhs_dims": lhs_roles,
                "kernel_dims": kernel_roles,
                "output_dims": output_roles,
                "strides": normalized_strides,
                "padding": normalized_padding,
                "lhs_dilation": normalized_lhs_dilation,
                "kernel_dilation": normalized_kernel_dilation,
                "feature_groups": groups,
                "output_spatial": tuple(output_spatial),
            }
        )

        output_extents = (batch_size, output_features, *output_spatial)
        output_size = output_shape.logical_size
        arithmetic = _conv_arithmetic(lhs, kernel)
        with executing(arithmetic):
            values: list[Any] = []
            for output_ordinal in range(output_size):
                output_coordinates = _logical_coordinates(
                    output_ordinal, output_extents
                )
                batch, output_feature = output_coordinates[:2]
                output_position = output_coordinates[2:]
                group = output_feature // (output_features // groups)
                terms = []
                for contraction_ordinal in range(
                    channels_per_group * _product(kernel_spatial)
                ):
                    contraction_coordinates = _logical_coordinates(
                        contraction_ordinal, (channels_per_group, *kernel_spatial)
                    )
                    channel, kernel_position = (
                        contraction_coordinates[0],
                        contraction_coordinates[1:],
                    )
                    input_position = []
                    valid = True
                    for dimension, kernel_coordinate in enumerate(kernel_position):
                        dilated = (
                            output_position[dimension] * normalized_strides[dimension]
                            + kernel_coordinate * normalized_kernel_dilation[dimension]
                            - normalized_padding[dimension][0]
                        )
                        if (
                            dilated < 0
                            or dilated % normalized_lhs_dilation[dimension] != 0
                        ):
                            valid = False
                            break
                        coordinate = dilated // normalized_lhs_dilation[dimension]
                        if coordinate >= input_spatial[dimension]:
                            valid = False
                            break
                        input_position.append(coordinate)
                    lhs_value = (
                        _read_role(
                            lhs,
                            lhs_roles,
                            (
                                batch,
                                group * channels_per_group + channel,
                                *input_position,
                            ),
                        )
                        if valid
                        else 0.0
                    )
                    kernel_value = _read_role(
                        kernel,
                        kernel_roles,
                        (output_feature, channel, *kernel_position),
                    )
                    terms.append(
                        arithmetic.convert(lhs_value) * arithmetic.convert(kernel_value)
                    )
                values.append(arithmetic.store(arithmetic.total(terms)))
        return _tensor_with_layout_like(
            lhs, output_layout, values, arithmetic.result_dtype
        )

    def backward(self, gradient: Any) -> tuple[Any, Any]:
        lhs, kernel = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        if gradient.dtype() is not DType.Float32:
            raise TypeError("gradient must have dtype DType.Float32")

        lhs_roles = self.ctx["lhs_dims"]
        kernel_roles = self.ctx["kernel_dims"]
        strides = self.ctx["strides"]
        padding = self.ctx["padding"]
        lhs_dilation = self.ctx["lhs_dilation"]
        kernel_dilation = self.ctx["kernel_dilation"]
        groups = self.ctx["feature_groups"]
        output_spatial = self.ctx["output_spatial"]

        lhs_extents = _mode_extents(lhs)
        kernel_extents = _mode_extents(kernel)
        batch_size = lhs_extents[lhs_roles[0]]
        input_features = lhs_extents[lhs_roles[1]]
        output_features = kernel_extents[kernel_roles[0]]
        channels_per_group = input_features // groups
        input_spatial = tuple(lhs_extents[mode] for mode in lhs_roles[2:])
        kernel_spatial = tuple(kernel_extents[mode] for mode in kernel_roles[2:])
        output_extents = (batch_size, output_features, *output_spatial)

        lhs_arithmetic = gradient_arithmetic(lhs)
        kernel_arithmetic = gradient_arithmetic(kernel)
        lhs_values = [0.0] * lhs.size()
        kernel_values = [0.0] * kernel.size()
        with executing(lhs_arithmetic):
            for output_ordinal in range(gradient.size()):
                output_coordinates = _logical_coordinates(
                    output_ordinal, output_extents
                )
                batch, output_feature = output_coordinates[:2]
                output_position = output_coordinates[2:]
                group = output_feature // (output_features // groups)
                incoming = lhs_arithmetic.convert(gradient[output_ordinal])
                for contraction_ordinal in range(
                    channels_per_group * _product(kernel_spatial)
                ):
                    contraction_coordinates = _logical_coordinates(
                        contraction_ordinal, (channels_per_group, *kernel_spatial)
                    )
                    channel, kernel_position = (
                        contraction_coordinates[0],
                        contraction_coordinates[1:],
                    )
                    input_position: list[int] = []
                    valid = True
                    for dimension, kernel_coordinate in enumerate(kernel_position):
                        dilated = (
                            output_position[dimension] * strides[dimension]
                            + kernel_coordinate * kernel_dilation[dimension]
                            - padding[dimension][0]
                        )
                        if dilated < 0 or dilated % lhs_dilation[dimension] != 0:
                            valid = False
                            break
                        coordinate = dilated // lhs_dilation[dimension]
                        if coordinate >= input_spatial[dimension]:
                            valid = False
                            break
                        input_position.append(coordinate)

                    kernel_ordinal = _role_ordinal(
                        kernel_roles,
                        (output_feature, channel, *kernel_position),
                        kernel,
                    )
                    kernel_value = kernel_arithmetic.convert(kernel[kernel_ordinal])
                    kernel_term = incoming * kernel_value
                    lhs_value = lhs_arithmetic.convert(0.0)
                    if valid:
                        lhs_ordinal = _role_ordinal(
                            lhs_roles,
                            (
                                batch,
                                group * channels_per_group + channel,
                                *input_position,
                            ),
                            lhs,
                        )
                        lhs_value = lhs_arithmetic.convert(lhs[lhs_ordinal])
                        lhs_values[lhs_ordinal] = binary32(
                            lhs_arithmetic.convert(lhs_values[lhs_ordinal])
                            + kernel_term
                        )
                    kernel_term = incoming * lhs_value
                    kernel_values[kernel_ordinal] = binary32(
                        kernel_arithmetic.convert(kernel_values[kernel_ordinal])
                        + kernel_term
                    )

        return (
            _detached_tensor_like(lhs, lhs_values, lhs_arithmetic.result_dtype),
            _detached_tensor_like(
                kernel, kernel_values, kernel_arithmetic.result_dtype
            ),
        )


def _product(values: Sequence[int]) -> int:
    """Multiply positive extents without importing a reduction helper."""
    result = 1
    for value in values:
        result *= value
    return result
