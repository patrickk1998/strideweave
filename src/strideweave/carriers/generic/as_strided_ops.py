"""Generic reference implementation for the logical ``as_strided`` view."""

from __future__ import annotations

from typing import Any

from ...core._representation import Subtensor, TensorRepresentation
from ...layout import Layout, Shape, Stride
from ..operation_helpers import (
    Operation,
    _detached_tensor_like,
    _require_layout,
    _require_live_tensor,
)

__all__ = ["GenericAsStridedOperation"]


def _nested_stride(shape: Shape, flat_strides: list[int]) -> Stride:
    """Rebuild ``shape`` hierarchy from one stride per leaf."""
    strides = iter(flat_strides)

    def level(shape_level: Any) -> list[Any]:
        return [
            next(strides) if isinstance(element, int) else level(element)
            for element in shape_level
        ]

    return Stride(level(shape.top_level))


def _compose_layout(A: Layout, B: Layout) -> Layout:
    """Compose layouts with work bounded by their hierarchical ranks."""
    try:
        candidate = Layout.compose(A, B)
    except (TypeError, ValueError, IndexError):
        candidate = None

    if candidate is not None and candidate.shape == B.shape:
        return candidate

    flat_A, _A_recipe = Layout.flatten_layout(A)
    flat_B, _B_recipe = Layout.flatten_layout(B)
    try:
        flat_candidate = Layout.compose(flat_A, flat_B)
    except (TypeError, ValueError, IndexError):
        flat_candidate = None

    if flat_candidate is not None and flat_candidate.shape == flat_B.shape:
        return Layout(
            B.shape,
            _nested_stride(
                B.shape, [stride for _shape, stride in flat_candidate.infix()]
            ),
        )

    # B is origin-based.  If its complete address range stays within A's first
    # coalesced logical mode, A.index(k) is exactly k times that mode's stride;
    # no mixed-radix carry can occur.  This covers useful noncanonical layouts
    # that the more general CuTe composition rejects, without coordinate probes.
    coalesced_modes = Layout.coalesce(flat_A).infix()
    if coalesced_modes and B.cosize <= coalesced_modes[0][0]:
        first_stride = coalesced_modes[0][1]
        return Layout(
            B.shape,
            _nested_stride(
                B.shape,
                [stride * first_stride for _shape, stride in flat_B.infix()],
            ),
        )

    raise ValueError("as_strided composition is not representable by a Layout")


def _compose_c0_representation(tensor: Any, mapping: Layout) -> Any:
    """Compose one origin-based mapping through every c0 layout.

    ``TensorRepresentation`` stores one placement layout per storage level.
    Only level zero and the adjacent mapping out of that level have ``c_0`` as
    their domain; all deeper layouts remain in their existing coordinate
    spaces.  Rebuilding the complete representation here makes universal and
    dtype-specific validation run before the view is returned.
    """

    from ...tensor import Tensor

    representation = tensor._representation
    subtensors = tuple(
        Subtensor(
            subtensor.dtype,
            subtensor.carrier,
            subtensor.offset,
            _compose_layout(subtensor.layout, mapping)
            if level == 0
            else subtensor.layout,
        )
        for level, subtensor in enumerate(representation.subtensors)
    )
    adjacent_layouts = tuple(
        _compose_layout(adjacent, mapping) if level == 0 else adjacent
        for level, adjacent in enumerate(representation.adjacent_layouts)
    )
    return Tensor._from_representation(
        TensorRepresentation(
            representation.logical_dtype,
            subtensors,
            adjacent_layouts,
        )
    )


class GenericAsStridedOperation(Operation):
    """Create a zero-copy logical-coordinate view for Generic tensors."""

    def _forward(self, tensor: Any, shape: Shape, stride: Stride) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape")
        if not isinstance(stride, Stride):
            raise TypeError("stride must be a Stride")

        mapping = Layout(shape, stride)
        if not mapping.is_injective:
            raise ValueError("as_strided mapping must be injective")
        if mapping.cosize > tensor.size():
            raise ValueError(
                "as_strided mapping exceeds the input logical coordinate domain"
            )

        output_layout = _compose_layout(tensor.layout, mapping)
        if not output_layout.is_injective:
            raise ValueError("as_strided composed placement must be injective")

        self.ctx["mapping_layout"] = mapping
        self.ctx["output_layout"] = output_layout
        return _compose_c0_representation(tensor, mapping)

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        output_layout = self.ctx["output_layout"]
        _require_layout(gradient, output_layout)

        mapping = self.ctx["mapping_layout"]
        values: list[Any] = [0] * tensor.size()
        for output_index in range(mapping.size):
            input_index = mapping.index(output_index)
            values[input_index] = gradient[output_index]
        return (_detached_tensor_like(tensor, values),)
