"""Generic reference implementations for masked selection and clamping.

The public dispatch and capability tables are assembled by the integration
layer.  This module owns the operation contracts themselves: simultaneous
structural broadcasting, central-plan validation, selected-branch reads, and
the exact ordered ``maximum``/``minimum`` VJP used by ``clamp``.
"""

from __future__ import annotations

import math
from typing import Any

from ...layout import Shape
from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _format_shape_profile,
    _require_layout,
    _require_live_tensor,
    _tensor_with_layout_like,
)
from .execution import (
    arithmetic_for_plan,
    executing,
    gradient_arithmetic,
    resolve_operation_plan,
)
from .numerics import (
    binary32,
    float32_errstate,
    float32_scalar,
    safe_maximum,
    safe_minimum,
)

__all__ = ["GenericClampOperation", "GenericSelectOperation"]


def _is_tensor(value: Any) -> bool:
    from ...tensor import Tensor

    return isinstance(value, Tensor)


def _require_dtype(value: Any, name: str, dtype: DType) -> Any:
    tensor = _require_live_tensor(value, name)
    if tensor.dtype() is not dtype:
        raise TypeError(f"{name} must have dtype DType.{dtype.name}")
    return tensor


def _require_same_carrier_class(tensors: tuple[Any, ...]) -> None:
    """Require all tensor operands to use one exact carrier implementation."""
    classes = {type(tensor.carrier) for tensor in tensors}
    if len(classes) != 1:
        raise ValueError("ternary tensor operands must use one exact carrier class")


def _common_level(levels: tuple[Any, ...], path: tuple[int, ...]) -> list[Any]:
    """Merge one shape-tree level without choosing a pairwise alignment order."""
    merged: list[Any] = []
    for index, extents in enumerate(zip(*levels, strict=True)):
        leaf_path = (*path, index)
        if isinstance(extents[0], int):
            non_singletons = {extent for extent in extents if extent != 1}
            if len(non_singletons) > 1:
                rendered = ".".join(str(part) for part in leaf_path)
                raise ValueError(
                    "Tensor extents are not broadcast-compatible at leaf "
                    f"{rendered}: extents={tuple(extents)}"
                )
            merged.append(next(iter(non_singletons), 1))
        else:
            merged.append(_common_level(tuple(extent for extent in extents), leaf_path))
    return merged


def _align_ternary_operands(*tensors: Any) -> tuple[tuple[Any, ...], Shape]:
    """Broadcast all operands against one common shape in one structural pass."""
    if not tensors:
        raise ValueError("at least one tensor is required for alignment")
    profile = tensors[0].layout.profile
    if any(tensor.layout.profile != profile for tensor in tensors[1:]):
        rendered = tuple(
            _format_shape_profile(t.layout.shape.top_level) for t in tensors
        )
        raise ValueError(
            "Tensor shape profiles are not congruent for ternary broadcasting: "
            f"profiles={rendered}"
        )
    shape = Shape(
        _common_level(
            tuple(tensor.layout.shape.top_level for tensor in tensors),
            (),
        )
    )
    aligned: list[Any] = []
    for tensor in tensors:
        if tensor.layout.shape == shape:
            aligned.append(tensor)
        else:
            aligned.append(
                tensor.carrier.dispatch_op("broadcast_to").forward(tensor, shape)
            )
    return tuple(aligned), shape


def _require_select_plan(condition: Any, on_true: Any, on_false: Any) -> None:
    """Resolve and gate the exact central select plan before result allocation."""
    plan = resolve_operation_plan(
        "select", condition.dtype(), on_true.dtype(), on_false.dtype()
    )
    arithmetic_for_plan(plan, type(condition.carrier))


def _require_clamp_plan(tensor: Any, lower: Any, upper: Any) -> None:
    """Resolve and gate the exact central clamp overload before allocation."""
    lower_operand = lower.dtype() if _is_tensor(lower) else lower
    upper_operand = upper.dtype() if _is_tensor(upper) else upper
    plan = resolve_operation_plan("clamp", tensor.dtype(), lower_operand, upper_operand)
    arithmetic_for_plan(plan, type(tensor.carrier))


def _extrema_multiplier(lhs: Any, rhs: Any, *, maximum: bool, lhs_side: bool) -> float:
    """Return the decided two-input maximum/minimum VJP multiplier."""
    lhs_value = float(lhs)
    rhs_value = float(rhs)
    if math.isnan(lhs_value) or math.isnan(rhs_value):
        return math.nan
    if lhs_value == rhs_value and math.isfinite(lhs_value):
        return 0.5
    selected_lhs = lhs_value > rhs_value if maximum else lhs_value < rhs_value
    return 1.0 if selected_lhs is lhs_side else 0.0


class GenericSelectOperation(Operation):
    """Generic Float32 masked selection with a Bool condition."""

    def _forward(self, condition: Any, on_true: Any, on_false: Any) -> Any:
        condition = _require_dtype(condition, "condition", DType.Bool)
        on_true = _require_dtype(on_true, "on_true", DType.Float32)
        on_false = _require_dtype(on_false, "on_false", DType.Float32)
        _require_same_carrier_class((condition, on_true, on_false))
        _require_select_plan(condition, on_true, on_false)
        (condition, on_true, on_false), shape = _align_ternary_operands(
            condition, on_true, on_false
        )

        # Store the aligned views so their broadcast nodes own reductions back
        # to each original value operand.  The Bool input is intentionally kept
        # in position zero; its VJP is always ``None``.
        self.store_inputs(condition, on_true, on_false)
        output_layout = _canonical_layout_for_shape(shape)
        self.ctx["output_layout"] = output_layout
        with float32_errstate():
            values = [
                binary32(on_true[index] if condition[index] else on_false[index])
                for index in range(shape.logical_size)
            ]
        return _tensor_with_layout_like(on_true, output_layout, values, DType.Float32)

    def backward(self, gradient: Any) -> tuple[None, Any, Any]:
        condition, on_true, on_false = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        arithmetic = gradient_arithmetic(on_true)
        with executing(arithmetic):
            true_values = []
            false_values = []
            for index in range(gradient.size()):
                incoming = arithmetic.convert(gradient[index])
                if condition[index]:
                    true_values.append(arithmetic.store(incoming))
                    false_values.append(arithmetic.store(float32_scalar(0.0)))
                else:
                    true_values.append(arithmetic.store(float32_scalar(0.0)))
                    false_values.append(arithmetic.store(incoming))
        return (
            None,
            _detached_tensor_like(on_true, true_values, arithmetic.result_dtype),
            _detached_tensor_like(on_false, false_values, arithmetic.result_dtype),
        )


class GenericClampOperation(Operation):
    """Generic Float32 clamp with ordered maximum/minimum semantics."""

    def _forward(self, tensor: Any, lower: Any, upper: Any) -> Any:
        tensor = _require_dtype(tensor, "tensor", DType.Float32)
        lower_tensor = (
            _require_dtype(lower, "lower", DType.Float32) if _is_tensor(lower) else None
        )
        upper_tensor = (
            _require_dtype(upper, "upper", DType.Float32) if _is_tensor(upper) else None
        )
        tensor_operands = tuple(
            operand
            for operand in (tensor, lower_tensor, upper_tensor)
            if operand is not None
        )
        _require_same_carrier_class(tensor_operands)
        _require_clamp_plan(tensor, lower, upper)

        all_tensors = tuple(
            operand
            for operand in (tensor, lower_tensor, upper_tensor)
            if operand is not None
        )
        aligned, shape = _align_ternary_operands(*all_tensors)
        aligned_index = 0
        tensor = aligned[aligned_index]
        aligned_index += 1
        if lower_tensor is not None:
            lower_tensor = aligned[aligned_index]
            aligned_index += 1
        if upper_tensor is not None:
            upper_tensor = aligned[aligned_index]

        lower_value = binary32(lower) if lower_tensor is None else None
        upper_value = binary32(upper) if upper_tensor is None else None
        self.ctx["output_layout"] = _canonical_layout_for_shape(shape)
        self.ctx["lower_is_tensor"] = lower_tensor is not None
        self.ctx["upper_is_tensor"] = upper_tensor is not None
        self.ctx["lower_value"] = lower_value
        self.ctx["upper_value"] = upper_value

        arithmetic = arithmetic_for_plan(
            resolve_operation_plan(
                "clamp",
                tensor.carrier.dtype(),
                lower_tensor.carrier.dtype() if lower_tensor is not None else lower,
                upper_tensor.carrier.dtype() if upper_tensor is not None else upper,
            ),
            type(tensor.carrier),
        )
        middle_values: list[Any] = []
        values: list[Any] = []
        with executing(arithmetic):
            for index in range(shape.logical_size):
                data_value = arithmetic.convert(tensor[index])
                low = arithmetic.convert(
                    lower_tensor[index] if lower_tensor is not None else lower_value
                )
                high = arithmetic.convert(
                    upper_tensor[index] if upper_tensor is not None else upper_value
                )
                middle = safe_maximum(data_value, low)
                middle_values.append(binary32(middle))
                values.append(binary32(safe_minimum(middle, high)))
        self.ctx["middle_values"] = middle_values
        stored_tensors = [tensor]
        if lower_tensor is not None:
            stored_tensors.append(lower_tensor)
        if upper_tensor is not None:
            stored_tensors.append(upper_tensor)
        self.store_inputs(*stored_tensors)
        return _tensor_with_layout_like(
            tensor, self.ctx["output_layout"], values, DType.Float32
        )

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        inputs = self.inputs()
        tensor = inputs[0]
        cursor = 1
        lower_tensor = inputs[cursor] if self.ctx["lower_is_tensor"] else None
        cursor += int(lower_tensor is not None)
        upper_tensor = inputs[cursor] if self.ctx["upper_is_tensor"] else None
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        lower_fixed = self.ctx["lower_value"]
        upper_fixed = self.ctx["upper_value"]
        arithmetic = gradient_arithmetic(tensor)
        tensor_values: list[Any] = []
        lower_values: list[Any] = []
        upper_values: list[Any] = []
        with executing(arithmetic):
            for index in range(gradient.size()):
                data_value = arithmetic.convert(tensor[index])
                low = arithmetic.convert(
                    lower_tensor[index] if lower_tensor is not None else lower_fixed
                )
                high = arithmetic.convert(
                    upper_tensor[index] if upper_tensor is not None else upper_fixed
                )
                middle = self.ctx["middle_values"][index]
                incoming = arithmetic.convert(gradient[index])
                middle_gradient = incoming * _extrema_multiplier(
                    middle, high, maximum=False, lhs_side=True
                )
                upper_gradient = incoming * _extrema_multiplier(
                    middle, high, maximum=False, lhs_side=False
                )
                tensor_values.append(
                    arithmetic.store(
                        middle_gradient
                        * _extrema_multiplier(
                            data_value, low, maximum=True, lhs_side=True
                        )
                    )
                )
                lower_values.append(
                    arithmetic.store(
                        middle_gradient
                        * _extrema_multiplier(
                            data_value, low, maximum=True, lhs_side=False
                        )
                    )
                )
                upper_values.append(arithmetic.store(upper_gradient))

        gradients = [
            _detached_tensor_like(tensor, tensor_values, arithmetic.result_dtype)
        ]
        if lower_tensor is not None:
            gradients.append(
                _detached_tensor_like(
                    lower_tensor, lower_values, arithmetic.result_dtype
                )
            )
        if upper_tensor is not None:
            gradients.append(
                _detached_tensor_like(
                    upper_tensor, upper_values, arithmetic.result_dtype
                )
            )
        return tuple(gradients)
