"""Central semantic definitions for the bounded v0 direct-operation set."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .carriers.dtype import DType, SimpleDType
from .carriers.generic.execution import executing, gradient_arithmetic
from .carriers.operation_helpers import (
    _canonical_layout_for_shape,
    _canonical_layout_from_modes,
    _detached_tensor_like,
    _mode_logical_size,
    _mode_shape,
)
from .carriers.operation_policy import (
    OperationPlan,
    operation_execution_options,
    resolve_operation_plan,
)
from .layout import Shape
from .operation_definition import (
    BoundOperationCall,
    OperandKind,
    OperandSpec,
    OperationDefinition,
    OperationSchema,
    OptionSpec,
    ResolvedInvocation,
    ResultSpec,
    VJPContext,
    _install_builtin,
)
from .tensor import Tensor


def _require_live(tensor: Tensor, name: str) -> Tensor:
    if tensor.carrier.is_released():
        raise RuntimeError(f"{name} carrier is released")
    return tensor


def _format_shape_profile(shape_level: object) -> str:
    parts = (
        "leaf" if isinstance(child, int) else _format_shape_profile(child)
        for child in cast(Any, shape_level)
    )
    return f"({', '.join(parts)})"


def _broadcast_shape(lhs: Tensor, rhs: Tensor) -> Shape:
    if lhs.layout.profile != rhs.layout.profile:
        lhs_rendered = _format_shape_profile(lhs.layout.shape.top_level)
        rhs_rendered = _format_shape_profile(rhs.layout.shape.top_level)
        raise ValueError(
            "Tensor shape profiles are not congruent: "
            f"lhs={lhs_rendered}, rhs={rhs_rendered}. "
            "Insert singleton modes with rearrange so both profiles match."
        )
    rendered_profile = _format_shape_profile(lhs.layout.shape.top_level)

    def common_level(
        lhs_level: Any, rhs_level: Any, path: tuple[int, ...]
    ) -> list[Any]:
        common: list[Any] = []
        for index, (lhs_extent, rhs_extent) in enumerate(
            zip(lhs_level, rhs_level, strict=True)
        ):
            leaf_path = (*path, index)
            if isinstance(lhs_extent, int):
                if lhs_extent == rhs_extent or rhs_extent == 1:
                    common.append(lhs_extent)
                elif lhs_extent == 1:
                    common.append(rhs_extent)
                else:
                    position = ".".join(str(component) for component in leaf_path)
                    raise ValueError(
                        "Tensor extents are not broadcast-compatible at leaf "
                        f"{position} within profile {rendered_profile}: "
                        f"lhs={lhs_extent}, rhs={rhs_extent}"
                    )
            else:
                common.append(common_level(lhs_extent, rhs_extent, leaf_path))
        return common

    return Shape(
        common_level(lhs.layout.shape.top_level, rhs.layout.shape.top_level, ())
    )


def _tensor(call: BoundOperationCall, index: int, name: str) -> Tensor:
    value = call.operands[index]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return _require_live(value, name)


def _accumulator(call: BoundOperationCall) -> object | None:
    return call.options.get("accumulator_dtype")


def _binary_resolver(name: str) -> Callable[[BoundOperationCall], ResolvedInvocation]:
    def resolve(call: BoundOperationCall) -> ResolvedInvocation:
        lhs = _tensor(call, 0, "lhs")
        rhs = _tensor(call, 1, "rhs")
        result_shape = _broadcast_shape(lhs, rhs)
        plan = resolve_operation_plan(name, lhs.dtype(), rhs.dtype())
        return ResolvedInvocation(
            call,
            plan,
            (
                ResultSpec(
                    plan.output, result_shape, _canonical_layout_for_shape(result_shape)
                ),
            ),
            saved_operands=(0, 1) if name == "elementwise_mul" else (),
        )

    return resolve


def _resolve_mul(call: BoundOperationCall) -> ResolvedInvocation:
    lhs, rhs = call.operands
    lhs_tensor = isinstance(lhs, Tensor)
    rhs_tensor = isinstance(rhs, Tensor)
    if lhs_tensor and rhs_tensor:
        left = _require_live(lhs, "lhs")
        right = _require_live(rhs, "rhs")
        result_shape = _broadcast_shape(left, right)
        plan = resolve_operation_plan("mul", left.dtype(), right.dtype())
        saved = (0, 1)
    elif lhs_tensor:
        left = _require_live(lhs, "lhs")
        result_shape = left.layout.shape
        plan = resolve_operation_plan("mul", left.dtype(), rhs)
        saved = ()
    elif rhs_tensor:
        right = _require_live(rhs, "rhs")
        result_shape = right.layout.shape
        forward = resolve_operation_plan("mul", right.dtype(), lhs)
        plan = OperationPlan(
            operation=forward.operation,
            operands=(forward.operands[1], forward.operands[0]),
            compute=forward.compute,
            accumulation=forward.accumulation,
            accumulator_dtype=forward.accumulator_dtype,
            output=forward.output,
        )
        saved = ()
    else:
        raise TypeError("mul requires at least one Tensor operand")
    return ResolvedInvocation(
        call,
        plan,
        (
            ResultSpec(
                plan.output, result_shape, _canonical_layout_for_shape(result_shape)
            ),
        ),
        saved_operands=saved,
    )


def _resolve_relu(call: BoundOperationCall) -> ResolvedInvocation:
    tensor = _tensor(call, 0, "tensor")
    plan = resolve_operation_plan("relu", tensor.dtype())
    shape = tensor.layout.shape
    return ResolvedInvocation(
        call,
        plan,
        (ResultSpec(plan.output, shape, _canonical_layout_for_shape(shape)),),
        saved_operands=(0,),
    )


def _require_two_mode(tensor: Tensor, name: str) -> Tensor:
    tensor = _require_live(tensor, name)
    if len(tensor.layout) != 2:
        raise ValueError(f"{name} must have a two-mode layout")
    return tensor


def _resolve_reduce_sum(call: BoundOperationCall) -> ResolvedInvocation:
    tensor = _require_two_mode(_tensor(call, 0, "tensor"), "tensor")
    if _mode_logical_size(tensor.layout, 1) == 0:
        raise ValueError("Reduction fibers must not be empty")
    output_shape = Shape(_mode_shape(tensor.layout, 0))
    plan = resolve_operation_plan(
        "reduce_sum", tensor.dtype(), accumulator_dtype=_accumulator(call)
    )
    return ResolvedInvocation(
        call,
        plan,
        (
            ResultSpec(
                plan.output, output_shape, _canonical_layout_for_shape(output_shape)
            ),
        ),
    )


def _resolve_matmul(call: BoundOperationCall) -> ResolvedInvocation:
    lhs = _require_two_mode(_tensor(call, 0, "lhs"), "lhs")
    rhs = _require_two_mode(_tensor(call, 1, "rhs"), "rhs")
    if _mode_logical_size(lhs.layout, 1) != _mode_logical_size(rhs.layout, 1):
        raise ValueError("Matmul inner dimensions must match")
    output_layout = _canonical_layout_from_modes(
        _mode_shape(lhs.layout, 0), _mode_shape(rhs.layout, 0)
    )
    plan = resolve_operation_plan(
        "matmul", lhs.dtype(), rhs.dtype(), accumulator_dtype=_accumulator(call)
    )
    return ResolvedInvocation(
        call,
        plan,
        (ResultSpec(plan.output, output_layout.shape, output_layout),),
        saved_operands=(0, 1),
    )


def _reference_binary(invocation: ResolvedInvocation) -> Tensor:
    lhs, rhs = cast(tuple[Tensor, Tensor], invocation.operands)
    return lhs.carrier.dispatch_op(invocation.name).forward(lhs, rhs)


def _reference_mul(invocation: ResolvedInvocation) -> Tensor:
    from .functional.api import mul

    lhs, rhs = invocation.operands
    return cast(Tensor, mul(lhs, rhs))


def _reference_relu(invocation: ResolvedInvocation) -> Tensor:
    (tensor,) = cast(tuple[Tensor], invocation.operands)
    return tensor.carrier.dispatch_op("relu").forward(tensor)


def _execution_options(invocation: ResolvedInvocation) -> object | None:
    accumulator_dtype = invocation.options.get("accumulator_dtype")
    if accumulator_dtype is None:
        return None
    return operation_execution_options(
        invocation.name,
        accumulator_dtype=cast(SimpleDType, accumulator_dtype),
    )


def _reference_reduce_sum(invocation: ResolvedInvocation) -> Tensor:
    (tensor,) = cast(tuple[Tensor], invocation.operands)
    operation = tensor.carrier.dispatch_op("reduce_sum")
    options = _execution_options(invocation)
    if options is None:
        return operation.forward(tensor)
    return operation.forward(tensor, options=options)


def _reference_matmul(invocation: ResolvedInvocation) -> Tensor:
    lhs, rhs = cast(tuple[Tensor, Tensor], invocation.operands)
    operation = lhs.carrier.dispatch_op("matmul")
    options = _execution_options(invocation)
    if options is None:
        return operation.forward(lhs, rhs)
    return operation.forward(lhs, rhs, options=options)


def _require_one_cotangent(cotangents: tuple[Tensor | None, ...]) -> Tensor:
    if len(cotangents) != 1 or cotangents[0] is None:
        raise ValueError("the v0 operation requires one Tensor cotangent")
    return cotangents[0]


def _flatten_extents(level: Any) -> list[int]:
    extents: list[int] = []
    for child in level:
        if isinstance(child, int):
            extents.append(child)
        else:
            extents.extend(_flatten_extents(child))
    return extents


def _sum_to_operand(target: Tensor, gradient: Tensor) -> Tensor:
    """Reduce a broadcast result gradient to one original operand Shape."""

    if target.layout.profile != gradient.layout.profile:
        raise ValueError("gradient and operand Shape profiles must match")
    target_extents = _flatten_extents(target.layout.shape.top_level)
    gradient_extents = _flatten_extents(gradient.layout.shape.top_level)
    if any(
        target_extent not in (1, gradient_extent)
        for target_extent, gradient_extent in zip(
            target_extents, gradient_extents, strict=True
        )
    ):
        raise ValueError("gradient Shape is not a broadcast of the operand Shape")
    target_layout = _canonical_layout_for_shape(target.layout.shape)
    arithmetic = gradient_arithmetic(target)
    values = [arithmetic.convert(0.0) for _ in range(target.size())]
    with executing(arithmetic):
        for ordinal in range(gradient.size()):
            coordinates = gradient.layout._cache.expand_key(ordinal)
            reduced = [
                0 if extent == 1 else coordinate
                for coordinate, extent in zip(coordinates, target_extents, strict=True)
            ]
            target_index = target_layout._cache.index_expanded(reduced)
            values[target_index] = arithmetic.convert(values[target_index]) + (
                arithmetic.convert(gradient[ordinal])
            )
        values = [arithmetic.store(value) for value in values]
    return _detached_tensor_like(target, values, arithmetic.result_dtype)


def _broadcast_value(tensor: Tensor, result: Tensor, ordinal: int) -> object:
    coordinates = result.layout._cache.expand_key(ordinal)
    extents = _flatten_extents(tensor.layout.shape.top_level)
    reduced = tuple(
        0 if extent == 1 else coordinate
        for coordinate, extent in zip(coordinates, extents, strict=True)
    )
    physical_index = tensor.layout._cache.index_expanded(reduced)
    return tensor.carrier[tensor.offset + physical_index]


def _pointwise_product(
    target: Tensor,
    cotangent: Tensor,
    factor: Tensor | object,
) -> Tensor:
    """Multiply a cotangent without assuming its carrier matches the primal."""

    arithmetic = gradient_arithmetic(target)
    with executing(arithmetic):
        values = [
            arithmetic.store(
                arithmetic.convert(cotangent[index])
                * arithmetic.convert(
                    _broadcast_value(factor, cotangent, index)
                    if isinstance(factor, Tensor)
                    else factor
                )
            )
            for index in range(cotangent.size())
        ]
    return _detached_tensor_like(cotangent, values, arithmetic.result_dtype)


def _vjp_add(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    lhs, rhs = cast(tuple[Tensor, Tensor], context.invocation.operands)
    return _sum_to_operand(lhs, cotangent), _sum_to_operand(rhs, cotangent)


def _vjp_elementwise_mul(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    lhs, rhs = cast(tuple[Tensor, Tensor], context.invocation.operands)
    lhs_gradient = _pointwise_product(lhs, cotangent, rhs)
    rhs_gradient = _pointwise_product(rhs, cotangent, lhs)
    return _sum_to_operand(lhs, lhs_gradient), _sum_to_operand(rhs, rhs_gradient)


def _vjp_mul(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    lhs, rhs = context.invocation.operands
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        lhs_gradient = _pointwise_product(lhs, cotangent, rhs)
        rhs_gradient = _pointwise_product(rhs, cotangent, lhs)
        return _sum_to_operand(lhs, lhs_gradient), _sum_to_operand(rhs, rhs_gradient)
    if isinstance(lhs, Tensor):
        return (_sum_to_operand(lhs, _pointwise_product(lhs, cotangent, rhs)),)
    assert isinstance(rhs, Tensor)
    return (_sum_to_operand(rhs, _pointwise_product(rhs, cotangent, lhs)),)


def _vjp_relu(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    (tensor,) = cast(tuple[Tensor], context.invocation.operands)
    arithmetic = gradient_arithmetic(tensor)
    with executing(arithmetic):
        values = [
            arithmetic.store(
                arithmetic.convert(cotangent[index])
                * arithmetic.convert(1 if tensor[index] > 0 else 0)
            )
            for index in range(tensor.size())
        ]
    return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)


def _vjp_reduce_sum(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    (tensor,) = cast(tuple[Tensor], context.invocation.operands)
    n_size = _mode_logical_size(tensor.layout, 0)
    m_size = _mode_logical_size(tensor.layout, 1)
    arithmetic = gradient_arithmetic(tensor)
    with executing(arithmetic):
        values = [
            arithmetic.store(arithmetic.convert(cotangent[i]))
            for _j in range(m_size)
            for i in range(n_size)
        ]
    return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)


def _vjp_matmul(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    cotangent = _require_one_cotangent(cotangents)
    lhs, rhs = cast(tuple[Tensor, Tensor], context.invocation.operands)
    n_size = _mode_logical_size(lhs.layout, 0)
    k_size = _mode_logical_size(lhs.layout, 1)
    m_size = _mode_logical_size(rhs.layout, 0)
    accumulator_dtype = context.invocation.plan.accumulator_dtype or DType.Float32
    lhs_arithmetic = gradient_arithmetic(lhs, accumulator_dtype)
    rhs_arithmetic = gradient_arithmetic(rhs, accumulator_dtype)
    with executing(lhs_arithmetic):
        lhs_values = [
            lhs_arithmetic.store(
                lhs_arithmetic.total(
                    lhs_arithmetic.convert(cotangent[i, j])
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
                    rhs_arithmetic.convert(cotangent[i, j])
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


_TENSOR_PAIR = OperationSchema(
    (
        OperandSpec("lhs", OperandKind.TENSOR, differentiable=True),
        OperandSpec("rhs", OperandKind.TENSOR, differentiable=True),
    )
)
_TENSOR = OperationSchema(
    (OperandSpec("tensor", OperandKind.TENSOR, differentiable=True),)
)
_ACCUMULATING_TENSOR = OperationSchema(
    _TENSOR.operands, (OptionSpec("accumulator_dtype", None),)
)
_ACCUMULATING_PAIR = OperationSchema(
    _TENSOR_PAIR.operands, (OptionSpec("accumulator_dtype", None),)
)
_MUL = OperationSchema(
    (
        OperandSpec("lhs", OperandKind.STRUCTURAL),
        OperandSpec("rhs", OperandKind.STRUCTURAL),
    )
)


def _definition(
    name: str,
    schema: OperationSchema,
    resolve: Callable[[BoundOperationCall], ResolvedInvocation],
    reference: Callable[[ResolvedInvocation], Tensor],
    vjp: Callable[[VJPContext, tuple[Tensor | None, ...]], tuple[Tensor | None, ...]],
) -> OperationDefinition:
    return OperationDefinition(name, schema, resolve, reference, vjp)


BUILTIN_OPERATION_DEFINITIONS = (
    _definition(
        "add", _TENSOR_PAIR, _binary_resolver("add"), _reference_binary, _vjp_add
    ),
    _definition(
        "elementwise_mul",
        _TENSOR_PAIR,
        _binary_resolver("elementwise_mul"),
        _reference_binary,
        _vjp_elementwise_mul,
    ),
    _definition("mul", _MUL, _resolve_mul, _reference_mul, _vjp_mul),
    _definition("relu", _TENSOR, _resolve_relu, _reference_relu, _vjp_relu),
    _definition(
        "reduce_sum",
        _ACCUMULATING_TENSOR,
        _resolve_reduce_sum,
        _reference_reduce_sum,
        _vjp_reduce_sum,
    ),
    _definition(
        "matmul",
        _ACCUMULATING_PAIR,
        _resolve_matmul,
        _reference_matmul,
        _vjp_matmul,
    ),
)

for _builtin in BUILTIN_OPERATION_DEFINITIONS:
    _install_builtin(_builtin)


__all__ = ["BUILTIN_OPERATION_DEFINITIONS"]
