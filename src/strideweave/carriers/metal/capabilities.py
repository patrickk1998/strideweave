"""Metal's exact executable operation-plan declaration.

The central dtype policy decides which plans exist.  This module makes the one
backend-local decision Metal is allowed to make: which of those resolved plans
its initial TileLang kernels execute faithfully.  The same filtered set is
sealed for introspection and required by every planned Metal forward path.
"""

from __future__ import annotations

from typing import Final

from ..dtype import DType, SimpleDType
from ..operation_capability import OperationCapability, require_capability
from ..operation_policy import (
    Accumulation,
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
    resolvable_plans,
    resolve_operation_plan,
)

__all__ = ["executable_plan_shape", "metal_capabilities"]

_FLOAT32_UNARY: Final[frozenset[str]] = frozenset(
    {
        "abs",
        "ceil",
        "cos",
        "erf",
        "exp2",
        "floor",
        "log",
        "log2",
        "neg",
        "recip",
        "relu",
        "round",
        "rsqrt",
        "sign",
        "sin",
        "sqrt",
    }
)
_CONVERTING_UNARY: Final[frozenset[str]] = frozenset(
    {
        "elu",
        "exp",
        "gelu",
        "leaky_relu",
        "sigmoid",
        "silu",
        "softplus",
        "tanh",
    }
)
_MIXED_BINARY: Final[frozenset[str]] = frozenset({"add", "elementwise_mul", "sub"})
_FLOAT32_BINARY: Final[frozenset[str]] = frozenset({"maximum", "minimum", "rem"})
_PREDICATES: Final[frozenset[str]] = frozenset({"eq", "le", "logical_not", "lt", "ne"})
_POINTWISE: Final[frozenset[str]] = frozenset(
    {
        *_FLOAT32_UNARY,
        *_CONVERTING_UNARY,
        *_MIXED_BINARY,
        *_FLOAT32_BINARY,
        *_PREDICATES,
        "clamp",
        "div",
        "mul",
        "pow",
        "select",
    }
)

_REDUCTION_ACCUMULATIONS: Final[dict[str, Accumulation]] = {
    "argmax": Accumulation.ARGMAX,
    "argmin": Accumulation.ARGMIN,
    "cumsum": Accumulation.SEQUENTIAL_BINARY32,
    "reduce_max": Accumulation.MAXIMUM,
    "reduce_min": Accumulation.MINIMUM,
    "reduce_prod": Accumulation.SEQUENTIAL_BINARY32_PRODUCT,
    "reduce_sum": Accumulation.FLOATING,
}
_REDUCTIONS: Final[frozenset[str]] = frozenset(_REDUCTION_ACCUMULATIONS)

_INDEXING_OPERANDS: Final[dict[str, tuple[SimpleDType, ...]]] = {
    "_sort_indices": (DType.Float32,),
    "_sort_values": (DType.Float32,),
    "_topk_indices": (DType.Float32,),
    "_topk_values": (DType.Float32,),
    "gather": (DType.Float32, DType.Int32),
    "scatter": (DType.Float32, DType.Int32, DType.Float32),
    "scatter_add": (DType.Float32, DType.Int32, DType.Float32),
}
_INDEXING: Final[frozenset[str]] = frozenset(_INDEXING_OPERANDS)
_CONTRACTIONS: Final[frozenset[str]] = frozenset({"conv_general", "matmul"})


def _tensor_operand(
    operand: OperandPlan,
    *,
    dtypes: tuple[SimpleDType, ...],
    convert_to: SimpleDType,
) -> bool:
    return (
        operand.role is OperandRole.TENSOR
        and any(operand.dtype is dtype for dtype in dtypes)
        and operand.convert_to is convert_to
    )


def _weak_float32_operand(operand: OperandPlan) -> bool:
    return (
        operand.role is OperandRole.WEAK_SCALAR
        and operand.dtype is None
        and operand.convert_to is DType.Float32
    )


def _binary32_pointwise_shape(plan: OperationPlan) -> bool:
    if (
        plan.operation not in _POINTWISE
        or plan.compute is not Arithmetic.BINARY32
        or plan.accumulation is not None
        or plan.accumulator_dtype is not None
    ):
        return False
    expected_output = DType.Bool if plan.operation in _PREDICATES else DType.Float32
    if plan.output is not expected_output:
        return False

    operands = plan.operands
    float_tensor = lambda operand: _tensor_operand(  # noqa: E731
        operand, dtypes=(DType.Float32,), convert_to=DType.Float32
    )
    converting_tensor = lambda operand: _tensor_operand(  # noqa: E731
        operand,
        dtypes=(DType.Float32, DType.Int32),
        convert_to=DType.Float32,
    )

    if plan.operation in _FLOAT32_UNARY or plan.operation == "logical_not":
        return len(operands) == 1 and float_tensor(operands[0])
    if plan.operation in _CONVERTING_UNARY:
        return len(operands) == 1 and converting_tensor(operands[0])
    if plan.operation in _PREDICATES:
        return len(operands) == 2 and all(float_tensor(item) for item in operands)
    if plan.operation in _FLOAT32_BINARY:
        return len(operands) == 2 and all(float_tensor(item) for item in operands)
    if plan.operation in _MIXED_BINARY:
        return (
            len(operands) == 2
            and all(converting_tensor(item) for item in operands)
            and not all(item.dtype is DType.Int32 for item in operands)
        )
    if plan.operation == "div":
        return len(operands) == 2 and all(converting_tensor(item) for item in operands)
    if plan.operation == "mul":
        if len(operands) != 2 or not converting_tensor(operands[0]):
            return False
        if operands[1].role is OperandRole.WEAK_SCALAR:
            return _weak_float32_operand(operands[1])
        return converting_tensor(operands[1]) and not all(
            item.dtype is DType.Int32 for item in operands
        )
    if plan.operation == "pow":
        if len(operands) != 2:
            return False
        lhs, rhs = operands
        return (
            (converting_tensor(lhs) and converting_tensor(rhs))
            or (converting_tensor(lhs) and _weak_float32_operand(rhs))
            or (_weak_float32_operand(lhs) and converting_tensor(rhs))
        )
    if plan.operation == "clamp":
        return (
            len(operands) == 3
            and float_tensor(operands[0])
            and all(
                float_tensor(item) or _weak_float32_operand(item)
                for item in operands[1:]
            )
        )
    if plan.operation == "select":
        return (
            len(operands) == 3
            and _tensor_operand(
                operands[0], dtypes=(DType.Bool,), convert_to=DType.Bool
            )
            and float_tensor(operands[1])
            and float_tensor(operands[2])
        )
    return False


def _reduction_shape(plan: OperationPlan) -> bool:
    if plan.operation not in _REDUCTIONS:
        return False
    expected_output = (
        DType.Int32 if plan.operation in {"argmax", "argmin"} else DType.Float32
    )
    return (
        plan.compute is Arithmetic.BINARY32
        and plan.accumulation is _REDUCTION_ACCUMULATIONS[plan.operation]
        and plan.accumulator_dtype
        is (DType.Float32 if plan.operation == "reduce_sum" else None)
        and plan.output is expected_output
        and len(plan.operands) == 1
        and _tensor_operand(
            plan.operands[0],
            dtypes=(DType.Float32,),
            convert_to=DType.Float32,
        )
    )


def _indexing_shape(plan: OperationPlan) -> bool:
    if plan.operation not in _INDEXING:
        return False
    expected_output = (
        DType.Int32
        if plan.operation in {"_sort_indices", "_topk_indices"}
        else DType.Float32
    )
    expected_accumulation = (
        Accumulation.SEQUENTIAL_BINARY32 if plan.operation == "scatter_add" else None
    )
    expected_operands = _INDEXING_OPERANDS[plan.operation]
    return (
        plan.compute is Arithmetic.BINARY32
        and plan.accumulation is expected_accumulation
        and plan.accumulator_dtype is None
        and plan.output is expected_output
        and len(plan.operands) == len(expected_operands)
        and all(
            _tensor_operand(operand, dtypes=(dtype,), convert_to=dtype)
            for operand, dtype in zip(plan.operands, expected_operands, strict=True)
        )
    )


def _contraction_shape(plan: OperationPlan) -> bool:
    if plan.operation not in _CONTRACTIONS:
        return False
    expected_accumulation = (
        Accumulation.FLOATING
        if plan.operation == "matmul"
        else Accumulation.SEQUENTIAL_BINARY32
    )
    expected_accumulator = DType.Float32 if plan.operation == "matmul" else None
    return (
        plan.compute is Arithmetic.BINARY32
        and plan.accumulation is expected_accumulation
        and plan.accumulator_dtype is expected_accumulator
        and plan.output is DType.Float32
        and len(plan.operands) == 2
        and all(
            _tensor_operand(
                operand,
                dtypes=(DType.Float32,),
                convert_to=DType.Float32,
            )
            for operand in plan.operands
        )
    )


def executable_plan_shape(plan: OperationPlan) -> bool:
    """Report whether an initial TileLang Metal kernel executes ``plan`` exactly."""
    if not isinstance(plan, OperationPlan):
        raise TypeError("plan must be an OperationPlan")
    if plan.operation in _POINTWISE:
        return _binary32_pointwise_shape(plan)
    if plan.operation in _REDUCTIONS:
        return _reduction_shape(plan)
    if plan.operation in _INDEXING:
        return _indexing_shape(plan)
    if plan.operation in _CONTRACTIONS:
        return _contraction_shape(plan)
    return False


def _require_executable_plan(plan: OperationPlan) -> OperationPlan:
    """Require ``plan`` against Metal's sealed declaration and return it."""
    from .carrier import Metal

    require_capability(Metal, plan)
    return plan


def _resolvable_metal_plans() -> tuple[OperationPlan, ...]:
    """Enumerate central plans over the dtypes Metal stores and computes with."""
    plans: dict[OperationPlan, None] = dict.fromkeys(
        resolvable_plans((DType.Float32, DType.Int32))
    )
    plans[
        resolve_operation_plan("select", DType.Bool, DType.Float32, DType.Float32)
    ] = None
    return tuple(plan for plan in plans if executable_plan_shape(plan))


def metal_capabilities() -> tuple[OperationCapability, ...]:
    """Return every exact central plan the initial Metal backend executes."""
    return tuple(
        OperationCapability.from_plan(plan) for plan in _resolvable_metal_plans()
    )
