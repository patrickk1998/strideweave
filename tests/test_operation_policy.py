"""Conformance fixtures for the backend-independent SimpleDType planner.

The expected plans below are explicit values, not results recomputed from the
resolver, so a policy change has to be re-baselined deliberately and cannot be
drifted into. Coverage is enumerated from the operation registry, so an
operation added without a fixture fails ``test_every_registered_operation_has_a
_fixture`` rather than passing silently.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from strideweave import CompoundDType, DType, SimpleDType
from strideweave.carriers.operation_policy import (
    INT32_MAX,
    INT32_MIN,
    POW_INTEGER_EXPONENT_MAX,
    SUPPORTED_TENSOR_DTYPES,
    Accumulation,
    Arithmetic,
    OperandRole,
    registered_operations,
    resolve_operation_plan,
)

F32 = DType.Float32
I32 = DType.Int32

BINARY32 = Arithmetic.BINARY32
CHECKED = Arithmetic.INT32_EXACT_CHECKED
EXACT = Arithmetic.INT32_EXACT
SEQUENTIAL = Accumulation.SEQUENTIAL_BINARY32
EXACT_SUM = Accumulation.EXACT_INTEGER

# One representative weak scalar per normalized kind, plus the bool case the
# policy deliberately normalizes as a weak float.
WEAK_INTEGER = 3
WEAK_FLOAT = 0.5
WEAK_BOOL = True

# Expected plans, keyed by (operation, operands). Each value is
# (operand conversions, compute, accumulation, output).
Expected = tuple[tuple[SimpleDType, ...], Arithmetic, Accumulation | None, SimpleDType]

_BINARY_ELEMENTWISE: dict[tuple[Any, ...], Expected] = {
    (F32, F32): ((F32, F32), BINARY32, None, F32),
    (F32, I32): ((F32, F32), BINARY32, None, F32),
    (I32, F32): ((F32, F32), BINARY32, None, F32),
    (I32, I32): ((I32, I32), CHECKED, None, I32),
}

_DIV: dict[tuple[Any, ...], Expected] = {
    (F32, F32): ((F32, F32), BINARY32, None, F32),
    (F32, I32): ((F32, F32), BINARY32, None, F32),
    (I32, F32): ((F32, F32), BINARY32, None, F32),
    (I32, I32): ((F32, F32), BINARY32, None, F32),
}

_MUL: dict[tuple[Any, ...], Expected] = {
    (F32, WEAK_INTEGER): ((F32, F32), BINARY32, None, F32),
    (F32, WEAK_FLOAT): ((F32, F32), BINARY32, None, F32),
    (F32, WEAK_BOOL): ((F32, F32), BINARY32, None, F32),
    (I32, WEAK_INTEGER): ((I32, I32), CHECKED, None, I32),
    (I32, WEAK_FLOAT): ((F32, F32), BINARY32, None, F32),
    (I32, WEAK_BOOL): ((F32, F32), BINARY32, None, F32),
}

_POW: dict[tuple[Any, ...], Expected] = {
    (F32, WEAK_INTEGER): ((F32, F32), BINARY32, None, F32),
    (F32, WEAK_FLOAT): ((F32, F32), BINARY32, None, F32),
    (F32, WEAK_BOOL): ((F32, F32), BINARY32, None, F32),
    (F32, -1): ((F32, F32), BINARY32, None, F32),
    (I32, WEAK_INTEGER): ((I32, I32), CHECKED, None, I32),
    (I32, 0): ((I32, I32), CHECKED, None, I32),
    (I32, POW_INTEGER_EXPONENT_MAX): ((I32, I32), CHECKED, None, I32),
    (I32, -1): ((F32, F32), BINARY32, None, F32),
    (I32, POW_INTEGER_EXPONENT_MAX + 1): ((F32, F32), BINARY32, None, F32),
    (I32, WEAK_FLOAT): ((F32, F32), BINARY32, None, F32),
    (I32, WEAK_BOOL): ((F32, F32), BINARY32, None, F32),
}

_RELU: dict[tuple[Any, ...], Expected] = {
    (F32,): ((F32,), BINARY32, None, F32),
    (I32,): ((I32,), EXACT, None, I32),
}

_REDUCE: dict[tuple[Any, ...], Expected] = {
    (F32,): ((F32,), BINARY32, SEQUENTIAL, F32),
    (I32,): ((I32,), CHECKED, EXACT_SUM, I32),
}

_MATMUL: dict[tuple[Any, ...], Expected] = {
    (F32, F32): ((F32, F32), BINARY32, SEQUENTIAL, F32),
    (F32, I32): ((F32, F32), BINARY32, SEQUENTIAL, F32),
    (I32, F32): ((F32, F32), BINARY32, SEQUENTIAL, F32),
    (I32, I32): ((I32, I32), EXACT, EXACT_SUM, I32),
}

_FLOATING_ACTIVATION: dict[tuple[Any, ...], Expected] = {
    (F32,): ((F32,), BINARY32, None, F32),
    (I32,): ((F32,), BINARY32, None, F32),
}

EXPECTED_PLANS: dict[str, dict[tuple[Any, ...], Expected]] = {
    "add": _BINARY_ELEMENTWISE,
    "sub": _BINARY_ELEMENTWISE,
    "elementwise_mul": _BINARY_ELEMENTWISE,
    "div": _DIV,
    "mul": _MUL,
    "pow": _POW,
    "relu": _RELU,
    "reduce": _REDUCE,
    "matmul": _MATMUL,
    "elu": _FLOATING_ACTIVATION,
    "exp": _FLOATING_ACTIVATION,
    "gelu": _FLOATING_ACTIVATION,
    "leaky_relu": _FLOATING_ACTIVATION,
    "sigmoid": _FLOATING_ACTIVATION,
    "silu": _FLOATING_ACTIVATION,
    "softplus": _FLOATING_ACTIVATION,
    "tanh": _FLOATING_ACTIVATION,
}


def _fixture_cases() -> list[tuple[str, tuple[Any, ...], Expected]]:
    return [
        (operation, operands, expected)
        for operation, entries in EXPECTED_PLANS.items()
        for operands, expected in entries.items()
    ]


@pytest.mark.parametrize(("operation", "operands", "expected"), _fixture_cases())
def test_resolved_plans_match_their_expected_fixtures(
    operation: str, operands: tuple[Any, ...], expected: Expected
) -> None:
    conversions, compute, accumulation, output = expected

    plan = resolve_operation_plan(operation, *operands)

    assert plan.operation == operation
    assert tuple(operand.convert_to for operand in plan.operands) == conversions
    assert plan.compute is compute
    assert plan.accumulation is accumulation
    assert plan.output is output


def test_every_registered_operation_has_a_fixture() -> None:
    registered = {spec.name for spec in registered_operations()}

    assert registered == set(EXPECTED_PLANS)


def test_every_supported_tensor_dtype_combination_is_planned() -> None:
    for spec in registered_operations():
        tensor_positions = [
            index for index, role in enumerate(spec.roles) if role is OperandRole.TENSOR
        ]
        combinations = itertools.product(
            SUPPORTED_TENSOR_DTYPES, repeat=len(tensor_positions)
        )
        for dtypes in combinations:
            operands: list[Any] = list(dtypes)
            if spec.roles[-1] is OperandRole.WEAK_SCALAR:
                operands.append(WEAK_INTEGER)

            plan = resolve_operation_plan(spec.name, *operands)

            assert plan.output in SUPPORTED_TENSOR_DTYPES


def test_a_plan_carries_no_field_derivable_from_another() -> None:
    # Autograd eligibility was once a plan field. It was exactly
    # `output is Float32`, and nothing consumed it — the tensor layer decides
    # differentiability from dtype for every tensor, plan-produced or not — so
    # it was removed rather than left to drift from the rule it duplicated.
    plan = resolve_operation_plan("add", DType.Float32, DType.Float32)

    assert not hasattr(plan, "differentiable")


def test_weak_scalar_operands_carry_no_dtype_of_their_own() -> None:
    plan = resolve_operation_plan("mul", DType.Int32, 2)

    tensor_operand, scalar_operand = plan.operands
    assert tensor_operand.role is OperandRole.TENSOR
    assert tensor_operand.dtype is DType.Int32
    assert scalar_operand.role is OperandRole.WEAK_SCALAR
    assert scalar_operand.dtype is None
    assert scalar_operand.convert_to is DType.Int32


def test_plans_are_immutable_and_hashable() -> None:
    plan = resolve_operation_plan("add", DType.Float32, DType.Float32)

    with pytest.raises(AttributeError):
        plan.output = DType.Int32  # type: ignore[misc]
    assert hash(plan) == hash(resolve_operation_plan("add", F32, F32))


def test_representation_preserving_operations_are_not_registered() -> None:
    registered = {spec.name for spec in registered_operations()}

    assert registered.isdisjoint({"view", "permute", "rearrange", "move"})


def test_an_unregistered_operation_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="no dtype plan is defined"):
        resolve_operation_plan("floor_div", DType.Int32, DType.Int32)


def test_a_wrong_operand_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="takes 2 operands"):
        resolve_operation_plan("add", DType.Float32)


def test_a_non_dtype_tensor_operand_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a DType"):
        resolve_operation_plan("add", "Float32", DType.Float32)


@pytest.mark.parametrize("category", [DType.Any, DType.Floating])
def test_legacy_opaque_categories_are_not_simple_promotion(category: Any) -> None:
    with pytest.raises(TypeError, match="legacy opaque storage category"):
        resolve_operation_plan("add", category, DType.Float32)


def test_an_abstract_category_is_rejected_as_a_relationship() -> None:
    with pytest.raises(TypeError, match="abstract category"):
        resolve_operation_plan("add", DType.Integer, DType.Float32)


def test_a_compound_operand_reports_the_deferred_capability() -> None:
    with pytest.raises(NotImplementedError, match="compound dtype 'MXFP4'"):
        resolve_operation_plan("add", DType.MXFP4, DType.Float32)


def test_a_registered_but_unimplemented_simple_dtype_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="no backend implements"):
        resolve_operation_plan("add", DType.Int8, DType.Int32)


def test_an_unimplemented_dtype_is_never_silently_widened() -> None:
    with pytest.raises(NotImplementedError, match="no backend implements"):
        resolve_operation_plan("exp", DType.E4M3)


@pytest.mark.parametrize("scalar", ["2", None, complex(1, 1)])
def test_a_non_real_weak_scalar_is_rejected(scalar: Any) -> None:
    with pytest.raises(TypeError, match="must be a real Python number"):
        resolve_operation_plan("mul", DType.Float32, scalar)


@pytest.mark.parametrize("scalar", [INT32_MAX + 1, INT32_MIN - 1])
def test_an_out_of_range_weak_integer_is_rejected_at_resolution(scalar: int) -> None:
    with pytest.raises(OverflowError, match="out of int32 range"):
        resolve_operation_plan("mul", DType.Int32, scalar)


def test_an_out_of_range_weak_integer_is_fine_for_a_floating_plan() -> None:
    plan = resolve_operation_plan("mul", DType.Float32, INT32_MAX + 1)

    assert plan.output is DType.Float32


def test_a_bool_scalar_selects_the_floating_plan() -> None:
    # Provisional policy: `True` is a weak float, so an Int32 tensor times a
    # bool yields Float32 rather than Int32.
    plan = resolve_operation_plan("mul", DType.Int32, True)

    assert plan.output is DType.Float32


def test_integer_pow_is_bounded_by_central_policy_not_a_backend_limit() -> None:
    inside = resolve_operation_plan("pow", DType.Int32, POW_INTEGER_EXPONENT_MAX)
    outside = resolve_operation_plan("pow", DType.Int32, POW_INTEGER_EXPONENT_MAX + 1)

    assert inside.output is DType.Int32
    assert outside.output is DType.Float32


def test_a_huge_integer_exponent_does_not_overflow_resolution() -> None:
    plan = resolve_operation_plan("pow", DType.Int32, 2**70)

    assert plan.output is DType.Float32


def test_integer_matmul_does_not_check_individual_products() -> None:
    plan = resolve_operation_plan("matmul", DType.Int32, DType.Int32)

    assert plan.compute is Arithmetic.INT32_EXACT
    assert plan.accumulation is Accumulation.EXACT_INTEGER


def test_integer_reduce_checks_only_the_final_narrowing() -> None:
    plan = resolve_operation_plan("reduce", DType.Int32)

    assert plan.compute is Arithmetic.INT32_EXACT_CHECKED
    assert plan.accumulation is Accumulation.EXACT_INTEGER


def test_operand_dtypes_are_matched_by_identity_not_equality() -> None:
    class Lookalike:
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(DType.Float32)

    with pytest.raises(TypeError, match="must be a DType"):
        resolve_operation_plan("add", Lookalike(), DType.Float32)


def test_a_compound_subclass_is_rejected_like_a_built_in_compound() -> None:
    class Planar(CompoundDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str, *, planes: tuple[SimpleDType, ...]) -> None:
            super().__init__(name, supertype=DType.Any, simple_types=planes)

    planar = Planar("PolicyPlanar32", planes=(DType.Float32, DType.Int32))

    with pytest.raises(NotImplementedError, match="compound dtype 'PolicyPlanar32'"):
        resolve_operation_plan("relu", planar)
