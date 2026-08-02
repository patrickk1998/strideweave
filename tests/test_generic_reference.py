"""Generic as the concrete SimpleDType behavioral reference.

These tests pin what makes Generic a reference rather than merely a Python
fallback: genuine IEEE-754 binary32 for ``Float32`` in forward *and* backward,
exact ``Int32`` with checked narrowing, normalized owned storage, and the
documented boundary where legacy opaque storage keeps its own arithmetic.
"""

from __future__ import annotations

import itertools
import subprocess
import sys

import numpy
import pytest

import strideweave as sw
from strideweave import CPU, DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.capabilities import generic_capabilities
from strideweave.carriers.generic.execution import (
    arithmetic_for_plan,
    executable_plan_shape,
)
from strideweave.carriers.operation_capability import (
    UnsupportedOperationPlan,
    capabilities_for_carrier_class,
    supports_operation_plan,
)
from strideweave.carriers.operation_policy import (
    Accumulation,
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
    resolvable_plans,
    resolve_operation_plan,
)

CONCRETE_DTYPES = (DType.Float32, DType.Int32)

ONE_MODE = Layout(Shape(2), Stride(1))
SCALAR = Layout(Shape(1), Stride(1))


def generic_tensor(values, dtype, layout=ONE_MODE):
    return Tensor(Generic(values, dtype=dtype), 0, layout)


def values_of(tensor):
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor):
    assert tensor.grad is not None
    return tensor.grad


def binary32(value):
    return float(numpy.float32(value))


# --- Storage normalization -------------------------------------------------


def test_float32_storage_holds_binary32_values():
    carrier = Generic([0.1, 1e40], dtype=DType.Float32)

    assert carrier[0] == binary32(0.1)
    assert carrier[0] != 0.1
    # An out-of-range magnitude becomes an infinity rather than raising.
    assert carrier[1] == float("inf")


def test_int32_storage_rejects_non_integers_and_out_of_range_values():
    with pytest.raises(TypeError, match="must be an integer"):
        Generic([1.5], dtype=DType.Int32)

    with pytest.raises(OverflowError, match="out of int32 range"):
        Generic([2**31], dtype=DType.Int32)


def test_concrete_storage_is_owned_rather_than_aliased():
    supplied = [1.0, 2.0]
    carrier = Generic(supplied, dtype=DType.Float32)

    supplied[0] = 99.0

    assert carrier[0] == 1.0


def test_legacy_storage_keeps_its_documented_aliasing():
    supplied = [1.0, 2.0]
    carrier = Generic(supplied, dtype=DType.Floating)

    supplied[0] = 99.0

    assert carrier[0] == 99.0


def test_mutation_normalizes_and_bumps_the_version():
    carrier = Generic([0.0], dtype=DType.Float32)
    before = carrier.version

    carrier[0] = 0.1

    assert carrier[0] == binary32(0.1)
    assert carrier.version != before

    with pytest.raises(TypeError, match="must be an integer"):
        Generic([0], dtype=DType.Int32)[0] = 0.5


def test_concrete_allocation_starts_at_zero_rather_than_none():
    carrier = Generic([1.0], dtype=DType.Float32).allocate_like(3)

    assert [carrier[i] for i in range(3)] == [0.0, 0.0, 0.0]
    assert Generic([1], dtype=DType.Int32).allocate_like(2)[0] == 0


# --- NumPy is lazy ---------------------------------------------------------


@pytest.mark.parametrize(
    "program",
    [
        "import strideweave",
        "import strideweave as sw; sw.CPU(4)",
        "import strideweave as sw; sw.Generic([1], dtype=sw.DType.Int32)",
    ],
)
def test_numpy_is_not_imported_without_concrete_float32(program: str) -> None:
    source = f"{program}\nimport sys\nassert 'numpy' not in sys.modules\n"

    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


# --- Binary32 forward semantics --------------------------------------------


def test_float32_arithmetic_rounds_at_every_step():
    lhs = generic_tensor([1.0, 0.1], DType.Float32)
    rhs = generic_tensor([2**-24, 0.2], DType.Float32)

    result = lhs + rhs

    assert result.dtype() is DType.Float32
    # 1.0 + 2**-24 is below the binary32 tie, so it rounds back to 1.0.
    assert result[0] == 1.0
    assert result[1] == binary32(binary32(0.1) + binary32(0.2))


def test_division_by_zero_is_ieee_rather_than_an_exception():
    lhs = generic_tensor([1.0, 0.0], DType.Float32)
    rhs = generic_tensor([0.0, 0.0], DType.Float32)

    result = lhs / rhs

    assert result[0] == float("inf")
    assert result[1] != result[1]  # NaN


def test_zero_base_negative_exponent_is_ieee():
    tensor = generic_tensor([0.0, 0.0], DType.Float32)

    assert (tensor**-1)[0] == float("inf")
    assert (tensor**0)[0] == 1.0


def test_exp_overflow_saturates_to_infinity():
    tensor = generic_tensor([1000.0, 0.0], DType.Float32)

    result = sw.exp(tensor)

    assert result[0] == float("inf")
    assert result[1] == 1.0


def test_reduce_can_widen_only_its_float32_accumulation():
    layout = Layout(Shape([1, 4]), Stride([1, 1]))
    values = [2**24, 1.0, 1.0, -(2**24)]
    tensor = generic_tensor(values, DType.Float32, layout)
    stored_before = values_of(tensor)

    widened = sw.reduce_sum(tensor, "a b -> a", accumulator_dtype=DType.Float64)
    explicit_default = sw.reduce_sum(
        tensor, "a b -> a", accumulator_dtype=DType.Float32
    )
    default = sw.reduce_sum(tensor, "a b -> a")

    assert stored_before == [binary32(value) for value in values]
    assert values_of(tensor) == stored_before
    assert tensor.dtype() is DType.Float32
    assert widened.dtype() is DType.Float32
    assert widened[0] == 2.0
    assert explicit_default[0] == default[0] == 0.0


def test_matmul_can_widen_only_its_float32_accumulation():
    layout = Layout(Shape([1, 4]), Stride([1, 1]))
    lhs = generic_tensor([2**24, 1.0, 1.0, -(2**24)], DType.Float32, layout)
    rhs = generic_tensor([1.0, 1.0, 1.0, 1.0], DType.Float32, layout)

    widened = sw.matmul(lhs, rhs, accumulator_dtype=DType.Float64)
    default = sw.matmul(lhs, rhs)

    assert lhs.dtype() is rhs.dtype() is DType.Float32
    assert widened.dtype() is default.dtype() is DType.Float32
    assert widened[0, 0] == 2.0
    assert default[0, 0] == 0.0


def test_int32_arithmetic_is_exact_and_checked():
    lhs = generic_tensor([2**31 - 1, 2], DType.Int32)
    rhs = generic_tensor([1, 3], DType.Int32)

    with pytest.raises(OverflowError, match="out of int32 range"):
        _ = lhs + rhs

    assert values_of(generic_tensor([2, 3], DType.Int32) * 2) == [4, 6]


def test_int32_reduction_accumulates_exactly_and_checks_only_the_result():
    maximum = 2**31 - 1
    layout = Layout(Shape([1, 3]), Stride([1, 1]))
    tensor = generic_tensor([maximum, maximum, -maximum], DType.Int32, layout)

    # The partial sum leaves Int32 range but the final sum does not.
    assert sw.reduce_sum(tensor, "a b -> a")[0] == maximum


def test_int32_reduction_overflow_is_reported():
    layout = Layout(Shape([1, 2]), Stride([1, 1]))
    tensor = generic_tensor([2**31 - 1, 1], DType.Int32, layout)

    with pytest.raises(OverflowError, match="out of int32 range"):
        sw.reduce_sum(tensor, "a b -> a")


# --- Promotion follows the shared plans ------------------------------------


@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "expected"),
    [
        (DType.Float32, DType.Float32, DType.Float32),
        (DType.Float32, DType.Int32, DType.Float32),
        (DType.Int32, DType.Float32, DType.Float32),
        (DType.Int32, DType.Int32, DType.Int32),
    ],
)
def test_binary_results_follow_the_resolved_plan(lhs_dtype, rhs_dtype, expected):
    lhs = generic_tensor([1, 2], lhs_dtype)
    rhs = generic_tensor([3, 4], rhs_dtype)

    assert (lhs + rhs).dtype() is expected


def test_division_is_floating_even_for_two_integer_operands():
    lhs = generic_tensor([1, 3], DType.Int32)
    rhs = generic_tensor([2, 2], DType.Int32)

    result = lhs / rhs

    assert result.dtype() is DType.Float32
    assert values_of(result) == [0.5, 1.5]


def test_relu_preserves_its_input_dtype():
    assert sw.relu(generic_tensor([-1, 2], DType.Int32)).dtype() is DType.Int32
    assert sw.relu(generic_tensor([-1.0, 2.0], DType.Float32)).dtype() is DType.Float32


def test_activations_are_floating_for_an_integer_input():
    result = sw.sigmoid(generic_tensor([0, 1], DType.Int32))

    assert result.dtype() is DType.Float32
    assert result[0] == 0.5


def test_integer_pow_stays_integer_only_for_a_preserving_exponent():
    tensor = generic_tensor([2, 3], DType.Int32)

    assert (tensor**2).dtype() is DType.Int32
    assert (tensor**-1).dtype() is DType.Float32
    assert (tensor**0.5).dtype() is DType.Float32


# --- Autograd --------------------------------------------------------------


def test_int32_results_are_not_differentiable():
    tensor = generic_tensor([1, 2], DType.Int32)

    assert not tensor.is_differentiable()
    with pytest.raises(RuntimeError, match="non-differentiable"):
        tensor.backward()


def test_float32_backward_computes_in_binary32():
    tensor = generic_tensor([0.1, 0.1], DType.Float32)

    result = sw.mul(tensor, 0.1)
    result.backward(generic_tensor([1.0, 1.0], DType.Float32))

    assert require_grad(tensor).dtype() is DType.Float32
    # The scalar is rounded to binary32 once and reused, so the gradient is
    # exactly the stored binary32 multiplier rather than the Python float.
    assert require_grad(tensor)[0] == binary32(0.1)


def test_division_backward_is_ieee_at_a_zero_divisor():
    lhs = generic_tensor([1.0, 1.0], DType.Float32)
    rhs = generic_tensor([0.0, 2.0], DType.Float32)

    result = lhs / rhs
    result.backward(generic_tensor([1.0, 1.0], DType.Float32))

    assert require_grad(lhs)[0] == float("inf")
    assert require_grad(rhs)[0] == float("-inf")


def test_pow_backward_at_a_zero_base_is_ieee():
    tensor = generic_tensor([0.0, 2.0], DType.Float32)

    result = tensor**-1
    result.backward(generic_tensor([1.0, 1.0], DType.Float32))

    assert require_grad(tensor)[0] == float("-inf")


def test_matmul_gradients_accumulate_in_binary32():
    layout = Layout(Shape([1, 2]), Stride([1, 1]))
    lhs = generic_tensor([1.0, 2.0], DType.Float32, layout)
    rhs = generic_tensor([3.0, 4.0], DType.Float32, layout)

    result = lhs @ rhs
    result.backward(Tensor(Generic([1.0], dtype=DType.Float32), 0, result.layout))

    assert require_grad(lhs).dtype() is DType.Float32
    assert values_of(require_grad(lhs)) == [3.0, 4.0]


def test_matmul_backward_reuses_the_forward_accumulator_dtype():
    lhs_layout = Layout(Shape([1, 1]), Stride([1, 1]))
    rhs_layout = Layout(Shape([4, 1]), Stride([1, 1]))
    lhs = generic_tensor([1.0], DType.Float32, lhs_layout)
    rhs = generic_tensor([2**24, 1.0, 1.0, -(2**24)], DType.Float32, rhs_layout)

    result = sw.matmul(lhs, rhs, accumulator_dtype=DType.Float64)
    result.backward(generic_tensor([1.0] * 4, DType.Float32, result.layout))

    assert require_grad(lhs).dtype() is DType.Float32
    assert require_grad(lhs)[0, 0] == 2.0


# --- The legacy boundary ---------------------------------------------------


def test_mixing_legacy_and_concrete_storage_stays_on_the_legacy_path():
    # Documented behavior, not a promotion result: a legacy Floating operand
    # keeps Generic's historical Python arithmetic, so the concrete operand's
    # binary32 semantics are downgraded to binary64 for this operation.
    legacy = generic_tensor([0.1, 0.1], DType.Floating)
    concrete = generic_tensor([0.2, 0.2], DType.Float32)

    result = legacy + concrete

    assert result.dtype() is DType.Floating
    assert result[0] == 0.1 + binary32(0.2)


def test_legacy_any_storage_is_never_routed_through_checked_integer():
    huge = 2**40
    legacy = generic_tensor([huge, huge], DType.Any)

    result = legacy + legacy

    assert result.dtype() is DType.Any
    assert result[0] == 2 * huge


def test_generic_and_cpu_agree_on_concrete_results():
    generic = generic_tensor([0.1, 0.2], DType.Float32)
    cpu_carrier = CPU(2, dtype=DType.Float32)
    cpu_carrier[0], cpu_carrier[1] = 0.1, 0.2
    cpu = Tensor(cpu_carrier, 0, ONE_MODE)

    assert values_of(generic * 3) == values_of(cpu * 3)
    assert values_of(generic + generic) == values_of(cpu + cpu)


# --- The capability boundary ------------------------------------------------
#
# Generic declares the plan shapes it executes, and `arithmetic_for_plan`
# refuses everything else before converting a single value. Two things are
# checked separately here. `executable_plan_shape` is the predicate those
# declarations are filtered through, so it is asked directly about shapes whose
# fields are deliberately *not* correlated the way the v0.2 matrix correlates
# them — shapes a future policy revision could produce. Execution is then
# exercised only through the shapes Generic actually declares, so no test has to
# widen the shipped backend to reach the adapter.


def plan_of(
    compute,
    output,
    accumulation=None,
    accumulator_dtype=None,
    convert_to=None,
    operation="relu",
):
    """Build a plan directly, bypassing the resolver's own correlations."""
    convert_to = output if convert_to is None else convert_to
    return OperationPlan(
        operation=operation,
        operands=(
            OperandPlan(
                role=OperandRole.TENSOR, dtype=convert_to, convert_to=convert_to
            ),
        ),
        compute=compute,
        accumulation=accumulation,
        accumulator_dtype=accumulator_dtype,
        output=output,
    )


def plan_of_capability(capability):
    """Return the plan whose shape ``capability`` accepts."""
    return OperationPlan(
        operation=capability.operation,
        operands=tuple(
            OperandPlan(
                role=operand.role, dtype=operand.dtype, convert_to=operand.convert_to
            )
            for operand in capability.operands
        ),
        compute=capability.compute,
        accumulation=capability.accumulation,
        accumulator_dtype=capability.accumulator_dtype,
        output=capability.output,
    )


MIXED_CONVERSION_PLAN = OperationPlan(
    operation="add",
    operands=(
        OperandPlan(
            role=OperandRole.TENSOR, dtype=DType.Float32, convert_to=DType.Float32
        ),
        OperandPlan(role=OperandRole.TENSOR, dtype=DType.Int32, convert_to=DType.Int32),
    ),
    compute=Arithmetic.BINARY32,
    accumulation=None,
    accumulator_dtype=None,
    output=DType.Float32,
)

# What each compute arithmetic implies about the rest of a plan, stated
# independently of the implementation so a table edited in one place fails here.
COMPUTE_REPRESENTATION = {
    Arithmetic.BINARY32: DType.Float32,
    Arithmetic.INT32_EXACT_CHECKED: DType.Int32,
    Arithmetic.INT32_EXACT: DType.Int32,
}
ACCUMULATING_OPERATIONS = frozenset(
    {
        "argmax",
        "argmin",
        "conv_general",
        "cumsum",
        "matmul",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "scatter_add",
    }
)

# One accumulator shape per (accumulation, accumulator_dtype) pair a capability
# may declare. Generic implements both floating accumulators for the two sum
# reductions whose association order the policy leaves to the backend; every
# other combining operation pins one order-normative rule and no accumulator
# dtype.
_FLOATING = frozenset(
    {
        (Accumulation.FLOATING, DType.Float32),
        (Accumulation.FLOATING, DType.Float64),
    }
)
_EXACT_INTEGER = frozenset({(Accumulation.EXACT_INTEGER, None)})


def _pinned(accumulation):
    return frozenset({(accumulation, None)})


EXPECTED_ACCUMULATIONS = {
    "argmax": _pinned(Accumulation.ARGMAX),
    "argmin": _pinned(Accumulation.ARGMIN),
    "conv_general": _pinned(Accumulation.SEQUENTIAL_BINARY32),
    "cumsum": _pinned(Accumulation.SEQUENTIAL_BINARY32),
    "matmul": _FLOATING | _EXACT_INTEGER,
    "reduce_max": _pinned(Accumulation.MAXIMUM),
    "reduce_min": _pinned(Accumulation.MINIMUM),
    "reduce_prod": _pinned(Accumulation.SEQUENTIAL_BINARY32_PRODUCT),
    "reduce_sum": _FLOATING | _EXACT_INTEGER,
    "scatter_add": _pinned(Accumulation.SEQUENTIAL_BINARY32),
}


def _operand_signature(operands):
    """Return the policy-relevant shape of capability operands."""
    return tuple(
        (operand.role, operand.dtype, operand.convert_to) for operand in operands
    )


# These operations deliberately store a logical result in a representation
# different from their compute arithmetic.  Keep their complete registered
# shapes explicit so the capability test does not turn into a blanket waiver
# for arbitrary Bool/Int32 outputs.
EXPECTED_NON_COMPUTE_OUTPUTS = {
    "_sort_indices": (
        DType.Int32,
        Arithmetic.BINARY32,
        None,
        ((OperandRole.TENSOR, DType.Float32, DType.Float32),),
    ),
    "_topk_indices": (
        DType.Int32,
        Arithmetic.BINARY32,
        None,
        ((OperandRole.TENSOR, DType.Float32, DType.Float32),),
    ),
    "argmax": (
        DType.Int32,
        Arithmetic.BINARY32,
        Accumulation.ARGMAX,
        ((OperandRole.TENSOR, DType.Float32, DType.Float32),),
    ),
    "argmin": (
        DType.Int32,
        Arithmetic.BINARY32,
        Accumulation.ARGMIN,
        ((OperandRole.TENSOR, DType.Float32, DType.Float32),),
    ),
    "eq": (
        DType.Bool,
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
    "le": (
        DType.Bool,
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
    "logical_not": (
        DType.Bool,
        Arithmetic.BINARY32,
        None,
        ((OperandRole.TENSOR, DType.Float32, DType.Float32),),
    ),
    "lt": (
        DType.Bool,
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
    "ne": (
        DType.Bool,
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
}

EXPECTED_MIXED_CONVERSIONS = {
    "gather": (
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Int32, DType.Int32),
        ),
    ),
    "scatter": (
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Int32, DType.Int32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
    "scatter_add": (
        Arithmetic.BINARY32,
        Accumulation.SEQUENTIAL_BINARY32,
        (
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Int32, DType.Int32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
    "select": (
        Arithmetic.BINARY32,
        None,
        (
            (OperandRole.TENSOR, DType.Bool, DType.Bool),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
            (OperandRole.TENSOR, DType.Float32, DType.Float32),
        ),
    ),
}

INCOHERENT_PLANS = {
    # A binary32 computation stored as Int32 is the truncation this boundary
    # exists for, from either conversion target.
    "binary32-compute-int32-output": plan_of(
        Arithmetic.BINARY32, DType.Int32, convert_to=DType.Float32
    ),
    "binary32-compute-int32-conversion": plan_of(
        Arithmetic.BINARY32, DType.Int32, convert_to=DType.Int32
    ),
    # ...and the opposite direction: exact integer arithmetic cannot produce or
    # be reached from a Float32 representation.
    "integer-compute-float32-output": plan_of(
        Arithmetic.INT32_EXACT_CHECKED, DType.Float32, convert_to=DType.Int32
    ),
    "integer-compute-float32-conversion": plan_of(
        Arithmetic.INT32_EXACT_CHECKED, DType.Int32, convert_to=DType.Float32
    ),
    "binary32-compute-exact-integer-accumulation": plan_of(
        Arithmetic.BINARY32,
        DType.Float32,
        accumulation=Accumulation.EXACT_INTEGER,
        accumulator_dtype=None,
        operation="reduce_sum",
    ),
    "integer-compute-binary32-accumulation": plan_of(
        Arithmetic.INT32_EXACT_CHECKED,
        DType.Int32,
        accumulation=Accumulation.FLOATING,
        accumulator_dtype=DType.Float32,
        operation="reduce_sum",
    ),
    # An operation whose loop combines terms needs an accumulation to combine
    # them with, and one that writes an element per element has nowhere to
    # apply one.
    "reduce-without-accumulation": plan_of(
        Arithmetic.BINARY32, DType.Float32, operation="reduce_sum"
    ),
    "matmul-without-accumulation": plan_of(
        Arithmetic.INT32_EXACT, DType.Int32, operation="matmul"
    ),
    "accumulating-elementwise": plan_of(
        Arithmetic.BINARY32,
        DType.Float32,
        accumulation=Accumulation.FLOATING,
        accumulator_dtype=DType.Float32,
        operation="relu",
    ),
    "mixed-conversion": MIXED_CONVERSION_PLAN,
    "unimplemented-representation": plan_of(
        Arithmetic.BINARY32, DType.Int8, convert_to=DType.Int8
    ),
}


@pytest.mark.parametrize("name", sorted(INCOHERENT_PLANS))
def test_the_predicate_rejects_a_cross_field_incoherent_plan(name):
    # Fields are accepted against each other, not one at a time: a plan whose
    # compute, conversion, accumulation and output describe different arithmetics
    # is one Generic cannot execute as written, whatever its individual fields.
    assert not executable_plan_shape(INCOHERENT_PLANS[name])


def test_the_predicate_accepts_every_plan_the_policy_resolves_for_generic():
    # Narrowing the predicate must not narrow what Generic supports: every shape
    # the central policy resolves over Generic's storage dtypes is still
    # executable, so nothing in the shipped backend is lost to the tightening.
    plans = {
        *resolvable_plans(CONCRETE_DTYPES),
        resolve_operation_plan("select", DType.Bool, DType.Float32, DType.Float32),
    }
    advertised = {
        plan_of_capability(capability) for capability in generic_capabilities()
    }
    declared = {
        plan_of_capability(capability)
        for capability in capabilities_for_carrier_class(Generic)
    }

    assert plans
    assert all(executable_plan_shape(plan) for plan in plans)
    assert advertised == plans
    assert declared == plans


def test_generic_advertises_no_cross_field_incoherent_capability():
    # The declarations are filtered through the predicate, so a future resolver
    # shape whose fields Generic would execute inconsistently cannot be
    # advertised. Checked against an independent statement of the matrix.
    for capability in generic_capabilities():
        representation = COMPUTE_REPRESENTATION[capability.compute]

        if capability.output is not representation:
            expected = EXPECTED_NON_COMPUTE_OUTPUTS.get(capability.operation)
            assert expected is not None
            assert (
                capability.output,
                capability.compute,
                capability.accumulation,
                _operand_signature(capability.operands),
            ) == expected
        else:
            expected_mixed = EXPECTED_MIXED_CONVERSIONS.get(capability.operation)
            if expected_mixed is None:
                assert all(
                    operand.convert_to is representation
                    for operand in capability.operands
                )
            else:
                assert (
                    capability.compute,
                    capability.accumulation,
                    _operand_signature(capability.operands),
                ) == expected_mixed
        if capability.operation in ACCUMULATING_OPERATIONS:
            assert (
                capability.accumulation,
                capability.accumulator_dtype,
            ) in EXPECTED_ACCUMULATIONS[capability.operation]
        else:
            assert capability.accumulation is None
            assert capability.accumulator_dtype is None


def test_a_binary32_reduction_is_never_accumulated_as_exact_integers():
    # The confirmed bug: an exact-integer accumulation over binary32 terms
    # int()-truncates each term, so 1.5 + 1.5 would reduce to 2 rather than 3.
    # The shape is refused before a value is converted, and the declared plan
    # Generic really runs for Float32 keeps the fractional terms.
    truncating = INCOHERENT_PLANS["binary32-compute-exact-integer-accumulation"]

    assert not supports_operation_plan(Generic, truncating)
    with pytest.raises(UnsupportedOperationPlan, match="Generic declares no"):
        arithmetic_for_plan(truncating, Generic)

    layout = Layout(Shape([1, 2]), Stride([1, 1]))
    assert (
        sw.reduce_sum(generic_tensor([1.5, 1.5], DType.Float32, layout), "a b -> a")[0]
        == 3.0
    )


def test_a_float32_compute_never_stores_its_result_as_int32():
    # The other confirmed bug: a Float32-conversion, binary32 plan with an Int32
    # output produced an Int32 adapter that stored 1.5 as 1.
    truncating = INCOHERENT_PLANS["binary32-compute-int32-output"]

    assert not executable_plan_shape(truncating)
    assert not supports_operation_plan(Generic, truncating)
    with pytest.raises(UnsupportedOperationPlan):
        arithmetic_for_plan(truncating, Generic)


@pytest.mark.parametrize("name", sorted(INCOHERENT_PLANS))
def test_an_undeclared_plan_shape_is_refused(name):
    # Refusing beats executing a nearby shape: a policy revision Generic has no
    # implementation for must be loud (policy section 10.5). The registry is the
    # only gate, so the refusal is observed through `arithmetic_for_plan` itself.
    with pytest.raises(UnsupportedOperationPlan, match="Generic declares no"):
        arithmetic_for_plan(INCOHERENT_PLANS[name], Generic)


def test_the_adapter_takes_its_result_dtype_from_the_plan_output():
    # `relu` on Int32 is the declared shape whose compute is the unchecked
    # INT32_EXACT: the policy uses it only where the result provably fits, so
    # the adapter must not add a check of its own.
    plan = resolve_operation_plan("relu", DType.Int32)
    arithmetic = arithmetic_for_plan(plan, Generic)

    assert arithmetic.result_dtype is DType.Int32
    assert arithmetic.store(2**40) == 2**40


def test_the_adapter_checks_an_accumulated_integer_narrowing():
    # Same compute, but the declared accumulating shape: `matmul` on Int32
    # checks the finished sum even though every term was exact.
    plan = resolve_operation_plan("matmul", DType.Int32, DType.Int32)
    arithmetic = arithmetic_for_plan(plan, Generic)

    assert plan.compute is Arithmetic.INT32_EXACT
    assert arithmetic.total([2**31, 2**31, -(2**31)]) == 2**31
    with pytest.raises(OverflowError, match="out of int32 range"):
        arithmetic.store(2**31)


def test_the_adapter_takes_its_accumulation_from_the_plan():
    plan = resolve_operation_plan("reduce_sum", DType.Float32)
    sequential = arithmetic_for_plan(plan, Generic)

    # Rounding at every step, not once at the end: 2**24 + 1 + 1 stays at 2**24
    # under sequential binary32 while an exact sum would reach 2**24 + 2.
    terms = [binary32(2**24), binary32(1.0), binary32(1.0)]
    assert float(sequential.total(terms)) == float(2**24)


def test_a_non_accumulating_plan_refuses_to_combine_terms():
    arithmetic = arithmetic_for_plan(
        resolve_operation_plan("relu", DType.Float32), Generic
    )

    with pytest.raises(ValueError, match="no accumulation"):
        arithmetic.total([1.0, 2.0])


def test_every_declared_capability_is_executable():
    # Advertised support and executable support are one set: every capability
    # Generic declares must build an arithmetic whose behavior matches it.
    for capability in capabilities_for_carrier_class(Generic):
        plan = plan_of_capability(capability)
        arithmetic = arithmetic_for_plan(plan, Generic)

        assert arithmetic.result_dtype is capability.output
        assert arithmetic.plan is plan
        if capability.accumulation is None:
            with pytest.raises(ValueError, match="no accumulation"):
                arithmetic.total([])
        else:
            assert arithmetic.total([]) is not None


def test_every_planned_operation_matches_a_declared_capability():
    # The other direction: every plan a real operation resolves is declared, so
    # no execution reaches a kernel the registry never accepted.
    for lhs_dtype, rhs_dtype in itertools.product(CONCRETE_DTYPES, repeat=2):
        lhs = generic_tensor([1, 2], lhs_dtype)
        rhs = generic_tensor([3, 4], rhs_dtype)

        for operation in ("add", "sub", "elementwise_mul", "div"):
            plan = resolve_operation_plan(operation, lhs_dtype, rhs_dtype)
            assert supports_operation_plan(Generic, plan)
            assert getattr(sw, operation)(lhs, rhs).dtype() is plan.output


def test_generic_is_a_closed_carrier_implementation():
    # Generic's factories, storage normalization and capability declarations are
    # all stated in terms of its exact class, so a subclass would advertise
    # support Generic's own allocation refuses to produce. Extending StrideWeave
    # means a sibling Carrier, normally composing one of these.
    with pytest.raises(TypeError, match="Generic is a closed carrier implementation"):
        type("QuietGeneric", (Generic,), {})

    # The exact class keeps executing what it declares.
    plan = resolve_operation_plan("relu", DType.Int32)
    assert arithmetic_for_plan(plan, Generic).result_dtype is DType.Int32
