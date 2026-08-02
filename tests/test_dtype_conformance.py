"""Generic and native CPU conform to the same SimpleDType policy.

``design/SimpleDType-operation-policy.md`` is the specification,
``strideweave.carriers.operation_policy`` is its executable form, and ``Generic``
is the behavioral reference. This suite proves that native ``CPU`` agrees with
``Generic`` — and that both agree with the resolved plan — for every operation
the policy registers, in forward and in backward.

Coverage is enumerated from the operation registry rather than sampled, so an
operation added to the policy without conformance coverage fails here (policy
section 10.3). Results are compared bit-exactly by default; tolerance is
admitted only for the transcendentals, and is named and justified where it is
defined below (policy section 10.4).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from itertools import product
from typing import Any, Self

import numpy
import pytest

import strideweave as sw
from strideweave import CPU, DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.operation_policy import (
    OperandRole,
    OperationOverload,
    OperationSpec,
    registered_operations,
    resolve_operation_plan,
)

ONE_MODE = Layout(Shape(4), Stride(1))
TWO_MODE = Layout(Shape([2, 2]), Stride([1, 2]))

# Operations whose operands must carry a contractible two-mode layout.
TWO_MODE_OPERATIONS = frozenset(
    {
        "argmax",
        "argmin",
        "cumsum",
        "matmul",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "scatter",
        "scatter_add",
        "_sort_indices",
        "_sort_values",
        "_topk_indices",
        "_topk_values",
    }
)

GATHER_INDEX_LAYOUT = Layout(Shape(2), Stride(1))
CONV_LHS_LAYOUT = Layout(Shape([1, 1, 4]), Stride([1, 1, 1]))
CONV_KERNEL_LAYOUT = Layout(Shape([1, 1, 1]), Stride([1, 1, 1]))

# Sample operands. The floating values exercise rounding (0.1 is not
# representable in binary32) and both signs; the integers stay small enough that
# every integer plan produces a representable result.
FLOAT_SAMPLE = (1.5, -2.25, 0.1, 3.0)
# Accumulating operations use integer-valued binary32 terms whose products and
# every possible partial sum are exact. Their bit comparison therefore checks
# structure and addressing without requiring Generic and CPU to associate a
# floating reduction in the same order.
FLOAT_ACCUMULATION_SAMPLE = (1.0, -2.0, 3.0, 4.0)

# The operations whose floating accumulation declares no normative association
# order, and therefore need the order-independent payload above. Every other
# combining operation pins its own order, so both backends must already agree
# on the ordinary sample.
FLOATING_ACCUMULATION_OPERATIONS = frozenset({"reduce_sum", "matmul"})
INT_SAMPLE = (1, -2, 3, 4)
BOOL_SAMPLE = (True, False, True, False)
SELECT_TRUE_SAMPLE = (1.0, 2.0, 3.0, 4.0)
SELECT_FALSE_SAMPLE = (10.0, 20.0, 30.0, 40.0)
CLAMP_INPUT_SAMPLE = (-2.0, -0.5, 0.5, 3.0)
CLAMP_LOWER_SAMPLE = (-1.0, -1.0, -1.0, -1.0)
CLAMP_UPPER_SAMPLE = (1.0, 1.0, 1.0, 1.0)

# Weak scalars covering both plan-selecting kinds and the pinned `bool` case.
WEAK_SCALARS = (2, 2.5, True)

# The transcendentals. `Generic` composes these from Python's double-precision
# libm and rounds the result to binary32 once, while CPU calls the single
# precision libm at every step, so the two may differ in the last binary32 ulp.
# These are the only operations this suite compares with a tolerance; every
# other operation is compared bit-exactly.
TRANSCENDENTAL_OPERATIONS = frozenset(
    {
        "elu",
        "erf",
        "exp",
        "gelu",
        "leaky_relu",
        "pow",
        "sigmoid",
        "silu",
        "softplus",
        "tanh",
    }
)

# The admitted spreads, in relative terms. Forward, a last-ulp libm difference
# is amplified by the cancellation in `gelu`'s ``0.5 * x * (1 + erf(x/sqrt 2))``
# at negative x. Backward, the derivative formulas amplify it further:
# ``1 - tanh(x)**2`` loses most of its significance as ``|tanh(x)|`` approaches
# one. Both bounds sit a few times above the worst observed divergence.
#
# These tolerances do not distinguish binary32 from binary64 evaluation of a
# transcendental — no comparison between two libm implementations can. That is
# pinned instead by exact comparison of the algebraic operations, where no libm
# is involved, and by exact dtype identity on every case.
LIBM_FORWARD_TOLERANCE = 2e-6
LIBM_BACKWARD_TOLERANCE = 2e-5


def test_erf_is_compared_with_platform_libm_tolerance() -> None:
    """Keep ``erf`` out of the bit-exact cross-platform comparison path."""
    assert "erf" in TRANSCENDENTAL_OPERATIONS


def generic_carrier(values: tuple[Any, ...], dtype: DType) -> Generic:
    return Generic(list(values), dtype=dtype)


def cpu_carrier(values: tuple[Any, ...], dtype: DType) -> CPU:
    carrier = CPU(len(values), dtype=dtype)
    for index, value in enumerate(values):
        carrier[index] = value
    return carrier


BACKENDS = {"generic": generic_carrier, "cpu": cpu_carrier}


def sample_for(dtype: DType, *, accumulating: bool = False) -> tuple[Any, ...]:
    if dtype is DType.Float32:
        return FLOAT_ACCUMULATION_SAMPLE if accumulating else FLOAT_SAMPLE
    if dtype is DType.Bool:
        return BOOL_SAMPLE
    return INT_SAMPLE


def values_of(tensor: Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def float32_from_bits(bits: int) -> float:
    """Construct a Python float carrying one exact binary32 bit pattern."""
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def float32_bits(value: float) -> int:
    """Return the binary32 storage pattern of a Python numeric value."""
    return struct.unpack("!I", struct.pack("!f", value))[0]


def ones_like(tensor: Tensor, backend: str) -> Tensor:
    size = tensor.layout._cache.cosize
    return Tensor(BACKENDS[backend]((1.0,) * size, DType.Float32), 0, tensor.layout)


@dataclass(frozen=True)
class Outcome:
    """What one backend did: a dtype and values, or the exception it raised."""

    dtype: DType | None = None
    values: tuple[Any, ...] = ()
    gradients: tuple[tuple[Any, ...] | None, ...] = ()
    has_graph: bool = False
    error: type[BaseException] | None = None
    message: str = ""


def operation_spec(operation: str) -> OperationSpec:
    """Return one registered operation specification by name."""
    return {candidate.name: candidate for candidate in registered_operations()}[
        operation
    ]


def operation_overload(operation: str, overload_index: int) -> OperationOverload:
    """Return an explicitly selected overload from the operation registry."""
    return operation_spec(operation).overloads[overload_index]


def layout_for_tensor(operation: str, tensor_index: int) -> Layout:
    """Choose a small valid layout for one tensor argument."""
    if operation == "gather" and tensor_index == 1:
        return GATHER_INDEX_LAYOUT
    if operation in {"scatter", "scatter_add"} and tensor_index == 1:
        return GATHER_INDEX_LAYOUT
    if operation == "conv_general":
        return CONV_LHS_LAYOUT if tensor_index == 0 else CONV_KERNEL_LAYOUT
    return TWO_MODE if operation in TWO_MODE_OPERATIONS else ONE_MODE


def values_for_tensor(
    operation: str,
    overload_index: int,
    tensor_index: int,
    dtype: DType,
) -> tuple[Any, ...]:
    """Provide values compatible with operation-specific shape parameters."""
    if operation in {"gather", "scatter", "scatter_add"} and tensor_index == 1:
        return (0, 1)
    if operation == "conv_general":
        return (1.5, -2.25, 0.1, 3.0) if tensor_index == 0 else (2.0,)
    if operation == "select":
        return (BOOL_SAMPLE, SELECT_TRUE_SAMPLE, SELECT_FALSE_SAMPLE)[tensor_index]
    if operation == "clamp":
        if tensor_index == 0:
            return CLAMP_INPUT_SAMPLE
        if overload_index in {0, 1} and tensor_index == 1:
            return CLAMP_LOWER_SAMPLE
        return CLAMP_UPPER_SAMPLE
    return sample_for(dtype, accumulating=operation in FLOATING_ACCUMULATION_OPERATIONS)


def invoke(operation: str, arguments: list[Any], backend: str) -> Tensor:
    """Invoke public and internal operations with their non-dtype parameters."""
    if operation in {
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
    }:
        return getattr(sw, operation)(arguments[0], "a b -> a")
    if operation == "cumsum":
        return sw.cumsum(arguments[0], 1)
    if operation == "gather":
        return sw.gather(arguments[0], arguments[1], 0)
    if operation in {"scatter", "scatter_add"}:
        return getattr(sw, operation)(arguments[0], arguments[1], arguments[2], 0)
    if operation == "conv_general":
        return (
            arguments[0]
            .carrier.dispatch_op(operation)
            .forward(
                arguments[0],
                arguments[1],
                (1,),
                ((0, 0),),
            )
        )
    if operation == "_sort_values":
        return (
            arguments[0].carrier.dispatch_op(operation).forward(arguments[0], 1, False)
        )
    if operation == "_sort_indices":
        return (
            arguments[0].carrier.dispatch_op(operation).forward(arguments[0], 1, False)
        )
    if operation == "_topk_values":
        return (
            arguments[0]
            .carrier.dispatch_op(operation)
            .forward(arguments[0], 1, 1, True)
        )
    if operation == "_topk_indices":
        return (
            arguments[0]
            .carrier.dispatch_op(operation)
            .forward(arguments[0], 1, 1, True)
        )
    return getattr(sw, operation)(*arguments)


def specification(
    operation: str,
    overload_index: int,
    dtypes: tuple[DType, ...],
    scalar: object,
):
    """Build the operand description for one selected registry overload."""
    overload = operation_overload(operation, overload_index)
    remaining = list(dtypes)
    if operation == "clamp":
        assert isinstance(scalar, tuple)
        clamp_scalars = iter(scalar)
    else:
        clamp_scalars = iter(())
    roles: list[tuple[OperandRole, Any]] = []
    for role in overload.roles:
        if role is OperandRole.TENSOR:
            roles.append((role, remaining.pop(0)))
        else:
            roles.append(
                (role, next(clamp_scalars) if operation == "clamp" else scalar)
            )
    return roles


def run(
    operation: str,
    overload_index: int,
    dtypes: tuple[DType, ...],
    scalar: object,
    backend: str,
):
    """Run one operation on one backend, forward and, if possible, backward."""
    roles = specification(operation, overload_index, dtypes, scalar)
    make = BACKENDS[backend]
    tensors: list[Tensor] = []
    arguments: list[Any] = []
    tensor_index = 0
    for role, value in roles:
        if role is OperandRole.TENSOR:
            layout = layout_for_tensor(operation, tensor_index)
            values = values_for_tensor(operation, overload_index, tensor_index, value)
            tensor = Tensor(make(values, value), 0, layout)
            tensors.append(tensor)
            arguments.append(tensor)
            tensor_index += 1
        else:
            arguments.append(value)

    try:
        result = invoke(operation, arguments, backend)
    except Exception as error:
        # The refusal is part of the outcome: both backends must refuse the
        # same invocations with the same exception type *and* the same message,
        # since a shared policy rejection should read identically wherever it
        # is raised.
        return Outcome(error=type(error), message=str(error))

    gradients: list[tuple[Any, ...] | None] = []
    has_graph = result.autograd_ctx is not None
    if has_graph:
        result.backward(ones_like(result, backend))
        for tensor in tensors:
            gradient = tensor.grad if tensor.is_differentiable() else None
            gradients.append(None if gradient is None else tuple(values_of(gradient)))
    return Outcome(
        dtype=result.dtype(),
        values=tuple(values_of(result)),
        gradients=tuple(gradients),
        has_graph=has_graph,
    )


def same_value(left: Any, right: Any, *, tolerance: float | None) -> bool:
    """Compare two element values, treating NaN as equal to NaN."""
    left_is_nan = isinstance(left, float) and math.isnan(left)
    right_is_nan = isinstance(right, float) and math.isnan(right)
    if left_is_nan or right_is_nan:
        return left_is_nan and right_is_nan
    if tolerance is None or not isinstance(left, float):
        # Bit-exact, including the sign of a zero and the sign of an infinity.
        return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return left == pytest.approx(right, rel=tolerance)


def same_values(left, right, *, tolerance: float | None) -> bool:
    return len(left) == len(right) and all(
        same_value(one, other, tolerance=tolerance)
        for one, other in zip(left, right, strict=True)
    )


def assert_conformant(
    generic: Outcome,
    cpu: Outcome,
    *,
    forward_tolerance: float | None,
    backward_tolerance: float | None,
):
    assert cpu.error is generic.error
    assert cpu.message == generic.message
    assert cpu.dtype is generic.dtype
    # Both backends must agree on whether a graph node was attached; comparing
    # gradients alone would pass if neither attached one.
    assert cpu.has_graph is generic.has_graph
    assert same_values(cpu.values, generic.values, tolerance=forward_tolerance)
    assert len(cpu.gradients) == len(generic.gradients)
    for cpu_gradient, generic_gradient in zip(
        cpu.gradients, generic.gradients, strict=True
    ):
        assert (cpu_gradient is None) is (generic_gradient is None)
        if cpu_gradient is not None and generic_gradient is not None:
            assert same_values(
                cpu_gradient, generic_gradient, tolerance=backward_tolerance
            )


def registered_cases() -> list[tuple[str, int, tuple[DType, ...], object]]:
    """Enumerate every registered operation over every supported combination.

    The scalar is varied only for the operations that take one. A scalar an
    operation never reads cannot change its outcome, so generating one case per
    scalar for the other operations would repeat identical invocations rather
    than widen coverage.
    """
    cases = []
    for spec in registered_operations():
        for overload_index, overload in enumerate(spec.overloads):
            tensor_count = sum(role is OperandRole.TENSOR for role in overload.roles)
            scalar_positions = tuple(
                index
                for index, role in enumerate(overload.roles)
                if role is OperandRole.WEAK_SCALAR
            )
            if spec.name == "clamp":
                clamp_bounds = {
                    (): (),
                    (1,): (-1.0,),
                    (2,): (1.0,),
                    (1, 2): (-1.0, 1.0),
                }
                scalars = (clamp_bounds[scalar_positions],)
            else:
                scalars = WEAK_SCALARS if scalar_positions else WEAK_SCALARS[:1]
            tensor_domains = overload.tensor_domains
            domains: list[tuple[DType, ...]] = []
            tensor_index = 0
            for role in overload.roles:
                if role is OperandRole.WEAK_SCALAR:
                    continue
                domains.append(
                    (DType.Float32, DType.Int32)
                    if tensor_domains is None
                    else tuple(tensor_domains[tensor_index])
                )
                tensor_index += 1
            assert len(domains) == tensor_count
            for dtypes in product(*domains):
                for scalar in scalars:
                    cases.append((spec.name, overload_index, dtypes, scalar))
    return cases


CASES = registered_cases()


def case_id(case) -> str:
    operation, overload_index, dtypes, scalar = case
    return (
        f"{operation}[{overload_index}]-"
        f"{'x'.join(dtype.name for dtype in dtypes)}-{scalar!r}"
    )


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_generic_and_cpu_agree_on_every_registered_operation(case):
    operation, overload_index, dtypes, scalar = case
    transcendental = operation in TRANSCENDENTAL_OPERATIONS

    generic = run(operation, overload_index, dtypes, scalar, "generic")
    cpu = run(operation, overload_index, dtypes, scalar, "cpu")

    assert_conformant(
        generic,
        cpu,
        forward_tolerance=LIBM_FORWARD_TOLERANCE if transcendental else None,
        backward_tolerance=LIBM_BACKWARD_TOLERANCE if transcendental else None,
    )


@pytest.mark.parametrize(
    ("operation", "values", "expected_bits", "expected_gradient"),
    [
        (
            "reduce_max",
            (float32_from_bits(0x7FC00011), 5.0, float32_from_bits(0x7FC00022)),
            0x7FC00011,
            (float("nan"),) * 3,
        ),
        (
            "reduce_min",
            (float32_from_bits(0x7FC00011), 5.0, float32_from_bits(0x7FC00022)),
            0x7FC00011,
            (float("nan"),) * 3,
        ),
        ("reduce_max", (-0.0, 0.0), 0x00000000, (2.0, 2.0)),
        ("reduce_min", (0.0, -0.0), 0x80000000, (2.0, 2.0)),
        ("reduce_max", (3.0, 3.0, -1.0), 0x40400000, (2.0, 2.0, 0.0)),
        ("reduce_min", (3.0, 3.0, 4.0), 0x40400000, (2.0, 2.0, 0.0)),
        ("reduce_max", (float("inf"), -float("inf"), 5.0), 0x7F800000, (4.0, 0.0, 0.0)),
        ("reduce_min", (float("inf"), -float("inf"), 5.0), 0xFF800000, (0.0, 4.0, 0.0)),
    ],
)
def test_extrema_reductions_share_ordered_float32_semantics(
    operation: str,
    values: tuple[float, ...],
    expected_bits: int,
    expected_gradient: tuple[float, ...],
) -> None:
    """Pin extrema payload, tie, infinity, signed-zero, and VJP conformance."""
    outcomes = {}
    layout = Layout(Shape([1, len(values)]), Stride([1, 1]))
    for backend, make in BACKENDS.items():
        tensor = Tensor(make(values, DType.Float32), 0, layout)
        result = getattr(sw, operation)(tensor, "a b -> a")
        result_bits = float32_bits(result[0])
        result.backward(Tensor(make((4.0,), DType.Float32), 0, result.layout))
        gradient = tensor.grad
        assert gradient is not None
        outcomes[backend] = (result_bits, tuple(values_of(gradient)))

    for result_bits, gradient in outcomes.values():
        assert result_bits == expected_bits
        assert same_values(gradient, expected_gradient, tolerance=None)
    assert outcomes["cpu"][0] == outcomes["generic"][0]


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_both_backends_produce_the_resolved_plan_output(case):
    operation, overload_index, dtypes, scalar = case
    roles = specification(operation, overload_index, dtypes, scalar)
    # Tensor operands are passed to the resolver as their storage dtype and weak
    # scalars as the value itself, which is exactly what `roles` already holds.
    plan = resolve_operation_plan(operation, *(value for _, value in roles))

    for backend in BACKENDS:
        assert (
            run(operation, overload_index, dtypes, scalar, backend).dtype is plan.output
        )


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_both_backends_attach_a_graph_exactly_when_the_plan_allows_one(case):
    operation, overload_index, dtypes, scalar = case
    roles = specification(operation, overload_index, dtypes, scalar)
    plan = resolve_operation_plan(operation, *(value for _, value in roles))

    # A floating result may participate in autograd, but a node is attached
    # only when an input is itself differentiable too, so an all-Int32
    # invocation of a floating operation yields a Float32 result with no graph.
    differentiable_input = any(dtype is DType.Float32 for dtype in dtypes)
    expected = plan.output is DType.Float32 and differentiable_input

    for backend in BACKENDS:
        assert (
            run(operation, overload_index, dtypes, scalar, backend).has_graph
            is expected
        )


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_no_enumerated_case_is_vacuous(case):
    # Every case above compares two successful results. If an invocation ever
    # started raising on both backends, the comparison would still pass while
    # testing nothing, so the enumeration asserts that it did not.
    operation, overload_index, dtypes, scalar = case

    outcome = run(operation, overload_index, dtypes, scalar, "generic")

    assert outcome.error is None
    assert len(outcome.values) > 0


# Scalars the policy refuses, and the reason it gives. Both backends must
# produce the identical diagnostic, since the rejection is the policy's.
REJECTED_SCALARS = ["text", None, complex(1, 2), [1]]


@pytest.mark.parametrize("operation", ["mul", "pow"])
@pytest.mark.parametrize("dtype", [DType.Float32, DType.Int32])
@pytest.mark.parametrize("scalar", REJECTED_SCALARS, ids=repr)
def test_both_backends_refuse_the_same_scalars_identically(dtype, operation, scalar):
    outcomes = {
        backend: run(operation, 1, (dtype,), scalar, backend) for backend in BACKENDS
    }

    assert outcomes["generic"].error is TypeError
    assert outcomes["cpu"].error is TypeError
    assert outcomes["cpu"].message == outcomes["generic"].message
    assert "real Python number" in outcomes["generic"].message


def test_every_registered_operation_is_reachable_as_a_public_function():
    # Enumeration is only exhaustive if every registered name is actually
    # invoked above; a policy operation with no public entry point would
    # otherwise be silently uncovered.
    for spec in registered_operations():
        if spec.public:
            assert callable(getattr(sw, spec.name))
    assert {case[0] for case in CASES} == {
        spec.name for spec in registered_operations()
    }


# --- Non-canonical layouts --------------------------------------------------
#
# The cases above use canonical layouts, whose logical indices address every
# physical slot. A strided layout leaves holes, and a result written through
# one must still hold values its dtype can represent in those holes — a
# `Generic` carrier normalizes everything it stores, so a placeholder would be
# rejected outright for Int32 and become a silent NaN for Float32.


GAPPED_ONE_MODE = Layout(Shape(4), Stride(2))
GAPPED_TWO_MODE = Layout(Shape([2, 2]), Stride([1, 4]))
GAPPED_OFFSET = 1


def gapped_tensor(
    dtype: DType, layout: Layout, backend: str, *, accumulating: bool = False
) -> Tensor:
    """Build a tensor at a non-zero offset over a layout with storage holes."""
    zero = 0.0 if dtype is DType.Float32 else 0
    size = GAPPED_OFFSET + layout._cache.cosize
    carrier = BACKENDS[backend]((zero,) * size, dtype)
    sample = sample_for(dtype, accumulating=accumulating)
    for logical_index in range(layout.shape.logical_size):
        carrier[GAPPED_OFFSET + layout.index(logical_index)] = sample[logical_index]
    return Tensor(carrier, GAPPED_OFFSET, layout)


def tensor_with_logical_values(
    values: tuple[Any, ...],
    dtype: DType,
    layout: Layout,
    backend: str,
) -> Tensor:
    """Build a zero-offset tensor by placing values through its layout."""
    zero = 0.0 if dtype is DType.Float32 else False if dtype is DType.Bool else 0
    physical = [zero] * layout.cosize
    for logical_index, value in enumerate(values):
        physical[layout.index(logical_index)] = value
    return Tensor(BACKENDS[backend](tuple(physical), dtype), 0, layout)


GAPPED_INVOCATIONS = {
    "add": lambda tensor: tensor + tensor,
    "sub": lambda tensor: tensor - tensor,
    "elementwise_mul": lambda tensor: sw.elementwise_mul(tensor, tensor),
    "div": lambda tensor: tensor / tensor,
    "mul": lambda tensor: sw.mul(tensor, 2),
    "pow": lambda tensor: tensor**2,
    "relu": sw.relu,
    "exp": sw.exp,
    "reduce_sum": lambda tensor: sw.reduce_sum(tensor, "a b -> a"),
    "matmul": lambda tensor: tensor @ tensor,
}


@pytest.mark.parametrize("operation", sorted(GAPPED_INVOCATIONS))
@pytest.mark.parametrize("dtype", [DType.Float32, DType.Int32])
def test_backends_agree_over_a_layout_with_storage_holes(dtype, operation):
    invoke = GAPPED_INVOCATIONS[operation]
    layout = GAPPED_TWO_MODE if operation in TWO_MODE_OPERATIONS else GAPPED_ONE_MODE
    tolerance = (
        LIBM_FORWARD_TOLERANCE if operation in TRANSCENDENTAL_OPERATIONS else None
    )

    results = {
        backend: invoke(
            gapped_tensor(
                dtype,
                layout,
                backend,
                accumulating=operation in FLOATING_ACCUMULATION_OPERATIONS,
            )
        )
        for backend in BACKENDS
    }

    assert results["cpu"].dtype() is results["generic"].dtype()
    assert same_values(
        tuple(values_of(results["cpu"])),
        tuple(values_of(results["generic"])),
        tolerance=tolerance,
    )


@pytest.mark.parametrize("dtype", [DType.Float32, DType.Int32])
def test_binary_pointwise_result_uses_injective_canonical_storage(dtype):
    tensor = gapped_tensor(dtype, GAPPED_ONE_MODE, "generic")

    result = tensor + tensor

    assert result.layout == Layout(Shape(4), Stride(1))
    assert result.layout.is_injective
    assert result.carrier.size() == result.size()


def binary_pointwise_operations() -> tuple[str, ...]:
    """Enumerate tensor-pair operations whose plans have no accumulation."""
    tensor_pair = (OperandRole.TENSOR, OperandRole.TENSOR)
    exercised = {"add", "sub", "elementwise_mul", "div"}
    operations = []
    for spec in registered_operations():
        if spec.name not in exercised or not any(
            overload.roles == tensor_pair for overload in spec.overloads
        ):
            continue
        plans = (
            resolve_operation_plan(spec.name, lhs_dtype, rhs_dtype)
            for lhs_dtype, rhs_dtype in product((DType.Float32, DType.Int32), repeat=2)
        )
        if all(plan.accumulation is None for plan in plans):
            operations.append(spec.name)
    return tuple(operations)


BINARY_POINTWISE_OPERATIONS = binary_pointwise_operations()


@pytest.mark.parametrize("operation", BINARY_POINTWISE_OPERATIONS)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_every_registered_binary_pointwise_operation_aligns_broadcast_inputs(
    operation,
    backend,
):
    lhs_layout = Layout(Shape([2, 3]), Stride([1, 2]))
    rhs_layout = Layout(Shape([2, 1]), Stride([1, 2]))
    lhs = tensor_with_logical_values(
        (2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
        DType.Float32,
        lhs_layout,
        backend,
    )
    rhs = tensor_with_logical_values(
        (2.0, 4.0),
        DType.Float32,
        rhs_layout,
        backend,
    )
    compute = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "elementwise_mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
    }[operation]

    result = getattr(sw, operation)(lhs, rhs)

    expanded_rhs = (2.0, 4.0, 2.0, 4.0, 2.0, 4.0)
    expected = tuple(
        compute(lhs_value, rhs_value)
        for lhs_value, rhs_value in zip(
            (2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            expanded_rhs,
            strict=True,
        )
    )
    assert tuple(values_of(result)) == expected
    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert result.layout.is_injective


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_binary_pointwise_alignment_accepts_same_shape_with_different_strides(backend):
    lhs = tensor_with_logical_values(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        DType.Float32,
        Layout(Shape([2, 3]), Stride([1, 2])),
        backend,
    )
    rhs = tensor_with_logical_values(
        (10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        DType.Float32,
        Layout(Shape([2, 3]), Stride([3, 1])),
        backend,
    )

    result = sw.elementwise_mul(lhs, rhs)

    assert tuple(values_of(result)) == (10.0, 40.0, 90.0, 160.0, 250.0, 360.0)
    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert result.layout.is_injective


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_binary_pointwise_alignment_broadcasts_at_hierarchy_depth(backend):
    lhs_layout = Layout(Shape([2, [3, 2]]), Stride([1, [2, 6]]))
    rhs_layout = Layout(Shape([2, [1, 2]]), Stride([1, [2, 2]]))
    lhs_values = tuple(float(value) for value in range(1, 13))
    rhs_values = (10.0, 20.0, 30.0, 40.0)
    lhs = tensor_with_logical_values(lhs_values, DType.Float32, lhs_layout, backend)
    rhs = tensor_with_logical_values(rhs_values, DType.Float32, rhs_layout, backend)

    result = sw.add(lhs, rhs)

    expanded_rhs = (
        10.0,
        20.0,
        10.0,
        20.0,
        10.0,
        20.0,
        30.0,
        40.0,
        30.0,
        40.0,
        30.0,
        40.0,
    )
    assert tuple(values_of(result)) == tuple(
        lhs_value + rhs_value
        for lhs_value, rhs_value in zip(lhs_values, expanded_rhs, strict=True)
    )
    assert result.layout == Layout(
        Shape([2, [3, 2]]),
        Stride([1, [2, 6]]),
    )
    assert result.layout.is_injective


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_binary_pointwise_alignment_refuses_noncongruent_profiles(backend):
    lhs = tensor_with_logical_values(
        (1.0, 2.0, 3.0, 4.0),
        DType.Float32,
        Layout(Shape([2, 2]), Stride([1, 2])),
        backend,
    )
    rhs = tensor_with_logical_values(
        (1.0, 2.0, 3.0, 4.0),
        DType.Float32,
        Layout(Shape([[2, 2]]), Stride([[1, 2]])),
        backend,
    )

    with pytest.raises(
        ValueError,
        match="Tensor shape profiles are not congruent",
    ) as error:
        sw.add(lhs, rhs)

    assert str(error.value) == (
        "Tensor shape profiles are not congruent: "
        "lhs=(leaf, leaf), rhs=((leaf, leaf)). "
        "Insert singleton modes with rearrange so both profiles match."
    )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_binary_pointwise_alignment_refuses_incompatible_leaf_extents(backend):
    lhs = tensor_with_logical_values(
        tuple(float(value) for value in range(6)),
        DType.Float32,
        Layout(Shape([2, 3]), Stride([1, 2])),
        backend,
    )
    rhs = tensor_with_logical_values(
        tuple(float(value) for value in range(8)),
        DType.Float32,
        Layout(Shape([2, 4]), Stride([1, 2])),
        backend,
    )

    with pytest.raises(
        ValueError,
        match="Tensor extents are not broadcast-compatible",
    ) as error:
        sw.add(lhs, rhs)

    assert str(error.value) == (
        "Tensor extents are not broadcast-compatible at leaf 1 "
        "within profile (leaf, leaf): lhs=3, rhs=4"
    )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_select_broadcasts_all_operands_and_routes_branch_gradients(backend):
    condition = tensor_with_logical_values(
        (True, False),
        DType.Bool,
        Layout(Shape([2, 1]), Stride([1, 2])),
        backend,
    )
    on_true = tensor_with_logical_values(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        DType.Float32,
        Layout(Shape([2, 3]), Stride([1, 2])),
        backend,
    )
    on_false = tensor_with_logical_values(
        (10.0, 20.0, 30.0),
        DType.Float32,
        Layout(Shape([1, 3]), Stride([1, 1])),
        backend,
    )

    result = sw.select(condition, on_true, on_false)
    result.backward(ones_like(result, backend))

    assert result.dtype() is DType.Float32
    assert tuple(values_of(result)) == (1.0, 10.0, 3.0, 20.0, 5.0, 30.0)
    assert not condition.is_differentiable()
    assert on_true.grad is not None
    assert tuple(values_of(on_true.grad)) == (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    assert on_false.grad is not None
    assert tuple(values_of(on_false.grad)) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_clamp_broadcasts_tensor_bounds_and_reduces_their_gradients(backend):
    tensor = tensor_with_logical_values(
        (-2.0, 0.0, 0.5, 2.0, 3.0, -3.0),
        DType.Float32,
        Layout(Shape([2, 3]), Stride([1, 2])),
        backend,
    )
    lower = tensor_with_logical_values(
        (-1.0, -2.0),
        DType.Float32,
        Layout(Shape([2, 1]), Stride([1, 2])),
        backend,
    )
    upper = tensor_with_logical_values(
        (1.0, 1.5, 2.0),
        DType.Float32,
        Layout(Shape([1, 3]), Stride([1, 1])),
        backend,
    )

    result = sw.clamp(tensor, lower, upper)
    result.backward(ones_like(result, backend))

    assert tuple(values_of(result)) == (-1.0, 0.0, 0.5, 1.5, 2.0, -2.0)
    assert tensor.grad is not None
    assert tuple(values_of(tensor.grad)) == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert lower.grad is not None
    assert tuple(values_of(lower.grad)) == (1.0, 1.0)
    assert upper.grad is not None
    assert tuple(values_of(upper.grad)) == (0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("tensor_lower", "tensor_upper"),
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["tensor-tensor", "tensor-scalar", "scalar-tensor", "scalar-scalar"],
)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_clamp_tensor_and_scalar_overloads_route_only_tensor_gradients(
    backend, tensor_lower, tensor_upper
):
    tensor = tensor_with_logical_values(
        CLAMP_INPUT_SAMPLE, DType.Float32, ONE_MODE, backend
    )
    lower_tensor = tensor_with_logical_values(
        CLAMP_LOWER_SAMPLE, DType.Float32, ONE_MODE, backend
    )
    upper_tensor = tensor_with_logical_values(
        CLAMP_UPPER_SAMPLE, DType.Float32, ONE_MODE, backend
    )
    lower = lower_tensor if tensor_lower else -1.0
    upper = upper_tensor if tensor_upper else 1.0

    result = sw.clamp(tensor, lower, upper)
    result.backward(ones_like(result, backend))

    assert tuple(values_of(result)) == (-1.0, -0.5, 0.5, 1.0)
    assert tensor.grad is not None
    assert tuple(values_of(tensor.grad)) == (0.0, 1.0, 1.0, 0.0)
    if tensor_lower:
        assert lower_tensor.grad is not None
        assert tuple(values_of(lower_tensor.grad)) == (1.0, 0.0, 0.0, 0.0)
    else:
        assert lower_tensor.grad is None
    if tensor_upper:
        assert upper_tensor.grad is not None
        assert tuple(values_of(upper_tensor.grad)) == (0.0, 0.0, 0.0, 1.0)
    else:
        assert upper_tensor.grad is None


def test_concrete_storage_rejects_a_placeholder_rather_than_storing_nan():
    # NumPy turns None into NaN rather than raising, so Float32 storage checks
    # the type itself; Int32 already rejected it.
    with pytest.raises(TypeError, match="must be a real number"):
        Generic([None], dtype=DType.Float32)
    with pytest.raises(TypeError, match="must be an integer"):
        Generic([None], dtype=DType.Int32)


# --- Exact integer behavior -------------------------------------------------


INT32_MAX = 2**31 - 1

ONE = Layout(Shape(1), Stride(1))
PAIR = Layout(Shape([1, 2]), Stride([1, 1]))
CONTRACTION = Layout(Shape([1, 1]), Stride([1, 1]))


def int_tensor(values, layout: Layout, backend: str) -> Tensor:
    return Tensor(BACKENDS[backend](tuple(values), DType.Int32), 0, layout)


OVERFLOW_CASES = {
    "add": lambda make: make([INT32_MAX], ONE) + make([1], ONE),
    "sub": lambda make: make([-INT32_MAX], ONE) - make([2], ONE),
    "elementwise_mul": lambda make: make([50_000], ONE) * make([50_000], ONE),
    "scalar_mul": lambda make: make([INT32_MAX], ONE) * 2,
    "pow": lambda make: sw.pow(make([3], ONE), 40),
    "reduce_sum": lambda make: sw.reduce_sum(make([INT32_MAX, 1], PAIR), "a b -> a"),
    "matmul": lambda make: make([INT32_MAX], CONTRACTION) @ make([2], CONTRACTION),
    "out_of_range_scalar": lambda make: make([1], ONE) * 2**40,
}


@pytest.mark.parametrize("case", sorted(OVERFLOW_CASES))
def test_both_backends_reject_the_same_integer_overflows(case):
    invoke = OVERFLOW_CASES[case]

    for backend in BACKENDS:

        def make(values, layout, backend=backend):
            return int_tensor(values, layout, backend)

        with pytest.raises(OverflowError, match="out of int32 range"):
            invoke(make)


def test_both_backends_accumulate_integers_exactly_before_narrowing():
    # A partial sum outside Int32 range is not an error: only the finished sum
    # is narrowed, so terms that cancel are not rejected.
    for backend in BACKENDS:
        tensor = int_tensor([INT32_MAX, INT32_MAX, -INT32_MAX], PAIR, backend)
        wide = Tensor(tensor.carrier, 0, Layout(Shape([1, 3]), Stride([1, 1])))

        assert sw.reduce_sum(wide, "a b -> a")[0] == INT32_MAX


def contraction(values, backend: str) -> Tensor:
    """A 1xN Int32 tensor laid out so `@` contracts the whole second mode."""
    layout = Layout(Shape([1, len(values)]), Stride([1, 1]))
    return int_tensor(values, layout, backend)


# Contractions whose partial sums leave int64 range even though the finished
# sum is representable: a fixed-width accumulator narrower than the exact sum
# would reject these, which is the defect this pins.
CANCELLING_CONTRACTIONS = {
    "three maximal products cancel": (
        [INT32_MAX] * 6,
        [INT32_MAX, INT32_MAX, INT32_MAX, -INT32_MAX, -INT32_MAX, -INT32_MAX],
        0,
    ),
    "cancellation leaves the maximum": (
        [INT32_MAX] * 3,
        [INT32_MAX, -INT32_MAX, 1],
        INT32_MAX,
    ),
    "cancellation leaves the minimum": (
        [INT32_MAX, INT32_MAX, -(2**31)],
        [INT32_MAX, -INT32_MAX, 1],
        -(2**31),
    ),
}


@pytest.mark.parametrize("case", sorted(CANCELLING_CONTRACTIONS))
def test_both_backends_contract_cancelling_products_exactly(case):
    lhs_values, rhs_values, expected = CANCELLING_CONTRACTIONS[case]

    for backend in BACKENDS:
        result = contraction(lhs_values, backend) @ contraction(rhs_values, backend)

        assert result.dtype() is DType.Int32
        assert result[0] == expected


@pytest.mark.parametrize(
    ("lhs_values", "rhs_values"),
    [
        ([INT32_MAX, INT32_MAX], [INT32_MAX, INT32_MAX]),
        ([INT32_MAX, 1], [2, 1]),
        ([-(2**31), 1], [2, 0]),
        ([INT32_MAX] * 4, [INT32_MAX, INT32_MAX, -INT32_MAX, 1]),
    ],
)
def test_both_backends_still_reject_uncancelled_contractions(lhs_values, rhs_values):
    # A wider accumulator must not turn overflow into a wrapped result: the
    # finished sum is what gets checked, and these do not fit.
    for backend in BACKENDS:
        with pytest.raises(OverflowError, match="out of int32 range"):
            _ = contraction(lhs_values, backend) @ contraction(rhs_values, backend)


def test_relu_preserves_large_integers_without_rounding_through_float():
    layout = Layout(Shape(3), Stride(1))
    for backend in BACKENDS:
        tensor = int_tensor([INT32_MAX, INT32_MAX - 1, -INT32_MAX], layout, backend)

        assert values_of(sw.relu(tensor)) == [INT32_MAX, INT32_MAX - 1, 0]


# --- IEEE special values ----------------------------------------------------


def float_tensor(values, layout: Layout, backend: str) -> Tensor:
    return Tensor(BACKENDS[backend](tuple(values), DType.Float32), 0, layout)


def classify(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return repr(value)


def test_division_singularities_agree_in_forward_and_backward():
    classes = {}
    for backend in BACKENDS:
        lhs = float_tensor([1.0, 0.0, -1.0, 1.0], ONE_MODE, backend)
        rhs = float_tensor([0.0, 0.0, 0.0, 2.0], ONE_MODE, backend)

        result = lhs / rhs
        result.backward(ones_like(result, backend))
        lhs_grad = lhs.grad
        rhs_grad = rhs.grad
        assert lhs_grad is not None and rhs_grad is not None

        classes[backend] = (
            [classify(value) for value in values_of(result)],
            [classify(value) for value in values_of(lhs_grad)],
            [classify(value) for value in values_of(rhs_grad)],
        )

    assert classes["cpu"] == classes["generic"]
    forward, lhs_grad_classes, _ = classes["generic"]
    assert forward[:3] == ["+inf", "nan", "-inf"]
    assert lhs_grad_classes[0] == "+inf"


def test_zero_base_pow_singularities_agree_in_forward_and_backward():
    for exponent, expected in ((-1, "+inf"), (0, "1.0")):
        classes = {}
        for backend in BACKENDS:
            tensor = float_tensor([0.0, 0.0, 0.0, 0.0], ONE_MODE, backend)

            result = tensor**exponent
            result.backward(ones_like(result, backend))
            gradient = tensor.grad
            assert gradient is not None

            classes[backend] = (
                [classify(value) for value in values_of(result)],
                [classify(value) for value in values_of(gradient)],
            )

        assert classes["cpu"] == classes["generic"]
        assert classes["generic"][0][0] == expected


def test_overflowing_magnitudes_saturate_rather_than_raise():
    for backend in BACKENDS:
        tensor = float_tensor([1000.0, 0.0, 1000.0, 0.0], ONE_MODE, backend)

        assert classify(sw.exp(tensor)[0]) == "+inf"
        assert classify(sw.exp(tensor)[1]) == "1.0"


def test_scalar_multiplication_rounds_the_scalar_once_on_both_backends():
    # The scalar is materialized in binary32 before the loop and reused in
    # backward, so the gradient is the stored multiplier, not the Python float.
    for backend in BACKENDS:
        tensor = float_tensor([0.1, 0.1, 0.1, 0.1], ONE_MODE, backend)

        result = sw.mul(tensor, 0.1)
        result.backward(ones_like(result, backend))
        gradient = tensor.grad
        assert gradient is not None

        assert values_of(gradient)[0] == float(numpy.float32(0.1))


class DriftingScalar(float):
    """A real number whose ``float`` conversion changes after the first call.

    The policy does not promise how many times a backend converts a scalar, but
    it does require the scalar to be materialized once and that value reused, so
    a backend must not pick up a different number in backward than it multiplied
    by in forward.
    """

    def __new__(cls) -> Self:
        return super().__new__(cls, 2.0)

    def __init__(self) -> None:
        self.conversions = 0

    def __float__(self) -> float:
        self.conversions += 1
        return 2.0 if self.conversions == 1 else 3.0


@pytest.mark.parametrize("operation", ["mul", "pow"])
def test_a_scalar_is_materialized_once_and_reused_in_backward(operation):
    gradients = {}
    for backend in BACKENDS:
        tensor = float_tensor([1.0] * 4, ONE_MODE, backend)

        result = getattr(sw, operation)(tensor, DriftingScalar())
        result.backward(ones_like(result, backend))
        gradient = tensor.grad
        assert gradient is not None

        gradients[backend] = values_of(gradient)

    # 1 * 2 and 1 ** 2 both differentiate to 2 at x = 1; a re-converted scalar
    # would make it 3.
    assert gradients["generic"] == [2.0] * 4
    assert gradients["cpu"] == gradients["generic"]


def test_reduce_backward_broadcasts_each_row_gradient_to_its_own_row():
    # The enumerated cases pass a gradient of ones, which cannot see a row
    # ordering mistake. Distinct per-row gradients can, and they pin the order
    # the one-pass repeated-gradient construction has to preserve.
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    gradients = {}
    for backend in BACKENDS:
        tensor = float_tensor([1.0, 2.0, 3.0, 4.0], layout, backend)

        result = sw.reduce_sum(tensor, "a b -> a")
        row_gradient = Tensor(
            BACKENDS[backend]((10.0, 20.0), DType.Float32), 0, result.layout
        )
        result.backward(row_gradient)
        gradient = tensor.grad
        assert gradient is not None

        gradients[backend] = values_of(gradient)

    assert gradients["generic"] == [10.0, 20.0, 10.0, 20.0]
    assert gradients["cpu"] == gradients["generic"]


def test_matmul_gradients_accumulate_identically():
    gradients = {}
    for backend in BACKENDS:
        lhs = float_tensor([1.0, 2.0, 0.1, 4.0], TWO_MODE, backend)
        rhs = float_tensor([0.1, 3.0, 5.0, 0.2], TWO_MODE, backend)

        result = lhs @ rhs
        result.backward(ones_like(result, backend))
        lhs_grad = lhs.grad
        rhs_grad = rhs.grad
        assert lhs_grad is not None and rhs_grad is not None

        gradients[backend] = (values_of(lhs_grad), values_of(rhs_grad))

    assert gradients["cpu"] == gradients["generic"]


# --- Movement keeps values and dtype identical ------------------------------


@pytest.mark.parametrize(
    ("dtype", "values"),
    [(DType.Float32, FLOAT_SAMPLE), (DType.Int32, INT_SAMPLE)],
)
def test_values_survive_a_round_trip_between_the_backends(dtype, values):
    source = Tensor(generic_carrier(values, dtype), 0, ONE_MODE)
    # Both carriers hold the same encoding, so the expectation is the stored
    # value rather than the Python literal: 0.1 is already binary32 here.
    expected = values_of(source)
    zero = 0.0 if dtype is DType.Float32 else 0

    on_cpu = sw.move(source, CPU(4, dtype=dtype))
    assert on_cpu.dtype() is dtype
    assert values_of(on_cpu) == expected

    # Moving releases the source carrier, so read before moving back.
    back_on_generic = sw.move(on_cpu, Generic([zero] * 4, dtype=dtype))
    assert back_on_generic.dtype() is dtype
    assert values_of(back_on_generic) == expected


# --- The documented boundaries ----------------------------------------------


def test_legacy_generic_storage_is_outside_simple_promotion():
    # Legacy opaque categories keep Generic's historical Python arithmetic and
    # are deliberately not planned; the resolver says so rather than guessing.
    with pytest.raises(TypeError, match="legacy opaque storage category"):
        resolve_operation_plan("add", DType.Floating, DType.Float32)

    legacy = Tensor(Generic([0.1, 0.1, 0.1, 0.1], dtype=DType.Floating), 0, ONE_MODE)

    assert (legacy + legacy).dtype() is DType.Floating


def test_compound_operands_report_a_deferred_capability_on_both_backends():
    with pytest.raises(NotImplementedError, match="compound dtype"):
        resolve_operation_plan("add", DType.MXFP8_E4M3, DType.Float32)

    # No carrier stores a compound dtype yet, so the deferral is reported at
    # construction on both backends rather than at operation time.
    for constructor in (
        lambda: Generic([0], dtype=DType.MXFP8_E4M3),
        lambda: CPU(1, dtype=DType.MXFP8_E4M3),
    ):
        with pytest.raises(ValueError, match="compound dtype"):
            constructor()
