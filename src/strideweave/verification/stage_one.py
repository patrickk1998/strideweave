"""Device-independent Stage One certification of the public CPU oracle path."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import strideweave as sw
from strideweave import (
    CPU,
    DType,
    FileBacked,
    Generic,
    Layout,
    Shape,
    SimpleDType,
    Stride,
    Tensor,
)

from .classification import MOVEMENT_CLASSIFICATIONS, classify_cpu_kernel_plans
from .comparison import compare_float32, gamma_bound
from .model import (
    CaseDescriptor,
    ClassificationDisposition,
    Deviations,
    EvidenceRecord,
    KernelDescriptor,
    KernelPlanDescriptor,
    OracleCertificate,
    PlanKey,
    Tolerance,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
)
from .payloads import (
    AnalyticCase,
    EncodedBoolPayload,
    EncodedFloat32Payload,
    EncodedInputs,
    EncodedInt32Payload,
    adversarial_float32_payload,
    analytic_cases,
    arbitrary_float32_payload,
    exact_structural_payload,
    wide_exponent_float32_payload,
)
from .reporting import make_verification_report

ResultTransform = Callable[[str, tuple[float | int, ...]], tuple[float | int, ...]]
Payload = EncodedFloat32Payload | EncodedInt32Payload | EncodedBoolPayload


@dataclass(frozen=True, slots=True)
class StageOneResult:
    """Stage One evidence report and certificates authorizing Stage Two."""

    report: VerificationReport
    certificates: tuple[OracleCertificate, ...]


@dataclass(frozen=True, slots=True)
class _NumericalWitness:
    """Prepared deterministic inputs and tolerance for one numerical case."""

    case_id: str
    payloads: tuple[Payload, ...]
    contraction_length: int
    shapes: tuple[tuple[int, ...], ...]
    seed: int
    tolerance: Tolerance


def _dtype(name: str) -> SimpleDType:
    dtype = getattr(DType, name)
    if not isinstance(dtype, SimpleDType):
        raise TypeError(f"verification plan dtype {name!r} is not simple")
    return dtype


def _tensor(
    values: Iterable[float | int], dtype: SimpleDType, layout: Layout, cpu: bool
) -> Tensor:
    zero: bool | int | float = 0.0
    if dtype is DType.Bool:
        zero = False
    elif dtype is DType.Int32:
        zero = 0
    physical: list[float | int] = [zero] * layout.cosize
    for logical_index, value in enumerate(values):
        physical[layout.index(logical_index)] = value
    if cpu:
        carrier = CPU(layout.cosize, dtype=dtype)
        for index, value in enumerate(physical):
            carrier[index] = value
    else:
        carrier = Generic(physical, dtype=dtype)
    return Tensor(carrier, 0, layout)


def _values(tensor: Tensor) -> tuple[float | int, ...]:
    return tuple(tensor[index] for index in range(tensor.size()))


def _tensor_dtypes(plan: PlanKey) -> tuple[SimpleDType, ...]:
    dtypes = []
    for role, dtype, _ in plan.operands:
        if role != "TENSOR":
            continue
        if dtype is None:
            raise ValueError("tensor plan operand has no storage dtype")
        dtypes.append(_dtype(dtype))
    return tuple(dtypes)


def _payload_dtype_name(payload: Payload) -> str:
    if isinstance(payload, EncodedFloat32Payload):
        return "Float32"
    return "Bool" if isinstance(payload, EncodedBoolPayload) else "Int32"


def _payload(dtype: SimpleDType, values: Iterable[float | int]) -> Payload:
    materialized = tuple(values)
    if dtype is DType.Bool:
        return EncodedBoolPayload.from_values(bool(value) for value in materialized)
    if dtype is DType.Int32:
        return EncodedInt32Payload.from_values(int(value) for value in materialized)
    return EncodedFloat32Payload.from_values(materialized)


def _payloads_for_plan(
    plan: PlanKey, values: tuple[tuple[float | int, ...], ...]
) -> tuple[Payload, ...]:
    dtypes = _tensor_dtypes(plan)
    if len(dtypes) != len(values):
        raise ValueError("verification payload count does not match the execution plan")
    return tuple(
        _payload(dtype, operand) for dtype, operand in zip(dtypes, values, strict=True)
    )


def _case(
    kernel: KernelDescriptor,
    test_class: VerificationClass,
    case_id: str,
    payloads: tuple[Payload, ...],
    outcome: VerificationOutcome,
    deviations: Deviations,
    mismatches: int | None,
    *,
    k: int | None = None,
    diagnostic: str | None = None,
    plan: PlanKey | None = None,
    shapes: tuple[tuple[int, ...], ...] | None = None,
    seed: int | None = 0,
    tolerance: Tolerance | None = None,
) -> EvidenceRecord:
    hashes = EncodedInputs(payloads).input_hashes
    if plan is None:
        input_dtypes = tuple(_payload_dtype_name(item) for item in payloads)
        output_dtype = "Float32"
        accumulator_dtype = None
    else:
        input_dtypes = tuple(dtype.name for dtype in _tensor_dtypes(plan))
        output_dtype = plan.output
        accumulator_dtype = plan.accumulator_dtype
    if shapes is None:
        shapes = tuple((len(item.bits),) for item in payloads)
    return EvidenceRecord(
        stage=VerificationStage.ORACLE,
        test_class=test_class,
        case=CaseDescriptor(
            kernel.operation,
            kernel.kernel_id,
            kernel.variant,
            input_dtypes,
            output_dtype,
            shapes,
            accumulator_dtype,
            k,
            seed,
            case_id,
            plan,
        ),
        target_input_bit_hashes=hashes,
        oracle_input_bit_hashes=hashes,
        tolerance=tolerance or Tolerance(version="stage-one-v1"),
        deviations=deviations,
        mismatches=mismatches,
        outcome=outcome,
        diagnostic=diagnostic,
    )


def _error_record(
    kernel: KernelDescriptor,
    test_class: VerificationClass,
    case_id: str,
    error: RuntimeError | ValueError,
    *,
    payloads: tuple[Payload, ...],
    k: int | None = None,
    plan: PlanKey | None = None,
    shapes: tuple[tuple[int, ...], ...] | None = None,
    seed: int | None = 0,
    tolerance: Tolerance | None = None,
) -> EvidenceRecord:
    """Record a recoverable execution failure with its prepared case context."""
    return _case(
        kernel,
        test_class,
        case_id,
        payloads,
        VerificationOutcome.ERROR,
        Deviations(None, None, None),
        None,
        diagnostic=f"{type(error).__name__}: {error}",
        k=k,
        plan=plan,
        shapes=shapes,
        seed=seed,
        tolerance=tolerance,
    )


def _operation_options(plan: PlanKey) -> dict[str, SimpleDType]:
    if plan.accumulator_dtype is None:
        return {}
    return {"accumulator_dtype": _dtype(plan.accumulator_dtype)}


def _plan_id(plan: PlanKey) -> str:
    operands = tuple(
        dtype if role == "TENSOR" else f"weak-{convert_to}"
        for role, dtype, convert_to in plan.operands
    )
    parts = (*operands, plan.compute, plan.accumulation, plan.accumulator_dtype)
    return "-".join(part.lower() for part in parts if part is not None)


def _weak_scalar(plan: PlanKey) -> float | int:
    weak_operands = [
        convert_to for role, _, convert_to in plan.operands if role == "WEAK_SCALAR"
    ]
    if len(weak_operands) != 1:
        raise ValueError("expected one weak scalar in this execution plan")
    return _weak_scalar_value(weak_operands[0], 0)


def _weak_scalar_value(convert_to: str | None, position: int) -> float | int:
    """Return the pinned weak scalar one plan position materializes.

    Weak scalars select a plan rather than carrying a dtype, so the value is
    fixed per position: a bound pair uses distinct values so a `clamp` case
    exercises an ordered interval rather than a degenerate point.
    """
    if convert_to == "Int32":
        return (2, -1)[position % 2]
    return (0.5, -0.5)[position % 2]


def _call_arguments(plan: PlanKey, tensors: tuple[Tensor, ...]) -> tuple[object, ...]:
    """Interleave tensor and weak scalar operands in the operation's own order."""
    arguments: list[object] = []
    tensor_index = 0
    weak_index = 0
    for role, _, convert_to in plan.operands:
        if role == "TENSOR":
            arguments.append(tensors[tensor_index])
            tensor_index += 1
            continue
        arguments.append(_weak_scalar_value(convert_to, weak_index))
        weak_index += 1
    return tuple(arguments)


# The operations that combine one two-mode tensor along its second mode. Each
# is addressed through its public description form so the verified path is the
# one a caller uses rather than a private two-mode primitive.
DESCRIBED_REDUCTIONS = frozenset(
    {
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
    }
)

# The operations whose kernel combines many terms, and which therefore need a
# two-mode contraction layout rather than the flat elementwise one.
CONTRACTING_OPERATIONS = DESCRIBED_REDUCTIONS | {
    "matmul",
    "cumsum",
    "conv_general",
}

# Selection operations are dispatched by their internal names because `sort`
# and `topk` package a value call and an index call as one public result.
_SELECTION_OPERATIONS = frozenset(
    {"_sort_values", "_sort_indices", "_topk_values", "_topk_indices"}
)

# Operations whose second operand is a logical index rather than a value, so
# its payload must stay inside the addressed extent.
INDEX_OPERAND_POSITIONS: dict[str, tuple[int, ...]] = {
    "gather": (1,),
    "scatter": (1,),
    "scatter_add": (1,),
}


@dataclass(frozen=True, slots=True)
class _CaseShape:
    """The layouts, element counts, and reported shapes one case executes on."""

    operand_layouts: tuple[Layout, ...]
    counts: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    contraction_length: int | None


def _flat(count: int) -> Layout:
    return Layout(Shape(count), Stride(1))


def _two_mode(rows: int, columns: int) -> Layout:
    return Layout(Shape([rows, columns]), Stride([1, rows]))


def _case_shape(operation: str, operand_count: int, k: int) -> _CaseShape:
    """Return the case geometry one operation's public call accepts.

    Args:
        operation: Dispatch name of the operation under certification.
        operand_count: Number of tensor operands its plan declares.
        k: Contraction or fiber length for the operations that combine terms.

    Returns:
        The per-operand layouts, element counts, reported shapes, and
        contraction length for one case.
    """
    if operation == "conv_general":
        lhs = Layout(Shape([1, 1, k]), Stride([1, 1, 1]))
        kernel = Layout(Shape([1, 1, 1]), Stride([1, 1, 1]))
        return _CaseShape((lhs, kernel), (k, 1), ((1, 1, k), (1, 1, 1)), k)
    if operation == "gather":
        source = _two_mode(2, 2)
        return _CaseShape((source, _flat(1)), (4, 1), ((2, 2), (1,)), None)
    if operation in {"scatter", "scatter_add"}:
        base = _two_mode(2, 2)
        updates = Layout(Shape([1, 2]), Stride([1, 1]))
        return _CaseShape(
            (base, _flat(1), updates), (4, 1, 2), ((2, 2), (1,), (1, 2)), None
        )
    if operation in _SELECTION_OPERATIONS:
        layout = _two_mode(2, 2)
        return _CaseShape(
            (layout,) * operand_count, (4,) * operand_count, ((2, 2),), None
        )
    if operation in CONTRACTING_OPERATIONS:
        layout = Layout(Shape([1, k]), Stride([1, 1]))
        return _CaseShape(
            (layout,) * operand_count,
            (k,) * operand_count,
            ((1, k),) * operand_count,
            k,
        )
    layout = _flat(4)
    return _CaseShape(
        (layout,) * operand_count, (4,) * operand_count, ((4,),) * operand_count, None
    )


def _dispatched(operation: str, tensors: tuple[Tensor, ...], *arguments: object):
    """Invoke an internal operation through the carrier that owns it."""
    return tensors[0].carrier.dispatch_op(operation).forward(*tensors, *arguments)


def _execute(
    descriptor: KernelPlanDescriptor,
    payloads: tuple[Payload, ...],
    layout: Layout,
    cpu: bool,
    *,
    operand_layouts: tuple[Layout, ...] | None = None,
) -> tuple[float | int, ...]:
    dtypes = _tensor_dtypes(descriptor.plan)
    layouts = operand_layouts or (layout,) * len(dtypes)
    tensors = tuple(
        _tensor(payload.values(), dtype, operand_layout, cpu)
        for payload, dtype, operand_layout in zip(
            payloads, dtypes, layouts, strict=True
        )
    )
    operation = descriptor.kernel.operation
    options = _operation_options(descriptor.plan)
    if operation in DESCRIBED_REDUCTIONS:
        result = getattr(sw, operation)(tensors[0], "a b -> a", **options)
    elif operation == "matmul":
        result = sw.matmul(*tensors, **options)
    elif operation == "cumsum":
        result = sw.cumsum(tensors[0], 1)
    elif operation == "conv_general":
        result = _dispatched(operation, tensors, (1,), ((0, 0),))
    elif operation == "gather":
        result = sw.gather(tensors[0], tensors[1], 0)
    elif operation in {"scatter", "scatter_add"}:
        result = getattr(sw, operation)(*tensors, 0)
    elif operation in {"_sort_values", "_sort_indices"}:
        result = _dispatched(operation, tensors, 1, False)
    elif operation in {"_topk_values", "_topk_indices"}:
        result = _dispatched(operation, tensors, 1, 1, True)
    else:
        result = getattr(sw, operation)(*_call_arguments(descriptor.plan, tensors))
    return _values(result)


def _comparison(
    expected: tuple[float | int, ...], actual: tuple[float | int, ...], output: str
) -> tuple[bool, Deviations, int]:
    if output == "Float32":
        comparison = compare_float32(expected, actual)
        return comparison.mismatches == 0, comparison.deviations, comparison.mismatches
    matches = expected == actual
    return (
        matches,
        Deviations(
            0.0 if matches else float("inf"), 0.0 if matches else float("inf"), 0
        ),
        0 if matches else 1,
    )


def _pinned_payloads(descriptor: KernelPlanDescriptor) -> tuple[Payload, ...]:
    """Prepare the small signed witness every exact plan is checked on first."""
    shape = _descriptor_shape(descriptor)
    operation = descriptor.kernel.operation
    index_positions = INDEX_OPERAND_POSITIONS.get(operation, ())
    values: list[tuple[float | int, ...]] = []
    for position, count in enumerate(shape.counts):
        if position in index_positions:
            values.append((0,) * count)
            continue
        pinned = (-2, -1, 1, 2)
        values.append(tuple(pinned[index % len(pinned)] for index in range(count)))
    return _payloads_for_plan(descriptor.plan, tuple(values))


def _descriptor_shape(descriptor: KernelPlanDescriptor) -> _CaseShape:
    return _case_shape(
        descriptor.kernel.operation, len(_tensor_dtypes(descriptor.plan)), 4
    )


def _exact_witness_record(
    descriptor: KernelPlanDescriptor,
    payloads: tuple[Payload, ...],
    transform: ResultTransform | None,
    *,
    case_suffix: str,
    seed: int | None = 0,
    tolerance: Tolerance | None = None,
) -> EvidenceRecord:
    """Run one encoded exact witness against Generic and CPU.

    Exact witnesses differ only in how their payloads and evidence metadata are
    prepared. Keeping execution, optional mutation, comparison, and evidence
    construction here makes those paths share the same contract.
    """
    shape = _descriptor_shape(descriptor)
    expected = _execute(
        descriptor,
        payloads,
        shape.operand_layouts[0],
        False,
        operand_layouts=shape.operand_layouts,
    )
    actual = _execute(
        descriptor,
        payloads,
        shape.operand_layouts[0],
        True,
        operand_layouts=shape.operand_layouts,
    )
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    matches, deviations, mismatches = _comparison(
        expected, actual, descriptor.plan.output
    )
    return _case(
        descriptor.kernel,
        VerificationClass.EXACT_ARITHMETIC,
        f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-{case_suffix}",
        payloads,
        VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
        deviations,
        mismatches,
        k=shape.contraction_length,
        plan=descriptor.plan,
        shapes=shape.shapes,
        seed=seed,
        tolerance=tolerance,
    )


def _exact_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    """Run the fixed signed exact witness for one active kernel plan."""
    return _exact_witness_record(
        descriptor,
        _pinned_payloads(descriptor),
        transform,
        case_suffix="exact",
    )


def _arbitrary_exact_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    seed = 1000 + sum(ord(character) for character in _plan_id(descriptor.plan))
    """Run the seeded arbitrary finite exact witness for one active plan."""
    return _exact_witness_record(
        descriptor,
        _arbitrary_payloads(descriptor, seed),
        transform,
        case_suffix="arbitrary-finite",
        seed=seed,
        tolerance=Tolerance(version="bit-exact-arbitrary-finite-v1"),
    )


def _arbitrary_payloads(
    descriptor: KernelPlanDescriptor, seed: int
) -> tuple[Payload, ...]:
    """Prepare the deterministic arbitrary exact witness before execution."""
    shape = _descriptor_shape(descriptor)
    operation = descriptor.kernel.operation
    index_positions = INDEX_OPERAND_POSITIONS.get(operation, ())
    payloads: list[Payload] = []
    for index, dtype in enumerate(_tensor_dtypes(descriptor.plan)):
        operand_seed = seed + index * 101
        count = shape.counts[index]
        generator = random.Random(operand_seed)
        if index in index_positions:
            # A logical index must address the operation's own extent, so it is
            # drawn from that extent rather than from the arbitrary range.
            payloads.append(
                EncodedInt32Payload.from_values(
                    tuple(generator.randrange(2) for _ in range(count))
                )
            )
            continue
        if dtype is DType.Bool:
            payloads.append(
                EncodedBoolPayload.from_values(
                    bool(generator.getrandbits(1)) for _ in range(count)
                )
            )
            continue
        if dtype is DType.Float32:
            payloads.append(arbitrary_float32_payload(operand_seed, count))
            continue
        values = tuple(generator.randint(-10_000, 10_000) for _ in range(count))
        if operation == "div" and index == 1:
            values = tuple(value if value != 0 else 1 for value in values)
        payloads.append(EncodedInt32Payload.from_values(values))
    return tuple(payloads)


def _contraction_payloads(
    descriptor: KernelPlanDescriptor,
    lhs: tuple[float | int, ...],
    rhs: tuple[float | int, ...] | None,
) -> tuple[Payload, ...]:
    values = (lhs,) if rhs is None else (lhs, rhs)
    return _payloads_for_plan(descriptor.plan, values)


# The exactly representable magnitudes a structural payload draws from. Every
# partial sum and product of four of these stays exactly representable in
# binary32, so a structural comparison checks traversal and addressing without
# depending on which association a backend chooses.
_EXACT_MAGNITUDES = (1.0, -2.0, 4.0, -8.0, 2.0, -1.0, 8.0, -4.0)


def _structural_payloads(descriptor: KernelPlanDescriptor) -> tuple[Payload, ...]:
    """Prepare payloads whose every legal partial result is exact."""
    operation = descriptor.kernel.operation
    shape = _descriptor_shape(descriptor)
    if operation in {"reduce_sum", "matmul"}:
        source = exact_structural_payload(17, 4, product=operation == "matmul")
        lhs = tuple(int(value) for value in source.lhs.values())
        rhs = (
            None
            if source.rhs is None
            else tuple(int(value) for value in source.rhs.values())
        )
        return _contraction_payloads(descriptor, lhs, rhs)
    index_positions = INDEX_OPERAND_POSITIONS.get(operation, ())
    values: list[tuple[float | int, ...]] = []
    for position, count in enumerate(shape.counts):
        if position in index_positions:
            values.append(tuple(index % 2 for index in range(count)))
            continue
        offset = position * 3
        values.append(
            tuple(
                _EXACT_MAGNITUDES[(offset + index) % len(_EXACT_MAGNITUDES)]
                for index in range(count)
            )
        )
    return _payloads_for_plan(descriptor.plan, tuple(values))


def _structural_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    shape = _descriptor_shape(descriptor)
    k = shape.contraction_length
    payloads = _structural_payloads(descriptor)
    expected = _execute(
        descriptor,
        payloads,
        shape.operand_layouts[0],
        False,
        operand_layouts=shape.operand_layouts,
    )
    actual = _execute(
        descriptor,
        payloads,
        shape.operand_layouts[0],
        True,
        operand_layouts=shape.operand_layouts,
    )
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    matches, deviations, mismatches = _comparison(
        expected, actual, descriptor.plan.output
    )
    return _case(
        descriptor.kernel,
        VerificationClass.STRUCTURAL,
        f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-structural",
        payloads,
        VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
        deviations,
        mismatches,
        k=k,
        diagnostic=None
        if matches
        else "encoded structural results are not bit-identical",
        plan=descriptor.plan,
        shapes=shape.shapes,
        seed=17,
    )


def _analytic_layout_case(operation: str) -> AnalyticCase:
    if operation == "reduce_sum":
        return AnalyticCase(
            "reduce-hierarchical-addressing",
            operation,
            ((1.0, 2.0, 4.0, 8.0),),
            (15.0,),
        )
    return AnalyticCase(
        "matmul-hierarchical-addressing",
        operation,
        ((1.0, 2.0, 4.0, 8.0), (8.0, 4.0, 2.0, 1.0)),
        (32.0,),
    )


def _analytic_cases_for(operation: str) -> tuple[AnalyticCase, ...]:
    """Return the independently executable analytic witnesses for an operation."""
    return (
        *[case for case in analytic_cases() if case.operation == operation],
        _analytic_layout_case(operation),
    )


def _analytic_record(
    descriptor: KernelPlanDescriptor,
    transform: ResultTransform | None,
    case: AnalyticCase,
) -> EvidenceRecord:
    """Execute one analytic witness so failures do not suppress later witnesses."""
    operation = descriptor.kernel.operation
    if case.operation != operation:
        raise ValueError("analytic witness operation does not match execution plan")
    k = len(case.inputs[0])
    layout = (
        Layout(Shape([1, [2, 2]]), Stride([1, [5, 2]]))
        if case.case_id.endswith("hierarchical-addressing")
        else Layout(Shape([1, k]), Stride([1, 1]))
    )
    payloads = _contraction_payloads(
        descriptor,
        case.inputs[0],
        case.inputs[1] if len(case.inputs) == 2 else None,
    )
    expected = _execute(descriptor, payloads, layout, False)
    actual = _execute(descriptor, payloads, layout, True)
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    generic_matches, _, _ = _comparison(case.expected, expected, descriptor.plan.output)
    target_matches, deviations, mismatches = _comparison(
        case.expected, actual, descriptor.plan.output
    )
    matches = generic_matches and target_matches
    return _case(
        descriptor.kernel,
        VerificationClass.ANALYTIC,
        f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-{case.case_id}",
        payloads,
        VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
        deviations,
        mismatches,
        k=k,
        diagnostic=None
        if matches
        else "result disagrees with the independent analytic witness",
        plan=descriptor.plan,
        shapes=((1, k),) * len(payloads),
    )


def _controlled_wide_payload(seed: int, count: int) -> EncodedFloat32Payload:
    source = wide_exponent_float32_payload(seed, count)
    exponents = (-10, -7, -4, -1, 1, 4, 7, 10)
    if count != len(exponents):
        raise ValueError("controlled wide payload requires eight terms")
    values = []
    for bits, exponent in zip(source.bits, exponents, strict=True):
        sign = -1.0 if bits & (1 << 31) else 1.0
        significand = 1.0 + (bits & 0x007F_FFFF) / 2**23
        values.append(math.ldexp(sign * significand, exponent))
    return EncodedFloat32Payload.from_values(values)


def _numerical_witness(descriptor: KernelPlanDescriptor) -> _NumericalWitness:
    """Prepare the shared deterministic witness for normal and error evidence."""
    k = 8
    dtypes = _tensor_dtypes(descriptor.plan)
    prefix = f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}"
    if all(dtype is DType.Int32 for dtype in dtypes):
        payloads = _payloads_for_plan(
            descriptor.plan, tuple(tuple(range(1, k + 1)) for _ in dtypes)
        )
        return _NumericalWitness(
            f"{prefix}-numerical-integer-exact",
            payloads,
            k,
            ((1, k),) * len(payloads),
            73,
            Tolerance(version="exact-integer-v1"),
        )

    wide = _controlled_wide_payload(73, k)
    float_operand = next(
        index for index, dtype in enumerate(dtypes) if dtype is DType.Float32
    )
    payloads = _payloads_for_plan(
        descriptor.plan,
        tuple(
            wide.values() if index == float_operand else (1,) * k
            for index in range(len(dtypes))
        ),
    )
    if descriptor.kernel.operation == "reduce_sum":
        terms = payloads[0].values()
    else:
        terms = tuple(
            lhs * rhs
            for lhs, rhs in zip(payloads[0].values(), payloads[1].values(), strict=True)
        )
    unit_roundoff = (
        2.0**-53 if descriptor.plan.accumulator_dtype == "Float64" else 2.0**-24
    )
    # Both Generic and CPU may choose any legal association. Each path is
    # bounded by gamma-K, so their pairwise separation is bounded by twice it.
    allowed = 2.0 * gamma_bound(unit_roundoff, k, sum(abs(term) for term in terms))
    return _NumericalWitness(
        f"{prefix}-numerical-wide-exponents",
        payloads,
        k,
        ((1, k),) * len(payloads),
        73,
        Tolerance(absolute=allowed, version="stage-one-two-path-gamma-v1"),
    )


def _numerical_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    witness = _numerical_witness(descriptor)
    layout = Layout(
        Shape([1, witness.contraction_length]),
        Stride([1, 1]),
    )
    if all(dtype is DType.Int32 for dtype in _tensor_dtypes(descriptor.plan)):
        expected = _execute(descriptor, witness.payloads, layout, False)
        actual = _execute(descriptor, witness.payloads, layout, True)
        if transform is not None:
            actual = transform(descriptor.kernel.kernel_id, actual)
        matches, deviations, mismatches = _comparison(
            expected, actual, descriptor.plan.output
        )
        return _case(
            descriptor.kernel,
            VerificationClass.NUMERICAL,
            witness.case_id,
            witness.payloads,
            VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
            deviations,
            mismatches,
            k=witness.contraction_length,
            diagnostic=None if matches else "integer numerical result is not exact",
            plan=descriptor.plan,
            shapes=witness.shapes,
            seed=witness.seed,
            tolerance=witness.tolerance,
        )
    expected = _execute(descriptor, witness.payloads, layout, False)
    actual = _execute(descriptor, witness.payloads, layout, True)
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    comparison = compare_float32(expected, actual)
    maximum_absolute = comparison.deviations.maximum_absolute
    if maximum_absolute is None:
        raise RuntimeError("completed numerical comparison has no absolute deviation")
    passed = maximum_absolute <= witness.tolerance.absolute
    return _case(
        descriptor.kernel,
        VerificationClass.NUMERICAL,
        witness.case_id,
        witness.payloads,
        VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED,
        comparison.deviations,
        comparison.mismatches,
        k=witness.contraction_length,
        diagnostic=None
        if passed
        else f"maximum absolute deviation exceeds {witness.tolerance.absolute}",
        plan=descriptor.plan,
        shapes=witness.shapes,
        seed=witness.seed,
        tolerance=witness.tolerance,
    )


def _error_context(
    descriptor: KernelPlanDescriptor,
    label: str,
    *,
    analytic_case: AnalyticCase | None = None,
) -> tuple[
    str,
    tuple[Payload, ...],
    int | None,
    tuple[tuple[int, ...], ...],
    int | None,
    Tolerance | None,
]:
    """Prepare reproducible evidence metadata for one recoverable case failure."""
    prefix = f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}"
    shape = _descriptor_shape(descriptor)
    if label == "exact":
        return (
            f"{prefix}-exact-error",
            _pinned_payloads(descriptor),
            shape.contraction_length,
            shape.shapes,
            0,
            None,
        )
    if label == "arbitrary":
        seed = 1000 + sum(ord(character) for character in _plan_id(descriptor.plan))
        return (
            f"{prefix}-arbitrary-finite-error",
            _arbitrary_payloads(descriptor, seed),
            shape.contraction_length,
            shape.shapes,
            seed,
            Tolerance(version="bit-exact-arbitrary-finite-v1"),
        )
    if label == "structural":
        shape = _descriptor_shape(descriptor)
        payloads = _structural_payloads(descriptor)
        return (
            f"{prefix}-structural-error",
            payloads,
            shape.contraction_length,
            shape.shapes,
            17,
            None,
        )
    if label == "numerical":
        witness = _numerical_witness(descriptor)
        return (
            f"{prefix}-numerical-error",
            witness.payloads,
            witness.contraction_length,
            witness.shapes,
            witness.seed,
            witness.tolerance,
        )
    if label == "analytic" and analytic_case is not None:
        k = len(analytic_case.inputs[0])
        payloads = _contraction_payloads(
            descriptor,
            analytic_case.inputs[0],
            analytic_case.inputs[1] if len(analytic_case.inputs) == 2 else None,
        )
        return (
            f"{prefix}-{analytic_case.case_id}-error",
            payloads,
            k,
            ((1, k),) * len(payloads),
            0,
            None,
        )
    raise ValueError(f"unknown Stage One recoverable case label {label!r}")


def _recoverable_error_record(
    descriptor: KernelPlanDescriptor,
    test_class: VerificationClass,
    label: str,
    error: RuntimeError | ValueError,
    *,
    analytic_case: AnalyticCase | None = None,
) -> EvidenceRecord:
    """Convert a recoverable execution error into complete durable evidence."""
    case_id, payloads, k, shapes, seed, tolerance = _error_context(
        descriptor, label, analytic_case=analytic_case
    )
    return _error_record(
        descriptor.kernel,
        test_class,
        case_id,
        error,
        payloads=payloads,
        k=k,
        plan=descriptor.plan,
        shapes=shapes,
        seed=seed,
        tolerance=tolerance,
    )


def _required_classes(
    descriptor: KernelPlanDescriptor,
) -> tuple[VerificationClass, ...]:
    return descriptor.classes


def _movement_comparison(
    expected: EncodedFloat32Payload, actual: EncodedFloat32Payload
) -> bool:
    """Return whether a movement result preserves every encoded Float32 bit."""
    return actual.bits == expected.bits


def _movement_records() -> tuple[EvidenceRecord, ...]:
    payload = adversarial_float32_payload()
    base_layout = Layout(Shape([2, 5]), Stride([1, 2]))

    def source() -> Tensor:
        return _tensor(payload.values(), DType.Float32, base_layout, True)

    operations = {
        "move": lambda tensor: (
            sw.move(tensor, FileBacked(dtype=DType.Float32)),
            payload.values(),
        ),
        # `view` is a dispatched structural operation rather than a public
        # function, so it is exercised through the carrier that owns it.
        "view": lambda tensor: (
            tensor.carrier.dispatch_op("view").forward(
                tensor, (slice(None), slice(1, 5))
            ),
            tuple(tensor[row, column] for column in range(1, 5) for row in range(2)),
        ),
        "permute": lambda tensor: (
            sw.permute(tensor, 1, 0),
            tuple(tensor[row, column] for row in range(2) for column in range(5)),
        ),
        "rearrange": lambda tensor: (
            sw.rearrange(tensor, "a b -> b a"),
            tuple(tensor[row, column] for row in range(2) for column in range(5)),
        ),
        "broadcast_to": lambda tensor: (
            sw.broadcast_to(tensor, Shape([2, 10])),
            tuple(tensor[0, column] for column in range(10) for _ in range(2)),
        ),
    }
    broadcast_source_layout = Layout(Shape([1, 10]), Stride([1, 1]))
    source_shapes = {
        "broadcast_to": ((1, 10),),
        "move": ((2, 5),),
        "view": ((2, 5),),
        "permute": ((2, 5),),
        "rearrange": ((2, 5),),
    }
    missing_cases = set(MOVEMENT_CLASSIFICATIONS) - set(operations)
    if missing_cases:
        operation = min(missing_cases)
        raise ValueError(
            f"movement classification {operation!r} has no verification case"
        )
    if set(operations) != set(MOVEMENT_CLASSIFICATIONS):
        raise ValueError("movement verification cases do not match classifications")
    records = []
    for operation in MOVEMENT_CLASSIFICATIONS:
        execute = operations[operation]
        kernel = KernelDescriptor(
            operation, f"movement.{operation}", "default", operation
        )
        source_shape = source_shapes[operation]
        prepared_error = _case(
            kernel,
            VerificationClass.BIT_EXACT,
            f"movement-{operation}-adversarial-bits",
            (payload,),
            VerificationOutcome.ERROR,
            Deviations(None, None, None),
            None,
            shapes=source_shape,
        )
        try:
            tensor = (
                _tensor(payload.values(), DType.Float32, broadcast_source_layout, True)
                if operation == "broadcast_to"
                else source()
            )
            result, expected_values = execute(tensor)
            actual = EncodedFloat32Payload.from_values(_values(result))
            expected = EncodedFloat32Payload.from_values(expected_values)
            passed = _movement_comparison(expected, actual)
        except (RuntimeError, ValueError) as error:
            records.append(
                replace(
                    prepared_error,
                    diagnostic=f"{type(error).__name__}: {error}",
                )
            )
            continue
        records.append(
            _case(
                kernel,
                VerificationClass.BIT_EXACT,
                f"movement-{operation}-adversarial-bits",
                (payload,),
                VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED,
                Deviations(
                    0.0 if passed else float("inf"),
                    0.0 if passed else float("inf"),
                    0,
                ),
                0 if passed else 1,
                diagnostic=None
                if passed
                else f"{operation} changed encoded Float32 bits or logical mapping",
                shapes=source_shape,
            )
        )
    return tuple(records)


def _unique_classes(
    requirements: Iterable[tuple[PlanKey, tuple[VerificationClass, ...]]],
) -> tuple[VerificationClass, ...]:
    classes: list[VerificationClass] = []
    for _, plan_classes in requirements:
        for test_class in plan_classes:
            if test_class not in classes:
                classes.append(test_class)
    return tuple(classes)


def run_stage_one(result_transform: ResultTransform | None = None) -> StageOneResult:
    """Certify the Generic/CPU oracle pair for every active CPU plan.

    Args:
        result_transform: Optional test hook that mutates CPU results before
            comparison, allowing failure evidence to be exercised.

    Returns:
        Stage One evidence report and certificates for plans that passed.

    Examples:
        >>> result = run_stage_one()
        >>> bool(result.report.records)
        True
    """
    descriptors = classify_cpu_kernel_plans()
    records: list[EvidenceRecord] = list(_movement_records())
    records_by_kernel: dict[KernelDescriptor, list[EvidenceRecord]] = {}
    requirements_by_kernel: dict[
        KernelDescriptor, list[tuple[PlanKey, tuple[VerificationClass, ...]]]
    ] = {}
    for descriptor in descriptors:
        kernel_records = records_by_kernel.setdefault(descriptor.kernel, [])
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            record = _case(
                descriptor.kernel,
                VerificationClass.DEFERRED,
                f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-deferred",
                (),
                VerificationOutcome.DEFERRED,
                Deviations(0.0, 0.0, 0),
                0,
                diagnostic=descriptor.deferred_reason,
                plan=descriptor.plan,
                seed=None,
            )
            records.append(record)
            kernel_records.append(record)
            continue
        required = _required_classes(descriptor)
        descriptor_records = []
        # The required classes decide which cases run, so a kernel added to the
        # manifest is covered by its classification rather than by an
        # operation-name branch that would silently skip it.
        case_functions = (
            (VerificationClass.EXACT_ARITHMETIC, "exact", _exact_record),
            (VerificationClass.EXACT_ARITHMETIC, "arbitrary", _arbitrary_exact_record),
            (VerificationClass.STRUCTURAL, "structural", _structural_record),
            (VerificationClass.NUMERICAL, "numerical", _numerical_record),
        )
        for test_class, label, case_function in case_functions:
            if test_class not in required:
                continue
            try:
                result = case_function(descriptor, result_transform)
                descriptor_records.extend(
                    result if isinstance(result, tuple) else (result,)
                )
            except (RuntimeError, ValueError) as error:
                descriptor_records.append(
                    _recoverable_error_record(descriptor, test_class, label, error)
                )
        if VerificationClass.ANALYTIC in required:
            for analytic_case in _analytic_cases_for(descriptor.kernel.operation):
                try:
                    descriptor_records.append(
                        _analytic_record(descriptor, result_transform, analytic_case)
                    )
                except (RuntimeError, ValueError) as error:
                    descriptor_records.append(
                        _recoverable_error_record(
                            descriptor,
                            VerificationClass.ANALYTIC,
                            "analytic",
                            error,
                            analytic_case=analytic_case,
                        )
                    )
        records.extend(descriptor_records)
        kernel_records.extend(descriptor_records)
        requirements_by_kernel.setdefault(descriptor.kernel, []).append(
            (descriptor.plan, required)
        )

    certificates = []
    for kernel, requirements in requirements_by_kernel.items():
        try:
            certificates.append(
                OracleCertificate.from_records(
                    kernel,
                    _unique_classes(requirements),
                    tuple(records_by_kernel[kernel]),
                    required_plan_classes=tuple(requirements),
                )
            )
        except ValueError:
            pass
    certificate_tuple = tuple(certificates)
    return StageOneResult(
        make_verification_report(tuple(records), certificate_tuple), certificate_tuple
    )
