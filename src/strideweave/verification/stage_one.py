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
    EncodedFloat32Payload,
    EncodedInputs,
    EncodedInt32Payload,
    adversarial_float32_payload,
    analytic_cases,
    arbitrary_float32_payload,
    exact_structural_payload,
    wide_exponent_float32_payload,
)

ResultTransform = Callable[[str, tuple[float | int, ...]], tuple[float | int, ...]]
Payload = EncodedFloat32Payload | EncodedInt32Payload


@dataclass(frozen=True, slots=True)
class StageOneResult:
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
    zero = 0 if dtype is DType.Int32 else 0.0
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


def _payload(dtype: SimpleDType, values: Iterable[float | int]) -> Payload:
    materialized = tuple(values)
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
        input_dtypes = tuple(
            "Float32" if isinstance(item, EncodedFloat32Payload) else "Int32"
            for item in payloads
        )
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
    return 2 if weak_operands[0] == "Int32" else 0.5


def _execute(
    descriptor: KernelPlanDescriptor,
    payloads: tuple[Payload, ...],
    layout: Layout,
    cpu: bool,
) -> tuple[float | int, ...]:
    tensors = tuple(
        _tensor(payload.values(), dtype, layout, cpu)
        for payload, dtype in zip(
            payloads, _tensor_dtypes(descriptor.plan), strict=True
        )
    )
    operation = descriptor.kernel.operation
    options = _operation_options(descriptor.plan)
    if operation in {"add", "sub", "elementwise_mul", "div"}:
        result = getattr(sw, operation)(*tensors)
    elif operation in {"mul", "pow"}:
        result = getattr(sw, operation)(tensors[0], _weak_scalar(descriptor.plan))
    elif operation == "reduce":
        result = sw.reduce(tensors[0], **options)
    elif operation == "matmul":
        result = sw.matmul(*tensors, **options)
    else:
        result = getattr(sw, operation)(tensors[0])
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


def _exact_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    values = tuple((-2, -1, 1, 2) for _ in _tensor_dtypes(descriptor.plan))
    payloads = _payloads_for_plan(descriptor.plan, values)
    layout = Layout(Shape(4), Stride(1))
    expected = _execute(descriptor, payloads, layout, False)
    actual = _execute(descriptor, payloads, layout, True)
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    matches, deviations, mismatches = _comparison(
        expected, actual, descriptor.plan.output
    )
    return _case(
        descriptor.kernel,
        VerificationClass.EXACT_ARITHMETIC,
        f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-exact",
        payloads,
        VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
        deviations,
        mismatches,
        plan=descriptor.plan,
        shapes=((4,),) * len(payloads),
    )


def _arbitrary_exact_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    seed = 1000 + sum(ord(character) for character in _plan_id(descriptor.plan))
    materialized = _arbitrary_payloads(descriptor, seed)
    layout = Layout(Shape(4), Stride(1))
    expected = _execute(descriptor, materialized, layout, False)
    actual = _execute(descriptor, materialized, layout, True)
    if transform is not None:
        actual = transform(descriptor.kernel.kernel_id, actual)
    matches, deviations, mismatches = _comparison(
        expected, actual, descriptor.plan.output
    )
    return _case(
        descriptor.kernel,
        VerificationClass.EXACT_ARITHMETIC,
        f"{descriptor.kernel.kernel_id}-{_plan_id(descriptor.plan)}-arbitrary-finite",
        materialized,
        VerificationOutcome.PASSED if matches else VerificationOutcome.FAILED,
        deviations,
        mismatches,
        plan=descriptor.plan,
        shapes=((4,),) * len(materialized),
        seed=seed,
        tolerance=Tolerance(version="bit-exact-arbitrary-finite-v1"),
    )


def _arbitrary_payloads(
    descriptor: KernelPlanDescriptor, seed: int
) -> tuple[Payload, ...]:
    """Prepare the deterministic arbitrary exact witness before execution."""
    payloads: list[Payload] = []
    for index, dtype in enumerate(_tensor_dtypes(descriptor.plan)):
        operand_seed = seed + index * 101
        if dtype is DType.Float32:
            payloads.append(arbitrary_float32_payload(operand_seed, 4))
            continue
        generator = random.Random(operand_seed)
        values = tuple(generator.randint(-10_000, 10_000) for _ in range(4))
        if descriptor.kernel.operation == "div" and index == 1:
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


def _structural_record(
    descriptor: KernelPlanDescriptor, transform: ResultTransform | None
) -> EvidenceRecord:
    k = 4
    source = exact_structural_payload(
        17, k, product=descriptor.kernel.operation == "matmul"
    )
    lhs = tuple(int(value) for value in source.lhs.values())
    rhs = (
        None
        if source.rhs is None
        else tuple(int(value) for value in source.rhs.values())
    )
    payloads = _contraction_payloads(descriptor, lhs, rhs)
    layout = Layout(Shape([1, k]), Stride([1, 1]))
    expected = _execute(descriptor, payloads, layout, False)
    actual = _execute(descriptor, payloads, layout, True)
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
        shapes=((1, k),) * len(payloads),
        seed=17,
    )


def _analytic_layout_case(operation: str) -> AnalyticCase:
    if operation == "reduce":
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
    if descriptor.kernel.operation == "reduce":
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
    if label == "exact":
        payloads = _payloads_for_plan(
            descriptor.plan,
            tuple((-2, -1, 1, 2) for _ in _tensor_dtypes(descriptor.plan)),
        )
        return (
            f"{prefix}-exact-error",
            payloads,
            None,
            ((4,),) * len(payloads),
            0,
            None,
        )
    if label == "arbitrary":
        seed = 1000 + sum(ord(character) for character in _plan_id(descriptor.plan))
        payloads = _arbitrary_payloads(descriptor, seed)
        return (
            f"{prefix}-arbitrary-finite-error",
            payloads,
            None,
            ((4,),) * len(payloads),
            seed,
            Tolerance(version="bit-exact-arbitrary-finite-v1"),
        )
    if label == "structural":
        k = 4
        source = exact_structural_payload(
            17, k, product=descriptor.kernel.operation == "matmul"
        )
        payloads = _contraction_payloads(
            descriptor,
            tuple(int(value) for value in source.lhs.values()),
            None
            if source.rhs is None
            else tuple(int(value) for value in source.rhs.values()),
        )
        return (
            f"{prefix}-structural-error",
            payloads,
            k,
            ((1, k),) * len(payloads),
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
        "view": lambda tensor: (
            sw.view(tensor, (slice(None), slice(1, 5))),
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
        if descriptor.kernel.operation in {"reduce", "matmul"}:
            case_functions = (
                (VerificationClass.STRUCTURAL, "structural", _structural_record),
                (VerificationClass.NUMERICAL, "numerical", _numerical_record),
            )
        else:
            case_functions = (
                (VerificationClass.EXACT_ARITHMETIC, "exact", _exact_record),
                (
                    VerificationClass.EXACT_ARITHMETIC,
                    "arbitrary",
                    _arbitrary_exact_record,
                ),
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
        if (
            descriptor.kernel.operation in {"reduce", "matmul"}
            and VerificationClass.ANALYTIC in required
        ):
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
    return StageOneResult(VerificationReport(tuple(records)), tuple(certificates))
