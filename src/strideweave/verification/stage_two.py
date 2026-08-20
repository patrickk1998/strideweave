"""Provider-neutral Stage Two verification of selected target profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal

import strideweave as sw
from strideweave import DType, Layout, Shape, Stride
from strideweave.carriers.operation_policy import operation_execution_options

from .classification import (
    LogicalKernel,
    PlanClassification,
    VerificationProfile,
    classify_profile_plans,
    profile_subjects,
    verification_profile,
)
from .comparison import compare_float32, gamma_bound
from .model import (
    ClassificationDisposition,
    Deviations,
    EvidenceRecord,
    KernelDescriptor,
    OracleCertificate,
    PlanKey,
    Tolerance,
    VerificationClass,
    VerificationOutcome,
    VerificationStage,
)
from .payloads import EncodedFloat32Payload
from .provenance import (
    CompilationBundle,
    CompilationInput,
    CompilationReceipt,
    GeneratedArtifact,
    SpecializationAxis,
    installed_compilation_bundle,
    make_compilation_bundle,
    make_compilation_runtime,
    make_compilation_target,
    make_compilation_toolchain,
    make_jit_specialization_receipt,
)
from .reporting import certificate_value, validate_report
from .stage_one import (
    OracleStageResult,
    _analytic_cases_for,
    _analytic_record,
    _arbitrary_exact_record,
    _case,
    _descriptor_requirement_records,
    _error_record,
    _exact_record,
    _movement_records,
    _movement_requirement_records,
    _numerical_record,
    _oracle_requirement_records,
    _plan_id,
    _recoverable_error_record,
    _required_case_catalog,
    _structural_record,
    _tensor,
    _values,
)

CaseKind = Literal["structural", "numerical"]
OperationName = Literal["reduce_sum", "matmul"]


@dataclass(frozen=True, slots=True)
class TargetStageResult:
    """Raw target-stage evidence produced before v3 report binding."""

    records: tuple[EvidenceRecord, ...]
    receipts: tuple[CompilationReceipt, ...] = ()


@dataclass(frozen=True, slots=True)
class _TargetExecution:
    """One provider execution's evidence and optional exact receipt."""

    records: tuple[EvidenceRecord, ...]
    receipt: CompilationReceipt | None = None


@dataclass(frozen=True, slots=True)
class _TargetRuntime:
    """Private provider adapter for target preparation, execution, and decode."""

    execute: Callable[
        [PlanClassification, str, OracleStageResult, Callable[[], None]],
        _TargetExecution,
    ]
    movement: Callable[
        [VerificationProfile, OracleStageResult, Callable[[], None]],
        tuple[EvidenceRecord, ...],
    ]
    synchronize: Callable[[], None]
    preflight: Callable[[], None]
    current_compilation: Callable[
        [VerificationProfile, tuple[CompilationReceipt, ...]], CompilationBundle
    ]


@dataclass(frozen=True, slots=True)
class _TargetCase:
    """One deterministic CPU target/oracle contraction witness."""

    case_id: str
    operation: OperationName
    kind: CaseKind
    lhs_layout: Layout
    rhs_layout: Layout | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTargetCase:
    """Static facts shared by target execution and installed-graph admission."""

    operation: OperationName
    payloads: tuple[EncodedFloat32Payload, ...]
    contraction_length: int
    shapes: tuple[tuple[int, ...], ...]
    test_class: VerificationClass
    tolerance: Tolerance


def _reduce_second_mode(tensor, *, accumulator_dtype: DType | None = None):
    """Reduce a two-mode tensor through its ordinary public capability.

    Args:
        tensor: Two-mode CPU tensor to reduce over its second mode.
        accumulator_dtype: Floating accumulator to request, or ``None`` for the
            backend default.

    Returns:
        The reduced tensor.
    """
    operation = tensor.carrier.dispatch_op("reduce_sum")
    if accumulator_dtype is None:
        return operation.forward(tensor)
    options = operation_execution_options(
        "reduce_sum", accumulator_dtype=accumulator_dtype
    )
    return operation.forward(tensor, options=options)


def _matrix_shape(layout: Layout) -> tuple[int, int]:
    if len(layout.shape) != 2:
        raise ValueError("Stage Two contractions require two-mode layouts")
    return (layout.shape[0].size, layout.shape[1].size)


def _matrix_values(
    shape: tuple[int, int], value_at: Callable[[int, int], float]
) -> tuple[float, ...]:
    rows, columns = shape
    return tuple(
        value_at(row, column) for column in range(columns) for row in range(rows)
    )


def _structural_values(
    operation: OperationName,
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int] | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    lhs = _matrix_values(
        lhs_shape, lambda row, column: float((3 * row + 2 * column) % 7 - 3)
    )
    if operation == "reduce_sum":
        return lhs, None
    if rhs_shape is None:
        raise ValueError("Stage Two matmul case requires a right-hand layout")
    rhs = _matrix_values(
        rhs_shape, lambda row, column: float((2 * row - column) % 5 - 2)
    )
    return lhs, rhs


def _numerical_values(
    operation: OperationName,
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int] | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    cancellation = (float(2**24), 1.0, 1.0, float(-(2**24)))
    lhs = _matrix_values(
        lhs_shape,
        lambda row, column: (
            cancellation[column % len(cancellation)] * (1.0 if row % 2 == 0 else -1.0)
        ),
    )
    if operation == "reduce_sum":
        return lhs, None
    if rhs_shape is None:
        raise ValueError("Stage Two matmul case requires a right-hand layout")
    rhs = _matrix_values(
        rhs_shape,
        lambda row, column: 1.0 if (row + column) % 2 == 0 else -1.0,
    )
    return lhs, rhs


def _target_cases() -> tuple[_TargetCase, ...]:
    flat_reduce = Layout(Shape([3, 16]), Stride([16, 1]))
    hierarchical_reduce = Layout(Shape([[2, 2], 12]), Stride([[12, 24], 1]))
    flat_lhs = Layout(Shape([3, 16]), Stride([16, 1]))
    flat_rhs = Layout(Shape([2, 16]), Stride([16, 1]))
    hierarchical_lhs = Layout(Shape([[2, 2], 12]), Stride([[12, 24], 1]))
    hierarchical_rhs = Layout(Shape([[3, 1], 12]), Stride([[12, 36], 1]))
    return (
        _TargetCase("reduce-flat-structural", "reduce_sum", "structural", flat_reduce),
        _TargetCase(
            "reduce-hierarchical-numerical",
            "reduce_sum",
            "numerical",
            hierarchical_reduce,
        ),
        _TargetCase(
            "matmul-flat-structural",
            "matmul",
            "structural",
            flat_lhs,
            flat_rhs,
        ),
        _TargetCase(
            "matmul-hierarchical-numerical",
            "matmul",
            "numerical",
            hierarchical_lhs,
            hierarchical_rhs,
        ),
    )


def _float64_oracle_requirements(
    kernel: LogicalKernel,
) -> tuple[tuple[PlanKey, tuple[VerificationClass, ...]], ...]:
    """Return the active CPU oracle plan/classes required by one target."""
    return tuple(
        (descriptor.plan, descriptor.classes)
        for descriptor in classify_profile_plans(verification_profile("cpu-compiled"))
        if descriptor.kernel.operation == kernel.operation
        and descriptor.disposition is ClassificationDisposition.ACTIVE
        and (
            kernel.operation not in {"reduce_sum", "matmul"}
            or descriptor.plan.accumulator_dtype == "Float64"
        )
    )


def _certificate_authorization(
    authorizations: dict[str, str], kernel: LogicalKernel
) -> str | None:
    """Return the prevalidated CPU certificate authorizing one target kernel.

    Target and oracle kernel IDs are intentionally not compared: a target is
    authorized by its logical operation, complete CPU oracle plan scope, and
    required verification classes.
    """
    required_plan_classes = _float64_oracle_requirements(kernel)
    if not required_plan_classes:
        return None
    return authorizations.get(kernel.operation)


def _oracle_requirement_key(
    operation: str,
    kernel_id: str,
    variant: str,
    plan: PlanKey | None,
    test_class: VerificationClass,
    case_id: str,
) -> tuple[str, str, str, PlanKey | None, VerificationClass, str]:
    """Return one exact identity in the registered Stage One catalog."""
    return operation, kernel_id, variant, plan, test_class, case_id


def _oracle_authorizations_from_records(
    profile: VerificationProfile,
    records: tuple[EvidenceRecord, ...],
    certificates: tuple[OracleCertificate, ...],
) -> dict[str, str]:
    """Reconstruct exact authorizations from a complete registered oracle graph."""

    if VerificationStage.ORACLE not in profile.stages:
        raise ValueError("registered profile does not support oracle verification")

    descriptors = classify_profile_plans(profile)
    active_by_kernel: dict[LogicalKernel, list[PlanClassification]] = {}
    expected = set()
    for descriptor in descriptors:
        kernel = descriptor.kernel
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            expected.add(
                _oracle_requirement_key(
                    kernel.operation,
                    kernel.kernel_id,
                    kernel.variant,
                    descriptor.plan,
                    VerificationClass.DEFERRED,
                    f"{kernel.kernel_id}-{_plan_id(descriptor.plan)}-deferred",
                )
            )
            continue
        active_by_kernel.setdefault(kernel, []).append(descriptor)
        expected.update(
            _oracle_requirement_key(
                kernel.operation,
                kernel.kernel_id,
                kernel.variant,
                plan,
                test_class,
                case_id,
            )
            for plan, test_class, case_id in _required_case_catalog(descriptor)
        )
    for subject in profile_subjects(profile):
        expected.add(
            _oracle_requirement_key(
                subject.operation,
                f"movement.{subject.operation}",
                "default",
                None,
                VerificationClass.BIT_EXACT,
                f"movement-{subject.operation}-adversarial-bits",
            )
        )

    observed = [
        _oracle_requirement_key(
            record.case.operation,
            record.case.kernel_id,
            record.case.variant,
            record.case.plan,
            record.test_class,
            record.case.case_id,
        )
        for record in records
    ]
    if any(record.stage is not VerificationStage.ORACLE for record in records):
        raise ValueError("oracle report contains non-oracle evidence")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            "oracle report does not match the exact registered witness catalog"
        )

    expected_certificates: dict[tuple[str, str], OracleCertificate] = {}
    for kernel, kernel_descriptors in active_by_kernel.items():
        requirements = tuple(
            (descriptor.plan, descriptor.classes) for descriptor in kernel_descriptors
        )
        required_classes: list[VerificationClass] = []
        required_cases = []
        for descriptor in kernel_descriptors:
            for test_class in descriptor.classes:
                if test_class not in required_classes:
                    required_classes.append(test_class)
            required_cases.extend(_required_case_catalog(descriptor))
        kernel_records = tuple(
            record
            for record in records
            if record.case.kernel_id == kernel.kernel_id
            and record.case.variant == kernel.variant
        )
        if any(
            record.outcome is not VerificationOutcome.PASSED
            for record in kernel_records
            if (record.case.plan, record.test_class, record.case.case_id)
            in set(required_cases)
        ):
            continue
        certificate = OracleCertificate.from_records(
            KernelDescriptor(kernel.operation, kernel.kernel_id, kernel.variant, ""),
            tuple(required_classes),
            kernel_records,
            required_plan_classes=requirements,
            required_cases=tuple(required_cases),
        )
        expected_certificates[(kernel.kernel_id, kernel.variant)] = certificate

    supplied = {
        (certificate.kernel_id, certificate.variant): certificate
        for certificate in certificates
    }
    if len(supplied) != len(certificates) or supplied != expected_certificates:
        raise ValueError(
            "oracle certificate set does not match complete report evidence"
        )
    authorizations: dict[str, str] = {}
    for kernel in active_by_kernel:
        certificate = expected_certificates.get((kernel.kernel_id, kernel.variant))
        if certificate is None:
            continue
        if kernel.operation in authorizations:
            raise ValueError(
                "oracle profile has multiple certificates for one operation"
            )
        authorizations[kernel.operation] = certificate_value(certificate)[
            "certificate_digest"
        ]
    return authorizations


def _validated_oracle_authorizations(stage_one: OracleStageResult) -> dict[str, str]:
    """Validate the complete registered oracle graph and return operation digests."""

    profile = verification_profile("cpu-compiled")
    header = stage_one.report.header
    if header is None:
        raise ValueError("oracle report has no provenance header")
    if (
        header.oracle_profile != profile.profile_id
        or header.selected_target_profile != profile.profile_id
    ):
        raise ValueError(
            "oracle report profile roles do not match the registered oracle"
        )
    validate_report(header, stage_one.report.records)
    authorizations = _oracle_authorizations_from_records(
        profile, stage_one.report.records, stage_one.certificates
    )
    expected_header_certificates = tuple(
        sorted(
            certificate_value(certificate)["certificate_digest"]
            for certificate in stage_one.certificates
        )
    )
    header_certificate_ids = tuple(
        item["certificate_digest"] for item in header.certificates
    )
    if header_certificate_ids != expected_header_certificates:
        raise ValueError("oracle report certificate facts do not match its result")
    return authorizations


def _blocked_target_result(
    profile: VerificationProfile, diagnostic: str
) -> TargetStageResult:
    """Return target obligations without resolving or invoking a provider runtime."""
    oracle_requirements = _oracle_requirement_records(
        verification_profile("cpu-compiled")
    )
    return TargetStageResult(
        tuple(
            _blocked_requirement_record(record, diagnostic)
            if record.outcome is not VerificationOutcome.DEFERRED
            else record
            for record in _target_requirement_records(profile, oracle_requirements)
        )
    )


def _maximum_term_sum(
    operation: OperationName,
    lhs_values: tuple[float, ...],
    lhs_shape: tuple[int, int],
    rhs_values: tuple[float, ...] | None,
    rhs_shape: tuple[int, int] | None,
) -> float:
    lhs_rows, k = lhs_shape
    if operation == "reduce_sum":
        return max(
            sum(abs(lhs_values[row + lhs_rows * column]) for column in range(k))
            for row in range(lhs_rows)
        )
    if rhs_values is None or rhs_shape is None:
        raise ValueError("Stage Two matmul numerical case requires right-hand values")
    rhs_rows, rhs_k = rhs_shape
    if rhs_k != k:
        raise ValueError("Stage Two matmul contraction lengths must agree")
    return max(
        sum(
            abs(lhs_values[row + lhs_rows * column])
            * abs(rhs_values[column_row + rhs_rows * column])
            for column in range(k)
        )
        for row in range(lhs_rows)
        for column_row in range(rhs_rows)
    )


def _prepare_target_case(
    kernel: LogicalKernel, target_case: _TargetCase
) -> _PreparedTargetCase:
    """Prepare one contraction case without allocating or executing a provider."""

    if kernel.operation == "reduce_sum":
        operation: OperationName = "reduce_sum"
    elif kernel.operation == "matmul":
        operation = "matmul"
    else:
        raise ValueError(f"unsupported Stage Two contraction {kernel.operation!r}")
    if target_case.operation != operation:
        raise ValueError("Stage Two target case does not match its logical operation")
    lhs_shape = _matrix_shape(target_case.lhs_layout)
    rhs_shape = (
        None
        if target_case.rhs_layout is None
        else _matrix_shape(target_case.rhs_layout)
    )
    value_factory = (
        _structural_values if target_case.kind == "structural" else _numerical_values
    )
    values, rhs_values = value_factory(operation, lhs_shape, rhs_shape)
    k = lhs_shape[1]
    payloads = [EncodedFloat32Payload.from_values(values)]
    if operation == "matmul":
        if rhs_values is None or target_case.rhs_layout is None:
            raise ValueError(
                "Stage Two matmul case requires right-hand values and layout"
            )
        payloads.append(EncodedFloat32Payload.from_values(rhs_values))
    test_class = (
        VerificationClass.STRUCTURAL
        if target_case.kind == "structural"
        else VerificationClass.NUMERICAL
    )
    if test_class is VerificationClass.STRUCTURAL:
        tolerance = Tolerance(absolute=0.0, version="bit-exact-structural-v1")
    else:
        tolerance = Tolerance(
            absolute=gamma_bound(
                2.0**-24,
                k,
                _maximum_term_sum(operation, values, lhs_shape, rhs_values, rhs_shape),
            ),
            version="stage-two-float32-gamma-k-v1",
        )
    return _PreparedTargetCase(
        operation,
        tuple(payloads),
        k,
        (lhs_shape,) if rhs_shape is None else (lhs_shape, rhs_shape),
        test_class,
        tolerance,
    )


def _target_case_requirement(
    kernel: LogicalKernel, plan: PlanKey, target_case: _TargetCase
) -> EvidenceRecord:
    """Return one pure model-owned Stage Two contraction template."""

    prepared = _prepare_target_case(kernel, target_case)
    return _case(
        kernel,
        prepared.test_class,
        f"{kernel.kernel_id}-{target_case.case_id}",
        prepared.payloads,
        VerificationOutcome.PASSED,
        Deviations(0.0, 0.0, 0),
        0,
        k=prepared.contraction_length,
        shapes=prepared.shapes,
        plan=plan,
        tolerance=prepared.tolerance,
        stage=VerificationStage.TARGET,
    )


def _synchronize_target() -> None:
    """Synchronize the selected target before its results are decoded.

    The installed CPU target completes synchronously. Keeping this explicit is
    deliberate: provider runtimes with queued work replace this hook, and the
    ordering remains observable in Stage Two tests.
    """


def _target_case(
    kernel: LogicalKernel,
    plan: PlanKey,
    target_case: _TargetCase,
    synchronize: Callable[[], None],
) -> EvidenceRecord:
    prepared = _prepare_target_case(kernel, target_case)
    payloads = prepared.payloads
    try:
        lhs_target = _tensor(
            payloads[0].values(), DType.Float32, target_case.lhs_layout, True
        )
        lhs_oracle = _tensor(
            payloads[0].values(), DType.Float32, target_case.lhs_layout, True
        )
        if kernel.operation == "reduce_sum":
            # The production layouts are hierarchical, so the two-mode
            # reduction primitive is dispatched directly rather than lowered
            # from a description that would first rearrange the operand away.
            target = _reduce_second_mode(lhs_target)
            oracle = _reduce_second_mode(lhs_oracle, accumulator_dtype=DType.Float64)
        else:
            if target_case.rhs_layout is None or len(payloads) != 2:
                raise ValueError("Stage Two matmul payload preparation is incomplete")
            rhs_target = _tensor(
                payloads[1].values(), DType.Float32, target_case.rhs_layout, True
            )
            rhs_oracle = _tensor(
                payloads[1].values(), DType.Float32, target_case.rhs_layout, True
            )
            target = sw.matmul(lhs_target, rhs_target)
            oracle = sw.matmul(lhs_oracle, rhs_oracle, accumulator_dtype=DType.Float64)
        synchronize()
        comparison = compare_float32(_values(oracle), _values(target))
    except (RuntimeError, ValueError) as error:
        return replace(
            _error_record(
                kernel,
                prepared.test_class,
                f"{kernel.kernel_id}-{target_case.case_id}",
                error,
                payloads=payloads,
                k=prepared.contraction_length,
                shapes=prepared.shapes,
                tolerance=prepared.tolerance,
                plan=plan,
            ),
            stage=VerificationStage.TARGET,
        )
    maximum_absolute = comparison.deviations.maximum_absolute
    if maximum_absolute is None:
        raise RuntimeError("completed numerical comparison has no absolute deviation")
    passed = maximum_absolute <= prepared.tolerance.absolute
    if prepared.test_class is VerificationClass.STRUCTURAL:
        passed = comparison.mismatches == 0
    record = _case(
        kernel,
        prepared.test_class,
        f"{kernel.kernel_id}-{target_case.case_id}",
        payloads,
        VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED,
        comparison.deviations,
        comparison.mismatches,
        k=prepared.contraction_length,
        diagnostic=(
            None
            if passed
            else "encoded Float32 results are not bit-identical"
            if prepared.test_class is VerificationClass.STRUCTURAL
            and comparison.mismatches
            else f"maximum absolute deviation exceeds {prepared.tolerance.absolute}"
        ),
        shapes=prepared.shapes,
        plan=plan,
        tolerance=prepared.tolerance,
    )
    return replace(
        record,
        stage=VerificationStage.TARGET,
    )


def _target_record(record: EvidenceRecord, certificate_digest: str) -> EvidenceRecord:
    """Mark a completed provider execution as target evidence."""

    return replace(
        record,
        stage=VerificationStage.TARGET,
        consumed_certificate_digest=certificate_digest,
    )


def _deferred_target_record(descriptor: PlanClassification) -> EvidenceRecord:
    """Retain an explicit target deferral for one classified plan."""

    return replace(
        _case(
            descriptor.kernel,
            VerificationClass.DEFERRED,
            f"{descriptor.kernel.kernel_id}-{descriptor.plan.operation}-deferred",
            (),
            VerificationOutcome.DEFERRED,
            Deviations(0.0, 0.0, 0),
            0,
            diagnostic=descriptor.deferred_reason,
            plan=descriptor.plan,
            seed=None,
        ),
        stage=VerificationStage.TARGET,
    )


def _cpu_target_requirement_records(
    profile: VerificationProfile,
    oracle_records: tuple[EvidenceRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    """Return the installed CPU target graph without allocating or executing."""

    del oracle_records
    records: list[EvidenceRecord] = [
        replace(
            record,
            stage=VerificationStage.TARGET,
            case=replace(record.case, case_id=f"stage-two-{record.case.case_id}"),
        )
        for record in _movement_requirement_records(profile)
    ]
    for descriptor in classify_profile_plans(profile):
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            records.append(_deferred_target_record(descriptor))
            continue
        if descriptor.kernel.operation not in {"reduce_sum", "matmul"}:
            records.extend(
                replace(record, stage=VerificationStage.TARGET)
                for record in _descriptor_requirement_records(descriptor)
            )
            continue
        records.extend(
            _target_case_requirement(descriptor.kernel, descriptor.plan, target_case)
            for target_case in _target_cases()
            if target_case.operation == descriptor.kernel.operation
        )
        records.extend(
            replace(record, stage=VerificationStage.TARGET)
            for record in _descriptor_requirement_records(descriptor)
            if record.test_class is VerificationClass.ANALYTIC
        )
    return tuple(records)


def _synthetic_target_requirement_records(
    profile: VerificationProfile,
    oracle_records: tuple[EvidenceRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    """Return the test JIT target graph from current model-owned oracle templates."""

    records = [
        replace(
            record,
            stage=VerificationStage.TARGET,
            case=replace(record.case, case_id=f"stage-two-{record.case.case_id}"),
        )
        for record in oracle_records
        if record.case.kernel_id.startswith("movement.")
    ]
    for descriptor in classify_profile_plans(profile):
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            records.append(_deferred_target_record(descriptor))
            continue
        source = tuple(
            record
            for record in oracle_records
            if record.case.operation == descriptor.kernel.operation
            and record.case.plan == descriptor.plan
        )
        if not source:
            raise ValueError(
                "synthetic target requirement has no matching oracle witness"
            )
        records.extend(
            replace(
                record,
                stage=VerificationStage.TARGET,
                case=replace(
                    record.case,
                    kernel_id=descriptor.kernel.kernel_id,
                    variant=descriptor.kernel.variant,
                ),
            )
            for record in source
        )
    return tuple(records)


def _target_requirement_records(
    profile: VerificationProfile,
    oracle_records: tuple[EvidenceRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    """Return one registered target's independent, execution-free case graph."""

    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    registered = verification_profile(profile.profile_id)
    if profile != registered:
        raise ValueError("profile does not match its registered descriptor")
    if VerificationStage.TARGET not in profile.stages:
        raise ValueError(
            f"profile {profile.profile_id!r} does not support target verification"
        )
    factories = {
        "cpu-compiled": _cpu_target_requirement_records,
        "synthetic-jit": _synthetic_target_requirement_records,
    }
    try:
        factory = factories[profile.profile_id]
    except KeyError as error:
        raise RuntimeError(
            f"profile {profile.profile_id!r} has no registered requirement provider"
        ) from error
    return factory(profile, oracle_records)


def _target_requirements_for_descriptor(
    requirements: tuple[EvidenceRecord, ...],
    descriptor: PlanClassification,
) -> tuple[EvidenceRecord, ...]:
    """Return exact current target attempts belonging to one active plan."""

    return tuple(
        record
        for record in requirements
        if record.case.kernel_id == descriptor.kernel.kernel_id
        and record.case.variant == descriptor.kernel.variant
        and record.case.plan == descriptor.plan
    )


def _blocked_requirement_record(
    record: EvidenceRecord, diagnostic: str
) -> EvidenceRecord:
    """Preserve one exact target requirement while recording missing authority."""

    return replace(
        record,
        outcome=VerificationOutcome.BLOCKED,
        deviations=Deviations(0.0, 0.0, 0),
        mismatches=0,
        diagnostic=diagnostic,
        compilation_receipt_id=None,
        consumed_certificate_digest=None,
    )


def _errored_requirement_record(
    record: EvidenceRecord,
    error: RuntimeError | ValueError,
    certificate_digest: str,
) -> EvidenceRecord:
    """Preserve one exact target requirement after a provider-wide failure."""

    return replace(
        record,
        outcome=VerificationOutcome.ERROR,
        deviations=Deviations(None, None, None),
        mismatches=None,
        diagnostic=f"{type(error).__name__}: {error}",
        compilation_receipt_id=None,
        consumed_certificate_digest=certificate_digest,
    )


def _generic_target_records(
    descriptor: PlanClassification,
    certificate_digest: str,
    synchronize: Callable[[], None],
) -> tuple[EvidenceRecord, ...]:
    """Execute all non-contraction target classes through public CPU capability."""

    records: list[EvidenceRecord] = []
    case_functions = (
        (VerificationClass.EXACT_ARITHMETIC, _exact_record),
        (VerificationClass.EXACT_ARITHMETIC, _arbitrary_exact_record),
        (VerificationClass.STRUCTURAL, _structural_record),
        (VerificationClass.NUMERICAL, _numerical_record),
    )
    for test_class, case_function in case_functions:
        if test_class not in descriptor.classes:
            continue
        try:
            completed = case_function(descriptor, None, synchronize=synchronize)
            evidence = completed if isinstance(completed, tuple) else (completed,)
            records.extend(
                _target_record(record, certificate_digest) for record in evidence
            )
        except (RuntimeError, ValueError) as error:
            records.append(
                _target_record(
                    _recoverable_error_record(descriptor, test_class, "target", error),
                    certificate_digest,
                )
            )
    if VerificationClass.ANALYTIC in descriptor.classes:
        for analytic_case in _analytic_cases_for(descriptor.kernel.operation):
            try:
                records.append(
                    _target_record(
                        _analytic_record(
                            descriptor,
                            None,
                            analytic_case,
                            synchronize=synchronize,
                        ),
                        certificate_digest,
                    )
                )
            except (RuntimeError, ValueError) as error:
                records.append(
                    _target_record(
                        _recoverable_error_record(
                            descriptor,
                            VerificationClass.ANALYTIC,
                            "analytic",
                            error,
                            analytic_case=analytic_case,
                        ),
                        certificate_digest,
                    )
                )
    return tuple(records)


def _target_records_for_descriptor(
    descriptor: PlanClassification,
    certificate_digest: str,
    synchronize: Callable[[], None],
) -> tuple[EvidenceRecord, ...]:
    """Execute one complete active target-plan obligation."""

    if descriptor.kernel.operation not in {"reduce_sum", "matmul"}:
        return _generic_target_records(descriptor, certificate_digest, synchronize)
    records: list[EvidenceRecord] = []
    for target_case in _target_cases():
        if target_case.operation != descriptor.kernel.operation:
            continue
        records.append(
            replace(
                _target_case(
                    descriptor.kernel, descriptor.plan, target_case, synchronize
                ),
                consumed_certificate_digest=certificate_digest,
            )
        )
    if VerificationClass.ANALYTIC in descriptor.classes:
        for analytic_case in _analytic_cases_for(descriptor.kernel.operation):
            try:
                records.append(
                    _target_record(
                        _analytic_record(
                            descriptor,
                            None,
                            analytic_case,
                            synchronize=synchronize,
                        ),
                        certificate_digest,
                    )
                )
            except (RuntimeError, ValueError) as error:
                records.append(
                    _target_record(
                        _recoverable_error_record(
                            descriptor,
                            VerificationClass.ANALYTIC,
                            "analytic",
                            error,
                            analytic_case=analytic_case,
                        ),
                        certificate_digest,
                    )
                )
    return tuple(records)


def _cpu_target_records(
    descriptor: PlanClassification,
    certificate_digest: str,
    oracle_result: OracleStageResult,
    synchronize: Callable[[], None],
) -> _TargetExecution:
    """Execute the installed CPU target through its public capabilities."""

    del oracle_result
    return _TargetExecution(
        _target_records_for_descriptor(descriptor, certificate_digest, synchronize)
    )


def _cpu_movement_records(
    profile: VerificationProfile,
    oracle_result: OracleStageResult,
    synchronize: Callable[[], None],
) -> tuple[EvidenceRecord, ...]:
    """Execute required movement subjects through the installed CPU target."""

    del oracle_result
    return tuple(
        replace(
            movement,
            stage=VerificationStage.TARGET,
            case=replace(movement.case, case_id=f"stage-two-{movement.case.case_id}"),
        )
        for movement in _movement_records(profile, synchronize=synchronize)
    )


def _synthetic_target_records(
    descriptor: PlanClassification,
    certificate_digest: str,
    oracle_result: OracleStageResult,
    synchronize: Callable[[], None],
) -> _TargetExecution:
    """Test-only JIT adapter that materializes target evidence from its oracle input.

    This runtime deliberately does not call the installed CPU target executor.
    It isolates target-selection, receipt binding, and synchronization tests
    until a real provider registers its own adapter.
    """

    source = tuple(
        record
        for record in oracle_result.report.records
        if record.case.operation == descriptor.kernel.operation
        and record.case.plan == descriptor.plan
    )
    if not source:
        raise ValueError("synthetic target has no matching oracle witness")
    synchronize()
    records = tuple(
        replace(
            record,
            stage=VerificationStage.TARGET,
            case=replace(
                record.case,
                kernel_id=descriptor.kernel.kernel_id,
                variant=descriptor.kernel.variant,
            ),
            consumed_certificate_digest=certificate_digest,
        )
        for record in source
    )
    return _TargetExecution(records, _synthetic_jit_receipt(descriptor))


def _synthetic_movement_records(
    profile: VerificationProfile,
    oracle_result: OracleStageResult,
    synchronize: Callable[[], None],
) -> tuple[EvidenceRecord, ...]:
    """Materialize test-only target movement evidence without CPU target fallback."""

    records = []
    for record in oracle_result.report.records:
        if not record.case.kernel_id.startswith("movement."):
            continue
        synchronize()
        records.append(
            replace(
                record,
                stage=VerificationStage.TARGET,
                case=replace(record.case, case_id=f"stage-two-{record.case.case_id}"),
            )
        )
    return tuple(records)


def _cpu_current_compilation(
    profile: VerificationProfile, receipts: tuple[CompilationReceipt, ...]
) -> CompilationBundle:
    """Resolve the current installed CPU bundle through the neutral adapter."""
    del receipts
    return installed_compilation_bundle(profile)


def _target_runtime(profile: VerificationProfile) -> _TargetRuntime:
    """Resolve the private runtime adapter for a registered target profile."""

    runtimes = {
        "cpu-compiled": _TargetRuntime(
            _cpu_target_records,
            _cpu_movement_records,
            _synchronize_target,
            lambda: None,
            _cpu_current_compilation,
        ),
        "synthetic-jit": _TargetRuntime(
            _synthetic_target_records,
            _synthetic_movement_records,
            _synchronize_target,
            _synthetic_runtime_unavailable,
            _synthetic_compilation_unavailable,
        ),
    }
    try:
        return runtimes[profile.profile_id]
    except KeyError as error:
        raise RuntimeError(
            f"profile {profile.profile_id!r} has no registered target runtime"
        ) from error


def _require_target_runtime(profile: VerificationProfile) -> None:
    """Preflight a selected runtime before public verification begins."""

    _target_runtime(profile).preflight()


def _current_profile_compilation_bundle(
    profile: VerificationProfile, receipts: tuple[CompilationReceipt, ...]
) -> CompilationBundle:
    """Resolve current exact receipt facts behind one neutral runtime adapter."""
    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    registered = verification_profile(profile.profile_id)
    if profile != registered:
        raise ValueError("profile does not match its registered descriptor")
    if type(receipts) is not tuple or any(
        receipt.profile_id != profile.profile_id for receipt in receipts
    ):
        raise ValueError("selected receipts do not match their verification profile")
    runtime = _target_runtime(profile)
    runtime.preflight()
    bundle = runtime.current_compilation(profile, receipts)
    if not isinstance(bundle, CompilationBundle):
        raise TypeError("provider compilation resolver must return a CompilationBundle")
    if any(receipt.profile_id != profile.profile_id for receipt in bundle.receipts):
        raise ValueError("provider compilation bundle crosses verification profiles")
    return make_compilation_bundle(bundle.receipts)


def _synthetic_runtime_unavailable() -> None:
    """Reject public use of the internal synthetic target runtime."""

    raise RuntimeError(
        "profile 'synthetic-jit' is an internal Stage Two test runtime and "
        "has no installed provider runtime"
    )


def _synthetic_compilation_unavailable(
    profile: VerificationProfile, receipts: tuple[CompilationReceipt, ...]
) -> CompilationBundle:
    """Reject provenance rebinding for the unavailable internal test runtime."""
    del profile, receipts
    raise RuntimeError(
        "profile 'synthetic-jit' is an internal Stage Two test runtime and "
        "has no installed provider provenance"
    )


def _synthetic_jit_receipt(descriptor: PlanClassification) -> CompilationReceipt:
    """Return test-only exact JIT receipts for synthetic target executions."""

    provider = "synthetic-jit-test-runtime"
    target = make_compilation_target(
        architecture="synthetic-arch",
        vendor="strideweave-test",
        operating_system="synthetic-os",
        abi="synthetic-abi",
        endianness="little",
        pointer_bits=64,
    )
    toolchain = make_compilation_toolchain(
        provider=provider,
        compiler_id="synthetic-jit",
        compiler_version="1",
        target_triple="synthetic-arch-synthetic-os",
        build_system="stage-two-test-seam",
    )
    runtime = make_compilation_runtime({"runtime": "synthetic-jit", "version": "1"})
    plan_json = json.dumps(
        asdict(descriptor.plan), separators=(",", ":"), sort_keys=True
    )
    digest = hashlib.sha256(plan_json.encode()).hexdigest()
    executable = hashlib.sha256(
        f"{descriptor.kernel.kernel_id}:{plan_json}:runtime".encode()
    ).hexdigest()
    uri = f"synthetic://source/{descriptor.kernel.operation}.tile"
    return make_jit_specialization_receipt(
        profile_id="synthetic-jit",
        provider=provider,
        target=target,
        toolchain=toolchain,
        runtime=runtime,
        logical_kernel=descriptor.kernel,
        declared_input_uris=(uri,),
        inputs=(CompilationInput(0, uri, "source", digest),),
        compile_options=("synthetic-jit-opt=1",),
        artifacts=(
            GeneratedArtifact(0, "generated-host-source", digest),
            GeneratedArtifact(1, "generated-device-source", executable),
            GeneratedArtifact(2, "runtime-executable", executable, executable),
        ),
        specialization_axes=(
            SpecializationAxis("logical_kernel", descriptor.kernel.kernel_id),
            SpecializationAxis("plan", plan_json),
        ),
        generated_host_source_artifact_ordinal=0,
        generated_device_source_artifact_ordinal=1,
        runtime_artifact_ordinals=(2,),
    )


def run_target_stage(
    profile: VerificationProfile, oracle_result: OracleStageResult
) -> TargetStageResult:
    """Verify one registered target profile using accepted CPU oracle evidence.

    Args:
        profile: Registered profile that declares the target verification stage.
        oracle_result: Complete Stage One CPU oracle result and certificates.

    Returns:
        Raw target evidence to bind with both profiles' compilation receipts.

    Examples:
        >>> from strideweave.verification import (
        ...     run_oracle_stage,
        ...     run_target_stage,
        ...     verification_profile,
        ... )
        >>> result = run_target_stage(
        ...     verification_profile("cpu-compiled"),
        ...     run_oracle_stage(verification_profile("cpu-compiled")),
        ... )
        >>> bool(result.records)
        True
    """
    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    registered_profile = verification_profile(profile.profile_id)
    if profile != registered_profile:
        raise ValueError("profile does not match its registered descriptor")
    if VerificationStage.TARGET not in profile.stages:
        raise ValueError(
            f"profile {profile.profile_id!r} does not support target verification"
        )
    if not isinstance(oracle_result, OracleStageResult):
        raise TypeError("oracle_result must be an OracleStageResult")
    try:
        authorizations = _validated_oracle_authorizations(oracle_result)
    except (TypeError, ValueError) as error:
        return _blocked_target_result(
            profile,
            "Stage One oracle result is invalid or incomplete: "
            f"{type(error).__name__}: {error}",
        )
    if not authorizations:
        return _blocked_target_result(
            profile,
            "Stage One oracle certificate is absent or no required scope passed",
        )
    runtime = _target_runtime(profile)
    receipts: list[CompilationReceipt] = []
    records = list(runtime.movement(profile, oracle_result, runtime.synchronize))
    target_requirements = _target_requirement_records(
        profile,
        _oracle_requirement_records(verification_profile("cpu-compiled")),
    )
    for descriptor in classify_profile_plans(profile):
        kernel = descriptor.kernel
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            records.append(_deferred_target_record(descriptor))
            continue
        authorization = _certificate_authorization(authorizations, kernel)
        if authorization is None:
            records.extend(
                _blocked_requirement_record(
                    requirement,
                    "Stage One oracle certificate is absent, invalid, or lacks required scope",
                )
                for requirement in _target_requirements_for_descriptor(
                    target_requirements, descriptor
                )
            )
            continue
        try:
            execution = runtime.execute(
                descriptor, authorization, oracle_result, runtime.synchronize
            )
        except (RuntimeError, ValueError) as error:
            records.extend(
                _errored_requirement_record(
                    requirement,
                    error,
                    authorization,
                )
                for requirement in _target_requirements_for_descriptor(
                    target_requirements, descriptor
                )
            )
            continue
        produced = execution.records
        receipt = execution.receipt
        if receipt is not None:
            receipts.append(receipt)
            produced = tuple(
                replace(record, compilation_receipt_id=receipt.receipt_id)
                for record in produced
            )
        records.extend(produced)
    return TargetStageResult(tuple(records), tuple(receipts))
