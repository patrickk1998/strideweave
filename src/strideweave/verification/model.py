"""Immutable records shared by staged kernel verification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class VerificationStage(Enum):
    ORACLE = "stage_one"
    TARGET = "stage_two"


class VerificationClass(Enum):
    BIT_EXACT = "bit_exact"
    EXACT_ARITHMETIC = "exact_arithmetic"
    STRUCTURAL = "structural"
    ANALYTIC = "analytic"
    NUMERICAL = "numerical"
    DEFERRED = "deferred"


class VerificationOutcome(Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class ClassificationDisposition(Enum):
    ACTIVE = "active"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class KernelDescriptor:
    operation: str
    kernel_id: str
    variant: str
    pybind_name: str


@dataclass(frozen=True, slots=True)
class PlanKey:
    operation: str
    operands: tuple[tuple[str, str | None, str | None], ...]
    compute: str
    accumulation: str | None
    accumulator_dtype: str | None
    output: str

    @classmethod
    def from_plan_like(cls, plan: Any) -> PlanKey:
        return cls(
            operation=plan.operation,
            operands=tuple(
                (
                    operand.role.name,
                    None if operand.dtype is None else operand.dtype.name,
                    None if operand.convert_to is None else operand.convert_to.name,
                )
                for operand in plan.operands
            ),
            compute=plan.compute.name,
            accumulation=None if plan.accumulation is None else plan.accumulation.name,
            accumulator_dtype=(
                None if plan.accumulator_dtype is None else plan.accumulator_dtype.name
            ),
            output=plan.output.name,
        )


@dataclass(frozen=True, slots=True)
class KernelPlanDescriptor:
    kernel: KernelDescriptor
    plan: PlanKey
    classes: tuple[VerificationClass, ...]
    disposition: ClassificationDisposition
    deferred_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CaseDescriptor:
    operation: str
    kernel_id: str
    variant: str
    input_dtypes: tuple[str, ...]
    output_dtype: str
    shapes: tuple[tuple[int, ...], ...]
    accumulator_dtype: str | None
    contraction_length: int | None
    seed: int | None
    case_id: str
    plan: PlanKey | None = None


@dataclass(frozen=True, slots=True)
class Tolerance:
    absolute: float = 0.0
    relative: float = 0.0
    ulps: int = 0
    version: str = "exact-v1"


@dataclass(frozen=True, slots=True)
class Deviations:
    maximum_absolute: float | None
    maximum_relative: float | None
    maximum_ulps: int | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    stage: VerificationStage
    test_class: VerificationClass
    case: CaseDescriptor
    target_input_bit_hashes: tuple[str, ...]
    oracle_input_bit_hashes: tuple[str, ...]
    tolerance: Tolerance
    deviations: Deviations
    mismatches: int | None
    outcome: VerificationOutcome
    diagnostic: str | None = None
    schema_version: str = "strideweave.kernel-evidence.v1"

    def __post_init__(self) -> None:
        if self.target_input_bit_hashes != self.oracle_input_bit_hashes:
            raise ValueError("target and oracle input bit hashes must match")

    def as_json_object(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage"] = self.stage.value
        value["test_class"] = self.test_class.value
        value["outcome"] = self.outcome.value
        return _json_safe(value)

    def to_jsonl(self) -> str:
        return json.dumps(
            self.as_json_object(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    records: tuple[EvidenceRecord, ...]
    schema_version: str = "strideweave.kernel-verification.v1"

    def to_jsonl(self) -> str:
        ordered = sorted(
            self.records,
            key=lambda record: (
                record.case.case_id,
                record.case.kernel_id,
                record.case.variant,
            ),
        )
        return (
            ""
            if not ordered
            else "\n".join(record.to_jsonl() for record in ordered) + "\n"
        )


@dataclass(frozen=True, slots=True)
class OracleCertificate:
    kernel_id: str
    variant: str
    certified_classes: tuple[VerificationClass, ...]
    evidence_digest: str

    @classmethod
    def from_records(
        cls,
        kernel: KernelDescriptor,
        required_classes: tuple[VerificationClass, ...],
        records: tuple[EvidenceRecord, ...],
        *,
        required_plan_classes: tuple[
            tuple[PlanKey, tuple[VerificationClass, ...]], ...
        ] = (),
    ) -> OracleCertificate:
        observed = {
            record.test_class
            for record in records
            if record.case.kernel_id == kernel.kernel_id
            and record.case.variant == kernel.variant
            and record.outcome is VerificationOutcome.PASSED
        }
        missing = tuple(
            test_class for test_class in required_classes if test_class not in observed
        )
        if missing:
            names = ", ".join(test_class.value for test_class in missing)
            raise ValueError(
                f"cannot certify {kernel.kernel_id}: missing passed classes {names}"
            )
        if any(
            record.case.kernel_id == kernel.kernel_id
            and record.case.variant == kernel.variant
            and record.test_class in required_classes
            and record.outcome is not VerificationOutcome.PASSED
            for record in records
        ):
            raise ValueError(
                f"cannot certify {kernel.kernel_id}: required evidence did not pass"
            )
        for plan, plan_classes in required_plan_classes:
            plan_records = tuple(
                record
                for record in records
                if record.case.kernel_id == kernel.kernel_id
                and record.case.variant == kernel.variant
                and record.case.plan == plan
            )
            passed = {
                record.test_class
                for record in plan_records
                if record.outcome is VerificationOutcome.PASSED
            }
            missing_plan_classes = tuple(
                test_class for test_class in plan_classes if test_class not in passed
            )
            if missing_plan_classes:
                names = ", ".join(
                    test_class.value for test_class in missing_plan_classes
                )
                raise ValueError(
                    f"cannot certify {kernel.kernel_id}: plan {plan!r} is missing "
                    f"passed classes {names}"
                )
            if any(
                record.test_class in plan_classes
                and record.outcome is not VerificationOutcome.PASSED
                for record in plan_records
            ):
                raise ValueError(
                    f"cannot certify {kernel.kernel_id}: required plan evidence did not pass"
                )
        serialized = VerificationReport(records).to_jsonl().encode()
        return cls(
            kernel.kernel_id,
            kernel.variant,
            required_classes,
            hashlib.sha256(serialized).hexdigest(),
        )
