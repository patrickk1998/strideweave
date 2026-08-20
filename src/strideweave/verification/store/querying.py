"""Offline factual queries over the schema-v3 evidence graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..classification import verification_profile
from ..model import VerificationReport
from ..provenance import CompilationReceipt, compilation_receipt_json_object
from ._identity import _canonical_json, _todo_provenance_digest
from .base import EvidenceStore, SQLStatement, VerificationStoreError


def _required(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise VerificationStoreError(f"{field} must be a non-empty string")
    return value


def _optional(value: object, field: str) -> str | None:
    return None if value is None else _required(value, field)


def _registered_profile_id(value: object, field: str) -> str:
    """Validate one selector against the neutral registry before store access."""

    profile_id = _required(value, field)
    try:
        verification_profile(profile_id)
    except ValueError as error:
        raise VerificationStoreError(
            f"{field} names an unknown verification profile {profile_id!r}"
        ) from error
    return profile_id


def _row_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if type(value) is not str:
        raise VerificationStoreError(
            f"verification store returned invalid {field} query data"
        )
    return value


def _row_optional(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    return None if value is None else _row_text(row, field)


def _json(value: object, field: str) -> Mapping[str, object]:
    try:
        result = json.loads(value) if type(value) is str else None
    except json.JSONDecodeError as error:
        raise VerificationStoreError(
            f"verification store returned invalid {field} JSON"
        ) from error
    if type(result) is not dict:
        raise VerificationStoreError(
            f"verification store returned invalid {field} data"
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class StatusObservation:
    """One unaggregated producer observation and its selected-target facts."""

    run_id: str
    evidence_id: str
    observation_id: str
    selected_target_profile: str
    oracle_profile: str
    receipt_id: str | None
    receipt_kind: str | None
    provider: str | None
    stage: str
    test_class: str
    case_id: str
    operation: str
    kernel_id: str
    variant: str
    outcome: str
    producer_id: str
    source_commit: str | None
    recorded_at_utc: str
    artifact_locator: str | None
    artifact_digest: str | None

    def as_json_object(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def query_status(
    store: EvidenceStore,
    *,
    selected_target_profile: str,
    kernel_id: str | None = None,
    variant: str | None = None,
    test_class: str | None = None,
    case_id: str | None = None,
    producer_id: str | None = None,
) -> tuple[StatusObservation, ...]:
    """Return selected-profile observations in deterministic factual order."""

    profile = _registered_profile_id(selected_target_profile, "selected_target_profile")
    filters = (
        ("e.kernel_id", kernel_id, "kernel_id"),
        ("e.variant", variant, "variant"),
        ("e.test_class", test_class, "test_class"),
        ("e.case_id", case_id, "case_id"),
        ("o.producer_id", producer_id, "producer_id"),
    )
    clauses = ["r.selected_target_profile = ?"]
    parameters: list[str] = [profile]
    for column, value, name in filters:
        parsed = _optional(value, name)
        if parsed is not None:
            clauses.append(f"{column} = ?")
            parameters.append(parsed)
    rows = store.query(
        SQLStatement(
            "SELECT r.run_id, e.evidence_id, o.observation_id, r.selected_target_profile, r.oracle_profile, "
            "e.receipt_id, c.receipt_kind, c.provider, e.stage, e.test_class, e.case_id, e.operation_name, "
            "e.kernel_id, e.variant, e.outcome, o.producer_id, o.source_commit, o.recorded_at_utc, "
            "o.artifact_locator, o.artifact_digest FROM observations o JOIN evidence e ON e.evidence_id=o.evidence_id "
            "JOIN verification_runs r ON r.run_id=e.run_id LEFT JOIN compilation_receipts c ON c.receipt_id=e.receipt_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY e.kernel_id, e.variant, e.test_class, e.case_id, o.producer_id, o.observation_id",
            tuple(parameters),
        )
    )
    return tuple(
        StatusObservation(
            run_id=_row_text(row, "run_id"),
            evidence_id=_row_text(row, "evidence_id"),
            observation_id=_row_text(row, "observation_id"),
            selected_target_profile=_row_text(row, "selected_target_profile"),
            oracle_profile=_row_text(row, "oracle_profile"),
            receipt_id=_row_optional(row, "receipt_id"),
            receipt_kind=_row_optional(row, "receipt_kind"),
            provider=_row_optional(row, "provider"),
            stage=_row_text(row, "stage"),
            test_class=_row_text(row, "test_class"),
            case_id=_row_text(row, "case_id"),
            operation=_row_text(row, "operation_name"),
            kernel_id=_row_text(row, "kernel_id"),
            variant=_row_text(row, "variant"),
            outcome=_row_text(row, "outcome"),
            producer_id=_row_text(row, "producer_id"),
            source_commit=_row_optional(row, "source_commit"),
            recorded_at_utc=_row_text(row, "recorded_at_utc"),
            artifact_locator=_row_optional(row, "artifact_locator"),
            artifact_digest=_row_optional(row, "artifact_digest"),
        )
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class IdentityDifference:
    """One independently reported provenance-axis difference."""

    axis: str
    stored: tuple[str, ...]
    current: tuple[str, ...]

    def as_json_object(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "stored": list(self.stored),
            "current": list(self.current),
        }


@dataclass(frozen=True, slots=True)
class RunStaleness:
    """Exact differences between one stored report and a supplied current report."""

    run_id: str
    differences: tuple[IdentityDifference, ...]

    @property
    def is_stale(self) -> bool:
        return bool(self.differences)

    def as_json_object(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "is_stale": self.is_stale,
            "differences": [item.as_json_object() for item in self.differences],
        }


def _receipt_axis(receipt: CompilationReceipt, axis: str) -> str:
    receipt_value = compilation_receipt_json_object(receipt)
    value: object
    if axis == "provider":
        value = receipt_value["provider"]
    elif axis == "target":
        value = receipt_value["target"]
    elif axis == "toolchain":
        value = receipt_value["toolchain"]
    elif axis == "runtime":
        value = receipt_value["runtime"]
    elif axis == "specialization":
        value = receipt_value.get("specialization_axes", [])
    elif axis == "generated_source":
        value = [
            item
            for item in receipt_value["artifacts"]
            if item["artifact_kind"]
            in {"generated-host-source", "generated-device-source"}
        ]
    elif axis == "compilation_input_closure":
        value = {
            "compile_options": receipt_value["compile_options"],
            "input_closure_digest": receipt_value["input_closure_digest"],
            "inputs": receipt_value["inputs"],
        }
    elif axis == "executable_artifact":
        value = receipt_value["artifacts"]
    else:
        raise AssertionError(axis)
    return _canonical_json(value)


def _receipt_scope(receipt: CompilationReceipt, *, include_specialization: bool) -> str:
    value = compilation_receipt_json_object(receipt)
    scope: dict[str, object] = {
        "kind": value["kind"],
        "logical_kernel": value["logical_kernel"],
        "profile_id": value["profile_id"],
    }
    if include_specialization:
        scope["specialization_axes"] = value.get("specialization_axes", [])
    return _canonical_json(scope)


def _specializations_by_base(
    receipts: tuple[CompilationReceipt, ...],
) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {}
    for receipt in receipts:
        base = _receipt_scope(receipt, include_specialization=False)
        specialization = _receipt_axis(receipt, "specialization")
        result.setdefault(base, set()).add(specialization)
    return {base: frozenset(values) for base, values in result.items()}


def _differences(
    stored: VerificationReport, current: VerificationReport
) -> tuple[IdentityDifference, ...]:
    if stored.header is None or current.header is None:  # pragma: no cover
        raise VerificationStoreError("verification report has no v3 header")
    differences: list[IdentityDifference] = []
    stored_receipts = stored.header.compilation_bundle.receipts
    current_receipts = current.header.compilation_bundle.receipts
    stored_specializations = _specializations_by_base(stored_receipts)
    current_specializations = _specializations_by_base(current_receipts)
    for axis in (
        "provider",
        "target",
        "toolchain",
        "runtime",
        "specialization",
        "generated_source",
        "compilation_input_closure",
        "executable_artifact",
    ):

        def scoped_value(
            receipt: CompilationReceipt, side: dict[str, frozenset[str]]
        ) -> str:
            base = _receipt_scope(receipt, include_specialization=False)
            include_specialization = axis == "specialization" or side.get(base) == (
                current_specializations
                if side is stored_specializations
                else stored_specializations
            ).get(base)
            return _canonical_json(
                {
                    "scope": _receipt_scope(
                        receipt, include_specialization=include_specialization
                    ),
                    "value": _receipt_axis(receipt, axis),
                }
            )

        before = tuple(
            sorted(
                scoped_value(item, stored_specializations) for item in stored_receipts
            )
        )
        after = tuple(
            sorted(
                scoped_value(item, current_specializations) for item in current_receipts
            )
        )
        if before != after:
            differences.append(IdentityDifference(axis, before, after))
    report_axes = (
        (
            "verification",
            stored.header.verification_spec,
            current.header.verification_spec,
        ),
        (
            "tolerance",
            stored.header.tolerance_policies,
            current.header.tolerance_policies,
        ),
        ("oracle", stored.header.oracle_references, current.header.oracle_references),
    )
    for axis, before, after in report_axes:
        encoded_before, encoded_after = (
            (_canonical_json(before),),
            (_canonical_json(after),),
        )
        if encoded_before != encoded_after:
            differences.append(IdentityDifference(axis, encoded_before, encoded_after))
    return tuple(differences)


def query_stale(
    store: EvidenceStore,
    *,
    selected_target_profile: str,
    current_report: VerificationReport,
) -> tuple[RunStaleness, ...]:
    """Compare stored reports with a supplied local baseline without network access."""

    profile = _required(selected_target_profile, "selected_target_profile")
    if not isinstance(current_report, VerificationReport):
        raise TypeError("current_report must be a VerificationReport")
    if (
        current_report.header is None
        or current_report.header.selected_target_profile != profile
    ):
        raise VerificationStoreError(
            "current report does not match selected_target_profile"
        )
    rows = store.query(
        SQLStatement(
            "SELECT run_id, report_json FROM verification_runs WHERE selected_target_profile=? ORDER BY run_id",
            (profile,),
        )
    )
    result = []
    for row in rows:
        try:
            stored = VerificationReport.from_jsonl(_row_text(row, "report_json"))
        except ValueError as error:
            raise VerificationStoreError(
                "verification store contains an invalid v3 report"
            ) from error
        result.append(
            RunStaleness(_row_text(row, "run_id"), _differences(stored, current_report))
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MissingRequirement:
    """One current selected-target requirement without a matching observation."""

    requirement_id: str
    case_id: str
    kernel_id: str
    variant: str
    test_class: str

    def as_json_object(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def query_todo(
    store: EvidenceStore,
    *,
    selected_target_profile: str,
    current_report: VerificationReport,
) -> tuple[MissingRequirement, ...]:
    """Return the stable unranked missing-requirement difference for one profile."""

    profile = _registered_profile_id(selected_target_profile, "selected_target_profile")
    if not isinstance(current_report, VerificationReport):
        raise TypeError("current_report must be a VerificationReport")
    if current_report.header is None:  # narrowed for the profile and bundle below.
        raise VerificationStoreError("current report has no v3 header")
    if current_report.header.selected_target_profile != profile:
        raise VerificationStoreError(
            "current report does not match selected_target_profile"
        )
    rows = store.query(
        SQLStatement(
            "SELECT DISTINCT e.requirement_id FROM verification_runs r JOIN evidence e ON e.run_id=r.run_id JOIN observations o ON o.evidence_id=e.evidence_id WHERE r.todo_provenance_digest=? ORDER BY e.requirement_id",
            (_todo_provenance_digest(current_report),),
        )
    )
    observed = {_row_text(row, "requirement_id") for row in rows}
    missing = []
    for requirement in current_report.header.verification_spec["requirements"]:
        if requirement["requirement_id"] not in observed:
            case = requirement["case"]
            missing.append(
                MissingRequirement(
                    requirement["requirement_id"],
                    case["case_id"],
                    case["kernel_id"],
                    case["variant"],
                    requirement["test_class"],
                )
            )
    return tuple(
        sorted(
            missing,
            key=lambda item: (
                item.kernel_id,
                item.variant,
                item.test_class,
                item.case_id,
                item.requirement_id,
            ),
        )
    )
