"""Offline, read-only factual queries over recorded verification evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from ..api import test_backend
from ..model import VerificationReport
from .base import EvidenceStore, SQLStatement, VerificationStoreError
from .recording import _report_manifest


def _required_text(value: str, field: str) -> str:
    if type(value) is not str or not value:
        raise VerificationStoreError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


def _string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if type(value) is not str:
        raise VerificationStoreError(
            f"verification store returned invalid {field} query data"
        )
    return value


def _optional_string(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if type(value) is not str:
        raise VerificationStoreError(
            f"verification store returned invalid {field} query data"
        )
    return value


def _json_object(row: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = row.get(field)
    try:
        parsed = json.loads(value) if type(value) is str else None
    except json.JSONDecodeError as error:
        raise VerificationStoreError(
            f"verification store returned invalid {field} JSON"
        ) from error
    if type(parsed) is not dict:
        raise VerificationStoreError(
            f"verification store returned invalid {field} query data"
        )
    return MappingProxyType(parsed)


@dataclass(frozen=True, slots=True)
class StatusObservation:
    """One unaggregated producer observation and its exact evidence facts."""

    run_id: str
    evidence_id: str
    observation_id: str
    architecture: str
    target_id: str
    represented_target_id: str
    proxy_id: str | None
    stage: str
    test_class: str
    case_id: str
    operation: str
    kernel_id: str
    variant: str
    outcome: str
    deviations: Mapping[str, object]
    tolerance: Mapping[str, object]
    producer_id: str
    source_commit: str | None
    recorded_at_utc: str
    artifact_locator: str | None
    artifact_digest: str | None
    compilation_receipt_id: str | None
    source_closure_id: str | None
    toolchain_id: str | None
    verification_spec_id: str
    tolerance_policy_id: str | None
    oracle_reference_id: str | None
    consumed_certificate_digest: str | None
    native_manifest_digest: str

    def as_json_object(self) -> dict[str, object]:
        """Return stable machine-readable fields for this factual observation."""

        return {
            "architecture": self.architecture,
            "artifact_digest": self.artifact_digest,
            "artifact_locator": self.artifact_locator,
            "case_id": self.case_id,
            "compilation_receipt_id": self.compilation_receipt_id,
            "consumed_certificate_digest": self.consumed_certificate_digest,
            "deviations": dict(self.deviations),
            "evidence_id": self.evidence_id,
            "kernel_id": self.kernel_id,
            "native_manifest_digest": self.native_manifest_digest,
            "observation_id": self.observation_id,
            "operation": self.operation,
            "oracle_reference_id": self.oracle_reference_id,
            "outcome": self.outcome,
            "producer_id": self.producer_id,
            "proxy_id": self.proxy_id,
            "recorded_at_utc": self.recorded_at_utc,
            "represented_target_id": self.represented_target_id,
            "run_id": self.run_id,
            "source_closure_id": self.source_closure_id,
            "source_commit": self.source_commit,
            "stage": self.stage,
            "target_id": self.target_id,
            "test_class": self.test_class,
            "tolerance": dict(self.tolerance),
            "tolerance_policy_id": self.tolerance_policy_id,
            "toolchain_id": self.toolchain_id,
            "variant": self.variant,
            "verification_spec_id": self.verification_spec_id,
        }


def query_status(
    store: EvidenceStore,
    *,
    architecture: str,
    kernel_id: str | None = None,
    variant: str | None = None,
    test_class: str | None = None,
    producer_id: str | None = None,
) -> tuple[StatusObservation, ...]:
    """Return stable, unaggregated observations matching exact factual filters."""

    architecture = _required_text(architecture, "architecture")
    filters = (
        ("e.kernel_id", _optional_text(kernel_id, "kernel_id")),
        ("e.variant", _optional_text(variant, "variant")),
        ("e.test_class", _optional_text(test_class, "test_class")),
        ("o.producer_id", _optional_text(producer_id, "producer_id")),
    )
    clauses = ["t.architecture = ?"]
    parameters: list[str] = [architecture]
    for column, value in filters:
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    rows = store.query(
        SQLStatement(
            "SELECT r.run_id, e.evidence_id, o.observation_id, t.architecture, "
            "r.execution_target_id AS target_id, r.represented_target_id, r.proxy_id, "
            "e.stage, e.test_class, e.case_id, e.operation_name, e.kernel_id, "
            "e.variant, e.outcome, e.deviations_json, p.definition_json AS tolerance_json, "
            "o.producer_id, o.source_commit, o.recorded_at_utc, o.artifact_locator, "
            "o.artifact_digest, e.kernel_build_id, k.closure_id, k.toolchain_id, "
            "r.verification_spec_id, e.tolerance_policy_id, e.oracle_reference_id, "
            "e.consumed_certificate_digest, r.native_manifest_digest "
            "FROM observations o JOIN evidence e ON e.evidence_id = o.evidence_id "
            "JOIN verification_runs r ON r.run_id = e.run_id "
            "JOIN verification_targets t ON t.target_id = r.execution_target_id "
            "LEFT JOIN tolerance_policies p "
            "ON p.tolerance_policy_id = e.tolerance_policy_id "
            "LEFT JOIN kernel_builds k ON k.kernel_build_id = e.kernel_build_id "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY e.kernel_id, e.variant, e.test_class, e.case_id, "
            "o.producer_id, o.observation_id",
            tuple(parameters),
        )
    )
    return tuple(
        StatusObservation(
            run_id=_string(row, "run_id"),
            evidence_id=_string(row, "evidence_id"),
            observation_id=_string(row, "observation_id"),
            architecture=_string(row, "architecture"),
            target_id=_string(row, "target_id"),
            represented_target_id=_string(row, "represented_target_id"),
            proxy_id=_optional_string(row, "proxy_id"),
            stage=_string(row, "stage"),
            test_class=_string(row, "test_class"),
            case_id=_string(row, "case_id"),
            operation=_string(row, "operation_name"),
            kernel_id=_string(row, "kernel_id"),
            variant=_string(row, "variant"),
            outcome=_string(row, "outcome"),
            deviations=_json_object(row, "deviations_json"),
            tolerance=_json_object(row, "tolerance_json"),
            producer_id=_string(row, "producer_id"),
            source_commit=_optional_string(row, "source_commit"),
            recorded_at_utc=_string(row, "recorded_at_utc"),
            artifact_locator=_optional_string(row, "artifact_locator"),
            artifact_digest=_optional_string(row, "artifact_digest"),
            compilation_receipt_id=_optional_string(row, "kernel_build_id"),
            source_closure_id=_optional_string(row, "closure_id"),
            toolchain_id=_optional_string(row, "toolchain_id"),
            verification_spec_id=_string(row, "verification_spec_id"),
            tolerance_policy_id=_optional_string(row, "tolerance_policy_id"),
            oracle_reference_id=_optional_string(row, "oracle_reference_id"),
            consumed_certificate_digest=_optional_string(
                row, "consumed_certificate_digest"
            ),
            native_manifest_digest=_string(row, "native_manifest_digest"),
        )
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class IdentityDifference:
    """One independently reported identity-axis difference for a stored run."""

    axis: str
    stored: tuple[str, ...]
    current: tuple[str, ...]
    details: tuple[str, ...] = ()

    def as_json_object(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "current": list(self.current),
            "details": list(self.details),
            "stored": list(self.stored),
        }


@dataclass(frozen=True, slots=True)
class RunStaleness:
    """Exact identity differences between one stored run and current facts."""

    run_id: str
    differences: tuple[IdentityDifference, ...]

    @property
    def is_stale(self) -> bool:
        return bool(self.differences)

    def as_json_object(self) -> dict[str, object]:
        return {
            "differences": [item.as_json_object() for item in self.differences],
            "is_stale": self.is_stale,
            "run_id": self.run_id,
        }


@lru_cache(maxsize=1)
def _current_report() -> VerificationReport:
    return test_backend()


def _set_difference(
    axis: str, stored: set[str], current: set[str], details: tuple[str, ...] = ()
) -> IdentityDifference | None:
    if stored == current and not details:
        return None
    return IdentityDifference(
        axis, tuple(sorted(stored)), tuple(sorted(current)), details
    )


def _closure_details(
    store: EvidenceStore, run_id: str, current_report: VerificationReport
) -> tuple[str, ...]:
    stored_rows = store.query(
        SQLStatement(
            "SELECT k.kernel_id, k.variant, i.input_uri, i.content_digest "
            "FROM run_kernel_builds rb "
            "JOIN kernel_builds k ON k.kernel_build_id = rb.kernel_build_id "
            "JOIN source_closure_inputs i ON i.closure_id = k.closure_id "
            "WHERE rb.run_id = ? ORDER BY k.kernel_id, k.variant, i.input_uri",
            (run_id,),
        )
    )
    stored = {
        (
            _string(row, "kernel_id"),
            _string(row, "variant"),
            _string(row, "input_uri"),
        ): _string(row, "content_digest")
        for row in stored_rows
    }
    manifest = _report_manifest(current_report)
    current = {
        (
            receipt.kernel.kernel_id,
            receipt.kernel.variant,
            item.uri,
        ): item.content_digest
        for receipt in manifest.receipts
        for item in receipt.inputs
    }
    details = []
    for key in sorted(stored.keys() | current.keys()):
        kernel, variant, uri = key
        before = stored.get(key)
        after = current.get(key)
        if before == after:
            continue
        change = (
            "added" if before is None else "removed" if after is None else "changed"
        )
        details.append(f"{kernel}/{variant}:{uri}:{change}")
    return tuple(details)


def query_stale(store: EvidenceStore, *, architecture: str) -> tuple[RunStaleness, ...]:
    """Explain every stored run's identity differences from current local facts."""

    architecture = _required_text(architecture, "architecture")
    current_report = _current_report()
    manifest = _report_manifest(current_report)
    if manifest.target.architecture != architecture:
        raise VerificationStoreError(
            f"installed target architecture is {manifest.target.architecture!r}, "
            f"not {architecture!r}"
        )
    if current_report.header is None:  # pragma: no cover
        raise VerificationStoreError("current verification report has no header")
    run_rows = store.query(
        SQLStatement(
            "SELECT r.run_id, r.native_manifest_digest, r.verification_spec_id, "
            "r.execution_target_id FROM verification_runs r "
            "JOIN verification_targets t ON t.target_id = r.execution_target_id "
            "WHERE t.architecture = ? ORDER BY r.run_id",
            (architecture,),
        )
    )
    current_policy_ids = {
        item["tolerance_policy_id"] for item in current_report.header.tolerance_policies
    }
    current_oracle_ids = {
        item["oracle_reference_id"] for item in current_report.header.oracle_references
    }
    current_receipts = {receipt.receipt_id for receipt in manifest.receipts}
    results = []
    for row in run_rows:
        run_id = _string(row, "run_id")
        identity_rows = store.query(
            SQLStatement(
                "SELECT DISTINCT k.kernel_build_id, k.closure_id, k.toolchain_id, "
                "e.tolerance_policy_id, e.oracle_reference_id "
                "FROM run_kernel_builds rb "
                "JOIN kernel_builds k ON k.kernel_build_id = rb.kernel_build_id "
                "LEFT JOIN evidence e ON e.run_id = rb.run_id "
                "WHERE rb.run_id = ?",
                (run_id,),
            )
        )
        stored_receipts = {_string(item, "kernel_build_id") for item in identity_rows}
        stored_closures = {_string(item, "closure_id") for item in identity_rows}
        stored_toolchains = {_string(item, "toolchain_id") for item in identity_rows}
        stored_policies = {
            value
            for item in identity_rows
            if (value := _optional_string(item, "tolerance_policy_id")) is not None
        }
        stored_oracles = {
            value
            for item in identity_rows
            if (value := _optional_string(item, "oracle_reference_id")) is not None
        }
        differences = []
        axes = (
            _set_difference(
                "compilation_manifest",
                {_string(row, "native_manifest_digest")},
                {manifest.manifest_digest},
            ),
            _set_difference("compilation_receipts", stored_receipts, current_receipts),
            _set_difference(
                "source_closure",
                stored_closures,
                {receipt.closure_id for receipt in manifest.receipts},
                _closure_details(store, run_id, current_report),
            ),
            _set_difference(
                "toolchain", stored_toolchains, {manifest.toolchain.toolchain_id}
            ),
            _set_difference(
                "target",
                {_string(row, "execution_target_id")},
                {manifest.target.target_id},
            ),
            _set_difference(
                "verification_specification",
                {_string(row, "verification_spec_id")},
                {current_report.header.verification_spec["verification_spec_id"]},
            ),
            _set_difference("tolerance_policy", stored_policies, current_policy_ids),
            _set_difference("oracle", stored_oracles, current_oracle_ids),
        )
        differences.extend(item for item in axes if item is not None)
        results.append(RunStaleness(run_id, tuple(differences)))
    return tuple(results)


@dataclass(frozen=True, slots=True)
class MissingRequirement:
    """One current verification requirement with no matching observation."""

    requirement_id: str
    kernel_id: str
    variant: str
    test_class: str
    case_id: str
    operation: str

    def as_json_object(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "kernel_id": self.kernel_id,
            "operation": self.operation,
            "requirement_id": self.requirement_id,
            "test_class": self.test_class,
            "variant": self.variant,
        }


def query_todo(
    store: EvidenceStore, *, architecture: str
) -> tuple[MissingRequirement, ...]:
    """Return the stable unranked set of current requirements lacking evidence."""

    architecture = _required_text(architecture, "architecture")
    current_report = _current_report()
    manifest = _report_manifest(current_report)
    if manifest.target.architecture != architecture:
        raise VerificationStoreError(
            f"installed target architecture is {manifest.target.architecture!r}, "
            f"not {architecture!r}"
        )
    if current_report.header is None:  # pragma: no cover
        raise VerificationStoreError("current verification report has no header")
    matched_rows = store.query(
        SQLStatement(
            "SELECT DISTINCT e.requirement_id FROM evidence e "
            "JOIN observations o ON o.evidence_id = e.evidence_id "
            "JOIN verification_runs r ON r.run_id = e.run_id "
            "WHERE r.native_manifest_digest = ? AND r.verification_spec_id = ? "
            "AND r.execution_target_id = ?",
            (
                manifest.manifest_digest,
                current_report.header.verification_spec["verification_spec_id"],
                manifest.target.target_id,
            ),
        )
    )
    matched = {_string(row, "requirement_id") for row in matched_rows}
    missing = []
    for requirement in current_report.header.verification_spec["requirements"]:
        if requirement["requirement_id"] in matched:
            continue
        case = requirement["case"]
        missing.append(
            MissingRequirement(
                requirement_id=requirement["requirement_id"],
                kernel_id=case["kernel_id"],
                variant=case["variant"],
                test_class=requirement["test_class"],
                case_id=case["case_id"],
                operation=case["operation"],
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
            ),
        )
    )
