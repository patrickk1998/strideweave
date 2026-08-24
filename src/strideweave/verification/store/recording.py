"""Validated, atomic persistence of schema-v3 verification reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from ..classification import verification_profile
from ..model import (
    EvidenceRecord,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
)
from ..provenance import compilation_receipt_json_object
from ..reporting import _certificate_from_value, bind_report
from ..stage_one import _oracle_requirement_records
from ..stage_two import (
    _current_profile_compilation_bundle,
    _oracle_authorizations_from_records,
    _target_requirement_records,
)
from ._identity import _canonical_json, _digest, _thaw, _todo_provenance_digest
from .base import EvidenceStore, SQLStatement, VerificationStoreError


def _text(
    value: object, field: str, maximum: int, *, optional: bool = False
) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value or len(value) > maximum:
        raise VerificationStoreError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    return value


def _digest_text(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise VerificationStoreError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _recording_time(
    value: object, field: str, *, allow_naive_utc: bool = False
) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        result = value
    elif type(value) is str:
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise VerificationStoreError(
                f"{field} must be an ISO-8601 timestamp"
            ) from error
    else:
        raise VerificationStoreError(
            f"{field} must be a timezone-aware datetime or ISO-8601 timestamp"
        )
    if result.tzinfo is None:
        if not allow_naive_utc:
            raise VerificationStoreError(f"{field} must include a timezone")
        # Dolt returns its DATETIME(6) values without an offset. Only this
        # trusted readback path may recover the UTC normalization used on write.
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _insert(
    table: str, columns: Sequence[str], values: Sequence[object]
) -> SQLStatement:
    return SQLStatement(
        f"INSERT IGNORE INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        tuple(values),  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class RecordResult:
    """Immutable identities and row counts for one recorded v3 report."""

    run_id: str
    report_digest: str
    evidence_count: int
    observation_count: int


def _attempt_identity(record: EvidenceRecord) -> str:
    """Return current-model facts that identify one required execution attempt."""

    value = record.as_json_object()
    return _canonical_json(
        {
            "case": value["case"],
            "oracle_input_bit_hashes": value["oracle_input_bit_hashes"],
            "schema_version": value["schema_version"],
            "stage": value["stage"],
            "target_input_bit_hashes": value["target_input_bit_hashes"],
            "test_class": value["test_class"],
            "tolerance": value["tolerance"],
        }
    )


def _validate_installed_verification_graph(report: VerificationReport) -> None:
    """Compare untrusted evidence to the complete execution-free current graph."""

    header = report.header
    if header is None:  # pragma: no cover - checked by the public caller.
        raise VerificationStoreError("verification report has no v3 provenance header")
    try:
        oracle_profile = verification_profile(header.oracle_profile)
        target_profile = verification_profile(header.selected_target_profile)
        oracle_requirements = _oracle_requirement_records(oracle_profile)
        target_requirements = _target_requirement_records(
            target_profile, oracle_requirements
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise VerificationStoreError(
            f"could not reconstruct the current installed verification graph: {error}"
        ) from error

    expected = tuple(
        _attempt_identity(record)
        for record in (*oracle_requirements, *target_requirements)
    )
    observed = tuple(_attempt_identity(record) for record in report.records)
    if (
        len(expected) != len(set(expected))
        or len(observed) != len(set(observed))
        or set(observed) != set(expected)
    ):
        raise VerificationStoreError(
            "report evidence does not match the complete installed verification graph"
        )

    try:
        certificates = tuple(
            _certificate_from_value(
                _thaw(value), f"report header.certificates[{index}]"
            )
            for index, value in enumerate(header.certificates)
        )
        oracle_records = tuple(
            record
            for record in report.records
            if record.stage is VerificationStage.ORACLE
        )
        authorizations = _oracle_authorizations_from_records(
            oracle_profile, oracle_records, certificates
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise VerificationStoreError(
            f"report oracle certificates do not match current requirements: {error}"
        ) from error

    compiled_oracle_operations = {
        receipt.logical_kernel.operation
        for receipt in header.compilation_bundle.receipts
        if receipt.profile_id == header.oracle_profile
    }
    for record in report.records:
        if record.stage is not VerificationStage.TARGET:
            continue
        requires_authorization = (
            record.case.operation in compiled_oracle_operations
            and record.outcome
            not in {VerificationOutcome.BLOCKED, VerificationOutcome.DEFERRED}
        )
        expected_digest = authorizations.get(record.case.operation)
        if requires_authorization and (
            expected_digest is None
            or record.consumed_certificate_digest != expected_digest
        ):
            raise VerificationStoreError(
                "target evidence does not consume its exact current oracle certificate"
            )
        if (
            record.outcome is VerificationOutcome.BLOCKED
            and expected_digest is not None
        ):
            raise VerificationStoreError(
                "target evidence is blocked despite an exact current oracle certificate"
            )


def _validate_current_report(report: VerificationReport) -> None:
    """Rebind installed identities before a store is even initialized."""

    if report.header is None:
        raise VerificationStoreError("verification report has no v3 provenance header")
    if report.schema_version != "strideweave.kernel-verification.v3":
        raise VerificationStoreError(
            "record accepts only schema-v3 verification reports"
        )
    profile_ids = tuple(
        dict.fromkeys(
            (
                report.header.oracle_profile,
                report.header.selected_target_profile,
            )
        )
    )
    for profile_id in profile_ids:
        stored = tuple(
            receipt
            for receipt in report.header.compilation_bundle.receipts
            if receipt.profile_id == profile_id
        )
        try:
            profile = verification_profile(profile_id)
            current = _current_profile_compilation_bundle(profile, stored).receipts
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise VerificationStoreError(
                f"could not resolve current compilation provenance for profile "
                f"{profile_id!r}: {error}"
            ) from error
        if tuple(compilation_receipt_json_object(item) for item in stored) != tuple(
            compilation_receipt_json_object(item) for item in current
        ):
            raise VerificationStoreError(
                f"report compilation receipts for profile {profile_id!r} are stale "
                "or do not match the current provider environment"
            )
    _validate_installed_verification_graph(report)
    try:
        records, header = bind_report(
            report.records,
            (),
            selected_target_profile=report.header.selected_target_profile,
            oracle_profile=report.header.oracle_profile,
            compilation_bundle=report.header.compilation_bundle,
            certificate_facts_override=report.header.certificates,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise VerificationStoreError(
            f"could not validate report against current provenance: {error}"
        ) from error
    if (
        records != report.records
        or header.as_json_object() != report.header.as_json_object()
    ):
        raise VerificationStoreError(
            "report provenance is stale or does not match this StrideWeave build"
        )


def _report_rows(
    report: VerificationReport,
    *,
    producer_id: str,
    source_commit: str | None,
    artifact_locator: str | None,
    artifact_digest: str | None,
    recorded_at: datetime,
) -> tuple[RecordResult, tuple[SQLStatement, ...]]:
    """Build one complete immutable v3 graph without touching a store."""

    if report.header is None:  # pragma: no cover - model guarantee.
        raise VerificationStoreError("verification report has no v3 provenance header")
    report_json = report.to_jsonl()
    report_digest = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
    run_id = _digest({"report_digest": report_digest})
    header = report.header
    statements: list[SQLStatement] = [
        _insert(
            "verification_runs",
            (
                "run_id",
                "report_digest",
                "report_schema",
                "selected_target_profile",
                "oracle_profile",
                "bundle_id",
                "todo_provenance_digest",
                "header_digest",
                "report_json",
            ),
            (
                run_id,
                report_digest,
                report.schema_version,
                header.selected_target_profile,
                header.oracle_profile,
                header.compilation_bundle.bundle_id,
                _todo_provenance_digest(report),
                header.header_digest,
                report_json,
            ),
        )
    ]
    for receipt in header.compilation_bundle.receipts:
        receipt_json = _canonical_json(compilation_receipt_json_object(receipt))
        statements.extend(
            (
                _insert(
                    "compilation_receipts",
                    (
                        "receipt_id",
                        "receipt_kind",
                        "profile_id",
                        "provider",
                        "logical_kernel_id",
                        "logical_kernel_variant",
                        "receipt_json",
                    ),
                    (
                        receipt.receipt_id,
                        receipt.kind,
                        receipt.profile_id,
                        receipt.provider,
                        receipt.logical_kernel.kernel_id,
                        receipt.logical_kernel.variant,
                        receipt_json,
                    ),
                ),
                _insert(
                    "run_compilation_receipts",
                    ("run_id", "receipt_id"),
                    (run_id, receipt.receipt_id),
                ),
            )
        )
    for record in report.records:
        record_json = _canonical_json(record.as_json_object())
        evidence_id = _digest({"record": json.loads(record_json), "run_id": run_id})
        statements.append(
            _insert(
                "evidence",
                (
                    "evidence_id",
                    "run_id",
                    "receipt_id",
                    "requirement_id",
                    "stage",
                    "test_class",
                    "case_id",
                    "operation_name",
                    "kernel_id",
                    "variant",
                    "outcome",
                    "record_json",
                ),
                (
                    evidence_id,
                    run_id,
                    record.compilation_receipt_id,
                    record.requirement_id,
                    record.stage.value,
                    record.test_class.value,
                    record.case.case_id,
                    record.case.operation,
                    record.case.kernel_id,
                    record.case.variant,
                    record.outcome.value,
                    record_json,
                ),
            )
        )
        identity = {
            "artifact_digest": artifact_digest,
            "artifact_locator": artifact_locator,
            "evidence_id": evidence_id,
            "producer_id": producer_id,
            "source_commit": source_commit,
        }
        observation_id = _digest(identity)
        observation_json = _canonical_json(
            {
                **identity,
                "observation_id": observation_id,
                "recorded_at_utc": recorded_at.isoformat(),
            }
        )
        statements.append(
            _insert(
                "observations",
                (
                    "observation_id",
                    "evidence_id",
                    "producer_id",
                    "source_commit",
                    "recorded_at_utc",
                    "artifact_locator",
                    "artifact_digest",
                    "observation_json",
                ),
                (
                    observation_id,
                    evidence_id,
                    producer_id,
                    source_commit,
                    recorded_at,
                    artifact_locator,
                    artifact_digest,
                    observation_json,
                ),
            )
        )
    return RecordResult(
        run_id, report_digest, len(report.records), len(report.records)
    ), tuple(statements)


def _record_validated_report(
    report: VerificationReport,
    store: EvidenceStore,
    *,
    producer_id: str,
    source_commit: str | None = None,
    artifact_locator: str | None = None,
    artifact_digest: str | None = None,
    recorded_at: datetime | None = None,
) -> RecordResult:
    """Atomically record a report that has already passed provenance rebinding."""

    if not isinstance(report, VerificationReport):
        raise TypeError("report must be a VerificationReport")
    if not isinstance(store, EvidenceStore):
        raise TypeError("store must implement EvidenceStore")
    producer = _text(producer_id, "producer_id", 255)
    assert producer is not None
    commit = _text(source_commit, "source_commit", 255, optional=True)
    locator = _text(artifact_locator, "artifact_locator", 1024, optional=True)
    artifact = _digest_text(artifact_digest, "artifact_digest", optional=True)
    result, statements = _report_rows(
        report,
        producer_id=producer,
        source_commit=commit,
        artifact_locator=locator,
        artifact_digest=artifact,
        recorded_at=_recording_time(recorded_at, "recorded_at"),
    )
    store.execute_transaction(statements)
    return result


def record_report(
    report: VerificationReport, store: EvidenceStore, **kwargs: object
) -> RecordResult:
    """Reconcile and atomically record one schema-v3 report as factual evidence."""

    _validate_current_report(report)
    return _record_validated_report(report, store, **kwargs)  # type: ignore[arg-type]
