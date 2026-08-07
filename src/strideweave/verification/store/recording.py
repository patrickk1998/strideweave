"""Validated, atomic persistence of provenance-complete verification reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from ..model import KernelDescriptor, VerificationReport
from ..provenance import CompilationManifest, parse_compilation_manifest
from ..reporting import bind_report
from .base import (
    EvidenceStore,
    SQLStatement,
    VerificationStoreError,
    is_diagnostic_closure_input,
)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _thaw(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _require_text(value: object, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > maximum:
        raise VerificationStoreError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    return value


def _require_digest(value: object, field: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VerificationStoreError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _recording_time(value: object, field: str) -> datetime:
    if value is None:
        result = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        result = value
        if result.tzinfo is None:
            raise VerificationStoreError(f"{field} must be a timezone-aware datetime")
    elif type(value) is str:
        try:
            result = datetime.fromisoformat(
                value.replace(" ", "T").replace("Z", "+00:00")
            )
        except ValueError as error:
            raise VerificationStoreError(
                f"{field} must be an ISO-8601 timestamp"
            ) from error
        if result.tzinfo is None:
            # Dolt's DATETIME column returns the UTC recording time without an
            # offset. Snapshot text is therefore normalized as UTC on import.
            result = result.replace(tzinfo=timezone.utc)
    else:
        raise VerificationStoreError(
            f"{field} must be a timezone-aware datetime or ISO-8601 timestamp"
        )
    return result.astimezone(timezone.utc)


def _validate_observation_provenance(
    *,
    producer_id: object,
    source_commit: object,
    artifact_locator: object,
    artifact_digest: object,
    recorded_at: object,
) -> tuple[str, str | None, str | None, str | None, datetime]:
    producer = _require_text(producer_id, "producer_id", 255)
    if producer is None:
        raise VerificationStoreError("producer_id is required")
    return (
        producer,
        _require_text(source_commit, "source_commit", 255),
        _require_text(artifact_locator, "artifact_locator", 1024),
        _require_digest(artifact_digest, "artifact_digest"),
        _recording_time(recorded_at, "recorded_at"),
    )


def _insert(
    table: str, columns: Sequence[str], values: Sequence[object]
) -> SQLStatement:
    placeholders = ", ".join("?" for _ in columns)
    return SQLStatement(
        f"INSERT IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values),  # type: ignore[arg-type]
    )


def _report_manifest(report: VerificationReport) -> CompilationManifest:
    if report.header is None:  # pragma: no cover - model construction prevents this.
        raise VerificationStoreError("verification report has no provenance header")
    kernels = tuple(
        KernelDescriptor(
            operation=item["kernel"]["operation"],
            kernel_id=item["kernel"]["kernel_id"],
            variant=item["kernel"]["variant"],
            pybind_name=item["kernel"]["pybind_name"],
            owning_source=item["kernel"]["owning_source"],
        )
        for item in report.header.compilation["kernel_receipts"]
    )
    return parse_compilation_manifest(
        _thaw(report.header.compilation["manifest"]), kernels=kernels
    )


def _validate_current_report(report: VerificationReport) -> None:
    if report.header is None:  # pragma: no cover - model construction prevents this.
        raise VerificationStoreError("verification report has no provenance header")
    try:
        rebound_records, current_header = bind_report(
            report.records,
            (),
            certificate_facts_override=report.header.certificates,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise VerificationStoreError(
            f"could not validate report against current provenance: {error}"
        ) from error
    if (
        rebound_records != report.records
        or current_header.as_json_object() != report.header.as_json_object()
    ):
        raise VerificationStoreError(
            "report provenance is stale or does not match this StrideWeave build"
        )


@dataclass(frozen=True, slots=True)
class RecordResult:
    """Immutable identities and row counts for one recorded report."""

    run_id: str
    report_digest: str
    evidence_count: int
    observation_count: int


def _target_statements(manifest: CompilationManifest) -> list[SQLStatement]:
    target = manifest.target
    target_value = asdict(target)
    toolchain = manifest.toolchain
    toolchain_value = asdict(toolchain)
    return [
        _insert(
            "verification_targets",
            (
                "target_id",
                "architecture",
                "vendor",
                "operating_system",
                "abi",
                "endianness",
                "pointer_bits",
                "descriptor_json",
            ),
            (
                target.target_id,
                target.architecture,
                target.vendor,
                target.operating_system,
                target.abi,
                target.endianness,
                target.pointer_bits,
                _canonical_json(target_value),
            ),
        ),
        _insert(
            "build_toolchains",
            (
                "toolchain_id",
                "provider_kind",
                "compiler_id",
                "compiler_version",
                "target_triple",
                "build_system",
                "descriptor_json",
            ),
            (
                toolchain.toolchain_id,
                toolchain.provider_kind,
                toolchain.compiler_id,
                toolchain.compiler_version,
                toolchain.target_triple,
                toolchain.build_system,
                _canonical_json(toolchain_value),
            ),
        ),
    ]


def _compilation_statements(manifest: CompilationManifest) -> list[SQLStatement]:
    statements: list[SQLStatement] = []
    closures: dict[str, object] = {}
    for receipt in manifest.receipts:
        closure_value = {
            "compile_invocation": receipt.compile_invocation,
            "inputs": tuple(asdict(item) for item in receipt.inputs),
            "owning_source": receipt.kernel.owning_source,
        }
        if receipt.closure_id not in closures:
            closures[receipt.closure_id] = closure_value
            statements.append(
                _insert(
                    "source_closures",
                    (
                        "closure_id",
                        "hash_algorithm",
                        "root_kind",
                        "root_uri",
                        "descriptor_json",
                    ),
                    (
                        receipt.closure_id,
                        "sha256",
                        "repository-source",
                        receipt.kernel.owning_source,
                        _canonical_json(closure_value),
                    ),
                )
            )
            # The descriptor above already carries the complete closure, so
            # ordinals stay those of the complete input sequence and only the
            # diagnostic members become rows of their own.
            for ordinal, source_input in enumerate(receipt.inputs):
                if not is_diagnostic_closure_input(source_input.input_kind):
                    continue
                statements.append(
                    _insert(
                        "source_closure_inputs",
                        (
                            "closure_id",
                            "input_ordinal",
                            "input_kind",
                            "input_uri",
                            "content_digest",
                            "descriptor_json",
                        ),
                        (
                            receipt.closure_id,
                            ordinal,
                            source_input.input_kind,
                            source_input.uri,
                            source_input.content_digest,
                            _canonical_json(asdict(source_input)),
                        ),
                    )
                )
        receipt_value = {
            "artifact_digest": receipt.shared_artifact_digest,
            "closure_id": receipt.closure_id,
            "compile_invocation": receipt.compile_invocation,
            "compile_invocation_digest": receipt.compile_invocation_digest,
            "framework_name": receipt.framework_name,
            "framework_version": receipt.framework_version,
            "kernel": asdict(receipt.kernel),
            "object_digest": receipt.object_digest,
            "provider_kind": receipt.provider_kind,
            "receipt_id": receipt.receipt_id,
            "receipt_schema": receipt.receipt_schema,
            "specialization": receipt.specialization,
            "target_id": receipt.target.target_id,
            "toolchain_id": receipt.toolchain.toolchain_id,
        }
        statements.append(
            _insert(
                "kernel_builds",
                (
                    "kernel_build_id",
                    "receipt_schema",
                    "provider_kind",
                    "framework_name",
                    "framework_version",
                    "kernel_id",
                    "variant",
                    "operation_name",
                    "artifact_digest",
                    "artifact_locator",
                    "compile_invocation_digest",
                    "target_id",
                    "toolchain_id",
                    "closure_id",
                    "specialization_json",
                    "receipt_json",
                ),
                (
                    receipt.receipt_id,
                    receipt.receipt_schema,
                    receipt.provider_kind,
                    receipt.framework_name,
                    receipt.framework_version,
                    receipt.kernel.kernel_id,
                    receipt.kernel.variant,
                    receipt.kernel.operation,
                    receipt.object_digest,
                    None,
                    receipt.compile_invocation_digest,
                    receipt.target.target_id,
                    receipt.toolchain.toolchain_id,
                    receipt.closure_id,
                    _canonical_json(receipt.specialization),
                    _canonical_json(receipt_value),
                ),
            )
        )
    return statements


def _specification_statements(report: VerificationReport) -> list[SQLStatement]:
    if report.header is None:  # pragma: no cover
        raise VerificationStoreError("verification report has no provenance header")
    specification = report.header.verification_spec
    manifest = report.header.compilation["manifest"]
    spec_id = specification["verification_spec_id"]
    statements = [
        _insert(
            "verification_specs",
            (
                "verification_spec_id",
                "spec_schema",
                "manifest_digest",
                "definition_json",
            ),
            (
                spec_id,
                specification["spec_schema"],
                manifest["manifest_digest"],
                _canonical_json(specification),
            ),
        )
    ]
    for requirement in specification["requirements"]:
        case = requirement["case"]
        deferred = requirement["test_class"] == "deferred"
        statements.append(
            _insert(
                "verification_requirements",
                (
                    "requirement_id",
                    "verification_spec_id",
                    "kernel_id",
                    "variant",
                    "operation_name",
                    "test_class",
                    "case_id",
                    "disposition",
                    "deferred_reason",
                    "plan_json",
                    "requirement_json",
                ),
                (
                    requirement["requirement_id"],
                    spec_id,
                    case["kernel_id"],
                    case["variant"],
                    case["operation"],
                    requirement["test_class"],
                    case["case_id"],
                    "deferred" if deferred else "active",
                    "declared deferred coverage" if deferred else None,
                    None if case["plan"] is None else _canonical_json(case["plan"]),
                    _canonical_json(requirement),
                ),
            )
        )
    return statements


def _policy_and_oracle_statements(report: VerificationReport) -> list[SQLStatement]:
    if report.header is None:  # pragma: no cover
        raise VerificationStoreError("verification report has no provenance header")
    statements: list[SQLStatement] = []
    for policy in report.header.tolerance_policies:
        statements.append(
            _insert(
                "tolerance_policies",
                (
                    "tolerance_policy_id",
                    "policy_schema",
                    "comparison_kind",
                    "definition_json",
                ),
                (
                    policy["tolerance_policy_id"],
                    policy["policy_schema"],
                    "absolute-relative-ulps",
                    _canonical_json(policy),
                ),
            )
        )
    for oracle in report.header.oracle_references:
        inputs = oracle["inputs"]
        closure_id = oracle["implementation_digest"]
        closure_value = {
            "inputs": inputs,
            "oracle_kind": oracle["oracle_kind"],
        }
        statements.append(
            _insert(
                "source_closures",
                (
                    "closure_id",
                    "hash_algorithm",
                    "root_kind",
                    "root_uri",
                    "descriptor_json",
                ),
                (
                    closure_id,
                    "sha256",
                    "python-package",
                    "src/strideweave",
                    _canonical_json(closure_value),
                ),
            )
        )
        for ordinal, source_input in enumerate(inputs):
            statements.append(
                _insert(
                    "source_closure_inputs",
                    (
                        "closure_id",
                        "input_ordinal",
                        "input_kind",
                        "input_uri",
                        "content_digest",
                        "descriptor_json",
                    ),
                    (
                        closure_id,
                        ordinal,
                        "source",
                        source_input["uri"],
                        source_input["content_digest"],
                        _canonical_json(source_input),
                    ),
                )
            )
        statements.append(
            _insert(
                "oracle_references",
                (
                    "oracle_reference_id",
                    "oracle_kind",
                    "implementation_digest",
                    "source_closure_id",
                    "kernel_build_id",
                    "descriptor_json",
                ),
                (
                    oracle["oracle_reference_id"],
                    oracle["oracle_kind"],
                    oracle["implementation_digest"],
                    closure_id,
                    None,
                    _canonical_json(oracle),
                ),
            )
        )
    return statements


def _run_and_evidence_statements(
    report: VerificationReport,
    manifest: CompilationManifest,
    *,
    producer_id: str,
    source_commit: str | None,
    recorded_at: datetime,
    artifact_locator: str | None,
    artifact_digest: str | None,
) -> tuple[str, str, list[SQLStatement]]:
    if report.header is None:  # pragma: no cover
        raise VerificationStoreError("verification report has no provenance header")
    report_json = report.to_jsonl()
    report_digest = hashlib.sha256(report_json.encode()).hexdigest()
    run_id = _digest({"report_digest": report_digest})
    spec_id = report.header.verification_spec["verification_spec_id"]
    statements = [
        _insert(
            "verification_runs",
            (
                "run_id",
                "report_schema",
                "report_digest",
                "native_manifest_digest",
                "verification_spec_id",
                "execution_target_id",
                "represented_target_id",
                "proxy_id",
                "report_json",
            ),
            (
                run_id,
                report.schema_version,
                report_digest,
                manifest.manifest_digest,
                spec_id,
                manifest.target.target_id,
                manifest.target.target_id,
                None,
                report_json,
            ),
        )
    ]
    receipt_ids = {receipt.receipt_id for receipt in manifest.receipts}
    for receipt_id in sorted(receipt_ids):
        statements.append(
            _insert(
                "run_kernel_builds",
                ("run_id", "kernel_build_id"),
                (run_id, receipt_id),
            )
        )
    for record in report.records:
        record_value = record.as_json_object()
        evidence_id = _digest({"record": record_value, "run_id": run_id})
        input_payload_digest = _digest(record.target_input_bit_hashes)
        plan = record_value["case"]["plan"]
        statements.append(
            _insert(
                "evidence",
                (
                    "evidence_id",
                    "run_id",
                    "requirement_id",
                    "kernel_build_id",
                    "tolerance_policy_id",
                    "oracle_reference_id",
                    "consumed_certificate_digest",
                    "stage",
                    "test_class",
                    "case_id",
                    "operation_name",
                    "kernel_id",
                    "variant",
                    "outcome",
                    "input_payload_digest",
                    "target_input_hashes_json",
                    "oracle_input_hashes_json",
                    "plan_json",
                    "shapes_json",
                    "deviations_json",
                    "mismatch_count",
                    "diagnostic",
                    "evidence_json",
                ),
                (
                    evidence_id,
                    run_id,
                    record.requirement_id,
                    record.compilation_receipt_id,
                    record.tolerance_policy_id,
                    record.oracle_reference_id,
                    record.consumed_certificate_digest,
                    record.stage.value,
                    record.test_class.value,
                    record.case.case_id,
                    record.case.operation,
                    record.case.kernel_id,
                    record.case.variant,
                    record.outcome.value,
                    input_payload_digest,
                    _canonical_json(record.target_input_bit_hashes),
                    _canonical_json(record.oracle_input_bit_hashes),
                    None if plan is None else _canonical_json(plan),
                    _canonical_json(record.case.shapes),
                    _canonical_json(record_value["deviations"]),
                    record.mismatches,
                    record.diagnostic,
                    _canonical_json(record_value),
                ),
            )
        )
        observation_identity = {
            "artifact_digest": artifact_digest,
            "artifact_locator": artifact_locator,
            "evidence_id": evidence_id,
            "producer_id": producer_id,
            "source_commit": source_commit,
        }
        observation_id = _digest(observation_identity)
        observation_value = {
            **observation_identity,
            "observation_id": observation_id,
            "recorded_at_utc": recorded_at.isoformat(),
        }
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
                    _canonical_json(observation_value),
                ),
            )
        )
    return run_id, report_digest, statements


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
    """Atomically persist one report whose provenance is already validated.

    This is the boundary between deciding that a report describes this build
    and writing its facts. It exists so persistence can be exercised against
    prevalidated evidence without re-running the native reconciliation that
    :func:`record_report` performs, and it is not a public entry point:
    reaching the store without that reconciliation is only ever correct for
    evidence a caller has already established as current.
    """

    producer, commit, locator, artifact, timestamp = _validate_observation_provenance(
        producer_id=producer_id,
        source_commit=source_commit,
        artifact_locator=artifact_locator,
        artifact_digest=artifact_digest,
        recorded_at=recorded_at,
    )
    manifest = _report_manifest(report)

    statements = _target_statements(manifest)
    statements.extend(_compilation_statements(manifest))
    statements.extend(_specification_statements(report))
    statements.extend(_policy_and_oracle_statements(report))
    run_id, report_digest, run_statements = _run_and_evidence_statements(
        report,
        manifest,
        producer_id=producer,
        source_commit=commit,
        recorded_at=timestamp,
        artifact_locator=locator,
        artifact_digest=artifact,
    )
    statements.extend(run_statements)
    store.execute_transaction(statements)
    return RecordResult(
        run_id=run_id,
        report_digest=report_digest,
        evidence_count=len(report.records),
        observation_count=len(report.records),
    )


def record_report(
    report: VerificationReport,
    store: EvidenceStore,
    *,
    producer_id: str,
    source_commit: str | None = None,
    artifact_locator: str | None = None,
    artifact_digest: str | None = None,
    recorded_at: datetime | None = None,
) -> RecordResult:
    """Validate and atomically record one report as immutable raw evidence.

    Validation against the installed compilation, specification, oracle, and
    tolerance facts completes before the store is initialized. Repeating the
    same report and observation identity is idempotent; a different producer,
    source commit, or artifact identity creates a coexisting observation.
    """

    if not isinstance(report, VerificationReport):
        raise TypeError("report must be a VerificationReport")
    if not isinstance(store, EvidenceStore):
        raise TypeError("store must implement EvidenceStore")
    _validate_current_report(report)
    return _record_validated_report(
        report,
        store,
        producer_id=producer_id,
        source_commit=source_commit,
        artifact_locator=artifact_locator,
        artifact_digest=artifact_digest,
        recorded_at=recorded_at,
    )
