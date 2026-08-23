from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from evidence_doubles import RecordingEvidenceStore

import strideweave.verification.stage_one as stage_one_module
import strideweave.verification.stage_two as stage_two_module
import strideweave.verification.store.recording as recording
from strideweave.verification import (
    CompilationInput,
    Deviations,
    GeneratedArtifact,
    JITSpecializationReceipt,
    SpecializationAxis,
    Tolerance,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    run_oracle_stage,
    run_target_stage,
    verification_profile,
)
from strideweave.verification.provenance import (
    make_compilation_bundle,
    make_compilation_runtime,
    make_compilation_toolchain,
    make_jit_specialization_receipt,
)
from strideweave.verification.reporting import (
    _certificate_from_value,
    bind_report,
    certificate_value,
    make_verification_report,
)
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
)


def _rebound_report(
    report: VerificationReport,
    records,
    *,
    certificate_facts=None,
) -> VerificationReport:
    """Return canonical strict-loaded v3 bytes around an intentionally changed graph."""

    assert report.header is not None
    enriched, header = bind_report(
        tuple(records),
        (),
        selected_target_profile=report.header.selected_target_profile,
        oracle_profile=report.header.oracle_profile,
        compilation_bundle=report.header.compilation_bundle,
        certificate_facts_override=(
            report.header.certificates
            if certificate_facts is None
            else certificate_facts
        ),
    )
    rebound = VerificationReport(enriched, header.schema_version, header)
    return VerificationReport.from_jsonl(rebound.to_jsonl())


def _assert_public_record_rejects_graph(report: VerificationReport) -> None:
    store = RecordingEvidenceStore()
    with pytest.raises(VerificationStoreError, match="installed verification graph"):
        recording.record_report(report, store, producer_id="producer")
    assert store.initialized == 0
    assert store.transactions == []


def test_validated_mixed_report_records_one_complete_graph(synthetic_report) -> None:
    store = RecordingEvidenceStore()
    result = recording._record_validated_report(
        synthetic_report, store, producer_id="producer"
    )
    assert result.evidence_count == len(synthetic_report.records)
    tables = store.tables()
    assert tables["verification_runs"] == 1
    assert tables["compilation_receipts"] == 2
    assert tables["run_compilation_receipts"] == 2
    assert tables["evidence"] == len(synthetic_report.records)
    assert tables["observations"] == len(synthetic_report.records)


def test_public_record_validates_each_receipt_exactly_once(
    monkeypatch: pytest.MonkeyPatch, synthetic_report
) -> None:
    assert synthetic_report.header is not None
    validation_calls = 0
    regenerated_receipts: list[str] = []
    validate = recording._validate_current_report

    def validate_once(report) -> None:
        nonlocal validation_calls
        validation_calls += 1
        validate(report)

    def regenerate(profile, receipts):
        del profile
        regenerated_receipts.extend(receipt.receipt_id for receipt in receipts)
        return make_compilation_bundle(receipts)

    monkeypatch.setattr(recording, "_validate_current_report", validate_once)
    monkeypatch.setattr(recording, "_current_profile_compilation_bundle", regenerate)
    monkeypatch.setattr(
        recording, "_validate_installed_verification_graph", lambda _report: None
    )
    monkeypatch.setattr(
        recording,
        "bind_report",
        lambda *_args, **_kwargs: (synthetic_report.records, synthetic_report.header),
    )
    store = RecordingEvidenceStore()

    recording.record_report(synthetic_report, store, producer_id="producer")

    assert validation_calls == 1
    assert regenerated_receipts == [
        receipt.receipt_id
        for receipt in synthetic_report.header.compilation_bundle.receipts
    ]
    assert len(store.transactions) == 1


def test_caller_naive_recording_time_is_rejected(synthetic_report) -> None:
    from datetime import datetime

    with pytest.raises(VerificationStoreError, match="timezone"):
        recording._record_validated_report(
            synthetic_report,
            RecordingEvidenceStore(),
            producer_id="producer",
            recorded_at=datetime(2026, 1, 1),
        )


def test_public_record_rejects_stale_cpu_receipt_before_store(
    monkeypatch: pytest.MonkeyPatch, synthetic_report
) -> None:
    store = RecordingEvidenceStore()
    monkeypatch.setattr(
        recording, "_current_profile_compilation_bundle", lambda *_args: object()
    )
    with pytest.raises(VerificationStoreError, match="cpu-compiled"):
        recording.record_report(synthetic_report, store, producer_id="producer")
    assert store.transactions == []


def test_public_record_rejects_unavailable_target_provenance_before_store_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_report
) -> None:
    resolve = recording._current_profile_compilation_bundle

    def allow_current_cpu(profile, receipts):
        if profile.profile_id == "cpu-compiled":
            return make_compilation_bundle(receipts)
        return resolve(profile, receipts)

    monkeypatch.setattr(
        recording, "_current_profile_compilation_bundle", allow_current_cpu
    )
    path = tmp_path / "unavailable-target-store"
    store = DoltEvidenceStore(path)

    with pytest.raises(
        VerificationStoreError, match="no installed provider provenance"
    ):
        recording.record_report(synthetic_report, store, producer_id="producer")

    assert not path.exists()


def test_public_record_accepts_a_neutrally_rebound_installed_jit_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = run_oracle_stage(verification_profile("cpu-compiled"))
    target = run_target_stage(verification_profile("synthetic-jit"), oracle)
    assert oracle.report.header is not None
    report = make_verification_report(
        (*oracle.report.records, *target.records),
        oracle.certificates,
        selected_target_profile="synthetic-jit",
        oracle_profile="cpu-compiled",
        compilation_bundle=make_compilation_bundle(
            (*oracle.report.header.compilation_bundle.receipts, *target.receipts)
        ),
    )
    monkeypatch.setattr(
        recording,
        "_current_profile_compilation_bundle",
        lambda profile, receipts: make_compilation_bundle(receipts),
    )
    store = RecordingEvidenceStore()

    result = recording.record_report(report, store, producer_id="producer")

    assert result.evidence_count == len(report.records)
    assert len(store.transactions) == 1


def test_public_record_accepts_the_current_compiled_target(backend_report) -> None:
    store = RecordingEvidenceStore()

    result = recording.record_report(backend_report, store, producer_id="producer")

    assert result.evidence_count == len(backend_report.records)
    assert len(store.transactions) == 1


def test_current_graph_reconciliation_does_not_execute_verification(
    monkeypatch: pytest.MonkeyPatch, backend_report
) -> None:
    def fail_if_executed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("recording admission executed verification")

    monkeypatch.setattr(stage_one_module, "_execute", fail_if_executed)
    monkeypatch.setattr(stage_one_module, "_movement_records", fail_if_executed)
    monkeypatch.setattr(stage_two_module, "_target_case", fail_if_executed)
    store = RecordingEvidenceStore()

    recording.record_report(backend_report, store, producer_id="producer")

    assert len(store.transactions) == 1


def test_public_record_rejects_a_canonical_report_missing_one_target_witness(
    backend_report,
) -> None:
    removed = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.TARGET
        and record.outcome is VerificationOutcome.PASSED
    )
    report = _rebound_report(
        backend_report,
        tuple(record for record in backend_report.records if record != removed),
    )

    assert len(report.records) == len(backend_report.records) - 1
    _assert_public_record_rejects_graph(report)


def test_public_record_rejects_missing_and_extra_installed_requirements(
    backend_report,
) -> None:
    deferred_oracle = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.ORACLE
        and record.outcome is VerificationOutcome.DEFERRED
    )
    missing = _rebound_report(
        backend_report,
        tuple(record for record in backend_report.records if record != deferred_oracle),
    )
    _assert_public_record_rejects_graph(missing)

    source = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.TARGET
        and record.outcome is VerificationOutcome.PASSED
    )
    extra = replace(
        source,
        case=replace(source.case, case_id=f"{source.case.case_id}-unregistered"),
    )
    _assert_public_record_rejects_graph(
        _rebound_report(backend_report, (*backend_report.records, extra))
    )


@pytest.mark.parametrize("axis", ("case", "tolerance"))
def test_public_record_rejects_changed_installed_requirement_facts(
    backend_report, axis
) -> None:
    source = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.TARGET
        and record.outcome is VerificationOutcome.PASSED
    )
    if axis == "case":
        changed = replace(
            source,
            case=replace(source.case, seed=(source.case.seed or 0) + 1),
        )
    else:
        changed = replace(
            source,
            tolerance=Tolerance(
                absolute=source.tolerance.absolute + 1.0,
                relative=source.tolerance.relative,
                ulps=source.tolerance.ulps,
                version=source.tolerance.version,
            ),
        )
    report = _rebound_report(
        backend_report,
        tuple(
            changed if record == source else record for record in backend_report.records
        ),
    )
    _assert_public_record_rejects_graph(report)


def test_public_record_rejects_missing_current_certificate_facts(
    backend_report,
) -> None:
    assert backend_report.header is not None
    certificate = next(
        value
        for value in backend_report.header.certificates
        if value["kernel_id"] == "cpu.add"
    )
    digest = certificate["certificate_digest"]

    def block_dependency(record):
        if (
            record.stage is VerificationStage.TARGET
            and record.consumed_certificate_digest == digest
        ):
            return replace(
                record,
                outcome=VerificationOutcome.BLOCKED,
                deviations=Deviations(0.0, 0.0, 0),
                mismatches=0,
                diagnostic="forged missing oracle authorization",
                compilation_receipt_id=None,
                consumed_certificate_digest=None,
            )
        return record

    report = _rebound_report(
        backend_report,
        tuple(block_dependency(record) for record in backend_report.records),
        certificate_facts=tuple(
            value
            for value in backend_report.header.certificates
            if value["certificate_digest"] != digest
        ),
    )
    store = RecordingEvidenceStore()
    with pytest.raises(VerificationStoreError, match="oracle certificates"):
        recording.record_report(report, store, producer_id="producer")
    assert store.transactions == []


def test_public_record_rejects_changed_current_certificate_scope(
    backend_report,
) -> None:
    assert backend_report.header is not None
    certificate_fact = next(
        value
        for value in backend_report.header.certificates
        if value["kernel_id"] == "cpu.add"
    )
    certificate = _certificate_from_value(
        recording._thaw(certificate_fact), "changed certificate"
    )
    plan, classes = certificate.certified_plan_classes[0]
    changed = replace(
        certificate,
        evidence_digest="0" * 64,
        certified_plan_classes=(
            *certificate.certified_plan_classes,
            (replace(plan, output="forged-output"), classes),
        ),
    )
    changed_fact = certificate_value(changed)
    changed_digest = changed_fact["certificate_digest"]
    records = tuple(
        replace(record, consumed_certificate_digest=changed_digest)
        if record.consumed_certificate_digest == certificate_fact["certificate_digest"]
        else record
        for record in backend_report.records
    )
    report = _rebound_report(
        backend_report,
        records,
        certificate_facts=tuple(
            changed_fact
            if value["certificate_digest"] == certificate_fact["certificate_digest"]
            else value
            for value in backend_report.header.certificates
        ),
    )

    store = RecordingEvidenceStore()
    with pytest.raises(VerificationStoreError, match="oracle certificates"):
        recording.record_report(report, store, producer_id="producer")
    assert store.transactions == []


def test_public_record_rejects_a_blocked_edge_with_a_current_certificate(
    backend_report,
) -> None:
    source = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.TARGET
        and record.consumed_certificate_digest is not None
        and record.outcome is VerificationOutcome.PASSED
    )
    blocked = replace(
        source,
        outcome=VerificationOutcome.BLOCKED,
        deviations=Deviations(0.0, 0.0, 0),
        mismatches=0,
        diagnostic="forged blocked dependency",
        compilation_receipt_id=None,
        consumed_certificate_digest=None,
    )
    report = _rebound_report(
        backend_report,
        tuple(
            blocked if record == source else record for record in backend_report.records
        ),
    )
    store = RecordingEvidenceStore()
    with pytest.raises(VerificationStoreError, match="blocked despite"):
        recording.record_report(report, store, producer_id="producer")
    assert store.transactions == []


def test_public_record_accepts_current_failed_oracle_facts_and_blocked_targets(
    backend_report,
) -> None:
    assert backend_report.header is not None
    certificate = next(
        value
        for value in backend_report.header.certificates
        if value["kernel_id"] == "cpu.add"
    )
    digest = certificate["certificate_digest"]
    failed_oracle = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.ORACLE
        and record.case.kernel_id == "cpu.add"
        and record.outcome is VerificationOutcome.PASSED
    )

    def changed_outcome(record):
        if record == failed_oracle:
            return replace(
                record,
                outcome=VerificationOutcome.FAILED,
                deviations=Deviations(math.inf, math.inf, 1),
                mismatches=1,
                diagnostic="injected current oracle disagreement",
            )
        if (
            record.stage is VerificationStage.TARGET
            and record.consumed_certificate_digest == digest
        ):
            return replace(
                record,
                outcome=VerificationOutcome.BLOCKED,
                deviations=Deviations(0.0, 0.0, 0),
                mismatches=0,
                diagnostic="current oracle authorization did not pass",
                compilation_receipt_id=None,
                consumed_certificate_digest=None,
            )
        return record

    report = _rebound_report(
        backend_report,
        tuple(changed_outcome(record) for record in backend_report.records),
        certificate_facts=tuple(
            value
            for value in backend_report.header.certificates
            if value["certificate_digest"] != digest
        ),
    )
    store = RecordingEvidenceStore()

    result = recording.record_report(report, store, producer_id="producer")

    assert result.evidence_count == len(report.records)
    assert len(store.transactions) == 1


def test_record_cli_rejects_incomplete_current_graph_before_store_creation(
    tmp_path: Path, backend_report
) -> None:
    removed = next(
        record
        for record in backend_report.records
        if record.stage is VerificationStage.TARGET
        and record.outcome is VerificationOutcome.PASSED
    )
    report = _rebound_report(
        backend_report,
        tuple(record for record in backend_report.records if record != removed),
    )
    report_path = tmp_path / "incomplete-v3.jsonl"
    report.write(report_path)
    store_path = tmp_path / "absent-store"

    assert (
        status_main(
            [
                "record",
                "--report",
                str(report_path),
                "--producer",
                "producer",
                "--store",
                str(store_path),
            ]
        )
        == 2
    )
    assert not store_path.exists()


def _stale_jit_receipt(
    receipt: JITSpecializationReceipt, axis: str
) -> JITSpecializationReceipt:
    provider = receipt.provider
    toolchain = receipt.toolchain
    runtime = receipt.runtime
    inputs = receipt.inputs
    options = receipt.compile_options
    artifacts = receipt.artifacts
    specialization_axes = receipt.specialization_axes
    if axis == "provider":
        provider = "stale-provider"
        toolchain = make_compilation_toolchain(
            provider=provider,
            compiler_id=toolchain.compiler_id,
            compiler_version=toolchain.compiler_version,
            target_triple=toolchain.target_triple,
            build_system=toolchain.build_system,
        )
    elif axis == "runtime":
        runtime = make_compilation_runtime({"provider_runtime": "stale"})
    elif axis == "toolchain":
        toolchain = make_compilation_toolchain(
            provider=provider,
            compiler_id=toolchain.compiler_id,
            compiler_version="stale",
            target_triple=toolchain.target_triple,
            build_system=toolchain.build_system,
        )
    elif axis == "generated-source":
        source = artifacts[0]
        artifacts = (
            GeneratedArtifact(source.ordinal, source.artifact_kind, "f" * 64),
            *artifacts[1:],
        )
    elif axis == "specialization":
        specialization_axes = (SpecializationAxis("block_size", 1024),)
    elif axis == "closure":
        source = inputs[0]
        inputs = (
            CompilationInput(source.ordinal, source.uri, source.input_kind, "e" * 64),
        )
    elif axis == "artifact":
        executable = artifacts[-1]
        artifacts = (
            *artifacts[:-1],
            GeneratedArtifact(
                executable.ordinal,
                executable.artifact_kind,
                "d" * 64,
                "d" * 64,
            ),
        )
    else:  # pragma: no cover - parameter list owns this helper.
        raise AssertionError(f"unknown stale JIT axis {axis!r}")
    return make_jit_specialization_receipt(
        profile_id=receipt.profile_id,
        provider=provider,
        target=receipt.target,
        toolchain=toolchain,
        runtime=runtime,
        logical_kernel=receipt.logical_kernel,
        declared_input_uris=receipt.declared_input_uris,
        inputs=inputs,
        compile_options=options,
        artifacts=artifacts,
        specialization_axes=specialization_axes,
        generated_host_source_artifact_ordinal=(
            receipt.generated_host_source_artifact_ordinal
        ),
        generated_device_source_artifact_ordinal=(
            receipt.generated_device_source_artifact_ordinal
        ),
        runtime_artifact_ordinals=receipt.runtime_artifact_ordinals,
    )


@pytest.mark.parametrize(
    "axis",
    (
        "provider",
        "runtime",
        "toolchain",
        "generated-source",
        "specialization",
        "closure",
        "artifact",
    ),
)
def test_public_record_rejects_stale_selected_jit_provenance_before_store(
    monkeypatch: pytest.MonkeyPatch, synthetic_report, axis
) -> None:
    assert synthetic_report.header is not None
    target_receipt = next(
        receipt
        for receipt in synthetic_report.header.compilation_bundle.receipts
        if receipt.profile_id == "synthetic-jit"
    )
    assert isinstance(target_receipt, JITSpecializationReceipt)
    stale = _stale_jit_receipt(target_receipt, axis)

    def resolve(profile, receipts):
        if profile.profile_id == "synthetic-jit":
            return make_compilation_bundle((stale,))
        return make_compilation_bundle(receipts)

    monkeypatch.setattr(recording, "_current_profile_compilation_bundle", resolve)
    store = RecordingEvidenceStore()

    with pytest.raises(VerificationStoreError, match=r"synthetic-jit.*stale"):
        recording.record_report(synthetic_report, store, producer_id="producer")

    assert store.transactions == []


def test_strict_report_loading_does_not_resolve_installed_providers(
    monkeypatch: pytest.MonkeyPatch, synthetic_report
) -> None:
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("strict report loading consulted an installed provider")

    monkeypatch.setattr(
        recording, "_current_profile_compilation_bundle", fail_if_called
    )

    loaded = VerificationReport.from_jsonl(synthetic_report.to_jsonl())

    assert loaded == synthetic_report


def test_public_record_rejects_semantically_inconsistent_evidence_before_store(
    backend_report,
) -> None:
    report = VerificationReport.from_jsonl(backend_report.to_jsonl())
    assert report.header is not None
    record = next(
        item
        for item in report.records
        if item.stage.value == "stage_two" and item.outcome.value == "passed"
    )
    object.__setattr__(record, "mismatches", 1)
    object.__setattr__(
        record,
        "deviations",
        Deviations(math.inf, math.inf, 0xFFFF_FFFF),
    )
    store = RecordingEvidenceStore()

    with pytest.raises(
        VerificationStoreError, match="passed exact_arithmetic evidence"
    ):
        recording.record_report(report, store, producer_id="producer")

    assert store.transactions == []


def test_cli_rejects_non_v3_report_before_store_path_exists(tmp_path: Path) -> None:
    report = tmp_path / "v2.jsonl"
    report.write_text(
        '{"schema_version":"strideweave.kernel-verification.v2"}\n',
        encoding="utf-8",
    )
    store = tmp_path / "fresh-store"
    assert (
        status_main(
            [
                "record",
                "--report",
                str(report),
                "--producer",
                "p",
                "--store",
                str(store),
            ]
        )
        == 2
    )
    assert not store.exists()


def _count(store: DoltEvidenceStore, table: str) -> int:
    value = store.query(SQLStatement(f"SELECT COUNT(*) AS count FROM {table}"))[0][
        "count"
    ]
    assert type(value) is int
    return value


def test_fresh_v3_store_is_idempotent_and_keeps_producers(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    first = recording._record_validated_report(synthetic_report, store, producer_id="a")
    assert (
        recording._record_validated_report(synthetic_report, store, producer_id="a")
        == first
    )
    recording._record_validated_report(synthetic_report, store, producer_id="b")
    assert _count(store, "verification_runs") == 1
    assert _count(store, "observations") == 2 * len(synthetic_report.records)


def test_v2_store_is_rejected_before_new_migration_or_write(
    evidence_store_path, synthetic_report
) -> None:
    path = evidence_store_path()
    initial = DoltEvidenceStore(path)
    recording._record_validated_report(synthetic_report, initial, producer_id="p")
    initial.execute_transaction(
        (
            SQLStatement(
                "UPDATE schema_migrations SET migration_name='0001_raw_evidence.sql'"
            ),
        )
    )
    rejected = DoltEvidenceStore(path)
    with pytest.raises(
        VerificationStoreError, match="new --store path or manually recreate"
    ):
        recording._record_validated_report(synthetic_report, rejected, producer_id="q")
    assert _count(initial, "observations") == len(synthetic_report.records)


def test_unknown_existing_store_is_rejected_without_bootstrapping_migrations(
    evidence_store_path, synthetic_report
) -> None:
    path = evidence_store_path()
    initial = DoltEvidenceStore(path)
    recording._record_validated_report(synthetic_report, initial, producer_id="p")
    initial.execute_transaction((SQLStatement("DROP TABLE schema_migrations"),))
    rejected = DoltEvidenceStore(path)
    with pytest.raises(VerificationStoreError, match="unknown or corrupt"):
        recording._record_validated_report(synthetic_report, rejected, producer_id="q")
    assert not initial.query(SQLStatement("SHOW TABLES LIKE 'schema_migrations'"))


def test_failed_transaction_rolls_back_all_prior_statements(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record = recording._record_validated_report
    _record(synthetic_report, store, producer_id="p")
    before = _count(store, "verification_runs")
    with pytest.raises(VerificationStoreError):
        store.execute_transaction(
            (
                SQLStatement(
                    "INSERT INTO verification_runs (run_id, report_digest, report_schema, selected_target_profile, oracle_profile, bundle_id, todo_provenance_digest, header_digest, report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "f" * 64,
                        "e" * 64,
                        "strideweave.kernel-verification.v3",
                        "synthetic-jit",
                        "cpu-compiled",
                        "d" * 64,
                        "b" * 64,
                        "c" * 64,
                        "{}",
                    ),
                ),
                SQLStatement("SELECT this_is_invalid_sql"),
            )
        )
    assert _count(store, "verification_runs") == before
