import hashlib
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import strideweave as sw
import strideweave.verification.api as verification_api
import strideweave.verification.classification as classification
import strideweave.verification.reporting as reporting
import strideweave.verification.stage_one as stage_one_module
import strideweave.verification.stage_two as stage_two_module
from strideweave.verification import (
    ClassificationDisposition,
    Deviations,
    KernelDescriptor,
    OracleCertificate,
    OracleStageResult,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    classify_profile_plans,
    installed_compilation_bundle,
    run_oracle_stage,
    verification_profile,
)
from strideweave.verification.comparison import Comparison
from strideweave.verification.provenance import make_compilation_bundle
from strideweave.verification.reporting import make_verification_report
from strideweave.verification.stage_two import run_target_stage

_ORACLE_PROFILE = verification_profile("cpu-compiled")


def _oracle_stage():
    return run_oracle_stage(_ORACLE_PROFILE)


def test_verify_backend_runs_both_stages_and_returns_deterministic_evidence(tmp_path):
    output = tmp_path / "verification.jsonl"

    report = sw.verify_backend("cpu-compiled", output=output)
    repeated = sw.verify_backend("cpu-compiled")
    stage_one = _oracle_stage()
    target = run_target_stage(verification_profile("cpu-compiled"), stage_one)
    serialized = report.to_jsonl()
    serialized_digest = hashlib.sha256(serialized.encode()).digest()
    loaded = VerificationReport.load(output)

    assert len(report.records) == len(stage_one.report.records) + len(target.records)
    assert hashlib.sha256(repeated.to_jsonl().encode()).digest() == serialized_digest
    assert hashlib.sha256(output.read_bytes()).digest() == serialized_digest
    assert hashlib.sha256(loaded.to_jsonl().encode()).digest() == serialized_digest
    stage_two = [
        record for record in report.records if record.stage is VerificationStage.TARGET
    ]
    assert len(stage_two) == len(target.records)
    assert all(
        record.outcome in {VerificationOutcome.PASSED, VerificationOutcome.DEFERRED}
        for record in stage_two
    )
    assert all(
        record.target_input_bit_hashes == record.oracle_input_bit_hashes
        for record in report.records
    )
    assert report.header is not None
    certificate_ids = {
        item["certificate_digest"] for item in report.header.certificates
    }
    certified_target_records = [
        record
        for record in stage_two
        if record.case.kernel_id in {"cpu.reduce_sum", "cpu.matmul"}
    ]
    assert certified_target_records
    assert all(
        record.consumed_certificate_digest in certificate_ids
        for record in certified_target_records
    )
    assert all(record.requirement_id != "unbound" for record in report.records)
    assert all(record.tolerance_policy_id != "unbound" for record in report.records)
    assert all(record.oracle_reference_id != "unbound" for record in report.records)


def test_verify_backend_reports_a_stale_native_extension_before_stage_one(monkeypatch):
    monkeypatch.setattr(classification, "import_module", lambda name: SimpleNamespace())

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Stage One ran before the native API preflight")

    monkeypatch.setattr(verification_api, "run_oracle_stage", fail_if_called)

    with pytest.raises(RuntimeError, match="stale or incompatible") as caught:
        sw.verify_backend("cpu-compiled")

    assert "_cpu_native_kernel_metadata" in str(caught.value)
    assert "uv sync --reinstall-package strideweave --group dev" in str(caught.value)
    assert isinstance(caught.value.__cause__, AttributeError)


def test_verify_backend_preserves_errors_from_a_present_native_binding(monkeypatch):
    class BindingFailure(AttributeError):
        pass

    def fail_inside_binding():
        raise BindingFailure("native binding failed internally")

    monkeypatch.setattr(
        classification,
        "import_module",
        lambda name: SimpleNamespace(_cpu_native_kernel_metadata=fail_inside_binding),
    )

    with pytest.raises(BindingFailure, match="failed internally"):
        sw.verify_backend("cpu-compiled")


def test_target_stage_blocks_without_resolving_a_runtime_when_certificates_are_absent(
    monkeypatch,
):
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("target runtime resolved without oracle authorization")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)
    report = run_target_stage(
        verification_profile("cpu-compiled"),
        OracleStageResult(VerificationReport(()), ()),
    )
    blocked = [
        record
        for record in report.records
        if record.outcome is VerificationOutcome.BLOCKED
    ]

    assert blocked
    profile = verification_profile("cpu-compiled")
    active = tuple(
        descriptor
        for descriptor in classify_profile_plans(profile)
        if descriptor.disposition is ClassificationDisposition.ACTIVE
    )
    movement = classification.profile_subjects(profile)
    assert {record.case.kernel_id for record in blocked} == {
        *(descriptor.kernel.kernel_id for descriptor in active),
        *(f"movement.{subject.operation}" for subject in movement),
    }
    assert all(record.diagnostic for record in blocked)
    assert {record.case.operation for record in blocked} == {
        *(descriptor.kernel.operation for descriptor in active),
        *(subject.operation for subject in movement),
    }
    assert not runtime_resolved


@pytest.mark.parametrize(
    ("test_class", "case_suffix"),
    (
        (
            VerificationClass.EXACT_ARITHMETIC,
            "abs-float32-binary32-arbitrary-finite",
        ),
        (VerificationClass.ANALYTIC, "hierarchical-addressing"),
    ),
)
def test_incomplete_rebuilt_certificate_cannot_authorize_target_runtime(
    monkeypatch, test_class, case_suffix
):
    stage_one = _oracle_stage()
    removed = next(
        record
        for record in stage_one.report.records
        if record.test_class is test_class and record.case.case_id.endswith(case_suffix)
    )
    if test_class is VerificationClass.EXACT_ARITHMETIC:
        assert removed.case.case_id == "cpu.abs-float32-binary32-arbitrary-finite"
    records = tuple(record for record in stage_one.report.records if record != removed)
    certificate = next(
        candidate
        for candidate in stage_one.certificates
        if (candidate.kernel_id, candidate.variant)
        == (removed.case.kernel_id, removed.case.variant)
    )
    incomplete_kernel_records = tuple(
        record
        for record in records
        if (record.case.kernel_id, record.case.variant)
        == (certificate.kernel_id, certificate.variant)
    )
    rebuilt = OracleCertificate.from_records(
        KernelDescriptor(
            removed.case.operation,
            removed.case.kernel_id,
            removed.case.variant,
            "",
        ),
        certificate.certified_classes,
        incomplete_kernel_records,
        required_plan_classes=certificate.certified_plan_classes,
    )
    certificates = tuple(
        rebuilt if candidate == certificate else candidate
        for candidate in stage_one.certificates
    )
    forged_report = make_verification_report(records, certificates)
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("incomplete oracle evidence resolved a target runtime")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)

    result = run_target_stage(
        verification_profile("cpu-compiled"),
        OracleStageResult(forged_report, certificates),
    )

    assert not runtime_resolved
    assert result.records
    assert all(
        record.outcome in {VerificationOutcome.BLOCKED, VerificationOutcome.DEFERRED}
        for record in result.records
    )


def test_changed_oracle_profile_role_blocks_before_target_runtime(monkeypatch):
    forged = _oracle_stage()
    assert forged.report.header is not None
    object.__setattr__(forged.report.header, "oracle_profile", "synthetic-jit")
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("forged profile role resolved a target runtime")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)

    result = run_target_stage(verification_profile("cpu-compiled"), forged)

    assert not runtime_resolved
    assert any(
        record.outcome is VerificationOutcome.BLOCKED for record in result.records
    )


@pytest.mark.parametrize("changed_fact", ("operation", "plan", "class"))
def test_changed_required_oracle_fact_blocks_before_target_runtime(
    monkeypatch, changed_fact
):
    forged = _oracle_stage()
    required = next(
        record
        for record in forged.report.records
        if record.case.case_id == "cpu.abs-float32-binary32-arbitrary-finite"
    )
    if changed_fact == "operation":
        changed = replace(required, case=replace(required.case, operation="forged"))
    elif changed_fact == "plan":
        assert required.case.plan is not None
        changed = replace(
            required,
            case=replace(
                required.case,
                plan=replace(required.case.plan, output="Int32"),
            ),
        )
    else:
        changed = replace(required, test_class=VerificationClass.STRUCTURAL)
    object.__setattr__(
        forged.report,
        "records",
        tuple(
            changed if record == required else record
            for record in forged.report.records
        ),
    )
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("changed oracle requirement resolved a target runtime")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)

    result = run_target_stage(verification_profile("cpu-compiled"), forged)

    assert not runtime_resolved
    assert any(
        record.outcome is VerificationOutcome.BLOCKED for record in result.records
    )


def test_synthetic_jit_uses_the_same_neutral_selection_and_cpu_certificate():
    oracle = _oracle_stage()

    result = run_target_stage(verification_profile("synthetic-jit"), oracle)

    contractions = [
        record
        for record in result.records
        if record.case.operation in {"reduce_sum", "matmul"}
    ]
    assert contractions
    assert {record.case.kernel_id.split(".")[0] for record in contractions} == {
        "synthetic-jit"
    }
    assert all(
        record.consumed_certificate_digest is not None for record in contractions
    )
    assert result.receipts
    assert all(receipt.kind == "jit-specialization" for receipt in result.receipts)

    bundle = make_compilation_bundle(
        (
            *installed_compilation_bundle(
                verification_profile("cpu-compiled")
            ).receipts,
            *result.receipts,
        )
    )
    report = make_verification_report(
        (*oracle.report.records, *result.records),
        oracle.certificates,
        selected_target_profile="synthetic-jit",
        oracle_profile="cpu-compiled",
        compilation_bundle=bundle,
    )
    assert report.header is not None
    assert {
        record.case.operation
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.compilation_receipt_id is not None
    } == {
        descriptor.kernel.operation
        for descriptor in classify_profile_plans(verification_profile("synthetic-jit"))
        if descriptor.disposition is ClassificationDisposition.ACTIVE
    }
    assert len(result.receipts) == sum(
        descriptor.disposition is ClassificationDisposition.ACTIVE
        for descriptor in classify_profile_plans(verification_profile("synthetic-jit"))
    )
    multi_plan_kernel = next(
        descriptor.kernel
        for descriptor in classify_profile_plans(verification_profile("synthetic-jit"))
        if sum(
            candidate.kernel == descriptor.kernel
            for candidate in classify_profile_plans(
                verification_profile("synthetic-jit")
            )
        )
        > 1
    )
    receipt_ids = {
        receipt.receipt_id
        for receipt in result.receipts
        if receipt.logical_kernel == multi_plan_kernel
    }
    assert len(receipt_ids) > 1


def test_synthetic_runtime_does_not_invoke_cpu_target_execution(monkeypatch):
    def fail_cpu_target(*args, **kwargs):
        del args, kwargs
        raise AssertionError("synthetic target fell back to the CPU runtime")

    monkeypatch.setattr(stage_two_module, "_cpu_target_records", fail_cpu_target)

    result = run_target_stage(verification_profile("synthetic-jit"), _oracle_stage())

    assert result.records


def test_target_stage_covers_every_classified_plan_and_subject():
    profile = verification_profile("cpu-compiled")
    result = run_target_stage(profile, _oracle_stage())

    for descriptor in classify_profile_plans(profile):
        evidence = [
            record
            for record in result.records
            if record.case.kernel_id == descriptor.kernel.kernel_id
            and record.case.variant == descriptor.kernel.variant
            and record.case.plan == descriptor.plan
        ]
        assert evidence, descriptor
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            assert {record.outcome for record in evidence} == {
                VerificationOutcome.DEFERRED
            }
        else:
            assert set(descriptor.classes).issubset(
                {record.test_class for record in evidence}
            )
    assert {
        record.case.operation
        for record in result.records
        if record.case.kernel_id.startswith("movement.")
    } == {"move", "view", "permute", "rearrange", "broadcast_to"}


def test_executed_noncontraction_target_requires_its_consumed_certificate():
    report = sw.verify_backend("cpu-compiled")
    assert report.header is not None
    target = next(
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.operation == "add"
        and record.outcome is VerificationOutcome.PASSED
    )
    changed = tuple(
        replace(record, consumed_certificate_digest=None)
        if record == target
        else record
        for record in report.records
    )

    with pytest.raises(ValueError, match="consumed Stage One certificate"):
        reporting.bind_report(
            changed,
            (),
            selected_target_profile="cpu-compiled",
            oracle_profile="cpu-compiled",
            compilation_bundle=report.header.compilation_bundle,
            certificate_facts_override=report.header.certificates,
        )


def test_target_synchronizes_before_result_decode(monkeypatch):
    synchronized = False
    synchronization_calls = 0
    original_values = stage_two_module._values

    def synchronize():
        nonlocal synchronized, synchronization_calls
        synchronized = True
        synchronization_calls += 1

    def values(tensor):
        assert synchronized, "target result decoded before synchronization"
        return original_values(tensor)

    monkeypatch.setattr(stage_two_module, "_synchronize_target", synchronize)
    monkeypatch.setattr(stage_two_module, "_values", values)

    result = run_target_stage(verification_profile("cpu-compiled"), _oracle_stage())

    assert result.records
    assert synchronized
    assert synchronization_calls > 20


def test_generic_target_execution_synchronizes_before_stage_one_decode(monkeypatch):
    oracle = _oracle_stage()
    pending_target_result = False
    target_executions = 0
    original_execute = stage_one_module._execute
    original_values = stage_one_module._values

    def execute(descriptor, payloads, layout, cpu, **kwargs):
        nonlocal pending_target_result, target_executions
        if cpu and descriptor.kernel.operation == "add":
            pending_target_result = True
            target_executions += 1
        return original_execute(descriptor, payloads, layout, cpu, **kwargs)

    def synchronize():
        nonlocal pending_target_result
        pending_target_result = False

    def values(tensor):
        assert not pending_target_result, (
            "decoded generic target before synchronization"
        )
        return original_values(tensor)

    monkeypatch.setattr(stage_one_module, "_execute", execute)
    monkeypatch.setattr(stage_one_module, "_values", values)
    monkeypatch.setattr(stage_two_module, "_synchronize_target", synchronize)

    result = run_target_stage(verification_profile("cpu-compiled"), oracle)

    assert result.records
    assert target_executions


def test_verify_backend_rejects_invalid_targets_before_native_work(
    monkeypatch, tmp_path
):
    called = False

    def fail_if_called():
        nonlocal called
        called = True
        raise AssertionError("native work must not start")

    monkeypatch.setattr(
        verification_api, "require_native_verification_api", fail_if_called
    )

    with pytest.raises(TypeError, match="target"):
        sw.verify_backend(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown verification profile"):
        sw.verify_backend("not-a-profile")
    with pytest.raises(TypeError, match="output"):
        sw.verify_backend("cpu-compiled", output=object())  # type: ignore[arg-type]

    assert not called
    assert not (tmp_path / "report.jsonl").exists()


def test_verify_backend_preserves_existing_output_when_atomic_replace_fails(
    monkeypatch, tmp_path
):
    output = tmp_path / "report.jsonl"
    output.write_text("previous report\n", encoding="utf-8")

    def fail_replace(source, destination):
        del source, destination
        raise OSError("injected replace failure")

    monkeypatch.setattr("strideweave.verification.api.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        sw.verify_backend("cpu-compiled", output=output)

    assert output.read_text(encoding="utf-8") == "previous report\n"


def test_verify_backend_requires_an_installed_target_runtime_before_stage_one(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Stage One must not run without a target runtime")

    monkeypatch.setattr(verification_api, "run_oracle_stage", fail_if_called)

    with pytest.raises(RuntimeError, match="no installed provider runtime"):
        sw.verify_backend("synthetic-jit")


def test_target_stage_rejects_a_forged_registered_profile_before_runtime(monkeypatch):
    profile = verification_profile("cpu-compiled")
    forged = replace(profile, provider="forged-provider")

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("runtime must not initialize for a forged profile")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_called)

    with pytest.raises(ValueError, match="registered descriptor"):
        run_target_stage(forged, _oracle_stage())


def test_target_runtime_errors_are_evidence_and_do_not_stop_other_plans(monkeypatch):
    oracle = _oracle_stage()
    original_runtime = stage_two_module._target_runtime

    cpu_runtime = original_runtime(verification_profile("cpu-compiled"))

    def execute(descriptor, certificate_digest, oracle_result, synchronize):
        if descriptor.kernel.operation == "add":
            raise RuntimeError("injected target runtime failure")
        return cpu_runtime.execute(
            descriptor, certificate_digest, oracle_result, synchronize
        )

    runtime = stage_two_module._TargetRuntime(
        execute,
        cpu_runtime.movement,
        cpu_runtime.synchronize,
        cpu_runtime.preflight,
        cpu_runtime.current_compilation,
    )
    monkeypatch.setattr(stage_two_module, "_target_runtime", lambda profile: runtime)

    result = run_target_stage(verification_profile("cpu-compiled"), oracle)

    add_errors = [
        record
        for record in result.records
        if record.case.operation == "add"
        and record.outcome is VerificationOutcome.ERROR
    ]
    assert add_errors
    assert any(
        record.case.operation == "sub" and record.outcome is VerificationOutcome.PASSED
        for record in result.records
    )


def test_stage_two_rejects_forged_variant_certificate_without_target_execution(
    monkeypatch,
):
    stage_one = _oracle_stage()
    forged = replace(
        next(
            certificate
            for certificate in stage_one.certificates
            if certificate.kernel_id == "cpu.reduce_sum"
        ),
        variant="forged",
    )
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("unauthorized target runtime resolved")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)
    report = run_target_stage(
        verification_profile("cpu-compiled"), replace(stage_one, certificates=(forged,))
    )

    reduce_records = [
        record for record in report.records if record.case.kernel_id == "cpu.reduce_sum"
    ]
    assert not runtime_resolved
    assert all(
        record.outcome is VerificationOutcome.BLOCKED for record in reduce_records
    )


def test_stage_two_rejects_certificate_without_float64_plan_scope(monkeypatch):
    stage_one = _oracle_stage()
    forged = OracleCertificate("cpu.reduce_sum", "default", (), "0" * 64)
    runtime_resolved = False

    def fail_if_resolved(profile):
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("unauthorized target runtime resolved")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)
    report = run_target_stage(
        verification_profile("cpu-compiled"), replace(stage_one, certificates=(forged,))
    )

    reduce_records = [
        record for record in report.records if record.case.kernel_id == "cpu.reduce_sum"
    ]
    assert not runtime_resolved
    assert all(
        record.outcome is VerificationOutcome.BLOCKED for record in reduce_records
    )


def test_stage_two_emits_each_declared_movement_subject():
    report = run_target_stage(verification_profile("cpu-compiled"), _oracle_stage())
    movement = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.kernel_id.startswith("movement.")
    ]

    assert {record.case.kernel_id.removeprefix("movement.") for record in movement} == {
        "move",
        "view",
        "permute",
        "rearrange",
        "broadcast_to",
    }
    assert all(record.outcome is VerificationOutcome.PASSED for record in movement)
    assert {record.case.operation for record in movement} == {
        "move",
        "view",
        "permute",
        "rearrange",
        "broadcast_to",
    }
    assert {record.case.operation: record.case.shapes for record in movement} == {
        "move": ((2, 5),),
        "view": ((2, 5),),
        "permute": ((2, 5),),
        "rearrange": ((2, 5),),
        "broadcast_to": ((1, 10),),
    }


def test_verify_backend_records_target_movement_processing_errors_and_continues(
    monkeypatch,
):
    original_comparison = stage_one_module._movement_comparison
    calls = 0

    def fail_first_movement_in_each_stage(expected, actual):
        nonlocal calls
        calls += 1
        if calls in {1, 6}:
            raise ValueError("injected movement comparison failure")
        return original_comparison(expected, actual)

    monkeypatch.setattr(
        stage_one_module, "_movement_comparison", fail_first_movement_in_each_stage
    )

    report = sw.verify_backend("cpu-compiled")
    errors = [
        record
        for record in report.records
        if record.case.kernel_id == "movement.move"
        and record.outcome is VerificationOutcome.ERROR
    ]

    assert {record.stage for record in errors} == {
        VerificationStage.ORACLE,
        VerificationStage.TARGET,
    }
    assert all(record.case.shapes == ((2, 5),) for record in errors)
    assert all(record.mismatches is None for record in errors)
    assert all(record.deviations.maximum_absolute is None for record in errors)
    assert any(
        record.stage is VerificationStage.TARGET
        and record.case.kernel_id == "movement.view"
        and record.outcome is VerificationOutcome.PASSED
        for record in report.records
    )
    assert any(
        record.stage is VerificationStage.TARGET
        and record.case.kernel_id == "cpu.reduce_sum"
        and record.outcome is VerificationOutcome.PASSED
        for record in report.records
    )


def test_stage_two_records_target_errors_and_continues(monkeypatch):
    stage_one = _oracle_stage()
    original_matmul = stage_two_module.sw.matmul

    def failing_matmul(*args, **kwargs):
        raise ValueError("injected target failure")

    monkeypatch.setattr(stage_two_module.sw, "matmul", failing_matmul)
    report = run_target_stage(verification_profile("cpu-compiled"), stage_one)
    monkeypatch.setattr(stage_two_module.sw, "matmul", original_matmul)

    errors = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.kernel_id == "cpu.matmul"
    ]
    assert errors
    assert all(record.outcome is VerificationOutcome.ERROR for record in errors)
    assert all(record.case.shapes for record in errors)
    assert all(record.case.contraction_length is not None for record in errors)
    assert all(record.target_input_bit_hashes for record in errors)
    assert all(record.deviations.maximum_absolute is None for record in errors)
    assert all(record.mismatches is None for record in errors)
    assert any(
        record.case.kernel_id == "cpu.reduce_sum"
        and record.outcome is VerificationOutcome.PASSED
        for record in report.records
    )


def test_stage_two_numerical_tolerance_is_versioned_and_observable():
    report = sw.verify_backend("cpu-compiled")
    numerical = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.test_class.value == "numerical"
    ]

    assert numerical
    assert all(
        record.tolerance.version == "stage-two-float32-gamma-k-v1"
        for record in numerical
    )
    assert all(record.tolerance.absolute > 0.0 for record in numerical)
    assert all(
        record.deviations.maximum_absolute is not None
        and record.deviations.maximum_absolute >= 0.0
        for record in numerical
    )


def test_stage_two_structural_evidence_requires_bit_identity(monkeypatch):
    def zero_deviation_mismatch(expected, actual):
        del expected, actual
        return Comparison(Deviations(0.0, 0.0, 0), 1, 1, 0)

    monkeypatch.setattr(stage_two_module, "compare_float32", zero_deviation_mismatch)

    report = run_target_stage(verification_profile("cpu-compiled"), _oracle_stage())
    structural = [
        record
        for record in report.records
        if record.test_class is VerificationClass.STRUCTURAL
        and record.case.case_id.endswith(("flat-structural", "hierarchical-numerical"))
    ]
    numerical = [
        record
        for record in report.records
        if record.test_class is VerificationClass.NUMERICAL
    ]

    assert all(record.outcome is VerificationOutcome.FAILED for record in structural)
    assert all(record.outcome is VerificationOutcome.PASSED for record in numerical)


def test_stage_two_uses_multi_output_flat_and_hierarchical_contractions():
    report = run_target_stage(verification_profile("cpu-compiled"), _oracle_stage())
    contractions = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.kernel_id in {"cpu.reduce_sum", "cpu.matmul"}
        and record.case.case_id.endswith(("flat-structural", "hierarchical-numerical"))
    ]

    assert contractions
    assert {record.case.contraction_length for record in contractions} == {12, 16}
    assert all(record.case.shapes for record in contractions)
    assert all(record.case.shapes[0][0] > 1 for record in contractions)
    assert {record.case.case_id.split("-")[2] for record in contractions} == {
        "flat",
        "hierarchical",
    }
    assert any(
        record.case.kernel_id == "cpu.matmul"
        and record.case.shapes == ((4, 12), (3, 12))
        for record in contractions
    )


def test_stage_two_detects_matmul_output_ordering_fault(monkeypatch):
    stage_one = _oracle_stage()
    original_matmul = stage_two_module.sw.matmul
    target_results = []

    def reordered_target(lhs, rhs, *, accumulator_dtype=None):
        result = original_matmul(lhs, rhs, accumulator_dtype=accumulator_dtype)
        if accumulator_dtype is None:
            target_results.append(result)
        return result

    original_values = stage_two_module._values

    def reordered_values(tensor):
        values = original_values(tensor)
        return (
            tuple(reversed(values))
            if any(tensor is target for target in target_results)
            else values
        )

    monkeypatch.setattr(stage_two_module.sw, "matmul", reordered_target)
    monkeypatch.setattr(stage_two_module, "_values", reordered_values)

    report = run_target_stage(verification_profile("cpu-compiled"), stage_one)
    matmul = [
        record
        for record in report.records
        if record.case.kernel_id == "cpu.matmul"
        and record.stage is VerificationStage.TARGET
        and record.case.case_id.endswith(("flat-structural", "hierarchical-numerical"))
    ]

    assert all(record.outcome is VerificationOutcome.FAILED for record in matmul), [
        (record.case.case_id, record.case.plan, record.test_class, record.outcome)
        for record in matmul
    ]


def test_installed_package_exposes_verify_backend():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import strideweave as sw; assert sw.verify_backend('cpu-compiled').records",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
