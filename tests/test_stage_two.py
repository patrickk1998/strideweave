import hashlib
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import strideweave as sw
import strideweave.verification.api as verification_api
import strideweave.verification.classification as classification
import strideweave.verification.stage_one as stage_one_module
import strideweave.verification.stage_two as stage_two_module
from strideweave.verification import (
    Deviations,
    OracleCertificate,
    StageOneResult,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    run_stage_one,
)
from strideweave.verification.comparison import Comparison
from strideweave.verification.stage_two import run_stage_two


def test_backend_runs_both_stages_and_returns_deterministic_evidence(tmp_path):
    output = tmp_path / "verification.jsonl"

    report = sw.test_backend(output)
    repeated = sw.test_backend()
    stage_one = run_stage_one()
    serialized = report.to_jsonl()
    serialized_digest = hashlib.sha256(serialized.encode()).digest()
    loaded = VerificationReport.load(output)

    assert len(report.records) == len(stage_one.report.records) + 9
    assert hashlib.sha256(repeated.to_jsonl().encode()).digest() == serialized_digest
    assert hashlib.sha256(output.read_bytes()).digest() == serialized_digest
    assert hashlib.sha256(loaded.to_jsonl().encode()).digest() == serialized_digest
    stage_two = [
        record for record in report.records if record.stage is VerificationStage.TARGET
    ]
    assert len(stage_two) == 9
    assert all(record.outcome is VerificationOutcome.PASSED for record in stage_two)
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


def test_backend_reports_a_stale_native_extension_before_stage_one(monkeypatch):
    monkeypatch.setattr(classification, "_carrier", SimpleNamespace())

    def fail_if_called():
        raise AssertionError("Stage One ran before the native API preflight")

    monkeypatch.setattr(verification_api, "run_stage_one", fail_if_called)

    with pytest.raises(RuntimeError, match="stale or incompatible") as caught:
        sw.test_backend()

    assert "_cpu_native_kernel_metadata" in str(caught.value)
    assert "uv sync --reinstall-package strideweave --group dev" in str(caught.value)
    assert isinstance(caught.value.__cause__, AttributeError)


def test_backend_preserves_errors_from_a_present_native_binding(monkeypatch):
    class BindingFailure(AttributeError):
        pass

    def fail_inside_binding():
        raise BindingFailure("native binding failed internally")

    monkeypatch.setattr(
        classification,
        "_carrier",
        SimpleNamespace(_cpu_native_kernel_metadata=fail_inside_binding),
    )

    with pytest.raises(BindingFailure, match="failed internally"):
        sw.test_backend()


def test_stage_two_blocks_operations_without_local_oracle_certificates():
    report = run_stage_two(StageOneResult(VerificationReport(()), ()))
    blocked = [
        record
        for record in report.records
        if record.outcome is VerificationOutcome.BLOCKED
    ]

    assert len(blocked) == 4
    assert {record.case.kernel_id for record in blocked} == {
        "cpu.reduce_sum",
        "cpu.matmul",
    }
    assert all(record.diagnostic for record in blocked)
    assert {record.case.operation for record in blocked} == {
        "reduce_sum",
        "matmul",
    }


def test_stage_two_rejects_forged_variant_certificate_without_target_execution(
    monkeypatch,
):
    stage_one = run_stage_one()
    forged = replace(
        next(
            certificate
            for certificate in stage_one.certificates
            if certificate.kernel_id == "cpu.reduce_sum"
        ),
        variant="forged",
    )
    target_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal target_called
        target_called = True
        raise AssertionError("unauthorized target executed")

    monkeypatch.setattr(stage_two_module, "_target_case", fail_if_called)
    report = run_stage_two(replace(stage_one, certificates=(forged,)))

    reduce_records = [
        record for record in report.records if record.case.kernel_id == "cpu.reduce_sum"
    ]
    assert not target_called
    assert all(
        record.outcome is VerificationOutcome.BLOCKED for record in reduce_records
    )


def test_stage_two_rejects_certificate_without_float64_plan_scope(monkeypatch):
    stage_one = run_stage_one()
    forged = OracleCertificate("cpu.reduce_sum", "default", (), "0" * 64)
    target_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal target_called
        target_called = True
        raise AssertionError("unauthorized target executed")

    monkeypatch.setattr(stage_two_module, "_target_case", fail_if_called)
    report = run_stage_two(replace(stage_one, certificates=(forged,)))

    reduce_records = [
        record for record in report.records if record.case.kernel_id == "cpu.reduce_sum"
    ]
    assert not target_called
    assert all(
        record.outcome is VerificationOutcome.BLOCKED for record in reduce_records
    )


def test_stage_two_emits_each_declared_movement_subject():
    report = run_stage_two(run_stage_one())
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


def test_backend_records_stage_two_movement_processing_errors_and_continues(
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

    report = sw.test_backend()
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
    stage_one = run_stage_one()
    original_matmul = stage_two_module.sw.matmul

    def failing_matmul(*args, **kwargs):
        raise ValueError("injected target failure")

    monkeypatch.setattr(stage_two_module.sw, "matmul", failing_matmul)
    report = run_stage_two(stage_one)
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
    report = sw.test_backend()
    numerical = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.test_class.value == "numerical"
    ]

    assert len(numerical) == 2
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

    report = run_stage_two(run_stage_one())
    structural = [
        record
        for record in report.records
        if record.test_class is VerificationClass.STRUCTURAL
    ]
    numerical = [
        record
        for record in report.records
        if record.test_class is VerificationClass.NUMERICAL
    ]

    assert all(record.outcome is VerificationOutcome.FAILED for record in structural)
    assert all(record.outcome is VerificationOutcome.PASSED for record in numerical)


def test_stage_two_uses_multi_output_flat_and_hierarchical_contractions():
    report = run_stage_two(run_stage_one())
    contractions = [
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.kernel_id in {"cpu.reduce_sum", "cpu.matmul"}
    ]

    assert len(contractions) == 4
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
    stage_one = run_stage_one()
    original_matmul = stage_two_module.sw.matmul
    target_result_ids = set()

    def reordered_target(lhs, rhs, *, accumulator_dtype=None):
        result = original_matmul(lhs, rhs, accumulator_dtype=accumulator_dtype)
        if accumulator_dtype is None:
            target_result_ids.add(id(result))
        return result

    original_values = stage_two_module._values

    def reordered_values(tensor):
        values = original_values(tensor)
        return tuple(reversed(values)) if id(tensor) in target_result_ids else values

    monkeypatch.setattr(stage_two_module.sw, "matmul", reordered_target)
    monkeypatch.setattr(stage_two_module, "_values", reordered_values)

    report = run_stage_two(stage_one)
    matmul = [
        record
        for record in report.records
        if record.case.kernel_id == "cpu.matmul"
        and record.stage is VerificationStage.TARGET
    ]

    assert all(record.outcome is VerificationOutcome.FAILED for record in matmul)


def test_installed_package_exposes_test_backend():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import strideweave as sw; assert sw.test_backend().records",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
