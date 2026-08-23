"""Pure-Python, schema-v3 mixed-receipt verification reports for store tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from unittest.mock import patch

import strideweave.verification.reporting as reporting_module
from strideweave.verification.classification import LogicalKernel
from strideweave.verification.model import (
    CaseDescriptor,
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
from strideweave.verification.provenance import (
    CompilationInput,
    GeneratedArtifact,
    SpecializationAxis,
    make_compilation_bundle,
    make_compilation_runtime,
    make_compilation_target,
    make_compilation_toolchain,
    make_compiled_executable_receipt,
    make_jit_specialization_receipt,
)
from strideweave.verification.reporting import (
    certificate_value,
    make_verification_report,
)

TARGET = "synthetic-jit"
"""Selected target profile represented by these self-contained reports."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _common(profile: str, provider: str):
    target = make_compilation_target(
        architecture="synthetic-arch64",
        vendor="synthetic",
        operating_system="test",
        abi="synthetic",
        endianness="little",
        pointer_bits=64,
    )
    toolchain = make_compilation_toolchain(
        provider=provider,
        compiler_id="SyntheticCompiler",
        compiler_version="1",
        target_triple="synthetic-arch64-test",
        build_system="synthetic-build",
    )
    runtime = make_compilation_runtime({"provider_runtime": "1"})
    inputs = (
        CompilationInput(
            0, f"synthetic://{profile}/source", "source", _digest(profile + "source")
        ),
    )
    return target, toolchain, runtime, inputs


def _alternate_oracle_value():
    inputs = [
        {
            "content_digest": _digest("alternate-generic-oracle-source"),
            "uri": "src/strideweave/synthetic_alternate_oracle.py",
        }
    ]
    value = {
        "implementation_digest": reporting_module._digest(inputs),
        "inputs": inputs,
        "oracle_kind": "generic-reference",
        "oracle_schema": "strideweave.kernel-oracle-reference.v1",
    }
    return {"oracle_reference_id": reporting_module._digest(value), **value}


def _make_synthetic_report(
    *,
    specialization: int = 128,
    specification_variant: bool = False,
    tolerance_variant: bool = False,
    oracle_variant: bool = False,
):
    """Return a complete target report with CPU and JIT receipt dependencies."""
    cpu_kernel = LogicalKernel("cpu-compiled", "add", "cpu.add", "default")
    cpu_target, cpu_toolchain, cpu_runtime, cpu_inputs = _common("cpu-compiled", "cpu")
    cpu_artifacts = (
        GeneratedArtifact(0, "compiled-object", _digest("cpu-object")),
        GeneratedArtifact(
            1, "shared-executable", _digest("cpu-executable"), _digest("cpu-executable")
        ),
    )
    cpu_receipt = make_compiled_executable_receipt(
        profile_id="cpu-compiled",
        provider="cpu",
        target=cpu_target,
        toolchain=cpu_toolchain,
        runtime=cpu_runtime,
        logical_kernel=cpu_kernel,
        declared_input_uris=(cpu_inputs[0].uri,),
        inputs=cpu_inputs,
        compile_options=("-O2",),
        artifacts=cpu_artifacts,
        compiled_object_artifact_ordinals=(0,),
        shared_executable_artifact_ordinal=1,
    )
    jit_kernel = LogicalKernel(TARGET, "add", "jit.add", "default")
    jit_target, jit_toolchain, jit_runtime, jit_inputs = _common(
        TARGET, "synthetic-jit"
    )
    jit_artifacts = (
        GeneratedArtifact(0, "generated-host-source", _digest(f"host{specialization}")),
        GeneratedArtifact(
            1, "generated-device-source", _digest(f"device{specialization}")
        ),
        GeneratedArtifact(
            2,
            "runtime-binary",
            _digest(f"binary{specialization}"),
            _digest(f"binary{specialization}"),
        ),
    )
    jit_receipt = make_jit_specialization_receipt(
        profile_id=TARGET,
        provider="synthetic-jit",
        target=jit_target,
        toolchain=jit_toolchain,
        runtime=jit_runtime,
        logical_kernel=jit_kernel,
        declared_input_uris=(jit_inputs[0].uri,),
        inputs=jit_inputs,
        compile_options=("--jit",),
        artifacts=jit_artifacts,
        specialization_axes=(SpecializationAxis("block_size", specialization),),
        generated_host_source_artifact_ordinal=0,
        generated_device_source_artifact_ordinal=1,
        runtime_artifact_ordinals=(2,),
    )
    hashes = (_digest("input"),)
    plan = PlanKey(
        operation="add",
        operands=(("LEFT", "Float32", None), ("RIGHT", "Float32", None)),
        compute="Float32",
        accumulation=None,
        accumulator_dtype=None,
        output="Float32",
    )
    case_suffix = "/alternate-specification" if specification_variant else ""
    tolerance = (
        Tolerance(absolute=0.25, relative=0.125, ulps=2, version="synthetic-v2")
        if tolerance_variant
        else Tolerance()
    )
    records = (
        EvidenceRecord(
            VerificationStage.ORACLE,
            VerificationClass.EXACT_ARITHMETIC,
            CaseDescriptor(
                "add",
                "cpu.add",
                "default",
                ("Float32", "Float32"),
                "Float32",
                ((2,), (2,)),
                None,
                None,
                1,
                f"cpu.add/oracle{case_suffix}",
                plan,
            ),
            hashes,
            hashes,
            tolerance,
            Deviations(0.0, 0.0, 0),
            0,
            VerificationOutcome.PASSED,
        ),
        EvidenceRecord(
            VerificationStage.TARGET,
            VerificationClass.EXACT_ARITHMETIC,
            CaseDescriptor(
                "add",
                "jit.add",
                "default",
                ("Float32", "Float32"),
                "Float32",
                ((2,), (2,)),
                None,
                None,
                1,
                f"jit.add/target{case_suffix}",
                plan,
            ),
            hashes,
            hashes,
            tolerance,
            Deviations(0.0, 0.0, 0),
            0,
            VerificationOutcome.PASSED,
        ),
    )
    certificate = OracleCertificate.from_records(
        KernelDescriptor("add", "cpu.add", "default", "synthetic"),
        (VerificationClass.EXACT_ARITHMETIC,),
        (records[0],),
        required_plan_classes=((plan, (VerificationClass.EXACT_ARITHMETIC,)),),
    )
    records = (
        records[0],
        replace(
            records[1],
            compilation_receipt_id=jit_receipt.receipt_id,
            consumed_certificate_digest=certificate_value(certificate)[
                "certificate_digest"
            ],
        ),
    )

    def build_report():
        return make_verification_report(
            records,
            (certificate,),
            selected_target_profile=TARGET,
            oracle_profile="cpu-compiled",
            compilation_bundle=make_compilation_bundle((cpu_receipt, jit_receipt)),
        )

    if oracle_variant:
        with patch.object(
            reporting_module,
            "_generic_oracle_value",
            return_value=_alternate_oracle_value(),
        ):
            report = build_report()
    else:
        report = build_report()
    return report, certificate


_REPORT_AND_CERTIFICATES = {
    128: _make_synthetic_report(),
    256: _make_synthetic_report(specialization=256),
}
_REPORTS = {
    specialization: report_and_certificate[0]
    for specialization, report_and_certificate in _REPORT_AND_CERTIFICATES.items()
}
_CERTIFICATES = {
    specialization: report_and_certificate[1]
    for specialization, report_and_certificate in _REPORT_AND_CERTIFICATES.items()
}
_PROVENANCE_VARIANTS = {
    "verification": _make_synthetic_report(specification_variant=True)[0],
    "tolerance": _make_synthetic_report(tolerance_variant=True)[0],
    "oracle": _make_synthetic_report(oracle_variant=True)[0],
}


def _make_cpu_report(report, certificate):
    assert report.header is not None
    cpu_bundle = make_compilation_bundle(
        tuple(
            receipt
            for receipt in report.header.compilation_bundle.receipts
            if receipt.profile_id == "cpu-compiled"
        )
    )
    return make_verification_report(
        tuple(
            record
            for record in report.records
            if record.stage is VerificationStage.ORACLE
        ),
        (certificate,),
        selected_target_profile="cpu-compiled",
        oracle_profile="cpu-compiled",
        compilation_bundle=cpu_bundle,
    )


_CPU_REPORTS = {
    specialization: _make_cpu_report(report, _CERTIFICATES[specialization])
    for specialization, report in _REPORTS.items()
}


def _make_contradicting_report(report, certificate):
    """Build the pure conflicting variant before marked-test guards run."""

    records = tuple(
        replace(record, outcome=VerificationOutcome.FAILED, mismatches=1)
        if record.stage is VerificationStage.TARGET
        else record
        for record in report.records
    )
    assert report.header is not None
    return make_verification_report(
        records,
        (certificate,),
        selected_target_profile=report.header.selected_target_profile,
        oracle_profile=report.header.oracle_profile,
        compilation_bundle=report.header.compilation_bundle,
    )


_CONTRADICTING_REPORTS = {
    specialization: _make_contradicting_report(report, _CERTIFICATES[specialization])
    for specialization, report in _REPORTS.items()
}


def _make_target_outcome_report(report, certificate, outcome, *, alternate=False):
    """Build one pure report whose target requirement has a factual outcome."""

    def target_outcome(record):
        if record.stage is not VerificationStage.TARGET:
            return record
        if outcome is VerificationOutcome.FAILED:
            return replace(record, outcome=outcome, mismatches=2 if alternate else 1)
        if outcome is VerificationOutcome.ERROR:
            return replace(
                record,
                outcome=outcome,
                deviations=Deviations(None, None, None),
                mismatches=None,
                diagnostic=(
                    "RuntimeError: alternate synthetic target failure"
                    if alternate
                    else "RuntimeError: synthetic target failure"
                ),
            )
        if outcome is VerificationOutcome.BLOCKED:
            return replace(
                record,
                outcome=outcome,
                mismatches=0,
                diagnostic=(
                    "alternate synthetic oracle authorization unavailable"
                    if alternate
                    else "synthetic oracle authorization unavailable"
                ),
                compilation_receipt_id=None,
                consumed_certificate_digest=None,
            )
        if outcome is VerificationOutcome.DEFERRED:
            return replace(
                record,
                test_class=VerificationClass.DEFERRED,
                outcome=outcome,
                mismatches=0,
                diagnostic=(
                    "alternate synthetic target verification deferred"
                    if alternate
                    else "synthetic target verification deferred"
                ),
                compilation_receipt_id=None,
                consumed_certificate_digest=None,
            )
        raise ValueError(f"unsupported synthetic target outcome {outcome!r}")

    assert report.header is not None
    return make_verification_report(
        tuple(target_outcome(record) for record in report.records),
        (certificate,),
        selected_target_profile=report.header.selected_target_profile,
        oracle_profile=report.header.oracle_profile,
        compilation_bundle=report.header.compilation_bundle,
    )


_TARGET_OUTCOME_REPORTS = {
    (specialization, outcome, alternate): _make_target_outcome_report(
        report, _CERTIFICATES[specialization], outcome, alternate=alternate
    )
    for specialization, report in _REPORTS.items()
    for outcome in (
        VerificationOutcome.FAILED,
        VerificationOutcome.ERROR,
        VerificationOutcome.BLOCKED,
        VerificationOutcome.DEFERRED,
    )
    for alternate in (False, True)
}


def synthetic_report(*, specialization: int = 128):
    """Return a cached pure report, or an alternate JIT specialization."""

    if specialization in _REPORTS:
        return _REPORTS[specialization]
    return _make_synthetic_report(specialization=specialization)[0]


def contradicting_report(report):
    """Return a separately content-addressed report with a failed JIT fact."""

    for specialization, cached in _REPORTS.items():
        if report is cached:
            return _CONTRADICTING_REPORTS[specialization]
    raise ValueError(
        "contradicting_report requires one of the cached synthetic reports"
    )


def target_outcome_report(report, outcome, *, alternate=False):
    """Return a cached report with the requested factual target outcome."""

    for specialization, cached in _REPORTS.items():
        if report is cached:
            try:
                return _TARGET_OUTCOME_REPORTS[specialization, outcome, alternate]
            except KeyError as error:
                raise ValueError(
                    f"unsupported synthetic target outcome {outcome!r}"
                ) from error
    raise ValueError("target_outcome_report requires one of the cached reports")


def provenance_variant(axis):
    """Return a cached report differing along one todo provenance axis."""

    try:
        return _PROVENANCE_VARIANTS[axis]
    except KeyError as error:
        raise ValueError(f"unsupported synthetic provenance axis {axis!r}") from error


def cpu_report(*, specialization=128):
    """Return a cached pure CPU-profile report for wrong-target query tests."""

    try:
        return _CPU_REPORTS[specialization]
    except KeyError as error:
        raise ValueError(
            f"unsupported synthetic CPU specialization {specialization!r}"
        ) from error
