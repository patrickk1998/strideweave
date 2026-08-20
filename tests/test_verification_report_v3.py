from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from strideweave.verification import (
    CaseDescriptor,
    Deviations,
    EvidenceRecord,
    KernelDescriptor,
    LogicalKernel,
    OracleCertificate,
    PlanKey,
    Tolerance,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
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

_SOURCE_DIGEST = "1" * 64
_OBJECT_DIGEST = "2" * 64
_EXECUTABLE_DIGEST = "3" * 64
_HOST_DIGEST = "4" * 64
_DEVICE_DIGEST = "5" * 64


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target():
    return make_compilation_target(
        architecture="test-arch",
        vendor="test-vendor",
        operating_system="test-os",
        abi="test-abi",
        endianness="little",
        pointer_bits=64,
    )


def _runtime():
    return make_compilation_runtime({"runtime": "test", "version": "1"})


def _toolchain(provider: str):
    return make_compilation_toolchain(
        provider=provider,
        compiler_id="test-compiler",
        compiler_version="1.0",
        target_triple="test-arch-test-os",
        build_system="test-build",
    )


def _compiled_receipt(*, profile_id: str = "cpu-compiled"):
    provider = "test-compiled-provider"
    return make_compiled_executable_receipt(
        profile_id=profile_id,
        provider=provider,
        target=_target(),
        toolchain=_toolchain(provider),
        runtime=_runtime(),
        logical_kernel=LogicalKernel(profile_id, "add", "cpu.add", "default"),
        declared_input_uris=("provider://source/add.cpp",),
        inputs=(
            CompilationInput(0, "provider://source/add.cpp", "source", _SOURCE_DIGEST),
        ),
        compile_options=("-O2",),
        artifacts=(
            GeneratedArtifact(0, "compiled-object", _OBJECT_DIGEST),
            GeneratedArtifact(
                1,
                "shared-executable",
                _EXECUTABLE_DIGEST,
                _EXECUTABLE_DIGEST,
            ),
        ),
        compiled_object_artifact_ordinals=(0,),
        shared_executable_artifact_ordinal=1,
    )


def _jit_receipt(block_size: int):
    provider = "test-jit-provider"
    return make_jit_specialization_receipt(
        profile_id="synthetic-jit",
        provider=provider,
        target=_target(),
        toolchain=_toolchain(provider),
        runtime=_runtime(),
        logical_kernel=LogicalKernel("synthetic-jit", "add", "cpu.add", "default"),
        declared_input_uris=("provider://source/add.py",),
        inputs=(
            CompilationInput(0, "provider://source/add.py", "source", _SOURCE_DIGEST),
        ),
        compile_options=("tilelang-opt=2",),
        artifacts=(
            GeneratedArtifact(0, "generated-host-source", _HOST_DIGEST),
            GeneratedArtifact(1, "generated-device-source", _DEVICE_DIGEST),
            GeneratedArtifact(
                2,
                "runtime-executable",
                _EXECUTABLE_DIGEST,
                _EXECUTABLE_DIGEST,
            ),
        ),
        specialization_axes=(SpecializationAxis("block_size", block_size),),
        generated_host_source_artifact_ordinal=0,
        generated_device_source_artifact_ordinal=1,
        runtime_artifact_ordinals=(2,),
    )


def _record(
    *,
    stage: VerificationStage,
    case_id: str,
    receipt_id: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        stage=stage,
        test_class=VerificationClass.EXACT_ARITHMETIC,
        case=CaseDescriptor(
            operation="add",
            kernel_id="cpu.add",
            variant="default",
            input_dtypes=("Float32", "Float32"),
            output_dtype="Float32",
            shapes=((4,), (4,)),
            accumulator_dtype=None,
            contraction_length=None,
            seed=7,
            case_id=case_id,
            plan=PlanKey(
                operation="add",
                operands=(
                    ("Lhs", "Float32", None),
                    ("Rhs", "Float32", None),
                ),
                compute="Float32",
                accumulation=None,
                accumulator_dtype=None,
                output="Float32",
            ),
        ),
        target_input_bit_hashes=("a" * 64, "b" * 64),
        oracle_input_bit_hashes=("a" * 64, "b" * 64),
        tolerance=Tolerance(),
        deviations=Deviations(0.0, 0.0, 0),
        mismatches=0,
        outcome=VerificationOutcome.PASSED,
        compilation_receipt_id=receipt_id,
    )


def _semantic_record(
    *,
    test_class: VerificationClass,
    outcome: VerificationOutcome,
    stage: VerificationStage,
    tolerance: Tolerance,
    deviations: Deviations,
    mismatches: int | None,
    diagnostic: str | None,
) -> EvidenceRecord:
    return replace(
        _record(stage=stage, case_id=f"semantic-{test_class.value}-{outcome.value}"),
        test_class=test_class,
        outcome=outcome,
        tolerance=tolerance,
        deviations=deviations,
        mismatches=mismatches,
        diagnostic=diagnostic,
    )


def _mixed_report() -> VerificationReport:
    cpu = _compiled_receipt()
    jit_64 = _jit_receipt(64)
    jit_128 = _jit_receipt(128)
    bundle = make_compilation_bundle((jit_128, cpu, jit_64))
    oracle = _record(stage=VerificationStage.ORACLE, case_id="a-oracle")
    assert oracle.case.plan is not None
    certificate = OracleCertificate.from_records(
        KernelDescriptor("add", "cpu.add", "default", "test-provider"),
        (VerificationClass.EXACT_ARITHMETIC,),
        (oracle,),
        required_plan_classes=(
            (oracle.case.plan, (VerificationClass.EXACT_ARITHMETIC,)),
        ),
    )
    target = replace(
        _record(
            stage=VerificationStage.TARGET,
            case_id="b-target",
            receipt_id=jit_64.receipt_id,
        ),
        consumed_certificate_digest=certificate_value(certificate)[
            "certificate_digest"
        ],
    )
    return make_verification_report(
        (oracle, target),
        (certificate,),
        selected_target_profile="synthetic-jit",
        oracle_profile="cpu-compiled",
        compilation_bundle=bundle,
    )


def _rehash_bundle_and_header(header: dict[str, Any]) -> None:
    bundle = header["compilation_bundle"]
    bundle["bundle_id"] = _digest(
        {
            "receipts": bundle["receipts"],
            "schema_version": bundle["schema_version"],
        }
    )
    header["header_digest"] = _digest(
        {key: value for key, value in header.items() if key != "header_digest"}
    )


def _mutate_report(
    report: VerificationReport,
    mutation: Callable[[dict[str, Any], list[dict[str, Any]]], None],
) -> str:
    values = [json.loads(line) for line in report.to_jsonl().splitlines()]
    header = values[0]
    records = values[1:]
    mutation(header, records)
    return (
        "\n".join(
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
            for value in (header, *records)
        )
        + "\n"
    )


def _replace_target_binding(
    header: dict[str, Any],
    records: list[dict[str, Any]],
    receipt_id: str | None,
) -> None:
    target_record = next(record for record in records if record["stage"] == "stage_two")
    target_record["compilation_receipt_id"] = receipt_id
    binding = next(
        item
        for item in header["evidence_bindings"]
        if item["requirement_id"] == target_record["requirement_id"]
    )
    binding["compilation_receipt_id"] = receipt_id
    binding["binding_id"] = _digest(
        {key: value for key, value in binding.items() if key != "binding_id"}
    )
    header["header_digest"] = _digest(
        {key: value for key, value in header.items() if key != "header_digest"}
    )


def test_mixed_compiled_and_jit_report_is_deterministic_and_self_contained() -> None:
    report = _mixed_report()
    serialized = report.to_jsonl()
    loaded = VerificationReport.from_jsonl(serialized)

    assert loaded == report
    assert loaded.to_jsonl() == serialized
    assert report.header is not None
    assert [item.kind for item in report.header.compilation_bundle.receipts] == [
        "compiled-executable",
        "jit-specialization",
        "jit-specialization",
    ]
    assert all(record.compilation_receipt_id for record in report.records)
    jit_receipts = tuple(
        item
        for item in report.header.compilation_bundle.receipts
        if item.kind == "jit-specialization"
    )
    assert jit_receipts[0].logical_kernel == jit_receipts[1].logical_kernel
    assert jit_receipts[0].receipt_id != jit_receipts[1].receipt_id


def test_loading_uses_only_report_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = _mixed_report().to_jsonl()

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("installed provider facts must not be consulted")

    monkeypatch.setattr(
        "strideweave.verification.reporting.installed_compilation_bundle",
        unavailable,
    )
    monkeypatch.setattr(
        "strideweave.verification.provenance._installed_manifest_value",
        unavailable,
    )

    assert VerificationReport.from_jsonl(serialized).to_jsonl() == serialized


@pytest.mark.parametrize(
    (
        "test_class",
        "outcome",
        "stage",
        "tolerance",
        "deviations",
        "mismatches",
        "diagnostic",
    ),
    [
        (
            VerificationClass.EXACT_ARITHMETIC,
            VerificationOutcome.PASSED,
            VerificationStage.ORACLE,
            Tolerance(),
            Deviations(0.0, 0.0, 0),
            0,
            None,
        ),
        (
            VerificationClass.BIT_EXACT,
            VerificationOutcome.PASSED,
            VerificationStage.TARGET,
            Tolerance(),
            Deviations(0.0, 0.0, 0),
            0,
            None,
        ),
        (
            VerificationClass.STRUCTURAL,
            VerificationOutcome.FAILED,
            VerificationStage.ORACLE,
            Tolerance(),
            Deviations(math.inf, math.inf, 1),
            1,
            "structural mismatch",
        ),
        (
            VerificationClass.ANALYTIC,
            VerificationOutcome.FAILED,
            VerificationStage.ORACLE,
            Tolerance(),
            Deviations(0.0, 0.0, 0),
            0,
            "independent oracle disagreed",
        ),
        (
            VerificationClass.NUMERICAL,
            VerificationOutcome.PASSED,
            VerificationStage.TARGET,
            Tolerance(absolute=1.0, relative=1.0, ulps=8),
            Deviations(0.5, 0.25, 3),
            1,
            None,
        ),
        (
            VerificationClass.ANALYTIC,
            VerificationOutcome.ERROR,
            VerificationStage.ORACLE,
            Tolerance(),
            Deviations(None, None, None),
            None,
            "RuntimeError: unavailable",
        ),
        (
            VerificationClass.NUMERICAL,
            VerificationOutcome.BLOCKED,
            VerificationStage.TARGET,
            Tolerance(),
            Deviations(0.0, 0.0, 0),
            0,
            "oracle certificate unavailable",
        ),
        (
            VerificationClass.DEFERRED,
            VerificationOutcome.DEFERRED,
            VerificationStage.ORACLE,
            Tolerance(),
            Deviations(0.0, 0.0, 0),
            0,
            "vendor implementation",
        ),
    ],
)
def test_evidence_semantic_matrix_accepts_consistent_facts(
    test_class: VerificationClass,
    outcome: VerificationOutcome,
    stage: VerificationStage,
    tolerance: Tolerance,
    deviations: Deviations,
    mismatches: int | None,
    diagnostic: str | None,
) -> None:
    record = _semantic_record(
        test_class=test_class,
        outcome=outcome,
        stage=stage,
        tolerance=tolerance,
        deviations=deviations,
        mismatches=mismatches,
        diagnostic=diagnostic,
    )

    assert record.outcome is outcome


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: replace(
                record,
                mismatches=1,
                deviations=Deviations(math.inf, math.inf, 0xFFFF_FFFF),
            ),
            "passed exact_arithmetic",
        ),
        (
            lambda record: replace(
                record, test_class=VerificationClass.BIT_EXACT, mismatches=1
            ),
            "passed bit_exact",
        ),
        (
            lambda record: replace(
                record,
                test_class=VerificationClass.STRUCTURAL,
                deviations=Deviations(1.0, 0.0, 1),
            ),
            "passed structural",
        ),
        (
            lambda record: replace(
                record,
                test_class=VerificationClass.ANALYTIC,
                mismatches=1,
            ),
            "passed analytic",
        ),
        (
            lambda record: replace(
                record,
                test_class=VerificationClass.NUMERICAL,
                tolerance=Tolerance(absolute=0.25),
                deviations=Deviations(0.5, 0.25, 3),
                mismatches=1,
            ),
            "exceeds its absolute tolerance",
        ),
        (
            lambda record: replace(record, diagnostic="unexpected"),
            "passed evidence cannot carry",
        ),
        (
            lambda record: replace(
                record, outcome=VerificationOutcome.FAILED, mismatches=0
            ),
            "failed evidence requires",
        ),
        (
            lambda record: replace(
                record,
                outcome=VerificationOutcome.ERROR,
                diagnostic="RuntimeError: unavailable",
            ),
            "error evidence must have absent",
        ),
        (
            lambda record: replace(
                record,
                outcome=VerificationOutcome.ERROR,
                deviations=Deviations(None, None, None),
                mismatches=None,
            ),
            "error evidence requires a diagnostic",
        ),
        (
            lambda record: replace(
                record,
                outcome=VerificationOutcome.BLOCKED,
                diagnostic="authorization unavailable",
            ),
            "blocked evidence must belong",
        ),
        (
            lambda record: replace(
                record,
                stage=VerificationStage.TARGET,
                outcome=VerificationOutcome.BLOCKED,
                deviations=Deviations(1.0, 0.0, 1),
                mismatches=1,
                diagnostic="authorization unavailable",
            ),
            "blocked evidence must have zero",
        ),
        (
            lambda record: replace(
                record,
                stage=VerificationStage.TARGET,
                outcome=VerificationOutcome.BLOCKED,
                diagnostic="authorization unavailable",
                compilation_receipt_id="a" * 64,
            ),
            "blocked evidence cannot reference",
        ),
        (
            lambda record: replace(
                record,
                outcome=VerificationOutcome.DEFERRED,
                diagnostic="vendor implementation",
            ),
            "deferred verification class",
        ),
        (
            lambda record: replace(
                record,
                test_class=VerificationClass.DEFERRED,
                outcome=VerificationOutcome.DEFERRED,
            ),
            "deferred evidence requires a diagnostic",
        ),
        (
            lambda record: replace(record, deviations=Deviations(0.0, None, 0)),
            "either all present or all absent",
        ),
        (
            lambda record: replace(record, tolerance=Tolerance(absolute=-1.0)),
            "non-negative",
        ),
        (
            lambda record: replace(record, deviations=Deviations(math.nan, 0.0, 0)),
            "not NaN",
        ),
        (
            lambda record: replace(record, mismatches=-1),
            "non-negative integer",
        ),
    ],
)
def test_evidence_semantic_matrix_rejects_contradictory_facts(
    mutation: Callable[[EvidenceRecord], EvidenceRecord], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mutation(_record(stage=VerificationStage.ORACLE, case_id="semantic-invalid"))


def test_loader_rejects_green_exact_record_with_mismatches_and_infinities() -> None:
    def mutation(_header: dict[str, Any], records: list[dict[str, Any]]) -> None:
        oracle = next(item for item in records if item["stage"] == "stage_one")
        oracle["mismatches"] = 1
        oracle["deviations"] = {
            "maximum_absolute": "Infinity",
            "maximum_relative": "Infinity",
            "maximum_ulps": 0xFFFF_FFFF,
        }

    with pytest.raises(ValueError, match=r"JSONL line 2: passed exact_arithmetic"):
        VerificationReport.from_jsonl(_mutate_report(_mixed_report(), mutation))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda header, _records: header.update(
                schema_version="strideweave.kernel-verification.v2"
            ),
            "unsupported report schema",
        ),
        (
            lambda header, _records: header["compilation_bundle"]["receipts"][1].update(
                kind="future-receipt"
            ),
            "unknown discriminator",
        ),
        (
            lambda header, _records: header["compilation_bundle"]["receipts"][1].update(
                shared_executable_artifact_ordinal=2
            ),
            "fields do not match",
        ),
    ],
)
def test_loader_rejects_unsupported_unknown_and_kind_crossed_headers(
    mutation: Callable[[dict[str, Any], list[dict[str, Any]]], None],
    message: str,
) -> None:
    def rehashed_mutation(
        header: dict[str, Any], records: list[dict[str, Any]]
    ) -> None:
        mutation(header, records)
        _rehash_bundle_and_header(header)

    with pytest.raises(ValueError, match=message):
        VerificationReport.from_jsonl(
            _mutate_report(_mixed_report(), rehashed_mutation)
        )


def test_loader_rejects_forged_receipt_reference_after_internal_rehash() -> None:
    report = _mixed_report()
    assert report.header is not None
    cpu_receipt_id = next(
        item.receipt_id
        for item in report.header.compilation_bundle.receipts
        if item.kind == "compiled-executable"
    )

    with pytest.raises(
        ValueError, match="does not match its profile and logical kernel"
    ):
        VerificationReport.from_jsonl(
            _mutate_report(
                report,
                lambda header, records: _replace_target_binding(
                    header, records, cpu_receipt_id
                ),
            )
        )


def test_loader_rejects_receiptless_error_when_compilation_receipt_exists() -> None:
    def mutation(header: dict[str, Any], records: list[dict[str, Any]]) -> None:
        target = next(item for item in records if item["stage"] == "stage_two")
        target["outcome"] = "error"
        target["diagnostic"] = "execution failed"
        target["deviations"] = {
            "maximum_absolute": None,
            "maximum_relative": None,
            "maximum_ulps": None,
        }
        target["mismatches"] = None
        _replace_target_binding(header, records, None)

    with pytest.raises(
        ValueError, match="compiled evidence has no compilation receipt"
    ):
        VerificationReport.from_jsonl(_mutate_report(_mixed_report(), mutation))


def test_report_rejects_receipt_kind_that_disagrees_with_profile() -> None:
    cpu = _compiled_receipt()
    invalid_target = _compiled_receipt(profile_id="synthetic-jit")
    bundle = make_compilation_bundle((cpu, invalid_target))

    with pytest.raises(ValueError, match="kind does not match"):
        make_verification_report(
            (
                _record(stage=VerificationStage.ORACLE, case_id="a-oracle"),
                _record(
                    stage=VerificationStage.TARGET,
                    case_id="b-target",
                    receipt_id=invalid_target.receipt_id,
                ),
            ),
            selected_target_profile="synthetic-jit",
            oracle_profile="cpu-compiled",
            compilation_bundle=bundle,
        )


def test_loader_requires_canonical_json_newlines_and_evidence_order() -> None:
    serialized = _mixed_report().to_jsonl()
    lines = serialized.splitlines()

    with pytest.raises(ValueError, match="canonical newline"):
        VerificationReport.from_jsonl(serialized.rstrip("\n"))

    header = json.loads(lines[0])
    noncanonical_header = json.dumps(header, sort_keys=False) + "\n"
    with pytest.raises(ValueError, match=r"line 1.*not canonical JSON"):
        VerificationReport.from_jsonl(noncanonical_header)

    reordered = "\n".join((lines[0], lines[2], lines[1])) + "\n"
    with pytest.raises(ValueError, match=r"JSONL line 3.*canonical order"):
        VerificationReport.from_jsonl(reordered)

    record = json.loads(lines[1])
    noncanonical_record = (
        "\n".join((lines[0], json.dumps(record, sort_keys=True), lines[2])) + "\n"
    )
    with pytest.raises(ValueError, match=r"JSONL line 2.*not canonical JSON"):
        VerificationReport.from_jsonl(noncanonical_record)
