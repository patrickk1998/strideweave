import copy
import hashlib
import json
import math
import re
import struct
from dataclasses import replace
from typing import Any, cast

import pytest

import strideweave as sw
import strideweave.verification.reporting as reporting
from strideweave.verification import (
    MOVEMENT_CLASSIFICATIONS,
    CaseDescriptor,
    ClassificationDisposition,
    Deviations,
    EncodedFloat32Payload,
    EncodedInputs,
    EncodedInt32Payload,
    EvidenceRecord,
    KernelDescriptor,
    OracleCertificate,
    PlanKey,
    Tolerance,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    VerificationSummary,
    adversarial_float32_payload,
    analytic_cases,
    arbitrary_float32_payload,
    classify_cpu_kernel_plans,
    compare_float32,
    exact_structural_payload,
    float32_ulp_distance,
    gamma_bound,
    native_cpu_kernel_manifest,
    require_complete_classification,
    wide_exponent_float32_payload,
)
from strideweave.verification.cli import main as verification_cli


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rehash_report_header(header: dict[str, Any]) -> None:
    header["header_digest"] = _canonical_digest(
        {key: value for key, value in header.items() if key != "header_digest"}
    )


def _rewire_certificate_digest(
    header: dict[str, Any], evidence: list[dict[str, Any]], certificate: dict[str, Any]
) -> None:
    previous = certificate["certificate_digest"]
    certificate["certificate_digest"] = _canonical_digest(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_digest"
        }
    )
    for record in evidence:
        if record["consumed_certificate_digest"] == previous:
            record["consumed_certificate_digest"] = certificate["certificate_digest"]
    _rehash_report_header(header)


def test_every_native_kernel_and_executable_plan_is_explicitly_classified():
    manifest = native_cpu_kernel_manifest()
    classified = require_complete_classification(manifest)
    plans = classify_cpu_kernel_plans()

    # Completeness is checked against the registry rather than a pinned count,
    # so a kernel added in C++ without a classification fails closed here.
    assert manifest
    assert len(classified) == len(manifest)
    assert {descriptor.kernel for descriptor in plans} == set(manifest)
    assert all(descriptor.classes or descriptor.deferred_reason for descriptor in plans)
    assert all(
        (descriptor.disposition is ClassificationDisposition.DEFERRED)
        == bool(descriptor.deferred_reason)
        for descriptor in plans
    )
    assert set(MOVEMENT_CLASSIFICATIONS) == {
        "move",
        "view",
        "permute",
        "rearrange",
        "broadcast_to",
    }


def test_pow_classification_is_plan_specific():
    pow_plans = [
        item for item in classify_cpu_kernel_plans() if item.plan.operation == "pow"
    ]

    assert any(
        item.disposition is ClassificationDisposition.ACTIVE
        and item.classes == (VerificationClass.EXACT_ARITHMETIC,)
        for item in pow_plans
    )
    assert any(
        item.disposition is ClassificationDisposition.DEFERRED
        and "vendor math" in (item.deferred_reason or "")
        for item in pow_plans
    )


def test_an_unknown_or_duplicate_manifest_entry_fails_closed():
    manifest = native_cpu_kernel_manifest()
    unknown = KernelDescriptor("future", "cpu.future", "default", "_CPUFuture")

    with pytest.raises(ValueError, match="does not exactly match"):
        require_complete_classification((*manifest, unknown))
    with pytest.raises(ValueError, match="duplicate kernel/variant"):
        require_complete_classification((*manifest, manifest[0]))


def test_encoded_inputs_materialize_target_and_oracle_from_identical_bits():
    floating = EncodedFloat32Payload.from_values([0.1, -0.0, 2**24])
    integer = EncodedInt32Payload.from_values([-(2**31), 0, 2**31 - 1])
    inputs = EncodedInputs((floating, integer))

    assert inputs.target_values() == inputs.oracle_values()
    assert inputs.input_hashes == (floating.bit_hash, integer.bit_hash)
    assert floating.bits[0] == struct.unpack("<I", struct.pack("<f", 0.1))[0]
    assert floating.bits[1] == 0x8000_0000


def test_float32_payload_bits_require_non_boolean_integral_uint32_words():
    payload = EncodedFloat32Payload.from_bits((0, 0xFFFF_FFFF))

    assert payload.bits == (0, 0xFFFF_FFFF)
    with pytest.raises(TypeError, match="integral uint32"):
        EncodedFloat32Payload.from_bits((1.5, True))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integral uint32"):
        EncodedFloat32Payload.from_bits(("1",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fit uint32"):
        EncodedFloat32Payload.from_bits((-1,))
    with pytest.raises(ValueError, match="fit uint32"):
        EncodedFloat32Payload.from_bits((0x1_0000_0000,))


def test_payload_generators_are_deterministic_and_cover_wide_finite_values():
    assert arbitrary_float32_payload(12, 20) == arbitrary_float32_payload(12, 20)
    assert arbitrary_float32_payload(12, 20) != arbitrary_float32_payload(13, 20)
    arbitrary = arbitrary_float32_payload(12, 20)
    assert all(math.isfinite(value) for value in arbitrary.values())
    assert all(value != 0.0 for value in arbitrary.values())
    assert any(not value.is_integer() for value in arbitrary.values())
    assert len({(bits >> 23) & 0xFF for bits in arbitrary.bits}) > 5

    wide = wide_exponent_float32_payload(7, 64)
    assert wide == wide_exponent_float32_payload(7, 64)
    assert all(math.isfinite(value) for value in wide.values())
    assert len({(bits >> 23) & 0xFF for bits in wide.bits}) > 20


def test_adversarial_payload_preserves_signed_zero_and_nan_payloads():
    payload = adversarial_float32_payload()

    assert payload.bits[0:2] == (0, 0x8000_0000)
    assert payload.bits[-2:] == (0x7FC0_0001, 0x7FC1_2345)
    assert math.isnan(payload.values()[-1])


@pytest.mark.parametrize(
    ("mantissa_bits", "length"), [(24, 1), (24, 3), (24, 17), (53, 17)]
)
@pytest.mark.parametrize("product", [False, True])
def test_exact_structural_bounds_protect_every_partial(mantissa_bits, length, product):
    payload = exact_structural_payload(
        9, length, product=product, rows=3, mantissa_bits=mantissa_bits
    )
    power = 2 if product else 1

    assert length * payload.operand_bound**power <= 2**mantissa_bits
    assert (length + 1) * payload.operand_bound**power > 2**mantissa_bits or (
        payload.operand_bound + 1
    ) ** power * length > 2**mantissa_bits


def test_analytic_cases_are_fixed_and_independently_named():
    cases = analytic_cases()

    assert len(cases) >= 5
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.operation for case in cases} == {"reduce_sum", "matmul"}


def test_bit_comparison_distinguishes_signed_zero_and_nan_payloads():
    positive_zero = struct.unpack("<f", struct.pack("<I", 0))[0]
    negative_zero = struct.unpack("<f", struct.pack("<I", 0x8000_0000))[0]
    nan_one = struct.unpack("<f", struct.pack("<I", 0x7FC0_0001))[0]
    nan_two = struct.unpack("<f", struct.pack("<I", 0x7FC0_0002))[0]

    zeros = compare_float32([positive_zero], [negative_zero])
    nans = compare_float32([nan_one], [nan_two])

    assert zeros.mismatches == zeros.signed_zero_mismatches == 1
    assert zeros.deviations.maximum_absolute == 0.0
    assert float32_ulp_distance(positive_zero, negative_zero) == 0
    assert nans.mismatches == nans.nan_payload_mismatches == 1


def test_numerical_tolerance_cannot_mask_signed_zero_or_nan_payload_mismatches():
    positive_zero = struct.unpack("<f", struct.pack("<I", 0))[0]
    negative_zero = struct.unpack("<f", struct.pack("<I", 0x8000_0000))[0]
    nan_one = struct.unpack("<f", struct.pack("<I", 0x7FC0_0001))[0]
    nan_two = struct.unpack("<f", struct.pack("<I", 0x7FC0_0002))[0]
    permissive = Tolerance(absolute=math.inf, relative=math.inf, ulps=0xFFFF_FFFF)

    assert not compare_float32([positive_zero], [negative_zero]).within(permissive)
    assert not compare_float32([nan_one], [nan_two]).within(permissive)
    assert compare_float32([1.0], [1.0 + 2.0**-23]).within(
        Tolerance(absolute=1e-6, relative=1e-6, ulps=1)
    )


def test_numerical_comparison_is_cancellation_safe_and_tracks_ulps():
    one = struct.unpack("<f", struct.pack("<I", 0x3F80_0000))[0]
    next_one = struct.unpack("<f", struct.pack("<I", 0x3F80_0001))[0]

    comparison = compare_float32([0.0, one], [1e-30, next_one])

    assert comparison.deviations.maximum_absolute is not None
    assert comparison.deviations.maximum_absolute > 0.0
    assert comparison.deviations.maximum_relative == 1.0
    assert comparison.deviations.maximum_ulps is not None
    assert comparison.deviations.maximum_ulps >= 1


def test_gamma_bound_uses_the_analytic_gamma_k_envelope():
    unit_roundoff = 2.0**-53
    expected = (16 * unit_roundoff) / (1 - 16 * unit_roundoff) * 12.5

    assert gamma_bound(unit_roundoff, 16, 12.5) == expected
    with pytest.raises(ValueError, match="undefined"):
        gamma_bound(0.5, 2, 1.0)
    for epsilon, magnitude in (
        (math.nan, 1.0),
        (math.inf, 1.0),
        (2.0**-53, math.nan),
        (2.0**-53, math.inf),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            gamma_bound(epsilon, 1, magnitude)


def make_record(case_id: str, outcome=VerificationOutcome.PASSED) -> EvidenceRecord:
    return EvidenceRecord(
        stage=VerificationStage.ORACLE,
        test_class=VerificationClass.NUMERICAL,
        case=CaseDescriptor(
            operation="reduce",
            kernel_id="cpu.reduce_sum",
            variant="default",
            input_dtypes=("Float32",),
            output_dtype="Float32",
            shapes=((1, 4),),
            accumulator_dtype="Float64",
            contraction_length=4,
            seed=11,
            case_id=case_id,
        ),
        target_input_bit_hashes=("abc",),
        oracle_input_bit_hashes=("abc",),
        tolerance=Tolerance(absolute=1e-12, version="gamma-k-v1"),
        deviations=Deviations(math.inf, math.inf, 0xFFFF_FFFF),
        mismatches=1,
        outcome=outcome,
        diagnostic="payload mismatch",
    )


def test_jsonl_evidence_is_versioned_complete_finite_and_deterministic():
    first = make_record("b")
    second = make_record("a", VerificationOutcome.DEFERRED)
    report = VerificationReport((first, second))

    serialized = report.to_jsonl()
    parsed = [json.loads(line) for line in serialized.splitlines()]

    assert serialized == VerificationReport((second, first)).to_jsonl()
    assert parsed[0]["schema_version"] == "strideweave.kernel-verification.v2"
    assert parsed[0]["evidence_schema"] == "strideweave.kernel-evidence.v2"
    assert [item["case"]["case_id"] for item in parsed[1:]] == ["a", "b"]
    assert parsed[1]["schema_version"] == "strideweave.kernel-evidence.v2"
    assert parsed[1]["case"]["operation"] == "reduce"
    assert parsed[1]["case"]["accumulator_dtype"] == "Float64"
    assert parsed[1]["deviations"]["maximum_absolute"] == "Infinity"
    assert "NaN" not in serialized


@pytest.mark.parametrize(
    "records",
    [
        (),
        (make_record("unsupported-report-schema"),),
    ],
    ids=("empty", "populated"),
)
def test_verification_report_rejects_unsupported_schema_versions(records):
    with pytest.raises(ValueError, match="unsupported report schema version 'future'"):
        VerificationReport(records, schema_version="future")


def test_verification_report_accepts_current_schema_version():
    report = VerificationReport(
        (make_record("current-report-schema"),),
        schema_version="strideweave.kernel-verification.v2",
    )

    assert VerificationReport.from_jsonl(report.to_jsonl()).schema_version == (
        "strideweave.kernel-verification.v2"
    )


def test_verification_report_round_trips_strict_jsonl_and_files(tmp_path):
    plan = PlanKey(
        "reduce_sum",
        (("TENSOR", "Float32", "Float32"),),
        "BINARY32",
        "FLOATING",
        "Float64",
        "Float32",
    )
    passed = replace(
        make_record("passed"), case=replace(make_record("passed").case, plan=plan)
    )
    blocked = replace(
        make_record("blocked", VerificationOutcome.BLOCKED),
        deviations=Deviations(None, None, None),
        mismatches=None,
        diagnostic="certificate unavailable",
    )
    errored = replace(
        make_record("errored", VerificationOutcome.ERROR),
        deviations=Deviations(math.nan, -math.inf, None),
        mismatches=None,
    )
    deferred = make_record("deferred", VerificationOutcome.DEFERRED)
    report = VerificationReport((passed, blocked, errored, deferred))

    serialized = report.to_jsonl()
    loaded = VerificationReport.from_jsonl(serialized)
    path = tmp_path / "evidence.jsonl"
    report.write(path)

    assert loaded.to_jsonl() == serialized
    assert VerificationReport.load(path).to_jsonl() == serialized
    assert any(
        math.isnan(record.deviations.maximum_absolute or 0.0)
        for record in loaded.records
    )
    assert any(record.case.plan == plan for record in loaded.records)


def test_verification_report_round_trips_integer_float_fields(tmp_path):
    record = replace(
        make_record("integer-float-fields"),
        tolerance=Tolerance(absolute=0, relative=1, ulps=2, version="integer-values"),
        deviations=Deviations(3, 4, 5),
    )
    report = VerificationReport((record,))
    path = tmp_path / "integer-floats.jsonl"

    serialized = report.to_jsonl()
    loaded = VerificationReport.from_jsonl(serialized)
    report.write(path)
    loaded_record = loaded.records[0]

    assert json.loads(serialized.splitlines()[1])["tolerance"]["absolute"] == 0
    assert type(loaded_record.tolerance.absolute) is float
    assert type(loaded_record.tolerance.relative) is float
    assert type(loaded_record.deviations.maximum_absolute) is float
    assert type(loaded_record.deviations.maximum_relative) is float
    assert VerificationReport.load(path).records == loaded.records


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ("{", "Expecting property name"),
        ('{"schema_version":"future"}', "fields do not match"),
        ('{"schema_version":"future","schema_version":"future"}', "duplicate field"),
    ],
)
def test_verification_report_load_reports_json_parse_failures_by_line(
    serialized, message
):
    with pytest.raises(ValueError, match=rf"JSONL line 1: .*{message}"):
        VerificationReport.from_jsonl(serialized)


def test_prototype_v1_evidence_only_files_are_rejected_not_migrated():
    old_record = make_record("prototype").as_json_object()
    old_record["schema_version"] = "strideweave.kernel-evidence.v1"

    with pytest.raises(ValueError, match="JSONL line 1: report header"):
        VerificationReport.from_jsonl(json.dumps(old_record) + "\n")


@pytest.mark.parametrize(
    ("line_index", "field", "expected"),
    [
        (
            0,
            "timestamp",
            "JSONL line 1: report header fields do not match: "
            "missing=[], unexpected=['timestamp']",
        ),
        (
            1,
            "database",
            "JSONL line 2: record has unexpected fields database",
        ),
    ],
)
def test_verification_report_rejects_status_store_fields(
    line_index: int, field: str, expected: str
) -> None:
    lines = (
        VerificationReport((make_record("status-store-field"),)).to_jsonl().splitlines()
    )
    value = json.loads(lines[line_index])
    value[field] = "not-report-metadata"
    lines[line_index] = json.dumps(value, separators=(",", ":"), sort_keys=True)

    with pytest.raises(ValueError, match=re.escape(expected)) as caught:
        VerificationReport.from_jsonl("\n".join(lines) + "\n")

    assert str(caught.value) == expected


@pytest.mark.parametrize(
    "mutation",
    ("missing", "unknown", "duplicate", "mismatched_receipt", "forged"),
)
def test_provenance_complete_report_loading_fails_closed_for_header_tampering(
    mutation: str,
) -> None:
    lines = VerificationReport((make_record("tamper"),)).to_jsonl().splitlines()
    header = json.loads(lines[0])
    evidence = json.loads(lines[1])
    if mutation == "missing":
        del header["compilation"]
    elif mutation == "unknown":
        header["unexpected"] = True
    elif mutation == "duplicate":
        header["tolerance_policies"].append(
            copy.deepcopy(header["tolerance_policies"][0])
        )
        _rehash_report_header(header)
    elif mutation == "mismatched_receipt":
        evidence["compilation_receipt_id"] = next(
            item["receipt_id"]
            for item in header["compilation"]["kernel_receipts"]
            if item["kernel"]["kernel_id"] != evidence["case"]["kernel_id"]
        )
    else:
        header["compilation"]["manifest"]["manifest_digest"] = "0" * 64
        _rehash_report_header(header)
    serialized = (
        json.dumps(header, separators=(",", ":"), sort_keys=True)
        + "\n"
        + json.dumps(evidence, separators=(",", ":"), sort_keys=True)
        + "\n"
    )

    with pytest.raises(
        ValueError, match=r"fields do not match|duplicate|does not match"
    ):
        VerificationReport.from_jsonl(serialized)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("schema_version", "future"),
        lambda value: value.pop("outcome"),
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value.__setitem__("outcome", "unknown"),
        lambda value: value["case"].__setitem__("shapes", ["not-a-shape"]),
        lambda value: value.__setitem__("oracle_input_bit_hashes", ["different"]),
        lambda value: value["tolerance"].__setitem__("absolute", True),
        lambda value: value["deviations"].__setitem__("maximum_relative", []),
        lambda value: value["tolerance"].__setitem__("absolute", 10**400),
    ],
    ids=(
        "schema",
        "missing",
        "unexpected",
        "enum",
        "nested_type",
        "model_invariant",
        "float_boolean",
        "float_collection",
        "unrepresentable_float_integer",
    ),
)
def test_verification_report_load_fails_closed_for_invalid_evidence(mutate):
    report = VerificationReport((make_record("invalid"),))
    lines = report.to_jsonl().splitlines()
    value = json.loads(lines[1])
    mutate(value)
    serialized = lines[0] + "\n" + json.dumps(value, separators=(",", ":")) + "\n"

    with pytest.raises(ValueError, match=r"JSONL line 2|reference does not match"):
        VerificationReport.from_jsonl(serialized)


def test_verification_report_load_rejects_nonstandard_json_nonfinite_values():
    serialized = VerificationReport((make_record("nonstandard"),)).to_jsonl()
    header, evidence = serialized.splitlines()
    serialized = header + "\n" + evidence.replace('"Infinity"', "Infinity") + "\n"

    with pytest.raises(ValueError, match="JSONL line 2: non-standard JSON constant"):
        VerificationReport.from_jsonl(serialized)


def test_verification_report_summary_repr_description_and_views_are_bounded():
    passed = make_record("passed")
    deferred = make_record("deferred", VerificationOutcome.DEFERRED)
    failed = replace(
        make_record("failed", VerificationOutcome.FAILED),
        stage=VerificationStage.TARGET,
        test_class=VerificationClass.STRUCTURAL,
        case=replace(make_record("failed").case, kernel_id="synthetic.reduce_sum"),
    )
    errored = replace(
        make_record("errored", VerificationOutcome.ERROR),
        test_class=VerificationClass.ANALYTIC,
    )
    blocked = replace(
        make_record("blocked", VerificationOutcome.BLOCKED),
        stage=VerificationStage.TARGET,
        test_class=VerificationClass.DEFERRED,
    )
    report = VerificationReport((passed, deferred, failed, errored, blocked))

    summary = report.summary()

    assert isinstance(summary, VerificationSummary)
    assert summary.total == 5
    assert dict(summary.outcomes) == {
        VerificationOutcome.PASSED: 1,
        VerificationOutcome.FAILED: 1,
        VerificationOutcome.ERROR: 1,
        VerificationOutcome.BLOCKED: 1,
        VerificationOutcome.DEFERRED: 1,
    }
    assert dict(summary.stages) == {
        VerificationStage.ORACLE: 3,
        VerificationStage.TARGET: 2,
    }
    assert dict(summary.classes)[VerificationClass.NUMERICAL] == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.errors == 1
    assert summary.blocked == 1
    assert summary.deferred == 1
    assert not summary.gate_passed
    assert tuple(record.case.case_id for record in report.passed.records) == ("passed",)
    assert tuple(record.case.case_id for record in report.deferred.records) == (
        "deferred",
    )
    assert tuple(record.case.case_id for record in report.problems.records) == (
        "failed",
        "errored",
        "blocked",
    )
    assert "EvidenceRecord" not in repr(report)
    assert len(repr(report)) < 160
    assert repr(summary) == (
        "VerificationSummary(total=5, passed=1, failed=1, errors=1, blocked=1, "
        "deferred=1, gate_passed=False)"
    )
    assert len(repr(summary)) < 160
    assert report.describe() == (
        "Verification report: 5 records; gate passed: no. "
        "Outcomes: passed=1, failed=1, error=1, blocked=1, deferred=1. "
        "Stages: stage_one=3, stage_two=2. "
        "Classes: bit_exact=0, exact_arithmetic=0, structural=1, analytic=1, "
        "numerical=2, deferred=1."
    )


def test_verification_report_empty_and_deferred_only_reports_pass_the_gate():
    empty = VerificationReport(())
    deferred = VerificationReport(
        (make_record("deferred", VerificationOutcome.DEFERRED),)
    )

    assert empty.summary().gate_passed
    assert deferred.summary().gate_passed
    assert deferred.problems.records == ()
    assert deferred.deferred.records == deferred.records


def test_verification_report_select_composes_all_supported_filters():
    first = replace(
        make_record("first"),
        case=replace(
            make_record("first").case,
            operation="matmul",
            kernel_id="synthetic.matmul",
            variant="wide",
        ),
    )
    second = replace(
        make_record("second", VerificationOutcome.FAILED),
        stage=VerificationStage.TARGET,
        test_class=VerificationClass.STRUCTURAL,
        case=replace(
            make_record("second").case,
            operation="matmul",
            kernel_id="synthetic.matmul",
            variant="wide",
        ),
    )
    other = replace(
        make_record("other", VerificationOutcome.FAILED),
        case=replace(make_record("other").case, operation="reduce_sum"),
    )
    report = VerificationReport((first, second, other))

    selected = report.select(
        stage=(VerificationStage.TARGET,),
        outcomes=(VerificationOutcome.FAILED, VerificationOutcome.BLOCKED),
        test_class=VerificationClass.STRUCTURAL,
        operation="matmul",
        kernel_id="synthetic.matmul",
        variant="wide",
    ).select(outcomes=VerificationOutcome.FAILED)

    assert tuple(record.case.case_id for record in selected.records) == ("second",)
    assert selected.schema_version == report.schema_version
    with pytest.raises(TypeError, match="outcomes"):
        report.select(outcomes=cast(Any, "failed"))


def test_report_filtering_preserves_embedded_historical_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = sw.test_backend()
    assert report.header is not None
    original_header = report.header.as_json_object()

    def moved_installed_provenance(*args: object, **kwargs: object) -> object:
        raise AssertionError("read-only filtering consulted installed provenance")

    monkeypatch.setattr(reporting, "_compilation_value", moved_installed_provenance)
    monkeypatch.setattr(reporting, "_generic_oracle_value", moved_installed_provenance)

    assert report.select() is report
    selected = report.select(
        stage=VerificationStage.TARGET,
        operation="reduce_sum",
    )
    assert selected.header is not None
    assert selected.header.compilation == report.header.compilation
    assert {
        item["oracle_reference_id"] for item in selected.header.oracle_references
    } == {record.oracle_reference_id for record in selected.records}
    consumed = {
        record.consumed_certificate_digest
        for record in selected.records
        if record.consumed_certificate_digest is not None
    }
    assert {
        item["certificate_digest"] for item in selected.header.certificates
    } == consumed
    receipt_ids = {
        item["receipt_id"] for item in selected.header.compilation["kernel_receipts"]
    }
    assert all(
        record.compilation_receipt_id in receipt_ids
        for record in selected.records
        if record.compilation_receipt_id is not None
    )
    assert VerificationReport.from_jsonl(selected.to_jsonl()) == selected
    partial = report.select(
        operation="reduce_sum", test_class=VerificationClass.NUMERICAL
    )
    assert any(record.stage is VerificationStage.ORACLE for record in partial.records)
    assert VerificationReport.from_jsonl(partial.to_jsonl()) == partial
    assert report.header.as_json_object() == original_header


def test_problem_view_keeps_certificates_with_unrelated_stage_one_failures() -> None:
    original = sw.test_backend()
    assert original.header is not None
    unrelated_stage_one = next(
        record
        for record in original.records
        if record.stage is VerificationStage.ORACLE
        and record.case.kernel_id not in {"cpu.reduce_sum", "cpu.matmul"}
        and record.outcome is VerificationOutcome.PASSED
    )
    certified_stage_two = next(
        record
        for record in original.records
        if record.stage is VerificationStage.TARGET
        and record.case.kernel_id == "cpu.reduce_sum"
        and record.outcome is VerificationOutcome.PASSED
    )
    changed = tuple(
        replace(record, outcome=VerificationOutcome.FAILED, diagnostic="mismatch")
        if record in {unrelated_stage_one, certified_stage_two}
        else record
        for record in original.records
    )
    records, header = reporting.bind_report(
        changed, (), certificate_facts_override=original.header.certificates
    )
    report = VerificationReport(records, header.schema_version, header)

    problems = report.problems

    assert {record.case.case_id for record in problems.records} == {
        unrelated_stage_one.case.case_id,
        certified_stage_two.case.case_id,
    }
    assert any(record.stage is VerificationStage.ORACLE for record in problems.records)
    loaded = VerificationReport.from_jsonl(problems.to_jsonl())
    assert loaded.to_jsonl() == problems.to_jsonl()


@pytest.mark.parametrize(
    "mutation",
    (
        "evidence_digest",
        "certified_classes",
        "plan_coverage",
        "plan_classes",
        "kernel",
        "variant",
    ),
)
def test_report_loading_reconstructs_stage_one_certificates_from_evidence(
    mutation: str,
) -> None:
    lines = sw.test_backend().to_jsonl().splitlines()
    header = json.loads(lines[0])
    evidence = [json.loads(line) for line in lines[1:]]
    certificate = next(
        item for item in header["certificates"] if item["kernel_id"] == "cpu.reduce_sum"
    )
    if mutation == "evidence_digest":
        certificate["evidence_digest"] = "0" * 64
    elif mutation == "certified_classes":
        certificate["certified_classes"].pop()
    elif mutation == "plan_coverage":
        certificate["certified_plan_classes"].pop()
    elif mutation == "plan_classes":
        certificate["certified_plan_classes"][0]["classes"].pop()
    elif mutation == "kernel":
        certificate["kernel_id"] = "cpu.wrong_reduce_sum"
    else:
        certificate["variant"] = "wrong"
    _rewire_certificate_digest(header, evidence, certificate)
    serialized = "\n".join(
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        for value in (header, *evidence)
    )

    with pytest.raises(ValueError, match="certificate"):
        VerificationReport.from_jsonl(serialized + "\n")


def test_verification_report_cli_uses_model_description_and_gate_exit_code(
    tmp_path, capsys
):
    report = VerificationReport((make_record("passing"),))
    path = tmp_path / "passing.jsonl"
    report.write(path)

    exit_code = verification_cli([str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == report.describe() + "\n"
    assert captured.err == ""


def test_verification_report_cli_filters_verbose_records_and_machine_json(
    tmp_path, capsys
):
    passed = make_record("passing")
    failed = replace(
        make_record("failed", VerificationOutcome.FAILED),
        stage=VerificationStage.TARGET,
        case=replace(
            make_record("failed").case,
            operation="matmul",
            kernel_id="synthetic.matmul",
            variant="wide",
        ),
    )
    report = VerificationReport((passed, failed))
    path = tmp_path / "mixed.jsonl"
    report.write(path)

    verbose_exit = verification_cli(
        [
            str(path),
            "--problems",
            "--stage",
            "stage_two",
            "--operation",
            "matmul",
            "--kernel",
            "synthetic.matmul",
            "--variant",
            "wide",
            "--outcome",
            "failed",
            "--verbose",
        ]
    )
    verbose = capsys.readouterr()
    json_exit = verification_cli([str(path), "--json", "--verbose"])
    first_json = capsys.readouterr()
    repeated_json_exit = verification_cli([str(path), "--json", "--verbose"])
    second_json = capsys.readouterr()

    assert verbose_exit == 1
    assert "Verification report: 1 records; gate passed: no." in verbose.out
    assert (
        "case_id=failed stage=stage_two operation=matmul kernel_id=synthetic.matmul "
        "variant=wide class=numerical outcome=failed deviations="
    ) in verbose.out
    assert "EvidenceRecord(" not in verbose.out
    assert json_exit == repeated_json_exit == 1
    assert first_json.out == second_json.out
    payload = json.loads(first_json.out)
    assert payload["outcomes"] == {
        "blocked": 0,
        "deferred": 0,
        "error": 0,
        "failed": 1,
        "passed": 1,
    }
    assert [record["case_id"] for record in payload["records"]] == ["failed", "passing"]


def test_verification_report_cli_returns_two_for_invalid_reports_and_usage(
    tmp_path, capsys
):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{\n", encoding="utf-8")

    assert verification_cli([str(malformed)]) == 2
    assert "JSONL line 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage_error:
        verification_cli([str(malformed), "--stage", "not-a-stage"])
    assert usage_error.value.code == 2


def test_evidence_rejects_independently_encoded_input_hashes():
    record = make_record("hashes")
    with pytest.raises(ValueError, match="input bit hashes must match"):
        EvidenceRecord(
            stage=record.stage,
            test_class=record.test_class,
            case=record.case,
            target_input_bit_hashes=("target",),
            oracle_input_bit_hashes=("oracle",),
            tolerance=record.tolerance,
            deviations=record.deviations,
            mismatches=record.mismatches,
            outcome=record.outcome,
        )


def test_oracle_certificate_requires_every_required_class_to_pass():
    kernel = KernelDescriptor("reduce", "cpu.reduce_sum", "default", "_CPU")
    record = make_record("certificate")

    certificate = OracleCertificate.from_records(
        kernel, (VerificationClass.NUMERICAL,), (record,)
    )

    assert len(certificate.evidence_digest) == 64
    assert certificate.certified_plan_classes == ()
    with pytest.raises(ValueError, match="missing passed classes"):
        OracleCertificate.from_records(
            kernel, (VerificationClass.STRUCTURAL,), (record,)
        )


def test_oracle_certificate_requires_passed_evidence_for_each_plan():
    kernel = KernelDescriptor("reduce", "cpu.reduce_sum", "default", "_CPU")
    float32_plan = PlanKey(
        "reduce",
        (("TENSOR", "Float32", "Float32"),),
        "BINARY32",
        "FLOATING",
        "Float32",
        "Float32",
    )
    float64_plan = replace(float32_plan, accumulator_dtype="Float64")
    record = make_record("per-plan")
    record = replace(record, case=replace(record.case, plan=float32_plan))

    with pytest.raises(ValueError, match=r"plan .*missing passed classes"):
        OracleCertificate.from_records(
            kernel,
            (VerificationClass.NUMERICAL,),
            (record,),
            required_plan_classes=(
                (float32_plan, (VerificationClass.NUMERICAL,)),
                (float64_plan, (VerificationClass.NUMERICAL,)),
            ),
        )
