import json
import math
import struct
from dataclasses import replace

import pytest

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
    assert [item["case"]["case_id"] for item in parsed] == ["a", "b"]
    assert parsed[0]["schema_version"] == "strideweave.kernel-evidence.v1"
    assert parsed[0]["case"]["operation"] == "reduce"
    assert parsed[0]["case"]["accumulator_dtype"] == "Float64"
    assert parsed[0]["deviations"]["maximum_absolute"] == "Infinity"
    assert "NaN" not in serialized


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
