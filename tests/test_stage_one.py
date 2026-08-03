import math

import pytest

import strideweave.verification.classification as classification
import strideweave.verification.stage_one as stage_one_module
from strideweave.verification import (
    ClassificationDisposition,
    VerificationClass,
    VerificationOutcome,
    run_stage_one,
)


def test_stage_one_emits_evidence_and_certifies_every_active_kernel():
    result = run_stage_one()
    active = [
        descriptor
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.disposition is ClassificationDisposition.ACTIVE
    ]

    assert all(
        record.outcome in {VerificationOutcome.PASSED, VerificationOutcome.DEFERRED}
        for record in result.report.records
    )
    assert {
        (certificate.kernel_id, certificate.variant)
        for certificate in result.certificates
    } == {
        (descriptor.kernel.kernel_id, descriptor.kernel.variant)
        for descriptor in active
    }
    for descriptor in active:
        plan_records = [
            record
            for record in result.report.records
            if record.case.plan == descriptor.plan
            and record.case.kernel_id == descriptor.kernel.kernel_id
        ]
        passed_classes = {
            record.test_class
            for record in plan_records
            if record.outcome is VerificationOutcome.PASSED
        }
        assert set(descriptor.classes) <= passed_classes
    movement = [
        record
        for record in result.report.records
        if record.test_class is VerificationClass.BIT_EXACT
        and record.case.kernel_id.startswith("movement.")
    ]
    assert {record.case.kernel_id.removeprefix("movement.") for record in movement} == {
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
    assert result.report.to_jsonl().count("\n") == len(result.report.records)
    assert len(
        {
            (record.case.kernel_id, record.case.variant, record.case.case_id)
            for record in result.report.records
        }
    ) == len(result.report.records)


def test_stage_one_represents_vendor_work_as_deferred_not_passed():
    result = run_stage_one()
    deferred_descriptors = [
        descriptor
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.disposition is ClassificationDisposition.DEFERRED
    ]
    deferred = [
        record
        for record in result.report.records
        if record.outcome is VerificationOutcome.DEFERRED
    ]

    assert {record.case.plan for record in deferred} == {
        descriptor.plan for descriptor in deferred_descriptors
    }
    assert all(record.diagnostic for record in deferred)
    active_kernel_ids = {
        descriptor.kernel.kernel_id
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.disposition is ClassificationDisposition.ACTIVE
    }
    assert not {
        record.case.kernel_id
        for record in deferred
        if record.case.kernel_id not in active_kernel_ids
    } & {certificate.kernel_id for certificate in result.certificates}


def test_every_active_kernel_in_the_expanded_manifest_is_certified():
    # The manifest is the whole native operation set, not a subset the runner
    # happens to know how to call: an operation reaching C++ without a Stage One
    # case leaves its kernel uncertified rather than silently passing.
    result = run_stage_one()
    descriptors = classification.classify_cpu_kernel_plans()

    active = {
        descriptor.kernel.kernel_id
        for descriptor in descriptors
        if descriptor.disposition is ClassificationDisposition.ACTIVE
    }
    certified = {certificate.kernel_id for certificate in result.certificates}

    assert active
    assert active <= certified
    assert not [
        record
        for record in result.report.records
        if record.outcome in {VerificationOutcome.FAILED, VerificationOutcome.ERROR}
    ]


def test_every_deferred_plan_states_a_concrete_reason():
    deferred = [
        descriptor
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.disposition is ClassificationDisposition.DEFERRED
    ]

    assert deferred
    assert all(descriptor.classes == () for descriptor in deferred)
    assert all(descriptor.deferred_reason for descriptor in deferred)
    # Only the vendor math library and the floating `pow` that calls it are
    # deferred; nothing else is allowed to opt out of certification.
    assert {descriptor.deferred_reason for descriptor in deferred} == {
        classification.VENDOR_TRANSCENDENTAL_REASON,
        "floating pow depends on the vendor math library",
    }


def test_both_accumulator_plans_of_each_sum_reduction_are_certified():
    result = run_stage_one()

    numerical = {
        (record.case.operation, record.case.accumulator_dtype)
        for record in result.report.records
        if record.test_class is VerificationClass.NUMERICAL
        and record.outcome is VerificationOutcome.PASSED
        and record.case.output_dtype == "Float32"
    }

    assert numerical == {
        ("reduce_sum", "Float32"),
        ("reduce_sum", "Float64"),
        ("matmul", "Float32"),
        ("matmul", "Float64"),
    }


def test_a_wrong_addressing_mutation_removes_the_matmul_certificate():
    def corrupt(kernel_id, values):
        if kernel_id == "cpu.matmul":
            return tuple(value + 1 for value in values)
        return values

    result = run_stage_one(corrupt)

    assert "cpu.matmul" not in {
        certificate.kernel_id for certificate in result.certificates
    }
    assert any(
        record.case.kernel_id == "cpu.matmul"
        and record.outcome is VerificationOutcome.FAILED
        for record in result.report.records
    )


def test_an_excessive_numerical_residual_fails_certification():
    def corrupt(kernel_id, values):
        if kernel_id == "cpu.reduce_sum":
            return tuple(value + 0.25 for value in values)
        return values

    result = run_stage_one(corrupt)

    assert "cpu.reduce_sum" not in {
        certificate.kernel_id for certificate in result.certificates
    }
    numerical = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.reduce_sum"
        and record.test_class is VerificationClass.NUMERICAL
    ]
    assert all(record.outcome is VerificationOutcome.FAILED for record in numerical)


def test_stage_one_numerical_cases_include_a_wide_exponent_distribution():
    result = run_stage_one()
    wide = [
        record
        for record in result.report.records
        if record.case.case_id.endswith("numerical-wide-exponents")
    ]

    expected_plans = {
        descriptor.plan
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.disposition is ClassificationDisposition.ACTIVE
        and descriptor.kernel.operation in {"reduce_sum", "matmul"}
        and any(dtype == "Float32" for _, dtype, _ in descriptor.plan.operands)
    }
    assert {record.case.plan for record in wide} == expected_plans
    assert all(record.case.contraction_length == 8 for record in wide)
    assert all(record.outcome is VerificationOutcome.PASSED for record in wide)
    assert all(record.tolerance.absolute > 0.0 for record in wide)
    assert all(
        record.tolerance.version == "stage-one-two-path-gamma-v1" for record in wide
    )


def test_exact_certification_distinguishes_the_sign_of_zero():
    def corrupt(kernel_id, values):
        if kernel_id == "cpu.relu":
            return (-0.0, *values[1:])
        return values

    result = run_stage_one(corrupt)

    assert "cpu.relu" not in {
        certificate.kernel_id for certificate in result.certificates
    }
    relu = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.relu"
    ]
    assert relu[0].outcome is VerificationOutcome.FAILED
    assert relu[0].mismatches == 1


def test_arbitrary_finite_exact_case_detects_fractional_result_mutation():
    def corrupt(kernel_id, values):
        if kernel_id == "cpu.add":
            return tuple(value + 0.25 for value in values)
        return values

    result = run_stage_one(corrupt)
    arbitrary_add = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.add"
        and record.case.case_id.endswith("arbitrary-finite")
    ]

    assert "cpu.add" not in {
        certificate.kernel_id for certificate in result.certificates
    }
    assert arbitrary_add
    assert all(record.outcome is VerificationOutcome.FAILED for record in arbitrary_add)
    assert all(record.case.seed is not None for record in arbitrary_add)


def test_both_exact_witnesses_share_result_mutation_and_fail_closed():
    def corrupt(kernel_id, values):
        if kernel_id == "cpu.add":
            return tuple(value + 0.25 for value in values)
        return values

    result = run_stage_one(corrupt)
    add_exact = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.add"
        and record.test_class is VerificationClass.EXACT_ARITHMETIC
    ]

    assert any(record.case.case_id.endswith("-exact") for record in add_exact)
    assert any(
        record.case.case_id.endswith("-arbitrary-finite") for record in add_exact
    )
    assert all(record.outcome is VerificationOutcome.FAILED for record in add_exact)
    assert all(
        record.mismatches is not None and record.mismatches > 0 for record in add_exact
    )


def test_both_exact_witnesses_record_the_same_recoverable_execution_error(
    monkeypatch,
):
    original_execute = stage_one_module._execute

    def fail_cpu_add(descriptor, payloads, layout, cpu, **options):
        if cpu and descriptor.kernel.operation == "add":
            raise RuntimeError("injected exact witness failure")
        return original_execute(descriptor, payloads, layout, cpu, **options)

    monkeypatch.setattr(stage_one_module, "_execute", fail_cpu_add)
    result = run_stage_one()
    add_errors = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.add"
        and record.test_class is VerificationClass.EXACT_ARITHMETIC
        and record.outcome is VerificationOutcome.ERROR
    ]

    assert any(record.case.case_id.endswith("-exact-error") for record in add_errors)
    assert any(
        record.case.case_id.endswith("-arbitrary-finite-error") for record in add_errors
    )
    assert all(
        record.diagnostic is not None
        and "injected exact witness failure" in record.diagnostic
        for record in add_errors
    )


def test_public_stage_one_fails_closed_on_a_stale_classification(monkeypatch):
    monkeypatch.setitem(
        classification._CLASSIFICATIONS,
        ("cpu.removed", "default"),
        (VerificationClass.EXACT_ARITHMETIC,),
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        run_stage_one()


def test_stage_one_fails_closed_when_a_movement_subject_has_no_case(monkeypatch):
    monkeypatch.setitem(
        classification.MOVEMENT_CLASSIFICATIONS,
        "unimplemented_movement",
        (VerificationClass.BIT_EXACT,),
    )

    with pytest.raises(ValueError, match="no verification case"):
        run_stage_one()


@pytest.mark.parametrize(
    "failure_point",
    ("execution", "result_extraction", "result_encoding", "comparison"),
)
def test_stage_one_records_movement_processing_errors_and_continues(
    monkeypatch, failure_point
):
    if failure_point == "execution":
        monkeypatch.setattr(
            stage_one_module.sw,
            "move",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected movement execution failure")
            ),
        )
    elif failure_point == "result_extraction":
        original_values = stage_one_module._values
        calls = 0

        def fail_extraction_once(tensor):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("injected movement extraction failure")
            return original_values(tensor)

        monkeypatch.setattr(stage_one_module, "_values", fail_extraction_once)
    elif failure_point == "result_encoding":
        original_encode = stage_one_module.EncodedFloat32Payload.from_values
        calls = 0

        def fail_encoding_once(values):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected movement encoding failure")
            return original_encode(values)

        monkeypatch.setattr(
            stage_one_module.EncodedFloat32Payload, "from_values", fail_encoding_once
        )
    else:
        original_comparison = stage_one_module._movement_comparison
        calls = 0

        def fail_comparison_once(expected, actual):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("injected movement comparison failure")
            return original_comparison(expected, actual)

        monkeypatch.setattr(
            stage_one_module, "_movement_comparison", fail_comparison_once
        )

    result = run_stage_one()

    errors = [
        record
        for record in result.report.records
        if record.case.kernel_id == "movement.move"
        and record.outcome is VerificationOutcome.ERROR
    ]
    assert len(errors) == 1
    error = errors[0]
    assert error.case.operation == "move"
    assert error.case.variant == "default"
    assert error.case.shapes == ((2, 5),)
    assert error.target_input_bit_hashes == (
        stage_one_module.adversarial_float32_payload().bit_hash,
    )
    assert error.target_input_bit_hashes == error.oracle_input_bit_hashes
    assert error.deviations.maximum_absolute is None
    assert error.deviations.maximum_relative is None
    assert error.deviations.maximum_ulps is None
    assert error.mismatches is None
    assert any(
        record.case.kernel_id == "movement.view"
        and record.outcome is VerificationOutcome.PASSED
        for record in result.report.records
    )
    assert any(
        record.case.kernel_id == "cpu.reduce_sum"
        and record.outcome is VerificationOutcome.PASSED
        for record in result.report.records
    )


def test_stage_one_records_recoverable_case_errors_and_continues(monkeypatch):
    original_structural = stage_one_module._structural_record

    def failing_reduce(descriptor, transform):
        if descriptor.kernel.operation == "reduce_sum":
            raise ValueError("injected reduce failure")
        return original_structural(descriptor, transform)

    monkeypatch.setattr(stage_one_module, "_structural_record", failing_reduce)
    result = run_stage_one()

    errors = [
        record
        for record in result.report.records
        if record.case.kernel_id == "cpu.reduce_sum"
        and record.outcome is VerificationOutcome.ERROR
    ]
    assert errors
    assert all(record.case.plan is not None for record in errors)
    assert all(record.case.shapes for record in errors)
    assert all(record.target_input_bit_hashes for record in errors)
    assert all(record.deviations.maximum_absolute is None for record in errors)
    assert all(record.mismatches is None for record in errors)
    assert "cpu.reduce_sum" not in {
        certificate.kernel_id for certificate in result.certificates
    }
    assert "cpu.matmul" in {
        certificate.kernel_id for certificate in result.certificates
    }


@pytest.mark.parametrize("operation", ("reduce_sum", "matmul"))
@pytest.mark.parametrize("error_type", (RuntimeError, ValueError))
def test_stage_one_numerical_errors_retain_the_resolved_tolerance(
    monkeypatch, operation, error_type
):
    baseline = run_stage_one()
    expected = next(
        record
        for record in baseline.report.records
        if record.case.operation == operation
        and record.test_class is VerificationClass.NUMERICAL
        and record.case.plan is not None
        and record.case.plan.output == "Float32"
        and record.outcome is VerificationOutcome.PASSED
    )
    original_execute = stage_one_module._execute

    def fail_after_numerical_preparation(descriptor, payloads, layout, cpu, **options):
        if descriptor.plan == expected.case.plan and layout.size == 8:
            raise error_type(f"injected {operation} numerical execution failure")
        return original_execute(descriptor, payloads, layout, cpu, **options)

    monkeypatch.setattr(stage_one_module, "_execute", fail_after_numerical_preparation)
    result = run_stage_one()

    error = next(
        record
        for record in result.report.records
        if record.case.operation == operation
        and record.test_class is VerificationClass.NUMERICAL
        and record.case.plan == expected.case.plan
        and record.outcome is VerificationOutcome.ERROR
    )
    assert error.tolerance == expected.tolerance
    assert error.tolerance.absolute is not None
    assert math.isfinite(error.tolerance.absolute)
    assert error.tolerance.absolute > 0.0
    assert error.tolerance.version == "stage-one-two-path-gamma-v1"
    assert error.target_input_bit_hashes == expected.target_input_bit_hashes
    assert error.case.shapes == expected.case.shapes
    assert error.case.contraction_length == expected.case.contraction_length
    assert error.case.seed == expected.case.seed
    assert error.deviations.maximum_absolute is None
    assert error.mismatches is None
    assert expected.case.kernel_id not in {
        certificate.kernel_id for certificate in result.certificates
    }
    assert any(
        record.case.operation == operation
        and record.test_class is VerificationClass.NUMERICAL
        and record.case.plan != expected.case.plan
        and record.outcome is VerificationOutcome.PASSED
        for record in result.report.records
    )


def test_stage_one_integer_numerical_errors_keep_exact_tolerance():
    descriptor = next(
        descriptor
        for descriptor in classification.classify_cpu_kernel_plans()
        if descriptor.kernel.operation == "reduce_sum"
        and all(
            dtype == "Int32"
            for role, dtype, _ in descriptor.plan.operands
            if role == "TENSOR"
        )
    )

    record = stage_one_module._recoverable_error_record(
        descriptor,
        VerificationClass.NUMERICAL,
        "numerical",
        RuntimeError("injected integer numerical execution failure"),
    )

    assert record.outcome is VerificationOutcome.ERROR
    assert record.tolerance.absolute == 0.0
    assert record.tolerance.version == "exact-integer-v1"
    assert record.deviations.maximum_absolute is None
    assert record.mismatches is None


def test_stage_one_preserves_other_analytic_witnesses_after_one_error(monkeypatch):
    original_analytic = stage_one_module._analytic_record
    seen_reduce_witnesses = 0
    failed_plan = None

    def fail_second_reduce_witness(descriptor, transform, case):
        nonlocal failed_plan, seen_reduce_witnesses
        if descriptor.kernel.operation == "reduce_sum":
            seen_reduce_witnesses += 1
            if seen_reduce_witnesses == 2:
                failed_plan = descriptor.plan
                raise RuntimeError("injected second analytic failure")
        return original_analytic(descriptor, transform, case)

    monkeypatch.setattr(
        stage_one_module, "_analytic_record", fail_second_reduce_witness
    )
    result = run_stage_one()

    assert failed_plan is not None
    affected = [
        record
        for record in result.report.records
        if record.case.plan == failed_plan
        and record.test_class is VerificationClass.ANALYTIC
    ]
    assert any(record.outcome is VerificationOutcome.ERROR for record in affected)
    assert sum(record.outcome is VerificationOutcome.PASSED for record in affected) >= 2
