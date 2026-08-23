from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from evidence_doubles import RecordingEvidenceStore

import strideweave as sw
import strideweave.carriers.metal._executor as pointwise_executor
import strideweave.carriers.metal._jit as metal_jit
import strideweave.verification.stage_one as stage_one
import strideweave.verification.stage_two as stage_two
import strideweave.verification.store.recording as recording
from strideweave.carriers.metal._address_plan import (
    address_plan,
    address_plan_from_key,
)
from strideweave.carriers.metal._verification import (
    capture_metal_compilations,
    current_metal_compilation_bundle,
    metal_compilation_receipt,
    metal_kernel_metadata,
)
from strideweave.carriers.metal.capabilities import metal_capabilities
from strideweave.verification import (
    ClassificationDisposition,
    PlanKey,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    classify_profile_plans,
    kernel_manifest,
    profile_subjects,
    run_oracle_stage,
    run_target_stage,
    verification_profile,
)
from strideweave.verification.provenance import (
    JITSpecializationReceipt,
    SpecializationAxis,
    make_compilation_bundle,
    make_jit_specialization_receipt,
)
from strideweave.verification.reporting import bind_report
from strideweave.verification.store import VerificationStoreError

pytestmark = pytest.mark.metal


def _tensor(values: tuple[float, ...]) -> sw.Tensor:
    layout = sw.Layout(sw.Shape(len(values)), sw.Stride(1))
    carrier = sw.Metal(len(values)).new_like(values)
    return sw.Tensor(carrier, 0, layout)


@pytest.fixture(scope="module")
def current_metal_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[VerificationReport, Path]:
    output = tmp_path_factory.mktemp("metal-verification") / "report.jsonl"
    return sw.verify_backend("metal-tilelang", output=output), output


def _receipt_with_malformed_axis(
    receipt: JITSpecializationReceipt,
) -> JITSpecializationReceipt:
    malformed_axes = (
        SpecializationAxis(receipt.specialization_axes[0].name, ()),
        *receipt.specialization_axes[1:],
    )
    return make_jit_specialization_receipt(
        profile_id=receipt.profile_id,
        provider=receipt.provider,
        target=receipt.target,
        toolchain=receipt.toolchain,
        runtime=receipt.runtime,
        logical_kernel=receipt.logical_kernel,
        declared_input_uris=receipt.declared_input_uris,
        inputs=receipt.inputs,
        compile_options=receipt.compile_options,
        artifacts=receipt.artifacts,
        specialization_axes=malformed_axes,
        generated_host_source_artifact_ordinal=(
            receipt.generated_host_source_artifact_ordinal
        ),
        generated_device_source_artifact_ordinal=(
            receipt.generated_device_source_artifact_ordinal
        ),
        runtime_artifact_ordinals=receipt.runtime_artifact_ordinals,
    )


def _receipt_with_axis_value(
    receipt: JITSpecializationReceipt,
    name: str,
    value: object,
) -> JITSpecializationReceipt:
    axes = tuple(
        SpecializationAxis(
            axis.name,
            metal_jit._canonical_value(value) if axis.name == name else axis.value,
        )
        for axis in receipt.specialization_axes
    )
    return make_jit_specialization_receipt(
        profile_id=receipt.profile_id,
        provider=receipt.provider,
        target=receipt.target,
        toolchain=receipt.toolchain,
        runtime=receipt.runtime,
        logical_kernel=receipt.logical_kernel,
        declared_input_uris=receipt.declared_input_uris,
        inputs=receipt.inputs,
        compile_options=receipt.compile_options,
        artifacts=receipt.artifacts,
        specialization_axes=axes,
        generated_host_source_artifact_ordinal=(
            receipt.generated_host_source_artifact_ordinal
        ),
        generated_device_source_artifact_ordinal=(
            receipt.generated_device_source_artifact_ordinal
        ),
        runtime_artifact_ordinals=receipt.runtime_artifact_ordinals,
    )


def _key_from_receipt(
    receipt: JITSpecializationReceipt,
) -> metal_jit.MetalSpecializationKey:
    return metal_jit.MetalSpecializationKey(
        metal_jit.LogicalKernel(
            receipt.logical_kernel.kernel_id,
            receipt.logical_kernel.variant,
        ),
        tuple(
            (axis.name, cast(metal_jit.CanonicalValue, axis.value))
            for axis in receipt.specialization_axes
        ),
    )


def _plan_mutations(plan: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    operation, operands_value, compute, accumulation, accumulator_dtype, output = plan
    operands = cast(tuple[tuple[str, str | None, str], ...], operands_value)
    role, dtype, convert_to = operands[0]
    changed_role = (
        ("weak_scalar", None, convert_to)
        if role == "tensor"
        else ("tensor", "Float32", convert_to)
    )
    changed_dtype = "Int32" if dtype != "Int32" else "Float32"
    changed_conversion = "Int32" if convert_to != "Int32" else "Float32"
    changed_compute = "int32_exact" if compute != "int32_exact" else "binary32"
    changed_accumulation = (
        "maximum" if accumulation != "maximum" else "sequential_binary32"
    )
    changed_accumulator = "Int32" if accumulator_dtype != "Int32" else "Float32"
    changed_output = "Int32" if output != "Int32" else "Float32"
    return (
        (
            "operation",
            (
                "sub" if operation != "sub" else "add",
                operands,
                compute,
                accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "operand role",
            (
                operation,
                (changed_role, *operands[1:]),
                compute,
                accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "operand dtype",
            (
                operation,
                ((role, changed_dtype, convert_to), *operands[1:]),
                compute,
                accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "operand conversion",
            (
                operation,
                ((role, dtype, changed_conversion), *operands[1:]),
                compute,
                accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "compute",
            (
                operation,
                operands,
                changed_compute,
                accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "accumulation",
            (
                operation,
                operands,
                compute,
                changed_accumulation,
                accumulator_dtype,
                output,
            ),
        ),
        (
            "accumulator dtype",
            (
                operation,
                operands,
                compute,
                accumulation,
                changed_accumulator,
                output,
            ),
        ),
        (
            "output",
            (
                operation,
                operands,
                compute,
                accumulation,
                accumulator_dtype,
                changed_output,
            ),
        ),
    )


def _report_with_replaced_receipt(
    report: VerificationReport,
    original: JITSpecializationReceipt,
    replacement: JITSpecializationReceipt,
) -> VerificationReport:
    assert report.header is not None
    records = tuple(
        replace(
            record,
            compilation_receipt_id=(
                replacement.receipt_id
                if record.compilation_receipt_id == original.receipt_id
                else record.compilation_receipt_id
            ),
            supporting_compilation_receipt_ids=tuple(
                replacement.receipt_id if value == original.receipt_id else value
                for value in record.supporting_compilation_receipt_ids
            ),
        )
        for record in report.records
    )
    bundle = make_compilation_bundle(
        tuple(
            replacement if receipt == original else receipt
            for receipt in report.header.compilation_bundle.receipts
        )
    )
    rebound, header = bind_report(
        records,
        (),
        selected_target_profile=report.header.selected_target_profile,
        oracle_profile=report.header.oracle_profile,
        compilation_bundle=bundle,
        certificate_facts_override=report.header.certificates,
    )
    return VerificationReport(rebound, header.schema_version, header)


def test_cached_pointwise_recipe_does_not_retain_user_storage() -> None:
    lhs = _tensor((1.0, 2.0))
    rhs = _tensor((3.0, 4.0))
    storage = lhs.carrier._require_storage()  # type: ignore[attr-defined]
    storage_reference = weakref.ref(storage)

    result = pointwise_executor.execute_expression(
        "add",
        (lhs, rhs),
        output_layout=lhs.layout,
        convert_dtypes=(sw.DType.Float32, sw.DType.Float32),
    )
    lhs.carrier._runtime.synchronize()  # type: ignore[attr-defined]
    assert len(pointwise_executor._JIT_CACHE) > 0

    del result, lhs, rhs, storage
    gc.collect()

    assert storage_reference() is None


def test_full_metal_report_records_in_a_fresh_process(tmp_path: Path) -> None:
    report = tmp_path / "metal-report.jsonl"
    store = tmp_path / "evidence-store"
    verification = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import strideweave as sw, sys; "
                "sw.verify_backend('metal-tilelang', output=sys.argv[1])"
            ),
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert verification.returncode == 0, verification.stderr
    assert report.is_file()

    recording_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "strideweave.verification.status_cli",
            "record",
            "--report",
            str(report),
            "--producer",
            "cross-process-metal-test",
            "--store",
            str(store),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert recording_process.returncode == 0, recording_process.stderr
    payload = json.loads(recording_process.stdout.splitlines()[-1])
    assert payload["evidence_count"] > 0
    assert payload["store"] == str(store)
    assert store.is_dir()


def test_metal_profile_manifest_and_classifications_are_exact() -> None:
    profile = verification_profile("metal-tilelang")
    manifest = kernel_manifest(profile)
    classifications = classify_profile_plans(profile)

    assert (profile.carrier, profile.provider) == ("Metal", "jit-specialization")
    assert {
        (kernel.operation, kernel.kernel_id, kernel.variant) for kernel in manifest
    } == set(metal_kernel_metadata())
    assert {classification.plan for classification in classifications} == {
        PlanKey.from_plan_like(capability) for capability in metal_capabilities()
    }
    assert {classification.kernel.operation for classification in classifications} == {
        capability.operation for capability in metal_capabilities()
    }


def test_real_metal_specialization_binds_complete_v3_receipt() -> None:
    profile = verification_profile("metal-tilelang")
    logical_kernel = next(
        kernel for kernel in kernel_manifest(profile) if kernel.operation == "add"
    )
    lhs = _tensor((1.0, 2.0))
    rhs = _tensor((3.0, 4.0))

    with capture_metal_compilations() as compilations:
        result = sw.add(lhs, rhs)
        result.carrier.get_value(result.offset)

    assert len(compilations) == 1
    receipt = metal_compilation_receipt(logical_kernel, compilations[0])
    assert receipt.kind == "jit-specialization"
    assert receipt.logical_kernel == logical_kernel
    assert receipt.specialization_axes
    assert {item.artifact_kind for item in receipt.artifacts} >= {
        "generated-host-source",
        "generated-device-source",
    }
    assert any(item.input_kind == "source" for item in receipt.inputs)
    assert current_metal_compilation_bundle(profile, (receipt,)).receipts == (receipt,)


def test_metal_receipt_rejects_a_mismatched_logical_kernel() -> None:
    profile = verification_profile("metal-tilelang")
    add_kernel = next(
        kernel for kernel in kernel_manifest(profile) if kernel.operation == "add"
    )
    lhs = _tensor((1.0, 2.0, 3.0))
    rhs = _tensor((4.0, 5.0, 6.0))
    with capture_metal_compilations() as compilations:
        result = sw.add(lhs, rhs)
        result.carrier.get_value(result.offset)

    forged = replace(add_kernel, operation="sub")
    with pytest.raises(ValueError, match="does not match"):
        metal_compilation_receipt(forged, compilations[0])


def test_current_metal_bundle_fails_closed_when_recipe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = verification_profile("metal-tilelang")
    logical_kernel = next(
        kernel for kernel in kernel_manifest(profile) if kernel.operation == "add"
    )
    lhs = _tensor((1.0, 2.0))
    rhs = _tensor((3.0, 4.0))
    with capture_metal_compilations() as compilations:
        result = sw.add(lhs, rhs)
        result.carrier.get_value(result.offset)
    receipt = metal_compilation_receipt(logical_kernel, compilations[0])

    def unavailable(_key) -> None:
        raise RuntimeError("specialization recipe unavailable")

    monkeypatch.setattr(metal_jit, "recompile_specialization", unavailable)
    with pytest.raises(RuntimeError, match="recipe unavailable"):
        current_metal_compilation_bundle(profile, (receipt,))


def test_metal_stage_two_blocks_before_resolving_or_launching_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = run_oracle_stage(verification_profile("cpu-compiled"))
    incomplete = replace(oracle, certificates=())
    runtime_resolved = False

    def reject_runtime(_profile) -> None:
        nonlocal runtime_resolved
        runtime_resolved = True
        raise AssertionError("blocked Metal target resolved its runtime")

    monkeypatch.setattr(stage_two, "_target_runtime", reject_runtime)
    result = run_target_stage(verification_profile("metal-tilelang"), incomplete)

    assert not runtime_resolved
    assert result.records
    assert all(
        record.outcome in {VerificationOutcome.BLOCKED, VerificationOutcome.DEFERRED}
        for record in result.records
    )
    assert not result.receipts


def test_metal_case_error_preserves_later_independent_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = next(
        item
        for item in classify_profile_plans(verification_profile("metal-tilelang"))
        if item.kernel.operation == "add"
        and item.disposition is ClassificationDisposition.ACTIVE
    )
    oracle = run_oracle_stage(verification_profile("cpu-compiled"))

    def fail_exact(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected independent Metal case failure")

    monkeypatch.setattr(stage_two, "_exact_record", fail_exact)
    execution = stage_two._metal_target_records(
        descriptor,
        "0" * 64,
        oracle,
        stage_two._metal_synchronize,
    )

    assert any(
        record.outcome is VerificationOutcome.ERROR for record in execution.records
    )
    assert any(
        record.outcome is VerificationOutcome.PASSED for record in execution.records
    )
    assert execution.receipts


def test_pre_jit_error_never_binds_later_case_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exact = stage_two._exact_record
    injected = False

    def fail_first_add(descriptor, *args, **kwargs):
        nonlocal injected
        if descriptor.kernel.operation == "add" and not injected:
            injected = True
            raise RuntimeError("injected pre-JIT add failure")
        return original_exact(descriptor, *args, **kwargs)

    monkeypatch.setattr(stage_two, "_exact_record", fail_first_add)
    report = sw.verify_backend("metal-tilelang")
    assert report.header is not None
    error = next(
        record
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.operation == "add"
        and record.outcome is VerificationOutcome.ERROR
    )
    assert error.compilation_receipt_id is None
    assert not error.supporting_compilation_receipt_ids

    later_add_receipts = {
        record.compilation_receipt_id
        for record in report.records
        if record.stage is VerificationStage.TARGET
        and record.case.operation == "add"
        and record.outcome is VerificationOutcome.PASSED
    }
    assert None not in later_add_receipts
    assert len(later_add_receipts) > 1
    serialized = report.to_jsonl()
    assert VerificationReport.from_jsonl(serialized).to_jsonl() == serialized

    store = RecordingEvidenceStore()
    recording.record_report(report, store, producer_id="pre-jit-error-test")
    assert store.transactions


def test_metal_report_is_complete_deterministic_and_offline(
    monkeypatch: pytest.MonkeyPatch,
    current_metal_report: tuple[VerificationReport, Path],
) -> None:
    profile = verification_profile("metal-tilelang")
    report, output = current_metal_report
    serialized = report.to_jsonl()

    pending_specialization = False
    specialization_selections = 0
    synchronization_calls = 0
    original_publish = metal_jit._publish_compilation
    original_synchronize = stage_two._metal_synchronize
    original_stage_one_values = stage_one._values
    original_stage_two_values = stage_two._values

    def publish_before_launch(key, compiled) -> None:
        nonlocal pending_specialization, specialization_selections
        pending_specialization = True
        specialization_selections += 1
        original_publish(key, compiled)

    def synchronize_before_decode() -> None:
        nonlocal pending_specialization, synchronization_calls
        original_synchronize()
        pending_specialization = False
        synchronization_calls += 1

    def stage_one_values(tensor: sw.Tensor):
        if isinstance(tensor.carrier, sw.Metal):
            assert not pending_specialization, (
                "Metal result decoded before synchronization"
            )
        return original_stage_one_values(tensor)

    def stage_two_values(tensor: sw.Tensor):
        if isinstance(tensor.carrier, sw.Metal):
            assert not pending_specialization, (
                "Metal result decoded before synchronization"
            )
        return original_stage_two_values(tensor)

    monkeypatch.setattr(metal_jit, "_publish_compilation", publish_before_launch)
    monkeypatch.setattr(stage_two, "_metal_synchronize", synchronize_before_decode)
    monkeypatch.setattr(stage_one, "_values", stage_one_values)
    monkeypatch.setattr(stage_two, "_values", stage_two_values)
    repeated = sw.verify_backend(profile.profile_id)

    assert specialization_selections > 0
    assert synchronization_calls > 0
    assert not pending_specialization
    assert (
        hashlib.sha256(repeated.to_jsonl().encode()).digest()
        == hashlib.sha256(serialized.encode()).digest()
    )
    assert output.read_text(encoding="utf-8") == serialized

    target_records = tuple(
        record for record in report.records if record.stage is VerificationStage.TARGET
    )
    assert target_records
    assert {record.outcome for record in target_records} <= {
        VerificationOutcome.PASSED,
        VerificationOutcome.DEFERRED,
    }
    for descriptor in classify_profile_plans(profile):
        evidence = tuple(
            record
            for record in target_records
            if record.case.kernel_id == descriptor.kernel.kernel_id
            and record.case.variant == descriptor.kernel.variant
            and record.case.plan == descriptor.plan
        )
        assert evidence, descriptor
        if descriptor.disposition is ClassificationDisposition.DEFERRED:
            assert len(evidence) == 1
            assert evidence[0].outcome is VerificationOutcome.DEFERRED
        else:
            assert set(descriptor.classes) <= {record.test_class for record in evidence}

    movement = tuple(
        record
        for record in target_records
        if record.case.kernel_id.startswith("movement.")
    )
    assert {record.case.operation for record in movement} == {
        subject.operation for subject in profile_subjects(profile)
    }
    assert all(record.outcome is VerificationOutcome.PASSED for record in movement)

    assert report.header is not None
    receipts = {
        receipt.receipt_id: receipt
        for receipt in report.header.compilation_bundle.receipts
        if receipt.profile_id == profile.profile_id
    }
    executed = tuple(
        record
        for record in target_records
        if record.case.plan is not None
        and record.outcome is not VerificationOutcome.DEFERRED
    )
    assert executed
    assert all(record.compilation_receipt_id in receipts for record in executed)
    indexed = tuple(
        record
        for record in executed
        if record.case.operation in {"gather", "scatter", "scatter_add"}
    )
    assert indexed
    assert all(record.supporting_compilation_receipt_ids for record in indexed)
    assert all(
        receipts[receipt_id].logical_kernel.operation == "validate_indices"
        for record in indexed
        for receipt_id in record.supporting_compilation_receipt_ids
    )
    for receipt in receipts.values():
        assert receipt.kind == "jit-specialization"
        assert receipt.inputs
        assert receipt.declared_input_uris == tuple(item.uri for item in receipt.inputs)
        assert {item.artifact_kind for item in receipt.artifacts} >= {
            "generated-host-source",
            "generated-device-source",
        }

    specializations_by_kernel: dict[object, set[str]] = {}
    for receipt in receipts.values():
        specializations_by_kernel.setdefault(receipt.logical_kernel, set()).add(
            receipt.receipt_id
        )
    assert any(
        len(receipt_ids) > 1 for receipt_ids in specializations_by_kernel.values()
    )

    store = RecordingEvidenceStore()
    recorded = recording.record_report(report, store, producer_id="metal-test")
    assert recorded.evidence_count == len(report.records)
    assert store.tables()["compilation_receipts"] == len(
        report.header.compilation_bundle.receipts
    )

    selected = report.select(operation="add")
    assert selected.header is not None
    selected_receipts = tuple(
        receipt
        for receipt in selected.header.compilation_bundle.receipts
        if receipt.profile_id == profile.profile_id
    )
    original_recompile = metal_jit.recompile_specialization
    launches = 0

    def drift_without_launch(key):
        compiled = original_recompile(key)

        def reject_launch(*_args: object, **_kwargs: object) -> None:
            nonlocal launches
            launches += 1
            raise AssertionError("current-fact regeneration launched a kernel")

        return metal_jit.CompiledMetalKernel(
            reject_launch,
            replace(
                compiled.facts,
                host_source=compiled.facts.host_source + "\n// current drift",
            ),
        )

    with monkeypatch.context() as drift:
        drift.setattr(metal_jit, "recompile_specialization", drift_without_launch)
        current = current_metal_compilation_bundle(profile, selected_receipts)
        assert current.receipts != selected_receipts
        stale_store = RecordingEvidenceStore()
        with pytest.raises(VerificationStoreError, match="stale"):
            recording.record_report(
                selected,
                stale_store,
                producer_id="stale-metal-test",
            )
        assert stale_store.initialized == 0
        assert not stale_store.transactions
        assert launches == 0

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline loading consulted installed provider facts")

    monkeypatch.setattr(
        "strideweave.verification.reporting.installed_compilation_bundle",
        unavailable,
    )
    monkeypatch.setattr(
        "strideweave.verification.provenance._installed_manifest_value",
        unavailable,
    )
    monkeypatch.setattr(metal_jit, "recompile_specialization", unavailable)
    monkeypatch.setattr(
        "strideweave.carriers.metal._verification.load_metal_runtime",
        unavailable,
    )
    assert VerificationReport.from_jsonl(serialized).to_jsonl() == serialized


def test_malformed_content_addressed_metal_recipe_fails_before_store_mutation(
    current_metal_report: tuple[VerificationReport, Path],
) -> None:
    report, _output = current_metal_report
    assert report.header is not None
    target_receipt = next(
        receipt
        for receipt in report.header.compilation_bundle.receipts
        if isinstance(receipt, JITSpecializationReceipt)
        and receipt.profile_id == "metal-tilelang"
        and receipt.logical_kernel.operation == "add"
    )
    malformed_receipt = _receipt_with_malformed_axis(target_receipt)
    malformed_report = _report_with_replaced_receipt(
        report, target_receipt, malformed_receipt
    )
    profile = verification_profile("metal-tilelang")

    with pytest.raises(ValueError, match="canonical specialization value"):
        current_metal_compilation_bundle(profile, (malformed_receipt,))

    store = RecordingEvidenceStore()
    with pytest.raises(
        VerificationStoreError,
        match=r"could not resolve current compilation provenance.*canonical",
    ):
        recording.record_report(
            malformed_report,
            store,
            producer_id="malformed-metal-test",
        )
    assert store.initialized == 0
    assert not store.transactions


def test_every_plan_bearing_metal_recipe_rejects_each_semantic_plan_mutation(
    monkeypatch: pytest.MonkeyPatch,
    current_metal_report: tuple[VerificationReport, Path],
) -> None:
    report, _output = current_metal_report
    assert report.header is not None
    profile = verification_profile("metal-tilelang")
    receipts = tuple(
        receipt
        for receipt in report.header.compilation_bundle.receipts
        if isinstance(receipt, JITSpecializationReceipt)
        and receipt.profile_id == profile.profile_id
    )
    runtime_loads = 0

    def fail_runtime_load() -> object:
        nonlocal runtime_loads
        runtime_loads += 1
        raise AssertionError("inconsistent plan loaded the Metal runtime")

    monkeypatch.setattr(
        "strideweave.carriers.metal._verification.load_metal_runtime",
        fail_runtime_load,
    )
    covered_families: set[str] = set()
    covered_mutations: set[str] = set()
    example: tuple[JITSpecializationReceipt, JITSpecializationReceipt] | None = None
    for receipt in receipts:
        axes = {
            axis.name: metal_jit._decoded_value(
                cast(metal_jit.CanonicalValue, axis.value)
            )
            for axis in receipt.specialization_axes
        }
        plan_value = axes.get("plan")
        if type(plan_value) is not tuple or len(plan_value) != 6:
            continue
        plan = cast(tuple[object, ...], plan_value)
        metal_jit.validate_specialization_recipe(_key_from_receipt(receipt))
        covered_families.add(receipt.logical_kernel.kernel_id)
        for mutation, mutated_plan in _plan_mutations(plan):
            mutated = _receipt_with_axis_value(receipt, "plan", mutated_plan)
            with pytest.raises(ValueError, match="plan"):
                current_metal_compilation_bundle(profile, (mutated,))
            covered_mutations.add(mutation)
            if (
                example is None
                and receipt.logical_kernel.operation == "add"
                and mutation == "compute"
            ):
                example = receipt, mutated

    assert covered_families == {
        "metal.conv_general",
        "metal.indexing",
        "metal.matmul",
        "metal.pointwise",
        "metal.reduction",
        "metal.scan",
        "metal.selection",
    }
    assert covered_mutations == {
        "accumulation",
        "accumulator dtype",
        "compute",
        "operand conversion",
        "operand dtype",
        "operand role",
        "operation",
        "output",
    }
    assert example is not None
    original, mutated = example
    original_axes = {
        axis.name: metal_jit._decoded_value(cast(metal_jit.CanonicalValue, axis.value))
        for axis in original.specialization_axes
    }
    mutated_axes = {
        axis.name: metal_jit._decoded_value(cast(metal_jit.CanonicalValue, axis.value))
        for axis in mutated.specialization_axes
    }
    assert cast(tuple[object, ...], original_axes["plan"])[2] == "binary32"
    assert cast(tuple[object, ...], mutated_axes["plan"])[2] == "int32_exact"
    assert original.receipt_id != mutated.receipt_id
    with pytest.raises(ValueError, match="current executable operation plan"):
        metal_jit.validate_specialization_recipe(_key_from_receipt(mutated))

    malformed_report = _report_with_replaced_receipt(report, original, mutated)
    store = RecordingEvidenceStore()
    with pytest.raises(
        VerificationStoreError,
        match=r"could not resolve current compilation provenance.*current executable",
    ):
        recording.record_report(
            malformed_report,
            store,
            producer_id="mismatched-metal-plan-test",
        )
    assert store.initialized == 0
    assert not store.transactions
    assert runtime_loads == 0


def test_every_generated_metal_recipe_variant_rejects_wrong_tensor_cardinality(
    monkeypatch: pytest.MonkeyPatch,
    current_metal_report: tuple[VerificationReport, Path],
) -> None:
    report, _output = current_metal_report
    assert report.header is not None
    receipts = tuple(
        receipt
        for receipt in report.header.compilation_bundle.receipts
        if isinstance(receipt, JITSpecializationReceipt)
        and receipt.profile_id == "metal-tilelang"
    )
    assert receipts
    runtime_loads = 0

    def fail_runtime_load() -> object:
        nonlocal runtime_loads
        runtime_loads += 1
        raise AssertionError("inconsistent recipe loaded the Metal runtime")

    monkeypatch.setattr(
        "strideweave.carriers.metal._verification.load_metal_runtime",
        fail_runtime_load,
    )
    covered_variants: set[tuple[str, str]] = set()
    for receipt in receipts:
        axes = {
            axis.name: metal_jit._decoded_value(
                cast(metal_jit.CanonicalValue, axis.value)
            )
            for axis in receipt.specialization_axes
        }
        if "address_plans" in axes:
            plans = cast(tuple[object, ...], axes["address_plans"])
            index = next(
                index
                for index, plan in enumerate(plans)
                if plan != ("strideweave.metal.scalar-address.v1",)
            )
            count = len(address_plan_from_key(plans[index]).addresses)
            replacement = sw.Layout(sw.Shape(count + 1), sw.Stride(0))
            plans = (
                *plans[:index],
                address_plan(replacement).key,
                *plans[index + 1 :],
            )
            axis_name, axis_value = "address_plans", plans
        else:
            axis_name = next(
                name
                for name in (
                    "first_address_plan",
                    "update_address_plan",
                    "source_address_plan",
                    "gradient_address_plan",
                    "address_plan",
                    "output_address_plan",
                )
                if name in axes
            )
            original_plan = address_plan_from_key(axes[axis_name])
            replacement = sw.Layout(
                sw.Shape(len(original_plan.addresses) + 1), sw.Stride(0)
            )
            axis_value = address_plan(replacement).key
        mutated = _receipt_with_axis_value(receipt, axis_name, axis_value)
        with pytest.raises((RuntimeError, ValueError), match="cardinality"):
            current_metal_compilation_bundle(
                verification_profile("metal-tilelang"),
                (mutated,),
            )
        covered_variants.add(
            (receipt.logical_kernel.operation, receipt.logical_kernel.variant)
        )

    assert len(covered_variants) >= 10
    output_receipt = next(
        receipt
        for receipt in receipts
        if any(
            axis.name == "output_address_plan"
            and len(
                address_plan_from_key(
                    metal_jit._decoded_value(cast(metal_jit.CanonicalValue, axis.value))
                ).addresses
            )
            > 1
            for axis in receipt.specialization_axes
        )
    )
    output_axis = next(
        axis
        for axis in output_receipt.specialization_axes
        if axis.name == "output_address_plan"
    )
    output_plan = address_plan_from_key(
        metal_jit._decoded_value(cast(metal_jit.CanonicalValue, output_axis.value))
    )
    aliasing_output = address_plan(
        sw.Layout(sw.Shape(len(output_plan.addresses)), sw.Stride(0))
    ).key
    mutated_output = _receipt_with_axis_value(
        output_receipt, "output_address_plan", aliasing_output
    )
    with pytest.raises(ValueError, match="must be injective"):
        current_metal_compilation_bundle(
            verification_profile("metal-tilelang"),
            (mutated_output,),
        )
    assert runtime_loads == 0
