"""Deterministic pure-Python verification facts for real-Dolt store tests.

Persistence, query, publication, and refresh behavior is about what the store
does with already-validated evidence, not about how that evidence was produced.
The facts below are therefore built entirely in Python: no native extension, no
installed compilation manifest, and no report binding against this build. That
keeps every native-facing path — ``sw.test_backend``, native kernel metadata,
``load_compilation_manifest``, and ``bind_report`` — inside the unmarked suite
the sanitizer job runs, while the marked suites still persist and read a
complete, schema-valid provenance graph.

The facts are shaped like a real report rather than reduced: two kernels owned
by two sources, closures carrying both project-owned and external members, an
oracle reference, tolerance policies, deferred coverage, and a resolved plan.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import strideweave.verification.provenance as provenance
import strideweave.verification.reporting as reporting
from strideweave.verification import (
    VerificationOutcome,
    VerificationReport,
)
from strideweave.verification.model import (
    CaseDescriptor,
    Deviations,
    EvidenceRecord,
    KernelDescriptor,
    PlanKey,
    Tolerance,
    VerificationClass,
    VerificationStage,
)

ARCHITECTURE = "synthetic-arch64"
"""Installed-manifest architecture token these facts describe."""

PROVIDER_KIND = "synthetic-cpu"

_SOURCE_NAMES = ("synthetic_alpha", "synthetic_beta")


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _identified(identity: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Return one content-addressed descriptor keyed by its own digest."""

    return {field: _digest(dict(identity)), **identity}


def _target() -> dict[str, Any]:
    return _identified(
        {
            "abi": "synthetic-abi",
            "architecture": ARCHITECTURE,
            "endianness": "little",
            "operating_system": "synthetic-os",
            "pointer_bits": 64,
            "vendor": "synthetic-vendor",
        },
        "target_id",
    )


def _toolchain() -> dict[str, Any]:
    return _identified(
        {
            "build_system": "synthetic-build",
            "compiler_id": "SyntheticCompiler",
            "compiler_version": "1.2.3",
            "provider_kind": PROVIDER_KIND,
            "target_triple": "synthetic-arch64-unknown-synthetic-os",
        },
        "toolchain_id",
    )


def _source_inputs(name: str) -> list[dict[str, str]]:
    """Return one closure's complete, sorted, mixed-ownership input sequence."""

    external = _content_digest(f"external:{name}")
    inputs = [
        {
            "content_digest": _content_digest(f"build-input:{name}"),
            "input_kind": "build_input",
            "uri": "CMakeLists.txt",
        },
        {
            "content_digest": _content_digest(f"generated:{name}"),
            "input_kind": "generated_header",
            "uri": f"build/{name}_dispatch.hpp",
        },
        {
            "content_digest": external,
            "input_kind": "external_header",
            "uri": f"cpp-external://sha256/{external}/vector",
        },
        {
            "content_digest": _content_digest(f"header:{name}"),
            "input_kind": "header",
            "uri": f"src/strideweave/{name}.hpp",
        },
        {
            "content_digest": _content_digest(f"source:{name}"),
            "input_kind": "source",
            "uri": f"src/strideweave/{name}.cpp",
        },
    ]
    return sorted(inputs, key=lambda item: item["uri"])


def _source(name: str) -> dict[str, Any]:
    owning_source = f"src/strideweave/{name}.cpp"
    inputs = _source_inputs(name)
    invocation = ["c++", "-O2", "-std=c++20", "-c", owning_source]
    closure_value = {
        "inputs": [
            {
                "content_digest": item["content_digest"],
                "input_kind": item["input_kind"],
                "uri": item["uri"],
            }
            for item in inputs
        ],
        "invocation": invocation,
    }
    return {
        "closure_id": _digest(closure_value),
        "compile_invocation": invocation,
        "compile_invocation_digest": _digest(invocation),
        "inputs": inputs,
        "object_digest": _content_digest(f"object:{name}"),
        "source": owning_source,
    }


def _manifest() -> dict[str, Any]:
    return _identified(
        {
            "artifact_digest": _content_digest("artifact:synthetic"),
            "provider_kind": PROVIDER_KIND,
            "receipt_schema": provenance._RECEIPT_SCHEMA,
            "schema": provenance._MANIFEST_SCHEMA,
            "sources": [_source(name) for name in _SOURCE_NAMES],
            "target": _target(),
            "toolchain": _toolchain(),
        },
        "manifest_digest",
    )


def _kernels() -> tuple[KernelDescriptor, ...]:
    return tuple(
        KernelDescriptor(
            operation=name.removeprefix("synthetic_"),
            kernel_id=f"cpu.{name}",
            variant="default",
            pybind_name=f"_{name}",
            owning_source=f"src/strideweave/{name}.cpp",
        )
        for name in _SOURCE_NAMES
    )


def _oracle_reference() -> dict[str, Any]:
    inputs = [
        {
            "content_digest": _content_digest(f"oracle:{uri}"),
            "uri": uri,
        }
        for uri in ("src/strideweave/synthetic_oracle.py",)
    ]
    return _identified(
        {
            "implementation_digest": _digest(inputs),
            "inputs": inputs,
            "oracle_kind": "synthetic-reference",
            "oracle_schema": reporting._ORACLE_SCHEMA,
        },
        "oracle_reference_id",
    )


def _plan(operation: str) -> PlanKey:
    return PlanKey(
        operation=operation,
        operands=(("LEFT", "Float32", None), ("RIGHT", "Float32", None)),
        compute="Float32",
        accumulation=None,
        accumulator_dtype=None,
        output="Float32",
    )


def _case(
    kernel: KernelDescriptor, label: str, *, with_plan: bool = True
) -> CaseDescriptor:
    return CaseDescriptor(
        operation=kernel.operation,
        kernel_id=kernel.kernel_id,
        variant=kernel.variant,
        input_dtypes=("Float32", "Float32"),
        output_dtype="Float32",
        shapes=((2, 3), (2, 3)),
        accumulator_dtype=None,
        contraction_length=None,
        seed=7,
        case_id=f"{kernel.kernel_id}/{label}",
        plan=_plan(kernel.operation) if with_plan else None,
    )


def _record(
    kernel: KernelDescriptor,
    label: str,
    *,
    stage: VerificationStage,
    test_class: VerificationClass,
    outcome: VerificationOutcome,
    with_plan: bool = True,
) -> EvidenceRecord:
    hashes = (_content_digest(f"input:{kernel.kernel_id}:{label}"),)
    return EvidenceRecord(
        stage=stage,
        test_class=test_class,
        case=_case(kernel, label, with_plan=with_plan),
        target_input_bit_hashes=hashes,
        oracle_input_bit_hashes=hashes,
        tolerance=Tolerance(absolute=0.0, relative=0.0, ulps=0, version="exact-v1"),
        deviations=Deviations(
            maximum_absolute=0.0, maximum_relative=0.0, maximum_ulps=0
        ),
        mismatches=0,
        outcome=outcome,
        diagnostic=None
        if outcome is not VerificationOutcome.DEFERRED
        else "declared deferred coverage",
    )


def _raw_records(kernels: Sequence[KernelDescriptor]) -> tuple[EvidenceRecord, ...]:
    first, second = kernels
    # Ordered by case_id, which is the canonical JSONL order, so a report and
    # its own serialized form hold their evidence in the same sequence.
    return (
        _record(
            first,
            "bit-exact",
            stage=VerificationStage.ORACLE,
            test_class=VerificationClass.BIT_EXACT,
            outcome=VerificationOutcome.PASSED,
            with_plan=False,
        ),
        _record(
            first,
            "numerical",
            stage=VerificationStage.TARGET,
            test_class=VerificationClass.NUMERICAL,
            outcome=VerificationOutcome.PASSED,
        ),
        _record(
            second,
            "deferred",
            stage=VerificationStage.TARGET,
            test_class=VerificationClass.DEFERRED,
            outcome=VerificationOutcome.DEFERRED,
        ),
        _record(
            second,
            "structural",
            stage=VerificationStage.TARGET,
            test_class=VerificationClass.STRUCTURAL,
            outcome=VerificationOutcome.PASSED,
        ),
    )


def _bind(
    records: Iterable[EvidenceRecord],
    *,
    compilation: Mapping[str, Any],
    receipt_ids: Mapping[tuple[str, str], str],
    oracle: Mapping[str, Any],
) -> VerificationReport:
    """Bind raw records to these synthetic facts the way a real report is bound.

    This mirrors the shape of a bound report rather than calling the production
    binder, which would reconcile against the installed native build.
    """

    compilation_value = reporting._thaw(compilation)
    oracle_value = reporting._thaw(oracle)
    requirements: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    bound: list[EvidenceRecord] = []
    for record in records:
        requirement = reporting._requirement_value(record)
        tolerance = reporting._tolerance_value(record)
        requirements[requirement["requirement_id"]] = requirement
        policies[tolerance["tolerance_policy_id"]] = tolerance
        bound.append(
            replace(
                record,
                requirement_id=requirement["requirement_id"],
                compilation_receipt_id=receipt_ids[
                    (record.case.kernel_id, record.case.variant)
                ],
                tolerance_policy_id=tolerance["tolerance_policy_id"],
                oracle_reference_id=oracle_value["oracle_reference_id"],
            )
        )
    spec_value = {
        "requirements": [requirements[key] for key in sorted(requirements)],
        "spec_schema": reporting._SPEC_SCHEMA,
    }
    header_value: dict[str, Any] = {
        "certificates": [],
        "compilation": compilation_value,
        "evidence_schema": reporting._EVIDENCE_SCHEMA,
        "oracle_references": [oracle_value],
        "schema_version": reporting._REPORT_SCHEMA,
        "tolerance_policies": [policies[key] for key in sorted(policies)],
        "verification_spec": _identified(spec_value, "verification_spec_id"),
    }
    header_value["header_digest"] = _digest(header_value)
    header = reporting.parse_report_header(header_value)
    return VerificationReport(tuple(bound), header.schema_version, header)


def synthetic_report() -> VerificationReport:
    """Return one deterministic, provenance-complete report built in Python.

    Returns:
        A report whose header, compilation manifest, specification, tolerance
        policies, and oracle reference are internally consistent and depend on
        no installed native build.
    """

    manifest = _manifest()
    kernels = _kernels()
    parsed = provenance.parse_compilation_manifest(manifest, kernels=kernels)
    receipt_ids = {
        (receipt.kernel.kernel_id, receipt.kernel.variant): receipt.receipt_id
        for receipt in parsed.receipts
    }
    compilation = {
        "kernel_receipts": [
            {
                "kernel": reporting._kernel_value(receipt.kernel),
                "receipt_id": receipt.receipt_id,
            }
            for receipt in parsed.receipts
        ],
        "manifest": manifest,
    }
    oracle = _oracle_reference()
    return _bind(
        _raw_records(kernels),
        compilation=compilation,
        receipt_ids=receipt_ids,
        oracle=oracle,
    )


def contradicting_report(report: VerificationReport) -> VerificationReport:
    """Return the same facts with one independently observed failing outcome.

    Args:
        report: Report whose first record is replaced by a failing observation.

    Returns:
        A rebound report that disagrees with ``report`` on exactly one case.
    """

    assert report.header is not None
    changed = (
        replace(
            report.records[0],
            outcome=VerificationOutcome.FAILED,
            diagnostic="independent producer observed a mismatch",
        ),
        *report.records[1:],
    )
    compilation = report.header.compilation
    receipt_ids = {
        (item["kernel"]["kernel_id"], item["kernel"]["variant"]): item["receipt_id"]
        for item in compilation["kernel_receipts"]
    }
    return _bind(
        changed,
        compilation=compilation,
        receipt_ids=receipt_ids,
        oracle=report.header.oracle_references[0],
    )
