"""Provenance binding and strict report-header validation."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from importlib import resources
from types import MappingProxyType
from typing import Any

from .model import (
    EvidenceRecord,
    KernelDescriptor,
    OracleCertificate,
    PlanKey,
    ReportHeader,
    VerificationClass,
    VerificationReport,
    VerificationStage,
    _parse_plan,
)
from .provenance import load_compilation_manifest, parse_compilation_manifest

_REPORT_SCHEMA = "strideweave.kernel-verification.v2"
_EVIDENCE_SCHEMA = "strideweave.kernel-evidence.v2"
_SPEC_SCHEMA = "strideweave.kernel-verification-spec.v1"
_ORACLE_SCHEMA = "strideweave.kernel-oracle-reference.v1"
_TOLERANCE_SCHEMA = "strideweave.kernel-tolerance-policy.v1"
_GENERIC_ORACLE_ROOTS = (
    "carriers/generic/execution.py",
    "carriers/generic/ops.py",
    "carriers/generic/reduction_ops.py",
    "verification/comparison.py",
    "verification/payloads.py",
    "verification/stage_one.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_object(value: Any, field: str, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    missing = sorted(fields - value.keys())
    unexpected = sorted(value.keys() - fields)
    if missing or unexpected:
        raise ValueError(
            f"{field} fields do not match: missing={missing!r}, unexpected={unexpected!r}"
        )
    return value


def _require_digest(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _raw_compilation_manifest() -> dict[str, Any]:
    resource = resources.files("strideweave.verification").joinpath(
        "_native_provenance.json"
    )
    value = json.loads(resource.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("installed compilation manifest must be an object")
    return value


def _kernel_value(kernel: KernelDescriptor) -> dict[str, str]:
    return {
        "kernel_id": kernel.kernel_id,
        "operation": kernel.operation,
        "owning_source": kernel.owning_source,
        "pybind_name": kernel.pybind_name,
        "variant": kernel.variant,
    }


def _requirement_value(record: EvidenceRecord) -> dict[str, Any]:
    case = record.as_json_object()["case"]
    value = {
        "case": case,
        "stage": record.stage.value,
        "test_class": record.test_class.value,
    }
    return {"requirement_id": _digest(value), **value}


def _tolerance_value(record: EvidenceRecord) -> dict[str, Any]:
    definition = {
        "absolute": float(record.tolerance.absolute),
        "relative": float(record.tolerance.relative),
        "ulps": record.tolerance.ulps,
        "version": record.tolerance.version,
    }
    value = {
        "definition": definition,
        "policy_schema": _TOLERANCE_SCHEMA,
    }
    return {"tolerance_policy_id": _digest(value), **value}


def _module_name(uri: str) -> tuple[str, bool]:
    if uri == "__init__.py":
        return "strideweave", True
    if uri.endswith("/__init__.py"):
        return f"strideweave.{uri.removesuffix('/__init__.py').replace('/', '.')}", True
    return f"strideweave.{uri.removesuffix('.py').replace('/', '.')}", False


def _module_resource(package: Any, module: str) -> tuple[str, Any, bool] | None:
    if module == "strideweave":
        candidates = (("__init__.py", True),)
    elif module.startswith("strideweave."):
        relative = module.removeprefix("strideweave.").replace(".", "/")
        candidates = ((f"{relative}.py", False), (f"{relative}/__init__.py", True))
    else:
        return None
    for uri, is_package in candidates:
        resource = package.joinpath(*uri.split("/"))
        if resource.is_file():
            return uri, resource, is_package
    return None


def _imported_modules(source: str, module: str, *, is_package: bool) -> set[str]:
    imported: set[str] = set()
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                try:
                    base = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError):
                    continue
            else:
                base = node.module or ""
            if base:
                imported.add(base)
                imported.update(f"{base}.{alias.name}" for alias in node.names)
    return {
        name
        for name in imported
        if name == "strideweave" or name.startswith("strideweave.")
    }


def _generic_oracle_input_uris(package: Any) -> tuple[str, ...]:
    """Return the reviewed Generic roots and their local static import closure."""

    pending = [_module_name(uri)[0] for uri in _GENERIC_ORACLE_ROOTS]
    discovered: dict[str, str] = {}
    while pending:
        module = pending.pop()
        if module in discovered:
            continue
        resolved = _module_resource(package, module)
        if resolved is None:
            continue
        uri, resource, is_package = resolved
        source = resource.read_text(encoding="utf-8")
        discovered[module] = uri
        pending.extend(
            sorted(
                _imported_modules(source, module, is_package=is_package)
                - discovered.keys(),
                reverse=True,
            )
        )
    return tuple(sorted(discovered.values()))


def _generic_oracle_value() -> dict[str, Any]:
    package = resources.files("strideweave")
    inputs = []
    for uri in _generic_oracle_input_uris(package):
        data = package.joinpath(uri).read_bytes()
        inputs.append(
            {
                "content_digest": hashlib.sha256(data).hexdigest(),
                "uri": f"src/strideweave/{uri}",
            }
        )
    closure_id = _digest(inputs)
    value = {
        "implementation_digest": closure_id,
        "inputs": inputs,
        "oracle_kind": "generic-reference",
        "oracle_schema": _ORACLE_SCHEMA,
    }
    return {"oracle_reference_id": _digest(value), **value}


def _plan_value(plan: PlanKey | None) -> object:
    return asdict(plan) if plan is not None else None


def certificate_value(certificate: OracleCertificate) -> dict[str, Any]:
    """Return one immutable, digest-backed oracle certificate fact."""

    value = {
        "certified_classes": [item.value for item in certificate.certified_classes],
        "certified_plan_classes": [
            {
                "classes": [item.value for item in classes],
                "plan": _plan_value(plan),
            }
            for plan, classes in certificate.certified_plan_classes
        ],
        "evidence_digest": certificate.evidence_digest,
        "kernel_id": certificate.kernel_id,
        "variant": certificate.variant,
    }
    return {"certificate_digest": _digest(value), **value}


def _certificate_from_value(value: Any, field: str) -> OracleCertificate:
    data = _require_object(
        value,
        field,
        frozenset(
            {
                "certificate_digest",
                "certified_classes",
                "certified_plan_classes",
                "evidence_digest",
                "kernel_id",
                "variant",
            }
        ),
    )
    for name in ("kernel_id", "variant"):
        if type(data[name]) is not str or not data[name]:
            raise ValueError(f"{field}.{name} must be a non-empty string")
    _require_digest(data["evidence_digest"], f"{field}.evidence_digest")
    class_values = data["certified_classes"]
    if type(class_values) is not list:
        raise ValueError(f"{field}.certified_classes must be an array")
    try:
        certified_classes = tuple(
            VerificationClass(item) if type(item) is str else VerificationClass("")
            for item in class_values
        )
    except ValueError as error:
        raise ValueError(
            f"{field}.certified_classes contains an invalid class"
        ) from error
    if len(set(certified_classes)) != len(certified_classes):
        raise ValueError(f"{field}.certified_classes contains a duplicate")
    plan_values = data["certified_plan_classes"]
    if type(plan_values) is not list:
        raise ValueError(f"{field}.certified_plan_classes must be an array")
    certified_plan_classes = []
    seen_plans = set()
    for index, item in enumerate(plan_values):
        item_field = f"{field}.certified_plan_classes[{index}]"
        plan_data = _require_object(item, item_field, frozenset({"classes", "plan"}))
        classes_value = plan_data["classes"]
        if type(classes_value) is not list:
            raise ValueError(f"{item_field}.classes must be an array")
        try:
            classes = tuple(
                VerificationClass(item) if type(item) is str else VerificationClass("")
                for item in classes_value
            )
        except ValueError as error:
            raise ValueError(
                f"{item_field}.classes contains an invalid class"
            ) from error
        if len(set(classes)) != len(classes):
            raise ValueError(f"{item_field}.classes contains a duplicate")
        plan = _parse_plan(_thaw(plan_data["plan"]), f"{item_field}.plan")
        if plan in seen_plans:
            raise ValueError(
                f"{field}.certified_plan_classes contains a duplicate plan"
            )
        seen_plans.add(plan)
        certified_plan_classes.append((plan, classes))
    certificate = OracleCertificate(
        data["kernel_id"],
        data["variant"],
        certified_classes,
        data["evidence_digest"],
        tuple(certified_plan_classes),
    )
    if _canonical_json(certificate_value(certificate)) != _canonical_json(data):
        raise ValueError(f"{field} identity does not match")
    return certificate


def _stage_one_requirements(
    records: Sequence[EvidenceRecord],
) -> tuple[tuple[PlanKey, tuple[VerificationClass, ...]], ...]:
    requirements: list[tuple[PlanKey, list[VerificationClass]]] = []
    positions: dict[PlanKey, int] = {}
    for record in records:
        if record.test_class is VerificationClass.DEFERRED:
            continue
        plan = record.case.plan
        if plan is None:
            raise ValueError("certified Stage One evidence has no capability plan")
        position = positions.get(plan)
        if position is None:
            position = len(requirements)
            positions[plan] = position
            requirements.append((plan, []))
        classes = requirements[position][1]
        if record.test_class not in classes:
            classes.append(record.test_class)
    result = tuple(
        (
            plan,
            tuple(
                test_class for test_class in VerificationClass if test_class in classes
            ),
        )
        for plan, classes in requirements
    )
    return tuple(sorted(result, key=lambda item: _canonical_json(_plan_value(item[0]))))


def _unique_requirement_classes(
    requirements: Sequence[tuple[PlanKey, tuple[VerificationClass, ...]]],
) -> tuple[VerificationClass, ...]:
    result: list[VerificationClass] = []
    for _, classes in requirements:
        for test_class in classes:
            if test_class not in result:
                result.append(test_class)
    return tuple(result)


def _validate_certificate_evidence(
    certificates: Sequence[OracleCertificate], records: Sequence[EvidenceRecord]
) -> None:
    stage_one_records = tuple(
        record for record in records if record.stage is VerificationStage.ORACLE
    )
    for certificate in certificates:
        matching = tuple(
            record
            for record in stage_one_records
            if (record.case.kernel_id, record.case.variant)
            == (certificate.kernel_id, certificate.variant)
        )
        if not matching:
            # A filtered report retains every certificate consumed by its target
            # rows, but may omit that certificate's Stage One evidence even when
            # unrelated Stage One rows remain in the same view. Validation is
            # therefore local to one certificate's kernel and variant.
            continue
        observed_requirements = _stage_one_requirements(matching)
        observed_by_plan = dict(observed_requirements)
        claimed_by_plan = dict(certificate.certified_plan_classes)
        if any(
            plan not in claimed_by_plan
            or not set(classes).issubset(claimed_by_plan[plan])
            for plan, classes in observed_requirements
        ):
            raise ValueError(
                "Stage One certificate plan/class coverage disagrees with report evidence"
            )
        complete = all(
            plan in observed_by_plan and set(classes).issubset(observed_by_plan[plan])
            for plan, classes in certificate.certified_plan_classes
        )
        if not complete:
            # Class- or plan-filtered reports carry only part of an already
            # validated source certificate's evidence and cannot recompute its
            # evidence digest locally.
            continue
        kernel = KernelDescriptor(
            matching[0].case.operation,
            certificate.kernel_id,
            certificate.variant,
            "embedded-report",
        )
        try:
            reconstructed = OracleCertificate.from_records(
                kernel,
                _unique_requirement_classes(observed_requirements),
                matching,
                required_plan_classes=observed_requirements,
            )
        except ValueError as error:
            raise ValueError(
                f"Stage One certificate disagrees with report evidence: {error}"
            ) from error
        if reconstructed != certificate:
            raise ValueError("Stage One certificate disagrees with report evidence")


def _compilation_value() -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    manifest = load_compilation_manifest()
    kernel_receipts = [
        {
            "kernel": _kernel_value(receipt.kernel),
            "receipt_id": receipt.receipt_id,
        }
        for receipt in manifest.receipts
    ]
    value = {
        "kernel_receipts": kernel_receipts,
        "manifest": _raw_compilation_manifest(),
    }
    receipt_ids = {
        (receipt.kernel.kernel_id, receipt.kernel.variant): receipt.receipt_id
        for receipt in manifest.receipts
    }
    return value, receipt_ids


def bind_report(
    records: Sequence[EvidenceRecord],
    certificates: Sequence[OracleCertificate],
    *,
    certificate_facts_override: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[tuple[EvidenceRecord, ...], ReportHeader]:
    """Bind raw records to current immutable compilation and oracle facts."""

    compilation, receipt_ids = _compilation_value()
    requirements_by_id = {
        value["requirement_id"]: value
        for value in (_requirement_value(record) for record in records)
    }
    requirements = [requirements_by_id[key] for key in sorted(requirements_by_id)]
    spec_value = {"requirements": requirements, "spec_schema": _SPEC_SCHEMA}
    verification_spec = {
        "verification_spec_id": _digest(spec_value),
        **spec_value,
    }
    tolerance_by_id = {
        value["tolerance_policy_id"]: value
        for value in (_tolerance_value(record) for record in records)
    }
    tolerance_policies = [tolerance_by_id[key] for key in sorted(tolerance_by_id)]
    oracle = _generic_oracle_value()
    certificate_facts = sorted(
        (
            [_thaw(value) for value in certificate_facts_override]
            if certificate_facts_override is not None
            else [certificate_value(certificate) for certificate in certificates]
        ),
        key=lambda value: value["certificate_digest"],
    )
    certificate_by_kernel = {
        (value["kernel_id"], value["variant"]): value["certificate_digest"]
        for value in certificate_facts
    }
    enriched = []
    for record in records:
        requirement = _requirement_value(record)
        tolerance = _tolerance_value(record)
        kernel_key = (record.case.kernel_id, record.case.variant)
        consumed = None
        if record.stage is VerificationStage.TARGET and record.case.kernel_id in {
            "cpu.reduce_sum",
            "cpu.matmul",
        }:
            consumed = certificate_by_kernel.get(kernel_key)
        enriched.append(
            replace(
                record,
                requirement_id=requirement["requirement_id"],
                compilation_receipt_id=receipt_ids.get(kernel_key),
                tolerance_policy_id=tolerance["tolerance_policy_id"],
                oracle_reference_id=oracle["oracle_reference_id"],
                consumed_certificate_digest=consumed,
            )
        )
    header_value = {
        "certificates": certificate_facts,
        "compilation": compilation,
        "evidence_schema": _EVIDENCE_SCHEMA,
        "oracle_references": [oracle],
        "schema_version": _REPORT_SCHEMA,
        "tolerance_policies": tolerance_policies,
        "verification_spec": verification_spec,
    }
    header_value["header_digest"] = _digest(header_value)
    header = parse_report_header(header_value)
    validate_report(header, tuple(enriched))
    return tuple(enriched), header


def make_verification_report(
    records: Sequence[EvidenceRecord], certificates: Sequence[OracleCertificate] = ()
) -> VerificationReport:
    """Construct a provenance-complete report from raw evidence and certificates."""

    enriched, header = bind_report(records, certificates)
    return VerificationReport(enriched, header.schema_version, header)


def report_header_json_object(header: ReportHeader) -> dict[str, Any]:
    """Return a mutable canonical JSON object for one immutable header."""

    return {
        "certificates": _thaw(header.certificates),
        "compilation": _thaw(header.compilation),
        "evidence_schema": _EVIDENCE_SCHEMA,
        "header_digest": header.header_digest,
        "oracle_references": _thaw(header.oracle_references),
        "schema_version": header.schema_version,
        "tolerance_policies": _thaw(header.tolerance_policies),
        "verification_spec": _thaw(header.verification_spec),
    }


def _kernel_from_value(value: Any, field: str) -> KernelDescriptor:
    data = _require_object(
        value,
        field,
        frozenset(
            {"kernel_id", "operation", "owning_source", "pybind_name", "variant"}
        ),
    )
    if any(type(data[key]) is not str or not data[key] for key in data):
        raise ValueError(f"{field} fields must be non-empty strings")
    return KernelDescriptor(
        data["operation"],
        data["kernel_id"],
        data["variant"],
        data["pybind_name"],
        data["owning_source"],
    )


def parse_report_header(value: Any) -> ReportHeader:
    """Strictly parse one self-contained report header without installed provenance."""

    fields = frozenset(
        {
            "certificates",
            "compilation",
            "evidence_schema",
            "header_digest",
            "oracle_references",
            "schema_version",
            "tolerance_policies",
            "verification_spec",
        }
    )
    data = _require_object(value, "report header", fields)
    if data["schema_version"] != _REPORT_SCHEMA:
        raise ValueError(
            f"unsupported report schema version {data['schema_version']!r}"
        )
    if data["evidence_schema"] != _EVIDENCE_SCHEMA:
        raise ValueError(
            f"unsupported evidence schema version {data['evidence_schema']!r}"
        )
    identity = {key: data[key] for key in fields - {"header_digest"}}
    if _digest(identity) != data["header_digest"]:
        raise ValueError("report header digest does not match its contents")
    compilation = _require_object(
        data["compilation"],
        "report header.compilation",
        frozenset({"kernel_receipts", "manifest"}),
    )
    receipt_values = compilation["kernel_receipts"]
    if type(receipt_values) is not list:
        raise ValueError("report header.compilation.kernel_receipts must be an array")
    kernels = []
    expected_receipts = {}
    for index, raw_receipt in enumerate(receipt_values):
        field = f"report header.compilation.kernel_receipts[{index}]"
        receipt = _require_object(
            raw_receipt, field, frozenset({"kernel", "receipt_id"})
        )
        kernel = _kernel_from_value(receipt["kernel"], f"{field}.kernel")
        key = (kernel.kernel_id, kernel.variant)
        if key in expected_receipts:
            raise ValueError("report header contains a duplicate kernel receipt")
        kernels.append(kernel)
        expected_receipts[key] = _require_digest(
            receipt["receipt_id"], f"{field}.receipt_id"
        )
    parsed_compilation = parse_compilation_manifest(
        compilation["manifest"], kernels=tuple(kernels)
    )
    observed_receipts = {
        (receipt.kernel.kernel_id, receipt.kernel.variant): receipt.receipt_id
        for receipt in parsed_compilation.receipts
    }
    if observed_receipts != expected_receipts:
        raise ValueError("report header compilation receipt identities do not match")
    verification_spec = _require_object(
        data["verification_spec"],
        "report header.verification_spec",
        frozenset({"requirements", "spec_schema", "verification_spec_id"}),
    )
    if verification_spec["spec_schema"] != _SPEC_SCHEMA:
        raise ValueError("report header has an unsupported verification specification")
    spec_identity = {
        key: verification_spec[key] for key in ("requirements", "spec_schema")
    }
    if _digest(spec_identity) != verification_spec["verification_spec_id"]:
        raise ValueError("verification specification identity does not match")
    for collection_name, id_name in (
        ("tolerance_policies", "tolerance_policy_id"),
        ("oracle_references", "oracle_reference_id"),
        ("certificates", "certificate_digest"),
    ):
        collection = data[collection_name]
        if type(collection) is not list:
            raise ValueError(f"report header.{collection_name} must be an array")
        seen = set()
        for index, item in enumerate(collection):
            if type(item) is not dict or id_name not in item:
                raise ValueError(
                    f"report header.{collection_name}[{index}] must contain {id_name}"
                )
            identifier = _require_digest(
                item[id_name], f"report header.{collection_name}[{index}].{id_name}"
            )
            identity_value = {
                key: field for key, field in item.items() if key != id_name
            }
            if _digest(identity_value) != identifier:
                raise ValueError(
                    f"report header.{collection_name}[{index}] identity does not match"
                )
            if identifier in seen:
                raise ValueError(
                    f"report header.{collection_name} contains a duplicate"
                )
            seen.add(identifier)
    for index, certificate in enumerate(data["certificates"]):
        _certificate_from_value(certificate, f"report header.certificates[{index}]")
    return ReportHeader(
        schema_version=_REPORT_SCHEMA,
        header_digest=_require_digest(data["header_digest"], "header_digest"),
        compilation=_freeze(compilation),
        verification_spec=_freeze(verification_spec),
        tolerance_policies=tuple(_freeze(item) for item in data["tolerance_policies"]),
        oracle_references=tuple(_freeze(item) for item in data["oracle_references"]),
        certificates=tuple(_freeze(item) for item in data["certificates"]),
    )


def validate_report(header: ReportHeader, records: Sequence[EvidenceRecord]) -> None:
    """Validate exact header coverage and every per-case provenance reference."""

    requirements = {
        item["requirement_id"]: item
        for item in header.verification_spec["requirements"]
    }
    if len(requirements) != len(header.verification_spec["requirements"]):
        raise ValueError("verification specification contains duplicate requirements")
    receipts_by_kernel = {
        (item["kernel"]["kernel_id"], item["kernel"]["variant"]): item["receipt_id"]
        for item in header.compilation["kernel_receipts"]
    }
    receipt_ids = set(receipts_by_kernel.values())
    policy_ids = {item["tolerance_policy_id"] for item in header.tolerance_policies}
    oracle_ids = {item["oracle_reference_id"] for item in header.oracle_references}
    certificates_by_kernel = {
        (item["kernel_id"], item["variant"]): item["certificate_digest"]
        for item in header.certificates
    }
    if len(certificates_by_kernel) != len(header.certificates):
        raise ValueError("report header contains duplicate certificate kernel/variant")
    certificate_ids = set(certificates_by_kernel.values())
    observed_requirements = set()
    for record in records:
        if record.schema_version != _EVIDENCE_SCHEMA:
            raise ValueError(
                f"unsupported evidence schema version {record.schema_version!r}"
            )
        if record.requirement_id not in requirements:
            raise ValueError("evidence references an unknown verification requirement")
        if _requirement_value(record)["requirement_id"] != record.requirement_id:
            raise ValueError("evidence requirement reference does not match its case")
        if record.requirement_id in observed_requirements:
            raise ValueError("report contains duplicate evidence for one requirement")
        observed_requirements.add(record.requirement_id)
        if (
            record.compilation_receipt_id is not None
            and record.compilation_receipt_id not in receipt_ids
        ):
            raise ValueError("evidence references an unknown compilation receipt")
        if (
            record.case.kernel_id.startswith("cpu.")
            and record.compilation_receipt_id is None
        ):
            raise ValueError("native CPU evidence has no compilation receipt")
        expected_receipt = receipts_by_kernel.get(
            (record.case.kernel_id, record.case.variant)
        )
        if (
            expected_receipt is not None
            and record.compilation_receipt_id != expected_receipt
        ):
            raise ValueError("evidence compilation receipt does not match its kernel")
        if record.tolerance_policy_id not in policy_ids:
            raise ValueError("evidence references an unknown tolerance policy")
        if (
            _tolerance_value(record)["tolerance_policy_id"]
            != record.tolerance_policy_id
        ):
            raise ValueError(
                "evidence tolerance reference does not match its thresholds"
            )
        if record.oracle_reference_id not in oracle_ids:
            raise ValueError("evidence references an unknown oracle")
        if (
            record.consumed_certificate_digest is not None
            and record.consumed_certificate_digest not in certificate_ids
        ):
            raise ValueError("evidence references an unknown Stage One certificate")
        requires_certificate = (
            record.stage is VerificationStage.TARGET
            and record.case.kernel_id in {"cpu.reduce_sum", "cpu.matmul"}
            and record.outcome.value != "blocked"
        )
        if requires_certificate and record.consumed_certificate_digest is None:
            raise ValueError("Stage Two evidence has no consumed Stage One certificate")
        expected_certificate = certificates_by_kernel.get(
            (record.case.kernel_id, record.case.variant)
        )
        if (
            record.consumed_certificate_digest is not None
            and record.consumed_certificate_digest != expected_certificate
        ):
            raise ValueError("Stage Two certificate does not match its kernel")
    if observed_requirements != requirements.keys():
        raise ValueError(
            "report evidence does not exactly cover its verification specification"
        )
    certificates = tuple(
        _certificate_from_value(_thaw(item), f"report header.certificates[{index}]")
        for index, item in enumerate(header.certificates)
    )
    _validate_certificate_evidence(certificates, records)


def subset_report(
    report: VerificationReport, records: tuple[EvidenceRecord, ...]
) -> VerificationReport:
    """Return a self-contained provenance report for a filtered record subset."""

    if records == report.records:
        return report
    if report.header is None:  # pragma: no cover - model construction prevents this.
        raise ValueError("verification report has no provenance header")
    requirement_ids = {record.requirement_id for record in records}
    requirements = [
        _thaw(item)
        for item in report.header.verification_spec["requirements"]
        if item["requirement_id"] in requirement_ids
    ]
    spec_value = {
        "requirements": requirements,
        "spec_schema": report.header.verification_spec["spec_schema"],
    }
    verification_spec = {
        "verification_spec_id": _digest(spec_value),
        **spec_value,
    }
    policy_ids = {record.tolerance_policy_id for record in records}
    policies = [
        _thaw(item)
        for item in report.header.tolerance_policies
        if item["tolerance_policy_id"] in policy_ids
    ]
    oracle_ids = {record.oracle_reference_id for record in records}
    oracles = [
        _thaw(item)
        for item in report.header.oracle_references
        if item["oracle_reference_id"] in oracle_ids
    ]
    certificate_digests = {
        record.consumed_certificate_digest
        for record in records
        if record.consumed_certificate_digest is not None
    }
    certificate_values = tuple(
        item
        for item in report.header.certificates
        if item["certificate_digest"] in certificate_digests
    )
    header_value = {
        "certificates": [_thaw(item) for item in certificate_values],
        "compilation": _thaw(report.header.compilation),
        "evidence_schema": _EVIDENCE_SCHEMA,
        "oracle_references": oracles,
        "schema_version": report.header.schema_version,
        "tolerance_policies": policies,
        "verification_spec": verification_spec,
    }
    header_value["header_digest"] = _digest(header_value)
    header = parse_report_header(header_value)
    return VerificationReport(records, header.schema_version, header)
