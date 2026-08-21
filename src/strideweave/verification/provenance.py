"""Strict provider-neutral compilation receipt and bundle model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from .classification import LogicalKernel, VerificationProfile

_BUNDLE_SCHEMA = "strideweave.compilation-bundle.v3"
_RECEIPT_SCHEMA = "strideweave.compilation-receipt.v3"

# The installed CPU build still emits this private source-build description. It is
# adapted into v3 receipts and is never embedded in a v3 report.
_INSTALLED_MANIFEST_SCHEMA = "strideweave.kernel-compilation-manifest.v1"
_INSTALLED_RECEIPT_SCHEMA = "strideweave.kernel-compilation-receipt.v1"

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")
_LOCAL_PATH_OPTION_PREFIXES = ("-I/", "-L/", "-F/", "--sysroot=/")
_FORBIDDEN_OBSERVATION_FACT_NAMES = frozenset(
    {
        "cache_location",
        "cache_path",
        "ci",
        "ci_job",
        "commit",
        "observed_at",
        "producer_observation",
        "source_commit",
        "timestamp",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


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


def _require_string(value: Any, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_path_free_fact(value: Any, field: str) -> str:
    fact = _require_string(value, field)
    if (
        fact.startswith(("/", "\\"))
        or _WINDOWS_ABSOLUTE_PATH.search(fact) is not None
        or fact.startswith(_LOCAL_PATH_OPTION_PREFIXES)
    ):
        raise ValueError(f"{field} must not contain a local absolute path")
    return fact


def _require_provenance_fact_name(value: Any, field: str) -> str:
    name = _require_string(value, field)
    if name.lower().replace("-", "_") in _FORBIDDEN_OBSERVATION_FACT_NAMES:
        raise ValueError(f"{field} names a forbidden observation-only fact")
    return name


def _require_digest(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if _DIGEST_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _require_ordinal(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_stable_uri(value: Any, field: str) -> str:
    uri = _require_string(value, field)
    if (
        uri.startswith(("/", "\\", "file:"))
        or _WINDOWS_ABSOLUTE_PATH.match(uri) is not None
        or ".." in uri.replace("\\", "/").split("/")
    ):
        raise ValueError(f"{field} must be a stable path-free provider URI")
    return uri


def _sequence(value: Any, field: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an array")
    return value


def _require_tuple(value: Any, field: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field} must be an immutable tuple")
    return value


def _freeze_specialization_value(value: Any, field: str) -> object:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return _require_path_free_fact(value, field)
    if type(value) is float:
        try:
            _canonical_json(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a finite JSON value") from error
        return value
    if type(value) in (list, tuple):
        return tuple(
            _freeze_specialization_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(
        f"{field} must be a JSON scalar or an immutable sequence of JSON values"
    )


def _specialization_json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_specialization_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CompilationInput:
    """One ordered content-addressed provider compilation input."""

    ordinal: int
    uri: str
    input_kind: str
    content_digest: str

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal, "compilation input.ordinal")
        _require_stable_uri(self.uri, "compilation input.uri")
        _require_string(self.input_kind, "compilation input.input_kind")
        _require_digest(self.content_digest, "compilation input.content_digest")


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    """One ordered typed generated artifact bound by a compilation receipt."""

    ordinal: int
    artifact_kind: str
    content_digest: str
    executable_digest: str | None = None

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal, "generated artifact.ordinal")
        _require_string(self.artifact_kind, "generated artifact.artifact_kind")
        _require_digest(self.content_digest, "generated artifact.content_digest")
        if self.executable_digest is not None:
            _require_digest(
                self.executable_digest, "generated artifact.executable_digest"
            )


@dataclass(frozen=True, slots=True)
class SpecializationAxis:
    """One ordered compilation-affecting JIT specialization axis."""

    name: str
    value: object

    def __post_init__(self) -> None:
        _require_provenance_fact_name(self.name, "specialization axis.name")
        object.__setattr__(
            self,
            "value",
            _freeze_specialization_value(self.value, "specialization axis.value"),
        )


@dataclass(frozen=True, slots=True)
class CompilationTarget:
    """Exact execution target associated with a compilation receipt."""

    target_id: str
    architecture: str
    vendor: str
    operating_system: str
    abi: str
    endianness: str
    pointer_bits: int

    def __post_init__(self) -> None:
        identity = {
            "abi": _require_string(self.abi, "target.abi"),
            "architecture": _require_string(self.architecture, "target.architecture"),
            "endianness": _require_string(self.endianness, "target.endianness"),
            "operating_system": _require_string(
                self.operating_system, "target.operating_system"
            ),
            "pointer_bits": self.pointer_bits,
            "vendor": _require_string(self.vendor, "target.vendor"),
        }
        if type(self.pointer_bits) is not int or self.pointer_bits <= 0:
            raise ValueError("target.pointer_bits must be a positive integer")
        _require_digest(self.target_id, "target.target_id")
        if _digest(identity) != self.target_id:
            raise ValueError("target.target_id does not match its canonical descriptor")


@dataclass(frozen=True, slots=True)
class CompilationToolchain:
    """Compiler or JIT toolchain identity associated with a receipt."""

    toolchain_id: str
    provider: str
    compiler_id: str
    compiler_version: str
    target_triple: str
    build_system: str

    def __post_init__(self) -> None:
        identity = {
            "build_system": _require_string(
                self.build_system, "toolchain.build_system"
            ),
            "compiler_id": _require_string(self.compiler_id, "toolchain.compiler_id"),
            "compiler_version": _require_string(
                self.compiler_version, "toolchain.compiler_version"
            ),
            "provider": _require_string(self.provider, "toolchain.provider"),
            "target_triple": _require_string(
                self.target_triple, "toolchain.target_triple"
            ),
        }
        _require_digest(self.toolchain_id, "toolchain.toolchain_id")
        if _digest(identity) != self.toolchain_id:
            raise ValueError(
                "toolchain.toolchain_id does not match its canonical descriptor"
            )


@dataclass(frozen=True, slots=True)
class CompilationRuntime:
    """Ordered provider-reported runtime facts associated with a receipt."""

    runtime_id: str
    facts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_tuple(self.facts, "runtime.facts")
        names = []
        for index, fact in enumerate(self.facts):
            if type(fact) is not tuple or len(fact) != 2:
                raise ValueError(f"runtime.facts[{index}] must be a name/value pair")
            name, value = fact
            names.append(
                _require_provenance_fact_name(name, f"runtime.facts[{index}][0]")
            )
            _require_path_free_fact(value, f"runtime.facts[{index}][1]")
        if tuple(names) != tuple(sorted(set(names))):
            raise ValueError("runtime facts must have unique names in canonical order")
        _require_digest(self.runtime_id, "runtime.runtime_id")
        if _digest(_runtime_facts_value(self.facts)) != self.runtime_id:
            raise ValueError("runtime.runtime_id does not match its canonical facts")


@dataclass(frozen=True, slots=True)
class CompiledExecutableReceipt:
    """Immutable provenance for one precompiled logical kernel executable."""

    kind: Literal["compiled-executable"]
    schema_version: str
    receipt_id: str
    profile_id: str
    provider: str
    target: CompilationTarget
    toolchain: CompilationToolchain
    runtime: CompilationRuntime
    logical_kernel: LogicalKernel
    declared_input_uris: tuple[str, ...]
    inputs: tuple[CompilationInput, ...]
    compile_options: tuple[str, ...]
    input_closure_digest: str
    artifacts: tuple[GeneratedArtifact, ...]
    compiled_object_artifact_ordinals: tuple[int, ...]
    shared_executable_artifact_ordinal: int

    def __post_init__(self) -> None:
        _validate_receipt(self)


@dataclass(frozen=True, slots=True)
class JITSpecializationReceipt:
    """Immutable provenance for one exact JIT-compiled specialization."""

    kind: Literal["jit-specialization"]
    schema_version: str
    receipt_id: str
    profile_id: str
    provider: str
    target: CompilationTarget
    toolchain: CompilationToolchain
    runtime: CompilationRuntime
    logical_kernel: LogicalKernel
    declared_input_uris: tuple[str, ...]
    inputs: tuple[CompilationInput, ...]
    compile_options: tuple[str, ...]
    input_closure_digest: str
    artifacts: tuple[GeneratedArtifact, ...]
    specialization_axes: tuple[SpecializationAxis, ...]
    generated_host_source_artifact_ordinal: int
    generated_device_source_artifact_ordinal: int
    runtime_artifact_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_receipt(self)


CompilationReceipt: TypeAlias = CompiledExecutableReceipt | JITSpecializationReceipt


@dataclass(frozen=True, slots=True)
class CompilationBundle:
    """Canonical content-addressed collection of compilation receipts."""

    schema_version: str
    bundle_id: str
    receipts: tuple[CompilationReceipt, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported compilation bundle schema {self.schema_version!r}"
            )
        _require_digest(self.bundle_id, "compilation bundle.bundle_id")
        _require_tuple(self.receipts, "compilation bundle.receipts")
        identities = [receipt.receipt_id for receipt in self.receipts]
        if len(set(identities)) != len(identities):
            raise ValueError("compilation bundle contains a duplicate receipt")
        expected = tuple(sorted(self.receipts, key=_receipt_sort_key))
        if self.receipts != expected:
            raise ValueError("compilation bundle receipts are not in canonical order")
        if _digest(_bundle_identity_value(self.receipts)) != self.bundle_id:
            raise ValueError("compilation bundle.bundle_id does not match its contents")

    def receipt_for(self, receipt_id: str) -> CompilationReceipt:
        """Return the unique receipt with one exact content identity."""

        _require_digest(receipt_id, "receipt_id")
        matches = tuple(
            receipt for receipt in self.receipts if receipt.receipt_id == receipt_id
        )
        if len(matches) != 1:
            raise ValueError(f"compilation bundle has no unique receipt {receipt_id}")
        return matches[0]


def make_compilation_target(
    *,
    architecture: str,
    vendor: str,
    operating_system: str,
    abi: str,
    endianness: str,
    pointer_bits: int,
) -> CompilationTarget:
    """Construct one validated content-addressed compilation target."""

    identity = {
        "abi": abi,
        "architecture": architecture,
        "endianness": endianness,
        "operating_system": operating_system,
        "pointer_bits": pointer_bits,
        "vendor": vendor,
    }
    return CompilationTarget(_digest(identity), **identity)


def make_compilation_toolchain(
    *,
    provider: str,
    compiler_id: str,
    compiler_version: str,
    target_triple: str,
    build_system: str,
) -> CompilationToolchain:
    """Construct one validated content-addressed compilation toolchain."""

    identity = {
        "build_system": build_system,
        "compiler_id": compiler_id,
        "compiler_version": compiler_version,
        "provider": provider,
        "target_triple": target_triple,
    }
    return CompilationToolchain(
        toolchain_id=_digest(identity),
        provider=provider,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        target_triple=target_triple,
        build_system=build_system,
    )


def make_compilation_runtime(
    facts: Mapping[str, str] | Sequence[tuple[str, str]],
) -> CompilationRuntime:
    """Construct one validated content-addressed ordered runtime descriptor."""

    raw_facts = facts.items() if isinstance(facts, Mapping) else facts
    canonical = tuple(sorted(tuple(raw_facts)))
    return CompilationRuntime(_digest(_runtime_facts_value(canonical)), canonical)


def make_compiled_executable_receipt(
    *,
    profile_id: str,
    provider: str,
    target: CompilationTarget,
    toolchain: CompilationToolchain,
    runtime: CompilationRuntime,
    logical_kernel: LogicalKernel,
    declared_input_uris: Sequence[str],
    inputs: Sequence[CompilationInput],
    compile_options: Sequence[str],
    artifacts: Sequence[GeneratedArtifact],
    compiled_object_artifact_ordinals: Sequence[int],
    shared_executable_artifact_ordinal: int,
) -> CompiledExecutableReceipt:
    """Construct a strict content-addressed compiled-executable receipt."""

    common = _prepare_common_receipt_values(
        profile_id=profile_id,
        provider=provider,
        target=target,
        toolchain=toolchain,
        runtime=runtime,
        logical_kernel=logical_kernel,
        declared_input_uris=declared_input_uris,
        inputs=inputs,
        compile_options=compile_options,
        artifacts=artifacts,
    )
    object_ordinals = tuple(compiled_object_artifact_ordinals)
    identity = _receipt_identity_from_values(
        kind="compiled-executable",
        common=common,
        compiled_object_artifact_ordinals=object_ordinals,
        shared_executable_artifact_ordinal=shared_executable_artifact_ordinal,
    )
    return CompiledExecutableReceipt(
        kind="compiled-executable",
        schema_version=_RECEIPT_SCHEMA,
        receipt_id=_digest(identity),
        compiled_object_artifact_ordinals=object_ordinals,
        shared_executable_artifact_ordinal=shared_executable_artifact_ordinal,
        **common,
    )


def make_jit_specialization_receipt(
    *,
    profile_id: str,
    provider: str,
    target: CompilationTarget,
    toolchain: CompilationToolchain,
    runtime: CompilationRuntime,
    logical_kernel: LogicalKernel,
    declared_input_uris: Sequence[str],
    inputs: Sequence[CompilationInput],
    compile_options: Sequence[str],
    artifacts: Sequence[GeneratedArtifact],
    specialization_axes: Sequence[SpecializationAxis],
    generated_host_source_artifact_ordinal: int,
    generated_device_source_artifact_ordinal: int,
    runtime_artifact_ordinals: Sequence[int],
) -> JITSpecializationReceipt:
    """Construct a strict content-addressed JIT-specialization receipt."""

    common = _prepare_common_receipt_values(
        profile_id=profile_id,
        provider=provider,
        target=target,
        toolchain=toolchain,
        runtime=runtime,
        logical_kernel=logical_kernel,
        declared_input_uris=declared_input_uris,
        inputs=inputs,
        compile_options=compile_options,
        artifacts=artifacts,
    )
    axes = tuple(specialization_axes)
    runtime_ordinals = tuple(runtime_artifact_ordinals)
    identity = _receipt_identity_from_values(
        kind="jit-specialization",
        common=common,
        specialization_axes=axes,
        generated_host_source_artifact_ordinal=generated_host_source_artifact_ordinal,
        generated_device_source_artifact_ordinal=generated_device_source_artifact_ordinal,
        runtime_artifact_ordinals=runtime_ordinals,
    )
    return JITSpecializationReceipt(
        kind="jit-specialization",
        schema_version=_RECEIPT_SCHEMA,
        receipt_id=_digest(identity),
        specialization_axes=axes,
        generated_host_source_artifact_ordinal=generated_host_source_artifact_ordinal,
        generated_device_source_artifact_ordinal=generated_device_source_artifact_ordinal,
        runtime_artifact_ordinals=runtime_ordinals,
        **common,
    )


def make_compilation_bundle(
    receipts: Sequence[CompilationReceipt],
) -> CompilationBundle:
    """Construct a canonical content-addressed compilation bundle."""

    ordered = tuple(sorted(tuple(receipts), key=_receipt_sort_key))
    return CompilationBundle(
        schema_version=_BUNDLE_SCHEMA,
        bundle_id=_digest(_bundle_identity_value(ordered)),
        receipts=ordered,
    )


def _prepare_common_receipt_values(
    *,
    profile_id: str,
    provider: str,
    target: CompilationTarget,
    toolchain: CompilationToolchain,
    runtime: CompilationRuntime,
    logical_kernel: LogicalKernel,
    declared_input_uris: Sequence[str],
    inputs: Sequence[CompilationInput],
    compile_options: Sequence[str],
    artifacts: Sequence[GeneratedArtifact],
) -> dict[str, Any]:
    input_values = tuple(inputs)
    option_values = tuple(compile_options)
    return {
        "profile_id": profile_id,
        "provider": provider,
        "target": target,
        "toolchain": toolchain,
        "runtime": runtime,
        "logical_kernel": logical_kernel,
        "declared_input_uris": tuple(declared_input_uris),
        "inputs": input_values,
        "compile_options": option_values,
        "input_closure_digest": _digest(
            _input_closure_value(input_values, option_values)
        ),
        "artifacts": tuple(artifacts),
    }


def _validate_receipt(receipt: CompilationReceipt) -> None:
    if receipt.schema_version != _RECEIPT_SCHEMA:
        raise ValueError(
            f"unsupported compilation receipt schema {receipt.schema_version!r}"
        )
    _require_digest(receipt.receipt_id, "compilation receipt.receipt_id")
    _require_string(receipt.profile_id, "compilation receipt.profile_id")
    provider = _require_string(receipt.provider, "compilation receipt.provider")
    if receipt.toolchain.provider != provider:
        raise ValueError("receipt provider does not match its toolchain provider")
    kernel = receipt.logical_kernel
    for field in ("profile_id", "operation", "kernel_id", "variant"):
        _require_string(getattr(kernel, field), f"logical kernel.{field}")
    if kernel.profile_id != receipt.profile_id:
        raise ValueError("receipt profile does not match its logical kernel profile")

    for field in (
        "declared_input_uris",
        "inputs",
        "compile_options",
        "artifacts",
    ):
        _require_tuple(getattr(receipt, field), f"compilation receipt.{field}")

    declared = tuple(
        _require_stable_uri(uri, f"declared_input_uris[{index}]")
        for index, uri in enumerate(receipt.declared_input_uris)
    )
    if len(set(declared)) != len(declared):
        raise ValueError("declared compilation inputs contain a duplicate URI")
    _validate_ordered_values(receipt.inputs, "compilation inputs")
    observed = tuple(item.uri for item in receipt.inputs)
    missing = tuple(uri for uri in declared if uri not in observed)
    undeclared = tuple(uri for uri in observed if uri not in declared)
    if missing or undeclared:
        raise ValueError(
            "compilation input closure does not match provider declaration: "
            f"missing={missing!r}, undeclared={undeclared!r}"
        )
    if declared != observed:
        raise ValueError(
            "declared compilation input URIs must match provider input order"
        )
    if not receipt.inputs or not any(
        item.input_kind == "source" for item in receipt.inputs
    ):
        raise ValueError("compilation input closure must contain an owning source")
    for index, option in enumerate(receipt.compile_options):
        try:
            _require_path_free_fact(option, f"compile_options[{index}]")
        except ValueError as error:
            raise ValueError(
                "compile options must contain non-empty path-free strings"
            ) from error
    expected_closure = _digest(
        _input_closure_value(receipt.inputs, receipt.compile_options)
    )
    if receipt.input_closure_digest != expected_closure:
        raise ValueError("input closure digest does not match inputs and options")

    _validate_ordered_values(receipt.artifacts, "generated artifacts")
    if not receipt.artifacts:
        raise ValueError("compilation receipt must contain generated artifacts")
    artifacts = {artifact.ordinal: artifact for artifact in receipt.artifacts}
    if isinstance(receipt, CompiledExecutableReceipt):
        if receipt.kind != "compiled-executable":
            raise ValueError("compiled receipt has an invalid discriminator")
        objects = receipt.compiled_object_artifact_ordinals
        _require_tuple(objects, "compiled receipt.compiled_object_artifact_ordinals")
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("compiled receipt must name unique object artifacts")
        if tuple(objects) != tuple(sorted(objects)):
            raise ValueError("compiled object artifacts are not in canonical order")
        for ordinal in objects:
            _require_artifact_kind(artifacts, ordinal, "compiled-object")
        _require_artifact_kind(
            artifacts,
            receipt.shared_executable_artifact_ordinal,
            "shared-executable",
        )
        claimed = {*objects, receipt.shared_executable_artifact_ordinal}
        if claimed != artifacts.keys():
            raise ValueError("compiled receipt has unclaimed generated artifacts")
    else:
        if receipt.kind != "jit-specialization":
            raise ValueError("JIT receipt has an invalid discriminator")
        axes = receipt.specialization_axes
        _require_tuple(axes, "JIT receipt.specialization_axes")
        axis_names = tuple(axis.name for axis in axes)
        if not axes or len(set(axis_names)) != len(axis_names):
            raise ValueError("JIT specialization axes must be non-empty and unique")
        host = receipt.generated_host_source_artifact_ordinal
        device = receipt.generated_device_source_artifact_ordinal
        if host == device:
            raise ValueError("generated host and device sources must be distinct")
        _require_artifact_kind(artifacts, host, "generated-host-source")
        _require_artifact_kind(artifacts, device, "generated-device-source")
        runtime_ordinals = receipt.runtime_artifact_ordinals
        _require_tuple(runtime_ordinals, "JIT receipt.runtime_artifact_ordinals")
        if len(set(runtime_ordinals)) != len(runtime_ordinals):
            raise ValueError("JIT runtime artifact ordinals contain a duplicate")
        if tuple(runtime_ordinals) != tuple(sorted(runtime_ordinals)):
            raise ValueError("JIT runtime artifacts are not in canonical order")
        for ordinal in runtime_ordinals:
            artifact = artifacts.get(ordinal)
            if artifact is None:
                raise ValueError("JIT receipt references an unknown runtime artifact")
            if artifact.artifact_kind in {
                "generated-host-source",
                "generated-device-source",
            }:
                raise ValueError(
                    "generated source cannot be claimed as a runtime artifact"
                )
        if {host, device, *runtime_ordinals} != artifacts.keys():
            raise ValueError("JIT receipt has unclaimed generated artifacts")

    if _digest(_receipt_identity_value(receipt)) != receipt.receipt_id:
        raise ValueError("compilation receipt.receipt_id does not match its contents")


def _validate_ordered_values(
    values: Sequence[CompilationInput] | Sequence[GeneratedArtifact], field: str
) -> None:
    ordinals = tuple(item.ordinal for item in values)
    if ordinals != tuple(range(len(values))):
        raise ValueError(f"{field} must use unique contiguous canonical ordinals")
    inputs = tuple(item for item in values if isinstance(item, CompilationInput))
    if inputs and len(inputs) == len(values):
        uris = tuple(item.uri for item in inputs)
        if len(set(uris)) != len(uris):
            raise ValueError(f"{field} contain a duplicate URI")


def _require_artifact_kind(
    artifacts: Mapping[int, GeneratedArtifact], ordinal: int, expected: str
) -> GeneratedArtifact:
    _require_ordinal(ordinal, "artifact reference")
    artifact = artifacts.get(ordinal)
    if artifact is None or artifact.artifact_kind != expected:
        raise ValueError(f"artifact {ordinal} must have kind {expected!r}")
    return artifact


def _runtime_facts_value(facts: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in facts]


def _target_value(target: CompilationTarget) -> dict[str, Any]:
    return {
        "abi": target.abi,
        "architecture": target.architecture,
        "endianness": target.endianness,
        "operating_system": target.operating_system,
        "pointer_bits": target.pointer_bits,
        "target_id": target.target_id,
        "vendor": target.vendor,
    }


def _toolchain_value(toolchain: CompilationToolchain) -> dict[str, str]:
    return {
        "build_system": toolchain.build_system,
        "compiler_id": toolchain.compiler_id,
        "compiler_version": toolchain.compiler_version,
        "provider": toolchain.provider,
        "target_triple": toolchain.target_triple,
        "toolchain_id": toolchain.toolchain_id,
    }


def _runtime_value(runtime: CompilationRuntime) -> dict[str, Any]:
    return {
        "facts": _runtime_facts_value(runtime.facts),
        "runtime_id": runtime.runtime_id,
    }


def _kernel_value(kernel: LogicalKernel) -> dict[str, str]:
    return {
        "kernel_id": kernel.kernel_id,
        "operation": kernel.operation,
        "profile_id": kernel.profile_id,
        "variant": kernel.variant,
    }


def _input_value(value: CompilationInput) -> dict[str, Any]:
    return {
        "content_digest": value.content_digest,
        "input_kind": value.input_kind,
        "ordinal": value.ordinal,
        "uri": value.uri,
    }


def _artifact_value(value: GeneratedArtifact) -> dict[str, Any]:
    return {
        "artifact_kind": value.artifact_kind,
        "content_digest": value.content_digest,
        "executable_digest": value.executable_digest,
        "ordinal": value.ordinal,
    }


def _axis_value(value: SpecializationAxis) -> dict[str, object]:
    return {
        "name": value.name,
        "value": _specialization_json_value(value.value),
    }


def _input_closure_value(
    inputs: Sequence[CompilationInput], compile_options: Sequence[str]
) -> dict[str, Any]:
    return {
        "compile_options": list(compile_options),
        "inputs": [_input_value(item) for item in inputs],
    }


def compilation_receipt_json_object(
    receipt: CompilationReceipt,
) -> dict[str, Any]:
    """Return the canonical mutable JSON object for one validated receipt."""

    value = _receipt_identity_value(receipt)
    value["receipt_id"] = receipt.receipt_id
    return value


def _receipt_identity_from_values(
    *,
    kind: Literal["compiled-executable", "jit-specialization"],
    common: Mapping[str, Any],
    schema_version: str = _RECEIPT_SCHEMA,
    compiled_object_artifact_ordinals: tuple[int, ...] = (),
    shared_executable_artifact_ordinal: int | None = None,
    specialization_axes: tuple[SpecializationAxis, ...] = (),
    generated_host_source_artifact_ordinal: int | None = None,
    generated_device_source_artifact_ordinal: int | None = None,
    runtime_artifact_ordinals: tuple[int, ...] = (),
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "artifacts": [_artifact_value(item) for item in common["artifacts"]],
        "compile_options": list(common["compile_options"]),
        "declared_input_uris": list(common["declared_input_uris"]),
        "input_closure_digest": common["input_closure_digest"],
        "inputs": [_input_value(item) for item in common["inputs"]],
        "kind": kind,
        "logical_kernel": _kernel_value(common["logical_kernel"]),
        "profile_id": common["profile_id"],
        "provider": common["provider"],
        "runtime": _runtime_value(common["runtime"]),
        "schema_version": schema_version,
        "target": _target_value(common["target"]),
        "toolchain": _toolchain_value(common["toolchain"]),
    }
    if kind == "compiled-executable":
        value.update(
            {
                "compiled_object_artifact_ordinals": list(
                    compiled_object_artifact_ordinals
                ),
                "shared_executable_artifact_ordinal": shared_executable_artifact_ordinal,
            }
        )
    else:
        value.update(
            {
                "generated_device_source_artifact_ordinal": generated_device_source_artifact_ordinal,
                "generated_host_source_artifact_ordinal": generated_host_source_artifact_ordinal,
                "runtime_artifact_ordinals": list(runtime_artifact_ordinals),
                "specialization_axes": [
                    _axis_value(item) for item in specialization_axes
                ],
            }
        )
    return value


def _receipt_identity_value(receipt: CompilationReceipt) -> dict[str, Any]:
    common = {
        "artifacts": receipt.artifacts,
        "compile_options": receipt.compile_options,
        "declared_input_uris": receipt.declared_input_uris,
        "input_closure_digest": receipt.input_closure_digest,
        "inputs": receipt.inputs,
        "logical_kernel": receipt.logical_kernel,
        "profile_id": receipt.profile_id,
        "provider": receipt.provider,
        "runtime": receipt.runtime,
        "target": receipt.target,
        "toolchain": receipt.toolchain,
    }
    if isinstance(receipt, CompiledExecutableReceipt):
        return _receipt_identity_from_values(
            kind=receipt.kind,
            common=common,
            schema_version=receipt.schema_version,
            compiled_object_artifact_ordinals=(
                receipt.compiled_object_artifact_ordinals
            ),
            shared_executable_artifact_ordinal=(
                receipt.shared_executable_artifact_ordinal
            ),
        )
    return _receipt_identity_from_values(
        kind=receipt.kind,
        common=common,
        schema_version=receipt.schema_version,
        specialization_axes=receipt.specialization_axes,
        generated_host_source_artifact_ordinal=(
            receipt.generated_host_source_artifact_ordinal
        ),
        generated_device_source_artifact_ordinal=(
            receipt.generated_device_source_artifact_ordinal
        ),
        runtime_artifact_ordinals=receipt.runtime_artifact_ordinals,
    )


def _receipt_sort_key(receipt: CompilationReceipt) -> tuple[str, ...]:
    return (
        receipt.kind,
        receipt.profile_id,
        receipt.logical_kernel.operation,
        receipt.logical_kernel.kernel_id,
        receipt.logical_kernel.variant,
        receipt.receipt_id,
    )


def _bundle_identity_value(
    receipts: Sequence[CompilationReceipt],
) -> dict[str, Any]:
    return {
        "receipts": [compilation_receipt_json_object(item) for item in receipts],
        "schema_version": _BUNDLE_SCHEMA,
    }


def compilation_bundle_json_object(bundle: CompilationBundle) -> dict[str, Any]:
    """Return the canonical mutable JSON object for one validated bundle."""

    return {
        "bundle_id": bundle.bundle_id,
        **_bundle_identity_value(bundle.receipts),
    }


def _parse_target(value: Any, field: str) -> CompilationTarget:
    data = _require_object(
        value,
        field,
        frozenset(
            {
                "abi",
                "architecture",
                "endianness",
                "operating_system",
                "pointer_bits",
                "target_id",
                "vendor",
            }
        ),
    )
    return CompilationTarget(
        target_id=_require_digest(data["target_id"], f"{field}.target_id"),
        architecture=_require_string(data["architecture"], f"{field}.architecture"),
        vendor=_require_string(data["vendor"], f"{field}.vendor"),
        operating_system=_require_string(
            data["operating_system"], f"{field}.operating_system"
        ),
        abi=_require_string(data["abi"], f"{field}.abi"),
        endianness=_require_string(data["endianness"], f"{field}.endianness"),
        pointer_bits=data["pointer_bits"],
    )


def _parse_toolchain(value: Any, field: str) -> CompilationToolchain:
    data = _require_object(
        value,
        field,
        frozenset(
            {
                "build_system",
                "compiler_id",
                "compiler_version",
                "provider",
                "target_triple",
                "toolchain_id",
            }
        ),
    )
    return CompilationToolchain(
        toolchain_id=_require_digest(data["toolchain_id"], f"{field}.toolchain_id"),
        provider=_require_string(data["provider"], f"{field}.provider"),
        compiler_id=_require_string(data["compiler_id"], f"{field}.compiler_id"),
        compiler_version=_require_string(
            data["compiler_version"], f"{field}.compiler_version"
        ),
        target_triple=_require_string(data["target_triple"], f"{field}.target_triple"),
        build_system=_require_string(data["build_system"], f"{field}.build_system"),
    )


def _parse_runtime(value: Any, field: str) -> CompilationRuntime:
    data = _require_object(value, field, frozenset({"facts", "runtime_id"}))
    facts = []
    for index, item in enumerate(_sequence(data["facts"], f"{field}.facts")):
        fact_field = f"{field}.facts[{index}]"
        fact = _require_object(item, fact_field, frozenset({"name", "value"}))
        facts.append(
            (
                _require_string(fact["name"], f"{fact_field}.name"),
                _require_string(fact["value"], f"{fact_field}.value"),
            )
        )
    return CompilationRuntime(
        _require_digest(data["runtime_id"], f"{field}.runtime_id"), tuple(facts)
    )


def _parse_logical_kernel(value: Any, field: str) -> LogicalKernel:
    from .classification import LogicalKernel

    data = _require_object(
        value,
        field,
        frozenset({"kernel_id", "operation", "profile_id", "variant"}),
    )
    return LogicalKernel(
        profile_id=_require_string(data["profile_id"], f"{field}.profile_id"),
        operation=_require_string(data["operation"], f"{field}.operation"),
        kernel_id=_require_string(data["kernel_id"], f"{field}.kernel_id"),
        variant=_require_string(data["variant"], f"{field}.variant"),
    )


def _parse_inputs(value: Any, field: str) -> tuple[CompilationInput, ...]:
    result = []
    for index, item in enumerate(_sequence(value, field)):
        item_field = f"{field}[{index}]"
        data = _require_object(
            item,
            item_field,
            frozenset({"content_digest", "input_kind", "ordinal", "uri"}),
        )
        result.append(
            CompilationInput(
                ordinal=_require_ordinal(data["ordinal"], f"{item_field}.ordinal"),
                uri=_require_stable_uri(data["uri"], f"{item_field}.uri"),
                input_kind=_require_string(
                    data["input_kind"], f"{item_field}.input_kind"
                ),
                content_digest=_require_digest(
                    data["content_digest"], f"{item_field}.content_digest"
                ),
            )
        )
    return tuple(result)


def _parse_artifacts(value: Any, field: str) -> tuple[GeneratedArtifact, ...]:
    result = []
    for index, item in enumerate(_sequence(value, field)):
        item_field = f"{field}[{index}]"
        data = _require_object(
            item,
            item_field,
            frozenset(
                {"artifact_kind", "content_digest", "executable_digest", "ordinal"}
            ),
        )
        executable = data["executable_digest"]
        result.append(
            GeneratedArtifact(
                ordinal=_require_ordinal(data["ordinal"], f"{item_field}.ordinal"),
                artifact_kind=_require_string(
                    data["artifact_kind"], f"{item_field}.artifact_kind"
                ),
                content_digest=_require_digest(
                    data["content_digest"], f"{item_field}.content_digest"
                ),
                executable_digest=(
                    None
                    if executable is None
                    else _require_digest(executable, f"{item_field}.executable_digest")
                ),
            )
        )
    return tuple(result)


def _parse_ordinals(value: Any, field: str) -> tuple[int, ...]:
    return tuple(
        _require_ordinal(item, f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field))
    )


_COMMON_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "compile_options",
        "declared_input_uris",
        "input_closure_digest",
        "inputs",
        "kind",
        "logical_kernel",
        "profile_id",
        "provider",
        "receipt_id",
        "runtime",
        "schema_version",
        "target",
        "toolchain",
    }
)
_COMPILED_RECEIPT_FIELDS = _COMMON_RECEIPT_FIELDS | {
    "compiled_object_artifact_ordinals",
    "shared_executable_artifact_ordinal",
}
_JIT_RECEIPT_FIELDS = _COMMON_RECEIPT_FIELDS | {
    "generated_device_source_artifact_ordinal",
    "generated_host_source_artifact_ordinal",
    "runtime_artifact_ordinals",
    "specialization_axes",
}


def parse_compilation_receipt(value: Any, field: str) -> CompilationReceipt:
    """Strictly parse one v3 receipt using only supplied facts."""

    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    kind = value.get("kind")
    if kind == "compiled-executable":
        data = _require_object(value, field, _COMPILED_RECEIPT_FIELDS)
    elif kind == "jit-specialization":
        data = _require_object(value, field, _JIT_RECEIPT_FIELDS)
    else:
        raise ValueError(f"{field}.kind has unknown discriminator {kind!r}")
    schema = _require_string(data["schema_version"], f"{field}.schema_version")
    if schema != _RECEIPT_SCHEMA:
        raise ValueError(f"unsupported compilation receipt schema {schema!r}")
    declared = tuple(
        _require_stable_uri(item, f"{field}.declared_input_uris[{index}]")
        for index, item in enumerate(
            _sequence(data["declared_input_uris"], f"{field}.declared_input_uris")
        )
    )
    options = tuple(
        _require_string(item, f"{field}.compile_options[{index}]")
        for index, item in enumerate(
            _sequence(data["compile_options"], f"{field}.compile_options")
        )
    )
    common: dict[str, Any] = {
        "kind": kind,
        "schema_version": schema,
        "receipt_id": _require_digest(data["receipt_id"], f"{field}.receipt_id"),
        "profile_id": _require_string(data["profile_id"], f"{field}.profile_id"),
        "provider": _require_string(data["provider"], f"{field}.provider"),
        "target": _parse_target(data["target"], f"{field}.target"),
        "toolchain": _parse_toolchain(data["toolchain"], f"{field}.toolchain"),
        "runtime": _parse_runtime(data["runtime"], f"{field}.runtime"),
        "logical_kernel": _parse_logical_kernel(
            data["logical_kernel"], f"{field}.logical_kernel"
        ),
        "declared_input_uris": declared,
        "inputs": _parse_inputs(data["inputs"], f"{field}.inputs"),
        "compile_options": options,
        "input_closure_digest": _require_digest(
            data["input_closure_digest"], f"{field}.input_closure_digest"
        ),
        "artifacts": _parse_artifacts(data["artifacts"], f"{field}.artifacts"),
    }
    if kind == "compiled-executable":
        return CompiledExecutableReceipt(
            **common,
            compiled_object_artifact_ordinals=_parse_ordinals(
                data["compiled_object_artifact_ordinals"],
                f"{field}.compiled_object_artifact_ordinals",
            ),
            shared_executable_artifact_ordinal=_require_ordinal(
                data["shared_executable_artifact_ordinal"],
                f"{field}.shared_executable_artifact_ordinal",
            ),
        )
    axes = []
    for index, item in enumerate(
        _sequence(data["specialization_axes"], f"{field}.specialization_axes")
    ):
        axis_field = f"{field}.specialization_axes[{index}]"
        axis = _require_object(item, axis_field, frozenset({"name", "value"}))
        axes.append(
            SpecializationAxis(
                _require_string(axis["name"], f"{axis_field}.name"),
                axis["value"],
            )
        )
    return JITSpecializationReceipt(
        **common,
        specialization_axes=tuple(axes),
        generated_host_source_artifact_ordinal=_require_ordinal(
            data["generated_host_source_artifact_ordinal"],
            f"{field}.generated_host_source_artifact_ordinal",
        ),
        generated_device_source_artifact_ordinal=_require_ordinal(
            data["generated_device_source_artifact_ordinal"],
            f"{field}.generated_device_source_artifact_ordinal",
        ),
        runtime_artifact_ordinals=_parse_ordinals(
            data["runtime_artifact_ordinals"], f"{field}.runtime_artifact_ordinals"
        ),
    )


def parse_compilation_bundle(value: Any) -> CompilationBundle:
    """Strictly parse one self-contained v3 compilation bundle."""

    data = _require_object(
        value,
        "compilation bundle",
        frozenset({"bundle_id", "receipts", "schema_version"}),
    )
    receipts = tuple(
        parse_compilation_receipt(item, f"compilation bundle.receipts[{index}]")
        for index, item in enumerate(
            _sequence(data["receipts"], "compilation bundle.receipts")
        )
    )
    return CompilationBundle(
        schema_version=_require_string(
            data["schema_version"], "compilation bundle.schema_version"
        ),
        bundle_id=_require_digest(data["bundle_id"], "compilation bundle.bundle_id"),
        receipts=receipts,
    )


@dataclass(frozen=True, slots=True)
class _InstalledSource:
    source: str
    inputs: tuple[CompilationInput, ...]
    compile_options: tuple[str, ...]
    object_digest: str


def _parse_installed_source(value: Any, index: int) -> _InstalledSource:
    field = f"installed manifest.sources[{index}]"
    data = _require_object(
        value,
        field,
        frozenset(
            {
                "closure_id",
                "compile_invocation",
                "compile_invocation_digest",
                "inputs",
                "object_digest",
                "source",
            }
        ),
    )
    options = tuple(
        _require_string(item, f"{field}.compile_invocation[{option_index}]")
        for option_index, item in enumerate(
            _sequence(data["compile_invocation"], f"{field}.compile_invocation")
        )
    )
    if _digest(options) != data["compile_invocation_digest"]:
        raise ValueError(f"{field}.compile_invocation_digest does not match")
    raw_inputs = _sequence(data["inputs"], f"{field}.inputs")
    inputs = tuple(
        CompilationInput(
            ordinal=input_index,
            uri=_require_stable_uri(item["uri"], f"{field}.inputs[{input_index}].uri"),
            input_kind=_require_string(
                item["input_kind"], f"{field}.inputs[{input_index}].input_kind"
            ),
            content_digest=_require_digest(
                item["content_digest"],
                f"{field}.inputs[{input_index}].content_digest",
            ),
        )
        for input_index, raw_item in enumerate(raw_inputs)
        for item in (
            _require_object(
                raw_item,
                f"{field}.inputs[{input_index}]",
                frozenset({"content_digest", "input_kind", "uri"}),
            ),
        )
    )
    source = _require_stable_uri(data["source"], f"{field}.source")
    if not any(item.uri == source and item.input_kind == "source" for item in inputs):
        raise ValueError(f"{field} has no owning source input")
    legacy_closure = {
        "inputs": [
            {
                "content_digest": item.content_digest,
                "input_kind": item.input_kind,
                "uri": item.uri,
            }
            for item in inputs
        ],
        "invocation": list(options),
    }
    if _digest(legacy_closure) != data["closure_id"]:
        raise ValueError(f"{field}.closure_id does not match")
    return _InstalledSource(
        source=source,
        inputs=inputs,
        compile_options=options,
        object_digest=_require_digest(data["object_digest"], f"{field}.object_digest"),
    )


def _installed_manifest_value() -> dict[str, Any]:
    resource = resources.files("strideweave.verification").joinpath(
        "_native_provenance.json"
    )
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "installed StrideWeave build has no valid compilation provenance"
        ) from error
    if type(value) is not dict:
        raise RuntimeError(
            "installed StrideWeave build has no valid compilation provenance"
        )
    return value


@lru_cache(maxsize=1)
def _installed_cpu_compilation_bundle() -> CompilationBundle:
    from strideweave import _carrier

    from .classification import kernel_manifest, verification_profile

    profile = verification_profile("cpu-compiled")
    logical_kernels = kernel_manifest(profile)
    raw = _installed_manifest_value()
    fields = frozenset(
        {
            "artifact_digest",
            "manifest_digest",
            "provider_kind",
            "receipt_schema",
            "schema",
            "sources",
            "target",
            "toolchain",
        }
    )
    data = _require_object(raw, "installed manifest", fields)
    if data["schema"] != _INSTALLED_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported installed manifest schema {data['schema']!r}")
    if data["receipt_schema"] != _INSTALLED_RECEIPT_SCHEMA:
        raise ValueError(
            f"unsupported installed receipt schema {data['receipt_schema']!r}"
        )
    if (
        _digest({key: data[key] for key in fields - {"manifest_digest"}})
        != data["manifest_digest"]
    ):
        raise ValueError("installed manifest digest does not match its contents")
    provider = _require_string(data["provider_kind"], "installed manifest.provider")
    legacy_target = _require_object(
        data["target"],
        "installed manifest.target",
        frozenset(
            {
                "abi",
                "architecture",
                "endianness",
                "operating_system",
                "pointer_bits",
                "target_id",
                "vendor",
            }
        ),
    )
    target = _parse_target(legacy_target, "installed manifest.target")
    legacy_toolchain = _require_object(
        data["toolchain"],
        "installed manifest.toolchain",
        frozenset(
            {
                "build_system",
                "compiler_id",
                "compiler_version",
                "provider_kind",
                "target_triple",
                "toolchain_id",
            }
        ),
    )
    if legacy_toolchain["provider_kind"] != provider:
        raise ValueError("installed manifest provider does not match its toolchain")
    if (
        _digest(
            {
                key: value
                for key, value in legacy_toolchain.items()
                if key != "toolchain_id"
            }
        )
        != legacy_toolchain["toolchain_id"]
    ):
        raise ValueError("installed manifest toolchain identity does not match")
    toolchain = make_compilation_toolchain(
        provider=provider,
        compiler_id=_require_string(legacy_toolchain["compiler_id"], "compiler_id"),
        compiler_version=_require_string(
            legacy_toolchain["compiler_version"], "compiler_version"
        ),
        target_triple=_require_string(
            legacy_toolchain["target_triple"], "target_triple"
        ),
        build_system=_require_string(legacy_toolchain["build_system"], "build_system"),
    )
    runtime = make_compilation_runtime(
        {
            "execution_model": "compiled-shared-executable",
            "framework": "strideweave",
        }
    )
    sources = tuple(
        _parse_installed_source(item, index)
        for index, item in enumerate(
            _sequence(data["sources"], "installed manifest.sources")
        )
    )
    by_source = {source.source: source for source in sources}
    if len(by_source) != len(sources):
        raise ValueError("installed manifest contains a duplicate owning source")

    native_metadata = tuple(_carrier._cpu_native_kernel_metadata())
    native_by_identity = {(entry[0], entry[2]): entry for entry in native_metadata}
    if len(native_by_identity) != len(native_metadata):
        raise ValueError("installed native metadata contains a duplicate kernel")
    if {(item.operation, item.variant) for item in logical_kernels} != set(
        native_by_identity
    ):
        raise ValueError(
            "installed compilation inputs do not match logical kernel metadata"
        )
    source_names = {entry[4] for entry in native_metadata}
    if source_names != by_source.keys():
        raise ValueError(
            "installed compilation sources do not match native kernel metadata"
        )
    shared_digest = _require_digest(
        data["artifact_digest"], "installed manifest.artifact_digest"
    )
    receipts = []
    for kernel in logical_kernels:
        source = by_source[native_by_identity[(kernel.operation, kernel.variant)][4]]
        artifacts = (
            GeneratedArtifact(0, "compiled-object", source.object_digest),
            GeneratedArtifact(
                1,
                "shared-executable",
                shared_digest,
                executable_digest=shared_digest,
            ),
        )
        receipts.append(
            make_compiled_executable_receipt(
                profile_id=profile.profile_id,
                provider=provider,
                target=target,
                toolchain=toolchain,
                runtime=runtime,
                logical_kernel=kernel,
                declared_input_uris=tuple(item.uri for item in source.inputs),
                inputs=source.inputs,
                compile_options=source.compile_options,
                artifacts=artifacts,
                compiled_object_artifact_ordinals=(0,),
                shared_executable_artifact_ordinal=1,
            )
        )
    return make_compilation_bundle(receipts)


def installed_compilation_bundle(profile: VerificationProfile) -> CompilationBundle:
    """Return installed compiled-executable receipts for one registered profile.

    Args:
        profile: Registered verification profile whose pre-execution compilation
            facts are requested.

    Returns:
        Immutable, content-addressed schema-v3 bundle for the installed profile.

    Examples:
        >>> from strideweave.verification import (
        ...     installed_compilation_bundle,
        ...     verification_profile,
        ... )
        >>> bundle = installed_compilation_bundle(
        ...     verification_profile("cpu-compiled")
        ... )
        >>> bool(bundle.receipts)
        True
    """

    from .classification import VerificationProfile, verification_profile

    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    registered = verification_profile(profile.profile_id)
    if registered is not profile and registered != profile:
        raise ValueError("profile does not match the registered verification profile")
    if profile.profile_id != "cpu-compiled":
        raise RuntimeError(
            f"profile {profile.profile_id!r} has no pre-execution installed bundle"
        )
    return _installed_cpu_compilation_bundle()
