"""Strict framework-neutral compilation receipt model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from .classification import native_cpu_kernel_manifest
from .model import KernelDescriptor

_MANIFEST_SCHEMA = "strideweave.kernel-compilation-manifest.v1"
_RECEIPT_SCHEMA = "strideweave.kernel-compilation-receipt.v1"


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


def _require_digest(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


@dataclass(frozen=True, slots=True)
class CompilationInput:
    """One content-addressed input in a kernel source closure."""

    uri: str
    input_kind: str
    content_digest: str


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


@dataclass(frozen=True, slots=True)
class CompilationToolchain:
    """Compiler and build-system identity associated with a receipt."""

    toolchain_id: str
    provider_kind: str
    compiler_id: str
    compiler_version: str
    target_triple: str
    build_system: str


@dataclass(frozen=True, slots=True)
class CompilationReceipt:
    """Immutable provenance for one native kernel and variant."""

    receipt_id: str
    receipt_schema: str
    provider_kind: str
    kernel: KernelDescriptor
    target: CompilationTarget
    toolchain: CompilationToolchain
    closure_id: str
    inputs: tuple[CompilationInput, ...]
    compile_invocation: tuple[str, ...]
    compile_invocation_digest: str
    object_digest: str
    shared_artifact_digest: str
    framework_name: str | None
    framework_version: str | None
    specialization: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CompilationManifest:
    """Strict installed manifest of current native kernel receipts."""

    manifest_digest: str
    schema: str
    provider_kind: str
    artifact_digest: str
    target: CompilationTarget
    toolchain: CompilationToolchain
    receipts: tuple[CompilationReceipt, ...]

    def receipt_for(self, kernel_id: str, variant: str) -> CompilationReceipt:
        """Return the exact receipt for a kernel identity or fail closed."""

        matches = tuple(
            receipt
            for receipt in self.receipts
            if receipt.kernel.kernel_id == kernel_id
            and receipt.kernel.variant == variant
        )
        if len(matches) != 1:
            raise ValueError(
                f"compilation manifest has no unique receipt for {kernel_id}/{variant}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class _SourceCompilation:
    source: str
    closure_id: str
    inputs: tuple[CompilationInput, ...]
    compile_invocation: tuple[str, ...]
    compile_invocation_digest: str
    object_digest: str


def _parse_target(value: Any) -> CompilationTarget:
    fields = frozenset(
        {
            "abi",
            "architecture",
            "endianness",
            "operating_system",
            "pointer_bits",
            "target_id",
            "vendor",
        }
    )
    data = _require_object(value, "target", fields)
    identity = {key: data[key] for key in fields - {"target_id"}}
    if _digest(identity) != data["target_id"]:
        raise ValueError("target.target_id does not match its canonical descriptor")
    pointer_bits = data["pointer_bits"]
    if type(pointer_bits) is not int or pointer_bits <= 0:
        raise ValueError("target.pointer_bits must be a positive integer")
    return CompilationTarget(
        target_id=_require_digest(data["target_id"], "target.target_id"),
        architecture=_require_string(data["architecture"], "target.architecture"),
        vendor=_require_string(data["vendor"], "target.vendor"),
        operating_system=_require_string(
            data["operating_system"], "target.operating_system"
        ),
        abi=_require_string(data["abi"], "target.abi"),
        endianness=_require_string(data["endianness"], "target.endianness"),
        pointer_bits=pointer_bits,
    )


def _parse_toolchain(value: Any) -> CompilationToolchain:
    fields = frozenset(
        {
            "build_system",
            "compiler_id",
            "compiler_version",
            "provider_kind",
            "target_triple",
            "toolchain_id",
        }
    )
    data = _require_object(value, "toolchain", fields)
    identity = {key: data[key] for key in fields - {"toolchain_id"}}
    if _digest(identity) != data["toolchain_id"]:
        raise ValueError("toolchain.toolchain_id does not match its descriptor")
    return CompilationToolchain(
        toolchain_id=_require_digest(data["toolchain_id"], "toolchain.toolchain_id"),
        provider_kind=_require_string(data["provider_kind"], "toolchain.provider_kind"),
        compiler_id=_require_string(data["compiler_id"], "toolchain.compiler_id"),
        compiler_version=_require_string(
            data["compiler_version"], "toolchain.compiler_version"
        ),
        target_triple=_require_string(data["target_triple"], "toolchain.target_triple"),
        build_system=_require_string(data["build_system"], "toolchain.build_system"),
    )


def _parse_source(value: Any, index: int) -> _SourceCompilation:
    field = f"sources[{index}]"
    fields = frozenset(
        {
            "closure_id",
            "compile_invocation",
            "compile_invocation_digest",
            "inputs",
            "object_digest",
            "source",
        }
    )
    data = _require_object(value, field, fields)
    invocation_value = data["compile_invocation"]
    if type(invocation_value) is not list or any(
        type(item) is not str or not item for item in invocation_value
    ):
        raise ValueError(f"{field}.compile_invocation must contain non-empty strings")
    invocation = tuple(invocation_value)
    if _digest(invocation) != data["compile_invocation_digest"]:
        raise ValueError(f"{field}.compile_invocation_digest does not match")
    inputs_value = data["inputs"]
    if type(inputs_value) is not list:
        raise ValueError(f"{field}.inputs must be an array")
    inputs = []
    for input_index, raw_input in enumerate(inputs_value):
        input_field = f"{field}.inputs[{input_index}]"
        input_data = _require_object(
            raw_input,
            input_field,
            frozenset({"content_digest", "input_kind", "uri"}),
        )
        inputs.append(
            CompilationInput(
                uri=_require_string(input_data["uri"], f"{input_field}.uri"),
                input_kind=_require_string(
                    input_data["input_kind"], f"{input_field}.input_kind"
                ),
                content_digest=_require_digest(
                    input_data["content_digest"], f"{input_field}.content_digest"
                ),
            )
        )
    if tuple(item.uri for item in inputs) != tuple(
        sorted({item.uri for item in inputs})
    ):
        raise ValueError(f"{field}.inputs must have unique, sorted URIs")
    source = _require_string(data["source"], f"{field}.source")
    if not any(item.uri == source and item.input_kind == "source" for item in inputs):
        raise ValueError(f"{field} has no owning source input")
    closure_value = {
        "inputs": tuple(
            {
                "content_digest": item.content_digest,
                "input_kind": item.input_kind,
                "uri": item.uri,
            }
            for item in inputs
        ),
        "invocation": invocation,
    }
    if _digest(closure_value) != data["closure_id"]:
        raise ValueError(f"{field}.closure_id does not match its inputs and invocation")
    return _SourceCompilation(
        closure_id=_require_digest(data["closure_id"], f"{field}.closure_id"),
        compile_invocation=invocation,
        compile_invocation_digest=_require_digest(
            data["compile_invocation_digest"], f"{field}.compile_invocation_digest"
        ),
        inputs=tuple(inputs),
        object_digest=_require_digest(data["object_digest"], f"{field}.object_digest"),
        source=source,
    )


def parse_compilation_manifest(
    value: Any, *, kernels: Sequence[KernelDescriptor] | None = None
) -> CompilationManifest:
    """Strictly validate and construct one compilation manifest."""

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
    data = _require_object(value, "manifest", fields)
    schema = _require_string(data["schema"], "manifest.schema")
    if schema != _MANIFEST_SCHEMA:
        raise ValueError(f"unsupported compilation manifest schema {schema!r}")
    receipt_schema = _require_string(data["receipt_schema"], "manifest.receipt_schema")
    if receipt_schema != _RECEIPT_SCHEMA:
        raise ValueError(f"unsupported compilation receipt schema {receipt_schema!r}")
    identity = {key: data[key] for key in fields - {"manifest_digest"}}
    if _digest(identity) != data["manifest_digest"]:
        raise ValueError("manifest.manifest_digest does not match its contents")
    target = _parse_target(data["target"])
    toolchain = _parse_toolchain(data["toolchain"])
    provider_kind = _require_string(data["provider_kind"], "manifest.provider_kind")
    if toolchain.provider_kind != provider_kind:
        raise ValueError("manifest provider does not match toolchain provider")
    sources_value = data["sources"]
    if type(sources_value) is not list:
        raise ValueError("manifest.sources must be an array")
    sources = tuple(
        _parse_source(item, index) for index, item in enumerate(sources_value)
    )
    by_source = {source.source: source for source in sources}
    if len(by_source) != len(sources):
        raise ValueError("manifest contains a duplicate owning source")
    kernel_values = tuple(
        kernels if kernels is not None else native_cpu_kernel_manifest()
    )
    kernel_sources = {kernel.owning_source for kernel in kernel_values}
    if not kernel_sources or "" in kernel_sources or kernel_sources != by_source.keys():
        raise ValueError(
            "compilation manifest sources do not exactly match native kernel metadata"
        )
    artifact_digest = _require_digest(
        data["artifact_digest"], "manifest.artifact_digest"
    )
    receipts = []
    for kernel in kernel_values:
        source = by_source[kernel.owning_source]
        receipt_value = {
            "closure_id": source.closure_id,
            "compile_invocation_digest": source.compile_invocation_digest,
            "kernel_id": kernel.kernel_id,
            "object_digest": source.object_digest,
            "provider_kind": provider_kind,
            "receipt_schema": receipt_schema,
            "target_id": target.target_id,
            "toolchain_id": toolchain.toolchain_id,
            "variant": kernel.variant,
        }
        receipts.append(
            CompilationReceipt(
                receipt_id=_digest(receipt_value),
                receipt_schema=receipt_schema,
                provider_kind=provider_kind,
                kernel=kernel,
                target=target,
                toolchain=toolchain,
                closure_id=source.closure_id,
                inputs=source.inputs,
                compile_invocation=source.compile_invocation,
                compile_invocation_digest=source.compile_invocation_digest,
                object_digest=source.object_digest,
                shared_artifact_digest=artifact_digest,
                framework_name=None,
                framework_version=None,
                specialization=MappingProxyType({}),
            )
        )
    return CompilationManifest(
        manifest_digest=_require_digest(
            data["manifest_digest"], "manifest.manifest_digest"
        ),
        schema=schema,
        provider_kind=provider_kind,
        artifact_digest=artifact_digest,
        target=target,
        toolchain=toolchain,
        receipts=tuple(sorted(receipts, key=lambda item: item.kernel.kernel_id)),
    )


@lru_cache(maxsize=1)
def load_compilation_manifest() -> CompilationManifest:
    """Load and strictly validate the installed native compilation manifest."""

    resource = resources.files("strideweave.verification").joinpath(
        "_native_provenance.json"
    )
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "installed StrideWeave build has no valid kernel compilation manifest"
        ) from error
    return parse_compilation_manifest(value)
