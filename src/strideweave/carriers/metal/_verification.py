"""Private Metal verification identity and TileLang receipt provider."""

from __future__ import annotations

import hashlib
import platform
import struct
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ._jit import (
    CompiledMetalKernel,
    MetalSpecializationKey,
    observe_compilations,
)
from ._jit import (
    LogicalKernel as MetalLogicalKernel,
)
from ._runtime import load_metal_runtime

_PROFILE_ID = "metal-tilelang"
_PROVIDER = "tilelang.JITKernel"

_POINTWISE_OPERATIONS = frozenset(
    {
        "abs",
        "add",
        "ceil",
        "clamp",
        "cos",
        "div",
        "elementwise_mul",
        "elu",
        "eq",
        "erf",
        "exp",
        "exp2",
        "floor",
        "gelu",
        "le",
        "leaky_relu",
        "log",
        "log2",
        "logical_not",
        "lt",
        "maximum",
        "minimum",
        "mul",
        "ne",
        "neg",
        "pow",
        "recip",
        "relu",
        "rem",
        "round",
        "rsqrt",
        "select",
        "sigmoid",
        "sign",
        "silu",
        "sin",
        "softplus",
        "sqrt",
        "sub",
        "tanh",
    }
)
_REDUCTION_OPERATIONS = frozenset(
    {
        "argmax",
        "argmin",
        "cumsum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
    }
)
_INDEXING_OPERATIONS = frozenset(
    {
        "_sort_indices",
        "_sort_values",
        "_topk_indices",
        "_topk_values",
        "gather",
        "scatter",
        "scatter_add",
    }
)
_CONTRACTION_VARIANTS = {
    "conv_general": "conv_forward",
    "matmul": "matmul_forward",
}

_FAMILY_SOURCE_FILES = {
    "metal.pointwise": ("_executor.py",),
    "metal.reduction": ("_kernel_support.py", "_reduction_executor.py"),
    "metal.scan": ("_kernel_support.py", "_reduction_executor.py"),
    "metal.indexing": ("_kernel_support.py", "_indexing_executor.py"),
    "metal.selection": ("_kernel_support.py", "_indexing_executor.py"),
    "metal.matmul": ("_kernel_support.py", "_contraction_executor.py"),
    "metal.conv_general": ("_kernel_support.py", "_contraction_executor.py"),
}


@dataclass(frozen=True, slots=True)
class MetalCompilation:
    """One exact specialization selected by a Metal execution."""

    key: MetalSpecializationKey
    compiled: CompiledMetalKernel


def _expected_metal_logical_kernel(operation: str) -> MetalLogicalKernel:
    if operation == "validate_indices":
        return MetalLogicalKernel("metal.indexing", "validate_indices")
    if operation in _POINTWISE_OPERATIONS:
        return MetalLogicalKernel("metal.pointwise", operation)
    if operation in _REDUCTION_OPERATIONS:
        family = "metal.scan" if operation == "cumsum" else "metal.reduction"
        return MetalLogicalKernel(family, operation)
    if operation in _INDEXING_OPERATIONS:
        family = (
            "metal.selection"
            if operation.startswith(("_sort", "_topk"))
            else "metal.indexing"
        )
        return MetalLogicalKernel(family, operation)
    try:
        variant = _CONTRACTION_VARIANTS[operation]
    except KeyError as error:
        raise ValueError(
            f"Metal operation {operation!r} has no verification logical kernel"
        ) from error
    family = "metal.matmul" if operation == "matmul" else "metal.conv_general"
    return MetalLogicalKernel(family, variant)


def metal_kernel_metadata() -> tuple[tuple[str, str, str], ...]:
    """Return the finite Metal verification manifest without loading its runtime."""
    from .capabilities import metal_capabilities

    operations = sorted({capability.operation for capability in metal_capabilities()})
    return tuple(
        (
            operation,
            _expected_metal_logical_kernel(operation).operation,
            _expected_metal_logical_kernel(operation).variant,
        )
        for operation in operations
    )


@contextmanager
def capture_metal_compilations() -> Iterator[list[MetalCompilation]]:
    """Capture each exact specialization selected inside the context."""
    captured: list[MetalCompilation] = []

    def record(key: MetalSpecializationKey, compiled: CompiledMetalKernel) -> None:
        captured.append(MetalCompilation(key, compiled))

    with observe_compilations(record):
        yield captured


def _source_inputs(family: str):
    from strideweave.verification.provenance import CompilationInput

    try:
        family_files = _FAMILY_SOURCE_FILES[family]
    except KeyError as error:
        raise ValueError(f"unknown Metal JIT source family {family!r}") from error
    names = ("_jit.py", "_address_plan.py", *family_files)
    unique_names = tuple(dict.fromkeys(names))
    root = Path(__file__).parent
    inputs = []
    for ordinal, name in enumerate(unique_names):
        path = root / name
        try:
            contents = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Metal JIT provenance cannot read required source {name!r}"
            ) from error
        inputs.append(
            CompilationInput(
                ordinal,
                f"strideweave://source/carriers/metal/{name}",
                "source" if name == family_files[-1] else "python-jit-input",
                hashlib.sha256(contents).hexdigest(),
            )
        )
    return tuple(inputs)


def _target():
    from strideweave.verification.provenance import make_compilation_target

    return make_compilation_target(
        architecture=platform.machine(),
        vendor="Apple",
        operating_system=f"macOS-{platform.mac_ver()[0] or platform.release()}",
        abi="metal-mps-torch",
        endianness=sys.byteorder,
        pointer_bits=struct.calcsize("P") * 8,
    )


def _toolchain(tilelang_version: str):
    from strideweave.verification.provenance import make_compilation_toolchain

    return make_compilation_toolchain(
        provider=_PROVIDER,
        compiler_id="tilelang-metal",
        compiler_version=tilelang_version,
        target_triple=f"{platform.machine()}-apple-macos-metal",
        build_system="tilelang-jit",
    )


def _runtime(tilelang_version: str, torch_version: str):
    from strideweave.verification.provenance import make_compilation_runtime

    return make_compilation_runtime(
        {
            "execution_backend": "torch",
            "python": platform.python_version(),
            "tilelang": tilelang_version,
            "torch": torch_version,
        }
    )


def metal_compilation_receipt(logical_kernel, compilation: MetalCompilation):
    """Bind one observed TileLang specialization to a strict schema-v3 receipt."""
    from strideweave.verification.classification import LogicalKernel
    from strideweave.verification.provenance import (
        GeneratedArtifact,
        SpecializationAxis,
        make_jit_specialization_receipt,
    )

    if not isinstance(logical_kernel, LogicalKernel):
        raise TypeError("logical_kernel must be a verification LogicalKernel")
    if logical_kernel.profile_id != _PROFILE_ID:
        raise ValueError("Metal receipt logical kernel has the wrong profile")
    if not isinstance(compilation, MetalCompilation):
        raise TypeError("compilation must be a MetalCompilation")
    validate_metal_compilation(compilation)
    expected = _expected_metal_logical_kernel(logical_kernel.operation)
    if compilation.key.logical_kernel != expected:
        raise ValueError(
            "observed TileLang specialization does not match the verification "
            "logical kernel"
        )
    if (logical_kernel.kernel_id, logical_kernel.variant) != (
        expected.operation,
        expected.variant,
    ):
        raise ValueError("Metal verification manifest disagrees with JIT identity")

    facts = compilation.compiled.facts
    inputs = _source_inputs(expected.operation)
    artifacts = [
        GeneratedArtifact(0, "generated-host-source", facts.host_source_digest),
        GeneratedArtifact(1, "generated-device-source", facts.device_source_digest),
    ]
    runtime_ordinals: tuple[int, ...] = ()
    if facts.runtime_artifact_digest is not None:
        artifacts.append(
            GeneratedArtifact(
                2,
                "runtime-executable",
                facts.runtime_artifact_digest,
                facts.runtime_artifact_digest,
            )
        )
        runtime_ordinals = (2,)
    return make_jit_specialization_receipt(
        profile_id=_PROFILE_ID,
        provider=_PROVIDER,
        target=_target(),
        toolchain=_toolchain(facts.tilelang_version),
        runtime=_runtime(facts.tilelang_version, facts.torch_version),
        logical_kernel=logical_kernel,
        declared_input_uris=tuple(item.uri for item in inputs),
        inputs=inputs,
        compile_options=("execution-backend=torch", "target=metal"),
        artifacts=tuple(artifacts),
        specialization_axes=tuple(
            SpecializationAxis(name, value) for name, value in compilation.key.axes
        ),
        generated_host_source_artifact_ordinal=0,
        generated_device_source_artifact_ordinal=1,
        runtime_artifact_ordinals=runtime_ordinals,
    )


def metal_observed_compilation_receipt(compilation: MetalCompilation):
    """Bind any supported case-local Metal launch to its exact JIT receipt."""
    from strideweave.verification.classification import LogicalKernel

    if not isinstance(compilation, MetalCompilation):
        raise TypeError("compilation must be a MetalCompilation")
    validate_metal_compilation(compilation)
    selected = compilation.key.logical_kernel
    operation = None
    if selected == _expected_metal_logical_kernel("validate_indices"):
        operation = "validate_indices"
    else:
        for candidate, kernel_id, variant in metal_kernel_metadata():
            if (kernel_id, variant) == (selected.operation, selected.variant):
                operation = candidate
                break
    if operation is None:
        raise ValueError(
            "observed TileLang specialization has no verification receipt identity"
        )
    logical_kernel = LogicalKernel(
        _PROFILE_ID,
        operation,
        selected.operation,
        selected.variant,
    )
    return metal_compilation_receipt(logical_kernel, compilation)


def validate_metal_compilation(compilation: MetalCompilation) -> None:
    """Validate provider facts and the source closure before one JIT launch."""
    if not isinstance(compilation, MetalCompilation):
        raise TypeError("compilation must be a MetalCompilation")
    facts = compilation.compiled.facts
    if facts.provider != _PROVIDER or facts.target != "metal":
        raise ValueError("TileLang compilation facts name an unexpected provider")
    if facts.execution_backend != "torch":
        raise ValueError("TileLang compilation facts name an unexpected backend")
    _source_inputs(compilation.key.logical_kernel.operation)


def metal_compilation_matches(logical_kernel, compilation: MetalCompilation) -> bool:
    """Report whether one observed specialization implements one manifest kernel."""
    from strideweave.verification.classification import LogicalKernel

    if not isinstance(logical_kernel, LogicalKernel):
        raise TypeError("logical_kernel must be a verification LogicalKernel")
    if not isinstance(compilation, MetalCompilation):
        raise TypeError("compilation must be a MetalCompilation")
    return compilation.key.logical_kernel == _expected_metal_logical_kernel(
        logical_kernel.operation
    )


def require_metal_verification_runtime() -> None:
    """Fail before target evidence when the optional Metal runtime is unavailable."""
    load_metal_runtime()


def synchronize_metal_verification() -> None:
    """Synchronize pending target work before verification decodes results."""
    load_metal_runtime().synchronize()


def current_metal_compilation_bundle(profile, receipts: Sequence[object]):
    """Regenerate current JIT facts for observed Metal receipts without launch."""
    from strideweave.verification.classification import VerificationProfile
    from strideweave.verification.provenance import (
        JITSpecializationReceipt,
        make_compilation_bundle,
    )

    from ._jit import (
        CanonicalValue,
        MetalSpecializationKey,
        recompile_specialization,
        validate_specialization_recipe,
    )

    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    if profile.profile_id != _PROFILE_ID:
        raise ValueError("Metal compilation resolver received another profile")
    keys = []
    for receipt in receipts:
        if not isinstance(receipt, JITSpecializationReceipt):
            raise ValueError("Metal compilation bundle requires JIT receipts")
        if receipt.profile_id != _PROFILE_ID or receipt.provider != _PROVIDER:
            raise ValueError("Metal JIT receipt names an unexpected profile/provider")
        expected_kernel = _expected_metal_logical_kernel(
            receipt.logical_kernel.operation
        )
        if (receipt.logical_kernel.kernel_id, receipt.logical_kernel.variant) != (
            expected_kernel.operation,
            expected_kernel.variant,
        ):
            raise ValueError("Metal JIT receipt logical kernel is inconsistent")
        key = MetalSpecializationKey(
            expected_kernel,
            tuple(
                (axis.name, cast(CanonicalValue, axis.value))
                for axis in receipt.specialization_axes
            ),
        )
        validate_specialization_recipe(key)
        keys.append((receipt, key))
    runtime = load_metal_runtime()
    tilelang_version = str(runtime.tilelang.__version__)
    torch_version = str(runtime.torch.__version__)
    expected_runtime = _runtime(tilelang_version, torch_version)
    expected_target = _target()
    expected_toolchain = _toolchain(tilelang_version)
    regenerated = []
    for receipt, key in keys:
        compiled = recompile_specialization(key)
        current = metal_compilation_receipt(
            receipt.logical_kernel,
            MetalCompilation(key, compiled),
        )
        if (
            current.runtime != expected_runtime
            or current.target != expected_target
            or current.toolchain != expected_toolchain
        ):
            raise ValueError("current Metal JIT runtime facts are inconsistent")
        regenerated.append(current)
    return make_compilation_bundle(tuple(regenerated))


__all__: list[str] = []
