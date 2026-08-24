from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from importlib import resources
from typing import Any

import pytest

import strideweave.verification.stage_two as stage_two
from strideweave.verification import (
    CompiledExecutableReceipt,
    LogicalKernel,
    kernel_manifest,
    verification_profile,
)
from strideweave.verification.provenance import (
    CompilationInput,
    GeneratedArtifact,
    SpecializationAxis,
    compilation_bundle_json_object,
    installed_compilation_bundle,
    make_compilation_bundle,
    make_compilation_runtime,
    make_compilation_target,
    make_compilation_toolchain,
    make_compiled_executable_receipt,
    make_jit_specialization_receipt,
    parse_compilation_bundle,
    parse_compilation_receipt,
)
from strideweave.verification.reporting import (
    _canonical_case_alias_uris,
    _generic_oracle_input_uris,
    _generic_oracle_value,
)

_SOURCE_DIGEST = "1" * 64
_OBJECT_DIGEST = "2" * 64
_EXECUTABLE_DIGEST = "3" * 64
_HOST_DIGEST = "4" * 64
_DEVICE_DIGEST = "5" * 64


def _target():
    return make_compilation_target(
        architecture="test-arch",
        vendor="test-vendor",
        operating_system="test-os",
        abi="test-abi",
        endianness="little",
        pointer_bits=64,
    )


def _toolchain(provider: str):
    return make_compilation_toolchain(
        provider=provider,
        compiler_id="test-compiler",
        compiler_version="1.0",
        target_triple="test-arch-test-os",
        build_system="test-build",
    )


def _runtime():
    return make_compilation_runtime({"runtime": "test", "version": "1"})


def _cpu_receipt():
    provider = "test-compiled-provider"
    return make_compiled_executable_receipt(
        profile_id="cpu-compiled",
        provider=provider,
        target=_target(),
        toolchain=_toolchain(provider),
        runtime=_runtime(),
        logical_kernel=LogicalKernel("cpu-compiled", "add", "cpu.add", "default"),
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


def _jit_receipt(*, block_size: int = 64):
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
        specialization_axes=(
            SpecializationAxis("block_size", block_size),
            SpecializationAxis("dtype", "Float32"),
        ),
        generated_host_source_artifact_ordinal=0,
        generated_device_source_artifact_ordinal=1,
        runtime_artifact_ordinals=(2,),
    )


def test_installed_bundle_covers_exactly_the_cpu_logical_manifest() -> None:
    profile = verification_profile("cpu-compiled")
    bundle = installed_compilation_bundle(profile)

    assert bundle == installed_compilation_bundle(profile)
    assert all(isinstance(item, CompiledExecutableReceipt) for item in bundle.receipts)
    assert {
        (
            receipt.logical_kernel.operation,
            receipt.logical_kernel.kernel_id,
            receipt.logical_kernel.variant,
        )
        for receipt in bundle.receipts
    } == {
        (kernel.operation, kernel.kernel_id, kernel.variant)
        for kernel in kernel_manifest(profile)
    }
    assert parse_compilation_bundle(compilation_bundle_json_object(bundle)) == bundle
    assert "kernel-compilation-manifest.v1" not in str(
        compilation_bundle_json_object(bundle)
    )
    with pytest.raises(FrozenInstanceError):
        bundle.bundle_id = "0" * 64  # type: ignore[misc]


def test_installed_bundle_requires_one_registered_profile() -> None:
    with pytest.raises(TypeError, match="VerificationProfile"):
        installed_compilation_bundle("cpu-compiled")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no pre-execution installed bundle"):
        installed_compilation_bundle(verification_profile("synthetic-jit"))


def test_neutral_provider_adapter_rebinds_before_runtime_preflight(monkeypatch) -> None:
    profile = verification_profile("synthetic-jit")
    receipt = _jit_receipt()
    baseline = stage_two._target_runtime(profile)
    preflight_calls = 0
    resolution_calls = 0

    def preflight() -> None:
        nonlocal preflight_calls
        preflight_calls += 1

    def current_compilation(selected_profile, receipts):
        nonlocal resolution_calls
        resolution_calls += 1
        assert selected_profile == profile
        return make_compilation_bundle(receipts)

    runtime = replace(
        baseline,
        preflight=preflight,
        current_compilation=current_compilation,
    )
    monkeypatch.setattr(stage_two, "_target_runtime", lambda selected: runtime)

    current = stage_two._current_profile_compilation_bundle(profile, (receipt,))

    assert current == make_compilation_bundle((receipt,))
    assert preflight_calls == 0
    assert resolution_calls == 1


def test_specialization_changes_receipt_identity_not_logical_scope() -> None:
    first = _jit_receipt(block_size=64)
    second = _jit_receipt(block_size=128)
    bundle = make_compilation_bundle((second, _cpu_receipt(), first))

    assert first.receipt_id != second.receipt_id
    assert first.logical_kernel == second.logical_kernel
    assert [item.kind for item in bundle.receipts] == [
        "compiled-executable",
        "jit-specialization",
        "jit-specialization",
    ]
    assert parse_compilation_bundle(compilation_bundle_json_object(bundle)) == bundle


def test_receipt_and_runtime_sequences_remain_deeply_immutable() -> None:
    receipt = _jit_receipt()

    with pytest.raises(ValueError, match="immutable tuple"):
        replace(receipt, inputs=list(receipt.inputs))
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(receipt.runtime, facts=list(receipt.runtime.facts))
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(
            make_compilation_bundle((receipt,)),
            receipts=[receipt],
        )


def test_receipts_exclude_local_paths_and_observation_only_facts() -> None:
    with pytest.raises(ValueError, match="path-free"):
        replace(_cpu_receipt(), compile_options=("-I/tmp/local-header",))
    with pytest.raises(ValueError, match="observation-only"):
        make_compilation_runtime({"cache_path": "provider-cache"})
    with pytest.raises(ValueError, match="observation-only"):
        SpecializationAxis("timestamp", "2026-08-19T00:00:00Z")


@pytest.mark.parametrize(
    ("declared", "inputs", "message"),
    [
        (
            ("provider://source/add.py", "provider://source/missing.py"),
            (
                CompilationInput(
                    0, "provider://source/add.py", "source", _SOURCE_DIGEST
                ),
            ),
            "missing",
        ),
        (
            (),
            (
                CompilationInput(
                    0, "provider://source/add.py", "source", _SOURCE_DIGEST
                ),
            ),
            "undeclared",
        ),
        (
            ("provider://header/add.hpp",),
            (
                CompilationInput(
                    0, "provider://header/add.hpp", "header", _SOURCE_DIGEST
                ),
            ),
            "owning source",
        ),
    ],
)
def test_receipt_rejects_incomplete_or_undeclared_input_closure(
    declared: tuple[str, ...], inputs: tuple[CompilationInput, ...], message: str
) -> None:
    provider = "test-compiled-provider"
    with pytest.raises(ValueError, match=message):
        make_compiled_executable_receipt(
            profile_id="cpu-compiled",
            provider=provider,
            target=_target(),
            toolchain=_toolchain(provider),
            runtime=_runtime(),
            logical_kernel=LogicalKernel("cpu-compiled", "add", "cpu.add", "default"),
            declared_input_uris=declared,
            inputs=inputs,
            compile_options=("-O2",),
            artifacts=(
                GeneratedArtifact(0, "compiled-object", _OBJECT_DIGEST),
                GeneratedArtifact(1, "shared-executable", _EXECUTABLE_DIGEST),
            ),
            compiled_object_artifact_ordinals=(0,),
            shared_executable_artifact_ordinal=1,
        )


def test_receipt_rejects_duplicate_ordinals_uris_and_local_paths() -> None:
    provider = "test-compiled-provider"
    common = {
        "profile_id": "cpu-compiled",
        "provider": provider,
        "target": _target(),
        "toolchain": _toolchain(provider),
        "runtime": _runtime(),
        "logical_kernel": LogicalKernel("cpu-compiled", "add", "cpu.add", "default"),
        "compile_options": ("-O2",),
        "artifacts": (
            GeneratedArtifact(0, "compiled-object", _OBJECT_DIGEST),
            GeneratedArtifact(1, "shared-executable", _EXECUTABLE_DIGEST),
        ),
        "compiled_object_artifact_ordinals": (0,),
        "shared_executable_artifact_ordinal": 1,
    }
    duplicate = (
        CompilationInput(0, "provider://source/add.cpp", "source", _SOURCE_DIGEST),
        CompilationInput(0, "provider://source/add2.cpp", "source", _SOURCE_DIGEST),
    )
    with pytest.raises(ValueError, match="unique contiguous canonical ordinals"):
        make_compiled_executable_receipt(
            declared_input_uris=(
                "provider://source/add.cpp",
                "provider://source/add2.cpp",
            ),
            inputs=duplicate,
            **common,
        )
    with pytest.raises(ValueError, match="duplicate URI"):
        make_compiled_executable_receipt(
            declared_input_uris=(
                "provider://source/add.cpp",
                "provider://source/add.cpp",
            ),
            inputs=(
                CompilationInput(
                    0, "provider://source/add.cpp", "source", _SOURCE_DIGEST
                ),
                CompilationInput(
                    1, "provider://source/add.cpp", "source", _SOURCE_DIGEST
                ),
            ),
            **common,
        )
    with pytest.raises(ValueError, match="stable path-free"):
        CompilationInput(0, "/tmp/add.cpp", "source", _SOURCE_DIGEST)


@pytest.mark.parametrize(
    ("artifacts", "host", "device", "runtime_ordinals", "message"),
    [
        (
            (
                GeneratedArtifact(0, "generated-device-source", _DEVICE_DIGEST),
                GeneratedArtifact(1, "runtime-executable", _EXECUTABLE_DIGEST),
            ),
            0,
            0,
            (1,),
            "host and device sources must be distinct",
        ),
        (
            (
                GeneratedArtifact(0, "generated-host-source", _HOST_DIGEST),
                GeneratedArtifact(1, "generated-device-source", _DEVICE_DIGEST),
                GeneratedArtifact(2, "runtime-executable", _EXECUTABLE_DIGEST),
            ),
            0,
            1,
            (),
            "unclaimed generated artifacts",
        ),
    ],
)
def test_jit_receipt_requires_generated_sources_and_every_runtime_artifact(
    artifacts: tuple[GeneratedArtifact, ...],
    host: int,
    device: int,
    runtime_ordinals: tuple[int, ...],
    message: str,
) -> None:
    provider = "test-jit-provider"
    with pytest.raises(ValueError, match=message):
        make_jit_specialization_receipt(
            profile_id="synthetic-jit",
            provider=provider,
            target=_target(),
            toolchain=_toolchain(provider),
            runtime=_runtime(),
            logical_kernel=LogicalKernel("synthetic-jit", "add", "cpu.add", "default"),
            declared_input_uris=("provider://source/add.py",),
            inputs=(
                CompilationInput(
                    0, "provider://source/add.py", "source", _SOURCE_DIGEST
                ),
            ),
            compile_options=("tilelang-opt=2",),
            artifacts=artifacts,
            specialization_axes=(SpecializationAxis("block_size", 64),),
            generated_host_source_artifact_ordinal=host,
            generated_device_source_artifact_ordinal=device,
            runtime_artifact_ordinals=runtime_ordinals,
        )


def test_receipt_parser_rejects_v2_unknown_crossed_and_forged_facts() -> None:
    raw = compilation_bundle_json_object(make_compilation_bundle((_jit_receipt(),)))
    receipt = raw["receipts"][0]

    v2 = copy.deepcopy(receipt)
    v2["schema_version"] = "strideweave.kernel-compilation-receipt.v2"
    with pytest.raises(ValueError, match="unsupported compilation receipt schema"):
        parse_compilation_receipt(v2, "receipt")

    unknown = copy.deepcopy(receipt)
    unknown["kind"] = "future-provider"
    with pytest.raises(ValueError, match="unknown discriminator"):
        parse_compilation_receipt(unknown, "receipt")

    crossed = copy.deepcopy(receipt)
    crossed["shared_executable_artifact_ordinal"] = 2
    with pytest.raises(ValueError, match=r"unexpected=.*shared_executable"):
        parse_compilation_receipt(crossed, "receipt")

    forged = copy.deepcopy(receipt)
    forged["receipt_id"] = "0" * 64
    with pytest.raises(ValueError, match="receipt_id does not match"):
        parse_compilation_receipt(forged, "receipt")


def test_bundle_parser_rejects_duplicate_receipts_and_noncanonical_order() -> None:
    cpu = _cpu_receipt()
    jit = _jit_receipt()
    raw = compilation_bundle_json_object(make_compilation_bundle((cpu, jit)))

    duplicate = copy.deepcopy(raw)
    duplicate["receipts"] = [duplicate["receipts"][0]] * 2
    with pytest.raises(ValueError, match="duplicate receipt"):
        parse_compilation_bundle(duplicate)

    reversed_bundle = copy.deepcopy(raw)
    reversed_bundle["receipts"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        parse_compilation_bundle(reversed_bundle)


def test_generic_oracle_closure_follows_transitive_implementation_imports() -> None:
    package = resources.files("strideweave")
    inputs = _generic_oracle_input_uris(package)

    assert "carriers/generic/helpers.py" in inputs
    assert "carriers/operation_capability.py" in inputs
    assert "carriers/operation_helpers.py" in inputs


def test_generic_oracle_closure_deduplicates_only_identical_case_aliases() -> None:
    contents = {
        "Alias.py": b"same",
        "alias.py": b"same",
        "Distinct.py": b"first",
        "distinct.py": b"second",
    }

    class Resource:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def read_bytes(self) -> bytes:
            return self._value

    class Package:
        def joinpath(self, *parts: str) -> Resource:
            return Resource(contents["/".join(parts)])

    assert _canonical_case_alias_uris(Package(), tuple(contents)) == (
        "Distinct.py",
        "alias.py",
        "distinct.py",
    )


def test_generic_oracle_reference_is_hash_seed_independent() -> None:
    command = (
        sys.executable,
        "-c",
        (
            "import json; "
            "from strideweave.verification.reporting import _generic_oracle_value; "
            "print(json.dumps(_generic_oracle_value(), sort_keys=True))"
        ),
    )
    values = []
    for seed in ("0", "1", "4", "7"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values.append(json.loads(process.stdout))

    assert all(value == values[0] for value in values[1:])


def test_generic_oracle_reference_changes_with_a_transitive_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _generic_oracle_value()
    package = resources.files("strideweave")

    class ChangedResource:
        def __init__(self, resource: Any, parts: tuple[str, ...] = ()) -> None:
            self._resource = resource
            self._parts = parts

        def joinpath(self, *descendants: str) -> ChangedResource:
            return ChangedResource(
                self._resource.joinpath(*descendants), self._parts + descendants
            )

        def is_file(self) -> bool:
            return self._resource.is_file()

        def read_text(self, *, encoding: str) -> str:
            return self._resource.read_text(encoding=encoding)

        def read_bytes(self) -> bytes:
            value = self._resource.read_bytes()
            if self._parts == ("carriers/generic/helpers.py",):
                return value + b"\n# changed\n"
            return value

    monkeypatch.setattr(
        "strideweave.verification.reporting.resources.files",
        lambda _package: ChangedResource(package),
    )

    assert (
        _generic_oracle_value()["oracle_reference_id"]
        != original["oracle_reference_id"]
    )
