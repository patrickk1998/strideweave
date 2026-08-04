from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from typing import Any

import pytest

from strideweave.verification import (
    load_compilation_manifest,
    native_cpu_kernel_manifest,
)
from strideweave.verification.provenance import parse_compilation_manifest
from strideweave.verification.reporting import (
    _generic_oracle_input_uris,
    _generic_oracle_value,
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _raw_manifest() -> dict[str, Any]:
    resource = resources.files("strideweave.verification").joinpath(
        "_native_provenance.json"
    )
    value = json.loads(resource.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _rehash_source(source: dict[str, Any]) -> None:
    source["compile_invocation_digest"] = _digest(source["compile_invocation"])
    source["closure_id"] = _digest(
        {
            "inputs": source["inputs"],
            "invocation": source["compile_invocation"],
        }
    )


def _rehash_manifest(manifest: dict[str, Any]) -> None:
    manifest["manifest_digest"] = _digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )


def _source(manifest: dict[str, Any], suffix: str) -> dict[str, Any]:
    return next(
        source for source in manifest["sources"] if source["source"].endswith(suffix)
    )


def test_installed_manifest_covers_exactly_the_native_kernel_sources() -> None:
    manifest = load_compilation_manifest()
    kernels = native_cpu_kernel_manifest()

    assert {
        (receipt.kernel.kernel_id, receipt.kernel.variant)
        for receipt in manifest.receipts
    } == {(kernel.kernel_id, kernel.variant) for kernel in kernels}
    assert {receipt.kernel.owning_source for receipt in manifest.receipts} == {
        kernel.owning_source for kernel in kernels
    }
    assert all(receipt.inputs for receipt in manifest.receipts)
    assert all(receipt.framework_name is None for receipt in manifest.receipts)
    assert all(receipt.specialization == {} for receipt in manifest.receipts)


def test_operation_source_changes_only_its_own_closure() -> None:
    raw = _raw_manifest()
    original = parse_compilation_manifest(raw)
    changed = copy.deepcopy(raw)
    add = _source(changed, "/add.cpp")
    owning_input = next(
        item for item in add["inputs"] if item["input_kind"] == "source"
    )
    owning_input["content_digest"] = "f" * 64
    _rehash_source(add)
    _rehash_manifest(changed)

    updated = parse_compilation_manifest(changed)

    assert (
        updated.receipt_for("cpu.add", "default").closure_id
        != original.receipt_for("cpu.add", "default").closure_id
    )
    assert (
        updated.receipt_for("cpu.abs", "default").closure_id
        == original.receipt_for("cpu.abs", "default").closure_id
    )


def test_shared_header_changes_every_affected_closure() -> None:
    raw = _raw_manifest()
    original = parse_compilation_manifest(raw)
    changed = copy.deepcopy(raw)
    affected_sources = []
    for source in changed["sources"]:
        for item in source["inputs"]:
            if item["uri"].endswith("/_cpu_registry.hpp"):
                item["content_digest"] = "e" * 64
                affected_sources.append(source["source"])
                _rehash_source(source)
                break
    _rehash_manifest(changed)

    updated = parse_compilation_manifest(changed)

    assert affected_sources
    for receipt in original.receipts:
        if receipt.kernel.owning_source in affected_sources:
            assert (
                updated.receipt_for(
                    receipt.kernel.kernel_id, receipt.kernel.variant
                ).closure_id
                != receipt.closure_id
            )


def test_flags_and_toolchain_change_their_exact_identity_axes() -> None:
    raw = _raw_manifest()
    original = parse_compilation_manifest(raw)
    changed_flag = copy.deepcopy(raw)
    add = _source(changed_flag, "/add.cpp")
    add["compile_invocation"].append("-DSTRIDEWEAVE_TEST_FLAG=1")
    _rehash_source(add)
    _rehash_manifest(changed_flag)

    flag_manifest = parse_compilation_manifest(changed_flag)
    assert (
        flag_manifest.receipt_for("cpu.add", "default").receipt_id
        != original.receipt_for("cpu.add", "default").receipt_id
    )
    assert (
        flag_manifest.receipt_for("cpu.abs", "default").receipt_id
        == original.receipt_for("cpu.abs", "default").receipt_id
    )

    changed_toolchain = copy.deepcopy(raw)
    changed_toolchain["toolchain"]["compiler_version"] += " changed"
    changed_toolchain["toolchain"]["toolchain_id"] = _digest(
        {
            key: value
            for key, value in changed_toolchain["toolchain"].items()
            if key != "toolchain_id"
        }
    )
    _rehash_manifest(changed_toolchain)
    toolchain_manifest = parse_compilation_manifest(changed_toolchain)
    assert toolchain_manifest.toolchain.toolchain_id != original.toolchain.toolchain_id
    assert all(
        changed_receipt.receipt_id
        != original.receipt_for(
            changed_receipt.kernel.kernel_id, changed_receipt.kernel.variant
        ).receipt_id
        for changed_receipt in toolchain_manifest.receipts
    )


def test_commit_only_movement_is_absent_from_compilation_identity() -> None:
    raw = _raw_manifest()
    manifest = parse_compilation_manifest(raw)

    assert "commit" not in json.dumps(raw).lower()
    assert parse_compilation_manifest(copy.deepcopy(raw)) == manifest


def test_generic_oracle_closure_follows_transitive_implementation_imports() -> None:
    package = resources.files("strideweave")
    inputs = _generic_oracle_input_uris(package)

    assert "carriers/generic/helpers.py" in inputs
    assert "carriers/operation_capability.py" in inputs
    assert "carriers/operation_helpers.py" in inputs


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


@pytest.mark.parametrize("mutation", ["missing", "unknown", "forged"])
def test_malformed_or_incomplete_provenance_fails_closed(mutation: str) -> None:
    raw = _raw_manifest()
    if mutation == "missing":
        del raw["sources"][0]["inputs"]
    elif mutation == "unknown":
        raw["sources"][0]["unexpected"] = True
    else:
        raw["sources"][0]["closure_id"] = "0" * 64
    _rehash_manifest(raw)

    with pytest.raises(ValueError, match=r"fields do not match|closure_id"):
        parse_compilation_manifest(raw)
