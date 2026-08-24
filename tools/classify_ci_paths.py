"""Classify a GitHub compare payload for ordinary and block-device CI jobs."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

_NON_CODE_PREFIXES = (
    ".agents/",
    ".beads/",
    ".claude/",
    ".codex/",
    "assets/",
    "docs/",
    "openspec/",
)
_NON_CODE_FILES = frozenset(
    {
        ".nojekyll",
        "AGENTS.md",
        "CLAUDE.md",
        "CNAME",
        "CONTRIBUTING.md",
        "INVARIANTS.md",
        "SECURITY.md",
        "index.html",
        "llms.md",
        "properdocs.yml",
        "skills-lock.json",
    }
)

_BLOCK_DEVICE_FILES = frozenset(
    {
        ".github/workflows/ci.yml",
        "CMakeLists.txt",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/strideweave/__init__.py",
        "src/strideweave/_block_device.pyi",
        "src/strideweave/_carrier.pyi",
        "src/strideweave/carriers/__init__.py",
        "src/strideweave/carriers/__init__.pyi",
        "src/strideweave/carriers/_built_in_capabilities.py",
        "src/strideweave/carriers/base.py",
        "src/strideweave/functional/api.py",
        "src/strideweave/operation.py",
        "src/strideweave/operation.pyi",
        "tests/test_carrier.py",
        "tests/test_classify_ci_paths.py",
        "tests/test_dlpack.py",
        "tests/test_docstrings.py",
        "tests/test_dtype.py",
        "tests/test_evictable.py",
        "tests/test_move.py",
        "tests/test_operation.py",
        "tests/test_operation_capability.py",
        "tests/test_tensor.py",
        "tests/test_tensor_representation.py",
        "tools/classify_ci_paths.py",
        "uv.lock",
    }
)
_BLOCK_DEVICE_PREFIXES = (
    "src/strideweave/carriers/block_device/",
    "src/strideweave/carriers/cpu/",
    "src/strideweave/carriers/evictable/",
    "src/strideweave/carriers/move/",
    "tests/test_block_device",
)
_KNOWN_BLOCK_IRRELEVANT_PREFIXES = (
    "src/strideweave/einops/",
    "src/strideweave/nn/",
    "src/strideweave/verification/",
    "tests/test_einops",
    "tests/test_module",
    "tests/test_verification",
)


@dataclass(frozen=True)
class Classification:
    """Two independent job-gate decisions for one complete path set."""

    code: bool
    block_device: bool


def _is_code_path(path: str) -> bool:
    return not (
        path in _NON_CODE_FILES
        or any(path.startswith(prefix) for prefix in _NON_CODE_PREFIXES)
    )


def _is_block_device_path(path: str) -> bool:
    if path in _BLOCK_DEVICE_FILES or any(
        path.startswith(prefix) for prefix in _BLOCK_DEVICE_PREFIXES
    ):
        return True
    if not _is_code_path(path):
        return False
    if path == ".github/workflows/specs.yml" or any(
        path.startswith(prefix) for prefix in _KNOWN_BLOCK_IRRELEVANT_PREFIXES
    ):
        return False
    # An unclassified code or build path is uncertain. Run fail-open so a new
    # integration surface cannot silently bypass the real-device job.
    return True


def classify_paths(paths: Iterable[str]) -> Classification:
    """Classify every new and previous path from a nonempty comparison."""
    materialized = tuple(paths)
    if not materialized:
        raise ValueError("comparison contains no paths")
    if any(not isinstance(path, str) or not path for path in materialized):
        raise ValueError("comparison contains an unreadable path")
    return Classification(
        code=any(_is_code_path(path) for path in materialized),
        block_device=any(_is_block_device_path(path) for path in materialized),
    )


def classify_compare_payload(payload: Mapping[str, Any]) -> Classification:
    """Classify one complete GitHub compare response or reject uncertainty."""
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("compare payload has no readable file list")
    if not files or len(files) >= 300:
        raise ValueError("compare file list is empty or may be truncated")

    paths: list[str] = []
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("compare file entry is unreadable")
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("compare file entry has no filename")
        paths.append(filename)
        previous = entry.get("previous_filename")
        if previous is not None:
            if not isinstance(previous, str) or not previous:
                raise ValueError("compare rename has no readable previous filename")
            paths.append(previous)
    return classify_paths(paths)


def main(*, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    """Read one compare payload and emit a compact JSON classification."""
    try:
        payload = json.load(stdin)
        if not isinstance(payload, Mapping):
            raise ValueError("compare payload must be a JSON object")
        result = classify_compare_payload(payload)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"classification failed: {error}", file=sys.stderr)
        return 1
    json.dump(
        {"code": result.code, "block_device": result.block_device},
        stdout,
        separators=(",", ":"),
    )
    stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
