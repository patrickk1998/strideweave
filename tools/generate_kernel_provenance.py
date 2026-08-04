"""Generate deterministic native C++ per-source compilation provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "strideweave.kernel-compilation-manifest.v1"
_CACHE_SCHEMA = "strideweave.kernel-provenance-dependency-cache.v2"


def canonical_json(value: object) -> str:
    """Encode one identity value using the repository canonical JSON form."""

    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def digest_value(value: object) -> str:
    """Return the SHA-256 digest of one canonical identity value."""

    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def digest_file(path: Path) -> str:
    """Return the SHA-256 digest of one compilation input or artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compile_arguments(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if type(arguments) is list and all(type(value) is str for value in arguments):
        return arguments
    command = entry.get("command")
    if type(command) is str:
        return shlex.split(command)
    raise ValueError("compile command must contain string arguments or command")


def _find_compile_entry(
    compile_commands: list[dict[str, Any]], source: Path
) -> dict[str, Any]:
    matches = []
    for entry in compile_commands:
        directory = Path(entry.get("directory", "."))
        candidate = Path(entry.get("file", ""))
        if not candidate.is_absolute():
            candidate = directory / candidate
        if candidate.resolve() == source.resolve():
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            f"source {source} must have exactly one compile command, found {len(matches)}"
        )
    return matches[0]


def _object_path(arguments: list[str], directory: Path) -> Path:
    for index, argument in enumerate(arguments):
        if argument == "-o" and index + 1 < len(arguments):
            result = Path(arguments[index + 1])
            return result if result.is_absolute() else directory / result
        if argument.startswith("/Fo") and len(argument) > 3:
            result = Path(argument[3:])
            return result if result.is_absolute() else directory / result
    raise ValueError("compile command has no object output")


def _dependency_arguments(arguments: list[str], source: Path) -> list[str]:
    result = [arguments[0]]
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-o", "-MF", "-MT", "-MQ"}:
            skip_next = True
            continue
        if argument == "-c" or argument in {"-M", "-MM", "-MD", "-MMD", "-MP"}:
            continue
        if argument.startswith(("/Fo", "-Wp,-M")):
            continue
        try:
            if Path(argument).resolve() == source.resolve():
                continue
        except OSError:
            pass
        result.append(argument)
    result.extend(("-M", "-MT", "strideweave-kernel", str(source)))
    return result


def _parse_dependency_rule(
    value: str, *, directory: Path, source: Path
) -> tuple[Path, ...]:
    flattened = value.replace("\\\n", " ")
    separator = flattened.find(":")
    if separator < 0:
        raise ValueError(f"compiler emitted no dependency rule for {source}")
    dependencies = []
    for value in shlex.split(flattened[separator + 1 :]):
        dependency = Path(value)
        if not dependency.is_absolute():
            dependency = directory / dependency
        dependencies.append(dependency.resolve())
    if source.resolve() not in dependencies:
        raise ValueError(f"compiler dependency rule omitted owning source {source}")
    return tuple(dict.fromkeys(dependencies))


def discover_dependencies(entry: dict[str, Any], source: Path) -> tuple[Path, ...]:
    """Ask the compiler for the source's complete transitive dependency set."""

    arguments = _compile_arguments(entry)
    directory = Path(entry["directory"])
    completed = subprocess.run(
        _dependency_arguments(arguments, source),
        cwd=directory,
        capture_output=True,
        check=True,
        text=True,
    )
    return _parse_dependency_rule(completed.stdout, directory=directory, source=source)


def _dependency_file_path(entry: dict[str, Any]) -> Path | None:
    arguments = _compile_arguments(entry)
    directory = Path(entry["directory"])
    for index, argument in enumerate(arguments[1:], start=1):
        if argument == "-MF" and index + 1 < len(arguments):
            result = Path(arguments[index + 1])
            return result if result.is_absolute() else directory / result
        if argument.startswith("-MF") and len(argument) > 3:
            result = Path(argument[3:])
            return result if result.is_absolute() else directory / result
        for prefix in ("-Wp,-MD,", "-Wp,-MMD,"):
            if argument.startswith(prefix):
                result = Path(argument[len(prefix) :])
                return result if result.is_absolute() else directory / result
    return None


def _compiled_dependencies(
    entry: dict[str, Any], source: Path
) -> tuple[Path, ...] | None:
    dependency_file = _dependency_file_path(entry)
    if dependency_file is None or not dependency_file.is_file():
        return None
    return _parse_dependency_rule(
        dependency_file.read_text(encoding="utf-8"),
        directory=Path(entry["directory"]),
        source=source,
    )


def _dependency_cache_path(cache_dir: Path, source: Path, project_root: Path) -> Path:
    relative = source.resolve().relative_to(project_root).as_posix()
    return cache_dir / f"{hashlib.sha256(relative.encode()).hexdigest()}.json"


def _cached_dependencies(
    *,
    entry: dict[str, Any],
    source: Path,
    project_root: Path,
    cache_dir: Path,
    dependency_context: object,
) -> tuple[Path, ...]:
    # A depfile records the dependencies selected by the actual compilation and
    # therefore detects include-search shadowing even when every previously
    # cached input is unchanged. Some build systems consume or remove depfiles;
    # the fallback cache is then reusable only while both its inputs and the
    # compiled object receipt remain unchanged. Any recompilation re-runs the
    # compiler dependency probe, even when it emits byte-identical object code.
    if (compiled := _compiled_dependencies(entry, source)) is not None:
        return compiled
    cache_path = _dependency_cache_path(cache_dir, source, project_root)
    arguments = _compile_arguments(entry)
    object_path = _object_path(arguments, Path(entry["directory"]))
    object_stat = object_path.stat()
    signature = digest_value(
        {
            "arguments": arguments,
            "context": dependency_context,
            "directory": str(Path(entry["directory"]).resolve()),
            "object": {
                "content_digest": digest_file(object_path),
                "modified_ns": object_stat.st_mtime_ns,
                "size": object_stat.st_size,
            },
        }
    )
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            type(cached) is not dict
            or set(cached) != {"dependencies", "schema", "signature", "source"}
            or cached["schema"] != _CACHE_SCHEMA
            or cached["signature"] != signature
            or cached["source"] != str(source.resolve())
            or type(cached["dependencies"]) is not list
        ):
            raise ValueError
        dependencies = []
        for item in cached["dependencies"]:
            if (
                type(item) is not dict
                or set(item) != {"content_digest", "path"}
                or type(item["content_digest"]) is not str
                or type(item["path"]) is not str
            ):
                raise ValueError
            path = Path(item["path"])
            if not path.is_file() or digest_file(path) != item["content_digest"]:
                raise ValueError
            dependencies.append(path.resolve())
        if source.resolve() not in dependencies:
            raise ValueError
        return tuple(dependencies)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        dependencies = discover_dependencies(entry, source)

    cache_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "dependencies": [
            {"content_digest": digest_file(path), "path": str(path.resolve())}
            for path in dependencies
        ],
        "schema": _CACHE_SCHEMA,
        "signature": signature,
        "source": str(source.resolve()),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_dir,
        prefix=f".{cache_path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(canonical_json(value) + "\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, cache_path)
    return dependencies


def _normalized_token(token: str, project_root: Path, build_root: Path) -> str:
    normalized = token.replace(str(project_root), "${PROJECT}").replace(
        str(build_root), "${BUILD}"
    )
    if normalized != token:
        return normalized
    external_prefixes = (
        "--sysroot=",
        "-idirafter",
        "-iquote",
        "-isysroot",
        "-isystem",
        "-I",
        "-F",
    )
    for prefix in external_prefixes:
        if token.startswith(prefix):
            suffix = token[len(prefix) :]
            if suffix and Path(suffix).is_absolute():
                return f"{prefix}${{EXTERNAL}}/{Path(suffix).name}"
    if Path(token).is_absolute():
        return f"${{EXTERNAL}}/{Path(token).name}"
    return normalized


def _normalized_invocation(
    entry: dict[str, Any], project_root: Path, build_root: Path
) -> tuple[str, ...]:
    arguments = _compile_arguments(entry)
    normalized = []
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument == "-o":
            skip_next = True
            continue
        if argument.startswith("/Fo"):
            continue
        normalized.append(_normalized_token(argument, project_root, build_root))
    return tuple(normalized)


def _input_record(
    path: Path,
    project_root: Path,
    build_root: Path,
    *,
    kind: str,
) -> dict[str, str]:
    resolved = path.resolve()
    digest = digest_file(resolved)
    try:
        uri = resolved.relative_to(project_root).as_posix()
    except ValueError:
        try:
            relative = resolved.relative_to(build_root).as_posix()
        except ValueError:
            # Compiler and SDK installations move between otherwise equivalent
            # build hosts. External inputs therefore use a content-addressed,
            # provider-owned URI rather than persisting an unstable local path.
            uri = f"cpp-external://sha256/{digest}/{resolved.name}"
        else:
            uri = f"cpp-build:///{relative}"
    return {
        "content_digest": digest,
        "input_kind": kind,
        "uri": uri,
    }


def build_source_record(
    *,
    entry: dict[str, Any],
    source: Path,
    project_root: Path,
    build_root: Path,
    build_inputs: tuple[Path, ...],
    cache_dir: Path | None = None,
    dependency_context: object = None,
) -> dict[str, object]:
    """Build one content-addressed per-source closure from compiler dependencies."""

    dependencies = (
        discover_dependencies(entry, source)
        if cache_dir is None
        else _cached_dependencies(
            entry=entry,
            source=source,
            project_root=project_root,
            cache_dir=cache_dir,
            dependency_context=dependency_context,
        )
    )
    input_by_uri: dict[str, dict[str, str]] = {}
    for dependency in dependencies:
        if dependency == source.resolve():
            kind = "source"
        else:
            try:
                dependency.relative_to(project_root)
            except ValueError:
                try:
                    dependency.relative_to(build_root)
                except ValueError:
                    kind = "external_header"
                else:
                    kind = "generated_header"
            else:
                kind = "header"
        record = _input_record(
            dependency,
            project_root,
            build_root,
            kind=kind,
        )
        input_by_uri[record["uri"]] = record
    for build_input in build_inputs:
        record = _input_record(
            build_input,
            project_root,
            build_root,
            kind="build_input",
        )
        input_by_uri[record["uri"]] = record
    inputs = tuple(input_by_uri[key] for key in sorted(input_by_uri))
    invocation = _normalized_invocation(entry, project_root, build_root)
    closure_value = {"inputs": inputs, "invocation": invocation}
    arguments = _compile_arguments(entry)
    object_path = _object_path(arguments, Path(entry["directory"]))
    if not object_path.is_file():
        raise ValueError(f"compiled object is missing for {source}: {object_path}")
    return {
        "closure_id": digest_value(closure_value),
        "compile_invocation": invocation,
        "compile_invocation_digest": digest_value(invocation),
        "inputs": inputs,
        "object_digest": digest_file(object_path),
        "source": source.resolve().relative_to(project_root).as_posix(),
    }


def generate_manifest(arguments: argparse.Namespace) -> dict[str, object]:
    """Generate the complete deterministic manifest selected by CLI arguments."""

    project_root = arguments.project_root.resolve()
    compile_commands_path = arguments.compile_commands.resolve()
    build_root = compile_commands_path.parent
    raw_commands = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    if type(raw_commands) is not list or any(
        type(entry) is not dict for entry in raw_commands
    ):
        raise ValueError("compile_commands.json must contain an array of objects")
    compile_commands: list[dict[str, Any]] = raw_commands
    compiler = Path(arguments.compiler).resolve()
    compiler_version = subprocess.run(
        (str(compiler), "--version"), capture_output=True, check=True, text=True
    ).stdout.splitlines()[0]
    target_triple = subprocess.run(
        (str(compiler), "-dumpmachine"), capture_output=True, check=True, text=True
    ).stdout.strip()
    dependency_context = {
        "compiler_id": arguments.compiler_id,
        "compiler_version": compiler_version,
        "target_triple": target_triple,
    }
    sources = tuple(sorted((path.resolve() for path in arguments.source), key=str))
    source_records = tuple(
        build_source_record(
            entry=_find_compile_entry(compile_commands, source),
            source=source,
            project_root=project_root,
            build_root=build_root,
            build_inputs=tuple(arguments.build_input),
            cache_dir=arguments.cache_dir,
            dependency_context=dependency_context,
        )
        for source in sources
    )
    target = {
        "abi": arguments.abi,
        "architecture": arguments.architecture,
        "endianness": arguments.endianness,
        "operating_system": arguments.operating_system,
        "pointer_bits": arguments.pointer_bits,
        "vendor": arguments.vendor,
    }
    target["target_id"] = digest_value(target)
    toolchain = {
        "build_system": f"cmake-{arguments.cmake_version}",
        "compiler_id": arguments.compiler_id,
        "compiler_version": compiler_version,
        "provider_kind": "cpp-cmake",
        "target_triple": target_triple,
    }
    toolchain["toolchain_id"] = digest_value(toolchain)
    manifest: dict[str, object] = {
        "artifact_digest": digest_file(arguments.artifact),
        "provider_kind": "cpp-cmake",
        "receipt_schema": "strideweave.kernel-compilation-receipt.v1",
        "schema": SCHEMA,
        "sources": source_records,
        "target": target,
        "toolchain": toolchain,
    }
    manifest["manifest_digest"] = digest_value(manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--cmake-version", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--operating-system", required=True)
    parser.add_argument("--vendor", default="unknown")
    parser.add_argument("--abi", default="unknown")
    parser.add_argument("--endianness", choices=("little", "big"), required=True)
    parser.add_argument("--pointer-bits", type=int, required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--build-input", action="append", type=Path, default=[])
    return parser


def main() -> int:
    """Generate the manifest selected by command-line arguments."""

    arguments = _parser().parse_args()
    manifest = generate_manifest(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
