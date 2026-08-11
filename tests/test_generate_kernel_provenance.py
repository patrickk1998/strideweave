from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest

from tools import generate_kernel_provenance as generator

# A compiler launcher such as ccache is invoked as "<launcher> <compiler> ...",
# so CMake writes it as argument zero and the real compiler becomes argument
# one. These tests execute the probe through a shim that merely execs its
# arguments, which reproduces that shape without making ccache a dependency.
_LAUNCHER = "strideweave-test-launcher"

# These tests are the only ones in the suite that spawn a real compiler, and in
# the native-sanitizers job they would inherit the preloaded sanitizer runtime
# and its options. A compiler is not instrumented, so that buys nothing, and it
# costs: the runtime would police the compiler's allocator, and any diagnostic
# it raised would abort the compiler under abort_on_error and land in the same
# report directory the job reserves for StrideWeave's own diagnostics, where the
# failure steps would print it as though StrideWeave had produced it. Clearing
# these from the environment the fixture leaves behind covers both the probe the
# generator runs and the compile the tests run, because neither passes an
# explicit env. This is a property of running the generator from inside the
# instrumented suite, not of the generator, which CMake invokes at build time
# with none of these set — so it is fixed here rather than in the tool.
_SANITIZER_ENVIRONMENT = (
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
    "ASAN_OPTIONS",
    "UBSAN_OPTIONS",
)


def _entry(
    source: Path, object_path: Path, *, dependency_file: Path | None = None
) -> dict[str, Any]:
    arguments = [
        "/toolchains/clang++",
        "-I",
        "/external/includes",
        "-c",
        str(source),
        "-o",
        str(object_path),
    ]
    if dependency_file is not None:
        arguments.extend(("-MD", "-MF", str(dependency_file)))
    return {
        "arguments": arguments,
        "directory": str(source.parent),
        "file": str(source),
    }


def test_source_closure_keeps_external_and_generated_headers_without_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    build = tmp_path / "build"
    external = tmp_path / "sdk" / "include"
    source = project / "ops" / "add.cpp"
    project_header = project / "include" / "shared.hpp"
    generated_header = build / "generated" / "config.hpp"
    external_header = external / "vendor.hpp"
    object_path = build / "add.o"
    for path, content in (
        (source, "source"),
        (project_header, "project"),
        (generated_header, "generated"),
        (external_header, "external"),
        (object_path, "object"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    dependencies = (source, project_header, generated_header, external_header)
    monkeypatch.setattr(generator, "discover_dependencies", lambda *_: dependencies)

    record = generator.build_source_record(
        entry=_entry(source, object_path),
        source=source,
        project_root=project,
        build_root=build,
        build_inputs=(),
    )
    raw_inputs = cast(tuple[dict[str, str], ...], record["inputs"])
    inputs = {item["input_kind"]: item for item in raw_inputs}

    assert inputs["header"]["uri"] == "include/shared.hpp"
    assert inputs["generated_header"]["uri"] == "cpp-build:///generated/config.hpp"
    assert inputs["external_header"]["uri"].startswith("cpp-external://sha256/")
    assert str(tmp_path) not in str(record)


def test_external_header_content_changes_the_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    build = tmp_path / "build"
    source = project / "ops" / "add.cpp"
    external_header = tmp_path / "sdk" / "include" / "vendor.hpp"
    object_path = build / "add.o"
    for path, content in (
        (source, "source"),
        (external_header, "before"),
        (object_path, "object"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        generator,
        "discover_dependencies",
        lambda *_: (source, external_header),
    )
    arguments = {
        "entry": _entry(source, object_path),
        "source": source,
        "project_root": project,
        "build_root": build,
        "build_inputs": (),
    }

    original = generator.build_source_record(**arguments)
    external_header.write_text("after", encoding="utf-8")
    changed = generator.build_source_record(**arguments)

    assert changed["closure_id"] != original["closure_id"]


def test_dependency_probe_requests_system_headers() -> None:
    arguments = generator._dependency_arguments(
        ["clang++", "-MMD", "-c", "input.cpp", "-o", "input.o"],
        Path("input.cpp"),
    )

    assert "-M" in arguments
    assert "-MM" not in arguments


class _LauncherProject(NamedTuple):
    compiler: str
    source: Path
    header: Path
    include: Path
    project: Path
    build: Path


@pytest.fixture
def launcher_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _LauncherProject:
    """Build a compilable source tree and put an exec-only launcher on PATH."""

    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("executing the dependency probe requires a C++ compiler")
    project = tmp_path / "project"
    build = tmp_path / "build"
    include = project / "include"
    for directory in (project / "ops", include, build):
        directory.mkdir(parents=True)
    header = include / "shared.hpp"
    header.write_text("#pragma once\nint shared_value();\n", encoding="utf-8")
    source = project / "ops" / "add.cpp"
    source.write_text(
        '#include "shared.hpp"\nint add() { return shared_value(); }\n',
        encoding="utf-8",
    )
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    launcher = launcher_directory / _LAUNCHER
    launcher.write_text('#!/bin/sh\nexec "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{launcher_directory}{os.pathsep}{os.environ['PATH']}")
    for variable in _SANITIZER_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    return _LauncherProject(
        compiler=compiler,
        source=source,
        header=header,
        include=include,
        project=project,
        build=build,
    )


def _compile_command(project: _LauncherProject, object_path: Path) -> list[str]:
    return [
        project.compiler,
        "-I",
        str(project.include),
        "-c",
        str(project.source),
        "-o",
        str(object_path),
    ]


def test_dependency_probe_keeps_a_launcher_and_its_compiler() -> None:
    source = Path("input.cpp")
    bare = generator._dependency_arguments(
        ["/usr/bin/c++", "-DX", "-c", "input.cpp", "-o", "input.o"], source
    )

    launched = generator._dependency_arguments(
        ["ccache", "/usr/bin/c++", "-DX", "-c", "input.cpp", "-o", "input.o"], source
    )

    assert bare[0] == "/usr/bin/c++"
    assert launched[0] == "ccache"
    assert launched[1] == "/usr/bin/c++"
    # The launcher only prepends: every other argument, and the appended probe
    # flags, are rebuilt exactly as they are for a bare compiler entry.
    assert launched == ["ccache", *bare]
    assert launched[-4:] == ["-M", "-MT", "strideweave-kernel", "input.cpp"]


def test_dependency_probe_keeps_a_launcher_from_a_command_string() -> None:
    source = Path("input.cpp")
    entry = {
        "command": "ccache /usr/bin/c++ -DX -c input.cpp -o input.o",
        "directory": ".",
        "file": "input.cpp",
    }

    launched = generator._dependency_arguments(
        generator._compile_arguments(entry), source
    )

    assert launched[0] == "ccache"
    assert launched[1] == "/usr/bin/c++"
    assert launched[-4:] == ["-M", "-MT", "strideweave-kernel", "input.cpp"]


def test_the_launcher_fixture_clears_an_inherited_sanitizer_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiler spawned by these tests must not inherit the runtime.

    The variables are set before the fixture is requested so that the assertion
    means something off the sanitizer job too; without that, a developer machine
    that never sets them would pass this vacuously.
    """

    for variable in _SANITIZER_ENVIRONMENT:
        monkeypatch.setenv(variable, "leaked-into-the-compiler")

    request.getfixturevalue("launcher_project")

    # Read them from a real child, the way the compiler would see them, rather
    # than from os.environ, which would not prove inheritance.
    report = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'printf "%s" "${LD_PRELOAD-}${DYLD_INSERT_LIBRARIES-}'
            '${ASAN_OPTIONS-}${UBSAN_OPTIONS-}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert report.stdout == ""


def test_a_launcher_prefixed_probe_discovers_the_same_dependencies(
    launcher_project: _LauncherProject,
) -> None:
    build = launcher_project.build
    directory = str(build)
    bare = _compile_command(launcher_project, build / "bare.o")
    launched = [_LAUNCHER, *_compile_command(launcher_project, build / "launched.o")]

    bare_dependencies = generator.discover_dependencies(
        {"arguments": bare, "directory": directory}, launcher_project.source
    )
    launched_dependencies = generator.discover_dependencies(
        {"arguments": launched, "directory": directory}, launcher_project.source
    )

    assert launcher_project.source.resolve() in bare_dependencies
    assert launcher_project.header.resolve() in bare_dependencies
    assert launched_dependencies == bare_dependencies


def _launcher_records(
    launcher_project: _LauncherProject,
) -> tuple[dict[str, object], dict[str, object]]:
    build = launcher_project.build
    bare = _compile_command(launcher_project, build / "bare.o")
    launched = [_LAUNCHER, *_compile_command(launcher_project, build / "launched.o")]
    records = []
    for arguments in (bare, launched):
        subprocess.run(arguments, check=True, cwd=build)
        records.append(
            generator.build_source_record(
                entry={"arguments": arguments, "directory": str(build)},
                source=launcher_project.source,
                project_root=launcher_project.project,
                build_root=build,
                build_inputs=(),
            )
        )
    return records[0], records[1]


def test_a_launcher_records_the_same_dependency_closure(
    launcher_project: _LauncherProject,
) -> None:
    bare_record, launched_record = _launcher_records(launcher_project)

    assert launched_record["inputs"] == bare_record["inputs"]
    assert launched_record["object_digest"] == bare_record["object_digest"]
    assert launched_record["source"] == bare_record["source"]


def test_a_launcher_moves_its_compiler_into_the_recorded_invocation(
    launcher_project: _LauncherProject,
) -> None:
    """Pin the one recorded axis a compiler launcher does move.

    Argument zero is the executable and is never recorded, so a bare entry omits
    its compiler while a launched entry omits the launcher and records the
    compiler it runs as an ordinary argument. The dependency closure is
    unaffected, but `closure_id` covers the invocation too, so adopting a
    launcher re-identifies every per-source closure once.
    """

    bare_record, launched_record = _launcher_records(launcher_project)
    bare_invocation = cast(tuple[str, ...], bare_record["compile_invocation"])
    launched_invocation = cast(tuple[str, ...], launched_record["compile_invocation"])
    compiler_token = f"${{EXTERNAL}}/{Path(launcher_project.compiler).name}"

    assert compiler_token not in bare_invocation
    assert launched_invocation == (compiler_token, *bare_invocation)
    assert _LAUNCHER not in launched_invocation
    assert launched_record["closure_id"] != bare_record["closure_id"]


def test_dependency_cache_reuses_unchanged_sources_and_invalidates_affected_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    build = tmp_path / "build"
    cache = build / "cache"
    shared = project / "include" / "shared.hpp"
    sources = (project / "ops" / "add.cpp", project / "ops" / "abs.cpp")
    objects = (build / "add.o", build / "abs.o")
    for path in (*sources, *objects, shared):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    calls: list[Path] = []

    def discover(_entry: dict[str, Any], source: Path) -> tuple[Path, ...]:
        calls.append(source)
        return source, shared

    monkeypatch.setattr(generator, "discover_dependencies", discover)

    def build_all() -> None:
        for source, object_path in zip(sources, objects, strict=True):
            generator.build_source_record(
                entry=_entry(source, object_path),
                source=source,
                project_root=project,
                build_root=build,
                build_inputs=(),
                cache_dir=cache,
                dependency_context={"compiler": "test"},
            )

    build_all()
    assert calls == list(sources)
    calls.clear()
    build_all()
    assert calls == []

    sources[0].write_text("changed source", encoding="utf-8")
    build_all()
    assert calls == [sources[0]]
    calls.clear()

    shared.write_text("changed shared header", encoding="utf-8")
    build_all()
    assert calls == list(sources)


def test_dependency_cache_follows_compile_depfiles_when_a_header_is_shadowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    build = tmp_path / "build"
    cache = build / "cache"
    earlier_include = project / "earlier"
    later_include = project / "later"
    selected_before = later_include / "shared.hpp"
    source = project / "ops" / "add.cpp"
    unrelated_source = project / "ops" / "abs.cpp"
    object_path = build / "add.o"
    unrelated_object = build / "abs.o"
    dependency_file = build / "add.o.d"
    unrelated_dependency_file = build / "abs.o.d"
    for path, content in (
        (source, "#include <shared.hpp>"),
        (unrelated_source, "unrelated"),
        (selected_before, "selected later header"),
        (object_path, "object before"),
        (unrelated_object, "unrelated object"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    dependency_file.write_text(f"add.o: {source} {selected_before}\n", encoding="utf-8")
    unrelated_dependency_file.write_text(
        f"abs.o: {unrelated_source}\n", encoding="utf-8"
    )
    calls: list[Path] = []

    def unexpected_discovery(_entry: dict[str, Any], owning_source: Path):
        calls.append(owning_source)
        raise AssertionError("compile depfiles should avoid dependency probing")

    monkeypatch.setattr(generator, "discover_dependencies", unexpected_discovery)
    arguments = {
        "project_root": project,
        "build_root": build,
        "build_inputs": (),
        "cache_dir": cache,
        "dependency_context": {"compiler": "test"},
    }
    original = generator.build_source_record(
        entry=_entry(source, object_path, dependency_file=dependency_file),
        source=source,
        **arguments,
    )
    unrelated_original = generator.build_source_record(
        entry=_entry(
            unrelated_source,
            unrelated_object,
            dependency_file=unrelated_dependency_file,
        ),
        source=unrelated_source,
        **arguments,
    )

    selected_after = earlier_include / "shared.hpp"
    selected_after.parent.mkdir(parents=True)
    selected_after.write_text("selected earlier header", encoding="utf-8")
    object_path.write_text("object after", encoding="utf-8")
    dependency_file.write_text(f"add.o: {source} {selected_after}\n", encoding="utf-8")
    changed = generator.build_source_record(
        entry=_entry(source, object_path, dependency_file=dependency_file),
        source=source,
        **arguments,
    )
    unrelated_changed = generator.build_source_record(
        entry=_entry(
            unrelated_source,
            unrelated_object,
            dependency_file=unrelated_dependency_file,
        ),
        source=unrelated_source,
        **arguments,
    )

    original_inputs = cast(tuple[dict[str, str], ...], original["inputs"])
    changed_inputs = cast(tuple[dict[str, str], ...], changed["inputs"])
    original_uris = {item["uri"] for item in original_inputs}
    changed_uris = {item["uri"] for item in changed_inputs}
    assert selected_before.relative_to(project).as_posix() in original_uris
    assert selected_after.relative_to(project).as_posix() in changed_uris
    assert selected_before.relative_to(project).as_posix() not in changed_uris
    assert changed["closure_id"] != original["closure_id"]
    assert unrelated_changed["closure_id"] == unrelated_original["closure_id"]
    assert calls == []
