from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tools import generate_kernel_provenance as generator


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
