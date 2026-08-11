"""The boundary between instrumented native work and the real-Dolt suites.

CI runs the sanitizers over everything except ``dolt_integration`` and
``dolt_lifecycle``, so those two selections must contain no native work at all.
``conftest.native_work_forbidden`` makes that true by construction for every
marked item; these tests prove the guard itself covers the whole boundary and
actually refuses, and that the two selections stay disjoint and exhaustive.
"""

from __future__ import annotations

import importlib
import sys

import conftest
import pytest
import synthetic_evidence

import strideweave as sw
import strideweave.verification.store.recording as recording_module
from strideweave._carrier import (  # pyright: ignore[reportMissingModuleSource]
    _cpu_native_kernel_metadata as captured_native_kernel_metadata,
)
from strideweave.verification.classification import native_cpu_kernel_manifest
from strideweave.verification.provenance import load_compilation_manifest
from strideweave.verification.reporting import bind_report

_MARKERS = ("dolt_integration", "dolt_lifecycle")


def unrelated_callable() -> str:
    """Return a sentinel from a callable unrelated to native verification."""

    return "unrelated"


def test_every_binding_that_reaches_the_installed_build_is_guarded() -> None:
    # Discovery is by identity over imported modules, so a re-export under a new
    # name is guarded without being listed. This asserts the known import paths
    # are found, which is what fails if discovery itself regresses.
    importlib.import_module("strideweave.verification.status_cli")
    bindings = {(module.__name__, name) for module, name in conftest.native_bindings()}

    assert ("strideweave", "test_backend") in bindings
    assert ("strideweave.verification.api", "test_backend") in bindings
    assert ("strideweave.verification.store.querying", "test_backend") in bindings
    assert ("strideweave.verification.provenance", "load_compilation_manifest") in (
        bindings
    )
    assert ("strideweave.verification.reporting", "load_compilation_manifest") in (
        bindings
    )
    assert ("strideweave.verification.reporting", "bind_report") in bindings
    assert ("strideweave.verification.store.recording", "bind_report") in bindings
    assert (
        "strideweave.verification.classification",
        "native_cpu_kernel_manifest",
    ) in (bindings)
    for module_name, attribute in conftest.NATIVE_ENTRY_POINTS:
        assert (module_name, attribute) in bindings


def test_the_guard_refuses_every_native_entry_point_and_restores_it() -> None:
    original = sw.test_backend

    with conftest.native_work_forbidden() as bindings:
        assert bindings
        for module, name in bindings:
            with pytest.raises(AssertionError, match="must not run"):
                getattr(module, name)()
        assert recording_module.bind_report is not bind_report

    assert sw.test_backend is original
    assert recording_module.bind_report is bind_report
    assert (
        load_compilation_manifest
        is sys.modules["strideweave.verification.provenance"].load_compilation_manifest
    )


def test_the_guard_refuses_the_compiled_metadata_export_itself() -> None:
    # Guarding only the Python wrapper would leave the compiled export as a
    # second way into the same native work, so a marked test could read native
    # metadata while deselected from the sanitizers. The export is what the
    # wrapper calls, so this is the call that has to refuse.
    carrier = importlib.import_module("strideweave._carrier")
    original = carrier._cpu_native_kernel_metadata

    with conftest.native_work_forbidden():
        with pytest.raises(AssertionError, match="must not run"):
            carrier._cpu_native_kernel_metadata()

    assert carrier._cpu_native_kernel_metadata is original
    assert len(carrier._cpu_native_kernel_metadata()) == len(
        native_cpu_kernel_manifest()
    )


def test_the_guard_refuses_a_precaptured_test_module_alias() -> None:
    original = captured_native_kernel_metadata

    bindings = {(module.__name__, name) for module, name in conftest.native_bindings()}
    assert (__name__, "captured_native_kernel_metadata") in bindings

    with conftest.native_work_forbidden():
        with pytest.raises(AssertionError, match="must not run"):
            captured_native_kernel_metadata()
        assert unrelated_callable() == "unrelated"

    assert captured_native_kernel_metadata is original
    assert captured_native_kernel_metadata()


def test_the_guard_restores_precaptured_aliases_after_an_exception() -> None:
    original = captured_native_kernel_metadata

    def fail_inside_guard() -> None:
        with conftest.native_work_forbidden():
            raise RuntimeError("context body failed")

    with pytest.raises(RuntimeError, match="context body failed"):
        fail_inside_guard()

    assert captured_native_kernel_metadata is original
    assert captured_native_kernel_metadata()


def test_the_marked_facts_need_no_installed_build() -> None:
    # The synthetic facts the marked suites persist are what makes the guard
    # satisfiable, so they must build with every native entry point refused.
    with conftest.native_work_forbidden():
        report = synthetic_evidence.synthetic_report()
        changed = synthetic_evidence.contradicting_report(report)

    assert report.header is not None
    assert report.records
    assert changed.records != report.records
    manifest = recording_module._report_manifest(report)
    assert manifest.target.architecture == synthetic_evidence.ARCHITECTURE


def test_the_two_ci_selections_are_disjoint_exhaustive_and_derived() -> None:
    # Every collected test, not only the half this run selected, so the claim
    # that the markers partition the suite is checked against all of it.
    items = conftest.COLLECTED_ITEMS
    marked = [
        item for item in items if any(marker in item.keywords for marker in _MARKERS)
    ]

    assert items
    assert [item for item in items if all(m in item.keywords for m in _MARKERS)] == []
    for item in items:
        fixtures = frozenset(getattr(item, "fixturenames", ()))
        needs_session_server = bool(fixtures & conftest._REAL_DOLT_FIXTURES)
        owns_servers = "dolt_lifecycle" in item.keywords
        assert ("dolt_integration" in item.keywords) == (
            needs_session_server and not owns_servers
        )
        # This is what keeps native work out of the uninstrumented selection at
        # collection time rather than only when a marked test happens to run.
        assert not (item in marked and "backend_report" in fixtures)


def test_a_process_that_escaped_the_sanitizer_runtime_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parallel sanitizer job must not degrade quietly to a plain test run."""

    config = pytest.Config.fromdictargs({}, [])
    monkeypatch.setenv("STRIDEWEAVE_EXPECT_SANITIZERS", "1")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setattr(conftest, "sanitizer_runtime_loaded", lambda: False)

    with pytest.raises(pytest.UsageError, match="gw3 is not running under"):
        conftest.pytest_configure(config)

    monkeypatch.setattr(conftest, "sanitizer_runtime_loaded", lambda: True)
    conftest.pytest_configure(config)


def test_the_sanitizer_guard_is_inert_outside_the_sanitizer_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every other job runs uninstrumented and must be unaffected."""

    config = pytest.Config.fromdictargs({}, [])
    monkeypatch.delenv("STRIDEWEAVE_EXPECT_SANITIZERS", raising=False)
    monkeypatch.setattr(conftest, "sanitizer_runtime_loaded", lambda: False)

    conftest.pytest_configure(config)
