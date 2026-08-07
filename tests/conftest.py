"""Shared fixtures for tests that exercise a real local Dolt store.

Every ``dolt_integration`` test in one session shares a single managed server
rooted at one data directory and works in its own freshly created database, so
the suite proves real Dolt behavior without paying for a server per test. The
``dolt_lifecycle`` tests are the deliberate exception: they own the manager
itself and start, and may concurrently run, several independent servers.

The marker itself is derived rather than written by hand: a test is a real-Dolt
test exactly when it asks for one of the fixtures below. That keeps the two CI
partitions exact, because a test cannot leave the sanitizer-instrumented suite
without also asking for the external Dolt server it was deselected for.

Because the marked selection is the one CI does not instrument, it must also
contain no native work. ``native_work_forbidden`` enforces that at runtime: for
the whole of a marked item — its fixture setup, call, and teardown — every
module binding that reaches ``sw.test_backend``, the compiled kernel metadata
export, the installed compilation manifest, or report binding refuses to run.
That includes module-level aliases captured by tests and fixtures before the
context opens. References held only in local variables, closures, or object
state are outside this pragmatic accidental-use guard; it is not an adversarial
sandbox. Marked tests therefore use the deterministic pure-Python facts in
``synthetic_evidence``, and the native construction and validation of a report
stays in the unmarked, sanitizer-instrumented suite.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
import synthetic_evidence

import strideweave as sw
import strideweave.verification.store.querying as querying_module
from strideweave.verification import VerificationReport
from strideweave.verification.store.server import (
    ManagedDoltServer,
    acquire_server,
    shutdown_servers,
)

_LEGACY_STORE_NAME = "legacy-verification-store"
_REAL_DOLT_FIXTURES = frozenset(
    {
        "dolt_data_directory",
        "dolt_server",
        "evidence_store_path",
        "legacy_evidence_store_path",
    }
)
_UNINSTRUMENTED_MARKERS = frozenset({"dolt_integration", "dolt_lifecycle"})
COLLECTED_ITEMS: list[pytest.Item] = []
"""Every item collected this session, before ``-m`` deselects a selection."""
# What reads the installed native build. The compiled metadata export is listed
# alongside its Python wrapper because the wrapper is not the only way to reach
# it: calling the export directly is native work too, so the boundary holds only
# when the export itself refuses. Every other name that reaches native
# provenance does so through one of these.
NATIVE_ENTRY_POINTS = (
    ("strideweave._carrier", "_cpu_native_kernel_metadata"),
    ("strideweave.verification.api", "test_backend"),
    ("strideweave.verification.classification", "native_cpu_kernel_manifest"),
    ("strideweave.verification.provenance", "load_compilation_manifest"),
    ("strideweave.verification.reporting", "_raw_compilation_manifest"),
    ("strideweave.verification.reporting", "bind_report"),
)


def native_bindings() -> tuple[tuple[ModuleType, str], ...]:
    """Return every imported module attribute bound to a native entry point.

    Re-exports and module-level aliases in test or fixture namespaces are found
    by identity rather than by name, so a new import path for one of these
    functions is guarded without being listed anywhere.

    Returns:
        Each ``(module, attribute)`` pair that currently reaches native work.
    """

    targets = [
        getattr(importlib.import_module(module), attribute)
        for module, attribute in NATIVE_ENTRY_POINTS
    ]
    bindings = []
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        for name, value in tuple(vars(module).items()):
            if any(value is target for target in targets):
                bindings.append((module, name))
    return tuple(bindings)


@contextmanager
def native_work_forbidden() -> Iterator[tuple[tuple[ModuleType, str], ...]]:
    """Make every binding that reaches the installed native build refuse to run."""

    bindings = native_bindings()
    saved = [(module, name, getattr(module, name)) for module, name in bindings]

    def refuse(name: str) -> Callable[..., object]:
        def refused(*arguments: object, **keywords: object) -> object:
            raise AssertionError(
                f"{name} reaches the installed native build and must not run in a "
                "dolt_integration or dolt_lifecycle test, which CI does not "
                "instrument; use the pure-Python facts in synthetic_evidence"
            )

        return refused

    for module, name in bindings:
        setattr(module, name, refuse(f"{module.__name__}.{name}"))
    try:
        yield bindings
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark exactly the tests that need the real session Dolt server.

    ``fixturenames`` is the complete closure of what a test resolves, so a
    fixture that itself depends on the session server marks its users too. A
    test that owns server processes of its own already carries
    ``dolt_lifecycle`` and is left alone, which keeps the two selections
    disjoint.

    The complete list is kept because ``-m`` removes the other selection from
    the session afterwards, and whether the two selections partition every test
    is a fact about all of them rather than about the half that ran.
    """

    COLLECTED_ITEMS.clear()
    COLLECTED_ITEMS.extend(items)
    for item in items:
        if "dolt_lifecycle" in item.keywords:
            continue
        if _REAL_DOLT_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.dolt_integration)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Iterator[None]:
    """Forbid native work for the whole of an uninstrumented item's protocol.

    Wrapping the protocol rather than the call covers fixture setup, so a
    higher-scoped fixture first resolved for a marked test cannot smuggle
    native work past the boundary either.
    """

    if _UNINSTRUMENTED_MARKERS.isdisjoint(item.keywords):
        return (yield)
    with native_work_forbidden():
        return (yield)


@pytest.fixture(scope="session")
def backend_report() -> VerificationReport:
    """Return this build's own report, produced by running local verification.

    Constructing it is native work, so only unmarked tests may ask for it.
    """

    return sw.test_backend()


@pytest.fixture(scope="session")
def synthetic_report() -> VerificationReport:
    """Return the deterministic pure-Python report the marked suites persist."""

    return synthetic_evidence.synthetic_report()


@pytest.fixture
def synthetic_current_facts(
    synthetic_report: VerificationReport, monkeypatch: pytest.MonkeyPatch
) -> VerificationReport:
    """Make staleness and coverage queries compare against the synthetic facts.

    ``query_stale`` and ``query_todo`` are defined against whatever this build
    currently verifies. Substituting that one source keeps the comparison
    itself real while leaving the local verification run — which is native
    work — to the instrumented suite.
    """

    monkeypatch.setattr(querying_module, "_current_report", lambda: synthetic_report)
    return synthetic_report


def _require_dolt() -> None:
    """Skip the calling test when no local Dolt runtime is installed."""

    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


def _live_dolt_server_count() -> int:
    """Return how many live ``dolt sql-server`` children this process has."""

    listing = subprocess.run(
        ("ps", "-A", "-o", "pid=,ppid=,args="),
        capture_output=True,
        check=True,
        text=True,
    )
    count = 0
    for line in listing.stdout.splitlines():
        pieces = line.split(maxsplit=2)
        if len(pieces) != 3 or not pieces[1].isdigit():
            continue
        if int(pieces[1]) == os.getpid() and " sql-server" in f" {pieces[2]}":
            count += 1
    return count


@pytest.fixture
def live_dolt_server_count() -> Callable[[], int]:
    """Return a counter of this process's live ``dolt sql-server`` children."""

    return _live_dolt_server_count


@pytest.fixture(scope="session")
def dolt_data_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the one directory whose databases the session server serves."""

    return tmp_path_factory.mktemp("dolt-evidence")


@pytest.fixture(scope="session")
def legacy_evidence_store_path(dolt_data_directory: Path) -> Path:
    """Return a store an earlier StrideWeave version created with ``dolt init``.

    A running server only serves the databases it found when it started, which
    is exactly how a user's existing store looks: already on disk before
    StrideWeave starts anything. The session server therefore depends on this
    fixture so the store exists first.
    """

    _require_dolt()
    path = dolt_data_directory / _LEGACY_STORE_NAME
    path.mkdir(parents=True)
    subprocess.run(
        (
            "dolt",
            "init",
            "--name",
            "StrideWeave Verification Store",
            "--email",
            "verification-store@strideweave.invalid",
        ),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return path


@pytest.fixture(scope="session")
def dolt_server(
    dolt_data_directory: Path, legacy_evidence_store_path: Path
) -> Iterator[ManagedDoltServer]:
    """Return the single managed server shared by every integration test."""

    _require_dolt()
    assert legacy_evidence_store_path.is_dir()
    server = acquire_server(dolt_data_directory)
    yield server
    shutdown_servers((dolt_data_directory,))


@pytest.fixture
def evidence_store_path(
    dolt_server: ManagedDoltServer,
    dolt_data_directory: Path,
    request: pytest.FixtureRequest,
) -> Iterator[Callable[..., Path]]:
    """Return a factory of unique store paths served by the session server.

    Each path names a database that does not exist yet, so a test still
    exercises real creation and migration. Every database a test claims is
    dropped afterwards, which is what keeps the shared server's state isolated
    between tests.
    """

    claimed: list[str] = []
    prefix = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in request.node.name
    )[:64]

    def make(label: str = "store") -> Path:
        name = f"{prefix}-{label}-{len(claimed)}"
        claimed.append(name)
        return dolt_data_directory / name

    yield make

    connection = dolt_server.connect()
    try:
        with connection.cursor() as cursor:
            for name in claimed:
                cursor.execute(f"DROP DATABASE IF EXISTS `{name}`")
        connection.commit()
    finally:
        connection.close()
    for name in claimed:
        shutil.rmtree(dolt_data_directory / name, ignore_errors=True)
