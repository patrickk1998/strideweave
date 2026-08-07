from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import strideweave.verification.store.server as server_module
from strideweave.verification.store import VerificationStoreError
from strideweave.verification.store.server import (
    ManagedDoltServer,
    acquire_server,
    shutdown_servers,
    started_server_directories,
    validate_runtime,
)

_DETACHED_EXIT_SETTLE_SECONDS = 1.0


def _require_dolt() -> None:
    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


@pytest.fixture(autouse=True)
def _stop_servers_started_by_a_test() -> Iterator[None]:
    """Stop only what a test started, leaving the shared session server alone."""

    before = set(started_server_directories())
    yield
    shutdown_servers(set(started_server_directories()) - before)


# One process starts a managed server, forks, and lets the child exit normally
# so its inherited interpreter-exit cleanup runs. The parent then reports
# whether its own server and private state survived that child's exit.
_FORK_SCRIPT = '''
"""Start a managed server, fork, and report what each process observed."""

import json
import os
import sys
from pathlib import Path

from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
)
from strideweave.verification.store.server import (
    acquire_server,
    started_server_directories,
)

ANSWER = SQLStatement("SELECT 1 AS answer")

data_directory = Path(sys.argv[1])
child_data_directory = Path(sys.argv[2])
child_report = Path(sys.argv[3])
parent_report = Path(sys.argv[4])

server = acquire_server(data_directory)
private_directory = server._private_directory
assert private_directory is not None
store = DoltEvidenceStore(data_directory / "verification-store")
store.initialize()
inherited_session = store._session()

if os.fork() == 0:
    observed = {}
    try:
        observed["inherited_owned"] = server.is_owned()
        observed["inherited_running"] = server.is_running()
        observed["inherited_registry"] = [
            str(entry) for entry in started_server_directories()
        ]
        try:
            server.connect()
            observed["inherited_connection"] = "opened"
        except VerificationStoreError as error:
            observed["inherited_connection"] = str(error)
        try:
            session = store._session()
            observed["inherited_session"] = (
                "reused" if session is inherited_session else "reopened"
            )
        except VerificationStoreError as error:
            observed["inherited_session"] = f"refused: {error}"
        own_store = DoltEvidenceStore(child_data_directory / "verification-store")
        own_store.initialize()
        observed["own_answer"] = [dict(row) for row in own_store.query(ANSWER)]
        own = acquire_server(child_data_directory)
        observed["own_running"] = own.is_running()
        observed["own_private_directory"] = str(own._private_directory)
        observed["own_process_id"] = own.process_id
    except BaseException as error:
        observed["failure"] = f"{type(error).__name__}: {error}"
    child_report.write_text(json.dumps(observed), encoding="utf-8")
    # Exit normally, so the inherited interpreter-exit cleanup runs here.
    sys.exit(0)

_, status = os.waitpid(-1, 0)
parent_report.write_text(
    json.dumps(
        {
            "child_status": status,
            "private_directory": str(private_directory),
            "private_directory_exists": private_directory.is_dir(),
            "server_running": server.is_running(),
            "server_process_id": server.process_id,
            "answer": [dict(row) for row in store.query(ANSWER)],
        }
    ),
    encoding="utf-8",
)
'''


def _initialized_store(directory: Path) -> Path:
    directory.mkdir(parents=True)
    subprocess.run(
        (
            "dolt",
            "init",
            "--name",
            "StrideWeave Verification Store",
            "--email",
            "verification-store@strideweave.invalid",
        ),
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return directory


def _child_process_ids() -> set[int]:
    """Return the live child process identifiers of this test process.

    The listing command is itself a child of this process while it runs, so it
    excludes its own identifier from the answer.
    """

    listing = subprocess.Popen(
        ("ps", "-A", "-o", "pid=,ppid="), stdout=subprocess.PIPE, text=True
    )
    output, _ = listing.communicate()
    assert listing.returncode == 0
    children: set[int] = set()
    for line in output.splitlines():
        pieces = line.split()
        if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
            if int(pieces[1]) == os.getpid() and int(pieces[0]) != listing.pid:
                children.add(int(pieces[0]))
    return children


def _fake_dolt(directory: Path, sql_server_body: str) -> str:
    """Install a stand-in runtime that reports a supported version."""

    executable = directory / "dolt"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "version" ]; then\n'
        '  echo "dolt version 1.40.0"\n'
        "  exit 0\n"
        "fi\n"
        f"{sql_server_body}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return str(executable)


@pytest.mark.dolt_lifecycle
def test_repeated_acquisitions_of_one_data_directory_share_a_single_server(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")

    first = acquire_server(data_directory)
    second = acquire_server(data_directory)
    third = acquire_server(data_directory / "." / ".." / "data")

    assert first is second is third
    assert first.is_running()
    server_children = {
        child for child in _child_process_ids() if child == first.process_id
    }
    assert server_children == {first.process_id}


@pytest.mark.dolt_lifecycle
def test_distinct_data_directories_are_served_independently(tmp_path: Path) -> None:
    _require_dolt()
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    _initialized_store(first_directory / "verification-store")
    _initialized_store(second_directory / "verification-store")

    first = acquire_server(first_directory)
    second = acquire_server(second_directory)

    assert first is not second
    assert first.port != second.port
    assert first.process_id != second.process_id
    assert first.is_running() and second.is_running()


@pytest.mark.dolt_lifecycle
def test_concurrent_acquisition_starts_exactly_one_server(tmp_path: Path) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    acquired: list[ManagedDoltServer] = []
    barrier = threading.Barrier(4)

    def acquire() -> None:
        barrier.wait()
        acquired.append(acquire_server(data_directory))

    threads = [threading.Thread(target=acquire) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(acquired) == 4
    assert len({id(server) for server in acquired}) == 1
    assert len({server.process_id for server in acquired}) == 1


@pytest.mark.dolt_lifecycle
def test_a_shutdown_during_startup_never_returns_an_unregistered_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    interrupted: list[ManagedDoltServer] = []

    class ShutdownWhileStarting(ManagedDoltServer):
        """Release the first server from the registry just before it starts."""

        def ensure_started(self) -> None:
            if not interrupted:
                interrupted.append(self)
                shutdown_servers((data_directory,))
            super().ensure_started()

    monkeypatch.setattr(server_module, "ManagedDoltServer", ShutdownWhileStarting)
    before = _child_process_ids()

    server = acquire_server(data_directory)

    assert interrupted and server is not interrupted[0]
    assert interrupted[0].is_retired()
    assert not interrupted[0].is_running()
    assert server.is_running()
    assert data_directory.resolve() in started_server_directories()
    assert _child_process_ids() - before == {server.process_id}


def test_a_failed_startup_cannot_be_restarted_by_a_waiting_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    monkeypatch.setattr(server_module, "_READINESS_TIMEOUT_SECONDS", 1.0)
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    executable = _fake_dolt(tmp_path, "sleep 30\n")
    created: list[ManagedDoltServer] = []

    class RecordingServer(ManagedDoltServer):
        """Expose the server objects one acquisition creates."""

        def __init__(self, directory: Path, runtime: str):
            super().__init__(directory, runtime)
            created.append(self)

    monkeypatch.setattr(server_module, "ManagedDoltServer", RecordingServer)
    before = _child_process_ids()

    with pytest.raises(VerificationStoreError, match="did not become ready"):
        acquire_server(data_directory, executable=executable)

    assert len(created) == 1
    assert created[0].is_retired()
    assert data_directory.resolve() not in started_server_directories()
    assert _child_process_ids() - before == set()
    # A caller that was waiting on this object while it failed reaches it only
    # now, and is refused rather than starting a server nothing tracks.
    with pytest.raises(VerificationStoreError, match="stopped before it started"):
        created[0].ensure_started()
    assert _child_process_ids() - before == set()


@pytest.mark.dolt_lifecycle
def test_concurrent_acquisition_and_shutdown_leave_no_untracked_server(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    before = _child_process_ids()
    acquired: list[ManagedDoltServer] = []
    failures: list[BaseException] = []
    barrier = threading.Barrier(2)

    def acquire() -> None:
        barrier.wait()
        try:
            acquired.append(acquire_server(data_directory))
        except BaseException as error:
            failures.append(error)

    for _ in range(3):
        thread = threading.Thread(target=acquire)
        thread.start()
        barrier.wait()
        shutdown_servers((data_directory,))
        thread.join()

    shutdown_servers((data_directory,))

    assert not failures
    assert len(acquired) == 3
    assert _child_process_ids() - before == set()


@pytest.mark.dolt_lifecycle
def test_a_data_directory_stays_unavailable_until_its_shutdown_completes(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    resolved = data_directory.resolve()
    before = _child_process_ids()
    stopped = acquire_server(data_directory)
    stopped_process_id = stopped.process_id
    begun = threading.Event()
    release = threading.Event()
    complete_shutdown = stopped.shutdown

    def paused_shutdown() -> None:
        """Hold the directory after shutdown began but before termination."""

        begun.set()
        release.wait(timeout=60.0)
        complete_shutdown()

    stopped.shutdown = paused_shutdown  # type: ignore[method-assign]
    acquired: list[ManagedDoltServer] = []
    failures: list[BaseException] = []

    def acquire() -> None:
        try:
            acquired.append(acquire_server(data_directory))
        except BaseException as error:  # pragma: no cover - reported by the test.
            failures.append(error)

    stopper = threading.Thread(target=lambda: shutdown_servers((data_directory,)))
    acquirer = threading.Thread(target=acquire)
    stopper.start()
    try:
        assert begun.wait(timeout=30.0)
        # The registry no longer serves this directory, but the process that
        # holds its database locks has not exited, so nothing may replace it.
        assert resolved not in started_server_directories()
        acquirer.start()
        acquirer.join(timeout=2.0)

        assert acquirer.is_alive()
        assert acquired == [] and failures == []
        assert stopped.is_running()
        assert _child_process_ids() - before == {stopped_process_id}
    finally:
        release.set()
        stopper.join(timeout=90.0)
        if acquirer.ident is not None:
            acquirer.join(timeout=90.0)

    assert not stopper.is_alive() and not acquirer.is_alive()
    assert failures == []
    assert len(acquired) == 1
    replacement = acquired[0]
    assert replacement is not stopped
    assert replacement.is_running()
    assert not stopped.is_running()
    assert resolved in started_server_directories()
    assert _child_process_ids() - before == {replacement.process_id}


def _wait_until_gone(process_id: int) -> bool:
    """Report whether one process is gone, allowing for a brief exit delay."""

    deadline = time.monotonic() + 10.0
    while True:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@pytest.mark.dolt_lifecycle
def test_a_forked_child_neither_stops_nor_reuses_inherited_server_state(
    tmp_path: Path,
) -> None:
    _require_dolt()
    if not hasattr(os, "fork"):
        pytest.skip("this platform does not fork")
    script = tmp_path / "fork_lifecycle.py"
    script.write_text(_FORK_SCRIPT, encoding="utf-8")
    data_directory = tmp_path / "parent-data"
    child_data_directory = tmp_path / "child-data"
    data_directory.mkdir()
    child_data_directory.mkdir()
    parent_report = tmp_path / "parent.json"
    child_report = tmp_path / "child.json"
    environment = dict(os.environ)
    package_root = str(Path(server_module.__file__).resolve().parents[3])
    environment["PYTHONPATH"] = os.pathsep.join(
        piece for piece in (package_root, environment.get("PYTHONPATH")) if piece
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            str(data_directory),
            str(child_data_directory),
            str(child_report),
            str(parent_report),
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    child = json.loads(child_report.read_text(encoding="utf-8"))
    parent = json.loads(parent_report.read_text(encoding="utf-8"))
    assert "failure" not in child, child.get("failure")
    # The child owns none of what it inherited, so it cannot mistake the
    # parent's server or session for transport of its own.
    assert child["inherited_owned"] is False
    assert child["inherited_running"] is False
    assert child["inherited_registry"] == []
    assert "is not running" in child["inherited_connection"]
    assert child["inherited_session"] != "reused"
    # Meanwhile the parent's own server survived that child's normal exit.
    assert parent["child_status"] == 0
    assert parent["private_directory_exists"] is True
    assert parent["server_running"] is True
    assert parent["answer"] == [{"answer": 1}]
    # The child's own state is supported and is cleaned up by its own exit.
    assert child["own_answer"] == [{"answer": 1}]
    assert child["own_running"] is True
    assert not Path(child["own_private_directory"]).exists()
    assert _wait_until_gone(child["own_process_id"])
    # The parent's exit still stops its server and removes its private state.
    assert not Path(parent["private_directory"]).exists()
    assert _wait_until_gone(parent["server_process_id"])


@pytest.mark.dolt_lifecycle
def test_a_served_database_answers_parameterized_queries(tmp_path: Path) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    server = acquire_server(data_directory)
    hostile = "value'); DROP TABLE t; --\\"

    connection = server.connect("verification-store")
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE t (a INT PRIMARY KEY, b VARCHAR(128))")
            cursor.execute("INSERT INTO t VALUES (%s, %s)", (1, hostile))
            connection.commit()
            cursor.execute("SELECT b FROM t WHERE a = %s", (1,))
            rows = cursor.fetchall()
    finally:
        connection.close()

    assert rows == ((hostile,),)


@pytest.mark.dolt_lifecycle
def test_one_server_serves_several_isolated_databases(tmp_path: Path) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "first-store")
    _initialized_store(data_directory / "second-store")
    server = acquire_server(data_directory)

    connection = server.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            databases = {str(row[0]) for row in cursor.fetchall()}
    finally:
        connection.close()

    assert {"first-store", "second-store"} <= databases


@pytest.mark.dolt_lifecycle
def test_teardown_leaves_no_server_child_and_refuses_further_connections(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    server = acquire_server(data_directory)
    process_id = server.process_id

    shutdown_servers((data_directory,))

    assert not server.is_running()
    assert process_id not in _child_process_ids()
    with pytest.raises(VerificationStoreError, match="server is not running"):
        server.connect()
    assert acquire_server(data_directory) is not server


@pytest.mark.dolt_lifecycle
def test_teardown_removes_the_private_state_and_leaves_no_detached_child(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")
    server = acquire_server(data_directory)
    private_directory = server._private_directory
    assert private_directory is not None and private_directory.is_dir()

    shutdown_servers((data_directory,))

    # A detached exit child of the server outlives the process it was started
    # from and recreates this directory shortly after teardown removed it, so
    # the answer is read once that window has passed rather than immediately.
    time.sleep(_DETACHED_EXIT_SETTLE_SECONDS)
    assert not private_directory.exists()


def test_a_server_that_exits_during_startup_reports_its_own_output(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    executable = _fake_dolt(
        tmp_path,
        'echo "cannot open database directory" >&2\nexit 1\n',
    )

    with pytest.raises(
        VerificationStoreError, match="exited during startup"
    ) as failure:
        acquire_server(data_directory, executable=executable)

    assert "cannot open database directory" in str(failure.value)


def test_a_server_that_never_becomes_ready_reports_its_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    monkeypatch.setattr(server_module, "_READINESS_TIMEOUT_SECONDS", 1.0)
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    executable = _fake_dolt(
        tmp_path, 'echo "waiting on an unavailable lock" >&2\nsleep 30\n'
    )

    with pytest.raises(VerificationStoreError, match="did not become ready") as failure:
        acquire_server(data_directory, executable=executable)

    assert "waiting on an unavailable lock" in str(failure.value)


def test_a_failed_startup_leaves_no_server_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    monkeypatch.setattr(server_module, "_READINESS_TIMEOUT_SECONDS", 1.0)
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    executable = _fake_dolt(tmp_path, "sleep 30\n")
    before = _child_process_ids()

    with pytest.raises(VerificationStoreError, match="did not become ready"):
        acquire_server(data_directory, executable=executable)

    assert _child_process_ids() - before == set()


def test_the_runtime_version_is_probed_once_per_executable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    probes: list[str] = []
    original = server_module._run_version_probe

    def counting_probe(resolved: str) -> str:
        probes.append(resolved)
        return original(resolved)

    monkeypatch.setattr(server_module, "_run_version_probe", counting_probe)
    server_module._clear_runtime_cache()
    executable = _fake_dolt(tmp_path, "exit 1\n")

    assert validate_runtime(executable) == executable
    assert validate_runtime(executable) == executable
    assert probes == [executable]

    Path(executable).write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "version" ]; then\n'
        '  echo "dolt version 1.41.0"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    Path(executable).chmod(0o755)

    assert validate_runtime(executable) == executable
    assert probes == [executable, executable]


def test_an_incompatible_runtime_is_refused_after_a_cached_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("the stand-in runtime is a POSIX shell script")
    executable = _fake_dolt(tmp_path, "exit 1\n")
    assert validate_runtime(executable) == executable

    monkeypatch.setattr(server_module, "_MINIMUM_DOLT_VERSION", (999, 0, 0))

    with pytest.raises(VerificationStoreError, match=r"runtime .* is incompatible"):
        validate_runtime(executable)


def test_a_missing_runtime_is_refused_before_any_server_starts(tmp_path: Path) -> None:
    with pytest.raises(VerificationStoreError, match=r"install Dolt 1\.40 or newer"):
        acquire_server(tmp_path, executable="missing-strideweave-dolt")


def test_configuration_scalars_survive_interpolation_and_quoting() -> None:
    encoded = server_module._yaml_scalar("C:\\data\\${HOME} evidence")

    assert encoded == '"C:\\\\data\\\\$${HOME} evidence"'


@pytest.mark.dolt_lifecycle
def test_the_server_writes_its_configuration_outside_the_data_directory(
    tmp_path: Path,
) -> None:
    _require_dolt()
    data_directory = tmp_path / "data"
    _initialized_store(data_directory / "verification-store")

    acquire_server(data_directory)

    assert sorted(entry.name for entry in data_directory.iterdir()) == [
        ".dolt",
        "verification-store",
    ]


def test_server_lifecycle_is_not_part_of_the_public_store_surface() -> None:
    import strideweave.verification as verification
    import strideweave.verification.store as store

    lifecycle_names = {
        "ManagedDoltServer",
        "acquire_server",
        "server",
        "shutdown_servers",
        "validate_runtime",
    }

    assert lifecycle_names.isdisjoint(store.__all__)
    assert lifecycle_names.isdisjoint(getattr(verification, "__all__", ()))
    assert not any(
        name.startswith(("server", "port", "connect", "shutdown"))
        for name in dir(store.DoltEvidenceStore)
        if not name.startswith("_")
    )


def test_a_managed_server_is_not_usable_before_it_starts(tmp_path: Path) -> None:
    server = ManagedDoltServer(tmp_path, "dolt")

    assert not server.is_running()
    with pytest.raises(VerificationStoreError, match="server is not running"):
        _ = server.port
    with pytest.raises(VerificationStoreError, match="server is not running"):
        _ = server.process_id
