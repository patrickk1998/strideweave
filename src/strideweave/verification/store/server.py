"""Process-local lifecycle for the internally managed Dolt SQL server.

The verification store speaks the MySQL-compatible Dolt server protocol instead
of launching one ``dolt`` command per statement. This module owns that server:
it validates the runtime once per executable identity, starts at most one server
per resolved data directory, hands out parameterized client connections, and
terminates every server it started before the interpreter exits.

Nothing here is public StrideWeave API. Users neither configure nor operate the
server; they continue to name a store path, and the store acquires whatever
transport it needs.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .base import VerificationStoreError

if TYPE_CHECKING:
    from types import ModuleType

    from pymysql.connections import Connection

_MINIMUM_DOLT_VERSION: Final = (1, 40, 0)
_SERVER_HOST: Final = "127.0.0.1"
_SERVER_USER: Final = "root"
_READINESS_TIMEOUT_SECONDS: Final = 60.0
_READINESS_POLL_SECONDS: Final = 0.02
_TEARDOWN_TIMEOUT_SECONDS: Final = 15.0
# One shutdown terminates gracefully and then forcibly, so a caller waiting for
# a directory to be released allows for both, plus the process exit itself.
_RELEASE_TIMEOUT_SECONDS: Final = 2 * _TEARDOWN_TIMEOUT_SECONDS + 5.0
_STARTUP_ATTEMPTS: Final = 3
_ACQUISITION_ATTEMPTS: Final = 3
_CONNECT_TIMEOUT_SECONDS: Final = 10.0
_DIAGNOSTIC_CHARACTERS: Final = 4000

_MISSING_RUNTIME_DIAGNOSTIC: Final = (
    "verification store runtime is unavailable; install Dolt 1.40 or newer "
    "or configure its executable path"
)


class _ServerExited(VerificationStoreError):
    """Report a server that terminated before it accepted any connection."""


class _ServerRetired(VerificationStoreError):
    """Report a server object that may never run again.

    Retirement is what keeps a registered server and a running process the same
    thing: a server is retired when it is shut down or when its startup failed,
    and acquisition answers this by taking a fresh server rather than starting
    one nobody is tracking. It is internal, and :func:`acquire_server` handles
    every instance it can provoke.
    """


_runtime_lock = threading.Lock()
_probed_versions: dict[tuple[Any, ...], tuple[int, int, int]] = {}

# Stopping is a registry state of its own. A directory leaves ``_servers`` when
# its shutdown begins but stays in ``_stopping`` until that server's process has
# exited and released the database locks it holds, so the directory has one
# owner at every moment rather than briefly none. Waiters are woken through the
# condition rather than by polling.
_servers_lock = threading.Condition()
_servers: dict[Path, ManagedDoltServer] = {}
_stopping: set[Path] = set()


def client_module() -> ModuleType:
    """Import the MySQL-compatible client lazily, on first store transport use."""

    import pymysql

    return pymysql


def _parse_version(output: str) -> tuple[int, int, int]:
    for token in output.replace("-", " ").split():
        pieces = token.lstrip("v").split(".")
        if len(pieces) >= 3 and all(piece.isdigit() for piece in pieces[:3]):
            return (int(pieces[0]), int(pieces[1]), int(pieces[2]))
    raise VerificationStoreError(
        "verification store runtime returned an unrecognized version"
    )


def _executable_identity(resolved: str) -> tuple[Any, ...]:
    """Return the identity that decides whether a runtime probe still applies."""

    try:
        status = os.stat(resolved)
    except OSError as error:
        raise VerificationStoreError(_MISSING_RUNTIME_DIAGNOSTIC) from error
    return (os.path.realpath(resolved), status.st_size, status.st_mtime_ns)


def _run_version_probe(resolved: str) -> str:
    """Ask one Dolt executable for its version, returning raw command output."""

    try:
        completed = subprocess.run(
            (resolved, "version"),
            cwd=Path.cwd(),
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise VerificationStoreError(_MISSING_RUNTIME_DIAGNOSTIC) from error
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout).strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        raise VerificationStoreError(
            f"verification store runtime version check failed{suffix}"
        ) from error
    return completed.stdout


def validate_runtime(executable: str) -> str:
    """Resolve one Dolt executable and require a compatible version.

    The version probe runs once per executable identity for the life of the
    process, while the compatibility comparison itself is repeated, so replacing
    the binary is noticed and a supported installation is probed only once.

    Args:
        executable: Executable name or path to resolve on the current PATH.

    Returns:
        The absolute path of the validated executable.

    Raises:
        VerificationStoreError: If the runtime is missing or too old.
    """

    resolved = shutil.which(executable)
    if resolved is None:
        raise VerificationStoreError(_MISSING_RUNTIME_DIAGNOSTIC)
    identity = _executable_identity(resolved)
    with _runtime_lock:
        version = _probed_versions.get(identity)
        if version is None:
            version = _parse_version(_run_version_probe(resolved))
            _probed_versions[identity] = version
    if version < _MINIMUM_DOLT_VERSION:
        required = ".".join(str(part) for part in _MINIMUM_DOLT_VERSION)
        observed = ".".join(str(part) for part in version)
        raise VerificationStoreError(
            f"verification store runtime {observed} is incompatible; "
            f"version {required} or newer is required"
        )
    return resolved


def _clear_runtime_cache() -> None:
    """Forget every probed runtime version, so the next validation reprobes."""

    with _runtime_lock:
        _probed_versions.clear()


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind((_SERVER_HOST, 0))
        return int(reservation.getsockname()[1])


def _yaml_scalar(value: str) -> str:
    """Encode one string as a Dolt server configuration scalar.

    Dolt expands ``${NAME}`` and ``$$`` in the raw configuration text before it
    parses YAML, so a literal ``$`` is doubled. The remaining quoting is JSON,
    which YAML accepts, and which escapes the quotes and backslashes that
    Windows paths carry.
    """

    return json.dumps(value).replace("$", "$$")


class ManagedDoltServer:
    """One Dolt SQL server this process started for one data directory.

    Instances are created by :func:`acquire_server` rather than directly. Each
    server owns a private configuration directory, listens on a loopback port
    reserved for this process, and serves every Dolt database under its data
    directory.
    """

    def __init__(self, data_directory: Path, executable: str):
        self._data_directory = data_directory
        self._executable = executable
        self._owner_process_id = os.getpid()
        self._lock = threading.Lock()
        self._retired = False
        self._private_directory: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_path: Path | None = None
        self._port: int | None = None

    @property
    def data_directory(self) -> Path:
        """Return the resolved directory whose databases this server serves."""

        return self._data_directory

    @property
    def port(self) -> int:
        """Return the loopback port a running server listens on."""

        if self._port is None:
            raise VerificationStoreError("verification store server is not running")
        return self._port

    @property
    def process_id(self) -> int:
        """Return the operating-system identifier of the running server."""

        if self._process is None:
            raise VerificationStoreError("verification store server is not running")
        return self._process.pid

    def is_owned(self) -> bool:
        """Report whether this process, rather than a forked ancestor, owns this.

        A forked child inherits every live server object, but none of those
        servers are its children: it may neither wait on them nor terminate
        them, and their private state belongs to the process that started them.
        """

        return self._owner_process_id == os.getpid()

    def is_running(self) -> bool:
        """Report whether this server was started and has not exited or stopped.

        A server inherited across a fork is never running for this process, so
        an inherited object is not mistaken for usable transport.
        """

        process = self._process
        return self.is_owned() and process is not None and process.poll() is None

    def is_retired(self) -> bool:
        """Report whether this server was shut down or failed to start.

        A retired server never runs again, so the answer only ever changes from
        false to true and is readable without holding this server's lock.
        """

        return self._retired

    def connect(self, database: str | None = None) -> Connection[Any]:
        """Open one parameterized client connection to this server.

        Args:
            database: Served database to select, or ``None`` for a connection
                that selects none.

        Returns:
            An open PyMySQL connection with autocommit disabled, so callers own
            their transaction boundaries.

        Raises:
            VerificationStoreError: If the server is not running or refuses the
                connection.
        """

        return self._connect(database, multi_statement=False)

    def connect_multi_statement(self, database: str | None = None) -> Connection[Any]:
        """Open one connection that accepts a repository-owned statement script.

        Only scripts the repository ships — the migration files — are sent this
        way. Ordinary sessions stay single-statement, where no parameter value
        can extend the statement it belongs to.

        Args:
            database: Served database to select, or ``None`` for a connection
                that selects none.

        Returns:
            An open PyMySQL connection with autocommit disabled and client-side
            multi-statement support enabled.

        Raises:
            VerificationStoreError: If the server is not running or refuses the
                connection.
        """

        return self._connect(database, multi_statement=True)

    def _connect(
        self, database: str | None, *, multi_statement: bool
    ) -> Connection[Any]:
        if not self.is_running():
            raise VerificationStoreError("verification store server is not running")
        try:
            return self._open_connection(database, multi_statement=multi_statement)
        except client_module().Error as error:
            raise VerificationStoreError(
                f"verification store server refused a connection{self._diagnostic()}"
            ) from error

    def _open_connection(
        self, database: str | None, *, multi_statement: bool = False
    ) -> Connection[Any]:
        client = client_module()
        return client.connect(
            host=_SERVER_HOST,
            port=self.port,
            user=_SERVER_USER,
            password="",
            database=database,
            autocommit=False,
            charset="utf8mb4",
            local_infile=False,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            client_flag=(
                client.constants.CLIENT.MULTI_STATEMENTS if multi_statement else 0
            ),
        )

    def ensure_started(self) -> None:
        """Start this server if it is not already accepting connections.

        Startup, retirement, and shutdown share one lock, so a server that was
        retired while a caller waited its turn is never started afterwards, and
        a startup failure retires this object before the next waiter sees it.

        Raises:
            _ServerRetired: If this server was shut down, already failed to
                start, or belongs to a forked ancestor, in which case the
                caller acquires a fresh one.
            VerificationStoreError: If the server does not become ready.
        """

        with self._lock:
            if self._retired:
                raise _ServerRetired(
                    "verification store server was stopped before it started"
                )
            if not self.is_owned():
                raise _ServerRetired(
                    "verification store server belongs to another process"
                )
            if self._process is not None and self._process.poll() is None:
                return
            try:
                self._start()
            except BaseException:
                self._retired = True
                raise

    def _start(self) -> None:
        # A port reserved from the ephemeral range is free when it is chosen but
        # is only bound once the server runs, so a server that exits immediately
        # is retried on a fresh port. A server that starts and stays unready is
        # a real failure and is reported after one attempt.
        for attempt in range(_STARTUP_ATTEMPTS):
            try:
                self._launch()
                return
            except _ServerExited:
                if attempt == _STARTUP_ATTEMPTS - 1:
                    raise

    def _launch(self) -> None:
        private = Path(tempfile.mkdtemp(prefix="strideweave-evidence-server-"))
        self._private_directory = private
        self._log_path = private / "server.log"
        self._port = _reserve_local_port()
        configuration = private / "server.yaml"
        configuration.write_text(self._configuration_text(private), encoding="utf-8")
        environment = dict(os.environ)
        environment["DOLT_ROOT_PATH"] = str(private / "root")
        # Dolt otherwise spawns a detached child at exit to flush usage events.
        # That child is reparented away from this process, so it both escapes
        # the lifecycle this module guarantees and recreates the private
        # directory moments after teardown removed it.
        environment["DOLT_DISABLE_EVENT_FLUSH"] = "true"
        # The log is a file rather than a captured pipe: nothing in this process
        # drains the server's output, so a pipe would deadlock a chatty server.
        with self._log_path.open("wb") as log:
            process = subprocess.Popen(
                (self._executable, "sql-server", "--config", str(configuration)),
                cwd=private,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        self._process = process
        try:
            self._await_readiness(process)
        except BaseException:
            self._terminate()
            raise

    def _configuration_text(self, private: Path) -> str:
        return (
            "log_level: warning\n"
            "listener:\n"
            f"  host: {_yaml_scalar(_SERVER_HOST)}\n"
            f"  port: {self.port}\n"
            f"data_dir: {_yaml_scalar(str(self._data_directory))}\n"
            f"cfg_dir: {_yaml_scalar(str(private / 'config'))}\n"
            f"privilege_file: {_yaml_scalar(str(private / 'privileges.db'))}\n"
            "branch_control_file: "
            f"{_yaml_scalar(str(private / 'branch_control.db'))}\n"
        )

    def _await_readiness(self, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
        while True:
            if process.poll() is not None:
                raise _ServerExited(
                    "verification store server exited during startup"
                    f"{self._diagnostic()}"
                )
            try:
                self._open_connection(None).close()
                return
            except client_module().Error:
                if time.monotonic() >= deadline:
                    raise VerificationStoreError(
                        "verification store server did not become ready within "
                        f"{_READINESS_TIMEOUT_SECONDS:.0f} seconds"
                        f"{self._diagnostic()}"
                    ) from None
                time.sleep(_READINESS_POLL_SECONDS)

    def _diagnostic(self) -> str:
        """Return the tail of the server log as an appendable diagnostic."""

        if self._log_path is None:
            return ""
        try:
            output = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        output = output.strip()[-_DIAGNOSTIC_CHARACTERS:]
        return f": {output}" if output else ""

    def shutdown(self) -> None:
        """Stop this server permanently, leaving no server child behind.

        Shutting down retires the object, so a caller that is about to start it
        — or is waiting to — never revives a server the registry has already
        released.
        """

        with self._lock:
            self._retired = True
            self._terminate()

    def _terminate(self) -> None:
        owned = self.is_owned()
        process = self._process
        self._process = None
        self._port = None
        if not owned:
            # An inherited server is another process's child and its private
            # directory is that process's live state, so this call only forgets
            # what it inherited: it must neither signal nor delete any of it.
            self._log_path = None
            self._private_directory = None
            return
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_TEARDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_TEARDOWN_TIMEOUT_SECONDS)
        private = self._private_directory
        self._private_directory = None
        self._log_path = None
        if private is not None:
            shutil.rmtree(private, ignore_errors=True)


def acquire_server(
    data_directory: str | os.PathLike[str], *, executable: str = "dolt"
) -> ManagedDoltServer:
    """Return the running server for one data directory, starting it if needed.

    Every caller naming the same resolved directory shares one server for the
    life of the process; distinct directories are served independently.

    Args:
        data_directory: Existing directory whose Dolt databases are served.
        executable: Executable name or path of the Dolt runtime.

    A concurrent shutdown retires the server it releases, so this never returns
    a running server the registry has stopped tracking; acquisition simply
    starts a fresh one. A directory whose shutdown is still in progress is not
    yet replaceable — the old process still holds its database locks — so this
    waits for that shutdown to finish rather than starting a second server that
    would be refused. A startup failure retires and unregisters its server too,
    leaving the directory acquirable again.

    Returns:
        The started :class:`ManagedDoltServer` for that directory.

    Raises:
        VerificationStoreError: If the runtime is unusable, the server does not
            become ready, a shutdown does not release the directory in time, or
            repeated concurrent shutdowns keep stopping the server being
            started.
    """

    resolved_executable = validate_runtime(executable)
    resolved_directory = Path(data_directory).expanduser().resolve()
    for _ in range(_ACQUISITION_ATTEMPTS):
        with _servers_lock:
            _await_release(resolved_directory)
            server = _servers.get(resolved_directory)
            if server is None or server.is_retired():
                server = ManagedDoltServer(resolved_directory, resolved_executable)
                _servers[resolved_directory] = server
        try:
            server.ensure_started()
        except _ServerRetired:
            _release_server(resolved_directory, server)
            continue
        except BaseException:
            _release_server(resolved_directory, server)
            raise
        with _servers_lock:
            if _servers.get(resolved_directory) is server:
                return server
        # A shutdown released this server while this call was starting it. Stop
        # what this call started rather than return a server no registry entry,
        # and therefore no interpreter-exit cleanup, still covers.
        server.shutdown()
    raise VerificationStoreError(
        "verification store server was stopped repeatedly while it was starting"
    )


def _await_release(resolved_directory: Path) -> None:
    """Wait until no shutdown still owns one data directory.

    The caller holds ``_servers_lock``. A shutdown that has begun keeps its
    directory reserved until its process has exited, so waiting here is what
    stops a replacement from racing the old process for the database locks.

    Raises:
        VerificationStoreError: If the shutdown does not finish in time, which
            leaves the directory reserved rather than contended.
    """

    if resolved_directory not in _stopping:
        return
    deadline = time.monotonic() + _RELEASE_TIMEOUT_SECONDS
    while resolved_directory in _stopping:
        if not _servers_lock.wait(timeout=max(deadline - time.monotonic(), 0.0)):
            raise VerificationStoreError(
                "verification store server did not release "
                f"{resolved_directory} within "
                f"{_RELEASE_TIMEOUT_SECONDS:.0f} seconds"
            )


def _release_server(resolved_directory: Path, server: ManagedDoltServer) -> None:
    """Drop one unusable server from the registry, keeping any replacement.

    A server reaching this never started or has already terminated, so its
    directory is free the moment the entry goes.
    """

    with _servers_lock:
        if _servers.get(resolved_directory) is server:
            del _servers[resolved_directory]
        _servers_lock.notify_all()


def started_server_directories() -> tuple[Path, ...]:
    """Return the resolved data directories this process currently serves."""

    with _servers_lock:
        return tuple(_servers)


def shutdown_servers(
    data_directories: Iterable[str | os.PathLike[str]] | None = None,
) -> None:
    """Stop the servers this process started.

    Each selected directory stays reserved from the moment its entry is removed
    until its server has terminated, so an acquisition running alongside this
    waits for the old process rather than starting a replacement the old one
    would refuse.

    Args:
        data_directories: Data directories whose servers should stop, or
            ``None`` to stop every server this process started.
    """

    with _servers_lock:
        if data_directories is None:
            selected = tuple(_servers.items())
            _servers.clear()
        else:
            wanted = {
                Path(directory).expanduser().resolve() for directory in data_directories
            }
            selected = tuple((key, _servers[key]) for key in wanted if key in _servers)
            for key in wanted:
                _servers.pop(key, None)
        _stopping.update(key for key, _ in selected)
    for key, server in selected:
        try:
            server.shutdown()
        finally:
            with _servers_lock:
                _stopping.discard(key)
                _servers_lock.notify_all()


def _forget_inherited_lifecycle() -> None:
    """Reset this module in a freshly forked child.

    A child inherits the registry, its locks — possibly held by threads that do
    not exist here — and the interpreter-exit cleanup. None of the inherited
    servers are this process's children, so the child starts with an empty
    registry and free locks: its own exit stops only what it started itself,
    and a store used here acquires its own server rather than reusing a socket
    and a process that belong to its parent. The probed runtime versions
    describe executables on disk, so they stay valid and are retained.
    """

    global _runtime_lock, _servers, _servers_lock, _stopping

    _runtime_lock = threading.Lock()
    _servers_lock = threading.Condition()
    _servers = {}
    # A shutdown a parent thread was running is not this process's to finish,
    # and none of the inherited directories are reserved here.
    _stopping = set()


atexit.register(shutdown_servers)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_lifecycle)
