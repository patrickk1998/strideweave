"""Local Dolt implementation of the verification-store boundary.

Statements reach Dolt over a persistent MySQL-compatible session on the
internally managed server rather than through one ``dolt`` process per
statement. Each store path names one served database inside the data directory
that holds it, so several stores under one directory share a single server.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .base import EvidenceStore, SQLStatement, SQLValue, VerificationStoreError
from .server import ManagedDoltServer, acquire_server, client_module, validate_runtime

if TYPE_CHECKING:
    from pymysql.connections import Connection

_PROJECT_DIRECTORY = "strideweave/kernel-evidence"
_BULK_STATEMENT_ROWS = 1000


def default_store_path() -> Path:
    """Return the platform-local verification-store path without creating it."""

    override = os.environ.get("STRIDEWEAVE_STATUS_HOME")
    if override:
        return Path(override).expanduser() / _PROJECT_DIRECTORY
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData/Local"
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local/share"
    return base / _PROJECT_DIRECTORY


def _bind_value(value: SQLValue) -> object:
    """Return one statement value in the form the client driver sends as data.

    The value never becomes part of a statement's text. Timestamps are
    normalized to naive UTC, which is how the schema stores them, and the
    unrepresentable values the store refuses are rejected here rather than
    silently encoded.
    """

    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("verification-store SQL values must be finite")
        return value
    if type(value) is datetime:
        if value.tzinfo is None:
            raise ValueError("verification-store timestamps must include a timezone")
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if type(value) is str or type(value) is bytes:
        return value
    raise TypeError(f"unsupported verification-store SQL value {type(value).__name__}")


def _prepare(statement: SQLStatement) -> tuple[str, tuple[object, ...] | None]:
    """Translate one internal statement into driver text and bound parameters."""

    placeholders = statement.template.count("?")
    if placeholders != len(statement.parameters):
        raise ValueError(
            "verification-store SQL placeholder count does not match values"
        )
    if not statement.parameters:
        return statement.template, None
    text = "%s".join(
        piece.replace("%", "%%") for piece in statement.template.split("?")
    )
    return text, tuple(_bind_value(value) for value in statement.parameters)


def _decode_value(value: object) -> object:
    """Return one stored column value in the store's deterministic Python form."""

    if type(value) is datetime:
        return value.isoformat(sep=" ", timespec="microseconds")
    if type(value) is date:
        return value.isoformat()
    if type(value) is bytes:
        return value.decode("utf-8", errors="replace")
    return value


def _quoted_identifier(name: str, subject: str) -> str:
    if not name or "`" in name or "\x00" in name:
        raise VerificationStoreError(
            f"verification store {subject} {name!r} is not a usable database name"
        )
    return f"`{name}`"


class DoltEvidenceStore(EvidenceStore):
    """Automatically managed local verification store backed by Dolt."""

    def __init__(
        self, path: str | os.PathLike[str] | None = None, *, executable: str = "dolt"
    ):
        self._path = Path(path) if path is not None else default_store_path()
        self._executable = executable
        self._initialized = False
        self._connection: Connection[Any] | None = None
        self._session_process_id: int | None = None

    @property
    def path(self) -> Path:
        """Return the configured path without initializing the store."""

        return self._path

    def initialize(self) -> None:
        """Create or validate the local store and apply checked migrations."""

        if self._initialized:
            return
        self._validate_location()
        self._validate_runtime()
        self._create_database_if_needed()
        self._apply_migrations()
        self._initialized = True

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        """Execute an explicit atomic transaction of safely encoded statements."""

        self.initialize()
        if not statements:
            return
        connection = self._session()
        try:
            connection.begin()
            with connection.cursor() as cursor:
                for text, parameter_sets in _batched(statements):
                    if parameter_sets is None:
                        cursor.execute(text)
                    elif len(parameter_sets) == 1:
                        cursor.execute(text, parameter_sets[0])
                    else:
                        cursor.executemany(text, parameter_sets)
            connection.commit()
        except client_module().Error as error:
            self._discard_failed_transaction(connection)
            raise VerificationStoreError(
                f"verification store operation failed{_diagnostic(error)}"
            ) from error

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        """Execute one read-only query and return immutable decoded rows."""

        self.initialize()
        return self._query(statement, "query")

    def _query(
        self, statement: SQLStatement, subject: str
    ) -> tuple[Mapping[str, object], ...]:
        text, parameters = _prepare(statement)
        connection = self._session()
        client = client_module()
        try:
            with connection.cursor(client.cursors.DictCursor) as cursor:
                cursor.execute(text, parameters)
                rows = cursor.fetchall()
            # A read opens a transaction whose snapshot would otherwise hide
            # every later write from this same session. Ending it by rollback
            # rather than commit releases the snapshot without asking Dolt to
            # write a working set for a query that changed nothing.
            connection.rollback()
        except client.Error as error:
            self._discard_failed_transaction(connection)
            raise VerificationStoreError(
                f"verification store {subject} failed{_diagnostic(error)}"
            ) from error
        return tuple(
            MappingProxyType(
                {name: _decode_value(value) for name, value in row.items()}
            )
            for row in rows
        )

    def _validate_location(self) -> None:
        candidate = self._path.expanduser().absolute()
        for parent in (candidate, *candidate.parents):
            if parent.name == ".beads" or (parent / ".beads").is_dir():
                raise VerificationStoreError(
                    "verification store path must be separate from a Beads workspace"
                )

    def _validate_runtime(self) -> None:
        self._executable = validate_runtime(self._executable)

    def _data_directory(self) -> Path:
        return self._path.expanduser().absolute().parent

    def _database(self) -> str:
        return self._path.expanduser().absolute().name

    def _server(self) -> ManagedDoltServer:
        self._data_directory().mkdir(parents=True, exist_ok=True)
        return acquire_server(self._data_directory(), executable=self._executable)

    def _session(self) -> Connection[Any]:
        if self._session_process_id not in (None, os.getpid()):
            self._discard_inherited_session()
        connection = self._connection
        if connection is not None and connection.open:
            return connection
        self._connection = self._server().connect(self._database())
        self._session_process_id = os.getpid()
        return self._connection

    def _discard_inherited_session(self) -> None:
        """Release a session inherited across a fork without speaking to it.

        The session belongs to the process that opened it. Closing it normally
        would write client protocol traffic onto a socket that process is still
        using, so the inherited connection is dropped and its own descriptor
        released directly; the parent's connection is unaffected.
        """

        connection = self._connection
        self._connection = None
        self._session_process_id = None
        if connection is None:
            return
        release = getattr(connection, "_force_close", None)
        if callable(release):
            release()

    def _discard_failed_transaction(self, connection: Connection[Any]) -> None:
        """Leave the store unchanged and the session reusable after a failure."""

        try:
            connection.rollback()
        except client_module().Error:
            connection.close()
        if self._connection is not None and not self._connection.open:
            self._connection = None

    def _create_database_if_needed(self) -> None:
        if (self._path / ".dolt").is_dir():
            self._server()
            return
        if self._path.exists():
            if any(self._path.iterdir()):
                raise VerificationStoreError(
                    "verification store path exists but is not an initialized store"
                )
            # Dolt refuses to create a database over an existing directory, and
            # this one is provably empty, so it is released before creation.
            self._path.rmdir()
        server = self._server()
        identifier = _quoted_identifier(self._database(), "path")
        connection = server.connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE {identifier}")
            connection.commit()
        except client_module().Error as error:
            raise VerificationStoreError(
                f"verification store could not be created{_diagnostic(error)}"
            ) from error
        finally:
            connection.close()

    def _migration_files(self) -> tuple[tuple[int, str, str, str], ...]:
        root = resources.files("strideweave.verification.store.migrations")
        migrations = []
        for entry in root.iterdir():
            if not entry.name.endswith(".sql"):
                continue
            prefix, separator, _ = entry.name.partition("_")
            if not separator or not prefix.isdigit():
                raise VerificationStoreError(
                    f"verification store has invalid migration name {entry.name!r}"
                )
            sql = entry.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            migrations.append((int(prefix), entry.name, checksum, sql))
        migrations.sort(key=lambda item: item[0])
        versions = tuple(item[0] for item in migrations)
        if versions != tuple(range(1, len(migrations) + 1)):
            raise VerificationStoreError(
                "verification store migrations must be a contiguous sequence starting at 1"
            )
        return tuple(migrations)

    def _apply_migrations(self) -> None:
        migrations = self._migration_files()
        table_rows = self._query(
            SQLStatement("SHOW TABLES LIKE 'schema_migrations'"), "migration"
        )
        if table_rows:
            applied_rows = self._query(
                SQLStatement(
                    "SELECT version, migration_name, migration_checksum "
                    "FROM schema_migrations ORDER BY version"
                ),
                "migration",
            )
        else:
            applied_rows = ()
        if len(applied_rows) > len(migrations):
            raise VerificationStoreError(
                "verification store schema is newer than this StrideWeave installation"
            )
        for index, row in enumerate(applied_rows):
            version, name, checksum, _ = migrations[index]
            if (
                row.get("version") != version
                or row.get("migration_name") != name
                or row.get("migration_checksum") != checksum
            ):
                raise VerificationStoreError(
                    f"verification store migration history disagrees at version {version}"
                )
        for version, name, checksum, sql in migrations[len(applied_rows) :]:
            self._apply_one_migration(version, name, checksum, sql)

    def _apply_one_migration(
        self, version: int, name: str, checksum: str, sql: str
    ) -> None:
        """Apply one migration script and its history row in one transaction.

        The script holds several repository-owned statements, so it runs on a
        separate multi-statement session. The ordinary store session stays
        single-statement, where no parameter can extend a statement.
        """

        applied_at = datetime.now(timezone.utc)
        insert_text, insert_parameters = _prepare(
            SQLStatement(
                "INSERT INTO schema_migrations "
                "(version, migration_name, migration_checksum, applied_at_utc) "
                "VALUES (?, ?, ?, ?)",
                (version, name, checksum, applied_at),
            )
        )
        connection = self._server().connect_multi_statement(self._database())
        try:
            connection.begin()
            with connection.cursor() as cursor:
                cursor.execute(sql)
                while cursor.nextset():
                    pass
                cursor.execute(insert_text, insert_parameters)
            connection.commit()
        except client_module().Error as error:
            raise VerificationStoreError(
                f"verification store migration {name} failed{_diagnostic(error)}"
            ) from error
        finally:
            connection.close()


def _diagnostic(error: BaseException) -> str:
    text = str(error).strip()
    return f": {text}" if text else ""


def _batched(
    statements: Sequence[SQLStatement],
) -> list[tuple[str, list[tuple[object, ...]] | None]]:
    """Group consecutive statements sharing one template into bounded batches.

    Statement order is preserved, so foreign-key ordering inside a transaction
    is unchanged; only repetitions of one template are combined, and each batch
    is capped so a large ingestion sends bounded statements rather than one
    unbounded statement or thousands of individually parsed ones.
    """

    batches: list[tuple[str, list[tuple[object, ...]] | None]] = []
    for statement in statements:
        text, parameters = _prepare(statement)
        if parameters is None:
            batches.append((text, None))
            continue
        if batches:
            previous_text, previous_parameters = batches[-1]
            if (
                previous_text == text
                and previous_parameters is not None
                and len(previous_parameters) < _BULK_STATEMENT_ROWS
            ):
                previous_parameters.append(parameters)
                continue
        batches.append((text, [parameters]))
    return batches
