"""Local Dolt implementation of the verification-store boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .base import EvidenceStore, SQLStatement, SQLValue, VerificationStoreError

_PROJECT_DIRECTORY = "strideweave/kernel-evidence"
_MINIMUM_DOLT_VERSION = (1, 40, 0)


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


def _sql_literal(value: SQLValue) -> str:
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "TRUE" if value else "FALSE"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("verification-store SQL values must be finite")
        return repr(value)
    if type(value) is datetime:
        if value.tzinfo is None:
            raise ValueError("verification-store timestamps must include a timezone")
        normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
        value = normalized.isoformat(timespec="microseconds")
    if type(value) is str:
        encoded = value.encode("utf-8").hex()
        return f"CONVERT(0x{encoded} USING utf8mb4)"
    if type(value) is bytes:
        return f"0x{value.hex()}"
    raise TypeError(f"unsupported verification-store SQL value {type(value).__name__}")


def _render_statement(statement: SQLStatement) -> str:
    pieces = statement.template.split("?")
    if len(pieces) - 1 != len(statement.parameters):
        raise ValueError(
            "verification-store SQL placeholder count does not match values"
        )
    rendered = [pieces[0]]
    for value, suffix in zip(statement.parameters, pieces[1:], strict=True):
        rendered.extend((_sql_literal(value), suffix))
    return "".join(rendered)


def _parse_version(output: str) -> tuple[int, int, int]:
    for token in output.replace("-", " ").split():
        pieces = token.lstrip("v").split(".")
        if len(pieces) >= 3 and all(piece.isdigit() for piece in pieces[:3]):
            return tuple(int(piece) for piece in pieces[:3])  # type: ignore[return-value]
    raise VerificationStoreError(
        "verification store runtime returned an unrecognized version"
    )


class DoltEvidenceStore(EvidenceStore):
    """Automatically managed local verification store backed by Dolt."""

    def __init__(
        self, path: str | os.PathLike[str] | None = None, *, executable: str = "dolt"
    ):
        self._path = Path(path) if path is not None else default_store_path()
        self._executable = executable
        self._initialized = False

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
        self._create_repository_if_needed()
        self._apply_migrations()
        self._initialized = True

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        """Execute an explicit atomic transaction of safely encoded statements."""

        self.initialize()
        if not statements:
            return
        body = "\n".join(f"{_render_statement(statement)};" for statement in statements)
        self._run_sql(f"START TRANSACTION;\n{body}\nCOMMIT;\n")

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        """Execute one read-only query and return immutable decoded rows."""

        self.initialize()
        completed = self._run(
            ("sql", "--result-format", "json", "--query", _render_statement(statement))
        )
        try:
            payload = json.loads(completed.stdout)
            rows = payload.get("rows", [])
            if type(rows) is not list or any(type(row) is not dict for row in rows):
                raise ValueError
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise VerificationStoreError(
                "verification store returned malformed query data"
            ) from error
        return tuple(MappingProxyType(row) for row in rows)

    def _validate_location(self) -> None:
        candidate = self._path.expanduser().absolute()
        for parent in (candidate, *candidate.parents):
            if parent.name == ".beads" or (parent / ".beads").is_dir():
                raise VerificationStoreError(
                    "verification store path must be separate from a Beads workspace"
                )

    def _validate_runtime(self) -> None:
        resolved = shutil.which(self._executable)
        if resolved is None:
            raise VerificationStoreError(
                "verification store runtime is unavailable; install Dolt 1.40 or newer "
                "or configure its executable path"
            )
        self._executable = resolved
        completed = self._run(("version",), cwd=Path.cwd())
        version = _parse_version(completed.stdout)
        if version < _MINIMUM_DOLT_VERSION:
            required = ".".join(str(part) for part in _MINIMUM_DOLT_VERSION)
            observed = ".".join(str(part) for part in version)
            raise VerificationStoreError(
                f"verification store runtime {observed} is incompatible; "
                f"version {required} or newer is required"
            )

    def _create_repository_if_needed(self) -> None:
        if (self._path / ".dolt").is_dir():
            return
        if self._path.exists() and any(self._path.iterdir()):
            raise VerificationStoreError(
                "verification store path exists but is not an initialized store"
            )
        self._path.mkdir(parents=True, exist_ok=True)
        self._run(
            (
                "init",
                "--name",
                "StrideWeave Verification Store",
                "--email",
                "verification-store@strideweave.invalid",
            )
        )

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
        table_rows = self._query_without_initialization(
            "SHOW TABLES LIKE 'schema_migrations'"
        )
        if table_rows:
            applied_rows = self._query_without_initialization(
                "SELECT version, migration_name, migration_checksum "
                "FROM schema_migrations ORDER BY version"
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
            applied_at = datetime.now(timezone.utc)
            insert = _render_statement(
                SQLStatement(
                    "INSERT INTO schema_migrations "
                    "(version, migration_name, migration_checksum, applied_at_utc) "
                    "VALUES (?, ?, ?, ?)",
                    (version, name, checksum, applied_at),
                )
            )
            self._run_sql(f"START TRANSACTION;\n{sql}\n{insert};\nCOMMIT;\n")

    def _query_without_initialization(
        self, query: str
    ) -> tuple[Mapping[str, Any], ...]:
        completed = self._run(("sql", "--result-format", "json", "--query", query))
        try:
            payload = json.loads(completed.stdout)
            rows = payload.get("rows", [])
            if type(rows) is not list or any(type(row) is not dict for row in rows):
                raise ValueError
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise VerificationStoreError(
                "verification store returned malformed migration data"
            ) from error
        return tuple(rows)

    def _run_sql(self, script: str) -> None:
        self._run(("sql",), input_text=script)

    def _run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                (self._executable, *arguments),
                cwd=cwd or self._path,
                input=input_text,
                capture_output=True,
                check=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise VerificationStoreError(
                "verification store runtime is unavailable; install Dolt 1.40 or newer"
            ) from error
        except subprocess.CalledProcessError as error:
            diagnostic = (error.stderr or error.stdout).strip()
            suffix = f": {diagnostic}" if diagnostic else ""
            raise VerificationStoreError(
                f"verification store operation failed{suffix}"
            ) from error
