from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import strideweave.verification.store.dolt as dolt_store_module
import strideweave.verification.store.server as server_module
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
    default_store_path,
)

StorePathFactory = Callable[..., Path]


def _target_statement(target_id: str) -> SQLStatement:
    return SQLStatement(
        "INSERT INTO verification_targets "
        "(target_id, architecture, vendor, operating_system, abi, "
        "endianness, pointer_bits, descriptor_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (target_id, "x86_64", "test", "test", "test", "little", 64, "{}"),
    )


def test_default_store_path_uses_the_explicit_status_home_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_home = tmp_path / "status-home"
    monkeypatch.setenv("STRIDEWEAVE_STATUS_HOME", str(status_home))

    path = default_store_path()
    store = DoltEvidenceStore()

    assert path == status_home / "strideweave/kernel-evidence"
    assert store.path == path
    assert not status_home.exists()


def test_first_use_creates_and_repeated_use_validates_the_store(
    evidence_store_path: StorePathFactory,
) -> None:
    path = evidence_store_path()
    store = DoltEvidenceStore(path)

    assert not path.exists()
    store.initialize()
    store.initialize()
    DoltEvidenceStore(path).initialize()

    tables = {
        next(iter(row.values())) for row in store.query(SQLStatement("SHOW TABLES"))
    }
    assert {
        "verification_targets",
        "target_proxies",
        "build_toolchains",
        "source_closures",
        "source_closure_inputs",
        "kernel_builds",
        "verification_specs",
        "verification_requirements",
        "tolerance_policies",
        "oracle_references",
        "verification_runs",
        "evidence",
        "observations",
    } <= tables


def test_transaction_rollback_leaves_no_partial_facts(
    evidence_store_path: StorePathFactory,
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    target = "a" * 64

    with pytest.raises(VerificationStoreError, match="operation failed"):
        store.execute_transaction(
            (
                _target_statement(target),
                SQLStatement("INSERT INTO table_that_does_not_exist VALUES (1)"),
            )
        )

    rows = store.query(
        SQLStatement(
            "SELECT target_id FROM verification_targets WHERE target_id = ?", (target,)
        )
    )
    assert rows == ()


def test_sql_values_are_data_not_executable_sql(
    evidence_store_path: StorePathFactory,
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    hostile = "value'); DROP TABLE verification_targets; --\\"
    target = "b" * 64
    store.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO verification_targets "
                "(target_id, architecture, vendor, operating_system, abi, "
                "endianness, pointer_bits, descriptor_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (target, hostile, "test", "test", "test", "little", 64, "{}"),
            ),
        )
    )

    rows = store.query(
        SQLStatement(
            "SELECT architecture FROM verification_targets WHERE target_id = ?",
            (target,),
        )
    )
    assert rows[0]["architecture"] == hostile


def test_changed_migration_checksum_is_refused(
    evidence_store_path: StorePathFactory,
) -> None:
    path = evidence_store_path()
    store = DoltEvidenceStore(path)
    store.initialize()
    store.execute_transaction(
        (
            SQLStatement(
                "UPDATE schema_migrations SET migration_checksum = ? WHERE version = 1",
                ("0" * 64,),
            ),
        )
    )

    with pytest.raises(VerificationStoreError, match="history disagrees at version 1"):
        DoltEvidenceStore(path).initialize()


def test_unknown_migration_version_is_refused_as_schema_skew(
    evidence_store_path: StorePathFactory,
) -> None:
    path = evidence_store_path()
    store = DoltEvidenceStore(path)
    store.initialize()
    store.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO schema_migrations "
                "(version, migration_name, migration_checksum, applied_at_utc) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP(6))",
                (2, "0002_unknown.sql", "0" * 64),
            ),
        )
    )

    with pytest.raises(VerificationStoreError, match="schema is newer"):
        DoltEvidenceStore(path).initialize()


def test_a_beads_workspace_path_is_refused_before_creation(tmp_path: Path) -> None:
    (tmp_path / ".beads").mkdir()
    path = tmp_path / "verification-store"

    with pytest.raises(VerificationStoreError, match="separate from a Beads workspace"):
        DoltEvidenceStore(path).initialize()
    assert not path.exists()


def test_a_missing_runtime_has_an_actionable_store_diagnostic(tmp_path: Path) -> None:
    store = DoltEvidenceStore(
        tmp_path / "verification-store", executable="missing-strideweave-dolt"
    )

    with pytest.raises(VerificationStoreError, match=r"install Dolt 1\.40 or newer"):
        store.initialize()
    assert not store.path.exists()


def test_an_incompatible_runtime_has_an_actionable_store_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dolt_server: object
) -> None:
    monkeypatch.setattr(server_module, "_MINIMUM_DOLT_VERSION", (999, 0, 0))
    store = DoltEvidenceStore(tmp_path / "verification-store")

    with pytest.raises(VerificationStoreError, match=r"runtime .* is incompatible"):
        store.initialize()
    assert not store.path.exists()


def test_a_store_created_by_an_earlier_version_stays_readable_and_migratable(
    legacy_evidence_store_path: Path, dolt_server: object
) -> None:
    store = DoltEvidenceStore(legacy_evidence_store_path)

    store.initialize()
    store.execute_transaction((_target_statement("c" * 64),))

    versions = store.query(
        SQLStatement("SELECT version, migration_name FROM schema_migrations")
    )
    assert [row["version"] for row in versions] == [1]
    assert versions[0]["migration_name"] == "0001_raw_evidence.sql"
    assert DoltEvidenceStore(legacy_evidence_store_path).query(
        SQLStatement(
            "SELECT target_id FROM verification_targets WHERE target_id = ?",
            ("c" * 64,),
        )
    ) == ({"target_id": "c" * 64},)


def test_stores_sharing_one_data_directory_stay_isolated(
    evidence_store_path: StorePathFactory,
) -> None:
    first = DoltEvidenceStore(evidence_store_path("first"))
    second = DoltEvidenceStore(evidence_store_path("second"))

    first.execute_transaction((_target_statement("d" * 64),))
    second.initialize()

    assert (
        first.query(SQLStatement("SELECT COUNT(*) AS n FROM verification_targets"))[0][
            "n"
        ]
        == 1
    )
    assert (
        second.query(SQLStatement("SELECT COUNT(*) AS n FROM verification_targets"))[0][
            "n"
        ]
        == 0
    )


def test_store_operations_launch_no_dolt_sql_subprocesses(
    evidence_store_path: StorePathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[tuple[str, ...]] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    def recording_run(arguments: Sequence[str], **keywords: Any) -> Any:
        launches.append(tuple(arguments))
        return real_run(arguments, **keywords)

    def recording_popen(arguments: Sequence[str], **keywords: Any) -> Any:
        launches.append(tuple(arguments))
        return real_popen(arguments, **keywords)

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    store = DoltEvidenceStore(evidence_store_path())

    store.initialize()
    store.execute_transaction((_target_statement("e" * 64),))
    store.query(SQLStatement("SELECT target_id FROM verification_targets"))
    store.query(SQLStatement("SHOW TABLES"))

    subcommands = [arguments[1] for arguments in launches if len(arguments) > 1]
    assert "sql" not in subcommands
    assert "init" not in subcommands
    # The session server is already running and a cached runtime probe may
    # already have run for this executable identity.
    assert set(subcommands) <= {"version"}


def test_every_real_dolt_test_shares_one_session_server(
    dolt_data_directory: Path,
    dolt_server: object,
    live_dolt_server_count: Callable[[], int],
) -> None:
    assert server_module.started_server_directories() == (dolt_data_directory,)
    assert live_dolt_server_count() == 1


def test_repeated_statements_become_bounded_bulk_batches() -> None:
    template = "INSERT INTO verification_targets VALUES (?)"
    other = "INSERT INTO observations VALUES (?)"
    statements = (
        [SQLStatement(template, (index,)) for index in range(2500)]
        + [SQLStatement(other, ("x",))]
        + [SQLStatement(template, (9,))]
        + [SQLStatement("DELETE FROM verification_targets")]
    )

    batches = dolt_store_module._batched(statements)

    assert [(text, None if rows is None else len(rows)) for text, rows in batches] == [
        ("INSERT INTO verification_targets VALUES (%s)", 1000),
        ("INSERT INTO verification_targets VALUES (%s)", 1000),
        ("INSERT INTO verification_targets VALUES (%s)", 500),
        ("INSERT INTO observations VALUES (%s)", 1),
        ("INSERT INTO verification_targets VALUES (%s)", 1),
        ("DELETE FROM verification_targets", None),
    ]


def test_a_statement_placeholder_is_never_confused_with_a_format_specifier() -> None:
    text, parameters = dolt_store_module._prepare(
        SQLStatement("SELECT ? WHERE name LIKE '%s%'", ("value",))
    )

    assert text == "SELECT %s WHERE name LIKE '%%s%%'"
    assert parameters == ("value",)
