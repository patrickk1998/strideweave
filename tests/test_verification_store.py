from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import strideweave.verification.store.dolt as dolt_store_module
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
    default_store_path,
)


def _require_dolt() -> None:
    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


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


def test_first_use_creates_and_repeated_use_validates_the_store(tmp_path: Path) -> None:
    _require_dolt()
    path = tmp_path / "verification-store"
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


def test_transaction_rollback_leaves_no_partial_facts(tmp_path: Path) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "verification-store")
    target = "a" * 64

    with pytest.raises(VerificationStoreError, match="operation failed"):
        store.execute_transaction(
            (
                SQLStatement(
                    "INSERT INTO verification_targets "
                    "(target_id, architecture, vendor, operating_system, abi, "
                    "endianness, pointer_bits, descriptor_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (target, "x86_64", "test", "test", "test", "little", 64, "{}"),
                ),
                SQLStatement("INSERT INTO table_that_does_not_exist VALUES (1)"),
            )
        )

    rows = store.query(
        SQLStatement(
            "SELECT target_id FROM verification_targets WHERE target_id = ?", (target,)
        )
    )
    assert rows == ()


def test_sql_values_are_data_not_executable_sql(tmp_path: Path) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "verification-store")
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


def test_changed_migration_checksum_is_refused(tmp_path: Path) -> None:
    _require_dolt()
    path = tmp_path / "verification-store"
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


def test_unknown_migration_version_is_refused_as_schema_skew(tmp_path: Path) -> None:
    _require_dolt()
    path = tmp_path / "verification-store"
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
    _require_dolt()
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_dolt()
    monkeypatch.setattr(dolt_store_module, "_MINIMUM_DOLT_VERSION", (999, 0, 0))
    store = DoltEvidenceStore(tmp_path / "verification-store")

    with pytest.raises(VerificationStoreError, match=r"runtime .* is incompatible"):
        store.initialize()
    assert not store.path.exists()
