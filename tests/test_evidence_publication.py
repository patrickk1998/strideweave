from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import strideweave as sw
import strideweave.verification.status_cli as status_cli_module
import strideweave.verification.store.publication as publication_module
from strideweave.verification import VerificationOutcome, VerificationReport
from strideweave.verification.provenance import load_compilation_manifest
from strideweave.verification.reporting import bind_report
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    EvidenceStore,
    SQLStatement,
    VerificationStoreError,
    publish_evidence,
    query_status,
    record_report,
    refresh_evidence,
)
from strideweave.verification.store.dolt import _render_statement


class _TrackingStore(EvidenceStore):
    def __init__(self, inner: EvidenceStore) -> None:
        self.inner = inner
        self.queries: list[SQLStatement] = []

    def initialize(self) -> None:
        self.inner.initialize()

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        self.inner.execute_transaction(statements)

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        self.queries.append(statement)
        return self.inner.query(statement)


class _LookupStore(EvidenceStore):
    def __init__(self, rows: Mapping[str, Mapping[str, object]]) -> None:
        self.rows = rows
        self.queries: list[SQLStatement] = []

    def initialize(self) -> None:
        raise AssertionError("batch lookup test should not initialize a store")

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        raise AssertionError("batch lookup test should not execute a transaction")

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        self.queries.append(statement)
        return tuple(
            self.rows[value]
            for value in statement.parameters
            if type(value) is str and value in self.rows
        )


class _PublicationGraphStore(EvidenceStore):
    def __init__(self, rows: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
        self.rows = rows
        self.queries: list[SQLStatement] = []

    def initialize(self) -> None:
        raise AssertionError("publication selection test should not initialize a store")

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        raise AssertionError("publication selection test should not write to a store")

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        self.queries.append(statement)
        _, _, table_and_where = statement.template.partition(" FROM ")
        table, _, where_and_order = table_and_where.partition(" WHERE ")
        where, _, _ = where_and_order.partition(" ORDER BY ")
        column = where.partition(" IN (")[0] if " IN (" in where else "producer_id"
        values = set(statement.parameters)
        rows = [row for row in self.rows[table] if row[column] in values]
        return tuple(
            sorted(
                rows,
                key=lambda row: publication_module._query_row_sort_key(table, row),
            )
        )


def _require_dolt() -> None:
    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


@pytest.fixture(scope="module")
def backend_report() -> VerificationReport:
    return sw.test_backend()


def _changed_report(report: VerificationReport) -> VerificationReport:
    assert report.header is not None
    changed = (
        replace(
            report.records[0],
            outcome=VerificationOutcome.FAILED,
            diagnostic="independent contributor mismatch",
        ),
        *report.records[1:],
    )
    records, header = bind_report(
        changed, (), certificate_facts_override=report.header.certificates
    )
    return VerificationReport(records, header.schema_version, header)


def _rewrite_snapshot(path: Path, mutate: Callable[[dict[str, object]], None]) -> Path:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    identity = {key: value[key] for key in ("producer_id", "schema_version", "tables")}
    value["snapshot_digest"] = hashlib.sha256(
        json.dumps(
            identity, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_independent_stores_publish_and_refresh_namespaced_observations(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    remote = tmp_path / "exchange"
    alice = DoltEvidenceStore(tmp_path / "alice-store")
    bob = DoltEvidenceStore(tmp_path / "bob-store")
    record_report(backend_report, alice, producer_id="alice")
    record_report(_changed_report(backend_report), bob, producer_id="bob")

    alice_publish = publish_evidence(
        alice, producer_id="alice", publish_destination=remote.as_uri()
    )
    repeated = publish_evidence(
        alice, producer_id="alice", publish_destination=remote.as_uri()
    )
    bob_publish = publish_evidence(
        bob, producer_id="bob", publish_destination=remote.as_uri()
    )

    assert repeated == alice_publish
    assert alice_publish.snapshot_digest != bob_publish.snapshot_digest
    assert len(tuple(remote.glob("contributors/*/current.json"))) == 2

    alice_refresh = refresh_evidence(alice, read_source=remote)
    bob_refresh = refresh_evidence(bob, read_source=remote)
    repeated_refresh = refresh_evidence(alice, read_source=remote)

    assert alice_refresh.snapshot_count == 2
    assert bob_refresh.snapshot_count == 2
    assert repeated_refresh == alice_refresh
    architecture = load_compilation_manifest().target.architecture
    alice_status = query_status(alice, architecture=architecture)
    bob_status = query_status(bob, architecture=architecture)
    expected = 2 * len(backend_report.records)
    assert len(alice_status) == expected
    assert len(bob_status) == expected
    assert {item.producer_id for item in alice_status} == {"alice", "bob"}
    assert {item.producer_id for item in bob_status} == {"alice", "bob"}

    bob.execute_transaction(
        (
            SQLStatement(
                "UPDATE observations SET artifact_locator = ? LIMIT 1",
                ("conflicting-local-value",),
            ),
        )
    )
    with pytest.raises(VerificationStoreError, match="conflicting immutable"):
        refresh_evidence(bob, read_source=remote)


def test_publication_configuration_is_explicit_and_actionable(
    tmp_path: Path,
    backend_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    record_report(backend_report, store, producer_id="producer")
    monkeypatch.delenv("STRIDEWEAVE_STATUS_PUBLISH_DESTINATION", raising=False)
    monkeypatch.delenv("STRIDEWEAVE_STATUS_READ_SOURCE", raising=False)

    with pytest.raises(
        VerificationStoreError, match=r"publish_destination.*configured"
    ):
        publish_evidence(store, producer_id="producer")
    with pytest.raises(VerificationStoreError, match=r"read_source.*configured"):
        refresh_evidence(store)
    with pytest.raises(VerificationStoreError, match="unsupported transport"):
        publish_evidence(
            store,
            producer_id="producer",
            publish_destination="https://evidence.invalid/project",
        )


def test_exchange_work_is_bounded_to_one_producer_current_snapshot_and_incoming_keys(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    remote = tmp_path / "exchange"
    source = DoltEvidenceStore(tmp_path / "source-store")
    record_report(backend_report, source, producer_id="alice")
    record_report(backend_report, source, producer_id="bob")
    tracked_source = _TrackingStore(source)

    first = publish_evidence(
        tracked_source, producer_id="alice", publish_destination=remote
    )

    assert tracked_source.queries
    assert all(" WHERE " in query.template for query in tracked_source.queries)
    current = next(remote.glob("contributors/*/current.json"))
    snapshot = json.loads(current.read_text(encoding="utf-8"))
    assert {row["producer_id"] for row in snapshot["tables"]["observations"]} == {
        "alice"
    }

    record_report(backend_report, source, producer_id="alice", source_commit="second")
    second = publish_evidence(source, producer_id="alice", publish_destination=remote)

    assert second.snapshot_digest != first.snapshot_digest
    assert len(tuple(remote.glob("contributors/*/current.json"))) == 1
    current.with_name("ignored-history.json").write_text("not json", encoding="utf-8")

    destination = _TrackingStore(DoltEvidenceStore(tmp_path / "destination-store"))
    refresh_evidence(destination, read_source=remote)

    assert destination.queries
    assert all(" WHERE " in query.template for query in destination.queries)


def test_refresh_lookup_batches_preserve_existing_rows_and_cross_batch_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 560
    monkeypatch.setattr(publication_module, "_QUERY_ARGUMENT_BUDGET", budget)
    tables = {table: [] for table in publication_module._TABLE_COLUMNS}
    rows = []
    for index in range(24):
        observation_id = f"{index:064x}"
        row: dict[str, object] = {
            column: None for column in publication_module._TABLE_COLUMNS["observations"]
        }
        row.update(
            observation_id=observation_id,
            evidence_id="e" * 64,
            producer_id="producer",
        )
        rows.append(row)
    tables["observations"] = rows
    snapshot = {"tables": tables}
    existing = {
        rows[0]["observation_id"]: rows[0],
        rows[-1]["observation_id"]: rows[-1],
    }
    store = _LookupStore(existing)

    statements = publication_module._merge_statements(store, (snapshot,))

    observation_queries = [
        query for query in store.queries if " FROM observations " in query.template
    ]
    assert len(observation_queries) > 1
    assert all(
        len(_render_statement(query).encode("utf-8")) <= budget
        for query in store.queries
    )
    assert len(statements) == len(rows) - len(existing)
    assert {
        statement.parameters[0]
        for statement in statements
        if statement.template.startswith("INSERT INTO observations")
    } == {row["observation_id"] for row in rows[1:-1]}

    conflicting = dict(rows[-1])
    conflicting["artifact_locator"] = "conflicting-local-value"
    conflict_store = _LookupStore(
        {
            rows[-1]["observation_id"]: conflicting,
        }
    )
    with pytest.raises(VerificationStoreError, match="conflicting immutable"):
        publication_module._merge_statements(conflict_store, (snapshot,))

    empty_store = _LookupStore({})
    empty_snapshot = {
        "tables": {table: [] for table in publication_module._TABLE_COLUMNS}
    }
    assert publication_module._merge_statements(empty_store, (empty_snapshot,)) == ()
    assert empty_store.queries == []


def test_publication_selection_batches_preserve_the_complete_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = "producer"
    run_id = "r" * 64
    specification_id = "s" * 64
    tables: dict[str, list[dict[str, object]]] = {
        table: [] for table in publication_module._TABLE_COLUMNS
    }

    def row(table: str, **values: object) -> dict[str, object]:
        result: dict[str, object] = {
            column: None for column in publication_module._TABLE_COLUMNS[table]
        }
        result.update(values)
        return result

    for index in range(32):
        evidence_id = f"{index:064x}"
        tables["observations"].append(
            row(
                "observations",
                observation_id=f"{index + 32:064x}",
                evidence_id=evidence_id,
                producer_id=producer,
            )
        )
        tables["evidence"].append(
            row("evidence", evidence_id=evidence_id, run_id=run_id)
        )
    tables["verification_runs"].append(
        row(
            "verification_runs",
            run_id=run_id,
            verification_spec_id=specification_id,
        )
    )
    tables["verification_specs"].append(
        row(
            "verification_specs",
            verification_spec_id=specification_id,
        )
    )

    unbatched_store = _PublicationGraphStore(tables)
    monkeypatch.setattr(publication_module, "_QUERY_ARGUMENT_BUDGET", 1024 * 1024)
    unbatched = publication_module._snapshot(unbatched_store, producer)

    budget = 1024
    batched_store = _PublicationGraphStore(tables)
    monkeypatch.setattr(publication_module, "_QUERY_ARGUMENT_BUDGET", budget)
    batched = publication_module._snapshot(batched_store, producer)

    evidence_queries = [
        query for query in batched_store.queries if " FROM evidence " in query.template
    ]
    assert len(evidence_queries) > 1
    assert all(
        len(_render_statement(query).encode("utf-8")) <= budget
        for query in batched_store.queries
    )
    assert batched == unbatched
    snapshot_tables = batched["tables"]
    assert isinstance(snapshot_tables, dict)
    for table, rows in snapshot_tables.items():
        keys = [publication_module._row_key(table, item) for item in rows]
        assert len(keys) == len(set(keys))


@pytest.mark.parametrize("mutation", ["evidence_identity", "missing_dependency"])
def test_refresh_rejects_digest_valid_inconsistent_snapshots_before_store_creation(
    tmp_path: Path,
    backend_report: VerificationReport,
    mutation: str,
) -> None:
    _require_dolt()
    producer_store = DoltEvidenceStore(tmp_path / "producer-store")
    remote = tmp_path / "exchange"
    record_report(backend_report, producer_store, producer_id="producer")
    publish_evidence(producer_store, producer_id="producer", publish_destination=remote)
    path = next(remote.glob("contributors/*/*.json"))

    def mutate(value: dict[str, object]) -> None:
        tables = value["tables"]
        assert isinstance(tables, dict)
        if mutation == "evidence_identity":
            rows = tables["evidence"]
            assert isinstance(rows, list)
            rows[0]["evidence_id"] = "f" * 64
        else:
            rows = tables["source_closures"]
            assert isinstance(rows, list)
            rows.pop()

    _rewrite_snapshot(path, mutate)
    consumer_path = tmp_path / "consumer-store"

    with pytest.raises(VerificationStoreError, match="facts do not match"):
        refresh_evidence(DoltEvidenceStore(consumer_path), read_source=remote)

    assert not (consumer_path / ".dolt").exists()


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("artifact_digest", "x", "artifact_digest"),
        ("source_commit", "s" * 256, "source_commit"),
        ("artifact_locator", "", "artifact_locator"),
        ("producer_id", "p" * 256, "producer_id"),
        ("recorded_at_utc", "not-a-time", "recorded_at"),
    ),
)
def test_refresh_rejects_digest_valid_malformed_observation_provenance(
    tmp_path: Path,
    backend_report: VerificationReport,
    field: str,
    invalid: str,
    message: str,
) -> None:
    _require_dolt()
    producer_store = DoltEvidenceStore(tmp_path / "producer-store")
    remote = tmp_path / "exchange"
    record_report(backend_report, producer_store, producer_id="producer")
    publish_evidence(producer_store, producer_id="producer", publish_destination=remote)
    path = next(remote.glob("contributors/*/current.json"))

    def mutate(value: dict[str, object]) -> None:
        tables = value["tables"]
        assert isinstance(tables, dict)
        observations = tables["observations"]
        assert isinstance(observations, list)
        if field == "producer_id":
            value["producer_id"] = invalid
        for row in observations:
            row[field] = invalid
            identity = {
                key: row[key]
                for key in (
                    "artifact_digest",
                    "artifact_locator",
                    "evidence_id",
                    "producer_id",
                    "source_commit",
                )
            }
            row["observation_id"] = hashlib.sha256(
                json.dumps(
                    identity,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            row["observation_json"] = json.dumps(
                {
                    **identity,
                    "observation_id": row["observation_id"],
                    "recorded_at_utc": row["recorded_at_utc"],
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )

    _rewrite_snapshot(path, mutate)
    if field == "producer_id":
        namespace = hashlib.sha256(invalid.encode()).hexdigest()
        destination = remote / "contributors" / namespace / "current.json"
        destination.parent.mkdir(parents=True)
        path.rename(destination)
    consumer_path = tmp_path / "consumer-store"

    with pytest.raises(VerificationStoreError, match=message):
        refresh_evidence(DoltEvidenceStore(consumer_path), read_source=remote)

    assert not (consumer_path / ".dolt").exists()


def test_cli_exchange_occurs_only_behind_explicit_flags(
    tmp_path: Path,
    backend_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_dolt()
    report_path = tmp_path / "report.jsonl"
    store_path = tmp_path / "evidence-store"
    remote = tmp_path / "exchange"
    backend_report.write(report_path)

    def unexpected_exchange(*args: object, **kwargs: object):
        raise AssertionError("exchange should require an explicit flag")

    original_publish = status_cli_module.publish_evidence
    original_refresh = status_cli_module.refresh_evidence
    monkeypatch.setattr(status_cli_module, "publish_evidence", unexpected_exchange)
    monkeypatch.setattr(status_cli_module, "refresh_evidence", unexpected_exchange)
    assert (
        status_main(
            [
                "record",
                "--report",
                str(report_path),
                "--producer",
                "cli-producer",
                "--store",
                str(store_path),
            ]
        )
        == 0
    )
    assert (
        status_main(
            [
                "status",
                "--arch",
                load_compilation_manifest().target.architecture,
                "--store",
                str(store_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    monkeypatch.setattr(status_cli_module, "publish_evidence", original_publish)
    monkeypatch.setattr(status_cli_module, "refresh_evidence", original_refresh)
    assert (
        status_main(
            [
                "record",
                "--report",
                str(report_path),
                "--producer",
                "cli-producer",
                "--store",
                str(store_path),
                "--publish",
                "--publish-destination",
                str(remote),
            ]
        )
        == 0
    )
    consumer = tmp_path / "consumer-store"
    assert (
        status_main(
            [
                "status",
                "--arch",
                load_compilation_manifest().target.architecture,
                "--store",
                str(consumer),
                "--refresh",
                "--read-source",
                str(remote),
            ]
        )
        == 0
    )
    assert "Refreshed 1 contributor snapshots" in capsys.readouterr().out
