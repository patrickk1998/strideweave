from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
import synthetic_evidence
from evidence_doubles import RecordingEvidenceStore

import strideweave.verification.status_cli as status_cli_module
import strideweave.verification.store.publication as publication_module
import strideweave.verification.store.recording as recording_module
from strideweave.verification import VerificationReport
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    EvidenceStore,
    SQLStatement,
    VerificationStoreError,
    publish_evidence,
    query_status,
    refresh_evidence,
)
from strideweave.verification.store.base import render_statement


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


StorePathFactory = Callable[..., Path]
ARCHITECTURE = synthetic_evidence.ARCHITECTURE


def _record(
    report: VerificationReport,
    store: EvidenceStore,
    producer: str,
    *,
    source_commit: str | None = None,
) -> recording_module.RecordResult:
    return recording_module._record_validated_report(
        report, store, producer_id=producer, source_commit=source_commit
    )


@pytest.fixture(scope="module")
def published_exchange(
    tmp_path_factory: pytest.TempPathFactory,
    dolt_data_directory: Path,
    dolt_server: object,
    synthetic_report: VerificationReport,
) -> Path:
    """Return one exchange holding a producer's complete published evidence.

    The rejection cases below differ only in how they corrupt a valid snapshot,
    so the evidence they corrupt is recorded and published once rather than once
    per case. Each case copies this exchange before mutating it.
    """

    store = DoltEvidenceStore(dolt_data_directory / "published-exchange-producer")
    remote = tmp_path_factory.mktemp("published-exchange")
    _record(synthetic_report, store, "producer")
    publish_evidence(store, producer_id="producer", publish_destination=remote)
    return remote


def _copied_exchange(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


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
    tmp_path: Path,
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
) -> None:
    remote = tmp_path / "exchange"
    alice = DoltEvidenceStore(evidence_store_path("alice"))
    bob = DoltEvidenceStore(evidence_store_path("bob"))
    _record(synthetic_report, alice, "alice")
    _record(synthetic_evidence.contradicting_report(synthetic_report), bob, "bob")

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
    alice_status = query_status(alice, architecture=ARCHITECTURE)
    bob_status = query_status(bob, architecture=ARCHITECTURE)
    expected = 2 * len(synthetic_report.records)
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


def test_closure_members_an_earlier_version_stored_are_never_republished(
    tmp_path: Path,
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
) -> None:
    remote = tmp_path / "exchange"
    source = DoltEvidenceStore(evidence_store_path("legacy-source"))
    _record(synthetic_report, source, "producer")
    closure = source.query(
        SQLStatement(
            "SELECT closure_id FROM source_closures WHERE root_kind = ? "
            "ORDER BY closure_id LIMIT 1",
            ("repository-source",),
        )
    )[0]["closure_id"]
    assert isinstance(closure, str)
    external_digest = "b" * 64
    source.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO source_closure_inputs (closure_id, input_ordinal, "
                "input_kind, input_uri, content_digest, descriptor_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    closure,
                    100000,
                    "external_header",
                    f"cpp-external://sha256/{external_digest}/legacy.hpp",
                    external_digest,
                    "{}",
                ),
            ),
        )
    )

    publish_evidence(source, producer_id="producer", publish_destination=remote)

    # This version records external members only inside closure identity, so a
    # snapshot carrying member rows for them would disagree with the facts its
    # own embedded reports rebuild and could never be refreshed anywhere.
    current = next(remote.glob("contributors/*/current.json"))
    snapshot = json.loads(current.read_text(encoding="utf-8"))
    members = snapshot["tables"]["source_closure_inputs"]
    assert members
    assert all(row["input_kind"] != "external_header" for row in members)
    destination = DoltEvidenceStore(evidence_store_path("legacy-destination"))
    result = refresh_evidence(destination, read_source=remote)
    assert result.observation_count == len(synthetic_report.records)


def test_publication_configuration_is_explicit_and_actionable(
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record(synthetic_report, store, "producer")
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
    tmp_path: Path,
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
) -> None:
    remote = tmp_path / "exchange"
    source = DoltEvidenceStore(evidence_store_path("source"))
    _record(synthetic_report, source, "alice")
    _record(synthetic_report, source, "bob")
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

    _record(synthetic_report, source, "alice", source_commit="second")
    second = publish_evidence(source, producer_id="alice", publish_destination=remote)

    assert second.snapshot_digest != first.snapshot_digest
    assert len(tuple(remote.glob("contributors/*/current.json"))) == 1
    current.with_name("ignored-history.json").write_text("not json", encoding="utf-8")

    destination = _TrackingStore(DoltEvidenceStore(evidence_store_path("destination")))
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
        len(render_statement(query).encode("utf-8")) <= budget
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
        len(render_statement(query).encode("utf-8")) <= budget
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
    evidence_store_path: StorePathFactory,
    published_exchange: Path,
    mutation: str,
) -> None:
    remote = _copied_exchange(published_exchange, tmp_path / "exchange")
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
    consumer_path = evidence_store_path("consumer")

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
    evidence_store_path: StorePathFactory,
    published_exchange: Path,
    field: str,
    invalid: str,
    message: str,
) -> None:
    remote = _copied_exchange(published_exchange, tmp_path / "exchange")
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
    consumer_path = evidence_store_path("consumer")

    with pytest.raises(VerificationStoreError, match=message):
        refresh_evidence(DoltEvidenceStore(consumer_path), read_source=remote)

    assert not (consumer_path / ".dolt").exists()


def test_cli_record_publishes_only_behind_an_explicit_flag(
    tmp_path: Path,
    backend_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Reading and validating a real report against the installed build is
    # native work, so the store and the exchange are substituted and only the
    # command's own decision to exchange is under test here.
    report_path = tmp_path / "report.jsonl"
    remote = tmp_path / "exchange"
    backend_report.write(report_path)
    monkeypatch.setattr(status_cli_module, "DoltEvidenceStore", RecordingEvidenceStore)

    def unexpected_exchange(*args: object, **kwargs: object) -> object:
        raise AssertionError("exchange should require an explicit flag")

    monkeypatch.setattr(status_cli_module, "publish_evidence", unexpected_exchange)
    monkeypatch.setattr(status_cli_module, "refresh_evidence", unexpected_exchange)
    arguments = [
        "record",
        "--report",
        str(report_path),
        "--producer",
        "cli-producer",
        "--store",
        str(tmp_path / "evidence"),
    ]

    assert status_main(arguments) == 0
    capsys.readouterr()

    published: list[tuple[str, object]] = []

    def publish(store: object, *, producer_id: str, publish_destination: object):
        published.append((producer_id, publish_destination))
        return publication_module.PublishResult(
            producer_id=producer_id,
            destination=str(publish_destination),
            observation_count=len(backend_report.records),
            snapshot_digest="d" * 64,
        )

    monkeypatch.setattr(status_cli_module, "publish_evidence", publish)

    assert (
        status_main([*arguments, "--publish", "--publish-destination", str(remote)])
        == 0
    )
    assert published == [("cli-producer", str(remote))]
    assert f"Published snapshot {'d' * 64}" in capsys.readouterr().out


def test_cli_status_refreshes_only_behind_an_explicit_flag(
    tmp_path: Path,
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = tmp_path / "exchange"
    producer_store = DoltEvidenceStore(evidence_store_path("producer"))
    _record(synthetic_report, producer_store, "cli-producer")
    publish_evidence(
        producer_store, producer_id="cli-producer", publish_destination=remote
    )
    consumer = evidence_store_path("consumer")

    def unexpected_refresh(*args: object, **kwargs: object) -> object:
        raise AssertionError("refresh should require an explicit flag")

    monkeypatch.setattr(status_cli_module, "refresh_evidence", unexpected_refresh)
    assert (
        status_main(["status", "--arch", ARCHITECTURE, "--store", str(consumer)]) == 0
    )
    assert "Observations for architecture" in capsys.readouterr().out

    monkeypatch.undo()
    assert (
        status_main(
            [
                "status",
                "--arch",
                ARCHITECTURE,
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
