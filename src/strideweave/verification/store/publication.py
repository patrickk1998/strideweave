"""Canonical, atomic local-file exchange for the schema-v3 evidence graph."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..model import VerificationReport
from ..provenance import compilation_receipt_json_object
from ._identity import _canonical_json, _digest, _todo_provenance_digest
from .base import (
    EvidenceStore,
    SQLStatement,
    SQLValue,
    VerificationStoreError,
    render_statement,
)
from .recording import _recording_time

_SCHEMA = "strideweave.kernel-evidence-snapshot.v3"
_PUBLISH_ENVIRONMENT = "STRIDEWEAVE_STATUS_PUBLISH_DESTINATION"
_READ_ENVIRONMENT = "STRIDEWEAVE_STATUS_READ_SOURCE"
_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "verification_runs": (
        "run_id",
        "report_digest",
        "report_schema",
        "selected_target_profile",
        "oracle_profile",
        "bundle_id",
        "todo_provenance_digest",
        "header_digest",
        "report_json",
    ),
    "compilation_receipts": (
        "receipt_id",
        "receipt_kind",
        "profile_id",
        "provider",
        "logical_kernel_id",
        "logical_kernel_variant",
        "receipt_json",
    ),
    "run_compilation_receipts": ("run_id", "receipt_id"),
    "evidence": (
        "evidence_id",
        "run_id",
        "receipt_id",
        "requirement_id",
        "stage",
        "test_class",
        "case_id",
        "operation_name",
        "kernel_id",
        "variant",
        "outcome",
        "record_json",
    ),
    "observations": (
        "observation_id",
        "evidence_id",
        "producer_id",
        "source_commit",
        "recorded_at_utc",
        "artifact_locator",
        "artifact_digest",
        "observation_json",
    ),
}
_PRIMARY_KEYS = {
    "verification_runs": ("run_id",),
    "compilation_receipts": ("receipt_id",),
    "run_compilation_receipts": ("run_id", "receipt_id"),
    "evidence": ("evidence_id",),
    "observations": ("observation_id",),
}
_BATCH_BYTES = 64 * 1024


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise VerificationStoreError(f"{field} must be a non-empty string")
    return value


def _endpoint(
    value: str | os.PathLike[str] | None, *, environment: str, field: str
) -> Path:
    configured = os.environ.get(environment) if value is None else os.fspath(value)
    if not configured:
        raise VerificationStoreError(
            f"{field} is not configured; pass it explicitly or set {environment}"
        )
    parsed = urlparse(configured)
    if parsed.scheme not in ("", "file"):
        raise VerificationStoreError(
            f"{field} uses unsupported transport {parsed.scheme!r}; only local and file endpoints are supported"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise VerificationStoreError(
                f"{field} file endpoint must name the local host"
            )
        return Path(unquote(parsed.path)).expanduser()
    return Path(configured).expanduser()


def _rows(
    store: EvidenceStore, statement: str, values: tuple[SQLValue, ...] = ()
) -> list[dict[str, object]]:
    return [dict(row) for row in store.query(SQLStatement(statement, values))]


def _in_batches(
    values: Sequence[str], prefix: str, suffix: str
) -> tuple[tuple[str, ...], ...]:
    """Split a deterministic IN-list before its rendered SQL reaches the budget."""

    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    for value in values:
        candidate = (*current, value)
        statement = SQLStatement(
            prefix + ", ".join("?" for _ in candidate) + ")" + suffix, candidate
        )
        if current and len(render_statement(statement).encode("utf-8")) > _BATCH_BYTES:
            batches.append(tuple(current))
            current = [value]
        else:
            current.append(value)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _rows_by_ids(
    store: EvidenceStore,
    *,
    select: str,
    column: str,
    identifiers: Sequence[str],
    order: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    suffix = f" ORDER BY {order}"
    for batch in _in_batches(
        tuple(sorted(set(identifiers))), f"{select} WHERE {column} IN (", suffix
    ):
        rows.extend(
            _rows(
                store,
                f"{select} WHERE {column} IN ({', '.join('?' for _ in batch)}){suffix}",
                batch,
            )
        )
    return rows


def _snapshot_rows(
    store: EvidenceStore, producer: str
) -> dict[str, list[dict[str, object]]]:
    observations = _rows(
        store,
        "SELECT observation_id, evidence_id, producer_id, source_commit, recorded_at_utc, artifact_locator, artifact_digest, observation_json FROM observations WHERE producer_id=? ORDER BY observation_id",
        (producer,),
    )
    evidence_ids = tuple(
        _text(row["evidence_id"], "observation.evidence_id") for row in observations
    )
    if not evidence_ids:
        raise VerificationStoreError(
            f"verification store has no observations for producer {producer!r}"
        )
    evidence = _rows_by_ids(
        store,
        select="SELECT evidence_id, run_id, receipt_id, requirement_id, stage, test_class, case_id, operation_name, kernel_id, variant, outcome, record_json FROM evidence",
        column="evidence_id",
        identifiers=evidence_ids,
        order="evidence_id",
    )
    run_ids = tuple(sorted(_text(row["run_id"], "evidence.run_id") for row in evidence))
    runs = _rows_by_ids(
        store,
        select="SELECT run_id, report_digest, report_schema, selected_target_profile, oracle_profile, bundle_id, todo_provenance_digest, header_digest, report_json FROM verification_runs",
        column="run_id",
        identifiers=run_ids,
        order="run_id",
    )
    links = _rows_by_ids(
        store,
        select="SELECT run_id, receipt_id FROM run_compilation_receipts",
        column="run_id",
        identifiers=run_ids,
        order="run_id, receipt_id",
    )
    receipt_ids = tuple(
        sorted(
            _text(row["receipt_id"], "run_compilation_receipts.receipt_id")
            for row in links
        )
    )
    receipts = _rows_by_ids(
        store,
        select="SELECT receipt_id, receipt_kind, profile_id, provider, logical_kernel_id, logical_kernel_variant, receipt_json FROM compilation_receipts",
        column="receipt_id",
        identifiers=receipt_ids,
        order="receipt_id",
    )
    return {
        "verification_runs": runs,
        "compilation_receipts": receipts,
        "run_compilation_receipts": links,
        "evidence": evidence,
        "observations": observations,
    }


def _snapshot(store: EvidenceStore, producer_id: str) -> dict[str, object]:
    producer = _text(producer_id, "producer_id")
    identity = {
        "producer_id": producer,
        "schema_version": _SCHEMA,
        "tables": _snapshot_rows(store, producer),
    }
    return {**identity, "snapshot_digest": _digest(identity)}


@dataclass(frozen=True, slots=True)
class PublishResult:
    producer_id: str
    snapshot_digest: str
    observation_count: int
    destination: str


def publish_evidence(
    store: EvidenceStore,
    *,
    producer_id: str,
    publish_destination: str | os.PathLike[str] | None = None,
) -> PublishResult:
    """Publish one producer's complete v3 graph to an explicit local endpoint."""
    snapshot = _snapshot(store, producer_id)
    root = _endpoint(
        publish_destination,
        environment=_PUBLISH_ENVIRONMENT,
        field="publish_destination",
    )
    destination = (
        root / "contributors" / hashlib.sha256(producer_id.encode()).hexdigest()
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "current.json"
    contents = _canonical_json(snapshot) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination, delete=False
    ) as stream:
        stream.write(contents)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    tables = snapshot["tables"]
    assert isinstance(tables, Mapping)
    return PublishResult(
        producer_id,
        _text(snapshot["snapshot_digest"], "snapshot_digest"),
        len(tables["observations"]),
        str(root),
    )


def _canonical_object(text: str, source: Path) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise VerificationStoreError(
            f"read_source contains malformed snapshot {source.name!r}"
        ) from error
    if type(value) is not dict or set(value) != {
        "producer_id",
        "schema_version",
        "snapshot_digest",
        "tables",
    }:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has invalid fields"
        )
    if _canonical_json(value) + "\n" != text:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} is not canonical JSON"
        )
    if value["schema_version"] != _SCHEMA:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has unsupported v2 or unknown schema"
        )
    identity = {key: value[key] for key in ("producer_id", "schema_version", "tables")}
    if value["snapshot_digest"] != _digest(identity):
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has a mismatched digest"
        )
    producer = _text(value["producer_id"], "snapshot producer_id")
    if source.parent.name != hashlib.sha256(producer.encode()).hexdigest():
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} is in the wrong producer namespace"
        )
    tables = value["tables"]
    if type(tables) is not dict or set(tables) != set(_TABLE_COLUMNS):
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has invalid tables"
        )
    for table, columns in _TABLE_COLUMNS.items():
        if type(tables[table]) is not list:
            raise VerificationStoreError(
                f"read_source snapshot {source.name!r} has non-array {table}"
            )
        for row in tables[table]:
            if type(row) is not dict or set(row) != set(columns):
                raise VerificationStoreError(
                    f"read_source snapshot {source.name!r} has invalid {table} row"
                )
    _validate_graph(value, source)
    return value


def _validate_graph(snapshot: Mapping[str, object], source: Path) -> None:
    tables = snapshot["tables"]
    assert isinstance(tables, Mapping)
    producer = _text(snapshot["producer_id"], "snapshot producer_id")
    indexed = {
        name: {
            tuple(row[key] for key in _PRIMARY_KEYS[name]): row for row in tables[name]
        }
        for name in _TABLE_COLUMNS
    }
    if any(len(indexed[name]) != len(tables[name]) for name in _TABLE_COLUMNS):
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has duplicate fact identities"
        )
    runs = indexed["verification_runs"]
    receipts = indexed["compilation_receipts"]
    links = indexed["run_compilation_receipts"]
    evidence = indexed["evidence"]
    observed_evidence_ids = {
        row["evidence_id"] for row in indexed["observations"].values()
    }
    if set(evidence) != {(item,) for item in observed_evidence_ids}:
        raise VerificationStoreError("snapshot contains orphan or missing evidence")
    expected_run_ids = {row["run_id"] for row in evidence.values()}
    if set(runs) != {(item,) for item in expected_run_ids}:
        raise VerificationStoreError("snapshot contains orphan or missing runs")
    for run in runs.values():
        try:
            report = VerificationReport.from_jsonl(
                _text(run["report_json"], "report_json")
            )
        except ValueError as error:
            raise VerificationStoreError(
                "snapshot contains an invalid v3 report"
            ) from error
        if (
            report.header is None
            or run["report_digest"]
            != hashlib.sha256(run["report_json"].encode()).hexdigest()
            or run["run_id"] != _digest({"report_digest": run["report_digest"]})
        ):
            raise VerificationStoreError(
                "snapshot run identity does not match its report"
            )
        if (
            run["report_schema"] != report.schema_version
            or run["selected_target_profile"] != report.header.selected_target_profile
            or run["oracle_profile"] != report.header.oracle_profile
            or run["bundle_id"] != report.header.compilation_bundle.bundle_id
            or run["todo_provenance_digest"] != _todo_provenance_digest(report)
            or run["header_digest"] != report.header.header_digest
        ):
            raise VerificationStoreError(
                "snapshot run columns disagree with its report"
            )
        expected = {
            receipt.receipt_id: _canonical_json(
                compilation_receipt_json_object(receipt)
            )
            for receipt in report.header.compilation_bundle.receipts
        }
        actual = {
            link["receipt_id"]
            for link in links.values()
            if link["run_id"] == run["run_id"]
        }
        if actual != set(expected):
            raise VerificationStoreError(
                "snapshot omits or adds report receipt relationships"
            )
        for receipt_id, receipt_json in expected.items():
            row = receipts.get((receipt_id,))
            if row is None or row["receipt_json"] != receipt_json:
                raise VerificationStoreError(
                    "snapshot receipt facts do not match embedded report"
                )
            receipt = json.loads(receipt_json)
            if (
                row["receipt_kind"] != receipt["kind"]
                or row["profile_id"] != receipt["profile_id"]
                or row["provider"] != receipt["provider"]
                or row["logical_kernel_id"] != receipt["logical_kernel"]["kernel_id"]
                or row["logical_kernel_variant"] != receipt["logical_kernel"]["variant"]
            ):
                raise VerificationStoreError(
                    "snapshot receipt columns disagree with its receipt"
                )
        expected_records = {
            _digest(
                {"record": record.as_json_object(), "run_id": run["run_id"]}
            ): _canonical_json(record.as_json_object())
            for record in report.records
        }
        actual_evidence = {
            row["evidence_id"]: row
            for row in evidence.values()
            if row["run_id"] == run["run_id"]
        }
        if set(actual_evidence) != set(expected_records) or any(
            row["record_json"] != expected_records[key]
            for key, row in actual_evidence.items()
        ):
            raise VerificationStoreError(
                "snapshot evidence facts do not match embedded report"
            )
        for record in report.records:
            evidence_id = _digest(
                {"record": record.as_json_object(), "run_id": run["run_id"]}
            )
            row = actual_evidence[evidence_id]
            if (
                row["receipt_id"] != record.compilation_receipt_id
                or row["requirement_id"] != record.requirement_id
                or row["stage"] != record.stage.value
                or row["test_class"] != record.test_class.value
                or row["case_id"] != record.case.case_id
                or row["operation_name"] != record.case.operation
                or row["kernel_id"] != record.case.kernel_id
                or row["variant"] != record.case.variant
                or row["outcome"] != record.outcome.value
            ):
                raise VerificationStoreError(
                    "snapshot evidence columns disagree with its record"
                )
    expected_receipts = {link["receipt_id"] for link in links.values()}
    if set(receipts) != {(item,) for item in expected_receipts}:
        raise VerificationStoreError("snapshot contains orphan or missing receipts")
    if any(
        (link["run_id"],) not in runs or (link["receipt_id"],) not in receipts
        for link in links.values()
    ):
        raise VerificationStoreError("snapshot contains invalid receipt relationships")
    for observation in indexed["observations"].values():
        if (
            observation["producer_id"] != producer
            or (observation["evidence_id"],) not in evidence
        ):
            raise VerificationStoreError(
                "snapshot observation has an invalid relationship"
            )
        identity = {
            "artifact_digest": observation["artifact_digest"],
            "artifact_locator": observation["artifact_locator"],
            "evidence_id": observation["evidence_id"],
            "producer_id": producer,
            "source_commit": observation["source_commit"],
        }
        if observation["observation_id"] != _digest(identity):
            raise VerificationStoreError("snapshot observation identity does not match")
        try:
            payload = json.loads(
                _text(observation["observation_json"], "observation_json")
            )
            recorded_at = _recording_time(
                observation["recorded_at_utc"], "recorded_at_utc", allow_naive_utc=True
            )
        except (json.JSONDecodeError, VerificationStoreError) as error:
            raise VerificationStoreError("snapshot observation is malformed") from error
        if _canonical_json(payload) != observation["observation_json"]:
            raise VerificationStoreError("snapshot observation JSON is not canonical")
        expected = {
            **identity,
            "observation_id": observation["observation_id"],
            "recorded_at_utc": recorded_at.isoformat(),
        }
        if payload != expected:
            raise VerificationStoreError(
                "snapshot observation JSON disagrees with its row facts"
            )


def _merge(
    store: EvidenceStore, snapshots: Sequence[Mapping[str, object]]
) -> tuple[SQLStatement, ...]:
    incoming: dict[tuple[str, tuple[object, ...]], Mapping[str, object]] = {}
    for snapshot in snapshots:
        tables = snapshot["tables"]
        assert isinstance(tables, Mapping)
        for table, columns in _TABLE_COLUMNS.items():
            for row in tables[table]:
                assert isinstance(row, Mapping)
                key = (table, tuple(row[column] for column in _PRIMARY_KEYS[table]))
                previous = incoming.setdefault(key, row)
                if previous != row:
                    raise VerificationStoreError(
                        f"refresh found conflicting immutable {table} identity {key[1]!r}"
                    )
    statements: list[SQLStatement] = []
    for (table, key), row in incoming.items():
        columns = _TABLE_COLUMNS[table]
        predicate = " AND ".join(f"{column}=?" for column in _PRIMARY_KEYS[table])
        existing = store.query(
            SQLStatement(
                f"SELECT {', '.join(columns)} FROM {table} WHERE {predicate}",
                key,  # type: ignore[arg-type]
            )
        )
        if existing:
            existing_row = dict(existing[0])
            if existing_row != dict(row):
                raise VerificationStoreError(
                    f"refresh found conflicting immutable {table} identity {key!r}"
                )
            continue
        statements.append(
            SQLStatement(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                tuple(row[column] for column in columns),  # type: ignore[arg-type]
            )
        )
    return tuple(statements)


@dataclass(frozen=True, slots=True)
class RefreshResult:
    snapshot_count: int
    observation_count: int
    read_source: str


def refresh_evidence(
    store: EvidenceStore, *, read_source: str | os.PathLike[str] | None = None
) -> RefreshResult:
    """Strictly validate every complete snapshot, then atomically merge them."""
    root = _endpoint(read_source, environment=_READ_ENVIRONMENT, field="read_source")
    paths = tuple(sorted((root / "contributors").glob("*/current.json")))
    if not paths:
        raise VerificationStoreError(
            f"read_source {str(root)!r} has no published contributor snapshots"
        )
    snapshots = tuple(
        _canonical_object(path.read_text(encoding="utf-8"), path) for path in paths
    )
    store.execute_transaction(_merge(store, snapshots))
    return RefreshResult(
        len(snapshots),
        sum(
            len(tables["observations"])
            for snapshot in snapshots
            for tables in (snapshot["tables"],)
            if isinstance(tables, Mapping)
        ),
        str(root),
    )  # type: ignore[index]
