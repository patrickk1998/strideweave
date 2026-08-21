from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import synthetic_evidence

from strideweave.verification import VerificationReport
from strideweave.verification.store import (
    DoltEvidenceStore,
    publish_evidence,
    refresh_evidence,
)
from strideweave.verification.store.base import (
    SQLStatement,
    VerificationStoreError,
    render_statement,
)
from strideweave.verification.store.publication import (
    _BATCH_BYTES,
    _in_batches,
    _validate_graph,
)
from strideweave.verification.store.recording import (
    _canonical_json,
    _digest,
    _record_validated_report,
)


def test_snapshot_validation_requires_complete_graph() -> None:
    with pytest.raises((KeyError, VerificationStoreError)):
        _validate_graph(
            {"producer_id": "p", "tables": {}},
            __import__("pathlib").Path("current.json"),
        )


def test_synthetic_reports_are_distinct_for_specializations() -> None:
    first = synthetic_evidence.synthetic_report(specialization=128)
    second = synthetic_evidence.synthetic_report(specialization=256)
    assert first.header is not None and second.header is not None
    assert (
        first.header.compilation_bundle.bundle_id
        != second.header.compilation_bundle.bundle_id
    )


def test_publish_refresh_round_trip_is_idempotent(
    evidence_store_path, tmp_path, synthetic_report
) -> None:
    source = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, source, producer_id="p")
    exchange = tmp_path / "exchange"
    publish_evidence(source, producer_id="p", publish_destination=exchange)
    destination = DoltEvidenceStore(evidence_store_path())
    first = refresh_evidence(destination, read_source=exchange)
    second = refresh_evidence(destination, read_source=exchange)
    assert (
        first.observation_count
        == second.observation_count
        == len(synthetic_report.records)
    )
    count = destination.query(
        SQLStatement("SELECT COUNT(*) AS count FROM observations")
    )[0]["count"]
    assert count == len(synthetic_report.records)


def test_query_batches_include_the_final_order_by_in_rendered_budget() -> None:
    prefix = "SELECT value FROM facts WHERE id IN ("
    suffix = " ORDER BY value, id"
    values = tuple(f"{index:064x}" for index in range(2_000))
    batches = _in_batches(values, prefix, suffix)
    assert len(batches) > 1
    assert all(
        len(
            render_statement(
                SQLStatement(
                    prefix + ", ".join("?" for _ in batch) + ")" + suffix, batch
                )
            ).encode("utf-8")
        )
        <= _BATCH_BYTES
        for batch in batches
    )


def _published_snapshot(
    source: DoltEvidenceStore, root: Path, report: VerificationReport
) -> tuple[Path, dict[str, Any]]:
    _record_validated_report(report, source, producer_id="p")
    publish_evidence(source, producer_id="p", publish_destination=root)
    path = next((root / "contributors").glob("*/current.json"))
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "field,value", [("schema_version", "v2"), ("snapshot_digest", "0" * 64)]
)
def test_bad_envelope_rejects_before_destination_initialization(
    evidence_store_path, tmp_path, synthetic_report, field, value
) -> None:
    root = tmp_path / "exchange"
    path, snapshot = _published_snapshot(
        DoltEvidenceStore(evidence_store_path()), root, synthetic_report
    )
    snapshot[field] = value
    path.write_text(_canonical_json(snapshot) + "\n", encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(VerificationStoreError):
        refresh_evidence(DoltEvidenceStore(destination), read_source=root)
    assert not destination.exists()


@pytest.mark.parametrize(
    "table,column,value",
    [("evidence", "outcome", "failed"), ("observations", "producer_id", "other")],
)
def test_rehashed_graph_tampering_rejects_before_destination_mutation(
    evidence_store_path, tmp_path, synthetic_report, table, column, value
) -> None:
    root = tmp_path / "exchange"
    path, snapshot = _published_snapshot(
        DoltEvidenceStore(evidence_store_path()), root, synthetic_report
    )
    snapshot["tables"][table][0][column] = value
    identity = {
        key: snapshot[key] for key in ("producer_id", "schema_version", "tables")
    }
    snapshot["snapshot_digest"] = _digest(identity)
    path.write_text(_canonical_json(snapshot) + "\n", encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(VerificationStoreError):
        refresh_evidence(DoltEvidenceStore(destination), read_source=root)
    assert not destination.exists()


@pytest.mark.parametrize("value", (None, "not-a-digest", "0" * 64))
def test_todo_provenance_snapshot_tampering_rejects_before_destination_mutation(
    evidence_store_path, tmp_path, synthetic_report, value
) -> None:
    root = tmp_path / "exchange"
    path, snapshot = _published_snapshot(
        DoltEvidenceStore(evidence_store_path()), root, synthetic_report
    )
    run = snapshot["tables"]["verification_runs"][0]
    if value is None:
        del run["todo_provenance_digest"]
    else:
        run["todo_provenance_digest"] = value
    identity = {
        key: snapshot[key] for key in ("producer_id", "schema_version", "tables")
    }
    snapshot["snapshot_digest"] = _digest(identity)
    path.write_text(_canonical_json(snapshot) + "\n", encoding="utf-8")
    destination = tmp_path / "destination"

    with pytest.raises(VerificationStoreError):
        refresh_evidence(DoltEvidenceStore(destination), read_source=root)

    assert not destination.exists()


def test_existing_immutable_conflict_rejects_refresh_without_new_rows(
    evidence_store_path, tmp_path, synthetic_report
) -> None:
    root = tmp_path / "exchange"
    source = DoltEvidenceStore(evidence_store_path())
    _published_snapshot(source, root, synthetic_report)
    destination = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, destination, producer_id="p")
    before = destination.query(
        SQLStatement("SELECT COUNT(*) AS count FROM observations")
    )[0]["count"]
    destination.execute_transaction(
        (SQLStatement("UPDATE compilation_receipts SET provider='forged' LIMIT 1"),)
    )
    with pytest.raises(VerificationStoreError, match="conflicting immutable"):
        refresh_evidence(destination, read_source=root)
    after = destination.query(
        SQLStatement("SELECT COUNT(*) AS count FROM observations")
    )[0]["count"]
    assert after == before


def test_existing_todo_provenance_conflict_rejects_refresh_without_new_rows(
    evidence_store_path, tmp_path, synthetic_report
) -> None:
    root = tmp_path / "exchange"
    source = DoltEvidenceStore(evidence_store_path())
    _published_snapshot(source, root, synthetic_report)
    destination = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, destination, producer_id="p")
    before = destination.query(
        SQLStatement("SELECT COUNT(*) AS count FROM observations")
    )[0]["count"]
    destination.execute_transaction(
        (
            SQLStatement(
                "UPDATE verification_runs SET todo_provenance_digest=?",
                ("0" * 64,),
            ),
        )
    )

    with pytest.raises(VerificationStoreError, match="conflicting immutable"):
        refresh_evidence(destination, read_source=root)

    after = destination.query(
        SQLStatement("SELECT COUNT(*) AS count FROM observations")
    )[0]["count"]
    assert after == before
