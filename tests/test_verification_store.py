from __future__ import annotations

from strideweave.verification.store import DoltEvidenceStore, SQLStatement
from strideweave.verification.store.dolt import default_store_path
from strideweave.verification.store.recording import _record_validated_report


def test_default_store_path_is_lazy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STRIDEWEAVE_STATUS_HOME", str(tmp_path))
    path = default_store_path()
    assert path == tmp_path / "strideweave/kernel-evidence"
    assert not path.exists()


def test_fresh_store_persists_and_indexes_todo_provenance(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, store, producer_id="producer")

    run = store.query(
        SQLStatement("SELECT todo_provenance_digest FROM verification_runs LIMIT 1")
    )[0]
    indexes = store.query(SQLStatement("SHOW INDEX FROM verification_runs"))

    assert isinstance(run["todo_provenance_digest"], str)
    assert len(run["todo_provenance_digest"]) == 64
    assert any(
        row.get("Key_name") == "verification_runs_todo_provenance"
        and row.get("Column_name") == "todo_provenance_digest"
        for row in indexes
    )
    assert any(
        row.get("Key_name") == "verification_runs_todo_provenance"
        and row.get("Column_name") == "run_id"
        for row in indexes
    )
