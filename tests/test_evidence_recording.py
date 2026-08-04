from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import strideweave as sw
import strideweave.verification.store.recording as recording_module
from strideweave.verification import (
    EvidenceRecord,
    OracleCertificate,
    VerificationReport,
)
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
    record_report,
)


def _require_dolt() -> None:
    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


@pytest.fixture(scope="module")
def backend_report() -> VerificationReport:
    return sw.test_backend()


def _count(store: DoltEvidenceStore, table: str) -> int:
    rows = store.query(SQLStatement(f"SELECT COUNT(*) AS count FROM {table}"))
    value = rows[0]["count"]
    assert isinstance(value, int)
    return value


def test_record_persists_every_raw_fact_and_observation(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    artifact_digest = "a" * 64

    result = record_report(
        backend_report,
        store,
        producer_id="test-runner",
        source_commit="commit-123",
        artifact_locator="artifact://verification/report.jsonl",
        artifact_digest=artifact_digest,
    )

    assert result.evidence_count == len(backend_report.records)
    assert _count(store, "verification_runs") == 1
    assert _count(store, "evidence") == len(backend_report.records)
    assert _count(store, "observations") == len(backend_report.records)
    assert _count(store, "kernel_builds") > 0
    assert _count(store, "source_closure_inputs") > 0
    assert _count(store, "tolerance_policies") > 0
    assert _count(store, "oracle_references") > 0
    metadata = store.query(
        SQLStatement(
            "SELECT o.producer_id, o.source_commit, o.artifact_locator, "
            "o.artifact_digest, e.outcome, e.deviations_json, e.plan_json, "
            "t.definition_json AS tolerance_json, k.receipt_json, "
            "c.descriptor_json AS closure_json, b.descriptor_json AS toolchain_json, "
            "v.descriptor_json AS target_json "
            "FROM observations o JOIN evidence e ON e.evidence_id = o.evidence_id "
            "JOIN tolerance_policies t ON t.tolerance_policy_id = e.tolerance_policy_id "
            "LEFT JOIN kernel_builds k ON k.kernel_build_id = e.kernel_build_id "
            "LEFT JOIN source_closures c ON c.closure_id = k.closure_id "
            "LEFT JOIN build_toolchains b ON b.toolchain_id = k.toolchain_id "
            "LEFT JOIN verification_targets v ON v.target_id = k.target_id "
            "WHERE e.plan_json IS NOT NULL AND k.kernel_build_id IS NOT NULL LIMIT 1"
        )
    )[0]
    assert metadata["producer_id"] == "test-runner"
    assert metadata["source_commit"] == "commit-123"
    assert metadata["artifact_locator"] == "artifact://verification/report.jsonl"
    assert metadata["artifact_digest"] == artifact_digest
    assert metadata["outcome"] in {"passed", "deferred"}
    assert metadata["deviations_json"]
    assert metadata["plan_json"]
    assert metadata["tolerance_json"]
    assert metadata["receipt_json"]
    assert metadata["closure_json"]
    assert metadata["toolchain_json"]
    assert metadata["target_json"]
    deferred_count = store.query(
        SQLStatement(
            "SELECT COUNT(*) AS count FROM evidence WHERE outcome = 'deferred'"
        )
    )[0]["count"]
    assert isinstance(deferred_count, int)
    assert deferred_count > 0


def test_duplicate_ingestion_is_idempotent_and_distinct_producers_coexist(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    first = record_report(backend_report, store, producer_id="producer-a")
    repeated = record_report(backend_report, store, producer_id="producer-a")

    assert repeated == first
    assert _count(store, "verification_runs") == 1
    assert _count(store, "evidence") == len(backend_report.records)
    assert _count(store, "observations") == len(backend_report.records)

    record_report(
        backend_report,
        store,
        producer_id="producer-b",
        source_commit="different-observation",
    )

    assert _count(store, "verification_runs") == 1
    assert _count(store, "evidence") == len(backend_report.records)
    assert _count(store, "observations") == 2 * len(backend_report.records)
    producers = store.query(
        SQLStatement(
            "SELECT DISTINCT producer_id FROM observations ORDER BY producer_id"
        )
    )
    assert tuple(row["producer_id"] for row in producers) == (
        "producer-a",
        "producer-b",
    )


def test_stale_provenance_fails_before_the_store_is_created(
    tmp_path: Path,
    backend_report: VerificationReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_dolt()
    path = tmp_path / "evidence-store"
    original_bind = recording_module.bind_report

    def stale_bind(
        records: Sequence[EvidenceRecord],
        certificates: Sequence[OracleCertificate],
        *,
        certificate_facts_override: Sequence[Mapping[str, Any]] | None = None,
    ):
        records, header = original_bind(
            records,
            certificates,
            certificate_facts_override=certificate_facts_override,
        )
        return records, replace(header, header_digest="0" * 64)

    monkeypatch.setattr(recording_module, "bind_report", stale_bind)

    with pytest.raises(VerificationStoreError, match="stale"):
        record_report(backend_report, DoltEvidenceStore(path), producer_id="producer")
    assert not path.exists()


def test_inconsistent_certificate_fails_before_the_store_is_created(
    tmp_path: Path,
    backend_report: VerificationReport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = backend_report.to_jsonl().splitlines()
    header = json.loads(lines[0])
    evidence = [json.loads(line) for line in lines[1:]]
    certificate = next(
        item for item in header["certificates"] if item["kernel_id"] == "cpu.reduce_sum"
    )
    previous = certificate["certificate_digest"]
    certificate["evidence_digest"] = "0" * 64
    certificate["certificate_digest"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in certificate.items()
                if key != "certificate_digest"
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for record in evidence:
        if record["consumed_certificate_digest"] == previous:
            record["consumed_certificate_digest"] = certificate["certificate_digest"]
    header["header_digest"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in header.items() if key != "header_digest"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    report_path = tmp_path / "inconsistent.jsonl"
    report_path.write_text(
        "\n".join(
            json.dumps(value, separators=(",", ":"), sort_keys=True)
            for value in (header, *evidence)
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "evidence-store"

    status = status_main(
        [
            "record",
            "--report",
            str(report_path),
            "--producer",
            "certificate-test",
            "--store",
            str(store_path),
        ]
    )

    assert status == 2
    assert "certificate" in capsys.readouterr().err
    assert not store_path.exists()


def test_record_cli_persists_locally_without_publication(
    tmp_path: Path,
    backend_report: VerificationReport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_dolt()
    report_path = tmp_path / "report.jsonl"
    store_path = tmp_path / "evidence-store"
    backend_report.write(report_path)

    status = status_main(
        [
            "record",
            "--report",
            str(report_path),
            "--producer",
            "cli-test",
            "--store",
            str(store_path),
            "--json",
        ]
    )

    assert status == 0
    assert '"evidence_count":' in capsys.readouterr().out
    assert _count(DoltEvidenceStore(store_path), "verification_runs") == 1
