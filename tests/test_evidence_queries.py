from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import strideweave as sw
from strideweave.verification import VerificationOutcome, VerificationReport
from strideweave.verification.provenance import load_compilation_manifest
from strideweave.verification.reporting import bind_report
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    query_stale,
    query_status,
    query_todo,
    record_report,
)


def _require_dolt() -> None:
    if shutil.which("dolt") is None:
        pytest.skip("local Dolt runtime is not installed")


@pytest.fixture(scope="module")
def backend_report() -> VerificationReport:
    return sw.test_backend()


def _architecture() -> str:
    return load_compilation_manifest().target.architecture


def _count(store: DoltEvidenceStore, table: str) -> int:
    value = store.query(SQLStatement(f"SELECT COUNT(*) AS count FROM {table}"))[0][
        "count"
    ]
    assert isinstance(value, int)
    return value


def test_status_filters_stably_and_preserves_contradictory_observations(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    original = backend_report.records[0]
    changed_records = (
        replace(
            original,
            outcome=VerificationOutcome.FAILED,
            diagnostic="independent producer observed a mismatch",
        ),
        *backend_report.records[1:],
    )
    assert backend_report.header is not None
    changed_records, changed_header = bind_report(
        changed_records,
        (),
        certificate_facts_override=backend_report.header.certificates,
    )
    changed_report = VerificationReport(
        changed_records, changed_header.schema_version, changed_header
    )
    record_report(backend_report, store, producer_id="producer-a")
    record_report(changed_report, store, producer_id="producer-b")

    observations = query_status(store, architecture=_architecture())
    order = tuple(
        (
            item.kernel_id,
            item.variant,
            item.test_class,
            item.case_id,
            item.producer_id,
            item.observation_id,
        )
        for item in observations
    )
    assert order == tuple(sorted(order))
    contradictory = tuple(
        item for item in observations if item.case_id == original.case.case_id
    )
    assert {item.producer_id for item in contradictory} == {
        "producer-a",
        "producer-b",
    }
    assert {item.outcome for item in contradictory} == {"passed", "failed"}

    filtered = query_status(
        store,
        architecture=_architecture(),
        kernel_id=original.case.kernel_id,
        variant=original.case.variant,
        test_class=original.test_class.value,
        producer_id="producer-b",
    )
    assert filtered
    assert all(item.producer_id == "producer-b" for item in filtered)
    assert all(item.kernel_id == original.case.kernel_id for item in filtered)


def test_stale_reports_each_changed_identity_axis_and_closure_member(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    result = record_report(backend_report, store, producer_id="producer")
    manifest = load_compilation_manifest()
    assert backend_report.header is not None
    fake_target = "1" * 64
    fake_toolchain = "2" * 64
    fake_specification = "3" * 64
    fake_policy = "4" * 64
    fake_oracle = "5" * 64
    closure_input = store.query(
        SQLStatement(
            "SELECT i.closure_id, i.input_ordinal FROM run_kernel_builds rb "
            "JOIN kernel_builds k ON k.kernel_build_id = rb.kernel_build_id "
            "JOIN source_closure_inputs i ON i.closure_id = k.closure_id "
            "WHERE rb.run_id = ? ORDER BY k.kernel_id, i.input_ordinal LIMIT 1",
            (result.run_id,),
        )
    )[0]
    closure_id = closure_input["closure_id"]
    input_ordinal = closure_input["input_ordinal"]
    assert isinstance(closure_id, str)
    assert isinstance(input_ordinal, int)
    store.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO verification_targets SELECT ?, architecture, vendor, "
                "operating_system, abi, endianness, pointer_bits, descriptor_json "
                "FROM verification_targets WHERE target_id = ?",
                (fake_target, manifest.target.target_id),
            ),
            SQLStatement(
                "INSERT INTO build_toolchains SELECT ?, provider_kind, compiler_id, "
                "compiler_version, target_triple, build_system, descriptor_json "
                "FROM build_toolchains WHERE toolchain_id = ?",
                (fake_toolchain, manifest.toolchain.toolchain_id),
            ),
            SQLStatement(
                "INSERT INTO verification_specs SELECT ?, spec_schema, manifest_digest, "
                "definition_json FROM verification_specs WHERE verification_spec_id = ?",
                (
                    fake_specification,
                    backend_report.header.verification_spec["verification_spec_id"],
                ),
            ),
            SQLStatement(
                "INSERT INTO tolerance_policies SELECT ?, policy_schema, comparison_kind, "
                "definition_json FROM tolerance_policies LIMIT 1",
                (fake_policy,),
            ),
            SQLStatement(
                "INSERT INTO oracle_references SELECT ?, oracle_kind, "
                "implementation_digest, source_closure_id, kernel_build_id, "
                "descriptor_json FROM oracle_references LIMIT 1",
                (fake_oracle,),
            ),
            SQLStatement(
                "UPDATE verification_runs SET native_manifest_digest = ?, "
                "verification_spec_id = ?, execution_target_id = ?, "
                "represented_target_id = ? WHERE run_id = ?",
                ("0" * 64, fake_specification, fake_target, fake_target, result.run_id),
            ),
            SQLStatement(
                "UPDATE kernel_builds SET toolchain_id = ? WHERE toolchain_id = ? LIMIT 1",
                (fake_toolchain, manifest.toolchain.toolchain_id),
            ),
            SQLStatement(
                "UPDATE evidence SET tolerance_policy_id = ?, oracle_reference_id = ? "
                "WHERE run_id = ? LIMIT 1",
                (fake_policy, fake_oracle, result.run_id),
            ),
            SQLStatement(
                "UPDATE source_closure_inputs SET content_digest = ? "
                "WHERE closure_id = ? AND input_ordinal = ?",
                (
                    "f" * 64,
                    closure_id,
                    input_ordinal,
                ),
            ),
        )
    )

    stale = query_stale(store, architecture=_architecture())

    assert len(stale) == 1
    axes = {item.axis for item in stale[0].differences}
    assert {
        "compilation_manifest",
        "source_closure",
        "target",
        "toolchain",
        "verification_specification",
        "tolerance_policy",
        "oracle",
    } <= axes
    closure = next(
        item for item in stale[0].differences if item.axis == "source_closure"
    )
    assert any(detail.endswith(":changed") for detail in closure.details)


def test_todo_is_a_read_only_deterministic_unranked_set_difference(
    tmp_path: Path, backend_report: VerificationReport
) -> None:
    _require_dolt()
    store = DoltEvidenceStore(tmp_path / "evidence-store")
    store.initialize()
    before = tuple(_count(store, table) for table in ("verification_runs", "evidence"))

    missing = query_todo(store, architecture=_architecture())

    assert len(missing) == len(backend_report.records)
    keys = tuple(
        (item.kernel_id, item.variant, item.test_class, item.case_id)
        for item in missing
    )
    assert keys == tuple(sorted(keys))
    assert (
        tuple(_count(store, table) for table in ("verification_runs", "evidence"))
        == before
    )

    record_report(backend_report, store, producer_id="producer")
    counts = tuple(_count(store, table) for table in ("verification_runs", "evidence"))
    assert query_todo(store, architecture=_architecture()) == ()
    assert query_stale(store, architecture=_architecture())[0].differences == ()
    assert (
        tuple(_count(store, table) for table in ("verification_runs", "evidence"))
        == counts
    )


def test_query_cli_emits_stable_json(
    tmp_path: Path,
    backend_report: VerificationReport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_dolt()
    store_path = tmp_path / "evidence-store"
    record_report(
        backend_report, DoltEvidenceStore(store_path), producer_id="cli-producer"
    )

    status = status_main(
        [
            "status",
            "--arch",
            _architecture(),
            "--producer",
            "cli-producer",
            "--store",
            str(store_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == len(backend_report.records)
    assert all(
        item["producer_id"] == "cli-producer" for item in payload["observations"]
    )
