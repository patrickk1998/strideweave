from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import synthetic_evidence

import strideweave.verification.store.recording as recording_module
from strideweave.verification import VerificationReport
from strideweave.verification.status_cli import main as status_main
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    query_stale,
    query_status,
    query_todo,
)

StorePathFactory = Callable[..., Path]
ARCHITECTURE = synthetic_evidence.ARCHITECTURE


def _text(row: Mapping[str, object], field: str) -> str:
    value = row[field]
    assert isinstance(value, str)
    return value


def _count(store: DoltEvidenceStore, table: str) -> int:
    value = store.query(SQLStatement(f"SELECT COUNT(*) AS count FROM {table}"))[0][
        "count"
    ]
    assert isinstance(value, int)
    return value


def _record(
    report: VerificationReport, store: DoltEvidenceStore, producer: str
) -> recording_module.RecordResult:
    return recording_module._record_validated_report(
        report, store, producer_id=producer
    )


def test_status_filters_stably_and_preserves_contradictory_observations(
    evidence_store_path: StorePathFactory, synthetic_report: VerificationReport
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    original = synthetic_report.records[0]
    changed_report = synthetic_evidence.contradicting_report(synthetic_report)
    _record(synthetic_report, store, "producer-a")
    _record(changed_report, store, "producer-b")

    observations = query_status(store, architecture=ARCHITECTURE)
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
        architecture=ARCHITECTURE,
        kernel_id=original.case.kernel_id,
        variant=original.case.variant,
        test_class=original.test_class.value,
        producer_id="producer-b",
    )
    assert filtered
    assert all(item.producer_id == "producer-b" for item in filtered)
    assert all(item.kernel_id == original.case.kernel_id for item in filtered)


def test_stale_reports_each_changed_identity_axis_and_closure_member(
    evidence_store_path: StorePathFactory, synthetic_current_facts: VerificationReport
) -> None:
    report = synthetic_current_facts
    store = DoltEvidenceStore(evidence_store_path())
    result = _record(report, store, "producer")
    manifest = recording_module._report_manifest(report)
    assert report.header is not None
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
                    report.header.verification_spec["verification_spec_id"],
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

    stale = query_stale(store, architecture=ARCHITECTURE)

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


def test_stored_closure_members_are_project_owned_while_identity_stays_complete(
    evidence_store_path: StorePathFactory, synthetic_report: VerificationReport
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record(synthetic_report, store, "producer")

    stored_kinds = {
        row["input_kind"]
        for row in store.query(
            SQLStatement("SELECT DISTINCT input_kind FROM source_closure_inputs")
        )
    }
    closure = store.query(
        SQLStatement(
            "SELECT closure_id, descriptor_json FROM source_closures "
            "WHERE root_kind = ? ORDER BY closure_id LIMIT 1",
            ("repository-source",),
        )
    )[0]
    described = json.loads(_text(closure, "descriptor_json"))["inputs"]
    members = store.query(
        SQLStatement(
            "SELECT input_ordinal, input_uri FROM source_closure_inputs "
            "WHERE closure_id = ? ORDER BY input_ordinal",
            (_text(closure, "closure_id"),),
        )
    )

    # The closure descriptor, and therefore the content-addressed closure_id,
    # still binds every transitive input; only project- and build-owned members
    # become rows, at the ordinals they hold in the complete input sequence.
    assert "external_header" in {item["input_kind"] for item in described}
    assert "external_header" not in stored_kinds
    assert stored_kinds <= {"build_input", "generated_header", "header", "source"}
    assert tuple((row["input_ordinal"], row["input_uri"]) for row in members) == tuple(
        (ordinal, item["uri"])
        for ordinal, item in enumerate(described)
        if item["input_kind"] != "external_header"
    )


def test_an_external_only_closure_change_is_stale_without_a_member_difference(
    evidence_store_path: StorePathFactory, synthetic_current_facts: VerificationReport
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    result = _record(synthetic_current_facts, store, "producer")
    build = store.query(
        SQLStatement(
            "SELECT k.kernel_build_id, k.closure_id FROM run_kernel_builds rb "
            "JOIN kernel_builds k ON k.kernel_build_id = rb.kernel_build_id "
            "WHERE rb.run_id = ? ORDER BY k.kernel_build_id LIMIT 1",
            (result.run_id,),
        )
    )[0]
    rebound = "e" * 64
    # A changed external header moves nothing this store keeps a member row for:
    # the project-owned members and their digests are identical, and only the
    # content-addressed closure identity differs.
    store.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO source_closures SELECT ?, hash_algorithm, root_kind, "
                "root_uri, descriptor_json FROM source_closures WHERE closure_id = ?",
                (rebound, _text(build, "closure_id")),
            ),
            SQLStatement(
                "INSERT INTO source_closure_inputs SELECT ?, input_ordinal, "
                "input_kind, input_uri, content_digest, descriptor_json "
                "FROM source_closure_inputs WHERE closure_id = ?",
                (rebound, _text(build, "closure_id")),
            ),
            SQLStatement(
                "UPDATE kernel_builds SET closure_id = ? WHERE kernel_build_id = ?",
                (rebound, _text(build, "kernel_build_id")),
            ),
        )
    )

    stale = query_stale(store, architecture=ARCHITECTURE)

    closure = next(
        item for item in stale[0].differences if item.axis == "source_closure"
    )
    assert rebound in closure.stored
    assert rebound not in closure.current
    assert closure.details == ()


def test_closure_members_an_earlier_version_stored_report_no_difference(
    evidence_store_path: StorePathFactory, synthetic_current_facts: VerificationReport
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record(synthetic_current_facts, store, "producer")
    closure = store.query(
        SQLStatement(
            "SELECT closure_id FROM source_closures WHERE root_kind = ? "
            "ORDER BY closure_id LIMIT 1",
            ("repository-source",),
        )
    )[0]
    external_digest = "b" * 64
    store.execute_transaction(
        (
            SQLStatement(
                "INSERT INTO source_closure_inputs (closure_id, input_ordinal, "
                "input_kind, input_uri, content_digest, descriptor_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _text(closure, "closure_id"),
                    100000,
                    "external_header",
                    f"cpp-external://sha256/{external_digest}/legacy.hpp",
                    external_digest,
                    "{}",
                ),
            ),
        )
    )

    stale = query_stale(store, architecture=ARCHITECTURE)

    # A store an earlier version wrote still holds external members. The
    # current manifest no longer offers them individually, so comparing them
    # would report an unchanged report as stale.
    assert stale[0].differences == ()


def test_todo_is_a_read_only_deterministic_unranked_set_difference(
    evidence_store_path: StorePathFactory, synthetic_current_facts: VerificationReport
) -> None:
    report = synthetic_current_facts
    store = DoltEvidenceStore(evidence_store_path())
    store.initialize()
    before = tuple(_count(store, table) for table in ("verification_runs", "evidence"))

    missing = query_todo(store, architecture=ARCHITECTURE)

    assert len(missing) == len(report.records)
    keys = tuple(
        (item.kernel_id, item.variant, item.test_class, item.case_id)
        for item in missing
    )
    assert keys == tuple(sorted(keys))
    assert (
        tuple(_count(store, table) for table in ("verification_runs", "evidence"))
        == before
    )

    _record(report, store, "producer")
    counts = tuple(_count(store, table) for table in ("verification_runs", "evidence"))
    assert query_todo(store, architecture=ARCHITECTURE) == ()
    assert query_stale(store, architecture=ARCHITECTURE)[0].differences == ()
    assert (
        tuple(_count(store, table) for table in ("verification_runs", "evidence"))
        == counts
    )


def test_query_cli_emits_stable_json(
    evidence_store_path: StorePathFactory,
    synthetic_report: VerificationReport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_path = evidence_store_path()
    _record(synthetic_report, DoltEvidenceStore(store_path), "cli-producer")

    status = status_main(
        [
            "status",
            "--arch",
            ARCHITECTURE,
            "--producer",
            "cli-producer",
            "--store",
            str(store_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == len(synthetic_report.records)
    assert all(
        item["producer_id"] == "cli-producer" for item in payload["observations"]
    )
