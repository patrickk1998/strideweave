from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from evidence_doubles import RecordingEvidenceStore

import strideweave.verification.status_cli as status_cli
import strideweave.verification.store.recording as recording
from strideweave.carriers.metal._jit import MetalRecipeError, _decoded_value
from strideweave.verification.provenance import make_compilation_bundle
from strideweave.verification.store import VerificationStoreError


def test_help_explains_v3_only_recreate_boundary(capsys) -> None:
    assert status_cli.main(["record", "--help"]) == 0
    output = capsys.readouterr().out
    assert "Schema-v3 only" in output
    assert "manually recreate" in output


def test_todo_help_describes_missing_observations_not_passes(capsys) -> None:
    assert status_cli.main(["todo", "--help"]) == 0
    output = capsys.readouterr().out
    assert "without a matching observation" in output
    assert "without a pass" not in output


def test_status_requires_target(capsys) -> None:
    assert status_cli.main(["status"]) == 2
    assert "--target" in capsys.readouterr().err


@dataclass
class _Store:
    path: Path


@dataclass
class _Record:
    run_id: str = "r" * 64
    report_digest: str = "d" * 64
    evidence_count: int = 1
    observation_count: int = 1


@dataclass
class _Refresh:
    snapshot_count: int = 1
    observation_count: int = 1
    read_source: str = "exchange"


def test_cli_success_commands_dispatch_without_implicit_network_or_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    stores: list[_Store] = []
    monkeypatch.setattr(
        status_cli,
        "DoltEvidenceStore",
        lambda path: stores.append(_Store(Path(path or "default"))) or stores[-1],
    )
    monkeypatch.setattr(status_cli.VerificationReport, "load", lambda _: object())
    monkeypatch.setattr(
        status_cli, "record_report", lambda *_args, **_kwargs: _Record()
    )
    monkeypatch.setattr(
        status_cli,
        "publish_evidence",
        lambda *_args, **_kwargs: type(
            "P",
            (),
            {
                "producer_id": "p",
                "snapshot_digest": "s" * 64,
                "observation_count": 1,
                "destination": "exchange",
            },
        )(),
    )
    monkeypatch.setattr(status_cli, "query_status", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        status_cli, "refresh_evidence", lambda *_args, **_kwargs: _Refresh()
    )
    monkeypatch.setattr(status_cli, "_baseline", lambda _: object())
    monkeypatch.setattr(status_cli, "query_stale", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(status_cli, "query_todo", lambda *_args, **_kwargs: ())
    report = tmp_path / "report.jsonl"
    report.write_text("unused", encoding="utf-8")
    assert (
        status_cli.main(
            [
                "record",
                "--report",
                str(report),
                "--producer",
                "p",
                "--publish",
                "--json",
            ]
        )
        == 0
    )
    assert (
        status_cli.main(["status", "--target", "synthetic-jit", "--refresh", "--json"])
        == 0
    )
    assert status_cli.main(["stale", "--target", "synthetic-jit", "--json"]) == 0
    assert status_cli.main(["todo", "--target", "synthetic-jit", "--json"]) == 0
    assert len(stores) == 4
    assert capsys.readouterr().err == ""


def test_record_cli_validates_each_receipt_once_through_public_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys, synthetic_report
) -> None:
    assert synthetic_report.header is not None
    validation_calls = 0
    regenerated_receipts: list[str] = []
    validate = recording._validate_current_report

    def validate_once(report) -> None:
        nonlocal validation_calls
        validation_calls += 1
        validate(report)

    def regenerate(profile, receipts):
        del profile
        regenerated_receipts.extend(receipt.receipt_id for receipt in receipts)
        return make_compilation_bundle(receipts)

    monkeypatch.setattr(recording, "_validate_current_report", validate_once)
    monkeypatch.setattr(recording, "_current_profile_compilation_bundle", regenerate)
    monkeypatch.setattr(
        recording, "_validate_installed_verification_graph", lambda _report: None
    )
    monkeypatch.setattr(
        recording,
        "bind_report",
        lambda *_args, **_kwargs: (synthetic_report.records, synthetic_report.header),
    )
    store = RecordingEvidenceStore(tmp_path / "store")
    monkeypatch.setattr(
        status_cli.VerificationReport, "load", lambda _: synthetic_report
    )
    monkeypatch.setattr(status_cli, "DoltEvidenceStore", lambda _path: store)

    exit_code = status_cli.main(
        [
            "record",
            "--report",
            str(tmp_path / "report.jsonl"),
            "--producer",
            "producer",
            "--store",
            str(store.path),
        ]
    )

    assert exit_code == 0
    assert validation_calls == 1
    assert regenerated_receipts == [
        receipt.receipt_id
        for receipt in synthetic_report.header.compilation_bundle.receipts
    ]
    assert len(store.transactions) == 1
    assert "Recorded schema-v3 run" in capsys.readouterr().out


@pytest.mark.parametrize("existing", [False, True])
def test_record_cli_validation_failure_is_once_and_does_not_mutate_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
    existing: bool,
) -> None:
    store_path = tmp_path / "store"
    marker = store_path / "marker"
    if existing:
        store_path.mkdir()
        marker.write_text("unchanged", encoding="utf-8")
    validation_calls = 0

    def reject(_report) -> None:
        nonlocal validation_calls
        validation_calls += 1
        raise VerificationStoreError("stale report")

    monkeypatch.setattr(status_cli.VerificationReport, "load", lambda _: object())
    monkeypatch.setattr(recording, "_validate_current_report", reject)

    exit_code = status_cli.main(
        [
            "record",
            "--report",
            str(tmp_path / "report.jsonl"),
            "--producer",
            "producer",
            "--store",
            str(store_path),
        ]
    )

    assert exit_code == 2
    assert validation_calls == 1
    if existing:
        assert marker.read_text(encoding="utf-8") == "unchanged"
        assert tuple(store_path.iterdir()) == (marker,)
    else:
        assert not store_path.exists()
    assert "stale report" in capsys.readouterr().err


@pytest.mark.parametrize("existing", [False, True])
def test_record_cli_normalizes_provider_recipe_failure_without_store_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
    synthetic_report,
    existing: bool,
) -> None:
    store_path = tmp_path / "store"
    marker = store_path / "marker"
    if existing:
        store_path.mkdir()
        marker.write_text("unchanged", encoding="utf-8")

    def provider_recipe(profile, receipts):
        if profile.profile_id == "synthetic-jit":
            _decoded_value(("float", "0x1p+999999999"))
        return make_compilation_bundle(receipts)

    monkeypatch.setattr(
        status_cli.VerificationReport, "load", lambda _: synthetic_report
    )
    monkeypatch.setattr(
        recording, "_current_profile_compilation_bundle", provider_recipe
    )

    exit_code = status_cli.main(
        [
            "record",
            "--report",
            str(tmp_path / "report.jsonl"),
            "--producer",
            "producer",
            "--store",
            str(store_path),
        ]
    )

    assert exit_code == 2
    if existing:
        assert marker.read_text(encoding="utf-8") == "unchanged"
        assert tuple(store_path.iterdir()) == (marker,)
    else:
        assert not store_path.exists()
    error = capsys.readouterr().err
    assert "canonical float specialization value is invalid" in error
    assert "Traceback" not in error


def test_record_cli_normalizes_metal_plan_mismatch_without_store_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
    synthetic_report,
) -> None:
    store_path = tmp_path / "store"

    def provider_recipe(profile, receipts):
        del profile, receipts
        raise MetalRecipeError(
            "Metal recipe plan is not a current executable operation plan"
        )

    monkeypatch.setattr(
        status_cli.VerificationReport, "load", lambda _: synthetic_report
    )
    monkeypatch.setattr(
        recording, "_current_profile_compilation_bundle", provider_recipe
    )

    exit_code = status_cli.main(
        [
            "record",
            "--report",
            str(tmp_path / "report.jsonl"),
            "--producer",
            "producer",
            "--store",
            str(store_path),
        ]
    )

    assert exit_code == 2
    assert not store_path.exists()
    error = capsys.readouterr().err
    assert "current executable operation plan" in error
    assert "Traceback" not in error


@pytest.mark.parametrize(
    "command",
    [
        ["status", "--target", "missing", "--refresh"],
        ["stale", "--target", "missing"],
        ["todo", "--target", "missing"],
    ],
)
def test_invalid_target_fails_before_store_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command, capsys
) -> None:
    path = tmp_path / "store"

    def fail_if_store_constructed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid target constructed a Dolt store")

    monkeypatch.setattr(status_cli, "DoltEvidenceStore", fail_if_store_constructed)
    monkeypatch.setattr(
        status_cli,
        "_baseline",
        lambda _: (_ for _ in ()).throw(
            AssertionError("invalid target constructed a current baseline")
        ),
    )
    assert status_cli.main([*command, "--store", str(path)]) == 2
    assert not path.exists()
    error = capsys.readouterr().err
    assert "--target" in error
    assert "unknown verification profile" in error


def test_cli_exchange_failure_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        status_cli, "DoltEvidenceStore", lambda path: _Store(Path(path))
    )
    monkeypatch.setattr(status_cli, "query_status", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        status_cli,
        "refresh_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            VerificationStoreError("tampered snapshot")
        ),
    )
    assert (
        status_cli.main(
            [
                "status",
                "--target",
                "synthetic-jit",
                "--refresh",
                "--store",
                str(tmp_path / "store"),
            ]
        )
        == 2
    )
    assert "tampered snapshot" in capsys.readouterr().err
