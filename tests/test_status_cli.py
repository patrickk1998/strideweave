from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import strideweave.verification.status_cli as status_cli
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
        status_cli.recording_module, "_validate_current_report", lambda _: None
    )
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
