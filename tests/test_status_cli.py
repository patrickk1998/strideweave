from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import strideweave.verification.status_cli as status_cli_module
from strideweave.verification.status_cli import main


@pytest.mark.parametrize(
    ("arguments", "fragments"),
    (
        (
            ["--help"],
            ("record", "status", "stale", "todo", "offline", "Exit status", "Examples"),
        ),
        (
            ["record", "--help"],
            (
                "provenance-complete v2",
                "offline by default",
                "Default",
                "--publish",
                "Exit status",
                "Examples",
            ),
        ),
        (
            ["status", "--help"],
            (
                "Contradictory outcomes",
                "Architecture syntax",
                "offline and read-only",
                "--refresh",
                "Exit status",
                "Examples",
            ),
        ),
        (
            ["stale", "--help"],
            (
                "closure-member",
                "Architecture syntax",
                "offline and read-only",
                "Exit status",
                "Examples",
            ),
        ),
        (
            ["todo", "--help"],
            (
                "deterministic, unranked set difference",
                "Architecture syntax",
                "offline and read-only",
                "Exit status",
                "Examples",
            ),
        ),
    ),
)
def test_every_help_path_is_complete_and_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    fragments: tuple[str, ...],
) -> None:
    status_home = tmp_path / "status-home"
    exchange = tmp_path / "exchange"
    monkeypatch.setenv("STRIDEWEAVE_STATUS_HOME", str(status_home))
    monkeypatch.setenv("STRIDEWEAVE_STATUS_PUBLISH_DESTINATION", str(exchange))
    monkeypatch.setenv("STRIDEWEAVE_STATUS_READ_SOURCE", str(exchange))

    def unexpected_side_effect(*args: object, **kwargs: object):
        raise AssertionError("help must not initialize storage or exchange data")

    monkeypatch.setattr(status_cli_module, "DoltEvidenceStore", unexpected_side_effect)
    monkeypatch.setattr(status_cli_module, "publish_evidence", unexpected_side_effect)
    monkeypatch.setattr(status_cli_module, "refresh_evidence", unexpected_side_effect)

    with pytest.raises(SystemExit) as exit_info:
        main(arguments)

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert all(fragment in output for fragment in fragments)
    assert not status_home.exists()
    assert not exchange.exists()


@pytest.mark.parametrize(
    "arguments", ((), ("record",), ("status",), ("stale",), ("todo",))
)
def test_packaged_command_help_exits_zero_without_creating_configured_paths(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    executable = Path(sys.executable).parent / "strideweave-kernel-status"
    status_home = tmp_path / "status-home"
    exchange = tmp_path / "exchange"
    environment = {
        **os.environ,
        "STRIDEWEAVE_STATUS_HOME": str(status_home),
        "STRIDEWEAVE_STATUS_PUBLISH_DESTINATION": str(exchange),
        "STRIDEWEAVE_STATUS_READ_SOURCE": str(exchange),
    }

    completed = subprocess.run(
        (str(executable), *arguments, "--help"),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage: strideweave-kernel-status" in completed.stdout
    assert not status_home.exists()
    assert not exchange.exists()


def test_command_errors_use_the_documented_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "record",
            "--report",
            str(tmp_path / "missing.jsonl"),
            "--producer",
            "test-producer",
        ]
    )

    assert status == 2
    assert "strideweave-kernel-status: error:" in capsys.readouterr().err
