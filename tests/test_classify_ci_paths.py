import io
import json

import pytest

from tools.classify_ci_paths import (
    Classification,
    classify_compare_payload,
    classify_paths,
    main,
)


def payload(*paths: str) -> dict[str, object]:
    return {"files": [{"filename": path} for path in paths]}


@pytest.mark.parametrize(
    "path",
    [
        "src/strideweave/carriers/block_device/carrier.py",
        "src/strideweave/carriers/move/ops.py",
        "src/strideweave/carriers/evictable/carrier.py",
        "src/strideweave/__init__.py",
        "tests/test_block_device_move.py",
        "CMakeLists.txt",
        "pyproject.toml",
        ".github/workflows/ci.yml",
    ],
)
def test_block_device_paths_run_both_job_families(path: str) -> None:
    assert classify_compare_payload(payload(path)) == Classification(
        code=True, block_device=True
    )


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("docs/guide.md", False),
        ("openspec/changes/example/proposal.md", False),
        ("src/strideweave/verification/api.py", True),
        ("tests/test_verification_report.py", True),
        ("src/strideweave/nn/module.py", True),
        (".github/workflows/specs.yml", True),
    ],
)
def test_known_irrelevant_paths_skip_the_real_device_job(path: str, code: bool) -> None:
    assert classify_compare_payload(payload(path)) == Classification(
        code=code, block_device=False
    )


def test_rename_classifies_both_old_and_new_paths() -> None:
    comparison = {
        "files": [
            {
                "filename": "docs/retired-carrier.md",
                "previous_filename": "src/strideweave/carriers/block_device/carrier.py",
            }
        ]
    }

    assert classify_compare_payload(comparison) == Classification(
        code=True, block_device=True
    )


def test_unknown_code_path_runs_fail_open() -> None:
    assert classify_compare_payload(payload("future-runtime/config.xyz")) == (
        Classification(code=True, block_device=True)
    )


@pytest.mark.parametrize(
    "comparison",
    [
        {},
        {"files": []},
        {"files": [{"filename": f"docs/{index}.md"} for index in range(300)]},
        {"files": [{"status": "modified"}]},
        {"files": [{"filename": "new.py", "previous_filename": 1}]},
    ],
)
def test_uncertain_compare_payloads_refuse_a_skip_verdict(
    comparison: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="compare"):
        classify_compare_payload(comparison)


def test_empty_direct_path_list_refuses_a_skip_verdict() -> None:
    with pytest.raises(ValueError, match="no paths"):
        classify_paths([])


def test_cli_emits_both_verdicts() -> None:
    stdin = io.StringIO(json.dumps(payload("tests/test_block_device.py")))
    stdout = io.StringIO()

    assert main(stdin=stdin, stdout=stdout) == 0
    assert json.loads(stdout.getvalue()) == {"code": True, "block_device": True}


def test_cli_failure_returns_nonzero_for_workflow_fail_open_handling() -> None:
    assert main(stdin=io.StringIO("not json"), stdout=io.StringIO()) == 1
