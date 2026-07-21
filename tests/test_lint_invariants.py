from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from tools.lint_invariants import check_source, discover_files, run


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("value = tensor[[i, j]]\n", "SW001"),
        ("value = tensor.dtype() == DType.Float32\n", "SW002"),
        ("class Backend(Carrier):\n    def is_mutable(self): return True\n", "SW003"),
        ("count = tensor.layout.size\n", "SW004"),
    ],
)
def test_invariant_checker_reports_each_rule(source: str, code: str):
    diagnostics = check_source(source)

    assert [diagnostic.code for diagnostic in diagnostics] == [code]


@pytest.mark.parametrize(
    "source",
    [
        "value = tensor[i, j]\n",
        "value = tensor.dtype() is DType.Float32\n",
        "class Backend(Carrier):\n    def _is_mutable(self): return True\n",
        "count = tensor.size()\n",
        "count = layout.size\n",
    ],
)
def test_invariant_checker_accepts_canonical_forms(source: str):
    assert check_source(source) == []


def test_invariant_checker_supports_narrow_line_suppressions():
    diagnostics = check_source(
        "first = tensor[[i, j]]  # strideweave-lint: ignore=SW001\n"
        "second = tensor[[i, j]]\n"
    )

    assert [(diagnostic.code, diagnostic.line) for diagnostic in diagnostics] == [
        ("SW001", 2)
    ]


def test_invariant_checker_orders_multiple_diagnostics():
    diagnostics = check_source(
        "dtype_matches = tensor.dtype() == DType.Float32\nvalue = tensor[[i, j]]\n"
    )

    assert [(diagnostic.code, diagnostic.line) for diagnostic in diagnostics] == [
        ("SW002", 1),
        ("SW001", 2),
    ]


def test_invariant_checker_discovers_python_files_and_excludes_caches(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    included = source / "included.py"
    included.write_text("value = 1\n")
    excluded_directory = source / "__pycache__"
    excluded_directory.mkdir()
    (excluded_directory / "excluded.py").write_text("value = 2\n")
    (source / "schema_pb2.py").write_text("value = 3\n")

    assert discover_files([source]) == [included.resolve()]


def test_invariant_checker_exit_codes_and_output(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("value = tensor[[i, j]]\n")
    stdout = StringIO()
    stderr = StringIO()

    assert run([source], stdout=stdout, stderr=stderr) == 1
    assert "SW001" in stdout.getvalue()
    assert stderr.getvalue() == ""

    missing = tmp_path / "missing.py"
    assert run([missing], stdout=stdout, stderr=stderr) == 2
    assert "path does not exist" in stderr.getvalue()


def test_invariant_checker_reports_syntax_errors(tmp_path: Path):
    source = tmp_path / "invalid.py"
    source.write_text("if:\n")
    stdout = StringIO()
    stderr = StringIO()

    assert run([source], stdout=stdout, stderr=stderr) == 2
    assert "syntax error" in stderr.getvalue()
