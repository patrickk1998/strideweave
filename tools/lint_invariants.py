"""Check repository-specific Python source invariants."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

DEFAULT_PATHS = ("src", "tests", "examples")
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
}
EXCLUDED_FILE_SUFFIXES = (".generated.py", "_pb2.py", "_pb2_grpc.py")
SUPPRESSION_PATTERN = re.compile(r"strideweave-lint:\s*ignore=(?P<codes>[A-Z0-9, ]+)")


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One source invariant violation."""

    path: Path
    line: int
    column: int
    code: str
    message: str

    def render(self, root: Path) -> str:
        """Render the diagnostic relative to the invocation directory."""

        try:
            display_path = self.path.relative_to(root)
        except ValueError:
            display_path = self.path
        return f"{display_path}:{self.line}:{self.column}: {self.code} {self.message}"


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _is_dtype_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "dtype"
            and not node.args
            and not node.keywords
        )
    parts = _attribute_parts(node)
    return len(parts) >= 2 and parts[-2] == "DType"


def _is_carrier_base(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "Carrier") or (
        isinstance(node, ast.Attribute) and node.attr == "Carrier"
    )


class InvariantVisitor(ast.NodeVisitor):
    """Collect StrideWeave-specific diagnostics from one Python AST."""

    def __init__(self, path: Path, suppressions: dict[int, set[str]]) -> None:
        self.path = path
        self.suppressions = suppressions
        self.diagnostics: list[Diagnostic] = []

    def _report(self, node: ast.AST, code: str, message: str) -> None:
        line = getattr(node, "lineno", 1)
        if code in self.suppressions.get(line, set()):
            return
        self.diagnostics.append(
            Diagnostic(
                self.path,
                line,
                getattr(node, "col_offset", 0) + 1,
                code,
                message,
            )
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.List):
            self._report(
                node,
                "SW001",
                "use tuple-style coordinates (tensor[i, j]), not tensor[[i, j]]",
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = (node.left, *node.comparators)
        if any(isinstance(operator, (ast.Eq, ast.NotEq)) for operator in node.ops):
            if any(_is_dtype_expression(operand) for operand in operands):
                self._report(
                    node,
                    "SW002",
                    "compare dtype tags with 'is' or 'is not'",
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(_is_carrier_base(base) for base in node.bases):
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if statement.name == "is_mutable":
                        self._report(
                            statement,
                            "SW003",
                            "override _is_mutable; Carrier owns public is_mutable",
                        )
                    if statement.name == "dispatch_op":
                        self._report(
                            statement,
                            "RT001",
                            "override _dispatch_op; Carrier owns public dispatch_op",
                        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_execute_lowered"
        ):
            self._report(
                node,
                "RT011",
                "use the sealed execute_lowered_operation helper for delegation",
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "_forward":
            receiver = node.func.value
            is_super_call = (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "super"
            )
            if not is_super_call:
                self._report(
                    node,
                    "RT011",
                    "delegate operations with execute_lowered_operation, not _forward",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr == "size"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "layout"
        ):
            self._report(
                node,
                "SW004",
                "use tensor.size() when a tensor object is available",
            )
        self.generic_visit(node)


def _read_suppressions(source: str) -> dict[int, set[str]]:
    suppressions: dict[int, set[str]] = {}
    try:
        tokens = tokenize.generate_tokens(
            iter(source.splitlines(keepends=True)).__next__
        )
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            match = SUPPRESSION_PATTERN.search(token.string)
            if match is None:
                continue
            codes = {
                code.strip() for code in match.group("codes").split(",") if code.strip()
            }
            suppressions.setdefault(token.start[0], set()).update(codes)
    except (IndentationError, tokenize.TokenError):
        pass
    return suppressions


def check_source(source: str, path: Path = Path("<string>")) -> list[Diagnostic]:
    """Return invariant diagnostics for a Python source string."""

    tree = ast.parse(source, filename=str(path))
    visitor = InvariantVisitor(path, _read_suppressions(source))
    visitor.visit(tree)
    return sorted(visitor.diagnostics)


def discover_files(paths: list[Path]) -> list[Path]:
    """Resolve Python files beneath explicit files and directories."""

    discovered: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else path.rglob("*.py")
        for candidate in candidates:
            if candidate.suffix != ".py":
                continue
            if any(part in EXCLUDED_PARTS for part in candidate.parts):
                continue
            if candidate.name.endswith(EXCLUDED_FILE_SUFFIXES):
                continue
            discovered.add(candidate.resolve())
    return sorted(discovered)


def run(
    paths: list[Path], *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr
) -> int:
    """Run all invariant rules over paths and return a process-style exit code."""

    root = Path.cwd().resolve()
    try:
        files = discover_files(paths)
    except FileNotFoundError as exc:
        print(f"lint-invariants: path does not exist: {exc.args[0]}", file=stderr)
        return 2

    diagnostics: list[Diagnostic] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            diagnostics.extend(check_source(source, path))
        except (OSError, UnicodeError) as exc:
            print(f"lint-invariants: cannot read {path}: {exc}", file=stderr)
            return 2
        except SyntaxError as exc:
            line = exc.lineno or 1
            column = exc.offset or 1
            print(f"{path}:{line}:{column}: syntax error: {exc.msg}", file=stderr)
            return 2

    for diagnostic in sorted(diagnostics):
        print(diagnostic.render(root), file=stdout)
    return 1 if diagnostics else 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line paths and run repository invariant checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, default=[Path(path) for path in DEFAULT_PATHS]
    )
    arguments = parser.parse_args(argv)
    return run(arguments.paths)


if __name__ == "__main__":
    raise SystemExit(main())
