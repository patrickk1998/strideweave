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


CARRIER_BASES = ("Carrier", "DependentCarrier")

# Public capability queries a carrier answers. Carrier owns them so that one
# resolution decides both introspection and enforcement: a backend states what
# it executes through a declaration or a dependent carrier's generator, never by
# answering the question itself.
CAPABILITY_QUERIES = (
    "operation_capabilities",
    "supports_operation_plan",
    "unsupported_plan_reason",
    "require_operation_plan",
)


def _is_carrier_base(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in CARRIER_BASES) or (
        isinstance(node, ast.Attribute) and node.attr in CARRIER_BASES
    )


def _is_attribute_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def _is_super_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    return (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Name)
        and receiver.func.id == "super"
    )


class InvariantVisitor(ast.NodeVisitor):
    """Collect StrideWeave-specific diagnostics from one Python AST."""

    def __init__(self, path: Path, suppressions: dict[int, set[str]]) -> None:
        self.path = path
        self.suppressions = suppressions
        self.diagnostics: list[Diagnostic] = []
        self._function_names: list[str] = []
        self._delegated_dispatch_names: list[set[str]] = []

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
                    if statement.name == "supports_storage_dtype":
                        self._report(
                            statement,
                            "RT012",
                            "override _supports_storage_dtype; Carrier owns "
                            "public supports_storage_dtype",
                        )
                    if statement.name in CAPABILITY_QUERIES:
                        self._report(
                            statement,
                            "RT013",
                            f"do not override {statement.name}; declare capabilities "
                            "for the class or generate them in "
                            "_generate_operation_capabilities",
                        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_names.append(node.name)
        self._delegated_dispatch_names.append(set())
        self.generic_visit(node)
        self._delegated_dispatch_names.pop()
        self._function_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_names.append(node.name)
        self._delegated_dispatch_names.append(set())
        self.generic_visit(node)
        self._delegated_dispatch_names.pop()
        self._function_names.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._function_names and self._function_names[-1] == "_dispatch_op":
            assigned_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            delegated_names = self._delegated_dispatch_names[-1]
            delegated_names.difference_update(assigned_names)
            if _is_attribute_call(node.value, "dispatch_op"):
                delegated_names.update(assigned_names)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            self._function_names
            and self._function_names[-1] == "_dispatch_op"
            and isinstance(node.target, ast.Name)
        ):
            delegated_names = self._delegated_dispatch_names[-1]
            delegated_names.discard(node.target.id)
            if _is_attribute_call(node.value, "dispatch_op"):
                delegated_names.add(node.target.id)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        in_dispatch_hook = (
            self._function_names and self._function_names[-1] == "_dispatch_op"
        )
        returns_delegated_call = _is_attribute_call(node.value, "dispatch_op")
        returns_delegated_name = (
            isinstance(node.value, ast.Name)
            and bool(self._delegated_dispatch_names)
            and node.value.id in self._delegated_dispatch_names[-1]
        )
        if in_dispatch_hook and (returns_delegated_call or returns_delegated_name):
            self._report(
                node,
                "RT011",
                "wrap a nested dispatch_op result in a composite-owned operation",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_dispatch_op":
            if not _is_super_call(node):
                self._report(
                    node,
                    "RT011",
                    "do not call another carrier's _dispatch_op; wrap its public "
                    "dispatch_op result in a composite-owned operation",
                )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_execute_lowered"
        ):
            self._report(
                node,
                "RT011",
                "use the sealed execute_lowered_operation helper for delegation",
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "_forward":
            if not _is_super_call(node):
                self._report(
                    node,
                    "RT011",
                    "delegate operations with execute_lowered_operation, not _forward",
                )
        elif (
            self._function_names
            and self._function_names[-1] == "_forward"
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward"
            and not _is_super_call(node)
        ):
            self._report(
                node,
                "RT011",
                "delegate operations with execute_lowered_operation, not forward",
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
