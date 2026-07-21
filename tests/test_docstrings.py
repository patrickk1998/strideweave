import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import cast

import strideweave as sw
import strideweave.einops as einops
import strideweave.friendly as friendly
import strideweave.nn as nn
import strideweave.operation as operation

NATIVE_TOP_LEVEL_EXPORTS = {
    "CPU",
    "Carrier",
    "Operation",
    "Tensor",
}
EINOPS_SYNTAX_FUNCTIONS = {
    "einsum",
    "lex",
    "parse_einsum",
    "parse_layout_ref",
    "parse_rearrange",
    "parse_reduce",
    "rearrange",
    "reduce",
}
EINOPS_SEMANTIC_FUNCTIONS = {
    "einsum",
    "parse_einsum",
    "parse_layout_ref",
    "parse_rearrange",
    "parse_reduce",
    "rearrange",
    "reduce",
}
EINOPS_TENSOR_FUNCTIONS = {
    "einsum",
    "rearrange",
    "reduce",
}


def assert_has_docstring(value: object, name: str) -> None:
    assert inspect.getdoc(value), f"{name} is missing a docstring"


def assert_signature_params_documented(
    value: Callable[..., object], name: str, docstring: str
) -> None:
    for parameter_name in inspect.signature(value).parameters:
        if parameter_name == "self":
            continue
        assert parameter_name in docstring, (
            f"{name} docstring must document {parameter_name!r}"
        )


def assert_function_doc_contract(value: Callable[..., object], name: str) -> None:
    docstring = inspect.getdoc(value)
    assert docstring, f"{name} is missing a docstring"
    assert "Args:" in docstring, f"{name} docstring must document inputs"
    assert "Returns:" in docstring, f"{name} docstring must document its output"
    assert "Examples:" in docstring, f"{name} docstring must provide an example"
    assert_signature_params_documented(value, name, docstring)


def assert_class_doc_contract(value: type, name: str) -> None:
    # Classes document their constructor: inputs and an example, plus every
    # constructor parameter (including ones inherited from a base ``__init__``,
    # such as ``name``). A class docstring describes construction, not a return
    # value, so ``Returns:`` is not required.
    docstring = inspect.getdoc(value)
    assert docstring, f"{name} is missing a docstring"
    assert "Args:" in docstring, f"{name} docstring must document constructor inputs"
    assert "Examples:" in docstring, f"{name} docstring must provide an example"
    assert_signature_params_documented(value, name, docstring)


def test_public_modules_have_docstrings():
    assert_has_docstring(sw, "strideweave")
    assert_has_docstring(einops, "strideweave.einops")
    assert_has_docstring(nn, "strideweave.nn")
    assert_has_docstring(friendly, "strideweave.friendly")


def test_friendly_public_exports_have_docstrings():
    friendly_public_exports = cast(list[str], getattr(friendly, "__all__"))
    for name in friendly_public_exports:
        value = getattr(friendly, name)
        qualified_name = f"sw.friendly.{name}"
        assert inspect.isfunction(value), f"{qualified_name} must be a function export"
        assert_function_doc_contract(value, qualified_name)


def test_nn_public_exports_have_docstrings():
    nn_public_exports = cast(list[str], getattr(nn, "__all__"))
    for name in nn_public_exports:
        value = getattr(nn, name)
        qualified_name = f"sw.nn.{name}"
        assert inspect.isclass(value), f"{qualified_name} must be a class export"
        assert_class_doc_contract(value, qualified_name)
        for method_name, method in inspect.getmembers(value, inspect.isfunction):
            if method_name.startswith("_") or method.__qualname__.split(".")[0] != name:
                continue
            assert_function_doc_contract(method, f"{qualified_name}.{method_name}")


def test_native_carrier_ownership_methods_document_their_semantics():
    mutable_docstring = cast(str, inspect.getdoc(sw.Carrier.is_mutable))
    owned_docstring = cast(str, inspect.getdoc(sw.Carrier.is_owned))

    assert "public carrier interfaces" in mutable_docstring
    assert "intrinsic mutability" in mutable_docstring
    assert "exclusively owns" in owned_docstring
    assert "public mutation" in owned_docstring
    for docstring in (mutable_docstring, owned_docstring):
        assert "Returns:" in docstring
        assert "Examples:" in docstring


def test_einops_public_exports_have_docstrings():
    einops_public_exports = cast(list[str], getattr(einops, "__all__"))
    for name in einops_public_exports:
        value = getattr(einops, name)
        qualified_name = f"sw.einops.{name}"
        assert_has_docstring(value, qualified_name)
        if inspect.isfunction(value):
            assert_function_doc_contract(value, qualified_name)

            docstring = cast(str, inspect.getdoc(value))
            if name in EINOPS_SYNTAX_FUNCTIONS:
                assert "Syntax:" in docstring, (
                    f"{qualified_name} docstring must document description syntax"
                )
            if name in EINOPS_SEMANTIC_FUNCTIONS:
                assert "Semantics:" in docstring, (
                    f"{qualified_name} docstring must document command semantics"
                )
            if name in EINOPS_TENSOR_FUNCTIONS:
                assert "Mode assumptions:" in docstring, (
                    f"{qualified_name} docstring must document tensor mode assumptions"
                )


def test_documentable_top_level_exports_have_docstrings():
    assert NATIVE_TOP_LEVEL_EXPORTS < set(sw.__all__)

    for name in sw.__all__:
        if name in NATIVE_TOP_LEVEL_EXPORTS:
            continue
        value = getattr(sw, name)
        qualified_name = f"sw.{name}"
        assert_has_docstring(value, qualified_name)
        if inspect.isfunction(value):
            assert_function_doc_contract(value, qualified_name)


def test_operation_runtime_exports_match_stub_declarations():
    stub_path = Path(__file__).parents[1] / "src/strideweave/operation.pyi"
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    all_assignment = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        )
    )
    stub_exports = ast.literal_eval(all_assignment.value)
    declared_names = {
        statement.name
        for statement in tree.body
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef))
    }
    declared_names.update(
        alias.asname or alias.name
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    )

    assert operation.__all__ == stub_exports
    assert set(stub_exports) <= declared_names
