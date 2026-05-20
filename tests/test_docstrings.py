import inspect
from collections.abc import Callable
from typing import cast

import neotorch
import neotorch.einops

NATIVE_TOP_LEVEL_EXPORTS = {
    "CPU",
    "Data",
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


def assert_function_doc_contract(value: Callable[..., object], name: str) -> None:
    docstring = inspect.getdoc(value)
    assert docstring, f"{name} is missing a docstring"
    assert "Args:" in docstring, f"{name} docstring must document inputs"
    assert "Returns:" in docstring, f"{name} docstring must document its output"
    assert "Examples:" in docstring, f"{name} docstring must provide an example"

    for parameter_name in inspect.signature(value).parameters:
        assert parameter_name in docstring, (
            f"{name} docstring must document {parameter_name!r}"
        )


def test_public_modules_have_docstrings():
    assert_has_docstring(neotorch, "neotorch")
    assert_has_docstring(neotorch.einops, "neotorch.einops")


def test_einops_public_exports_have_docstrings():
    einops_public_exports = cast(list[str], getattr(neotorch.einops, "__all__"))
    for name in einops_public_exports:
        value = getattr(neotorch.einops, name)
        qualified_name = f"neotorch.einops.{name}"
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
    assert NATIVE_TOP_LEVEL_EXPORTS < set(neotorch.__all__)

    for name in neotorch.__all__:
        if name in NATIVE_TOP_LEVEL_EXPORTS:
            continue
        value = getattr(neotorch, name)
        qualified_name = f"neotorch.{name}"
        assert_has_docstring(value, qualified_name)
        if inspect.isfunction(value):
            assert_function_doc_contract(value, qualified_name)
