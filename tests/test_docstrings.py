import inspect
from collections.abc import Callable
from typing import cast

import neotorch
import neotorch.einops
import neotorch.friendly
import neotorch.nn

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
    assert_has_docstring(neotorch, "neotorch")
    assert_has_docstring(neotorch.einops, "neotorch.einops")
    assert_has_docstring(neotorch.nn, "neotorch.nn")
    assert_has_docstring(neotorch.friendly, "neotorch.friendly")


def test_friendly_public_exports_have_docstrings():
    friendly_public_exports = cast(list[str], getattr(neotorch.friendly, "__all__"))
    for name in friendly_public_exports:
        value = getattr(neotorch.friendly, name)
        qualified_name = f"neotorch.friendly.{name}"
        assert inspect.isfunction(value), f"{qualified_name} must be a function export"
        assert_function_doc_contract(value, qualified_name)


def test_nn_public_exports_have_docstrings():
    nn_public_exports = cast(list[str], getattr(neotorch.nn, "__all__"))
    for name in nn_public_exports:
        value = getattr(neotorch.nn, name)
        qualified_name = f"neotorch.nn.{name}"
        assert inspect.isclass(value), f"{qualified_name} must be a class export"
        assert_class_doc_contract(value, qualified_name)
        for method_name, method in inspect.getmembers(value, inspect.isfunction):
            if method_name.startswith("_") or method.__qualname__.split(".")[0] != name:
                continue
            assert_function_doc_contract(method, f"{qualified_name}.{method_name}")


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
