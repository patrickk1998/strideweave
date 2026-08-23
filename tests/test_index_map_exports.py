import doctest
import inspect
from typing import Any, cast

import pytest

import strideweave as sw
import strideweave.layout as layout_api

INDEX_MAP_EXPORTS = {
    "IndexMap",
    "Permutation",
    "Product",
    "Swizzle",
    "SwizzleStage",
}


def test_index_map_exports_match_both_public_import_surfaces():
    assert INDEX_MAP_EXPORTS <= set(sw.__all__)
    assert INDEX_MAP_EXPORTS <= set(layout_api.__all__)
    for name in INDEX_MAP_EXPORTS:
        assert getattr(sw, name) is getattr(layout_api, name)

    assert sw.Layout is layout_api.Layout
    assert sw.Shape is layout_api.Shape
    assert not hasattr(sw, "_ComposedIndexMap")
    assert not hasattr(layout_api, "_ComposedIndexMap")


@pytest.mark.parametrize(
    "public_class",
    [sw.IndexMap, sw.Permutation, sw.Product, sw.Swizzle, sw.SwizzleStage],
)
def test_index_map_class_docstrings_document_construction_and_execute_examples(
    public_class: type,
):
    docstring = inspect.getdoc(public_class)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Examples:" in docstring
    for parameter_name in inspect.signature(public_class).parameters:
        assert parameter_name in docstring

    example = doctest.DocTestParser().get_doctest(
        docstring,
        {},
        public_class.__qualname__,
        None,
        0,
    )
    assert example.examples
    doctest.DebugRunner().run(example)


@pytest.mark.parametrize(
    "index_map",
    [
        sw.Permutation([1, 0], 2),
        sw.Product(sw.Permutation([1, 0], 2), sw.Permutation([0, 1], 2)),
        sw.Swizzle(sw.Shape(4), sw.SwizzleStage(1, 0, 1)),
    ],
)
def test_tensor_placement_remains_layout_only(index_map: sw.IndexMap):
    tensor_type = cast(Any, sw.Tensor)

    with pytest.raises(TypeError, match="placement must be a Layout"):
        tensor_type(sw.Generic([0.0] * index_map.codomain_size), 0, index_map)
