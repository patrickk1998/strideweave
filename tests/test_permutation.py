from typing import Any, cast

import pytest

from strideweave.core.index_map import IndexMap
from strideweave.core.layout import Shape
from strideweave.core.permutation import Permutation


def test_permutation_preserves_sparse_lookup_metadata_and_key_forms():
    permutation = Permutation([4, 3], 10)

    assert permutation.shape == Shape(2)
    assert permutation.size == 2
    assert permutation.codomain_size == 10
    assert permutation.values == (4, 3)
    assert permutation.is_injective is True
    for key in (0, (0,), [0]):
        assert permutation.index(key) == 4
        assert permutation(key) == 4
    for key in (1, (1,), [1]):
        assert permutation.index(key) == 3
        assert permutation(key) == 3


def test_permutation_preserves_unused_codomain_space_above_its_image():
    permutation = Permutation([4, 3], 10)

    assert max(permutation.values) == 4
    assert permutation.codomain_size == 10


def test_permutation_composition_closes_by_lookup_with_the_outer_bound():
    inner = Permutation([1, 0], 3)
    outer = Permutation([4, 2, 1], 6)

    result = outer.compose(inner)

    assert isinstance(result, IndexMap)
    assert isinstance(result, Permutation)
    assert result.shape == Shape(2)
    assert result.codomain_size == 6
    assert result.values == (2, 4)
    assert result.is_injective is True
    assert [result(index) for index in range(result.size)] == [2, 4]


def test_permutation_composition_uses_declared_bounds_not_the_lookup_image():
    inner = Permutation([4, 1], 5)
    outer = Permutation([6, 4, 2, 0, 5], 9)

    result = outer.compose(inner)

    assert isinstance(result, Permutation)
    assert result.values == (5, 4)
    assert result.codomain_size == 9


@pytest.mark.parametrize(
    ("values", "codomain_size"),
    [
        (object(), 10),
        ([4, 3.0], 10),
        ([False], 2),
        ([True], 2),
        ([4, 3], 10.0),
        ([0], False),
        ([0], True),
    ],
)
def test_permutation_rejects_wrong_argument_and_entry_kinds(
    values: object,
    codomain_size: object,
):
    permutation_type = cast(Any, Permutation)

    with pytest.raises(TypeError):
        permutation_type(values, codomain_size)


def test_permutation_rejects_wrong_constructor_arities():
    permutation_type = cast(Any, Permutation)

    with pytest.raises(TypeError):
        permutation_type()
    with pytest.raises(TypeError):
        permutation_type([1])
    with pytest.raises(TypeError):
        permutation_type([1], 2, True)


@pytest.mark.parametrize(
    ("values", "codomain_size", "message"),
    [
        ([], 10, "at least one entry"),
        ([4, 4], 10, "pairwise distinct"),
        ([-1, 3], 10, "non-negative"),
        ([4, 10], 10, "smaller than codomain_size"),
        ([4, 11], 10, "smaller than codomain_size"),
        ([0], 0, "codomain_size must be positive"),
        ([0], -1, "codomain_size must be positive"),
    ],
)
def test_permutation_rejects_invalid_values_and_codomain_bounds(
    values: list[int],
    codomain_size: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        Permutation(values, codomain_size)


def test_permutation_copies_values_and_rejects_ordinary_semantic_mutation():
    values = [4, 3]
    permutation = Permutation(values, 10)

    values[0] = 9
    values.append(8)

    assert isinstance(permutation.values, tuple)
    assert permutation.values == (4, 3)
    assert permutation(0) == 4
    assert permutation(1) == 3
    with pytest.raises(AttributeError, match="immutable"):
        setattr(permutation, "values", (9, 8))
    with pytest.raises(AttributeError, match="immutable"):
        delattr(permutation, "codomain_size")
    assert permutation.values == (4, 3)
    assert [permutation(index) for index in range(permutation.size)] == [4, 3]
