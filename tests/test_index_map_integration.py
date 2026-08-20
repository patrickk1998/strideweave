import pytest

import strideweave as sw
from strideweave.core.index_map import IndexMap
from strideweave.core.layout import Shape
from strideweave.core.permutation import Permutation
from strideweave.core.product import Product


def test_product_packs_real_permutation_children():
    first = Permutation([4, 3], 10)
    second = Permutation([2, 9], 15)

    product = Product(first, second)

    assert product.shape == Shape(2, 2)
    assert product.target_shape == Shape(10, 15)
    assert (first(0), second(0)) == (4, 2)
    assert product((0, 0)) == 24
    assert (first(1), second(1)) == (3, 9)
    assert product((1, 1)) == 93


@pytest.mark.parametrize(
    "inner",
    [
        sw.Permutation([3, 1], 4),
        sw.Swizzle(sw.Shape(4), sw.SwizzleStage(1, 0, 1)),
        sw.Product(sw.Permutation([1, 0], 2), sw.Permutation([0, 1], 2)),
    ],
)
def test_layout_composes_sibling_maps_through_every_supported_call_form(
    inner: sw.IndexMap,
):
    outer = sw.Layout(sw.Shape(5), sw.Stride(2))
    results = (
        outer.compose(inner),
        outer.compose(inner=inner),
        sw.Layout.compose(outer, inner),
        sw.Layout.compose(A=outer, B=inner),
    )

    for result in results:
        assert isinstance(result, IndexMap)
        assert not isinstance(result, sw.Layout)
        assert result.shape == inner.shape
        assert result.codomain_size == outer.codomain_size == 9
        assert result.is_injective is True
        for coordinate in range(inner.size):
            assert result(coordinate) == outer(inner(coordinate))


def test_layout_composes_a_private_generic_chain_without_changing_its_metadata():
    innermost = sw.Swizzle(sw.Shape(4), sw.SwizzleStage(1, 0, 1))
    middle = sw.Permutation([3, 1, 4, 0], 5)
    generic_inner = middle.compose(innermost)
    outer = sw.Layout(sw.Shape(6), sw.Stride(2))

    result = outer.compose(generic_inner)

    assert not isinstance(generic_inner, sw.Permutation)
    assert not isinstance(result, sw.Layout)
    assert result.shape == innermost.shape
    assert result.codomain_size == outer.codomain_size
    assert result.is_injective is True
    for coordinate in range(result.size):
        assert result(coordinate) == outer(generic_inner(coordinate))


def test_layout_composition_preserves_hierarchical_product_coordinates():
    first = sw.Permutation([1, 0], 2)
    second = sw.Permutation([0, 1], 2)
    third = sw.Permutation([1, 0], 2)
    inner = sw.Product(first, sw.Product(second, third))
    outer = sw.Layout(sw.Shape(8), sw.Stride(2))

    result = outer.compose(inner)

    assert result.shape == sw.Shape(2, [2, 2])
    assert result.codomain_size == outer.codomain_size == 15
    for coordinate in ((1, (0, 1)), (1, 2), [1, [0, 1]], 5):
        assert result(coordinate) == outer(inner(coordinate))


def test_layout_sibling_composition_uses_declared_bounds_and_metadata():
    inner = sw.Swizzle(sw.Shape(4))
    smaller_bound_outer = sw.Layout(sw.Shape(5), sw.Stride(2))
    colliding_outer = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 1]))
    unknown_outer = sw.Layout(sw.Shape([3, 2]), sw.Stride([2, 3]))

    assert smaller_bound_outer.compose(inner).codomain_size == 9
    assert colliding_outer.compose(inner).is_injective is None
    assert unknown_outer.compose(inner).is_injective is None

    oversized = sw.Permutation([5, 0], 6)
    with pytest.raises(ValueError, match="inner codomain exceeds"):
        smaller_bound_outer.compose(oversized)


def test_power_of_two_and_non_power_of_two_identities_preserve_an_operand():
    permutation = sw.Permutation([3, 1, 0, 2], 4)
    swizzle_identity = sw.Swizzle(sw.Shape(4))
    compact_layout_identity = sw.Layout(sw.Shape(3), sw.Stride(1))
    ternary_permutation = sw.Permutation([2, 0], 3)

    assert swizzle_identity.compose(permutation) is permutation
    assert permutation.compose(swizzle_identity) is permutation
    assert compact_layout_identity.compose(ternary_permutation) is ternary_permutation


def test_identity_composition_keeps_a_generic_result_when_metadata_differs():
    outer_identity = sw.Layout(sw.Shape(5), sw.Stride(1))
    inner = sw.Permutation([1, 0], 2)

    result = outer_identity.compose(inner)

    assert isinstance(result, sw.IndexMap)
    assert not isinstance(result, sw.Layout | sw.Permutation)
    assert result.shape == inner.shape
    assert result.codomain_size == outer_identity.codomain_size == 5
    assert [result(index) for index in range(result.size)] == [1, 0]
