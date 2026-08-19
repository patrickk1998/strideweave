from itertools import product as cartesian_product

import pytest

from strideweave.core.index_map import IndexMap, _compose_generic
from strideweave.core.layout import Shape
from strideweave.core.product import Product


class _LookupMap(IndexMap):
    _values: tuple[int, ...]

    def __init__(
        self,
        values: tuple[int, ...],
        codomain_size: int,
        is_injective: bool | None = True,
    ) -> None:
        super().__init__(Shape(len(values)), codomain_size, is_injective)
        object.__setattr__(self, "_values", values)

    def _index_ordinal(self, index: int) -> int:
        return self._values[index]

    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)


class _RankZeroMap(IndexMap):
    _value: int

    def __init__(self, value: int, codomain_size: int) -> None:
        super().__init__(Shape(), codomain_size, True)
        object.__setattr__(self, "_value", value)

    def _index_ordinal(self, index: int) -> int:
        return self._value

    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)


class _ShapedMap(IndexMap):
    _values: tuple[int, ...]

    def __init__(
        self,
        shape: Shape,
        values: tuple[int, ...],
        codomain_size: int,
    ) -> None:
        super().__init__(shape, codomain_size, True)
        object.__setattr__(self, "_values", values)

    def _index_ordinal(self, index: int) -> int:
        return self._values[index]

    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)


def test_flat_product_exposes_owned_two_and_three_child_metadata():
    first = _LookupMap((1, 0), 3)
    second = _LookupMap((2, 0, 1), 4)
    third = _LookupMap((4, 1), 5)
    supplied_children = [first, second]

    two_child = Product(*supplied_children)
    three_child = Product(first, second, third)
    supplied_children[0] = third

    assert two_child.children == (first, second)
    assert two_child.shape == Shape(2, 3)
    assert two_child.target_shape == Shape(3, 4)
    assert two_child.codomain_size == 12
    assert three_child.children == (first, second, third)
    assert three_child.shape == Shape(2, 3, 2)
    assert three_child.target_shape == Shape(3, 4, 5)
    assert three_child.codomain_size == 60


def test_flat_product_uses_first_mode_fastest_packing_for_every_coordinate():
    first = _LookupMap((1, 0), 3)
    second = _LookupMap((2, 0, 1), 4)
    third = _LookupMap((4, 1), 5)
    two_child = Product(first, second)
    three_child = Product(first, second, third)

    for first_coordinate, second_coordinate in cartesian_product(range(2), range(3)):
        expected = first(first_coordinate) + 3 * second(second_coordinate)
        assert two_child((first_coordinate, second_coordinate)) == expected

    for coordinates in cartesian_product(range(2), range(3), range(2)):
        first_result = first(coordinates[0])
        second_result = second(coordinates[1])
        third_result = third(coordinates[2])
        expected = first_result + 3 * (second_result + 4 * third_result)
        assert three_child(coordinates) == expected


def test_product_matches_the_canonical_sparse_packing_example():
    first = _LookupMap((4, 3), 10)
    second = _LookupMap((2, 9), 15)
    product = Product(first, second)

    assert product.shape == Shape(2, 2)
    assert product.target_shape == Shape(10, 15)
    assert (first(0), second(0)) == (4, 2)
    assert product((0, 0)) == 24
    assert (first(1), second(1)) == (3, 9)
    assert product((1, 1)) == 93


def test_flat_product_accepts_shared_scalar_tuple_and_list_coordinate_forms():
    product = Product(
        _LookupMap((1, 0), 3),
        _LookupMap((2, 0, 1), 4),
    )

    for ordinal in range(product.size):
        coordinate = product.shape.decode(ordinal)

        assert product.index(ordinal) == product.index(coordinate)
        assert product.index(ordinal) == product.index(list(coordinate))


def test_product_preserves_two_rank_zero_children_as_explicit_modes():
    first = _RankZeroMap(2, 4)
    second = _RankZeroMap(3, 5)

    product = Product(first, second)

    assert product.children == (first, second)
    assert product.shape == Shape([], [])
    assert product.size == 1
    assert product.target_shape == Shape(4, 5)
    assert product.codomain_size == 20
    for coordinate in (0, ((), ()), [[], []], (0, 0), [0, 0], ((), [])):
        assert product(coordinate) == 14


def test_product_preserves_mixed_rank_zero_modes_in_child_order():
    rank_zero = _RankZeroMap(2, 4)
    ordinary = _LookupMap((1, 3), 5)

    rank_zero_first = Product(rank_zero, ordinary)
    rank_zero_last = Product(ordinary, rank_zero)

    assert rank_zero_first.shape == Shape([], 2)
    assert rank_zero_first.target_shape == Shape(4, 5)
    assert rank_zero_first(((), 1)) == 14
    assert rank_zero_first((0, 1)) == 14
    assert rank_zero_first([[], 1]) == 14
    assert rank_zero_first(1) == 14
    assert rank_zero_last.shape == Shape(2, [])
    assert rank_zero_last.target_shape == Shape(5, 4)
    assert rank_zero_last((1, ())) == 13
    assert rank_zero_last((1, 0)) == 13


def test_product_preserves_nested_rank_zero_product_grouping():
    first = _RankZeroMap(2, 4)
    second = _RankZeroMap(3, 5)
    ordinary = _LookupMap((1, 4), 5)
    inner = Product(first, second)

    nested = Product(inner, ordinary)
    variadic = Product(first, second, ordinary)

    assert nested.shape == Shape([[], []], 2)
    assert nested.target_shape == Shape([4, 5], 5)
    assert variadic.shape == Shape([], [], 2)
    assert variadic.target_shape == Shape(4, 5, 5)
    assert nested.shape != variadic.shape
    assert nested.target_shape != variadic.target_shape
    for nested_coordinate, variadic_coordinate in (
        ((((), ()), 1), ((), (), 1)),
        ([[[], []], 1], [[], [], 1]),
        ((0, 1), (0, 0, 1)),
        (1, 1),
    ):
        assert nested(nested_coordinate) == 94
        assert variadic(variadic_coordinate) == 94


def test_product_accepts_ordinary_children_with_nested_empty_modes():
    one_empty_level = _ShapedMap(Shape([[]]), (3,), 4)
    mixed_level = _ShapedMap(Shape([2, []]), (1, 3), 4)
    ordinary = _LookupMap((4, 0, 2), 5)

    shallow = Product(one_empty_level, ordinary)
    deep = Product(mixed_level, _RankZeroMap(2, 3), ordinary)

    assert shallow.shape == Shape([[]], 3)
    assert shallow((((),), 1)) == 3
    assert shallow(1) == 3
    assert deep.shape == Shape([2, []], [], 3)
    assert deep.target_shape == Shape(4, 3, 5)
    for ordinal in range(deep.size):
        coordinate = deep.shape.decode(ordinal)
        first_result = mixed_level(coordinate[0])
        second_result = deep.children[1](coordinate[1])
        third_result = ordinary(coordinate[2])
        expected = first_result + 4 * (second_result + 3 * third_result)

        assert deep.shape.encode(coordinate) == ordinal
        assert deep(coordinate) == expected
        assert deep(ordinal) == expected


@pytest.mark.parametrize(
    "coordinate",
    [
        (((), ()),),
        (((), ()), 1, 0),
        (((1,), ()), 1),
        (1, 1),
    ],
)
def test_nested_empty_product_rejects_invalid_coordinate_forms(coordinate: object):
    nested = Product(
        Product(_RankZeroMap(1, 2), _RankZeroMap(0, 2)),
        _LookupMap((1, 0), 2),
    )

    with pytest.raises(
        ValueError,
        match=r"Coordinate (does not match shape hierarchy|is not in domain of shape)",
    ):
        nested(coordinate)


def test_generic_composition_preserves_nested_empty_product_shape():
    nested = Product(
        Product(_RankZeroMap(1, 2), _RankZeroMap(0, 2)),
        _LookupMap((1, 0), 2),
    )
    outer = _LookupMap(tuple(reversed(range(nested.codomain_size))), 8)

    composed = outer.compose(nested)

    assert composed.shape == nested.shape
    assert composed.codomain_size == outer.codomain_size
    assert [composed(index) for index in range(composed.size)] == [
        outer(nested(index)) for index in range(nested.size)
    ]


def test_flat_product_participates_in_generic_composition():
    outer = Product(
        _LookupMap((1, 0), 2),
        _LookupMap((2, 0, 1), 3),
    )
    inner = _LookupMap((5, 0, 3), outer.size)

    composed = outer.compose(inner)

    assert isinstance(composed, IndexMap)
    assert composed.shape == inner.shape
    assert composed.codomain_size == outer.codomain_size
    assert [composed(index) for index in range(inner.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_aligned_flat_products_compose_componentwise():
    outer = Product(
        _LookupMap((2, 0, 1), 4),
        _LookupMap((1, 0), 3),
    )
    inner = Product(
        _LookupMap((2, 0), 3),
        _LookupMap((1, 0), 2),
    )

    composed = outer.compose(inner)

    assert isinstance(composed, Product)
    assert len(composed.children) == 2
    assert composed.shape == inner.shape
    assert composed.target_shape == outer.target_shape
    assert [composed(index) for index in range(inner.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_aligned_nested_products_preserve_the_explicit_expression_tree():
    outer = Product(
        Product(
            _LookupMap((1, 0), 3),
            _LookupMap((1, 0), 3),
        ),
        _LookupMap((1, 0), 3),
    )
    inner = Product(
        Product(
            _LookupMap((1, 0), 2),
            _LookupMap((1, 0), 2),
        ),
        _LookupMap((1, 0), 2),
    )

    composed = outer.compose(inner)

    assert isinstance(composed, Product)
    assert isinstance(composed.children[0], Product)
    assert len(composed.children) == 2
    assert len(composed.children[0].children) == 2
    assert composed.shape == inner.shape
    assert [composed(index) for index in range(inner.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_unaligned_product_trees_use_generic_composition_without_flattening():
    leaf = _LookupMap((1, 0), 2)
    outer = Product(Product(leaf, leaf), leaf)
    inner = Product(leaf, leaf, leaf)

    composed = outer.compose(inner)

    assert isinstance(composed, IndexMap)
    assert not isinstance(composed, Product)
    assert composed.shape == inner.shape
    assert composed.codomain_size == outer.codomain_size
    assert [composed(index) for index in range(inner.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_smaller_aligned_product_leaf_bounds_use_generic_composition():
    outer = Product(
        _LookupMap((2, 0, 1), 4),
        _LookupMap((1, 0), 3),
    )
    inner = Product(
        _LookupMap((1, 0), 2),
        _LookupMap((1, 0), 2),
    )

    composed = outer.compose(inner)

    assert isinstance(composed, IndexMap)
    assert not isinstance(composed, Product)
    assert inner.codomain_size < outer.size
    assert [composed(index) for index in range(inner.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_aligned_products_close_over_rank_zero_and_nested_empty_modes():
    outer = Product(
        Product(_RankZeroMap(0, 2), _RankZeroMap(0, 3)),
        _ShapedMap(Shape([[]]), (0,), 4),
    )
    inner = Product(
        Product(_RankZeroMap(0, 1), _RankZeroMap(0, 1)),
        _RankZeroMap(0, 1),
    )

    composed = outer.compose(inner)

    assert isinstance(composed, Product)
    assert isinstance(composed.children[0], Product)
    assert composed.shape == inner.shape
    assert composed.shape == Shape([[], []], [])
    assert composed(0) == outer(inner(0))


def test_product_preserves_explicit_domain_and_target_hierarchy():
    first = _LookupMap((1, 0), 3)
    second = _LookupMap((2, 0, 1), 4)
    third = _LookupMap((4, 1), 5)
    inner = Product(first, second)

    nested = Product(inner, third)
    variadic = Product(first, second, third)

    assert nested.children == (inner, third)
    assert nested.shape == Shape([2, 3], 2)
    assert nested.target_shape == Shape([3, 4], 5)
    assert variadic.children == (first, second, third)
    assert variadic.shape == Shape(2, 3, 2)
    assert variadic.target_shape == Shape(3, 4, 5)
    assert nested.shape != variadic.shape
    assert nested.target_shape != variadic.target_shape

    for coordinates in cartesian_product(range(2), range(3), range(2)):
        first_coordinate, second_coordinate, third_coordinate = coordinates
        nested_coordinate = (
            (first_coordinate, second_coordinate),
            third_coordinate,
        )
        expected = variadic(coordinates)

        assert nested(nested_coordinate) == expected


def test_nested_product_accepts_scalar_forms_for_each_hierarchical_subtree():
    first = _LookupMap((1, 0), 3)
    second = _LookupMap((2, 0, 1), 4)
    third = _LookupMap((4, 1), 5)
    nested = Product(Product(first, second), third)

    for ordinal in range(nested.size):
        coordinate = nested.shape.decode(ordinal)
        inner_coordinate, third_coordinate = coordinate
        inner_ordinal = nested.children[0].shape.encode(inner_coordinate)
        expected = nested(coordinate)

        assert nested((inner_ordinal, third_coordinate)) == expected
        assert nested([inner_ordinal, third_coordinate]) == expected
        assert nested(ordinal) == expected


def test_product_propagates_tri_state_injectivity_by_identity():
    injective = _LookupMap((1, 0), 2, True)
    non_injective = _LookupMap((0, 0), 1, False)
    unknown = _LookupMap((1, 0), 2, None)

    assert Product(injective, injective).is_injective is True
    assert Product(injective, unknown).is_injective is None
    assert Product(injective, non_injective).is_injective is False
    assert Product(non_injective, unknown).is_injective is False


def test_flat_product_rejects_ordinary_semantic_mutation():
    product = Product(
        _LookupMap((1, 0), 2),
        _LookupMap((2, 0, 1), 3),
    )

    for name, value in (
        ("children", ()),
        ("shape", Shape(1)),
        ("target_shape", Shape(1)),
        ("codomain_size", 1),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(product, name, value)
        with pytest.raises(AttributeError, match="immutable"):
            delattr(product, name)


@pytest.mark.parametrize("children", [(), (_LookupMap((0,), 1),)])
def test_flat_product_requires_at_least_two_children(children: tuple[IndexMap, ...]):
    with pytest.raises(ValueError, match="at least two"):
        Product(*children)


@pytest.mark.parametrize(
    "children",
    [
        (_LookupMap((0,), 1), object()),
        (object(), _LookupMap((0,), 1)),
    ],
)
def test_flat_product_rejects_non_map_children(children: tuple[object, object]):
    with pytest.raises(TypeError, match="IndexMap"):
        Product(*children)  # type: ignore[arg-type]
