from importlib import import_module
from typing import Any, Protocol, cast

import neotorch
import pytest
from neotorch import Layout, Node, Shape, Stride, Tree


class NativeIndexModule(Protocol):
    def get_index(self, layout: Layout, key: Any) -> int: ...


native_index = cast(NativeIndexModule, import_module("neotorch._index"))


def test_public_api_imports():
    assert neotorch.Layout is Layout
    assert neotorch.Node is Node
    assert neotorch.Shape is Shape
    assert neotorch.Stride is Stride
    assert neotorch.Tree is Tree


def test_tree_reshape():
    assert Tree(1, Tree(1, 1)).reshape(["A", "B", "C"]) == ["A", ["B", "C"]]


def test_tree_id_leaf_marker():
    marker = Node.id(2)

    assert marker.id == 2
    assert Tree(marker).recipe == [Node.Leaf]
    assert Tree(marker).size == 1


def test_tree_id_leaf_rejects_invalid_ids():
    invalid_id: Any = "0"

    with pytest.raises(TypeError):
        Node.id(invalid_id)
    with pytest.raises(ValueError):
        Node.id(-1)


def test_shape_creation_and_indexing():
    shape = Shape([1, [2, [3, 3], [3, 3], [3, 3]]])

    assert shape[0] == Shape(1)
    assert shape[1][0] == Shape(2)


def test_shape_variadic_creation_matches_list_creation():
    assert Shape(1, 2, [3, 4]) == Shape([1, 2, [3, 4]])


def test_shape_concat():
    assert Shape([1, 2, 3]) == Shape.concat(Shape(1), Shape([2, 3]))


def test_shape_append():
    assert Shape([1, [2, 3]]) == Shape.append(Shape(1), Shape([2, 3]))


def test_stride_creation_and_indexing():
    stride = Stride([1, [2, [3, 3], [3, 3], [3, 3]]])

    assert stride[0] == Stride(1)
    assert stride[1][0] == Stride(2)


def test_stride_variadic_creation_matches_list_creation():
    assert Stride(1, 2, [3, 4]) == Stride([1, 2, [3, 4]])


def test_stride_allows_zero_for_singleton_layouts():
    assert Stride(0) == Stride([0])

    with pytest.raises(ValueError):
        Stride(-1)


def test_layout_slicing():
    layout = Layout(Shape([1, 2, [3, 4]]), Stride([1, 1, [2, 6]]))

    assert layout[1:] == Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]]))


def test_layout_divide():
    assert Layout.divide(
        Layout(Shape([4, 2, 3]), Stride([2, 1, 8])), Layout.leaf(4, 2)
    ) == Layout(Shape([[2, 2], [2, 3]]), Stride([[4, 1], [2, 8]]))


def test_layout_zipped_divide():
    correct_layout = Layout(
        Shape([[3, [2, 4]], [3, [2, 2]]]),
        Stride([[177, [13, 2]], [59, [26, 1]]]),
    )

    assert (
        Layout.zipped_divide(
            Layout(Shape([9, [4, 8]]), Stride([59, [13, 1]])),
            [Layout.leaf(3, 3), Layout(Shape([2, 4]), Stride([1, 8]))],
        )
        == correct_layout
    )


def test_layout_divide_tiler_and_index():
    layout_a = Layout(Shape([9, [4, 8]]), Stride([59, [13, 1]]))
    tiler = [Layout.leaf(3, 3), Layout(Shape([2, 4]), Stride([1, 8]))]

    assert Layout.divide_tiler(layout_a, tiler).index([1, 1]) == 177 + 13


def test_layout_get_index_flat_coordinate_key():
    layout = Layout(Shape([3, 4]), Stride([2, 10]))

    assert Layout.get_index(layout, [2, 3]) == 34


def test_layout_get_index_flat_integer_key_expansion():
    layout = Layout(Shape([3, 4]), Stride([2, 10]))

    assert Layout.get_index(layout, 5) == 14


def test_layout_get_index_nested_coordinate_key():
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))

    assert Layout.get_index(layout, [1, [2, 3]]) == 321


def test_layout_get_index_out_of_domain_key_raises_value_error():
    layout = Layout(Shape([3, 4]), Stride([2, 10]))

    with pytest.raises(ValueError):
        Layout.get_index(layout, [3, 0])

    with pytest.raises(ValueError):
        Layout.get_index(layout, 12)


def test_layout_index_and_call_match_static_get_index():
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    key = [1, [2, 3]]

    assert layout.index(key) == Layout.get_index(layout, key)
    assert layout(key) == Layout.get_index(layout, key)


def test_native_get_index_matches_python_get_index():
    flat_layout = Layout(Shape([3, 4]), Stride([2, 10]))
    nested_layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))

    cases = [
        (flat_layout, [2, 3]),
        (flat_layout, 5),
        (nested_layout, [1, [2, 3]]),
    ]

    for layout, key in cases:
        assert native_index.get_index(layout, key) == Layout.get_index(layout, key)


def test_native_get_index_matches_python_value_error_cases():
    flat_layout = Layout(Shape([3, 4]), Stride([2, 10]))
    nested_layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))

    cases = [
        (flat_layout, [3, 0]),
        (flat_layout, 12),
        (nested_layout, [1, [3, 0]]),
    ]

    for layout, key in cases:
        with pytest.raises(ValueError):
            Layout.get_index(layout, key)
        with pytest.raises(ValueError):
            native_index.get_index(layout, key)


def test_layout_leaf_edge_case():
    assert Layout(Shape(1), Stride(1)) == Layout(Shape([1]), Stride([1]))
    assert Layout(Shape(1), Stride(1)).is_leaf


def test_layout_concat():
    assert Layout(Shape([1, 2, 3]), Stride([1, 2, 6])) == Layout.concat(
        Layout(Shape([1, 2]), Stride([1, 2])), Layout(Shape(3), Stride(6))
    )


def test_layout_append():
    assert Layout(Shape([1, [2, 3]]), Stride([1, [2, 6]])) == Layout.append(
        Layout(Shape(1), Stride(1)), Layout(Shape([2, 3]), Stride([2, 6]))
    )


def test_layout_extract_profile():
    extracted = Layout.extract_profile(
        Layout(Shape([1, [2, 3]]), Stride([1, [2, 6]])), Tree(1, 1)
    )

    assert extracted[0] == Layout(Shape(1), Stride(1))
    assert extracted[1] == Layout(Shape([2, 3]), Stride([2, 6]))


def test_layout_extract_profile_defaults_to_prefix_leaf_traversal():
    extracted = Layout.extract_profile(Layout(Shape([1, [2, 3]]), Stride([5, [7, 11]])))

    assert extracted == [
        Layout(Shape(1), Stride(5)),
        Layout(Shape(2), Stride(7)),
        Layout(Shape(3), Stride(11)),
    ]


def test_layout_extract_profile_ignores_id_leaf_values():
    extracted = Layout.extract_profile(
        Layout(Shape([1, [2, 3]]), Stride([1, [2, 6]])),
        Tree(Node.id(7), Node.id(0)),
    )

    assert extracted[0] == Layout(Shape(1), Stride(1))
    assert extracted[1] == Layout(Shape([2, 3]), Stride([2, 6]))


def test_layout_rearrange_default_selection_permute_flat_leaves():
    layout = Layout(Shape([2, 3, 4]), Stride([1, 2, 6]))

    assert Layout.rearrange(layout, Tree(Node.id(2), Node.id(0), Node.id(1))) == Layout(
        Shape([4, 2, 3]), Stride([6, 1, 2])
    )


def test_layout_rearrange_selection_extracts_subtrees():
    layout = Layout(Shape([1, [2, 3]]), Stride([5, [7, 14]]))

    assert Layout.rearrange(
        layout,
        Tree(Node.id(1), Node.id(0)),
        Tree(1, 1),
    ) == Layout(Shape([[2, 3], 1]), Stride([[7, 14], 5]))


def test_layout_rearrange_plain_leaves_insert_singleton_modes():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))

    assert Layout.rearrange(layout, Tree(Node.id(1), Node.Leaf, Node.id(0))) == Layout(
        Shape([3, 1, 2]), Stride([2, 0, 1])
    )


def test_layout_rearrange_preserves_nested_output_structure():
    layout = Layout(Shape([2, 3, 4]), Stride([1, 2, 6]))

    assert Layout.rearrange(
        layout,
        Tree(Tree(Node.id(1), Node.id(2)), Node.id(0)),
    ) == Layout(Shape([[3, 4], 2]), Stride([[2, 6], 1]))


def test_layout_rearrange_allows_omitting_singleton_leaf_ids():
    layout = Layout(Shape([1, 2, 1, 3]), Stride([7, 1, 11, 2]))

    assert Layout.rearrange(layout, Tree(Node.id(3), Node.id(1))) == Layout(
        Shape([3, 2]), Stride([2, 1])
    )


def test_layout_rearrange_rejects_invalid_id_coverage():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))

    with pytest.raises(ValueError):
        Layout.rearrange(layout, Tree(Node.id(0), Node.id(0)))
    with pytest.raises(ValueError):
        Layout.rearrange(layout, Tree(Node.id(0)))
    with pytest.raises(ValueError):
        Layout.rearrange(layout, Tree(Node.id(2), Node.id(0), Node.id(1)))


def test_layout_reverse_rearrange_inverts_simple_permutation():
    reverse_output, reverse_selection = Layout.reverse_rearrange(
        Tree(Node.id(2), Node.id(0), Node.id(1)),
        Tree(1, 1, 1),
    )

    assert reverse_output == Tree(Node.id(1), Node.id(2), Node.id(0))
    assert reverse_selection == Tree(1, 1, 1)


def test_layout_reverse_rearrange_inverts_inserted_singletons():
    reverse_output, reverse_selection = Layout.reverse_rearrange(
        Tree(Node.id(1), Node.Leaf, Node.id(0)),
        Tree(1, 1),
    )

    assert reverse_output == Tree(Node.id(2), Node.id(0))
    assert reverse_selection == Tree(1, 1, 1)


def test_layout_reverse_rearrange_preserves_nested_output_structure():
    reverse_output, reverse_selection = Layout.reverse_rearrange(
        Tree(Tree(Node.id(1), Node.id(2)), Node.id(0)),
        Tree(1, 1, 1),
    )

    assert reverse_output == Tree(Node.id(2), Node.id(0), Node.id(1))
    assert reverse_selection == Tree(Tree(1, 1), 1)


def test_layout_reverse_rearrange_reconstructs_omitted_singleton_ids():
    reverse_output, reverse_selection = Layout.reverse_rearrange(
        Tree(Node.id(0)),
        Tree(1, 1),
    )

    assert reverse_output == Tree(Node.id(0), Node.Leaf)
    assert reverse_selection == Tree(1)


def test_layout_reverse_rearrange_composes_to_identity_for_permutation():
    layout = Layout(Shape([2, 3, 4]), Stride([1, 2, 6]))
    output = Tree(Node.id(2), Node.id(0), Node.id(1))
    selection = Tree(1, 1, 1)

    forward = Layout.rearrange(layout, output, selection)
    reverse_output, reverse_selection = Layout.reverse_rearrange(output, selection)

    assert Layout.rearrange(forward, reverse_output, reverse_selection) == layout


def test_layout_reverse_rearrange_composes_to_identity_with_inserted_singleton():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    output = Tree(Node.id(1), Node.Leaf, Node.id(0))
    selection = Tree(1, 1)

    forward = Layout.rearrange(layout, output, selection)
    reverse_output, reverse_selection = Layout.reverse_rearrange(output, selection)

    assert Layout.rearrange(forward, reverse_output, reverse_selection) == layout


def test_layout_reverse_rearrange_restores_omitted_singleton_shape():
    layout = Layout(Shape([2, 1, 3]), Stride([1, 99, 2]))
    output = Tree(Node.id(2), Node.id(0))
    selection = Tree(1, 1, 1)

    forward = Layout.rearrange(layout, output, selection)
    reverse_output, reverse_selection = Layout.reverse_rearrange(output, selection)
    restored = Layout.rearrange(forward, reverse_output, reverse_selection)

    assert restored.shape == layout.shape
    assert restored == Layout(Shape([2, 1, 3]), Stride([1, 0, 2]))


def test_layout_permute_reorders_top_level_modes():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))

    assert Layout.permute(layout, (1, 0, 2)) == Layout(
        Shape([[3, 4], 2, 5]), Stride([[2, 6], 1, 24])
    )


def test_layout_permute_accepts_variadic_and_list_orders():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))

    assert Layout.permute(layout, 1, 0) == Layout(Shape([3, 2]), Stride([2, 1]))
    assert Layout.permute(layout, [1, 0]) == Layout(Shape([3, 2]), Stride([2, 1]))


def test_layout_permute_identity_order_returns_equal_layout():
    layout = Layout(Shape([[2, 3], 4]), Stride([[1, 2], 6]))

    assert Layout.permute(layout, 0, 1) == layout


def test_layout_permute_rejects_invalid_orders():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    non_integer_dim: Any = "0"

    with pytest.raises(ValueError):
        Layout.permute(layout, 0, 0)
    with pytest.raises(ValueError):
        Layout.permute(layout, 0)
    with pytest.raises(ValueError):
        Layout.permute(layout, -1, 0)
    with pytest.raises(ValueError):
        Layout.permute(layout, 0, 2)
    with pytest.raises(TypeError):
        Layout.permute(layout, non_integer_dim, 1)


def test_layout_permute_composes_with_inverse_to_identity():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    order = (1, 2, 0)
    inverse_order = (2, 0, 1)

    assert Layout.permute(Layout.permute(layout, order), inverse_order) == layout


def test_layout_compose_with_leaf_left_hand_side():
    l1 = Layout(Shape([5]), Stride([2]))
    l2 = Layout(Shape([3, [5, 2]]), Stride([2, [6, 30]]))

    assert Layout.compose(l1, l2) == Layout(Shape([3, [5, 2]]), Stride([4, [12, 60]]))


def test_layout_flattening():
    true_recipe = [Node.Leaf, Node.Push, Node.Leaf, Node.Leaf, Node.Pop]
    flat_layout, recipe = Layout.flatten_layout(
        Layout(Shape([1, [2, 3]]), Stride([1, [2, 6]]))
    )

    assert flat_layout == Layout(Shape([1, 2, 3]), Stride([1, 2, 6]))
    assert recipe == true_recipe
    for el in flat_layout:
        assert el.is_leaf


def test_layout_coalesce():
    layout = Layout(Shape([1, [1, 2]]), Stride([1, [5, 25]]))

    assert Layout.coalesce(layout) == Layout(Shape(2), Stride(25))


def test_layout_coalesce_by_mode():
    layout = Layout(Shape([1, [1, 2]]), Stride([1, [5, 25]]))

    assert Layout.coalesce_by_mode(layout, Tree(1, 1)) == Layout(
        Shape([1, 2]), Stride([1, 25])
    )


def test_layout_choose_large_stride():
    layout = Layout(Shape([3, 6, 2, 8]), Stride([1, 3, 18, 36]))

    assert Layout.choose(layout, 72) == Layout(
        Shape([1, 1, 1, 4]), Stride([72, 72, 72, 72])
    )


def test_layout_choose_partial_modes():
    layout = Layout(Shape([3, 6, 2, 8]), Stride([1, 3, 18, 36]))

    assert Layout.choose(layout, 9) == Layout(
        Shape([1, 2, 2, 8]), Stride([9, 9, 18, 36])
    )


def test_layout_modout():
    layout = Layout(Shape([1, 2, 2, 8]), Stride([9, 9, 18, 36]))

    assert Layout.modout(layout, 16) == Layout(
        Shape([1, 2, 2, 4]), Stride([9, 9, 18, 36])
    )


def test_layout_compose_layouts():
    layout_a = Layout(Shape([6, 2]), Stride([8, 2]))
    layout_b = Layout(Shape([4, 3]), Stride([3, 1]))

    assert Layout.compose(layout_a, layout_b) == Layout(
        Shape([[2, 2], 3]), Stride([[24, 2], 8])
    )


def test_layout_compose_leaf_layout_with_layout():
    layout_a = Layout(Shape(20), Stride(2))
    layout_b = Layout(Shape([5, 4]), Stride([4, 1]))

    assert Layout.compose(layout_a, layout_b) == Layout(Shape([5, 4]), Stride([8, 2]))


def test_layout_compose_with_shape():
    layout_a = Layout(Shape([20, 50]), Stride([1, 20]))

    assert Layout.compose(layout_a, Shape([5, 2])) == Layout(
        Shape([5, 2]), Stride([1, 20])
    )


def test_layout_compose_with_tiler():
    layout_a = Layout(Shape([12, [4, 8]]), Stride([59, [13, 1]]))
    tile = (Layout.leaf(3, 4), Layout.leaf(8, 2))

    assert Layout.compose(layout_a, tile) == Layout(
        Shape([3, [2, 4]]), Stride([236, [26, 1]])
    )


def test_layout_append_nested_layout():
    assert Layout.append(
        Layout(Shape([1, 2]), Stride([1, 2])),
        Layout(Shape([3, 4]), Stride([1, 2])),
    ) == Layout(Shape([1, 2, [3, 4]]), Stride([1, 2, [1, 2]]))


def test_layout_concat_nested_layout():
    assert Layout.concat(
        Layout(Shape([1, 2]), Stride([1, 2])),
        Layout(Shape([3, 4]), Stride([1, 2])),
    ) == Layout(Shape([1, 2, 3, 4]), Stride([1, 2, 1, 2]))


def test_layout_complement():
    assert Layout.complement(Layout(Shape([2, 2]), Stride([1, 6])), 24) == Layout(
        Shape([3, 2]), Stride([2, 12])
    )
    assert Layout.complement(Layout.leaf(4, 2), 24) == Layout(
        Shape([2, 3]), Stride([1, 8])
    )
