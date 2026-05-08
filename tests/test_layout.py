import neotorch
from neotorch import Layout, Node, Shape, Stride, Tree


def test_public_api_imports():
    assert neotorch.Layout is Layout
    assert neotorch.Node is Node
    assert neotorch.Shape is Shape
    assert neotorch.Stride is Stride
    assert neotorch.Tree is Tree


def test_tree_reshape():
    assert Tree(1, Tree(1, 1)).reshape(["A", "B", "C"]) == ["A", ["B", "C"]]


def test_shape_creation_and_indexing():
    shape = Shape([1, [2, [3, 3], [3, 3], [3, 3]]])

    assert shape[0] == Shape(1)
    assert shape[1][0] == Shape(2)


def test_shape_concat():
    assert Shape([1, 2, 3]) == Shape.concat(Shape(1), Shape([2, 3]))


def test_shape_append():
    assert Shape([1, [2, 3]]) == Shape.append(Shape(1), Shape([2, 3]))


def test_stride_creation_and_indexing():
    stride = Stride([1, [2, [3, 3], [3, 3], [3, 3]]])

    assert stride[0] == Stride(1)
    assert stride[1][0] == Stride(2)


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
