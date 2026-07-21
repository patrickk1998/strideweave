import random

import pytest

import strideweave.friendly as friendly
from strideweave import CPU, DType, Layout, Shape, Stride


def test_column_major_matches_hand_built_layouts():
    assert friendly.column_major(2, 3) == Layout(Shape([2, 3]), Stride([1, 2]))
    assert friendly.column_major(2, 3, 4) == Layout(Shape([2, 3, 4]), Stride([1, 2, 6]))
    assert friendly.column_major(5) == Layout(Shape(5), Stride(1))


def test_row_major_matches_hand_built_layouts():
    assert friendly.row_major(2, 3) == Layout(Shape([2, 3]), Stride([3, 1]))
    assert friendly.row_major(2, 3, 4) == Layout(Shape([2, 3, 4]), Stride([12, 4, 1]))


def test_layout_builders_reject_invalid_extents():
    with pytest.raises(ValueError, match="at least one extent"):
        friendly.column_major()
    with pytest.raises(ValueError, match="positive"):
        friendly.row_major(2, 0)


def test_tensor_from_nested_lists_round_trips_coordinates():
    tensor = friendly.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert tensor.layout == friendly.column_major(2, 3)
    assert type(tensor.carrier) is CPU
    for i in range(2):
        for j in range(3):
            assert tensor[i, j] == pytest.approx(1.0 + 3 * i + j)


def test_tensor_from_flat_list_builds_one_mode_tensor():
    tensor = friendly.tensor([5.0, 6.0, 7.0])

    assert tensor.layout == friendly.column_major(3)
    assert friendly.to_list(tensor) == [5.0, 6.0, 7.0]


def test_tensor_with_explicit_layout_fills_logical_order():
    tensor = friendly.tensor([1.0, 2.0, 3.0, 4.0], layout=friendly.row_major(2, 2))

    # Logical order enumerates the first mode fastest.
    assert tensor[0, 0] == 1.0
    assert tensor[1, 0] == 2.0
    assert tensor[0, 1] == 3.0
    assert tensor[1, 1] == 4.0


def test_tensor_rejects_ragged_and_empty_values():
    with pytest.raises(ValueError, match="rectangular"):
        friendly.tensor([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="empty"):
        friendly.tensor([[], []])
    with pytest.raises(ValueError, match="logical size"):
        friendly.tensor([1.0], layout=friendly.column_major(2))


def test_creation_helpers_fill_expected_values():
    assert friendly.to_list(friendly.zeros(2, 2)) == [0.0] * 4
    assert friendly.to_list(friendly.ones(3)) == [1.0] * 3
    assert friendly.to_list(friendly.full(2, value=7.5)) == [7.5, 7.5]
    assert friendly.to_list(friendly.arange(4)) == [0.0, 1.0, 2.0, 3.0]


def test_creation_helpers_support_int32():
    tensor = friendly.ones(2, dtype=DType.Int32)

    assert tensor.dtype() is DType.Int32
    assert friendly.to_list(tensor) == [1, 1]


def test_rand_and_randn_are_reproducible_with_seeded_rng():
    first = friendly.to_list(friendly.rand(2, 3, rng=random.Random(0)))
    second = friendly.to_list(friendly.rand(2, 3, rng=random.Random(0)))

    assert first == second
    assert all(0.0 <= value < 1.0 for value in first)
    assert friendly.to_list(friendly.randn(4, rng=random.Random(1))) == (
        friendly.to_list(friendly.randn(4, rng=random.Random(1)))
    )


def test_sum_reduces_to_scalar_layout_and_backpropagates():
    tensor = friendly.tensor([[1.0, 2.0], [3.0, 4.0]])

    total = friendly.sum(tensor)

    assert total.layout == Layout(Shape(1), Stride(1))
    assert friendly.item(total) == pytest.approx(10.0)

    total.backward()

    assert tensor.grad is not None
    assert friendly.to_list(tensor.grad) == pytest.approx([1.0] * 4)


def test_sum_handles_one_and_three_mode_tensors():
    assert friendly.item(friendly.sum(friendly.arange(4))) == pytest.approx(6.0)
    three_mode = friendly.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    assert friendly.item(friendly.sum(three_mode)) == pytest.approx(10.0)


def test_mean_value_and_gradient():
    tensor = friendly.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = friendly.mean(tensor)

    assert friendly.item(result) == pytest.approx(2.5)

    result.backward()

    assert tensor.grad is not None
    assert friendly.to_list(tensor.grad) == pytest.approx([0.25] * 4)


def test_item_rejects_non_scalar_tensors():
    with pytest.raises(ValueError, match="exactly one element"):
        friendly.item(friendly.ones(2))


def test_to_list_enumerates_logical_order():
    tensor = friendly.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Column-major logical order: first mode fastest.
    assert friendly.to_list(tensor) == [1.0, 3.0, 2.0, 4.0]
