from typing import Any

import neotorch
import pytest
from neotorch import (
    DataType,
    Generic,
    GenericAddOperation,
    GenericEvictable,
    GenericMatmulOperation,
    GenericReduceSumOperation,
    GenericScalarMulOperation,
    Layout,
    Shape,
    Stride,
)
from neotorch.tensor import Tensor


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_tensor_public_api_imports():
    assert neotorch.Tensor is Tensor


def test_tensor_constructor_exposes_read_only_api():
    data = Generic(["alpha", "beta", "gamma", "delta"])
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 0, layout)

    assert tensor.data is data
    assert tensor.offset == 0
    assert tensor.layout is layout
    assert tensor.size() == layout.shape.logical_size
    assert tensor.dtype() is DataType.Any
    assert tensor.device() is type(data)
    assert tensor.autograd_ctx is None
    assert tensor.grad is None

    with pytest.raises(AttributeError):
        setattr(tensor, "data", data)
    with pytest.raises(AttributeError):
        setattr(tensor, "offset", 1)
    with pytest.raises(AttributeError):
        setattr(tensor, "layout", layout)


def test_tensor_autograd_fields_are_writable():
    data = Generic(["alpha"])
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(data, 0, layout)
    operation = GenericAddOperation()
    grad = Tensor(Generic([1]), 0, layout)

    tensor.autograd_ctx = operation
    tensor.grad = grad

    assert tensor.autograd_ctx is operation
    assert tensor.grad is grad


def test_tensor_mutability_delegates_to_backing_data():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    mutable_tensor = Tensor(Generic(range(4)), 0, layout)
    immutable_tensor = Tensor(Generic(range(4), mutable=False), 0, layout)

    assert mutable_tensor.is_mutable()
    assert not immutable_tensor.is_mutable()


def test_tensor_indexes_flat_coordinate_key():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 5, layout)

    assert tensor[2, 3] == data[5 + layout.index([2, 3])]


def test_tensor_single_integer_key_uses_layout_expansion():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 5, layout)

    assert tensor[5] == data[5 + layout.index(5)]


def test_tensor_indexes_nested_layout_key():
    data = Generic(range(400))
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    tensor = Tensor(data, 7, layout)

    assert tensor[1, [2, 3]] == data[7 + layout.index([1, [2, 3]])]


def test_tensor_accepts_tuple_and_list_coordinate_keys():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    assert tensor[1, 2] == data[layout.index([1, 2])]
    assert tensor[[1, 2]] == data[layout.index([1, 2])]
    assert tensor[[1, 2]] == tensor[1, 2]


def test_tensor_list_coordinate_key_matches_tuple_key_for_hierarchical_mode():
    layout = Layout(Shape([2, 3, [2, 2]]), Stride([1, 2, [6, 12]]))
    tensor = Tensor(Generic(range(24)), 0, layout)

    assert tensor[[1, 2, [1, 1]]] == tensor[1, 2, [1, 1]]


def test_tensor_out_of_domain_keys_raise_layout_errors():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    with pytest.raises(ValueError):
        tensor[3, 0]
    with pytest.raises(ValueError):
        tensor[12]


def test_tensor_rejects_slice_and_non_integer_keys():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)
    string_key: Any = "x"
    mixed_key: Any = [1, "x"]
    slice_key: Any = slice(1, 2)

    with pytest.raises(TypeError):
        tensor[slice_key]
    with pytest.raises(TypeError):
        tensor[string_key]
    with pytest.raises(TypeError):
        tensor[mixed_key]


def test_tensor_setitem_updates_flat_coordinate_key():
    values: list[Any] = list(range(64))
    data = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 5, layout)
    data_index = 5 + layout.index([2, 3])

    tensor[2, 3] = "updated"

    assert values[data_index] == "updated"
    assert tensor[2, 3] == "updated"


def test_tensor_setitem_uses_integer_key_expansion():
    values: list[Any] = list(range(64))
    data = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 5, layout)
    data_index = 5 + layout.index(5)

    tensor[5] = "updated"

    assert values[data_index] == "updated"
    assert tensor[5] == "updated"


def test_tensor_setitem_updates_nested_layout_key():
    values: list[Any] = list(range(400))
    data = Generic(values)
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    tensor = Tensor(data, 7, layout)
    data_index = 7 + layout.index([1, [2, 3]])

    tensor[1, [2, 3]] = "updated"

    assert values[data_index] == "updated"
    assert tensor[1, [2, 3]] == "updated"


def test_tensor_setitem_accepts_tuple_and_list_coordinate_keys():
    values: list[Any] = list(range(64))
    data = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    tensor[1, 2] = "tuple"
    tensor[[2, 3]] = "list"
    tensor[[1, 2]] = "list-overwrite"

    assert values[layout.index([1, 2])] == "list-overwrite"
    assert tensor[1, 2] == "list-overwrite"
    assert tensor[[1, 2]] == tensor[1, 2]
    assert values[layout.index([2, 3])] == "list"


def test_tensor_setitem_list_coordinate_key_matches_tuple_key_for_hierarchical_mode():
    values: list[Any] = list(range(24))
    layout = Layout(Shape([2, 3, [2, 2]]), Stride([1, 2, [6, 12]]))
    tensor = Tensor(Generic(values), 0, layout)

    tensor[[1, 2, [1, 1]]] = "updated"

    assert tensor[1, 2, [1, 1]] == "updated"
    assert tensor[[1, 2, [1, 1]]] == tensor[1, 2, [1, 1]]
    assert values[layout.index([1, 2, [1, 1]])] == "updated"


def test_tensor_setitem_out_of_domain_keys_raise_layout_errors():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    with pytest.raises(ValueError):
        tensor[3, 0] = "updated"
    with pytest.raises(ValueError):
        tensor[12] = "updated"


def test_tensor_setitem_rejects_slice_and_non_integer_keys():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)
    string_key: Any = "x"
    mixed_key: Any = [1, "x"]
    slice_key: Any = slice(1, 2)

    with pytest.raises(TypeError):
        tensor[slice_key] = "updated"
    with pytest.raises(TypeError):
        tensor[string_key] = "updated"
    with pytest.raises(TypeError):
        tensor[mixed_key] = "updated"


def test_tensor_setitem_rejects_immutable_backing_data():
    data = Generic(range(64), mutable=False)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    with pytest.raises(RuntimeError):
        tensor[1, 2] = "updated"


def test_tensor_add_public_api_imports():
    assert neotorch.GenericAddOperation is GenericAddOperation
    assert neotorch.GenericScalarMulOperation is GenericScalarMulOperation
    assert neotorch.GenericReduceSumOperation is GenericReduceSumOperation
    assert neotorch.GenericMatmulOperation is GenericMatmulOperation
    assert neotorch.add is not None
    assert neotorch.mul is not None
    assert neotorch.reduce is not None
    assert neotorch.matmul is not None


def test_tensor_add_with_generic_data_returns_autograd_tensor():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = lhs + rhs

    assert tensor_values(result) == [11, 22, 33, 44]
    assert result.layout == layout
    assert result.dtype() is DataType.Any
    assert result.device() is Generic
    assert isinstance(result.autograd_ctx, GenericAddOperation)
    assert result.autograd_ctx.inputs() == (lhs, rhs)


def test_tensor_add_function_matches_operator():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = neotorch.add(lhs, rhs)

    assert tensor_values(result) == [11, 22, 33, 44]
    assert isinstance(result.autograd_ctx, GenericAddOperation)


def test_tensor_add_rejects_mismatched_layouts():
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([1, 2])))
    rhs = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([2, 1])))

    with pytest.raises(ValueError):
        _ = lhs + rhs


def test_tensor_add_rejects_non_tensor_operand():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    with pytest.raises(TypeError):
        _ = lhs + 1


def test_tensor_add_rejects_non_generic_data(tmp_path):
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(GenericEvictable([1, 2, 3, 4], tmp_path / "lhs.pkl"), 0, layout)
    rhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    with pytest.raises(TypeError):
        _ = lhs + rhs


def test_tensor_scalar_mul_accepts_any_layout_and_preserves_layout():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    tensor = Tensor(Generic(list(range(10))), 0, layout)

    left_result = tensor * 2
    right_result = 3 * tensor
    function_result = neotorch.mul(tensor, 4)

    assert left_result.layout == layout
    assert right_result.layout == layout
    assert function_result.layout == layout
    assert tensor_values(left_result) == [0, 2, 8, 10, 16, 18]
    assert tensor_values(right_result) == [0, 3, 12, 15, 24, 27]
    assert tensor_values(function_result) == [0, 4, 16, 20, 32, 36]
    assert isinstance(left_result.autograd_ctx, GenericScalarMulOperation)


def test_tensor_scalar_mul_backward_scales_gradient():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    tensor = Tensor(Generic(list(range(10))), 0, layout)
    result = tensor * 5
    gradient = Tensor(Generic([1] * 10), 0, layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [5, 5, 5, 5, 5, 5]
    assert type(tensor_grad.data) is type(tensor.data)


def test_tensor_scalar_mul_rejects_non_numeric_scalar_and_non_generic_data(tmp_path):
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    evictable = Tensor(GenericEvictable([1, 2, 3, 4], tmp_path / "data.pkl"), 0, layout)

    with pytest.raises(TypeError):
        _ = tensor * "x"

    with pytest.raises(TypeError):
        _ = evictable * 2


def test_tensor_reduce_sums_second_mode():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, layout)

    result = neotorch.reduce(tensor)

    assert result.layout == Layout(Shape(2), Stride(1))
    assert tensor_values(result) == [9, 12]
    assert isinstance(result.autograd_ctx, GenericReduceSumOperation)


def test_tensor_reduce_preserves_hierarchical_first_mode_with_column_major_layout():
    layout = Layout(Shape([[2, 2], 3]), Stride([[1, 2], 4]))
    tensor = Tensor(Generic(range(1, 13)), 0, layout)

    result = neotorch.reduce(tensor)

    assert result.layout == Layout(Shape([2, 2]), Stride([1, 2]))
    assert tensor_values(result) == [15, 18, 21, 24]


def test_tensor_reduce_backward_copies_gradient_over_second_mode():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, layout)
    result = neotorch.reduce(tensor)
    gradient = Tensor(Generic([10, 20]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [10, 20, 10, 20, 10, 20]
    assert type(tensor_grad.data) is type(tensor.data)


def test_tensor_reduce_rejects_non_two_mode_or_non_generic_data(tmp_path):
    one_mode = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
    evictable = Tensor(
        GenericEvictable([1, 2, 3, 4], tmp_path / "data.pkl"),
        0,
        Layout(Shape([2, 2]), Stride([1, 2])),
    )

    with pytest.raises(ValueError):
        neotorch.reduce(one_mode)

    with pytest.raises(TypeError):
        neotorch.reduce(evictable)


def test_tensor_matmul_computes_nk_by_mk_to_nm():
    a = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    b = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = a @ b

    assert result.layout == Layout(Shape([2, 4]), Stride([1, 2]))
    assert tensor_values(result) == [1, 2, 3, 4, 5, 6, 9, 12]
    assert isinstance(result.autograd_ctx, GenericMatmulOperation)


def test_tensor_matmul_preserves_hierarchical_row_modes():
    a = Tensor(
        Generic(range(1, 13)),
        0,
        Layout(Shape([[2, 2], 3]), Stride([[1, 2], 4])),
    )
    b = Tensor(
        Generic([1, 0, 0, 1, 0, 0]),
        0,
        Layout(Shape([[2, 1], 3]), Stride([[1, 2], 2])),
    )

    result = neotorch.matmul(a, b)

    assert result.layout == Layout(Shape([[2, 2], [2, 1]]), Stride([[1, 2], [4, 8]]))
    assert tensor_values(result) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_tensor_matmul_backward_computes_input_gradients():
    a = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    b = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )
    result = a @ b
    gradient = Tensor(Generic([1] * 8), 0, result.layout)

    result.backward(gradient)
    a_grad = require_grad(a)
    b_grad = require_grad(b)

    assert tensor_values(a_grad) == [2, 2, 2, 2, 2, 2]
    assert tensor_values(b_grad) == [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11]
    assert type(a_grad.data) is type(a.data)
    assert type(b_grad.data) is type(b.data)


def test_tensor_matmul_rejects_invalid_shapes_and_data(tmp_path):
    a = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([1, 2])))
    bad_k = Tensor(Generic([1, 2, 3]), 0, Layout(Shape([1, 3]), Stride([1, 1])))
    one_mode = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
    evictable = Tensor(
        GenericEvictable([1, 2, 3, 4], tmp_path / "data.pkl"), 0, a.layout
    )

    with pytest.raises(ValueError):
        _ = a @ bad_k

    with pytest.raises(ValueError):
        _ = a @ one_mode

    with pytest.raises(TypeError):
        _ = evictable @ a


def test_tensor_backward_on_leaf_creates_detached_generic_grad():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    gradient = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    tensor.backward(gradient)
    gradient[0] = 999
    tensor_grad = require_grad(tensor)

    assert tensor.grad is not gradient
    assert type(tensor_grad.data) is type(tensor.data)
    assert tensor_values(tensor_grad) == [10, 20, 30, 40]


def test_tensor_backward_through_add_sets_input_grads():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.backward(gradient)
    result_grad = require_grad(result)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert tensor_values(result_grad) == [1, 2, 3, 4]
    assert tensor_values(lhs_grad) == [1, 2, 3, 4]
    assert tensor_values(rhs_grad) == [1, 2, 3, 4]
    assert type(lhs_grad.data) is type(lhs.data)
    assert type(rhs_grad.data) is type(rhs.data)


def test_tensor_backward_accumulates_repeated_calls():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    result.backward(gradient)
    result.backward(gradient)
    result_grad = require_grad(result)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert tensor_values(result_grad) == [2, 2, 2, 2]
    assert tensor_values(lhs_grad) == [2, 2, 2, 2]
    assert tensor_values(rhs_grad) == [2, 2, 2, 2]


def test_tensor_backward_accumulates_shared_input_contributions():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    result = tensor + tensor
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [2, 4, 6, 8]


def test_tensor_backward_rejects_invalid_gradient():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    wrong_layout_gradient = Tensor(
        Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([2, 1]))
    )
    invalid_gradient: Any = "gradient"

    with pytest.raises(TypeError):
        tensor.backward(invalid_gradient)

    with pytest.raises(ValueError):
        tensor.backward(wrong_layout_gradient)


def test_tensor_rejects_negative_offset():
    data = Generic(range(4))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))

    with pytest.raises(ValueError):
        Tensor(data, -1, layout)


def test_tensor_rejects_storage_that_exceeds_data_size():
    data = Generic(range(4))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))

    with pytest.raises(ValueError):
        Tensor(data, 1, layout)


def test_tensor_storage_validation_uses_cosize_not_logical_size():
    layout = Layout(Shape([2, 2]), Stride([1, 10]))

    with pytest.raises(ValueError):
        Tensor(Generic(range(11)), 0, layout)

    tensor = Tensor(Generic(range(12)), 0, layout)
    assert tensor[1, 1] == 11


def test_tensor_propagates_backing_data_lifecycle_errors(tmp_path):
    path = tmp_path / "data.pkl"
    data = GenericEvictable(range(16), path)
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 3, layout)

    assert tensor[1, 1] == 6

    data.evict()
    with pytest.raises(RuntimeError):
        tensor[1, 1]

    data.promote()
    assert tensor[1, 1] == 6


def test_tensor_setitem_propagates_backing_data_lifecycle_errors(tmp_path):
    path = tmp_path / "data.pkl"
    data = GenericEvictable(range(16), path)
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 3, layout)

    tensor[1, 1] = "updated"
    assert tensor[1, 1] == "updated"

    data.evict()
    with pytest.raises(RuntimeError):
        tensor[1, 1] = "evicted"

    data.promote()
    assert tensor[1, 1] == "updated"
