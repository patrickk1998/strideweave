import math
from collections.abc import Iterable
from typing import Any

import pytest

import strideweave as sw
from strideweave import (
    CPU,
    Carrier,
    DType,
    Generic,
    GenericAddOperation,
    GenericDivOperation,
    GenericElementwiseMulOperation,
    GenericExpOperation,
    GenericMatmulOperation,
    GenericPowOperation,
    GenericReduceSumOperation,
    GenericScalarMulOperation,
    GenericSubOperation,
    GenericViewOperation,
    Layout,
    Node,
    PermuteOperation,
    RearrangeOperation,
    Shape,
    Stride,
    Tree,
)
from strideweave.tensor import Tensor


class UnsupportedData(Carrier):
    def __init__(self, values: list[Any]):
        super().__init__()
        self.values = values

    def size(self) -> int:
        return len(self.values)

    def dtype(self) -> DType:
        return DType.Any

    def get_value(self, index: int) -> Any:
        return self.values[index]

    def new_like(
        self, values: Iterable[Any], *, mutable: bool = True
    ) -> "UnsupportedData":
        return UnsupportedData(list(values))

    def empty_like(
        self, size: int, *, mutable: bool = True, dtype: DType | None = None
    ) -> "UnsupportedData":
        return UnsupportedData([None] * size)

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("UnsupportedData does not implement scatter")


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def tensor_with_logical_values(values: Iterable[Any], layout: Layout) -> Tensor:
    physical_values: list[Any] = [None] * layout._cache.cosize
    for logical_index, value in enumerate(values):
        physical_values[layout.index(logical_index)] = value
    return Tensor(Generic(physical_values), 0, layout)


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_tensor_public_api_imports():
    assert sw.Tensor is Tensor
    assert sw.GenericViewOperation is GenericViewOperation
    assert sw.view is not None


def test_tensor_constructor_exposes_read_only_api():
    carrier = Generic(["alpha", "beta", "gamma", "delta"], dtype=DType.Any)
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(carrier, 0, layout)

    assert tensor.carrier is carrier
    assert tensor.offset == 0
    assert tensor.layout is layout
    assert tensor.size() == layout.shape.logical_size
    assert tensor.dtype() is DType.Any
    assert tensor.carrier_type() is type(carrier)
    assert tensor.autograd_ctx is None
    with pytest.raises(RuntimeError, match="grad is not available"):
        tensor.grad

    with pytest.raises(AttributeError):
        setattr(tensor, "data", carrier)
    with pytest.raises(AttributeError):
        setattr(tensor, "offset", 1)
    with pytest.raises(AttributeError):
        setattr(tensor, "layout", layout)


def test_tensor_autograd_fields_are_writable():
    carrier = Generic(["alpha"])
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(carrier, 0, layout)
    operation = GenericAddOperation()
    grad = Tensor(Generic([1]), 0, layout)

    tensor.autograd_ctx = operation
    tensor.grad = grad

    assert tensor.autograd_ctx is operation
    assert tensor.grad is grad


def test_tensor_mutability_delegates_to_backing_carrier():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    mutable_tensor = Tensor(Generic(range(4)), 0, layout)
    immutable_tensor = Tensor(Generic(range(4), mutable=False), 0, layout)

    assert mutable_tensor.is_mutable()
    assert not immutable_tensor.is_mutable()


def test_tensor_version_increments_on_in_place_setitem():
    carrier = Generic([1, 2, 3, 4])
    tensor = Tensor(carrier, 0, Layout(Shape([2, 2]), Stride([1, 2])))

    assert tensor.version == 0
    assert carrier.version == 0

    tensor[1, 0] = 10

    assert tensor.version == 1
    assert carrier.version == 1

    tensor[1, 1] = 20

    assert tensor.version == 2
    assert carrier.version == 2


def test_tensor_version_is_shared_by_views_and_same_data_tensors():
    carrier = Generic(list(range(6)))
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(carrier, 0, layout)
    alias = Tensor(carrier, 0, layout)
    view = tensor[1, :]

    assert tensor.version == 0
    assert alias.version == 0
    assert view.version == 0

    view[0] = 99

    assert tensor.version == 1
    assert alias.version == 1
    assert view.version == 1


def test_tensor_indexes_flat_coordinate_key():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 5, layout)

    assert tensor[2, 3] == carrier[5 + layout.index([2, 3])]


def test_tensor_single_integer_key_uses_layout_expansion():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 5, layout)

    assert tensor[5] == carrier[5 + layout.index(5)]


def test_tensor_indexes_nested_layout_key():
    carrier = Generic(range(400))
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    tensor = Tensor(carrier, 7, layout)

    assert tensor[1, [2, 3]] == carrier[7 + layout.index([1, [2, 3]])]


def test_tensor_accepts_tuple_and_list_coordinate_keys():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)

    assert tensor[1, 2] == carrier[layout.index([1, 2])]
    assert (
        tensor[[1, 2]]  # strideweave-lint: ignore=SW001
        == carrier[layout.index([1, 2])]
    )
    assert tensor[[1, 2]] == tensor[1, 2]  # strideweave-lint: ignore=SW001


def test_tensor_view_with_leaf_slice_shares_data_and_updates_layout_and_offset():
    carrier = Generic(range(64))
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(carrier, 3, layout)

    view = tensor[2, 2:5]

    assert view.carrier is carrier
    assert view.offset == 3 + layout.index([2, 2])
    assert view.layout == Layout(Shape(3), Stride(5))
    assert tensor_values(view) == [tensor[2, j] for j in range(2, 5)]
    assert isinstance(view.autograd_ctx, GenericViewOperation)


def test_tensor_view_can_keep_non_leaf_mode_whole():
    carrier = Generic(range(128))
    layout = Layout(Shape([10, [2, 3]]), Stride([1, [10, 20]]))
    tensor = Tensor(carrier, 0, layout)

    view = tensor[0, :]

    assert view.carrier is carrier
    assert view.offset == tensor.offset
    assert view.layout == Layout(Shape([[2, 3]]), Stride([[10, 20]]))
    assert tensor_values(view) == [tensor[0, j] for j in range(6)]


def test_tensor_view_with_full_slices_preserves_layout_and_offset():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 5, layout)

    view = tensor[:, :]

    assert view.carrier is carrier
    assert view.offset == 5
    assert view.layout == layout
    assert tensor_values(view) == tensor_values(tensor)


def test_tensor_view_requires_slice_for_getitem_dispatch():
    carrier = Generic(range(64))
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(carrier, 0, layout)

    assert tensor[2, 3] == carrier[layout.index([2, 3])]
    assert not isinstance(tensor[2, 3], Tensor)


def test_tensor_view_rejects_missing_and_extra_modes():
    tensor = Tensor(Generic(range(64)), 0, Layout(Shape([5, 10]), Stride([1, 5])))

    with pytest.raises(ValueError, match="exactly one key per top-level mode"):
        tensor[2:5]
    with pytest.raises(ValueError, match="exactly one key per top-level mode"):
        tensor[:, :, :]


def test_tensor_view_rejects_invalid_slices_and_integer_keys():
    nested_tensor = Tensor(
        Generic(range(128)), 0, Layout(Shape([10, [2, 3]]), Stride([1, [10, 20]]))
    )
    flat_tensor = Tensor(Generic(range(64)), 0, Layout(Shape([5, 10]), Stride([1, 5])))

    with pytest.raises(ValueError, match="whole slices are supported for non-leaf"):
        nested_tensor[:, 1:3]
    with pytest.raises(ValueError, match="View slices do not support steps"):
        flat_tensor[:, ::2]
    with pytest.raises(ValueError, match="View integer key is out of domain"):
        flat_tensor[5, :]
    with pytest.raises(ValueError, match="View slices must be non-empty"):
        flat_tensor[:, 3:3]


def test_tensor_list_coordinate_key_matches_tuple_key_for_hierarchical_mode():
    layout = Layout(Shape([2, 3, [2, 2]]), Stride([1, 2, [6, 12]]))
    tensor = Tensor(Generic(range(24)), 0, layout)

    assert (
        tensor[[1, 2, [1, 1]]]  # strideweave-lint: ignore=SW001
        == tensor[1, 2, [1, 1]]
    )


def test_tensor_out_of_domain_keys_raise_layout_errors():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)

    with pytest.raises(ValueError, match="Key is not in domain of shape"):
        tensor[3, 0]
    with pytest.raises(ValueError, match="Key is not in domain of shape"):
        tensor[12]


def test_tensor_rejects_non_integer_scalar_keys():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)
    string_key: Any = "x"
    mixed_key: Any = [1, "x"]

    with pytest.raises(TypeError):
        tensor[string_key]
    with pytest.raises(TypeError):
        tensor[mixed_key]


def test_tensor_setitem_updates_flat_coordinate_key():
    values: list[Any] = list(range(64))
    carrier = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 5, layout)
    carrier_index = 5 + layout.index([2, 3])

    tensor[2, 3] = "updated"

    assert values[carrier_index] == "updated"
    assert tensor[2, 3] == "updated"


def test_tensor_setitem_uses_integer_key_expansion():
    values: list[Any] = list(range(64))
    carrier = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 5, layout)
    carrier_index = 5 + layout.index(5)

    tensor[5] = "updated"

    assert values[carrier_index] == "updated"
    assert tensor[5] == "updated"


def test_tensor_setitem_updates_nested_layout_key():
    values: list[Any] = list(range(400))
    carrier = Generic(values)
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    tensor = Tensor(carrier, 7, layout)
    carrier_index = 7 + layout.index([1, [2, 3]])

    tensor[1, [2, 3]] = "updated"

    assert values[carrier_index] == "updated"
    assert tensor[1, [2, 3]] == "updated"


def test_tensor_setitem_accepts_tuple_and_list_coordinate_keys():
    values: list[Any] = list(range(64))
    carrier = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)

    tensor[1, 2] = "tuple"
    tensor[[2, 3]] = "list"  # strideweave-lint: ignore=SW001
    tensor[[1, 2]] = "list-overwrite"  # strideweave-lint: ignore=SW001

    assert values[layout.index([1, 2])] == "list-overwrite"
    assert tensor[1, 2] == "list-overwrite"
    assert tensor[[1, 2]] == tensor[1, 2]  # strideweave-lint: ignore=SW001
    assert values[layout.index([2, 3])] == "list"


def test_tensor_setitem_list_coordinate_key_matches_tuple_key_for_hierarchical_mode():
    values: list[Any] = list(range(24))
    layout = Layout(Shape([2, 3, [2, 2]]), Stride([1, 2, [6, 12]]))
    tensor = Tensor(Generic(values), 0, layout)

    tensor[[1, 2, [1, 1]]] = "updated"  # strideweave-lint: ignore=SW001

    assert tensor[1, 2, [1, 1]] == "updated"
    assert (
        tensor[[1, 2, [1, 1]]]  # strideweave-lint: ignore=SW001
        == tensor[1, 2, [1, 1]]
    )
    assert values[layout.index([1, 2, [1, 1]])] == "updated"


def test_tensor_setitem_out_of_domain_keys_raise_layout_errors():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)

    with pytest.raises(ValueError, match="Key is not in domain of shape"):
        tensor[3, 0] = "updated"
    with pytest.raises(ValueError, match="Key is not in domain of shape"):
        tensor[12] = "updated"


def test_tensor_setitem_rejects_slice_and_non_integer_keys():
    carrier = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)
    string_key: Any = "x"
    mixed_key: Any = [1, "x"]
    slice_key: Any = slice(1, 2)

    with pytest.raises(TypeError):
        tensor[slice_key] = "updated"
    with pytest.raises(TypeError):
        tensor[string_key] = "updated"
    with pytest.raises(TypeError):
        tensor[mixed_key] = "updated"


def test_tensor_setitem_rejects_immutable_backing_carrier():
    carrier = Generic(range(64), mutable=False)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(carrier, 0, layout)

    with pytest.raises(RuntimeError):
        tensor[1, 2] = "updated"


def test_tensor_add_public_api_imports():
    assert sw.GenericAddOperation is GenericAddOperation
    assert sw.GenericSubOperation is GenericSubOperation
    assert sw.GenericElementwiseMulOperation is GenericElementwiseMulOperation
    assert sw.GenericDivOperation is GenericDivOperation
    assert sw.GenericExpOperation is GenericExpOperation
    assert sw.GenericPowOperation is GenericPowOperation
    assert sw.GenericScalarMulOperation is GenericScalarMulOperation
    assert sw.GenericReduceSumOperation is GenericReduceSumOperation
    assert sw.GenericMatmulOperation is GenericMatmulOperation
    assert sw.RearrangeOperation is RearrangeOperation
    assert sw.PermuteOperation is PermuteOperation
    assert sw.add is not None
    assert sw.elementwise_mul is not None
    assert sw.mul is not None
    assert sw.div is not None
    assert sw.exp is not None
    assert sw.pow is not None
    assert sw.reduce is not None
    assert sw.matmul is not None
    assert sw.rearrange is not None
    assert sw.permute is not None


def test_tensor_add_with_generic_data_returns_autograd_tensor():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = lhs + rhs

    assert tensor_values(result) == [11, 22, 33, 44]
    assert result.layout == layout
    assert result.dtype() is DType.Floating
    assert result.carrier_type() is Generic
    assert isinstance(result.autograd_ctx, GenericAddOperation)
    assert result.autograd_ctx.inputs() == (lhs, rhs)


def test_tensor_add_function_matches_operator():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = sw.add(lhs, rhs)

    assert tensor_values(result) == [11, 22, 33, 44]
    assert isinstance(result.autograd_ctx, GenericAddOperation)


def test_tensor_sub_function_matches_operator_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    rhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result = lhs - rhs

    assert tensor_values(result) == [9, 18, 27, 36]
    assert tensor_values(sw.sub(lhs, rhs)) == [9, 18, 27, 36]
    assert isinstance(result.autograd_ctx, GenericSubOperation)
    assert result.autograd_ctx.inputs() == (lhs, rhs)

    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)
    result.backward(gradient)

    assert lhs.grad is not None and tensor_values(lhs.grad) == [1, 1, 1, 1]
    assert rhs.grad is not None and tensor_values(rhs.grad) == [-1, -1, -1, -1]


def test_tensor_neg_function_matches_operator_and_backpropagates():
    layout = Layout(Shape(2), Stride(1))
    tensor = Tensor(Generic([2, -3]), 0, layout)

    result = -tensor

    assert tensor_values(result) == [-2, 3]
    assert tensor_values(sw.neg(tensor)) == [-2, 3]
    assert isinstance(result.autograd_ctx, GenericScalarMulOperation)

    result.backward(Tensor(Generic([1, 1]), 0, layout))

    assert tensor.grad is not None and tensor_values(tensor.grad) == [-1, -1]


def test_tensor_any_dtype_disables_autograd_interfaces():
    layout = Layout(Shape(2), Stride(1))
    tensor = Tensor(Generic([1, 2], dtype=DType.Any), 0, layout)
    gradient = Tensor(Generic([1, 1]), 0, layout)

    assert tensor.dtype() is DType.Any
    assert not tensor.is_differentiable()
    with pytest.raises(RuntimeError, match="grad is not available"):
        tensor.grad
    with pytest.raises(RuntimeError, match="backward is not available"):
        tensor.backward(gradient)
    with pytest.raises(RuntimeError, match="retain_grad is not available"):
        tensor.retain_grad()
    with pytest.raises(RuntimeError, match="autograd_ctx is not available"):
        tensor.autograd_ctx = object()


def test_tensor_generic_any_operations_do_not_build_autograd_graphs():
    layout = Layout(Shape(2), Stride(1))
    lhs = Tensor(Generic([1, 2], dtype=DType.Any), 0, layout)
    rhs = Tensor(Generic([10, 20], dtype=DType.Any), 0, layout)

    result = lhs + rhs

    assert result.dtype() is DType.Any
    assert tensor_values(result) == [11, 22]
    assert result.autograd_ctx is None


def test_tensor_generic_mixed_any_floating_only_accumulates_floating_grad():
    layout = Layout(Shape(2), Stride(1))
    any_tensor = Tensor(Generic([1, 2], dtype=DType.Any), 0, layout)
    floating_tensor = Tensor(Generic([10, 20]), 0, layout)

    result = any_tensor + floating_tensor
    result.backward(Tensor(Generic([3, 4]), 0, layout))
    floating_grad = require_grad(floating_tensor)

    assert result.dtype() is DType.Floating
    assert isinstance(result.autograd_ctx, GenericAddOperation)
    assert tensor_values(floating_grad) == [3, 4]
    with pytest.raises(RuntimeError, match="grad is not available"):
        any_tensor.grad


def test_tensor_generic_any_non_integer_result_ops_promote_to_floating():
    layout = Layout(Shape(2), Stride(1))
    tensor = Tensor(Generic([2, 4], dtype=DType.Any), 0, layout)
    rhs = Tensor(Generic([4, 2], dtype=DType.Any), 0, layout)

    div_result = tensor / rhs
    exp_result = sw.exp(tensor)
    sigmoid_result = sw.sigmoid(tensor)
    pow_result = tensor**-1

    assert div_result.dtype() is DType.Floating
    assert exp_result.dtype() is DType.Floating
    assert sigmoid_result.dtype() is DType.Floating
    assert pow_result.dtype() is DType.Floating
    assert div_result.autograd_ctx is None
    assert exp_result.autograd_ctx is None
    assert sigmoid_result.autograd_ctx is None
    assert pow_result.autograd_ctx is None


def test_tensor_add_preserves_generic_data_class():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = lhs + rhs

    assert tensor_values(result) == [11, 22, 33, 44]
    assert type(result.carrier) is Generic


def test_tensor_add_rejects_mismatched_layouts():
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([1, 2])))
    rhs = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([2, 1])))

    with pytest.raises(ValueError, match="Tensor layouts must match"):
        _ = lhs + rhs


def test_tensor_add_rejects_non_tensor_operand():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    with pytest.raises(TypeError):
        _ = lhs + 1


def test_tensor_add_rejects_mismatched_data_classes():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(CPU(4), 0, layout)

    with pytest.raises(TypeError):
        _ = lhs + rhs


def test_tensor_add_rejects_unsupported_data_class():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(UnsupportedData([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(UnsupportedData([1, 2, 3, 4]), 0, layout)

    with pytest.raises(NotImplementedError):
        _ = lhs + rhs


def test_tensor_scalar_mul_accepts_any_layout_and_preserves_layout():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    tensor = Tensor(Generic(list(range(10))), 0, layout)

    left_result = tensor * 2
    right_result = 3 * tensor
    function_result = sw.mul(tensor, 4)

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
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_scalar_mul_rejects_non_numeric_scalar():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    with pytest.raises(TypeError):
        _ = tensor * "x"


def test_tensor_elementwise_mul_forward_and_backward():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([5, 6, 7, 8]), 0, layout)
    gradient = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    result = lhs * rhs
    function_result = sw.elementwise_mul(lhs, rhs)
    result.backward(gradient)

    assert tensor_values(result) == [5, 12, 21, 32]
    assert tensor_values(function_result) == [5, 12, 21, 32]
    assert isinstance(result.autograd_ctx, GenericElementwiseMulOperation)
    assert tensor_values(require_grad(lhs)) == [50, 120, 210, 320]
    assert tensor_values(require_grad(rhs)) == [10, 40, 90, 160]


def test_tensor_div_forward_and_backward():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([8, 9, 10, 12]), 0, layout)
    rhs = Tensor(Generic([2, 3, 5, 4]), 0, layout)
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result = lhs / rhs
    function_result = sw.div(lhs, rhs)
    result.backward(gradient)

    assert tensor_values(result) == pytest.approx([4, 3, 2, 3])
    assert tensor_values(function_result) == pytest.approx([4, 3, 2, 3])
    assert isinstance(result.autograd_ctx, GenericDivOperation)
    assert tensor_values(require_grad(lhs)) == pytest.approx([0.5, 2 / 3, 0.6, 1])
    assert tensor_values(require_grad(rhs)) == pytest.approx([-2, -2, -1.2, -3])


def test_tensor_exp_forward_and_backward():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([0, 1, 2, 3]), 0, layout)
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result = sw.exp(tensor)
    result.backward(gradient)

    expected = [math.exp(value) for value in [0, 1, 2, 3]]
    assert tensor_values(result) == pytest.approx(expected)
    assert isinstance(result.autograd_ctx, GenericExpOperation)
    assert tensor_values(require_grad(tensor)) == pytest.approx(
        [grad * value for grad, value in zip([1, 2, 3, 4], expected, strict=True)]
    )


def test_tensor_pow_scalar_forward_and_backward():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result = tensor**3
    function_result = sw.pow(tensor, 3)
    result.backward(gradient)

    assert tensor_values(result) == [1, 8, 27, 64]
    assert tensor_values(function_result) == [1, 8, 27, 64]
    assert isinstance(result.autograd_ctx, GenericPowOperation)
    assert tensor_values(require_grad(tensor)) == [3, 24, 81, 192]


def test_tensor_elementwise_operations_reject_invalid_inputs():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    mismatched_layout = Tensor(
        Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([2, 1]))
    )

    with pytest.raises(ValueError, match="Tensor layouts must match"):
        _ = tensor * mismatched_layout
    with pytest.raises(TypeError):
        _ = tensor / 2
    with pytest.raises(TypeError):
        _ = tensor ** "x"


def test_tensor_reduce_sums_second_mode():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, layout)

    result = sw.reduce(tensor)

    assert result.layout == Layout(Shape(2), Stride(1))
    assert tensor_values(result) == [9, 12]
    assert isinstance(result.autograd_ctx, GenericReduceSumOperation)


def test_tensor_reduce_preserves_hierarchical_first_mode_with_column_major_layout():
    layout = Layout(Shape([[2, 2], 3]), Stride([[1, 2], 4]))
    tensor = Tensor(Generic(range(1, 13)), 0, layout)

    result = sw.reduce(tensor)

    assert result.layout == Layout(Shape([2, 2]), Stride([1, 2]))
    assert tensor_values(result) == [15, 18, 21, 24]


def test_tensor_reduce_backward_copies_gradient_over_second_mode():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, layout)
    result = sw.reduce(tensor)
    gradient = Tensor(Generic([10, 20]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [10, 20, 10, 20, 10, 20]
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_reduce_rejects_non_two_mode_tensor():
    one_mode = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))

    with pytest.raises(ValueError, match="tensor must have a two-mode layout"):
        sw.reduce(one_mode)


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

    result = sw.matmul(a, b)

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
    assert type(a_grad.carrier) is type(a.carrier)
    assert type(b_grad.carrier) is type(b.carrier)


def test_tensor_matmul_rejects_invalid_shapes_and_carrier():
    a = Tensor(Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([1, 2])))
    bad_k = Tensor(Generic([1, 2, 3]), 0, Layout(Shape([1, 3]), Stride([1, 1])))
    one_mode = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
    cpu_tensor = Tensor(CPU(4), 0, a.layout)

    with pytest.raises(ValueError, match="Matmul inner dimensions must match"):
        _ = a @ bad_k

    with pytest.raises(ValueError, match="rhs must have a two-mode layout"):
        _ = a @ one_mode

    with pytest.raises(TypeError):
        _ = cpu_tensor @ a


def test_tensor_rearrange_forward_returns_view_with_rearranged_layout():
    carrier = Generic([1, 2, 3, 4, 5, 6])
    tensor = Tensor(carrier, 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = sw.rearrange(tensor, Tree(Node.id(1), Node.id(0)))

    assert result.carrier is carrier
    assert result.offset == tensor.offset
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, RearrangeOperation)
    assert result.autograd_ctx.inputs() == (tensor,)


def test_tensor_rearrange_forward_accepts_explicit_selection():
    tensor = Tensor(
        Generic(range(36)),
        0,
        Layout(Shape([1, [2, 3]]), Stride([5, [7, 14]])),
    )

    result = sw.rearrange(
        tensor,
        Tree(Node.id(1), Node.id(0)),
        Tree(1, 1),
    )

    assert result.layout == Layout(Shape([[2, 3], 1]), Stride([[7, 14], 5]))


def test_tensor_rearrange_forward_allows_omitted_singleton_ids():
    tensor = Tensor(
        Generic(range(6)),
        0,
        Layout(Shape([2, 1, 3]), Stride([1, 99, 2])),
    )

    result = sw.rearrange(tensor, Tree(Node.id(2), Node.id(0)))

    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 0, 2]


def test_tensor_rearrange_backward_inverts_permutation():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    result = sw.rearrange(tensor, Tree(Node.id(1), Node.id(0)))
    gradient = Tensor(Generic([10, 40, 20, 50, 30, 60]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_grad.layout == tensor.layout
    assert tensor_values(tensor_grad) == [10, 40, 20, 50, 30, 60]
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_rearrange_backward_preserves_original_singleton_strides():
    layout = Layout(Shape([2, 1, 3]), Stride([1, 99, 2]))
    tensor = Tensor(Generic(range(6)), 0, layout)
    result = sw.rearrange(tensor, Tree(Node.id(2), Node.id(0)))
    gradient = Tensor(Generic([10, 40, 20, 50, 30, 60]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_grad.layout == layout
    assert tensor_values(tensor_grad) == [10, 40, 20, 50, 30, 60]


def test_tensor_rearrange_rejects_invalid_inputs():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    invalid_output: Any = object()
    invalid_selection: Any = "selection"

    with pytest.raises(TypeError):
        sw.rearrange(tensor, invalid_output)
    with pytest.raises(
        ValueError, match="Rearrange command must contain one '->' arrow"
    ):
        sw.rearrange(tensor, "output")
    with pytest.raises(TypeError):
        sw.rearrange(tensor, Tree(Node.id(0), Node.id(1)), invalid_selection)
    with pytest.raises(ValueError, match="must include every extracted layout"):
        sw.rearrange(tensor, Tree(Node.id(0)))


def test_tensor_permute_forward_returns_view_with_permuted_layout():
    carrier = Generic([1, 2, 3, 4, 5, 6])
    tensor = Tensor(carrier, 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = sw.permute(tensor, 1, 0)

    assert result.carrier is carrier
    assert result.offset == tensor.offset
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, PermuteOperation)
    assert result.autograd_ctx.inputs() == (tensor,)


def test_tensor_permute_accepts_tuple_and_list_orders():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    tuple_result = sw.permute(tensor, (1, 0))
    list_result = sw.permute(tensor, [1, 0])

    assert tuple_result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert list_result.layout == tuple_result.layout


def test_tensor_permute_preserves_hierarchical_modes():
    tensor = Tensor(
        Generic(range(120)),
        0,
        Layout(Shape([[2, 3], 4, 5]), Stride([[1, 2], 6, 24])),
    )
    nested_key = [1, 2]

    result = sw.permute(tensor, 1, 0, 2)

    assert result.layout == Layout(Shape([4, [2, 3], 5]), Stride([6, [1, 2], 24]))
    assert result[3, nested_key, 4] == tensor[nested_key, 3, 4]


def test_tensor_permute_backward_inverts_permutation():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    result = sw.permute(tensor, 1, 0)
    gradient = Tensor(Generic([10, 40, 20, 50, 30, 60]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_grad.layout == tensor.layout
    assert tensor_values(tensor_grad) == [10, 40, 20, 50, 30, 60]
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_permute_backward_accumulates_repeated_calls():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    result = sw.permute(tensor, 1, 0)
    gradient = Tensor(Generic([1, 4, 2, 5, 3, 6]), 0, result.layout)

    result.backward(gradient)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [2, 8, 4, 10, 6, 12]


def test_tensor_permute_rejects_invalid_orders():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    non_integer_dim: Any = "0"

    with pytest.raises(ValueError, match="must reorder every layout mode"):
        sw.permute(tensor, 0, 0)
    with pytest.raises(ValueError, match="must reorder every layout mode"):
        sw.permute(tensor, 0)
    with pytest.raises(ValueError, match="must reorder every layout mode"):
        sw.permute(tensor, -1, 0)
    with pytest.raises(ValueError, match="must reorder every layout mode"):
        sw.permute(tensor, 0, 2)
    with pytest.raises(TypeError):
        sw.permute(tensor, non_integer_dim, 1)


def test_tensor_backward_on_leaf_creates_detached_generic_grad():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    gradient = Tensor(Generic([10, 20, 30, 40]), 0, layout)

    tensor.backward(gradient)
    gradient[0] = 999
    tensor_grad = require_grad(tensor)

    assert tensor.grad is not gradient
    assert type(tensor_grad.carrier) is type(tensor.carrier)
    assert tensor_values(tensor_grad) == [10, 20, 30, 40]


def test_tensor_backward_without_gradient_on_scalar_leaf_creates_unit_grad():
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(Generic([7]), 0, layout)

    tensor.backward()
    tensor_grad = require_grad(tensor)

    assert tensor_grad.layout == layout
    assert tensor_values(tensor_grad) == [1]
    assert type(tensor_grad.carrier) is Generic


def test_tensor_backward_without_gradient_through_scalar_operation_sets_input_grad():
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(Generic([3]), 0, layout)
    result = tensor * 2

    result.backward()
    tensor_grad = require_grad(tensor)

    assert result.grad is None
    assert tensor_values(tensor_grad) == [2]
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_backward_without_gradient_on_scalar_accumulates_repeated_calls():
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(Generic([3]), 0, layout)
    result = tensor * 2

    result.backward()
    result.backward()
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [4]


def test_tensor_backward_without_gradient_retains_non_leaf_scalar_grad():
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(Generic([3]), 0, layout)
    result = tensor * 2

    result.retain_grad()
    result.backward()

    assert tensor_values(require_grad(result)) == [1]
    assert tensor_values(require_grad(tensor)) == [2]


def test_tensor_backward_explicit_scalar_gradient_overrides_implicit_gradient():
    layout = Layout(Shape(1), Stride(1))
    tensor = Tensor(Generic([3]), 0, layout)
    result = tensor * 2
    gradient = Tensor(Generic([5]), 0, layout)

    result.retain_grad()
    result.backward(gradient)

    assert tensor_values(require_grad(result)) == [5]
    assert tensor_values(require_grad(tensor)) == [10]


def test_tensor_backward_without_gradient_rejects_non_scalar_tensor():
    logical_size_one_layout = Layout(Shape([1, 1]), Stride([1, 1]))
    logical_size_two_layout = Layout(Shape(2), Stride(1))
    logical_size_one_tensor = Tensor(Generic([1]), 0, logical_size_one_layout)
    logical_size_two_tensor = Tensor(Generic([1, 2]), 0, logical_size_two_layout)

    with pytest.raises(
        ValueError, match=r"Tensor.backward requires a gradient for non-scalar tensors"
    ):
        logical_size_one_tensor.backward()
    with pytest.raises(
        ValueError, match=r"Tensor.backward requires a gradient for non-scalar tensors"
    ):
        logical_size_two_tensor.backward()


def test_tensor_backward_without_gradient_on_cpu_scalar_uses_cpu_grad():
    layout = Layout(Shape(1), Stride(1))
    carrier = CPU(1)
    carrier[0] = 7.0
    tensor = Tensor(carrier, 0, layout)

    tensor.backward()
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == pytest.approx([1.0])
    assert type(tensor_grad.carrier) is CPU


def test_tensor_backward_through_add_sets_input_grads():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert result.grad is None
    assert tensor_values(lhs_grad) == [1, 2, 3, 4]
    assert tensor_values(rhs_grad) == [1, 2, 3, 4]
    assert type(lhs_grad.carrier) is type(lhs.carrier)
    assert type(rhs_grad.carrier) is type(rhs.carrier)


def test_tensor_retain_grad_keeps_non_leaf_gradient():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.retain_grad()
    result.backward(gradient)
    result_grad = require_grad(result)

    assert tensor_values(result_grad) == [1, 2, 3, 4]


def test_tensor_retain_grad_false_disables_non_leaf_gradient_retention():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.retain_grad()
    result.retain_grad(False)
    result.backward(gradient)

    assert result.grad is None
    assert tensor_values(require_grad(lhs)) == [1, 2, 3, 4]
    assert tensor_values(require_grad(rhs)) == [1, 2, 3, 4]


def test_tensor_backward_on_no_grad_result_does_not_propagate_to_inputs():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    with sw.no_grad():
        result = lhs + rhs

    assert result.autograd_ctx is None

    result.backward(gradient)
    result_grad = require_grad(result)

    assert tensor_values(result_grad) == [1, 2, 3, 4]
    assert lhs.grad is None
    assert rhs.grad is None


def test_tensor_backward_accumulates_repeated_calls():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    result.backward(gradient)
    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert result.grad is None
    assert tensor_values(lhs_grad) == [2, 2, 2, 2]
    assert tensor_values(rhs_grad) == [2, 2, 2, 2]


def test_tensor_retained_non_leaf_grad_accumulates_repeated_calls():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    rhs = Tensor(Generic([10, 20, 30, 40]), 0, layout)
    result = lhs + rhs
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    result.retain_grad()
    result.backward(gradient)
    result.backward(gradient)
    result_grad = require_grad(result)

    assert tensor_values(result_grad) == [2, 2, 2, 2]


def test_tensor_backward_only_retains_selected_non_leaf_grads_in_chain():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    x = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    y = x * 2
    z = y * 3
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    z.backward(gradient)

    assert z.grad is None
    assert y.grad is None
    assert tensor_values(require_grad(x)) == [6, 6, 6, 6]


def test_tensor_backward_retains_requested_non_leaf_grads_in_chain():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    x = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    y = x * 2
    z = y * 3
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    y.retain_grad()
    z.retain_grad()
    z.backward(gradient)

    assert tensor_values(require_grad(z)) == [1, 1, 1, 1]
    assert tensor_values(require_grad(y)) == [3, 3, 3, 3]
    assert tensor_values(require_grad(x)) == [6, 6, 6, 6]


def test_tensor_backward_accumulates_shared_input_contributions():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    result = tensor + tensor
    gradient = Tensor(Generic([1, 2, 3, 4]), 0, layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(tensor_grad) == [2, 4, 6, 8]


def test_tensor_backward_rejects_input_modified_in_place_after_forward():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    result = tensor * tensor
    gradient = Tensor(Generic([1, 1, 1, 1]), 0, layout)

    tensor[0, 0] = 10

    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(gradient)


def test_tensor_backward_rejects_view_input_modified_through_source_after_forward():
    layout = Layout(Shape([2, 3]), Stride([1, 2]))
    tensor = Tensor(Generic(list(range(6))), 0, layout)
    view = tensor[1, :]
    result = view * view
    gradient = tensor_with_logical_values([1, 1, 1], view.layout)

    tensor[1, 0] = 99

    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(gradient)


def test_tensor_view_backward_scatters_gradient_into_source_layout():
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(Generic(range(50)), 0, layout)
    view = tensor[2, 2:5]
    gradient = tensor_with_logical_values([10, 20, 30], view.layout)

    view.backward(gradient)
    tensor_grad = require_grad(tensor)

    expected = [0] * tensor.size()
    for value, j in zip([10, 20, 30], range(2, 5), strict=True):
        expected[layout.index([2, j])] = value
    assert view.grad is None
    assert tensor_values(tensor_grad) == expected
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_tensor_view_backward_handles_non_leaf_whole_mode():
    layout = Layout(Shape([10, [2, 3]]), Stride([1, [10, 20]]))
    tensor = Tensor(Generic(range(128)), 0, layout)
    view = tensor[0, :]
    gradient = tensor_with_logical_values([1, 2, 3, 4, 5, 6], view.layout)

    view.backward(gradient)
    tensor_grad = require_grad(tensor)

    expected = [0] * tensor.size()
    for j, value in enumerate([1, 2, 3, 4, 5, 6]):
        expected[layout.index([0, j])] = value
    assert tensor_values(tensor_grad) == expected


def test_tensor_view_created_under_no_grad_does_not_propagate():
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(Generic(range(50)), 0, layout)

    with sw.no_grad():
        view = tensor[2, 2:5]

    gradient = tensor_with_logical_values([10, 20, 30], view.layout)
    view.backward(gradient)

    assert view.autograd_ctx is None
    assert tensor_values(require_grad(view)) == [10, 20, 30]
    assert tensor.grad is None


def test_tensor_backward_rejects_invalid_gradient():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(Generic([1, 2, 3, 4]), 0, layout)
    wrong_layout_gradient = Tensor(
        Generic([1, 2, 3, 4]), 0, Layout(Shape([2, 2]), Stride([2, 1]))
    )
    invalid_gradient: Any = "gradient"

    with pytest.raises(TypeError):
        tensor.backward(invalid_gradient)

    with pytest.raises(ValueError, match="gradient layout must match tensor layout"):
        tensor.backward(wrong_layout_gradient)


def test_tensor_rejects_negative_offset():
    carrier = Generic(range(4))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))

    with pytest.raises(ValueError, match="Tensor offset must be non-negative"):
        Tensor(carrier, -1, layout)


def test_tensor_rejects_storage_that_exceeds_data_size():
    carrier = Generic(range(4))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))

    with pytest.raises(ValueError, match="Tensor storage exceeds carrier size"):
        Tensor(carrier, 1, layout)


def test_tensor_storage_validation_uses_cosize_not_logical_size():
    layout = Layout(Shape([2, 2]), Stride([1, 10]))

    with pytest.raises(ValueError, match="Tensor storage exceeds carrier size"):
        Tensor(Generic(range(11)), 0, layout)

    tensor = Tensor(Generic(range(12)), 0, layout)
    assert tensor[1, 1] == 11


def test_tensor_propagates_released_backing_data_errors():
    carrier = Generic(range(16))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(carrier, 3, layout)

    assert tensor[1, 1] == 6

    carrier.release()
    with pytest.raises(RuntimeError):
        tensor[1, 1]


def test_tensor_setitem_propagates_released_backing_data_errors():
    carrier = Generic(range(16))
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(carrier, 3, layout)

    tensor[1, 1] = "updated"
    assert tensor[1, 1] == "updated"

    carrier.release()
    with pytest.raises(RuntimeError):
        tensor[1, 1] = "released"


def test_tensor_rejects_negative_index_keys():
    tensor = Tensor(
        Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2]))
    )

    with pytest.raises(ValueError, match="not in domain"):
        tensor[-1, 2]
    with pytest.raises(ValueError, match="not in domain"):
        tensor[-1]
    with pytest.raises(ValueError, match="not in domain"):
        tensor[[0, -1]]  # strideweave-lint: ignore=SW001
    with pytest.raises(ValueError, match="not in domain"):
        tensor[-1, 0] = 9.0


def test_tensor_backward_deep_graph_does_not_exhaust_recursion():
    leaf = Tensor(Generic([1.0]), 0, Layout(Shape(1), Stride(1)))

    output = leaf
    for _ in range(3000):
        output = sw.mul(output, 1.0)
    output.backward()

    assert leaf.grad is not None
    assert leaf.grad[0] == 1.0


def test_tensor_backward_shared_subgraph_runs_each_operation_once():
    # A doubling chain of depth 60 re-traverses 2**60 paths under naive
    # per-path backward propagation; topological propagation visits each
    # operation once and finishes immediately.
    leaf = Tensor(Generic([1.0]), 0, Layout(Shape(1), Stride(1)))

    output = leaf
    for _ in range(60):
        output = sw.add(output, output)
    output.backward()

    assert leaf.grad is not None
    assert leaf.grad[0] == 2.0**60


def test_tensor_backward_diamond_graph_accumulates_gradients():
    layout = Layout(Shape(2), Stride(1))
    leaf = Tensor(Generic([1.0, 2.0]), 0, layout)

    doubled = sw.mul(leaf, 2.0)
    tripled = sw.mul(leaf, 3.0)
    combined = sw.add(doubled, tripled)
    combined.backward(Tensor(Generic([1.0, 1.0]), 0, layout))

    assert leaf.grad is not None
    assert [leaf.grad[0], leaf.grad[1]] == [5.0, 5.0]


def test_tensor_backward_retains_summed_gradient_on_interior_tensor():
    layout = Layout(Shape(1), Stride(1))
    leaf = Tensor(Generic([1.0]), 0, layout)

    interior = sw.mul(leaf, 2.0)
    interior.retain_grad()
    combined = sw.add(interior, interior)
    combined.backward()

    assert interior.grad is not None
    assert interior.grad[0] == 2.0
    assert leaf.grad is not None
    assert leaf.grad[0] == 4.0
