from typing import Any

import neotorch
import pytest
from neotorch import DataType, Generic, GenericEvictable, Layout, Shape, Stride
from neotorch.tensor import Tensor


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

    with pytest.raises(AttributeError):
        setattr(tensor, "data", data)
    with pytest.raises(AttributeError):
        setattr(tensor, "offset", 1)
    with pytest.raises(AttributeError):
        setattr(tensor, "layout", layout)


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

    assert tensor[[2, 3]] == data[5 + layout.index([2, 3])]


def test_tensor_single_integer_key_uses_layout_expansion():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 5, layout)

    assert tensor[5] == data[5 + layout.index(5)]


def test_tensor_indexes_nested_layout_key():
    data = Generic(range(400))
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))
    tensor = Tensor(data, 7, layout)

    assert tensor[[1, [2, 3]]] == data[7 + layout.index([1, [2, 3]])]


def test_tensor_accepts_tuple_and_list_coordinate_keys():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    assert tensor[1, 2] == data[layout.index([1, 2])]
    assert tensor[[1, 2]] == data[layout.index([1, 2])]


def test_tensor_out_of_domain_keys_raise_layout_errors():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    with pytest.raises(ValueError):
        tensor[[3, 0]]
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

    tensor[[2, 3]] = "updated"

    assert values[data_index] == "updated"
    assert tensor[[2, 3]] == "updated"


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

    tensor[[1, [2, 3]]] = "updated"

    assert values[data_index] == "updated"
    assert tensor[[1, [2, 3]]] == "updated"


def test_tensor_setitem_accepts_tuple_and_list_coordinate_keys():
    values: list[Any] = list(range(64))
    data = Generic(values)
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    tensor[1, 2] = "tuple"
    tensor[[2, 3]] = "list"

    assert values[layout.index([1, 2])] == "tuple"
    assert values[layout.index([2, 3])] == "list"


def test_tensor_setitem_out_of_domain_keys_raise_layout_errors():
    data = Generic(range(64))
    layout = Layout(Shape([3, 4]), Stride([2, 10]))
    tensor = Tensor(data, 0, layout)

    with pytest.raises(ValueError):
        tensor[[3, 0]] = "updated"
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
        tensor[[1, 2]] = "updated"


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
    assert tensor[[1, 1]] == 11


def test_tensor_propagates_backing_data_lifecycle_errors(tmp_path):
    path = tmp_path / "data.pkl"
    data = GenericEvictable(range(16), path)
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 3, layout)

    assert tensor[[1, 1]] == 6

    data.evict()
    with pytest.raises(RuntimeError):
        tensor[[1, 1]]

    data.promote()
    assert tensor[[1, 1]] == 6


def test_tensor_setitem_propagates_backing_data_lifecycle_errors(tmp_path):
    path = tmp_path / "data.pkl"
    data = GenericEvictable(range(16), path)
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 3, layout)

    tensor[[1, 1]] = "updated"
    assert tensor[[1, 1]] == "updated"

    data.evict()
    with pytest.raises(RuntimeError):
        tensor[[1, 1]] = "evicted"

    data.promote()
    assert tensor[[1, 1]] == "updated"
