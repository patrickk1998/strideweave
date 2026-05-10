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
