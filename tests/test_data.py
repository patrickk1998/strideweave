from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol, cast

import neotorch
import pytest
from neotorch import Data, DataType, Generic


class NativeDataModule(Protocol):
    _VectorDataForTest: Callable[[list[Any]], Data]


native_data = cast(NativeDataModule, import_module("neotorch._data"))


class PythonData(Data):
    def __init__(self, values: list[Any]):
        super().__init__()
        self.values = values

    def size(self) -> int:
        return len(self.values)

    def type(self) -> DataType:
        return DataType.Any

    def get_value(self, index: int) -> Any:
        return self.values[index]


def test_data_public_api_imports():
    assert neotorch.Data is Data
    assert neotorch.DataType is DataType
    assert neotorch.Generic is Generic
    assert DataType.Any.value == "Any"


def test_python_data_subclass_implements_data_contract():
    data = PythonData(["alpha", 2, None])

    assert isinstance(data, Data)
    assert data.size() == 3
    assert data.type() is DataType.Any
    assert data.get_value(1) == 2
    assert data[0] == "alpha"
    assert data[2] is None


def test_python_data_subclass_getitem_validates_index():
    data = PythonData(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        data[-1]

    with pytest.raises(IndexError):
        data[1]

    with pytest.raises(TypeError):
        data[non_int_index]


def test_native_data_subclass_implements_data_contract():
    data = native_data._VectorDataForTest(["alpha", 2, None])

    assert isinstance(data, Data)
    assert data.size() == 3
    assert data.type() is DataType.Any
    assert data.get_value(1) == 2
    assert data[0] == "alpha"
    assert data[2] is None


def test_native_data_subclass_getitem_validates_index():
    data = native_data._VectorDataForTest(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        data[-1]

    with pytest.raises(IndexError):
        data[1]

    with pytest.raises(TypeError):
        data[non_int_index]


def test_generic_data_wraps_indexable_iterable():
    values = ["alpha", 2, None]
    data = Generic(values)

    assert isinstance(data, Data)
    assert data.size() == 3
    assert data.type() is DataType.Any
    assert data.get_value(1) == 2
    assert data[0] == "alpha"
    assert data[2] is None

    values[1] = "updated"
    assert data[1] == "updated"


def test_generic_data_wraps_tuple_and_range():
    tuple_data = Generic(("alpha", "beta"))
    range_data = Generic(range(3))

    assert tuple_data.size() == 2
    assert tuple_data[1] == "beta"
    assert range_data.size() == 3
    assert range_data[2] == 2


def test_generic_data_materializes_one_pass_iterable():
    iterator = iter(["alpha", "beta", "gamma"])
    data = Generic(iterator)

    assert data.size() == 3
    assert data[0] == "alpha"
    assert data[2] == "gamma"
    assert list(iterator) == []


def test_generic_data_requires_iterable_input():
    non_iterable: Any = 1

    with pytest.raises(TypeError):
        Generic(non_iterable)


def test_generic_data_getitem_validates_index():
    data = Generic(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        data[-1]

    with pytest.raises(IndexError):
        data[1]

    with pytest.raises(TypeError):
        data[non_int_index]
