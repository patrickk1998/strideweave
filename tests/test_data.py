from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol, cast

import neotorch
import pytest
from neotorch import Data, DataType


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
