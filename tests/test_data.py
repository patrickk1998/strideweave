from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any, Protocol, cast

import neotorch
import pytest
from neotorch import (
    CPU,
    Data,
    DataType,
    Generic,
    Layout,
    Shape,
    Stride,
)
from neotorch.tensor import Tensor


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

    def new_like(self, values: Iterable[Any], *, mutable: bool = True) -> "PythonData":
        return type(self)(list(values))

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("PythonData does not implement scatter")


class PythonMutableData(PythonData):
    def is_mutable(self) -> bool:
        return True

    def set_value(self, index: int, value: Any) -> None:
        self.values[index] = value


def test_data_public_api_imports():
    assert neotorch.Data is Data
    assert neotorch.CPU is CPU
    assert neotorch.DataType is DataType
    assert neotorch.Generic is Generic
    assert DataType.Any.value == "Any"
    assert DataType.Floating.value == "Floating"
    assert DataType.Float32.value == "Float32"
    assert DataType.Int32.value == "Int32"


def test_data_default_dispatch_op_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Data.dispatch_op("add")


def test_generic_data_dispatch_op_returns_supported_operations():
    cases = {
        "add": neotorch.GenericAddOperation,
        "div": neotorch.GenericDivOperation,
        "elu": neotorch.GenericELUOperation,
        "elementwise_mul": neotorch.GenericElementwiseMulOperation,
        "exp": neotorch.GenericExpOperation,
        "gelu": neotorch.GenericGELUOperation,
        "leaky_relu": neotorch.GenericLeakyReLUOperation,
        "matmul": neotorch.GenericMatmulOperation,
        "mul": neotorch.GenericScalarMulOperation,
        "permute": neotorch.PermuteOperation,
        "pow": neotorch.GenericPowOperation,
        "rearrange": neotorch.RearrangeOperation,
        "reduce": neotorch.GenericReduceSumOperation,
        "relu": neotorch.GenericReLUOperation,
        "sigmoid": neotorch.GenericSigmoidOperation,
        "silu": neotorch.GenericSiLUOperation,
        "softplus": neotorch.GenericSoftplusOperation,
        "tanh": neotorch.GenericTanhOperation,
        "view": neotorch.GenericViewOperation,
    }

    for operation_name, operation_type in cases.items():
        assert isinstance(Generic.dispatch_op(operation_name), operation_type)


def test_generic_data_dispatch_op_rejects_unknown_operation():
    with pytest.raises(NotImplementedError):
        Generic.dispatch_op("unknown")


def test_python_data_subclass_implements_data_contract():
    data = PythonData(["alpha", 2, None])

    assert isinstance(data, Data)
    assert not data.is_mutable()
    assert data.size() == 3
    assert data.type() is DataType.Any
    assert data.get_value(1) == 2
    assert data[0] == "alpha"
    assert data[2] is None


def test_python_data_subclass_new_like_creates_matching_data_class():
    data = PythonData(["alpha"])

    new_data = data.new_like(["beta", "gamma"])

    assert type(new_data) is PythonData
    assert new_data.size() == 2
    assert new_data[0] == "beta"


def test_native_data_subclass_new_like_creates_matching_data_class():
    data = native_data._VectorDataForTest(["alpha"])

    new_data = data.new_like(["beta", "gamma"])

    assert type(new_data) is type(data)
    assert new_data.size() == 2
    assert new_data[0] == "beta"


def test_python_data_subclass_getitem_validates_index():
    data = PythonData(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        data[-1]

    with pytest.raises(IndexError):
        data[1]

    with pytest.raises(TypeError):
        data[non_int_index]


def test_python_data_subclass_rejects_setitem_by_default():
    data = PythonData(["alpha"])

    with pytest.raises(RuntimeError):
        data[0] = "updated"

    assert data[0] == "alpha"


def test_python_mutable_data_subclass_setitem_updates_values():
    data = PythonMutableData(["alpha", "beta"])
    non_int_index: Any = "0"

    assert data.is_mutable()

    data[1] = "updated"

    assert data.get_value(1) == "updated"
    assert data[1] == "updated"

    with pytest.raises(IndexError):
        data[-1] = "negative"

    with pytest.raises(IndexError):
        data[2] = "large"

    with pytest.raises(TypeError):
        data[non_int_index] = "non-int"


def test_native_data_subclass_implements_data_contract():
    data = native_data._VectorDataForTest(["alpha", 2, None])

    assert isinstance(data, Data)
    assert not data.is_mutable()
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


def test_native_data_subclass_rejects_setitem_by_default():
    data = native_data._VectorDataForTest(["alpha"])

    with pytest.raises(RuntimeError):
        data[0] = "updated"

    assert data[0] == "alpha"


def test_generic_data_wraps_indexable_iterable():
    values = ["alpha", 2, None]
    data = Generic(values, dtype=DataType.Any)

    assert isinstance(data, Data)
    assert data.is_mutable()
    assert data.size() == 3
    assert data.type() is DataType.Any
    assert data.get_value(1) == 2
    assert data[0] == "alpha"
    assert data[2] is None

    values[1] = "updated"
    assert data[1] == "updated"


def test_generic_data_defaults_to_floating_dtype():
    data = Generic([1, 2, 3])

    assert data.type() is DataType.Floating


def test_generic_data_new_like_preserves_or_overrides_dtype():
    data = Generic([1, 2, 3])

    preserved = data.new_like([4, 5])
    overridden = data.new_like(["alpha"], dtype=DataType.Any)

    assert preserved.type() is DataType.Floating
    assert overridden.type() is DataType.Any


def test_generic_data_rejects_invalid_dtype():
    with pytest.raises(ValueError):
        Generic([1], dtype=DataType.Int32)

    with pytest.raises(TypeError):
        Generic([1], dtype="Floating")  # type: ignore[arg-type]


def test_generic_data_mutates_backing_list_by_default():
    values = ["alpha", "beta"]
    data = Generic(values)

    data[1] = "updated"

    assert values[1] == "updated"
    assert data[1] == "updated"


def test_generic_data_can_be_immutable():
    values = ["alpha", "beta"]
    data = Generic(values, mutable=False)

    assert not data.is_mutable()

    with pytest.raises(RuntimeError):
        data[1] = "updated"

    assert values[1] == "beta"
    assert data[1] == "beta"


def test_generic_data_materializes_non_settable_inputs_when_mutable():
    tuple_data = Generic(("alpha", "beta"))
    range_data = Generic(range(3))
    iterator = iter(["left", "right"])
    iterator_data = Generic(iterator)

    tuple_data[1] = "updated"
    range_data[2] = "updated"
    iterator_data[0] = "updated"

    assert tuple_data[1] == "updated"
    assert range_data[2] == "updated"
    assert iterator_data[0] == "updated"
    assert list(iterator) == []


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


def test_generic_data_setitem_validates_index():
    data = Generic(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        data[-1] = "negative"

    with pytest.raises(IndexError):
        data[1] = "large"

    with pytest.raises(TypeError):
        data[non_int_index] = "non-int"


def test_generic_data_scatter_maps_source_values_into_destination_storage():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    destination_values = [0] * 50
    destination_data = Generic(destination_values)
    destination = Tensor(destination_data, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(3), Stride(5))

    destination_data.scatter(source, destination, mapping, 12)

    assert destination_values[12] == 10
    assert destination_values[17] == 20
    assert destination_values[22] == 30
    assert sum(destination_values) == 60


def test_generic_data_scatter_validates_mapping_shape():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    destination_data = Generic([0] * 50)
    destination = Tensor(destination_data, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(4), Stride(5))

    with pytest.raises(ValueError):
        destination_data.scatter(source, destination, mapping, 12)


def test_generic_data_scatter_uses_destination_mutability():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    mapping = Layout(Shape(3), Stride(5))
    immutable_data = Generic([0] * 50, mutable=False)
    immutable_destination = Tensor(
        immutable_data, 0, Layout(Shape([5, 10]), Stride([1, 5]))
    )

    with pytest.raises(RuntimeError):
        immutable_data.scatter(source, immutable_destination, mapping, 12)


def test_cpu_data_scatter_maps_source_values_into_destination_storage():
    source_data = CPU(3)
    for index, value in enumerate([10.0, 20.0, 30.0]):
        source_data[index] = value
    source = Tensor(source_data, 0, Layout(Shape(3), Stride(1)))
    destination_data = CPU(50)
    destination = Tensor(destination_data, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(3), Stride(5))

    destination_data.scatter(source, destination, mapping, 12)

    assert destination_data[12] == pytest.approx(10.0)
    assert destination_data[17] == pytest.approx(20.0)
    assert destination_data[22] == pytest.approx(30.0)
    assert destination_data[0] == pytest.approx(0.0)
