from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any, Protocol, cast

import pytest

import strideweave as sw
from strideweave import (
    CPU,
    Carrier,
    DType,
    Generic,
    Layout,
    Shape,
    Stride,
)
from strideweave.tensor import Tensor


class NativeCarrierModule(Protocol):
    _VectorCarrierForTest: Callable[[list[Any]], Carrier]


native_carrier = cast(NativeCarrierModule, import_module("strideweave._carrier"))


class PythonCarrier(Carrier):
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
    ) -> "PythonCarrier":
        return type(self)(list(values))

    def empty_like(
        self, size: int, *, mutable: bool = True, dtype: DType | None = None
    ) -> "PythonCarrier":
        return type(self)([None] * size)

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("PythonCarrier does not implement scatter")


class PythonMutableCarrier(PythonCarrier):
    def _is_mutable(self) -> bool:
        return True

    def set_value(self, index: int, value: Any) -> None:
        self.values[index] = value


class DispatchOperation(sw.Operation):
    def _forward(self, *inputs: Any) -> Tensor:
        return Tensor(Generic([1.0]), 0, Layout(Shape(1), Stride(1)))

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        return ()


class DispatchingPythonCarrier(PythonCarrier):
    def _dispatch_op(self, operation_name: str) -> sw.Operation:
        if operation_name != "custom":
            raise NotImplementedError
        return DispatchOperation()


class InvalidDispatchCarrier(PythonCarrier):
    def _dispatch_op(self, operation_name: str) -> object:
        return object()


class CachedDispatchCarrier(PythonCarrier):
    def __init__(self, values: list[Any]):
        super().__init__(values)
        self.operation = DispatchOperation()

    def _dispatch_op(self, operation_name: str) -> sw.Operation:
        return self.operation


def test_data_public_api_imports():
    assert sw.Carrier is Carrier
    assert sw.CPU is CPU
    assert sw.DType is DType
    assert sw.Generic is Generic
    assert DType.Any.value == "Any"
    assert DType.Floating.value == "Floating"
    assert DType.Float32.value == "Float32"
    assert DType.Int32.value == "Int32"


def test_data_default_dispatch_op_raises_not_implemented():
    carrier = PythonCarrier([])
    with pytest.raises(NotImplementedError):
        carrier.dispatch_op("add")


def test_python_carrier_dispatch_policy_uses_hook_and_exact_class_metadata():
    carrier = DispatchingPythonCarrier([])

    first = carrier.dispatch_op("custom")
    second = carrier.dispatch_op("custom")

    assert type(first) is DispatchOperation
    assert first is not second
    assert first._operation_name == "custom"
    assert first._dispatch_carrier_class is DispatchingPythonCarrier


def test_carrier_dispatch_policy_requires_operation_result():
    with pytest.raises(TypeError, match="_dispatch_op must return an Operation"):
        InvalidDispatchCarrier([]).dispatch_op("custom")


def test_carrier_dispatch_policy_rejects_cached_dispatched_operation():
    carrier = CachedDispatchCarrier([])

    first = carrier.dispatch_op("first")
    first.ctx["state"] = "retained"

    with pytest.raises(TypeError, match="fresh Operation"):
        carrier.dispatch_op("second")

    assert first is carrier.operation
    assert first._operation_name == "first"
    assert first._dispatch_carrier_class is CachedDispatchCarrier
    assert first.ctx == {"state": "retained"}


def test_generic_data_dispatch_op_returns_supported_operations():
    carrier = Generic([])
    cases = {
        "add": sw.GenericAddOperation,
        "div": sw.GenericDivOperation,
        "elu": sw.GenericELUOperation,
        "elementwise_mul": sw.GenericElementwiseMulOperation,
        "exp": sw.GenericExpOperation,
        "gelu": sw.GenericGELUOperation,
        "leaky_relu": sw.GenericLeakyReLUOperation,
        "matmul": sw.GenericMatmulOperation,
        "mul": sw.GenericScalarMulOperation,
        "permute": sw.PermuteOperation,
        "pow": sw.GenericPowOperation,
        "rearrange": sw.RearrangeOperation,
        "reduce": sw.GenericReduceSumOperation,
        "relu": sw.GenericReLUOperation,
        "sigmoid": sw.GenericSigmoidOperation,
        "silu": sw.GenericSiLUOperation,
        "softplus": sw.GenericSoftplusOperation,
        "sub": sw.GenericSubOperation,
        "tanh": sw.GenericTanhOperation,
        "view": sw.GenericViewOperation,
    }

    for operation_name, operation_type in cases.items():
        first = carrier.dispatch_op(operation_name)
        second = carrier.dispatch_op(operation_name)
        assert type(first) is operation_type
        assert type(second) is operation_type
        assert first is not second
        assert first._operation_name == operation_name
        assert first._dispatch_carrier_class is Generic


def test_generic_data_dispatch_op_rejects_unknown_operation():
    with pytest.raises(NotImplementedError):
        Generic([]).dispatch_op("unknown")


@pytest.mark.parametrize("carrier_class", [Carrier, Generic, CPU])
def test_data_operation_dispatch_is_instance_only(carrier_class):
    with pytest.raises(TypeError):
        carrier_class.dispatch_op("relu")  # type: ignore[misc]


def test_generic_empty_like_allocates_requested_storage_and_dtype():
    result = Generic([], dtype=DType.Any).empty_like(
        3, mutable=False, dtype=DType.Floating
    )

    assert result.size() == 3
    assert result.dtype() is DType.Floating
    assert not result.is_mutable()


def test_python_data_subclass_implements_data_contract():
    carrier = PythonCarrier(["alpha", 2, None])

    assert isinstance(carrier, Carrier)
    assert not carrier.is_mutable()
    assert carrier.size() == 3
    assert carrier.dtype() is DType.Any
    assert carrier.get_value(1) == 2
    assert carrier[0] == "alpha"
    assert carrier[2] is None


def test_python_data_subclass_new_like_creates_matching_data_class():
    carrier = PythonCarrier(["alpha"])

    new_carrier = carrier.new_like(["beta", "gamma"])

    assert type(new_carrier) is PythonCarrier
    assert new_carrier.size() == 2
    assert new_carrier[0] == "beta"


def test_native_carrier_subclass_new_like_creates_matching_data_class():
    carrier = native_carrier._VectorCarrierForTest(["alpha"])

    new_carrier = carrier.new_like(["beta", "gamma"])

    assert type(new_carrier) is type(carrier)
    assert new_carrier.size() == 2
    assert new_carrier[0] == "beta"


def test_python_data_subclass_getitem_validates_index():
    carrier = PythonCarrier(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        carrier[-1]

    with pytest.raises(IndexError):
        carrier[1]

    with pytest.raises(TypeError):
        carrier[non_int_index]


def test_python_data_subclass_rejects_setitem_by_default():
    carrier = PythonCarrier(["alpha"])

    with pytest.raises(RuntimeError):
        carrier[0] = "updated"

    assert carrier[0] == "alpha"


def test_python_mutable_data_subclass_setitem_updates_values():
    carrier = PythonMutableCarrier(["alpha", "beta"])
    non_int_index: Any = "0"

    assert carrier.is_mutable()

    carrier[1] = "updated"

    assert carrier.get_value(1) == "updated"
    assert carrier[1] == "updated"

    with pytest.raises(IndexError):
        carrier[-1] = "negative"

    with pytest.raises(IndexError):
        carrier[2] = "large"

    with pytest.raises(TypeError):
        carrier[non_int_index] = "non-int"


def test_native_carrier_subclass_implements_data_contract():
    carrier = native_carrier._VectorCarrierForTest(["alpha", 2, None])

    assert isinstance(carrier, Carrier)
    assert not carrier.is_mutable()
    assert carrier.size() == 3
    assert carrier.dtype() is DType.Any
    assert carrier.get_value(1) == 2
    assert carrier[0] == "alpha"
    assert carrier[2] is None


def test_native_carrier_subclass_getitem_validates_index():
    carrier = native_carrier._VectorCarrierForTest(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        carrier[-1]

    with pytest.raises(IndexError):
        carrier[1]

    with pytest.raises(TypeError):
        carrier[non_int_index]


def test_native_carrier_subclass_rejects_setitem_by_default():
    carrier = native_carrier._VectorCarrierForTest(["alpha"])

    with pytest.raises(RuntimeError):
        carrier[0] = "updated"

    assert carrier[0] == "alpha"


def test_generic_data_wraps_indexable_iterable():
    values = ["alpha", 2, None]
    carrier = Generic(values, dtype=DType.Any)

    assert isinstance(carrier, Carrier)
    assert carrier.is_mutable()
    assert carrier.size() == 3
    assert carrier.dtype() is DType.Any
    assert carrier.get_value(1) == 2
    assert carrier[0] == "alpha"
    assert carrier[2] is None

    values[1] = "updated"
    assert carrier[1] == "updated"


def test_generic_data_defaults_to_floating_dtype():
    carrier = Generic([1, 2, 3])

    assert carrier.dtype() is DType.Floating


def test_generic_data_new_like_preserves_or_overrides_dtype():
    carrier = Generic([1, 2, 3])

    preserved = carrier.new_like([4, 5])
    overridden = carrier.new_like(["alpha"], dtype=DType.Any)

    assert preserved.dtype() is DType.Floating
    assert overridden.dtype() is DType.Any


def test_generic_data_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=DType.Int32)

    with pytest.raises(TypeError):
        Generic([1], dtype="Floating")  # type: ignore[arg-type]


def test_generic_data_mutates_backing_list_by_default():
    values = ["alpha", "beta"]
    carrier = Generic(values)

    carrier[1] = "updated"

    assert values[1] == "updated"
    assert carrier[1] == "updated"


def test_generic_data_can_be_immutable():
    values = ["alpha", "beta"]
    carrier = Generic(values, mutable=False)

    assert not carrier.is_mutable()

    with pytest.raises(RuntimeError):
        carrier[1] = "updated"

    assert values[1] == "beta"
    assert carrier[1] == "beta"


def test_generic_data_materializes_non_settable_inputs_when_mutable():
    tuple_carrier = Generic(("alpha", "beta"))
    range_carrier = Generic(range(3))
    iterator = iter(["left", "right"])
    iterator_carrier = Generic(iterator)

    tuple_carrier[1] = "updated"
    range_carrier[2] = "updated"
    iterator_carrier[0] = "updated"

    assert tuple_carrier[1] == "updated"
    assert range_carrier[2] == "updated"
    assert iterator_carrier[0] == "updated"
    assert list(iterator) == []


def test_generic_data_wraps_tuple_and_range():
    tuple_carrier = Generic(("alpha", "beta"))
    range_carrier = Generic(range(3))

    assert tuple_carrier.size() == 2
    assert tuple_carrier[1] == "beta"
    assert range_carrier.size() == 3
    assert range_carrier[2] == 2


def test_generic_data_materializes_one_pass_iterable():
    iterator = iter(["alpha", "beta", "gamma"])
    carrier = Generic(iterator)

    assert carrier.size() == 3
    assert carrier[0] == "alpha"
    assert carrier[2] == "gamma"
    assert list(iterator) == []


def test_generic_data_requires_iterable_input():
    non_iterable: Any = 1

    with pytest.raises(TypeError):
        Generic(non_iterable)


def test_generic_data_getitem_validates_index():
    carrier = Generic(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        carrier[-1]

    with pytest.raises(IndexError):
        carrier[1]

    with pytest.raises(TypeError):
        carrier[non_int_index]


def test_generic_data_setitem_validates_index():
    carrier = Generic(["alpha"])
    non_int_index: Any = "0"

    with pytest.raises(IndexError):
        carrier[-1] = "negative"

    with pytest.raises(IndexError):
        carrier[1] = "large"

    with pytest.raises(TypeError):
        carrier[non_int_index] = "non-int"


def test_generic_public_write_paths_increment_version():
    carrier = Generic([0, 0])

    carrier[0] = 1
    assert carrier.version == 1

    carrier.set_value(1, 2)
    assert carrier.version == 2


def test_generic_data_scatter_maps_source_values_into_destination_storage():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    destination_values = [0] * 50
    destination_carrier = Generic(destination_values)
    destination = Tensor(destination_carrier, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(3), Stride(5))

    destination_carrier.scatter(source, destination, mapping, 12)

    assert destination_values[12] == 10
    assert destination_values[17] == 20
    assert destination_values[22] == 30
    assert sum(destination_values) == 60
    assert destination_carrier.version > 0


def test_generic_data_scatter_validates_mapping_shape():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    destination_carrier = Generic([0] * 50)
    destination = Tensor(destination_carrier, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(4), Stride(5))

    with pytest.raises(ValueError, match="mapping shape must match"):
        destination_carrier.scatter(source, destination, mapping, 12)


def test_generic_data_scatter_uses_destination_mutability():
    source = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))
    mapping = Layout(Shape(3), Stride(5))
    immutable_carrier = Generic([0] * 50, mutable=False)
    immutable_destination = Tensor(
        immutable_carrier, 0, Layout(Shape([5, 10]), Stride([1, 5]))
    )

    with pytest.raises(RuntimeError):
        immutable_carrier.scatter(source, immutable_destination, mapping, 12)


def test_cpu_data_scatter_maps_source_values_into_destination_storage():
    source_carrier = CPU(3)
    for index, value in enumerate([10.0, 20.0, 30.0]):
        source_carrier[index] = value
    source = Tensor(source_carrier, 0, Layout(Shape(3), Stride(1)))
    destination_carrier = CPU(50)
    destination = Tensor(destination_carrier, 0, Layout(Shape([5, 10]), Stride([1, 5])))
    mapping = Layout(Shape(3), Stride(5))

    destination_carrier.scatter(source, destination, mapping, 12)

    assert destination_carrier[12] == pytest.approx(10.0)
    assert destination_carrier[17] == pytest.approx(20.0)
    assert destination_carrier[22] == pytest.approx(30.0)
    assert destination_carrier[0] == pytest.approx(0.0)
    assert destination_carrier.version > 0
