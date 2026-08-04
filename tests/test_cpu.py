import math
from array import array
from collections.abc import Iterable
from importlib import import_module
from typing import Any

import pytest

import strideweave as sw
from strideweave import (
    CPU,
    Carrier,
    DType,
    GenericViewOperation,
    Layout,
    Operation,
    PermuteOperation,
    RearrangeOperation,
    Shape,
    SqueezeOperation,
    Stride,
    UnsqueezeOperation,
)
from strideweave.carriers.cpu.capabilities import cpu_capabilities
from strideweave.carriers.operation_helpers import execute_lowered_operation
from strideweave.tensor import Tensor

_carrier = import_module("strideweave._carrier")

# The complete native kernel manifest, stated here independently of the
# registry so a kernel added, renamed, or dropped in C++ fails this test rather
# than silently changing what CPU dispatches.
CPU_KERNEL_METADATA = (
    ("_sort_indices", "cpu.sort_indices", "default", "_CPUSortOperation"),
    ("_sort_values", "cpu.sort_values", "default", "_CPUSortOperation"),
    ("_topk_indices", "cpu.topk_indices", "default", "_CPUSortOperation"),
    ("_topk_values", "cpu.topk_values", "default", "_CPUSortOperation"),
    ("abs", "cpu.abs", "default", "_CPUAbsOperation"),
    ("add", "cpu.add", "default", "_CPUAddOperation"),
    ("argmax", "cpu.argmax", "default", "_CPUArgmaxOperation"),
    ("argmin", "cpu.argmin", "default", "_CPUArgminOperation"),
    ("ceil", "cpu.ceil", "default", "_CPUCeilOperation"),
    ("clamp", "cpu.clamp", "default", "_CPUClampOperation"),
    ("conv_general", "cpu.conv_general", "default", "_CPUConvGeneralOperation"),
    ("cos", "cpu.cos", "default", "_CPUCosOperation"),
    ("cumsum", "cpu.cumsum", "default", "_CPUCumsumOperation"),
    ("div", "cpu.div", "default", "_CPUDivOperation"),
    (
        "elementwise_mul",
        "cpu.elementwise_mul",
        "default",
        "_CPUElementwiseMulOperation",
    ),
    ("elu", "cpu.elu", "default", "_CPUELUOperation"),
    ("eq", "cpu.eq", "default", "_CPUEqOperation"),
    ("erf", "cpu.erf", "default", "_CPUErfOperation"),
    ("exp", "cpu.exp", "default", "_CPUExpOperation"),
    ("exp2", "cpu.exp2", "default", "_CPUExp2Operation"),
    ("floor", "cpu.floor", "default", "_CPUFloorOperation"),
    ("gather", "cpu.gather", "default", "_CPUGatherOperation"),
    ("gelu", "cpu.gelu", "default", "_CPUGELUOperation"),
    ("le", "cpu.le", "default", "_CPULeOperation"),
    ("leaky_relu", "cpu.leaky_relu", "default", "_CPULeakyReLUOperation"),
    ("log", "cpu.log", "default", "_CPULogOperation"),
    ("log2", "cpu.log2", "default", "_CPULog2Operation"),
    ("logical_not", "cpu.logical_not", "default", "_CPULogicalNotOperation"),
    ("lt", "cpu.lt", "default", "_CPULtOperation"),
    ("matmul", "cpu.matmul", "default", "_CPUMatmulOperation"),
    ("maximum", "cpu.maximum", "default", "_CPUMaximumOperation"),
    ("minimum", "cpu.minimum", "default", "_CPUMinimumOperation"),
    ("mul", "cpu.scalar_mul", "default", "_CPUScalarMulOperation"),
    ("ne", "cpu.ne", "default", "_CPUNeOperation"),
    ("neg", "cpu.neg", "default", "_CPUNegOperation"),
    ("pow", "cpu.pow", "default", "_CPUPowOperation"),
    ("recip", "cpu.recip", "default", "_CPURecipOperation"),
    ("reduce_max", "cpu.reduce_max", "default", "_CPUReduceMaxOperation"),
    ("reduce_min", "cpu.reduce_min", "default", "_CPUReduceMinOperation"),
    ("reduce_prod", "cpu.reduce_prod", "default", "_CPUReduceProdOperation"),
    ("reduce_sum", "cpu.reduce_sum", "default", "_CPUReduceSumOperation"),
    ("relu", "cpu.relu", "default", "_CPUReLUOperation"),
    ("rem", "cpu.rem", "default", "_CPURemOperation"),
    ("round", "cpu.round", "default", "_CPURoundOperation"),
    ("rsqrt", "cpu.rsqrt", "default", "_CPURsqrtOperation"),
    ("scatter", "cpu.scatter", "default", "_CPUScatterOperation"),
    ("scatter_add", "cpu.scatter_add", "default", "_CPUScatterOperation"),
    ("select", "cpu.select", "default", "_CPUSelectOperation"),
    ("sigmoid", "cpu.sigmoid", "default", "_CPUSigmoidOperation"),
    ("sign", "cpu.sign", "default", "_CPUSignOperation"),
    ("silu", "cpu.silu", "default", "_CPUSiLUOperation"),
    ("sin", "cpu.sin", "default", "_CPUSinOperation"),
    ("softplus", "cpu.softplus", "default", "_CPUSoftplusOperation"),
    ("sqrt", "cpu.sqrt", "default", "_CPUSqrtOperation"),
    ("sub", "cpu.sub", "default", "_CPUSubOperation"),
    ("tanh", "cpu.tanh", "default", "_CPUTanhOperation"),
)


def make_cpu_carrier(
    values: Iterable[float | int], dtype: DType = DType.Float32
) -> CPU:
    materialized = list(values)
    carrier = CPU(len(materialized), dtype=dtype)
    for index, value in enumerate(materialized):
        carrier[index] = value
    return carrier


def make_cpu_tensor(
    values: Iterable[float | int],
    layout: Layout,
    dtype: DType = DType.Float32,
) -> Tensor:
    return Tensor(make_cpu_carrier(values, dtype), 0, layout)


def make_cpu_tensor_with_logical_values(
    values: Iterable[float | int],
    layout: Layout,
    dtype: DType = DType.Float32,
) -> Tensor:
    carrier = CPU(layout._cache.cosize, dtype=dtype)
    for logical_index, value in enumerate(values):
        carrier[layout.index(logical_index)] = value
    return Tensor(carrier, 0, layout)


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_cpu_data_contract_and_mutation():
    carrier = CPU(3)

    assert isinstance(carrier, Carrier)
    assert carrier.is_mutable()
    assert carrier.size() == 3
    assert carrier.dtype() is DType.Float32
    assert carrier.pointer() > 0
    assert [carrier[i] for i in range(carrier.size())] == [0.0, 0.0, 0.0]

    carrier[1] = 2.5
    carrier.set_value(2, 3)

    assert carrier.get_value(1) == pytest.approx(2.5)
    assert carrier[2] == pytest.approx(3.0)
    assert carrier.version == 2


@pytest.mark.parametrize(
    ("dtype", "zero"),
    [(DType.Float32, 0.0), (DType.Int32, 0)],
)
def test_cpu_owned_allocations_are_zero_initialized_by_default(dtype, zero):
    carrier = CPU(3, dtype=dtype)

    assert [carrier[i] for i in range(carrier.size())] == [zero, zero, zero]


@pytest.mark.parametrize(
    ("dtype", "values"),
    [(DType.Float32, [1.5, -2.5, 3.5]), (DType.Int32, [1, -2, 3])],
)
@pytest.mark.parametrize("use_factory", [False, True])
def test_cpu_empty_allocation_is_writable_before_reading(dtype, values, use_factory):
    if use_factory:
        carrier = CPU(0, dtype=dtype).allocate_like(3, empty=True)
    else:
        carrier = CPU(3, dtype=dtype, empty=True)

    for index, value in enumerate(values):
        carrier[index] = value

    assert [carrier[i] for i in range(carrier.size())] == pytest.approx(values)


def test_cpu_data_can_be_immutable():
    carrier = CPU(2, mutable=False)

    assert not carrier.is_mutable()

    with pytest.raises(RuntimeError):
        carrier[0] = 1.0
    with pytest.raises(RuntimeError):
        carrier.set_value(0, 1.0)

    assert carrier[0] == pytest.approx(0.0)


def test_cpu_data_validates_constructor_inputs():
    invalid_pointer: Any = "0"

    with pytest.raises(ValueError, match="CPU size must be non-negative"):
        CPU(-1)
    with pytest.raises(ValueError, match="CPU pointer must be a positive integer"):
        CPU(1, 0)
    with pytest.raises(TypeError):
        CPU(1, invalid_pointer)


def test_cpu_data_can_wrap_external_float32_pointer():
    values = array("f", [1.5, 2.5, 3.5])
    carrier = CPU(len(values), values.buffer_info()[0], empty=True)

    assert carrier[0] == pytest.approx(1.5)
    assert carrier[2] == pytest.approx(3.5)

    carrier[1] = 9.5

    assert values[1] == pytest.approx(9.5)


def test_cpu_int32_data_contract_and_pointer_storage():
    values = array("i", [1, -2, 3])
    carrier = CPU(len(values), values.buffer_info()[0], dtype=DType.Int32)

    assert carrier.dtype() is DType.Int32
    assert [carrier[i] for i in range(carrier.size())] == [1, -2, 3]

    carrier[1] = 9

    assert carrier[1] == 9
    assert values[1] == 9


def test_cpu_int32_data_validates_writes_and_dtype():
    carrier = CPU(1, dtype=DType.Int32)

    carrier[0] = 7
    assert carrier[0] == 7

    with pytest.raises(TypeError):
        carrier[0] = 1.5

    with pytest.raises(OverflowError):
        carrier[0] = 2**31

    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(1, dtype=DType.Floating)


def test_cpu_new_like_allocates_cpu_and_zero_fills_gap_placeholders():
    carrier = CPU(1)

    new_carrier = carrier.new_like([1.0, None, 3.0])

    assert type(new_carrier) is CPU
    assert new_carrier.size() == 3
    assert [new_carrier[i] for i in range(new_carrier.size())] == [1.0, 0.0, 3.0]


def test_cpu_new_like_preserves_or_overrides_dtype():
    carrier = CPU(1, dtype=DType.Int32)

    preserved = carrier.new_like([1, None, 3])
    overridden = carrier.new_like([1.5, None, 3.5], dtype=DType.Float32)

    assert preserved.dtype() is DType.Int32
    assert [preserved[i] for i in range(preserved.size())] == [1, 0, 3]
    assert overridden.dtype() is DType.Float32
    assert [overridden[i] for i in range(overridden.size())] == [1.5, 0.0, 3.5]


def test_cpu_int32_tensor_disables_autograd_interfaces():
    tensor = make_cpu_tensor([1, 2], Layout(Shape(2), Stride(1)), DType.Int32)
    gradient = make_cpu_tensor([1.0, 1.0], tensor.layout)

    assert not tensor.is_differentiable()
    with pytest.raises(RuntimeError, match="grad is not available"):
        tensor.grad
    with pytest.raises(RuntimeError, match="backward is not available"):
        tensor.backward(gradient)
    with pytest.raises(RuntimeError, match="retain_grad is not available"):
        tensor.retain_grad()
    with pytest.raises(RuntimeError, match="autograd_ctx is not available"):
        tensor.autograd_ctx = object()


def test_cpu_dispatch_op_returns_supported_operations():
    carrier = CPU(1)

    for (
        operation_name,
        _kernel_id,
        _variant,
        operation_type_name,
    ) in CPU_KERNEL_METADATA:
        first = carrier.dispatch_op(operation_name)
        second = carrier.dispatch_op(operation_name)
        assert type(first).__name__ == operation_type_name
        assert getattr(_carrier, operation_type_name) is type(first)
        assert type(second) is type(first)
        assert isinstance(first, Operation)
        assert first is not second
        assert first._operation_name == operation_name
        assert first._dispatch_carrier_class is CPU

    assert type(carrier.dispatch_op("broadcast_to")).__name__ == "BroadcastOperation"
    assert isinstance(carrier.dispatch_op("permute"), PermuteOperation)
    assert isinstance(carrier.dispatch_op("rearrange"), RearrangeOperation)
    assert isinstance(carrier.dispatch_op("squeeze"), SqueezeOperation)
    assert isinstance(carrier.dispatch_op("unsqueeze"), UnsqueezeOperation)
    assert isinstance(carrier.dispatch_op("view"), GenericViewOperation)

    with pytest.raises(NotImplementedError):
        carrier.dispatch_op("unknown")


def test_cpu_native_kernel_metadata_is_complete_stable_and_unique():
    metadata = _carrier._cpu_native_kernel_metadata()

    assert tuple(entry[:4] for entry in metadata) == CPU_KERNEL_METADATA
    assert tuple(entry[0] for entry in metadata) == tuple(
        sorted(entry[0] for entry in metadata)
    )
    assert len({entry[0] for entry in metadata}) == len(metadata)
    # Sort and top-k share one operation class, so their four dispatch names
    # carry four distinct kernel IDs rather than one shared ID.
    assert len({entry[1] for entry in metadata}) == len(metadata)
    assert {entry[2] for entry in metadata} == {"default"}
    assert all(
        entry[4].startswith("src/strideweave/carriers/cpu/native/ops/")
        and entry[4].endswith(".cpp")
        for entry in metadata
    )
    # Structural operations preserve dtype and layout instead of computing, so
    # they are Python-backed and carry no native kernel metadata.
    assert not {
        "as_strided",
        "broadcast_to",
        "permute",
        "rearrange",
        "reshape",
        "squeeze",
        "unsqueeze",
        "view",
    } & {entry[0] for entry in metadata}


def test_every_executable_cpu_capability_has_a_native_kernel():
    dispatch_names = {entry[0] for entry in CPU_KERNEL_METADATA}

    assert {capability.operation for capability in cpu_capabilities()} <= dispatch_names


def test_cpu_registry_rejects_duplicate_dispatch_names_and_kernel_ids():
    first = (
        "first",
        "cpu.first",
        "default",
        "_CPUFirstOperation",
        "src/first.cpp",
    )

    with pytest.raises(RuntimeError, match="duplicate CPU dispatch name 'first'"):
        _carrier._validate_cpu_native_registry_for_test(
            (
                first,
                (
                    "first",
                    "cpu.second",
                    "default",
                    "_CPUSecondOperation",
                    "src/second.cpp",
                ),
            )
        )

    with pytest.raises(RuntimeError, match=r"duplicate CPU kernel ID 'cpu\.first'"):
        _carrier._validate_cpu_native_registry_for_test(
            (
                first,
                (
                    "second",
                    "cpu.first",
                    "default",
                    "_CPUSecondOperation",
                    "src/second.cpp",
                ),
            )
        )


def test_cpu_is_a_closed_carrier_implementation():
    # CPU owns storage and capability claims stated in terms of its exact class,
    # so it is extended by composition rather than by specialization.
    with pytest.raises(TypeError, match="CPU is a closed carrier implementation"):
        type("CustomCPU", (CPU,), {})


class CpuBackedCarrier(Carrier):
    """An independent carrier composing a CPU carrier, as Evictable composes.

    This is the supported way to build a backend on top of an existing one: a
    sibling `Carrier` owning a CPU carrier and wrapping delegated operations in
    an adapter that owns lowering, result wrapping, and the autograd boundary.
    """

    def __init__(self, size: int, dtype: DType = DType.Float32):
        super().__init__()
        self._inner = CPU(size, dtype=dtype)
        self.dispatched: list[str] = []

    @classmethod
    def _from_inner(cls, inner: CPU) -> Any:
        result = cls.__new__(cls)
        Carrier.__init__(result)
        result._inner = inner
        result.dispatched = []
        return result

    def size(self) -> int:
        return self._inner.size()

    def dtype(self) -> DType:
        return self._inner.dtype()

    def get_value(self, index: int) -> Any:
        return self._inner[index]

    def _is_mutable(self) -> bool:
        return True

    def set_value(self, index: int, value: Any) -> None:
        self._inner[index] = value

    def new_like(self, values: Iterable[Any], *, mutable: bool = True) -> Any:
        materialized = list(values)
        result = type(self)(len(materialized), dtype=self.dtype())
        for index, value in enumerate(materialized):
            result[index] = value
        return result

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> Any:
        return type(self)(size, dtype=self.dtype() if dtype is None else dtype)

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError("CpuBackedCarrier does not implement scatter")

    def _dispatch_op(self, operation_name: str) -> Any:
        self.dispatched.append(operation_name)
        if operation_name == "custom_relu":
            return sw.GenericReLUOperation()
        return CpuBackedOperation(self._inner.dispatch_op(operation_name))


class CpuBackedOperation(Operation):
    """Own one outer autograd node while delegating computation to CPU."""

    def __init__(self, primary_operation: Operation) -> None:
        super().__init__()
        self.primary_operation = primary_operation

    @staticmethod
    def _lower_tensor(tensor: Any) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError("operation tensor inputs must be Tensors")
        if not isinstance(tensor.carrier, CpuBackedCarrier):
            raise TypeError("CpuBackedOperation requires CpuBackedCarrier inputs")
        return Tensor(tensor.carrier._inner, tensor.offset, tensor.layout)

    @staticmethod
    def _wrap_tensor(tensor: Any) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError("nested operation must return a Tensor")
        if type(tensor.carrier) is not CPU:
            raise TypeError("nested operation result must be backed by CPU")
        return Tensor(
            CpuBackedCarrier._from_inner(tensor.carrier),
            tensor.offset,
            tensor.layout,
        )

    def _forward(self, *inputs: Any) -> Tensor:
        lowered_arguments = tuple(
            self._lower_tensor(value) if isinstance(value, Tensor) else value
            for value in inputs
        )
        self.primary_operation.store_inputs(
            *(value for value in lowered_arguments if isinstance(value, Tensor))
        )
        primary_result = execute_lowered_operation(
            self.primary_operation, *lowered_arguments
        )
        return self._wrap_tensor(primary_result)

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        lowered_gradient = self._lower_tensor(gradient)
        primary_gradients = tuple(self.primary_operation.backward(lowered_gradient))
        if len(primary_gradients) != len(self.inputs()):
            raise ValueError("nested operation returned wrong number of gradients")
        return tuple(
            None if value is None else self._wrap_tensor(value)
            for value in primary_gradients
        )


class InvalidDispatchCpuBackedCarrier(CpuBackedCarrier):
    def _dispatch_op(self, operation_name: str) -> Any:
        return object()


def test_a_composed_carrier_supplies_its_own_operations():
    carrier = CpuBackedCarrier(1)

    first = carrier.dispatch_op("custom_relu")
    second = carrier.dispatch_op("custom_relu")

    assert carrier.dispatched == ["custom_relu", "custom_relu"]
    assert type(first) is sw.GenericReLUOperation
    assert type(second) is sw.GenericReLUOperation
    assert first is not second
    assert first._operation_name == "custom_relu"
    assert first._dispatch_carrier_class is CpuBackedCarrier


def test_a_composed_carrier_lowers_standard_operations_onto_cpu():
    carrier = CpuBackedCarrier(2)
    carrier[0] = -1.0
    carrier[1] = 2.0
    tensor = Tensor(carrier, 0, Layout(Shape(2), Stride(1)))

    custom = carrier.dispatch_op("custom_relu")
    result = sw.relu(tensor)

    assert carrier.dispatched == ["custom_relu", "relu"]
    assert type(custom) is sw.GenericReLUOperation
    assert type(result.carrier) is CpuBackedCarrier
    assert tensor_values(result) == [0.0, 2.0]
    adapter = result.autograd_ctx
    assert type(adapter) is CpuBackedOperation
    assert type(adapter.primary_operation).__name__ == "_CPUReLUOperation"
    (lowered_input,) = adapter.primary_operation.inputs()
    assert type(lowered_input.carrier) is CPU
    # Only the composite-owned adapter is visible in the graph and dispatch
    # metadata; the nested CPU operation remains behind lowered execution.
    assert custom._dispatch_carrier_class is CpuBackedCarrier
    assert adapter._dispatch_carrier_class is CpuBackedCarrier
    assert adapter._operation_name == "relu"

    gradient_carrier = CpuBackedCarrier(2)
    gradient_carrier[0] = 3.0
    gradient_carrier[1] = 4.0
    result.backward(Tensor(gradient_carrier, 0, result.layout))

    assert tensor.grad is not None
    assert type(tensor.grad.carrier) is CpuBackedCarrier
    assert tensor_values(tensor.grad) == [0.0, 4.0]


def test_a_composed_carrier_dispatch_hook_result_is_validated():
    with pytest.raises(TypeError, match="_dispatch_op must return an Operation"):
        InvalidDispatchCpuBackedCarrier(1).dispatch_op("invalid")


def test_cpu_allocate_like_allocates_requested_storage_and_dtype():
    result = CPU(0, dtype=DType.Int32).allocate_like(
        3, mutable=False, dtype=DType.Float32
    )

    assert result.size() == 3
    assert result.dtype() is DType.Float32
    assert not result.is_mutable()
    assert [result[i] for i in range(result.size())] == [0.0, 0.0, 0.0]


def test_cpu_tensor_constructor_reports_float32_device():
    carrier = make_cpu_carrier([1.0, 2.0, 3.0, 4.0])
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(carrier, 0, layout)

    assert tensor.dtype() is DType.Float32
    assert tensor.carrier_type() is CPU
    assert tensor_values(tensor) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_cpu_add_uses_native_operation_and_no_grad_state():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)
    rhs = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], layout)

    result = lhs + rhs

    assert result.layout == layout
    assert result.dtype() is DType.Float32
    assert result.carrier_type() is CPU
    assert tensor_values(result) == pytest.approx([11.0, 22.0, 33.0, 44.0])
    autograd_ctx = result.autograd_ctx
    assert autograd_ctx is not None
    assert type(autograd_ctx).__name__ == "_CPUAddOperation"
    assert autograd_ctx.inputs() == (lhs, rhs)

    with sw.no_grad():
        disabled_result = lhs + rhs

    assert disabled_result.autograd_ctx is None


def test_cpu_sub_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], layout)
    rhs = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)
    gradient = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)

    result = lhs - rhs

    assert result.layout == layout
    assert result.dtype() is DType.Float32
    assert result.carrier_type() is CPU
    assert tensor_values(result) == pytest.approx([9.0, 18.0, 27.0, 36.0])
    autograd_ctx = result.autograd_ctx
    assert autograd_ctx is not None
    assert type(autograd_ctx).__name__ == "_CPUSubOperation"
    assert autograd_ctx.inputs() == (lhs, rhs)

    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    # d(lhs) = gradient; d(rhs) = -gradient.
    assert tensor_values(lhs_grad) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert tensor_values(rhs_grad) == pytest.approx([-1.0, -2.0, -3.0, -4.0])
    assert type(lhs_grad.carrier) is CPU
    assert type(rhs_grad.carrier) is CPU

    with sw.no_grad():
        disabled_result = lhs - rhs

    assert disabled_result.autograd_ctx is None


def test_cpu_int32_sub_keeps_int32_without_autograd():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([10, 20, 30, 40], layout, DType.Int32)
    rhs = make_cpu_tensor([1, 2, 3, 4], layout, DType.Int32)

    result = lhs - rhs

    assert result.dtype() is DType.Int32
    assert tensor_values(result) == [9, 18, 27, 36]
    assert result.autograd_ctx is None


def test_cpu_int32_add_and_elementwise_mul_keep_int32_without_autograd():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([1, 2, 3, 4], layout, DType.Int32)
    rhs = make_cpu_tensor([10, 20, 30, 40], layout, DType.Int32)

    add_result = lhs + rhs
    mul_result = sw.elementwise_mul(lhs, rhs)

    assert add_result.dtype() is DType.Int32
    assert tensor_values(add_result) == [11, 22, 33, 44]
    assert add_result.autograd_ctx is None
    assert mul_result.dtype() is DType.Int32
    assert tensor_values(mul_result) == [10, 40, 90, 160]
    assert mul_result.autograd_ctx is None


def test_cpu_mixed_int32_float32_promotes_and_only_float_accumulates_grad():
    layout = Layout(Shape(2), Stride(1))
    int_tensor = make_cpu_tensor([1, 2], layout, DType.Int32)
    float_tensor = make_cpu_tensor([10.0, 20.0], layout)

    result = int_tensor + float_tensor
    result.backward(make_cpu_tensor([3.0, 4.0], layout))
    float_grad = require_grad(float_tensor)

    assert result.dtype() is DType.Float32
    assert tensor_values(result) == pytest.approx([11.0, 22.0])
    assert type(result.autograd_ctx).__name__ == "_CPUAddOperation"
    assert tensor_values(float_grad) == pytest.approx([3.0, 4.0])
    with pytest.raises(RuntimeError, match="grad is not available"):
        int_tensor.grad


def test_cpu_operation_output_has_independent_storage():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)
    rhs = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], layout)

    result = lhs + rhs
    lhs[0] = 100.0
    rhs[0] = 1000.0

    assert result.carrier is not lhs.carrier
    assert result.carrier is not rhs.carrier
    assert tensor_values(result) == pytest.approx([11.0, 22.0, 33.0, 44.0])


def test_cpu_operation_reads_external_float32_pointer_storage():
    values = array("f", [1.0, 2.0, 3.0, 4.0])
    carrier = CPU(len(values), values.buffer_info()[0])
    tensor = Tensor(carrier, 0, Layout(Shape([2, 2]), Stride([1, 2])))

    result = tensor * 2
    values[1] = 20.0
    updated_result = tensor * 2

    assert tensor_values(result) == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert tensor_values(updated_result) == pytest.approx([2.0, 40.0, 6.0, 8.0])


def test_cpu_tensor_view_uses_generic_operation_and_shares_storage():
    carrier = make_cpu_carrier(range(50))
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(carrier, 0, layout)

    view = tensor[2, 2:5]

    assert view.carrier is carrier
    assert view.offset == layout.index([2, 2])
    assert view.layout == Layout(Shape(3), Stride(5))
    assert tensor_values(view) == pytest.approx([tensor[2, j] for j in range(2, 5)])
    assert isinstance(view.autograd_ctx, GenericViewOperation)


def test_cpu_scalar_mul_accepts_strided_layout_and_backpropagates_cpu_grad():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    tensor = make_cpu_tensor(range(10), layout)
    gradient = make_cpu_tensor([1.0] * 10, layout)

    result = tensor * 5
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(result) == pytest.approx([0, 5, 20, 25, 40, 45])
    assert tensor_values(tensor_grad) == pytest.approx([5, 5, 5, 5, 5, 5])
    assert type(tensor_grad.carrier) is CPU


def test_cpu_scalar_mul_uses_expanded_keys_for_hierarchical_strides():
    layout = Layout(Shape([2, [3, 2]]), Stride([1, [10, 3]]))
    tensor = make_cpu_tensor(range(layout._cache.cosize), layout)

    result = tensor * 2

    assert tensor_values(result) == pytest.approx(
        [tensor[i] * 2 for i in range(tensor.size())]
    )


def test_cpu_int32_scalar_mul_promotes_for_non_integral_scalar():
    layout = Layout(Shape(3), Stride(1))
    tensor = make_cpu_tensor([2, 3, 4], layout, DType.Int32)

    int_result = tensor * 3
    float_result = tensor * 2.5

    assert int_result.dtype() is DType.Int32
    assert tensor_values(int_result) == [6, 9, 12]
    assert int_result.autograd_ctx is None
    assert float_result.dtype() is DType.Float32
    assert tensor_values(float_result) == pytest.approx([5.0, 7.5, 10.0])
    assert float_result.autograd_ctx is None


def test_cpu_elementwise_mul_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    lhs = make_cpu_tensor(range(layout._cache.cosize), layout)
    rhs = make_cpu_tensor(range(10, 10 + layout._cache.cosize), layout)

    result = sw.elementwise_mul(lhs, rhs)
    gradient = make_cpu_tensor([1.0] * result.layout.cosize, result.layout)
    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert type(result.autograd_ctx).__name__ == "_CPUElementwiseMulOperation"
    assert tensor_values(result) == pytest.approx(
        [lhs[i] * rhs[i] for i in range(lhs.size())]
    )
    assert tensor_values(lhs_grad) == pytest.approx([rhs[i] for i in range(rhs.size())])
    assert tensor_values(rhs_grad) == pytest.approx([lhs[i] for i in range(lhs.size())])
    assert type(lhs_grad.carrier) is CPU
    assert type(rhs_grad.carrier) is CPU


def test_cpu_int32_non_integer_result_operations_promote_to_float32():
    layout = Layout(Shape(2), Stride(1))
    lhs = make_cpu_tensor([2, 3], layout, DType.Int32)
    rhs = make_cpu_tensor([4, 2], layout, DType.Int32)

    div_result = lhs / rhs
    exp_result = sw.exp(lhs)
    sigmoid_result = sw.sigmoid(lhs)
    pow_result = lhs**-1

    assert div_result.dtype() is DType.Float32
    assert tensor_values(div_result) == pytest.approx([0.5, 1.5])
    assert exp_result.dtype() is DType.Float32
    assert tensor_values(exp_result) == pytest.approx([math.exp(2), math.exp(3)])
    assert sigmoid_result.dtype() is DType.Float32
    assert tensor_values(sigmoid_result) == pytest.approx(
        [1.0 / (1.0 + math.exp(-2)), 1.0 / (1.0 + math.exp(-3))]
    )
    assert pow_result.dtype() is DType.Float32
    assert tensor_values(pow_result) == pytest.approx([0.5, 1 / 3])
    assert div_result.autograd_ctx is None
    assert exp_result.autograd_ctx is None
    assert sigmoid_result.autograd_ctx is None
    assert pow_result.autograd_ctx is None


def test_cpu_int32_pow_relu_reduce_and_matmul_preserve_int32():
    layout = Layout(Shape(3), Stride(1))
    tensor = make_cpu_tensor([-2, 3, 4], layout, DType.Int32)
    reduce_tensor = make_cpu_tensor(
        [1, 2, 3, 4, 5, 6], Layout(Shape([2, 3]), Stride([1, 2])), DType.Int32
    )
    lhs = make_cpu_tensor(
        [1, 2, 3, 4, 5, 6], Layout(Shape([2, 3]), Stride([1, 2])), DType.Int32
    )
    rhs = make_cpu_tensor(
        [1, 0, 0, 1, 0, 1], Layout(Shape([2, 3]), Stride([1, 2])), DType.Int32
    )

    pow_result = tensor**2
    relu_result = sw.relu(tensor)
    reduce_result = sw.reduce_sum(reduce_tensor, "a b -> a")
    matmul_result = lhs @ rhs

    assert pow_result.dtype() is DType.Int32
    assert tensor_values(pow_result) == [4, 9, 16]
    assert relu_result.dtype() is DType.Int32
    assert tensor_values(relu_result) == [0, 3, 4]
    assert reduce_result.dtype() is DType.Int32
    assert tensor_values(reduce_result) == [9, 12]
    assert matmul_result.dtype() is DType.Int32
    assert tensor_values(matmul_result) == [1, 2, 8, 10]
    assert pow_result.autograd_ctx is None
    assert relu_result.autograd_ctx is None
    assert reduce_result.autograd_ctx is None
    assert matmul_result.autograd_ctx is None


def test_cpu_int32_relu_preserves_large_values_without_float_rounding():
    layout = Layout(Shape(3), Stride(1))
    max_int32 = 2**31 - 1
    tensor = make_cpu_tensor(
        [max_int32, max_int32 - 1, -max_int32], layout, DType.Int32
    )

    result = sw.relu(tensor)

    assert result.dtype() is DType.Int32
    assert tensor_values(result) == [max_int32, max_int32 - 1, 0]


def test_cpu_int32_operations_raise_on_overflow():
    one_mode = Layout(Shape(1), Stride(1))
    matmul_layout = Layout(Shape([1, 1]), Stride([1, 1]))
    two_mode = Layout(Shape([1, 2]), Stride([1, 1]))
    max_int32 = 2**31 - 1

    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([max_int32], one_mode, DType.Int32) + make_cpu_tensor(
            [1], one_mode, DType.Int32
        )
    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([max_int32], one_mode, DType.Int32) * 2
    with pytest.raises(OverflowError):
        _ = sw.elementwise_mul(
            make_cpu_tensor([50_000], one_mode, DType.Int32),
            make_cpu_tensor([50_000], one_mode, DType.Int32),
        )
    with pytest.raises(OverflowError):
        sw.reduce_sum(
            make_cpu_tensor([max_int32, 1], two_mode, DType.Int32), "a b -> a"
        )
    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([max_int32], matmul_layout, DType.Int32) @ make_cpu_tensor(
            [2], matmul_layout, DType.Int32
        )


def test_cpu_int32_hierarchical_layout_uses_expanded_keys():
    # This layout has a storage gap, so the kernel must iterate logical expanded
    # keys instead of assuming raw contiguous storage order.
    layout = Layout(Shape([[2, 2]]), Stride([[1, 3]]))
    tensor = make_cpu_tensor_with_logical_values([1, 2, 3, 4], layout, DType.Int32)

    result = tensor * 2

    assert result.dtype() is DType.Int32
    assert tensor_values(result) == [2, 4, 6, 8]


def test_cpu_div_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([8.0, 9.0, 10.0, 12.0], layout)
    rhs = make_cpu_tensor([2.0, 3.0, 5.0, 4.0], layout)
    gradient = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)

    result = lhs / rhs
    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert type(result.autograd_ctx).__name__ == "_CPUDivOperation"
    assert tensor_values(result) == pytest.approx([4.0, 3.0, 2.0, 3.0])
    assert tensor_values(lhs_grad) == pytest.approx([0.5, 2.0 / 3.0, 0.6, 1.0])
    assert tensor_values(rhs_grad) == pytest.approx([-2.0, -2.0, -1.2, -3.0])
    assert type(lhs_grad.carrier) is CPU
    assert type(rhs_grad.carrier) is CPU


def test_cpu_exp_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = make_cpu_tensor([0.0, 1.0, 2.0, 3.0], layout)
    gradient = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)

    result = sw.exp(tensor)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    expected = [math.exp(value) for value in [0.0, 1.0, 2.0, 3.0]]
    assert type(result.autograd_ctx).__name__ == "_CPUExpOperation"
    assert tensor_values(result) == pytest.approx(expected)
    assert tensor_values(tensor_grad) == pytest.approx(
        [
            grad * value
            for grad, value in zip([1.0, 2.0, 3.0, 4.0], expected, strict=True)
        ]
    )
    assert type(tensor_grad.carrier) is CPU


def test_cpu_pow_scalar_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)
    gradient = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)

    result = tensor**3
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert type(result.autograd_ctx).__name__ == "_CPUPowOperation"
    assert tensor_values(result) == pytest.approx([1.0, 8.0, 27.0, 64.0])
    assert tensor_values(tensor_grad) == pytest.approx([3.0, 24.0, 81.0, 192.0])
    assert type(tensor_grad.carrier) is CPU


def test_cpu_reduce_sums_second_mode_and_backpropagates():
    tensor = make_cpu_tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Layout(Shape([2, 3]), Stride([1, 2])),
    )

    result = sw.reduce_sum(tensor, "a b -> a")
    gradient = make_cpu_tensor([10.0, 20.0], result.layout)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert result.layout == Layout(Shape(2), Stride(1))
    assert tensor_values(result) == pytest.approx([9.0, 12.0])
    assert type(result.autograd_ctx).__name__ == "_CPUReduceSumOperation"
    assert tensor_values(tensor_grad) == pytest.approx([10, 20, 10, 20, 10, 20])
    assert type(tensor_grad.carrier) is CPU


def test_cpu_reduce_uses_expanded_keys_for_hierarchical_modes():
    layout = Layout(Shape([[2, 2], [3, 2]]), Stride([[1, 5], [20, 7]]))
    tensor = make_cpu_tensor(range(layout._cache.cosize), layout)

    result = sw.reduce_sum(tensor, "a b -> a")
    gradient = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], result.layout)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert tensor_values(result) == pytest.approx(
        [sum(tensor[i, j] for j in range(6)) for i in range(4)]
    )
    assert tensor_values(tensor_grad) == pytest.approx(
        [gradient[i] for j in range(6) for i in range(4)]
    )


def test_cpu_matmul_computes_output_and_input_gradients():
    a = make_cpu_tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Layout(Shape([2, 3]), Stride([1, 2])),
    )
    b = make_cpu_tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = a @ b
    gradient = make_cpu_tensor([1.0] * 8, result.layout)
    result.backward(gradient)
    a_grad = require_grad(a)
    b_grad = require_grad(b)

    assert result.layout == Layout(Shape([2, 4]), Stride([1, 2]))
    assert tensor_values(result) == pytest.approx([1, 2, 3, 4, 5, 6, 9, 12])
    assert type(result.autograd_ctx).__name__ == "_CPUMatmulOperation"
    assert tensor_values(a_grad) == pytest.approx([2, 2, 2, 2, 2, 2])
    assert tensor_values(b_grad) == pytest.approx(
        [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11]
    )
    assert type(a_grad.carrier) is CPU
    assert type(b_grad.carrier) is CPU


def test_cpu_matmul_uses_expanded_keys_for_hierarchical_contract_mode():
    a_layout = Layout(Shape([2, [2, 2]]), Stride([1, [7, 3]]))
    b_layout = Layout(Shape([3, [2, 2]]), Stride([2, [11, 5]]))
    a = make_cpu_tensor(range(a_layout._cache.cosize), a_layout)
    b = make_cpu_tensor(range(b_layout._cache.cosize), b_layout)

    result = a @ b
    gradient = make_cpu_tensor([1.0] * result.layout._cache.cosize, result.layout)
    result.backward(gradient)
    a_grad = require_grad(a)
    b_grad = require_grad(b)

    assert tensor_values(result) == pytest.approx(
        [sum(a[i, k] * b[j, k] for k in range(4)) for j in range(3) for i in range(2)]
    )
    assert tensor_values(a_grad) == pytest.approx(
        [sum(b[j, k] for j in range(3)) for k in range(4) for _i in range(2)]
    )
    assert tensor_values(b_grad) == pytest.approx(
        [sum(a[i, k] for i in range(2)) for k in range(4) for _j in range(3)]
    )


def test_cpu_view_operations_reuse_python_layout_operations():
    carrier = make_cpu_carrier([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor = Tensor(carrier, 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = sw.permute(tensor, 1, 0)

    assert result.carrier is carrier
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, PermuteOperation)


def test_cpu_int32_pow_handles_large_exponents():
    tensor = make_cpu_tensor([1, 0, -1], Layout(Shape(3), Stride(1)), DType.Int32)

    even = sw.pow(tensor, 2**30)
    odd = sw.pow(tensor, 2**24 - 1)

    assert even.dtype() is DType.Int32
    assert [even[0], even[1], even[2]] == [1, 0, 1]
    assert [odd[0], odd[1], odd[2]] == [1, 0, -1]


def test_cpu_int32_pow_overflow_raises():
    tensor = make_cpu_tensor([3], Layout(Shape(1), Stride(1)), DType.Int32)

    with pytest.raises(OverflowError):
        sw.pow(tensor, 40)


def test_cpu_bool_scalar_multiplies_as_float():
    tensor = make_cpu_tensor([3], Layout(Shape(1), Stride(1)), DType.Int32)

    result = sw.mul(tensor, True)

    assert result.dtype() is DType.Float32
    assert result[0] == 3.0
