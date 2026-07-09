import math
from array import array
from collections.abc import Iterable
from typing import Any

import neotorch
import pytest
from neotorch import (
    CPU,
    Data,
    DataType,
    GenericViewOperation,
    Layout,
    Operation,
    PermuteOperation,
    RearrangeOperation,
    Shape,
    Stride,
)
from neotorch.tensor import Tensor


def make_cpu_data(
    values: Iterable[float | int], dtype: DataType = DataType.Float32
) -> CPU:
    materialized = list(values)
    data = CPU(len(materialized), dtype=dtype)
    for index, value in enumerate(materialized):
        data[index] = value
    return data


def make_cpu_tensor(
    values: Iterable[float | int],
    layout: Layout,
    dtype: DataType = DataType.Float32,
) -> Tensor:
    return Tensor(make_cpu_data(values, dtype), 0, layout)


def make_cpu_tensor_with_logical_values(
    values: Iterable[float | int],
    layout: Layout,
    dtype: DataType = DataType.Float32,
) -> Tensor:
    data = CPU(layout._cache.cosize, dtype=dtype)
    for logical_index, value in enumerate(values):
        data[layout.index(logical_index)] = value
    return Tensor(data, 0, layout)


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_cpu_data_contract_and_mutation():
    data = CPU(3)

    assert isinstance(data, Data)
    assert data.is_mutable()
    assert data.size() == 3
    assert data.type() is DataType.Float32
    assert data.pointer() > 0
    assert [data[i] for i in range(data.size())] == [0.0, 0.0, 0.0]

    data[1] = 2.5
    data.set_value(2, 3)

    assert data.get_value(1) == pytest.approx(2.5)
    assert data[2] == pytest.approx(3.0)


def test_cpu_data_can_be_immutable():
    data = CPU(2, mutable=False)

    assert not data.is_mutable()

    with pytest.raises(RuntimeError):
        data[0] = 1.0
    with pytest.raises(RuntimeError):
        data.set_value(0, 1.0)

    assert data[0] == pytest.approx(0.0)


def test_cpu_data_validates_constructor_inputs():
    invalid_pointer: Any = "0"

    with pytest.raises(ValueError):
        CPU(-1)
    with pytest.raises(ValueError):
        CPU(1, 0)
    with pytest.raises(TypeError):
        CPU(1, invalid_pointer)


def test_cpu_data_can_wrap_external_float32_pointer():
    values = array("f", [1.5, 2.5, 3.5])
    data = CPU(len(values), values.buffer_info()[0])

    assert data[0] == pytest.approx(1.5)
    assert data[2] == pytest.approx(3.5)

    data[1] = 9.5

    assert values[1] == pytest.approx(9.5)


def test_cpu_int32_data_contract_and_pointer_storage():
    values = array("i", [1, -2, 3])
    data = CPU(len(values), values.buffer_info()[0], dtype=DataType.Int32)

    assert data.type() is DataType.Int32
    assert [data[i] for i in range(data.size())] == [1, -2, 3]

    data[1] = 9

    assert data[1] == 9
    assert values[1] == 9


def test_cpu_int32_data_validates_writes_and_dtype():
    data = CPU(1, dtype=DataType.Int32)

    data[0] = 7
    assert data[0] == 7

    with pytest.raises(TypeError):
        data[0] = 1.5

    with pytest.raises(OverflowError):
        data[0] = 2**31

    with pytest.raises(ValueError):
        CPU(1, dtype=DataType.Floating)


def test_cpu_new_like_allocates_cpu_and_zero_fills_gap_placeholders():
    data = CPU(1)

    new_data = data.new_like([1.0, None, 3.0])

    assert type(new_data) is CPU
    assert new_data.size() == 3
    assert [new_data[i] for i in range(new_data.size())] == [1.0, 0.0, 3.0]


def test_cpu_new_like_preserves_or_overrides_dtype():
    data = CPU(1, dtype=DataType.Int32)

    preserved = data.new_like([1, None, 3])
    overridden = data.new_like([1.5, None, 3.5], dtype=DataType.Float32)

    assert preserved.type() is DataType.Int32
    assert [preserved[i] for i in range(preserved.size())] == [1, 0, 3]
    assert overridden.type() is DataType.Float32
    assert [overridden[i] for i in range(overridden.size())] == [1.5, 0.0, 3.5]


def test_cpu_int32_tensor_disables_autograd_interfaces():
    tensor = make_cpu_tensor([1, 2], Layout(Shape(2), Stride(1)), DataType.Int32)
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
    native_cases = {
        "add": "_CPUAddOperation",
        "div": "_CPUDivOperation",
        "elu": "_CPUELUOperation",
        "elementwise_mul": "_CPUElementwiseMulOperation",
        "exp": "_CPUExpOperation",
        "gelu": "_CPUGELUOperation",
        "leaky_relu": "_CPULeakyReLUOperation",
        "matmul": "_CPUMatmulOperation",
        "mul": "_CPUScalarMulOperation",
        "pow": "_CPUPowOperation",
        "reduce": "_CPUReduceSumOperation",
        "relu": "_CPUReLUOperation",
        "sigmoid": "_CPUSigmoidOperation",
        "silu": "_CPUSiLUOperation",
        "softplus": "_CPUSoftplusOperation",
        "tanh": "_CPUTanhOperation",
    }

    for operation_name, operation_type_name in native_cases.items():
        operation = CPU.dispatch_op(operation_name)
        assert type(operation).__name__ == operation_type_name
        assert isinstance(operation, Operation)

    assert isinstance(CPU.dispatch_op("permute"), PermuteOperation)
    assert isinstance(CPU.dispatch_op("rearrange"), RearrangeOperation)
    assert isinstance(CPU.dispatch_op("view"), GenericViewOperation)

    with pytest.raises(NotImplementedError):
        CPU.dispatch_op("unknown")


def test_cpu_tensor_constructor_reports_float32_device():
    data = make_cpu_data([1.0, 2.0, 3.0, 4.0])
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = Tensor(data, 0, layout)

    assert tensor.dtype() is DataType.Float32
    assert tensor.device() is CPU
    assert tensor_values(tensor) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_cpu_add_uses_native_operation_and_no_grad_state():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)
    rhs = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], layout)

    result = lhs + rhs

    assert result.layout == layout
    assert result.dtype() is DataType.Float32
    assert result.device() is CPU
    assert tensor_values(result) == pytest.approx([11.0, 22.0, 33.0, 44.0])
    autograd_ctx = result.autograd_ctx
    assert autograd_ctx is not None
    assert type(autograd_ctx).__name__ == "_CPUAddOperation"
    assert autograd_ctx.inputs() == (lhs, rhs)

    with neotorch.no_grad():
        disabled_result = lhs + rhs

    assert disabled_result.autograd_ctx is None


def test_cpu_int32_add_and_elementwise_mul_keep_int32_without_autograd():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    lhs = make_cpu_tensor([1, 2, 3, 4], layout, DataType.Int32)
    rhs = make_cpu_tensor([10, 20, 30, 40], layout, DataType.Int32)

    add_result = lhs + rhs
    mul_result = lhs * rhs

    assert add_result.dtype() is DataType.Int32
    assert tensor_values(add_result) == [11, 22, 33, 44]
    assert add_result.autograd_ctx is None
    assert mul_result.dtype() is DataType.Int32
    assert tensor_values(mul_result) == [10, 40, 90, 160]
    assert mul_result.autograd_ctx is None


def test_cpu_mixed_int32_float32_promotes_and_only_float_accumulates_grad():
    layout = Layout(Shape(2), Stride(1))
    int_tensor = make_cpu_tensor([1, 2], layout, DataType.Int32)
    float_tensor = make_cpu_tensor([10.0, 20.0], layout)

    result = int_tensor + float_tensor
    result.backward(make_cpu_tensor([3.0, 4.0], layout))
    float_grad = require_grad(float_tensor)

    assert result.dtype() is DataType.Float32
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

    assert result.data is not lhs.data
    assert result.data is not rhs.data
    assert tensor_values(result) == pytest.approx([11.0, 22.0, 33.0, 44.0])


def test_cpu_operation_reads_external_float32_pointer_storage():
    values = array("f", [1.0, 2.0, 3.0, 4.0])
    data = CPU(len(values), values.buffer_info()[0])
    tensor = Tensor(data, 0, Layout(Shape([2, 2]), Stride([1, 2])))

    result = tensor * 2
    values[1] = 20.0
    updated_result = tensor * 2

    assert tensor_values(result) == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert tensor_values(updated_result) == pytest.approx([2.0, 40.0, 6.0, 8.0])


def test_cpu_tensor_view_uses_generic_operation_and_shares_storage():
    data = make_cpu_data(range(50))
    layout = Layout(Shape([5, 10]), Stride([1, 5]))
    tensor = Tensor(data, 0, layout)

    view = tensor[2, 2:5]

    assert view.data is data
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
    assert type(tensor_grad.data) is CPU


def test_cpu_scalar_mul_uses_expanded_keys_for_hierarchical_strides():
    layout = Layout(Shape([2, [3, 2]]), Stride([1, [10, 3]]))
    tensor = make_cpu_tensor(range(layout._cache.cosize), layout)

    result = tensor * 2

    assert tensor_values(result) == pytest.approx(
        [tensor[i] * 2 for i in range(tensor.size())]
    )


def test_cpu_int32_scalar_mul_promotes_for_non_integral_scalar():
    layout = Layout(Shape(3), Stride(1))
    tensor = make_cpu_tensor([2, 3, 4], layout, DataType.Int32)

    int_result = tensor * 3
    float_result = tensor * 2.5

    assert int_result.dtype() is DataType.Int32
    assert tensor_values(int_result) == [6, 9, 12]
    assert int_result.autograd_ctx is None
    assert float_result.dtype() is DataType.Float32
    assert tensor_values(float_result) == pytest.approx([5.0, 7.5, 10.0])
    assert float_result.autograd_ctx is None


def test_cpu_elementwise_mul_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 3]), Stride([1, 4]))
    lhs = make_cpu_tensor(range(layout._cache.cosize), layout)
    rhs = make_cpu_tensor(range(10, 10 + layout._cache.cosize), layout)
    gradient = make_cpu_tensor([1.0] * layout._cache.cosize, layout)

    result = lhs * rhs
    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert type(result.autograd_ctx).__name__ == "_CPUElementwiseMulOperation"
    assert tensor_values(result) == pytest.approx(
        [lhs[i] * rhs[i] for i in range(lhs.size())]
    )
    assert tensor_values(lhs_grad) == pytest.approx([rhs[i] for i in range(rhs.size())])
    assert tensor_values(rhs_grad) == pytest.approx([lhs[i] for i in range(lhs.size())])
    assert type(lhs_grad.data) is CPU
    assert type(rhs_grad.data) is CPU


def test_cpu_int32_non_integer_result_operations_promote_to_float32():
    layout = Layout(Shape(2), Stride(1))
    lhs = make_cpu_tensor([2, 3], layout, DataType.Int32)
    rhs = make_cpu_tensor([4, 2], layout, DataType.Int32)

    div_result = lhs / rhs
    exp_result = neotorch.exp(lhs)
    sigmoid_result = neotorch.sigmoid(lhs)
    pow_result = lhs**-1

    assert div_result.dtype() is DataType.Float32
    assert tensor_values(div_result) == pytest.approx([0.5, 1.5])
    assert exp_result.dtype() is DataType.Float32
    assert tensor_values(exp_result) == pytest.approx([math.exp(2), math.exp(3)])
    assert sigmoid_result.dtype() is DataType.Float32
    assert tensor_values(sigmoid_result) == pytest.approx(
        [1.0 / (1.0 + math.exp(-2)), 1.0 / (1.0 + math.exp(-3))]
    )
    assert pow_result.dtype() is DataType.Float32
    assert tensor_values(pow_result) == pytest.approx([0.5, 1 / 3])
    assert div_result.autograd_ctx is None
    assert exp_result.autograd_ctx is None
    assert sigmoid_result.autograd_ctx is None
    assert pow_result.autograd_ctx is None


def test_cpu_int32_pow_relu_reduce_and_matmul_preserve_int32():
    layout = Layout(Shape(3), Stride(1))
    tensor = make_cpu_tensor([-2, 3, 4], layout, DataType.Int32)
    reduce_tensor = make_cpu_tensor(
        [1, 2, 3, 4, 5, 6], Layout(Shape([2, 3]), Stride([1, 2])), DataType.Int32
    )
    lhs = make_cpu_tensor(
        [1, 2, 3, 4, 5, 6], Layout(Shape([2, 3]), Stride([1, 2])), DataType.Int32
    )
    rhs = make_cpu_tensor(
        [1, 0, 0, 1, 0, 1], Layout(Shape([2, 3]), Stride([1, 2])), DataType.Int32
    )

    pow_result = tensor**2
    relu_result = neotorch.relu(tensor)
    reduce_result = neotorch.reduce(reduce_tensor)
    matmul_result = lhs @ rhs

    assert pow_result.dtype() is DataType.Int32
    assert tensor_values(pow_result) == [4, 9, 16]
    assert relu_result.dtype() is DataType.Int32
    assert tensor_values(relu_result) == [0, 3, 4]
    assert reduce_result.dtype() is DataType.Int32
    assert tensor_values(reduce_result) == [9, 12]
    assert matmul_result.dtype() is DataType.Int32
    assert tensor_values(matmul_result) == [1, 2, 8, 10]
    assert pow_result.autograd_ctx is None
    assert relu_result.autograd_ctx is None
    assert reduce_result.autograd_ctx is None
    assert matmul_result.autograd_ctx is None


def test_cpu_int32_relu_preserves_large_values_without_float_rounding():
    layout = Layout(Shape(3), Stride(1))
    max_int32 = 2**31 - 1
    tensor = make_cpu_tensor(
        [max_int32, max_int32 - 1, -max_int32], layout, DataType.Int32
    )

    result = neotorch.relu(tensor)

    assert result.dtype() is DataType.Int32
    assert tensor_values(result) == [max_int32, max_int32 - 1, 0]


def test_cpu_int32_operations_raise_on_overflow():
    one_mode = Layout(Shape(1), Stride(1))
    matmul_layout = Layout(Shape([1, 1]), Stride([1, 1]))
    two_mode = Layout(Shape([1, 2]), Stride([1, 1]))
    max_int32 = 2**31 - 1

    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([max_int32], one_mode, DataType.Int32) + make_cpu_tensor(
            [1], one_mode, DataType.Int32
        )
    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([max_int32], one_mode, DataType.Int32) * 2
    with pytest.raises(OverflowError):
        _ = make_cpu_tensor([50_000], one_mode, DataType.Int32) * make_cpu_tensor(
            [50_000], one_mode, DataType.Int32
        )
    with pytest.raises(OverflowError):
        neotorch.reduce(make_cpu_tensor([max_int32, 1], two_mode, DataType.Int32))
    with pytest.raises(OverflowError):
        _ = make_cpu_tensor(
            [max_int32], matmul_layout, DataType.Int32
        ) @ make_cpu_tensor([2], matmul_layout, DataType.Int32)


def test_cpu_int32_hierarchical_layout_uses_expanded_keys():
    # This layout has a storage gap, so the kernel must iterate logical expanded
    # keys instead of assuming raw contiguous storage order.
    layout = Layout(Shape([[2, 2]]), Stride([[1, 3]]))
    tensor = make_cpu_tensor_with_logical_values([1, 2, 3, 4], layout, DataType.Int32)

    result = tensor * 2

    assert result.dtype() is DataType.Int32
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
    assert type(lhs_grad.data) is CPU
    assert type(rhs_grad.data) is CPU


def test_cpu_exp_uses_native_operation_and_backpropagates():
    layout = Layout(Shape([2, 2]), Stride([1, 2]))
    tensor = make_cpu_tensor([0.0, 1.0, 2.0, 3.0], layout)
    gradient = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], layout)

    result = neotorch.exp(tensor)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    expected = [math.exp(value) for value in [0.0, 1.0, 2.0, 3.0]]
    assert type(result.autograd_ctx).__name__ == "_CPUExpOperation"
    assert tensor_values(result) == pytest.approx(expected)
    assert tensor_values(tensor_grad) == pytest.approx(
        [grad * value for grad, value in zip([1.0, 2.0, 3.0, 4.0], expected)]
    )
    assert type(tensor_grad.data) is CPU


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
    assert type(tensor_grad.data) is CPU


def test_cpu_reduce_sums_second_mode_and_backpropagates():
    tensor = make_cpu_tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Layout(Shape([2, 3]), Stride([1, 2])),
    )

    result = neotorch.reduce(tensor)
    gradient = make_cpu_tensor([10.0, 20.0], result.layout)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert result.layout == Layout(Shape(2), Stride(1))
    assert tensor_values(result) == pytest.approx([9.0, 12.0])
    assert type(result.autograd_ctx).__name__ == "_CPUReduceSumOperation"
    assert tensor_values(tensor_grad) == pytest.approx([10, 20, 10, 20, 10, 20])
    assert type(tensor_grad.data) is CPU


def test_cpu_reduce_uses_expanded_keys_for_hierarchical_modes():
    layout = Layout(Shape([[2, 2], [3, 2]]), Stride([[1, 5], [20, 7]]))
    tensor = make_cpu_tensor(range(layout._cache.cosize), layout)

    result = neotorch.reduce(tensor)
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
    assert type(a_grad.data) is CPU
    assert type(b_grad.data) is CPU


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
    data = make_cpu_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor = Tensor(data, 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = neotorch.permute(tensor, 1, 0)

    assert result.data is data
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, PermuteOperation)


def test_cpu_int32_pow_handles_large_exponents():
    tensor = make_cpu_tensor([1, 0, -1], Layout(Shape(3), Stride(1)), DataType.Int32)

    even = neotorch.pow(tensor, 2**30)
    # The exponent is carried as float32, so the odd exponent must stay
    # within float32's exact-integer range.
    odd = neotorch.pow(tensor, 2**24 - 1)

    assert even.dtype() is DataType.Int32
    assert [even[0], even[1], even[2]] == [1, 0, 1]
    assert [odd[0], odd[1], odd[2]] == [1, 0, -1]


def test_cpu_int32_pow_overflow_raises():
    tensor = make_cpu_tensor([3], Layout(Shape(1), Stride(1)), DataType.Int32)

    with pytest.raises(OverflowError):
        neotorch.pow(tensor, 40)


def test_cpu_bool_scalar_multiplies_as_float():
    tensor = make_cpu_tensor([3], Layout(Shape(1), Stride(1)), DataType.Int32)

    result = neotorch.mul(tensor, True)

    assert result.dtype() is DataType.Float32
    assert result[0] == 3.0
