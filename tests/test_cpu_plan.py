"""Decisions native CPU no longer makes on its own.

CPU holds no promotion table of its own (RT013). That every registered
operation produces the plan's dtype, on both backends, is enumerated in
``tests/test_dtype_conformance.py``; what remains here are the specific
native behaviors the routing changed or that only CPU can exhibit.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

import strideweave as sw
from strideweave import CPU, DType, Layout, Shape, Stride, Tensor
from strideweave.carriers.operation_capability import (
    UnsupportedOperationPlan,
    capabilities_for_carrier_class,
    require_capability,
    supports_operation_plan,
)
from strideweave.carriers.operation_policy import (
    Accumulation,
    Arithmetic,
    OperandRole,
    resolve_operation_plan,
)

ONE_MODE = Layout(Shape(4), Stride(1))
TWO_MODE = Layout(Shape([2, 2]), Stride([1, 2]))


def cpu_tensor(dtype: DType, layout: Layout) -> Tensor:
    carrier = CPU(4, dtype=dtype)
    values = (True, False, True, False) if dtype is DType.Bool else (1, 2, 3, 4)
    for index, value in enumerate(values):
        carrier[index] = value
    return Tensor(carrier, 0, layout)


def test_int32_pow_with_a_float_valued_exponent_is_floating():
    # A weak float exponent takes the floating path even when its value happens
    # to be integral: the exponent's kind, not its value, selects the plan.
    tensor = cpu_tensor(DType.Int32, ONE_MODE)

    result = sw.pow(tensor, 2.0)

    assert result.dtype() is DType.Float32
    assert result[1] == pytest.approx(4.0)


def test_int32_pow_with_a_bool_exponent_is_floating():
    tensor = cpu_tensor(DType.Int32, ONE_MODE)

    assert sw.pow(tensor, True).dtype() is DType.Float32


def test_int32_pow_uses_the_exact_exponent_beyond_float_precision():
    # 2**24 + 1 is not representable in binary32, so an exponent carried as a
    # float would round to an even one and lose the sign of a negative base.
    carrier = CPU(1, dtype=DType.Int32)
    carrier[0] = -1
    tensor = Tensor(carrier, 0, Layout(Shape(1), Stride(1)))

    result = sw.pow(tensor, 2**24 + 1)

    assert result.dtype() is DType.Int32
    assert result[0] == -1


def test_plan_errors_are_raised_before_any_kernel_runs():
    tensor = cpu_tensor(DType.Int32, ONE_MODE)

    # Resolution precedes the kernel, so a scalar the integer plan cannot
    # represent is rejected by the policy rather than by a native cast.
    with pytest.raises(OverflowError, match="out of int32 range"):
        sw.mul(tensor, 2**40)
    with pytest.raises(TypeError, match="must be a real Python number"):
        sw.mul(tensor, "2")
    with pytest.raises(TypeError, match="must be a real Python number"):
        sw.pow(tensor, None)


def test_unregistered_operations_keep_their_own_dispatch():
    # Representation-preserving operations are deliberately unplanned, so CPU
    # still reaches them without consulting the policy.
    tensor = cpu_tensor(DType.Int32, TWO_MODE)

    result = sw.permute(tensor, 1, 0)

    assert result.dtype() is DType.Int32
    assert result.carrier is tensor.carrier


# --- The capability boundary ------------------------------------------------
#
# A resolved plan is not permission to run: CPU executes only the shapes its
# carrier class declared, and refuses everything else before it allocates the
# result or releases the GIL.


def test_cpu_declares_only_shapes_its_kernels_implement():
    # An accumulating kernel needs an accumulation and an elementwise kernel has
    # nowhere to apply one, so the two are declared apart rather than left to a
    # branch to sort out.
    for operation in ("reduce_sum", "matmul"):
        for capability in capabilities_for_carrier_class(CPU, operation):
            assert capability.accumulation is not None
    accumulating_operations = {
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
        "cumsum",
        "matmul",
        "conv_general",
        "scatter_add",
    }
    for capability in capabilities_for_carrier_class(CPU):
        if capability.operation not in accumulating_operations:
            assert capability.accumulation is None


def test_cpu_declares_both_floating_accumulator_plans():
    # The accumulator dtype is an exact capability dimension, so CPU declares
    # one capability per accumulator its kernels implement rather than one
    # floating capability a kernel then reinterprets.
    for operation in ("reduce_sum", "matmul"):
        floating = {
            capability.accumulator_dtype
            for capability in capabilities_for_carrier_class(CPU, operation)
            if capability.compute is Arithmetic.BINARY32
        }
        assert floating == {DType.Float32, DType.Float64}


def _float32_row(values):
    carrier = CPU(len(values), dtype=DType.Float32)
    for index, value in enumerate(values):
        carrier[index] = value
    return Tensor(carrier, 0, Layout(Shape([1, len(values)]), Stride([1, 1])))


# Summing these four Float32 terms left to right cancels to exactly zero in
# binary32 and to exactly 2.0 in binary64, so the two accumulators are
# distinguishable from identical encoded storage.
CANCELLING_TERMS = (2**24, 1.0, 1.0, -(2**24))


def test_public_cpu_reduce_sum_widens_only_the_accumulation():
    tensor = _float32_row(CANCELLING_TERMS)
    stored = [tensor[0, index] for index in range(4)]

    widened = sw.reduce_sum(tensor, "a b -> a", accumulator_dtype=DType.Float64)
    explicit = sw.reduce_sum(tensor, "a b -> a", accumulator_dtype=DType.Float32)
    default = sw.reduce_sum(tensor, "a b -> a")

    assert [tensor[0, index] for index in range(4)] == stored
    assert tensor.dtype() is widened.dtype() is DType.Float32
    assert widened[0] == 2.0
    assert explicit[0] == default[0] == 0.0


def test_public_cpu_matmul_widens_only_the_accumulation():
    lhs = _float32_row(CANCELLING_TERMS)
    rhs = _float32_row((1.0, 1.0, 1.0, 1.0))

    widened = sw.matmul(lhs, rhs, accumulator_dtype=DType.Float64)
    default = sw.matmul(lhs, rhs)

    assert widened.dtype() is default.dtype() is DType.Float32
    assert widened[0, 0] == 2.0
    assert default[0, 0] == 0.0


def test_cpu_accumulator_selection_is_execution_local():
    tensor = _float32_row(CANCELLING_TERMS)

    widened = sw.reduce_sum(tensor, "a b -> a", accumulator_dtype=DType.Float64)
    default = sw.reduce_sum(tensor, "a b -> a")
    widened_again = sw.reduce_sum(tensor, "a b -> a", accumulator_dtype=DType.Float64)

    assert widened[0] == 2.0
    assert default[0] == 0.0
    assert widened_again[0] == 2.0


def test_cpu_matmul_backward_reuses_the_forward_accumulator():
    # Backward contracts the same terms, so a forward that asked for the wider
    # accumulator must not silently narrow its gradient contraction.
    lhs_carrier = CPU(1, dtype=DType.Float32)
    lhs_carrier[0] = 1.0
    lhs = Tensor(lhs_carrier, 0, Layout(Shape([1, 1]), Stride([1, 1])))
    rhs_carrier = CPU(4, dtype=DType.Float32)
    for index, value in enumerate(CANCELLING_TERMS):
        rhs_carrier[index] = value
    rhs = Tensor(rhs_carrier, 0, Layout(Shape([4, 1]), Stride([1, 1])))

    result = sw.matmul(lhs, rhs, accumulator_dtype=DType.Float64)
    gradient_carrier = CPU(4, dtype=DType.Float32)
    for index in range(4):
        gradient_carrier[index] = 1.0
    result.backward(Tensor(gradient_carrier, 0, result.layout))

    assert lhs.grad is not None
    assert lhs.grad.dtype() is DType.Float32
    assert lhs.grad[0, 0] == 2.0


def test_cpu_rejects_an_unimplemented_accumulator_before_allocation():
    # An accumulator dtype CPU has no kernel for is refused by the same exact
    # capability match, not approximated by the nearest floating kernel.
    unsupported = dataclasses.replace(
        resolve_operation_plan("reduce_sum", DType.Float32),
        accumulator_dtype=DType.Int32,
    )

    assert not supports_operation_plan(CPU, unsupported)
    with pytest.raises(UnsupportedOperationPlan, match="CPU declares no"):
        require_capability(CPU, unsupported)


@pytest.mark.parametrize("operation", ["reduce_sum", "matmul"])
def test_an_accumulating_plan_without_an_accumulation_is_unsupported(operation):
    # The confirmed bug this boundary exists for: an Int32 reduce plan carrying
    # no accumulation entered the floating branch and wrote Float32 bit patterns
    # into Int32 storage.
    operands = (DType.Int32,) * (2 if operation == "matmul" else 1)
    resolved = resolve_operation_plan(operation, *operands)
    stripped = dataclasses.replace(resolved, accumulation=None)

    assert supports_operation_plan(CPU, resolved)
    assert not supports_operation_plan(CPU, stripped)
    with pytest.raises(UnsupportedOperationPlan, match="CPU declares no"):
        require_capability(CPU, stripped)


@pytest.mark.parametrize(
    "overrides",
    [
        {"compute": Arithmetic.BINARY32},
        {"output": DType.Float32},
        {"accumulation": Accumulation.FLOATING},
    ],
    ids=["compute", "output", "accumulation"],
)
def test_a_plan_disagreeing_with_the_kernel_is_unsupported(overrides):
    integer_reduce = resolve_operation_plan("reduce_sum", DType.Int32)

    assert not supports_operation_plan(
        CPU, dataclasses.replace(integer_reduce, **overrides)
    )


def test_an_injected_unsupported_plan_never_reaches_a_kernel():
    # The native bridge resolves and then requires a capability while it still
    # holds the GIL, so a plan the policy would never produce is refused before
    # the result carrier exists rather than mis-executed into Int32 storage.
    source = """
import dataclasses

import strideweave as sw
from strideweave import CPU, DType, Layout, Shape, Stride, Tensor
from strideweave.carriers import operation_policy
from strideweave.carriers.operation_capability import UnsupportedOperationPlan

resolve = operation_policy.resolve_operation_plan


def injected(operation, *operands, **options):
    plan = resolve(operation, *operands, **options)
    if operation == "reduce_sum":
        return dataclasses.replace(plan, accumulation=None)
    return plan


operation_policy.resolve_operation_plan = injected

carrier = CPU(4, dtype=DType.Int32)
for index, value in enumerate((1, 2, 3, 4)):
    carrier[index] = value
tensor = Tensor(carrier, 0, Layout(Shape([2, 2]), Stride([1, 2])))

try:
    sw.reduce_sum(tensor, "a b -> a")
except UnsupportedOperationPlan as error:
    assert "CPU declares no" in str(error), error
else:
    raise AssertionError("an undeclared reduce plan reached a kernel")
"""

    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_every_declared_capability_executes_on_a_real_tensor():
    # Advertised support and executable support are one set: every capability
    # CPU declares reaches a kernel that produces exactly its output dtype.
    executed = 0
    for capability in capabilities_for_carrier_class(CPU):
        arguments = _arguments_for(capability)
        if arguments is None:
            continue
        result = _execute_capability(capability.operation, arguments)
        executed += 1

        assert result.dtype() is capability.output
    assert executed == len(capabilities_for_carrier_class(CPU))


def _arguments_for(capability):
    """Build real operands matching one capability's operand shape."""
    operation = capability.operation
    if operation in {"_sort_values", "_sort_indices", "_topk_values", "_topk_indices"}:
        return (cpu_tensor(DType.Float32, ONE_MODE),)
    if operation == "conv_general":
        lhs = cpu_tensor(DType.Float32, Layout(Shape([1, 1, 2]), Stride([1, 1, 1])))
        kernel = cpu_tensor(DType.Float32, Layout(Shape([1, 1, 1]), Stride([1, 1, 1])))
        return (lhs, kernel, (1,), ((0, 0),))
    if operation == "gather":
        source = cpu_tensor(DType.Float32, Layout(Shape([2, 2]), Stride([1, 2])))
        indices = cpu_tensor(DType.Int32, Layout(Shape(1), Stride(1)))
        return (source, indices, 0)
    if operation in {"scatter", "scatter_add"}:
        base = cpu_tensor(DType.Float32, TWO_MODE)
        indices = cpu_tensor(DType.Int32, Layout(Shape(1), Stride(1)))
        updates = cpu_tensor(DType.Float32, Layout(Shape([1, 2]), Stride([1, 1])))
        return (base, indices, updates, 0)

    reducing_operations = {
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
    }
    layout = TWO_MODE if operation in reducing_operations | {"matmul"} else ONE_MODE
    arguments = []
    for operand in capability.operands:
        if operand.role is OperandRole.TENSOR:
            arguments.append(cpu_tensor(operand.dtype, layout))
        else:
            arguments.append(3 if operand.convert_to is DType.Int32 else 0.5)
    if operation in reducing_operations:
        arguments.append("a b -> a")
    elif operation == "cumsum":
        arguments.append(0)
    return tuple(arguments)


def _execute_capability(operation: str, arguments: tuple[object, ...]):
    """Call public wrappers for private sort/top-k capability names."""
    tensor = arguments[0] if arguments else None
    if operation == "_sort_values":
        assert isinstance(tensor, Tensor)
        return sw.sort(tensor).values
    if operation == "_sort_indices":
        assert isinstance(tensor, Tensor)
        return sw.sort(tensor).indices
    if operation == "_topk_values":
        assert isinstance(tensor, Tensor)
        return sw.topk(tensor, 2).values
    if operation == "_topk_indices":
        assert isinstance(tensor, Tensor)
        return sw.topk(tensor, 2).indices
    return getattr(sw, operation)(*arguments)
