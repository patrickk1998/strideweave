from __future__ import annotations

from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.operation_capability import (
    OperationCapability,
    capabilities_for_carrier_class,
)
from strideweave.carriers.operation_policy import OperandRole

pytestmark = pytest.mark.metal

_ONE_MODE = sw.Layout(sw.Shape(4), sw.Stride(1))
_TWO_MODE = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))


def _capability_id(capability: OperationCapability) -> str:
    operand_ids: list[str] = []
    for operand in capability.operands:
        if operand.role is OperandRole.TENSOR:
            assert operand.dtype is not None
            operand_ids.append(
                f"tensor-{operand.dtype.name}-to-{operand.convert_to.name}"
            )
        else:
            operand_ids.append(f"scalar-to-{operand.convert_to.name}")
    operands = "-".join(operand_ids)
    accumulator = (
        "none"
        if capability.accumulator_dtype is None
        else capability.accumulator_dtype.name
    )
    return f"{capability.operation}-{operands}-{accumulator}-{capability.output.name}"


_METAL_CAPABILITIES = capabilities_for_carrier_class(sw.Metal)


def _metal_tensor(dtype: sw.DType, layout: sw.Layout) -> sw.Tensor:
    carrier = sw.Metal(layout.cosize, dtype=dtype)
    if dtype is sw.DType.Bool:
        values: tuple[Any, ...] = (True, False, True, False)
    elif dtype is sw.DType.Int32:
        values = (0, 1, 0, 1)
    else:
        values = (1.0, 2.0, 3.0, 4.0)
    for index in range(layout.cosize):
        carrier[index] = values[index % len(values)]
    return sw.Tensor(carrier, 0, layout)


def _arguments_for(capability: OperationCapability) -> tuple[object, ...]:
    operation = capability.operation
    if operation in {
        "_sort_values",
        "_sort_indices",
        "_topk_values",
        "_topk_indices",
    }:
        return (_metal_tensor(sw.DType.Float32, _ONE_MODE),)
    if operation == "conv_general":
        lhs = _metal_tensor(
            sw.DType.Float32,
            sw.Layout(sw.Shape([1, 1, 2]), sw.Stride([1, 1, 1])),
        )
        kernel = _metal_tensor(
            sw.DType.Float32,
            sw.Layout(sw.Shape([1, 1, 1]), sw.Stride([1, 1, 1])),
        )
        return (lhs, kernel, (1,), ((0, 0),))
    if operation == "gather":
        source = _metal_tensor(sw.DType.Float32, _TWO_MODE)
        indices = _metal_tensor(sw.DType.Int32, sw.Layout(sw.Shape(1), sw.Stride(1)))
        return (source, indices, 0)
    if operation in {"scatter", "scatter_add"}:
        base = _metal_tensor(sw.DType.Float32, _TWO_MODE)
        indices = _metal_tensor(sw.DType.Int32, sw.Layout(sw.Shape(1), sw.Stride(1)))
        updates = _metal_tensor(
            sw.DType.Float32,
            sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1])),
        )
        return (base, indices, updates, 0)

    reductions = {
        "argmax",
        "argmin",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
    }
    layout = _TWO_MODE if operation in reductions | {"matmul"} else _ONE_MODE
    arguments: list[object] = []
    for operand in capability.operands:
        if operand.role is OperandRole.TENSOR:
            assert operand.dtype is not None
            arguments.append(_metal_tensor(operand.dtype, layout))
        else:
            arguments.append(0.5)
    if operation in reductions:
        arguments.append("a b -> a")
    elif operation == "cumsum":
        arguments.append(0)
    return tuple(arguments)


def _execute_capability(operation: str, arguments: tuple[object, ...]) -> sw.Tensor:
    tensor = arguments[0] if arguments else None
    if operation == "_sort_values":
        assert isinstance(tensor, sw.Tensor)
        return sw.sort(tensor).values
    if operation == "_sort_indices":
        assert isinstance(tensor, sw.Tensor)
        return sw.sort(tensor).indices
    if operation == "_topk_values":
        assert isinstance(tensor, sw.Tensor)
        return sw.topk(tensor, 2).values
    if operation == "_topk_indices":
        assert isinstance(tensor, sw.Tensor)
        return sw.topk(tensor, 2).indices
    result = getattr(sw, operation)(*arguments)
    assert isinstance(result, sw.Tensor)
    return result


@pytest.mark.parametrize(
    "capability",
    _METAL_CAPABILITIES,
    ids=_capability_id,
)
def test_every_declared_metal_capability_executes_on_a_real_tensor(
    capability: OperationCapability,
) -> None:
    arguments = _arguments_for(capability)

    result = _execute_capability(capability.operation, arguments)
    result.carrier.get_value(result.offset)

    assert type(result.carrier) is sw.Metal
    assert result.dtype() is capability.output
    assert capability in result.carrier.operation_capabilities(capability.operation)
