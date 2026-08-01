"""Focused Generic capability checks for select and clamp plans."""

from __future__ import annotations

from dataclasses import replace

from strideweave import DType
from strideweave.carriers.generic.capabilities import generic_capabilities
from strideweave.carriers.generic.execution import executable_plan_shape
from strideweave.carriers.operation_policy import OperandRole, resolve_operation_plan


def test_generic_derives_select_and_all_clamp_overloads_from_the_registry() -> None:
    entries = generic_capabilities()
    select_entries = [entry for entry in entries if entry.operation == "select"]
    clamp_entries = [entry for entry in entries if entry.operation == "clamp"]

    assert len(select_entries) == 1
    assert tuple(operand.dtype for operand in select_entries[0].operands) == (
        DType.Bool,
        DType.Float32,
        DType.Float32,
    )
    assert {
        tuple(operand.role for operand in entry.operands) for entry in clamp_entries
    } == {
        (OperandRole.TENSOR, OperandRole.TENSOR, OperandRole.TENSOR),
        (OperandRole.TENSOR, OperandRole.TENSOR, OperandRole.WEAK_SCALAR),
        (OperandRole.TENSOR, OperandRole.WEAK_SCALAR, OperandRole.TENSOR),
        (OperandRole.TENSOR, OperandRole.WEAK_SCALAR, OperandRole.WEAK_SCALAR),
    }


def test_generic_accepts_only_exact_select_and_clamp_operand_shapes() -> None:
    select = resolve_operation_plan("select", DType.Bool, DType.Float32, DType.Float32)
    assert executable_plan_shape(select)
    malformed_select = replace(
        select,
        operands=(
            replace(select.operands[0], dtype=DType.Float32),
            *select.operands[1:],
        ),
    )
    assert not executable_plan_shape(malformed_select)

    clamp_inputs = (
        (DType.Float32, DType.Float32, DType.Float32),
        (DType.Float32, DType.Float32, 1),
        (DType.Float32, 1, DType.Float32),
        (DType.Float32, 1, 2),
    )
    clamps = [resolve_operation_plan("clamp", *operands) for operands in clamp_inputs]
    assert all(executable_plan_shape(plan) for plan in clamps)
    assert not executable_plan_shape(
        replace(clamps[0], operands=clamps[0].operands[:2])
    )
