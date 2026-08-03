"""Stage Two verification of ordinary CPU Float32 target execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import strideweave as sw
from strideweave import DType, Layout, Shape, Stride
from strideweave.carriers.operation_policy import operation_execution_options

from .comparison import compare_float32, gamma_bound
from .model import (
    Deviations,
    KernelDescriptor,
    Tolerance,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
)
from .payloads import EncodedFloat32Payload
from .stage_one import (
    StageOneResult,
    _case,
    _error_record,
    _movement_records,
    _tensor,
    _values,
)

CaseKind = Literal["structural", "numerical"]
OperationName = Literal["reduce_sum", "matmul"]


@dataclass(frozen=True, slots=True)
class _TargetCase:
    """One deterministic CPU target/oracle contraction witness."""

    case_id: str
    operation: OperationName
    kind: CaseKind
    lhs_layout: Layout
    rhs_layout: Layout | None = None


def _reduce_second_mode(tensor, *, accumulator_dtype: DType | None = None):
    """Reduce a two-mode tensor through its ordinary public capability.

    Args:
        tensor: Two-mode CPU tensor to reduce over its second mode.
        accumulator_dtype: Floating accumulator to request, or ``None`` for the
            backend default.

    Returns:
        The reduced tensor.
    """
    operation = tensor.carrier.dispatch_op("reduce_sum")
    if accumulator_dtype is None:
        return operation.forward(tensor)
    options = operation_execution_options(
        "reduce_sum", accumulator_dtype=accumulator_dtype
    )
    return operation.forward(tensor, options=options)


def _matrix_shape(layout: Layout) -> tuple[int, int]:
    if len(layout.shape) != 2:
        raise ValueError("Stage Two contractions require two-mode layouts")
    return (layout.shape[0].size, layout.shape[1].size)


def _matrix_values(
    shape: tuple[int, int], value_at: Callable[[int, int], float]
) -> tuple[float, ...]:
    rows, columns = shape
    return tuple(
        value_at(row, column) for column in range(columns) for row in range(rows)
    )


def _structural_values(
    operation: OperationName,
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int] | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    lhs = _matrix_values(
        lhs_shape, lambda row, column: float((3 * row + 2 * column) % 7 - 3)
    )
    if operation == "reduce_sum":
        return lhs, None
    if rhs_shape is None:
        raise ValueError("Stage Two matmul case requires a right-hand layout")
    rhs = _matrix_values(
        rhs_shape, lambda row, column: float((2 * row - column) % 5 - 2)
    )
    return lhs, rhs


def _numerical_values(
    operation: OperationName,
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int] | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    cancellation = (float(2**24), 1.0, 1.0, float(-(2**24)))
    lhs = _matrix_values(
        lhs_shape,
        lambda row, column: (
            cancellation[column % len(cancellation)] * (1.0 if row % 2 == 0 else -1.0)
        ),
    )
    if operation == "reduce_sum":
        return lhs, None
    if rhs_shape is None:
        raise ValueError("Stage Two matmul case requires a right-hand layout")
    rhs = _matrix_values(
        rhs_shape,
        lambda row, column: 1.0 if (row + column) % 2 == 0 else -1.0,
    )
    return lhs, rhs


def _target_cases() -> tuple[_TargetCase, ...]:
    flat_reduce = Layout(Shape([3, 16]), Stride([16, 1]))
    hierarchical_reduce = Layout(Shape([[2, 2], 12]), Stride([[12, 24], 1]))
    flat_lhs = Layout(Shape([3, 16]), Stride([16, 1]))
    flat_rhs = Layout(Shape([2, 16]), Stride([16, 1]))
    hierarchical_lhs = Layout(Shape([[2, 2], 12]), Stride([[12, 24], 1]))
    hierarchical_rhs = Layout(Shape([[3, 1], 12]), Stride([[12, 36], 1]))
    return (
        _TargetCase("reduce-flat-structural", "reduce_sum", "structural", flat_reduce),
        _TargetCase(
            "reduce-hierarchical-numerical",
            "reduce_sum",
            "numerical",
            hierarchical_reduce,
        ),
        _TargetCase(
            "matmul-flat-structural",
            "matmul",
            "structural",
            flat_lhs,
            flat_rhs,
        ),
        _TargetCase(
            "matmul-hierarchical-numerical",
            "matmul",
            "numerical",
            hierarchical_lhs,
            hierarchical_rhs,
        ),
    )


def _blocked(kernel: KernelDescriptor, test_class: VerificationClass, suffix: str):
    payload = EncodedFloat32Payload.from_values(())
    return replace(
        _case(
            kernel,
            test_class,
            f"{kernel.kernel_id}-{suffix}",
            (payload,),
            VerificationOutcome.BLOCKED,
            Deviations(0.0, 0.0, 0),
            0,
            diagnostic="Stage One Float64 oracle certificate is absent",
        ),
        stage=VerificationStage.TARGET,
    )


def _maximum_term_sum(
    operation: OperationName,
    lhs_values: tuple[float, ...],
    lhs_shape: tuple[int, int],
    rhs_values: tuple[float, ...] | None,
    rhs_shape: tuple[int, int] | None,
) -> float:
    lhs_rows, k = lhs_shape
    if operation == "reduce_sum":
        return max(
            sum(abs(lhs_values[row + lhs_rows * column]) for column in range(k))
            for row in range(lhs_rows)
        )
    if rhs_values is None or rhs_shape is None:
        raise ValueError("Stage Two matmul numerical case requires right-hand values")
    rhs_rows, rhs_k = rhs_shape
    if rhs_k != k:
        raise ValueError("Stage Two matmul contraction lengths must agree")
    return max(
        sum(
            abs(lhs_values[row + lhs_rows * column])
            * abs(rhs_values[column_row + rhs_rows * column])
            for column in range(k)
        )
        for row in range(lhs_rows)
        for column_row in range(rhs_rows)
    )


def _target_case(kernel: KernelDescriptor, target_case: _TargetCase):
    if kernel.operation == "reduce_sum":
        operation: OperationName = "reduce_sum"
    elif kernel.operation == "matmul":
        operation = "matmul"
    else:
        raise ValueError(f"unsupported Stage Two contraction {kernel.operation!r}")
    lhs_shape = _matrix_shape(target_case.lhs_layout)
    rhs_shape = (
        None
        if target_case.rhs_layout is None
        else _matrix_shape(target_case.rhs_layout)
    )
    value_factory = (
        _structural_values if target_case.kind == "structural" else _numerical_values
    )
    values, rhs_values = value_factory(operation, lhs_shape, rhs_shape)
    k = lhs_shape[1]
    payload = EncodedFloat32Payload.from_values(values)
    payloads = [payload]
    if kernel.operation == "matmul":
        if rhs_values is None or target_case.rhs_layout is None:
            raise ValueError(
                "Stage Two matmul case requires right-hand values and layout"
            )
        rhs_payload = EncodedFloat32Payload.from_values(rhs_values)
        payloads.append(rhs_payload)
    test_class = (
        VerificationClass.STRUCTURAL
        if target_case.kind == "structural"
        else VerificationClass.NUMERICAL
    )
    if test_class is VerificationClass.STRUCTURAL:
        allowed = 0.0
        version = "bit-exact-structural-v1"
    else:
        allowed = gamma_bound(
            2.0**-24,
            k,
            _maximum_term_sum(operation, values, lhs_shape, rhs_values, rhs_shape),
        )
        version = "stage-two-float32-gamma-k-v1"
    shapes = (lhs_shape,) if rhs_shape is None else (lhs_shape, rhs_shape)
    try:
        lhs_target = _tensor(
            payload.values(), DType.Float32, target_case.lhs_layout, True
        )
        lhs_oracle = _tensor(
            payload.values(), DType.Float32, target_case.lhs_layout, True
        )
        if kernel.operation == "reduce_sum":
            # The production layouts are hierarchical, so the two-mode
            # reduction primitive is dispatched directly rather than lowered
            # from a description that would first rearrange the operand away.
            target = _reduce_second_mode(lhs_target)
            oracle = _reduce_second_mode(lhs_oracle, accumulator_dtype=DType.Float64)
        else:
            if target_case.rhs_layout is None or len(payloads) != 2:
                raise ValueError("Stage Two matmul payload preparation is incomplete")
            rhs_target = _tensor(
                payloads[1].values(), DType.Float32, target_case.rhs_layout, True
            )
            rhs_oracle = _tensor(
                payloads[1].values(), DType.Float32, target_case.rhs_layout, True
            )
            target = sw.matmul(lhs_target, rhs_target)
            oracle = sw.matmul(lhs_oracle, rhs_oracle, accumulator_dtype=DType.Float64)
        comparison = compare_float32(_values(oracle), _values(target))
    except (RuntimeError, ValueError) as error:
        return replace(
            _error_record(
                kernel,
                test_class,
                f"{kernel.kernel_id}-{target_case.case_id}-error",
                error,
                payloads=tuple(payloads),
                k=k,
                shapes=shapes,
                tolerance=Tolerance(absolute=allowed, version=version),
            ),
            stage=VerificationStage.TARGET,
        )
    maximum_absolute = comparison.deviations.maximum_absolute
    if maximum_absolute is None:
        raise RuntimeError("completed numerical comparison has no absolute deviation")
    passed = maximum_absolute <= allowed
    if test_class is VerificationClass.STRUCTURAL:
        passed = comparison.mismatches == 0
    record = _case(
        kernel,
        test_class,
        f"{kernel.kernel_id}-{target_case.case_id}",
        tuple(payloads),
        VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED,
        comparison.deviations,
        comparison.mismatches,
        k=k,
        diagnostic=(
            None
            if passed
            else "encoded Float32 results are not bit-identical"
            if test_class is VerificationClass.STRUCTURAL and comparison.mismatches
            else f"maximum absolute deviation exceeds {allowed}"
        ),
        shapes=shapes,
    )
    return replace(
        record,
        stage=VerificationStage.TARGET,
        tolerance=Tolerance(absolute=allowed, version=version),
    )


def run_stage_two(stage_one: StageOneResult) -> VerificationReport:
    certified = {certificate.kernel_id for certificate in stage_one.certificates}
    records = [
        replace(
            movement,
            stage=VerificationStage.TARGET,
            case=replace(movement.case, case_id=f"stage-two-{movement.case.case_id}"),
        )
        for movement in _movement_records()
    ]
    cases_by_operation = {
        operation: tuple(
            case for case in _target_cases() if case.operation == operation
        )
        for operation in ("reduce_sum", "matmul")
    }
    for operation, kernel_id in (
        ("reduce_sum", "cpu.reduce_sum"),
        ("matmul", "cpu.matmul"),
    ):
        kernel = KernelDescriptor(operation, kernel_id, "default", "")
        if kernel_id not in certified:
            records.extend(
                (
                    _blocked(
                        kernel, VerificationClass.STRUCTURAL, "structural-blocked"
                    ),
                    _blocked(kernel, VerificationClass.NUMERICAL, "numerical-blocked"),
                )
            )
            continue
        for case in cases_by_operation[operation]:
            records.append(_target_case(kernel, case))
    return VerificationReport(tuple(records))
