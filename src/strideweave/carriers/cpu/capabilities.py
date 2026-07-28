"""Native CPU's declaration of the operation plans its kernels execute.

A CPU kernel makes three structural commitments a plan has to agree with. It
reads and writes one element type for the whole loop, chosen from the plan's
``compute``; it allocates its result from ``output``; and it either combines
terms or does not, decided by which kernel the operation dispatches to rather
than by anything in the plan. A plan disagreeing with any of those describes a
loop this backend has not written, so it is declared here as unsupported and
refused before allocation rather than routed to the nearest branch that
compiles.

The declarations are derived by resolving the central policy and filtering it
through :func:`executable_plan_shape`, so the shapes CPU advertises, the shapes
its native bridge accepts, and the shapes its kernels implement are one
decision. Native code consults these entries through
:func:`strideweave.carriers.operation_capability.require_capability` while the
GIL is still held, before it allocates or releases (``CPP001``).
"""

from __future__ import annotations

from typing import Final

from ..dtype import DType, SimpleDType
from ..operation_capability import OperationCapability
from ..operation_policy import (
    SUPPORTED_TENSOR_DTYPES,
    Accumulation,
    Arithmetic,
    OperationPlan,
    resolvable_plans,
)

__all__ = ["cpu_capabilities", "executable_plan_shape"]

# The dtypes CPU stores, and therefore the only dtypes a declaration may name in
# a tensor position or as a result (`RT012`).
_STORAGE_DTYPES: Final[tuple[SimpleDType, ...]] = (DType.Float32, DType.Int32)

# The operations whose kernels combine terms. Every other kernel writes one
# result per element and has nowhere to apply an accumulation, so declaring one
# for it would advertise a loop that does not exist.
_ACCUMULATING_OPERATIONS: Final[frozenset[str]] = frozenset({"reduce", "matmul"})

# The element type each compute arithmetic runs a kernel in.
_COMPUTE_DTYPES: Final[dict[Arithmetic, SimpleDType]] = {
    Arithmetic.BINARY32: DType.Float32,
    Arithmetic.INT32_EXACT_CHECKED: DType.Int32,
    Arithmetic.INT32_EXACT: DType.Int32,
}

# The accumulator each compute arithmetic's combining kernel uses: a sequential
# binary32 sum, or the exact 128-bit integer accumulator narrowed once at the
# end.
_COMPUTE_ACCUMULATIONS: Final[dict[Arithmetic, Accumulation]] = {
    Arithmetic.BINARY32: Accumulation.SEQUENTIAL_BINARY32,
    Arithmetic.INT32_EXACT_CHECKED: Accumulation.EXACT_INTEGER,
    Arithmetic.INT32_EXACT: Accumulation.EXACT_INTEGER,
}


def executable_plan_shape(plan: OperationPlan) -> bool:
    """Report whether a CPU kernel exists for ``plan`` exactly as written.

    This asks only what the native backend implements. Which plan is correct for
    a given operand pair is central policy, decided before a plan reaches a
    kernel.

    Args:
        plan: A resolved operation plan.

    Returns:
        ``True`` when the operands convert uniformly into the element type
        ``compute`` names, the result is stored in that same type, and the
        accumulation is present exactly for the operations whose kernels
        combine terms.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.cpu.capabilities import executable_plan_shape
        >>> executable_plan_shape(resolve_operation_plan("reduce", DType.Int32))
        True
    """
    element = _COMPUTE_DTYPES.get(plan.compute)
    if element is None:
        return False
    if any(operand.convert_to is not element for operand in plan.operands):
        return False
    # `_COMPUTE_DTYPES` names only dtypes CPU stores, so agreeing with `element`
    # is already agreeing with `_STORAGE_DTYPES` (`RT012`).
    if plan.output is not element:
        return False
    if plan.operation not in _ACCUMULATING_OPERATIONS:
        return plan.accumulation is None
    return plan.accumulation is _COMPUTE_ACCUMULATIONS[plan.compute]


def cpu_capabilities() -> tuple[OperationCapability, ...]:
    """Return the capabilities native CPU declares, in a deterministic order.

    Returns:
        One capability per distinct plan shape the policy resolves over CPU's
        storage dtypes and a CPU kernel implements.

    Examples:
        >>> from strideweave.carriers.cpu.capabilities import cpu_capabilities
        >>> "matmul" in {entry.operation for entry in cpu_capabilities()}
        True
    """
    return tuple(
        OperationCapability.from_plan(plan)
        for plan in resolvable_plans(
            tuple(
                dtype for dtype in SUPPORTED_TENSOR_DTYPES if dtype in _STORAGE_DTYPES
            )
        )
        if executable_plan_shape(plan)
    )
