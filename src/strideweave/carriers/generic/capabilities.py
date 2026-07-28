"""Generic's declaration of the operation plans it executes faithfully.

Generic is the policy's reference backend, so its declared capabilities are
close to — but deliberately not identical with — everything the central resolver
can produce. Two independent conditions have to hold for a shape to be declared:
the central policy must actually resolve to it for dtypes Generic stores, and
Generic's own primitives must be able to assemble it
(:func:`~strideweave.carriers.generic.execution.executable_plan_shape`). A
future policy revision that plans an encoding Generic has no storage or
arithmetic for therefore produces no capability, and Generic refuses the plan
instead of quietly executing the nearest shape it does implement.

Declaring by resolution rather than by hand is what keeps the advertised set and
the executable set one decision: a shape reaches this module only by being
resolved exactly as an operation would resolve it at run time.
"""

from __future__ import annotations

from typing import Final

from ..dtype import SimpleDType
from ..operation_capability import OperationCapability
from ..operation_policy import SUPPORTED_TENSOR_DTYPES, resolvable_plans
from .execution import executable_plan_shape
from .numerics import is_concrete_simple_dtype

__all__ = ["generic_capabilities"]

# The dtypes a declaration may name in a tensor position: those the central
# policy plans for, narrowed to those Generic actually stores. An encoding a
# future policy plans but Generic has no storage for produces no capability.
_TENSOR_DTYPES: Final[tuple[SimpleDType, ...]] = tuple(
    dtype for dtype in SUPPORTED_TENSOR_DTYPES if is_concrete_simple_dtype(dtype)
)


def generic_capabilities() -> tuple[OperationCapability, ...]:
    """Return the capabilities Generic declares, in a deterministic order.

    Returns:
        One capability per distinct plan shape the policy resolves over
        Generic's storage dtypes and Generic's primitives can assemble.

    Examples:
        >>> from strideweave.carriers.generic.capabilities import generic_capabilities
        >>> "matmul" in {entry.operation for entry in generic_capabilities()}
        True
    """
    return tuple(
        OperationCapability.from_plan(plan)
        for plan in resolvable_plans(_TENSOR_DTYPES)
        if executable_plan_shape(plan)
    )
