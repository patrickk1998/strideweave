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

from itertools import product
from typing import Any, Final

from ..dtype import DType, SimpleDType
from ..operation_capability import OperationCapability
from ..operation_policy import (
    SUPPORTED_TENSOR_DTYPES,
    WEAK_SCALAR_PROBES,
    OperandRole,
    OperationPlan,
    registered_operations,
)
from .execution import executable_plan_shape
from .numerics import is_concrete_simple_dtype

__all__ = ["generic_capabilities"]

# The dtypes a declaration may name in a tensor position: those the central
# policy plans for, narrowed to those Generic actually stores. An encoding a
# future policy plans but Generic has no storage for produces no capability.
_DEFAULT_TENSOR_DTYPES: Final[tuple[SimpleDType, ...]] = tuple(
    dtype for dtype in SUPPORTED_TENSOR_DTYPES if is_concrete_simple_dtype(dtype)
)

# Generic additionally stores Bool. It participates only where an overload's
# registry-owned tensor domain names it explicitly (currently select's
# condition), never as a locally invented promotion candidate.
_STORAGE_DTYPES: Final[tuple[SimpleDType, ...]] = (
    DType.Float32,
    DType.Int32,
    DType.Bool,
)


def _resolvable_generic_plans() -> tuple[OperationPlan, ...]:
    """Enumerate central-registry plans over dtypes Generic can store.

    Overloads without explicit tensor domains use the policy's supported input
    dtypes. Explicit domains are intersected with Generic storage, allowing
    registry-declared Bool positions without making Bool a general arithmetic
    input. Every promotion and output decision remains owned by the overload's
    resolver.
    """

    weak_scalars: tuple[Any, ...] = WEAK_SCALAR_PROBES
    plans: dict[OperationPlan, None] = {}
    for spec in registered_operations():
        for overload in spec.overloads:
            tensor_domain_index = 0
            candidates: list[tuple[Any, ...]] = []
            for role in overload.roles:
                if role is OperandRole.WEAK_SCALAR:
                    candidates.append(weak_scalars)
                    continue
                domains = overload.tensor_domains
                if domains is None:
                    candidates.append(_DEFAULT_TENSOR_DTYPES)
                else:
                    declared = domains[tensor_domain_index]
                    candidates.append(
                        tuple(
                            dtype
                            for dtype in _STORAGE_DTYPES
                            if any(dtype is allowed for allowed in declared)
                        )
                    )
                tensor_domain_index += 1
            for operands in product(*candidates):
                plans[overload.rule(*operands)] = None
    return tuple(plans)


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
        for plan in _resolvable_generic_plans()
        if executable_plan_shape(plan)
    )
