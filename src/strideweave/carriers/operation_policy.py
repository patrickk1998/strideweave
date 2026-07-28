"""Backend-independent operation planning for simple dtypes.

This module is the single executable statement of the policy specified in
``design/SimpleDType-operation-policy.md``. For one operation and its operand
dtypes it decides which operands are converted, what arithmetic runs, how terms
combine, what dtype the result carries, and whether that result may participate
in autograd. ``Generic``, native ``CPU``, and future accelerator carriers
resolve a plan here and execute it; a backend never carries a promotion table of
its own.

The policy is an intentional starting point rather than a compatibility
promise. Revising it means changing the specification, this resolver, its
expected-plan fixtures, and every backend conformance expectation together; a
change that lands in one backend first is a policy fork.

Only ``SimpleDType`` operands are planned. Legacy opaque categories
(``DType.Any``, ``DType.Floating``), compound descriptors, and registered but
unimplemented simple encodings each raise their own documented error rather
than resolving to a guessed plan.

Native operations resolve a plan while still holding the GIL and then release it
to run the kernel loop (``CPP001``). Every error this module raises is
value-independent, so resolution never has to read an element.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import product
from numbers import Integral, Real
from typing import Any, Final

from .dtype import DType, SimpleDType

__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "POW_INTEGER_EXPONENT_MAX",
    "SUPPORTED_TENSOR_DTYPES",
    "WEAK_SCALAR_PROBES",
    "Accumulation",
    "Arithmetic",
    "OperandPlan",
    "OperandRole",
    "OperationPlan",
    "OperationSpec",
    "WeakScalarKind",
    "registered_operations",
    "resolvable_plans",
    "resolve_operation_plan",
]

INT32_MIN: Final = -(2**31)
INT32_MAX: Final = 2**31 - 1

# The largest exponent `pow` treats as integer-preserving. This bound is central
# policy rather than any backend's representable range: every backend switches to
# the floating path at the same exponent.
POW_INTEGER_EXPONENT_MAX: Final = INT32_MAX


class OperandRole(Enum):
    """What kind of operand one position of an operation takes."""

    TENSOR = "tensor"
    WEAK_SCALAR = "weak_scalar"


class WeakScalarKind(Enum):
    """The normalized kind of a weak Python scalar operand.

    A weak scalar has no dtype of its own and never forces a width; it only
    selects between plans. ``bool`` normalizes to :attr:`FLOAT` rather than
    :attr:`INTEGER`, which is existing pinned behavior flagged as provisional in
    the policy specification.
    """

    INTEGER = "integer"
    FLOAT = "float"


class Arithmetic(Enum):
    """The arithmetic an operation's per-element computation performs.

    ``BINARY32`` is IEEE-754 binary32 with round-to-nearest-even, no FMA
    contraction, no reassociation, and no wider intermediate; IEEE special
    values propagate rather than raise. ``INT32_EXACT_CHECKED`` evaluates
    exactly over the integers and raises ``OverflowError`` if the exact result
    leaves ``Int32``. ``INT32_EXACT`` is exact integer arithmetic that provably
    cannot leave ``Int32``, so no check is required.
    """

    BINARY32 = "binary32"
    INT32_EXACT_CHECKED = "int32_exact_checked"
    INT32_EXACT = "int32_exact"


class Accumulation(Enum):
    """How an operation that combines many terms combines them.

    ``SEQUENTIAL_BINARY32`` initializes to ``+0.0`` and adds each term in
    ascending logical index order, rounding in binary32 at every step; the order
    is normative, so pairwise or blocked summation does not conform.
    ``EXACT_INTEGER`` accumulates exactly over the integers and checks only the
    final narrowing into ``Int32``, so an intermediate partial sum may leave
    ``Int32`` range.
    """

    SEQUENTIAL_BINARY32 = "sequential_binary32"
    EXACT_INTEGER = "exact_integer"


@dataclass(frozen=True, slots=True)
class OperandPlan:
    """How one operand is materialized before the computation runs.

    Args:
        role: Whether this position takes a tensor or a weak Python scalar.
        dtype: The tensor operand's storage dtype, or ``None`` for a weak
            scalar, which has no dtype of its own.
        convert_to: The dtype this operand is materialized into. Equal to
            ``dtype`` when a tensor operand needs no conversion.
    """

    role: OperandRole
    dtype: SimpleDType | None
    convert_to: SimpleDType


@dataclass(frozen=True, slots=True)
class OperationPlan:
    """The resolved dtype policy for one operation invocation.

    Args:
        operation: The operation name the plan was resolved for.
        operands: One :class:`OperandPlan` per operand, positionally.
        compute: Arithmetic for the per-element computation.
        accumulation: How terms combine, or ``None`` for an operation that
            combines no terms.
        output: The dtype the result carrier reports. Autograd eligibility is
            not a separate field: a result participates in autograd exactly
            when its dtype is floating, which is the framework-wide rule the
            tensor layer already applies to every tensor, plan-produced or not.
    """

    operation: str
    operands: tuple[OperandPlan, ...]
    compute: Arithmetic
    accumulation: Accumulation | None
    output: SimpleDType


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One registered operation and the operand roles it takes.

    Args:
        name: Operation name, matching the ``dispatch_op`` name.
        roles: The role of each operand position.
        rule: Resolver for this operation's plan.
    """

    name: str
    roles: tuple[OperandRole, ...]
    rule: Callable[..., OperationPlan]


# The concrete simple dtypes carriers store and kernels implement today.
SUPPORTED_TENSOR_DTYPES: Final[tuple[SimpleDType, ...]] = (DType.Float32, DType.Int32)

# One weak scalar per branch a weak scalar can select: a non-negative weak
# integer keeps `mul` and `pow` integer, a negative one sends `pow` to the
# floating path, and a weak float sends both there. Enumerating coverage with
# these probes reaches every plan a weak scalar can produce.
WEAK_SCALAR_PROBES: Final[tuple[Any, ...]] = (3, -1, 0.5)

_DEFERRED_COMPOUND_MESSAGE: Final = (
    "operation planning is not implemented for compound dtype {name!r}: a "
    "compound representation needs representation-aware planning over its "
    "simple_types planes, which is not implemented"
)


def _is_supported_tensor_dtype(dtype: DType) -> bool:
    return any(dtype is candidate for candidate in SUPPORTED_TENSOR_DTYPES)


def _require_tensor_dtype(value: object, name: str) -> SimpleDType:
    """Validate one tensor operand's storage dtype against the supported set."""
    if not isinstance(value, DType):
        raise TypeError(f"{name} must be a DType")
    if value.is_compound():
        raise NotImplementedError(_DEFERRED_COMPOUND_MESSAGE.format(name=value.name))
    if not isinstance(value, SimpleDType):
        if value.is_opaque_storage():
            raise TypeError(
                f"{name} is the legacy opaque storage category DType.{value.name}, "
                "which is not a simple dtype and takes no part in simple "
                "promotion; legacy Generic arithmetic is a separate path"
            )
        raise TypeError(
            f"{name} is the abstract category DType.{value.name}, which describes "
            "a relationship rather than a representation"
        )
    if not _is_supported_tensor_dtype(value):
        raise NotImplementedError(
            f"no backend implements operations on DType.{value.name}; the "
            "supported simple dtypes are DType.Float32 and DType.Int32"
        )
    return value


def _normalize_weak_scalar(value: object, name: str) -> WeakScalarKind:
    """Normalize a weak Python scalar to the kind that selects its plan."""
    if isinstance(value, bool):
        # Provisional: `True` behaves as a weak float, so an Int32 tensor times
        # a bool yields Float32. Existing pinned behavior; see the policy's
        # provisional choices.
        return WeakScalarKind.FLOAT
    if isinstance(value, Integral):
        return WeakScalarKind.INTEGER
    if isinstance(value, Real):
        return WeakScalarKind.FLOAT
    raise TypeError(f"{name} must be a real Python number")


def _require_int32_scalar(value: Any, name: str) -> None:
    """Reject a weak integer an integer plan could not represent exactly."""
    if not INT32_MIN <= value <= INT32_MAX:
        raise OverflowError(f"{name} is out of int32 range")


def _tensor_operand(dtype: SimpleDType, convert_to: SimpleDType) -> OperandPlan:
    return OperandPlan(role=OperandRole.TENSOR, dtype=dtype, convert_to=convert_to)


def _weak_scalar_operand(convert_to: SimpleDType) -> OperandPlan:
    return OperandPlan(role=OperandRole.WEAK_SCALAR, dtype=None, convert_to=convert_to)


def _plan(
    operation: str,
    operands: tuple[OperandPlan, ...],
    *,
    compute: Arithmetic,
    output: SimpleDType,
    accumulation: Accumulation | None = None,
) -> OperationPlan:
    """Build a plan from the decisions one rule made."""
    return OperationPlan(
        operation=operation,
        operands=operands,
        compute=compute,
        accumulation=accumulation,
        output=output,
    )


def _both_integer(lhs: SimpleDType, rhs: SimpleDType) -> bool:
    return lhs is DType.Int32 and rhs is DType.Int32


def _floating_binary_operands(
    lhs: SimpleDType, rhs: SimpleDType
) -> tuple[OperandPlan, ...]:
    return (
        _tensor_operand(lhs, DType.Float32),
        _tensor_operand(rhs, DType.Float32),
    )


def _resolve_binary_elementwise(
    operation: str, lhs_dtype: object, rhs_dtype: object
) -> OperationPlan:
    """Resolve ``add``, ``sub``, and ``elementwise_mul``.

    Two integer operands stay integer with checked overflow; any floating
    operand converts the other and produces a floating result.
    """
    lhs = _require_tensor_dtype(lhs_dtype, "lhs_dtype")
    rhs = _require_tensor_dtype(rhs_dtype, "rhs_dtype")
    if _both_integer(lhs, rhs):
        return _plan(
            operation,
            (_tensor_operand(lhs, DType.Int32), _tensor_operand(rhs, DType.Int32)),
            compute=Arithmetic.INT32_EXACT_CHECKED,
            output=DType.Int32,
        )
    return _plan(
        operation,
        _floating_binary_operands(lhs, rhs),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_div(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
    """Resolve ``div``, which is always floating; there is no integer path."""
    lhs = _require_tensor_dtype(lhs_dtype, "lhs_dtype")
    rhs = _require_tensor_dtype(rhs_dtype, "rhs_dtype")
    return _plan(
        "div",
        _floating_binary_operands(lhs, rhs),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_mul(tensor_dtype: object, scalar: object) -> OperationPlan:
    """Resolve ``mul``, a tensor scaled by a weak Python scalar."""
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    kind = _normalize_weak_scalar(scalar, "scalar")
    if tensor is DType.Int32 and kind is WeakScalarKind.INTEGER:
        _require_int32_scalar(scalar, "scalar")
        return _plan(
            "mul",
            (_tensor_operand(tensor, DType.Int32), _weak_scalar_operand(DType.Int32)),
            compute=Arithmetic.INT32_EXACT_CHECKED,
            output=DType.Int32,
        )
    return _plan(
        "mul",
        (
            _tensor_operand(tensor, DType.Float32),
            _weak_scalar_operand(DType.Float32),
        ),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _exponent_preserves_integer(exponent: Any, kind: WeakScalarKind) -> bool:
    """Report whether an exponent keeps an integer base closed over ``Int32``.

    A negative exponent produces reciprocals and a non-integral one produces
    roots, so only a weak integer in ``[0, POW_INTEGER_EXPONENT_MAX]``
    preserves an integer result.
    """
    return kind is WeakScalarKind.INTEGER and 0 <= exponent <= POW_INTEGER_EXPONENT_MAX


def _resolve_pow(tensor_dtype: object, exponent: object) -> OperationPlan:
    """Resolve ``pow``, a tensor raised to a weak Python scalar exponent."""
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    kind = _normalize_weak_scalar(exponent, "exponent")
    if tensor is DType.Int32 and _exponent_preserves_integer(exponent, kind):
        return _plan(
            "pow",
            (_tensor_operand(tensor, DType.Int32), _weak_scalar_operand(DType.Int32)),
            compute=Arithmetic.INT32_EXACT_CHECKED,
            output=DType.Int32,
        )
    return _plan(
        "pow",
        (
            _tensor_operand(tensor, DType.Float32),
            _weak_scalar_operand(DType.Float32),
        ),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_floating_activation(operation: str, tensor_dtype: object) -> OperationPlan:
    """Resolve an activation that is floating regardless of its input dtype."""
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    return _plan(
        operation,
        (_tensor_operand(tensor, DType.Float32),),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_relu(tensor_dtype: object) -> OperationPlan:
    """Resolve ``relu``, which preserves its input dtype.

    Selecting between an element and zero cannot overflow, so an integer input
    needs no check.
    """
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    integer = tensor is DType.Int32
    return _plan(
        "relu",
        (_tensor_operand(tensor, tensor),),
        compute=Arithmetic.INT32_EXACT if integer else Arithmetic.BINARY32,
        output=tensor,
    )


def _resolve_reduce(tensor_dtype: object) -> OperationPlan:
    """Resolve ``reduce``, a sum that preserves its input dtype."""
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    integer = tensor is DType.Int32
    return _plan(
        "reduce",
        (_tensor_operand(tensor, tensor),),
        compute=Arithmetic.INT32_EXACT_CHECKED if integer else Arithmetic.BINARY32,
        output=tensor,
        accumulation=(
            Accumulation.EXACT_INTEGER if integer else Accumulation.SEQUENTIAL_BINARY32
        ),
    )


def _resolve_matmul(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
    """Resolve ``matmul``, which promotes like binary arithmetic but accumulates.

    Two integer operands compute each product with :attr:`Arithmetic.INT32_EXACT`
    rather than a checked product: only the final narrowed sum must fit
    ``Int32``, so checking each product would reject contractions whose terms
    legitimately cancel.
    """
    lhs = _require_tensor_dtype(lhs_dtype, "lhs_dtype")
    rhs = _require_tensor_dtype(rhs_dtype, "rhs_dtype")
    if _both_integer(lhs, rhs):
        return _plan(
            "matmul",
            (_tensor_operand(lhs, DType.Int32), _tensor_operand(rhs, DType.Int32)),
            compute=Arithmetic.INT32_EXACT,
            output=DType.Int32,
            accumulation=Accumulation.EXACT_INTEGER,
        )
    return _plan(
        "matmul",
        _floating_binary_operands(lhs, rhs),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
        accumulation=Accumulation.SEQUENTIAL_BINARY32,
    )


_TENSOR_PAIR: Final = (OperandRole.TENSOR, OperandRole.TENSOR)
_TENSOR_AND_SCALAR: Final = (OperandRole.TENSOR, OperandRole.WEAK_SCALAR)
_ONE_TENSOR: Final = (OperandRole.TENSOR,)

# Activations that produce a floating result regardless of their input dtype.
_FLOATING_ACTIVATIONS: Final = (
    "elu",
    "exp",
    "gelu",
    "leaky_relu",
    "sigmoid",
    "silu",
    "softplus",
    "tanh",
)


def _floating_activation_spec(operation: str) -> OperationSpec:
    def rule(tensor_dtype: object) -> OperationPlan:
        return _resolve_floating_activation(operation, tensor_dtype)

    return OperationSpec(name=operation, roles=_ONE_TENSOR, rule=rule)


def _binary_elementwise_spec(operation: str) -> OperationSpec:
    def rule(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
        return _resolve_binary_elementwise(operation, lhs_dtype, rhs_dtype)

    return OperationSpec(name=operation, roles=_TENSOR_PAIR, rule=rule)


def _build_registry() -> dict[str, OperationSpec]:
    """Build the central registry every planned operation is reached through.

    Coverage is enumerated from this registry rather than hand-maintained, so an
    operation added without an expected-plan fixture fails conformance
    enumeration instead of passing silently. Representation-preserving
    operations (``view``, ``permute``, ``rearrange``, ``move``) are deliberately
    absent: they preserve dtype exactly and the planner has no opinion on them.
    """
    specs = [
        *(
            _binary_elementwise_spec(operation)
            for operation in ("add", "sub", "elementwise_mul")
        ),
        OperationSpec(name="div", roles=_TENSOR_PAIR, rule=_resolve_div),
        OperationSpec(name="mul", roles=_TENSOR_AND_SCALAR, rule=_resolve_mul),
        OperationSpec(name="pow", roles=_TENSOR_AND_SCALAR, rule=_resolve_pow),
        OperationSpec(name="relu", roles=_ONE_TENSOR, rule=_resolve_relu),
        OperationSpec(name="reduce", roles=_ONE_TENSOR, rule=_resolve_reduce),
        OperationSpec(name="matmul", roles=_TENSOR_PAIR, rule=_resolve_matmul),
        *(_floating_activation_spec(operation) for operation in _FLOATING_ACTIVATIONS),
    ]
    return {spec.name: spec for spec in specs}


_REGISTRY: Final[dict[str, OperationSpec]] = _build_registry()


def registered_operations() -> tuple[OperationSpec, ...]:
    """Return every operation the planner covers, ordered by name.

    Returns:
        The registered specifications, so conformance suites can enumerate
        coverage exhaustively rather than sampling it.

    Examples:
        >>> from strideweave.carriers.operation_policy import registered_operations
        >>> "matmul" in {spec.name for spec in registered_operations()}
        True
    """
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def resolvable_plans(
    tensor_dtypes: tuple[SimpleDType, ...] = SUPPORTED_TENSOR_DTYPES,
    weak_scalars: tuple[Any, ...] = WEAK_SCALAR_PROBES,
) -> tuple[OperationPlan, ...]:
    """Return every distinct plan this policy resolves over the given operands.

    Coverage is enumerated from the operation registry rather than listed by
    hand, so an operation added to the policy is planned here too. A backend
    uses this to derive the shapes it might be asked to execute before deciding
    which of them it implements; the enumeration itself expresses no backend's
    reach.

    Args:
        tensor_dtypes: Storage dtypes to plan each tensor operand with.
        weak_scalars: Values to plan each weak scalar operand with. The default
            probes select every branch a weak scalar can choose between.

    Returns:
        The distinct resolved plans, deduplicated because many operand
        combinations resolve identically, ordered by operation name and then by
        the order the operands were enumerated in.

    Examples:
        >>> from strideweave.carriers.operation_policy import resolvable_plans
        >>> {plan.operation for plan in resolvable_plans()} >= {"add", "matmul"}
        True
    """
    candidates = {
        OperandRole.TENSOR: tuple(tensor_dtypes),
        OperandRole.WEAK_SCALAR: tuple(weak_scalars),
    }
    plans: dict[OperationPlan, None] = {}
    for spec in registered_operations():
        for operands in product(*(candidates[role] for role in spec.roles)):
            plans[spec.rule(*operands)] = None
    return tuple(plans)


def resolve_operation_plan(operation: str, *operands: object) -> OperationPlan:
    """Resolve the shared :class:`OperationPlan` for one operation invocation.

    This is the single entry point every backend uses. Tensor operands are given
    as their carrier storage dtype and weak scalar operands as the Python value
    itself, positionally, in the operation's own argument order.

    Args:
        operation: Operation name, matching the ``dispatch_op`` name (for
            example ``"add"``, ``"mul"``, ``"pow"``, ``"matmul"``, ``"reduce"``,
            ``"relu"``, or ``"exp"``).
        *operands: One value per operand position: a ``SimpleDType`` for a
            tensor operand, a real Python number for a weak scalar operand.

    Returns:
        The resolved immutable plan.

    Raises:
        NotImplementedError: If ``operation`` is not registered, or an operand is
            a compound dtype or a registered but unimplemented simple dtype.
        TypeError: If the operand count is wrong, an operand is not a ``DType``
            where a tensor is expected, an operand is an abstract or legacy
            opaque category, or a weak scalar is not a real Python number.
        OverflowError: If a weak integer scalar an integer plan would use is
            outside ``Int32`` range.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> plan = resolve_operation_plan("add", DType.Int32, DType.Float32)
        >>> plan.output is DType.Float32
        True
        >>> plan.compute
        <Arithmetic.BINARY32: 'binary32'>
    """
    spec = _REGISTRY.get(operation)
    if spec is None:
        raise NotImplementedError(
            f"no dtype plan is defined for operation {operation!r}"
        )
    if len(operands) != len(spec.roles):
        raise TypeError(
            f"{operation} takes {len(spec.roles)} operands, got {len(operands)}"
        )
    return spec.rule(*operands)
