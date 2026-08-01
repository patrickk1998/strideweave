"""Backend-independent operation planning for simple dtypes.

This module is the single executable statement of the policy specified in
``design/SimpleDType-operation-policy.md``. For one operation and its operand
dtypes it decides which operands are converted, what arithmetic runs, how terms
combine, and what dtype the result carries. Autograd participation follows from
the tensor layer's floating-dtype rule. ``Generic``, native ``CPU``, and future accelerator carriers
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
    "OperationOverload",
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
    ``SEQUENTIAL_BINARY32_PRODUCT`` is the corresponding ordered product.
    ``MAXIMUM`` and ``MINIMUM`` apply the operation's pinned Float32 NaN and
    signed-zero rules in logical order. ``ARGMAX`` and ``ARGMIN`` additionally
    retain the first winning logical ordinal. ``EXACT_INTEGER`` accumulates
    exactly over the integers and checks only the final narrowing into
    ``Int32``, so an intermediate partial sum may leave ``Int32`` range.
    """

    SEQUENTIAL_BINARY32 = "sequential_binary32"
    SEQUENTIAL_BINARY32_PRODUCT = "sequential_binary32_product"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
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
class OperationOverload:
    """One positional operand-role signature and its plan resolver.

    An operation may expose several overloads under one dispatch name.  The
    overload owns only dtype-policy operands: tensor positions receive their
    storage :class:`DType`, and weak-scalar positions receive the Python scalar
    value.  Shape, axis, ordering, and other non-dtype parameters remain the
    operation implementation's responsibility and are not smuggled into the
    promotion policy as weak scalars.

    Args:
        roles: Positional roles accepted by this overload.
        rule: Resolver that returns the exact plan for those operands.
        tensor_domains: Optional allowed dtype domain for each tensor role, in
            tensor-role order. ``None`` uses the caller-provided enumeration
            domain. Runtime validation remains the resolver's responsibility;
            these domains make exhaustive capability enumeration precise for
            overloads that intentionally accept only part of that domain.
    """

    roles: tuple[OperandRole, ...]
    rule: Callable[..., OperationPlan]
    tensor_domains: tuple[tuple[SimpleDType, ...], ...] | None = None

    def __post_init__(self) -> None:
        tensor_count = sum(role is OperandRole.TENSOR for role in self.roles)
        if self.tensor_domains is not None and len(self.tensor_domains) != tensor_count:
            raise ValueError(
                "tensor_domains must contain one dtype domain per tensor role"
            )
        if self.tensor_domains is not None and any(
            not domain for domain in self.tensor_domains
        ):
            raise ValueError("a tensor dtype domain must not be empty")


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One registered operation and all dtype-policy overloads it accepts.

    Args:
        name: Operation name, matching the ``dispatch_op`` name.
        overloads: Non-empty immutable overload sequence. Role signatures must
            be unique so overload selection and exhaustive plan enumeration are
            deterministic.
        public: Whether the dispatch name itself is a public single-result
            function. Internal value/index operations may be planned while a
            public wrapper packages several single-result calls.
        dtype_operand_positions: Positions in the operation's full forward-call
            argument list that participate in dtype planning. ``None`` means
            every forward argument is a policy operand. This distinguishes
            axis, shape, ordering, and convolution configuration arguments from
            genuine weak-scalar operands without duplicating that knowledge in
            carrier adapters.
    """

    name: str
    overloads: tuple[OperationOverload, ...]
    public: bool = True
    dtype_operand_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.overloads:
            raise ValueError("an operation spec must declare at least one overload")
        signatures = tuple(overload.roles for overload in self.overloads)
        if len(set(signatures)) != len(signatures):
            raise ValueError(
                f"operation {self.name!r} declares one operand-role signature twice"
            )
        if self.dtype_operand_positions is not None:
            if any(position < 0 for position in self.dtype_operand_positions):
                raise ValueError("dtype operand positions must be non-negative")
            if tuple(sorted(set(self.dtype_operand_positions))) != (
                self.dtype_operand_positions
            ):
                raise ValueError(
                    "dtype operand positions must be unique and increasing"
                )
            arities = {len(overload.roles) for overload in self.overloads}
            if arities != {len(self.dtype_operand_positions)}:
                raise ValueError(
                    "dtype operand positions must match every overload arity"
                )

    @property
    def roles(self) -> tuple[OperandRole, ...]:
        """Return the sole overload's roles for single-overload compatibility.

        Consumers that enumerate the registry must iterate :attr:`overloads`.
        This property keeps the established single-overload API explicit while
        refusing to silently hide additional overloads.
        """
        if len(self.overloads) != 1:
            raise AttributeError(
                f"operation {self.name!r} has multiple overloads; iterate overloads"
            )
        return self.overloads[0].roles

    @property
    def rule(self) -> Callable[..., OperationPlan]:
        """Return the sole overload's resolver for compatibility."""
        if len(self.overloads) != 1:
            raise AttributeError(
                f"operation {self.name!r} has multiple overloads; iterate overloads"
            )
        return self.overloads[0].rule


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


def _require_float32_tensor_dtype(value: object, name: str) -> SimpleDType:
    """Require the concrete Float32 tensor domain used by v0 primitives."""
    dtype = _require_tensor_dtype(value, name)
    if dtype is not DType.Float32:
        raise TypeError(f"{name} must be DType.Float32")
    return dtype


def _require_int32_tensor_dtype(value: object, name: str) -> SimpleDType:
    """Require the concrete Int32 tensor domain used for logical indices."""
    dtype = _require_tensor_dtype(value, name)
    if dtype is not DType.Int32:
        raise TypeError(f"{name} must be DType.Int32")
    return dtype


def _require_bool_tensor_dtype(value: object, name: str) -> SimpleDType:
    """Require the concrete Bool tensor domain used by masked selection."""
    if value is DType.Bool:
        return DType.Bool
    _require_tensor_dtype(value, name)
    raise TypeError(f"{name} must be DType.Bool")


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


def _resolve_float32_unary(operation: str, tensor_dtype: object) -> OperationPlan:
    tensor = _require_float32_tensor_dtype(tensor_dtype, "tensor_dtype")
    return _plan(
        operation,
        (_tensor_operand(tensor, DType.Float32),),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_float32_binary(
    operation: str, lhs_dtype: object, rhs_dtype: object
) -> OperationPlan:
    lhs = _require_float32_tensor_dtype(lhs_dtype, "lhs_dtype")
    rhs = _require_float32_tensor_dtype(rhs_dtype, "rhs_dtype")
    return _plan(
        operation,
        (_tensor_operand(lhs, DType.Float32), _tensor_operand(rhs, DType.Float32)),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_float32_predicate(
    operation: str, lhs_dtype: object, rhs_dtype: object
) -> OperationPlan:
    lhs = _require_float32_tensor_dtype(lhs_dtype, "lhs_dtype")
    rhs = _require_float32_tensor_dtype(rhs_dtype, "rhs_dtype")
    return _plan(
        operation,
        (_tensor_operand(lhs, DType.Float32), _tensor_operand(rhs, DType.Float32)),
        compute=Arithmetic.BINARY32,
        output=DType.Bool,
    )


def _resolve_logical_not(tensor_dtype: object) -> OperationPlan:
    tensor = _require_float32_tensor_dtype(tensor_dtype, "tensor_dtype")
    return _plan(
        "logical_not",
        (_tensor_operand(tensor, DType.Float32),),
        compute=Arithmetic.BINARY32,
        output=DType.Bool,
    )


def _resolve_float32_reduction(
    operation: str,
    tensor_dtype: object,
    accumulation: Accumulation,
    *,
    output: SimpleDType = DType.Float32,
) -> OperationPlan:
    tensor = _require_float32_tensor_dtype(tensor_dtype, "tensor_dtype")
    return _plan(
        operation,
        (_tensor_operand(tensor, DType.Float32),),
        compute=Arithmetic.BINARY32,
        accumulation=accumulation,
        output=output,
    )


def _resolve_float32_selection(
    operation: str,
    tensor_dtype: object,
    *,
    output: SimpleDType,
) -> OperationPlan:
    tensor = _require_float32_tensor_dtype(tensor_dtype, "tensor_dtype")
    return _plan(
        operation,
        (_tensor_operand(tensor, DType.Float32),),
        compute=Arithmetic.BINARY32,
        output=output,
    )


def _resolve_gather(data_dtype: object, indices_dtype: object) -> OperationPlan:
    data = _require_float32_tensor_dtype(data_dtype, "data_dtype")
    indices = _require_int32_tensor_dtype(indices_dtype, "indices_dtype")
    return _plan(
        "gather",
        (
            _tensor_operand(data, DType.Float32),
            _tensor_operand(indices, DType.Int32),
        ),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_scatter(
    operation: str,
    base_dtype: object,
    indices_dtype: object,
    updates_dtype: object,
) -> OperationPlan:
    base = _require_float32_tensor_dtype(base_dtype, "base_dtype")
    indices = _require_int32_tensor_dtype(indices_dtype, "indices_dtype")
    updates = _require_float32_tensor_dtype(updates_dtype, "updates_dtype")
    return _plan(
        operation,
        (
            _tensor_operand(base, DType.Float32),
            _tensor_operand(indices, DType.Int32),
            _tensor_operand(updates, DType.Float32),
        ),
        compute=Arithmetic.BINARY32,
        accumulation=(
            Accumulation.SEQUENTIAL_BINARY32 if operation == "scatter_add" else None
        ),
        output=DType.Float32,
    )


def _resolve_select(
    condition_dtype: object,
    on_true_dtype: object,
    on_false_dtype: object,
) -> OperationPlan:
    condition = _require_bool_tensor_dtype(condition_dtype, "condition_dtype")
    on_true = _require_float32_tensor_dtype(on_true_dtype, "on_true_dtype")
    on_false = _require_float32_tensor_dtype(on_false_dtype, "on_false_dtype")
    return _plan(
        "select",
        (
            _tensor_operand(condition, DType.Bool),
            _tensor_operand(on_true, DType.Float32),
            _tensor_operand(on_false, DType.Float32),
        ),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_clamp(
    tensor_dtype: object,
    lower: object,
    upper: object,
    *,
    lower_role: OperandRole,
    upper_role: OperandRole,
) -> OperationPlan:
    tensor = _require_float32_tensor_dtype(tensor_dtype, "tensor_dtype")

    def bound_operand(value: object, role: OperandRole, name: str) -> OperandPlan:
        if role is OperandRole.TENSOR:
            dtype = _require_float32_tensor_dtype(value, f"{name}_dtype")
            return _tensor_operand(dtype, DType.Float32)
        _normalize_weak_scalar(value, name)
        return _weak_scalar_operand(DType.Float32)

    return _plan(
        "clamp",
        (
            _tensor_operand(tensor, DType.Float32),
            bound_operand(lower, lower_role, "lower"),
            bound_operand(upper, upper_role, "upper"),
        ),
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


def _resolve_tensor_pow(base_dtype: object, exponent_dtype: object) -> OperationPlan:
    """Resolve tensor-tensor power through the Float32 v0 primitive."""
    base = _require_tensor_dtype(base_dtype, "base_dtype")
    exponent = _require_tensor_dtype(exponent_dtype, "exponent_dtype")
    return _plan(
        "pow",
        (
            _tensor_operand(base, DType.Float32),
            _tensor_operand(exponent, DType.Float32),
        ),
        compute=Arithmetic.BINARY32,
        output=DType.Float32,
    )


def _resolve_reverse_pow(base: object, exponent_dtype: object) -> OperationPlan:
    """Resolve a weak scalar base raised to a tensor exponent."""
    _normalize_weak_scalar(base, "base")
    exponent = _require_tensor_dtype(exponent_dtype, "exponent_dtype")
    return _plan(
        "pow",
        (
            _weak_scalar_operand(DType.Float32),
            _tensor_operand(exponent, DType.Float32),
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


def _resolve_reduce_sum(tensor_dtype: object) -> OperationPlan:
    """Resolve ``reduce_sum``, which preserves its input dtype."""
    tensor = _require_tensor_dtype(tensor_dtype, "tensor_dtype")
    integer = tensor is DType.Int32
    return _plan(
        "reduce_sum",
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
_SCALAR_AND_TENSOR: Final = (OperandRole.WEAK_SCALAR, OperandRole.TENSOR)
_THREE_TENSORS: Final = (
    OperandRole.TENSOR,
    OperandRole.TENSOR,
    OperandRole.TENSOR,
)
_TWO_TENSORS_AND_SCALAR: Final = (
    OperandRole.TENSOR,
    OperandRole.TENSOR,
    OperandRole.WEAK_SCALAR,
)
_TENSOR_SCALAR_TENSOR: Final = (
    OperandRole.TENSOR,
    OperandRole.WEAK_SCALAR,
    OperandRole.TENSOR,
)
_TENSOR_AND_TWO_SCALARS: Final = (
    OperandRole.TENSOR,
    OperandRole.WEAK_SCALAR,
    OperandRole.WEAK_SCALAR,
)
_ONE_TENSOR: Final = (OperandRole.TENSOR,)
_FLOAT32_DOMAIN: Final = (DType.Float32,)
_INT32_DOMAIN: Final = (DType.Int32,)

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

    return _single_overload_spec(operation, _ONE_TENSOR, rule)


def _binary_elementwise_spec(operation: str) -> OperationSpec:
    def rule(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
        return _resolve_binary_elementwise(operation, lhs_dtype, rhs_dtype)

    return _single_overload_spec(operation, _TENSOR_PAIR, rule)


def _float32_unary_spec(operation: str) -> OperationSpec:
    def rule(tensor_dtype: object) -> OperationPlan:
        return _resolve_float32_unary(operation, tensor_dtype)

    return _single_overload_spec(
        operation,
        _ONE_TENSOR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN,),
    )


def _float32_binary_spec(operation: str) -> OperationSpec:
    def rule(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
        return _resolve_float32_binary(operation, lhs_dtype, rhs_dtype)

    return _single_overload_spec(
        operation,
        _TENSOR_PAIR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
    )


def _float32_predicate_spec(operation: str) -> OperationSpec:
    def rule(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
        return _resolve_float32_predicate(operation, lhs_dtype, rhs_dtype)

    return _single_overload_spec(
        operation,
        _TENSOR_PAIR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
    )


def _float32_reduction_spec(
    operation: str,
    accumulation: Accumulation,
    *,
    output: SimpleDType = DType.Float32,
    dtype_operand_positions: tuple[int, ...] | None = None,
) -> OperationSpec:
    def rule(tensor_dtype: object) -> OperationPlan:
        return _resolve_float32_reduction(
            operation, tensor_dtype, accumulation, output=output
        )

    return _single_overload_spec(
        operation,
        _ONE_TENSOR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN,),
        dtype_operand_positions=dtype_operand_positions,
    )


def _float32_contraction_spec(operation: str) -> OperationSpec:
    def rule(lhs_dtype: object, rhs_dtype: object) -> OperationPlan:
        lhs = _require_float32_tensor_dtype(lhs_dtype, "lhs_dtype")
        rhs = _require_float32_tensor_dtype(rhs_dtype, "rhs_dtype")
        return _plan(
            operation,
            (
                _tensor_operand(lhs, DType.Float32),
                _tensor_operand(rhs, DType.Float32),
            ),
            compute=Arithmetic.BINARY32,
            accumulation=Accumulation.SEQUENTIAL_BINARY32,
            output=DType.Float32,
        )

    return _single_overload_spec(
        operation,
        _TENSOR_PAIR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
        dtype_operand_positions=(0, 1),
    )


def _selection_spec(
    operation: str, *, output: SimpleDType, public: bool
) -> OperationSpec:
    def rule(tensor_dtype: object) -> OperationPlan:
        return _resolve_float32_selection(operation, tensor_dtype, output=output)

    return _single_overload_spec(
        operation,
        _ONE_TENSOR,
        rule,
        tensor_domains=(_FLOAT32_DOMAIN,),
        public=public,
        dtype_operand_positions=(0,),
    )


def _single_overload_spec(
    name: str,
    roles: tuple[OperandRole, ...],
    rule: Callable[..., OperationPlan],
    *,
    tensor_domains: tuple[tuple[SimpleDType, ...], ...] | None = None,
    public: bool = True,
    dtype_operand_positions: tuple[int, ...] | None = None,
) -> OperationSpec:
    """Build the common one-overload operation specification."""
    return OperationSpec(
        name=name,
        overloads=(OperationOverload(roles, rule, tensor_domains),),
        public=public,
        dtype_operand_positions=dtype_operand_positions,
    )


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
        _single_overload_spec("div", _TENSOR_PAIR, _resolve_div),
        OperationSpec(
            name="mul",
            overloads=(
                OperationOverload(
                    _TENSOR_PAIR,
                    lambda lhs, rhs: _resolve_binary_elementwise("mul", lhs, rhs),
                ),
                OperationOverload(_TENSOR_AND_SCALAR, _resolve_mul),
            ),
        ),
        OperationSpec(
            name="pow",
            overloads=(
                OperationOverload(_TENSOR_PAIR, _resolve_tensor_pow),
                OperationOverload(_TENSOR_AND_SCALAR, _resolve_pow),
                OperationOverload(_SCALAR_AND_TENSOR, _resolve_reverse_pow),
            ),
        ),
        *(
            _float32_binary_spec(operation)
            for operation in ("maximum", "minimum", "rem")
        ),
        *(_float32_predicate_spec(operation) for operation in ("eq", "ne", "lt", "le")),
        _single_overload_spec(
            "logical_not",
            _ONE_TENSOR,
            _resolve_logical_not,
            tensor_domains=(_FLOAT32_DOMAIN,),
        ),
        _single_overload_spec("relu", _ONE_TENSOR, _resolve_relu),
        _single_overload_spec("reduce_sum", _ONE_TENSOR, _resolve_reduce_sum),
        _float32_reduction_spec(
            "reduce_prod", Accumulation.SEQUENTIAL_BINARY32_PRODUCT
        ),
        _float32_reduction_spec("reduce_max", Accumulation.MAXIMUM),
        _float32_reduction_spec("reduce_min", Accumulation.MINIMUM),
        _float32_reduction_spec("argmax", Accumulation.ARGMAX, output=DType.Int32),
        _float32_reduction_spec("argmin", Accumulation.ARGMIN, output=DType.Int32),
        _float32_reduction_spec(
            "cumsum",
            Accumulation.SEQUENTIAL_BINARY32,
            dtype_operand_positions=(0,),
        ),
        _single_overload_spec("matmul", _TENSOR_PAIR, _resolve_matmul),
        _float32_contraction_spec("conv_general"),
        *(_floating_activation_spec(operation) for operation in _FLOATING_ACTIVATIONS),
        *(
            _float32_unary_spec(operation)
            for operation in (
                "neg",
                "abs",
                "sign",
                "recip",
                "sqrt",
                "rsqrt",
                "exp2",
                "log",
                "log2",
                "sin",
                "cos",
                "erf",
                "floor",
                "ceil",
                "round",
            )
        ),
        _single_overload_spec(
            "gather",
            _TENSOR_PAIR,
            _resolve_gather,
            tensor_domains=(_FLOAT32_DOMAIN, _INT32_DOMAIN),
            dtype_operand_positions=(0, 1),
        ),
        _single_overload_spec(
            "scatter",
            _THREE_TENSORS,
            lambda base, indices, updates: _resolve_scatter(
                "scatter", base, indices, updates
            ),
            tensor_domains=(
                _FLOAT32_DOMAIN,
                _INT32_DOMAIN,
                _FLOAT32_DOMAIN,
            ),
            dtype_operand_positions=(0, 1, 2),
        ),
        _single_overload_spec(
            "scatter_add",
            _THREE_TENSORS,
            lambda base, indices, updates: _resolve_scatter(
                "scatter_add", base, indices, updates
            ),
            tensor_domains=(
                _FLOAT32_DOMAIN,
                _INT32_DOMAIN,
                _FLOAT32_DOMAIN,
            ),
            dtype_operand_positions=(0, 1, 2),
        ),
        _single_overload_spec(
            "select",
            _THREE_TENSORS,
            _resolve_select,
            tensor_domains=(
                (DType.Bool,),
                _FLOAT32_DOMAIN,
                _FLOAT32_DOMAIN,
            ),
        ),
        OperationSpec(
            name="clamp",
            overloads=(
                OperationOverload(
                    _THREE_TENSORS,
                    lambda tensor, lower, upper: _resolve_clamp(
                        tensor,
                        lower,
                        upper,
                        lower_role=OperandRole.TENSOR,
                        upper_role=OperandRole.TENSOR,
                    ),
                    (_FLOAT32_DOMAIN, _FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
                ),
                OperationOverload(
                    _TWO_TENSORS_AND_SCALAR,
                    lambda tensor, lower, upper: _resolve_clamp(
                        tensor,
                        lower,
                        upper,
                        lower_role=OperandRole.TENSOR,
                        upper_role=OperandRole.WEAK_SCALAR,
                    ),
                    (_FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
                ),
                OperationOverload(
                    _TENSOR_SCALAR_TENSOR,
                    lambda tensor, lower, upper: _resolve_clamp(
                        tensor,
                        lower,
                        upper,
                        lower_role=OperandRole.WEAK_SCALAR,
                        upper_role=OperandRole.TENSOR,
                    ),
                    (_FLOAT32_DOMAIN, _FLOAT32_DOMAIN),
                ),
                OperationOverload(
                    _TENSOR_AND_TWO_SCALARS,
                    lambda tensor, lower, upper: _resolve_clamp(
                        tensor,
                        lower,
                        upper,
                        lower_role=OperandRole.WEAK_SCALAR,
                        upper_role=OperandRole.WEAK_SCALAR,
                    ),
                    (_FLOAT32_DOMAIN,),
                ),
            ),
        ),
        *(
            _selection_spec(operation, output=output, public=False)
            for operation, output in (
                ("_sort_values", DType.Float32),
                ("_sort_indices", DType.Int32),
                ("_topk_values", DType.Float32),
                ("_topk_indices", DType.Int32),
            )
        ),
    ]
    registry = {spec.name: spec for spec in specs}
    if len(registry) != len(specs):
        raise ValueError("the operation registry declares one name twice")
    return registry


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
        for overload in spec.overloads:
            tensor_domain_index = 0
            operand_candidates: list[tuple[Any, ...]] = []
            for role in overload.roles:
                if role is OperandRole.WEAK_SCALAR:
                    operand_candidates.append(candidates[role])
                    continue
                declared_domains = overload.tensor_domains
                declared = (
                    tuple(tensor_dtypes)
                    if declared_domains is None
                    else declared_domains[tensor_domain_index]
                )
                tensor_domain_index += 1
                operand_candidates.append(
                    tuple(
                        dtype
                        for dtype in tensor_dtypes
                        if any(dtype is allowed for allowed in declared)
                    )
                )
            for operands in product(*operand_candidates):
                plan = overload.rule(*operands)
                _validate_overload_plan(spec, overload, plan)
                plans[plan] = None
    return tuple(plans)


def _operand_role(value: object) -> OperandRole | None:
    """Classify an operand only when its role is unambiguous.

    Invalid values deliberately return ``None`` so the selected resolver owns
    the established diagnostic (for example, ``lhs_dtype must be a DType`` or
    ``scalar must be a real Python number``) instead of overload selection
    replacing it with a generic mismatch.
    """
    if isinstance(value, DType):
        return OperandRole.TENSOR
    if isinstance(value, Real):
        return OperandRole.WEAK_SCALAR
    return None


def _select_overload(
    spec: OperationSpec, operands: tuple[object, ...]
) -> OperationOverload:
    """Choose one overload deterministically without swallowing rule errors."""
    same_arity = tuple(
        overload for overload in spec.overloads if len(overload.roles) == len(operands)
    )
    if not same_arity:
        arities = tuple(sorted({len(overload.roles) for overload in spec.overloads}))
        expected = (
            str(arities[0])
            if len(arities) == 1
            else "one of " + ", ".join(str(arity) for arity in arities)
        )
        raise TypeError(f"{spec.name} takes {expected} operands, got {len(operands)}")

    classified = tuple(_operand_role(value) for value in operands)
    compatible = tuple(
        overload
        for overload in same_arity
        if all(
            actual is None or actual is declared
            for actual, declared in zip(classified, overload.roles, strict=True)
        )
    )
    # Unknown invalid values intentionally choose the first otherwise-compatible
    # declaration so its resolver produces the precise dtype/scalar diagnostic.
    if compatible:
        # A non-DType invalid value is more plausibly a malformed weak scalar
        # than a malformed dtype when both overloads share the same arity. This
        # preserves the established ``must be a real Python number`` diagnostic
        # for calls such as ``mul(tensor_dtype, "bad")`` without weakening the
        # single-overload tensor validation used by operations such as ``add``.
        return max(
            compatible,
            key=lambda overload: sum(
                actual is None and declared is OperandRole.WEAK_SCALAR
                for actual, declared in zip(classified, overload.roles, strict=True)
            ),
        )
    declared = ", ".join(
        "(" + ", ".join(role.value for role in overload.roles) + ")"
        for overload in same_arity
    )
    raise TypeError(
        f"{spec.name} operands do not match any declared role signature: {declared}"
    )


def _validate_overload_plan(
    spec: OperationSpec, overload: OperationOverload, plan: OperationPlan
) -> None:
    """Reject a resolver whose returned plan disagrees with its registration."""
    if plan.operation != spec.name:
        raise ValueError(
            f"operation {spec.name!r} resolver returned a plan for {plan.operation!r}"
        )
    roles = tuple(operand.role for operand in plan.operands)
    if roles != overload.roles:
        raise ValueError(
            f"operation {spec.name!r} resolver returned operand roles {roles!r}; "
            f"expected {overload.roles!r}"
        )


def resolve_operation_plan(operation: str, *operands: object) -> OperationPlan:
    """Resolve the shared :class:`OperationPlan` for one operation invocation.

    This is the single entry point every backend uses. Tensor operands are given
    as their carrier storage dtype and weak scalar operands as the Python value
    itself, positionally, in the operation's own argument order.

    Args:
        operation: Operation name, matching the ``dispatch_op`` name (for
            example ``"add"``, ``"mul"``, ``"pow"``, ``"matmul"``, ``"reduce_sum"``,
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
    overload = _select_overload(spec, operands)
    plan = overload.rule(*operands)
    _validate_overload_plan(spec, overload, plan)
    return plan
