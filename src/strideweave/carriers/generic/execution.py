"""Plan-driven arithmetic execution for the Generic reference carrier.

Generic runs one code path per operation. What differs between a legacy opaque
tensor and a concrete simple-dtype tensor is the *arithmetic* the path executes,
so the operation resolves an arithmetic here and expresses its formula in terms
of that object's primitives.

Concrete operands resolve a plan from
:mod:`strideweave.carriers.operation_policy` and execute it faithfully:
``Float32`` in IEEE-754 binary32, ``Int32`` exactly with checked narrowing.
Operands that are not concrete simple dtypes — the legacy ``DType.Any`` and
``DType.Floating`` storage — keep Generic's historical Python arithmetic, and a
tensor mixing legacy and concrete storage stays on that legacy path rather than
silently selecting a concrete plan for it.

Which plan shapes Generic executes is a single decision recorded in the backend
capability registry and declared in
:mod:`strideweave.carriers.generic.capabilities`. The primitives below —
conversions, accumulations, and narrowings — are the implementation those
declarations describe, so this module carries no second table of accepted
shapes: :func:`arithmetic_for_plan` requires a declared capability and refuses
everything else before converting a value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Final

from ..dtype import DType
from ..operation_capability import require_capability
from ..operation_helpers import _require_number
from ..operation_policy import Accumulation as AccumulationKind
from ..operation_policy import Arithmetic as ArithmeticKind
from ..operation_policy import OperandRole, OperationPlan, resolve_operation_plan
from .numerics import (
    binary32,
    checked_int32,
    float32_errstate,
    float32_scalar,
    is_concrete_simple_dtype,
)

__all__ = [
    "GenericArithmetic",
    "arithmetic_for_plan",
    "binary_arithmetic",
    "executable_plan_shape",
    "executing",
    "extrema_total",
    "gradient_arithmetic",
    "scalar_arithmetic",
    "unary_arithmetic",
]


class GenericArithmetic:
    """The arithmetic one Generic operation executes, plus its result dtype.

    Args:
        result_dtype: Dtype the operation's result carrier reports, or ``None``
            to preserve the operand's dtype the way Generic historically does.
        plan: The resolved plan this arithmetic executes, or ``None`` on the
            legacy path.
    """

    def __init__(
        self, result_dtype: DType | None, plan: OperationPlan | None = None
    ) -> None:
        self.result_dtype = result_dtype
        self.plan = plan

    @property
    def is_planned(self) -> bool:
        """Whether this arithmetic executes a resolved plan."""
        return self.plan is not None

    def scope(self) -> Any:
        """Return the context an operation loop runs inside."""
        return nullcontext()

    def convert(self, value: Any) -> Any:
        """Materialize one operand value into the compute representation."""
        return value

    def total(self, values: Iterable[Any]) -> Any:
        """Combine many terms in the plan's accumulation order."""
        result: Any = None
        for value in values:
            result = value if result is None else result + value
        return 0 if result is None else result

    def store(self, value: Any) -> Any:
        """Narrow a computed value into the result's stored representation."""
        return value


def _float32_binary32(value: Any) -> Any:
    return float32_scalar(value)


# How an operand is materialized into the dtype its OperandPlan names. This is
# the only place `convert_to` is realized, so a conversion target Generic cannot
# materialize is rejected rather than approximated.
_CONVERSIONS: dict[DType, Any] = {
    DType.Float32: _float32_binary32,
    DType.Int32: int,
}


def _sequential_binary32(values: Iterable[Any]) -> Any:
    """Add each term in the caller's order, rounding in binary32 every step."""
    result = float32_scalar(0.0)
    for value in values:
        result = result + value
    return result


def _sequential_binary32_product(values: Iterable[Any]) -> Any:
    """Multiply each term in caller order, rounding in binary32 each step."""
    result = float32_scalar(1.0)
    for value in values:
        result = result * value
    return result


def _exact_integer(values: Iterable[Any]) -> Any:
    """Add every term exactly; only the final narrowing is checked."""
    return sum(int(value) for value in values)


def _is_nan(value: Any) -> bool:
    """Return whether a planned Float32 value is NaN."""
    try:
        return math.isnan(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _zero_sign(value: Any) -> int:
    """Return the sign of a numeric zero as ``-1`` or ``1``."""
    return -1 if math.copysign(1.0, float(value)) < 0 else 1


def _prefer_zero(candidate: Any, winner: Any, *, maximum: bool) -> bool:
    """Return whether a signed-zero candidate wins a max/min tie."""
    if candidate != 0 or winner != 0:
        return False
    candidate_sign = _zero_sign(candidate)
    winner_sign = _zero_sign(winner)
    return candidate_sign > winner_sign if maximum else candidate_sign < winner_sign


def extrema_total(values: Iterable[Any], *, maximum: bool) -> Any:
    """Return a NumPy-compatible maximum or minimum in logical order.

    The first NaN is retained. Equal finite values keep the first winner,
    except that mixed signed zeros select positive zero for a maximum and
    negative zero for a minimum.
    """
    values = list(values)
    if not values:
        return float("nan")
    winner = values[0]
    winner_is_nan = _is_nan(winner)
    for value in values[1:]:
        value_is_nan = _is_nan(value)
        if winner_is_nan:
            continue
        if value_is_nan:
            winner = value
            winner_is_nan = True
        elif (value > winner if maximum else value < winner) or _prefer_zero(
            value, winner, maximum=maximum
        ):
            winner = value
    return winner


def _maximum(values: Iterable[Any]) -> Any:
    """Return a NumPy-compatible Float32 maximum in logical order."""
    return extrema_total(values, maximum=True)


def _minimum(values: Iterable[Any]) -> Any:
    """Return a NumPy-compatible Float32 minimum in logical order."""
    return extrema_total(values, maximum=False)


def _argmax(values: Iterable[Any]) -> int:
    """Return the first Float32 maximum ordinal in logical order.

    NaN wins over all numeric values and the first NaN wins among NaNs. Equal
    numeric values, including signed zeros, retain their first ordinal.
    """
    values = list(values)
    if not values:
        return 0
    winner = 0
    winner_value = values[0]
    winner_is_nan = _is_nan(winner_value)
    for index, value in enumerate(values[1:], 1):
        value_is_nan = _is_nan(value)
        if winner_is_nan:
            continue
        if value_is_nan or value > winner_value:
            winner = index
            winner_value = value
            winner_is_nan = value_is_nan
    return winner


def _argmin(values: Iterable[Any]) -> int:
    """Return the first Float32 minimum ordinal in logical order.

    NaN wins over all numeric values and the first NaN wins among NaNs. Equal
    numeric values, including signed zeros, retain their first ordinal.
    """
    values = list(values)
    if not values:
        return 0
    winner = 0
    winner_value = values[0]
    winner_is_nan = _is_nan(winner_value)
    for index, value in enumerate(values[1:], 1):
        value_is_nan = _is_nan(value)
        if winner_is_nan:
            continue
        if value_is_nan or value < winner_value:
            winner = index
            winner_value = value
            winner_is_nan = value_is_nan
    return winner


# How terms combine, one entry per Accumulation member.
_ACCUMULATIONS: dict[AccumulationKind, Any] = {
    AccumulationKind.SEQUENTIAL_BINARY32: _sequential_binary32,
    AccumulationKind.SEQUENTIAL_BINARY32_PRODUCT: _sequential_binary32_product,
    AccumulationKind.EXACT_INTEGER: _exact_integer,
    AccumulationKind.MAXIMUM: _maximum,
    AccumulationKind.MINIMUM: _minimum,
    AccumulationKind.ARGMAX: _argmax,
    AccumulationKind.ARGMIN: _argmin,
}

# The operations whose Generic implementations combine many terms into one
# result. Every other operation writes one result per element and has nowhere to
# apply an accumulation, so declaring one for it would advertise a loop Generic
# does not run.
_ACCUMULATING_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
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
)

# Gather/scatter and select have intentionally mixed representations: value
# operands compute in Float32 while logical indices remain Int32 and select's
# condition remains Bool. The operation policy specifies these exact shapes;
# Generic merely recognizes them rather than choosing a promotion locally.
_MIXED_CONVERSION_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"gather", "scatter", "scatter_add", "select"}
)

_MIXED_CONVERSION_TARGETS: Final[dict[str, tuple[DType, ...]]] = {
    "gather": (DType.Float32, DType.Int32),
    "scatter": (DType.Float32, DType.Int32, DType.Float32),
    "scatter_add": (DType.Float32, DType.Int32, DType.Float32),
    "select": (DType.Bool, DType.Float32, DType.Float32),
}

type _OperandShape = tuple[OperandRole, DType | None, DType]

_EXACT_OPERAND_SHAPES: Final[dict[str, frozenset[tuple[_OperandShape, ...]]]] = {
    "select": frozenset(
        {
            (
                (OperandRole.TENSOR, DType.Bool, DType.Bool),
                (OperandRole.TENSOR, DType.Float32, DType.Float32),
                (OperandRole.TENSOR, DType.Float32, DType.Float32),
            )
        }
    ),
    "clamp": frozenset(
        {
            (
                (OperandRole.TENSOR, DType.Float32, DType.Float32),
                (
                    lower_role,
                    DType.Float32 if lower_role is OperandRole.TENSOR else None,
                    DType.Float32,
                ),
                (
                    upper_role,
                    DType.Float32 if upper_role is OperandRole.TENSOR else None,
                    DType.Float32,
                ),
            )
            for lower_role in (OperandRole.TENSOR, OperandRole.WEAK_SCALAR)
            for upper_role in (OperandRole.TENSOR, OperandRole.WEAK_SCALAR)
        }
    ),
}

_SPECIAL_ACCUMULATIONS: Final[dict[str, dict[ArithmeticKind, AccumulationKind]]] = {
    "reduce_sum": {
        ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
        ArithmeticKind.INT32_EXACT_CHECKED: AccumulationKind.EXACT_INTEGER,
    },
    "reduce_prod": {
        ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32_PRODUCT,
    },
    "reduce_max": {ArithmeticKind.BINARY32: AccumulationKind.MAXIMUM},
    "reduce_min": {ArithmeticKind.BINARY32: AccumulationKind.MINIMUM},
    "argmax": {ArithmeticKind.BINARY32: AccumulationKind.ARGMAX},
    "argmin": {ArithmeticKind.BINARY32: AccumulationKind.ARGMIN},
    "cumsum": {ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32},
    "matmul": {
        ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
        ArithmeticKind.INT32_EXACT: AccumulationKind.EXACT_INTEGER,
    },
    "conv_general": {
        ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
    },
    "scatter_add": {
        ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
    },
}

_DIFFERENT_OUTPUT_OPERATIONS: Final[dict[str, DType]] = {
    "eq": DType.Bool,
    "ne": DType.Bool,
    "lt": DType.Bool,
    "le": DType.Bool,
    "logical_not": DType.Bool,
    "argmax": DType.Int32,
    "argmin": DType.Int32,
    "_sort_indices": DType.Int32,
    "_topk_indices": DType.Int32,
}

# The single element representation each arithmetic kernel computes in. Most
# Generic operations convert every operand into this dtype and store the result
# in it. Predicate/index operations intentionally store Bool/Int32 instead,
# while gather/scatter/select have dedicated mixed operand loops.
_COMPUTE_DTYPES: Final[dict[ArithmeticKind, DType]] = {
    ArithmeticKind.BINARY32: DType.Float32,
    ArithmeticKind.INT32_EXACT_CHECKED: DType.Int32,
    ArithmeticKind.INT32_EXACT: DType.Int32,
}

# The default accumulation associated with each compute arithmetic. Operations
# with specialized reductions (product, extrema, arg reductions, convolution,
# and scatter-add) are narrowed further by `_SPECIAL_ACCUMULATIONS` below.
_COMPUTE_ACCUMULATIONS: Final[dict[ArithmeticKind, AccumulationKind]] = {
    ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
    ArithmeticKind.INT32_EXACT_CHECKED: AccumulationKind.EXACT_INTEGER,
    ArithmeticKind.INT32_EXACT: AccumulationKind.EXACT_INTEGER,
}

# How a computed value is narrowed into the dtype `output` names. Bool predicate
# outputs and Int32 index outputs intentionally differ from the Float32 compute
# representation; these stores are still concrete Generic representations.
_STORES: dict[DType, Any] = {
    DType.Bool: bool,
    DType.Float32: binary32,
    DType.Int32: checked_int32,
}


def _plan_conversion(plan: OperationPlan) -> DType:
    """Return the compute dtype for a plan's arithmetic adapter.

    Most Generic arithmetic uses one conversion target for every operand.
    Gather, scatter, and select are deliberate exceptions: their dedicated
    loops preserve Int32 indices or Bool conditions while values use Float32.
    """
    conversions = {operand.convert_to for operand in plan.operands}
    if len(conversions) == 1:
        return conversions.pop()
    # Mixed plans are executed by dedicated indexing/selection loops that
    # convert operands independently. Returning the value representation keeps
    # their capability adapters inspectable; they never route through the
    # single-conversion GenericArithmetic.convert path.
    if plan.operation in _MIXED_CONVERSION_OPERATIONS:
        return DType.Float32
    raise ValueError("Generic plans must use one compute representation")


def _plan_store(plan: OperationPlan) -> Any:
    """Return the narrowing that writes a computed value into ``plan.output``.

    ``output`` selects the representation; ``compute`` says whether reaching it
    needs a check. :attr:`Arithmetic.INT32_EXACT` is the policy's promise that
    the result provably fits, except that exact integer accumulations still
    check their final narrowing; Bool predicates and Int32 indices are selected
    explicitly by the resolved output dtype.
    """
    if plan.output is DType.Int32 and plan.compute is ArithmeticKind.INT32_EXACT:
        # An accumulated integer result is still checked at its final narrowing,
        # because a sum of exact terms need not fit.
        if plan.accumulation is not AccumulationKind.EXACT_INTEGER:
            return int
    return _STORES[plan.output]


def _plan_total(plan: OperationPlan) -> Any:
    """Return the term combination ``plan.accumulation`` asks for, if any."""
    if plan.accumulation is None:
        return None
    return _ACCUMULATIONS[plan.accumulation]


def executable_plan_shape(plan: OperationPlan) -> bool:
    """Report whether Generic's primitives can assemble ``plan`` as written.

    This is the predicate Generic's capability declarations are filtered
    through, so the set of shapes it advertises and the set its adapter can
    actually build stay one decision. It asks only what this backend
    implements; which plan is correct for a given operand pair is central
    policy, decided before a plan ever reaches here.

    A plan is executable as a whole or not at all. Accepting each field
    separately would advertise combinations Generic cannot honor together — a
    binary32 computation stored as ``Int32`` truncates every fractional value,
    and an exact-integer accumulation over binary32 terms truncates each term
    before adding it — so the fields are checked against each other rather than
    one at a time.

    Args:
        plan: A resolved operation plan.

    Returns:
        ``True`` when the operation's operand conversions, compute arithmetic,
        output dtype, and accumulation agree with a Generic implementation.
        Gather/scatter/select use their explicit mixed conversion shapes;
        predicates return Bool and arg/selection index operations return Int32
        while computing their comparison or ordering in Float32. Clamp accepts
        exactly its four tensor/weak-scalar role signatures.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.generic.execution import executable_plan_shape
        >>> executable_plan_shape(resolve_operation_plan("reduce_sum", DType.Int32))
        True
    """
    exact_shapes = _EXACT_OPERAND_SHAPES.get(plan.operation)
    operand_shape = tuple(
        (operand.role, operand.dtype, operand.convert_to) for operand in plan.operands
    )
    if exact_shapes is not None and operand_shape not in exact_shapes:
        return False

    representation = _COMPUTE_DTYPES.get(plan.compute)
    if representation is None:
        return False
    if plan.operation not in _MIXED_CONVERSION_OPERATIONS and any(
        operand.convert_to is not representation for operand in plan.operands
    ):
        return False
    if plan.operation in _MIXED_CONVERSION_OPERATIONS:
        expected = _MIXED_CONVERSION_TARGETS[plan.operation]
        if tuple(operand.convert_to for operand in plan.operands) != expected:
            return False
    expected_output = _DIFFERENT_OUTPUT_OPERATIONS.get(plan.operation, representation)
    if plan.output is not expected_output:
        return False
    if plan.operation not in _ACCUMULATING_OPERATIONS:
        return plan.accumulation is None
    if plan.operation in _SPECIAL_ACCUMULATIONS:
        return (
            _SPECIAL_ACCUMULATIONS[plan.operation].get(plan.compute)
            is plan.accumulation
        )
    return plan.accumulation is _COMPUTE_ACCUMULATIONS[plan.compute]


class _PlannedArithmetic(GenericArithmetic):
    """The arithmetic a resolved plan describes, assembled from its fields.

    Every field is realized by exactly one part of this object: ``convert_to``
    by :meth:`convert`, ``compute`` by the conversion and error-state scope,
    ``accumulation`` by :meth:`total`, and ``output`` by :meth:`store` and the
    reported result dtype. Nothing is re-derived from ``compute`` alone, so a
    revised policy cannot leave Generic executing a stale implication.
    """

    def __init__(self, plan: OperationPlan) -> None:
        super().__init__(plan.output, plan)
        conversion = _plan_conversion(plan)
        self._convert = _CONVERSIONS[conversion]
        self._store = _plan_store(plan)
        self._total = _plan_total(plan)
        self._binary32 = plan.compute is ArithmeticKind.BINARY32

    def scope(self) -> Any:
        # Binary32 computation raises no exceptions at IEEE singularities, so
        # NumPy's error state is suppressed for the whole operation.
        return float32_errstate() if self._binary32 else nullcontext()

    def convert(self, value: Any) -> Any:
        return self._convert(value)

    def total(self, values: Iterable[Any]) -> Any:
        if self._total is None:
            raise ValueError(
                f"Generic cannot combine terms for {self.plan.operation!r}: "
                "its plan declares no accumulation"
                if self.plan is not None
                else "Generic cannot combine terms without an accumulation"
            )
        return self._total(values)

    def store(self, value: Any) -> Any:
        return self._store(value)


class _GradientArithmetic(GenericArithmetic):
    """Binary32 arithmetic for a backward pass.

    Backward is governed by the policy's fixed backward rule rather than by a
    forward plan: a gradient is always ``Float32`` computed in binary32,
    whatever dtype the forward operation produced. It therefore carries no plan
    and is built directly rather than through :func:`arithmetic_for_plan`.
    """

    def __init__(self) -> None:
        super().__init__(DType.Float32, None)

    def scope(self) -> Any:
        return float32_errstate()

    def convert(self, value: Any) -> Any:
        return float32_scalar(value)

    def total(self, values: Iterable[Any]) -> Any:
        return _sequential_binary32(values)

    def store(self, value: Any) -> Any:
        return binary32(value)


def arithmetic_for_plan(plan: OperationPlan, carrier_class: type) -> GenericArithmetic:
    """Build the arithmetic that executes ``plan`` on ``carrier_class``.

    This is Generic's adapter boundary, and the backend capability registry is
    the only gate it has: a plan the carrier class declares no capability for is
    refused here, before any value is converted, rather than assembled out of
    whichever primitives happen to fit. The accepted plan's fields are then the
    sole source of execution behavior.

    Args:
        plan: A resolved operation plan.
        carrier_class: The exact carrier class about to execute the plan, whose
            declared capabilities decide whether it may.

    Returns:
        The arithmetic executing exactly what ``plan`` describes.

    Raises:
        UnsupportedOperationPlan: If ``carrier_class`` declares no capability
            for that exact shape.
    """
    require_capability(carrier_class, plan)
    return _PlannedArithmetic(plan)


def _concrete_dtype(tensor: Any) -> DType | None:
    dtype = tensor.carrier.dtype()
    return dtype if is_concrete_simple_dtype(dtype) else None


def _executing_class(tensor: Any) -> type:
    """Return the exact carrier class whose capabilities gate this operation.

    Dispatch reaches an operation through one carrier, and that carrier's exact
    class is what declares capabilities, so a `Generic` subclass declaring extra
    shapes is asked about its own set rather than its base's.
    """
    return type(tensor.carrier)


def _planned(operation: str, *operands: Any) -> OperationPlan | None:
    """Resolve a plan, or return ``None`` when any operand is legacy storage."""
    if any(operand is None for operand in operands):
        return None
    return resolve_operation_plan(operation, *operands)


def binary_arithmetic(
    operation: str, lhs: Any, rhs: Any, legacy_dtype: DType
) -> GenericArithmetic:
    """Resolve the arithmetic for a two-tensor operation.

    Args:
        operation: Registered operation name.
        lhs: Left tensor operand.
        rhs: Right tensor operand.
        legacy_dtype: Result dtype Generic reports on its legacy path.

    Returns:
        The arithmetic to execute, planned when both operands are concrete.

    Raises:
        UnsupportedOperationPlan: If the operands' carrier class declares no
            capability for the resolved plan.
    """
    plan = _planned(operation, _concrete_dtype(lhs), _concrete_dtype(rhs))
    if plan is None:
        return GenericArithmetic(legacy_dtype)
    return arithmetic_for_plan(plan, _executing_class(lhs))


def unary_arithmetic(
    operation: str, tensor: Any, legacy_dtype: DType | None
) -> GenericArithmetic:
    """Resolve the arithmetic for a one-tensor operation.

    Args:
        operation: Registered operation name.
        tensor: The tensor operand.
        legacy_dtype: Result dtype Generic reports on its legacy path, or
            ``None`` to preserve the operand's dtype.

    Returns:
        The arithmetic to execute, planned when the operand is concrete.

    Raises:
        UnsupportedOperationPlan: If the operand's carrier class declares no
            capability for the resolved plan.
    """
    plan = _planned(operation, _concrete_dtype(tensor))
    if plan is None:
        return GenericArithmetic(legacy_dtype)
    return arithmetic_for_plan(plan, _executing_class(tensor))


def scalar_arithmetic(
    operation: str,
    tensor: Any,
    scalar: Any,
    legacy_dtype: DType,
    scalar_name: str = "scalar",
) -> GenericArithmetic:
    """Resolve the arithmetic for a tensor-and-weak-scalar operation.

    The scalar is validated by whichever rule governs the path taken. A
    concrete tensor lets the shared policy reject the scalar, so every backend
    refuses the same values with the same diagnostic; legacy opaque storage has
    no plan and keeps its own historical check, which admits any
    ``numbers.Number`` rather than only a real one.

    Args:
        operation: Registered operation name.
        tensor: The tensor operand.
        scalar: The weak Python scalar operand.
        legacy_dtype: Result dtype Generic reports on its legacy path.
        scalar_name: Operand name used in the legacy path's diagnostic.

    Returns:
        The arithmetic to execute, planned when the tensor is concrete.

    Raises:
        UnsupportedOperationPlan: If the tensor's carrier class declares no
            capability for the resolved plan.
    """
    dtype = _concrete_dtype(tensor)
    if dtype is None:
        _require_number(scalar, scalar_name)
        return GenericArithmetic(legacy_dtype)
    return arithmetic_for_plan(
        resolve_operation_plan(operation, dtype, scalar), _executing_class(tensor)
    )


def gradient_arithmetic(tensor: Any) -> GenericArithmetic:
    """Resolve the arithmetic a backward pass executes for ``tensor``.

    Gradients are always ``Float32`` for a concrete differentiable operand, so
    backward runs the same binary32 mechanics as forward — including at IEEE
    singularities, which produce infinities and NaNs rather than exceptions.

    Args:
        tensor: The forward operand a gradient is being produced for.

    Returns:
        Binary32 arithmetic for a concrete operand, legacy arithmetic otherwise.
    """
    if _concrete_dtype(tensor) is None:
        return GenericArithmetic(None)
    return _GradientArithmetic()


@contextmanager
def executing(arithmetic: GenericArithmetic) -> Iterator[GenericArithmetic]:
    """Run an operation loop inside its arithmetic's error-state scope.

    The scope is entered once per operation rather than once per element, so
    IEEE error state is never configured inside the element loop.

    Args:
        arithmetic: The arithmetic whose scope the loop runs in.

    Yields:
        The same arithmetic, for convenient ``with`` binding.

    Examples:
        >>> from strideweave.carriers.generic.execution import (
        ...     GenericArithmetic,
        ...     executing,
        ... )
        >>> with executing(GenericArithmetic(None)) as arithmetic:
        ...     arithmetic.store(2)
        2
    """
    with arithmetic.scope():
        yield arithmetic
