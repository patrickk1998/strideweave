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

from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Final

from ..dtype import DType
from ..operation_capability import require_capability
from ..operation_helpers import _require_number
from ..operation_policy import Accumulation as AccumulationKind
from ..operation_policy import Arithmetic as ArithmeticKind
from ..operation_policy import OperationPlan, resolve_operation_plan
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


def _exact_integer(values: Iterable[Any]) -> Any:
    """Add every term exactly; only the final narrowing is checked."""
    return sum(int(value) for value in values)


# How terms combine, one entry per Accumulation member.
_ACCUMULATIONS: dict[AccumulationKind, Any] = {
    AccumulationKind.SEQUENTIAL_BINARY32: _sequential_binary32,
    AccumulationKind.EXACT_INTEGER: _exact_integer,
}

# The operations whose Generic implementations combine many terms into one
# result. Every other operation writes one result per element and has nowhere to
# apply an accumulation, so declaring one for it would advertise a loop Generic
# does not run.
_ACCUMULATING_OPERATIONS: Final[frozenset[str]] = frozenset({"matmul", "reduce"})

# The single element representation each compute arithmetic runs in. Generic
# materializes one representation per operation, so this is simultaneously the
# dtype every operand converts into and the dtype the result is stored in: a
# binary32 computation cannot be stored as `Int32` without truncating, and an
# exact-integer computation cannot be reached from `Float32` operands. Every
# representation named here is a key of both `_CONVERSIONS` and `_STORES`, which
# is what keeps an accepted shape assemblable from the primitives above.
_COMPUTE_DTYPES: Final[dict[ArithmeticKind, DType]] = {
    ArithmeticKind.BINARY32: DType.Float32,
    ArithmeticKind.INT32_EXACT_CHECKED: DType.Int32,
    ArithmeticKind.INT32_EXACT: DType.Int32,
}

# The accumulation each compute arithmetic combines terms with when it combines
# them at all. Binary32 terms must round at every step and integer terms must
# stay exact, so the pairing is the arithmetic's, not the operation's.
_COMPUTE_ACCUMULATIONS: Final[dict[ArithmeticKind, AccumulationKind]] = {
    ArithmeticKind.BINARY32: AccumulationKind.SEQUENTIAL_BINARY32,
    ArithmeticKind.INT32_EXACT_CHECKED: AccumulationKind.EXACT_INTEGER,
    ArithmeticKind.INT32_EXACT: AccumulationKind.EXACT_INTEGER,
}

# How a computed value is narrowed into the dtype `output` names. Together with
# `_CONVERSIONS` and `_ACCUMULATIONS` these are Generic's implemented primitives:
# a plan shape they cannot assemble is one Generic never declares a capability
# for, so it is refused at the capability boundary rather than approximated here.
_STORES: dict[DType, Any] = {
    DType.Float32: binary32,
    DType.Int32: checked_int32,
}


def _plan_conversion(plan: OperationPlan) -> DType:
    """Return the single dtype every operand of ``plan`` is materialized into.

    Generic computes one element representation per operation, so a uniform
    conversion target is part of what makes a plan shape executable at all.
    """
    (target,) = {operand.convert_to for operand in plan.operands}
    return target


def _plan_store(plan: OperationPlan) -> Any:
    """Return the narrowing that writes a computed value into ``plan.output``.

    ``output`` selects the representation; ``compute`` says whether reaching it
    needs a check. :attr:`Arithmetic.INT32_EXACT` is the policy's promise that
    the result provably fits, and the one accumulating exception is explicit
    below, so nothing here re-derives the result dtype from the arithmetic.
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
        ``True`` when every operand converts into the single representation
        ``compute`` runs in, the result is stored in that same representation,
        and an accumulation of that arithmetic's kind is present exactly for the
        operations whose Generic loops combine terms.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.generic.execution import executable_plan_shape
        >>> executable_plan_shape(resolve_operation_plan("reduce", DType.Int32))
        True
    """
    representation = _COMPUTE_DTYPES.get(plan.compute)
    if representation is None:
        return False
    if any(operand.convert_to is not representation for operand in plan.operands):
        return False
    if plan.output is not representation:
        return False
    if plan.operation not in _ACCUMULATING_OPERATIONS:
        return plan.accumulation is None
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
