"""Backend capability registry for already-resolved operation plans.

:mod:`strideweave.carriers.operation_policy` decides *what* an operation must do
for a set of operand dtypes. This module records *which of those decisions a
particular backend can actually carry out*. The two are deliberately separate:
policy is central and identical for every carrier, while capability is local to
one exact carrier class and describes only its implementation reach.

A capability is an exact resolved-plan shape. A backend declares the shapes it
faithfully executes, and every planned execution must be matched against those
declarations before the backend allocates a result or enters a kernel. Nothing
here promotes, rewrites, or substitutes: a plan that matches no declaration is
refused with one explicit diagnostic
(:class:`UnsupportedOperationPlan`), never replaced by a nearby shape the
backend happens to implement. That refusal is what ``RT013`` means by "a backend
that cannot execute a plan raises".

Capabilities are owned by an exact carrier class, the same granularity carrier
dispatch and profiling already use, and ownership stops there: a class's
declarations are exactly the entries declared for it, never entries resolved
through its bases. Support is a claim about an implementation, and an
implementation is not inherited — a class that declares nothing supports
nothing, and declaring one shape somewhere in a hierarchy cannot make another
class advertise a kernel it never wrote.

That exact-class model is the default, and it fits an *independent* carrier: one
whose reach is fixed by the kernels it ships. A carrier assembled out of other
carriers is not independent — what it can execute depends on the instances it
was handed — so :class:`DependentCarrier` is the second ownership model, where
capabilities belong to one constructed instance rather than to a class. Both
models live here because both answer the same question about the same entries;
which one a carrier uses is a property of the carrier, not of the caller.

Two paths reach the registry. Every shipped backend declares once, during
carrier-package initialization, through the framework-internal bootstrap in
:mod:`strideweave.carriers._built_in_capabilities` — including the storage and
composition carriers, whose declaration is the empty set they intentionally
state. That path is private and exported nowhere, so it is not an escape hatch a
caller can reach. Everything else goes through
:func:`register_operation_capabilities`, the one public path, which accepts only
an independent ``Carrier`` implementation — never ``object``, the ``Carrier``
root, an unrelated class, a ``DependentCarrier`` class, or a sealed built-in —
so no public call can widen what a shipped backend claims to execute. A backend
built by composition declares the shapes it executes for its own exact class,
exactly as a shipped one does.

Either way a class declares exactly once, and declaring seals: the complete set
is published and closed in one lock transaction, so no observer sees part of a
declaration and no later call can add to one. Observation seals too. An
independent class that is queried, or whose plan is required, before it has
declared is sealed against the empty set right there, which makes first
observation the final answer instead of a temporary one. Without that, a
carrier that snapshots another class's reach — as a dependent carrier does —
could be told "nothing" and then be silently wrong once a later registration
widened it.

The same stored entries answer both questions asked of a backend: whether an
execution may proceed, and what a caller enumerating support is told. There is
no second informational table to drift out of agreement with the executed one.

Only forward plans are gated here. A backward pass is governed by the policy's
fixed backward rule — every gradient is ``Float32`` computed in binary32,
whatever dtype the forward operation produced — rather than by a resolved plan,
so it carries no plan to match and deliberately does not consult this registry.
See section 9 of ``design/SimpleDType-operation-policy.md``.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable, Mapping, MutableMapping, MutableSet
from dataclasses import dataclass
from functools import partial
from typing import Final, cast

from .base import Carrier
from .dtype import SimpleDType
from .operation_policy import (
    Accumulation,
    Arithmetic,
    OperandRole,
    OperationPlan,
)

__all__ = [
    "DependentCarrier",
    "OperandCapability",
    "OperationCapability",
    "UnsupportedOperationPlan",
    "capabilities_for_carrier_class",
    "matching_capability",
    "register_operation_capabilities",
    "require_capability",
    "supports_operation_plan",
    "unsupported_plan_reason",
]


class UnsupportedOperationPlan(NotImplementedError):
    """A backend was asked to execute a plan it declares no capability for.

    Raised by :func:`require_capability` before any allocation or kernel work, so
    an unsupported plan fails once, explicitly, rather than falling through to a
    branch implementing a different shape.
    """


@dataclass(frozen=True, slots=True)
class OperandCapability:
    """One operand position of an executable plan shape.

    Mirrors :class:`~strideweave.carriers.operation_policy.OperandPlan`: a
    capability accepts an operand exactly when the plan's role, storage dtype,
    and conversion target all match.

    Args:
        role: Whether this position takes a tensor or a weak Python scalar.
        dtype: The tensor operand's storage dtype, or ``None`` for a weak
            scalar, which has no dtype of its own.
        convert_to: The dtype this operand is materialized into before the
            computation runs.
    """

    role: OperandRole
    dtype: SimpleDType | None
    convert_to: SimpleDType

    def __post_init__(self) -> None:
        if not isinstance(self.role, OperandRole):
            raise TypeError("role must be an OperandRole")
        if not isinstance(self.convert_to, SimpleDType):
            raise TypeError("convert_to must be a SimpleDType")
        if self.role is OperandRole.TENSOR:
            if not isinstance(self.dtype, SimpleDType):
                raise TypeError("a tensor operand capability needs a storage dtype")
        elif self.dtype is not None:
            raise TypeError("a weak scalar operand capability has no storage dtype")


@dataclass(frozen=True, slots=True)
class OperationCapability:
    """One resolved-plan shape a backend declares it executes faithfully.

    The fields are exactly the fields of an
    :class:`~strideweave.carriers.operation_policy.OperationPlan`, because a
    capability answers one question about a plan: can this backend run *this*,
    as written. It carries no promotion rule, no preferred alternative, and no
    execution behavior of its own.

    Args:
        operation: Operation name the shape applies to.
        operands: One :class:`OperandCapability` per operand, positionally.
        compute: Arithmetic the per-element computation performs.
        accumulation: How terms combine, or ``None`` for a shape that combines
            no terms. This distinction is operation-specific and is matched
            exactly: an accumulating backend branch must not be reached by a
            plan that declares no accumulation.
        output: The dtype the result carrier reports.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import OperationCapability
        >>> plan = resolve_operation_plan("add", DType.Int32, DType.Int32)
        >>> OperationCapability.from_plan(plan).matches(plan)
        True
    """

    operation: str
    operands: tuple[OperandCapability, ...]
    compute: Arithmetic
    accumulation: Accumulation | None
    output: SimpleDType

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str):
            raise TypeError("operation must be a str")
        if not isinstance(self.operands, tuple):
            raise TypeError("operands must be a tuple of OperandCapability")
        if not self.operands:
            raise ValueError("an operation capability needs at least one operand")
        if not all(isinstance(operand, OperandCapability) for operand in self.operands):
            raise TypeError("operands must be a tuple of OperandCapability")
        if not isinstance(self.compute, Arithmetic):
            raise TypeError("compute must be an Arithmetic")
        if self.accumulation is not None and not isinstance(
            self.accumulation, Accumulation
        ):
            raise TypeError("accumulation must be an Accumulation or None")
        if not isinstance(self.output, SimpleDType):
            raise TypeError("output must be a SimpleDType")

    @classmethod
    def from_plan(cls, plan: OperationPlan) -> OperationCapability:
        """Return the capability that accepts exactly ``plan``'s shape.

        Backends declare their support by resolving the plans they are known to
        execute and recording those shapes, so a declaration cannot describe a
        shape the central policy never produces without saying so explicitly.

        Args:
            plan: A resolved operation plan.

        Returns:
            The capability matching that plan and no other shape.

        Examples:
            >>> from strideweave.carriers.dtype import DType
            >>> from strideweave.carriers.operation_policy import (
            ...     resolve_operation_plan,
            ... )
            >>> from strideweave.carriers.operation_capability import (
            ...     OperationCapability,
            ... )
            >>> plan = resolve_operation_plan("relu", DType.Int32)
            >>> OperationCapability.from_plan(plan).output is DType.Int32
            True
        """
        return cls(
            operation=plan.operation,
            operands=tuple(
                OperandCapability(
                    role=operand.role,
                    dtype=operand.dtype,
                    convert_to=operand.convert_to,
                )
                for operand in plan.operands
            ),
            compute=plan.compute,
            accumulation=plan.accumulation,
            output=plan.output,
        )

    def matches(self, plan: OperationPlan) -> bool:
        """Report whether ``plan`` is exactly the shape this capability accepts.

        Args:
            plan: A resolved operation plan.

        Returns:
            ``True`` when every field agrees, including operand count, order,
            per-operand conversion, and the presence or absence of an
            accumulation.

        Examples:
            >>> from strideweave.carriers.dtype import DType
            >>> from strideweave.carriers.operation_policy import (
            ...     resolve_operation_plan,
            ... )
            >>> from strideweave.carriers.operation_capability import (
            ...     OperationCapability,
            ... )
            >>> capability = OperationCapability.from_plan(
            ...     resolve_operation_plan("add", DType.Int32, DType.Int32)
            ... )
            >>> capability.matches(resolve_operation_plan("add", DType.Float32,
            ...                                          DType.Int32))
            False
        """
        return _plan_key(plan) == _capability_key(self)


# The exact shape two entries must agree on to be the same capability. Dtype
# descriptors are registry singletons (`SW002`), so their hashing and comparison
# inside this key are identity, not value, comparisons.
_CapabilityKey = tuple[
    str,
    tuple[tuple[OperandRole, SimpleDType | None, SimpleDType], ...],
    Arithmetic,
    Accumulation | None,
    SimpleDType,
]


def _capability_key(capability: OperationCapability) -> _CapabilityKey:
    return (
        capability.operation,
        tuple(
            (operand.role, operand.dtype, operand.convert_to)
            for operand in capability.operands
        ),
        capability.compute,
        capability.accumulation,
        capability.output,
    )


def _plan_key(plan: OperationPlan) -> _CapabilityKey:
    return (
        plan.operation,
        tuple(
            (operand.role, operand.dtype, operand.convert_to)
            for operand in plan.operands
        ),
        plan.compute,
        plan.accumulation,
        plan.output,
    )


def _sort_key(capability: OperationCapability) -> tuple[str, ...]:
    """Return the total order enumeration reports capabilities in.

    Enumeration must be deterministic across processes, so the order is derived
    from stable names rather than from registration order or object identity.
    """
    return (
        capability.operation,
        "|".join(
            f"{operand.role.value}:"
            f"{'' if operand.dtype is None else operand.dtype.name}:"
            f"{operand.convert_to.name}"
            for operand in capability.operands
        ),
        capability.compute.value,
        "" if capability.accumulation is None else capability.accumulation.value,
        capability.output.name,
    )


_LOCK: Final = threading.Lock()

# Capabilities exactly as each class declared them, and the whole answer for
# that class: nothing is resolved through a base. A class present here is
# declared *and* sealed, because a declaration is complete and is published in
# one lock transaction: the stored mapping is written once and never mutated, so
# a query observes either no declaration or the whole of it, and what it
# observes cannot widen afterwards. Absence means no answer has been fixed yet.
_DECLARED: MutableMapping[type, Mapping[_CapabilityKey, OperationCapability]] = (
    weakref.WeakKeyDictionary()
)

# The shipped classes, declared by the framework-internal bootstrap during
# carrier-package initialization. Sealing is not what sets them apart — every
# declared class is sealed — only the diagnostic a caller trying to declare for
# one gets.
_SHIPPED: MutableSet[type] = weakref.WeakSet()

_EMPTY: Final[Mapping[_CapabilityKey, OperationCapability]] = {}


def _require_carrier_class(carrier_class: object) -> type:
    if not isinstance(carrier_class, type):
        raise TypeError("carrier_class must be a carrier class, not an instance")
    return carrier_class


def _require_declarable_carrier_class(carrier_class: object) -> type:
    """Validate a class any declaration path may name.

    Declaring is a claim that this exact class executes a shape, so only a
    concrete independent ``Carrier`` implementation can be named: ``object`` and
    the ``Carrier`` root are shared by every backend, a class outside the
    hierarchy implements no carrier at all, and a ``DependentCarrier`` class
    cannot answer for its instances, which reach different carriers.
    """
    _require_carrier_class(carrier_class)
    name = carrier_class.__name__  # type: ignore[union-attr]
    if carrier_class is Carrier:
        raise TypeError(
            "capabilities cannot be declared for the Carrier base class; declare "
            "them for the exact carrier class whose implementation executes them"
        )
    if not issubclass(cast(type, carrier_class), Carrier):
        raise TypeError(
            f"{name} is not a Carrier implementation; capabilities are declared "
            "for the exact carrier class whose implementation executes them"
        )
    if issubclass(cast(type, carrier_class), DependentCarrier):
        raise TypeError(
            f"{name} is a DependentCarrier implementation, whose capabilities "
            "depend on the carriers each instance composes; it generates them "
            "per instance instead of declaring them for its class"
        )
    return cast(type, carrier_class)


def _is_independent_carrier_class(carrier_class: type) -> bool:
    """Report whether this exact class is one that owns a class declaration.

    Independence is what makes a class-level answer meaningful: the ``Carrier``
    root and a class outside the hierarchy implement no backend, and a
    ``DependentCarrier`` class answers per instance instead.
    """
    return (
        carrier_class is not Carrier
        and issubclass(carrier_class, Carrier)
        and not issubclass(carrier_class, DependentCarrier)
    )


def _require_extensible_carrier_class(carrier_class: object) -> type:
    """Validate a class a public caller may declare capabilities for.

    Beyond being a declarable carrier class, its answer must still be open:
    a class declares once, and a shipped backend or an already-observed class
    has had its answer fixed.
    """
    _require_declarable_carrier_class(carrier_class)
    if _DECLARED.get(cast(type, carrier_class)) is not None:
        # Reported here so an ineligible class is refused before its entries are
        # examined; the check is repeated under the lock, which is authoritative.
        raise _declaration_closed_error(cast(type, carrier_class))
    return cast(type, carrier_class)


def _require_plan(plan: object) -> OperationPlan:
    if not isinstance(plan, OperationPlan):
        raise TypeError("plan must be an OperationPlan")
    return plan


def _observed_capabilities(
    carrier_class: type,
) -> Mapping[_CapabilityKey, OperationCapability]:
    """Return the sealed answer for ``carrier_class``, keyed by shape.

    There is no traversal: a class that declared nothing supports nothing, even
    when a base declared something, because a base's declarations describe a
    base's implementation.

    Observing an independent class that has not declared seals its empty set
    first, so first observation is the final answer. That is what lets a
    dependent carrier snapshot another class's reach and know the snapshot
    cannot go stale: a class cannot answer "nothing" now and something later.
    The seal is taken under the lock, but only once — a class already sealed is
    read without one, which keeps the execution gate lock-free.
    """
    declared = _DECLARED.get(carrier_class)
    if declared is not None:
        return declared
    if not _is_independent_carrier_class(carrier_class):
        return _EMPTY
    with _LOCK:
        declared = _DECLARED.get(carrier_class)
        if declared is None:
            declared = _EMPTY
            _DECLARED[carrier_class] = declared
        return declared


def _validated_entries(
    capabilities: Iterable[OperationCapability],
) -> tuple[OperationCapability, ...]:
    """Materialize ``capabilities``, rejecting anything that is not an entry."""
    entries = tuple(capabilities)
    for entry in entries:
        if not isinstance(entry, OperationCapability):
            raise TypeError("capabilities must be OperationCapability entries")
    return entries


def _sealed_error(carrier_class: type) -> TypeError:
    return TypeError(
        f"{carrier_class.__name__} is a shipped backend whose declared "
        "capabilities are sealed; a carrier composing it declares for its own "
        "class the shapes it executes"
    )


def _redeclaration_error(carrier_class: type) -> TypeError:
    return TypeError(
        f"{carrier_class.__name__} has already declared the operation "
        "capabilities its class executes; a class declares its complete set "
        "once, and a class first observed without a declaration is sealed "
        "against a later one"
    )


def _declaration_closed_error(carrier_class: type) -> TypeError:
    """Return the refusal for a class whose answer is already fixed."""
    if carrier_class in _SHIPPED:
        return _sealed_error(carrier_class)
    return _redeclaration_error(carrier_class)


def _complete_declaration(
    carrier_class: type, entries: tuple[OperationCapability, ...]
) -> Mapping[_CapabilityKey, OperationCapability]:
    """Return the whole set ``entries`` declares, or refuse a repeated shape.

    Built before the lock is taken, so the locked region is one check and one
    write and a rejected declaration never touches the registry.
    """
    declared: dict[_CapabilityKey, OperationCapability] = {}
    for entry in entries:
        key = _capability_key(entry)
        if key in declared:
            raise ValueError(
                f"{carrier_class.__name__} already declares a capability for "
                f"{_describe_capability(entry)}"
            )
        declared[key] = entry
    return declared


def _is_sealed(carrier_class: type) -> bool:
    """Report whether ``carrier_class``'s capability answer is already fixed.

    Only the built-in bootstrap consults this, so re-running initialization is a
    no-op rather than an error.
    """
    return _DECLARED.get(_require_carrier_class(carrier_class)) is not None


def _declare_built_in_capabilities(
    carrier_class: type, capabilities: Iterable[OperationCapability]
) -> None:
    """Record a shipped backend's capabilities once and seal them.

    This is the framework-internal declaration path, reached only from
    :mod:`strideweave.carriers._built_in_capabilities` during carrier-package
    initialization. It is deliberately private and exported nowhere: sealing is
    what makes a shipped backend's advertised reach a property of its
    implementation rather than of whatever a later caller registered, and a
    public entry point here would be a way around that.

    An empty ``capabilities`` is a declaration too — the sealed statement that a
    storage or composition carrier plans no operation of its own.

    Args:
        carrier_class: The exact shipped carrier class the capabilities belong
            to.
        capabilities: Every capability that class executes. There is no second
            call: the class is sealed against further declarations.

    Raises:
        TypeError: If ``carrier_class`` is not an eligible ``Carrier``
            implementation, or an entry is not an :class:`OperationCapability`.
        RuntimeError: If the class has already declared its built-in
            capabilities.
        ValueError: If one shape is declared twice.
    """
    _require_declarable_carrier_class(carrier_class)
    entries = _validated_entries(capabilities)
    declaration = _complete_declaration(carrier_class, entries)
    # Publishing and marking the class shipped are one transaction, so a
    # concurrent public registration never sees a shipped backend as an
    # ordinary undeclared class.
    with _LOCK:
        if _DECLARED.get(carrier_class) is not None:
            raise RuntimeError(
                f"{carrier_class.__name__} has already declared its built-in "
                "capabilities"
            )
        _SHIPPED.add(carrier_class)
        _DECLARED[carrier_class] = declaration


def register_operation_capabilities(
    carrier_class: type, capabilities: Iterable[OperationCapability]
) -> None:
    """Declare the complete set of plan shapes ``carrier_class`` executes.

    The entries are stored against this exact class and are visible to no other
    class, so extending the framework cannot widen what any other backend claims
    to support. A carrier built by composition declares here the shapes it
    executes, including those it executes by lowering onto a carrier it owns.

    This is one complete declaration rather than the first of several, and it
    must come before anything observes the class. A declared class is sealed by
    the same call that declares it, and a class first observed without a
    declaration is sealed against the empty set, so whatever a caller — or a
    dependent carrier snapshotting this class — is told first stays true. A
    carrier that declares its shapes in its own module, at import time, is
    always in time.

    Args:
        carrier_class: The exact carrier class the capabilities belong to. It
            must be an independent ``Carrier`` implementation of its own whose
            answer is still open: the ``Carrier`` root, a class outside the
            carrier hierarchy, a ``DependentCarrier`` class, a shipped backend,
            and a class already declared or already observed are all rejected.
        capabilities: Every capability entry this class executes. Declaring the
            same shape twice in one call is rejected, and an empty declaration
            is the complete statement that the class executes no planned
            operation.

    Raises:
        TypeError: If ``carrier_class`` is not a class, is not an eligible
            ``Carrier`` implementation, has already declared or been observed,
            or an entry is not an :class:`OperationCapability`.
        ValueError: If one declaration names the same shape twice.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import (
        ...     OperationCapability,
        ...     register_operation_capabilities,
        ...     supports_operation_plan,
        ... )
        >>> class TinyCarrier(sw.Carrier): ...
        >>> plan = resolve_operation_plan("relu", DType.Int32)
        >>> register_operation_capabilities(
        ...     TinyCarrier, [OperationCapability.from_plan(plan)]
        ... )
        >>> supports_operation_plan(TinyCarrier, plan)
        True
    """
    _require_extensible_carrier_class(carrier_class)
    entries = _validated_entries(capabilities)
    declaration = _complete_declaration(carrier_class, entries)
    with _LOCK:
        # The authoritative check: every path that fixes a class's answer —
        # another declaration, the built-in bootstrap, or a first observation —
        # publishes under this same lock, so no registration can slip between
        # one of them and its seal.
        if _DECLARED.get(carrier_class) is not None:
            raise _declaration_closed_error(carrier_class)
        _DECLARED[carrier_class] = declaration


# --- Carriers whose capabilities depend on the carriers they compose ---------

# One frozen snapshot per finalized dependent carrier, keyed by instance
# identity. Identity rather than the instance itself, because a dependent
# carrier is an open extension point: a subclass defining ``__eq__``,
# ``__hash__``, or ``__slots__`` must not be able to merge two instances'
# snapshots or lose its own. Each entry is dropped when its carrier is
# collected, and the stored weak reference is what proves a surviving entry
# still belongs to the live carrier at that address rather than to a reused one.
#
# Reusing an address is what makes the identity key delicate, so the two
# defenses are stated explicitly. First, this relies on CPython running a
# weakref callback synchronously while the referent is being deallocated, before
# its memory can back another object: an entry is therefore always removed
# before its identity can be handed out again. Second, that ordering is not
# trusted on its own — every read re-checks that the stored reference still
# resolves to the carrier being asked about, so an entry that somehow outlived
# its carrier is ignored rather than answered with.
#
# The lock is reentrant because a callback can fire during a collection
# triggered inside a locked region on this same thread.
_INSTANCE_LOCK: Final = threading.RLock()


@dataclass(frozen=True, slots=True)
class _InstanceSnapshot:
    """One dependent instance's frozen capabilities and the carrier they answer for."""

    reference: weakref.ref
    entries: Mapping[_CapabilityKey, OperationCapability]


_FROZEN: Final[MutableMapping[int, _InstanceSnapshot]] = {}


def _forget_instance_snapshot(identity: int, reference: weakref.ref) -> None:
    """Drop a collected carrier's snapshot, freeing its identity for reuse."""
    with _INSTANCE_LOCK:
        snapshot = _FROZEN.get(identity)
        if snapshot is not None and snapshot.reference is reference:
            del _FROZEN[identity]


def _require_dependent_carrier(carrier: object) -> DependentCarrier:
    if not isinstance(carrier, DependentCarrier):
        raise TypeError("carrier must be a DependentCarrier instance")
    return carrier


def _frozen_instance_snapshot(carrier: DependentCarrier) -> _InstanceSnapshot | None:
    snapshot = _FROZEN.get(id(carrier))
    if snapshot is None or snapshot.reference() is not carrier:
        return None
    return snapshot


def _already_finalized_error(name: str) -> RuntimeError:
    return RuntimeError(
        f"this {name} instance has already finalized its operation capabilities; "
        "a dependent carrier finalizes once, at the end of construction"
    )


def _freeze_instance_capabilities(carrier: DependentCarrier) -> None:
    """Generate, validate, and publish ``carrier``'s capabilities once.

    Generation runs outside the lock — it is the concrete carrier's own code and
    may reach the carriers it composes — and its result is materialized once, so
    a generator handing back a one-shot iterator is not consumed twice. Nothing
    is published until every entry has been validated, so a generator that
    raises, yields a non-entry, or repeats a shape leaves the carrier
    unfinalized rather than half-declared.

    Finalization is a single-threaded construction step. The concrete
    implementation must not expose the unfinished instance or invoke this
    function concurrently; the publication lock protects the snapshot registry,
    but does not serialize calls to extension-controlled generation code.
    """
    _require_dependent_carrier(carrier)
    name = type(carrier).__name__
    if _frozen_instance_snapshot(carrier) is not None:
        raise _already_finalized_error(name)
    entries = _validated_entries(carrier._generate_operation_capabilities())
    frozen: dict[_CapabilityKey, OperationCapability] = {}
    for entry in sorted(entries, key=_sort_key):
        key = _capability_key(entry)
        if key in frozen:
            raise ValueError(
                f"this {name} instance generated two capabilities for "
                f"{_describe_capability(entry)}"
            )
        frozen[key] = entry
    identity = id(carrier)
    reference = weakref.ref(carrier, partial(_forget_instance_snapshot, identity))
    with _INSTANCE_LOCK:
        # Keep publication defensive even though concurrent finalization is
        # outside the supported construction contract.
        if _frozen_instance_snapshot(carrier) is not None:
            raise _already_finalized_error(name)
        _FROZEN[identity] = _InstanceSnapshot(reference=reference, entries=frozen)


def _instance_capabilities(
    carrier: DependentCarrier,
) -> Mapping[_CapabilityKey, OperationCapability]:
    """Return the capabilities ``carrier`` froze, keyed by shape.

    The snapshot is the whole answer for that instance: a dependent carrier
    resolves nothing through its class, which cannot know which carriers this
    instance was handed.

    Raises:
        RuntimeError: If ``carrier`` has not finalized its capabilities yet, so
            an unfinished hierarchy is never mistaken for one supporting
            nothing.
    """
    _require_dependent_carrier(carrier)
    snapshot = _frozen_instance_snapshot(carrier)
    if snapshot is None:
        raise RuntimeError(
            f"this {type(carrier).__name__} instance has not finalized its "
            "operation capabilities; a dependent carrier calls "
            "_finalize_dependent_capabilities() once its dependencies are in "
            "place, before it answers a capability query"
        )
    return snapshot.entries


class DependentCarrier(Carrier):
    """Carrier whose executable plans depend on the carriers it composes.

    ``Carrier`` itself is the *independent* model: its exact class declares the
    plan shapes its own kernels execute, once, for every instance. That is the
    right model for a backend whose reach is fixed by the code it ships, and it
    stays the default. A carrier assembled from other carriers is different —
    one hierarchy over ``CPU`` tiers and one over object-storage tiers execute
    different plans while sharing a class — so this extension category moves
    capability ownership from the class to the constructed instance.

    Subclassing is the supported way to use it, and a concrete dependent carrier
    may still be closed to further subclassing itself. Two protected members
    make up the contract:

    - ``_generate_operation_capabilities`` is unimplemented here. How an
      instance decides what it executes is entirely its own: it may enumerate a
      composed carrier's capabilities, filter them by what it can store, or
      build entries outright.
    - ``_finalize_dependent_capabilities`` freezes what that hook generated.
      The base never calls it, because only the concrete carrier knows when its
      dependencies are complete; construction calls it explicitly as its last
      step, in the thread constructing the instance, before the instance becomes
      reachable elsewhere. Concurrent finalization and exposing an unfinished
      instance are outside this cooperative extension contract. Until
      finalization has completed the instance answers no capability query, and a
      later call is an error, so a snapshot is never observed half-built or
      quietly replaced.

    Capabilities are frozen against the dependencies the instance ended up
    with; they are structural, not a report of how the carrier currently feels.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.operation_capability import (
        ...     OperationCapability,
        ... )
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> class Mirror(sw.DependentCarrier):
        ...     def __init__(self, inner):
        ...         super().__init__()
        ...         self._inner = inner
        ...         self._finalize_dependent_capabilities()
        ...
        ...     def _generate_operation_capabilities(self):
        ...         return self._inner.operation_capabilities()
        >>> plan = resolve_operation_plan("relu", sw.DType.Int32)
        >>> OperationCapability.from_plan(plan) in Mirror(
        ...     sw.CPU(1, dtype=sw.DType.Int32)
        ... )._operation_capability_snapshot()
        True
    """

    def _generate_operation_capabilities(self) -> Iterable[OperationCapability]:
        """Return every plan shape this instance executes faithfully.

        Implemented by the concrete dependent carrier and called once, from
        :meth:`_finalize_dependent_capabilities`. The algorithm is the
        implementation's own; the only requirements are that it yields
        :class:`OperationCapability` entries and that it does not repeat one
        exact shape.

        Returns:
            The capabilities to freeze for this instance.

        Raises:
            NotImplementedError: Always, unless the concrete carrier overrides
                it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_generate_operation_capabilities() to say which plans this "
            "instance executes"
        )

    def _finalize_dependent_capabilities(self) -> None:
        """Freeze this instance's generated capabilities, once.

        Call this as the last step of construction, after every dependency is
        installed and validated, from the same thread performing construction
        and before the unfinished instance becomes reachable elsewhere. This
        lifecycle hook is not synchronized for concurrent calls. It is the point
        at which the instance starts answering capability queries. Failure
        publishes nothing, so a carrier whose finalization raised is left
        unfinalized rather than holding a partial set.

        Raises:
            NotImplementedError: If the concrete carrier implements no
                generator.
            TypeError: If the generator yields something other than an
                :class:`OperationCapability`.
            ValueError: If it generates one exact shape twice.
            RuntimeError: If this instance has already finalized.

        Examples:
            >>> import strideweave as sw
            >>> class Empty(sw.DependentCarrier):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._finalize_dependent_capabilities()
            ...
            ...     def _generate_operation_capabilities(self):
            ...         return ()
            >>> Empty()._operation_capability_snapshot()
            ()
        """
        _freeze_instance_capabilities(self)

    def _operation_capability_snapshot(
        self, operation: str | None = None
    ) -> tuple[OperationCapability, ...]:
        """Return this instance's frozen capabilities, in a stable order.

        Args:
            operation: Restrict the result to one operation name, or ``None``
                for every operation.

        Returns:
            The frozen capabilities ordered by operation name, then by operand
            shape, compute, accumulation, and output.

        Raises:
            RuntimeError: If this instance has not finalized yet.

        Examples:
            >>> import strideweave as sw
            >>> class Empty(sw.DependentCarrier):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._finalize_dependent_capabilities()
            ...
            ...     def _generate_operation_capabilities(self):
            ...         return ()
            >>> Empty()._operation_capability_snapshot("relu")
            ()
        """
        entries = _instance_capabilities(self).values()
        if operation is not None:
            entries = [entry for entry in entries if entry.operation == operation]
        return tuple(sorted(entries, key=_sort_key))


# --- Answering from a set of entries ----------------------------------------
#
# One implementation of each question, so a class-declaration query and a
# carrier query cannot drift: they differ only in which entries they resolve.


def _enumerate(
    entries: Mapping[_CapabilityKey, OperationCapability], operation: str | None
) -> tuple[OperationCapability, ...]:
    selected = entries.values()
    if operation is not None:
        selected = [entry for entry in selected if entry.operation == operation]
    return tuple(sorted(selected, key=_sort_key))


def _match(
    entries: Mapping[_CapabilityKey, OperationCapability], plan: OperationPlan
) -> OperationCapability | None:
    return entries.get(_plan_key(_require_plan(plan)))


def _reason(
    name: str,
    entries: Mapping[_CapabilityKey, OperationCapability],
    plan: OperationPlan,
) -> str | None:
    if _match(entries, plan) is not None:
        return None
    if not _enumerate(entries, plan.operation):
        return f"{name} declares no operation-plan capability for {plan.operation!r}"
    return (
        f"{name} declares no operation-plan capability for this {plan.operation!r} "
        f"plan: {_describe_plan(plan)}"
    )


def _require(
    name: str,
    entries: Mapping[_CapabilityKey, OperationCapability],
    plan: OperationPlan,
) -> OperationCapability:
    capability = _match(entries, plan)
    if capability is None:
        raise UnsupportedOperationPlan(_reason(name, entries, plan))
    return capability


# --- Asking a carrier class -------------------------------------------------
#
# What an exact class declared, which is what an independent backend's
# execution gate and a declaration-oriented caller ask about.


def capabilities_for_carrier_class(
    carrier_class: type, operation: str | None = None
) -> tuple[OperationCapability, ...]:
    """Return the capabilities ``carrier_class`` executes, in a stable order.

    The entries are the same immutable objects execution matches against, so an
    enumerated shape is executable and an executable shape is enumerated.

    Enumerating an independent class that has not declared seals its empty set,
    so this answer is final rather than provisional.

    Args:
        carrier_class: The exact carrier class to enumerate.
        operation: Restrict the result to one operation name, or ``None`` for
            every operation.

    Returns:
        The matching capabilities ordered by operation name, then by operand
        shape, compute, accumulation, and output.

    Raises:
        TypeError: If ``carrier_class`` is not a class.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import (
        ...     OperationCapability,
        ...     capabilities_for_carrier_class,
        ...     register_operation_capabilities,
        ... )
        >>> class SmallCarrier(sw.Carrier): ...
        >>> register_operation_capabilities(
        ...     SmallCarrier,
        ...     [
        ...         OperationCapability.from_plan(
        ...             resolve_operation_plan("relu", DType.Float32)
        ...         )
        ...     ],
        ... )
        >>> [entry.operation for entry in capabilities_for_carrier_class(SmallCarrier)]
        ['relu']
    """
    _require_carrier_class(carrier_class)
    return _enumerate(_observed_capabilities(carrier_class), operation)


def matching_capability(
    carrier_class: type, plan: OperationPlan
) -> OperationCapability | None:
    """Return the capability ``carrier_class`` executes ``plan`` under, if any.

    Asking seals an independent class that has not declared, exactly as
    enumerating it does: the answer a caller acts on is the final one.

    Args:
        carrier_class: The exact carrier class to query.
        plan: A resolved operation plan.

    Returns:
        The single matching capability, or ``None`` when the backend declares
        no capability for that exact shape.

    Raises:
        TypeError: If ``carrier_class`` is not a class or ``plan`` is not an
            :class:`~strideweave.carriers.operation_policy.OperationPlan`.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import matching_capability
        >>> class BareCarrier: ...
        >>> matching_capability(
        ...     BareCarrier, resolve_operation_plan("relu", DType.Float32)
        ... ) is None
        True
    """
    _require_carrier_class(carrier_class)
    return _match(_observed_capabilities(carrier_class), plan)


def supports_operation_plan(carrier_class: type, plan: OperationPlan) -> bool:
    """Report whether ``carrier_class`` executes ``plan`` as written.

    Support means faithful execution of an already-resolved plan. It is not a
    claim about promotion, which no backend owns.

    Args:
        carrier_class: The exact carrier class to query.
        plan: A resolved operation plan.

    Returns:
        ``True`` when a declared capability matches the plan exactly.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import (
        ...     supports_operation_plan,
        ... )
        >>> class BareCarrier: ...
        >>> supports_operation_plan(
        ...     BareCarrier, resolve_operation_plan("add", DType.Int32, DType.Int32)
        ... )
        False
    """
    return matching_capability(carrier_class, plan) is not None


def unsupported_plan_reason(carrier_class: type, plan: OperationPlan) -> str | None:
    """Return why ``carrier_class`` cannot execute ``plan``, or ``None``.

    The reason distinguishes an operation the backend implements nothing for
    from an operation it implements other shapes of, which is the distinction a
    caller needs to tell "wrong carrier" from "wrong dtype combination".

    Args:
        carrier_class: The exact carrier class to query.
        plan: A resolved operation plan.

    Returns:
        A stable explanatory sentence, or ``None`` when the plan is supported.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import (
        ...     unsupported_plan_reason,
        ... )
        >>> class BareCarrier: ...
        >>> reason = unsupported_plan_reason(
        ...     BareCarrier, resolve_operation_plan("relu", DType.Float32)
        ... )
        >>> "no operation-plan capability" in reason
        True
    """
    _require_carrier_class(carrier_class)
    return _reason(carrier_class.__name__, _observed_capabilities(carrier_class), plan)


def require_capability(carrier_class: type, plan: OperationPlan) -> OperationCapability:
    """Return the capability ``plan`` executes under, or refuse the plan.

    This is the acceptance gate a backend calls before allocating a result or
    entering a kernel. Refusing is the only alternative to executing the plan as
    written: a backend never substitutes a shape it does support.

    Args:
        carrier_class: The exact carrier class about to execute the plan.
        plan: A resolved operation plan.

    Returns:
        The matching capability.

    Raises:
        UnsupportedOperationPlan: If the backend declares no capability for that
            exact shape.
        TypeError: If ``carrier_class`` is not a class or ``plan`` is not an
            :class:`~strideweave.carriers.operation_policy.OperationPlan`.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.carriers.operation_capability import (
        ...     UnsupportedOperationPlan,
        ...     require_capability,
        ... )
        >>> class BareCarrier: ...
        >>> try:
        ...     require_capability(
        ...         BareCarrier, resolve_operation_plan("relu", DType.Float32)
        ...     )
        ... except UnsupportedOperationPlan as error:
        ...     print(error)
        BareCarrier declares no operation-plan capability for 'relu'
    """
    _require_carrier_class(carrier_class)
    return _require(carrier_class.__name__, _observed_capabilities(carrier_class), plan)


# --- Asking a carrier -------------------------------------------------------
#
# The public surface a user reaches through a carrier object. It answers from
# whichever set owns that carrier — its exact class's sealed declarations, or a
# dependent instance's frozen snapshot — so a caller never has to know which
# ownership model a backend uses, and enforcement and introspection cannot
# disagree because they read the same entries through the same resolution.


def _require_carrier(carrier: object) -> object:
    if isinstance(carrier, type):
        raise TypeError("carrier must be a carrier instance, not a class")
    if not isinstance(carrier, Carrier):
        raise TypeError("carrier must be a Carrier instance")
    return carrier


def _carrier_capabilities(
    carrier: object,
) -> Mapping[_CapabilityKey, OperationCapability]:
    """Return the entries that decide what ``carrier`` executes, keyed by shape.

    Ownership is selected structurally rather than by name: a dependent carrier
    answers from the snapshot it froze, and every other carrier answers from its
    exact class, with no traversal of bases in either case.
    """
    _require_carrier(carrier)
    if isinstance(carrier, DependentCarrier):
        return _instance_capabilities(carrier)
    return _observed_capabilities(type(carrier))


def carrier_operation_capabilities(
    carrier: object, operation: str | None = None
) -> tuple[OperationCapability, ...]:
    """Return the capabilities ``carrier`` executes, in a stable order.

    Args:
        carrier: The carrier instance to enumerate.
        operation: Restrict the result to one operation name, or ``None`` for
            every operation.

    Returns:
        The matching capabilities ordered by operation name, then by operand
        shape, compute, accumulation, and output.

    Raises:
        TypeError: If ``carrier`` is a class rather than a carrier instance.
        RuntimeError: If ``carrier`` is a dependent carrier that has not
            finalized its capabilities.

    Examples:
        >>> import strideweave as sw
        >>> [
        ...     entry.output.name
        ...     for entry in sw.CPU(1).operation_capabilities("relu")
        ... ]
        ['Float32', 'Int32']
    """
    return _enumerate(_carrier_capabilities(carrier), operation)


def carrier_supports_operation_plan(carrier: object, plan: OperationPlan) -> bool:
    """Report whether ``carrier`` executes ``plan`` as written.

    Args:
        carrier: The carrier instance to query.
        plan: A resolved operation plan.

    Returns:
        ``True`` when one of the carrier's capabilities matches the plan
        exactly.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> sw.CPU(1).supports_operation_plan(
        ...     resolve_operation_plan("relu", sw.DType.Int32)
        ... )
        True
    """
    return _match(_carrier_capabilities(carrier), plan) is not None


def carrier_unsupported_plan_reason(carrier: object, plan: OperationPlan) -> str | None:
    """Return why ``carrier`` cannot execute ``plan``, or ``None``.

    Args:
        carrier: The carrier instance to query.
        plan: A resolved operation plan.

    Returns:
        A stable explanatory sentence, or ``None`` when the plan is supported.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> sw.CPU(1).unsupported_plan_reason(
        ...     resolve_operation_plan("relu", sw.DType.Int32)
        ... ) is None
        True
    """
    return _reason(type(carrier).__name__, _carrier_capabilities(carrier), plan)


def require_carrier_capability(
    carrier: object, plan: OperationPlan
) -> OperationCapability:
    """Return the capability ``carrier`` executes ``plan`` under, or refuse it.

    This is the acceptance gate for a carrier whose reach is per instance, and
    it reaches the same decision the class-oriented gate does for an
    independent one.

    Args:
        carrier: The carrier instance about to execute the plan.
        plan: A resolved operation plan.

    Returns:
        The matching capability.

    Raises:
        UnsupportedOperationPlan: If the carrier has no capability for that
            exact shape.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> sw.CPU(1).require_operation_plan(
        ...     resolve_operation_plan("relu", sw.DType.Int32)
        ... ).compute.value
        'int32_exact'
    """
    return _require(type(carrier).__name__, _carrier_capabilities(carrier), plan)


def _describe_operand(
    role: OperandRole, dtype: SimpleDType | None, convert_to: SimpleDType
) -> str:
    if role is OperandRole.TENSOR and dtype is not None:
        return f"tensor {dtype.name}->{convert_to.name}"
    return f"weak scalar ->{convert_to.name}"


def _describe(
    operation: str,
    operands: str,
    compute: Arithmetic,
    accumulation: Accumulation | None,
    output: SimpleDType,
) -> str:
    combination = (
        "no accumulation"
        if accumulation is None
        else f"{accumulation.value} accumulation"
    )
    return (
        f"{operation} with operands ({operands}), {compute.value} compute, "
        f"{combination}, {output.name} output"
    )


def _describe_capability(capability: OperationCapability) -> str:
    operands = ", ".join(
        _describe_operand(operand.role, operand.dtype, operand.convert_to)
        for operand in capability.operands
    )
    return _describe(
        capability.operation,
        operands,
        capability.compute,
        capability.accumulation,
        capability.output,
    )


def _describe_plan(plan: OperationPlan) -> str:
    operands = ", ".join(
        _describe_operand(operand.role, operand.dtype, operand.convert_to)
        for operand in plan.operands
    )
    return _describe(
        plan.operation, operands, plan.compute, plan.accumulation, plan.output
    )
