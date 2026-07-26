"""Module-owned dtype contract specifications and ownership guards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

# The predicates that answer which kind of descriptor an object is. Every
# contract class fixes all of them for its own descriptors, so each owns the
# whole set rather than only the one it overrides.
_KIND_PREDICATES: Final = (
    "is_category",
    "is_compound",
    "is_opaque_storage",
    "is_simple",
)

# What each contract class owns: its stored fields and the accessors reporting
# its part of a descriptor's representation. The policy is layered — a class is
# bound by every layer above it in its MRO and its own layer binds its
# subclasses — and it is defined here, in module state, rather than as class
# attributes the classes themselves carry. Ownership expressed on the class
# would be ordinary class state: replacing or deleting it is an ordinary
# assignment, and a policy that can be switched off that way protects nothing.
# Reading these frozensets instead means weakening any class attribute, legacy
# ``_OWNED_MEMBERS`` metadata included, cannot unlock a single owned member.
_ROOT_OWNED: Final = frozenset(
    {
        # Attribute lookup and mutation themselves. An implementation that
        # intercepted them would decide what every accessor above answers —
        # ``__getattribute__`` returning 64 for the ``bits`` of an 8-bit
        # descriptor, say — which is the same drift between observable
        # representation and recorded structure that owning the accessors
        # prevents, reached one level lower. Sealing likewise relies on the
        # model's own ``__setattr__`` and ``__delattr__`` staying in place.
        "__delattr__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "_finalized",
        "_name",
        "_structure",
        "_supertype",
        # The names the model used to dispatch identity policy through. They
        # are reserved rather than defined: a descriptor's validation, its
        # canonical structure layers, its structural-uniqueness disposition,
        # and the conflict it reports belong to the contract specifications
        # below, so an implementation that defines one of these — expecting
        # the model to call it — is told at class creation that it does not,
        # instead of quietly weakening nothing.
        "_contract_structure",
        "_structural_key",
        "_structure_conflict",
        "_validate_finalized",
        "is_subtype_of",
        "name",
        "structure",
        "supertype",
        "supertypes",
        "value",
    }
)
_CATEGORY_OWNED: Final = frozenset({"_opaque_storage", *_KIND_PREDICATES})
_SIMPLE_OWNED: Final = frozenset({"_bits", "bits", *_KIND_PREDICATES})
_COMPOUND_OWNED: Final = frozenset(
    {"_simple_types", "num_carriers", "simple_types", *_KIND_PREDICATES}
)
_BLOCK_SCALED_OWNED: Final = frozenset(
    {
        "_element",
        "_levels",
        "bits_per_element",
        "element",
        "levels",
        "num_axes",
    }
)


@dataclass(frozen=True, slots=True)
class _ContractSpec:
    """One contract class's whole framework-owned share of dtype identity.

    A specification is data this module holds *about* a contract class, never
    behavior the class carries. That is the point: policy expressed as a method
    is dispatched virtually, so an implementation could override it and decide
    what its own identity is — omitting a canonical layer from its fingerprint,
    declining the uniqueness its contract imposes, or weakening the validation
    it still claims to satisfy. Reading these frozen records instead means a
    descriptor is validated, fingerprinted, and made unique by the contracts it
    inherits, whatever its class defines. The one contribution an implementation
    makes to its representation is :meth:`DType.structure_extension`.

    Args:
        owned: Members this contract class owns, which no subclass, base, or
            later class assignment may define or replace.
        layer: Returns exactly the fragment this contract class contributes to a
            descriptor's canonical structure, without its bases' fragments.
        validate: Checks a completely initialized descriptor against this
            contract, or ``None`` when the contract adds no check of its own.
        structural_conflict: Present only for a contract whose descriptors are
            additionally unique by structure; it reports the descriptor that
            already claimed a representation. ``None`` leaves descriptors of the
            contract unique by name alone.
    """

    # The callables receive a descriptor belonging to the contract class's own
    # subtree, which is narrower than ``DType`` but not something the checker
    # can express through a mapping keyed by class.
    owned: frozenset[str]
    layer: Callable[[Any], tuple[object, ...]]
    validate: Callable[[Any], None] | None = None
    structural_conflict: Callable[[Any], str] | None = None


# The contract specifications, keyed by the class each one governs. Entries are
# added once, immediately after the class they describe is created and before
# any subclass of it can exist, so the mapping is complete for every class the
# checks below ever see. It lives in module state rather than as class
# attributes because ownership carried on a class would be ordinary class
# state: replacing or deleting it is an ordinary assignment, and a policy that
# can be switched off that way protects nothing. Mutating this mapping directly
# is unsupported rather than guarded, exactly like reaching into any other
# private module state.
_CONTRACT_SPECS: Final[dict[type, _ContractSpec]] = {}


def _declare_contract(cls: type, spec: _ContractSpec) -> None:
    """Record the contract the dtype model owns on behalf of ``cls``."""
    _CONTRACT_SPECS[cls] = spec


def _contract_specs(cls: type) -> tuple[_ContractSpec, ...]:
    """Return the contracts binding ``cls``, from most general to most specific.

    The method resolution order is what composes them, so a class is bound by
    every contract it inherits and by nothing else: a ``CompoundDType``
    implementation resolves the root and compound contracts, and a
    ``BlockScaledDType`` subclass resolves those two plus the block-scaled one.
    Reversing the MRO puts the root first, which is the order the layered
    ``super()``-chained hooks this replaced produced, so the composed structure
    is the same sequence a base-first chain built.
    """
    return tuple(
        spec
        for base in reversed(cls.__mro__)
        if (spec := _CONTRACT_SPECS.get(base)) is not None
    )


def _owned_state(cls: type, *, inherited_only: bool) -> set[str]:
    """Return the names of stored fields and accessors the dtype model owns.

    ``inherited_only`` selects what a subclass may not introduce, excluding the
    class's own definitions; otherwise the class's own contributions count too,
    which is what a later assignment must not replace.

    An implementation class outside this module contributes the slots it
    declares, so a further subclass cannot shadow an extension's own storage
    either. That reads a class attribute, but it can only ever add to the
    module-owned layers, never subtract from them: a model-owned member stays
    owned however an implementation declares — or redeclares — its ``__slots__``.
    """
    owned: set[str] = set()
    for base in cls.__mro__[1:] if inherited_only else cls.__mro__:
        contract = _CONTRACT_SPECS.get(base)
        if contract is not None:
            owned.update(contract.owned)
            continue
        slots = base.__dict__.get("__slots__", ())
        owned.update((slots,) if isinstance(slots, str) else slots)
    return owned


def _reject_shadowed_state(cls: type) -> None:
    """Reject a descriptor class that redefines state the dtype model owns.

    A descriptor's stored fields and the accessors that report its
    representation must agree with the structure recorded for its identity.
    Redefining either — a slot a base already declared, or an owned accessor —
    would reintroduce exactly that drift, so it is refused when the class is
    created rather than discovered through a descriptor that lies about itself.
    The same applies to the reserved names the model once dispatched identity
    policy through, which a class defining one would expect to be called. An
    implementation extends the model by adding fields and, when its
    representation carries state of its own, by overriding
    :meth:`DType.structure_extension`.
    """
    shadowed = sorted(_owned_state(cls, inherited_only=True).intersection(vars(cls)))
    if shadowed:
        raise TypeError(
            f"{cls.__name__} must not redefine {', '.join(shadowed)}: a "
            "descriptor's stored state, structural accessors, and identity "
            "policy are owned by the dtype model, because an override could "
            "report or claim a representation that disagrees with the "
            "registered identity"
        )


def _reject_shadowing_bases(cls: type) -> None:
    """Reject a descriptor class inheriting owned members from a mixin.

    A descriptor class may be assembled from several bases, and what the class
    body does not define its bases still supply. A mixin contributing an owned
    accessor — or a ``__getattribute__`` deciding what every accessor answers —
    makes exactly the descriptor the direct check refuses, only spelled one base
    away, and a mixin listed *after* the contract class is not harmless either:
    the model defines no ``__getattr__``, so that one takes effect wherever it
    sits. The whole initial hierarchy is therefore checked here.

    Only the hierarchy as defined is checked. A class and its bases are trusted
    to stay what they were when the class was created; later mutation of a mixin
    is outside the supported contract rather than something the model watches.
    """
    owned: set[str] = set()
    for spec in _contract_specs(cls):
        owned.update(spec.owned)
    for base in cls.__mro__:
        # ``object`` supplies the attribute machinery every class starts with,
        # and a contract class supplies the members it owns by definition.
        # ``cls`` itself is reported by the direct check instead.
        if base is cls or base is object or base in _CONTRACT_SPECS:
            continue
        shadowed = sorted(owned.intersection(vars(base)))
        if shadowed:
            raise TypeError(
                f"{cls.__name__} inherits {', '.join(shadowed)} from "
                f"{base.__name__}: a descriptor's stored state, structural "
                "accessors, identity policy, and attribute machinery are owned "
                "by the dtype model, because a base that supplies one could "
                "report a representation that disagrees with the registered "
                "identity"
            )


def _reject_owned_assignment(cls: type, name: str) -> None:
    """Reject replacing owned state on an existing descriptor class.

    The same rule the class body is checked against applies afterwards:
    patching an owned accessor onto a class would let its descriptors report a
    representation that disagrees with their registered identity. It is a
    module function rather than a metaclass method so that the check itself is
    not one more class attribute an assignment could replace.
    """
    if name in _owned_state(cls, inherited_only=False):
        raise AttributeError(
            f"{cls.__name__}.{name} is owned by the dtype model and cannot "
            "be replaced: a descriptor's stored state, structural accessors, "
            "and identity policy must keep agreeing with its registered "
            "identity"
        )


def _reject_shadowed_instance_state(dtype: object) -> None:
    """Reject a descriptor whose own attributes shadow model-owned members.

    A descriptor class need not declare ``__slots__``, so an implementation may
    keep its fields in an instance dictionary. Those fields are its own, but an
    entry named like a model-owned member is not: an instance attribute takes
    precedence over an inherited method, so a descriptor could publish a
    ``structure`` or an ``is_compound`` of its own while the structure recorded
    for its identity — and carried by its pickle — said something else. The
    completed descriptor is therefore checked before anything about it is
    computed, and again once every representation hook has run and before
    anything is claimed or sealed; finalization fails as a whole rather than
    sealing a descriptor that misreports itself.

    The rule addresses a contract violation, not hostile code: an
    implementation that mutates a descriptor after it is finalized, or that
    reaches around attribute access, has left the supported model behind.
    """
    state = getattr(dtype, "__dict__", None)
    if not isinstance(state, dict):
        return
    shadowed = sorted(
        _owned_state(type(dtype), inherited_only=False).intersection(state)
    )
    if shadowed:
        raise TypeError(
            f"{type(dtype).__name__} assigned {', '.join(shadowed)} on itself: a "
            "descriptor's stored state and structural accessors are owned by "
            "the dtype model, because an instance attribute could report a "
            "representation that disagrees with the registered identity"
        )
