"""Core dtype descriptors, registry transactions, and identity model."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self, cast

from .contracts import (
    _CATEGORY_OWNED,
    _COMPOUND_OWNED,
    _ROOT_OWNED,
    _SIMPLE_OWNED,
    _contract_specs,
    _ContractSpec,
    _declare_contract,
    _reject_owned_assignment,
    _reject_shadowed_instance_state,
    _reject_shadowed_state,
    _reject_shadowing_bases,
)
from .structure import (
    _described_structure_value,
    _encoded_leaf,
    _encoded_structure,
    _require_encoded_structure,
    _structure_difference,
)

if TYPE_CHECKING:
    from .block_scaled import BlockScaledDType


_REGISTRY: dict[str, DType] = {}

# Descriptors whose contract additionally requires uniqueness by structure —
# currently the block-scaled formats — claim their completed, model-composed
# structure here.
_STRUCTURES: dict[object, DType] = {}

# One lock guards every registry read and write. Claiming a name — and, for a
# block-scaled descriptor, a structure — must be indivisible, or two concurrent
# constructors could both pass their availability checks and return distinct
# descriptors while only one stayed reachable, breaking SW002 identity. It is an
# ``RLock`` so a registration performed while the registry is already held (such
# as nested built-in initialization) cannot deadlock against itself.
_REGISTRY_LOCK: Final = threading.RLock()


def _normalize_lookup_name(name: object) -> str:
    """Return an exact string that is safe to use as a registry key.

    A genuine ``str`` subclass is accepted like it is during descriptor
    construction, but its custom hashing and equality must never run while the
    registry lock is held. Inspecting the object's actual type also avoids
    consulting a spoofed ``__class__`` attribute.
    """
    if not issubclass(type(name), str):
        raise TypeError("A dtype lookup name must be a string")
    return str.__str__(cast("str", name))


def _registered_dtype(name: str) -> DType:
    """Return the registered dtype named ``name``."""
    name = _normalize_lookup_name(name)
    with _REGISTRY_LOCK:
        dtype = _REGISTRY.get(name)
    if dtype is None:
        raise LookupError(f"No StrideWeave dtype named {name!r} is registered")
    return dtype


def _resolve_pickled_dtype(name: str, structure: tuple[object, ...]) -> DType:
    """Resolve a pickled descriptor against the receiving process's registry.

    A pickle carries a descriptor's name and complete structure rather than a
    rebuildable definition, because identity — not a copy — is what SW002
    guarantees. The receiving process must therefore already hold the matching
    registration: built-ins register when ``strideweave`` is imported, while a
    dtype registered dynamically must be registered again by the receiver. The
    structure spans every referenced descriptor, so a receiver that registered
    the same names over different categories, widths, planes, or scale levels is
    rejected instead of quietly substituting its own representation.

    Raises:
        LookupError: If no dtype of that name is registered here.
        ValueError: If the registered dtype describes a different representation
            than the pickled one, which would otherwise be substituted silently.
    """
    name = _normalize_lookup_name(name)
    with _REGISTRY_LOCK:
        dtype = _REGISTRY.get(name)
    if dtype is None:
        raise LookupError(
            f"No StrideWeave dtype named {name!r} is registered in this "
            "process; a dynamically registered dtype must be registered again "
            "before a pickle referring to it can be loaded"
        )
    if dtype._structure != structure:
        where, registered, pickled = _structure_difference(dtype._structure, structure)
        raise ValueError(
            f"Registered dtype {name!r} does not describe the pickled "
            f"representation: {where} is {_described_structure_value(registered)} "
            f"here and {_described_structure_value(pickled)} in the pickle; a "
            "differently defined descriptor is never substituted for the "
            "pickled one"
        )
    return dtype


def _referenced_structure(dtype: DType | None) -> object:
    """Return the complete identity structure of a referenced descriptor.

    A reference expands to the referenced descriptor's name *and* its whole
    structure, which itself expands the descriptors it references. Comparing two
    structures therefore compares the entire reachable descriptor graph rather
    than a set of names that a receiving process could have defined differently.

    Only a descriptor that finished finalization *and* is the descriptor its
    name resolves to may be referenced. Proving canonical registry identity —
    rather than trusting a recorded structure — is what keeps a graph honest:
    an object whose own registration was rejected still carries the structure it
    had computed, and must not become the supertype or plane of a descriptor
    that does register. It also keeps the expansion finite, because a descriptor
    is registered only after the constructors of everything it names, so the
    graph is acyclic by construction and a descriptor that tries to describe
    itself is rejected explicitly.
    """
    if dtype is None:
        return _encoded_leaf(None)
    # Read the name defensively: a referenced object that never finished
    # initializing may not have one, and the diagnostic below has to survive
    # that rather than fail with an attribute error of its own.
    name = cast("str | None", getattr(dtype, "_name", None))
    structure = getattr(dtype, "_structure", None)
    if name is None or structure is None or not getattr(dtype, "_finalized", False):
        raise ValueError(
            f"Dtype {name!r} is not finalized, so its structure cannot be "
            "referenced: a descriptor may describe only descriptors that were "
            "already registered, never itself"
        )
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(name)
    if registered is not dtype:
        raise ValueError(
            f"Dtype {name!r} is not the descriptor registered under that name, "
            "so its structure cannot be referenced: a descriptor may describe "
            "only canonical registered descriptors"
        )
    return (_encoded_leaf(name), *structure)


def _canonical_planes(
    owner: type, simple_types: Iterable[SimpleDType]
) -> tuple[SimpleDType, ...]:
    """Return the descriptor-owned copy of an externally supplied plane mapping.

    Copying is what makes the mapping canonical: the descriptor stops depending
    on the caller's collection, so mutating that collection afterwards cannot
    change a registered descriptor's planes, carrier count, recorded structure,
    or pickle identity. The copy is validated here, before any part of it is
    published or referenced.
    """
    try:
        planes = tuple(simple_types)
    except TypeError as error:
        raise TypeError(
            f"{owner.__name__} simple_types must be an iterable of SimpleDType "
            f"descriptors, not {type(simple_types).__name__}"
        ) from error
    if not planes:
        raise ValueError(
            f"{owner.__name__} simple_types must name at least one representation plane"
        )
    for position, plane in enumerate(planes):
        if not isinstance(plane, SimpleDType):
            raise TypeError(
                f"{owner.__name__} simple_types[{position}] must be a "
                f"SimpleDType descriptor, not {type(plane).__name__}"
            )
    return planes


def _require_name_available(name: str) -> None:
    """Reject ``name`` if a descriptor already claims it, without mutating.

    Callers hold :data:`_REGISTRY_LOCK`, so the answer stays true through the
    commit that follows it.
    """
    if name in _REGISTRY:
        raise ValueError(f"A StrideWeave dtype named {name!r} is already registered")


def _finalize(dtype: DType) -> None:
    """Validate a completely initialized descriptor and publish it atomically.

    This is the single finalization boundary of the dtype model, reached from
    :meth:`_DTypeNamespace.__call__` once the most-derived ``__init__`` has
    returned. Running here rather than inside a constructor is what makes the
    transaction safe: validation sees the finished descriptor, and the name and
    any structural uniqueness key are claimed together in one critical section.
    A constructor that raises after its base initialized therefore leaves
    nothing behind, and a reader can never observe an unsealed descriptor
    because the seal is applied before the registry entry that exposes it.
    A finalization that fails is undone completely: the object keeps no
    structure and no seal, so an implementation that leaked ``self`` out of a
    rejected constructor holds an inert object rather than a usable descriptor.

    Every decision here reads the contract specifications the descriptor's type
    resolves through its MRO, so validation, the canonical structure layers, and
    structural uniqueness are the model's rather than the descriptor's. The one
    thing the descriptor itself contributes is
    :meth:`DType.structure_extension`.
    """
    specs = _contract_specs(type(dtype))
    name = getattr(dtype, "_name", None)
    if name is None:
        raise TypeError(
            f"{type(dtype).__name__} did not initialize its DType base: a "
            "descriptor implementation must call super().__init__ with its "
            "name before construction completes"
        )
    if type(name) is not str:
        # The base normalizes the name it is given, so a non-exact string here
        # was substituted afterwards. It never reaches the registry, where
        # hashing and comparing it would run user code during a commit.
        raise TypeError(
            f"{type(dtype).__name__} replaced its registered name with a "
            f"{type(name).__name__}: a dtype name must be an exact string"
        )
    _reject_shadowed_instance_state(dtype)
    for spec in specs:
        if spec.validate is not None:
            spec.validate(dtype)
    # The structure is computed once, from a complete descriptor, and recorded
    # before the descriptor becomes referenceable: pickling, structural
    # uniqueness, and every later reference read this one value.
    extension = dtype.structure_extension()
    if type(extension) is not tuple:
        raise TypeError(
            f"{type(dtype).__name__}.structure_extension must return a tuple, "
            f"not {type(extension).__name__}"
        )
    # Each contract contributes its own fragment and nothing else, so the
    # canonical layers of a descriptor are exactly those of the contracts it
    # inherits; only the trailing extension is the implementation's.
    structure = (
        *(fragment for spec in specs for fragment in spec.layer(dtype)),
        _encoded_structure(dtype, extension),
    )
    _require_encoded_structure(dtype, structure)
    object.__setattr__(dtype, "_structure", structure)
    try:
        # A contract that makes its descriptors unique by structure as well as
        # by name is keyed on the completed structure recorded above, so the key
        # is derived from the published identity rather than reported by the
        # descriptor.
        conflict = next(
            (
                spec.structural_conflict
                for spec in specs
                if spec.structural_conflict is not None
            ),
            None,
        )
        # ``structure_extension`` has now run, and it assigns to ``self`` as
        # naturally as ``__init__`` does. The scan above cannot see what it
        # added, so the authoritative one runs here: after the last callback and
        # before anything is claimed or sealed, so a descriptor that ended up
        # shadowing an owned member is rejected rather than published reporting
        # a structure other than the recorded one.
        _reject_shadowed_instance_state(dtype)
        with _REGISTRY_LOCK:
            _require_name_available(name)
            if conflict is not None:
                claimed = _STRUCTURES.get(structure)
                if claimed is not None:
                    raise ValueError(conflict(claimed))
                _STRUCTURES[structure] = dtype
            # Sealing precedes publication: the entry below is what makes the
            # descriptor reachable, so it must already be immutable by then.
            object.__setattr__(dtype, "_finalized", True)
            try:
                _REGISTRY[name] = dtype
            except BaseException:
                # The structure was claimed first, so a failed name commit must
                # release it; reserving a representation whose name never landed
                # would lock that representation out for the rest of the
                # process.
                if conflict is not None:
                    del _STRUCTURES[structure]
                raise
    except BaseException:
        _discard_finalization(dtype)
        raise


def _discard_finalization(dtype: DType) -> None:
    """Return a descriptor whose finalization failed to its unfinished state.

    Nothing published survives a failed transaction, so nothing recorded on the
    object may either. Clearing the seal and the structure keeps an object that
    escaped a rejected constructor from passing the reference-eligibility check
    or from looking immutable while it is not registered anywhere.
    """
    object.__setattr__(dtype, "_finalized", False)
    try:
        object.__delattr__(dtype, "_structure")
    except AttributeError:
        pass


# Installed built-in names, held in module state rather than on the class so
# that ordinary ``DType`` attribute assignment cannot shadow the bookkeeping the
# namespace protection reads.
_INSTALLED_BUILTINS: set[str] = set()

# Set once the separate built-in bootstrap module finishes. After it, the
# ``DType`` namespace is exactly those bindings and accepts no further descriptor
# attribute, installed or assigned.
_BUILTINS_SEALED = False


class _DTypeNamespace(type):
    """Metaclass freezing the descriptor namespace of ``DType``.

    Built-in descriptors are reachable as ``DType.Float32`` and through
    ``DType.from_name``. Both must name the same object for the lifetime of the
    process, so an installed binding rejects reassignment and deletion, and no
    descriptor may be added to the namespace afterwards: a dtype registered at
    runtime is reached through :meth:`DType.from_name` and
    :meth:`DType.registered` instead.
    """

    # Declared here so the construction hook below can read the flag that
    # ``DType.__init_subclass__`` maintains on every descriptor class.
    _abstract: bool

    def __call__(cls, *args: object, **kwargs: object) -> Any:
        """Construct, validate, seal, and publish a descriptor as one step.

        Every descriptor is created through here, so this is where the model
        decides that a class may be instantiated at all and where the finished
        object becomes the registered identity. An implementation therefore
        never orders registry mutation itself: it initializes its own fields and
        describes its structure, and :func:`_finalize` publishes the result. The
        return type stays the constructed class: a metaclass ``__call__`` cannot
        restate that in the type system, and the stub deliberately omits this
        hook so callers keep the precise constructor signatures.
        """
        if cls._abstract:
            raise TypeError(
                f"{cls.__name__} is abstract; construct a registered "
                "descriptor implementation such as DTypeCategory, SimpleDType, "
                "or BlockScaledDType"
            )
        descriptor = cast("DType", super().__call__(*args, **kwargs))
        _finalize(descriptor)
        return descriptor

    def _install_builtin(cls, name: str, dtype: DType) -> None:
        """Bind ``dtype`` as a protected built-in attribute named ``name``."""
        if name in _INSTALLED_BUILTINS:
            raise AttributeError(f"DType.{name} is already installed")
        if _BUILTINS_SEALED:
            raise AttributeError(
                f"DType.{name} cannot be installed: the built-in dtype namespace "
                "is frozen once StrideWeave has finished importing"
            )
        type.__setattr__(cls, name, dtype)
        _INSTALLED_BUILTINS.add(name)

    def __setattr__(cls, name: str, value: object) -> None:
        if name in _INSTALLED_BUILTINS:
            raise AttributeError(
                f"DType.{name} is a built-in dtype binding and cannot be reassigned"
            )
        if isinstance(value, DType):
            raise AttributeError(
                f"DType.{name} cannot be added: the DType namespace holds exactly "
                "the built-in descriptors, and a dtype registered afterwards is "
                "reached through DType.from_name and DType.registered"
            )
        _reject_owned_assignment(cls, name)
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if name in _INSTALLED_BUILTINS:
            raise AttributeError(
                f"DType.{name} is a built-in dtype binding and cannot be deleted"
            )
        _reject_owned_assignment(cls, name)
        super().__delattr__(name)


class DType(metaclass=_DTypeNamespace):
    """Immutable root descriptor of the StrideWeave dtype hierarchy.

    ``DType`` is abstract: every descriptor is a :class:`DTypeCategory`, a
    :class:`SimpleDType`, or a :class:`CompoundDType`. Descriptors are compared
    by identity. The attributes of this class are exactly the built-in
    descriptors, and that namespace is frozen; a descriptor registered
    afterwards is reached through :meth:`from_name` and :meth:`registered` and
    cannot be installed here.

    Args:
        name: Unique registered name of the descriptor.
        supertype: Category that immediately encloses it, or ``None`` at the
            root of the hierarchy.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.Float32.is_simple()
        True
        >>> sw.DType.Float32.is_subtype_of(sw.DType.Floating)
        True
        >>> extension = sw.SimpleDType(
        ...     "Float16", bits=16, supertype=sw.DType.Floating
        ... )
        >>> sw.DType.from_name("Float16") is extension
        True
    """

    __slots__ = ("_finalized", "_name", "_structure", "_supertype")

    Any: ClassVar[DTypeCategory]
    Floating: ClassVar[DTypeCategory]
    Integer: ClassVar[DTypeCategory]
    Float32: ClassVar[SimpleDType]
    Int32: ClassVar[SimpleDType]
    Int8: ClassVar[SimpleDType]
    E8M0: ClassVar[SimpleDType]
    E5M2: ClassVar[SimpleDType]
    E4M3: ClassVar[SimpleDType]
    E3M2: ClassVar[SimpleDType]
    E2M3: ClassVar[SimpleDType]
    E2M1: ClassVar[SimpleDType]
    MXFP8_E4M3: ClassVar[BlockScaledDType]
    MXFP8_E5M2: ClassVar[BlockScaledDType]
    MXFP6_E3M2: ClassVar[BlockScaledDType]
    MXFP6_E2M3: ClassVar[BlockScaledDType]
    MXFP4: ClassVar[BlockScaledDType]
    MXINT8: ClassVar[BlockScaledDType]
    NVFP4: ClassVar[BlockScaledDType]

    _abstract: ClassVar[bool] = True

    def __init_subclass__(cls, *, abstract: bool = True, **kwargs: object) -> None:
        """Record whether ``cls`` is a constructible descriptor implementation.

        Abstractness is declared rather than inferred, so a subclass that adds
        nothing — and therefore describes no representation of its own — stays
        abstract instead of silently becoming a constructible dtype whose
        descriptors would claim registry names. The class is also refused if it
        redefines state the model owns.
        """
        super().__init_subclass__(**kwargs)
        cls._abstract = abstract
        _reject_shadowed_state(cls)
        _reject_shadowing_bases(cls)

    def __init__(self, name: str, *, supertype: DTypeCategory | None = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("A dtype name must be a non-empty string")
        if supertype is not None and not isinstance(supertype, DTypeCategory):
            raise TypeError("A dtype supertype must be a DTypeCategory")
        # Keep an exact ``str``: a subclass instance used as a registry key
        # would hash and compare through user code during a commit, so a name
        # that merely behaves like a string never reaches the registry.
        self._name = str.__str__(name)
        self._supertype = supertype

    @classmethod
    def registered(cls) -> tuple[Self, ...]:
        """Return every registered descriptor that is an instance of ``cls``.

        This spans both built-in and later-registered descriptors, so it is the
        way to discover extension dtypes, which are never installed as
        :class:`DType` attributes.
        """
        # ``isinstance(dtype, cls)`` establishes membership in ``cls``; the cast
        # only restates that for the checker, which cannot narrow through the
        # untyped registry values.
        with _REGISTRY_LOCK:
            snapshot = tuple(_REGISTRY.values())
        return tuple(
            cast("Self", dtype) for dtype in snapshot if isinstance(dtype, cls)
        )

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Return the registered descriptor named ``name`` and typed as ``cls``.

        Built-in names resolve to the same object as the matching
        :class:`DType` attribute; a name registered afterwards resolves here
        only, because registration does not extend the class namespace.
        """
        dtype = _registered_dtype(name)
        if not isinstance(dtype, cls):
            raise LookupError(
                f"Registered dtype {name!r} is not a {cls.__name__} descriptor"
            )
        return cast("Self", dtype)

    @property
    def name(self) -> str:
        """Return the unique registered name of this descriptor."""
        return self._name

    @property
    def value(self) -> str:
        """Return :attr:`name`, the pre-descriptor compatibility alias.

        ``DType`` used to be an ``Enum`` whose members carried their canonical
        string as ``value``. The descriptor model makes ``name`` the canonical
        field; ``value`` is a read-only alias kept so existing callers such as
        ``tensor.dtype().value`` keep working, and it can never diverge from
        ``name``.
        """
        return self._name

    @property
    def supertype(self) -> DTypeCategory | None:
        """Return the immediately enclosing category, or ``None`` at the root."""
        return self._supertype

    def supertypes(self) -> tuple[DTypeCategory, ...]:
        """Return the enclosing categories, innermost first."""
        chain: list[DTypeCategory] = []
        current = self._supertype
        while current is not None:
            chain.append(current)
            current = current.supertype
        return tuple(chain)

    def is_simple(self) -> bool:
        """Return whether this descriptor is one fixed-width scalar encoding.

        This classifies the *representation*, not backend availability: a simple
        dtype is a single scalar encoding of an exact width rather than a
        composition of subtensors. Whether any carrier accepts it is a separate,
        carrier-by-carrier question, so a registered encoding that no current
        carrier stores — ``E4M3``, or a simple extension — still reports
        ``True``.

        Returns:
            ``True`` for a :class:`SimpleDType`, ``False`` for a category or a
            compound descriptor.

        Examples:
            >>> import strideweave as sw
            >>> sw.DType.Float32.is_simple()
            True
            >>> sw.DType.E4M3.is_simple()
            True
            >>> sw.DType.MXFP4.is_simple()
            False
        """
        return False

    def is_category(self) -> bool:
        """Return whether this descriptor is an abstract category."""
        return False

    def is_compound(self) -> bool:
        """Return whether this descriptor needs several simple-dtype planes."""
        return False

    def is_opaque_storage(self) -> bool:
        """Return whether legacy carriers may store this descriptor opaquely."""
        return False

    def is_subtype_of(self, other: DType) -> bool:
        """Return whether this descriptor is ``other`` or is enclosed by it."""
        if not isinstance(other, DType):
            raise TypeError("A dtype subtype query requires a DType")
        current: DType | None = self
        while current is not None:
            if current is other:
                return True
            current = current.supertype
        return False

    def structure(self) -> tuple[object, ...]:
        """Return the recorded structure that defines this representation.

        The structure is computed once, when the descriptor is finalized, from
        the complete descriptor and the whole graph it references. It is the
        authority for structural identity: pickling carries it, structural
        uniqueness keys are built from it, and a descriptor that references this
        one embeds it. Because it is recorded rather than recomputed, it can
        never drift from the identity the registry published.

        Its canonical layers come from the contracts this descriptor's class
        inherits and are assembled by the dtype model, so an implementation can
        add to its representation through :meth:`structure_extension` but never
        omit a field its contract describes.

        Returns:
            Immutable nested tuple describing this descriptor and, recursively,
            every descriptor it names.

        Examples:
            >>> import strideweave as sw
            >>> sw.DType.Float32.structure() == sw.DType.Float32.structure()
            True
            >>> sw.DType.Float32.structure() == sw.DType.Int32.structure()
            False
        """
        return self._structure

    def structure_extension(self) -> tuple[object, ...]:
        """Return the structure this class adds beyond its contract.

        This is the only part of a descriptor's identity an implementation
        contributes; the canonical layers before it belong to the contracts the
        class inherits. Override this when a descriptor implementation carries
        state that makes its representation distinct — anything two descriptors
        could differ in while their contract structure matched. The result is
        appended to the descriptor's structure, so it takes part in the pickle
        compatibility check and, for a kind that is unique by structure, in that
        uniqueness.

        It is consulted exactly once, during finalization, and the result is
        recorded. :meth:`structure` is therefore the authority afterwards: an
        implementation that computes this from state it keeps mutating changes
        only what this method returns, never the registered identity.

        Returns:
            Tuple of exact strings, numbers, ``None``, :data:`Whole`, and
            tuples of those. The default is empty, which means the class adds
            nothing to the representation its contract already describes.

        Examples:
            >>> import strideweave as sw
            >>> class Tagged(sw.CompoundDType, abstract=False):
            ...     __slots__ = ("_tag",)
            ...
            ...     def __init__(self, name, *, tag):
            ...         super().__init__(
            ...             name,
            ...             supertype=sw.DType.Any,
            ...             simple_types=(sw.DType.Float32,),
            ...         )
            ...         self._tag = tag
            ...
            ...     def structure_extension(self):
            ...         return (self._tag,)
            >>> Tagged("TaggedPlane", tag="row-major").structure_extension()
            ('row-major',)
        """
        return ()

    def __setattr__(self, name: str, value: object) -> None:
        # Descriptor state is open only while its constructor runs; finalization
        # seals it before the descriptor becomes reachable, so an implementation
        # assigns its own fields normally and every published descriptor is
        # immutable for the rest of the process.
        if getattr(self, "_finalized", False):
            raise AttributeError(f"{type(self).__name__} descriptors are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_finalized", False):
            raise AttributeError(f"{type(self).__name__} descriptors are immutable")
        object.__delattr__(self, name)

    def __copy__(self) -> DType:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> DType:
        return self

    def __reduce__(self) -> tuple[object, ...]:
        from ..dtype import _unpickle_dtype

        return (_unpickle_dtype, (self._name, self._structure))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r})"


def _root_layer(dtype: DType) -> tuple[object, ...]:
    """Return the root contract's own fragment of a descriptor's structure.

    Each contract's fragment names the contract rather than the most derived
    class, so a subclass describes the same representation as the contract it
    inherits unless it adds structure of its own through
    :meth:`DType.structure_extension`. Every descriptor it references is
    expanded through :func:`_referenced_structure`.
    """
    return (_encoded_leaf("DType"), _referenced_structure(dtype._supertype))


def _root_validation(dtype: DType) -> None:
    """Check a completely initialized descriptor against the root contract.

    This runs after the most-derived ``__init__`` returned and before the
    descriptor is sealed or reachable, so it may read any property the
    implementation defines.
    """
    if not isinstance(dtype, (DTypeCategory, SimpleDType, CompoundDType)):
        raise TypeError(
            f"{type(dtype).__name__} is not a StrideWeave descriptor kind: "
            "every dtype is a DTypeCategory, a SimpleDType, or a CompoundDType"
        )


_declare_contract(
    DType, _ContractSpec(_ROOT_OWNED, layer=_root_layer, validate=_root_validation)
)


class DTypeCategory(DType, abstract=False):
    """Abstract dtype category such as ``Floating`` or ``Integer``.

    A category expresses a relationship between descriptors. It has no bit
    width and is not itself a representation, simple or compound. ``Any`` and ``Floating``
    additionally carry the legacy opaque-storage disposition: ``Generic``
    accepts both, and ``FileBacked`` accepts ``Floating``. ``Integer`` carries
    no such disposition, and neither does a category registered later, so no
    carrier accepts it.

    Args:
        name: Unique registered name, such as ``"Integer"``.
        supertype: Enclosing category, or ``None`` for a root category.
        opaque_storage: Whether legacy carriers may store values tagged with
            this category as Python objects or width-unspecified numbers. It
            records the disposition only; a carrier still accepts exactly the
            descriptors its own documented set names.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.Integer.is_category()
        True
        >>> sw.DType.Integer.is_simple()
        False
    """

    __slots__ = ("_opaque_storage",)

    def __init__(
        self,
        name: str,
        *,
        supertype: DTypeCategory | None = None,
        opaque_storage: bool = False,
    ) -> None:
        super().__init__(name, supertype=supertype)
        self._opaque_storage = bool(opaque_storage)

    def is_category(self) -> bool:
        """Return ``True``: every category is abstract."""
        return True

    def is_opaque_storage(self) -> bool:
        """Return whether legacy carriers may store this category opaquely."""
        return self._opaque_storage


def _category_layer(dtype: DTypeCategory) -> tuple[object, ...]:
    """Return the category contract's own fragment of a structure."""
    return (_encoded_leaf("DTypeCategory"), _encoded_leaf(dtype._opaque_storage))


_declare_contract(DTypeCategory, _ContractSpec(_CATEGORY_OWNED, layer=_category_layer))


class SimpleDType(DType, abstract=False):
    """One fixed-width scalar encoding, storable homogeneously by one carrier.

    A simple dtype declares an exact positive bit width and the category that
    encloses it, so ``Float32`` is a member of ``Floating`` and ``Int32`` is a
    member of ``Integer``. Being simple describes the encoding, not carrier
    support: ``Float32`` and ``Int32`` are the concrete simple storage dtypes
    carriers accept today, while the registered narrow encodings such as
    ``E4M3`` are structural only.

    Args:
        name: Unique registered name, such as ``"Float32"``.
        bits: Exact storage width of one element, in bits.
        supertype: Category this encoding belongs to.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.Int32.bits
        32
        >>> sw.DType.Int32.supertype is sw.DType.Integer
        True
    """

    __slots__ = ("_bits",)

    def __init__(self, name: str, *, bits: int, supertype: DTypeCategory) -> None:
        if not isinstance(supertype, DTypeCategory):
            raise TypeError("A simple dtype must belong to a DTypeCategory")
        super().__init__(name, supertype=supertype)
        if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
            raise ValueError("A simple dtype must declare an exact positive bit width")
        # Keep an exact ``int``: the width takes part in this descriptor's
        # structure, which is hashed as a registry key and compared across
        # processes, so a subclass instance must not reach it.
        self._bits = int(bits)

    @property
    def bits(self) -> int:
        """Return the exact storage width of one element, in bits."""
        return self._bits

    def is_simple(self) -> bool:
        """Return ``True``: this is one fixed-width scalar encoding.

        The answer describes the representation and says nothing about which
        carriers accept it, so a registered encoding that no carrier stores yet
        still reports ``True``.

        Returns:
            ``True`` for every :class:`SimpleDType`, independently of carrier
            support.

        Examples:
            >>> import strideweave as sw
            >>> sw.DType.E4M3.is_simple()
            True
        """
        return True


def _simple_layer(dtype: SimpleDType) -> tuple[object, ...]:
    """Return the simple contract's own fragment of a structure."""
    return (_encoded_leaf("SimpleDType"), _encoded_leaf(dtype._bits))


_declare_contract(SimpleDType, _ContractSpec(_SIMPLE_OWNED, layer=_simple_layer))


class CompoundDType(DType):
    """Descriptor for a value represented by several simple-dtype planes.

    A compound descriptor is never carrier storage itself. Its ``simple_types``
    give the ordered simple dtype of each plane, so plane ``i`` is stored by a
    carrier whose dtype is ``simple_types[i]``. ``CompoundDType`` is itself
    abstract, because a plane mapping is the whole content of a compound
    representation and this class defines none.

    A subclass is the supported way to add a compound representation. It
    declares ``abstract=False``, initializes its own fields in ``__init__``, and
    hands its planes to ``super().__init__``; no registry API is involved::

        class Planar(sw.CompoundDType, abstract=False):
            __slots__ = ("_label",)

            def __init__(self, name, *, planes, label):
                super().__init__(name, supertype=sw.DType.Any, simple_types=planes)
                self._label = label

    The planes are copied into a tuple this class owns, and ``simple_types`` and
    ``num_carriers`` read that copy, so they cannot be overridden and cannot
    drift afterwards — mutating whatever collection was passed in changes
    nothing about the registered descriptor.

    Args:
        name: Unique registered name of the descriptor.
        supertype: Category that immediately encloses it, or ``None``.
        simple_types: Ordered planes of the representation, as any iterable of
            registered :class:`SimpleDType` descriptors. At least one is
            required.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.MXFP8_E4M3.is_compound()
        True
        >>> sw.DType.MXFP8_E4M3.num_carriers
        2
    """

    # ``simple_types`` and ``num_carriers`` report the canonical plane mapping
    # this class owns, so :data:`_COMPOUND_OWNED` keeps them unoverridable. An
    # override could serve a live view of mutable state, which would let a
    # finalized descriptor's observable representation disagree with the
    # structure recorded for its identity and its pickle.
    __slots__ = ("_simple_types",)

    def __init__(
        self,
        name: str,
        *,
        supertype: DTypeCategory | None = None,
        simple_types: Iterable[SimpleDType],
    ) -> None:
        super().__init__(name, supertype=supertype)
        self._simple_types = _canonical_planes(type(self), simple_types)

    def is_compound(self) -> bool:
        """Return ``True``: every compound dtype spans several planes."""
        return True

    @property
    def simple_types(self) -> tuple[SimpleDType, ...]:
        """Return the ordered simple dtype of every representation plane."""
        return self._simple_types

    @property
    def num_carriers(self) -> int:
        """Return how many carriers one value of this dtype occupies."""
        return len(self._simple_types)


def _compound_layer(dtype: CompoundDType) -> tuple[object, ...]:
    """Return the compound contract's own fragment of a structure.

    The plane mapping is the common compound contract, so it is expanded
    centrally: every implementation's structure describes the complete dtype of
    each plane rather than only its name.
    """
    return (
        _encoded_leaf("CompoundDType"),
        tuple(_referenced_structure(plane) for plane in dtype._simple_types),
    )


def _compound_validation(dtype: CompoundDType) -> None:
    """Require the compound base to have taken ownership of the planes."""
    if getattr(dtype, "_simple_types", None) is None:
        raise TypeError(
            f"{type(dtype).__name__} did not initialize its compound base: a "
            "compound descriptor must pass simple_types to super().__init__"
        )


_declare_contract(
    CompoundDType,
    _ContractSpec(
        _COMPOUND_OWNED, layer=_compound_layer, validate=_compound_validation
    ),
)
