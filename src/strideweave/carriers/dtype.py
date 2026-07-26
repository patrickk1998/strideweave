"""Runtime dtype descriptors and the registry shared by StrideWeave carriers.

StrideWeave distinguishes descriptors that describe *what a value is* from
descriptors that describe *how one carrier stores it*:

- :class:`DTypeCategory` is an abstract relationship such as ``Floating`` or
  ``Integer``. A category has no bit width and is not itself a representation.
- :class:`SimpleDType` is one fixed-width scalar encoding rather than a
  composition of subtensors, so a single carrier could store it homogeneously.
  ``Float32`` and ``Int32`` are the concrete simple storage dtypes carriers
  support today; ``Int8``, ``E8M0``, ``E5M2``, ``E4M3``,
  ``E3M2``, ``E2M3``, and ``E2M1`` are registered simple encodings as well, but
  structural ones that no carrier stores yet. Being simple describes the
  encoding, never a promise that some carrier accepts it.
- :class:`CompoundDType` describes a logical value whose physical representation
  is composed from several simple-dtype planes. :class:`BlockScaledDType` is the
  block-scaled case: one simple element dtype plus a linear chain of simple
  scale :class:`Level` entries. A compound descriptor is never carrier storage
  itself; its ``simple_types`` are.
- ``DType.Any`` and ``DType.Floating`` additionally carry the legacy *opaque
  storage* disposition, the one way a category is accepted as storage:
  ``Generic`` accepts both for Python-object and width-unspecified numeric
  values, and ``FileBacked`` accepts ``Floating``. They are explicitly opaque
  rather than fixed-size, so they never claim a bit width. Each carrier's
  accepted set is exact, and ``Integer`` belongs to none of them.

Every registered descriptor is an immutable singleton, which keeps the ``SW002``
identity-comparison invariant valid. Constructing one is a single transaction
owned by the dtype implementation rather than a sequence an implementation has to order
correctly: the most-derived ``__init__`` runs first, the complete descriptor is
then validated, and only afterwards is it sealed against mutation and published
under the registry lock. A constructor that raises — at any depth — therefore
leaves neither its name nor its structure claimed, and no partially initialized
descriptor is ever reachable through :meth:`DType.from_name`. Descriptor state
is writable only inside that construction window, which is what lets an
extension assign its own fields in ``__init__`` without touching the registry —
its own fields only, because a descriptor whose state shadows a member the model
owns is rejected instead of published.

What makes a descriptor's identity is likewise the model's rather than the
descriptor's. Each contract class — the root, categories, simple, compound, and
block-scaled dtypes — has one immutable specification held in private module state,
carrying the members it owns, the fragment it contributes to a canonical
structure, the validation it requires, and whether its descriptors are
additionally unique by structure. A descriptor is finalized against the
specifications its class resolves through its method resolution order, general
to specific, so an implementation inherits its contract whole: it cannot omit a
canonical layer from its fingerprint, decline the uniqueness its contract
imposes, or weaken a check it still claims to satisfy. The one contribution an
implementation makes to its representation is
:meth:`DType.structure_extension`.

Those checks are guardrails for cooperative extensions, not a sandbox. The model
validates a descriptor class's initial hierarchy when the class is created,
validates the completed descriptor during finalization, seals registered
instances, and freezes the built-in namespace. Beyond that it relies on the
extension contract: from the moment a descriptor class is created, that
extension class and every base contributing behavior to it must stay as they
were, and representation-bearing state must be initialized before registration
and described by a pure, stable :meth:`DType.structure_extension`. Concealing a
descriptor's own state, reassigning ``__slots__``, mutating an extension class
or a participating mixin after descriptor class creation, ``object.__setattr__``,
``type.__setattr__``, and reaching into the dtype implementation's private state are
unsupported: they forfeit the guarantees above rather than being intercepted.

Lookup is deliberately split by origin:

- The *built-in* descriptors are the protected class attributes of
  :class:`DType`, such as ``DType.Float32``. They are fixed at import and cannot
  be reassigned or deleted, and the namespace accepts no further descriptor
  attribute afterwards.
- A descriptor registered later by constructing a :class:`SimpleDType`,
  :class:`DTypeCategory`, or :class:`BlockScaledDType` is reached through the
  registry APIs :meth:`DType.from_name` and :meth:`DType.registered`.
  Registration never installs a class attribute, so extensions cannot collide
  with the :class:`DType` namespace and the attribute surface stays typable.

``name`` is a descriptor's canonical string; ``value`` is a read-only alias for
it kept for callers written against the previous ``Enum`` model.
Block-scaled descriptors are additionally unique by structure, so two
descriptors describe the same representation only when they are the same
object.

Pickling preserves identity by resolving a descriptor's name and structure
against the *receiving* process's registry rather than rebuilding a descriptor.
A built-in therefore unpickles anywhere ``strideweave`` is imported, while a
dtype registered dynamically requires the receiving process to have registered a
matching descriptor first; without it, loading raises :class:`LookupError`, and
a same-named descriptor of a different structure raises :class:`ValueError`
instead of being substituted.

A structure describes the *complete referenced graph*, not a set of names: each
contract the descriptor's class inherits contributes its own fields, and every
descriptor it names —
supertype, compound plane, block element, or scale — is expanded into that
descriptor's own structure recursively. A receiver that registered the same
names over different widths, categories, opaque dispositions, planes, or scale
levels is therefore rejected rather than silently substituted. An
implementation that carries state beyond its contract adds it through
:meth:`DType.structure_extension`.

Every leaf of a structure is encoded as a string naming its exact type
alongside its value, so comparison is type-exact rather than numeric: ``True``,
``1``, and ``1.0`` are three different representations, and a ``NaN`` gets one
deterministic spelling instead of a value that is not even equal to itself.
Floats are recorded in their exact hexadecimal form, which round-trips without
precision loss. Shipping a dtype definition itself — a
cross-process descriptor schema — is deliberately not part of this model and
remains possible future work.

These descriptors are structural only. Registering a narrow element or scale
encoding such as ``E4M3`` does not give any carrier the ability to store it, and
no quantization, rounding, or dequantization semantics are defined here.
"""

from __future__ import annotations

from importlib import import_module

from ._dtype import model as _model
from ._dtype.block_scaled import BlockScaledDType, Level, SymbolicBits
from ._dtype.contracts import _CONTRACT_SPECS, _ContractSpec  # noqa: F401
from ._dtype.model import CompoundDType, DType, DTypeCategory, SimpleDType
from ._dtype.storage import validate_storage_dtype
from ._dtype.structure import Whole, WholeExtent


def _unpickle_dtype(name: str, structure: tuple[object, ...]) -> DType:
    """Resolve a pickled descriptor through the internal registry model."""
    return _model._resolve_pickled_dtype(name, structure)


# Preserve the historical public module identity of exported classes and pickle
# payloads even though their implementations now live in private modules.
for _public_type in (
    DType,
    DTypeCategory,
    SimpleDType,
    CompoundDType,
    WholeExtent,
    Level,
    SymbolicBits,
    BlockScaledDType,
):
    type.__setattr__(_public_type, "__module__", __name__)
del _public_type

# Private compatibility aliases remain readable for diagnostics. Code that
# needs to replace module-owned state for failure injection must patch
# ``strideweave.carriers._dtype.model`` directly.
_REGISTRY = _model._REGISTRY
_STRUCTURES = _model._STRUCTURES
_REGISTRY_LOCK = _model._REGISTRY_LOCK

# Built-ins are installed only after every contract class exists and the public
# module identities above have been restored.
_builtins = import_module("strideweave.carriers._dtype.builtins")

del _builtins


__all__ = [
    "BlockScaledDType",
    "CompoundDType",
    "DType",
    "DTypeCategory",
    "Level",
    "SimpleDType",
    "SymbolicBits",
    "Whole",
    "WholeExtent",
    "validate_storage_dtype",
]
