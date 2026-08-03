# StrideWeave

[![CI](https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml)

StrideWeave is a research tensor and autograd framework built around hierarchical,
CuTe-style layouts. A tensor combines a carrier, a physical offset, and a
layout. A carrier owns or references storage and dispatches the operations it
supports; StrideWeave deliberately has no separate device abstraction.

The project is currently a tested prototype rather than a complete PyTorch
replacement. It provides native CPU kernels, a Python reference carrier,
autograd, hierarchical layout transformations, a small module system, and
ergonomic layers (`strideweave.nn` with a minimal layer library and optimizer,
`strideweave.friendly` with tensor factories), but does not yet include
accelerator carriers.

## Core Model

- `Tensor(carrier, offset, layout)` references storage owned by a `Carrier`.
  This public constructor creates the conventional one-subtensor case of the
  authoritative internal representation: one logical dtype, an ordered tuple
  of carrier-backed subtensors, one placement `Layout` per level, and one
  adjacent `Layout` between each pair of levels. The carrier, offset, and layout
  properties read subtensor zero rather than parallel fields.
- `Layout` describes hierarchical `Shape` and `Stride` trees and maps logical
  coordinates to physical storage indices.
- `layout.profile` exposes the shape tree's leaf-and-nesting recipe without its
  extents or strides. `layout.is_injective` reports whether every logical
  coordinate maps to a distinct physical offset, including exact detection of
  stride-zero and overlapping non-zero layouts. `layout.broadcast_to(shape)`
  widens only extent-one leaves at the same hierarchical positions, setting
  their strides to zero; it never flattens, rank-aligns, inserts, removes, or
  reorders modes. Layout complement is undefined for non-injective layouts and
  refuses them explicitly.
- `Tiler` is the public type alias for a read-only sequence of `Layout` values.
  Layout composition APIs use tilers to describe one tile per leading
  hierarchical mode: `Layout.compose` accepts a tiler directly, while
  `Layout.divide_tiler` and `Layout.zipped_divide` use its layouts to divide
  the corresponding leading modes. Lists, tuples, and other compatible
  sequences are accepted.
- `layout.size` is the logical element count, while `layout.cosize` is the
  physical storage size the layout addresses (one past its largest offset).
  They are equal for compact layouts but `cosize` is larger for strided or
  hierarchical ones, so back a tensor with `cosize` elements — e.g. a strided
  `Layout(Shape([2, 3]), Stride([1, 4]))` has `size` 6 but `cosize` 10, so it
  needs `CPU(10)`. The `strideweave.friendly` and `strideweave.nn` layers allocate
  through `layout.cosize` for exactly this reason.
- `layout.uniform_preimage_extent(target_shape)` proves, from the immutable
  hierarchical shape/stride algebra alone, whether a layout uniformly covers a
  target coordinate space and returns its replication extent. The proof is
  rank-bounded and never enumerates logical coordinates.
- Operations dispatch through `tensor.carrier.dispatch_op(operation_name)`.
  The base `Carrier` method owns the shared dispatch policy: it calls the
  backend `_dispatch_op` factory hook, requires a fresh `Operation`, and tags
  that operation with its canonical name and exact dispatching carrier class.
  Custom carrier implementations must override `_dispatch_op`, not
  `dispatch_op`. Dispatch is uniformly instance-based; class-level
  `dispatch_op` calls are not part of the public contract. A carrier composing
  another handles custom names in its own `_dispatch_op` and returns a
  composite-owned operation adapter for delegated names. The adapter owns the
  visible autograd node, obtains a fresh nested operation through the owned
  carrier's public `dispatch_op`, lowers tensor arguments into representations
  that operation accepts, invokes it through sealed lowered execution, and wraps
  results and gradients back into the composite representation.
- Python and native operations inherit from the shared native `Operation` base.
  `Operation._forward` is a protected implementation hook and must not be
  invoked directly. Call public `forward`, or use the framework-owned sealed
  lowered-execution path from a composite adapter, so operation preflight,
  result validation, profiling, and autograd bookkeeping remain intact. Direct
  `_forward` calls are unsupported. The v0 c0 layout views (`permute`, `slice`,
  `reshape`, `as_strided`, `broadcast_to`, `squeeze`, and
  `unsqueeze`) are the only operations that may
  execute on a validated multi-subtensor representation; all other operations
  retain the one-subtensor preflight.
- Views may use different layouts and offsets while sharing the same carrier.
- In the generalized representation, placement layout `L_i` maps level
  coordinate space `c_i` into carrier `i`, while adjacent layout `S_i` maps
  `c_i` to an integer decoded in `c_(i+1).shape`. Both are ordinary CuTe-style
  `Layout` values; their structural positions distinguish physical placement
  from logical grouping. Universal validation checks the logical dtype's
  storage schema, identity-matched carrier dtypes, exact carrier-class
  homogeneity, offsets, `cosize` bounds, and adjacent source/target
  compatibility before any dtype-specific rule runs.
- Current Tensor operations take a structural one-subtensor fast path. Native
  CPU access, views, results, movement, scatter, autograd, and DLPack all read
  carrier, offset, layout, dtype, and version state through that authoritative
  representation. Autograd snapshots the complete ordered version token of
  every unique constituent carrier. Validated multi-subtensor representations
  support only the pure c0 layout views named above: they preserve every
  carrier, offset, deeper layout, and adjacent level while rebuilding L0/S0
  and revalidating the complete representation. Coordinate indexing, mutation, arithmetic,
  movement, scatter, backward, and DLPack still reject them before allocation
  or carrier mutation until their remaining per-plane semantics are implemented.
- External representation rules annotate `validate` with the public
  `RepresentationValidationContext` protocol, available from both
  `strideweave` and `strideweave.carriers.dtype`. Its read-only fields expose
  the logical dtype, ordered storage dtypes, placement and adjacent layouts,
  and level shapes only after universal validation succeeds.

For example, this creates a two-mode column-major tensor backed by a Python carrier:

```python
import strideweave as sw

layout = sw.Layout(
    sw.Shape([2, 3]),
    sw.Stride([1, 2]),
)
tensor = sw.Tensor(
    sw.Generic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    0,
    layout,
)

assert tensor[1, 2] == 6.0
```

The core namespace deliberately exposes only these composable primitives.
High-level factories such as `tensor`, `zeros`, and `ones` live in the
separate `strideweave.friendly` submodule (see Ergonomic Layers below); callers
working with the core API construct the carrier and layout explicitly.

## Carriers

StrideWeave currently provides four carrier implementations:

- `Generic(values, mutable=True, dtype=DType.Floating)` stores Python
  objects. It supports differentiable `Floating` values, non-differentiable
  arbitrary `Any` values, and the concrete simple dtypes `Float32`, `Int32`,
  and `Bool`, for which it is StrideWeave's behavioral reference
  implementation. Concrete `Bool` values are normalized to Python `bool` and
  never participate in numeric promotion or autograd.
- `CPU(size, pointer=None, *, mutable=True, dtype=DType.Float32, empty=False)`
  owns native memory or references a caller-provided address. It supports
  `Float32`, `Int32`, and `Bool` (one byte per Boolean). Owned storage is
  zero-initialized unless `empty=True`
  opts into unspecified initial values; `empty` never initializes or changes
  caller-owned memory supplied through `pointer`. Its native sources are
  organized per operation: `carriers/cpu/native/_cpu.cpp` owns only carrier
  storage and module glue, each numerical operation owns a translation unit
  under `carriers/cpu/native/ops` carrying its formula, stable kernel ID,
  binding, and single explicit registration, and the shared traversal,
  alignment, reduction, and accumulator mechanics live in the neighbouring
  headers. Duplicate dispatch names or kernel IDs are rejected rather than
  overwritten.
- `FileBacked(filename=None, mutable=True, dtype=DType.Floating)` stores raw
  numeric values in a temporary binary file. It is intended for storage and
  movement rather than direct tensor computation.
- `Evictable(primary, secondary)` composes two carriers into a memory
  hierarchy. Computation uses promoted primary storage; `evict()` moves values
  to secondary storage and blocks access until `promote()` restores them. Its
  constructor takes exclusive ownership of both supplied carriers.

These four are closed implementations: `Carrier` is the extension interface and
stays open, but `Generic`, `CPU`, `FileBacked`, and `Evictable` reject subclass
creation with a message naming the supported alternative, and each is declared
`@final` on every import path, so a type checker reports the same closure before
the program runs. Each states its
allocation factories, storage normalization, dispatch metadata, and capability
declarations in terms of its exact class — `Evictable` in terms of its exact
instances — so a specialization would inherit claims it cannot honor: a
`Generic` subclass would advertise every plan `Generic` executes while
`Generic.new_like` refused to allocate a result for it.

A new backend is therefore a sibling `Carrier`, normally composed from the
existing ones the way `Evictable` composes a memory hierarchy: it owns the
carriers it delegates storage to, implements `_dispatch_op` for the operations
it defines itself, and returns one of its own adapters for operations delegated
to an owned carrier. Returning the owned carrier's operation directly is not
lowering: outer tensors would reach an implementation that accepts only the
inner carrier, and the outer carrier would own neither result wrapping nor the
autograd boundary. Composition keeps the new backend's exact class in dispatch
metadata, profiler filtering, capability declarations, results, and gradients,
which is what a subclass silently shared.

A composed backend has one further choice. `Carrier` is the *independent* model,
where an exact class declares the plans its own implementation executes. When
what a backend can execute instead depends on the carrier instances it was
handed, it implements `DependentCarrier` and generates its capabilities per
instance, as described under [Backend Capabilities](#backend-capabilities).

### Dtype Descriptors

Dtype descriptors form a small immutable hierarchy rooted at `DType`, and every
registered descriptor is a singleton, so tags are compared with `is`. The
attributes of `DType` are exactly the built-in descriptors. That namespace is
frozen: reassigning or deleting a binding raises, and so does binding any
further descriptor as a class attribute, so class lookup and `DType.from_name`
can never disagree.

Constructing a descriptor is one transaction owned by the dtype model rather
than a sequence an implementation has to order correctly. The most-derived
`__init__` runs first and simply assigns its own fields; the finished descriptor
is then validated, sealed against mutation, and published under the registry
lock. Three properties follow from that single finalization boundary:

- A constructor that raises at any depth registers nothing, so a failed name and
  its structure both stay free for a later descriptor.
- A concurrent lookup never observes a partly initialized descriptor, because
  the entry that makes it reachable is written after it is complete and sealed.
- Claiming is thread-atomic: one lock guards every registry read and write, so a
  name — and, for a block-scaled descriptor, a structure — is checked and
  claimed indivisibly. Two threads constructing the same name therefore produce
  exactly one descriptor, and every descriptor a constructor returns is the
  identity that `DType.from_name` reports.

The hierarchy has three descriptor kinds, plus the legacy opaque disposition:

- `SimpleDType` is one fixed-width scalar encoding rather than a composition
  of subtensors, so a single carrier could store it homogeneously, and each
  reports an exact `bits` width. `DType.Float32`, `DType.Int32`, and
  `DType.Bool` (8 bits) are the concrete simple storage dtypes carriers support
  today; `DType.Float64` is an accumulator-only descriptor used by operation
  plans and is not accepted as carrier storage; the registry also defines the
  simple encodings `Int8`, `E8M0`, `E5M2`, `E4M3`,
  `E3M2`, `E2M3`, and `E2M1`, which are structural only. Being simple describes
  the encoding, not carrier support.
- `DTypeCategory` is an abstract relationship with no bit width. `DType.Any` is
  the root category, and `DType.Floating` and `DType.Integer` enclose the
  matching simple dtypes. A category is not itself a representation.
- `DType.Any` and `DType.Floating` additionally carry the legacy *opaque
  storage* disposition, which is the one way a category is accepted as storage:
  `Generic` accepts exactly those two for Python-object and width-unspecified
  numeric values, and `FileBacked` accepts `Floating` alongside the concrete
  simple dtypes. `DType.Integer` carries no such disposition, so no carrier
  accepts it.
- `CompoundDType` describes a logical value whose physical representation is
  composed from several simple-dtype planes. It is never carrier storage
  itself; its ordered `simple_types` are the per-plane storage dtypes a
  multi-plane implementation would use, one carrier per plane. A compound
  descriptor may also carry an ordered immutable tuple of
  `RepresentationRule` values. The rule sequence is copied into descriptor
  ownership and contributes once to canonical structure and pickle identity;
  an empty sequence is valid. Rules add reusable representation constraints and
  run only after the universal representation checks have succeeded.
  `LevelExtent(level, extent)` is the first reusable rule: an integer extent
  requires every coordinate in adjacent target level `level + 1` to group
  exactly that many coordinates from level `level`, while `Whole` requires one
  target coordinate covering the complete source level. It validates logical
  grouping only, not physical placement or numerical encoding.

Descriptors describe representation; they do not decide what an operation
computes in or returns. That policy is specified in
[`design/SimpleDType-operation-policy.md`](design/SimpleDType-operation-policy.md)
and implemented by the backend-independent planner in
`strideweave.carriers.operation_policy`, which resolves overload selection,
promotion, arithmetic, accumulation kind and accumulator dtype, and result dtype
for the supported simple dtype operands. Floating reduction association order is
backend-defined. The registry declares each operation once, including overloads
such as tensor/tensor versus tensor/weak-scalar `mul` and `pow`, mixed-output
operations such as Float32 predicates returning Bool or index reductions
returning Int32, and the `dtype_operand_positions` that identify which full
operation-call arguments participate in dtype planning. Shape, axis, ordering,
and other execution parameters never enter promotion as accidental weak
scalars. `select` has a Bool/F32/F32 overload, while `clamp` has four exact
tensor/weak-real bound overloads; these role signatures are selected centrally.
Autograd eligibility follows from the result dtype by the same
floating-dtype rule the tensor layer applies to every tensor, so the plan does
not restate it. The policy is a deliberately evolvable starting point rather
than a compatibility promise.

`Generic` executes those plans and is the reference every other backend
conforms to. Native `CPU` resolves the same plans: each operation asks the
planner for its overload, promotion, arithmetic, accumulation, and output dtype
while the GIL is still held, then releases it to run the kernel that plan
selected. No backend carries a promotion table of its own, so Generic and CPU
agree on Float32, Int32, and Bool storage/results by construction rather than by
parallel maintenance.

#### Backend Capabilities

Resolving a plan is not permission to run it. Policy decides what an operation
*must* do; a backend's capabilities record which of those resolved plans it
*can* faithfully do, and a plan matching none of them is refused before any
storage is allocated rather than executed as the nearest shape the backend
implements. Support therefore means faithful execution of an already-resolved
plan — never that the backend chose the promotion.

A carrier reports its backend's capabilities without running a kernel:

```python
carrier = sw.CPU(4, dtype=sw.DType.Int32)
plan = sw.carriers.operation_policy.resolve_operation_plan("relu", sw.DType.Int32)

carrier.operation_capabilities("relu")     # immutable descriptors, stable order
carrier.supports_operation_plan(plan)      # True
carrier.unsupported_plan_reason(plan)      # None when supported
carrier.require_operation_plan(plan)       # the capability, or raises
```

`operation_capabilities(operation_name=None)` returns immutable
`OperationCapability` descriptors — operation, per-operand source dtype and
conversion target, compute arithmetic, accumulation kind and accumulator dtype,
and output dtype — in a
deterministic order. The four queries answer from whichever set owns that
carrier: an independent carrier's exact class declarations, or the frozen
snapshot a dependent carrier generated for that instance. A caller asks the
carrier and never has to know which, and a new backend of either kind gets the
right answer without overriding these methods. The query describes the carrier's
capabilities rather than its own storage dtype, so a `Float32` CPU carrier still
reports the `Int32` plans CPU executes. An operation the backend has nothing for
yields an empty tuple and a reason naming the operation; a plan whose shape is
unsupported yields a reason describing the shape; `require_operation_plan` raises
`UnsupportedOperationPlan` in both cases. These are the same entries execution
is accepted against, so an advertised plan is executable and an executed plan
was advertised.

Capabilities belong to an exact carrier class and are never resolved through its
bases: a class that declares nothing supports nothing. Every shipped independent
carrier — `Generic`, `CPU`, and `FileBacked` — declares its entries once during
carrier-package initialization and is sealed afterwards, through a
framework-internal path that is exported nowhere. `FileBacked` declares the
empty set: it plans no operation of its own, and declaring that makes it a
stated fact rather than an unclaimed class. `Evictable` declares nothing at all,
because what a hierarchy executes depends on the tiers it was handed; it
generates its capabilities per instance, as described below. Registration
accepts only an independent `Carrier`
implementation — never `object`, the `Carrier` root, an unrelated class, a
`DependentCarrier` class, or a sealed built-in — so no call can widen what a
shipped carrier claims to execute.

A class declares once, and its answer is fixed from then on. The complete set is
published and sealed in the same call, so no observer sees part of a declaration
and no second call can add to one — including after an empty declaration, which
is itself a complete statement. Observation seals as well: querying or requiring
a plan of an independent class that has not declared seals its empty set on the
spot, so first observation is the final answer rather than a provisional one.
That is what lets one carrier snapshot another class's reach without the
snapshot going stale; the practical rule is to declare a custom carrier's
capabilities in its own module, at import time, before anything can ask.

A custom carrier declares its own capabilities against its exact class, which is
what that class executes and nothing else:

```python
from strideweave.carriers.operation_capability import (
    OperationCapability,
    register_operation_capabilities,
)

class MyBackend(sw.Carrier):
    def __init__(self, size):
        super().__init__()
        self._inner = sw.CPU(size, dtype=sw.DType.Int32)

    def _dispatch_op(self, operation_name):
        nested = self._inner.dispatch_op(operation_name)
        # The composite-owned adapter described below.
        return MyBackendOperation(nested)

register_operation_capabilities(MyBackend, [OperationCapability.from_plan(plan)])
```

`MyBackendOperation` is a composite-owned adapter, not a marker wrapper around
the nested operation. Like the complete `EvictableOperation` implementation, it
lowers every outer tensor argument into the representation `nested` accepts,
calls `execute_lowered_operation`, and wraps results and gradients back into
`MyBackend` tensors. Consequently, the capabilities registered above are exactly
the plans that this outer adapter can lower and restore faithfully.

##### Carriers Whose Reach Depends on What They Compose

Declaring for an exact class fits an *independent* carrier: one whose reach is
fixed by the kernels it ships, so every instance of it executes the same plans.
`Carrier` is that default model. A carrier assembled out of other carriers is
not independent — a hierarchy over `CPU` tiers and one over object-storage tiers
execute different plans while sharing a class — so `sw.DependentCarrier` is the
second extension category, and it moves capability ownership from the class to
the constructed instance:

```python
class Mirror(sw.DependentCarrier):
    def __init__(self, inner):
        super().__init__()
        self._inner = inner
        self._finalize_dependent_capabilities()   # last step of construction

    def _generate_operation_capabilities(self):
        return self._inner.operation_capabilities()
```

`_generate_operation_capabilities` is unimplemented by the base: how an instance
decides what it executes is entirely its own, whether that means enumerating a
composed carrier's capabilities, filtering them by what it can store, or
building entries outright.

Finalization is explicit and cooperative rather than automatic. The base never
calls the generator itself — only the concrete carrier knows when its
dependencies are complete — so construction calls
`_finalize_dependent_capabilities()` as its last step, in the same thread, before
the unfinished instance becomes reachable anywhere else. Finalization is not a
thread-safe initialization protocol: concurrent calls and leaking `self` during
construction are outside the supported extension contract. Within that
construction step, the call materializes the generated iterable once, rejects
anything that is not an `OperationCapability` and any exact shape generated
twice, and freezes a deterministically ordered immutable snapshot. Before it,
the instance answers no capability query, which keeps a half-built hierarchy
from advertising an empty set a caller then trusts; a later call is an error, so
a snapshot is never quietly replaced; and a generation that fails publishes
nothing rather than a partial set. Capabilities are frozen against the
dependencies the instance ended up with, so two instances of one dependent class
may answer differently, and a dependent class cannot declare class-level
capabilities at all — it knows nothing about the carriers its instances will be
handed. Until an instance finalizes, its four capability queries raise rather
than answering. Unlike the shipped concrete carriers, `DependentCarrier` and the
dependent carriers built on it stay open to subclassing — but not to overriding
those four queries, which `Carrier` owns so that introspection and enforcement
cannot disagree.

`Evictable` is the shipped example. At the end of construction it takes the
plans its own promoted primary executes — asked of the carrier, so a primary
that is itself dependent composes through its snapshot — and keeps exactly those
whose output dtype *both* tiers can store, because a result only the primary
could hold is a value the hierarchy could never evict. It rewrites no plan,
chooses no promotion, and keeps no operation table of its own. A hierarchy over
`Int32` CPU storage therefore advertises and executes `relu(Int32)`, while a
hierarchy whose secondary tier can only store `Int32` does not advertise
`exp(Int32) -> Float32` and refuses it once, before lowering or any kernel work,
rather than computing a result it could not keep. The snapshot is structural, so
eviction, promotion, and release do not change it, and results and gradients are
hierarchies that generate their own.

Like the policy itself, the capability surface is evolvable rather than a
compatibility promise.

One consequence is worth calling out for `pow`: an exponent preserves an
`Int32` result only when it is a weak *integer* in `[0, 2**31 - 1]`. A float
exponent takes the floating path even when its value is integral, so
`int_tensor ** 2` stays `Int32` while `int_tensor ** 2.0` is `Float32`. On the
integer path the exponent is used exactly rather than carried through a float,
so an exponent above `2**24` keeps its parity — `(-1) ** (2**24 + 1)` is `-1`,
where a float-carried exponent rounded to an even one and returned `1`.

### Generic Reference Semantics

On concrete storage `Generic` implements the encodings faithfully rather than
approximating them with Python's own numeric types:

- `Float32` storage holds binary32-exact values, and arithmetic rounds to
  binary32 at every step. IEEE singularities are results rather than
  exceptions, in forward and backward alike: dividing by zero yields `±inf`,
  `0.0 / 0.0` and `0.0 ** -1` follow IEEE-754, and an overflowing magnitude
  saturates to an infinity instead of raising.
- Floating `reduce_sum` and `matmul` advertise both the default `Float32`
  accumulator and an explicit `Float64` accumulator. Widening happens only
  after loading already encoded `Float32` terms; products and stored outputs
  remain `Float32`. A matmul backward contraction reuses that call's retained
  accumulator choice without treating the option as a tensor input.
- `Int32` storage holds in-range integers. Arithmetic is exact and narrowing is
  checked, so an out-of-range result raises `OverflowError` rather than
  wrapping. A reduction accumulates exactly and checks only the final sum, so a
  partial sum may legitimately leave `Int32` range. `Int32` tensors are not
  differentiable.
- `Bool` storage contains only `False` and `True`. Comparisons and
  `logical_not` produce Bool tensors, which are non-differentiable; Bool is not
  implicitly promoted into Float32 or Int32 arithmetic. Generic normalizes
  stored values to Python `bool`, while CPU uses one-byte `0`/`1` storage.
- Concrete storage is normalized and owned. Values are converted when stored,
  including through `carrier[i] = value` and `scatter`, and the carrier copies
  the supplied sequence, so no caller-held alias can place an unrepresentable
  value or change stored values without the version counter observing it.
  Legacy `Any` and `Floating` storage keeps its documented aliasing behavior.
- NumPy supplies the binary32 mechanics and is imported lazily on first
  concrete `Float32` use, so importing StrideWeave, or using only `CPU`,
  `Int32`, or the legacy dtypes, never loads it.

The legacy dtypes are outside this policy. An operation whose operands mix
legacy `Any`/`Floating` storage with concrete storage stays on Generic's
historical Python arithmetic rather than silently selecting a concrete plan,
which means the concrete operand's binary32 semantics are downgraded to
binary64 for that operation. Legacy `Any` values are never routed through
checked integer arithmetic.

`DType` and `CompoundDType` are abstract at runtime, as is any subclass that
does not declare `abstract=False`, so a class that describes no representation
of its own cannot produce descriptors that claim registry names. The kinds are
also closed: every
descriptor is a `DTypeCategory`, a `SimpleDType`, or a `CompoundDType`, and a
subclass of the root that is none of them is rejected.

A concrete compound representation is added by subclassing, using public APIs
only. The subclass declares `abstract=False`, hands its ordered planes to
`super().__init__`, and keeps whatever fields it needs:

```python
class Planar(sw.CompoundDType, abstract=False):
    __slots__ = ("_label",)

    def __init__(self, name, *, planes, label):
        super().__init__(name, supertype=sw.DType.Any, simple_types=planes)
        self._label = label

pair = Planar("Planar32", planes=(sw.DType.Float32, sw.DType.Int32), label="pair")
assert sw.CompoundDType.from_name("Planar32") is pair
assert pair.num_carriers == 2
```

`CompoundDType` copies the planes into a tuple it owns and validates that copy,
so a mapping that is empty, not iterable, or not all `SimpleDType` descriptors
never reaches the registry, and mutating the collection the caller passed in
changes nothing afterwards.

More generally, a descriptor's stored fields and the accessors that report its
representation — `name`, `value`, `supertype`, `supertypes()`, `structure()`,
`is_subtype_of()`, the kind predicates, `bits`, `simple_types`, `num_carriers`,
`element`, `levels`, `num_axes`, and `bits_per_element` — are owned by the
model, and a subclass that redefines one of them, whether as a slot, a property,
or a method, is refused when the class is created. The attribute machinery
itself — `__getattribute__`, `__getattr__`, `__setattr__`, and `__delattr__` —
is owned on the same terms, because intercepting attribute access decides what
every accessor answers. Each would otherwise let a
registered descriptor report a representation that disagrees with the structure
recorded for its identity and its pickle. A subclass extends the model by adding
its own fields and, when its representation carries state of its own, by
overriding `structure_extension()`.

Ownership is layered by contract class — the root descriptor, categories,
simple, compound, and block-scaled dtypes each own what they define, and every
class below inherits that — and the layers live in the dtype module rather than
on the classes themselves. Assigning to or deleting a class attribute therefore
cannot weaken the policy: the same rule applies to a later `setattr` on a
descriptor class as to its class body. An implementation's own declared slots
stay protected against a further subclass on top of that.

Those layers are part of a wider rule: a descriptor's identity is composed by
the model, not reported by the descriptor. Each contract class has one immutable
specification held in the dtype module, carrying the members it owns, the
fragment it contributes to a canonical structure, the validation it requires,
and whether its descriptors are additionally unique by structure. Finalizing a
descriptor resolves the specifications its class inherits through the method
resolution order, general to specific, and applies them in that order — so a
`CompoundDType` implementation is bound by the root and compound contracts, and
a `BlockScaledDType` subclass by those two plus the block-scaled one. Nothing
about identity is dispatched through the descriptor, so an implementation cannot
omit a canonical layer from its fingerprint, decline the uniqueness its contract
imposes, or weaken a check it still claims to satisfy. The names the model once
dispatched through — `_contract_structure`, `_structural_key`,
`_structure_conflict`, and `_validate_finalized` — are reserved on the same
terms as the accessors, so defining one is refused at class creation rather than
silently ignored. Contracts are not registrable: an external representation
inherits a recognized one and is first-class through registration, subtype and
kind queries, discovery, copying, pickling, and `structure_extension()`.

The check spans the whole hierarchy a descriptor class is built from, not only
its class body: a mixin supplying an owned accessor or an attribute-interception
method is refused at class creation, whichever side of the contract class it is
listed on, and the error names the member and the base it came from. Mixins that
add behavior of their own remain ordinary bases. Only the hierarchy as defined
is checked — a class and its bases are trusted to stay what they were when the
descriptor class was created.

The rule covers a descriptor's own state as well as its class. Declaring
`__slots__` is optional, so an implementation may keep its fields in an instance
dictionary — but an entry named like an owned member, such as a `structure` or
an `is_compound` assigned in `__init__`, would take precedence over the
inherited accessor, so finalization rejects the descriptor before it is sealed
or registered. The check runs again after finalization has consulted
`structure_extension()`, the one hook that runs after the constructor and can
assign to `self` just as `__init__` does, so an accidental shadow introduced
there is caught before anything is claimed. Its name and structural key stay
free and the rejected object keeps neither structure nor seal.

Together these rules describe guardrails, not a sandbox. What the model actually
does is bounded and worth stating exactly. It validates a descriptor class's
initial hierarchy when the class is created, validates the completed descriptor
during finalization — as its constructor leaves it and again after the hooks
that describe it — seals every registered instance against mutation, and keeps
the built-in `DType` namespace frozen. Within that, both slotted and
dictionary-backed implementations are supported, an implementation's own fields
stay its own, and a registered descriptor's model-owned accessors agree with the
structure recorded for its identity, which is what pickling and structural
uniqueness use.

The rest is contract rather than enforcement. From the moment a descriptor
class is created, that extension class and every base contributing behavior to
it must stay as they were: Python classes remain mutable, and the model checks a
hierarchy when it is defined instead of watching it afterwards.
Representation-bearing state must be initialized before registration and
described by a pure, stable
`structure_extension()`, read exactly once during finalization — a field an
implementation keeps changing afterwards alters nothing about the registered
identity, and an object an extension field merely refers to is not deep-frozen.

The following are therefore unsupported rather than intercepted, and code that
uses one forfeits the registry, structural-identity, pickle, and immutability
guarantees above: a custom `__dict__` or other concealment of a descriptor's own
state, reassigning `__slots__`, mutating an extension class or a participating
mixin after the descriptor class is created, `object.__setattr__`,
`type.__setattr__`, and reaching into the dtype module's private state. The
dtype model is an API-integrity boundary against mistakes and drift in
cooperative code, not a defense against hostile metaprogramming sharing the
process.

Field assignment inside `__init__` works because descriptor state is open
exactly until finalization seals it; every published descriptor is immutable. An
implementation that omits `super().__init__` is rejected rather than published
without a name or without planes, and a construction that fails leaves the
object inert: it keeps no structure and no seal, so it cannot later be used as
another descriptor's supertype or plane.

Descriptors expose `name`, `supertype`, `supertypes()`, `is_simple()`,
`is_category()`, `is_compound()`, `is_opaque_storage()`, and
`is_subtype_of(other)`. The kind predicates classify the *representation*, not
backend availability: `is_simple()` reports that a dtype is one fixed-width
scalar encoding, which stays true for an encoding no carrier accepts. There is
deliberately no global "is this storable" predicate, because whether a dtype can
be stored is a decision of an exact carrier class together with a dtype, not a
property of the descriptor. `E4M3` is a perfectly well-formed simple dtype that
no carrier accepts today, and `is_opaque_storage()` reports a descriptor's
legacy disposition rather than permission from any carrier.

`name` is the canonical descriptor field. `value` is a read-only compatibility
alias returning the same string, kept because `DType` was previously an `Enum`
whose members exposed `value`; code such as `tensor.dtype().value` therefore
still works and can never disagree with `name`.

`DType.registered()` and `DType.from_name(name)` query the registry and narrow
to the receiving class in both value and type, so `SimpleDType.registered()`
returns only simple dtypes and `SimpleDType.from_name("E4M3").bits` type-checks
without a cast. Constructing a `SimpleDType` or `DTypeCategory` registers it
under a unique name and extends the hierarchy.

Registration deliberately does not install a `DType` class attribute, which
would let extensions collide with the built-in namespace and leave the
attribute surface untypable. Extensions are therefore discovered through the
registry rather than by attribute lookup:

```python
extension = sw.SimpleDType("Float16", bits=16, supertype=sw.DType.Floating)
assert sw.DType.from_name("Float16") is extension  # registry lookup works
assert extension in sw.SimpleDType.registered()    # enumeration includes it
assert not hasattr(sw.DType, "Float16")            # the namespace is unchanged
sw.DType.Float16 = extension                       # AttributeError: the namespace is frozen
```

Copying and pickling preserve identity rather than producing a duplicate
descriptor. A pickle carries the descriptor's name and its structure, which the
*receiving* process resolves against its own registry; it never ships a dtype
definition. A built-in therefore unpickles in any process that imports
`strideweave`, while a dynamically registered dtype requires that process to
register a matching descriptor first:

```python
payload = pickle.dumps(sw.SimpleDType("Float16", bits=16, supertype=sw.DType.Floating))

# In the receiving process, registration must happen before the load.
recreated = sw.SimpleDType("Float16", bits=16, supertype=sw.DType.Floating)
assert pickle.loads(payload) is recreated
```

Without that registration the load raises `LookupError`, and if the receiving
process registered `Float16` with a different width or category the load raises
`ValueError` rather than silently substituting a different representation.

That check covers the whole referenced graph, not a list of names. A structure
records each contract class's own fields and expands every descriptor it names —
supertype, compound plane, block element, scale — into that descriptor's
structure recursively, so a receiver whose `E4M3` is 16 bits, whose `Floating`
is not opaque, or whose compound planes differ is rejected even though every
name matches. The error names the field that actually differs. `structure()`
returns the recorded value, which is computed once at finalization and is the
authority for structural identity.

A descriptor may reference only descriptors that are *canonically registered* —
finalized and still the descriptor their name resolves to. That keeps the graph
acyclic and the expansion finite, and it keeps an object whose own registration
was rejected from becoming the supertype or plane of a descriptor that does
register.

Every leaf of a structure is stored as a string naming its exact type alongside
its value, so structures never compare through ordinary Python equality:
`True`, `1`, and `1.0` are three different representations rather than one, and
a `NaN` tag gets a single deterministic spelling instead of a value that is not
equal to itself. Floats are recorded in exact hexadecimal form, so a tag
round-trips through a pickle without precision loss.

An implementation whose representation carries state beyond its contract
declares it by overriding `structure_extension()` — the only part of a
descriptor's identity an implementation controls — which returns exact strings,
numbers, `None`, `Whole`, and tuples of those. Two descriptors that would
otherwise be structurally identical are then distinguished for both pickle
compatibility and structural uniqueness:

```python
class Tagged(sw.CompoundDType, abstract=False):
    __slots__ = ("_tag",)

    def __init__(self, name, *, tag):
        super().__init__(
            name,
            supertype=sw.DType.Any,
            simple_types=(sw.DType.Float32,),
        )
        self._tag = tag

    def structure_extension(self):
        return (self._tag,)
```

Shipping the dtype definition itself — a cross-process descriptor schema — is
deliberately out of scope and remains possible future work.

The narrow simple encodings `Int8`, `E8M0`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, and
`E2M1` are registered so the block-scaled formats below can name their elements
and scales. They are structural descriptors only: no carrier stores them and no
kernel interprets them yet, so `SimpleDType.registered()` deliberately returns
more encodings than any carrier accepts.

### Block-Scaled Descriptors

`BlockScaledDType` is the currently implemented compound dtype: one simple
element dtype plus a linear chain of simple scale `Level` entries. A level's
`block` extent is measured in the *previous* level's coordinate space, so each
level coarsens the grouping below it, and only the final level may use the
symbolic `Whole` extent that produces a single scale for the entire tensor.
`Whole` is a true singleton — `WholeExtent()`, copying, and unpickling all yield
it — so equivalent whole-scaled formats cannot bypass structural uniqueness.

- `simple_types` maps position to plane dtype, so plane `i` is stored by a
  carrier whose dtype is `simple_types[i]`; `num_carriers` is its length.
- `num_axes` counts the blocking axes a tensor of this dtype must be given,
  which excludes `Whole` levels.
- `bits_per_element` returns a `float`, or a `SymbolicBits` value with an
  `evaluate(element_count)` method when the final level is `Whole`.
- `representation_rules` contains one `LevelExtent(i, levels[i].block)` per
  scale level. The rules are derived from the dtype's existing level chain and
  use the same generic representation validator available to external compound
  formats; Tensor validation contains no block-scaled-specific branch.

The registered formats are `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP6_E3M2`,
`MXFP6_E2M3`, `MXFP4`, `MXINT8`, and `NVFP4`. Block extents are fixed by each
format, so no public signature other than the structural `Level` definition
accepts one. Descriptors are unique by structure as well as by name: two
descriptors describe the same representation only when they are the same
object. That uniqueness is anchored at `BlockScaledDType` itself, so a subclass
that adds no structure of its own describes an already registered
representation and is rejected instead of becoming a second identity for it.

These descriptors are structural. Block-scaled tensors, tilers, quantization,
requantization, and dispatch eligibility are not implemented.

The runtime implementation is organized behind the stable
`strideweave.carriers.dtype` facade. Private modules separate canonical
structure encoding, contract ownership, the descriptor and registry model,
block-scaled definitions, built-in installation, and carrier-facing storage
validation. This organization is internal: public import paths, class module
identities, structure fingerprints, and pickle resolution continue to use
`strideweave.carriers.dtype`.

### Carrier Storage Dtypes

A carrier holds elements of exactly one dtype, so each carrier accepts a fixed
set of descriptors:

| Carrier | Accepted storage dtypes |
| --- | --- |
| `Generic` | `Any`, `Floating` (legacy opaque storage), `Float32`, `Int32`, `Bool` |
| `CPU` | `Float32`, `Int32`, `Bool` |
| `FileBacked` | `Floating`, `Float32`, `Int32` |
| `Evictable` | Whatever both composed tiers accept, which must match |

Each table row is the exact accepted set: membership is checked by object
identity — never by equality, including in the native CPU parser, so an object
that merely compares equal to `DType.Float32` is rejected like any other
unsupported dtype — and every registered descriptor outside a carrier's row is
rejected at construction.
`Any` and `Floating` are accepted only where the table lists them, as legacy
opaque storage; the category `Integer` is accepted nowhere; the narrow simple
encodings have no carrier support yet; and a compound descriptor is rejected
with an error naming the deferred capability rather than partially constructing
a carrier that could hold only one of its planes. Every carrier gives that same diagnostic, including
the native CPU parser. Multi-plane storage — one carrier per entry of
`simple_types` — is future work; carrier-authoring code can reuse
`strideweave.carriers.dtype.validate_storage_dtype` to get the same behavior.
`Float64` is likewise outside every row: it currently names widened accumulator
arithmetic only, not storage any carrier may allocate.

The same table is readable at run time, without allocating anything:

```python
sw.CPU(4, dtype=sw.DType.Float32).supports_storage_dtype(sw.DType.Int32)   # True
sw.CPU(4).supports_storage_dtype(sw.DType.Integer)                         # False
```

`supports_storage_dtype(dtype)` asks whether the carrier's *implementation* can
allocate that dtype at all. It is structural rather than a report of state: it
allocates nothing, changes nothing, and is unaffected by size, mutability,
ownership, eviction residency, release, or which dtype the carrier currently
holds, so a `Float32` CPU carrier still reports `Int32`. A dtype outside the
carrier's row — an abstract category, an unimplemented narrow encoding, or a
compound descriptor whose per-plane storage is deferred — is unsupported rather
than an error; only a non-`DType` argument raises. `Evictable` reports the
intersection of its tiers, because a value it cannot evict is a value it cannot
hold. This is what lets a composed carrier decide, before any work begins,
whether it could store an operation's result.

`Carrier` owns the public query and its validation; a carrier implementation
states its accepted set through the protected `_supports_storage_dtype(dtype)`
hook, exactly as `_is_mutable()` works. The conservative default claims only
the dtype the instance currently holds, which is the most any carrier can be
assumed to allocate without saying so, so a custom carrier that can allocate
more overrides the hook.

Only `Floating` and `Float32` tensors participate in autograd. That set is an
explicit pair rather than a `Floating` category query, because the narrow
floating encodings have no numerical semantics yet.

Every carrier exposes `allocate_like(size, *, mutable=True, dtype=None,
empty=False)` for fresh size-based allocation. The default requests initialized
storage; `empty=True` permits a backend to skip initialization, so callers must
write every element they will read. `new_like(values, ...)` remains the separate
factory for materializing supplied values.

Carriers may be mutable or immutable. Mutating shared storage increments a version
counter visible through `tensor.version`. Calling `release()` permanently
releases a carrier's storage. Eviction and promotion belong specifically to
the composite `Evictable` carrier rather than the base `Carrier` or `Tensor` APIs.
A carrier owned by a composite carrier remains readable through retained aliases
while that tier is live, but rejects direct mutation, scatter, release, and
move operations. A tier may be released and replaced during a hierarchy
transition, after which an alias to the old tier is no longer readable. The
owning carrier retains privileged access so mutation through the composite
interface continues to follow its normal mutability and version rules.

`is_mutable()` reports whether public interfaces may currently write the carrier,
not only whether its storage was constructed mutable. Consequently, an
owned child reports `False` while its mutable owning composite may report
`True`. Carrier implementations define their intrinsic storage capability
through the private `_is_mutable()` hook; ownership is applied centrally by
`Carrier`.

```python
import strideweave as sw

primary = sw.Generic([1.0])
carrier = sw.Evictable(primary, sw.Generic([0.0]))

assert carrier.is_mutable()
assert primary.is_owned()
assert not primary.is_mutable()

carrier[0] = 2.0

try:
    primary[0] = 3.0
except RuntimeError:
    pass
```

## Operations

The public functional API includes the following v0 surface:

- views and layout composition: slice indexing, `as_strided`, `reshape`,
  `permute`, `rearrange`, `broadcast_to`, `broadcast_in_dim`, `squeeze`, and
  `unsqueeze`. These are zero-copy layout operations; `flip` and materializing
  `contiguous`/`copy` remain deferred.
- unary elementwise operations: `neg`, `abs`, `sign`, `recip`, `sqrt`, `rsqrt`,
  `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `erf`, `floor`, `ceil`, and
  `round`, plus the composite activations `relu`, `sigmoid`, `tanh`, `gelu`,
  `silu`, `softplus`, `elu`, and `leaky_relu`.
- binary elementwise operations: `add`, `sub`, `mul`, `elementwise_mul`, `div`,
  `pow`, `maximum`, `minimum`, and `rem`. `mul` and `pow` also accept the
  documented weak-scalar overloads.
- predicates: `eq`, `ne`, `lt`, and `le` return non-differentiable Bool
  tensors. `gt` and `ge` are argument-swapping composites, and `logical_not`
  returns Bool.
- masked selection: `select` consumes a Bool condition and two Float32 value
  tensors; `clamp` consumes a Float32 tensor with Float32 tensor or weak-real
  bounds. Both use the shared structural broadcasting rule.
- reductions and scans: `reduce_sum`, `reduce_prod`, `reduce_max`,
  `reduce_min`, `argmax`, `argmin`, and inclusive `cumsum`. The former
  `sw.reduce` name is removed rather than retained as a compatibility alias.
- contractions and indexing: two-mode `matmul`, layout-described `einsum`,
  Float32 grouped `conv_general`, Float32/Int32 `gather`, and functional
  `scatter` and `scatter_add`.
- selection: `sort` and `topk` return named `(values, indices)` results with
  Float32 values and Int32 indices.
- storage movement: `move`.

`reduce_sum(..., accumulator_dtype=...)` and
`matmul(..., accumulator_dtype=...)` select the floating accumulator without
changing input encoding or planned output dtype. `None` uses the backend's
default `Float32` accumulator; `DType.Float64` requests widened accumulation and
must match an advertised backend capability. `Generic` and native `CPU` both
advertise and execute the two choices for `reduce_sum` and `matmul`; CPU widens
already encoded Float32 terms into a native `double` accumulator without adding
Float64 carrier storage, and matmul backward reuses the accumulator its forward
call selected. The `tensor @ other` spelling keeps the default; callers
requesting `Float64` use `sw.matmul`. Exact-integer plans reject a floating accumulator request, and the
operations whose accumulation order is normative — `cumsum`, `conv_general`,
`scatter_add`, the product and extrema reductions, and the arg reductions —
accept no accumulator option at all.

Binary pointwise operations (`add`, `sub`, `mul`, `elementwise_mul`, `div`,
`pow`, `maximum`, `minimum`, `rem`, and the Float32 predicates) align
operands structurally before dispatch. Their shape trees must share the same
`Layout.profile`; corresponding leaf extents must be equal or one must be 1.
Extent-one leaves widen with stride zero at that exact hierarchical position.
There is no NumPy-style rank alignment, flattening, insertion, removal, or
reordering. Differing strides are accepted when shapes already match. Results
use an injective canonical layout over the common logical shape rather than
inheriting broadcast strides. Each widening is a differentiable zero-copy
`broadcast_to` view saved as the pointwise operation's input. Its backward pass
sums the cotangent over every widened leaf and restores the pre-broadcast
layout, including when the leaf is nested in a hierarchical mode.

Broadcast operands also have defined semantics outside the pointwise class.
Reducing a stride-zero mode of extent N sums N equal logical reads, so it scales
the stored value by N. Matmul treats stride-zero kept modes as repeated output
rows or columns and stride-zero contracted modes as repeated factors in the dot
product. Both operations compute backward values in injective logical storage;
the broadcast node, or broadcast-leaf accumulation, then sums contributions
back to the underlying storage. Generic and CPU follow the same rule.

`broadcast_in_dim` is a composite convenience wrapper. It accepts an explicit
`broadcast_dimensions` sequence naming the target top-level positions occupied
by the source modes, inserts extent-one modes with `unsqueeze` for omitted
positions, and finishes with `broadcast_to`. It does not infer rank alignment,
reorder modes, or insert inside nested modes; callers needing those layouts use
`rearrange` explicitly first.

`select(condition, on_true, on_false)` aligns all three tensors simultaneously:
the condition is Bool, both value branches are Float32, and singleton leaves
widen through differentiable stride-zero views. Pairwise alignment order cannot
change the common shape. The forward read is masked: only the selected branch is
read at each coordinate, so an NaN in an unselected branch does not propagate.
The Bool condition is non-differentiable; backward routes the cotangent to the
selected branch and zero to the other, then reduces any broadcast views.

`clamp(tensor, lower, upper)` accepts Float32 tensor bounds or weak real scalar
bounds. Tensor operands use the same simultaneous structural broadcasting, and
weak scalars are converted to Float32 by the central policy. Its forward and
backward semantics are exactly the ordered composition
`minimum(maximum(tensor, lower), upper)`, including NaNs, signed zero,
infinities, equal-winner splits, and `lower > upper`; no independent bound-order
validation is performed. Tensor bounds receive staged VJPs, while weak scalar
bounds do not receive gradients.

Reductions lower a hierarchical description to a two-mode intermediate whose
second mode is the reduction fiber. `reduce_sum`, `reduce_prod`, `reduce_max`,
and `reduce_min` return Float32 results; `argmax` and `argmin` return Int32 first-
winner indices. `cumsum` is an inclusive scan over one explicit top-level mode.
Fibers must be non-empty, and `keepdims` is composed with `unsqueeze` rather
than a reduction option.

`conv_general` is a grouped Float32 cross-correlation over arbitrary spatial
rank. It supports explicit positive strides, non-negative padding, input and
kernel dilation, feature groups, and top-level dimension-role permutations;
the kernel is not reversed. `gather` replaces one top-level mode using Int32
indices. `scatter` requires distinct indices, while `scatter_add` accumulates
repeated indices in logical order. Both return fresh Float32 tensors.

`sort` and `topk` operate on one top-level mode and return named tuples with
Float32 values and Int32 source indices. Their ordering and tie behavior are
defined by the Generic reference and matched by CPU; the internal value/index
dispatch names are not public operations.

`Generic` provides Python reference implementations. `CPU` provides native C++
kernels that use cached expanded layout keys and release the GIL in hot loops.
`FileBacked` does not dispatch computational operations.

An Evictable tensor dispatches through a public `EvictableOperation` adapter.
Each adapter owns one fresh operation from the primary carrier, lowers its
inputs to temporary primary-backed tensors, and invokes the primary operation's
generic lowered-execution route and `backward` method. Lowered execution shares
framework execution hooks and result validation with regular execution but does
not attach an inner autograd node or discard delegated state. The adapter is the
sole visible autograd node and wraps primary results and gradients back into the
same hierarchy. CPU and Generic implementations therefore do not need
composition-specific code. Before it lowers anything, the adapter resolves the
logical plan the central policy gives for its operation name and its *outer*
operands plus typed execution options and requires that plan against the
hierarchy's own frozen
capabilities, so no plan the hierarchy does not advertise reaches a nested
allocation or a kernel. Only an operation name the policy does not register and
legacy opaque operand storage skip planning and keep their documented behavior;
a registered operation handed operands its shape does not accept is refused by
the resolver at that gate rather than further down.
New operation results allocate only their promoted primary storage. Their
secondary tier remains empty until the first eviction provisions it.

StrideWeave layout descriptions preserve hierarchical modes and therefore do not
have standard flat einops semantics. String forms include:

```python
transposed = sw.rearrange(tensor, "a b -> b a")
summed = sw.reduce_sum(tensor, "a (b c) -> a b")
contracted = sw.einsum(lhs, rhs, "a b, c b -> a c")
batched = sw.einsum(lhs, rhs, "b i k, b j k -> b i j")
```

The native lexer and Python parsers compile these descriptions into layout
trees and cache successful specifications. Einsum classifies each shared symbol
by its output presence: an omitted shared symbol is contracted, while a retained
shared symbol is a batch dimension. One-sided symbols are free dimensions and
must appear in the output. A contraction without batch symbols keeps the
two-mode matmul lowering. Batched contractions align both operands over their
union symbol space with differentiable singleton broadcasts, multiply
elementwise, and reduce only omitted shared symbols. This general lowering
materializes the union-shaped product; it does not currently use a native
batched-matmul kernel.

## Operation Profiling

`profile` is a single-use context manager that records carrier-dispatched
operation executions on the current thread. It records execution attempts, not
dispatch factory lookups. Events contain the canonical operation name, exact
dispatching carrier and implementation classes, monotonic start time, inclusive
and self synchronous host wall time, thread identity, parent relationship, and
success status. With `record_shapes=True`, tensor argument positions also carry
immutable snapshots of their hierarchical shapes; non-tensor positions are
represented by `None`.
Typed execution options travel through the profiled call separately and are not
reported as positional inputs or saved as autograd tensors.

```python
import strideweave as sw

tensor = sw.Tensor(
    sw.CPU(6),
    0,
    sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2])),
)

with sw.profile(carriers={sw.CPU}, record_shapes=True) as prof:
    result = sw.relu(tensor)

events = prof.events()
averages = prof.key_averages(group_by_input_shape=True)
print(prof.table(sort_by="self_total_time_ns"))
```

`carriers=None` records every exact carrier class; an iterable such as
`{CPU, Evictable}` selects only those exact classes and does not retain carrier
instances. Nested composite execution is visible when selected: an Evictable
operation produces an outer Evictable event and a nested event for its promoted
CPU or Generic operation. Filtering out that nested event does not charge its
time to the parent's self time. Aggregates are derived from the immutable raw
events by operation name and carrier class, optionally adding input shapes to
the grouping key.

Profiling state is thread-local, so work on another thread requires its own
context, and a context must exit on the thread that entered it. A rejected
cross-thread exit abandons that registration so the owner thread recovers
before its next dispatched operation or profiler context. Timings measure
synchronous host wall time only; asynchronous
accelerator activity is not modeled. Directly instantiated operations without
carrier dispatch metadata and unannotated registry move operations are excluded.
Results become available after the context exits, including when its body raises;
the original exception still propagates.

## Autograd

Operations attach an autograd context when gradient construction is enabled,
the result is differentiable, and at least one tensor input is differentiable.
Backward traversal is iterative and topological, so shared subgraphs accumulate
their pending gradients before their operation runs.

- `backward(gradient=None, retain_graph=False)` releases the saved inputs,
  versions, and operation context for every reached operation after a successful
  traversal. Calling backward through that graph again raises an error naming
  `retain_graph=True`; pass that flag on an earlier traversal when the same
  graph must be reused.
- Graph nodes remain attached after their saved state is released so a second
  traversal fails explicitly rather than treating a former result as a leaf.
  A shared subgraph is released by whichever reachable root traverses it first.
- Non-scalar tensors require an explicit gradient in `backward(gradient)`.
- An exact shape `[1]` is a scalar and may call `backward()` with an implicit
  gradient of one.
- Leaf tensors accumulate `.grad` by default, including across distinct
  forward graphs and retained repeated traversals.
- Non-leaf tensors retain `.grad` only after `retain_grad()`.
- `no_grad()`, `is_grad_enabled()`, and `set_grad_enabled()` control the
  thread-local graph-building state.
- `Any` and `Int32` tensors reject gradient APIs.
- Backward validates saved input versions and raises if required storage was
  modified in place after the forward pass.

Views are differentiable. Their backward path scatters gradients into a tensor
with the original input layout.

Gradient buffers are always injective. When a leaf tensor uses stride-zero
broadcast modes, `.grad` sums all logical contributions that address the same
input storage slot and represents each resulting sum in a canonical injective
layout with the tensor's logical shape. This supports broadcast aliasing at any
hierarchy depth. Other non-injective layouts, such as overlapping non-zero
strides, are refused explicitly in autograd rather than producing an
under-counted gradient.

### Functional gradients

`sw.grad(output, inputs, cotangents, *, batched=False, retain_graph=False)`
computes vector-Jacobian products without reading or modifying any tensor's
`.grad` field. It returns one gradient per requested input in positional order;
an input unreachable from `output` produces `None`. The unbatched form accepts
one cotangent whose layout exactly matches `output.layout`.

With `batched=True`, one tensor represents K cotangents. Its layout has a
prepended leaf batch mode followed by modes exactly equal to `output.layout`;
each batch slice is a zero-copy offset view. Every reachable input produces one
stacked gradient tensor with a prepended batch mode. That mode's stride is the
single-gradient layout's `cosize`, not its logical size, so inputs with storage
holes keep adjacent gradient slices disjoint. The native traversal discovers
the shared operation topology once, propagates each of the K cotangents through
it independently, and releases saved graph state only once after the last pass
unless `retain_graph=True`.

Functional gradients do not build a differentiable backward graph, so there is
no `create_graph` parameter and double backward or Hessian construction is not
supported. Forward-mode JVPs are also not implemented.

## Modules

`Module` provides basic PyTorch-like structure: subclasses implement `forward`,
and `__call__` delegates to it. Assigning public `Parameter` or `Module`
attributes registers them for `modules()`, `parameters()`, and
`get_named_parameters()` traversal. Optional module and parameter names can
override attribute-name segments.

Buffers, state dictionaries, training/evaluation modes, and hooks are not
implemented yet. A minimal layer library and optimizer live in `strideweave.nn`
(see Ergonomic Layers).

## Ergonomic Layers

The core carriers stay composable primitives; user-facing ergonomics live
in two submodule-only packages that are built entirely from the public
primitives and are not re-exported at the top level.

### strideweave.nn

`import strideweave.nn as nn` provides `Linear`, activation module wrappers
(`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softplus`, `ELU`, `LeakyReLU`),
`MSELoss`, and an `SGD` optimizer.

Carrier and layout requirements differ per component and follow from what
each one actually does, rather than a blanket `strideweave.nn` restriction:

- The activation wrappers are thin `Module` adapters that delegate to the
  corresponding functional operation, so they carry no hyperparameters and
  inherit its input support: they accept any carrier and layout the underlying
  op accepts (e.g. a one-mode `Generic` tensor), not just CPU `Float32`.
- `Linear` holds CPU `Float32` parameters and uses matmul plus a differentiable
  stride-zero bias broadcast, so it requires CPU inputs in the flat
  column-major `[batch, features]` convention below.
- `MSELoss` is composed from carrier-dispatched operations (`sub`, `pow`,
  reduction), so it works on any carrier that supports them (CPU or Generic);
  it does require the prediction and target to share a flat two-mode layout.
- `SGD` writes elementwise through each parameter's layout, so it works on any
  mutable parameter with a compatible gradient — no CPU or two-mode
  requirement.

Conventions: `Linear` inputs are flat column-major `[batch, features]` tensors
(`Layout(Shape([rows, cols]), Stride([1, rows]))`) and its weights are
`[out_features, in_features]`; because matmul contracts the second mode of
both operands, `x @ weight` yields `[batch, out_features]` directly. There is
a differentiable broadcasting primitive, so `Linear` permutes its
`[out_features, 1]` bias to `[1, out_features]` and applies
`broadcast_to(..., Shape([batch, out_features]))`. The view shares bias
storage, and its backward pass sums the bias gradient over the batch.

`SGD.step()` mutates parameter storage in place and therefore bumps carrier
versions: the required per-iteration ordering is forward, `backward()`, then
`step()`, and graphs built before a step cannot be backwarded afterwards.
Gradients accumulate until `SGD.zero_grad()` resets them to `None`.

`MSELoss` returns an exact single-mode `Shape(1)` scalar, so
`loss.backward()` needs no explicit gradient.

### strideweave.friendly

`import strideweave.friendly as F` provides compact layout builders
(`column_major`, `row_major`), CPU tensor factories (`tensor` from nested
lists, `zeros`, `ones`, `full`, `arange`, `rand`, `randn`), scalar reductions
(`sum`, `mean`, both returning `Shape(1)` tensors that support implicit
`backward()`), and value extraction (`item`, `to_list`).

End-to-end training examples live in `examples/train_mlp_cpu.py` (raw
primitives) and `examples/train_mlp_cpu_friendly.py` (same model using the
helpers).

## Interoperability And Movement

CPU tensors support DLPack export through `__dlpack__` and
`__dlpack_device__`. Hierarchical shapes and strides are flattened for the
DLPack representation. Generic, FileBacked, and Evictable carriers do not support
DLPack, and copy or cross-device exports are not implemented.

`move(tensor, destination)` dispatches on the exact source and destination carrier
classes. CPU-to-FileBacked and FileBacked-to-CPU moves use native bulk copies;
unregistered pairs use an elementwise fallback. A successful move releases the
source carrier, and autograd moves gradients back into fresh source-class
storage. Moving a broadcast tensor preserves its exact stride-zero layout and
copies its `cosize` physical span rather than materializing `size` logical
elements. Its backward consumes an injective same-shape gradient before the
broadcast node performs the required summation.

Evictable resolves the move registry for each transition and routes move
operations through the framework-owned sealed lowered-execution path, so
residency changes receive shared result validation without adding autograd
nodes. Element access, forward operations, and scatter are unavailable while
values are evicted. Backward may still run while an operation result is evicted because
the result storage is not read; saved inputs and the supplied gradient must be
promoted.

Residency transitions publish replacement tiers only after a move succeeds. If
a move implementation raises, the prior residency state and ownership remain
valid and the transition may be retried.

Ownership guards apply to carrier interfaces. Explicit external-memory
escape hatches such as `CPU.pointer()` and direct writes to a `FileBacked` path
remain the caller's responsibility and cannot participate in version tracking.
The same applies to direct mutation of a mutable container originally supplied
to `Generic`, because the container remains an alias of Generic storage.

## Current Boundaries

- No CUDA, Metal, or other accelerator carriers.
- No carrier stores `Float64`; it is currently an accumulator-only descriptor
  whose execution remains gated by each backend's exact capabilities.
- High-level tensor creation lives only in `strideweave.friendly` and is
  CPU-backed; other carriers are constructed from primitives.
- Explicit structural views are available through `as_strided`, `reshape`,
  `permute`, `broadcast_to`, `broadcast_in_dim`, `squeeze`, and `unsqueeze`;
  slice indexing supplies the corresponding positive-step view. They preserve
  hierarchy, expand only extent-one leaves where applicable, and never infer
  rank alignment. Binary pointwise operations additionally perform implicit
  singleton alignment using differentiable views. Reductions, scans,
  contractions, convolution, indexing, selection, and movement remain
  separate capabilities.
- `as_strided(tensor, shape, stride)` uses `Layout(shape, stride)` as an
  origin-based logical mapping `B` from output coordinates to flattened input
  `c_0` ordinals; unlike PyTorch, its stride is not a direct physical-storage
  reinterpretation. The output placement is the representable composition
  `Layout.compose(L_0, B)`, and multi-subtensor views compose `S_0` with the
  same mapping while preserving deeper layouts. The mapping and composed
  placement must be injective, and `B.cosize` must fit the input logical size.
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
- Evictable tensors must be promoted before access or computation, and binary
  Evictable operations require matching primary and secondary carrier classes.
- Public compound tensor construction and multi-subtensor coordinate indexing, mutation,
  non-view operations, movement, release orchestration, and DLPack export
  remain deferred. The pure c0 layout views (`permute`, `broadcast_to`,
  `slice`, `reshape`, `as_strided`, `squeeze`, and `unsqueeze`) are the narrow
  validated multi-subtensor
  exception; the internal representation and optional-rule contracts exist so
  the remaining paths can be added without a parallel public tensor type.
- `strideweave.nn` covers only `Linear`, elementwise activations, `MSELoss`, and
  `SGD`; there are no buffers, state dictionaries, training/evaluation modes,
  or hooks.

## Local Kernel Verification

`test_backend(output=None)` verifies the installed CPU backend without CI,
network, or database access. It reruns Stage One over the active native kernel
capability plans, including mixed and exact-integer plans and the public CPU
Float64 accumulator capabilities for `reduce_sum` and `matmul`. Coverage is
enumerated from the native kernel manifest, so every registered kernel is either
actively certified or explicitly deferred with a stated reason; a kernel added
in C++ without a classification fails the manifest check rather than passing
silently. A kernel certificate requires every class assigned to every active
plan to pass. Stage Two then runs
ordinary CPU Float32 target execution only where an exact kernel-ID/variant
certificate is reconstructed from Stage One evidence and covers the active
Float64 oracle plan/classes required for that target. Its contraction catalog includes multi-output flat and
hierarchical layouts, recording each operand's effective matrix shape and
contraction length.
Movement verification emits separate bit-exact cases for move, view, permute,
rearrange, and broadcast-to views, each using an adversarial Float32 payload.
The correctly rounded and exact-integer kernels are compared bit for bit against
`Generic` on seeded arbitrary finite encoded inputs, while the operations whose
accumulation order is normative — `cumsum`, `conv_general`, `scatter_add`, and
`reduce_prod` — use payloads whose every legal partial result is exactly
representable. Exactly representable structural cases are likewise bit-exact; numerical Stage
One cases use the analytic, order-independent
`stage-one-two-path-gamma-v1` envelope, while Stage Two uses the versioned,
case-specific `stage-two-float32-gamma-k-v1` envelope. Every evidence case also
records its explicit operation name with its kernel ID and variant. Target and
oracle decode one payload encoded once at its declared
Float32 or Int32 storage dtype, so source quantization error is outside the
comparison.
Floating `pow`, vendor-transcendental kernels, and autograd certification remain
visible v0 deferrals rather than being reported as passes.

```python
report = sw.test_backend()
sw.test_backend("kernel-evidence.jsonl")
```

The returned immutable report contains passed, failed, errored, blocked, and
deferred attempts. Stage Two currently covers movement, reduce, and matmul. A
missing, forged, or incomplete Stage One certificate blocks its dependent Stage Two cases
without executing them. JSONL output is deterministic and versioned; it
intentionally contains no CI status, Dolt state, toolchain hash, transitive
closure hash, autotune cache, or status-aggregation record.
Use `report.write(path)` to save it and `VerificationReport.load(path)` to
strictly reload it. The evidence-only v1 JSONL wire format carries an evidence
schema on each line, while the model accepts only its current report schema
version before any report can be serialized. Loading accepts only the current
evidence schema and canonical JSON encoding, reconstructs the immutable nested
evidence model, and identifies the JSONL line for malformed or invalid evidence.

```python
summary = report.summary()
assert summary.gate_passed
print(report.describe())
failed_or_blocked = report.problems
stage_one_records = report.select(stage=sw.verification.VerificationStage.ORACLE)
```

Reports keep their complete immutable `records` tuple for detailed inspection,
while their REPL representation and `describe()` output remain bounded. The
summary counts every outcome, pipeline stage, and verification class; its
`passed`, `failed`, `errors`, `blocked`, and `deferred` accessors expose the
common immutable totals without replacing the ordered aggregates. `errors` is
plural because it is a count, while the serialized and CLI outcome spelling
remains `error`. Deferred coverage is counted separately and does not by itself
fail the gate.
`VerificationReport.__doc__` documents the Stage One/Stage Two record model and
its navigation APIs. In a REPL, use `repr(report)` for compact outcome counts,
`report.summary()` for immutable structured counts, `report.describe()` for
plain text, `report.select(...)` for composable filters, and the `passed`,
`deferred`, and `problems` views for common selections.

The inspection-only `strideweave-verify-report` CLI loads the same strict model
parser used by Python; it does not run tests, contact a network, or write status
data. It exits 0 when the selected evidence has no failed, errored, or blocked
record; 1 when the selected correctness gate fails; and 2 for malformed reports
or command usage.

```bash
strideweave-verify-report kernel-evidence.jsonl
strideweave-verify-report kernel-evidence.jsonl --problems --verbose
strideweave-verify-report kernel-evidence.jsonl --stage stage_two --operation matmul
strideweave-verify-report kernel-evidence.jsonl --json --outcome failed
```

`--kernel`, `--variant`, `--class`, and repeatable `--outcome` apply the same
exact filters as `report.select()`. `--verbose` adds flat per-case metadata;
with `--json`, it adds a stable `records` array instead.
Recoverable public `RuntimeError` and `ValueError` execution failures are
recorded case by case, so independent witnesses continue. Their records retain
the prepared operation, kernel, plan, payload hashes, logical shapes,
contraction length, seed, and tolerance; deviation and mismatch measurements
are `null` because no comparison completed.

## Development

The package requires Python 3.12 or newer and builds its native modules with
scikit-build-core and pybind11.

Before designing a change, read the cross-cutting contracts in
[`INVARIANTS.md`](INVARIANTS.md). It records the canonical implementation choices and
whether each invariant is enforced by AST lint, Ruff, behavioral tests, native builds,
or code review.

```bash
uv sync --group dev
uv run pytest tests
uv run ruff format --check .
uv run ruff check .
uv run python tools/lint_invariants.py
uv run pyright
uv build
npm ci
npm run duplication
find src/strideweave -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec uv run clang-format --dry-run --Werror {} +
CMAKE_ARGS="-DSTRIDEWEAVE_STRICT_WARNINGS=ON" uv build
```

The repository invariant checker uses Python's built-in AST and reports
StrideWeave-specific source contracts without importing the package. Native sanitizer
coverage runs in Linux CI with `STRIDEWEAVE_SANITIZERS=ON`; it instruments the extension
modules with AddressSanitizer and UndefinedBehaviorSanitizer before running the full
Python test suite.

The duplication gate uses the exact `jscpd` version locked by npm and the checked-in
`.jscpd.json` configuration to scan production code under `src/`. The post-binary-
operation-refactor baseline was 4.7% duplicated lines with 5-line/50-token minimum
clones; CI blocks results above 5.0%. The scanner respects `.gitignore`, and the
configuration explicitly excludes non-production, generated, dependency, cache,
report, and build artifacts.

The test suite covers layouts, carriers, tensor indexing and mutation,
autograd, operations and activations, hierarchical command parsing, DLPack,
movement, modules, and public docstrings. `tests/test_dtype_conformance.py`
additionally enumerates the operation policy's registry and compares `Generic`
against native `CPU` for every registered operation, so a backend that drifts
from the shared plans fails there rather than in review.

## License

StrideWeave is licensed under the Apache License, Version 2.0. See `LICENSE`.
