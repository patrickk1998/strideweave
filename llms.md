# StrideWeave

StrideWeave is a research tensor and autograd framework built around hierarchical,
CuTe-style layouts. A tensor combines a carrier, a physical offset, and a
layout. A carrier owns or references storage and dispatches the operations it
supports; StrideWeave deliberately has no separate device abstraction.

The project is currently a tested prototype rather than a complete PyTorch
replacement. It provides native CPU kernels, a Python reference carrier,
autograd, hierarchical layout transformations, a small module system, and a
user-facing tensor layer (`strideweave.friendly`), with a minimal model layer
(`strideweave.nn`) above it as a backstop. It does not yet include accelerator
carriers.

## Specifications And This Document

The normative contracts live in `openspec/specs/<capability>/spec.md` and are
published at <https://strideweave.org/spec/>. A spec states external behavior in
testable terms: what a conforming implementation must do, for someone who has
never seen this repository.

This document is the mental model. It says what the pieces are, why they are
shaped the way they are, and how they fit together, so that reading a spec — or a
source file — starts from the right picture. It is deliberately **lossy** about
anything a spec owns: it gives the shape of a rule and names the spec that states
it exactly. Where the two disagree, the spec is authoritative and this file is the
bug.

The looseness is the point. This file grew long because it was the only source of
truth, so every clarification had to be defended here. Re-enumerating a
spec-owned contract rebuilds exactly that. When behavior changes, update the
owning spec; update this file only when the mental model itself changes.

| Area | Owning spec |
| --- | --- |
| Immutable shaped-domain index maps, coordinate conversion, composition, and specialized map kinds | [`index-maps`](openspec/specs/index-maps/spec.md) |
| Hierarchical layout values, coordinate mapping, structural transforms, broadcasting, tiling algebra | [`core-layout`](openspec/specs/core-layout/spec.md) |
| Tensor construction, representation validation, carrier ownership, multi-subtensor boundaries | [`core-tensor-representation`](openspec/specs/core-tensor-representation/spec.md) |
| Zero-copy views, their layout transformations, failures, and reverse-mode behavior | [`core-tensor-views`](openspec/specs/core-tensor-views/spec.md) |
| Reverse-mode graph construction, backward traversal, accumulation, functional VJPs | [`autograd`](openspec/specs/autograd/spec.md) |
| The layout command language behind every description string, and the lowering each command compiles to | [`layout-commands`](openspec/specs/layout-commands/spec.md) |
| DLPack export, the carrier hook enabling it, and cross-carrier movement | [`interop-movement`](openspec/specs/interop-movement/spec.md) |
| Dtype identity, hierarchy, discovery, immutability, extension, serialization | [`dtype-descriptors`](openspec/specs/dtype-descriptors/spec.md) |
| Compound storage schemas, representation rules, block-scaled structure | [`dtype-representations`](openspec/specs/dtype-representations/spec.md) |
| Carrier construction, storage dtypes, value access, factories, mutation, versioning, release | [`carrier-storage`](openspec/specs/carrier-storage/spec.md) |
| Operation dispatch, exact-class metadata, backend extension, preflight, composite lowering | [`carrier-dispatch`](openspec/specs/carrier-dispatch/spec.md) |
| Evictable hierarchy construction, ownership, residency transitions, lifecycle | [`carrier-composition`](openspec/specs/carrier-composition/spec.md) |
| Overload selection, promotion, arithmetic, accumulation, result dtype | [`operation-dtype-policy`](openspec/specs/operation-dtype-policy/spec.md) |
| Capability descriptors, registration, introspection, enforcement | [`backend-capabilities`](openspec/specs/backend-capabilities/spec.md) |
| Profiling lifecycle, event evidence, nesting, timing, aggregation, reporting | [`operation-profiling`](openspec/specs/operation-profiling/spec.md) |
| Staged local verification of the installed backend | [`kernel-verification`](openspec/specs/kernel-verification/spec.md) |
| Verification report format, strict loading, filtering, summaries, CLI | [`verification-report`](openspec/specs/verification-report/spec.md) |
| Raw evidence persistence, provenance, store lifecycle, contributor exchange | [`kernel-evidence-tracking`](openspec/specs/kernel-evidence-tracking/spec.md) |
| `strideweave.friendly` creation, layout builders, scalar reductions, value extraction | [`friendly-tensor-creation`](openspec/specs/friendly-tensor-creation/spec.md), [`friendly-layout-builders`](openspec/specs/friendly-layout-builders/spec.md), [`friendly-scalar-reductions`](openspec/specs/friendly-scalar-reductions/spec.md), [`friendly-value-extraction`](openspec/specs/friendly-value-extraction/spec.md) |

Areas this document still states in full, because no spec owns them yet, are
operation semantics beyond dtype planning — structural alignment, reductions,
convolution, indexing, selection — the module system, and `strideweave.nn`.
Those sections are correspondingly precise: they are the contract until a spec
takes it over.

## Core Model

- `Tensor(carrier, offset, layout)` references storage owned by a `Carrier`.
  This public constructor creates the conventional one-subtensor case of the
  authoritative internal representation: one logical dtype, an ordered tuple
  of carrier-backed subtensors, one placement `Layout` per level, and one
  adjacent `Layout` between each pair of levels. The carrier, offset, and layout
  properties read subtensor zero rather than parallel fields.
- `IndexMap` is the immutable algebra for mapping a hierarchical `Shape` domain
  into a flat integer range. Its `codomain_size` is a declared exclusive bound,
  not the number of reached indices or an inferred image size. `Shape.encode`
  and `Shape.decode` are the shared first-mode-fastest (colexicographic)
  coordinate authority, including nested subtree-scalar and rank-zero forms.
  [`index-maps`](openspec/specs/index-maps/spec.md) owns the exact contract.
- The four concrete IndexMap siblings have distinct structural roles. `Layout`
  uses parallel `Shape` and `Stride` trees; `Permutation` is an explicit sparse
  lookup; `Swizzle` is a sequence of validated XOR-field stages; and `Product`
  packs child maps while retaining their explicit expression tree. Composition
  consumes the inner map first. Closed same-kind pairs retain their specialized
  kind, while other compatible pairs use a private flattened result that is
  exposed only as `IndexMap`.
- `Layout` remains the physical specialization. `layout.profile` exposes the
  shape tree's leaf-and-nesting recipe without its extents or strides, and
  `layout.is_injective` reports whether every logical coordinate maps to a
  distinct physical offset. `layout.broadcast_to(shape)` widens only extent-one
  leaves at the same hierarchical positions, setting their strides to zero; it
  never flattens, rank-aligns, inserts, removes, or reorders modes. That refusal
  to guess is the layout algebra's defining choice.
- `Tiler` is the public type alias for a read-only sequence of `Layout` values,
  used by the composition APIs (`Layout.compose`, `Layout.divide_tiler`,
  `Layout.zipped_divide`) to describe one tile per leading hierarchical mode.
- `layout.size` is the logical element count, while `layout.cosize` — also
  exposed as its IndexMap `codomain_size` — is the physical storage size the
  Layout addresses (one past its largest offset).
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
  The base `Carrier` method owns the shared dispatch policy and tags each fresh
  operation with its canonical name and exact dispatching carrier class; custom
  carriers override the `_dispatch_op` factory hook rather than `dispatch_op`.
  Dispatch is uniformly instance-based. A carrier composing another returns a
  composite-owned adapter for delegated names: the adapter owns the visible
  autograd node, lowers tensor arguments into the representation the nested
  operation accepts, runs it through sealed lowered execution, and wraps results
  and gradients back into the composite representation.
- Python and native operations inherit from the shared native `Operation` base.
  `Operation._forward` is a protected implementation hook and must not be
  invoked directly. Call public `forward`, or use the framework-owned sealed
  lowered-execution path from a composite adapter, so operation preflight,
  result validation, profiling, and autograd bookkeeping remain intact.
- Views may use different layouts and offsets while sharing the same carrier.
- In the generalized representation, placement layout `L_i` maps level
  coordinate space `c_i` into carrier `i`, while adjacent layout `S_i` maps
  `c_i` to an integer decoded in `c_(i+1).shape`. Both are ordinary CuTe-style
  `Layout` values; their structural positions distinguish physical placement
  from logical grouping. Universal validation checks that structure — storage
  schema, carrier dtypes and classes, offsets, `cosize` bounds, adjacent
  compatibility — before any dtype-specific rule runs.
- Current Tensor operations take a structural one-subtensor fast path, and
  native CPU access, views, results, movement, scatter, autograd, and DLPack all
  read carrier, offset, layout, dtype, and version state through the
  authoritative representation. Validated multi-subtensor representations
  support only the pure c0 layout views (`permute`, `slice`, `reshape`,
  `as_strided`, `broadcast_to`, `squeeze`, `unsqueeze`), which preserve every
  carrier, offset, and deeper layout while rebuilding and revalidating the top
  level; everything else refuses them before allocation or carrier mutation.
- An external representation rule annotates `validate` with the public
  `RepresentationValidationContext` protocol, whose read-only fields become
  available only after universal validation succeeds.

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

Three specs state this model exactly:
[`core-layout`](openspec/specs/core-layout/spec.md) for the layout algebra and
its coordinate mapping, transforms, broadcasting, and tiling;
[`core-tensor-representation`](openspec/specs/core-tensor-representation/spec.md)
for construction, the ordered representation, validation, and the
multi-subtensor boundary; and
[`core-tensor-views`](openspec/specs/core-tensor-views/spec.md) for the
zero-copy views and their reverse-mode behavior. Dispatch itself is
[`carrier-dispatch`](openspec/specs/carrier-dispatch/spec.md).

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

[`carrier-storage`](openspec/specs/carrier-storage/spec.md) states what each of
these carriers accepts, allocates, and reports, and
[`carrier-composition`](openspec/specs/carrier-composition/spec.md) states
`Evictable`'s ownership and residency behavior.

These four are closed implementations: `Carrier` is the extension interface and
stays open, but `Generic`, `CPU`, `FileBacked`, and `Evictable` reject subclass
creation with a message naming the supported alternative, and each is declared
`@final` on every import path, so a type checker reports the same closure before
the program runs. Each states its allocation factories, storage normalization,
dispatch metadata, and capability declarations in terms of its exact class —
`Evictable` in terms of its exact instances — so a specialization would inherit
claims it cannot honor: a `Generic` subclass would advertise every plan
`Generic` executes while `Generic.new_like` refused to allocate a result for it.

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
attributes of `DType` are exactly the built-in descriptors, in a namespace that
is frozen against rebinding, so class lookup and `DType.from_name` can never
disagree.

Constructing a descriptor is one transaction owned by the dtype model rather
than a sequence an implementation has to order correctly: the most-derived
`__init__` assigns its own fields, and the finished descriptor is then
validated, sealed against mutation, and published under the registry lock. A
constructor that raises registers nothing, a concurrent lookup never observes a
partly initialized descriptor, and claiming a name is thread-atomic.
[`dtype-descriptors`](openspec/specs/dtype-descriptors/spec.md) states the
identity, registration, extension, and serialization contract exactly.

Descriptors expose `name`, `supertype`, `supertypes()`, `is_simple()`,
`is_category()`, `is_compound()`, `is_opaque_storage()`, and
`is_subtype_of(other)`, and the registry is queried through `DType.registered()`
and `DType.from_name(name)`, both narrowed to the receiving class. The kind
predicates classify the *representation*, not backend availability: there is
deliberately no global "is this storable" predicate, because storability is a
decision of an exact carrier class together with a dtype. `E4M3` is a
well-formed simple dtype that no carrier accepts today.

The hierarchy has three descriptor kinds, plus the legacy opaque disposition:

- `SimpleDType` is one fixed-width scalar encoding rather than a composition
  of subtensors, so a single carrier could store it homogeneously, and each
  reports an exact `bits` width. `DType.Float32`, `DType.Int32`, and
  `DType.Bool` (8 bits) are the concrete simple storage dtypes carriers support
  today; `DType.Float64` is an accumulator-only descriptor used by operation
  plans and is not accepted as carrier storage; the registry also defines the
  simple encodings `Int8`, `E8M0`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, and `E2M1`,
  which exist so the block-scaled formats can name their elements and scales and
  are otherwise structural only. Being simple describes the encoding, not
  carrier support.
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
  descriptor may also carry reusable `RepresentationRule` values, which
  constrain logical grouping only and run after the universal representation
  checks succeed. `LevelExtent(level, extent)` is the first such rule.
  [`dtype-representations`](openspec/specs/dtype-representations/spec.md) owns
  the compound and rule contract.

Descriptors describe representation; they do not decide what an operation
computes in or returns. That policy is specified in
[`operation-dtype-policy`](openspec/specs/operation-dtype-policy/spec.md) and
implemented by the backend-independent planner in
`strideweave.carriers.operation_policy`, which resolves overload selection,
promotion, arithmetic, accumulation kind and accumulator dtype, and result dtype
from the operands alone. Shape, axis, ordering, and other execution parameters
never enter promotion as accidental weak scalars, and autograd eligibility
follows from the result dtype rather than being restated per plan. Floating
reduction association order is backend-defined. The policy is a deliberately
evolvable starting point rather than a compatibility promise.

`Generic` executes those plans and is the reference every other backend
conforms to. Native `CPU` resolves the same plans: each operation asks the
planner for its overload, promotion, arithmetic, accumulation, and output dtype
while the GIL is still held, then releases it to run the kernel that plan
selected. No backend carries a promotion table of its own, so Generic and CPU
agree on Float32, Int32, and Bool storage/results by construction rather than by
parallel maintenance.

One consequence of the policy is worth calling out for `pow`: an exponent
preserves an `Int32` result only when it is a weak *integer* in
`[0, 2**31 - 1]`. A float exponent takes the floating path even when its value
is integral, so `int_tensor ** 2` stays `Int32` while `int_tensor ** 2.0` is
`Float32`. On the integer path the exponent is used exactly rather than carried
through a float, so an exponent above `2**24` keeps its parity.

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

The four queries answer from whichever set owns that carrier: an independent
carrier's exact-class declarations, or the frozen snapshot a dependent carrier
generated for that instance. A caller asks the carrier and never has to know
which. The query describes the carrier's *capabilities* rather than its own
storage dtype, so a `Float32` CPU carrier still reports the `Int32` plans CPU
executes, and these are the same entries execution is accepted against, so an
advertised plan is executable and an executed plan was advertised. An
unsupported plan yields a reason naming the operation or the shape, and
`require_operation_plan` raises `UnsupportedOperationPlan`.
[`backend-capabilities`](openspec/specs/backend-capabilities/spec.md) states the
descriptor fields, ordering, registration, sealing, and refusal contract.

Capabilities belong to an exact carrier class and are never resolved through its
bases: a class that declares nothing supports nothing. A class declares once —
`Generic`, `CPU`, and `FileBacked` during carrier-package initialization,
`FileBacked` declaring the empty set as a stated fact — and its answer is fixed
from then on, because publication and sealing happen in the same call and first
observation seals an undeclared class's empty set. That is what lets one carrier
snapshot another class's reach without the snapshot going stale; the practical
rule is to declare a custom carrier's capabilities in its own module, at import
time, before anything can ask.

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
decides what it executes is entirely its own. Finalization is explicit and
cooperative rather than automatic, because only the concrete carrier knows when
its dependencies are complete — construction calls
`_finalize_dependent_capabilities()` as its last step, in the same thread,
before the unfinished instance becomes reachable anywhere else, and the call
materializes, validates, and freezes the snapshot once. Before it, the instance
answers no capability query, which keeps a half-built hierarchy from advertising
an empty set a caller then trusts. This is a construction protocol, not a
thread-safe initialization one: concurrent calls and leaking `self` during
construction are outside the supported extension contract.

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

### Extending The Dtype Model

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

The governing idea is that a descriptor's identity is *composed by the model,
not reported by the descriptor*. Each contract class — the root descriptor,
categories, simple, compound, block-scaled — owns one immutable specification
carrying the members it defines, the fragment it contributes to a canonical
structure, the validation it requires, and whether its descriptors are
additionally unique by structure. Finalizing a descriptor resolves those
specifications through the method resolution order and applies them general to
specific. Nothing about identity is dispatched through the descriptor, so an
implementation cannot omit a canonical layer from its fingerprint, decline the
uniqueness its contract imposes, or weaken a check it still claims to satisfy.
Model-owned accessors and the names the model once dispatched through are
reserved: redefining one — in the class body, in a mixin, or as an instance
attribute that would shadow it — is refused rather than silently honored. An
implementation extends the model by adding its own fields and, when its
representation carries state of its own, by overriding `structure_extension()`,
the one hook whose value it controls.

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
descriptor. A pickle carries the descriptor's name and its recursively expanded
structure, which the *receiving* process resolves against its own registry; it
never ships a dtype definition. A built-in therefore unpickles anywhere
`strideweave` is imported, while a dynamically registered dtype requires the
receiving process to register a matching descriptor first, and a receiver whose
matching name has a different width, category, or planes is rejected rather than
silently substituted. Shipping the dtype definition itself — a cross-process
descriptor schema — is deliberately out of scope and remains possible future
work.

Together these rules describe guardrails, not a sandbox. The model validates a
descriptor class's hierarchy when the class is created, validates the completed
descriptor during finalization, seals every registered instance, and keeps the
built-in namespace frozen. Beyond that it is contract rather than enforcement:
Python classes stay mutable, and code that reaches around the model — a custom
`__dict__`, reassigned `__slots__`, a mixin mutated after the fact,
`object.__setattr__`, or the dtype module's private state — forfeits the
registry, structural-identity, pickle, and immutability guarantees rather than
being intercepted. The dtype model is an API-integrity boundary against mistakes
and drift in cooperative code, not a defense against hostile metaprogramming
sharing the process.

The runtime implementation is organized behind the stable
`strideweave.carriers.dtype` facade. Private modules separate canonical
structure encoding, contract ownership, the descriptor and registry model,
block-scaled definitions, built-in installation, and carrier-facing storage
validation. This organization is internal: public import paths, class module
identities, structure fingerprints, and pickle resolution continue to use
`strideweave.carriers.dtype`.

### Block-Scaled Descriptors

`BlockScaledDType` is the currently implemented compound dtype: one simple
element dtype plus a linear chain of simple scale `Level` entries. A level's
`block` extent is measured in the *previous* level's coordinate space, so each
level coarsens the grouping below it, and only the final level may use the
symbolic `Whole` extent that produces a single scale for the entire tensor.
`Whole` is a true singleton, so equivalent whole-scaled formats cannot bypass
structural uniqueness. `simple_types` maps position to plane dtype, `num_axes`
counts the blocking axes a tensor of this dtype must be given, and
`representation_rules` derives one `LevelExtent` per scale level from that same
chain, using the generic representation validator available to external compound
formats — Tensor validation contains no block-scaled-specific branch.

The registered formats are `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP6_E3M2`,
`MXFP6_E2M3`, `MXFP4`, `MXINT8`, and `NVFP4`. Descriptors are unique by
structure as well as by name, anchored at `BlockScaledDType` itself, so a
subclass adding no structure of its own describes an already registered
representation and is rejected instead of becoming a second identity for it.
These descriptors are structural: block-scaled tensors, tilers, quantization,
requantization, and dispatch eligibility are not implemented.
[`dtype-representations`](openspec/specs/dtype-representations/spec.md) states
the level algebra, derived rules, and registered formats exactly.

### Generic Reference Semantics

On concrete storage `Generic` implements the encodings faithfully rather than
approximating them with Python's own numeric types. `Float32` storage holds
binary32-exact values and rounds to binary32 at every step, with IEEE
singularities as results rather than exceptions in forward and backward alike;
`Int32` arithmetic is exact with checked narrowing, so an out-of-range result
raises `OverflowError` rather than wrapping, while a reduction accumulates
exactly and checks only the final sum; `Bool` storage contains only `False` and
`True`, is non-differentiable, and is never implicitly promoted. Floating
`reduce_sum` and `matmul` advertise both the default `Float32` accumulator and
an explicit `Float64` one, widening only after loading already encoded `Float32`
terms. Concrete storage is normalized and owned — values are converted when
stored, and the carrier copies the supplied sequence — so no caller-held alias
can place an unrepresentable value or change stored values without the version
counter observing it. NumPy supplies the binary32 mechanics and is imported
lazily on first concrete `Float32` use, so importing StrideWeave, or using only
`CPU`, `Int32`, or the legacy dtypes, never loads it.
[`carrier-storage`](openspec/specs/carrier-storage/spec.md) and
[`operation-dtype-policy`](openspec/specs/operation-dtype-policy/spec.md) state
these encodings and their arithmetic exactly.

The legacy dtypes are outside this policy. An operation whose operands mix
legacy `Any`/`Floating` storage with concrete storage stays on Generic's
historical Python arithmetic rather than silently selecting a concrete plan,
which means the concrete operand's binary32 semantics are downgraded to
binary64 for that operation. Legacy `Any` values are never routed through
checked integer arithmetic.

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
identity rather than equality, in the native CPU parser as well as in Python,
and every registered descriptor outside a carrier's row is rejected at
construction with the same diagnostic — including a compound descriptor, whose
multi-plane storage (one carrier per entry of `simple_types`) is future work,
and `Float64`, which currently names widened accumulator arithmetic only.
Carrier-authoring code reuses
`strideweave.carriers.dtype.validate_storage_dtype` to get that behavior.

The same table is readable at run time, without allocating anything:

```python
sw.CPU(4, dtype=sw.DType.Float32).supports_storage_dtype(sw.DType.Int32)   # True
sw.CPU(4).supports_storage_dtype(sw.DType.Integer)                         # False
```

`supports_storage_dtype(dtype)` asks whether the carrier's *implementation* can
allocate that dtype at all. It is structural rather than a report of state: it
allocates nothing and is unaffected by size, mutability, ownership, eviction
residency, release, or the dtype the carrier currently holds. `Evictable`
reports the intersection of its tiers, because a value it cannot evict is a
value it cannot hold — which is what lets a composed carrier decide, before any
work begins, whether it could store an operation's result. `Carrier` owns the
public query and its validation; an implementation states its accepted set
through the protected `_supports_storage_dtype(dtype)` hook, exactly as
`_is_mutable()` works, and the conservative default claims only the dtype the
instance currently holds.

Fresh storage comes from two factories every carrier exposes: `allocate_like`
for size-based allocation, where `empty=True` permits a backend to skip
initialization so callers must write every element they read, and `new_like` for
materializing supplied values.

Only `Floating` and `Float32` tensors participate in autograd. That set is an
explicit pair rather than a `Floating` category query, because the narrow
floating encodings have no numerical semantics yet.

Carriers may be mutable or immutable, mutation increments a version counter
visible through `tensor.version`, and `release()` permanently releases storage.
`is_mutable()` reports whether public interfaces may currently write the
carrier, not only whether its storage was constructed mutable, so a carrier
owned by a composite reports `False` while its mutable owning composite reports
`True`; ownership is applied centrally by `Carrier` over each implementation's
`_is_mutable()` hook. Eviction and promotion belong specifically to `Evictable`
rather than to the base `Carrier` or `Tensor` APIs.
[`carrier-storage`](openspec/specs/carrier-storage/spec.md) and
[`carrier-composition`](openspec/specs/carrier-composition/spec.md) own the
allocation factories, mutation and versioning rules, and the ownership and
residency contract.

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
changing input encoding or planned output dtype. Those two are the only
operations that take the option, because they are the only sum reductions whose
association order is not observable; every other combining operation pins a
normative order and offers no accumulator choice. `None` uses the backend's
default `Float32` accumulator, `DType.Float64` requests widened accumulation and
must match an advertised backend capability, and the `tensor @ other` spelling
keeps the default, so callers requesting `Float64` use `sw.matmul`. Matmul
backward reuses the accumulator its forward call selected rather than treating
the choice as a tensor input.
[`operation-dtype-policy`](openspec/specs/operation-dtype-policy/spec.md) states
which operations are configurable and how a request is refused.

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

Reductions take a reduce command and lower it to a two-mode intermediate whose
second mode is the reduction fiber — that lowering is the command language's,
described under Layout Descriptions below; what each reduction then does with
the fiber is its own. `reduce_sum`, `reduce_prod`, `reduce_max`, and
`reduce_min` return Float32 results; `argmax` and `argmin` return Int32
first-winner indices. `cumsum` is an inclusive scan over one explicit top-level
mode. Fibers must be non-empty, and `keepdims` is composed with `unsqueeze`
rather than a reduction option.

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

An Evictable tensor dispatches through a public `EvictableOperation` adapter,
which is the worked example of the composite lowering described under Core
Model: it owns one fresh primary-carrier operation, lowers inputs to temporary
primary-backed tensors, runs them through the framework's sealed
lowered-execution route, and is the sole visible autograd node, so CPU and
Generic implementations need no composition-specific code. Before it lowers
anything, it resolves the plan the central policy gives for its *outer* operands
and requires that plan against the hierarchy's own frozen capabilities, so a
plan the hierarchy does not advertise never reaches a nested allocation or a
kernel. New operation results allocate only their promoted primary storage; the
secondary tier stays empty until the first eviction provisions it.
[`carrier-dispatch`](openspec/specs/carrier-dispatch/spec.md) and
[`carrier-composition`](openspec/specs/carrier-composition/spec.md) state the
adapter, lowering, and residency contract.

### Layout Descriptions

StrideWeave layout descriptions preserve hierarchical modes and therefore do not
have standard flat einops semantics. String forms include:

```python
transposed = sw.rearrange(tensor, "a b -> b a")
summed = sw.reduce_sum(tensor, "a (b c) -> a b")
contracted = sw.einsum(lhs, rhs, "a b, c b -> a c")
batched = sw.einsum(lhs, rhs, "b i k, b j k -> b i j")
```

The native lexer and Python parsers compile these descriptions into layout trees
and cache successful specifications. The classification that matters for reading
the code is by *output presence*: a shared symbol omitted from the output is
contracted, a shared symbol retained in it is a batch dimension, and a one-sided
symbol is free and must appear in the output. That distinction picks the
lowering — a contraction without batch symbols keeps the two-mode matmul path,
while a batched contraction aligns both operands over their union symbol space
with differentiable singleton broadcasts, multiplies elementwise, and reduces
only the omitted shared symbols. The general path materializes the union-shaped
product; there is no native batched-matmul kernel yet.

[`layout-commands`](openspec/specs/layout-commands/spec.md) states the command
language exactly — lexing, the layout-reference grammar, the three command
forms, their diagnostics, and the lowering each compiles to. It is one contract
rather than a parser surface plus separate operations, and it governs *every*
public entry point that takes a description string, not only the
`strideweave.einops` exports: a command that compiles for `rearrange` or
`einsum` compiles identically for `reduce_sum`, `reduce_max`, `argmin`, and
their siblings, which then contribute their own operation semantics on top. The
layout, view, dtype, dispatch, and autograd contracts the lowering executes stay
with their own specs.

## Operation Profiling

`profile` is a single-use context manager that records carrier-dispatched
operation computations on the current thread. The boundary it records is
deliberately narrow: the post-preflight computation-and-result-validation step,
not dispatch factory lookups, option validation, structural preflight, or
autograd attachment. Events carry the canonical operation name, exact
dispatching carrier and implementation classes, timing, thread identity, parent
relationship, and success status, plus shape snapshots when
`record_shapes=True`.

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

Selection is by exact carrier class (`carriers=None` records every one), so
nested composite execution is visible when selected: an Evictable operation
produces an outer Evictable event and a nested event for its promoted CPU or
Generic operation, and filtering the nested event out does not charge its time
to the parent's self time. Aggregates derive from the immutable raw events.

Profiling state is thread-local, so work on another thread requires its own
context, and a context must exit on the thread that entered it. Timings measure
synchronous host wall time only; asynchronous accelerator activity is not
modeled. [`operation-profiling`](openspec/specs/operation-profiling/spec.md)
states the recorded boundary, event fields, nesting and timing arithmetic,
thread rules, exclusions, and report determinism exactly.

## Autograd

Operations attach an autograd context when gradient construction is enabled,
the result is differentiable, and at least one tensor input is differentiable.
Backward traversal is iterative and topological, so shared subgraphs accumulate
their pending gradients before their operation runs. Differentiability follows
the logical dtype — only `Floating` and `Float32` tensors participate, and `Any`
and `Int32` tensors reject the gradient APIs.

Three properties are worth carrying in your head; the rest is contract:

- **Saved state is released once.** `backward(gradient=None,
  retain_graph=False)` releases the saved inputs, versions, and operation
  context for every node it reaches, and nodes stay attached afterwards so a
  second traversal fails explicitly rather than treating a former result as a
  leaf. A shared subgraph is released by whichever reachable root traverses it
  first. `retain_graph=True` on the earlier traversal is how a graph is reused.
- **A cotangent must address its tensor exactly.** Non-scalar tensors require an
  explicit gradient; an exact shape `[1]` is the scalar case that may call
  `backward()` with an implicit one. Leaf tensors accumulate `.grad` by default,
  non-leaves retain it only after `retain_grad()`, and `no_grad()`,
  `is_grad_enabled()`, and `set_grad_enabled()` control the thread-local
  graph-building state.
- **Gradient buffers are always injective.** When a leaf tensor uses stride-zero
  broadcast modes, `.grad` sums every logical contribution addressing the same
  storage slot and represents the result in a canonical injective layout with
  the tensor's logical shape, at any hierarchy depth. Other non-injective
  layouts, such as overlapping non-zero strides, are refused rather than
  producing an under-counted gradient. Views are differentiable on the same
  terms: their backward path scatters gradients into a tensor with the original
  input layout.

Backward also validates saved input versions, so storage modified in place after
the forward pass raises instead of silently differentiating the wrong values.

### Functional gradients

`sw.grad(output, inputs, cotangents, *, batched=False, retain_graph=False)`
computes vector-Jacobian products without reading or modifying any tensor's
`.grad` field, returning one gradient per requested input in positional order
and `None` for an input unreachable from `output`.

With `batched=True`, one tensor represents K cotangents: its layout is a
prepended leaf batch mode followed by modes exactly equal to `output.layout`,
and every reachable input produces one stacked gradient whose batch stride is
the single-gradient layout's `cosize` rather than its logical size, so inputs
with storage holes keep adjacent slices disjoint. The native traversal discovers
the shared topology once and propagates each cotangent through it independently.

[`autograd`](openspec/specs/autograd/spec.md) states graph construction,
traversal, accumulation, saved-version validation, and the functional VJP
contract exactly — including that reverse mode is the complete differentiation
surface, so `create_graph`, double backward, and forward-mode JVPs are refused
rather than approximated.

## Modules

`Module` provides basic PyTorch-like structure: subclasses implement `forward`,
and `__call__` delegates to it. Assigning public `Parameter` or `Module`
attributes registers them for `modules()`, `parameters()`, and
`get_named_parameters()` traversal. Optional module and parameter names can
override attribute-name segments.

Buffers, state dictionaries, training/evaluation modes, and hooks are not
implemented yet. A minimal layer library and optimizer live in `strideweave.nn`
(see Ergonomic Layers below).

## Ergonomic Layers

The core carriers stay composable primitives. Two submodule-only packages sit
above them, built entirely from the public primitives and not re-exported at the
top level. They are different layers rather than peers:

- `strideweave.friendly` is the user interface over tensor computation. It
  removes the boilerplate of assembling a carrier, an offset, and a layout by
  hand, and its behavior is specified.
- `strideweave.nn` is a further layer above that, for models rather than
  tensors. It is an unpolished backstop — enough to train the example MLP —
  rather than a maintained model library, and is deliberately unspecified: see
  the note on `strideweave.nn` below.

### strideweave.friendly

`import strideweave.friendly as F` provides compact layout builders
(`column_major`, `row_major`), CPU tensor factories (`tensor` from nested
lists, `zeros`, `ones`, `full`, `arange`, `rand`, `randn`), scalar reductions
(`sum`, `mean`, both returning `Shape(1)` tensors that support implicit
`backward()`), and value extraction (`item`, `to_list`).

All thirteen exports are spec-owned:
[`friendly-tensor-creation`](openspec/specs/friendly-tensor-creation/spec.md),
[`friendly-layout-builders`](openspec/specs/friendly-layout-builders/spec.md),
[`friendly-scalar-reductions`](openspec/specs/friendly-scalar-reductions/spec.md),
and [`friendly-value-extraction`](openspec/specs/friendly-value-extraction/spec.md)
state their value ordering, dtype selection, allocation, layout, reverse-mode
behavior, and diagnostics.

### strideweave.nn

`import strideweave.nn as nn` provides `Linear`, activation module wrappers
(`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softplus`, `ELU`, `LeakyReLU`),
`MSELoss`, and an `SGD` optimizer. There is deliberately no `strideweave.nn`
spec: the layer is provisional, and specifying it would freeze behavior that is
expected to change.

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

End-to-end training examples live in `examples/train_mlp_cpu.py` (raw
primitives) and `examples/train_mlp_cpu_friendly.py` (same model using the
helpers).

## Interoperability And Movement

CPU tensors export DLPack through `__dlpack__` and `__dlpack_device__`, with
hierarchical shapes and strides flattened for the DLPack representation. Export
is the interesting boundary: it is zero-copy, same-device, and opt-in per
carrier through a `dlpack_info` hook, so Generic, FileBacked, and Evictable do
not participate, a multi-subtensor tensor is refused rather than partially
described, and there is no import, copy, or cross-device path.

`move(tensor, destination)` dispatches on the exact source and destination
carrier class *pair* against an explicit process-global registry: CPU-to-
FileBacked and FileBacked-to-CPU use native bulk copies, and every other pair —
including a subclass of a registered class, and `(CPU, CPU)` — falls back to
elementwise copying until registered on its own. A new pair is a public
`MoveOperation` subclass implementing the protected `_copy` hook, registered for
its exact pair. A successful move releases the source carrier only after that
hook returns, and autograd moves gradients back into fresh source-class storage.
Moving a broadcast tensor preserves its exact stride-zero layout and copies its
`cosize` physical span rather than materializing `size` logical elements, so its
backward consumes an injective same-shape gradient before the broadcast node
performs the required summation.

Evictable is the framework's own consumer of that registry: it resolves a move
per residency transition and routes it through the sealed lowered-execution
path, so a transition receives shared result validation without adding an
autograd node. Element access, forward operations, and scatter are unavailable
while values are evicted, while backward may still run against an evicted
result because the result storage is not read. Replacement tiers publish only
after a move succeeds, so a failed transition leaves the prior residency and
ownership valid and retryable.

Ownership guards apply to carrier interfaces, not to memory. Explicit
external-memory escape hatches — `CPU.pointer()`, direct writes to a
`FileBacked` path, mutation of the container originally handed to `Generic` —
remain the caller's responsibility and cannot participate in version tracking.

[`interop-movement`](openspec/specs/interop-movement/spec.md) states the export
structure, versioning, mutability advisory, capsule lifetime, dtype eligibility,
carrier hook, and the move dispatch, registration, and autograd contract;
[`carrier-composition`](openspec/specs/carrier-composition/spec.md) states the
residency transitions themselves.

## Current Boundaries

What is deliberately not built yet, and why it is safe to leave undone:

- No CUDA, Metal, or other accelerator carriers. The carrier interface is the
  extension point for one; nothing in the core assumes a device abstraction.
- No carrier stores `Float64`. It names widened accumulator arithmetic only, and
  a request for it is gated by each backend's exact capabilities rather than
  silently downgraded.
- High-level tensor creation lives only in `strideweave.friendly` and is
  CPU-backed; tensors over other carriers are assembled from primitives.
- Structural views, implicit singleton alignment in binary pointwise operations,
  and the separate reduction, scan, contraction, convolution, indexing,
  selection, and movement capabilities are the whole shape-manipulation surface.
  There is no flat-layout rank inference anywhere, by design — see
  [`core-tensor-views`](openspec/specs/core-tensor-views/spec.md) for what each
  view guarantees, including `as_strided`'s origin-based logical mapping, which
  is not PyTorch's physical-storage reinterpretation.
- The broader IndexMap algebra does not make every map a Tensor placement.
  Tensor placement and adjacent grouping, tiling, broadcasting, complement,
  `as_strided`, and native layout caches remain Layout-only because they depend
  on stride structure and physical `cosize`. Permutation, Swizzle, Product, and
  private generic compositions are mapping metadata, not alternate storage
  representations.
- IndexMaps are currently static immutable structures. Callable- or
  Tensor-backed maps, data-dependent selection, a public explicit
  materialization operation, inverse maps, and equality or hashing for generic
  expression graphs remain deferred.
- Public compound tensor construction, and multi-subtensor coordinate indexing,
  mutation, non-view operations, movement, release orchestration, and DLPack
  export, remain deferred; the pure c0 layout views are the narrow validated
  exception. The internal representation and the optional-rule contract exist
  precisely so those paths can be added later without a parallel public tensor
  type, which is why the restriction is a refusal rather than a missing class.
  [`core-tensor-representation`](openspec/specs/core-tensor-representation/spec.md)
  bounds it exactly.
- Block-scaled descriptors are structural only: no tensors, tilers,
  quantization, requantization, or dispatch eligibility.
- `FileBacked` supports storage and movement, not computation, and declares that
  as an empty capability set rather than leaving it unstated.
- Evictable tensors must be promoted before access or computation, and binary
  Evictable operations require matching primary and secondary carrier classes.
- `strideweave.nn` covers only `Linear`, elementwise activations, `MSELoss`, and
  `SGD`; there are no buffers, state dictionaries, training/evaluation modes,
  or hooks. It is a backstop for the examples, not a model library.

## Local Kernel Verification

`test_backend(output=None)` verifies the installed CPU backend without CI,
network, or database access.

```python
report = sw.test_backend()
sw.test_backend("kernel-evidence.jsonl")
```

Its organizing idea is that coverage is *enumerated from the build rather than
asserted by the test*. Cases are derived from the native kernel manifest and the
backend's active capability plans, so every registered kernel is either actively
certified or explicitly deferred with a stated reason, and a kernel added in C++
without a classification fails the manifest check rather than passing silently.
Stage One compares the correctly rounded and exact-integer kernels bit for bit
against `Generic` on seeded arbitrary finite encoded inputs; operations whose
accumulation order is normative use payloads whose every legal partial result is
exactly representable, so a comparison never encodes an association order the
contract does not fix. Stage Two runs ordinary target execution only where an
exact kernel-ID/variant certificate reconstructed from Stage One evidence covers
the plan it depends on — a missing, forged, or incomplete certificate blocks its
dependent cases without executing them. Floating `pow`,
vendor-transcendental kernels, and autograd certification remain visible v0
deferrals rather than being reported as passes.
[`kernel-verification`](openspec/specs/kernel-verification/spec.md) states the
classification, staging, payload, envelope, and gate contract exactly.

The returned report is immutable and provenance-complete: alongside passed,
failed, errored, blocked, and deferred attempts, it binds the native compilation
manifest, target and toolchain, per-kernel receipts and source closures,
verification requirements, tolerance policies, Generic reference identity, and
Stage One certificates. It deliberately contains no wall-clock timestamp, CI
status, database state, source commit, or status aggregation — those are facts
about a run's environment, not about the build's behavior.

```python
summary = report.summary()
assert summary.gate_passed
print(report.describe())
failed_or_blocked = report.problems
stage_one_records = report.select(stage=sw.verification.VerificationStage.ORACLE)
```

`report.write(path)` saves the deterministic versioned JSONL and
`VerificationReport.load(path)` strictly reloads it. Loading is the model-owned
inverse: it validates the header and every nested content identity without
consulting the currently installed build, and reconstructs each embedded Stage
One certificate from the report's own evidence. This is a fail-closed
consistency check for implementation drift and stale certificate wiring, not an
authenticity or producer-trust protocol. The inspection-only
`strideweave-verify-report` CLI loads through that same strict parser and
neither runs tests, contacts a network, nor writes status data:

```bash
strideweave-verify-report kernel-evidence.jsonl --problems --verbose
strideweave-verify-report kernel-evidence.jsonl --stage stage_two --operation matmul
```

[`verification-report`](openspec/specs/verification-report/spec.md) states the
format, loading rules, filters, summaries, and CLI exit behavior.

### Raw evidence persistence

The persistence layer is a local-first store of *raw facts, never confidence
claims*. Its behavioral contract is
[`kernel-evidence-tracking`](openspec/specs/kernel-evidence-tracking/spec.md).
Three properties shape everything else about it:

- Identity is content-addressed and complete. A kernel's compilation identity
  covers its source, the compiler-reported complete transitive input closure,
  definitions, flags, target, toolchain, and compiled object, so what
  invalidates evidence is a changed input rather than a new commit. Project and
  build inputs use stable relative URIs while external inputs are path-free and
  content-addressed, so a report retains no local installation paths.
- Facts are additive. Contradictory producer observations coexist rather than
  being resolved by timestamp; `status` is a factual inventory, `stale` explains
  identity differences axis by axis, and `todo` is an unranked deterministic set
  difference.
- Access is explicit. `test_backend`, imports, and every `--help` path are
  offline and mutation-free; a store is touched only by an explicit
  record/query command, and a network endpoint only under `--publish` or
  `--refresh`.

```bash
strideweave-kernel-status record --report kernel-evidence.jsonl \
  --producer local-dev --source-commit "$REVISION"
strideweave-kernel-status status --arch arm64 --kernel cpu.matmul --json
strideweave-kernel-status stale --arch arm64
```

Recording first rebinds the complete report against the installed build's
compilation, specification, tolerance, oracle, and certificate facts, so a
stale, incomplete, mismatched, or forged report fails before the store is
created. The store lives in the platform application-data directory —
`STRIDEWEAVE_STATUS_HOME` replaces only its base, `--store PATH` overrides one
command — and is separate from Beads, source Git, and the working tree.

StrideWeave owns the Dolt process model rather than exposing it: the runtime is
resolved and version-checked once per executable identity, and SQL transport is
an internally managed `dolt sql-server`, at most one per resolved data directory
per process, on a loopback port, terminated before interpreter exit. No
lifecycle API, environment variable, or command-line option selects a host,
port, or server. Registration and running are one condition, so a shutdown or
failed startup retires the server it releases and a racing acquisition waits for
that process to exit rather than starting a replacement it would refuse; a
forked child inherits an empty registry and never signals or tears down a
parent's server. Every store operation runs over one persistent session with
bound parameters and bounded multi-row batches — no `dolt sql` subprocess, and
no statement that grows with the size of a report.

Optional contributor exchange is opt-in per command: each producer receives an
opaque namespace holding one atomically replaced content-addressed snapshot, and
refresh validates the envelope, identities, relationship graph, and embedded
evidence before it initializes or mutates anything, merging atomically. Local
paths and `file:` endpoints are the initial transport; no central service or
account is required.

Confidence policies, ranked verification levels, autotuning, real JIT adapters,
MSVC/Visual Studio provenance, manual assembly/sanitizer levels, and CI
integration remain future work.

## Development

The package requires Python 3.12 or newer and builds its native modules with
scikit-build-core and pybind11.

Development workflow lives in [`CONTRIBUTING.md`](CONTRIBUTING.md): environment
setup, the native rebuild step, the local verification suite, the test markers
and the native boundary they enforce, and what each CI job does and why. Before
designing a change, read the cross-cutting contracts in
[`INVARIANTS.md`](INVARIANTS.md). It records the canonical implementation choices and
whether each invariant is enforced by AST lint, Ruff, behavioral tests, native builds,
or code review.

## License

StrideWeave is licensed under the Apache License, Version 2.0. See `LICENSE`.
