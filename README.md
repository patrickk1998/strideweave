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
- `Layout` describes hierarchical `Shape` and `Stride` trees and maps logical
  coordinates to physical storage indices.
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
- Operations dispatch through `tensor.carrier.dispatch_op(operation_name)`.
  The base `Carrier` method owns the shared dispatch policy: it calls the
  backend `_dispatch_op` factory hook, requires a fresh `Operation`, and tags
  that operation with its canonical name and exact dispatching carrier class.
  Custom carrier implementations must override `_dispatch_op`, not
  `dispatch_op`. Dispatch is uniformly instance-based; class-level
  `dispatch_op` calls are not part of the public contract. A Python subclass of
  `CPU` may extend the native registry by handling custom names in `_dispatch_op`
  and delegating standard names through `super()._dispatch_op(...)`.
- Python and native operations inherit from the shared native `Operation` base.
- Views may use different layouts and offsets while sharing the same carrier.

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
  objects. It supports differentiable `Floating` values and non-differentiable
  arbitrary `Any` values.
- `CPU(size, pointer=None, *, mutable=True, dtype=DType.Float32, empty=False)`
  owns native memory or references a caller-provided address. It supports
  `Float32` and `Int32`. Owned storage is zero-initialized unless `empty=True`
  opts into unspecified initial values; `empty` never initializes or changes
  caller-owned memory supplied through `pointer`.
- `FileBacked(filename=None, mutable=True, dtype=DType.Floating)` stores raw
  numeric values in a temporary binary file. It is intended for storage and
  movement rather than direct tensor computation.
- `Evictable(primary, secondary)` composes two carriers into a memory
  hierarchy. Computation uses promoted primary storage; `evict()` moves values
  to secondary storage and blocks access until `promote()` restores them. Its
  constructor takes exclusive ownership of both supplied carriers.

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
  reports an exact `bits` width. `DType.Float32` and
  `DType.Int32` are the concrete simple storage dtypes carriers support today;
  the registry also defines the simple encodings `Int8`, `E8M0`, `E5M2`, `E4M3`,
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
  multi-plane implementation would use, one carrier per plane.

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
| `Generic` | `Any`, `Floating` (legacy opaque storage) |
| `CPU` | `Float32`, `Int32` |
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

The public functional API includes:

- arithmetic: `add`, `sub`, `neg`, `mul`, `elementwise_mul`, `div`, `pow`, and
  `exp` (`sub` is implemented natively for CPU carriers and in Python for
  Generic carriers; `neg` is a composition of scalar `mul`);
- activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`,
  and `leaky_relu`;
- layout operations: `view`, `permute`, and `rearrange`;
- contractions: `reduce`, `matmul`, and `einsum`;
- storage movement: `move`.

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
composition-specific code.
New operation results allocate only their promoted primary storage. Their
secondary tier remains empty until the first eviction provisions it.

StrideWeave layout descriptions preserve hierarchical modes and therefore do not
have standard flat einops semantics. String forms include:

```python
transposed = sw.rearrange(tensor, "a b -> b a")
summed = sw.reduce(tensor, "a (b c) -> a b")
contracted = sw.einsum(lhs, rhs, "a b, c b -> a c")
```

The native lexer and Python parsers compile these descriptions into layout
trees and cache successful specifications.

## Operation Profiling

`profile` is a single-use context manager that records carrier-dispatched
operation executions on the current thread. It records execution attempts, not
dispatch factory lookups. Events contain the canonical operation name, exact
dispatching carrier and implementation classes, monotonic start time, inclusive
and self synchronous host wall time, thread identity, parent relationship, and
success status. With `record_shapes=True`, tensor argument positions also carry
immutable snapshots of their hierarchical shapes; non-tensor positions are
represented by `None`.

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

- Non-scalar tensors require an explicit gradient in `backward(gradient)`.
- An exact shape `[1]` is a scalar and may call `backward()` with an implicit
  gradient of one.
- Leaf tensors accumulate `.grad` by default.
- Non-leaf tensors retain `.grad` only after `retain_grad()`.
- `no_grad()`, `is_grad_enabled()`, and `set_grad_enabled()` control the
  thread-local graph-building state.
- `Any` and `Int32` tensors reject gradient APIs.
- Backward validates saved input versions and raises if required storage was
  modified in place after the forward pass.

Views are differentiable. Their backward path scatters gradients into a tensor
with the original input layout.

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
- `Linear` holds CPU `Float32` parameters and uses matmul plus the ones-column
  bias trick, so it requires CPU inputs in the flat column-major `[batch,
  features]` convention below.
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
no broadcasting primitive, so `Linear` broadcasts its `[out_features, 1]`
bias by contracting a constant ones column against it: `ones[batch, 1] @
bias[out, 1]` produces a tile whose layout matches the matmul output and
whose backward pass sums the bias gradient over the batch.

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
source carrier, and autograd moves gradients back into fresh source-class storage.

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
- High-level tensor creation lives only in `strideweave.friendly` and is
  CPU-backed; other carriers are constructed from primitives.
- No general broadcasting system; binary operations require compatible layouts
  and generally the same backing carrier (`strideweave.nn` composes around
  this with matmul-based bias broadcasting).
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
- Evictable tensors must be promoted before access or computation, and binary
  Evictable operations require matching primary and secondary carrier classes.
- `strideweave.nn` covers only `Linear`, elementwise activations, `MSELoss`, and
  `SGD`; there are no buffers, state dictionaries, training/evaluation modes,
  or hooks.

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
movement, modules, and public docstrings.

## License

StrideWeave is licensed under the Apache License, Version 2.0. See `LICENSE`.
