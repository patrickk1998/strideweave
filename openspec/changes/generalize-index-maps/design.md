## Context

See `proposal.md` for motivation. Today `Shape`, `Stride`, and `Layout` live in
one Python module, and `Layout` owns both recursive scalar expansion and a
native `_LayoutCache`. `Layout.compose` is a normal static function that can be
called as `Layout.compose(A, B)` or `A.compose(B)` and also accepts the
Layout-specific `Shape` and `Tiler` forms. The Tensor representation requires
ordinary Layouts for placement and adjacent grouping under RT014.

The current value objects are only partly immutable. Their tuple-backed trees
cannot be edited, but cached metadata and a Layout's `shape`, `stride`, and
cache references can be reassigned independently. That permits stale derived
state. The generalized algebra adds more value graphs and therefore needs one
mechanical immutability boundary before those graphs are composed.

## Goals / Non-Goals

**Goals:**

- Establish a small public map hierarchy whose common evaluation and
  composition rules do not depend on Tensor or carrier concepts.
- Keep affine Layout behavior and its native hot path compatible while moving
  coordinate normalization to Shape-owned shared machinery.
- Keep each specialized map independently implementable before cross-kind
  composition is integrated.
- Make the module boundaries suitable for a future extraction without making
  a separate package part of this change.

**Non-Goals:**

- No generic callable-map escape hatch or eager table materialization.
- No Tensor-backed or asynchronously produced map metadata.
- No partial carrier, promotion/eviction, gather/scatter, autograd, kernel, or
  inference-helper changes.
- No inverse-map API, hashing guarantee, or standalone library extraction.

## Decisions

### 1. Put normalization and ordinal conversion on Shape

`Shape.encode` and `Shape.decode` will own recursive first-mode-fastest conversion.
IndexMap evaluation will normalize through Shape before a concrete map applies
its own mapping. Layout's Python and native cache paths will use the same
semantic conversion and retain existing list/tuple/scalar inputs.

This avoids duplicating recursive coordinate parsing across four siblings and
makes Product packing an ordinary use of the inverse operation. Keeping the
logic only on Layout was rejected because Permutation, Swizzle, and Product
would then either depend on an affine type or grow subtly different parsers.

The native Layout cache remains specialized: it may keep flattened expanded
keys and indices for CPP002, but its results must be tested against the Shape
conversion authority.

### 2. Use an abstract public contract and a private generic expression

`IndexMap` will provide the shared read-only surface, coordinate normalization,
composition validation, and common dispatch to specialized composition hooks.
Its public role is a superclass and result type, not a constructor for arbitrary
functions. A private concrete composed-map implementation will hold an
immutable ordered child chain.

Generic composition will flatten nested generic chains and remove identity
nodes only when the result shape and codomain metadata remain identical.
Evaluation will stream one scalar through that chain and will not enumerate the
domain. Injectivity inference is deliberately conservative: an innermost known
collision remains a collision, an all-injective chain is injective, and other
cases are unknown because a preceding image may avoid a later map's collision.

An eager Permutation fallback was rejected because it changes construction
cost from structural to domain-sized, loses algebraic provenance, and is a poor
fit for large maps. A public `ComposedIndexMap` was rejected because callers
need the behavior, not a second expression-building API that would become a
compatibility commitment.

### 3. Preserve Layout as the affine specialization

Layout will inherit IndexMap and define `codomain_size` as an alias of its
existing `cosize`. Its exact Boolean injectivity analysis, profile,
broadcasting, complement, tiling, and static/class-qualified call forms remain
specialized.

Tensor storage validation and operation-result allocation will continue to use
`Layout.cosize` under RT002. The new `codomain_size` spelling is an IndexMap
view of the same Layout value, not an alternative allocation extent: a gapped
Layout with `size == 6` and `cosize == 10` still requires ten carrier elements.

The existing `compose` implementation will remain the Layout/Layout closure
path only when a rank-bounded structural check proves that its result is exact
and preserves the inner Layout's accepted hierarchical coordinate forms. That
specialized result may refine an existing inner leaf into a nested Layout mode,
because Shape's subtree-scalar rule still accepts the original leaf coordinate,
but it may not remove or reorder an existing coordinate level. A compatible
Layout/Layout pair that cannot meet that proof will use the same private generic
composition representation as a cross-kind pair. The generic result preserves
`inner.shape` exactly and evaluates the inner map before the outer map without
enumerating either domain.

The public method will continue to accept `Shape` and `Tiler` and return Layout
for those convenience forms. IndexMap siblings enter through the common
composition path and fall back structurally when Layout has no closed
representation. The private generic chain is introduced with the foundation so
Layout/Layout composition is total before sibling integration; the later
composition phase extends that representation across all map kinds and adds the
specified identity simplifications.

Treating every IndexMap as a legal Tensor placement was rejected. RT014 depends
on Layout-specific `cosize`, stride structure, and native caching; map
generality is a separate algebraic capability.

### 4. Model Permutation as an immutable explicit lookup

Permutation will copy and validate its input into a tuple, infer its one-mode
shape from the tuple length, and retain an explicit flat codomain bound. Its
constructor will reject repeats, so injectivity is exact. Permutation
composition can therefore lower by indexing the outer tuple with the inner
tuple while retaining the outer bound.

Inferring the bound from `max(values) + 1` was rejected because gaps above the
image can be semantically meaningful when the result feeds another shaped
domain. Allowing repeated values was rejected because routed expert IDs with
repetition are runtime Tensor data, whereas this type is immutable mapping
metadata.

### 5. Represent Swizzle as a sequence of involutive stages

`SwizzleStage` will be a frozen public value. Its two equal-width XOR fields
are disjoint and fit in the fixed domain bit width, making each stage
involutive. Swizzle copies stages into a tuple, requires a power-of-two domain,
and fixes its codomain size to its domain size.

Swizzle composition concatenates `inner.stages` before `outer.stages`, then a
small stack normalization cancels adjacent equal involutions. Closure does not
depend on discovering a global normal form: any remaining stage sequence is
still a Swizzle, and complete cancellation leaves a zero-stage Swizzle.

A single opaque bit-permutation callable was rejected because it would make
validation, equality, composition closure, and later code generation
uninspectable. Restricting identity to Swizzle was also rejected; compact
Layout remains the identity for arbitrary non-power-of-two spaces.

### 6. Preserve Product's explicit expression tree

Product stores an immutable tuple of at least two children. Variadic
construction makes three or more modes concise, but nested Product children are
not flattened. Domain Shape construction preserves each child's shape as one
mode, and target Shape construction preserves the corresponding Product target
as one nested mode. Evaluation splits the Product coordinate by that tree,
evaluates children, and calls `target_shape.encode` exactly once.

Product/Product composition lowers componentwise only when both explicit trees
align recursively: corresponding Product nodes have equal arity, and each
corresponding non-Product inner child has `codomain_size` exactly equal to the
outer child's `size`. Those equal leaf radices make the inner Product's target
encoding congruent with the outer Product's domain decoding. A pair that
satisfies total flat codomain containment but lacks either tree alignment or
exact leaf-radix congruence remains composable through the private generic
expression. Automatic Product flattening was rejected because it would erase a
user's intended hierarchy; binary-only construction was rejected because
ordinary multi-mode products would become unnecessarily clumsy.

### 7. Enforce immutability at every public value boundary

Shape, Stride, Layout, IndexMap siblings, SwizzleStage, and exposed sequences
will reject semantic field assignment and deletion after construction.
Constructor collections will be copied. Internal initialization and cache
population may use private mutation, but a Layout's Shape/Stride identity cannot
diverge from the data used to construct its native cache.

Hashability is deferred. Immutability is required for stable composition and
caches, but choosing equality and hash semantics for generic expression graphs
is a separate public contract.

### 8. Keep exports, documentation, and invariants synchronized

Runtime and stub exports will add `IndexMap`, `Permutation`, `Product`,
`Swizzle`, and `SwizzleStage` at the top-level and layout facade, with public
docstrings and examples. This preserves RT005 and RT006.

`llms.md` Core Model will describe shaped-domain IndexMaps, flat
`codomain_size`, the four concrete kinds, composition closure/fallback, and
Shape coordinate conversion. Current Boundaries will state that Tensor
placement remains Layout-only and that dynamic/callable/materialized maps are
deferred.

`INVARIANTS.md` must change with implementation:

- RT002 continues to require Tensor and operation-result storage to cover
  `Layout.cosize`; introducing `codomain_size` does not replace that allocation
  rule.
- RT014 remains Layout-specific and will explicitly distinguish the broader
  IndexMap algebra from Tensor placement and adjacent grouping.
- RT015 will be expanded so the shared immutable
  domain/codomain/composition and tri-state injectivity rules are
  cross-cutting, while Layout broadcasting, exact injectivity, and complement
  remain their existing canonical choices.
- RT005 and RT006 govern the new public exports and stubs.
- CPP002 continues to require native kernels to consume cached expanded Layout
  keys and indices; the shared semantic normalization must not introduce Python
  coordinate reconstruction into hot loops.

## Risks / Trade-offs

- **[Risk] Layout inheritance changes dispatch or static-call compatibility**
  → Preserve `Layout.compose(A, B)` and `A.compose(B)` in tests alongside Shape
  and Tiler regression coverage before adding sibling dispatch.
- **[Risk] Immutability breaks code that relied on accidental field mutation**
  → Treat such mutation as unsupported stale-cache behavior, document the
  intentional tightening, and retain copy/construction APIs for producing new
  values.
- **[Risk] Python and native coordinate normalization diverge**
  → Build shared conformance vectors for complete scalars, subtree scalars,
  nested tuples/lists, rank-zero shapes, and failures; keep the native cache as
  a derived acceleration only.
- **[Risk] Generic composition becomes a deep call chain**
  → Flatten private generic nodes into one immutable child tuple and evaluate
  iteratively. Defer materialization until a measured workload needs it.
- **[Risk] Legacy Layout composition returns an inexact affine result**
  → Specialize only after rank-bounded divisibility, compactness, and coordinate-
  structure checks establish the existing lowering's exactness. Use the generic
  chain for every other bound-compatible pair; do not probe by enumerating the
  domain or hide algebra failures behind exception handling.
- **[Risk] Swizzle terminology is confused with arbitrary permutation**
  → Keep the stage formula and bit-width validation public and exact; use
  Permutation for explicit arbitrary finite lookups.
- **[Risk] Product hierarchy and flat containment disagree**
  → Use structure only to decide specialized lowering. Flat containment still
  decides whether the mathematical composition is valid.

## Migration Plan

1. Add shared Shape conversion and IndexMap infrastructure, mechanically freeze
   Shape/Stride/Layout, preserve all existing Layout behavior and caches, and
   make Layout/Layout composition total through exact specialization or the
   private generic-chain fallback.
2. Add the three sibling map types and their independent constructor,
   evaluation, immutability, and injectivity tests.
3. Integrate specialized and generic composition, then add cross-kind,
   identity, and explicit Product-tree coverage.
4. Align runtime/stub exports and public docstrings; update `llms.md` and
   `INVARIANTS.md`; run the complete repository quality gates.

Before release, the accepted delta specs are synchronized through the normal
OpenSpec review/archive flow. Rollback is a source-level revert of the complete
integrated change; no persisted data or wire format is migrated.
