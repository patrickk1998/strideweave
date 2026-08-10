---
title: Core Tensor Views
publish: true
status: stable
order: 3
summary: Zero-copy Tensor views, hierarchical layout transformations, failures, and reverse-mode behavior.
---

# core-tensor-views Specification

## Purpose

Define StrideWeave's zero-copy Tensor views, including carrier sharing,
hierarchical layout transformations, explicit failures, and reverse-mode
behavior.

## Terminology

| Term | Meaning |
| --- | --- |
| source Tensor | The Tensor supplied to a view operation. |
| view Tensor | The Tensor returned by a successful view operation. |
| zero-copy | Reuses the source Tensor's constituent carriers without copying carrier values or allocating replacement carriers. |
| Tensor offset | The non-negative base carrier index to which a placement layout's linear index is added. This is distinct from a layout's linear index. |
| leading layouts | Placement layout `L_0` and, when present, adjacent layout `S_0`, whose domain is the Tensor's public logical-coordinate space. |
| source ordinal | An integer in `[0, source.size())` that identifies a logical coordinate from the source Tensor in first-mode-fastest order. |
| cotangent | The derivative value propagated backward from an operation's output during reverse-mode differentiation. |

## Requirements

### Requirement: Views share carriers through fresh representations

Every successful view operation SHALL return a new Tensor with the source
Tensor's logical dtype and identical constituent carrier objects. The view
operation SHALL construct and validate a fresh authoritative representation,
reuse carrier values in place, and leave every source carrier's version and
release state unchanged.

If the transformed representation violates a
`core-tensor-representation` requirement, the view operation SHALL propagate
the exception produced for that exact representation violation, SHALL return
no Tensor, and SHALL NOT mutate or release any source carrier.

#### Scenario: Create a zero-copy view
- **WHEN** a supported view transformation produces a valid representation
- **THEN** the view Tensor has a fresh representation, the same logical dtype,
  and the identical constituent carriers as the source Tensor
- **AND** every source carrier retains its previous version and release state

#### Scenario: Reject an invalid transformed representation
- **WHEN** a view transformation produces a representation that fails
  representation validation
- **THEN** the operation reports that validation error, returns no Tensor, and
  mutates and releases no source carrier

### Requirement: Multi-subtensor views transform only leading layouts

For a validated multi-subtensor Tensor, `permute`, slice indexing, `reshape`,
`as_strided`, `broadcast_to`, `broadcast_in_dim`, `squeeze`, and `unsqueeze`
SHALL apply their transformation to `L_0` and `S_0`. They SHALL preserve every
placement layout `L_i` for `i > 0`, every adjacent layout `S_i` for `i > 0`,
every subtensor carrier and Tensor offset, the logical dtype, and the order of
all representation entries.

If transforming `L_0` or `S_0` violates an operation-specific condition, the
operation SHALL fail with the exception specified by that operation's
requirement. If the transformed complete representation instead violates a
`core-tensor-representation` requirement, the operation SHALL propagate the
exception produced for that exact representation violation. Both failure
paths SHALL return no Tensor and SHALL NOT mutate or release any carrier.

#### Scenario: Transform a multi-subtensor leading domain
- **WHEN** one of the listed views validly transforms a multi-subtensor Tensor
- **THEN** the same transformation is applied to `L_0` and `S_0`, every deeper
  layout and Tensor offset is preserved, and the complete result is validated

#### Scenario: Reject an invalid multi-subtensor transformation
- **WHEN** the transformation is invalid for either `L_0`, `S_0`, or the
  resulting complete representation
- **THEN** the operation reports the applicable error, returns no Tensor, and
  mutates and releases no carrier

### Requirement: Slice indexing creates a view only when a slice is present

In `tensor[key]`, `tensor` is the source Tensor and each normalized entry of
`key` selects from the corresponding source top-level mode. An integer selects
one logical coordinate and removes that mode from a view result. A `slice`
selects a regularly spaced range and preserves that mode in a view result.

Tensor indexing SHALL return a new view Tensor when the key contains at least
one `slice`. The key SHALL contain exactly one integer or `slice` per top-level
mode. For a one-subtensor source, an all-integer key SHALL return the carrier
value at the source Tensor offset plus the source placement layout's linear
index for the supplied logical coordinate. The all-integer path SHALL leave the
source Tensor and carrier unchanged.

For a leaf mode, an omitted slice step SHALL default to `1`. Slice bounds SHALL
use Python slice normalization, the normalized step SHALL be positive, the
normalized interval SHALL be non-empty, and the interval width SHALL be
divisible by the step. The output mode SHALL have extent
`(stop - start) / step` and stride `source_stride * step`. An integer key SHALL
remove its selected top-level mode. A non-leaf mode SHALL accept only a whole
slice or one in-domain integer over that mode's first-mode-fastest logical
coordinates.

For a one-subtensor source, the view Tensor offset SHALL equal the source
Tensor offset plus the source placement layout's linear index at the first
selected logical coordinate. For a multi-subtensor source, every normalized
slice start and every integer selection SHALL equal zero, and all Tensor
offsets SHALL remain unchanged.

If the key count differs from the top-level mode count, the operation SHALL
fail with `ValueError`. If a key or slice step has the wrong type, it SHALL
fail with `TypeError`. If an integer is outside its mode, a slice step is not
positive, a slice is empty, a slice interval is not divisible by its step, or
a non-leaf slice is not whole, it SHALL fail with `ValueError`. If a
multi-subtensor selection has a nonzero normalized origin, it SHALL fail with
`NotImplementedError`. If a multi-subtensor key contains only integers,
coordinate access SHALL fail with `NotImplementedError`. Every failure SHALL
raise before returning a value or Tensor and SHALL NOT mutate or release any
carrier.

#### Scenario: Slice a leaf mode with the default step
- **WHEN** source layout `Shape([5, 10]), Stride([1, 5])` is indexed by
  `(2, slice(1, 9))`
- **THEN** the view Tensor offset increases by the source linear index at
  logical coordinate `(2, 1)` and the view layout is
  `Layout(Shape(8), Stride(5))`

#### Scenario: Slice a leaf mode with a positive step
- **WHEN** source layout `Shape([5, 10]), Stride([1, 5])` is indexed by
  `(2, slice(1, 9, 2))`
- **THEN** the view Tensor offset increases by the source linear index at
  logical coordinate `(2, 1)` and the view layout is
  `Layout(Shape(4), Stride(10))`

#### Scenario: Preserve a non-leaf mode with a whole slice
- **WHEN** a source layout with shape `[10, [2, 3]]` is indexed by `(0, :)`
- **THEN** the view layout preserves `[2, 3]` as one hierarchical mode

#### Scenario: Reject an invalid slice key
- **WHEN** a key violates any key-count, key-type, mode-domain, step,
  interval, or non-leaf-slice condition above
- **THEN** slice-view construction reports the specified `TypeError` or
  `ValueError`, returns no Tensor, and mutates and releases no carrier

#### Scenario: Reject a nonzero multi-subtensor slice origin
- **WHEN** a multi-subtensor source is sliced with a normalized nonzero start
  or a nonzero integer selection
- **THEN** slice-view construction fails with `NotImplementedError`, returns no
  Tensor, and mutates and releases no carrier

### Requirement: Tensor rearrangement applies the core layout result

In `rearrange(tensor, output, selection)`, `tensor` is the source Tensor,
`selection` identifies the source sublayouts available to the rearrangement,
and `output` describes the hierarchy constructed from those identified
sublayouts and inserted extent-one leaves. A string `output` describes the same
selection and output hierarchy through the public StrideWeave rearrangement
syntax.

`rearrange` SHALL return a new view Tensor whose output layout is derived by
applying the `core-layout` rearrangement contract to the source layout with the
supplied output tree and effective selection tree. When the selection tree is
not supplied, it SHALL default to every source leaf in depth-first,
left-to-right order. A string description SHALL be parsed into output and
selection trees and SHALL then use the same rearrangement rule. A caller SHALL
supply either that string or a `Tree` output. The returned view Tensor SHALL
preserve the source Tensor's carrier and Tensor offset.

If a string output is combined with an explicit selection tree, `rearrange`
SHALL fail with `TypeError`. If the output or selection has the wrong type, it
SHALL fail with `TypeError`. If the description or requested layout
rearrangement is invalid, it SHALL fail with `ValueError`. If the source has
multiple subtensors, it SHALL fail with `NotImplementedError`. Every failure
SHALL return no Tensor and SHALL NOT mutate or release any carrier.

#### Scenario: Rearrange a one-subtensor Tensor
- **WHEN** a valid output tree or rearrangement description reorders the source
  layout
- **THEN** the view Tensor shares the source carrier and Tensor offset and uses
  the rearranged layout

#### Scenario: Use the default rearrangement selection
- **WHEN** a caller supplies a valid output tree without a selection tree
- **THEN** `rearrange` extracts source leaves in depth-first, left-to-right
  order before constructing the output layout

#### Scenario: Reject an invalid rearrangement
- **WHEN** the inputs violate a type, description, source-identifier, omitted-
  source, or multi-subtensor condition above
- **THEN** `rearrange` reports the specified error, returns no Tensor, and
  mutates and releases no carrier

### Requirement: Tensor permutation reorders complete top-level modes

In `permute(tensor, *order)`, `tensor` is the source Tensor and `order[j]`
identifies the source top-level mode that becomes top-level mode `j` of the
returned view Tensor.

`permute(tensor, *order)` SHALL return a new view Tensor whose leading layouts
equal the results of applying the `core-layout` permutation to the source's
leading layouts. The returned view Tensor SHALL preserve every Tensor offset. A
nested mode SHALL move as one mode. The order SHALL contain every top-level
mode index exactly once. For a source with no top-level modes, the empty order
SHALL satisfy this requirement.

If an order entry is not an integer, `permute` SHALL fail with `TypeError`. If
the order omits, duplicates, or names an out-of-domain top-level mode,
`permute` SHALL fail with `ValueError`. Each failure SHALL return no Tensor and
SHALL NOT mutate or release any carrier.

#### Scenario: Permute hierarchical top-level modes
- **WHEN** a Tensor with shape `[[2, 3], 4, 5]` is permuted by `(1, 0, 2)`
- **THEN** the view Tensor has shape `[4, [2, 3], 5]`, preserves every Tensor
  offset, and shares every carrier

#### Scenario: Reject an invalid permutation
- **WHEN** the order has a non-integer, duplicate, missing, or out-of-domain
  entry
- **THEN** `permute` reports `TypeError` or `ValueError` as specified, returns
  no Tensor, and mutates and releases no carrier

### Requirement: Reshape preserves first-mode-fastest linear order

In `reshape(tensor, target)`, `tensor` is the source Tensor and `target`
describes the requested output coordinate hierarchy. `target` changes how the
same source ordinal sequence is grouped into modes without changing that
sequence.

`reshape(tensor, target)` SHALL require `target` to be a `Shape` with
`target.size == tensor.size()`. A source leading layout SHALL be
reshape-compatible when visiting its logical coordinates in
first-mode-fastest order produces the linear-index sequence
`0, b, 2b, ..., (source.size - 1)b` for some integer `b >= 0`.
On success, `reshape` SHALL return a new view Tensor whose leading layouts
preserve `target` hierarchy and assign its leaves first-mode-fastest compact
strides scaled by the corresponding `b`. The returned view Tensor SHALL
preserve every Tensor offset.

For a multi-subtensor Tensor, both `L_0` and `S_0` SHALL independently satisfy
that reshape-compatible condition for the same target, using their respective
base strides.

If `target` is not a `Shape`, `reshape` SHALL fail with `TypeError`. If the
sizes differ or any transformed leading layout is not reshape-compatible,
`reshape` SHALL fail with `ValueError`. Every failure SHALL return no Tensor
and SHALL NOT copy values, mutate carriers, or release carriers.

#### Scenario: Reshape a layout with regular holes
- **WHEN** source layout `Shape([2, 3]), Stride([2, 4])` is reshaped to
  `Shape([3, [2]])`
- **THEN** the view layout is `Layout(Shape([3, [2]]), Stride([2, [6]]))`, the
  Tensor offset is preserved, and logical values remain in the same
  first-mode-fastest order

#### Scenario: Reject a size mismatch
- **WHEN** `target.size` differs from `tensor.size()`
- **THEN** `reshape` fails with `ValueError`, returns no Tensor, and copies and
  mutates no carrier value

#### Scenario: Reject a reshape that requires copying
- **WHEN** no base stride satisfies the reshape-compatible condition for a
  transformed leading layout
- **THEN** `reshape` fails with `ValueError`, returns no Tensor, and copies and
  mutates no carrier value

### Requirement: As-strided composes an origin-based selector layout

In `as_strided(tensor, shape, stride)`, `tensor` is the source Tensor, `shape`
describes the output logical-coordinate domain, and `stride` describes how
coordinates in that domain select source ordinals.

`as_strided(tensor, shape, stride)` SHALL require `shape` to be a `Shape` and
`stride` to be a congruent `Stride`. Together, `shape` and `stride` SHALL define
selector layout `B` over the output logical coordinates. For each output
logical coordinate `q`, linear index `B(q)` SHALL be interpreted as a source
ordinal identifying the logical coordinate read from the source Tensor. `B`
SHALL be injective, `B(0)` SHALL equal zero, and `B.cosize` SHALL be at most
`tensor.size()`.

On success, `as_strided` SHALL return a new view Tensor with shape `shape`. For
every output logical coordinate `q`, the returned view placement layout SHALL
return `L_0(B(q))`. For a multi-subtensor source, the returned first adjacent
layout SHALL return `S_0(B(q))`. Every composed placement layout SHALL be
injective, every composition SHALL be representable as a `Layout`, and the
returned view Tensor SHALL preserve every Tensor offset and deeper layout.

If `shape` or `stride` has the wrong type, `as_strided` SHALL fail with
`TypeError`. If their hierarchy differs, `B` aliases source ordinals,
`B.cosize > tensor.size()`, a required composition is not representable as a
`Layout`, or a composed placement aliases carrier entries, `as_strided` SHALL
fail with `ValueError`. Every failure SHALL return no Tensor and SHALL NOT copy
values, mutate carriers, or release carriers.

#### Scenario: Select source ordinals from a noncanonical layout
- **WHEN** source layout `Shape([5, 4]), Stride([4, 1])` is viewed with selector
  layout `Shape([2, 2]), Stride([1, 2])`
- **THEN** the view layout is `Layout(Shape([2, 2]), Stride([4, 8]))`, every
  Tensor offset is preserved, and every carrier is shared

#### Scenario: Reject an aliasing selector layout
- **WHEN** `B` maps two output logical coordinates to the same source ordinal
- **THEN** `as_strided` fails with `ValueError`, returns no Tensor, and copies
  and mutates no carrier value

#### Scenario: Reject a selector layout outside the source
- **WHEN** `B.cosize > tensor.size()`
- **THEN** `as_strided` fails with `ValueError`, returns no Tensor, and copies
  and mutates no carrier value

#### Scenario: Reject an aliasing composed placement
- **WHEN** composing `B` with a source placement layout produces a
  non-injective placement layout
- **THEN** `as_strided` fails with `ValueError`, returns no Tensor, and copies
  and mutates no carrier value

### Requirement: Broadcast-to widens only matching singleton leaves

In `broadcast_to(tensor, target)`, `tensor` is the source Tensor and `target`
describes the requested output shape. Matching source and target leaves
identify the same logical coordinate, while a target coordinate at a widened
source extent-one leaf reads source coordinate zero at that leaf.

`broadcast_to(tensor, target)` SHALL require `target` to be a `Shape` accepted
by the `core-layout` broadcasting contract for every transformed leading
layout. On success, `broadcast_to` SHALL return a new view Tensor whose shape is
`target` and whose transformed leading layouts equal the results of applying
that broadcasting contract to the corresponding source leading layouts and
`target`. The returned view Tensor SHALL preserve every Tensor offset and
carrier. Each widened extent-one leaf SHALL use stride zero in the
corresponding returned view layout.

If `target` is not a `Shape`, `broadcast_to` SHALL fail with `TypeError`. If a
shape profile differs, a non-singleton extent would change, or any transformed
leading layout cannot broadcast to `target`, `broadcast_to` SHALL fail with
`ValueError`. Every failure SHALL return no Tensor and SHALL NOT copy values,
mutate carriers, or release carriers.

#### Scenario: Broadcast a nested singleton leaf
- **WHEN** a Tensor with layout `Shape([2, [1, 3]]), Stride([1, [2, 2]])` is
  broadcast to `Shape([2, [4, 3]])`
- **THEN** the view layout is
  `Layout(Shape([2, [4, 3]]), Stride([1, [0, 2]]))`, every Tensor offset is
  preserved, and every carrier is shared

#### Scenario: Reject an incompatible broadcast target
- **WHEN** the target has a different profile or changes an extent other than
  one
- **THEN** `broadcast_to` fails with `ValueError`, returns no Tensor, and
  copies and mutates no carrier value

### Requirement: Broadcast-in-dim inserts only explicitly omitted modes

In `broadcast_in_dim(tensor, target, broadcast_dimensions)`, `tensor` is the
source Tensor, `target` describes the requested output shape, and
`broadcast_dimensions[i]` identifies the target top-level mode occupied by
source top-level mode `i`. A target top-level mode absent from
`broadcast_dimensions` is a new broadcast-only mode.

`broadcast_in_dim(tensor, target, broadcast_dimensions)` SHALL require
`target` to be a `Shape`. The `broadcast_dimensions` sequence SHALL contain
one integer target-mode index for each source top-level mode in strictly
increasing order. Each named target mode SHALL hold the corresponding source
mode. On success, `broadcast_in_dim` SHALL return a new view Tensor with shape
`target`. It SHALL produce the returned view Tensor by first inserting each
omitted target mode as an extent-one, stride-zero top-level mode and then
applying `broadcast_to` with `target`.

If `target` is not a `Shape` or a `broadcast_dimensions` entry is not an
integer, `broadcast_in_dim` SHALL fail with `TypeError`. If the sequence count
differs from the source top-level mode count, contains an out-of-domain target
mode, or is not strictly increasing, it SHALL fail with `ValueError`. If the
resulting structural broadcast is invalid, it SHALL report the
`broadcast_to` failure. Every failure SHALL return no final view Tensor and
SHALL NOT copy values, mutate carriers, or release carriers.

#### Scenario: Insert an omitted leading mode
- **WHEN** a Tensor with `Layout(Shape(2), Stride(1))` is broadcast to
  `Shape([3, 2])` with `broadcast_dimensions=(1,)`
- **THEN** the view layout is `Layout(Shape([3, 2]), Stride([0, 1]))`, the
  Tensor offset is preserved, and the carrier is shared

#### Scenario: Reject an invalid target-mode sequence
- **WHEN** `broadcast_dimensions` has the wrong count, a non-integer entry, an
  out-of-domain entry, a duplicate, or decreasing entries
- **THEN** `broadcast_in_dim` reports `TypeError` or `ValueError` as specified,
  returns no final view Tensor, and copies and mutates no carrier value

### Requirement: Unsqueeze and squeeze transform explicit top-level modes

In `unsqueeze(tensor, dim)`, `tensor` is the source Tensor and `dim` identifies
the top-level mode index at which the returned view inserts a new mode. In
`squeeze(tensor, dim)`, `dim` identifies the existing source top-level mode
removed from the returned view. A negative `dim` identifies the corresponding
mode by counting backward from the applicable insertion or source-mode range.

`unsqueeze(tensor, dim)` SHALL require integer input `dim` and normalize a
negative value against the insertion range. On success, `unsqueeze` SHALL
return a new view Tensor whose leading layouts insert one extent-one,
stride-zero top-level mode at the selected mode index. For a source with `n`
top-level modes, accepted normalized insertion indices SHALL lie in
`[0, n + 1)`.

`squeeze(tensor, dim)` SHALL require integer input `dim`, normalize a negative
value against the source top-level mode count, and require the selected mode to
be a leaf with extent one. On success, `squeeze` SHALL return a new view Tensor
whose leading layouts remove exactly the selected top-level mode. Both returned
view Tensors SHALL preserve every Tensor offset.

If `dim` is not an integer, each operation SHALL fail with `TypeError`. If the
normalized mode index is out of domain, each operation SHALL fail with
`ValueError`. If `squeeze` selects a nested mode, a non-leaf mode, or an extent
other than one, it SHALL fail with `ValueError`. Every failure SHALL return no
Tensor and SHALL NOT copy values, mutate carriers, or release carriers.

#### Scenario: Insert and remove a singleton top-level mode
- **WHEN** `unsqueeze` inserts a mode and `squeeze` removes that same mode
- **THEN** both views share every carrier and preserve every Tensor offset, and
  the final leading layouts equal the source leading layouts

#### Scenario: Reject an invalid singleton-mode transformation
- **WHEN** `dim` has the wrong type, is out of domain, or `squeeze` selects a
  mode that is not a top-level extent-one leaf
- **THEN** the operation reports `TypeError` or `ValueError` as specified,
  returns no Tensor, and copies and mutates no carrier value

### Requirement: One-subtensor view backward applies the inverse logical transformation

In `view.backward(gradient, retain_graph)`, `view` is the differentiated view
Tensor, `gradient` is the cotangent for the view output, and `retain_graph`
indicates whether the recorded reverse-mode graph remains available after the
call. When `gradient` is not supplied, it represents a unit cotangent for a
scalar view. When `retain_graph` is not supplied, it represents a request to
release the recorded graph after the call.

`gradient` SHALL default to `None`. When `gradient` is `None`, the view shape
SHALL consist of exactly one leaf mode with extent one, and `backward` SHALL use
a unit cotangent with the view layout. If `gradient` is `None` for any other
view shape, `backward` SHALL fail with `ValueError` before modifying any
`.grad`. If a supplied `gradient` is not a Tensor, `backward` SHALL fail with
`TypeError` before modifying any `.grad`. `retain_graph` SHALL default to
`False`; `False` SHALL release the recorded graph after successful propagation,
and `True` SHALL preserve it for another backward call.

When reverse-mode differentiation records a one-subtensor view, the view SHALL
save the source Tensor and its carrier version. Slice indexing and `as_strided`
SHALL scatter cotangent values to the selected source ordinals and assign zero
to every unselected source ordinal. `rearrange`, `permute`, `reshape`,
`squeeze`, and `unsqueeze` SHALL apply the inverse logical transformation.
`broadcast_to` SHALL sum all cotangent contributions that correspond to each
source logical coordinate. `broadcast_in_dim` SHALL compose the `unsqueeze`
and `broadcast_to` reverse transformations.

The resulting source gradient SHALL use fresh carrier storage of the source
carrier's exact class. When the source placement layout is injective, the
source gradient SHALL restore that layout. When the source aliases only
through stride-zero broadcast modes, the source gradient SHALL use an
injective layout with the source shape and SHALL aggregate contributions as
required by `RT017`. After successful gradient propagation and required
`.grad` accumulation, a public `backward` call on the view Tensor SHALL return
`None`.

If an injective view receives a cotangent whose layout differs from the view
layout, reverse-mode differentiation SHALL fail with `ValueError`. If a
broadcast view receives a cotangent with a different shape or a non-injective
layout, it SHALL fail with `ValueError`. If a saved constituent carrier version
has changed after view creation, it SHALL fail with `RuntimeError` before
producing the source gradient. Each failure SHALL NOT modify the source
Tensor's `.grad` through that view operation.

#### Scenario: Scatter a slice cotangent
- **WHEN** reverse-mode differentiation receives a valid cotangent for a
  positive-step slice view
- **THEN** the source gradient restores selected cotangent values at their
  source ordinals and contains zero at every omitted source ordinal

#### Scenario: Reduce a broadcast cotangent
- **WHEN** a valid injective cotangent is supplied for a broadcast view
- **THEN** the source gradient sums every contribution associated with each
  pre-broadcast logical coordinate

#### Scenario: Invert a structural view
- **WHEN** a valid cotangent is supplied for `rearrange`, `permute`, `reshape`,
  `squeeze`, or `unsqueeze`
- **THEN** the source gradient applies the inverse logical transformation and
  restores the injective source layout

#### Scenario: Reject an invalid view cotangent
- **WHEN** a cotangent violates the applicable layout, shape, or injectivity
  condition above
- **THEN** reverse-mode differentiation fails with `ValueError` and does not
  modify the source Tensor's `.grad` through the view operation

#### Scenario: Reject a view whose source was mutated
- **WHEN** a saved constituent carrier version changes after view creation and
  before reverse-mode differentiation reaches the view
- **THEN** reverse-mode differentiation fails with `RuntimeError` and does not
  modify the source Tensor's `.grad` through the view operation

### Requirement: No-grad view creation detaches the view from its source

When a view is created while gradient recording is disabled, the view Tensor
SHALL have no autograd operation connecting it to the source Tensor. A later
backward call on that view SHALL treat it as a leaf and SHALL leave the source
Tensor's `.grad` unchanged. After successfully accumulating the detached view's
gradient, that backward call SHALL return `None`.

If a later backward call supplies an invalid cotangent for the detached view,
it SHALL fail with the ordinary Tensor backward `TypeError` or `ValueError`
before modifying either the detached view's `.grad` or the source Tensor's
`.grad`.

#### Scenario: Backward through a no-grad view
- **WHEN** a valid view is created with gradient recording disabled and later
  receives a valid cotangent
- **THEN** the view accumulates its own gradient as a leaf and the source
  Tensor's `.grad` remains unchanged

#### Scenario: Reject an invalid cotangent for a no-grad view
- **WHEN** backward on the detached view receives a non-Tensor cotangent or a
  cotangent with an invalid layout
- **THEN** backward reports `TypeError` or `ValueError` as applicable and
  modifies neither the detached view's `.grad` nor the source Tensor's `.grad`
