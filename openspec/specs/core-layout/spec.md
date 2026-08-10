---
title: Core Layout
publish: true
status: stable
order: 1
summary: Hierarchical layout values, coordinate indexing, structural transforms, broadcasting, and tiling algebra.
---

# core-layout Specification

## Purpose

Define StrideWeave's public hierarchical layout values and the observable
coordinate-to-scalar-index mapping, structural queries, broadcasting,
transformations, and tiling algebra on which tensor behavior depends.

## Terminology

| Term | Meaning |
| --- | --- |
| extent | The positive integer count associated with one leaf mode of a `Shape`. |

## Requirements

### Requirement: Shape and stride values preserve hierarchy

In `Shape(*items)`, each integer item is an extent and each nested iterable is a
nested mode. In `Stride(*items)`, each integer item is the linear-index change
for one increment of the congruent Shape leaf, and each nested iterable mirrors
a nested Shape mode. In `shape[key]` and `stride[key]`, `key` identifies one
top-level mode or a top-level slice to preserve in the result.

`Shape(*items)` SHALL return a Shape containing the described hierarchy of
positive integer extents. `Stride(*items)` SHALL return a Stride containing the
described hierarchy of non-negative integer stride values. A single iterable
and the equivalent variadic items SHALL return equal values. `shape[key]` SHALL
return a Shape and `stride[key]` SHALL return a Stride preserving the selected
hierarchy. If a Shape extent is less than one or a Stride value is negative,
construction SHALL fail with `ValueError` before returning a value.

#### Scenario: Construct equivalent hierarchical values

- **WHEN** a caller constructs `Shape(2, [3, 4])` and `Shape([2, [3, 4]])`
- **THEN** the two shapes compare equal
- **AND** the corresponding variadic and single-iterable `Stride` forms compare equal

#### Scenario: Reject invalid leaves

- **WHEN** a caller supplies a shape extent less than one or a negative stride
- **THEN** construction raises a value error

### Requirement: Layout construction requires structural congruence

In `Layout(shape, stride)`, `shape` defines the hierarchical logical-coordinate
domain and `stride` defines the linear-index contribution of each congruent
Shape leaf.

`Layout(shape, stride)` SHALL return a Layout containing that Shape and Stride
when they have exactly the same leaf and nesting structure. If `shape` is not a
Shape, `stride` is not a Stride, or their hierarchies differ, construction SHALL
fail with `ValueError` before returning a Layout.

#### Scenario: Reject a mismatched shape and stride tree

- **WHEN** a caller pairs `Shape([2, [3, 4]])` with `Stride([1, 2, 6])`
- **THEN** layout construction raises a value error identifying the structural mismatch

### Requirement: Public tree markers describe layout structure

In `Tree(*items)`, an ordinary integer or `Node.Leaf` denotes one leaf and a
nested Tree denotes one nested subtree. In `Node.id(n)`, `n` identifies one
source sublayout consumed by a later tree-based transformation.

`Tree(*items)` SHALL return a Tree preserving the described leaves and nesting.
`Node.id(n)` SHALL return an identified leaf marker containing non-negative
integer source identifier `n` without changing the Tree recipe or leaf count.
Reading `tree.recipe` SHALL return the ordered list of public `Node` markers
describing that hierarchy, and reading `tree.size` SHALL return its leaf count
as an integer.
If a Tree item is neither an integer, a leaf marker, nor a Tree, Tree
construction SHALL fail with `ValueError` before returning a Tree.
If `n` is not an integer, `Node.id(n)` SHALL fail with `TypeError`; if `n` is
negative, it SHALL fail with `ValueError`. Each failure SHALL return no marker.

#### Scenario: Identified leaves retain leaf structure

- **WHEN** a caller constructs `Tree(Node.id(2))`
- **THEN** its recipe contains one ordinary leaf marker
- **AND** its structural size is one

#### Scenario: Reject an invalid source identifier

- **WHEN** a caller supplies a negative or non-integer identifier to `Node.id`
- **THEN** construction raises a type or value error

### Requirement: Layouts map hierarchical coordinates to scalar indices

In `Layout.get_index(layout, key)`, `layout` is the Layout being evaluated and
`key` identifies either a hierarchical logical coordinate or a non-negative
integer identifying one logical coordinate in first-mode-fastest order.
`layout.index(key)` and `layout(key)` give `key` the same meaning.

Each indexing form SHALL return the same non-negative scalar index, equal to
the sum of each leaf coordinate multiplied by its corresponding stride. If the
`layout` argument to `Layout.get_index` is not a Layout, the operation SHALL
fail with `ValueError` before returning an index. The scalar index has no
intrinsic physical-storage meaning: a Tensor placement may use it to address a
carrier, while an adjacent Tensor layout may decode it as a coordinate in
another hierarchical shape. A coordinate SHALL preserve the shape hierarchy
or use a non-negative integer at any addressed hierarchy level; an integer key
SHALL expand in mixed-radix order with the first mode varying fastest.

#### Scenario: Index a nested coordinate

- **WHEN** `Layout(Shape([2, [3, 4]]), Stride([1, [10, 100]]))` is indexed by `[1, [2, 3]]`
- **THEN** the returned scalar index is `321`

#### Scenario: Expand a linear coordinate

- **WHEN** `Layout(Shape([3, 4]), Stride([2, 10]))` is indexed by the integer `5`
- **THEN** the integer expands to coordinate `[2, 1]`
- **AND** the returned scalar index is `14`

### Requirement: Layout indexing enforces the logical domain

Every coordinate component and linear key SHALL be non-negative and less than
the corresponding logical extent. For a negative, missing, extra, or
out-of-domain coordinate, indexing SHALL fail with `ValueError` before
returning a scalar index.

#### Scenario: Reject an out-of-domain coordinate

- **WHEN** `Layout(Shape([3, 4]), Stride([2, 10]))` is indexed by `[3, 0]`, `12`, or a negative coordinate
- **THEN** indexing raises a value error identifying that the key is outside the shape domain

### Requirement: Logical size and index-space cosize remain distinct

For a Shape, `size` is the count of logical coordinates described by its
extents. For a Layout, `size` is the size of its Shape and `cosize` is the
minimum origin-based scalar-index span containing every reachable scalar index.

Reading `shape.size` and reading `layout.size` SHALL return the product of all
Shape extents. Reading `layout.cosize` SHALL return one plus the greatest scalar
index reachable by a logical coordinate and therefore SHALL report the
Layout's minimum origin-based scalar-index span. A Tensor placement uses that
span when validating carrier storage, but storage is an interpretation supplied
by the Tensor rather than by Layout. Index gaps and stride-zero aliasing SHALL
NOT change the logical size.

#### Scenario: A gapped layout has a larger index span than size

- **WHEN** a caller inspects `Layout(Shape([2, 3]), Stride([1, 4]))`
- **THEN** `size` is `6`
- **AND** `cosize` is `10`

### Requirement: A layout profile records hierarchy only

For a Layout, `profile` is the ordered leaf-and-nesting recipe of its Shape
tree.

Reading `layout.profile` SHALL return that recipe as a list of public `Node`
markers. Extents and stride values SHALL NOT affect the returned profile, and
the query SHALL NOT flatten or reorder modes.

#### Scenario: Equal hierarchy yields equal profiles

- **WHEN** two layouts have shape trees `[2, [3, 4]]` and `[7, [8, 9]]` with arbitrary congruent strides
- **THEN** both profiles are `[Leaf, Push, Leaf, Leaf, Pop]`

### Requirement: Injectivity detects every scalar-index collision

For a Layout, `is_injective` asks whether its logical-coordinate to
scalar-index function has any collision.

Reading `layout.is_injective` SHALL return a Boolean that is true exactly when
distinct logical coordinates map to distinct scalar indices. It SHALL detect collisions caused by either
stride-zero broadcast modes or overlapping non-zero strides, while allowing
injective layouts whose scalar-index image contains gaps.

#### Scenario: Distinguish holes from collisions

- **WHEN** callers inspect layouts with shape/stride pairs `([2, 3], [1, 4])`, `([4, 2], [0, 1])`, and `([4, 2], [1, 1])`
- **THEN** the first layout is injective
- **AND** the two layouts with scalar-index collisions are not injective

### Requirement: Layout broadcasting preserves the shape profile

In `layout.broadcast_to(target)`, `layout` is the source Layout and `target`
describes the requested output Shape. Corresponding leaves are determined by
their position in the shared hierarchical profile.

`layout.broadcast_to(target)` SHALL require a Shape target with the same
hierarchical profile and SHALL return a new Layout whose Shape is `target`.
Each equal source and target extent SHALL retain its stride. When corresponding
extents differ, a source extent of one SHALL widen by changing its stride to
zero. If `target` is not a Shape, the operation SHALL fail with `TypeError`. If
the profile differs or any other extent would change, it SHALL fail with
`ValueError`. Each failure SHALL return no Layout. Broadcasting SHALL NOT
flatten, rank-align, insert, remove, or reorder modes.

#### Scenario: Widen a nested singleton leaf

- **WHEN** `Layout(Shape([2, [1, 3]]), Stride([1, [2, 2]]))` is broadcast to `Shape([2, [4, 3]])`
- **THEN** the result is `Layout(Shape([2, [4, 3]]), Stride([1, [0, 2]]))`

#### Scenario: Reject implicit structural alignment

- **WHEN** a flat three-mode layout is broadcast to a target that nests two of those modes
- **THEN** broadcasting raises a value error identifying the different shape profiles

### Requirement: Uniform preimage extent is an algebraic layout query

In `layout.uniform_preimage_extent(target_shape)`, `layout` supplies the source
coordinate space and scalar-index function, while `target_shape` supplies the
target coordinate space whose first-mode-fastest ordinal range is the intended
codomain.

For a layout `L`, the preimage of a scalar index `j` SHALL mean the set of
source coordinates `c` for which `L(c) = j`. A target `Shape` SHALL identify
its coordinate space with scalar indices from zero through `target.size - 1`
using the same mixed-radix convention as linear indexing. `L` SHALL be a
mapping onto that target only when every source coordinate maps inside that
scalar-index range. It SHALL be a surjection when, additionally, every target
scalar index has at least one preimage. The surjection SHALL be uniform when
every target coordinate is reached by exactly the same number `k` of source
coordinates. That number `k` SHALL be the uniform preimage extent.

`layout.uniform_preimage_extent(target_shape)` SHALL return `k` exactly when it
algebraically proves such a uniform surjection. It SHALL return `None` for
non-uniform collisions, gaps in the target scalar-index range, or the wrong
target size. If `target_shape` is not a Shape, the operation SHALL fail with
`TypeError` before returning a result. The result SHALL be determined without
enumerating logical coordinates.

#### Scenario: Prove uniform stride-zero replication

- **WHEN** `Layout(Shape([4, 3]), Stride([0, 1]))` is queried against `Shape(3)`
- **THEN** the returned uniform preimage extent is `4`
- **AND** each target scalar index `0`, `1`, and `2` has four source coordinates in its preimage

#### Scenario: Reject a non-uniform mapping

- **WHEN** `Layout(Shape([2, 2]), Stride([1, 1]))` is queried against `Shape(3)`
- **THEN** the result is `None`
- **AND** target scalar indices `0`, `1`, and `2` have unequal preimage cardinalities `1`, `2`, and `1`

### Requirement: Layout combination distinguishes concatenation from nesting

In `Layout.concat(first, second)` and `first + second`, `first` supplies the
leading top-level modes and `second` supplies the following top-level modes. In
`Layout.append(first, second)`, `second` is one sublayout to append as a single
top-level mode unless it is a leaf. In `layout[key]`, `key` identifies one
top-level mode or top-level slice of the source Layout.

`Layout.concat(first, second)` and `first + second` SHALL return equal Layouts
with every top-level mode of `second` following every top-level mode of
`first`. `Layout.append(first, second)` SHALL return a Layout that preserves a
non-leaf `second` as one nested top-level mode; for a leaf `second`, it SHALL
return the concatenation result. `layout[key]` SHALL return a Layout containing
the corresponding Shape and Stride subtree.

#### Scenario: Concatenate and append a nested layout

- **WHEN** layout `[1, 2]` is combined with layout `[3, 4]`
- **THEN** concatenation has shape `[1, 2, 3, 4]`
- **AND** append has shape `[1, 2, [3, 4]]`

### Requirement: Rearrangement uses explicit hierarchical trees

In `Layout.rearrange(layout, output, selection)`, `layout` is the source Layout,
`selection` identifies the source sublayouts available to the transformation,
and `output` describes the hierarchy constructed from those sublayouts and
inserted extent-one leaves.

When `selection` is not supplied, it SHALL default to extracting every source
leaf Layout in depth-first, left-to-right order. When `selection` is supplied,
rearrangement SHALL extract each source subtree corresponding to a
selection-tree leaf and preserve that subtree as one sublayout.
`Layout.rearrange` SHALL return a new Layout whose hierarchy is constructed
from the identified extracted sublayouts. An unidentified `Node.Leaf` in
`output` SHALL insert an extent-one, stride-zero mode. If `output` is not a
Tree, a supplied selection hierarchy differs from the source hierarchy, a
source identifier is duplicated or out of range, or an omitted extracted
sublayout has size greater than one, rearrangement SHALL fail with `ValueError`
before returning a Layout.

#### Scenario: Reorder sources and insert a singleton

- **WHEN** layout `Shape([2, 3])`, `Stride([1, 2])` is rearranged with no selection tree and output `Tree(Node.id(1), Node.Leaf, Node.id(0))`
- **THEN** the result has shape `[3, 1, 2]`
- **AND** the result has stride `[2, 0, 1]`

#### Scenario: Select a nested subtree as one source

- **WHEN** layout `Shape([1, [2, 3]])`, `Stride([5, [7, 14]])` is rearranged with selection `Tree(Node.Leaf, Node.Leaf)` and output `Tree(Node.id(1), Node.id(0))`
- **THEN** the selected nested subtree remains one extracted sublayout
- **AND** the result has shape `[[2, 3], 1]` and stride `[[7, 14], 5]`

#### Scenario: Reject missing non-singleton sources

- **WHEN** an output tree omits an extracted source layout whose logical size is greater than one
- **THEN** rearrangement raises a value error

### Requirement: Permutation reorders complete top-level modes

In `Layout.permute(layout, *order)`, `layout` is the source Layout and
`order[j]` identifies the source top-level mode that becomes top-level mode `j`
of the result. `order` may be supplied as variadic integers or as one iterable.

`Layout.permute(layout, *order)` SHALL return a new Layout with the requested
top-level mode order. The order SHALL contain every source top-level mode
exactly once. A nested mode SHALL move as one mode without flattening, and
applying the inverse order SHALL return the original Layout. If an order entry
is not an integer, permutation SHALL fail with `TypeError`. If the order omits,
duplicates, or names an out-of-domain mode, permutation SHALL fail with
`ValueError`. Each failure SHALL return no Layout.

#### Scenario: Permute a nested top-level mode

- **WHEN** layout shape `[2, [3, 4], 5]` and stride `[1, [2, 6], 24]` are permuted by `(1, 0, 2)`
- **THEN** the result has shape `[[3, 4], 2, 5]`
- **AND** the result has stride `[[2, 6], 1, 24]`

### Requirement: Layout composition preserves hierarchical function composition

In `Layout.compose(A, B)`, `A` is the outer Layout. A Layout `B` maps each
result coordinate to the logical coordinate consumed by `A`; a Shape `B`
describes one unit-stride tile for each corresponding leading mode of `A`; and
a Tiler `B` is an ordered `Sequence[Layout]` containing those leading-mode
tiles.

For a Layout `B`, `Layout.compose(A, B)` SHALL return a Layout whose scalar
index for coordinate `q` is `A(B(q))`. For a Shape `B`, it SHALL return the
result of applying one unit-stride tile to each corresponding leading mode. For
a Tiler `B`, it SHALL return the result of applying one tile to each
corresponding leading mode and preserving unmentioned trailing modes. Lists and
tuples containing equal tile Layouts SHALL return equal results.

#### Scenario: Compose two layouts

- **WHEN** `Layout(Shape(20), Stride(2))` is composed with `Layout(Shape([5, 4]), Stride([4, 1]))`
- **THEN** the result is `Layout(Shape([5, 4]), Stride([8, 2]))`

#### Scenario: Compose with list and tuple tilers

- **WHEN** the same valid tiles are supplied as a list and as a tuple
- **THEN** both composition results compare equal

### Requirement: Layout complement requires an injective congruent layout

In `Layout.complement(layout, cotarget)`, `layout` supplies the represented
coordinate factors and `cotarget` is the positive exclusive upper bound of the
scalar-index range that the combined factors must span.

`Layout.complement(layout, cotarget)` SHALL return a Layout covering the
coordinate factors omitted by `layout` so their combined Layout spans
`[0, cotarget)` without overlap. If `layout` is non-injective, incongruent with
`cotarget`, or larger than `cotarget`, complementation SHALL fail with
`ValueError` before returning a Layout.

#### Scenario: Complement a gapped injective layout

- **WHEN** `Layout(Shape([2, 2]), Stride([1, 6]))` is complemented to `24`
- **THEN** the result is `Layout(Shape([3, 2]), Stride([2, 12]))`

#### Scenario: Reject complement of an overlapping layout

- **WHEN** a layout with shape `[4, 2]` and stride `[1, 1]` is complemented
- **THEN** complementation raises a value error identifying self-overlap

### Requirement: Layout division exposes tile and remainder structure

In `Layout.divide(A, B)`, `A` is the source Layout and Layout `B` describes one
tile of its logical-coordinate domain. In `Layout.divide_tiler(A, B)` and
`Layout.zipped_divide(A, B)`, Tiler `B` is an ordered `Sequence[Layout]` whose
entries describe tiles for corresponding leading modes of `A`.

`Layout.divide(A, B)` SHALL return the Layout produced by composing `A` with
`B` and the complement of `B` over `A.size`, preserving the resulting tile and
remainder hierarchy. `Layout.divide_tiler(A, B)` SHALL return a Layout that
divides each corresponding leading mode and preserves unmentioned trailing
modes. `Layout.zipped_divide(A, B)` SHALL return a Layout that groups the tile
portions and remainder portions into separate leading hierarchical modes
before preserving unmentioned trailing modes. Lists and tuples containing
equal tile Layouts SHALL return equal division results.

#### Scenario: Divide one layout

- **WHEN** `Layout(Shape([4, 2, 3]), Stride([2, 1, 8]))` is divided by `Layout(Shape(4), Stride(2))`
- **THEN** the result is `Layout(Shape([[2, 2], [2, 3]]), Stride([[4, 1], [2, 8]]))`

#### Scenario: Divide with sequence tilers

- **WHEN** the same valid leading-mode tiler is supplied as a list and as a tuple to tiler or zipped division
- **THEN** the corresponding list and tuple results compare equal
