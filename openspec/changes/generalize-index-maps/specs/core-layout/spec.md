## MODIFIED Requirements

### Requirement: Logical size and index-space cosize remain distinct

For a Shape, `size` is the count of logical coordinates described by its
extents. For a Layout, `size` is the size of its Shape, `cosize` is the minimum
origin-based scalar-index span containing every reachable scalar index, and
`codomain_size` is the flat exclusive upper bound inherited from IndexMap.

Reading `shape.size` and reading `layout.size` SHALL return the product of all
Shape extents. Reading `layout.cosize` SHALL return one plus the greatest scalar
index reachable by a logical coordinate and therefore SHALL report the
Layout's minimum origin-based scalar-index span. Reading
`layout.codomain_size` SHALL return exactly `layout.cosize`. A Tensor placement
uses that span when validating carrier storage, but storage is an
interpretation supplied by the Tensor rather than by Layout. The logical size
SHALL remain the product of Shape extents when the Layout image has index gaps
or stride-zero aliases.

#### Scenario: A gapped layout has a larger index span than size

- **WHEN** a caller inspects `Layout(Shape([2, 3]), Stride([1, 4]))`
- **THEN** `size` is `6`
- **AND** `cosize` and `codomain_size` are both `10`

### Requirement: Layout composition preserves hierarchical function composition

In `outer.compose(inner)` and the equivalent class-qualified
`Layout.compose(outer, inner)`, `outer` is the Layout evaluated second and
`inner` is the value evaluated or applied first. An IndexMap `inner` maps each
result coordinate to the scalar coordinate consumed by `outer`; a Shape
`inner` describes one unit-stride tile for each corresponding leading mode of
`outer`; and a Tiler `inner` is an ordered `Sequence[Layout]` containing those
leading-mode tiles. A list, tuple, or other Sequence whose every member is a
Layout SHALL be interpreted as a Tiler.

For an IndexMap `inner`, composition SHALL follow the common IndexMap
compatibility and evaluation contract. When `inner` is a Layout, composition
SHALL return the Layout representation of that affine hierarchical function.
When a compatible sibling IndexMap has no specialized closure with Layout,
composition SHALL return the generic IndexMap result.

For a Shape `inner`, composition SHALL return a Layout produced by applying one
unit-stride tile to each corresponding leading mode. For a Tiler `inner`, it
SHALL return a Layout produced by applying one tile to each corresponding
leading mode and preserving unmentioned trailing modes. Lists and tuples
containing equal tile Layouts SHALL return equal results. These Shape and Tiler
forms SHALL remain Layout-specific conveniences rather than IndexMap operands.

The instance form SHALL require exactly one `inner` argument, and the
class-qualified form SHALL require exactly one `outer` and one `inner` argument;
an arity mismatch SHALL raise `TypeError`. A class-qualified `outer` of another
type, an `inner` that is neither an IndexMap nor Shape nor Tiler, or a Tiler
containing a non-Layout member SHALL raise `TypeError`. The common IndexMap
failure contract SHALL govern IndexMap bounds. A Shape or Tiler with more
top-level tiles than `outer`, or a tile that cannot compose with its
corresponding leading Layout mode, SHALL raise `ValueError`. Every failure SHALL
return no map.

#### Scenario: Compose two layouts

- **WHEN** `Layout(Shape(20), Stride(2))` is composed with `Layout(Shape([5, 4]), Stride([4, 1]))`
- **THEN** the result is `Layout(Shape([5, 4]), Stride([8, 2]))`

#### Scenario: Compose with list and tuple tilers

- **WHEN** the same valid tiles are supplied as a list and as a tuple
- **THEN** both composition results compare equal

#### Scenario: Compose Layout with a sibling map

- **WHEN** a Layout is composed with a compatible Swizzle or Permutation and no specialized closure rule applies
- **THEN** the result is a generic IndexMap that evaluates the sibling first and the Layout second

#### Scenario: Preserve trailing modes for an empty tiler

- **WHEN** a Layout is composed with an empty Shape, list, or tuple
- **THEN** composition returns a Layout equal to the outer Layout

#### Scenario: Reject an invalid Layout composition input

- **WHEN** class-qualified composition receives a non-Layout outer, a Tiler contains a non-Layout member, or a Shape or Tiler supplies too many or incompatible tiles
- **THEN** composition raises the specified type or value error and returns no map

#### Scenario: Reject invalid Layout composition arity

- **WHEN** either Layout composition syntax receives fewer or more arguments than its declared form
- **THEN** composition raises `TypeError` and returns no map

## ADDED Requirements

### Requirement: Layout is the affine IndexMap specialization

Every Layout SHALL be an IndexMap whose `shape` is its hierarchical logical
domain, whose `codomain_size` equals its Layout-specific `cosize`, and whose
coordinate evaluation and exact Boolean `is_injective` answer satisfy the
common IndexMap contract. `Layout` SHALL remain the type used for Tensor
physical placement and adjacent logical grouping. Those Tensor roles SHALL
continue to require a Layout.

#### Scenario: Layout participates in the public map hierarchy

- **WHEN** a caller constructs any valid Layout
- **THEN** it satisfies `isinstance(layout, IndexMap)`
- **AND** its existing shape, stride, indexing, profile, broadcasting, complement, and tiling operations remain available

### Requirement: Layout value graphs are immutable

A successfully constructed Shape, Stride, or Layout SHALL expose immutable
semantic state. Assigning or deleting its hierarchy, size, depth, shape,
stride, or other semantic fields SHALL fail with `AttributeError`. Its public
shape, stride, evaluation, and size properties SHALL remain stable for the
value's lifetime.

#### Scenario: Semantic reassignment is rejected

- **WHEN** a caller attempts to replace a Layout's shape or stride or a Shape's logical size
- **THEN** assignment raises an attribute error
- **AND** subsequent Layout indexing returns the same result as before the attempt
