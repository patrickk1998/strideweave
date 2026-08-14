## Purpose

Define immutable shaped-domain maps to flat integer codomains, including sparse
permutations, bit swizzles, Cartesian products, and closed or structural
composition.

## ADDED Requirements

### Requirement: Index maps share one shaped-domain contract

An `IndexMap` maps every coordinate in a hierarchical `shape` to one integer in
a flat codomain. `shape` SHALL be a `Shape`; `size` SHALL equal `shape.size`;
and `codomain_size` SHALL be a positive integer exclusive upper bound such that
every returned index is in `[0, codomain_size)`. `codomain_size` SHALL describe
only that flat bound. The declared bound SHALL remain authoritative when the
map's image contains fewer distinct indices or occupies a smaller prefix.

The public `IndexMap` superclass SHALL expose read-only `shape`, `size`,
`codomain_size`, and `is_injective` properties, an `index(key)` method, an
equivalent direct-call form `map(key)`, and `outer.compose(inner)`. The public
concrete map kinds SHALL be `Layout`, `Permutation`, `Swizzle`, and `Product`.
Direct construction of the abstract `IndexMap` superclass SHALL fail with
`TypeError`; callers SHALL construct a concrete map kind instead. Reading
`shape` SHALL return the map's Shape, reading `size` and `codomain_size` SHALL
return integers, and reading `is_injective` SHALL return the tri-state result
defined below. `index(key)` and `map(key)` SHALL return the map's integer result
for `key`. Each evaluation form SHALL require exactly one key; an arity
mismatch SHALL raise `TypeError`. A generic composition result SHALL satisfy
`isinstance(result, IndexMap)`.

`is_injective` SHALL be `True` when the map is proven injective, `False` when a
collision is proven, and `None` when neither result is established.

#### Scenario: A declared codomain may contain unused indices

- **WHEN** a map has `shape == Shape(2)`, returns indices `4` and `3`, and declares `codomain_size == 10`
- **THEN** its `size` is `2`
- **AND** its flat codomain is `[0, 10)` even though its image is `{3, 4}`

#### Scenario: Call syntax is index syntax

- **WHEN** any IndexMap instance is evaluated with a valid key using `index(key)` and direct call syntax
- **THEN** both forms return the same integer

#### Scenario: Reject direct abstract construction

- **WHEN** a caller attempts to construct `IndexMap` directly
- **THEN** construction raises `TypeError` and returns no map

### Requirement: Shapes decode and encode coordinates first-mode-fastest

In `shape.decode(index)`, `index` is a non-negative scalar ordinal in the
coordinate space described by `shape`. In `shape.encode(coordinate)`,
`coordinate` is a valid hierarchical coordinate for `shape`. At any hierarchy
node, a non-negative scalar coordinate SHALL stand for the complete subtree at
that node and SHALL decode recursively using the same rule.

For sibling mode sizes `s0, s1, ...`, encoding SHALL use first-mode-fastest order:
`c0 + s0 * (c1 + s1 * (...))`, where a nested mode's coordinate is first
encoded within that subtree. `shape.decode(index)` SHALL return the congruent
nested tuple coordinate, and `shape.encode(coordinate)` SHALL return its scalar
ordinal. The two operations SHALL be inverses throughout the shape domain.

`shape.decode(index)` SHALL require an integer `index` and SHALL return the
nested tuple coordinate identified by that ordinal. `shape.encode(coordinate)`
SHALL require `coordinate` to be an integer, tuple, or list and SHALL return its
integer ordinal. An IndexMap key SHALL have those same accepted kinds and SHALL
represent either the complete coordinate as an integer or a tuple/list that
preserves the shape hierarchy while optionally replacing any subtree with its
integer ordinal. All equivalent forms SHALL return the same map index.

A decode argument or encode/map key of another kind SHALL raise `TypeError`.
A Shape encode or decode call with zero or more than one argument SHALL raise
`TypeError`.
A negative or out-of-domain integer, a component outside its corresponding
extent, or a structurally incomplete or excessive tuple/list SHALL raise
`ValueError` before returning a coordinate or index.

#### Scenario: Decode a scalar nested coordinate

- **WHEN** `Shape(5, [20, 4])` normalizes coordinate `(4, 42)`
- **THEN** the nested scalar `42` decodes to `(2, 2)`
- **AND** the complete coordinate is `(4, (2, 2))`
- **AND** its encoded ordinal is `214`

#### Scenario: A complete scalar is equivalent to the hierarchical coordinate

- **WHEN** a map with `shape == Shape(5, [20, 4])` is indexed by `214`, `(4, 42)`, and `(4, (2, 2))`
- **THEN** all three keys return the same map index

#### Scenario: Reject an invalid subtree scalar

- **WHEN** `Shape(5, [20, 4])` is given coordinate `(4, 80)` or a coordinate with missing or extra modes
- **THEN** coordinate encoding and map indexing raise a value error and return no result

#### Scenario: Reject an invalid coordinate argument kind

- **WHEN** `shape.decode` receives a non-integer or `shape.encode` or map indexing receives a value that is not an integer, tuple, or list
- **THEN** the operation raises `TypeError` and returns no result

### Requirement: Rank-zero shapes contain one coordinate

`Shape()` SHALL describe a rank-zero domain of size one. Its sole coordinate
SHALL be accepted as scalar `0`, empty tuple `()`, or empty list `[]`.
`Shape().decode(0)` SHALL return `()`, and `Shape().encode(())` SHALL return
`0`. A scalar other than zero or a non-empty structured coordinate SHALL raise
`ValueError` and return no result.

#### Scenario: Evaluate a rank-zero map

- **WHEN** a map has `shape == Shape()` and is indexed by `0`, `()`, and `[]`
- **THEN** all three forms identify its one logical coordinate

### Requirement: Permutations are explicit partial injections

In `Permutation(values, codomain_size)`, `values[i]` is the flat result for
scalar domain ordinal `i`, and `codomain_size` is the required flat exclusive
upper bound. `values` SHALL be a finite sequence whose entries are integers,
and `codomain_size` SHALL be a required integer argument. Supplying another
argument kind, an entry of another kind, or any constructor arity other than
the declared two inputs SHALL raise `TypeError`. The sequence SHALL contain at
least one entry; every entry SHALL be non-negative and less than the positive
`codomain_size`; and entries SHALL be pairwise distinct. An empty sequence,
duplicate value, negative value, out-of-codomain value, or non-positive
codomain size SHALL raise `ValueError` before returning a map.

Construction SHALL return an immutable `Permutation` with
`shape == Shape(len(values))`, a read-only tuple `values`, the declared
`codomain_size`, and `is_injective is True`. Its index for coordinate `q` SHALL
be `values[shape.encode(q)]`.

#### Scenario: Preserve gaps in a permutation codomain

- **WHEN** a caller constructs `Permutation([4, 3], codomain_size=10)`
- **THEN** coordinates `0` and `1` return `4` and `3`
- **AND** `shape` is `Shape(2)`, `size` is `2`, and `codomain_size` remains `10`

#### Scenario: Reject a repeated target

- **WHEN** a caller constructs a Permutation whose values contain the same integer twice
- **THEN** construction raises a value error and returns no map

### Requirement: Swizzle stages are invertible XOR field transforms

In `SwizzleStage(bits, base, shift)`, `bits` is the positive width of each of
two bit fields, `base` is the non-negative position of the lower field, and
non-zero signed `shift` gives both the field separation and XOR direction. The
two fields SHALL be disjoint, so `abs(shift) >= bits`.

For `shift > 0`, a stage SHALL XOR the `bits`-wide field beginning at
`base + shift` into the field beginning at `base`. For `shift < 0`, it SHALL
XOR the field beginning at `base` into the field beginning at
`base - shift`. The source field remains unchanged, so applying the same stage
twice SHALL be identity. A stage SHALL expose read-only `bits`, `base`, and
`shift` values, and successful construction SHALL return that immutable
SwizzleStage. Non-integer arguments or constructor arity other than the declared
three inputs SHALL fail with `TypeError`; a non-positive width, negative base,
zero shift, or overlapping fields SHALL fail with `ValueError`.

#### Scenario: Apply an XOR field stage

- **WHEN** `Swizzle(Shape(16), SwizzleStage(bits=2, base=0, shift=2))` evaluates binary index `1101`
- **THEN** the upper field `11` is XORed into the lower field `01`
- **AND** the result is binary index `1110`

### Requirement: Swizzles compose structured stages over fixed binary spaces

In `Swizzle(shape, *stages)`, `shape` supplies the domain and every `stage`
SHALL be a `SwizzleStage` applied in argument order. `shape.size` SHALL be a
power of two, including one, and the returned immutable Swizzle SHALL have
`codomain_size == size`, a read-only tuple `stages`, and
`is_injective is True`. Every stage field SHALL fit within the
`log2(shape.size)`-bit index width. A non-Shape domain or non-SwizzleStage stage
SHALL fail with `TypeError`; omitting the required `shape` SHALL also fail with
`TypeError`. A non-power-of-two domain or out-of-width stage SHALL fail with
`ValueError`.

Zero stages SHALL produce the identity Swizzle. When two compatible Swizzles
have equal size, `outer.compose(inner)` SHALL return a Swizzle whose evaluation
applies the inner stages followed by the outer stages. Adjacent equal stages
SHALL cancel, including to the zero-stage identity. Compatible Swizzles with
different sizes SHALL use generic composition rather than return a Swizzle.

#### Scenario: Construct the one-point identity swizzle

- **WHEN** a caller constructs `Swizzle(Shape())` with no stages
- **THEN** the result maps its sole coordinate to zero and has an empty stages tuple

#### Scenario: Cancel two equal stages

- **WHEN** a Swizzle containing one valid stage is composed with an equal Swizzle
- **THEN** the result is a zero-stage Swizzle over the same shape

#### Scenario: Reject a stage outside the binary domain

- **WHEN** a caller supplies overlapping stage fields or a stage whose upper field exceeds the Swizzle domain bit width
- **THEN** construction raises a value error and returns no Swizzle

#### Scenario: Use Layout identity outside binary domains

- **WHEN** an identity map is required for a domain whose size is not a power of two
- **THEN** a compact first-mode-fastest Layout represents that identity
- **AND** constructing a Swizzle for that domain raises a value error

### Requirement: Products pack child maps as a Cartesian map

In `Product(*children)`, each child is an IndexMap and supplies one ordered
Cartesian mode. Construction SHALL require at least two children and SHALL
return an immutable Product exposing them as a read-only tuple `children`.
Supplying fewer than two children SHALL fail with `ValueError`; supplying a
non-IndexMap child SHALL fail with `TypeError`.

The Product `shape` SHALL contain each child shape as its corresponding mode,
preserving the hierarchy of a non-leaf child shape. Its read-only
`target_shape` SHALL contain each ordinary child's `codomain_size` as the
corresponding target mode and SHALL preserve a child Product's target hierarchy
as a nested mode. `codomain_size` SHALL equal `target_shape.size`.

For a Product coordinate, each child SHALL evaluate its corresponding
coordinate and the Product SHALL encode the ordered child results in
`target_shape` using first-mode-fastest order. `is_injective` SHALL be `False` if
any child is non-injective, `True` if every child is injective, and `None`
otherwise.

#### Scenario: Pack two sparse selections

- **WHEN** `p0 == Permutation([4, 3], 10)` and `p1 == Permutation([2, 9], 15)` are combined as `Product(p0, p1)`
- **THEN** the Product has `shape == Shape(2, 2)` and `target_shape == Shape(10, 15)`
- **AND** coordinate `(0, 0)` evaluates child coordinate `(4, 2)` and returns `24`
- **AND** coordinate `(1, 1)` evaluates child coordinate `(3, 9)` and returns `93`

#### Scenario: Preserve explicit Product nesting

- **WHEN** the same maps are constructed as `Product(Product(a, b), c)` and `Product(a, b, c)`
- **THEN** the first Product preserves the first two domain and target modes as nested modes
- **AND** the second Product has three flat top-level domain and target modes

#### Scenario: Reject an invalid Product constructor

- **WHEN** a caller supplies fewer than two children or includes a value that is not an IndexMap
- **THEN** construction raises the specified value or type error and returns no Product

### Requirement: Composition uses declared codomain containment

In `outer.compose(inner)`, `inner` supplies the result domain and is evaluated
first, while `outer` consumes its flat result as a scalar coordinate.
Composition SHALL require both operands to be IndexMaps and SHALL require
`inner.codomain_size <= outer.size`. A non-IndexMap operand SHALL fail with
`TypeError`; a declared inner codomain larger than the outer domain SHALL fail
with `ValueError` before returning a map.

For a compatible pair, the result SHALL have `shape == inner.shape`,
`codomain_size == outer.codomain_size`, and evaluation
`outer.compose(inner)(q) == outer(inner(q))`. Compatibility SHALL depend on the
declared bound. Instance composition SHALL require exactly one `inner`
argument; supplying zero or more than one SHALL raise `TypeError`. Successful
composition SHALL return the specialized or generic IndexMap established by
the requirements below.

#### Scenario: Compose through a smaller declared codomain

- **WHEN** an inner map declares `codomain_size == 8` and an outer map has `size == 10`
- **THEN** `outer.compose(inner)` succeeds and follows function-composition order

#### Scenario: Reject a codomain that exceeds the outer domain

- **WHEN** an inner map declares `codomain_size == 11` and an outer map has `size == 10`
- **THEN** composition raises a value error and returns no map

#### Scenario: Reject invalid composition arity

- **WHEN** a caller invokes instance composition with zero or more than one inner argument
- **THEN** composition raises `TypeError` and returns no map

### Requirement: Closed compositions retain specialized map kinds

Compatible Permutation composition SHALL return a Permutation whose values are
the outer lookup at each inner value and whose `codomain_size` is the outer
bound. Equal-size Swizzle composition SHALL return the staged Swizzle described
by the Swizzle requirement.

Compatible Product composition SHALL return a Product of component
compositions only when the two explicit Product trees have equal arity at each
aligned Product node and every aligned child composition is compatible. The
result SHALL preserve that explicit tree. A Product pair that is compatible as
flat maps but lacks that structural alignment SHALL use generic composition.

#### Scenario: Compose two permutations by lookup

- **WHEN** compatible Permutations `outer` and `inner` are composed
- **THEN** result value `i` equals `outer.values[inner.values[i]]`
- **AND** the result is a Permutation with the outer codomain size

#### Scenario: Lower an aligned Product composition componentwise

- **WHEN** outer and inner Products have matching explicit Product trees and compatible corresponding children
- **THEN** their composition is a Product containing each corresponding child composition

### Requirement: Other compatible compositions return a generic IndexMap

A compatible composition that is not closed under a specialized result rule
SHALL return an immutable IndexMap whose evaluation preserves the composition
law and whose public type contract is IndexMap.

The generic result SHALL report `is_injective is False` when the inner map is
known non-injective, `True` when every composed child is known injective, and
`None` otherwise.

An identity map is an IndexMap whose `codomain_size == size` and whose result is
`shape.encode(q)` for every coordinate `q`. Composition SHALL return the
non-identity operand unchanged when doing so preserves the required result
`shape` and `codomain_size`; otherwise it SHALL preserve the same function in
the appropriate specialized or generic result.

#### Scenario: Compose a Layout and Swizzle structurally

- **WHEN** a compatible Layout and Swizzle are composed and no specialized closure rule applies
- **THEN** the result is an IndexMap with the inner shape and outer codomain size
- **AND** evaluating coordinate `q` returns `outer(inner(q))`

#### Scenario: Preserve an operand across compatible identity composition

- **WHEN** an identity composition has the same result shape and codomain size as its non-identity operand
- **THEN** composition returns that non-identity map unchanged

#### Scenario: Preserve unknown injectivity conservatively

- **WHEN** a generic composition has an injective inner map and an outer map whose injectivity is unknown
- **THEN** the composed map reports `is_injective is None`

### Requirement: Index map values are immutable

Every public IndexMap, SwizzleStage, and publicly exposed child or stage
sequence SHALL be immutable after successful construction. Assigning or
deleting a semantic public field SHALL fail with `AttributeError`. A
constructed map SHALL own immutable copies of collection inputs so its public
values and evaluations remain stable when the caller later mutates those
inputs.

#### Scenario: Constructor inputs do not remain mutable aliases

- **WHEN** a caller mutates a list previously supplied as Permutation values or Product children
- **THEN** the constructed map's values, children, and evaluations remain unchanged
