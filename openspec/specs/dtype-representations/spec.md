---
title: Dtype Representations
publish: true
status: stable
order: 30
summary: Compound storage schemas, declarative representation rules, and block-scaled dtype structure.
---

# dtype-representations Specification

## Purpose

Define how compound dtypes describe ordered storage planes and reusable logical
representation constraints, including the built-in block-scaled formats.

## Terminology

| Term | Meaning |
| --- | --- |
| plane | One ordered homogeneous physical-storage component of a compound logical value, backed by one carrier whose storage dtype is identical to the plane's `SimpleDType`. |
| storage schema | The ordered tuple of `SimpleDType` identities that fixes the number, position, and required storage dtype of a compound dtype's planes. |
| representation rule | An immutable declarative logical-validity constraint that contributes to compound dtype identity and is evaluated against read-only representation facts only after universal Tensor-representation validation. |
| level | One position in a block-scaled dtype's inner-to-outer scale chain, pairing a simple-dtype scale plane with a grouping extent measured in the coordinate space immediately below it. |
| adjacent layout | An ordinary `Layout` that maps coordinates at representation level `i` to integer coordinates in level `i + 1`, expressing logical grouping rather than physical placement. |
| `Whole` | The singleton symbolic extent that groups an entire source level into one target coordinate. |

## Requirements

### Requirement: Compound dtypes own an ordered storage schema

A concrete subclass of `CompoundDType` SHALL declare `abstract=False` and call
`CompoundDType(name, *, supertype=None, simple_types,
representation_rules=())`. `name` names the compound dtype's unique registry
key and SHALL satisfy the registered-name contract in `dtype-descriptors`.
`supertype` names the compound dtype's immediately enclosing category; it SHALL
be optional, SHALL default to `None`, and when provided SHALL be a registered
`DTypeCategory`. `simple_types` names the ordered simple-dtype plane required at
each representation position and SHALL be a non-empty iterable containing only
registered `SimpleDType` descriptors. `representation_rules` names the ordered
logical representation constraints applied after universal validation; it
SHALL be optional, SHALL default to the empty sequence, and SHALL be an iterable
containing only `RepresentationRule` objects.

Construction SHALL copy `simple_types` and `representation_rules` into
immutable tuples owned by the descriptor. When `supertype` is neither `None`
nor a registered `DTypeCategory`, construction SHALL fail with `TypeError`.
When `simple_types`
is non-iterable or contains an invalid element, construction SHALL fail with
`TypeError`; when `simple_types` is empty, construction SHALL fail with
`ValueError`. When `representation_rules` is non-iterable or contains an
invalid element, construction SHALL fail with `TypeError`. Each failure SHALL
occur before `name` is registered. `simple_types` SHALL return the owned storage
schema, `num_carriers` SHALL return its length, `representation_rules` SHALL
return the owned rule tuple, and `is_compound()` SHALL return `True`.

#### Scenario: Construct a planar extension

- **WHEN** a concrete compound subclass supplies two registered simple dtypes
  and no representation rules
- **THEN** it registers with those two planes in order, `num_carriers == 2`,
  and an empty `representation_rules` tuple

#### Scenario: Isolate descriptor state from caller mutation

- **WHEN** the caller mutates a collection previously supplied as planes or
  rules
- **THEN** the registered descriptor's schema, rules, structure, and carrier
  count remain unchanged

### Requirement: Compound dtypes describe logical values rather than homogeneous storage

A `CompoundDType` SHALL identify a logical value assembled from one carrier per
entry of `simple_types`, in the same order. The compound descriptor itself
SHALL NOT be a homogeneous carrier storage dtype. Tensor-side enforcement of
the schema and rules is defined by `core-tensor-representation`; carrier-side
storage rejection is defined by `carrier-storage`.

#### Scenario: Interpret plane position

- **WHEN** a compound dtype reports schema `(D_0, D_1, ..., D_n)`
- **THEN** representation plane `i` requires storage dtype identical to `D_i`

### Requirement: Representation rules have immutable structural identity

`RepresentationRule` SHALL be subclassed to define a declarative constraint.
The rule class identity and its `structure_extension()` result SHALL determine
the immutable value returned by `structure()`. `structure_extension()` SHALL
default to `()` and SHALL return a tuple containing only permitted descriptor
structure values; another return type or an unsupported value SHALL fail with
`TypeError` during rule construction.

A rule subclass SHALL implement `validate(context)`. `context` names the
read-only universally validated representation facts against which the rule is
evaluated and SHALL satisfy `RepresentationValidationContext`. Successfully
completed validation SHALL return `None`. After construction, assigning or
deleting rule attributes SHALL fail with `AttributeError`, and shallow or deep
copy SHALL return the same rule object. A rule subclass that shadows
framework-owned identity or immutability members SHALL fail with `TypeError`
when defined.

#### Scenario: Construct a reusable rule

- **WHEN** a rule subclass supplies immutable state through
  `structure_extension()` and implements `validate(context)`
- **THEN** the constructed rule has a stable structure and can be reused by
  more than one compound dtype

#### Scenario: Refuse mutation after rule construction

- **WHEN** a caller assigns or deletes an attribute on a finalized rule
- **THEN** the operation fails with `AttributeError`

### Requirement: Rules consume a read-only validated context

`RepresentationValidationContext` SHALL expose `logical_dtype`, the ordered
`storage_dtypes`, the ordered `placement_layouts`, the ordered
`adjacent_layouts`, and the ordered `level_shapes`. Each sequence SHALL be an
immutable tuple. The context SHALL describe facts already accepted by the
universal validation in `core-tensor-representation`.

A representation rule SHALL inspect that context without mutating descriptors,
layouts, carriers, the context, or other external state. Rule exceptions SHALL
propagate as representation-construction failures as specified by
`core-tensor-representation`.

#### Scenario: Inspect validated representation facts

- **WHEN** a rule's `validate` method is invoked
- **THEN** every public context field reports the corresponding universally
  validated logical dtype, storage schema, layouts, or level shapes

### Requirement: Whole is one persistent symbolic extent

`WholeExtent()` SHALL return the singleton object exported as `Whole`.
Constructing, copying, deep-copying, or unpickling a whole extent SHALL return
that same identity. `repr(Whole)` SHALL be `Whole`.

#### Scenario: Reconstruct Whole

- **WHEN** a caller constructs or deserializes a `WholeExtent`
- **THEN** the result is identical to `Whole`

### Requirement: A Level defines one scale dtype and grouping extent

`Level(scale, block)` SHALL return an immutable, hashable level. `scale` names
the `SimpleDType` stored at that scale level. `block` names the positive number
of coordinates from the preceding level grouped under one scale, or `Whole`
for a single scale covering that complete level.

When `scale` is not a `SimpleDType`, construction SHALL fail with `TypeError`.
When `block` is neither an integer other than `bool` nor `Whole`, construction
SHALL fail with `TypeError`; when integer `block` is zero or negative,
construction SHALL fail with `ValueError`. `is_whole()` SHALL return whether
`block is Whole`.

#### Scenario: Construct a fixed block level

- **WHEN** a caller constructs `Level(DType.E8M0, 32)`
- **THEN** it records that scale identity and a block extent of 32 and reports
  `is_whole() == False`

#### Scenario: Construct a whole level

- **WHEN** a caller constructs a level whose block is `Whole`
- **THEN** it reports `is_whole() == True`

### Requirement: LevelExtent validates uniform logical grouping

`LevelExtent(level, extent)` SHALL construct an immutable representation rule.
`level` names the zero-based adjacent edge and SHALL be a non-negative exact
integer. `extent` names the required number of source coordinates grouped under
each target coordinate, or the complete source level when it is `Whole`.
`extent` SHALL be a positive exact integer or `Whole`. When `level` or `extent`
has an invalid type, construction SHALL fail with `TypeError`; when integer
`level` or `extent` has an invalid range, construction SHALL fail with
`ValueError`. The `level` and `extent` properties SHALL return those values.

For integer `extent`, `validate(context)` SHALL require the source cardinality
to be exactly `extent` times the target cardinality and each target coordinate
to have exactly `extent` source preimages. For `Whole`, it SHALL require one
target coordinate with every source coordinate as a preimage. A missing edge,
cardinality mismatch, or non-uniform preimage count SHALL fail with
`ValueError`.

#### Scenario: Accept fixed uniform grouping

- **WHEN** the selected adjacent layout maps exactly 32 source coordinates to
  every target coordinate and the source cardinality is 32 times the target
- **THEN** `LevelExtent(level, 32).validate(context)` returns `None`

#### Scenario: Accept whole-level grouping

- **WHEN** the selected adjacent layout maps every source coordinate to one
  target coordinate
- **THEN** `LevelExtent(level, Whole).validate(context)` returns `None`

### Requirement: BlockScaledDType derives its complete plane chain

`BlockScaledDType(name, *, element, levels)` SHALL construct a compound dtype
whose logical value is its encoded element multiplied by one scale from each
level. `name` names the block-scaled format's unique registry key and SHALL
satisfy the registered-name contract in `dtype-descriptors`. `element` names
the simple dtype of the encoded logical element and SHALL be a
`SimpleDType`. `levels` names the ordered scale and grouping chain and SHALL be
a non-empty iterable of `Level` values ordered from innermost to outermost;
only the final level MAY use `Whole`.

When `element` is invalid or `levels` contains an invalid entry, construction
SHALL fail with `TypeError`. When `levels` is empty or contains a non-final
`Whole` level, construction SHALL fail with `ValueError` before `name` is
registered. The `element` and `levels` properties SHALL return the normalized
immutable values supplied through `element` and `levels`. `simple_types` SHALL
equal `element` followed by each level's scale,
and `representation_rules` SHALL contain
`LevelExtent(i, levels[i].block)` for every level in order. Its supertype SHALL
be `DType.Any`.

Two block-scaled descriptors with the same complete element and level-chain
structure SHALL be one forbidden duplicate representation even when their
names or implementation subclasses differ; attempting to construct the second
SHALL fail with `ValueError`.

#### Scenario: Derive planes and rules

- **WHEN** a block-scaled dtype has element `E2M1` and levels using `E4M3` then
  `Float32`
- **THEN** its planes are `(E2M1, E4M3, Float32)` and it has one corresponding
  `LevelExtent` rule per scale level

#### Scenario: Reject a structurally duplicate format

- **WHEN** a caller constructs a second block-scaled descriptor with a complete
  element and level chain already registered
- **THEN** construction fails with `ValueError` and the original remains the
  unique identity for that representation

### Requirement: Block-scaled properties report axes and storage cost

`num_axes` SHALL return the count of integer-block levels and SHALL exclude a
final `Whole` level. `bits_per_element` SHALL include the element width and
each fixed scale width divided by the cumulative block extents below it.

When every level has an integer block, `bits_per_element` SHALL return the
concrete floating-point cost. When the final level is `Whole`, it SHALL return
`SymbolicBits(constant, whole_scale_bits)`, where `constant` is the cost before
the whole scale and `whole_scale_bits` is that final scale width.

`SymbolicBits(constant, whole_scale_bits)` SHALL construct and return an
immutable storage-cost expression. `constant` names the fixed bits per logical
element before the whole-tensor scale contribution. `whole_scale_bits` names
the bit width of that whole-tensor scale. The `constant` and
`whole_scale_bits` properties SHALL return those supplied values.

`SymbolicBits.evaluate(element_count)` SHALL return
`constant + whole_scale_bits / element_count`. `element_count` names the
positive number of logical elements across which the whole-tensor scale cost is
amortized. `element_count` SHALL be an integer other than `bool`; when it is
not, evaluation SHALL fail with `TypeError`. When `element_count` is below one,
evaluation SHALL fail with `ValueError`.

#### Scenario: Evaluate a whole-scale cost

- **WHEN** `DType.NVFP4.bits_per_element.evaluate(1024)` is called
- **THEN** it returns `4.53125`

### Requirement: Built-in block-scaled formats have fixed structures

The built-in formats SHALL be `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP6_E3M2`,
`MXFP6_E2M3`, `MXFP4`, `MXINT8`, and `NVFP4`.

Each MX format SHALL use one `E8M0` scale per block of 32 encoded elements.
Their element encodings SHALL respectively be `E4M3`, `E5M2`, `E3M2`, `E2M3`,
`E2M1`, and `Int8`. `NVFP4` SHALL use `E2M1` elements, one `E4M3` scale per
block of 16, and one `Float32` scale for `Whole`.

#### Scenario: Inspect an MX representation

- **WHEN** a caller inspects `DType.MXFP4`
- **THEN** its element is `DType.E2M1` and its sole level uses `DType.E8M0`
  with block extent 32

#### Scenario: Inspect NVFP4

- **WHEN** a caller inspects `DType.NVFP4`
- **THEN** its scale levels are `(Level(DType.E4M3, 16),
  Level(DType.Float32, Whole))` in that order
