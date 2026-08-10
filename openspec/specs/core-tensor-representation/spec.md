---
title: Core Tensor Representation
publish: true
status: stable
order: 2
summary: Tensor construction, authoritative representation, validation, carrier ownership, and multi-subtensor boundaries.
---

# core-tensor-representation Specification

## Purpose

Define how one logical Tensor is constructed from carrier-backed storage and
how its ordered representation is validated, observed, owned, and restricted.

## Terminology

| Term | Meaning |
| --- | --- |
| logical dtype | The dtype of each logical value exposed by a Tensor. |
| storage dtype | The dtype of values held by one constituent carrier. |
| storage schema | The logical dtype's ordered sequence of required storage dtypes. |
| Tensor offset | The non-negative base carrier index to which a placement layout's linear index is added. |
| subtensor | One storage dtype, carrier, Tensor offset, and placement layout in a Tensor representation. |
| placement layout | A Layout whose linear index selects a carrier value relative to a Tensor offset. |
| adjacent layout | A Layout whose linear index identifies a logical coordinate in the next representation level. |

## Requirements

### Requirement: Conventional Tensor construction creates one subtensor

In `Tensor(carrier, offset, layout)`, `carrier` supplies the stored values and
their dtype, `offset` identifies the carrier index corresponding to layout
linear index zero, and `layout` maps each Tensor logical coordinate to the
linear index added to `offset`.

`Tensor(carrier, offset, layout)` SHALL return a new Tensor backed by the
conventional one-subtensor representation. Its logical dtype and sole storage
dtype SHALL be the carrier's dtype, its sole subtensor SHALL retain the
supplied carrier, offset, and placement Layout, and it SHALL have no adjacent
layouts.

The offset SHALL be a non-negative integer. If the offset is not an integer,
construction SHALL fail with `TypeError`. If the offset is negative or
`offset + layout.cosize` exceeds the carrier size, construction SHALL fail with
`ValueError`. Each failure SHALL occur before returning a Tensor and SHALL NOT
mutate or release the carrier.

#### Scenario: Construct a conventional Tensor
- **WHEN** a caller supplies a carrier, a non-negative offset, and a `Layout`
  whose addressed scalar-index span fits the carrier
- **THEN** the resulting Tensor has exactly one subtensor with that carrier,
  offset, and placement layout, the carrier dtype as its logical and storage
  dtype, and zero adjacent layouts

#### Scenario: Reject a negative offset
- **WHEN** a caller constructs a Tensor with a negative offset
- **THEN** construction fails with an error identifying that Tensor offsets
  must be non-negative before the Tensor becomes usable

#### Scenario: Reject insufficient carrier storage
- **WHEN** a caller supplies a placement for which
  `offset + layout.cosize` exceeds the carrier size
- **THEN** construction fails with an error identifying that the placement
  exceeds the carrier size

### Requirement: Tensor state is projected from one authoritative representation

Every Tensor SHALL have one authoritative representation consisting of one
logical dtype, an ordered non-empty sequence of carrier-backed subtensors, and
an ordered sequence of adjacent layouts. Each subtensor SHALL contain its
storage dtype, carrier, non-negative offset, and placement layout.

For a Tensor, `carrier`, `offset`, and `layout` query the storage state of
subtensor zero; `dtype()` queries its logical dtype; `size()` queries the count
of logical coordinates in its primary placement Shape; and `carrier_type()`
queries the exact carrier class used by subtensor zero.

Reading `carrier` SHALL return subtensor zero's carrier, reading `offset` SHALL
return its integer Tensor offset, and reading `layout` SHALL return its
placement Layout. `dtype()` SHALL return the representation's logical dtype,
`size()` SHALL return subtensor zero's placement size as an integer, and
`carrier_type()` SHALL return the exact type of subtensor zero's carrier. None
of these queries SHALL modify the Tensor, its representation, or any carrier.
Their returns SHALL remain consistent for Tensors produced by construction,
views, and operations.

#### Scenario: Read conventional Tensor properties
- **WHEN** a caller reads `carrier`, `offset`, `layout`, `dtype()`, `size()`, or
  `carrier_type()` from a conventional Tensor
- **THEN** each result is derived from the Tensor's logical dtype or subtensor
  zero as specified above

#### Scenario: Observe a framework-produced Tensor
- **WHEN** a view or operation produces a Tensor with a fresh validated
  representation
- **THEN** its public properties agree with that representation rather than
  with stale state from an input Tensor

### Requirement: Tensor validates a dtype-provided ordered storage schema

In framework-owned representation construction, `logical_dtype` describes the
Tensor's logical values, `subtensors` supplies the ordered carrier-backed
storage levels, and `adjacent_layouts` supplies the transitions between
consecutive levels. Each subtensor's position identifies the corresponding
position in the logical dtype's storage schema.

Successful representation construction SHALL produce one validated
authoritative representation containing the supplied logical dtype, ordered
subtensors, and ordered adjacent layouts. Validation SHALL obtain the ordered
storage dtypes from the logical dtype's storage schema. For a schema
`(D_0, ..., D_(n-1))`, the representation SHALL contain exactly `n`
subtensors, subtensor `i` SHALL have storage dtype identical to `D_i`, and the
representation SHALL contain exactly `n - 1` adjacent layouts.

If the logical dtype provides no storage schema, representation construction
SHALL fail with `ValueError` identifying the missing Tensor storage schema. If
the subtensor or adjacent-layout count differs from the schema, or a
subtensor's storage dtype is not identical to its corresponding schema entry,
construction SHALL fail with `ValueError`. Each failure SHALL occur before
returning a representation or invoking a dtype-specific rule and SHALL NOT
mutate or release a carrier. This capability does not define which dtypes
provide a schema or what storage dtypes their schemas contain.

#### Scenario: Validate an ordered storage schema
- **WHEN** the logical dtype supplies an ordered storage schema
  `(D_0, ..., D_(n-1))`
- **THEN** validation accepts only `n` subtensors with the identical storage
  dtype at each corresponding position and exactly one adjacent layout between
  each consecutive pair

#### Scenario: Reject a storage-schema count or order mismatch
- **WHEN** the representation has the wrong number of subtensors or a
  subtensor's storage dtype is not identical to the schema entry at that
  position
- **THEN** representation construction fails with an error identifying the
  expected storage-schema count or position before any dtype-specific
  rule runs

#### Scenario: Reject a dtype without a storage schema
- **WHEN** the logical dtype provides no Tensor storage schema
- **THEN** representation construction fails with an error identifying that
  the logical dtype has no Tensor storage schema before any dtype-specific rule
  runs

### Requirement: Universal validation protects every subtensor

Universal validation consumes the complete candidate representation produced
from the logical dtype, ordered subtensors, and ordered adjacent layouts. Its
successful result permits construction to continue to dtype-specific rules; it
does not transform or replace any candidate field.

Before any dtype-specific representation rule runs, the system SHALL validate
the complete storage schema and every subtensor. Each subtensor's carrier SHALL
be a Carrier whose dtype is identical to the subtensor storage dtype. Every
carrier in one representation SHALL have the same exact carrier class. Every
Tensor offset SHALL be a non-negative integer, every placement SHALL be a
Layout, every reported carrier size SHALL be non-negative, and every placement
SHALL satisfy `offset + layout.cosize <= carrier.size()`.

If a subtensor carrier or placement has the wrong type, a Tensor offset is not
an integer, or two carriers have different exact classes, validation SHALL fail
with `TypeError`. If a carrier dtype is not identical to its storage dtype, a
Tensor offset or carrier size is negative, or a placement exceeds its carrier,
validation SHALL fail with `ValueError`. Each failure SHALL occur before
returning a representation or invoking a dtype-specific rule and SHALL NOT
mutate or release a carrier.

#### Scenario: Accept a universally valid representation
- **WHEN** all subtensors match their ordered storage dtypes, use one exact
  carrier class, have valid offsets and layouts, and fit their carriers
- **THEN** universal validation completes before dtype-specific rules are
  evaluated

#### Scenario: Reject a carrier dtype mismatch
- **WHEN** a subtensor's carrier dtype is not the identical dtype object named
  by that subtensor
- **THEN** universal validation fails with an error identifying the subtensor
  position and carrier-dtype mismatch before any dtype-specific rule runs

#### Scenario: Reject mixed carrier classes
- **WHEN** two subtensors use carriers of different exact classes, including a
  base class and its subclass
- **THEN** universal validation fails with an error identifying the conflicting
  positions and exact carrier classes before any dtype-specific rule runs

#### Scenario: Reject an invalid subtensor field
- **WHEN** a subtensor has a non-Carrier carrier, a non-integer offset, a
  non-Layout placement, or its carrier reports a negative size
- **THEN** universal validation fails with an error identifying the subtensor
  position and invalid field before any dtype-specific rule runs

#### Scenario: Reject an out-of-bounds placement
- **WHEN** any subtensor has a negative offset or a placement whose
  scalar-index span exceeds its carrier
- **THEN** universal validation fails with an error identifying the subtensor
  position and violated offset or carrier-size boundary before any
  dtype-specific rule runs

### Requirement: Adjacent layouts map between representation levels

For representation level `i`, placement layout `L_i` consumes a logical
coordinate at that level and its linear index selects a value from subtensor
`i`. Adjacent layout `S_i` consumes the same logical coordinate and its linear
index identifies the logical coordinate used at level `i + 1`.

For consecutive placement layouts `L_i` and `L_(i+1)`, adjacent layout `S_i`
SHALL map logical coordinates in `L_i.shape` to linear indices decoded as
logical coordinates in `L_(i+1).shape`. Accordingly, `S_i.shape` SHALL equal
`L_i.shape`, and every linear index reached by `S_i` SHALL lie in
`[0, L_(i+1).size)`.

Both roles SHALL use ordinary Layout values. For placement layout `L_i`, the
Tensor SHALL use linear index `j` to access subtensor `i`'s carrier at
`offset_i + j`. For adjacent layout `S_i`, the Tensor SHALL decode linear index
`j` as a logical coordinate in `L_(i+1).shape`. Successful adjacent-layout
validation SHALL return control to representation construction without
modifying either Layout or any carrier.

#### Scenario: Accept an adjacent level map
- **WHEN** `S_i` has the source placement shape as its domain and every linear
  index it reaches lies in `[0, L_(i+1).size)`
- **THEN** representation validation accepts the adjacent relationship

#### Scenario: Reject the wrong adjacent domain
- **WHEN** `S_i.shape` differs from `L_i.shape`
- **THEN** representation construction fails with `ValueError` identifying
  adjacent level `i` and the required source placement shape, mutates no
  carrier, and invokes no dtype-specific rule

#### Scenario: Reject an adjacent result outside the target shape
- **WHEN** `S_i` reaches a linear index at least as large as
  `L_(i+1).size`
- **THEN** representation construction fails with `ValueError` identifying
  adjacent level `i`, the reached linear index, and `L_(i+1).size`, mutates no
  carrier, and invokes no dtype-specific rule

### Requirement: Dtype-provided rules receive only validated context

In `rule.validate(context)`, `rule` is one representation rule supplied by the
logical dtype and `context` is the read-only description of the universally
validated candidate representation. Its fields identify the logical dtype,
ordered storage dtypes, ordered placement layouts, ordered adjacent layouts,
and ordered level Shapes.

When a logical dtype supplies an ordered sequence of representation rules,
Tensor validation SHALL invoke those rules in order only after universal
validation succeeds. Each rule SHALL receive a read-only
`RepresentationValidationContext` containing the logical dtype, ordered
storage dtypes, ordered placement layouts, ordered adjacent layouts, and
ordered level shapes.

Each successful `rule.validate(context)` call SHALL return `None`. If a rule
raises an exception, representation construction SHALL propagate that
exception before returning a representation and SHALL NOT invoke later rules,
mutate a carrier, or release a carrier. The public
`RepresentationValidationContext` protocol SHALL be available from the
top-level package and the dtype namespace, and the public rule-validation
contract SHALL use that protocol. Reading each context field SHALL return the
corresponding value described above without modifying the context. This
capability does not define how a dtype stores rules or how rules contribute to
dtype identity.

#### Scenario: Invoke ordered rules with validated state
- **WHEN** universal validation succeeds and the logical dtype supplies
  representation rules
- **THEN** each representation rule receives the same frozen context in its
  declared order

#### Scenario: Stop before rules on universal failure
- **WHEN** the representation violates its storage schema, subtensor, carrier,
  placement, or adjacent-layout requirements
- **THEN** no dtype-specific representation rule is invoked

#### Scenario: Observe the public validation protocol
- **WHEN** an external representation rule inspects its validation argument
- **THEN** it can read all five documented context fields but cannot mutate the
  context

### Requirement: Tensor representations own carriers and track all versions

A representation's carrier references identify the storage objects on which
its Tensor depends. A mutation snapshot identifies each distinct constituent
carrier together with the version observed when the snapshot is taken.

A Tensor representation SHALL retain strong references to every constituent
carrier and SHALL NOT automatically release a carrier when the representation
or a view is discarded. The system SHALL permit multiple representations and
views to share the same carrier.

For mutation detection, the system SHALL snapshot the identity and version of
every unique constituent carrier, preserving the order of each carrier's first
subtensor occurrence. Taking the snapshot SHALL return the ordered identity and
version pairs without modifying any carrier. It SHALL NOT collapse that state
to only subtensor zero, a maximum, or a sum.

#### Scenario: Keep carrier storage alive
- **WHEN** all external carrier references are removed while a Tensor
  representation still refers to the carrier
- **THEN** the carrier remains alive and is not automatically released

#### Scenario: Share a carrier across representations
- **WHEN** multiple Tensors or views refer to the same carrier
- **THEN** each representation remains valid without taking exclusive
  ownership or automatically releasing the shared carrier

#### Scenario: Snapshot unique carrier versions in level order
- **WHEN** a representation contains repeated and distinct carrier identities
  across its subtensors
- **THEN** its mutation snapshot contains each unique carrier identity and
  current version once, ordered by first occurrence

### Requirement: Public multi-subtensor behavior remains narrowly bounded

The public `Tensor(carrier, offset, layout)` constructor SHALL remain the only
public representation-construction form and SHALL create only one-subtensor
Tensors. The system SHALL permit framework-owned construction to create
validated multi-subtensor Tensors without making the representation constructor
public.

Validated multi-subtensor Tensors SHALL be accepted only by the pure leading-
level layout views `permute`, positive-step `slice`, view-only `reshape`,
`as_strided`, `broadcast_to`, `broadcast_in_dim`, `squeeze`, and `unsqueeze`.
Other operations, including coordinate access, mutation, arithmetic, movement,
scatter, backward, and DLPack export, SHALL reject a multi-subtensor Tensor
before allocation, carrier mutation, or other operation effects. Detailed
view transformation and autograd semantics are outside this capability.

Each admitted view SHALL return the validated multi-subtensor view Tensor
specified by `core-tensor-views`. Every rejected operation SHALL raise its
one-subtensor-required error before returning a value.

#### Scenario: Public construction remains conventional
- **WHEN** a caller uses the public Tensor constructor
- **THEN** the caller supplies one carrier, offset, and placement layout and
  receives a one-subtensor Tensor

#### Scenario: Permit a supported pure layout view
- **WHEN** a validated multi-subtensor Tensor is passed to one of the listed
  leading-level layout views within that view's existing domain
- **THEN** the view may produce another fully validated multi-subtensor Tensor

#### Scenario: Reject an unsupported multi-subtensor operation before effects
- **WHEN** a validated multi-subtensor Tensor is passed to any other operation
- **THEN** the operation fails with an error identifying that it requires a
  one-subtensor Tensor before allocation, carrier mutation, kernel execution,
  or autograd bookkeeping
