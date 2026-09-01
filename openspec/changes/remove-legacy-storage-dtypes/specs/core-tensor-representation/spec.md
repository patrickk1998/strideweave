## MODIFIED Requirements

### Requirement: Conventional Tensor construction creates one subtensor

In `Tensor(carrier, offset, layout)`, `carrier` supplies the stored values and
their dtype, `offset` identifies the carrier index corresponding to layout
linear index zero, and `layout` maps each Tensor logical coordinate to the
linear index added to `offset`.

`Tensor(carrier, offset, layout)` SHALL return a new Tensor backed by the
conventional one-subtensor representation when the carrier's dtype is a
`SimpleDType`. Its logical dtype and sole storage dtype SHALL be that identical
simple descriptor, its sole subtensor SHALL retain the supplied carrier,
offset, and placement Layout, and it SHALL have no adjacent layouts. A carrier
whose dtype is a category or compound descriptor SHALL fail with `ValueError`
identifying that the dtype cannot provide the conventional one-subtensor
storage schema.

The offset SHALL be a non-negative integer. If the offset is not an integer,
construction SHALL fail with `TypeError`. If the offset is negative or
`offset + layout.cosize` exceeds the carrier size, construction SHALL fail with
`ValueError`. Each failure SHALL occur before returning a Tensor and SHALL NOT
mutate or release the carrier.

#### Scenario: Construct a conventional Tensor

- **WHEN** a caller supplies a simple-dtype carrier, a non-negative offset, and
  a `Layout` whose addressed scalar-index span fits the carrier
- **THEN** the resulting Tensor has exactly one subtensor with that carrier,
  offset, and placement layout, the carrier's identical simple dtype as its
  logical and storage dtype, and zero adjacent layouts

#### Scenario: Reject a negative offset

- **WHEN** a caller constructs a Tensor with a negative offset
- **THEN** construction fails with an error identifying that Tensor offsets
  must be non-negative before the Tensor becomes usable

#### Scenario: Reject insufficient carrier storage

- **WHEN** a caller supplies a placement for which
  `offset + layout.cosize` exceeds the carrier size
- **THEN** construction fails with an error identifying that the placement
  exceeds the carrier size

#### Scenario: Reject a category-backed conventional Tensor

- **WHEN** a custom carrier reports a `DTypeCategory` as its storage dtype and
  a caller supplies it to `Tensor(carrier, offset, layout)`
- **THEN** construction fails with `ValueError` before returning a Tensor or
  mutating or releasing the carrier

### Requirement: Tensor validates a dtype-provided ordered storage schema

In framework-owned representation construction, `logical_dtype` describes the
Tensor's logical values, `subtensors` supplies the ordered carrier-backed
storage levels, and `adjacent_layouts` supplies the transitions between
consecutive levels. Each subtensor's position identifies the corresponding
position in the logical dtype's storage schema.

A `SimpleDType` SHALL provide the one-entry storage schema containing that
identical descriptor. A `CompoundDType` SHALL provide the ordered storage
schema in its `simple_types`. A `DTypeCategory` SHALL provide no Tensor storage
schema. Consequently, every representation with more than one subtensor SHALL
use a compound logical dtype, while a simple logical dtype SHALL use exactly one
same-dtype subtensor.

Successful representation construction SHALL produce one validated
authoritative representation containing the supplied logical dtype, ordered
subtensors, and ordered adjacent layouts. For a schema
`(D_0, ..., D_(n-1))`, the representation SHALL contain exactly `n`
subtensors, subtensor `i` SHALL have storage dtype identical to `D_i`, and the
representation SHALL contain exactly `n - 1` adjacent layouts.

If the logical dtype is a category or otherwise provides no storage schema,
representation construction SHALL fail with `ValueError` identifying the
missing Tensor storage schema. If the subtensor or adjacent-layout count differs
from the schema, or a subtensor's storage dtype is not identical to its
corresponding schema entry, construction SHALL fail with `ValueError`. Each
failure SHALL occur before returning a representation or invoking a
dtype-specific rule and SHALL NOT mutate, release, or claim ownership of a
carrier.

#### Scenario: Validate a simple storage schema

- **WHEN** the logical dtype is a `SimpleDType`
- **THEN** validation accepts exactly one subtensor whose storage dtype is that
  identical descriptor and accepts no adjacent layouts

#### Scenario: Validate an ordered storage schema

- **WHEN** a `CompoundDType` supplies ordered `simple_types`
  `(D_0, ..., D_(n-1))`
- **THEN** validation accepts only `n` subtensors with the identical storage
  dtype at each corresponding position and exactly one adjacent layout between
  each consecutive pair

#### Scenario: Reject a storage-schema count or order mismatch

- **WHEN** the representation has the wrong number of subtensors or a
  subtensor's storage dtype is not identical to the schema entry at that
  position
- **THEN** representation construction fails with an error identifying the
  expected storage-schema count or position before any dtype-specific rule
  runs

#### Scenario: Reject a dtype without a storage schema

- **WHEN** the logical dtype is `DType.Any`, `DType.Floating`, `DType.Integer`,
  or an extension `DTypeCategory`
- **THEN** representation construction fails with `ValueError` identifying
  that the logical dtype has no Tensor storage schema before any
  dtype-specific rule or carrier side effect
