---
title: Friendly Tensor Creation
publish: true
status: stable
order: 10
summary: CPU-backed friendly tensor factories, value ordering, dtype selection, allocation, and diagnostics.
---

# friendly-tensor-creation Specification

## Purpose

Define the friendly API for constructing tensors over fresh CPU storage from
nested values, explicit layouts, constant fills, ranges, and random samples.
The capability preserves core `Layout`, `Tensor`, `CPU`, and dtype semantics
while removing the need for callers to assemble those primitives manually.

## Requirements

### Requirement: Friendly creation is an explicit submodule surface

The `strideweave.friendly` submodule SHALL export `tensor`, `full`, `zeros`,
`ones`, `arange`, `rand`, and `randn`. Each successful call MUST return an
ordinary public `Tensor` usable by the core API.

#### Scenario: Import the creation surface

- **WHEN** a caller imports `strideweave.friendly`
- **THEN** each documented creation function is available from that submodule
- **AND** a successful call to any creation function returns a `Tensor`

### Requirement: Extent factories allocate compact CPU tensors

For `full`, `zeros`, `ones`, `rand`, and `randn`, `extents` SHALL mean one or
more positive integer extents for flat top-level modes in the supplied order.
For `arange`, `count` SHALL mean the positive integer extent of its single
mode. Each successful call MUST allocate fresh storage in an exact `CPU`
carrier, use offset zero, and return a tensor whose layout is the compact
column-major layout for those extents. Allocation SHALL cover `layout.cosize`.

An empty `extents` input SHALL raise `ValueError` with `at least one extent is
required`. A non-integer extent or an extent less than one, including `count`,
SHALL raise `ValueError` with `extents must be positive integers`. Both failures
MUST occur before tensor storage is allocated.

#### Scenario: Allocate a two-mode tensor

- **WHEN** a caller invokes `zeros(2, 3)`
- **THEN** the result has `column_major(2, 3)` as its layout
- **AND** the result has offset zero over fresh `CPU` storage

#### Scenario: Reject missing extents

- **WHEN** a caller invokes an extent factory without an extent
- **THEN** it raises `ValueError` with `at least one extent is required`
- **AND** no tensor storage is allocated

#### Scenario: Reject invalid extents

- **WHEN** an extent factory receives a non-integer extent or an extent less than one
- **THEN** it raises `ValueError` with `extents must be positive integers`
- **AND** no tensor storage is allocated

### Requirement: Creation dtype selects CPU storage

For `tensor`, `full`, `zeros`, `ones`, and `arange`, `dtype` SHALL mean the
storage dtype requested from the result's CPU carrier. It SHALL be optional and
default to `DType.Float32`. A successful call MUST return a tensor with the
selected dtype and apply that dtype's CPU storage conversion to every written
value.

The selected dtype SHALL satisfy the accepted CPU storage-dtype contract. A
dtype rejected by CPU construction MUST fail before a result is returned and
SHALL expose the CPU construction diagnostic. `rand` and `randn` SHALL always
return `DType.Float32` tensors.

#### Scenario: Use the default dtype

- **WHEN** a caller invokes `tensor`, `full`, `zeros`, `ones`, or `arange` without `dtype`
- **THEN** the returned tensor's dtype is `DType.Float32`
- **AND** its values use CPU Float32 storage conversion

#### Scenario: Select Int32 storage

- **WHEN** a caller invokes `ones(2, dtype=DType.Int32)`
- **THEN** the result's dtype is `DType.Int32`
- **AND** both logical values are the integer `1`

#### Scenario: Select Bool storage

- **WHEN** a caller invokes `zeros(2, dtype=DType.Bool)`
- **THEN** the result's dtype is `DType.Bool`
- **AND** both logical values are `False`

#### Scenario: Reject a dtype CPU cannot store

- **WHEN** a caller supplies a dtype outside the accepted CPU storage set
- **THEN** CPU construction raises its documented dtype error
- **AND** no result is returned

### Requirement: Nested values define outermost-first coordinates

For `tensor(values, *, layout=None, dtype=DType.Float32)`, `values` SHALL mean a
non-empty rectangular nesting of lists or tuples whose leaves are values for
the selected dtype. With `layout=None`, each nesting level SHALL define one flat
mode, ordered from the outermost level to the innermost. The value at
`values[i][j]...` MUST be readable from the result at coordinate `[i, j, ...]`,
and the returned tensor SHALL use a compact column-major layout.

The factory SHALL raise `ValueError` with `values must be a non-empty list or
tuple` when `values` has no list-or-tuple level, with `values must not contain
empty levels` when an inferred level is empty, and with `values must be
rectangular across every level` when the nesting is ragged or its leaves occur
at inconsistent depths. A leaf rejected by CPU storage conversion SHALL expose
that conversion failure.

#### Scenario: Create from a rectangular matrix

- **WHEN** a caller invokes `tensor([[1.0, 2.0], [3.0, 4.0]])`
- **THEN** the result has layout `column_major(2, 2)`
- **AND** `result[1, 0]` is `3.0` and `result[0, 1]` is `2.0`

#### Scenario: Reject ragged values

- **WHEN** a caller invokes `tensor([[1.0, 2.0], [3.0]])`
- **THEN** tensor creation raises `ValueError` with `values must be rectangular across every level`
- **AND** no result is returned

#### Scenario: Reject an empty nested level

- **WHEN** a caller invokes `tensor([[], []])`
- **THEN** tensor creation raises `ValueError` with `values must not contain empty levels`
- **AND** no result is returned

### Requirement: Explicit layouts consume values by logical coordinates

The optional `layout` input to `tensor` SHALL mean the exact public `Layout`
used by the result and SHALL default to `None`. When `layout` is supplied,
`values` SHALL mean a finite iterable yielding exactly one value for each
logical coordinate. Integer keys from zero through `layout.size - 1` SHALL
identify those logical coordinates in the core first-mode-fastest order, and
the layout SHALL map each coordinate to the scalar index where the corresponding
yielded value is stored. The factory MUST allocate `layout.cosize` CPU storage
elements, preserving storage positions the layout does not address.

A non-`None` `layout` that is not a `Layout` SHALL raise `TypeError` with
`layout must be a Layout` before `values` is consumed or a layout property is
accessed. A `values` object that is not iterable SHALL raise `TypeError` during
iterable conversion. An exception raised by the iterable SHALL propagate
unchanged. A yielded-value count different from `layout.size` SHALL raise
`ValueError` with `values length must equal the layout logical size` before
storage is allocated.

#### Scenario: Fill a row-major layout

- **WHEN** a caller invokes `tensor([1.0, 2.0, 3.0, 4.0], layout=row_major(2, 2))`
- **THEN** the returned tensor uses the supplied row-major layout
- **AND** coordinates `(0, 0)`, `(1, 0)`, `(0, 1)`, and `(1, 1)` read `1.0`, `2.0`, `3.0`, and `4.0`

#### Scenario: Accept a finite generator

- **WHEN** a caller supplies a finite generator yielding exactly `layout.size` values
- **THEN** `tensor` consumes it once and returns a tensor using the supplied layout
- **AND** the yielded values are assigned to first-mode-fastest logical coordinates

#### Scenario: Reject an invalid explicit layout before consuming values

- **WHEN** a caller supplies a non-`None` object that is not a `Layout` as `layout`
- **THEN** tensor creation raises `TypeError` with `layout must be a Layout`
- **AND** the supplied `values` iterable is not consumed

#### Scenario: Reject non-iterable explicit values

- **WHEN** a caller supplies a valid layout and a `values` object that is not iterable
- **THEN** Python raises `TypeError` during iterable conversion
- **AND** no tensor storage is allocated

#### Scenario: Reject the wrong explicit value count

- **WHEN** a caller supplies one value with a layout whose `size` is two
- **THEN** tensor creation raises `ValueError` with `values length must equal the layout logical size`
- **AND** no tensor storage is allocated

### Requirement: Deterministic factories define every logical value

For `full(*extents, value, dtype=DType.Float32)`, `value` SHALL mean the value
assigned to every logical coordinate and SHALL be a required keyword-only
input. The function MUST return a fresh tensor whose every logical coordinate
contains `value` after conversion to `dtype`. `zeros` MUST return a fresh tensor
whose every logical coordinate contains the selected dtype's zero, and `ones`
MUST return one whose every logical coordinate contains the selected dtype's
one. `arange(count, *, dtype=DType.Float32)` MUST return a fresh one-mode tensor
whose coordinates from zero through `count - 1` contain those ascending values
converted to `dtype`.

#### Scenario: Fill with one caller value

- **WHEN** a caller invokes `full(2, value=7.5)`
- **THEN** the result contains two logical values
- **AND** both values are the `DType.Float32` representation of `7.5`

#### Scenario: Build an ascending range

- **WHEN** a caller invokes `arange(4)`
- **THEN** the result has the one-mode layout `column_major(4)`
- **AND** its logical coordinate values are `0.0`, `1.0`, `2.0`, and `3.0` in order

### Requirement: Random factories accept an optional generator

For `rand(*extents, rng=None)` and `randn(*extents, rng=None)`, `rng` SHALL mean
the optional random generator that supplies samples and SHALL default to
`None`. When `rng` is `None`, the factory SHALL use an independently initialized
generator for that call. A supplied generator SHALL provide `random()` for
`rand` or `gauss(0.0, 1.0)` for `randn`; absence of the required method SHALL
raise `AttributeError` before a result is returned.

`rand` MUST return a fresh Float32 CPU tensor containing one uniform sample in
`[0, 1)` per logical coordinate. `randn` MUST return a fresh Float32 CPU tensor
containing one standard-normal sample per logical coordinate. Samples SHALL be
assigned to first-mode-fastest logical coordinates in generator call order.

#### Scenario: Reproduce uniform samples

- **WHEN** two `rand` calls use equal extents and separately seeded `random.Random` instances with the same seed
- **THEN** their tensors contain the same logical coordinate values
- **AND** every value is at least zero and less than one

#### Scenario: Reproduce normal samples

- **WHEN** two `randn` calls use equal extents and separately seeded `random.Random` instances with the same seed
- **THEN** their tensors contain the same logical coordinate values
- **AND** both results use `DType.Float32` CPU storage

#### Scenario: Reject a generator without the required method

- **WHEN** `rand` receives an object without `random()` or `randn` receives one without `gauss()`
- **THEN** Python raises `AttributeError`
- **AND** no tensor result is returned
