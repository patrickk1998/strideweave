---
title: Friendly Layout Builders
publish: true
status: stable
order: 90
summary: Compact flat column-major and row-major layouts built from positive mode extents.
---

# friendly-layout-builders Specification

## Purpose

Define the friendly helpers that construct common compact flat layouts from
mode extents. The returned values retain the public `Layout` contract and can
be used anywhere the core API accepts a caller-built layout.

## Requirements

### Requirement: Friendly layout builders are explicit submodule exports

The `strideweave.friendly` submodule SHALL export `column_major` and
`row_major`. Each successful call MUST return an ordinary public `Layout`.

#### Scenario: Import the layout-builder surface

- **WHEN** a caller imports `strideweave.friendly`
- **THEN** `column_major` and `row_major` are available from that submodule
- **AND** each successful call returns a `Layout`

### Requirement: Builder extents define flat modes

For `column_major(*extents)` and `row_major(*extents)`, `extents` SHALL mean one
or more positive integer extents for flat top-level modes in the supplied
order. Each function MUST return a `Layout` whose `Shape` contains those
extents, whose `size` and `cosize` equal their product, and whose mapping is
compact and injective.

An empty `extents` input SHALL raise `ValueError` with `at least one extent is
required`. A non-integer extent or an extent less than one SHALL raise
`ValueError` with `extents must be positive integers`.

#### Scenario: Build a one-mode layout

- **WHEN** a caller invokes either layout builder with extent `5`
- **THEN** it returns a layout equal to `Layout(Shape(5), Stride(1))`
- **AND** the layout has `size` and `cosize` equal to `5`

#### Scenario: Reject missing extents

- **WHEN** a caller invokes `column_major()` or `row_major()`
- **THEN** the builder raises `ValueError` with `at least one extent is required`
- **AND** no `Layout` is returned

#### Scenario: Reject an invalid extent

- **WHEN** a caller supplies a non-integer extent or an extent less than one
- **THEN** the builder raises `ValueError` with `extents must be positive integers`
- **AND** no `Layout` is returned

### Requirement: Column-major strides make first-mode coordinates adjacent

`column_major(*extents)` SHALL assign stride one to the first mode and SHALL
assign every later mode the product of all preceding extents. Incrementing the
first coordinate by one while the other coordinates stay fixed MUST increment
the mapped scalar index by one. Integer keys SHALL identify logical coordinates
in the core first-mode-fastest order independently of this stride choice.

#### Scenario: Build a three-mode column-major layout

- **WHEN** a caller invokes `column_major(2, 3, 4)`
- **THEN** it returns a layout equal to `Layout(Shape([2, 3, 4]), Stride([1, 2, 6]))`
- **AND** adjacent first-mode coordinates map to adjacent scalar indices

### Requirement: Row-major strides make last-mode coordinates adjacent

`row_major(*extents)` SHALL assign stride one to the last mode and SHALL assign
every preceding mode the product of all following extents. Incrementing the
last coordinate by one while the other coordinates stay fixed MUST increment
the mapped scalar index by one. Integer keys SHALL identify logical coordinates
in the core first-mode-fastest order independently of this stride choice.

#### Scenario: Build a three-mode row-major layout

- **WHEN** a caller invokes `row_major(2, 3, 4)`
- **THEN** it returns a layout equal to `Layout(Shape([2, 3, 4]), Stride([12, 4, 1]))`
- **AND** adjacent last-mode coordinates map to adjacent scalar indices
