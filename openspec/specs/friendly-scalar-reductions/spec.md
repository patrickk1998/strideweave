---
title: Friendly Scalar Reductions
publish: true
status: stable
order: 100
summary: Whole-tensor sum and mean reductions with scalar layouts and reverse-mode behavior.
---

# friendly-scalar-reductions Specification

## Purpose

Define the friendly whole-tensor `sum` and `mean` conveniences for flat
tensors. These helpers compose public operations and preserve their carrier,
dtype, dispatch, and reverse-mode semantics.

## Requirements

### Requirement: Friendly scalar reductions are explicit submodule exports

The `strideweave.friendly` submodule SHALL export `sum` and `mean`. Each
successful call MUST return an ordinary public `Tensor`.

#### Scenario: Import the scalar-reduction surface

- **WHEN** a caller imports `strideweave.friendly`
- **THEN** `sum` and `mean` are available from that submodule
- **AND** each successful call returns a `Tensor`

### Requirement: Reduction inputs are flat tensors with bounded mode count

For `sum(tensor)` and `mean(tensor)`, `tensor` SHALL mean the input `Tensor`
whose complete logical value set is reduced. Its layout MUST contain from one
through 26 top-level modes, and each top-level mode MUST be an integer extent
rather than a hierarchical shape.

A layout with no mode or any hierarchical top-level mode SHALL raise
`ValueError` with `tensor must be a flat tensor with integer modes`. A flat
layout with more than 26 top-level modes SHALL raise `ValueError` with `tensor
has too many modes to reduce`. Both failures MUST occur before reduction
dispatch. An object that does not provide the required Tensor interface SHALL
raise `AttributeError` at its first missing Tensor attribute.

#### Scenario: Accept a flat three-mode tensor

- **WHEN** a caller passes a tensor with three integer top-level modes to `sum`
- **THEN** every logical element participates in the reduction
- **AND** the call returns one scalar tensor

#### Scenario: Reject a hierarchical mode

- **WHEN** `sum` or `mean` receives a tensor with a hierarchical top-level mode
- **THEN** it raises `ValueError` with `tensor must be a flat tensor with integer modes`
- **AND** no reduction operation is dispatched

#### Scenario: Reject more than 26 modes

- **WHEN** `sum` or `mean` receives a flat tensor with more than 26 top-level modes
- **THEN** it raises `ValueError` with `tensor has too many modes to reduce`
- **AND** no reduction operation is dispatched

#### Scenario: Reject an object without the Tensor interface

- **WHEN** `sum` or `mean` receives an object without a `layout` attribute
- **THEN** Python raises `AttributeError`
- **AND** no reduction operation is dispatched

### Requirement: Sum returns an implicit-backward scalar

`sum(tensor)` SHALL add every logical element using public `reduce_sum`
semantics and MUST return the total in a tensor with exact layout
`Layout(Shape(1), Stride(1))`. For an injective differentiable input, calling
`backward()` on that scalar SHALL propagate one unit of cotangent to every
logical input element; aliased inputs SHALL retain the core autograd
accumulation semantics.

#### Scenario: Sum a matrix

- **WHEN** a caller sums a Float32 tensor containing `1.0`, `2.0`, `3.0`, and `4.0`
- **THEN** the result has exact scalar layout `Layout(Shape(1), Stride(1))` and value `10.0`
- **AND** implicit `backward()` produces a logical input gradient of four ones

#### Scenario: Sum a one-mode tensor

- **WHEN** a caller invokes `sum(arange(4))`
- **THEN** the scalar result contains `6.0`
- **AND** its value can be consumed by the friendly `item` helper

### Requirement: Mean scales the whole-tensor sum by logical size

`mean(tensor)` SHALL return `sum(tensor)` multiplied by the reciprocal of
`tensor.size()`. Its result MUST retain the exact scalar layout
`Layout(Shape(1), Stride(1))` and the public dtype, carrier dispatch, and
autograd behavior of that composition. For a differentiable tensor of logical
size `N` with an injective layout, implicit `backward()` SHALL propagate `1 / N`
to every logical input element; aliased inputs SHALL retain the core autograd
accumulation semantics.

#### Scenario: Average a matrix

- **WHEN** a caller averages a Float32 tensor containing `1.0`, `2.0`, `3.0`, and `4.0`
- **THEN** the scalar result contains `2.5`
- **AND** implicit `backward()` produces a logical input gradient of four `0.25` values

### Requirement: Reduction results preserve carrier implementation

The friendly reductions SHALL accept a tensor on any carrier that can execute
the public operations their definitions require. `sum` MUST use that carrier's
whole-tensor `reduce_sum` dispatch, and `mean` MUST additionally use its
weak-scalar multiplication dispatch. On success, the result carrier MUST have
the same exact implementation class as the carrier through which the input
operation was dispatched. Unsupported plans SHALL retain the public operation
diagnostics.

#### Scenario: Reduce through a non-CPU supporting carrier

- **WHEN** `sum` receives a flat tensor whose carrier supports its resolved whole-tensor reduction plan
- **THEN** reduction dispatch occurs through that carrier
- **AND** the result carrier has the same exact implementation class
