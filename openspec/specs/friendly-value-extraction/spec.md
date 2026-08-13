---
title: Friendly Value Extraction
publish: true
status: stable
order: 110
summary: Scalar and flat-list extraction through tensor logical coordinates and carrier values.
---

# friendly-value-extraction Specification

## Purpose

Define the friendly helpers that extract Python values from readable tensors.
Extraction follows the tensor's offset and hierarchical layout rather than
assuming compact storage or exposing physical storage order.

## Requirements

### Requirement: Friendly value extraction is an explicit submodule surface

The `strideweave.friendly` submodule SHALL export `item` and `to_list`. `item`
MUST return one Python value on success, and `to_list` MUST return a new Python
list on success.

#### Scenario: Import the value-extraction surface

- **WHEN** a caller imports `strideweave.friendly`
- **THEN** `item` and `to_list` are available from that submodule
- **AND** successful calls return Python values rather than Tensor views

### Requirement: Item extracts exactly one logical value

For `item(tensor)`, `tensor` SHALL mean the readable input `Tensor` whose sole
logical value is requested. When `tensor.size()` is one, `item` MUST use integer
key zero to identify the sole logical coordinate, map that coordinate through
the tensor's layout and offset, and return the carrier's Python value. The
tensor and carrier version SHALL remain unchanged.

When `tensor.size()` is not one, `item` SHALL raise `ValueError` with `item
requires a tensor with exactly one element` before carrier storage is read. An
object without a `size` method SHALL raise `AttributeError` before a value is
read. A carrier value-read failure SHALL propagate unchanged.

#### Scenario: Extract a scalar reduction

- **WHEN** a caller passes a readable size-one tensor containing `3.0` to `item`
- **THEN** `item` returns the Python value `3.0`
- **AND** the input tensor and carrier version remain unchanged

#### Scenario: Reject a multi-element tensor

- **WHEN** a caller passes a tensor of logical size two to `item`
- **THEN** it raises `ValueError` with `item requires a tensor with exactly one element`
- **AND** carrier storage is not read

#### Scenario: Reject an object without the Tensor interface

- **WHEN** a caller passes an object without a `size` method to `item`
- **THEN** Python raises `AttributeError`
- **AND** no carrier value is read

### Requirement: To-list follows logical coordinate order

For `to_list(tensor)`, `tensor` SHALL mean the readable input `Tensor` whose
logical values are requested. The function MUST return a new Python list of
length `tensor.size()`. Integer keys from zero through `tensor.size() - 1`
SHALL identify logical coordinates in the core first-mode-fastest order; the
tensor's layout and offset SHALL map each coordinate to the carrier value placed
at the corresponding list position. The tensor and carrier version SHALL remain
unchanged.

An object without a `layout` attribute SHALL raise `AttributeError` before a
value is read. A carrier value-read failure SHALL propagate unchanged.

#### Scenario: Extract a column-major matrix

- **WHEN** a tensor has coordinate values `[[1.0, 2.0], [3.0, 4.0]]` in `column_major(2, 2)` layout
- **THEN** `to_list` returns `[1.0, 3.0, 2.0, 4.0]`
- **AND** successive list positions correspond to first-mode-fastest logical coordinates

#### Scenario: Respect a tensor view offset

- **WHEN** `to_list` receives a readable tensor view with a nonzero offset
- **THEN** every returned value is read through that view's offset and layout
- **AND** the view's carrier version remains unchanged

#### Scenario: Reject an object without the Tensor interface

- **WHEN** a caller passes an object without a `layout` attribute to `to_list`
- **THEN** Python raises `AttributeError`
- **AND** no carrier value is read

### Requirement: Extraction preserves readable carrier values

For a readable tensor, both extraction helpers SHALL return values in the
carrier's documented Python representation and MUST leave the carrier identity,
residency, release state, and version unchanged. If the carrier's value-read
entry point raises, the same exception SHALL be visible to the caller.

#### Scenario: Extract Int32 values

- **WHEN** `to_list` receives a readable `DType.Int32` tensor containing two integer values
- **THEN** it returns those values as Python integers
- **AND** the tensor's carrier identity and version remain unchanged
