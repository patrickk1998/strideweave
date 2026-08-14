---
title: Interoperability and Movement
publish: true
status: stable
order: 85
summary: Zero-copy DLPack export, the carrier hook that enables it, and cross-carrier tensor movement with its dispatch registry.
---

# interop-movement Specification

## Purpose

Define how a StrideWeave Tensor hands its storage to a foreign array library
through the DLPack protocol, how a carrier opts into that export, and how
`move` relocates a tensor's values from one carrier instance into another.

DLPack itself is an external standard. This capability specifies only the
decisions a reimplementer cannot read out of that standard: which protocol
versions are produced, which carriers and dtypes can be exported at all, what
the exported buffer's shape, strides, and byte offset mean for a hierarchical
layout, how mutability is advertised, how long the exported memory stays valid,
and what export does — and does not — do to gradient tracking and carrier
version tracking.

## Terminology

| Term | Meaning |
| --- | --- |
| export | Producing a DLPack capsule from a Tensor through `__dlpack__`, without copying or converting storage. |
| consumer | The foreign library that receives an exported capsule and reads its managed tensor. |
| legacy capsule | A `PyCapsule` named `dltensor` holding an unversioned `DLManagedTensor`. |
| versioned capsule | A `PyCapsule` named `dltensor_versioned` holding a `DLManagedTensorVersioned` carrying an explicit protocol version and a flags word. |
| managed tensor | The DLPack structure inside a capsule, owning the shape and stride arrays it points at and carrying the deleter that frees them. |
| exporting carrier | The carrier whose storage the exported managed tensor addresses. |
| DLPack device | The `(device_type, device_id)` pair the exporting carrier reports for its storage. |
| leaf modes | A layout's Shape tree traversed depth-first to its leaf extents, in tree order, paired with the corresponding leaf strides, using the leaf-mode term `core-layout` defines. |
| cosize | A layout's minimum origin-based scalar-index span, as `core-layout` defines it: one plus the greatest scalar index a logical coordinate reaches. It is independent of logical size, so a gapped or stride-zero layout may span more or fewer slots than it has coordinates. |
| move | Relocating a tensor's values into a caller-supplied destination carrier instance, releasing the source carrier. |
| move operation | An `Operation` subclass of `MoveOperation` that performs one move for a carrier pair. |
| move registry | The process-global mapping from an exact `(source carrier class, destination carrier class)` pair to a move operation class. |

## Requirements

### Requirement: DLPack export produces version 1.0 or the legacy structure on request

A Tensor SHALL expose `__dlpack__(stream=None, *, max_version=None,
dl_device=None, copy=None)` and `__dlpack_device__()`. All four `__dlpack__`
inputs are optional and SHALL default to `None`; `stream` SHALL be positional or
keyword and the other three SHALL be keyword-only.

`max_version` names the highest DLPack protocol version the consumer is
prepared to read, as a `(major, minor)` pair, and SHALL select the produced
structure. When `max_version` is `None` or its major component is `0`,
`__dlpack__` SHALL return a legacy capsule. When its major component is at
least `1`, `__dlpack__` SHALL return a versioned capsule whose version field is
exactly major `1`, minor `0`, regardless of how much higher the consumer's
ceiling is. The producer SHALL cap its advertised version at 1.0 and SHALL
succeed for every higher ceiling a consumer offers.

`max_version` SHALL be `None`, a tuple, or a list; any other object SHALL fail
with `TypeError`. It SHALL hold exactly two elements, and both SHALL be
non-negative integers; a different element count and a negative component SHALL
each fail with `ValueError`. The failure for a non-integer element is
unspecified.

`__dlpack_device__()` SHALL take no arguments and SHALL return the exporting
carrier's DLPack device as a two-element tuple. It SHALL depend only on the
carrier's DLPack support and the one-subtensor requirement, so it SHALL succeed
for a tensor whose dtype `__dlpack__` would reject.

#### Scenario: Produce a versioned capsule for a 1.x consumer

- **WHEN** a caller passes `max_version=(1, 0)` or any higher major version
- **THEN** the returned capsule is named `dltensor_versioned` and its managed
  tensor reports version 1.0

#### Scenario: Fall back to the legacy structure

- **WHEN** a caller omits `max_version` or passes a major version of `0`
- **THEN** the returned capsule is named `dltensor` and holds an unversioned
  managed tensor

#### Scenario: Reject a malformed version ceiling

- **WHEN** `max_version` is not a tuple or list
- **THEN** the call fails with `TypeError`
- **AND** a two-element requirement violation or a negative component fails
  with `ValueError`

#### Scenario: Report a device for a non-exportable dtype

- **WHEN** a caller reads `__dlpack_device__()` on a `Bool` CPU tensor
- **THEN** the call returns the CPU device
- **AND** exporting that same tensor fails on its dtype

### Requirement: Export is zero-copy, same-device, and stream-agnostic

Export SHALL alias the exporting carrier's existing storage. It SHALL never
allocate a copy, convert a dtype, or migrate storage to another device.

`copy` names the consumer's demand about duplication: a true value demands a
fresh copy the consumer may own outright, and a false value or `None` accepts
the alias. A truthy `copy` SHALL fail with `BufferError` because copy exports
are not implemented; every falsy value SHALL produce the alias.

`dl_device` names the device the consumer wants the result to live on, as a
`(device_type, device_id)` pair. It SHALL be `None`, a tuple, or a list; any
other object SHALL fail with `TypeError`. It SHALL hold exactly two integer
elements, and a different element count SHALL fail with `ValueError`; the
failure for a non-integer element is unspecified. When `dl_device` is supplied,
it SHALL equal the exporting carrier's DLPack device component for component,
and any other device SHALL fail with `BufferError` because cross-device exports
are not implemented.

`stream` names the consumer's compute stream, on which a producer for an
asynchronous device would order its pending writes before the consumer reads.
Export performs no work a stream could order, so `stream` SHALL be accepted in
any form and SHALL leave the result identical to omitting it.

#### Scenario: Refuse a copy export

- **WHEN** a consumer calls `__dlpack__(copy=True)`
- **THEN** the call fails with `BufferError` and no capsule is produced

#### Scenario: Refuse a cross-device export

- **WHEN** a consumer requests a `dl_device` other than the device the
  exporting carrier reports
- **THEN** the call fails with `BufferError`

#### Scenario: Accept the tensor's own device

- **WHEN** a consumer passes the same `dl_device` the tensor reports from
  `__dlpack_device__()`
- **THEN** the export succeeds and aliases the same storage

### Requirement: The exported buffer describes the layout's flattened leaf modes

The exported managed tensor's `ndim`, `shape`, and `strides` SHALL be the
tensor layout's leaf modes and their leaf strides, in order, so a hierarchical
StrideWeave layout SHALL appear to the consumer as a flat strided array with
one dimension per leaf mode. Strides SHALL be expressed in elements, as DLPack
requires, and SHALL be exported unchanged, including the stride-zero modes of a
broadcast view.

`data` SHALL be the exporting carrier's storage pointer and `byte_offset` SHALL
be the tensor's offset scaled by the exported dtype's item size, so the
consumer addresses the tensor's own window rather than the carrier's start.

The managed tensor SHALL own the shape and stride arrays it points at, and its
deleter SHALL free them.

#### Scenario: Flatten a hierarchical layout

- **WHEN** a tensor whose layout has nested modes is exported
- **THEN** the consumer observes one dimension per leaf mode with that mode's
  leaf stride

#### Scenario: Preserve broadcast strides

- **WHEN** a broadcast view with a stride-zero mode is exported
- **THEN** the consumer observes stride zero for that mode and reads the same
  storage element for every coordinate along it

#### Scenario: Address the tensor's window

- **WHEN** a tensor with a non-zero offset is exported
- **THEN** the managed tensor's byte offset selects that window and the
  consumer's first element is the tensor's first element

### Requirement: Only Float32 and Int32 logical dtypes are exportable

Export SHALL map logical dtype `Float32` to the DLPack float code with 32 bits
and one lane, and logical dtype `Int32` to the DLPack int code with 32 bits and
one lane. Every other logical dtype, including `Bool`, `Floating`, `Any`, and
every compound dtype, SHALL fail with `BufferError`.

The carrier's DLPack opt-in SHALL be resolved before the dtype, so a tensor
that fails both checks SHALL report the carrier failure. The dtype failure is
therefore observable only on a carrier that supports DLPack.

#### Scenario: Export a supported dtype

- **WHEN** a `Float32` or `Int32` tensor on an exporting carrier is exported
- **THEN** the consumer receives a 32-bit single-lane float or int array
  aliasing the same storage

#### Scenario: Reject an unsupported dtype

- **WHEN** a `Bool` tensor on an exporting carrier is exported
- **THEN** the call fails with `BufferError` naming the supported dtypes

### Requirement: A carrier opts into DLPack through `dlpack_info`

DLPack support SHALL be a per-carrier decision made by the public
`dlpack_info()` carrier hook rather than a property of the Tensor. The default
hook SHALL fail with `BufferError`, so a carrier that does not override it SHALL
be non-exportable and both `__dlpack__` and `__dlpack_device__` SHALL fail
identically.

A carrier that overrides the hook SHALL return a value convertible to a
dictionary carrying the keys `pointer`, `device_type`, and `device_id`, holding
the integer address of the storage the tensor's offset is relative to and the
two DLPack device components. A value that is not convertible SHALL fail with
`TypeError`, and a converted value missing any of those keys SHALL fail with
`KeyError`. Additional keys SHALL be ignored, and an exception the hook raises
SHALL propagate unchanged. The failure for a non-integer value at one of those
keys is unspecified.

When `pointer` is zero for a tensor with elements, the call SHALL fail with
`BufferError` because a DLPack data pointer must be non-null; a released
carrier reporting a null pointer SHALL therefore fail rather than export
dangling storage.

The reported device SHALL be used verbatim for both `__dlpack_device__` and the
exported managed tensor, and `dl_device` validation SHALL compare against it, so
a carrier for any device reports and exports that device.

The built-in carriers SHALL support DLPack as follows.

| Carrier | DLPack export |
| --- | --- |
| `CPU` | Supported; reports its storage pointer with device type `1` (CPU) and device id `0` |
| `Generic` | Unsupported; fails with `BufferError` |
| `FileBacked` | Unsupported; fails with `BufferError` |
| `Evictable` | Unsupported in both residencies, as `carrier-composition` requires; fails with `BufferError` naming the Evictable carrier |

#### Scenario: Export a CPU tensor

- **WHEN** a `Float32` CPU tensor is exported
- **THEN** `__dlpack_device__()` reports `(1, 0)` and the consumer aliases the
  carrier's memory

#### Scenario: Reject a non-exporting carrier

- **WHEN** a Generic, FileBacked, or Evictable tensor is exported or queried
  for its DLPack device
- **THEN** both calls fail with `BufferError` and no capsule is produced

#### Scenario: Report a foreign device verbatim

- **WHEN** a carrier whose hook reports a non-CPU device type exports
- **THEN** `__dlpack_device__()` and the managed tensor both carry that device
- **AND** a `dl_device` request naming it is accepted

### Requirement: Versioned exports advertise mutability as an advisory read-only flag

A versioned capsule's flags word SHALL set the DLPack read-only bit exactly
when the Tensor is not publicly mutable at the moment of export, and SHALL be
zero otherwise. Public mutability SHALL be the same predicate `tensor
.is_mutable()` reports, so a carrier constructed immutable and a carrier
currently owned by another carrier both SHALL export read-only.

A legacy capsule carries no flags word, so a legacy export SHALL convey no
mutability information at all. A producer that needs the read-only signal
honored SHALL therefore be exported to a consumer that requests version 1.0 or
higher.

The flag SHALL be advisory. StrideWeave SHALL neither prevent nor detect a
consumer's writes through the exported buffer, and violating the flag SHALL
have no StrideWeave-side effect beyond whatever the modified storage causes on
the next read.

#### Scenario: Mark an immutable tensor read-only

- **WHEN** a tensor whose carrier is not publicly mutable is exported with
  `max_version=(1, 0)`
- **THEN** the managed tensor's read-only flag is set and a consumer that
  honors it exposes a non-writable array

#### Scenario: Leave a mutable tensor writable

- **WHEN** a tensor whose carrier is publicly mutable is exported with
  `max_version=(1, 0)`
- **THEN** the flags word is zero and a consumer may write through the alias

#### Scenario: Lose the signal on a legacy export

- **WHEN** an immutable tensor is exported without `max_version`
- **THEN** the legacy capsule carries no flag and the consumer cannot learn
  that the tensor is read-only

### Requirement: Export changes neither version tracking nor gradient tracking

Exporting SHALL leave the exporting carrier's version, the tensor's
`autograd_ctx`, and every tensor's `.grad` exactly as they were, and SHALL
record no autograd node. A tensor SHALL remain exportable while it participates
in an autograd graph, and a gradient tensor SHALL be exportable under exactly
these rules.

A write a consumer performs through an exported buffer bypasses every carrier
mutation entry point, so the exporting carrier's version SHALL remain at its
pre-write value and every version token snapshotted from it SHALL still compare
equal. The saved-input version validation `autograd` defines rejects a traversal
only when a saved input's current version differs from its snapshot, so a
traversal over storage a consumer modified SHALL succeed and SHALL consume the
modified values.

Exporting mutably is consequently outside the guarantees StrideWeave can
enforce, in the same class as the other explicit escape hatches such as
`CPU.pointer()` and direct writes to a `FileBacked` path. Detecting such a
write is left to the caller, who alone knows whether a consumer will write to a
buffer taken from a live autograd input.

#### Scenario: Leave the version unchanged across export

- **WHEN** a tensor is exported and the consumer only reads
- **THEN** the carrier's version before and after the export is identical

#### Scenario: Miss a foreign write during backward

- **WHEN** a consumer writes through an exported buffer that a recorded
  forward operation saved as an input, and backward then runs
- **THEN** the carrier version is unchanged, no saved-version error is raised,
  and the gradient reflects the modified storage

#### Scenario: Export without disturbing the graph

- **WHEN** a non-leaf tensor is exported
- **THEN** its `autograd_ctx` and `.grad` are unchanged and backward still
  traverses the same graph

### Requirement: A capsule keeps its producer alive but not released storage

An exported capsule SHALL hold a strong reference to the exporting Tensor, so
the Tensor and its carrier SHALL remain alive for as long as the managed tensor
does, even when the caller drops every other reference.

Ownership SHALL follow the DLPack convention. A consumer takes ownership by
renaming the capsule to `used_dltensor` or `used_dltensor_versioned` and then
becomes responsible for calling the managed tensor's deleter. The capsule's own
deleter SHALL invoke the managed tensor's deleter only for a capsule that was
never consumed, and SHALL do nothing for a consumed one, so storage SHALL be
released exactly once on either path. Dropping the producer's reference SHALL
be safe during interpreter finalization.

That reference SHALL keep objects alive, not storage. `release()` on the
exporting carrier, and any `move` that releases it, SHALL relinquish the
storage the exported buffer addresses — freeing it when the carrier owns that
storage — and SHALL succeed while a capsule is outstanding, leaving the
exported buffer pointing at memory the exporting carrier no longer holds.
Keeping an exported buffer usable therefore requires the caller to keep the
exporting tensor unreleased and unmoved for as long as the consumer reads it.

#### Scenario: Outlive the producing expression

- **WHEN** a consumer imports a capsule from a temporary Tensor and every
  StrideWeave-side reference is dropped
- **THEN** the consumer's array still reads the original values

#### Scenario: Free once for an unconsumed capsule

- **WHEN** a capsule is produced and then garbage collected without being
  consumed
- **THEN** its deleter runs once and drops the producer reference

#### Scenario: Do not outlive released storage

- **WHEN** the exporting carrier is released, or the tensor is moved, while a
  capsule is outstanding
- **THEN** the release or move succeeds and the exported buffer no longer
  addresses storage the exporting carrier holds

### Requirement: Multi-subtensor tensors reject DLPack export

`__dlpack__` and `__dlpack_device__` SHALL require an authoritative
representation holding exactly one subtensor. For a validated multi-subtensor
Tensor, each SHALL fail with `NotImplementedError`, identifying DLPack export
as the unimplemented operation, before producing a capsule or touching any
constituent carrier. This is the DLPack instance of the general boundary
`core-tensor-representation` defines; per-plane export semantics are outside
this capability.

#### Scenario: Reject a multi-subtensor export

- **WHEN** a validated multi-subtensor Tensor is exported or queried for its
  DLPack device
- **THEN** both calls fail with `NotImplementedError`
- **AND** every constituent carrier's version and release state is unchanged

### Requirement: DLPack is export-only

StrideWeave SHALL provide no DLPack import: no `from_dlpack` entry point, no
capsule-consuming Tensor or carrier constructor, and no other way to adopt
foreign memory as a carrier. Interoperability SHALL therefore flow one way, and
gradient tracking, version tokens, and ownership SHALL be specified for the
export direction alone.

#### Scenario: Offer no adoption path

- **WHEN** a caller looks for a way to wrap a foreign DLPack capsule as a
  StrideWeave Tensor
- **THEN** the public surface offers none

### Requirement: `move` relocates values into a caller-supplied carrier

`move(tensor, destination)` SHALL fill `destination` with `tensor`'s values,
release the source carrier, and return a Tensor backed by `destination` at
offset zero with the source tensor's layout unchanged. `tensor` names the
Tensor to relocate and SHALL be a Tensor. `destination` names a live, publicly
mutable `Carrier` instance whose dtype is the tensor's dtype.

Move SHALL size the destination by the layout's `cosize` physical span rather
than by its logical size, so moving a broadcast view SHALL preserve its exact
stride-zero layout and SHALL require only `cosize` destination slots rather
than one slot per logical coordinate. A destination that is empty and supports
allocation SHALL be allocated to exactly that span; a destination that is
smaller SHALL fail.

Logical values SHALL be identical whichever move operation runs, while the
contents of destination slots that the layout does not address SHALL be
unspecified: a bulk copy carries the whole physical span including holes, and
the elementwise fallback leaves those slots at their prior values.

A validated multi-subtensor Tensor SHALL be rejected first, under the boundary
`core-tensor-representation` defines. Every other validation SHALL happen before
any copy, and SHALL reject in this order:

| Condition | Failure |
| --- | --- |
| `tensor` is not a Tensor | `TypeError` |
| the source carrier is released | `RuntimeError` |
| the source carrier is owned by another carrier | `RuntimeError` |
| `destination` is not a `Carrier` instance | `TypeError` |
| `destination` is the tensor's own carrier | `ValueError` |
| `destination` is released | `RuntimeError` |
| `destination` is not publicly mutable | `RuntimeError` |
| `destination`'s dtype is not the tensor's dtype | `TypeError` |
| the dispatched operation pins a carrier class the pair does not match | `TypeError` |
| `destination` is smaller than the layout's `cosize` and cannot allocate | `ValueError` |

#### Scenario: Move and release the source

- **WHEN** a live tensor is moved into a live mutable destination of matching
  dtype
- **THEN** the result is backed by that destination, holds the same logical
  values under the same layout, and the source carrier is released so further
  access through it fails

#### Scenario: Allocate an empty destination

- **WHEN** the destination is an empty carrier that supports allocation
- **THEN** it is allocated to the layout's `cosize` and receives the values

#### Scenario: Move a broadcast view without materializing it

- **WHEN** a stride-zero broadcast view of two elements over six coordinates is
  moved
- **THEN** the destination needs only two slots and the result keeps the same
  stride-zero layout

#### Scenario: Reject an unusable destination before copying

- **WHEN** the destination is released, immutable, of another dtype, the
  tensor's own carrier, or too small
- **THEN** the move fails with the corresponding error and neither carrier is
  modified or released

### Requirement: Move dispatches on the exact carrier class pair

The concrete move operation SHALL be selected from the move registry, which is
owned by neither carrier, by the exact `(type(source carrier), type(destination
carrier))` pair. `dispatch_move(source_class, destination_class)` names that
pair through the carrier class of the tensor and the carrier class of the
destination, and SHALL return the registered class for that exact pair or
`ElementwiseMoveOperation` when the pair is unregistered.

Dispatch SHALL match both classes by identity, so a registration SHALL apply to
its exact pair alone and a subclass of a registered carrier class SHALL resolve
to the elementwise fallback until registered on its own.

The registry SHALL be pre-populated with `CpuToFileBackedMoveOperation` for
`(CPU, FileBacked)` and `FileBackedToCpuMoveOperation` for `(FileBacked, CPU)`,
which copy the tensor's whole physical byte span with one native transfer.
Every other pair, including `(CPU, CPU)`, SHALL use `ElementwiseMoveOperation`,
which copies logical values one at a time.

#### Scenario: Select a registered bulk operation

- **WHEN** a CPU tensor is moved into a FileBacked destination, or the reverse
- **THEN** the registered native bulk operation for that pair runs

#### Scenario: Fall back for an unregistered pair

- **WHEN** the pair has no registration
- **THEN** `ElementwiseMoveOperation` runs and copies logical values

#### Scenario: Do not inherit a registration

- **WHEN** a carrier class is registered and a subclass of it is used as the
  source
- **THEN** dispatch returns the elementwise fallback rather than the parent's
  registration

### Requirement: Move registration is explicit, validated, and process-global

`register_move_operation(source_class, destination_class, operation_class)`
SHALL record one operation for one exact pair and SHALL return `None`.
`source_class` and `destination_class` name the carrier classes of the tensor
and the destination and SHALL be `Carrier` subclasses; `operation_class` names
the handler and SHALL be a `MoveOperation` subclass. A non-conforming argument
SHALL fail with `TypeError` identifying which one.

The registry SHALL be process-global, and its built-in entries SHALL already be
in place for the first caller. Registering a pair that already has an entry
SHALL fail with `ValueError` naming both class names and SHALL retain the
existing entry, so replacing a registration SHALL require an explicit
unregistration first. Re-registering a built-in pair SHALL therefore fail the
same way rather than succeed as a no-op.

`unregister_move_operation(source_class, destination_class)` names the same
pair, SHALL remove and return its registered operation class, and SHALL fail
with `KeyError` naming both class names when the pair has none. It and
`dispatch_move` SHALL accept `Carrier` subclasses; their behavior for any other
argument is unspecified.

`registered_move_operation(source_class, destination_class, operation_class)`
names the same three inputs and SHALL be a context manager that registers the
pair on entry, yields `operation_class`, and unregisters on exit, including
when the block raises.

`move`, `MoveOperation`, and the concrete move operation classes SHALL be
reachable from the top-level package; `dispatch_move`,
`register_move_operation`, `unregister_move_operation`, and
`registered_move_operation` SHALL be reachable from the movement module
`strideweave.carriers.move`.

#### Scenario: Refuse to overwrite a registration

- **WHEN** a caller registers a pair that already has an operation
- **THEN** the call fails with `ValueError` and the existing registration is
  retained

#### Scenario: Remove a registration

- **WHEN** a caller unregisters a registered pair
- **THEN** the removed class is returned and a second unregistration fails with
  `KeyError`

#### Scenario: Scope a registration to a block

- **WHEN** a `registered_move_operation` block exits, whether normally or by
  raising
- **THEN** the pair is unregistered again

### Requirement: A move operation performs its transfer through the copy hook

`MoveOperation` SHALL be public and open for subclassing. A subclass SHALL
implement the protected `_copy(tensor, destination, output, element_count)`
hook, which performs the transfer for one move and SHALL return `None`.
`tensor` names the Tensor being relocated and `destination` names the receiving
carrier instance, both as the caller supplied them; `output` names the result
Tensor already bound to that destination; and `element_count` names the number
of storage slots the transfer covers, which SHALL be the layout's `cosize`.

The hook SHALL run only after every validation succeeds, and the operation
SHALL release the source carrier only after the hook returns.

#### Scenario: Run a custom copy

- **WHEN** a registered subclass handles a pair
- **THEN** its `_copy` hook receives the layout's `cosize` as the element count
  and fills the destination

### Requirement: A move operation may pin the carrier pair it supports

A `MoveOperation` subclass MAY pin `source_class` and `destination_class` class
attributes to the exact carrier classes it supports. When either is pinned and
the supplied carrier's exact class differs, the forward call SHALL fail with
`TypeError` naming the operation and the required class, before copying
anything or releasing the source. Leaving both unpinned SHALL accept any pair,
as `ElementwiseMoveOperation` does.

#### Scenario: Reject a mismatched carrier class

- **WHEN** an operation pinned to a CPU source is invoked with a Generic source
- **THEN** the call fails with `TypeError` and the source carrier is not
  released

### Requirement: Move participates in autograd across the carrier boundary

A move of a differentiable tensor while graph construction is enabled SHALL
record an autograd node on its result, following the graph-construction
contract in `autograd`, so gradients cross carrier boundaries.

Move's backward SHALL take exactly one cotangent, which SHALL satisfy the
cotangent-layout contract `autograd` defines for the move result. Move adds two
requirements of its own: the cotangent SHALL be a live Tensor of the source
tensor's shape, so a non-Tensor SHALL fail with `TypeError` and a released
carrier SHALL fail with `RuntimeError`; and its layout SHALL be injective, so a
non-injective layout SHALL fail with `ValueError` stating that an injective
gradient layout is required. A shape mismatch SHALL fail with `ValueError`.

Backward SHALL return one gradient, materialized in fresh storage of the
*source* carrier's class through the released-carrier `new_like` guarantee
`carrier-storage` defines, carrying the cotangent's logical values under the
source tensor's layout when that layout is injective and under the canonical
injective layout for its shape otherwise. The returned gradient SHALL be
detached. Moving a broadcast view SHALL therefore hand an injective same-shape
gradient back to the broadcast node, which performs the summation.

#### Scenario: Move a gradient back to the source carrier class

- **WHEN** a CPU tensor is moved to FileBacked and backward propagates a
  cotangent through the move node
- **THEN** the gradient is a detached Tensor in fresh CPU storage holding the
  cotangent's logical values

#### Scenario: Reject a non-injective cotangent

- **WHEN** a stride-zero cotangent reaches a move node's backward
- **THEN** the call fails with `ValueError` requiring an injective gradient
  layout

## Non-Coverage

This capability does not define:

- DLPack import. There is no way to adopt foreign memory, so no import-side
  ownership, versioning, or gradient contract exists.
- Copy exports, cross-device exports, or stream synchronization semantics
  beyond accepting and ignoring a stream argument.
- Export from any built-in carrier other than `CPU`, or of any dtype other than
  `Float32` and `Int32`.
- Per-plane DLPack export or movement for validated multi-subtensor tensors,
  whose rejection boundary belongs to `core-tensor-representation`.
- Whether a move appears in profiling evidence, which `operation-profiling`
  defines; public `move` executions are excluded there.
- Evictable residency transitions and Evictable's DLPack refusal, which resolve
  the same move registry through framework-owned lowered execution and are
  defined by `carrier-composition`.
- Graph construction, traversal, cotangent layout validation, and saved-version
  validation, all defined by `autograd`.
- The released-carrier `new_like` guarantee that move's backward relies on, and
  the release semantics of each built-in carrier, both defined by
  `carrier-storage`.
- Logical size, `cosize`, leaf modes, and every other layout property this
  capability reads, all defined by `core-layout`.
