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

`move(tensor, destination)` SHALL remain the blocking single-Tensor surface and
SHALL behave exactly as `move_async(tensor, destination).wait()`. It SHALL fill
`destination` with `tensor`'s values, release the source carrier at successful
terminal commit, and return a Tensor backed by `destination` at offset zero
with the source Tensor's Layout unchanged. It SHALL use the same validation,
route, transfer, source-release, autograd, and publication path as
`move_async`.

Move SHALL size the destination by the Layout's `cosize` physical span rather
than by logical size, so moving a broadcast view SHALL preserve its exact
stride-zero Layout and require only `cosize` destination slots. A destination
that is empty and supports allocation SHALL have exactly that span when the
move publishes successfully; a destination that is smaller and cannot allocate
SHALL fail.

Logical values SHALL be identical whichever route runs, while destination
slots that the Layout does not address SHALL remain unspecified: a bulk copy
may carry the whole physical span including holes, and the elementwise fallback
may leave those slots at their prior values.

A validated multi-subtensor Tensor SHALL be rejected first, under the boundary
`core-tensor-representation` defines. Every other validation SHALL happen
before submission in this order:

| Condition | Failure |
| --- | --- |
| `tensor` is not a Tensor | `TypeError` |
| the source carrier is released | `RuntimeError` |
| the source carrier is owned by another carrier | `RuntimeError` |
| `destination` is not a `Carrier` instance | `TypeError` |
| `destination` is the Tensor's own carrier | `ValueError` |
| `destination` is released | `RuntimeError` |
| `destination` is not publicly mutable | `RuntimeError` |
| `destination`'s dtype is not the Tensor's dtype | `TypeError` |
| the resolved route pins a carrier class the pair does not match | `TypeError` |
| `destination` is smaller than the Layout's `cosize` and cannot allocate | `ValueError` |

#### Scenario: Move and release the source

- **WHEN** a live Tensor is moved into a live mutable destination of matching
  dtype
- **THEN** the call blocks until the result is backed by that destination with
  the same logical values and Layout and the source has been released

#### Scenario: Allocate an empty destination

- **WHEN** the destination is an empty carrier that supports allocation
- **THEN** it is allocated to the Layout's `cosize` and receives the values

#### Scenario: Move a broadcast view without materializing it

- **WHEN** a stride-zero broadcast view of two elements over six coordinates is
  moved
- **THEN** the destination needs only two slots and the result keeps the same
  stride-zero Layout

#### Scenario: Reject an unusable destination before copying

- **WHEN** the destination is released, immutable, of another dtype, the
  Tensor's own carrier, or too small
- **THEN** the move fails with the corresponding error and neither carrier is
  modified or released

### Requirement: Move dispatches on the exact carrier class pair

For a definition-backed source, the transfer route SHALL be selected from that
exact source class's `CarrierDefinition` by the exact
`(type(source carrier), type(destination carrier))` pair. A definition-free
legacy pair SHALL be selected from the legacy process-global registry. An exact
pair with no applicable route SHALL
use `ElementwiseMoveOperation` only when the applicable definitions or legacy
contract permit fallback.

Dispatch SHALL match both classes by identity. A route or legacy registration
SHALL apply to its exact pair alone; a subclass of either registered class
SHALL not inherit it.

Definition-free shipped carriers SHALL retain the accepted exact bulk routes
and elementwise fallback rules installed by the legacy movement authority,
including CPU-to-FileBacked, FileBacked-to-CPU, and the supported Metal pairs.
This delivery SHALL NOT install `CarrierDefinition` or `TransferRoute`
declarations for Generic, CPU, Metal, FileBacked, BlockDeviceCarrier, or
Evictable. TiledEvictable's internal transfers between configured
definition-free dependency carriers SHALL use their legacy exact-pair movement
path; a public move whose exact source is TiledEvictable remains governed by
its own definition and route rules above.

#### Scenario: Select a registered bulk operation

- **WHEN** an exact source definition declares a bulk route to the exact
  destination class
- **THEN** that route prepares the move

#### Scenario: Fall back for an unregistered pair

- **WHEN** the exact pair has no route and both definitions permit elementwise
  fallback
- **THEN** `ElementwiseMoveOperation` copies logical values

#### Scenario: Do not inherit a registration

- **WHEN** a carrier class has a route and a subclass is used as source or
  destination
- **THEN** the parent's route is not selected

### Requirement: Move registration is explicit, validated, and process-global

`register_move_operation(source_class, destination_class, operation_class)`
SHALL record `operation_class` for the exact `(source_class,
destination_class)` pair in the process-global legacy move registry and SHALL
return `None`. `source_class` and `destination_class` SHALL each be a
`Carrier` subclass, and `operation_class` SHALL be a `MoveOperation` subclass.
A non-conforming argument SHALL fail with `TypeError` identifying that
argument. Registering an exact pair that already has an entry, including a
built-in entry, SHALL fail with `ValueError` naming both carrier classes and
SHALL retain the existing entry.

`dispatch_move(source_class, destination_class)` SHALL select by class identity
and SHALL return the operation class registered for that exact pair. When the
exact pair has no entry, it SHALL return `ElementwiseMoveOperation`; a
registration for a parent class or a pair differing in either class SHALL not
match. `unregister_move_operation(source_class, destination_class)` SHALL
remove and return the operation class registered for that exact pair. When the
exact pair has no entry, it SHALL fail with `KeyError` naming both carrier
classes. `dispatch_move` and `unregister_move_operation` SHALL accept
`Carrier` subclasses; their behavior for other arguments is unspecified.

`registered_move_operation(source_class, destination_class, operation_class)`
SHALL return a context manager with the same input validation and duplicate
rejection as `register_move_operation`. Entering the context SHALL register the
exact pair and yield `operation_class`. Exiting SHALL unregister that pair,
whether the context body returns normally or raises. The context manager SHALL
not suppress an exception raised by its body.

The legacy registry SHALL be initialized before its first caller with every
built-in entry required by the movement specifications, including
`CpuToFileBackedMoveOperation` for `(CPU, FileBacked)` and
`FileBackedToCpuMoveOperation` for `(FileBacked, CPU)`. Every unregistered
pair, including `(CPU, CPU)`, SHALL use the elementwise fallback.

For any of these four legacy registry APIs, after validating that the named
source and destination are `Carrier` subclasses, a pair for which either exact
class has a `CarrierDefinition` SHALL fail with `RuntimeError` directing the
caller to express movement with `TransferRoute`. No legacy entry SHALL shadow
or mutate a definition-owned route. Defining a carrier after its exact legacy
pair has been observed SHALL obey the common definition sealing rule rather
than silently changing dispatch.

`move`, `MoveOperation`, and every concrete move-operation class specified for
a built-in route, including `ElementwiseMoveOperation`,
`CpuToFileBackedMoveOperation`, and `FileBackedToCpuMoveOperation`, SHALL be
importable from the top-level package. `dispatch_move`,
`register_move_operation`, `unregister_move_operation`, and
`registered_move_operation` SHALL be importable from
`strideweave.carriers.move`.

#### Scenario: Register and dispatch an exact legacy pair

- **WHEN** a caller registers a `MoveOperation` subclass for a definition-free
  exact source and destination pair
- **THEN** registration returns `None` and `dispatch_move` returns that same
  operation class only for that exact pair

#### Scenario: Refuse to overwrite a registration

- **WHEN** a caller registers a definition-free pair that already has an
  operation
- **THEN** the call fails with `ValueError` and retains the existing entry

#### Scenario: Reject registration for a defined carrier

- **WHEN** a caller invokes a legacy registry API for a pair involving an exact
  definition-backed class
- **THEN** the call fails with `RuntimeError` directing the caller to
  `TransferRoute`, and its definition-owned routes remain unchanged

#### Scenario: Remove a registration

- **WHEN** a caller unregisters a registered definition-free pair
- **THEN** the removed class is returned and a second unregistration fails
  with `KeyError`

#### Scenario: Do not match a different exact pair

- **WHEN** a registered source or destination class is replaced by a subclass,
  or neither exact class has a registration
- **THEN** `dispatch_move` returns `ElementwiseMoveOperation` and
  `unregister_move_operation` for that different pair fails with `KeyError`

#### Scenario: Scope a registration to a block

- **WHEN** a `registered_move_operation` block for a definition-free pair
  exits normally or by raising
- **THEN** the pair is unregistered again and an exception raised by the block
  remains observable

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

When graph construction is enabled for a differentiable source,
`move_async` SHALL resolve and capture a permitted exact reverse route before
submitting any forward work. Absence of a permitted reverse route SHALL fail
initiation without returning a handle. Successful forward completion SHALL
attach exactly one visible move autograd node to the published Tensor and save
the source class, source Layout, version, and route identity required by
backward. Provider transfers SHALL create no nested visible node.

Move's backward SHALL take exactly one cotangent satisfying the cotangent
Layout contract `autograd` defines. It SHALL be a live Tensor of the source
Tensor's shape; a non-Tensor SHALL fail with `TypeError`, a released carrier
with `RuntimeError`, a shape mismatch with `ValueError`, and a non-injective
Layout with `ValueError` stating that an injective gradient Layout is required.

Backward SHALL initiate the captured reverse route through the same
asynchronous movement primitive and wait for it because the current autograd
engine is synchronous. It SHALL return one detached gradient in fresh storage
of the source carrier's exact class, carrying the cotangent's logical values
under the source Tensor's Layout when injective and under the canonical
injective Layout for its shape otherwise. A moved broadcast view SHALL
therefore hand an injective same-shape gradient to the broadcast node, which
performs summation.

#### Scenario: Reject a differentiable one-way route early

- **WHEN** a differentiable source has a forward route but no permitted exact
  reverse route
- **THEN** `move_async` fails before submitting the forward transfer

#### Scenario: Move a gradient back to the source carrier class

- **WHEN** a CPU Tensor is moved to FileBacked and backward propagates a
  cotangent through the move node
- **THEN** backward waits for the captured reverse movement and returns a
  detached equivalent gradient in fresh CPU storage

#### Scenario: Reject a non-injective cotangent

- **WHEN** a stride-zero cotangent reaches a move node's backward
- **THEN** the call fails with `ValueError` requiring an injective gradient
  Layout

### Requirement: Metal movement uses exact registered bulk pairs

The shipped movement registry SHALL include exact-class CPU-to-Metal,
Metal-to-CPU, and Metal-to-Metal bulk operations. These registrations SHALL not
apply to subclasses. Every unregistered exact pair SHALL retain the ordinary
elementwise fallback or documented refusal rather than inheriting a nearby
registration.

Metal MAY serve as an Evictable tier only when every exact movement edge needed
by the hierarchy's construction, promotion, eviction, results, and gradients
is registered and supports the hierarchy's dtype. An unavailable edge SHALL
make construction fail before tier ownership transfers.

#### Scenario: Select each Metal bulk pair

- **WHEN** movement uses CPU-to-Metal, Metal-to-CPU, or Metal-to-Metal exact classes
- **THEN** the corresponding registered bulk operation performs the transfer

#### Scenario: Refuse an unusable Metal hierarchy

- **WHEN** an Evictable hierarchy would require a missing Metal movement edge
- **THEN** construction fails before either supplied tier becomes owned

### Requirement: Metal movement preserves the complete Tensor value boundary

A successful Metal movement SHALL allocate fresh destination storage of the
same dtype and physical `layout.cosize`, copy the complete physical span
addressed from the Tensor offset including holes, preserve the exact
hierarchical Layout and logical values, return a Tensor over the destination,
and release the source only after transfer completion. Stride-zero broadcast
layouts SHALL remain layouts rather than being materialized by logical size.

The implementation SHALL synchronize Metal work before CPU-visible values are
decoded, before a source whose work may still be pending is released, and before
the move is reported complete to a caller that observes host values. Metal-to-
Metal movement MAY remain asynchronous internally only when destination use and
source release preserve the same completion ordering.

Validation, allocation, copy, or synchronization failure SHALL leave the source
unreleased and usable, expose no partially published destination Tensor, and
preserve both carrier versions. A successful move SHALL not count as an in-place
value mutation of either carrier.

#### Scenario: Move a hierarchical physical span to Metal

- **WHEN** a Tensor with holes or nested modes moves from CPU to Metal
- **THEN** destination storage has the same dtype and `cosize`, and the returned
  Tensor has the exact source offset-relative values and Layout

#### Scenario: Synchronize before host decode and release

- **WHEN** a Metal-to-CPU move follows pending Metal work
- **THEN** synchronization completes before host values are decoded or Metal
  source storage is released

#### Scenario: Preserve the source after failure

- **WHEN** any Metal move step fails
- **THEN** the source remains unreleased and usable and no destination is published

### Requirement: Metal movement preserves reverse-mode carrier boundaries

A differentiable successful move SHALL attach the ordinary move autograd
boundary. Backward SHALL require an injective same-shape cotangent, use the
registered inverse exact carrier pair, and return a fresh gradient Tensor in
the source carrier class with the source layout semantics. A missing inverse
pair or failed transfer SHALL raise without substituting another carrier.

#### Scenario: Move a gradient back from Metal

- **WHEN** backward crosses a CPU-to-Metal forward move
- **THEN** the gradient synchronizes as needed and returns through Metal-to-CPU
  into fresh CPU storage

### Requirement: Metal exposes no new public device-interchange surface

Metal SHALL not opt into DLPack export in this change. Public APIs SHALL not
expose the private PyTorch MPS storage object or promise adoption of PyTorch,
Metal, or another device buffer. Friendly tensor factories SHALL remain
CPU-backed. These restrictions SHALL not prevent registered `move` operations.

Operation profiling of Metal SHALL measure the existing synchronous host
dispatch or launch boundary and SHALL not require device synchronization merely
to close an event. Device work MAY outlive that event; operations whose public
result semantics require host observation SHALL synchronize at their own
boundary.

#### Scenario: Reject Metal DLPack export

- **WHEN** a caller requests DLPack from a Metal Tensor
- **THEN** export fails through the documented non-exporting-carrier path

#### Scenario: Profile without forcing completion

- **WHEN** a Metal operation launches asynchronously inside a profiling context
- **THEN** event completion does not itself synchronize the device

### Requirement: Move routes are exact and definition-owned

An exact carrier class MAY declare outgoing `TransferRoute`
entries in its `CarrierDefinition`. Each route SHALL name the exact source
carrier class that owns the definition, one exact destination carrier class,
a stable route identifier, and one TransferProvider whose `prepare` callback
evaluates the request's dtype, Layout, device, resource, and alignment
constraints. Route matching SHALL use class identity and SHALL not inherit a
route from either carrier's superclass.

A definition-backed pair SHALL resolve only through its declared route. When
the exact source definition declares no matching route, movement SHALL use the
documented elementwise fallback only if both exact definitions permit that
fallback; otherwise it SHALL fail with `NotImplementedError` naming both exact
classes before allocation or submission. A route
provider SHALL implement transfer mechanics and SHALL not redefine move
validation, source release, autograd, completion, or publication semantics.

#### Scenario: Resolve an exact custom route

- **WHEN** a definition-backed custom source declares a route to one exact
  custom destination class
- **THEN** movement prepares that route rather than a route declared for a
  parent or sibling class

#### Scenario: Refuse inherited transfer support

- **WHEN** a subclass has no exact definition route for a pair supported by
  its parent
- **THEN** that parent's route is not selected

### Requirement: move_async validates one movement request

`move_async(tensor, destination)` SHALL validate and eagerly initiate one
logical move and return `AwaitMove[Tensor]`. `tensor` SHALL be a Tensor and
`destination` SHALL be a live, publicly mutable Carrier of the Tensor's dtype.
The Tensor's source Carrier and `destination` SHALL be different Carrier
objects; using the same Carrier object in both roles SHALL fail with
`ValueError` before allocation, submission, version change, publication, or
release.

`move_async` SHALL not accept Tensor or destination sequences. Passing a
sequence in either position SHALL fail with `TypeError` before allocation,
submission, version change, publication, or release.

#### Scenario: Initiate one move

- **WHEN** a valid Tensor and destination are supplied
- **THEN** `move_async` returns an `AwaitMove` after submitting all immediately
  available transfer work

#### Scenario: Reject batched movement

- **WHEN** a caller supplies Tensor or destination sequences
- **THEN** `move_async` raises `TypeError` before any route allocates, copies,
  releases, or submits work

### Requirement: Await results share one completion contract

`AwaitResult[T]` SHALL be the read-only public completion base for
framework-created `AwaitMove[T]`, `AwaitProjection`, and `AwaitResidency`
handles. It SHALL expose the `done: bool` property, blocking `wait() -> T`, and
no Python coroutine protocol. `done` SHALL become true
exactly when a terminal success or failure has been recorded. `wait()` SHALL
block only until that terminal state and SHALL then return the completed value
or raise the recorded exception.

Completion SHALL be thread-safe and idempotent. Repeated or concurrent calls to
`wait()` SHALL observe the same terminal outcome; a successful
object result SHALL have the same identity on every observation. Completion
callbacks and profiling finalization SHALL run at most once. The public
completion interface SHALL expose neither `.await()` nor `__await__`.

`AwaitMove[Tensor]` SHALL complete with the moved Tensor;
`AwaitProjection` SHALL complete with one compact ordinary Tensor; and
`AwaitResidency` SHALL complete with `None`.

`move_async`, `AwaitResult`, and `AwaitMove` SHALL be importable from both the
top-level package and `strideweave.carriers.move`. `AwaitProjection` and
`AwaitResidency` SHALL be importable from
`strideweave.carriers.tiled_evictable`.

#### Scenario: Wait for a move more than once

- **WHEN** two threads call `wait()` on one successful `AwaitMove[Tensor]`
- **THEN** both receive the identical moved Tensor after one terminal commit

#### Scenario: Wait for a failed projection

- **WHEN** projection has recorded a transfer failure
- **THEN** every `wait()` raises that same terminal failure

### Requirement: Move submission is eager but does not force host synchronization

After validation and route resolution, `move_async` SHALL initiate every
provider transfer that can start without waiting for another asynchronous
transfer before returning. From that point until terminal completion, the
source value and supplied destination SHALL remain valid for the request and
SHALL be protected from conflicting mutation, release, or movement. The call
SHALL not wait for a device event, I/O completion, result validation, source
release, or terminal publication merely to return the handle.

Validation or route resolution that fails before submission SHALL raise
directly from `move_async` and SHALL not return a handle. A failure after the
first provider submission SHALL become the handle's terminal failure and SHALL
be observed through `wait()`.

#### Scenario: Return while a device copy is pending

- **WHEN** a transfer provider submits an asynchronous device copy
- **THEN** `move_async` returns an incomplete `AwaitMove` without synchronizing
  the calling thread

### Requirement: One asynchronous move publishes at one terminal boundary

Until transfer and result validation complete successfully, no result Tensor
SHALL be published and the source SHALL not be released. On success, the
destination value and result Tensor SHALL publish before the source carrier is
released, and the handle SHALL then record success. On failure, the source
SHALL remain live and logically unchanged, no result SHALL be published, and
the source and destination SHALL again satisfy their existing movement failure
contracts before the handle records failure.

#### Scenario: Fail one asynchronous move

- **WHEN** provider completion or result validation fails
- **THEN** the source remains live, no result Tensor is returned, and `wait()`
  raises the recorded terminal exception

### Requirement: Pending movement preserves lifetime through completion

After successful initiation and until terminal completion, every source,
destination, provider resource, and autograd value needed by the move SHALL
remain valid even if the caller drops the handle. Mutation, release, or another
move of a participating source SHALL fail with `RuntimeError` during that
interval. Reads MAY continue only when they observe the captured logical
version; a route that cannot preserve that value SHALL reject initiation before
provider submission.

Dropping all caller references to an incomplete handle SHALL not release a
source early, expose an incomplete destination, cancel the move, or block the
dropping thread. The request SHALL still reach one terminal outcome, restore
the usability required by that outcome, and relinquish every resource it kept
unavailable. Cancellation SHALL not be exposed by this public contract.

#### Scenario: Drop a pending handle

- **WHEN** a caller discards its last `AwaitMove` reference while a device
  event is pending
- **THEN** dropping the handle does not block, and the transfer still preserves
  participating values and resources until its terminal outcome

### Requirement: Profiling distinguishes submission from completion

When operation profiling is enabled, a move SHALL record host submission and
terminal completion as distinct events correlated to the same logical move and
request. Provider execution time SHALL include asynchronous device or
storage activity through completion rather than only enqueue latency. Calling
`wait()` repeatedly SHALL not duplicate the terminal event.

#### Scenario: Profile an asynchronous Metal move

- **WHEN** Metal submission returns before its device event and the move later
  completes
- **THEN** profiling reports correlated submission and completion timing for
  one logical move

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
