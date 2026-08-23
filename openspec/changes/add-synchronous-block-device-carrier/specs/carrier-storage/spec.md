## MODIFIED Requirements

### Requirement: Storage support is exact, structural, and allocation-free

`supports_storage_dtype(dtype)` asks whether the carrier implementation can
allocate homogeneous storage for `dtype`. `dtype` SHALL be a `DType`; otherwise,
the call SHALL fail with `TypeError`. It SHALL recognize descriptors by object
identity and return `False`, rather than fail, for a valid descriptor outside
the implementation's accepted set.

The answer SHALL allocate nothing, mutate nothing, and remain unchanged by the
instance's current dtype, size, intrinsic mutability, ownership, residency, or
release state. An ordinary custom carrier that does not widen its support SHALL
conservatively report support only for the dtype it currently holds.

The accepted sets SHALL be:

| Carrier | Supported storage dtypes |
| --- | --- |
| `Generic` | `DType.Any`, `DType.Floating`, `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `CPU` | `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `FileBacked` | `DType.Floating`, `DType.Float32`, `DType.Int32` |
| `BlockDeviceCarrier` | `DType.Float32` |
| `Evictable` | The identity-based intersection reported by its primary and secondary tiers |

Every descriptor outside the applicable row SHALL be unsupported, including
`DType.Integer`, `DType.Float64`, narrow structural encodings, and compound
dtypes.

#### Scenario: Query support independently of current storage

- **WHEN** a `CPU` currently holding `Float32` is asked whether it supports
  `DType.Int32`
- **THEN** it returns `True` without changing or allocating storage

#### Scenario: Query block-device storage support

- **WHEN** a live or released `BlockDeviceCarrier` is asked about
  `DType.Float32` and `DType.Int32`
- **THEN** it returns `True` and `False`, respectively, without allocation or
  device I/O

#### Scenario: Query a valid unsupported descriptor

- **WHEN** any carrier is asked whether it supports a compound dtype
- **THEN** it returns `False`

### Requirement: Every successful public mutation advances the version

`version` SHALL expose a non-negative monotonic integer. Every successful
public mutation that can change stored values SHALL increment the visible
carrier version, including indexed assignment, `set_value`, and a completed
`scatter` call. An indexed assignment or `set_value` call SHALL increment it
exactly once. A scatter implementation MAY perform multiple constituent writes
and therefore MAY increment it more than once.

A mutation that fails during preflight, before its destination can change,
SHALL leave both values and version unchanged. A synchronous block-device scalar
write or CPU/block-device bulk move SHALL instead increment its destination
version exactly once after preflight succeeds and immediately before its first
transfer attempt. That increment SHALL remain visible when a later terminal
transfer failure leaves the destination unchanged or possibly partially changed.

Storage allocation, construction, release, and Evictable residency transitions
SHALL NOT count as value mutations and SHALL NOT increment the visible version.
Mutation through an Evictable wrapper SHALL increment the wrapper version once;
the wrapper SHALL remain the version authority visible to its Tensors.

#### Scenario: Version an indexed write

- **WHEN** a public indexed assignment succeeds
- **THEN** the carrier version is exactly one greater than before the call

#### Scenario: Preserve a started-transfer version after failure

- **WHEN** a validated block-device scalar write or CPU/block-device bulk move
  advances the destination version and then fails during transfer
- **THEN** the destination version remains exactly one greater than before the
  call so saved-version validation can detect the possible mutation

#### Scenario: Preserve version across eviction and promotion

- **WHEN** an Evictable carrier evicts or promotes without a logical value
  write
- **THEN** its visible version remains unchanged

## ADDED Requirements

### Requirement: BlockDevice opens one explicit supported device node

`BlockDevice(path)` SHALL open and return one process-local block-device arena.
`path` names the exact device node to open and SHALL be a string or a Python
filesystem-path object whose filesystem representation is a string; another
value or a bytes-valued representation SHALL fail with `TypeError`. An exception
raised while obtaining a filesystem representation SHALL propagate unchanged.

On Linux, the opened node SHALL be a block-special device. On macOS, it
SHALL be a buffered block-special device such as `/dev/diskN`. A regular file,
directory, or macOS raw character device such as `/dev/rdiskN` SHALL fail with
`ValueError`. On another operating system, construction SHALL fail with
`NotImplementedError`.

Construction SHALL open the supplied path for reading and writing, validate the
exact opened device resource rather than discovering or substituting another node, and
obtain a positive byte capacity, positive logical block size, and positive
physical block size from it. Capacity SHALL be a multiple of the logical block
size, and the physical block size SHALL be a multiple of the logical block size.
A path that cannot be opened SHALL propagate `OSError`; inconsistent geometry
SHALL fail with `ValueError`; geometry whose capacity cannot be returned exactly by
`capacity_bytes` or used to address every valid Float32 physical slot SHALL fail
with `OverflowError`. Every construction failure SHALL close any device resource it
opened and SHALL perform no device write.

`path`, `capacity_bytes`, `logical_block_size`, `physical_block_size`, and
`allocation_alignment` SHALL return the path string and discovered immutable
geometry. `path` SHALL equal the exact string produced from the accepted input by
Python's filesystem-path protocol, with no absolute-path conversion, lexical
normalization, or symlink resolution. `allocation_alignment` SHALL be the greater
of the logical and physical block sizes. `is_closed()` and `is_faulted()` SHALL
return the current lifecycle state without device I/O.

#### Scenario: Open an explicit Linux block device

- **WHEN** a caller supplies a readable and writable Linux block-special node
  with valid geometry
- **THEN** construction returns an open `BlockDevice` reporting that exact
  node's capacity and block sizes without writing it

#### Scenario: Open a buffered macOS block device

- **WHEN** a caller supplies a readable and writable macOS `/dev/diskN`
  block-special node with valid geometry
- **THEN** construction returns an open `BlockDevice` reporting that exact
  node's capacity and block sizes without writing it

#### Scenario: Reject an unsupported storage object

- **WHEN** the supplied path opens a regular file or a macOS raw character node
- **THEN** construction fails with `ValueError`, closes the opened device resource, and
  leaves the storage object unchanged

#### Scenario: Preserve an accepted relative path representation

- **WHEN** a path-like input returns the relative string `devices/model-tier`
- **THEN** `BlockDevice.path` returns exactly `devices/model-tier` after the exact
  opened node is validated

#### Scenario: Reject a bytes-valued path representation

- **WHEN** a path-like input returns bytes instead of a string
- **THEN** construction fails with `TypeError` before opening a device resource or
  writing storage

### Requirement: BlockDevice allocates isolated aligned carrier extents

`device.allocate(size, *, mutable=True, dtype=DType.Float32, empty=False)` SHALL
return a fresh `BlockDeviceCarrier` backed by the receiving `BlockDevice`.
`size` names the requested number of Float32 physical slots and SHALL support
Python's integer-index protocol; another value SHALL fail with `TypeError`, and a
negative value SHALL fail with `ValueError`. A value for which `size * 4`, its
aligned padded extent length, or its candidate end offset cannot be computed exactly
as a non-negative device byte position SHALL fail with `OverflowError` before
allocation. `mutable` names the carrier's
intrinsic mutability, SHALL be optional, SHALL default to `True`, and SHALL be an
exact Python `bool`; another value SHALL fail with `TypeError`. `dtype` names the
storage dtype, SHALL be optional, and SHALL default to `DType.Float32`; a
non-`DType` SHALL fail with `TypeError`, and any descriptor other than the
identical `DType.Float32` SHALL fail with `ValueError`. `empty` names whether
initialization may be skipped, SHALL be optional, SHALL default to `False`, and
SHALL be an exact Python `bool`; another value SHALL fail with `TypeError`.

For a live carrier, its payload byte span SHALL mean the contiguous unpadded
`size() * 4` bytes that hold its Float32 physical slots, beginning at
`device_offset_bytes`. A Tensor's Layout SHALL map its logical coordinates into
those physical slots; the payload byte span SHALL be determined only by carrier
size. For a positive `size`, the returned carrier SHALL reserve one contiguous
padded byte extent whose
start and byte length are multiples of `allocation_alignment` and whose length
covers the payload byte span. Its reserved extent SHALL overlap no other live
allocation from that `BlockDevice`. A zero-size carrier SHALL have an empty payload
byte span and reserve no bytes. Allocation SHALL select the lowest suitable aligned
unreserved byte range and SHALL fail with `MemoryError` without reserving bytes or
changing any existing carrier when no such range exists.

The carrier's `device`, `device_offset_bytes`, and `extent_bytes` SHALL return
the same `BlockDevice`, reserved byte offset, and padded byte length; a zero-size
carrier SHALL return zero for both byte values. With `empty=False`, every Float32
physical slot SHALL contain zero before the carrier is returned. With `empty=True`,
initial physical-slot values SHALL have no readable-value guarantee until written.
Callers SHALL write every physical slot before reading it. If initialization or
carrier creation fails after reserving a range, the complete range SHALL become
available again and no carrier SHALL be returned.

Every `BlockDeviceCarrier` allocation SHALL keep its construction-time size for its
live lifetime and SHALL act as a fixed-size `move` destination using only its
existing physical slots. A positive-size CPU Tensor moved into a zero-size
BlockDeviceCarrier SHALL therefore fail with
`ValueError` as an undersized destination before changing either carrier, allocating
an extent, issuing device I/O, advancing a version, faulting the device, or releasing
the source.

Allocation isolation SHALL cover carriers produced through `allocate`,
`new_like`, and `allocate_like` on the same `BlockDevice` instance. It SHALL make
no allocation-coordination claim across processes or across distinct
`BlockDevice` instances that open the same underlying device.

`BlockDeviceCarrier` SHALL be public for imports and runtime type checks but
SHALL have no independent direct-construction form. Calling
`BlockDeviceCarrier(...)` SHALL fail with `TypeError` directing the caller to
`BlockDevice.allocate` before opening storage, reserving an extent, or issuing
device I/O.

`device.allocate(...)` SHALL fail with `RuntimeError` when the receiving device
is faulted or closed, before reserving an extent or issuing device I/O. A closed
device remains directly reachable through the `BlockDevice` object after a
successful `close()` even though no carrier object may coexist with that close.

#### Scenario: Allocate two non-overlapping carriers

- **WHEN** one device allocates two positive-size carriers
- **THEN** both extents are aligned and disjoint, and writes through either
  carrier leave the other's physical slots unchanged

#### Scenario: Validate allocation flags exactly

- **WHEN** `mutable` or `empty` is an integer, truthy object, or other value
  whose exact type is not `bool`
- **THEN** allocation fails with `TypeError` before reserving an extent or
  issuing device I/O

#### Scenario: Fail allocation atomically

- **WHEN** no aligned free extent can cover the requested size or initialization
  fails
- **THEN** allocation raises, returns no carrier, leaves every pre-existing carrier
  and extent unchanged, and makes no device bytes unavailable to a later allocation

#### Scenario: Reject allocation on a closed device

- **WHEN** a caller successfully closes a BlockDevice and then calls
  `device.allocate(...)` on that same device object
- **THEN** allocation fails with `RuntimeError` before reserving an extent or
  issuing device I/O

#### Scenario: Reject move growth of a zero-size block allocation

- **WHEN** a positive-size CPU Tensor is moved into a live mutable
  BlockDeviceCarrier returned by `device.allocate(0)`
- **THEN** move fails with `ValueError`, the block carrier remains zero-size, and
  neither carrier, device health, extent availability, nor lifecycle changes

### Requirement: One BlockDevice serializes storage and lifecycle operations

Allocation, extent-availability changes, release, factories, lifecycle transitions,
scalar transfers, and bulk transfers associated with one `BlockDevice` SHALL be
ordered so at most one such operation proceeds on that device at a time, including
operations on disjoint carrier extents. Operations associated with distinct
`BlockDevice` instances have no cross-device serialization guarantee.

A `release()` call that races an active transfer on the same device SHALL wait
until that transfer returns or raises. Only then SHALL release return the
carrier's extent, so a later allocation SHALL never reuse bytes while the
earlier transfer can still access them.

Caller-controlled work directly invoked while a public API obtains and normalizes
its inputs SHALL complete before the corresponding ordered device effect begins.
This covered work includes the filesystem-path protocol, integer-index protocol,
iterable steps and length hints, and Float32 conversion. It MAY reenter the same
BlockDevice and SHALL neither deadlock nor observe a partly committed outer
operation. If it raises, the outer call SHALL fail without allocation, extent
change, device I/O, version change, lifecycle change, or device fault.

A cleanup or finalization callback directly induced by completing, failing, or
cleaning up that operation SHALL begin only after the operation's device effects
and public lifecycle, version, and failure state have reached the state that its
caller can observe. Such a callback MAY reenter the same BlockDevice and SHALL
neither deadlock nor observe a partly committed outer operation.

Signals, tracing or profiling hooks, cyclic-garbage-collection finalizers, and
other asynchronous interpreter callbacks that are not directly induced by the
operation are outside the v0 callback and reentrancy guarantee.

After input processing completes and immediately before effects, the operation
SHALL revalidate every mutable precondition relevant to that call, including device
health and open state, carrier lifecycle and allocation identity, ownership,
mutability, bounds, and available capacity. A concurrent state change that makes a
precondition false SHALL produce the ordinary public failure before effects. In
particular, `new_like(values)` SHALL completely materialize and Float32-normalize
`values` before it reserves an extent or writes the device.

#### Scenario: Serialize disjoint transfers

- **WHEN** two threads start transfers through different carriers allocated by
  one device
- **THEN** one transfer finishes or fails before the other enters its device
  transfer

#### Scenario: Wait before reusing a released extent

- **WHEN** release races a transfer already active on the same device
- **THEN** release waits for the transfer to finish and the extent remains
  unavailable to later allocation until release completes

#### Scenario: Reenter during input normalization

- **WHEN** an integer-index conversion, Float32 conversion, or `new_like` iterable
  performs a same-device operation before returning its normalized value
- **THEN** the nested operation can complete without deadlock, and the outer
  operation subsequently revalidates current state before effects

#### Scenario: Reenter during operation-induced cleanup

- **WHEN** completing, failing, or cleaning up an ordered device operation directly
  invokes a cleanup or finalization callback that performs a same-device operation
- **THEN** the outer operation's observable state is settled before the callback
  begins, and the nested operation completes or raises from that state without
  deadlock or exposure of an intermediate outer state

#### Scenario: Fail materialization without device effects

- **WHEN** `new_like(values)` encounters an iteration, length-hint, or Float32
  conversion exception while materializing `values`
- **THEN** it propagates that exception without allocation, extent change, device
  I/O, version change, lifecycle change, or device fault

### Requirement: BlockDeviceCarrier provides complete synchronous Float32 access

A live `BlockDeviceCarrier` SHALL expose the ordinary carrier `size`, `dtype`,
indexed access, `get_value`, `set_value`, mutability, ownership, and version
contracts for its Float32 physical slots. A successful read SHALL return the exact
binary32 value stored in the selected slot. A successful write SHALL normalize
the value to Float32, advance the carrier version exactly once at transfer start,
complete its positioned device transfer synchronously, and return `None`.

Every storage transfer SHALL first prove that its byte range lies inside both
the carrier's payload byte span and padded reserved device extent. It SHALL retry
an interrupted transfer and SHALL advance the byte position after a positive short
completion until the requested range is complete. A zero-byte completion before
the requested range is complete SHALL be a terminal I/O failure and SHALL raise
`RuntimeError` identifying the incomplete transfer. A terminal operating-system
I/O error SHALL propagate as `OSError`.

`BlockDeviceCarrier.scatter(...)` SHALL fail with `NotImplementedError` before
device I/O, value change, or version change. DLPack behavior is owned by
`interop-movement`.

#### Scenario: Round-trip one Float32 slot

- **WHEN** a caller writes and then reads one valid slot on a supported device
- **THEN** the write returns `None`, the version advances once, and the read
  returns the normalized binary32 value

#### Scenario: Reject a physical-slot out-of-range access before I/O

- **WHEN** an index is negative or at least the carrier size
- **THEN** access fails with `IndexError` before device I/O, version change, or
  device fault

#### Scenario: Complete a short or interrupted transfer

- **WHEN** a transfer is interrupted or completes a positive prefix
- **THEN** it resumes at the remaining byte position and succeeds only after
  every requested byte has transferred

### Requirement: Terminal device transfer failures make shared storage fail closed

After preflight succeeds, any terminal device-transfer error or zero-byte
completion before the requested transfer is complete SHALL fault the associated
`BlockDevice`, including an error on the first transfer attempt. The triggering
operation SHALL propagate `OSError` for an operating-system error and SHALL raise
`RuntimeError` for a premature zero completion. Type, dtype, mutability,
ownership, lifecycle, and bounds failures detected before transfer SHALL leave the
device healthy.

Once faulted, every carrier from that device SHALL reject value access, mutation,
allocation, and factories with `RuntimeError` before further device I/O. Carrier
release, lifecycle queries, and device close SHALL remain available. A write
whose transfer started SHALL retain its one destination-version increment even
when no byte is known to have changed. No later public storage access SHALL
expose possibly partial device values.

Every non-triggering live carrier SHALL retain its reported device, size, offset,
and extent without overlap or reassignment. The triggering carrier SHALL likewise
retain its extent and lifecycle until an explicit release, and each explicit release
SHALL retire that carrier's relationship with its original extent exactly once.
On a terminally faulted device this cleanup SHALL make no capacity-reuse promise,
because all later allocation is rejected. These cleanup operations SHALL continue
to permit device close after the last carrier object is gone.

#### Scenario: Fault after a terminal write failure

- **WHEN** a validated scalar write starts and then encounters a terminal error
  or premature zero completion
- **THEN** the write raises, its version remains advanced once, the device
  becomes faulted, and every carrier from that device rejects later storage
  access

#### Scenario: Keep preflight failures non-faulting

- **WHEN** an operation rejects an invalid dtype, lifecycle state, ownership,
  mutability, index, or byte range before transfer
- **THEN** it leaves the device health, destination version, and stored values
  unchanged

#### Scenario: Release storage after a fault

- **WHEN** a carrier's device is faulted
- **THEN** releasing the carrier returns `None`, marks it released, zeros its
  reported size, device offset, and extent length, performs no further device I/O,
  and permits eventual device close after the carrier object is destroyed

### Requirement: Carrier release retires its extent relationship exactly once

`release()` on an unowned live `BlockDeviceCarrier` SHALL wait for any active
operation on its device, retire the carrier's relationship with its complete extent,
mark the carrier released, set its reported size, device offset, and extent length
to zero, and return `None`. It SHALL be idempotent and SHALL perform that retirement
at most once. On an open healthy device, adjacent released extents SHALL become
jointly reusable, so a later aligned allocation that fits only their combined span
SHALL succeed without overlapping another live carrier. On a faulted device,
release SHALL preserve the same public carrier-state transition but no later
allocation or factory may observe or reuse the retired capacity. Destruction of an
unreleased carrier SHALL perform the same one-time retirement before the object
relationship with the device ends.

A released carrier SHALL retain its dtype, structural storage support,
intrinsic mutability setting, and access to its `BlockDevice`. While the carrier is
reachable, `carrier.device` SHALL return that same open device and its permitted
lifecycle and factory operations SHALL remain usable as specified. Value
access and storage-dependent operations SHALL fail with `RuntimeError`. Its
`new_like(values, *, mutable=True, dtype=None)` and
`allocate_like(size, *, mutable=True, dtype=None, empty=False)` factories SHALL
retain their existing input, default, return, and failure contracts while
allocating fresh independent extents from the same device. For both factories,
`mutable` SHALL be an exact Python `bool`; another value SHALL fail with
`TypeError`. For `allocate_like`, `empty` SHALL likewise be an exact Python
`bool`; another value SHALL fail with `TypeError`. A `None` value materialized
by `new_like` SHALL produce a Float32 zero in that physical slot.
`allocate_like(size, empty=False)` SHALL return a fresh same-device extent with
every Float32 physical slot initialized to zero. `allocate_like(size, empty=True)`
SHALL return a fresh same-device empty allocation whose initial physical-slot
values have no readable-value guarantee until written; callers SHALL write every
physical slot before reading it. The factories SHALL remain usable while the
prototype's device is open and healthy and SHALL fail with `RuntimeError` when it
is faulted. A successfully closed device cannot retain a live or released carrier
prototype because the close guard requires every carrier object to have been
destroyed.

#### Scenario: Reuse adjacent released space

- **WHEN** two adjacent carriers are released and a later request fits only
  their combined range
- **THEN** allocation succeeds in their jointly reusable aligned span without
  overlapping another live carrier

#### Scenario: Allocate from a released prototype

- **WHEN** `new_like(values)` is called on a released `BlockDeviceCarrier` whose
  device remains open and healthy
- **THEN** it returns a fresh carrier on the same device without reading the
  released extent

#### Scenario: Select BlockDeviceCarrier factory initialization

- **WHEN** `allocate_like(size, empty=False)` and `allocate_like(size, empty=True)`
  are called on live or released `BlockDeviceCarrier` prototypes whose device is
  open and healthy
- **THEN** each returns a fresh independent same-device extent, the initialized
  result contains Float32 zero in every physical slot, and the empty result makes
  no readable-value guarantee until every slot is written

#### Scenario: Reject a non-Boolean factory flag

- **WHEN** a BlockDeviceCarrier factory receives a `mutable` or `empty` value
  whose exact type is not `bool`
- **THEN** it fails with `TypeError` before allocation or device I/O

#### Scenario: Release twice

- **WHEN** a caller invokes `release()` more than once on one carrier
- **THEN** every call returns `None`, the public released state is unchanged, and
  cleanup of the original extent occurs at most once

### Requirement: BlockDevice close waits for every carrier object

`close()` SHALL close the opened `BlockDevice` resource and return `None` only when
no `BlockDeviceCarrier` object created from that device remains alive. This
guard SHALL include released carriers because they remain valid factory
prototypes. When any such object remains, `close()` SHALL fail with
`RuntimeError` and leave the device open and retryable. Once close succeeds it
SHALL be idempotent, `is_closed()` SHALL return `True`, and allocation and
device-dependent operations SHALL fail with `RuntimeError`. Close SHALL leave
management of the caller-supplied device node to the caller.

`BlockDevice.__enter__()` SHALL return the same device.
`BlockDevice.__exit__(exc_type, exc_value, traceback)` SHALL accept the exception
type, exception value, and traceback supplied by Python for the managed block;
each SHALL be `None` when the block completed normally, and otherwise SHALL
describe the exception being unwound. `__exit__()` SHALL always call `close()`.
When close succeeds, it SHALL return `False`, so any managed-block exception
propagates unchanged. When close fails, `__exit__()` SHALL propagate the close
exception; if the managed block also raised, the close exception SHALL take
precedence and the original managed-block exception SHALL remain available as its
ordinary Python exception context. While a carrier object is reachable, its
BlockDevice SHALL remain open and available through `carrier.device` even when
external device references are dropped.

#### Scenario: Refuse close with a surviving carrier

- **WHEN** a live or released carrier object from a device remains reachable and
  a caller invokes `close()`
- **THEN** close fails with `RuntimeError`, the device remains open, and the
  carrier's existing lifecycle behavior remains available

#### Scenario: Close after the last carrier object is gone

- **WHEN** every carrier object from the device has been destroyed
- **THEN** `close()` closes the opened device resource, returns `None`, and a repeated call
  also returns `None`

#### Scenario: Preserve context-manager exception behavior

- **WHEN** `__exit__(exc_type, exc_value, traceback)` runs after normal completion,
  a managed-block exception, a close failure, or simultaneous managed-block and
  close exceptions
- **THEN** it always attempts close, returns `False` after successful close,
  propagates a close failure when one occurs, and retains a simultaneous
  managed-block exception through ordinary Python exception chaining
