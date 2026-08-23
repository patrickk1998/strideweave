## MODIFIED Requirements

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
| `BlockDeviceCarrier` | Unsupported; fails with `BufferError` |
| `Evictable` | Unsupported in both residencies, as `carrier-composition` requires; fails with `BufferError` naming the Evictable carrier |

#### Scenario: Export a CPU tensor

- **WHEN** a `Float32` CPU tensor is exported
- **THEN** `__dlpack_device__()` reports `(1, 0)` and the consumer aliases the
  carrier's memory

#### Scenario: Reject a non-exporting carrier

- **WHEN** a Generic, FileBacked, BlockDeviceCarrier, or Evictable tensor is
  exported or queried for its DLPack device
- **THEN** both calls fail with `BufferError` and no capsule is produced

#### Scenario: Report a foreign device verbatim

- **WHEN** a carrier whose hook reports a non-CPU device type exports
- **THEN** `__dlpack_device__()` and the managed tensor both carry that device
- **AND** a `dl_device` request naming it is accepted

### Requirement: Move dispatches on the exact carrier class pair

The concrete move operation SHALL be selected from the move registry, which is
owned by neither carrier, by the exact `(type(source carrier), type(destination
carrier))` pair. `dispatch_move(source_class, destination_class)` names that
pair through the carrier class of the tensor and the carrier class of the
destination, and SHALL return the registered class for that exact pair or
`ElementwiseMoveOperation` when the pair is unregistered and neither exact class is
`BlockDeviceCarrier`. When an unregistered pair contains `BlockDeviceCarrier`,
`dispatch_move` SHALL fail with `NotImplementedError`.

Dispatch SHALL match both classes by identity, so a registration SHALL apply to
its exact pair alone and a subclass of a registered carrier class SHALL resolve
to the elementwise fallback until registered on its own when neither resulting
exact class is BlockDeviceCarrier. An unregistered subclass pair involving
BlockDeviceCarrier SHALL follow the unsupported-block-pair failure above.

Before `move(tensor, destination)` asks `dispatch_move` to select an operation, it
SHALL perform the inherited validation sequence through destination dtype
validation. A non-Tensor, released or owned source, non-Carrier destination, same
source and destination carrier, released or immutable destination, or mismatched
destination dtype SHALL therefore report its inherited error before an unsupported
block pair can report `NotImplementedError`. Standalone `dispatch_move` has no
instances to validate and SHALL apply the exact-class policy directly.

The registry SHALL be pre-populated with `CpuToFileBackedMoveOperation` for
`(CPU, FileBacked)`, `FileBackedToCpuMoveOperation` for `(FileBacked, CPU)`,
`CpuToBlockDeviceMoveOperation` for `(CPU, BlockDeviceCarrier)`, and
`BlockDeviceToCpuMoveOperation` for `(BlockDeviceCarrier, CPU)`. Each built-in
pair SHALL copy the tensor's whole physical byte span with a specialized bulk
transfer. The two registered CPU/block pairs SHALL be the complete supported set
involving BlockDeviceCarrier. Every other pair involving block storage, including
block-to-block and pairs with Generic, FileBacked, or a custom carrier, SHALL fail
with `NotImplementedError`. A public `move` that encounters this refusal SHALL fail
before copy or any destination size, version, lifecycle, offset, or extent change,
device I/O, device fault, or source release. Every other unregistered non-block pair, including
`(CPU, CPU)`, SHALL use `ElementwiseMoveOperation`, which copies logical values one
at a time.

#### Scenario: Select a registered bulk operation

- **WHEN** a CPU tensor is moved into a FileBacked destination, or the reverse
- **THEN** the registered native bulk operation for that pair runs

#### Scenario: Select a registered block-device bulk operation

- **WHEN** a CPU tensor is moved into a BlockDeviceCarrier destination, or the
  reverse
- **THEN** the registered synchronous bulk operation for that exact pair runs

#### Scenario: Fall back for an unregistered pair

- **WHEN** the pair has no registration and neither exact class is
  BlockDeviceCarrier
- **THEN** `ElementwiseMoveOperation` runs and copies logical values

#### Scenario: Reject an unsupported block-device pair

- **WHEN** a caller dispatches or invokes move for BlockDeviceCarrier-to-
  BlockDeviceCarrier, Generic-to-BlockDeviceCarrier, BlockDeviceCarrier-to-
  FileBacked, or another unregistered pair involving block storage
- **THEN** the call fails with `NotImplementedError` before allocation, copy,
  device I/O, version change, device fault, or source release

#### Scenario: Preserve move validation before unsupported dispatch

- **WHEN** `move` receives both an invalid source or destination under the inherited
  validation sequence and an otherwise unsupported pair involving
  BlockDeviceCarrier
- **THEN** it reports the earlier inherited validation error before dispatch and
  leaves values, versions, allocations, lifecycle, device health, and source release
  unchanged

#### Scenario: Do not inherit a registration

- **WHEN** a carrier class is registered and a subclass of it is used as the
  source in a pair that does not involve BlockDeviceCarrier
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
in place for the first caller. For an entry not protected by the block policy,
registering a pair that already has an entry SHALL fail with `ValueError` naming
both class names and SHALL retain the existing entry, so replacing it SHALL
require an explicit unregistration first. Re-registering an ordinary built-in
pair SHALL therefore fail the same way rather than succeed as a no-op.

The exact `(CPU, BlockDeviceCarrier)` and `(BlockDeviceCarrier, CPU)` entries
SHALL be protected built-ins. Public registration SHALL fail with `ValueError`
before registry mutation when either class is BlockDeviceCarrier, whether the
call attempts to replace one of those entries or install any other block pair.
Public unregistration SHALL likewise fail with `ValueError` and retain the
protected entry when the pair is one of the two exact built-ins, and SHALL fail
with `ValueError` without mutation for every other pair involving
BlockDeviceCarrier.

For pairs that do not involve BlockDeviceCarrier,
`unregister_move_operation(source_class, destination_class)` names the same
pair, SHALL remove and return its registered operation class, and SHALL fail
with `KeyError` naming both class names when the pair has none. It and
`dispatch_move` SHALL accept `Carrier` subclasses; their behavior for any other
argument is unspecified.

`registered_move_operation(source_class, destination_class, operation_class)`
names the same three inputs and SHALL be a context manager that registers an
ordinary non-block pair on entry, yields `operation_class`, and unregisters on
exit, including when the block raises. When either pair class is
BlockDeviceCarrier, entering the context SHALL fail with `ValueError` before it
yields or changes the registry, so it cannot temporarily shadow a protected
built-in or install an unsupported pair.

`move`, `MoveOperation`, and the concrete move operation classes SHALL be
reachable from the top-level package; `dispatch_move`,
`register_move_operation`, `unregister_move_operation`, and
`registered_move_operation` SHALL be reachable from the movement module
`strideweave.carriers.move`.

#### Scenario: Refuse to overwrite a registration

- **WHEN** a caller registers a non-block pair that already has an operation
- **THEN** the call fails with `ValueError` and the existing registration is
  retained

#### Scenario: Protect the built-in block registrations

- **WHEN** a caller attempts to register, unregister, replace, or temporarily
  shadow either exact CPU/block built-in entry
- **THEN** the public registry call fails with `ValueError`, the built-in remains
  registered, and no context body is entered

#### Scenario: Refuse another block registration

- **WHEN** a caller attempts permanent or context-managed registration or
  unregistration for another exact pair involving BlockDeviceCarrier
- **THEN** the public registry call fails with `ValueError` before registry,
  allocation, carrier, or device state changes

#### Scenario: Remove a registration

- **WHEN** a caller unregisters a registered non-block pair
- **THEN** the removed class is returned and a second unregistration fails with
  `KeyError`

#### Scenario: Scope a registration to a block

- **WHEN** a `registered_move_operation` block for a non-block pair exits,
  whether normally or by raising
- **THEN** the pair is unregistered again

## ADDED Requirements

### Requirement: Block movement policy covers every public operation entrypoint

Every public execution of `MoveOperation`, including `move()`, direct
`MoveOperation.forward`, direct `ElementwiseMoveOperation.forward`, a concrete
built-in operation, and an extension-defined subclass, SHALL apply one shared
block movement policy after the inherited validation sequence and any pinned-class
checks but before destination allocation, output Tensor construction, copy, device
I/O, version change, device fault, or source release.

When either exact endpoint class is BlockDeviceCarrier, execution SHALL proceed only
when the pair and exact operation class are `(CPU, BlockDeviceCarrier,
CpuToBlockDeviceMoveOperation)` or `(BlockDeviceCarrier, CPU,
BlockDeviceToCpuMoveOperation)`. Every other combination, including a direct
ElementwiseMoveOperation, an unpinned or pinned custom subclass, a supported pair
presented to the wrong operation class, BlockDeviceCarrier-to-BlockDeviceCarrier,
or a block pair with Generic, FileBacked, or a custom carrier, SHALL fail with
`NotImplementedError` before effects. Pairs that do not involve BlockDeviceCarrier
SHALL retain the inherited registry, fallback, pinning, and extension behavior.

#### Scenario: Reject a direct generic block move

- **WHEN** a caller directly invokes ElementwiseMoveOperation or an
  extension-defined MoveOperation for a pair involving BlockDeviceCarrier
- **THEN** execution fails with `NotImplementedError` before destination allocation,
  output construction, copy, device I/O, version change, device fault, or source
  release

#### Scenario: Require the exact operation for a supported block pair

- **WHEN** a valid exact CPU/block pair is supplied directly to any operation class
  other than its corresponding built-in specialized operation
- **THEN** execution fails with `NotImplementedError` before effects

#### Scenario: Preserve direct-operation validation precedence

- **WHEN** a direct MoveOperation call has an inherited source or destination
  validation failure, or violates that operation's pinned carrier class, and also
  falls outside the supported block policy
- **THEN** the earlier inherited or pinned-class error is reported before the block
  policy, with no allocation, copy, device I/O, version change, device fault, or
  source release

### Requirement: CPU and block-device moves transfer the physical Float32 span

`CpuToBlockDeviceMoveOperation` SHALL pin `CPU` as its source class and
`BlockDeviceCarrier` as its destination class.
`BlockDeviceToCpuMoveOperation` SHALL pin `BlockDeviceCarrier` as its source
class and `CPU` as its destination class. Both SHALL support only the identical
`DType.Float32`, as required by ordinary `move` validation, and SHALL be public
concrete `MoveOperation` classes reachable wherever the existing concrete move
operations are public.

A BlockDeviceCarrier destination SHALL be fixed-size for movement. A destination
smaller than the source layout's `cosize`, including a zero-size carrier returned by
`device.allocate(0)`, SHALL fail with `ValueError` before allocation, copy, device
I/O, destination version change, device fault, or source release. A sufficiently
sized destination SHALL continue through the exact-pair transfer contract below.

For this Float32 path, a Tensor `offset` and the copy hook's `element_count` SHALL
each name a count of physical carrier slots. `element_count` SHALL equal the source
layout's `cosize`. Each copy SHALL derive `offset_bytes = offset * 4` and
`transfer_bytes = element_count * 4`, then transfer exactly `transfer_bytes` bytes.
The transfer SHALL therefore preserve the complete physical span, including storage
holes and the exact stride-zero or gapped layout, without iterating logical
coordinates.

For CPU-to-block movement, `tensor` SHALL name the CPU source and `output` SHALL
name the block destination. The read SHALL begin at the CPU carrier base pointer
plus `tensor.offset * 4`, and the write SHALL begin at
`output.carrier.device_offset_bytes + output.offset * 4`. For block-to-CPU
movement, `tensor` SHALL name the block source and `output` SHALL name the CPU
destination. The read SHALL begin at
`tensor.carrier.device_offset_bytes + tensor.offset * 4`, and the write SHALL
begin at the CPU carrier base pointer plus `output.offset * 4`. As the ordinary
`move` contract requires, `output.offset` SHALL be zero in both directions.

Before transfer, the move SHALL validate the ordinary `move` contract and the
address arithmetic for each named endpoint. For the block endpoint, let
`block_offset_bytes = block_tensor.offset * 4` and
`block_end_bytes = (block_tensor.offset + element_count) * 4`. The
allocation-relative half-open window `[block_offset_bytes, block_end_bytes)` SHALL
fit within `[0, block_tensor.carrier.size() * 4)`, the allocation-relative length
of the payload byte span defined by `carrier-storage`, and within
`[0, extent_bytes)`. The corresponding absolute device window SHALL begin at
`device_offset_bytes + block_offset_bytes` and end at
`device_offset_bytes + block_end_bytes`. It SHALL fit within both the absolute
payload byte span
`[device_offset_bytes, device_offset_bytes + block_tensor.carrier.size() * 4)` and
the padded reserved extent
`[device_offset_bytes, device_offset_bytes + extent_bytes)`. A CPU Tensor offset
SHALL contribute only to its CPU pointer calculation; block-carrier bounds SHALL
be evaluated from the block Tensor's offset.

For every Tensor and carrier obtainable through the public API, the representation,
allocation, and ordinary `move` contracts SHALL make each derived endpoint
representable and keep the block window within its payload byte span and reserved
extent. The exact-pair operations SHALL still reject a released carrier, faulted
device, wrong pinned carrier class, dtype mismatch, undersized destination, or
ordinary ownership or mutability violation before transfer, destination version
change, or source release. Released or faulted storage and ownership or mutability
violations SHALL fail with `RuntimeError`; a wrong pinned class or dtype SHALL fail
with `TypeError`; and an undersized destination SHALL fail with `ValueError`.

#### Scenario: Reject a zero-size block destination

- **WHEN** a positive-size CPU Tensor is moved into a live mutable zero-size
  BlockDeviceCarrier
- **THEN** move fails with `ValueError` before growing the destination or changing
  either carrier or device state

#### Scenario: Round-trip a gapped tensor through block storage

- **WHEN** a CPU Float32 tensor whose layout `cosize` includes holes moves to a
  BlockDeviceCarrier and then moves back to CPU
- **THEN** both bulk operations transfer the complete `cosize` span, the final
  Tensor retains the original layout, and every logical value is unchanged

#### Scenario: Move a nonzero-offset CPU view into an exact-size block destination

- **WHEN** a CPU Float32 Tensor has a nonzero offset and is moved into a
  `BlockDeviceCarrier` whose payload byte span length is exactly
  `layout.cosize * 4` bytes
- **THEN** the move reads from the CPU base pointer plus `tensor.offset * 4`,
  writes the complete physical span at block output offset zero, and succeeds
  with block-carrier bounds evaluated from that block output offset

#### Scenario: Move a nonzero-offset block view into an exact-size CPU destination

- **WHEN** a BlockDeviceCarrier-backed Float32 Tensor has a nonzero offset whose
  allocation-relative `(tensor.offset + layout.cosize) * 4` endpoint fits its
  payload byte span and is moved into a CPU carrier containing exactly
  `layout.cosize` physical slots
- **THEN** the move reads from `device_offset_bytes + tensor.offset * 4`, writes
  the complete physical span at CPU output offset zero, and succeeds

#### Scenario: Reject released or faulted block storage before copying

- **WHEN** an exact CPU/block-device move names a released BlockDeviceCarrier or a
  carrier whose BlockDevice is faulted
- **THEN** move fails with `RuntimeError` before device I/O, destination mutation,
  destination version change, or source release

#### Scenario: Reject invalid exact-operation inputs before copying

- **WHEN** a concrete CPU/block-device move operation receives a wrong pinned
  carrier class or mismatched dtype, or ordinary move validation finds an owned
  source or an immutable or undersized destination
- **THEN** move reports the ordinary public error before device I/O, destination
  mutation, destination version change, device fault, or source release

#### Scenario: Accept an exact-fit block endpoint

- **WHEN** the block endpoint's allocation-relative byte window ends exactly at
  its payload byte span length and does not exceed the padded reserved extent
- **THEN** byte-range preflight accepts that endpoint

### Requirement: CPU and block-device move completion controls mutation and release

After all preflight validation succeeds and immediately before the first bulk
transfer attempt, a CPU/block-device move SHALL advance the destination
carrier's version exactly once. Successful completion SHALL return the ordinary
destination-backed Tensor, leave the destination version advanced once, leave
the block device healthy, and release the source carrier only after the complete
byte span has transferred synchronously.

An interrupted transfer SHALL be retried, and a positive short completion SHALL
advance the byte position and continue. A terminal I/O error or zero-byte
completion before the span is complete SHALL fail the move and fault the
associated `BlockDevice` as defined by `carrier-storage`. The source carrier
SHALL remain unreleased, no result Tensor SHALL be returned, and the destination
version SHALL retain its one transfer-start increment.

For CPU-to-block failure, the CPU source SHALL remain live and unchanged, and
possibly partial block bytes SHALL become inaccessible through the faulted
device. For block-to-CPU failure, the block source SHALL remain unreleased but
its device SHALL be faulted. Destination bytes in the transferred prefix MAY
contain copied source bytes, including a Float32 slot only partly overwritten by
a non-slot-aligned short completion; every byte beyond that prefix SHALL retain
its pre-move value. The failed move SHALL make no atomic rollback claim for that
caller-supplied CPU destination; its advanced version SHALL expose the possible
mutation to saved-version validation.

A failure before transfer start SHALL leave source and destination values, versions,
reported sizes, offsets, extents, lifecycle, and later allocation availability
unchanged. After a transfer starts, each endpoint SHALL remain live and associated
with its same carrier and extent until explicit release, and every other live
allocation SHALL remain disjoint from that extent.

#### Scenario: Keep the CPU source after a block write failure

- **WHEN** a CPU-to-block move starts and then encounters a terminal transfer
  failure
- **THEN** move raises, the CPU source remains live and unchanged, the block
  destination version remains advanced once, the block device becomes faulted,
  and no result Tensor is returned

#### Scenario: Expose a partial CPU destination through its version

- **WHEN** a block-to-CPU move starts and then encounters a terminal device-read
  failure
- **THEN** move raises without releasing the block source, returns no Tensor,
  faults the block device, leaves the CPU destination version advanced once,
  and permits only bytes in the transferred destination prefix to differ

#### Scenario: Release only after synchronous success

- **WHEN** every byte of a CPU/block-device move completes successfully
- **THEN** the destination version has advanced exactly once, the source is
  released only after completion, and the result is usable with the unchanged
  source layout

#### Scenario: Leave state unchanged on preflight failure

- **WHEN** CPU/block-device move validation fails before its first transfer
  attempt
- **THEN** neither carrier's values, version, allocation, lifecycle, or device
  health changes
