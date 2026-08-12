---
title: Carrier Storage
publish: true
status: stable
order: 40
summary: Carrier construction, dtype support, value access, factories, mutation, versioning, and release.
---

# carrier-storage Specification

## Purpose

Define the storage-facing contract shared by carriers and the concrete storage
behavior of Generic, CPU, FileBacked, and Evictable carriers.

## Terminology

| Term | Meaning |
| --- | --- |
| carrier | A `Carrier` instance that owns or references a one-dimensional homogeneous physical storage sequence and serves as its dtype-support, lifecycle, mutation/version, and operation-dispatch boundary. |
| intrinsic mutability | Whether the carrier's storage implementation permits writes. |
| public mutability | Whether public writes are currently permitted after intrinsic mutability and ownership are combined. |
| initialized allocation | Fresh storage whose readable slots contain the storage dtype's zero. |
| empty allocation | Fresh storage whose initial contents carry no readable-value guarantee because the backend may skip initialization; every slot must be written before it is read. |
| released carrier | A carrier in the terminal state where its storage has been permanently relinquished and storage-dependent access is unavailable, while dtype identity, structural storage support, and `new_like` prototype construction remain available. |

## Requirements

### Requirement: Carrier exposes one homogeneous storage sequence

For a live carrier, `size()` SHALL return its non-negative number of physical
slots and `dtype()` SHALL return the single `DType` describing every slot.
`get_value(index)` and `carrier[index]` SHALL return the value at the physical
`index`. `set_value(index, value)` and `carrier[index] = value` SHALL write one
slot when public mutation is permitted and SHALL return `None`. `index` names
the physical storage position to read or write. `value` names the new stored
value and SHALL satisfy the receiver's dtype normalization contract.

`index` SHALL be interpreted through Python's integer-index protocol and SHALL
fall within `[0, size())`. When `index` does not support that protocol, access
SHALL fail with `TypeError`; when `index` is negative or out of range, access
SHALL fail with `IndexError`. A write to a carrier that is not publicly mutable
SHALL fail with `RuntimeError` before `value` changes the stored slot.

#### Scenario: Read and write one slot

- **WHEN** a live mutable carrier has size three and a caller reads or writes
  an index in `[0, 3)`
- **THEN** the operation observes or updates that physical slot

#### Scenario: Reject an invalid physical index

- **WHEN** a caller accesses a negative index or an index at least `size()`
- **THEN** access fails with `IndexError`

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
| `Evictable` | The identity-based intersection reported by its primary and secondary tiers |

Every descriptor outside the applicable row SHALL be unsupported, including
`DType.Integer`, `DType.Float64`, narrow structural encodings, and compound
dtypes.

#### Scenario: Query support independently of current storage

- **WHEN** a `CPU` currently holding `Float32` is asked whether it supports
  `DType.Int32`
- **THEN** it returns `True` without changing or allocating storage

#### Scenario: Query a valid unsupported descriptor

- **WHEN** any carrier is asked whether it supports a compound dtype
- **THEN** it returns `False`

### Requirement: Unsupported construction fails before storage is exposed

For every concrete carrier constructor and factory, `dtype` names the requested
homogeneous storage dtype. `dtype` SHALL be a `DType` recognized by that
carrier's support set. When `dtype` is not a `DType`, the call SHALL fail with
`TypeError`. When `dtype` is a valid but unsupported descriptor, the call SHALL
fail with `ValueError`. When the descriptor is compound, the `ValueError` SHALL
explain that homogeneous carrier storage cannot hold its multiple simple-dtype
planes. Construction failure SHALL expose no partially constructed storage.

#### Scenario: Reject an unsupported simple encoding

- **WHEN** a CPU constructor or factory receives `DType.Float64`
- **THEN** it fails with `ValueError` before allocating usable storage

#### Scenario: Reject compound homogeneous storage

- **WHEN** any shipped carrier constructor receives a compound dtype
- **THEN** it fails with `ValueError` identifying the deferred multi-plane
  storage requirement

### Requirement: Public storage helpers share the identity-based storage rules

`validate_storage_dtype(dtype, *, carrier, accepted)` SHALL validate one
candidate against the exact descriptor identities in `accepted`. `dtype` names
the candidate object, `carrier` names the carrier in diagnostics, and
`accepted` names the ordered tuple of supported DType identities. It SHALL
return the unchanged `dtype` when supported, fail with `TypeError` when `dtype`
is not a `DType`, and fail with `ValueError` when `dtype` is a valid but
unsupported descriptor. When `dtype` is compound, the failure SHALL identify
the deferred multi-plane storage requirement.

`accepts_storage_dtype(dtype, accepted)` SHALL return whether `dtype` is
identical to any descriptor in `accepted`. `dtype` names the candidate object,
and `accepted` names the tuple of accepted descriptor identities. The function
SHALL perform identity rather than equality matching and SHALL return `False`
when `dtype` is not identical to any member of `accepted`.

`storage_zero(dtype)` SHALL return the initialized slot value associated with
the supplied storage dtype. `dtype` names the storage descriptor whose
initialized value is requested. The return value SHALL be `0.0` for
`DType.Float32`, `0` for `DType.Int32`, `False` for `DType.Bool`, and `None`
for a dtype without a defined concrete stored zero, including the legacy opaque
categories.

#### Scenario: Validate an exact accepted identity

- **WHEN** `validate_storage_dtype` receives a candidate identical to one entry
  of `accepted`
- **THEN** it returns that same descriptor object

#### Scenario: Obtain a concrete storage zero

- **WHEN** `storage_zero` receives `DType.Bool`
- **THEN** it returns the Python boolean `False`

### Requirement: Generic stores legacy objects and normalized concrete values

`Generic(values, *, mutable=True, dtype=DType.Floating)` SHALL consume an
iterable of stored values and return a Generic carrier. `values` names the
ordered values to place in storage and SHALL be iterable. `mutable` states the
carrier's intrinsic mutability, SHALL be optional, and SHALL default to `True`;
`dtype` names the homogeneous storage dtype, SHALL be optional, and SHALL
default to `DType.Floating`. When `values` is not iterable, construction SHALL
fail with `TypeError`.

For `DType.Any` and `DType.Floating`, sized indexable `values` MAY remain
aliased; when `mutable=True` and `values` cannot be assigned by index, Generic
SHALL materialize a mutable sequence. For `DType.Float32`,
`DType.Int32`, and `DType.Bool`, Generic SHALL materialize storage it owns and
normalize every value before the carrier is returned. Float32 values SHALL be
binary32-exact Python floats, Int32 values SHALL be integers in
`[-2**31, 2**31 - 1]`, and Bool values SHALL be Python `bool` values. Invalid
concrete values SHALL fail with `TypeError` or `OverflowError` as appropriate.

The same concrete normalization SHALL apply to later writes, including indexed
writes and scatter writes, so a caller-held alias cannot bypass it.

#### Scenario: Own concrete input storage

- **WHEN** a caller constructs a Float32 Generic from a mutable input sequence
  and later mutates that sequence
- **THEN** the carrier's stored values do not change

#### Scenario: Preserve legacy aliasing

- **WHEN** a caller constructs mutable legacy opaque Generic storage from a
  mutable sized indexable sequence
- **THEN** carrier writes and caller writes remain visible through the shared
  sequence

### Requirement: CPU owns or wraps typed host storage

`CPU(size, pointer=None, *, mutable=True, dtype=DType.Float32, empty=False)`
SHALL return host storage with `size` physical slots. `size` names the slot
count and SHALL be a non-negative integer; when `size` is not an integer,
construction SHALL fail with `TypeError`, and when `size` is negative,
construction SHALL fail with `ValueError`. `pointer` names an optional external
memory address, SHALL be optional, and SHALL default to `None`. `mutable` states
the carrier's intrinsic mutability, SHALL be optional, and SHALL default to
`True`. `dtype` names the homogeneous storage dtype, SHALL be optional, and
SHALL default to `DType.Float32`. `empty` states whether initialization of a
newly owned allocation may be skipped, SHALL be optional, and SHALL default to
`False`.

When `pointer is None`, CPU SHALL own a new allocation. Unless `empty=True`,
Float32, Int32, and Bool slots SHALL initially contain `0.0`, `0`, and `False`
respectively. When `pointer` is supplied, it names an existing memory address
and SHALL be a positive integer; another type SHALL fail with `TypeError` and a
non-positive value SHALL fail with `ValueError`. CPU SHALL wrap that address
without taking ownership or changing its contents, and `empty` SHALL not alter
external memory.

`pointer()` SHALL return the positive integer address used by the carrier.
Float32 reads and writes SHALL use binary32 values, Int32 writes SHALL require
in-range integers, and Bool writes SHALL require a Python `bool`; invalid typed
writes SHALL fail before the slot changes.

#### Scenario: Create initialized owned CPU storage

- **WHEN** a caller constructs an owned CPU with `empty=False`
- **THEN** every slot is readable and contains the zero for the selected dtype

#### Scenario: Wrap external memory

- **WHEN** a caller supplies a positive integer pointer and a valid CPU dtype
- **THEN** the CPU reads and writes that external storage without initializing
  or owning it

### Requirement: FileBacked owns a temporary raw numeric file

`FileBacked(filename=None, *, mutable=True, dtype=DType.Floating)` SHALL create
a raw numeric file inside a hidden per-process temporary directory and return
an initially empty carrier. `filename` names a bare file within that directory
and SHALL be optional; `filename` SHALL default to `None`, and `None` SHALL
request a generated unique name. When `filename` contains path components,
construction SHALL fail with `ValueError`; when `filename` duplicates an
existing file, construction SHALL fail without replacing that file. `mutable`
states the carrier's intrinsic mutability, SHALL be optional, and SHALL default
to `True`. `dtype` names the homogeneous storage dtype, SHALL be optional, and
SHALL default to `DType.Floating`.

`path` SHALL return the carrier's file path. The file SHALL encode Floating,
Float32, or Int32 values according to the selected storage dtype. New or
extended slots SHALL read as zero. Int32 writes SHALL require integer values.
Deleting or releasing the carrier SHALL remove its file, and process shutdown
SHALL remove the hidden session directory.

#### Scenario: Generate file-backed storage

- **WHEN** a caller omits `filename`
- **THEN** the result owns a uniquely named empty file in the hidden session
  directory

#### Scenario: Reject a path-like filename

- **WHEN** `filename` contains a directory separator
- **THEN** construction fails with `ValueError` and creates no requested file

### Requirement: new_like materializes values in matching storage

`new_like(values, *, mutable=True, dtype=None)` SHALL consume the iterable
`values` and return a fresh carrier of the same concrete carrier kind as the
receiver. `values` names the ordered values to materialize in the fresh storage
and SHALL be iterable. `mutable` states the fresh carrier's intrinsic
mutability, SHALL be optional, and SHALL default to `True`. `dtype` names the
fresh carrier's requested storage dtype, SHALL be optional, and SHALL default
to `None`; `None` SHALL preserve the receiver's dtype, while a supplied
descriptor SHALL request that supported dtype. The result SHALL own or
otherwise establish fresh storage independent of the receiver and SHALL remain
usable even when the receiver has been released.

The result size SHALL equal the number of materialized values. Each non-`None`
value SHALL be stored using the result carrier's normalization. A `None` hole
in CPU or FileBacked input SHALL contain the result dtype's zero. Generic
concrete storage SHALL likewise contain only representable normalized values.

For an Evictable receiver, the result SHALL be a promoted Evictable hierarchy:
its primary SHALL be produced by the primary tier's value factory, its secondary
SHALL be a fresh mutable zero-size allocation of the result dtype, and it SHALL
preserve the primary and secondary carrier kinds.

#### Scenario: Preserve dtype by default

- **WHEN** `new_like` is called without a dtype override
- **THEN** the fresh result has the receiver's dtype and stores the supplied
  values

#### Scenario: Override factory dtype

- **WHEN** `new_like` receives a dtype supported by the carrier kind
- **THEN** the fresh result reports that dtype and normalizes values for it

### Requirement: allocate_like creates fresh size-based storage

`allocate_like(size, *, mutable=True, dtype=None, empty=False)` SHALL return a
fresh carrier of the same concrete carrier kind as the receiver. `size` names
the requested slot count and SHALL support Python's integer-index protocol; a
`size` value that does not support that protocol SHALL fail with `TypeError`,
and a negative `size` SHALL fail with `ValueError`. `mutable` states the fresh
carrier's intrinsic mutability, SHALL
be optional, and SHALL default to `True`. `dtype` names the fresh carrier's
requested storage dtype, SHALL be optional, and SHALL default to `None`; `None`
SHALL preserve the receiver's dtype. `empty` states whether the backend may
skip initialization, SHALL be optional, and SHALL default to `False`.

With `empty=False`, every slot SHALL be initialized to `0.0` for Float32, `0`
for Int32, `False` for Bool, and `None` for legacy opaque Generic storage.
CPU MAY skip initialization when `empty=True`; callers SHALL write every slot
before reading it. Generic and FileBacked SHALL accept `empty=True` while
retaining their initialized behavior.

For an Evictable receiver, the result SHALL be promoted, its primary SHALL have
the requested size, and its fresh mutable secondary SHALL have size zero until
eviction provisions it.

#### Scenario: Allocate initialized storage

- **WHEN** `allocate_like(3)` is called on a concrete storage carrier
- **THEN** it returns independent size-three storage of the receiver's dtype
  with each slot containing that dtype's zero

#### Scenario: Request an empty CPU allocation

- **WHEN** CPU `allocate_like` receives `empty=True`
- **THEN** it returns writable storage of the requested size without promising
  a readable initial value

### Requirement: Public mutability combines storage policy and ownership

`is_mutable()` SHALL return whether public interfaces may currently modify the
carrier. An intrinsically immutable carrier SHALL always return `False`.
A mutable carrier SHALL return `False` while exclusively owned by another
carrier, except within the owner's access scope. `is_owned()` SHALL return
whether such an exclusive owner exists. Ownership semantics are defined fully
by `carrier-composition`.

All public write paths SHALL use this same answer, including indexed writes,
`set_value`, `scatter`, direct release, version increments, and direct moves.

#### Scenario: Query an immutable carrier

- **WHEN** a carrier was constructed with `mutable=False`
- **THEN** `is_mutable()` returns `False` and a public write fails with
  `RuntimeError`

#### Scenario: Query an owned mutable tier

- **WHEN** a mutable tier has been claimed by a composite carrier
- **THEN** its retained alias reports `is_owned() == True` and
  `is_mutable() == False`

### Requirement: Every successful public mutation advances the version

`version` SHALL expose a non-negative monotonic integer. Every successful
public mutation that can change stored values SHALL increment the visible
carrier version, including indexed assignment, `set_value`, and a completed
`scatter` call. An indexed assignment or `set_value` call SHALL increment it
exactly once. A scatter implementation MAY perform multiple constituent writes
and therefore MAY increment it more than once. A failed mutation SHALL leave
both values and version unchanged.

Storage allocation, construction, release, and Evictable residency transitions
SHALL NOT count as value mutations and SHALL NOT increment the visible version.
Mutation through an Evictable wrapper SHALL increment the wrapper version once;
the wrapper SHALL remain the version authority visible to its Tensors.

#### Scenario: Version an indexed write

- **WHEN** a public indexed assignment succeeds
- **THEN** the carrier version is exactly one greater than before the call

#### Scenario: Preserve version across eviction and promotion

- **WHEN** an Evictable carrier evicts or promotes without a logical value
  write
- **THEN** its visible version remains unchanged

### Requirement: Scatter maps logical source values into destination storage

`scatter(to_scatter, scatter_onto, mapping, mapping_offset=0)` SHALL write
source Tensor values into the receiver's storage and return `None`.
`to_scatter` and `scatter_onto` name single-subtensor Tensors; `mapping` names a
`Layout` whose shape equals the source layout shape; `mapping_offset` names an
optional non-negative integer offset and SHALL default to zero.

`scatter_onto` SHALL be backed by the receiver. For each source logical index
`i`, the destination physical index SHALL be
`scatter_onto.offset + mapping_offset + mapping.index(i)`. When `to_scatter` or
`scatter_onto` is not a Tensor, or when `mapping` is not a `Layout`, the call
SHALL fail with `TypeError`. When `scatter_onto` is backed by a different
receiver, when `mapping.shape` differs from the source layout shape, or when
either Tensor has a multi-subtensor representation, the call SHALL fail with
`ValueError`. When `mapping_offset` is invalid or a computed destination index
is outside the receiver, the call SHALL fail before an out-of-range write.
Public mutability and dtype normalization SHALL apply to every write.

Generic and CPU SHALL implement this contract. Evictable SHALL require a
promoted hierarchy, lower to its primary tier, and advance the wrapper version
once. FileBacked SHALL fail with `NotImplementedError` because it does not
support scatter.

#### Scenario: Scatter through a mapping

- **WHEN** valid source and destination Tensors and a matching mapping are
  supplied to a mutable Generic or CPU receiver
- **THEN** each source logical value is stored at the mapped destination index
  and the receiver version advances

### Requirement: Release permanently invalidates storage access

`release()` SHALL permanently release the carrier's storage and return `None`.
It SHALL be idempotent. `is_released()` SHALL return `False` before release and
`True` afterwards. After release, element access, mutation, and operations that
require existing storage SHALL fail with `RuntimeError`. Carrier-specific size
queries MAY report zero after release, but `dtype()`, structural dtype support,
and `new_like` SHALL remain usable.

Direct release of an owned carrier SHALL fail with `RuntimeError`; the owner
MAY release it through its access scope. Releasing a FileBacked carrier SHALL
remove its file. Releasing an Evictable SHALL release both tiers and prevent
subsequent residency transitions.

#### Scenario: Release a carrier twice

- **WHEN** a caller releases an unowned carrier and calls `release()` again
- **THEN** both calls return `None`, the carrier remains released, and no
  storage reappears

#### Scenario: Create fresh storage from a released prototype

- **WHEN** `new_like(values)` is called on a released carrier
- **THEN** it returns fresh usable storage without reading the released values
