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
| `Generic` | `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `CPU` | `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `Metal` | `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `FileBacked` | `DType.Float32`, `DType.Int32` |
| `Evictable` | The identity-based intersection reported by its primary and secondary tiers |

Every descriptor outside the applicable row SHALL be unsupported, including
every `DTypeCategory`, `DType.Float64`, narrow structural encodings, and
compound dtypes.

#### Scenario: Query support independently of current storage

- **WHEN** a `Metal` currently holding `Float32` is asked whether it supports
  `DType.Int32`
- **THEN** it returns `True` without changing or allocating storage

#### Scenario: Query a valid unsupported descriptor

- **WHEN** any carrier is asked whether it supports a compound dtype
- **THEN** it returns `False`

#### Scenario: Refuse category storage structurally

- **WHEN** Generic or FileBacked is asked whether it supports `DType.Any`,
  `DType.Floating`, `DType.Integer`, or an extension category
- **THEN** it returns `False` without allocating or mutating storage

### Requirement: Public storage helpers share the identity-based storage rules

`validate_storage_dtype(dtype, *, carrier, accepted)` SHALL validate one
candidate against the exact descriptor identities in `accepted`. `dtype` names
the candidate object, `carrier` names the carrier in diagnostics, and
`accepted` names the ordered tuple of supported DType identities. It SHALL
return the unchanged `dtype` when supported, fail with `TypeError` when `dtype`
is not a `DType`, and fail with `ValueError` when `dtype` is a valid but
unsupported descriptor. When `dtype` is compound, the failure SHALL identify
the deferred multi-plane storage requirement. When `dtype` is a category, the
failure SHALL identify the concrete accepted storage dtypes rather than treating
the category as storage.

`accepts_storage_dtype(dtype, accepted)` SHALL return whether `dtype` is
identical to any descriptor in `accepted`. `dtype` names the candidate object,
and `accepted` names the tuple of accepted descriptor identities. The function
SHALL perform identity rather than equality matching and SHALL return `False`
when `dtype` is not identical to any member of `accepted`.

`storage_zero(dtype)` SHALL return the initialized slot value associated with
the supplied storage dtype. `dtype` names the storage descriptor whose
initialized value is requested. The return value SHALL be `0.0` for
`DType.Float32`, `0` for `DType.Int32`, `False` for `DType.Bool`, and `None`
for a descriptor without a defined concrete stored zero. A `None` result SHALL
not make that descriptor supported carrier storage.

#### Scenario: Validate an exact accepted identity

- **WHEN** `validate_storage_dtype` receives a candidate identical to one entry
  of `accepted`
- **THEN** it returns that same descriptor object

#### Scenario: Reject a category through storage validation

- **WHEN** `validate_storage_dtype` receives a category outside `accepted`
- **THEN** it fails with `ValueError` naming the accepted concrete storage
  dtypes

#### Scenario: Obtain a concrete storage zero

- **WHEN** `storage_zero` receives `DType.Bool`
- **THEN** it returns the Python boolean `False`

## ADDED Requirements

### Requirement: Generic stores normalized concrete values

`Generic(values, *, mutable=True, dtype)` SHALL consume an iterable of stored
values and return a Generic carrier. `values` names the ordered values to place
in storage and SHALL be iterable. `mutable` states the carrier's intrinsic
mutability, SHALL be optional, and SHALL default to `True`. `dtype` names the
homogeneous storage dtype, SHALL be required, SHALL be keyword-only, and SHALL
be exactly `DType.Float32`, `DType.Int32`, or `DType.Bool`. Omitting `dtype`
SHALL fail with `TypeError` identifying the missing required keyword-only
argument; supplying a dtype positionally SHALL fail with `TypeError`
identifying a positional-argument mismatch. A non-`DType` value SHALL fail with
`TypeError`, and any other descriptor identity, including every category,
SHALL fail with `ValueError`. Each failure SHALL expose no carrier storage.

Generic SHALL consume `values` into storage it owns and normalize every value
before the carrier is returned. Float32 values SHALL be binary32-exact Python
floats, Int32 values SHALL be integers in `[-2**31, 2**31 - 1]`, and Bool
values SHALL be Python `bool` values. When `values` is not iterable,
construction SHALL fail with `TypeError`. Invalid stored values SHALL fail with
`TypeError` or `OverflowError` as appropriate.

The same normalization SHALL apply to later writes, including indexed writes
and scatter writes, so a caller-held alias cannot bypass it.

#### Scenario: Construct with an explicit concrete dtype

- **WHEN** a caller constructs `Generic(values, dtype=DType.Float32)`
- **THEN** the result owns normalized Float32 storage independent of `values`

#### Scenario: Require the keyword-only dtype

- **WHEN** a caller invokes `Generic(values)` or supplies a dtype as a second
  positional argument
- **THEN** construction fails with `TypeError` identifying either the omitted
  required keyword-only `dtype` or the positional-argument mismatch and exposes
  no carrier storage

#### Scenario: Own concrete input storage

- **WHEN** a caller constructs a Generic from a mutable input sequence and
  later mutates that sequence
- **THEN** the carrier's stored values do not change

#### Scenario: Reject category-backed Generic storage

- **WHEN** a caller supplies `DType.Any`, `DType.Floating`, `DType.Integer`, or
  an extension category as `dtype`
- **THEN** construction fails with `ValueError` before storage is exposed

## MODIFIED Requirements

### Requirement: FileBacked owns a temporary raw numeric file

`FileBacked(filename=None, *, mutable=True, dtype)` SHALL create a raw numeric
file inside a hidden per-process temporary directory and return an initially
empty carrier. `filename` names a bare file within that directory and SHALL be
optional; `filename` SHALL default to `None`, and `None` SHALL request a
generated unique name. When `filename` contains path components, construction
SHALL fail with `ValueError`; when `filename` duplicates an existing file,
construction SHALL fail without replacing that file. `mutable` states the
carrier's intrinsic mutability, SHALL be optional, and SHALL default to `True`.
`dtype` names the homogeneous storage dtype, SHALL be required, SHALL be
keyword-only, and SHALL be exactly `DType.Float32` or `DType.Int32`.

Omitting `dtype` SHALL fail with `TypeError` identifying the missing required
keyword-only argument; supplying a dtype positionally SHALL fail with
`TypeError` identifying a positional-argument mismatch. A non-`DType` value
SHALL fail with `TypeError`, and any other descriptor identity, including every
category, SHALL fail with `ValueError`. Each failure SHALL occur before creating
the requested file.

`path` SHALL return the carrier's file path. The file SHALL encode Float32 or
Int32 values according to the selected storage dtype. New or extended slots
SHALL read as zero. Int32 writes SHALL require integer values. Deleting or
releasing the carrier SHALL remove its file, and process shutdown SHALL remove
the hidden session directory.

#### Scenario: Generate file-backed storage

- **WHEN** a caller invokes `FileBacked(dtype=DType.Float32)` and omits
  `filename`
- **THEN** the result owns a uniquely named empty Float32 file in the hidden
  session directory

#### Scenario: Require the keyword-only dtype before file creation

- **WHEN** a caller invokes `FileBacked()` or supplies a dtype as a positional
  argument
- **THEN** construction fails with `TypeError` identifying either the omitted
  required keyword-only `dtype` or the positional-argument mismatch and creates
  no file

#### Scenario: Reject a path-like filename

- **WHEN** `filename` contains a directory separator
- **THEN** construction fails with `ValueError` and creates no requested file

#### Scenario: Reject category-backed file storage

- **WHEN** a caller supplies `DType.Any`, `DType.Floating`, `DType.Integer`, or
  an extension category as `dtype`
- **THEN** construction fails with `ValueError` before creating a file

### Requirement: allocate_like creates fresh size-based storage

`allocate_like(size, *, mutable=True, dtype=None, empty=False)` SHALL return a
fresh carrier of the same concrete carrier kind as the receiver. `size` names
the requested slot count and SHALL support Python's integer-index protocol; a
`size` value that does not support that protocol SHALL fail with `TypeError`,
and a negative `size` SHALL fail with `ValueError`. `mutable` states the fresh
carrier's intrinsic mutability, SHALL be optional, and SHALL default to `True`.
`dtype` names the fresh carrier's requested storage dtype, SHALL be optional,
and SHALL default to `None`; `None` SHALL preserve the receiver's dtype.
`empty` states whether the backend may skip initialization, SHALL be optional,
and SHALL default to `False`.

With `empty=False`, every slot SHALL be initialized to `0.0` for Float32, `0`
for Int32, and `False` for Bool. CPU MAY skip initialization when `empty=True`;
callers SHALL write every slot before reading it. Generic and FileBacked SHALL
accept `empty=True` while retaining their initialized behavior. A supplied
`dtype` outside the receiver implementation's concrete accepted set SHALL fail
under the common storage validation contract before fresh storage is exposed.

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

#### Scenario: Reject a category override

- **WHEN** a Generic or FileBacked factory receives a category as its explicit
  `dtype` override
- **THEN** it fails with `ValueError` before exposing fresh storage

## REMOVED Requirements

### Requirement: Generic stores legacy objects and normalized concrete values

**Reason**: Generic no longer accepts category-backed opaque storage, so the
legacy aliasing contract and the requirement name that advertises it no longer
describe any supported behavior.

**Migration**: Supply the required concrete `dtype` and rely on Generic's owned,
normalized storage contract in `Generic stores normalized concrete values`.
