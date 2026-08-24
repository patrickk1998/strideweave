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
| `Metal` | `DType.Float32`, `DType.Int32`, `DType.Bool` |
| `FileBacked` | `DType.Floating`, `DType.Float32`, `DType.Int32` |
| `Evictable` | The identity-based intersection reported by its primary and secondary tiers |

Every descriptor outside the applicable row SHALL be unsupported, including
`DType.Integer`, `DType.Float64`, narrow structural encodings, and compound
dtypes.

#### Scenario: Query support independently of current storage

- **WHEN** a `Metal` currently holding `Float32` is asked whether it supports
  `DType.Int32`
- **THEN** it returns `True` without changing or allocating storage

#### Scenario: Query a valid unsupported descriptor

- **WHEN** any carrier is asked whether it supports a compound dtype
- **THEN** it returns `False`

### Requirement: Scatter maps logical source values into destination storage

`scatter(to_scatter, scatter_onto, mapping, mapping_offset=0)` SHALL write
source Tensor values into receiver storage and return `None`. The two Tensor
arguments SHALL be single-subtensor; `mapping` SHALL be a `Layout` with source
shape; and `mapping_offset` SHALL be a non-negative integer defaulting to zero.
For source logical index `i`, the destination physical index SHALL be
`scatter_onto.offset + mapping_offset + mapping.index(i)`.

Invalid argument types SHALL raise `TypeError`. A foreign destination,
mismatched mapping shape, multi-subtensor argument, invalid offset, or
out-of-range destination SHALL fail before any write. Public mutability and
dtype normalization SHALL apply. A failed call SHALL preserve values and
version. A successful Metal call SHALL complete all writes before returning and
advance the visible Metal version at least once; aliases SHALL observe it.

Generic, CPU, and Metal SHALL implement this carrier mutation. Evictable SHALL
require promotion, lower to its primary tier, and advance the wrapper version
once. FileBacked SHALL raise `NotImplementedError`. This carrier method is
distinct from the computational `scatter` operation, which returns a new Tensor
under operation dispatch and its own dtype and autograd contract.

#### Scenario: Scatter through a mapping

- **WHEN** valid source, destination, and mapping values are supplied to a
  mutable Generic, CPU, or Metal receiver
- **THEN** every source value is stored at its mapped destination and the
  receiver version advances

#### Scenario: Fail Metal scatter atomically

- **WHEN** Metal scatter validation or execution fails
- **THEN** destination values and visible version remain unchanged

## ADDED Requirements

### Requirement: Metal owns typed accelerator storage

`Metal(size, *, mutable=True, dtype=DType.Float32, empty=False)` SHALL return
accelerator storage with `size` physical slots. `size` names the slot count and
SHALL be a non-negative integer; another type SHALL fail with `TypeError`, and a
negative value SHALL fail with `ValueError`. `mutable` states intrinsic
mutability, SHALL be optional, and SHALL default to `True`. `dtype` names the
homogeneous storage dtype, SHALL be optional, and SHALL default to
`DType.Float32`. `empty` states whether initialization may be skipped, SHALL be
optional, and SHALL default to `False`.

Unless `empty=True`, Float32, Int32, and Bool slots SHALL initially contain
`0.0`, `0`, and `False` respectively. Reads and successful writes SHALL obey the
common physical-index, normalization, mutability, version, ownership, and
release contracts. `new_like` and `allocate_like` SHALL return fresh Metal
storage and preserve the receiver's dtype unless an accepted dtype override is
supplied.

Construction SHALL fail with `RuntimeError` before exposing storage when the
required Metal runtime is unavailable. Missing optional accelerator
dependencies SHALL not prevent CPU-only import or execution, and attempting to
construct Metal without them SHALL fail with an actionable `RuntimeError`.

#### Scenario: Allocate initialized Metal storage

- **WHEN** a caller constructs `Metal(3, dtype=DType.Float32)` on an available
  Metal runtime
- **THEN** it returns live mutable size-three Float32 storage whose slots read
  as `0.0`

#### Scenario: Preserve CPU-only use without accelerator dependencies

- **WHEN** the optional Metal dependency set is absent and a caller uses only
  CPU functionality
- **THEN** importing and executing CPU functionality succeeds without loading
  the Metal runtime

#### Scenario: Refuse unavailable Metal construction

- **WHEN** a caller constructs Metal without an available supported Metal
  runtime
- **THEN** construction fails with `RuntimeError` before exposing partial
  storage
