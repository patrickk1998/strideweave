## Why

StrideWeave still permits abstract dtype categories to act as physical storage,
creating a second, width-unspecified execution model beside its concrete dtype
policy. This change removes that legacy storage role while preserving the
public category hierarchy used to relate concrete and extension dtypes.

## What Changes

- **BREAKING**: Remove category-backed physical storage and require callers to
  select a concrete dtype when constructing `Generic` or `FileBacked`.
- **BREAKING**: Remove `DTypeCategory`'s `opaque_storage` input and the public
  `is_opaque_storage()` query, including that state's contribution to canonical
  descriptor structures and pickle compatibility.
- Retain `DType.Any`, `DType.Floating`, and `DType.Integer` as immutable abstract
  category identities and hierarchy nodes.
- Define Tensor storage schemas solely from representations: a `SimpleDType`
  uses one identical storage subtensor and a `CompoundDType` uses its ordered
  simple-dtype planes; a category has no Tensor storage schema.
- Make the central concrete dtype policy the only operation-planning path,
  reject every abstract category uniformly, and remove legacy Generic and
  Evictable bypass behavior.
- Restrict differentiation to the exact `DType.Float32` logical dtype.

This proposal is a non-normative scope index for intended behavior. The delta
specs listed below are the sole behavioral authority.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `dtype-descriptors`: Categories become relationship-only descriptors, and
  canonical structure and pickle compatibility no longer include an opaque
  storage disposition.
- `core-tensor-representation`: Simple and compound descriptors provide the
  only Tensor storage schemas; categories are rejected as logical storage
  representations.
- `carrier-storage`: Generic and FileBacked accept only concrete storage dtypes
  and require an explicit keyword-only `dtype` at construction.
- `autograd`: Only exact Float32 logical values participate in reverse mode.
- `operation-dtype-policy`: Every category receives the same unsupported
  planning disposition and Generic has no alternate opaque execution policy.
- `carrier-dispatch`: Planned Generic and Evictable execution has no legacy
  opaque bypass around central planning and capability preflight.

## Impact

- Public constructor signatures, dtype descriptor APIs, carrier support
  introspection, Generic/FileBacked storage, Tensor validation, operation
  planning, dispatch, Evictable lowering, and autograd eligibility change.
- `dtype-representations`, `carrier-composition`, and `interop-movement` remain
  consistent without requirement changes: compound planes stay concrete,
  hierarchy tiers must still share one dtype, and moves still preserve an
  explicitly constructed source and destination dtype.
- `llms.md` sections covering dtype descriptors, carrier storage, Generic
  reference semantics, operations, autograd, and current boundaries require
  reconciliation after implementation.
- Relevant invariant IDs are SW002 and its structure/pickle clauses, RT012,
  RT012a, RT012b, RT013c, RT013r, RT013s, RT013t, and RT014.
