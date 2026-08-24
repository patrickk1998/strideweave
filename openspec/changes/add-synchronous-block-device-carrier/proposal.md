## Why

Models can exceed available DRAM while local SSD capacity remains available for
inactive weights. This change scopes an ephemeral, storage-only tensor memory tier
over one user-supplied logical block device shared by multiple tensors.

This proposal is a non-normative scope index. The effective delta specs are the sole
authority for whether observable behavior is a feature or a bug.

## What Changes

- Add public `BlockDevice` and `BlockDeviceCarrier` types for process-local,
  Float32-only allocation and synchronous access on a supported device node.
- Add exact synchronous CPU-to-block and block-to-CPU movement while refusing other
  block movement pairs, computation, DLPack, and block-backed Evictable tiers.
- Define shared-device allocation, lifecycle, failure, callback/reentrancy, and
  fixed-size destination behavior within the existing carrier contracts.
- Cover Linux block nodes and buffered macOS block nodes; defer persistence,
  discovery, striping, multi-process coordination, asynchronous I/O, direct I/O,
  raw macOS nodes, device-side computation, and performance guarantees.

## Capabilities

### New Capabilities

None. The new types extend existing carrier storage, dispatch, capability, movement,
and composition domains.

### Modified Capabilities

- `carrier-storage`: Block-device construction, shared extent allocation, Float32
  access, ordering, failure, factories, release, and device lifecycle.
- `carrier-dispatch`: Closed storage-only dispatch behavior for
  `BlockDeviceCarrier`.
- `backend-capabilities`: Complete sealed empty capability declaration for
  `BlockDeviceCarrier`.
- `interop-movement`: DLPack refusal, exact CPU/block movement, unsupported-pair
  refusal, fixed-size destinations, and partial-failure state.
- `carrier-composition`: Refusal of `BlockDeviceCarrier` in either Evictable tier.

## Impact

- Public API: `BlockDevice`, `BlockDeviceCarrier`, and two exact move-operation
  classes become public exports without changing existing signatures.
- Runtime and platform surface: Linux and macOS block-device access, shared extent
  management, and exact CPU/block movement.
- Documentation and invariants: later implementation updates `llms.md`, the affected
  invariant entries, and the published `carrier-storage` Purpose/front matter.
- Validation: deterministic semantic coverage plus disposable Linux real-device
  integration and local buffered-macOS compatibility evidence.
