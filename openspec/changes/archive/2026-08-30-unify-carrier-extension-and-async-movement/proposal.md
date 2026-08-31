## Why

Adding BlockDevice and Metal exposed repeated carrier-specific seams, while
TiledEvictable needs eager partial movement, dependent execution, and
carrier-specific residency control. The first delivery should test whether one
generic extension boundary can express those requirements without also
migrating every shipped carrier in the same change.

This proposal introduces the generic extension surface and uses
TiledEvictable as its first shipped consumer. Existing Generic, CPU, Metal,
FileBacked, BlockDeviceCarrier, and Evictable implementations retain their
current storage, dispatch, capability, operation, and movement behavior.

## What Changes

- Add immutable exact-class carrier definitions separating storage, kernel
  execution, transfers, dependent composition, and typed carrier-specific
  facets.
- Add public namespaced operation definitions and exact-independent-carrier
  kernel packs for definition-backed extension carriers. Existing shipped
  compute carriers remain definition-free and do not accept packs.
- Define `KernelExecutionInterface` as the carrier/pack compatibility boundary;
  concrete compiled launch ABIs remain provider details.
- Add eager asynchronous single-Tensor movement returning a blocking completion
  handle with `done` and `wait()`. Python coroutine integration, cancellation,
  and transactional multi-move batching are outside this delivery.
- Add TiledEvictable as the only definition-backed shipped carrier. It composes
  definition-free compute and storage carriers through their existing public
  capabilities, dispatch, factories, and movement behavior without concrete
  dependency-class branches.
- Give raw TiledEvictable operands a bounded central operation surface:
  `add`, `elementwise_mul`, `mul` (tensor/tensor, tensor/weak-scalar, and
  weak-scalar/tensor), `relu`, `reduce_sum`, and `matmul`. Explicit projection
  returns an ordinary compute-carrier Tensor that retains that carrier's
  complete existing operation surface.
- Add eager projection, asynchronous residency transitions, functional
  scatter, and full-shaped tiled gradients. This delivery uses
  correctness-first behavior without tiled kernels, graph-wide residency
  scheduling, or cache optimization.
- Preserve all existing definition-free carrier, capability, operation, and
  movement extension behavior. Migrating shipped carriers and removing the
  compatibility boundary belongs to a later change.

## Capabilities

### New Capabilities

- `carrier-extension`: Exact carrier definitions, public kernel packs,
  carrier execution interfaces, typed facets, provider responsibilities, and
  definition-free compatibility.
- `operation-semantics`: Public carrier-neutral custom operation definitions
  and the bounded central operation surface used by direct tiled execution.
- `tiled-carrier-composition`: Tiled selection, eager projection, residency,
  direct full-Tensor dispatch, gather/scatter, lifecycle, and full logical
  gradients.

### Modified Capabilities

- `backend-capabilities`: Definition-backed capabilities derive from kernel
  patterns or dependent composition while existing shipped declarations remain
  unchanged.
- `carrier-dispatch`: Exact definition-backed dispatch is added alongside the
  existing definition-free carrier path, and TiledEvictable uses the dependent
  path for its bounded direct operation surface.
- `interop-movement`: Existing blocking exact-carrier movement gains eager
  single-request completion handles used by tiled residency and projection.

## Impact

The change affects the public carrier-extension and custom-operation APIs,
TiledEvictable construction and control surfaces, single-Tensor movement,
operation dispatch at the definition-backed boundary, autograd construction,
profiling, public exports, typing, and documentation. It does not change the
implementation authority or public behavior of Generic, CPU, Metal,
FileBacked, BlockDeviceCarrier, or Evictable.

The affected `llms.md` headings are **Specifications And This Document**,
**Core Model**, **Carriers** (including **Backend Capabilities**), **Operations**,
**Operation Profiling**, **Autograd**, **Interoperability And Movement**, and
**Current Boundaries**.

The affected invariant entries are `RT001`, `RT001a`, `RT001b`, `RT002`,
`RT003`, `RT004`, `RT005`, `RT006`, `RT007`, `RT008`, `RT009`, `RT011`,
`RT011a`, `RT011b`, `RT012`, `RT012a`, `RT012b`, `RT012c`, `RT013`,
`RT013a`, `RT013b`, `RT013c`, `RT013d`, `RT013e`, `RT013f`, `RT013g`,
`RT013h`, `RT013i`, `RT013j`, `RT013l`, `RT013m`, `RT013n`, `RT013o`,
`RT013p`, `RT014d`, `RT015`, `RT016`, `RT016b`, `RT017`, `RT017a`, and
`RT022`.
