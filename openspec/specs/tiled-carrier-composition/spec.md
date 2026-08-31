---
title: Tiled Carrier Composition
publish: true
status: stable
order: 82
summary: Tiled storage geometry, independent residency, projection, scatter, direct execution, lifecycle, and autograd.
---

# tiled-carrier-composition Specification

## Purpose

Define TiledEvictable tile geometry, independent residency, projection,
scatter, bounded direct execution, lifecycle, and reverse-mode behavior.

## Requirements

### Requirement: TiledEvictable owns one flat logical storage extent

A live TiledEvictable SHALL expose one dtype identity, one flat physical-slot
extent, one canonical tile grid and tile geometry, and primary and secondary
tier kinds capable of storing every tile. The tile mapping SHALL map each
logical Tensor coordinate that addresses the carrier to exactly one tile
coordinate and one within-tile coordinate.

The Tensor's shape and Layout SHALL describe the unchanged full logical Tensor,
and the Tensor SHALL be tile-footprint compatible with the carrier. Residency
transitions SHALL not alter that shape, Layout, dtype, logical storage extent,
Tensor identity, or visible value version. Tile selection SHALL not be
represented by changing the Tensor's placement Layout.

`TiledEvictable` SHALL be importable from both the top-level package and
`strideweave.carriers.tiled_evictable`.

TiledEvictable SHALL be the only shipped carrier whose execution and extension
surface is installed through a `CarrierDefinition` in this delivery. Its
definition SHALL compose its configured definition-free compute and backing
carriers through their existing public allocation, capability, dispatch, and
movement authorities. Composition SHALL NOT install definitions on, or change
the behavior of, Generic, CPU, Metal, FileBacked, BlockDeviceCarrier, or
Evictable.

#### Scenario: Promote one expert tile

- **WHEN** a weight Tensor shaped `(expert, output, input)` uses one whole
  expert per tile and one expert is promoted
- **THEN** the Tensor retains its full shape, Layout, identity, and carrier
  while only that tile's residency changes

### Requirement: Tensor Layout compatibility preserves exact carrier tile footprints

For an in-range tile coordinate `t`, the **carrier tile footprint** SHALL mean
the finite set of physical slots that the TiledEvictable's canonical tile
geometry assigns to carrier tile `t`. For a full Tensor backed by that carrier,
the **Tensor tile footprint** at `t` SHALL mean the set of physical slots
reached by adding the Tensor's physical offset to the result of applying its
Layout to every logical Tensor coordinate whose carrier tile coordinate is
`t`.

The Tensor SHALL be **tile-footprint compatible** with the carrier exactly when,
for every in-range tile coordinate `t`, the logical coordinates in Tensor tile
`t` map injectively and the Tensor tile footprint at `t` equals the carrier tile
footprint at that same `t`. The Tensor's physical offset and Layout SHALL be the
sole authority for associating within-tile logical coordinates with slots in
that footprint. Every within-tile ordering whose image is exactly the carrier
tile footprint SHALL therefore be compatible, including different injective
within-tile orderings over the same physical slots.

Every public TiledEvictable entry point that accepts a full Tensor SHALL enforce
this one compatibility rule before residency, transfer, provider, value,
version, ownership, or autograd state changes. A Tensor whose logical tile
reaches slots owned by multiple carrier tiles, reaches another tile's
footprint, repeats a physical slot, or omits a slot in its corresponding carrier
tile footprint SHALL fail with `ValueError`.

#### Scenario: Accept a within-tile permutation

- **WHEN** every logical Tensor tile reaches exactly its corresponding carrier
  tile footprint and its Layout permutes the association between within-tile
  coordinates and those physical slots
- **THEN** the Tensor is tile-footprint compatible and projection, scatter,
  direct execution, and backward preserve the logical values defined by that
  Layout

#### Scenario: Reject one logical tile spanning physical tiles

- **WHEN** the physical slots reached by one logical Tensor tile belong to more
  than one carrier tile footprint
- **THEN** the accepting TiledEvictable entry point fails with `ValueError`
  before residency, transfer, provider, value, version, ownership, or autograd
  state changes

#### Scenario: Reject aliasing or incomplete tile coverage

- **WHEN** coordinates within one logical Tensor tile repeat a physical slot or
  fail to reach every slot in the corresponding carrier tile footprint
- **THEN** the accepting TiledEvictable entry point fails with `ValueError`
  before residency, transfer, provider, value, version, ownership, or autograd
  state changes

#### Scenario: Reject tile-coordinate remapping

- **WHEN** a logical Tensor tile reaches exactly the footprint assigned to a
  different carrier tile coordinate
- **THEN** the accepting TiledEvictable entry point fails with `ValueError`
  before residency, transfer, provider, value, version, ownership, or autograd
  state changes

### Requirement: TileSelection is an ordered Cartesian product

`TileSelection(per_axis)` SHALL immutably describe the tiles gathered into one
compact Tensor. `per_axis` SHALL be a finite iterable containing one non-empty
finite iterable of integer tile coordinates for each selected grid axis. The
constructor SHALL preserve the caller's coordinate order on every axis and
SHALL expose it as `per_axis: tuple[tuple[int, ...], ...]`.

The selection's source tile coordinates SHALL be the complete Cartesian
product. Its public `coordinates: tuple[tuple[int, ...], ...]` SHALL enumerate
those source coordinates in compact StrideWeave first-mode-fastest order: the
first grid axis varies fastest, and each axis uses its order in `per_axis`.
This is the same ordinal authority as `Shape.encode` and `Shape.decode`. The
selection SHALL expose `rank: int` as `len(per_axis)` and
`compact_grid_shape: tuple[int, ...]`, whose entries are the selected counts
per axis. For compact coordinate `c`, the corresponding source tile coordinate
SHALL be
`tuple(per_axis[i][c[i]] for i in range(rank))`.

Construction SHALL fail with `TypeError` when `per_axis` or an axis is not a
finite iterable, or when a coordinate is not an integer. It SHALL fail with
`ValueError` when there are no axes, an axis is empty, an axis contains a
duplicate, or a coordinate is negative. Binding the selection to a tiled grid
SHALL fail with `ValueError` for rank mismatch or an out-of-range coordinate.
`TileSelection` SHALL expose no sparse-coordinate constructor and SHALL expose
no `Product`, `Permutation`, or other IndexMap as part of its public contract.

`TileSelection` SHALL be importable from both the top-level package and
`strideweave.carriers.tiled_evictable`.

#### Scenario: Select three experts in router order

- **WHEN** the expert-axis selection is `[2, 7, 19]` and every remaining tile
  axis is fully selected
- **THEN** the compact expert axis contains tiles 2, 7, and 19 in that order

#### Scenario: Include every recombined corner

- **WHEN** a two-axis selection uses coordinates `[0, 1]` on both axes
- **THEN** its `coordinates` contains all four coordinate pairs including
  `(1, 1)`

#### Scenario: Enumerate a multi-axis selection first-mode-fastest

- **WHEN** a two-axis selection uses `[7, 2]` on axis zero and `[4, 9]` on
  axis one
- **THEN** `coordinates` is `((7, 4), (2, 4), (7, 9), (2, 9))`

### Requirement: Project eagerly returns a completion handle for a compact Tensor

`carrier.project(tensor, selection, destination=None)` SHALL validate and
eagerly initiate projection, then return `AwaitProjection`, an
`AwaitResult[Tensor]`. `carrier` SHALL be the exact TiledEvictable backing the
full logical `tensor`; `selection` SHALL be an immutable `TileSelection` bound
to that carrier's tile grid. A non-Tensor or a Tensor backed by another carrier
SHALL fail with `TypeError`; a non-`TileSelection` SHALL fail with `TypeError`;
and a rank-, range-, or geometry-incompatible selection SHALL fail with
`ValueError`. A released `carrier` SHALL fail with `RuntimeError`. These
failures SHALL occur before residency or transfer work.

`destination` SHALL default to a fresh ordinary carrier allocated from the
tiled definition's configured compute carrier. If supplied, it SHALL be a live,
publicly mutable ordinary Carrier of the configured compute carrier's exact
class, with matching dtype and sufficient or allocatable compact physical
extent. A TiledEvictable, another dependent carrier, a non-Carrier, or a
different exact carrier class SHALL fail with `TypeError`; a released or
publicly immutable carrier SHALL fail with `RuntimeError`; an incompatible
dtype SHALL fail with `TypeError`; and insufficient non-allocatable extent
SHALL fail with `ValueError`. These failures SHALL occur before work starts.

After validation, `project` SHALL start transfers for missing selected tiles
and every gather step whose inputs are immediately available before returning.
It SHALL keep the selected source values and destination valid until terminal
completion and SHALL not wait for transfer or device completion merely to
return the handle. `AwaitProjection.wait()` SHALL return the same ordinary
compact destination Tensor after terminal success. The result
SHALL have dense Layout, preserve the `TileSelection.coordinates` order and
dtype, and contain exactly the selected logical values.

Projection SHALL not change the source Tensor's shape, Layout, carrier,
identity, or logical values, though selected tiles MAY be primary-resident
after successful projection. Projection SHALL create no mutable carrier-wide
selection state. Mutation of the original Python inputs after return SHALL not
change the selection, destination, source version, or result observed by the
pending projection.

`AwaitProjection` and the TiledEvictable `project` method SHALL be available
through `strideweave.carriers.tiled_evictable`; `AwaitProjection` SHALL also be
importable from the top-level package.

#### Scenario: Overlap projection with computation

- **WHEN** a valid selection contains evicted tiles whose transfer remains
  pending after initiation
- **THEN** `project` begins promotion and gather, returns an incomplete
  `AwaitProjection`, and lets the caller perform unrelated computation before
  calling `wait()`

#### Scenario: Observe a completed projection repeatedly

- **WHEN** projection completes successfully
- **THEN** repeated `wait()` observations return the same
  compact Tensor object

#### Scenario: Keep the full Tensor unchanged

- **WHEN** projection of three experts completes
- **THEN** the source still denotes the full expert Tensor while only its
  residency metadata may differ

### Requirement: Projection is an autograd-visible gather

When graph construction is enabled and the full source requires gradients,
successful projection SHALL attach exactly one visible gather node to the
published compact Tensor. `AwaitProjection` itself SHALL not be an autograd
node and SHALL not be accepted as a Tensor operand. Backward SHALL use the
source identity, immutable selection, Layout, value version, and tile geometry
captured by successful projection and SHALL apply the existing saved-version
validation contract before producing a gradient.

Backward SHALL scatter-add the compact cotangent into a full-shaped gradient
Tensor backed by a fresh, distinct TiledEvictable with the primal's dtype,
Layout, geometry, and tier kinds. Selected contributions SHALL accumulate;
untouched gradient tiles SHALL be valid implicit zero. Multiple graph paths
with overlapping selections SHALL sum their contributions on overlap.

#### Scenario: Backpropagate through project then operation

- **WHEN** an operation consumes the Tensor returned by
  `carrier.project(source, selection).wait()`
- **THEN** backward runs the operation's compact VJP and then the projection
  node scatter-adds its compact input gradient into `source.grad`

#### Scenario: Accumulate overlapping projections

- **WHEN** two projection nodes select overlapping tiles and contribute
  gradients
- **THEN** the full TiledEvictable gradient contains their sum on overlaps and
  implicit zero on untouched tiles

### Requirement: Residency state is independent of executable selection shape

Each tile SHALL have observable residency metadata identifying its current
location, validity, and dirty state. The resident tile set MAY be any subset of
the grid and need not form a `TileSelection`. Selection tensor-shapedness SHALL
constrain only one projection or explicit scatter request.

An invalid tile SHALL not be readable until initialized by promotion from a
valid tier, explicit scatter, or its declared implicit value. A dirty tile
SHALL identify a primary-resident value that must be preserved before primary
storage is discarded.

#### Scenario: Cache unrelated tiles

- **WHEN** policy retains three primary tiles that do not form a Cartesian
  product
- **THEN** residency remains valid even though those tiles cannot be consumed
  together as one `TileSelection`

### Requirement: Residency remains valid while tiled work needs it

From initiation until projection, scatter, direct full execution, or tiled
backward no longer needs a tile, that tile's required value SHALL remain valid
and SHALL not be evicted, replaced, or made inaccessible to that work. Missing
values SHALL be obtained through the common asynchronous movement completion
contract before they are consumed.

An overlapping promotion or eviction SHALL not commit a conflicting residency
change while earlier tiled work still needs the value. Failure before provider
submission SHALL leave residency, validity, dirtiness, values, and visible
version unchanged. Failure after submission SHALL publish no partial
destination or result, and every affected Tensor and Carrier SHALL again be
usable when the failure becomes terminal.

#### Scenario: Eviction races with execution

- **WHEN** an eviction request overlaps tiles still needed by an executing
  projection
- **THEN** eviction cannot commit those tiles while projection still needs
  their values

### Requirement: A direct tiled operand means the full logical Tensor

A Tensor backed by TiledEvictable SHALL be accepted directly only by the
central definition-backed operations `add`, `elementwise_mul`, `mul`, `relu`,
`reduce_sum`, and `matmul`; `mul` SHALL include its tensor/tensor,
tensor/weak-scalar, and weak-scalar/tensor overloads. Passing the Tensor
directly SHALL mean its complete logical shape and every tile, never only its
current resident set. Any other operation name on a raw TiledEvictable Tensor
SHALL remain unsupported in this delivery.

Before computing, the operation SHALL make every required logical value
available and MAY block because ordinary operation dispatch is synchronous.
Its allocating result SHALL be the ordinary dense compute-carrier Tensor
declared by central semantics; it SHALL not be wrapped back into
TiledEvictable. The direct operation SHALL expose exactly one operation and
autograd boundary; provider computation and residency movement used to produce
that result SHALL create no additional visible autograd nodes.

An `AwaitProjection`, `AwaitMove`, or `AwaitResidency` SHALL not be a Tensor and
SHALL fail ordinary operand validation with `TypeError`; callers SHALL first
call `wait()`. A Tensor returned from explicit projection SHALL already be an
ordinary compact compute Tensor and SHALL retain the compute carrier's complete
legacy operation surface, so a subsequent operation SHALL create its normal
compute-carrier node after the separate projection gather node.

#### Scenario: Execute on a partially resident full Tensor

- **WHEN** an operation receives a full TiledEvictable Tensor while only some
  tiles are primary-resident
- **THEN** the operation consumes every logical tile and returns the same dense
  result as if all tiles had already been resident

#### Scenario: Reject a pending projection operand

- **WHEN** an operation receives `carrier.project(tensor, selection)` without
  waiting
- **THEN** it fails Tensor operand validation rather than silently blocking

#### Scenario: Execute on an explicitly projected Tensor

- **WHEN** the caller waits for three selected experts and passes that compact
  Tensor to an operation
- **THEN** the operation consumes only the three compact experts and returns a
  compact dense result

### Requirement: Gather preserves public TileSelection ordering

Projection gather SHALL map each compact tile coordinate `c` to
`tuple(selection.per_axis[i][c[i]] for i in range(selection.rank))` and SHALL
pack the resulting source coordinates in `selection.coordinates` order. The
Tensor's Layout SHALL govern affine placement within each source tile. Gather
SHALL preserve source dtype and logical values and SHALL validate the
selection against the full tile grid before work begins.

No concrete IndexMap composition SHALL be part of the `TileSelection`, project,
or gather public contract. Implementations MAY use accepted IndexMap types
internally provided that the public mapping and first-mode-fastest ordering are
unchanged.

#### Scenario: Gather a permuted selection

- **WHEN** a selection orders tiles `[7, 2, 19]`
- **THEN** gather packs them in that compact order without expressing the
  permutation as a global affine Layout

#### Scenario: Gather a multi-axis selection

- **WHEN** `selection.coordinates` is
  `((7, 4), (2, 4), (7, 9), (2, 9))`
- **THEN** successive compact tiles contain those source tiles in exactly that
  first-mode-fastest order

### Requirement: Scatter is the asynchronous projection counterpart

`carrier.scatter(compact, template, selection, reduction="replace")` SHALL
validate and eagerly initiate functional scatter and return
`AwaitMove[Tensor]`. `compact` SHALL be an ordinary Tensor backed by the exact
configured compute carrier class;
`template` SHALL be a full Tensor backed by `carrier`; `selection` SHALL be a
compatible ordered `TileSelection`. `reduction` SHALL accept `"replace"` and
`"add"` and default to `"replace"`. Other values SHALL fail with `ValueError`.

The compact shape and dtype SHALL exactly match the selected region. A
non-Tensor `compact` or `template`, a compact Tensor backed by a dependent
carrier or a different exact carrier class, a template backed by another
carrier, or a non-`TileSelection` SHALL fail with `TypeError`. A released input
or carrier SHALL fail with
`RuntimeError`; a compact dtype mismatch SHALL fail with `TypeError`; and a
compact shape, selection rank, selection range, or tile-geometry mismatch SHALL
fail with `ValueError`. Validation SHALL complete before transfer submission or
any logical value, residency, version, ownership, or release-state change.

Work SHALL start eagerly after validation. `wait()` SHALL return a fresh
full-shaped Tensor backed by a new TiledEvictable with the template's shape,
Layout, dtype, geometry, and tier kinds. Scatter SHALL not release or mutate
`compact` or `template`. `scatter` SHALL be available on TiledEvictable through
`strideweave.carriers.tiled_evictable`.

Replace SHALL publish compact values at selected tiles. Add SHALL combine them
with selected existing or implicit-zero values using central scatter-add
arithmetic. The selected set SHALL commit atomically, become valid and dirty,
and advance the fresh destination's value version exactly once. Unselected
tiles SHALL be invalid for an ordinary partial result or valid implicit zero
when the result is a gradient. Failure SHALL publish no Tensor and clean up all
request state so both inputs are usable again at terminal failure.

For a `TileSelection`, projecting the result with the same selection
SHALL reproduce `compact` after replace scatter. Scatter SHALL not be specified
as a two-sided inverse because the full result also contains unselected state.
Scatter-add SHALL be the linear adjoint used by projection backward.

#### Scenario: Scatter a compact result

- **WHEN** a compact Tensor matches a selected three-expert region
- **THEN** replace scatter returns a completion handle whose completed full
  result has those three valid dirty tiles

#### Scenario: Project after replace scatter

- **WHEN** replace scatter completes and its result is projected with the same
  selection
- **THEN** the completed compact values equal the original `compact` Tensor

#### Scenario: Reject a compact shape mismatch

- **WHEN** compact shape differs from the selection's compact shape
- **THEN** scatter fails before transfer submission or any input or residency
  state change

### Requirement: Scatter participates in autograd

Successful differentiable scatter SHALL attach one visible scatter node to the
completed full result; its `AwaitMove` SHALL not be a graph node. Backward for
the compact input SHALL project the incoming full cotangent through the saved
selection. Replace SHALL propagate no gradient from overwritten selected
template values and SHALL preserve template gradients only where the result
semantically retained template values. Add SHALL propagate the selected
cotangent to both contributing values according to central addition semantics.

#### Scenario: Differentiate replace scatter

- **WHEN** a scalar loss consumes tiles written by differentiable `compact`
- **THEN** scatter backward projects the full cotangent at the saved selection
  into `compact.grad`

### Requirement: Residency requests use arbitrary immutable tile sets

`TileSet(coordinates=())` SHALL capture an immutable finite set of full-rank
tile coordinates. It SHALL expose
`coordinates: tuple[tuple[int, ...], ...]`, ordered with the first coordinate
axis varying fastest by sorting on each coordinate's reversed tuple.
Construction SHALL fail with `TypeError` when
`coordinates` is not a finite iterable, an entry is not a finite iterable, or
a component is not an integer. It SHALL fail with `ValueError` for mixed ranks,
a negative component, or a duplicate coordinate. An empty rank-agnostic
TileSet SHALL be valid. Binding a non-empty TileSet to a tiled grid SHALL fail
with `ValueError` for rank mismatch or an out-of-range coordinate.

`TileSelection.tile_set` SHALL expose a TileSet containing its unique public
`coordinates`; TileSet SHALL not require a Cartesian product.

`ResidencyPlan(promote=None, retain=None, evict=None)` SHALL capture one
immutable control-plane decision and expose those three fields as TileSets.
Each omitted or `None` argument SHALL become an empty TileSet; a non-TileSet
argument SHALL fail with `TypeError`. A tile occurring in more than one field
SHALL fail with `ValueError` before residency work.

`TileSet` and `ResidencyPlan` SHALL be importable from both the top-level
package and `strideweave.carriers.tiled_evictable`.

#### Scenario: Represent a non-product cache set

- **WHEN** a TileSet contains `(0, 0)`, `(0, 1)`, and `(1, 0)` without `(1, 1)`
- **THEN** it is valid even though it is not a compact `TileSelection` domain

### Requirement: Residency transitions have asynchronous and blocking forms

The tiled residency facet SHALL expose
`promote_async(tiles) -> AwaitResidency` and
`evict_async(tiles) -> AwaitResidency`. `tiles` SHALL be an immutable TileSet;
a TileSelection SHALL be accepted as its `tile_set`. Any other input SHALL fail
with `TypeError`; a rank- or range-incompatible set SHALL fail with
`ValueError`; and a released TiledEvictable SHALL fail with `RuntimeError`.
Each method SHALL validate the complete set, start all immediately available
tile transfers, and return without waiting for terminal completion. Validation
failure SHALL change no value, residency, validity, dirtiness, ownership, or
visible version.

Blocking `promote(tiles)` and `evict(tiles)` SHALL behave exactly as their
asynchronous counterpart followed by `.wait()` and SHALL return `None`.
Promotion SHALL make every requested tile valid on primary. Eviction SHALL
preserve every dirty value on secondary before relinquishing primary
residency. Repeating an already-satisfied request SHALL be idempotent. A failed
residency request SHALL retain the complete set's prior residency, values,
validity, dirtiness, ownership, and visible version.

Promotion and eviction SHALL be residency transitions, not differentiable data
operations: they SHALL create no autograd node and SHALL not change the Tensor
or carrier identity. Projection and scatter SHALL remain the separate
differentiable data APIs.

An operation, projection, or scatter whose required tiles overlap an
independently initiated incomplete residency request SHALL fail with
`RuntimeError` directing the caller to wait for that request. It SHALL not
silently join or synchronously block on another public handle. Residency work
initiated as part of the same projection, scatter, operation, or backward is
part of that request and SHALL not be treated as an independent conflict.

`AwaitResidency` and these four residency methods SHALL be available through
`strideweave.carriers.tiled_evictable`; `AwaitResidency` SHALL also be
importable from the top-level package.

#### Scenario: Prefetch without changing autograd

- **WHEN** `promote_async(selection)` completes before projection
- **THEN** residency changes but the full Tensor identity and autograd graph do
  not

#### Scenario: Evict a selected dirty tile set

- **WHEN** selected primary tiles are dirty and `evict(tiles)` succeeds
- **THEN** all values commit to secondary together and the method returns
  `None`

#### Scenario: Refuse hidden waiting on external transition

- **WHEN** an operation needs a tile covered by another caller's incomplete
  `evict_async`
- **THEN** it raises `RuntimeError` rather than waiting for or joining that
  handle

### Requirement: Residency policy can optimize but cannot define correctness

`ResidencyPolicy` SHALL be a runtime-checkable structural protocol whose one
required method is
`plan(required: TileSet, purpose: str) -> ResidencyPlan`. `required` SHALL be
the immutable TileSet of values the affected request must be able to read or
write; `purpose` SHALL be exactly one of `"operation"`, `"projection"`,
`"scatter"`, or `"backward"`.
The TiledEvictable construction API SHALL include the keyword
`residency_policy`, accepting one `ResidencyPolicy` instance or `None`,
defaulting to `None`, and the selected value SHALL be immutable after
construction. A non-`None` value that does not satisfy the protocol SHALL fail
construction with `TypeError` before an instance is exposed.

The callback SHALL return a `ResidencyPlan`. Another return type SHALL fail
with `TypeError`; a plan incompatible with the carrier grid SHALL fail with
`ValueError`. An exception raised by the callback SHALL propagate before any
provider work for the affected request. For a valid plan, every `promote` tile
SHALL be primary-valid before that request consumes it; every `retain` tile
that is primary-valid SHALL remain so until the request is terminal; and every
`evict` tile SHALL be evicted after the request no longer needs it. Required
values SHALL be made available regardless of whether the policy included them
in `promote`, and an eviction SHALL be deferred until it cannot invalidate live
work or a value saved for backward.

Router outputs and token-to-expert assignments SHALL remain Tensor data. The
policy receives only the derived immutable requirement and SHALL not redefine
selection, operation results, autograd reachability, or gradients. Without a
policy, every forward and backward SHALL remain correct by making its required
values available on demand. The policy SHALL be invoked only for an operation
or backward request that actually executes; an unreachable graph branch that
receives no gradient SHALL require no policy call or promotion.

`ResidencyPolicy` SHALL be importable from both the top-level package and
`strideweave.carriers.tiled_evictable`.

#### Scenario: Run backward without a policy

- **WHEN** saved tiles are evicted and no scheduler is configured
- **THEN** backward makes them available and computes the same gradient

#### Scenario: Skip an unreachable branch

- **WHEN** graph discovery sees a node whose output receives no gradient
- **THEN** policy need not promote that node's tiles

#### Scenario: Reject an invalid policy result before execution

- **WHEN** `ResidencyPolicy.plan` returns an object other than
  `ResidencyPlan`
- **THEN** the affected request fails with `TypeError` before provider work or
  any residency change

### Requirement: Tiled release ends all owned residency

`TiledEvictable.release()` SHALL release all tile storage and metadata, mark the
carrier released, and return `None`. It SHALL be idempotent. If any projection,
scatter, residency request, move, or operation involving the carrier is not yet
terminal, `release()` SHALL fail with `RuntimeError` directing the caller to
wait for that work; it SHALL neither cancel the work nor release any tile.
After release, projection, scatter, promotion, eviction, value access, and
dispatch SHALL fail with `RuntimeError`.

#### Scenario: Release after completed tiled work

- **WHEN** a TiledEvictable with tiles in both tiers has no incomplete work and
  is released
- **THEN** both tiers' resources become unavailable and no later residency or
  execution request succeeds
