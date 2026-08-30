## ADDED Requirements

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

## MODIFIED Requirements

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
