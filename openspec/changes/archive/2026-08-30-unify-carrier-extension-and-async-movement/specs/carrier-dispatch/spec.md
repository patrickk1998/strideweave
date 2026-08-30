## ADDED Requirements

### Requirement: Definition-backed dispatch preserves central operation semantics

When an exact carrier class has a `CarrierDefinition`, dispatch of a registered
operation name SHALL return a fresh `Operation` bound to that exact carrier and
the name's immutable central `OperationDefinition`. Calling the Operation SHALL
complete central input binding and semantic resolution before asking a kernel
or composite provider to prepare the resulting `ResolvedInvocation`.

A provider SHALL receive only the resolved invocation and the public
carrier-extension context its callback specifies. It SHALL prepare execution
without constructing another OperationDefinition or changing operand, option,
dtype, shape, effect, result, saved-state, or VJP semantics.

#### Scenario: Dispatch a custom definition-backed operation

- **WHEN** a definition-backed custom carrier dispatches a registered custom
  operation
- **THEN** it returns a fresh Operation using the central definition and the
  exact carrier provider prepares only matching execution

#### Scenario: Dispatch a storage-only carrier

- **WHEN** a definition-backed storage carrier dispatches a computational name
  without a kernel or composite provider
- **THEN** dispatch fails with `NotImplementedError` identifying that name

## MODIFIED Requirements

### Requirement: Dispatch is an instance operation

`carrier.dispatch_op(operation_name)` SHALL require a `Carrier` instance and a
string dispatch name. `carrier` names the instance whose exact class owns the
dispatch definition. `operation_name` names the canonical operation requested
and SHALL be a string. Calling `dispatch_op` on `Carrier` or another carrier
class without a carrier instance SHALL fail with `TypeError`; passing a
non-string name SHALL fail with `TypeError`.

For an exact class with a `CarrierDefinition`, the call SHALL resolve a fresh
Operation bound to the registered semantic definition and that exact carrier.
An exact legacy class with no definition SHALL ask that instance's dispatch
hook for the implementation. A
storage-only definition and the default legacy hook SHALL fail with
`NotImplementedError` identifying the unsupported name.

#### Scenario: Dispatch from an instance

- **WHEN** a carrier instance's definition supports a dispatch name
- **THEN** `dispatch_op` returns a fresh Operation bound to the central
  definition and that exact carrier instance

#### Scenario: Dispatch from a legacy instance

- **WHEN** a definition-free legacy carrier instance supports a dispatch name
  through its hook
- **THEN** `dispatch_op` returns the hook's operation implementation

#### Scenario: Reject class dispatch

- **WHEN** `dispatch_op` is called on a carrier class without an instance
- **THEN** the call fails with `TypeError`

### Requirement: Every dispatch returns a fresh Operation

Definition-backed dispatch and the legacy dispatch hook SHALL each return an
`Operation` that has not previously been dispatched. Another return type SHALL
make `dispatch_op` fail with `TypeError`. Returning an already dispatched or
cached operation SHALL make `dispatch_op` fail with `TypeError` and SHALL not
replace the operation's original dispatch metadata or state.

Two successful calls for the same name SHALL return distinct operation
objects. This freshness SHALL isolate resolved invocations, provider or
composite completion state, saved inputs, execution options, and other
invocation state.

#### Scenario: Dispatch the same name twice

- **WHEN** a carrier dispatches one supported name twice
- **THEN** the results have distinct identities and independent state while
  sharing the same central semantic definition

#### Scenario: Reject a cached operation

- **WHEN** a legacy hook returns an `Operation` that a previous dispatch
  already marked
- **THEN** dispatch fails with `TypeError` and the first dispatch metadata is
  retained

### Requirement: Carrier is open while shipped implementations are closed

`Carrier` SHALL remain open for new sibling implementations, including Python
and native subclasses that either implement the existing storage and dispatch
contract or register a `CarrierDefinition`. `DependentCarrier` SHALL remain
open for dependent implementations. The existing storage, dispatch-hook,
capability, and movement registration path SHALL remain available for every
exact class without a `CarrierDefinition`.

Every shipped concrete carrier, including Generic, CPU, Metal, FileBacked,
Evictable, the block-storage carrier, and TiledEvictable, SHALL be closed at
runtime and declared final on its public typed import paths. `BlockDevice`
itself SHALL remain the existing storage resource and allocator used by the
definition-free block carrier; it SHALL not be treated as a Tensor carrier or
as a `StorageProvider`. An attempt to
subclass a shipped concrete carrier SHALL fail with `TypeError` identifying it
as closed and directing extension to a sibling `Carrier`.

#### Scenario: Implement a sibling carrier

- **WHEN** a caller subclasses `Carrier`, supplies storage behavior, and
  registers a valid `CarrierDefinition`
- **THEN** instances participate in generic dispatch, capabilities, movement,
  and optional facet contracts according to that definition

#### Scenario: Allocate through a block device resource

- **WHEN** `BlockDevice.allocate` creates a BlockDeviceCarrier
- **THEN** the resulting Tensor is backed by the block carrier and the device
  remains a distinct resource rather than a definition or Tensor carrier

#### Scenario: Reject specialization of a shipped carrier

- **WHEN** a caller attempts to subclass any shipped concrete carrier
- **THEN** class creation fails with the common closed-carrier `TypeError`

### Requirement: Generic and CPU dispatch the supported operation surface

Generic and CPU SHALL retain fresh definition-free implementations for every
supported computational name registered by `operation-dtype-policy`. Metal
SHALL retain fresh definition-free implementations for every such registered
name for which it advertises at least one executable plan. All three SHALL dispatch
the shared representation-preserving names `as_strided`, `broadcast_to`,
`permute`, `rearrange`, `reshape`, `squeeze`, `unsqueeze`, and `view`.

The planned dispatch names SHALL include `add`, `sub`, `mul`,
`elementwise_mul`, `div`, `pow`, `neg`, `abs`, `sign`, `recip`, `sqrt`,
`rsqrt`, `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `erf`, `floor`, `ceil`,
`round`, `maximum`, `minimum`, `rem`, `eq`, `ne`, `lt`, `le`, `logical_not`,
`relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`, `leaky_relu`,
`reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `argmax`, `argmin`,
`cumsum`, `matmul`, `conv_general`, `gather`, `scatter`, `scatter_add`,
`select`, and `clamp`; the backend SHALL also provide the implementation needed
to return both observable values and indices for `sort` and `topk` without the
contract fixing private dispatch names.

An unknown name SHALL fail with `NotImplementedError`. FileBacked and the
block-storage carrier SHALL retain their existing storage-only rejection of
every computational dispatch name.

#### Scenario: Dispatch a supported Generic operation

- **WHEN** Generic dispatches a supported name twice
- **THEN** it returns two fresh existing Generic or shared operation
  implementations carrying Generic dispatch metadata

#### Scenario: Dispatch every registered Metal name

- **WHEN** Metal dispatches a registered computational name whose plan it
  advertises
- **THEN** a fresh existing Metal operation implementation executes through
  the definition-free TileLang path

#### Scenario: Refuse FileBacked computation

- **WHEN** FileBacked receives a computational dispatch name
- **THEN** dispatch fails with `NotImplementedError`

#### Scenario: Refuse block-storage computation

- **WHEN** the block-storage carrier receives a computational dispatch name
- **THEN** dispatch fails with `NotImplementedError`

### Requirement: Composite adapters own the visible dispatch boundary

A definition-free carrier that executes through another carrier SHALL retain
its existing fresh composite adapter, operand translation, lowered execution,
result restoration, dispatch metadata, and autograd behavior.

TiledEvictable SHALL dispatch one fresh definition-backed Operation for each
name in its bounded direct operation surface. Its CompositeProvider SHALL
receive the immutable `ResolvedInvocation` through the documented
`prepare(carrier, invocation)` callback and SHALL return only
`PreparedComposite` or `Unsupported`. It MAY prepare conforming work through a
definition-free dependency's existing public dispatch and lowered-execution
behavior without changing that dependency class.

The outer TiledEvictable Operation SHALL remain the sole visible autograd and
dispatch node for that direct call. Dependency work SHALL create no additional
visible node on the published result. The framework SHALL retain the outer
exact-class dispatch metadata, validate `ProviderResult` against the outer
invocation, and apply only the central OperationDefinition's VJP. A direct
operation SHALL mean the Tensor's complete logical value, never its current
resident tile set, and SHALL publish the ordinary dense compute-carrier result
declared by the tiled composite contract. A compact Tensor obtained by
explicitly waiting for projection SHALL already be an ordinary compute Tensor
and subsequent operations SHALL dispatch directly on that carrier.

#### Scenario: Execute through a composed backend

- **WHEN** an Evictable carrier runs an operation implemented by its promoted
  primary
- **THEN** its existing definition-free adapter restores the result to
  Evictable and remains the visible operation boundary

#### Scenario: Execute directly on a full tiled Tensor

- **WHEN** an operation receives a Tensor backed by TiledEvictable
- **THEN** it consumes the full logical Tensor and returns the declared dense
  compute Tensor with one visible outer operation node

#### Scenario: Execute after explicit projection

- **WHEN** an `AwaitProjection` completes and its compact Tensor is passed to
  an operation
- **THEN** the operation dispatches normally on that compact Tensor's compute
  carrier and creates its own ordinary operation node
