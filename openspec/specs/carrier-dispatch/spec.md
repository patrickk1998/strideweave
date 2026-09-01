---
title: Carrier Dispatch
publish: true
status: stable
order: 70
summary: Instance operation dispatch, exact-class metadata, backend extension, preflight, and composite lowering.
---

# carrier-dispatch Specification

## Purpose

Define how a carrier creates stateful operation instances, identifies the exact
dispatching implementation, and composes operations across carrier boundaries.

## Terminology

| Term | Meaning |
| --- | --- |
| dispatch name | The canonical string identifier by which a carrier is asked for an operation implementation and by which the resulting execution is identified in dispatch metadata, dtype planning when registered, and profiling. |
| dispatched operation | A fresh stateful `Operation` instance bound by metadata to one dispatch name and the exact dispatching carrier class, isolating saved inputs, context, execution options, and other invocation state at that public carrier boundary. |
| dispatch hook | The protected carrier-extension factory `_dispatch_op(operation_name)`, which supplies a fresh carrier-specific `Operation` while public dispatch retains validation, freshness enforcement, and metadata ownership. |
| lowered execution | The framework-controlled path for running a nested operation while preserving ordinary preflight, execution-option validation, result validation, profiling and computation hooks, and delegated backward state, but creating no nested visible autograd node. |
| composite adapter | A fresh outer-carrier `Operation` that is the sole visible dispatch and autograd boundary for delegated work, owns one fresh nested operation, enforces the outer capability gate for planned operations, translates outer operands into the nested representation, and restores results and gradients to the outer representation. |
## Requirements
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

### Requirement: Dispatch metadata records name and exact carrier class

After a successful dispatch, the operation's `_operation_name` SHALL equal the
requested string and `_dispatch_carrier_class` SHALL be the exact Python class
of the carrier instance. Metadata SHALL not be inherited from a base-class
declaration or replaced with a nested implementation class.

The exact dispatch carrier class SHALL remain attached through forward,
backward, profiling, and composite adaptation. It identifies the public
carrier boundary that produced the operation.

#### Scenario: Dispatch from a custom subclass

- **WHEN** an open custom Carrier subclass dispatches an operation
- **THEN** the operation metadata names the requested operation and that exact
  custom class

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

### Requirement: Planned execution passes the backend capability gate

Before a registered dtype-planned operation allocates output or performs backend
work, the implementation SHALL resolve the central `OperationPlan` and require
an exact matching backend capability as defined by `backend-capabilities`. The
implementation SHALL execute operand conversions, arithmetic, accumulation,
and output dtype from that accepted plan rather than deriving a local policy.

Generic, CPU, Metal, and a composite adapter for a registered operation SHALL
apply this preflight. An unsupported resolved plan SHALL raise
`UnsupportedOperationPlan` before result allocation, compilation, or kernel
entry. An operand dtype that central planning rejects SHALL propagate that
planning failure before capability lookup or backend work. Only an operation
name absent from the dtype-policy registry MAY retain a documented unplanned
path; no category-backed Generic operation receives an exception.

#### Scenario: Refuse a plan before backend work

- **WHEN** dispatch reaches a resolved plan the carrier does not advertise
- **THEN** execution raises `UnsupportedOperationPlan` before allocating,
  compiling, or entering the implementation

#### Scenario: Reject a category before backend work

- **WHEN** a registered Generic operation is presented with an abstract dtype
  category
- **THEN** central planning fails with `TypeError` before result allocation,
  capability execution, or kernel entry

### Requirement: Execution options are validated at the dispatched boundary

`operation.forward(*inputs, options=None)` SHALL accept only the optional
keyword `options`. `inputs` names the ordered positional operation operands.
`options` names validated non-tensor execution options; it SHALL be optional,
SHALL default to `None`, and when non-`None` SHALL be an
`OperationExecutionOptions` for the operation's dispatch name. An unknown
keyword or an `options` value of another type SHALL fail with `TypeError`;
`options` bound to another operation SHALL fail with `ValueError`.

Validated options SHALL be available to the implementation and SHALL not be
treated as positional tensor inputs or saved autograd inputs. Lowered execution
SHALL apply the same options validation.

#### Scenario: Reject options for another operation

- **WHEN** a dispatched matmul receives options validated for reduce_sum
- **THEN** forward fails with `ValueError` before computation

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

### Requirement: Evictable lowering requires compatible promoted hierarchies

Evictable dispatch SHALL require a promoted primary and SHALL return a fresh
`EvictableOperation` owning the primary's fresh operation.
`EvictableOperation(primary_operation)` SHALL construct and return that
single-use adapter. `primary_operation` names the fresh operation dispatched by
the promoted primary and SHALL be an `Operation`; when `primary_operation` is
not an `Operation`, construction SHALL fail with `TypeError`. Forward SHALL be
single-use; a second forward call SHALL fail with `RuntimeError`.

Every Tensor operand lowered by an Evictable adapter SHALL be backed by an
Evictable, have one subtensor, and be promoted. All Evictable operands SHALL
have the same exact primary carrier class and the same exact secondary carrier
class. A missing Tensor input or wrong Tensor carrier SHALL fail with
`TypeError`; mismatched hierarchy classes SHALL fail with `TypeError`; an
evicted input SHALL fail with `RuntimeError` requiring promotion.

For every operation registered in the dtype policy, the adapter SHALL resolve
the plan from the outer operands and validated options and require it against
the outer hierarchy's snapshot before lowering. A planning failure, including
the `TypeError` for an abstract dtype category, SHALL propagate before nested
allocation, lowering, or execution. The adapter SHALL have no category-backed
planning or capability bypass.

The adapter SHALL wrap newly allocated primary results into a fresh hierarchy
of the same tier classes, leaving its secondary empty until first eviction. A
representation-preserving operation whose primary result reuses the same
primary carrier SHALL reuse the same Evictable carrier.

#### Scenario: Refuse mismatched hierarchies

- **WHEN** one Evictable operation receives operands with different exact
  primary or secondary carrier classes
- **THEN** forward fails with `TypeError` before nested execution

#### Scenario: Lower a supported outer plan

- **WHEN** compatible promoted hierarchies advertise the resolved outer plan
- **THEN** the adapter lowers to the primary, executes once, and restores an
  Evictable result using the same hierarchy kinds

#### Scenario: Reject an abstract category before lowering

- **WHEN** a registered operation receives an Evictable operand whose logical
  dtype is an abstract category
- **THEN** central planning fails with `TypeError` before nested allocation,
  lowering, or execution

### Requirement: Evictable backward restores outer gradients

After a successful forward, `adapter.backward(gradient)` SHALL refresh the nested
operation's saved primary-backed inputs from the current promoted forms of the
original outer inputs, then invoke the nested operation's backward. `gradient`
names the incoming cotangent and SHALL be an Evictable Tensor or a Tensor backed
by the exact expected primary class. When `gradient` is another value or is
backed by another carrier class, backward SHALL fail with `TypeError`.

The nested backward SHALL return one Tensor or `None` per saved outer Tensor
input. A wrong result count SHALL fail with `ValueError`; another result type
SHALL fail with `TypeError`. Each Tensor gradient SHALL be wrapped in a new
Evictable hierarchy matching its corresponding input, and `None` entries SHALL
remain `None`.

An operation result need not be promoted for backward, but every saved input
needed by backward SHALL be promoted. Mutation of a saved outer input after
forward SHALL still be detected by the outer operation's version validation.

#### Scenario: Backpropagate through an adapter

- **WHEN** forward completed and all saved inputs needed by backward are
  promoted and unmodified
- **THEN** backward delegates through retained nested state and returns
  Evictable gradients matching the original input hierarchies

#### Scenario: Reject an evicted saved input

- **WHEN** a saved input remains evicted at backward time
- **THEN** backward fails with `RuntimeError` requiring promotion

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
