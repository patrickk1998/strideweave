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

`carrier.dispatch_op(operation_name)` SHALL require a Carrier instance and a
string dispatch name. `carrier` names the Carrier instance whose implementation
is asked to dispatch. `operation_name` names the canonical operation requested
and SHALL be a string. Calling `dispatch_op` on `Carrier` or another carrier
class without a `carrier` instance SHALL fail with `TypeError`; passing a
non-string `operation_name` SHALL fail with `TypeError`. The call SHALL ask that
instance's dispatch hook for the operation implementation.

The default Carrier hook SHALL fail with `NotImplementedError` identifying the
unsupported dispatch name. A storage-only carrier MAY use that default for
every name.

#### Scenario: Dispatch from an instance

- **WHEN** a carrier instance supports a dispatch name
- **THEN** `dispatch_op` returns an operation for that instance's implementation

#### Scenario: Reject class dispatch

- **WHEN** `dispatch_op` is called on a carrier class without an instance
- **THEN** the call fails with `TypeError`

### Requirement: Every dispatch returns a fresh Operation

The dispatch hook SHALL return an `Operation` that has not previously been
dispatched. Another return type SHALL make `dispatch_op` fail with `TypeError`.
Returning an already dispatched or cached operation SHALL make `dispatch_op`
fail with `TypeError` and SHALL not replace the operation's original dispatch
metadata or state.

Two successful calls for the same name SHALL return distinct operation
objects. This freshness SHALL isolate context, saved inputs, execution options,
and other state between invocations.

#### Scenario: Dispatch the same name twice

- **WHEN** a carrier dispatches one supported name twice
- **THEN** the results have the expected implementation type but distinct
  identities and independent state

#### Scenario: Reject a cached operation

- **WHEN** a hook returns an Operation that a previous dispatch already marked
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
and native subclasses that implement its storage and dispatch contract.
`DependentCarrier` SHALL remain open for dependent implementations.

Generic, CPU, FileBacked, and Evictable SHALL be closed implementations at
runtime and SHALL be declared final on their public typed import paths. An
attempt to subclass any of those four SHALL fail with `TypeError` identifying
it as a closed carrier implementation and directing extension to a sibling
Carrier.

#### Scenario: Implement a sibling carrier

- **WHEN** a caller subclasses Carrier and supplies the required storage and
  dispatch behavior
- **THEN** instances participate in the ordinary dispatch contract

#### Scenario: Reject specialization of a shipped carrier

- **WHEN** a caller attempts to subclass Generic, CPU, FileBacked, or Evictable
- **THEN** class creation fails with the common closed-carrier `TypeError`

### Requirement: Generic and CPU dispatch the supported operation surface

Generic and CPU SHALL dispatch fresh implementations for the supported
computational names registered by `operation-dtype-policy` and the shared
representation-preserving names `as_strided`, `broadcast_to`, `permute`,
`rearrange`, `reshape`, `squeeze`, `unsqueeze`, and `view`.

The planned dispatch names SHALL include `add`, `sub`, `mul`,
`elementwise_mul`, `div`, `pow`, `neg`, `abs`, `sign`, `recip`, `sqrt`,
`rsqrt`, `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `erf`, `floor`, `ceil`,
`round`, `maximum`, `minimum`, `rem`, `eq`, `ne`, `lt`, `le`, `logical_not`,
`relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`, `leaky_relu`,
`reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `argmax`, `argmin`,
`cumsum`, `matmul`, `conv_general`, `gather`, `scatter`, `scatter_add`,
`select`, `clamp`, and the internal value/index dispatch names for sort and
topk.

An unknown name SHALL fail with `NotImplementedError`. FileBacked SHALL fail
with `NotImplementedError` for every computational dispatch name.

#### Scenario: Dispatch a supported Generic operation

- **WHEN** Generic dispatches a supported name twice
- **THEN** it returns two fresh Generic or shared operation implementations
  with Generic dispatch metadata

#### Scenario: Refuse FileBacked computation

- **WHEN** FileBacked receives a computational dispatch name
- **THEN** dispatch fails with `NotImplementedError`

### Requirement: Planned execution passes the backend capability gate

Before a planned simple-dtype operation allocates output or performs backend
work, the implementation SHALL resolve the central `OperationPlan` and require
an exact matching backend capability as defined by `backend-capabilities`.
The implementation SHALL execute operand conversions, arithmetic,
accumulation, and output dtype from that accepted plan rather than deriving a
local policy.

Generic and CPU SHALL apply this preflight to their planned operations. An
unsupported plan SHALL raise `UnsupportedOperationPlan` before result
allocation or kernel entry. An operation name absent from the policy and a
legacy opaque Generic operation MAY retain their documented unplanned path.

#### Scenario: Refuse a plan before backend work

- **WHEN** dispatch reaches a resolved plan the carrier does not advertise
- **THEN** execution raises `UnsupportedOperationPlan` before allocating or
  entering the implementation

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

A carrier that executes through another carrier SHALL return its own fresh
composite adapter from dispatch. The adapter SHALL own one fresh nested
operation. It SHALL translate every outer Tensor argument into the nested
representation, invoke the nested operation through lowered execution, and
translate the result back into the outer carrier representation.

The adapter SHALL remain the sole visible autograd and dispatch node. Lowered
execution SHALL run the nested forward computation and retain nested state
needed by the adapter without attaching a nested autograd node to the returned
outer Tensor. The adapter SHALL preserve its own outer exact-class dispatch
metadata; the owned nested operation SHALL preserve its nested metadata.

#### Scenario: Execute through a composed backend

- **WHEN** a composite carrier dispatches and runs an operation implemented by
  its owned inner carrier
- **THEN** the result is backed by the composite carrier and its autograd
  context points to the outer adapter rather than the nested operation

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

For a registered simple-dtype operation, the adapter SHALL resolve the plan
from the outer operands and validated options and require it against the outer
hierarchy's snapshot before lowering. The adapter SHALL wrap newly allocated
primary results into a fresh hierarchy of the same tier classes, leaving its
secondary empty until first eviction. A representation-preserving operation
whose primary result reuses the same primary carrier SHALL reuse the same
Evictable carrier.

#### Scenario: Refuse mismatched hierarchies

- **WHEN** one Evictable operation receives operands with different exact
  primary or secondary carrier classes
- **THEN** forward fails with `TypeError` before nested execution

#### Scenario: Lower a supported outer plan

- **WHEN** compatible promoted hierarchies advertise the resolved outer plan
- **THEN** the adapter lowers to the primary, executes once, and restores an
  Evictable result using the same hierarchy kinds

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
