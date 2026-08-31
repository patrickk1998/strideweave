## Purpose

Define public carrier-neutral operation meaning so validation, results,
effects, reference execution, and automatic differentiation remain identical
across built-in and extension kernel implementations.

## ADDED Requirements

### Requirement: Operation extension values have one stable import surface

`define_operation`, `OperationDefinition`, `OperationSchema`, `OperandSpec`,
`OptionSpec`, `OperandKind`, `REQUIRED`, `NON_DIFFERENTIABLE`, `ResultSpec`,
`ResolvedInvocation`, `BoundOperationCall`, `VJPContext`, and
`ProviderContractError` SHALL be
importable from `strideweave.operation`.

`define_operation`, `OperationDefinition`, `OperationSchema`, `OperandSpec`,
`OptionSpec`, `OperandKind`, `REQUIRED`, `NON_DIFFERENTIABLE`, `ResultSpec`,
and `ProviderContractError` SHALL also be importable from the top-level
package. The module and top-level exports SHALL name the same objects.

#### Scenario: Import custom-operation contracts

- **WHEN** an extension imports schema, definition, resolution, result, and VJP
  types from their documented paths
- **THEN** it needs no backend-private or framework-internal type

### Requirement: OperationSchema binds operands and options before semantic resolution

`OperandKind` SHALL be an enum with the string-valued members
`TENSOR="tensor"`, `WEAK_SCALAR="weak_scalar"`, and
`STRUCTURAL="structural"`; its members SHALL use these explicit values rather
than generated ordinal values.

`OperandSpec(name, kind, differentiable=False)` SHALL be immutable. `name`
SHALL be a non-empty Python identifier, `kind` SHALL be an `OperandKind`, and
`differentiable` SHALL be a `bool` that MAY be true only for `TENSOR`.
Invalid field types SHALL fail with `TypeError`; an empty or invalid name or a
differentiable non-Tensor operand SHALL fail with `ValueError`.

`OptionSpec(name, default=REQUIRED)` SHALL be immutable. `name` SHALL be a
non-empty Python identifier. `REQUIRED` SHALL be the one public immutable
marker for an option with no default. An invalid name type SHALL fail with
`TypeError`; an empty or invalid name SHALL fail with `ValueError`.

`OperationSchema(operands, options=())` SHALL materialize finite iterables of
`OperandSpec` and `OptionSpec` once and expose immutable tuples. `operands`
SHALL be non-empty. A wrong entry type SHALL fail with `TypeError`; an empty
operand tuple, duplicate name, or a name shared by an operand and option SHALL
fail with `ValueError`.

Calling an OperationDefinition SHALL bind positional operands in schema order
and keyword options by exact name. Missing or extra operands, an unknown option,
or two values for one option SHALL fail with `TypeError`; an omitted required
option SHALL fail with `TypeError`; omitted optional values SHALL use their
declared defaults. A TENSOR operand SHALL be a Tensor, a WEAK_SCALAR SHALL be a
supported Python weak scalar and not a Tensor, and STRUCTURAL SHALL be passed
unchanged for semantic validation. A kind mismatch SHALL fail with `TypeError`.
Successful binding SHALL produce a framework-created immutable
`BoundOperationCall` exposing `definition`, ordered `operands`, and a read-only
option mapping before invoking `resolve`.

#### Scenario: Bind defaults and preserve operand order

- **WHEN** a custom operation is called with every positional operand and omits
  one optional option
- **THEN** its resolve callback receives one immutable bound call with operands
  in schema order and the declared default present

#### Scenario: Reject a Tensor in a weak-scalar role

- **WHEN** a Tensor is supplied for a `WEAK_SCALAR` OperandSpec
- **THEN** binding raises `TypeError` before semantic resolution or carrier work

### Requirement: The direct tiled operation set has complete semantic definitions

The built-in names `add`, `elementwise_mul`, `mul`, `relu`, `reduce_sum`, and
`matmul` SHALL each resolve through one immutable `OperationDefinition` for
definition-backed dispatch. `mul` SHALL include its existing tensor/tensor,
tensor/weak-scalar, and weak-scalar/tensor overloads. Each definition SHALL
identify its ordered operand kinds and execution options, validation and
normalization, central dtype plan, result Shape and Layout, allocation effects,
differentiability, saved forward context, reference behavior, and
vector-Jacobian product.

No other existing built-in operation SHALL be migrated to OperationDefinition
by this capability. Definition-free carriers SHALL retain their current
operation implementations for every name, including these six. A compact
ordinary Tensor returned by TiledEvictable projection SHALL therefore retain
the complete operation surface of its compute carrier, while raw direct
TiledEvictable dispatch SHALL expose only these six names.

Resolution SHALL depend only on the semantic definition, normalized public
inputs, and validated execution options. It SHALL complete before capability
enforcement or any provider callback. A carrier provider SHALL select
executable details without replacing any resolved semantic field.

#### Scenario: Resolve one direct tiled operation

- **WHEN** TiledEvictable directly dispatches one of the six central names
- **THEN** resolution fixes validation, dtype, result, save, and differentiation
  before its composite provider prepares legacy dependency work

#### Scenario: Preserve a definition-free operation

- **WHEN** Generic, CPU, or Metal dispatches any existing built-in name
- **THEN** it uses its current definition-free operation implementation rather
  than the new definition-backed adapter

#### Scenario: Reject invalid input before backend work

- **WHEN** an operand or option violates the operation definition
- **THEN** resolution raises the documented public error before capability,
  kernel, composite, or other provider work

### Requirement: Namespaced custom operations use the same public definition contract

`define_operation(name, schema, resolve, reference=None, vjp=...)` SHALL
atomically register and return one callable immutable `OperationDefinition`.
`name` SHALL be a non-empty dot-namespaced string whose segments are Python
identifiers and whose first segment is outside the reserved `strideweave` and
built-in namespaces. `schema` SHALL be an `OperationSchema`.
`resolve(call: BoundOperationCall) -> ResolvedInvocation` SHALL be callable and
SHALL receive the successfully bound call. `reference` SHALL be `None` or a
callable with
`reference(invocation: ResolvedInvocation) -> Tensor | tuple[Tensor, ...]` and
SHALL default to `None`. `vjp` SHALL be supplied explicitly as either
`vjp(context: VJPContext, cotangents: tuple[Tensor | None, ...]) ->
tuple[Tensor | None, ...]` or the public immutable `NON_DIFFERENTIABLE` marker.

Invalid argument types SHALL fail with `TypeError`. An invalid or reserved
name, a duplicate registered name, or omitted `vjp` SHALL fail with
`ValueError` and SHALL register no operation. Registration SHALL be one-shot;
a registered definition SHALL have no replacement or unregistration operation.
The returned OperationDefinition SHALL expose immutable `name`, `schema`,
`resolve`, `reference`, and `vjp` fields.

On each call, a resolve exception SHALL propagate before capability or provider
work. A non-ResolvedInvocation return SHALL fail with `TypeError`. An
invocation whose bound-call identity, operation name, operand roles, option
values, or result/effect fields conflict with the definition SHALL fail with
`ValueError` before capability matching. The framework SHALL not call
`reference` or `vjp` to infer missing resolution fields.

The resolve callback MAY inspect normalized operand kind, Shape, Layout, dtype,
gradient intent, and option values. It SHALL not select semantics from an exact
Carrier class, provider, device, stream, resource, or current residency state;
calls differing only in those execution details SHALL resolve to the same
plan, results, effects, save recipe, and kernel identifier.

Calling the returned definition with its declared operands and options SHALL
perform ordinary carrier dispatch and return the Tensor or Tensor tuple
declared by its result contract. A carrier with no matching kernel pattern or
CompositeProvider preparation SHALL raise `UnsupportedOperationPlan`.
Registering a kernel pattern or facet without an operation definition SHALL
not create or redefine an operation.

#### Scenario: Define and call a custom fused operation

- **WHEN** an extension defines a complete `acme.flash_attention` operation
  and an input carrier has a matching kernel pattern
- **THEN** calling the definition executes through ordinary dispatch and
  returns the centrally declared result

#### Scenario: Require an explicit gradient declaration

- **WHEN** a custom definition supplies neither a VJP nor
  `NON_DIFFERENTIABLE`
- **THEN** definition fails with `ValueError` before the operation name becomes
  observable

#### Scenario: Leave another carrier unsupported

- **WHEN** a custom operation is called with a carrier whose sealed execution
  set has no matching pattern or composite translation
- **THEN** it fails with `UnsupportedOperationPlan` without acquiring
  backend-local semantics

### Requirement: Resolved invocations fully describe backend obligations

`ResultSpec(dtype, shape, layout, alias_of=None, *, atol=0.0, rtol=0.0)` SHALL
be an immutable result contract. `dtype` SHALL be a DType identity, `shape` a
Shape, `layout` a Layout with that exact Shape, and `alias_of` either `None` for
an allocating result or the non-negative positional index of a Tensor operand
that the result aliases by exact Tensor object identity. `alias_of` SHALL not
describe a distinct Tensor view that merely shares carrier storage with an
operand. `atol` and `rtol` SHALL be finite non-negative real numbers other than
`bool` and SHALL define reference-comparison tolerance.
Invalid field types SHALL fail with `TypeError`; Shape disagreement, a negative
or non-finite tolerance, or an out-of-range or non-Tensor alias index when
bound SHALL fail with `ValueError`.

`ResolvedInvocation(call, plan, results, mutated_operands=(),
saved_operands=(), kernel_id=None)` SHALL be immutable. `call` SHALL be the
framework-created `BoundOperationCall` being resolved. `plan` SHALL be the
exact `OperationPlan` whose operation name and ordered operand roles match that
call. `results` SHALL be a non-empty finite iterable of `ResultSpec` values.
`mutated_operands` and `saved_operands` SHALL be finite iterables of unique
non-negative positional Tensor-operand indices and SHALL default to empty.
`kernel_id` SHALL be a non-empty semantic identifier string and SHALL default
to `call.definition.name` when `None`.

Construction SHALL materialize each iterable once. A wrong field or entry type
SHALL fail with `TypeError`; an empty result set, duplicate index, index out of
range, non-Tensor index, plan mismatch, result Shape/Layout mismatch, or empty
kernel identifier SHALL fail with `ValueError`. The invocation SHALL expose
`definition`, canonical operation `name`, ordered normalized operands,
read-only options, exact dtype `plan`, result contracts, mutation effects,
saved-operand recipe, and kernel identifier for the lifetime of forward and
backward.

Kernel preparation SHALL either preserve the entire resolved contract or
report the invocation unsupported. It SHALL not change operand conversion,
arithmetic, accumulation, result dtype, result Layout, aliasing, mutation,
error precedence, or gradient meaning to use a nearby kernel.

#### Scenario: Refuse a nearby kernel

- **WHEN** a candidate kernel differs from the resolved invocation in dtype,
  result Layout, accumulation, or effect
- **THEN** the candidate reports unsupported instead of rewriting the
  invocation

### Requirement: Reference execution provides a carrier-neutral oracle

Each of the six central direct tiled operations SHALL provide reference
execution for every plan it makes available to definition-backed dispatch. A
custom operation with a `reference` callable SHALL make that reference
available for conformance and for execution by an eligible definition-backed
carrier whose sealed patterns include it. A custom operation with
`reference=None` SHALL gain no execution fallback merely because it is
registered. Definition-free Generic SHALL retain its existing reference
implementation and SHALL not acquire custom-operation patterns through this
capability.

Any carrier advertising the same resolved invocation SHALL agree with the
reference on result count, shape, Layout, dtype, logical values within the
corresponding ResultSpec's `atol` and `rtol`, aliasing or mutation effects, error
conditions, and differentiability. Scheduling, compilation, intermediate
physical representation, and completion timing MAY differ without changing
those observable results.

The framework SHALL normalize a single Tensor return to a one-element result
tuple and SHALL preserve a Tensor tuple in order. Another return type or a
non-Tensor tuple entry SHALL fail with `TypeError`; an empty tuple or a result
that conflicts with the invocation's ResultSpecs SHALL fail with
`ProviderContractError` before publication.

#### Scenario: Compare an extension kernel with its reference

- **WHEN** a custom accelerator kernel and its registered reference execute the
  same resolved invocation
- **THEN** their observable results and effects agree

#### Scenario: Omit a definition-backed fallback deliberately

- **WHEN** a custom operation has no reference and a definition-backed carrier
  has no exact kernel pattern for it
- **THEN** calling it on that carrier raises `UnsupportedOperationPlan`

### Requirement: Capability and execution select from the same kernel patterns

An independent definition-backed carrier SHALL advertise an operation plan
only when a sealed built-in or extension kernel pattern matches that plan.
Forward execution SHALL select from those same matching patterns under the
preference and runtime-fallback rules in `carrier-extension`. A static pattern
mismatch SHALL raise `UnsupportedOperationPlan` before invoking any pattern's
prepare callback.

Invocation-specific preparation rejection SHALL leave the static capability
set unchanged. If another statically matching candidate accepts, it SHALL
execute; if every candidate rejects, execution SHALL raise the combined stable
unsupported reason without partial mutation or publication.

#### Scenario: Keep introspection and execution aligned

- **WHEN** a carrier's sealed set has no pattern for one resolved plan
- **THEN** capability introspection omits it and execution rejects it before
  backend work

#### Scenario: Fall back after an optimized rejection

- **WHEN** a higher-preference candidate rejects one runtime shape and a lower
  candidate accepts it
- **THEN** the lower candidate executes while static capability introspection
  remains unchanged

### Requirement: Automatic differentiation belongs to the semantic definition

When a resolved invocation has differentiable Tensor operands and its
definition does not use `NON_DIFFERENTIABLE`, the framework SHALL save exactly
the operands named by `saved_operands`, their versions, the immutable
invocation, and the published outputs. It SHALL create one immutable
`VJPContext` exposing `invocation`, `outputs: tuple[Tensor, ...]`, and
`saved_operands: tuple[Tensor, ...]` in the declared saved-index order.

Backward SHALL call the definition's VJP with that context and one cotangent
entry per result in result order. Each cotangent SHALL be a Tensor satisfying
the result's cotangent contract or `None`. The callback SHALL return a tuple
with one Tensor or `None` for every schema operand whose
`OperandSpec.differentiable` is true, in operand order. Another return type or
entry type SHALL fail with `TypeError`; the wrong tuple length or an
incompatible gradient Shape SHALL fail with `ValueError`; an incompatible
gradient dtype SHALL fail with `TypeError`. Returned Tensor gradients SHALL
enter ordinary autograd accumulation. A definition using
`NON_DIFFERENTIABLE` SHALL attach no differentiable node and SHALL never receive
a VJP call.

After semantic resolution and the bound carrier's released-state check, but
before capability lookup, result-carrier resolution, provider selection,
preparation, submission, mutation, result publication, or graph attachment,
the framework SHALL reject an invocation with `ValueError` when all of the
following hold: graph construction is enabled; the definition has a callable
VJP rather than `NON_DIFFERENTIABLE`; at least one actual Tensor operand whose
`OperandSpec.differentiable` is true participates; and any result that would
otherwise receive the definition-backed differentiable node has `alias_of` set.
The rejection SHALL be atomic across every result in a multi-result invocation.
A released definition-backed carrier SHALL still raise its released-state
`RuntimeError` first.

When graph construction is disabled or the definition uses
`NON_DIFFERENTIABLE`, an exact-object alias MAY proceed to provider execution
and SHALL still undergo the declared identity validation. A distinct Tensor
view that shares storage with an operand but is not that exact Tensor object
SHALL not be rejected by this rule and SHALL retain ordinary view/autograd
behavior.

Kernel packs, kernel providers, and composite providers SHALL use that VJP and
SHALL not supply competing gradient mathematics for a built-in operation. A
custom fused backward MAY be expressed as another complete namespaced
operation invoked by the custom operation's VJP. Saved-input version checks and
graph-lifetime rules SHALL apply at the one central Operation boundary before
the VJP callback runs.

#### Scenario: Backpropagate through two carrier kernels

- **WHEN** two independent carriers execute the same differentiable
  invocation
- **THEN** both use the same semantic VJP and return gradients matching the
  reference behavior

#### Scenario: Use a fused namespaced backward

- **WHEN** a custom operation's VJP invokes a separately defined namespaced
  backward operation
- **THEN** that backward operation undergoes ordinary semantic resolution and
  carrier kernel selection

#### Scenario: Detect a modified saved input

- **WHEN** a saved input is mutated after forward and before backward
- **THEN** the framework raises the ordinary saved-version error before
  provider execution

#### Scenario: Reject a differentiable exact-object alias before provider work

- **WHEN** graph construction is enabled and a definition with a callable VJP
  resolves a differentiable result whose `alias_of` names an actual
  schema-differentiable Tensor operand
- **THEN** the call raises `ValueError` before capability, result-carrier, or
  provider callbacks and leaves the operand and its existing graph unchanged

#### Scenario: Reject one graph-building alias atomically among multiple results

- **WHEN** a multi-result invocation would build differentiable nodes and any
  one of its result contracts declares `alias_of`
- **THEN** the whole invocation raises `ValueError` before provider work or
  publication of any result

#### Scenario: Permit a non-differentiable exact-object alias

- **WHEN** a `NON_DIFFERENTIABLE` definition declares and returns an exact
  operand alias
- **THEN** provider execution proceeds and publication validates exact object
  identity without attaching a differentiable node

#### Scenario: Permit a no-grad exact-object alias

- **WHEN** graph construction is disabled for an otherwise differentiable
  invocation that declares an exact operand alias
- **THEN** provider execution proceeds and publication validates exact object
  identity without attaching a differentiable node

#### Scenario: Preserve a distinct storage-sharing view

- **WHEN** a result is a distinct Tensor object that shares storage with an
  operand and its ResultSpec does not declare `alias_of`
- **THEN** the exact-object alias prohibition does not reject it and ordinary
  view and autograd rules remain in force

### Requirement: Composite translation preserves the declared public boundary

A dependent carrier's CompositeProvider SHALL receive the exact outer carrier
instance and immutable `ResolvedInvocation` through
`result_carriers(carrier, invocation)` and then
`prepare(carrier, invocation)`. The result callback SHALL return one exact
Carrier class per ResultSpec. Preparation SHALL return `PreparedComposite` or
`Unsupported` before submission. Submitted work SHALL return a
`ProviderResult` directly or through `ProviderCompletion[ProviderResult]`.
Another preparation or submission return type SHALL fail with `TypeError`.

How prepared composite work uses dependency carriers SHALL not be part of this
public contract. The outer Operation SHALL remain the sole visible operation
and autograd node for the direct composite call, and the result SHALL preserve
the central operation's values, dtype, Shape, Layout, effects, and VJP. The
framework SHALL validate the ProviderResult against the invocation and the
exact result-carrier tuple before publication.

Definition-free dependent carriers, including Evictable, SHALL retain their
existing composite and autograd behavior outside this OperationDefinition
path. For a full logical TiledEvictable Tensor operand, direct operation
dispatch SHALL consume the complete logical Tensor and publish the dense
configured-compute-carrier result. It SHALL derive neither operands nor output
Shape from the currently resident tile set.

#### Scenario: Preserve Evictable composition

- **WHEN** Evictable lowers an operation onto its promoted primary
- **THEN** its existing definition-free adapter restores the result and remains
  its one visible node

#### Scenario: Publish a full tiled operation compactly

- **WHEN** an operation receives a full logical TiledEvictable Tensor directly
- **THEN** the outer Operation applies the operation to the full logical Shape,
  publishes a dense configured-compute-carrier result, and retains the full
  selection needed for backward

### Requirement: Result publication validates provider output

`ProviderContractError` SHALL be a `RuntimeError` subtype identifying provider
output that conflicts with a `ResolvedInvocation`.

After kernel or composite completion and before returning a Tensor, the
framework SHALL validate the ProviderResult's result count, carrier role,
physical storage size, dtype identity, Shape, Layout, aliasing, authorized
mutation effects, and successful completion against the resolved invocation.
A non-aliased independent-provider result SHALL use the exact dispatch carrier
class; an aliased result permitted to reach provider execution SHALL be the
declared operand object; and a composite result SHALL use the exact class at
its corresponding `CompositeProvider.result_carriers` position.
A mismatch SHALL raise `ProviderContractError` naming the conflicting field,
publish no result, attach no successful autograd node, and advance no public
mutation version.

When kernel or composite submission returns `ProviderCompletion`, the ordinary
Operation call SHALL wait for its terminal outcome, validate the resulting
ProviderResult, and only then return or raise. It SHALL not return the provider
completion object to the operation caller.

#### Scenario: Reject the wrong provider dtype

- **WHEN** a submitted kernel completes with output storage whose dtype differs
  from the resolved output dtype
- **THEN** publication fails before a Tensor with that dtype becomes observable
