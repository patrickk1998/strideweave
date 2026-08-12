---
title: Backend Capabilities
publish: true
status: stable
order: 60
summary: Exact operation-plan capability descriptors, registration, introspection, and enforcement.
---

# backend-capabilities Specification

## Purpose

Define how a carrier states and enforces the exact resolved operation plans it
can execute, for both class-independent backends and instance-dependent
compositions.

## Terminology

| Term | Meaning |
| --- | --- |
| capability | An immutable, executor-free descriptor asserting that a backend can faithfully execute exactly one resolved operation plan; it mirrors the plan and gates execution without selecting, rewriting, or approximating dtype policy. |
| independent carrier | A carrier whose executable plans are fixed by its exact class. |
| dependent carrier | A `DependentCarrier` whose executable plans depend on the carrier instances it composes. |
| declaration | The complete, one-time capability set assigned to an independent exact class. |
| snapshot | The complete, one-time capability set frozen for one dependent instance. |
| observation | A class-level capability query that consumes an independent exact class's authoritative answer and, if that class has not declared, atomically seals an empty capability set as its final answer. |

## Requirements

### Requirement: A capability mirrors every plan matching dimension

`OperandCapability(role, dtype, convert_to)` SHALL describe one operand
position. `role` names whether the operand is a tensor or weak scalar and SHALL
be an `OperandRole`. `dtype` names the source storage dtype; for a tensor role,
`dtype` SHALL be a `SimpleDType`, and for a weak-scalar role, `dtype` SHALL be
`None`. `convert_to` names the materialization dtype and SHALL be a
`SimpleDType`. When `role`, `dtype`, or `convert_to` violates these constraints,
construction SHALL fail with `TypeError`.

`OperationCapability(operation, operands, compute, accumulation,
accumulator_dtype, output)` SHALL describe one exact executable plan shape.
`operation` names the dispatch operation and SHALL be a string. `operands`
names the ordered operand positions and SHALL be a non-empty tuple of
`OperandCapability` values. `compute` names the per-element arithmetic and
SHALL be an `Arithmetic`. `accumulation` names the term-combination rule and
SHALL be an `Accumulation` or `None`. `accumulator_dtype` names the concrete
floating accumulator and SHALL be a `SimpleDType` or `None`. `output` names the
result storage dtype and SHALL be a `SimpleDType`. When `operation`, `operands`,
`compute`, `accumulation`, `accumulator_dtype`, or `output` has an invalid type
under its named constraint, construction SHALL fail with `TypeError`; when
`operands` is empty, construction SHALL fail with `ValueError`.

Capability and operand records SHALL be immutable and hashable and SHALL expose
no executor or mutable registry.

#### Scenario: Construct a valid capability

- **WHEN** every field describes one coherent resolved-plan shape
- **THEN** construction returns an immutable capability exposing those fields

#### Scenario: Reject a weak scalar with storage dtype

- **WHEN** an OperandCapability has weak-scalar role and a non-`None` dtype
- **THEN** construction fails with `TypeError`

### Requirement: Capabilities match plans exactly

`OperationCapability.from_plan(plan)` SHALL return the immutable capability
whose fields mirror `plan`. For `from_plan`, `plan` names the resolved
`OperationPlan` to mirror. `capability.matches(plan)` SHALL return `True` only
when the operation name, operand count and order, every operand role, source
dtype identity, conversion target identity, compute arithmetic, accumulation
including its presence or absence, accumulator dtype identity, and output dtype
identity all agree. For `matches`, `plan` names the resolved `OperationPlan` to
test against the capability.

A difference in any one field SHALL return `False`. Matching SHALL not promote,
rewrite, or substitute a nearby shape.

#### Scenario: Round-trip a resolved plan

- **WHEN** a capability is constructed with `OperationCapability.from_plan`
- **THEN** `matches` returns `True` for that plan

#### Scenario: Reject a near match

- **WHEN** a plan differs only in operand order, conversion, accumulation, or
  output
- **THEN** `matches` returns `False`

### Requirement: Independent capabilities belong to one exact class

An independent carrier's declaration SHALL belong only to the exact class
named by that declaration. Capability lookup SHALL perform no base-class
traversal. A base declaration SHALL not widen a subclass, a subclass
declaration SHALL not widen a base, and sibling declarations SHALL remain
isolated. An exact class with no declaration SHALL support no plans.

#### Scenario: Query an undeclared subclass

- **WHEN** a base carrier class declared a capability and its distinct subclass
  declared none
- **THEN** the subclass does not support the base class's capability

### Requirement: Public registration is complete, atomic, and one-shot

`register_operation_capabilities(carrier_class, capabilities)` SHALL declare
the complete capability set for an eligible custom independent Carrier class
and return `None`. `carrier_class` names that exact implementation;
`capabilities` names every shape it executes and SHALL be an iterable of
`OperationCapability` objects. An empty iterable SHALL be a complete empty
declaration.

The call SHALL validate the entire iterable, reject a duplicate exact shape
with `ValueError`, and publish either the whole immutable declaration or
nothing. A non-class, the `Carrier` root, an unrelated class, a
`DependentCarrier` class, an already declared or observed class, and any
shipped concrete backend SHALL fail with `TypeError`. A non-capability entry
SHALL fail with `TypeError`. A rejected declaration SHALL leave an eligible
class open when no prior declaration or observation closed it.

Successful declaration SHALL seal the class. A second declaration SHALL fail
and SHALL not change the original set.

#### Scenario: Declare a custom independent backend

- **WHEN** an eligible custom carrier declares a valid set before observation
- **THEN** the complete set becomes its final exact-class capability answer

#### Scenario: Reject duplicate shapes atomically

- **WHEN** one declaration contains the same exact capability shape twice
- **THEN** registration fails with `ValueError` and publishes none of its
  entries

### Requirement: First observation seals an undeclared independent class empty

Calling any class-level enumeration, matching, support, reason, or requirement
query SHALL observe that exact class. If it has no declaration, observation
SHALL atomically seal its empty set. A later registration attempt SHALL fail
with `TypeError`, so the first observed answer never changes.

#### Scenario: Observe before declaring

- **WHEN** a custom independent class is queried before it registers
  capabilities
- **THEN** the query reports no support and permanently seals that empty answer

### Requirement: Class-level queries share one deterministic set

`capabilities_for_carrier_class(carrier_class, operation=None)` SHALL return a
tuple of that exact class's capabilities ordered deterministically by stable
field names. `carrier_class` names the exact carrier implementation whose
declaration is queried and SHALL be a class. `operation` names an optional
dispatch-name filter; it SHALL be optional and SHALL default to `None`, while a
string SHALL restrict the tuple to that operation. When `carrier_class` is not
a class, the query SHALL fail with `TypeError`.

`matching_capability(carrier_class, plan)` SHALL return the single exact match
or `None`. `supports_operation_plan(carrier_class, plan)` SHALL return whether
that match exists. `unsupported_plan_reason(carrier_class, plan)` SHALL return
`None` when supported; otherwise it SHALL return a stable sentence that
distinguishes no capability for the operation from other capabilities for that
operation but no match for this shape. `require_capability(carrier_class,
plan)` SHALL return the exact match or raise `UnsupportedOperationPlan` with the
same reason.

For each of these matching, support, reason, and requirement queries,
`carrier_class` names the exact carrier implementation whose declaration is
queried and `plan` names the resolved plan to compare. `carrier_class` SHALL be
a class and `plan` SHALL be an `OperationPlan`; when either constraint is
violated, the query SHALL fail with `TypeError`. All queries SHALL use the same
stored entries.

#### Scenario: Restrict class enumeration by operation

- **WHEN** a class capability enumeration supplies an operation name
- **THEN** it returns only entries with that name in deterministic order

#### Scenario: Require an unsupported shape

- **WHEN** the class has entries for the operation but none exactly matches the
  plan
- **THEN** `require_capability` raises `UnsupportedOperationPlan` whose reason
  describes the unmatched plan shape

### Requirement: Carrier queries select the correct ownership model

`carrier.operation_capabilities(operation_name=None)` SHALL return immutable
capabilities in deterministic order. `operation_name` names an optional
dispatch-name filter; it SHALL be optional and SHALL default to `None`, while a
string SHALL restrict the result to that name.
`carrier.supports_operation_plan(plan)` SHALL return an exact-match boolean.
`carrier.unsupported_plan_reason(plan)` SHALL return `None` or the stable
unsupported reason. `carrier.require_operation_plan(plan)` SHALL return the
matching capability or raise `UnsupportedOperationPlan`.

For each carrier support, reason, and requirement query, `plan` names the
resolved `OperationPlan` to compare and SHALL be an `OperationPlan`; another
value SHALL make the query fail with `TypeError`.

For an independent carrier, these methods SHALL read its exact class's sealed
declaration. For a dependent carrier, they SHALL read that instance's frozen
snapshot. They SHALL describe the implementation's reach rather than filter by
the querying carrier's current storage dtype, size, residency, mutability, or
release state. Passing a carrier class where a carrier instance is required
SHALL fail with `TypeError`.

#### Scenario: Query a Float32 CPU for Int32 plans

- **WHEN** a CPU instance currently stores Float32
- **THEN** its capability enumeration still includes the Int32 plan shapes the
  CPU implementation executes

### Requirement: Unsupported plans fail before execution work

`UnsupportedOperationPlan` SHALL be a `NotImplementedError` subtype identifying
a resolved plan that the carrier does not execute. Every planned backend
execution SHALL require an exact matching capability before allocating result
storage, entering a kernel, or lowering onto another carrier. Failure SHALL
raise `UnsupportedOperationPlan` once and SHALL not execute a nearby supported
shape.

#### Scenario: Gate a backend execution

- **WHEN** a resolved plan has no exact matching carrier capability
- **THEN** execution raises `UnsupportedOperationPlan` before allocation or
  kernel work

### Requirement: Shipped carrier declarations are complete and sealed

Generic and CPU SHALL each expose exactly the planned shapes they faithfully
execute. FileBacked SHALL expose a complete empty declaration. Those independent
classes SHALL be sealed before they are publicly observable and SHALL reject
public attempts to add capabilities.

Evictable SHALL have no class declaration because its reach depends on its
tiers. Its instance snapshot behavior is defined below.

#### Scenario: Query a storage-only backend

- **WHEN** a FileBacked carrier's capabilities are enumerated
- **THEN** the result is the immutable empty tuple and no later registration
  can widen it

### Requirement: DependentCarrier owns capabilities per finalized instance

`DependentCarrier` SHALL remain open for subclassing. A concrete subclass SHALL
implement `_generate_operation_capabilities()` to return the capabilities one
constructed instance executes. The base implementation SHALL fail with
`NotImplementedError`.

The concrete constructor SHALL call `_finalize_dependent_capabilities()` once,
after its dependencies are valid and before the instance is exposed. The call
SHALL materialize the generator once, require only capability entries, reject a
duplicate exact shape, sort them deterministically, freeze the complete
snapshot for that instance identity, and return `None`. Invalid entries SHALL
fail with `TypeError`; duplicates SHALL fail with `ValueError`; generation
failure SHALL publish no partial snapshot. A second finalization SHALL fail with
`RuntimeError` and leave the first snapshot unchanged.

Before finalization, every carrier capability query SHALL fail with
`RuntimeError`. Two instances of the same dependent class MAY freeze different
sets, including when the instances compare equal. A dependent class SHALL fail
with `TypeError` if passed to class-level capability registration.

#### Scenario: Freeze two different dependent instances

- **WHEN** two instances of one dependent class compose dependencies with
  different executable plans
- **THEN** each answers from its own immutable snapshot

#### Scenario: Fail generation atomically

- **WHEN** capability generation raises, yields an invalid entry, or repeats a
  shape
- **THEN** finalization fails and subsequent public queries report that the
  instance remains unfinalized

### Requirement: Evictable snapshots executable and storable primary plans

At successful construction, Evictable SHALL freeze the capabilities advertised
by its primary carrier instance whose output dtype is supported by both tiers.
It SHALL preserve those exact plans without rewriting or choosing promotion,
and SHALL query a dependent primary through that primary instance's snapshot.

The snapshot SHALL be structural and SHALL remain unchanged by eviction,
promotion, or release. A generated operation result or gradient SHALL be a new
Evictable hierarchy with its own snapshot. If snapshot generation fails,
construction SHALL publish no Evictable and SHALL relinquish any tier ownership
already claimed.

#### Scenario: Drop a result the hierarchy cannot evict

- **WHEN** the primary advertises a plan whose output dtype the secondary tier
  cannot store
- **THEN** the Evictable instance omits that plan and refuses it before any
  lowering or execution
