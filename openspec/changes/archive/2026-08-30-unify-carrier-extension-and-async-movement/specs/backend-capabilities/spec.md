## ADDED Requirements

### Requirement: Definition-backed independent capabilities derive from executable patterns

For an exact independent carrier class with a registered CarrierDefinition,
capability enumeration SHALL equal the complete immutable duplicate-free union
of `OperationCapability` values declared in the built-in and KernelPack
patterns frozen for that exact class. Every pattern plan SHALL name that
pattern's registered operation and SHALL be the same exact-match descriptor
used to admit a `ResolvedInvocation` to runtime preparation. Finalization SHALL
collapse the same exact plan declared by multiple fallback candidates, sort by
the existing deterministic capability order, and publish either the complete
set or no set.

A definition with no kernel provider SHALL derive the empty set. Transfer
routes and typed facets SHALL not add computational capabilities.
Invocation-specific Shape, Layout, alignment, resource, device, mutability,
release, or residency state SHALL not change the static set. Runtime candidate
fallback SHALL select only among patterns already contributing the matched
plan.

#### Scenario: Derive built-in and extension reach

- **WHEN** a definition-backed custom accelerator freezes built-in patterns and
  an eligible exact-carrier extension pack before observation
- **THEN** capability enumeration reports the deterministic union of their
  matched plans without a separate capability declaration

#### Scenario: Derive storage-only reach

- **WHEN** a definition-backed custom storage carrier has storage and transfer
  routes but no kernel provider
- **THEN** its operation capability tuple is empty

#### Scenario: Preserve capability during runtime fallback

- **WHEN** one matching optimized candidate rejects an invocation-specific
  condition and another matching candidate executes
- **THEN** capability enumeration retains the one matched operation plan

### Requirement: Dependent capabilities derive through composition without kernel ownership

For a definition-backed `DependentCarrier`, capability finalization SHALL use
the complete finite iterable returned by
`CompositeProvider.capabilities(carrier)`. The provider SHALL include an outer
`OperationCapability` only when the instance's validated dependencies can
execute it and its `result_carriers` callback can name conforming exact result
classes.
The framework SHALL validate every entry, reject duplicates, apply the existing
deterministic capability order, and freeze the tuple before exposing the
instance. The dependent instance SHALL own no KernelProvider or KernelPack.

The derived snapshot SHALL be immutable for that instance and SHALL not change
when dependency residency, mutability, release state, or runtime kernel
selection changes. Two instances of the same dependent class MAY derive
different snapshots from different dependency instances.

#### Scenario: Derive TiledEvictable reach from CUDA

- **WHEN** a TiledEvictable instance uses a CUDA compute carrier whose exact
  class includes a CUDA kernel pack
- **THEN** the tiled instance derives reachable outer plans through CUDA
  without attaching or copying that pack

#### Scenario: Exclude an unpublishable result

- **WHEN** a dependency supports an inner plan but the composite cannot publish
  conforming exact result Carrier classes
- **THEN** the dependent instance omits the corresponding outer capability

## MODIFIED Requirements

### Requirement: Public registration is complete, atomic, and one-shot

`register_operation_capabilities(carrier_class, capabilities)` SHALL declare
the complete capability set for an eligible definition-free custom independent
Carrier class and return `None`.
`carrier_class` names that exact implementation; `capabilities` names every
shape it executes and SHALL be an iterable of `OperationCapability` objects.
An empty iterable SHALL be a complete empty declaration.

The call SHALL validate the entire iterable, reject a duplicate exact shape
with `ValueError`, and publish either the whole immutable declaration or
nothing. A non-class, the `Carrier` root, an unrelated class, a
`DependentCarrier` class, an already declared or observed class, a class with a
registered CarrierDefinition, and any shipped concrete backend SHALL fail with
`TypeError`. A non-capability entry SHALL fail with `TypeError`. A rejected
declaration SHALL leave an eligible class open when no prior declaration,
definition, or observation closed it.

Successful declaration SHALL seal the class. A second declaration SHALL fail
and SHALL not change the original set.

#### Scenario: Declare a custom independent backend

- **WHEN** an eligible definition-free custom carrier declares a valid set
  before observation
- **THEN** the complete set becomes its final exact-class capability answer

#### Scenario: Reject duplicate shapes atomically

- **WHEN** one declaration contains the same exact capability shape twice
- **THEN** registration fails with `ValueError` and publishes none of its
  entries

#### Scenario: Reject separate capabilities for a definition-backed carrier

- **WHEN** a custom carrier has registered a CarrierDefinition and then calls
  `register_operation_capabilities`
- **THEN** registration fails with `TypeError` and its pattern-derived set
  remains the only independent capability authority

### Requirement: Shipped carrier declarations are complete and sealed

Generic, CPU, and Metal SHALL retain the complete exact-class capability sets
declared by their existing definition-free implementations. Metal's set SHALL
contain at least one executable plan for every computational operation name
registered by `operation-dtype-policy`; it need not contain every plan exposed
by Generic or CPU. FileBacked and BlockDeviceCarrier SHALL retain their
complete empty definition-free declarations. Those independent exact classes
SHALL remain sealed against later public capability registration and, because
they have no CarrierDefinition, SHALL reject kernel-pack registration.

Evictable SHALL retain its definition-free per-instance dependent capability
snapshot. TiledEvictable SHALL have no independent class-level capability or
kernel pack; its CarrierDefinition SHALL contain a CompositeProvider, and each
instance SHALL freeze the subset of the bounded central tiled operation surface
that its composed compute carrier can execute and whose results it can publish.

#### Scenario: Cover the complete Metal dispatch-name set

- **WHEN** Metal's sealed capability set is compared with the central
  registered computational operation names
- **THEN** every registered name has at least one Metal capability and its
  definition-free declaration remains the authority

#### Scenario: Query a storage-only backend

- **WHEN** a FileBacked or BlockDeviceCarrier instance's capabilities are
  enumerated
- **THEN** the result is its existing immutable empty tuple and no definition
  or pack is installed by this capability

#### Scenario: Keep a dependent class pack-free

- **WHEN** Evictable or TiledEvictable capabilities are queried
- **THEN** the answer comes from that instance's definition-free or
  definition-backed dependent snapshot rather than an attached pack

### Requirement: DependentCarrier owns capabilities per finalized instance

`DependentCarrier` SHALL remain open for subclassing. A definition-backed
concrete subclass SHALL derive the capabilities one constructed instance can
execute through its `CompositeProvider` and validated dependencies. A legacy
definition-free concrete subclass SHALL implement
`_generate_operation_capabilities()` while it remains definition-free; the
legacy base implementation SHALL fail with `NotImplementedError`.

The concrete constructor SHALL finalize dependent capabilities once, after its
dependencies are valid and before the instance is exposed. Definition-backed
finalization SHALL materialize
`CompositeProvider.capabilities(carrier)` once. Legacy finalization SHALL
materialize the generator once. Both paths SHALL require only capability
entries, reject a duplicate exact shape, sort deterministically, freeze the
complete snapshot for that instance identity, and return `None`. Invalid
entries SHALL fail with
`TypeError`; duplicates SHALL fail with `ValueError`; derivation or generation
failure SHALL publish no partial snapshot. A second finalization SHALL fail
with `RuntimeError` and leave the first snapshot unchanged.

Before finalization, every carrier capability query SHALL fail with
`RuntimeError`. Two instances of the same dependent class MAY freeze different
sets, including when the instances compare equal. A dependent class SHALL fail
with `TypeError` if passed to class-level capability or kernel-pack
registration.

#### Scenario: Freeze two different dependent instances

- **WHEN** two instances of one dependent class compose dependencies with
  different executable plans
- **THEN** each answers from its own immutable derived snapshot

#### Scenario: Fail generation atomically

- **WHEN** composite translation or a legacy generator raises, yields an
  invalid entry, or repeats a shape
- **THEN** finalization fails and subsequent public queries report that the
  instance remains unfinalized

#### Scenario: Reject dependent pack registration

- **WHEN** a dependent carrier class is passed to kernel-pack registration
- **THEN** registration fails with `TypeError` and no instance snapshot changes
