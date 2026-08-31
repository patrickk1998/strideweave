## Purpose

Define the public exact-class carrier-authoring and kernel-extension contracts
that separate common Tensor behavior from storage, execution, transfer,
composition, and carrier-specific control planes.

## ADDED Requirements

### Requirement: Public extension values have one stable import surface

`CarrierDefinition`, `StorageProvider`, `KernelProvider`, `TransferProvider`,
`TransferRoute`, `CompositeProvider`, `CarrierFacet`,
`KernelExecutionInterface`, `KernelPack`, `KernelPattern`, `Unsupported`,
`PreparedKernel`, `PreparedTransfer`, `PreparedComposite`, `ProviderResult`,
`ProviderCompletion`, `TransferRequest`, `register_carrier_definition`, and
`register_kernel_pack` SHALL be importable from
`strideweave.carriers.extension`.

`CarrierDefinition`, `StorageProvider`, `KernelProvider`, `TransferRoute`,
`CompositeProvider`, `CarrierFacet`, `KernelExecutionInterface`, `KernelPack`,
`KernelPattern`, `Unsupported`, `register_carrier_definition`, and
`register_kernel_pack` SHALL also be importable from the top-level package.
The module and top-level exports SHALL name the same objects.

#### Scenario: Import a carrier-authoring surface

- **WHEN** an extension imports the common definition, provider, route, pack,
  pattern, and unsupported-result types
- **THEN** the documented module paths expose one identical set of public
  objects without requiring a backend-private import

### Requirement: A carrier definition is immutable and exact-class scoped

`CarrierDefinition(storage, kernels=None, transfers=(), composite=None,
facets=())` SHALL describe one carrier implementation. `storage` SHALL be a
`StorageProvider`. `kernels` SHALL be a `KernelProvider` or `None` and SHALL
default to `None`. `transfers` SHALL be a finite iterable of `TransferRoute`
values and SHALL default to empty. `composite` SHALL be a `CompositeProvider`
or `None` and SHALL default to `None`. `facets` SHALL be a finite iterable of
`CarrierFacet` instances and SHALL default to empty.

Construction SHALL materialize the iterables once, expose them as immutable
tuples, and leave every field immutable. Another field or entry type SHALL fail
with `TypeError`; two routes with the same exact source and destination classes
or two facets of the same exact facet class SHALL fail with `ValueError` before
a definition is returned.

`register_carrier_definition(carrier_class, definition)` SHALL bind one complete
definition to the exact class named by `carrier_class` and return `None`.
`carrier_class` SHALL be a concrete `Carrier` subclass other than `Carrier`
itself, and `definition` SHALL be a `CarrierDefinition`; another input SHALL
fail with `TypeError`. Every route's source class SHALL be exactly
`carrier_class`; a mismatch SHALL fail with `TypeError`. Registration SHALL
validate the complete definition and publish all providers or none. Lookup and
execution SHALL perform no base-class traversal.

#### Scenario: Register an exact sibling carrier

- **WHEN** a custom Carrier class registers a valid definition before its first
  observation
- **THEN** every instance of that exact class uses that immutable definition

#### Scenario: Keep a subclass independent

- **WHEN** a subclass of a definition-backed class has no exact definition
- **THEN** it acquires none of the base class's storage, kernels, routes,
  facets, or operation reach

### Requirement: Storage providers define the complete generic storage surface

`StorageProvider(supports_dtype, physical_size, allocate, release, read,
write, elementwise_fallback=False)` SHALL store six required callbacks and one
boolean fallback permission as immutable fields. Each callback argument SHALL
be callable and `elementwise_fallback` SHALL be a `bool`; another value SHALL
fail construction with `TypeError`.

The framework SHALL invoke the callbacks with these signatures and contracts:

| Callback | Contract |
| --- | --- |
| `supports_dtype(carrier, dtype) -> bool` | Receives the exact definition-backed Carrier instance being initialized or queried and a DType identity, and returns a `bool`; `False` rejects construction or allocation for that dtype. |
| `physical_size(carrier) -> int` | Receives an exact-class Carrier instance and returns its non-negative physical slot count. |
| `allocate(carrier, slots) -> None` | Receives a live mutable exact-class Carrier and a non-negative integer extent, establishes storage of that extent, and returns `None`. |
| `release(carrier) -> None` | Relinquishes the exact instance's provider-owned storage and returns `None`; the framework supplies at most one non-idempotent release transition. |
| `read(carrier, index) -> object` | Returns the value at an in-range non-negative physical slot in a form accepted by the carrier's dtype. |
| `write(carrier, index, value) -> None` | Writes one dtype-compatible value to an in-range non-negative physical slot and returns `None`. |

Before the framework first invokes `supports_dtype` for an instance, that exact
instance SHALL have established every definition-required configuration on
which structural storage support can depend, including validated dependency
tier instances. The framework SHALL pass that same exact instance when
validating its construction or allocation and when answering a later
`supports_storage_dtype` query. A fixed provider MAY ignore `carrier`; a
composite provider MAY inspect already-established dependency configuration.

`supports_dtype` SHALL recognize `dtype` by object identity, allocate and
mutate nothing, and return an answer independent of the instance's current
dtype, physical size, mutability, ownership, residency, and release state. It
SHALL remain available after release. The framework SHALL validate exact
carrier identity before every callback that receives a carrier, and SHALL
validate callback-specific liveness, mutability, slot extent, and index range
before invoking callbacks for which those conditions apply. A callback return
of the wrong public type SHALL fail with `TypeError`; a negative
`physical_size` SHALL fail with `ValueError`; and provider exceptions SHALL
propagate without changing Tensor metadata, public versions, ownership, or
graph state.
Elementwise movement fallback SHALL be permitted for a definition-backed pair
only when both exact definitions' storage providers set
`elementwise_fallback=True`.

A storage provider MAY use an independent public resource object, but that
resource SHALL remain neither a `StorageProvider` nor the Carrier of a Tensor.
The existing `BlockDevice` and `BlockDeviceCarrier` SHALL remain on their
definition-free legacy allocation path in this delivery: `BlockDevice` remains
the separate handle whose public `allocate` operation creates or manages an
actual `BlockDeviceCarrier`; it is not retrofitted into `StorageProvider`.

#### Scenario: Allocate from a separate resource

- **WHEN** a custom definition-backed carrier's storage provider uses a
  separate device resource to allocate storage
- **THEN** the resulting Tensor is backed by the custom carrier while the
  device resource remains a distinct provider-owned object

#### Scenario: Reject an invalid provider return

- **WHEN** `physical_size` returns a negative value or `allocate` returns a
  non-`None` value
- **THEN** the framework fails before publishing a Tensor or version change

#### Scenario: Ignore the carrier for fixed dtype support

- **WHEN** a definition-backed independent carrier uses a storage provider
  whose supported dtype set is fixed across instances
- **THEN** the framework supplies the exact carrier and DType identity, and the
  provider may ignore the carrier while returning its identity-based boolean
  answer

#### Scenario: Derive composite dtype support from configured tiers

- **WHEN** a TiledEvictable has established its configured primary and
  secondary tier instances before storage support validation
- **THEN** its storage provider receives that exact TiledEvictable and reports
  the identity-based intersection of the dtypes supported by both tiers

### Requirement: Provider completion and prepared work have one public contract

`ProviderCompletion[T]` SHALL be a runtime-checkable structural protocol with
the `done: bool` property and blocking `wait() -> T` method. It SHALL expose no
`__await__` method or other Python coroutine protocol.
Completion SHALL be thread-safe and idempotent: repeated or concurrent waits
SHALL observe the same terminal value identity or the same terminal exception,
and `done` SHALL become true exactly at that terminal outcome.
A non-boolean `done` value or an object missing any protocol member SHALL fail
provider-result acceptance with `TypeError`.

`ProviderResult(outputs)` SHALL materialize a non-empty finite iterable of
Tensor values once, expose `outputs: tuple[Tensor, ...]`, and remain immutable.
A non-Tensor entry SHALL fail with `TypeError`; an empty iterable SHALL fail
with `ValueError`.

`PreparedKernel(interface, submit)`, `PreparedTransfer(submit)`, and
`PreparedComposite(submit)` SHALL be immutable prepared-work records. The
kernel record SHALL require a `KernelExecutionInterface` instance; every
`submit` SHALL be a zero-argument callable, or construction SHALL fail with
`TypeError`. The framework SHALL invoke a prepared record at most once. Kernel
and composite submission SHALL return either `ProviderResult` or
`ProviderCompletion[ProviderResult]`; transfer submission SHALL return either
`None` or `ProviderCompletion[None]`. Another return type SHALL fail with
`TypeError` as the request's terminal provider failure. These completion
objects SHALL not themselves become Tensor or autograd objects.

#### Scenario: Complete a JIT kernel asynchronously

- **WHEN** a prepared JIT kernel submits device work and returns an incomplete
  `ProviderCompletion[ProviderResult]`
- **THEN** the framework observes its stable terminal result without treating
  the completion as an operation result

#### Scenario: Reject a malformed prepared result

- **WHEN** a kernel submit callback returns a Tensor instead of
  `ProviderResult` or `ProviderCompletion[ProviderResult]`
- **THEN** execution fails with `TypeError` before result publication

### Requirement: Transfer routes expose exact preparation and submission

`TransferRequest` SHALL be a framework-created immutable record exposing the
fields `tensor: Tensor`, `destination: Carrier`, `source_class: type[Carrier]`,
`destination_class: type[Carrier]`, `dtype: DType`, `layout: Layout`, and
`physical_span: int`. The class fields SHALL equal the exact runtime classes,
and `physical_span` SHALL equal `layout.cosize`.

`TransferProvider(prepare)` SHALL require one callable immutable field; another
value SHALL fail construction with `TypeError`.
`prepare(request: TransferRequest) -> PreparedTransfer | Unsupported` SHALL
validate request-specific dtype, Layout, device, resource, and alignment
constraints without changing either Carrier, submitting transfer work, or
publishing state. Another return type SHALL fail with `TypeError`. Returning
`Unsupported(reason)` SHALL permit documented elementwise fallback only when
both exact storage providers permit it; otherwise movement SHALL raise
`NotImplementedError` naming the exact pair and reason. Another exception SHALL be the terminal
preparation failure and SHALL not select another route.

`TransferRoute(source_class, destination_class, route_id, provider)` SHALL be
an immutable directional route. Both endpoint arguments SHALL be concrete
`Carrier` subclasses, `route_id` SHALL be a non-empty string, and `provider`
SHALL be a `TransferProvider`; invalid field types SHALL fail with `TypeError`
and an empty identifier SHALL fail with `ValueError`. Route matching SHALL use
both exact class identities. Successful preparation SHALL submit the returned
`PreparedTransfer`; its terminal `None` means destination contents are ready
for framework validation and transactional publication.

#### Scenario: Reject an invocation-specific transfer condition

- **WHEN** an exact route cannot transfer one source Layout and returns
  `Unsupported` before submission
- **THEN** movement uses elementwise fallback only when both storage providers
  opted into it, with neither Carrier changed by preparation

### Requirement: Independent and dependent definitions have distinct execution ownership

`KernelProvider(interface, patterns=())` SHALL require one
`KernelExecutionInterface` instance and a finite iterable of `KernelPattern`
values. It SHALL materialize and expose the patterns as an immutable tuple.
Another interface or pattern type SHALL fail with `TypeError`, and duplicate
`kernel_id` values SHALL fail with `ValueError`.

`CompositeProvider(capabilities, result_carriers, prepare)` SHALL require three
callables.
`capabilities(carrier) -> Iterable[OperationCapability]` SHALL receive the
constructed dependent Carrier and return its complete finite executable set.
`result_carriers(carrier, invocation) -> tuple[type[Carrier], ...]` SHALL
return one exact result Carrier class per `ResultSpec` in result order.
`prepare(carrier, invocation) -> PreparedComposite | Unsupported` SHALL receive
that same carrier and one `ResolvedInvocation`; it SHALL preserve that
invocation's semantics and either return prepared work or reject it before
submission and public state change. A callback return or yielded entry of the
wrong type, including a result class that is not a concrete Carrier subclass,
SHALL fail with `TypeError`; a duplicate capability or wrong result tuple
length SHALL fail with `ValueError`; another callback exception SHALL
propagate. Capability
results SHALL be materialized once, sorted by the existing deterministic
capability order, and frozen for the instance. Result-carrier validation SHALL
complete before `prepare` is invoked.

An independent `Carrier` definition MAY supply a `KernelProvider` and SHALL
supply no `CompositeProvider`. A `DependentCarrier` definition SHALL supply a
`CompositeProvider` and SHALL supply no `KernelProvider`. Supplying an invalid
combination SHALL make registration fail with `TypeError` without publishing
the definition. An independent definition with no kernel provider SHALL be
storage-only. A dependent carrier SHALL derive execution through its validated
dependency instances and SHALL never own or copy their KernelPacks.

A definition-backed dependent carrier MAY compose definition-free Carrier
instances. Its capability, result-carrier, and preparation callbacks SHALL use
the dependencies' existing public capability, dispatch, allocation, and
movement behavior without requiring a CarrierDefinition on those exact
dependency classes. Whether a dependency is definition-backed SHALL not change
the dependent carrier's public Tensor semantics.

#### Scenario: Define a custom storage-only carrier

- **WHEN** an independent carrier registers storage and transfer providers but
  no kernel provider
- **THEN** it can allocate and move storage while advertising no computational
  capability

#### Scenario: Reject kernels on a dependent carrier

- **WHEN** a dependent carrier definition supplies a kernel provider
- **THEN** definition registration fails with `TypeError` and publishes none of
  its providers

#### Scenario: Compose a definition-free compute carrier

- **WHEN** TiledEvictable is constructed with a definition-free CPU or Metal
  compute carrier
- **THEN** its CompositeProvider derives and prepares supported work through
  that carrier's existing public behavior without attaching a definition or
  kernel pack to the dependency class

### Requirement: Definition and extension registration seal on first observation

One successful carrier-definition registration SHALL fix that exact class's
storage, kernel-provider interface, transfer, composite, and facet fields. A
second definition registration SHALL fail with `TypeError` and leave the first
definition unchanged.

Before the first instance is constructed or any capability query observes the
exact class, eligible kernel packs MAY be registered against its kernel
provider. The first such observation SHALL
atomically freeze the complete registered pack set together with the
definition. Definition or pack registration after observation SHALL fail with
`TypeError` and leave every prior observable answer unchanged.

For a dependent carrier, construction SHALL validate its dependencies and
obtain the complete `CompositeProvider.capabilities` iterable before exposing
the instance. Failure SHALL expose no partially configured instance. Two
instances of the same dependent class MAY derive different reach from
different dependencies while sharing one class definition.

#### Scenario: Add a pack before construction

- **WHEN** an eligible pack is registered after its target definition but
  before the target class is observed
- **THEN** the first observation freezes the provider's built-in patterns and
  the pack's patterns as one immutable execution set

#### Scenario: Reject a late pack

- **WHEN** a caller registers a pack after constructing or querying its target
  carrier class
- **THEN** registration fails with `TypeError` and capability and dispatch
  answers remain unchanged

### Requirement: Providers have disjoint responsibilities

A `StorageProvider` SHALL implement only its documented dtype, extent,
allocation, release, and physical value callbacks. A `KernelProvider` SHALL
contain one execution interface and its built-in kernel patterns. A
`TransferProvider` SHALL prepare only its route's exact directional transfer.
A `CompositeProvider` SHALL report one dependent instance's complete
capabilities and exact result Carrier roles and prepare execution that preserves an outer
`ResolvedInvocation`. A `CarrierFacet` SHALL own only its declared
carrier-specific behavior.

Framework operation meaning, dtype planning, capability derivation,
autograd-node construction, public versions, move commit, ownership checks,
provider-result validation, and result publication SHALL remain
framework-owned. A provider SHALL use `Unsupported`, a prepared-work record,
`ProviderResult`, and `ProviderCompletion` exactly as specified rather than
redefining those behaviors.

A resource used by a storage provider SHALL retain its own public type and
SHALL not thereby become a `StorageProvider` or the carrier of a Tensor.

#### Scenario: Allocate from a separate resource

- **WHEN** a storage provider uses a device resource to allocate a carrier
- **THEN** the resulting Tensor is backed by the allocated carrier while the
  resource remains a distinct object

### Requirement: Kernel execution interfaces cover JIT and precompiled execution

A `KernelExecutionInterface(version)` SHALL be an immutable compatibility base
for carrier-specific execution context. `version` SHALL be a non-empty string;
another type SHALL fail with `TypeError` and an empty value with `ValueError`.
Its public compatibility identity SHALL be `(type(interface),
interface.version)`. Concrete subclasses MAY add immutable device, stream,
addressing, workspace, or submission facilities used by kernels for that exact
carrier, but those additions SHALL not change the common identity rule.

A KernelPattern preparation callback SHALL receive the exact interface
instance owned by the carrier's KernelProvider. It MAY return a prepared
precompiled executable or JIT-compile a specialization and return it as
`PreparedKernel(interface, submit)`. The prepared record's interface identity
SHALL equal the provider interface identity or execution SHALL fail with
`TypeError` before submission. A concrete compiled specialization MAY have its
own native launch ABI; that ABI SHALL not widen the carrier classes or interface
versions accepted by its pack.

#### Scenario: Prepare a JIT specialization

- **WHEN** a matching pattern for a definition-backed custom accelerator
  receives a supported resolved invocation and its execution interface
- **THEN** it may JIT-compile and return a PreparedKernel whose submission uses
  that same interface identity

#### Scenario: Use a precompiled kernel through the same boundary

- **WHEN** a matching precompiled kernel already exists
- **THEN** preparation may select it without changing result, capability,
  completion, or interface semantics

### Requirement: A kernel pack belongs to one exact independent carrier

`KernelPack(namespace, version, carrier_class, execution_interface, patterns)`
SHALL describe one extension pack. `namespace` and `version` SHALL be non-empty
strings. `carrier_class` SHALL be an exact independent concrete `Carrier` class
with a registered definition and kernel provider. `execution_interface` SHALL
be a `KernelExecutionInterface` whose exact type and version match that
provider. `patterns` SHALL be a non-empty finite iterable of `KernelPattern`
values and SHALL be materialized once as an immutable tuple.

Construction SHALL fail with `TypeError` for a wrong namespace, version,
carrier-class, interface, pattern collection, or pattern entry type, and with
`ValueError` for an empty namespace, version, or collection or duplicate
`kernel_id` within the pack. Exact target eligibility and compatibility with
the target provider SHALL be validated by `register_kernel_pack`.

`register_kernel_pack(pack)` SHALL validate and attach the complete pack to its
exact target class and return `None`. Another `pack` type, invalid field type,
the `Carrier` root, an unrelated class, a `DependentCarrier` class, a
storage-only class, or an execution-interface mismatch SHALL fail with
`TypeError`. An empty namespace, version, or pattern collection, a duplicate
`(namespace, version)` identity, or a duplicate `kernel_id` in the complete
target execution set SHALL fail with `ValueError`.
A rejected pack SHALL contribute no pattern.

Pack attachment SHALL use exact carrier-class identity. Sharing a device
category, compiler framework, storage representation, or execution-interface
type SHALL not attach a pack to a base, subclass, sibling, or dependent
carrier. A dependent carrier that composes the target MAY derive reach through
that target but SHALL not own, copy, or register the pack.

#### Scenario: Attach a CUDA kernel pack only to CUDA

- **WHEN** a CUDA-specific CUTLASS pack names the exact independent CUDA
  carrier and matching CUDA execution interface
- **THEN** only that exact carrier's sealed kernel set includes the pack

#### Scenario: Reject a CUDA pack on Metal

- **WHEN** the same CUTLASS pack names a Metal carrier or Metal execution
  interface
- **THEN** registration fails with `TypeError` and Metal's kernel set is
  unchanged

#### Scenario: Reject a pack on a composite

- **WHEN** a pack names Evictable, TiledEvictable, or another dependent carrier
- **THEN** registration fails with `TypeError` before any pattern is attached

### Requirement: Kernel patterns are authoritative and deterministically selected

`Unsupported(reason)` SHALL be an immutable rejection value whose `reason`
SHALL be a non-empty string; invalid input SHALL fail with `TypeError` or
`ValueError`, respectively.

`KernelPattern(operation, plans, kernel_id, prepare, preference=0)` SHALL name
one executable candidate. `operation` SHALL be one registered immutable
`OperationDefinition`. `plans` SHALL be a non-empty finite iterable of exact
`OperationCapability` values for that operation. `kernel_id` SHALL be a stable
non-empty string within the target execution set. `prepare` SHALL be callable
with `(invocation: ResolvedInvocation, interface:
KernelExecutionInterface)` and return `PreparedKernel` or `Unsupported`.
`preference` SHALL be a signed integer other than `bool` and SHALL default to
`0`. Invalid field or entry types SHALL fail with `TypeError`; empty plans or
identifier, duplicate plans, or a plan whose operation string differs from
`operation.name` SHALL fail with `ValueError`.

At runtime, a pattern SHALL match only when its `operation` is the invocation's
definition and one of its plans exactly matches `invocation.plan`. Matching
candidates SHALL prepare in descending preference, then built-in pattern order,
then sorted pack `(namespace, version)` identity and pack pattern order.
Returning `Unsupported` SHALL cause the next candidate to prepare and SHALL
perform no provider submission, public mutation, result publication, or version
change. Another preparation exception SHALL be terminal and SHALL not trigger
fallback. If every candidate returns `Unsupported`, execution SHALL raise
`UnsupportedOperationPlan` containing the ordered rejection reasons.

Capability enumeration SHALL be exactly the immutable duplicate-free union of
the plans in the provider's built-in patterns and frozen packs. A successful
PreparedKernel SHALL come from the exact pattern whose plan authorized the
invocation.

#### Scenario: Prefer an optimized kernel with fallback

- **WHEN** an optimized pattern has higher preference but rejects one runtime
  alignment and a general matching pattern accepts it
- **THEN** the general pattern prepares and executes without changing static
  capability introspection

#### Scenario: Break equal-preference ties deterministically

- **WHEN** two equal-preference patterns statically match one invocation
- **THEN** built-in and sorted pack identity plus pattern order selects one
  stable preparation order

#### Scenario: Treat a preparation exception as terminal

- **WHEN** the first matching pattern raises an exception instead of returning
  `Unsupported`
- **THEN** execution propagates that exception without preparing a lower
  preference fallback or publishing a result

### Requirement: Carrier-specific facets are typed and semantically isolated

`CarrierFacet` SHALL be an open marker base class for immutable
carrier-specific providers. `carrier.facet(facet_type)` SHALL require an exact
`CarrierFacet` subclass and return the facet instance registered under that
exact type, or `None` when absent. `carrier.require_facet(facet_type)` SHALL
return the same instance or fail with `NotImplementedError` identifying the
missing exact facet type. Another `facet_type` SHALL fail with `TypeError`.

A facet SHALL add only its declared carrier-specific surface. It SHALL not
widen storage dtype support, kernel-pack eligibility, computational capability,
movement routes, or built-in operation meaning. Carrier-specific convenience
methods MAY delegate to a facet without adding those methods to every Carrier.

#### Scenario: Query an absent residency facet

- **WHEN** a storage-only carrier is queried for a tiled-residency facet it did
  not register
- **THEN** `facet` returns `None` and `require_facet` fails with
  `NotImplementedError`

### Requirement: Definition-free carriers retain the existing extension path

An exact Carrier class with no registered definition SHALL retain the
accepted storage hooks, dispatch hook, manual capability declaration, and
movement registration behavior. An exact class with a registered definition
SHALL use only the definition-backed path and SHALL reject manual capability or
movement authority for that class.

Generic, CPU, FileBacked, Evictable, BlockDeviceCarrier, and Metal SHALL remain
definition-free and SHALL retain their existing authorities and observable
behavior. TiledEvictable SHALL be the only definition-backed shipped carrier.
Public kernel-pack registration SHALL therefore reject those definition-free
shipped independent carriers, while a custom definition-backed independent
carrier remains eligible under the exact-class rules.

#### Scenario: Run a legacy custom carrier

- **WHEN** a definition-free custom carrier supplies the legacy contracts
- **THEN** its existing dispatch, capability, and movement behavior remains
  available

#### Scenario: Preserve a shipped legacy carrier

- **WHEN** Generic, CPU, Metal, FileBacked, BlockDeviceCarrier, or Evictable is
  constructed or observed
- **THEN** it uses the same definition-free storage, dispatch, capability, and
  movement authority it used before this capability was added

#### Scenario: Reject mixed authority

- **WHEN** a definition-backed class attempts a manual capability or movement
  registration
- **THEN** registration fails without changing its definition-derived answer
