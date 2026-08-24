## Context

See `proposal.md` for motivation. StrideWeave currently has two numerical
implementations: Generic is the behavioral reference and CPU is the native
implementation. Dispatch and capabilities belong to an exact carrier class,
while local verification assumes CPU at its public entry point, classification
layer, execution helpers, Stage Two catalog, and compilation-manifest binding.

TileLang is a Python JIT whose Metal execution adapter accepts PyTorch MPS
tensors. The first implementation can therefore establish accelerator behavior
without adding native Objective-C++ storage code, but the executed artifact is
runtime-generated and cannot honestly reuse the current AOT-only provenance
identity unchanged.

The verification/provenance and Metal carrier surfaces can be developed
independently, but the delivered behavior is one coherent contract: Metal
execution is not complete until it passes through the same fail-closed staged
verification, direct v3 provenance, and v3-only evidence-tracking boundary.

## Goals / Non-Goals

**Goals:**

- Replace the prototype verification API with explicit provider-neutral profile
  selection while retaining CPU as the shipped oracle profile.
- Give every central registered computational dispatch name at least one exact
  executable Metal capability.
- Make slow and serial implementations acceptable when they are the simplest
  faithful form of current reference semantics.
- Bind each executed Metal specialization to the generated code and complete
  compilation inputs used by that case.
- Preserve fail-closed correctness while allowing compiled-executable and JIT
  providers to expose different observable compilation artifacts.

**Non-Goals:**

- Complete parity with every CPU dtype or accumulator plan.
- Direct `MTLBuffer` allocation or a non-PyTorch TileLang argument adapter.
- Kernel optimization, layout normalization, autotuning, asynchronous profiling,
  CUDA, or ROCm.
- New numerical-accuracy contracts for the currently deferred vendor
  transcendental and floating-`pow` domains.
- Provenance certification of autograd itself; ordinary tests continue to cover
  forward/backward behavior.

## Decisions

### Use explicit provider-neutral verification profiles

`verify_backend(target, *, output=None)` requires an explicit registered target
profile. Neutral profile, logical-kernel, classification, stage, and certificate
interfaces supply the facts generic algorithms need; shipped CPU compiled and
Metal/TileLang JIT profiles provide those facts without CPU/Metal branches in
the generic verifier. Prototype public names and CPU wrappers are removed rather
than retained as compatibility shims.

Alternative considered: accept an arbitrary carrier class directly. Rejected
for this change because a carrier class alone cannot provide the required kernel
manifest, classification, synchronization, or compilation provenance, and a
public plugin-registration protocol would expand the contract materially.

### Keep CPU Stage One and make Stage Two target-generic

Stage One continues to establish Generic/CPU oracle evidence. Stage Two consumes
a selected backend descriptor and enumerates that backend's complete classified
capability set. Oracle authorization is matched by operation, plan obligations,
and verification class rather than requiring the target and oracle to share a
kernel ID. Metal evidence names `metal.*` kernels while consuming the relevant
`cpu.*` certificate.

The CPU Stage Two catalog retains its correctness coverage. Metal
Stage Two covers every active Metal capability, emits explicit deferrals for
the existing vendor-math domains, and fails completeness if a registered
operation name lacks a kernel, capability, or classification.

Alternative considered: compare Metal directly with Generic in a new test
harness. Rejected because it would bypass the certificate and provenance model
the change is intended to exercise.

### Use PyTorch MPS storage as the temporary Metal allocation boundary

The closed `Metal` carrier owns one one-dimensional MPS tensor representing
physical storage. Tensor layouts and offsets remain StrideWeave metadata.
Results are allocated by the carrier and passed to TileLang as explicit output
buffers. Host reads and CPU movement synchronize before decoding values; ordinary
operation launches remain asynchronous where dependency ordering is preserved by
the runtime.

TileLang, PyTorch, and their Metal runtime support are optional dependencies.
Importing StrideWeave and using CPU must not import or require them. Metal
construction performs the lazy availability check and returns actionable errors.
The dependency versions are pinned after an ABI spike establishes one working
combination.

Alternative considered: implement Objective-C++ Metal allocation first. Deferred
to a follow-on PR because it couples storage lifetime and the TileLang argument
ABI before functional and provenance boundaries are established.

### Specialize simple kernels and use temporary address maps

The initial layout bridge materializes and caches logical-to-physical address
maps for immutable layouts, including offsets, holes, hierarchy, permutation,
and stride-zero broadcast. Kernels operate over logical coordinates and indirect
through these maps. Cache keys include the exact layout/address signature and
every operation parameter that changes generated code.

Pointwise kernels assign one logical output per thread. Reductions, scans,
matmul, convolution, scatter-add, and per-slice selection use deliberately
serial loops wherever parallel association, collisions, or tie handling could
violate the reference order. This makes coverage precede optimization.

Alternative considered: require stride sorting and layout normalization before
the first operation. Deferred because it creates a second correctness-critical
algebraic feature and is unnecessary for a rough implementation.

### Cover operation names without claiming CPU plan parity

Metal stores Float32, Int32, and Bool. It advertises every faithful Float32 plan
needed to cover the registered operation-name set plus Bool and Int32 plans
needed for predicates, conditions, indices, and value/index outputs. It omits
Float64 accumulator plans and Int32 arithmetic plans whose exact checked or wide
accumulation semantics the first kernels cannot implement.

Capabilities are still derived from the central policy and filtered through one
Metal executable-plan predicate. Dispatch name coverage is tested separately
from exact capability/execution agreement so neither property can mask the
other.

### Treat structural operations and movement as carrier infrastructure

Metal reuses the existing carrier-neutral implementations of `as_strided`,
`broadcast_to`, `permute`, `rearrange`, `reshape`, `squeeze`, `unsqueeze`, and
`view`. It registers exact-class CPU-to-Metal, Metal-to-CPU, and Metal-to-Metal
bulk moves. Movement copies the physical `cosize` span and preserves the Tensor
layout instead of materializing logical values.

### Introduce provenance-complete v3 for mixed AOT and JIT evidence

Version v3 cleanly replaces the prototype format and represents a deterministic
compilation bundle containing CPU compiled-executable
receipts and runtime TileLang JIT receipts. A logical Metal kernel and variant
has one classification identity; each compile-time specialization has a distinct
receipt containing:

- the complete specialization key;
- provider, target, toolchain, framework, OS, and relevant runtime versions;
- normalized compile options;
- the operation-owned source and complete content-addressed closure;
- generated host and Metal device source digests;
- the runtime artifact digest when the provider exposes it.

Evidence points to the exact primary specialization receipt and separately
relates every case-local supporting specialization, such as index validation. A
pre-compilation error has no receipt and report binding never borrows one from a
later case. Loading remains offline and validates only report bytes. Online
recording independently regenerates current JIT artifact facts from a
provider-owned specialization recipe that is reconstructable in a later process
without retaining invocation tensors or launching the computational kernel; it
does not treat incoming generated-artifact fields as their own baseline. Canonical
axes are transport data rather than trusted runtime inputs: recording first decodes
them into immutable family-typed recipes that validate exact fields, types, plans,
address cardinalities, and dimensional relationships before address/JIT caches or
the optional TileLang runtime are consulted.
Fresh stores use v3; v2 reports, stores, and
snapshots are rejected before mutation with actionable new-store or manual
recreation guidance. Nothing migrates, wraps, rewrites, preserves, or deletes v2
state automatically.

Alternative considered: identify every shape as a logical kernel variant.
Rejected because it makes the classification set unbounded and conflates
semantic coverage with compilation specialization.

### Replace temporary matrix coverage only with equivalent evidence

Temporary ordinary Metal tests may stabilize
storage, dispatch, kernels, movement, and autograd without depending on the
unfinished generic interface. During integration, exhaustive duplicated target
matrix tests are removed only after equivalent provenance-backed cases exist.
Focused carrier, error-contract, dispatch freshness, and autograd unit tests
remain because kernel evidence does not replace API tests.

## Invariant Preservation

- `RT001`, `RT011`: Metal dispatches through its exact carrier instance and
  returns a fresh Metal-owned visible operation; TileLang remains below that
  boundary.
- `RT002`, `RT004`: physical allocation and movement use `layout.cosize`, and
  factories preserve or explicitly override dtype, mutability, initialization,
  and freshness.
- `RT003`, `RT007`, `RT022`: public Metal mutation advances carrier versions,
  saved inputs retain versions, and graph release remains owned by the existing
  Operation/Tensor machinery.
- `RT009`, `RT020`: exact-class moves preserve dtype, layout, physical holes,
  values, failure atomicity, and gradient movement.
- `RT012`: the accepted-storage table is revised to add Metal Float32, Int32,
  and Bool while Float64 remains accumulator-only.
- `RT013`: Metal resolves central plans, advertises exact filtered capabilities,
  honors pinned accumulation order, and refuses unsupported shapes before
  allocation or JIT compilation.
- `RT016`, `RT017`: shared alignment remains authoritative, outputs and
  gradients allocate injective layouts, and address maps preserve stride-zero
  input semantics.
- `RT023`: classification completeness becomes backend-selectable, CPU Stage One
  remains the certificate authority, Metal Stage Two is fail-closed, and every
  attempted Metal plan receives passed, failed, error, blocked, or deferred
  evidence.
- `RT024`: compiled and JIT receipts become active immutable provenance facts;
  persistence continues to store facts rather than confidence judgments and
  `verify_backend` remains local and network-free.

`llms.md` must change in the sections identified by `proposal.md`.
`INVARIANTS.md` must revise `RT007`, `RT012`, `RT023` including `RT023e`, and
`RT024` and add a Metal/JIT
invariant if implementation reveals a separate canonical synchronization or
receipt-boundary rule rather than an instance of those existing invariants.

## Risks / Trade-offs

- **[TileLang Metal lacks a required primitive or dtype]** → The ABI spike tests
  every required storage dtype and representative kernel family before the
  Metal branch expands; missing high-level primitives are replaced with simple
  scalar formulas or serial loops, and an operation is not advertised until it
  executes faithfully.
- **[PyTorch MPS behavior leaks into the public carrier contract]** → Keep the
  owned tensor private, expose only Carrier semantics, and treat direct Metal
  allocation as an implementation replacement in a follow-on change.
- **[Asynchronous errors are recorded as passes]** → The backend descriptor owns
  an explicit synchronization hook used before decoding or classifying every
  verification result.
- **[Address maps consume excessive memory]** → Use small deterministic evidence
  shapes and bounded ordinary tests; normalization and generated index arithmetic
  remain follow-on optimizations.
- **[JIT provenance cannot recover one provider fact]** → Fail closed and report
  the missing fact rather than emitting an executable Metal evidence row without
  a complete receipt.
- **[Verification and Metal surfaces drift across shared model types]** → Freeze
  observable profile and v3 receipt contracts before integration and keep
  temporary Metal tests independent of verification internals.
- **[A v2 store is accidentally mutated as v3]** → Detect its schema before any
  mutation, fail with actionable new-store guidance, and exercise record, query,
  publication, refresh, and cross-install loading on fresh v3 stores.

## Delivery Plan

1. Implement the neutral verification/provenance surface and Metal carrier
   surface against the same reviewed contracts.
2. Connect Metal Stage Two, synchronization, JIT receipts, direct v3 report
   binding, and fresh v3-only tracking.
3. Replace superseded temporary operation-matrix tests with provenance-backed
   Metal coverage while retaining focused API and autograd tests.
4. Update public stubs, documentation, invariants, and optional dependency
   guidance.
5. Run OpenSpec validation, CPU regression tests, pure provenance/store tests,
   ordinary Metal tests, provenance-backed Metal verification, Ruff, pyright,
   duplication checks, and applicable native gates.
6. Submit the complete reviewed change through one pull request.

Rollback is removal or reversion before merge. CPU-only imports remain usable
without optional Metal dependencies. V2 verification reports and stores are not
runtime-migrated; callers use the new API and a fresh v3 store.
