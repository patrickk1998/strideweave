# Staged Kernel Correctness And Evidence Taxonomy

Version: v0.2 (raw-evidence persistence contract)

This document defines the correctness and persistence boundary used by
StrideWeave's staged kernel verification. `test_backend` remains an offline,
deterministic correctness runner: it does not open a database, inspect source
control, publish, or mutate verification status. Recording a completed report is
a separate explicit operation.

The persistence layer stores raw, provenance-complete facts. It does not assign
confidence, rank evidence classes, choose a preferred producer, or collapse
contradictory observations. Bit-exact and tolerance-based evidence are distinct
test classes, not rungs in a quality ladder. A future confidence policy may
interpret the raw facts, but its version and conclusions are outside this
schema and this implementation.

## 1. Encoded-input boundary

A case converts source values into the declared kernel storage encoding exactly
once. The resulting immutable 32-bit words are the case input. Target and oracle
both decode those same words, and evidence records their identical canonical
hashes. Consequently the comparison measures kernel behavior after storage
encoding; it never charges either implementation for source quantization error.

Float64 is currently an accumulator descriptor, not a carrier storage encoding.
An oracle may widen a Float32 value only after loading the already encoded term.

## 2. Manifest and plan classification

Native CPU kernels declare a stable kernel ID and variant in their owning
translation unit. Classification is keyed by that identity and expanded over
the exact executable capability plans. Completeness is fail-closed: unknown,
duplicate, missing, or stale entries are errors, as is an executable plan with no
classification.

The v0 classes are:

- `bit_exact`: representation-preserving movement and structural operations;
- `exact_arithmetic`: finite algebraic operations and exact-integer pow plans;
- `structural`: reductions whose generated integer-valued partials are exactly
  representable in the selected floating format;
- `analytic`: independent small witnesses with known answers;
- `numerical`: reductions checked against an explicit numerical envelope;
- `deferred`: a visible disposition for floating pow and vendor-transcendental
  operations, never a silent pass.

Reduce and matmul plans participate in structural, analytic, and numerical
classes. Python-backed view, permute, rearrange, broadcast, and movement subjects
are classified separately from the native numerical manifest.

## 3. Deterministic payload families

Generators accept explicit seeds and have no global random state. The foundation
provides arbitrary Float32 bit patterns, wide finite exponents, adversarial bits
(signed zero, subnormals, infinities, and distinct quiet-NaN payloads), exact
structural integers, and fixed analytic cases.

For precision `p`, contraction length `K`, and `r` multiplicative factors per
term, the structural generator chooses an integer bound `B` satisfying

`K * B**r <= 2**p`.

Thus every product and every signed prefix sum is within the exactly
representable integer range. Reduction uses `r=1`; matmul uses `r=2`.

## 4. Comparisons and numerical bounds

Bit identity compares encoded words, so `+0` differs from `-0` and distinct NaN
payloads differ. Numerical evidence reports maximum absolute error, symmetric
relative error using `max(abs(expected), abs(actual))`, and ordered Float32 ULP
distance. Signed zeros have zero numerical ULP distance while remaining bitwise
different. Non-finite numeric deviations are serialized as canonical strings.

For `K` accumulated terms and unit roundoff `u`, numerical reduction evidence
uses the analytic envelope

`gamma_K * sum(abs(terms))`, where `gamma_K = K*u/(1-K*u)` and `K*u < 1`.

That is the bound for one accumulation path. Stage One compares two paths that
may associate independently, so its observable pairwise envelope is twice this
quantity. Stage Two uses the conservative v0 target-versus-certified-oracle
envelope recorded on its evidence.

Absolute error remains authoritative under cancellation; relative-to-result
error alone is never used.

## 5. Evidence, provenance, and certificates

Each attempted case emits one immutable evidence object containing stage, class,
explicit operation, kernel and variant, the exact classified plan when
applicable, dtypes, shapes, accumulator, contraction length, seed and case ID,
tolerance-policy identity, deviations, mismatch count, outcome, diagnostic, and
input-payload hashes. JSON uses sorted keys and standard finite values only. A
certificate is issued only when every required class for every active plan of
its kernel/variant has passed and no required record failed, blocked, errored,
or remained deferred.

The provenance-complete report replaces the prototype v1 evidence-only wire
format directly; v1 files are rejected rather than migrated. One strict header
binds the report schema, native manifest, complete required-coverage set,
verification specification, execution target, optional explicit target proxy,
toolchain identities, compilation receipts, tolerance policies, and oracle
references. Every case refers to those header objects by content digest. Every
Stage Two case also records the exact Stage One certificate digest it consumed.
The report is self-contained: a recorder never consults a different installed
build or changed verifier to reconstruct missing provenance.

Canonical reports exclude wall-clock timestamps, database state, producer
identity, publication state, source-control commit, and artifact location. Those
facts belong to the observation envelope added during explicit recording. This
keeps repeat execution deterministic and ensures that commit-only movement does
not alter compilation identity.

Prepared case metadata is retained when a recoverable public execution
`RuntimeError` or `ValueError` occurs: the error record still contains operation,
kernel, variant, exact plan where applicable, dtypes, logical shapes, payload
hashes, accumulator, contraction length, seed, and tolerance. Its deviations
and mismatch count are `null`, denoting unavailable measurements rather than a
successful zero. Independent witnesses continue after such an error. Manifest
and classification-completeness failures remain fail-fast.

## 6. Staged execution

Stage One exercises every active classified CPU capability plan against the
device-independent Generic reference, including widened Float64 accumulation
for `reduce_sum` and `matmul`. Coverage is driven by the native kernel manifest
and each plan's declared classes rather than by an operation-name branch, so a
kernel reaching C++ without a case leaves its certificate unissued. Every active
exact-arithmetic plan keeps its small fixed witness and additionally receives
distinct seeded, finite, storage-encoded operands with fractional values and
varied exponents; division denominators are nonzero, and a logical index operand
is drawn from the extent it addresses. The order-normative floating combiners —
`cumsum`, `conv_general`, `scatter_add`, and `reduce_prod` — use structural
payloads whose every legal partial result is exactly representable, so their
evidence checks traversal and addressing without pinning an association.
Structural payloads for `reduce_sum` and `matmul` come from the mantissa-bound
generator, analytic cases are checked independently against their literal known
answers, and numerical evidence uses a deterministic controlled wide-exponent
payload. Its order-independent pairwise envelope is twice gamma-K because both
Generic and CPU may choose a different legal association.
Floating `pow`, vendor-transcendental kernels, and autograd certification are
explicit v0 deferrals, each recorded with a concrete reason rather than a pass.
Stage Two uses ordinary CPU Float32 accumulation as the target and covers
movement, `reduce_sum`, and `matmul` only. Its deterministic
contraction catalog contains multi-output flat and hierarchical layouts with
materially different M, N, and K dimensions; evidence records the effective
matrix shape of each operand and the case-specific contraction length. Each
numerical output is bounded with its own K-dependent gamma envelope using that
output's sum of term magnitudes. It may execute a `reduce_sum` or `matmul` case only
when the matching Stage One oracle certificate is present; missing certification
produces deterministic blocked evidence and no target execution.

Report validation treats a complete embedded Stage One evidence set as the
independent source for its certificate. It derives the kernel and variant,
required class union, per-capability-plan class coverage, and evidence digest,
then requires the reconstructed certificate and every Stage Two reference to
match exactly. This catches stale construction, wrong record selection, and
ordinary implementation wiring mistakes; it is not a signature, an
authenticity boundary, or a producer-trust protocol. A filtered report may
retain a consumed certificate while carrying no Stage One rows, or only a class-
or plan-filtered subset. Such a view validates the available coverage and keeps
the already validated source certificate, but necessarily cannot recompute the
digest of omitted evidence.

The local `test_backend(output=None)` entry point reruns both stages and may
write their combined deterministic report. It remains offline and mutation-free
regardless of whether a local evidence store exists. Hypothesis generation,
empirical tolerance calibration, and quantizer validation remain outside the
verifier.

## 7. Compilation receipts and build identity

A compilation receipt is a framework-neutral, immutable description of one
kernel build. Its identity covers:

- stable kernel ID, variant, and public operation;
- provider and receipt-schema identities;
- exact execution target and toolchain descriptors;
- per-source compiled-object digest and compilation-invocation digest;
- one ordered source closure containing the operation-owned source, every
  transitive included header, generated source or build input, relevant
  definition, and flag; and
- provider-specific framework and specialization metadata, which are empty for
  the current C++/CMake provider.

Closure inputs are named by stable project-relative or provider-defined URIs and
content digests. The C++ provider requests the compiler's complete dependency
set, including project, generated, external-user, SDK, standard-library, and
compiler headers. Build-tree inputs use build-relative provider URIs; inputs
outside the project and build trees use path-free content-addressed C++ provider
URIs, so installation prefixes and private local paths do not enter reports. A
source or transitive-header change therefore changes only the closures containing
it; a flag or toolchain change changes the corresponding build identity; a Git
commit change alone does not. The shared extension digest is retained at the
manifest/report boundary as artifact provenance but is not the per-kernel
invalidation key, because unrelated operation kernels share that extension.

The Generic oracle reference begins from a reviewed set of execution,
comparison, payload, and Stage One Python roots and derives their complete local
static-import closure. Shared implementation modules such as Generic scalar
helpers, operation policy, operation capabilities, and alignment helpers are
therefore identity inputs without relying on an ad hoc leaf-file list. External
Python distributions remain part of the declared runtime/toolchain environment,
not copied into this source closure.

Only native C++ FP32 kernel receipts are generated now. The same receipt can
later describe CuTe DSL, Triton, or TileLang JIT compilation by naming a
different provider, recording generated sources and runtime compiler inputs in
the closure, and populating framework/specialization metadata. No JIT adapter or
dependency on those frameworks is part of the current scope.

The current C++ provider is narrower than the framework-neutral receipt model.
It supports CMake compiler IDs `GNU`, `Clang`, and `AppleClang` when the compiler
is invoked in GNU-compatible driver mode and accepts `-M`, `--version`, and
`-dumpmachine`. It also requires a Makefile- or Ninja-family CMake generator
that actually writes `compile_commands.json`. The MSVC driver and Visual Studio
generators are therefore unsupported for source and editable builds in this
version; adding their different dependency-discovery and compiler-identity
protocol is deferred to a future provider. A supported prebuilt wheel carries
the generated manifest, so installing that wheel does not exercise this
provider and leaves `test_backend` and explicit local status-store operations
available on the wheel's supported platforms.

## 8. Immutable persistence schema

The checked-in MySQL-compatible Dolt migration at
`src/strideweave/verification/store/migrations/0001_raw_evidence.sql` is the
normative storage shape. Canonical JSON stored in that schema is validated
before insertion and hashed with SHA-256; every 64-character primary key is the
digest of the complete canonical fact represented by its row.

The tables separate target descriptors and explicit proxy statements,
toolchains, source closures and ordered closure inputs, kernel builds,
verification specifications and their requirements, tolerance policies, oracle
references, deterministic runs, per-case evidence, and producer observations.
A proxy always names both the target that actually executed and the target it
represents. It is displayed as a fact and never silently treated as direct
hardware evidence.

Runs and evidence contain only report facts. An observation links one evidence
row to producer identity, optional source commit, recording time, and optional
artifact locator/digest. Observation identity is derived from producer,
evidence, source commit, and artifact identity rather than recording time, so
recording the same report twice is idempotent; genuinely different evidence or
observation provenance remains distinct. Conflicting outcomes coexist because
neither a semantic requirement nor a producer name is a mutable unique key.

The schema uses no auto-increment identity and no mutable "current" status row.
Independent contributors can insert content-addressed facts and producer-scoped
observations on separate internal branches. Identical rows converge by identity;
different rows remain available for factual queries. Application code inserts a
complete report in one transaction and never updates or deletes evidence facts.

## 9. Local-store lifecycle

The evidence database is separate from the source repository, its Git history,
and Beads. The default location is the platform application-data directory under
StrideWeave's stable project identity; `STRIDEWEAVE_STATUS_HOME` is the explicit
test and advanced-user override. Importing StrideWeave, asking any root command
or subcommand for `--help`, and running `test_backend` do not locate, create, or
migrate a store.

The first command that needs persistence creates the local store and applies the
ordered checked-in migrations in one transaction. `schema_migrations` records
the version, filename, checksum, and application time. Startup validates every
applied checksum and refuses unknown, missing, reordered, or modified migration
history. It also refuses a location inside a Beads database. Repeated
initialization and ingestion are idempotent, and a failed migration or report
recording leaves no partial facts.

The storage adapter is internal and backend-neutral. Its ordinary vocabulary is
"verification store"; Dolt processes, repositories, branches, and remotes are
an implementation detail. A supported local Dolt executable is invoked through
one argument-safe adapter with explicit transactions. No long-running server,
DoltHub account, central service, network, or source-tree database is required.

## 10. Factual queries and publication

`status` is an observation inventory: it reports matching runs, classes,
outcomes, deviations, tolerances, producers, artifacts, direct/proxy target
facts, and exact identities. It does not resolve contradictions by recency.
`stale` compares recorded identities with the currently installed manifest and
explains differences independently for source closure inputs, toolchain,
target, verification specification, tolerance policy, and oracle. `todo` is the
deterministically sorted set difference between current manifest requirements
and matching recorded evidence. It is unranked and makes no scarce-hardware or
risk-priority claim. All three are offline and read-only.

Optional sharing is expressed through backend-neutral `read_source` and
`publish_destination` configuration. Publication uses internal contributor
namespaces derived from an explicit producer identity, preserves append-only
observations, and surfaces conflicts. Each namespace holds one atomically
replaced current content-addressed snapshot. Publication selects only that
producer's relationship graph in SQL; refresh reads only current snapshots and
looks up only incoming primary keys in the destination, so neither side scans
unrelated or historical evidence. Refresh validates the contributor
namespace and snapshot digest, canonical encodings and their content identities,
the complete relationship graph, embedded reports and evidence, and observation
provenance before initializing or changing the destination. Exact rows are
idempotent by explicit comparison; new rows use strict inserts in one atomic
transaction. Only an explicit `record --publish` or `status --refresh` may use
the network. Users never need database init, clone, branch, remote, pull, or push
commands, and no provider such as DoltHub, GitHub, or a central server is
hard-coded.

## 11. Explicitly deferred scope

The following are not represented as implemented behavior or schema tables:

- a confidence lattice, evidence ranking, semantic prioritization, or manual
  verification levels;
- autotuning measurements or caches;
- actual CuTe DSL, Triton, or TileLang adapters;
- assembly inspection or sanitizer evidence levels; and
- CI recording, publication, or status integration.

Those features require later versioned policy or schema work. Raw evidence must
remain usable without adopting any of them.
