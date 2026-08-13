---
title: Kernel Evidence Tracking
publish: true
status: stable
order: 90
summary: Provenance-complete recording, local factual queries, and explicit contributor exchange for immutable kernel evidence.
---

# kernel-evidence-tracking Specification

## Purpose

Define how provenance-complete kernel verification reports become immutable
local facts, how callers inspect those facts, and how contributors explicitly
exchange validated snapshots.

## Terminology

| Term | Meaning |
| --- | --- |
| tracking report | A provenance-complete v2 verification report accepted by `VerificationReport.load`, carrying the compilation, verification, tolerance, oracle, certificate, case, and outcome identities required for tracking. Tracking consumes those identities as one validated input and does not redefine their verification semantics. |
| compilation receipt | The content-addressed provenance record for one exact kernel and variant, binding its provider, target, toolchain, compile invocation, compiled object, operation-owned source closure, and shared artifact provenance. |
| source closure | The complete ordered transitive input set reported for one compilation, including the operation-owned source and every project, generated, external-user, SDK, standard-library, compiler, and build input selected by that action. |
| current baseline | The installed build's locally constructed compilation manifest, verification specification, tolerance policies, oracle references, and required evidence cases. |
| evidence | One immutable raw verification outcome and its exact requirement, report, compilation, tolerance, oracle, payload, and certificate relationships. |
| observation | One producer's immutable provenance for retaining an evidence fact, including its producer identity and optional source-revision and artifact metadata. |
| direct target | A run whose execution target and represented target have the same exact target identity. |
| local store | The append-oriented local state containing targets, toolchains, source closures, receipts, verification specifications, tolerance policies, oracle references, runs, evidence, and producer observations. |
| contributor snapshot | One producer namespace's canonical, content-addressed current relationship graph of immutable tracking facts and observations. |

## Boundaries

This baseline captures direct-target evidence from the current native C++
provider. It does not define a compiler or build-generator support matrix,
proxy-target ingestion, JIT-framework providers, confidence or risk policy,
autotuning, CI integration, historical snapshot retention, or network exchange
transports. Those areas require separate behavior-change decisions. Status and
todo results remain factual and unranked within this capability.

## Requirements

### Requirement: The installed compilation manifest identifies every native kernel build

`load_compilation_manifest()` SHALL return an immutable
`CompilationManifest` whose receipts cover exactly the installed native kernel
and variant metadata. Each receipt SHALL bind the provider, exact target,
toolchain, kernel identity and variant, compile invocation, compiled object,
operation-owned source closure, and shared artifact provenance. The receipt and
closure identities SHALL determine per-kernel invalidation, while the shared
artifact digest SHALL remain provenance for the containing extension.

The manifest and every nested target, toolchain, invocation, closure, and
receipt identity SHALL be reconstructed from canonical content. A semantic
schema, identity, coverage, or uniqueness failure SHALL raise `ValueError`.
An absent installed manifest resource or malformed JSON resource SHALL raise
`RuntimeError`. Source-revision metadata SHALL contribute to observation
identity rather than compilation identity.

#### Scenario: Change one operation-owned source

- **WHEN** one operation-owned source changes and every other compilation input
  remains identical
- **THEN** the receipts using that source receive a different closure and
  receipt identity
- **AND** receipts for unaffected sources retain their identities

#### Scenario: Reject semantic manifest corruption

- **WHEN** the installed manifest parses as JSON but has a mismatched manifest,
  target, toolchain, invocation, or closure identity
- **THEN** `load_compilation_manifest()` raises `ValueError`

#### Scenario: Reject an unavailable installed manifest

- **WHEN** the installed manifest resource is absent or is not valid JSON
- **THEN** `load_compilation_manifest()` raises `RuntimeError`

### Requirement: Source closures retain complete portable identity and actionable diagnostics

Each source closure SHALL bind the complete ordered transitive input set
reported for the exact compilation action. Project and generated inputs SHALL
use stable provider-relative identifiers. External inputs SHALL use path-free
content-addressed identifiers, so the same content has portable identity across
contributor installations.

The closure identity SHALL cover every input. Per-member diagnostics SHALL
retain project-owned sources and headers, generated headers, and build inputs
at their positions in the complete sequence. An external-input change SHALL be
observable through the complete closure identity; project and build inputs
SHALL additionally provide individual changed-member diagnostics.

#### Scenario: Track a project-header change

- **WHEN** one project header changes between a stored receipt and the current
  baseline
- **THEN** stale reports a source-closure identity difference
- **AND** identifies that project header as changed

#### Scenario: Track an external-header change portably

- **WHEN** only an external header changes between a stored receipt and the
  current baseline
- **THEN** stale reports a source-closure identity difference from its complete
  content-addressed identity

### Requirement: Record admits only reports bound to the current baseline

`strideweave-kernel-status record` SHALL load the required `--report PATH` as a
tracking report and SHALL require `--producer PRODUCER` as a non-empty producer
identity. Before initializing the selected local store, it SHALL reconcile the
report's compilation manifest, receipts, verification specification, tolerance
policies, oracle references, and embedded Stage One certificates with the
current baseline. A current and internally consistent report SHALL proceed to
one atomic persistence operation. A stale, incomplete, inconsistent, forged,
or malformed report SHALL terminate with process status 2 before local state is
created or changed.

The command SHALL accept these optional observation inputs:

- `--source-commit REVISION` names a non-empty source revision of at most 255
  characters and SHALL default to absent;
- `--artifact LOCATOR` names a non-empty retained report or artifact locator of
  at most 1024 characters and SHALL default to absent;
- `--artifact-digest SHA256` names a 64-character lowercase SHA-256 artifact
  digest and SHALL default to absent.

An invalid observation input SHALL terminate with process status 2 before
persistence.

#### Scenario: Record a current report

- **WHEN** the report agrees with the current baseline and every observation
  input is valid
- **THEN** record atomically persists every report outcome and one producer
  observation per outcome
- **AND** returns process status 0

#### Scenario: Reject stale report provenance

- **WHEN** the report's compilation or verification identities differ from the
  current baseline
- **THEN** record returns process status 2 before the selected local store is
  initialized

### Requirement: Persistence retains immutable direct-target facts

Each accepted report SHALL be stored as one direct-target run whose execution
and represented target identities are equal. Persistence SHALL retain every raw
outcome in the report, including passed, failed, errored, blocked, and deferred
evidence, together with its exact compilation, requirement, tolerance, oracle,
payload, and certificate relationships.

Run, evidence, and observation identities SHALL be derived from their canonical
facts. Repeating the same report and observation provenance SHALL resolve to
the same identities and facts. A different producer, source revision, artifact
locator, or artifact digest SHALL resolve to a distinct coexisting observation
over the same evidence. Status SHALL return contradictory observations as
independent facts, regardless of their recording times.

#### Scenario: Repeat identical ingestion

- **WHEN** the same report and observation provenance are recorded twice
- **THEN** both commands resolve to the same run, evidence, and observation
  identities
- **AND** the stored factual counts remain unchanged

#### Scenario: Retain contradictory observations

- **WHEN** different producers record different outcomes for one case
- **THEN** both observations coexist and status returns each one independently

### Requirement: Local state is lazy, isolated, compatible, and atomic

The default store location SHALL be the platform application-data base followed
by `strideweave/kernel-evidence`: `~/Library/Application Support` on macOS,
`${XDG_DATA_HOME:-~/.local/share}` on Linux, and `%LOCALAPPDATA%` on Windows
with `~/AppData/Local` as its fallback. `STRIDEWEAVE_STATUS_HOME` SHALL replace
the platform base while preserving the stable suffix. Every command SHALL
accept `--store PATH` as an optional complete location override and SHALL
default to that platform location.

Resolving a default or overridden path SHALL only select local state.
Initialization SHALL occur on the first explicit persistence or query
operation. Distinct selected store paths SHALL keep their facts isolated. A
compatible earlier store SHALL be upgraded atomically while preserving its
facts. An incompatible runtime or schema history SHALL terminate the command
with process status 2 before partially applying an operation.

Every recording and refresh SHALL commit all of its facts atomically. A failure
SHALL leave the previously committed state intact.

#### Scenario: Inspect command help with an unused store path

- **WHEN** a caller invokes a help path while a store override names a path that
  does not exist
- **THEN** help returns process status 0 and that path remains uninitialized

#### Scenario: Fail during an atomic write

- **WHEN** one fact in a recording or refresh operation cannot be persisted
- **THEN** the operation returns process status 2
- **AND** the store retains exactly its previously committed facts

### Requirement: Status returns every matching factual observation

`strideweave-kernel-status status` SHALL require `--arch ARCH` as the exact
case-sensitive architecture token. It SHALL accept `--kernel KERNEL`,
`--variant VARIANT`, `--class CLASS`, and `--producer PRODUCER` as optional
exact non-empty filters, each defaulting to absent. The command SHALL return
every observation matching the architecture and all supplied filters in stable
kernel, variant, class, case, producer, and observation-identity order.

In text mode, status SHALL print the architecture and total followed by one
line per observation containing its kernel, variant, class, case, outcome,
producer, run, and observation identities. With `--json`, it SHALL return a
compact object containing `architecture`, the ordered `observations` array, and
`total`. Each JSON observation SHALL expose its evidence, target, compilation,
verification, tolerance, oracle, certificate, outcome, producer,
source-revision, recording-time, and artifact facts.

#### Scenario: Filter contradictory observations

- **WHEN** two producers recorded different outcomes for one case and one
  producer filter is supplied
- **THEN** status returns only that producer's matching observation

#### Scenario: Return all factual observations as JSON

- **WHEN** status receives only the required architecture and `--json`
- **THEN** it returns every matching observation in a deterministically ordered
  JSON array

### Requirement: Stale explains current-baseline identity differences independently

`strideweave-kernel-status stale` SHALL require `--arch ARCH` as the exact
case-sensitive installed-target architecture. It SHALL construct the current
baseline locally and compare every stored run for that architecture.
Compilation manifest, receipt, source closure, toolchain, target, verification
specification, tolerance policy, and oracle differences SHALL be reported as
independent axes. Project-owned and generated changed members SHALL appear as
source-closure details when available. A current run SHALL be identified with
an empty difference set.

In text mode, stale SHALL print the architecture and run count followed by each
run's current identity or individual difference lines. With `--json`, it SHALL
return a compact object containing `architecture`, the ordered `runs` array,
and `total`. A requested architecture different from the installed target or a
failure to construct the baseline SHALL terminate with process status 2. A
successful stale query SHALL leave the stored facts unchanged.

#### Scenario: Explain several changed axes

- **WHEN** a stored run differs from the current baseline in target, toolchain,
  closure, specification, tolerance, and oracle identities
- **THEN** stale returns each changed axis independently
- **AND** includes actionable project-member details where available

#### Scenario: Report a current run

- **WHEN** every stored identity matches the current baseline
- **THEN** stale identifies that run with an empty difference set

### Requirement: Todo returns the stable unranked current requirement difference

`strideweave-kernel-status todo` SHALL require `--arch ARCH` as the exact
case-sensitive installed-target architecture. It SHALL construct the current
baseline locally and return each current verification requirement lacking a
matching observation under the current manifest, verification specification,
and target identities. Results SHALL be sorted by kernel, variant,
verification class, and case.

In text mode, todo SHALL print the architecture and missing count followed by
one line per missing requirement containing its kernel, variant, class, case,
and requirement identities. With `--json`, it SHALL return a compact object
containing `architecture`, the ordered `missing` array, and `total`; each JSON
missing-requirement object SHALL additionally expose the operation identity. A
requested architecture different from the installed target or a failure to
construct the baseline SHALL terminate with process status 2. A successful
todo query SHALL leave the stored facts unchanged.

#### Scenario: Query an empty local store

- **WHEN** the selected store has no observations matching the current baseline
- **THEN** todo returns every current requirement in deterministic order

#### Scenario: Query complete current evidence

- **WHEN** every current requirement has a matching observation
- **THEN** todo returns an empty result with total zero

### Requirement: Publication atomically replaces one producer's current snapshot

Passing `--publish` to record SHALL, after successful local persistence,
publish the named producer's contributor snapshot. `--publish` SHALL be an
optional flag defaulting to false. `--publish-destination ENDPOINT` SHALL be an
optional scheme-free local path string or local `file:` endpoint and SHALL default to
`STRIDEWEAVE_STATUS_PUBLISH_DESTINATION`; supplying it while `--publish` is
false SHALL terminate with process status 2. A true `--publish` with no explicit
or configured destination SHALL also terminate with process status 2.

The destination SHALL assign the producer an opaque namespace containing one
canonical content-addressed current snapshot. That snapshot SHALL contain
exactly the producer's observations and the complete immutable relationship
graph they require. Publishing changed facts SHALL atomically replace that
producer's current snapshot. Repeating an unchanged publication SHALL resolve
to the same snapshot digest and bytes.

In text mode, record SHALL append the snapshot digest and destination to its
local recording result. With `--json`, its result SHALL add a `publication`
object containing `producer_id`, `snapshot_digest`, `observation_count`, and
`destination`.

#### Scenario: Publish two contributors

- **WHEN** two producers publish to one destination
- **THEN** each producer receives a distinct opaque namespace containing its
  canonical current snapshot

#### Scenario: Repeat an unchanged publication

- **WHEN** a producer republishes unchanged observations
- **THEN** publication returns the same snapshot identity and canonical bytes

### Requirement: Refresh validates every current snapshot before one atomic merge

Passing `--refresh` to status SHALL refresh contributor snapshots before
running the factual query. `--refresh` SHALL be an optional flag defaulting to
false. `--read-source ENDPOINT` SHALL be an optional scheme-free local path
string or local `file:` endpoint and SHALL default to
`STRIDEWEAVE_STATUS_READ_SOURCE`;
supplying it while `--refresh` is false SHALL terminate with process status 2.
A true `--refresh` with no explicit or configured source, or with no contributor
current snapshots, SHALL also terminate with process status 2.

Before initializing or changing the selected local store, refresh SHALL
validate every snapshot's schema, canonical encoding, producer namespace,
snapshot and contained fact identities, complete relationship graph, embedded
tracking reports, evidence, and observation provenance. Valid snapshots SHALL
be merged in one atomic operation. An exact existing fact SHALL resolve
idempotently. An existing content identity with different immutable fields
SHALL terminate the refresh with process status 2 and preserve the previously
committed local state.

In text mode, refreshed status SHALL prepend the source and snapshot count to
the ordinary status result. With `--json`, its result SHALL add a `refresh`
object containing `read_source`, `snapshot_count`, and `observation_count`.

#### Scenario: Repeat a valid refresh

- **WHEN** the same contributor snapshots are refreshed twice
- **THEN** both refreshes report the same snapshot and observation counts
- **AND** local factual counts remain unchanged on the second refresh

#### Scenario: Reject an inconsistent snapshot atomically

- **WHEN** a snapshot has a valid outer digest but an inconsistent embedded
  evidence identity or relationship
- **THEN** refresh returns process status 2 before changing the selected local
  store

### Requirement: Exchange access occurs only through explicit local publication or refresh

The supported exchange endpoint set SHALL be exactly scheme-free local path
strings and `file:` endpoints whose host is empty or `localhost`. A path string
with a URI-scheme prefix, another URI scheme, or a non-local `file:` host SHALL
terminate publication or refresh with process status 2.

An ordinary record command SHALL complete after current-baseline validation and
local persistence. A record command with `--publish` SHALL additionally access
its selected publication destination. An ordinary status command SHALL query
the selected local store. A status command with `--refresh` SHALL first access
its selected read source. Stale and todo SHALL use the selected local store and
the locally constructed current baseline. Package imports, report inspection,
and backend verification SHALL remain independent of tracking storage and
exchange state.

#### Scenario: Record locally

- **WHEN** record is invoked with its required inputs and the default false
  `--publish` flag
- **THEN** it validates and persists the report using only the selected local
  store

#### Scenario: Query local status

- **WHEN** status is invoked with its required architecture and the default
  false `--refresh` flag
- **THEN** it returns observations from the selected local store

#### Scenario: Refuse an unsupported endpoint

- **WHEN** publication or refresh receives an HTTP endpoint
- **THEN** the command returns process status 2 with an unsupported-transport
  diagnostic

### Requirement: The kernel-status CLI has stable common output and failure behavior

`strideweave-kernel-status` SHALL provide exactly the `record`, `status`,
`stale`, and `todo` subcommands described above. Every subcommand SHALL accept
`--store PATH` with the platform default and `--json` as an optional flag
defaulting to false. Text mode SHALL be the default. With `--json`, each command
SHALL emit compact JSON with sorted object keys and deterministically ordered
arrays.

Successful commands SHALL return process status 0. Invalid arguments, input
files, observation provenance, architecture, current baseline, local state, or
exchange data SHALL return process status 2 with an actionable diagnostic on
standard error. The root command and every subcommand `--help` path SHALL return
process status 0 and describe all inputs, defaults, examples, and exit behavior
while leaving configured local and exchange paths uninitialized.

Record text SHALL identify the run, evidence count, observation count, and
selected store. Record JSON SHALL contain `run_id`, `report_digest`,
`evidence_count`, `observation_count`, and `store`, plus the conditional
publication object defined above. Status, stale, and todo outputs SHALL follow
their command-specific result contracts.

#### Scenario: Inspect every help path

- **WHEN** a caller invokes root or subcommand `--help`
- **THEN** the command returns process status 0 with inputs, defaults, examples,
  and exit behavior
- **AND** configured local and exchange paths remain uninitialized

#### Scenario: Report invalid command usage

- **WHEN** a required input is absent or an option interaction is invalid
- **THEN** the command returns process status 2 with an actionable diagnostic
  on standard error
