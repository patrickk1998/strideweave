## MODIFIED Requirements

### Requirement: Local state is lazy, isolated, compatible, and atomic

The default store location SHALL remain the platform application-data base
followed by `strideweave/kernel-evidence`. `STRIDEWEAVE_STATUS_HOME` SHALL
replace only that base, and every command SHALL continue accepting optional
`--store PATH` as a complete override.

Path resolution and every help path SHALL leave an absent store uninitialized.
The first explicit persistence or query SHALL initialize a fresh schema-v3
store, and distinct paths SHALL remain isolated. Every recording and refresh
SHALL commit atomically; failure SHALL retain exactly the prior committed state.

A store whose schema is v2 SHALL be detected before migrations, table changes,
or any other mutation. Every command against it SHALL return process status 2
with an actionable diagnostic requiring a new store path or an explicit manual
recreation after the user has handled existing data. The system SHALL not
migrate, rewrite, delete, or preserve v2 rows as v3 facts automatically. An
unknown or corrupt schema SHALL likewise fail before mutation.

#### Scenario: Inspect command help with an unused store path

- **WHEN** help is invoked while `--store` names an absent path
- **THEN** it returns status 0 and leaves that path uninitialized

#### Scenario: Fail during an atomic write

- **WHEN** one fact in a recording or refresh cannot be persisted
- **THEN** status 2 is returned and prior facts remain unchanged

#### Scenario: Reject a v2 store before mutation

- **WHEN** any command opens a schema-v2 store
- **THEN** it returns status 2 before mutation and explains how to select a new
  store or recreate the old path manually

## ADDED Requirements

### Requirement: Tracking stores schema-v3 selected-target facts directly

`record` SHALL accept only a strictly loaded schema-v3 report and reconcile its
selected target profile, CPU oracle dependencies, compilation bundle, exact
compiled-executable and JIT-specialization receipts, verification requirements,
tolerances, certificates, and evidence graph against the current installed
baseline before store initialization. A v2 report or any stale, incomplete,
forged, or inconsistent v3 graph SHALL return status 2 before mutation.

One verification run SHALL have exactly one selected target profile. CPU oracle
certificates, evidence, and receipts used by another target SHALL be stored as
dependencies of that selected-target run, not as a second target run. Immutable
content identities and producer observations SHALL coexist append-orientedly;
exact repetition SHALL remain idempotent and contradictory observations SHALL
remain independent raw facts.

#### Scenario: Create and record into a fresh v3 store

- **WHEN** a current valid v3 report is recorded to an absent store
- **THEN** one atomic operation initializes v3 and stores every selected-target
  outcome, dependency, relationship, and producer observation

#### Scenario: Record mixed compiled and JIT provenance

- **WHEN** a Metal report references CPU compiled receipts and TileLang JIT receipts
- **THEN** one Metal-selected run retains both receipt kinds and their exact graph

#### Scenario: Reject a v2 report before store creation

- **WHEN** `record` receives a schema-v2 report and the store path is absent
- **THEN** it returns status 2 with v3 guidance and leaves the path absent

### Requirement: Queries describe selected profiles and provenance axes factually

`status` SHALL filter and return observations for selected target profile,
logical kernel, variant, class, case, and producer in deterministic order,
including referenced oracle dependencies and receipt discriminators. `stale`
SHALL compare provider, target, toolchain, runtime, specialization, generated
source, compilation-input closure, executable artifact, verification,
tolerance, and oracle axes independently. `todo` SHALL return the stable
unranked difference between the selected profile's current complete
requirements and matching observations.

All successful queries SHALL be offline, factual, and mutation-free. Invalid
selectors, unavailable current baselines, or v2 state SHALL return status 2.

#### Scenario: Report selected Metal status

- **WHEN** status selects `metal-tilelang`
- **THEN** it returns matching Metal observations and their CPU oracle dependencies

#### Scenario: Explain a changed JIT specialization

- **WHEN** stored and current JIT receipts differ in specialization or generated source
- **THEN** stale reports each changed axis independently

#### Scenario: Compute target-specific todo

- **WHEN** some current Metal requirements lack matching observations
- **THEN** todo returns exactly those requirements in deterministic unranked order

### Requirement: Publication and refresh exchange complete v3 graphs atomically

Publication SHALL create one canonical content-addressed current schema-v3
snapshot per producer containing exactly its observations and complete required
run, evidence, receipt, certificate, oracle, and compilation relationships.
Refresh SHALL strictly validate every snapshot envelope, schema, identity,
canonical encoding, and complete relationship graph before store initialization
or mutation, then merge all accepted snapshots atomically and idempotently.

V2 snapshots and reports, missing mixed-receipt relationships, forged content,
or immutable identity conflicts SHALL return status 2 without changing the
destination. Ordinary record and query paths SHALL remain local; only explicit
publication or refresh SHALL access a configured exchange endpoint.

#### Scenario: Publish a mixed-provenance snapshot

- **WHEN** a producer publishes Metal observations
- **THEN** its snapshot contains the selected-target facts and complete CPU and
  TileLang dependency graph

#### Scenario: Refresh a valid v3 snapshot

- **WHEN** a complete canonical v3 snapshot is refreshed twice
- **THEN** both merges succeed atomically and the second changes no factual counts

#### Scenario: Reject a v2 snapshot atomically

- **WHEN** refresh encounters a schema-v2 snapshot
- **THEN** it returns status 2 before initializing or changing the destination

### Requirement: Tracking commands use v3-only replacement interfaces

The tracking CLI and public tracking APIs SHALL operate only on schema-v3
reports, stores, queries, publications, and refreshes. Successful commands SHALL
return status 0. Invalid usage, report schema, store schema, baseline, or
exchange input SHALL return status 2 with an actionable diagnostic on standard
error. Help SHALL describe the v3-only boundary, fresh-store behavior, and v2
create/recreate guidance without initializing storage or accessing a network.

No compatibility wrapper, automatic migration, preservation mode, or deletion
option for v2 SHALL be part of the replacement surface.

#### Scenario: Explain v2 replacement from help

- **WHEN** a caller inspects command help
- **THEN** it sees v3-only inputs, exit behavior, and manual new-store guidance
