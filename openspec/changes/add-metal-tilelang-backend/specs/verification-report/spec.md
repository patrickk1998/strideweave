## MODIFIED Requirements

### Requirement: Reports bind complete provenance before exposure

Every schema-v3 report SHALL begin with one immutable header containing the
selected target profile, oracle profile, complete required coverage, tolerance
policies, oracle identity, oracle certificates, and one `CompilationBundle`.
The bundle SHALL contain every compilation receipt referenced by evidence.

Every oracle or target execution record SHALL reference its exact receipt ID.
Every certificate-gated target record SHALL reference the exact reconstructed
oracle certificate it consumed. A JIT execution SHALL reference the receipt for
the exact specialization launched, not merely its logical kernel identity.

Report construction SHALL validate complete classifications, receipt and
certificate content identities, evidence references, payload hashes, plan
obligations, target selection, and the complete relationship graph before
exposing the report. Missing, duplicate, unexpected, stale, inconsistent, or
forged facts SHALL raise `ValueError`. Evidence SHALL use v3 receipts directly;
v2 compilation records SHALL not be nested, wrapped, or adapted.

#### Scenario: Bind one native record

- **WHEN** CPU compiled-executable evidence is bound
- **THEN** it references the exact CPU receipt in the report bundle

#### Scenario: Reject certificate-required rows without a header

- **WHEN** certificate-gated evidence is constructed without its bound header
- **THEN** construction raises `ValueError` before report exposure

#### Scenario: Reject a forged reference

- **WHEN** evidence names a receipt or certificate whose content does not match
- **THEN** report construction raises `ValueError`

#### Scenario: Bind CPU oracle and Metal target records

- **WHEN** a Metal target report is completed
- **THEN** its one bundle contains referenced CPU compiled-executable receipts
  and exact TileLang JIT-specialization receipts

### Requirement: Installed compilation provenance exposes exact immutable receipts

`installed_compilation_bundle(profile)` SHALL take a registered verification
profile and return a validated immutable schema-v3 `CompilationBundle` for the
installed compiled-executable inputs known before execution. A non-profile
value SHALL raise `TypeError`; unavailable or inconsistent installed facts
SHALL raise `RuntimeError` or `ValueError` before evidence exposure.

JIT providers SHALL add a specialization receipt only after compilation
succeeds and all required inputs and generated artifacts are available. Failed
compilation SHALL expose no receipt. Bundle receipt order SHALL be canonical by
discriminator, profile, logical kernel, and receipt ID, independent of
discovery, compilation, or cache order.

Receipts and bundles SHALL be immutable, content-addressed, and deterministic.
They SHALL not include local absolute paths, timestamps, source commits, CI
facts, cache locations, or producer observations.

#### Scenario: Resolve one installed kernel receipt

- **WHEN** the CPU compiled profile is available and internally consistent
- **THEN** its bundle contains the exact immutable compiled-executable receipts

#### Scenario: Reject an inconsistent installed manifest

- **WHEN** installed compilation inputs disagree with declared receipt facts
- **THEN** bundle creation raises `ValueError` before verification evidence exists

#### Scenario: Publish a JIT receipt only after compilation

- **WHEN** TileLang compilation fails or omits required generated artifacts
- **THEN** no specialization receipt is added to the bundle

### Requirement: Report loading is strict, offline, and line-diagnostic

`VerificationReport.load(path)` and `VerificationReport.from_jsonl(text)` SHALL
accept only canonical schema-v3 JSONL and reconstruct the entire report without
consulting an installed backend, compilation provider, accelerator runtime,
store, network, or source tree. The loader SHALL validate the header, bundle,
receipt discriminators and fields, complete input closures, typed artifacts,
receipt and certificate identities, evidence references, ordering, and every
nested relationship before returning immutable values.

Malformed JSON, non-canonical encoding, unsupported schemas, unknown receipt
discriminators, missing or extra fields, duplicate identities, invalid ordering,
incomplete closure, forged digests, and mismatched references SHALL raise the
documented `ValueError` carrying the one-based failing line when attributable.
Schema-v2 and prototype evidence-only reports SHALL be rejected directly and
SHALL not be migrated or adapted.

#### Scenario: Reject prototype evidence-only JSONL

- **WHEN** input begins with evidence rather than a complete v3 header
- **THEN** loading raises the line-diagnostic `ValueError`

#### Scenario: Reject an earlier report schema

- **WHEN** a schema-v2 report is supplied
- **THEN** loading rejects it as unsupported without migration

#### Scenario: Validate a report on another installation

- **WHEN** canonical v3 JSONL is loaded where neither Metal nor TileLang exists
- **THEN** all provenance is validated from report bytes alone

## ADDED Requirements

### Requirement: Compilation bundles discriminate executable and JIT receipts

A `CompilationBundle` SHALL contain an ordered immutable tuple of receipts.
Every receipt SHALL have exactly one discriminator: `compiled-executable` or
`jit-specialization`. Both kinds SHALL bind schema version, receipt ID, profile,
provider, target, toolchain and runtime facts, logical kernel identity, complete
ordered provider-declared compilation inputs, and ordered typed generated
artifacts. Receipt ID SHALL be the deterministic digest of its canonical facts.

Each compilation input SHALL bind a stable path-free URI, input kind, ordinal,
and content digest. Each generated artifact SHALL bind an artifact kind, ordinal,
content digest, and optional provider-declared executable digest. Any
compilation-affecting fact not represented by the declared closure SHALL make
receipt construction fail with `ValueError` as undeclared rather than silently
remaining outside identity.

A `compiled-executable` receipt SHALL additionally bind ordered compiled object
artifacts and the shared executable artifact used by the CPU profile. A
`jit-specialization` receipt SHALL additionally bind the exact ordered
specialization axes and values, generated host source, generated device source,
and every executable or runtime artifact the provider exposes. A provider that
cannot supply a required source or declared artifact SHALL fail before evidence
exposure. The two kinds SHALL not be required to expose identical artifacts.

Canonical JSON SHALL order object keys lexicographically and retain tuple order
for closures, specialization axes, and artifacts. Unknown discriminators,
missing kind-specific fields, fields belonging only to the other kind,
duplicate ordinals or URIs, non-canonical ordering, incomplete closure, or
forged digests SHALL raise `ValueError`.

#### Scenario: Bind a CPU compiled executable

- **WHEN** CPU receipt facts cover its complete declared inputs, objects, and
  shared executable
- **THEN** a deterministic `compiled-executable` receipt is produced

#### Scenario: Distinguish two Metal specializations

- **WHEN** two TileLang launches differ in one compilation-affecting axis
- **THEN** their `jit-specialization` receipts have distinct IDs while their
  logical classification identity may remain the same

#### Scenario: Reject incomplete generated-code provenance

- **WHEN** a JIT receipt omits generated host or device source or a declared
  compilation input
- **THEN** receipt construction raises `ValueError` before evidence exposure

#### Scenario: Reject a v2 receipt payload

- **WHEN** a v2 native receipt is supplied where a v3 receipt is required
- **THEN** validation rejects it rather than wrapping or translating it
