---
title: Verification Report
publish: true
status: stable
order: 100
summary: Portable provenance-complete verification reports, strict loading, filtering, summaries, and inspection.
---

# verification-report Specification

## Purpose

Define the immutable, provenance-complete report format and the APIs and CLI for
strictly loading, filtering, summarizing, and inspecting local verification.

## Terminology

| Term | Meaning |
| --- | --- |
| evidence record | One immutable attempted-case fact identifying its stage, verification class, deterministic case and plan metadata, shared input-bit hashes, tolerance, deviations, mismatch count, outcome, diagnostic, and bound provenance references. |
| verification outcome | Exactly one of `passed`, `failed`, `error`, `blocked`, or `deferred`; it states the result of an attempted case and is not a confidence judgment. |
| report header | The first JSONL object, binding the report schema, evidence schema, compilation facts, complete verification requirements, tolerance policies, Generic oracle references, Stage One certificates, and its own content digest. |
| provenance-complete report | One immutable report whose header and evidence form a closed, referentially valid set that can be validated without consulting the currently installed build. |
| report view | A provenance-complete immutable report containing an exact filtered subset of another report's evidence while retaining the dependencies consumed by that subset. |
| correctness gate | The Boolean report result that passes exactly when no selected evidence outcome is `failed`, `error`, or `blocked`; `deferred` remains visible but does not alone fail the gate. |

## Requirements

### Requirement: Exported report values have explicit field contracts

`VerificationOutcome` SHALL expose exactly `passed`, `failed`, `error`,
`blocked`, and `deferred`. The exported direct-construction report-side records
SHALL accept these annotated field types:

| Value | Supported constructor fields and result |
| --- | --- |
| `CaseDescriptor` | `operation: str`, `kernel_id: str`, `variant: str`, `input_dtypes: tuple[str, ...]`, `output_dtype: str`, `shapes: tuple[tuple[int, ...], ...]`, `accumulator_dtype: str | None`, `contraction_length: int | None`, `seed: int | None`, `case_id: str`, and `plan: PlanKey | None = None`; returns one deterministic case identity. |
| `Tolerance` | `absolute: float = 0.0`, `relative: float = 0.0`, `ulps: int = 0`, and `version: str = "exact-v1"`; returns one versioned threshold holder. |
| `Deviations` | `maximum_absolute: float | None`, `maximum_relative: float | None`, and `maximum_ulps: int | None`; returns observed comparison maxima or absent measurements. |
| `VerificationSummary` | `total: int`, `outcomes: tuple[tuple[VerificationOutcome, int], ...]`, `stages: tuple[tuple[VerificationStage, int], ...]`, `classes: tuple[tuple[VerificationClass, int], ...]`, and `gate_passed: bool`; returns one aggregate holder. |
| `ReportHeader` | `schema_version: str`, `header_digest: str`, `compilation: Mapping[str, Any]`, `verification_spec: Mapping[str, Any]`, `tolerance_policies: tuple[Mapping[str, Any], ...]`, `oracle_references: tuple[Mapping[str, Any], ...]`, and `certificates: tuple[Mapping[str, Any], ...]`; returns one header field holder. |
| `CompilationInput` | `uri: str`, `input_kind: str`, and `content_digest: str`; returns one content-addressed compilation-input holder. |
| `CompilationTarget` | `target_id: str`, `architecture: str`, `vendor: str`, `operating_system: str`, `abi: str`, `endianness: str`, and `pointer_bits: int`; returns one target-identity holder. |
| `CompilationToolchain` | `toolchain_id: str`, `provider_kind: str`, `compiler_id: str`, `compiler_version: str`, `target_triple: str`, and `build_system: str`; returns one toolchain-identity holder. |
| `CompilationReceipt` | `receipt_id: str`, `receipt_schema: str`, `provider_kind: str`, `kernel: KernelDescriptor`, `target: CompilationTarget`, `toolchain: CompilationToolchain`, `closure_id: str`, `inputs: tuple[CompilationInput, ...]`, `compile_invocation: tuple[str, ...]`, `compile_invocation_digest: str`, `object_digest: str`, `shared_artifact_digest: str`, `framework_name: str | None`, `framework_version: str | None`, and `specialization: Mapping[str, object]`; returns one receipt field holder. |
| `CompilationManifest` | `manifest_digest: str`, `schema: str`, `provider_kind: str`, `artifact_digest: str`, `target: CompilationTarget`, `toolchain: CompilationToolchain`, and `receipts: tuple[CompilationReceipt, ...]`; returns one compilation field holder. |

Within those supported types, direct construction SHALL return a shallow-frozen
field holder whose attributes are protected from reassignment. It SHALL retain
supplied field objects, including mutable mappings. Strict validated
construction through manifest/report loaders SHALL own type, canonicality,
content-identity, and deep-immutability enforcement under the rules below.

#### Scenario: Construct a report-side field holder

- **WHEN** a caller supplies every declared `CaseDescriptor` field using its annotated type
- **THEN** construction returns a shallow-frozen case holder retaining those exact field objects

### Requirement: Evidence records preserve every attempted-case fact

`EvidenceRecord(stage: VerificationStage, test_class: VerificationClass, case:
CaseDescriptor, target_input_bit_hashes: tuple[str, ...],
oracle_input_bit_hashes: tuple[str, ...], tolerance: Tolerance, deviations:
Deviations, mismatches: int | None, outcome: VerificationOutcome,
diagnostic: str | None = None, requirement_id: str = "unbound",
compilation_receipt_id: str | None = None, tolerance_policy_id: str = "unbound",
oracle_reference_id: str = "unbound", consumed_certificate_digest: str | None =
None, schema_version: str = "strideweave.kernel-evidence.v2")` SHALL return an
immutable record containing one
`VerificationStage` (`stage_one` or `stage_two`), one `VerificationClass`
(`bit_exact`, `exact_arithmetic`, `structural`, `analytic`, `numerical`, or
`deferred`), one deterministic `CaseDescriptor`, matching ordered target and
oracle input-bit hashes, one versioned `Tolerance`, observed `Deviations`, an
nullable mismatch count, one `VerificationOutcome`, an optional diagnostic defaulting to `None`, and
content-bound requirement, compilation-receipt, tolerance-policy, Generic-oracle,
and consumed-certificate references.

The case descriptor SHALL retain operation, kernel ID, variant, ordered input
dtypes, output dtype, effective shapes, optional accumulator dtype, optional
contraction length, optional seed, stable case ID, and optional exact executable
plan. A completed comparison SHALL retain maximum absolute,
symmetric-relative, and Float32-ULP deviations. A recoverable execution error
SHALL retain the prepared metadata and represent unavailable comparison
measurements as absent values. A blocked Stage Two case SHALL retain its
prepared identity, zero absolute, relative, and ULP deviations, zero mismatches,
and its authorization diagnostic.

Target and oracle input-bit hash tuples SHALL be identical by construction.
Constructing an evidence record with unequal tuples SHALL fail with `ValueError`.
Serialization SHALL encode non-finite deviation values as canonical strings so
the result remains strict JSON containing only standard number literals.

`record.as_json_object()` SHALL accept no inputs and return a mutable JSON-safe
object containing every evidence field with enum values represented by their
wire strings. `record.to_jsonl()` SHALL accept no inputs and return that object
as one compact, deterministically key-ordered JSON line without a trailing
newline.

#### Scenario: Preserve a recoverable execution error

- **WHEN** one verification case raises a recoverable execution error after its inputs and tolerance are prepared
- **THEN** its immutable error record retains the prepared case, plan, hashes, tolerance, provenance references, and diagnostic while unavailable comparison measurements remain absent

#### Scenario: Reject independently encoded inputs

- **WHEN** an evidence record is constructed with different target and oracle input-bit hash tuples
- **THEN** construction fails with `ValueError`

### Requirement: Reports bind complete provenance before exposure

`VerificationReport(records: tuple[EvidenceRecord, ...], schema_version: str =
"strideweave.kernel-verification.v2", header: ReportHeader | None = None)` SHALL take
`records`, the complete immutable evidence tuple; `schema_version`, the report
wire-format version that defaults to the current provenance-complete v2 value;
and `header`, an optional immutable report header that defaults to `None`.
Another schema version SHALL fail with `ValueError`.

When `header` is `None`, construction SHALL bind the evidence with an empty
certificate set to the current
compilation manifest and per-kernel receipts, target and toolchain, complete
verification requirements, versioned tolerance policies, Generic oracle
identity, and record references before exposing the report. This path SHALL
succeed only when every record can be bound without a consumed certificate; a
non-blocked Stage Two `reduce_sum` or `matmul` record that requires a certificate
SHALL make construction fail with `ValueError`. Certificate-bearing complete
reports SHALL be returned by `test_backend()` or reconstructed from JSONL using
a supplied validated header. When `header` is supplied, construction SHALL
validate it and every record reference as a closed set.

The Generic oracle identity SHALL be content-derived from the reviewed Generic
roots and their complete local static-import closure. Each native CPU evidence
record SHALL identify the exact compilation receipt for its kernel and variant.
Each non-blocked Stage Two `reduce_sum` or `matmul` record SHALL identify the
exact Stage One certificate it consumed. The report SHALL reject missing,
duplicate, unexpected, unknown, mismatched, malformed, or forged identities and
references with `ValueError`.

The canonical report wire format SHALL consist exactly of its bound compilation,
verification-requirement, tolerance-policy, Generic-oracle, certificate, and
evidence facts. Observation state such as wall-clock timestamps, CI status,
database state, source commits, producers, publication, confidence, and
autotuning belongs to the separate evidence-tracking model.

`header.as_json_object()` SHALL accept no inputs and return a mutable JSON-safe
object reflecting the header fields. For a header exposed by a validated report,
the result SHALL be the canonical header object including its content-derived
digest.

#### Scenario: Bind one native record

- **WHEN** `test_backend()` binds native CPU evidence and its Stage One certificates, or a supplied validated header is loaded with those facts
- **THEN** the exposed record references its exact compilation receipt, tolerance policy, Generic oracle, verification requirement, and consumed certificate when Stage Two requires one

#### Scenario: Reject certificate-required rows without a header

- **WHEN** `VerificationReport(records, header=None)` receives a non-blocked Stage Two reduction or matmul record
- **THEN** construction fails with `ValueError` because the implicit empty certificate set cannot bind that row

#### Scenario: Reject a forged reference

- **WHEN** a record names a receipt, tolerance policy, oracle, requirement, or certificate not present under the matching header identity
- **THEN** report construction fails with `ValueError` before returning a report

### Requirement: Installed compilation provenance exposes exact immutable receipts

`load_compilation_manifest()` SHALL accept no inputs, read the installed native
compilation manifest once per process, strictly validate its current schema and
content identities against the installed native kernel metadata, and return an
immutable `CompilationManifest`. A missing or malformed installed resource
SHALL fail with `RuntimeError`; an unsupported schema, invalid field, mismatched
digest, duplicate source, source/manifest mismatch, or other provenance
inconsistency SHALL fail with `ValueError`.

The manifest SHALL expose its content digest, schema, provider kind, shared
artifact digest, exact `CompilationTarget`, exact `CompilationToolchain`, and a
deterministically ordered immutable receipt tuple. Each `CompilationReceipt`
SHALL bind one exact native kernel/variant to its receipt schema, provider,
target, toolchain, complete content-addressed input closure, normalized compile
invocation and digest, compiled-object digest, shared-artifact digest, and any
framework or specialization facts.

`manifest.receipt_for(kernel_id, variant)` SHALL take exact string `kernel_id`
and `variant` identities and return the unique matching receipt. An identity for
which exactly one receipt does not exist SHALL fail with `ValueError`.

#### Scenario: Resolve one installed kernel receipt

- **WHEN** `receipt_for` receives a kernel ID and variant that occur exactly once in the validated installed manifest
- **THEN** it returns the immutable receipt whose kernel, target, toolchain, closure, invocation, object, and artifact identities are bound together

#### Scenario: Reject an inconsistent installed manifest

- **WHEN** the installed manifest has an invalid digest, duplicate owning source, or a source set that differs from native kernel metadata
- **THEN** strict manifest loading fails before a report binds any native evidence

### Requirement: Stage One certificates are evidence-reconstructable

An embedded Stage One certificate SHALL name one exact kernel ID and variant,
its certified-class union, exact executable-plan/class coverage, evidence
digest, and content-derived certificate digest. When a report contains the
complete matching Stage One evidence, loading SHALL independently reconstruct
the certificate and require the same plan/class coverage and evidence digest.

A report view that omits some or all evidence behind an already validated
certificate SHALL retain that certificate when selected target evidence
consumes it. Validation of such a view SHALL require every available matching
Stage One row to agree with the certificate, while recognizing that an omitted
evidence digest cannot be recomputed from the subset.

#### Scenario: Reconstruct complete certificate evidence

- **WHEN** a loaded report contains all Stage One rows for an embedded certificate
- **THEN** loading reconstructs its kernel, variant, plan/class obligations, class union, and evidence digest and rejects any disagreement

#### Scenario: Preserve a certificate in a target-only view

- **WHEN** filtering retains Stage Two evidence but omits the Stage One rows behind its consumed certificate
- **THEN** the returned report view retains the already validated certificate as a dependency and remains provenance-complete for the selected rows

### Requirement: JSONL serialization is canonical and deterministic

`report.to_jsonl()` SHALL accept no inputs and return newline-terminated UTF-8
text whose first line is the strict report header and whose remaining lines are
all evidence records ordered by case ID, kernel ID, and variant. Every object
SHALL use deterministic key ordering and compact canonical JSON. Empty reports
SHALL still contain their required header line.

`report.write(path)` SHALL take `path`, a string or filesystem path-like
destination, replace it with `report.to_jsonl()` encoded as UTF-8, and return
`None`. Filesystem failures SHALL remain observable as the corresponding I/O
error.

For the same immutable compilation, specification, tolerance, oracle,
certificate, and evidence facts, serialization SHALL produce identical bytes.
The v2 provenance-complete report format SHALL replace prototype v1 directly;
the strict loader SHALL treat v1 evidence-only files as invalid input and fail
with `ValueError`.

#### Scenario: Serialize the same facts twice

- **WHEN** one report is serialized twice without changing any bound fact
- **THEN** both results are byte-identical canonical JSONL ending in one newline

#### Scenario: Serialize an empty report

- **WHEN** a report contains no evidence records
- **THEN** `to_jsonl()` returns exactly one canonical header line followed by a newline

### Requirement: Report loading is strict, offline, and line-diagnostic

`VerificationReport.from_jsonl(text)` SHALL take `text`, a decoded JSONL string,
and return the validated immutable report. A non-string input SHALL fail with
`TypeError`. The first line SHALL be a current v2 provenance header. Missing or
blank header lines, blank evidence lines, malformed JSON, duplicate object keys,
non-standard `NaN` or infinity literals, unknown or missing fields, unsupported
schema versions, invalid enum values, malformed nested values, duplicate
requirements or evidence, incomplete header coverage, and any provenance or
certificate mismatch SHALL fail with `ValueError` identifying the one-based
JSONL line at which parsing failed when applicable.

Loading SHALL validate every nested identity exclusively from the supplied
header and evidence bytes. Consequently the installed compilation manifest,
current source tree, network, status store, and database are outside its inputs.

`VerificationReport.load(path)` SHALL take `path`, a string or filesystem
path-like input, read it as UTF-8, and return
`VerificationReport.from_jsonl(...)`. Filesystem failures SHALL remain
observable as the corresponding I/O error.

#### Scenario: Reject prototype evidence-only JSONL

- **WHEN** the first line is a prototype v1 evidence record instead of a current provenance header
- **THEN** loading fails with a line-1 `ValueError` identifying the required provenance header

#### Scenario: Validate a report on another installation

- **WHEN** a provenance-complete current report is loaded where the installed build differs or is unavailable
- **THEN** loading validates solely from the report's bound facts and returns the same immutable evidence model

### Requirement: Summary and bounded text expose the complete selected gate

`report.summary()` SHALL accept no inputs and return an immutable
`VerificationSummary` containing the total record count, one count for every
declared outcome, stage, and verification class in declaration order including
zero counts, and the correctness gate. Its `passed`, `failed`, `errors`,
`blocked`, and `deferred` properties SHALL return the corresponding outcome
counts.

The correctness gate SHALL be false exactly when at least one selected record
is failed, errored, or blocked. Empty and deferred-only reports SHALL pass.

`report.describe()` SHALL accept no inputs and return deterministic bounded
plain text containing total, gate, complete outcome counts, complete stage
counts, and complete class counts. `repr(report)` and `repr(report.summary())`
SHALL contain only bounded aggregate counts and gate state.

#### Scenario: Summarize deferred-only evidence

- **WHEN** every selected record has the deferred outcome
- **THEN** the summary counts those records, reports zero failures, errors, and blocks, and sets `gate_passed` to true

#### Scenario: Summarize one blocked target

- **WHEN** the selected evidence contains one blocked Stage Two record
- **THEN** the blocked count is one and `gate_passed` is false

### Requirement: Report selection composes exact filters and preserves provenance

`report.select(*, stage=None, outcomes=None, test_class=None, operation=None,
kernel_id=None, variant=None)` SHALL take optional exact filters that all default
to `None`. `stage`, `outcomes`, and `test_class` SHALL each accept one value of
its corresponding enum or a collection of those enum values; another value
SHALL fail with `TypeError`. `operation`, `kernel_id`, and `variant` SHALL each
accept a string or `None`; another value SHALL fail with `TypeError`.

The result SHALL be a provenance-complete immutable report view containing only
records that match every supplied filter. Selection SHALL be invariant under
filter ordering. `report.passed` SHALL equal selection of `passed`;
`report.deferred` SHALL equal selection of `deferred`; and `report.problems`
SHALL equal selection of failed, errored, and blocked outcomes while excluding
deferred records.

The report view SHALL narrow verification requirements, tolerance policies, and
oracle references to those selected records, retain compilation facts required
to validate exact receipts, and retain every Stage One certificate consumed by
the selected target evidence.

#### Scenario: Compose all exact filters

- **WHEN** selection supplies stage, outcomes, class, operation, kernel ID, and variant
- **THEN** the result contains exactly the records satisfying all six constraints and its summary and gate describe only that subset

#### Scenario: Select problems

- **WHEN** a report contains passed, failed, error, blocked, and deferred records
- **THEN** `report.problems` contains only failed, error, and blocked records and remains a loadable provenance-complete report

### Requirement: The inspection CLI uses the strict report model

`strideweave-verify-report REPORT` SHALL take `REPORT`, the filesystem path to a
verification JSONL file, load it through the same strict model parser, apply
optional filters, and print a deterministic summary derived exclusively from
the loaded immutable report. Kernel execution and external network, database,
status, and mutation capabilities are outside this inspection command.

The CLI SHALL accept `--problems`; exact `--operation`, `--kernel`, and
`--variant` filters; enum-constrained `--stage` and `--class` filters; repeatable
enum-constrained `--outcome`; `--verbose`; and `--json`. Filters SHALL compose
with the public report-selection semantics. `--problems` SHALL first restrict
the report to failed, error, and blocked evidence, after which all other filters
still apply.

Default text output SHALL print `report.describe()`. `--verbose` SHALL add one
deterministic flat line per selected case with case identity, stage, operation,
kernel, variant, class, outcome, deviations, tolerance, and provenance
references. `--json` SHALL print stable compact JSON containing the complete
summary; with `--verbose`, it SHALL additionally contain a stable `records`
array with those flat fields.

The CLI SHALL exit `0` when the selected correctness gate passes, `1` when it
fails, and `2` when report loading or command usage fails. A load failure SHALL
write a diagnostic to standard error. Help and argument validation SHALL remain
side-effect-free.

#### Scenario: Inspect only selected problems

- **WHEN** the CLI receives `REPORT --problems --stage stage_two --verbose`
- **THEN** it prints the selected Stage Two failed, error, and blocked cases and exits according to that selected subset's gate

#### Scenario: Reject malformed report input

- **WHEN** `REPORT` cannot be read or fails strict report validation
- **THEN** the CLI writes an error diagnostic to standard error, exits `2`, and leaves verification and persistence to their owning capabilities

### Requirement: Reports contain raw facts and bound provenance

The report wire format and inspection APIs SHALL contain immutable raw
verification attempts and their bound compilation, requirement, tolerance,
oracle, and certificate provenance. The separate evidence-tracking capability
SHALL own producer observations, source commits, database identities, stored
status/staleness/todo queries, publication, refresh, contributor exchange,
confidence lattices, risk rankings, and autotuning.

#### Scenario: Inspect a portable report

- **WHEN** a report is serialized, transferred, loaded, filtered, and summarized
- **THEN** its results depend only on its immutable bound facts and its field set remains exactly the raw report/provenance contract
