## MODIFIED Requirements

### Requirement: Native kernels and executable CPU plans have complete classifications

`verification_profiles()` SHALL return immutable `VerificationProfile` records
sorted by `profile_id`. Each record SHALL expose exact string `profile_id`,
`carrier`, and `provider` fields plus ordered `VerificationStage` roles. The
stages SHALL be `oracle` and `target`. Shipped profiles SHALL include
`cpu-compiled`, usable in both roles, and `metal-tilelang`, usable as a target.

`verification_profile(profile_id)` SHALL resolve an exact string selector. A
non-string selector SHALL raise `TypeError`; an unknown selector SHALL raise
`ValueError` before provider initialization, compilation, execution, or output
mutation. `kernel_manifest(profile)` SHALL return immutable `LogicalKernel`
records binding `profile_id`, `operation`, `kernel_id`, and `variant`.

`classify_profile_plans(profile)` SHALL return immutable `PlanClassification`
records covering every plan advertised by the profile. Each SHALL bind one
logical kernel, one exact `PlanKey`, ordered `VerificationClass` values, an
active or deferred `ClassificationDisposition`, and a non-empty reason for a
deferral. `PlanKey.from_plan_like(plan)` SHALL normalize the operation, ordered
operand roles and dtypes, conversions, compute dtype, accumulation kind and
dtype, and output dtype.

Completeness SHALL raise `ValueError` before execution for any duplicate,
missing, stale, unknown, or mismatched manifest kernel, plan, operation,
classification, movement subject, or structural subject. `metal-tilelang`
SHALL cover at least one executable plan for every registered computational
dispatch name. Logical classification identity SHALL remain bounded
independently of runtime JIT specialization identity.

The public profile, logical-kernel, plan-key, and classification records SHALL
be immutable. Lookup, manifest, classification, and staged-verification
functions SHALL own canonicality, completeness, and deep validation.

#### Scenario: Classify the complete installed backend

- **WHEN** every logical kernel, executable plan, operation, and required
  structural subject has exactly one current classification
- **THEN** classification returns their complete exact pairing

#### Scenario: Reject stale classification metadata

- **WHEN** a profile gains, loses, or duplicates a logical kernel or plan
  without its classifications changing
- **THEN** completeness validation raises `ValueError` before execution

#### Scenario: Keep JIT specialization out of classification

- **WHEN** two TileLang executions specialize one logical variant differently
- **THEN** they share one logical classification and use distinct receipts

### Requirement: Active and deferred dispositions remain explicit

An active plan SHALL name every required verification class. A deferred plan
SHALL carry no active classes and a non-empty reason, and its applicable stage
SHALL emit exactly one deferred outcome. The CPU profile SHALL preserve current
vendor-transcendental and floating-`pow` deferrals while exact-integer `pow`
remains active. Each profile SHALL explicitly classify required movement and
structural subjects independently of numerical kernels.

#### Scenario: Report a vendor-transcendental plan

- **WHEN** a CPU oracle plan is deferred for vendor-math accuracy
- **THEN** evidence carries its concrete reason and no certificate covers it

#### Scenario: Split pow by executable plan

- **WHEN** CPU has active integer and deferred floating `pow` plans
- **THEN** each receives its own exact disposition and evidence

### Requirement: Stage One records every required attempt and certifies complete passing coverage

`run_oracle_stage(profile)` SHALL require a profile registered for `oracle` and
return an immutable `OracleStageResult` containing a report and certificates.
The shipped oracle SHALL be `cpu-compiled`. Test-only result alteration SHALL
exist only behind non-public test seams and SHALL not be an API input.

The oracle stage SHALL attempt every required class for every active plan, emit
required movement and structural cases, and emit one record for every deferred
plan. It SHALL preserve signed-zero witnesses, exactly representable structural
witnesses, independent analytic expectations, deterministic wide-exponent
inputs, exact plan identities, and versioned tolerance policies.

A recoverable case error SHALL produce complete `error` evidence retaining the
prepared operation, logical kernel, variant, class, plan, hashes, shapes,
contraction length, seed, and tolerance, and SHALL not suppress independent
cases. An `OracleCertificate` SHALL bind oracle profile, logical kernel,
operation, exact oracle plan, complete classes, and deterministic evidence
digest. Missing, failed, errored, forged, or incomplete evidence SHALL prevent
certificate issuance.

#### Scenario: Certify every active plan of one kernel

- **WHEN** every required witness passes for one logical oracle obligation
- **THEN** its certificate binds the operation, plan, classes, and evidence

#### Scenario: Fail closed on one required witness

- **WHEN** one required witness fails or errors
- **THEN** evidence remains, independent cases continue, and no certificate
  covers the incomplete obligation

### Requirement: Stage Two runs only behind an exact valid certificate

`run_target_stage(profile, oracle_result)` SHALL require a target-stage profile
and a valid accepted `OracleStageResult`, then verify every active target plan
plus required movement and structural subjects. Invalid roles or oracle results
SHALL fail before target preparation or execution.

Authorization SHALL require a reconstructed oracle certificate matching the
target operation, complete oracle-plan obligations, and every required class.
Target and oracle kernel IDs MAY differ. Float32 `reduce_sum` and `matmul`
targets SHALL require certified CPU Float64-accumulator oracle coverage.

Absent, forged, inconsistent, or incomplete authorization SHALL emit `blocked`
evidence instead of executing. Explicit oracle deferrals SHALL emit deferred
target evidence. Exact cases SHALL preserve bit identity; structural cases
SHALL use exactly representable witnesses; numerical cases SHALL use the
versioned single-path Float32 gamma-K envelope while recording absolute,
symmetric-relative, and ULP deviations. Catalogs SHALL cover non-compact,
hierarchical, and multi-output layouts and record effective shapes.

Recoverable errors SHALL produce complete `error` evidence without suppressing
independent cases. The profile SHALL synchronize target work before decoding,
comparison, or evidence classification.

#### Scenario: Block a target without Float64 oracle scope

- **WHEN** authorization omits the required Float64 plan or class
- **THEN** blocked evidence replaces affected target execution

#### Scenario: Run an authorized hierarchical contraction

- **WHEN** CPU oracle evidence authorizes a hierarchical target case
- **THEN** synchronization precedes decode and the result records hashes,
  shapes, contraction length, deviations, tolerance, and outcome

#### Scenario: Continue after a target error

- **WHEN** one target case raises a recoverable execution error
- **THEN** complete error evidence is retained and independent cases continue

### Requirement: Deferred verification domains remain explicit boundaries

Vendor-transcendental and floating-`pow` accuracy, autograd certification,
confidence ranking, autotuning, unsupported compilation providers, and CI
integration SHALL remain explicit deferred domains. The shipped TileLang JIT
provider and Metal target SHALL be active. Local verification SHALL read only
installed and deterministic local facts, return an immutable report, and write
only an optional caller destination. Evidence persistence and exchange SHALL
remain owned by `kernel-evidence-tracking`.

#### Scenario: Inspect a local report with deferred work

- **WHEN** a selected profile contains a deferred plan
- **THEN** its report retains deferred evidence without persisting it

## ADDED Requirements

### Requirement: Backend verification selects one explicit target profile

`verify_backend(target, *, output=None)` SHALL require `target`, a string
registered target `profile_id`, and accept optional string or path-like
`output`, defaulting to `None`. Invalid types SHALL raise `TypeError`; an
unknown target SHALL raise `ValueError` before provider initialization,
compilation, execution, or output mutation.

The call SHALL resolve the profile, require its provider and runtime, run the
`cpu-compiled` oracle stage, run the selected target stage, and bind evidence,
certificates, and compilation receipts directly into one schema-v3
`VerificationReport`. Missing runtime or dependencies SHALL raise actionable
`RuntimeError` before affected target execution is represented as passing.

The call SHALL return the immutable report. If `output` is supplied, it SHALL
atomically replace only that path with canonical UTF-8 JSONL after the report is
complete. I/O errors SHALL remain observable without partial replacement.
CI state, Git history, wall-clock time, evidence stores, exchange, and network
resources SHALL not be execution inputs. No prototype selector or report
compatibility mode SHALL exist.

#### Scenario: Verify the compiled CPU target

- **WHEN** `verify_backend("cpu-compiled")` runs on a compatible installation
- **THEN** it returns a deterministic v3 CPU oracle-and-target report

#### Scenario: Verify the Metal TileLang target

- **WHEN** `verify_backend("metal-tilelang")` runs on available Metal
- **THEN** it returns CPU oracle dependencies and synchronized Metal evidence

#### Scenario: Write the returned report atomically

- **WHEN** `verify_backend(target, output=path)` succeeds
- **THEN** `path` is atomically replaced with the returned canonical report

## REMOVED Requirements

### Requirement: test_backend is local, deterministic, and optionally writes one report

**Reason**: The CPU-oriented prototype entry point is replaced before it has
external consumers by an explicit provider-neutral target-profile API.

**Migration**: Use `verify_backend(target, output=path)`. Prototype names,
arguments, report schemas, and output compatibility are not retained.
