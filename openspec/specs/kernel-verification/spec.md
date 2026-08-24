---
title: Kernel Verification
publish: true
status: stable
order: 90
summary: Fail-closed classification and staged local verification of selected installed CPU and Metal targets.
---

# kernel-verification Specification

## Purpose

Define fail-closed classification and staged local verification of explicit
installed target profiles, using Generic to certify CPU oracle behavior and
that CPU authority to validate targets such as TileLang Metal.

## Terminology

| Term | Meaning |
| --- | --- |
| verification profile | One immutable installed oracle or target identity with an exact carrier, compilation provider, stage roles, logical-kernel manifest, executable plans, and required movement or structural subjects. |
| logical kernel | One bounded operation, stable kernel ID, and variant identity in a verification profile; compilation-affecting shapes, layouts, dtypes, plans, and options remain specialization facts rather than creating new logical kernels. |
| native kernel | One compiled CPU operation implementation identified by its operation name, stable kernel ID, variant, native Python/pybind binding name, and owning source identity. |
| executable CPU plan | One exact operation plan advertised by the installed CPU carrier under `backend-capabilities`, including operand roles and dtypes, conversions, compute arithmetic, accumulation, accumulator dtype, and output dtype. |
| verification classification | The complete ordered verification-class obligation assigned to one profile logical kernel and exact executable plan, refined into an active or deferred disposition independently of any runtime JIT specialization. |
| encoded payload | An immutable, content-hashed operand representation from which both target and oracle values are decoded, placing source quantization outside the comparison boundary. |
| Stage One | Local oracle validation that compares the installed CPU implementation with Generic for every active executable CPU plan and issues plan-scoped certificates only for complete passing coverage. |
| Stage One certificate | A digest-backed fact for one exact kernel/variant identifying every certified verification class, every certified executable-plan/class obligation, and the evidence digest from which it was issued. |
| Stage Two | Local validation of one explicit installed target profile, including CPU or TileLang Metal, whose active computational cases run only behind matching reconstructed Stage One authority and whose queued target work synchronizes before host-visible comparison. |
| JIT specialization | One compilation identity beneath a logical JIT kernel, distinguished by every compilation-affecting plan, dtype, layout, shape, and option axis and bound to its exact receipt without expanding the logical classification surface. |
| recoverable case error | A `RuntimeError` or `ValueError` raised while preparing or executing one independent verification case; it becomes error evidence while other cases continue. |

## Requirements

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

### Requirement: Encoded inputs define one immutable comparison boundary

`EncodedFloat32Payload.from_bits(bits)` SHALL take `bits`, an iterable of
non-Boolean integral words in the inclusive unsigned-32-bit range, and return an
immutable tuple of canonical words plus the SHA-256 hash of their little-endian
bytes. A Boolean or non-integral word SHALL fail with `TypeError`; an out-of-range
word SHALL fail with `ValueError`.

`EncodedFloat32Payload.from_values(values)` SHALL take `values`, an iterable of
Python `int` or `float` values encodable as IEEE-754 binary32, quantize each
value exactly once, and return the immutable canonical words and content hash.
`EncodedInt32Payload.from_values(values)` SHALL take `values`,
an iterable of Python integers in the inclusive signed-32-bit range, and return
their immutable unsigned canonical words and content hash.

The exported payload record constructors SHALL have these supported types:
`EncodedFloat32Payload(bits: tuple[int, ...], bit_hash: str)` and
`EncodedInt32Payload(bits: tuple[int, ...], bit_hash: str)` return shallow-frozen
payload holders; `EncodedInputs(operands)` SHALL take `operands`, a tuple whose
members are public Float32 or Int32 payloads or Boolean payloads produced by the
staged verifier, and return a shallow-frozen ordered operand collection;
`ExactStructuralPayload(lhs: EncodedFloat32Payload, rhs:
EncodedFloat32Payload | None, contraction_length: int, operand_bound: int,
mantissa_bits: int)` returns the generated exactness facts; and
`AnalyticCase(case_id: str, operation: str, inputs: tuple[tuple[float, ...],
...], expected: tuple[float, ...])` returns one analytic witness. Direct
construction SHALL prevent field reassignment while retaining the supplied
field objects. The public factories SHALL own type, hash, canonicality,
exactness, and deep-immutability validation.

`EncodedInputs(operands)` SHALL take `operands`, the ordered encoded operand
payloads. Its `target_values()` and `oracle_values()` methods SHALL accept no
inputs and return ordered tuples decoded from those same immutable payloads;
its `input_hashes` property SHALL return the payload hashes in operand order.
`EncodedFloat32Payload.values()` and `EncodedInt32Payload.values()` SHALL accept
no inputs and return immutable decoded tuples without changing the stored words.
Every evidence record SHALL therefore carry identical
target and oracle input-bit hashes; constructing a record with unequal hash
tuples SHALL fail with `ValueError`.

`arbitrary_float32_payload(seed, count)` and
`wide_exponent_float32_payload(seed, count)` SHALL take integer deterministic
generator `seed` and non-negative integer element `count`, return finite encoded
Float32 payloads, and fail with `ValueError` when `count` is negative.
`adversarial_float32_payload()` SHALL accept no inputs and return fixed encodings
covering positive and negative zero, subnormals, finite extremes, infinities,
and distinct NaN payloads.

`analytic_cases()` SHALL accept no inputs and return an immutable tuple of
independently named deterministic reduction and matmul witnesses, each carrying
its operation, complete input sequences, and expected result sequence.

`exact_structural_payload(seed, contraction_length, *, product, rows=1,
mantissa_bits=24)` SHALL take integer deterministic `seed`, positive integer
reduction-fiber `contraction_length`, Boolean product-mode selector `product`,
positive integer fiber count `rows`, and positive integer exact-mantissa budget
`mantissa_bits`. It SHALL return an `ExactStructuralPayload` whose `lhs` contains
`rows * contraction_length` values, whose `rhs` contains the same count in
product mode and is `None` otherwise, and whose operand bound makes every legal
partial sum or product exactly representable under that budget. Non-positive
lengths or rows and a contraction length for which the computed operand bound is
less than one SHALL fail with `ValueError`. Positive integer mantissa budgets
are the supported public domain.

#### Scenario: Decode one payload for both executions

- **WHEN** target and oracle tensors are prepared for one verification case
- **THEN** both value sequences are decoded from the same immutable encoded payload and their recorded input-bit hashes are identical

#### Scenario: Preserve adversarial Float32 identity

- **WHEN** the fixed adversarial payload is decoded and re-encoded
- **THEN** positive and negative zero and each distinct NaN payload retain their original binary32 words

### Requirement: Exact and numerical comparisons expose distinct policies

`float32_bits(value)` SHALL take a Python numeric `value` encodable as binary32
and return its unsigned IEEE-754 binary32 word.
`float32_ulp_distance(expected, actual)` SHALL take binary32-
encodable reference `expected` and produced `actual` values and return their
non-negative ordered binary32 ULP distance. Identical NaN encodings SHALL have
distance zero, distinct NaN
encodings SHALL have maximum distance, and the two zero signs SHALL have ULP
distance zero while remaining distinguishable by exact comparison.

`compare_float32(expected, actual)` SHALL take equal-length iterables of
binary32-encodable reference and target values and return an immutable
`Comparison` containing mismatch count, signed-zero mismatch count, NaN-payload
mismatch count, and maximum absolute, symmetric-relative, and Float32-ULP
deviations. Unequal lengths SHALL fail with `ValueError`; element encoding
uses the supported binary32-encodable input domain. `Comparison(deviations:
Deviations, mismatches: int, signed_zero_mismatches: int,
nan_payload_mismatches: int)` SHALL assemble those immutable fields.

`comparison.within(tolerance)` SHALL take one `Tolerance` and return a Boolean.
It SHALL return `False` for any signed-zero or NaN-payload mismatch or any absent
deviation. Otherwise it SHALL return `True` when `mismatches == 0`, or when the
maximum absolute, symmetric-relative, and ULP deviations are respectively less
than or equal to `tolerance.absolute`, `tolerance.relative`, and
`tolerance.ulps`; it SHALL return `False` for every other comparison.

`gamma_bound(unit_roundoff, terms, sum_absolute_terms)` SHALL take positive
finite numeric accumulator unit roundoff, non-negative integer term count, and
non-negative finite numeric sum of absolute exact terms and return the analytic
gamma-K absolute error envelope. Invalid signs, non-finite inputs, or
`terms * unit_roundoff >= 1` SHALL fail with `ValueError`.

Exact verification classes SHALL compare encoded results bit for bit. Numerical
classes SHALL record absolute, symmetric-relative, and ULP deviations even when
their acceptance criterion uses the versioned absolute analytic envelope.
Stage One floating reductions and matmuls SHALL use twice the gamma-K bound for
two independently associated paths; Stage Two SHALL use the single-path
Float32 gamma-K bound. Integer numerical cases SHALL be exact.

#### Scenario: Refuse tolerance masking of representational differences

- **WHEN** expected and actual results differ only by zero sign or NaN payload
- **THEN** exact comparison records the mismatch and the comparison gate remains failed for every numerical tolerance

#### Scenario: Compare two independently associated sums

- **WHEN** Stage One compares Generic and CPU floating accumulation over the same encoded terms
- **THEN** the evidence records the `stage-one-two-path-gamma-v1` policy with an absolute bound equal to twice the applicable gamma-K envelope

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
