---
title: Kernel Verification
publish: true
status: stable
order: 90
summary: Fail-closed classification and staged local verification of the installed CPU backend.
---

# kernel-verification Specification

## Purpose

Define fail-closed classification and staged local verification of the installed
CPU backend against StrideWeave's Generic reference semantics.

## Terminology

| Term | Meaning |
| --- | --- |
| native kernel | One compiled CPU operation implementation identified by its operation name, stable kernel ID, variant, native Python/pybind binding name, and owning source identity. |
| executable CPU plan | One exact operation plan advertised by the installed CPU carrier under `backend-capabilities`, including operand roles and dtypes, conversions, compute arithmetic, accumulation, accumulator dtype, and output dtype. |
| verification classification | The complete ordered verification-class obligation assigned to one native kernel/variant and refined into an active or deferred disposition for each executable CPU plan. |
| encoded payload | An immutable, content-hashed operand representation from which both target and oracle values are decoded, placing source quantization outside the comparison boundary. |
| Stage One | Local oracle validation that compares the installed CPU implementation with Generic for every active executable CPU plan and issues plan-scoped certificates only for complete passing coverage. |
| Stage One certificate | A digest-backed fact for one exact kernel/variant identifying every certified verification class, every certified executable-plan/class obligation, and the evidence digest from which it was issued. |
| Stage Two | Local target validation of ordinary CPU Float32 movement, reduction, and matmul behavior, with reduction and matmul execution gated by a matching Stage One certificate for the required Float64 oracle plan. |
| recoverable case error | A `RuntimeError` or `ValueError` raised while preparing or executing one independent verification case; it becomes error evidence while other cases continue. |

## Requirements

### Requirement: Native kernels and executable CPU plans have complete classifications

`native_cpu_kernel_manifest()` SHALL accept no inputs and return the installed
native kernels as an immutable tuple of `KernelDescriptor` values.
`classifications_for_kernel(kernel)` SHALL take `kernel`, the native kernel to
classify, and return its ordered verification classes. An unknown kernel ID and
variant SHALL fail with `ValueError` identifying that no classification exists.

`require_complete_classification(kernels)` SHALL take `kernels`, an iterable of
native kernel descriptors, and return an immutable one-to-one pairing between
each descriptor and its classes. A duplicate kernel/variant, a manifest entry
without a classification, or a classification without a manifest entry SHALL
fail with `ValueError` before verification cases run.

`classify_cpu_kernel_plans()` SHALL accept no inputs and return an immutable
tuple of `KernelPlanDescriptor` values covering every executable CPU plan. Each
descriptor SHALL bind one exact native kernel, one exact executable CPU plan,
its ordered required classes, an active or deferred disposition, and a concrete
reason for every deferral. A CPU capability without native metadata or a native
kernel with no executable CPU plan SHALL fail with `ValueError`.

`PlanKey.from_plan_like(plan)` SHALL take `plan`, an `OperationPlan` or
`OperationCapability` exposing `operation`; ordered `operands` whose entries
expose `role.name`, optional `dtype.name`, and optional `convert_to.name`;
`compute.name`; optional `accumulation.name`; optional
`accumulator_dtype.name`; and `output.name`. It SHALL return the immutable
normalized identity of those values.

`VerificationStage`, `VerificationClass`, and `ClassificationDisposition` SHALL
expose the stage, class, and active/deferred enum values used by this spec.
`MOVEMENT_CLASSIFICATIONS` SHALL expose the five movement operation names mapped
to their ordered bit-exact obligation. The exported direct-construction records
SHALL accept these annotated field types:

| Value | Supported constructor fields and result |
| --- | --- |
| `KernelDescriptor` | `operation: str`, `kernel_id: str`, `variant: str`, `pybind_name: str`, and `owning_source: str = ""`; returns one native-kernel identity. |
| `PlanKey` | `operation: str`, `operands: tuple[tuple[str, str | None, str | None], ...]`, `compute: str`, `accumulation: str | None`, `accumulator_dtype: str | None`, and `output: str`; returns one normalized exact-plan identity. |
| `KernelPlanDescriptor` | `kernel: KernelDescriptor`, `plan: PlanKey`, `classes: tuple[VerificationClass, ...]`, `disposition: ClassificationDisposition`, and `deferred_reason: str | None = None`; returns one classified executable-plan obligation. |
| `StageOneResult` | `report: VerificationReport` and `certificates: tuple[OracleCertificate, ...]`; returns the Stage One report/certificate pair. |
| `OracleCertificate` | `kernel_id: str`, `variant: str`, `certified_classes: tuple[VerificationClass, ...]`, `evidence_digest: str`, and `certified_plan_classes: tuple[tuple[PlanKey, tuple[VerificationClass, ...]], ...] = ()`; returns one certificate fact. |

Within those supported types, direct construction SHALL return a shallow-frozen
record whose fields are protected from reassignment and whose nested field
objects retain their supplied identities. The validated factories and staged-
verification functions below SHALL own type, canonicality, content-identity,
and deep-immutability validation.

#### Scenario: Classify the complete installed backend

- **WHEN** every native kernel/variant and every executable CPU plan has exactly one current classification
- **THEN** classification returns the complete exact pairing of manifest entries and current classifications

#### Scenario: Reject stale classification metadata

- **WHEN** the native manifest gains, loses, or duplicates a kernel/variant without the classification set changing with it
- **THEN** completeness validation fails with `ValueError` before Stage One attempts any case

### Requirement: Active and deferred dispositions remain explicit

An active executable CPU plan SHALL name every verification class required for
its Stage One certificate. A deferred executable CPU plan SHALL carry an empty
active-class tuple and a non-empty reason. Stage One SHALL represent every such
plan with exactly one `deferred` outcome.

The current baseline SHALL defer vendor-transcendental accuracy and the floating
`pow` plan that depends on the vendor math library. Exact-integer `pow` SHALL
remain active. Movement subjects `move`, `view`, `permute`, `rearrange`, and
`broadcast_to` SHALL each have an explicit bit-exact case independent of the
native numerical-kernel manifest.

#### Scenario: Report a vendor-transcendental plan

- **WHEN** Stage One reaches an executable CPU plan whose numerical contract is deferred because it uses the vendor math library
- **THEN** it emits deferred evidence carrying the plan and concrete reason while certificate scope includes only active passing obligations

#### Scenario: Split pow by executable plan

- **WHEN** the installed `pow` kernel has both checked-integer and floating executable plans
- **THEN** the integer plan receives its active exact obligation while the floating plan receives its own deferred outcome and reason

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

`run_stage_one(result_transform=None)` SHALL take optional `result_transform`, a
test-only callable that defaults to `None`. When supplied, the callable SHALL
receive the kernel ID string and immutable result tuple and return the
replacement result tuple used for comparison. The function SHALL return a
`StageOneResult` containing an immutable Stage One report and certificate tuple.
With the default, Stage One SHALL attempt every required class for every active
executable CPU plan, emit the explicit movement cases, and emit one deferred
record for every deferred plan.

A `RuntimeError` or `ValueError` raised by the result transform SHALL become
recoverable error evidence for that case.

Stage One exact witnesses SHALL preserve signed zero and exercise deterministic
arbitrary finite encoded inputs. Structural witnesses SHALL use payloads whose
every legal partial result is exactly representable. Analytic witnesses SHALL
use independently named fixed expected results. Numerical witnesses SHALL use
deterministic wide-exponent inputs and the executable plan's accumulator dtype.
Every attempted case SHALL identify its exact plan when it is plan-backed.

A recoverable case error SHALL produce an `error` outcome retaining the prepared
operation, kernel, variant, class, plan, payload hashes, shapes, contraction
length, seed, and versioned tolerance, with unavailable comparison measurements
represented as absent values. Stage One SHALL continue attempting independent
cases after such an error.

A Stage One certificate SHALL be issued for an exact kernel/variant only when
every required class for every active executable plan of that kernel/variant has
passed and all required evidence for that scope has a passing outcome. The
certificate SHALL record the union of certified classes, exact per-plan class
coverage, and a deterministic evidence digest. Missing, failed, errored, or
incomplete required evidence SHALL prevent certificate issuance.

`OracleCertificate.from_records(kernel, required_classes, records, *,
required_plan_classes=())` SHALL take `kernel`, the exact native kernel to
certify; `required_classes`, its complete class union; `records`, the Stage One
evidence to digest; and optional `required_plan_classes`, the exact plan/class
obligations defaulting to the empty tuple. It SHALL return the immutable
certificate when all required evidence passed. Missing or non-passing required
class or plan evidence SHALL fail with `ValueError`.

#### Scenario: Certify every active plan of one kernel

- **WHEN** all required witnesses pass for every active executable CPU plan mapped to one kernel/variant
- **THEN** Stage One emits one certificate whose plan/class scope and evidence digest cover all of those obligations

#### Scenario: Fail closed on one required witness

- **WHEN** one required exact, structural, analytic, or numerical witness fails or errors
- **THEN** its evidence remains in the report, independent cases continue, and certificate scope includes only complete passing kernel/variant obligations

### Requirement: Stage Two runs only behind an exact valid certificate

Stage Two SHALL verify movement subjects and the current Float32 `reduce_sum`
and `matmul` target surface. For `reduce_sum` and `matmul`, authorization SHALL
require a Stage One certificate for the same kernel ID and variant whose
reconstructed evidence digest is valid and whose per-plan scope covers the
active Float64 accumulator plan and all required classes.

Stage Two SHALL execute a reduction or matmul target case only after the exact
certificate authorization succeeds. When authorization is absent, forged,
inconsistent, or incomplete, it SHALL instead emit separate `blocked`
structural and numerical records containing the prepared case identity, zero
absolute, relative, and ULP deviations, zero mismatches, and a diagnostic that
explains the missing or invalid certificate. Movement evidence SHALL remain
independent of those certificates.

Authorized structural target cases SHALL require encoded Float32 bit identity.
Authorized numerical target cases SHALL compare ordinary Float32 accumulation
with the Float64-accumulator oracle under the versioned single-path Float32
gamma-K absolute envelope. The target catalog SHALL include flat and
hierarchical multi-output contraction layouts and SHALL record effective operand
shapes and contraction length. Recoverable target errors SHALL become complete
`error` evidence, and Stage Two SHALL continue every independent case.

#### Scenario: Block a target without Float64 oracle scope

- **WHEN** a certificate matches the kernel and variant but omits the active Float64 accumulator plan or one of its required classes
- **THEN** Stage Two selects the blocked structural and numerical evidence path in place of affected target execution

#### Scenario: Run an authorized hierarchical contraction

- **WHEN** a matching reconstructed certificate covers the required Float64 oracle plan and classes
- **THEN** Stage Two executes the hierarchical Float32 target and Float64 oracle, records their shared input hashes, effective shapes, contraction length, deviations, tolerance, and passed or failed outcome

### Requirement: test_backend is local, deterministic, and optionally writes one report

`test_backend(output=None)` SHALL take `output`, an optional filesystem path for
the complete verification JSONL; it SHALL accept a string or filesystem
path-like value and default to `None`. It SHALL first
require a compatible installed native verification binding, run Stage One, run
Stage Two using that Stage One result, bind the combined evidence and
certificates into one `VerificationReport`, optionally replace `output` with the
report's deterministic UTF-8 JSONL, and return the same immutable report model.

When the installed native extension lacks the required verification binding,
the call SHALL fail with `RuntimeError` before Stage One and identify the command
needed to rebuild the active environment. Errors raised by a present native
binding SHALL propagate unchanged. A failure replacing `output` SHALL remain observable as the
corresponding filesystem I/O error.

With `output=None`, the call SHALL return the bound report as its only produced
artifact. With a path, it SHALL additionally replace exactly that destination
with the report bytes. In all modes its inputs SHALL be the installed
verification binding and deterministic local verification facts; CI state,
source Git history, wall-clock time, evidence databases, status stores,
contributor exchange, and network resources are therefore outside this call's
execution inputs.

#### Scenario: Verify without persistence

- **WHEN** `test_backend()` runs against a compatible installed CPU backend
- **THEN** it produces exactly the complete deterministic returned report as its sole artifact

#### Scenario: Write the same returned report

- **WHEN** `test_backend(output=path)` succeeds
- **THEN** it returns the report and replaces `path` with UTF-8 bytes equal to `report.to_jsonl()`

### Requirement: Deferred verification domains remain explicit boundaries

The local kernel-verification capability SHALL treat vendor-transcendental and
floating-`pow` accuracy, autograd certification, actual JIT-framework adapters,
confidence or risk ranking, autotuning, unsupported compiler-provenance
providers, and CI integration as deferred domains. Their absence SHALL remain
visible through deferred evidence or documented capability boundaries rather
than being represented as passing verification.

Local verification SHALL read installed verification inputs, perform its staged
case execution, return its immutable report, and optionally write only the
caller-supplied report path. A separate evidence-tracking capability SHALL own
evidence persistence, producer observations, stored status/staleness/todo
queries, publication, refresh, and contributor exchange.

#### Scenario: Inspect a local report with deferred work

- **WHEN** the installed backend contains a plan whose verification domain is explicitly deferred
- **THEN** the report retains a deferred record and returns it to the caller while the separate evidence-tracking capability remains the owner of persistence, publication, refresh, and confidence ranking
