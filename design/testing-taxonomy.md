# Staged Kernel Correctness Taxonomy

Version: v0.1 (database-independent PR 1 foundation)

This document defines the correctness boundary used by StrideWeave's staged
kernel verification. It covers local evidence and certificates only. CI status,
Dolt persistence, toolchain identity, transitive closure hashes, and autotuning
records are deliberately deferred.

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

## 5. Evidence and certificates

Each attempted case emits one versioned JSON object containing stage, class,
explicit operation, kernel and variant, the exact classified plan when applicable, dtypes, shapes,
accumulator, contraction length, seed and case ID, tolerances, deviations,
mismatch count, outcome, diagnostic, and input hashes. This unmerged v1 format
adds the explicit operation field without a schema-version bump; released
schemas will receive normal versioned migration treatment. JSON uses sorted keys
and standard finite values only. A certificate is issued only when every required
class for every active plan of its kernel/variant has passed and no required
record failed, blocked, errored, or remained deferred.

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

The local `test_backend(output=None)` entry point reruns both stages and may
write their combined deterministic JSONL evidence. PR 1 deliberately provides
no CI integration, Dolt persistence, closure or toolchain hashing, status
aggregation, or autotune cache; those facilities belong to PR 2. Hypothesis
generation, empirical tolerance calibration, and quantizer validation are also
outside the v0 verifier.
