# SimpleDType Operation Policy — v0.2

**Status:** current normative specification for simple-dtype operation planning.
**Stability:** *intentional starting point, not a compatibility promise.* Every rule
here is expected to be revisable as the GPU backends land and as users report what
they actually need. Section 11 defines how a revision is made. Nothing in this
document should be read as a permanent guarantee to callers.

This document specifies *policy only*. It does not implement the resolver; that is
`main-28v.6`. It exists so the resolver, `Generic`, native `CPU`, and future CUDA
and mROC kernels all encode one decision rather than rediscovering semantics in
backend code.

---

## 1. What this policy decides

For one operation invocation with known operand dtypes, this policy decides:

1. which operands are converted, and to what, before any arithmetic;
2. what arithmetic the per-element computation performs;
3. what arithmetic combines terms, for the operations that combine terms;
4. what dtype the result carrier reports, from which autograd eligibility
   follows by the framework's existing floating-dtype rule;
5. and, where the combination is not supported, which error is raised.

It does *not* decide layout compatibility, carrier compatibility, gradient
formulas, or kernel scheduling. Those remain each operation's business. The policy
constrains the dtype, rounding, and overflow behavior those formulas must exhibit.

---

## 2. Scope boundaries

### 2.1 In scope: concrete simple dtypes

The planner accepts `SimpleDType` operands only. In v0.1 the supported set is
exactly the two dtypes carriers store and kernels implement:

- `DType.Float32`
- `DType.Int32`

### 2.2 Registered but unsupported simple encodings

`Int8`, `E8M0`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, and `E2M1` are registered
structural descriptors with no carrier storage and no kernel. Passing one to the
planner is an explicit `NotImplementedError`, never a promotion to `Float32`.
Silently widening an unimplemented narrow encoding would be the exact
backend-local guesswork this policy exists to remove.

### 2.3 Out of scope: legacy opaque categories

`DType.Any` and `DType.Floating` are `DTypeCategory` descriptors carrying the
legacy opaque-storage disposition (`RT012`). They are **not** simple dtypes and
have no entry in this matrix. The planner rejects them with a message naming the
legacy disposition explicitly.

`Generic`'s existing `Any`/`Floating` arithmetic is unchanged by this policy and
continues to run on its own legacy path. A tensor pairing legacy opaque storage
with concrete `Float32`/`Int32` storage is **not** silently planned: the mixed
case is a documented legacy behavior, not a promotion result. This is the direct
answer to the earlier review finding that `Any` must not masquerade as a checked
integer.

### 2.4 Out of scope: compound dtypes

`CompoundDType` (including `BlockScaledDType`) requires representation-aware
operation planning — element and scale planes, tilers, quantization rules — none
of which the simple planner models. Every compound operand raises
`NotImplementedError` naming the deferred capability, using the same wording shape
as `validate_storage_dtype`'s carrier-side rejection so users see one consistent
story about what is deferred.

### 2.5 Out of scope: representation-preserving operations

`view`, `permute`, `rearrange`, and `move` neither promote nor compute. They
preserve the operand's dtype exactly, including legacy opaque and (eventually)
compound dtypes. They are deliberately **not** registered in the operation-policy
registry: registering them would imply the planner has an opinion about them and
would pollute exhaustive conformance enumeration with entries that assert nothing.

---

## 3. Operand normalization

### 3.1 Tensor operands

A tensor operand contributes its carrier storage dtype, matched by identity
(`SW002`). No equality comparison, no name lookup, no duck typing.

### 3.2 Weak Python scalars

Two operations take a Python scalar rather than a tensor: `mul` (the multiplier)
and `pow` (the exponent). Such a scalar is *weak*: it has no dtype of its own and
never forces a width, it only selects between plans.

Normalization, in order:

| Python value | Normalized kind | Notes |
| --- | --- | --- |
| `bool` (`True`/`False`) | **weak float** | Provisional; see §12.1 |
| `int` (non-`bool`, `numbers.Integral`) | **weak integer** | |
| `float` and other `numbers.Real` | **weak float** | |
| anything else, including `complex` | — | `TypeError` |

A weak scalar is materialized **once, before the kernel loop**, into the plan's
compute arithmetic, and the same materialized value is used by both forward and
backward. Specifically:

- into `BINARY32`: rounded to binary32 with round-to-nearest-even. A caller's
  `0.1` is `float(numpy.float32(0.1))` for every backend, in forward *and* in
  backward. Backends must not carry the Python `float` into backward and round
  only in forward; that mismatch was a confirmed review finding.
- into `INT32_EXACT_CHECKED`: required to be exactly representable in `Int32`.
  A weak integer outside `[-2**31, 2**31 - 1]` raises `OverflowError` at plan
  resolution, independent of the tensor's values. This is value-independent on
  purpose: a plan must be resolvable before the data is read.

---

## 4. Arithmetic semantics

The plan names an arithmetic, not merely a width. v0.1 defines exactly three.

### 4.1 `BINARY32`

IEEE-754 binary32 throughout:

- round-to-nearest-even on every primitive operation and on every conversion into
  binary32;
- no FMA contraction, no reassociation, no wider intermediate precision — a
  backend that computes in double and rounds once at the end is **not**
  conforming, because it produces different results for the double-rounding cases
  the conformance suite pins;
- IEEE special values propagate rather than raise: `1.0 / 0.0` is `+inf`,
  `0.0 / 0.0` is `NaN`, `pow` follows IEEE-754 `pow` at its singularities, and
  signed zero is preserved. A backend must not raise `ZeroDivisionError` or a
  Python floating-point exception at a supported singularity, in forward or in
  backward. This applies to `Generic` in particular: it must scope its
  floating-point error state around the operation loop rather than per element,
  and must not let Python's `float` semantics turn an IEEE result into an
  exception.

`Int32 → Float32` conversion is round-to-nearest-even and is therefore lossy above
`2**24`. That loss is intentional and pinned, not an accident to be worked around.

### 4.2 `INT32_EXACT_CHECKED`

Evaluate over the mathematical integers exactly, then narrow to `Int32`:

- the exact result is computed as if in unbounded integers;
- if the exact result lies outside `[-2**31, 2**31 - 1]`, raise `OverflowError`;
- wrapping, saturation, and undefined behavior are all non-conforming.

A backend using a fixed-width intermediate (native CPU computes one element in
`int64`, which represents every `Int32` sum, difference, and product exactly)
conforms as long as it raises `OverflowError` rather than wrapping when its own
intermediate cannot represent the exact value. See §5.2 for the accumulation
consequence, where a fixed width is a harder constraint.

### 4.3 `INT32_EXACT`

Exact integer arithmetic that provably cannot leave `Int32`, so no check is
required. In v0.1 this is used by `relu` alone, which selects between an existing
element and zero and therefore cannot overflow.

---

## 5. Combining terms

### 5.1 Why accumulation is a separate axis

`reduce` and `matmul` combine many terms. Their observable result depends on the
*order and width* of that combination, not only on the element arithmetic, and the
conformance suite must pin it. Elementwise operations combine nothing, so their
plans carry no accumulation at all.

### 5.2 The two accumulation rules

**`SEQUENTIAL_BINARY32`** — initialize to `+0.0`, then add each term in ascending
logical index order over the reduced mode (`reduce`) or the contracted mode
(`matmul`), each addition rounding in binary32. Order is normative: pairwise or
blocked summation is a different, non-conforming result.

**`EXACT_INTEGER`** — accumulate exactly over the mathematical integers, then
narrow the final sum to `Int32` with the `INT32_EXACT_CHECKED` check. Two
consequences are deliberate:

- an intermediate partial sum **may** leave `Int32` range as long as the final sum
  does not, so `[2**31 - 1, 2**31 - 1, -(2**31 - 1)]` reduces successfully;
- a backend whose accumulator is a fixed width must raise `OverflowError` rather
  than wrap, **and must be wide enough that it never rejects a sum whose terms
  cancel**. `matmul` makes the second requirement concrete: its terms are
  products, and three products of `2**31 - 1` already exceed `int64`, so an
  `int64` accumulator would reject a contraction whose later terms cancel back
  into `Int32` — a result `Generic` computes. Native CPU therefore accumulates
  in an exact 128-bit signed value, which is exact for any contraction of fewer
  than `2**64` products, beyond what any layout can address. `Generic`
  (unbounded Python integers) and CPU agree on every tensor that can exist.

---

## 6. The `OperationPlan` contract

Derived from the distinctions §7's matrix actually makes — no field exists for a
distinction v0.1 never draws.

```python
class OperandRole(Enum):
    TENSOR = "tensor"
    WEAK_SCALAR = "weak_scalar"

class Arithmetic(Enum):
    BINARY32 = "binary32"                        # §4.1
    INT32_EXACT_CHECKED = "int32_exact_checked"  # §4.2
    INT32_EXACT = "int32_exact"                  # §4.3

class Accumulation(Enum):
    SEQUENTIAL_BINARY32 = "sequential_binary32"  # §5.2
    EXACT_INTEGER = "exact_integer"              # §5.2

@dataclass(frozen=True, slots=True)
class OperandPlan:
    role: OperandRole
    dtype: SimpleDType | None   # storage dtype; None for a weak scalar
    convert_to: SimpleDType     # dtype this operand is materialized into

@dataclass(frozen=True, slots=True)
class OperationPlan:
    operation: str
    operands: tuple[OperandPlan, ...]
    compute: Arithmetic
    accumulation: Accumulation | None
    output: SimpleDType
```

Plans are immutable and hashable, so a backend may cache one per
`(operation, operand dtypes)` key without defensive copying.

Field-by-field justification:

- **`operation`** — plans are compared and enumerated by operation in fixtures and
  error messages.
- **`operands`** — carries the per-operand conversion. This is a real distinction:
  `add(Float32, Int32)` converts one operand and not the other, and the activation
  entries convert an `Int32` input to `Float32`. It also records what the plan was
  resolved *from*, which is what makes exhaustive fixture comparison meaningful.
- **`compute`** — the element arithmetic. Not derivable from `output`: `relu` on
  `Int32` is `INT32_EXACT` while `add` on `Int32` is `INT32_EXACT_CHECKED`, and
  both output `Int32`.
- **`accumulation`** — `None` for every elementwise entry, set only for `reduce`
  and `matmul`. A field that is absent for most entries earns its place by
  distinguishing them.
- **`output`** — the dtype the result carrier reports. Consumed at allocation time,
  and the sole source of the result's storage narrowing; a backend must not
  re-derive it from `compute`.

**A removed field (v0.2):** `differentiable` was a plan field in v0.1, defined as
`output is DType.Float32` and justified as a policy claim that a future
non-differentiable float dtype could separate from the arithmetic. It was removed
because nothing consumed it: the tensor layer decides differentiability from a
tensor's dtype, for every tensor, whether or not a plan produced it. A duplicated
derivation no backend reads is a drift risk, not a contract. The "diff." column of
§7's tables still documents the resulting behavior; it describes the result, not a
field. If a non-differentiable floating dtype is ever registered, the rule that
changes is the tensor layer's, and that is where the distinction belongs.

**A deliberately absent field:** there is no separate accumulator *dtype*. In v0.1
no entry accumulates in a dtype the `Accumulation` value does not already imply,
so a second dtype field would be unjustified. §12.2 records the alternative.

**A deliberately absent distinction:** `compute` never disagrees with `output`
about width in v0.1, because there is no mixed-width compute yet. That equality is
asserted by the fixtures as a *derived observation*, not assumed by the resolver —
a mixed-precision revision changes the matrix without changing the plan's shape.

---

## 7. The v0.1 operation policy matrix

`F32` = `DType.Float32`, `I32` = `DType.Int32`. "conv." lists non-identity operand
conversions. Every listed entry is supported; anything not listed is §8's error.

### 7.1 Elementwise binary — `add`, `sub`, `elementwise_mul`

| lhs | rhs | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 | — | `BINARY32` | — | F32 | yes |
| F32 | I32 | rhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | F32 | lhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | I32 | — | `INT32_EXACT_CHECKED` | — | I32 | no |

### 7.2 Elementwise binary — `div`

Division is always floating; v0.1 has no integer-division semantics.

| lhs | rhs | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 | — | `BINARY32` | — | F32 | yes |
| F32 | I32 | rhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | F32 | lhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | I32 | both→F32 | `BINARY32` | — | F32 | yes |

Division by zero is IEEE (§4.1), including in backward.

### 7.3 Tensor × weak scalar — `mul`

| tensor | scalar | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | weak int | scalar→F32 | `BINARY32` | — | F32 | yes |
| F32 | weak float | scalar→F32 | `BINARY32` | — | F32 | yes |
| I32 | weak int | scalar→I32 | `INT32_EXACT_CHECKED` | — | I32 | no |
| I32 | weak float | tensor→F32, scalar→F32 | `BINARY32` | — | F32 | yes |

A weak integer outside `Int32` range with an `I32` tensor raises at resolution
(§3.2). A weak integer above `2**24` with an `F32` tensor rounds (§4.1).

### 7.4 Tensor × weak scalar — `pow`

The exponent preserves an integer result only when it is a weak integer in
`[0, 2**31 - 1]`. A negative exponent produces reciprocals, and a non-integral one
produces roots; neither is closed over `Int32`. The upper bound is central policy,
not a native limit, so every backend agrees on where the floating path begins.

| tensor | exponent | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | any weak scalar | exp→F32 | `BINARY32` | — | F32 | yes |
| I32 | weak int in `[0, 2**31 - 1]` | — | `INT32_EXACT_CHECKED` | — | I32 | no |
| I32 | weak int `< 0` or `> 2**31 - 1` | tensor→F32, exp→F32 | `BINARY32` | — | F32 | yes |
| I32 | weak float | tensor→F32, exp→F32 | `BINARY32` | — | F32 | yes |

`pow` follows IEEE-754 `pow` at its singularities (`0 ** 0` is `1.0`,
`0 ** -1` is `+inf`), in forward and in backward.

### 7.5 Unary elementwise — always-floating activations

Applies to `exp`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`, and
`leaky_relu`.

| input | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- |
| F32 | — | `BINARY32` | — | F32 | yes |
| I32 | input→F32 | `BINARY32` | — | F32 | yes |

The "diff." column describes the *result*: a floating result may participate in
autograd. A node is still attached only when at least one input is itself
differentiable, so an `I32` input yields a `Float32` result with no graph — the
existing rule in `README.md`, unchanged.

### 7.6 Unary elementwise — `relu`

`relu` preserves its input dtype; selecting between an element and zero cannot
overflow.

| input | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- |
| F32 | — | `BINARY32` | — | F32 | yes |
| I32 | — | `INT32_EXACT` | — | I32 | no |

### 7.7 Reduction — `reduce` (sum over the second mode)

| input | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- |
| F32 | — | `BINARY32` | `SEQUENTIAL_BINARY32` | F32 | yes |
| I32 | — | `INT32_EXACT_CHECKED` | `EXACT_INTEGER` | I32 | no |

### 7.8 Contraction — `matmul`

Category promotion matches §7.1; the accumulation differs because many terms
combine.

| lhs | rhs | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 | — | `BINARY32` | `SEQUENTIAL_BINARY32` | F32 | yes |
| F32 | I32 | rhs→F32 | `BINARY32` | `SEQUENTIAL_BINARY32` | F32 | yes |
| I32 | F32 | lhs→F32 | `BINARY32` | `SEQUENTIAL_BINARY32` | F32 | yes |
| I32 | I32 | — | `INT32_EXACT` | `EXACT_INTEGER` | I32 | no |

The `I32 × I32` element compute is `INT32_EXACT` rather than
`INT32_EXACT_CHECKED`: an individual product is not required to fit `Int32`, only
the final narrowed sum is, so checking each product would reject contractions whose
terms legitimately cancel.

---

## 8. Errors

Every unsupported case has exactly one documented disposition. None of them fall
back to a guessed plan.

| Condition | Error | Raised at |
| --- | --- | --- |
| Unregistered operation name | `NotImplementedError` | resolution |
| Operand is not a `DType` | `TypeError` | resolution |
| Operand is `Any`, `Floating`, or `Integer` | `TypeError`, naming the legacy opaque disposition and that legacy `Generic` arithmetic is a separate path (§2.3) | resolution |
| Operand is a `CompoundDType` | `NotImplementedError`, naming the deferred per-plane capability (§2.4) | resolution |
| Operand is a registered but unimplemented `SimpleDType` | `NotImplementedError`, naming the supported set (§2.2) | resolution |
| Weak scalar is not a supported Python number | `TypeError` | resolution |
| Weak integer outside `Int32` range in an integer plan | `OverflowError` | resolution |
| Exact integer result outside `Int32` range | `OverflowError` | execution |
| Fixed-width accumulator cannot hold an exact partial sum | `OverflowError` | execution |

Resolution-time errors are value-independent: a plan resolves — or fails — before
any element is read, which is what lets native CPU resolve once while holding the
GIL and then release it (`CPP001`).

---

## 9. Backward policy

The policy fixes the dtype and rounding of backward, not the gradient formulas.

- Every gradient tensor is `Float32` and every backward computation uses
  `BINARY32`, including its accumulations, which follow §5.2's ordering rule.
- Gradients are produced only for operands whose storage dtype is `Float32`. An
  `Int32` operand of a `Float32`-output operation receives none; `Int32` tensors
  reject the gradient APIs entirely.
- An operation whose plan has a non-floating `output` attaches no autograd node and
  raises on `backward()`, as today.
- A weak scalar's backward uses the *same* binary32-rounded value the forward used
  (§3.2).
- IEEE singularities propagate in backward exactly as in forward (§4.1): a
  division-by-zero backward, a zero-base `pow` backward, and a `pow` backward with
  a negative exponent all produce IEEE values rather than exceptions.

These five rules are precisely the earlier review's confirmed backward findings,
promoted from bug fixes to policy so a backend cannot regress them quietly.

---

## 10. Conformance obligations

The resolver is the single executable statement of this policy. It follows that:

1. **No backend defines promotion or result-dtype policy locally.** `Generic`,
   native `CPU`, and every future backend resolve a plan and execute it. A native
   promotion table is a policy fork and is not permitted.
2. **`Generic` is the behavioral reference.** Where a documented rule and a
   `Generic` result disagree, `Generic` is wrong. Where `CPU` and `Generic`
   disagree and both satisfy the rule, the rule is underspecified and this
   document must be amended.
3. **Coverage is enumerated, not sampled.** The operation registry lists every
   planned operation with its operand roles, so the conformance suite can
   enumerate registered coverage exhaustively and compare it against explicit
   expected-plan fixtures. An operation added without a fixture fails the
   enumeration rather than passing silently.
4. **Exactness is the default.** Forward and backward results are compared
   bit-exactly. A tolerance is admissible only for a transcendental whose
   difference is a justified libm divergence, and each such tolerance is named and
   justified individually at its assertion.
5. **A backend may refuse, but may not differ.** A backend that cannot execute a
   plan raises; it never substitutes a nearby plan it can execute. Which plan
   shapes a backend does execute is recorded, per exact carrier class, in
   `strideweave.carriers.operation_capability` — an implementation-reach
   registry, not a second policy: it accepts or refuses an already-resolved
   plan and never selects one. The entries execution is accepted against are the
   entries capability introspection reports, so a backend cannot advertise a
   shape it would refuse or run one it never declared.

---

## 11. Revising this policy

A policy change is one change touching all of:

1. this document, with its version heading incremented;
2. the resolver, so the executable policy matches;
3. the resolver's expected-plan fixtures;
4. the Generic/CPU conformance expectations that pinned the old behavior;
5. `README.md` where the change is user-visible, and `INVARIANTS.md` where it
   alters a cross-cutting rule.

A change that lands in a backend first is a policy fork and should be rejected in
review regardless of its merits. The fixtures are intended to be *easy to
re-baseline deliberately* and *impossible to drift into*: they are explicit
expected values, not assertions recomputed from the resolver.

---

## 12. Provisional choices and open questions

Flagged rather than settled. Each is implementable as written; each is a plausible
thing for a user or a GPU backend to change.

**12.1 `bool` is a weak float.** `sw.mul(int32_tensor, True)` returns `Float32`
`3.0`, not `Int32` `3`. This is existing pinned behavior
(`tests/test_cpu.py::test_cpu_bool_scalar_multiplies_as_float`), preserved
deliberately rather than by accident. The defensible alternative is to reject
`bool` outright: promoting `True` to `1.0` and silently changing the result dtype
is the kind of accident a central policy exists to prevent. Deferred because it is
a user-visible break with no current demand.

**12.2 No accumulator dtype field.** Justified in §6 for v0.1. Mixed-precision
accumulation (bf16 storage, fp32 accumulate) is the obvious future case that
splits `Accumulation` into a rule plus a dtype. Adding the field then is a
mechanical widening of the plan; adding it now would be an unused dimension.

**12.3 `div` has no integer path.** `I32 / I32` yields `Float32`. Neither
truncating nor flooring integer division is offered, because picking one silently
is worse than not offering one. A future explicit `floor_div` operation is the
expected resolution, not a change to `div`.

**12.4 Mixed-width compute is unrepresented.** Every v0.1 entry computes at the
output width. The plan shape anticipates the split (§6) but the matrix does not
exercise it.

**12.5 `SEQUENTIAL_BINARY32` will constrain GPUs.** A strictly ordered binary32
accumulation is not what a parallel reduction wants. This is the rule most likely
to be revised once a CUDA backend exists — probably into an explicitly
order-unspecified accumulation with a documented tolerance, chosen centrally here
rather than assumed by the first backend that finds sequential summation
inconvenient.

**12.6 The `pow` integer bound is policy, not a limit.** `2**31 - 1` is chosen so
every backend switches to the floating path at the same exponent. It is not
derived from any backend's representable range.

---

## 13. What implementation must add

Recorded here so the implementing tasks do not have to rediscover it, and so this
document does not claim enforcement that does not exist yet.

- **`INVARIANTS.md` carries this policy as `RT013`,** added with the resolver.
  Its resolver clauses are Test-enforced by the expected-plan fixtures; its
  backend clauses are Review-enforced. The entry carried an adoption boundary
  while the carriers still applied their own rules; both `Generic` and native
  `CPU` now resolve and execute plans, and the boundary has been removed from
  the entry rather than left to go stale.
- **`README.md`** documents delivered behavior: its dtype and operation sections
  describe the promotion both carriers now deliver, including the one behavior
  the routing changed — a float-valued exponent no longer preserves an `Int32`
  `pow` result on native `CPU`.
- **The operation registry** (§10.3) is part of the resolver, not of this
  document. It must list operand roles per operation so coverage enumeration is
  derived rather than hand-maintained.
