# SimpleDType Operation Policy — v0.3

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

The planner accepts concrete simple dtype operands only. The v0 operation set
uses the three dtypes carriers store and kernels implement:

- `DType.Float32` for numeric computation and differentiable results;
- `DType.Int32` for checked integer storage and index results; and
- `DType.Bool` for masked conditions and predicate/logical results.

Bool is deliberately a narrow domain: it is not implicitly promoted into
Float32 or Int32 arithmetic. Ordinary numeric operations reject Bool operands;
`select` alone consumes a Bool tensor as its condition, while predicates and
`logical_not` produce Bool results.

`DType.Float64` is additionally registered as an accumulator-only descriptor.
It may appear in `OperationPlan.accumulator_dtype`, but it is not a tensor
operand, output, or carrier storage dtype.

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
with concrete simple storage is **not** silently planned: the mixed case is a
documented legacy behavior, not a promotion result. This is the direct answer to
the earlier review finding that `Any` must not masquerade as a checked integer.

### 2.4 Out of scope: compound dtypes

`CompoundDType` (including `BlockScaledDType`) requires representation-aware
operation planning — element and scale planes, tilers, quantization rules — none
of which the simple planner models. Every compound operand raises
`NotImplementedError` naming the deferred capability, using the same wording shape
as `validate_storage_dtype`'s carrier-side rejection so users see one consistent
story about what is deferred.

### 2.5 Out of scope: representation-preserving operations

`slice`, `as_strided`, `reshape`, `view`, `permute`, `rearrange`, broadcast
views, and `move` neither promote nor compute. They
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

Three operations take Python scalars as optional weak operands: `mul` (the
multiplier), `pow` (the exponent), and `clamp` (either bound). Such a scalar is
*weak*: it has no dtype of its own and never forces a width; it only selects
between plans.

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

The plan names an arithmetic, not merely a width. v0.3 defines exactly three.

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
required. In v0.3 this is used by `relu` alone, which selects between an existing
element and zero and therefore cannot overflow.

---

## 5. Combining terms

### 5.1 Why accumulation is a separate axis

`reduce_*`, `cumsum`, `matmul`, `conv_general`, and `scatter_add` combine many
terms. Their observable result depends on the *domain, width, and order* of that
combination, not only on the element arithmetic, and the conformance suite must
pin whichever of those the operation makes observable. Elementwise operations
combine nothing, so their plans carry no accumulation at all.

The policy therefore separates two questions. `reduce_sum` and `matmul` expose
only the finished sum, so the plan pins the accumulator's arithmetic domain and
dtype and leaves traversal and association order to the backend; parallel kernels
are not forced into a sequential schedule, and `accumulator_dtype` is the one
knob a caller may turn. Every other combining operation makes its per-term
ordering observable — `cumsum` publishes each prefix, `conv_general` and
`scatter_add` pin a reference schedule, and the product, extrema, and arg
reductions pin NaN, signed-zero, and first-winner rules — so their accumulation
stays fully normative and declares no `accumulator_dtype`.

### 5.2 The accumulation rules

**`FLOATING`** — combine terms in the plan's `accumulator_dtype`. The default is
`DType.Float32`; an explicit `DType.Float64` request widens already-encoded input
values before accumulation and narrows once to the unchanged planned output dtype.
For matmul, each product is still formed in the plan's binary32 compute arithmetic
before that encoded term widens into the accumulator.
The backend chooses traversal and association order. A capability therefore claims
the floating kind and accumulator dtype, not one summation schedule. `reduce_sum`
and `matmul` are the only operations that use it.

**`SEQUENTIAL_BINARY32`** — initialize to `+0.0`, then add each term in ascending
logical index order over the operation's combined mode, each addition rounding in
binary32. Order is normative: pairwise or blocked summation is a different,
non-conforming result. `cumsum`, `conv_general`, and `scatter_add` use it, and it
declares no `accumulator_dtype`.

**`SEQUENTIAL_BINARY32_PRODUCT`, `MAXIMUM`, `MINIMUM`, `ARGMAX`, `ARGMIN`** — the
ordered product and the extrema and arg reductions, each applying its pinned
Float32 NaN, signed-zero, and first-winner rules in ascending logical index
order. They declare no `accumulator_dtype`.

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
distinction v0.3 never draws.

```python
class OperandRole(Enum):
    TENSOR = "tensor"
    WEAK_SCALAR = "weak_scalar"

class Arithmetic(Enum):
    BINARY32 = "binary32"                        # §4.1
    INT32_EXACT_CHECKED = "int32_exact_checked"  # §4.2
    INT32_EXACT = "int32_exact"                  # §4.3

class Accumulation(Enum):
    FLOATING = "floating"                        # §5.2
    SEQUENTIAL_BINARY32 = "sequential_binary32"  # §5.2
    SEQUENTIAL_BINARY32_PRODUCT = "sequential_binary32_product"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
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
    accumulator_dtype: SimpleDType | None
    output: SimpleDType

@dataclass(frozen=True, slots=True)
class OperationOverload:
    roles: tuple[OperandRole, ...]
    rule: Callable[..., OperationPlan]
    tensor_domains: tuple[tuple[SimpleDType, ...], ...] | None = None

@dataclass(frozen=True, slots=True)
class OperationSpec:
    name: str
    overloads: tuple[OperationOverload, ...]
    public: bool = True
    dtype_operand_positions: tuple[int, ...] | None = None
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
- **`accumulation`** — `None` for elementwise, predicate, gather, overwrite,
  and ordering entries. Reductions, scans, contraction, and additive scatter
  name their combination rule explicitly, and every rule but `FLOATING` fixes
  the combination order as well.
- **`accumulator_dtype`** — `Float32` or `Float64` for `FLOATING`; `None` for
  non-accumulating operations, exact mathematical-integer accumulation, and
  every order-normative floating rule. It is independent of `output`: widened
  accumulation still stores the policy's original result dtype.
- **`output`** — the dtype the result carrier reports. Consumed at allocation time,
  and the sole source of the result's storage narrowing; a backend must not
  re-derive it from `compute`.

**A removed field (v0.2):** `differentiable` was a plan field in v0.2, defined as
`output is DType.Float32` and justified as a policy claim that a future
non-differentiable float dtype could separate from the arithmetic. It was removed
because nothing consumed it: the tensor layer decides differentiability from a
tensor's dtype, for every tensor, whether or not a plan produced it. A duplicated
derivation no backend reads is a drift risk, not a contract. The "diff." column of
§7's tables still documents the resulting behavior; it describes the result, not a
field. If a non-differentiable floating dtype is ever registered, the rule that
changes is the tensor layer's, and that is where the distinction belongs.

**A deliberately absent distinction:** `compute` never disagrees with `output`
about width in v0.3, because there is no mixed-width element compute yet. That
equality is asserted by the fixtures as a *derived observation*, not assumed by
the resolver — a mixed-precision revision changes the matrix without changing the
plan's shape. `accumulator_dtype` is not that distinction: it widens only the
combination of already-encoded terms, and §12.2 records why it is an execution
option rather than a storage or compute change.

`compute` and `output` are deliberately independent. Predicate plans consume
binary32 values and store Bool, while arg-reduction and index-result selection
consume binary32 values and store Int32. Backends therefore validate exact plan
shapes per operation; they never apply a global `compute == output` shortcut.

The registry contains one `OperationSpec` per dispatch name and one or more
immutable overloads per spec. Overload signatures are unique and selected by
the exact tensor/weak-scalar role pattern. Per-overload tensor domains make
exhaustive capability enumeration precise for Float32-only primitives. Axis,
shape, ordering flags, and other non-dtype parameters are operation semantics,
not weak scalars, and never enter the dtype resolver. The optional
`dtype_operand_positions` field identifies that planned subset of the full
forward call centrally; without it every call argument is a policy operand and
an extra argument remains an arity error. Internal single-output
selection operations are registered with `public=False`; public `sort` and
`topk` package their value/index calls without pretending a plan has two outputs.

---

## 7. The v0.3 operation policy matrix

`F32` = `DType.Float32`, `F64` = `DType.Float64`, and `I32` = `DType.Int32`.
"conv." lists non-identity operand
conversions. Every listed entry is supported; anything not listed is §8's error.

### 7.1 Elementwise binary — `add`, `sub`, `elementwise_mul`

| lhs | rhs | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 | — | `BINARY32` | — | F32 | yes |
| F32 | I32 | rhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | F32 | lhs→F32 | `BINARY32` | — | F32 | yes |
| I32 | I32 | — | `INT32_EXACT_CHECKED` | — | I32 | no |

### 7.2 Elementwise binary — `div`

Division is always floating; v0.3 has no integer-division semantics.

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

### 7.7 Reduction — `reduce_sum` (sum over the second mode)

| input | conv. | compute | accum. | accum. dtype | output | diff. |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | — | `BINARY32` | `FLOATING` | F32 default; F64 explicit | F32 | yes |
| I32 | — | `INT32_EXACT_CHECKED` | `EXACT_INTEGER` | — | I32 | no |

### 7.8 Contraction — `matmul`

Category promotion matches §7.1; the accumulation differs because many terms
combine.

| lhs | rhs | conv. | compute | accum. | accum. dtype | output | diff. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 | — | `BINARY32` | `FLOATING` | F32 default; F64 explicit | F32 | yes |
| F32 | I32 | rhs→F32 | `BINARY32` | `FLOATING` | F32 default; F64 explicit | F32 | yes |
| I32 | F32 | lhs→F32 | `BINARY32` | `FLOATING` | F32 default; F64 explicit | F32 | yes |
| I32 | I32 | — | `INT32_EXACT` | `EXACT_INTEGER` | — | I32 | no |

The `I32 × I32` element compute is `INT32_EXACT` rather than
`INT32_EXACT_CHECKED`: an individual product is not required to fit `Int32`, only
the final narrowed sum is, so checking each product would reject contractions whose
terms legitimately cancel.

### 7.9 Float32 unary and binary primitives

The v0 Float32-only unary primitives (`neg`, `abs`, `sign`, `recip`, `sqrt`,
`rsqrt`, `exp2`, `log`, `log2`, `sin`, `cos`, `erf`, `floor`, `ceil`, and
`round`) resolve one F32 tensor to `BINARY32`, no accumulation, F32 output.
`maximum`, `minimum`, and `rem` resolve two F32 tensors to the same plan shape.
Their numerical and VJP semantics are specified in `design/minimal-op-set.md`.

`mul` has tensor-tensor and tensor-weak-scalar overloads. Its tensor pair uses
the §7.1 promotion matrix; `elementwise_mul` remains the compatibility dispatch
name for the same tensor-pair policy. `pow` has tensor-tensor,
tensor-weak-scalar, and weak-scalar-tensor overloads. Tensor-tensor and reverse
power materialize both operands as F32 and return F32; tensor-weak-scalar keeps
the existing bounded Int32-preserving branch in §7.4.

### 7.10 Predicates and logical negation

`eq`, `ne`, `lt`, and `le` consume `(F32, F32)` and store Bool.
`logical_not` consumes F32 and stores Bool. They use `BINARY32` as the operand
compute semantics, have no accumulation, and their Bool results are
non-differentiable. No Bool arithmetic or implicit Bool promotion is planned.

### 7.11 Remaining reductions and scan

`reduce_prod`, `reduce_max`, and `reduce_min` consume F32 and return F32 with
`SEQUENTIAL_BINARY32_PRODUCT`, `MAXIMUM`, and `MINIMUM` respectively. `argmax`
and `argmin` consume F32, use `ARGMAX`/`ARGMIN`, and return Int32. `cumsum`
consumes and returns F32 with `SEQUENTIAL_BINARY32`. Operation-specific order,
NaN, tie, and VJP rules live in `design/minimal-op-set.md`.

### 7.12 Irregular indexing

`gather` resolves `(F32 data, I32 indices) -> F32`. `scatter` and
`scatter_add` resolve `(F32 base, I32 indices, F32 updates) -> F32`; only
`scatter_add` declares `SEQUENTIAL_BINARY32`, recording its deterministic
ordered collision accumulation. Axis and shape arguments are outside dtype
planning.

### 7.13 Sort and top-k lowering

The internal operations `_sort_values` and `_topk_values` resolve
`F32 -> F32`; `_sort_indices` and `_topk_indices` resolve `F32 -> I32`. All
four have `BINARY32` input compute semantics and no accumulation. Their specs
are non-public; public `sort` and `topk` package corresponding single-output
operations.

### 7.14 Convolution

`conv_general` resolves two F32 tensor operands to F32 with `BINARY32`
products and `SEQUENTIAL_BINARY32` accumulation. Dimension numbers, strides,
dilation, padding, and feature groups are operation semantics outside dtype
planning. Their exact validation, traversal, and VJP rules live in
`design/minimal-op-set.md`.

### 7.15 Masked selection and clamp

`select` is a three-tensor overload with the condition fixed to Bool and both
value operands fixed to Float32:

| condition | on_true | on_false | conv. | compute | accum. | output | diff. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bool | F32 | F32 | — | `BINARY32` | — | F32 | yes for value inputs |

The operation's three tensor operands are aligned **simultaneously** through the
shared structural broadcast rule. Pairwise alignment order is not observable.
At each common logical coordinate it reads only the selected value branch;
unselected NaNs therefore do not contaminate the result. The Bool condition is
non-differentiable. Backward routes the upstream cotangent to the selected value
branch and zero to the other, then the differentiable broadcast views reduce
those cotangents to each original value shape.

`clamp` has four overloads, all with a Float32 data tensor and Float32 output:

| data | lower | upper | conversion | compute | accum. | output |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | F32 tensor | F32 tensor | — | `BINARY32` | — | F32 |
| F32 | F32 tensor | weak real | upper→F32 | `BINARY32` | — | F32 |
| F32 | weak real | F32 tensor | lower→F32 | `BINARY32` | — | F32 |
| F32 | weak real | weak real | bounds→F32 | `BINARY32` | — | F32 |

Weak bounds are normalized by the common weak-scalar rules and do not receive
gradients. Tensor bounds use the same simultaneous structural broadcasting as
`select`. Forward and backward are defined as the ordered stages
`middle = maximum(data, lower)` followed by `result = minimum(middle, upper)`;
the implementation must not add a separate bound-order check. This preserves
the specified NaN, signed-zero, infinity, and `lower > upper` behavior. Its VJP
first applies the `minimum` VJP to `(middle, upper, g)`, then the `maximum` VJP
to `(data, lower, grad_middle)`, including equal-winner splits and NaN
gradients at each stage. A fused kernel is conforming only when it is
observationally identical to these two stages.

---

## 8. Errors

Every unsupported case has exactly one documented disposition. None of them fall
back to a guessed plan.

| Condition | Error | Raised at |
| --- | --- | --- |
| Unregistered operation name | `NotImplementedError` | resolution |
| Operand is not a `DType` | `TypeError` | resolution |
| Operand is `Any`, `Floating`, or `Integer` | `TypeError`, naming the legacy opaque disposition and that legacy `Generic` arithmetic is a separate path (§2.3) | resolution |
| Bool appears in an ordinary numeric operand position | `TypeError`; Bool is accepted only by the `select` condition overload and is never implicitly promoted | resolution |
| Operand is a `CompoundDType` | `NotImplementedError`, naming the deferred per-plane capability (§2.4) | resolution |
| Operand is a registered but unimplemented `SimpleDType` | `NotImplementedError`, naming the supported operation-specific domains (§2.2) | resolution |
| Weak scalar is not a supported Python number | `TypeError` | resolution |
| Weak integer outside `Int32` range in an integer plan | `OverflowError` | resolution |
| Unknown execution option | `TypeError` | option validation |
| `accumulator_dtype` on a non-accumulating operation | `TypeError` | resolution |
| Accumulator descriptor is not `Float32` or `Float64` | `TypeError` or `NotImplementedError`, according to whether it is a simple descriptor | resolution |
| Floating accumulator requested for an exact-integer plan | `TypeError` | resolution |
| Exact integer result outside `Int32` range | `OverflowError` | execution |
| Fixed-width accumulator cannot hold an exact partial sum | `OverflowError` | execution |

Resolution-time errors are value-independent: a plan resolves — or fails — before
any element is read, which is what lets native CPU resolve once while holding the
GIL and then release it (`CPP001`).

---

## 9. Backward policy

The policy fixes the dtype and rounding of backward, not the gradient formulas.

- Every gradient tensor is stored as `Float32`, and every backward operation
  forms its terms with `BINARY32` arithmetic. A backward contraction combines
  those already encoded terms in the forward call's accumulator dtype. Forward
  accumulator options are execution state, not tensor inputs; a backward formula
  that reuses the forward contraction plan must retain them.
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
- `select` carries gradients only through the selected Float32 value branch;
  its Bool condition and any unselected value are non-differentiable for that
  invocation. `clamp` follows the staged `minimum`-then-`maximum` VJP in §7.15,
  including equal-winner splitting and NaN propagation. Weak scalar bounds do
  not receive gradients.

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
   planned operation with all operand-role overloads and dtype domains, so the conformance suite can
   enumerate registered coverage exhaustively and compare it against explicit
   expected-plan fixtures. An operation added without a fixture fails the
   enumeration rather than passing silently.
4. **Exactness is the default.** Same-plan forward and backward conformance
   results are compared bit-exactly. A tolerance is admissible there only for a
   transcendental whose difference is a justified libm divergence, and each such
   tolerance is named and justified individually at its assertion. Staged kernel
   verification is a different comparison: its numerical reduce and matmul cases
   compare a Float32-accumulator target with a Float64-accumulator oracle under the
   explicit analytic or versioned conservative envelope in
   `design/testing-taxonomy.md`.
5. **A backend may refuse, but may not differ.** A backend that cannot execute a
   plan raises; it never substitutes a nearby plan it can execute. Which plan
   shapes a backend does execute is recorded, per exact carrier class, in
   `strideweave.carriers.operation_capability` — an implementation-reach
   registry, not a second policy: it accepts or refuses an already-resolved
   plan and never selects one. The entries execution is accepted against are the
   entries capability introspection reports, so a backend cannot advertise a
   shape it would refuse or run one it never declared.
6. **Selection capabilities are exact and shared.** Generic and CPU advertise
   the Bool/F32/F32 `select` plan and every clamp overload (F32 data with tensor
   and/or weak-real bounds) with F32 output. Both backends must align all tensor
   operands simultaneously, read only the selected `select` branch, and make
   `clamp` observationally equivalent to its ordered `maximum`/`minimum`
   stages; no backend may silently widen Bool or replace a weak bound with an
   integer plan.

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

**12.2 Accumulator dtype is an execution option.** `Float64` is registered as a
simple descriptor so plans and capabilities can name it, but no carrier accepts it
as storage. The option changes neither input encoding nor output dtype, and only
the order-agnostic `FLOATING` rule declares one; mixed-precision element compute
(bf16 storage, fp32 arithmetic) remains a separate future revision.

**12.3 `div` has no integer path.** `I32 / I32` yields `Float32`. Neither
truncating nor flooring integer division is offered, because picking one silently
is worse than not offering one. A future explicit `floor_div` operation is the
expected resolution, not a change to `div`.

**12.4 Mixed-width element compute is unrepresented.** Every v0.3 entry computes
at the
output width. The plan shape anticipates the split (§6) but the matrix does not
exercise it.

**12.5 Floating order is backend-defined.** Capability exactness covers the
floating accumulation kind and dtype. Numerical conformance uses the tolerance
appropriate to the declared kernel class rather than requiring two backends to
choose the same association order.

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
