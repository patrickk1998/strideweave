# Broadcast and Alignment Policy — v0.1

**Status:** proposed normative specification for forward operand alignment.
**Stability:** *intentional starting point, not a compatibility promise.* Section 12
defines how a revision is made.

This document specifies *alignment policy only*: how operands of differing layouts are
made compatible before an operation runs, and how a symbol shared by two einsum
operands is classified. It does not decide dtype promotion — that is
`design/SimpleDType-operation-policy.md` — and it does not implement the resolver.
Implementation is tracked by `main-hvt.1` through `main-hvt.4`.

The motivating requirement is that `"b i k, b j k -> b i j"` must work. Today it raises
`Unknown dimension symbol 'b'`.

---

## 1. What this policy decides

For one operation invocation with known operand layouts, this policy decides:

1. whether two operand layouts are compatible;
2. which extents expand, and to what;
3. what layout the result carries;
4. how an einsum symbol is classified as batch, contraction, or free;
5. and, where operands are incompatible, which error is raised.

It does *not* decide dtype, carrier compatibility, gradient formulas beyond the
broadcast node's own, or kernel scheduling.

---

## 2. Scope boundaries

### 2.1 In scope

Every public operation, with its obligation stated in section 6.5. The alignment step in
front of dispatch, and the einsum symbol classification.

Binary elementwise operations are where alignment does visible work; most other
operations either need nothing or need an explicit refusal. Silence per operation is not
acceptable — an operation that already accepts a broadcast operand without anyone having
decided that it should is the failure mode section 6.5 exists to close.

### 2.2 Out of scope: batching as a separate mechanism

There is no batched-operation primitive in this policy, because it is not needed for
semantics. A batch mode is a mode present in both operands and in the output; under
the alignment rule it requires no special handling. A *fused* batched-matmul kernel is
a performance concern, deliberately deferred (section 11).

### 2.3 Out of scope: representation-preserving operations

`view`, `permute`, `rearrange`, and `move` do not align operands. `rearrange` already
*produces* the singleton modes this policy expands (section 4.3) but performs no
alignment of its own.

### 2.4 Out of scope: outputs

Broadcasting applies to operation *inputs*. Section 8 states the output rule, which is
a prohibition rather than a form of alignment.

---

## 3. Definitions

**Injective layout.** A layout is injective when no two distinct logical coordinates
map to the same physical offset. A layout with a stride-0 mode is non-injective; so is
one with overlapping non-zero strides, e.g. `Shape([4,2])` with `Stride([1,1])`.

**Broadcast mode.** A mode with extent > 1 and stride 0. Every logical coordinate along
it reads the same physical slot. Reads through such a layout already work correctly,
and `cosize` already reports the physical size (`Layout(Shape([4,2]), Stride([0,1]))`
has `size` 8 and `cosize` 2).

**Shape profile.** The structural signature of a shape tree — its recipe, already
computed by `Tree.get_recipe` (`src/strideweave/core/layout.py:82`). Two shapes share a
profile when their trees have identical structure, independent of extents.

**Leaf position.** A position in the shape tree identified by its path, not by a
flattened index.

---

## 4. The alignment rule

### 4.1 Structural congruence

Two layouts are **broadcast-compatible** when:

1. their shape trees share a profile; and
2. at every leaf position, the extents are equal, or one of them is 1.

Alignment expands each extent-1 leaf to its peer's extent by setting that leaf's stride
to 0. Where extents are already equal, nothing changes. Strides are otherwise
untouched: alignment never reorders, recopies, or restrides an operand.

Alignment is symmetric and total on compatible inputs: there is exactly one aligned
result for a given pair, so it is not a search.

### 4.2 Why not NumPy rank-align-right

Right-alignment is rejected for three independent reasons:

1. **A top-level mode is a tree.** "Align from the right" has no meaning at depth. Any
   flattening rule would have to pick one, silently.
2. **Mode position here is structural, not incidental.** `matmul` contracts mode 1;
   `reduce` keeps mode 0. Rank-shifting silently reinterprets which mode is which, and
   the reinterpretation is invisible at the call site.
3. **It is implicit shape substitution**, which `RT013` refuses everywhere else: a plan
   is "never replaced by a nearby shape the backend happens to implement." Broadcasting
   by rank shift would be the single place the framework guesses.

Structural congruence is NumPy's "extent 1 expands" rule lifted from flat vectors to
trees, matched positionally within the tree rather than right-aligned.

### 4.3 Ergonomics: insertion is already solved

The caller says *where* an axis goes using the existing einops surface, which already
emits stride-0 singletons at any depth:

```
"a b -> a 1 b"    ->  Shape<(2,1,3)>, Stride<(1,0,2)>
"a b -> (a 1) b"  ->  Shape<((2,1),3)>, Stride<((1,0),2)>
```

This policy supplies only the missing 1 → N widening. Insertion stays explicit and
positional; expansion is automatic once profiles are congruent. The einops description
is therefore the alignment contract, which is the honest place for it.

---

## 5. Where alignment happens

Alignment is a **lowering, not a kernel concern**. One shared align step runs in front
of binary dispatch and produces operands that satisfy the (relaxed) precondition. No
kernel is aware of broadcasting.

This mirrors how `einsum` already sandwiches `matmul` between rearranges, and how the
`Evictable` adapter lowers operands before invoking a nested operation.

Capability declarations are untouched: `operation_capability` entries are plan-shaped
over dtypes (`RT013`), not tensor-shape-shaped, so alignment adds no entries and
resolves the inner plan exactly once.

---

## 6. The elementwise precondition

The precondition changes from **layout equality** to **shape congruence**.

Layout equality is stricter than either backend needs:

- **Generic** kernels index purely logically — `compute(convert(lhs[i]), convert(rhs[i]))`
  over `range(lhs.size())` — so differing strides already compute correctly. The guard
  is `_require_same_layout` (`src/strideweave/carriers/operation_helpers.py:33`).
- **CPU native** builds one shared expanded key and each view resolves it through *its
  own* layout cache (`read_float_expanded` → `cache->index_expanded`,
  `src/strideweave/carriers/cpu/native/_cpu.cpp:172`). Differing strides already work.
  The guard is `require_same_layout` (`_cpu.cpp:166`). Operand agreement on
  `leaf_rank()` is exactly shape-tree congruence.

Both guards are therefore removable in favour of a congruence check. The expanded-key
cache must remain per-layout so `CPP002` continues to hold.

Layout equality survives as a **post-condition** in one place: `validate_gradient`
(`src/strideweave/core/native/_tensor.cpp:652`) still requires a cotangent to match its
tensor exactly. Section 10 explains why that is compatible with broadcasting.

### 6.1 One convention for the whole pointwise class

Binary pointwise operations get **one rule, stated once, with no per-operation content**:

> Operands must be shape-congruent (section 4.1). Extent-1 leaves expand to stride 0.
> The result is allocated in an injective canonical layout over the common logical shape.

`add`, `sub`, `elementwise_mul`, and `div` are this class today, and they already share
the implementation that enforces it (`_binary_elementwise_result` →
`_require_same_layout`, and `require_same_layout` on the native side). A future binary
pointwise operation inherits the rule by construction.

This mirrors `design/SimpleDType-operation-policy.md` section 7.1, which groups
`add`/`sub`/`elementwise_mul` as one class rather than giving each a row. Alignment is
the same kind of decision and is grouped the same way.

### 6.2 Why batching needs no separate rule here

A pointwise operation **consumes no mode**. Every mode of an aligned operand pair is
therefore one of exactly two things:

- matched extents in both operands — what a caller would call a *batch* mode;
- extent 1 in one operand — a *broadcast* mode.

The operation cannot tell these apart and has no reason to: it is defined per element, so
every mode is iterated identically. "Batching a pointwise operation" is not a feature to
add; it is what the alignment rule already does.

The batch/contraction distinction only becomes real for operations that *consume* a mode
— `reduce` consumes mode 1, `matmul` and `einsum` consume the contracted mode. Those are
the only places where "this mode is iterated" must be distinguished from "this mode is
consumed", and section 9 is where that distinction is drawn.

### 6.3 Where the rule is implemented

Alignment belongs in the **shared binary helper**, not in each operation. Placing it in
`_binary_elementwise_result` (Generic) and the shared binary dispatch path (CPU) makes it
structurally impossible for one pointwise operation to diverge from the convention or to
be added without it.

### 6.4 Enumerated conformance

The pointwise class is checked by **enumeration over the operation registry**, not by
sampled per-operation tests: every registered binary pointwise operation must exhibit
identical alignment behaviour. Adding an operation without the rule then fails a test
rather than shipping a silent divergence.

This follows the existing pattern in
`tests/test_operation_policy.py::test_every_registered_operation_has_a_fixture`.

### 6.5 Operations outside the pointwise class

| Operation | Alignment obligation |
| --- | --- |
| `add`, `sub`, `elementwise_mul`, `div` | The section 6.1 convention. |
| Unary elementwise — activations, `exp`, `neg` | None. One operand, indexed logically, so already correct at arbitrary hierarchy and stride, broadcast operands included. |
| Tensor × weak scalar — `mul`, `pow` | None. The scalar has no layout. |
| `reduce` | A broadcast operand is supported. Reducing a stride-zero mode of extent N sums N equal logical reads and therefore multiplies that stored value by N. Backward produces an injective logical cotangent; a differentiable broadcast node, or leaf broadcast-alias accumulation, sums it back to the pre-broadcast storage. |
| `matmul` | A broadcast operand is supported. A stride-zero kept mode produces repeated output rows or columns; a stride-zero contracted mode repeats the same stored factor in the dot product. Backward is computed in injective logical storage and then summed through the same broadcast-gradient rule. |
| `einsum` | Section 9. Alignment is how batch and free symbols are realized. |
| `view`, `permute`, `rearrange` | No alignment. `rearrange` *produces* the singleton modes section 4.3 expands. `view`'s backward scatters, so scattering into a non-injective layout falls under section 8. |
| `move` | Moving a broadcast tensor preserves its exact stride-zero layout and copies its `cosize` physical span; it does not materialize `size` logical elements. Backward accepts an injective same-shape cotangent, moves those logical values into fresh source-class storage, and lets the broadcast-gradient rule perform any required summation. |

These operations do not use pointwise alignment, but broadcast operands now
have explicit forward, storage, and backward semantics rather than inheriting
accidental behavior from logical indexing.

---

## 7. Result layouts

Every operation result is allocated in an **injective canonical layout**, derived from
the common logical shape. `_canonical_layout_from_modes` already does this.

A result never inherits an operand's broadcast strides.

---

## 8. Broadcast layouts are input-only

A non-injective layout is legal for an operation input and **illegal for an operation
output or a gradient buffer**. Writing through a non-injective layout aliases: several
logical writes land on one physical slot, and the last one wins.

This is not hypothetical. It is the mechanism of `main-xwm`, where a gradient buffer
inherited a broadcast layout and under-counted by exactly the broadcast factor with no
error raised. The rule is stated here so it is enforced rather than assumed, and it is
a candidate for a new `RT0xx` entry.

---

## 9. Einsum symbol classification

### 9.1 The uniform rule

Every symbol is classified by where it appears, with no special cases:

| appears in | in output | classification | treatment |
| --- | --- | --- | --- |
| both operands | no | **contraction** | reduced away |
| both operands | yes | **batch** | kept, aligned in lockstep |
| one operand | yes | **free** | broadcast into the other, kept |

Today only the first row exists. A shared symbol is *forced* to be a contraction
dimension and forbidden from the output (`_einsum_output_symbol_ids`,
`src/strideweave/einops/__init__.py:481`), which is why a batch symbol raises
`Unknown dimension symbol`.

The classification is the standard einsum distinction between a batch index and a
summation index. Nothing here is novel; what is new is that this framework can express
all three rows through one lowering.

### 9.2 The general lowering

For `"b i k, b j k -> b i j"`:

1. align `lhs` from `b i k` to the union space `b i j k`, with stride 0 on `j`;
2. align `rhs` from `b j k` to `b i j k`, with stride 0 on `i`;
3. elementwise multiply, giving `b i j k`;
4. rearrange to two modes `((b i j), (k))`;
5. `reduce` over mode 1, giving `(b i j)`;
6. rearrange to the requested output.

Steps 4 through 6 are exactly what the existing reduce path already does — `ReduceSpec`
builds `Tree(output, reduced)` in this shape. Steps 1 and 2 are the new part, and they
are the alignment rule applied unchanged.

Note that `b`, `i`, and `k` receive no individually special handling. A batch symbol is
simply one that is stride-0 in neither operand and is not reduced.

### 9.3 Matmul remains a fast path

The existing `matmul` lowering is retained for the flat, no-batch case and must not
regress. Section 11 covers the performance consequence.

---

## 10. Backward: broadcast-forward is reduce-backward

A mode that reads many logical positions from one physical slot must **sum** on the way
back. The gradient of a broadcast operand is the cotangent reduced over its broadcast
modes.

Alignment is therefore a **differentiable operation**, not a silent view rewrite. Two
reasons:

1. `validate_gradient` requires exact layout equality, so the gradient for a broadcast
   input must carry the *pre-broadcast* layout. A node produces that naturally, the same
   way `einsum` inserts rearranges.
2. The summing reduction then lives in exactly one `backward` implementation, instead of
   every operation being asked to remember to accumulate.

This is the general form of a rule the codebase already implements by hand. `nn.Linear`
broadcasts its bias with a ones-column matmul specifically to obtain "a tile whose
backward pass sums the bias gradient over the batch." Once this node exists, that trick
is deletable.

**Scope of non-injectivity.** Stride-0 is cheap to detect and to sum. General
overlapping strides are not. This policy supports exactly stride-0 non-injectivity and
requires other aliasing layouts in an autograd path to be **explicitly refused** rather
than silently mis-differentiated, which is today's behaviour.

---

## 11. Performance consequences

The general lowering materializes the full union index space: `b*i*j*k` elements for the
example, against `b*i*j` for a batched matmul. That is asymptotically worse, and it is
an accepted consequence of defining semantics first — the same posture that makes
`Generic` the behavioural reference while `CPU` supplies kernels.

Two deferred optimizations, neither of which may change results:

- a fused batched-matmul path recognizing the `(batch, free, contracted)` pattern
  before materializing the union space;
- fusing multiply-then-reduce so the intermediate is never allocated.

Both are separate issues. A caller who needs the fast path today writes the `matmul`
form directly.

---

## 12. Errors

| Condition | Error |
| --- | --- |
| Shape trees do not share a profile | `ValueError` naming both profiles |
| Congruent profiles, leaf extents differ and neither is 1 | `ValueError` naming the leaf position and both extents |
| `Layout.complement` on a non-injective layout | `ValueError`, consistent with its existing "overlaps with itself" / "is incongruent" refusals — **not** today's raw `ZeroDivisionError` |
| Non-stride-0 aliasing layout in an autograd path | `ValueError` refusing explicitly |
| Non-injective layout supplied as an output or gradient buffer | `ValueError` |

No condition falls back to a guessed alignment. Rank shifting is never attempted, so
"incompatible ranks" is a profile mismatch and reports as one.

---

## 13. Conformance obligations

1. `Generic` and `CPU` agree elementwise under broadcasting, extending
   `tests/test_dtype_conformance.py`.
2. Every operation result is injective — asserted generically over the operation
   registry, not sampled per operation.
3. Broadcast gradients are checked against torch, at top level and at hierarchy depth.
4. `einsum` batch symbols are checked against `torch.einsum`, with more than one batch
   symbol and at more than one position.
5. The no-batch case still dispatches through `matmul` — asserted on the path taken, not
   merely on the answer.
6. Public layout and einops APIs meet the `RT005` docstring contract, including
   `Syntax`, `Semantics`, and `Mode assumptions`.

---

## 14. Documentation obligations

- The README boundary "No general broadcasting system; binary operations require
  compatible layouts" is replaced by the rule in section 4.
- The README operations section documents batch symbols.
- `INVARIANTS.md` gains entries for the input-only rule (section 8) and injective
  results (section 7), or an existing entry is extended.
- `nn.Linear`'s documented ones-column bias trick is removed once section 10 lands.

---

## 15. Provisional choices and open questions

1. **Is `expand` public?** Section 4.3 assumes expansion is automatic and insertion is
   explicit via `rearrange`. An explicit `sw.expand` may still be wanted for
   readability. Unresolved.
2. **Extent-1 ambiguity.** When both peers have extent 1 at a position, nothing
   expands. That is consistent but means `1` never carries "unknown" semantics.
3. **Should extent-1 expansion require opt-in?** NumPy's rule is famously error-prone.
   A stricter variant would demand an explicit `expand` and treat silent 1 → N as an
   error. This policy takes the permissive option for ergonomics; revisit if it hides
   bugs in practice.
4. **Overlapping-stride injectivity detection** is specified as "refuse" rather than
   "support". If a real use appears, it needs its own gradient rule.

---

## 16. Revising this policy

A revision changes this document, the alignment implementation, its expected-behaviour
fixtures, and every backend conformance expectation **in one change**. A change that
lands in one backend first is a policy fork.
