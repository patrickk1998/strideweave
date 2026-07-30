# Autograd Graph Lifetime and Batched Cotangents — v0.1

**Status:** proposed normative specification for autograd graph lifetime and the
functional gradient entry point.
**Stability:** *intentional starting point, not a compatibility promise.* Section 11
defines how a revision is made.

This document specifies two changes to the autograd system: explicit graph lifetime with
PyTorch-compatible `retain_graph` semantics, and a functional gradient API accepting a
batch of cotangents. Implementation is tracked by `main-hvt.5` and `main-hvt.6`.

Cotangent batching here is **independent of forward batching**. It is a property of the
backward traversal, not of the graph that was built.

---

## 1. What this specifies

1. When autograd graph state is released, and what a caller sees afterwards.
2. The `retain_graph` parameter and its default.
3. A functional gradient entry point that returns gradients instead of accumulating them.
4. How a batch of K cotangents is represented, validated, and returned.

It does *not* specify forward-mode differentiation, double backward, or forward
batching. Sections 8 and 9 record why the first two are out of reach.

---

## 2. Graph lifetime

### 2.1 Current state

The graph is held by `result.autograd_ctx` pointing at the `Operation`, whose saved
state is `inputs_`, `input_versions_`, and the `ctx_` dict
(`src/strideweave/core/native/_operation.hpp:125`). **Nothing ever releases it.** There
is no `retain_graph` parameter because nothing frees; repeated `backward()` works and
accumulates.

A private `clear_inputs()` already exists (`_operation.hpp:131`), so the mechanism is
largely present.

### 2.2 The rule

`Tensor.backward(gradient=None, retain_graph=False)`.

After traversal completes, unless `retain_graph` is true, every operation reached during
phase 1 discovery is freed. Phase 1 already collects exactly this set into
`visited_operations` (`src/strideweave/core/native/_tensor.cpp:403`), including nodes
that received no gradient — phase 2 visits those with a `none` gradient so consumer
counts keep decrementing, and they are freed alongside the rest.

**Freeing** clears `inputs_`, `input_versions_`, and `ctx_`, and sets a freed flag.

### 2.3 Node structure survives freeing

`autograd_ctx` stays attached and the node remains in place; only saved state is
released.

The alternative — clearing `autograd_ctx` so results become leaves — is rejected
because it would silently convert a second `backward()` into `.grad` accumulation on a
former non-leaf. A wrong number, not an error.

A retained node carrying a freed flag gives a real diagnostic instead. The check belongs
where phase 2 is about to invoke `current.backward(...)`. Match PyTorch's wording:
backward through the graph a second time, specify `retain_graph=True`.

### 2.4 Shared subgraphs

A subgraph reachable from two roots is freed by whichever root backwards first, and the
second root then raises. This is PyTorch's behaviour. Reference counting is deliberately
**not** specified: it would make lifetime depend on graph topology in a way callers
cannot predict from their own code.

### 2.5 Interaction with version validation

`RT007` requires backward to reject inputs mutated in place after the forward pass.
Freeing releases the saved inputs, so a freed graph can no longer perform that check —
which is correct, because it can no longer perform backward at all. The freed diagnostic
must fire before any version check, so a caller who both mutated and freed sees the
lifetime error rather than a confusing version error.

---

## 3. Breaking change and migration

Free-by-default is a **breaking change**, chosen deliberately in favour of PyTorch
compatibility.

Repeated backward is currently pinned by five deliberately named tests, which must be
rewritten to pass `retain_graph=True` while asserting the same accumulated values:

| Test | Location |
| --- | --- |
| `test_tensor_permute_backward_accumulates_repeated_calls` | `tests/test_tensor.py:985` |
| `test_tensor_backward_without_gradient_on_scalar_accumulates_repeated_calls` | `tests/test_tensor.py:1054` |
| leaf accumulate-on-repeat case | `tests/test_tensor.py:1194` |
| retained non-leaf accumulate-on-repeat case | `tests/test_tensor.py:1212` |
| `test_repeated_backward_reuses_state_and_accumulates_gradients` | `tests/test_evictable.py:748` |

The README autograd section currently documents unlimited repeated backward and must be
rewritten. `RT007`'s evidence list references mutation-after-forward tests that remain
valid but now need `retain_graph=True` to reach the version check.

---

## 4. Why batched cotangents need a functional API

They cannot hang off `.backward()`.

A leaf's `.grad` carries the leaf's own layout, and `SGD.step()` reads it. K stacked
gradients do not fit that layout, and widening `.grad` to hold a batch would break the
optimizer and every caller that reads `.grad` expecting parameter-shaped values.

PyTorch reached the same conclusion: `is_grads_batched` exists only on
`torch.autograd.grad`, never on `Tensor.backward`. This specification follows it.

So the feature *is* the functional API. The framework currently has none — gradients only
ever land in `.grad`, and they accumulate.

---

## 5. The functional entry point

```
sw.grad(output, inputs, cotangents, *, batched=False, retain_graph=False)
    -> tuple[Tensor, ...]
```

Semantics:

- Returns one gradient per entry of `inputs`, positionally.
- **Never touches `.grad`.** A leaf that had no `.grad` still has none afterwards.
- `retain_graph` behaves exactly as in section 2.
- Traversal, accumulation of shared subgraphs, and version validation are otherwise
  unchanged: this is the same algorithm with a different sink.

An input not reachable from `output` yields `None` in its position rather than a zero
tensor, matching how the existing traversal treats a branch that received no gradient.

### 5.1 No `create_graph`

Every `backward` is hand-written against carrier primitives and returns detached results
(`src/strideweave/carriers/generic/ops.py:92`), so the backward pass builds no graph.
Double backward is unreachable.

The parameter is **omitted**, not accepted and ignored. Accepting it would advertise a
capability that does not exist — the same reasoning that makes `RT013` refuse to
substitute a nearby plan.

---

## 6. Cotangent representation

A batch of K cotangents is **one tensor with a prepended batch mode**, not a sequence of
tensors. Using a mode is the point of having hierarchical layouts: the batch is part of
the layout, so slicing cotangent *k* is offset arithmetic with no copy.

Validation, when `batched=True`:

1. `len(cotangents.layout) == len(output.layout) + 1`;
2. the trailing modes equal the output layout exactly.

Mode 0 is the batch; its extent is K and its stride is unconstrained. Slice *k* is
`Tensor(carrier, offset + k * batch_stride, output_layout)`.

When `batched=False`, `cotangents` is a single cotangent and the existing
`validate_gradient` rule applies unchanged — exact layout equality with `output`.

Requiring exact equality of the trailing modes, rather than mere congruence, is
deliberate: a cotangent is not an operand and is not subject to the alignment rule in
`design/broadcast-and-alignment-policy.md`. Broadcasting a cotangent would silently
change which VJP is computed.

---

## 7. Return representation

For each input, one tensor with a prepended batch mode:

- batch extent K;
- batch stride `input.layout.cosize`;
- trailing modes equal to `input.layout`;
- backed by a carrier of `K * input.layout.cosize`.

`cosize` rather than `size` is **required**, not an optimization: a strided or
hierarchical input layout addresses more physical slots than it has logical elements, so
a `size`-strided batch would overlap adjacent slices. This is the same reason `friendly`
and `nn` allocate through `cosize` (`RT002`). The conformance case is an input with
holes — `Shape([2,3])` with `Stride([1,4])`, `size` 6, `cosize` 10.

When the K cotangents are basis vectors of the output space, the returned stack is the
Jacobian.

---

## 8. Implementation strategy

Run **phase 1 discovery once, phase 2 K times**.

Phase 1 is topology-only: it counts consumers and collects reachable operations, and
neither depends on the cotangent. Re-running it per cotangent would be pure waste.

Phase 2 needs a **collect** mode: record gradients for the requested inputs instead of
accumulating into `.grad`. This is the one substantive change to the native traversal —
an optional collector that, when present, suppresses `.grad` accumulation.

The graph must be **retained across the K inner passes and freed exactly once** at the
end, which is why section 2 is a hard prerequisite rather than an adjacent feature.

Per-operation `backward` implementations are untouched. Each inner pass hands them a
single cotangent in the output's own layout, exactly as today. A fused variant that
pushes a batch mode through every `backward` is out of scope: it would require changing
every backward implementation and would interact with forward batching.

### 8.1 Dependency on `main-xwm`

The collect path allocates gradient buffers on the same helpers that currently
under-count through non-injective layouts (`_gradient_tensor` /
`_detached_tensor_like`), and the stacking scheme in section 7 relies on `cosize` being
respected. Batched cotangents over a broadcast input would be wrong in exactly the way
`main-xwm` describes. That bug is a prerequisite for trusting any batched result.

---

## 9. What this does not provide

- **Forward mode (JVP).** No `J·v`, so no efficient path for a tall Jacobian. A wide
  Jacobian via batched VJP is what this specification buys.
- **Double backward / Hessians.** Section 5.1.
- **Forward batching.** Separate concern; see
  `design/broadcast-and-alignment-policy.md`.
- **Per-sample gradients as a first-class feature.** Expressible as batched cotangents,
  but no dedicated API.

---

## 10. Conformance obligations

1. `backward()` frees by default; a second call raises naming `retain_graph`.
2. `backward(retain_graph=True)` reproduces today's accumulate-on-repeat values exactly.
3. Freeing releases saved input references observably — the saved tensors become
   collectable.
4. A freed shared subgraph reports the same error from either root.
5. The `Evictable` lowering path frees correctly; its adapter owns the visible node
   (`RT011`).
6. Unbatched `sw.grad` agrees with `backward()` plus reading `.grad`, on a multi-input
   graph containing a shared subgraph.
7. Batched cotangents over K basis vectors reconstruct a Jacobian matching
   `torch.autograd.grad(..., is_grads_batched=True)` or
   `torch.autograd.functional.jacobian`.
8. Batch stride uses `cosize`, verified on an input layout with holes.
9. A cotangent whose trailing modes do not equal the output layout is refused with one
   explicit diagnostic.
10. `sw.grad` meets the `RT005` docstring contract and its stub is aligned per `RT006`.

---

## 11. Provisional choices and open questions

1. **Multiple outputs.** `sw.grad` takes a single `output` because an `Operation`
   produces exactly one result (`_operation.hpp:64`). If multi-output operations arrive,
   the signature needs an output sequence with matching cotangents.
2. **Batch mode position.** Prepended is chosen for readability. Appending would keep
   per-sample slices contiguous, which may matter for a future fused implementation.
   Revisit if the fused path lands.
3. **Should `backward` gain `batched=`?** Currently no, per section 4. If `.grad` ever
   grows a batched form, revisit.
4. **`None` versus zeros** for unreachable inputs. `None` matches the existing traversal;
   PyTorch offers `allow_unused` and `materialize_grads`. Not specified here.
5. **Freeing granularity.** All-or-nothing per traversal. A partial free of only the
   subgraph that received gradient is conceivable but would make lifetime depend on
   cotangent sparsity.

---

## 12. Revising this policy

A revision changes this document, the traversal implementation, its expected-behaviour
fixtures, and every affected conformance expectation **in one change**.
