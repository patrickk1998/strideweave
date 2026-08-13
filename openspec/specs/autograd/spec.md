---
title: Autograd
publish: true
status: stable
order: 4
summary: Reverse-mode graph construction, backward traversal, gradient accumulation, saved-version validation, and functional vector-Jacobian products.
---

# autograd Specification

## Purpose

Define StrideWeave's reverse-mode automatic differentiation: when operations
record a differentiation graph, how backward traversal propagates and
accumulates cotangents, how saved forward state is validated and released, and
how functional vector-Jacobian products are computed without touching any
tensor's `.grad`.

## Terminology

| Term | Meaning |
| --- | --- |
| differentiable Tensor | A Tensor whose logical dtype participates in differentiation, reported by `tensor.is_differentiable()`. |
| autograd node | The `Operation` instance a forward call records on its result Tensor, readable and assignable as `tensor.autograd_ctx`. |
| autograd graph | The set of autograd nodes reachable from a Tensor by following each node's saved Tensor inputs to their own autograd nodes. |
| leaf Tensor | A differentiable Tensor whose `autograd_ctx` is `None`, including a Tensor a caller constructed directly and a result produced while graph construction was disabled. |
| non-leaf Tensor | A differentiable Tensor whose `autograd_ctx` is an autograd node. |
| cotangent | The derivative value propagated backward from an output during reverse-mode differentiation. |
| saved Tensor inputs | The positional Tensor arguments a forward call records for its autograd node, in argument order, readable as `operation.inputs()`. |
| saved input version | The ordered snapshot a forward call takes of every unique constituent carrier of a saved Tensor input, pairing each carrier's identity with its version, readable as `operation.input_versions()`. |
| reachable input | A Tensor that reverse-mode traversal from an output reaches through saved Tensor inputs and receives a gradient for. |
| canonical injective layout | The compact injective layout for a shape whose strides advance fastest in the first mode at every hierarchy depth. |
| graph release | Discarding an autograd node's saved Tensor inputs, saved input versions, and context so it can no longer run backward, while leaving the node attached to its result Tensor. |
| broadcast aliasing | Layout aliasing that remains only because one or more modes have stride zero, so every remaining collision maps a group of logical coordinates onto one storage slot. |

## Requirements

### Requirement: Gradient participation follows the logical dtype

`tensor.is_differentiable()` SHALL return `True` when the Tensor's logical
dtype is `Float32` or `Floating`, and `False` for every other logical dtype,
including `Any`, `Int32`, and `Bool`. Differentiability SHALL depend on the
logical dtype alone and SHALL be identical across carrier implementations.

Reading `tensor.grad`, calling `tensor.retain_grad(retain)`, calling
`tensor.backward(gradient, retain_graph)`, and assigning a value other than
`None` to `tensor.autograd_ctx` or `tensor.grad` SHALL fail with `RuntimeError`
when the Tensor is not differentiable. Assigning `None` to `tensor.grad` or
`tensor.autograd_ctx` SHALL succeed for every Tensor.

`tensor.grad` SHALL be `None` until a gradient is accumulated or assigned, and
`tensor.autograd_ctx` SHALL be `None` until a forward call records a node.

#### Scenario: Report differentiability from the logical dtype

- **WHEN** a caller reads `is_differentiable()` on a `Float32` Tensor and on a
  `Floating` Tensor
- **THEN** both report `True`
- **AND** an `Any`, `Int32`, or `Bool` Tensor reports `False`

#### Scenario: Reject gradient APIs on a non-differentiable Tensor

- **WHEN** a caller reads `grad`, calls `retain_grad()`, calls `backward()`, or
  assigns a non-`None` `autograd_ctx` on an `Int32` or `Any` Tensor
- **THEN** each call fails with `RuntimeError`
- **AND** no gradient is recorded for that Tensor

### Requirement: Operations record a reverse-mode node for differentiable results

A public operation forward call SHALL save its positional Tensor arguments and
their saved input versions when graph construction is enabled in the calling
thread and at least one positional Tensor argument is differentiable. It SHALL
then record itself as the result Tensor's autograd node when the result is also
differentiable.

When graph construction is disabled, when no positional Tensor argument is
differentiable, or when the result is not differentiable, the forward call
SHALL leave the result's `autograd_ctx` as `None` and SHALL retain no saved
Tensor inputs or saved input versions.

The saved Tensor inputs SHALL consist exactly of the call's positional Tensor
arguments in argument order, so a Tensor supplied several times SHALL be saved
once per positional occurrence, while a positional argument that is not a Tensor
and the `options` execution-option keyword SHALL remain outside them.

#### Scenario: Record a node for a differentiable result

- **WHEN** a caller invokes an operation whose positional arguments include a
  differentiable Tensor, a non-Tensor value, and an `options` keyword, and
  whose result is differentiable
- **THEN** the result's `autograd_ctx` is that operation
- **AND** the saved Tensor inputs contain only the positional Tensor arguments
  in argument order, each with its saved input version

#### Scenario: Record no node for a non-differentiable result

- **WHEN** an operation on `Any` or `Int32` Tensors produces a
  non-differentiable result
- **THEN** the result's `autograd_ctx` is `None`
- **AND** the operation retains no saved Tensor inputs

### Requirement: Graph construction state is thread-local and restorable

`is_grad_enabled()` SHALL return whether the current thread builds autograd
graphs, and SHALL be `True` for a thread that has not changed it.
`set_grad_enabled(enabled)` SHALL set that state for the current thread, where
`enabled` is the requested graph-construction state, and SHALL return `None`.
`no_grad()` SHALL return a context manager that disables graph construction for
the current thread and restores the state observed on entry when the block
exits, including when the block raises.

The state SHALL be thread-local, so a change in one thread SHALL leave every
other thread's state unchanged. Nested `no_grad()` blocks SHALL each restore
the state observed at their own entry, and graph construction SHALL resume for
operations invoked after the outermost block exits.

#### Scenario: Suppress graph construction inside a block

- **WHEN** a caller invokes an operation on differentiable Tensors inside a
  `no_grad()` block
- **THEN** the result's `autograd_ctx` is `None` and the operation retains no
  saved Tensor inputs
- **AND** `is_grad_enabled()` reports `True` again after the block exits

#### Scenario: Keep the state local to one thread

- **WHEN** a thread inside a `no_grad()` block starts another thread that reads
  `is_grad_enabled()`
- **THEN** the started thread reports `True`
- **AND** the entering thread still reports `False` inside its block

### Requirement: Backward traverses each recorded node once in topological order

`tensor.backward(gradient, retain_graph)` SHALL propagate `gradient` through
the autograd graph reachable from `tensor` and SHALL return `None` on success.
`gradient` SHALL mean the cotangent for `tensor` and SHALL default to `None`.
`retain_graph` SHALL mean whether the reached graph stays available for a later
traversal and SHALL default to `False`.

Traversal SHALL be iterative and topological: a node's backward SHALL run only
after every reached consumer of its result has contributed its cotangent, and
each reached node SHALL run backward at most once per traversal. Contributions
to the same Tensor SHALL be summed before that Tensor's producing node runs, so
a Tensor consumed several times in one graph SHALL receive the sum of its
contributions. A node SHALL run backward exactly when its result has an
accumulated cotangent, and SHALL run it with that accumulated cotangent.

#### Scenario: Sum contributions of a repeated input

- **WHEN** a caller adds a differentiable Tensor to itself and calls `backward`
  on the result with a cotangent
- **THEN** the input's `.grad` holds the sum of both contributions

#### Scenario: Accumulate a shared subgraph before propagating

- **WHEN** an output consumes one intermediate Tensor through two operations
  and `backward` runs on that output
- **THEN** the intermediate's producing node runs backward once with the summed
  cotangent
- **AND** the traversal reaches each leaf gradient exactly once

### Requirement: A scalar result supplies its own unit cotangent

When `gradient` is `None`, `tensor.backward` SHALL require `tensor` to have
exactly one top-level leaf mode whose extent is one, and SHALL use a unit
cotangent that has the tensor's layout and fresh storage of the tensor
carrier's exact implementation class.

For any other layout, including a layout of logical size one with more than one
mode such as `Layout(Shape([1, 1]), Stride([1, 1]))`, `backward` SHALL fail
with `ValueError` before modifying any `.grad`. A supplied `gradient` SHALL be
used in place of the implicit unit cotangent.

#### Scenario: Differentiate a scalar without an explicit cotangent

- **WHEN** a caller invokes `backward()` on a differentiable Tensor with layout
  `Layout(Shape(1), Stride(1))`
- **THEN** propagation uses a unit cotangent carried by the same carrier class
- **AND** each reached leaf receives the derivative of that scalar

#### Scenario: Reject an implicit cotangent for a non-scalar Tensor

- **WHEN** a caller invokes `backward()` with no `gradient` on a Tensor with
  layout `Layout(Shape([1, 1]), Stride([1, 1]))` or `Layout(Shape(2), Stride(1))`
- **THEN** the call fails with `ValueError`
- **AND** no `.grad` is modified

### Requirement: Cotangent layouts must address the target tensor exactly

A cotangent supplied to `backward`, produced by a node's backward, or supplied
to functional differentiation SHALL be a Tensor; any other value SHALL fail
with `TypeError`.

When the target Tensor's placement layout is injective, the cotangent layout
SHALL equal that layout exactly, and any other layout SHALL fail with
`ValueError`.

When the target Tensor's placement layout is not injective, its aliasing SHALL
be broadcast aliasing; any other aliasing SHALL fail with `ValueError`. For
such a target the cotangent SHALL have the target's shape in an injective
layout, and any other shape or a non-injective cotangent layout SHALL fail with
`ValueError`.

Each of these failures SHALL occur before the target Tensor's `.grad` is
modified.

#### Scenario: Reject a mismatched cotangent layout

- **WHEN** a caller supplies a cotangent whose layout differs from an injective
  target Tensor's layout
- **THEN** the call fails with `ValueError`
- **AND** the target Tensor's `.grad` is unchanged

#### Scenario: Refuse aliasing that broadcast modes do not explain

- **WHEN** reverse-mode differentiation reaches a Tensor with layout
  `Layout(Shape([4, 2]), Stride([1, 1]))`
- **THEN** the call fails with `ValueError`
- **AND** no under-counted gradient is produced

### Requirement: Leaf tensors accumulate detached gradients

A leaf Tensor reached by traversal SHALL accumulate its total cotangent into
`.grad`. The accumulated value SHALL be a detached copy in fresh storage of the
leaf carrier's exact implementation class, so a later change to the supplied
cotangent SHALL leave `.grad` unchanged, and `.grad` SHALL carry no autograd
node.

When `.grad` is already set, accumulation SHALL add the new contribution to the
existing value elementwise. Accumulation SHALL continue across repeated
traversals of a retained graph and across traversals of distinct forward
graphs, until a caller assigns `None` to `.grad`.

A `backward` call on a Tensor that has no autograd node SHALL accumulate the
supplied cotangent into that Tensor's `.grad` and SHALL propagate no further.

#### Scenario: Detach the accumulated gradient

- **WHEN** a caller passes a cotangent Tensor to `backward` on a leaf and then
  mutates that cotangent
- **THEN** the leaf's `.grad` holds the values captured at the call
- **AND** its carrier has the same exact implementation class as the leaf's carrier

#### Scenario: Accumulate across two traversals

- **WHEN** a caller runs `backward(gradient, retain_graph=True)` and then
  `backward(gradient)` on the same result
- **THEN** each reached leaf's `.grad` holds the sum of both traversals

### Requirement: Non-leaf tensors retain gradients only on request

A non-leaf Tensor SHALL leave `.grad` as `None` after traversal unless
`retain_grad(retain)` requested retention, where `retain` means whether the
Tensor keeps its own gradient and SHALL default to `True`. `retain_grad(True)`
SHALL cause every later traversal that reaches the Tensor to accumulate into
`.grad` with the leaf accumulation semantics above, and `retain_grad(False)`
SHALL disable that retention. `retain_grad` SHALL return `None`.

Retention SHALL be per Tensor, so retaining one intermediate in a chain SHALL
leave every other intermediate's `.grad` as `None`.

#### Scenario: Retain one intermediate gradient

- **WHEN** a caller calls `retain_grad()` on one intermediate of a chain and
  runs `backward` on the final result
- **THEN** that intermediate's `.grad` holds its cotangent
- **AND** every other intermediate's `.grad` is `None`

#### Scenario: Leave intermediates without gradients by default

- **WHEN** a caller runs `backward` on the result of a two-operation chain
  without calling `retain_grad`
- **THEN** the result's and the intermediate's `.grad` are `None`
- **AND** the leaf's `.grad` holds the chained derivative

### Requirement: Gradient buffers use injective layouts and aggregate broadcast aliases

Every gradient buffer autograd allocates SHALL have an injective layout. When
the target Tensor's placement layout is injective, the gradient SHALL preserve
that layout. When the target aliases only through broadcast aliasing, the
gradient SHALL use the canonical injective layout for the target's shape.

For such a broadcast target, the accumulated gradient SHALL sum every logical
contribution that addresses the same target storage slot, and SHALL place that
sum at each logical position associated with that slot. This SHALL hold at
every hierarchy depth and SHALL agree across carrier implementations.

#### Scenario: Sum contributions of a top-level broadcast mode

- **WHEN** reverse-mode differentiation reaches a leaf with layout
  `Layout(Shape([4, 2]), Stride([0, 1]))`
- **THEN** its `.grad` uses the canonical injective layout
  `Layout(Shape([4, 2]), Stride([1, 4]))`
- **AND** each logical position holds the sum of the contributions addressing
  its storage slot

#### Scenario: Sum contributions at hierarchy depth

- **WHEN** reverse-mode differentiation reaches a leaf whose nested mode is
  broadcast, such as `Shape([2, [4, 3]])` with a stride-zero nested mode
- **THEN** its `.grad` uses the canonical injective layout for that shape
- **AND** each replicated logical position holds the same summed value

### Requirement: Backward releases graph state unless the graph is retained

With `retain_graph=False`, a successful traversal SHALL release every node
reached during topology discovery, including nodes that received no cotangent.
Released nodes SHALL remain attached to their result Tensors so a former result
is never treated as a leaf.

A later traversal that reaches a released node SHALL fail with `RuntimeError`
naming `retain_graph=True`, and that failure SHALL occur before the node
validates saved input versions or runs backward. A subgraph shared by several
roots SHALL be released by whichever reachable root traverses it first, so a
traversal from another root afterwards SHALL fail the same way.

With `retain_graph=True`, the traversal SHALL preserve every reached node's
saved Tensor inputs, saved input versions, and context for another traversal.
A leaf has no node to release and SHALL keep accumulating normally.

#### Scenario: Release the graph after the default traversal

- **WHEN** a caller runs `backward(gradient)` on a result
- **THEN** each reached node reports no saved Tensor inputs, no saved input
  versions, and an empty context
- **AND** a second `backward(gradient)` on that result fails with `RuntimeError`
  naming `retain_graph=True`

#### Scenario: Release a shared subgraph from the first root

- **WHEN** two results share one intermediate node and a caller runs `backward`
  on the first result
- **THEN** `backward` on the second result fails with `RuntimeError` naming
  `retain_graph=True`

### Requirement: Backward rejects storage modified after the forward pass

Before running a node's backward, traversal SHALL compare each saved Tensor
input's current version against its saved input version. When any comparison
differs, the traversal SHALL fail with `RuntimeError` reporting in-place
modification.

Every successful public mutation of a carrier SHALL change the version observed
by every Tensor that shares that carrier, so mutating a Tensor, an alias, or a
view of saved storage after the forward pass SHALL invalidate a graph built
before that mutation. A caller that must reuse saved storage SHALL therefore
order a forward pass, its traversal, and then any mutation.

#### Scenario: Reject a traversal after an in-place mutation

- **WHEN** a caller builds a graph from a differentiable Tensor, then assigns a
  value into that Tensor through `tensor[i, j] = value`, and then calls
  `backward` on the result
- **THEN** the call fails with `RuntimeError` reporting in-place modification
- **AND** no `.grad` is modified by that traversal

#### Scenario: Reject a traversal after a mutation through the source

- **WHEN** a caller builds a graph from a view, then mutates the view's source
  Tensor through a coordinate assignment, and then calls `backward` on the
  result
- **THEN** the call fails with `RuntimeError` reporting in-place modification

#### Scenario: Complete a traversal before mutating parameters

- **WHEN** a caller runs a forward pass, calls `backward` on its result, and
  then mutates the parameters in place
- **THEN** the traversal succeeds and accumulates parameter gradients
- **AND** a graph built before that mutation is no longer traversable

### Requirement: Operation backward returns one gradient per saved tensor input

An operation reached by traversal SHALL receive the accumulated cotangent for
its result through its public `backward(gradient)` method and SHALL return one
value per saved Tensor input, in the saved order. A returned value SHALL be the
cotangent Tensor for that input, or `None` when the operation propagates no
cotangent to it.

When the returned count differs from the saved Tensor input count, traversal
SHALL fail with `ValueError`. A `None` entry and an entry for a
non-differentiable saved input SHALL contribute nothing to that input.

#### Scenario: Propagate only to the inputs an operation differentiates

- **WHEN** an operation with two saved Tensor inputs returns a cotangent for
  the first and `None` for the second
- **THEN** only the first input receives a contribution
- **AND** traversal continues through the first input's node, while the second
  input's `.grad` keeps the value it already held and its producing node is
  released with the rest of the traversed graph

#### Scenario: Reject a mismatched gradient count

- **WHEN** an operation returns fewer values than its saved Tensor input count
- **THEN** traversal fails with `ValueError`

### Requirement: Functional gradients collect vector-Jacobian products without touching `.grad`

`grad(output, inputs, cotangents, *, batched=False, retain_graph=False)` SHALL
return one vector-Jacobian product per requested input. `output` SHALL mean the
differentiable Tensor whose graph is traversed. `inputs` SHALL mean the
sequence of differentiable Tensors gradients are requested for. `cotangents`
SHALL mean the cotangent for `output`, or the batch of cotangents when
`batched` is true. `batched` SHALL mean whether the first cotangent mode
indexes independent products and SHALL default to `False`. `retain_graph` SHALL
mean whether the traversed graph stays available afterwards and SHALL default
to `False`.

The call SHALL return a tuple with one entry per requested input in the order
given: a gradient Tensor for a reachable input, and `None` for an input the
traversal does not reach. It SHALL neither read nor modify `.grad` on any
Tensor, and every returned gradient SHALL be a detached copy in fresh storage
of that input's exact carrier implementation class. For gradient buffers,
cotangent validation, graph release, and saved-version validation, it SHALL
follow the same rules as `backward`.

`grad` SHALL fail with `TypeError` when `inputs` is a Tensor rather than a
sequence of Tensors, when `inputs` is a value that cannot be iterated such as
`None` or an integer, when `output` or `cotangents` is not a Tensor, or when
`inputs` contains a value that is not a Tensor. It SHALL fail with `ValueError`
when `inputs` contains a non-differentiable Tensor, and with `RuntimeError` when
`output` is not differentiable. Each of these failures SHALL occur before any
traversal, so no gradient is returned and no `.grad` changes.

#### Scenario: Match backward on a shared multi-input graph

- **WHEN** a caller requests gradients for both inputs of a graph that consumes
  a shared intermediate, with `retain_graph=True`
- **THEN** the returned gradients equal the `.grad` values a later `backward`
  with the same cotangent accumulates
- **AND** no Tensor's `.grad` changed during the `grad` call

#### Scenario: Return None for an unreachable input

- **WHEN** a caller requests gradients for one Tensor in the graph and one
  Tensor that the output does not depend on
- **THEN** the tuple holds a gradient for the first input and `None` for the second

#### Scenario: Reject an inputs argument that is not a sequence of Tensors

- **WHEN** a caller passes a single Tensor, `None`, or an integer as `inputs`
- **THEN** the call fails with `TypeError`
- **AND** no gradient is returned and no Tensor's `.grad` changes

### Requirement: Batched functional gradients return one stacked gradient per reachable input

With `batched=True`, `cotangents` SHALL carry `K` cotangents for `output`: its
layout SHALL have one prepended leaf batch mode of extent `K` followed by modes
exactly equal to `output.layout`, and any other layout SHALL fail with
`ValueError`. The cotangent at batch index `k` SHALL be the cotangent whose
layout is `output.layout` and whose logical values are the values `cotangents`
holds at batch index `k`, and each SHALL satisfy the cotangent rules for
`output` stated above.

The call SHALL return one stacked gradient Tensor for every reachable input and
`None` for every input the traversal does not reach. A stacked gradient's layout
SHALL prepend a batch mode of extent `K` to the layout the same request returns
for a single cotangent, and its values at batch index `k` SHALL equal the
gradient the same request returns for cotangent `k` alone. The batch mode's
stride SHALL be the single-gradient layout's `cosize`, so the storage each batch
index addresses stays disjoint from every other batch index even when the
single-gradient layout addresses storage holes.

A batched call SHALL leave the traversed graph in the state one unbatched call
leaves it in: with `retain_graph=False` the graph SHALL be released once, after
the last of the `K` cotangents, so a later traversal fails with `RuntimeError`
naming `retain_graph=True`; with `retain_graph=True` the graph SHALL stay
available for a later traversal.

#### Scenario: Reconstruct a Jacobian from basis cotangents

- **WHEN** a caller passes basis cotangents in a batched cotangent Tensor for a
  two-element output
- **THEN** the returned stacked gradient holds one Jacobian row per batch index
- **AND** the requested input's `.grad` is still `None`

#### Scenario: Keep batch indices disjoint across storage holes

- **WHEN** the requested input's layout has a `cosize` larger than its `size`
- **THEN** the stacked gradient's batch stride equals the single-gradient
  layout's `cosize`
- **AND** each batch index holds the gradient for its own cotangent

#### Scenario: Reject a batched cotangent with a different trailing layout

- **WHEN** the batched cotangent's trailing modes differ from `output.layout`
- **THEN** the call fails with `ValueError` naming the required prepended leaf
  batch mode followed by the output layout

#### Scenario: Release the graph after the final cotangent

- **WHEN** a caller runs a batched `grad` with `retain_graph=True` and then a
  batched `grad` without it
- **THEN** the second call releases the graph after its last cotangent
- **AND** a third batched `grad` fails with `RuntimeError` naming
  `retain_graph=True`

### Requirement: Node-rooted traversal propagates a cotangent without a target Tensor

`Tensor.backwards_traversal(gradient, operation, retain_graph)` SHALL propagate
`gradient` through the autograd graph rooted at `operation` and SHALL return
`None` on success. `gradient` SHALL mean the cotangent for the result
`operation` produced. `operation` SHALL mean the autograd node the traversal
starts from, or `None` for no graph. `retain_graph` SHALL mean whether the
traversed graph stays available for a later traversal and SHALL default to
`False`.

When `operation` is `None`, the call SHALL return `None` and SHALL modify no
`.grad`. Otherwise the call SHALL run `operation`'s backward with `gradient`
and SHALL then follow the traversal, accumulation, gradient-buffer,
saved-version, and graph-release rules stated above for `tensor.backward`.

Because the call names no target Tensor, accumulation SHALL begin at
`operation`'s saved Tensor inputs: the call SHALL derive no implicit unit
cotangent and SHALL accumulate `gradient` into no Tensor's `.grad`, including
the `.grad` of a result whose `retain_grad` requested retention. `gradient`
SHALL be a cotangent that `operation`'s backward accepts; for any other value
the call SHALL report the failure that operation's backward raises, which is
`TypeError` for a value that is not a Tensor and `ValueError` for a cotangent
whose shape differs from the result's shape. When `operation` is a value that
is neither an autograd node nor `None`, the call SHALL fail with
`AttributeError` at the first autograd-node attribute that value lacks.

#### Scenario: Propagate from an autograd node

- **WHEN** a caller passes a matching cotangent and a result Tensor's autograd
  node to `Tensor.backwards_traversal`
- **THEN** the call returns `None` and each reached leaf accumulates the
  gradient that `backward` on that result with the same cotangent accumulates
- **AND** the result Tensor's own `.grad` is unchanged

#### Scenario: Traverse no graph

- **WHEN** a caller passes `None` as `operation`
- **THEN** the call returns `None`
- **AND** no Tensor's `.grad` changes

#### Scenario: Reject a second node-rooted traversal

- **WHEN** a caller runs `Tensor.backwards_traversal` twice on one node with
  `retain_graph` left at `False`
- **THEN** the second call fails with `RuntimeError` naming `retain_graph=True`

### Requirement: Reverse mode is the complete differentiation surface

The public differentiation surface SHALL consist of `tensor.backward(gradient,
retain_graph)`, `Tensor.backwards_traversal(gradient, operation,
retain_graph)`, `tensor.grad`, `tensor.retain_grad(retain)`,
`tensor.autograd_ctx`, `tensor.is_differentiable()`, `grad(output, inputs,
cotangents, *, batched, retain_graph)`, `is_grad_enabled()`,
`set_grad_enabled(enabled)`, and `no_grad()`.

`backward`, `backwards_traversal`, and `grad` SHALL accept exactly the
parameters named in this specification, so a `create_graph` argument SHALL fail
with `TypeError`. Every
gradient these entry points produce SHALL be detached, carrying no autograd
node, so differentiating a gradient SHALL be unavailable and second-order
results such as a Hessian SHALL require a caller-supplied method outside this
capability. Differentiation SHALL be reverse mode only; the surface provides no
forward-mode Jacobian-vector product.

#### Scenario: Reject a create_graph argument

- **WHEN** a caller passes `create_graph=True` to `backward`,
  `Tensor.backwards_traversal`, or `grad`
- **THEN** the call fails with `TypeError`

#### Scenario: Produce detached gradients

- **WHEN** a traversal accumulates `.grad` on a leaf or `grad` returns a
  gradient Tensor
- **THEN** that gradient Tensor's `autograd_ctx` is `None`
- **AND** calling `backward` on it propagates to no other Tensor

### Requirement: Autograd entry points require a one-subtensor Tensor

`tensor.backward(gradient, retain_graph)` SHALL require the receiving Tensor,
and `grad(output, inputs, cotangents)` SHALL require `output`, to have an
authoritative representation holding exactly one subtensor. For a validated
multi-subtensor Tensor, each SHALL fail with `NotImplementedError` before
allocating a gradient buffer or modifying any `.grad`.

#### Scenario: Reject a multi-subtensor output

- **WHEN** a caller calls `backward` on a validated multi-subtensor Tensor
- **THEN** the call fails with `NotImplementedError`
- **AND** no gradient buffer is allocated and no `.grad` is modified
