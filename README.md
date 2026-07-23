# StrideWeave

[![CI](https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml)

StrideWeave is a research tensor and autograd framework built around hierarchical,
CuTe-style layouts. A tensor combines a carrier, a physical offset, and a
layout. A carrier owns or references storage and dispatches the operations it
supports; StrideWeave deliberately has no separate device abstraction.

The project is currently a tested prototype rather than a complete PyTorch
replacement. It provides native CPU kernels, a Python reference carrier,
autograd, hierarchical layout transformations, a small module system, and
ergonomic layers (`strideweave.nn` with a minimal layer library and optimizer,
`strideweave.friendly` with tensor factories), but does not yet include
accelerator carriers.

## Core Model

- `Tensor(carrier, offset, layout)` references storage owned by a `Carrier`.
- `Layout` describes hierarchical `Shape` and `Stride` trees and maps logical
  coordinates to physical storage indices.
- `layout.size` is the logical element count, while `layout.cosize` is the
  physical storage size the layout addresses (one past its largest offset).
  They are equal for compact layouts but `cosize` is larger for strided or
  hierarchical ones, so back a tensor with `cosize` elements — e.g. a strided
  `Layout(Shape([2, 3]), Stride([1, 4]))` has `size` 6 but `cosize` 10, so it
  needs `CPU(10)`. The `strideweave.friendly` and `strideweave.nn` layers allocate
  through `layout.cosize` for exactly this reason.
- Operations dispatch through `tensor.carrier.dispatch_op(operation_name)`.
  The base `Carrier` method owns the shared dispatch policy: it calls the
  backend `_dispatch_op` factory hook, requires a fresh `Operation`, and tags
  that operation with its canonical name and exact dispatching carrier class.
  Custom carrier implementations must override `_dispatch_op`, not
  `dispatch_op`. Dispatch is uniformly instance-based; class-level
  `dispatch_op` calls are not part of the public contract. A Python subclass of
  `CPU` may extend the native registry by handling custom names in `_dispatch_op`
  and delegating standard names through `super()._dispatch_op(...)`.
- Python and native operations inherit from the shared native `Operation` base.
- Views may use different layouts and offsets while sharing the same carrier.

For example, this creates a two-mode column-major tensor backed by a Python carrier:

```python
import strideweave as sw

layout = sw.Layout(
    sw.Shape([2, 3]),
    sw.Stride([1, 2]),
)
tensor = sw.Tensor(
    sw.Generic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    0,
    layout,
)

assert tensor[1, 2] == 6.0
```

The core namespace deliberately exposes only these composable primitives.
High-level factories such as `tensor`, `zeros`, and `ones` live in the
separate `strideweave.friendly` submodule (see Ergonomic Layers below); callers
working with the core API construct the carrier and layout explicitly.

## Carriers

StrideWeave currently provides four carrier implementations:

- `Generic(values, mutable=True, dtype=DType.Floating)` stores Python
  objects. It supports differentiable `Floating` values and non-differentiable
  arbitrary `Any` values.
- `CPU(size, pointer=None, mutable=True, dtype=DType.Float32)` owns native
  memory or references a caller-provided address. It supports `Float32` and
  `Int32`.
- `FileBacked(filename=None, mutable=True, dtype=DType.Floating)` stores raw
  numeric values in a temporary binary file. It is intended for storage and
  movement rather than direct tensor computation.
- `Evictable(primary, secondary)` composes two carriers into a memory
  hierarchy. Computation uses promoted primary storage; `evict()` moves values
  to secondary storage and blocks access until `promote()` restores them. Its
  constructor takes exclusive ownership of both supplied carriers.

The available dtype tags are `Any`, `Floating`, `Float32`, and `Int32`. Only
`Floating` and `Float32` tensors participate in autograd.

Carriers may be mutable or immutable. Mutating shared storage increments a version
counter visible through `tensor.version`. Calling `release()` permanently
releases a carrier's storage. Eviction and promotion belong specifically to
the composite `Evictable` carrier rather than the base `Carrier` or `Tensor` APIs.
A carrier owned by a composite carrier remains readable through retained aliases
while that tier is live, but rejects direct mutation, scatter, release, and
move operations. A tier may be released and replaced during a hierarchy
transition, after which an alias to the old tier is no longer readable. The
owning carrier retains privileged access so mutation through the composite
interface continues to follow its normal mutability and version rules.

`is_mutable()` reports whether public interfaces may currently write the carrier,
not only whether its storage was constructed mutable. Consequently, an
owned child reports `False` while its mutable owning composite may report
`True`. Carrier implementations define their intrinsic storage capability
through the private `_is_mutable()` hook; ownership is applied centrally by
`Carrier`.

```python
import strideweave as sw

primary = sw.Generic([1.0])
carrier = sw.Evictable(primary, sw.Generic([0.0]))

assert carrier.is_mutable()
assert primary.is_owned()
assert not primary.is_mutable()

carrier[0] = 2.0

try:
    primary[0] = 3.0
except RuntimeError:
    pass
```

## Operations

The public functional API includes:

- arithmetic: `add`, `sub`, `neg`, `mul`, `elementwise_mul`, `div`, `pow`, and
  `exp` (`sub` is implemented natively for CPU carriers and in Python for
  Generic carriers; `neg` is a composition of scalar `mul`);
- activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`,
  and `leaky_relu`;
- layout operations: `view`, `permute`, and `rearrange`;
- contractions: `reduce`, `matmul`, and `einsum`;
- storage movement: `move`.

`Generic` provides Python reference implementations. `CPU` provides native C++
kernels that use cached expanded layout keys and release the GIL in hot loops.
`FileBacked` does not dispatch computational operations.

An Evictable tensor dispatches through a public `EvictableOperation` adapter.
Each adapter owns one fresh operation from the primary carrier, lowers its
inputs to temporary primary-backed tensors, and invokes the primary operation's
generic lowered-execution route and `backward` method. Lowered execution shares
framework execution hooks and result validation with regular execution but does
not attach an inner autograd node or discard delegated state. The adapter is the
sole visible autograd node and wraps primary results and gradients back into the
same hierarchy. CPU and Generic implementations therefore do not need
composition-specific code.
New operation results allocate only their promoted primary storage. Their
secondary tier remains empty until the first eviction provisions it.

StrideWeave layout descriptions preserve hierarchical modes and therefore do not
have standard flat einops semantics. String forms include:

```python
transposed = sw.rearrange(tensor, "a b -> b a")
summed = sw.reduce(tensor, "a (b c) -> a b")
contracted = sw.einsum(lhs, rhs, "a b, c b -> a c")
```

The native lexer and Python parsers compile these descriptions into layout
trees and cache successful specifications.

## Operation Profiling

`profile` is a single-use context manager that records carrier-dispatched
operation executions on the current thread. It records execution attempts, not
dispatch factory lookups. Events contain the canonical operation name, exact
dispatching carrier and implementation classes, monotonic start time, inclusive
and self synchronous host wall time, thread identity, parent relationship, and
success status. With `record_shapes=True`, tensor argument positions also carry
immutable snapshots of their hierarchical shapes; non-tensor positions are
represented by `None`.

```python
import strideweave as sw

tensor = sw.Tensor(
    sw.CPU(6),
    0,
    sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2])),
)

with sw.profile(carriers={sw.CPU}, record_shapes=True) as prof:
    result = sw.relu(tensor)

events = prof.events()
averages = prof.key_averages(group_by_input_shape=True)
print(prof.table(sort_by="self_total_time_ns"))
```

`carriers=None` records every exact carrier class; an iterable such as
`{CPU, Evictable}` selects only those exact classes and does not retain carrier
instances. Nested composite execution is visible when selected: an Evictable
operation produces an outer Evictable event and a nested event for its promoted
CPU or Generic operation. Filtering out that nested event does not charge its
time to the parent's self time. Aggregates are derived from the immutable raw
events by operation name and carrier class, optionally adding input shapes to
the grouping key.

Profiling state is thread-local, so work on another thread requires its own
context, and a context must exit on the thread that entered it. A rejected
cross-thread exit abandons that registration so the owner thread recovers
before its next dispatched operation or profiler context. Timings measure
synchronous host wall time only; asynchronous
accelerator activity is not modeled. Directly instantiated operations without
carrier dispatch metadata and unannotated registry move operations are excluded.
Results become available after the context exits, including when its body raises;
the original exception still propagates.

## Autograd

Operations attach an autograd context when gradient construction is enabled,
the result is differentiable, and at least one tensor input is differentiable.
Backward traversal is iterative and topological, so shared subgraphs accumulate
their pending gradients before their operation runs.

- Non-scalar tensors require an explicit gradient in `backward(gradient)`.
- An exact shape `[1]` is a scalar and may call `backward()` with an implicit
  gradient of one.
- Leaf tensors accumulate `.grad` by default.
- Non-leaf tensors retain `.grad` only after `retain_grad()`.
- `no_grad()`, `is_grad_enabled()`, and `set_grad_enabled()` control the
  thread-local graph-building state.
- `Any` and `Int32` tensors reject gradient APIs.
- Backward validates saved input versions and raises if required storage was
  modified in place after the forward pass.

Views are differentiable. Their backward path scatters gradients into a tensor
with the original input layout.

## Modules

`Module` provides basic PyTorch-like structure: subclasses implement `forward`,
and `__call__` delegates to it. Assigning public `Parameter` or `Module`
attributes registers them for `modules()`, `parameters()`, and
`get_named_parameters()` traversal. Optional module and parameter names can
override attribute-name segments.

Buffers, state dictionaries, training/evaluation modes, and hooks are not
implemented yet. A minimal layer library and optimizer live in `strideweave.nn`
(see Ergonomic Layers).

## Ergonomic Layers

The core carriers stay composable primitives; user-facing ergonomics live
in two submodule-only packages that are built entirely from the public
primitives and are not re-exported at the top level.

### strideweave.nn

`import strideweave.nn as nn` provides `Linear`, activation module wrappers
(`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softplus`, `ELU`, `LeakyReLU`),
`MSELoss`, and an `SGD` optimizer.

Carrier and layout requirements differ per component and follow from what
each one actually does, rather than a blanket `strideweave.nn` restriction:

- The activation wrappers are thin `Module` adapters that delegate to the
  corresponding functional operation, so they carry no hyperparameters and
  inherit its input support: they accept any carrier and layout the underlying
  op accepts (e.g. a one-mode `Generic` tensor), not just CPU `Float32`.
- `Linear` holds CPU `Float32` parameters and uses matmul plus the ones-column
  bias trick, so it requires CPU inputs in the flat column-major `[batch,
  features]` convention below.
- `MSELoss` is composed from carrier-dispatched operations (`sub`, `pow`,
  reduction), so it works on any carrier that supports them (CPU or Generic);
  it does require the prediction and target to share a flat two-mode layout.
- `SGD` writes elementwise through each parameter's layout, so it works on any
  mutable parameter with a compatible gradient — no CPU or two-mode
  requirement.

Conventions: `Linear` inputs are flat column-major `[batch, features]` tensors
(`Layout(Shape([rows, cols]), Stride([1, rows]))`) and its weights are
`[out_features, in_features]`; because matmul contracts the second mode of
both operands, `x @ weight` yields `[batch, out_features]` directly. There is
no broadcasting primitive, so `Linear` broadcasts its `[out_features, 1]`
bias by contracting a constant ones column against it: `ones[batch, 1] @
bias[out, 1]` produces a tile whose layout matches the matmul output and
whose backward pass sums the bias gradient over the batch.

`SGD.step()` mutates parameter storage in place and therefore bumps carrier
versions: the required per-iteration ordering is forward, `backward()`, then
`step()`, and graphs built before a step cannot be backwarded afterwards.
Gradients accumulate until `SGD.zero_grad()` resets them to `None`.

`MSELoss` returns an exact single-mode `Shape(1)` scalar, so
`loss.backward()` needs no explicit gradient.

### strideweave.friendly

`import strideweave.friendly as F` provides compact layout builders
(`column_major`, `row_major`), CPU tensor factories (`tensor` from nested
lists, `zeros`, `ones`, `full`, `arange`, `rand`, `randn`), scalar reductions
(`sum`, `mean`, both returning `Shape(1)` tensors that support implicit
`backward()`), and value extraction (`item`, `to_list`).

End-to-end training examples live in `examples/train_mlp_cpu.py` (raw
primitives) and `examples/train_mlp_cpu_friendly.py` (same model using the
helpers).

## Interoperability And Movement

CPU tensors support DLPack export through `__dlpack__` and
`__dlpack_device__`. Hierarchical shapes and strides are flattened for the
DLPack representation. Generic, FileBacked, and Evictable carriers do not support
DLPack, and copy or cross-device exports are not implemented.

`move(tensor, destination)` dispatches on the exact source and destination carrier
classes. CPU-to-FileBacked and FileBacked-to-CPU moves use native bulk copies;
unregistered pairs use an elementwise fallback. A successful move releases the
source carrier, and autograd moves gradients back into fresh source-class storage.

Evictable resolves the move registry for each transition and routes move
operations through the framework-owned sealed lowered-execution path, so
residency changes receive shared result validation without adding autograd
nodes. Element access, forward operations, and scatter are unavailable while
values are evicted. Backward may still run while an operation result is evicted because
the result storage is not read; saved inputs and the supplied gradient must be
promoted.

Residency transitions publish replacement tiers only after a move succeeds. If
a move implementation raises, the prior residency state and ownership remain
valid and the transition may be retried.

Ownership guards apply to carrier interfaces. Explicit external-memory
escape hatches such as `CPU.pointer()` and direct writes to a `FileBacked` path
remain the caller's responsibility and cannot participate in version tracking.
The same applies to direct mutation of a mutable container originally supplied
to `Generic`, because the container remains an alias of Generic storage.

## Current Boundaries

- No CUDA, Metal, or other accelerator carriers.
- High-level tensor creation lives only in `strideweave.friendly` and is
  CPU-backed; other carriers are constructed from primitives.
- No general broadcasting system; binary operations require compatible layouts
  and generally the same backing carrier (`strideweave.nn` composes around
  this with matmul-based bias broadcasting).
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
- Evictable tensors must be promoted before access or computation, and binary
  Evictable operations require matching primary and secondary carrier classes.
- `strideweave.nn` covers only `Linear`, elementwise activations, `MSELoss`, and
  `SGD`; there are no buffers, state dictionaries, training/evaluation modes,
  or hooks.

## Development

The package requires Python 3.12 or newer and builds its native modules with
scikit-build-core and pybind11.

Before designing a change, read the cross-cutting contracts in
[`INVARIANTS.md`](INVARIANTS.md). It records the canonical implementation choices and
whether each invariant is enforced by AST lint, Ruff, behavioral tests, native builds,
or code review.

```bash
uv sync --group dev
uv run pytest tests
uv run ruff format --check .
uv run ruff check .
uv run python tools/lint_invariants.py
uv run pyright
uv build
find src/strideweave -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec uv run clang-format --dry-run --Werror {} +
CMAKE_ARGS="-DSTRIDEWEAVE_STRICT_WARNINGS=ON" uv build
```

The repository invariant checker uses Python's built-in AST and reports
StrideWeave-specific source contracts without importing the package. Native sanitizer
coverage runs in Linux CI with `STRIDEWEAVE_SANITIZERS=ON`; it instruments the extension
modules with AddressSanitizer and UndefinedBehaviorSanitizer before running the full
Python test suite.

The test suite covers layouts, carriers, tensor indexing and mutation,
autograd, operations and activations, hierarchical command parsing, DLPack,
movement, modules, and public docstrings.

## License

StrideWeave is licensed under the Apache License, Version 2.0. See `LICENSE`.
