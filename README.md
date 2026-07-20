# neotorch

Neotorch is a research tensor and autograd framework built around hierarchical,
CuTe-style layouts. A tensor combines a data backend, a physical offset, and a
layout; tensor operations are selected by the backing data class rather than by
a separate device abstraction.

The project is currently a tested prototype rather than a complete PyTorch
replacement. It provides native CPU kernels, a Python reference backend,
autograd, hierarchical layout transformations, and a small module system, but
does not yet include optimizers, a layer library, or accelerator backends.

## Core Model

- `Tensor(data, offset, layout)` references storage owned by a `Data` object.
- `Layout` describes hierarchical `Shape` and `Stride` trees and maps logical
  coordinates to physical storage indices.
- Operations dispatch through `tensor.data.dispatch_op(operation_name)`.
  Dispatch is uniformly instance-based; class-level `dispatch_op` calls are
  not part of the public contract.
- Python and native operations inherit from the shared native `Operation` base.
- Views may use different layouts and offsets while sharing the same data.

For example, this creates a two-mode column-major tensor backed by Python data:

```python
import neotorch

layout = neotorch.Layout(
    neotorch.Shape([2, 3]),
    neotorch.Stride([1, 2]),
)
tensor = neotorch.Tensor(
    neotorch.Generic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    0,
    layout,
)

assert tensor[1, 2] == 6.0
```

There are not yet high-level factories such as `tensor`, `zeros`, or `ones`,
so callers currently construct the data and layout explicitly.

## Data Backends

Neotorch currently provides four data classes:

- `Generic(values, mutable=True, dtype=DataType.Floating)` stores Python
  objects. It supports differentiable `Floating` values and non-differentiable
  arbitrary `Any` values.
- `CPU(size, pointer=None, mutable=True, dtype=DataType.Float32)` owns native
  memory or references a caller-provided address. It supports `Float32` and
  `Int32`.
- `FileBacked(filename=None, mutable=True, dtype=DataType.Floating)` stores raw
  numeric values in a temporary binary file. It is intended for storage and
  movement rather than direct tensor computation.
- `Evictable(primary, secondary)` composes two data instances into a memory
  hierarchy. Computation uses promoted primary storage; `evict()` moves values
  to secondary storage and blocks access until `promote()` restores them. Its
  constructor takes exclusive ownership of both supplied data objects.

The available dtype tags are `Any`, `Floating`, `Float32`, and `Int32`. Only
`Floating` and `Float32` tensors participate in autograd.

Data may be mutable or immutable. Mutating shared storage increments a version
counter visible through `tensor.version`. Calling `release()` permanently
releases a data object's storage. Eviction and promotion belong specifically to
the composite `Evictable` backend rather than the base `Data` or `Tensor` APIs.
Data owned by a composite backend remains readable through retained aliases
while that tier is live, but rejects direct mutation, scatter, release, and
move operations. A tier may be released and replaced during a hierarchy
transition, after which an alias to the old tier is no longer readable. The
owning backend retains privileged access so mutation through the composite
interface continues to follow its normal mutability and version rules.

`is_mutable()` reports whether public interfaces may currently write the data,
not only whether its backend storage was constructed mutable. Consequently, an
owned child reports `False` while its mutable owning composite may report
`True`. Backend implementations define their intrinsic storage capability
through the private `_is_mutable()` hook; ownership is applied centrally by
`Data`.

```python
primary = neotorch.Generic([1.0])
data = neotorch.Evictable(primary, neotorch.Generic([0.0]))

assert data.is_mutable()
assert primary.is_owned()
assert not primary.is_mutable()

data[0] = 2.0

try:
    primary[0] = 3.0
except RuntimeError:
    pass
```

## Operations

The public functional API includes:

- arithmetic: `add`, `mul`, `elementwise_mul`, `div`, `pow`, and `exp`;
- activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`,
  and `leaky_relu`;
- layout operations: `view`, `permute`, and `rearrange`;
- contractions: `reduce`, `matmul`, and `einsum`;
- storage movement: `move`.

`Generic` provides Python reference implementations. `CPU` provides native C++
kernels that use cached expanded layout keys and release the GIL in hot loops.
`FileBacked` does not dispatch computational operations.

An Evictable tensor dispatches through a public `EvictableOperation` adapter.
Each adapter owns one fresh operation from the primary data class, lowers its
inputs to temporary primary-backed tensors, and invokes the primary operation's
`_forward` and `backward` methods directly. The adapter is the visible autograd
node and wraps primary results and gradients back into the same hierarchy. CPU
and Generic implementations therefore do not need composition-specific code.
New operation results allocate only their promoted primary storage. Their
secondary tier remains empty until the first eviction provisions it.

Neotorch layout descriptions preserve hierarchical modes and therefore do not
have standard flat einops semantics. String forms include:

```python
transposed = neotorch.rearrange(tensor, "a b -> b a")
summed = neotorch.reduce(tensor, "a (b c) -> a b")
contracted = neotorch.einsum(lhs, rhs, "a b, c b -> a c")
```

The native lexer and Python parsers compile these descriptions into layout
trees and cache successful specifications.

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

Buffers, state dictionaries, training/evaluation modes, hooks, optimizers, and
a standard layer library are not implemented yet.

## Interoperability And Movement

CPU tensors support DLPack export through `__dlpack__` and
`__dlpack_device__`. Hierarchical shapes and strides are flattened for the
DLPack representation. Generic, FileBacked, and Evictable data do not support
DLPack, and copy or cross-device exports are not implemented.

`move(tensor, destination)` dispatches on the exact source and destination data
classes. CPU-to-FileBacked and FileBacked-to-CPU moves use native bulk copies;
unregistered pairs use an elementwise fallback. A successful move releases the
source data, and autograd moves gradients back into fresh source-class storage.

Evictable resolves the move registry for each transition and calls move
`_forward` directly so residency changes do not add autograd nodes. Element
access, forward operations, and scatter are unavailable while values are
evicted. Backward may still run while an operation result is evicted because
the result storage is not read; saved inputs and the supplied gradient must be
promoted.

Residency transitions publish replacement tiers only after a move succeeds. If
a move implementation raises, the prior residency state and ownership remain
valid and the transition may be retried.

Ownership guards apply to data-class interfaces. Explicit external-memory
escape hatches such as `CPU.pointer()` and direct writes to a `FileBacked` path
remain the caller's responsibility and cannot participate in version tracking.
The same applies to direct mutation of a mutable container originally supplied
to `Generic`, because the container remains an alias of Generic storage.

## Current Boundaries

- No CUDA, Metal, or other accelerator backend.
- No high-level tensor creation functions.
- No general broadcasting system; binary operations require compatible layouts
  and generally the same backing data class.
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
- Evictable tensors must be promoted before access or computation, and binary
  Evictable operations require matching primary and secondary backend classes.
- The module API is foundational and does not yet constitute a neural-network
  training stack.

## Development

The package requires Python 3.12 or newer and builds its native modules with
scikit-build-core and pybind11.

```bash
uv pip install -e packages/neotorch
uv run pytest packages/neotorch/tests
uv run ruff format --check . --exclude notebooks/CuTe.py
uv run ruff check packages/neotorch
uv run pyright packages/neotorch
```

The test suite covers layouts, data backends, tensor indexing and mutation,
autograd, operations and activations, hierarchical command parsing, DLPack,
movement, modules, and public docstrings.
