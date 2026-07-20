# neotorch

Neotorch is a research tensor and autograd framework built around hierarchical,
CuTe-style layouts. A tensor combines a data backend, a physical offset, and a
layout; tensor operations are selected by the backing data class rather than by
a separate device abstraction.

The project is currently a tested prototype rather than a complete PyTorch
replacement. It provides native CPU kernels, a Python reference backend,
autograd, hierarchical layout transformations, a small module system, and
ergonomic layers (`neotorch.nn` with a minimal layer library and optimizer,
`neotorch.friendly` with tensor factories), but does not yet include
accelerator backends.

## Core Model

- `Tensor(data, offset, layout)` references storage owned by a `Data` object.
- `Layout` describes hierarchical `Shape` and `Stride` trees and maps logical
  coordinates to physical storage indices.
- `layout.size` is the logical element count, while `layout.cosize` is the
  physical storage size the layout addresses (one past its largest offset).
  They are equal for compact layouts but `cosize` is larger for strided or
  hierarchical ones, so back a tensor with `cosize` elements — e.g. a strided
  `Layout(Shape([2, 3]), Stride([1, 4]))` has `size` 6 but `cosize` 10, so it
  needs `CPU(10)`. The `neotorch.friendly` and `neotorch.nn` layers allocate
  through `layout.cosize` for exactly this reason.
- Operations dispatch through `type(tensor.data).dispatch_op(operation_name)`.
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

The core namespace deliberately exposes only these composable primitives.
High-level factories such as `tensor`, `zeros`, and `ones` live in the
separate `neotorch.friendly` submodule (see Ergonomic Layers below); callers
working with the core API construct the data and layout explicitly.

## Data Backends

Neotorch currently provides three data classes:

- `Generic(values, mutable=True, dtype=DataType.Floating)` stores Python
  objects. It supports differentiable `Floating` values and non-differentiable
  arbitrary `Any` values.
- `CPU(size, pointer=None, mutable=True, dtype=DataType.Float32)` owns native
  memory or references a caller-provided address. It supports `Float32` and
  `Int32`.
- `FileBacked(filename=None, mutable=True, dtype=DataType.Floating)` stores raw
  numeric values in a temporary binary file. It is intended for storage and
  movement rather than direct tensor computation.

The available dtype tags are `Any`, `Floating`, `Float32`, and `Int32`. Only
`Floating` and `Float32` tensors participate in autograd.

Data may be mutable or immutable. Mutating shared storage increments a version
counter visible through `tensor.version`. Calling `release()` permanently
releases a data object's storage; the older eviction and promotion interface no
longer exists.

## Operations

The public functional API includes:

- arithmetic: `add`, `sub`, `neg`, `mul`, `elementwise_mul`, `div`, `pow`, and
  `exp` (`sub` is a backend operation, implemented natively for CPU data and in
  Python for Generic data; `neg` is a composition of scalar `mul`);
- activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`,
  and `leaky_relu`;
- layout operations: `view`, `permute`, and `rearrange`;
- contractions: `reduce`, `matmul`, and `einsum`;
- storage movement: `move`.

`Generic` provides Python reference implementations. `CPU` provides native C++
kernels that use cached expanded layout keys and release the GIL in hot loops.
`FileBacked` does not dispatch computational operations.

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

Buffers, state dictionaries, training/evaluation modes, and hooks are not
implemented yet. A minimal layer library and optimizer live in `neotorch.nn`
(see Ergonomic Layers).

## Ergonomic Layers

The core data classes stay composable primitives; user-facing ergonomics live
in two submodule-only packages that are built entirely from the public
primitives and are not re-exported at the top level.

### neotorch.nn

`import neotorch.nn as nn` provides `Linear`, activation module wrappers
(`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softplus`, `ELU`, `LeakyReLU`),
`MSELoss`, and an `SGD` optimizer.

Backend and layout requirements differ per component and follow from what
each one actually does, rather than a blanket `neotorch.nn` restriction:

- The activation wrappers are thin `Module` adapters that delegate to the
  corresponding functional operation, so they carry no hyperparameters and
  inherit its input support: they accept any backend and layout the underlying
  op accepts (e.g. a one-mode `Generic` tensor), not just CPU `Float32`.
- `Linear` holds CPU `Float32` parameters and uses matmul plus the ones-column
  bias trick, so it requires CPU inputs in the flat column-major `[batch,
  features]` convention below.
- `MSELoss` is composed from backend-dispatched operations (`sub`, `pow`,
  reduction), so it works on any backend that supports them (CPU or Generic);
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

`SGD.step()` mutates parameter storage in place and therefore bumps data
versions: the required per-iteration ordering is forward, `backward()`, then
`step()`, and graphs built before a step cannot be backwarded afterwards.
Gradients accumulate until `SGD.zero_grad()` resets them to `None`.

`MSELoss` returns an exact single-mode `Shape(1)` scalar, so
`loss.backward()` needs no explicit gradient.

### neotorch.friendly

`import neotorch.friendly as F` provides compact layout builders
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
DLPack representation. Generic and FileBacked data do not support DLPack, and
copy or cross-device exports are not implemented.

`move(tensor, destination)` dispatches on the exact source and destination data
classes. CPU-to-FileBacked and FileBacked-to-CPU moves use native bulk copies;
unregistered pairs use an elementwise fallback. A successful move releases the
source data, and autograd moves gradients back into fresh source-class storage.

## Current Boundaries

- No CUDA, Metal, or other accelerator backend.
- High-level tensor creation lives only in `neotorch.friendly` and is
  CPU-backed; other data classes are constructed from primitives.
- No general broadcasting system; binary operations require compatible layouts
  and generally the same backing data class (`neotorch.nn` composes around
  this with matmul-based bias broadcasting).
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
- `neotorch.nn` covers only `Linear`, elementwise activations, `MSELoss`, and
  `SGD`; there are no buffers, state dictionaries, training/evaluation modes,
  or hooks.

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
