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

There are not yet high-level factories such as `tensor`, `zeros`, or `ones`,
so callers currently construct the data and layout explicitly.

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

- arithmetic: `add`, `mul`, `elementwise_mul`, `div`, `pow`, and `exp`;
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

Buffers, state dictionaries, training/evaluation modes, hooks, optimizers, and
a standard layer library are not implemented yet.

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
- No high-level tensor creation functions.
- No general broadcasting system; binary operations require compatible layouts
  and generally the same backing data class.
- DLPack support is export-only and currently CPU-only.
- `FileBacked` supports storage and movement, not direct computation.
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
