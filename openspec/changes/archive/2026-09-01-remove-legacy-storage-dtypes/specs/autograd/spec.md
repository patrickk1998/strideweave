## MODIFIED Requirements

### Requirement: Gradient participation follows the logical dtype

`tensor.is_differentiable()` SHALL return `True` only when the Tensor's logical
dtype is the exact descriptor `DType.Float32`, and SHALL return `False` for
every other logical dtype, including `Int32`, `Bool`, every `DTypeCategory`,
every other `SimpleDType`, and every `CompoundDType`. Differentiability SHALL
depend on the logical dtype alone and SHALL be identical across carrier
implementations.

Reading `tensor.grad`, calling `tensor.retain_grad(retain)`, calling
`tensor.backward(gradient, retain_graph)`, and assigning a value other than
`None` to `tensor.autograd_ctx` or `tensor.grad` SHALL fail with `RuntimeError`
when the Tensor is not differentiable. Assigning `None` to `tensor.grad` or
`tensor.autograd_ctx` SHALL succeed for every Tensor.

`tensor.grad` SHALL be `None` until a gradient is accumulated or assigned, and
`tensor.autograd_ctx` SHALL be `None` until a forward call records a node.

#### Scenario: Report differentiability from the logical dtype

- **WHEN** a caller reads `is_differentiable()` on a Float32 Tensor and on
  constructible Tensors of other logical dtypes
- **THEN** only the Float32 Tensor reports `True`

#### Scenario: Reject gradient APIs on a non-differentiable Tensor

- **WHEN** a caller reads `grad`, calls `retain_grad()`, calls `backward()`, or
  assigns a non-`None` `autograd_ctx` on an Int32 or Bool Tensor
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
once per positional occurrence, while a positional argument that is not a
Tensor and the `options` execution-option keyword SHALL remain outside them.

#### Scenario: Record a node for a differentiable result

- **WHEN** a caller invokes an operation whose positional arguments include a
  Float32 Tensor, a non-Tensor value, and an `options` keyword, and whose result
  is Float32
- **THEN** the result's `autograd_ctx` is that operation
- **AND** the saved Tensor inputs contain only the positional Tensor arguments
  in argument order, each with its saved input version

#### Scenario: Record no node for a non-differentiable result

- **WHEN** an operation on Int32 or Bool Tensors produces a
  non-differentiable result
- **THEN** the result's `autograd_ctx` is `None`
- **AND** the operation retains no saved Tensor inputs
