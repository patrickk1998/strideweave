# neotorch

## Tensor Indexing Style

Prefer `tensor[i, j]` style for coordinate indexing in source code and tests.

Use `tensor[[i, j]]` only when intentionally testing or documenting list-coordinate
key support, such as checking that `tensor[[i, j]]` behaves like `tensor[i, j]`.

## Public API Docstrings

Public Python API docstrings must document purpose and semantics, every input,
the return value, and at least one concrete usage example.

`packages/neotorch/tests/test_docstrings.py` generically checks public exports
listed in `neotorch.__all__` and `neotorch.einops.__all__`. Adding a Python
function to either public export list should not require changing that test
unless it needs a stricter contract.

Modify the docstring test only when:

- adding a new native/pybind export that cannot reasonably carry a Python
  docstring, in which case add it to the explicit native skip set;
- adding a new layout-command parser or tensor operation that should require
  `Syntax:`, `Semantics:`, or `Mode assumptions:` sections, in which case add it
  to the appropriate explicit set;
- changing the public docstring contract itself.

For Neotorch layout description APIs, avoid wording that implies standard
einops/PyTorch flat-layout semantics. Describe commands as Neotorch
hierarchical-layout descriptions, and document syntax, semantics, and tensor
mode assumptions explicitly.

Do not write vague or flat-layout docstrings like:

```python
def einsum(lhs, rhs, description):
    """Contract two tensors using an einops-style einsum description."""
```

Prefer docstrings that name Neotorch semantics and include inputs, output, and
an example:

```python
def einsum(lhs, rhs, description):
    """Contract two tensors using a Neotorch contraction description.

    Shared symbols are lowered into the second mode of two intermediate layouts
    and contracted with matmul.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor containing the requested contraction result.

    Examples:
        >>> neotorch.einsum(lhs, rhs, "a b, c b -> a c")
    """
```
