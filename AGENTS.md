# neotorch

## Project Orientation

Read `packages/neotorch/README.md` before making changes. It describes the
project's architecture, data backends, operation dispatch, autograd model,
supported public APIs, interoperability, and current limitations.

When planning or reviewing a change, check whether it would make any part of
the README inaccurate or incomplete. Propose corresponding README updates for
changes to architecture, data backends, dtypes, operation dispatch, autograd,
public capabilities, interoperability, development commands, or documented
limitations. When implementing such a change, update the README in the same
change unless the user explicitly excludes documentation work.

## Tensor Indexing Style

Prefer `tensor[i, j]` style for coordinate indexing in source code and tests.

Use `tensor[[i, j]]` only when intentionally testing or documenting list-coordinate
key support, such as checking that `tensor[[i, j]]` behaves like `tensor[i, j]`.

## Public API Docstrings

Public Python API docstrings must document purpose and semantics, every
relevant input, an appropriate output description, and at least one concrete
usage example. The exact contract depends on the export kind:

- Function exports document purpose, every input, the return value, and an
  example (`Args:`, `Returns:`, `Examples:`).
- Class exports in `neotorch.nn` document purpose, every constructor input
  (including inherited ones such as `name`), and an example (`Args:`,
  `Examples:`). A constructor documents construction rather than a return
  value, so `Returns:` is intentionally not required on the class docstring;
  public methods defined on the class are additionally checked under the full
  function contract.
- Other public class exports surfaced for historical import paths or `isinstance`
  checks — the `neotorch.operation` operation classes (for example
  `GenericAddOperation`, `GenericSubOperation`) and the native data/layout
  classes — are implementation and dispatch classes rather than
  constructor-driven user APIs. They are checked for docstring presence only
  and carry a one-line summary; do not add constructor-style `Args:`/`Examples:`
  sections to them.

`packages/neotorch/tests/test_docstrings.py` enforces this generically over the
public exports listed in `neotorch.__all__`, `neotorch.einops.__all__`,
`neotorch.nn.__all__`, and `neotorch.friendly.__all__`. Adding a Python
function or class to any of these public export lists should not require
changing that test unless it needs a stricter contract.

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
