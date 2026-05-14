# neotorch

## Tensor Indexing Style

Prefer `tensor[i, j]` style for coordinate indexing in source code and tests.

Use `tensor[[i, j]]` only when intentionally testing or documenting list-coordinate
key support, such as checking that `tensor[[i, j]]` behaves like `tensor[i, j]`.
