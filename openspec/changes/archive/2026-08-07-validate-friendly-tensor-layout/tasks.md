## 1. Explicit Layout Validation

- [x] 1.1 Validate a non-None explicit layout with `isinstance(layout, Layout)` before consuming values or accessing layout properties.
- [x] 1.2 Raise `TypeError("layout must be a Layout")` when validation fails.

## 2. Regression Coverage

- [x] 2.1 Add a test proving an invalid layout raises the specified TypeError before values are consumed.
- [x] 2.2 Verify the existing valid explicit-layout logical-order behavior remains passing.

## 3. Validation

- [x] 3.1 Run the focused friendly API tests.
- [x] 3.2 Run Ruff, the invariant checker, and Pyright.
- [x] 3.3 Validate the completed OpenSpec change.
