## Why

`strideweave.friendly.tensor` accepts an optional explicit layout but does not
validate that argument before accessing layout attributes. Invalid arguments
therefore produce incidental errors instead of a clear public API diagnostic.

## What Changes

- Validate that a non-`None` explicit `layout` is a `Layout`.
- Raise a clear `TypeError` before consuming values or accessing layout
  properties.
- Add regression coverage while preserving valid explicit-layout behavior.

## Capabilities

### New Capabilities

- `friendly-tensor-creation`: Define explicit-layout validation behavior for
  the friendly CPU tensor factory.

### Modified Capabilities

None. The project has no existing OpenSpec capability for friendly tensor
creation.

## Impact

- `src/strideweave/friendly/creation.py`: Add focused explicit-layout
  validation.
- `tests/test_friendly.py`: Add regression coverage for invalid layout
  arguments.
- `llms.md` and `INVARIANTS.md`: No changes expected because architecture,
  public capabilities, and cross-cutting invariants remain unchanged.
