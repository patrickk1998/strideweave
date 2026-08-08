---
title: Friendly Tensor Creation
publish: true
status: stable
order: 10
summary: How friendly CPU tensor creation validates an explicitly supplied layout before consuming the caller's values.
---

# friendly-tensor-creation Specification

## Purpose
Define how friendly CPU tensor creation validates an explicitly supplied layout
so callers receive stable diagnostics while valid layouts retain their existing
logical filling behavior.
## Requirements
### Requirement: Explicit layout arguments are validated

The friendly tensor creation API SHALL accept only a `Layout` or `None` for its
optional explicit layout argument. When the argument is neither, the API MUST
raise `TypeError` with a message identifying the layout requirement before it
consumes the supplied values or accesses layout properties.

#### Scenario: Invalid explicit layout

- **WHEN** a caller supplies a non-`None` object that is not a `Layout` as the
  explicit layout argument
- **THEN** tensor creation raises `TypeError` with the message
  `layout must be a Layout`
- **AND** the supplied values are not consumed

#### Scenario: Valid explicit layout

- **WHEN** a caller supplies a `Layout` and exactly one value for each logical
  element in that layout
- **THEN** tensor creation returns a CPU-backed tensor using that layout
- **AND** the values are stored in the layout's logical index order
