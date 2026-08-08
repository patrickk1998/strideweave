## Context

See `proposal.md` for motivation. The explicit-layout branch currently converts
`values` to a list and reads `layout.size` without first establishing that the
argument satisfies the documented `Layout | None` contract. Existing valid
layout construction already allocates by `layout.cosize` and fills in logical
index order; those mechanics must remain unchanged.

## Goals / Non-Goals

**Goals:**

- Establish a deterministic public validation boundary for the explicit layout
  argument.
- Validate the layout before any value iteration or layout-property access.
- Preserve all valid explicit-layout construction behavior.

**Non-Goals:**

- Changing nested-value inference, dtype validation, or value-count rules.
- Generalizing the factory to non-CPU carriers.
- Introducing a shared validation abstraction for unrelated APIs.

## Decisions

### Decision 1: Validate with `isinstance` at the public factory boundary

The explicit-layout branch will first require `isinstance(layout, Layout)` and
raise `TypeError("layout must be a Layout")` when it fails. This is preferred to
duck typing because the public annotation and downstream tensor construction
require the concrete layout contract. It is preferred to relying on downstream
attribute or constructor failures because those diagnostics vary with the
invalid object and may occur after values have been consumed.

### Decision 2: Cover validation order and compatibility in focused tests

The regression test will use values whose iteration would fail if reached, so
it proves that layout validation occurs first. Existing explicit-layout tests
continue to cover logical-order filling, avoiding duplicate compatibility
fixtures.

## Risks / Trade-offs

- **Risk:** A caller relying on an incidental `AttributeError` will now observe
  `TypeError`. **Mitigation:** This aligns runtime behavior with the documented
  public signature and gives a stable, argument-focused diagnostic.

## Migration Plan

No data or deployment migration is required. Rollback consists of reverting the
validation and its regression test.
