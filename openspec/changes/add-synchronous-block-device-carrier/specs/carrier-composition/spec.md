## MODIFIED Requirements

### Requirement: Evictable construction validates a complete two-tier hierarchy

`Evictable(primary, secondary)` SHALL return a promoted hierarchy and take
exclusive ownership of both supplied tiers. `primary` names the live carrier
whose current values initialize the hierarchy and whose exact class performs
normal operation dispatch. `secondary` names a distinct live mutable carrier
that can receive evicted values.

Both arguments SHALL be Carrier instances. Supported v0 tier kinds SHALL exclude
`BlockDeviceCarrier` in both positions. After validating that both arguments are
Carrier instances, construction SHALL validate this tier-kind restriction before
distinctness, lifecycle, ownership, mutability, dtype, or move-availability checks.
If either exact tier class is `BlockDeviceCarrier`, construction SHALL fail with
`TypeError` identifying the unsupported tier kind. It SHALL return no hierarchy,
claim no ownership, allocate no storage, issue no device I/O, and leave both tiers'
values, versions, reported sizes, offsets, extents, lifecycle, and device health
unchanged. Every existing extent SHALL remain associated with its current carrier,
and later allocation availability SHALL be unchanged.

For supported tier kinds, both arguments SHALL be distinct objects, SHALL be
unreleased and unowned, and SHALL have identical dtype descriptor identities.
The secondary SHALL be publicly mutable. The primary SHALL contain at least one
element. A non-Carrier SHALL fail with `TypeError`; identical tiers or an empty
primary SHALL fail with `ValueError`; a released, owned, or immutable required
tier SHALL fail with `RuntimeError`; mismatched dtypes SHALL fail with
`TypeError`.

Both directions between the exact tier classes SHALL have a registered move
implementation. If either direction is unavailable, construction SHALL fail
before returning a hierarchy.

#### Scenario: Construct a promoted hierarchy

- **WHEN** a live non-empty primary and a distinct live mutable secondary have
  supported tier kinds, the same dtype, and moves in both directions
- **THEN** construction returns a promoted Evictable that owns both tiers and
  reports the primary's size and dtype

#### Scenario: Reject mismatched tier dtypes

- **WHEN** the primary and secondary dtypes are not the same descriptor object
- **THEN** construction fails with `TypeError` and neither tier becomes owned

#### Scenario: Reject a block primary tier

- **WHEN** `primary` is a BlockDeviceCarrier and `secondary` is another Carrier
- **THEN** construction fails with `TypeError` before ordinary tier checks,
  ownership, allocation, or device I/O, leaving both tiers unchanged

#### Scenario: Reject a block secondary tier

- **WHEN** `secondary` is a BlockDeviceCarrier and `primary` is another Carrier
- **THEN** construction fails with `TypeError` before ordinary tier checks,
  ownership, allocation, or device I/O, leaving both tiers unchanged
