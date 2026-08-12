---
title: Carrier Composition
publish: true
status: stable
order: 80
summary: Evictable hierarchy construction, exclusive ownership, residency transitions, and lifecycle.
---

# carrier-composition Specification

## Purpose

Define how Evictable composes two carriers into one exclusively owned storage
hierarchy and moves values between promoted and evicted residency without
changing their logical identity.

## Terminology

| Term | Meaning |
| --- | --- |
| primary tier | The owned `Carrier` instance that holds the hierarchy's complete values while promoted and whose exact class performs normal value access and operation execution. |
| secondary tier | The distinct owned mutable `Carrier` instance used to receive and retain the hierarchy's complete values during eviction and whose exact class defines the evicted-storage and reverse-move boundary; while the hierarchy is promoted, this tier may be empty. |
| promoted | Residency state in which complete values are available through the primary tier. |
| evicted | Residency state in which complete values reside in the secondary tier and primary access is unavailable. |
| exclusive ownership | The relationship granting one composite privileged control over a child carrier's mutation, movement, release, and replacement while retained external aliases may read live storage but cannot perform those controlled actions. |
| transition | An `evict()` or `promote()` move that changes residency without changing logical values. |

## Requirements

### Requirement: Evictable construction validates a complete two-tier hierarchy

`Evictable(primary, secondary)` SHALL return a promoted hierarchy and take
exclusive ownership of both supplied tiers. `primary` names the live carrier
whose current values initialize the hierarchy and whose exact class performs
normal operation dispatch. `secondary` names a distinct live mutable carrier
that can receive evicted values.

Both arguments SHALL be Carrier instances, SHALL be distinct objects, SHALL be
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
  the same dtype and support moves in both directions
- **THEN** construction returns a promoted Evictable that owns both tiers and
  reports the primary's size and dtype

#### Scenario: Reject mismatched tier dtypes

- **WHEN** the primary and secondary dtypes are not the same descriptor object
- **THEN** construction fails with `TypeError` and neither tier becomes owned

### Requirement: Construction failure leaves ownership unchanged

Evictable construction SHALL validate ordinary tier preconditions before
claiming ownership. If claiming the secondary fails after the primary was
claimed, or if per-instance capability finalization fails after both were
claimed, construction SHALL relinquish every claim it made. No failed
construction SHALL leave a supplied tier owned by an unreachable hierarchy.

#### Scenario: Fail while claiming the secondary

- **WHEN** primary ownership succeeds and secondary ownership cannot be
  completed
- **THEN** construction fails and the primary is returned to its previous
  unowned state

#### Scenario: Fail while freezing hierarchy capabilities

- **WHEN** the hierarchy's dependent capability generation fails
- **THEN** construction returns no Evictable and both supplied tiers are
  unowned afterwards

### Requirement: The hierarchy exposes current tier identities

`primary` SHALL return the current primary-tier carrier and `secondary` SHALL
return the current secondary-tier carrier. These properties SHALL retain exact
carrier objects rather than copies. A transition MAY replace a tier with fresh
storage of the same exact class; after replacement, the property SHALL return
the replacement.

`size()` SHALL return the hierarchy's logical physical-slot count while live,
and `dtype()` SHALL return the dtype identity captured from the initial tiers.
`is_evicted()` SHALL return `False` after construction and promotion and `True`
after successful eviction.

#### Scenario: Inspect initial tiers

- **WHEN** an Evictable has just been constructed
- **THEN** `primary` and `secondary` are the supplied objects and
  `is_evicted()` is `False`

### Requirement: Child ownership makes retained aliases externally read-only

While a tier is owned by an Evictable, `tier.is_owned()` SHALL return `True`.
A retained alias MAY read the tier while its storage remains live, but public
mutation, scatter, direct version increment, release, and direct move SHALL
fail with `RuntimeError`. An intrinsically mutable owned tier SHALL report
`is_mutable() == False` to external callers.

The Evictable owner SHALL have privileged access sufficient to mutate, move,
release, or replace its children through the composite interface. That access
SHALL not make the tier publicly mutable to another thread or retained alias.
Ownership SHALL compose: an Evictable used as a tier of another hierarchy SHALL
be protected by the same rules.

#### Scenario: Read but do not mutate an owned tier alias

- **WHEN** a caller retains a live primary alias after constructing an
  Evictable
- **THEN** the alias can read its values but direct write or release fails with
  `RuntimeError`

#### Scenario: Mutate through the owner

- **WHEN** the outer Evictable is mutable and promoted
- **THEN** an indexed write through the Evictable succeeds even though the
  primary alias reports not mutable

### Requirement: Composite mutability is captured from the primary

At construction, Evictable SHALL capture whether public mutation was permitted
by the primary. `is_mutable()` on the hierarchy SHALL report that captured
intrinsic policy subject to any ownership of the hierarchy itself. The
secondary's requirement to be mutable SHALL not make an immutable-primary
hierarchy mutable.

Promotion SHALL preserve the hierarchy's captured mutability. When an immutable
hierarchy needs fresh primary storage, the transition MAY use mutable storage
internally to complete the move, but public mutation through the restored
hierarchy SHALL remain unavailable.

#### Scenario: Preserve an immutable hierarchy

- **WHEN** an Evictable was constructed from an immutable primary and later
  evicted and promoted
- **THEN** the hierarchy still reports `is_mutable() == False` and rejects
  public writes

### Requirement: Promoted state is required for public data access and execution

While promoted, `get_value`, indexed reads, permitted writes, scatter, and
operation dispatch SHALL act through the primary tier. While evicted, those
interfaces SHALL fail with `RuntimeError` directing the caller to `promote()`.
`is_evicted()`, `dtype()`, structural storage support, structural operation
capabilities, `promote()`, and release SHALL remain available in the evicted
state.

Evictable SHALL not expose a DLPack buffer in either residency state; a DLPack
request SHALL fail with `BufferError`.

#### Scenario: Block access while evicted

- **WHEN** a hierarchy is evicted and a caller reads, writes, scatters, or
  dispatches an operation
- **THEN** the call fails with `RuntimeError` requiring promotion

### Requirement: Eviction moves complete storage into the secondary tier

`evict()` SHALL return `None`. On a live promoted hierarchy, it SHALL move the
complete physical storage from primary to secondary through the registered
move for their exact classes, using lowered execution that creates no autograd
node.

If the current secondary is live and large enough, eviction SHALL use it as the
destination. Otherwise it SHALL allocate a fresh mutable secondary of the
hierarchy's size and dtype. After success, the hierarchy SHALL be evicted, the
secondary property SHALL identify the destination containing all values, and
the source primary storage SHALL be released by the move lifecycle. A replaced
secondary SHALL be released and relinquished.

Calling `evict()` while already evicted SHALL be an idempotent no-op returning
`None`.

#### Scenario: Evict into existing secondary storage

- **WHEN** the live secondary has at least the hierarchy size
- **THEN** eviction moves all values into that tier and marks the hierarchy
  evicted

#### Scenario: Provision an undersized secondary

- **WHEN** the current secondary is too small
- **THEN** eviction uses a fresh same-class secondary of sufficient size and
  makes it the owned secondary after success

### Requirement: Promotion restores fresh primary-class storage

`promote()` SHALL return `None`. On a live evicted hierarchy, it SHALL move the
complete physical storage from secondary to primary through the registered
reverse move, using lowered execution that creates no autograd node.

If the current primary remains live and large enough, promotion MAY use it as
the destination. Otherwise it SHALL allocate fresh primary storage of the
hierarchy's size, dtype, and internal transition mutability. After success, the
hierarchy SHALL be promoted, the primary property SHALL identify the complete
destination, and replaced primary storage SHALL be released and relinquished.

Calling `promote()` while already promoted SHALL be an idempotent no-op
returning `None`.

#### Scenario: Promote an evicted hierarchy

- **WHEN** a live hierarchy is evicted
- **THEN** promotion restores all values into storage of the original exact
  primary class and marks the hierarchy promoted

### Requirement: Residency transitions are failure-atomic

Each transition SHALL resolve the applicable move implementation when that
transition begins. If allocation, move dispatch, lowered execution, or result
validation fails, the hierarchy SHALL preserve its previous residency state,
current tier properties, logical values, ownership, size, dtype, mutability,
and visible version. Fresh temporary destinations SHALL be released and
relinquished.

A move result whose carrier is not the selected destination SHALL make the
transition fail with `RuntimeError` without committing the new state. A failed
transition SHALL remain retryable after its cause is corrected.

#### Scenario: Fail eviction without changing residency

- **WHEN** an eviction move raises after a temporary destination was allocated
- **THEN** the hierarchy remains promoted with its original tiers and values,
  the temporary destination is disposed, and a later eviction may retry

#### Scenario: Fail promotion without changing residency

- **WHEN** a promotion move raises
- **THEN** the hierarchy remains evicted with its secondary values intact and
  a later promotion may retry

### Requirement: Transitions preserve logical version identity

Eviction and promotion SHALL be storage-only transitions and SHALL not advance
the Evictable carrier's visible `version`. A successful public logical write
or scatter through the wrapper SHALL advance the wrapper version as defined by
`carrier-storage`. Moving to a new tier object SHALL not reset or substitute
the wrapper's version authority.

#### Scenario: Round-trip residency without mutation

- **WHEN** a hierarchy evicts and promotes with no public value write
- **THEN** its values and visible version equal those before the round trip

### Requirement: Evictable factories preserve the hierarchy pattern

The value and allocation factories defined by `carrier-storage` SHALL return
fresh promoted Evictable hierarchies using the receiver's exact primary and
secondary carrier kinds. The primary SHALL contain the materialized values or
requested allocation; the mutable secondary SHALL initially have size zero.
A dtype override SHALL have to be supported by both tier implementations.

#### Scenario: Allocate a fresh hierarchy

- **WHEN** `allocate_like` succeeds on an Evictable
- **THEN** the result is promoted with requested primary storage and a lazy
  zero-size secondary of the matching dtype

### Requirement: Evictable results retain composition ownership

An operation result or gradient restored through `carrier-dispatch` SHALL be
an Evictable that exclusively owns its new child storage. Retained aliases to
that result's child carriers SHALL therefore receive the same read-only
ownership protection as construction inputs. A layout-only result that reuses
the original primary SHALL reuse the original Evictable rather than attempting
to give the same child a second owner.

#### Scenario: Own an allocating result

- **WHEN** an Evictable operation allocates fresh primary result storage
- **THEN** the restored result hierarchy claims that storage and exposes no
  publicly mutable child alias

### Requirement: Release ends the whole hierarchy lifecycle

`release()` on a live unowned Evictable SHALL release both child tiers through
owner access, mark the hierarchy released, and return `None`. It SHALL be
idempotent. After release, `size()` SHALL return zero, `is_released()` SHALL be
`True`, and eviction, promotion, value access, mutation, scatter, and dispatch
SHALL fail with `RuntimeError`.

When an unreleased Evictable is destroyed without explicit release, it SHALL
relinquish ownership of each surviving child without releasing that child's
storage, allowing retained tier aliases to become publicly usable according to
their intrinsic mutability. A child already released by a residency transition
SHALL remain released.

#### Scenario: Release both tiers

- **WHEN** a caller releases a live hierarchy
- **THEN** both current tiers are released and no later residency transition
  succeeds

#### Scenario: Destroy without release

- **WHEN** an unreleased hierarchy becomes unreachable while retained live tier
  aliases remain
- **THEN** those surviving tiers are no longer owned and their storage was not
  released merely by destruction of the wrapper
