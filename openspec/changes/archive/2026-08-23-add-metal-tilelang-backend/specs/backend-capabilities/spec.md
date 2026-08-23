## MODIFIED Requirements

### Requirement: Shipped carrier declarations are complete and sealed

Generic, CPU, and Metal SHALL each expose exactly the planned shapes they
faithfully execute. Metal's declaration SHALL contain at least one executable
plan for every computational operation name registered by
`operation-dtype-policy`; it need not contain every plan exposed by Generic or
CPU. FileBacked SHALL expose a complete empty declaration. Those independent
classes SHALL be sealed before they are publicly observable and SHALL reject
public attempts to add capabilities.

Evictable SHALL have no class declaration because its reach depends on its
tiers. Its instance snapshot behavior is defined below.

#### Scenario: Cover the complete Metal dispatch-name set

- **WHEN** Metal's sealed capability declaration is compared with the central
  registered computational operation names
- **THEN** every registered name has at least one Metal capability and every
  Metal capability exactly matches a plan Metal executes

#### Scenario: Query a storage-only backend

- **WHEN** a FileBacked carrier's capabilities are enumerated
- **THEN** the result is the immutable empty tuple and no later registration
  can widen it
