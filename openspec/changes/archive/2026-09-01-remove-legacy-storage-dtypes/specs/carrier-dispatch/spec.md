## MODIFIED Requirements

### Requirement: Planned execution passes the backend capability gate

Before a registered dtype-planned operation allocates output or performs backend
work, the implementation SHALL resolve the central `OperationPlan` and require
an exact matching backend capability as defined by `backend-capabilities`. The
implementation SHALL execute operand conversions, arithmetic, accumulation,
and output dtype from that accepted plan rather than deriving a local policy.

Generic, CPU, Metal, and a composite adapter for a registered operation SHALL
apply this preflight. An unsupported resolved plan SHALL raise
`UnsupportedOperationPlan` before result allocation, compilation, or kernel
entry. An operand dtype that central planning rejects SHALL propagate that
planning failure before capability lookup or backend work. Only an operation
name absent from the dtype-policy registry MAY retain a documented unplanned
path; no category-backed Generic operation receives an exception.

#### Scenario: Refuse a plan before backend work

- **WHEN** dispatch reaches a resolved plan the carrier does not advertise
- **THEN** execution raises `UnsupportedOperationPlan` before allocating,
  compiling, or entering the implementation

#### Scenario: Reject a category before backend work

- **WHEN** a registered Generic operation is presented with an abstract dtype
  category
- **THEN** central planning fails with `TypeError` before result allocation,
  capability execution, or kernel entry

### Requirement: Evictable lowering requires compatible promoted hierarchies

Evictable dispatch SHALL require a promoted primary and SHALL return a fresh
`EvictableOperation` owning the primary's fresh operation.
`EvictableOperation(primary_operation)` SHALL construct and return that
single-use adapter. `primary_operation` names the fresh operation dispatched by
the promoted primary and SHALL be an `Operation`; when `primary_operation` is
not an `Operation`, construction SHALL fail with `TypeError`. Forward SHALL be
single-use; a second forward call SHALL fail with `RuntimeError`.

Every Tensor operand lowered by an Evictable adapter SHALL be backed by an
Evictable, have one subtensor, and be promoted. All Evictable operands SHALL
have the same exact primary carrier class and the same exact secondary carrier
class. A missing Tensor input or wrong Tensor carrier SHALL fail with
`TypeError`; mismatched hierarchy classes SHALL fail with `TypeError`; an
evicted input SHALL fail with `RuntimeError` requiring promotion.

For every operation registered in the dtype policy, the adapter SHALL resolve
the plan from the outer operands and validated options and require it against
the outer hierarchy's snapshot before lowering. A planning failure, including
the `TypeError` for an abstract dtype category, SHALL propagate before nested
allocation, lowering, or execution. The adapter SHALL have no category-backed
planning or capability bypass.

The adapter SHALL wrap newly allocated primary results into a fresh hierarchy
of the same tier classes, leaving its secondary empty until first eviction. A
representation-preserving operation whose primary result reuses the same
primary carrier SHALL reuse the same Evictable carrier.

#### Scenario: Refuse mismatched hierarchies

- **WHEN** one Evictable operation receives operands with different exact
  primary or secondary carrier classes
- **THEN** forward fails with `TypeError` before nested execution

#### Scenario: Lower a supported outer plan

- **WHEN** compatible promoted hierarchies advertise the resolved outer plan
- **THEN** the adapter lowers to the primary, executes once, and restores an
  Evictable result using the same hierarchy kinds

#### Scenario: Reject an abstract category before lowering

- **WHEN** a registered operation receives an Evictable operand whose logical
  dtype is an abstract category
- **THEN** central planning fails with `TypeError` before nested allocation,
  lowering, or execution
