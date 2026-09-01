## MODIFIED Requirements

### Requirement: One immutable plan carries the complete dtype decision

`OperandPlan(role, dtype, convert_to)` SHALL describe one positional operand.
`role` names whether the operand is a tensor or weak scalar and SHALL be an
`OperandRole`. `dtype` names the operand's source storage dtype; for a tensor
role, `dtype` SHALL be its source `SimpleDType`, and for a weak-scalar role,
`dtype` SHALL be `None`. `convert_to` names the simple dtype used for
computation and SHALL be a `SimpleDType`.

`OperationPlan(operation, operands, compute, accumulation,
accumulator_dtype, output)` SHALL describe the complete dtype decision for one
operation. `operation` names the registered operation and SHALL be a string;
`operands` names the ordered positional policy operands and SHALL be a tuple of
`OperandPlan` values. `compute` names the per-element arithmetic and SHALL be an
`Arithmetic`. `accumulation` names the term-combination rule and SHALL be an
`Accumulation` or `None` when no terms are combined. `accumulator_dtype` names
the concrete floating dtype used by floating accumulation and SHALL be a
`SimpleDType`, or `None` when the plan has no separate floating accumulator.
`output` names the result dtype and SHALL be a `SimpleDType`. These records and
the public enum values they contain SHALL be immutable and hashable.

An operation plan SHALL carry no duplicated autograd field. Autograd
eligibility SHALL follow from the result dtype under the framework-wide rule
that only exact `DType.Float32` participates.

#### Scenario: Inspect a mixed add plan

- **WHEN** `resolve_operation_plan("add", DType.Int32, DType.Float32)` succeeds
- **THEN** it returns an immutable plan that converts both operands to
  Float32, uses binary32 arithmetic, performs no accumulation, and outputs
  Float32

## ADDED Requirements

### Requirement: Central planning rejects abstract and unsupported dtypes explicitly

Simple-dtype planning SHALL accept the implemented tensor dtypes Float32 and
Int32, plus Bool only in the exact operand positions whose overload domain
requires Bool. Every `DTypeCategory`, including `DType.Any`,
`DType.Floating`, `DType.Integer`, and extension categories, SHALL fail with
`TypeError` identifying that an abstract category is not an operation storage
dtype. No category SHALL enter an alternate Generic execution path. A compound
dtype SHALL fail with `NotImplementedError` identifying deferred
representation-aware planning. A registered but unimplemented simple dtype
SHALL fail with `NotImplementedError` rather than widening to an implemented
dtype.

Dtype matching SHALL use descriptor identity. An object that merely compares
equal to a supported descriptor SHALL not select its plan. Every failure SHALL
occur before a backend allocates result storage or performs computation.

#### Scenario: Refuse an abstract category uniformly

- **WHEN** any registered operation receives `DType.Any`, `DType.Floating`,
  `DType.Integer`, or an extension category in a tensor-operand position
- **THEN** resolution fails with `TypeError` identifying the abstract category
  before backend allocation or computation

#### Scenario: Reject Generic category operands through central planning

- **WHEN** a Generic operation is presented with `DType.Any` or
  `DType.Floating` in a tensor-operand position
- **THEN** central resolution fails with `TypeError` before Generic allocation
  or execution, and no separate opaque arithmetic path runs

#### Scenario: Refuse an unimplemented simple encoding

- **WHEN** a tensor operand is represented by a registered narrow simple dtype
  with no operation implementation
- **THEN** resolution fails with `NotImplementedError` and does not substitute
  Float32 or Int32

#### Scenario: Refuse compound planning separately

- **WHEN** a tensor operand is represented by a `CompoundDType`
- **THEN** resolution fails with `NotImplementedError` identifying deferred
  representation-aware planning

## REMOVED Requirements

### Requirement: Unsupported dtype dispositions are explicit

**Reason**: The requirement's legacy Generic exception and its scenario name
contradict the uniform central rejection contract for abstract categories.

**Migration**: Use `Central planning rejects abstract and unsupported dtypes
explicitly`, which defines one category failure path and retains the distinct
failures for unimplemented simple and compound dtypes.
