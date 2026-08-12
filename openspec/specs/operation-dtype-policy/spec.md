---
title: Operation Dtype Policy
publish: true
status: stable
order: 50
summary: Backend-independent operation overloads, promotion, arithmetic, accumulation, and result dtype planning.
---

# operation-dtype-policy Specification

## Purpose

Define the central, backend-independent plan that determines how supported
simple-dtype operation operands are converted, computed, accumulated, and
stored.

## Terminology

| Term | Meaning |
| --- | --- |
| tensor operand | An operation operand represented to the planner by its storage `SimpleDType`. |
| weak scalar | A real Python scalar that selects a plan branch without carrying a dtype of its own. |
| operand plan | An immutable positional record fixing whether one operand is a tensor or weak scalar, its source storage dtype when it is a tensor, and the dtype into which it is materialized for computation. |
| operation plan | The immutable, backend-independent dtype-execution contract for one operation invocation, identifying the operation and fixing operand roles and conversions, compute arithmetic, accumulation behavior and accumulator dtype, and result dtype. |
| compute arithmetic | The numerical semantics applied to each computed term before any multi-term combination, including precision, rounding, overflow behavior, and permitted contraction, and distinct from accumulation. |
| accumulation | The numerical and ordering semantics for combining multiple computed terms, including combination precision, order or association, narrowing behavior, and extrema or argument tie behavior; `None` means that no terms are combined. |
| accumulator dtype | The concrete `SimpleDType` in which floating accumulation combines already encoded terms, independently of operand storage or conversion and result storage; it is an arithmetic choice and does not imply carrier storage support. |

## Requirements

### Requirement: One immutable plan carries the complete dtype decision

`OperandPlan(role, dtype, convert_to)` SHALL describe one positional operand.
`role` names whether the operand is a tensor or weak scalar and SHALL be an
`OperandRole`. `dtype` names the operand's source storage dtype; for a tensor
role, `dtype` SHALL be its source `SimpleDType`, and for a weak-scalar role,
`dtype` SHALL be `None`. `convert_to` names the materialization dtype and SHALL
be the `SimpleDType` into which that operand is materialized.

`OperationPlan(operation, operands, compute, accumulation,
accumulator_dtype, output)` SHALL describe one complete dtype decision.
`operation` names the dispatch operation and SHALL be a string. `operands`
names the ordered positional policy operands and SHALL be a tuple of
`OperandPlan` values. `compute` names the per-element arithmetic and SHALL be an
`Arithmetic`. `accumulation` names the term-combination rule and SHALL be an
`Accumulation` or `None` when no terms are combined. `accumulator_dtype` names
the concrete floating dtype used by floating accumulation and SHALL be a
`SimpleDType`, or `None` when the plan has no separate floating accumulator.
`output` names the result dtype and SHALL be a `SimpleDType`. These records and
the public enum values they contain SHALL be immutable and hashable.

An operation plan SHALL carry no duplicated autograd field. Autograd
eligibility SHALL follow from the result dtype under the framework-wide rule
that only `DType.Floating` and `DType.Float32` participate.

#### Scenario: Inspect a mixed add plan

- **WHEN** `resolve_operation_plan("add", DType.Int32, DType.Float32)` succeeds
- **THEN** it returns an immutable plan that converts both operands to
  Float32, uses binary32 arithmetic, performs no accumulation, and outputs
  Float32

### Requirement: Arithmetic and accumulation values have exact semantics

`Arithmetic.BINARY32` SHALL mean IEEE-754 binary32, round-to-nearest-even,
without wider intermediates, reassociation, or fused contraction.
`INT32_EXACT_CHECKED` SHALL mean exact integer arithmetic with an
`OverflowError` when the exact result leaves Int32 range.
`INT32_EXACT` SHALL mean exact integer arithmetic for a plan whose result is
known to remain in range.

`Accumulation.FLOATING` SHALL combine terms in the plan's separate floating
`accumulator_dtype`, with association order chosen by the backend.
`SEQUENTIAL_BINARY32` and `SEQUENTIAL_BINARY32_PRODUCT` SHALL combine in
ascending logical order, rounding in binary32 after every addition or
multiplication. `MAXIMUM` and `MINIMUM` SHALL apply the reference Float32 NaN
and signed-zero rules in logical order. `ARGMAX` and `ARGMIN` SHALL additionally
retain the first winning logical ordinal. `EXACT_INTEGER` SHALL combine exactly
over integers and check only the final narrowing to Int32.

#### Scenario: Distinguish exact reduction accumulation

- **WHEN** an Int32 sum has an intermediate partial sum outside Int32 range but
  a final result inside range
- **THEN** exact-integer accumulation succeeds because only the final narrowing
  is checked

### Requirement: Operation specifications declare positional overloads

`OperationOverload(roles, rule, tensor_domains=None)` SHALL describe one
positional `OperandRole` signature and its plan resolver. `roles` names the
ordered operand-role signature and SHALL contain `OperandRole` values. `rule`
names the deterministic resolver and SHALL be a callable that produces the
matching `OperationPlan`.
`tensor_domains` names the optional ordered dtype domain for each tensor role;
it SHALL be optional and SHALL default to `None`, meaning the caller supplies
the enumeration domain. When supplied, `tensor_domains` SHALL contain one
non-empty dtype domain per tensor role. When `tensor_domains` has the wrong
count or contains an empty domain, construction SHALL fail with `ValueError`.

`OperationSpec(name, overloads, public=True,
dtype_operand_positions=None)` SHALL describe one registered operation.
`name` names the operation's unique dispatch registry entry and SHALL be a
string. `overloads` names every accepted positional overload and SHALL be a
non-empty tuple with no duplicate role signature. `public` states whether the
dispatch name itself is a public single-result operation; it SHALL be a boolean,
SHALL be optional, and SHALL default to `True`.
`dtype_operand_positions` names the forward-argument positions that participate
in dtype policy; it SHALL be optional and SHALL default to `None`, meaning
every forward argument is a dtype-policy operand. When supplied,
`dtype_operand_positions` SHALL be a unique increasing sequence of non-negative
positions with the same length as every overload. When `overloads` is empty or
contains a duplicate role signature, or when `dtype_operand_positions`
violates its ordering, range, uniqueness, or length constraints, construction
SHALL fail with `ValueError`.

For a single-overload specification, `roles` and `rule` SHALL return that
overload's values. For a multiple-overload specification, either compatibility
property SHALL fail with `AttributeError` so it cannot hide the additional
overloads.

#### Scenario: Keep non-dtype parameters out of planning

- **WHEN** a registered operation identifies only selected forward arguments
  in `dtype_operand_positions`
- **THEN** axes, shapes, strides, padding, ordering flags, and other remaining
  arguments do not become weak-scalar operands

### Requirement: The registry is deterministic and exhaustively enumerable

`registered_operations()` SHALL return every registered `OperationSpec` in
operation-name order. Each operation name SHALL occur once. Representation-only
operations that perform no dtype computation SHALL not be registered as dtype
plans.

`resolvable_plans(tensor_dtypes=SUPPORTED_TENSOR_DTYPES,
weak_scalars=WEAK_SCALAR_PROBES)` SHALL enumerate every distinct plan selected
from every registered overload over the supplied tensor dtypes and weak scalar
probes. `tensor_dtypes` names the ordered dtype domain offered to each tensor
operand; it SHALL be optional and SHALL default to
`SUPPORTED_TENSOR_DTYPES`. `weak_scalars` names the ordered scalar probes
offered to weak-scalar operands; it SHALL be optional and SHALL default to
`WEAK_SCALAR_PROBES`. The result SHALL be deduplicated and deterministically
ordered by registry and operand enumeration order, and SHALL include every
supported configurable accumulator variant.

The default tensor domain SHALL be `(DType.Float32, DType.Int32)`. The supported
floating accumulator domain SHALL be `(DType.Float32, DType.Float64)`, and the
configurable operations SHALL be exactly `reduce_sum` and `matmul`.

#### Scenario: Enumerate policy coverage

- **WHEN** `resolvable_plans()` is called with its defaults
- **THEN** it includes all registered overload shapes and both floating
  accumulator choices for Float32 reduce_sum and matmul

### Requirement: Resolution selects exact roles before applying policy

`resolve_operation_plan(operation, *operands, accumulator_dtype=None,
options=None)` SHALL select the registered overload whose positional roles
match the operands, invoke its resolver, validate that the returned plan agrees
with the registration, and return that immutable plan. `operation` names the
dispatch operation. `operands` names the ordered positional values used for
dtype planning; each tensor operand SHALL be supplied as its storage dtype and
each weak scalar SHALL be supplied as its Python value.

`accumulator_dtype` names a directly requested floating accumulator dtype; it
SHALL be optional and SHALL default to `None`. `options` names prevalidated
execution options; it SHALL be optional and SHALL default to `None`. When both
`accumulator_dtype` and `options` are non-`None`, resolution SHALL fail with
`TypeError`. When `options` names a different operation, resolution SHALL fail
with `ValueError`.

When `operation` is unregistered, resolution SHALL fail with
`NotImplementedError`. When `operands` has the wrong count or an unmatched role
signature, contains a non-`DType` tensor operand, or contains a non-real weak
scalar, resolution SHALL fail with `TypeError`. Resolution SHALL depend only on
`operation`, operand dtype identities, weak-scalar values needed to select a
branch, and `options`; it SHALL read no tensor element.

#### Scenario: Select tensor and scalar overloads separately

- **WHEN** `mul` or `pow` receives a tensor dtype followed by another tensor
  dtype or a weak scalar
- **THEN** the planner selects the corresponding exact positional overload
  without treating the two signatures as interchangeable

#### Scenario: Reject an unknown operation

- **WHEN** `resolve_operation_plan` receives a name absent from the registry
- **THEN** it fails with `NotImplementedError`

### Requirement: Unsupported dtype dispositions are explicit

Simple-dtype planning SHALL accept the implemented tensor dtypes Float32 and
Int32, plus Bool only in the exact operand positions whose overload domain
requires Bool. The legacy opaque categories `Any` and `Floating` SHALL fail
with `TypeError` and remain on Generic's separate legacy execution path. An
abstract non-opaque category SHALL fail with `TypeError`. A compound dtype SHALL
fail with `NotImplementedError` identifying deferred representation-aware
planning. A registered but unimplemented simple dtype SHALL fail with
`NotImplementedError` rather than widening to an implemented dtype.

Dtype matching SHALL use descriptor identity. An object that merely compares
equal to a supported descriptor SHALL not select its plan.

#### Scenario: Refuse an unimplemented simple encoding

- **WHEN** a tensor operand is represented by a registered narrow simple dtype
  with no operation implementation
- **THEN** resolution fails with `NotImplementedError` and does not substitute
  Float32 or Int32

#### Scenario: Keep legacy opaque arithmetic separate

- **WHEN** a Generic operation contains an Any or Floating storage operand
- **THEN** it uses the documented legacy path rather than a simple-dtype plan

### Requirement: Weak scalars select branches without acquiring a dtype

A weak scalar SHALL be a real Python number. Integral values other than `bool`
SHALL select the integer kind; real non-integral values and `bool` SHALL select
the floating kind. Its `OperandPlan.dtype` SHALL be `None`, and `convert_to`
SHALL state the materialization dtype selected by the operation.

When an integer plan needs to represent the weak scalar as Int32, a value
outside `[-2**31, 2**31 - 1]` SHALL fail with `OverflowError`. The same value
MAY select a floating plan without that Int32 range failure.

#### Scenario: Treat bool as a weak floating scalar

- **WHEN** an Int32 tensor is multiplied by a Python `bool`
- **THEN** the bool selects the floating branch and the plan outputs Float32

#### Scenario: Reject an out-of-range integer scalar on an integer path

- **WHEN** a weak integer outside Int32 range would be materialized as Int32
- **THEN** resolution fails with `OverflowError`

### Requirement: Core promotion families resolve consistently

For tensor/tensor `add`, `sub`, `mul`, and `elementwise_mul`, two Int32 operands
SHALL remain Int32 and use checked exact arithmetic; any Float32 operand SHALL
convert both operands to Float32 and produce Float32 binary32 output. Division
SHALL convert both operands to Float32 and output Float32 for every Float32 or
Int32 pair.

Tensor/tensor `pow` SHALL convert every pair to Float32 and output Float32.
For tensor/weak-scalar `pow`, an Int32 base SHALL remain Int32 only for a weak
integer exponent in `[0, 2**31 - 1]`; a negative integer, larger integer,
floating scalar, or bool SHALL select Float32. Reverse weak-scalar/tensor pow
SHALL output Float32. Int32 tensor/weak-integer `mul` SHALL remain Int32;
floating or bool scalars SHALL select Float32.

`relu` SHALL preserve Float32 or Int32. The other floating activations SHALL
convert Int32 to Float32 and output Float32. The Float32-only unary and binary
operations SHALL require Float32 and output Float32. Float32 comparisons and
`logical_not` SHALL output Bool. Bool SHALL not participate in numeric
promotion.

#### Scenario: Preserve an integer exponent exactly

- **WHEN** an Int32 base is raised to a weak integer exponent at most
  `2**31 - 1`, including one above binary32's exact-integer range
- **THEN** the plan retains exact integer arithmetic and uses the original
  exponent without floating conversion

#### Scenario: Plan integer division as floating

- **WHEN** both operands of `div` are Int32
- **THEN** both convert to Float32 and the result is Float32

### Requirement: Structured operation families pin their dtype shapes

Float32 `reduce_sum` SHALL use floating accumulation and output Float32; Int32
`reduce_sum` SHALL use exact-integer accumulation and output Int32. Float32
`matmul` and mixed Float32/Int32 matmul SHALL convert terms to Float32, use
floating accumulation, and output Float32; Int32/Int32 matmul SHALL use exact
integer accumulation and output Int32.

Float32 `reduce_prod`, extrema reductions, arg reductions, and `cumsum` SHALL
use their corresponding ordered accumulation. Arg reductions SHALL output
Int32; the others SHALL output Float32. `conv_general` and `scatter_add` SHALL
use sequential binary32 accumulation. `gather`, `scatter`, Float32
`conv_general`, internal sort/topk value plans, and `select` SHALL output
Float32; internal sort/topk index plans and arg reductions SHALL output Int32.
`select` SHALL require Bool, Float32, Float32 tensor operands. `clamp` SHALL
require a Float32 value and each bound SHALL be either a Float32 tensor or weak
real scalar; all four bound-role combinations SHALL output Float32.

#### Scenario: Plan a masked selection

- **WHEN** select receives `(DType.Bool, DType.Float32, DType.Float32)`
- **THEN** it returns a Float32 binary32 plan with no accumulation

#### Scenario: Refuse a wrong structured operand dtype

- **WHEN** select, clamp, gather, scatter, or another restricted operation
  receives a dtype outside its declared operand domain
- **THEN** resolution fails rather than promoting it into the required shape

### Requirement: Accumulator options widen only accumulation

`OperationExecutionOptions(operation, accumulator_dtype=None)` and
`operation_execution_options(operation, *, accumulator_dtype=None)` SHALL
return immutable validated options for `reduce_sum` or `matmul`.
`operation` names the operation to which the execution options apply and SHALL
be `reduce_sum` or `matmul`. `accumulator_dtype` names the requested floating
accumulator dtype; it SHALL be optional and SHALL default to `None`, which
selects the policy default. Only `DType.Float32` and `DType.Float64` SHALL be
supported floating accumulators.

A non-configurable operation SHALL fail with `TypeError`. A non-simple
accumulator SHALL fail with `TypeError`; an unimplemented simple accumulator
SHALL fail with `NotImplementedError`. An exact-integer plan SHALL fail with
`TypeError` when asked to use a floating accumulator. Changing a floating
accumulator SHALL leave operand conversions, compute arithmetic, and output
dtype unchanged.

`plan_accumulator_variants(plan)` SHALL return `plan` first and every remaining
supported accumulator variant for a configurable floating plan; otherwise it
SHALL return `(plan,)`. `plan` names the already-resolved base `OperationPlan`
whose accumulator variants are requested.

#### Scenario: Widen a Float32 reduction accumulator

- **WHEN** reduce_sum options request `DType.Float64`
- **THEN** the plan accumulates already encoded Float32 terms in Float64 while
  retaining Float32 input conversion, compute, and output storage

### Requirement: Generic is the concrete semantic reference

For concrete Float32, Int32, and Bool plans, Generic SHALL execute the resolved
plan as the behavioral reference. Float32 storage and each arithmetic step
SHALL be binary32; IEEE singularities and overflow to infinity SHALL produce
IEEE values rather than Python arithmetic exceptions. Int32 arithmetic SHALL
be exact and checked at the narrowing point stated by the plan, raising
`OverflowError` rather than wrapping. Bool results SHALL contain only Python
booleans and SHALL be non-differentiable.

CPU SHALL resolve and execute the same plans and SHALL agree with Generic on
concrete results. Neither backend SHALL maintain a separate promotion or result
dtype policy. Generic and CPU SHALL both execute Float32 and Float64 accumulator
variants for `reduce_sum` and `matmul` while retaining Float32 result storage.

#### Scenario: Compare reference and CPU results

- **WHEN** Generic and CPU execute the same supported concrete plan on the same
  logical values
- **THEN** their result dtype and numerical behavior agree with that plan
