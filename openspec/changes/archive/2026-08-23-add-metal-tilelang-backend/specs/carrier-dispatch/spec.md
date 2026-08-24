## MODIFIED Requirements

### Requirement: Carrier is open while shipped implementations are closed

`Carrier` SHALL remain open for new sibling implementations, including Python
and native subclasses that implement its storage and dispatch contract.
`DependentCarrier` SHALL remain open for dependent implementations.

Generic, CPU, Metal, FileBacked, and Evictable SHALL be closed implementations
at runtime and SHALL be declared final on their public typed import paths. An
attempt to subclass any of those five SHALL fail with `TypeError` identifying
it as a closed carrier implementation and directing extension to a sibling
Carrier.

#### Scenario: Implement a sibling carrier

- **WHEN** a caller subclasses Carrier and supplies the required storage and
  dispatch behavior
- **THEN** instances participate in the ordinary dispatch contract

#### Scenario: Reject specialization of a shipped carrier

- **WHEN** a caller attempts to subclass Generic, CPU, Metal, FileBacked, or
  Evictable
- **THEN** class creation fails with the common closed-carrier `TypeError`

### Requirement: Generic and CPU dispatch the supported operation surface

Generic and CPU SHALL dispatch fresh implementations for every supported
computational name registered by `operation-dtype-policy`. Metal SHALL dispatch
fresh implementations for every such registered name for which it advertises
at least one executable plan. All three SHALL dispatch the shared
representation-preserving names `as_strided`, `broadcast_to`, `permute`,
`rearrange`, `reshape`, `squeeze`, `unsqueeze`, and `view`.

The planned dispatch names SHALL include `add`, `sub`, `mul`,
`elementwise_mul`, `div`, `pow`, `neg`, `abs`, `sign`, `recip`, `sqrt`,
`rsqrt`, `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `erf`, `floor`, `ceil`,
`round`, `maximum`, `minimum`, `rem`, `eq`, `ne`, `lt`, `le`, `logical_not`,
`relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softplus`, `elu`, `leaky_relu`,
`reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `argmax`, `argmin`,
`cumsum`, `matmul`, `conv_general`, `gather`, `scatter`, `scatter_add`,
`select` and `clamp`; the backend SHALL also provide the implementation needed
to return both observable values and indices for `sort` and `topk` without the
contract fixing private dispatch names.

An unknown name SHALL fail with `NotImplementedError`. FileBacked SHALL fail
with `NotImplementedError` for every computational dispatch name.

#### Scenario: Dispatch a supported Generic operation

- **WHEN** Generic dispatches a supported name twice
- **THEN** it returns two fresh Generic or shared operation implementations
  with Generic dispatch metadata

#### Scenario: Dispatch every registered Metal name

- **WHEN** Metal dispatches any registered computational name twice
- **THEN** it returns two fresh Metal operation implementations carrying Metal
  dispatch metadata

#### Scenario: Refuse FileBacked computation

- **WHEN** FileBacked receives a computational dispatch name
- **THEN** dispatch fails with `NotImplementedError`

### Requirement: Planned execution passes the backend capability gate

Before a planned simple-dtype operation allocates output or performs backend
work, the implementation SHALL resolve the central `OperationPlan` and require
an exact matching backend capability as defined by `backend-capabilities`.
The implementation SHALL execute operand conversions, arithmetic,
accumulation, and output dtype from that accepted plan rather than deriving a
local policy.

Generic, CPU, and Metal SHALL apply this preflight to their planned operations.
An unsupported plan SHALL raise `UnsupportedOperationPlan` before result
allocation, compilation, or kernel entry. An operation name absent from the
policy and a legacy opaque Generic operation MAY retain their documented
unplanned path.

#### Scenario: Refuse a plan before backend work

- **WHEN** dispatch reaches a resolved plan the carrier does not advertise
- **THEN** execution raises `UnsupportedOperationPlan` before allocating,
  compiling, or entering the implementation
