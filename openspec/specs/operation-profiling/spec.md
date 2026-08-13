---
title: Operation Profiling
publish: true
status: stable
order: 90
summary: Carrier-aware profiling lifecycle, execution evidence, nesting, timing, aggregation, and reporting.
---

# operation-profiling Specification

## Purpose

Define carrier-aware profiling of synchronous dispatched operation execution,
including lifecycle, immutable event evidence, nested timing, aggregation,
thread isolation, and deterministic reporting.

## Terminology

| Term | Meaning |
| --- | --- |
| profiled computation boundary | The synchronous computation-and-result-validation interval of a dispatched Operation, entered only after public or lowered execution has accepted its execution options and structural inputs, and exited after carrier computation either returns a valid Tensor or raises. |
| selected event | An immutable record created when a profiled computation boundary belongs to an exact dispatching Carrier class selected by the active Profiler. |
| inclusive time | Monotonic synchronous host wall time from entry to exit of one profiled computation boundary, including every nested dispatched computation interval. |
| self time | One boundary's inclusive time less the inclusive intervals of its immediately nested dispatched computations, whether or not those child events are selected; a deeper descendant is excluded exactly once through its enclosing immediate child interval. |
| input-shape snapshot | An immutable positional tuple containing each Tensor input's complete hierarchical Shape and `None` for each non-Tensor positional input. |
| aggregate key | A dispatch name and exact dispatching Carrier class, plus an input-shape snapshot when shape grouping is enabled. |

## Requirements

### Requirement: Public profiling names share both supported import paths

`profile`, `Profiler`, `ProfilerEvent`, and `ProfilerAggregate` SHALL each be
available from both `strideweave` and `strideweave.operation`. For each name,
the two import paths SHALL return the same object. `profile(...)` and
`Profiler(...)` SHALL return a `Profiler`; `ProfilerEvent(...)` SHALL return a
`ProfilerEvent`; and `ProfilerAggregate(...)` SHALL return a
`ProfilerAggregate`.

#### Scenario: Import the profiling surface from either namespace

- **WHEN** a caller imports all four profiling names from `strideweave` and
  `strideweave.operation`
- **THEN** each pair has identical object identity

### Requirement: Profilers select exact carrier classes

`profile(*, carriers=None, record_shapes=False)` and
`Profiler(*, carriers=None, record_shapes=False)` SHALL create a single-use
`Profiler`. `carriers` names the exact dispatching Carrier classes whose events
are selected. It SHALL be optional and SHALL default to `None`, meaning every
exact Carrier class is selected. An empty iterable SHALL select no Carrier
class. Selection SHALL use exact class identity rather than subclass or
instance relationships.

Every non-`None` `carriers` value SHALL be an iterable containing only Carrier
classes. A non-iterable, a Carrier instance, or an iterable containing a value
that is not a Carrier class SHALL fail with `TypeError`. `record_shapes` names
whether positional Tensor input shapes are captured; it SHALL be optional,
SHALL default to `False`, and a non-`bool` value SHALL fail with `TypeError`.

#### Scenario: Record all carrier classes by default

- **WHEN** a profiler created with default inputs observes a composite operation
  executed by an outer Evictable and an inner Generic
- **THEN** it records one event for each exact dispatching Carrier class

#### Scenario: Select one exact carrier class

- **WHEN** `carriers` contains Generic and an operation is dispatched by a
  distinct Carrier class that owns Generic storage
- **THEN** the distinct Carrier's execution is not selected as a Generic event

#### Scenario: Select no carrier classes

- **WHEN** `carriers` is an empty iterable
- **THEN** the completed profiler returns an empty event tuple

#### Scenario: Reject an invalid carrier filter

- **WHEN** `carriers` is not iterable or contains a Carrier instance or a class
  that is not a Carrier subclass
- **THEN** profiler creation fails with `TypeError`

#### Scenario: Reject an invalid shape option

- **WHEN** `record_shapes` is not a `bool`
- **THEN** profiler creation fails with `TypeError`

### Requirement: Profiler contexts are single-use and publish results on exit

Entering a Profiler SHALL activate it on the current thread and return that
same Profiler. At most one Profiler SHALL be active on a thread. Entering
another Profiler while one is active SHALL fail with `RuntimeError`. Re-entering
a Profiler whose entry has previously been attempted SHALL fail with
`RuntimeError` identifying the single-use contract. An entry attempt SHALL
consume the Profiler even when activation fails because another Profiler is
already active on that thread.

`events()`, `key_averages(...)`, and `table(...)` SHALL make results available
only after successful context exit; calling one before exit SHALL fail with
`RuntimeError`. Context exit SHALL deactivate the profiler, freeze its recorded
events, release its active recording resources, and return `False` so an
exception from the context body continues to propagate. Exiting a Profiler that
is not active SHALL fail with `RuntimeError`.

#### Scenario: Read results after normal exit

- **WHEN** a profiler context records an operation and exits normally
- **THEN** `events()` returns the frozen event tuple

#### Scenario: Refuse results before exit

- **WHEN** `events()`, `key_averages(...)`, or `table(...)` is called before the
  profiler context exits
- **THEN** the call fails with `RuntimeError` identifying result availability

#### Scenario: Reject a reused context

- **WHEN** a Profiler that has already been entered is entered again
- **THEN** entry fails with `RuntimeError` identifying the single-use contract

#### Scenario: Reject nested contexts on one thread

- **WHEN** a second Profiler is entered while another Profiler is active on the
  same thread
- **THEN** entry fails with `RuntimeError` identifying the active session
- **AND** a later entry attempt on that rejected Profiler fails with
  `RuntimeError` identifying the single-use contract

#### Scenario: Preserve results when the context body raises

- **WHEN** a selected dispatched execution raises inside a profiler context
- **AND** the failure occurs after its profiled computation boundary begins
- **THEN** context exit preserves its failed event and releases active recording
  resources
- **AND** the original exception continues to propagate

### Requirement: Events represent post-preflight dispatched computation attempts

A selected event SHALL represent one profiled computation boundary reached by a
dispatched Operation. Public and lowered execution SHALL enter that boundary
only after execution-option validation and structural Tensor preflight have
succeeded. The boundary SHALL include carrier computation and validation that
the computation returned a Tensor, and SHALL end immediately after that result
validation.

Dispatch factory lookup SHALL leave the active Profiler's event tuple
unchanged. A computation executed by an Operation without a dispatch name and
exact dispatching Carrier metadata SHALL leave it unchanged. The public
`move(tensor, destination)` operation SHALL leave it unchanged because its
execution does not reach a metadata-bearing dispatched computation boundary.

Both successful and failed attempts that enter a selected profiled computation
boundary SHALL be recorded. A validation or preflight failure before entry
SHALL leave the event tuple unchanged. After a failure following entry, the
Profiler SHALL restore its nesting state and record later independent attempts
in the same active context normally.

#### Scenario: Record a dispatched attempt

- **WHEN** a Generic or CPU Carrier dispatches and executes a supported
  operation in an active profiler context
- **THEN** the profiler records one event identifying that dispatched attempt

#### Scenario: Exclude dispatch lookup

- **WHEN** a Carrier returns a dispatched Operation but the Operation is not
  executed
- **THEN** no profiler event is recorded for that lookup

#### Scenario: Exclude a preflight failure

- **WHEN** a dispatched operation rejects execution options or structural
  Tensor inputs before entering its profiled computation boundary
- **THEN** the active Profiler's event tuple remains unchanged

#### Scenario: Exclude execution without dispatch metadata

- **WHEN** a directly constructed Operation executes without dispatch metadata
- **THEN** no profiler event is recorded for that execution

#### Scenario: Exclude public movement

- **WHEN** `move(tensor, destination)` completes inside an active Profiler
- **THEN** the active Profiler's event tuple remains unchanged

#### Scenario: Continue after an attempted execution fails

- **WHEN** one selected operation raises after entering its profiled computation
  boundary and a later selected operation succeeds in the same context
- **THEN** both events are retained in their execution-start order with their
  respective failure and success states

### Requirement: Raw events are immutable computation snapshots

`events()` SHALL return a tuple of immutable `ProfilerEvent` values in
execution-start order. Every event SHALL expose:

- `id`, a zero-based session-local identifier assigned in selected
  execution-start order;
- `parent_id`, the identifier of the nearest selected execution that was active
  when this execution started, or `None`;
- `name`, the dispatch name;
- `carrier_type`, the exact dispatching Carrier class;
- `implementation_type`, the exact executed Operation class;
- `input_shapes`, the optional positional shape snapshot;
- `start_time_ns`, a monotonic host start timestamp in nanoseconds;
- `duration_ns`, inclusive synchronous host wall time in nanoseconds;
- `self_time_ns`, synchronous host wall time excluding nested dispatched
  execution;
- `thread_id`, the Python thread identifier that executed the operation; and
- `succeeded`, which is `True` exactly when the profiled computation boundary
  returned a valid Tensor without raising.

#### Scenario: Inspect a successful event

- **WHEN** one selected operation returns a valid Tensor
- **THEN** `events()` returns an immutable event with id zero, the dispatch and
  implementation classes, non-negative timing fields, the executing thread,
  and `succeeded` equal to `True`

#### Scenario: Inspect a failed event

- **WHEN** one selected operation raises during execution
- **AND** its profiled computation boundary had begun
- **THEN** its immutable event has `succeeded` equal to `False` and retains its
  dispatch, nesting, thread, and timing evidence

### Requirement: Event values support direct immutable construction

`ProfilerEvent(id, parent_id, name, carrier_type, implementation_type,
input_shapes, start_time_ns, duration_ns, self_time_ns, thread_id, succeeded)`
SHALL construct an immutable `ProfilerEvent` storing the supplied values without
normalization or cross-field validation. All inputs SHALL be required and SHALL
be accepted positionally or by keyword. `id` names the session-local integer
identifier; `parent_id` names an integer parent identifier or `None`; `name`
names the dispatch string; `carrier_type` names the exact Carrier class;
`implementation_type` names the exact Operation implementation class;
`input_shapes` names an input-shape snapshot or `None`; `start_time_ns`,
`duration_ns`, and `self_time_ns` name integer timing values; `thread_id` names
the integer Python thread identifier; and `succeeded` names the Boolean result
state.

Omitting a required input, supplying too many positional inputs, binding an
input more than once, or supplying an unknown keyword SHALL fail with
`TypeError`.

#### Scenario: Construct an event value directly

- **WHEN** a caller supplies all eleven event inputs
- **THEN** `ProfilerEvent(...)` returns an immutable value exposing those exact
  inputs

#### Scenario: Reject an incomplete event construction

- **WHEN** a caller omits a required event input
- **THEN** construction fails with `TypeError`

### Requirement: Shape recording preserves positional hierarchy

When `record_shapes` is `False`, every event's `input_shapes` SHALL be `None`.
When `record_shapes` is `True`, `input_shapes` SHALL be an immutable tuple
aligned with the Operation's positional inputs. A Tensor position SHALL contain
an immutable snapshot of its complete hierarchical Shape using nested tuples of
integer extents. A non-Tensor position SHALL contain `None`. The snapshot SHALL
contain exactly one position per positional operation input; validated
execution options SHALL remain separate from it.

#### Scenario: Snapshot a hierarchical Tensor and a scalar

- **WHEN** a shape-recording profiler observes an operation whose positional
  inputs are a hierarchically shaped Tensor and a non-Tensor scalar
- **THEN** `input_shapes` contains the nested extent tuple followed by `None`

#### Scenario: Omit shape recording

- **WHEN** `record_shapes` is `False`
- **THEN** the event's complete `input_shapes` field is `None`

#### Scenario: Keep execution options outside shape positions

- **WHEN** an operation receives validated typed execution options in addition
  to its positional Tensor input
- **THEN** `input_shapes` contains only the Tensor input position

### Requirement: Nested events preserve selected ancestry and self time

Every dispatched execution active inside another dispatched execution SHALL be
treated as nested work, whether or not either execution is selected by the
carrier filter. When both are selected, the nested event's `parent_id` SHALL
name the nearest selected parent. When the parent is filtered out, a selected
nested event SHALL name the nearest remaining selected ancestor, or `None` when
there is none.

`duration_ns` SHALL equal the inclusive time of the profiled computation
boundary. `self_time_ns` SHALL be non-negative and SHALL equal that inclusive
time less the inclusive intervals of the boundary's immediately nested
dispatched computations. A deeper descendant SHALL be excluded exactly once as
part of its enclosing immediate child's interval. Time spent in a filtered-out
immediate child SHALL still be excluded from a selected parent's self time.

#### Scenario: Record both composite levels

- **WHEN** an Evictable adapter executes a nested Generic operation and both
  exact Carrier classes are selected
- **THEN** the outer event precedes the inner event
- **AND** the inner event's `parent_id` names the outer event
- **AND** the outer self time excludes the inner duration

#### Scenario: Hide nested work without charging it to the parent

- **WHEN** only the outer Carrier class is selected for a composite execution
- **THEN** only the outer event is returned
- **AND** its self time excludes the filtered nested execution interval

#### Scenario: Select only the nested execution

- **WHEN** only the inner Carrier class is selected for a composite execution
- **THEN** the inner event is returned as a root event
- **AND** its self time equals its inclusive duration when it has no nested
  dispatched work of its own

#### Scenario: Exclude three-level nesting exactly once

- **WHEN** a selected outer computation contains a child computation that
  contains a grandchild computation
- **THEN** the outer self time equals its inclusive time less the child
  inclusive time
- **AND** the grandchild interval is excluded from the outer self time exactly
  once through the child's inclusive interval

### Requirement: Profiling is local to the active thread

An active Profiler SHALL record only dispatched execution attempts made on the
thread that entered it. Work on another thread SHALL require its own Profiler.
Context exit SHALL occur on the entering thread. An exit attempted from another
thread SHALL fail with `RuntimeError`, abandon that session, and make it
inactive for further recording.

Before the owner thread begins another dispatched execution or enters another
Profiler, it SHALL recover from the abandoned session, discard the abandoned
session's events, and permit a fresh context to run. Destruction of an abandoned
Profiler SHALL permit normal process continuation and owner-thread recovery.

#### Scenario: Ignore another thread's operation

- **WHEN** a worker thread executes a dispatched operation while a Profiler is
  active only on the main thread
- **THEN** the main-thread Profiler records no event for the worker execution

#### Scenario: Reject cross-thread exit and recover

- **WHEN** another thread attempts to exit the active Profiler
- **THEN** exit fails with `RuntimeError` identifying the active-thread rule
- **AND** the owner thread can subsequently execute unprofiled work and enter a
  fresh Profiler without process failure

### Requirement: Aggregates are deterministic functions of raw events

`key_averages(*, group_by_input_shape=False)` SHALL return a tuple of immutable
`ProfilerAggregate` rows. `group_by_input_shape` names whether the event's
`input_shapes` participates in grouping; it SHALL be optional, SHALL default to
`False`, and a non-`bool` value SHALL fail with `TypeError`.

Rows SHALL always group by dispatch name and exact dispatching Carrier class.
When `group_by_input_shape` is `True`, they SHALL additionally group by the
captured `input_shapes`; when shape recording was disabled those keys remain
`None`. Returned rows SHALL use a deterministic ascending order by dispatch
name, Carrier module and qualified name, and shape representation.

Every row SHALL expose its grouping `name`, `carrier_type`, and `input_shapes`,
plus `count`, the number of raw events; `total_time_ns`, the sum of inclusive
durations; `self_total_time_ns`, the sum of self times; `mean_time_ns`, the
inclusive total divided by count; and `min_time_ns` and `max_time_ns`, the
minimum and maximum inclusive durations. Both successful and failed raw events
SHALL contribute according to the same rule.

#### Scenario: Aggregate repeated executions

- **WHEN** two events share a dispatch name and exact Carrier class and shape
  grouping is disabled
- **THEN** one immutable row reports count two and timing fields derived exactly
  from those two raw events

#### Scenario: Separate exact carrier classes

- **WHEN** nested events share a dispatch name but have different exact
  dispatching Carrier classes
- **THEN** `key_averages()` returns a separate row for each class

#### Scenario: Group by hierarchical input shape

- **WHEN** shape recording captured two different hierarchical input-shape
  tuples and `group_by_input_shape` is `True`
- **THEN** the events appear in separate aggregate rows carrying those shape
  keys

#### Scenario: Reject an invalid grouping option

- **WHEN** `group_by_input_shape` is not a `bool`
- **THEN** `key_averages(...)` fails with `TypeError`

### Requirement: Aggregate values support direct immutable construction

`ProfilerAggregate(name, carrier_type, input_shapes, count, total_time_ns,
self_total_time_ns, mean_time_ns, min_time_ns, max_time_ns)` SHALL construct an
immutable `ProfilerAggregate` storing the supplied values without normalization
or cross-field validation. All inputs SHALL be required and SHALL be accepted
positionally or by keyword. `name` names the dispatch string; `carrier_type`
names the exact dispatching Carrier class; `input_shapes` names the grouped
input-shape snapshot or `None`; `count` names the integer event count;
`total_time_ns` and `self_total_time_ns` name integer inclusive and self totals;
`mean_time_ns` names the floating-point inclusive mean; and `min_time_ns` and
`max_time_ns` name the integer inclusive extrema.

Omitting a required input, supplying too many positional inputs, binding an
input more than once, or supplying an unknown keyword SHALL fail with
`TypeError`.

#### Scenario: Construct an aggregate value directly

- **WHEN** a caller supplies all nine aggregate inputs
- **THEN** `ProfilerAggregate(...)` returns an immutable value exposing those
  exact inputs

#### Scenario: Reject an incomplete aggregate construction

- **WHEN** a caller omits a required aggregate input
- **THEN** construction fails with `TypeError`

### Requirement: Tables render validated deterministic aggregate views

`table(*, sort_by="self_total_time_ns", descending=True,
group_by_input_shape=False, row_limit=None)` SHALL return an aligned plain-text
table derived from `key_averages(...)`. `sort_by` names the aggregate field used
for primary ordering. A valid `sort_by` SHALL be one of the strings `name`,
`carrier_type`, `input_shapes`, `count`, `total_time_ns`,
`self_total_time_ns`, `mean_time_ns`, `min_time_ns`, or `max_time_ns`.
`descending` names the primary sort direction and SHALL be a `bool`.
`group_by_input_shape` SHALL have the same meaning and validation as in
`key_averages(...)`. `row_limit` names the optional maximum rows rendered; it
SHALL be `None` or a non-negative integer and SHALL default to `None`.

Ordering ties SHALL use the deterministic aggregate grouping key. Repeating a
table call over the same completed profiler and options SHALL return identical
text. A non-empty table SHALL include the selected aggregate rows and timing
columns in nanoseconds. An empty result SHALL still return the header and
separator. Enabling shape grouping SHALL include an input-shapes column.

A hashable `sort_by` value outside the supported string set SHALL fail with
`ValueError`. An unhashable `sort_by` value SHALL fail with `TypeError`. A
non-`bool` `descending` or `group_by_input_shape` SHALL fail with `TypeError`.
A `row_limit` that is negative, Boolean, or not an integer or `None` SHALL fail
with `ValueError`.

#### Scenario: Render a deterministic limited table

- **WHEN** a completed profiler renders a table sorted by count in descending
  order with `row_limit` equal to one
- **THEN** repeated calls return the same header, separator, and one aggregate
  row

#### Scenario: Render an empty table

- **WHEN** the profiler selected no events
- **THEN** `table()` returns its header and separator with no data rows

#### Scenario: Reject an unsupported sort field

- **WHEN** `sort_by` is a hashable value that does not name a supported
  aggregate field
- **THEN** `table(...)` fails with `ValueError`

#### Scenario: Reject an unhashable sort value

- **WHEN** `sort_by` is unhashable
- **THEN** `table(...)` fails with `TypeError`

#### Scenario: Reject an invalid row limit

- **WHEN** `row_limit` is negative, Boolean, or not an integer or `None`
- **THEN** `table(...)` fails with `ValueError`

### Requirement: Timing measures synchronous host execution

Profiler timing SHALL measure monotonic synchronous host wall time across the
profiled computation boundary. It SHALL begin after execution-option and
structural input preflight and SHALL end after Tensor-result validation but
before the public operation returns its result. Any asynchronous accelerator
activity that outlives the boundary's host return SHALL be outside the measured
interval. The timing fields SHALL represent synchronous host time exclusively;
device synchronization, device time, memory use, and hardware counters are
outside their semantic domain.

#### Scenario: Interpret an event duration

- **WHEN** a profiled computation boundary completes and returns while
  asynchronous device work could remain outstanding
- **THEN** the event duration describes only that completed synchronous host
  interval
