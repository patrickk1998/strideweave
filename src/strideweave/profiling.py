"""Public carrier-aware operation profiling API."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast

from .carriers.base import Carrier

type ShapeSnapshot = tuple[int | ShapeSnapshot, ...]
type InputShapes = tuple[ShapeSnapshot | None, ...]


class _RawEvent(Protocol):
    id: int
    parent_id: int | None
    name: str
    carrier_type: type[Carrier]
    implementation_type: type[Any]
    input_shapes: InputShapes | None
    start_time_ns: int
    duration_ns: int
    self_time_ns: int
    thread_id: int
    succeeded: bool


class _RawSession(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def _abandon(self) -> None: ...

    def events(self) -> tuple[_RawEvent, ...]: ...


class _RawSessionFactory(Protocol):
    def __call__(
        self,
        carrier_types: object | None = None,
        record_shapes: bool = False,
    ) -> _RawSession: ...


_operation = import_module("strideweave._operation")
_raw_session_factory = cast(
    _RawSessionFactory, getattr(_operation, "_RawProfilerSession")
)


@dataclass(frozen=True, slots=True)
class ProfilerEvent:
    """Immutable snapshot of one carrier-dispatched operation execution.

    Args:
        id: Session-local event identifier in execution-start order.
        parent_id: Identifier of the nearest recorded parent event, if any.
        name: Canonical dispatched operation name.
        carrier_type: Exact carrier class that dispatched the operation.
        implementation_type: Exact executed ``Operation`` implementation class.
        input_shapes: Hierarchical tensor shape snapshots by argument position, with
            ``None`` for non-tensor arguments or for the whole field when shape
            recording is disabled.
        start_time_ns: Monotonic host start timestamp in nanoseconds.
        duration_ns: Inclusive synchronous host wall time in nanoseconds.
        self_time_ns: Inclusive time minus nested dispatched operation time.
        thread_id: Python thread identifier that executed the operation.
        succeeded: Whether execution returned a valid tensor without raising.

    Examples:
        >>> with profile() as prof:
        ...     result = relu(tensor)
        >>> event = prof.events()[0]
        >>> event.name
        'relu'
    """

    id: int
    parent_id: int | None
    name: str
    carrier_type: type[Carrier]
    implementation_type: type[Any]
    input_shapes: InputShapes | None
    start_time_ns: int
    duration_ns: int
    self_time_ns: int
    thread_id: int
    succeeded: bool

    @classmethod
    def _from_raw(cls, event: _RawEvent) -> Self:
        return cls(
            id=event.id,
            parent_id=event.parent_id,
            name=event.name,
            carrier_type=event.carrier_type,
            implementation_type=event.implementation_type,
            input_shapes=event.input_shapes,
            start_time_ns=event.start_time_ns,
            duration_ns=event.duration_ns,
            self_time_ns=event.self_time_ns,
            thread_id=event.thread_id,
            succeeded=event.succeeded,
        )


@dataclass(frozen=True, slots=True)
class ProfilerAggregate:
    """Immutable timing summary for one profiler grouping key.

    Args:
        name: Canonical operation name shared by the grouped events.
        carrier_type: Exact dispatching carrier class shared by the events.
        input_shapes: Hierarchical input-shape key when shape grouping is enabled;
            otherwise ``None``.
        count: Number of execution attempts in the group.
        total_time_ns: Sum of inclusive host wall time in nanoseconds.
        self_total_time_ns: Sum of self host wall time in nanoseconds.
        mean_time_ns: Mean inclusive host wall time in nanoseconds.
        min_time_ns: Minimum inclusive host wall time in nanoseconds.
        max_time_ns: Maximum inclusive host wall time in nanoseconds.

    Examples:
        >>> averages = prof.key_averages()
        >>> averages[0].count >= 1
        True
    """

    name: str
    carrier_type: type[Carrier]
    input_shapes: InputShapes | None
    count: int
    total_time_ns: int
    self_total_time_ns: int
    mean_time_ns: float
    min_time_ns: int
    max_time_ns: int


@dataclass(slots=True)
class _AggregateAccumulator:
    count: int
    total_time_ns: int
    self_total_time_ns: int
    min_time_ns: int
    max_time_ns: int

    @classmethod
    def from_event(cls, event: ProfilerEvent) -> Self:
        return cls(
            count=1,
            total_time_ns=event.duration_ns,
            self_total_time_ns=event.self_time_ns,
            min_time_ns=event.duration_ns,
            max_time_ns=event.duration_ns,
        )

    def add(self, event: ProfilerEvent) -> None:
        self.count += 1
        self.total_time_ns += event.duration_ns
        self.self_total_time_ns += event.self_time_ns
        self.min_time_ns = min(self.min_time_ns, event.duration_ns)
        self.max_time_ns = max(self.max_time_ns, event.duration_ns)


def _carrier_key(carrier_type: type[Carrier]) -> tuple[str, str]:
    return carrier_type.__module__, carrier_type.__qualname__


def _aggregate_key(
    aggregate: ProfilerAggregate,
) -> tuple[str, tuple[str, str], str]:
    return (
        aggregate.name,
        _carrier_key(aggregate.carrier_type),
        repr(aggregate.input_shapes),
    )


class Profiler:
    """Single-use context manager for carrier-aware operation profiling.

    Args:
        carriers: Optional iterable of exact ``Carrier`` classes to record.
            ``None`` records every carrier class; an empty iterable records none.
        record_shapes: Whether events snapshot hierarchical tensor input shapes.

    Examples:
        >>> with Profiler(carriers={CPU}, record_shapes=True) as prof:
        ...     result = relu(tensor)
        >>> prof.events()[0].carrier_type is CPU
        True
    """

    def __init__(
        self,
        *,
        carriers: Iterable[type[Carrier]] | None = None,
        record_shapes: bool = False,
    ) -> None:
        self._session: _RawSession | None = None
        self._entered = False
        self._active = False
        self._completed = False
        self._events: tuple[ProfilerEvent, ...] = ()

        if not isinstance(record_shapes, bool):
            raise TypeError("record_shapes must be a bool")
        if carriers is None:
            carrier_types: frozenset[type[Carrier]] | None = None
        else:
            try:
                carrier_types = frozenset(carriers)
            except TypeError as exc:
                raise TypeError(
                    "carriers must be an iterable of Carrier classes"
                ) from exc
            if any(
                not isinstance(carrier_type, type)
                or not issubclass(carrier_type, Carrier)
                for carrier_type in carrier_types
            ):
                raise TypeError("carriers must contain only Carrier subclasses")

        self._session = _raw_session_factory(carrier_types, record_shapes)

    def __del__(self) -> None:
        session = self._session
        if self._active and session is not None:
            session._abandon()

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("Profiler contexts are single-use")
        self._entered = True
        session = self._session
        if session is None:
            raise RuntimeError("Profiler native session is unavailable")
        session.start()
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if not self._active:
            raise RuntimeError("Profiler context is not active")
        session = self._session
        if session is None:
            raise RuntimeError("Profiler native session is unavailable")
        session.stop()
        self._active = False
        raw_events = session.events()
        self._events = tuple(ProfilerEvent._from_raw(event) for event in raw_events)
        self._session = None
        self._completed = True
        return False

    def _require_completed(self) -> None:
        if not self._completed:
            raise RuntimeError("Profiler results are available only after context exit")

    def events(self) -> tuple[ProfilerEvent, ...]:
        """Return immutable raw events in execution-start order.

        Args:
            None.

        Returns:
            Tuple of immutable operation execution events.

        Examples:
            >>> with profile() as prof:
            ...     result = relu(tensor)
            >>> prof.events()[0].name
            'relu'
        """

        self._require_completed()
        return self._events

    def key_averages(
        self, *, group_by_input_shape: bool = False
    ) -> tuple[ProfilerAggregate, ...]:
        """Aggregate raw events by operation name and exact carrier class.

        Args:
            group_by_input_shape: Also include hierarchical input shapes in the
                grouping key. Shape recording must have been enabled to distinguish
                shapes; otherwise all events retain the ``None`` shape key.

        Returns:
            Deterministically ordered immutable aggregate rows.

        Examples:
            >>> rows = prof.key_averages(group_by_input_shape=True)
            >>> rows[0].count >= 1
            True
        """

        if not isinstance(group_by_input_shape, bool):
            raise TypeError("group_by_input_shape must be a bool")
        self._require_completed()
        grouped: dict[
            tuple[str, type[Carrier], InputShapes | None], _AggregateAccumulator
        ] = {}
        for event in self._events:
            input_shapes = event.input_shapes if group_by_input_shape else None
            key = (event.name, event.carrier_type, input_shapes)
            accumulator = grouped.get(key)
            if accumulator is None:
                grouped[key] = _AggregateAccumulator.from_event(event)
            else:
                accumulator.add(event)

        aggregates = []
        for (name, carrier_type, input_shapes), accumulator in grouped.items():
            aggregates.append(
                ProfilerAggregate(
                    name=name,
                    carrier_type=carrier_type,
                    input_shapes=input_shapes,
                    count=accumulator.count,
                    total_time_ns=accumulator.total_time_ns,
                    self_total_time_ns=accumulator.self_total_time_ns,
                    mean_time_ns=(accumulator.total_time_ns / accumulator.count),
                    min_time_ns=accumulator.min_time_ns,
                    max_time_ns=accumulator.max_time_ns,
                )
            )
        return tuple(sorted(aggregates, key=_aggregate_key))

    def table(
        self,
        *,
        sort_by: str = "self_total_time_ns",
        descending: bool = True,
        group_by_input_shape: bool = False,
        row_limit: int | None = None,
    ) -> str:
        """Render deterministic profiler aggregates as an aligned text table.

        Args:
            sort_by: Aggregate field used for the primary ordering. Supported values
                are ``name``, ``carrier_type``, ``input_shapes``, ``count``,
                ``total_time_ns``, ``self_total_time_ns``, ``mean_time_ns``,
                ``min_time_ns``, and ``max_time_ns``.
            descending: Whether the primary sort is descending. Ties always use the
                deterministic name, carrier, and shape grouping key.
            group_by_input_shape: Include hierarchical input shapes in grouping and
                display.
            row_limit: Optional non-negative maximum number of rows to render.

        Returns:
            Aligned plain-text table with nanosecond timing columns.

        Examples:
            >>> print(prof.table(sort_by="total_time_ns"))
            Name  Carrier  Calls  Total (ns)  Self total (ns)  Mean (ns)  Min (ns)  Max (ns)
            ...
        """

        valid_sort_fields = {
            "carrier_type",
            "count",
            "input_shapes",
            "max_time_ns",
            "mean_time_ns",
            "min_time_ns",
            "name",
            "self_total_time_ns",
            "total_time_ns",
        }
        if sort_by not in valid_sort_fields:
            raise ValueError(f"unsupported profiler table sort field: {sort_by!r}")
        if not isinstance(descending, bool):
            raise TypeError("descending must be a bool")
        if row_limit is not None and (
            not isinstance(row_limit, int)
            or isinstance(row_limit, bool)
            or row_limit < 0
        ):
            raise ValueError("row_limit must be a non-negative integer or None")

        aggregates = list(self.key_averages(group_by_input_shape=group_by_input_shape))

        def primary_key(aggregate: ProfilerAggregate) -> Any:
            if sort_by == "carrier_type":
                return _carrier_key(aggregate.carrier_type)
            if sort_by == "input_shapes":
                return repr(aggregate.input_shapes)
            return getattr(aggregate, sort_by)

        aggregates.sort(key=_aggregate_key)
        aggregates.sort(key=primary_key, reverse=descending)
        if row_limit is not None:
            aggregates = aggregates[:row_limit]

        headers = ["Name", "Carrier"]
        if group_by_input_shape:
            headers.append("Input shapes")
        headers.extend(
            [
                "Calls",
                "Total (ns)",
                "Self total (ns)",
                "Mean (ns)",
                "Min (ns)",
                "Max (ns)",
            ]
        )
        rows: list[list[str]] = []
        for aggregate in aggregates:
            row = [
                aggregate.name,
                ".".join(_carrier_key(aggregate.carrier_type)),
            ]
            if group_by_input_shape:
                row.append(repr(aggregate.input_shapes))
            row.extend(
                [
                    str(aggregate.count),
                    str(aggregate.total_time_ns),
                    str(aggregate.self_total_time_ns),
                    f"{aggregate.mean_time_ns:.1f}",
                    str(aggregate.min_time_ns),
                    str(aggregate.max_time_ns),
                ]
            )
            rows.append(row)

        widths = [
            max([len(headers[index]), *(len(row[index]) for row in rows)])
            for index in range(len(headers))
        ]
        header = "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(headers)
        )
        separator = "  ".join("-" * width for width in widths)
        body = [
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            for row in rows
        ]
        return "\n".join([header, separator, *body])


def profile(
    *,
    carriers: Iterable[type[Carrier]] | None = None,
    record_shapes: bool = False,
) -> Profiler:
    """Create a single-use carrier-aware operation profiling context.

    Args:
        carriers: Optional iterable of exact ``Carrier`` classes to record.
            ``None`` records all carriers. Carrier instances and unrelated classes
            are rejected.
        record_shapes: Whether events snapshot hierarchical tensor input shapes.

    Returns:
        Profiler context whose results become available after context exit.

    Examples:
        >>> import strideweave as sw
        >>> with sw.profile(carriers={sw.CPU}, record_shapes=True) as prof:
        ...     result = sw.relu(tensor)
        >>> prof.events()[0].name
        'relu'
    """

    return Profiler(carriers=carriers, record_shapes=record_shapes)


__all__ = [
    "Profiler",
    "ProfilerAggregate",
    "ProfilerEvent",
    "profile",
]
