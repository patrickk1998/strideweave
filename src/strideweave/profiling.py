"""Public carrier-aware operation profiling API."""

from __future__ import annotations

import threading
import time
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
_active_profiler = threading.local()


@dataclass(frozen=True, slots=True)
class ProfilerEvent:
    """Immutable snapshot of one operation or movement profiling event.

    Args:
        id: Session-local event identifier in execution-start order.
        parent_id: Identifier of the nearest recorded parent event, if any.
        name: Canonical dispatched operation name or movement phase.
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
        >>> import strideweave as sw
        >>> tensor = sw.Tensor(
        ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> with sw.profile() as prof:
        ...     result = sw.relu(tensor)
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
class _ExternalEvent:
    key: tuple[str, int, str]
    parent_key: tuple[str, int, str] | None
    name: str
    carrier_type: type[Carrier]
    implementation_type: type[Any]
    input_shapes: InputShapes | None
    start_time_ns: int
    duration_ns: int
    thread_id: int
    succeeded: bool


def _shape_snapshot(level: object) -> ShapeSnapshot:
    return tuple(
        child if isinstance(child, int) else _shape_snapshot(child)
        for child in cast(Iterable[object], level)
    )


class _MoveProfileToken:
    __slots__ = (
        "_carrier_type",
        "_completed",
        "_implementation_type",
        "_input_shapes",
        "_profiler",
        "_request_id",
        "_start_time_ns",
        "_submitted_time_ns",
    )

    def __init__(
        self,
        profiler: Profiler | None,
        request_id: int = -1,
        carrier_type: type[Carrier] | None = None,
        implementation_type: type[Any] | None = None,
        input_shapes: InputShapes | None = None,
    ) -> None:
        self._profiler = profiler
        self._request_id = request_id
        self._carrier_type = carrier_type
        self._implementation_type = implementation_type
        self._input_shapes = input_shapes
        self._start_time_ns = time.monotonic_ns()
        self._submitted_time_ns: int | None = None
        self._completed = False

    def submitted(self, succeeded: bool) -> None:
        profiler = self._profiler
        if profiler is None or self._submitted_time_ns is not None:
            return
        finished = time.monotonic_ns()
        self._submitted_time_ns = finished
        profiler._record_move_phase(
            request_id=self._request_id,
            phase="submit",
            parent_phase=None,
            name="move.submit",
            carrier_type=cast(type[Carrier], self._carrier_type),
            implementation_type=cast(type[Any], self._implementation_type),
            input_shapes=self._input_shapes,
            start_time_ns=self._start_time_ns,
            duration_ns=finished - self._start_time_ns,
            succeeded=succeeded,
        )

    def completed(self, succeeded: bool) -> None:
        profiler = self._profiler
        if profiler is None or self._completed:
            return
        self._completed = True
        started = self._submitted_time_ns
        if started is None:
            self.submitted(False)
            started = cast(int, self._submitted_time_ns)
        finished = time.monotonic_ns()
        profiler._record_move_phase(
            request_id=self._request_id,
            phase="complete",
            parent_phase="submit",
            name="move.complete",
            carrier_type=cast(type[Carrier], self._carrier_type),
            implementation_type=cast(type[Any], self._implementation_type),
            input_shapes=self._input_shapes,
            start_time_ns=started,
            duration_ns=finished - started,
            succeeded=succeeded,
        )
        profiler._finish_move_profile()

    def abandon(self) -> None:
        profiler = self._profiler
        if profiler is None or self._completed:
            return
        self._completed = True
        profiler._finish_move_profile()


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
        >>> import strideweave as sw
        >>> tensor = sw.Tensor(
        ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> with sw.profile() as prof:
        ...     result = sw.relu(tensor)
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
        >>> import strideweave as sw
        >>> tensor = sw.Tensor(
        ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> with sw.Profiler(carriers={sw.Generic}, record_shapes=True) as prof:
        ...     result = sw.relu(tensor)
        >>> prof.events()[0].carrier_type is sw.Generic
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
        self._native_events: tuple[_RawEvent, ...] = ()
        self._external_events: list[_ExternalEvent] = []
        self._events_dirty = False
        self._external_pending = 0
        self._next_external_request = 0
        self._external_lock = threading.RLock()

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

        self._carrier_types = carrier_types
        self._record_shapes = record_shapes
        self._session = _raw_session_factory(carrier_types, record_shapes)

    def __del__(self) -> None:
        session = self._session
        if self._active and session is not None:
            session._abandon()
        if getattr(_active_profiler, "current", None) is self:
            _active_profiler.current = None

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("Profiler contexts are single-use")
        self._entered = True
        session = self._session
        if session is None:
            raise RuntimeError("Profiler native session is unavailable")
        session.start()
        self._active = True
        _active_profiler.current = self
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
        if getattr(_active_profiler, "current", None) is self:
            _active_profiler.current = None
        raw_events = session.events()
        with self._external_lock:
            self._native_events = raw_events
            self._completed = True
            self._refresh_events_locked()
            self._events_dirty = False
        self._session = None
        return False

    def _begin_move_profile(
        self, tensor: Any, implementation_type: type[Any]
    ) -> _MoveProfileToken:
        carrier_type = type(tensor.carrier)
        carrier_types = self._carrier_types
        if carrier_types is not None and carrier_type not in carrier_types:
            return _MoveProfileToken(None)
        input_shapes = (
            (_shape_snapshot(tensor.layout.shape.top_level),)
            if self._record_shapes
            else None
        )
        with self._external_lock:
            request_id = self._next_external_request
            self._next_external_request += 1
            self._external_pending += 1
        return _MoveProfileToken(
            self,
            request_id,
            carrier_type,
            implementation_type,
            input_shapes,
        )

    def _record_move_phase(
        self,
        *,
        request_id: int,
        phase: str,
        parent_phase: str | None,
        name: str,
        carrier_type: type[Carrier],
        implementation_type: type[Any],
        input_shapes: InputShapes | None,
        start_time_ns: int,
        duration_ns: int,
        succeeded: bool,
    ) -> None:
        parent_key = (
            ("move", request_id, parent_phase) if parent_phase is not None else None
        )
        event = _ExternalEvent(
            key=("move", request_id, phase),
            parent_key=parent_key,
            name=name,
            carrier_type=carrier_type,
            implementation_type=implementation_type,
            input_shapes=input_shapes,
            start_time_ns=start_time_ns,
            duration_ns=duration_ns,
            thread_id=threading.get_ident(),
            succeeded=succeeded,
        )
        with self._external_lock:
            self._external_events.append(event)
            self._events_dirty = True

    def _finish_move_profile(self) -> None:
        with self._external_lock:
            if self._external_pending > 0:
                self._external_pending -= 1
            if self._completed and self._external_pending == 0 and self._events_dirty:
                self._refresh_events_locked()
                self._events_dirty = False

    def _refresh_events_locked(self) -> None:
        combined: list[
            tuple[
                tuple[str, int, str],
                tuple[str, int, str] | None,
                ProfilerEvent,
            ]
        ] = []
        for event in self._native_events:
            key = ("native", event.id, "event")
            parent_key = (
                ("native", event.parent_id, "event")
                if event.parent_id is not None
                else None
            )
            combined.append((key, parent_key, ProfilerEvent._from_raw(event)))
        for event in self._external_events:
            combined.append(
                (
                    event.key,
                    event.parent_key,
                    ProfilerEvent(
                        id=-1,
                        parent_id=None,
                        name=event.name,
                        carrier_type=event.carrier_type,
                        implementation_type=event.implementation_type,
                        input_shapes=event.input_shapes,
                        start_time_ns=event.start_time_ns,
                        duration_ns=event.duration_ns,
                        self_time_ns=event.duration_ns,
                        thread_id=event.thread_id,
                        succeeded=event.succeeded,
                    ),
                )
            )
        combined.sort(key=lambda entry: (entry[2].start_time_ns, entry[0]))
        ids = {key: index for index, (key, _parent, _event) in enumerate(combined)}
        self._events = tuple(
            ProfilerEvent(
                id=index,
                parent_id=(ids.get(parent_key) if parent_key is not None else None),
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
            for index, (_key, parent_key, event) in enumerate(combined)
        )

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
            >>> import strideweave as sw
            >>> tensor = sw.Tensor(
            ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
            ... )
            >>> with sw.profile() as prof:
            ...     result = sw.relu(tensor)
            >>> prof.events()[0].name
            'relu'
        """

        self._require_completed()
        with self._external_lock:
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
            >>> import strideweave as sw
            >>> tensor = sw.Tensor(
            ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
            ... )
            >>> with sw.profile(record_shapes=True) as prof:
            ...     result = sw.relu(tensor)
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
        for event in self.events():
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
            >>> import strideweave as sw
            >>> tensor = sw.Tensor(
            ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
            ... )
            >>> with sw.profile() as prof:
            ...     result = sw.relu(tensor)
            >>> "Total (ns)" in prof.table(sort_by="total_time_ns")
            True
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
        >>> tensor = sw.Tensor(
        ...     sw.Generic([-1.0], dtype=sw.DType.Float32), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> with sw.profile(carriers={sw.Generic}, record_shapes=True) as prof:
        ...     result = sw.relu(tensor)
        >>> prof.events()[0].name
        'relu'
    """

    return Profiler(carriers=carriers, record_shapes=record_shapes)


def _begin_move_profile(
    tensor: Any, implementation_type: type[Any]
) -> _MoveProfileToken:
    """Capture the active caller-thread profiler for one logical move."""

    profiler = getattr(_active_profiler, "current", None)
    if not isinstance(profiler, Profiler) or not profiler._active:
        return _MoveProfileToken(None)
    return profiler._begin_move_profile(tensor, implementation_type)


__all__ = [
    "Profiler",
    "ProfilerAggregate",
    "ProfilerEvent",
    "profile",
]
