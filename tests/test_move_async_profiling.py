from __future__ import annotations

import threading
import time

import pytest

import strideweave as sw
from strideweave.carriers.extension import PreparedTransfer, TransferRequest
from tests.test_move_async import (
    copying_prepare,
    fresh_carrier,
    make_tensor,
    register_pair,
)


class GateCompletion:
    """A provider completion whose wait can be released by the test."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._error: BaseException | None = None
        self.wait_started = threading.Event()

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self.wait_started.set()
        self._event.wait()
        if self._error is not None:
            raise self._error

    def finish(self, error: BaseException | None = None) -> None:
        self._error = error
        self._event.set()


def pending_route(
    source_class: type[sw.Carrier],
    destination_class: type[sw.Carrier],
    completion: GateCompletion,
) -> None:
    def prepare(request: TransferRequest) -> PreparedTransfer:
        prepared = copying_prepare(request)

        def submit() -> GateCompletion:
            prepared.submit()
            return completion

        return PreparedTransfer(submit)

    register_pair(source_class, destination_class, prepare)


def test_move_profile_correlates_submit_and_completion_after_context_exit() -> None:
    source_class = fresh_carrier("ProfileSource")
    destination_class = fresh_carrier("ProfileDestination")
    completion = GateCompletion()
    pending_route(source_class, destination_class, completion)

    source = make_tensor(source_class, [1.0, 2.0])
    destination = destination_class(2, dtype=sw.DType.Float32)
    with sw.profile(carriers={source_class}, record_shapes=True) as profiler:
        handle = sw.move_async(source, destination)
        assert not handle.done
        assert completion.wait_started.wait(timeout=2)

    submit_events = profiler.events()
    assert [event.name for event in submit_events] == ["move.submit"]
    assert submit_events[0].carrier_type is source_class
    assert submit_events[0].input_shapes == ((2,),)
    release_at = time.monotonic_ns()
    completion.finish()
    assert handle.wait()[0] == 1.0

    events = profiler.events()
    assert [event.name for event in events] == ["move.submit", "move.complete"]
    submit, complete = events
    assert complete.parent_id == submit.id
    assert complete.carrier_type is source_class
    assert complete.input_shapes == ((2,),)
    assert complete.implementation_type is submit.implementation_type
    assert complete.succeeded
    assert complete.start_time_ns <= release_at
    assert complete.start_time_ns + complete.duration_ns >= release_at


def test_move_profile_failure_is_terminal_and_repeated_wait_does_not_duplicate() -> (
    None
):
    source_class = fresh_carrier("ProfileFailureSource")
    destination_class = fresh_carrier("ProfileFailureDestination")
    completion = GateCompletion()
    pending_route(source_class, destination_class, completion)

    source = make_tensor(source_class, [3.0])
    destination = destination_class(1, dtype=sw.DType.Float32)
    failure = RuntimeError("profiled provider failure")
    with sw.profile(carriers={source_class}, record_shapes=True) as profiler:
        handle = sw.move_async(source, destination)
        assert completion.wait_started.wait(timeout=2)

    completion.finish(failure)
    with pytest.raises(RuntimeError) as first:
        handle.wait()
    with pytest.raises(RuntimeError) as second:
        handle.wait()

    assert first.value is failure
    assert second.value is failure
    events = profiler.events()
    assert [event.name for event in events] == ["move.submit", "move.complete"]
    submit, complete = events
    assert complete.parent_id == submit.id
    assert submit.succeeded
    assert not complete.succeeded
    assert complete.input_shapes == ((1,),)
    assert len(profiler.events()) == 2


def test_many_post_exit_completions_materialize_one_terminal_event_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    move_count = 40
    source_class = fresh_carrier("ProfileManySource")
    destination_class = fresh_carrier("ProfileManyDestination")
    completions = [GateCompletion() for _ in range(move_count)]
    remaining = iter(completions)

    def prepare(request: TransferRequest) -> PreparedTransfer:
        prepared = copying_prepare(request)
        completion = next(remaining)

        def submit() -> GateCompletion:
            prepared.submit()
            return completion

        return PreparedTransfer(submit)

    register_pair(source_class, destination_class, prepare)
    profiler = sw.profile(carriers={source_class}, record_shapes=True)
    refresh_count = 0
    original_refresh = getattr(profiler, "_refresh_events_locked")

    def count_refresh() -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh()

    monkeypatch.setattr(profiler, "_refresh_events_locked", count_refresh)
    handles = []
    with profiler:
        for index in range(move_count):
            source = make_tensor(source_class, [float(index)])
            destination = destination_class(1, dtype=sw.DType.Float32)
            handles.append(sw.move_async(source, destination))

    assert refresh_count == 1
    assert len(profiler.events()) == move_count
    finishers = [
        threading.Thread(target=completion.finish) for completion in completions
    ]
    for finisher in finishers:
        finisher.start()
    for finisher in finishers:
        finisher.join(timeout=2)
        assert not finisher.is_alive()
    for index, handle in enumerate(handles):
        assert handle.wait()[0] == float(index)

    assert refresh_count == 2
    events = profiler.events()
    assert len(events) == 2 * move_count
    assert [event.start_time_ns for event in events] == sorted(
        event.start_time_ns for event in events
    )
    assert [event.id for event in events] == list(range(2 * move_count))
    submits = {event.id: event for event in events if event.name == "move.submit"}
    completes = [event for event in events if event.name == "move.complete"]
    assert len(submits) == len(completes) == move_count
    assert {event.parent_id for event in completes} == set(submits)
    assert all(
        event.parent_id is not None
        and submits[event.parent_id].carrier_type is event.carrier_type is source_class
        and submits[event.parent_id].input_shapes == event.input_shapes == ((1,),)
        for event in completes
    )
    assert profiler.events() == events
    first_averages = profiler.key_averages(group_by_input_shape=True)
    assert profiler.key_averages(group_by_input_shape=True) == first_averages
    assert {(row.name, row.count) for row in first_averages} == {
        ("move.submit", move_count),
        ("move.complete", move_count),
    }
