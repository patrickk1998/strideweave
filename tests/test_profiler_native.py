from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Protocol, cast

import pytest

import strideweave as sw
from strideweave import CPU, DType, Evictable, Generic, Layout, Shape, Stride, Tensor


class RawEvent(Protocol):
    id: int
    parent_id: int | None
    name: str
    carrier_type: type[Any]
    implementation_type: type[Any]
    input_shapes: tuple[Any, ...] | None
    start_time_ns: int
    duration_ns: int
    self_time_ns: int
    thread_id: int
    succeeded: bool


class RawSession(Protocol):
    is_active: bool

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def _abandon(self) -> None: ...

    def events(self) -> tuple[RawEvent, ...]: ...


class RawSessionFactory(Protocol):
    def __call__(
        self,
        carrier_types: object | None = None,
        record_shapes: bool = False,
    ) -> RawSession: ...


class NativeOperationModule(Protocol):
    _RawProfilerSession: RawSessionFactory


native_operation = cast(NativeOperationModule, import_module("strideweave._operation"))


@contextmanager
def record(
    carrier_types: set[type[Any]] | None = None, *, record_shapes: bool = False
) -> Iterator[RawSession]:
    session = native_operation._RawProfilerSession(carrier_types, record_shapes)
    session.start()
    try:
        yield session
    finally:
        session.stop()


def generic_tensor(
    values: list[float],
    layout: Layout | None = None,
) -> Tensor:
    if layout is None:
        layout = Layout(Shape(len(values)), Stride(1))
    return Tensor(Generic(values), 0, layout)


def cpu_tensor(values: list[float]) -> Tensor:
    carrier = CPU(len(values), dtype=DType.Float32)
    for index, value in enumerate(values):
        carrier[index] = value
    return Tensor(carrier, 0, Layout(Shape(len(values)), Stride(1)))


def evictable_tensor(values: list[float]) -> Tensor:
    return Tensor(
        Evictable(Generic(values), Generic([])),
        0,
        Layout(Shape(len(values)), Stride(1)),
    )


@pytest.mark.parametrize(
    ("tensor_factory", "carrier_type"),
    [(generic_tensor, Generic), (cpu_tensor, CPU)],
)
def test_raw_profiler_records_dispatched_generic_and_cpu_execution(
    tensor_factory, carrier_type
):
    tensor = tensor_factory([-1.0, 2.0])

    with record() as session:
        sw.relu(tensor)

    (event,) = session.events()
    assert event.id == 0
    assert event.parent_id is None
    assert event.name == "relu"
    assert event.carrier_type is carrier_type
    assert event.implementation_type.__name__.endswith("ReLUOperation")
    assert event.input_shapes is None
    assert event.start_time_ns > 0
    assert event.duration_ns >= 0
    assert event.self_time_ns == event.duration_ns
    assert event.thread_id == threading.get_ident()
    assert event.succeeded


def test_raw_profiler_snapshots_hierarchical_shapes_and_non_tensor_positions():
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]]))
    tensor = generic_tensor([float(index) for index in range(24)], layout)

    with record(record_shapes=True) as session:
        sw.mul(tensor, 2.0)

    (event,) = session.events()
    assert event.input_shapes == ((2, (3, 4)), None)
    assert event.input_shapes is not None
    assert event.input_shapes[0] is layout.shape.top_level


def test_raw_profiler_records_nested_evictable_events_in_start_order():
    tensor = evictable_tensor([-1.0, 2.0])

    with record() as session:
        sw.relu(tensor)

    outer, inner = session.events()
    assert [outer.id, inner.id] == [0, 1]
    assert [outer.carrier_type, inner.carrier_type] == [Evictable, Generic]
    assert outer.parent_id is None
    assert inner.parent_id == outer.id
    assert outer.self_time_ns == outer.duration_ns - inner.duration_ns
    assert inner.self_time_ns == inner.duration_ns


def test_raw_profiler_exact_filter_keeps_hidden_child_time_out_of_parent_self_time():
    tensor = evictable_tensor([-1.0, 2.0])

    with record({Evictable}) as session:
        sw.relu(tensor)

    (event,) = session.events()
    assert event.carrier_type is Evictable
    assert event.parent_id is None
    assert event.self_time_ns < event.duration_ns


def test_raw_profiler_exact_filter_makes_selected_inner_event_a_root():
    tensor = evictable_tensor([-1.0, 2.0])

    with record({Generic}) as session:
        sw.relu(tensor)

    (event,) = session.events()
    assert event.carrier_type is Generic
    assert event.parent_id is None
    assert event.self_time_ns == event.duration_ns


def test_raw_profiler_records_failure_and_restores_execution_stack():
    tensor = generic_tensor([1.0])

    with record() as session:
        operation = tensor.carrier.dispatch_op("add")
        with pytest.raises(TypeError, match="rhs must be a Tensor"):
            operation.forward(tensor, "not a tensor")
        sw.relu(tensor)

    failed, succeeded = session.events()
    assert (failed.name, failed.succeeded) == ("add", False)
    assert (succeeded.name, succeeded.succeeded) == ("relu", True)
    assert failed.parent_id is None
    assert succeeded.parent_id is None


def test_raw_profiler_excludes_unannotated_operations_and_move_registry_work():
    tensor = generic_tensor([1.0])
    destination = Generic([0.0])

    with record() as session:
        sw.GenericReLUOperation().forward(tensor)
        sw.move(tensor, destination)

    assert session.events() == ()


def test_raw_profiler_session_is_current_thread_local():
    main_tensor = generic_tensor([1.0])
    worker_tensor = generic_tensor([2.0])

    with record() as session:
        worker = threading.Thread(target=sw.relu, args=(worker_tensor,))
        worker.start()
        worker.join()
        sw.relu(main_tensor)

    (event,) = session.events()
    assert event.thread_id == threading.get_ident()


def test_rejected_cross_thread_stop_is_abandoned_and_owner_thread_recovers():
    session = native_operation._RawProfilerSession(None, False)
    session.start()
    errors = []

    def reject_stop():
        try:
            session.stop()
        except RuntimeError as error:
            errors.append(error)

    worker = threading.Thread(target=reject_stop)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert str(errors[0]) == "Profiler session must be stopped on its active thread"
    assert not session.is_active

    for _ in range(10):
        sw.relu(generic_tensor([1.0]))

    assert session.events() == ()
    with record() as replacement:
        sw.relu(generic_tensor([2.0]))
    assert len(replacement.events()) == 1


def test_abandoned_profiler_finalization_recovers_without_process_failure():
    script = textwrap.dedent(
        """
        import gc
        import threading

        import strideweave as sw

        profiler = sw.profile()
        profiler.__enter__()
        transferred = [profiler]
        del profiler

        def reject_exit_and_drop():
            transferred_profiler = transferred.pop()
            try:
                transferred_profiler.__exit__(None, None, None)
            except RuntimeError:
                pass

        worker = threading.Thread(target=reject_exit_and_drop)
        worker.start()
        worker.join()
        gc.collect()

        tensor = sw.Tensor(
            sw.Generic([1.0]),
            0,
            sw.Layout(sw.Shape(1), sw.Stride(1)),
        )
        with sw.profile() as recovered:
            sw.relu(tensor)
        print(f"recovered {len(recovered.events())}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "recovered 1"


def test_raw_profiler_events_are_read_only_snapshots():
    with record(record_shapes=True) as session:
        sw.relu(generic_tensor([1.0]))

    (event,) = session.events()
    with pytest.raises(AttributeError):
        event.name = "changed"


def test_raw_profiler_session_is_single_use_and_rejects_nesting():
    outer = native_operation._RawProfilerSession(None, False)
    nested = native_operation._RawProfilerSession(None, False)

    outer.start()
    assert outer.is_active
    with pytest.raises(RuntimeError, match="already active"):
        nested.start()
    outer.stop()
    assert not outer.is_active
    with pytest.raises(RuntimeError, match="only be started once"):
        outer.start()
