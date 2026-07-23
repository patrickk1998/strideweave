from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import strideweave as sw
import strideweave.operation as operation
from strideweave import Evictable, Generic, Layout, Shape, Stride, Tensor


def tensor(values: list[float]) -> Tensor:
    return Tensor(
        Generic(values),
        0,
        Layout(Shape(len(values)), Stride(1)),
    )


def evictable_tensor(values: list[float]) -> Tensor:
    return Tensor(
        Evictable(Generic(values), Generic([])),
        0,
        Layout(Shape(len(values)), Stride(1)),
    )


def test_profiler_public_api_exports():
    for name in ("profile", "Profiler", "ProfilerEvent", "ProfilerAggregate"):
        assert name in sw.__all__
        assert name in operation.__all__
        assert getattr(sw, name) is getattr(operation, name)


def test_profiler_context_records_all_and_selected_exact_carriers():
    value = evictable_tensor([-1.0, 2.0])

    with sw.profile() as all_profiler:
        sw.relu(value)
    with sw.profile(carriers={Generic}) as generic_profiler:
        sw.relu(value)
    with sw.profile(carriers=set()) as empty_profiler:
        sw.relu(value)

    assert [event.carrier_type for event in all_profiler.events()] == [
        Evictable,
        Generic,
    ]
    assert [event.carrier_type for event in generic_profiler.events()] == [Generic]
    assert empty_profiler.events() == ()


def test_profiler_carrier_filter_uses_exact_classes():
    class CustomGeneric(Generic):
        pass

    value = Tensor(
        CustomGeneric([1.0]),
        0,
        Layout(Shape(1), Stride(1)),
    )

    with sw.profile(carriers={Generic}) as base_profiler:
        with pytest.raises(TypeError, match="rhs must be a Tensor"):
            value.carrier.dispatch_op("add").forward(value, "not a tensor")
    with sw.profile(carriers={CustomGeneric}) as custom_profiler:
        with pytest.raises(TypeError, match="rhs must be a Tensor"):
            value.carrier.dispatch_op("add").forward(value, "not a tensor")

    assert base_profiler.events() == ()
    (event,) = custom_profiler.events()
    assert event.carrier_type is CustomGeneric


@pytest.mark.parametrize(
    "carriers",
    [Generic([1.0]), [Generic([1.0])], [str], 1],
)
def test_profiler_rejects_invalid_carrier_filters(carriers):
    with pytest.raises(TypeError, match="Carrier"):
        sw.profile(carriers=carriers)


def test_profiler_rejects_non_boolean_options():
    with pytest.raises(TypeError, match="record_shapes"):
        sw.profile(record_shapes=1)  # type: ignore[arg-type]

    with sw.profile() as profiler:
        sw.relu(tensor([1.0]))

    with pytest.raises(TypeError, match="group_by_input_shape"):
        profiler.key_averages(group_by_input_shape=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="descending"):
        profiler.table(descending=1)  # type: ignore[arg-type]


def test_profiler_results_are_available_only_after_context_exit():
    profiler = sw.profile()

    with pytest.raises(RuntimeError, match="only after context exit"):
        profiler.events()
    with profiler:
        sw.relu(tensor([1.0]))
        with pytest.raises(RuntimeError, match="only after context exit"):
            profiler.key_averages()

    assert len(profiler.events()) == 1


def test_profiler_context_is_single_use_and_rejects_nesting():
    profiler = sw.profile()
    with profiler:
        pass

    with pytest.raises(RuntimeError, match="single-use"):
        with profiler:
            pass

    with sw.profile():
        with pytest.raises(RuntimeError, match="already active"):
            with sw.profile():
                pass


def test_profiler_preserves_events_when_context_body_raises():
    value = tensor([1.0])
    profiler = sw.profile()

    with pytest.raises(TypeError, match="rhs must be a Tensor"):
        with profiler:
            value.carrier.dispatch_op("add").forward(value, "not a tensor")

    (event,) = profiler.events()
    assert event.name == "add"
    assert not event.succeeded
    assert getattr(profiler, "_session") is None


def test_completed_profiler_releases_native_shape_event_storage():
    value = tensor([1.0, 2.0])

    with sw.profile(record_shapes=True) as profiler:
        for _ in range(1_000):
            sw.relu(value)

    events = profiler.events()
    assert len(events) == 1_000
    assert events[0].input_shapes == ((2,),)
    assert events[-1].input_shapes == ((2,),)
    assert getattr(profiler, "_session") is None


def test_profiler_events_are_immutable_and_keep_execution_start_order():
    value = tensor([1.0])

    with sw.profile() as profiler:
        sw.relu(value)
        sw.mul(value, 2.0)

    events = profiler.events()
    assert [(event.id, event.name) for event in events] == [(0, "relu"), (1, "mul")]
    with pytest.raises(FrozenInstanceError):
        events[0].name = "changed"  # type: ignore[misc]


def test_profiler_key_averages_derive_all_timing_fields_from_raw_events():
    value = tensor([1.0])

    with sw.profile() as profiler:
        sw.relu(value)
        sw.relu(value)
        sw.mul(value, 2.0)

    events = profiler.events()
    rows = {row.name: row for row in profiler.key_averages()}
    relu_events = [event for event in events if event.name == "relu"]
    relu = rows["relu"]

    assert relu.carrier_type is Generic
    assert relu.input_shapes is None
    assert relu.count == 2
    assert relu.total_time_ns == sum(event.duration_ns for event in relu_events)
    assert relu.self_total_time_ns == sum(event.self_time_ns for event in relu_events)
    assert relu.mean_time_ns == relu.total_time_ns / relu.count
    assert relu.min_time_ns == min(event.duration_ns for event in relu_events)
    assert relu.max_time_ns == max(event.duration_ns for event in relu_events)
    assert rows["mul"].count == 1


def test_profiler_key_averages_optionally_group_by_hierarchical_input_shape():
    one = tensor([1.0])
    two = tensor([1.0, 2.0])

    with sw.profile(record_shapes=True) as profiler:
        sw.relu(one)
        sw.relu(two)

    (combined,) = profiler.key_averages()
    grouped = profiler.key_averages(group_by_input_shape=True)

    assert combined.count == 2
    assert combined.input_shapes is None
    assert [(row.input_shapes, row.count) for row in grouped] == [
        (((1,),), 1),
        (((2,),), 1),
    ]


def test_profiler_aggregates_nested_carrier_types_separately():
    with sw.profile() as profiler:
        sw.relu(evictable_tensor([1.0]))

    rows = profiler.key_averages()
    assert [(row.name, row.carrier_type) for row in rows] == [
        ("relu", Evictable),
        ("relu", Generic),
    ]


def test_profiler_table_is_deterministic_sortable_and_limitable():
    value = tensor([1.0])

    with sw.profile() as profiler:
        sw.relu(value)
        sw.relu(value)
        sw.mul(value, 2.0)

    table = profiler.table(sort_by="count", descending=True)
    assert table == profiler.table(sort_by="count", descending=True)
    assert table.splitlines()[2].startswith("relu")
    assert len(profiler.table(row_limit=1).splitlines()) == 3

    with pytest.raises(ValueError, match="sort field"):
        profiler.table(sort_by="unknown")
    with pytest.raises(ValueError, match="row_limit"):
        profiler.table(row_limit=-1)
    with pytest.raises(ValueError, match="row_limit"):
        profiler.table(row_limit=True)


def test_profiler_table_handles_empty_and_shape_grouped_results():
    with sw.profile(carriers=set()) as empty:
        sw.relu(tensor([1.0]))
    with sw.profile(record_shapes=True) as shaped:
        sw.relu(tensor([1.0]))

    assert len(empty.table().splitlines()) == 2
    assert "Input shapes" in shaped.table(group_by_input_shape=True).splitlines()[0]
