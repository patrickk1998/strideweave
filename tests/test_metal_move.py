from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave.carriers.metal._jit import MetalJITCache
from strideweave.carriers.move import (
    CpuToMetalMoveOperation,
    MetalToCpuMoveOperation,
    MetalToMetalMoveOperation,
    dispatch_move,
)

pytestmark = pytest.mark.metal


def _cpu_tensor(
    physical_values: list[Any],
    *,
    dtype: sw.DType,
    layout: sw.Layout | None = None,
    offset: int = 0,
) -> sw.Tensor:
    carrier = sw.CPU(len(physical_values), dtype=dtype)
    for index, value in enumerate(physical_values):
        carrier[index] = value
    effective_layout = layout or sw.Layout(sw.Shape(len(physical_values)), sw.Stride(1))
    return sw.Tensor(carrier, offset, effective_layout)


def _logical_values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _physical_values(carrier: Any, count: int | None = None) -> list[Any]:
    size = carrier.size() if count is None else count
    return [carrier[index] for index in range(size)]


def _internal(carrier: object) -> Any:
    return cast(Any, carrier)


def test_exact_metal_move_pairs_are_eagerly_registered_and_exported() -> None:
    import strideweave.operation as operation

    assert dispatch_move(sw.CPU, sw.Metal) is CpuToMetalMoveOperation
    assert dispatch_move(sw.Metal, sw.CPU) is MetalToCpuMoveOperation
    assert dispatch_move(sw.Metal, sw.Metal) is MetalToMetalMoveOperation
    assert operation.CpuToMetalMoveOperation is CpuToMetalMoveOperation
    assert operation.MetalToCpuMoveOperation is MetalToCpuMoveOperation
    assert operation.MetalToMetalMoveOperation is MetalToMetalMoveOperation
    assert sw.CpuToMetalMoveOperation is CpuToMetalMoveOperation
    assert sw.MetalToCpuMoveOperation is MetalToCpuMoveOperation
    assert sw.MetalToMetalMoveOperation is MetalToMetalMoveOperation


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (sw.DType.Float32, [1.25, -2.5, 3.75]),
        (sw.DType.Int32, [1, -2, 3]),
        (sw.DType.Bool, [True, False, True]),
    ],
    ids=["float32", "int32", "bool"],
)
def test_cpu_metal_cpu_and_metal_metal_bulk_moves_preserve_dtype_and_versions(
    dtype: sw.DType, values: list[Any]
) -> None:
    source = _cpu_tensor(values, dtype=dtype)
    source_version = source.carrier.version
    first_metal = sw.Metal(len(values), dtype=dtype)
    first_version = first_metal.version

    on_metal = sw.move(source, first_metal)

    assert source.carrier.is_released()
    assert first_metal.version == first_version
    assert on_metal.dtype() is dtype
    assert _logical_values(on_metal) == values

    second_metal = sw.Metal(len(values) + 2, dtype=dtype)
    second_version = second_metal.version
    on_second_metal = sw.move(on_metal, second_metal)

    assert first_metal.is_released()
    assert second_metal.version == second_version
    assert _logical_values(on_second_metal) == values

    on_cpu = sw.move(on_second_metal, sw.CPU(len(values) + 2, dtype=dtype))

    assert second_metal.is_released()
    assert on_cpu.carrier.version == 0
    assert _logical_values(on_cpu) == values
    assert source_version == len(values)


@pytest.mark.parametrize(
    "layout",
    [
        sw.Layout(sw.Shape(3), sw.Stride(2)),
        sw.Layout(sw.Shape([2, [2, 2]]), sw.Stride([1, [3, 7]])),
    ],
    ids=["flat-holes", "hierarchical-holes"],
)
def test_bulk_moves_copy_exact_offset_physical_span_including_holes(
    layout: sw.Layout,
) -> None:
    offset = 2
    physical = [float(index + 10) for index in range(offset + layout.cosize + 2)]
    expected_span = physical[offset : offset + layout.cosize]
    source = _cpu_tensor(
        physical,
        dtype=sw.DType.Float32,
        layout=layout,
        offset=offset,
    )
    metal = sw.Metal(layout.cosize + 2, dtype=sw.DType.Float32)

    moved = sw.move(source, metal)

    assert moved.offset == 0
    assert moved.layout == layout
    assert _physical_values(metal, layout.cosize) == expected_span
    assert _physical_values(metal)[layout.cosize :] == [0.0, 0.0]

    cpu = sw.CPU(layout.cosize + 2, dtype=sw.DType.Float32)
    cpu[layout.cosize] = -91.0
    cpu[layout.cosize + 1] = -92.0
    restored = sw.move(moved, cpu)

    assert restored.layout == layout
    assert _physical_values(cpu, layout.cosize) == expected_span
    assert _physical_values(cpu)[layout.cosize :] == [-91.0, -92.0]


def test_bulk_move_preserves_stride_zero_layout_without_materializing_aliases() -> None:
    layout = sw.Layout(sw.Shape([4, 3]), sw.Stride([0, 1]))
    source = _cpu_tensor(
        [99.0, 2.0, 5.0, 7.0, 101.0],
        dtype=sw.DType.Float32,
        layout=layout,
        offset=1,
    )

    moved = sw.move(source, sw.Metal(layout.cosize, dtype=sw.DType.Float32))

    assert moved.layout == layout
    assert moved.carrier.size() == 3
    assert _physical_values(moved.carrier) == [2.0, 5.0, 7.0]
    assert [moved[i, j] for i in range(4) for j in range(3)] == [
        2.0,
        5.0,
        7.0,
    ] * 4


@dataclass
class _RuntimeSpy:
    wrapped: Any
    source: Any
    fail_call: int | None = None
    calls: int = 0

    @property
    def torch(self) -> Any:
        return self.wrapped.torch

    @property
    def tilelang(self) -> Any:
        return self.wrapped.tilelang

    def synchronize(self) -> None:
        self.calls += 1
        assert not self.source.is_released()
        self.wrapped.synchronize()
        if self.calls == self.fail_call:
            raise RuntimeError("injected MPS synchronization failure")


def test_cpu_to_metal_releases_only_after_sync_and_is_failure_atomic() -> None:
    source = _cpu_tensor([1.0, 2.0, 3.0], dtype=sw.DType.Float32)
    destination = sw.Metal(3, dtype=sw.DType.Float32)
    internal_destination = _internal(destination)
    original_storage = internal_destination._require_storage()
    original_values = _physical_values(destination)
    source_version = source.carrier.version
    destination_version = destination.version
    spy = _RuntimeSpy(internal_destination._runtime, source.carrier, fail_call=1)
    internal_destination._runtime = spy

    with pytest.raises(RuntimeError, match="injected MPS synchronization failure"):
        sw.move(source, destination)

    assert spy.calls == 1
    assert not source.carrier.is_released()
    assert source.carrier.version == source_version
    assert internal_destination._require_storage() is original_storage
    internal_destination._runtime = spy.wrapped
    assert _physical_values(destination) == original_values
    assert destination.version == destination_version


def test_metal_to_metal_sync_failure_keeps_both_visible_storages_unchanged() -> None:
    cpu = _cpu_tensor([4.0, 5.0], dtype=sw.DType.Float32)
    source = sw.move(cpu, sw.Metal(2, dtype=sw.DType.Float32))
    destination = sw.Metal(2, dtype=sw.DType.Float32)
    internal_source = _internal(source.carrier)
    internal_destination = _internal(destination)
    original_storage = internal_destination._require_storage()
    original_values = _physical_values(destination)
    spy = _RuntimeSpy(internal_source._runtime, source.carrier, fail_call=1)
    internal_source._runtime = spy

    with pytest.raises(RuntimeError, match="injected MPS synchronization failure"):
        sw.move(source, destination)

    assert not source.carrier.is_released()
    assert internal_destination._require_storage() is original_storage
    internal_destination._runtime.synchronize()
    assert _physical_values(destination) == original_values
    assert source.carrier.version == 0
    assert destination.version == 0


def test_metal_to_cpu_second_sync_failure_leaves_cpu_destination_untouched() -> None:
    source = sw.move(
        _cpu_tensor([4.0, 5.0], dtype=sw.DType.Float32),
        sw.Metal(2, dtype=sw.DType.Float32),
    )
    destination = sw.CPU(2, dtype=sw.DType.Float32)
    destination[0] = -1.0
    destination[1] = -2.0
    source_version = source.carrier.version
    destination_version = destination.version
    internal_source = _internal(source.carrier)
    spy = _RuntimeSpy(internal_source._runtime, source.carrier, fail_call=2)
    internal_source._runtime = spy

    with pytest.raises(RuntimeError, match="injected MPS synchronization failure"):
        sw.move(source, destination)

    assert spy.calls == 2
    assert not source.carrier.is_released()
    assert _logical_values(source) == [4.0, 5.0]
    assert _physical_values(destination) == [-1.0, -2.0]
    assert source.carrier.version == source_version
    assert destination.version == destination_version


@pytest.mark.parametrize(
    ("operation", "destination_factory"),
    [
        (
            MetalToCpuMoveOperation,
            lambda: sw.CPU(1, dtype=sw.DType.Float32),
        ),
        (
            MetalToMetalMoveOperation,
            lambda: sw.Metal(1, dtype=sw.DType.Float32),
        ),
    ],
    ids=["metal-to-cpu", "metal-to-metal"],
)
def test_zero_span_metal_source_copy_still_synchronizes_before_return(
    operation: type[Any], destination_factory: Any
) -> None:
    source = sw.move(
        _cpu_tensor([4.0], dtype=sw.DType.Float32),
        sw.Metal(1, dtype=sw.DType.Float32),
    )
    destination = destination_factory()
    output = sw.Tensor(destination, 0, source.layout)
    internal_source = _internal(source.carrier)
    spy = _RuntimeSpy(internal_source._runtime, source.carrier)
    internal_source._runtime = spy

    operation()._copy(source, destination, output, 0)

    assert spy.calls == 1
    assert not source.carrier.is_released()


@pytest.mark.parametrize(
    ("source_class", "inverse_class"),
    [
        (sw.CPU, MetalToCpuMoveOperation),
        (sw.Metal, CpuToMetalMoveOperation),
    ],
    ids=["cpu-to-metal-backward", "metal-to-cpu-backward"],
)
def test_metal_move_backward_uses_exact_inverse_and_preserves_caller_gradient(
    source_class: type, inverse_class: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    if source_class is sw.CPU:
        source = _cpu_tensor([1.0, 2.0], dtype=sw.DType.Float32)
        moved = sw.move(source, sw.Metal(2, dtype=sw.DType.Float32))
        gradient = sw.move(
            _cpu_tensor([7.0, 8.0], dtype=sw.DType.Float32),
            sw.Metal(2, dtype=sw.DType.Float32),
        )
    else:
        source = sw.move(
            _cpu_tensor([1.0, 2.0], dtype=sw.DType.Float32),
            sw.Metal(2, dtype=sw.DType.Float32),
        )
        moved = sw.move(source, sw.CPU(2, dtype=sw.DType.Float32))
        gradient = _cpu_tensor([7.0, 8.0], dtype=sw.DType.Float32)

    calls = 0
    original_copy = inverse_class._copy

    def spy_copy(
        self: Any,
        tensor: sw.Tensor,
        destination: Any,
        output: sw.Tensor,
        element_count: int,
    ) -> None:
        nonlocal calls
        calls += 1
        original_copy(self, tensor, destination, output, element_count)

    monkeypatch.setattr(inverse_class, "_copy", spy_copy)

    context = moved.autograd_ctx
    assert context is not None
    (source_gradient,) = context.backward(gradient)

    assert calls == 1
    assert type(source_gradient.carrier) is source_class
    assert source_gradient.carrier is not source.carrier
    assert _logical_values(source_gradient) == [7.0, 8.0]
    assert not gradient.carrier.is_released()
    assert _logical_values(gradient) == [7.0, 8.0]


def test_evictable_rejects_missing_exact_metal_edge_before_claiming_tiers() -> None:
    primary = sw.Metal(2, dtype=sw.DType.Float32)
    secondary = sw.FileBacked(dtype=sw.DType.Float32)

    with pytest.raises(RuntimeError, match="exact registered move operation"):
        sw.Evictable(primary, secondary)

    assert not primary.is_owned()
    assert not secondary.is_owned()


def test_evictable_metal_cpu_tiers_use_exact_edges_in_both_directions() -> None:
    primary_tensor = sw.move(
        _cpu_tensor([3.0, 6.0], dtype=sw.DType.Float32),
        sw.Metal(2, dtype=sw.DType.Float32),
    )
    hierarchy = sw.Evictable(
        primary_tensor.carrier,
        sw.CPU(2, dtype=sw.DType.Float32),
    )

    hierarchy.evict()
    assert hierarchy.is_evicted()
    hierarchy.promote()

    assert not hierarchy.is_evicted()
    assert type(hierarchy.primary) is sw.Metal
    assert _physical_values(hierarchy) == [3.0, 6.0]


def test_structural_views_preserve_metal_storage_without_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_compile(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("structural view attempted Metal JIT compilation")

    monkeypatch.setattr(MetalJITCache, "get_or_compile", unexpected_compile)
    tensor = sw.move(
        _cpu_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=sw.DType.Float32,
            layout=sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2])),
        ),
        sw.Metal(6, dtype=sw.DType.Float32),
    )
    storage = _internal(tensor.carrier)._require_storage()

    outputs = [
        tensor[1, :],
        sw.permute(tensor, 1, 0),
        sw.reshape(tensor, sw.Shape([3, 2])),
        sw.rearrange(
            tensor,
            sw.Tree(sw.Node.id(1), sw.Node.id(0)),
            sw.Tree(1, 1),
        ),
        sw.as_strided(tensor, sw.Shape(3), sw.Stride(2)),
    ]

    singleton = sw.move(
        _cpu_tensor(
            [2.0, 5.0, 7.0],
            dtype=sw.DType.Float32,
            layout=sw.Layout(sw.Shape([1, 3]), sw.Stride([1, 1])),
        ),
        sw.Metal(3, dtype=sw.DType.Float32),
    )
    singleton_storage = _internal(singleton.carrier)._require_storage()
    singleton_outputs = [
        sw.broadcast_to(singleton, sw.Shape([4, 3])),
        sw.unsqueeze(singleton, 1),
        sw.squeeze(sw.unsqueeze(singleton, 1), 1),
    ]

    assert all(output.carrier is tensor.carrier for output in outputs)
    assert all(
        _internal(output.carrier)._require_storage() is storage for output in outputs
    )
    assert all(output.carrier is singleton.carrier for output in singleton_outputs)
    assert all(
        _internal(output.carrier)._require_storage() is singleton_storage
        for output in singleton_outputs
    )
    first_add = tensor.carrier.dispatch_op("add")
    second_add = tensor.carrier.dispatch_op("add")
    assert first_add is not second_add
    assert first_add._operation_name == "add"
    assert first_add._dispatch_carrier_class is sw.Metal
