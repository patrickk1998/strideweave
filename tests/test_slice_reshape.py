from __future__ import annotations

from collections.abc import Iterable

import pytest

import strideweave as sw
from strideweave.carriers.shared_ops import ReshapeOperation
from strideweave.core._representation import Subtensor, TensorRepresentation


def _tensor_with_storage(
    values: Iterable[float], layout: sw.Layout, backend: str
) -> sw.Tensor:
    materialized = list(values)
    if backend == "generic":
        carrier = sw.Generic(materialized, dtype=sw.DType.Float32)
    else:
        carrier = sw.CPU(len(materialized), dtype=sw.DType.Float32)
        for index, value in enumerate(materialized):
            carrier[index] = value
    return sw.Tensor(carrier, 0, layout)


def _logical_storage(values: Iterable[float], layout: sw.Layout) -> list[float]:
    storage = [0.0] * layout.cosize
    for logical_index, value in enumerate(values):
        storage[layout.index(logical_index)] = value
    return storage


def _logical_values(tensor: sw.Tensor) -> list[float]:
    return [tensor[index] for index in range(tensor.size())]


@pytest.mark.parametrize("backend", ["generic", "cpu"])
def test_positive_step_slice_is_a_zero_copy_view(backend: str) -> None:
    source_layout = sw.Layout(sw.Shape([5, 10]), sw.Stride([1, 5]))
    source = _tensor_with_storage(range(source_layout.cosize), source_layout, backend)

    view = source[2, 1:9:2]

    assert view.carrier is source.carrier
    assert view.offset == source.offset + source_layout.index((2, 1))
    assert view.layout == sw.Layout(sw.Shape(4), sw.Stride(10))
    assert _logical_values(view) == [source[2, column] for column in (1, 3, 5, 7)]


@pytest.mark.parametrize("backend", ["generic", "cpu"])
def test_positive_step_slice_backward_scatters_omitted_coordinates(
    backend: str,
) -> None:
    source_layout = sw.Layout(sw.Shape([5, 10]), sw.Stride([1, 5]))
    source = _tensor_with_storage(range(source_layout.cosize), source_layout, backend)
    view = source[2, 1:9:2]
    gradient = _tensor_with_storage(
        _logical_storage([10.0, 20.0, 30.0, 40.0], view.layout), view.layout, backend
    )

    view.backward(gradient)

    assert source.grad is not None
    assert source.grad.layout == source_layout
    expected = [0.0] * source.size()
    for value, column in zip((10.0, 20.0, 30.0, 40.0), (1, 3, 5, 7), strict=True):
        expected[source_layout.index((2, column))] = value
    assert _logical_values(source.grad) == expected


@pytest.mark.parametrize("backend", ["generic", "cpu"])
def test_reshape_preserves_holes_and_hierarchical_first_mode_fast_order(
    backend: str,
) -> None:
    source_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 4]))
    source = _tensor_with_storage(
        _logical_storage([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], source_layout),
        source_layout,
        backend,
    )

    result = ReshapeOperation().forward(source, sw.Shape([3, [2]]))

    assert result.carrier is source.carrier
    assert result.offset == source.offset
    assert result.layout == sw.Layout(sw.Shape([3, [2]]), sw.Stride([2, [6]]))
    assert _logical_values(result) == _logical_values(source)


def test_reshape_backward_restores_the_exact_source_layout() -> None:
    source_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 4]))
    source = _tensor_with_storage(
        _logical_storage([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], source_layout),
        source_layout,
        "generic",
    )
    result = ReshapeOperation().forward(source, sw.Shape([3, 2]))
    gradient = _tensor_with_storage(
        _logical_storage([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], result.layout),
        result.layout,
        "generic",
    )

    result.backward(gradient)

    assert source.grad is not None
    assert source.grad.layout == source_layout
    assert _logical_values(source.grad) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_reshape_rejects_implicit_copies_and_size_mismatches() -> None:
    transposed_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 3]))
    transposed = sw.Tensor(
        sw.Generic([0.0] * transposed_layout.cosize, dtype=sw.DType.Float32),
        0,
        transposed_layout,
    )
    with pytest.raises(ValueError, match="coalesces to one logical leaf"):
        ReshapeOperation().forward(transposed, sw.Shape([3, 2]))

    source = sw.Tensor(
        sw.Generic([0.0] * 6, dtype=sw.DType.Float32),
        0,
        sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2])),
    )
    with pytest.raises(ValueError, match="same logical size"):
        ReshapeOperation().forward(source, sw.Shape([5]))
    with pytest.raises(ValueError, match="must not be less than 1"):
        ReshapeOperation().forward(source, sw.Shape([-1, 6]))


class _SliceReshapeDType(sw.CompoundDType, abstract=False):
    """Two-plane dtype used to verify c0-only reshape transformations."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            "SliceReshapeTestDType",
            supertype=sw.DType.Any,
            simple_types=(sw.DType.Float32, sw.DType.Float32),
        )


def test_reshape_transforms_multi_subtensor_l0_and_s0_only() -> None:
    dtype = _SliceReshapeDType()
    first = sw.Generic([1.0] * 4, dtype=sw.DType.Float32)
    second = sw.Generic([0.0] * 4, dtype=sw.DType.Float32)
    l0 = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    l1 = sw.Layout(sw.Shape(4), sw.Stride(1))
    s0 = sw.Layout(sw.Shape([2, 2]), sw.Stride([0, 0]))
    tensor = sw.Tensor._from_representation(
        TensorRepresentation(
            dtype,
            (
                Subtensor(sw.DType.Float32, first, 0, l0),
                Subtensor(sw.DType.Float32, second, 0, l1),
            ),
            (s0,),
        )
    )

    result = sw.reshape(tensor, sw.Shape([4]))

    assert result._representation.subtensors[0].layout == sw.Layout(
        sw.Shape(4), sw.Stride(1)
    )
    assert result._representation.subtensors[1].layout == l1
    assert result._representation.adjacent_layouts[0] == sw.Layout(
        sw.Shape(4), sw.Stride(0)
    )
