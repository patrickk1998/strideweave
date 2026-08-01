from __future__ import annotations

import pytest

import strideweave as sw
from strideweave.carriers.generic.as_strided_ops import GenericAsStridedOperation
from strideweave.core._representation import Subtensor, TensorRepresentation


def _tensor(values: list[float], shape: sw.Shape, stride: sw.Stride) -> sw.Tensor:
    return sw.Tensor(
        sw.Generic(values, dtype=sw.DType.Floating),
        0,
        sw.Layout(shape, stride),
    )


def _values(tensor: sw.Tensor) -> list[float]:
    return [tensor[index] for index in range(tensor.size())]


def test_as_strided_is_an_origin_based_zero_copy_view():
    source = _tensor([10.0, 20.0, 30.0, 40.0], sw.Shape(4), sw.Stride(1))

    view = GenericAsStridedOperation().forward(
        source,
        sw.Shape([2, 2]),
        sw.Stride([1, 2]),
    )

    assert view.carrier is source.carrier
    assert view.offset == source.offset
    assert view.layout == sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    assert _values(view) == [10.0, 20.0, 30.0, 40.0]


def test_as_strided_backward_scatter_leaves_unselected_ordinals_zero():
    source = _tensor([10.0, 20.0, 30.0, 40.0], sw.Shape(4), sw.Stride(1))
    view = GenericAsStridedOperation().forward(
        source,
        sw.Shape(2),
        sw.Stride(2),
    )
    gradient = _tensor([5.0, 0.0, 7.0], sw.Shape(2), sw.Stride(2))

    (source_gradient,) = view.autograd_ctx.backward(gradient)

    assert source_gradient.layout == source.layout
    assert _values(source_gradient) == [5.0, 0.0, 7.0, 0.0]


def test_as_strided_supports_hierarchical_mapping_shapes():
    source = _tensor(
        [float(index) for index in range(8)],
        sw.Shape([2, [2, 2]]),
        sw.Stride([1, [2, 4]]),
    )

    view = GenericAsStridedOperation().forward(
        source,
        sw.Shape([[2, 2], 2]),
        sw.Stride([[1, 2], 4]),
    )

    assert view.layout == sw.Layout(sw.Shape([[2, 2], 2]), sw.Stride([[1, 2], 4]))
    assert _values(view) == [float(index) for index in range(8)]


@pytest.mark.parametrize("backend", ["generic", "cpu"])
def test_as_strided_composes_a_mapping_inside_a_noncanonical_leading_mode(
    backend: str,
) -> None:
    layout = sw.Layout(sw.Shape([5, 4]), sw.Stride([4, 1]))
    if backend == "generic":
        carrier = sw.Generic([float(index) for index in range(layout.cosize)])
    else:
        carrier = sw.CPU(layout.cosize)
        for index in range(layout.cosize):
            carrier[index] = float(index)
    source = sw.Tensor(carrier, 0, layout)

    view = sw.as_strided(source, sw.Shape([2, 2]), sw.Stride([1, 2]))

    assert view.layout == sw.Layout(sw.Shape([2, 2]), sw.Stride([4, 8]))
    assert _values(view) == [0.0, 4.0, 8.0, 12.0]


@pytest.mark.parametrize("backend", ["generic", "cpu"])
def test_as_strided_large_logical_extent_does_not_enumerate_coordinates(
    backend: str,
) -> None:
    extent = 10**9
    carrier = sw.Generic([1.0]) if backend == "generic" else sw.CPU(1)
    source = sw.Tensor(
        carrier,
        0,
        sw.Layout(sw.Shape(extent), sw.Stride(0)),
    )

    with pytest.raises(ValueError, match="composed placement must be injective"):
        sw.as_strided(source, sw.Shape(extent), sw.Stride(1))


def test_as_strided_composes_only_c0_layouts_in_a_multi_subtensor_representation():
    class Planar(sw.CompoundDType, abstract=False):
        __slots__ = ()

        def __init__(self) -> None:
            super().__init__(
                "AsStridedTestPlanar",
                supertype=sw.DType.Any,
                simple_types=(sw.DType.Float32, sw.DType.Float32),
            )

    dtype = Planar()
    first = sw.Generic([1.0] * 4, dtype=sw.DType.Float32)
    second = sw.Generic([1.0] * 2, dtype=sw.DType.Float32)
    first_layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    second_layout = sw.Layout(sw.Shape(2), sw.Stride(1))
    adjacent = sw.Layout(sw.Shape([2, 2]), sw.Stride([0, 1]))
    source = sw.Tensor._from_representation(
        TensorRepresentation(
            dtype,
            (
                Subtensor(sw.DType.Float32, first, 0, first_layout),
                Subtensor(sw.DType.Float32, second, 0, second_layout),
            ),
            (adjacent,),
        )
    )

    view = sw.as_strided(source, sw.Shape([2, 2]), sw.Stride([2, 1]))

    assert view._representation.subtensors[0].layout == sw.Layout(
        sw.Shape([2, 2]), sw.Stride([2, 1])
    )
    assert view._representation.subtensors[1].layout == second_layout
    assert view._representation.adjacent_layouts[0] == sw.Layout(
        sw.Shape([2, 2]), sw.Stride([1, 0])
    )
    assert view._representation.subtensors[0].carrier is first
    assert view._representation.subtensors[1].carrier is second


@pytest.mark.parametrize(
    ("shape", "stride", "message"),
    [
        (sw.Shape([2, 2]), sw.Stride([0, 1]), "mapping must be injective"),
        (sw.Shape(2), sw.Stride(4), "exceeds the input logical coordinate domain"),
    ],
)
def test_as_strided_rejects_invalid_mapping_contract(
    shape: sw.Shape, stride: sw.Stride, message: str
):
    source = _tensor([1.0, 2.0, 3.0, 4.0], sw.Shape(4), sw.Stride(1))

    with pytest.raises(ValueError, match=message):
        GenericAsStridedOperation().forward(source, shape, stride)


def test_as_strided_rejects_a_composed_placement_alias():
    source = _tensor([1.0, 2.0, 3.0, 4.0], sw.Shape([2, 2]), sw.Stride([0, 1]))

    with pytest.raises(ValueError, match="composed placement must be injective"):
        GenericAsStridedOperation().forward(source, sw.Shape(2), sw.Stride(1))
