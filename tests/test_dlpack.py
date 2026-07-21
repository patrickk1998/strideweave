from collections.abc import Iterable
from typing import Any

import pytest

import strideweave as sw
from strideweave import CPU, DType, Generic, Layout, Shape, Stride, Tensor

np = pytest.importorskip("numpy")


def make_cpu_carrier(
    values: Iterable[float | int], dtype: DType = DType.Float32
) -> CPU:
    materialized = list(values)
    carrier = CPU(len(materialized), dtype=dtype)
    for index, value in enumerate(materialized):
        carrier[index] = value
    return carrier


def assert_dlpack_array_matches_tensor(array: Any, tensor: Tensor) -> None:
    assert array.shape == tuple(tensor.layout._cache.leaf_shapes)
    for logical_index in range(tensor.size()):
        expanded_key = tuple(tensor.layout._cache.expand_key(logical_index))
        assert array[expanded_key] == pytest.approx(tensor[logical_index])


def test_cpu_dlpack_device_reports_cpu():
    carrier = make_cpu_carrier([1.0, 2.0])
    tensor = Tensor(carrier, 0, Layout(Shape(2), Stride(1)))

    assert tensor.__dlpack_device__() == (1, 0)
    assert carrier.dlpack_info() == {
        "pointer": carrier.pointer(),
        "device_type": 1,
        "device_id": 0,
    }


def test_cpu_float32_dlpack_exports_flat_layout_with_strides():
    tensor = Tensor(
        make_cpu_carrier([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        0,
        Layout(Shape([2, 3]), Stride([1, 2])),
    )

    array = np.from_dlpack(tensor)

    assert array.dtype == np.dtype("float32")
    assert array.strides == (4, 8)
    np.testing.assert_allclose(
        array,
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
    )
    assert_dlpack_array_matches_tensor(array, tensor)


def test_cpu_dlpack_flattens_hierarchical_layout():
    tensor = Tensor(
        make_cpu_carrier([float(index) for index in range(24)]),
        0,
        Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]])),
    )

    array = np.from_dlpack(tensor)

    assert array.shape == (2, 3, 4)
    assert array.strides == (4, 8, 24)
    assert array[1, 2, 3] == pytest.approx(tensor[1, [2, 3]])
    assert_dlpack_array_matches_tensor(array, tensor)


def test_cpu_dlpack_honors_tensor_offset():
    tensor = Tensor(
        make_cpu_carrier([10.0, 20.0, 30.0, 40.0, 50.0]),
        2,
        Layout(Shape(3), Stride(1)),
    )

    array = np.from_dlpack(tensor)

    np.testing.assert_allclose(array, np.array([30.0, 40.0, 50.0], dtype=np.float32))
    assert_dlpack_array_matches_tensor(array, tensor)


def test_cpu_int32_dlpack_exports_int32_storage():
    tensor = Tensor(
        make_cpu_carrier([1, 2, 3, 4], DType.Int32),
        0,
        Layout(Shape([2, 2]), Stride([1, 2])),
    )

    array = np.from_dlpack(tensor)

    assert array.dtype == np.dtype("int32")
    np.testing.assert_array_equal(
        array,
        np.array([[1, 3], [2, 4]], dtype=np.int32),
    )
    assert_dlpack_array_matches_tensor(array, tensor)


def test_generic_dlpack_export_is_not_supported():
    tensor = Tensor(Generic([1.0, 2.0]), 0, Layout(Shape(2), Stride(1)))

    with pytest.raises(BufferError, match="DLPack is not supported"):
        tensor.__dlpack_device__()
    with pytest.raises(BufferError, match="DLPack is not supported"):
        tensor.__dlpack__()
    with pytest.raises(BufferError, match="DLPack is not supported"):
        np.from_dlpack(tensor)


def test_cpu_dlpack_rejects_unsupported_copy_and_device_requests():
    tensor = Tensor(make_cpu_carrier([1.0, 2.0]), 0, Layout(Shape(2), Stride(1)))

    with pytest.raises(BufferError, match="copy exports are not supported"):
        tensor.__dlpack__(copy=True)
    with pytest.raises(BufferError, match="cross-device exports are not supported"):
        tensor.__dlpack__(dl_device=(2, 0))


def test_top_level_tensor_exposes_dlpack_protocol():
    tensor = Tensor(make_cpu_carrier([1.0]), 0, Layout(Shape(1), Stride(1)))

    assert sw.Tensor is Tensor
    assert hasattr(tensor, "__dlpack__")
    assert hasattr(tensor, "__dlpack_device__")
