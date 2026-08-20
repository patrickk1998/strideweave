from __future__ import annotations

from typing import Any

import pytest

import strideweave as sw
import strideweave.carriers.metal._runtime as metal_runtime
import strideweave.carriers.metal.carrier as metal_carrier
from strideweave.carriers.base import Carrier


def _uninitialized_metal() -> sw.Metal:
    carrier = sw.Metal.__new__(sw.Metal)
    Carrier.__init__(carrier)
    return carrier


def test_metal_is_public_closed_typed_and_lazily_importable() -> None:
    from strideweave.carriers import Metal as carrier_export
    from strideweave.carriers.metal import Metal as compatibility_export

    assert carrier_export is sw.Metal
    assert compatibility_export is sw.Metal
    with pytest.raises(TypeError, match="closed carrier implementation"):
        type("MetalSubclass", (sw.Metal,), {})


def test_metal_validates_size_and_dtype_before_loading_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_runtime_load() -> Any:
        raise AssertionError("runtime loaded before argument validation")

    monkeypatch.setattr(metal_carrier, "load_metal_runtime", unexpected_runtime_load)

    with pytest.raises(TypeError):
        sw.Metal(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size must be non-negative"):
        sw.Metal(-1)
    with pytest.raises(TypeError, match="dtype must be a DType"):
        sw.Metal(1, dtype="Float32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Metal dtype must be"):
        sw.Metal(1, dtype=sw.DType.Float64)


def test_unavailable_runtime_does_not_publish_partial_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = _uninitialized_metal()

    def unavailable() -> Any:
        raise RuntimeError(
            "Metal is missing optional dependency 'tilelang'; "
            "install the optional dependency set with "
            "`pip install 'strideweave[metal]'`"
        )

    monkeypatch.setattr(metal_carrier, "load_metal_runtime", unavailable)

    with pytest.raises(
        RuntimeError, match=r"tilelang.*pip install 'strideweave\[metal\]'"
    ):
        sw.Metal.__init__(carrier, 2)

    assert not hasattr(carrier, "_storage")


def test_metal_storage_support_is_exact_without_loading_runtime() -> None:
    carrier = _uninitialized_metal()
    accepted = (sw.DType.Float32, sw.DType.Int32, sw.DType.Bool)

    for dtype in sw.DType.registered():
        expected = any(dtype is candidate for candidate in accepted)
        assert carrier.supports_storage_dtype(dtype) is expected, dtype.name

    with pytest.raises(TypeError, match="storage dtype must be a DType"):
        carrier.supports_storage_dtype("Float32")  # type: ignore[arg-type]


def test_metal_rejects_every_registered_unsupported_dtype_before_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = (sw.DType.Float32, sw.DType.Int32, sw.DType.Bool)

    def unexpected_runtime_load() -> Any:
        raise AssertionError("runtime loaded for an unsupported dtype")

    monkeypatch.setattr(metal_carrier, "load_metal_runtime", unexpected_runtime_load)

    for dtype in sw.DType.registered():
        if any(dtype is candidate for candidate in accepted):
            continue
        with pytest.raises(ValueError, match=r"^Metal "):
            sw.Metal(1, dtype=dtype)


@pytest.mark.metal
@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (sw.DType.Float32, [1.25, -2.5, 0.5], [1.25, -2.5, 0.5]),
        (sw.DType.Int32, [1, -2, 3], [1, -2, 3]),
        (sw.DType.Bool, [True, False, True], [True, False, True]),
    ],
    ids=["float32", "int32", "bool"],
)
def test_metal_constructs_typed_private_storage_and_mutates_once(
    dtype: sw.DType, values: list[Any], expected: list[Any]
) -> None:
    carrier = sw.Metal(len(values), dtype=dtype)

    assert carrier.size() == len(values)
    assert carrier.dtype() is dtype
    assert carrier.is_mutable()
    assert carrier.version == 0
    storage = getattr(carrier, "_storage")
    assert storage.ndim == 1
    assert storage.device.type == "mps"
    assert not hasattr(carrier, "tensor")
    assert [carrier[index] for index in range(carrier.size())] == [
        False if dtype is sw.DType.Bool else 0 for _ in values
    ]

    carrier.set_value(0, values[0])
    assert carrier.version == 1
    carrier[1] = values[1]
    assert carrier.version == 2
    carrier[2] = values[2]
    assert carrier.version == 3
    assert [carrier[index] for index in range(carrier.size())] == expected

    with pytest.raises(BufferError, match="DLPack is not supported"):
        carrier.dlpack_info()


@pytest.mark.metal
def test_metal_scalar_mutation_publishes_only_after_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = sw.Metal(2, dtype=sw.DType.Float32)
    carrier.set_value(0, 2.0)
    layout = sw.Layout(sw.Shape(2), sw.Stride(1))
    alias = sw.Tensor(carrier, 0, layout)
    original_storage = getattr(carrier, "_storage")
    original_version = carrier.version
    synchronize = metal_runtime.MetalRuntime.synchronize

    def fail_synchronization(_runtime: metal_runtime.MetalRuntime) -> None:
        raise RuntimeError("injected MPS synchronization failure")

    monkeypatch.setattr(metal_runtime.MetalRuntime, "synchronize", fail_synchronization)

    with pytest.raises(RuntimeError, match="injected MPS synchronization failure"):
        carrier[0] = 7.0

    assert getattr(carrier, "_storage") is original_storage
    assert carrier.version == original_version

    monkeypatch.setattr(metal_runtime.MetalRuntime, "synchronize", synchronize)
    assert carrier.get_value(0) == 2.0
    assert alias[0] == 2.0

    carrier[0] = 5.0
    assert carrier.version == original_version + 1
    assert carrier.get_value(0) == 5.0
    assert alias[0] == 5.0


@pytest.mark.metal
def test_metal_factories_are_fresh_typed_and_work_after_release() -> None:
    source = sw.Metal(2, dtype=sw.DType.Float32)
    initialized = source.new_like([1, None, 3], mutable=False)
    overridden = source.allocate_like(2, dtype=sw.DType.Bool)
    empty = source.allocate_like(2, dtype=sw.DType.Int32, empty=True)

    assert type(initialized) is sw.Metal
    assert initialized is not source
    assert initialized.dtype() is sw.DType.Float32
    assert [initialized[index] for index in range(3)] == [1.0, 0.0, 3.0]
    assert not initialized.is_mutable()
    assert initialized.version == 0
    assert overridden.dtype() is sw.DType.Bool
    assert [overridden[index] for index in range(2)] == [False, False]

    empty[0] = 7
    empty[1] = -3
    assert [empty[index] for index in range(2)] == [7, -3]

    source.release()
    assert source.is_released()
    assert source.size() == 0
    assert source.supports_storage_dtype(sw.DType.Int32)
    with pytest.raises(RuntimeError, match="Carrier is released"):
        source[0]

    after_release = source.new_like([True, None], dtype=sw.DType.Bool)
    assert [after_release[index] for index in range(2)] == [True, False]


@pytest.mark.metal
def test_metal_bounds_normalization_mutability_and_ownership() -> None:
    integer = sw.Metal(2, dtype=sw.DType.Int32)
    version = integer.version

    for index in (-1, 2):
        with pytest.raises(IndexError, match="Carrier index out of range"):
            integer.get_value(index)
        with pytest.raises(IndexError, match="Carrier index out of range"):
            integer.set_value(index, 1)
    with pytest.raises(TypeError):
        integer.get_value(0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an integer"):
        integer.set_value(0, True)
    with pytest.raises(OverflowError, match="out of int32 range"):
        integer.set_value(0, 2**31)
    assert integer.version == version

    immutable = sw.Metal(1, mutable=False)
    with pytest.raises(RuntimeError, match="Carrier is not mutable"):
        immutable[0] = 1.0
    assert immutable.version == 0

    token = integer._claim_ownership()
    assert integer.is_owned()
    assert not integer.is_mutable()
    assert integer.supports_storage_dtype(sw.DType.Bool)
    with pytest.raises(RuntimeError, match="Carrier is not mutable"):
        integer[0] = 1
    integer._begin_owner_access(token)
    try:
        integer[0] = 1
    finally:
        integer._end_owner_access(token)
    assert integer.version == version + 1
    integer._relinquish_ownership(token)
    assert integer.is_mutable()


@pytest.mark.metal
def test_metal_carrier_scatter_is_synchronized_versioned_and_atomic() -> None:
    source = sw.Tensor(
        sw.Generic([3.0, 4.0], dtype=sw.DType.Float32),
        0,
        sw.Layout(sw.Shape(2), sw.Stride(1)),
    )
    destination_carrier = sw.Metal(4, dtype=sw.DType.Float32)
    destination = sw.Tensor(
        destination_carrier,
        0,
        sw.Layout(sw.Shape(4), sw.Stride(1)),
    )
    mapping = sw.Layout(sw.Shape(2), sw.Stride(1))

    destination_carrier.scatter(source, destination, mapping, mapping_offset=1)

    assert [destination_carrier[index] for index in range(4)] == [0.0, 3.0, 4.0, 0.0]
    assert destination_carrier.version == 1

    before = [destination_carrier[index] for index in range(4)]
    version = destination_carrier.version
    with pytest.raises(IndexError, match="Carrier index out of range"):
        destination_carrier.scatter(source, destination, mapping, mapping_offset=3)
    assert [destination_carrier[index] for index in range(4)] == before
    assert destination_carrier.version == version

    invalid_source = sw.Tensor(
        sw.Generic(["not-a-float", 2.0], dtype=sw.DType.Any),
        0,
        sw.Layout(sw.Shape(2), sw.Stride(1)),
    )
    with pytest.raises(TypeError, match="real number"):
        destination_carrier.scatter(invalid_source, destination, mapping)
    assert [destination_carrier[index] for index in range(4)] == before
    assert destination_carrier.version == version
