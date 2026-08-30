"""Focused tests for direct operation dispatch on ``TiledEvictable`` tensors."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.definition_dispatch import execute_definition_invocation
from strideweave.operation_definition import _operation_definition

GRID = (2, 2)
TILE = (2, 2)
SHAPE = (4, 4)
SIZE = 16
DIRECT_NAMES = {"add", "elementwise_mul", "mul", "relu", "reduce_sum", "matmul"}


def _layout(shape: tuple[int, ...] = SHAPE) -> sw.Layout:
    strides: list[int] = []
    scale = 1
    for extent in shape:
        strides.append(scale)
        scale *= extent
    return sw.Layout(sw.Shape(shape), sw.Stride(tuple(strides)))


def _within_tile_permutation() -> sw.Layout:
    return sw.Layout(
        sw.Shape([[2, 2], [2, 2]]),
        sw.Stride([[4, 2], [1, 8]]),
    )


def _incompatible_layout(kind: str) -> sw.Layout:
    if kind == "spans":
        return sw.Layout(
            sw.Shape([[2, 2], [2, 2]]),
            sw.Stride([[1, 2], [8, 4]]),
        )
    if kind == "remaps":
        return sw.Layout(sw.Shape(SHAPE), sw.Stride((4, 1)))
    if kind == "aliases":
        return sw.Layout(sw.Shape(SHAPE), sw.Stride((1, 0)))
    assert kind == "omits"
    return sw.Layout(
        sw.Shape([[2, 2], [2, 2]]),
        sw.Stride([[1, 8], [2, 4]]),
    )


def _tiled(
    values: list[float],
    *,
    residency_policy: sw.ResidencyPolicy | None = None,
) -> sw.TiledEvictable:
    assert len(values) == SIZE
    return sw.TiledEvictable(
        sw.Generic(values, dtype=sw.DType.Float32),
        sw.Generic([], dtype=sw.DType.Float32),
        GRID,
        TILE,
        residency_policy=residency_policy,
    )


def _tensor(carrier: sw.TiledEvictable) -> sw.Tensor:
    return sw.Tensor(carrier, 0, _layout())


def _dense(values: list[float], shape: tuple[int, ...] = SHAPE) -> sw.Tensor:
    return sw.Tensor(
        sw.Generic(values, dtype=sw.DType.Float32),
        0,
        _layout(shape),
    )


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _partially_resident(values: list[float]) -> tuple[sw.TiledEvictable, sw.Tensor]:
    carrier = _tiled(values)
    carrier.evict(sw.TileSet(((1, 0),)))
    return carrier, _tensor(carrier)


def test_tiled_advertises_exact_direct_operation_boundary_and_fresh_metadata() -> None:
    carrier = _tiled([float(index) for index in range(SIZE)])

    assert {
        entry.operation for entry in carrier.operation_capabilities()
    } == DIRECT_NAMES
    assert {
        tuple(operand.role.value for operand in entry.operands)
        for entry in carrier.operation_capabilities("mul")
    } >= {
        ("tensor", "tensor"),
        ("tensor", "weak_scalar"),
        ("weak_scalar", "tensor"),
    }
    for name in sorted(DIRECT_NAMES):
        first = carrier.dispatch_op(name)
        second = carrier.dispatch_op(name)
        assert first is not second
        assert first._operation_name == name
        assert first._dispatch_carrier_class is sw.TiledEvictable

    with pytest.raises(NotImplementedError):
        carrier.dispatch_op("div")


@pytest.mark.parametrize(
    ("name", "invoke", "expected_shape", "expected"),
    [
        (
            "add",
            lambda tensor: sw.add(tensor, tensor),
            SHAPE,
            [float(2 * index) for index in range(SIZE)],
        ),
        (
            "elementwise_mul",
            lambda tensor: sw.elementwise_mul(tensor, tensor),
            SHAPE,
            [float(index * index) for index in range(SIZE)],
        ),
        (
            "mul",
            lambda tensor: sw.mul(tensor, 3.0),
            SHAPE,
            [float(3 * index) for index in range(SIZE)],
        ),
        (
            "relu",
            lambda tensor: sw.relu(tensor),
            SHAPE,
            [float(index) for index in range(SIZE)],
        ),
    ],
)
def test_direct_tiled_operations_consume_all_tiles_and_return_dense_results(
    name: str,
    invoke: Callable[[sw.Tensor], sw.Tensor],
    expected_shape: tuple[int, ...],
    expected: list[float],
) -> None:
    del name
    values = [float(index) for index in range(SIZE)]
    carrier, tensor = _partially_resident(values)
    source_version = carrier.version
    source_values = _values(tensor)

    result = invoke(tensor)

    assert isinstance(result, sw.Tensor)
    assert type(result.carrier) is sw.Generic
    assert result.carrier is not tensor.carrier
    assert result.offset == 0
    assert result.carrier.size() == result.layout.cosize
    assert result.dtype() is sw.DType.Float32
    assert result.layout.shape == sw.Shape(expected_shape)
    assert _values(result) == expected
    assert carrier.version == source_version
    assert _values(tensor) == source_values


def test_direct_operation_and_vjp_preserve_within_tile_permutation() -> None:
    physical_values = [float(index - 8) for index in range(SIZE)]
    layout = _within_tile_permutation()
    carrier = _tiled(physical_values)
    carrier.evict(sw.TileSet(((1, 0),)))
    tensor = sw.Tensor(carrier, 0, layout)
    dense = sw.Tensor(
        sw.Generic(physical_values, dtype=sw.DType.Float32),
        0,
        layout,
    )

    result = sw.relu(tensor)
    reference = sw.relu(dense)

    assert result.layout == reference.layout
    assert _values(result) == _values(reference)
    cotangent_values = [float(index + 1) for index in range(SIZE)]
    result.backward(
        sw.Tensor(
            sw.Generic(cotangent_values, dtype=sw.DType.Float32),
            0,
            result.layout,
        )
    )
    reference.backward(
        sw.Tensor(
            sw.Generic(cotangent_values, dtype=sw.DType.Float32),
            0,
            reference.layout,
        )
    )

    assert tensor.grad is not None
    assert dense.grad is not None
    assert tensor.grad.layout == tensor.layout
    assert _values(tensor.grad) == _values(dense.grad)


@pytest.mark.parametrize("kind", ["spans", "remaps", "aliases", "omits"])
def test_direct_operation_rejects_incompatible_tile_footprints_before_work(
    kind: str,
) -> None:
    carrier = _tiled([float(index) for index in range(SIZE)])
    tensor = sw.Tensor(carrier, 0, _incompatible_layout(kind))
    before = (
        carrier.version,
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        tuple(carrier[index] for index in range(SIZE)),
    )

    with pytest.raises(ValueError, match="tile footprints"):
        sw.relu(tensor)

    assert (
        carrier.version,
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        tuple(carrier[index] for index in range(SIZE)),
    ) == before


def test_released_state_precedes_dispatch_and_saved_footprint_failures() -> None:
    values = [float(index) for index in range(SIZE)]
    released = _tiled(values)
    incompatible = sw.Tensor(released, 0, _incompatible_layout("remaps"))
    capabilities = released.operation_capabilities()
    raw_operations = {
        "add": (released.dispatch_op("add"), (incompatible, incompatible)),
        "elementwise_mul": (
            released.dispatch_op("elementwise_mul"),
            (incompatible, incompatible),
        ),
        "mul": (released.dispatch_op("mul"), (incompatible, incompatible)),
        "relu": (released.dispatch_op("relu"), (incompatible,)),
        "reduce_sum": (released.dispatch_op("reduce_sum"), (incompatible,)),
        "matmul": (released.dispatch_op("matmul"), (incompatible, incompatible)),
    }
    resolved = _operation_definition("relu").resolve_call(incompatible)

    released.release()

    assert released.operation_capabilities() == capabilities
    for name in sorted(DIRECT_NAMES | {"div", "not_a_registered_operation"}):
        with pytest.raises(RuntimeError, match=r"^TiledEvictable is released$"):
            released.dispatch_op(name)
    for operation, operands in raw_operations.values():
        with pytest.raises(RuntimeError, match="carrier is released"):
            operation.forward(*operands)
    with pytest.raises(RuntimeError, match=r"^TiledEvictable is released$"):
        execute_definition_invocation(released, resolved)

    live = _tiled(values)
    live_incompatible = sw.Tensor(live, 0, _incompatible_layout("remaps"))
    before = (
        live.version,
        live.primary_tiles,
        live.secondary_tiles,
        live.valid_tiles,
        tuple(live[index] for index in range(SIZE)),
    )

    with pytest.raises(ValueError, match="tile footprints"):
        sw.relu(live_incompatible)

    assert (
        live.version,
        live.primary_tiles,
        live.secondary_tiles,
        live.valid_tiles,
        tuple(live[index] for index in range(SIZE)),
    ) == before


def test_direct_tiled_reduce_sum_preserves_shape_layout_and_dtype() -> None:
    values = [float(index) for index in range(SIZE)]
    _carrier, tensor = _partially_resident(values)

    result = sw.reduce_sum(tensor, "a b -> a")

    assert type(result.carrier) is sw.Generic
    assert result.dtype() is sw.DType.Float32
    assert result.layout.shape == sw.Shape((4,))
    assert result.layout.is_injective is True
    assert _values(result) == [24.0, 28.0, 32.0, 36.0]
    assert result.autograd_ctx is not None
    assert result.autograd_ctx._operation_name == "reduce_sum"
    assert result.autograd_ctx.inputs() == (tensor,)


def test_direct_tiled_matmul_preserves_shape_layout_and_dtype() -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    rhs_values = [float(2 * (index + 1)) for index in range(SIZE)]
    lhs_carrier, lhs = _partially_resident(lhs_values)
    rhs_carrier, rhs = _partially_resident(rhs_values)

    result = lhs_carrier.dispatch_op("matmul").forward(lhs, rhs)

    expected = []
    for flat in range(SIZE):
        row = flat % 4
        column = flat // 4
        expected.append(
            sum(
                lhs_values[row + 4 * inner] * rhs_values[column + 4 * inner]
                for inner in range(4)
            )
        )
    assert rhs_carrier is rhs.carrier
    assert type(result.carrier) is sw.Generic
    assert result.dtype() is sw.DType.Float32
    assert result.layout.shape == sw.Shape(SHAPE)
    assert _values(result) == expected


def test_direct_tiled_mixed_dtype_plan_matches_central_semantics() -> None:
    lhs = _tensor(_tiled([float(index) for index in range(SIZE)]))
    rhs_carrier = sw.TiledEvictable(
        sw.Generic(list(range(SIZE)), dtype=sw.DType.Int32),
        sw.Generic([], dtype=sw.DType.Int32),
        GRID,
        TILE,
    )
    rhs = sw.Tensor(rhs_carrier, 0, _layout())
    rhs_carrier.evict(sw.TileSet(((1, 0),)))

    result = sw.add(lhs, rhs)

    assert type(result.carrier) is sw.Generic
    assert result.dtype() is sw.DType.Float32
    assert _values(result) == [float(2 * index) for index in range(SIZE)]


def test_direct_tiled_broadcast_vjp_matches_dense_reference() -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    rhs_values = [2.0, 3.0, 4.0, 5.0]
    lhs_carrier, lhs = _partially_resident(lhs_values)
    rhs_carrier = sw.TiledEvictable(
        sw.Generic(rhs_values, dtype=sw.DType.Float32),
        sw.Generic([], dtype=sw.DType.Float32),
        (1, 2),
        (1, 2),
    )
    rhs = sw.Tensor(rhs_carrier, 0, _layout((1, 4)))
    rhs_carrier.evict(sw.TileSet(((0, 1),)))

    result = sw.elementwise_mul(lhs, rhs)
    result.backward(_dense([1.0] * SIZE))

    expected = [lhs_values[index] * rhs_values[index // 4] for index in range(SIZE)]
    assert _values(result) == expected
    assert lhs.grad is not None
    assert rhs.grad is not None
    assert isinstance(lhs.grad.carrier, sw.TiledEvictable)
    assert isinstance(rhs.grad.carrier, sw.TiledEvictable)
    assert _values(lhs.grad) == [rhs_values[index // 4] for index in range(SIZE)]
    assert _values(rhs.grad) == [
        sum(lhs_values[column * 4 : (column + 1) * 4]) for column in range(4)
    ]
    assert lhs_carrier is lhs.carrier


def test_direct_matmul_vjp_matches_dense_reference_with_tiled_gradients() -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    rhs_values = [float(index + 2) for index in range(SIZE)]
    _lhs_carrier, lhs = _partially_resident(lhs_values)
    _rhs_carrier, rhs = _partially_resident(rhs_values)
    dense_lhs = _dense(lhs_values)
    dense_rhs = _dense(rhs_values)

    result = sw.matmul(lhs, rhs)
    reference = sw.matmul(dense_lhs, dense_rhs)
    cotangent = _dense([1.0] * SIZE)
    result.backward(cotangent)
    reference.backward(_dense([1.0] * SIZE))

    assert lhs.grad is not None
    assert rhs.grad is not None
    assert dense_lhs.grad is not None
    assert dense_rhs.grad is not None
    assert isinstance(lhs.grad.carrier, sw.TiledEvictable)
    assert isinstance(rhs.grad.carrier, sw.TiledEvictable)
    assert _values(lhs.grad) == _values(dense_lhs.grad)
    assert _values(rhs.grad) == _values(dense_rhs.grad)


@pytest.mark.parametrize(
    "direction", ["tensor_scalar", "scalar_tensor", "tensor_tensor"]
)
def test_mul_dispatch_supports_all_declared_operand_roles(direction: str) -> None:
    lhs_carrier, lhs = _partially_resident([float(index + 1) for index in range(SIZE)])
    rhs_carrier, rhs = _partially_resident([float(index + 2) for index in range(SIZE)])

    if direction == "tensor_scalar":
        result = sw.mul(lhs, 2.0)
        expected = [float(2 * (index + 1)) for index in range(SIZE)]
    elif direction == "scalar_tensor":
        result = sw.mul(2.0, lhs)
        expected = [float(2 * (index + 1)) for index in range(SIZE)]
    else:
        result = sw.mul(lhs, rhs)
        expected = [float((index + 1) * (index + 2)) for index in range(SIZE)]

    assert lhs_carrier is lhs.carrier
    assert rhs_carrier is rhs.carrier
    assert result.dtype() is sw.DType.Float32
    assert result.layout.shape == sw.Shape(SHAPE)
    assert _values(result) == expected
    if direction == "scalar_tensor":
        operation = result.autograd_ctx
        assert operation is not None
        invocation = operation._resolved_invocation
        assert tuple(operand.role.value for operand in invocation.plan.operands) == (
            "weak_scalar",
            "tensor",
        )


def test_raw_tiled_unsupported_operation_is_rejected() -> None:
    _carrier, tensor = _partially_resident([float(index) for index in range(SIZE)])

    with pytest.raises((NotImplementedError, TypeError)):
        sw.div(tensor, tensor)
    with pytest.raises((NotImplementedError, TypeError)):
        sw.exp(tensor)


def test_pending_residency_handle_is_not_accepted_as_a_tensor_operand() -> None:
    carrier = _tiled([float(index) for index in range(SIZE)])
    carrier.evict(sw.TileSet(((0, 0),)))
    pending = carrier.promote_async(sw.TileSet(((0, 0),)))

    with pytest.raises(TypeError):
        sw.relu(pending)  # type: ignore[arg-type]
    pending.wait()


def test_projected_compact_tensor_uses_the_ordinary_compute_operation_surface() -> None:
    carrier = _tiled([float(index + 1) for index in range(SIZE)])
    source = _tensor(carrier)
    projected = carrier.project(source, sw.TileSelection(([1], [0]))).wait()

    assert type(projected.carrier) is sw.Generic
    result = sw.exp(projected)

    assert type(result.carrier) is sw.Generic
    assert result.layout.shape == projected.layout.shape
    assert _values(result) == pytest.approx(
        [math.exp(value) for value in _values(projected)]
    )


@pytest.mark.parametrize("operation", ["add", "elementwise_mul", "mul"])
def test_direct_pointwise_vjp_returns_full_shaped_tiled_gradients(
    operation: str,
) -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    rhs_values = [float(index + 2) for index in range(SIZE)]
    lhs_carrier, lhs = _partially_resident(lhs_values)
    rhs_carrier, rhs = _partially_resident(rhs_values)

    if operation == "add":
        result = sw.add(lhs, rhs)
    elif operation == "elementwise_mul":
        result = sw.elementwise_mul(lhs, rhs)
    else:
        result = sw.mul(lhs, rhs)
    result.backward(_dense([1.0] * SIZE))

    assert lhs.grad is not None
    assert rhs.grad is not None
    assert isinstance(lhs.grad.carrier, sw.TiledEvictable)
    assert isinstance(rhs.grad.carrier, sw.TiledEvictable)
    assert lhs.grad.layout.shape == sw.Shape(SHAPE)
    assert rhs.grad.layout.shape == sw.Shape(SHAPE)
    assert lhs.grad.carrier.valid_tiles == sw.TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert rhs.grad.carrier.valid_tiles == sw.TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    if operation == "add":
        assert _values(lhs.grad) == [1.0] * SIZE
        assert _values(rhs.grad) == [1.0] * SIZE
    elif operation == "elementwise_mul":
        assert _values(lhs.grad) == rhs_values
        assert _values(rhs.grad) == lhs_values
    else:
        assert _values(lhs.grad) == rhs_values
        assert _values(rhs.grad) == lhs_values
    assert lhs_carrier is lhs.carrier
    assert rhs_carrier is rhs.carrier


def test_direct_mul_scalar_vjp_accumulates() -> None:
    values = [float(index + 1) for index in range(SIZE)]
    carrier, tensor = _partially_resident(values)
    result = sw.mul(tensor, 3.0)

    result.backward(_dense([1.0] * SIZE), retain_graph=True)
    result.backward(_dense([1.0] * SIZE), retain_graph=True)
    assert tensor.grad is not None
    assert isinstance(tensor.grad.carrier, sw.TiledEvictable)
    assert _values(tensor.grad) == [6.0] * SIZE

    assert carrier.version == 0


def test_direct_elementwise_vjp_checks_saved_version() -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    lhs_carrier, lhs = _partially_resident(lhs_values)
    _rhs_carrier, rhs = _partially_resident([float(index + 2) for index in range(SIZE)])
    result = sw.elementwise_mul(lhs, rhs)

    lhs[0] = 99.0
    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(_dense([1.0] * SIZE))
    assert lhs_carrier.version > 0


def test_direct_relu_vjp_returns_a_full_shaped_tiled_gradient() -> None:
    values = [float(index - 8) for index in range(SIZE)]
    _carrier, tensor = _partially_resident(values)
    result = sw.relu(tensor)

    result.backward(_dense([1.0] * SIZE))

    assert tensor.grad is not None
    assert isinstance(tensor.grad.carrier, sw.TiledEvictable)
    assert tensor.grad.layout.shape == sw.Shape(SHAPE)
    assert _values(tensor.grad) == [0.0 if value <= 0 else 1.0 for value in values]


def test_direct_reduce_sum_vjp_returns_a_full_shaped_tiled_gradient() -> None:
    values = [float(index) for index in range(SIZE)]
    _carrier, tensor = _partially_resident(values)
    result = tensor.carrier.dispatch_op("reduce_sum").forward(tensor)

    result.backward(_dense([1.0] * 4, shape=(4,)))

    assert tensor.grad is not None
    assert isinstance(tensor.grad.carrier, sw.TiledEvictable)
    assert tensor.grad.layout.shape == sw.Shape(SHAPE)
    assert _values(tensor.grad) == [1.0] * SIZE


def test_direct_backward_promotes_saved_tiles_with_backward_policy_purpose() -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[sw.TileSet, str]] = []

        def plan(self, required: sw.TileSet, purpose: str) -> sw.ResidencyPlan:
            self.calls.append((required, purpose))
            return sw.ResidencyPlan()

    policy = RecordingPolicy()
    carrier = _tiled(
        [float(index - 8) for index in range(SIZE)],
        residency_policy=policy,
    )
    tensor = _tensor(carrier)
    result = sw.relu(tensor)
    all_tiles = sw.TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    carrier.evict(all_tiles)

    result.backward(_dense([1.0] * SIZE))

    assert policy.calls == [(all_tiles, "operation"), (all_tiles, "backward")]
    assert carrier.primary_tiles == all_tiles
    assert tensor.grad is not None
    assert _values(tensor.grad) == [0.0] * 9 + [1.0] * 7


def test_direct_operation_publishes_one_outer_autograd_node() -> None:
    lhs_values = [float(index + 1) for index in range(SIZE)]
    lhs_carrier, lhs = _partially_resident(lhs_values)
    rhs_carrier, rhs = _partially_resident([float(index + 2) for index in range(SIZE)])

    result = sw.elementwise_mul(lhs, rhs)
    operation = result.autograd_ctx

    assert operation is not None
    assert operation._operation_name == "elementwise_mul"
    assert operation._dispatch_carrier_class is sw.TiledEvictable
    assert operation.inputs() == (lhs, rhs)
    assert not any(
        getattr(value, "autograd_ctx", None) is not None for value in operation.inputs()
    )
    assert lhs_carrier is lhs.carrier
    assert rhs_carrier is rhs.carrier
