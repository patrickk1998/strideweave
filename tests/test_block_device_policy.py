from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

import pytest

import strideweave as sw
from strideweave import CPU, DType, FileBacked, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.block_device.carrier import BlockDevice
from strideweave.carriers.move import (
    CpuToFileBackedMoveOperation,
    ElementwiseMoveOperation,
    FileBackedToCpuMoveOperation,
    MoveOperation,
    dispatch_move,
    register_move_operation,
    registered_move_operation,
    unregister_move_operation,
)

_block_device = import_module("strideweave._block_device")

CAPACITY = 16 * 1024
LOGICAL_BLOCK_SIZE = 512
PHYSICAL_BLOCK_SIZE = 4096


def make_block_carrier(
    tmp_path: Path,
    *,
    size: int = 1,
    mutable: bool = True,
) -> tuple[sw.BlockDeviceCarrier, Any]:
    path = tmp_path / "block-policy-storage"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(CAPACITY))
    state = _block_device._TestIoState()
    handle = _block_device._open_block_device_for_test(
        str(path),
        CAPACITY,
        LOGICAL_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE,
        state,
    )
    device = BlockDevice._from_handle_for_test(str(path), handle)  # pyright: ignore[reportAttributeAccessIssue]
    return device.allocate(size, mutable=mutable, empty=True), state


def make_cpu_tensor(values: list[float | int], dtype: DType = DType.Float32) -> Tensor:
    carrier = CPU(len(values), dtype=dtype)
    tensor = Tensor(carrier, 0, Layout(Shape(len(values)), Stride(1)))
    for index, value in enumerate(values):
        tensor[index] = value
    return tensor


def tensor_for(carrier: sw.Carrier) -> Tensor:
    return Tensor(carrier, 0, Layout(Shape(carrier.size()), Stride(1)))


def block_state(carrier: sw.BlockDeviceCarrier, state: Any) -> tuple[Any, ...]:
    return (
        carrier.size(),
        carrier.device_offset_bytes,
        carrier.extent_bytes,
        carrier.version,
        carrier.is_released(),
        carrier.device.is_faulted(),
        state.read_attempts,
        state.write_attempts,
    )


@pytest.mark.parametrize(
    ("source_class", "destination_class"),
    [
        (CPU, sw.BlockDeviceCarrier),
        (sw.BlockDeviceCarrier, CPU),
        (Generic, sw.BlockDeviceCarrier),
        (sw.BlockDeviceCarrier, FileBacked),
        (sw.BlockDeviceCarrier, sw.BlockDeviceCarrier),
    ],
)
def test_public_registry_mutation_rejects_every_block_pair(
    source_class: type,
    destination_class: type,
) -> None:
    with pytest.raises(ValueError, match="protected built-ins"):
        register_move_operation(
            source_class, destination_class, ElementwiseMoveOperation
        )
    with pytest.raises(ValueError, match="protected built-ins"):
        unregister_move_operation(source_class, destination_class)


def test_block_registry_argument_validation_precedes_pair_policy() -> None:
    with pytest.raises(TypeError, match="source_class"):
        register_move_operation(
            object,
            sw.BlockDeviceCarrier,
            ElementwiseMoveOperation,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="destination_class"):
        register_move_operation(
            sw.BlockDeviceCarrier,
            object,
            ElementwiseMoveOperation,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="operation_class"):
        register_move_operation(
            CPU,
            sw.BlockDeviceCarrier,
            object,  # type: ignore[arg-type]
        )


def test_block_registry_context_never_enters_or_mutates() -> None:
    entered = False

    with pytest.raises(ValueError, match="protected built-ins"):
        with registered_move_operation(
            CPU, sw.BlockDeviceCarrier, ElementwiseMoveOperation
        ):
            entered = True

    assert not entered


def test_block_policy_uses_exact_identity_not_a_class_name() -> None:
    class CarrierBase(sw.Carrier):
        pass

    lookalike = type("BlockDeviceCarrier", (CarrierBase,), {})

    assert dispatch_move(lookalike, Generic) is ElementwiseMoveOperation
    register_move_operation(lookalike, Generic, ElementwiseMoveOperation)
    assert unregister_move_operation(lookalike, Generic) is ElementwiseMoveOperation


@pytest.mark.parametrize(
    "pair",
    [
        (Generic, sw.BlockDeviceCarrier),
        (sw.BlockDeviceCarrier, FileBacked),
        (sw.BlockDeviceCarrier, sw.BlockDeviceCarrier),
    ],
)
def test_dispatch_rejects_unregistered_block_pairs(pair: tuple[type, type]) -> None:
    with pytest.raises(NotImplementedError, match="block-device carrier pair"):
        dispatch_move(*pair)


def test_public_move_rejects_unsupported_block_pairs_before_effects(
    tmp_path: Path,
) -> None:
    destination, state = make_block_carrier(tmp_path)
    source = Generic([1.0], dtype=DType.Float32)
    tensor = tensor_for(source)
    before = block_state(destination, state)

    with pytest.raises(NotImplementedError, match="block-device carrier pair"):
        sw.move(tensor, destination)

    assert block_state(destination, state) == before
    assert not source.is_released()
    assert source[0] == 1.0


@pytest.mark.parametrize("operation_class", [MoveOperation, ElementwiseMoveOperation])
def test_direct_generic_move_operations_reject_a_valid_cpu_block_pair(
    tmp_path: Path,
    operation_class: type[MoveOperation],
) -> None:
    tensor = make_cpu_tensor([1.0])
    destination, state = make_block_carrier(tmp_path)
    before = block_state(destination, state)

    with pytest.raises(NotImplementedError, match="exact built-in"):
        operation_class().forward(tensor, destination)

    assert block_state(destination, state) == before
    assert not tensor.carrier.is_released()


def test_direct_unpinned_and_pinned_custom_operations_cannot_bypass_policy(
    tmp_path: Path,
) -> None:
    class CustomMove(MoveOperation):
        def _copy(
            self, tensor: Any, destination: Any, output: Any, element_count: int
        ) -> None:
            raise AssertionError("copy must remain unreachable")

    class PinnedCustomMove(CustomMove):
        source_class: ClassVar[type | None] = CPU
        destination_class: ClassVar[type | None] = sw.BlockDeviceCarrier

    for operation_class in (CustomMove, PinnedCustomMove):
        tensor = make_cpu_tensor([1.0])
        destination, state = make_block_carrier(tmp_path / operation_class.__name__)
        before = block_state(destination, state)

        with pytest.raises(NotImplementedError, match="exact built-in"):
            operation_class().forward(tensor, destination)

        assert block_state(destination, state) == before
        assert not tensor.carrier.is_released()


def test_pinned_class_errors_precede_block_policy(tmp_path: Path) -> None:
    block, _ = make_block_carrier(tmp_path)
    cpu_tensor = make_cpu_tensor([1.0])

    with pytest.raises(TypeError, match="FileBacked destination"):
        CpuToFileBackedMoveOperation().forward(cpu_tensor, block)
    with pytest.raises(TypeError, match="FileBacked source"):
        FileBackedToCpuMoveOperation().forward(tensor_for(block), CPU(1))

    assert not cpu_tensor.carrier.is_released()
    assert not block.is_released()


def test_public_block_move_preserves_inherited_validation_precedence(
    tmp_path: Path,
) -> None:
    block, state = make_block_carrier(tmp_path)

    with pytest.raises(TypeError, match="tensor must be a Tensor"):
        sw.move(object(), block)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="destination must be a Carrier"):
        sw.move(tensor_for(block), object())
    with pytest.raises(ValueError, match="own carrier"):
        sw.move(tensor_for(block), block)

    released_source = Generic([1.0], dtype=DType.Float32)
    released_tensor = tensor_for(released_source)
    released_source.release()
    with pytest.raises(RuntimeError, match="tensor carrier is released"):
        sw.move(released_tensor, block)

    owned_source = Generic([1.0], dtype=DType.Float32)
    hierarchy = sw.Evictable(owned_source, Generic([0.0], dtype=DType.Float32))
    with pytest.raises(RuntimeError, match="owned by another carrier"):
        sw.move(tensor_for(owned_source), block)
    assert hierarchy[0] == 1.0

    immutable, _ = make_block_carrier(tmp_path / "immutable", mutable=False)
    with pytest.raises(RuntimeError, match="must be mutable"):
        sw.move(make_cpu_tensor([1.0]), immutable)

    released, _ = make_block_carrier(tmp_path / "released")
    released.release()
    with pytest.raises(RuntimeError, match="destination carrier is released"):
        sw.move(make_cpu_tensor([1.0]), released)

    with pytest.raises(TypeError, match="dtype must match"):
        sw.move(make_cpu_tensor([1], DType.Int32), block)

    assert state.read_attempts == 0
    assert state.write_attempts == 0
    assert not block.device.is_faulted()


@pytest.mark.parametrize("block_position", ["primary", "secondary"])
def test_evictable_rejects_block_tiers_before_hierarchy_effects(
    tmp_path: Path,
    block_position: str,
) -> None:
    block, state = make_block_carrier(tmp_path)
    ordinary = Generic([0.0], dtype=DType.Float32)
    before = block_state(block, state)
    tiers = (block, ordinary) if block_position == "primary" else (ordinary, block)

    with pytest.raises(TypeError, match="not a supported Evictable tier"):
        sw.Evictable(*tiers)

    assert block_state(block, state) == before
    assert not block.is_owned()
    assert not ordinary.is_owned()


def test_evictable_validates_both_carrier_types_before_block_exclusion(
    tmp_path: Path,
) -> None:
    block, _ = make_block_carrier(tmp_path)

    with pytest.raises(TypeError, match="primary must be a Carrier"):
        sw.Evictable(object(), block)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="secondary must be a Carrier"):
        sw.Evictable(block, object())  # type: ignore[arg-type]


def test_evictable_block_exclusion_precedes_distinctness_and_lifecycle(
    tmp_path: Path,
) -> None:
    block, state = make_block_carrier(tmp_path)
    block.release()
    before = block_state(block, state)

    with pytest.raises(TypeError, match="not a supported Evictable tier"):
        sw.Evictable(block, block)

    assert block_state(block, state) == before
