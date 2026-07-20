import pytest
import strideweave as sw
from strideweave import (
    CPU,
    DType,
    FileBacked,
    Generic,
    Layout,
    Shape,
    Stride,
    Tensor,
)
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


def make_cpu_tensor(values, dtype=DType.Float32):
    carrier = CPU(len(values), dtype=dtype)
    layout = Layout(Shape(len(values)), Stride(1))
    tensor = Tensor(carrier, 0, layout)
    for index, value in enumerate(values):
        tensor[index] = value
    return tensor


def make_file_backed(dtype=DType.Float32):
    return FileBacked(dtype=dtype)


def tensor_values(tensor):
    return [tensor[index] for index in range(tensor.size())]


def test_move_cpu_tensor_to_file_backed_copies_values_and_releases_source():
    tensor = make_cpu_tensor([1.0, 2.0, 3.0])
    source_carrier = tensor.carrier
    destination = make_file_backed()

    moved = sw.move(tensor, destination)

    assert moved.carrier is destination
    assert tensor_values(moved) == [1.0, 2.0, 3.0]
    assert source_carrier.is_released()
    with pytest.raises(RuntimeError, match="released"):
        tensor[0]


def test_move_released_cpu_source_rejects_native_operations():
    tensor = make_cpu_tensor([1.0, 2.0])
    sw.move(tensor, make_file_backed())

    with pytest.raises(RuntimeError, match="released"):
        sw.relu(tensor)
    with pytest.raises(RuntimeError, match="released"):
        _ = tensor + tensor


def test_move_roundtrip_through_file_backed_backpropagates_to_cpu_leaf():
    tensor = make_cpu_tensor([1.0, 2.0, 3.0])

    file_backed = sw.move(tensor, make_file_backed())
    result = sw.move(file_backed, CPU(3, dtype=DType.Float32))

    assert type(result.carrier) is CPU
    assert tensor_values(result) == [1.0, 2.0, 3.0]

    gradient = make_cpu_tensor([10.0, 11.0, 12.0])
    result.backward(gradient)

    assert tensor.grad is not None
    assert type(tensor.grad.carrier) is CPU
    assert tensor_values(tensor.grad) == pytest.approx([10.0, 11.0, 12.0])


def test_move_records_autograd_context_with_source_input():
    tensor = make_cpu_tensor([1.0, 2.0])

    moved = sw.move(tensor, make_file_backed())

    context = moved.autograd_ctx
    assert context is not None
    assert isinstance(context, MoveOperation)
    assert type(context) is CpuToFileBackedMoveOperation
    assert context.inputs() == (tensor,)


def test_move_under_no_grad_builds_no_graph():
    tensor = make_cpu_tensor([1.0, 2.0])

    with sw.no_grad():
        moved = sw.move(tensor, make_file_backed())

    assert moved.autograd_ctx is None


def test_move_from_generic_source():
    layout = Layout(Shape(2), Stride(1))
    generic = Generic([4.0, 5.0])
    tensor = Tensor(generic, 0, layout)

    moved = sw.move(tensor, FileBacked(dtype=DType.Floating))

    assert tensor_values(moved) == [4.0, 5.0]
    assert generic.is_released()
    with pytest.raises(RuntimeError, match="released"):
        tensor[0]


def test_move_to_presized_cpu_destination():
    tensor = make_cpu_tensor([4.0, 5.0])

    moved = sw.move(tensor, CPU(2, dtype=DType.Float32))

    assert type(moved.carrier) is CPU
    assert tensor_values(moved) == [4.0, 5.0]


def test_move_rejects_non_data_destination():
    tensor = make_cpu_tensor([1.0])

    with pytest.raises(TypeError, match="Carrier instance"):
        sw.move(tensor, CPU)
    assert not tensor.carrier.is_released()


def test_move_rejects_destination_identical_to_source_carrier():
    tensor = make_cpu_tensor([1.0])

    with pytest.raises(ValueError, match="own carrier"):
        sw.move(tensor, tensor.carrier)
    assert not tensor.carrier.is_released()


def test_move_rejects_immutable_destination():
    tensor = make_cpu_tensor([1.0])

    with pytest.raises(RuntimeError, match="mutable"):
        sw.move(tensor, CPU(1, mutable=False))
    assert not tensor.carrier.is_released()


def test_move_rejects_undersized_destination():
    tensor = make_cpu_tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="too small"):
        sw.move(tensor, CPU(2, dtype=DType.Float32))
    assert not tensor.carrier.is_released()


def test_move_rejects_released_destination():
    tensor = make_cpu_tensor([1.0])
    destination = make_file_backed()
    destination.release()

    with pytest.raises(RuntimeError, match="released"):
        sw.move(tensor, destination)
    assert not tensor.carrier.is_released()


@pytest.mark.parametrize(
    ("source_dtype", "destination_dtype"),
    [
        (DType.Float32, DType.Floating),
        (DType.Float32, DType.Int32),
        (DType.Int32, DType.Float32),
    ],
)
def test_move_rejects_mismatched_dtypes(source_dtype, destination_dtype):
    values = [1, 2] if source_dtype is DType.Int32 else [1.0, 2.0]
    tensor = make_cpu_tensor(values, dtype=source_dtype)

    with pytest.raises(TypeError, match="dtype must match"):
        sw.move(tensor, FileBacked(dtype=destination_dtype))
    assert not tensor.carrier.is_released()
    assert tensor_values(tensor) == values


def test_move_rejects_generic_to_cpu_dtype_mismatch():
    layout = Layout(Shape(2), Stride(1))
    tensor = Tensor(Generic([1.0, 2.0]), 0, layout)

    with pytest.raises(TypeError, match="dtype must match"):
        sw.move(tensor, CPU(2, dtype=DType.Float32))
    assert not tensor.carrier.is_released()


def test_move_failed_copy_leaves_source_intact():
    layout = Layout(Shape(2), Stride(1))
    generic = Generic([1.0, "not a number"])
    tensor = Tensor(generic, 0, layout)

    with pytest.raises(Exception):
        sw.move(tensor, FileBacked(dtype=DType.Floating))

    assert not generic.is_released()
    assert tensor[0] == 1.0


def test_move_int32_tensors_between_cpu_and_file_backed():
    tensor = make_cpu_tensor([1, -2, 3], dtype=DType.Int32)

    moved = sw.move(tensor, FileBacked(dtype=DType.Int32))
    result = sw.move(moved, CPU(3, dtype=DType.Int32))

    assert tensor_values(result) == [1, -2, 3]
    assert not result.is_differentiable()
    assert result.autograd_ctx is None


def test_move_preserves_strided_layout():
    carrier = CPU(6, dtype=DType.Float32)
    layout = Layout(Shape([2, 3]), Stride([3, 1]))
    tensor = Tensor(carrier, 0, layout)
    for index in range(tensor.size()):
        tensor[index] = float(index)

    moved = sw.move(tensor, make_file_backed())

    assert moved.layout == layout
    assert tensor_values(moved) == [float(index) for index in range(6)]


def test_move_bulk_copies_offset_tensor_views():
    carrier = CPU(6, dtype=DType.Float32)
    full = Tensor(carrier, 0, Layout(Shape(6), Stride(1)))
    for index in range(6):
        full[index] = float(index)
    view = Tensor(carrier, 2, Layout(Shape(3), Stride(1)))

    moved = sw.move(view, make_file_backed())

    assert tensor_values(moved) == [2.0, 3.0, 4.0]


def test_move_bulk_copies_layouts_with_holes():
    carrier = CPU(5, dtype=DType.Float32)
    layout = Layout(Shape(3), Stride(2))
    tensor = Tensor(carrier, 0, layout)
    for index in range(tensor.size()):
        tensor[index] = float(index + 1)

    moved = sw.move(tensor, make_file_backed())
    result = sw.move(moved, CPU(5, dtype=DType.Float32))

    assert tensor_values(result) == [1.0, 2.0, 3.0]


def test_move_operation_is_exported_from_operation_module():
    assert MoveOperation is sw.MoveOperation


def test_dispatch_move_returns_registered_bulk_operations():
    assert dispatch_move(CPU, FileBacked) is CpuToFileBackedMoveOperation
    assert dispatch_move(FileBacked, CPU) is FileBackedToCpuMoveOperation


def test_dispatch_move_falls_back_to_elementwise_for_unregistered_pairs():
    assert dispatch_move(Generic, FileBacked) is ElementwiseMoveOperation
    assert dispatch_move(CPU, CPU) is ElementwiseMoveOperation


def test_registered_move_operation_is_used_by_public_move():
    calls = []

    class SpyMoveOperation(ElementwiseMoveOperation):
        def _copy(self, tensor, destination, output, element_count):
            calls.append(element_count)
            super()._copy(tensor, destination, output, element_count)

    with registered_move_operation(Generic, Generic, SpyMoveOperation):
        layout = Layout(Shape(2), Stride(1))
        tensor = Tensor(Generic([1.0, 2.0]), 0, layout)
        moved = sw.move(tensor, Generic([0.0, 0.0]))

    assert calls == [2]
    assert tensor_values(moved) == [1.0, 2.0]
    assert isinstance(moved.autograd_ctx, SpyMoveOperation)


def test_register_move_operation_rejects_duplicate_pair():
    with pytest.raises(ValueError, match="already registered"):
        register_move_operation(CPU, FileBacked, ElementwiseMoveOperation)


def test_register_move_operation_rejects_non_move_operation_class():
    with pytest.raises(TypeError, match="MoveOperation subclass"):
        register_move_operation(Generic, Generic, object)  # type: ignore[arg-type]


def test_register_move_operation_rejects_non_data_classes():
    with pytest.raises(TypeError, match="source_class must be a Carrier subclass"):
        register_move_operation(int, Generic, ElementwiseMoveOperation)
    with pytest.raises(TypeError, match="destination_class must be a Carrier subclass"):
        register_move_operation(Generic, int, ElementwiseMoveOperation)


def test_unregister_move_operation_removes_registration():
    register_move_operation(Generic, Generic, ElementwiseMoveOperation)

    removed = unregister_move_operation(Generic, Generic)

    assert removed is ElementwiseMoveOperation
    assert dispatch_move(Generic, Generic) is ElementwiseMoveOperation
    with pytest.raises(KeyError, match="no move operation is registered"):
        unregister_move_operation(Generic, Generic)


def test_registered_move_operation_unregisters_when_block_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with registered_move_operation(Generic, Generic, ElementwiseMoveOperation):
            assert dispatch_move(Generic, Generic) is ElementwiseMoveOperation
            raise RuntimeError("boom")

    with pytest.raises(KeyError, match="no move operation is registered"):
        unregister_move_operation(Generic, Generic)


def test_concrete_move_operation_rejects_wrong_source_class():
    layout = Layout(Shape(2), Stride(1))
    tensor = Tensor(Generic([1.0, 2.0]), 0, layout)
    destination = FileBacked(dtype=DType.Floating)

    with pytest.raises(TypeError, match="requires a CPU source"):
        CpuToFileBackedMoveOperation().forward(tensor, destination)
    assert not tensor.carrier.is_released()


def test_concrete_move_operation_rejects_wrong_destination_class():
    tensor = make_cpu_tensor([1.0, 2.0])
    destination = CPU(2, dtype=DType.Float32)

    with pytest.raises(TypeError, match="requires a FileBacked destination"):
        CpuToFileBackedMoveOperation().forward(tensor, destination)
    assert not tensor.carrier.is_released()


def test_released_data_new_like_still_creates_fresh_storage():
    tensor = make_cpu_tensor([1.0, 2.0])
    source_carrier = tensor.carrier
    sw.move(tensor, make_file_backed())

    fresh = source_carrier.new_like([7.0, 8.0])

    assert [fresh[index] for index in range(2)] == [7.0, 8.0]
