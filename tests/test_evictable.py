import gc

import neotorch
import pytest
from neotorch import (
    CPU,
    DataType,
    Evictable,
    EvictableOperation,
    FileBacked,
    Generic,
    Layout,
    Shape,
    Stride,
    Tensor,
)
from neotorch.data.move import ElementwiseMoveOperation, registered_move_operation


def flat_layout(size):
    return Layout(Shape(size), Stride(1))


def make_cpu_data(values, dtype=DataType.Float32, *, mutable=True):
    data = CPU(len(values), dtype=dtype, mutable=mutable)
    for index, value in enumerate(values):
        data[index] = value
    return data


def make_cpu_evictable(values, dtype=DataType.Float32, *, mutable=True):
    return Evictable(
        make_cpu_data(values, dtype, mutable=mutable),
        FileBacked(dtype=dtype),
    )


def make_tensor(data):
    return Tensor(data, 0, flat_layout(data.size()))


def values(tensor):
    return [tensor[index] for index in range(tensor.size())]


def adapter_for(tensor):
    adapter = tensor.autograd_ctx
    assert isinstance(adapter, EvictableOperation)
    return adapter


def evictable_data(tensor):
    data = tensor.data
    assert isinstance(data, Evictable)
    return data


def test_evictable_public_exports():
    assert neotorch.Evictable is Evictable
    assert neotorch.EvictableOperation is EvictableOperation


def test_evictable_constructor_exposes_promoted_hierarchy():
    primary = make_cpu_data([1.0, 2.0])
    secondary = FileBacked(dtype=DataType.Float32)

    data = Evictable(primary, secondary)

    assert data.primary is primary
    assert data.secondary is secondary
    assert data.size() == 2
    assert data.type() is DataType.Float32
    assert data.is_mutable()
    assert primary.is_owned()
    assert secondary.is_owned()
    assert not primary.is_mutable()
    assert not secondary.is_mutable()
    assert not data.is_evicted()
    assert data[1] == 2.0


@pytest.mark.parametrize(
    ("primary", "secondary", "error", "message"),
    [
        (object(), FileBacked(), TypeError, "primary"),
        (Generic([1.0]), object(), TypeError, "secondary"),
        (Generic([1.0]), Generic([0.0], dtype=DataType.Any), TypeError, "dtypes"),
        (Generic([]), Generic([]), ValueError, "at least one"),
        (
            Generic([1.0]),
            Generic([0.0], mutable=False),
            RuntimeError,
            "mutable",
        ),
    ],
)
def test_evictable_constructor_validation(primary, secondary, error, message):
    with pytest.raises(error, match=message):
        Evictable(primary, secondary)  # type: ignore[arg-type]


def test_evictable_constructor_rejects_identical_and_released_tiers():
    same = Generic([1.0])
    with pytest.raises(ValueError, match="distinct"):
        Evictable(same, same)

    released_primary = Generic([1.0])
    released_primary.release()
    with pytest.raises(RuntimeError, match="primary.*released"):
        Evictable(released_primary, Generic([0.0]))

    released_secondary = Generic([0.0])
    released_secondary.release()
    with pytest.raises(RuntimeError, match="secondary.*released"):
        Evictable(Generic([1.0]), released_secondary)


def test_evictable_constructor_rejects_data_owned_by_another_composition():
    primary = Generic([1.0])
    first = Evictable(primary, Generic([0.0]))

    with pytest.raises(RuntimeError, match="already owned"):
        Evictable(primary, Generic([0.0]))

    assert first[0] == 1.0


def test_rejected_owned_secondary_leaves_primary_unclaimed():
    owned_secondary = Generic([0.0])
    existing = Evictable(Generic([1.0]), owned_secondary)
    unclaimed_primary = Generic([2.0])

    with pytest.raises(RuntimeError, match="already owned"):
        Evictable(unclaimed_primary, owned_secondary)

    assert not unclaimed_primary.is_owned()
    assert unclaimed_primary.is_mutable()
    assert existing[0] == 1.0


def test_destroying_evictable_returns_unreleased_tiers_to_the_caller():
    primary = Generic([1.0])
    secondary = Generic([0.0])
    data = Evictable(primary, secondary)

    del data
    gc.collect()

    assert not primary.is_owned()
    assert not secondary.is_owned()
    primary[0] = 2.0
    assert primary[0] == 2.0


def test_owned_tier_aliases_are_read_only_and_cannot_be_released():
    primary = Generic([1.0])
    secondary = Generic([0.0])
    data = Evictable(primary, secondary)

    with pytest.raises(RuntimeError, match="not mutable"):
        primary[0] = 2.0
    with pytest.raises(RuntimeError, match="not mutable"):
        primary.set_value(0, 2.0)
    with pytest.raises(RuntimeError, match="not mutable"):
        secondary[0] = 2.0
    with pytest.raises(RuntimeError, match="owned"):
        primary._increment_version()
    with pytest.raises(RuntimeError, match="owned"):
        primary.release()

    assert data.version == 0
    assert primary.version == 0
    assert data[0] == 1.0


def test_owned_tier_cannot_be_moved_directly():
    primary = Generic([1.0])
    data = Evictable(primary, Generic([0.0]))
    destination = Generic([0.0])

    with pytest.raises(RuntimeError, match="cannot be moved directly"):
        neotorch.move(make_tensor(primary), destination)

    assert data[0] == 1.0
    assert destination[0] == 0.0


def test_ownership_guards_compose_for_nested_evictable_data():
    inner = Evictable(Generic([1.0]), Generic([0.0]))
    outer = Evictable(inner, Generic([0.0]))

    with pytest.raises(RuntimeError, match="cannot be modified directly"):
        inner.evict()

    outer[0] = 2.0
    outer.evict()
    outer.promote()

    assert outer[0] == 2.0


def test_wrapper_mutation_remains_available_and_updates_its_version_once():
    primary = Generic([1.0])
    data = Evictable(primary, Generic([0.0]))

    data[0] = 2.0

    assert data[0] == 2.0
    assert data.version == 1
    assert primary.version == 1
    assert not primary.is_mutable()


def test_evict_and_promote_roundtrip_releases_and_recreates_tiers():
    data = make_cpu_evictable([1.0, -2.0, 3.0])
    original_primary = data.primary
    original_secondary = data.secondary

    data.evict()

    assert data.is_evicted()
    assert original_primary.is_released()
    assert data.secondary is not original_secondary
    assert original_secondary.is_released()
    assert [data.secondary[i] for i in range(3)] == [1.0, -2.0, 3.0]
    with pytest.raises(RuntimeError, match="evicted"):
        _ = data[0]

    data.promote()

    assert not data.is_evicted()
    assert data.primary is not original_primary
    assert original_secondary.is_released()
    assert [data[i] for i in range(3)] == [1.0, -2.0, 3.0]


def test_evict_and_promote_are_idempotent():
    data = make_cpu_evictable([1.0])

    assert data.promote() is None
    primary = data.primary
    assert data.promote() is None
    assert data.primary is primary

    data.evict()
    secondary = data.secondary
    assert data.evict() is None
    assert data.secondary is secondary


def test_elementwise_fallback_moves_generic_hierarchy():
    data = Evictable(
        Generic(["a", "b"], dtype=DataType.Any),
        Generic([None, None], dtype=DataType.Any),
    )

    data.evict()
    data.promote()

    assert [data[i] for i in range(2)] == ["a", "b"]


def test_transition_reallocates_an_undersized_secondary_tier():
    data = Evictable(Generic([1.0, 2.0]), Generic([0.0]))

    data.evict()

    assert data.secondary.size() == 2
    assert [data.secondary[i] for i in range(2)] == [1.0, 2.0]


def test_failed_eviction_preserves_state_and_can_be_retried():
    attempts = 0

    class FailFirstMove(ElementwiseMoveOperation):
        def _copy(self, tensor, destination, output, element_count):
            nonlocal attempts
            attempts += 1
            output[0] = tensor[0]
            if attempts == 1:
                raise OSError("boom")
            super()._copy(tensor, destination, output, element_count)

    primary = Generic([1.0, 2.0])
    secondary = Generic([])
    data = Evictable(primary, secondary)

    with registered_move_operation(Generic, Generic, FailFirstMove):
        with pytest.raises(OSError, match="boom"):
            data.evict()

        assert not data.is_evicted()
        assert data.primary is primary
        assert data.secondary is secondary
        assert not primary.is_released()
        assert not secondary.is_released()
        assert primary.is_owned()
        assert secondary.is_owned()

        data.evict()

    assert data.is_evicted()
    assert values(make_tensor(data.secondary)) == [1.0, 2.0]
    assert secondary.is_released()
    assert not secondary.is_owned()


def test_failed_promotion_preserves_state_and_can_be_retried():
    attempts = 0

    class FailFirstMove(ElementwiseMoveOperation):
        def _copy(self, tensor, destination, output, element_count):
            nonlocal attempts
            attempts += 1
            output[0] = tensor[0]
            if attempts == 1:
                raise OSError("boom")
            super()._copy(tensor, destination, output, element_count)

    data = Evictable(Generic([1.0, 2.0]), Generic([]))
    original_primary = data.primary
    data.evict()
    evicted_secondary = data.secondary

    with registered_move_operation(Generic, Generic, FailFirstMove):
        with pytest.raises(OSError, match="boom"):
            data.promote()

        assert data.is_evicted()
        assert data.primary is original_primary
        assert original_primary.is_released()
        assert original_primary.is_owned()
        assert data.secondary is evicted_secondary
        assert not evicted_secondary.is_released()
        assert evicted_secondary.is_owned()

        data.promote()

    assert not data.is_evicted()
    assert [data[index] for index in range(2)] == [1.0, 2.0]
    assert not original_primary.is_owned()


def test_transitions_call_move_private_forward_not_autograd_forward():
    calls = []

    class SpyMove(ElementwiseMoveOperation):
        def forward(self, *inputs):
            calls.append("forward")
            return super().forward(*inputs)

        def _forward(self, tensor, destination):
            calls.append("_forward")
            return super()._forward(tensor, destination)

    with registered_move_operation(Generic, Generic, SpyMove):
        data = Evictable(Generic([1.0]), Generic([0.0]))
        data.evict()
        data.promote()

    assert calls == ["_forward", "_forward"]


def test_transitions_resolve_move_registry_when_each_transition_runs():
    calls = []

    class SpyMove(ElementwiseMoveOperation):
        def _forward(self, tensor, destination):
            calls.append("spy")
            return super()._forward(tensor, destination)

    data = Evictable(Generic([1.0]), Generic([0.0]))
    with registered_move_operation(Generic, Generic, SpyMove):
        data.evict()
    data.promote()

    with registered_move_operation(Generic, Generic, SpyMove):
        later = Evictable(Generic([2.0]), Generic([0.0]))
    later.evict()

    assert calls == ["spy"]


def test_immutable_hierarchy_remains_externally_read_only_after_promotion():
    prototype = make_cpu_data([1.0, 2.0])
    primary = prototype.new_like([1.0, 2.0], mutable=False)
    data = Evictable(primary, FileBacked(dtype=DataType.Float32))

    data.evict()
    data.promote()

    assert not data.is_mutable()
    assert not data.primary.is_mutable()
    with pytest.raises(RuntimeError, match="not mutable"):
        data.primary[0] = 3.0


def test_immutable_promotion_keeps_the_move_destination_without_copying():
    promotion_destinations = []

    class CaptureDestinationMove(ElementwiseMoveOperation):
        def _forward(self, tensor, destination):
            result = super()._forward(tensor, destination)
            promotion_destinations.append(destination)
            return result

    data = Evictable(
        Generic([1.0, 2.0], mutable=False),
        Generic([]),
    )
    with registered_move_operation(Generic, Generic, CaptureDestinationMove):
        data.evict()
        data.promote()

    assert not data.is_mutable()
    assert data.primary is promotion_destinations[-1]
    assert not data.primary.is_mutable()


def test_eviction_preserves_version_and_writes_increment_it():
    data = make_cpu_evictable([1.0, 2.0])
    initial = data.version

    data.evict()
    data.promote()

    assert data.version == initial
    data[0] = 4.0
    assert data.version == initial + 1


def test_release_releases_both_tiers_and_blocks_transitions():
    data = make_cpu_evictable([1.0])
    primary = data.primary
    secondary = data.secondary

    data.release()

    assert data.is_released()
    assert primary.is_released()
    assert secondary.is_released()
    assert data.size() == 0
    with pytest.raises(RuntimeError, match="released"):
        data.promote()
    with pytest.raises(RuntimeError, match="released"):
        data.evict()


def test_new_like_preserves_hierarchy_and_supports_dtype_change():
    data = make_cpu_evictable([1, 2], DataType.Int32)

    result = data.new_like([1.5, 2.5], dtype=DataType.Float32)

    assert isinstance(result, Evictable)
    assert type(result.primary) is CPU
    assert type(result.secondary) is FileBacked
    assert result.secondary.size() == 0
    assert result.type() is DataType.Float32
    assert [result[i] for i in range(2)] == pytest.approx([1.5, 2.5])


def test_empty_like_allocates_primary_and_leaves_secondary_lazy():
    data = make_cpu_evictable([1, 2], DataType.Int32)

    result = data.empty_like(4, mutable=False, dtype=DataType.Float32)

    assert result.size() == 4
    assert result.primary.size() == 4
    assert result.secondary.size() == 0
    assert result.type() is DataType.Float32
    assert not result.is_mutable()
    assert not result.primary.is_mutable()
    assert not result.secondary.is_mutable()
    assert result.primary.is_owned()
    assert result.secondary.is_owned()


def test_evicted_data_blocks_writes_operations_and_scatter():
    data = make_cpu_evictable([1.0, 2.0])
    tensor = make_tensor(data)
    data.evict()

    with pytest.raises(RuntimeError, match="evicted"):
        data[0] = 3.0
    with pytest.raises(RuntimeError, match="evicted"):
        neotorch.relu(tensor)
    with pytest.raises(RuntimeError, match="evicted"):
        data.scatter(tensor, tensor, tensor.layout)


@pytest.mark.parametrize("evicted", [False, True])
def test_evictable_never_supports_dlpack(evicted):
    tensor = make_tensor(make_cpu_evictable([1.0]))
    if evicted:
        evictable_data(tensor).evict()

    with pytest.raises(BufferError, match="not supported"):
        tensor.__dlpack_device__()
    with pytest.raises(BufferError, match="not supported"):
        tensor.__dlpack__()


def test_cpu_operation_adapter_owns_primary_operation_and_original_inputs():
    tensor = make_tensor(make_cpu_evictable([-1.0, 2.0]))

    result = neotorch.relu(tensor)

    adapter = adapter_for(result)
    assert type(adapter.primary_operation).__name__ == "_CPUReLUOperation"
    assert adapter.inputs() == (tensor,)
    (lowered,) = adapter.primary_operation.inputs()
    assert type(lowered.data) is CPU
    assert lowered.layout == tensor.layout
    assert values(result) == [0.0, 2.0]
    assert isinstance(result.data, Evictable)


def test_generic_operation_adapter_owns_generic_operation():
    tensor = make_tensor(Evictable(Generic([-1.0, 2.0]), Generic([0.0, 0.0])))

    result = neotorch.relu(tensor)

    assert type(adapter_for(result).primary_operation) is neotorch.GenericReLUOperation
    assert values(result) == [0, 2.0]


def test_layout_only_operation_reuses_same_evictable_data():
    tensor = make_tensor(make_cpu_evictable([1.0, 2.0, 3.0, 4.0]))

    view = neotorch.view(tensor, (slice(1, 3),))

    assert view.data is tensor.data
    adapter = adapter_for(view)
    assert isinstance(adapter.primary_operation, neotorch.GenericViewOperation)
    assert values(view) == [2.0, 3.0]


def test_allocating_operation_preserves_hierarchy():
    tensor = make_tensor(make_cpu_evictable([1.0, 2.0]))

    result = neotorch.mul(tensor, 3)

    assert isinstance(result.data, Evictable)
    assert type(result.data.primary) is CPU
    assert type(result.data.secondary) is FileBacked
    assert result.data.secondary.size() == 0
    assert values(result) == [3.0, 6.0]

    result.data.evict()

    assert result.data.secondary.size() == 2


def test_binary_operation_requires_matching_hierarchies():
    lhs = make_tensor(make_cpu_evictable([1.0, 2.0]))
    rhs = make_tensor(
        Evictable(
            make_cpu_data([3.0, 4.0]),
            CPU(2, dtype=DataType.Float32),
        )
    )

    with pytest.raises(TypeError, match="secondary data classes must match"):
        neotorch.add(lhs, rhs)

    plain = make_tensor(make_cpu_data([3.0, 4.0]))
    with pytest.raises(TypeError, match="backing data classes must match"):
        neotorch.add(lhs, plain)


def test_adapter_forward_is_single_use():
    tensor = make_tensor(make_cpu_evictable([1.0]))
    adapter = evictable_data(tensor).dispatch_op("relu")

    adapter.forward(tensor)

    with pytest.raises(RuntimeError, match="only be called once"):
        adapter.forward(tensor)


def test_no_grad_uses_adapter_without_attaching_graph():
    tensor = make_tensor(make_cpu_evictable([1.0]))

    with neotorch.no_grad():
        result = neotorch.relu(tensor)

    assert result.autograd_ctx is None
    assert isinstance(result.data, Evictable)


def test_adapter_preserves_generic_ctx_and_cpu_native_state():
    generic = make_tensor(Evictable(Generic([0.0]), Generic([0.0])))
    sigmoid = neotorch.sigmoid(generic)
    saved = adapter_for(sigmoid).primary_operation.ctx["saved_values"]
    assert saved == [0.5]

    cpu = make_tensor(make_cpu_evictable([2.0]))
    scaled = neotorch.mul(cpu, 4)
    assert adapter_for(scaled).primary_operation.ctx["scalar"] == 4.0

    matrix_data = make_cpu_evictable([1.0, 2.0, 3.0, 4.0])
    matrix = Tensor(
        matrix_data,
        0,
        Layout(Shape([2, 2]), Stride([1, 2])),
    )
    reduced = neotorch.reduce(matrix)
    assert "output_layout" in adapter_for(reduced).primary_operation.ctx


def test_backward_returns_evictable_gradients():
    tensor = make_tensor(make_cpu_evictable([-1.0, 2.0]))
    result = neotorch.relu(tensor)
    gradient = make_tensor(make_cpu_evictable([3.0, 4.0]))

    result.backward(gradient)

    assert tensor.grad is not None
    assert isinstance(tensor.grad.data, Evictable)
    assert type(tensor.grad.data.primary) is CPU
    assert type(tensor.grad.data.secondary) is FileBacked
    assert values(tensor.grad) == [0.0, 4.0]


def test_backward_refreshes_primary_inputs_after_evict_promote():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = neotorch.pow(tensor, 3)
    adapter = adapter_for(result)
    original_lowered = adapter.primary_operation.inputs()[0]

    evictable_data(tensor).evict()
    evictable_data(tensor).promote()
    result.backward(make_tensor(make_cpu_evictable([1.0])))

    refreshed = adapter.primary_operation.inputs()[0]
    assert refreshed.data is evictable_data(tensor).primary
    assert refreshed.data is not original_lowered.data
    assert values(tensor.grad) == pytest.approx([12.0])


def test_backward_does_not_require_the_operation_result_to_be_promoted():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = neotorch.mul(tensor, 3)
    evictable_data(result).evict()

    result.backward(make_tensor(make_cpu_evictable([1.0])))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([3.0])


def test_backward_fails_if_required_input_remains_evicted():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = neotorch.pow(tensor, 3)
    evictable_data(tensor).evict()

    with pytest.raises(RuntimeError, match="evicted"):
        result.backward(make_tensor(make_cpu_evictable([1.0])))


def test_mutation_after_forward_still_fails_version_validation():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = neotorch.pow(tensor, 3)
    tensor[0] = 4.0

    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(make_tensor(make_cpu_evictable([1.0])))


def test_primary_alias_cannot_silently_change_saved_autograd_input():
    primary = Generic([2.0])
    tensor = make_tensor(Evictable(primary, Generic([0.0])))
    result = neotorch.pow(tensor, 3)

    with pytest.raises(RuntimeError, match="not mutable"):
        primary[0] = 4.0

    result.backward(make_tensor(Evictable(Generic([1.0]), Generic([0.0]))))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([12.0])


def test_operation_results_claim_their_child_storage():
    tensor = make_tensor(make_cpu_evictable([2.0]))

    result = neotorch.mul(tensor, 3)
    result_data = evictable_data(result)

    assert result_data.primary.is_owned()
    assert result_data.secondary.is_owned()
    with pytest.raises(RuntimeError, match="not mutable"):
        result_data.primary[0] = 7.0


def test_repeated_backward_reuses_state_and_accumulates_gradients():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = neotorch.pow(tensor, 3)

    result.backward(make_tensor(make_cpu_evictable([1.0])))
    result.backward(make_tensor(make_cpu_evictable([1.0])))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([24.0])


def test_scalar_implicit_backward_uses_evictable_gradient():
    tensor = make_tensor(make_cpu_evictable([3.0]))
    result = neotorch.mul(tensor, 2)

    result.backward()

    assert tensor.grad is not None
    assert isinstance(tensor.grad.data, Evictable)
    assert values(tensor.grad) == [2.0]
