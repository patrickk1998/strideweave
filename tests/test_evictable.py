import gc

import pytest

import strideweave as sw
from strideweave import (
    CPU,
    DType,
    Evictable,
    EvictableOperation,
    FileBacked,
    Generic,
    Layout,
    Shape,
    Stride,
    Tensor,
)
from strideweave.carriers.move import (
    ElementwiseMoveOperation,
    registered_move_operation,
)


def flat_layout(size):
    return Layout(Shape(size), Stride(1))


def make_cpu_carrier(values, dtype=DType.Float32, *, mutable=True):
    carrier = CPU(len(values), dtype=dtype, mutable=mutable)
    for index, value in enumerate(values):
        carrier[index] = value
    return carrier


def make_cpu_evictable(values, dtype=DType.Float32, *, mutable=True):
    return Evictable(
        make_cpu_carrier(values, dtype, mutable=mutable),
        FileBacked(dtype=dtype),
    )


def make_tensor(carrier):
    return Tensor(carrier, 0, flat_layout(carrier.size()))


def values(tensor):
    return [tensor[index] for index in range(tensor.size())]


def adapter_for(tensor):
    adapter = tensor.autograd_ctx
    assert isinstance(adapter, EvictableOperation)
    return adapter


def evictable_carrier(tensor):
    carrier = tensor.carrier
    assert isinstance(carrier, Evictable)
    return carrier


def test_evictable_public_exports():
    assert sw.Evictable is Evictable
    assert sw.EvictableOperation is EvictableOperation


def test_evictable_constructor_exposes_promoted_hierarchy():
    primary = make_cpu_carrier([1.0, 2.0])
    secondary = FileBacked(dtype=DType.Float32)

    carrier = Evictable(primary, secondary)

    assert carrier.primary is primary
    assert carrier.secondary is secondary
    assert carrier.size() == 2
    assert carrier.dtype() is DType.Float32
    assert carrier.is_mutable()
    assert primary.is_owned()
    assert secondary.is_owned()
    assert not primary.is_mutable()
    assert not secondary.is_mutable()
    assert not carrier.is_evicted()
    assert carrier[1] == 2.0


@pytest.mark.parametrize(
    ("primary", "secondary", "error", "message"),
    [
        (object(), FileBacked(), TypeError, "primary"),
        (Generic([1.0]), object(), TypeError, "secondary"),
        (Generic([1.0]), Generic([0.0], dtype=DType.Any), TypeError, "dtypes"),
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
    with pytest.raises(RuntimeError, match=r"primary.*released"):
        Evictable(released_primary, Generic([0.0]))

    released_secondary = Generic([0.0])
    released_secondary.release()
    with pytest.raises(RuntimeError, match=r"secondary.*released"):
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
    carrier = Evictable(primary, secondary)

    del carrier
    gc.collect()

    assert not primary.is_owned()
    assert not secondary.is_owned()
    primary[0] = 2.0
    assert primary[0] == 2.0


def test_owned_tier_aliases_are_read_only_and_cannot_be_released():
    primary = Generic([1.0])
    secondary = Generic([0.0])
    carrier = Evictable(primary, secondary)

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

    assert carrier.version == 0
    assert primary.version == 0
    assert carrier[0] == 1.0


def test_owned_tier_cannot_be_moved_directly():
    primary = Generic([1.0])
    carrier = Evictable(primary, Generic([0.0]))
    destination = Generic([0.0])

    with pytest.raises(RuntimeError, match="cannot be moved directly"):
        sw.move(make_tensor(primary), destination)

    assert carrier[0] == 1.0
    assert destination[0] == 0.0


def test_ownership_guards_compose_for_nested_evictable_carrier():
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
    carrier = Evictable(primary, Generic([0.0]))

    carrier[0] = 2.0

    assert carrier[0] == 2.0
    assert carrier.version == 1
    assert primary.version == 1
    assert not primary.is_mutable()

    carrier.set_value(0, 3.0)

    assert carrier[0] == 3.0
    assert carrier.version == 2
    assert primary.version == 2


def test_evict_and_promote_roundtrip_releases_and_recreates_tiers():
    carrier = make_cpu_evictable([1.0, -2.0, 3.0])
    original_primary = carrier.primary
    original_secondary = carrier.secondary

    carrier.evict()

    assert carrier.is_evicted()
    assert original_primary.is_released()
    assert carrier.secondary is not original_secondary
    assert original_secondary.is_released()
    assert [carrier.secondary[i] for i in range(3)] == [1.0, -2.0, 3.0]
    with pytest.raises(RuntimeError, match="evicted"):
        _ = carrier[0]

    carrier.promote()

    assert not carrier.is_evicted()
    assert carrier.primary is not original_primary
    assert original_secondary.is_released()
    assert [carrier[i] for i in range(3)] == [1.0, -2.0, 3.0]


def test_evict_and_promote_are_idempotent():
    carrier = make_cpu_evictable([1.0])

    assert carrier.promote() is None
    primary = carrier.primary
    assert carrier.promote() is None
    assert carrier.primary is primary

    carrier.evict()
    secondary = carrier.secondary
    assert carrier.evict() is None
    assert carrier.secondary is secondary


def test_elementwise_fallback_moves_generic_hierarchy():
    carrier = Evictable(
        Generic(["a", "b"], dtype=DType.Any),
        Generic([None, None], dtype=DType.Any),
    )

    carrier.evict()
    carrier.promote()

    assert [carrier[i] for i in range(2)] == ["a", "b"]


def test_transition_reallocates_an_undersized_secondary_tier():
    carrier = Evictable(Generic([1.0, 2.0]), Generic([0.0]))

    carrier.evict()

    assert carrier.secondary.size() == 2
    assert [carrier.secondary[i] for i in range(2)] == [1.0, 2.0]


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
    carrier = Evictable(primary, secondary)

    with registered_move_operation(Generic, Generic, FailFirstMove):
        with pytest.raises(OSError, match="boom"):
            carrier.evict()

        assert not carrier.is_evicted()
        assert carrier.primary is primary
        assert carrier.secondary is secondary
        assert not primary.is_released()
        assert not secondary.is_released()
        assert primary.is_owned()
        assert secondary.is_owned()

        carrier.evict()

    assert carrier.is_evicted()
    assert values(make_tensor(carrier.secondary)) == [1.0, 2.0]
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

    carrier = Evictable(Generic([1.0, 2.0]), Generic([]))
    original_primary = carrier.primary
    carrier.evict()
    evicted_secondary = carrier.secondary

    with registered_move_operation(Generic, Generic, FailFirstMove):
        with pytest.raises(OSError, match="boom"):
            carrier.promote()

        assert carrier.is_evicted()
        assert carrier.primary is original_primary
        assert original_primary.is_released()
        assert original_primary.is_owned()
        assert carrier.secondary is evicted_secondary
        assert not evicted_secondary.is_released()
        assert evicted_secondary.is_owned()

        carrier.promote()

    assert not carrier.is_evicted()
    assert [carrier[index] for index in range(2)] == [1.0, 2.0]
    assert not original_primary.is_owned()


def test_transitions_use_lowered_execution_not_autograd_forward():
    calls = []

    class SpyMove(ElementwiseMoveOperation):
        def forward(self, *inputs):
            calls.append("forward")
            return super().forward(*inputs)

        def _execute_lowered(self, *inputs):
            calls.append("shadowed _execute_lowered")
            raise AssertionError("dynamic lowered execution was called")

        def _forward(self, tensor, destination):
            calls.append("_forward")
            return super()._forward(tensor, destination)

    with registered_move_operation(Generic, Generic, SpyMove):
        carrier = Evictable(Generic([1.0]), Generic([0.0]))
        carrier.evict()
        carrier.promote()

    assert calls == ["_forward", "_forward"]


def test_adapter_uses_sealed_lowered_execution_when_subclass_shadows_method():
    calls = []

    class ShadowedLoweredOperation(sw.Operation):
        def _execute_lowered(self, *inputs):
            calls.append("shadowed _execute_lowered")
            return inputs[0]

        def _forward(self, *inputs):
            calls.append("_forward")
            (tensor,) = inputs
            carrier = tensor.carrier.new_like([7.0])
            return Tensor(carrier, tensor.offset, tensor.layout)

        def backward(self, gradient):
            return (gradient,)

    tensor = make_tensor(Evictable(Generic([1.0]), Generic([0.0])))
    adapter = EvictableOperation(ShadowedLoweredOperation())

    result = adapter.forward(tensor)

    assert calls == ["_forward"]
    assert values(result) == [7.0]
    assert result.autograd_ctx is adapter


def test_transitions_resolve_move_registry_when_each_transition_runs():
    calls = []

    class SpyMove(ElementwiseMoveOperation):
        def _forward(self, tensor, destination):
            calls.append("spy")
            return super()._forward(tensor, destination)

    carrier = Evictable(Generic([1.0]), Generic([0.0]))
    with registered_move_operation(Generic, Generic, SpyMove):
        carrier.evict()
    carrier.promote()

    with registered_move_operation(Generic, Generic, SpyMove):
        later = Evictable(Generic([2.0]), Generic([0.0]))
    later.evict()

    assert calls == ["spy"]


def test_immutable_hierarchy_remains_externally_read_only_after_promotion():
    prototype = make_cpu_carrier([1.0, 2.0])
    primary = prototype.new_like([1.0, 2.0], mutable=False)
    carrier = Evictable(primary, FileBacked(dtype=DType.Float32))

    carrier.evict()
    carrier.promote()

    assert not carrier.is_mutable()
    assert not carrier.primary.is_mutable()
    with pytest.raises(RuntimeError, match="not mutable"):
        carrier.primary[0] = 3.0


def test_immutable_promotion_keeps_the_move_destination_without_copying():
    promotion_destinations = []

    class CaptureDestinationMove(ElementwiseMoveOperation):
        def _forward(self, tensor, destination):
            result = super()._forward(tensor, destination)
            promotion_destinations.append(destination)
            return result

    carrier = Evictable(
        Generic([1.0, 2.0], mutable=False),
        Generic([]),
    )
    with registered_move_operation(Generic, Generic, CaptureDestinationMove):
        carrier.evict()
        carrier.promote()

    assert not carrier.is_mutable()
    assert carrier.primary is promotion_destinations[-1]
    assert not carrier.primary.is_mutable()


def test_eviction_preserves_version_and_writes_increment_it():
    carrier = make_cpu_evictable([1.0, 2.0])
    initial = carrier.version

    carrier.evict()
    carrier.promote()

    assert carrier.version == initial
    carrier[0] = 4.0
    assert carrier.version == initial + 1


def test_release_releases_both_tiers_and_blocks_transitions():
    carrier = make_cpu_evictable([1.0])
    primary = carrier.primary
    secondary = carrier.secondary

    carrier.release()

    assert carrier.is_released()
    assert primary.is_released()
    assert secondary.is_released()
    assert carrier.size() == 0
    with pytest.raises(RuntimeError, match="released"):
        carrier.promote()
    with pytest.raises(RuntimeError, match="released"):
        carrier.evict()


def test_new_like_preserves_hierarchy_and_supports_dtype_change():
    carrier = make_cpu_evictable([1, 2], DType.Int32)

    result = carrier.new_like([1.5, 2.5], dtype=DType.Float32)

    assert isinstance(result, Evictable)
    assert type(result.primary) is CPU
    assert type(result.secondary) is FileBacked
    assert result.secondary.size() == 0
    assert result.dtype() is DType.Float32
    assert [result[i] for i in range(2)] == pytest.approx([1.5, 2.5])


def test_allocate_like_allocates_primary_and_leaves_secondary_lazy():
    carrier = make_cpu_evictable([1, 2], DType.Int32)

    result = carrier.allocate_like(4, mutable=False, dtype=DType.Float32, empty=True)

    assert result.size() == 4
    assert result.primary.size() == 4
    assert result.secondary.size() == 0
    assert result.dtype() is DType.Float32
    assert not result.is_mutable()
    assert not result.primary.is_mutable()
    assert not result.secondary.is_mutable()
    assert result.primary.is_owned()
    assert result.secondary.is_owned()


def test_evicted_data_blocks_writes_operations_and_scatter():
    carrier = make_cpu_evictable([1.0, 2.0])
    tensor = make_tensor(carrier)
    carrier.evict()

    with pytest.raises(RuntimeError, match="evicted"):
        carrier[0] = 3.0
    with pytest.raises(RuntimeError, match="evicted"):
        sw.relu(tensor)
    with pytest.raises(RuntimeError, match="evicted"):
        carrier.scatter(tensor, tensor, tensor.layout)


@pytest.mark.parametrize("evicted", [False, True])
def test_evictable_never_supports_dlpack(evicted):
    tensor = make_tensor(make_cpu_evictable([1.0]))
    if evicted:
        evictable_carrier(tensor).evict()

    with pytest.raises(BufferError, match="not supported"):
        tensor.__dlpack_device__()
    with pytest.raises(BufferError, match="not supported"):
        tensor.__dlpack__()


def test_cpu_operation_adapter_owns_primary_operation_and_original_inputs():
    tensor = make_tensor(make_cpu_evictable([-1.0, 2.0]))

    result = sw.relu(tensor)

    adapter = adapter_for(result)
    assert type(adapter.primary_operation).__name__ == "_CPUReLUOperation"
    assert adapter.inputs() == (tensor,)
    (lowered,) = adapter.primary_operation.inputs()
    assert type(lowered.carrier) is CPU
    assert lowered.layout == tensor.layout
    assert values(result) == [0.0, 2.0]
    assert isinstance(result.carrier, Evictable)


def test_generic_operation_adapter_owns_generic_operation():
    tensor = make_tensor(Evictable(Generic([-1.0, 2.0]), Generic([0.0, 0.0])))

    result = sw.relu(tensor)

    assert type(adapter_for(result).primary_operation) is sw.GenericReLUOperation
    assert values(result) == [0, 2.0]


def test_layout_only_operation_reuses_same_evictable_carrier():
    tensor = make_tensor(make_cpu_evictable([1.0, 2.0, 3.0, 4.0]))

    view = sw.view(tensor, (slice(1, 3),))

    assert view.carrier is tensor.carrier
    adapter = adapter_for(view)
    assert isinstance(adapter.primary_operation, sw.GenericViewOperation)
    assert values(view) == [2.0, 3.0]


def test_allocating_operation_preserves_hierarchy():
    tensor = make_tensor(make_cpu_evictable([1.0, 2.0]))

    result = sw.mul(tensor, 3)

    assert isinstance(result.carrier, Evictable)
    assert type(result.carrier.primary) is CPU
    assert type(result.carrier.secondary) is FileBacked
    assert result.carrier.secondary.size() == 0
    assert values(result) == [3.0, 6.0]

    result.carrier.evict()

    assert result.carrier.secondary.size() == 2


def test_binary_operation_requires_matching_hierarchies():
    lhs = make_tensor(make_cpu_evictable([1.0, 2.0]))
    rhs = make_tensor(
        Evictable(
            make_cpu_carrier([3.0, 4.0]),
            CPU(2, dtype=DType.Float32),
        )
    )

    with pytest.raises(TypeError, match="secondary carriers must match"):
        sw.add(lhs, rhs)

    plain = make_tensor(make_cpu_carrier([3.0, 4.0]))
    with pytest.raises(TypeError, match="backing carriers must match"):
        sw.add(lhs, plain)


def test_adapter_forward_is_single_use():
    tensor = make_tensor(make_cpu_evictable([1.0]))
    adapter = evictable_carrier(tensor).dispatch_op("relu")

    adapter.forward(tensor)

    with pytest.raises(RuntimeError, match="only be called once"):
        adapter.forward(tensor)


def test_adapter_and_primary_operation_keep_exact_dispatch_metadata():
    tensor = make_tensor(make_cpu_evictable([1.0]))

    adapter = evictable_carrier(tensor).dispatch_op("relu")

    assert adapter._operation_name == "relu"
    assert adapter._dispatch_carrier_class is Evictable
    assert adapter.primary_operation._operation_name == "relu"
    assert adapter.primary_operation._dispatch_carrier_class is CPU


def test_no_grad_uses_adapter_without_attaching_graph():
    tensor = make_tensor(make_cpu_evictable([1.0]))

    with sw.no_grad():
        result = sw.relu(tensor)

    assert result.autograd_ctx is None
    assert isinstance(result.carrier, Evictable)


def test_adapter_preserves_generic_ctx_and_cpu_native_state():
    generic = make_tensor(Evictable(Generic([0.0]), Generic([0.0])))
    sigmoid = sw.sigmoid(generic)
    saved = adapter_for(sigmoid).primary_operation.ctx["saved_values"]
    assert saved == [0.5]

    cpu = make_tensor(make_cpu_evictable([2.0]))
    scaled = sw.mul(cpu, 4)
    assert adapter_for(scaled).primary_operation.ctx["scalar"] == 4.0

    matrix_carrier = make_cpu_evictable([1.0, 2.0, 3.0, 4.0])
    matrix = Tensor(
        matrix_carrier,
        0,
        Layout(Shape([2, 2]), Stride([1, 2])),
    )
    reduced = sw.reduce(matrix)
    assert "output_layout" in adapter_for(reduced).primary_operation.ctx


def test_backward_returns_evictable_gradients():
    tensor = make_tensor(make_cpu_evictable([-1.0, 2.0]))
    result = sw.relu(tensor)
    gradient = make_tensor(make_cpu_evictable([3.0, 4.0]))

    result.backward(gradient)

    assert tensor.grad is not None
    assert isinstance(tensor.grad.carrier, Evictable)
    assert type(tensor.grad.carrier.primary) is CPU
    assert type(tensor.grad.carrier.secondary) is FileBacked
    assert values(tensor.grad) == [0.0, 4.0]


def test_backward_refreshes_primary_inputs_after_evict_promote():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.pow(tensor, 3)
    adapter = adapter_for(result)
    original_lowered = adapter.primary_operation.inputs()[0]

    evictable_carrier(tensor).evict()
    evictable_carrier(tensor).promote()
    result.backward(make_tensor(make_cpu_evictable([1.0])))

    refreshed = adapter.primary_operation.inputs()[0]
    assert refreshed.carrier is evictable_carrier(tensor).primary
    assert refreshed.carrier is not original_lowered.carrier
    assert values(tensor.grad) == pytest.approx([12.0])


def test_backward_does_not_require_the_operation_result_to_be_promoted():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.mul(tensor, 3)
    evictable_carrier(result).evict()

    result.backward(make_tensor(make_cpu_evictable([1.0])))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([3.0])


def test_backward_fails_if_required_input_remains_evicted():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.pow(tensor, 3)
    evictable_carrier(tensor).evict()

    with pytest.raises(RuntimeError, match="evicted"):
        result.backward(make_tensor(make_cpu_evictable([1.0])))


def test_mutation_after_forward_still_fails_version_validation():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.pow(tensor, 3)
    tensor[0] = 4.0

    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(make_tensor(make_cpu_evictable([1.0])))


def test_primary_alias_cannot_silently_change_saved_autograd_input():
    primary = Generic([2.0])
    tensor = make_tensor(Evictable(primary, Generic([0.0])))
    result = sw.pow(tensor, 3)

    with pytest.raises(RuntimeError, match="not mutable"):
        primary[0] = 4.0

    result.backward(make_tensor(Evictable(Generic([1.0]), Generic([0.0]))))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([12.0])


def test_operation_results_claim_their_child_storage():
    tensor = make_tensor(make_cpu_evictable([2.0]))

    result = sw.mul(tensor, 3)
    result_carrier = evictable_carrier(result)

    assert result_carrier.primary.is_owned()
    assert result_carrier.secondary.is_owned()
    with pytest.raises(RuntimeError, match="not mutable"):
        result_carrier.primary[0] = 7.0


def test_repeated_backward_reuses_state_and_accumulates_gradients():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.pow(tensor, 3)

    result.backward(
        make_tensor(make_cpu_evictable([1.0])),
        retain_graph=True,
    )
    result.backward(make_tensor(make_cpu_evictable([1.0])))

    assert tensor.grad is not None
    assert values(tensor.grad) == pytest.approx([24.0])


def test_backward_frees_the_visible_evictable_adapter():
    tensor = make_tensor(make_cpu_evictable([2.0]))
    result = sw.pow(tensor, 3)
    adapter = adapter_for(result)
    gradient = make_tensor(make_cpu_evictable([1.0]))

    result.backward(gradient)

    assert adapter.inputs() == ()
    assert adapter.ctx == {}
    assert adapter._autograd_state_freed
    with pytest.raises(
        RuntimeError,
        match=r"backward through the graph a second time.*retain_graph=True",
    ):
        result.backward(make_tensor(make_cpu_evictable([1.0])))


def test_scalar_implicit_backward_uses_evictable_gradient():
    tensor = make_tensor(make_cpu_evictable([3.0]))
    result = sw.mul(tensor, 2)

    result.backward()

    assert tensor.grad is not None
    assert isinstance(tensor.grad.carrier, Evictable)
    assert values(tensor.grad) == [2.0]


# --- Capabilities a hierarchy generates for itself ---------------------------
#
# What an Evictable can execute depends on the carriers it was handed, so it is
# a dependent carrier: it freezes the plans its own primary executes and whose
# results both tiers could store.


class _SingleDTypeCarrier(sw.Carrier):
    """A list-backed carrier that can allocate only the dtype it was built for.

    It overrides no storage-dtype hook, so it gets ``Carrier``'s conservative
    default. That makes it a secondary tier narrower than any shipped one,
    which is what distinguishes "the primary can execute this plan" from "this
    hierarchy could keep the result".
    """

    def __init__(self, dtype, size=0, *, mutable=True):
        super().__init__()
        self._dtype = dtype
        self._values = [0 if dtype is DType.Int32 else 0.0] * size
        self._mutable = mutable
        self.allocations = []

    def size(self):
        return len(self._values)

    def dtype(self):
        return self._dtype

    def _is_mutable(self):
        return self._mutable

    def get_value(self, index):
        return self._values[index]

    def set_value(self, index, value):
        self._values[index] = value

    def new_like(self, values, *, mutable=True, dtype=None):
        materialized = list(values)
        carrier = _SingleDTypeCarrier(
            self._dtype if dtype is None else dtype, mutable=mutable
        )
        carrier._values = materialized
        return carrier

    def allocate_like(self, size, *, mutable=True, dtype=None, empty=False):
        requested = self._dtype if dtype is None else dtype
        self.allocations.append(requested)
        return _SingleDTypeCarrier(requested, size, mutable=mutable)

    def scatter(self, to_scatter, scatter_onto, mapping, mapping_offset=0):
        raise NotImplementedError


def cpu_hierarchy(dtype, values=(1, 2, 3, 4)):
    return Evictable(make_cpu_carrier(list(values), dtype), FileBacked(dtype=dtype))


def keepable(primary, secondary):
    """Return the primary's plans whose result both tiers could store."""
    return tuple(
        entry
        for entry in primary.operation_capabilities()
        if primary.supports_storage_dtype(entry.output)
        and secondary.supports_storage_dtype(entry.output)
    )


def test_a_hierarchy_advertises_what_its_primary_executes_and_it_can_keep():
    primary = make_cpu_carrier([1.0, 2.0, 3.0, 4.0])
    secondary = FileBacked(dtype=DType.Float32)
    expected = keepable(primary, secondary)

    hierarchy = Evictable(primary, secondary)

    assert hierarchy.operation_capabilities() == expected
    assert expected
    # The class declares nothing: this answer belongs to the instance.
    assert (
        sw.carriers.operation_capability.capabilities_for_carrier_class(Evictable) == ()
    )


def test_a_hierarchy_drops_plans_whose_result_it_could_not_evict():
    # The primary executes exp(Int32) -> Float32, but this secondary can only
    # store Int32, so the hierarchy could never evict that result and does not
    # advertise the plan.
    primary = make_cpu_carrier([1, 2, 3, 4], DType.Int32)
    hierarchy = Evictable(primary, _SingleDTypeCarrier(DType.Int32, 4))

    outputs = {entry.output for entry in hierarchy.operation_capabilities()}

    assert outputs == {DType.Int32}
    assert DType.Float32 in {entry.output for entry in primary.operation_capabilities()}
    assert hierarchy.operation_capabilities("relu")
    assert hierarchy.operation_capabilities("exp") == ()


def test_two_hierarchies_of_one_class_advertise_different_plans():
    narrow = Evictable(
        make_cpu_carrier([1, 2, 3, 4], DType.Int32),
        _SingleDTypeCarrier(DType.Int32, 4),
    )
    wide = cpu_hierarchy(DType.Int32)

    assert narrow.operation_capabilities() != wide.operation_capabilities()
    assert set(narrow.operation_capabilities()) < set(wide.operation_capabilities())


def test_a_nested_hierarchy_composes_through_its_primary_instance_snapshot():
    inner = Evictable(
        make_cpu_carrier([1, 2, 3, 4], DType.Int32),
        _SingleDTypeCarrier(DType.Int32, 4),
    )

    outer = Evictable(inner, _SingleDTypeCarrier(DType.Int32, 4))

    # The outer hierarchy asked its primary, which is itself dependent, so the
    # inner instance's narrowing carries through without either one inspecting
    # a carrier class.
    assert outer.operation_capabilities() == inner.operation_capabilities()


def test_a_hierarchy_executes_every_plan_it_advertises():
    # Advertised support and executable support are one set, through real
    # tensors: this is what the old empty Evictable declaration got wrong, and
    # it is why relu on an Int32 hierarchy now runs instead of being reported
    # as unsupported.
    hierarchy = cpu_hierarchy(DType.Float32)
    advertised = hierarchy.operation_capabilities()
    executed = 0

    for capability in advertised:
        arguments = []
        for operand in capability.operands:
            if operand.role.value == "tensor":
                carrier = cpu_hierarchy(operand.dtype)
                layout = (
                    Layout(Shape([2, 2]), Stride([1, 2]))
                    if capability.operation in ("reduce", "matmul")
                    else flat_layout(4)
                )
                arguments.append(Tensor(carrier, 0, layout))
            else:
                arguments.append(3 if operand.convert_to is DType.Int32 else 0.5)
        result = getattr(sw, capability.operation)(*arguments)
        executed += 1

        assert result.dtype() is capability.output
        assert isinstance(result.carrier, Evictable)
    assert executed == len(advertised)


def test_an_unadvertised_plan_is_refused_before_any_work():
    secondary = _SingleDTypeCarrier(DType.Int32, 4)
    hierarchy = Evictable(make_cpu_carrier([1, 2, 3, 4], DType.Int32), secondary)
    tensor = make_tensor(hierarchy)
    version = hierarchy.version
    secondary.allocations.clear()

    with pytest.raises(NotImplementedError, match="Evictable declares no"):
        sw.exp(tensor)

    # Nothing was lowered, allocated, or computed: the refusal happened at the
    # hierarchy's own gate, not after its primary produced a result.
    assert secondary.allocations == []
    assert hierarchy.version == version
    assert hierarchy.is_evicted() is False


def test_capabilities_are_structural_across_residency_and_release():
    hierarchy = cpu_hierarchy(DType.Int32)
    advertised = hierarchy.operation_capabilities()

    hierarchy.evict()
    evicted = hierarchy.operation_capabilities()
    hierarchy.promote()
    promoted = hierarchy.operation_capabilities()
    hierarchy.release()

    assert evicted == advertised
    assert promoted == advertised
    assert hierarchy.operation_capabilities() == advertised


def test_results_and_gradients_advertise_their_own_capabilities():
    tensor = make_tensor(cpu_hierarchy(DType.Float32, (2.0, 3.0, 4.0, 5.0)))
    expected = evictable_carrier(tensor).operation_capabilities()

    result = sw.mul(tensor, 2)
    result.backward(make_tensor(cpu_hierarchy(DType.Float32, (1.0, 1.0, 1.0, 1.0))))

    assert evictable_carrier(result).operation_capabilities() == expected
    assert tensor.grad is not None
    assert evictable_carrier(tensor.grad).operation_capabilities() == expected


def test_a_legacy_opaque_hierarchy_keeps_its_documented_behavior():
    # Legacy opaque storage is outside simple-dtype planning: this hierarchy
    # resolves no plan for its own tensors, so the gate does not apply and the
    # documented legacy path still runs. Its advertised set still describes the
    # implementation rather than the dtype this instance happens to hold.
    primary = Generic([1.0, 2.0])
    secondary = Generic([0.0, 0.0])
    expected = keepable(primary, secondary)
    hierarchy = Evictable(primary, secondary)
    tensor = Tensor(hierarchy, 0, flat_layout(2))

    assert hierarchy.dtype() is DType.Floating
    assert hierarchy.operation_capabilities() == expected

    result = sw.relu(tensor)

    assert isinstance(result.carrier, Evictable)
    assert result.dtype() is DType.Floating
    assert values(result) == [1.0, 2.0]


def test_a_failed_capability_generation_leaves_no_owned_tiers():
    class _Unaskable(_SingleDTypeCarrier):
        """A tier that cannot answer the structural question generation asks."""

        def _supports_storage_dtype(self, dtype):
            raise RuntimeError("storage support is unavailable")

    primary = make_cpu_carrier([1.0])
    secondary = _Unaskable(DType.Float32, 1)

    with pytest.raises(RuntimeError, match="storage support is unavailable"):
        Evictable(primary, secondary)

    # Generation is the last step of construction, and its failure leaves no
    # usable hierarchy, so neither tier stays locked to an object nobody got.
    assert not primary.is_owned()
    assert not secondary.is_owned()
    assert primary.is_mutable()


@pytest.mark.parametrize(
    ("operation", "extra_tensor", "message"),
    [
        ("relu", True, "relu takes 1 operands, got 2"),
        ("mul", True, "scalar must be a real Python number"),
        ("pow", True, "exponent must be a real Python number"),
        ("add", False, "add takes 2 operands, got 1"),
    ],
    ids=["arity", "scalar-position", "exponent-position", "missing-operand"],
)
def test_an_operand_shape_the_operation_does_not_accept_is_refused_at_the_gate(
    operation, extra_tensor, message
):
    # A registered operation is always planned, so operands its shape does not
    # accept are refused by the central resolver at this hierarchy's own gate
    # rather than passed down to be diagnosed by whichever primary operation
    # happened to receive them.
    def narrow_hierarchy():
        tier = _SingleDTypeCarrier(DType.Int32, 4)
        return Evictable(make_cpu_carrier([1, 2, 3, 4], DType.Int32), tier), tier

    hierarchy, secondary = narrow_hierarchy()
    tensor = make_tensor(hierarchy)
    arguments = (
        (tensor, make_tensor(narrow_hierarchy()[0])) if extra_tensor else (tensor,)
    )
    secondary.allocations.clear()

    with pytest.raises(TypeError, match=message):
        hierarchy.dispatch_op(operation).forward(*arguments)

    assert secondary.allocations == []
    assert hierarchy.version == 0


def test_an_operation_the_policy_does_not_describe_is_not_planned():
    # Only a registered operation is planned; anything else keeps whatever the
    # primary carrier does with it, which for an unknown name is its own
    # dispatch refusal.
    hierarchy = cpu_hierarchy(DType.Int32)

    with pytest.raises(NotImplementedError, match="no_such_operation"):
        hierarchy.dispatch_op("no_such_operation")
