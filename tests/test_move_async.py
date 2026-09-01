from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave.carriers.extension import (
    PreparedTransfer,
    TransferProvider,
    TransferRequest,
    Unsupported,
)
from strideweave.carriers.move import (
    AwaitMove,
    AwaitResult,
    ElementwiseMoveOperation,
    register_move_operation,
)


class PendingCompletion:
    def __init__(self, *, on_wait: Callable[[], None] | None = None) -> None:
        self._event = threading.Event()
        self._error: BaseException | None = None
        self._on_wait = on_wait
        self.wait_calls = 0

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self.wait_calls += 1
        self._event.wait()
        if self._on_wait is not None:
            self._on_wait()
        if self._error is not None:
            raise self._error

    def finish(self, error: BaseException | None = None) -> None:
        self._error = error
        self._event.set()


def fresh_carrier(name: str) -> type[sw.Carrier]:
    return type(name, (sw.Carrier,), {})


def storage_provider(*, fallback: bool = False) -> sw.StorageProvider:
    def allocate(carrier: Any, slots: int) -> None:
        carrier.values = [None] * slots

    def release(carrier: Any) -> None:
        carrier.values = []

    def size(carrier: Any) -> int:
        return len(carrier.values)

    def read(carrier: Any, index: int) -> object:
        return carrier.values[index]

    def write(carrier: Any, index: int, value: object) -> None:
        carrier.values[index] = value

    return sw.StorageProvider(
        lambda _carrier, dtype: dtype in (sw.DType.Float32, sw.DType.Int32),
        size,
        allocate,
        release,
        read,
        write,
        elementwise_fallback=fallback,
    )


def make_tensor(
    carrier_class: type[sw.Carrier],
    values: list[float] | list[int],
    *,
    dtype: sw.SimpleDType = sw.DType.Float32,
    layout: sw.Layout | None = None,
) -> sw.Tensor:
    if layout is None:
        layout = sw.Layout(sw.Shape(len(values)), sw.Stride(1))
    carrier = carrier_class(len(values), dtype=dtype)
    for index, value in enumerate(values):
        carrier[index] = value
    return sw.Tensor(carrier, 0, layout)


def copying_prepare(
    request: TransferRequest,
) -> PreparedTransfer:
    def submit() -> None:
        source = cast(Any, request.tensor.carrier)
        destination = cast(Any, request.destination)
        for index in range(request.physical_span):
            destination.values[index] = source.values[request.tensor.offset + index]

    return PreparedTransfer(submit)


def route(
    source: type[sw.Carrier],
    destination: type[sw.Carrier],
    route_id: str,
    prepare: Callable[[TransferRequest], PreparedTransfer | Unsupported],
) -> sw.TransferRoute:
    return sw.TransferRoute(
        source,
        destination,
        route_id,
        TransferProvider(prepare),
    )


def register_pair(
    source: type[sw.Carrier],
    destination: type[sw.Carrier],
    forward_prepare: Callable[
        [TransferRequest], PreparedTransfer | Unsupported
    ] = copying_prepare,
    *,
    reverse: bool = True,
) -> None:
    sw.register_carrier_definition(
        source,
        sw.CarrierDefinition(
            storage_provider(),
            transfers=(route(source, destination, "forward", forward_prepare),),
        ),
    )
    reverse_routes = (
        (route(destination, source, "reverse", copying_prepare),) if reverse else ()
    )
    sw.register_carrier_definition(
        destination,
        sw.CarrierDefinition(storage_provider(), transfers=reverse_routes),
    )


def test_async_exports_are_identity_preserving_and_not_coroutines() -> None:
    import strideweave.carriers.move as movement

    assert sw.AwaitResult is AwaitResult is movement.AwaitResult
    assert sw.AwaitMove is AwaitMove is movement.AwaitMove
    assert sw.move_async is movement.move_async
    for name in ("AwaitResult", "AwaitMove", "move_async"):
        assert name in sw.__all__
        assert name in movement.__all__
    with pytest.raises(TypeError, match="created by the framework"):
        AwaitResult()
    assert not hasattr(AwaitResult, "__await__")
    assert not hasattr(AwaitResult, "await")


def test_pending_route_returns_immediately_and_wait_is_identity_stable() -> None:
    source_class = fresh_carrier("PendingSource")
    destination_class = fresh_carrier("PendingDestination")
    completion = PendingCompletion()
    submitted = threading.Event()

    def prepare(request: TransferRequest) -> PreparedTransfer:
        prepared = copying_prepare(request)

        def submit() -> PendingCompletion:
            cast(Callable[[], None], prepared.submit)()
            submitted.set()
            return completion

        return PreparedTransfer(submit)

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [1.0, 2.0])
    destination = destination_class(2, dtype=sw.DType.Float32)
    handle = sw.move_async(source, destination)

    assert submitted.is_set()
    assert isinstance(handle, AwaitMove)
    assert not handle.done
    assert not source.carrier.is_released()
    assert source.carrier.is_owned()
    assert destination.is_owned()

    results: list[sw.Tensor] = []
    waiters = [
        threading.Thread(target=lambda: results.append(handle.wait())) for _ in range(2)
    ]
    for waiter in waiters:
        waiter.start()
    completion.finish()
    for waiter in waiters:
        waiter.join()

    assert handle.done
    assert len(results) == 2
    assert results[0] is results[1] is handle.wait()
    assert [results[0][index] for index in range(2)] == [1.0, 2.0]
    assert source.carrier.is_released()
    assert not destination.is_owned()


def test_pending_move_rejects_mutation_release_and_overlapping_movement() -> None:
    source_class = fresh_carrier("ProtectedSource")
    destination_class = fresh_carrier("ProtectedDestination")
    completion = PendingCompletion()

    def prepare(request: TransferRequest) -> PreparedTransfer:
        prepared = copying_prepare(request)
        return PreparedTransfer(lambda: (prepared.submit(), completion)[1])

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [3.0])
    destination = destination_class(1, dtype=sw.DType.Float32)
    handle = sw.move_async(source, destination)

    with pytest.raises(RuntimeError, match="mutable"):
        source[0] = 4.0
    with pytest.raises(RuntimeError, match="owned"):
        source.carrier.release()
    with pytest.raises(RuntimeError, match="owned"):
        sw.move_async(source, destination_class(1, dtype=sw.DType.Float32))
    with pytest.raises(RuntimeError, match="mutable"):
        destination[0] = 5.0
    pending_destination_tensor = sw.Tensor(
        destination, 0, sw.Layout(sw.Shape(1), sw.Stride(1))
    )
    with pytest.raises(RuntimeError, match="owned"):
        sw.move_async(
            pending_destination_tensor,
            source_class(1, dtype=sw.DType.Float32),
        )

    completion.finish()
    assert handle.wait()[0] == 3.0
    assert destination.is_mutable()


def test_post_submission_failure_is_same_exception_and_restores_ownership() -> None:
    source_class = fresh_carrier("FailureSource")
    destination_class = fresh_carrier("FailureDestination")
    completion = PendingCompletion()

    register_pair(
        source_class,
        destination_class,
        lambda request: PreparedTransfer(lambda: completion),
    )
    source = make_tensor(source_class, [7.0])
    destination = destination_class(1, dtype=sw.DType.Float32)
    handle = sw.move_async(source, destination)
    failure = RuntimeError("device copy failed")
    completion.finish(failure)

    observed: list[BaseException] = []
    for _ in range(2):
        try:
            handle.wait()
        except BaseException as error:
            observed.append(error)
    assert observed == [failure, failure]
    assert handle.done
    assert not source.carrier.is_released()
    assert source[0] == 7.0
    assert not source.carrier.is_owned()
    assert not destination.is_owned()
    assert destination.is_mutable()


def test_source_release_occurs_only_after_provider_completion() -> None:
    source_class = fresh_carrier("ReleaseOrderSource")
    destination_class = fresh_carrier("ReleaseOrderDestination")
    source_holder: list[sw.Tensor] = []

    def observe_live_source() -> None:
        assert not source_holder[0].carrier.is_released()

    completion = PendingCompletion(on_wait=observe_live_source)

    def prepare(request: TransferRequest) -> PreparedTransfer:
        prepared = copying_prepare(request)
        return PreparedTransfer(lambda: (prepared.submit(), completion)[1])

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [2.0])
    source_holder.append(source)
    handle = sw.move_async(source, destination_class(1, dtype=sw.DType.Float32))
    completion.finish()
    assert handle.wait()[0] == 2.0
    assert source.carrier.is_released()


def test_dropped_handle_does_not_cancel_or_block_pending_move() -> None:
    source_class = fresh_carrier("DroppedSource")
    destination_class = fresh_carrier("DroppedDestination")
    completion = PendingCompletion()
    register_pair(
        source_class,
        destination_class,
        lambda request: PreparedTransfer(lambda: completion),
    )
    source = make_tensor(source_class, [1.0])
    handle = sw.move_async(source, destination_class(1, dtype=sw.DType.Float32))
    reference = weakref.ref(handle)
    del handle
    gc.collect()
    assert reference() is None
    assert not source.carrier.is_released()

    completion.finish()
    deadline = time.monotonic() + 2
    while not source.carrier.is_released() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert source.carrier.is_released()


def test_pending_move_retains_prepared_provider_resources() -> None:
    source_class = fresh_carrier("ResourceSource")
    destination_class = fresh_carrier("ResourceDestination")
    completion = PendingCompletion()
    resource_references: list[weakref.ReferenceType[object]] = []

    class ProviderResource:
        pass

    def prepare(request: TransferRequest) -> PreparedTransfer:
        resource = ProviderResource()
        resource_references.append(weakref.ref(resource))
        prepared = copying_prepare(request)

        def submit() -> PendingCompletion:
            assert resource is not None
            prepared.submit()
            return completion

        return PreparedTransfer(submit)

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [6], dtype=sw.DType.Int32)
    handle = sw.move_async(source, destination_class(1, dtype=sw.DType.Int32))

    gc.collect()
    assert resource_references[0]() is not None
    completion.finish()
    assert handle.wait()[0] == 6


def test_slow_preparation_protects_participants_from_concurrent_mutation() -> None:
    source_class = fresh_carrier("PreparingSource")
    destination_class = fresh_carrier("PreparingDestination")
    preparing = threading.Event()
    continue_preparing = threading.Event()

    def prepare(request: TransferRequest) -> PreparedTransfer:
        preparing.set()
        continue_preparing.wait()
        return copying_prepare(request)

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [3], dtype=sw.DType.Int32)
    destination = destination_class(1, dtype=sw.DType.Int32)
    handles: list[AwaitMove[sw.Tensor]] = []
    errors: list[BaseException] = []

    def initiate() -> None:
        try:
            handles.append(sw.move_async(source, destination))
        except BaseException as error:
            errors.append(error)

    initiator = threading.Thread(target=initiate)
    initiator.start()
    assert preparing.wait(timeout=2)
    with pytest.raises(RuntimeError, match="mutable"):
        source[0] = 4
    with pytest.raises(RuntimeError, match="owned"):
        source.carrier.release()
    with pytest.raises(RuntimeError, match="mutable"):
        destination[0] = 4
    continue_preparing.set()
    initiator.join(timeout=2)

    assert not initiator.is_alive()
    assert errors == []
    assert handles[0].wait()[0] == 3


def test_provider_callbacks_do_not_create_nested_autograd_nodes() -> None:
    source_class = fresh_carrier("NoNestedGraphSource")
    destination_class = fresh_carrier("NoNestedGraphDestination")
    observed_contexts: list[object | None] = []
    probe = sw.Tensor(
        sw.Generic([2.0], dtype=sw.DType.Float32),
        0,
        sw.Layout(sw.Shape(1), sw.Stride(1)),
    )
    increment = sw.Tensor(
        sw.Generic([1.0], dtype=sw.DType.Float32),
        0,
        sw.Layout(sw.Shape(1), sw.Stride(1)),
    )

    def execute_probe() -> None:
        observed_contexts.append(sw.add(probe, increment).autograd_ctx)

    completion = PendingCompletion(on_wait=execute_probe)

    def prepare(request: TransferRequest) -> PreparedTransfer:
        execute_probe()
        prepared = copying_prepare(request)

        def submit() -> PendingCompletion:
            execute_probe()
            prepared.submit()
            return completion

        return PreparedTransfer(submit)

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [2.0])
    handle = sw.move_async(source, destination_class(1, dtype=sw.DType.Float32))
    completion.finish()
    result = handle.wait()

    assert observed_contexts == [None, None, None]
    assert result.autograd_ctx is not None


def test_sequence_same_carrier_and_unsupported_route_errors_are_pre_submission() -> (
    None
):
    source_class = fresh_carrier("PreflightSource")
    destination_class = fresh_carrier("PreflightDestination")
    submissions: list[str] = []
    sw.register_carrier_definition(
        source_class,
        sw.CarrierDefinition(
            storage_provider(),
            transfers=(
                route(
                    source_class,
                    destination_class,
                    "reject",
                    lambda request: Unsupported("alignment"),
                ),
            ),
        ),
    )
    sw.register_carrier_definition(
        destination_class, sw.CarrierDefinition(storage_provider())
    )
    source = make_tensor(source_class, [1, 2], dtype=sw.DType.Int32)
    destination = destination_class(2, dtype=sw.DType.Int32)
    source_version = source.carrier.version
    destination_version = destination.version

    with pytest.raises(TypeError, match="tensor must be a Tensor"):
        sw.move_async([source], destination)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Carrier instance"):
        sw.move_async(source, [destination])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="own carrier"):
        sw.move_async(source, source.carrier)
    with pytest.raises(NotImplementedError, match="alignment"):
        sw.move_async(source, destination)

    assert submissions == []
    assert source.carrier.version == source_version
    assert destination.version == destination_version
    assert not source.carrier.is_owned()
    assert not destination.is_owned()


def test_forward_route_without_reverse_fails_before_submission() -> None:
    source_class = fresh_carrier("OneWaySource")
    destination_class = fresh_carrier("OneWayDestination")
    submissions: list[TransferRequest] = []

    def prepare(request: TransferRequest) -> PreparedTransfer:
        return PreparedTransfer(lambda: submissions.append(request))

    register_pair(source_class, destination_class, prepare, reverse=False)
    source = make_tensor(source_class, [1.0])
    destination = destination_class(1, dtype=sw.DType.Float32)
    with pytest.raises(NotImplementedError, match=r"reverse|transfer route"):
        sw.move_async(source, destination)
    assert submissions == []
    assert not source.carrier.is_owned()
    assert not destination.is_owned()


def test_definition_elementwise_fallback_requires_both_exact_permissions() -> None:
    source_class = fresh_carrier("FallbackSource")
    destination_class = fresh_carrier("FallbackDestination")
    sw.register_carrier_definition(
        source_class, sw.CarrierDefinition(storage_provider(fallback=True))
    )
    sw.register_carrier_definition(
        destination_class, sw.CarrierDefinition(storage_provider(fallback=True))
    )
    source = make_tensor(source_class, [4, 5], dtype=sw.DType.Int32)
    moved = sw.move_async(source, destination_class(2, dtype=sw.DType.Int32)).wait()
    assert [moved[index] for index in range(2)] == [4, 5]

    refusing_source = fresh_carrier("RefusingFallbackSource")
    refusing_destination = fresh_carrier("RefusingFallbackDestination")
    sw.register_carrier_definition(
        refusing_source, sw.CarrierDefinition(storage_provider(fallback=True))
    )
    sw.register_carrier_definition(
        refusing_destination, sw.CarrierDefinition(storage_provider())
    )
    with pytest.raises(
        NotImplementedError,
        match=r"RefusingFallbackSource, RefusingFallbackDestination",
    ):
        sw.move_async(
            make_tensor(refusing_source, [1], dtype=sw.DType.Int32),
            refusing_destination(1, dtype=sw.DType.Int32),
        )


def test_rejected_definition_route_uses_mutually_permitted_fallback() -> None:
    source_class = fresh_carrier("RejectedFallbackSource")
    destination_class = fresh_carrier("RejectedFallbackDestination")
    sw.register_carrier_definition(
        source_class,
        sw.CarrierDefinition(
            storage_provider(fallback=True),
            transfers=(
                route(
                    source_class,
                    destination_class,
                    "alignment-specific",
                    lambda request: Unsupported("unaligned"),
                ),
            ),
        ),
    )
    sw.register_carrier_definition(
        destination_class,
        sw.CarrierDefinition(storage_provider(fallback=True)),
    )
    source = make_tensor(source_class, [4, 5], dtype=sw.DType.Int32)

    moved = sw.move_async(source, destination_class(2, dtype=sw.DType.Int32)).wait()

    assert [moved[index] for index in range(2)] == [4, 5]


def test_definition_route_allocates_an_empty_destination_to_layout_cosize() -> None:
    source_class = fresh_carrier("EmptyDestinationSource")
    destination_class = fresh_carrier("EmptyDestinationTarget")
    register_pair(source_class, destination_class)
    layout = sw.Layout(sw.Shape(3), sw.Stride(2))
    source = make_tensor(
        source_class,
        [1, 0, 2, 0, 3],
        dtype=sw.DType.Int32,
        layout=layout,
    )
    destination = destination_class(0, dtype=sw.DType.Int32)

    moved = sw.move_async(source, destination).wait()

    assert destination.size() == layout.cosize == 5
    assert moved.layout == layout
    assert [moved[index] for index in range(3)] == [1, 2, 3]


@pytest.mark.parametrize("failure", ["submit", "completion", "result"])
def test_post_submission_contract_failures_are_terminal_handles(
    failure: str,
) -> None:
    source_class = fresh_carrier(f"ContractFailureSource{failure}")
    destination_class = fresh_carrier(f"ContractFailureTarget{failure}")

    class WrongResultCompletion:
        done = True

        def wait(self) -> object:
            return object()

    def prepare(request: TransferRequest) -> PreparedTransfer:
        del request
        if failure == "submit":

            def raise_on_submit() -> None:
                raise RuntimeError("submit failed")

            return PreparedTransfer(raise_on_submit)
        if failure == "completion":
            return PreparedTransfer(lambda: cast(Any, object()))
        return PreparedTransfer(lambda: cast(Any, WrongResultCompletion()))

    register_pair(source_class, destination_class, prepare)
    source = make_tensor(source_class, [1.0])
    destination = destination_class(1, dtype=sw.DType.Float32)
    handle = sw.move_async(source, destination)

    expected = RuntimeError if failure == "submit" else TypeError
    with pytest.raises(expected):
        handle.wait()
    assert handle.done
    assert not source.carrier.is_released()
    assert not source.carrier.is_owned()
    assert destination.is_mutable()


def test_routes_are_exact_and_parent_route_is_not_inherited() -> None:
    parent = fresh_carrier("RouteParent")
    child = type("RouteChild", (parent,), {})
    destination = fresh_carrier("RouteDestination")
    parent_prepares: list[TransferRequest] = []

    def prepare(request: TransferRequest) -> PreparedTransfer:
        parent_prepares.append(request)
        return copying_prepare(request)

    sw.register_carrier_definition(
        parent,
        sw.CarrierDefinition(
            storage_provider(),
            transfers=(route(parent, destination, "parent", prepare),),
        ),
    )
    sw.register_carrier_definition(child, sw.CarrierDefinition(storage_provider()))
    sw.register_carrier_definition(
        destination, sw.CarrierDefinition(storage_provider())
    )
    with pytest.raises(NotImplementedError, match="RouteChild, RouteDestination"):
        sw.move_async(
            make_tensor(child, [1], dtype=sw.DType.Int32),
            destination(1, dtype=sw.DType.Int32),
        )
    assert parent_prepares == []


def test_legacy_registry_rejects_definition_owned_pairs() -> None:
    source_class = fresh_carrier("DefinedRegistrySource")
    destination_class = fresh_carrier("DefinedRegistryDestination")
    register_pair(source_class, destination_class)
    with pytest.raises(RuntimeError, match="TransferRoute"):
        register_move_operation(
            source_class,
            destination_class,
            ElementwiseMoveOperation,
        )


def test_definition_route_autograd_uses_captured_exact_reverse_route() -> None:
    source_class = fresh_carrier("AutogradMoveSource")
    destination_class = fresh_carrier("AutogradMoveDestination")
    reverse_calls: list[TransferRequest] = []

    def reverse_prepare(request: TransferRequest) -> PreparedTransfer:
        reverse_calls.append(request)
        return copying_prepare(request)

    sw.register_carrier_definition(
        source_class,
        sw.CarrierDefinition(
            storage_provider(),
            transfers=(
                route(source_class, destination_class, "forward", copying_prepare),
            ),
        ),
    )
    sw.register_carrier_definition(
        destination_class,
        sw.CarrierDefinition(
            storage_provider(),
            transfers=(
                route(destination_class, source_class, "reverse", reverse_prepare),
            ),
        ),
    )
    source = make_tensor(source_class, [2.0, 3.0])
    result = sw.move_async(source, destination_class(2, dtype=sw.DType.Float32)).wait()
    assert result.autograd_ctx is not None
    gradient = make_tensor(destination_class, [8.0, 9.0])
    result.backward(gradient)

    assert len(reverse_calls) == 1
    assert source.grad is not None
    assert type(source.grad.carrier) is source_class
    assert [source.grad[index] for index in range(2)] == [8.0, 9.0]
    assert not gradient.carrier.is_released()


def test_move_async_is_not_awaitable() -> None:
    source = sw.Tensor(
        sw.Generic([1.0], dtype=sw.DType.Float32),
        0,
        sw.Layout(sw.Shape(1), sw.Stride(1)),
    )
    handle = sw.move_async(source, sw.Generic([0.0], dtype=sw.DType.Float32))

    async def reject() -> None:
        with pytest.raises(TypeError, match="can't be used in 'await' expression"):
            await cast(Any, handle)

    asyncio.run(reject())
    assert handle.wait()[0] == 1.0
