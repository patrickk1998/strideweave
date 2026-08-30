"""Eager single-Tensor movement with transactional blocking completion."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, cast

from ...core.tensor import Tensor
from ...profiling import _begin_move_profile, _MoveProfileToken
from .._background import _MOVE_BACKGROUND_EXECUTOR
from ..base import Carrier
from ..extension import (
    CarrierDefinition,
    PreparedTransfer,
    ProviderCompletion,
    TransferRequest,
    TransferRoute,
    Unsupported,
    _observe_carrier_definition,
    _provider_size,
    _require_provider_completion,
)
from ..operation_helpers import (
    _as_tensor,
    _canonical_layout_for_shape,
    _logical_values,
    _physical_values_for_layout,
    _require_live_tensor,
    _require_same_shape,
)
from .await_result import AwaitMove, _CompletionState
from .ops import (
    ElementwiseMoveOperation,
    MoveOperation,
    _require_supported_block_operation,
    _validate_move_inputs,
    dispatch_move,
)

_operation = import_module("strideweave._operation")


@dataclass(frozen=True, slots=True)
class _ResolvedMoveRoute:
    source_class: type[Carrier]
    destination_class: type[Carrier]
    route_id: str
    operation_class: type[MoveOperation] | None = None
    transfer_route: TransferRoute | None = None


@dataclass(slots=True)
class _MoveCommit:
    state: _CompletionState[Tensor]
    tensor: Tensor
    destination: Carrier
    operation: MoveOperation
    reverse_route: _ResolvedMoveRoute | None
    build_graph: bool
    captured_version: object
    required_size: int
    source_token: int
    destination_token: int
    profile: _MoveProfileToken
    prepared: PreparedTransfer | None
    completion: ProviderCompletion[object] | None


class _DefinitionMoveOperation(MoveOperation):
    """Autograd boundary for one definition-owned transfer route."""

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        del tensor, destination, output, element_count
        raise RuntimeError("definition transfer copying is provider-owned")


def _pair_name(source_class: type, destination_class: type) -> str:
    return f"({source_class.__name__}, {destination_class.__name__})"


def _definition_elementwise_route(
    source_class: type[Carrier], destination_class: type[Carrier]
) -> _ResolvedMoveRoute | None:
    source_definition = _observe_carrier_definition(source_class)
    destination_definition = _observe_carrier_definition(destination_class)
    if (
        source_definition is None
        or destination_definition is None
        or not source_definition.storage.elementwise_fallback
        or not destination_definition.storage.elementwise_fallback
    ):
        return None
    return _ResolvedMoveRoute(
        source_class,
        destination_class,
        "definition.elementwise",
        operation_class=ElementwiseMoveOperation,
    )


def _resolve_route(
    source_class: type[Carrier], destination_class: type[Carrier]
) -> _ResolvedMoveRoute:
    source_definition = _observe_carrier_definition(source_class)
    destination_definition = _observe_carrier_definition(destination_class)
    if source_definition is not None:
        route = next(
            (
                candidate
                for candidate in source_definition.transfers
                if candidate.source_class is source_class
                and candidate.destination_class is destination_class
            ),
            None,
        )
        if route is not None:
            return _ResolvedMoveRoute(
                source_class,
                destination_class,
                route.route_id,
                transfer_route=route,
            )
        fallback = _definition_elementwise_route(source_class, destination_class)
        if fallback is not None:
            return fallback
        raise NotImplementedError(
            "no definition-owned transfer route or mutually permitted "
            f"elementwise fallback exists for {_pair_name(source_class, destination_class)}"
        )
    if destination_definition is not None:
        raise NotImplementedError(
            "definition-free movement cannot target a definition-backed carrier "
            f"without an exact TransferRoute for {_pair_name(source_class, destination_class)}"
        )
    operation_class = dispatch_move(source_class, destination_class)
    return _ResolvedMoveRoute(
        source_class,
        destination_class,
        f"legacy.{operation_class.__module__}.{operation_class.__qualname__}",
        operation_class=operation_class,
    )


def _validate_route_pair(
    route: _ResolvedMoveRoute, tensor: Tensor, destination: Carrier
) -> None:
    if type(tensor.carrier) is not route.source_class:
        raise TypeError(
            f"move route {route.route_id!r} requires a "
            f"{route.source_class.__name__} source"
        )
    if type(destination) is not route.destination_class:
        raise TypeError(
            f"move route {route.route_id!r} requires a "
            f"{route.destination_class.__name__} destination"
        )
    operation_class = route.operation_class
    if operation_class is not None:
        if (
            operation_class.source_class is not None
            and operation_class.source_class is not route.source_class
        ):
            raise TypeError(
                f"{operation_class.__name__} requires a "
                f"{operation_class.source_class.__name__} source"
            )
        if (
            operation_class.destination_class is not None
            and operation_class.destination_class is not route.destination_class
        ):
            raise TypeError(
                f"{operation_class.__name__} requires a "
                f"{operation_class.destination_class.__name__} destination"
            )


def _prepare_transfer(
    route: _ResolvedMoveRoute, tensor: Tensor, destination: Carrier
) -> tuple[_ResolvedMoveRoute, PreparedTransfer | None]:
    declared = route.transfer_route
    if declared is None:
        return route, None
    request = TransferRequest(
        tensor,
        destination,
        route.source_class,
        route.destination_class,
        tensor.dtype(),
        tensor.layout,
        tensor.layout.cosize,
    )
    with _without_graph():
        prepared = declared.provider.prepare(request)
    if isinstance(prepared, Unsupported):
        fallback = _definition_elementwise_route(
            route.source_class, route.destination_class
        )
        if fallback is not None:
            return fallback, None
        raise NotImplementedError(
            f"transfer route {route.route_id!r} rejected "
            f"{_pair_name(route.source_class, route.destination_class)}: "
            f"{prepared.reason}"
        )
    if not isinstance(prepared, PreparedTransfer):
        raise TypeError(
            "TransferProvider.prepare must return PreparedTransfer or Unsupported"
        )
    return route, prepared


def _can_allocate_empty_destination(
    destination: Carrier, definition: CarrierDefinition | None
) -> bool:
    return definition is not None or callable(getattr(destination, "_allocate", None))


def _preflight_destination_extent(
    destination: Carrier,
    required_size: int,
    definition: CarrierDefinition | None,
) -> None:
    size = destination.size()
    if size < required_size and not (
        size == 0 and _can_allocate_empty_destination(destination, definition)
    ):
        raise ValueError("destination carrier is too small for the tensor layout")


def _allocate_destination(
    destination: Carrier,
    required_size: int,
    definition: CarrierDefinition | None,
) -> None:
    if destination.size() != 0 or required_size == 0:
        return
    if definition is None:
        allocate = getattr(destination, "_allocate")
        result = allocate(required_size)
        if result is not None:
            raise TypeError("destination allocation must return None")
    else:
        result = definition.storage.allocate(destination, required_size)
        if result is not None:
            raise TypeError("StorageProvider.allocate must return None")
        if _provider_size(destination, definition) != required_size:
            raise ValueError(
                "StorageProvider.allocate must establish the requested physical size"
            )


@contextmanager
def _owner_access(
    source: Carrier,
    destination: Carrier,
    source_token: int,
    destination_token: int,
) -> Iterator[None]:
    source._begin_owner_access(source_token)
    try:
        destination._begin_owner_access(destination_token)
        try:
            yield
        finally:
            destination._end_owner_access(destination_token)
    finally:
        source._end_owner_access(source_token)


@contextmanager
def _without_graph() -> Iterator[None]:
    previous = bool(_operation.is_grad_enabled())
    _operation.set_grad_enabled(False)
    try:
        yield
    finally:
        _operation.set_grad_enabled(previous)


def _claim_participants(source: Carrier, destination: Carrier) -> tuple[int, int]:
    source_token = source._claim_ownership()
    try:
        destination_token = destination._claim_ownership()
    except BaseException:
        source._relinquish_ownership(source_token)
        raise
    return source_token, destination_token


def _relinquish_participants(
    source: Carrier,
    destination: Carrier,
    source_token: int,
    destination_token: int,
) -> None:
    source_error: BaseException | None = None
    try:
        source._relinquish_ownership(source_token)
    except BaseException as error:
        source_error = error
    destination._relinquish_ownership(destination_token)
    if source_error is not None:
        raise source_error


def _operation_for_route(route: _ResolvedMoveRoute) -> MoveOperation:
    operation_class = route.operation_class
    if operation_class is None:
        return _DefinitionMoveOperation()
    return operation_class()


def _attach_autograd(
    operation: MoveOperation,
    output: Tensor,
    tensor: Tensor,
    destination: Carrier,
    reverse_route: _ResolvedMoveRoute | None,
    build_graph: bool,
) -> None:
    if not build_graph:
        return
    operation._async_reverse_route = reverse_route
    operation._async_destination_prototype = destination
    operation.store_inputs(tensor)
    output.autograd_ctx = operation


def _validate_completed_move(
    tensor: Tensor,
    destination: Carrier,
    captured_version: object,
    required_size: int,
) -> None:
    if tensor.carrier.is_released():
        raise RuntimeError("move provider released the source before publication")
    if tensor._version_token() != captured_version:
        raise RuntimeError("move source changed while transfer work was pending")
    if destination.is_released():
        raise RuntimeError("move provider released the destination")
    if destination.dtype() is not tensor.dtype():
        raise TypeError("destination dtype changed while transfer work was pending")
    if destination.size() < required_size:
        raise ValueError("destination carrier is too small after transfer completion")


def _commit_success(commit: _MoveCommit) -> None:
    try:
        with _owner_access(
            commit.tensor.carrier,
            commit.destination,
            commit.source_token,
            commit.destination_token,
        ):
            _validate_completed_move(
                commit.tensor,
                commit.destination,
                commit.captured_version,
                commit.required_size,
            )
            output = Tensor(commit.destination, 0, commit.tensor.layout)
            _attach_autograd(
                commit.operation,
                output,
                commit.tensor,
                commit.destination,
                commit.reverse_route,
                commit.build_graph,
            )
            commit.tensor.carrier.release()
        _relinquish_participants(
            commit.tensor.carrier,
            commit.destination,
            commit.source_token,
            commit.destination_token,
        )
    except BaseException as error:
        try:
            _relinquish_participants(
                commit.tensor.carrier,
                commit.destination,
                commit.source_token,
                commit.destination_token,
            )
        except BaseException:
            pass
        commit.profile.completed(False)
        commit.state.fail(error)
        return
    commit.profile.completed(True)
    commit.state.succeed(output)


def _fail_after_submission(commit: _MoveCommit, error: BaseException) -> None:
    try:
        _relinquish_participants(
            commit.tensor.carrier,
            commit.destination,
            commit.source_token,
            commit.destination_token,
        )
    except BaseException as cleanup_error:
        commit.profile.completed(False)
        commit.state.fail(cleanup_error)
    else:
        commit.profile.completed(False)
        commit.state.fail(error)


def _wait_for_provider(
    completion: ProviderCompletion[object],
    commit: _MoveCommit,
) -> None:
    try:
        with _without_graph():
            result = completion.wait()
        if result is not None:
            raise TypeError("transfer completion wait() must return None")
        if type(completion.done) is not bool or not completion.done:
            raise TypeError("transfer completion done must be true after wait()")
    except BaseException as error:
        _fail_after_submission(commit, error)
        return
    _commit_success(commit)


def _run_legacy_copy(
    *,
    state: _CompletionState[Tensor],
    tensor: Tensor,
    destination: Carrier,
    operation: MoveOperation,
    required_size: int,
    source_token: int,
    destination_token: int,
    commit: _MoveCommit,
    launch_ready: threading.Event,
) -> None:
    launch_ready.wait()
    try:
        with _owner_access(
            tensor.carrier,
            destination,
            source_token,
            destination_token,
        ):
            output = Tensor(destination, 0, tensor.layout)
            with _without_graph():
                result = operation._copy(tensor, destination, output, required_size)
            if result is not None:
                raise TypeError("MoveOperation._copy must return None")
    except BaseException as error:
        _fail_after_submission(commit, error)
        return
    _commit_success(commit)


def _move_async_with_route(
    tensor: object,
    destination: object,
    route: _ResolvedMoveRoute,
    *,
    build_graph: bool,
) -> AwaitMove[Tensor]:
    tensor = cast(Tensor, _validate_move_inputs(tensor, destination))
    destination = cast(Carrier, destination)
    _validate_route_pair(route, tensor, destination)
    required_size = tensor.layout.cosize
    source_token, destination_token = _claim_participants(tensor.carrier, destination)
    try:
        destination_definition = _observe_carrier_definition(type(destination))
        _preflight_destination_extent(
            destination, required_size, destination_definition
        )
        captured_version = tensor._version_token()
        destination_version = destination.version
        source_size = tensor.carrier.size()
        destination_size = destination.size()
        route, prepared = _prepare_transfer(route, tensor, destination)
        if (
            tensor._version_token() != captured_version
            or destination.version != destination_version
            or tensor.carrier.size() != source_size
            or destination.size() != destination_size
            or tensor.carrier.is_released()
            or destination.is_released()
        ):
            raise RuntimeError("transfer preparation changed a participating Carrier")
        reverse_route = (
            _resolve_route(type(destination), type(tensor.carrier))
            if build_graph
            else None
        )
        operation = _operation_for_route(route)
        _require_supported_block_operation(operation, tensor.carrier, destination)
        with _owner_access(
            tensor.carrier,
            destination,
            source_token,
            destination_token,
        ):
            _allocate_destination(destination, required_size, destination_definition)
    except BaseException:
        _relinquish_participants(
            tensor.carrier,
            destination,
            source_token,
            destination_token,
        )
        raise

    state: _CompletionState[Tensor] = _CompletionState()
    handle = cast(AwaitMove[Tensor], AwaitMove._create(state))
    profile = _begin_move_profile(tensor, type(operation))
    commit = _MoveCommit(
        state,
        tensor,
        destination,
        operation,
        reverse_route,
        build_graph,
        captured_version,
        required_size,
        source_token,
        destination_token,
        profile,
        prepared,
        None,
    )

    if prepared is None:
        launch_ready = threading.Event()
        worker = partial(
            _run_legacy_copy,
            state=state,
            tensor=tensor,
            destination=destination,
            operation=operation,
            required_size=required_size,
            source_token=source_token,
            destination_token=destination_token,
            commit=commit,
            launch_ready=launch_ready,
        )
        try:
            _MOVE_BACKGROUND_EXECUTOR.submit(worker)
        except BaseException:
            profile.abandon()
            _relinquish_participants(
                tensor.carrier,
                destination,
                source_token,
                destination_token,
            )
            raise
        profile.submitted(True)
        launch_ready.set()
        return handle

    try:
        with _owner_access(
            tensor.carrier,
            destination,
            source_token,
            destination_token,
        ):
            with _without_graph():
                submitted = prepared.submit()
    except BaseException as error:
        profile.submitted(False)
        _fail_after_submission(commit, error)
        return handle
    profile.submitted(True)
    if submitted is None:
        _commit_success(commit)
        return handle
    try:
        completion = _require_provider_completion(submitted)
    except BaseException as error:
        _fail_after_submission(commit, error)
        return handle
    commit.completion = completion
    try:
        completed = completion.done
    except BaseException as error:
        _fail_after_submission(commit, error)
        return handle
    if completed:
        _wait_for_provider(completion, commit)
        return handle
    try:
        _MOVE_BACKGROUND_EXECUTOR.submit(
            partial(_wait_for_provider, completion, commit)
        )
    except BaseException:
        _wait_for_provider(completion, commit)
    return handle


def move_async(tensor: object, destination: object) -> AwaitMove[Tensor]:
    """Eagerly move one Tensor into one caller-supplied Carrier.

    Validation and exact route preparation finish before submission. The
    returned blocking handle protects both carriers until transfer completion,
    publishes one Tensor and releases the source on success, or preserves a
    usable source and raises the same terminal exception from every ``wait()``
    on failure. The handle is deliberately not awaitable.

    Args:
        tensor: Single live Tensor whose physical Layout span should move.
        destination: Single live, publicly mutable Carrier of the same dtype.

    Returns:
        AwaitMove whose ``wait()`` returns the moved Tensor.

    Examples:
        >>> import strideweave as sw
        >>> source = sw.Tensor(
        ...     sw.Generic([1.0]), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> sw.move_async(source, sw.Generic([0.0])).wait()[0]
        1.0
    """

    tensor = cast(Tensor, _as_tensor(tensor, "tensor"))
    tensor._require_single_subtensor("operation execution")
    destination = cast(Carrier, destination)
    _validate_move_inputs(tensor, destination)
    route = _resolve_route(type(tensor.carrier), type(destination))
    build_graph = bool(_operation.is_grad_enabled() and tensor.is_differentiable())
    return _move_async_with_route(tensor, destination, route, build_graph=build_graph)


def _move_backward(operation: MoveOperation, gradient: object) -> tuple[Tensor]:
    (tensor,) = operation.inputs()
    tensor = cast(Tensor, tensor)
    gradient = cast(Tensor, _require_live_tensor(gradient, "gradient"))
    _require_same_shape(tensor, gradient)
    if not gradient.layout.is_injective:
        raise ValueError("Move backward requires an injective gradient layout")
    reverse_route = cast(
        _ResolvedMoveRoute | None,
        getattr(operation, "_async_reverse_route", None),
    )
    if reverse_route is None:
        raise RuntimeError("move autograd node has no captured reverse route")
    destination_prototype = cast(Carrier, operation._async_destination_prototype)
    target_layout = tensor.layout
    if not target_layout.is_injective:
        target_layout = _canonical_layout_for_shape(target_layout.shape)
    physical_values = _physical_values_for_layout(
        target_layout,
        _logical_values(gradient),
        0,
    )
    staging_carrier: Carrier | None = None
    source_destination: Carrier | None = None
    try:
        staging_carrier = destination_prototype.new_like(
            physical_values,
            mutable=True,
            dtype=tensor.dtype(),
        )
        source_destination = tensor.carrier.allocate_like(
            target_layout.cosize,
            mutable=True,
            dtype=tensor.dtype(),
        )
        staging_tensor = Tensor(staging_carrier, 0, target_layout)
        moved = _move_async_with_route(
            staging_tensor,
            source_destination,
            reverse_route,
            build_graph=False,
        ).wait()
    except BaseException:
        if staging_carrier is not None and not staging_carrier.is_released():
            staging_carrier.release()
        if source_destination is not None and not source_destination.is_released():
            source_destination.release()
        raise
    return (moved,)


__all__ = ["move_async"]
