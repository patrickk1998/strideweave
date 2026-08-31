"""Bounded full-value operation lowering for :class:`TiledEvictable`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from ...core.tensor import Tensor
from ...operation_definition import ResolvedInvocation
from ..extension import PreparedComposite, ProviderResult, Unsupported
from ..operation_capability import OperationCapability
from ..operation_helpers import execute_lowered_operation
from ..operation_policy import OperandRole, operation_execution_options
from .carrier import _coordinates, _owner_access, _validate_full_tensor
from .values import TileSet

if TYPE_CHECKING:
    from .carrier import TiledEvictable


_DIRECT_NAMES = frozenset(
    {"add", "elementwise_mul", "mul", "relu", "reduce_sum", "matmul"}
)


def _mode_extent(mode: object) -> int:
    return mode if isinstance(mode, int) else cast(Any, mode).logical_size


def _dispatch_tensor_dtype(capability: OperationCapability) -> object:
    return next(
        operand.dtype
        for operand in capability.operands
        if operand.role is OperandRole.TENSOR
    )


def composite_capabilities(
    carrier: TiledEvictable,
) -> tuple[OperationCapability, ...]:
    """Derive this instance's bounded plans from its configured compute tier."""

    capabilities: dict[OperationCapability, None] = {}
    for capability in carrier.primary.operation_capabilities():
        if capability.operation not in _DIRECT_NAMES:
            continue
        if _dispatch_tensor_dtype(capability) is not carrier.dtype():
            continue
        if not carrier.primary.supports_storage_dtype(capability.output):
            continue
        capabilities[capability] = None
        if (
            capability.operation == "mul"
            and len(capability.operands) == 2
            and capability.operands[0].role is OperandRole.TENSOR
            and capability.operands[1].role is OperandRole.WEAK_SCALAR
        ):
            capabilities[
                replace(
                    capability,
                    operands=(capability.operands[1], capability.operands[0]),
                )
            ] = None
    return tuple(capabilities)


def composite_result_carriers(
    carrier: TiledEvictable, invocation: object
) -> tuple[type, ...]:
    """Name the ordinary configured compute class for every dense result."""

    if not isinstance(invocation, ResolvedInvocation):
        raise TypeError("invocation must be a ResolvedInvocation")
    if any(
        not carrier.primary.supports_storage_dtype(spec.dtype)
        for spec in invocation.results
    ):
        raise RuntimeError("configured compute carrier cannot publish result dtype")
    return tuple(type(carrier.primary) for _ in invocation.results)


def _full_tiled_operands(
    carrier: TiledEvictable, invocation: ResolvedInvocation
) -> tuple[TiledEvictable, ...] | Unsupported:
    from .carrier import TiledEvictable

    operands: list[TiledEvictable] = []
    seen: set[int] = set()
    for value in invocation.operands:
        if not isinstance(value, Tensor):
            continue
        tiled = value.carrier
        if type(tiled) is not TiledEvictable:
            return Unsupported("every Tensor operand must be backed by TiledEvictable")
        tiled = cast(TiledEvictable, tiled)
        if tiled.is_released():
            raise RuntimeError("TiledEvictable is released")
        if type(tiled.primary) is not type(carrier.primary):
            return Unsupported("tiled operands use incompatible compute carriers")
        if (
            value.size() != tiled.size()
            or tuple(_mode_extent(mode) for mode in value.layout.shape.top_level)
            != tiled.logical_shape
        ):
            return Unsupported(
                "direct tiled operands must preserve the full logical geometry"
            )
        _validate_full_tensor(tiled, value, "direct tiled operand")
        identity = id(tiled)
        if identity not in seen:
            seen.add(identity)
            operands.append(tiled)
    if not operands:
        return Unsupported("direct tiled execution requires a Tensor operand")
    return tuple(operands)


def _materialize_tensor(tensor: Tensor) -> Tensor:
    from .carrier import TiledEvictable

    tiled = cast(TiledEvictable, tensor.carrier)
    values = [tiled[index] for index in range(tiled.size())]
    prototype = cast(Any, tiled)._primary_owned
    with cast(Any, tiled)._prototype_lock:
        with _owner_access(prototype) as primary:
            dense = primary.new_like(
                values,
                mutable=True,
                dtype=tiled.dtype(),
            )
    return Tensor(dense, tensor.offset, tensor.layout)


def _execution_options(invocation: ResolvedInvocation) -> object | None:
    accumulator_dtype = invocation.options.get("accumulator_dtype")
    if accumulator_dtype is None:
        return None
    return operation_execution_options(
        invocation.name,
        accumulator_dtype=cast(Any, accumulator_dtype),
    )


def _submit_composite(
    invocation: ResolvedInvocation,
    operands: tuple[TiledEvictable, ...],
) -> ProviderResult:
    temporary: list[Tensor] = []
    result: Tensor | None = None
    try:
        with ExitStack() as work:
            for tiled in operands:
                work.enter_context(
                    cast(Any, tiled)._pin_tiles_for_work(
                        TileSet(_coordinates(tiled.grid_shape)),
                        "operation",
                    )
                )
            lowered: list[object] = []
            for value in invocation.operands:
                if isinstance(value, Tensor):
                    materialized = _materialize_tensor(value)
                    temporary.append(materialized)
                    lowered.append(materialized)
                else:
                    lowered.append(value)
            tensor = next(value for value in lowered if isinstance(value, Tensor))
            operation = tensor.carrier.dispatch_op(invocation.name)
            arguments = tuple(lowered)
            if (
                invocation.name == "mul"
                and not isinstance(arguments[0], Tensor)
                and isinstance(arguments[1], Tensor)
                and not tensor.carrier.supports_operation_plan(invocation.plan)
            ):
                arguments = (arguments[1], arguments[0])
            options = _execution_options(invocation)
            if options is None:
                result = cast(
                    Tensor,
                    execute_lowered_operation(operation, *arguments),
                )
            else:
                result = cast(
                    Tensor,
                    execute_lowered_operation(
                        operation,
                        *arguments,
                        options=options,
                    ),
                )
        return ProviderResult((result,))
    except BaseException:
        if result is not None and not result.carrier.is_released():
            result.carrier.release()
        raise
    finally:
        for tensor in temporary:
            if not tensor.carrier.is_released():
                tensor.carrier.release()


def prepare_composite(
    carrier: TiledEvictable, invocation: object
) -> PreparedComposite | Unsupported:
    """Validate and seal one synchronous full-tiled dependency lowering."""

    if not isinstance(invocation, ResolvedInvocation):
        raise TypeError("invocation must be a ResolvedInvocation")
    if invocation.name not in _DIRECT_NAMES:
        return Unsupported("operation is outside the bounded direct tiled set")
    operands = _full_tiled_operands(carrier, invocation)
    if isinstance(operands, Unsupported):
        return operands
    return PreparedComposite(lambda: _submit_composite(invocation, operands))


@contextmanager
def composite_vjp_scope(
    carrier: TiledEvictable, invocation: ResolvedInvocation
) -> Iterator[None]:
    """Make every saved tiled value available for the central semantic VJP."""

    operands = _full_tiled_operands(carrier, invocation)
    if isinstance(operands, Unsupported):
        raise RuntimeError(operands.reason)
    saved_carriers = {
        id(value.carrier)
        for index in invocation.saved_operands
        if isinstance((value := invocation.operands[index]), Tensor)
    }
    with ExitStack() as work:
        for tiled in operands:
            if id(tiled) not in saved_carriers:
                continue
            work.enter_context(
                cast(Any, tiled)._pin_tiles_for_work(
                    TileSet(_coordinates(tiled.grid_shape)),
                    "backward",
                )
            )
        yield


__all__: list[str] = []
