from contextlib import AbstractContextManager

from ...core.tensor import Tensor
from ..base import Carrier
from ..extension import CarrierDefinition
from .await_result import AwaitMove
from .ops import MoveOperation

class _ResolvedMoveRoute: ...

def _resolve_route(
    source_class: type[Carrier], destination_class: type[Carrier]
) -> _ResolvedMoveRoute: ...
def _move_async_with_route(
    tensor: object,
    destination: object,
    route: _ResolvedMoveRoute,
    *,
    build_graph: bool,
) -> AwaitMove[Tensor]: ...
def _preflight_destination_extent(
    destination: Carrier,
    required_size: int,
    definition: CarrierDefinition | None,
) -> None: ...
def _allocate_destination(
    destination: Carrier,
    required_size: int,
    definition: CarrierDefinition | None,
) -> None: ...
def _without_graph() -> AbstractContextManager[None]: ...
def move_async(tensor: Tensor, destination: Carrier) -> AwaitMove[Tensor]: ...
def _move_backward(operation: MoveOperation, gradient: object) -> tuple[Tensor]: ...

__all__ = ["move_async"]
