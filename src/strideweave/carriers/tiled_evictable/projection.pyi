from ...tensor import Tensor
from ..move.await_result import AwaitResult

class AwaitProjection(AwaitResult[Tensor]): ...

__all__ = ["AwaitProjection"]
