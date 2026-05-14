from typing import Any

from ._operation import Operation as Operation
from .tensor import Tensor

class GenericAddOperation(Operation):
    def forward(self, *inputs: Any) -> Tensor: ...
    def backward(self, gradient: Tensor) -> tuple[Tensor, Tensor]: ...

def add(lhs: Tensor, rhs: Tensor) -> Tensor: ...
