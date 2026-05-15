from typing import Any

import neotorch
import pytest
from neotorch import Generic, Layout, Operation, Shape, Stride, Tensor


class EchoOperation(Operation):
    def _forward(self, *inputs: Any) -> Tensor:
        self.ctx["input_count"] = len(inputs)
        return make_tensor([10, 20])

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        return tuple(gradient for _ in self.inputs())


class NonTensorForwardOperation(Operation):
    def _forward(self, *inputs: Any) -> tuple[Any, ...]:
        return inputs

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        return tuple(gradient for _ in self.inputs())


class MissingForwardOperation(Operation):
    def backward(self, gradient: Any) -> tuple[Any, ...]:
        return tuple(gradient for _ in self.inputs())


class MissingBackwardOperation(Operation):
    def _forward(self, *inputs: Any) -> Tensor:
        return make_tensor([10, 20])


def make_tensor(values: list[Any]) -> Tensor:
    return Tensor(Generic(values), 0, Layout(Shape(len(values)), Stride(1)))


def test_operation_public_api_imports():
    assert neotorch.Operation is Operation


def test_python_operation_forward_stores_tensor_inputs_and_context():
    operation = EchoOperation()
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    result = operation.forward(lhs, "alpha", rhs)

    assert result.autograd_ctx is operation
    assert operation.inputs() == (lhs, rhs)
    assert operation.ctx["input_count"] == 3
    assert operation.backward("grad") == ("grad", "grad")


def test_operation_forward_requires_tensor_result():
    operation = NonTensorForwardOperation()
    tensor = make_tensor([1, 2])

    with pytest.raises(TypeError):
        operation.forward(tensor)


def test_operation_subclass_missing_forward_raises():
    operation = MissingForwardOperation()
    tensor = make_tensor([1, 2])

    with pytest.raises(TypeError):
        operation.forward(tensor)


def test_operation_subclass_missing_backward_raises():
    operation = MissingBackwardOperation()

    operation.forward(make_tensor([1, 2]))

    with pytest.raises(RuntimeError):
        operation.backward("grad")
