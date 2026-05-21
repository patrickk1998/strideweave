import math
import threading
from typing import Any

import neotorch
import pytest
from neotorch import Generic, GenericEvictable, Layout, Operation, Shape, Stride, Tensor
from neotorch.operation import is_grad_enabled, set_grad_enabled


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


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def test_operation_public_api_imports():
    assert neotorch.Operation is Operation
    assert neotorch.is_grad_enabled is is_grad_enabled
    assert neotorch.set_grad_enabled is set_grad_enabled
    assert neotorch.no_grad is not None
    assert neotorch.relu is not None
    assert neotorch.sigmoid is not None


def test_grad_is_enabled_by_default():
    assert is_grad_enabled()
    assert neotorch.is_grad_enabled()


def test_top_level_set_grad_enabled_updates_current_thread_state():
    previous = neotorch.is_grad_enabled()
    try:
        neotorch.set_grad_enabled(False)
        assert not neotorch.is_grad_enabled()

        neotorch.set_grad_enabled(True)
        assert neotorch.is_grad_enabled()
    finally:
        neotorch.set_grad_enabled(previous)


def test_python_operation_forward_stores_tensor_inputs_and_context():
    operation = EchoOperation()
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    result = operation.forward(lhs, "alpha", rhs)

    assert result.autograd_ctx is operation
    assert operation.inputs() == (lhs, rhs)
    assert operation.ctx["input_count"] == 3
    assert operation.backward("grad") == ("grad", "grad")


def test_no_grad_skips_input_storage_and_autograd_context():
    operation = EchoOperation()
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    with neotorch.no_grad():
        result = operation.forward(lhs, "alpha", rhs)

    assert result.autograd_ctx is None
    assert operation.inputs() == ()
    assert operation.ctx["input_count"] == 3
    assert is_grad_enabled()


def test_no_grad_nested_contexts_restore_previous_state():
    assert is_grad_enabled()

    with neotorch.no_grad():
        assert not is_grad_enabled()
        with neotorch.no_grad():
            assert not is_grad_enabled()
        assert not is_grad_enabled()

    assert is_grad_enabled()


def test_no_grad_restores_existing_disabled_state():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        with neotorch.no_grad():
            assert not is_grad_enabled()
        assert not is_grad_enabled()
    finally:
        set_grad_enabled(previous)


def test_no_grad_state_is_thread_local():
    worker_grad_states: list[bool] = []

    def read_grad_state() -> None:
        worker_grad_states.append(is_grad_enabled())

    with neotorch.no_grad():
        assert not is_grad_enabled()
        worker = threading.Thread(target=read_grad_state)
        worker.start()
        worker.join()

    assert worker_grad_states == [True]
    assert is_grad_enabled()


def test_grad_construction_resumes_after_no_grad_context():
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    with neotorch.no_grad():
        disabled_result = EchoOperation().forward(lhs, rhs)
    enabled_operation = EchoOperation()
    enabled_result = enabled_operation.forward(lhs, rhs)

    assert disabled_result.autograd_ctx is None
    assert enabled_result.autograd_ctx is enabled_operation
    assert enabled_operation.inputs() == (lhs, rhs)


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


def test_relu_forward_and_backward():
    tensor = make_tensor([-2, 0, 3])
    gradient = make_tensor([10, 20, 30])

    result = neotorch.relu(tensor)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert type(result.autograd_ctx).__name__ == "GenericReLUOperation"
    assert tensor_values(result) == [0, 0, 3]
    assert tensor_values(tensor_grad) == [0, 0, 30]


def test_sigmoid_forward_and_backward():
    values = [-2.0, 0.0, 1.0]
    gradient_values = [10.0, 20.0, 30.0]
    tensor = make_tensor(values)
    gradient = make_tensor(gradient_values)

    result = neotorch.sigmoid(tensor)
    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    expected = [1.0 / (1.0 + math.exp(-value)) for value in values]
    expected_grad = [
        grad * sigmoid_value * (1.0 - sigmoid_value)
        for grad, sigmoid_value in zip(gradient_values, expected)
    ]
    assert type(result.autograd_ctx).__name__ == "GenericSigmoidOperation"
    assert tensor_values(result) == pytest.approx(expected)
    assert tensor_values(tensor_grad) == pytest.approx(expected_grad)


def test_relu_and_sigmoid_propagate_evicted_data_errors(tmp_path):
    data = GenericEvictable([1.0], tmp_path / "data.pkl")
    tensor = Tensor(data, 0, Layout(Shape(1), Stride(1)))
    tensor.evict()

    with pytest.raises(RuntimeError, match="tensor data is evicted"):
        neotorch.relu(tensor)

    with pytest.raises(RuntimeError, match="tensor data is evicted"):
        neotorch.sigmoid(tensor)
