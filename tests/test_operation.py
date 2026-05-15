import threading
from typing import Any

import neotorch
import pytest
from neotorch import Generic, Layout, Operation, Shape, Stride, Tensor
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


def test_operation_public_api_imports():
    assert neotorch.Operation is Operation
    assert neotorch.is_grad_enabled is is_grad_enabled
    assert neotorch.set_grad_enabled is set_grad_enabled
    assert neotorch.no_grad is not None


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
