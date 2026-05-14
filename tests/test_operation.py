from typing import Any

import neotorch
import pytest
from neotorch import Operation


class EchoOperation(Operation):
    def forward(self, *inputs: Any) -> tuple[Any, ...]:
        self.store_inputs(*inputs)
        self.ctx["input_count"] = len(inputs)
        return inputs

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        return tuple(gradient for _ in self.inputs())


class MissingBackwardOperation(Operation):
    def forward(self, *inputs: Any) -> tuple[Any, ...]:
        self.store_inputs(*inputs)
        return inputs


def test_operation_public_api_imports():
    assert neotorch.Operation is Operation


def test_python_operation_subclass_stores_context_and_inputs():
    operation = EchoOperation()

    result = operation.forward("alpha", "beta")

    assert result == ("alpha", "beta")
    assert operation.inputs() == ("alpha", "beta")
    assert operation.ctx["input_count"] == 2
    assert operation.backward("grad") == ("grad", "grad")


def test_operation_subclass_missing_backward_raises():
    operation = MissingBackwardOperation()

    operation.forward("alpha")

    with pytest.raises(RuntimeError):
        operation.backward("grad")
