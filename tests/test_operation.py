import threading
from typing import Any

import pytest

import strideweave as sw
import strideweave.functional.api as functional_api
from strideweave import FileBacked, Generic, Layout, Operation, Shape, Stride, Tensor
from strideweave.carriers.operation_policy import operation_execution_options
from strideweave.operation import is_grad_enabled, set_grad_enabled


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
    assert sw.Operation is Operation
    assert sw.is_grad_enabled is is_grad_enabled
    assert sw.set_grad_enabled is set_grad_enabled
    assert sw.no_grad is not None
    assert sw.relu is not None
    assert sw.sigmoid is not None
    assert sw.tanh is not None
    assert sw.gelu is not None
    assert sw.silu is not None
    assert sw.softplus is not None
    assert sw.elu is not None
    assert sw.leaky_relu is not None


def test_grad_is_enabled_by_default():
    assert is_grad_enabled()
    assert sw.is_grad_enabled()


def test_top_level_set_grad_enabled_updates_current_thread_state():
    previous = sw.is_grad_enabled()
    try:
        sw.set_grad_enabled(False)
        assert not sw.is_grad_enabled()

        sw.set_grad_enabled(True)
        assert sw.is_grad_enabled()
    finally:
        sw.set_grad_enabled(previous)


def test_python_operation_forward_stores_tensor_inputs_and_context():
    operation = EchoOperation()
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    result = operation.forward(lhs, "alpha", rhs)

    assert result.autograd_ctx is operation
    assert operation.inputs() == (lhs, rhs)
    assert operation.ctx["input_count"] == 3
    assert operation.backward("grad") == ("grad", "grad")


def test_execution_options_are_not_positional_or_saved_autograd_inputs():
    operation = EchoOperation()
    tensor = make_tensor([1, 2])
    options = operation_execution_options("reduce_sum")

    result = operation.forward(tensor, "label", options=options)

    assert result.autograd_ctx is operation
    assert operation.ctx["input_count"] == 2
    assert operation.inputs() == (tensor,)
    assert operation.input_versions() == (tensor._version_token(),)
    assert operation._execution_options is options


def test_operation_forward_rejects_unknown_execution_option_names():
    with pytest.raises(TypeError, match="unknown operation execution option"):
        EchoOperation().forward(
            make_tensor([1, 2]),
            accumulator_dtype=sw.DType.Float64,  # pyright: ignore[reportCallIssue]
        )


def test_lowered_execution_preserves_typed_options_outside_inputs():
    operation = EchoOperation()
    saved = make_tensor([1, 2])
    options = operation_execution_options("matmul")
    operation.store_inputs(saved)

    operation._execute_lowered(  # strideweave-lint: ignore=RT011
        make_tensor([3, 4]), options=options
    )

    assert operation.inputs() == (saved,)
    assert operation._execution_options is options


def test_default_accumulations_use_the_positional_forward_path(monkeypatch):
    calls = []

    class SpyOperation:
        def forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "result"

    monkeypatch.setattr(functional_api, "_dispatch_unary", lambda *_: SpyOperation())
    monkeypatch.setattr(functional_api, "_dispatch_binary", lambda *_: SpyOperation())
    monkeypatch.setattr(
        functional_api,
        "operation_execution_options",
        lambda *_args, **_kwargs: pytest.fail("default path allocated options"),
    )

    assert functional_api._reduce_second_mode("tensor") == "result"
    assert functional_api._matmul_2mode("lhs", "rhs") == "result"
    assert calls == [(("tensor",), {}), (("lhs", "rhs"), {})]


def test_explicit_accumulations_pass_typed_options(monkeypatch):
    options = object()
    calls = []

    class SpyOperation:
        def forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "result"

    monkeypatch.setattr(functional_api, "_dispatch_unary", lambda *_: SpyOperation())
    monkeypatch.setattr(functional_api, "_dispatch_binary", lambda *_: SpyOperation())
    monkeypatch.setattr(
        functional_api, "operation_execution_options", lambda *_args, **_kwargs: options
    )

    functional_api._reduce_second_mode("tensor", accumulator_dtype=sw.DType.Float64)
    functional_api._matmul_2mode("lhs", "rhs", accumulator_dtype=sw.DType.Float32)
    assert calls == [
        (("tensor",), {"options": options}),
        (("lhs", "rhs"), {"options": options}),
    ]


def test_explicit_invalid_accumulators_are_rejected_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        functional_api,
        "_dispatch_unary",
        lambda *_: pytest.fail("reduce dispatched before validating its accumulator"),
    )
    monkeypatch.setattr(
        functional_api,
        "_dispatch_binary",
        lambda *_: pytest.fail("matmul dispatched before validating its accumulator"),
    )

    with pytest.raises(TypeError, match="accumulator_dtype must be a DType or None"):
        functional_api._reduce_second_mode(
            "tensor",
            accumulator_dtype="Float64",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="accumulator_dtype must be a DType or None"):
        functional_api._matmul_2mode(
            "lhs",
            "rhs",
            accumulator_dtype="Float64",  # type: ignore[arg-type]
        )


def test_public_reduce_validates_an_invalid_accumulator_before_file_backed_dispatch():
    carrier = FileBacked(dtype=sw.DType.Float32).new_like([1.0, 2.0])
    tensor = Tensor(carrier, 0, Layout(Shape([1, 2]), Stride([1, 1])))

    with pytest.raises(TypeError, match="accumulator_dtype must be a DType or None"):
        sw.reduce_sum(
            tensor,
            "a b -> a",
            accumulator_dtype="Float64",  # type: ignore[arg-type]
        )


def test_no_grad_skips_input_storage_and_autograd_context():
    operation = EchoOperation()
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    with sw.no_grad():
        result = operation.forward(lhs, "alpha", rhs)

    assert result.autograd_ctx is None
    assert operation.inputs() == ()
    assert operation.ctx["input_count"] == 3
    assert is_grad_enabled()


def test_no_grad_nested_contexts_restore_previous_state():
    assert is_grad_enabled()

    with sw.no_grad():
        assert not is_grad_enabled()
        with sw.no_grad():
            assert not is_grad_enabled()
        assert not is_grad_enabled()

    assert is_grad_enabled()


def test_no_grad_restores_existing_disabled_state():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        with sw.no_grad():
            assert not is_grad_enabled()
        assert not is_grad_enabled()
    finally:
        set_grad_enabled(previous)


def test_no_grad_state_is_thread_local():
    worker_grad_states: list[bool] = []

    def read_grad_state() -> None:
        worker_grad_states.append(is_grad_enabled())

    with sw.no_grad():
        assert not is_grad_enabled()
        worker = threading.Thread(target=read_grad_state)
        worker.start()
        worker.join()

    assert worker_grad_states == [True]
    assert is_grad_enabled()


def test_grad_construction_resumes_after_no_grad_context():
    lhs = make_tensor([1, 2])
    rhs = make_tensor([3, 4])

    with sw.no_grad():
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


def test_lowered_execution_requires_tensor_result():
    operation = NonTensorForwardOperation()

    with pytest.raises(TypeError, match="must return a Tensor"):
        operation._execute_lowered(  # strideweave-lint: ignore=RT011
            make_tensor([1, 2])
        )


def test_lowered_execution_preserves_delegated_state_without_attaching_graph():
    operation = EchoOperation()
    saved = make_tensor([1, 2])
    operation.store_inputs(saved)

    result = operation._execute_lowered(  # strideweave-lint: ignore=RT011
        make_tensor([3, 4])
    )

    assert result.autograd_ctx is None
    assert operation.inputs() == (saved,)
    assert operation.ctx["input_count"] == 1


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
