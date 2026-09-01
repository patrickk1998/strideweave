from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

import pytest

from strideweave import (
    CPU,
    DType,
    Evictable,
    FileBacked,
    Generic,
    Layout,
    Shape,
    Stride,
    Tensor,
)

ACTIVATION_LAYOUTS = (
    pytest.param(Layout(Shape([8, 16]), Stride([1, 8])), id="current_8x16"),
    pytest.param(Layout(Shape([64, 64]), Stride([1, 64])), id="large_64x64"),
    pytest.param(Layout(Shape([37, 53]), Stride([1, 37])), id="irregular_37x53"),
)
ACTIVATION_VALUE_SEED = 20260531
ACTIVATION_GRADIENT_SEED = 20260532
CARRIERS = ("generic", "cpu", "evictable_generic", "evictable_cpu")
ACTIVATION_OPERATION_NAMES = (
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "softplus",
    "elu",
    "leaky_relu",
)


@pytest.fixture(scope="module")
def activation_references(torch_reference: Any) -> tuple[Any, Any]:
    return torch_reference, import_module("torch.nn.functional")


def seeded_activation_values(torch: Any, seed: int, size: int) -> list[float]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    values = torch.randn(size, generator=generator, dtype=torch.float32)
    return values.mul(3.0).tolist()


def make_activation_tensor(
    values: Iterable[float], carrier_kind: str, layout: Layout
) -> Tensor:
    materialized = list(values)
    if carrier_kind == "generic":
        carrier: Any = Generic([0.0] * layout._cache.cosize, dtype=DType.Float32)
    elif carrier_kind == "cpu":
        carrier = CPU(layout._cache.cosize, dtype=DType.Float32)
    elif carrier_kind == "evictable_generic":
        carrier = Evictable(
            Generic([0.0] * layout._cache.cosize, dtype=DType.Float32),
            Generic([0.0] * layout._cache.cosize, dtype=DType.Float32),
        )
    elif carrier_kind == "evictable_cpu":
        carrier = Evictable(
            CPU(layout._cache.cosize, dtype=DType.Float32),
            FileBacked(dtype=DType.Float32),
        )
    else:
        raise ValueError(f"unknown activation test carrier: {carrier_kind}")

    for logical_index, value in enumerate(materialized):
        carrier[layout.index(logical_index)] = value
    return Tensor(carrier, 0, layout)


def tensor_values(tensor: Tensor) -> list[float]:
    return [tensor[i] for i in range(tensor.size())]


def assert_tensor_close(
    strideweave_tensor: Tensor,
    torch_tensor: Any,
    torch: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 2e-6,
) -> None:
    actual = torch.tensor(tensor_values(strideweave_tensor), dtype=torch.float32)
    torch.testing.assert_close(actual, torch_tensor.detach(), rtol=rtol, atol=atol)


def run_activation_case(
    operation_name: str,
    torch_activation: Callable[[Any], Any],
    carrier_kind: str,
    layout: Layout,
    torch: Any,
) -> None:
    values = seeded_activation_values(
        torch, ACTIVATION_VALUE_SEED, layout.shape.logical_size
    )
    gradient_values = seeded_activation_values(
        torch, ACTIVATION_GRADIENT_SEED, layout.shape.logical_size
    )
    tensor = make_activation_tensor(values, carrier_kind, layout)
    gradient = make_activation_tensor(gradient_values, carrier_kind, layout)
    operation = tensor.carrier.dispatch_op(operation_name)

    torch_input = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    torch_gradient = torch.tensor(gradient_values, dtype=torch.float32)

    result = operation.forward(tensor)
    torch_result = torch_activation(torch_input)

    result.backward(gradient)
    torch_result.backward(torch_gradient)

    expected_dtype = DType.Float32
    assert result.dtype() is expected_dtype
    assert_tensor_close(result, torch_result, torch)
    strideweave_grad = tensor.grad
    assert strideweave_grad is not None
    assert torch_input.grad is not None
    assert strideweave_grad.dtype() is expected_dtype
    assert_tensor_close(strideweave_grad, torch_input.grad, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_relu_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """ReLU: forward ``y = max(x, 0)``; backward ``dx = dy`` if ``x > 0`` else ``0``."""

    torch, _ = activation_references
    run_activation_case("relu", torch.relu, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_sigmoid_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """Sigmoid: forward ``y = 1 / (1 + exp(-x))``; backward ``dx = dy * y * (1 - y)``."""

    torch, _ = activation_references
    run_activation_case("sigmoid", torch.sigmoid, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_tanh_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """Tanh: forward ``y = tanh(x)``; backward ``dx = dy * (1 - y**2)``."""

    torch, _ = activation_references
    run_activation_case("tanh", torch.tanh, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_gelu_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """GELU: forward ``y = 0.5 * x * (1 + erf(x / sqrt(2)))``; backward ``dx = dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x**2) / sqrt(2 * pi))``."""

    torch, functional = activation_references
    run_activation_case("gelu", functional.gelu, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_silu_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """SiLU: forward ``y = x * sigmoid(x)``; backward ``dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))``."""

    torch, functional = activation_references
    run_activation_case("silu", functional.silu, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_softplus_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """Softplus: forward ``y = log(1 + exp(x))``; backward ``dx = dy * sigmoid(x)``."""

    torch, functional = activation_references
    run_activation_case("softplus", functional.softplus, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_elu_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """ELU: forward ``y = x`` if ``x > 0`` else ``exp(x) - 1``; backward ``dx = dy`` if ``x > 0`` else ``dy * exp(x)``."""

    torch, functional = activation_references
    run_activation_case("elu", functional.elu, carrier, layout, torch)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_leaky_relu_activation_matches_pytorch(
    carrier: str, layout: Layout, activation_references: tuple[Any, Any]
):
    """Leaky ReLU: forward ``y = x`` if ``x >= 0`` else ``0.01 * x``; backward ``dx = dy`` if ``x >= 0`` else ``0.01 * dy``."""

    torch, functional = activation_references
    run_activation_case("leaky_relu", functional.leaky_relu, carrier, layout, torch)


@pytest.mark.parametrize("operation_name", ACTIVATION_OPERATION_NAMES)
def test_activations_propagate_released_data_errors(operation_name: str):
    carrier = Generic([1.0], dtype=DType.Float32)
    tensor = Tensor(carrier, 0, Layout(Shape(1), Stride(1)))
    carrier.release()

    with pytest.raises(RuntimeError, match="released"):
        tensor.carrier.dispatch_op(operation_name).forward(tensor)
