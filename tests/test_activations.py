from collections.abc import Callable, Iterable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
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


def seeded_activation_values(seed: int, size: int) -> list[float]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    values = torch.randn(size, generator=generator, dtype=torch.float32)
    return values.mul(3.0).tolist()


def make_activation_tensor(
    values: Iterable[float], carrier_kind: str, layout: Layout
) -> Tensor:
    materialized = list(values)
    if carrier_kind == "generic":
        carrier: Any = Generic([0.0] * layout._cache.cosize)
    elif carrier_kind == "cpu":
        carrier = CPU(layout._cache.cosize, dtype=DType.Float32)
    elif carrier_kind == "evictable_generic":
        carrier = Evictable(
            Generic([0.0] * layout._cache.cosize),
            Generic([0.0] * layout._cache.cosize),
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
    torch_tensor: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    actual = torch.tensor(tensor_values(strideweave_tensor), dtype=torch.float32)
    torch.testing.assert_close(actual, torch_tensor.detach(), rtol=rtol, atol=atol)


def run_activation_case(
    operation_name: str,
    torch_activation: Callable[[torch.Tensor], torch.Tensor],
    carrier_kind: str,
    layout: Layout,
) -> None:
    values = seeded_activation_values(ACTIVATION_VALUE_SEED, layout.shape.logical_size)
    gradient_values = seeded_activation_values(
        ACTIVATION_GRADIENT_SEED, layout.shape.logical_size
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

    expected_dtype = DType.Floating if "generic" in carrier_kind else DType.Float32
    assert result.dtype() is expected_dtype
    assert_tensor_close(result, torch_result)
    strideweave_grad = tensor.grad
    assert strideweave_grad is not None
    assert torch_input.grad is not None
    assert strideweave_grad.dtype() is expected_dtype
    assert_tensor_close(strideweave_grad, torch_input.grad)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_relu_activation_matches_pytorch(carrier: str, layout: Layout):
    """ReLU: forward ``y = max(x, 0)``; backward ``dx = dy`` if ``x > 0`` else ``0``."""

    run_activation_case("relu", torch.relu, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_sigmoid_activation_matches_pytorch(carrier: str, layout: Layout):
    """Sigmoid: forward ``y = 1 / (1 + exp(-x))``; backward ``dx = dy * y * (1 - y)``."""

    run_activation_case("sigmoid", torch.sigmoid, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_tanh_activation_matches_pytorch(carrier: str, layout: Layout):
    """Tanh: forward ``y = tanh(x)``; backward ``dx = dy * (1 - y**2)``."""

    run_activation_case("tanh", torch.tanh, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_gelu_activation_matches_pytorch(carrier: str, layout: Layout):
    """GELU: forward ``y = 0.5 * x * (1 + erf(x / sqrt(2)))``; backward ``dx = dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x**2) / sqrt(2 * pi))``."""

    run_activation_case("gelu", F.gelu, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_silu_activation_matches_pytorch(carrier: str, layout: Layout):
    """SiLU: forward ``y = x * sigmoid(x)``; backward ``dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))``."""

    run_activation_case("silu", F.silu, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_softplus_activation_matches_pytorch(carrier: str, layout: Layout):
    """Softplus: forward ``y = log(1 + exp(x))``; backward ``dx = dy * sigmoid(x)``."""

    run_activation_case("softplus", F.softplus, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_elu_activation_matches_pytorch(carrier: str, layout: Layout):
    """ELU: forward ``y = x`` if ``x > 0`` else ``exp(x) - 1``; backward ``dx = dy`` if ``x > 0`` else ``dy * exp(x)``."""

    run_activation_case("elu", F.elu, carrier, layout)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("layout", ACTIVATION_LAYOUTS)
def test_leaky_relu_activation_matches_pytorch(carrier: str, layout: Layout):
    """Leaky ReLU: forward ``y = x`` if ``x >= 0`` else ``0.01 * x``; backward ``dx = dy`` if ``x >= 0`` else ``0.01 * dy``."""

    run_activation_case("leaky_relu", F.leaky_relu, carrier, layout)


@pytest.mark.parametrize("operation_name", ACTIVATION_OPERATION_NAMES)
def test_activations_propagate_released_data_errors(operation_name: str):
    carrier = Generic([1.0])
    tensor = Tensor(carrier, 0, Layout(Shape(1), Stride(1)))
    carrier.release()

    with pytest.raises(RuntimeError, match="released"):
        tensor.carrier.dispatch_op(operation_name).forward(tensor)
