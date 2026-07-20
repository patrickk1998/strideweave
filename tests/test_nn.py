import math
import random

import neotorch
import neotorch.nn as nn
import pytest
from neotorch import CPU, DataType, Layout, Shape, Stride, Tensor


def make_cpu_data(values, dtype=DataType.Float32):
    values = list(values)
    data = CPU(len(values), dtype=dtype)
    for i, value in enumerate(values):
        data[i] = value
    return data


def make_cpu_tensor(values, layout, dtype=DataType.Float32):
    return Tensor(make_cpu_data(values, dtype), 0, layout)


def column_major(rows, cols):
    return Layout(Shape([rows, cols]), Stride([1, rows]))


def tensor_values(tensor):
    return [
        tensor.data.get_value(tensor.offset + tensor.layout.index(i))
        for i in range(tensor.layout.size)
    ]


def logical_values(tensor, rows, cols):
    return [[tensor[i, j] for j in range(cols)] for i in range(rows)]


def set_logical_values(tensor, rows_of_values):
    for i, row in enumerate(rows_of_values):
        for j, value in enumerate(row):
            tensor[i, j] = value


def test_bias_tile_layout_matches_matmul_output_layout():
    batch, in_features, out_features = 3, 2, 4
    x = make_cpu_tensor(
        [float(i) for i in range(batch * in_features)],
        column_major(batch, in_features),
    )
    weight = make_cpu_tensor(
        [float(i) for i in range(out_features * in_features)],
        column_major(out_features, in_features),
    )
    ones = make_cpu_tensor([1.0] * batch, column_major(batch, 1))
    bias = make_cpu_tensor([10.0, 20.0, 30.0, 40.0], column_major(out_features, 1))

    product = x @ weight
    tile = ones @ bias

    assert product.layout == tile.layout == column_major(batch, out_features)
    combined = product + tile
    for i in range(batch):
        for j in range(out_features):
            expected = (
                sum(x[i, k] * weight[j, k] for k in range(in_features)) + bias[j, 0]
            )
            assert combined[i, j] == pytest.approx(expected)


def test_reduce_description_yields_exact_scalar_layout():
    tensor = make_cpu_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], column_major(2, 3))

    total = neotorch.reduce(tensor, "a b -> 1")

    assert total.layout == Layout(Shape(1), Stride(1))
    assert total[0] == pytest.approx(21.0)
    total.backward()
    assert tensor.grad is not None


def test_linear_forward_matches_reference_computation():
    layer = nn.Linear(2, 3, rng=random.Random(0))
    set_logical_values(layer.weight, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    set_logical_values(layer.bias, [[0.5], [-0.5], [1.5]])

    x = make_cpu_tensor([0.0] * 4, column_major(2, 2))
    set_logical_values(x, [[1.0, 2.0], [3.0, -1.0]])
    result = layer(x)

    assert result.layout == column_major(2, 3)
    assert logical_values(result, 2, 3) == [
        [pytest.approx(5.5), pytest.approx(10.5), pytest.approx(18.5)],
        [pytest.approx(1.5), pytest.approx(4.5), pytest.approx(10.5)],
    ]


def test_linear_bias_gradient_is_batch_sum_of_upstream_gradient():
    layer = nn.Linear(2, 3, rng=random.Random(0))
    x = make_cpu_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], column_major(3, 2))

    result = layer(x)
    upstream = make_cpu_tensor([0.0] * 9, result.layout)
    set_logical_values(upstream, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result.backward(upstream)

    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert logical_values(layer.bias.grad, 3, 1) == [
        [pytest.approx(12.0)],
        [pytest.approx(15.0)],
        [pytest.approx(18.0)],
    ]


def test_linear_weight_gradient_matches_reference():
    layer = nn.Linear(2, 2, rng=random.Random(0))
    x = make_cpu_tensor([0.0] * 4, column_major(2, 2))
    set_logical_values(x, [[1.0, 2.0], [3.0, 4.0]])

    result = layer(x)
    upstream = make_cpu_tensor([0.0] * 4, result.layout)
    set_logical_values(upstream, [[1.0, 0.0], [0.0, 1.0]])
    result.backward(upstream)

    assert layer.weight.grad is not None
    # d(weight[j, k]) = sum_i upstream[i, j] * x[i, k]
    for j in range(2):
        for k in range(2):
            expected = sum(upstream[i, j] * x[i, k] for i in range(2))
            assert layer.weight.grad[j, k] == pytest.approx(expected)


def test_linear_without_bias_registers_single_parameter():
    layer = nn.Linear(3, 2, bias=False, rng=random.Random(0))

    assert layer.bias is None
    assert layer.parameters() == (layer.weight,)

    x = make_cpu_tensor([1.0] * 6, column_major(2, 3))
    result = layer(x)
    expected = sum(layer.weight[0, k] for k in range(3))
    assert result[0, 0] == pytest.approx(expected)


def test_linear_init_respects_kaiming_uniform_bound():
    layer = nn.Linear(4, 8, rng=random.Random(0))
    bound = 1.0 / math.sqrt(4)

    values = tensor_values(layer.weight) + tensor_values(layer.bias)
    assert all(-bound <= value <= bound for value in values)
    assert any(value != 0.0 for value in values)


def test_linear_rejects_mismatched_input_features():
    layer = nn.Linear(2, 3, rng=random.Random(0))
    x = make_cpu_tensor([1.0] * 6, column_major(2, 3))

    with pytest.raises(ValueError, match="input features"):
        layer(x)


def test_linear_rejects_non_flat_input():
    layer = nn.Linear(4, 2, rng=random.Random(0))
    hierarchical = make_cpu_tensor(
        [1.0] * 8, Layout(Shape([2, [2, 2]]), Stride([1, [2, 4]]))
    )

    with pytest.raises(ValueError, match="flat two-mode"):
        layer(hierarchical)


def test_activation_modules_delegate_to_functional_ops():
    layout = Layout(Shape(3), Stride(1))
    values = [-1.0, 0.0, 2.0]
    cases = [
        (nn.ReLU(), neotorch.relu),
        (nn.Sigmoid(), neotorch.sigmoid),
        (nn.Tanh(), neotorch.tanh),
        (nn.GELU(), neotorch.gelu),
        (nn.SiLU(), neotorch.silu),
        (nn.Softplus(), neotorch.softplus),
        (nn.ELU(), neotorch.elu),
        (nn.LeakyReLU(), neotorch.leaky_relu),
    ]

    for module, function in cases:
        module_result = module(make_cpu_tensor(values, layout))
        function_result = function(make_cpu_tensor(values, layout))
        assert tensor_values(module_result) == pytest.approx(
            tensor_values(function_result)
        )
        assert type(module_result.autograd_ctx) is type(function_result.autograd_ctx)
        assert module.parameters() == ()


def test_activation_module_registers_as_submodule():
    class Model(neotorch.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.Tanh()

    model = Model()

    assert model.modules() == (model, model.activation)


def test_mse_loss_value_and_implicit_backward():
    prediction = make_cpu_tensor([0.0] * 4, column_major(2, 2))
    set_logical_values(prediction, [[1.0, 2.0], [3.0, 4.0]])
    target = make_cpu_tensor([0.0] * 4, column_major(2, 2))
    set_logical_values(target, [[0.0, 0.0], [0.0, 8.0]])

    loss = nn.MSELoss()(prediction, target)

    assert loss.layout == Layout(Shape(1), Stride(1))
    assert loss[0] == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 4.0)

    loss.backward()

    assert prediction.grad is not None
    # d(loss)/d(prediction) = 2 * (prediction - target) / N
    for i in range(2):
        for j in range(2):
            expected = 2.0 * (prediction[i, j] - target[i, j]) / 4.0
            assert prediction.grad[i, j] == pytest.approx(expected)


def test_mse_loss_rejects_mismatched_layouts():
    prediction = make_cpu_tensor([1.0] * 4, column_major(2, 2))
    target = make_cpu_tensor([1.0] * 4, column_major(4, 1))

    with pytest.raises(ValueError, match="layouts must match"):
        nn.MSELoss()(prediction, target)


def test_sgd_step_applies_learning_rate_and_skips_missing_grads():
    layer = nn.Linear(2, 2, rng=random.Random(0))
    x = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], column_major(2, 2))
    result = layer(x)
    result.backward(make_cpu_tensor([1.0] * 4, result.layout))

    weight_before = tensor_values(layer.weight)
    weight_grad = tensor_values(layer.weight.grad)
    assert layer.bias is not None
    layer.bias.grad = None
    bias_before = tensor_values(layer.bias)

    optimizer = nn.SGD(layer.parameters(), lr=0.5)
    optimizer.step()

    assert tensor_values(layer.weight) == pytest.approx(
        [w - 0.5 * g for w, g in zip(weight_before, weight_grad)]
    )
    assert tensor_values(layer.bias) == pytest.approx(bias_before)


def test_sgd_zero_grad_resets_gradients_to_none():
    layer = nn.Linear(2, 2, rng=random.Random(0))
    x = make_cpu_tensor([1.0] * 4, column_major(2, 2))
    result = layer(x)
    result.backward(make_cpu_tensor([1.0] * 4, result.layout))
    optimizer = nn.SGD(layer.parameters(), lr=0.1)

    assert layer.weight.grad is not None
    assert layer.bias is not None
    optimizer.zero_grad()
    assert layer.weight.grad is None
    assert layer.bias.grad is None


def test_gradients_accumulate_without_zero_grad():
    layer = nn.Linear(2, 2, bias=False, rng=random.Random(0))
    x = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], column_major(2, 2))

    layer(x).backward(make_cpu_tensor([1.0] * 4, column_major(2, 2)))
    single = tensor_values(layer.weight.grad)
    layer(x).backward(make_cpu_tensor([1.0] * 4, column_major(2, 2)))
    doubled = tensor_values(layer.weight.grad)

    assert doubled == pytest.approx([2.0 * value for value in single])


def test_backward_after_step_rejects_stale_graph():
    layer = nn.Linear(2, 2, bias=False, rng=random.Random(0))
    x = make_cpu_tensor([1.0, 2.0, 3.0, 4.0], column_major(2, 2))
    optimizer = nn.SGD(layer.parameters(), lr=0.1)

    result = layer(x)
    result.backward(make_cpu_tensor([1.0] * 4, result.layout))
    optimizer.step()

    with pytest.raises(RuntimeError):
        result.backward(make_cpu_tensor([1.0] * 4, result.layout))


def test_sgd_rejects_non_positive_learning_rate():
    layer = nn.Linear(2, 2, rng=random.Random(0))

    with pytest.raises(ValueError, match="lr must be positive"):
        nn.SGD(layer.parameters(), lr=0.0)


def test_mlp_training_loss_decreases():
    rng = random.Random(0)

    class MLP(neotorch.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(1, 8, rng=rng)
            self.activation = nn.Tanh()
            self.second = nn.Linear(8, 1, rng=rng)

        def forward(self, tensor):
            return self.second(self.activation(self.first(tensor)))

    batch = 8
    xs = [i / batch for i in range(batch)]
    targets = [2.0 * value - 1.0 for value in xs]
    x = make_cpu_tensor(xs, column_major(batch, 1))
    target = make_cpu_tensor(targets, column_major(batch, 1))

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = nn.SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(50):
        loss = criterion(model(x), target)
        losses.append(loss[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < 0.1 * losses[0]
