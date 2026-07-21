"""Train a tiny MLP regressor on sin(x) using the CPU carrier.

Runs a full-batch regression fitting ``y = sin(x)`` on 64 points in
``[-pi, pi]`` with a ``Linear(1, 16) -> Tanh -> Linear(16, 1)`` model,
``MSELoss``, and ``SGD``. Carrier tensors are deliberately built from raw
primitives (``CPU`` buffers filled element by element plus hand-written
``Layout`` objects) to exercise the low-level interface.

Usage:
    uv run python examples/train_mlp_cpu.py
"""

from __future__ import annotations

import math
import random

import strideweave as sw
import strideweave.nn as nn
from strideweave import CPU, Layout, Shape, Stride, Tensor


class MLP(sw.Module):
    """Two-layer tanh MLP mapping ``[batch, 1] -> [batch, 1]``."""

    def __init__(self, hidden: int, rng: random.Random) -> None:
        super().__init__()
        self.first = nn.Linear(1, hidden, rng=rng)
        self.activation = nn.Tanh()
        self.second = nn.Linear(hidden, 1, rng=rng)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.second(self.activation(self.first(tensor)))


def make_column_tensor(values: list[float]) -> Tensor:
    """Build a ``[len(values), 1]`` column tensor from raw primitives."""

    data = CPU(len(values))
    for i, value in enumerate(values):
        data[i] = value
    layout = Layout(Shape([len(values), 1]), Stride([1, len(values)]))
    return Tensor(data, 0, layout)


def main(epochs: int = 2000, lr: float = 0.05, log_every: int | None = 200) -> float:
    """Train the MLP and return the final training loss."""

    batch = 64
    xs = [-math.pi + 2.0 * math.pi * i / (batch - 1) for i in range(batch)]
    x = make_column_tensor(xs)
    target = make_column_tensor([math.sin(value) for value in xs])

    model = MLP(hidden=16, rng=random.Random(0))
    criterion = nn.MSELoss()
    optimizer = nn.SGD(model.parameters(), lr=lr)

    final_loss = math.inf
    for epoch in range(epochs):
        prediction = model(x)
        loss = criterion(prediction, target)
        final_loss = loss[0]
        if log_every is not None and epoch % log_every == 0:
            print(f"epoch {epoch:5d}  loss {final_loss:.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if log_every is not None:
        print(f"final loss {final_loss:.6f}")
    return final_loss


if __name__ == "__main__":
    final = main()
    assert final < 0.05, f"training did not converge: final loss {final:.6f}"
