"""Train a tiny MLP on sin(x) using the sw.friendly helpers.

Same task as ``train_mlp_cpu.py`` — full-batch regression of ``y = sin(x)``
on 64 points with a ``Linear(1, 16) -> Tanh -> Linear(16, 1)`` model — but
data construction and value extraction go through ``sw.friendly``
instead of raw ``CPU`` buffers and hand-written layouts.

Usage:
    uv run python examples/train_mlp_cpu_friendly.py
"""

from __future__ import annotations

import math
import random

import strideweave as sw
import strideweave.friendly as F
import strideweave.nn as nn


class MLP(sw.Module):
    """Two-layer tanh MLP mapping ``[batch, 1] -> [batch, 1]``."""

    def __init__(self, hidden: int, rng: random.Random) -> None:
        super().__init__()
        self.first = nn.Linear(1, hidden, rng=rng)
        self.activation = nn.Tanh()
        self.second = nn.Linear(hidden, 1, rng=rng)

    def forward(self, tensor: sw.Tensor) -> sw.Tensor:
        return self.second(self.activation(self.first(tensor)))


def main(epochs: int = 2000, lr: float = 0.05, log_every: int | None = 200) -> float:
    """Train the MLP and return the final training loss."""

    batch = 64
    xs = [-math.pi + 2.0 * math.pi * i / (batch - 1) for i in range(batch)]
    x = F.tensor([[value] for value in xs])
    target = F.tensor([[math.sin(value)] for value in xs])

    model = MLP(hidden=16, rng=random.Random(0))
    criterion = nn.MSELoss()
    optimizer = nn.SGD(model.parameters(), lr=lr)

    final_loss = math.inf
    for epoch in range(epochs):
        loss = criterion(model(x), target)
        final_loss = F.item(loss)
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
