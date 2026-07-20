"""Optimizers for updating strideweave parameters in place.

Parameter updates mutate backing storage and therefore increment the shared
version counters. Any autograd graph built before a ``step()`` call cannot be
backwarded afterwards; the required ordering per iteration is forward ->
backward -> ``step()``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..functional import no_grad
from ..module import Parameter


class SGD:
    """Stochastic gradient descent over an explicit parameter collection.

    ``step()`` applies ``parameter -= lr * parameter.grad`` elementwise
    through each parameter's layout. Updates are in-place carrier writes, so
    they bump parameter versions: call ``step()`` only after ``backward()``
    for the current iteration, and rebuild the graph with a fresh forward
    pass afterwards. Gradients accumulate across ``backward()`` calls until
    ``zero_grad()`` resets them.

    Args:
        parameters: Parameters to update, e.g. ``module.parameters()``.
        lr: Positive learning rate scaling each gradient step.

    Examples:
        >>> import random
        >>> import sw.nn as nn
        >>> layer = nn.Linear(2, 2, rng=random.Random(0))
        >>> optimizer = nn.SGD(layer.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
    """

    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        self._parameters = tuple(parameters)
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = float(lr)

    def step(self) -> None:
        """Apply one gradient descent update to every parameter with a grad.

        Parameters whose ``grad`` is ``None`` are skipped. The update writes
        directly into parameter storage, incrementing its version, so graphs
        built before this call can no longer be backwarded.

        Args:
            None.

        Returns:
            ``None``.

        Examples:
            >>> import random
            >>> import sw.nn as nn
            >>> layer = nn.Linear(2, 2, rng=random.Random(0))
            >>> nn.SGD(layer.parameters(), lr=0.1).step()  # no grads: no-op
        """

        with no_grad():
            for parameter in self._parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                for i in range(parameter.layout.size):
                    physical = parameter.offset + parameter.layout.index(i)
                    gradient_value = gradient.carrier[
                        gradient.offset + gradient.layout.index(i)
                    ]
                    parameter.carrier[physical] = (
                        parameter.carrier[physical] - self.lr * gradient_value
                    )

    def zero_grad(self) -> None:
        """Reset every tracked parameter's gradient to ``None``.

        Args:
            None.

        Returns:
            ``None``.

        Examples:
            >>> import random
            >>> import sw.nn as nn
            >>> layer = nn.Linear(2, 2, rng=random.Random(0))
            >>> nn.SGD(layer.parameters(), lr=0.1).zero_grad()
        """

        for parameter in self._parameters:
            parameter.grad = None
