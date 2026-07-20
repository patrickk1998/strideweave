"""Parameter initialization helpers for ``neotorch.nn`` modules.

These helpers write directly into a parameter's backing storage through its
layout, so they work for any physical layout. They are intended for module
construction time, before any forward pass builds an autograd graph over the
parameter.
"""

from __future__ import annotations

import math
import random

from ..module import Parameter


def fill_(parameter: Parameter, value: float) -> None:
    """Fill every logical element of a parameter with one value.

    Writes go through the parameter's layout, so only the logical elements
    are touched regardless of the physical stride pattern.

    Args:
        parameter: Parameter whose storage is filled in place.
        value: Value written to every logical element.

    Returns:
        ``None``.

    Examples:
        >>> import neotorch
        >>> from neotorch import CPU, Layout, Parameter, Shape, Stride, Tensor
        >>> from neotorch.nn.init import fill_
        >>> weight = Parameter(CPU(2), 0, Layout(Shape(2), Stride(1)))
        >>> fill_(weight, 3.0)
        >>> weight[1]
        3.0
    """

    for i in range(parameter.layout.size):
        parameter.data[parameter.offset + parameter.layout.index(i)] = value


def kaiming_uniform_(parameter: Parameter, fan_in: int, *, rng: random.Random) -> None:
    """Fill a parameter with Kaiming-style uniform values in place.

    Every logical element is drawn independently from
    ``uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))``.

    Args:
        parameter: Parameter whose storage is filled in place.
        fan_in: Positive input feature count that scales the uniform bound.
        rng: Random generator supplying the samples; pass a seeded
            ``random.Random`` for reproducible initialization.

    Returns:
        ``None``.

    Examples:
        >>> import random
        >>> import neotorch
        >>> from neotorch import CPU, Layout, Parameter, Shape, Stride
        >>> from neotorch.nn.init import kaiming_uniform_
        >>> weight = Parameter(CPU(4), 0, Layout(Shape([2, 2]), Stride([1, 2])))
        >>> kaiming_uniform_(weight, 2, rng=random.Random(0))
        >>> abs(weight[0, 0]) <= 1 / 2**0.5
        True
    """

    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    bound = 1.0 / math.sqrt(fan_in)
    for i in range(parameter.layout.size):
        parameter.data[parameter.offset + parameter.layout.index(i)] = rng.uniform(
            -bound, bound
        )
