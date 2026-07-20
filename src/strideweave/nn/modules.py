"""Layer and loss modules composed from strideweave primitive operations.

All 2-D activations and parameters follow the flat column-major convention
``Layout(Shape([rows, cols]), Stride([1, rows]))``. Inputs to :class:`Linear`
are ``[batch, in_features]`` tensors, weights are ``[out_features,
in_features]``, and matmul contracts the second mode of both operands, so no
transposes are needed.
"""

from __future__ import annotations

import random
from typing import Any

from ..carriers import CPU
from ..friendly import column_major, mean, ones
from ..functional import (
    elu,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    silu,
    softplus,
    tanh,
)
from ..module import Module, Parameter
from .init import kaiming_uniform_


def _require_flat_2mode(tensor: Any, name: str) -> tuple[int, int]:
    layout = tensor.layout
    if len(layout) != 2 or not layout.shape[0].is_int or not layout.shape[1].is_int:
        raise ValueError(f"{name} must be a flat two-mode tensor")
    return int(layout.shape[0]), int(layout.shape[1])


class Linear(Module):
    """Affine layer computing ``x @ weight + bias`` on flat two-mode tensors.

    The weight has shape ``[out_features, in_features]``; because matmul
    contracts the second mode of both operands, ``x[batch, in] @ weight[out,
    in]`` yields ``[batch, out]`` directly. The bias is stored as an
    ``[out_features, 1]`` parameter and broadcast without a broadcasting
    primitive by contracting a constant ones column against it:
    ``ones[batch, 1] @ bias[out, 1]`` produces a ``[batch, out]`` tile whose
    layout matches the matmul output, and whose backward pass sums the bias
    gradient over the batch.

    Parameters use CPU ``Float32`` storage and are initialized from
    ``uniform(-1/sqrt(in_features), 1/sqrt(in_features))``.

    Args:
        in_features: Size of the input feature mode.
        out_features: Size of the output feature mode.
        bias: Whether to learn an additive bias.
        rng: Random generator for parameter initialization; pass a seeded
            ``random.Random`` for reproducibility.
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import random
        >>> import sw.nn as nn
        >>> layer = nn.Linear(3, 2, rng=random.Random(0))
        >>> len(layer.parameters())
        2
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rng: random.Random | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        rng = rng if rng is not None else random.Random()

        weight_layout = column_major(out_features, in_features)
        self.weight = Parameter(CPU(weight_layout.cosize), 0, weight_layout)
        kaiming_uniform_(self.weight, in_features, rng=rng)

        if bias:
            bias_layout = column_major(out_features, 1)
            self.bias = Parameter(CPU(bias_layout.cosize), 0, bias_layout)
            kaiming_uniform_(self.bias, in_features, rng=rng)
        else:
            self.bias = None

    def forward(self, tensor: Any) -> Any:
        """Apply the affine transformation to a ``[batch, in_features]`` tensor.

        Args:
            tensor: Flat two-mode input tensor whose second mode extent equals
                ``in_features`` and whose backing carrier matches the
                parameters (CPU).

        Returns:
            ``[batch, out_features]`` tensor.

        Examples:
            >>> import random
            >>> import sw.nn as nn
            >>> from strideweave import CPU, Layout, Shape, Stride, Tensor
            >>> layer = nn.Linear(2, 4, rng=random.Random(0))
            >>> x = Tensor(CPU(6), 0, Layout(Shape([3, 2]), Stride([1, 3])))
            >>> layer(x).layout == Layout(Shape([3, 4]), Stride([1, 3]))
            True
        """

        batch, in_features = _require_flat_2mode(tensor, "tensor")
        if in_features != self.in_features:
            raise ValueError(
                f"tensor has {in_features} input features, expected {self.in_features}"
            )
        result = tensor @ self.weight
        if self.bias is not None:
            result = result + ones(batch, 1) @ self.bias
        return result


class MSELoss(Module):
    """Mean squared error between two flat two-mode tensors.

    The loss is composed from primitives: elementwise difference, square via
    ``pow``, reduction of all elements to the scalar layout ``Shape(1)``
    through the ``"a b -> 1"`` reduce description, and scaling by the
    reciprocal element count. The result supports ``backward()`` with an
    implicit unit gradient.

    Args:
        name: Optional module name override.

    Examples:
        >>> import sw.nn as nn
        >>> criterion = nn.MSELoss()
        >>> criterion.parameters()
        ()
    """

    def forward(self, prediction: Any, target: Any) -> Any:
        """Compute the mean squared error loss.

        Args:
            prediction: Flat two-mode prediction tensor.
            target: Target tensor with the same layout and backing carrier
                as ``prediction``. As an operation input it accumulates a
                (constant) gradient when it is a differentiable leaf.

        Returns:
            Scalar ``Shape(1)`` tensor holding the mean squared error.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import CPU, Layout, Shape, Stride, Tensor
            >>> layout = Layout(Shape([1, 2]), Stride([1, 1]))
            >>> pred = Tensor(CPU(2), 0, layout)
            >>> pred[0, 0] = 3.0
            >>> target = Tensor(CPU(2), 0, layout)
            >>> nn.MSELoss()(pred, target)[0]
            4.5
        """

        _require_flat_2mode(prediction, "prediction")
        if prediction.layout != target.layout:
            raise ValueError("prediction and target layouts must match")
        return mean((prediction - target) ** 2.0)


class ReLU(Module):
    """Module wrapper around :func:`strideweave.relu`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.ReLU().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the rectified linear unit elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with negative values replaced by zero.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([-1.0, 2.0]), 0, Layout(Shape(2), Stride(1)))
            >>> nn.ReLU()(x)[0]
            0.0
        """

        return relu(tensor)


class Sigmoid(Module):
    """Module wrapper around :func:`strideweave.sigmoid`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.Sigmoid().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the logistic sigmoid elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of sigmoid activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([0.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.Sigmoid()(x)[0]
            0.5
        """

        return sigmoid(tensor)


class Tanh(Module):
    """Module wrapper around :func:`strideweave.tanh`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.Tanh().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the hyperbolic tangent elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of tanh activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([0.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.Tanh()(x)[0]
            0.0
        """

        return tanh(tensor)


class GELU(Module):
    """Module wrapper around :func:`strideweave.gelu`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.GELU().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the Gaussian error linear unit elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of GELU activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([0.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.GELU()(x)[0]
            0.0
        """

        return gelu(tensor)


class SiLU(Module):
    """Module wrapper around :func:`strideweave.silu`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.SiLU().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the sigmoid-weighted linear unit elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of SiLU activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([0.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.SiLU()(x)[0]
            0.0
        """

        return silu(tensor)


class Softplus(Module):
    """Module wrapper around :func:`strideweave.softplus`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.Softplus().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the softplus function elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of softplus activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([0.0]), 0, Layout(Shape(1), Stride(1)))
            >>> round(nn.Softplus()(x)[0], 4)
            0.6931
        """

        return softplus(tensor)


class ELU(Module):
    """Module wrapper around :func:`strideweave.elu`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.ELU().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the exponential linear unit elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of ELU activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([1.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.ELU()(x)[0]
            1.0
        """

        return elu(tensor)


class LeakyReLU(Module):
    """Module wrapper around :func:`strideweave.leaky_relu`.

    Args:
        name: Optional module name override for parameter traversal.

    Examples:
        >>> import sw.nn as nn
        >>> nn.LeakyReLU().parameters()
        ()
    """

    def forward(self, tensor: Any) -> Any:
        """Apply the leaky rectified linear unit elementwise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor of leaky ReLU activations.

        Examples:
            >>> import sw.nn as nn
            >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
            >>> x = Tensor(Generic([2.0]), 0, Layout(Shape(1), Stride(1)))
            >>> nn.LeakyReLU()(x)[0]
            2.0
        """

        return leaky_relu(tensor)
