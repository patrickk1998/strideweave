"""Public tensor operation functions dispatched to data-backend operations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, cast, overload

from ..core.layout import Tree
from ..data.operation_helpers import _as_tensor

_REDUCE_DESCRIPTION_MISSING = object()

_operation = import_module("neotorch._operation")
_is_grad_enabled = cast(Callable[[], bool], _operation.is_grad_enabled)
_set_grad_enabled = cast(Callable[[bool], None], _operation.set_grad_enabled)

__all__ = [
    "add",
    "div",
    "einsum",
    "elementwise_mul",
    "elu",
    "exp",
    "gelu",
    "is_grad_enabled",
    "leaky_relu",
    "matmul",
    "mul",
    "no_grad",
    "permute",
    "pow",
    "rearrange",
    "reduce",
    "relu",
    "set_grad_enabled",
    "sigmoid",
    "silu",
    "softplus",
    "tanh",
    "view",
]


def _dispatch_unary(operation_name: str, tensor: Any) -> Any:
    tensor = _as_tensor(tensor, "tensor")
    return type(tensor.data).dispatch_op(operation_name)


def _dispatch_binary(operation_name: str, lhs: Any, rhs: Any) -> Any:
    lhs = _as_tensor(lhs, "lhs")
    rhs = _as_tensor(rhs, "rhs")
    if type(lhs.data) is not type(rhs.data):
        raise TypeError("Tensor backing data classes must match")
    return type(lhs.data).dispatch_op(operation_name)


def _reduce_second_mode(tensor: Any) -> Any:
    return _dispatch_unary("reduce", tensor).forward(tensor)


def _matmul_2mode(lhs: Any, rhs: Any) -> Any:
    return _dispatch_binary("matmul", lhs, rhs).forward(lhs, rhs)


def _rearrange_tree(tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
    return _dispatch_unary("rearrange", tensor).forward(tensor, output, selection)


def is_grad_enabled() -> bool:
    """Return whether autograd graph construction is enabled.

    The value is thread-local and controls whether operation calls attach
    autograd context to their result tensors.

    Args:
        None.

    Returns:
        ``True`` when operations build an autograd graph in the current thread.

    Examples:
        >>> import neotorch
        >>> neotorch.is_grad_enabled()
        True
    """

    return _is_grad_enabled()


def set_grad_enabled(enabled: bool) -> None:
    """Set the current thread's autograd graph construction state.

    Args:
        enabled: ``True`` to build autograd graphs for future operations in the
            current thread, or ``False`` to skip graph construction.

    Returns:
        ``None``.

    Examples:
        >>> import neotorch
        >>> previous = neotorch.is_grad_enabled()
        >>> neotorch.set_grad_enabled(False)
        >>> neotorch.set_grad_enabled(previous)
    """

    _set_grad_enabled(enabled)


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable autograd graph construction.

    The previous thread-local grad-enabled state is restored when the context
    exits, including when the block raises.

    Args:
        None.

    Returns:
        Context manager that yields ``None`` while gradients are disabled.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> with neotorch.no_grad():
        ...     y = neotorch.mul(x, 2)
        >>> y.autograd_ctx is None
        True
    """

    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)


def add(lhs: Any, rhs: Any) -> Any:
    """Add two tensors with matching layouts.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand with the same layout and backing data class.

    Returns:
        Tensor containing the elementwise sum.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([1, 2]), 0, layout)
        >>> y = Tensor(Generic([3, 4]), 0, layout)
        >>> neotorch.add(x, y)[1]
        6
    """

    return _dispatch_binary("add", lhs, rhs).forward(lhs, rhs)


def elementwise_mul(lhs: Any, rhs: Any) -> Any:
    """Multiply two tensors elementwise.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand with the same layout and backing data class.

    Returns:
        Tensor containing elementwise products.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([2, 3]), 0, layout)
        >>> y = Tensor(Generic([4, 5]), 0, layout)
        >>> neotorch.elementwise_mul(x, y)[1]
        15
    """

    return _dispatch_binary("elementwise_mul", lhs, rhs).forward(lhs, rhs)


def mul(tensor: Any, scalar: Any) -> Any:
    """Multiply a tensor by a scalar or another tensor.

    Tensor inputs dispatch to elementwise multiplication; non-tensor scalar
    inputs dispatch to scalar multiplication for the tensor's data backend.

    Args:
        tensor: Tensor operand to scale or multiply elementwise.
        scalar: Numerical scalar or tensor operand.

    Returns:
        Tensor containing scaled or elementwise multiplied values.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.mul(x, 10)[1]
        30
    """

    from ..core.tensor import Tensor

    if isinstance(scalar, Tensor):
        return elementwise_mul(tensor, scalar)
    return _dispatch_unary("mul", tensor).forward(tensor, scalar)


def div(lhs: Any, rhs: Any) -> Any:
    """Divide two tensors elementwise.

    Args:
        lhs: Numerator tensor.
        rhs: Denominator tensor with the same layout and backing data class.

    Returns:
        Tensor containing elementwise quotients.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([8, 9]), 0, layout)
        >>> y = Tensor(Generic([2, 3]), 0, layout)
        >>> neotorch.div(x, y)[1]
        3.0
    """

    return _dispatch_binary("div", lhs, rhs).forward(lhs, rhs)


def exp(tensor: Any) -> Any:
    """Apply the exponential function elementwise.

    Args:
        tensor: Tensor whose logical values should be exponentiated.

    Returns:
        Tensor containing ``math.exp`` applied to each element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.exp(x)[0]
        1.0
    """

    return _dispatch_unary("exp", tensor).forward(tensor)


def relu(tensor: Any) -> Any:
    """Apply the rectified linear unit function elementwise.

    ReLU maps negative values to ``0`` and keeps positive values unchanged. Its
    autograd derivative is ``0`` for values less than or equal to ``0`` and
    ``1`` for values greater than ``0``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing ``max(0, value)`` for each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([-1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.relu(x)[1]
        2
    """

    return _dispatch_unary("relu", tensor).forward(tensor)


def sigmoid(tensor: Any) -> Any:
    """Apply the logistic sigmoid function elementwise.

    Sigmoid maps each value ``x`` to ``1 / (1 + math.exp(-x))``. Its autograd
    derivative multiplies the incoming gradient by ``sigmoid(x) * (1 -
    sigmoid(x))``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the logistic sigmoid of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.sigmoid(x)[0]
        0.5
    """

    return _dispatch_unary("sigmoid", tensor).forward(tensor)


def tanh(tensor: Any) -> Any:
    """Apply the hyperbolic tangent function elementwise.

    Tanh maps each value ``x`` to ``math.tanh(x)``. Its autograd derivative
    multiplies the incoming gradient by ``1 - tanh(x) ** 2``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the hyperbolic tangent of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.tanh(x)[0]
        0.0
    """

    return _dispatch_unary("tanh", tensor).forward(tensor)


def gelu(tensor: Any) -> Any:
    """Apply the exact Gaussian error linear unit function elementwise.

    GELU maps each value ``x`` to ``0.5 * x * (1 + erf(x / sqrt(2)))`` using
    PyTorch's default exact formula. Its autograd derivative multiplies the
    incoming gradient by ``0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 *
    x**2) / sqrt(2 * pi)``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the exact GELU value of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.gelu(x)[0]
        0.0
    """

    return _dispatch_unary("gelu", tensor).forward(tensor)


def silu(tensor: Any) -> Any:
    """Apply the sigmoid linear unit function elementwise.

    SiLU maps each value ``x`` to ``x * sigmoid(x)``. Its autograd derivative
    multiplies the incoming gradient by ``sigmoid(x) + x * sigmoid(x) * (1 -
    sigmoid(x))``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the SiLU value of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.silu(x)[0]
        0.0
    """

    return _dispatch_unary("silu", tensor).forward(tensor)


def softplus(tensor: Any) -> Any:
    """Apply the softplus function elementwise.

    Softplus maps each value ``x`` to ``log(1 + exp(x))`` using a numerically
    stable equivalent formula. Its autograd derivative multiplies the incoming
    gradient by ``sigmoid(x)``.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the softplus value of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> round(neotorch.softplus(x)[0], 6)
        0.693147
    """

    return _dispatch_unary("softplus", tensor).forward(tensor)


def elu(tensor: Any) -> Any:
    """Apply the exponential linear unit function elementwise.

    ELU uses PyTorch's default ``alpha=1.0``. It maps ``x`` to ``x`` when
    ``x > 0`` and to ``exp(x) - 1`` otherwise. Its autograd derivative
    multiplies the incoming gradient by ``1`` when ``x > 0`` and by ``exp(x)``
    otherwise.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the ELU value of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.elu(x)[0]
        0.0
    """

    return _dispatch_unary("elu", tensor).forward(tensor)


def leaky_relu(tensor: Any) -> Any:
    """Apply the leaky rectified linear unit function elementwise.

    Leaky ReLU uses PyTorch's default negative slope ``0.01``. It maps ``x`` to
    ``x`` when ``x >= 0`` and to ``0.01 * x`` otherwise. Its autograd
    derivative multiplies the incoming gradient by ``1`` when ``x >= 0`` and by
    ``0.01`` otherwise.

    Args:
        tensor: Tensor whose logical values should be transformed.

    Returns:
        Tensor containing the leaky ReLU value of each input element.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([-2]), 0, Layout(Shape(1), Stride(1)))
        >>> neotorch.leaky_relu(x)[0]
        -0.02
    """

    return _dispatch_unary("leaky_relu", tensor).forward(tensor)


def pow(tensor: Any, exponent: Any) -> Any:
    """Raise each tensor element to a scalar exponent.

    Args:
        tensor: Tensor containing base values.
        exponent: Numerical scalar exponent.

    Returns:
        Tensor containing each element raised to ``exponent``.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> neotorch.pow(x, 3)[1]
        27
    """

    return _dispatch_unary("pow", tensor).forward(tensor, exponent)


@overload
def reduce(tensor: Any) -> Any: ...


@overload
def reduce(tensor: Any, description: str) -> Any: ...


def reduce(tensor: Any, description: Any = _REDUCE_DESCRIPTION_MISSING) -> Any:
    """Sum-reduce a tensor.

    With no description, the tensor must have two top-level modes and the second
    mode is summed. With a string description, omitted dimensions are reduced
    through the Neotorch hierarchical-layout lowering path.

    Args:
        tensor: Tensor to reduce.
        description: Optional Neotorch layout reduce command.

    Returns:
        Tensor containing the sum-reduced values.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.reduce(x)[1]
        12
        >>> neotorch.reduce(x, "a b -> a")[1]
        12
    """

    if description is _REDUCE_DESCRIPTION_MISSING:
        return _reduce_second_mode(tensor)
    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from ..einops import reduce as einops_reduce

    return einops_reduce(tensor, description)


def matmul(lhs: Any, rhs: Any) -> Any:
    """Multiply two two-mode tensors.

    The first mode of each input is kept, and the second modes must have the
    same logical size and are contracted with a dot product.

    Args:
        lhs: Left two-mode tensor.
        rhs: Right two-mode tensor with matching second-mode logical size.

    Returns:
        Tensor with layout formed from the first mode of each input.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.matmul(lhs, rhs)[1, 1]
        22
    """

    return _matmul_2mode(lhs, rhs)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    """Contract two tensors using a Neotorch contraction description.

    The string form is parsed by ``neotorch.einops`` and lowered into
    rearrange, matmul, and final rearrange operations.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor containing the requested contraction result.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.einsum(lhs, rhs, "a b, c b -> a c")[1, 1]
        22
    """

    if not isinstance(description, str):
        raise TypeError("description must be a str")

    from ..einops import einsum as einops_einsum

    return einops_einsum(lhs, rhs, description)


@overload
def rearrange(tensor: Any, output: str) -> Any: ...


@overload
def rearrange(tensor: Any, output: Tree, selection: Tree | None = None) -> Any: ...


def rearrange(tensor: Any, output: Tree | str, selection: Tree | None = None) -> Any:
    """Rearrange a tensor layout.

    Tree inputs call the lower-level rearrange operation directly. A string
    output is parsed as a Neotorch layout command and must not be combined with
    an explicit selection.

    Args:
        tensor: Tensor whose layout should be rearranged.
        output: Output Tree or Neotorch layout rearrange description.
        selection: Optional Tree selecting source layout subtrees.

    Returns:
        Tensor view with the rearranged layout.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Node, Shape, Stride, Tensor, Tree
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.rearrange(x, "a b -> b a")[2, 1]
        6
        >>> output = Tree(Node.id(1), Node.id(0))
        >>> selection = Tree(Node.Leaf, Node.Leaf)
        >>> neotorch.rearrange(x, output, selection)[2, 1]
        6
    """

    if isinstance(output, str):
        if selection is not None:
            raise TypeError(
                "String rearrange descriptions do not accept an explicit selection"
            )
        from ..einops import rearrange as einops_rearrange

        return einops_rearrange(tensor, output)
    return _rearrange_tree(tensor, output, selection)


def permute(tensor: Any, *order: Any) -> Any:
    """Permute top-level tensor layout modes.

    Args:
        tensor: Tensor whose layout modes should be reordered.
        order: Permutation of every top-level mode index.

    Returns:
        Tensor view with top-level modes in the requested order.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> neotorch.permute(x, 1, 0)[2, 1]
        6
    """

    return _dispatch_unary("permute", tensor).forward(tensor, *order)


def view(tensor: Any, key: Any) -> Any:
    """Create a tensor view from integer and slice keys.

    Integers select and remove a mode where supported; slices preserve a mode
    while adjusting the output layout and offset.

    Args:
        tensor: Tensor to view.
        key: Integer and slice key tuple for top-level modes.

    Returns:
        Tensor view sharing the input backing data with an adjusted layout.

    Examples:
        >>> import neotorch
        >>> from neotorch import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> row = neotorch.view(x, (1, slice(None)))
        >>> row[2]
        6
    """

    return _dispatch_unary("view", tensor).forward(tensor, key)
