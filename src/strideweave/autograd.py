"""Functional automatic-differentiation interfaces."""

from collections.abc import Sequence

from .tensor import Tensor


def grad(
    output: Tensor,
    inputs: Sequence[Tensor],
    cotangents: Tensor,
    *,
    batched: bool = False,
    retain_graph: bool = False,
) -> tuple[Tensor | None, ...]:
    """Return vector-Jacobian products without modifying tensor ``.grad`` fields.

    When ``batched`` is false, ``cotangents`` must have exactly the output
    layout. When it is true, ``cotangents`` has one prepended leaf batch mode
    followed by the output layout. The returned gradient for each reachable
    input then has the same prepended batch mode; an unreachable input yields
    ``None``.

    Args:
        output: Differentiable tensor whose vector-Jacobian product is computed.
        inputs: Differentiable tensors for which gradients are requested.
        cotangents: One cotangent, or a tensor of cotangents when ``batched`` is
            true.
        batched: Whether the first cotangent mode indexes independent
            vector-Jacobian products.
        retain_graph: Whether to preserve saved graph state for another
            backward traversal.

    Returns:
        Tuple containing one gradient tensor or ``None`` per requested input.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape(2), sw.Stride(1))
        >>> x = sw.Tensor(sw.Generic([2.0, 3.0]), 0, layout)
        >>> cotangent = sw.Tensor(sw.Generic([1.0, 1.0]), 0, layout)
        >>> (gradient,) = sw.grad(x * 2.0, (x,), cotangent)
        >>> [gradient[i] for i in range(gradient.size())]
        [2.0, 2.0]
    """
    if isinstance(inputs, Tensor):
        raise TypeError("grad inputs must be a sequence of Tensors")
    return Tensor._functional_grad(
        output,
        tuple(inputs),
        cotangents,
        batched=batched,
        retain_graph=retain_graph,
    )


__all__ = [
    "grad",
]
