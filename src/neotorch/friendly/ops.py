"""Ergonomic reduction and extraction helpers composed from primitives.

``sum`` and ``mean`` wrap the reduce-description recipe needed to collapse a
flat tensor to the scalar ``Shape(1)`` layout that supports implicit
``backward()``. ``item`` and ``to_list`` read logical values back out of any
tensor without hand-written index loops.
"""

from __future__ import annotations

from string import ascii_lowercase
from typing import Any

from ..functional import mul, reduce


def _require_flat_modes(tensor: Any, name: str) -> int:
    layout = tensor.layout
    modes = len(layout)
    if modes == 0 or any(not layout.shape[i].is_int for i in range(modes)):
        raise ValueError(f"{name} must be a flat tensor with integer modes")
    return modes


def sum(tensor: Any) -> Any:  # noqa: A001 - mirrors the PyTorch name
    """Sum every element of a flat tensor into a scalar ``Shape(1)`` tensor.

    The reduction runs through a generated ``"a b ... -> 1"`` reduce
    description, so the result has the exact scalar layout that permits
    ``backward()`` with an implicit unit gradient. The input must have flat
    integer top-level modes.

    Args:
        tensor: Flat tensor whose elements are summed.

    Returns:
        Scalar ``Shape(1)`` tensor holding the total.

    Examples:
        >>> from neotorch.friendly import sum, tensor
        >>> sum(tensor([[1.0, 2.0], [3.0, 4.0]]))[0]
        10.0
    """

    modes = _require_flat_modes(tensor, "tensor")
    if modes > len(ascii_lowercase):
        raise ValueError("tensor has too many modes to reduce")
    description = " ".join(ascii_lowercase[:modes]) + " -> 1"
    return reduce(tensor, description)


def mean(tensor: Any) -> Any:
    """Average every element of a flat tensor into a scalar tensor.

    Composed as ``sum(tensor)`` scaled by the reciprocal element count, so
    the result keeps the scalar ``Shape(1)`` layout and full autograd
    support.

    Args:
        tensor: Flat tensor whose elements are averaged.

    Returns:
        Scalar ``Shape(1)`` tensor holding the mean.

    Examples:
        >>> from neotorch.friendly import mean, tensor
        >>> mean(tensor([1.0, 2.0, 3.0]))[0]
        2.0
    """

    return mul(sum(tensor), 1.0 / tensor.layout.size)


def item(tensor: Any) -> Any:
    """Extract the single value of a size-one tensor.

    Args:
        tensor: Tensor whose layout has logical size one, such as a loss
            produced by ``sum`` or ``mean``.

    Returns:
        The tensor's only element as a Python value.

    Examples:
        >>> from neotorch.friendly import item, sum, tensor
        >>> item(sum(tensor([1.0, 2.0])))
        3.0
    """

    if tensor.layout.size != 1:
        raise ValueError("item requires a tensor with exactly one element")
    return tensor.data.get_value(tensor.offset + tensor.layout.index(0))


def to_list(tensor: Any) -> list[Any]:
    """Read every logical element of a tensor into a flat Python list.

    Elements are enumerated in logical index order (first mode fastest), the
    same order used by layout logical indexing.

    Args:
        tensor: Tensor whose values are extracted.

    Returns:
        List of the tensor's logical values.

    Examples:
        >>> from neotorch.friendly import tensor, to_list
        >>> to_list(tensor([[1.0, 2.0], [3.0, 4.0]]))
        [1.0, 3.0, 2.0, 4.0]
    """

    layout = tensor.layout
    return [
        tensor.data.get_value(tensor.offset + layout.index(i))
        for i in range(layout.size)
    ]
