"""Public tensor operations dispatched through their input carriers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib import import_module
from operator import index as operator_index
from typing import Any, NamedTuple, cast, overload

from ..carriers.dtype import SimpleDType
from ..carriers.operation_helpers import _as_tensor
from ..carriers.operation_policy import operation_execution_options
from ..core.layout import Node, Shape, Stride, Tree

_operation = import_module("strideweave._operation")
_is_grad_enabled = cast(Callable[[], bool], _operation.is_grad_enabled)
_set_grad_enabled = cast(Callable[[bool], None], _operation.set_grad_enabled)

__all__ = [
    "SortResult",
    "TopKResult",
    "abs",
    "add",
    "argmax",
    "argmin",
    "as_strided",
    "broadcast_in_dim",
    "broadcast_to",
    "ceil",
    "clamp",
    "conv_general",
    "cos",
    "cumsum",
    "div",
    "einsum",
    "elementwise_mul",
    "elu",
    "eq",
    "erf",
    "exp",
    "exp2",
    "floor",
    "gather",
    "ge",
    "gelu",
    "gt",
    "is_grad_enabled",
    "le",
    "leaky_relu",
    "log",
    "log2",
    "logical_not",
    "lt",
    "matmul",
    "maximum",
    "minimum",
    "move",
    "mul",
    "ne",
    "neg",
    "no_grad",
    "permute",
    "pow",
    "rearrange",
    "recip",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "relu",
    "rem",
    "reshape",
    "round",
    "rsqrt",
    "scatter",
    "scatter_add",
    "select",
    "set_grad_enabled",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "softplus",
    "sort",
    "sqrt",
    "squeeze",
    "sub",
    "tanh",
    "topk",
    "unsqueeze",
]


class SortResult(NamedTuple):
    """Values and source indices returned by :func:`sort`.

    The tuple supports both attribute access (``result.values`` and
    ``result.indices``) and ordinary two-item unpacking.

    Args:
        values: Sorted value tensor.
        indices: Int32 tensor of source ordinals corresponding to ``values``.

    Examples:
        >>> result = SortResult("sorted", "source indices")
        >>> result.values, result.indices
        ('sorted', 'source indices')
    """

    values: Any
    indices: Any


class TopKResult(NamedTuple):
    """Selected values and source indices returned by :func:`topk`.

    The tuple supports both attribute access (``result.values`` and
    ``result.indices``) and ordinary two-item unpacking.

    Args:
        values: Selected and sorted value tensor.
        indices: Int32 tensor of source ordinals corresponding to ``values``.

    Examples:
        >>> result = TopKResult("selected", "source indices")
        >>> result.values, result.indices
        ('selected', 'source indices')
    """

    values: Any
    indices: Any


def _dispatch_unary(operation_name: str, tensor: Any) -> Any:
    tensor = _as_tensor(tensor, "tensor")
    return tensor.carrier.dispatch_op(operation_name)


def _dispatch_binary(operation_name: str, lhs: Any, rhs: Any) -> Any:
    lhs = _as_tensor(lhs, "lhs")
    rhs = _as_tensor(rhs, "rhs")
    if type(lhs.carrier) is not type(rhs.carrier):
        raise TypeError("Tensor backing carriers must match")
    return lhs.carrier.dispatch_op(operation_name)


def _accumulator_options(
    operation_name: str, accumulator_dtype: SimpleDType | None
) -> Any:
    """Validate an optional accumulator request into execution options."""
    if accumulator_dtype is None:
        return None
    return operation_execution_options(
        operation_name, accumulator_dtype=accumulator_dtype
    )


def _reduce_second_mode(
    tensor: Any, *, accumulator_dtype: SimpleDType | None = None
) -> Any:
    options = _accumulator_options("reduce_sum", accumulator_dtype)
    operation = _dispatch_unary("reduce_sum", tensor)
    if options is None:
        return operation.forward(tensor)
    return operation.forward(tensor, options=options)


def _matmul_2mode(
    lhs: Any, rhs: Any, *, accumulator_dtype: SimpleDType | None = None
) -> Any:
    options = _accumulator_options("matmul", accumulator_dtype)
    operation = _dispatch_binary("matmul", lhs, rhs)
    if options is None:
        return operation.forward(lhs, rhs)
    return operation.forward(lhs, rhs, options=options)


def _rearrange_tree(tensor: Any, output: Tree, selection: Tree | None = None) -> Any:
    return _dispatch_unary("rearrange", tensor).forward(tensor, output, selection)


def _normalize_top_level_axis(axis: Any, rank: int, *, name: str = "axis") -> int:
    """Normalize one explicit top-level axis without accepting booleans."""

    if isinstance(axis, bool):
        raise TypeError(f"{name} must be an integer top-level mode")
    try:
        normalized = operator_index(axis)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer top-level mode") from exc
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"{name} {normalized} is out of range for rank {rank}")
    return normalized


def _normalize_permutation(spec: Any, rank: int, name: str) -> tuple[int, ...]:
    """Normalize an explicit top-level mode permutation."""

    if spec is None:
        return tuple(range(rank))
    if isinstance(spec, (str, bytes)):
        raise TypeError(f"{name} must be a permutation of top-level mode indices")
    try:
        values = tuple(operator_index(value) for value in spec)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a permutation of top-level mode indices"
        ) from exc
    if any(value < 0 or value >= rank for value in values):
        raise ValueError(f"{name} must be a permutation of all top-level modes")
    if len(values) != rank or set(values) != set(range(rank)):
        raise ValueError(f"{name} must be a permutation of all top-level modes")
    return values


def _reduction_intermediate(tensor: Any, axis: Any) -> Any:
    """Move one top-level mode into the second reduction mode.

    Reduction carrier operations intentionally consume a two-mode tensor.  The
    first mode is a hierarchical grouping of every mode other than ``axis``;
    the selected axis becomes the second mode.  This preserves StrideWeave's
    hierarchy and keeps reductions independent of flat-layout rank inference.
    """

    tensor = _as_tensor(tensor, "tensor")
    rank = len(tensor.layout)
    normalized_axis = _normalize_top_level_axis(axis, rank)
    if rank < 2:
        raise ValueError("reduction requires at least two top-level modes")
    kept = [index for index in range(rank) if index != normalized_axis]
    kept_tree: Tree | Any
    if len(kept) == 1:
        kept_tree = Tree(Node.id(kept[0]))
    else:
        kept_tree = Tree(*(Node.id(index) for index in kept))
    output = Tree(kept_tree, Node.id(normalized_axis))
    selection = Tree(*(Node.Leaf for _ in range(rank)))
    return _rearrange_tree(tensor, output, selection)


def _reduce_named_axis(operation_name: str, tensor: Any, axis: Any) -> Any:
    """Lower a named two-mode reduction over an explicit top-level axis."""

    intermediate = _reduction_intermediate(tensor, axis)
    return _dispatch_unary(operation_name, intermediate).forward(intermediate)


def _reduce_description(
    operation_name: str,
    tensor: Any,
    description: str,
    *,
    accumulator_dtype: SimpleDType | None = None,
) -> Any:
    """Lower a reduction description and dispatch the requested operation."""

    if not isinstance(description, str):
        raise TypeError("description must be a str")
    from ..einops import parse_reduce

    options = _accumulator_options(operation_name, accumulator_dtype)
    spec = parse_reduce(description)
    intermediate = _rearrange_tree(tensor, spec.rearrange_output, spec.selection)
    operation = _dispatch_unary(operation_name, intermediate)
    if options is None:
        return operation.forward(intermediate)
    return operation.forward(intermediate, options=options)


def is_grad_enabled() -> bool:
    """Return whether autograd graph construction is enabled.

    The value is thread-local and controls whether operation calls attach
    autograd context to their result tensors.

    Args:
        None.

    Returns:
        ``True`` when operations build an autograd graph in the current thread.

    Examples:
        >>> import strideweave as sw
        >>> sw.is_grad_enabled()
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
        >>> import strideweave as sw
        >>> previous = sw.is_grad_enabled()
        >>> sw.set_grad_enabled(False)
        >>> sw.set_grad_enabled(previous)
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> with sw.no_grad():
        ...     y = sw.mul(x, 2)
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
    """Add two structurally broadcast-compatible tensors.

    Shape trees must share a profile. At each leaf, equal extents are matched
    and an extent of one expands to its peer using stride zero. No rank
    alignment, flattening, insertion, or reordering is inferred.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand on the same backing carrier and with a
            structurally broadcast-compatible shape.

    Returns:
        Tensor containing the elementwise sum.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([1, 2]), 0, layout)
        >>> y = Tensor(Generic([3, 4]), 0, layout)
        >>> sw.add(x, y)[1]
        6
    """

    return _dispatch_binary("add", lhs, rhs).forward(lhs, rhs)


def broadcast_to(tensor: Any, target: Shape) -> Any:
    """Broadcast singleton leaves to a structurally congruent target shape.

    Forward returns a zero-copy view whose widened leaves have stride zero.
    Backward sums the incoming cotangent over those widened leaves and restores
    the input shape and layout. The target must have the same hierarchical
    profile, and only an input extent of one may widen; no rank alignment,
    insertion, removal, flattening, or reordering is inferred.

    Args:
        tensor: Tensor to broadcast.
        target: Hierarchical target shape with the same profile as the input.

    Returns:
        Differentiable stride-zero view with shape ``target``.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
        >>> x = sw.Tensor(sw.Generic([3.0, 4.0]), 0, layout)
        >>> y = sw.broadcast_to(x, sw.Shape([3, 2]))
        >>> [y[i, 1] for i in range(3)]
        [4.0, 4.0, 4.0]
    """

    tensor = _as_tensor(tensor, "tensor")
    if not isinstance(target, Shape):
        raise TypeError("target must be a Shape")
    return _dispatch_unary("broadcast_to", tensor).forward(tensor, target)


def as_strided(tensor: Any, shape: Shape, stride: Stride) -> Any:
    """Create a zero-copy view through an explicit logical-coordinate mapping.

    ``shape`` and ``stride`` define a layout ``B`` from output ``c_0``
    coordinates to flattened input ``c_0`` ordinals. They do not directly
    specify the view's physical placement stride. The result composes ``B``
    through every layout whose domain is ``c_0``: its placement is
    ``Layout.compose(L_0, B)``, and a multi-subtensor representation likewise
    uses ``Layout.compose(S_0, B)`` for its first adjacent layout. Deeper
    placement and adjacent layouts are preserved.

    Unlike PyTorch ``as_strided``, this operation does not reinterpret physical
    storage offsets. ``B`` is origin-based, must be injective, and must remain
    inside the input logical coordinate domain. The composed placement must
    also be injective, so overlapping or stride-zero views are deferred. A
    composition that cannot be represented by a hierarchical ``Layout`` is
    rejected instead of being approximated.

    Args:
        tensor: Tensor whose carrier and offset are shared by the view.
        shape: Hierarchical output shape of logical mapping ``B``.
        stride: Hierarchical logical-mapping stride of ``B``, with the same
            structure as ``shape``; it is not necessarily the output placement
            stride.

    Returns:
        Tensor view whose ``c_0`` layouts are composed with ``B`` without
        copying carrier storage.

    Semantics:
        For every output coordinate ``q``, the view reads input logical ordinal
        ``B(q)``. Its physical address is therefore ``L_0(B(q))``. In a
        multi-subtensor representation, ``S_0(B(q))`` selects the corresponding
        next-level coordinate.

    Mode assumptions:
        ``shape`` and ``stride`` describe the same hierarchy. ``B`` is
        injective and origin-based, ``B.cosize <= tensor.size()``, and all
        composed placements remain injective. No rank alignment, flattening,
        mode insertion, reordering, or physical-offset reinterpretation is
        inferred.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape([5, 4]), sw.Stride([4, 1]))
        >>> x = sw.Tensor(sw.Generic([float(i) for i in range(20)]), 0, layout)
        >>> y = sw.as_strided(x, sw.Shape([2, 2]), sw.Stride([1, 2]))
        >>> y.layout == sw.Layout(sw.Shape([2, 2]), sw.Stride([4, 8]))
        True
        >>> [y[i, j] for j in range(2) for i in range(2)]
        [0.0, 4.0, 8.0, 12.0]
    """

    tensor = _as_tensor(tensor, "tensor")
    if not isinstance(shape, Shape):
        raise TypeError("shape must be a Shape")
    if not isinstance(stride, Stride):
        raise TypeError("stride must be a Stride")
    return tensor.carrier.dispatch_op("as_strided").forward(tensor, shape, stride)


def reshape(tensor: Any, shape: Shape) -> Any:
    """Create a zero-copy view with a compatible hierarchical shape.

    Args:
        tensor: Tensor to reshape.
        shape: Target shape with the same logical element count.

    Returns:
        Tensor view with ``shape`` when the source layout is reshape-compatible.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([1.0, 2.0, 3.0, 4.0])
        >>> sw.reshape(x, sw.Shape([2, 2]))[1, 1]
        4.0
    """

    tensor = _as_tensor(tensor, "tensor")
    if not isinstance(shape, Shape):
        raise TypeError("shape must be a Shape")
    return tensor.carrier.dispatch_op("reshape").forward(tensor, shape)


def unsqueeze(tensor: Any, dim: Any) -> Any:
    """Insert one extent-one top-level mode as a zero-copy view.

    The new mode is inserted in the outermost hierarchical coordinate space;
    nested insertion is intentionally deferred. Negative dimensions are
    normalized against the insertion range ``0 .. rank``. The inserted mode
    uses canonical stride zero and is removed by the inverse VJP.

    Args:
        tensor: Tensor whose top-level layout should gain a singleton mode.
        dim: Insertion position, including the normalized negative positions
            from ``-1`` through ``-rank-1``.

    Returns:
        Tensor view sharing the input representation with one new singleton
        top-level mode.

    Examples:
        >>> import strideweave as sw
        >>> x = sw.Tensor(sw.Generic([1.0, 2.0]), 0,
        ...               sw.Layout(sw.Shape(2), sw.Stride(1)))
        >>> sw.unsqueeze(x, 0).layout.shape
        Shape<((1,), (2,))>
    """

    return _dispatch_unary("unsqueeze", tensor).forward(tensor, dim)


def squeeze(tensor: Any, dim: Any) -> Any:
    """Remove one explicitly selected extent-one top-level mode.

    Only a top-level leaf with extent one may be removed; nested leaves and an
    implicit ``dim=None`` form are intentionally deferred. Negative dimensions
    are normalized against the input's top-level rank.

    Args:
        tensor: Tensor whose top-level singleton mode should be removed.
        dim: Existing top-level mode index, with ordinary negative-index
            normalization.

    Returns:
        Tensor view sharing the input representation with the selected mode
        removed.

    Examples:
        >>> import strideweave as sw
        >>> x = sw.Tensor(sw.Generic([1.0, 2.0]), 0,
        ...               sw.Layout(sw.Shape([1, 2]), sw.Stride([0, 1])))
        >>> sw.squeeze(x, 0).layout.shape
        Shape<((2,),)>
    """

    return _dispatch_unary("squeeze", tensor).forward(tensor, dim)


def broadcast_in_dim(
    tensor: Any, target: Shape, broadcast_dimensions: Iterable[Any]
) -> Any:
    """Compose explicit singleton insertion with :func:`broadcast_to`.

    This helper follows StrideWeave's hierarchical rule rather than guessing
    rank alignment: ``broadcast_dimensions`` names the target top-level
    positions occupied by the existing source modes, in increasing order. Any
    omitted positions are inserted as extent-one modes before the structural
    broadcast. Nested insertion, reordering, and implicit rank alignment are
    intentionally unsupported; callers can use ``rearrange`` explicitly when
    they need a different hierarchy.

    Args:
        tensor: Tensor whose modes are to be placed into ``target``.
        target: Target hierarchical shape accepted by ``broadcast_to`` after
            singleton insertion.
        broadcast_dimensions: Strictly increasing target positions, one per
            source top-level mode.

    Returns:
        Differentiable zero-copy view with shape ``target``.

    Examples:
        >>> import strideweave as sw
        >>> x = sw.Tensor(sw.Generic([1.0, 2.0]), 0,
        ...               sw.Layout(sw.Shape(2), sw.Stride(1)))
        >>> y = sw.broadcast_in_dim(x, sw.Shape([3, 2]), (1,))
        >>> y.layout.stride
        Stride<(0, 1)>
    """

    tensor = _as_tensor(tensor, "tensor")
    if not isinstance(target, Shape):
        raise TypeError("target must be a Shape")
    try:
        dimensions = tuple(operator_index(value) for value in broadcast_dimensions)
    except TypeError as exc:
        raise TypeError("broadcast_dimensions must contain integers") from exc

    source_rank = len(tensor.layout)
    target_rank = len(target)
    if len(dimensions) != source_rank:
        raise ValueError(
            "broadcast_dimensions must contain one position per source mode"
        )
    if any(dim < 0 or dim >= target_rank for dim in dimensions):
        raise ValueError("broadcast_dimensions contains an out-of-range position")
    if tuple(sorted(set(dimensions))) != dimensions:
        raise ValueError("broadcast_dimensions must be strictly increasing")

    result = tensor
    occupied = set(dimensions)
    # Insert from left to right so each position is measured in the growing
    # target rank. Inserting from right to left would shift an earlier omitted
    # position past the source mode it was meant to precede.
    for dim in range(target_rank):
        if dim not in occupied:
            result = unsqueeze(result, dim)
    return broadcast_to(result, target)


def sub(lhs: Any, rhs: Any) -> Any:
    """Subtract two structurally broadcast-compatible tensors.

    Subtraction dispatches to the carrier's ``sub`` operation (implemented
    natively for CPU carriers and in Python for Generic carriers). Its autograd
    backward passes the incoming gradient to ``lhs`` and its negation to
    ``rhs``. Shape trees must share a profile, and only an extent of one
    expands to its peer; no rank alignment is inferred.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand on the same backing carrier and with a
            structurally broadcast-compatible shape.

    Returns:
        Tensor containing the elementwise difference.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([5, 7]), 0, layout)
        >>> y = Tensor(Generic([3, 4]), 0, layout)
        >>> sw.sub(x, y)[1]
        3
    """

    return _dispatch_binary("sub", lhs, rhs).forward(lhs, rhs)


def neg(tensor: Any) -> Any:
    """Negate a tensor elementwise.

    The negation is composed from existing primitives as ``mul(tensor, -1)``,
    so it dispatches through the same carrier operations and participates in
    autograd like scalar multiplication.

    Args:
        tensor: Tensor operand to negate.

    Returns:
        Tensor containing elementwise negated values.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, -3]), 0, Layout(Shape(2), Stride(1)))
        >>> sw.neg(x)[1]
        3
    """

    tensor = _as_tensor(tensor, "tensor")
    return tensor.carrier.dispatch_op("neg").forward(tensor)


def elementwise_mul(lhs: Any, rhs: Any) -> Any:
    """Multiply two structurally broadcast-compatible tensors elementwise.

    Shape trees must share a profile. Equal leaf extents match, and an extent
    of one expands to its peer using stride zero without rank alignment.

    Args:
        lhs: Left tensor operand.
        rhs: Right tensor operand on the same backing carrier and with a
            structurally broadcast-compatible shape.

    Returns:
        Tensor containing elementwise products.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([2, 3]), 0, layout)
        >>> y = Tensor(Generic([4, 5]), 0, layout)
        >>> sw.elementwise_mul(x, y)[1]
        15
    """

    return _dispatch_binary("elementwise_mul", lhs, rhs).forward(lhs, rhs)


def mul(lhs: Any, rhs: Any) -> Any:
    """Multiply tensors or a tensor by a weak scalar.

    Tensor/tensor calls use the carrier's ``mul`` binary dispatch. If one
    operand is a weak scalar, the tensor operand owns dispatch and the scalar
    is forwarded in its original direction (scalar-first calls are accepted
    for ergonomic parity with tensor arithmetic).

    Args:
        lhs: Tensor or numerical scalar operand.
        rhs: Tensor or numerical scalar operand.

    Returns:
        Tensor containing scaled or elementwise multiplied values.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> sw.mul(x, 10)[1]
        30
    """

    from ..core.tensor import Tensor

    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _dispatch_binary("mul", lhs, rhs).forward(lhs, rhs)
    if isinstance(lhs, Tensor):
        return lhs.carrier.dispatch_op("mul").forward(lhs, rhs)
    if isinstance(rhs, Tensor):
        return rhs.carrier.dispatch_op("mul").forward(rhs, lhs)
    raise TypeError("mul requires at least one Tensor operand")


def div(lhs: Any, rhs: Any) -> Any:
    """Divide two structurally broadcast-compatible tensors elementwise.

    Shape trees must share a profile. Equal leaf extents match, and an extent
    of one expands to its peer using stride zero without rank alignment.

    Args:
        lhs: Numerator tensor.
        rhs: Denominator tensor on the same backing carrier and with a
            structurally broadcast-compatible shape.

    Returns:
        Tensor containing elementwise quotients.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> layout = Layout(Shape(2), Stride(1))
        >>> x = Tensor(Generic([8, 9]), 0, layout)
        >>> y = Tensor(Generic([2, 3]), 0, layout)
        >>> sw.div(x, y)[1]
        3.0
    """

    return _dispatch_binary("div", lhs, rhs).forward(lhs, rhs)


def maximum(lhs: Any, rhs: Any) -> Any:
    """Select the elementwise maximum of two Float32 tensors.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Tensor containing the elementwise maxima.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.maximum(F.tensor([1.0]), F.tensor([2.0]))[0]
        2.0
    """

    return _dispatch_binary("maximum", lhs, rhs).forward(lhs, rhs)


def minimum(lhs: Any, rhs: Any) -> Any:
    """Select the elementwise minimum of two Float32 tensors.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Tensor containing the elementwise minima.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.minimum(F.tensor([1.0]), F.tensor([2.0]))[0]
        1.0
    """

    return _dispatch_binary("minimum", lhs, rhs).forward(lhs, rhs)


def rem(lhs: Any, rhs: Any) -> Any:
    """Compute the elementwise truncating remainder of two Float32 tensors.

    Args:
        lhs: Float32 dividend tensor.
        rhs: Structurally broadcast-compatible Float32 divisor tensor.

    Returns:
        Tensor containing ``fmod(lhs, rhs)`` values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.rem(F.tensor([5.0]), F.tensor([2.0]))[0]
        1.0
    """

    return _dispatch_binary("rem", lhs, rhs).forward(lhs, rhs)


def eq(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors for equality, returning Bool storage.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs == rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.eq(F.tensor([1.0]), F.tensor([1.0]))[0]
        True
    """

    return _dispatch_binary("eq", lhs, rhs).forward(lhs, rhs)


def ne(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors for inequality, returning Bool storage.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs != rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.ne(F.tensor([1.0]), F.tensor([2.0]))[0]
        True
    """

    return _dispatch_binary("ne", lhs, rhs).forward(lhs, rhs)


def lt(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors elementwise with ``<``.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs < rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.lt(F.tensor([1.0]), F.tensor([2.0]))[0]
        True
    """

    return _dispatch_binary("lt", lhs, rhs).forward(lhs, rhs)


def le(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors elementwise with ``<=``.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs <= rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.le(F.tensor([1.0]), F.tensor([1.0]))[0]
        True
    """

    return _dispatch_binary("le", lhs, rhs).forward(lhs, rhs)


def gt(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors elementwise with ``>``.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs > rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.gt(F.tensor([2.0]), F.tensor([1.0]))[0]
        True
    """

    return lt(rhs, lhs)


def ge(lhs: Any, rhs: Any) -> Any:
    """Compare two Float32 tensors elementwise with ``>=``.

    Args:
        lhs: Left Float32 tensor.
        rhs: Right structurally broadcast-compatible Float32 tensor.

    Returns:
        Non-differentiable Bool tensor indicating ``lhs >= rhs``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.ge(F.tensor([1.0]), F.tensor([1.0]))[0]
        True
    """

    return le(rhs, lhs)


def logical_not(tensor: Any) -> Any:
    """Return a Bool tensor indicating which Float32 values are zero.

    Args:
        tensor: Float32 tensor to test for logical falsity.

    Returns:
        Non-differentiable Bool tensor with ``True`` where ``tensor == 0``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.logical_not(F.tensor([0.0]))[0]
        True
    """

    return _dispatch_unary("logical_not", tensor).forward(_as_tensor(tensor, "tensor"))


def exp(tensor: Any) -> Any:
    """Apply the exponential function elementwise.

    Args:
        tensor: Tensor whose logical values should be exponentiated.

    Returns:
        Tensor containing ``math.exp`` applied to each element.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.exp(x)[0]
        1.0
    """

    return _dispatch_unary("exp", tensor).forward(tensor)


def abs(tensor: Any) -> Any:  # noqa: A001 - mirrors the tensor API
    """Apply elementwise absolute value to a Float32 tensor.

    Args:
        tensor: Float32 tensor to transform.

    Returns:
        Tensor containing the absolute value of each element.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.abs(F.tensor([-2.0]))[0]
        2.0
    """

    return _dispatch_unary("abs", tensor).forward(_as_tensor(tensor, "tensor"))


def sign(tensor: Any) -> Any:
    """Return the sign of each element of a Float32 tensor.

    Args:
        tensor: Float32 tensor to inspect.

    Returns:
        Tensor containing ``-1``, ``0``, or ``1`` for each element.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.sign(F.tensor([-2.0, 0.0, 3.0]))[2]
        1.0
    """

    return _dispatch_unary("sign", tensor).forward(_as_tensor(tensor, "tensor"))


def recip(tensor: Any) -> Any:
    """Compute the elementwise reciprocal of a Float32 tensor.

    Args:
        tensor: Float32 tensor whose reciprocal is requested.

    Returns:
        Tensor containing ``1 / tensor`` with IEEE special values preserved.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.recip(F.tensor([2.0]))[0]
        0.5
    """

    return _dispatch_unary("recip", tensor).forward(_as_tensor(tensor, "tensor"))


def sqrt(tensor: Any) -> Any:
    """Compute elementwise square roots for a Float32 tensor.

    Args:
        tensor: Float32 tensor whose square roots are requested.

    Returns:
        Tensor containing square roots of the input values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.sqrt(F.tensor([4.0]))[0]
        2.0
    """

    return _dispatch_unary("sqrt", tensor).forward(_as_tensor(tensor, "tensor"))


def rsqrt(tensor: Any) -> Any:
    """Compute elementwise reciprocal square roots for a Float32 tensor.

    Args:
        tensor: Float32 tensor whose reciprocal square roots are requested.

    Returns:
        Tensor containing ``1 / sqrt(tensor)``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.rsqrt(F.tensor([4.0]))[0]
        0.5
    """

    return _dispatch_unary("rsqrt", tensor).forward(_as_tensor(tensor, "tensor"))


def exp2(tensor: Any) -> Any:
    """Compute elementwise powers of two for a Float32 tensor.

    Args:
        tensor: Float32 tensor providing exponents.

    Returns:
        Tensor containing ``2 ** tensor``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.exp2(F.tensor([3.0]))[0]
        8.0
    """

    return _dispatch_unary("exp2", tensor).forward(_as_tensor(tensor, "tensor"))


def log(tensor: Any) -> Any:
    """Compute natural logarithms elementwise for a Float32 tensor.

    Args:
        tensor: Float32 tensor with positive values.

    Returns:
        Tensor containing natural logarithms.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> f"{sw.log(F.tensor([1.0]))[0]:.6f}"
        '0.000000'
    """

    return _dispatch_unary("log", tensor).forward(_as_tensor(tensor, "tensor"))


def log2(tensor: Any) -> Any:
    """Compute base-two logarithms elementwise for a Float32 tensor.

    Args:
        tensor: Float32 tensor with positive values.

    Returns:
        Tensor containing base-two logarithms.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.log2(F.tensor([8.0]))[0]
        3.0
    """

    return _dispatch_unary("log2", tensor).forward(_as_tensor(tensor, "tensor"))


def sin(tensor: Any) -> Any:
    """Compute elementwise sine for a Float32 tensor.

    Args:
        tensor: Float32 tensor containing angles in radians.

    Returns:
        Tensor containing sine values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> f"{sw.sin(F.tensor([0.0]))[0]:.6f}"
        '0.000000'
    """

    return _dispatch_unary("sin", tensor).forward(_as_tensor(tensor, "tensor"))


def cos(tensor: Any) -> Any:
    """Compute elementwise cosine for a Float32 tensor.

    Args:
        tensor: Float32 tensor containing angles in radians.

    Returns:
        Tensor containing cosine values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.cos(F.tensor([0.0]))[0]
        1.0
    """

    return _dispatch_unary("cos", tensor).forward(_as_tensor(tensor, "tensor"))


def erf(tensor: Any) -> Any:
    """Compute the elementwise Gauss error function for a Float32 tensor.

    Args:
        tensor: Float32 tensor to transform.

    Returns:
        Tensor containing ``erf(tensor)``.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.erf(F.tensor([0.0]))[0]
        0.0
    """

    return _dispatch_unary("erf", tensor).forward(_as_tensor(tensor, "tensor"))


def floor(tensor: Any) -> Any:
    """Round each Float32 tensor element toward negative infinity.

    Args:
        tensor: Float32 tensor to round.

    Returns:
        Tensor containing floor-rounded values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.floor(F.tensor([1.8]))[0]
        1.0
    """

    return _dispatch_unary("floor", tensor).forward(_as_tensor(tensor, "tensor"))


def ceil(tensor: Any) -> Any:
    """Round each Float32 tensor element toward positive infinity.

    Args:
        tensor: Float32 tensor to round.

    Returns:
        Tensor containing ceil-rounded values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.ceil(F.tensor([1.2]))[0]
        2.0
    """

    return _dispatch_unary("ceil", tensor).forward(_as_tensor(tensor, "tensor"))


def round(tensor: Any) -> Any:  # noqa: A001 - mirrors the tensor API
    """Round Float32 tensor elements using ties-to-even semantics.

    Args:
        tensor: Float32 tensor to round.

    Returns:
        Tensor containing rounded values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> sw.round(F.tensor([1.5]))[0]
        2.0
    """

    return _dispatch_unary("round", tensor).forward(_as_tensor(tensor, "tensor"))


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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([-1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> sw.relu(x)[1]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.sigmoid(x)[0]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.tanh(x)[0]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.gelu(x)[0]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.silu(x)[0]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> f"{sw.softplus(x)[0]:.6f}"
        '0.693147'
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([0]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.elu(x)[0]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([-2]), 0, Layout(Shape(1), Stride(1)))
        >>> sw.leaky_relu(x)[0]
        -0.02
    """

    return _dispatch_unary("leaky_relu", tensor).forward(tensor)


def pow(base: Any, exponent: Any) -> Any:
    """Raise a tensor or scalar base to a tensor or weak scalar exponent.

    Args:
        base: Tensor or weak scalar base.
        exponent: Tensor or weak scalar exponent.

    Returns:
        Tensor containing each element raised to ``exponent``.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([2, 3]), 0, Layout(Shape(2), Stride(1)))
        >>> sw.pow(x, 3)[1]
        27
    """

    from ..core.tensor import Tensor

    if isinstance(base, Tensor) and isinstance(exponent, Tensor):
        return _dispatch_binary("pow", base, exponent).forward(base, exponent)
    if isinstance(base, Tensor):
        return base.carrier.dispatch_op("pow").forward(base, exponent)
    if isinstance(exponent, Tensor):
        return exponent.carrier.dispatch_op("pow").forward(base, exponent)
    raise TypeError("pow requires at least one Tensor operand")


def reduce_sum(
    tensor: Any,
    description: str,
    *,
    accumulator_dtype: SimpleDType | None = None,
) -> Any:
    """Sum-reduce dimensions omitted by a StrideWeave description.

    A stride-zero reduced mode contributes once per logical coordinate, so
    reducing an extent-N broadcast mode scales its stored value by N; backward
    sums through the differentiable broadcast view. The accumulator dtype
    selects precision, not traversal or association; floating reduction order
    is backend-defined.

    Args:
        tensor: Tensor to reduce.
        description: Command such as ``"a b -> a"`` naming kept dimensions.
        accumulator_dtype: Floating accumulator dtype. ``None`` selects the
            backend's default ``Float32`` accumulator. ``DType.Float64``
            requests widened accumulation without changing the output dtype.

    Returns:
        Tensor containing sums over omitted dimensions.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[1.0, 2.0]])
        >>> sw.reduce_sum(x, "a b -> a")[0]
        3.0
    """

    return _reduce_description(
        "reduce_sum", tensor, description, accumulator_dtype=accumulator_dtype
    )


def reduce_prod(tensor: Any, description: str) -> Any:
    """Product-reduce dimensions omitted by a StrideWeave description.

    Args:
        tensor: Float32 tensor to reduce.
        description: Command naming dimensions to retain.

    Returns:
        Tensor containing sequential Float32 products over omitted dimensions.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[2.0, 3.0]])
        >>> sw.reduce_prod(x, "a b -> a")[0]
        6.0
    """

    return _reduce_description("reduce_prod", tensor, description)


def reduce_max(tensor: Any, description: str) -> Any:
    """Maximum-reduce dimensions omitted by a StrideWeave description.

    Args:
        tensor: Float32 tensor to reduce.
        description: Command naming dimensions to retain.

    Returns:
        Tensor containing maximum values over omitted dimensions.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[2.0, 3.0]])
        >>> sw.reduce_max(x, "a b -> a")[0]
        3.0
    """

    return _reduce_description("reduce_max", tensor, description)


def reduce_min(tensor: Any, description: str) -> Any:
    """Minimum-reduce dimensions omitted by a StrideWeave description.

    Args:
        tensor: Float32 tensor to reduce.
        description: Command naming dimensions to retain.

    Returns:
        Tensor containing minimum values over omitted dimensions.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[2.0, 3.0]])
        >>> sw.reduce_min(x, "a b -> a")[0]
        2.0
    """

    return _reduce_description("reduce_min", tensor, description)


def argmax(tensor: Any, description: str) -> Any:
    """Return first-winning indices after a described maximum reduction.

    Args:
        tensor: Float32 tensor to reduce.
        description: Command naming dimensions to retain.

    Returns:
        Int32 tensor containing source ordinals of maximum values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[2.0, 3.0]])
        >>> arg = sw.argmax(x, "a b -> a")
        >>> arg[0]
        1
    """

    return _reduce_description("argmax", tensor, description)


def argmin(tensor: Any, description: str) -> Any:
    """Return first-winning indices after a described minimum reduction.

    Args:
        tensor: Float32 tensor to reduce.
        description: Command naming dimensions to retain.

    Returns:
        Int32 tensor containing source ordinals of minimum values.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[2.0, 3.0]])
        >>> arg = sw.argmin(x, "a b -> a")
        >>> arg[0]
        0
    """

    return _reduce_description("argmin", tensor, description)


def cumsum(tensor: Any, dimension: Any) -> Any:
    """Compute an inclusive cumulative sum along one top-level mode.

    Args:
        tensor: Float32 tensor to scan.
        dimension: Explicit top-level mode index; negative indices count from
            the last mode.

    Returns:
        Tensor with the same shape containing inclusive cumulative sums.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[1.0, 2.0]])
        >>> sw.cumsum(x, 0)[0]
        1.0
    """

    tensor = _as_tensor(tensor, "tensor")
    dimension = _normalize_top_level_axis(
        dimension, len(tensor.layout), name="dimension"
    )
    return tensor.carrier.dispatch_op("cumsum").forward(tensor, dimension)


def conv_general(
    lhs: Any,
    kernel: Any,
    strides: Any,
    padding: Any,
    *,
    lhs_dilation: Any = None,
    kernel_dilation: Any = None,
    feature_groups: Any = 1,
    lhs_dims: Any = None,
    kernel_dims: Any = None,
    output_dims: Any = None,
) -> Any:
    """Apply grouped cross-correlation over explicit hierarchical modes.

    Args:
        lhs: Float32 input tensor with batch, feature, and spatial modes.
        kernel: Float32 kernel tensor with output-feature, input-feature, and
            spatial modes.
        strides: Positive per-spatial-mode convolution strides.
        padding: Non-negative ``(low, high)`` pair per spatial mode.
        lhs_dilation: Optional positive input dilation per spatial mode.
        kernel_dilation: Optional positive kernel dilation per spatial mode.
        feature_groups: Positive number of grouped feature contractions.
        lhs_dims: Keyword-only permutation naming lhs batch, feature, spatial
            modes in canonical order.
        kernel_dims: Keyword-only permutation naming kernel role modes in
            canonical order.
        output_dims: Keyword-only permutation for the returned mode order.

    Returns:
        Float32 tensor containing the grouped cross-correlation result.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> x = F.tensor([[[1.0, 2.0]]])
        >>> k = F.tensor([[[2.0]]])
        >>> y = sw.conv_general(x, k, (1,), ((0, 0),))
        >>> y[0, 0, 0]
        2.0
    """

    lhs = _as_tensor(lhs, "lhs")
    kernel = _as_tensor(kernel, "kernel")
    lhs_permutation = _normalize_permutation(lhs_dims, len(lhs.layout), "lhs_dims")
    kernel_permutation = _normalize_permutation(
        kernel_dims, len(kernel.layout), "kernel_dims"
    )
    output_permutation = _normalize_permutation(
        output_dims, len(lhs.layout), "output_dims"
    )
    canonical_lhs = permute(lhs, *lhs_permutation)
    canonical_kernel = permute(kernel, *kernel_permutation)
    operation = _dispatch_binary("conv_general", canonical_lhs, canonical_kernel)
    result = operation.forward(
        canonical_lhs,
        canonical_kernel,
        strides,
        padding,
        lhs_dilation,
        kernel_dilation,
        feature_groups,
    )
    if output_permutation != tuple(range(len(output_permutation))):
        result = permute(result, *output_permutation)
    return result


def _dispatch_ternary(operation_name: str, first: Any, second: Any, third: Any) -> Any:
    first = _as_tensor(first, "first")
    second = _as_tensor(second, "second")
    third = _as_tensor(third, "third")
    carrier_type = type(first.carrier)
    if (
        type(second.carrier) is not carrier_type
        or type(third.carrier) is not carrier_type
    ):
        raise TypeError("Tensor backing carriers must match")
    return first.carrier.dispatch_op(operation_name)


def select(condition: Any, on_true: Any, on_false: Any) -> Any:
    """Select values elementwise from two tensors using a Bool condition.

    All three tensors must be backed by the same exact carrier class and are
    aligned simultaneously by the carrier's structural broadcast rule. At
    each logical coordinate, only the selected value branch is read, so a NaN
    in the unselected branch does not contaminate the result. The Bool
    condition is non-differentiable; backward routes the cotangent to the
    selected value branch and zero to the other before reducing broadcast
    views to their original shapes.

    Args:
        condition: Bool tensor choosing ``on_true`` where true and ``on_false``
            where false.
        on_true: Float32 tensor supplying values for true condition elements.
        on_false: Float32 tensor supplying values for false condition elements.

    Returns:
        Float32 tensor containing the selected values.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape(2), sw.Stride(1))
        >>> condition = sw.Tensor(sw.Generic([True, False], dtype=sw.DType.Bool), 0, layout)
        >>> on_true = sw.Tensor(sw.Generic([1.0, 2.0], dtype=sw.DType.Float32), 0, layout)
        >>> on_false = sw.Tensor(sw.Generic([3.0, 4.0], dtype=sw.DType.Float32), 0, layout)
        >>> [sw.select(condition, on_true, on_false)[i] for i in range(2)]
        [1.0, 4.0]
    """

    condition = _as_tensor(condition, "condition")
    on_true = _as_tensor(on_true, "on_true")
    on_false = _as_tensor(on_false, "on_false")
    carrier_type = type(condition.carrier)
    if (
        type(on_true.carrier) is not carrier_type
        or type(on_false.carrier) is not carrier_type
    ):
        raise TypeError("Tensor backing carriers must match")
    return condition.carrier.dispatch_op("select").forward(condition, on_true, on_false)


def clamp(tensor: Any, lower: Any, upper: Any) -> Any:
    """Clamp a Float32 tensor between tensor or weak-scalar bounds.

    Each bound may be a structurally broadcast-compatible Float32 tensor or a
    real Python scalar. Tensor bounds must use the same exact carrier class as
    ``tensor``. The carrier executes the ordered stages
    ``middle = maximum(tensor, lower)`` and then
    ``result = minimum(middle, upper)`` without a separate bound-order check.
    Backward applies the corresponding minimum VJP followed by the maximum
    VJP, preserving their equal-winner splits and NaN gradients. Weak scalar
    bounds do not receive gradients.

    Args:
        tensor: Float32 tensor whose values are constrained.
        lower: Float32 tensor or weak scalar lower bound.
        upper: Float32 tensor or weak scalar upper bound.

    Returns:
        Float32 tensor with values constrained between ``lower`` and ``upper``.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape(3), sw.Stride(1))
        >>> x = sw.Tensor(sw.Generic([-2.0, 0.5, 3.0], dtype=sw.DType.Float32), 0, layout)
        >>> [sw.clamp(x, 0.0, 1.0)[i] for i in range(3)]
        [0.0, 0.5, 1.0]
    """

    from ..core.tensor import Tensor

    tensor = _as_tensor(tensor, "tensor")
    carrier_type = type(tensor.carrier)
    if isinstance(lower, Tensor) and type(lower.carrier) is not carrier_type:
        raise TypeError("Tensor backing carriers must match")
    if isinstance(upper, Tensor) and type(upper.carrier) is not carrier_type:
        raise TypeError("Tensor backing carriers must match")
    return tensor.carrier.dispatch_op("clamp").forward(tensor, lower, upper)


def gather(tensor: Any, indices: Any, axis: Any) -> Any:
    """Gather values along one top-level mode using Int32 indices.

    Args:
        tensor: Float32 source tensor.
        indices: Int32 tensor of logical indices replacing the selected mode.
        axis: Explicit top-level source mode, allowing negative indexing.

    Returns:
        Float32 tensor whose selected mode is replaced by ``indices`` shape.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> data = F.tensor([[10.0, 20.0]])
        >>> ids = F.tensor([1], dtype=sw.DType.Int32)
        >>> sw.gather(data, ids, 1)[0, 0]
        20.0
    """

    tensor = _as_tensor(tensor, "tensor")
    indices = _as_tensor(indices, "indices")
    axis = _normalize_top_level_axis(axis, len(tensor.layout))
    if type(tensor.carrier) is not type(indices.carrier):
        raise TypeError("Tensor backing carriers must match")
    return tensor.carrier.dispatch_op("gather").forward(tensor, indices, axis)


def scatter(base: Any, indices: Any, updates: Any, axis: Any) -> Any:
    """Functionally overwrite values selected by distinct Int32 indices.

    Args:
        base: Float32 tensor receiving updates.
        indices: Int32 tensor of distinct indices for ``axis``.
        updates: Float32 tensor with the gather-result shape.
        axis: Explicit top-level mode to update, allowing negative indexing.

    Returns:
        New Float32 tensor with selected values overwritten.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> base = F.tensor([1.0, 2.0])
        >>> ids = F.tensor([0], dtype=sw.DType.Int32)
        >>> updates = F.tensor([9.0])
        >>> sw.scatter(base, ids, updates, 0)[0]
        9.0
    """

    base = _as_tensor(base, "base")
    indices = _as_tensor(indices, "indices")
    updates = _as_tensor(updates, "updates")
    axis = _normalize_top_level_axis(axis, len(base.layout))
    carrier_type = type(base.carrier)
    if (
        type(indices.carrier) is not carrier_type
        or type(updates.carrier) is not carrier_type
    ):
        raise TypeError("Tensor backing carriers must match")
    return base.carrier.dispatch_op("scatter").forward(base, indices, updates, axis)


def scatter_add(base: Any, indices: Any, updates: Any, axis: Any) -> Any:
    """Functionally add updates at Int32 indices along one top-level mode.

    Args:
        base: Float32 tensor receiving updates.
        indices: Int32 tensor of logical indices; repeats accumulate in order.
        updates: Float32 tensor with the gather-result shape.
        axis: Explicit top-level mode to update, allowing negative indexing.

    Returns:
        New Float32 tensor with updates added at selected positions.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> base = F.tensor([1.0, 2.0])
        >>> ids = F.tensor([0], dtype=sw.DType.Int32)
        >>> updates = F.tensor([3.0])
        >>> sw.scatter_add(base, ids, updates, 0)[0]
        4.0
    """

    base = _as_tensor(base, "base")
    indices = _as_tensor(indices, "indices")
    updates = _as_tensor(updates, "updates")
    axis = _normalize_top_level_axis(axis, len(base.layout))
    carrier_type = type(base.carrier)
    if (
        type(indices.carrier) is not carrier_type
        or type(updates.carrier) is not carrier_type
    ):
        raise TypeError("Tensor backing carriers must match")
    return base.carrier.dispatch_op("scatter_add").forward(base, indices, updates, axis)


def sort(tensor: Any, axis: Any = -1, descending: Any = False) -> SortResult:
    """Sort values along one top-level mode and return source indices.

    Args:
        tensor: Float32 tensor to sort.
        axis: Top-level mode to sort, defaulting to the last mode.
        descending: Whether to sort largest-first.

    Returns:
        ``(values, indices)`` named tuple; indices are Int32 source ordinals.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> result = sw.sort(F.tensor([2.0, 1.0]))
        >>> result.values[0], result.indices[0]
        (1.0, 1)
    """

    tensor = _as_tensor(tensor, "tensor")
    axis = _normalize_top_level_axis(axis, len(tensor.layout))
    values = tensor.carrier.dispatch_op("_sort_values").forward(
        tensor, axis, descending
    )
    indices = tensor.carrier.dispatch_op("_sort_indices").forward(
        tensor, axis, descending
    )
    return SortResult(values, indices)


def topk(tensor: Any, k: Any, axis: Any = -1, largest: Any = True) -> TopKResult:
    """Select sorted top-k values along one top-level mode.

    Args:
        tensor: Float32 tensor to select from.
        k: Positive number of values to retain along ``axis``.
        axis: Top-level mode to select, defaulting to the last mode.
        largest: Whether to select largest values (otherwise smallest).

    Returns:
        ``(values, indices)`` named tuple; indices are Int32 source ordinals.

    Examples:
        >>> import strideweave as sw
        >>> import strideweave.friendly as F
        >>> result = sw.topk(F.tensor([2.0, 1.0]), 1)
        >>> result.values[0], result.indices[0]
        (2.0, 0)
    """

    tensor = _as_tensor(tensor, "tensor")
    axis = _normalize_top_level_axis(axis, len(tensor.layout))
    values = tensor.carrier.dispatch_op("_topk_values").forward(
        tensor, k, axis, largest
    )
    indices = tensor.carrier.dispatch_op("_topk_indices").forward(
        tensor, k, axis, largest
    )
    return TopKResult(values, indices)


def matmul(lhs: Any, rhs: Any, *, accumulator_dtype: SimpleDType | None = None) -> Any:
    """Multiply two two-mode tensors.

    The first mode of each input is kept, and the second modes must have the
    same logical size and are contracted with a dot product. Stride-zero modes
    are supported: a broadcast kept mode repeats rows or columns, while a
    broadcast contracted mode repeats its stored factor in the dot product.
    Backward computes logical gradients injectively and sums them through the
    broadcast view. The accumulator dtype selects precision, not traversal or
    association; floating contraction order is backend-defined.

    Args:
        lhs: Left two-mode tensor.
        rhs: Right two-mode tensor with matching second-mode logical size.
        accumulator_dtype: Floating accumulator dtype. ``None`` selects the
            backend's default ``Float32`` accumulator. ``DType.Float64``
            requests widened accumulation without changing the output dtype.

    Returns:
        Tensor with layout formed from the first mode of each input.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> sw.matmul(lhs, rhs)[1, 1]
        22
    """

    return _matmul_2mode(lhs, rhs, accumulator_dtype=accumulator_dtype)


def move(tensor: Any, destination: Any) -> Any:
    """Move a tensor's values into another carrier instance.

    The destination carrier object is filled with the tensor's values, the
    source carrier is released (further access raises), and a tensor backed by
    the destination is returned. Move is not owned by any carrier: the
    concrete move operation is dispatched on the (source, destination) carrier
    class pair, with native bulk-copy operations registered for
    CPU/FileBacked pairs and an elementwise fallback for every other pair.
    The destination dtype must match the tensor dtype (for example
    ``DType.Float32`` to ``DType.Float32``). Move participates in
    autograd: gradients flowing into the result are moved back into the
    source carrier during backward, so gradients cross carrier
    boundaries.

    A broadcast tensor keeps its exact stride-zero layout and copies only the
    layout's physical ``cosize`` span; move does not materialize one storage
    element per logical coordinate. Backward accepts an injective same-shape
    cotangent and returns logical values to fresh source-class storage before
    broadcast reduction.

    Logical tensor values are identical regardless of the dispatched
    operation, but storage contents at layout holes may differ: bulk
    operations copy the whole physical span including holes, while the
    elementwise fallback leaves destination hole values untouched. Do not
    rely on the storage contents of hole positions.

    Args:
        tensor: Tensor whose values should be moved.
        destination: Pre-constructed mutable ``Carrier`` instance with the
            tensor's dtype that receives the values, such as ``FileBacked()``
            or ``CPU(6)``.

    Returns:
        Tensor backed by ``destination`` with the input tensor's layout.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import FileBacked, Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2]), 0, Layout(Shape(2), Stride(1)))
        >>> moved = sw.move(x, FileBacked())
        >>> moved[1]
        2.0
    """

    from ..carriers.move.ops import dispatch_move

    tensor = _as_tensor(tensor, "tensor")
    operation_class = dispatch_move(type(tensor.carrier), type(destination))
    return operation_class().forward(tensor, destination)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    """Contract two tensors using a StrideWeave contraction description.

    Shared input symbols omitted from the output are contracted, shared symbols
    retained in the output are batch dimensions, and one-sided symbols are free
    dimensions.

    Syntax:
        ``description`` must be ``"lhs, rhs -> output"``. Every one-sided input
        symbol must appear exactly once in the output. Shared symbols may be
        omitted for contraction or retained for batching. Parentheses preserve
        hierarchical output grouping, and literal ``1`` inserts a singleton.

    Semantics:
        A description without batch symbols uses rearrange, two-mode matmul, and
        final rearrange. A description with batch symbols aligns both operands
        over their union symbol space, multiplies them elementwise, and sums
        only the shared symbols omitted from the output.

    Mode assumptions:
        Each input reference describes the corresponding hierarchical layout.
        Same-named symbols must have equal logical sizes. StrideWeave does not
        infer flat-layout rank alignment or reorder unspecified dimensions.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor containing the requested contraction result.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> sw.einsum(lhs, rhs, "a b, c b -> a c")[1, 1]
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
    output is parsed as a StrideWeave layout command and must not be combined with
    an explicit selection.

    Args:
        tensor: Tensor whose layout should be rearranged.
        output: Output Tree or StrideWeave layout rearrange description.
        selection: Optional Tree selecting source layout subtrees.

    Returns:
        Tensor view with the rearranged layout.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Node, Shape, Stride, Tensor, Tree
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> sw.rearrange(x, "a b -> b a")[2, 1]
        6
        >>> output = Tree(Node.id(1), Node.id(0))
        >>> selection = Tree(Node.Leaf, Node.Leaf)
        >>> sw.rearrange(x, output, selection)[2, 1]
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
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> sw.permute(x, 1, 0)[2, 1]
        6
    """

    return _dispatch_unary("permute", tensor).forward(tensor, *order)


def _view(tensor: Any, key: Any) -> Any:
    """Create the internal tensor view used by slice indexing.

    Integers select and remove a mode where supported; slices preserve a mode
    while adjusting the output layout and offset.

    Args:
        tensor: Tensor to view.
        key: Integer and slice key tuple for top-level modes.

    Returns:
        Tensor view sharing the input backing carrier with an adjusted layout.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> row = x[1, :]
        >>> row[2]
        6
    """

    return _dispatch_unary("view", tensor).forward(tensor, key)
