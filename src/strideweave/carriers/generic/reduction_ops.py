"""Generic reduction and scan operations.

The public carrier dispatch table is wired by the integration task.  Keeping the
implementations here makes the reduction family share one coordinate/fiber model
without coupling it to the existing elementwise operation module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from ...layout import Shape
from ..dtype import DType
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _mode_logical_size,
    _mode_shape,
    _require_layout,
    _require_live_tensor,
    _require_two_mode_tensor,
    _tensor_with_layout_like,
)
from ..operation_policy import OperationExecutionOptions
from .execution import executing, extrema_total, gradient_arithmetic, unary_arithmetic

__all__ = [
    "GenericArgMaxOperation",
    "GenericArgMinOperation",
    "GenericCumsumOperation",
    "GenericReduceMaxOperation",
    "GenericReduceMinOperation",
    "GenericReduceProdOperation",
    "GenericReduceSumOperation",
]


def _require_nonempty_fiber(size: int) -> None:
    if size <= 0:
        raise ValueError("Reduction fibers must be nonempty")


def _sum_values(arithmetic: Any, values: Iterable[Any]) -> Any:
    """Sequentially add values in the arithmetic's representation."""
    return arithmetic.total(values)


def _product_values(arithmetic: Any, values: Iterable[Any]) -> Any:
    """Sequentially multiply values in the arithmetic's representation."""
    result = arithmetic.convert(1.0)
    for value in values:
        # Store every intermediate so planned Float32 arithmetic rounds each
        # multiplication in binary32; legacy Generic storage remains unchanged.
        result = arithmetic.store(result * value)
    return result


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _select_arg(values: list[Any], *, maximum: bool) -> int:
    """Return the first winning ordinal, with NaN winning numeric values."""
    if not values:
        raise ValueError("Reduction fibers must be nonempty")
    winner = 0
    winner_value = values[0]
    winner_is_nan = _is_nan(winner_value)
    for index, candidate in enumerate(values[1:], 1):
        candidate_is_nan = _is_nan(candidate)
        if winner_is_nan:
            continue
        if candidate_is_nan:
            winner = index
            winner_value = candidate
            winner_is_nan = True
            continue
        if (maximum and candidate > winner_value) or (
            not maximum and candidate < winner_value
        ):
            winner = index
            winner_value = candidate
    return winner


def _reduce_forward(
    owner: Any,
    tensor: Any,
    operation: str,
    combine: Callable[[Any, list[Any]], Any],
    *,
    output_dtype: DType | None = None,
    options: OperationExecutionOptions | None = None,
) -> Any:
    tensor = _require_two_mode_tensor(tensor, "tensor")
    n_size = _mode_logical_size(tensor.layout, 0)
    m_size = _mode_logical_size(tensor.layout, 1)
    _require_nonempty_fiber(m_size)
    output_layout = _canonical_layout_for_shape(Shape(_mode_shape(tensor.layout, 0)))
    owner.ctx["output_layout"] = output_layout
    arithmetic = unary_arithmetic(operation, tensor, None, options=options)
    with executing(arithmetic):
        values = []
        for i in range(n_size):
            fiber = [arithmetic.convert(tensor[i, j]) for j in range(m_size)]
            result = combine(arithmetic, fiber)
            values.append(arithmetic.store(result))
    dtype = arithmetic.result_dtype if output_dtype is None else output_dtype
    return _tensor_with_layout_like(tensor, output_layout, values, dtype)


class GenericReduceSumOperation(Operation):
    """Generic two-mode sum reduction over the second mode."""

    def _forward(self, tensor: Any) -> Any:
        return _reduce_forward(
            self,
            tensor,
            "reduce_sum",
            lambda arithmetic, values: _sum_values(arithmetic, values),
            options=self._execution_options,
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        arithmetic = gradient_arithmetic(tensor)
        with executing(arithmetic):
            values = [
                arithmetic.store(arithmetic.convert(gradient[i]))
                for _j in range(m_size)
                for i in range(n_size)
            ]
        return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)


class GenericReduceProdOperation(Operation):
    """Generic two-mode product reduction over the second mode."""

    def _forward(self, tensor: Any) -> Any:
        return _reduce_forward(
            self,
            tensor,
            "reduce_prod",
            lambda arithmetic, values: _product_values(arithmetic, values),
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        arithmetic = gradient_arithmetic(tensor)
        with executing(arithmetic):
            values: list[Any] = []
            for j in range(m_size):
                for i in range(n_size):
                    terms = [
                        arithmetic.convert(tensor[i, k])
                        for k in range(m_size)
                        if k != j
                    ]
                    product_without_self = _product_values(arithmetic, terms)
                    values.append(
                        arithmetic.store(
                            arithmetic.convert(gradient[i]) * product_without_self
                        )
                    )
        return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)


class _GenericExtremeOperation(Operation):
    """Shared Generic implementation for max/min reduction and VJP."""

    operation: str
    maximum: bool

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_two_mode_tensor(tensor, "tensor")
        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        _require_nonempty_fiber(m_size)
        output_layout = _canonical_layout_for_shape(
            Shape(_mode_shape(tensor.layout, 0))
        )
        self.ctx["output_layout"] = output_layout
        arithmetic = unary_arithmetic(self.operation, tensor, None)
        winners: list[int] = []
        nan_results: list[bool] = []
        with executing(arithmetic):
            values = []
            for i in range(n_size):
                fiber = [arithmetic.convert(tensor[i, j]) for j in range(m_size)]
                # The resolved plan owns extrema ordering, NaN payload, and
                # signed-zero semantics.  Keep VJP metadata derived from that
                # exact result instead of maintaining a second reduction here.
                result = (
                    arithmetic.total(fiber)
                    if arithmetic.is_planned
                    else extrema_total(fiber, maximum=self.maximum)
                )
                result_is_nan = _is_nan(result)
                count = 0 if result_is_nan else sum(value == result for value in fiber)
                values.append(arithmetic.store(result))
                winners.append(count)
                nan_results.append(result_is_nan)
        self.ctx["winner_counts"] = winners
        self.ctx["nan_results"] = nan_results
        self.ctx["results"] = values
        return _tensor_with_layout_like(
            tensor, output_layout, values, arithmetic.result_dtype
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        arithmetic = gradient_arithmetic(tensor)
        counts = self.ctx["winner_counts"]
        nan_results = self.ctx["nan_results"]
        with executing(arithmetic):
            values: list[Any] = []
            for j in range(m_size):
                for i in range(n_size):
                    if nan_results[i]:
                        value = float("nan")
                    else:
                        result = self.ctx["results"][i]
                        is_winner = arithmetic.convert(tensor[i, j]) == result
                        value = (
                            arithmetic.convert(gradient[i]) / counts[i]
                            if is_winner
                            else arithmetic.convert(0.0)
                        )
                    values.append(arithmetic.store(value))
        return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)


class GenericReduceMaxOperation(_GenericExtremeOperation):
    """Generic two-mode maximum reduction with equal-winner VJP."""

    operation = "reduce_max"
    maximum = True


class GenericReduceMinOperation(_GenericExtremeOperation):
    """Generic two-mode minimum reduction with equal-winner VJP."""

    operation = "reduce_min"
    maximum = False


class _GenericArgOperation(Operation):
    """Shared Generic implementation for first-winning arg reductions."""

    operation: str
    maximum: bool

    def _forward(self, tensor: Any) -> Any:
        tensor = _require_two_mode_tensor(tensor, "tensor")
        n_size = _mode_logical_size(tensor.layout, 0)
        m_size = _mode_logical_size(tensor.layout, 1)
        _require_nonempty_fiber(m_size)
        if m_size > 2**31 - 1:
            raise OverflowError("Reduction fiber extent is out of Int32 range")
        output_layout = _canonical_layout_for_shape(
            Shape(_mode_shape(tensor.layout, 0))
        )
        self.ctx["output_layout"] = output_layout
        arithmetic = unary_arithmetic(self.operation, tensor, None)
        with executing(arithmetic):
            values = []
            for i in range(n_size):
                fiber = [arithmetic.convert(tensor[i, j]) for j in range(m_size)]
                values.append(_select_arg(fiber, maximum=self.maximum))
        return _tensor_with_layout_like(tensor, output_layout, values, DType.Int32)

    def backward(self, gradient: Any) -> tuple[Any]:
        raise RuntimeError("arg reductions are not differentiable")


class GenericArgMaxOperation(_GenericArgOperation):
    """Generic first-winning argmax reduction returning Int32 ordinals."""

    operation = "argmax"
    maximum = True


class GenericArgMinOperation(_GenericArgOperation):
    """Generic first-winning argmin reduction returning Int32 ordinals."""

    operation = "argmin"
    maximum = False


def _normalize_axis(axis: Any, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("cumsum dimension must be an integer top-level mode")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("cumsum dimension is out of range")
    return axis


def _decode_modes(flat: int, extents: list[int]) -> list[int]:
    coordinates: list[int] = []
    for extent in extents:
        coordinates.append(flat % extent)
        flat //= extent
    return coordinates


def _fiber_keys(extents: list[int], axis: int) -> Iterable[list[tuple[int, ...]]]:
    other_extents = [extent for index, extent in enumerate(extents) if index != axis]
    total = math.prod(other_extents)
    for fixed in range(total):
        fixed_coordinates = _decode_modes(fixed, other_extents)
        keys: list[tuple[int, ...]] = []
        cursor = 0
        for position in range(extents[axis]):
            coordinates: list[int] = []
            for index in range(len(extents)):
                if index == axis:
                    coordinates.append(position)
                else:
                    coordinates.append(fixed_coordinates[cursor])
                    cursor += 1
            keys.append(tuple(coordinates))
            cursor = 0
        yield keys


def _logical_index(key: tuple[int, ...], extents: list[int]) -> int:
    multiplier = 1
    result = 0
    for coordinate, extent in zip(key, extents, strict=True):
        result += coordinate * multiplier
        multiplier *= extent
    return result


class GenericCumsumOperation(Operation):
    """Generic inclusive cumsum over one explicitly selected top-level mode."""

    def _forward(self, tensor: Any, dimension: Any) -> Any:
        tensor = _require_live_tensor(tensor, "tensor")
        axis = _normalize_axis(dimension, len(tensor.layout))
        extents = [
            _mode_logical_size(tensor.layout, index)
            for index in range(len(tensor.layout))
        ]
        if not extents or any(extent <= 0 for extent in extents):
            raise ValueError("cumsum requires nonempty tensor modes")
        output_layout = _canonical_layout_for_shape(tensor.layout.shape)
        self.ctx["output_layout"] = output_layout
        self.ctx["dimension"] = axis
        arithmetic = unary_arithmetic("cumsum", tensor, None)
        values: list[Any] = [None] * tensor.size()
        with executing(arithmetic):
            for keys in _fiber_keys(extents, axis):
                running = arithmetic.convert(0.0)
                for key in keys:
                    running = arithmetic.store(
                        running + arithmetic.convert(tensor[key])
                    )
                    values[_logical_index(key, extents)] = arithmetic.store(running)
        return _tensor_with_layout_like(
            tensor, output_layout, values, arithmetic.result_dtype
        )

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_layout(gradient, self.ctx["output_layout"])
        axis = self.ctx["dimension"]
        extents = [
            _mode_logical_size(tensor.layout, index)
            for index in range(len(tensor.layout))
        ]
        arithmetic = gradient_arithmetic(tensor)
        values: list[Any] = [None] * tensor.size()
        with executing(arithmetic):
            for keys in _fiber_keys(extents, axis):
                running = arithmetic.convert(0.0)
                for key in reversed(keys):
                    running = arithmetic.store(
                        running + arithmetic.convert(gradient[key])
                    )
                    values[_logical_index(key, extents)] = arithmetic.store(running)
        return (_detached_tensor_like(tensor, values, arithmetic.result_dtype),)
