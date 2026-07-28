"""Concrete simple-dtype numerics for the Generic reference carrier.

``Generic`` is the behavioral reference for the policy specified in
``design/SimpleDType-operation-policy.md``, so its ``Float32`` arithmetic must be
genuine IEEE-754 binary32 rather than Python's binary64 approximated afterwards,
and its ``Int32`` arithmetic must be exact with checked narrowing.

NumPy supplies the binary32 mechanics. It is imported lazily, on the first
concrete ``Float32`` use, so importing StrideWeave — or using only ``CPU``,
``Int32``, or the legacy opaque dtypes — never pays for it. Floating-point error
state is scoped around whole operation loops rather than set up per element:
IEEE singularities are results, not exceptions, so division by zero and
overflow produce ``inf``/``NaN`` and never raise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from numbers import Integral, Real
from typing import Any

from ..dtype import DType

__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "binary32",
    "checked_int32",
    "float32_errstate",
    "float32_scalar",
    "is_concrete_simple_dtype",
    "normalize_storage_value",
    "normalize_storage_values",
    "safe_exp",
]

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

_numpy_module: Any = None


def _numpy() -> Any:
    """Return NumPy, importing it on first concrete ``Float32`` use."""
    global _numpy_module
    if _numpy_module is None:
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Generic Float32 storage and arithmetic require NumPy, which "
                "supplies StrideWeave's reference binary32 semantics"
            ) from exc
        _numpy_module = numpy
    return _numpy_module


@contextmanager
def float32_errstate() -> Iterator[None]:
    """Scope IEEE error state around a whole operation loop.

    Under this context NumPy reports no warnings for the IEEE results the policy
    requires, so a division by zero yields ``inf`` and an overflow yields
    ``inf`` instead of raising. Entering it once per operation, rather than once
    per element, keeps the reference implementation's cost proportional to the
    number of operations.
    """
    numpy = _numpy()
    with numpy.errstate(all="ignore"):
        yield


def binary32(value: Any) -> float:
    """Round ``value`` to IEEE-754 binary32, returned as a Python float.

    Rounding is round-to-nearest-even, and an out-of-range magnitude becomes an
    infinity rather than raising, so the result is exactly the binary32 value a
    native ``float`` kernel would hold.

    Args:
        value: Any real number to represent in binary32.

    Returns:
        The binary32 value, widened losslessly back to a Python float.

    Examples:
        >>> from strideweave.carriers.generic.numerics import binary32
        >>> binary32(0.1) == 0.1
        False
    """
    return float(_numpy().float32(value))


def float32_scalar(value: Any) -> Any:
    """Return ``value`` as a binary32 scalar that computes in binary32.

    Arithmetic between the returned scalars rounds to binary32 at every step and
    produces IEEE results — infinities and NaNs — rather than raising, which is
    what makes this the reference for ``Float32``. Use :func:`binary32` to widen
    a result back to a stored Python float.

    Args:
        value: Any real number to compute with in binary32.

    Returns:
        A binary32 scalar.

    Examples:
        >>> from strideweave.carriers.generic.numerics import float32_scalar
        >>> float(float32_scalar(1.0) / float32_scalar(0.0))
        inf
    """
    return _numpy().float32(value)


def checked_int32(value: Any, *, message: str = "Generic Int32 result") -> int:
    """Narrow an exact integer result to ``Int32``, raising on overflow.

    Args:
        value: The exact integer result to narrow.
        message: Subject of the overflow message.

    Returns:
        ``value`` unchanged, once it is known to fit ``Int32``.

    Raises:
        OverflowError: If ``value`` lies outside ``Int32`` range.

    Examples:
        >>> from strideweave.carriers.generic.numerics import checked_int32
        >>> checked_int32(7)
        7
    """
    if not INT32_MIN <= value <= INT32_MAX:
        raise OverflowError(f"{message} is out of int32 range")
    return int(value)


def safe_exp(value: Any) -> float:
    """Exponentiate, returning ``inf`` on overflow instead of raising.

    IEEE-754 defines an overflowing exponential as ``+inf``, but Python's
    ``math.exp`` raises ``OverflowError``. The policy requires the IEEE result,
    so this is the exponential every Generic path uses.

    Args:
        value: The exponent.

    Returns:
        ``exp(value)``, or ``inf`` where the true result overflows.

    Examples:
        >>> from strideweave.carriers.generic.numerics import safe_exp
        >>> safe_exp(10_000.0)
        inf
    """
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def is_concrete_simple_dtype(dtype: DType) -> bool:
    """Report whether ``dtype`` is one of the concrete simple storage dtypes.

    The legacy opaque categories ``DType.Any`` and ``DType.Floating`` are not
    concrete, so they stay on Generic's legacy arithmetic path.

    Args:
        dtype: The dtype to classify.

    Returns:
        ``True`` for ``DType.Float32`` and ``DType.Int32``.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.generic.numerics import (
        ...     is_concrete_simple_dtype,
        ... )
        >>> is_concrete_simple_dtype(DType.Floating)
        False
    """
    return dtype is DType.Float32 or dtype is DType.Int32


def _normalize_float32(value: Any, name: str) -> float:
    # NumPy converts several non-numbers (None among them) to NaN rather than
    # raising, so the type is checked here instead of relying on the cast.
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number for Float32 storage")
    try:
        return binary32(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number for Float32 storage") from exc


def _normalize_int32(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer for Int32 storage")
    return checked_int32(value, message=name)


_NORMALIZERS: dict[DType, Callable[[Any, str], Any]] = {
    DType.Float32: _normalize_float32,
    DType.Int32: _normalize_int32,
}


def normalize_storage_value(dtype: DType, value: Any, name: str = "value") -> Any:
    """Normalize one value into a concrete simple dtype's stored representation.

    A ``Float32`` carrier stores binary32-exact Python floats and an ``Int32``
    carrier stores in-range Python integers, so every stored value is already
    the value the encoding can hold. Storage on the legacy opaque dtypes is
    returned unchanged.

    Args:
        dtype: The carrier's storage dtype.
        value: The value being stored.
        name: Subject used in error messages.

    Returns:
        The normalized value to store.

    Raises:
        TypeError: If ``value`` cannot be represented in ``dtype``.
        OverflowError: If an integer value is outside ``Int32`` range.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.generic.numerics import (
        ...     normalize_storage_value,
        ... )
        >>> normalize_storage_value(DType.Int32, 5)
        5
    """
    normalizer = _NORMALIZERS.get(dtype)
    if normalizer is None:
        return value
    if dtype is DType.Float32:
        # One error-state scope for this conversion, so an out-of-range
        # magnitude becomes an infinity without emitting a warning.
        with float32_errstate():
            return normalizer(value, name)
    return normalizer(value, name)


def normalize_storage_values(dtype: DType, values: list[Any], name: str) -> list[Any]:
    """Normalize a whole sequence into a dtype's stored representation.

    The error-state scope is entered once for the sequence rather than once per
    value, so building storage costs one setup regardless of its length.

    Args:
        dtype: The carrier's storage dtype.
        values: The values being stored.
        name: Subject used in error messages.

    Returns:
        A new list of normalized values.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.generic.numerics import (
        ...     normalize_storage_values,
        ... )
        >>> normalize_storage_values(DType.Int32, [1, 2], "value")
        [1, 2]
    """
    normalizer = _NORMALIZERS.get(dtype)
    if normalizer is None:
        return list(values)
    scope = float32_errstate() if dtype is DType.Float32 else nullcontext()
    with scope:
        return [normalizer(value, name) for value in values]
