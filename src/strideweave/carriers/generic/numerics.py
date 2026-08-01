"""Concrete simple-dtype numerics for the Generic reference carrier.

``Generic`` is the behavioral reference for the policy specified in
``design/SimpleDType-operation-policy.md``, so its ``Float32`` arithmetic must be
genuine IEEE-754 binary32 rather than Python's binary64 approximated afterwards,
its ``Int32`` arithmetic must be exact with checked narrowing, and its ``Bool``
storage must remain a logical (non-numeric) representation.

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
    "safe_abs",
    "safe_ceil",
    "safe_cos",
    "safe_divide",
    "safe_erf",
    "safe_exp",
    "safe_exp2",
    "safe_expm1",
    "safe_floor",
    "safe_fmod",
    "safe_int_power_checked",
    "safe_log",
    "safe_log1p",
    "safe_log2",
    "safe_maximum",
    "safe_minimum",
    "safe_power",
    "safe_recip",
    "safe_round",
    "safe_sign",
    "safe_sin",
    "safe_sqrt",
    "safe_tanh",
    "safe_trunc",
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


def safe_int_power_checked(base: int, exponent: int) -> int:
    """Compute an ``Int32`` power without constructing an oversized integer.

    The integer ``pow`` plan permits exponents up to ``INT32_MAX``.  Python's
    built-in exponentiation would eagerly allocate an enormous intermediate for
    a value such as ``2 ** INT32_MAX`` before the result could be narrowed.  The
    exponentiation-by-squaring loop below checks every multiplication against
    the final ``Int32`` range first.  For ``|base| > 1`` the magnitude can only
    grow, so an overflowing intermediate proves that the final result cannot
    fit either.

    Args:
        base: An ``Int32`` base.
        exponent: A non-negative integer exponent.

    Returns:
        The exact power when it fits ``Int32``.

    Raises:
        OverflowError: If the exact result does not fit ``Int32``.
        ValueError: If ``exponent`` is negative.

    Examples:
        >>> from strideweave.carriers.generic.numerics import safe_int_power_checked
        >>> safe_int_power_checked(2, 30)
        1073741824
    """
    if exponent < 0:
        raise ValueError("Int32 power exponent must be non-negative")
    if exponent == 0:
        return 1
    if base == 0:
        return 0
    if base == 1:
        return 1
    if base == -1:
        return -1 if exponent & 1 else 1

    def multiply_checked(lhs: int, rhs: int) -> int:
        # Division by the positive magnitude avoids creating the overflowing
        # product at all.  The one exceptional lower bound is still covered by
        # the signed final range check.
        negative = (lhs < 0) != (rhs < 0)
        limit = -INT32_MIN if negative else INT32_MAX
        if abs(lhs) > limit // abs(rhs):
            raise OverflowError("Generic Int32 power result is out of int32 range")
        product = lhs * rhs
        if not INT32_MIN <= product <= INT32_MAX:
            raise OverflowError("Generic Int32 power result is out of int32 range")
        return product

    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply_checked(result, factor)
        remaining >>= 1
        if remaining:
            factor = multiply_checked(factor, factor)
    return result


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
    if _is_binary32_scalar(value):
        return _numpy_unary("exp", value)
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _is_binary32_scalar(value: Any) -> bool:
    """Identify NumPy's binary32 scalar without importing NumPy eagerly."""
    value_type = type(value)
    return value_type.__module__ == "numpy" and value_type.__name__ == "float32"


def _numpy_unary(name: str, value: Any) -> Any:
    numpy = _numpy()
    with float32_errstate():
        return getattr(numpy, name)(numpy.float32(value))


def _numpy_binary(name: str, lhs: Any, rhs: Any) -> Any:
    numpy = _numpy()
    with float32_errstate():
        return getattr(numpy, name)(numpy.float32(lhs), numpy.float32(rhs))


def _nan_or_inf_result(value: Any, *, nan: bool = False) -> float | None:
    """Return a Python IEEE result for legacy exceptional scalar paths."""
    numeric = float(value)
    if math.isnan(numeric):
        return math.nan
    if math.isinf(numeric):
        return math.nan if nan else numeric
    return None


def safe_abs(value: Any) -> Any:
    """Return ``abs(value)`` without changing IEEE singularities."""
    if _is_binary32_scalar(value):
        return _numpy_unary("abs", value)
    return abs(value)


def safe_sign(value: Any) -> Any:
    """Return NumPy-style sign, mapping both signed zeros to ``+0``."""
    if _is_binary32_scalar(value):
        return _numpy_unary("sign", value)
    numeric = float(value)
    if math.isnan(numeric):
        return math.nan
    if numeric > 0:
        return 1.0
    if numeric < 0:
        return -1.0
    return 0.0


def safe_recip(value: Any) -> Any:
    """Return the IEEE reciprocal, including zero and infinity cases."""
    if _is_binary32_scalar(value):
        return _numpy_unary("reciprocal", value)
    numeric = float(value)
    if math.isnan(numeric):
        return math.nan
    if numeric == 0.0:
        return math.copysign(math.inf, numeric)
    if math.isinf(numeric):
        return math.copysign(0.0, numeric)
    return 1.0 / numeric


def safe_sqrt(value: Any) -> Any:
    """Return square root, converting invalid negative inputs to ``NaN``."""
    if _is_binary32_scalar(value):
        return _numpy_unary("sqrt", value)
    try:
        return math.sqrt(value)
    except ValueError:
        return math.nan


def safe_exp2(value: Any) -> Any:
    """Return base-two exponential, saturating overflow at ``+inf``."""
    if _is_binary32_scalar(value):
        return _numpy_unary("exp2", value)
    try:
        return 2.0 ** float(value)
    except OverflowError:
        return math.inf


def safe_expm1(value: Any) -> Any:
    """Return ``exp(value) - 1`` with binary32-aware rounding.

    ``math.expm1`` is a binary64 operation even when its argument originated
    as a ``Float32`` value.  The Generic reference uses NumPy's scalar
    implementation for concrete binary32 values so the primitive rounds at
    the same boundary as the native CPU kernel.

    Args:
        value: The exponent argument.

    Returns:
        ``expm1(value)`` in the input's arithmetic representation.

    Examples:
        >>> from strideweave.carriers.generic.numerics import safe_expm1
        >>> safe_expm1(0.0)
        0.0
    """
    if _is_binary32_scalar(value):
        return _numpy_unary("expm1", value)
    return math.expm1(value)


def safe_log(value: Any) -> Any:
    """Return natural log with IEEE zero/invalid-domain results."""
    if _is_binary32_scalar(value):
        return _numpy_unary("log", value)
    numeric = float(value)
    if numeric == 0.0:
        return -math.inf
    if numeric < 0.0 or math.isnan(numeric):
        return math.nan
    return math.log(numeric)


def safe_log1p(value: Any) -> Any:
    """Return ``log1p(value)`` with binary32-aware rounding.

    Args:
        value: The value one is added to before taking the logarithm.

    Returns:
        ``log1p(value)`` in the input's arithmetic representation.

    Examples:
        >>> from strideweave.carriers.generic.numerics import safe_log1p
        >>> safe_log1p(0.0)
        0.0
    """
    if _is_binary32_scalar(value):
        return _numpy_unary("log1p", value)
    try:
        return math.log1p(value)
    except ValueError:
        return math.nan


def safe_log2(value: Any) -> Any:
    """Return base-two log with IEEE zero/invalid-domain results."""
    if _is_binary32_scalar(value):
        return _numpy_unary("log2", value)
    numeric = float(value)
    if numeric == 0.0:
        return -math.inf
    if numeric < 0.0 or math.isnan(numeric):
        return math.nan
    return math.log2(numeric)


def safe_sin(value: Any) -> Any:
    """Return sine, mapping infinite arguments to ``NaN``."""
    if _is_binary32_scalar(value):
        return _numpy_unary("sin", value)
    try:
        return math.sin(value)
    except ValueError:
        return math.nan


def safe_cos(value: Any) -> Any:
    """Return cosine, mapping infinite arguments to ``NaN``."""
    if _is_binary32_scalar(value):
        return _numpy_unary("cos", value)
    try:
        return math.cos(value)
    except ValueError:
        return math.nan


def safe_tanh(value: Any) -> Any:
    """Return hyperbolic tangent with IEEE signed-zero behavior."""
    if _is_binary32_scalar(value):
        return _numpy_unary("tanh", value)
    return math.tanh(value)


def safe_erf(value: Any) -> Any:
    """Return the error function, preserving IEEE NaN and infinity values."""
    if _is_binary32_scalar(value):
        # NumPy does not expose a scalar ``erf`` ufunc.  The platform libm
        # remains the reference implementation, but the returned value must be
        # rounded before the surrounding GELU primitive continues in binary32.
        return float32_scalar(math.erf(float(value)))
    return math.erf(value)


def safe_floor(value: Any) -> Any:
    """Return floor while preserving signed zero and exceptional values."""
    if _is_binary32_scalar(value):
        return _numpy_unary("floor", value)
    exceptional = _nan_or_inf_result(value)
    if exceptional is not None:
        return exceptional
    if float(value) == 0.0:
        return math.copysign(0.0, float(value))
    return math.floor(value)


def safe_ceil(value: Any) -> Any:
    """Return ceil while preserving signed zero and exceptional values."""
    if _is_binary32_scalar(value):
        return _numpy_unary("ceil", value)
    exceptional = _nan_or_inf_result(value)
    if exceptional is not None:
        return exceptional
    if float(value) == 0.0:
        return math.copysign(0.0, float(value))
    return math.ceil(value)


def safe_round(value: Any) -> Any:
    """Return ties-to-even round while preserving signed zero/exceptionals."""
    if _is_binary32_scalar(value):
        return _numpy_unary("rint", value)
    exceptional = _nan_or_inf_result(value)
    if exceptional is not None:
        return exceptional
    if float(value) == 0.0:
        return math.copysign(0.0, float(value))
    return round(value)


def safe_divide(lhs: Any, rhs: Any) -> Any:
    """Divide two values without allowing Python zero-division exceptions."""
    if _is_binary32_scalar(lhs) or _is_binary32_scalar(rhs):
        return _numpy_binary("divide", lhs, rhs)
    numerator = float(lhs)
    denominator = float(rhs)
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        return math.copysign(math.inf, numerator * denominator)
    return numerator / denominator


def safe_power(lhs: Any, rhs: Any) -> Any:
    """Evaluate power with NumPy Float32 exceptional-value semantics."""
    if _is_binary32_scalar(lhs) or _is_binary32_scalar(rhs):
        return _numpy_binary("power", lhs, rhs)
    try:
        return lhs**rhs
    except (OverflowError, ValueError, ZeroDivisionError):
        numeric_lhs = float(lhs)
        numeric_rhs = float(rhs)
        if numeric_lhs == 0.0 and numeric_rhs < 0.0:
            return math.copysign(math.inf, numeric_lhs)
        return math.nan


def safe_maximum(lhs: Any, rhs: Any) -> Any:
    """Evaluate NumPy-style maximum, including NaN propagation."""
    if _is_binary32_scalar(lhs) or _is_binary32_scalar(rhs):
        return _numpy_binary("maximum", lhs, rhs)
    if math.isnan(float(lhs)) or math.isnan(float(rhs)):
        return math.nan
    return max(lhs, rhs)


def safe_minimum(lhs: Any, rhs: Any) -> Any:
    """Evaluate NumPy-style minimum, including NaN propagation."""
    if _is_binary32_scalar(lhs) or _is_binary32_scalar(rhs):
        return _numpy_binary("minimum", lhs, rhs)
    if math.isnan(float(lhs)) or math.isnan(float(rhs)):
        return math.nan
    return min(lhs, rhs)


def safe_fmod(lhs: Any, rhs: Any) -> Any:
    """Evaluate truncating remainder, returning IEEE NaN at singularities."""
    if _is_binary32_scalar(lhs) or _is_binary32_scalar(rhs):
        return _numpy_binary("fmod", lhs, rhs)
    try:
        return math.fmod(lhs, rhs)
    except (ValueError, ZeroDivisionError):
        return math.nan


def safe_trunc(value: Any) -> Any:
    """Truncate toward zero without raising for non-finite values."""
    if _is_binary32_scalar(value):
        return _numpy_unary("trunc", value)
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return math.nan
    return math.trunc(value)


def is_concrete_simple_dtype(dtype: DType) -> bool:
    """Report whether ``dtype`` is one of the concrete simple storage dtypes.

    The legacy opaque categories ``DType.Any`` and ``DType.Floating`` are not
    concrete, so they stay on Generic's legacy arithmetic path.

    Args:
        dtype: The dtype to classify.

    Returns:
        ``True`` for ``DType.Float32``, ``DType.Int32``, and ``DType.Bool``.

    Examples:
        >>> from strideweave.carriers.dtype import DType
        >>> from strideweave.carriers.generic.numerics import (
        ...     is_concrete_simple_dtype,
        ... )
        >>> is_concrete_simple_dtype(DType.Floating)
        False
    """
    return dtype is DType.Float32 or dtype is DType.Int32 or dtype is DType.Bool


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


def _normalize_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool for Bool storage")
    return value


_NORMALIZERS: dict[DType, Callable[[Any, str], Any]] = {
    DType.Float32: _normalize_float32,
    DType.Int32: _normalize_int32,
    DType.Bool: _normalize_bool,
}


def normalize_storage_value(dtype: DType, value: Any, name: str = "value") -> Any:
    """Normalize one value into a concrete simple dtype's stored representation.

    A ``Float32`` carrier stores binary32-exact Python floats, an ``Int32``
    carrier stores in-range Python integers, and a ``Bool`` carrier stores
    Python booleans, so every stored value is already the value its encoding can
    hold. Storage on the legacy opaque dtypes is returned unchanged.

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
