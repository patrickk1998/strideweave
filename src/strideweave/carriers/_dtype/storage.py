"""Carrier-facing validation for homogeneous dtype storage."""

from __future__ import annotations

from .model import DType


def _join_alternatives(names: list[str]) -> str:
    """Join ``names`` into an English alternative list."""
    if len(names) < 3:
        return " or ".join(names)
    return f"{', '.join(names[:-1])}, or {names[-1]}"


def validate_storage_dtype(
    dtype: object, *, carrier: str, accepted: tuple[DType, ...]
) -> DType:
    """Validate a dtype supplied as one carrier's homogeneous storage.

    A carrier holds elements of exactly one dtype, so it accepts a fixed set of
    simple or legacy opaque descriptors. Compound descriptors are rejected with
    a message naming the deferred capability rather than partially constructing
    a carrier that could only hold one of their planes.

    Args:
        dtype: Candidate descriptor supplied to a carrier constructor.
        carrier: Carrier class name used in the error messages.
        accepted: Descriptors this carrier can store, in message order.

    Returns:
        The validated descriptor, unchanged.

    Raises:
        TypeError: If ``dtype`` is not a :class:`DType`.
        ValueError: If ``dtype`` is a descriptor this carrier cannot store.

    Examples:
        >>> from strideweave.carriers.dtype import DType, validate_storage_dtype
        >>> validate_storage_dtype(
        ...     DType.Float32, carrier="CPU", accepted=(DType.Float32, DType.Int32)
        ... ) is DType.Float32
        True
    """
    if not isinstance(dtype, DType):
        raise TypeError(f"{carrier} dtype must be a DType")
    if any(dtype is candidate for candidate in accepted):
        return dtype
    if dtype.is_compound():
        raise ValueError(
            f"{carrier} cannot store compound dtype {dtype.name!r}: a carrier "
            "holds one simple dtype, and a compound representation needs one "
            "carrier per simple_types plane, which is not implemented"
        )
    expected = _join_alternatives([f"DType.{candidate.name}" for candidate in accepted])
    raise ValueError(f"{carrier} dtype must be {expected}")
