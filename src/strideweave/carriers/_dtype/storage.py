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


def accepts_storage_dtype(dtype: DType, accepted: tuple[DType, ...]) -> bool:
    """Report whether ``accepted`` contains ``dtype``, matched by identity.

    The boolean counterpart of :func:`validate_storage_dtype`, for the
    structural question a carrier answers about what it could allocate rather
    than the refusal it raises about what it was handed. Both read the same
    accepted set, so what a carrier reports and what it accepts cannot drift.

    Args:
        dtype: The descriptor to look for.
        accepted: Descriptors this carrier can store.

    Returns:
        ``True`` when ``dtype`` is one of the accepted descriptors.

    Examples:
        >>> from strideweave.carriers.dtype import DType, accepts_storage_dtype
        >>> accepts_storage_dtype(DType.Int32, (DType.Float32, DType.Int32))
        True
    """
    # Descriptors are registry singletons (SW002): identity, never equality, so
    # an object with a spoofed __eq__ cannot claim to be storable here.
    return any(dtype is candidate for candidate in accepted)


def storage_zero(dtype: DType) -> object:
    """Return the value a fresh or unaddressed slot of ``dtype`` holds.

    Concrete simple storage is always representable, so an allocation that has
    not been written to, and a physical slot no logical index of a layout
    addresses, both hold that dtype's zero rather than a placeholder object.
    Legacy opaque storage (``DType.Any``, ``DType.Floating``) stores arbitrary
    Python objects and has no zero, so it keeps ``None``.

    Args:
        dtype: The storage dtype whose zero is wanted.

    Returns:
        ``0.0`` for ``DType.Float32``, ``0`` for ``DType.Int32``, and ``None``
        for every dtype without a defined stored zero.

    Examples:
        >>> from strideweave.carriers.dtype import DType, storage_zero
        >>> storage_zero(DType.Int32)
        0
        >>> storage_zero(DType.Floating) is None
        True
    """
    # Descriptors are identity singletons (SW002), and the built-in registry is
    # installed after this module is imported, so the comparison happens here
    # rather than in a module-level table.
    if dtype is DType.Float32:
        return 0.0
    if dtype is DType.Int32:
        return 0
    return None
