"""Canonical dtype-structure values and encoding helpers."""

from __future__ import annotations

from typing import ClassVar, Final, Self, cast


class WholeExtent:
    """Symbolic block extent covering everything below a scale level.

    A level whose extent is :data:`Whole` produces exactly one scale for the
    entire tensor, which is how per-tensor scaling is expressed without naming
    a concrete size. ``Whole`` is the only instance: constructing, copying,
    or unpickling a ``WholeExtent`` yields it, so structurally equal levels
    cannot be built from distinct whole extents.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.NVFP4.levels[-1].block is sw.Whole
        True
        >>> sw.WholeExtent() is sw.Whole
        True
    """

    __slots__ = ()

    _instance: ClassVar[WholeExtent | None] = None

    def __new__(cls) -> Self:
        if WholeExtent._instance is None:
            WholeExtent._instance = super().__new__(cls)
        return cast("Self", WholeExtent._instance)

    def __copy__(self) -> WholeExtent:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> WholeExtent:
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return (WholeExtent, ())

    def __repr__(self) -> str:
        return "Whole"


Whole: Final[WholeExtent] = WholeExtent()


def _structure_difference(
    registered: object, pickled: object, path: str = "structure"
) -> tuple[str, object, object]:
    """Return where two structures first differ, and both values there.

    A structure spans every referenced descriptor, so reporting the whole graph
    twice would bury the disagreement. Descending to the first differing leaf
    names the field that actually changed instead.
    """
    if type(registered) is tuple and type(pickled) is tuple:
        for position in range(min(len(registered), len(pickled))):
            if registered[position] != pickled[position]:
                return _structure_difference(
                    registered[position], pickled[position], f"{path}[{position}]"
                )
        if len(registered) != len(pickled):
            return (f"{path} length", len(registered), len(pickled))
    return (path, registered, pickled)


def _encoded_leaf(value: object) -> str:
    """Return the canonical encoding of one permitted structure value.

    Every leaf of a structure is encoded as a string that names its exact type
    alongside its value, because a structure is compared to decide identity.
    Plain Python equality would make ``True``, ``1``, and ``1.0`` one value and
    would make a ``NaN``-bearing structure unequal to itself, so neither the
    registry key nor a pickle comparison may rely on it. Floats are recorded in
    their exact hexadecimal form, which round-trips every finite value and gives
    infinities and ``NaN`` a single deterministic spelling.

    Raises:
        TypeError: If ``value`` is not a permitted structure leaf.
    """
    if value is None:
        return "none"
    if value is Whole:
        return "whole"
    kind = type(value)
    if kind is bool:
        return f"bool:{value!r}"
    if kind is int:
        return f"int:{value!r}"
    if kind is float:
        return f"float:{cast('float', value).hex()}"
    if kind is str:
        return f"str:{value}"
    raise TypeError(
        f"A dtype structure holds only exact strings, numbers, None, Whole, and "
        f"tuples of those, not {kind.__name__}"
    )


def _encoded_structure(dtype: object, value: object) -> object:
    """Return ``value`` with every leaf canonically encoded.

    Used for the part of a structure an implementation supplies, which is the
    only part this module did not build itself. Encoding *is* the validation:
    anything that is not a permitted leaf or a tuple of them is rejected before
    it can reach a registry key or a pickle.
    """
    if type(value) is tuple:
        return tuple(_encoded_structure(dtype, item) for item in value)
    try:
        return _encoded_leaf(value)
    except TypeError as error:
        raise TypeError(f"{type(dtype).__name__} structure: {error}") from error


def _require_encoded_structure(dtype: object, value: object) -> None:
    """Reject a structure holding anything but encoded leaves and tuples.

    The contract layers encode their own fields, so this only has to prove that
    the assembled result is exactly that: a tree of tuples whose leaves are
    strings. It keeps an implementation that reaches past the documented
    extension hook from recording a value that hashes or compares through user
    code.
    """
    if type(value) is tuple:
        for item in value:
            _require_encoded_structure(dtype, item)
        return
    if type(value) is not str:
        raise TypeError(
            f"{type(dtype).__name__} described its structure with a "
            f"{type(value).__name__}: a dtype structure holds only encoded "
            "leaves and tuples of them"
        )


def _described_structure_value(value: object) -> str:
    """Return the readable form of an encoded structure leaf.

    Diagnostics report the field that differs between two structures, so the
    encoding is undone for the message rather than shown as stored.
    """
    if type(value) is not str:
        return repr(value)
    if value == "none":
        return "None"
    if value == "whole":
        return "Whole"
    tag, separator, payload = value.partition(":")
    if not separator:
        return repr(value)
    if tag == "str":
        return repr(payload)
    if tag in ("bool", "int"):
        return payload
    if tag == "float":
        try:
            return repr(float.fromhex(payload))
        except ValueError:
            return repr(value)
    return repr(value)
