"""Base carrier class export and the closed-implementation rule.

``Carrier`` is StrideWeave's extension interface and stays open: a new backend
implements it directly. The shipped concrete implementations — ``Generic``,
``CPU``, ``FileBacked``, and ``Evictable`` — are closed instead. Each owns
storage invariants its own factories, capability declarations, and dispatch
metadata are stated in terms of its exact class, so a specialization inherits
claims it cannot honor: a ``Generic`` subclass would advertise every plan
``Generic`` executes while ``Generic.new_like`` refused to allocate a result for
it. Composition is the supported way to build on one, exactly as ``Evictable``
builds a memory hierarchy out of two carriers it owns.
"""

from importlib import import_module
from typing import Any, NoReturn, cast

_carrier = import_module("strideweave._carrier")
Carrier = cast(type[Any], _carrier.Carrier)

# The single wording every closed carrier refuses subclassing with. The native
# CPU binding repeats it in C++ (`_cpu.cpp`), and
# `tests/test_carrier.py::test_a_closed_carrier_states_one_refusal` pins the two
# together.
CLOSED_CARRIER_MESSAGE = (
    "{name} is a closed carrier implementation and cannot be subclassed; "
    "implement a sibling Carrier instead, normally by composing existing "
    "carriers and lowering operations onto them the way Evictable does"
)


def reject_carrier_subclass(name: str) -> NoReturn:
    """Refuse the creation of a subclass of a closed concrete carrier.

    Called from the closed carrier's ``__init_subclass__``, so the refusal
    happens when the subclass is defined rather than when one of the inherited
    contracts it cannot satisfy is first exercised.

    Args:
        name: Name of the closed carrier being subclassed.

    Raises:
        TypeError: Always, naming ``name`` and the supported alternative.

    Examples:
        >>> from strideweave.carriers.base import reject_carrier_subclass
        >>> try:
        ...     reject_carrier_subclass("Generic")
        ... except TypeError as error:
        ...     print(str(error).split(";")[0])
        Generic is a closed carrier implementation and cannot be subclassed
    """
    raise TypeError(CLOSED_CARRIER_MESSAGE.format(name=name))


__all__ = [
    "CLOSED_CARRIER_MESSAGE",
    "Carrier",
    "reject_carrier_subclass",
]
