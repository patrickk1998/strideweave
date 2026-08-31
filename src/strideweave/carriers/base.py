"""Base carrier class export and the closed-implementation rule.

``Carrier`` is StrideWeave's extension interface and stays open: a new backend
implements it directly. The shipped concrete implementations — ``Generic``,
``CPU``, ``FileBacked``, ``Metal``, ``BlockDeviceCarrier``, and ``Evictable`` —
plus definition-backed ``TiledEvictable`` are closed instead. Each owns
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
Carrier = cast(Any, _carrier.Carrier)

_native_carrier_init = Carrier.__init__
_native_size = Carrier.size
_native_dtype = Carrier.dtype
_native_get_value = Carrier.get_value
_native_new_like = Carrier.new_like
_native_allocate_like = Carrier.allocate_like


def _definition_carrier_init(
    self: Any,
    slots: int | None = None,
    *,
    dtype: Any = None,
    mutable: bool = True,
) -> None:
    _native_carrier_init(self)
    from .extension import (
        _initialize_definition_storage,
        _observe_carrier_definition,
    )

    definition = _observe_carrier_definition(type(self))
    if definition is None:
        if slots is not None or dtype is not None or mutable is not True:
            raise TypeError(
                "definition-free Carrier construction accepts no storage arguments"
            )
        return
    if slots is None or dtype is None:
        raise TypeError(
            "definition-backed Carrier construction requires slots and dtype"
        )
    _initialize_definition_storage(self, definition, slots, dtype, mutable)


def _definition_size(self: Any) -> int:
    from .extension import _definition_for_instance, _provider_size

    definition = _definition_for_instance(self)
    if definition is None:
        return cast(int, _native_size(self))
    if self.is_released():
        return 0
    return _provider_size(self, definition)


def _definition_dtype(self: Any) -> Any:
    from .extension import _definition_dtype as definition_dtype
    from .extension import _definition_for_instance

    if _definition_for_instance(self) is None:
        return _native_dtype(self)
    return definition_dtype(self)


def _definition_get_value(self: Any, index: int) -> Any:
    from .extension import _definition_for_instance, _provider_size

    definition = _definition_for_instance(self)
    if definition is None:
        return _native_get_value(self, index)
    if self.is_released():
        raise RuntimeError("Carrier is released")
    if type(index) is not int:
        raise TypeError("Carrier index must be an integer")
    if index < 0 or index >= _provider_size(self, definition):
        raise IndexError("Carrier index out of range")
    return definition.storage.read(self, index)


def _definition_set_value(self: Any, index: int, value: Any) -> None:
    from .extension import _definition_for_instance, _provider_size

    definition = _definition_for_instance(self)
    if definition is None:
        raise RuntimeError("Carrier is not mutable")
    if self.is_released():
        raise RuntimeError("Carrier is released")
    if not self.is_mutable():
        raise RuntimeError("Carrier is not mutable")
    if type(index) is not int:
        raise TypeError("Carrier index must be an integer")
    if index < 0 or index >= _provider_size(self, definition):
        raise IndexError("Carrier index out of range")
    result = definition.storage.write(self, index, value)
    if result is not None:
        raise TypeError("StorageProvider.write must return None")
    self._increment_version()


def _definition_new_like(
    self: Any,
    values: Any,
    *,
    mutable: bool = True,
    dtype: Any = None,
) -> Any:
    from .extension import _definition_for_instance

    definition = _definition_for_instance(self)
    if definition is None:
        return _native_new_like(self, values, mutable=mutable)
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError("values must be a finite iterable") from error
    result_dtype = self.dtype() if dtype is None else dtype
    result = type(self)(len(materialized), mutable=mutable, dtype=result_dtype)
    result_definition = cast(Any, definition)
    for index, value in enumerate(materialized):
        returned = result_definition.storage.write(result, index, value)
        if returned is not None:
            raise TypeError("StorageProvider.write must return None")
    return result


def _definition_allocate_like(
    self: Any,
    size: int,
    *,
    mutable: bool = True,
    dtype: Any = None,
    empty: bool = False,
) -> Any:
    from .extension import _definition_for_instance

    if _definition_for_instance(self) is None:
        return _native_allocate_like(
            self,
            size,
            mutable=mutable,
            dtype=dtype,
            empty=empty,
        )
    if type(empty) is not bool:
        raise TypeError("empty must be a bool")
    result_dtype = self.dtype() if dtype is None else dtype
    return type(self)(size, mutable=mutable, dtype=result_dtype)


def _definition_is_mutable(self: Any) -> bool:
    from .extension import _definition_for_instance, _definition_is_mutable

    if _definition_for_instance(self) is None:
        return False
    return _definition_is_mutable(self)


def _definition_supports_storage_dtype(self: Any, dtype: Any) -> bool:
    from .extension import _definition_for_instance

    definition = _definition_for_instance(self)
    if definition is None:
        return dtype is self.dtype()
    result = definition.storage.supports_dtype(self, dtype)
    if type(result) is not bool:
        raise TypeError("StorageProvider.supports_dtype must return a bool")
    return result


def _definition_release(self: Any) -> None:
    from .extension import _definition_for_instance

    definition = _definition_for_instance(self)
    if definition is None:
        return
    result = definition.storage.release(self)
    if result is not None:
        raise TypeError("StorageProvider.release must return None")


def _facet(self: Any, facet_type: type) -> Any:
    from .extension import _definition_facet

    return _definition_facet(self, facet_type)


def _require_facet(self: Any, facet_type: type) -> Any:
    result = _facet(self, facet_type)
    if result is None:
        name = getattr(facet_type, "__name__", repr(facet_type))
        raise NotImplementedError(
            f"{type(self).__name__} has no facet for exact type {name}"
        )
    return result


def _definition_dispatch_hook(self: Any, operation_name: str) -> Any:
    from .definition_dispatch import definition_dispatch_operation
    from .extension import _definition_for_instance

    if _definition_for_instance(self) is None:
        raise NotImplementedError(
            f"Carrier does not support operation {operation_name!r}"
        )
    return definition_dispatch_operation(self, operation_name)


def _execute_resolved_invocation(self: Any, invocation: Any) -> Any:
    from .definition_dispatch import DefinitionBackedOperation

    operation = self.dispatch_op(invocation.name)
    if not isinstance(operation, DefinitionBackedOperation):
        raise TypeError("resolved invocation requires a definition-backed operation")
    operation._supply_resolved_invocation(invocation)
    return operation.forward(*invocation.operands)


Carrier.__init__ = _definition_carrier_init
Carrier.size = _definition_size
Carrier.dtype = _definition_dtype
Carrier.get_value = _definition_get_value
Carrier.set_value = _definition_set_value
Carrier.new_like = _definition_new_like
Carrier.allocate_like = _definition_allocate_like
Carrier._is_mutable = _definition_is_mutable
Carrier._supports_storage_dtype = _definition_supports_storage_dtype
Carrier._release = _definition_release
Carrier._dispatch_op = _definition_dispatch_hook
Carrier._execute_resolved_invocation = _execute_resolved_invocation
Carrier.facet = _facet
Carrier.require_facet = _require_facet

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
