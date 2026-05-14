from __future__ import annotations

import pickle
from collections.abc import Iterable
from enum import Enum
from importlib import import_module
from os import PathLike, fspath
from typing import Any, Protocol, cast, runtime_checkable

Data = cast(type[Any], import_module("neotorch._data").Data)


@runtime_checkable
class _SizedIndexable(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> Any: ...


@runtime_checkable
class _MutableSizedIndexable(_SizedIndexable, Protocol):
    def __setitem__(self, index: int, value: Any, /) -> None: ...


class DataType(Enum):
    Any = "Any"


def _as_sized_indexable(
    values: Iterable[Any], class_name: str, *, mutable: bool
) -> _SizedIndexable:
    try:
        iter(values)
    except TypeError as exc:
        raise TypeError(f"{class_name} requires an iterable object") from exc

    if mutable:
        if isinstance(values, _MutableSizedIndexable):
            return values
        return list(values)

    if isinstance(values, _SizedIndexable):
        return values
    return list(values)


class Generic(Data):
    def __init__(self, values: Iterable[Any], *, mutable: bool = True):
        super().__init__()
        self._mutable = bool(mutable)
        self._values: _SizedIndexable | None = _as_sized_indexable(
            values, "Generic", mutable=self._mutable
        )

    def _require_values(self) -> _SizedIndexable:
        if self._values is None:
            raise RuntimeError("Data is evicted")
        return self._values

    def _require_mutable_values(self) -> _MutableSizedIndexable:
        if not self.is_mutable():
            raise RuntimeError("Data is not mutable")
        values = self._require_values()
        if not isinstance(values, _MutableSizedIndexable):
            raise RuntimeError("Data is not mutable")
        return values

    def size(self) -> int:
        return len(self._require_values())

    def type(self) -> DataType:
        return DataType.Any

    def get_value(self, index: int) -> Any:
        return self._require_values()[index]

    def is_mutable(self) -> bool:
        return self._mutable

    def set_value(self, index: int, value: Any) -> None:
        self._require_mutable_values()[index] = value

    def new_like(self, values: Iterable[Any], *, mutable: bool = True) -> Generic:
        if type(self) is not Generic:
            raise NotImplementedError(
                "Generic data factory only supports exact Generic data"
            )
        return Generic(values, mutable=mutable)


class GenericEvictable(Generic):
    def __init__(
        self, values: Iterable[Any], path: str | PathLike[str], *, mutable: bool = True
    ):
        super().__init__(values, mutable=mutable)
        self.path = fspath(path)
        self._size = super().size()

    def size(self) -> int:
        return self._size

    def is_evictable(self) -> bool:
        return True

    def _evict(self) -> None:
        with open(self.path, "wb") as file:
            pickle.dump(self._require_values(), file)
        self._values = None

    def _promote(self) -> None:
        with open(self.path, "rb") as file:
            self._values = pickle.load(file)


__all__ = [
    "Data",
    "DataType",
    "Generic",
    "GenericEvictable",
]
