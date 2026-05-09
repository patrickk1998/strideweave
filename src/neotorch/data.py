from collections.abc import Iterable
from enum import Enum
from importlib import import_module
from typing import Any, Protocol, cast, runtime_checkable

Data = cast(type[Any], import_module("neotorch._data").Data)


@runtime_checkable
class _SizedIndexable(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> Any: ...


class DataType(Enum):
    Any = "Any"


class Generic(Data):
    def __init__(self, values: Iterable[Any]):
        super().__init__()
        self.values: _SizedIndexable
        try:
            iter(values)
        except TypeError as exc:
            raise TypeError("Generic requires an iterable object") from exc

        if isinstance(values, _SizedIndexable):
            self.values = values
        else:
            self.values = list(values)

    def size(self) -> int:
        return len(self.values)

    def type(self) -> DataType:
        return DataType.Any

    def get_value(self, index: int) -> Any:
        return self.values[index]


__all__ = [
    "Data",
    "DataType",
    "Generic",
]
