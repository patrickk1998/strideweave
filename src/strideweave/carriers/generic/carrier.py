from __future__ import annotations

from collections.abc import Iterable
from operator import index as operator_index
from typing import Any, Protocol, runtime_checkable

from ..base import Carrier
from ..dtype import DType


@runtime_checkable
class _SizedIndexable(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> Any: ...


@runtime_checkable
class _MutableSizedIndexable(_SizedIndexable, Protocol):
    def __setitem__(self, index: int, value: Any, /) -> None: ...


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


def _validate_generic_dtype(dtype: DType) -> DType:
    if not isinstance(dtype, DType):
        raise TypeError("Generic dtype must be a DType")
    if dtype not in (DType.Any, DType.Floating):
        raise ValueError("Generic dtype must be DType.Any or DType.Floating")
    return dtype


class Generic(Carrier):
    """Python-backed carrier storage for generic StrideWeave tensors."""

    def __init__(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType = DType.Floating,
    ):
        super().__init__()
        self._mutable = bool(mutable)
        self._dtype = _validate_generic_dtype(dtype)
        self._values: _SizedIndexable | None = _as_sized_indexable(
            values, "Generic", mutable=self._mutable
        )

    def _require_values(self) -> _SizedIndexable:
        if self._values is None:
            if self.is_released():
                raise RuntimeError("Carrier is released")
            raise RuntimeError("Carrier storage is unavailable")
        return self._values

    def _release(self) -> None:
        self._values = None

    def _require_mutable_values(self) -> _MutableSizedIndexable:
        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        values = self._require_values()
        if not isinstance(values, _MutableSizedIndexable):
            raise RuntimeError("Carrier is not mutable")
        return values

    def size(self) -> int:
        return len(self._require_values())

    def dtype(self) -> DType:
        return self._dtype

    def get_value(self, index: int) -> Any:
        return self._require_values()[index]

    def _is_mutable(self) -> bool:
        return self._mutable

    def set_value(self, index: int, value: Any) -> None:
        self._require_mutable_values()[index] = value
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> Generic:
        if type(self) is not Generic:
            raise NotImplementedError(
                "Generic carrier factory only supports exact Generic carriers"
            )
        return Generic(
            values, mutable=mutable, dtype=self._dtype if dtype is None else dtype
        )

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> Generic:
        del empty
        normalized_size = operator_index(size)
        if normalized_size < 0:
            raise ValueError("Generic allocation size must be non-negative")
        return Generic(
            [None] * normalized_size,
            mutable=mutable,
            dtype=self._dtype if dtype is None else dtype,
        )

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        from ...layout import Layout
        from ...tensor import Tensor

        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        if not isinstance(to_scatter, Tensor):
            raise TypeError("to_scatter must be a Tensor")
        if not isinstance(scatter_onto, Tensor):
            raise TypeError("scatter_onto must be a Tensor")
        if not isinstance(mapping, Layout):
            raise TypeError("mapping must be a Layout")
        if scatter_onto.carrier is not self:
            raise ValueError("scatter_onto must be backed by this carrier")
        if mapping.shape != to_scatter.layout.shape:
            raise ValueError("mapping shape must match to_scatter layout shape")

        normalized_offset = operator_index(mapping_offset)
        if normalized_offset < 0:
            raise ValueError("mapping_offset must be non-negative")

        for logical_index in range(to_scatter.size()):
            carrier_index = (
                scatter_onto.offset + normalized_offset + mapping.index(logical_index)
            )
            self[carrier_index] = to_scatter[logical_index]

    def _dispatch_op(self, operation_name: str) -> Any:
        from ..shared_ops import (
            GenericViewOperation,
            PermuteOperation,
            RearrangeOperation,
        )
        from .ops import (
            GenericAddOperation,
            GenericDivOperation,
            GenericElementwiseMulOperation,
            GenericELUOperation,
            GenericExpOperation,
            GenericGELUOperation,
            GenericLeakyReLUOperation,
            GenericMatmulOperation,
            GenericPowOperation,
            GenericReduceSumOperation,
            GenericReLUOperation,
            GenericScalarMulOperation,
            GenericSigmoidOperation,
            GenericSiLUOperation,
            GenericSoftplusOperation,
            GenericSubOperation,
            GenericTanhOperation,
        )

        operations = {
            "add": GenericAddOperation,
            "div": GenericDivOperation,
            "elu": GenericELUOperation,
            "elementwise_mul": GenericElementwiseMulOperation,
            "exp": GenericExpOperation,
            "gelu": GenericGELUOperation,
            "leaky_relu": GenericLeakyReLUOperation,
            "matmul": GenericMatmulOperation,
            "mul": GenericScalarMulOperation,
            "permute": PermuteOperation,
            "pow": GenericPowOperation,
            "rearrange": RearrangeOperation,
            "reduce": GenericReduceSumOperation,
            "relu": GenericReLUOperation,
            "sigmoid": GenericSigmoidOperation,
            "silu": GenericSiLUOperation,
            "softplus": GenericSoftplusOperation,
            "sub": GenericSubOperation,
            "tanh": GenericTanhOperation,
            "view": GenericViewOperation,
        }
        try:
            operation_type = operations[operation_name]
        except KeyError as exc:
            raise NotImplementedError(
                f"Generic carrier does not support operation '{operation_name}'"
            ) from exc
        return operation_type()


__all__ = [
    "Generic",
]
