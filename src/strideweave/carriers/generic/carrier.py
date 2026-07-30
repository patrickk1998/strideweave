from __future__ import annotations

from collections.abc import Iterable
from operator import index as operator_index
from typing import Any, Protocol, final, runtime_checkable

from ..base import Carrier, reject_carrier_subclass
from ..dtype import (
    DType,
    accepts_storage_dtype,
    storage_zero,
    validate_storage_dtype,
)
from .numerics import (
    is_concrete_simple_dtype,
    normalize_storage_value,
    normalize_storage_values,
)


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


# Generic stores Python objects, so it accepts the legacy opaque-storage
# descriptors, and it is the behavioral reference for the concrete simple
# dtypes, so it accepts those too.
_GENERIC_DTYPES = (DType.Any, DType.Floating, DType.Float32, DType.Int32)


def _validate_generic_dtype(dtype: DType) -> DType:
    return validate_storage_dtype(dtype, carrier="Generic", accepted=_GENERIC_DTYPES)


# A fresh concrete allocation starts at its dtype's zero, per RT004's
# initialized storage contract; `storage_zero` is that rule's single source and
# returns None for legacy opaque storage, which keeps its historical fill.


def _normalized_storage(
    values: Iterable[Any], dtype: DType, class_name: str, *, mutable: bool
) -> _SizedIndexable:
    """Build this carrier's backing storage for ``dtype``.

    Concrete simple dtypes are normalized into a list this carrier owns, so no
    caller-held alias can later place a value the encoding cannot represent, or
    change stored values without the version counter observing it. Legacy opaque
    storage keeps its documented aliasing behavior.
    """
    if not is_concrete_simple_dtype(dtype):
        return _as_sized_indexable(values, class_name, mutable=mutable)
    try:
        supplied = list(values)
    except TypeError as exc:
        raise TypeError(f"{class_name} requires an iterable object") from exc
    return normalize_storage_values(dtype, supplied, f"{class_name} value")


@final
class Generic(Carrier):
    """Python-backed carrier storage for generic StrideWeave tensors.

    Generic accepts the legacy opaque-storage descriptors — ``DType.Floating``
    for differentiable numeric values and ``DType.Any`` for arbitrary objects —
    and the concrete simple dtypes ``DType.Float32`` and ``DType.Int32``, for
    which it is StrideWeave's behavioral reference implementation.

    Concrete storage is normalized and owned: a ``Float32`` carrier holds
    binary32-exact floats and an ``Int32`` carrier holds in-range integers,
    copied into storage this carrier owns. Legacy opaque storage continues to
    alias a mutable container the caller supplied.

    Generic is a closed implementation: extend StrideWeave with a sibling
    ``Carrier`` rather than a specialization of this one.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        reject_carrier_subclass("Generic")

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
        self._values: _SizedIndexable | None = _normalized_storage(
            values, self._dtype, "Generic", mutable=self._mutable
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

    def _supports_storage_dtype(self, dtype: DType) -> bool:
        """Report the dtypes Generic can allocate, whatever it holds now.

        The same accepted set its constructor validates against, so the
        reference backend cannot advertise storage it would then refuse.
        """
        return accepts_storage_dtype(dtype, _GENERIC_DTYPES)

    def set_value(self, index: int, value: Any) -> None:
        values = self._require_mutable_values()
        values[index] = normalize_storage_value(self._dtype, value)
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> Generic:
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
        allocated_dtype = self._dtype if dtype is None else dtype
        # A concrete carrier's storage always holds representable values, so
        # fresh allocations start at that dtype's zero rather than at None.
        initial: Any = storage_zero(allocated_dtype)
        return Generic(
            [initial] * normalized_size,
            mutable=mutable,
            dtype=allocated_dtype,
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
        to_scatter._require_single_subtensor("scatter")
        scatter_onto._require_single_subtensor("scatter")
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
            BroadcastOperation,
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
            "broadcast_to": BroadcastOperation,
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
