from __future__ import annotations

from collections.abc import Iterable
from operator import index as operator_index
from typing import Any, final

from ..base import Carrier, reject_carrier_subclass
from ..dtype import (
    DType,
    accepts_storage_dtype,
    storage_zero,
    validate_storage_dtype,
)
from .numerics import normalize_storage_value, normalize_storage_values

# Generic is the behavioral reference for these concrete storage dtypes.
_GENERIC_DTYPES = (DType.Float32, DType.Int32, DType.Bool)


def _validate_generic_dtype(dtype: DType) -> DType:
    return validate_storage_dtype(dtype, carrier="Generic", accepted=_GENERIC_DTYPES)


def _normalized_storage(values: Iterable[Any], dtype: DType) -> list[Any]:
    """Build normalized backing storage owned by this carrier."""
    try:
        supplied = list(values)
    except TypeError as exc:
        raise TypeError("Generic requires an iterable object") from exc
    return normalize_storage_values(dtype, supplied, "Generic value")


@final
class Generic(Carrier):
    """Python-backed carrier storage for generic StrideWeave tensors.

    Generic accepts the concrete simple dtypes ``DType.Float32``,
    ``DType.Int32``, and ``DType.Bool``, for which it is StrideWeave's
    behavioral reference implementation. Storage is normalized and owned: a
    ``Float32`` carrier holds
    binary32-exact floats, an ``Int32`` carrier holds in-range integers, and a
    ``Bool`` carrier holds normalized Python booleans, copied into storage this
    carrier owns.

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
        dtype: DType,
    ):
        super().__init__()
        self._mutable = bool(mutable)
        self._dtype = _validate_generic_dtype(dtype)
        self._values: list[Any] | None = _normalized_storage(values, self._dtype)

    def _require_values(self) -> list[Any]:
        if self._values is None:
            if self.is_released():
                raise RuntimeError("Carrier is released")
            raise RuntimeError("Carrier storage is unavailable")
        return self._values

    def _release(self) -> None:
        self._values = None

    def _require_mutable_values(self) -> list[Any]:
        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        values = self._require_values()
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
            ReshapeOperation,
            SqueezeOperation,
            UnsqueezeOperation,
        )
        from .as_strided_ops import GenericAsStridedOperation
        from .convolution_ops import GenericConvGeneralOperation
        from .indexing_ops import (
            GenericGatherOperation,
            GenericScatterAddOperation,
            GenericScatterOperation,
        )
        from .ops import (
            GenericAbsOperation,
            GenericAddOperation,
            GenericCeilOperation,
            GenericCosOperation,
            GenericDivOperation,
            GenericElementwiseMulOperation,
            GenericELUOperation,
            GenericErfOperation,
            GenericExp2Operation,
            GenericExpOperation,
            GenericFloorOperation,
            GenericGELUOperation,
            GenericLeakyReLUOperation,
            GenericLog2Operation,
            GenericLogOperation,
            GenericMatmulOperation,
            GenericMaximumOperation,
            GenericMinimumOperation,
            GenericNegOperation,
            GenericPowOperation,
            GenericRecipOperation,
            GenericReLUOperation,
            GenericRemOperation,
            GenericRoundOperation,
            GenericRsqrtOperation,
            GenericScalarMulOperation,
            GenericSigmoidOperation,
            GenericSignOperation,
            GenericSiLUOperation,
            GenericSinOperation,
            GenericSoftplusOperation,
            GenericSqrtOperation,
            GenericSubOperation,
            GenericTanhOperation,
        )
        from .predicate_ops import (
            GenericEqOperation,
            GenericLeOperation,
            GenericLogicalNotOperation,
            GenericLtOperation,
            GenericNeOperation,
        )
        from .reduction_ops import (
            GenericArgMaxOperation,
            GenericArgMinOperation,
            GenericCumsumOperation,
            GenericReduceMaxOperation,
            GenericReduceMinOperation,
            GenericReduceProdOperation,
        )
        from .reduction_ops import (
            GenericReduceSumOperation as GenericPlannedReduceSumOperation,
        )
        from .selection_ops import (
            GenericSortIndicesOperation,
            GenericSortValuesOperation,
            GenericTopKIndicesOperation,
            GenericTopKValuesOperation,
        )
        from .ternary_ops import GenericClampOperation, GenericSelectOperation

        operations = {
            "add": GenericAddOperation,
            "abs": GenericAbsOperation,
            "argmax": GenericArgMaxOperation,
            "argmin": GenericArgMinOperation,
            "as_strided": GenericAsStridedOperation,
            "broadcast_to": BroadcastOperation,
            "ceil": GenericCeilOperation,
            "clamp": GenericClampOperation,
            "conv_general": GenericConvGeneralOperation,
            "cos": GenericCosOperation,
            "cumsum": GenericCumsumOperation,
            "div": GenericDivOperation,
            "elu": GenericELUOperation,
            "elementwise_mul": GenericElementwiseMulOperation,
            "exp": GenericExpOperation,
            "exp2": GenericExp2Operation,
            "eq": GenericEqOperation,
            "erf": GenericErfOperation,
            "floor": GenericFloorOperation,
            "gather": GenericGatherOperation,
            "gelu": GenericGELUOperation,
            "le": GenericLeOperation,
            "leaky_relu": GenericLeakyReLUOperation,
            "logical_not": GenericLogicalNotOperation,
            "log": GenericLogOperation,
            "log2": GenericLog2Operation,
            "lt": GenericLtOperation,
            "matmul": GenericMatmulOperation,
            "maximum": GenericMaximumOperation,
            "minimum": GenericMinimumOperation,
            "mul": GenericScalarMulOperation,
            "ne": GenericNeOperation,
            "neg": GenericNegOperation,
            "permute": PermuteOperation,
            "pow": GenericPowOperation,
            "rearrange": RearrangeOperation,
            "reduce_max": GenericReduceMaxOperation,
            "reduce_min": GenericReduceMinOperation,
            "reduce_prod": GenericReduceProdOperation,
            "reduce_sum": GenericPlannedReduceSumOperation,
            "relu": GenericReLUOperation,
            "recip": GenericRecipOperation,
            "rem": GenericRemOperation,
            "reshape": ReshapeOperation,
            "round": GenericRoundOperation,
            "rsqrt": GenericRsqrtOperation,
            "scatter": GenericScatterOperation,
            "scatter_add": GenericScatterAddOperation,
            "select": GenericSelectOperation,
            "sigmoid": GenericSigmoidOperation,
            "silu": GenericSiLUOperation,
            "sign": GenericSignOperation,
            "sin": GenericSinOperation,
            "softplus": GenericSoftplusOperation,
            "sub": GenericSubOperation,
            "squeeze": SqueezeOperation,
            "sqrt": GenericSqrtOperation,
            "tanh": GenericTanhOperation,
            "unsqueeze": UnsqueezeOperation,
            "view": GenericViewOperation,
            "_sort_values": GenericSortValuesOperation,
            "_sort_indices": GenericSortIndicesOperation,
            "_topk_values": GenericTopKValuesOperation,
            "_topk_indices": GenericTopKIndicesOperation,
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
