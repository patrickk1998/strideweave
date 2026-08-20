"""PyTorch MPS-backed storage for the closed Metal carrier."""

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
from ..generic.numerics import normalize_storage_value, normalize_storage_values
from ._runtime import MetalRuntime, load_metal_runtime

_METAL_DTYPES = (DType.Float32, DType.Int32, DType.Bool)


def _validate_metal_dtype(dtype: DType) -> DType:
    return validate_storage_dtype(dtype, carrier="Metal", accepted=_METAL_DTYPES)


def _torch_dtype(runtime: MetalRuntime, dtype: DType) -> Any:
    if dtype is DType.Float32:
        return runtime.torch.float32
    if dtype is DType.Int32:
        return runtime.torch.int32
    return runtime.torch.bool


def _allocation_error(
    runtime: MetalRuntime, *, size: int, dtype: DType, error: Exception
) -> RuntimeError:
    torch_version = getattr(runtime.torch, "__version__", "<unknown>")
    tilelang_version = getattr(runtime.tilelang, "__version__", "<unknown>")
    return RuntimeError(
        "Metal could not allocate private MPS storage "
        f"(size={size}, dtype=DType.{dtype.name}, torch={torch_version}, "
        f"tilelang={tilelang_version}). Verify available Metal memory and the "
        "pinned optional runtime. "
        f"Original allocation error: {error}"
    )


@final
class Metal(Carrier):
    """One-dimensional typed storage on an Apple Metal device.

    The carrier owns a private PyTorch MPS tensor as its temporary allocation
    boundary. PyTorch and TileLang remain optional and are loaded only when a
    Metal carrier is constructed.

    Args:
        size: Number of physical storage slots to allocate.
        mutable: Whether public writes are permitted.
        dtype: Homogeneous storage dtype; ``Float32``, ``Int32``, or ``Bool``.
        empty: Whether zero-initialization may be skipped.

    Examples:
        >>> import strideweave as sw
        >>> carrier = sw.Metal(3, dtype=sw.DType.Float32)  # doctest: +SKIP
        >>> carrier.size()  # doctest: +SKIP
        3
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        reject_carrier_subclass("Metal")

    def __init__(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType = DType.Float32,
        empty: bool = False,
    ) -> None:
        super().__init__()
        normalized_size = operator_index(size)
        if normalized_size < 0:
            raise ValueError("Metal size must be non-negative")
        validated_dtype = _validate_metal_dtype(dtype)
        runtime = load_metal_runtime()
        try:
            factory = runtime.torch.empty if empty else runtime.torch.zeros
            storage = factory(
                normalized_size,
                dtype=_torch_dtype(runtime, validated_dtype),
                device="mps",
            )
        except Exception as exc:
            raise _allocation_error(
                runtime, size=normalized_size, dtype=validated_dtype, error=exc
            ) from exc
        self._publish_storage(
            runtime,
            storage,
            size=normalized_size,
            dtype=validated_dtype,
            mutable=mutable,
        )

    def _publish_storage(
        self,
        runtime: MetalRuntime,
        storage: Any,
        *,
        size: int,
        dtype: DType,
        mutable: bool,
    ) -> None:
        self._runtime = runtime
        self._storage: Any | None = storage
        self._size = size
        self._dtype = dtype
        self._mutable = bool(mutable)

    @classmethod
    def _from_values(cls, values: list[Any], *, mutable: bool, dtype: DType) -> Metal:
        runtime = load_metal_runtime()
        try:
            storage = runtime.torch.tensor(
                values, dtype=_torch_dtype(runtime, dtype), device="mps"
            )
        except Exception as exc:
            raise _allocation_error(
                runtime, size=len(values), dtype=dtype, error=exc
            ) from exc
        result = cls.__new__(cls)
        Carrier.__init__(result)
        result._publish_storage(
            runtime, storage, size=len(values), dtype=dtype, mutable=mutable
        )
        return result

    def _require_storage(self) -> Any:
        storage = self._storage
        if storage is None:
            if self.is_released():
                raise RuntimeError("Carrier is released")
            raise RuntimeError("Metal storage is unavailable")
        return storage

    def _validate_index(self, index: int) -> int:
        normalized = operator_index(index)
        if normalized < 0 or normalized >= self.size():
            raise IndexError("Carrier index out of range")
        return normalized

    def size(self) -> int:
        return self._size if self._storage is not None else 0

    def dtype(self) -> DType:
        return self._dtype

    def get_value(self, index: int) -> Any:
        storage = self._require_storage()
        normalized = self._validate_index(index)
        self._runtime.synchronize()
        return storage[normalized].item()

    def _is_mutable(self) -> bool:
        return self._mutable

    def _supports_storage_dtype(self, dtype: DType) -> bool:
        return accepts_storage_dtype(dtype, _METAL_DTYPES)

    def set_value(self, index: int, value: Any) -> None:
        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        storage = self._require_storage()
        normalized_index = self._validate_index(index)
        normalized_value = normalize_storage_value(
            self._dtype, value, name="Metal value"
        )
        staged = storage.clone()
        staged[normalized_index] = normalized_value
        self._runtime.synchronize()
        self._storage = staged
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> Metal:
        allocated_dtype = _validate_metal_dtype(self._dtype if dtype is None else dtype)
        materialized = list(values)
        zero = storage_zero(allocated_dtype)
        normalized = normalize_storage_values(
            allocated_dtype,
            [zero if value is None else value for value in materialized],
            "Metal value",
        )
        return Metal._from_values(normalized, mutable=mutable, dtype=allocated_dtype)

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> Metal:
        return Metal(
            size,
            mutable=mutable,
            dtype=self._dtype if dtype is None else dtype,
            empty=empty,
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

        destination_indices = [
            self._validate_index(
                scatter_onto.offset + normalized_offset + mapping.index(logical_index)
            )
            for logical_index in range(to_scatter.size())
        ]
        values = normalize_storage_values(
            self._dtype,
            [to_scatter[index] for index in range(to_scatter.size())],
            "Metal value",
        )
        if not values:
            return

        # Stage every write away from the visible carrier. Validation and
        # source reads above have already completed; a device failure below
        # therefore leaves both the original storage and version unchanged.
        staged = self._require_storage().clone()
        for destination_index, value in zip(destination_indices, values, strict=True):
            staged[destination_index] = value
        self._runtime.synchronize()
        self._storage = staged
        self._increment_version()

    def _dispatch_op(self, operation_name: str) -> Any:
        from ..generic.as_strided_ops import GenericAsStridedOperation
        from ..shared_ops import (
            BroadcastOperation,
            GenericViewOperation,
            PermuteOperation,
            RearrangeOperation,
            ReshapeOperation,
            SqueezeOperation,
            UnsqueezeOperation,
        )

        structural_operations = {
            "as_strided": GenericAsStridedOperation,
            "broadcast_to": BroadcastOperation,
            "permute": PermuteOperation,
            "rearrange": RearrangeOperation,
            "reshape": ReshapeOperation,
            "squeeze": SqueezeOperation,
            "unsqueeze": UnsqueezeOperation,
            "view": GenericViewOperation,
        }
        from .pointwise_ops import metal_pointwise_operation

        pointwise_operations = {
            "abs",
            "add",
            "ceil",
            "clamp",
            "cos",
            "div",
            "elementwise_mul",
            "elu",
            "eq",
            "erf",
            "exp",
            "exp2",
            "floor",
            "gelu",
            "le",
            "leaky_relu",
            "log",
            "log2",
            "logical_not",
            "lt",
            "maximum",
            "minimum",
            "mul",
            "ne",
            "neg",
            "pow",
            "recip",
            "relu",
            "rem",
            "round",
            "rsqrt",
            "select",
            "sigmoid",
            "sign",
            "silu",
            "sin",
            "softplus",
            "sqrt",
            "sub",
            "tanh",
        }
        if operation_name in pointwise_operations:
            return metal_pointwise_operation(operation_name)
        from .reduction_ops import metal_reduction_operation

        if operation_name in {
            "argmax",
            "argmin",
            "cumsum",
            "reduce_max",
            "reduce_min",
            "reduce_prod",
            "reduce_sum",
        }:
            return metal_reduction_operation(operation_name)
        from .indexing_ops import metal_indexing_operation

        if operation_name in {
            "_sort_indices",
            "_sort_values",
            "_topk_indices",
            "_topk_values",
            "gather",
            "scatter",
            "scatter_add",
        }:
            return metal_indexing_operation(operation_name)
        from .contraction_ops import metal_contraction_operation

        if operation_name in {"conv_general", "matmul"}:
            return metal_contraction_operation(operation_name)
        try:
            operation_type = structural_operations[operation_name]
        except KeyError as exc:
            raise NotImplementedError(
                f"Metal carrier does not support operation '{operation_name}'"
            ) from exc
        return operation_type()

    def _release(self) -> None:
        self._storage = None
        self._size = 0


__all__ = [
    "Metal",
]
