"""Composite data storage with explicit promoted and evicted tiers."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from typing import Any, cast

from ..base import Data
from ..dtype import DataType


def _new_like_with_dtype(
    prototype: Data,
    values: Iterable[Any],
    *,
    mutable: bool,
    dtype: DataType,
) -> Data:
    if dtype is prototype.type():
        return prototype.new_like(values, mutable=mutable)
    return cast(Any, prototype).new_like(values, mutable=mutable, dtype=dtype)


@contextmanager
def _owner_access(data: Data, token: int):
    data._begin_owner_access(token)
    try:
        yield
    finally:
        data._end_owner_access(token)


class Evictable(Data):
    """Compose primary and secondary data into an evictable memory hierarchy.

    The constructor takes exclusive ownership of both tiers. Retained aliases
    remain readable but reject mutation and release; mutation must go through
    this Evictable object. The primary tier is active after construction and is
    the only tier that permits value access or tensor operations. ``evict``
    moves the complete physical storage into the secondary tier, while
    ``promote`` restores it to fresh primary-class storage. Both transitions
    bypass move autograd.

    Args:
        primary: Live data containing the initial values and defining normal
            tensor operation dispatch. Ownership transfers to the result.
        secondary: Distinct live, mutable data receiving evicted values. It
            must have enough storage or support allocation from an empty state,
            and ownership transfers to the result.

    Returns:
        Promoted Evictable data backed by the supplied hierarchy.

    Examples:
        >>> from neotorch import CPU, DataType, Evictable, FileBacked
        >>> data = Evictable(
        ...     CPU(2, dtype=DataType.Float32),
        ...     FileBacked(dtype=DataType.Float32),
        ... )
        >>> data.is_evicted()
        False
        >>> data.is_mutable()
        True
        >>> data.primary.is_owned()
        True
        >>> data.primary.is_mutable()
        False
    """

    def __init__(self, primary: Data, secondary: Data) -> None:
        super().__init__()
        if not isinstance(primary, Data):
            raise TypeError("primary must be a Data instance")
        if not isinstance(secondary, Data):
            raise TypeError("secondary must be a Data instance")
        if primary is secondary:
            raise ValueError("primary and secondary must be distinct data objects")
        if primary.is_released():
            raise RuntimeError("primary data is released")
        if secondary.is_released():
            raise RuntimeError("secondary data is released")
        if primary.is_owned():
            raise RuntimeError("primary data is already owned by another data object")
        if secondary.is_owned():
            raise RuntimeError("secondary data is already owned by another data object")
        if primary.type() is not secondary.type():
            raise TypeError("primary and secondary dtypes must match")
        if not secondary.is_mutable():
            raise RuntimeError("secondary data must be mutable")

        size = primary.size()
        if size <= 0:
            raise ValueError("primary data must contain at least one element")
        from ..move import dispatch_move

        # Validate both directions now; resolve them again for every transition
        # so later registry overrides take effect.
        dispatch_move(type(primary), type(secondary))
        dispatch_move(type(secondary), type(primary))
        self._size = size
        self._dtype = primary.type()
        self._mutable = primary.is_mutable()
        primary_token = primary._claim_ownership()
        try:
            secondary_token = secondary._claim_ownership()
        except Exception:
            primary._relinquish_ownership(primary_token)
            raise
        self._primary = primary
        self._primary_token = primary_token
        self._secondary = secondary
        self._secondary_token = secondary_token
        self._evicted = False

    def __del__(self) -> None:
        for data_name, token_name in (
            ("_primary", "_primary_token"),
            ("_secondary", "_secondary_token"),
        ):
            try:
                data = getattr(self, data_name)
                token = getattr(self, token_name)
                if data.is_owned():
                    data._relinquish_ownership(token)
            except Exception:
                pass

    @property
    def primary(self) -> Data:
        """Return the current primary-tier data object.

        Returns:
            Externally read-only primary data object. It is released while this
            hierarchy is evicted and may be replaced during promotion.

        Examples:
            >>> hierarchy.primary is primary
            True
        """

        return self._primary

    @property
    def secondary(self) -> Data:
        """Return the current secondary-tier data object.

        Returns:
            Externally read-only secondary data object. It may be replaced
            after a promotion and subsequent eviction.

        Examples:
            >>> hierarchy.secondary is secondary
            True
        """

        return self._secondary

    def size(self) -> int:
        if self.is_released():
            return 0
        return self._size

    def type(self) -> DataType:
        return self._dtype

    def _is_mutable(self) -> bool:
        return self._mutable

    def is_evicted(self) -> bool:
        """Return whether values currently reside in secondary storage.

        Returns:
            ``True`` after eviction and before promotion.

        Examples:
            >>> data.evict()
            >>> data.is_evicted()
            True
        """

        return self._evicted

    def _require_promoted(self) -> Data:
        if self.is_released():
            raise RuntimeError("Data is released")
        if self._evicted:
            raise RuntimeError("Evictable data is evicted; call promote() first")
        return self._primary

    def _require_hierarchy_access(self) -> None:
        if self.is_owned() and not self._has_owner_access():
            raise RuntimeError(
                "Evictable data is owned by another data object and cannot be "
                "modified directly"
            )

    def get_value(self, index: int) -> Any:
        return self._require_promoted()[index]

    def set_value(self, index: int, value: Any) -> None:
        if not self.is_mutable():
            raise RuntimeError("Data is not mutable")
        primary = self._require_promoted()
        with _owner_access(primary, self._primary_token):
            primary[index] = value
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> Evictable:
        materialized = list(values)
        result_dtype = self._dtype if dtype is None else dtype
        primary = _new_like_with_dtype(
            self._primary,
            materialized,
            mutable=mutable,
            dtype=result_dtype,
        )
        secondary = self._secondary.empty_like(
            len(materialized),
            mutable=True,
            dtype=result_dtype,
        )
        return Evictable(primary, secondary)

    def empty_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DataType | None = None,
    ) -> Evictable:
        result_dtype = self._dtype if dtype is None else dtype
        primary = self._primary.empty_like(size, mutable=mutable, dtype=result_dtype)
        secondary = self._secondary.empty_like(size, mutable=True, dtype=result_dtype)
        return Evictable(primary, secondary)

    def _wrap_primary(self, primary: Data) -> Evictable:
        if primary is self._primary:
            return self
        if type(primary) is not type(self._primary):
            raise TypeError("operation result does not use the primary data class")
        secondary = self._secondary.empty_like(
            primary.size(),
            mutable=True,
            dtype=primary.type(),
        )
        return Evictable(primary, secondary)

    def _flat_tensor(self, data: Data) -> Any:
        from ...core.layout import Layout, Shape, Stride
        from ...core.tensor import Tensor

        return Tensor(data, 0, Layout(Shape(self._size), Stride(1)))

    def _fresh_destination(
        self, prototype: Data, prototype_token: int
    ) -> tuple[Data, int]:
        if not prototype.is_released() and prototype.size() >= self._size:
            return prototype, prototype_token
        destination = prototype.empty_like(
            self._size,
            mutable=True,
            dtype=self._dtype,
        )
        destination_token = destination._claim_ownership()
        if not prototype.is_released():
            with _owner_access(prototype, prototype_token):
                prototype.release()
        prototype._relinquish_ownership(prototype_token)
        return destination, destination_token

    def evict(self) -> None:
        """Move promoted values into secondary storage without autograd.

        Repeated calls while already evicted are no-ops. The move operation's
        ``_forward`` implementation is called directly, so the transition does
        not add an operation to any tensor graph.

        Returns:
            ``None``.

        Examples:
            >>> data.evict()
            >>> data.is_evicted()
            True
        """

        self._require_hierarchy_access()
        if self.is_released():
            raise RuntimeError("Data is released")
        if self._evicted:
            return
        from ..move import dispatch_move

        destination, destination_token = self._fresh_destination(
            self._secondary, self._secondary_token
        )
        operation = dispatch_move(type(self._primary), type(destination))
        with ExitStack() as stack:
            stack.enter_context(_owner_access(self._primary, self._primary_token))
            stack.enter_context(_owner_access(destination, destination_token))
            moved = operation()._forward(self._flat_tensor(self._primary), destination)
        if moved.data is not destination:
            raise RuntimeError("move operation did not return its destination data")
        self._secondary = moved.data
        self._secondary_token = destination_token
        self._evicted = True

    def promote(self) -> None:
        """Move evicted values back into fresh primary-class storage.

        Repeated calls while already promoted are no-ops. The transition calls
        move ``_forward`` directly and does not create an autograd node.

        Returns:
            ``None``.

        Examples:
            >>> data.evict()
            >>> data.promote()
            >>> data.is_evicted()
            False
        """

        self._require_hierarchy_access()
        if self.is_released():
            raise RuntimeError("Data is released")
        if not self._evicted:
            return
        from ..move import dispatch_move

        destination, destination_token = self._fresh_destination(
            self._primary, self._primary_token
        )
        operation = dispatch_move(type(self._secondary), type(destination))
        replaced_primary: tuple[Data, int] | None = None
        with ExitStack() as stack:
            stack.enter_context(_owner_access(self._secondary, self._secondary_token))
            stack.enter_context(_owner_access(destination, destination_token))
            moved = operation()._forward(
                self._flat_tensor(self._secondary), destination
            )
            if moved.data is not destination:
                raise RuntimeError("move operation did not return its destination data")
            primary = moved.data
            primary_token = destination_token
            if not self._mutable:
                immutable = primary.new_like(
                    (primary[index] for index in range(self._size)),
                    mutable=False,
                )
                primary.release()
                replaced_primary = (primary, primary_token)
                primary = immutable
                primary_token = primary._claim_ownership()
        if replaced_primary is not None:
            replaced_primary[0]._relinquish_ownership(replaced_primary[1])
        self._primary = primary
        self._primary_token = primary_token
        self._evicted = False

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        from ...core.tensor import Tensor

        if not self.is_mutable():
            raise RuntimeError("Data is not mutable")
        primary = self._require_promoted()
        if not isinstance(scatter_onto, Tensor) or scatter_onto.data is not self:
            raise ValueError("scatter_onto must be backed by this data object")

        def lower(tensor: Any, name: str) -> Any:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if isinstance(tensor.data, Evictable):
                data = tensor.data._require_promoted()
                return Tensor(data, tensor.offset, tensor.layout)
            return tensor

        lowered_source = lower(to_scatter, "to_scatter")
        lowered_destination = Tensor(primary, scatter_onto.offset, scatter_onto.layout)
        with _owner_access(primary, self._primary_token):
            primary.scatter(
                lowered_source,
                lowered_destination,
                mapping,
                mapping_offset,
            )
        self._increment_version()

    def dlpack_info(self) -> dict[str, int]:
        raise BufferError("DLPack is not supported for Evictable data")

    def dispatch_op(self, operation_name: str) -> Any:
        """Create an adapter around the promoted backend's operation.

        Args:
            operation_name: Registered operation name such as ``"relu"`` or
                ``"matmul"``.

        Returns:
            Fresh EvictableOperation owning a fresh primary operation.

        Examples:
            >>> operation = data.dispatch_op("relu")
            >>> type(operation).__name__
            'EvictableOperation'
        """

        from .ops import EvictableOperation

        primary = self._require_promoted()
        return EvictableOperation(primary.dispatch_op(operation_name))

    def _release(self) -> None:
        if not self._primary.is_released():
            with _owner_access(self._primary, self._primary_token):
                self._primary.release()
        if not self._secondary.is_released():
            with _owner_access(self._secondary, self._secondary_token):
                self._secondary.release()


__all__ = ["Evictable"]
