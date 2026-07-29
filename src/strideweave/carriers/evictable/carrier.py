"""Composite carrier storage with explicit promoted and evicted tiers."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from typing import Any, cast, final

from ..base import Carrier, reject_carrier_subclass
from ..dtype import DType
from ..operation_capability import DependentCarrier, OperationCapability
from ..operation_helpers import execute_lowered_operation


def _new_like_with_dtype(
    prototype: Carrier,
    values: Iterable[Any],
    *,
    mutable: bool,
    dtype: DType,
) -> Carrier:
    if dtype is prototype.dtype():
        return prototype.new_like(values, mutable=mutable)
    return cast(Any, prototype).new_like(values, mutable=mutable, dtype=dtype)


@contextmanager
def _owner_access(carrier: Carrier, token: int):
    carrier._begin_owner_access(token)
    try:
        yield
    finally:
        carrier._end_owner_access(token)


@final
class Evictable(DependentCarrier):
    """Compose primary and secondary carriers into an evictable memory hierarchy.

    The constructor takes exclusive ownership of both tiers. Retained aliases
    remain readable but reject mutation and release; mutation must go through
    this Evictable object. The primary tier is active after construction and is
    the only tier that permits value access or tensor operations. ``evict``
    moves the complete physical storage into the secondary tier, while
    ``promote`` restores it to fresh primary-class storage. Both transitions
    bypass move autograd.

    What a hierarchy can execute depends on the carriers it was handed, so
    Evictable is a ``DependentCarrier``: at the end of construction it freezes
    the plans its own primary executes and whose results both tiers could
    store. Two hierarchies of this same class therefore advertise different
    plans, and a plan only the primary could hold is refused before any work
    begins rather than producing a result the hierarchy could never evict.

    Evictable is itself the composition pattern a new backend follows, and is a
    closed implementation: extend StrideWeave with a sibling ``Carrier`` that
    owns its tiers rather than with a specialization of this one.

    Args:
        primary: Live carrier containing the initial values and defining normal
            tensor operation dispatch. Ownership transfers to the result.
        secondary: Distinct live, mutable carrier receiving evicted values. It
            must have enough storage or support allocation from an empty state,
            and ownership transfers to the result.

    Returns:
        Promoted Evictable carrier backed by the supplied hierarchy.

    Examples:
        >>> from strideweave import CPU, DType, Evictable, FileBacked
        >>> carrier = Evictable(
        ...     CPU(2, dtype=DType.Float32),
        ...     FileBacked(dtype=DType.Float32),
        ... )
        >>> carrier.is_evicted()
        False
        >>> carrier.is_mutable()
        True
        >>> carrier.primary.is_owned()
        True
        >>> carrier.primary.is_mutable()
        False
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        reject_carrier_subclass("Evictable")

    def __init__(self, primary: Carrier, secondary: Carrier) -> None:
        super().__init__()
        if not isinstance(primary, Carrier):
            raise TypeError("primary must be a Carrier instance")
        if not isinstance(secondary, Carrier):
            raise TypeError("secondary must be a Carrier instance")
        if primary is secondary:
            raise ValueError("primary and secondary must be distinct carriers")
        if primary.is_released():
            raise RuntimeError("primary carrier is released")
        if secondary.is_released():
            raise RuntimeError("secondary carrier is released")
        if primary.is_owned():
            raise RuntimeError("primary carrier is already owned by another carrier")
        if secondary.is_owned():
            raise RuntimeError("secondary carrier is already owned by another carrier")
        if primary.dtype() is not secondary.dtype():
            raise TypeError("primary and secondary dtypes must match")
        if not secondary.is_mutable():
            raise RuntimeError("secondary carrier must be mutable")

        size = primary.size()
        if size <= 0:
            raise ValueError("primary carrier must contain at least one element")
        from ..move import dispatch_move

        # Validate both directions now; resolve them again for every transition
        # so later registry overrides take effect.
        dispatch_move(type(primary), type(secondary))
        dispatch_move(type(secondary), type(primary))
        self._size = size
        self._dtype = primary.dtype()
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
        try:
            # The last step of construction: the tiers are installed and valid,
            # so what this hierarchy executes is now decided. A failure here
            # leaves no usable hierarchy, so the tiers it claimed go back to
            # their previous owners rather than staying locked to an object
            # nobody received.
            self._finalize_dependent_capabilities()
        except Exception:
            secondary._relinquish_ownership(secondary_token)
            primary._relinquish_ownership(primary_token)
            raise

    def __del__(self) -> None:
        for carrier_name, token_name in (
            ("_primary", "_primary_token"),
            ("_secondary", "_secondary_token"),
        ):
            try:
                carrier = getattr(self, carrier_name)
                token = getattr(self, token_name)
                if carrier.is_owned():
                    carrier._relinquish_ownership(token)
            except Exception:
                pass

    @property
    def primary(self) -> Carrier:
        """Return the current primary-tier carrier.

        Returns:
            Externally read-only primary carrier. It is released while this
            hierarchy is evicted and may be replaced during promotion.

        Examples:
            >>> hierarchy.primary is primary
            True
        """

        return self._primary

    @property
    def secondary(self) -> Carrier:
        """Return the current secondary-tier carrier.

        Returns:
            Externally read-only secondary carrier. It may be replaced
            after a promotion and subsequent eviction. Newly created operation
            results keep this tier empty until their first eviction.

        Examples:
            >>> hierarchy.secondary is secondary
            True
        """

        return self._secondary

    def size(self) -> int:
        if self.is_released():
            return 0
        return self._size

    def dtype(self) -> DType:
        return self._dtype

    def _is_mutable(self) -> bool:
        return self._mutable

    def _generate_operation_capabilities(self) -> tuple[OperationCapability, ...]:
        """Return the plans this hierarchy executes and could keep.

        A hierarchy executes on its promoted primary, so its reach starts as
        that instance's own reach — asked of the carrier, so a primary that is
        itself a dependent carrier composes through its snapshot rather than
        through its class. It is then narrowed by one structural condition: the
        hierarchy must be able to store the result in *both* tiers, because a
        result only the primary could hold would be a value it could never
        evict.

        Nothing here rewrites a plan, chooses a promotion, or inspects a carrier
        class: the entries are the primary's own, filtered.
        """
        return tuple(
            entry
            for entry in self._primary.operation_capabilities()
            if self.supports_storage_dtype(entry.output)
        )

    def _supports_storage_dtype(self, dtype: DType) -> bool:
        """Report the dtypes both tiers can store.

        A hierarchy must be able to hold its values in either tier, so it
        supports the intersection: a dtype only the promoted tier can allocate
        could not survive an eviction. The answer is structural and asks each
        tier the same structural question, so residency, ownership, and release
        do not change it.
        """
        return self._primary.supports_storage_dtype(
            dtype
        ) and self._secondary.supports_storage_dtype(dtype)

    def is_evicted(self) -> bool:
        """Return whether values currently reside in secondary storage.

        Returns:
            ``True`` after eviction and before promotion.

        Examples:
            >>> carrier.evict()
            >>> carrier.is_evicted()
            True
        """

        return self._evicted

    def _require_promoted(self) -> Carrier:
        if self.is_released():
            raise RuntimeError("Carrier is released")
        if self._evicted:
            raise RuntimeError("Evictable carrier is evicted; call promote() first")
        return self._primary

    def _require_hierarchy_access(self) -> None:
        if self.is_owned() and not self._has_owner_access():
            raise RuntimeError(
                "Evictable carrier is owned by another carrier and cannot be "
                "modified directly"
            )

    def get_value(self, index: int) -> Any:
        return self._require_promoted()[index]

    def set_value(self, index: int, value: Any) -> None:
        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        primary = self._require_promoted()
        with _owner_access(primary, self._primary_token):
            primary[index] = value
        self._increment_version()

    def new_like(
        self,
        values: Iterable[Any],
        *,
        mutable: bool = True,
        dtype: DType | None = None,
    ) -> Evictable:
        materialized = list(values)
        result_dtype = self._dtype if dtype is None else dtype
        primary = _new_like_with_dtype(
            self._primary,
            materialized,
            mutable=mutable,
            dtype=result_dtype,
        )
        secondary = self._secondary.allocate_like(
            0,
            mutable=True,
            dtype=result_dtype,
        )
        return Evictable(primary, secondary)

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: DType | None = None,
        empty: bool = False,
    ) -> Evictable:
        result_dtype = self._dtype if dtype is None else dtype
        primary = self._primary.allocate_like(
            size, mutable=mutable, dtype=result_dtype, empty=empty
        )
        secondary = self._secondary.allocate_like(
            0, mutable=True, dtype=result_dtype, empty=empty
        )
        return Evictable(primary, secondary)

    def _wrap_primary(self, primary: Carrier) -> Evictable:
        if primary is self._primary:
            return self
        if type(primary) is not type(self._primary):
            raise TypeError("operation result does not use the primary carrier")
        secondary = self._secondary.allocate_like(
            0,
            mutable=True,
            dtype=primary.dtype(),
        )
        return Evictable(primary, secondary)

    def _flat_tensor(self, carrier: Carrier) -> Any:
        from ...core.layout import Layout, Shape, Stride
        from ...core.tensor import Tensor

        return Tensor(carrier, 0, Layout(Shape(self._size), Stride(1)))

    def _prepare_destination(
        self, prototype: Carrier, prototype_token: int
    ) -> tuple[Carrier, int, bool]:
        if not prototype.is_released() and prototype.size() >= self._size:
            return prototype, prototype_token, False
        destination = prototype.allocate_like(
            self._size,
            mutable=True,
            dtype=self._dtype,
        )
        destination_token = destination._claim_ownership()
        return destination, destination_token, True

    @staticmethod
    def _dispose_tier(carrier: Carrier, token: int) -> None:
        try:
            if not carrier.is_released():
                with _owner_access(carrier, token):
                    carrier.release()
        finally:
            if carrier.is_owned():
                carrier._relinquish_ownership(token)

    def evict(self) -> None:
        """Move promoted values into secondary storage without autograd.

        Repeated calls while already evicted are no-ops. The move operation uses
        lowered execution, so the transition does not add an operation to any
        tensor graph.

        Returns:
            ``None``.

        Examples:
            >>> carrier.evict()
            >>> carrier.is_evicted()
            True
        """

        self._require_hierarchy_access()
        if self.is_released():
            raise RuntimeError("Carrier is released")
        if self._evicted:
            return
        from ..move import dispatch_move

        destination, destination_token, replaces_secondary = self._prepare_destination(
            self._secondary, self._secondary_token
        )
        operation = dispatch_move(type(self._primary), type(destination))
        try:
            with ExitStack() as stack:
                stack.enter_context(_owner_access(self._primary, self._primary_token))
                stack.enter_context(_owner_access(destination, destination_token))
                moved = execute_lowered_operation(
                    operation(), self._flat_tensor(self._primary), destination
                )
        except Exception:
            if replaces_secondary:
                self._dispose_tier(destination, destination_token)
            raise
        if moved.carrier is not destination:
            raise RuntimeError("move operation did not return its destination carrier")
        previous_secondary = self._secondary
        previous_secondary_token = self._secondary_token
        self._secondary = moved.carrier
        self._secondary_token = destination_token
        self._evicted = True
        if replaces_secondary:
            self._dispose_tier(previous_secondary, previous_secondary_token)

    def promote(self) -> None:
        """Move evicted values back into fresh primary-class storage.

        Repeated calls while already promoted are no-ops. The transition uses
        lowered move execution and does not create an autograd node.

        Returns:
            ``None``.

        Examples:
            >>> carrier.evict()
            >>> carrier.promote()
            >>> carrier.is_evicted()
            False
        """

        self._require_hierarchy_access()
        if self.is_released():
            raise RuntimeError("Carrier is released")
        if not self._evicted:
            return
        from ..move import dispatch_move

        destination, destination_token, replaces_primary = self._prepare_destination(
            self._primary, self._primary_token
        )
        operation = dispatch_move(type(self._secondary), type(destination))
        try:
            with ExitStack() as stack:
                stack.enter_context(
                    _owner_access(self._secondary, self._secondary_token)
                )
                stack.enter_context(_owner_access(destination, destination_token))
                moved = execute_lowered_operation(
                    operation(), self._flat_tensor(self._secondary), destination
                )
        except Exception:
            if replaces_primary:
                self._dispose_tier(destination, destination_token)
            raise
        if moved.carrier is not destination:
            raise RuntimeError("move operation did not return its destination carrier")
        previous_primary = self._primary
        previous_primary_token = self._primary_token
        self._primary = moved.carrier
        self._primary_token = destination_token
        self._evicted = False
        if replaces_primary:
            self._dispose_tier(previous_primary, previous_primary_token)

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        from ...core.tensor import Tensor

        if not self.is_mutable():
            raise RuntimeError("Carrier is not mutable")
        primary = self._require_promoted()
        if not isinstance(scatter_onto, Tensor) or scatter_onto.carrier is not self:
            raise ValueError("scatter_onto must be backed by this carrier object")
        scatter_onto._require_single_subtensor("scatter")

        def lower(tensor: Any, name: str) -> Any:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            tensor._require_single_subtensor("scatter")
            if isinstance(tensor.carrier, Evictable):
                carrier = tensor.carrier._require_promoted()
                return Tensor(carrier, tensor.offset, tensor.layout)
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
        raise BufferError("DLPack is not supported for Evictable carrier")

    def _dispatch_op(self, operation_name: str) -> Any:
        """Create an adapter around the promoted carrier's operation.

        Args:
            operation_name: Registered operation name such as ``"relu"`` or
                ``"matmul"``.

        Returns:
            Fresh EvictableOperation owning a fresh primary operation.

        Examples:
            >>> operation = carrier.dispatch_op("relu")
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
