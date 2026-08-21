"""Cross-carrier-class move operations and their dispatch registry."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, ClassVar, cast

from ..base import Carrier
from ..cpu import CPU
from ..dtype import DType, storage_zero
from ..file_backed.carrier import FileBacked
from ..metal import Metal
from ..operation_helpers import (
    Operation,
    _canonical_layout_for_shape,
    _detached_tensor_like,
    _logical_values,
    _physical_values_for_layout,
    _require_live_tensor,
    _require_same_shape,
    execute_lowered_operation,
)

_move = import_module("strideweave._move")
_copy_memory_to_file = cast(Callable[[str, int, int], None], _move.copy_memory_to_file)
_copy_file_to_memory = cast(
    Callable[[str, int, int, int], None], _move.copy_file_to_memory
)


class MoveOperation(Operation):
    """Base class for moving a tensor's values into another carrier instance.

    Forward validates the move, fills the destination carrier object through the
    ``_copy`` hook, and releases the source carrier. The destination dtype must
    match the tensor dtype. Concrete subclasses implement ``_copy`` and may
    pin ``source_class``/``destination_class`` to the carrier pair they
    support. Backward moves the gradient back into the source carrier, so
    gradients flow across carrier boundaries.

    Backward materializes the gradient in the *released* source carrier
    through ``new_like``, so every carrier must keep ``new_like`` usable
    after ``release()`` (it constructs fresh storage and reads nothing from
    the released instance).
    """

    source_class: ClassVar[type | None] = None
    destination_class: ClassVar[type | None] = None

    def _forward(self, tensor: Any, destination: Any) -> Any:
        from ...tensor import Tensor

        tensor = _require_live_tensor(tensor, "tensor")
        if tensor.carrier.is_released():
            raise RuntimeError("tensor carrier is released")
        if tensor.carrier.is_owned() and not tensor.carrier._has_owner_access():
            raise RuntimeError(
                "tensor carrier is owned by another carrier object and cannot be moved "
                "directly"
            )
        if not isinstance(destination, Carrier):
            raise TypeError("destination must be a Carrier instance")
        if destination is tensor.carrier:
            raise ValueError("destination must not be the tensor's own carrier")
        if destination.is_released():
            raise RuntimeError("destination carrier is released")
        if not destination.is_mutable():
            raise RuntimeError("destination carrier must be mutable")
        if destination.dtype() is not tensor.carrier.dtype():
            raise TypeError("destination dtype must match the tensor dtype")
        if (
            self.source_class is not None
            and type(tensor.carrier) is not self.source_class
        ):
            raise TypeError(
                f"{type(self).__name__} requires a {self.source_class.__name__} source"
            )
        if (
            self.destination_class is not None
            and type(destination) is not self.destination_class
        ):
            raise TypeError(
                f"{type(self).__name__} requires a "
                f"{self.destination_class.__name__} destination"
            )

        required_size = tensor.layout._cache.cosize
        allocate = getattr(destination, "_allocate", None)
        if destination.size() == 0 and allocate is not None:
            allocate(required_size)
        if destination.size() < required_size:
            raise ValueError("destination carrier is too small for the tensor layout")

        output = Tensor(destination, 0, tensor.layout)
        self._copy(tensor, destination, output, required_size)

        tensor.carrier.release()
        return output

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        raise NotImplementedError

    def backward(self, gradient: Any) -> tuple[Any]:
        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)
        if not gradient.layout.is_injective:
            raise ValueError("Move backward requires an injective gradient layout")
        return (_detached_tensor_like(tensor, _logical_values(gradient)),)


class ElementwiseMoveOperation(MoveOperation):
    """Move between any carrier pair by copying logical values one by one.

    Layout holes keep the destination's prior values. Used as the dispatch
    fallback for carrier pairs without a registered move operation.
    """

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        for i in range(tensor.size()):
            output[i] = tensor[i]


class CpuToFileBackedMoveOperation(MoveOperation):
    """Move a CPU tensor into FileBacked carrier with a native bulk byte copy.

    Copies the tensor's whole physical byte span, including layout holes,
    into the destination file in one native write.
    """

    source_class: ClassVar[type | None] = CPU
    destination_class: ClassVar[type | None] = FileBacked

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        itemsize = destination._itemsize
        _copy_memory_to_file(
            str(destination.path),
            tensor.carrier.pointer() + tensor.offset * itemsize,
            element_count * itemsize,
        )


class FileBackedToCpuMoveOperation(MoveOperation):
    """Move a FileBacked tensor into CPU carrier with a native bulk byte copy.

    Copies the tensor's whole physical byte span, including layout holes,
    from the source file into the destination memory in one native read.
    """

    source_class: ClassVar[type | None] = FileBacked
    destination_class: ClassVar[type | None] = CPU

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        itemsize = tensor.carrier._itemsize
        _copy_file_to_memory(
            str(tensor.carrier.path),
            tensor.offset * itemsize,
            destination.pointer(),
            element_count * itemsize,
        )


def _metal_itemsize(dtype: DType) -> int:
    if dtype is DType.Bool:
        return 1
    if dtype is DType.Float32 or dtype is DType.Int32:
        return 4
    raise TypeError(f"Metal movement does not support DType.{dtype.name}")


def _metal_torch_dtype(runtime: Any, dtype: DType) -> Any:
    if dtype is DType.Float32:
        return runtime.torch.float32
    if dtype is DType.Int32:
        return runtime.torch.int32
    if dtype is DType.Bool:
        return runtime.torch.bool
    raise TypeError(f"Metal movement does not support DType.{dtype.name}")


class _MetalMoveOperation(MoveOperation):
    """Shared exact-pair backward path for moves that cross Metal storage."""

    def _forward(self, tensor: Any, destination: Any) -> Any:
        output = super()._forward(tensor, destination)
        self.ctx["destination_carrier"] = destination
        return output

    def backward(self, gradient: Any) -> tuple[Any]:
        from ...tensor import Tensor

        (tensor,) = self.inputs()
        gradient = _require_live_tensor(gradient, "gradient")
        _require_same_shape(tensor, gradient)
        if not gradient.layout.is_injective:
            raise ValueError("Move backward requires an injective gradient layout")
        if self.source_class is None or self.destination_class is None:
            raise RuntimeError("Metal move operation is missing its exact carrier pair")

        inverse_class = _MOVE_OPERATIONS.get(
            (self.destination_class, self.source_class)
        )
        if inverse_class is None:
            raise RuntimeError(
                "Metal move backward requires an exact registered inverse move from "
                f"{self.destination_class.__name__} to {self.source_class.__name__}"
            )

        destination_prototype = self.ctx["destination_carrier"]
        if type(destination_prototype) is not self.destination_class:
            raise RuntimeError("Metal move destination carrier identity changed")

        target_layout = tensor.layout
        if not target_layout.is_injective:
            target_layout = _canonical_layout_for_shape(target_layout.shape)
        physical_values = _physical_values_for_layout(
            target_layout,
            _logical_values(gradient),
            storage_zero(tensor.carrier.dtype()),
        )

        staging_carrier: Any | None = None
        source_destination: Any | None = None
        try:
            # The caller-owned cotangent remains live: only this fresh staging
            # carrier is consumed by the registered inverse move.
            new_staging_carrier = destination_prototype.new_like(
                physical_values,
                mutable=True,
                dtype=tensor.carrier.dtype(),
            )
            staging_carrier = new_staging_carrier
            new_source_destination = tensor.carrier.allocate_like(
                target_layout.cosize,
                mutable=True,
                dtype=tensor.carrier.dtype(),
            )
            source_destination = new_source_destination
            staging_tensor = Tensor(new_staging_carrier, 0, target_layout)
            moved = execute_lowered_operation(
                inverse_class(), staging_tensor, new_source_destination
            )
        except Exception:
            if staging_carrier is not None and not staging_carrier.is_released():
                staging_carrier.release()
            if source_destination is not None and not source_destination.is_released():
                source_destination.release()
            raise
        return (moved,)


class CpuToMetalMoveOperation(_MetalMoveOperation):
    """Bulk-copy one exact CPU physical span into transactional Metal storage."""

    source_class: ClassVar[type | None] = CPU
    destination_class: ClassVar[type | None] = Metal

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        if element_count == 0:
            return
        itemsize = _metal_itemsize(tensor.carrier.dtype())
        raw_bytes = bytearray(
            ctypes.string_at(
                tensor.carrier.pointer() + tensor.offset * itemsize,
                element_count * itemsize,
            )
        )
        runtime = destination._runtime
        host = runtime.torch.frombuffer(
            raw_bytes,
            dtype=_metal_torch_dtype(runtime, tensor.carrier.dtype()),
            count=element_count,
        )
        device = host.to(device="mps")
        staged = destination._require_storage().clone()
        staged[:element_count].copy_(device)
        runtime.synchronize()
        destination._storage = staged


class MetalToCpuMoveOperation(_MetalMoveOperation):
    """Bulk-copy one exact Metal physical span into CPU storage."""

    source_class: ClassVar[type | None] = Metal
    destination_class: ClassVar[type | None] = CPU

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        runtime = tensor.carrier._runtime
        if element_count == 0:
            runtime.synchronize()
            return
        runtime.synchronize()
        temporary = (
            tensor.carrier._require_storage()[
                tensor.offset : tensor.offset + element_count
            ]
            .detach()
            .to(device="cpu")
            .contiguous()
        )
        runtime.synchronize()
        ctypes.memmove(
            destination.pointer(),
            temporary.data_ptr(),
            element_count * _metal_itemsize(tensor.carrier.dtype()),
        )


class MetalToMetalMoveOperation(_MetalMoveOperation):
    """Bulk-copy one exact Metal physical span into fresh visible MPS storage."""

    source_class: ClassVar[type | None] = Metal
    destination_class: ClassVar[type | None] = Metal

    def _copy(
        self, tensor: Any, destination: Any, output: Any, element_count: int
    ) -> None:
        if element_count == 0:
            tensor.carrier._runtime.synchronize()
            if destination._runtime is not tensor.carrier._runtime:
                destination._runtime.synchronize()
            return
        staged = destination._require_storage().clone()
        staged[:element_count].copy_(
            tensor.carrier._require_storage()[
                tensor.offset : tensor.offset + element_count
            ]
        )
        tensor.carrier._runtime.synchronize()
        if destination._runtime is not tensor.carrier._runtime:
            destination._runtime.synchronize()
        destination._storage = staged


_MOVE_OPERATIONS: dict[tuple[type, type], type[MoveOperation]] = {}


def register_move_operation(
    source_class: type,
    destination_class: type,
    operation_class: type[MoveOperation],
) -> None:
    """Register a concrete move operation for a carrier pair.

    Args:
        source_class: Carrier class of the tensor being moved.
        destination_class: Carrier class of the destination instance.
        operation_class: ``MoveOperation`` subclass handling the pair.

    Returns:
        ``None``.

    Examples:
        >>> from strideweave import Generic
        >>> from strideweave.carriers.move import (
        ...     ElementwiseMoveOperation,
        ...     dispatch_move,
        ...     register_move_operation,
        ... )
        >>> register_move_operation(Generic, Generic, ElementwiseMoveOperation)
        >>> dispatch_move(Generic, Generic) is ElementwiseMoveOperation
        True
    """

    if not (isinstance(source_class, type) and issubclass(source_class, Carrier)):
        raise TypeError("source_class must be a Carrier subclass")
    if not (
        isinstance(destination_class, type) and issubclass(destination_class, Carrier)
    ):
        raise TypeError("destination_class must be a Carrier subclass")
    if not (
        isinstance(operation_class, type) and issubclass(operation_class, MoveOperation)
    ):
        raise TypeError("operation_class must be a MoveOperation subclass")
    key = (source_class, destination_class)
    if key in _MOVE_OPERATIONS:
        raise ValueError(
            "a move operation is already registered for "
            f"({source_class.__name__}, {destination_class.__name__})"
        )
    _MOVE_OPERATIONS[key] = operation_class


def unregister_move_operation(
    source_class: type, destination_class: type
) -> type[MoveOperation]:
    """Remove the registered move operation for a carrier pair.

    Args:
        source_class: Carrier class of the tensor being moved.
        destination_class: Carrier class of the destination instance.

    Returns:
        The ``MoveOperation`` subclass that was registered for the pair.

    Examples:
        >>> from strideweave import Generic
        >>> from strideweave.carriers.move import (
        ...     ElementwiseMoveOperation,
        ...     register_move_operation,
        ...     unregister_move_operation,
        ... )
        >>> register_move_operation(Generic, Generic, ElementwiseMoveOperation)
        >>> unregister_move_operation(Generic, Generic) is ElementwiseMoveOperation
        True
    """

    try:
        return _MOVE_OPERATIONS.pop((source_class, destination_class))
    except KeyError:
        raise KeyError(
            "no move operation is registered for "
            f"({source_class.__name__}, {destination_class.__name__})"
        ) from None


@contextmanager
def registered_move_operation(
    source_class: type,
    destination_class: type,
    operation_class: type[MoveOperation],
) -> Iterator[type[MoveOperation]]:
    """Register a move operation for the duration of a ``with`` block.

    The operation is registered on entry and unregistered on exit, including
    when the block raises. Useful for tests and temporary carrier overrides.

    Args:
        source_class: Carrier class of the tensor being moved.
        destination_class: Carrier class of the destination instance.
        operation_class: ``MoveOperation`` subclass handling the pair.

    Returns:
        Context manager yielding ``operation_class``.

    Examples:
        >>> from strideweave import Generic
        >>> from strideweave.carriers.move import (
        ...     ElementwiseMoveOperation,
        ...     dispatch_move,
        ...     registered_move_operation,
        ... )
        >>> with registered_move_operation(
        ...     Generic, Generic, ElementwiseMoveOperation
        ... ):
        ...     selected = dispatch_move(Generic, Generic)
        >>> selected is ElementwiseMoveOperation
        True
    """

    register_move_operation(source_class, destination_class, operation_class)
    try:
        yield operation_class
    finally:
        unregister_move_operation(source_class, destination_class)


def dispatch_move(source_class: type, destination_class: type) -> type[MoveOperation]:
    """Return the move operation class for a carrier pair.

    Dispatch is exact-class: subclasses of a registered source or destination
    class do not inherit the registration and fall back to
    ``ElementwiseMoveOperation`` unless registered explicitly.

    Args:
        source_class: Carrier class of the tensor being moved.
        destination_class: Carrier class of the destination instance.

    Returns:
        The registered ``MoveOperation`` subclass for the exact pair, or
        ``ElementwiseMoveOperation`` when no operation is registered.

    Examples:
        >>> from strideweave import CPU, FileBacked
        >>> from strideweave.carriers.move import dispatch_move
        >>> dispatch_move(CPU, FileBacked)  # doctest: +ELLIPSIS
        <class '...CpuToFileBackedMoveOperation'>
    """

    return _MOVE_OPERATIONS.get(
        (source_class, destination_class), ElementwiseMoveOperation
    )


def _has_registered_move_operation(source_class: type, destination_class: type) -> bool:
    return (source_class, destination_class) in _MOVE_OPERATIONS


register_move_operation(CPU, FileBacked, CpuToFileBackedMoveOperation)
register_move_operation(FileBacked, CPU, FileBackedToCpuMoveOperation)
register_move_operation(CPU, Metal, CpuToMetalMoveOperation)
register_move_operation(Metal, CPU, MetalToCpuMoveOperation)
register_move_operation(Metal, Metal, MetalToMetalMoveOperation)


__all__ = [
    "CpuToFileBackedMoveOperation",
    "CpuToMetalMoveOperation",
    "ElementwiseMoveOperation",
    "FileBackedToCpuMoveOperation",
    "MetalToCpuMoveOperation",
    "MetalToMetalMoveOperation",
    "MoveOperation",
    "dispatch_move",
    "register_move_operation",
    "registered_move_operation",
    "unregister_move_operation",
]
