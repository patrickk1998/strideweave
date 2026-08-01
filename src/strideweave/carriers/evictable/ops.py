"""Autograd adapter for operations on promoted Evictable storage."""

from __future__ import annotations

from typing import Any

from ..dtype import SimpleDType
from ..operation_helpers import Operation, execute_lowered_operation
from ..operation_policy import (
    OperationPlan,
    registered_operations,
    resolve_operation_plan,
)
from .carrier import Evictable


class EvictableOperation(Operation):
    """Delegate one autograd operation to a promoted primary carrier.

    One adapter owns one stateful primary operation. Forward lowers Evictable
    tensors to temporary primary-backed tensors and uses the primary
    operation's lowered execution route. Backward refreshes only those lowered
    inputs before calling the same primary operation's ``backward``, preserving
    its context and native forward state.

    Args:
        primary_operation: Fresh operation returned by the primary carrier.

    Returns:
        Single-use forward adapter that remains reusable for backward.

    Examples:
        >>> from strideweave import CPU, EvictableOperation
        >>> adapter = EvictableOperation(CPU(1).dispatch_op("relu"))
        >>> adapter.primary_operation is not None
        True
    """

    def __init__(self, primary_operation: Any) -> None:
        super().__init__()
        if not isinstance(primary_operation, Operation):
            raise TypeError("primary_operation must be an Operation")
        self._primary_operation = primary_operation
        self._forward_complete = False
        self._output_carrier: Evictable | None = None

    @property
    def primary_operation(self) -> Any:
        """Return the operation owned by the primary carrier.

        Returns:
            Stateful CPU, Generic, or shared operation used for computation.

        Examples:
            >>> adapter.primary_operation
            <...Operation object ...>
        """

        return self._primary_operation

    def forward(self, *inputs: Any) -> Any:
        """Run one delegated forward pass and build its outer autograd node.

        Args:
            *inputs: Evictable-backed tensor arguments and non-tensor operation
                arguments accepted by the primary operation.

        Returns:
            Evictable-backed tensor produced by the primary operation.

        Examples:
            >>> result = adapter.forward(tensor)
        """

        if self._forward_complete:
            raise RuntimeError("EvictableOperation forward may only be called once")
        result = super().forward(*inputs)
        self._forward_complete = True
        return result

    @staticmethod
    def _lower_tensor(tensor: Any) -> Any:
        from ...core.tensor import Tensor

        if not isinstance(tensor, Tensor):
            raise TypeError("operation tensor inputs must be Tensors")
        if not isinstance(tensor.carrier, Evictable):
            raise TypeError("EvictableOperation requires Evictable tensor inputs")
        # A pure layout view may pass the outer multi-subtensor preflight, but
        # this adapter still only knows how to lower one primary plane. Refuse
        # before constructing a one-subtensor Tensor so no plane is silently
        # discarded at the composite boundary.
        tensor._require_single_subtensor("Evictable operation lowering")
        primary = tensor.carrier._require_promoted()
        return Tensor(primary, tensor.offset, tensor.layout)

    @staticmethod
    def _require_compatible(inputs: tuple[Any, ...]) -> Evictable:
        from ...core.tensor import Tensor

        carrier_inputs = [
            value.carrier
            for value in inputs
            if isinstance(value, Tensor) and isinstance(value.carrier, Evictable)
        ]
        if not carrier_inputs:
            raise TypeError("EvictableOperation requires a tensor input")
        first = carrier_inputs[0]
        primary_class = type(first._require_promoted())
        secondary_class = type(first.secondary)
        for carrier in carrier_inputs[1:]:
            if type(carrier._require_promoted()) is not primary_class:
                raise TypeError("Evictable primary carriers must match")
            if type(carrier.secondary) is not secondary_class:
                raise TypeError("Evictable secondary carriers must match")
        return first

    @staticmethod
    def _tensor_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        from ...core.tensor import Tensor

        return tuple(value for value in inputs if isinstance(value, Tensor))

    def _planned_operation(self, inputs: tuple[Any, ...]) -> OperationPlan | None:
        """Resolve the logical plan this call runs, or ``None`` for legacy storage.

        The plan is the one the central policy resolves from this operation's
        name and the *outer* operands, which is what the hierarchy must be able
        to execute and keep.

        Exactly two things are outside simple-dtype planning and keep their
        documented behavior: an operation the policy registry does not describe,
        and legacy opaque operand storage. Everything else is planned, so an
        operand list the operation's own shape does not accept — the wrong
        count, a tensor where a weak scalar belongs, a scalar where a tensor
        belongs — is refused by the central resolver here rather than passed
        down to be diagnosed by whichever primary operation happens to receive
        it.
        """
        from ...core.tensor import Tensor

        operation = self._primary_operation._operation_name
        if operation is None:
            return None
        spec = next(
            (
                candidate
                for candidate in registered_operations()
                if candidate.name == operation
            ),
            None,
        )
        if spec is None:
            return None
        policy_inputs = inputs
        if spec.dtype_operand_positions is not None:
            policy_inputs = tuple(
                inputs[position]
                for position in spec.dtype_operand_positions
                if position < len(inputs)
            )
        operands: list[Any] = []
        for value in policy_inputs:
            if not isinstance(value, Tensor):
                operands.append(value)
                continue
            dtype = value.carrier.dtype()
            if not isinstance(dtype, SimpleDType):
                return None
            operands.append(dtype)
        return resolve_operation_plan(operation, *operands)

    def _forward(self, *inputs: Any) -> Any:
        from ...core.tensor import Tensor

        hierarchy = self._require_compatible(inputs)
        plan = self._planned_operation(inputs)
        if plan is not None:
            # Before any lowering, nested allocation, or kernel work: a plan
            # this hierarchy does not advertise is refused once, here, rather
            # than after the primary has produced a result no hierarchy could
            # hold.
            hierarchy.require_operation_plan(plan)
        lowered_arguments = tuple(
            self._lower_tensor(value) if isinstance(value, Tensor) else value
            for value in inputs
        )
        lowered_tensors = self._tensor_inputs(lowered_arguments)
        self._primary_operation.store_inputs(*lowered_tensors)
        primary_result = execute_lowered_operation(
            self._primary_operation, *lowered_arguments
        )
        wrapped_carrier = hierarchy._wrap_primary(primary_result.carrier)
        self._output_carrier = wrapped_carrier
        return Tensor(wrapped_carrier, primary_result.offset, primary_result.layout)

    def backward(self, gradient: Any) -> tuple[Any, ...]:
        """Delegate backward through the retained primary operation state.

        Args:
            gradient: Incoming Evictable or compatible primary-backed tensor.

        Returns:
            Ordered tuple of Evictable gradients for the original tensor
            inputs, preserving ``None`` entries from the primary operation.

        Examples:
            >>> (input_gradient,) = adapter.backward(output_gradient)
        """

        from ...core.tensor import Tensor

        if not self._forward_complete or self._output_carrier is None:
            raise RuntimeError("EvictableOperation backward requires forward first")
        original_inputs = self.inputs()
        lowered_inputs = tuple(self._lower_tensor(value) for value in original_inputs)
        self._primary_operation.store_inputs(*lowered_inputs)

        if not isinstance(gradient, Tensor):
            raise TypeError("gradient must be a Tensor")
        expected_primary_class = type(self._output_carrier.primary)
        if isinstance(gradient.carrier, Evictable):
            lowered_gradient = self._lower_tensor(gradient)
        else:
            if type(gradient.carrier) is not expected_primary_class:
                raise TypeError("gradient carrier must match the primary carrier class")
            lowered_gradient = gradient

        primary_gradients = tuple(self._primary_operation.backward(lowered_gradient))
        if len(primary_gradients) != len(original_inputs):
            raise ValueError("primary operation returned wrong number of gradients")

        wrapped_gradients = []
        for original, primary_gradient in zip(
            original_inputs, primary_gradients, strict=True
        ):
            if primary_gradient is None:
                wrapped_gradients.append(None)
                continue
            if not isinstance(primary_gradient, Tensor):
                raise TypeError("primary operation gradients must be Tensors or None")
            wrapped_carrier = original.carrier._wrap_primary(primary_gradient.carrier)
            wrapped_gradients.append(
                Tensor(
                    wrapped_carrier,
                    primary_gradient.offset,
                    primary_gradient.layout,
                )
            )
        return tuple(wrapped_gradients)


__all__ = ["EvictableOperation"]
