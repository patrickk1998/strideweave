"""Framework-owned execution for exact definition-backed carriers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from ..operation_definition import (
    NON_DIFFERENTIABLE,
    ProviderContractError,
    ResolvedInvocation,
    VJPContext,
    _call_vjp,
    _differentiable_indices,
    _operation_definition,
)
from ..tensor import Tensor
from .extension import (
    CarrierDefinition,
    KernelPattern,
    PreparedComposite,
    PreparedKernel,
    ProviderResult,
    Unsupported,
    _definition_for_instance,
    _definition_kernel_patterns,
    _observe_carrier_definition,
    _provider_size,
    _require_provider_completion,
)
from .operation_capability import (
    UnsupportedOperationPlan,
    carrier_operation_capabilities,
    require_capability,
    require_carrier_capability,
)
from .operation_helpers import Operation

if TYPE_CHECKING:
    from .base import Carrier

_is_grad_enabled = cast(
    Callable[[], bool], import_module("strideweave._operation").is_grad_enabled
)


def _invocation_allows_autograd(invocation: ResolvedInvocation) -> bool:
    vjp = invocation.definition.vjp
    if vjp is NON_DIFFERENTIABLE or not callable(vjp):
        return False
    return any(
        isinstance(operand := invocation.operands[index], Tensor)
        and operand.is_differentiable()
        for index in _differentiable_indices(invocation)
    )


def _reject_graph_building_exact_alias(invocation: ResolvedInvocation) -> None:
    if not _is_grad_enabled() or not _invocation_allows_autograd(invocation):
        return
    if any(
        spec.alias_of is not None
        and cast(Tensor, invocation.operands[spec.alias_of]).is_differentiable()
        for spec in invocation.results
    ):
        raise ValueError(
            "definition-backed autograd prohibits exact-object aliased results"
        )


def _stored_values_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return left is right


@dataclass(frozen=True, slots=True)
class _CarrierMutationSnapshot:
    carrier: Carrier
    definition: CarrierDefinition
    values: tuple[object, ...]
    version: int
    is_released: bool
    is_owned: bool
    operand_indices: tuple[int, ...]

    def has_changed(self) -> bool:
        carrier = self.carrier
        if (
            carrier.version != self.version
            or carrier.is_released() is not self.is_released
            or carrier.is_owned() is not self.is_owned
        ):
            return True
        try:
            if _provider_size(carrier, self.definition) != len(self.values):
                return True
            current = tuple(
                self.definition.storage.read(carrier, index)
                for index in range(len(self.values))
            )
        except BaseException:
            return True
        return any(
            not _stored_values_equal(actual, expected)
            for actual, expected in zip(current, self.values, strict=True)
        )

    def restore(self) -> None:
        carrier = self.carrier
        storage = self.definition.storage
        try:
            current_size = _provider_size(carrier, self.definition)
        except BaseException:
            current_size = -1
        if carrier.is_released() or current_size != len(self.values):
            released = storage.release(carrier)
            if released is not None:
                raise TypeError("StorageProvider.release must return None")
            allocated = storage.allocate(carrier, len(self.values))
            if allocated is not None:
                raise TypeError("StorageProvider.allocate must return None")
        for index, value in enumerate(self.values):
            returned = storage.write(carrier, index, deepcopy(value))
            if returned is not None:
                raise TypeError("StorageProvider.write must return None")
        carrier._restore_definition_state(self.version, self.is_released)


class _ProviderMutationTransaction:
    __slots__ = ("_snapshots",)

    def __init__(self, invocation: ResolvedInvocation) -> None:
        by_carrier: dict[int, tuple[Carrier, list[int]]] = {}
        for index, operand in enumerate(invocation.operands):
            if not isinstance(operand, Tensor):
                continue
            identity = id(operand.carrier)
            entry = by_carrier.get(identity)
            if entry is None:
                by_carrier[identity] = (operand.carrier, [index])
            else:
                entry[1].append(index)

        snapshots: list[_CarrierMutationSnapshot] = []
        for carrier, operand_indices in by_carrier.values():
            definition = _definition_for_instance(carrier)
            if definition is None:
                raise TypeError(
                    "definition-backed provider operands require exact "
                    "CarrierDefinitions"
                )
            size = _provider_size(carrier, definition)
            values = tuple(
                deepcopy(definition.storage.read(carrier, index))
                for index in range(size)
            )
            snapshots.append(
                _CarrierMutationSnapshot(
                    carrier,
                    definition,
                    values,
                    carrier.version,
                    carrier.is_released(),
                    carrier.is_owned(),
                    tuple(operand_indices),
                )
            )
        self._snapshots = tuple(snapshots)

    def require_unchanged(self, boundary: str) -> None:
        if any(snapshot.has_changed() for snapshot in self._snapshots):
            raise ProviderContractError(
                f"provider mutated an operand during {boundary}"
            )

    def commit(self, mutated_operands: tuple[int, ...]) -> None:
        authorized = set(mutated_operands)
        for snapshot in self._snapshots:
            changed = snapshot.has_changed()
            carrier_authorized = bool(snapshot.operand_indices) and all(
                index in authorized for index in snapshot.operand_indices
            )
            if changed and not carrier_authorized:
                index = next(
                    index
                    for index in snapshot.operand_indices
                    if index not in authorized
                )
                raise ProviderContractError(
                    f"provider mutated unauthorized operand {index}"
                )
            if not carrier_authorized:
                continue
            if (
                snapshot.carrier.is_released() is not snapshot.is_released
                or snapshot.carrier.is_owned() is not snapshot.is_owned
                or _provider_size(snapshot.carrier, snapshot.definition)
                != len(snapshot.values)
            ):
                raise ProviderContractError(
                    "provider mutation changed carrier lifecycle or storage extent"
                )
            snapshot.carrier._restore_definition_state(
                snapshot.version, snapshot.is_released
            )
            snapshot.carrier._increment_version()

    def rollback(self) -> None:
        failures: list[BaseException] = []
        for snapshot in self._snapshots:
            if not snapshot.has_changed():
                continue
            try:
                snapshot.restore()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise ProviderContractError(
                "provider mutation rollback failed"
            ) from failures[0]


def _rollback_after_provider_failure(
    transaction: _ProviderMutationTransaction, error: BaseException
) -> None:
    try:
        transaction.rollback()
    except BaseException as rollback_error:
        raise rollback_error from error


def _matching_patterns(
    carrier_class: type[Carrier], invocation: ResolvedInvocation
) -> tuple[KernelPattern, ...]:
    indexed = tuple(enumerate(_definition_kernel_patterns(carrier_class)))
    matching = [
        (index, pattern)
        for index, pattern in indexed
        if pattern.operation is invocation.definition
        and any(capability.matches(invocation.plan) for capability in pattern.plans)
    ]
    matching.sort(key=lambda entry: (-entry[1].preference, entry[0]))
    return tuple(pattern for _index, pattern in matching)


def _completed_provider_result(value: object) -> ProviderResult:
    if isinstance(value, ProviderResult):
        return value
    completion = _require_provider_completion(value)
    result = completion.wait()
    if not isinstance(result, ProviderResult):
        raise TypeError("kernel ProviderCompletion.wait() must return a ProviderResult")
    if type(completion.done) is not bool or not completion.done:
        raise TypeError("provider completion done must be true after wait()")
    return result


def _validate_provider_result(
    carrier: Carrier,
    invocation: ResolvedInvocation,
    result: ProviderResult,
    *,
    result_carriers: tuple[type, ...] | None = None,
) -> tuple[Tensor, ...]:
    outputs = cast(tuple[Tensor, ...], result.outputs)
    if len(outputs) != len(invocation.results):
        raise ProviderContractError("provider result count conflicts")
    tensor_operands = tuple(
        operand for operand in invocation.operands if isinstance(operand, Tensor)
    )
    for index, (output, spec) in enumerate(
        zip(outputs, invocation.results, strict=True)
    ):
        if output.dtype() is not spec.dtype:
            raise ProviderContractError(f"provider result {index} dtype conflicts")
        if output.layout.shape != spec.shape:
            raise ProviderContractError(f"provider result {index} shape conflicts")
        if output.layout != spec.layout:
            raise ProviderContractError(f"provider result {index} layout conflicts")
        expected_carrier = (
            type(carrier) if result_carriers is None else result_carriers[index]
        )
        if type(output.carrier) is not expected_carrier:
            raise ProviderContractError(
                f"provider result {index} carrier role conflicts"
            )
        if spec.alias_of is not None:
            if output is not invocation.operands[spec.alias_of]:
                raise ProviderContractError(
                    f"provider result {index} aliasing conflicts"
                )
            continue
        if any(output is operand for operand in tensor_operands):
            raise ProviderContractError(
                f"provider result {index} unexpectedly aliases an operand"
            )
        if output.offset != 0:
            raise ProviderContractError(f"provider result {index} offset conflicts")
        if output.carrier.size() != output.layout.cosize:
            raise ProviderContractError(
                f"provider result {index} physical storage size conflicts"
            )
    return outputs


def _execute_kernel_invocation(
    carrier: Carrier,
    definition: CarrierDefinition,
    invocation: ResolvedInvocation,
) -> tuple[Tensor, ...]:
    provider = definition.kernels
    if provider is None:
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} is storage-only and executes no operations"
        )
    require_capability(type(carrier), invocation.plan)
    candidates = _matching_patterns(type(carrier), invocation)
    if not candidates:
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} has no matching kernel pattern for "
            f"{invocation.name!r}"
        )
    transaction = _ProviderMutationTransaction(invocation)
    try:
        rejections: list[str] = []
        prepared: PreparedKernel | None = None
        for pattern in candidates:
            candidate = pattern.prepare(invocation, provider.interface)
            transaction.require_unchanged("kernel preparation")
            if isinstance(candidate, Unsupported):
                rejections.append(f"{pattern.kernel_id}: {candidate.reason}")
                continue
            if not isinstance(candidate, PreparedKernel):
                raise TypeError(
                    "KernelPattern.prepare must return PreparedKernel or Unsupported"
                )
            if (
                candidate.interface.compatibility_identity
                != provider.interface.compatibility_identity
            ):
                raise TypeError(
                    "PreparedKernel interface does not match the KernelProvider "
                    "interface"
                )
            prepared = candidate
            break
        if prepared is None:
            details = "; ".join(rejections)
            raise UnsupportedOperationPlan(
                f"all matching kernel patterns rejected {invocation.name!r}: {details}"
            )
        submitted = prepared.submit()
        result = _completed_provider_result(submitted)
        outputs = _validate_provider_result(carrier, invocation, result)
        transaction.commit(invocation.mutated_operands)
        return outputs
    except BaseException as error:
        _rollback_after_provider_failure(transaction, error)
        raise


def _composite_result_carriers(
    carrier: Carrier,
    definition: CarrierDefinition,
    invocation: ResolvedInvocation,
) -> tuple[type, ...]:
    from .base import Carrier as CarrierType

    composite = definition.composite
    if composite is None:
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} has no CompositeProvider"
        )
    result_carriers = composite.result_carriers(carrier, invocation)
    if not isinstance(result_carriers, tuple):
        raise TypeError("CompositeProvider.result_carriers must return a tuple")
    if len(result_carriers) != len(invocation.results):
        raise ProviderContractError("composite result carrier count conflicts")
    for result_carrier in result_carriers:
        if (
            not isinstance(result_carrier, type)
            or result_carrier is CarrierType
            or not issubclass(result_carrier, CarrierType)
            or inspect.isabstract(result_carrier)
        ):
            raise TypeError(
                "CompositeProvider.result_carriers must contain concrete "
                "Carrier classes"
            )
    return cast(tuple[type, ...], result_carriers)


def _execute_composite_invocation(
    carrier: Carrier,
    definition: CarrierDefinition,
    invocation: ResolvedInvocation,
) -> tuple[Tensor, ...]:
    composite = definition.composite
    if composite is None:
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} has no CompositeProvider"
        )
    require_carrier_capability(carrier, invocation.plan)
    transaction = _ProviderMutationTransaction(invocation)
    try:
        result_carriers = _composite_result_carriers(carrier, definition, invocation)
        transaction.require_unchanged("composite result-carrier resolution")
        prepared = composite.prepare(carrier, invocation)
        transaction.require_unchanged("composite preparation")
        if isinstance(prepared, Unsupported):
            raise UnsupportedOperationPlan(
                f"{type(carrier).__name__} rejected {invocation.name!r}: "
                f"{prepared.reason}"
            )
        if not isinstance(prepared, PreparedComposite):
            raise TypeError(
                "CompositeProvider.prepare must return PreparedComposite or Unsupported"
            )
        result = _completed_provider_result(prepared.submit())
        outputs = _validate_provider_result(
            carrier,
            invocation,
            result,
            result_carriers=result_carriers,
        )
        transaction.commit(invocation.mutated_operands)
        return outputs
    except BaseException as error:
        _rollback_after_provider_failure(transaction, error)
        raise


def execute_definition_invocation(
    carrier: Carrier, invocation: ResolvedInvocation
) -> tuple[Tensor, ...]:
    """Execute one resolved invocation through its exact carrier definition."""

    definition = _observe_carrier_definition(type(carrier))
    if definition is None:
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} has no CarrierDefinition"
        )
    if carrier.is_released():
        raise RuntimeError(f"{type(carrier).__name__} is released")
    _reject_graph_building_exact_alias(invocation)
    if definition.kernels is not None:
        return _execute_kernel_invocation(carrier, definition, invocation)
    if definition.composite is not None:
        return _execute_composite_invocation(carrier, definition, invocation)
    raise UnsupportedOperationPlan(
        f"{type(carrier).__name__} is storage-only and executes no operations"
    )


class _DefinitionResultAutogradContext:
    """Route one published result cotangent through its shared definition VJP."""

    __slots__ = ("_index", "_owner")

    def __init__(self, owner: DefinitionBackedOperation, index: int) -> None:
        self._owner = owner
        self._index = index

    @property
    def _autograd_state_freed(self) -> bool:
        return bool(self._owner._autograd_state_freed)

    @property
    def _dispatch_carrier_class(self) -> type[Carrier] | None:
        return cast(Any, self._owner._dispatch_carrier_class)

    @property
    def _operation_name(self) -> str | None:
        return cast(str | None, self._owner._operation_name)

    def inputs(self) -> tuple[Tensor, ...]:
        return cast(tuple[Tensor, ...], self._owner.inputs())

    def validate_input_versions(self) -> None:
        self._owner.validate_input_versions()

    def backward(self, gradient: Tensor) -> tuple[Tensor | None, ...]:
        return self._owner._backward_result(self._index, gradient)

    def _release_autograd_state(self) -> None:
        self._owner._release_autograd_state()


class DefinitionBackedOperation(Operation):
    """Fresh dispatched Operation for one exact definition and carrier."""

    def __init__(self, carrier: Carrier, definition: object) -> None:
        super().__init__()
        self._definition_carrier = carrier
        self._definition = definition
        self._provided_invocation: ResolvedInvocation | None = None
        self._resolved_invocation: ResolvedInvocation | None = None
        self._published_outputs: tuple[Tensor, ...] = ()
        self._result_contexts: tuple[_DefinitionResultAutogradContext, ...] = ()
        self._saved_operands: tuple[Tensor, ...] = ()
        self._saved_versions: tuple[object, ...] = ()

    def _semantic_options(self) -> dict[str, object]:
        options = self._execution_options
        if options is None:
            return {}
        return {"accumulator_dtype": options.accumulator_dtype}

    def _supply_resolved_invocation(self, invocation: ResolvedInvocation) -> None:
        if self._provided_invocation is not None:
            raise RuntimeError("a resolved invocation was already supplied")
        if invocation.definition is not self._definition:
            raise ValueError("resolved invocation definition conflicts")
        self._provided_invocation = invocation

    def _accepts_multiple_results(self) -> bool:
        return True

    def _allows_autograd(self) -> bool:
        invocation = self._resolved_invocation
        return invocation is not None and _invocation_allows_autograd(invocation)

    def _autograd_context_for_result(self, index: int) -> object:
        if len(self._published_outputs) == 1:
            if index != 0:
                raise IndexError("definition result index is out of range")
            return self
        if not self._result_contexts:
            self._result_contexts = tuple(
                _DefinitionResultAutogradContext(self, result_index)
                for result_index in range(len(self._published_outputs))
            )
        try:
            return self._result_contexts[index]
        except IndexError as error:
            raise IndexError("definition result index is out of range") from error

    def _forward(self, *operands: object) -> Tensor | tuple[Tensor, ...]:
        definition = cast(Any, self._definition)
        invocation = self._provided_invocation
        if invocation is None:
            invocation = definition.resolve_call(*operands, **self._semantic_options())
        elif len(operands) != len(invocation.operands) or any(
            actual is not expected
            for actual, expected in zip(operands, invocation.operands, strict=True)
        ):
            raise ValueError("forward operands conflict with resolved invocation")
        tensor_operands = tuple(
            operand for operand in invocation.operands if isinstance(operand, Tensor)
        )
        if (
            not tensor_operands
            or tensor_operands[0].carrier is not self._definition_carrier
        ):
            raise TypeError(
                "definition-backed dispatch must execute on its bound first Tensor carrier"
            )
        if any(
            type(operand.carrier) is not type(self._definition_carrier)
            for operand in tensor_operands
        ):
            raise TypeError("Tensor backing carriers must match")
        self._provided_invocation = None
        outputs = execute_definition_invocation(self._definition_carrier, invocation)
        if (
            _is_grad_enabled()
            and _invocation_allows_autograd(invocation)
            and any(output.is_differentiable() for output in outputs)
        ):
            saved = tuple(
                cast(Tensor, invocation.operands[index])
                for index in invocation.saved_operands
            )
            self._resolved_invocation = invocation
            self._published_outputs = outputs
            self._saved_operands = saved
            self._saved_versions = tuple(operand._version_token() for operand in saved)
        else:
            self._resolved_invocation = None
            self._published_outputs = ()
            self._saved_operands = ()
            self._saved_versions = ()
        return outputs[0] if len(outputs) == 1 else outputs

    def validate_input_versions(self) -> None:
        for operand, version in zip(
            self._saved_operands, self._saved_versions, strict=True
        ):
            if operand._version_token() != version:
                raise RuntimeError(
                    "A tensor needed for gradient computation was modified "
                    "in-place: its representation version token changed"
                )

    def _backward_result(
        self, result_index: int, gradient: Tensor
    ) -> tuple[Tensor | None, ...]:
        invocation = self._resolved_invocation
        if invocation is None:
            raise RuntimeError("definition-backed operation has no resolved invocation")
        if result_index < 0 or result_index >= len(invocation.results):
            raise IndexError("definition result index is out of range")
        context = VJPContext(invocation, self._published_outputs, self._saved_operands)
        scope_factory = getattr(self._definition_carrier, "_composite_vjp_scope", None)
        scope = cast(
            AbstractContextManager[object],
            scope_factory(invocation) if callable(scope_factory) else nullcontext(),
        )
        with scope:
            cotangents: tuple[Tensor | None, ...] = tuple(
                gradient if index == result_index else None
                for index in range(len(invocation.results))
            )
            semantic_gradients = _call_vjp(context, cotangents)
        by_operand = dict(
            zip(
                _differentiable_indices(invocation),
                semantic_gradients,
                strict=True,
            )
        )
        return tuple(
            by_operand.get(index)
            for index, operand in enumerate(invocation.operands)
            if isinstance(operand, Tensor)
        )

    def backward(self, gradient: Tensor) -> tuple[Tensor | None, ...]:
        return self._backward_result(0, gradient)

    def _release_autograd_state(self) -> None:
        Operation._release_autograd_state(self)
        self._resolved_invocation = None
        self._provided_invocation = None
        self._published_outputs = ()
        self._result_contexts = ()
        self._saved_operands = ()
        self._saved_versions = ()


def definition_dispatch_operation(carrier: Carrier, operation_name: str) -> object:
    """Return a fresh dispatched operation for an exact carrier definition."""

    definition = _observe_carrier_definition(type(carrier))
    if definition is None:
        raise LookupError
    if carrier.is_released():
        raise RuntimeError(f"{type(carrier).__name__} is released")
    try:
        operation = _operation_definition(operation_name)
    except LookupError as error:
        raise NotImplementedError(
            f"{type(carrier).__name__} does not support operation {operation_name!r}"
        ) from error
    if definition.kernels is None:
        if definition.composite is None:
            raise UnsupportedOperationPlan(
                f"{type(carrier).__name__} is storage-only and does not support "
                f"operation {operation_name!r}"
            )
        if not carrier_operation_capabilities(carrier, operation_name):
            raise UnsupportedOperationPlan(
                f"{type(carrier).__name__} does not support operation "
                f"{operation_name!r}"
            )
    elif not any(
        pattern.operation is operation
        for pattern in _definition_kernel_patterns(type(carrier))
    ):
        raise UnsupportedOperationPlan(
            f"{type(carrier).__name__} has no kernel for operation {operation_name!r}"
        )
    return DefinitionBackedOperation(carrier, operation)


__all__ = ["DefinitionBackedOperation"]
