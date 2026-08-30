from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError, dataclass, replace
from threading import Barrier, Thread
from typing import Any, Self, cast

import pytest

import strideweave as sw
import strideweave.carriers.extension as extension
from strideweave.carriers.operation_capability import (
    register_operation_capabilities,
)


def storage_provider(
    *,
    supports_dtype: Any = None,
    allocate_result: object = None,
    release_result: object = None,
    write_result: object = None,
) -> sw.StorageProvider:
    def allocate(carrier: Any, slots: int) -> object:
        carrier.values = [0] * slots
        return allocate_result

    def release(carrier: Any) -> object:
        carrier.values.clear()
        return release_result

    def write(carrier: Any, index: int, value: object) -> object:
        carrier.values[index] = value
        return write_result

    def physical_size(carrier: Any) -> int:
        return len(carrier.values)

    def read(carrier: Any, index: int) -> object:
        return carrier.values[index]

    return sw.StorageProvider(
        supports_dtype or (lambda _carrier, dtype: dtype is sw.DType.Int32),
        physical_size,
        cast(Any, allocate),
        cast(Any, release),
        read,
        cast(Any, write),
        elementwise_fallback=True,
    )


class OneShot(Iterator[object]):
    def __init__(self, values: list[object]) -> None:
        self._values = iter(values)
        self.iterations = 0

    def __iter__(self) -> Self:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("iterable was consumed more than once")
        return self

    def __next__(self) -> object:
        return next(self._values)


def fresh_carrier(name: str) -> type[sw.Carrier]:
    return type(name, (sw.Carrier,), {})


def test_extension_foundation_has_identity_preserving_public_imports() -> None:
    top_level_names = {
        "CarrierDefinition",
        "CarrierFacet",
        "CompositeProvider",
        "KernelExecutionInterface",
        "KernelProvider",
        "StorageProvider",
        "TransferRoute",
        "Unsupported",
        "register_carrier_definition",
    }
    module_only_names = {
        "PreparedComposite",
        "PreparedKernel",
        "PreparedTransfer",
        "ProviderCompletion",
        "ProviderResult",
        "TransferProvider",
        "TransferRequest",
    }

    assert top_level_names <= set(sw.__all__)
    assert top_level_names | module_only_names <= set(extension.__all__)
    for name in top_level_names:
        assert getattr(sw, name) is getattr(extension, name)


@pytest.mark.parametrize(
    "field",
    ["supports_dtype", "physical_size", "allocate", "release", "read", "write"],
)
def test_storage_provider_requires_every_callback(field: str) -> None:
    callbacks = {
        "supports_dtype": lambda carrier, dtype: True,
        "physical_size": lambda carrier: 0,
        "allocate": lambda carrier, slots: None,
        "release": lambda carrier: None,
        "read": lambda carrier, index: None,
        "write": lambda carrier, index, value: None,
    }
    callbacks[field] = None

    with pytest.raises(TypeError, match=rf"{field} must be callable"):
        sw.StorageProvider(**callbacks)  # type: ignore[arg-type]


def test_storage_provider_requires_an_exact_boolean_fallback_and_is_immutable() -> None:
    with pytest.raises(TypeError, match="elementwise_fallback must be a bool"):
        sw.StorageProvider(
            lambda carrier, dtype: True,
            lambda carrier: 0,
            lambda carrier, slots: None,
            lambda carrier: None,
            lambda carrier, index: None,
            lambda carrier, index, value: None,
            elementwise_fallback=1,  # type: ignore[arg-type]
        )

    provider = storage_provider()
    with pytest.raises(FrozenInstanceError):
        provider.elementwise_fallback = False  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class MetricsFacet(sw.CarrierFacet):
    label: str


def test_definition_materializes_routes_and_facets_once() -> None:
    source = fresh_carrier("MaterializedSource")
    destination = fresh_carrier("MaterializedDestination")
    route = sw.TransferRoute(
        source,
        destination,
        "copy",
        extension.TransferProvider(lambda request: extension.Unsupported("no copy")),
    )
    routes = OneShot([route])
    facets = OneShot([MetricsFacet("metrics")])

    definition = sw.CarrierDefinition(
        storage_provider(),
        transfers=routes,  # type: ignore[arg-type]
        facets=facets,  # type: ignore[arg-type]
    )

    assert routes.iterations == facets.iterations == 1
    assert definition.transfers == (route,)
    assert definition.facets == (MetricsFacet("metrics"),)


def test_definition_rejects_duplicate_routes_and_exact_facet_types() -> None:
    source = fresh_carrier("DuplicateSource")
    destination = fresh_carrier("DuplicateDestination")
    route = sw.TransferRoute(
        source,
        destination,
        "first",
        extension.TransferProvider(lambda request: extension.Unsupported("no copy")),
    )
    duplicate = sw.TransferRoute(
        source,
        destination,
        "second",
        extension.TransferProvider(lambda request: extension.Unsupported("no copy")),
    )

    with pytest.raises(ValueError, match="duplicate exact transfer routes"):
        sw.CarrierDefinition(storage_provider(), transfers=(route, duplicate))
    with pytest.raises(ValueError, match="duplicate exact facet types"):
        sw.CarrierDefinition(
            storage_provider(),
            facets=(MetricsFacet("one"), MetricsFacet("two")),
        )


def test_definition_registration_requires_its_exact_source_class() -> None:
    source = fresh_carrier("RouteSource")
    other_source = fresh_carrier("OtherRouteSource")
    destination = fresh_carrier("RouteDestination")
    route = sw.TransferRoute(
        other_source,
        destination,
        "copy",
        extension.TransferProvider(lambda request: extension.Unsupported("no copy")),
    )

    with pytest.raises(TypeError, match="exact source_class"):
        sw.register_carrier_definition(
            source,
            sw.CarrierDefinition(storage_provider(), transfers=(route,)),
        )


def test_minimal_definition_backed_carrier_uses_generic_storage_and_facets() -> None:
    carrier_class = fresh_carrier("DefinitionStorage")
    facet = MetricsFacet("ready")
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(storage_provider(), facets=(facet,)),
    )

    carrier = carrier_class(3, dtype=sw.DType.Int32)

    assert carrier.size() == 3
    assert carrier.dtype() is sw.DType.Int32
    assert carrier.supports_storage_dtype(sw.DType.Int32)
    assert not carrier.supports_storage_dtype(sw.DType.Float32)
    assert carrier.is_mutable()
    assert carrier.facet(MetricsFacet) is facet
    assert carrier.require_facet(MetricsFacet) is facet
    carrier[1] = 7
    assert carrier[1] == 7
    assert carrier.version == 1

    copied = carrier.new_like(iter([4, 5]), mutable=False)
    assert type(copied) is carrier_class
    assert [copied[index] for index in range(2)] == [4, 5]
    assert copied.version == 0
    assert not copied.is_mutable()

    allocated = carrier.allocate_like(2)
    assert type(allocated) is carrier_class
    assert allocated.size() == 2
    assert allocated.version == 0
    assert carrier.operation_capabilities() == ()
    with pytest.raises(sw.UnsupportedOperationPlan, match="storage-only"):
        carrier.dispatch_op("relu")

    carrier.release()
    carrier.release()
    assert carrier.is_released()
    assert carrier.size() == 0
    with pytest.raises(RuntimeError, match="DefinitionStorage is released"):
        carrier.dispatch_op("not_a_registered_operation")
    with pytest.raises(RuntimeError, match="released"):
        _ = carrier[0]


def test_fixed_storage_dtype_provider_may_ignore_the_exact_carrier() -> None:
    carrier_class = fresh_carrier("FixedStorageSupport")
    allocated: list[sw.Carrier] = []
    provider = storage_provider(
        supports_dtype=lambda _carrier, dtype: dtype is sw.DType.Int32
    )
    original_allocate = provider.allocate

    def allocate(carrier: sw.Carrier, slots: int) -> None:
        allocated.append(carrier)
        original_allocate(carrier, slots)

    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(replace(cast(Any, provider), allocate=allocate)),
    )

    carrier = carrier_class(1, dtype=sw.DType.Int32)
    assert allocated == [carrier]
    assert carrier.supports_storage_dtype(sw.DType.Int32)
    assert not carrier.supports_storage_dtype(sw.DType.Float32)

    with pytest.raises(TypeError, match="does not support storage dtype Float32"):
        carrier_class(1, dtype=sw.DType.Float32)
    with pytest.raises(TypeError, match="does not support storage dtype Float32"):
        carrier.allocate_like(1, dtype=sw.DType.Float32)
    assert allocated == [carrier]


def test_storage_dtype_provider_receives_the_exact_configured_instance() -> None:
    observed: list[sw.Carrier] = []

    class InstanceStorageSupport(sw.Carrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.DType,
            supported_dtype: sw.DType,
            mutable: bool = True,
        ) -> None:
            self.supported_dtype = supported_dtype
            super().__init__(slots, dtype=dtype, mutable=mutable)

    def supports_dtype(carrier: sw.Carrier, dtype: sw.DType) -> bool:
        observed.append(carrier)
        return cast(Any, carrier).supported_dtype is dtype

    sw.register_carrier_definition(
        InstanceStorageSupport,
        sw.CarrierDefinition(storage_provider(supports_dtype=supports_dtype)),
    )

    carrier = InstanceStorageSupport(
        1,
        dtype=sw.DType.Int32,
        supported_dtype=sw.DType.Int32,
    )
    assert observed == [carrier]
    assert carrier.supports_storage_dtype(sw.DType.Int32)
    assert not carrier.supports_storage_dtype(sw.DType.Float32)
    assert observed == [carrier, carrier, carrier]

    carrier.release()
    assert carrier.supports_storage_dtype(sw.DType.Int32)
    assert observed[-1] is carrier


def test_facet_lookup_is_exact_typed_and_absent_on_legacy_carriers() -> None:
    @dataclass(frozen=True, slots=True)
    class DerivedMetricsFacet(MetricsFacet):
        pass

    carrier = sw.Generic([1.0])
    assert carrier.facet(MetricsFacet) is None
    with pytest.raises(NotImplementedError, match="MetricsFacet"):
        carrier.require_facet(MetricsFacet)
    with pytest.raises(TypeError, match="CarrierFacet subclass"):
        carrier.facet(object)

    carrier_class = fresh_carrier("ExactFacetCarrier")
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(
            storage_provider(), facets=(DerivedMetricsFacet("derived"),)
        ),
    )
    custom = carrier_class(1, dtype=sw.DType.Int32)
    assert custom.facet(MetricsFacet) is None
    assert isinstance(custom.facet(DerivedMetricsFacet), DerivedMetricsFacet)


def test_definition_scope_is_exact_and_does_not_traverse_a_base_class() -> None:
    base = fresh_carrier("DefinedBase")
    sw.register_carrier_definition(base, sw.CarrierDefinition(storage_provider()))
    subclass = type("UndefinedSubclass", (base,), {})

    base(1, dtype=sw.DType.Int32)
    with pytest.raises(TypeError, match="definition-free Carrier construction"):
        subclass(1, dtype=sw.DType.Int32)


def test_definition_registration_is_one_shot_and_seals_on_first_observation() -> None:
    defined = fresh_carrier("DefinedOnce")
    definition = sw.CarrierDefinition(storage_provider())
    sw.register_carrier_definition(defined, definition)
    with pytest.raises(TypeError, match="already has a CarrierDefinition"):
        sw.register_carrier_definition(defined, definition)

    observed = fresh_carrier("ObservedWithoutDefinition")
    observed()
    with pytest.raises(TypeError, match="already observed"):
        sw.register_carrier_definition(observed, definition)
    with pytest.raises(TypeError, match="already observed"):
        register_operation_capabilities(observed, ())


def test_definition_and_manual_capability_registration_are_one_atomic_choice() -> None:
    for attempt in range(20):
        carrier_class = fresh_carrier(f"AuthorityRace{attempt}")
        definition = sw.CarrierDefinition(storage_provider())
        barrier = Barrier(2)
        outcomes: list[tuple[str, object]] = []

        def register_definition() -> None:
            barrier.wait()
            try:
                sw.register_carrier_definition(carrier_class, definition)
                outcomes.append(("definition", None))
            except Exception as error:
                outcomes.append(("definition", error))

        def register_capabilities() -> None:
            barrier.wait()
            try:
                register_operation_capabilities(carrier_class, ())
                outcomes.append(("capabilities", None))
            except Exception as error:
                outcomes.append(("capabilities", error))

        threads = [
            Thread(target=register_definition),
            Thread(target=register_capabilities),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        successes = [name for name, error in outcomes if error is None]
        failures = [error for _, error in outcomes if error is not None]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], TypeError)


def test_definition_registration_rejects_manual_capability_authority() -> None:
    carrier_class = fresh_carrier("DefinitionCapabilityBoundary")
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(storage_provider()),
    )

    with pytest.raises(TypeError, match="CarrierDefinition"):
        register_operation_capabilities(carrier_class, ())


def test_definition_registration_rejects_invalid_composition_ownership() -> None:
    independent = fresh_carrier("IndependentComposite")
    composite = sw.CompositeProvider(
        lambda carrier: (),
        lambda carrier, invocation: (),
        lambda carrier, invocation: extension.Unsupported("unsupported"),
    )
    with pytest.raises(TypeError, match="must not supply a composite"):
        sw.register_carrier_definition(
            independent,
            sw.CarrierDefinition(storage_provider(), composite=composite),
        )


def test_definition_backed_dependent_derives_without_legacy_generator() -> None:
    capability_calls = 0

    def capabilities(carrier: object) -> tuple[()]:
        nonlocal capability_calls
        capability_calls += 1
        return ()

    composite = sw.CompositeProvider(
        capabilities,
        lambda carrier, invocation: (),
        lambda carrier, invocation: extension.Unsupported("unsupported"),
    )

    class DefinedDependent(sw.DependentCarrier):
        def __init__(self) -> None:
            super().__init__(1, dtype=sw.DType.Int32)
            self._finalize_dependent_capabilities()

        def _generate_operation_capabilities(self) -> tuple[()]:
            raise AssertionError(
                "definition-backed dependents do not use legacy generation"
            )

    sw.register_carrier_definition(
        DefinedDependent,
        sw.CarrierDefinition(storage_provider(), composite=composite),
    )

    carrier = DefinedDependent()
    assert carrier.operation_capabilities() == ()
    assert capability_calls == 1

    dependent = type("DependentDefinition", (sw.DependentCarrier,), {})
    with pytest.raises(TypeError, match="requires composite"):
        sw.register_carrier_definition(
            dependent,
            sw.CarrierDefinition(storage_provider()),
        )
    with pytest.raises(TypeError, match="forbids kernels"):
        sw.register_carrier_definition(
            dependent,
            sw.CarrierDefinition(
                storage_provider(),
                kernels=sw.KernelProvider(sw.KernelExecutionInterface("v1")),
                composite=composite,
            ),
        )


@pytest.mark.parametrize(
    "shipped",
    [sw.Generic, sw.CPU, sw.Metal, sw.FileBacked, sw.BlockDeviceCarrier, sw.Evictable],
)
def test_shipped_carriers_remain_definition_free(shipped: type[sw.Carrier]) -> None:
    assert extension._peek_carrier_definition(shipped) is None
    with pytest.raises(TypeError, match="definition-free shipped carrier"):
        sw.register_carrier_definition(
            shipped,
            sw.CarrierDefinition(storage_provider()),
        )


def test_block_device_resource_is_not_a_storage_provider_or_tensor_carrier() -> None:
    assert not issubclass(sw.BlockDevice, sw.Carrier)
    assert not issubclass(sw.BlockDevice, sw.StorageProvider)


def test_provider_completion_is_blocking_only_and_validates_done() -> None:
    class Completion:
        done = True

        def wait(self) -> object:
            return self

    completion = Completion()
    assert isinstance(completion, extension.ProviderCompletion)
    assert extension._require_provider_completion(completion) is completion
    assert not hasattr(completion, "__await__")

    class WrongDone(Completion):
        done = 1

    with pytest.raises(TypeError, match="done must be a bool"):
        extension._require_provider_completion(WrongDone())

    class Awaitable(Completion):
        def __await__(self) -> Iterable[object]:
            return iter(())

    with pytest.raises(TypeError, match="must not expose a coroutine"):
        extension._require_provider_completion(Awaitable())


def test_prepared_values_validate_callbacks_and_provider_result_tensors() -> None:
    interface = sw.KernelExecutionInterface("v1")
    assert (
        extension.PreparedKernel(interface, cast(Any, lambda: None)).interface
        is interface
    )
    assert callable(extension.PreparedTransfer(lambda: None).submit)
    assert callable(extension.PreparedComposite(cast(Any, lambda: None)).submit)
    with pytest.raises(TypeError, match="submit must be callable"):
        extension.PreparedTransfer(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outputs must not be empty"):
        extension.ProviderResult(())
    with pytest.raises(TypeError, match="Tensor values"):
        extension.ProviderResult((object(),))  # type: ignore[arg-type]

    tensor = sw.Tensor(sw.Generic([1.0]), 0, sw.Layout(sw.Shape(1), sw.Stride(1)))
    assert extension.ProviderResult((tensor,)).outputs == (tensor,)


def test_storage_provider_public_return_contracts_fail_closed() -> None:
    bad_support = fresh_carrier("BadSupportReturn")
    sw.register_carrier_definition(
        bad_support,
        sw.CarrierDefinition(
            storage_provider(supports_dtype=lambda _carrier, _dtype: 1)
        ),
    )
    with pytest.raises(TypeError, match="supports_dtype must return a bool"):
        bad_support(1, dtype=sw.DType.Int32)

    bad_allocate = fresh_carrier("BadAllocateReturn")
    sw.register_carrier_definition(
        bad_allocate,
        sw.CarrierDefinition(storage_provider(allocate_result=True)),
    )
    with pytest.raises(TypeError, match="allocate must return None"):
        bad_allocate(1, dtype=sw.DType.Int32)

    bad_write = fresh_carrier("BadWriteReturn")
    sw.register_carrier_definition(
        bad_write,
        sw.CarrierDefinition(storage_provider(write_result=True)),
    )
    carrier = bad_write(1, dtype=sw.DType.Int32)
    with pytest.raises(TypeError, match="write must return None"):
        carrier[0] = 1
    assert carrier.version == 0

    bad_release = fresh_carrier("BadReleaseReturn")
    sw.register_carrier_definition(
        bad_release,
        sw.CarrierDefinition(storage_provider(release_result=True)),
    )
    carrier = bad_release(1, dtype=sw.DType.Int32)
    with pytest.raises(TypeError, match="release must return None"):
        carrier.release()
    assert not carrier.is_released()


@pytest.mark.parametrize(
    ("physical_size", "error", "message"),
    [
        (lambda carrier: "1", TypeError, "must return an integer"),
        (lambda carrier: -1, ValueError, "must not return a negative"),
        (lambda carrier: 0, ValueError, "requested physical size"),
    ],
)
def test_storage_provider_physical_size_is_validated(
    physical_size: Any,
    error: type[Exception],
    message: str,
) -> None:
    carrier_class = fresh_carrier(f"BadPhysicalSize{message}")
    provider = replace(cast(Any, storage_provider()), physical_size=physical_size)
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(provider),
    )
    with pytest.raises(error, match=message):
        carrier_class(1, dtype=sw.DType.Int32)


def test_kernel_interface_and_foundation_records_validate_public_fields() -> None:
    with pytest.raises(TypeError, match="version must be a string"):
        sw.KernelExecutionInterface(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="version must not be empty"):
        sw.KernelExecutionInterface("")
    interface = sw.KernelExecutionInterface("v1")
    assert interface.compatibility_identity == (type(interface), "v1")
    with pytest.raises(AttributeError, match="immutable"):
        interface.version = "v2"  # type: ignore[misc]

    with pytest.raises(TypeError, match="KernelExecutionInterface"):
        sw.KernelProvider(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KernelPattern"):
        sw.KernelProvider(interface, (object(),))
    with pytest.raises(TypeError, match="reason must be a string"):
        sw.Unsupported(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must not be empty"):
        sw.Unsupported("")


def test_transfer_route_validates_exact_carrier_classes_and_identifier() -> None:
    source = fresh_carrier("ValidatedRouteSource")
    destination = fresh_carrier("ValidatedRouteDestination")
    provider = extension.TransferProvider(
        lambda request: extension.Unsupported("unsupported")
    )

    with pytest.raises(TypeError, match="source_class"):
        sw.TransferRoute(object, destination, "copy", provider)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="destination_class"):
        sw.TransferRoute(source, object, "copy", provider)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="route_id must not be empty"):
        sw.TransferRoute(source, destination, "", provider)
    with pytest.raises(TypeError, match="provider must be a TransferProvider"):
        sw.TransferRoute(source, destination, "copy", object())  # type: ignore[arg-type]
