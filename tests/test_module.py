from typing import Any

import pytest

import strideweave as sw
from strideweave import Generic, Layout, Module, Parameter, Shape, Stride, Tensor


def make_tensor(value: Any) -> Tensor:
    return Tensor(Generic([value]), 0, Layout(Shape(1), Stride(1)))


def make_parameter(value: Any) -> Parameter:
    return Parameter(make_tensor(value))


class EchoModule(Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self.calls.append((args, kwargs))
        return args, kwargs


def test_module_public_api_imports():
    assert sw.Module is Module
    assert sw.Parameter is Parameter


def test_parameter_is_tensor_and_can_wrap_existing_tensor():
    tensor = make_tensor("value")

    parameter = Parameter(tensor)

    assert isinstance(parameter, Tensor)
    assert parameter.carrier is tensor.carrier
    assert parameter.offset == tensor.offset
    assert parameter.layout == tensor.layout
    assert parameter[0] == "value"


def test_parameter_supports_direct_tensor_constructor_arguments():
    carrier = Generic(["value"])
    layout = Layout(Shape(1), Stride(1))

    parameter = Parameter(carrier, 0, layout)

    assert isinstance(parameter, Tensor)
    assert parameter.carrier is carrier
    assert parameter.offset == 0
    assert parameter.layout == layout
    assert parameter[0] == "value"


def test_parameter_supports_optional_name_metadata():
    tensor = make_tensor("value")

    wrapped = Parameter(tensor, name="wrapped")
    direct = Parameter(tensor.carrier, 0, tensor.layout, name="direct")

    assert wrapped.name == "wrapped"
    assert direct.name == "direct"


def test_parameter_rejects_ambiguous_constructor_arguments():
    tensor = make_tensor("value")

    with pytest.raises(TypeError):
        Parameter(tensor, 0, tensor.layout)

    with pytest.raises(TypeError):
        Parameter(tensor.carrier)


def test_parameter_rejects_invalid_names():
    tensor = make_tensor("value")

    with pytest.raises(ValueError):
        Parameter(tensor, name="")

    with pytest.raises(ValueError):
        Parameter(tensor, name="bad.name")

    with pytest.raises(TypeError):
        Parameter(tensor, name=123)  # type: ignore[arg-type]


def test_module_supports_optional_name_metadata():
    module = Module(name="block")

    assert module.name == "block"


def test_module_call_delegates_to_forward_with_args_and_kwargs():
    module = EchoModule()

    result = module("x", "y", scale=3)

    assert result == (("x", "y"), {"scale": 3})
    assert module.calls == [(("x", "y"), {"scale": 3})]


def test_module_base_forward_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Module().forward("x")


def test_module_registers_direct_parameters_and_submodules():
    root = Module()
    child = Module()
    root_weight = make_parameter("root")
    child_weight = make_parameter("child")
    root.weight = root_weight
    child.weight = child_weight
    root.child = child

    assert root_weight[0] == "root"
    assert getattr(root, "child") is child
    assert root.modules() == (root, child)
    assert root.parameters() == (root_weight, child_weight)
    assert root.get_named_parameters() == (
        ("weight", root_weight),
        ("child.weight", child_weight),
    )


def test_module_traversal_preserves_recursive_insertion_order():
    root = Module()
    first = Module()
    second = Module()
    nested = Module()
    root_weight = make_parameter("root")
    first_weight = make_parameter("first")
    nested_weight = make_parameter("nested")
    second_weight = make_parameter("second")
    root.root_weight = root_weight
    first.first_weight = first_weight
    nested.nested_weight = nested_weight
    second.second_weight = second_weight
    first.nested = nested
    root.first = first
    root.second = second

    assert root.modules() == (root, first, nested, second)
    assert root.parameters() == (
        root_weight,
        first_weight,
        nested_weight,
        second_weight,
    )
    assert root.get_named_parameters() == (
        ("root_weight", root_weight),
        ("first.first_weight", first_weight),
        ("first.nested.nested_weight", nested_weight),
        ("second.second_weight", second_weight),
    )


def test_module_reassignment_and_deletion_update_registries():
    root = Module()
    parameter = make_parameter("parameter")
    child = Module()
    raw_tensor = make_tensor("raw")

    root.slot = parameter
    assert root.parameters() == (parameter,)

    root.slot = child
    assert root.parameters() == ()
    assert root.modules() == (root, child)

    root.slot = raw_tensor
    assert root.parameters() == ()
    assert root.modules() == (root,)

    root.slot = parameter
    del root.slot
    assert root.parameters() == ()
    assert not hasattr(root, "slot")


def test_module_does_not_register_private_or_plain_tensor_attributes():
    root = Module()
    parameter = make_parameter("parameter")
    raw_tensor = make_tensor("raw")

    root._private_parameter = parameter
    root.raw_tensor = raw_tensor

    assert root.parameters() == ()
    assert root.modules() == (root,)
    assert getattr(root, "_private_parameter") is parameter
    assert getattr(root, "raw_tensor") is raw_tensor


def test_module_deduplicates_shared_submodules_and_parameters():
    root = Module()
    shared_child = Module()
    shared_parameter = make_parameter("shared")
    shared_child.weight = shared_parameter

    root.first = shared_child
    root.second = shared_child

    assert root.modules() == (root, shared_child)
    assert root.parameters() == (shared_parameter,)
    assert root.get_named_parameters() == (("first.weight", shared_parameter),)


def test_module_deduplicates_shared_direct_parameters():
    root = Module()
    shared_parameter = make_parameter("shared")

    root.first = shared_parameter
    root.second = shared_parameter

    assert root.parameters() == (shared_parameter,)
    assert root.get_named_parameters() == (("first", shared_parameter),)


def test_named_parameters_use_explicit_parameter_names():
    root = Module()
    parameter = Parameter(make_tensor("value"), name="kernel")

    root.weight = parameter

    assert root.get_named_parameters() == (("kernel", parameter),)


def test_named_parameters_use_explicit_child_module_names():
    root = Module()
    child = Module(name="encoder")
    parameter = make_parameter("value")
    child.weight = parameter
    root.layer = child

    assert root.get_named_parameters() == (("encoder.weight", parameter),)


def test_named_parameters_compose_nested_explicit_names():
    root = Module(name="root_name")
    child = Module(name="encoder")
    grandchild = Module(name="projection")
    parameter = Parameter(make_tensor("value"), name="kernel")
    grandchild.weight = parameter
    child.inner = grandchild
    root.layer = child

    assert root.get_named_parameters() == (("encoder.projection.kernel", parameter),)


def test_root_module_name_is_not_prefixed_in_own_named_parameters():
    root = Module(name="root_name")
    parameter = make_parameter("value")
    root.weight = parameter

    assert root.get_named_parameters() == (("weight", parameter),)


def test_mutating_names_updates_later_named_parameter_output():
    root = Module()
    child = Module()
    parameter = make_parameter("value")
    child.weight = parameter
    root.layer = child

    assert root.get_named_parameters() == (("layer.weight", parameter),)

    child.name = "encoder"
    parameter.name = "kernel"

    assert root.get_named_parameters() == (("encoder.kernel", parameter),)


def test_assigning_module_name_does_not_register_it():
    root = Module()
    root.name = "model"

    assert root.name == "model"
    assert root.modules() == (root,)
    assert root.parameters() == ()


def test_module_rejects_invalid_names():
    with pytest.raises(ValueError):
        Module(name="")

    with pytest.raises(ValueError):
        Module(name="bad.name")

    with pytest.raises(TypeError):
        Module(name=123)  # type: ignore[arg-type]

    module = Module()
    with pytest.raises(ValueError):
        module.name = ""

    with pytest.raises(ValueError):
        module.name = "bad.name"

    with pytest.raises(TypeError):
        module.name = 123  # type: ignore[assignment]
