from __future__ import annotations

import copy
import gc
import inspect
import pickle
import weakref
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, get_type_hints

import pytest

import strideweave as sw
from strideweave.core._representation import (
    Subtensor,
    TensorRepresentation,
)


class PlaneCarrier(sw.Carrier):
    """Minimal mutable carrier accepting an arbitrary canonical storage dtype."""

    def __init__(self, values: Iterable[object], dtype: sw.DType) -> None:
        super().__init__()
        self._values = list(values)
        self._dtype = dtype

    def size(self) -> int:
        return len(self._values)

    def dtype(self) -> sw.DType:
        return self._dtype

    def get_value(self, index: int) -> object:
        return self._values[index]

    def set_value(self, index: int, value: object) -> None:
        self._values[index] = value

    def _is_mutable(self) -> bool:
        return True

    def new_like(
        self, values: Iterable[object], *, mutable: bool = True
    ) -> PlaneCarrier:
        return type(self)(values, self._dtype)

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: sw.DType | None = None,
        empty: bool = False,
    ) -> PlaneCarrier:
        return type(self)([None] * size, self._dtype if dtype is None else dtype)

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError


class OtherPlaneCarrier(PlaneCarrier):
    """Distinct exact carrier class for homogeneity validation."""


class VirtualPlaneCarrier(sw.Carrier):
    """Carrier exposing a large virtual extent without allocating storage."""

    def __init__(self, size: int, dtype: sw.DType) -> None:
        super().__init__()
        self._size = size
        self._dtype = dtype

    def size(self) -> int:
        return self._size

    def dtype(self) -> sw.DType:
        return self._dtype

    def get_value(self, index: int) -> object:
        raise NotImplementedError

    def set_value(self, index: int, value: object) -> None:
        raise NotImplementedError

    def _is_mutable(self) -> bool:
        return False

    def new_like(
        self, values: Iterable[object], *, mutable: bool = True
    ) -> VirtualPlaneCarrier:
        raise NotImplementedError

    def allocate_like(
        self,
        size: int,
        *,
        mutable: bool = True,
        dtype: sw.DType | None = None,
        empty: bool = False,
    ) -> VirtualPlaneCarrier:
        raise NotImplementedError

    def scatter(
        self,
        to_scatter: Any,
        scatter_onto: Any,
        mapping: Any,
        mapping_offset: int = 0,
    ) -> None:
        raise NotImplementedError


class PlanarDType(sw.CompoundDType, abstract=False):
    """Simple external compound descriptor used by representation tests."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        *,
        planes: Iterable[sw.SimpleDType],
        rules: Iterable[sw.RepresentationRule] = (),
    ) -> None:
        super().__init__(
            name,
            supertype=sw.DType.Any,
            simple_types=planes,
            representation_rules=rules,
        )


class StructuredSparsityLikeDType(sw.CompoundDType, abstract=False):
    """External metadata format reusing the public grouping rule."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        *,
        planes: Iterable[sw.SimpleDType],
        extent: int | sw.WholeExtent,
    ) -> None:
        super().__init__(
            name,
            supertype=sw.DType.Any,
            simple_types=planes,
            representation_rules=(sw.LevelExtent(0, extent),),
        )


MULTI_FLOAT_DTYPE = PlanarDType(
    "TestMultiFloatTensorDType",
    planes=(sw.DType.Float32, sw.DType.Float32),
)


class TargetCardinality(sw.RepresentationRule):
    """Require one adjacent target level with a fixed cardinality."""

    __slots__ = ("_cardinality",)

    def __init__(self, cardinality: int) -> None:
        self._cardinality = cardinality

    def structure_extension(self) -> tuple[object, ...]:
        return (self._cardinality,)

    def validate(self, context: sw.RepresentationValidationContext) -> None:
        if len(context.level_shapes) < 2:
            raise ValueError("TargetCardinality requires a second level")
        if context.level_shapes[1].logical_size != self._cardinality:
            raise ValueError(
                f"TargetCardinality expected {self._cardinality} target coordinates"
            )


class RuleReached(sw.RepresentationRule):
    """Rule used to prove universal validation runs before optional checks."""

    __slots__ = ()

    def validate(self, context: sw.RepresentationValidationContext) -> None:
        raise RuntimeError("optional rule reached")


def placement(shape: sw.Shape, stride: sw.Stride) -> sw.Layout:
    return sw.Layout(shape, stride)


def grouped_pair(
    dtype: sw.CompoundDType,
    *,
    element_carrier: PlaneCarrier | None = None,
    metadata_carrier: PlaneCarrier | None = None,
    adjacent: sw.Layout | None = None,
) -> TensorRepresentation:
    element_carrier = element_carrier or PlaneCarrier(
        [0.0, 0.0, 0.0, 0.0], dtype.simple_types[0]
    )
    metadata_carrier = metadata_carrier or PlaneCarrier([0, 0], dtype.simple_types[1])
    element_layout = placement(sw.Shape([2, 2]), sw.Stride([1, 2]))
    metadata_layout = placement(sw.Shape(2), sw.Stride(1))
    grouping = adjacent or placement(sw.Shape([2, 2]), sw.Stride([0, 1]))
    return TensorRepresentation(
        dtype,
        (
            Subtensor(dtype.simple_types[0], element_carrier, 0, element_layout),
            Subtensor(dtype.simple_types[1], metadata_carrier, 0, metadata_layout),
        ),
        (grouping,),
    )


def multi_float_tensor(
    first: sw.Carrier,
    second: sw.Carrier,
) -> sw.Tensor:
    representation = TensorRepresentation(
        MULTI_FLOAT_DTYPE,
        (
            Subtensor(
                sw.DType.Float32,
                first,
                0,
                placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
            ),
            Subtensor(
                sw.DType.Float32,
                second,
                0,
                placement(sw.Shape(2), sw.Stride(1)),
            ),
        ),
        (placement(sw.Shape([2, 2]), sw.Stride([0, 1])),),
    )
    return sw.Tensor._from_representation(representation)


@pytest.mark.parametrize(
    "carrier",
    [
        sw.Generic([1.0, 2.0], dtype=sw.DType.Floating),
        sw.Generic([1.0, 2.0], dtype=sw.DType.Float32),
        sw.Generic([1, 2], dtype=sw.DType.Int32),
        sw.CPU(2, dtype=sw.DType.Float32),
        sw.CPU(2, dtype=sw.DType.Int32),
    ],
)
def test_public_tensor_constructor_uses_authoritative_one_subtensor_state(carrier):
    layout = placement(sw.Shape(2), sw.Stride(1))
    tensor = sw.Tensor(carrier, 0, layout)
    representation = tensor._representation

    assert isinstance(representation, TensorRepresentation)
    assert representation.is_single_subtensor
    assert representation.primary.carrier is tensor.carrier is carrier
    assert representation.primary.offset == tensor.offset == 0
    assert representation.primary.layout is tensor.layout is layout
    assert representation.logical_dtype is tensor.dtype() is carrier.dtype()
    assert tensor._version_token() == representation._version_token()


def test_native_tensor_stores_no_parallel_carrier_offset_or_layout_state():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "strideweave"
        / "core"
        / "native"
        / "_tensor.cpp"
    ).read_text(encoding="utf-8")

    assert "py::object representation_;" in source
    assert "py::object carrier_;" not in source
    assert "Index offset_;" not in source
    assert "py::object layout_;" not in source


def test_views_and_results_construct_fresh_authoritative_representations():
    tensor = sw.Tensor(
        sw.Generic([1.0, 2.0, 3.0, 4.0], dtype=sw.DType.Float32),
        0,
        placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
    )

    view = tensor[:, 1]
    result = sw.relu(tensor)

    assert isinstance(view._representation, TensorRepresentation)
    assert view._representation.primary.carrier is tensor.carrier
    assert view._representation.primary.offset == tensor.offset + 2
    assert view.carrier is view._representation.primary.carrier
    assert view.layout is view._representation.primary.layout
    assert isinstance(result._representation, TensorRepresentation)
    assert result.carrier is result._representation.primary.carrier
    assert result.dtype() is result._representation.logical_dtype


def test_multi_subtensor_accessors_read_the_authoritative_representation():
    first = sw.Generic([1.0] * 4, dtype=sw.DType.Float32)
    second = sw.Generic([1.0] * 2, dtype=sw.DType.Float32)
    tensor = multi_float_tensor(first, second)

    assert tensor.carrier is first
    assert tensor.offset == 0
    assert tensor.layout is tensor._representation.primary.layout
    assert tensor.dtype() is MULTI_FLOAT_DTYPE
    assert tensor.carrier_type() is sw.Generic
    assert tensor.size() == 4
    assert not tensor.is_mutable()
    version_token = tensor._version_token()
    assert version_token == tensor._representation._version_token()
    assert isinstance(version_token, tuple)
    assert len(version_token) == 2


def test_multi_subtensor_entry_points_fail_before_generic_carrier_effects():
    first = sw.Generic([1.0] * 4, dtype=sw.DType.Float32)
    second = sw.Generic([1.0] * 2, dtype=sw.DType.Float32)
    tensor = multi_float_tensor(first, second)
    destination = sw.Generic([], dtype=sw.DType.Float32)
    versions = (first.version, second.version)

    with pytest.raises(NotImplementedError, match="Multi-subtensor Tensor indexing"):
        tensor[0]
    with pytest.raises(NotImplementedError, match="Multi-subtensor Tensor mutation"):
        tensor[0] = 2.0
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor operation execution",
    ):
        tensor[:, 0]
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor operation execution",
    ):
        sw.add(tensor, tensor)
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor operation execution",
    ):
        sw.move(tensor, destination)
    with pytest.raises(NotImplementedError, match="Multi-subtensor Tensor backward"):
        tensor.backward()
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor DLPack export",
    ):
        tensor.__dlpack_device__()
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor DLPack export",
    ):
        tensor.__dlpack__()
    with pytest.raises(NotImplementedError, match="Multi-subtensor Tensor scatter"):
        first.scatter(
            tensor,
            tensor,
            placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
        )

    assert (first.version, second.version) == versions
    assert not first.is_released()
    assert not second.is_released()
    assert destination.size() == 0


def test_multi_subtensor_native_cpu_access_fails_before_mutation():
    first = sw.CPU(4, dtype=sw.DType.Float32)
    second = sw.CPU(2, dtype=sw.DType.Float32)
    tensor = multi_float_tensor(first, second)
    versions = (first.version, second.version)

    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor operation execution",
    ):
        sw.relu(tensor)
    with pytest.raises(
        NotImplementedError,
        match="Multi-subtensor Tensor native CPU access",
    ):
        first.scatter(
            tensor,
            tensor,
            placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
        )

    assert (first.version, second.version) == versions


def test_autograd_saves_the_complete_representation_version_token():
    tensor = sw.Tensor(
        sw.Generic([1.0, 2.0], dtype=sw.DType.Float32),
        0,
        placement(sw.Shape(2), sw.Stride(1)),
    )

    result = sw.relu(tensor)
    operation = result.autograd_ctx

    assert operation is not None
    assert operation.input_versions() == (tensor._version_token(),)


def test_simple_and_legacy_opaque_storage_have_one_plane():
    layout = placement(sw.Shape(3), sw.Stride(1))
    concrete = PlaneCarrier([1.0, 2.0, 3.0], sw.DType.Float32)
    opaque = PlaneCarrier([1.0, 2.0, 3.0], sw.DType.Floating)

    concrete_representation = TensorRepresentation(
        sw.DType.Float32,
        (Subtensor(sw.DType.Float32, concrete, 0, layout),),
    )
    opaque_representation = TensorRepresentation(
        sw.DType.Floating,
        (Subtensor(sw.DType.Floating, opaque, 0, layout),),
    )

    assert concrete_representation.is_single_subtensor
    assert concrete_representation.primary is concrete_representation.subtensors[0]
    assert opaque_representation.primary.carrier is opaque


def test_abstract_non_opaque_categories_have_no_storage_schema():
    carrier = PlaneCarrier([1], sw.DType.Integer)
    with pytest.raises(ValueError, match="abstract dtype category"):
        TensorRepresentation(
            sw.DType.Integer,
            (
                Subtensor(
                    sw.DType.Integer,
                    carrier,
                    0,
                    placement(sw.Shape(1), sw.Stride(1)),
                ),
            ),
        )


def test_compound_schema_requires_ordered_planes_and_adjacent_cardinality():
    dtype = PlanarDType(
        "TestRepresentationSchema",
        planes=(sw.DType.Float32, sw.DType.Int32),
    )
    element_layout = placement(sw.Shape([2, 2]), sw.Stride([1, 2]))
    element = PlaneCarrier([0.0] * 4, sw.DType.Float32)

    with pytest.raises(ValueError, match="requires 2 subtensors"):
        TensorRepresentation(
            dtype,
            (Subtensor(sw.DType.Float32, element, 0, element_layout),),
        )

    metadata = PlaneCarrier([0, 0], sw.DType.Int32)
    subtensors = (
        Subtensor(sw.DType.Float32, element, 0, element_layout),
        Subtensor(
            sw.DType.Int32,
            metadata,
            0,
            placement(sw.Shape(2), sw.Stride(1)),
        ),
    )
    with pytest.raises(ValueError, match="requires 1 adjacent layouts"):
        TensorRepresentation(dtype, subtensors)
    with pytest.raises(ValueError, match="subtensor 0 must use storage dtype Float32"):
        TensorRepresentation(
            dtype,
            (
                Subtensor(sw.DType.Int32, metadata, 0, element_layout),
                subtensors[1],
            ),
            (placement(sw.Shape([2, 2]), sw.Stride([0, 1])),),
        )


def test_carrier_dtype_identity_and_exact_class_are_validated():
    dtype = PlanarDType(
        "TestRepresentationCarrierContract",
        planes=(sw.DType.Float32, sw.DType.Int32),
    )
    wrong_dtype = PlaneCarrier([0.0] * 4, sw.DType.Int32)
    with pytest.raises(ValueError, match="carrier dtype must be identical"):
        grouped_pair(dtype, element_carrier=wrong_dtype)

    metadata = OtherPlaneCarrier([0, 0], sw.DType.Int32)
    with pytest.raises(TypeError, match="one exact class"):
        grouped_pair(dtype, metadata_carrier=metadata)


def test_offsets_and_placement_cosize_must_fit_storage():
    carrier = PlaneCarrier([0.0, 0.0], sw.DType.Float32)
    strided = placement(sw.Shape(2), sw.Stride(2))
    with pytest.raises(ValueError, match="placement storage exceeds carrier size"):
        TensorRepresentation(
            sw.DType.Float32,
            (Subtensor(sw.DType.Float32, carrier, 0, strided),),
        )
    with pytest.raises(ValueError, match="offset must be non-negative"):
        TensorRepresentation(
            sw.DType.Float32,
            (
                Subtensor(
                    sw.DType.Float32,
                    carrier,
                    -1,
                    placement(sw.Shape(1), sw.Stride(1)),
                ),
            ),
        )


def test_adjacent_layouts_use_source_domain_and_decode_in_target_shape():
    dtype = PlanarDType(
        "TestRepresentationAdjacentContract",
        planes=(sw.DType.Float32, sw.DType.Int32),
    )
    with pytest.raises(ValueError, match="source placement shape"):
        grouped_pair(
            dtype,
            adjacent=placement(sw.Shape(4), sw.Stride(0)),
        )
    with pytest.raises(ValueError, match="outside level 1 shape cardinality 2"):
        grouped_pair(
            dtype,
            adjacent=placement(sw.Shape([2, 2]), sw.Stride([0, 2])),
        )

    representation = grouped_pair(dtype)
    assert representation.adjacent_layouts[0].index((1, 1)) == 1
    assert representation.subtensors[1].layout.shape.logical_size == 2


def test_optional_rules_are_canonical_immutable_dtype_state():
    supplied = [TargetCardinality(2)]
    dtype = PlanarDType(
        "TestCanonicalRepresentationRules",
        planes=(sw.DType.Float32, sw.DType.Int32),
        rules=supplied,
    )
    structure = dtype.structure()
    supplied.clear()

    assert dtype.representation_rules
    assert structure[4] == (dtype.representation_rules[0].structure(),)
    assert copy.copy(dtype.representation_rules[0]) is dtype.representation_rules[0]
    assert copy.deepcopy(dtype.representation_rules[0]) is dtype.representation_rules[0]
    assert pickle.loads(pickle.dumps(dtype)) is dtype
    assert dtype.structure() == structure
    with pytest.raises(AttributeError, match="rules are immutable"):
        dtype.representation_rules[0]._cardinality = 3  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="immutable"):
        dtype.representation_rules = ()  # type: ignore[misc]


def test_rule_identity_and_immutability_cannot_be_overridden():
    with pytest.raises(TypeError, match="must not redefine structure"):

        class MisreportingRule(sw.RepresentationRule):
            __slots__ = ()

            def structure(self) -> tuple[object, ...]:
                return ()

            def validate(self, context: sw.RepresentationValidationContext) -> None:
                return None


def test_compound_rules_are_validated_before_dtype_registration():
    with pytest.raises(
        TypeError,
        match=r"representation_rules\[0\] must be a RepresentationRule",
    ):
        PlanarDType(
            "TestInvalidRepresentationRule",
            planes=(sw.DType.Float32, sw.DType.Int32),
            rules=(object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        sw.DType.from_name("TestInvalidRepresentationRule")


@pytest.mark.parametrize(
    ("level", "extent", "error", "match"),
    [
        (True, 2, TypeError, "level must be an integer"),
        (-1, 2, ValueError, "level must be non-negative"),
        (0, True, TypeError, "extent must be an integer or Whole"),
        (0, 0, ValueError, "extent must be positive or Whole"),
    ],
)
def test_level_extent_requires_a_valid_level_and_extent(level, extent, error, match):
    with pytest.raises(error, match=match):
        sw.LevelExtent(level, extent)


def test_level_extent_accepts_arbitrary_cute_groupings_with_uniform_preimages():
    dtype = StructuredSparsityLikeDType(
        "TestStructuredGrouping",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=2,
    )

    representation = grouped_pair(dtype)

    assert representation.logical_dtype is dtype
    assert dtype.representation_rules == (dtype.representation_rules[0],)
    assert isinstance(dtype.representation_rules[0], sw.LevelExtent)
    assert dtype.representation_rules[0].level == 0
    assert dtype.representation_rules[0].extent == 2


def test_level_extent_rejects_nonuniform_or_wrong_cardinality_groupings():
    uneven = StructuredSparsityLikeDType(
        "TestStructuredUnevenGrouping",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=2,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"TestStructuredUnevenGrouping LevelExtent rule at level 0.*"
            "overlaps targets or leaves holes"
        ),
    ):
        grouped_pair(
            uneven,
            adjacent=placement(sw.Shape([2, 2]), sw.Stride([0, 0])),
        )

    wrong_target = StructuredSparsityLikeDType(
        "TestStructuredWrongTarget",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=2,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"TestStructuredWrongTarget LevelExtent rule at level 0.*"
            "requires target cardinality 2"
        ),
    ):
        TensorRepresentation(
            wrong_target,
            (
                Subtensor(
                    sw.DType.Float32,
                    PlaneCarrier([0.0] * 4, sw.DType.Float32),
                    0,
                    placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
                ),
                Subtensor(
                    sw.DType.Int32,
                    PlaneCarrier([0] * 3, sw.DType.Int32),
                    0,
                    placement(sw.Shape(3), sw.Stride(1)),
                ),
            ),
            (placement(sw.Shape([2, 2]), sw.Stride([0, 1])),),
        )


def test_whole_level_extent_requires_one_target_coordinate():
    dtype = StructuredSparsityLikeDType(
        "TestStructuredWholeGrouping",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=sw.Whole,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"TestStructuredWholeGrouping LevelExtent rule at level 0.*"
            "complete source level to one target coordinate"
        ),
    ):
        grouped_pair(
            dtype,
            adjacent=placement(sw.Shape([2, 2]), sw.Stride([0, 0])),
        )


def test_whole_level_extent_accepts_uniform_zero_stride_grouping():
    dtype = StructuredSparsityLikeDType(
        "TestStructuredWholeAccepted",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=sw.Whole,
    )
    representation = TensorRepresentation(
        dtype,
        (
            Subtensor(
                sw.DType.Float32,
                PlaneCarrier([0.0] * 4, sw.DType.Float32),
                0,
                placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
            ),
            Subtensor(
                sw.DType.Int32,
                PlaneCarrier([0], sw.DType.Int32),
                0,
                placement(sw.Shape(1), sw.Stride(1)),
            ),
        ),
        (placement(sw.Shape([2, 2]), sw.Stride([0, 0])),),
    )

    assert representation.logical_dtype is dtype


def test_level_extent_validation_is_independent_of_tensor_cardinality():
    billion = 1_000_000_000
    dtype = StructuredSparsityLikeDType(
        "TestLargeAlgebraicGrouping",
        planes=(sw.DType.Float32, sw.DType.Int32),
        extent=billion,
    )
    representation = TensorRepresentation(
        dtype,
        (
            Subtensor(
                sw.DType.Float32,
                VirtualPlaneCarrier(billion * billion, sw.DType.Float32),
                0,
                placement(
                    sw.Shape([billion, billion]),
                    sw.Stride([1, billion]),
                ),
            ),
            Subtensor(
                sw.DType.Int32,
                VirtualPlaneCarrier(billion, sw.DType.Int32),
                0,
                placement(sw.Shape(billion), sw.Stride(1)),
            ),
        ),
        (
            placement(
                sw.Shape([billion, billion]),
                sw.Stride([0, 1]),
            ),
        ),
    )

    assert representation.logical_dtype is dtype
    source = inspect.getsource(sw.LevelExtent.validate)
    assert "uniform_preimage_extent" in source
    assert "range(" not in source
    assert "counts" not in source


def test_a_level_extent_names_a_missing_level_in_its_diagnostic():
    dtype = PlanarDType(
        "TestMissingRuleLevel",
        planes=(sw.DType.Float32, sw.DType.Int32),
        rules=(sw.LevelExtent(1, 2),),
    )
    with pytest.raises(
        ValueError,
        match="TestMissingRuleLevel LevelExtent rule at level 1 has no adjacent layout",
    ):
        grouped_pair(dtype)


def test_block_scaled_dtypes_derive_one_level_extent_per_scale_level():
    for dtype in sw.BlockScaledDType.registered():
        assert len(dtype.representation_rules) == len(dtype.levels)
        for level, (rule, descriptor) in enumerate(
            zip(dtype.representation_rules, dtype.levels, strict=True)
        ):
            assert isinstance(rule, sw.LevelExtent)
            assert (rule.level, rule.extent) == (level, descriptor.block)

    mxfp4 = sw.DType.MXFP4
    mxfp4_representation = TensorRepresentation(
        mxfp4,
        (
            Subtensor(
                sw.DType.E2M1,
                PlaneCarrier([0] * 32, sw.DType.E2M1),
                0,
                placement(sw.Shape(32), sw.Stride(1)),
            ),
            Subtensor(
                sw.DType.E8M0,
                PlaneCarrier([0], sw.DType.E8M0),
                0,
                placement(sw.Shape(1), sw.Stride(1)),
            ),
        ),
        (placement(sw.Shape(32), sw.Stride(0)),),
    )
    assert mxfp4_representation.logical_dtype is mxfp4

    nvfp4 = sw.DType.NVFP4
    nvfp4_representation = TensorRepresentation(
        nvfp4,
        (
            Subtensor(
                sw.DType.E2M1,
                PlaneCarrier([0] * 32, sw.DType.E2M1),
                0,
                placement(sw.Shape([16, 2]), sw.Stride([1, 16])),
            ),
            Subtensor(
                sw.DType.E4M3,
                PlaneCarrier([0] * 2, sw.DType.E4M3),
                0,
                placement(sw.Shape(2), sw.Stride(1)),
            ),
            Subtensor(
                sw.DType.Float32,
                PlaneCarrier([0.0], sw.DType.Float32),
                0,
                placement(sw.Shape(1), sw.Stride(1)),
            ),
        ),
        (
            placement(sw.Shape([16, 2]), sw.Stride([0, 1])),
            placement(sw.Shape(2), sw.Stride(0)),
        ),
    )
    assert nvfp4_representation.logical_dtype is nvfp4


def test_level_extent_state_survives_copy_pickle_and_dtype_discovery():
    rule = sw.LevelExtent(1, sw.Whole)
    restored = pickle.loads(pickle.dumps(rule))

    assert copy.copy(rule) is rule
    assert copy.deepcopy(rule) is rule
    assert isinstance(restored, sw.LevelExtent)
    assert restored.level == rule.level
    assert restored.extent is sw.Whole
    assert restored.structure() == rule.structure()
    assert sw.DType.MXFP4 in sw.BlockScaledDType.registered()
    built_in_rule = sw.DType.MXFP4.representation_rules[0]
    assert isinstance(built_in_rule, sw.LevelExtent)
    assert built_in_rule.extent == 32


def test_generic_representation_validation_has_no_format_subtype_branch():
    from strideweave.core import _representation

    source = inspect.getsource(_representation)
    assert "BlockScaledDType" not in source
    assert "StructuredSparsityLikeDType" not in source


def test_empty_rules_are_valid_and_rule_validation_follows_universal_checks():
    plain = PlanarDType(
        "TestEmptyRepresentationRules",
        planes=(sw.DType.Float32, sw.DType.Int32),
    )
    assert plain.representation_rules == ()
    assert grouped_pair(plain).logical_dtype is plain

    ruled = PlanarDType(
        "TestRepresentationRuleOrdering",
        planes=(sw.DType.Float32, sw.DType.Int32),
        rules=(RuleReached(),),
    )
    wrong_dtype = PlaneCarrier([0.0] * 4, sw.DType.Int32)
    with pytest.raises(ValueError, match="carrier dtype must be identical"):
        grouped_pair(ruled, element_carrier=wrong_dtype)
    with pytest.raises(RuntimeError, match="optional rule reached"):
        grouped_pair(ruled)


def test_validation_context_is_frozen_and_rules_receive_validated_shapes():
    captured: list[sw.RepresentationValidationContext] = []

    class CaptureContext(sw.RepresentationRule):
        __slots__ = ()

        def validate(self, context: sw.RepresentationValidationContext) -> None:
            captured.append(context)

    dtype = PlanarDType(
        "TestRepresentationValidationContext",
        planes=(sw.DType.Float32, sw.DType.Int32),
        rules=(TargetCardinality(2), CaptureContext()),
    )
    assert grouped_pair(dtype).logical_dtype is dtype

    context = captured[0]
    assert context.logical_dtype is dtype
    assert context.storage_dtypes == dtype.simple_types
    assert len(context.placement_layouts) == 2
    assert len(context.adjacent_layouts) == 1
    assert context.level_shapes == tuple(
        layout.shape for layout in context.placement_layouts
    )
    with pytest.raises(FrozenInstanceError):
        context.level_shapes = ()  # type: ignore[misc]


def test_validation_context_is_a_precise_public_protocol():
    import strideweave.carriers as carriers
    import strideweave.carriers.dtype as dtype

    assert (
        sw.RepresentationValidationContext is carriers.RepresentationValidationContext
    )
    assert sw.RepresentationValidationContext is dtype.RepresentationValidationContext
    assert get_type_hints(sw.RepresentationRule.validate) == {
        "context": sw.RepresentationValidationContext,
        "return": type(None),
    }

    expected = {
        "logical_dtype": sw.DType,
        "storage_dtypes": tuple[sw.DType, ...],
        "placement_layouts": tuple[sw.Layout, ...],
        "adjacent_layouts": tuple[sw.Layout, ...],
        "level_shapes": tuple[sw.Shape, ...],
    }
    actual = {
        name: get_type_hints(getattr(sw.RepresentationValidationContext, name).fget)[
            "return"
        ]
        for name in expected
    }
    assert actual == expected


def test_representation_owns_carriers_without_auto_release_and_views_may_alias():
    carrier = PlaneCarrier([1.0, 2.0], sw.DType.Float32)
    carrier_ref = weakref.ref(carrier)
    layout = placement(sw.Shape(2), sw.Stride(1))
    first = TensorRepresentation(
        sw.DType.Float32,
        (Subtensor(sw.DType.Float32, carrier, 0, layout),),
    )
    second = TensorRepresentation(
        sw.DType.Float32,
        (Subtensor(sw.DType.Float32, carrier, 0, layout),),
    )
    del carrier
    gc.collect()

    assert carrier_ref() is first.primary.carrier is second.primary.carrier
    alias = first.primary.carrier
    del first
    del second
    gc.collect()
    assert not alias.is_released()
    assert alias[1] == 2.0


def test_version_token_records_each_unique_constituent_carrier():
    dtype = PlanarDType(
        "TestRepresentationVersionToken",
        planes=(sw.DType.Float32, sw.DType.Float32),
    )
    shared = PlaneCarrier([0.0] * 6, sw.DType.Float32)
    representation = TensorRepresentation(
        dtype,
        (
            Subtensor(
                sw.DType.Float32,
                shared,
                0,
                placement(sw.Shape([2, 2]), sw.Stride([1, 2])),
            ),
            Subtensor(
                sw.DType.Float32,
                shared,
                4,
                placement(sw.Shape(2), sw.Stride(1)),
            ),
        ),
        (placement(sw.Shape([2, 2]), sw.Stride([0, 1])),),
    )
    before = representation._version_token()
    shared[0] = 1.0
    after = representation._version_token()

    assert len(before) == len(after) == 1
    assert before[0][0] == after[0][0] == id(shared)
    assert after[0][1] == before[0][1] + 1
