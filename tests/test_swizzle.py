from typing import Any, cast

import pytest

from strideweave.core.index_map import IndexMap
from strideweave.core.layout import Layout, Shape, Stride
from strideweave.core.swizzle import Swizzle, SwizzleStage


def test_swizzle_stage_preserves_positive_and_negative_field_metadata():
    positive = SwizzleStage(bits=2, base=1, shift=4)
    negative = SwizzleStage(bits=3, base=2, shift=-5)

    assert (positive.bits, positive.base, positive.shift) == (2, 1, 4)
    assert (negative.bits, negative.base, negative.shift) == (3, 2, -5)


@pytest.mark.parametrize("shift", [3, -3])
def test_swizzle_stage_accepts_fields_that_touch_without_overlapping(shift):
    stage = SwizzleStage(bits=3, base=0, shift=shift)

    assert stage.shift == shift


@pytest.mark.parametrize("bits", [0, -1])
def test_swizzle_stage_rejects_nonpositive_field_width(bits):
    with pytest.raises(ValueError, match="bits must be positive"):
        SwizzleStage(bits=bits, base=0, shift=1)


def test_swizzle_stage_rejects_a_negative_base():
    with pytest.raises(ValueError, match="base must be non-negative"):
        SwizzleStage(bits=1, base=-1, shift=1)


def test_swizzle_stage_rejects_a_zero_shift():
    with pytest.raises(ValueError, match="shift must be non-zero"):
        SwizzleStage(bits=1, base=0, shift=0)


@pytest.mark.parametrize("shift", [2, -2])
def test_swizzle_stage_rejects_overlapping_fields(shift):
    with pytest.raises(ValueError, match="fields must not overlap"):
        SwizzleStage(bits=3, base=0, shift=shift)


@pytest.mark.parametrize(
    ("bits", "base", "shift"),
    [
        (1.0, 0, 1),
        (1, 0.0, 1),
        (1, 0, 1.0),
        (True, 0, 1),
        (1, False, 1),
        (1, 0, True),
    ],
)
def test_swizzle_stage_rejects_noninteger_inputs(bits, base, shift):
    with pytest.raises(TypeError):
        SwizzleStage(bits=bits, base=base, shift=shift)


def test_swizzle_stage_rejects_invalid_constructor_arity():
    stage_type = cast(Any, SwizzleStage)

    with pytest.raises(TypeError):
        stage_type(1, 0)
    with pytest.raises(TypeError):
        stage_type(1, 0, 1, 2)


def test_swizzle_stage_rejects_ordinary_assignment_and_deletion():
    stage = SwizzleStage(bits=2, base=1, shift=-3)

    for name, value in (("bits", 1), ("base", 0), ("shift", 3)):
        with pytest.raises(AttributeError):
            setattr(stage, name, value)
        with pytest.raises(AttributeError):
            delattr(stage, name)

    assert (stage.bits, stage.base, stage.shift) == (2, 1, -3)


def test_zero_stage_swizzle_is_identity_including_for_the_one_point_domain():
    scalar_identity = Swizzle(Shape())
    identity = Swizzle(Shape(8))

    assert scalar_identity.stages == ()
    assert scalar_identity(0) == 0
    assert scalar_identity(()) == 0
    assert scalar_identity([]) == 0
    assert [identity(index) for index in range(identity.size)] == list(range(8))


def test_positive_swizzle_stage_xors_the_upper_field_into_the_lower_field():
    swizzle = Swizzle(Shape(16), SwizzleStage(2, 0, 2))

    assert swizzle(0b1101) == 0b1110


def test_negative_swizzle_stage_xors_the_lower_field_into_the_upper_field():
    swizzle = Swizzle(Shape(16), SwizzleStage(2, 0, -2))

    assert swizzle(0b1101) == 0b1001


@pytest.mark.parametrize(
    "stage",
    [
        SwizzleStage(bits=2, base=1, shift=3),
        SwizzleStage(bits=2, base=1, shift=-3),
    ],
)
def test_one_stage_swizzle_is_involutive_and_changes_only_the_destination(stage):
    swizzle = Swizzle(Shape(64), stage)
    destination_mask = ((1 << stage.bits) - 1) << stage.base
    source_mask = destination_mask
    if stage.shift > 0:
        source_mask <<= stage.shift
    else:
        destination_mask <<= -stage.shift
    preserved_mask = (swizzle.size - 1) ^ destination_mask

    for index in range(swizzle.size):
        result = swizzle(index)

        assert swizzle(result) == index
        assert result & source_mask == index & source_mask
        assert result & preserved_mask == index & preserved_mask


def test_swizzle_uses_shared_hierarchical_key_normalization():
    swizzle = Swizzle(Shape(2, [2, 2]), SwizzleStage(1, 0, 1))

    for key in (6, (0, 3), (0, (1, 1)), [0, [1, 1]]):
        assert swizzle.index(key) == 7
        assert swizzle(key) == 7


def test_swizzle_copies_stages_into_immutable_containment():
    stage = SwizzleStage(1, 0, 1)
    stages = [stage]
    swizzle = Swizzle(Shape(4), *stages)

    stages.clear()

    assert swizzle.stages == (stage,)
    assert isinstance(swizzle.stages, tuple)
    with pytest.raises(AttributeError, match="immutable"):
        swizzle.stages = ()  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del swizzle.stages  # type: ignore[misc]


def test_swizzle_reports_exact_index_map_metadata():
    shape = Shape(4)
    swizzle = Swizzle(shape, SwizzleStage(1, 0, 1))

    assert isinstance(swizzle, IndexMap)
    assert swizzle.shape is shape
    assert swizzle.size == 4
    assert swizzle.codomain_size == 4
    assert swizzle.is_injective is True


def test_swizzle_participates_in_generic_composition_fallback():
    outer = Swizzle(Shape(4), SwizzleStage(1, 0, 1))
    inner = Layout(Shape(2), Stride(2))

    result = outer.compose(inner)

    assert isinstance(result, IndexMap)
    assert not isinstance(result, Swizzle)
    assert result.shape == inner.shape
    assert result.codomain_size == outer.codomain_size
    assert [result(index) for index in range(result.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_equal_size_swizzle_composition_keeps_inner_before_outer_stage_order():
    lower = SwizzleStage(bits=1, base=0, shift=1)
    upper = SwizzleStage(bits=1, base=1, shift=1)
    inner = Swizzle(Shape(8), lower)
    outer = Swizzle(Shape(8), upper)

    result = outer.compose(inner)

    assert isinstance(result, Swizzle)
    assert result.stages == (lower, upper)
    assert result(0b100) == 0b110
    assert [result(index) for index in range(result.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_swizzle_composition_cancels_only_adjacent_equal_stages():
    lower = SwizzleStage(bits=1, base=0, shift=1)
    upper = SwizzleStage(bits=1, base=1, shift=1)

    cancelled = Swizzle(Shape(8), upper, lower).compose(Swizzle(Shape(8), lower, upper))
    separated = Swizzle(Shape(8), lower).compose(Swizzle(Shape(8), lower, upper))

    assert isinstance(cancelled, Swizzle)
    assert cancelled.stages == ()
    assert [cancelled(index) for index in range(8)] == list(range(8))
    assert isinstance(separated, Swizzle)
    assert separated.stages == (lower, upper, lower)


def test_equal_size_swizzle_composition_preserves_the_inner_hierarchical_shape():
    stage = SwizzleStage(bits=1, base=0, shift=1)
    inner = Swizzle(Shape([2, 2]), stage)
    outer = Swizzle(Shape(4), SwizzleStage(bits=1, base=0, shift=-1))

    result = outer.compose(inner)

    assert isinstance(result, Swizzle)
    assert result.shape == inner.shape
    for coordinate in ((1, 0), (1, 1), 2, 3):
        assert result(coordinate) == outer(inner(coordinate))


def test_different_size_swizzles_use_lazy_generic_composition():
    inner = Swizzle(Shape(4), SwizzleStage(bits=1, base=0, shift=1))
    outer = Swizzle(Shape(8), SwizzleStage(bits=1, base=1, shift=1))

    result = outer.compose(inner)

    assert isinstance(result, IndexMap)
    assert not isinstance(result, Swizzle)
    assert result.shape == inner.shape
    assert result.codomain_size == outer.codomain_size
    assert [result(index) for index in range(result.size)] == [
        outer(inner(index)) for index in range(inner.size)
    ]


def test_swizzle_rejects_an_omitted_or_non_shape_domain():
    swizzle_type = cast(Any, Swizzle)

    with pytest.raises(TypeError):
        swizzle_type()
    with pytest.raises(TypeError, match="shape must be a Shape"):
        swizzle_type(object())


def test_swizzle_rejects_a_non_stage_argument():
    swizzle_type = cast(Any, Swizzle)

    with pytest.raises(TypeError, match="every stage must be a SwizzleStage"):
        swizzle_type(Shape(4), object())


def test_swizzle_rejects_a_non_power_of_two_domain():
    with pytest.raises(ValueError, match="domain size must be a power of two"):
        Swizzle(Shape(3))


@pytest.mark.parametrize("shift", [2, -2])
def test_swizzle_accepts_a_stage_that_exactly_fits_the_domain_bit_width(shift):
    stage = SwizzleStage(bits=2, base=0, shift=shift)

    swizzle = Swizzle(Shape(16), stage)

    assert swizzle.stages == (stage,)


@pytest.mark.parametrize("shift", [2, -2])
def test_swizzle_rejects_a_stage_that_exceeds_the_domain_by_one_bit(shift):
    stage = SwizzleStage(bits=2, base=1, shift=shift)

    with pytest.raises(ValueError, match="exceed the domain bit width"):
        Swizzle(Shape(16), stage)


def test_swizzle_applies_noncommuting_stages_in_argument_order():
    lower = SwizzleStage(bits=1, base=0, shift=1)
    upper = SwizzleStage(bits=1, base=1, shift=1)
    lower_then_upper = Swizzle(Shape(8), lower, upper)
    upper_then_lower = Swizzle(Shape(8), upper, lower)

    assert lower_then_upper(0b100) == 0b110
    assert upper_then_lower(0b100) == 0b111


def test_swizzle_rejects_ordinary_metadata_assignment_and_deletion():
    swizzle = Swizzle(Shape(4), SwizzleStage(1, 0, 1))

    for name, value in (
        ("shape", Shape(1)),
        ("size", 1),
        ("codomain_size", 1),
        ("is_injective", False),
        ("stages", ()),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(swizzle, name, value)
        with pytest.raises(AttributeError, match="immutable"):
            delattr(swizzle, name)

    assert swizzle.shape == Shape(4)
    assert swizzle.codomain_size == 4
    assert swizzle.is_injective is True
    assert swizzle.stages == (SwizzleStage(1, 0, 1),)
