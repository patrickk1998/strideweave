from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import Self

import pytest

import strideweave as sw
import strideweave.carriers.tiled_evictable as tiled


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


def test_tiled_values_have_identity_preserving_public_imports() -> None:
    assert sw.TileSelection is tiled.TileSelection
    assert sw.TileSet is tiled.TileSet
    assert sw.ResidencyPlan is tiled.ResidencyPlan
    assert {"TileSelection", "TileSet", "ResidencyPlan"} <= set(sw.__all__)
    assert tiled.__all__ == [
        "AwaitProjection",
        "AwaitResidency",
        "ResidencyPlan",
        "ResidencyPolicy",
        "TileSelection",
        "TileSet",
        "TiledEvictable",
        "TiledResidencyFacet",
    ]


def test_tile_selection_preserves_axis_order_and_enumerates_first_mode_fastest() -> (
    None
):
    selection = sw.TileSelection(([7, 2], [4, 9]))

    assert selection.per_axis == ((7, 2), (4, 9))
    assert selection.coordinates == ((7, 4), (2, 4), (7, 9), (2, 9))
    assert selection.rank == 2
    assert selection.compact_grid_shape == (2, 2)


def test_tile_selection_covers_single_axis_and_every_recombined_corner() -> None:
    assert sw.TileSelection(([2, 7, 19],)).coordinates == ((2,), (7,), (19,))
    assert sw.TileSelection(([0, 1], [0, 1])).coordinates == (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    )


def test_tile_selection_materializes_each_input_once_and_owns_the_snapshot() -> None:
    first_axis = OneShot([2, 7, 19])
    second_axis = OneShot([4])
    per_axis = OneShot([first_axis, second_axis])

    selection = sw.TileSelection(per_axis)  # type: ignore[arg-type]

    assert per_axis.iterations == first_axis.iterations == second_axis.iterations == 1
    assert selection.per_axis == ((2, 7, 19), (4,))


@pytest.mark.parametrize("per_axis", [None, 3, object()])
def test_tile_selection_rejects_a_non_iterable_outer_input(per_axis: object) -> None:
    with pytest.raises(TypeError, match="per_axis must be a finite iterable"):
        sw.TileSelection(per_axis)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", [None, 3, object()])
def test_tile_selection_rejects_a_non_iterable_axis(axis: object) -> None:
    with pytest.raises(TypeError, match=r"per_axis\[0\] must be a finite iterable"):
        sw.TileSelection((axis,))  # type: ignore[arg-type]


@pytest.mark.parametrize("per_axis", [(), ((),)])
def test_tile_selection_rejects_missing_or_empty_axes(per_axis: object) -> None:
    with pytest.raises(ValueError, match=r"axis|at least one"):
        sw.TileSelection(per_axis)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [1.0, float("inf"), "1", True, None])
def test_tile_selection_rejects_non_integer_coordinates(coordinate: object) -> None:
    with pytest.raises(TypeError, match="tile coordinates must be integers"):
        sw.TileSelection(((coordinate,),))  # type: ignore[list-item]


def test_tile_selection_rejects_negative_and_duplicate_coordinates() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sw.TileSelection(((-1,),))
    with pytest.raises(ValueError, match="duplicate"):
        sw.TileSelection(((2, 2),))


def test_tile_selection_grid_binding_validates_rank_and_range() -> None:
    selection = sw.TileSelection(([2, 0], [1]))

    selection._validate_grid_shape((3, 2))
    with pytest.raises(ValueError, match="rank"):
        selection._validate_grid_shape((3,))
    with pytest.raises(ValueError, match="outside"):
        selection._validate_grid_shape((2, 2))


def test_tile_selection_converts_to_a_deterministically_ordered_tile_set() -> None:
    selection = sw.TileSelection(([7, 2], [4, 9]))

    assert selection.tile_set == sw.TileSet(selection.coordinates)
    assert selection.tile_set.coordinates == ((2, 4), (7, 4), (2, 9), (7, 9))


def test_tile_set_accepts_arbitrary_non_product_coordinates_and_sorts_them() -> None:
    tiles = sw.TileSet(((0, 1), (1, 0), (0, 0)))

    assert tiles.coordinates == ((0, 0), (1, 0), (0, 1))
    assert (1, 1) not in tiles.coordinates


def test_empty_tile_set_is_rank_agnostic() -> None:
    tiles = sw.TileSet()

    assert tiles.coordinates == ()
    tiles._validate_grid_shape((3,))
    tiles._validate_grid_shape((2, 4, 5))


@pytest.mark.parametrize("coordinates", [None, 3, object()])
def test_tile_set_rejects_a_non_iterable_outer_input(coordinates: object) -> None:
    with pytest.raises(TypeError, match="coordinates must be a finite iterable"):
        sw.TileSet(coordinates)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [None, 3, object()])
def test_tile_set_rejects_a_non_iterable_coordinate(coordinate: object) -> None:
    with pytest.raises(TypeError, match=r"coordinates\[0\] must be a finite iterable"):
        sw.TileSet((coordinate,))  # type: ignore[arg-type]


@pytest.mark.parametrize("component", [1.0, float("nan"), float("inf"), "1", True])
def test_tile_set_rejects_non_integer_components(component: object) -> None:
    with pytest.raises(TypeError, match="components must be integers"):
        sw.TileSet(((component,),))  # type: ignore[list-item]


def test_tile_set_rejects_negative_mixed_rank_and_duplicate_coordinates() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sw.TileSet(((-1,),))
    with pytest.raises(ValueError, match="same rank"):
        sw.TileSet(((0,), (0, 1)))
    with pytest.raises(ValueError, match="duplicate"):
        sw.TileSet(((0, 1), (0, 1)))


def test_tile_set_grid_binding_validates_nonempty_rank_and_range() -> None:
    tiles = sw.TileSet(((0, 1), (2, 0)))

    tiles._validate_grid_shape((3, 2))
    with pytest.raises(ValueError, match="rank"):
        tiles._validate_grid_shape((3,))
    with pytest.raises(ValueError, match="outside"):
        tiles._validate_grid_shape((2, 2))


def test_residency_plan_defaults_to_empty_sets_and_preserves_inputs() -> None:
    promote = sw.TileSet(((0,),))
    plan = sw.ResidencyPlan(promote=promote)

    assert plan.promote is promote
    assert plan.retain == sw.TileSet()
    assert plan.evict == sw.TileSet()


@pytest.mark.parametrize("field", ["promote", "retain", "evict"])
def test_residency_plan_requires_tile_set_fields(field: str) -> None:
    arguments = {field: ((0,),)}
    with pytest.raises(TypeError, match=rf"{field} must be a TileSet"):
        sw.ResidencyPlan(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [("promote", "retain"), ("promote", "evict"), ("retain", "evict")],
)
def test_residency_plan_rejects_overlap_between_any_fields(
    lhs: str,
    rhs: str,
) -> None:
    arguments = {lhs: sw.TileSet(((1,),)), rhs: sw.TileSet(((1,),))}
    with pytest.raises(ValueError, match="must be disjoint"):
        sw.ResidencyPlan(**arguments)  # type: ignore[arg-type]


def test_tiled_values_are_immutable_equal_hashable_and_repr_stable() -> None:
    selection = sw.TileSelection(([2, 7], [4]))
    same_selection = sw.TileSelection(((2, 7), (4,)))
    tiles = selection.tile_set
    plan = sw.ResidencyPlan(promote=tiles)

    assert selection == same_selection
    assert hash(selection) == hash(same_selection)
    assert repr(selection) == "TileSelection(per_axis=((2, 7), (4,)))"
    assert repr(tiles).startswith("TileSet(coordinates=")
    assert hash(plan) == hash(sw.ResidencyPlan(promote=tiles))
    with pytest.raises(FrozenInstanceError):
        selection.per_axis = ((0,),)  # type: ignore[misc]
