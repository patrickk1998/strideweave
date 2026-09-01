"""Contract tests for TiledEvictable construction and tile residency."""

from __future__ import annotations

import ast
import gc
import inspect
import threading
import time
import weakref
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave import DType, Generic
from strideweave.carriers import extension
from strideweave.carriers.tiled_evictable import (
    AwaitResidency,
    ResidencyPlan,
    ResidencyPolicy,
    TiledEvictable,
    TiledResidencyFacet,
    TileSelection,
    TileSet,
)
from strideweave.carriers.tiled_evictable.carrier import _TileState

GRID = (2, 2)
TILE = (2, 2)
SLOTS = 16


def _carrier(values: list[float] | None = None) -> Generic:
    return Generic([] if values is None else values, dtype=DType.Float32)


def _tiled(
    *,
    primary_values: list[float] | None = None,
    secondary_values: list[float] | None = None,
    residency_policy: ResidencyPolicy | None = None,
) -> TiledEvictable:
    return TiledEvictable(
        _carrier(primary_values),
        _carrier(secondary_values),
        GRID,
        TILE,
        residency_policy=residency_policy,
    )


def _all_values() -> list[float]:
    return [float(index) for index in range(SLOTS)]


def _state(carrier: TiledEvictable, coordinate: tuple[int, int]) -> _TileState:
    return carrier.tile_state(coordinate)


def test_tiled_exports_and_facet_identity() -> None:
    import strideweave.carriers.tiled_evictable as tiled

    assert sw.TiledEvictable is tiled.TiledEvictable
    assert sw.AwaitResidency is tiled.AwaitResidency
    assert sw.ResidencyPolicy is tiled.ResidencyPolicy
    assert sw.TiledResidencyFacet is tiled.TiledResidencyFacet
    assert _tiled().require_facet(TiledResidencyFacet) is not None


def test_construction_exposes_flat_extent_and_canonical_geometry() -> None:
    carrier = _tiled(primary_values=_all_values())

    assert carrier.grid_shape == GRID
    assert carrier.tile_shape == TILE
    assert carrier.logical_shape == (4, 4)
    assert carrier.size() == SLOTS
    assert carrier.dtype() is DType.Float32
    assert carrier.primary.size() == SLOTS
    assert carrier.secondary.size() == 0
    assert carrier.primary_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert carrier.secondary_tiles == TileSet()
    assert carrier.valid_tiles == carrier.primary_tiles
    assert carrier.dirty_tiles == TileSet()


@pytest.mark.parametrize(
    ("primary_factory", "secondary_factory", "supported"),
    [
        (
            lambda: sw.Generic([0.0], dtype=DType.Float32),
            lambda: sw.Generic([], dtype=DType.Float32),
            (DType.Float32, DType.Int32, DType.Bool),
        ),
        (
            lambda: sw.CPU(1, dtype=DType.Float32),
            lambda: sw.FileBacked(dtype=DType.Float32),
            (DType.Float32, DType.Int32),
        ),
        (
            lambda: sw.Generic([0.0], dtype=DType.Float32),
            lambda: sw.FileBacked(dtype=DType.Float32),
            (DType.Float32, DType.Int32),
        ),
    ],
    ids=("generic-generic", "cpu-file-backed", "generic-file-backed"),
)
def test_storage_dtype_support_is_the_state_independent_tier_intersection(
    primary_factory: Callable[[], sw.Carrier],
    secondary_factory: Callable[[], sw.Carrier],
    supported: tuple[DType, ...],
) -> None:
    carrier = TiledEvictable(primary_factory(), secondary_factory(), (1,), (1,))
    registered = DType.registered()
    expected = tuple(
        any(dtype is supported_dtype for supported_dtype in supported)
        for dtype in registered
    )

    def query_without_storage_work() -> tuple[bool, ...]:
        before = (
            carrier.version,
            carrier.primary.size(),
            carrier.secondary.size(),
            carrier.primary_tiles,
            carrier.secondary_tiles,
            carrier.valid_tiles,
            carrier.dirty_tiles,
        )
        answers = tuple(carrier.supports_storage_dtype(dtype) for dtype in registered)
        after = (
            carrier.version,
            carrier.primary.size(),
            carrier.secondary.size(),
            carrier.primary_tiles,
            carrier.secondary_tiles,
            carrier.valid_tiles,
            carrier.dirty_tiles,
        )
        assert after == before
        return answers

    assert query_without_storage_work() == expected
    with pytest.raises(TypeError, match="dtype must be a DType"):
        carrier.supports_storage_dtype(object())  # type: ignore[arg-type]

    only_tile = TileSet(((0,),))
    carrier.evict(only_tile)
    assert query_without_storage_work() == expected
    carrier.promote(only_tile)
    assert query_without_storage_work() == expected

    carrier.release()
    release_state = (
        carrier.version,
        carrier.is_released(),
        carrier.primary.is_released(),
        carrier.secondary.is_released(),
    )
    assert (
        tuple(carrier.supports_storage_dtype(dtype) for dtype in registered) == expected
    )
    assert (
        carrier.version,
        carrier.is_released(),
        carrier.primary.is_released(),
        carrier.secondary.is_released(),
    ) == release_state


def test_tile_state_reports_location_validity_and_dirty_metadata() -> None:
    carrier = _tiled(primary_values=_all_values())

    initial = _state(carrier, (0, 0))
    assert initial.location == "primary"
    assert initial.valid is True
    assert initial.dirty is False

    carrier[0] = 99.0
    assert carrier.tile_state((0, 0)).dirty is True
    carrier.evict(TileSet(((0, 0),)))
    evicted = _state(carrier, (0, 0))
    assert evicted.location == "secondary"
    assert evicted.valid is True
    assert evicted.dirty is False


@pytest.mark.parametrize(
    ("primary", "secondary", "grid", "tile", "error"),
    [
        (object(), _carrier(), GRID, TILE, TypeError),
        (_carrier(), object(), GRID, TILE, TypeError),
        (_carrier([1.0]), Generic([0], dtype=DType.Int32), GRID, TILE, TypeError),
        (_carrier([1.0]), _carrier(), GRID, TILE, ValueError),
        (_carrier(), _carrier(), (2,), TILE, ValueError),
        (_carrier(), _carrier(), GRID, (0, 2), ValueError),
    ],
)
def test_construction_validation_is_preflight_and_atomic(
    primary: object,
    secondary: object,
    grid: tuple[int, ...],
    tile: tuple[int, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TiledEvictable(primary, secondary, grid, tile)  # type: ignore[arg-type]

    for candidate in (primary, secondary):
        if isinstance(candidate, sw.Carrier):
            assert not candidate.is_owned()


def test_empty_tiers_start_with_invalid_tiles() -> None:
    carrier = _tiled()

    assert carrier.primary_tiles == TileSet()
    assert carrier.secondary_tiles == TileSet()
    assert carrier.valid_tiles == TileSet()
    assert carrier.dirty_tiles == TileSet()
    with pytest.raises(RuntimeError, match=r"invalid|resident|promot"):
        carrier[0]
    with pytest.raises(RuntimeError, match=r"invalid|initialization"):
        carrier[0] = 1.0
    assert carrier.valid_tiles == TileSet()


def test_promote_and_evict_async_have_blocking_equivalents() -> None:
    carrier = _tiled(primary_values=_all_values())
    selected = TileSet(((0, 1), (1, 0)))

    carrier.evict(selected)
    assert carrier.primary_tiles == TileSet(((0, 0), (1, 1)))
    assert carrier.secondary_tiles == selected
    assert carrier.valid_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))

    pending = carrier.promote_async(selected)
    assert isinstance(pending, AwaitResidency)
    assert pending.wait() is None
    assert carrier.primary_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert carrier.secondary_tiles == TileSet()
    assert carrier.promote(selected) is None
    assert carrier.evict(selected) is None
    assert carrier.evict(selected) is None


def test_selection_is_accepted_for_residency_without_changing_its_order() -> None:
    carrier = _tiled(primary_values=_all_values())
    selection = TileSelection(([1, 0], [1]))

    carrier.evict_async(selection).wait()

    assert carrier.secondary_tiles == selection.tile_set
    assert selection.coordinates == ((1, 1), (0, 1))


def test_dirty_values_survive_partial_eviction_and_promotion() -> None:
    carrier = _tiled(primary_values=_all_values())
    carrier[0] = 123.0
    dirty = TileSet(((0, 0),))
    assert carrier.dirty_tiles == dirty

    carrier.evict(dirty)
    assert carrier.secondary_tiles == dirty
    assert carrier.dirty_tiles == TileSet()
    carrier.promote(dirty)
    assert carrier[0] == 123.0
    assert carrier.valid_tiles == carrier.primary_tiles


def test_multiaxis_mapping_round_trips_every_value_through_arbitrary_residency() -> (
    None
):
    expected = _all_values()
    carrier = _tiled(primary_values=expected)
    selected = TileSet(((0, 0), (1, 0), (1, 1)))

    carrier.evict(selected)
    assert [carrier[index] for index in range(SLOTS)] == expected
    carrier.promote(TileSet(((1, 0),)))
    assert [carrier[index] for index in range(SLOTS)] == expected
    assert carrier.primary_tiles == TileSet(((0, 1), (1, 0)))
    assert carrier.secondary_tiles == TileSet(((0, 0), (1, 1)))


def test_implicit_zero_tiles_materialize_on_promotion_without_version_change() -> None:
    template = _tiled(primary_values=_all_values())
    carrier = cast(Any, template)._implicit_zero_like()
    selected = TileSet(((0, 0), (1, 1)))
    version = carrier.version

    assert carrier.valid_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert carrier.primary_tiles == TileSet()
    assert carrier.tile_state((0, 0)).location == "implicit"

    carrier.promote(selected)

    assert carrier.primary_tiles == selected
    assert carrier.tile_state((0, 0)).location == "primary"
    assert carrier[0] == 0.0
    assert carrier[15] == 0.0
    assert carrier.version == version


def test_residency_validation_does_not_change_state() -> None:
    carrier = _tiled(primary_values=_all_values())
    before = (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.version,
    )

    with pytest.raises(ValueError, match=r"range|grid|coordinate"):
        carrier.evict(TileSet(((2, 0),)))
    with pytest.raises(TypeError):
        carrier.promote(object())  # type: ignore[arg-type]

    assert (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.version,
    ) == before


def test_overlapping_incomplete_residency_request_is_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = _tiled(primary_values=_all_values())
    gate = threading.Event()
    original = cast(Any, carrier)._run_transition

    def blocked(_self: TiledEvictable, *args: object, **kwargs: object) -> None:
        gate.wait(timeout=5)
        original(*args, **kwargs)

    monkeypatch.setattr(type(carrier), "_run_transition", blocked)
    first = carrier.evict_async(TileSet(((0, 0),)))
    assert not first.done
    with pytest.raises(RuntimeError, match=r"wait|conflict|pending"):
        carrier.promote_async(TileSet(((0, 0),)))
    with pytest.raises(RuntimeError, match=r"wait|pending"):
        carrier.release()
    assert not carrier.is_released()
    gate.set()
    first.wait()


def test_request_remains_pending_until_its_handle_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = _tiled(primary_values=_all_values())
    tile = TileSet(((0, 0),))
    committed = threading.Event()
    finish = threading.Event()
    original = cast(Any, carrier)._commit_transition

    def block_after_commit(
        _self: TiledEvictable, *args: object, **kwargs: object
    ) -> None:
        original(*args, **kwargs)
        committed.set()
        finish.wait(timeout=5)

    monkeypatch.setattr(type(carrier), "_commit_transition", block_after_commit)
    pending = carrier.evict_async(tile)
    assert committed.wait(timeout=5)
    try:
        assert carrier.secondary_tiles == tile
        assert not pending.done
        with pytest.raises(RuntimeError, match=r"wait|pending"):
            carrier.release()
        with pytest.raises(RuntimeError, match=r"wait|conflict|pending"):
            carrier.promote_async(tile)
    finally:
        finish.set()
    pending.wait()


def test_disjoint_residency_requests_can_stage_concurrently() -> None:
    carrier = _tiled(primary_values=_all_values())
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    handles: list[AwaitResidency] = []

    def submit(tile: tuple[int, int]) -> None:
        try:
            barrier.wait(timeout=5)
            handles.append(carrier.evict_async(TileSet((tile,))))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=submit, args=((0, 0),)),
        threading.Thread(target=submit, args=((1, 1),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    for handle in handles:
        handle.wait()

    assert errors == []
    assert carrier.secondary_tiles == TileSet(((0, 0), (1, 1)))


def test_many_tile_residency_uses_bounded_workers_and_survives_dropped_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background = import_module("strideweave.carriers._background")
    async_move = import_module("strideweave.carriers.move.async_move")
    tile_count = 32
    carrier = TiledEvictable(
        Generic([float(index) for index in range(tile_count)], dtype=DType.Float32),
        Generic([], dtype=DType.Float32),
        (tile_count,),
        (1,),
    )
    requested = TileSet((index,) for index in range(tile_count))
    gate = threading.Event()
    started = 0
    started_condition = threading.Condition()
    staged: list[tuple[int, ...]] = []
    original_run = cast(Any, async_move)._run_legacy_copy
    original_stage = cast(Any, carrier)._stage_transfer

    def delayed_run(**kwargs: object) -> None:
        nonlocal started
        with started_condition:
            started += 1
            started_condition.notify_all()
        gate.wait(timeout=5)
        original_run(**kwargs)

    def observe_stage(
        _self: TiledEvictable,
        coordinate: tuple[int, ...],
        source: object,
        target: object,
        route: object,
    ) -> object:
        staged.append(coordinate)
        return original_stage(coordinate, source, target, route)

    monkeypatch.setattr(async_move, "_run_legacy_copy", delayed_run)
    monkeypatch.setattr(type(carrier), "_stage_transfer", observe_stage)
    pending = carrier.evict_async(requested)

    assert staged == list(requested.coordinates)
    with started_condition:
        assert started_condition.wait_for(
            lambda: (
                started == cast(Any, background)._MOVE_BACKGROUND_EXECUTOR.max_workers
            ),
            timeout=2,
        )
    assert not pending.done
    move_workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(
            cast(Any, background)._MOVE_BACKGROUND_EXECUTOR.thread_name_prefix
        )
    ]
    residency_workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(
            cast(Any, background)._TILED_RESIDENCY_EXECUTOR.thread_name_prefix
        )
    ]
    assert (
        len(move_workers) <= cast(Any, background)._MOVE_BACKGROUND_EXECUTOR.max_workers
    )
    assert (
        len(residency_workers)
        <= cast(Any, background)._TILED_RESIDENCY_EXECUTOR.max_workers
    )

    reference = weakref.ref(pending)
    del pending
    gc.collect()
    assert reference() is None
    gate.set()
    deadline = time.monotonic() + 5
    while cast(Any, carrier)._pending_requests and time.monotonic() < deadline:
        time.sleep(0.001)

    assert cast(Any, carrier)._pending_requests == 0
    assert carrier.secondary_tiles == requested
    carrier.promote(requested)
    assert carrier.primary_tiles == requested


def test_failed_multi_tile_request_restores_the_complete_prior_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = _tiled(primary_values=_all_values())
    requested = TileSet(((0, 0), (1, 0)))
    before = (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.dirty_tiles,
        tuple(carrier[index] for index in range(SLOTS)),
        carrier.version,
    )
    original = cast(Any, carrier)._stage_transfer

    def fail_second(
        _self: TiledEvictable,
        coordinate: tuple[int, ...],
        source: object,
        target: object,
        route: object,
    ) -> object:
        if coordinate == (1, 0):
            raise RuntimeError("injected staging failure")
        return original(coordinate, source, target, route)

    monkeypatch.setattr(type(carrier), "_stage_transfer", fail_second)
    pending = carrier.evict_async(requested)
    with pytest.raises(RuntimeError, match="injected staging failure"):
        pending.wait()

    assert (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.dirty_tiles,
        tuple(carrier[index] for index in range(SLOTS)),
        carrier.version,
    ) == before
    assert not carrier.is_owned()


def test_multi_tile_completion_failure_rolls_back_and_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = _tiled(primary_values=_all_values())
    requested = TileSet(((0, 0), (1, 0)))
    before = (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.dirty_tiles,
        tuple(carrier[index] for index in range(SLOTS)),
        carrier.version,
    )
    original = cast(Any, carrier)._stage_transfer
    injected = False

    class FailingCompletion:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def wait(self) -> object:
            cast(Any, self._handle).wait()
            raise RuntimeError("injected completion failure")

    def fail_one_completion(
        _self: TiledEvictable,
        coordinate: tuple[int, ...],
        source: object,
        target: object,
        route: object,
    ) -> object:
        nonlocal injected
        transfer = original(coordinate, source, target, route)
        if not injected and coordinate == (1, 0):
            injected = True
            transfer.handle = FailingCompletion(transfer.handle)
        return transfer

    monkeypatch.setattr(type(carrier), "_stage_transfer", fail_one_completion)
    pending = carrier.evict_async(requested)
    with pytest.raises(RuntimeError, match="injected completion failure"):
        pending.wait()

    assert (
        carrier.primary_tiles,
        carrier.secondary_tiles,
        carrier.valid_tiles,
        carrier.dirty_tiles,
        tuple(carrier[index] for index in range(SLOTS)),
        carrier.version,
    ) == before
    carrier.evict(requested)
    assert carrier.secondary_tiles == requested


def test_policy_is_structural_and_receives_immutable_requirements() -> None:
    calls: list[tuple[TileSet, str]] = []

    class Policy:
        def plan(self, required: TileSet, purpose: str) -> ResidencyPlan:
            calls.append((required, purpose))
            assert isinstance(required, TileSet)
            return ResidencyPlan(promote=required)

    policy = Policy()
    carrier = _tiled(primary_values=_all_values(), residency_policy=policy)
    carrier.promote(TileSet(((1, 1),)))
    assert carrier.residency_policy is policy
    assert calls == []  # explicit residency transitions do not execute a work request


def test_policy_promote_retain_and_evict_order_around_pinned_work() -> None:
    calls: list[tuple[TileSet, str]] = []
    required = TileSet(((1, 1),))
    promote = TileSet(((0, 0),))
    retain = TileSet(((1, 0),))
    evict = TileSet(((0, 1),))

    class Policy:
        def plan(self, required: TileSet, purpose: str) -> ResidencyPlan:
            calls.append((required, purpose))
            return ResidencyPlan(promote=promote, retain=retain, evict=evict)

    carrier = _tiled(primary_values=_all_values(), residency_policy=Policy())
    carrier.evict(promote)
    version = carrier.version

    with cast(Any, carrier)._pin_tiles_for_work(required, "operation") as plan:
        assert plan == ResidencyPlan(promote=promote, retain=retain, evict=evict)
        assert promote.coordinates[0] in carrier.primary_tiles.coordinates
        assert evict.coordinates[0] in carrier.primary_tiles.coordinates
        with pytest.raises(RuntimeError, match="pinned"):
            carrier.evict(retain)
        with pytest.raises(RuntimeError, match="pinned"):
            carrier.evict(required)

    assert calls == [(required, "operation")]
    assert evict.coordinates[0] in carrier.secondary_tiles.coordinates
    assert carrier.version == version


def test_work_pins_required_tiles_before_internal_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = TileSet(((0, 0),))
    carrier = _tiled(primary_values=_all_values())
    carrier.evict(required)
    original = cast(Any, carrier)._transition_async
    observed = False

    def observe_pins(
        _self: TiledEvictable,
        kind: str,
        tiles: object,
        *,
        pinned_owner: set[tuple[int, ...]] | None = None,
    ) -> AwaitResidency:
        nonlocal observed
        if kind == "promote" and pinned_owner:
            observed = True
            with pytest.raises(RuntimeError, match="pinned"):
                carrier.promote(required)
        return cast(
            AwaitResidency,
            original(kind, tiles, pinned_owner=pinned_owner),
        )

    monkeypatch.setattr(type(carrier), "_transition_async", observe_pins)
    with cast(Any, carrier)._pin_tiles_for_work(required, "projection"):
        assert carrier.primary_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert observed


def test_no_policy_promotes_required_values_for_work() -> None:
    required = TileSet(((0, 0),))
    carrier = _tiled(primary_values=_all_values())
    carrier.evict(required)

    with cast(Any, carrier)._pin_tiles_for_work(required, "projection") as plan:
        assert plan == ResidencyPlan(promote=required)
        assert carrier.primary_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))


@pytest.mark.parametrize(
    "result", [object(), ResidencyPlan(promote=TileSet(((2, 0),)))]
)
def test_invalid_policy_result_fails_before_residency_work(result: object) -> None:
    class Policy:
        def plan(self, required: TileSet, purpose: str) -> object:
            del required, purpose
            return result

    carrier = _tiled(primary_values=_all_values(), residency_policy=cast(Any, Policy()))
    required = TileSet(((0, 0),))
    before = (carrier.primary_tiles, carrier.secondary_tiles, carrier.version)

    with pytest.raises((TypeError, ValueError)):
        with cast(Any, carrier)._pin_tiles_for_work(required, "backward"):
            raise AssertionError("invalid policy must not begin work")

    assert (carrier.primary_tiles, carrier.secondary_tiles, carrier.version) == before


def test_policy_exception_propagates_before_residency_work() -> None:
    class Policy:
        def plan(self, required: TileSet, purpose: str) -> ResidencyPlan:
            del required, purpose
            raise LookupError("policy failed")

    carrier = _tiled(primary_values=_all_values(), residency_policy=Policy())
    before = (carrier.primary_tiles, carrier.secondary_tiles, carrier.version)

    with pytest.raises(LookupError, match="policy failed"):
        with cast(Any, carrier)._pin_tiles_for_work(TileSet(((0, 0),)), "scatter"):
            raise AssertionError("failing policy must not begin work")

    assert (carrier.primary_tiles, carrier.secondary_tiles, carrier.version) == before


def test_invalid_policy_is_rejected_before_construction() -> None:
    with pytest.raises(TypeError):
        _tiled(residency_policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        _tiled(residency_policy=cast(Any, type("BadPolicy", (), {"plan": 1})()))


def test_release_is_idempotent_and_blocks_future_residency() -> None:
    carrier = _tiled(primary_values=_all_values())
    primary = carrier.primary
    secondary = carrier.secondary
    assert primary.is_owned()
    assert secondary.is_owned()
    carrier.evict(TileSet(((0, 0),)))
    assert carrier.release() is None
    assert carrier.release() is None
    assert carrier.is_released()
    assert primary.is_released() and not primary.is_owned()
    assert secondary.is_released() and not secondary.is_owned()
    with pytest.raises(RuntimeError):
        carrier.promote(TileSet(((0, 0),)))
    with pytest.raises(RuntimeError):
        carrier.evict(TileSet(((0, 0),)))


def test_definition_backing_is_unique_and_not_concrete_backend_branching() -> None:
    module = cast(ModuleType, inspect.getmodule(TiledEvictable))
    source = inspect.getsource(module)
    assert "CarrierDefinition" in source
    assert "isinstance(primary, (CPU" not in source
    assert "isinstance(primary, (Generic" not in source
    assert TiledEvictable is not Generic
    assert extension._peek_carrier_definition(TiledEvictable) is not None
    for shipped in (
        sw.Generic,
        sw.CPU,
        sw.Metal,
        sw.FileBacked,
        sw.BlockDeviceCarrier,
        sw.Evictable,
    ):
        assert extension._peek_carrier_definition(shipped) is None

    forbidden = {
        "Generic",
        "CPU",
        "Metal",
        "FileBacked",
        "BlockDeviceCarrier",
        "Evictable",
    }
    package = Path(inspect.getfile(TiledEvictable)).parent
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        concrete_references = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert concrete_references.isdisjoint(forbidden), source_path

    definition = extension._peek_carrier_definition(TiledEvictable)
    assert definition is not None
    with pytest.raises(TypeError, match=r"already observed|already has"):
        sw.register_carrier_definition(TiledEvictable, definition)


def test_factories_preserve_geometry_tier_kinds_policy_and_validity() -> None:
    class Policy:
        def plan(self, required: TileSet, purpose: str) -> ResidencyPlan:
            del purpose
            return ResidencyPlan(promote=required)

    policy = Policy()
    template = _tiled(primary_values=_all_values(), residency_policy=policy)

    initialized = template.new_like([7.0] * SLOTS, mutable=False)
    assert initialized.grid_shape == GRID
    assert initialized.tile_shape == TILE
    assert type(initialized.primary) is type(template.primary)
    assert type(initialized.secondary) is type(template.secondary)
    assert initialized.residency_policy is policy
    assert initialized.valid_tiles == initialized.primary_tiles
    assert not initialized.is_mutable()
    assert tuple(initialized[index] for index in range(SLOTS)) == (7.0,) * SLOTS

    invalid = template.allocate_like(SLOTS, empty=True)
    assert invalid.valid_tiles == TileSet()
    zeroed = template.allocate_like(SLOTS)
    assert zeroed.valid_tiles == TileSet(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert zeroed.primary_tiles == TileSet()
    assert tuple(zeroed[index] for index in range(SLOTS)) == (0.0,) * SLOTS
