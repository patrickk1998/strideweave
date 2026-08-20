from __future__ import annotations

from collections.abc import Mapping

import pytest
import synthetic_evidence
from evidence_doubles import RecordingEvidenceStore

import strideweave.verification.stage_two as stage_two_module
from strideweave.verification import (
    VerificationOutcome,
    VerificationReport,
    verification_profiles,
)
from strideweave.verification.store import (
    DoltEvidenceStore,
    SQLStatement,
    VerificationStoreError,
    query_stale,
    query_status,
    query_todo,
)
from strideweave.verification.store._identity import _todo_provenance_digest
from strideweave.verification.store.querying import _differences
from strideweave.verification.store.recording import _record_validated_report


class _QueryStore(RecordingEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[SQLStatement] = []

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        self.queries.append(statement)
        return ()


class _TodoStore(RecordingEvidenceStore):
    def __init__(self, report, observed_requirement_ids) -> None:
        super().__init__()
        self.report = report
        self.observed_requirement_ids = tuple(observed_requirement_ids)
        self.queries: list[SQLStatement] = []

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        self.queries.append(statement)
        if statement.parameters != (_todo_provenance_digest(self.report),):
            return ()
        return tuple(
            {"requirement_id": requirement_id}
            for requirement_id in self.observed_requirement_ids
        )


def test_staleness_explains_jit_axes_without_dataclass_encoding_failure() -> None:
    current = synthetic_evidence.synthetic_report(specialization=256)
    stored = synthetic_evidence.synthetic_report(specialization=128)
    axes = {item.axis for item in _differences(stored, current)}
    assert {"specialization", "generated_source", "executable_artifact"} <= axes


def test_staleness_reports_current_report_as_current() -> None:
    report = synthetic_evidence.synthetic_report()
    assert not _differences(report, report)


def test_todo_provenance_digest_uses_only_stable_matching_axes() -> None:
    report = synthetic_evidence.synthetic_report()
    digest = _todo_provenance_digest(report)
    assert digest == _todo_provenance_digest(
        synthetic_evidence.contradicting_report(report)
    )
    for outcome in (
        VerificationOutcome.FAILED,
        VerificationOutcome.ERROR,
        VerificationOutcome.BLOCKED,
    ):
        assert digest == _todo_provenance_digest(
            synthetic_evidence.target_outcome_report(report, outcome)
        )
    deferred = synthetic_evidence.target_outcome_report(
        report, VerificationOutcome.DEFERRED
    )
    assert _todo_provenance_digest(deferred) == _todo_provenance_digest(
        synthetic_evidence.target_outcome_report(
            report, VerificationOutcome.DEFERRED, alternate=True
        )
    )
    assert {
        _todo_provenance_digest(synthetic_evidence.provenance_variant(axis))
        for axis in ("verification", "tolerance", "oracle")
    }.isdisjoint({digest})
    assert digest != _todo_provenance_digest(
        synthetic_evidence.synthetic_report(specialization=256)
    )
    assert digest != _todo_provenance_digest(synthetic_evidence.cpu_report())


@pytest.mark.parametrize("profile", ["", None])
def test_query_selectors_require_target_profile(profile) -> None:
    with pytest.raises(VerificationStoreError):
        query_status(RecordingEvidenceStore(), selected_target_profile=profile)  # type: ignore[arg-type]


def test_status_rejects_an_unknown_profile_before_store_query() -> None:
    store = _QueryStore()

    with pytest.raises(VerificationStoreError, match="unknown verification profile"):
        query_status(store, selected_target_profile="missing-profile")

    assert store.queries == []


@pytest.mark.parametrize(
    "profile_id", tuple(profile.profile_id for profile in verification_profiles())
)
def test_status_accepts_every_registered_profile_without_resolving_a_runtime(
    monkeypatch: pytest.MonkeyPatch, profile_id: str
) -> None:
    store = _QueryStore()

    def fail_if_resolved(*args, **kwargs):
        del args, kwargs
        raise AssertionError("status query resolved a provider runtime")

    monkeypatch.setattr(stage_two_module, "_target_runtime", fail_if_resolved)

    assert not query_status(store, selected_target_profile=profile_id)
    assert len(store.queries) == 1


def test_status_preserves_every_valid_selector_in_deterministic_order() -> None:
    store = _QueryStore()

    query_status(
        store,
        selected_target_profile="synthetic-jit",
        kernel_id="jit.add",
        variant="default",
        test_class="exact_arithmetic",
        case_id="jit.add/target",
        producer_id="producer",
    )

    assert len(store.queries) == 1
    assert store.queries[0].parameters == (
        "synthetic-jit",
        "jit.add",
        "default",
        "exact_arithmetic",
        "jit.add/target",
        "producer",
    )


def test_todo_matches_equivalent_requirements_without_loading_stored_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = synthetic_evidence.synthetic_report()
    stored = synthetic_evidence.contradicting_report(current)
    assert current.to_jsonl() != stored.to_jsonl()
    assert current.header is not None
    assert stored.header is not None
    assert (
        current.header.verification_spec["verification_spec_id"]
        == stored.header.verification_spec["verification_spec_id"]
    )
    requirement_ids = tuple(
        value["requirement_id"]
        for value in current.header.verification_spec["requirements"]
    )
    store = _TodoStore(stored, requirement_ids)
    monkeypatch.setattr(VerificationReport, "from_jsonl", None)

    assert not query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=current,
    )
    assert len(store.queries) == 1
    assert store.queries[0].parameters == (_todo_provenance_digest(current),)
    assert "report_json" not in store.queries[0].template
    assert "r.todo_provenance_digest=?" in store.queries[0].template


def test_todo_keeps_a_truly_unobserved_requirement_in_stable_order() -> None:
    current = synthetic_evidence.synthetic_report()
    assert current.header is not None
    requirements = tuple(current.header.verification_spec["requirements"])
    store = _TodoStore(
        synthetic_evidence.contradicting_report(current),
        (requirements[0]["requirement_id"],),
    )

    missing = query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=current,
    )

    assert [item.requirement_id for item in missing] == [
        requirements[1]["requirement_id"]
    ]


@pytest.mark.parametrize("axis", ("verification", "tolerance", "oracle"))
def test_todo_rejects_observations_from_a_changed_provenance_axis(axis) -> None:
    current = synthetic_evidence.synthetic_report()
    stale = synthetic_evidence.provenance_variant(axis)
    assert current.header is not None
    requirement_ids = tuple(
        value["requirement_id"]
        for value in current.header.verification_spec["requirements"]
    )
    store = _TodoStore(stale, requirement_ids)

    missing = query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=current,
    )

    assert {item.requirement_id for item in missing} == set(requirement_ids)


def test_status_filters_and_includes_oracle_dependencies(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, store, producer_id="producer-a")
    _record_validated_report(synthetic_report, store, producer_id="producer-b")
    observations = query_status(store, selected_target_profile="synthetic-jit")
    assert {item.stage for item in observations} == {"stage_one", "stage_two"}
    assert {item.producer_id for item in observations} == {"producer-a", "producer-b"}
    filtered = query_status(
        store,
        selected_target_profile="synthetic-jit",
        kernel_id="jit.add",
        producer_id="producer-b",
    )
    assert len(filtered) == 1
    assert filtered[0].receipt_kind == "jit-specialization"


def test_stale_and_todo_are_scoped_to_exact_current_graph(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path())
    _record_validated_report(synthetic_report, store, producer_id="producer")
    current = synthetic_evidence.synthetic_report(specialization=256)
    stale = query_stale(
        store, selected_target_profile="synthetic-jit", current_report=current
    )
    assert len(stale) == 1
    assert {item.axis for item in stale[0].differences} >= {
        "specialization",
        "generated_source",
        "executable_artifact",
    }
    missing = query_todo(
        store, selected_target_profile="synthetic-jit", current_report=current
    )
    assert {item.case_id for item in missing} == {"cpu.add/oracle", "jit.add/target"}
    assert not query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=synthetic_report,
    )


def test_todo_does_not_match_observations_for_another_target(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path("wrong-target"))
    _record_validated_report(
        synthetic_evidence.cpu_report(), store, producer_id="cpu-producer"
    )

    missing = query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=synthetic_report,
    )

    assert {item.case_id for item in missing} == {
        "cpu.add/oracle",
        "jit.add/target",
    }


def test_todo_matches_many_equivalent_distinct_runs(
    evidence_store_path, synthetic_report
) -> None:
    store = DoltEvidenceStore(evidence_store_path("many-runs"))
    reports = (
        synthetic_report,
        synthetic_evidence.contradicting_report(synthetic_report),
        synthetic_evidence.target_outcome_report(
            synthetic_report, VerificationOutcome.ERROR
        ),
        synthetic_evidence.target_outcome_report(
            synthetic_report, VerificationOutcome.BLOCKED
        ),
    )
    for index, report in enumerate(reports):
        _record_validated_report(report, store, producer_id=f"producer-{index}")
    current = synthetic_evidence.target_outcome_report(
        synthetic_report, VerificationOutcome.FAILED, alternate=True
    )

    assert not query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=current,
    )
    assert store.query(SQLStatement("SELECT COUNT(*) AS count FROM verification_runs"))[
        0
    ]["count"] == len(reports)


@pytest.mark.parametrize("axis", ("verification", "tolerance", "oracle"))
def test_todo_keeps_real_stale_axis_observations_missing(
    evidence_store_path, synthetic_report, axis
) -> None:
    store = DoltEvidenceStore(evidence_store_path(f"stale-{axis}"))
    _record_validated_report(
        synthetic_evidence.provenance_variant(axis),
        store,
        producer_id="stale-producer",
    )

    missing = query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=synthetic_report,
    )

    assert len(missing) == len(synthetic_report.records)


@pytest.mark.parametrize(
    "outcome",
    (
        VerificationOutcome.FAILED,
        VerificationOutcome.ERROR,
        VerificationOutcome.BLOCKED,
        VerificationOutcome.DEFERRED,
    ),
)
def test_todo_subtracts_every_matching_factual_observation(
    evidence_store_path, synthetic_report, outcome
) -> None:
    report = synthetic_evidence.target_outcome_report(synthetic_report, outcome)
    current = (
        synthetic_evidence.target_outcome_report(
            synthetic_report, outcome, alternate=True
        )
        if outcome is VerificationOutcome.DEFERRED
        else synthetic_report
    )
    assert report.to_jsonl() != current.to_jsonl()
    assert report.header is not None
    assert current.header is not None
    assert (
        report.header.verification_spec["verification_spec_id"]
        == current.header.verification_spec["verification_spec_id"]
    )
    store = DoltEvidenceStore(evidence_store_path(outcome.value))
    _record_validated_report(report, store, producer_id="producer-a")
    _record_validated_report(report, store, producer_id="producer-b")

    assert not query_todo(
        store,
        selected_target_profile="synthetic-jit",
        current_report=current,
    )
    target = query_status(
        store,
        selected_target_profile="synthetic-jit",
        kernel_id="jit.add",
    )
    assert len(target) == 2
    assert {item.producer_id for item in target} == {"producer-a", "producer-b"}
    assert {item.outcome for item in target} == {outcome.value}
