"""Immutable records shared by staged kernel verification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection
from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any

_EVIDENCE_SCHEMA_VERSION = "strideweave.kernel-evidence.v1"
_REPORT_SCHEMA_VERSION = "strideweave.kernel-verification.v1"
_NONFINITE_FLOATS = {
    "NaN": math.nan,
    "Infinity": math.inf,
    "-Infinity": -math.inf,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while refusing ambiguous duplicate members."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject JSON extensions such as bare ``NaN`` during report loading."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_fields(
    value: dict[str, Any], field: str, expected: frozenset[str]
) -> None:
    observed = frozenset(value)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields {', '.join(unexpected)}")
        raise ValueError(f"{field} has {'; '.join(details)}")


def _require_string(value: Any, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a string")
    return value


def _require_optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field)


def _require_integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _require_optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, field)


def _require_float(value: Any, field: str) -> float:
    if type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{field} must be representable as a finite JSON number"
            ) from exc
        if math.isfinite(number):
            return number
        raise ValueError(f"{field} must be a finite JSON number")
    if type(value) is str and value in _NONFINITE_FLOATS:
        return _NONFINITE_FLOATS[value]
    raise ValueError(
        f"{field} must be a finite JSON number or one of the canonical non-finite strings"
    )


def _require_optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, field)


def _require_strings(value: Any, field: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an array")
    return tuple(
        _require_string(item, f"{field}[{index}]") for index, item in enumerate(value)
    )


def _parse_plan(value: Any, field: str) -> PlanKey:
    data = _require_mapping(value, field)
    _require_exact_fields(
        data,
        field,
        frozenset(
            {
                "operation",
                "operands",
                "compute",
                "accumulation",
                "accumulator_dtype",
                "output",
            }
        ),
    )
    operands_value = data["operands"]
    if type(operands_value) is not list:
        raise ValueError(f"{field}.operands must be an array")
    operands: list[tuple[str, str | None, str | None]] = []
    for index, operand in enumerate(operands_value):
        operand_field = f"{field}.operands[{index}]"
        if type(operand) is not list or len(operand) != 3:
            raise ValueError(f"{operand_field} must be a three-item array")
        operands.append(
            (
                _require_string(operand[0], f"{operand_field}[0]"),
                _require_optional_string(operand[1], f"{operand_field}[1]"),
                _require_optional_string(operand[2], f"{operand_field}[2]"),
            )
        )
    return PlanKey(
        operation=_require_string(data["operation"], f"{field}.operation"),
        operands=tuple(operands),
        compute=_require_string(data["compute"], f"{field}.compute"),
        accumulation=_require_optional_string(
            data["accumulation"], f"{field}.accumulation"
        ),
        accumulator_dtype=_require_optional_string(
            data["accumulator_dtype"], f"{field}.accumulator_dtype"
        ),
        output=_require_string(data["output"], f"{field}.output"),
    )


def _parse_case(value: Any) -> CaseDescriptor:
    data = _require_mapping(value, "case")
    _require_exact_fields(
        data,
        "case",
        frozenset(
            {
                "operation",
                "kernel_id",
                "variant",
                "input_dtypes",
                "output_dtype",
                "shapes",
                "accumulator_dtype",
                "contraction_length",
                "seed",
                "case_id",
                "plan",
            }
        ),
    )
    shapes_value = data["shapes"]
    if type(shapes_value) is not list:
        raise ValueError("case.shapes must be an array")
    shapes: list[tuple[int, ...]] = []
    for shape_index, shape in enumerate(shapes_value):
        if type(shape) is not list:
            raise ValueError(f"case.shapes[{shape_index}] must be an array")
        shapes.append(
            tuple(
                _require_integer(extent, f"case.shapes[{shape_index}][{extent_index}]")
                for extent_index, extent in enumerate(shape)
            )
        )
    plan_value = data["plan"]
    return CaseDescriptor(
        operation=_require_string(data["operation"], "case.operation"),
        kernel_id=_require_string(data["kernel_id"], "case.kernel_id"),
        variant=_require_string(data["variant"], "case.variant"),
        input_dtypes=_require_strings(data["input_dtypes"], "case.input_dtypes"),
        output_dtype=_require_string(data["output_dtype"], "case.output_dtype"),
        shapes=tuple(shapes),
        accumulator_dtype=_require_optional_string(
            data["accumulator_dtype"], "case.accumulator_dtype"
        ),
        contraction_length=_require_optional_integer(
            data["contraction_length"], "case.contraction_length"
        ),
        seed=_require_optional_integer(data["seed"], "case.seed"),
        case_id=_require_string(data["case_id"], "case.case_id"),
        plan=None if plan_value is None else _parse_plan(plan_value, "case.plan"),
    )


def _parse_tolerance(value: Any) -> Tolerance:
    data = _require_mapping(value, "tolerance")
    _require_exact_fields(
        data, "tolerance", frozenset({"absolute", "relative", "ulps", "version"})
    )
    return Tolerance(
        absolute=_require_float(data["absolute"], "tolerance.absolute"),
        relative=_require_float(data["relative"], "tolerance.relative"),
        ulps=_require_integer(data["ulps"], "tolerance.ulps"),
        version=_require_string(data["version"], "tolerance.version"),
    )


def _parse_deviations(value: Any) -> Deviations:
    data = _require_mapping(value, "deviations")
    _require_exact_fields(
        data,
        "deviations",
        frozenset({"maximum_absolute", "maximum_relative", "maximum_ulps"}),
    )
    return Deviations(
        maximum_absolute=_require_optional_float(
            data["maximum_absolute"], "deviations.maximum_absolute"
        ),
        maximum_relative=_require_optional_float(
            data["maximum_relative"], "deviations.maximum_relative"
        ),
        maximum_ulps=_require_optional_integer(
            data["maximum_ulps"], "deviations.maximum_ulps"
        ),
    )


def _parse_evidence_record(value: Any) -> EvidenceRecord:
    data = _require_mapping(value, "record")
    _require_exact_fields(
        data,
        "record",
        frozenset(
            {
                "stage",
                "test_class",
                "case",
                "target_input_bit_hashes",
                "oracle_input_bit_hashes",
                "tolerance",
                "deviations",
                "mismatches",
                "outcome",
                "diagnostic",
                "schema_version",
            }
        ),
    )
    schema_version = _require_string(data["schema_version"], "record.schema_version")
    if schema_version != _EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence schema version {schema_version!r}")
    try:
        stage = VerificationStage(_require_string(data["stage"], "record.stage"))
        test_class = VerificationClass(
            _require_string(data["test_class"], "record.test_class")
        )
        outcome = VerificationOutcome(
            _require_string(data["outcome"], "record.outcome")
        )
    except ValueError as exc:
        raise ValueError(f"invalid enum value: {exc}") from exc
    return EvidenceRecord(
        stage=stage,
        test_class=test_class,
        case=_parse_case(data["case"]),
        target_input_bit_hashes=_require_strings(
            data["target_input_bit_hashes"], "record.target_input_bit_hashes"
        ),
        oracle_input_bit_hashes=_require_strings(
            data["oracle_input_bit_hashes"], "record.oracle_input_bit_hashes"
        ),
        tolerance=_parse_tolerance(data["tolerance"]),
        deviations=_parse_deviations(data["deviations"]),
        mismatches=_require_optional_integer(data["mismatches"], "record.mismatches"),
        outcome=outcome,
        diagnostic=_require_optional_string(data["diagnostic"], "record.diagnostic"),
        schema_version=schema_version,
    )


class VerificationStage(Enum):
    """Evidence pipeline stage that produced a verification record."""

    ORACLE = "stage_one"
    TARGET = "stage_two"


class VerificationClass(Enum):
    """Semantic class of verification evidence required for certification."""

    BIT_EXACT = "bit_exact"
    EXACT_ARITHMETIC = "exact_arithmetic"
    STRUCTURAL = "structural"
    ANALYTIC = "analytic"
    NUMERICAL = "numerical"
    DEFERRED = "deferred"


class VerificationOutcome(Enum):
    """Result state emitted for an attempted verification case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class ClassificationDisposition(Enum):
    """Whether a kernel plan is executable now or explicitly deferred."""

    ACTIVE = "active"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class KernelDescriptor:
    """Stable native kernel identity and its public binding name."""

    operation: str
    kernel_id: str
    variant: str
    pybind_name: str


@dataclass(frozen=True, slots=True)
class PlanKey:
    """Immutable normalized identity of one resolved operation capability plan."""

    operation: str
    operands: tuple[tuple[str, str | None, str | None], ...]
    compute: str
    accumulation: str | None
    accumulator_dtype: str | None
    output: str

    @classmethod
    def from_plan_like(cls, plan: Any) -> PlanKey:
        return cls(
            operation=plan.operation,
            operands=tuple(
                (
                    operand.role.name,
                    None if operand.dtype is None else operand.dtype.name,
                    None if operand.convert_to is None else operand.convert_to.name,
                )
                for operand in plan.operands
            ),
            compute=plan.compute.name,
            accumulation=None if plan.accumulation is None else plan.accumulation.name,
            accumulator_dtype=(
                None if plan.accumulator_dtype is None else plan.accumulator_dtype.name
            ),
            output=plan.output.name,
        )


@dataclass(frozen=True, slots=True)
class KernelPlanDescriptor:
    """Native kernel paired with its resolved plan and evidence requirements."""

    kernel: KernelDescriptor
    plan: PlanKey
    classes: tuple[VerificationClass, ...]
    disposition: ClassificationDisposition
    deferred_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CaseDescriptor:
    """Deterministic operation, input, shape, and plan metadata for one case."""

    operation: str
    kernel_id: str
    variant: str
    input_dtypes: tuple[str, ...]
    output_dtype: str
    shapes: tuple[tuple[int, ...], ...]
    accumulator_dtype: str | None
    contraction_length: int | None
    seed: int | None
    case_id: str
    plan: PlanKey | None = None


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Versioned numerical thresholds attached to one evidence record."""

    absolute: float = 0.0
    relative: float = 0.0
    ulps: int = 0
    version: str = "exact-v1"


@dataclass(frozen=True, slots=True)
class Deviations:
    """Maximum absolute, relative, and ULP differences observed in a case."""

    maximum_absolute: float | None
    maximum_relative: float | None
    maximum_ulps: int | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Immutable JSON-serializable evidence for one verification attempt."""

    stage: VerificationStage
    test_class: VerificationClass
    case: CaseDescriptor
    target_input_bit_hashes: tuple[str, ...]
    oracle_input_bit_hashes: tuple[str, ...]
    tolerance: Tolerance
    deviations: Deviations
    mismatches: int | None
    outcome: VerificationOutcome
    diagnostic: str | None = None
    schema_version: str = _EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.target_input_bit_hashes != self.oracle_input_bit_hashes:
            raise ValueError("target and oracle input bit hashes must match")

    def as_json_object(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage"] = self.stage.value
        value["test_class"] = self.test_class.value
        value["outcome"] = self.outcome.value
        return _json_safe(value)

    def to_jsonl(self) -> str:
        return json.dumps(
            self.as_json_object(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """Immutable aggregate counts and gate status for one verification report.

    The ``outcomes``, ``stages``, and ``classes`` tuples retain the declaration
    order of their respective enums and include zero counts. ``gate_passed``
    is true exactly when the report has no failed, errored, or blocked records;
    deferred records remain visible coverage gaps and do not by themselves fail
    the gate. The direct count accessors make common outcome totals convenient
    without replacing those complete aggregates. ``errors`` is plural because
    it is a count; the serialized and CLI outcome label remains ``error``.

    Args:
        total: Number of evidence records represented by the summary.
        outcomes: Count for every :class:`VerificationOutcome`.
        stages: Count for every :class:`VerificationStage`.
        classes: Count for every :class:`VerificationClass`.
        gate_passed: Whether no failed, errored, or blocked record is present.

    Examples:
        >>> from strideweave.verification import VerificationReport
        >>> VerificationReport(()).summary().gate_passed
        True
    """

    total: int
    outcomes: tuple[tuple[VerificationOutcome, int], ...]
    stages: tuple[tuple[VerificationStage, int], ...]
    classes: tuple[tuple[VerificationClass, int], ...]
    gate_passed: bool

    def __repr__(self) -> str:
        """Return a bounded representation containing outcome counts and gate status."""
        return (
            "VerificationSummary("
            f"total={self.total}, "
            f"passed={self.passed}, "
            f"failed={self.failed}, "
            f"errors={self.errors}, "
            f"blocked={self.blocked}, "
            f"deferred={self.deferred}, "
            f"gate_passed={self.gate_passed}"
            ")"
        )

    def _outcome_count(self, outcome: VerificationOutcome) -> int:
        """Return one outcome's count from the authoritative aggregate tuple."""
        return dict(self.outcomes)[outcome]

    @property
    def passed(self) -> int:
        """Return the number of evidence records with the ``passed`` outcome."""
        return self._outcome_count(VerificationOutcome.PASSED)

    @property
    def failed(self) -> int:
        """Return the number of evidence records with the ``failed`` outcome."""
        return self._outcome_count(VerificationOutcome.FAILED)

    @property
    def errors(self) -> int:
        """Return the number of evidence records with the ``error`` outcome."""
        return self._outcome_count(VerificationOutcome.ERROR)

    @property
    def blocked(self) -> int:
        """Return the number of evidence records with the ``blocked`` outcome."""
        return self._outcome_count(VerificationOutcome.BLOCKED)

    @property
    def deferred(self) -> int:
        """Return the number of evidence records with the ``deferred`` outcome."""
        return self._outcome_count(VerificationOutcome.DEFERRED)


def _normalized_filter[EnumType: Enum](
    value: EnumType | Collection[EnumType] | None,
    enum_type: type[EnumType],
    field: str,
) -> frozenset[EnumType] | None:
    """Normalize one enum or a collection of enums for report selection."""
    if value is None:
        return None
    if isinstance(value, enum_type):
        return frozenset((value,))
    if not isinstance(value, Collection) or not all(
        isinstance(item, enum_type) for item in value
    ):
        raise TypeError(
            f"{field} must be a {enum_type.__name__} or a collection of them"
        )
    return frozenset(value)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Collection of deterministic evidence records from a local run.

    A report records every Stage One oracle and Stage Two target attempt.
    Records retain their case metadata, outcome, tolerance, observed
    deviations, and matching target/oracle input hashes. Outcomes distinguish
    passed comparisons from failed comparisons, execution errors, blocked
    Stage Two evidence, and explicitly deferred coverage. A deferred record is
    visible in summaries but does not alone make ``gate_passed`` false.

    Use :meth:`summary` for immutable counts, :meth:`describe` for deterministic
    text, and :meth:`select` or the :attr:`passed`, :attr:`deferred`, and
    :attr:`problems` views to navigate the authoritative ``records`` tuple.
    JSONL serialization is stable and reloads through a strict, line-numbered
    evidence-schema validator. The evidence-only v1 wire format has no report
    header, so this model accepts only its current report schema version rather
    than silently serializing a report a v1 reader would misidentify.

    Args:
        records: Immutable evidence records collected by local verification.
        schema_version: Version of the report format represented by this model.

    Examples:
        >>> from strideweave.verification import VerificationReport
        >>> report = VerificationReport(())
        >>> report.to_jsonl()
        ''
    """

    records: tuple[EvidenceRecord, ...]
    schema_version: str = _REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Refuse report versions that the evidence-only wire format cannot encode."""
        if self.schema_version != _REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema version {self.schema_version!r}"
            )

    def __repr__(self) -> str:
        """Return a bounded representation containing only outcome counts."""
        summary = self.summary()
        return (
            "VerificationReport("
            f"total={summary.total}, "
            f"passed={summary.passed}, "
            f"failed={summary.failed}, "
            f"errors={summary.errors}, "
            f"blocked={summary.blocked}, "
            f"deferred={summary.deferred}, "
            f"gate_passed={summary.gate_passed}"
            ")"
        )

    def summary(self) -> VerificationSummary:
        """Return immutable aggregate counts for the report's evidence.

        ``gate_passed`` is true precisely when there are no failed, errored,
        or blocked records. Explicitly deferred records remain counted as
        uncovered evidence but do not themselves fail the gate.

        Returns:
            Immutable total, outcome, stage, and test-class counts plus gate status.

        Examples:
            >>> from strideweave.verification import VerificationReport
            >>> VerificationReport(()).summary().total
            0
        """
        outcomes = tuple(
            (
                outcome,
                sum(record.outcome is outcome for record in self.records),
            )
            for outcome in VerificationOutcome
        )
        stages = tuple(
            (stage, sum(record.stage is stage for record in self.records))
            for stage in VerificationStage
        )
        classes = tuple(
            (
                test_class,
                sum(record.test_class is test_class for record in self.records),
            )
            for test_class in VerificationClass
        )
        problem_outcomes = {
            VerificationOutcome.FAILED,
            VerificationOutcome.ERROR,
            VerificationOutcome.BLOCKED,
        }
        return VerificationSummary(
            total=len(self.records),
            outcomes=outcomes,
            stages=stages,
            classes=classes,
            gate_passed=all(
                record.outcome not in problem_outcomes for record in self.records
            ),
        )

    def describe(self) -> str:
        """Return deterministic human-readable text describing report coverage.

        Returns:
            A plain-text summary of outcomes, stages, classes, and gate status.

        Examples:
            >>> from strideweave.verification import VerificationReport
            >>> VerificationReport(()).describe()
            'Verification report: 0 records; gate passed: yes. Outcomes: passed=0, failed=0, error=0, blocked=0, deferred=0. Stages: stage_one=0, stage_two=0. Classes: bit_exact=0, exact_arithmetic=0, structural=0, analytic=0, numerical=0, deferred=0.'
        """
        summary = self.summary()
        outcomes = ", ".join(
            f"{outcome.value}={count}" for outcome, count in summary.outcomes
        )
        stages = ", ".join(f"{stage.value}={count}" for stage, count in summary.stages)
        classes = ", ".join(
            f"{test_class.value}={count}" for test_class, count in summary.classes
        )
        gate = "yes" if summary.gate_passed else "no"
        return (
            f"Verification report: {summary.total} records; gate passed: {gate}. "
            f"Outcomes: {outcomes}. Stages: {stages}. Classes: {classes}."
        )

    def select(
        self,
        *,
        stage: VerificationStage | Collection[VerificationStage] | None = None,
        outcomes: VerificationOutcome | Collection[VerificationOutcome] | None = None,
        test_class: VerificationClass | Collection[VerificationClass] | None = None,
        operation: str | None = None,
        kernel_id: str | None = None,
        variant: str | None = None,
    ) -> VerificationReport:
        """Return an immutable report containing records matching all filters.

        Args:
            stage: One pipeline stage or collection of stages to retain.
            outcomes: One outcome or collection of outcomes to retain.
            test_class: One verification class or collection of classes to retain.
            operation: Exact operation name to retain.
            kernel_id: Exact native kernel identifier to retain.
            variant: Exact kernel variant to retain.

        Returns:
            A report containing only records that match every supplied filter.

        Examples:
            >>> from strideweave.verification import VerificationReport, VerificationOutcome
            >>> VerificationReport(()).select(outcomes=VerificationOutcome.FAILED).records
            ()
        """
        stages = _normalized_filter(stage, VerificationStage, "stage")
        selected_outcomes = _normalized_filter(
            outcomes, VerificationOutcome, "outcomes"
        )
        classes = _normalized_filter(test_class, VerificationClass, "test_class")
        for field, value in (
            ("operation", operation),
            ("kernel_id", kernel_id),
            ("variant", variant),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"{field} must be a string or None")
        return type(self)(
            tuple(
                record
                for record in self.records
                if (stages is None or record.stage in stages)
                and (selected_outcomes is None or record.outcome in selected_outcomes)
                and (classes is None or record.test_class in classes)
                and (operation is None or record.case.operation == operation)
                and (kernel_id is None or record.case.kernel_id == kernel_id)
                and (variant is None or record.case.variant == variant)
            ),
            schema_version=self.schema_version,
        )

    @property
    def passed(self) -> VerificationReport:
        """Return the report view containing only passed evidence records."""
        return self.select(outcomes=VerificationOutcome.PASSED)

    @property
    def deferred(self) -> VerificationReport:
        """Return the report view containing only explicitly deferred records."""
        return self.select(outcomes=VerificationOutcome.DEFERRED)

    @property
    def problems(self) -> VerificationReport:
        """Return failed, errored, and blocked records, excluding deferrals."""
        return self.select(
            outcomes=(
                VerificationOutcome.FAILED,
                VerificationOutcome.ERROR,
                VerificationOutcome.BLOCKED,
            )
        )

    def to_jsonl(self) -> str:
        """Serialize all evidence as deterministic, newline-terminated JSONL.

        Returns:
            Canonically ordered JSONL, or an empty string when there are no records.

        Examples:
            >>> from strideweave.verification import VerificationReport
            >>> VerificationReport(()).to_jsonl()
            ''
        """
        ordered = sorted(
            self.records,
            key=lambda record: (
                record.case.case_id,
                record.case.kernel_id,
                record.case.variant,
            ),
        )
        return (
            ""
            if not ordered
            else "\n".join(record.to_jsonl() for record in ordered) + "\n"
        )

    @classmethod
    def from_jsonl(cls, text: str) -> VerificationReport:
        """Load a deterministic report from strict JSONL evidence.

        Args:
            text: UTF-8-decoded JSONL produced by :meth:`to_jsonl` or
                :meth:`write`.

        Returns:
            A report reconstructing every validated evidence record.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If any line has malformed JSON, an invalid schema, an
                invalid nested value, or violates an evidence-model invariant.

        Examples:
            >>> from strideweave.verification import VerificationReport
            >>> VerificationReport.from_jsonl("").records
            ()
        """
        if type(text) is not str:
            raise TypeError("text must be a string")
        records: list[EvidenceRecord] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise ValueError(f"JSONL line {line_number}: blank lines are not valid")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
                records.append(_parse_evidence_record(value))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"JSONL line {line_number}: {exc}") from exc
        return cls(tuple(records))

    @classmethod
    def load(cls, path: str | PathLike[str]) -> VerificationReport:
        """Read UTF-8 JSONL evidence from a filesystem path.

        Args:
            path: File containing JSONL previously created by :meth:`write`.

        Returns:
            The strictly validated report stored at ``path``.

        Examples:
            >>> from pathlib import Path
            >>> from strideweave.verification import VerificationReport
            >>> path = Path("verification.jsonl")
            >>> VerificationReport(()).write(path)
            >>> VerificationReport.load(path).records
            ()
        """
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | PathLike[str]) -> None:
        """Write deterministic UTF-8 JSONL evidence to a filesystem path.

        Args:
            path: Destination file to replace with this report's JSONL.

        Returns:
            ``None`` after the complete report is written.

        Examples:
            >>> from pathlib import Path
            >>> from strideweave.verification import VerificationReport
            >>> VerificationReport(()).write(Path("verification.jsonl"))
        """
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class OracleCertificate:
    """Digest-backed proof that a kernel's required oracle cases passed."""

    kernel_id: str
    variant: str
    certified_classes: tuple[VerificationClass, ...]
    evidence_digest: str
    certified_plan_classes: tuple[
        tuple[PlanKey, tuple[VerificationClass, ...]], ...
    ] = ()

    @classmethod
    def from_records(
        cls,
        kernel: KernelDescriptor,
        required_classes: tuple[VerificationClass, ...],
        records: tuple[EvidenceRecord, ...],
        *,
        required_plan_classes: tuple[
            tuple[PlanKey, tuple[VerificationClass, ...]], ...
        ] = (),
    ) -> OracleCertificate:
        observed = {
            record.test_class
            for record in records
            if record.case.kernel_id == kernel.kernel_id
            and record.case.variant == kernel.variant
            and record.outcome is VerificationOutcome.PASSED
        }
        missing = tuple(
            test_class for test_class in required_classes if test_class not in observed
        )
        if missing:
            names = ", ".join(test_class.value for test_class in missing)
            raise ValueError(
                f"cannot certify {kernel.kernel_id}: missing passed classes {names}"
            )
        if any(
            record.case.kernel_id == kernel.kernel_id
            and record.case.variant == kernel.variant
            and record.test_class in required_classes
            and record.outcome is not VerificationOutcome.PASSED
            for record in records
        ):
            raise ValueError(
                f"cannot certify {kernel.kernel_id}: required evidence did not pass"
            )
        for plan, plan_classes in required_plan_classes:
            plan_records = tuple(
                record
                for record in records
                if record.case.kernel_id == kernel.kernel_id
                and record.case.variant == kernel.variant
                and record.case.plan == plan
            )
            passed = {
                record.test_class
                for record in plan_records
                if record.outcome is VerificationOutcome.PASSED
            }
            missing_plan_classes = tuple(
                test_class for test_class in plan_classes if test_class not in passed
            )
            if missing_plan_classes:
                names = ", ".join(
                    test_class.value for test_class in missing_plan_classes
                )
                raise ValueError(
                    f"cannot certify {kernel.kernel_id}: plan {plan!r} is missing "
                    f"passed classes {names}"
                )
            if any(
                record.test_class in plan_classes
                and record.outcome is not VerificationOutcome.PASSED
                for record in plan_records
            ):
                raise ValueError(
                    f"cannot certify {kernel.kernel_id}: required plan evidence did not pass"
                )
        serialized = VerificationReport(records).to_jsonl().encode()
        return cls(
            kernel.kernel_id,
            kernel.variant,
            required_classes,
            hashlib.sha256(serialized).hexdigest(),
            required_plan_classes,
        )
