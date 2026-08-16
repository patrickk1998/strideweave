"""Inspection-only command-line interface for verification JSONL reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .model import (
    EvidenceRecord,
    VerificationClass,
    VerificationOutcome,
    VerificationReport,
    VerificationStage,
    VerificationSummary,
)


def _parser() -> argparse.ArgumentParser:
    """Build the standard-library argument parser for report inspection."""
    parser = argparse.ArgumentParser(
        prog="strideweave-verify-report",
        description="Inspect deterministic StrideWeave verification JSONL evidence.",
    )
    parser.add_argument("report", help="JSONL evidence written by sw.test_backend().")
    parser.add_argument(
        "--problems",
        action="store_true",
        help="Keep failed, errored, and blocked evidence only.",
    )
    parser.add_argument("--operation", help="Keep one exact operation name.")
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in VerificationStage],
        help="Keep one evidence pipeline stage.",
    )
    parser.add_argument(
        "--class",
        dest="test_class",
        choices=[test_class.value for test_class in VerificationClass],
        help="Keep one verification class.",
    )
    parser.add_argument(
        "--outcome",
        action="append",
        choices=[outcome.value for outcome in VerificationOutcome],
        help="Keep an outcome; repeat this option to retain multiple outcomes.",
    )
    parser.add_argument("--kernel", help="Keep one exact native kernel ID.")
    parser.add_argument("--variant", help="Keep one exact kernel variant.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include one flat, per-case evidence line in text output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit stable machine-readable JSON instead of human-readable text.",
    )
    return parser


def _summary_json(summary: VerificationSummary) -> dict[str, Any]:
    """Return a JSON-safe, deterministically ordered summary payload."""
    return {
        "classes": {test_class.value: count for test_class, count in summary.classes},
        "gate_passed": summary.gate_passed,
        "outcomes": {outcome.value: count for outcome, count in summary.outcomes},
        "stages": {stage.value: count for stage, count in summary.stages},
        "total": summary.total,
    }


def _record_json(record: EvidenceRecord) -> dict[str, Any]:
    """Return the stable, flat evidence fields used by verbose CLI output."""
    return {
        "case_id": record.case.case_id,
        "class": record.test_class.value,
        "compilation_receipt_id": record.compilation_receipt_id,
        "consumed_certificate_digest": record.consumed_certificate_digest,
        "deviations": record.as_json_object()["deviations"],
        "kernel_id": record.case.kernel_id,
        "operation": record.case.operation,
        "oracle_reference_id": record.oracle_reference_id,
        "outcome": record.outcome.value,
        "stage": record.stage.value,
        "tolerance": record.as_json_object()["tolerance"],
        "tolerance_policy_id": record.tolerance_policy_id,
        "verification_requirement_id": record.requirement_id,
        "variant": record.case.variant,
    }


def _format_record(record: EvidenceRecord) -> str:
    """Render a stable flat case description without recursive repr output."""
    value = _record_json(record)
    deviations = json.dumps(value["deviations"], separators=(",", ":"), sort_keys=True)
    tolerance = json.dumps(value["tolerance"], separators=(",", ":"), sort_keys=True)
    return (
        f"case_id={value['case_id']} stage={value['stage']} "
        f"operation={value['operation']} kernel_id={value['kernel_id']} "
        f"variant={value['variant']} class={value['class']} "
        f"outcome={value['outcome']} deviations={deviations} tolerance={tolerance} "
        f"requirement_id={value['verification_requirement_id']} "
        f"compilation_receipt_id={value['compilation_receipt_id']} "
        f"tolerance_policy_id={value['tolerance_policy_id']} "
        f"oracle_reference_id={value['oracle_reference_id']} "
        f"consumed_certificate_digest={value['consumed_certificate_digest']}"
    )


def _filtered_report(
    report: VerificationReport, arguments: argparse.Namespace
) -> VerificationReport:
    """Apply the parser's explicit filters through the public report API."""
    if arguments.problems:
        report = report.problems
    return report.select(
        stage=None if arguments.stage is None else VerificationStage(arguments.stage),
        outcomes=(
            None
            if arguments.outcome is None
            else tuple(VerificationOutcome(outcome) for outcome in arguments.outcome)
        ),
        test_class=(
            None
            if arguments.test_class is None
            else VerificationClass(arguments.test_class)
        ),
        operation=arguments.operation,
        kernel_id=arguments.kernel,
        variant=arguments.variant,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect one verification JSONL file and return a stable process status.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        ``0`` when the selected report has no failed, errored, or blocked
        evidence, ``1`` when its correctness gate fails, or ``2`` when the
        report cannot be loaded. Argument syntax errors use argparse's standard
        exit status ``2``.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     status = main([f"{directory}/missing.jsonl"])
        >>> status
        2
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        report = VerificationReport.load(arguments.report)
    except (OSError, TypeError, ValueError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    report = _filtered_report(report, arguments)
    if arguments.json:
        payload: dict[str, Any] = _summary_json(report.summary())
        if arguments.verbose:
            payload["records"] = [_record_json(record) for record in report.records]
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(report.describe())
        if arguments.verbose:
            for record in report.records:
                print(_format_record(record))
    return 0 if report.summary().gate_passed else 1


if __name__ == "__main__":  # pragma: no cover - package entry point uses main().
    raise SystemExit(main())
