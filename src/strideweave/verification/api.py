"""Public entry point for local backend verification."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from .classification import require_native_verification_api
from .model import VerificationReport
from .reporting import make_verification_report
from .stage_one import run_stage_one
from .stage_two import run_stage_two


def test_backend(output: str | PathLike[str] | None = None) -> VerificationReport:
    """Verify the installed CPU backend through its public operation capabilities.

    Stage One first checks every explicitly classified CPU capability plan and
    certifies the public CPU Float64 accumulator as an oracle. Stage Two then
    checks ordinary CPU Float32 execution only for operations whose local oracle
    certificate passed. The function performs no network, CI, database, or
    status mutation.

    Args:
        output: Optional filesystem path that receives deterministic, versioned
            provenance-complete JSONL report for all Stage One and Stage Two attempts.

    Returns:
        Immutable report containing the combined local evidence records.

    Examples:
        >>> import strideweave as sw
        >>> report = sw.test_backend()
        >>> len(report.records) > 0
        True
    """
    require_native_verification_api()
    stage_one = run_stage_one()
    stage_two = run_stage_two(stage_one)
    report = make_verification_report(
        (*stage_one.report.records, *stage_two.records), stage_one.certificates
    )
    if output is not None:
        Path(output).write_text(report.to_jsonl(), encoding="utf-8")
    return report
