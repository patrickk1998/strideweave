"""Public entry point for local backend verification."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

from .classification import require_native_verification_api, verification_profile
from .model import VerificationReport, VerificationStage
from .provenance import installed_compilation_bundle, make_compilation_bundle
from .reporting import make_verification_report
from .stage_one import run_oracle_stage
from .stage_two import _require_target_runtime, run_target_stage


def _atomic_write(path: Path, contents: str) -> None:
    """Atomically replace one caller-selected report destination."""

    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def verify_backend(
    target: str, *, output: str | PathLike[str] | None = None
) -> VerificationReport:
    """Verify one explicit installed target against the CPU oracle.

    Stage One certifies every classified CPU oracle obligation. Stage Two then
    selects the requested target profile and runs only target evidence whose
    operation and complete CPU certificate scope are accepted locally. The
    function performs no network, CI, database, or evidence-store mutation.

    Args:
        target: Exact registered target verification profile identifier.
        output: Optional string or path-like destination atomically replaced
            with canonical schema-v3 JSONL after verification completes.

    Returns:
        Immutable provenance-complete report for the CPU oracle and target.

    Examples:
        >>> import strideweave as sw
        >>> report = sw.verify_backend("cpu-compiled")
        >>> len(report.records) > 0
        True
    """
    if type(target) is not str:
        raise TypeError("target must be a string")
    if output is not None and not isinstance(output, (str, PathLike)):
        raise TypeError("output must be a string, path-like value, or None")
    profile = verification_profile(target)
    if VerificationStage.TARGET not in profile.stages:
        raise ValueError(f"profile {target!r} is not a target verification profile")
    require_native_verification_api()
    _require_target_runtime(profile)
    oracle_profile = verification_profile("cpu-compiled")
    oracle_bundle = installed_compilation_bundle(oracle_profile)
    stage_one = run_oracle_stage(oracle_profile)
    stage_two = run_target_stage(profile, stage_one)
    compilation_bundle = make_compilation_bundle(
        (*oracle_bundle.receipts, *stage_two.receipts)
    )
    report = make_verification_report(
        (*stage_one.report.records, *stage_two.records),
        stage_one.certificates,
        selected_target_profile=profile.profile_id,
        oracle_profile=oracle_profile.profile_id,
        compilation_bundle=compilation_bundle,
    )
    if output is not None:
        _atomic_write(Path(output), report.to_jsonl())
    return report
