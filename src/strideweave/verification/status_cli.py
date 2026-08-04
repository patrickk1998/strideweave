"""Local-first command-line access to immutable verification evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .model import VerificationReport
from .store import (
    DoltEvidenceStore,
    VerificationStoreError,
    publish_evidence,
    query_stale,
    query_status,
    query_todo,
    record_report,
    refresh_evidence,
)

_FORMATTER = argparse.RawDescriptionHelpFormatter
_STORE_HELP = (
    "Local store directory. Default: the platform application-data directory "
    "under strideweave/kernel-evidence; STRIDEWEAVE_STATUS_HOME overrides its base."
)


def _query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arch",
        required=True,
        metavar="ARCH",
        help=(
            "Exact installed-manifest architecture token, such as arm64 or x86_64; "
            "values are case-sensitive and are not normalized."
        ),
    )
    parser.add_argument("--store", metavar="PATH", help=_STORE_HELP)
    parser.add_argument(
        "--json", action="store_true", help="Emit stable machine-readable output."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strideweave-kernel-status",
        formatter_class=_FORMATTER,
        description=(
            "Record provenance-complete kernel reports and inspect immutable raw "
            "evidence.\n\n"
            "Ordinary recording and queries are local and offline. Exchange is "
            "performed only by record --publish or status --refresh."
        ),
        epilog=(
            "Exit status:\n"
            "  0  Command or help completed successfully.\n"
            "  2  Invalid arguments, report, configuration, store, or exchange data.\n\n"
            "Examples:\n"
            "  strideweave-kernel-status record --report evidence.jsonl "
            "--producer local-dev\n"
            "  strideweave-kernel-status status --arch arm64\n"
            "  strideweave-kernel-status stale --arch arm64\n"
            "  strideweave-kernel-status todo --arch arm64 --json\n\n"
            "Run 'strideweave-kernel-status COMMAND --help' for command arguments."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    record = subcommands.add_parser(
        "record",
        formatter_class=_FORMATTER,
        help="Record one provenance-complete JSONL report locally.",
        description=(
            "Validate a provenance-complete v2 JSONL report against the installed "
            "build, then atomically record every raw outcome and producer observation.\n\n"
            "The command is offline by default. --publish is the only option that "
            "inspects a publication destination."
        ),
        epilog=(
            "Defaults and configuration:\n"
            "  The store uses the platform application-data default unless --store "
            "is set.\n"
            "  --publish-destination defaults to "
            "STRIDEWEAVE_STATUS_PUBLISH_DESTINATION.\n\n"
            "Exit status: 0 on success; 2 for arguments, validation, store, or "
            "publication errors.\n\n"
            "Examples:\n"
            "  strideweave-kernel-status record --report evidence.jsonl "
            "--producer local-dev\n"
            "  strideweave-kernel-status record --report evidence.jsonl "
            "--producer ci-arm64 --source-commit abc123 --artifact report://run/7 "
            "--artifact-digest <64-lowercase-hex>\n"
            "  strideweave-kernel-status record --report evidence.jsonl "
            "--producer local-dev --publish --publish-destination /shared/evidence"
        ),
    )
    record.add_argument(
        "--report",
        required=True,
        metavar="PATH",
        help="Provenance-complete v2 verification JSONL file.",
    )
    record.add_argument(
        "--producer", required=True, help="Stable identity of the evidence producer."
    )
    record.add_argument(
        "--source-commit",
        metavar="REVISION",
        help="Optional source revision observed by the producer; not a build identity.",
    )
    record.add_argument(
        "--artifact",
        metavar="LOCATOR",
        help="Optional link or locator for the retained report or build artifact.",
    )
    record.add_argument(
        "--artifact-digest",
        metavar="SHA256",
        help="Optional 64-character lowercase SHA-256 artifact digest.",
    )
    record.add_argument("--store", metavar="PATH", help=_STORE_HELP)
    record.add_argument(
        "--publish",
        action="store_true",
        help="Explicitly publish this producer's observations after recording.",
    )
    record.add_argument(
        "--publish-destination",
        metavar="ENDPOINT",
        help=(
            "Local path or file: endpoint used only with --publish. Default: "
            "STRIDEWEAVE_STATUS_PUBLISH_DESTINATION."
        ),
    )
    record.add_argument(
        "--json", action="store_true", help="Emit stable machine-readable output."
    )
    status = subcommands.add_parser(
        "status",
        formatter_class=_FORMATTER,
        help="List factual producer observations without aggregation.",
        description=(
            "List every matching producer observation in deterministic order. "
            "Contradictory outcomes remain separate and timestamps do not select a "
            "winner.\n\n"
            "The query is offline and read-only unless --refresh explicitly inspects "
            "a configured read source."
        ),
        epilog=(
            "Architecture syntax: pass the exact case-sensitive installed-manifest "
            "token, for example arm64 or x86_64.\n"
            "Defaults: local platform store; --read-source defaults to "
            "STRIDEWEAVE_STATUS_READ_SOURCE.\n"
            "Exit status: 0 on success; 2 for arguments, store, configuration, or "
            "refresh errors.\n\n"
            "Examples:\n"
            "  strideweave-kernel-status status --arch arm64\n"
            "  strideweave-kernel-status status --arch arm64 --kernel cpu.matmul "
            "--variant default --class numerical --producer local-dev --json\n"
            "  strideweave-kernel-status status --arch arm64 --refresh "
            "--read-source /shared/evidence"
        ),
    )
    _query_arguments(status)
    status.add_argument("--kernel", help="Keep one exact native kernel ID.")
    status.add_argument("--variant", help="Keep one exact kernel variant.")
    status.add_argument(
        "--class", dest="test_class", help="Keep one exact verification class."
    )
    status.add_argument("--producer", help="Keep one exact producer identity.")
    status.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly refresh contributor snapshots before querying.",
    )
    status.add_argument(
        "--read-source",
        metavar="ENDPOINT",
        help=(
            "Local path or file: endpoint used only with --refresh. Default: "
            "STRIDEWEAVE_STATUS_READ_SOURCE."
        ),
    )
    stale = subcommands.add_parser(
        "stale",
        formatter_class=_FORMATTER,
        help="Explain stored identity differences from the installed build.",
        description=(
            "Compare stored runs with a locally generated baseline for the installed "
            "build. Report manifest, receipt, closure-member, toolchain, target, "
            "specification, tolerance, and oracle identity differences independently.\n\n"
            "This command is offline and read-only; it runs local verification to "
            "construct current identities and never refreshes contributor data."
        ),
        epilog=(
            "Architecture syntax: exact case-sensitive installed-manifest token, such "
            "as arm64 or x86_64.\n"
            "Default: local platform store. Exit status: 0 on success; 2 for "
            "arguments, verification, or store errors.\n\n"
            "Examples:\n"
            "  strideweave-kernel-status stale --arch arm64\n"
            "  strideweave-kernel-status stale --arch arm64 --json "
            "--store /tmp/kernel-evidence"
        ),
    )
    _query_arguments(stale)
    todo = subcommands.add_parser(
        "todo",
        formatter_class=_FORMATTER,
        help="List current requirements with no matching observation.",
        description=(
            "Compute the deterministic, unranked set difference between current local "
            "verification requirements and matching recorded observations.\n\n"
            "This command is offline and read-only; it runs local verification to "
            "construct current requirements and never assigns risk or priority."
        ),
        epilog=(
            "Architecture syntax: exact case-sensitive installed-manifest token, such "
            "as arm64 or x86_64.\n"
            "Default: local platform store. Exit status: 0 on success; 2 for "
            "arguments, verification, or store errors.\n\n"
            "Examples:\n"
            "  strideweave-kernel-status todo --arch arm64\n"
            "  strideweave-kernel-status todo --arch arm64 --json "
            "--store /tmp/kernel-evidence"
        ),
    )
    _query_arguments(todo)
    return parser


def _record(arguments: argparse.Namespace) -> tuple[dict[str, object], str]:
    if arguments.publish_destination is not None and not arguments.publish:
        raise ValueError("--publish-destination requires --publish")
    report = VerificationReport.load(arguments.report)
    store = DoltEvidenceStore(arguments.store)
    result = record_report(
        report,
        store,
        producer_id=arguments.producer,
        source_commit=arguments.source_commit,
        artifact_locator=arguments.artifact,
        artifact_digest=arguments.artifact_digest,
    )
    payload: dict[str, object] = {
        "evidence_count": result.evidence_count,
        "observation_count": result.observation_count,
        "report_digest": result.report_digest,
        "run_id": result.run_id,
        "store": str(store.path),
    }
    text = (
        f"Recorded run {result.run_id}: {result.evidence_count} evidence rows, "
        f"{result.observation_count} observations in {store.path}."
    )
    if arguments.publish:
        publication = publish_evidence(
            store,
            producer_id=arguments.producer,
            publish_destination=arguments.publish_destination,
        )
        payload["publication"] = {
            "destination": publication.destination,
            "observation_count": publication.observation_count,
            "producer_id": publication.producer_id,
            "snapshot_digest": publication.snapshot_digest,
        }
        text += (
            f" Published snapshot {publication.snapshot_digest} to "
            f"{publication.destination}."
        )
    return payload, text


def _status(arguments: argparse.Namespace) -> tuple[dict[str, object], str]:
    if arguments.read_source is not None and not arguments.refresh:
        raise ValueError("--read-source requires --refresh")
    store = DoltEvidenceStore(arguments.store)
    refresh = None
    if arguments.refresh:
        refresh = refresh_evidence(store, read_source=arguments.read_source)
    observations = query_status(
        store,
        architecture=arguments.arch,
        kernel_id=arguments.kernel,
        variant=arguments.variant,
        test_class=arguments.test_class,
        producer_id=arguments.producer,
    )
    payload: dict[str, object] = {
        "architecture": arguments.arch,
        "observations": [item.as_json_object() for item in observations],
        "total": len(observations),
    }
    if refresh is not None:
        payload["refresh"] = {
            "observation_count": refresh.observation_count,
            "read_source": refresh.read_source,
            "snapshot_count": refresh.snapshot_count,
        }
    lines = [f"Observations for architecture {arguments.arch}: {len(observations)}."]
    if refresh is not None:
        lines.insert(
            0,
            f"Refreshed {refresh.snapshot_count} contributor snapshots from "
            f"{refresh.read_source}.",
        )
    lines.extend(
        f"kernel={item.kernel_id} variant={item.variant} class={item.test_class} "
        f"case={item.case_id} outcome={item.outcome} producer={item.producer_id} "
        f"run={item.run_id} observation={item.observation_id}"
        for item in observations
    )
    return payload, "\n".join(lines)


def _stale(arguments: argparse.Namespace) -> tuple[dict[str, object], str]:
    runs = query_stale(DoltEvidenceStore(arguments.store), architecture=arguments.arch)
    payload: dict[str, object] = {
        "architecture": arguments.arch,
        "runs": [run.as_json_object() for run in runs],
        "total": len(runs),
    }
    lines = [f"Stored runs for architecture {arguments.arch}: {len(runs)}."]
    for run in runs:
        if not run.differences:
            lines.append(f"run={run.run_id} identities=current")
            continue
        for difference in run.differences:
            details = ",".join(difference.details) or "none"
            lines.append(
                f"run={run.run_id} axis={difference.axis} "
                f"stored={','.join(difference.stored)} "
                f"current={','.join(difference.current)} details={details}"
            )
    return payload, "\n".join(lines)


def _todo(arguments: argparse.Namespace) -> tuple[dict[str, object], str]:
    missing = query_todo(
        DoltEvidenceStore(arguments.store), architecture=arguments.arch
    )
    payload: dict[str, object] = {
        "architecture": arguments.arch,
        "missing": [item.as_json_object() for item in missing],
        "total": len(missing),
    }
    lines = [f"Missing requirements for architecture {arguments.arch}: {len(missing)}."]
    lines.extend(
        f"kernel={item.kernel_id} variant={item.variant} class={item.test_class} "
        f"case={item.case_id} requirement={item.requirement_id}"
        for item in missing
    )
    return payload, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local kernel-evidence command and return its process status."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        handlers = {
            "record": _record,
            "stale": _stale,
            "status": _status,
            "todo": _todo,
        }
        payload, text = handlers[arguments.command](arguments)
    except (OSError, TypeError, ValueError, VerificationStoreError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
