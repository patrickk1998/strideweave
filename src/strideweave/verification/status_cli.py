"""Local-first command line access to schema-v3 verification evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .api import verify_backend
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
from .store.querying import _registered_profile_id

_STORE_HELP = "Local fresh v3 store directory; STRIDEWEAVE_STATUS_HOME replaces the platform-data base."
_V3_HELP = "Schema-v3 only. Existing v2 stores are never migrated: select a new --store path or manually recreate the old path after handling its data."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strideweave-kernel-status",
        description="Record and inspect factual schema-v3 verification evidence.\n\n"
        + _V3_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "record", help="Validate and record one schema-v3 report.", description=_V3_HELP
    )
    record.add_argument("--report", required=True)
    record.add_argument("--producer", required=True)
    record.add_argument("--source-commit")
    record.add_argument("--artifact")
    record.add_argument("--artifact-digest")
    record.add_argument("--store", help=_STORE_HELP)
    record.add_argument("--publish", action="store_true")
    record.add_argument("--publish-destination")
    record.add_argument("--json", action="store_true")
    for name, help_text in (
        ("status", "List unaggregated observations."),
        ("stale", "Explain each independent stale provenance axis."),
        ("todo", "List unranked current requirements without a matching observation."),
    ):
        command = commands.add_parser(
            name,
            help=help_text,
            description=f"{help_text}\n\n{_V3_HELP}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        command.add_argument(
            "--target", required=True, help="Exact selected target profile identifier."
        )
        command.add_argument("--store", help=_STORE_HELP)
        command.add_argument("--json", action="store_true")
        if name == "status":
            command.add_argument("--kernel")
            command.add_argument("--variant")
            command.add_argument("--class", dest="test_class")
            command.add_argument("--case")
            command.add_argument("--producer")
            command.add_argument("--refresh", action="store_true")
            command.add_argument("--read-source")
    return parser


def _record(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    if args.publish_destination and not args.publish:
        raise ValueError("--publish-destination requires --publish")
    report = VerificationReport.load(args.report)
    store = DoltEvidenceStore(args.store)
    result = record_report(
        report,
        store,
        producer_id=args.producer,
        source_commit=args.source_commit,
        artifact_locator=args.artifact,
        artifact_digest=args.artifact_digest,
    )
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "report_digest": result.report_digest,
        "evidence_count": result.evidence_count,
        "observation_count": result.observation_count,
        "store": str(store.path),
    }
    if args.publish:
        publication = publish_evidence(
            store,
            producer_id=args.producer,
            publish_destination=args.publish_destination,
        )
        payload["publication"] = {
            "producer_id": publication.producer_id,
            "snapshot_digest": publication.snapshot_digest,
            "observation_count": publication.observation_count,
            "destination": publication.destination,
        }
    return payload, f"Recorded schema-v3 run {result.run_id}."


def _status(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    if args.read_source and not args.refresh:
        raise ValueError("--read-source requires --refresh")
    store = DoltEvidenceStore(args.store)
    refresh = (
        refresh_evidence(store, read_source=args.read_source) if args.refresh else None
    )
    values = query_status(
        store,
        selected_target_profile=args.target,
        kernel_id=args.kernel,
        variant=args.variant,
        test_class=args.test_class,
        case_id=args.case,
        producer_id=args.producer,
    )
    payload: dict[str, object] = {
        "target": args.target,
        "observations": [item.as_json_object() for item in values],
        "total": len(values),
    }
    if refresh:
        payload["refresh"] = {
            "snapshot_count": refresh.snapshot_count,
            "observation_count": refresh.observation_count,
            "read_source": refresh.read_source,
        }
    return payload, f"Observations for {args.target}: {len(values)}."


def _baseline(target: str) -> VerificationReport:
    return verify_backend(target)


def _stale(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    values = query_stale(
        DoltEvidenceStore(args.store),
        selected_target_profile=args.target,
        current_report=_baseline(args.target),
    )
    return {
        "target": args.target,
        "runs": [item.as_json_object() for item in values],
        "total": len(values),
    }, f"Stored runs for {args.target}: {len(values)}."


def _todo(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    values = query_todo(
        DoltEvidenceStore(args.store),
        selected_target_profile=args.target,
        current_report=_baseline(args.target),
    )
    return {
        "target": args.target,
        "requirements": [item.as_json_object() for item in values],
        "total": len(values),
    }, f"Missing requirements for {args.target}: {len(values)}."


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"status", "stale", "todo"}:
            _registered_profile_id(args.target, "--target")
        payload, text = {
            "record": _record,
            "status": _status,
            "stale": _stale,
            "todo": _todo,
        }[args.command](args)
        print(json.dumps(payload, sort_keys=True) if args.json else text)
        return 0
    except SystemExit as exit_error:
        return exit_error.code if type(exit_error.code) is int else 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        VerificationStoreError,
    ) as error:
        print(f"strideweave-kernel-status: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
