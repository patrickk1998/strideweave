"""Explicit contributor snapshot publication and refresh."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote, urlparse

from ..model import VerificationReport
from . import recording
from .base import (
    EvidenceStore,
    SQLStatement,
    VerificationStoreError,
    is_diagnostic_closure_input,
)

_SNAPSHOT_SCHEMA = "strideweave.kernel-evidence-snapshot.v1"
_PUBLISH_ENVIRONMENT = "STRIDEWEAVE_STATUS_PUBLISH_DESTINATION"
_READ_ENVIRONMENT = "STRIDEWEAVE_STATUS_READ_SOURCE"
# Dolt receives read-only queries through one ``--query`` process argument. Keep
# publication and refresh lookups comfortably below macOS's 1 MiB ARG_MAX even
# after accounting for the executable arguments and environment inherited by
# the child process.
_QUERY_ARGUMENT_BUDGET = 256 * 1024
_TABLE_COLUMNS = MappingProxyType(
    {
        "verification_targets": (
            "target_id",
            "architecture",
            "vendor",
            "operating_system",
            "abi",
            "endianness",
            "pointer_bits",
            "descriptor_json",
        ),
        "target_proxies": (
            "proxy_id",
            "execution_target_id",
            "represented_target_id",
            "proxy_kind",
            "statement_json",
        ),
        "build_toolchains": (
            "toolchain_id",
            "provider_kind",
            "compiler_id",
            "compiler_version",
            "target_triple",
            "build_system",
            "descriptor_json",
        ),
        "source_closures": (
            "closure_id",
            "hash_algorithm",
            "root_kind",
            "root_uri",
            "descriptor_json",
        ),
        "source_closure_inputs": (
            "closure_id",
            "input_ordinal",
            "input_kind",
            "input_uri",
            "content_digest",
            "descriptor_json",
        ),
        "kernel_builds": (
            "kernel_build_id",
            "receipt_schema",
            "provider_kind",
            "framework_name",
            "framework_version",
            "kernel_id",
            "variant",
            "operation_name",
            "artifact_digest",
            "artifact_locator",
            "compile_invocation_digest",
            "target_id",
            "toolchain_id",
            "closure_id",
            "specialization_json",
            "receipt_json",
        ),
        "verification_specs": (
            "verification_spec_id",
            "spec_schema",
            "manifest_digest",
            "definition_json",
        ),
        "verification_requirements": (
            "requirement_id",
            "verification_spec_id",
            "kernel_id",
            "variant",
            "operation_name",
            "test_class",
            "case_id",
            "disposition",
            "deferred_reason",
            "plan_json",
            "requirement_json",
        ),
        "tolerance_policies": (
            "tolerance_policy_id",
            "policy_schema",
            "comparison_kind",
            "definition_json",
        ),
        "oracle_references": (
            "oracle_reference_id",
            "oracle_kind",
            "implementation_digest",
            "source_closure_id",
            "kernel_build_id",
            "descriptor_json",
        ),
        "verification_runs": (
            "run_id",
            "report_schema",
            "report_digest",
            "native_manifest_digest",
            "verification_spec_id",
            "execution_target_id",
            "represented_target_id",
            "proxy_id",
            "report_json",
        ),
        "run_kernel_builds": ("run_id", "kernel_build_id"),
        "evidence": (
            "evidence_id",
            "run_id",
            "requirement_id",
            "kernel_build_id",
            "tolerance_policy_id",
            "oracle_reference_id",
            "consumed_certificate_digest",
            "stage",
            "test_class",
            "case_id",
            "operation_name",
            "kernel_id",
            "variant",
            "outcome",
            "input_payload_digest",
            "target_input_hashes_json",
            "oracle_input_hashes_json",
            "plan_json",
            "shapes_json",
            "deviations_json",
            "mismatch_count",
            "diagnostic",
            "evidence_json",
        ),
        "observations": (
            "observation_id",
            "evidence_id",
            "producer_id",
            "source_commit",
            "recorded_at_utc",
            "artifact_locator",
            "artifact_digest",
            "observation_json",
        ),
    }
)
_PRIMARY_KEYS = MappingProxyType(
    {
        "verification_targets": ("target_id",),
        "target_proxies": ("proxy_id",),
        "build_toolchains": ("toolchain_id",),
        "source_closures": ("closure_id",),
        "source_closure_inputs": ("closure_id", "input_ordinal"),
        "kernel_builds": ("kernel_build_id",),
        "verification_specs": ("verification_spec_id",),
        "verification_requirements": ("requirement_id",),
        "tolerance_policies": ("tolerance_policy_id",),
        "oracle_references": ("oracle_reference_id",),
        "verification_runs": ("run_id",),
        "run_kernel_builds": ("run_id", "kernel_build_id"),
        "evidence": ("evidence_id",),
        "observations": ("observation_id",),
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise VerificationStoreError(f"{field} must be a non-empty string")
    return value


def _row_text(row: Mapping[str, object], field: str) -> str:
    return _text(row.get(field), field)


def _query_table(
    store: EvidenceStore,
    table: str,
    where: str,
    parameters: tuple[object, ...],
) -> list[dict[str, object]]:
    columns = _TABLE_COLUMNS[table]
    rows = store.query(
        SQLStatement(
            _query_template(table, where),
            parameters,  # type: ignore[arg-type]
        )
    )
    return [{column: row.get(column) for column in columns} for row in rows]


def _query_template(table: str, where: str) -> str:
    columns = _TABLE_COLUMNS[table]
    order = ", ".join(_PRIMARY_KEYS[table])
    return f"SELECT {', '.join(columns)} FROM {table} WHERE {where} ORDER BY {order}"


def _query_literal_size(value: object) -> int:
    if type(value) is str:
        return len("CONVERT(0x USING utf8mb4)") + 2 * len(value.encode("utf-8"))
    if type(value) is int:
        return len(str(value).encode("utf-8"))
    raise VerificationStoreError("snapshot query values must be strings or integers")


def _query_term_batches(
    table: str,
    terms: Sequence[tuple[str, tuple[object, ...]]],
    *,
    prefix: str = "",
    separator: str,
    suffix: str = "",
    subject: str,
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    empty_query_size = len(_query_template(table, prefix + suffix).encode("utf-8"))
    batches: list[tuple[str, tuple[object, ...]]] = []
    batch_terms: list[str] = []
    batch_parameters: list[object] = []
    rendered_size = empty_query_size
    for term, parameters in terms:
        placeholder_count = term.count("?")
        if placeholder_count != len(parameters):  # pragma: no cover - internal.
            raise VerificationStoreError("snapshot query term shape is invalid")
        separator_size = len(separator.encode("utf-8")) if batch_terms else 0
        addition_size = (
            separator_size
            + len(term.encode("utf-8"))
            - placeholder_count
            + sum(_query_literal_size(value) for value in parameters)
        )
        if batch_terms and rendered_size + addition_size > _QUERY_ARGUMENT_BUDGET:
            batches.append(
                (
                    prefix + separator.join(batch_terms) + suffix,
                    tuple(batch_parameters),
                )
            )
            batch_terms = []
            batch_parameters = []
            rendered_size = empty_query_size
            addition_size -= separator_size
        if rendered_size + addition_size > _QUERY_ARGUMENT_BUDGET:
            raise VerificationStoreError(
                f"snapshot {table} {subject} exceeds the query argument budget"
            )
        batch_terms.append(term)
        batch_parameters.extend(parameters)
        rendered_size += addition_size
    if batch_terms:
        batches.append(
            (
                prefix + separator.join(batch_terms) + suffix,
                tuple(batch_parameters),
            )
        )
    return tuple(batches)


def _key_query_batches(
    table: str, keys: set[tuple[object, ...]]
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    fields = _PRIMARY_KEYS[table]
    clause = "(" + " AND ".join(f"{field} = ?" for field in fields) + ")"
    ordered_keys = sorted(keys, key=lambda item: tuple(str(value) for value in item))
    terms = []
    for key in ordered_keys:
        if len(key) != len(fields):  # pragma: no cover - internal table contract.
            raise VerificationStoreError("snapshot primary-key shape is invalid")
        terms.append((clause, key))
    return _query_term_batches(
        table,
        terms,
        separator=" OR ",
        subject="identity",
    )


def _value_query_batches(
    table: str, column: str, values: set[str]
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    terms = [("?", (value,)) for value in sorted(values)]
    return _query_term_batches(
        table,
        terms,
        prefix=f"{column} IN (",
        separator=", ",
        suffix=")",
        subject="selection value",
    )


def _query_sort_value(value: object) -> tuple[int, str | int]:
    if type(value) is str:
        return (0, value)
    if type(value) is int:
        return (1, value)
    raise VerificationStoreError(
        "snapshot primary-key values must be strings or integers"
    )


def _query_row_sort_key(
    table: str, row: Mapping[str, object]
) -> tuple[tuple[int, str | int], ...]:
    return tuple(_query_sort_value(row.get(field)) for field in _PRIMARY_KEYS[table])


def _query_by_values(
    store: EvidenceStore, table: str, column: str, values: set[str]
) -> list[dict[str, object]]:
    rows = []
    for where, parameters in _value_query_batches(table, column, values):
        rows.extend(_query_table(store, table, where, parameters))
    rows.sort(key=lambda row: _query_row_sort_key(table, row))
    return rows


def _optional_texts(rows: Sequence[Mapping[str, object]], field: str) -> set[str]:
    return {
        value
        for row in rows
        if (value := row.get(field)) is not None and type(value) is str
    }


def _snapshot_rows(
    store: EvidenceStore, producer_id: str
) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {
        table: [] for table in _TABLE_COLUMNS
    }
    selected["observations"] = _query_table(
        store, "observations", "producer_id = ?", (producer_id,)
    )
    evidence_ids = {_row_text(row, "evidence_id") for row in selected["observations"]}
    selected["evidence"] = _query_by_values(
        store, "evidence", "evidence_id", evidence_ids
    )
    run_ids = {_row_text(row, "run_id") for row in selected["evidence"]}
    selected["verification_runs"] = _query_by_values(
        store, "verification_runs", "run_id", run_ids
    )
    selected["run_kernel_builds"] = _query_by_values(
        store, "run_kernel_builds", "run_id", run_ids
    )
    build_ids = {
        _row_text(row, "kernel_build_id") for row in selected["run_kernel_builds"]
    }
    build_ids.update(_optional_texts(selected["evidence"], "kernel_build_id"))
    oracle_ids = _optional_texts(selected["evidence"], "oracle_reference_id")
    selected["oracle_references"] = _query_by_values(
        store, "oracle_references", "oracle_reference_id", oracle_ids
    )
    build_ids.update(_optional_texts(selected["oracle_references"], "kernel_build_id"))
    selected["kernel_builds"] = _query_by_values(
        store, "kernel_builds", "kernel_build_id", build_ids
    )
    closure_ids = {_row_text(row, "closure_id") for row in selected["kernel_builds"]}
    closure_ids.update(
        _optional_texts(selected["oracle_references"], "source_closure_id")
    )
    selected["source_closures"] = _query_by_values(
        store, "source_closures", "closure_id", closure_ids
    )
    # A store an earlier version wrote may still hold external members. They
    # are not part of what this version records, so they are not republished:
    # a snapshot carrying them would disagree with the facts its own embedded
    # reports rebuild.
    selected["source_closure_inputs"] = [
        row
        for row in _query_by_values(
            store, "source_closure_inputs", "closure_id", closure_ids
        )
        if is_diagnostic_closure_input(row.get("input_kind"))
    ]
    toolchain_ids = {
        _row_text(row, "toolchain_id") for row in selected["kernel_builds"]
    }
    selected["build_toolchains"] = _query_by_values(
        store, "build_toolchains", "toolchain_id", toolchain_ids
    )
    target_ids = {
        value
        for row in selected["verification_runs"]
        for field in ("execution_target_id", "represented_target_id")
        if (value := row.get(field)) is not None and type(value) is str
    }
    target_ids.update(_row_text(row, "target_id") for row in selected["kernel_builds"])
    selected["verification_targets"] = _query_by_values(
        store, "verification_targets", "target_id", target_ids
    )
    proxy_ids = _optional_texts(selected["verification_runs"], "proxy_id")
    selected["target_proxies"] = _query_by_values(
        store, "target_proxies", "proxy_id", proxy_ids
    )
    specification_ids = {
        _row_text(row, "verification_spec_id") for row in selected["verification_runs"]
    }
    selected["verification_specs"] = _query_by_values(
        store, "verification_specs", "verification_spec_id", specification_ids
    )
    selected["verification_requirements"] = _query_by_values(
        store,
        "verification_requirements",
        "verification_spec_id",
        specification_ids,
    )
    policy_ids = _optional_texts(selected["evidence"], "tolerance_policy_id")
    selected["tolerance_policies"] = _query_by_values(
        store, "tolerance_policies", "tolerance_policy_id", policy_ids
    )
    return selected


def _snapshot(store: EvidenceStore, producer_id: str) -> dict[str, object]:
    producer = _text(producer_id, "producer_id")
    tables = _snapshot_rows(store, producer)
    if not tables["observations"]:
        raise VerificationStoreError(
            f"verification store has no observations for producer {producer!r}"
        )
    identity: dict[str, object] = {
        "producer_id": producer,
        "schema_version": _SNAPSHOT_SCHEMA,
        "tables": tables,
    }
    return {**identity, "snapshot_digest": _digest(identity)}


def _configured_endpoint(
    value: str | os.PathLike[str] | None, *, environment: str, field: str
) -> Path:
    configured = os.environ.get(environment) if value is None else os.fspath(value)
    if not configured:
        raise VerificationStoreError(
            f"{field} is not configured; pass it explicitly or set {environment}"
        )
    parsed = urlparse(configured)
    if parsed.scheme not in ("", "file"):
        raise VerificationStoreError(
            f"{field} uses unsupported transport {parsed.scheme!r}; "
            "this installation supports local and file endpoints"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise VerificationStoreError(
                f"{field} file endpoint must name the local host"
            )
        return Path(unquote(parsed.path)).expanduser()
    return Path(configured).expanduser()


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Factual result of one explicit contributor publication."""

    producer_id: str
    snapshot_digest: str
    observation_count: int
    destination: str


def publish_evidence(
    store: EvidenceStore,
    *,
    producer_id: str,
    publish_destination: str | os.PathLike[str] | None = None,
) -> PublishResult:
    """Publish one producer's immutable facts to a configured endpoint."""

    snapshot = _snapshot(store, producer_id)
    root = _configured_endpoint(
        publish_destination,
        environment=_PUBLISH_ENVIRONMENT,
        field="publish_destination",
    )
    namespace = hashlib.sha256(producer_id.encode()).hexdigest()
    destination = root / "contributors" / namespace
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_digest = _text(snapshot["snapshot_digest"], "snapshot_digest")
    path = destination / "current.json"
    text = _canonical_json(snapshot) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=".snapshot-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    tables = snapshot["tables"]
    if type(tables) is not dict:  # pragma: no cover - constructed above.
        raise VerificationStoreError("constructed publication snapshot is malformed")
    observations = tables["observations"]
    return PublishResult(
        producer_id=producer_id,
        snapshot_digest=snapshot_digest,
        observation_count=len(observations),
        destination=str(root),
    )


def _parse_snapshot(text: str, source: Path) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise VerificationStoreError(
            f"read_source contains malformed snapshot {source.name!r}"
        ) from error
    if type(value) is not dict or set(value) != {
        "producer_id",
        "schema_version",
        "snapshot_digest",
        "tables",
    }:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has invalid fields"
        )
    if value["schema_version"] != _SNAPSHOT_SCHEMA:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has unsupported schema"
        )
    identity = {key: value[key] for key in ("producer_id", "schema_version", "tables")}
    if _digest(identity) != value["snapshot_digest"]:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has a mismatched digest"
        )
    producer = _text(value["producer_id"], "snapshot producer_id")
    expected_namespace = hashlib.sha256(producer.encode()).hexdigest()
    if source.parent.name != expected_namespace:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} is in the wrong producer namespace"
        )
    if source.name != "current.json" and source.stem != value["snapshot_digest"]:
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} filename does not match its digest"
        )
    tables = value["tables"]
    if type(tables) is not dict or set(tables) != set(_TABLE_COLUMNS):
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} has invalid tables"
        )
    for table, columns in _TABLE_COLUMNS.items():
        rows = tables[table]
        if type(rows) is not list:
            raise VerificationStoreError(
                f"read_source snapshot {source.name!r} table {table!r} is not an array"
            )
        for row in rows:
            if type(row) is not dict or set(row) != set(columns):
                raise VerificationStoreError(
                    f"read_source snapshot {source.name!r} table {table!r} "
                    "has an invalid row"
                )
            if any(
                value is not None and type(value) not in (str, int, float, bool)
                for value in row.values()
            ):
                raise VerificationStoreError(
                    f"read_source snapshot {source.name!r} table {table!r} "
                    "has an unsupported value"
                )
    if any(row["producer_id"] != producer for row in tables["observations"]):
        raise VerificationStoreError(
            f"read_source snapshot {source.name!r} crosses producer namespaces"
        )
    _validate_snapshot_facts(value, source)
    return value


def _parse_canonical_json(value: object, field: str) -> object:
    if type(value) is not str:
        raise VerificationStoreError(f"{field} must be canonical JSON text")
    try:
        decoded = json.loads(value)
        canonical = _canonical_json(decoded)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise VerificationStoreError(f"{field} must be canonical JSON text") from error
    if canonical != value:
        raise VerificationStoreError(f"{field} must use canonical JSON encoding")
    return decoded


def _statement_row(statement: SQLStatement) -> tuple[str, dict[str, object]]:
    prefix = "INSERT IGNORE INTO "
    if not statement.template.startswith(prefix):  # pragma: no cover - internal.
        raise VerificationStoreError("report fact builder emitted an invalid statement")
    table, _, remainder = statement.template.removeprefix(prefix).partition(" (")
    columns_text, separator, _ = remainder.partition(") VALUES")
    if not separator or table not in _TABLE_COLUMNS:  # pragma: no cover - internal.
        raise VerificationStoreError("report fact builder emitted an invalid statement")
    columns = tuple(item.strip() for item in columns_text.split(","))
    if columns != _TABLE_COLUMNS[table]:  # pragma: no cover - internal.
        raise VerificationStoreError("report fact columns disagree with the snapshot")
    return table, dict(zip(columns, statement.parameters, strict=True))


def _expected_report_rows(
    report: VerificationReport,
) -> dict[str, list[dict[str, object]]]:
    manifest = recording._report_manifest(report)
    statements = [
        *recording._target_statements(manifest),
        *recording._compilation_statements(manifest),
        *recording._specification_statements(report),
        *recording._policy_and_oracle_statements(report),
    ]
    _, _, run_statements = recording._run_and_evidence_statements(
        report,
        manifest,
        producer_id="snapshot-validation",
        source_commit=None,
        recorded_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
        artifact_locator=None,
        artifact_digest=None,
    )
    statements.extend(run_statements)
    expected = {table: [] for table in _TABLE_COLUMNS}
    for statement in statements:
        table, row = _statement_row(statement)
        if table != "observations":
            expected[table].append(row)
    return expected


def _validate_observation(
    row: Mapping[str, object], producer: str, evidence_ids: set[object]
) -> None:
    if row["evidence_id"] not in evidence_ids:
        raise VerificationStoreError("snapshot observation references missing evidence")
    (
        validated_producer,
        source_commit,
        artifact_locator,
        artifact_digest,
        recorded_at,
    ) = recording._validate_observation_provenance(
        producer_id=row["producer_id"],
        source_commit=row["source_commit"],
        artifact_locator=row["artifact_locator"],
        artifact_digest=row["artifact_digest"],
        recorded_at=row["recorded_at_utc"],
    )
    if validated_producer != producer:
        raise VerificationStoreError("snapshot observation crosses producer namespaces")
    identity = {
        "artifact_digest": artifact_digest,
        "artifact_locator": artifact_locator,
        "evidence_id": row["evidence_id"],
        "producer_id": validated_producer,
        "source_commit": source_commit,
    }
    expected_id = _digest(identity)
    if row["observation_id"] != expected_id:
        raise VerificationStoreError("snapshot observation identity does not match")
    value = _parse_canonical_json(row["observation_json"], "observation_json")
    expected = {
        **identity,
        "observation_id": expected_id,
        "recorded_at_utc": value.get("recorded_at_utc")
        if isinstance(value, Mapping)
        else None,
    }
    if value != expected:
        raise VerificationStoreError("snapshot observation JSON disagrees with its row")
    expected_time = recording._recording_time(
        expected["recorded_at_utc"], "observation_json time"
    )
    if recorded_at != expected_time:
        raise VerificationStoreError("snapshot observation timestamp does not match")


def _validate_snapshot_facts(snapshot: Mapping[str, object], source: Path) -> None:
    tables = snapshot["tables"]
    if not isinstance(tables, Mapping):  # pragma: no cover - parsed immediately above.
        raise VerificationStoreError("parsed publication snapshot is malformed")
    producer = _text(snapshot["producer_id"], "snapshot producer_id")
    observed: dict[str, dict[tuple[object, ...], dict[str, object]]] = {
        table: {} for table in _TABLE_COLUMNS
    }
    for table, columns in _TABLE_COLUMNS.items():
        for raw_row in tables[table]:  # type: ignore[index, union-attr]
            row = dict(raw_row)
            key = _row_key(table, row)
            if key in observed[table]:
                raise VerificationStoreError(
                    f"read_source snapshot {source.name!r} has duplicate {table} identity"
                )
            observed[table][key] = row
            for column in columns:
                if column.endswith("_json") and column != "report_json":
                    value = row[column]
                    if value is not None:
                        _parse_canonical_json(value, f"{table}.{column}")

    expected: dict[str, dict[tuple[object, ...], dict[str, object]]] = {
        table: {} for table in _TABLE_COLUMNS
    }
    for run in observed["verification_runs"].values():
        report_text = _text(run["report_json"], "verification_runs.report_json")
        try:
            report = VerificationReport.from_jsonl(report_text)
        except (TypeError, ValueError) as error:
            raise VerificationStoreError(
                "snapshot contains an invalid embedded verification report"
            ) from error
        for table, rows in _expected_report_rows(report).items():
            for row in rows:
                key = _row_key(table, row)
                previous = expected[table].get(key)
                if previous is not None and previous != row:
                    raise VerificationStoreError(
                        f"snapshot reports conflict on immutable {table} identity"
                    )
                expected[table][key] = row

    for table in _TABLE_COLUMNS:
        if table == "observations":
            continue
        if observed[table] != expected[table]:
            raise VerificationStoreError(
                f"snapshot {table} facts do not match its embedded reports"
            )
    evidence_ids = {key[0] for key in observed["evidence"]}
    for observation in observed["observations"].values():
        _validate_observation(observation, producer, evidence_ids)


def _row_key(table: str, row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in _PRIMARY_KEYS[table])


def _merge_statements(
    store: EvidenceStore, snapshots: Sequence[Mapping[str, object]]
) -> tuple[SQLStatement, ...]:
    incoming_keys: dict[str, set[tuple[object, ...]]] = {
        table: set() for table in _TABLE_COLUMNS
    }
    for snapshot in snapshots:
        tables = snapshot["tables"]
        if not isinstance(tables, Mapping):  # pragma: no cover - parsed above.
            raise VerificationStoreError("parsed publication snapshot is malformed")
        for table in _TABLE_COLUMNS:
            incoming_keys[table].update(
                _row_key(table, row)
                for row in tables[table]  # type: ignore[index, union-attr]
            )
    existing: dict[str, list[dict[str, object]]] = {
        table: [] for table in _TABLE_COLUMNS
    }
    for table, keys in incoming_keys.items():
        for where, parameters in _key_query_batches(table, keys):
            existing[table].extend(_query_table(store, table, where, parameters))
    known = {
        (table, _row_key(table, row)): row
        for table, rows in existing.items()
        for row in rows
    }
    statements = []
    for snapshot in snapshots:
        tables = snapshot["tables"]
        if not isinstance(tables, Mapping):  # pragma: no cover - parsed above.
            raise VerificationStoreError("parsed publication snapshot is malformed")
        for table, columns in _TABLE_COLUMNS.items():
            rows = tables[table]
            if not isinstance(rows, Sequence):  # pragma: no cover - parsed above.
                raise VerificationStoreError("parsed publication rows are malformed")
            for raw_row in rows:
                if not isinstance(raw_row, Mapping):  # pragma: no cover
                    raise VerificationStoreError("parsed publication row is malformed")
                row = dict(raw_row)
                identity = (table, _row_key(table, row))
                previous = known.get(identity)
                if previous is not None:
                    if previous != row:
                        raise VerificationStoreError(
                            f"refresh found conflicting immutable {table} identity "
                            f"{identity[1]!r}"
                        )
                    continue
                known[identity] = row
                statements.append(
                    SQLStatement(
                        f"INSERT INTO {table} ({', '.join(columns)}) "
                        f"VALUES ({', '.join('?' for _ in columns)})",
                        tuple(row[column] for column in columns),  # type: ignore[arg-type]
                    )
                )
    return tuple(statements)


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Factual result of one explicit contributor refresh."""

    snapshot_count: int
    observation_count: int
    read_source: str


def refresh_evidence(
    store: EvidenceStore,
    *,
    read_source: str | os.PathLike[str] | None = None,
) -> RefreshResult:
    """Validate and atomically merge all configured contributor snapshots."""

    root = _configured_endpoint(
        read_source, environment=_READ_ENVIRONMENT, field="read_source"
    )
    contributor_root = root / "contributors"
    if not contributor_root.is_dir():
        raise VerificationStoreError(
            f"read_source {str(root)!r} has no published contributor snapshots"
        )
    paths = tuple(sorted(contributor_root.glob("*/current.json")))
    if not paths:
        raise VerificationStoreError(
            f"read_source {str(root)!r} has no published contributor snapshots"
        )
    snapshots = tuple(
        _parse_snapshot(path.read_text(encoding="utf-8"), path) for path in paths
    )
    statements = _merge_statements(store, snapshots)
    store.execute_transaction(statements)
    observation_count = sum(
        len(snapshot["tables"]["observations"])  # type: ignore[index, arg-type]
        for snapshot in snapshots
    )
    return RefreshResult(
        snapshot_count=len(snapshots),
        observation_count=observation_count,
        read_source=str(root),
    )
