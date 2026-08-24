-- Fresh StrideWeave schema-v3 verification-evidence graph. Earlier schemas
-- are deliberately rejected by the adapter before this script can run.
CREATE TABLE schema_migrations (
    version BIGINT NOT NULL, migration_name VARCHAR(255) NOT NULL,
    migration_checksum CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    applied_at_utc DATETIME(6) NOT NULL, PRIMARY KEY (version),
    UNIQUE KEY schema_migrations_name (migration_name)
);
CREATE TABLE verification_runs (
    run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    report_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    report_schema VARCHAR(255) NOT NULL, selected_target_profile VARCHAR(255) NOT NULL,
    oracle_profile VARCHAR(255) NOT NULL, bundle_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    todo_provenance_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    header_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL, report_json LONGTEXT NOT NULL,
    PRIMARY KEY (run_id), UNIQUE KEY verification_runs_report_digest (report_digest),
    KEY verification_runs_todo_provenance (todo_provenance_digest, run_id)
);
CREATE TABLE compilation_receipts (
    receipt_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    receipt_kind VARCHAR(64) NOT NULL, profile_id VARCHAR(255) NOT NULL, provider VARCHAR(255) NOT NULL,
    logical_kernel_id VARCHAR(255) NOT NULL, logical_kernel_variant VARCHAR(255) NOT NULL, receipt_json LONGTEXT NOT NULL,
    PRIMARY KEY (receipt_id)
);
CREATE TABLE run_compilation_receipts (
    run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL, receipt_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    PRIMARY KEY (run_id, receipt_id), FOREIGN KEY (run_id) REFERENCES verification_runs(run_id),
    FOREIGN KEY (receipt_id) REFERENCES compilation_receipts(receipt_id)
);
CREATE TABLE evidence (
    evidence_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL, run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    receipt_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL, requirement_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    stage VARCHAR(64) NOT NULL, test_class VARCHAR(64) NOT NULL, case_id VARCHAR(255) NOT NULL, operation_name VARCHAR(255) NOT NULL,
    kernel_id VARCHAR(255) NOT NULL, variant VARCHAR(255) NOT NULL, outcome VARCHAR(32) NOT NULL, record_json LONGTEXT NOT NULL,
    PRIMARY KEY (evidence_id), KEY evidence_run(run_id), FOREIGN KEY (run_id) REFERENCES verification_runs(run_id),
    FOREIGN KEY (receipt_id) REFERENCES compilation_receipts(receipt_id)
);
CREATE TABLE observations (
    observation_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL, evidence_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    producer_id VARCHAR(255) NOT NULL, source_commit VARCHAR(255) NULL, recorded_at_utc DATETIME(6) NOT NULL,
    artifact_locator VARCHAR(1024) NULL, artifact_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL, observation_json LONGTEXT NOT NULL,
    PRIMARY KEY (observation_id), KEY observations_evidence(evidence_id), KEY observations_producer(producer_id),
    FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
);
