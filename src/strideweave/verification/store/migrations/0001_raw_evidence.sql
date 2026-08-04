-- StrideWeave kernel-evidence store schema v1.
--
-- All 64-character identifiers are lowercase SHA-256 digests of the canonical
-- facts named by their row. There are deliberately no auto-increment keys:
-- identical facts produced on independent contributor branches use identical
-- keys, while distinct observations never contend for a shared sequence.

CREATE TABLE schema_migrations (
    version BIGINT NOT NULL,
    migration_name VARCHAR(255) NOT NULL,
    migration_checksum CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    applied_at_utc DATETIME(6) NOT NULL,
    PRIMARY KEY (version),
    UNIQUE KEY schema_migrations_name (migration_name)
);

CREATE TABLE verification_targets (
    target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    architecture VARCHAR(255) NOT NULL,
    vendor VARCHAR(255) NOT NULL,
    operating_system VARCHAR(255) NOT NULL,
    abi VARCHAR(255) NOT NULL,
    endianness VARCHAR(16) NOT NULL,
    pointer_bits SMALLINT UNSIGNED NOT NULL,
    descriptor_json LONGTEXT NOT NULL,
    PRIMARY KEY (target_id)
);

CREATE TABLE target_proxies (
    proxy_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    execution_target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    represented_target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    proxy_kind VARCHAR(255) NOT NULL,
    statement_json LONGTEXT NOT NULL,
    PRIMARY KEY (proxy_id),
    CONSTRAINT target_proxies_execution_target_fk
        FOREIGN KEY (execution_target_id) REFERENCES verification_targets (target_id),
    CONSTRAINT target_proxies_represented_target_fk
        FOREIGN KEY (represented_target_id) REFERENCES verification_targets (target_id)
);

CREATE TABLE build_toolchains (
    toolchain_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    provider_kind VARCHAR(255) NOT NULL,
    compiler_id VARCHAR(255) NOT NULL,
    compiler_version VARCHAR(255) NOT NULL,
    target_triple VARCHAR(255) NOT NULL,
    build_system VARCHAR(255) NOT NULL,
    descriptor_json LONGTEXT NOT NULL,
    PRIMARY KEY (toolchain_id)
);

CREATE TABLE source_closures (
    closure_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    hash_algorithm VARCHAR(64) NOT NULL,
    root_kind VARCHAR(255) NOT NULL,
    root_uri VARCHAR(1024) NOT NULL,
    descriptor_json LONGTEXT NOT NULL,
    PRIMARY KEY (closure_id)
);

CREATE TABLE source_closure_inputs (
    closure_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    input_ordinal BIGINT UNSIGNED NOT NULL,
    input_kind VARCHAR(255) NOT NULL,
    input_uri VARCHAR(1024) NOT NULL,
    content_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    descriptor_json LONGTEXT NOT NULL,
    PRIMARY KEY (closure_id, input_ordinal),
    CONSTRAINT source_closure_inputs_closure_fk
        FOREIGN KEY (closure_id) REFERENCES source_closures (closure_id)
);

CREATE TABLE kernel_builds (
    kernel_build_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    receipt_schema VARCHAR(255) NOT NULL,
    provider_kind VARCHAR(255) NOT NULL,
    framework_name VARCHAR(255) NULL,
    framework_version VARCHAR(255) NULL,
    kernel_id VARCHAR(255) NOT NULL,
    variant VARCHAR(255) NOT NULL,
    operation_name VARCHAR(255) NOT NULL,
    artifact_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    artifact_locator VARCHAR(1024) NULL,
    compile_invocation_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    toolchain_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    closure_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    specialization_json LONGTEXT NOT NULL,
    receipt_json LONGTEXT NOT NULL,
    PRIMARY KEY (kernel_build_id),
    CONSTRAINT kernel_builds_target_fk
        FOREIGN KEY (target_id) REFERENCES verification_targets (target_id),
    CONSTRAINT kernel_builds_toolchain_fk
        FOREIGN KEY (toolchain_id) REFERENCES build_toolchains (toolchain_id),
    CONSTRAINT kernel_builds_closure_fk
        FOREIGN KEY (closure_id) REFERENCES source_closures (closure_id)
);

CREATE TABLE verification_specs (
    verification_spec_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    spec_schema VARCHAR(255) NOT NULL,
    manifest_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    definition_json LONGTEXT NOT NULL,
    PRIMARY KEY (verification_spec_id)
);

CREATE TABLE verification_requirements (
    requirement_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    verification_spec_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    kernel_id VARCHAR(255) NOT NULL,
    variant VARCHAR(255) NOT NULL,
    operation_name VARCHAR(255) NOT NULL,
    test_class VARCHAR(64) NOT NULL,
    case_id VARCHAR(255) NOT NULL,
    disposition VARCHAR(32) NOT NULL,
    deferred_reason LONGTEXT NULL,
    plan_json LONGTEXT NULL,
    requirement_json LONGTEXT NOT NULL,
    PRIMARY KEY (requirement_id),
    KEY verification_requirements_spec (verification_spec_id),
    CONSTRAINT verification_requirements_spec_fk
        FOREIGN KEY (verification_spec_id)
        REFERENCES verification_specs (verification_spec_id)
);

CREATE TABLE tolerance_policies (
    tolerance_policy_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    policy_schema VARCHAR(255) NOT NULL,
    comparison_kind VARCHAR(64) NOT NULL,
    definition_json LONGTEXT NOT NULL,
    PRIMARY KEY (tolerance_policy_id)
);

CREATE TABLE oracle_references (
    oracle_reference_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    oracle_kind VARCHAR(255) NOT NULL,
    implementation_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    source_closure_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    kernel_build_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    descriptor_json LONGTEXT NOT NULL,
    PRIMARY KEY (oracle_reference_id),
    CONSTRAINT oracle_references_closure_fk
        FOREIGN KEY (source_closure_id) REFERENCES source_closures (closure_id),
    CONSTRAINT oracle_references_kernel_build_fk
        FOREIGN KEY (kernel_build_id) REFERENCES kernel_builds (kernel_build_id)
);

CREATE TABLE verification_runs (
    run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    report_schema VARCHAR(255) NOT NULL,
    report_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    native_manifest_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    verification_spec_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    execution_target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    represented_target_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    proxy_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    report_json LONGTEXT NOT NULL,
    PRIMARY KEY (run_id),
    UNIQUE KEY verification_runs_report_digest (report_digest),
    CONSTRAINT verification_runs_spec_fk
        FOREIGN KEY (verification_spec_id)
        REFERENCES verification_specs (verification_spec_id),
    CONSTRAINT verification_runs_execution_target_fk
        FOREIGN KEY (execution_target_id) REFERENCES verification_targets (target_id),
    CONSTRAINT verification_runs_represented_target_fk
        FOREIGN KEY (represented_target_id) REFERENCES verification_targets (target_id),
    CONSTRAINT verification_runs_proxy_fk
        FOREIGN KEY (proxy_id) REFERENCES target_proxies (proxy_id)
);

CREATE TABLE run_kernel_builds (
    run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    kernel_build_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    PRIMARY KEY (run_id, kernel_build_id),
    CONSTRAINT run_kernel_builds_run_fk
        FOREIGN KEY (run_id) REFERENCES verification_runs (run_id),
    CONSTRAINT run_kernel_builds_build_fk
        FOREIGN KEY (kernel_build_id) REFERENCES kernel_builds (kernel_build_id)
);

CREATE TABLE evidence (
    evidence_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    run_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    requirement_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    kernel_build_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    tolerance_policy_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    oracle_reference_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    consumed_certificate_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    stage VARCHAR(64) NOT NULL,
    test_class VARCHAR(64) NOT NULL,
    case_id VARCHAR(255) NOT NULL,
    operation_name VARCHAR(255) NOT NULL,
    kernel_id VARCHAR(255) NOT NULL,
    variant VARCHAR(255) NOT NULL,
    outcome VARCHAR(32) NOT NULL,
    input_payload_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    target_input_hashes_json LONGTEXT NOT NULL,
    oracle_input_hashes_json LONGTEXT NOT NULL,
    plan_json LONGTEXT NULL,
    shapes_json LONGTEXT NOT NULL,
    deviations_json LONGTEXT NOT NULL,
    mismatch_count BIGINT UNSIGNED NULL,
    diagnostic LONGTEXT NULL,
    evidence_json LONGTEXT NOT NULL,
    PRIMARY KEY (evidence_id),
    KEY evidence_run (run_id),
    KEY evidence_requirement (requirement_id),
    CONSTRAINT evidence_run_fk
        FOREIGN KEY (run_id) REFERENCES verification_runs (run_id),
    CONSTRAINT evidence_requirement_fk
        FOREIGN KEY (requirement_id)
        REFERENCES verification_requirements (requirement_id),
    CONSTRAINT evidence_kernel_build_fk
        FOREIGN KEY (kernel_build_id) REFERENCES kernel_builds (kernel_build_id),
    CONSTRAINT evidence_tolerance_fk
        FOREIGN KEY (tolerance_policy_id)
        REFERENCES tolerance_policies (tolerance_policy_id),
    CONSTRAINT evidence_oracle_fk
        FOREIGN KEY (oracle_reference_id)
        REFERENCES oracle_references (oracle_reference_id)
);

CREATE TABLE observations (
    observation_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    evidence_id CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    producer_id VARCHAR(255) NOT NULL,
    source_commit VARCHAR(255) NULL,
    recorded_at_utc DATETIME(6) NOT NULL,
    artifact_locator VARCHAR(1024) NULL,
    artifact_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    observation_json LONGTEXT NOT NULL,
    PRIMARY KEY (observation_id),
    KEY observations_evidence (evidence_id),
    KEY observations_producer (producer_id),
    CONSTRAINT observations_evidence_fk
        FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id)
);
