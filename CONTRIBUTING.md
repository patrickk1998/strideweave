# Contributing

StrideWeave is an early-stage research project. Before proposing a change, read
`llms.md` for the current architecture, `INVARIANTS.md` for cross-cutting design
constraints, and `AGENTS.md` for repository-specific engineering and documentation
conventions.

## Development Setup

StrideWeave requires Python 3.12 or newer and uses `uv` for its development
environment.

```bash
git clone https://github.com/patrickk1998/strideweave.git
cd strideweave
uv sync --group dev
```

## Verification

Run the complete local verification suite before opening a pull request:

```bash
uv run pytest tests -m "not dolt_integration and not dolt_lifecycle"
uv run pytest tests -m "dolt_integration or dolt_lifecycle"
uv run ruff format --check .
uv run ruff check .
uv run python tools/lint_invariants.py
uv run pyright
uv build
find src/strideweave -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec uv run clang-format --dry-run --Werror {} +
CMAKE_ARGS="-DSTRIDEWEAVE_STRICT_WARNINGS=ON" uv build
git diff --check
```

The repository invariant checker is a dependency-free AST pass over `src`, `tests`, and
`examples`. CI also builds the native extensions with strict compiler warnings and runs
a separate Linux AddressSanitizer/UndefinedBehaviorSanitizer test job.

The two marked test suites need a local Dolt runtime and skip without one. They
run in CI's own `dolt-integration` job, which is the only job that installs
Dolt. `dolt_integration` shares one managed server for the whole session;
`dolt_lifecycle` is the server manager's own test and starts several
independently owned servers. The sanitizer job deselects both markers because
their work happens inside an uninstrumented external Dolt process; see `CPP009`
in `INVARIANTS.md` for the exact boundary.

Do not mark a whole test file. `dolt_integration` is derived in
`tests/conftest.py` from the fixtures a test resolves, so asking for
`evidence_store_path` or another session-server fixture is what moves a test
into that job; add `dolt_lifecycle` by hand only to a test that starts server
processes of its own.

A marked test may not run native code at all, and the guard in
`tests/conftest.py` enforces that: throughout a marked item's setup, call, and
teardown, every binding of `sw.test_backend`, native kernel metadata,
`load_compilation_manifest`, and `bind_report` refuses. Give a marked test the
deterministic pure-Python facts from `tests/synthetic_evidence.py` and persist
them through the internal post-validation boundary in the recording module.
Cover the public `record_report`, report binding, and provenance reconciliation
with unmarked tests against a non-Dolt store, so that coverage stays in the
instrumented job.

Maintainers: the `dolt-integration` job is new, so add it to the repository
ruleset's required status checks. Until that is done, a pull request can merge
without it having passed.

When a change adds, removes, or materially changes a cross-cutting invariant, update
`INVARIANTS.md` in the same pull request. Every registry entry must state its canonical
implementation, enforcement type, and stable evidence locations so authors and coding
agents can apply it before code is generated.

Changes to public behavior or architecture should update `llms.md`. Public
Python APIs must follow the docstring contract documented in `AGENTS.md`.

## Pull Requests

Keep changes focused and include tests proportional to their behavioral impact.
Explain compatibility implications, especially for carrier dispatch, layouts,
autograd, native bindings, and public APIs.
