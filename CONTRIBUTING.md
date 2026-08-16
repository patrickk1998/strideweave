# Contributing

StrideWeave is an early-stage research project. Before proposing a change, read
`llms.md` for the architectural mental model, the capability spec under
`openspec/specs/` that owns the behavior you are changing, `INVARIANTS.md` for
cross-cutting design constraints, and `AGENTS.md` for repository-specific engineering
and documentation conventions. The specs are also published at
<https://strideweave.org/spec/>.

`INVARIANTS.md` in particular is design input rather than review material: it records
the canonical implementation choices and whether each invariant is enforced by AST lint,
Ruff, behavioral tests, native builds, or code review, so read it before designing a
change rather than after writing one.

## Development Setup

StrideWeave requires Python 3.12 or newer, uses `uv` for its development
environment, and builds its native modules with scikit-build-core and pybind11.

```bash
git clone https://github.com/patrickk1998/strideweave.git
cd strideweave
uv sync --group dev
```

After changing native C++ sources, rebuild and reinstall StrideWeave in the
active development environment before running the test suite or
`sw.test_backend()`:

```bash
uv sync --reinstall-package strideweave --group dev
```

`uv build` creates a distribution artifact, but it does not reinstall the
editable native extension imported by the active environment. If that extension
is older than the Python verification sources, `sw.test_backend()` fails closed
with the rebuild command instead of skipping native verification.

## Verification

Run the complete local verification suite before opening a pull request:

```bash
uv run pytest tests -m "not dolt_integration and not dolt_lifecycle"
uv run pytest tests -m "dolt_integration or dolt_lifecycle"
uv run pytest --doctest-modules src/strideweave
uv run ruff format --check .
uv run ruff check .
uv run python tools/lint_invariants.py
uv run pyright
uv build
npm ci
npm run duplication
find src/strideweave -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec uv run clang-format --dry-run --Werror {} +
CMAKE_ARGS="-DSTRIDEWEAVE_STRICT_WARNINGS=ON" uv build
git diff --check
```

The repository invariant checker is a dependency-free pass over `src`, `tests`, and
`examples` that uses Python's built-in AST, so it reports StrideWeave-specific source
contracts without importing the package.

The test suite covers layouts, carriers, tensor indexing and mutation,
autograd, operations and activations, hierarchical command parsing, DLPack,
movement, modules, and public docstrings. `tests/test_dtype_conformance.py`
additionally enumerates the operation policy's registry and compares `Generic`
against native `CPU` for every registered operation, so a backend that drifts
from the shared plans fails there rather than in review.

The Python docstring gate collects modules under `src/strideweave` explicitly,
with whitespace normalization but without global ellipsis matching. Pytest's
module collector does not inspect docstrings compiled into pybind extensions,
so the 25 examples in `src/strideweave/core/native/_carrier.cpp` remain a known
native coverage gap pending a custom collector.

`--durations=10` is on by default, so every run reports its slowest steps.

## Test Markers And The Native Boundary

Tests that need a real local Dolt store carry the `dolt_integration` marker and
share one managed server for the whole session, defined by the fixtures in
`tests/conftest.py`. `evidence_store_path` hands each test a unique database in
that one data directory and drops it afterwards, so the tests keep real
creation, migration, transaction, query, publication, refresh, and conflict
coverage while paying for one server rather than one per test. `dolt_lifecycle`
is the server manager's own test suite and deliberately starts, and may
concurrently run, several independently owned servers. Both markers skip when no
Dolt runtime is installed. Select or exclude them with pytest's `-m`:

```bash
uv run pytest tests -m "not dolt_integration and not dolt_lifecycle"
uv run pytest tests -m "dolt_integration or dolt_lifecycle"
```

Do not mark a whole test file. `dolt_integration` is derived from each test's
fixture closure rather than written by hand, so a test leaves the
sanitizer-instrumented suite exactly when it asks for the server it was
deselected for; asking for `evidence_store_path` or another session-server
fixture is what moves a test into that job. Add `dolt_lifecycle` by hand only to
a test that starts server processes of its own.

A marked test may not run native code at all, because the sanitizer job
deselects both markers — their work happens inside an uninstrumented external
Dolt process. See `CPP009` in `INVARIANTS.md` for that deselection and `CPP009a`
for the guard's exact boundary. The
promise that makes this exact is that the deselected selection contains no
native work at all: for the whole of a marked item — fixture setup, call, and teardown
— `tests/conftest.py` replaces every imported binding of `sw.test_backend`,
native kernel metadata, the installed compilation manifest, and report binding
with one that refuses, including module-level aliases that tests or fixtures
captured before the marked protocol began. This is a pragmatic accidental-use
guard over imported module attributes, not an adversarial sandbox over
references hidden in locals, closures, or object state.

Give a marked test the deterministic pure-Python facts from
`tests/synthetic_evidence.py`: a complete, internally consistent provenance
graph — two kernels owned by two sources, closures carrying both project-owned
and external members, a specification, tolerance policies, an oracle reference,
deferred coverage, and a resolved plan — that depends on no installed native
build. Persist them through the internal post-validation boundary in
`recording.py` rather than the public `record_report(...)`, which still fails
closed by rebinding a report against this build before any store is touched.
Cover that public path, report construction, report binding, provenance
reconciliation, the CLI's own validation, and their rejection paths with
unmarked tests against a non-Dolt store, so that coverage stays in the
instrumented job.

`tests/test_native_boundary.py` proves the guard covers every imported binding
of the native entry points — including the compiled
`_cpu_native_kernel_metadata` export itself and aliases captured in test or
fixture modules, not only the Python wrapper that calls it — that it refuses and
restores them, and that the two selections stay disjoint, exhaustive, and
derived. The markers are narrow rather than per file, so the store's pure
helper, in-memory-double, stand-in-runtime, and path-default tests run in both
the regular and sanitizer jobs. Everything else in the store files — SQL
rendering and batching, publication selection and merge over in-memory doubles,
stand-in-runtime failures, and the default store path — stays in the regular
suite.

## Continuous Integration

CI runs five separately visible code checks: `test` (the non-Dolt suite plus
Python docstring examples, formatting, lint, invariants, native formatting,
type checking, and the distribution build), `dolt-integration`,
`native-strict-warnings`, `native-sanitizers`, and `duplication`. A sixth job,
`changes`, gates them. It
classifies the paths a change touches and the five code checks run only when
that classification is anything other than a purely non-code change, so a pull
request that only adds an OpenSpec spec, an agent skill, or repository prose
does not build the extension three times to prove nothing. The non-code set is
`openspec/`, `.agents/`, `.codex/`, `.claude/`, `.beads/`, `docs/`, `assets/`,
`index.html`, `CNAME`, `.nojekyll`, `properdocs.yml`, `skills-lock.json`, and
the root prose files other than `README.md` and `LICENSE`, which the
distribution build consumes. Nothing in that set is read by any of the five;
the one script that reads `openspec/specs/` is `scripts/gen_spec_pages.py`,
which belongs to the specs site workflow and filters its own triggers. Ruff's
own discovery is kept aligned with that set through `extend-exclude` in
`pyproject.toml`, so a Python helper added under a non-code directory cannot
land unformatted and then fail `test` for the next unrelated change.

The gate is structured around two hazards. The first is that `test`,
`duplication`, `native-strict-warnings`, and `native-sanitizers` are required
status checks, and a required check that never reports blocks a pull request
permanently rather than failing it, so the triggers deliberately carry no
`paths` filter: the workflow always starts and the jobs skip individually,
because a job skipped by `if:` reports `skipped` and satisfies the requirement.
The second is that a wrong classification must be wrong in the safe direction.
Classification is therefore a deny list rather than an allow list — an
unrecognised path runs the full CI — `README.md`, `LICENSE`, and `.github/**`
are held out of the non-code set because the distribution build fails without
the first two and a workflow edit has to exercise the third, and a missing base
commit, a failed or unreadable compare response, or a change set large enough
for the compare endpoint to truncate its file list all resolve to running
everything. A rename is classified by both of its endpoints, because the
compare endpoint reports one entry whose `filename` is the new path, so reading
that alone would let a source file move into a documentation directory
unexamined. The dependent jobs compare against `false` rather than `true` and
carry `!cancelled()`, so a gate that crashes or writes no verdict runs the full
CI instead of silently skipping it.

`dolt-integration` is the only job that installs a Dolt runtime — a pinned,
checksum-verified release — and it runs both real-Dolt suites. The
`dolt_integration` suite shares one session server and asserts that exactly one
`dolt sql-server` runs and that no `dolt sql` subprocess is launched; the
`dolt_lifecycle` suite is the manager's own test and deliberately starts, and
may concurrently run, several independently owned servers.

Native sanitizer coverage runs in Linux CI with `STRIDEWEAVE_SANITIZERS=ON`,
instrumenting the extension modules with AddressSanitizer and
UndefinedBehaviorSanitizer. It deselects the two Dolt markers for the reason
given above, so marked tests persist pure-Python evidence while report
construction, binding, provenance reconciliation, and their rejection paths stay
instrumented.

That sanitizer job is the only one that runs pytest in parallel, with `-n auto`
over pytest-xdist; every other job runs serially. Sanitizers cost roughly 6.8x,
and the cost is a long flat tail across the whole suite rather than a few slow
tests, so worker parallelism is what shortens it. Parallelism changes how a
sanitizer diagnosis reaches a human, and the job is arranged around that.
`--capture=no` is unsupported under xdist, so a report can no longer surface
through the worker's own stderr; `ASAN_OPTIONS` and `UBSAN_OPTIONS` therefore
share one `log_path` prefix under `sanitizer-reports/`, where each process
writes `sanitizer.<pid>`. Because `abort_on_error=1` and `halt_on_error=1` kill
the process on the first diagnostic, that file is what survives: xdist names the
test the crashed worker was running, and the report file explains why. Two
`if: failure()` steps then print every report into the log and upload the
directory as a `sanitizer-reports` artifact. The two option strings share one
prefix deliberately — ASan and UBSan share a runtime and therefore the common
`log_path` flag, so separate prefixes would only file an AddressSanitizer report
under a `ubsan` name. The job also exports `STRIDEWEAVE_EXPECT_SANITIZERS=1`,
which makes `tests/conftest.py` assert in the controller and in every worker
that the process actually inherited the preloaded runtime, since a worker that
escaped `LD_PRELOAD` would pass every test while silently reducing the job to an
ordinary test run.

The duplication gate uses the exact `jscpd` version locked by npm and the checked-in
`.jscpd.json` configuration to scan production code under `src/`. The post-binary-
operation-refactor baseline was 4.7% duplicated lines with 5-line/50-token minimum
clones; CI blocks results above 5.0%. The scanner respects `.gitignore`, and the
configuration explicitly excludes non-production, generated, dependency, cache,
report, and build artifacts.

Maintainers: the `dolt-integration` job is new, so add it to the repository
ruleset's required status checks. Until that is done, a pull request can merge
without it having passed.

## Documentation

Behavior is documented spec-first. A change to external behavior updates the owning
capability spec under `openspec/specs/` in the same pull request, through the OpenSpec
workflow described in `AGENTS.md`. Specs are the contract; they state what a conforming
implementation must do for someone who has never seen this repository.

`llms.md` is the architectural mental model, not a second copy of the contract. Update it
when a change makes that model inaccurate — a new or removed subsystem, a changed
relationship between parts, a public capability that needs orienting explanation, or a
documented boundary that no longer holds — and do not restate spec-owned rules there. Its
"Specifications And This Document" section maps each area to its owning spec. Where no
spec owns the behavior yet, `llms.md` is still the statement of record and must be updated
with the change; `AGENTS.md` lists those areas.

When a change adds, removes, or materially changes a cross-cutting invariant, update
`INVARIANTS.md` in the same pull request. Every registry entry must state its canonical
implementation, enforcement type, and stable evidence locations so authors and coding
agents can apply it before code is generated.

Public Python APIs must follow the docstring contract documented in `AGENTS.md`.

## Pull Requests

Keep changes focused and include tests proportional to their behavioral impact.
Explain compatibility implications, especially for carrier dispatch, layouts,
autograd, native bindings, and public APIs.
