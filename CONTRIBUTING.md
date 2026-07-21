# Contributing

StrideWeave is an early-stage research project. Before proposing a change, read
`README.md` for the current architecture and `AGENTS.md` for repository-specific
engineering and documentation conventions.

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
uv run pytest tests
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

Changes to public behavior or architecture should update `README.md`. Public
Python APIs must follow the docstring contract documented in `AGENTS.md`.

## Pull Requests

Keep changes focused and include tests proportional to their behavioral impact.
Explain compatibility implications, especially for carrier dispatch, layouts,
autograd, native bindings, and public APIs.
