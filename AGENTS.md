# strideweave

## Project Orientation

Read `README.md` and `INVARIANTS.md` before making changes. The README describes the
project's architecture, carriers, operation dispatch, autograd model, supported public
APIs, interoperability, and current limitations. The invariant registry contains
cross-cutting constraints that must shape code during design and generation, before
linting, tests, or review detect violations.

When planning or reviewing a change, check whether it would make any part of
the README inaccurate or incomplete. Propose corresponding README updates for
changes to architecture, carriers, dtypes, operation dispatch, autograd,
public capabilities, interoperability, development commands, or documented
limitations. When implementing such a change, update the README in the same
change unless the user explicitly excludes documentation work.

Identify the relevant invariant IDs while planning a change. When adding, removing, or
materially changing a cross-cutting invariant, update `INVARIANTS.md` and its stated
enforcement evidence in the same change.

## Tensor Indexing Style

Prefer `tensor[i, j]` style for coordinate indexing in source code and tests.

Use `tensor[[i, j]]` only when intentionally testing or documenting list-coordinate
key support, such as checking that `tensor[[i, j]]` behaves like `tensor[i, j]`.

## Public API Docstrings

Public Python API docstrings must document purpose and semantics, every
relevant input, an appropriate output description, and at least one concrete
usage example. The exact contract depends on the export kind:

- Function exports document purpose, every input, the return value, and an
  example (`Args:`, `Returns:`, `Examples:`).
- Class exports in `strideweave.nn` document purpose, every constructor input
  (including inherited ones such as `name`), and an example (`Args:`,
  `Examples:`). A constructor documents construction rather than a return
  value, so `Returns:` is intentionally not required on the class docstring;
  public methods defined on the class are additionally checked under the full
  function contract.
- Other public class exports surfaced for historical import paths or `isinstance`
  checks — the `strideweave.operation` operation classes (for example
  `GenericAddOperation`, `GenericSubOperation`) and the native carrier/layout
  classes — are implementation and dispatch classes rather than
  constructor-driven user APIs. They are checked for docstring presence only
  and carry a one-line summary; do not add constructor-style `Args:`/`Examples:`
  sections to them.

`tests/test_docstrings.py` enforces this generically over the
public exports listed in `strideweave.__all__`, `strideweave.einops.__all__`,
`strideweave.nn.__all__`, and `strideweave.friendly.__all__`. Adding a Python
function or class to any of these public export lists should not require
changing that test unless it needs a stricter contract.

Modify the docstring test only when:

- adding a new native/pybind export that cannot reasonably carry a Python
  docstring, in which case add it to the explicit native skip set;
- adding a new layout-command parser or tensor operation that should require
  `Syntax:`, `Semantics:`, or `Mode assumptions:` sections, in which case add it
  to the appropriate explicit set;
- changing the public docstring contract itself.

For StrideWeave layout description APIs, avoid wording that implies standard
einops/PyTorch flat-layout semantics. Describe commands as StrideWeave
hierarchical-layout descriptions, and document syntax, semantics, and tensor
mode assumptions explicitly.

Do not write vague or flat-layout docstrings like:

```python
def einsum(lhs, rhs, description):
    """Contract two tensors using an einops-style einsum description."""
```

Prefer docstrings that name StrideWeave semantics and include inputs, output, and
an example:

```python
def einsum(lhs, rhs, description):
    """Contract two tensors using a StrideWeave contraction description.

    Shared symbols are lowered into the second mode of two intermediate layouts
    and contracted with matmul.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor containing the requested contraction result.

    Examples:
        >>> import strideweave as sw
        >>> sw.einsum(lhs, rhs, "a b, c b -> a c")
    """
```

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
