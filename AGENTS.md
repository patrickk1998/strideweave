# strideweave

## Project Orientation

Read `llms.md` and `INVARIANTS.md` before making changes. `llms.md` is the
architectural mental model — carriers, operation dispatch, autograd, the public
surface, interoperability, and current boundaries. It is deliberately lossy about
behavior that a spec under `openspec/specs/` already owns; its "Specifications And
This Document" section maps each area to its owning spec, and where the two
disagree the spec is authoritative. The invariant registry contains cross-cutting
constraints that must shape code during design and generation, before linting,
tests, or review detect violations. Contributor workflow — environment setup,
verification commands, test markers, and CI — lives in `CONTRIBUTING.md`.

When a change alters external behavior, update the owning spec in the same change,
through the OpenSpec flow below. Update `llms.md` when the change makes its mental
model inaccurate or incomplete: a new or removed subsystem, a changed
architectural relationship, a public capability that needs orienting explanation,
or a documented boundary that no longer holds. Do not restate a spec-owned
contract in `llms.md` — two maintained statements of one contract is the
duplication this split removed. Where behavior has no owning spec yet — operation
semantics beyond dtype planning, the module system, `strideweave.nn` — `llms.md`
is still the statement of record and must be updated in the same change unless
the user explicitly excludes documentation work.

Identify the relevant invariant IDs while planning a change. When adding, removing, or
materially changing a cross-cutting invariant, update `INVARIANTS.md` and its stated
enforcement evidence in the same change.

## OpenSpec and Beads Workflow

OpenSpec sits above the Beads implementation lifecycle. OpenSpec artifacts define the
approved intent, behavioral requirements, design constraints, and acceptance boundary
that the planner and reviewer use; they do not authorize direct implementation.

- Do not use `openspec-apply-change` in this repository, even when that skill is
  installed. The apply phase is owned by the Beads workflow below.
- Treat an OpenSpec `tasks.md` as planning input and acceptance-oriented decomposition,
  not as the executable work queue.
- After the required OpenSpec artifacts are coherent, the planner/reviewer agent uses
  `create-task` to translate them into dependency-aware Beads implementation tasks, a
  review bead, and the appropriate worktree.
- An implementer agent uses `do-task` to claim and complete ready Beads work. It does
  not implement directly from the OpenSpec change.
- The planner/reviewer agent reviews the implementation against the OpenSpec proposal,
  specs, design, repository invariants, and Beads acceptance criteria. Review findings
  become Beads fix work through `create-task`, and review remains dependent on every
  outstanding fix.
- Repeat implementation and review until the review is approved. Archive the OpenSpec
  change only after the review bead and all implementation or fix beads are closed.
  Use `openspec-sync-specs` before that point only when explicitly requested.
- `openspec-onboard` is a tutorial. For repository work, stop its walkthrough before
  its direct implementation phase and enter the planner-to-Beads flow above.

### Spec Publication

`openspec/specs/*/spec.md` feeds the public specifications site at
`https://strideweave.org/spec/`, built by `.github/workflows/specs.yml` from
`scripts/gen_spec_pages.py`. Publication is opt-in per spec through YAML front matter:

```yaml
---
title: Friendly Tensor Creation
publish: true
status: stable
order: 10
summary: One sentence describing what the spec governs.
---
```

A spec with no front matter, or with `publish` unset or false, is silently omitted from
the site. That default is deliberate — an unfinished contract should not publish by
accident — but nothing warns about the omission, so a spec that should be public stays
invisible until someone adds the block. When a change archives a spec whose behavior is
delivered and intended to be public, add or update its front matter in the same change.

`status` renders a banner on the page unless it is `stable`, so use `stable` only for
delivered behavior. `order` sorts the index; `title` and `summary` populate its table.

Preview what is being held back with `SW_DOCS_INCLUDE_INTERNAL=1 uv run --no-project
properdocs build --strict`, which also renders unarchived changes under
`openspec/changes/`. Never publish those: an unarchived change is a proposal, not the
contract. Front matter is invisible to `openspec validate`, so both consumers accept
the same file.

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
