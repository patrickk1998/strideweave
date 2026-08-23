---
name: openspec-onboard
description: Guided onboarding for OpenSpec - walk through a complete workflow cycle with narration and real codebase work.
allowed-tools: Bash(openspec:*)
license: MIT
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.8.0"
---

Guide the user through their first complete OpenSpec workflow cycle. This is a teaching experience—you'll do real work in their codebase while explaining each step.

**StrideWeave project override:** Teach the project-local `spec-driven` workflow:
explore → concise non-normative proposal → normative delta specs → Beads planning and
implementation → review → archive. Do not create `design.md` or `tasks.md`, do not use
OpenSpec apply, and stop the OpenSpec walkthrough before implementation so a later
`create-task`/`do-task` workflow can take over.

**Store selection:** If the user names a store (a store is a standalone OpenSpec repo registered on this machine) or the work lives in one, run `openspec store list --json` to discover registered store ids, then pass `--store <id>` on the commands that read or write specs and changes (`new change`, `status`, `instructions`, `list`, `show`, `validate`, `archive`, `doctor`, `context`, `view`). Once selected, treat `--store <id>` as sticky for the rest of the workflow. Every unscoped example of those commands below is shorthand: before running it, append the flag. For example, run `openspec status --change "<name>" --json --store "<id>"`, not the unscoped form shown below. Other commands do not take the flag. Hints printed by commands already carry the flag; keep it on follow-ups. Without a store, commands act on the nearest local `openspec/` root.

---

## Preflight

Before starting, check if the OpenSpec CLI is installed:

```bash
# Unix/macOS
openspec --version 2>&1 || echo "CLI_NOT_INSTALLED"
# Windows (PowerShell)
# if (Get-Command openspec -ErrorAction SilentlyContinue) { openspec --version } else { echo "CLI_NOT_INSTALLED" }
```

**If CLI not installed:**
> OpenSpec CLI is not installed. Install it first, then come back to `/openspec-onboard`.

Stop here if not installed.

---

## Phase 1: Welcome

Display:

```
## Welcome to OpenSpec!

I'll walk you through a complete change cycle—from idea to implementation—using a real task in your codebase. Along the way, you'll learn the workflow by doing it.

**What we'll do:**
1. Pick a small, real behavioral change
2. Explore the problem and owning specs
3. Create a change container
4. Write a concise non-normative proposal
5. Write and validate the normative delta specs
6. Hand the contract to the later Beads implementation workflow

**Time:** ~15-20 minutes

Let's start by finding something to work on.
```

---

## Phase 2: Task Selection

### Codebase Analysis

Scan the codebase for small improvement opportunities. Look for:

1. **TODO/FIXME comments** - Search for `TODO`, `FIXME`, `HACK`, `XXX` in code files
2. **Missing error handling** - `catch` blocks that swallow errors, risky operations without try-catch
3. **Functions without tests** - Cross-reference `src/` with test directories
4. **Type issues** - `any` types in TypeScript files (`: any`, `as any`)
5. **Debug artifacts** - `console.log`, `console.debug`, `debugger` statements in non-debug code
6. **Missing validation** - User input handlers without validation

Also check recent git activity:
```bash
# Unix/macOS
git log --oneline -10 2>/dev/null || echo "No git history"
# Windows (PowerShell)
# git log --oneline -10 2>$null; if ($LASTEXITCODE -ne 0) { echo "No git history" }
```

### Present Suggestions

From your analysis, present 3-4 specific suggestions:

```
## Task Suggestions

Based on scanning your codebase, here are some good starter tasks:

**1. [Most promising task]**
   Location: `src/path/to/file.ts:42`
   Scope: ~1-2 files, ~20-30 lines
   Why it's good: [brief reason]

**2. [Second task]**
   Location: `src/another/file.ts`
   Scope: ~1 file, ~15 lines
   Why it's good: [brief reason]

**3. [Third task]**
   Location: [location]
   Scope: [estimate]
   Why it's good: [brief reason]

**4. Something else?**
   Tell me what you'd like to work on.

Which task interests you? (Pick a number or describe your own)
```

**If nothing found:** Fall back to asking what the user wants to build:
> I didn't find obvious quick wins in your codebase. What's something small you've been meaning to add or fix?

### Scope Guardrail

If the user picks or describes something too large (major feature, multi-day work):

```
That's a valuable task, but it's probably larger than ideal for your first OpenSpec run-through.

For learning the workflow, smaller is better—it lets you see the full cycle without getting stuck in implementation details.

**Options:**
1. **Slice it smaller** - What's the smallest useful piece of [their task]? Maybe just [specific slice]?
2. **Pick something else** - One of the other suggestions, or a different small task?
3. **Do it anyway** - If you really want to tackle this, we can. Just know it'll take longer.

What would you prefer?
```

Let the user override if they insist—this is a soft guardrail.

---

## Phase 3: Explore Demo

Once a task is selected, briefly demonstrate explore mode:

```
Before we create a change, let me quickly show you **explore mode**—it's how you think through problems before committing to a direction.
```

Spend 1-2 minutes investigating the relevant code:
- Read the file(s) involved
- Draw a quick ASCII diagram if it helps
- Note any considerations

```
## Quick Exploration

[Your brief analysis—what you found, any considerations]

┌─────────────────────────────────────────┐
│   [Optional: ASCII diagram if helpful]  │
└─────────────────────────────────────────┘

Explore mode (`/openspec-explore`) is for this kind of thinking—investigating before implementing. You can use it anytime you need to think through a problem.

Now let's create a change to hold our work.
```

**PAUSE** - Wait for user acknowledgment before proceeding.

---

## Phase 4: Create the Change

**EXPLAIN:**
```
## Creating a Change

A StrideWeave OpenSpec change is a container for proposed behavioral contract
changes. It lives at the `changeRoot` reported by
`openspec status --change "<name>" --json` and holds a non-normative proposal
plus normative delta specs.

Let me create one for our task.
```

**DO:** Create the change with a derived kebab-case name:
```bash
openspec new change "<derived-name>"
```

**SHOW:**
```
Created: <changeRoot from status JSON>

The artifact structure:
```
<changeRoot>/
├── .openspec.yaml
├── proposal.md    ← Concise, non-normative change index
└── specs/         ← Normative behavioral deltas
```

Now let's fill in the first artifact—the proposal.
```

---

## Phase 5: Proposal

**EXPLAIN:**
```
## The Proposal

The proposal captures **why** we're making this change and **what** it involves at a high level. It's the "elevator pitch" for the work.

I'll draft one based on our task.
```

**DO:** Draft the proposal content (don't save yet):

`<capability-path>` is the spec directory relative to `specs/` (for example,
`user-auth` or `identity/user-auth`). Use the exact existing path for modified
capabilities. For new capabilities, follow the project's established spec
organization.

```
Here's a draft proposal:

---

## Why

[1-2 sentences explaining the problem/opportunity]

## What Changes

[Bullet points of what will be different]

## Capabilities

### New Capabilities
- `<capability-path>`: [brief description]

### Modified Capabilities
<!-- If modifying existing behavior -->
- `<existing-capability-path>`: [brief description]

## Impact

- `src/path/to/file.ts`: [what changes]
- [other files if applicable]

---

Does this capture the intent? I can adjust before we save it.
```

**PAUSE** - Wait for user approval/feedback.

After approval, save the proposal:
```bash
openspec instructions proposal --change "<name>" --json
```
Then write the content to the `resolvedOutputPath` from `openspec instructions proposal --change "<name>" --json`.

```
Proposal saved. This is your "why" document—you can always come back and refine it as understanding evolves.

Next up: specs.
```

---

## Phase 6: Specs

**EXPLAIN:**
```
## Specs

Specs define **what** we're building in precise, testable terms. They use a requirement/scenario format that makes expected behavior crystal clear.

For a small task like this, we might only need one spec file.
```

**DO:** Resolve where the spec file should be created:
```bash
openspec instructions specs --change "<name>" --json
# Use resolvedOutputPath from the JSON. If it is a glob, choose the concrete file path using the schema instruction and the change's context.
```

Draft the spec content:

```
Here's the spec:

---

## ADDED Requirements

### Requirement: <Name>

<Description of what the system should do>

#### Scenario: <Scenario name>

- **WHEN** <trigger condition>
- **THEN** <expected outcome>
- **AND** <additional outcome if needed>

---

This format—WHEN/THEN/AND—makes requirements testable. You can literally read them as test cases.
```

Save to the concrete file path chosen from `resolvedOutputPath`.

---

## Phase 7: Validate and Hand Off

Run:

```bash
openspec status --change "<name>" --json
openspec validate "<name>" --type change --strict --no-interactive
git diff --check
```

Confirm that the resolved artifact list contains only `proposal` and `specs`,
and that both are complete. Explain the authority split:

- the proposal records intent and affected capability paths but is non-normative;
- the effective specs decide whether observable behavior is a feature or bug;
- Beads will later own implementation design, threat models, decomposition,
  dependencies, validation, and review.

Summarize the change and stop. Tell the user:

```
The normative artifacts are ready for review. When you want to plan
implementation, ask me to run create-task.
```

Do not create Beads work, implement, archive, commit, or push during this
onboarding invocation. Those are separate, explicitly requested workflows.

---

## Phase 8: Recap & Next Steps

```
## Congratulations!

You completed StrideWeave's normative OpenSpec phase:

1. **Explore** - Thought through the problem
2. **New** - Created a change container
3. **Proposal** - Indexed WHY and affected capabilities
4. **Specs** - Defined the observable contract
5. **Validate** - Proved the normative delta is structurally sound
6. **Handoff** - Stopped before Beads planning and implementation

The later repository lifecycle is create-task → do-task → review → archive.

---

## Command Reference

**Core workflow:**

 | Command                    | What it does                                      |
 |----------------------------|---------------------------------------------------|
 | `/openspec-propose`        | Create proposal and normative delta specs         |
 | `/openspec-explore`        | Think through behavioral requirements             |
 | `create-task`              | Prepare Beads implementation and review work      |
 | `do-task`                  | Implement assigned ready Beads work               |
 | `/openspec-archive-change` | Archive after implementation review is complete   |

**Additional commands** (only if installed - availability depends on your profile):

 | Command            | What it does                                             |
 |--------------------|----------------------------------------------------------|
 | `/openspec-new-change`      | Start a new change, step through artifacts one at a time |
 | `/openspec-continue-change` | Continue working on an existing change                   |
 | `/openspec-ff-change`       | Fast-forward: create all artifacts at once               |
 | `/openspec-verify-change`   | Verify implementation matches the effective specs         |

---

## What's Next?

Try `/openspec-propose` on something you actually want to build. You've got the rhythm now!
```

---

## Graceful Exit Handling

### User wants to stop mid-way

If the user says they need to stop, want to pause, or seem disengaged:

```
No problem! Your change is saved at the `changeRoot` reported by `openspec status --change "<name>" --json`.

To pick up where we left off later:
- `/openspec-continue-change <name>` - Resume normative artifact creation (if installed; otherwise `openspec status --change "<name>" --json` shows the next artifact)
- `create-task` - Prepare implementation only after the normative artifacts are complete and reviewed

The work won't be lost. Come back whenever you're ready.
```

Exit gracefully without pressure.

### User just wants command reference

If the user says they just want to see the commands or skip the tutorial:

```
## OpenSpec Quick Reference

**Core workflow:**

 | Command                           | What it does                                      |
 |-----------------------------------|---------------------------------------------------|
 | `/openspec-propose <name>`        | Create proposal and normative delta specs         |
 | `/openspec-explore`               | Explore behavior without implementation           |
 | `create-task`                     | Prepare Beads implementation and review work      |
 | `do-task`                         | Implement assigned ready Beads work               |
 | `/openspec-archive-change <name>` | Archive after review closes                       |

**Additional commands** (only if installed - availability depends on your profile):

 | Command                   | What it does                        |
 |---------------------------|-------------------------------------|
 | `/openspec-new-change <name>`      | Start a new change, step by step    |
 | `/openspec-continue-change <name>` | Continue an existing change         |
 | `/openspec-ff-change <name>`       | Fast-forward: all artifacts at once |
 | `/openspec-verify-change <name>`   | Verify implementation               |

Try `/openspec-propose` to start your first change.
```

Exit gracefully.

---

## Guardrails

- **Follow the EXPLAIN → DO → SHOW → PAUSE pattern** at key transitions (after explore and the proposal draft)
- **Stop before implementation**—this tutorial teaches the normative phase and handoff
- **Don't skip phases** even if the change is small—the goal is teaching the workflow
- **Pause for acknowledgment** at marked points, but don't over-pause
- **Handle exits gracefully**—never pressure the user to continue
- **Use a real bounded behavioral change**—don't simulate or use fake examples
- **Adjust scope gently**—guide toward smaller tasks but respect user choice
