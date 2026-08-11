---
name: capture-spec
description: Capture already-implemented behavior directly as an OpenSpec main spec under openspec/specs/, bypassing the change lifecycle. Use for brownfield spec migration when the code exists, nothing needs implementing, and a proposal/design/tasks cycle would be empty ceremony.
---

# capture-spec

Write the contract for behavior that already ships, straight to
`openspec/specs/<capability-path>/spec.md`.

## When to use this

Use it when the behavior is already implemented and the job is to write down what
it guarantees. There is no work to schedule, so the change lifecycle
(proposal → design → tasks → apply → archive) has nothing to carry.

Do **not** use it when behavior is new or changing. That goes through
`openspec-propose` and then the planner-to-Beads flow in CLAUDE.md. If capture
reveals that the implementation is wrong rather than unspecified, stop and say so
— fixing it is a change, not a capture.

## Steps

### 1. Fix the scope

One spec per public API function. Semantically equivalent surfaces of the same
function share a single spec that lists every name and syntax form (for example
`matmul` and the `@` operator). Private helpers get no spec.

The capability path is a kebab-case directory under `openspec/specs/`. Run
`openspec spec list` first: if a spec already covers the function, edit that file
rather than adding a second one.

### 2. Load the authoring rules

Read `openspec/config.yaml` and apply its `context` block and every entry in
`rules.specs`. This step is mandatory and easy to skip — `openspec instructions`
only delivers those rules inside a change, so nothing hands them to you here.

### 3. Gather evidence before writing a line

For the function under capture, collect:

- its section in `llms.md`
- the tests that pin its behavior
- the source signature and docstring — every parameter, default, raised error,
  and return value

You are writing down confirmed behavior plus maintainer intent. Where the two
diverge, or where intent is unresolved, **ask the user**. Do not promote observed
behavior to a contract to fill a gap, and do not invent a requirement to satisfy
validation.

### 4. Write the spec

Main-spec format — never delta headers (`## ADDED Requirements` and friends are
change-only and silently truncate the parsed section).

```markdown
---
title: Matmul
publish: true
status: stable
order: 20
summary: One sentence describing what the spec governs.
---

# <capability-path> Specification

## Purpose
Two sentences on what the capability is for. Under 50 characters fails
`validate --strict`.

## Requirements

### Requirement: <stable name>

<Normative text using SHALL/MUST.>

#### Scenario: <name>

- **WHEN** <condition>
- **THEN** <observable outcome>
- **AND** <further outcome>
```

Mechanical constraints the validator enforces:

- scenarios take exactly four hashtags — three fails silently
- every `### Requirement:` lives inside `## Requirements`
- requirement names are unique within the file
- every requirement has at least one scenario and a SHALL/MUST in its body

Front matter drives the public site. Use `publish: true` with `status: stable`
only for delivered behavior; `order` is the next unused multiple of 10 across
`openspec/specs/`.

### 5. Validate

```bash
openspec validate --specs --strict
```

This is a format gate, not a correctness gate — it never reads your source, and
it checks none of the `rules.specs`. A pass means the file parses and nothing
vanished; correctness rests on the evidence from step 3.

### 6. Report

State the capability path written, the evidence each requirement rests on, any
behavior deliberately left uncaptured, and every question about intent that is
still open.

## Guardrails

- Never edit source or tests. Capture describes; it does not change.
- Never touch `openspec/changes/`. This skill creates no change, proposal,
  design, or tasks file.
- Leave a spec unwritten rather than guess at a contract.
