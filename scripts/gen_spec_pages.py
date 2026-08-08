"""Build docs pages from openspec/specs/ without copying files into docs/.

Runs under mkdocs-gen-files at build time. Every spec whose front matter says
`publish: true` becomes a page; everything else is skipped, so internal specs
stay internal. Nav is emitted as a literate-nav SUMMARY.md.

Set SW_DOCS_INCLUDE_INTERNAL=1 to preview unpublished specs locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import mkdocs_gen_files
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_ROOT = REPO_ROOT / "openspec" / "specs"
CHANGES_ROOT = REPO_ROOT / "openspec" / "changes"
INCLUDE_INTERNAL = os.environ.get("SW_DOCS_INCLUDE_INTERNAL") == "1"


def split_front_matter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). Tolerates files with no front matter."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text
    return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")


def humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def collect_specs() -> list[dict]:
    specs: list[dict] = []
    if not SPEC_ROOT.is_dir():
        return specs
    for spec_file in sorted(SPEC_ROOT.glob("*/spec.md")):
        domain = spec_file.parent.name
        meta, body = split_front_matter(spec_file.read_text(encoding="utf-8"))
        if not meta.get("publish", False) and not INCLUDE_INTERNAL:
            continue
        specs.append(
            {
                "domain": domain,
                "title": meta.get("title") or humanize(domain),
                "status": meta.get("status", "draft"),
                "summary": meta.get("summary", ""),
                "order": meta.get("order", 1000),
                "published": bool(meta.get("publish", False)),
                "body": body,
                "src": spec_file.relative_to(REPO_ROOT).as_posix(),
            }
        )
    specs.sort(key=lambda s: (s["order"], s["title"]))
    return specs


def active_changes() -> list[dict]:
    """In-flight OpenSpec changes, with their delta specs.

    Only rendered when SW_DOCS_INCLUDE_INTERNAL=1, since an unarchived change
    is a proposal, not the contract. Deltas are shown as-is rather than
    speculatively merged into the base spec: a wrong merge preview is worse
    than no merge preview. /opsx:archive does the real merge.
    """
    if not CHANGES_ROOT.is_dir() or not INCLUDE_INTERNAL:
        return []
    changes = []
    for change_dir in sorted(CHANGES_ROOT.iterdir()):
        if not change_dir.is_dir() or change_dir.name == "archive":
            continue
        deltas = []
        for delta in sorted(change_dir.glob("specs/**/*.md")):
            deltas.append(
                {
                    "domain": delta.parent.name,
                    "body": split_front_matter(delta.read_text(encoding="utf-8"))[1],
                }
            )
        proposal = change_dir / "proposal.md"
        changes.append(
            {
                "name": change_dir.name,
                "deltas": deltas,
                "proposal": (
                    split_front_matter(proposal.read_text(encoding="utf-8"))[1]
                    if proposal.is_file()
                    else ""
                ),
            }
        )
    return changes


def write_change_pages(changes: list[dict]) -> None:
    for change in changes:
        out_path = f"changes/{change['name']}.md"
        with mkdocs_gen_files.open(out_path, "w") as fh:
            fh.write(f"# Proposed: {change['name']}\n\n")
            fh.write(
                '!!! danger "Unmerged proposal"\n\n'
                "    Not archived. These deltas are not part of the "
                "specification yet and this page is never published.\n\n"
            )
            if change["proposal"]:
                fh.write("## Proposal\n\n" + change["proposal"] + "\n\n")
            for delta in change["deltas"]:
                fh.write(f"## Delta: `{delta['domain']}`\n\n")
                fh.write(delta["body"] + "\n\n")


def write_spec_page(spec: dict) -> str:
    out_path = f"{spec['domain']}.md"
    with mkdocs_gen_files.open(out_path, "w") as fh:
        badge = "" if spec["published"] else '!!! warning "Internal preview"\n\n'
        if badge:
            fh.write(badge + "    Not published. Local preview only.\n\n")
        if spec["status"] != "stable":
            fh.write(f'!!! note "Status: {spec["status"]}"\n\n')
        fh.write(spec["body"])
    # Makes the "edit this page" pencil point at the real source file.
    mkdocs_gen_files.set_edit_path(out_path, f"../{spec['src']}")
    return out_path


def write_index(specs: list[dict], changes: list[dict]) -> None:
    lines = ["# Strideweave Specifications", ""]
    lines.append(
        "Behavior contracts for Strideweave subsystems. Generated from "
        "`openspec/specs/` at build time.\n"
    )
    lines.append("| Specification | Status | Summary |")
    lines.append("| --- | --- | --- |")
    for s_ in specs:
        lines.append(
            f"| [{s_['title']}]({s_['domain']}.md) | `{s_['status']}` | {s_['summary']} |"
        )
    if changes:
        lines.append("\n## Changes in flight (local preview only)\n")
        for c in changes:
            lines.append(f"- [`{c['name']}`](changes/{c['name']}.md)")
    with mkdocs_gen_files.open("index.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_nav(specs: list[dict], changes: list[dict]) -> None:
    lines = ["- [Overview](index.md)", "- Specifications"]
    for s in specs:
        lines.append(f"    - [{s['title']}]({s['domain']}.md)")
    if changes:
        lines.append("- Proposed changes")
        for c in changes:
            lines.append(f"    - [{c['name']}](changes/{c['name']}.md)")
    with mkdocs_gen_files.open("SUMMARY.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    specs = collect_specs()
    changes = active_changes()
    for spec in specs:
        write_spec_page(spec)
    write_change_pages(changes)
    write_index(specs, changes)
    write_nav(specs, changes)
    print(
        f"[gen_spec_pages] emitted {len(specs)} spec page(s), "
        f"{len(changes)} in-flight change page(s)"
    )


main()
