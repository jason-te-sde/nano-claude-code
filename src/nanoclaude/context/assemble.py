"""Putting one request's context together.

Order is fixed and the first three sections are byte-stable across a session,
because that is what makes prompt caching work -- on Anthropic through explicit
breakpoints, on OpenAI-compatible endpoints through an unchanged prefix. The
environment block is deliberately *after* them: it contains git state, which
changes, so it must not sit inside the cached prefix.

Nothing here retrieves code. See spec 7.2.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from nanoclaude.context.instructions import load_instructions
from nanoclaude.context.projectmap import project_map


@dataclass(frozen=True, slots=True)
class AssembledContext:
    system: str
    environment: str


def git_state(root: str) -> str:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not branch:
        return ""
    changed = len(status.splitlines())
    return f"git branch: {branch} ({changed} file(s) with uncommitted changes)"


def environment_block(root: str, *, depth: int = 3) -> str:
    parts = [f"Working directory: {root}"]
    state = git_state(root)
    if state:
        parts.append(state)
    parts.append("Project structure:\n" + project_map(root, depth=depth))
    return "\n\n".join(parts)


def assemble(
    root: str, *, cwd: str, home: str | None, tool_protocol: str | None = None, depth: int = 3
) -> AssembledContext:
    from nanoclaude.prompts import build_system_prompt

    instructions = load_instructions(root, cwd=cwd, home=home)
    return AssembledContext(
        system=build_system_prompt(root, instructions=instructions, tool_protocol=tool_protocol),
        environment=environment_block(root, depth=depth),
    )
