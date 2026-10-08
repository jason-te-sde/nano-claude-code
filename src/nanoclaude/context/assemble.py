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

from nanoclaude.context.git import run_git
from nanoclaude.context.instructions import load_instructions
from nanoclaude.context.projectmap import project_map


@dataclass(frozen=True, slots=True)
class AssembledContext:
    system: str
    environment: str


def git_state(root: str) -> str:
    """The branch and how many files differ, or "" when there is nothing to say.

    Nothing to say is also what a failed question is: a count of zero that only means
    git could not answer would be a statement of fact the model would believe.
    """
    try:
        branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        status = run_git(root, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return ""
    if not branch or status.returncode != 0:
        return ""
    changed = len(status.stdout.strip().splitlines())
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
