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
    """The branch, or "" when there is nothing to say.

    Only the branch: how many files differ is a question about the working tree, which git
    answers by reading it, and reading a repository somebody else prepared is how it runs a
    program that repository names (see ``context/git.py``). Nothing to say is also what a
    failed question is: a branch name printed by a git that then failed (an unborn branch
    prints ``HEAD`` and exits non-zero) would be a statement the model would believe.
    """
    try:
        done = run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return ""
    branch = done.stdout.strip()
    if done.returncode != 0 or not branch:
        return ""
    return f"git branch: {branch}"


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
