"""Fixtures for the tool tests that need something outside the sandbox to point at."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.tools.base import ToolContext


@dataclass(frozen=True)
class Layout:
    """A project that is the whole sandbox, and a directory beside it that is not."""

    ctx: ToolContext
    project: Path
    outside: Path


@pytest.fixture
def layout(tmp_path: Path, ctx: ToolContext) -> Layout:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    sandbox = Sandbox((str(project),))
    rooted = replace(
        ctx, sandbox=sandbox, policy=replace(ctx.policy, sandbox=sandbox), root=sandbox.roots[0]
    )
    return Layout(rooted, project, outside)
