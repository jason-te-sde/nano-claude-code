"""What the context tests share: the policy and the redactor a session would hand over."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from nanoclaude.context.assemble import AssembledContext, assemble
from nanoclaude.context.instructions import load_instructions
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox


def policy_for(root: Path, *, deny: Sequence[str] = ()) -> Policy:
    """The policy of a default session whose sandbox is ``root``, with ``deny`` rules."""
    return Policy(
        sandbox=Sandbox((str(root),)),
        rules=RuleSet.build(allow=["Read", "Grep", "Glob"], ask=["Write", "Edit"], deny=list(deny)),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
    )


def load(
    root: Path,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    deny: Sequence[str] = (),
    redactor: Redactor | None = None,
) -> str:
    """The instructions a session in ``root`` would load, from ``cwd`` (default ``root``)."""
    return load_instructions(
        str(root),
        cwd=str(cwd or root),
        home=None if home is None else str(home),
        policy=policy_for(root, deny=deny),
        redactor=redactor or Redactor(),
    )


def assemble_in(
    root: Path, *, deny: Sequence[str] = (), home: Path | None = None
) -> AssembledContext:
    """The context a session in ``root`` would assemble."""
    return assemble(
        str(root),
        cwd=str(root),
        home=None if home is None else str(home),
        policy=policy_for(root, deny=deny),
        redactor=Redactor(),
    )
