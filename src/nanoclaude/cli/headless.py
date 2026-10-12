"""What ``ncc -p`` says about the calls it could not ask about.

Headless mode has nobody to ask, so a call the policy wanted confirmed is declined (spec 3.1).
That is the right behaviour in CI, and it must not be silent: a script that wanted a file
written finds out that it was not from the model's apology on stdout, if at all. Each declined
call is one line on stderr, in the form of spec 17.9: what was declined, with the id of the rule
that asked in the parentheses every refusal has (``refused (sandbox.outside-root): ...``), a dash,
what to do. It starts ``ncc:`` and not ``error:``, because the run has not failed and its exit
code is what it would have been.

Only a call that was asked about is reported here. One the policy denies outright is not: nobody
was to be asked, the model is given the refusal with its rule id as the tool's result, and the
audit records it.
"""

from __future__ import annotations

import contextlib
from pathlib import PurePosixPath

from rich.console import Console
from rich.text import Text

from nanoclaude.agent.ui import Approval, AutoDecline
from nanoclaude.cli.render import MAX_SUMMARY_CHARS, visible
from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.policy import (
    EDIT_TOOLS,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
)

#: What every such line begins with, as ``error: `` begins every error.
DECLINED_PREFIX = "ncc: "


def _subject_shown(request: PermissionRequest, root: str) -> str:
    """What a call is about, on one line: a path inside the root relative to it, anything else
    (a command, a path outside the root) as it is, with control characters spelled out and a
    long one cut short."""
    subject = request.subject
    if subject in request.resolved_paths:
        with contextlib.suppress(ValueError):  # outside the root: shown as it is
            subject = str(PurePosixPath(subject).relative_to(root))
    shown = visible(subject)
    if len(shown) > MAX_SUMMARY_CHARS:
        shown = shown[: MAX_SUMMARY_CHARS - 3] + "..."
    return shown


def _advice(tool: str, mode: PermissionMode) -> str:
    """What lets a headless run do what was declined.

    A file edit is let through by the mode made for that, unless it is already on. Anything
    else is allowed by a rule, which has to be in the person's own config: a project's file
    cannot add one, so it is not named as a place to put it.
    """
    if tool in EDIT_TOOLS and mode is PermissionMode.DEFAULT:
        return "pass --mode accept-edits to let -p edit files"
    return f'add "{visible(tool)}" to allow under [permissions] in your config to let -p run it'


def declined_line(
    tool: str, request: PermissionRequest, *, rule: str, root: str, mode: PermissionMode
) -> str:
    """``declined (<rule>) <tool> <subject> — <what to do>``, without the prefix.

    ``rule`` is the id of the rule that produced the ask (``default.ask`` when nothing
    matched, ``rule.ask`` when a rule did), so that a script that reads the line can tell the
    two apart and knows which rule to change.
    """
    return (
        f"declined ({visible(rule)}) {visible(tool)} {_subject_shown(request, root)} "
        f"— {_advice(tool, mode)}"
    )


class ReportingDecline(AutoDecline):
    """The front end of ``-p``: declines whatever needs asking, and says so on ``err``."""

    def __init__(self, err: Console, *, root: str, mode: PermissionMode) -> None:
        self._err = err
        self._root = root
        self._mode = mode

    async def confirm(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> Approval:
        line = declined_line(call.name, request, rule=result.rule, root=self._root, mode=self._mode)
        self._err.print(Text.assemble((DECLINED_PREFIX, "bold"), line))
        return Approval.NO
