# src/nanoclaude/agent/ui.py
"""The only way anything under agent/ reaches a human.

Four implementations exist: the REPL, headless mode, the test doubles here, and
whatever someone importing the library writes. Keeping the protocol small is
what makes that true -- every method added here is a method three other
implementations have to grow.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.policy import PermissionRequest, PermissionResult
from nanoclaude.providers.base import ModelReply
from nanoclaude.tools.base import ToolOutcome


class Approval(StrEnum):
    ONCE = "once"
    ALWAYS = "always"
    NO = "no"


@runtime_checkable
class UI(Protocol):
    async def confirm(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> Approval: ...

    def on_reply(self, reply: ModelReply) -> None: ...

    def on_decision(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> None: ...

    def on_outcome(self, call: ToolUseBlock, outcome: ToolOutcome) -> None: ...

    def on_output(self, text: str) -> None: ...

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None: ...


class SilentUI:
    """Does nothing and declines everything. Base for the others."""

    # Parameters carry the leading underscore convention rather than the bare
    # names UI.confirm declares: the answer here never depends on them, and
    # that is true for every override below too. Still positional-only in
    # practice -- run_batch always calls confirm(call, request, result)
    # positionally -- so the rename changes nothing a caller can observe.
    async def confirm(
        self, _call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
    ) -> Approval:
        return Approval.NO

    def on_reply(self, reply: ModelReply) -> None: ...

    def on_decision(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> None: ...

    def on_outcome(self, call: ToolUseBlock, outcome: ToolOutcome) -> None: ...

    def on_output(self, text: str) -> None: ...

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None: ...


class AutoApprove(SilentUI):
    """Says yes to anything the policy merely wants confirmed. Tests and --yes."""

    async def confirm(
        self, _call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
    ) -> Approval:
        return Approval.ONCE


class AutoDecline(SilentUI):
    """The correct default when there is no terminal to ask at."""
