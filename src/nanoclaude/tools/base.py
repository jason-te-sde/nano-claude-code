# src/nanoclaude/tools/base.py
"""What a tool is, and the two questions it answers separately.

:meth:`Tool.permission_request` says what a call *would* touch -- resolved
paths, the shell command -- and changes nothing. :meth:`Tool.run` does the work
and is only reached after the policy has seen that answer. A tool that resolves
its own paths inside ``run`` has quietly moved the sandbox check to after the
fact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from nanoclaude.permissions.policy import PermissionRequest, Policy
from nanoclaude.permissions.redact import Redactor
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.fs import FileStamp


class ToolArgumentError(ValueError):
    """The model called a tool with arguments that do not make sense."""


#: CSI (colour, cursor movement, clear screen) and OSC (window title) sequences.
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
#: Every C0 control except tab and newline, plus DEL and the C1 controls
#: (U+0080 to U+009F). Carriage returns go too: a progress bar redrawing itself is
#: noise in a transcript. The C1 block is the 8-bit form of the escape sequences
#: above (U+009B is a CSI, U+009D an OSC, U+0090 a DCS, U+009C a string terminator),
#: which a terminal that honours C1 in UTF-8 acts on as it would on the two-byte
#: form. The bidirectional controls (U+202A to U+202E, U+2066 to U+2069) are not here
#: on purpose: they do not act on a terminal, and file contents can hold them, so
#: stripping them from what Read returns would leave Edit unable to match the lines
#: that have them.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Remove terminal control sequences from anything a tool returns."""
    return _CONTROL.sub("", _ANSI.sub("", text))


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    tool_use_id: str
    content: str
    is_error: bool = False
    #: Files this call looked at, as (resolved path, stamp). The only way the
    #: loop's read state is updated, so an edit cannot claim a file was read.
    observed: tuple[tuple[str, FileStamp], ...] = ()


@dataclass(frozen=True, slots=True)
class ToolContext:
    sandbox: Sandbox
    policy: Policy
    redactor: Redactor
    read_state: Mapping[str, FileStamp]
    root: str

    # Declared unhashable rather than left to the default: read_state is a
    # mapping, and callers pass a plain dict. frozen=True's generated __hash__
    # would report isinstance(x, Hashable) as True and then raise naming
    # "dict" rather than this class -- the decision ToolSpec makes in
    # providers/base.py, for the same reason.
    __hash__ = None  # type: ignore[assignment]

    def resolve(self, raw: str) -> str:
        if not raw:
            raise ToolArgumentError("path must not be empty")
        return self.sandbox.resolve(raw, base=self.root)

    def display(self, resolved: str) -> str:
        path, root = PurePosixPath(resolved), PurePosixPath(self.root)
        return str(path.relative_to(root)) if root in path.parents else resolved


@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def read_only(self) -> bool:
        """True if the tool cannot change anything.

        Read-only calls in one batch run concurrently; everything else runs one
        at a time. A property of the tool, not of the policy -- a user granting
        Write for the session has not made writes commutative.
        """
        ...

    def spec(self) -> ToolSpec: ...

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest: ...

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome: ...


def require_str(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolArgumentError(f"{key} must be a string, got {type(value).__name__}")
    return value


def optional_int(arguments: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolArgumentError(f"{key} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ToolArgumentError(f"{key} must be at least {minimum}, got {value}")
    return value


def ok(call_id: str, content: str, observed: tuple[tuple[str, FileStamp], ...] = ()) -> ToolOutcome:
    return ToolOutcome(call_id, sanitize(content), is_error=False, observed=observed)


def failed(call_id: str, content: str) -> ToolOutcome:
    return ToolOutcome(call_id, sanitize(content), is_error=True)
