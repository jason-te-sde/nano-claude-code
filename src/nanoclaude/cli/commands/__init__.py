# src/nanoclaude/cli/commands/__init__.py
"""Slash commands. Spec 17.7 lists them; this module is the implementation.

The eleven here are what v0.1 has. ``/agents`` and ``/mcp`` describe subsystems that
arrive in v0.3, and ``/rewind``, ``/diff`` and ``/checkpoints`` in v0.2, so they are
absent rather than stubbed. ``/exit`` is not in spec 17.7; it is here because the
REPL has to be leavable by a word as well as by Ctrl+D.

A handler prints its own result. It does not catch what goes wrong in the model or the
store: the REPL does, for every command and every turn alike, and says what failed
without ending the session.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

from rich.console import Console
from rich.text import Text

from nanoclaude.agent.session import Session
from nanoclaude.cli.render import cost_panel, plain, show_error, show_notice, status_panel
from nanoclaude.config.load import ConfigError
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from nanoclaude.permissions.policy import PermissionMode
from nanoclaude.providers.base import ModelError
from nanoclaude.tools.base import sanitize

Handler = Callable[[list[str], Session, Console], Awaitable[None]]

#: The modes ``/mode`` can switch to. Not ``bypass``: spec 6.2 makes that a startup choice,
#: made with a flag whose long spelling is the point, and a slash command would make it
#: one line away from a session that was started with the protections on.
SETTABLE_MODES = (PermissionMode.DEFAULT, PermissionMode.PLAN, PermissionMode.ACCEPT_EDITS)

#: How much of one tool result an export keeps.
MAX_EXPORTED_RESULT_CHARS = 2000


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    summary: str
    handler: Handler
    #: True for the command that ends the REPL, and for no other.
    ends_repl: bool = False


def _fence(content: str) -> str:
    """A code fence longer than any run of backticks inside ``content``."""
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _fenced(content: str, language: str = "") -> str:
    fence = _fence(content)
    return f"{fence}{language}\n{content}\n{fence}"


def _clipped(content: str) -> str:
    if len(content) <= MAX_EXPORTED_RESULT_CHARS:
        return content
    hidden = len(content) - MAX_EXPORTED_RESULT_CHARS
    return f"{content[:MAX_EXPORTED_RESULT_CHARS]}\n[{hidden} more characters not exported]"


def export_markdown(session_id: str, messages: Sequence[Message]) -> str:
    """The conversation as a markdown document.

    Text goes through :func:`~nanoclaude.tools.base.sanitize`: the file will be read by
    something that prints it, an ``cat`` in a terminal included, and the model's words
    must not be able to act on that terminal any more than they can on this one. The
    model's private reasoning is not part of the conversation and is left out. A tool
    result is cut to ``MAX_EXPORTED_RESULT_CHARS``, and a fence is made longer than any
    run of backticks in what it holds, so a file that contains a fence cannot end the
    block around it early.
    """
    sections = [f"# Session {sanitize(session_id)}"]
    for message in messages:
        parts = [f"## {message.role}"]
        for block in message.blocks:
            if isinstance(block, TextBlock):
                parts.append(sanitize(block.text))
            elif isinstance(block, ThinkingBlock):
                continue
            elif isinstance(block, ToolUseBlock):
                arguments = json.dumps(dict(block.arguments), ensure_ascii=False, default=str)
                parts.append(f"**tool call** `{sanitize(block.name)}`")
                parts.append(_fenced(arguments, "json"))
            elif isinstance(block, ToolResultBlock):
                parts.append("**tool result** (error)" if block.is_error else "**tool result**")
                parts.append(_fenced(_clipped(sanitize(block.content))))
            else:
                assert_never(block)
        sections.append("\n\n".join(parts))
    return "\n\n".join(sections) + "\n"


def _export_target(root: str, name: str) -> Path:
    """Where ``/export <name>`` writes.

    A relative name is relative to the project and not to wherever the process happened to
    start, and an absolute one, or one that begins with ``~``, is taken as it is.
    """
    return Path(root) / Path(name).expanduser()


def _write_new(target: Path, document: str) -> None:
    """Write ``document`` to a file that does not exist yet: never over one that does."""
    with target.open("x", encoding="utf-8") as handle:
        handle.write(document)


async def _help(_args: list[str], _session: Session, console: Console) -> None:
    width = max(len(name) for name in COMMANDS) + 1
    for command in sorted(COMMANDS.values(), key=lambda c: c.name):
        label = f"/{command.name}".ljust(width)
        console.print(Text.assemble((label, "bold"), "  ", command.summary))


async def _clear(_args: list[str], session: Session, console: Console) -> None:
    # Through the session, which stores the change at once. Replacing its state here would
    # leave the cleared conversation in the store, and a resume would bring it back.
    session.clear()
    show_notice(console, "conversation cleared — the old one stays in the session archive")


async def _compact(args: list[str], session: Session, console: Console) -> None:
    before, _ = await session.context_usage()
    # The session says whether it summarised anything. The figures cannot: closing what an
    # interrupted turn left open adds a note first, which makes the context larger when
    # nothing at all was compacted.
    if not await session.compact(" ".join(args) or None):
        show_notice(console, "nothing to compact — the recent turns are kept as they are")
        return
    after, _ = await session.context_usage()
    show_notice(console, f"compacted — context was {before} tokens, now {after}")


async def _status(_args: list[str], session: Session, console: Console) -> None:
    used, available = await session.context_usage()
    console.print(status_panel(session, used, available))


async def _cost(_args: list[str], session: Session, console: Console) -> None:
    console.print(cost_panel(session.router, session.router.prices))


async def _model(args: list[str], session: Session, console: Console) -> None:
    config = session.router.config
    if not args:
        for alias, entry in config.models.items():
            marker = "*" if alias == config.roles.main else " "
            console.print(plain(f"{marker} {alias}: {entry.adapter}/{entry.model}"))
        return
    alias = args[0]
    if alias not in config.models:
        defined = ", ".join(sorted(config.models))
        show_error(console, f"no model named {alias!r} — choose one of: {defined}")
        return
    try:
        # Through the router, which decides which client answers each role and which model
        # a call is billed to. Changing only the session's copy of the config would change
        # what /status shows and leave the old model answering.
        await session.router.route("main", alias)
    except (ConfigError, ModelError) as exc:
        show_error(console, str(exc))
        return
    # In step with the router, but only in the roles: the session's own config may have
    # been changed since (a turn limit given on the command line) and keeps that.
    session.config = replace(session.config, roles=session.router.config.roles)
    show_notice(console, f"main role now uses {alias}")


async def _mode(args: list[str], session: Session, console: Console) -> None:
    names = [mode.value for mode in SETTABLE_MODES]
    if not args:
        choices = f"{', '.join(names[:-1])} or {names[-1]}"
        show_notice(console, f"mode: {session.policy.mode} (choose {choices})")
        return
    requested = args[0]
    if requested == PermissionMode.BYPASS:
        show_error(
            console,
            "bypass mode is set at startup — restart with: ncc --dangerously-skip-permissions",
        )
        return
    if requested not in names:
        show_error(
            console,
            f"unknown mode {requested!r} — choose one of {', '.join(names)} "
            "(bypass is set at startup, with --dangerously-skip-permissions)",
        )
        return
    session.policy = replace(session.policy, mode=PermissionMode(requested))
    # The executor holds the policy it judges calls by, not the session's attribute.
    session.executor.policy = session.policy
    show_notice(console, f"permission mode: {requested}")


async def _tools(_args: list[str], session: Session, console: Console) -> None:
    for tool in sorted(session.registry, key=lambda t: t.name):
        console.print(plain(f"{tool.name} (read-only)" if tool.read_only else tool.name))


async def _resume(_args: list[str], _session: Session, console: Console) -> None:
    show_error(console, "/resume is a startup flag — restart with: ncc --resume <id>")


async def _export(args: list[str], session: Session, console: Console) -> None:
    name = " ".join(args) or f"session-{session.session_id}.md"
    target = _export_target(session.root, name)
    document = export_markdown(session.session_id, session.state.transcript.messages)
    try:
        _write_new(target, document)
    except FileExistsError:
        show_error(console, f"{target} already exists — give another name, or remove it first")
    except OSError as exc:
        show_error(console, f"could not write {target} — {exc.strerror or exc}")
    else:
        show_notice(console, f"written to {target}")


async def _init(_args: list[str], _session: Session, console: Console) -> None:
    show_error(console, "/init is a subcommand — run: ncc init")


async def _exit(args: list[str], session: Session, console: Console) -> None:
    """Does nothing itself: ``ends_repl`` is what ends the REPL."""


COMMANDS: dict[str, SlashCommand] = {
    c.name: c
    for c in (
        SlashCommand("help", "list these commands", _help),
        SlashCommand(
            "clear",
            "start a fresh conversation (the old one stays in the session archive)",
            _clear,
        ),
        SlashCommand(
            "compact",
            "summarise the conversation so far; words after it say what to keep",
            _compact,
        ),
        SlashCommand("status", "models, mode, sandbox, turns, context", _status),
        SlashCommand("cost", "tokens and money this run, by role", _cost),
        SlashCommand("model", "show the models, or switch the main role: /model <alias>", _model),
        SlashCommand(
            "mode", "show or switch the permission mode: default, plan, accept-edits", _mode
        ),
        SlashCommand("tools", "list the available tools", _tools),
        SlashCommand("resume", "how to resume an earlier session", _resume),
        SlashCommand(
            "export", "write the conversation to a markdown file: /export [file]", _export
        ),
        SlashCommand("init", "how to create a configuration", _init),
        SlashCommand("exit", "leave (Ctrl+D does the same)", _exit, ends_repl=True),
    )
}


async def dispatch(name: str, args: list[str], session: Session, console: Console) -> bool:
    """Run the command called ``name``; True if the REPL should go on, False if it should end.

    A name nobody defined is an error line and not an exception: it is something a person
    typed, and the answer to it is a pointer to ``/help``.
    """
    command = COMMANDS.get(name)
    if command is None:
        show_error(console, f"unknown command /{name} — try /help")
        return True
    await command.handler(args, session, console)
    return not command.ends_repl
