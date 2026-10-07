"""The ``ncc`` command: argument parsing, wiring, exit codes.

Headless mode is not a reduced REPL. It is the same Session with a different UI
implementation, one that never asks and therefore refuses anything the policy wanted
confirmed. That is the right behaviour in CI: approving writes because nobody is watching
is exactly backwards.

Exit codes are a contract. A script needs to tell "the agent finished but the task was not
done" from "the provider rejected the key", and both from "you typed the flag wrong".

Streams are a contract too. stdout carries the answer and nothing else, written exactly as
the model wrote it: not through Rich, which would read ``[...]`` in code as markup and wrap
a long line at 80 columns when the output is a file. Everything that went wrong is one line
on stderr in the form of spec 17.9, and a run that fails with nothing to show leaves stdout
empty.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, assert_never

from rich.console import Console
from rich.text import Text

from nanoclaude import __version__
from nanoclaude.agent.loop import Done, StopReason
from nanoclaude.agent.router import Router
from nanoclaude.agent.session import (
    Session,
    SessionChangedError,
    UnknownSessionError,
    new_session_id,
)
from nanoclaude.agent.ui import UI, AutoDecline
from nanoclaude.cli.render import ConsoleUI, show_error, show_notice
from nanoclaude.cli.repl import run_repl
from nanoclaude.config.load import CONFIG_DIRNAME, ConfigError, expand_root, load_config
from nanoclaude.config.schema import ROLES
from nanoclaude.conversation.budget import ContextTooSmallError
from nanoclaude.conversation.store import Store
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.providers.base import CredentialsError, ModelError
from nanoclaude.providers.capabilities import CapabilityCache
from nanoclaude.tools.base import sanitize
from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.todo import TodoState

EXIT_CODES = {
    "completed": 0,
    "stopped": 1,
    "usage": 2,
    "config": 3,
    "provider": 4,
    "interrupted": 130,
}

#: Where a provider error that does not say what to do is sent.
_PROVIDER_ADVICE = "try again, or choose another model with --model"


def nanoclaude_home() -> Path:
    """The directory that holds ``.nanoclaude``: the person's home, unless told otherwise."""
    return Path(os.environ.get("NANOCLAUDE_HOME", str(Path.home())))


class UsageError(Exception):
    """``ncc`` was invoked wrongly. The message is the person's line, as spec 17.9 words it."""


def _turns(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncc",
        description=(
            "A terminal coding agent that works with Anthropic, OpenAI-compatible and local models."
        ),
        # A flag is spelled out. With abbreviations on, --dangerously-skip would turn the
        # permission checks off, and spec 6.2 gives that flag its long name on purpose.
        allow_abbrev=False,
    )
    parser.add_argument("-p", "--print", dest="prompt", metavar="PROMPT", help="run once and exit")
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--root", default=".", help="working directory (default: cwd)")
    parser.add_argument("--model", metavar="ALIAS", help="model to use for the main role")
    parser.add_argument("--role", action="append", default=[], metavar="ROLE=ALIAS")
    parser.add_argument("--mode", choices=[mode.value for mode in PermissionMode])
    parser.add_argument("--add-dir", action="append", default=[], metavar="PATH")
    parser.add_argument("--max-turns", type=_turns, metavar="N")
    again = parser.add_mutually_exclusive_group()
    again.add_argument("-c", "--continue", dest="continue_last", action="store_true")
    again.add_argument("-r", "--resume", metavar="SESSION_ID")
    parser.add_argument("--allow-secrets", action="store_true")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--version", action="version", version=f"nano-claude-code {__version__}")
    return parser


@dataclass(frozen=True, slots=True)
class _Settings:
    """What the flags ask for, checked, before anything is opened or read."""

    root: str
    #: The sandbox: ``root`` first, then every ``--add-dir``.
    roots: tuple[str, ...]
    mode: PermissionMode
    #: ``--role`` pairs, as (role, alias), in the order given.
    roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What a headless run has to say, split by where it goes."""

    #: What ``--output-format json`` prints.
    payload: dict[str, Any]
    #: What text mode writes to stdout: the model's words, and never the program's own.
    text: str
    #: The line for stderr, in the form of spec 17.9; None when the run finished.
    error: str | None
    code: int


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse ends --help, --version and a mistake in the arguments by raising, and this
        # function does not end the process: the console script does, with what it returns.
        return exc.code if isinstance(exc.code, int) else 1
    err = _stderr_console(args)
    home = nanoclaude_home()
    try:
        if args.prompt is not None and not args.prompt.strip():
            raise UsageError(
                "--print needs a prompt — pass the task as its argument, "
                'as in: ncc -p "explain main.py"'
            )
        settings = _settings(args)
        _warn_about_switches(args, err)
        if args.prompt is None:
            return asyncio.run(_run_interactive(args, settings, home, _stdout_console(args)))
        outcome = asyncio.run(_run_headless(args, settings, home))
        _report(outcome, args.output_format, err)
        return outcome.code
    except (UsageError, UnknownSessionError) as exc:
        show_error(err, str(exc))
        return EXIT_CODES["usage"]
    except (ConfigError, ContextTooSmallError, CredentialsError) as exc:
        # A key that is missing or refused is mended in the configuration, as the rest are.
        show_error(err, str(exc))
        return EXIT_CODES["config"]
    except SessionChangedError as exc:
        show_error(err, str(exc))
        return EXIT_CODES["stopped"]
    except ModelError as exc:
        show_error(err, _with_advice(str(exc), _PROVIDER_ADVICE))
        return EXIT_CODES["provider"]
    except KeyboardInterrupt:
        show_notice(err, "interrupted")
        return EXIT_CODES["interrupted"]


def _settings(args: argparse.Namespace) -> _Settings:
    mode = _permission_mode(args)
    roles = _role_overrides(args.role)
    root = _directory("--root", args.root)
    return _Settings(
        root=root,
        roots=(root, *(_directory("--add-dir", raw) for raw in args.add_dir)),
        mode=mode,
        roles=roles,
    )


def _permission_mode(args: argparse.Namespace) -> PermissionMode:
    """The mode the flags ask for. Bypass takes the flag with the long name, and nothing else.

    Spec 6.2 makes turning the permission checks off a decision made at start, with a flag
    whose spelling is the point. ``--mode bypass`` is the same decision spelled the short
    way, so it is refused alone; and the flag beside a mode that asks for protection is two
    answers to one question, so it is refused too and not settled quietly in the flag's favour.
    """
    skipping = args.dangerously_skip_permissions
    if args.mode == PermissionMode.BYPASS and not skipping:
        raise UsageError(
            "--mode bypass needs --dangerously-skip-permissions "
            "— pass that flag as well, or choose another mode"
        )
    if skipping:
        if args.mode not in (None, PermissionMode.BYPASS):
            raise UsageError(
                f"--mode {args.mode} contradicts --dangerously-skip-permissions "
                "— pass only one of them"
            )
        return PermissionMode.BYPASS
    return PermissionMode(args.mode) if args.mode else PermissionMode.DEFAULT


def _role_overrides(pairs: list[str]) -> tuple[tuple[str, str], ...]:
    overrides = []
    for pair in pairs:
        role, _, alias = pair.partition("=")
        if role not in ROLES or not alias:
            raise UsageError(
                f"--role expects ROLE=ALIAS, not {pair!r} — ROLE is one of: {', '.join(ROLES)}"
            )
        overrides.append((role, alias))
    return tuple(overrides)


def _directory(flag: str, raw: str) -> str:
    """A directory as the person wrote it, made absolute (``~`` and ``..`` included).

    Never ``Path(raw).resolve()``: that turns ``~/lib`` into ``<cwd>/~/lib``, which is
    absolute, so the sandbox takes it without a word and the directory it was meant to
    open stays shut.
    """
    try:
        path = expand_root(raw)
    except ConfigError as exc:
        raise UsageError(str(exc)) from exc
    if not Path(path).is_dir():
        raise UsageError(f"{flag} {raw} is not a directory — pass one that exists")
    return path


def _warn_about_switches(args: argparse.Namespace, err: Console) -> None:
    """Say, as spec 6.2 and 6.5 ask, when a protection has been switched off."""
    if args.dangerously_skip_permissions:
        _warn(
            err,
            "permission prompts are off — tool calls run without asking. The sandbox, "
            "credentials files, dangerous commands and your deny rules are still enforced",
        )
    if args.allow_secrets:
        _warn(
            err,
            "--allow-secrets is on — credentials files can be read, and what is read goes "
            "to the model unredacted",
        )


def _warn(err: Console, message: str) -> None:
    err.print(Text.assemble(("warning: ", "bold red"), message))


def _no_color(args: argparse.Namespace) -> bool:
    return args.no_color or bool(os.environ.get("NO_COLOR"))


def _stdout_console(args: argparse.Namespace) -> Console:
    """What the REPL writes to. What it prints from outside is built as Text, which Rich never
    parses as markup, so the console is left as Rich makes it."""
    return Console(no_color=_no_color(args))


def _stderr_console(args: argparse.Namespace) -> Console:
    """Where every complaint goes: stderr, never parsed as markup, never wrapped.

    Rich wraps at 80 columns when its output is not a terminal, which would cut a path in
    an error message in two for whoever greps the log.
    """
    return Console(
        stderr=True,
        markup=False,
        highlight=False,
        emoji=False,
        soft_wrap=True,
        no_color=_no_color(args),
    )


def _with_advice(message: str, advice: str) -> str:
    """``message`` as spec 17.9 words an error: what happened, an em dash, what to do.

    A message that already says what to do is left as it is.
    """
    detail = " ".join(message.split()).rstrip(".") or "the model request failed"
    return detail if " — " in detail else f"{detail} — {advice}"


def _report(outcome: _Outcome, output_format: str, err: Console) -> None:
    if output_format == "json":
        print(json.dumps(outcome.payload, indent=2))
    else:
        _write_result(outcome.text)
    sys.stdout.flush()
    if outcome.error is not None:
        show_error(err, outcome.error)


def _write_result(text: str) -> None:
    """The answer, exactly: with a final newline if it had none, and nothing else added.

    Piped to a file it is what the model wrote. Shown on a terminal it is text from
    outside, and a terminal acts on the control sequences inside it, so those are removed.
    """
    shown = sanitize(text) if sys.stdout.isatty() else text
    if not shown:
        return
    sys.stdout.write(shown if shown.endswith("\n") else shown + "\n")


def _build_session(
    args: argparse.Namespace, settings: _Settings, home: Path, console: Console | None
) -> Session:
    config = load_config(home=str(home), project=settings.root, env=dict(os.environ))
    if args.model:
        config = replace(config, roles=replace(config.roles, main=args.model))
    for role, alias in settings.roles:
        config = replace(config, roles=replace(config.roles, **{role: alias}))
    if args.max_turns is not None:
        config = replace(config, limits=replace(config.limits, max_turns=args.max_turns))

    policy = Policy(
        sandbox=Sandbox(settings.roots),
        rules=RuleSet.build(
            allow=list(config.permissions.allow),
            ask=list(config.permissions.ask),
            deny=list(config.permissions.deny),
        ),
        mode=settings.mode,
        secret_paths=SECRET_PATH_PATTERNS,
        allow_secrets=args.allow_secrets,
    )
    state_dir = home / CONFIG_DIRNAME
    store = Store(state_dir / "sessions.db")
    store.open()
    try:
        session_id, resume = _which_session(args, store, settings.root)
        router = Router(config, CapabilityCache(state_dir / "capabilities.json"))
        todo = TodoState()
        # One redactor for what the tools return and for what the audit records: with
        # --allow-secrets a credentials file is read, and the redactor must let it through.
        redactor = Redactor(enabled=not args.allow_secrets)
        return Session(
            root=settings.root,
            config=config,
            router=router,
            registry=default_registry(todo),
            policy=policy,
            ui=_ui(console, settings.root, router),
            store=store,
            audit=AuditLog(store, redactor),
            session_id=session_id,
            resume=resume,
            todo_state=todo,
            redactor=redactor,
            home=str(home),
        )
    except BaseException:
        # No session exists to close the store, and nobody else has it.
        store.close()
        raise


def _ui(console: Console | None, root: str, router: Router) -> UI:
    """The front end for the session: none that asks, for headless, and ConsoleUI for the REPL.

    ConsoleUI is told where the project is, so that it shows ``src/a.py`` and not an absolute
    path, and how to find the model behind a role, so that its waiting indicator names the
    model and not the role.
    """
    if console is None:
        return AutoDecline()
    return ConsoleUI(console, root=root, model_of=_model_of(router))


def _model_of(router: Router) -> Callable[[str], str]:
    def model_of(role: str) -> str:
        # Read when asked and not when built: /model changes which model plays a role.
        config = router.config
        return config.models[config.roles.alias_for(role)].model

    return model_of


def _which_session(args: argparse.Namespace, store: Store, root: str) -> tuple[str, bool]:
    """The id of the session to run in, and whether it is one that was stored before.

    A resumed session is taken up through the session's own resume path and not by handing
    its id to the constructor, which would try to create the row again. ``--continue`` is the
    latest session started in this directory, not the latest in the store: someone working in
    one project is not taken back into another's conversation. Resuming by id is held to the
    same rule, since a session's sandbox was the directory it began in.
    """
    if args.resume:
        row = store.session_row(args.resume)
        if row is not None and os.path.realpath(row.cwd) != root:
            raise UsageError(
                f"session {row.id} was started in {row.cwd}, not in {root} "
                f"— run ncc from {row.cwd} (or pass --root {row.cwd}), or leave out --resume"
            )
        return args.resume, True
    if args.continue_last:
        latest = store.latest_session_id(cwd=root)
        if latest is None:
            raise UsageError(
                f"no earlier session was started in {root} — leave out --continue to start one"
            )
        return latest, True
    return new_session_id(), False


async def _run_interactive(
    args: argparse.Namespace, settings: _Settings, home: Path, console: Console
) -> int:
    session = _build_session(args, settings, home, console)
    try:
        return await run_repl(session, console, str(home / CONFIG_DIRNAME / "history"))
    finally:
        await session.aclose()


async def _run_headless(args: argparse.Namespace, settings: _Settings, home: Path) -> _Outcome:
    session = _build_session(args, settings, home, None)
    try:
        try:
            done = await session.follow_up(args.prompt)
        except ModelError as exc:
            kept = session.cut_off_text
            if kept is None:
                raise
            # The stream broke after text had arrived and the session kept it: that text is
            # the result there is, and the run still failed.
            return _Outcome(
                payload=_payload(session, kept, "cut_off", session.state.turn, is_error=True),
                text=kept,
                error=_with_advice(str(exc), _PROVIDER_ADVICE),
                code=EXIT_CODES["provider"],
            )
        return _finished(session, done)
    finally:
        await session.aclose()


def _finished(session: Session, done: Done) -> _Outcome:
    reason = done.reason
    completed = reason is StopReason.COMPLETED
    # The words of the model are the result. The loop's closing line and the unsuitable-model
    # message are the program's own, and belong on stderr.
    models_words = reason in (StopReason.COMPLETED, StopReason.REFUSAL, StopReason.MAX_TOKENS)
    return _Outcome(
        payload=_payload(session, done.text, reason.value, done.state.turn, is_error=not completed),
        text=done.text if models_words else "",
        error=_stop_line(session, done),
        code=EXIT_CODES["completed" if completed else "stopped"],
    )


def _stop_line(session: Session, done: Done) -> str | None:
    """Why a prompt stopped before it finished, in the form of spec 17.9."""
    match done.reason:
        case StopReason.COMPLETED:
            return None
        case StopReason.TURN_LIMIT:
            limit = session.config.limits.max_turns
            return (
                f"the turn limit of {limit} was reached before the task was finished "
                "— raise it with --max-turns, or give the model a smaller task"
            )
        case StopReason.MODEL_UNSUITABLE:
            # Already worded for a person, and names the model.
            return done.text
        case StopReason.REFUSAL:
            return (
                "the model declined to continue "
                "— rephrase the request, or try another model with --model"
            )
        case StopReason.MAX_TOKENS:
            return (
                "the reply was cut off at the model's output limit "
                "— ask for less in one request, or use a model that can write more (--model)"
            )
        case _:
            assert_never(done.reason)


def _payload(
    session: Session, result: str, stop_reason: str, turns: int, *, is_error: bool
) -> dict[str, Any]:
    usage = session.router.total_usage()
    return {
        "session_id": session.session_id,
        "result": result,
        "stop_reason": stop_reason,
        "turns": turns,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
        },
        "cost_usd": session.router.total_cost(),
        "is_error": is_error,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
