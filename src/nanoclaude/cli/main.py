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
import ast
import asyncio
import contextlib
import json
import os
import re
import shlex
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, assert_never

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
from nanoclaude.cli.init import run_init
from nanoclaude.cli.render import ConsoleUI, show_error
from nanoclaude.cli.repl import run_repl
from nanoclaude.config.load import CONFIG_DIRNAME, ConfigError, expand_root, load_config
from nanoclaude.config.schema import ROLES
from nanoclaude.conversation.budget import ContextTooSmallError
from nanoclaude.conversation.store import Store, same_directory
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox, is_within
from nanoclaude.providers.base import CredentialsError, ModelError
from nanoclaude.providers.capabilities import CACHE_FILENAME, CapabilityCache
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
    # Not a failure: whoever reads the output closed it before the end (``| head -1``). The
    # shell's own number for a process killed by SIGPIPE, which is what this stands for.
    "output_closed": 141,
}

#: Where a provider error that does not say what to do is sent.
_PROVIDER_ADVICE = "try again, or choose another model with --model"


def nanoclaude_home() -> Path:
    """The directory that holds ``.nanoclaude``: the person's home, unless told otherwise.

    ``NANOCLAUDE_HOME`` may start with ``~``, as every other path a person writes may, and
    it is expanded here: the loader expands it too, so a path left as it was would be one
    place for what writes the config and another for what reads it.

    Raises :class:`~nanoclaude.config.load.ConfigError` for ``~name`` with no such user.
    """
    given = os.environ.get("NANOCLAUDE_HOME")
    if given is None:
        return Path.home()
    try:
        return Path(given).expanduser()
    except RuntimeError as exc:
        raise ConfigError(
            f"NANOCLAUDE_HOME is {given!r}, which starts with ~ but names no home directory "
            "\u2014 write the full path, or unset it"
        ) from exc


class UsageError(Exception):
    """``ncc`` was invoked wrongly. The message is the person's line, as spec 17.9 words it."""


class StoreOpenError(Exception):
    """The session store could not be opened. The message names the file and the reason."""

    def __init__(self, path: Path, cause: Exception) -> None:
        reason = (cause.strerror if isinstance(cause, OSError) else None) or str(cause)
        super().__init__(
            f"cannot open the session store at {path} ({reason or type(cause).__name__}) "
            "— check that the directory exists and that you can write to it"
        )


def _turns(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise UsageError(f"--max-turns needs a whole number of at least 1 — got {text!r}")
    return value


#: What ``--mode`` may be set to by itself. Bypass is also a mode, but only with the flag
#: whose name says so, and what is offered to someone who typed a mode wrongly is not that.
_MODES = tuple(mode.value for mode in PermissionMode if mode is not PermissionMode.BYPASS)
_OUTPUT_FORMATS = ("text", "json")

_FLAGS_HELP = "run ncc --help for the flags there are"

#: The commands ncc has, beside running a prompt. ``ncc init`` is the only one.
_COMMANDS = ("init",)
#: What argparse calls the place a command goes, in the message it words for a wrong one.
_COMMAND = "COMMAND"

_EXPECTED_ONE = re.compile(r"argument (?P<flag>\S+): expected one argument")
_INVALID_CHOICE = re.compile(
    r"argument (?P<flag>\S+): invalid choice: (?P<value>.*?) \(choose from", re.DOTALL
)
_NOT_ALLOWED = re.compile(r"argument (?P<flag>\S+): not allowed with argument (?P<other>\S+)")


def _long(flag: str) -> str:
    """``--resume`` for argparse's ``-r/--resume``."""
    return flag.rpartition("/")[2]


def _flag_error(message: str) -> str:
    """What argparse found wrong, as spec 17.9 words an error: what happened, a dash, what to do.

    argparse words these for itself and puts what was typed into them as it was. The line is
    printed through ``show_error``, which makes it one line without anything in it for a
    terminal to act on; this gives it the form. What is not one of the cases it knows is
    still said, with where to look.
    """
    if match := _EXPECTED_ONE.fullmatch(message):
        flag = _long(match["flag"])
        if flag == "--print":
            return "-p needs a prompt — pass the task as its argument"
        return f"{flag} needs a value — pass one"
    if match := _INVALID_CHOICE.match(message):
        flag = _long(match["flag"])
        try:
            value = ast.literal_eval(match["value"])
        except (ValueError, SyntaxError):
            value = match["value"]
        if flag == _COMMAND:
            # What was typed where a command goes is most often a task, said without -p.
            return (
                f"{value} is not a command \u2014 the only command is {', '.join(_COMMANDS)}; "
                "to give ncc a task, pass it with -p"
            )
        what, choices = ("mode", _MODES) if flag == "--mode" else ("format", _OUTPUT_FORMATS)
        return f"{flag} {value} is not a {what} — choose one of {', '.join(choices)}"
    if match := _NOT_ALLOWED.fullmatch(message):
        first, second = _long(match["flag"]), _long(match["other"])
        return f"{first} and {second} cannot be combined — pass only one"
    return f"{message} — {_FLAGS_HELP}"


class _Parser(argparse.ArgumentParser):
    """An argument parser that reports a mistake as :class:`UsageError`, and does not exit.

    argparse prints its usage and ``ncc: error: ...`` and ends the process, with what was
    typed echoed as it was. ``--help`` and ``--version`` are not mistakes and still print to
    stdout and exit 0.
    """

    def error(self, message: str) -> NoReturn:
        raise UsageError(_flag_error(message))

    def parse_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        parsed, extra = self.parse_known_args(args, namespace)
        if extra:
            plural = "s" if len(extra) > 1 else ""
            raise UsageError(f"unrecognized argument{plural} {' '.join(extra)} — {_FLAGS_HELP}")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="ncc",
        description=(
            "A terminal coding agent that works with Anthropic, OpenAI-compatible and local models."
        ),
        # A flag is spelled out. With abbreviations on, --dangerously-skip would turn the
        # permission checks off, and spec 6.2 gives that flag its long name on purpose.
        allow_abbrev=False,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=_COMMANDS,
        metavar=_COMMAND,
        help="init: pick a provider, give a key, and write the config; run it once, first",
    )
    parser.add_argument("-p", "--print", dest="prompt", metavar="PROMPT", help="run once and exit")
    # No default: whether it was given at all is what says it has no use without a prompt.
    parser.add_argument("--output-format", choices=_OUTPUT_FORMATS)
    # No default: whether it was given at all is what says it was meant. Left out it is the
    # working directory, and a working directory that would put a whole home directory in
    # the sandbox is refused (see _refuse_a_root_that_is_too_wide).
    parser.add_argument(
        "--root",
        default=None,
        help="working directory (default: cwd, but never a home directory or / unless named)",
    )
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
        # argparse ends --help and --version by raising, and this function does not end the
        # process: the console script does, with what it returns.
        return exc.code if isinstance(exc.code, int) else 1
    except UsageError as exc:
        # Nothing was parsed, so whether colour was asked off is read from what was typed.
        flagged = "--no-color" in (sys.argv[1:] if argv is None else argv)
        show_error(_stderr_console(flagged), str(exc))
        return EXIT_CODES["usage"]
    err = _stderr_console(args.no_color)
    try:
        home = nanoclaude_home()
        if args.command == "init":
            return _init(args, home, err, sys.argv[1:] if argv is None else argv)
        if args.prompt is not None and not args.prompt.strip():
            raise UsageError("-p needs a prompt — pass the task as its argument")
        if args.prompt is None and args.output_format is not None:
            raise UsageError(
                "--output-format needs -p — it shapes the answer to one prompt; "
                "pass a prompt with -p"
            )
        if args.prompt is None and not _stdin_is_terminal():
            raise UsageError(
                "no prompt and no terminal — pass a prompt with -p, or run ncc in a terminal"
            )
        settings = _settings(args, home)
        _warn_about_switches(args, err)
        if args.prompt is None:
            return asyncio.run(
                _run_interactive(args, settings, home, _stdout_console(args.no_color))
            )
        outcome = asyncio.run(_run_headless(args, settings, home))
        _report(outcome, as_json=args.output_format == "json", err=err)
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
    except StoreOpenError as exc:
        show_error(err, str(exc))
        return EXIT_CODES["stopped"]
    except BrokenPipeError:
        _abandon_stdout()
        return EXIT_CODES["output_closed"]
    except KeyboardInterrupt as exc:
        # Ctrl+C has no words of its own; one that came with a message (ncc init, interrupted
        # after it wrote the config) has, and a person who is told nothing more was done when
        # a file was written is told something false.
        show_error(err, str(exc) or "interrupted \u2014 nothing more was done; run again to retry")
        return EXIT_CODES["interrupted"]
    except Exception as exc:
        # Last, after every exception that has a line and a code of its own. A script reads
        # the exit code and the first line of stderr, and a traceback is neither.
        show_error(err, _unexpected(exc))
        return EXIT_CODES["stopped"]


def _init(args: argparse.Namespace, home: Path, err: Console, typed: Sequence[str]) -> int:
    """``ncc init``: the questions, the config and the check, in ``home``.

    It asks, so it has no use without somebody to answer: with no terminal it says so in its
    own words, before it asks anything, and not as the no-prompt error of a run that lost
    its ``-p``. Every flag but ``--no-color`` shapes a session and init starts none, so one
    that is given is refused and not ignored: a setting the person believes is in force and
    is not is what this program refuses everywhere else.
    """
    for word in typed:
        if word.startswith("-") and word != "--no-color":
            raise UsageError(
                f"ncc init takes no {word} \u2014 run it on its own "
                "(--no-color is the only flag that goes with it)"
            )
    if not _stdin_is_terminal():
        raise UsageError("ncc init asks questions \u2014 run it in a terminal")
    try:
        return run_init(_stdout_console(args.no_color), home=home)
    except EOFError:
        show_error(
            err,
            "the input ended before ncc init was finished \u2014 nothing was written; "
            "run ncc init again",
        )
        return EXIT_CODES["stopped"]


def _settings(args: argparse.Namespace, home: Path) -> _Settings:
    mode = _permission_mode(args)
    roles = _role_overrides(args.role)
    # Given and empty is not left out: "$UNSET" in a script says something went wrong, and a
    # run that goes on as if the flag were not there does something else than was meant.
    if args.resume is not None and not args.resume.strip():
        raise UsageError("--resume needs a session id — pass one, or use --continue")
    if args.model is not None and not args.model.strip():
        raise UsageError("--model needs a model alias — pass one, or leave the flag out")
    for role, alias in roles:
        if role == "main" and args.model is not None and alias != args.model:
            raise UsageError(
                f"--model {args.model} and --role main={alias} say different things about the "
                "main role — pass only one of them"
            )
    root = _directory("--root", "." if args.root is None else args.root)
    if args.root is None:
        _refuse_a_root_that_is_too_wide(root, home)
    return _Settings(
        root=root,
        roots=(root, *(_directory("--add-dir", raw) for raw in args.add_dir)),
        mode=mode,
        roles=roles,
    )


def _refuse_a_root_that_is_too_wide(root: str, home: Path) -> None:
    """Stop, as a usage error, when the directory ncc was run in would put too much in reach.

    The sandbox is whatever the root holds, and a root that holds a home directory holds the
    session store (the whole conversation, tool output included), the shell's startup files
    and every other project there is. That is never what running ``ncc`` somewhere means, so
    it is refused unless ``--root`` names it, which says it was meant. Three cases, from the
    widest: the filesystem root; a home directory, the person's own and the one ncc keeps its
    files in, which are the same unless ``NANOCLAUDE_HOME`` says otherwise; and a directory
    that holds one of them.
    """
    if root == "/":
        raise UsageError(
            "the current directory is the filesystem root — every file on this machine would "
            "be in reach; run ncc from a project directory, or pass --root / if you mean it"
        )
    homes = [os.path.realpath(home)]
    with contextlib.suppress(RuntimeError):  # no home can be found for this user at all
        homes.append(os.path.realpath(Path.home()))
    if root in homes:
        raise UsageError(
            "the current directory is your home directory — every file in it would be in "
            "reach; run ncc from a project directory, or pass --root ~ if you mean it"
        )
    if any(is_within(root, held) for held in homes):
        raise UsageError(
            "the current directory contains your home directory — every file in it would be "
            f"in reach; run ncc from a project directory, or pass --root {shlex.quote(root)} "
            "if you mean it"
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
    if not raw.strip():
        # expand_root("") is the working directory, and the sandbox would take it for one
        # the person had named. An empty value is what an unset variable gives.
        raise UsageError(f"{flag} needs a directory — pass one, or leave the flag out")
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


def _stdin_is_terminal() -> bool:
    """Whether there is somebody to type at: stdin is a terminal, and not a pipe or a file.

    A stdin that is closed, or not there at all, is not one.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:
        return False


def _colourless(flag: bool) -> bool:
    return flag or bool(os.environ.get("NO_COLOR"))


def _stdout_console(no_color: bool) -> Console:
    """What the REPL writes to. What it prints from outside is built as Text, which Rich never
    parses as markup, so the console is left as Rich makes it."""
    return Console(no_color=_colourless(no_color))


def _stderr_console(no_color: bool) -> Console:
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
        no_color=_colourless(no_color),
    )


def _is_busy(exc: BaseException) -> bool:
    """A store that another process holds a lock on: not a bug, and gone by itself."""
    return isinstance(exc, sqlite3.OperationalError) and any(
        word in str(exc).lower() for word in ("locked", "busy")
    )


def _unexpected(exc: Exception) -> str:
    """What a person is told of an exception nobody planned for, as spec 17.9 words it."""
    if _is_busy(exc):
        return "the session store is busy — another ncc may be using it; try again"
    detail = " ".join(str(exc).split())
    what = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return f"unexpected error ({what}) — this is a bug; please report it"


def _with_advice(message: str, advice: str) -> str:
    """``message`` as spec 17.9 words an error: what happened, an em dash, what to do.

    A message that already says what to do is left as it is.
    """
    detail = " ".join(message.split()).rstrip(".") or "the model request failed"
    return detail if " — " in detail else f"{detail} — {advice}"


def _report(outcome: _Outcome, *, as_json: bool, err: Console) -> None:
    if as_json:
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
    It is written as UTF-8 bytes, so that a locale that cannot write a character does not
    stop the reply; one that cannot be encoded at all (a lone surrogate) becomes ``?``.
    """
    shown = sanitize(text) if sys.stdout.isatty() else text
    if not shown:
        return
    sys.stdout.buffer.write(
        (shown if shown.endswith("\n") else shown + "\n").encode("utf-8", errors="replace")
    )


def _abandon_stdout() -> None:
    """Point stdout at nowhere, after the reader closed it.

    Python flushes stdout once more as it exits, finds the pipe closed and prints a complaint
    of its own to stderr, with another exit status. What there was to write has been given up
    on, so what is left over is written to the bit bucket.
    """
    # A stdout with no descriptor behind it has nothing to flush.
    with contextlib.suppress(OSError, ValueError):
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


def _build_session(
    args: argparse.Namespace, settings: _Settings, home: Path, console: Console | None
) -> Session:
    config = load_config(home=str(home), project=settings.root, env=dict(os.environ))
    if args.model is not None:
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
    try:
        _open(store)
        session_id, resume = _which_session(args, store, settings.root)
        router = Router(config, CapabilityCache(state_dir / CACHE_FILENAME))
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


def _open(store: Store) -> None:
    """Open the store. What cannot be opened is reported with where it was looked for.

    A store that is only busy is left for the general handler, which words that for itself.
    """
    try:
        store.open()
    except (OSError, sqlite3.Error) as exc:
        if _is_busy(exc):
            raise
        raise StoreOpenError(store.path, exc) from exc


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
    if args.resume is not None:
        row = store.session_row(args.resume)
        if row is not None and not same_directory(row.cwd, root):
            there = shlex.quote(row.cwd)
            raise UsageError(
                f"session {row.id} was started in {row.cwd}, not in {root} "
                f"— run ncc from {there} (or pass --root {there}), or leave out --resume"
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
            cause = exc.__cause__
            if not (isinstance(cause, ModelError) and cause.partial is not None):
                raise
            # The stream broke midway. What arrived is kept by the session when there was any,
            # and the advice it gives is for somebody at a prompt: a script is told how to go on.
            kept = session.cut_off_text
            line = _cut_off_line(exc, kept is not None)
            if kept is None:
                raise ModelError(line) from exc
            # That text is the result there is, and the run still failed.
            return _Outcome(
                payload=_payload(session, kept, "cut_off", session.state.turn, is_error=True),
                text=kept,
                error=line,
                code=EXIT_CODES["provider"],
            )
        return _finished(session, done)
    finally:
        await session.aclose()


def _cut_off_line(error: ModelError, kept: bool) -> str:
    """The session's line about a reply that broke off, with what a script can do for its advice.

    The line is "what happened — what to do", and what happened is the part before the last
    dash: the cause may have a dash in it, and the advice is the part after.
    """
    happened, dash, _advice = str(error).rpartition(" \u2014 ")
    happened = happened if dash else str(error)
    rest = "what arrived is kept; " if kept else ""
    return f'{happened} \u2014 {rest}run: ncc --continue -p "continue"'


def _finished(session: Session, done: Done) -> _Outcome:
    reason = done.reason
    completed = reason is StopReason.COMPLETED
    # The words of the model are the result. The loop's closing line and the unsuitable-model
    # message are the program's own, and belong on stderr.
    models_words = reason in (StopReason.COMPLETED, StopReason.REFUSAL, StopReason.MAX_TOKENS)
    result = _as_written(done) if models_words else done.text
    return _Outcome(
        payload=_payload(session, result, reason.value, done.state.turn, is_error=not completed),
        text=result if models_words else "",
        error=_stop_line(session, done),
        code=EXIT_CODES["completed" if completed else "stopped"],
    )


def _as_written(done: Done) -> str:
    """The reply as the model wrote it. ``done.text`` is stripped, and the indentation of the
    first line of a file is code."""
    last = done.state.transcript.last()
    return last.text() if last is not None and last.role == "assistant" else done.text


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
