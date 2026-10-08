"""The first sixty seconds.

Someone who has just run pipx install has a key in a password manager and no idea what
a role is. This asks three questions, writes a commented config, verifies the key with
one cheap request, and stops. Everything else is a default they can read later in the
file itself, which is why the file is written with comments rather than as minimal TOML.

The key is never written. The file names an environment variable and init prints the
export line, because a config file with a secret in it is a config file that ends up in a
dotfiles repository. The pasted key goes to one place only: the verification request.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from rich.console import Console
from rich.markup import escape

from nanoclaude.agent.router import Router
from nanoclaude.cli.render import show_error
from nanoclaude.config.load import CONFIG_DIRNAME, CONFIG_FILENAME, ConfigError, load_config
from nanoclaude.config.schema import ROLES, Config, LimitsConfig, PermissionsConfig
from nanoclaude.conversation.transcript import Transcript, user_text
from nanoclaude.providers.base import EmptyReplyError, ModelError, ModelRequest
from nanoclaude.providers.capabilities import CACHE_FILENAME, CapabilityCache
from nanoclaude.tools.base import sanitize

#: Asks one question and returns the answer, as ``ask(message, choices=..., default=...,
#: password=...)``. ``choices`` are the answers that are accepted, ``default`` is what Enter
#: gives, and ``password`` keeps the answer off the screen.
Ask = Callable[..., str]

#: Sends one small request with the model ``alias`` names and returns what is wrong, or None.
Verifier = Callable[[Config, str], Coroutine[Any, Any, str | None]]


@dataclass(frozen=True, slots=True)
class Preset:
    """A provider as the person would name it, and what ncc needs to know to talk to it."""

    key: str
    label: str
    adapter: str
    default_model: str
    #: The environment variable the key is read from; empty for a provider that needs none.
    key_env: str
    base_url: str | None = None
    needs_key: bool = True


#: In the order they are offered: the number a person types is the position in this table.
PRESETS: dict[str, Preset] = {
    "anthropic": Preset(
        "anthropic", "Anthropic (Claude)", "anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY"
    ),
    "openai": Preset(
        "openai",
        "OpenAI",
        "openai_compat",
        "gpt-5",
        "OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
    ),
    "openrouter": Preset(
        "openrouter",
        "OpenRouter (many models, one key)",
        "openai_compat",
        "deepseek/deepseek-v4-pro",
        "OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "ollama": Preset(
        "ollama", "Ollama (local, free)", "ollama", "qwen3-coder", "", needs_key=False
    ),
}

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_string(text: str) -> str:
    """``text`` as a TOML basic string: what was typed is a value, and never more TOML.

    Quotes, backslashes and control characters are escaped, so that a model name with a
    newline in it cannot add a section to the file it is written into.
    """
    out = []
    for char in text:
        if char in '"\\':
            out.append("\\" + char)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _toml_key(name: str) -> str:
    return name if _BARE_KEY.fullmatch(name) else _toml_string(name)


def _list(items: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_string(item) for item in items) + "]"


def render_config(preset: Preset, alias: str, model: str) -> str:
    """The config file for ``preset``: every role on one model, and a comment for each choice.

    Written with comments because the file is the documentation: what init does not ask is
    a default the person reads here when they want to change it. The permission lists are
    the program's own defaults plus one deny rule, so a person who never edits the file
    gets what they would have got without it.
    """
    permissions, limits = PermissionsConfig(), LimitsConfig()
    base_url = f"base_url    = {_toml_string(preset.base_url)}\n" if preset.base_url else ""
    key_env = f"api_key_env = {_toml_string(preset.key_env)}\n" if preset.needs_key else ""
    roles = "\n".join(f"{role:<7} = {_toml_string(alias)}" for role in ROLES)
    return f"""\
# nano-claude-code configuration.
# Everything here has a default; delete anything you do not want to change.
# A key is never kept in this file: a model names the environment variable that holds
# its key (api_key_env), and ncc reads the key from there.

[models.{_toml_key(alias)}]
adapter     = {_toml_string(preset.adapter)}
model       = {_toml_string(model)}
{base_url}{key_env}
# Roles let one session use different models for different jobs. Exploring a
# codebase is most of the tokens and none of the difficulty, so pointing
# `explore` at a cheap or local model is the easiest saving available.
# Add another [models.<alias>] block and name it here.
[roles]
{roles}

# What ncc may do without asking, what it asks about, and what it never does.
[permissions]
allow = {_list(permissions.allow)}
ask   = {_list(permissions.ask)}
deny  = ["Read(**/.env*)"]

# Guard rails on one run: how many model turns it may take, and how long a shell command
# may run, in seconds.
[limits]
max_turns      = {limits.max_turns}
bash_timeout_s = {limits.bash_timeout_s:g}
"""


def ask_on_terminal(
    message: str,
    *,
    choices: Sequence[str] | None = None,
    default: str | None = None,
    password: bool = False,
    input: Input | None = None,
    output: Output | None = None,
) -> str:
    """Ask one question on the terminal; the answer, without the spaces around it.

    ``choices`` are the answers that are accepted, in any case, and the question is asked
    again until it gets one; the answer given back is the choice as it is spelled there.
    Enter alone gives ``default``. ``password`` keeps the answer off the screen: a mask is
    drawn in its place, and a default is neither shown nor taken. ``input`` and ``output``
    are the terminal's unless given.

    Every question gets a prompt of its own. A prompt keeps what it was given, and Up brings
    it back: the model's question must not be able to show the key.

    Raises ``KeyboardInterrupt`` for Ctrl+C and ``EOFError`` for Ctrl+D at an empty line, as
    ``input()`` does.
    """
    shown_default = default if default is not None and not password else None
    label = message
    if choices:
        label += f" [{'/'.join(choices)}]"
    if shown_default:
        label += f" ({shown_default})"
    prompt = f"{label}: "
    while True:
        session: PromptSession[str] = PromptSession(input=input, output=output)
        answer = session.prompt(prompt, is_password=password).strip()
        if not answer and shown_default is not None:
            answer = shown_default
        if not choices:
            return answer
        for choice in choices:
            if choice.lower() == answer.lower():
                return choice
        prompt = f"{message} (choose one of {', '.join(choices)}): "


#: How long the check waits for the provider. A model that has to be loaded into memory first
#: (a local one, cold) can take most of this; a person who has waited this long has a problem
#: that another few seconds will not mend.
VERIFY_TIMEOUT_S = 60.0


async def verify(config: Config, alias: str) -> str | None:
    """Send one small request with the model ``alias`` names. None if it was answered.

    The request goes through the client ncc builds for the alias, so what is checked is what
    will run: the key from the environment the config was loaded with, the address, the
    parameter the provider wants its output cap in, no temperature. The cap is eight tokens.
    A reply cut off at it is still an answer, whether it holds thinking or, from a model
    that thinks where nobody can see, nothing at all (``EmptyReplyError``): the key, the
    model and the request were accepted, which is all that is read.

    Whatever the provider or the network does comes back as a line for the person and is
    never raised, with the key cut out of it: a provider can say a key back in its error,
    and a library can put a header in an exception. A missing or refused key is a
    ``CredentialsError`` and is reported like the rest. ``alias`` must be one of the
    config's models, as it is for ncc itself.

    The 60 seconds bound the wait for an answer, not the work behind the wait. A name is
    looked up in a thread of the event loop's own, which nothing can interrupt, so a lookup
    that hangs outlasts the limit: the wait is given up, the line is returned, and the loop
    then waits for the thread when it closes (without limit on Python 3.11, for up to five
    minutes on 3.12 and later). That is accepted, and no thread of ours is added to get
    around it: a machine whose resolver hangs has a problem of which this is not the worst.
    """
    key = config.api_key_for(alias)
    flaw = _key_flaw(key)
    if flaw is not None:
        return _flawed_key_line(
            f"the key in {config.api_key_env_for(alias)}", flaw, config.api_key_env_for(alias)
        )
    request = ModelRequest("Reply with: ok", Transcript((user_text("ping"),)), (), 8)
    # A router for its way of making the client. The cache is a path that is never read: no
    # capability is asked for, and if one ever were, a file under /dev/null cannot be made.
    router = Router(
        replace(config, roles=replace(config.roles, main=alias)), CapabilityCache(Path(os.devnull))
    )
    try:
        client = await router.client_for("main")
        await asyncio.wait_for(client.complete(request), VERIFY_TIMEOUT_S)
    except TimeoutError:
        return f"no answer within {VERIFY_TIMEOUT_S:g} seconds — check your network connection"
    except EmptyReplyError:
        return None
    except ModelError as exc:
        return _redact(str(exc), key)
    except Exception as exc:
        # Whatever it is, init ends in a line for the person and not in a traceback.
        return _redact(f"{type(exc).__name__}: {exc}", key)
    finally:
        await router.aclose()
    return None


def _outside(text: str) -> str:
    """Text that came from outside (a provider's error, a path), safe to put in a template.

    Never read as markup, so ``[/]`` in a path cannot raise; stripped of anything a terminal
    would act on; and one line, so that it cannot start a line of its own.
    """
    return escape(" ".join(sanitize(text).split()))


def _say(console: Console, markup: str) -> None:
    """One line. Not wrapped: a path cut in two at column 80 is a path nobody can copy."""
    console.print(markup, soft_wrap=True, highlight=False)


def _redact(text: str, key: str) -> str:
    """``text`` without the key in it, wherever it appears.

    A provider is free to say a key back in its error, and a library to put a header value
    in an exception, and a library writes one as a bytes literal: a line break in the key is
    ``\\n`` there, which the key's own text is not found in. So the key is cut out as it is
    and in the two escaped forms it can take. This is the one place where what is shown is
    checked against the key. An empty key is nothing to hide, and replacing it would put the
    marker between every two characters.
    """
    if not key:
        return text
    for form in (key, repr(key)[1:-1], key.encode("unicode_escape").decode()):
        text = text.replace(form, "[hidden]")
    return text


#: What a key may be made of: the printable ASCII characters but the space. A key is sent as
#: the value of a header, and anything else is not one: a line break, which is a paste that
#: took two lines; a space or a tab, which is a paste that took a word of the page around it;
#: a character that is not ASCII, such as a zero-width space or a non-breaking space, which
#: a page puts into text that is copied from it and which no one can see.
_NOT_A_KEY_CHARACTER = re.compile(r"[^\x21-\x7e]")


def _key_flaw(key: str) -> str | None:
    """What is wrong with ``key`` as the value of a header, or None; without ever quoting it.

    It says what kind of character, and where it is (counting from 1, in the key as it is):
    what is said about a key is the one thing that may not hold the key. An empty key is not
    a flaw here: it is a key that is missing, which has words of its own.
    """
    found = _NOT_A_KEY_CHARACTER.search(key)
    if found is None:
        return None
    character = found.group()
    if character == " ":
        what = "a space"
    elif character == "\t":
        what = "a tab"
    elif character in "\r\n":
        what = "a line break"
    elif ord(character) < 0x20 or ord(character) == 0x7F:
        what = "a control character"
    else:
        what = "a character that is not plain ASCII"
    return f"{what} at position {found.start() + 1}"


def _flawed_key_line(whose: str, flaw: str, variable: str) -> str:
    """The line for a key that was not sent, in the form of spec 17.9."""
    return (
        f"{whose} contains {flaw} \u2014 it was not sent anywhere; copy the key again, exactly "
        f"as it was given and with nothing around it, and set {variable} to it"
    )


class _TargetExists(Exception):
    """The config is there, and the write was not told that replacing it was agreed to."""


def _make_private_directories(directory: Path) -> None:
    """Make ``directory`` and every part of it that is missing, each with mode 0700.

    Only what is made here is touched: a directory that was already there keeps its mode.
    ``mkdir`` filters its mode by the umask, so that with a umask of 0177 or tighter (any bit
    of 0300) a new directory would lose its owner's write or search bit and nothing could be
    made in it; the mode is set again after, and the directory that holds a private config
    is private too.
    """
    missing = []
    for candidate in (directory, *directory.parents):
        if candidate.exists():
            break
        missing.append(candidate)
    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        candidate.chmod(0o700)


def _write_private(path: Path, text: str, *, replace: bool) -> None:
    """Write ``text`` to ``path``, which exists with mode 0600 from the moment it does.

    The file is made by ``os.open`` with the mode in its arguments, so that no other user
    can read it between its being made and its being made private; that mode is filtered by
    the umask (0277 makes it 0400, 0777 makes it 0000), so it is set on the descriptor too.
    It is then moved into place: a file that was already there is replaced by one that is
    private, whatever its own mode was, and one that was interrupted leaves no half of a
    config. ``O_EXCL`` refuses a temporary name that is already there, a link included, so
    that nothing is written through one.

    ``replace`` says whether replacing a config is agreed to. When it is not, the file is
    put in place by a hard link, which fails if anything is there, a dangling link included,
    and :class:`_TargetExists` is raised and nothing is replaced: a config that appeared
    after the last look is somebody's work.
    """
    _make_private_directories(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text)
        if replace:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise _TargetExists from exc
            with contextlib.suppress(OSError):
                temporary.unlink()
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _what_is_there(target: Path) -> str:
    """What the config that is there is, for the question about replacing it.

    A link is said to be one, with where it leads, because replacing it is not replacing
    what it leads to: the link goes, a regular file takes its place, and the file it led
    to (often one in a dotfiles repository) is left as it is.
    """
    if target.is_symlink():
        try:
            leads_to = _outside(str(target.readlink()))
        except OSError:
            return f"{_outside(str(target))} already exists."
        return (
            f"{_outside(str(target))} is a link to {leads_to}. Answering y replaces the link "
            f"with a regular file; {leads_to} is left as it is."
        )
    return f"{_outside(str(target))} already exists."


def _may_replace(console: Console, ask: Ask, target: Path) -> bool:
    """Ask whether the config that is there may be replaced; True only for a plain yes."""
    _say(console, f"[yellow]{_what_is_there(target)}[/yellow]")
    if ask("overwrite?", choices=["y", "n"], default="n") != "y":
        _say(console, "[dim]left alone[/dim]")
        return False
    return True


def _publish(console: Console, ask: Ask, target: Path, text: str, agreed: bool) -> bool:
    """Put the config in place; False when the person, asked, would not have it replaced.

    Raises :class:`~nanoclaude.config.load.ConfigError` when it cannot be written.
    """
    try:
        try:
            _write_private(target, text, replace=agreed)
        except _TargetExists:
            if not _may_replace(console, ask, target):
                return False
            _write_private(target, text, replace=True)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError(
            f"cannot write {target} ({reason}) — check that the directory exists and that you "
            "can write to it"
        ) from exc
    return True


def run_init(
    console: Console,
    *,
    home: Path,
    ask: Ask = ask_on_terminal,
    env: Mapping[str, str] | None = None,
    verify: Verifier = verify,
) -> int:
    """Ask which provider, which model and (if one is needed) the key; write the config.

    ``home`` is the directory that holds ``.nanoclaude``. ``ask`` asks the questions and
    ``verify`` sends the check; ``env`` is the environment, this process's unless given. The
    key is used for one request and kept nowhere: not in the file, not in this process's
    environment, not on the screen.

    Returns the exit status: 0 when the setup was finished or left alone, even if the key
    could not be verified, because the file is written and the key is the person's to fix.
    Raises :class:`~nanoclaude.config.load.ConfigError` when the file cannot be written.
    """
    environment = dict(os.environ if env is None else env)
    target = home / CONFIG_DIRNAME / CONFIG_FILENAME
    agreed = os.path.lexists(target)
    if agreed and not _may_replace(console, ask, target):
        return 0

    _say(console, "[bold]Which provider?[/bold]")
    presets = list(PRESETS.values())
    for number, offered in enumerate(presets, start=1):
        _say(console, f"  {number}. {offered.label}")
    picked = ask("choice", choices=[str(n) for n in range(1, len(presets) + 1)], default="1")
    preset = presets[int(picked) - 1]

    key, whose = "", "the key you pasted"
    if preset.needs_key:
        key = environment.get(preset.key_env, "")
        if key:
            whose = f"the key in {preset.key_env}"
            _say(console, f"[dim]{preset.key_env} is already set; using it[/dim]")
        else:
            while not key:
                key = ask(
                    "paste your key (hidden; it is checked once and never saved)", password=True
                ).strip()
                if not key:
                    _say(
                        console,
                        "[yellow]a key is needed[/yellow] \u2014 paste it, or press Ctrl+C to stop",
                    )
    model = ask("model", default=preset.default_model).strip() or preset.default_model

    alias = preset.key
    if not _publish(console, ask, target, render_config(preset, alias, model), agreed):
        return 0
    _say(console, f"[green]wrote[/green] {_outside(str(target))}")
    _forget_what_was_learned_about_models(console, target.parent / CACHE_FILENAME)

    if key and not environment.get(preset.key_env):
        _say(
            console,
            "\nncc reads the key from your environment, and init does not save it. Add this to "
            "your shell profile, with your key in place of <your key>:\n"
            f"  [bold]export {preset.key_env}='<your key>'[/bold]\n",
        )

    # What the check uses is what ncc will use: the file just written, loaded as ncc loads it,
    # with the key it was given. It is added to this copy of the environment and nowhere else.
    held = {preset.key_env: key} if key else {}
    config = load_config(home=str(home), project=None, env={**environment, **held})
    _check_the_key(console, verify, config, alias, key, whose, target)
    _say(
        console,
        "\nTry it:\n  [bold]ncc[/bold]\n  [bold]ncc -p 'what does this project do?'[/bold]",
    )
    return 0


class InterruptedWhileChecking(KeyboardInterrupt):
    """Ctrl+C after the config was written. Its message is the line for the person.

    A ``KeyboardInterrupt``, so that whatever ends a run on Ctrl+C ends this one the same
    way, with the same status; the command prints the message and not its generic line.
    """


def _check_the_key(
    console: Console,
    check: Verifier,
    config: Config,
    alias: str,
    key: str,
    whose: str,
    config_path: Path,
) -> None:
    """Look at the key, and send the check if it is one that can be sent; say what came of it.

    A key that is not plain (see ``_key_flaw``) is not sent: the line says what is wrong and
    where, and nothing else is done with it. A check that fails is a warning and not an error
    of init's own: the file is written, and the key is the person's to fix.
    """
    flaw = _key_flaw(key)
    if flaw is not None:
        show_error(console, _flawed_key_line(whose, flaw, config.api_key_env_for(alias)))
        return
    _say(console, "[dim]checking with one small request...[/dim]")
    try:
        problem = asyncio.run(check(config, alias))
    except KeyboardInterrupt as exc:
        raise InterruptedWhileChecking(
            "interrupted while the key was being checked \u2014 the config was written to "
            f"{config_path} and the key was not checked; run ncc to try it, or ncc init to "
            "start over"
        ) from exc
    if problem:
        _say(console, f"[yellow]could not verify:[/yellow] {_outside(_redact(problem, key))}")
    else:
        _say(console, "[green]verified[/green] the provider answered")


def _forget_what_was_learned_about_models(console: Console, cache: Path) -> None:
    """Clear the capability cache, so that every model is probed again on its next use.

    A new setup is a reason to ask again what the models can do; a cache that cannot be
    removed is said, with the file, and does not stop the setup.
    """
    try:
        CapabilityCache(cache).clear()
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        _say(
            console,
            f"[yellow]could not remove {_outside(str(cache))}[/yellow] ({_outside(reason)}) "
            "— delete it so that every model is probed again",
        )
