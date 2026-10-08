"""Tests for ``ncc init``: the first sixty seconds.

What these hold the command to, beyond "it writes a file":

* every config it can write is one the loader accepts, so a first run never ends in the
  configuration error that init exists to prevent;
* the key goes to the verification request and nowhere else: not the file, not the screen,
  not an error message, not the prompt's echo, not the environment of the process;
* the file is private from the moment it exists;
* an existing file is replaced only when the person says so.

No test here holds a real key, and none sends a request anywhere but to a transport made in
the test. Every key-shaped string is built when the test runs.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import pty
import secrets
import select
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

from nanoclaude.cli import init as init_module
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.init import (
    PRESETS,
    Preset,
    _redact,
    ask_on_terminal,
    render_config,
    run_init,
    verify,
)
from nanoclaude.cli.main import EXIT_CODES, main, nanoclaude_home
from nanoclaude.config.load import ConfigError, load_config
from nanoclaude.config.schema import ROLES, Config, PermissionsConfig
from nanoclaude.conversation.transcript import Transcript, user_text
from nanoclaude.providers.base import ModelRequest
from nanoclaude.providers.capabilities import (
    CONSERVATIVE_DEFAULT,
    CapabilityCache,
    capabilities_for,
)
from nanoclaude.providers.openai_compat import OpenAICompatClient
from nanoclaude.testing.scripted import says
from tests.cli.helpers import INTERRUPTED, colour_codes, plain_console, run_ncc


def write_home(home: Path, text: str) -> None:
    (home / ".nanoclaude").mkdir(parents=True, exist_ok=True)
    (home / ".nanoclaude" / "config.toml").write_text(text)


def config_path(home: Path) -> Path:
    return home / ".nanoclaude" / "config.toml"


@pytest.fixture
def key() -> str:
    """A string shaped like a key, made up when the test runs: it is nobody's."""
    return "sk-" + secrets.token_urlsafe(24)


def choice_of(name: str) -> str:
    """What a person types to pick the preset ``name``: its place in the menu."""
    return str(list(PRESETS).index(name) + 1)


class Answers:
    """The person, as a script: one answer for each question, in order.

    ``None`` is Enter, which takes the default the question offers. A question nobody
    scripted fails the test and names the question: a prompt that appears when it should
    not is what this is here to catch.
    """

    def __init__(self, *answers: str | None) -> None:
        self._answers = iter(answers)
        #: Every question asked, with the keywords it was asked with.
        self.asked: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, message: str, **kwargs: Any) -> str:
        self.asked.append((message, kwargs))
        try:
            answer = next(self._answers)
        except StopIteration:
            raise AssertionError(f"unscripted question: {message!r} {kwargs}") from None
        return kwargs.get("default", "") if answer is None else answer

    def hidden(self) -> list[str]:
        """The questions asked with the answer kept off the screen."""
        return [message for message, kwargs in self.asked if kwargs.get("password")]


class Racing(Answers):
    """Answers, and something that happens while one question waits for its answer.

    ``do`` runs when ``when`` is asked, before it is answered: the person is at the keyboard
    and another process is not.
    """

    def __init__(self, *answers: str | None, when: str, do: Callable[[], None]) -> None:
        super().__init__(*answers)
        self._when, self._do = when, do

    def __call__(self, message: str, **kwargs: Any) -> str:
        if message == self._when:
            self._do()
        return super().__call__(message, **kwargs)


class Verifier:
    """Stands in for the request that checks the key; says what it was asked, answers as told."""

    def __init__(self, problem: str | None = None) -> None:
        self.problem = problem
        self.calls: list[tuple[Config, str]] = []
        #: Whether the config file was on disk when each request went out.
        self.file_was_there: list[bool] = []
        self._home: Path | None = None

    async def __call__(self, config: Config, alias: str) -> str | None:
        self.calls.append((config, alias))
        self.file_was_there.append(self._home is not None and config_path(self._home).exists())
        return self.problem


@dataclass
class Ran:
    code: int
    out: str
    ask: Answers
    verify: Verifier


def init(
    home: Path,
    *answers: str | None,
    env: Mapping[str, str] | None = None,
    verify: Verifier | None = None,
) -> Ran:
    """``ncc init`` in ``home``, with scripted answers, an empty environment and no network."""
    console, buffer = plain_console(100)
    asking, checking = Answers(*answers), verify or Verifier()
    checking._home = home
    code = run_init(console, home=home, ask=asking, env={} if env is None else env, verify=checking)
    return Ran(code, buffer.getvalue(), asking, checking)


# --------------------------------------------------------------------------
# What a preset renders to
# --------------------------------------------------------------------------


def test_each_preset_is_filed_under_its_own_key():
    assert [preset.key for preset in PRESETS.values()] == list(PRESETS)


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_renders_a_config_the_loader_accepts(tmp_path, name):
    preset = PRESETS[name]
    write_home(tmp_path, render_config(preset, preset.key, preset.default_model))
    config = load_config(home=str(tmp_path), project=None, env={})
    entry = config.models[preset.key]
    assert (entry.adapter, entry.model, entry.base_url) == (
        preset.adapter,
        preset.default_model,
        preset.base_url,
    )
    assert config.api_key_env_for(preset.key) == preset.key_env
    assert {role: config.roles.alias_for(role) for role in ROLES} == dict.fromkeys(
        ROLES, preset.key
    )


@pytest.mark.parametrize("name", list(PRESETS))
def test_the_permissions_it_writes_are_the_ones_the_program_ships(tmp_path, name):
    # A list in the person's file replaces the default, so a list that is shorter than the
    # default takes a tool out of it without anyone having said so.
    preset = PRESETS[name]
    write_home(tmp_path, render_config(preset, preset.key, preset.default_model))
    config = load_config(home=str(tmp_path), project=None, env={})
    shipped = PermissionsConfig()
    assert config.permissions.allow == shipped.allow
    assert config.permissions.ask == shipped.ask
    assert config.permissions.deny == ("Read(**/.env*)",)


def test_the_rendered_config_is_commented():
    rendered = render_config(PRESETS["anthropic"], "sonnet", "claude-sonnet-5")
    assert rendered.lstrip().startswith("#")
    assert "explore" in rendered


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_section_of_the_rendered_config_is_introduced_by_a_comment(name):
    # The file is the documentation: what init does not ask is a default read here.
    preset = PRESETS[name]
    lines = render_config(preset, preset.key, preset.default_model).splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("[")]
    assert len(headers) == 4
    for index in headers:
        before = [line for line in lines[:index] if line.strip()]
        assert before[-1].startswith("#"), f"no comment above {lines[index]}"


def test_the_comment_above_the_roles_says_what_a_role_is_for():
    lines = render_config(PRESETS["anthropic"], "a", "m").splitlines()
    above = lines[: lines.index("[roles]")]
    assert any(line.startswith("#") and "explore" in line for line in above)


def test_the_rendered_config_names_the_variable_and_never_holds_a_key():
    entry = tomllib.loads(render_config(PRESETS["anthropic"], "sonnet", "claude-sonnet-5"))[
        "models"
    ]["sonnet"]
    assert entry["api_key_env"] == "ANTHROPIC_API_KEY"
    assert "api_key" not in entry


def test_a_local_model_is_not_given_a_key_variable():
    preset = PRESETS["ollama"]
    entry = tomllib.loads(render_config(preset, "local", preset.default_model))["models"]["local"]
    assert "api_key_env" not in entry


def test_the_anthropic_preset_offers_the_current_model():
    assert PRESETS["anthropic"].default_model == "claude-sonnet-5-5"


def test_the_model_it_replaced_is_still_usable_by_name(tmp_path):
    # claude-sonnet-5 is legacy, not gone: someone who types it gets it, and the program
    # knows what it can do, so it is not treated as an unknown model.
    write_home(tmp_path, render_config(PRESETS["anthropic"], "anthropic", "claude-sonnet-5"))
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models["anthropic"].model == "claude-sonnet-5"
    assert capabilities_for("anthropic", "claude-sonnet-5").native_tools


@pytest.mark.parametrize(
    "model",
    [
        'x"\n[limits]\nmax_turns = 1\n#',
        "back\\slash",
        'say "hi"',
        "tab\there",
        "bell\x07",
        "del\x7f",
        "[models.evil]",
        "caf\u00e9-\u6a21\u578b",
    ],
)
def test_what_is_typed_as_a_model_is_a_value_and_never_more_toml(tmp_path, model):
    preset = PRESETS["anthropic"]
    rendered = render_config(preset, preset.key, model)
    assert tomllib.loads(rendered)["models"][preset.key]["model"] == model
    write_home(tmp_path, rendered)
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models[preset.key].model == model
    assert config.limits.max_turns == 40
    assert list(config.models) == [preset.key]


@pytest.mark.parametrize("alias", ["my model", "a.b", 'q"uote', "ünï"])
def test_an_alias_that_is_not_a_bare_toml_key_is_quoted(alias):
    preset = PRESETS["anthropic"]
    rendered = render_config(preset, alias, preset.default_model)
    assert list(tomllib.loads(rendered)["models"]) == [alias]


# --------------------------------------------------------------------------
# What it asks, and what it writes
# --------------------------------------------------------------------------


def test_a_written_config_parses_has_every_role_and_loads(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    assert ran.code == 0
    written = tomllib.loads(config_path(tmp_path).read_text())
    assert set(written["roles"]) == set(ROLES)
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.roles.main == "anthropic"


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_can_be_chosen_and_its_config_loads(tmp_path, key, name):
    answers = [choice_of(name)] + ([key] if PRESETS[name].needs_key else []) + [None]
    assert init(tmp_path, *answers).code == 0
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models[name].model == PRESETS[name].default_model


def test_enter_at_the_model_question_takes_the_current_anthropic_model(tmp_path, key):
    init(tmp_path, "1", key, None)
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models["anthropic"].model == "claude-sonnet-5-5"


def test_the_model_it_replaced_can_still_be_asked_for_by_name(tmp_path, key):
    init(tmp_path, "1", key, "claude-sonnet-5")
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models["anthropic"].model == "claude-sonnet-5"


def test_the_model_question_offers_the_presets_model_as_its_default(tmp_path, key):
    ran = init(tmp_path, choice_of("openai"), key, None)
    (message, kwargs) = ran.ask.asked[-1]
    assert kwargs["default"] == "gpt-5", message


def test_a_model_name_with_stray_spaces_is_the_name_without_them(tmp_path, key):
    init(tmp_path, "1", key, "  claude-haiku-4-5 ")
    config = load_config(home=str(tmp_path), project=None, env={})
    assert config.models["anthropic"].model == "claude-haiku-4-5"


def test_the_ollama_preset_asks_for_no_key(tmp_path):
    ran = init(tmp_path, choice_of("ollama"), "qwen3-coder")
    assert ran.code == 0
    assert ran.ask.hidden() == []
    assert len(ran.ask.asked) == 2


def test_the_menu_lists_every_preset_by_the_number_that_picks_it(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    for number, preset in enumerate(PRESETS.values(), start=1):
        assert f"{number}. {preset.label}" in ran.out
    (_, kwargs) = ran.ask.asked[0]
    assert kwargs["choices"] == [str(n) for n in range(1, len(PRESETS) + 1)]
    assert kwargs["default"] == "1"


# --------------------------------------------------------------------------
# The key: it reaches the verification request and nothing else
# --------------------------------------------------------------------------


def test_the_key_is_asked_for_with_the_answer_kept_off_the_screen(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    assert len(ran.ask.hidden()) == 1
    # And only the key: a model name typed in the dark would be a question nobody could check.
    assert [kwargs.get("password", False) for _, kwargs in ran.ask.asked] == [False, True, False]


def test_the_key_is_not_in_the_file_and_not_on_the_screen(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    assert key not in config_path(tmp_path).read_text()
    assert key not in ran.out


def test_the_screen_names_the_variable_and_tells_the_person_where_the_key_goes(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    assert "ANTHROPIC_API_KEY" in ran.out
    assert "export ANTHROPIC_API_KEY=" in ran.out
    assert key not in ran.out


def test_a_key_pasted_with_spaces_or_a_newline_is_the_key_without_them(tmp_path, key):
    ran = init(tmp_path, "1", f"  {key}\n", None)
    (config, alias) = ran.verify.calls[0]
    assert config.api_key_for(alias) == key


def test_the_key_the_check_is_given_is_the_one_that_was_pasted(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    (config, alias) = ran.verify.calls[0]
    assert alias == "anthropic"
    assert config.api_key_for(alias) == key


def test_the_key_is_not_put_in_the_environment_of_this_process(tmp_path, key, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    console, _ = plain_console()
    run_init(console, home=tmp_path, ask=Answers("1", key, None), verify=Verifier())
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert key not in os.environ.values()


def test_a_key_already_in_the_environment_is_used_and_not_asked_for(tmp_path, key):
    ran = init(tmp_path, "1", None, env={"ANTHROPIC_API_KEY": key})
    assert ran.ask.hidden() == []
    (config, alias) = ran.verify.calls[0]
    assert config.api_key_for(alias) == key
    assert "ANTHROPIC_API_KEY is already set" in ran.out
    # Nothing to add to a profile: it is there already. And nothing of it is shown.
    assert "export" not in ran.out
    assert key not in ran.out
    assert key not in config_path(tmp_path).read_text()


def test_a_key_in_the_environment_for_another_provider_is_not_used(tmp_path, key):
    ran = init(tmp_path, choice_of("openai"), key, None, env={"ANTHROPIC_API_KEY": "other"})
    assert len(ran.ask.hidden()) == 1
    (config, alias) = ran.verify.calls[0]
    assert config.api_key_for(alias) == key


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------


def test_the_config_is_written_under_the_home_it_was_given(tmp_path, key):
    init(tmp_path, "1", key, None)
    assert config_path(tmp_path).is_file()


def test_the_config_is_not_world_readable_whatever_the_umask(tmp_path, key):
    before = os.umask(0)  # with no mask, any file made without a private mode is open to all
    try:
        init(tmp_path, "1", key, None)
    finally:
        os.umask(before)
    assert config_path(tmp_path).stat().st_mode & 0o777 == 0o600


def removable(root: Path) -> None:
    """Make everything under ``root`` ours to remove again, whatever a test left it as.

    A test that makes the umask hostile and finds a regression leaves directories nobody can
    enter, and pytest then fails to clean them up in every session after it.
    """
    pending = [root] if root.exists() else []
    while pending:
        path = pending.pop()
        if path.is_symlink():
            continue
        path.chmod(0o700)  # before it is listed: a directory with no mode cannot be
        if path.is_dir():
            pending.extend(path.iterdir())


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o277, 0o777], ids=lambda m: f"umask{m:04o}")
def test_the_config_is_0600_whatever_the_umask(tmp_path, key, mask):
    # The mode given to os.open is filtered by the umask, so that a umask of 0277 makes the
    # file 0400 and 0777 makes it 0000: not private, but unreadable, and a config nobody can
    # read is the error init exists to prevent. The mode is set on the descriptor too.
    before = os.umask(mask)
    try:
        ran = init(tmp_path, "1", key, None)
        observed = config_path(tmp_path).stat().st_mode & 0o777
    finally:
        os.umask(before)
        removable(tmp_path / ".nanoclaude")
    assert ran.code == 0
    assert observed == 0o600
    assert [p.name for p in (tmp_path / ".nanoclaude").iterdir()] == ["config.toml"]


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o177, 0o277, 0o777], ids=lambda m: f"umask{m:04o}")
def test_a_directory_init_makes_is_private_and_usable_whatever_the_umask(tmp_path, key, mask):
    # mkdir filters its mode by the umask too: with 0177 or tighter (any bit of 0300) the new
    # directory would lose its owner's write or search bit, and nothing could be made in it.
    home = tmp_path / "a" / "b"
    before = os.umask(mask)
    try:
        ran = init(home, "1", key, None)
        modes = {
            made: made.stat().st_mode & 0o777
            for made in (tmp_path / "a", home, home / ".nanoclaude")
        }
    finally:
        os.umask(before)
        removable(tmp_path / "a")
    assert ran.code == 0
    assert modes == dict.fromkeys(modes, 0o700)
    assert config_path(home).is_file()


def test_a_dangling_link_where_the_directory_goes_is_an_error_and_is_not_made_into_one(
    tmp_path, key
):
    (tmp_path / ".nanoclaude").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ConfigError, match=r"cannot write .*config\.toml"):
        run_init(
            plain_console()[0],
            home=tmp_path,
            ask=Answers("1", key, None),
            env={},
            verify=Verifier(),
        )
    assert (tmp_path / ".nanoclaude").is_symlink()
    assert not (tmp_path / "nowhere").exists()


def test_a_directory_that_was_already_there_is_left_as_it_is(tmp_path, key):
    tmp_path.chmod(0o755)
    (tmp_path / ".nanoclaude").mkdir(mode=0o750)
    (tmp_path / ".nanoclaude").chmod(0o750)
    init(tmp_path, "1", key, None)
    assert tmp_path.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / ".nanoclaude").stat().st_mode & 0o777 == 0o750


def test_the_config_is_private_from_the_moment_it_exists(tmp_path, key, monkeypatch):
    # Not written first and made private after: between the two, any other user could read it.
    # So every file made under the home is made with the private mode, by the call that makes it.
    created: list[tuple[Path, int]] = []
    real_open = os.open

    def watching(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        if flags & os.O_CREAT:
            created.append((Path(os.fsdecode(path)), mode))
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", watching)
    before = os.umask(0)
    try:
        init(tmp_path, "1", key, None)
    finally:
        os.umask(before)
    made = [(path, oct(mode)) for path, mode in created if tmp_path in path.parents]
    assert made, "no file was made through os.open, so nothing says how it was made"
    assert {mode for _, mode in made} == {oct(0o600)}, made


def test_nothing_but_the_config_is_left_in_the_directory(tmp_path, key):
    init(tmp_path, "1", key, None)
    assert [p.name for p in (tmp_path / ".nanoclaude").iterdir()] == ["config.toml"]


def test_the_directory_is_made_when_it_is_not_there(tmp_path, key):
    home = tmp_path / "a" / "b"
    home.mkdir(parents=True)
    assert init(home, "1", key, None).code == 0
    assert config_path(home).is_file()


def test_a_config_that_cannot_be_written_is_an_error_that_names_the_file(tmp_path, key):
    home = tmp_path / "home"
    home.write_text("a file where the directory should go")
    console, _ = plain_console()
    with pytest.raises(ConfigError, match=r"cannot write .*config\.toml") as caught:
        run_init(console, home=home, ask=Answers("1", key, None), env={}, verify=Verifier())
    assert key not in str(caught.value)


def test_a_write_that_fails_leaves_nothing_behind_and_does_not_touch_what_was_there(tmp_path, key):
    # A directory where the file goes: the new config is made, and then cannot take its place.
    target = config_path(tmp_path)
    target.mkdir(parents=True)
    (target / "keep").write_text("mine")
    console, _ = plain_console()
    with pytest.raises(ConfigError, match=r"cannot write .*config\.toml"):
        run_init(
            console, home=tmp_path, ask=Answers("y", "1", key, None), env={}, verify=Verifier()
        )
    assert [p.name for p in (tmp_path / ".nanoclaude").iterdir()] == ["config.toml"]
    assert (target / "keep").read_text() == "mine"


# --------------------------------------------------------------------------
# An existing config
# --------------------------------------------------------------------------


def test_an_existing_config_is_not_overwritten_without_consent(tmp_path):
    write_home(tmp_path, "# mine\n")
    ran = init(tmp_path, "n")
    assert config_path(tmp_path).read_text() == "# mine\n"
    assert ran.code == 0
    assert len(ran.ask.asked) == 1, "nothing is asked after a no"
    assert ran.verify.calls == []
    assert "left alone" in ran.out


def test_a_config_that_is_a_link_is_said_to_be_one_with_where_it_goes_and_what_a_yes_does(tmp_path):
    dotfile = tmp_path / "dotfiles" / "config.toml"
    dotfile.parent.mkdir()
    dotfile.write_text("# mine\n")
    (tmp_path / ".nanoclaude").mkdir()
    config_path(tmp_path).symlink_to(dotfile)
    ran = init(tmp_path, "n")
    line = " ".join(ran.out.split())
    assert f"{config_path(tmp_path)} is a link to {dotfile}." in line
    assert "Answering y replaces the link with a regular file" in line
    assert f"{dotfile} is left as it is" in line
    assert config_path(tmp_path).is_symlink() and dotfile.read_text() == "# mine\n"


def test_a_yes_replaces_the_link_and_leaves_what_it_led_to_alone(tmp_path, key):
    dotfile = tmp_path / "dotfiles" / "config.toml"
    dotfile.parent.mkdir()
    dotfile.write_text("# mine\n")
    (tmp_path / ".nanoclaude").mkdir()
    config_path(tmp_path).symlink_to(dotfile)
    init(tmp_path, "y", "1", key, None)
    assert not config_path(tmp_path).is_symlink() and config_path(tmp_path).is_file()
    assert dotfile.read_text() == "# mine\n"
    assert config_path(tmp_path).stat().st_mode & 0o777 == 0o600


def test_a_link_is_said_as_it_was_made_and_never_as_markup(tmp_path):
    (tmp_path / ".nanoclaude").mkdir()
    config_path(tmp_path).symlink_to("../[red]dot[bold]/config.toml")
    ran = init(tmp_path, "n")
    assert "is a link to ../[red]dot[bold]/config.toml." in " ".join(ran.out.split())


def test_a_link_that_leads_nowhere_is_a_config_that_is_there_and_is_asked_about_first(tmp_path):
    (tmp_path / ".nanoclaude").mkdir()
    config_path(tmp_path).symlink_to(tmp_path / "nowhere")
    ran = init(tmp_path, "n")
    assert [m for m, _ in ran.ask.asked] == ["overwrite?"], "before anything else is asked"
    assert f"is a link to {tmp_path / 'nowhere'}." in " ".join(ran.out.split())
    assert config_path(tmp_path).is_symlink() and not (tmp_path / "nowhere").exists()


def test_a_link_that_cannot_be_read_is_still_asked_about_and_not_a_crash(tmp_path, monkeypatch):
    # It was a link when it was looked at, and is not when it is read: somebody is at work.
    (tmp_path / ".nanoclaude").mkdir()
    config_path(tmp_path).symlink_to(tmp_path / "nowhere")

    def vanished(self: Path) -> Path:
        raise FileNotFoundError(self)

    monkeypatch.setattr(Path, "readlink", vanished)
    ran = init(tmp_path, "n")
    assert f"{config_path(tmp_path)} already exists." in " ".join(ran.out.split())
    assert [m for m, _ in ran.ask.asked] == ["overwrite?"]


def test_a_config_that_is_not_a_link_is_not_called_one(tmp_path):
    write_home(tmp_path, "# mine\n")
    assert "is a link" not in init(tmp_path, "n").out


def test_an_existing_config_is_asked_about_with_no_as_the_answer_enter_gives(tmp_path):
    write_home(tmp_path, "# mine\n")
    ran = init(tmp_path, None)
    (_, kwargs) = ran.ask.asked[0]
    assert kwargs["default"] == "n"
    assert kwargs["choices"] == ["y", "n"]
    assert config_path(tmp_path).read_text() == "# mine\n"


def test_an_existing_config_is_replaced_when_the_person_says_yes(tmp_path, key):
    write_home(tmp_path, "# mine\n")
    ran = init(tmp_path, "y", "1", key, None)
    assert ran.code == 0
    assert "# mine" not in config_path(tmp_path).read_text()
    assert load_config(home=str(tmp_path), project=None, env={}).roles.main == "anthropic"


def test_a_replaced_config_is_private_even_if_the_old_one_was_not(tmp_path, key):
    write_home(tmp_path, "# mine\n")
    config_path(tmp_path).chmod(0o644)
    init(tmp_path, "y", "1", key, None)
    assert config_path(tmp_path).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("answer", ["", "yes", "Y", "no", "sure"])
def test_only_a_plain_y_is_consent(tmp_path, answer):
    # What reaches here from a real prompt is one of the choices; this is a script that
    # does not hold to them, and an answer that is not y is not a yes.
    write_home(tmp_path, "# mine\n")
    init(tmp_path, answer)
    assert config_path(tmp_path).read_text() == "# mine\n"


# A file that was not there when the questions began, and is there when the answer is written:
# somebody else's work, made while the person was choosing a model. The answer to "is there a
# config?" was no, and that is not consent to replace one.


def appears(home: Path, text: str = "# somebody else's\n") -> Callable[[], None]:
    return lambda: write_home(home, text)


def test_a_config_that_appears_while_the_questions_are_asked_is_not_replaced_unasked(tmp_path, key):
    console, buffer = plain_console(100)
    asking = Racing("1", key, None, "n", when="model", do=appears(tmp_path))
    checking = Verifier()
    code = run_init(console, home=tmp_path, ask=asking, env={}, verify=checking)
    assert code == 0
    assert config_path(tmp_path).read_text() == "# somebody else's\n"
    assert checking.calls == [], (
        "and nothing was checked, or cleared, for a setup that did not happen"
    )
    assert "left alone" in buffer.getvalue()
    assert [p.name for p in (tmp_path / ".nanoclaude").iterdir()] == ["config.toml"]


def test_the_question_about_a_config_that_appeared_is_the_one_asked_about_any_other(tmp_path, key):
    asking = Racing("1", key, None, "n", when="model", do=appears(tmp_path))
    run_init(plain_console()[0], home=tmp_path, ask=asking, env={}, verify=Verifier())
    (message, kwargs) = asking.asked[-1]
    assert (message, kwargs["choices"], kwargs["default"]) == ("overwrite?", ["y", "n"], "n")
    assert len(asking.asked) == 4


def test_a_config_that_appeared_is_replaced_when_the_person_then_says_yes(tmp_path, key):
    cache = seeded_cache(tmp_path)
    asking = Racing("1", key, None, "y", when="model", do=appears(tmp_path))
    checking = Verifier()
    code = run_init(plain_console()[0], home=tmp_path, ask=asking, env={}, verify=checking)
    assert code == 0
    assert load_config(home=str(tmp_path), project=None, env={}).roles.main == "anthropic"
    assert config_path(tmp_path).stat().st_mode & 0o777 == 0o600
    assert len(checking.calls) == 1
    assert cache.get("ollama", "qwen3-coder") is None
    assert [p.name for p in (tmp_path / ".nanoclaude").iterdir()] == ["config.toml"]


def test_declining_the_config_that_appeared_leaves_the_cache_as_it_was(tmp_path, key):
    cache = seeded_cache(tmp_path)
    asking = Racing("1", key, None, "n", when="model", do=appears(tmp_path))
    run_init(plain_console()[0], home=tmp_path, ask=asking, env={}, verify=Verifier())
    assert cache.get("ollama", "qwen3-coder") == CONSERVATIVE_DEFAULT


def test_a_dangling_link_that_appears_is_a_config_that_appeared(tmp_path, key):
    def link() -> None:
        (tmp_path / ".nanoclaude").mkdir(exist_ok=True)
        config_path(tmp_path).symlink_to(tmp_path / "nowhere")

    asking = Racing("1", key, None, "n", when="model", do=link)
    console, buffer = plain_console(200)
    run_init(console, home=tmp_path, ask=asking, env={}, verify=Verifier())
    assert config_path(tmp_path).is_symlink() and not config_path(tmp_path).exists()
    assert len(asking.asked) == 4
    assert f"is a link to {tmp_path / 'nowhere'}." in " ".join(buffer.getvalue().split())


def test_a_config_that_was_agreed_to_is_replaced_without_a_second_question(tmp_path, key):
    write_home(tmp_path, "# mine\n")
    ran = init(tmp_path, "y", "1", key, None)
    assert [m for m, _ in ran.ask.asked].count("overwrite?") == 1


def test_a_file_planted_at_the_temporary_name_is_refused_and_nothing_goes_through_it(tmp_path, key):
    victim = tmp_path / "victim"
    victim.write_text("precious")
    (tmp_path / ".nanoclaude").mkdir()
    (tmp_path / ".nanoclaude" / f".config.toml.{os.getpid()}.tmp").symlink_to(victim)
    with pytest.raises(ConfigError, match=r"cannot write .*config\.toml"):
        run_init(
            plain_console()[0],
            home=tmp_path,
            ask=Answers("1", key, None),
            env={},
            verify=Verifier(),
        )
    assert victim.read_text() == "precious"
    assert not config_path(tmp_path).exists()


@pytest.mark.parametrize("agreed", [False, True])
def test_an_interrupt_in_the_middle_of_the_write_leaves_no_temporary_file(
    tmp_path, key, monkeypatch, agreed
):
    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace" if agreed else "link", interrupted)
    answers = ["y", "1", key, None] if agreed else ["1", key, None]
    if agreed:
        write_home(tmp_path, "# mine\n")
    with pytest.raises(KeyboardInterrupt):
        run_init(
            plain_console()[0], home=tmp_path, ask=Answers(*answers), env={}, verify=Verifier()
        )
    left = sorted(p.name for p in (tmp_path / ".nanoclaude").iterdir())
    assert left == (["config.toml"] if agreed else [])
    if agreed:
        assert config_path(tmp_path).read_text() == "# mine\n"


# --------------------------------------------------------------------------
# What ncc has learned about models is forgotten when the setup is redone
# --------------------------------------------------------------------------


def seeded_cache(home: Path) -> CapabilityCache:
    cache = CapabilityCache(home / ".nanoclaude" / "capabilities.json")
    cache.put("ollama", "qwen3-coder", CONSERVATIVE_DEFAULT)
    return cache


def test_running_init_forgets_the_cached_capabilities_so_every_model_is_probed_again(tmp_path, key):
    cache = seeded_cache(tmp_path)
    assert cache.get("ollama", "qwen3-coder") is not None
    init(tmp_path, "1", key, None)
    assert cache.get("ollama", "qwen3-coder") is None
    assert not (tmp_path / ".nanoclaude" / "capabilities.json").exists()


def test_a_write_that_fails_leaves_what_was_learned_as_it_was(tmp_path, key):
    # Forgotten when the setup is redone, and a setup that did not happen is not one.
    cache = seeded_cache(tmp_path)
    config_path(tmp_path).mkdir()  # a directory where the file goes: the write cannot finish
    with pytest.raises(ConfigError):
        run_init(
            plain_console()[0],
            home=tmp_path,
            ask=Answers("y", "1", key, None),
            env={},
            verify=Verifier(),
        )
    assert cache.get("ollama", "qwen3-coder") == CONSERVATIVE_DEFAULT


def test_declining_to_replace_the_config_leaves_the_cache_as_it_was(tmp_path):
    cache = seeded_cache(tmp_path)
    write_home(tmp_path, "# mine\n")
    init(tmp_path, "n")
    assert cache.get("ollama", "qwen3-coder") == CONSERVATIVE_DEFAULT


def test_a_cache_that_cannot_be_cleared_is_said_and_does_not_stop_the_setup(tmp_path, key):
    (tmp_path / ".nanoclaude").mkdir()
    (tmp_path / ".nanoclaude" / "capabilities.json").mkdir()
    ran = init(tmp_path, "1", key, None)
    assert ran.code == 0
    assert config_path(tmp_path).is_file()
    assert "capabilities.json" in ran.out
    assert "probed again" in ran.out


def test_the_reason_a_cache_cannot_be_removed_is_not_read_as_markup(tmp_path, key, monkeypatch):
    def refuse(self: CapabilityCache) -> None:
        raise OSError(5, "no [/] space [bold]left")

    monkeypatch.setattr(CapabilityCache, "clear", refuse)
    ran = init(tmp_path, "1", key, None)
    assert "(no [/] space [bold]left)" in ran.out


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def test_the_key_is_checked_after_the_config_is_on_disk_with_what_was_written(tmp_path, key):
    ran = init(tmp_path, "1", key, "claude-haiku-4-5")
    assert ran.verify.file_was_there == [True]
    (config, alias) = ran.verify.calls[0]
    assert config.models[alias].model == "claude-haiku-4-5"
    assert "verified" in ran.out


def test_a_failed_check_is_reported_and_the_file_is_still_written(tmp_path, key):
    ran = init(tmp_path, "1", key, None, verify=Verifier("the provider rejected your credentials"))
    assert ran.code == 0
    assert "could not verify" in ran.out
    assert "rejected" in ran.out
    assert "verified the provider" not in ran.out
    assert config_path(tmp_path).exists()


def test_what_the_provider_says_is_shown_as_it_was_and_is_not_read_as_markup(tmp_path, key):
    problem = "no such model [/] [bold]x[/bold] [red"
    ran = init(tmp_path, "1", key, None, verify=Verifier(problem))
    assert problem in ran.out


def test_the_path_is_shown_as_it_is_and_is_not_read_as_markup(tmp_path, key):
    home = tmp_path / "[red]home[bold]"
    home.mkdir()
    ran = init(home, "1", key, None)
    assert f"wrote {config_path(home)}" in ran.out


def test_the_path_of_an_existing_config_is_not_read_as_markup_either(tmp_path):
    home = tmp_path / "[red]home[bold]"
    write_home(home, "# mine\n")
    ran = init(home, "n")
    assert f"{config_path(home)} already exists" in ran.out


def test_what_the_provider_says_cannot_move_the_cursor_or_start_a_line(tmp_path, key):
    problem = "bad\x1b[2J\x1b]0;owned\x07 news\nerror: second line"
    ran = init(tmp_path, "1", key, None, verify=Verifier(problem))
    assert "\x1b" not in ran.out and "\x07" not in ran.out
    (line,) = [line for line in ran.out.splitlines() if "could not verify" in line]
    assert line.endswith("bad news error: second line")


def test_the_key_is_not_shown_even_when_the_provider_says_it_back(tmp_path, key):
    problem = f"Incorrect API key provided: {key}. You can find your key at https://example.test"
    ran = init(tmp_path, "1", key, None, verify=Verifier(problem))
    assert "Incorrect API key provided" in ran.out
    assert key not in ran.out


def test_a_provider_that_says_the_key_back_in_pieces_of_a_sentence_is_still_not_shown(
    tmp_path, key
):
    problem = f"bad key {key}{key} and again {key}"
    ran = init(tmp_path, "1", key, None, verify=Verifier(problem))
    assert key not in ran.out


def test_the_check_is_not_asked_to_run_when_the_person_said_no(tmp_path):
    write_home(tmp_path, "# mine\n")
    assert init(tmp_path, "n").verify.calls == []


def test_it_ends_by_saying_how_to_start(tmp_path, key):
    ran = init(tmp_path, "1", key, None)
    assert "ncc -p" in ran.out


# --------------------------------------------------------------------------
# The check itself: one small request, through the client ncc would use
# --------------------------------------------------------------------------


def config_for(home: Path, name: str, key: str | None, model: str | None = None) -> Config:
    """The config ``ncc init`` writes for a preset, loaded as ncc loads it."""
    preset = PRESETS[name]
    write_home(home, render_config(preset, name, model or preset.default_model))
    env = {preset.key_env: key} if key and preset.key_env else {}
    return load_config(home=str(home), project=None, env=env)


Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class Wire:
    """What went out, and how to answer it: every request ``verify`` makes ends up here."""

    requests: list[httpx.Request]
    clients: list[httpx.AsyncClient]
    answer: Handler

    def body(self) -> dict[str, Any]:
        (request,) = self.requests
        sent: dict[str, Any] = json.loads(request.content)
        return sent


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    """Put a transport made here under every HTTP client the providers build.

    Nothing leaves the process: a request that is made is a request this test can read.
    """
    sent = Wire([], [], lambda _request: httpx.Response(500))
    real = httpx.AsyncClient

    def transport(request: httpx.Request) -> httpx.Response:
        sent.requests.append(request)
        return sent.answer(request)

    def make(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        client = real(*args, **{**kwargs, "transport": httpx.MockTransport(transport)})
        sent.clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make)
    return sent


def sse(*events: tuple[str, dict[str, Any]]) -> httpx.Response:
    text = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})


def anthropic_reply(*blocks: tuple[str, dict[str, Any]], stop: str = "end_turn") -> httpx.Response:
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"message": {"model": "m", "usage": {"input_tokens": 9}}})
    ]
    for index, (kind, delta) in enumerate(blocks):
        events += [
            ("content_block_start", {"index": index, "content_block": {"type": kind}}),
            ("content_block_delta", {"index": index, "delta": delta}),
            ("content_block_stop", {"index": index}),
        ]
    events += [
        ("message_delta", {"delta": {"stop_reason": stop}, "usage": {"output_tokens": 8}}),
        ("message_stop", {}),
    ]
    return sse(*events)


def chat_reply(*chunks: dict[str, Any]) -> httpx.Response:
    text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})


OK_ANTHROPIC = anthropic_reply(("text", {"type": "text_delta", "text": "ok"}))


def says_ok(request: httpx.Request) -> httpx.Response:
    if "api/chat" in str(request.url):
        return httpx.Response(
            200,
            content=b'{"message": {"role": "assistant", "content": "ok"}, "done": true}\n',
        )
    if "chat/completions" in str(request.url):
        return chat_reply(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )
    return anthropic_reply(("text", {"type": "text_delta", "text": "ok"}))


async def test_a_provider_that_answers_verifies_the_key(tmp_path, key, wire):
    wire.answer = says_ok
    assert await verify(config_for(tmp_path, "anthropic", key), "anthropic") is None
    (request,) = wire.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"


async def test_the_key_goes_to_the_provider_in_the_header_and_nowhere_else(tmp_path, key, wire):
    wire.answer = says_ok
    await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    (request,) = wire.requests
    assert request.headers["x-api-key"] == key
    assert key not in str(request.url)
    assert key.encode() not in request.content


async def test_the_request_is_small_and_names_the_model_and_no_temperature(tmp_path, key, wire):
    wire.answer = says_ok
    await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    body = wire.body()
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 8
    assert "temperature" not in body  # current Claude models refuse any other value
    assert "tools" not in body
    assert len(body["messages"]) == 1


async def test_the_model_checked_is_the_one_the_alias_names(tmp_path, key, wire):
    wire.answer = says_ok
    config = config_for(tmp_path, "anthropic", key, "claude-haiku-4-5")
    await verify(config, "anthropic")
    assert wire.body()["model"] == "claude-haiku-4-5"


async def test_the_model_checked_is_the_one_asked_for_even_when_it_is_not_the_main_one(
    tmp_path, key, wire
):
    write_home(
        tmp_path,
        '[models.big]\nadapter = "anthropic"\nmodel = "claude-sonnet-5-5"\n'
        '[models.small]\nadapter = "anthropic"\nmodel = "claude-haiku-4-5"\n'
        '[roles]\nmain = "big"\n',
    )
    config = load_config(home=str(tmp_path), project=None, env={"ANTHROPIC_API_KEY": key})
    wire.answer = says_ok
    assert await verify(config, "small") is None
    assert wire.body()["model"] == "claude-haiku-4-5"


async def test_a_reply_that_stops_at_the_cap_with_only_thinking_in_it_still_verifies(
    tmp_path, key, wire
):
    # A thinking model spends eight tokens thinking and is cut off. That is a 200: the key,
    # the model and the request were all accepted, which is all this reads.
    wire.answer = lambda _request: anthropic_reply(
        ("thinking", {"type": "thinking_delta", "thinking": "let me"}), stop="max_tokens"
    )
    assert await verify(config_for(tmp_path, "anthropic", key), "anthropic") is None


async def test_openai_is_sent_the_parameter_its_own_host_wants(tmp_path, key, wire):
    wire.answer = says_ok
    config = config_for(tmp_path, "openai", key)
    assert await verify(config, "openai") is None
    (request,) = wire.requests
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {key}"
    body = wire.body()
    assert body["max_completion_tokens"] == 8
    assert "max_tokens" not in body and "temperature" not in body


async def test_another_openai_compatible_host_is_sent_max_tokens(tmp_path, key, wire):
    wire.answer = says_ok
    await verify(config_for(tmp_path, "openrouter", key), "openrouter")
    (request,) = wire.requests
    assert request.url.host == "openrouter.ai"
    assert request.headers["authorization"] == f"Bearer {key}"
    assert wire.body()["max_tokens"] == 8


async def test_a_local_model_is_asked_with_no_key_at_all(tmp_path, wire):
    wire.answer = says_ok
    assert await verify(config_for(tmp_path, "ollama", None), "ollama") is None
    (request,) = wire.requests
    assert request.url.host == "localhost"
    assert "authorization" not in request.headers and "x-api-key" not in request.headers


async def test_a_reply_cut_at_the_cap_before_any_text_verifies_on_the_openai_wire_too(
    tmp_path, key, wire
):
    # What a reasoning model sends when its eight tokens went on thinking nobody can see:
    # a finished stream whose only content is the reason it stopped.
    wire.answer = lambda _request: chat_reply(
        {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    )
    assert await verify(config_for(tmp_path, "openai", key), "openai") is None


async def test_a_local_thinking_model_cut_at_the_cap_before_any_text_verifies_too(tmp_path, wire):
    body = b'{"message": {"role": "assistant", "content": "", "thinking": "hm"}, "done": true}\n'
    wire.answer = lambda _request: httpx.Response(200, content=body)
    assert await verify(config_for(tmp_path, "ollama", None), "ollama") is None


# What answers where a provider should, and is not one: a captive portal's page, a proxy with
# nothing behind it, a reply that was not streamed. A 200 proves something answered, not that
# the key was accepted; "verified" is for a reply stream.
NOT_REPLIES = {
    "a captive portal's page": lambda _request: httpx.Response(
        200, content=b"<html>Sign in to the network</html>", headers={"content-type": "text/html"}
    ),
    "an empty body": lambda _request: httpx.Response(200, content=b""),
    "a whole reply that was not streamed": lambda _request: httpx.Response(
        200, json={"choices": [{"message": {"content": "ok"}}], "content": [{"text": "ok"}]}
    ),
}
WIRES = {"anthropic": "anthropic", "openai": "openai", "ollama": "ollama"}


@pytest.mark.parametrize("answer", NOT_REPLIES)
@pytest.mark.parametrize("name", WIRES)
async def test_a_200_that_is_not_a_reply_stream_does_not_verify_a_key(
    tmp_path, key, wire, name, answer
):
    wire.answer = NOT_REPLIES[answer]
    problem = await verify(config_for(tmp_path, name, None if name == "ollama" else key), name)
    assert problem is not None and "not a reply stream" in problem


@pytest.mark.parametrize("name", WIRES)
async def test_a_redirect_does_not_verify_a_key(tmp_path, key, wire, name):
    wire.answer = lambda _request: httpx.Response(
        302, headers={"location": "https://portal.example/login"}, content=b"<html></html>"
    )
    problem = await verify(config_for(tmp_path, name, None if name == "ollama" else key), name)
    assert problem is not None and "redirect (HTTP 302)" in problem


@pytest.mark.parametrize("name", WIRES)
def test_a_captive_portal_page_is_reported_as_could_not_verify(tmp_path, key, wire, name):
    wire.answer = NOT_REPLIES["a captive portal's page"]
    answers = [choice_of(name)] + ([key] if PRESETS[name].needs_key else []) + [None]
    console, buffer = plain_console(120)
    assert run_init(console, home=tmp_path, ask=Answers(*answers), env={}) == 0
    assert "could not verify" in buffer.getvalue()
    assert "verified the provider" not in buffer.getvalue()
    assert config_path(tmp_path).is_file()


async def test_a_rejected_key_is_reported_in_the_providers_words(tmp_path, key, wire):
    wire.answer = lambda _request: httpx.Response(
        401, json={"error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    )
    problem = await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    assert problem is not None
    assert "rejected your credentials" in problem and "invalid x-api-key" in problem


async def test_a_rejected_key_the_provider_says_back_is_not_in_what_is_returned(
    tmp_path, key, wire
):
    wire.answer = lambda _request: httpx.Response(
        401, json={"error": {"message": f"Incorrect API key provided: {key}."}}
    )
    problem = await verify(config_for(tmp_path, "openai", key), "openai")
    assert problem is not None
    assert "Incorrect API key provided" in problem
    assert key not in problem


async def test_a_model_the_provider_does_not_know_is_reported(tmp_path, key, wire):
    wire.answer = lambda _request: httpx.Response(404, json={"error": {"message": "no such model"}})
    problem = await verify(config_for(tmp_path, "openai", key), "openai")
    assert problem is not None and "no such model" in problem


async def test_a_provider_that_cannot_be_reached_is_reported(tmp_path, wire):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    wire.answer = refuse
    problem = await verify(config_for(tmp_path, "ollama", None), "ollama")
    assert problem is not None and "could not reach" in problem and "refused" in problem


async def test_an_exception_nobody_planned_for_is_a_message_and_not_a_traceback(
    tmp_path, key, wire
):
    def explode(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    wire.answer = explode
    problem = await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    assert problem == "RuntimeError: boom"


async def test_an_exception_that_holds_the_key_is_not_returned_with_it(tmp_path, key, wire):
    # A library can put a header value into the message of what it raises.
    def explode(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"Illegal header value {key!r}")

    wire.answer = explode
    problem = await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    assert problem is not None and "Illegal header value" in problem
    assert key not in problem


async def test_a_missing_key_is_reported_without_a_request_being_made(tmp_path, wire):
    problem = await verify(config_for(tmp_path, "anthropic", None), "anthropic")
    assert problem is not None and "ANTHROPIC_API_KEY" in problem
    assert wire.requests == []


async def test_a_provider_that_never_answers_is_given_up_on(tmp_path, key, wire, monkeypatch):
    monkeypatch.setattr(init_module, "VERIFY_TIMEOUT_S", 0.05)

    async def never(_request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    wire.answer = never
    problem = await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    assert problem is not None and "no answer" in problem


@pytest.mark.parametrize("fails", [False, True])
async def test_the_client_is_closed_however_the_check_ends(tmp_path, key, wire, fails):
    wire.answer = (lambda _request: httpx.Response(401, json={})) if fails else says_ok
    await verify(config_for(tmp_path, "anthropic", key), "anthropic")
    assert wire.clients and all(client.is_closed for client in wire.clients)


# --------------------------------------------------------------------------
# The whole of it: the questions, the file, and the real request to a stand-in provider
# --------------------------------------------------------------------------


def test_init_checks_the_key_with_the_real_request_when_nothing_else_is_given(tmp_path, key, wire):
    wire.answer = says_ok
    console, buffer = plain_console(100)
    code = run_init(console, home=tmp_path, ask=Answers("1", key, None), env={})
    assert code == 0
    assert "verified" in buffer.getvalue()
    assert wire.requests[0].headers["x-api-key"] == key
    assert key not in buffer.getvalue()
    assert key not in config_path(tmp_path).read_text()


def test_a_key_the_provider_refuses_is_reported_and_the_file_stays(tmp_path, key, wire):
    wire.answer = lambda _request: httpx.Response(
        401, json={"error": {"message": f"bad key {key}"}}
    )
    console, buffer = plain_console(100)
    code = run_init(console, home=tmp_path, ask=Answers("1", key, None), env={})
    assert code == 0
    assert (
        "could not verify" in buffer.getvalue() and "rejected your credentials" in buffer.getvalue()
    )
    assert key not in buffer.getvalue()
    assert config_path(tmp_path).is_file()


@pytest.mark.parametrize("blank", ["", "   ", " \t "])
def test_a_blank_key_is_asked_for_again_and_never_sent_to_the_check(tmp_path, key, blank):
    ran = init(tmp_path, "1", blank, "", key, None)
    keys = [(m, kw) for m, kw in ran.ask.asked if kw.get("password")]
    assert len(keys) == 3 and all(kw == {"password": True} for _, kw in keys)
    assert "a key is needed" in ran.out
    assert ran.out.count("a key is needed") == 2
    (config, alias) = ran.verify.calls[0]
    assert len(ran.verify.calls) == 1 and config.api_key_for(alias) == key
    assert "could not verify" not in ran.out


def test_the_key_is_asked_for_again_before_the_model_is_asked_about(tmp_path, key):
    ran = init(tmp_path, "1", "", key, None)
    assert [bool(kw.get("password")) for _, kw in ran.ask.asked] == [False, True, True, False]


def test_nothing_is_written_while_the_key_is_still_missing(tmp_path):
    # Ctrl+C at the key question: the person gave up, and there is nothing to show for it.
    def gives_up(message: str, **kwargs: Any) -> str:
        if kwargs.get("password"):
            raise KeyboardInterrupt
        return "1"

    with pytest.raises(KeyboardInterrupt):
        run_init(plain_console()[0], home=tmp_path, ask=gives_up, env={}, verify=Verifier())
    assert not (tmp_path / ".nanoclaude").exists()


# --------------------------------------------------------------------------
# The questions as a terminal asks them
# --------------------------------------------------------------------------


@dataclass
class Terminal:
    """A terminal for one conversation: the keys the person presses, and what the screen shows."""

    keys: PipeInput
    screen: io.StringIO

    def ask(self, typed: str, message: str = "question", **kwargs: Any) -> str:
        """Press ``typed``, as the answer to a question, and return the answer ``ask`` gives."""
        self.keys.send_text(typed)
        return ask_on_terminal(
            message, input=self.keys, output=PlainTextOutput(self.screen), **kwargs
        )


@pytest.fixture
def terminal() -> Iterator[Terminal]:
    with create_pipe_input() as keys:
        yield Terminal(keys, io.StringIO())


ENTER = "\r"


def test_what_is_typed_is_the_answer(terminal):
    assert terminal.ask(f"claude-haiku-4-5{ENTER}") == "claude-haiku-4-5"


def test_spaces_around_the_answer_are_not_part_of_it(terminal):
    assert terminal.ask(f"  x  {ENTER}") == "x"


def test_enter_alone_gives_the_default_and_the_default_is_shown(terminal):
    assert terminal.ask(ENTER, "model", default="gpt-5") == "gpt-5"
    assert "model (gpt-5):" in terminal.screen.getvalue()


def test_a_question_with_no_default_is_not_answered_by_enter(terminal):
    assert terminal.ask(ENTER) == ""


def test_the_choices_are_shown_and_one_of_them_is_the_answer(terminal):
    assert terminal.ask(f"2{ENTER}", "choice", choices=["1", "2", "3"], default="1") == "2"
    assert "choice [1/2/3] (1):" in terminal.screen.getvalue()


def test_something_that_is_not_a_choice_is_asked_again(terminal):
    answer = terminal.ask(f"9{ENTER}2{ENTER}", "choice", choices=["1", "2"], default="1")
    assert answer == "2"
    assert "choose one of 1, 2" in terminal.screen.getvalue()


def test_a_choice_typed_in_the_other_case_is_the_choice(terminal):
    # The Ctrl+D is for a prompt that asks again: it ends the question and fails the test,
    # where it would otherwise wait for a key that is never going to be typed.
    assert terminal.ask(f"Y{ENTER}\x04", "overwrite?", choices=["y", "n"], default="n") == "y"


def test_enter_at_a_question_with_choices_takes_the_default(terminal):
    assert terminal.ask(ENTER, "overwrite?", choices=["y", "n"], default="n") == "n"


def test_a_key_is_read_and_never_drawn(terminal, key):
    assert terminal.ask(f"{key}{ENTER}", "paste your key", password=True) == key
    screen = terminal.screen.getvalue()
    assert key not in screen
    assert "paste your key:" in screen


def test_a_hidden_question_neither_shows_nor_takes_a_default(terminal):
    assert terminal.ask(ENTER, "paste your key", password=True, default="hunter2") == ""
    assert "hunter2" not in terminal.screen.getvalue()


def test_not_a_letter_of_the_key_is_drawn(terminal, key):
    terminal.ask(f"{key}{ENTER}", "paste your key", password=True)
    drawn = set(terminal.screen.getvalue().replace("paste your key:", ""))
    assert not drawn & set(key), drawn


def test_what_was_typed_for_one_question_cannot_be_called_back_at_the_next(terminal, key):
    # A prompt keeps what it was given and Up brings it back. The key's prompt is not the
    # model's: Up at the model question must show the model's own history, which is none.
    terminal.ask(f"{key}{ENTER}", "paste your key", password=True)
    answer = terminal.ask(f"\x1b[A{ENTER}", "model", default="gpt-5")
    assert answer == "gpt-5"
    assert key not in terminal.screen.getvalue()


def test_ctrl_c_ends_the_question_the_way_it_ends_any_other(terminal):
    with pytest.raises(KeyboardInterrupt):
        terminal.ask("\x03")


def test_ctrl_d_at_an_empty_line_says_the_input_has_ended(terminal):
    with pytest.raises(EOFError):
        terminal.ask("\x04")


# --------------------------------------------------------------------------
# The command: ncc init
# --------------------------------------------------------------------------


@dataclass
class Machine:
    """A machine with nothing set up: a home of its own, and ncc told to keep its files apart."""

    home: Path
    ncc: Path
    project: Path


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Machine:
    home, ncc, project = tmp_path / "home", tmp_path / "ncc", tmp_path / "project"
    for directory in (home, ncc, project):
        directory.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(ncc))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(project)
    return Machine(home, ncc, project)


def asked_by(
    monkeypatch: pytest.MonkeyPatch, ask: Callable[..., str], verify: Verifier | None = None
) -> None:
    """Have ``ncc init`` asked its questions by ``ask``, and check with ``verify`` if given."""

    def run(console: Any, *, home: Path) -> int:
        if verify is None:
            return run_init(console, home=home, ask=ask)
        return run_init(console, home=home, ask=ask, verify=verify)

    monkeypatch.setattr(ncc_main, "run_init", run)


def scripted(
    monkeypatch: pytest.MonkeyPatch, *answers: str | None, verify: Verifier | None = None
) -> Answers:
    """Have ``ncc init`` asked its questions by a script, and check with ``verify`` if given."""
    asking = Answers(*answers)
    asked_by(monkeypatch, asking, verify)
    return asking


def not_called(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make ``run_init`` fail the test if it is reached; the list says whether it was."""
    reached: list[str] = []

    def run(*_args: Any, **_kwargs: Any) -> int:
        reached.append("run_init")
        raise AssertionError("ncc init went on to ask its questions")

    monkeypatch.setattr(ncc_main, "run_init", run)
    return reached


def test_init_is_run_by_the_command_and_writes_where_ncc_keeps_its_files(
    machine, monkeypatch, capsys, key
):
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    code, out, err = run_ncc(capsys, "init")
    assert (code, err) == (0, "")
    assert "wrote" in out and "verified" in out
    assert config_path(machine.ncc).is_file()
    assert list(machine.home.iterdir()) == [], "nothing goes in HOME when ncc was told where"


def test_init_with_no_terminal_says_why_in_its_own_words_and_asks_nothing(
    machine, monkeypatch, capsys
):
    monkeypatch.setattr(ncc_main, "_stdin_is_terminal", lambda: False)
    reached = not_called(monkeypatch)
    code, out, err = run_ncc(capsys, "init")
    assert code == EXIT_CODES["usage"] == 2
    assert out == ""
    assert err == "error: ncc init asks questions \u2014 run it in a terminal\n"
    assert reached == []
    assert list(machine.ncc.iterdir()) == []


def test_init_with_no_terminal_leaves_an_existing_config_alone_without_asking(
    machine, monkeypatch, capsys
):
    write_home(machine.ncc, "# mine\n")
    monkeypatch.setattr(ncc_main, "_stdin_is_terminal", lambda: False)
    reached = not_called(monkeypatch)
    assert run_ncc(capsys, "init")[0] == EXIT_CODES["usage"]
    assert config_path(machine.ncc).read_text() == "# mine\n"
    assert reached == []


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["init", "-p", "hello"], "-p"),
        (["init", "--model", "m"], "--model"),
        (["init", "--root", "."], "--root"),
        (["init", "--mode", "plan"], "--mode"),
        (["init", "-c"], "-c"),
        (["init", "--allow-secrets"], "--allow-secrets"),
        (["init", "--dangerously-skip-permissions"], "--dangerously-skip-permissions"),
        (["--max-turns", "3", "init"], "--max-turns"),
        (["--output-format", "json", "init"], "--output-format"),
    ],
)
def test_init_takes_no_flag_that_shapes_a_session_and_does_not_ignore_one(
    machine, monkeypatch, capsys, argv, flag
):
    reached = not_called(monkeypatch)
    code, out, err = run_ncc(capsys, *argv)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        f"error: ncc init takes no {flag} \u2014 run it on its own "
        "(--no-color is the only flag that goes with it)\n"
    )
    assert reached == []


def test_a_flag_is_refused_when_ncc_is_run_as_the_console_script_runs_it(
    machine, monkeypatch, capsys
):
    # The script calls main() with no arguments, and main reads what was typed from sys.argv.
    monkeypatch.setattr(sys, "argv", ["ncc", "init", "--model", "m"])
    reached = not_called(monkeypatch)
    code = main()
    captured = capsys.readouterr()
    assert code == EXIT_CODES["usage"] and captured.out == ""
    assert captured.err.startswith("error: ncc init takes no --model")
    assert reached == []


@pytest.mark.parametrize("argv", [["init", "--no-color"], ["--no-color", "init"]])
def test_init_takes_no_color(machine, monkeypatch, capsys, key, argv):
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    code, out, _ = run_ncc(capsys, *argv)
    assert code == 0
    assert "\x1b[" not in out


def test_init_is_in_colour_where_colour_is_wanted_and_in_none_where_it_is_not(
    machine, monkeypatch, capsys, key
):
    monkeypatch.setenv("FORCE_COLOR", "1")  # as if the output were a colour terminal
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    assert colour_codes(run_ncc(capsys, "init")[1]), "the premise: it is coloured"
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    assert colour_codes(run_ncc(capsys, "init", "--no-color")[1]) == []


def test_a_word_that_is_not_a_command_is_said_so_and_a_task_goes_with_p(machine, capsys):
    code, out, err = run_ncc(capsys, "fix the failing test")
    assert code == EXIT_CODES["usage"] and out == ""
    assert err == (
        "error: fix the failing test is not a command \u2014 the only command is init; "
        "to give ncc a task, pass it with -p\n"
    )


def test_a_word_after_init_is_an_unrecognized_argument(machine, capsys):
    code, out, err = run_ncc(capsys, "init", "now")
    assert code == EXIT_CODES["usage"] and out == ""
    assert err.startswith("error: unrecognized argument now")


def test_help_lists_the_command(capsys):
    code, out, _ = run_ncc(capsys, "--help")
    assert code == 0
    assert "init" in out


def test_ctrl_c_at_a_question_ends_init_like_any_interrupted_run_with_nothing_written(
    machine, monkeypatch, capsys
):
    def interrupted(_message: str, **_kwargs: Any) -> str:
        raise KeyboardInterrupt

    asked_by(monkeypatch, interrupted)
    code, _, err = run_ncc(capsys, "init")
    assert (code, err) == (EXIT_CODES["interrupted"], INTERRUPTED)
    assert list(machine.ncc.iterdir()) == []


class Interrupting(Verifier):
    """A check that the person stops with Ctrl+C while it is waiting for the provider."""

    async def __call__(self, config: Config, alias: str) -> str | None:
        await super().__call__(config, alias)
        raise KeyboardInterrupt


def test_ctrl_c_during_the_check_says_the_config_was_written_and_the_key_was_not_checked(
    machine, monkeypatch, capsys, key
):
    scripted(monkeypatch, "1", key, None, verify=Interrupting())
    code, out, err = run_ncc(capsys, "init")
    assert code == EXIT_CODES["interrupted"] == 130
    assert config_path(machine.ncc).is_file(), "it was written before the check began"
    assert err == (
        f"error: interrupted while the key was being checked \u2014 the config was written to "
        f"{config_path(machine.ncc)} and the key was not checked; run ncc to try it, or ncc init "
        "to start over\n"
    )
    assert key not in out + err


def test_ctrl_c_at_a_question_is_still_the_plain_interrupted_line(machine, monkeypatch, capsys):
    def interrupted(_message: str, **_kwargs: Any) -> str:
        raise KeyboardInterrupt

    asked_by(monkeypatch, interrupted)
    assert run_ncc(capsys, "init")[2] == INTERRUPTED


def test_the_input_ending_at_a_question_is_said_and_nothing_is_written(
    machine, monkeypatch, capsys
):
    def ended(_message: str, **_kwargs: Any) -> str:
        raise EOFError

    asked_by(monkeypatch, ended)
    code, _, err = run_ncc(capsys, "init")
    assert code == EXIT_CODES["stopped"]
    assert err == (
        "error: the input ended before ncc init was finished \u2014 nothing was written; "
        "run ncc init again\n"
    )
    assert list(machine.ncc.iterdir()) == []


def test_a_config_that_cannot_be_written_is_one_line_and_a_configuration_error(
    machine, monkeypatch, capsys, key
):
    where = machine.project / "a-file"
    where.write_text("not a directory")
    monkeypatch.setenv("NANOCLAUDE_HOME", str(where))
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    code, out, err = run_ncc(capsys, "init")
    assert code == EXIT_CODES["config"]
    assert err.startswith("error: cannot write ") and err.count("\n") == 1
    assert key not in out + err


# -- where ncc keeps its files -----------------------------------------------------------


def test_the_home_ncc_is_told_about_may_start_with_a_tilde(machine, monkeypatch):
    monkeypatch.setenv("NANOCLAUDE_HOME", "~/ncc")
    assert nanoclaude_home() == machine.home / "ncc"


def test_with_no_home_given_ncc_keeps_its_files_in_the_users_home(machine, monkeypatch):
    monkeypatch.delenv("NANOCLAUDE_HOME")
    assert nanoclaude_home() == machine.home


def test_init_in_a_home_given_with_a_tilde_writes_in_the_home_directory_and_ncc_finds_it(
    machine, monkeypatch, capsys, key, serve
):
    # Written in <cwd>/~/ncc, and then not found by the loader, which expands the tilde: init
    # reported success and the next command said to run ncc init.
    monkeypatch.setenv("NANOCLAUDE_HOME", "~/ncc")
    scripted(monkeypatch, "1", key, None, verify=Verifier())
    code, out, err = run_ncc(capsys, "init")
    assert (code, err) == (0, "")
    assert config_path(machine.home / "ncc").is_file()
    assert not (machine.project / "~").exists()
    assert str(machine.home / "ncc") in out
    serve(anthropic=[says("fine")])
    assert run_ncc(capsys, "--root", str(machine.project), "-p", "hi")[:2] == (0, "fine\n")


def test_a_home_that_names_nobodys_home_directory_is_one_line_and_a_configuration_error(
    machine, monkeypatch, capsys
):
    monkeypatch.setenv("NANOCLAUDE_HOME", "~nobody_by_this_name_exists/ncc")
    reached = not_called(monkeypatch)
    code, out, err = run_ncc(capsys, "init")
    assert code == EXIT_CODES["config"] and out == ""
    assert err.startswith("error: NANOCLAUDE_HOME is '~nobody_by_this_name_exists/ncc'")
    assert err.count("\n") == 1 and reached == []


# -- the key, from the command's side: where it can be found afterwards -------------------


def files_holding(root: Path, secret: str) -> list[Path]:
    return [
        path for path in root.rglob("*") if path.is_file() and secret.encode() in path.read_bytes()
    ]


def test_the_key_is_in_no_file_and_on_neither_stream_after_a_whole_run(
    machine, monkeypatch, capsys, tmp_path, key, wire
):
    wire.answer = says_ok
    scripted(monkeypatch, "1", key, None)
    code, out, err = run_ncc(capsys, "init")
    assert code == 0 and "verified" in out
    assert wire.requests[0].headers["x-api-key"] == key, "the check was made with it"
    assert files_holding(tmp_path, key) == []
    assert key not in out and key not in err
    assert key not in os.environ.values()


def test_the_key_is_in_no_file_and_on_neither_stream_when_the_check_fails_and_says_it_back(
    machine, monkeypatch, capsys, tmp_path, key, wire
):
    wire.answer = lambda _request: httpx.Response(
        401, json={"error": {"message": f"Incorrect API key provided: {key}"}}
    )
    scripted(monkeypatch, "1", key, None)
    code, out, err = run_ncc(capsys, "init")
    assert code == 0 and "could not verify" in out
    assert files_holding(tmp_path, key) == []
    assert key not in out and key not in err


def test_the_key_is_in_no_file_and_on_neither_stream_when_the_request_blows_up(
    machine, monkeypatch, capsys, tmp_path, key, wire
):
    def explode(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"cannot send {key!r}")

    wire.answer = explode
    scripted(monkeypatch, "1", key, None)
    code, out, err = run_ncc(capsys, "init")
    assert code == 0 and "RuntimeError" in out
    assert files_holding(tmp_path, key) == []
    assert key not in out and key not in err


# --------------------------------------------------------------------------
# On a real terminal: the prompt that is the default, with nothing scripted but the keys
# --------------------------------------------------------------------------

#: ``run_init`` as ncc calls it, with the questions it asks by default and the check replaced:
#: this is about the prompt and the file, and no request leaves the machine.
TERMINAL_HARNESS = """
import sys
from pathlib import Path
from rich.console import Console
from nanoclaude.cli.init import run_init

async def answered(config, alias):
    return None

sys.exit(run_init(Console(), home=Path(sys.argv[1]), env={}, verify=answered))
"""

#: What a terminal sends back when asked where its cursor is, which prompt_toolkit asks.
CURSOR_AT_ORIGIN = b"\x1b[1;1R"
ASK_CURSOR = b"\x1b[6n"


def type_into(command: list[str], answers: list[tuple[bytes, bytes]]) -> tuple[int, bytes]:
    """Run ``command`` on a pseudo-terminal; type each answer once its question is on screen.

    Returns the exit status and everything the program drew. A question that never appears
    fails the test, naming it and what had been drawn, and the program is stopped.
    """
    master, slave = pty.openpty()
    process = subprocess.Popen(  # noqa: S603 - the interpreter running these tests, and a fixed script
        command,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
        start_new_session=True,
    )
    os.close(slave)
    drawn, since, asked, located = b"", 0, 0, 0
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 4096)
                except OSError:  # the program has gone and so has its end of the terminal
                    break
                if not chunk:
                    break
                drawn += chunk
                while located < drawn.count(ASK_CURSOR):
                    os.write(master, CURSOR_AT_ORIGIN)
                    located += 1
            if asked < len(answers) and answers[asked][0] in drawn[since:]:
                os.write(master, answers[asked][1])
                since, asked = len(drawn), asked + 1
        else:
            pytest.fail(f"timed out after {asked} answers; drawn so far: {drawn[-300:]!r}")
        return process.wait(timeout=10), drawn
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def test_on_a_real_terminal_the_key_is_typed_in_the_dark_and_the_config_is_private(tmp_path, key):
    status, drawn = type_into(
        [sys.executable, "-c", TERMINAL_HARNESS, str(tmp_path)],
        [
            (b"choice [1/2/3/4]", b"1\r"),
            (b"paste your key", key.encode() + b"\r"),
            (b"model (claude-sonnet-5-5)", b"\r"),
        ],
    )
    assert status == 0, drawn
    assert key.encode() not in drawn, "the terminal showed the key"
    assert b"*" in drawn, "and drew the mask that says a key is being typed"
    assert b"verified" in drawn
    assert config_path(tmp_path).stat().st_mode & 0o777 == 0o600
    assert key not in config_path(tmp_path).read_text()
    assert load_config(home=str(tmp_path), project=None, env={}).models["anthropic"].model == (
        "claude-sonnet-5-5"
    )


def test_on_a_real_terminal_a_choice_that_is_not_one_is_asked_again(tmp_path):
    status, drawn = type_into(
        [sys.executable, "-c", TERMINAL_HARNESS, str(tmp_path)],
        [
            (b"choice [1/2/3/4]", b"7\r"),
            (b"choose one of 1, 2, 3, 4", b"4\r"),
            (b"model (qwen3-coder)", b"\r"),
        ],
    )
    assert status == 0, drawn
    assert load_config(home=str(tmp_path), project=None, env={}).roles.main == "ollama"


# Each hosted preset's default model, written out by hand from the provider's own list.
# OpenRouter's public model list (https://openrouter.ai/api/v1/models), checked
# 2026-10-07: deepseek/deepseek-v3 is not on it; deepseek/deepseek-v4-pro is, with tools.
CHECKED_DEFAULTS = {
    "anthropic": "claude-sonnet-5-5",
    "openrouter": "deepseek/deepseek-v4-pro",
}


@pytest.mark.parametrize(("name", "model"), sorted(CHECKED_DEFAULTS.items()))
def test_a_presets_default_model_is_one_its_provider_lists(name, model):
    assert PRESETS[name].default_model == model


@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if p.needs_key])
def test_the_default_model_of_a_hosted_preset_is_one_the_program_knows(name):
    # A model the program knows nothing about gets the conservative default: an 8K window and
    # no tools, which is not what anybody who pressed Enter at the model question meant.
    preset = PRESETS[name]
    assert capabilities_for(preset.adapter, preset.default_model) != CONSERVATIVE_DEFAULT


def test_the_default_openrouter_model_is_given_the_window_it_has():
    preset = PRESETS["openrouter"]
    assert capabilities_for(preset.adapter, preset.default_model).context_window == 1_024_000


# --------------------------------------------------------------------------
# A key the HTTP library refuses
# --------------------------------------------------------------------------
# The key reaches the check request and nothing else, and that has to hold for a key the
# HTTP library will not send. What it says about one is a message with the header in it, in
# the escaped form of a bytes literal, which no search for the key itself would find: so the
# key is looked at before anything is sent, and what is shown is checked against every form
# it can take. These tests use a real client against a socket on this machine, because a
# transport made for tests never looks at a header.


@dataclass
class Server:
    """A server on this machine that answers like a provider, and what it was sent."""

    url: str
    received: list[tuple[str, dict[str, str], bytes]]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    seen: list[tuple[str, dict[str, str], bytes]] = []
    reply = (
        b'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}\n\n'
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Server(f"http://127.0.0.1:{httpd.server_address[1]}", seen)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


@pytest.fixture
def local_preset(server: Server, monkeypatch: pytest.MonkeyPatch) -> str:
    """A provider on this machine, offered as the last choice in the menu."""
    monkeypatch.setitem(
        PRESETS,
        "local",
        Preset(
            "local",
            "A server on this machine",
            "openai_compat",
            "test-model",
            "LOCAL_TEST_API_KEY",
            base_url=f"{server.url}/v1",
        ),
    )
    return choice_of("local")


def forms_of(secret: str) -> list[str]:
    """Every way a key can be written into a message: as it is, and escaped."""
    return [secret, repr(secret)[1:-1], secret.encode("unicode_escape").decode()]


async def refused_by_the_library(secret: str, url: str) -> str:
    """What a real client says when it is given ``secret`` to send, as the check would see it."""
    client = OpenAICompatClient(secret, model="m", base_url=f"{url}/v1")
    try:
        await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 8))
    except Exception as exc:  # the message is what is looked at
        return str(exc)
    finally:
        await client.aclose()
    raise AssertionError("the library sent a key it should have refused")


@pytest.mark.parametrize("flaw", ["\n", "\r", "\r\n"])
async def test_the_http_library_refuses_a_key_with_a_line_break_and_says_it_back_escaped(
    server, key, flaw
):
    # The premise of everything below: a message that holds the key, in a form the key's own
    # text does not appear in.
    secret = f"{key}{flaw}{key[:8]}"
    message = await refused_by_the_library(secret, server.url)
    assert secret not in message, "the premise: not in its own form"
    assert any(form in message for form in forms_of(secret)[1:]), message
    assert server.received == []


@pytest.mark.parametrize(
    "flaw", ["\n", "\r", "\r\n", "\u200b", "\u00a0", "\u00e9", "\x07", "'\"", "\\"]
)
def test_what_is_shown_has_the_key_cut_out_of_every_form_it_can_take(key, flaw):
    secret = f"{key}{flaw}{key[:8]}"
    for form in forms_of(secret):
        shown = _redact(f"Illegal header value b'Bearer {form}' in the request", secret)
        assert form not in shown
        assert "Illegal header value" in shown


@pytest.mark.parametrize("flaw", ["\n", "\r"])
async def test_even_a_key_that_got_past_the_check_is_not_said_back_by_the_library(
    tmp_path, server, key, flaw, monkeypatch
):
    # The backstop on its own: the check that looks at the key first is switched off, so the
    # real client is handed the key, refuses it, and says it back in its own words.
    monkeypatch.setattr(init_module, "_key_flaw", lambda _key: None)
    secret = f"{key}{flaw}{key[:8]}"
    write_home(
        tmp_path,
        f'[models.local]\nadapter = "openai_compat"\nmodel = "m"\n'
        f'base_url = "{server.url}/v1"\napi_key_env = "LOCAL_TEST_API_KEY"\n',
    )
    config = load_config(home=str(tmp_path), project=None, env={"LOCAL_TEST_API_KEY": secret})
    problem = await verify(config, "local")
    assert problem is not None and "Illegal header value" in problem
    assert not any(form in problem for form in forms_of(secret))
    assert server.received == []


BAD_KEYS = [
    ("\n", "a line break"),
    ("\r", "a line break"),
    ("\r\n", "a line break"),
    (" ", "a space"),
    ("\t", "a tab"),
    ("\x07", "a control character"),
    ("\x7f", "a control character"),
    ("\u200b", "a character that is not plain ASCII"),
    ("\u00a0", "a character that is not plain ASCII"),
    ("\u00e9", "a character that is not plain ASCII"),
]


@pytest.mark.parametrize(("flaw", "what"), BAD_KEYS)
def test_a_pasted_key_with_something_inside_it_is_not_sent_and_the_line_says_where(
    tmp_path, key, wire, flaw, what
):
    secret = f"{key}{flaw}{key[:8]}"
    wire.answer = says_ok
    console, buffer = plain_console(200)
    code = run_init(console, home=tmp_path, ask=Answers("1", secret, None), env={})
    out = buffer.getvalue()
    assert code == 0 and config_path(tmp_path).is_file()
    assert wire.requests == [], "nothing was sent"
    assert f"error: the key you pasted contains {what} at position {len(key) + 1} \u2014" in out
    assert "not sent anywhere" in out
    assert "verified" not in out and "could not verify" not in out
    for form in forms_of(secret):
        assert form not in out
    assert flaw.strip() == "" or flaw not in out  # the character itself is never shown


def test_the_line_counts_from_the_first_character_that_is_wrong(tmp_path, key, wire):
    console, buffer = plain_console(200)
    run_init(console, home=tmp_path, ask=Answers("1", f"ab cd\ne{key}", None), env={})
    assert "contains a space at position 3 \u2014" in buffer.getvalue()


def test_a_key_already_in_the_environment_is_looked_at_the_same_way(tmp_path, key, wire):
    secret = f"{key} {key[:4]}"
    console, buffer = plain_console(200)
    code = run_init(
        console, home=tmp_path, ask=Answers("1", None), env={"ANTHROPIC_API_KEY": secret}
    )
    out = buffer.getvalue()
    assert code == 0 and wire.requests == []
    assert (
        f"error: the key in ANTHROPIC_API_KEY contains a space at position {len(key) + 1} \u2014"
        in out
    )
    assert not any(form in out for form in forms_of(secret))


@pytest.mark.parametrize(("flaw", "what"), [("\n", "a line break"), (" ", "a space")])
async def test_the_check_itself_will_not_send_a_key_that_is_not_plain(
    tmp_path, key, wire, flaw, what
):
    # Whoever calls it: what is sent is looked at where it is sent.
    secret = f"{key}{flaw}x"
    problem = await verify(config_for(tmp_path, "anthropic", secret), "anthropic")
    assert problem is not None
    assert f"contains {what} at position {len(key) + 1}" in problem
    assert "ANTHROPIC_API_KEY" in problem
    assert not any(form in problem for form in forms_of(secret))
    assert wire.requests == []


@pytest.mark.parametrize("name", ["anthropic", "openai", "openrouter"])
async def test_a_key_of_plain_characters_is_sent_whatever_it_looks_like(tmp_path, wire, name):
    # Every printable ASCII character but the space: what a key may be made of.
    secret = "".join(chr(c) for c in range(0x21, 0x7F))
    wire.answer = says_ok
    assert await verify(config_for(tmp_path, name, secret), name) is None
    assert len(wire.requests) == 1


def test_a_good_key_reaches_a_real_server_in_its_header_and_a_bad_one_never_does(
    tmp_path, key, server, local_preset
):
    # The positive control, so that "nothing arrived" means something: the same run, with a
    # key that is whole, reaches the socket.
    console, buffer = plain_console(200)
    code = run_init(console, home=tmp_path, ask=Answers(local_preset, key, None), env={})
    assert code == 0 and "verified" in buffer.getvalue()
    ((path, headers, _),) = server.received
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == f"Bearer {key}"

    server.received.clear()
    bad = f"{key}\n{key[:8]}"
    console, buffer = plain_console(200)
    run_init(console, home=tmp_path, ask=Answers("y", local_preset, bad, None), env={})
    assert server.received == []
    out = buffer.getvalue()
    assert f"contains a line break at position {len(key) + 1}" in out
    assert not any(form in out for form in forms_of(bad))
