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
import json
import os
import secrets
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from nanoclaude.cli import init as init_module
from nanoclaude.cli.init import PRESETS, render_config, run_init, verify
from nanoclaude.config.load import ConfigError, load_config
from nanoclaude.config.schema import ROLES, Config, PermissionsConfig
from nanoclaude.providers.capabilities import (
    CONSERVATIVE_DEFAULT,
    CapabilityCache,
    capabilities_for,
)
from tests.cli.helpers import plain_console


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


def test_a_blank_key_is_reported_by_the_check_and_does_not_break_the_setup(tmp_path, wire):
    console, buffer = plain_console(100)
    code = run_init(console, home=tmp_path, ask=Answers("1", "   ", None), env={})
    assert code == 0
    assert "could not verify" in buffer.getvalue() and "ANTHROPIC_API_KEY" in buffer.getvalue()
    assert wire.requests == []
    assert config_path(tmp_path).is_file()
    # Nothing was pasted, so there is no key to tell the person to keep; and an empty key is
    # not "redacted" out of the message, which would put the marker between every two letters.
    assert "<your key>" not in buffer.getvalue()
    assert "[hidden]" not in buffer.getvalue()
