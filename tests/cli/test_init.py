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

import os
import secrets
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.cli.init import PRESETS, render_config, run_init
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
