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

import tomllib
from pathlib import Path

import pytest

from nanoclaude.cli.init import PRESETS, render_config
from nanoclaude.config.load import load_config
from nanoclaude.config.schema import ROLES, PermissionsConfig
from nanoclaude.providers.capabilities import capabilities_for


def write_home(home: Path, text: str) -> None:
    (home / ".nanoclaude").mkdir(parents=True, exist_ok=True)
    (home / ".nanoclaude" / "config.toml").write_text(text)


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
