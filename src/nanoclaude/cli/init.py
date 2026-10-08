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

import re
from dataclasses import dataclass

from nanoclaude.config.schema import ROLES, LimitsConfig, PermissionsConfig


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
        "deepseek/deepseek-v3",
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

[limits]
max_turns      = {limits.max_turns}
bash_timeout_s = {limits.bash_timeout_s:g}
"""
