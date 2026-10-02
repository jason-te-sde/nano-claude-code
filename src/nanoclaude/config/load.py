"""Loading and validating configuration.

Two files, project over user, merged key by key rather than wholesale -- a
project that wants a tighter turn limit should not have to restate its models.
A list in the user's file replaces the default, because the person's own file
says what they want. A list in the project's file stacks onto the one beneath it,
each rule once and in order, because a repository's config is not the person's
own: it can add a rule and can never take one away, and an empty list is a no-op.
It can add deny and ask rules, which only narrow what runs. It cannot add allow
rules, which widen it: a repository must not be able to grant itself permissions.
Nor can it re-point an alias the home file defines at another adapter, which would
send the key set up for one vendor to another's endpoint; it can change that
alias's model and the other fields that do not route.

Validation happens here rather than at first use. "Role explore points at a
model you have not defined" is a sentence worth reading before the first
request, not three turns in. Each file is checked on its own, before the two are
merged, so that a message can name the file that holds the mistake; what only the
merged result can show -- a model with no adapter, a role that points nowhere --
is checked afterwards.

A repository's file is not the person's own, so a project config may choose
models and roles but may not say where requests go or which key to send: naming
`base_url` or `api_key_env` there is an error, and belongs in the home file. A
project config that could would let a cloned repository aim the person's key at
whatever host it liked. The consequence is that a project config cannot define an
`openai_compat` model, which needs a base_url; it can select one the home file
defines.

Nothing here is silently ignored. A key that nothing reads, a section this
version does not have, a misspelled role: each is an error, because the
alternative is a setting the person believes is in force and is not.

Keys are never read from the file. The file names an environment variable; the
value comes from the environment. A config file full of secrets is a config
file someone will commit.

Everything a person can get wrong here is a :class:`ConfigError`, worded in the
form spec 17.9 gives for what people are shown: what went wrong, an em-dash,
what to do about it.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

from nanoclaude.config.schema import (
    DIFF_STYLES,
    KNOWN_ADAPTERS,
    ROLES,
    THEMES,
    Config,
    LimitsConfig,
    ModelConfig,
    PermissionsConfig,
    RolesConfig,
    UiConfig,
    key_env_name,
)
from nanoclaude.permissions.rules import Rule

CONFIG_FILENAME = "config.toml"
CONFIG_DIRNAME = ".nanoclaude"

#: Checks one value found at ``where`` (a dotted key) in the file at ``path``, and
#: returns it in the form the rest of the program uses.
Checker = Callable[[object, str, Path], Any]

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ConfigError(RuntimeError):
    """Configuration is missing, or does not make sense."""


def _expand(raw: str) -> Path:
    try:
        return Path(raw).expanduser()
    except RuntimeError as exc:
        raise ConfigError(
            f"cannot expand {raw!r} — it starts with ~ but no home directory is "
            "known for it; write the full path"
        ) from exc


def expand_root(raw: str) -> str:
    """A sandbox root as a person typed it, made absolute.

    ``Sandbox`` refuses a root that is not absolute, deliberately, so that a root
    never depends on where the process happens to be standing. But ``~/src/lib``
    is what people write on a command line and in a file, and a path in quotes
    reaches us with its tilde unexpanded. Expand it here, before the ``Sandbox``
    is built, and anchor a relative path to the working directory, which is what
    a flag such as ``--add-dir ../lib`` means.
    """
    return str(_expand(raw).resolve())


# ---------------------------------------------------------------------------
# Reading one file
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any] | None:
    """The file's tables, or None when there is no such file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"the config file {path} is not valid UTF-8 text — save it as UTF-8"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"could not read {path} ({exc.strerror or exc}) — check that it is a readable file"
        ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"the config file {path} is not valid TOML ({exc}) — fix the syntax error it names"
        ) from exc


# ---------------------------------------------------------------------------
# Checking one file: types, keys and ranges, each with its own message
# ---------------------------------------------------------------------------


def _describe(value: object) -> str:
    # bool is a subclass of int, so it has to be asked about first.
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, int):
        return "a whole number"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a table"
    return "a date or time"


def _wrong(where: str, path: Path, expected: str, value: object, fix: str) -> ConfigError:
    return ConfigError(f"{where} in {path} must be {expected}, not {_describe(value)} — {fix}")


def _table(value: object, where: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _wrong(where, path, "a table", value, f"write it as a [{where}] section")
    return value


def _string(value: object, where: str, path: Path) -> str:
    if not isinstance(value, str):
        raise _wrong(where, path, "a string", value, "put the value in quotes")
    return value


def _boolean(value: object, where: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise _wrong(where, path, "true or false", value, "write it without quotes")
    return value


def _number(value: object, where: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _wrong(where, path, "a number", value, "write it without quotes")
    return float(value)


def _whole(minimum: int) -> Checker:
    def check(value: object, where: str, path: Path) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _wrong(
                where, path, "a whole number", value, "write it without quotes or a decimal point"
            )
        if value < minimum:
            raise ConfigError(
                f"{where} in {path} is {value}, below the minimum of {minimum} "
                f"— set it to {minimum} or more"
            )
        return value

    return check


def _seconds(value: object, where: str, path: Path) -> float:
    number = _number(value, where, path)
    # Written as "not above zero" rather than "at or below" so that nan, which
    # compares false both ways, is refused too.
    if not number > 0:
        raise ConfigError(f"{where} in {path} is {value} — set a positive number of seconds")
    return number


def _fraction(value: object, where: str, path: Path) -> float:
    number = _number(value, where, path)
    if not 0 < number <= 1:
        raise ConfigError(
            f"{where} in {path} is {value} — set a share of the context window "
            "above 0 and at most 1, such as 0.7"
        )
    return number


def _choice(options: tuple[str, ...]) -> Checker:
    def check(value: object, where: str, path: Path) -> str:
        text = _string(value, where, path)
        if text not in options:
            raise ConfigError(
                f"{where} in {path} is {text!r} — set it to one of: {', '.join(options)}"
            )
        return text

    return check


def _url(value: object, where: str, path: Path) -> str:
    text = _string(value, where, path)
    if not text.startswith(("http://", "https://")):
        raise ConfigError(f"{where} in {path} must start with http:// or https:// — add the scheme")
    return text


def _env_name(value: object, where: str, path: Path) -> str:
    text = _string(value, where, path)
    if not _ENV_NAME.fullmatch(text):
        # The value is deliberately not repeated: the usual way to get here is
        # pasting the key itself, and an error message is exactly where it would
        # be printed to a terminal and into a log.
        raise ConfigError(
            f"{where} in {path} must be the name of an environment variable, such as "
            "ANTHROPIC_API_KEY — name the variable that holds the key, never the key itself"
        )
    return text


def _alias(value: object, where: str, path: Path) -> str:
    text = _string(value, where, path)
    if not text:
        raise ConfigError(
            f"{where} in {path} is empty — name a model defined under [models], "
            "or delete the line to leave it unset"
        )
    return text


def _strings(value: object, where: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _wrong(where, path, "a list of strings", value, 'write it like ["Read", "Grep"]')
    return tuple(_string(item, f"{where}[{index}]", path) for index, item in enumerate(value))


def _rules(value: object, where: str, path: Path) -> tuple[str, ...]:
    rules = _strings(value, where, path)
    for index, rule in enumerate(rules):
        try:
            Rule.parse(rule)
        except ValueError as exc:
            raise ConfigError(
                f"{where}[{index}] in {path} is not a valid rule ({exc}) — rules look like "
                "Tool, Tool(subject) or Tool(prefix:*)"
            ) from exc
    # Each rule once, so a list never differs from itself after a project file is
    # laid over it: stacking an empty list is then exactly a no-op.
    return tuple(dict.fromkeys(rules))


def _fields(
    value: object, fields: Mapping[str, Checker], where: str, path: Path, *, noun: str = "key"
) -> dict[str, Any]:
    """A table whose keys are exactly ``fields``, each value checked by its own checker."""
    table = _table(value, where, path)
    for key in table:
        if key not in fields:
            raise ConfigError(
                f"unknown {noun} {key!r} in [{where}] of {path} "
                f"— the {noun}s are {', '.join(fields)}"
            )
    return {key: fields[key](item, f"{where}.{key}", path) for key, item in table.items()}


def _section(fields: Mapping[str, Checker], *, noun: str = "key") -> Checker:
    def check(value: object, where: str, path: Path) -> dict[str, Any]:
        return _fields(value, fields, where, path, noun=noun)

    return check


_MODEL_FIELDS: dict[str, Checker] = {
    "adapter": _string,
    "model": _string,
    "base_url": _url,
    "api_key_env": _env_name,
    "context_window": _whole(1),
    "native_tools": _boolean,
}


def _model_entry(value: object, where: str, path: Path) -> dict[str, Any]:
    entry = _table(value, where, path)
    if "api_key" in entry:
        raise ConfigError(
            f"{where} in {path} has an api_key — keys never go in a config file, so name "
            "the environment variable that holds it with api_key_env"
        )
    return _fields(entry, _MODEL_FIELDS, where, path)


def _models(value: object, where: str, path: Path) -> dict[str, Any]:
    table = _table(value, where, path)
    return {alias: _model_entry(entry, f"{where}.{alias}", path) for alias, entry in table.items()}


_SECTION_CHECKS: dict[str, Checker] = {
    "models": _models,
    "roles": _section(dict.fromkeys(ROLES, _alias), noun="role"),
    "permissions": _section({"allow": _rules, "ask": _rules, "deny": _rules}),
    "limits": _section(
        {
            "max_turns": _whole(1),
            "bash_timeout_s": _seconds,
            "output_cap_bytes": _whole(1),
            "compact_soft": _fraction,
            "compact_hard": _fraction,
            # full_compact raises for anything lower: it would have to fold the
            # person's current request into the summary. Refused here, with the
            # key in the message, rather than as a traceback from inside it.
            "keep_recent_turns": _whole(1),
        }
    ),
    "ui": _section({"theme": _choice(THEMES), "diff_style": _choice(DIFF_STYLES)}),
}


class _Home(NamedTuple):
    """What lies beneath a project's file: where the home config is, and the models it defines."""

    path: Path
    models: Mapping[str, Mapping[str, Any]]


#: What a project config may not say about a model, and why it is the person's to say.
_NOT_IN_A_PROJECT = {
    "base_url": (
        "say where requests go",
        "; that also means a project config cannot define an openai_compat model, which needs one",
    ),
    "api_key_env": ("say which key to send", ""),
}


def _refuse_credential_routing(raw: dict[str, Any], path: Path, home: Path) -> None:
    """Refuse a project config that names a model's endpoint or key variable.

    Checked on presence, not on value, and before the types are: a wrong value is
    still an attempt, and the useful answer is where the key belongs.
    """
    models = raw.get("models")
    if not isinstance(models, dict):
        return
    for alias, entry in models.items():
        if not isinstance(entry, dict):
            continue
        for key, (what, note) in _NOT_IN_A_PROJECT.items():
            if key in entry:
                raise ConfigError(
                    f"models.{alias}.{key} in {path} is not allowed in a project config "
                    f"— a repository's file is not trusted to {what}, so set it in {home}{note}"
                )


def _refuse_widening(raw: dict[str, Any], path: Path, home: Path) -> None:
    """Refuse a project config that grants permissions.

    Checked on presence, like the keys above: the only allow list a project may
    have is an empty one, and anything else, a malformed rule included, is an
    attempt to grant something.
    """
    permissions = raw.get("permissions")
    if isinstance(permissions, dict) and permissions.get("allow", []) != []:
        raise ConfigError(
            f"permissions.allow in {path} is not allowed in a project config "
            "— a repository's file may add deny and ask rules but cannot grant itself "
            f"permissions, so add allow rules to {home} if you trust the project"
        )


def _refuse_adapter_change(layer: dict[str, Any], path: Path, home: _Home) -> None:
    """Refuse a project config that re-points an alias the home config defines.

    The key set up for one vendor would be sent to another's endpoint: an alias that
    reaches the person's own Ollama box, given an Anthropic adapter, would carry
    their Anthropic key there. An alias the project defines itself is not affected,
    and neither is a field that does not route: the model, its window, its tools.
    Run on the checked layer, so the adapter is known to be a string.
    """
    for alias, entry in layer.get("models", {}).items():
        beneath = home.models.get(alias)
        if beneath is None or "adapter" not in entry:
            continue
        was = beneath.get("adapter")
        if entry["adapter"] != was:
            what = f"adapter {was!r}" if was is not None else "no adapter"
            raise ConfigError(
                f"models.{alias}.adapter in {path} is {entry['adapter']!r}, but {home.path} "
                f"defines {alias!r} with {what} — a project config cannot re-point a home "
                "alias, because the key set up for one vendor would be sent to another's "
                f"endpoint; give the project its own alias, or change the adapter in {home.path}"
            )


def _check_layer(raw: dict[str, Any], path: Path, *, home: _Home | None = None) -> dict[str, Any]:
    """Check one file. ``home`` is given for a project's file: where the person's own file
    is, and what it defines, which a project's file may not overstep."""
    for name in raw:
        if name not in _SECTION_CHECKS:
            raise ConfigError(
                f"unknown section {name!r} in {path} "
                f"— the sections are {', '.join(_SECTION_CHECKS)}"
            )
    if home is not None:
        _refuse_credential_routing(raw, path, home.path)
        _refuse_widening(raw, path, home.path)
    layer = {name: _SECTION_CHECKS[name](value, name, path) for name, value in raw.items()}
    if home is not None:
        _refuse_adapter_change(layer, path, home)
    return layer


# ---------------------------------------------------------------------------
# Checking what only the merged result can show
# ---------------------------------------------------------------------------


def check_model(
    alias: str,
    adapter: str | None,
    model: str | None,
    base_url: str | None,
    *,
    home_config: Path | None = None,
) -> None:
    """Refuse a model entry that cannot be used, naming what it lacks.

    Also called by the router before it builds a client, so a ``ModelConfig`` made
    by hand gets the same answer as one read from a file instead of being sent to
    whichever adapter happens to be the fallback. The loader passes ``home_config``,
    the one file that may give an ``openai_compat`` model its base_url.
    """
    known = ", ".join(KNOWN_ADAPTERS)
    if adapter is None:
        raise ConfigError(f"model {alias!r} has no adapter — set adapter to one of: {known}")
    if adapter not in KNOWN_ADAPTERS:
        raise ConfigError(f"model {alias!r} uses adapter {adapter!r} — use one of: {known}")
    if not model:
        raise ConfigError(
            f'model {alias!r} has no model id — set model = "<id>" in [models.{alias}]'
        )
    if adapter == "openai_compat" and not base_url:
        where = (
            f", in {home_config}; a project config cannot set it, so the model has to be "
            "defined there"
            if home_config is not None
            else ""
        )
        raise ConfigError(
            f"model {alias!r} uses openai_compat but has no base_url — set base_url to "
            f"the endpoint, such as https://api.openai.com/v1{where}"
        )


def check_roles(roles: RolesConfig, models: Mapping[str, ModelConfig]) -> None:
    """Refuse a role that points at a model nobody defined, naming the ones that exist.

    Also called by the router when it is built: a ``--model`` or ``--role`` flag
    replaces an alias after the file was loaded, and a typo there should end the
    run before the first request rather than on the first use of that role.
    """
    for role in ROLES:
        alias = roles.alias_for(role)
        if alias not in models:
            raise ConfigError(
                f"role {role!r} points at model {alias!r}, which is not defined "
                f"— define [models.{alias}] or point the role at one of: "
                f"{', '.join(sorted(models))}"
            )


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def _union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    """Both lists in order, each entry once."""
    return tuple(dict.fromkeys((*first, *second)))


def _merge(base: dict[str, Any], overlay: dict[str, Any], *, stack: bool) -> dict[str, Any]:
    """Tables merge key by key, a list stacks onto the one beneath it when ``stack``, and
    anything else is replaced. A list is a tuple here: ``_rules`` made it one."""
    merged = dict(base)
    for key, value in overlay.items():
        beneath = merged.get(key)
        if isinstance(value, dict) and isinstance(beneath, dict):
            merged[key] = _merge(beneath, value, stack=stack)
        elif stack and isinstance(value, tuple) and isinstance(beneath, tuple):
            merged[key] = _union(beneath, value)
        else:
            merged[key] = value
    return merged


def _build(
    raw: dict[str, Any], paths: list[Path], env: Mapping[str, str], user_path: Path
) -> Config:
    models: dict[str, ModelConfig] = {}
    for alias, entry in raw.get("models", {}).items():
        check_model(
            alias,
            entry.get("adapter"),
            entry.get("model"),
            entry.get("base_url"),
            home_config=user_path,
        )
        models[alias] = ModelConfig(**entry)
    if not models:
        raise ConfigError(
            f"no [models.<alias>] sections found in {', '.join(map(str, paths))} "
            "— add one, or run: ncc init"
        )

    roles_raw = raw.get("roles", {})
    main = roles_raw.get("main", next(iter(models)))
    roles = RolesConfig(**{role: roles_raw.get(role, main) for role in ROLES})
    check_roles(roles, models)

    # Only the variables that hold a model's key: a config that is passed around
    # and printed should not carry the rest of the process's environment.
    held = {key_env_name(entry) for entry in models.values()} - {""}
    return Config(
        models=models,
        roles=roles,
        limits=LimitsConfig(**raw.get("limits", {})),
        permissions=PermissionsConfig(**raw.get("permissions", {})),
        ui=UiConfig(**raw.get("ui", {})),
        env={name: env[name] for name in held if name in env},
    )


def load_config(*, home: str, project: str | None, env: Mapping[str, str]) -> Config:
    """Defaults, then the home file, then the project file; keys from ``env``.

    ``home`` and ``project`` are directories, each expected to hold a
    ``.nanoclaude/config.toml``; a leading ``~`` in either is expanded. At least
    one of the two files must exist. A project's deny and ask lists stack onto the
    home lists, it may not add allow rules, its models may not name a ``base_url`` or
    ``api_key_env``, and it may not re-point an alias the home file defines.
    """
    user_path = _expand(home) / CONFIG_DIRNAME / CONFIG_FILENAME
    layers = [(user_path, False)]  # (file, is it a project's)
    if project:
        project_path = _expand(project) / CONFIG_DIRNAME / CONFIG_FILENAME
        # Run from the home directory, the "project" file is the home file: the
        # person's own, read once as that, not a second time as a repository's.
        if project_path.resolve() != user_path.resolve():
            layers.append((project_path, True))

    found: list[Path] = []
    # The permission lists begin as the defaults, so that a project, which adds to
    # the list beneath it, adds to the defaults when the home file says nothing.
    merged: dict[str, Any] = {"permissions": asdict(PermissionsConfig())}
    for path, is_project in layers:
        raw = _read(path)
        if raw is not None:
            found.append(path)
            beneath = _Home(user_path, merged.get("models", {})) if is_project else None
            layer = _check_layer(raw, path, home=beneath)
            merged = _merge(merged, layer, stack=is_project)
    if not found:
        raise ConfigError(f"no configuration found — run: ncc init (it writes {user_path})")
    return _build(merged, found, env, user_path)
