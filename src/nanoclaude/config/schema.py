"""Typed configuration, mirroring spec 17.6.

These are plain values. Nothing here checks them: ``load.py`` does that, once,
with the file in hand, so that a message can say which file to fix.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

KNOWN_ADAPTERS = ("anthropic", "openai_compat", "ollama")
ROLES = ("main", "explore", "plan", "verify", "compact", "title")
THEMES = ("auto", "light", "dark")
DIFF_STYLES = ("unified",)

#: A server's name in ``[mcp.<name>]``. No hyphen: the name is part of the registry name
#: ``mcp__<server>__<tool>``, which the text tool protocol writes as an identifier.
MCP_SERVER_NAME = r"[a-z0-9][a-z0-9_]{0,23}"

#: The transports a server may name. Streamable HTTP is not built, and when it is, it is
#: ``"http"``.
MCP_TRANSPORTS = ("stdio",)

#: The seconds a server's ``timeout`` may be, and what it is when the server does not say.
MCP_TIMEOUT_RANGE = (1, 600)
MCP_DEFAULT_TIMEOUT = 30

#: The variable a model's key is read from when its config does not name one.
#: Empty means the adapter needs no key.
DEFAULT_KEY_ENV: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai_compat": "OPENAI_API_KEY",
        "ollama": "",
    }
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    adapter: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    context_window: int | None = None
    native_tools: bool | None = None


def check_role(role: str) -> None:
    """Refuse a name that is not one of the six roles.

    A caller handing over a role it made up is a mistake in the program, not in
    anything a person wrote, so this is a ValueError and not a ConfigError.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; the roles are {', '.join(ROLES)}")


def key_env_name(entry: ModelConfig) -> str:
    """The variable this model's key is read from; empty when it needs none."""
    return entry.api_key_env or DEFAULT_KEY_ENV.get(entry.adapter, "")


@dataclass(frozen=True, slots=True)
class RolesConfig:
    main: str
    explore: str
    plan: str
    verify: str
    compact: str
    title: str

    def alias_for(self, role: str) -> str:
        """The alias of the model this role is routed to."""
        check_role(role)
        alias: str = getattr(self, role)
        return alias


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    max_turns: int = 40
    bash_timeout_s: float = 120.0
    output_cap_bytes: int = 100_000
    compact_soft: float = 0.70
    compact_hard: float = 0.85
    keep_recent_turns: int = 3


@dataclass(frozen=True, slots=True)
class PermissionsConfig:
    allow: tuple[str, ...] = ("Read", "Grep", "Glob")
    ask: tuple[str, ...] = ("Bash", "Write", "Edit", "Git", "TodoWrite")
    deny: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiConfig:
    theme: str = "auto"
    diff_style: str = "unified"


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One ``[mcp.<name>]``: a program ncc starts, speaks to over stdio and takes tools from.

    ``timeout`` is the seconds that one request to the server may take. ``env`` is added to
    the few variables a server is given from ncc's own environment, and is the one way a
    credential reaches it.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    # Never shown by repr, as Config.env is not: this is where a server's credentials go.
    env: dict[str, str] = field(default_factory=dict, repr=False)
    transport: str = "stdio"
    enabled: bool = True
    timeout: int = MCP_DEFAULT_TIMEOUT

    # Declared unhashable rather than left to the default: env is a dict. See Config.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Config:
    models: dict[str, ModelConfig]
    roles: RolesConfig
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    #: The MCP servers of the person's own file, by name, in the order the file gives them.
    mcp: dict[str, McpServerConfig] = field(default_factory=dict)
    # Never shown by repr: this holds API keys, and a repr ends up in logs, in
    # assertion output and in exception messages.
    env: dict[str, str] = field(default_factory=dict, repr=False)

    # Declared unhashable rather than left to the default: models and env are
    # dicts. Left to frozen=True's default eq=True, dataclass would generate a
    # real __hash__ -- isinstance(x, Hashable) would say True -- that only raises
    # when actually called, and names "dict" rather than this class.
    __hash__ = None  # type: ignore[assignment]

    def api_key_env_for(self, alias: str) -> str:
        """The variable this model's key is read from; empty when it needs none."""
        return key_env_name(self.models[alias])

    def api_key_for(self, alias: str) -> str:
        """The key itself, or an empty string when its variable is not set."""
        name = self.api_key_env_for(alias)
        return self.env.get(name, "") if name else ""
