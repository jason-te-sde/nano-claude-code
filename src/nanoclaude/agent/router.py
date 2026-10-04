"""Which model plays which role, and what each of them cost.

The reason this exists: exploring a codebase is most of the tokens in a coding
session and almost none of the difficulty. Sending that to a cheap or local
model while the main conversation stays on an expensive one is the single
largest saving available, and it is only possible because the provider
boundary is neutral.

Cost is reported per role, and the total is ``None`` if any model in the
session has no price. A partial total presented as a total is worse than no
total. The same rule holds inside a role: its cost is the sum of what each call
cost at that call's own model, so switching models mid-session neither reprices
what was already spent nor lets an unpriced stretch pass as free.

A router refuses a configuration that routes a role to a model nobody defined
as soon as it is built. A ``--model`` or ``--role`` flag replaces an alias after
the file was loaded and checked, and a typo there should end the run before the
first request.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace

from nanoclaude.config.load import check_model, check_roles
from nanoclaude.config.schema import ROLES, Config, ModelConfig, check_role
from nanoclaude.providers.anthropic import DEFAULT_BASE_URL as ANTHROPIC_BASE_URL
from nanoclaude.providers.anthropic import AnthropicClient
from nanoclaude.providers.base import ModelClient, ModelError, Usage
from nanoclaude.providers.capabilities import (
    Capabilities,
    CapabilityCache,
    resolve_capabilities,
)
from nanoclaude.providers.ollama import DEFAULT_BASE_URL as OLLAMA_BASE_URL
from nanoclaude.providers.ollama import OllamaClient, probe_ollama
from nanoclaude.providers.openai_compat import OpenAICompatClient
from nanoclaude.providers.pricing import PriceBook


@dataclass(frozen=True, slots=True)
class RoleUsage:
    """What one role has used and what that cost; ``cost`` is None when not known."""

    usage: Usage = field(default_factory=Usage)
    model: str = ""
    adapter: str = ""
    cost: float | None = 0.0


class Router:
    """Hands out the client and capabilities for each role, and keeps its ledger."""

    def __init__(
        self,
        config: Config,
        cache: CapabilityCache,
        clients: Mapping[str, ModelClient] | None = None,
        *,
        prices: PriceBook | None = None,
    ) -> None:
        """``clients`` maps a model alias to a client to use for it instead of building one."""
        check_roles(config.roles, config.models)
        self.config = config
        self.cache = cache
        self.prices = prices if prices is not None else PriceBook.load()
        self._clients: dict[str, ModelClient] = dict(clients or {})
        self._capabilities: dict[str, Capabilities] = {}
        self._usage: dict[str, RoleUsage] = {}

    async def client_for(self, role: str) -> ModelClient:
        """The client for the model this role is routed to; one per model, built on first use."""
        alias = self.config.roles.alias_for(role)
        if alias not in self._clients:
            self._clients[alias] = self._build(role, alias, self.config.models[alias])
        return self._clients[alias]

    async def route(self, role: str, alias: str) -> None:
        """Send ``role`` to the model defined as ``alias``, from the next request on.

        What ``/model`` does. The router, not the session, decides which client answers a
        role and which model a call is billed to, so a front end that changed only its own
        copy of the config would show one model and talk to another. The one this changes
        is ``self.config``; a caller holding another copy of the config keeps its roles in
        step with it.

        The client is built here and not when the next request needs it: a model that
        cannot be used (its key is not set) is refused now, to whoever asked for it, with
        the old route still in force, and not on the next prompt in the middle of a
        conversation. Nothing changes when it raises.

        Raises :class:`~nanoclaude.config.load.ConfigError` for an alias nobody defined,
        ``ValueError`` for a role that is not one of the six, and
        :class:`~nanoclaude.providers.base.ModelError` for a model that cannot be built.
        """
        check_role(role)
        roles = replace(self.config.roles, **{role: alias})
        check_roles(roles, self.config.models)
        previous = self.config
        self.config = replace(previous, roles=roles)
        try:
            await self.client_for(role)
        except BaseException:
            self.config = previous
            raise

    def _build(self, role: str, alias: str, entry: ModelConfig) -> ModelClient:
        # Also catches a ModelConfig made by hand, which the loader never saw: an
        # adapter nobody knows must be an error here and not whichever branch
        # below happens to come last.
        check_model(alias, entry.adapter, entry.model, entry.base_url)
        variable = self.config.api_key_env_for(alias)
        key = self.config.api_key_for(alias)
        if entry.adapter == "anthropic":
            # The adapter's own message names ANTHROPIC_API_KEY whatever the config
            # says, and cannot name the role. This one names both.
            if not key:
                raise ModelError(
                    f"role {role!r} uses model {alias!r}, which needs {variable} "
                    "— set it, or run: ncc init",
                    retryable=False,
                )
            return AnthropicClient(
                key, model=entry.model, base_url=entry.base_url or ANTHROPIC_BASE_URL
            )
        if entry.adapter == "openai_compat":
            # One adapter serves OpenRouter, Groq, DeepSeek and the rest, each with
            # its own variable, so the adapter is told which one to name when the
            # key is missing.
            return OpenAICompatClient(
                key, model=entry.model, base_url=entry.base_url or "", api_key_env=variable
            )
        return OllamaClient(model=entry.model, base_url=entry.base_url or OLLAMA_BASE_URL)

    async def capabilities_for(self, role: str) -> Capabilities:
        """What the model for this role can do: the table, the cache, a probe, then the config."""
        alias = self.config.roles.alias_for(role)
        if alias not in self._capabilities:
            entry = self.config.models[alias]
            probe = (
                functools.partial(
                    probe_ollama, entry.model, base_url=entry.base_url or OLLAMA_BASE_URL
                )
                if entry.adapter == "ollama"
                else None
            )
            self._capabilities[alias] = await resolve_capabilities(
                entry.adapter,
                entry.model,
                cache=self.cache,
                probe=probe,
                overrides={
                    "context_window": entry.context_window,
                    "native_tools": entry.native_tools,
                },
            )
        return self._capabilities[alias]

    def record(self, role: str, usage: Usage, adapter: str, model: str) -> None:
        """Add one call's usage to a role, priced at the model that made the call."""
        check_role(role)
        current = self._usage.get(role, RoleUsage())
        call_cost = self.prices.cost(usage, adapter, model)
        cost = None if current.cost is None or call_cost is None else current.cost + call_cost
        self._usage[role] = RoleUsage(current.usage + usage, model, adapter, cost)

    def by_role(self) -> dict[str, RoleUsage]:
        """Each role that has been used, in the order reports list the roles."""
        return {role: self._usage[role] for role in ROLES if role in self._usage}

    def total_usage(self) -> Usage:
        """Everything every role has used."""
        return sum((entry.usage for entry in self._usage.values()), Usage())

    def total_cost(self) -> float | None:
        """Dollars across every role, or None if any role's cost is not known."""
        costs = [entry.cost for entry in self._usage.values()]
        if None in costs:
            return None
        return sum((cost for cost in costs if cost is not None), 0.0)

    async def aclose(self) -> None:
        """Close every client held, and keep closing if one of them fails to."""
        async with AsyncExitStack() as stack:
            for client in self._clients.values():
                closer = getattr(client, "aclose", None)
                if closer is not None:
                    stack.push_async_callback(closer)
